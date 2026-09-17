import asyncio
import unittest
from unittest.mock import ANY, AsyncMock

from services.llm.providers.coedit_batch import CoEdITBatcher, NativeBatchObservation

ALLOCATOR = {"baseline_allocated": 10, "baseline_reserved": 20, "peak_allocated": 30,
             "peak_reserved": 40, "final_allocated": 10, "final_reserved": 20}
def evidence(size):
    return {"batch_size": size, "execution_started": 1, "execution_ended": 2,
            "cuda_synchronized": True, "allocator": ALLOCATOR,
            "decoder_steps": [64] * size, "max_output_tokens": 64}
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager
from services.llm.resource_manager.protocol import ProviderResponse


class CoEdITBatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.worker = AsyncMock()
        self.worker.frame_limit = 4096

    async def test_concurrent_calls_use_one_ordered_child_batch(self):
        self.worker.call.return_value = {
            "outputs": ["one-fixed", "two-fixed"],
            "observation": dict(evidence(2), execution_started=10, execution_ended=11),
        }
        batcher = CoEdITBatcher(self.worker, 2, 1.0, 64)
        first = asyncio.create_task(batcher.submit("r1", "fix one", "one"))
        second = asyncio.create_task(batcher.submit("r2", "fix two", "two"))
        self.assertEqual(await first, ("one-fixed", ANY))
        self.assertEqual(await second, ("two-fixed", ANY))
        self.worker.call.assert_awaited_once_with(
            "execute_batch", items=[
                {"instruction": "fix one", "texts": ["one"]},
                {"instruction": "fix two", "texts": ["two"]},
            ])
        observation = first.result()[1]
        from services.llm.providers.coedit_batch import AllocatorObservation
        self.assertEqual(observation, NativeBatchObservation(2, 10, 11, True, AllocatorObservation(10, 20, 30, 40, 10, 20), (64, 64), 64))
        await batcher.close()

    async def test_full_native_batch_preserves_32_row_cardinality_and_request_correlation(self):
        size = 32
        request_ids = tuple(f"request-{index}" for index in range(size))
        self.worker.call.return_value = {
            "outputs": [f"fixed-{index}" for index in range(size)],
            "observation": evidence(size),
        }
        batcher = CoEdITBatcher(self.worker, size, 0, 64)
        calls = [asyncio.create_task(batcher.submit(request_id, "fix", f"text-{index}"))
                 for index, request_id in enumerate(request_ids)]
        results = await asyncio.gather(*calls)
        self.assertEqual([output for output, _ in results], [f"fixed-{index}" for index in range(size)])
        observation = batcher.drain_observations()
        self.assertEqual(len(observation), 1)
        self.assertEqual(observation[0]["batch_size"], size)
        self.assertEqual(observation[0]["request_ids"], request_ids)
        await batcher.close()

    async def test_hard_bound_and_pending_cancellation(self):
        release = asyncio.Event()
        async def hold(_operation, **_kwargs):
            await release.wait()
            return {"outputs": ["one-fixed", "two-fixed"], "observation": evidence(2)}
        self.worker.call.side_effect = hold
        batcher = CoEdITBatcher(self.worker, 2, 1.0, 64)
        first = asyncio.create_task(batcher.submit("r1", "i", "one"))
        second = asyncio.create_task(batcher.submit("r2", "i", "two"))
        await asyncio.sleep(0)
        self.assertEqual(len(batcher.pending), 0)
        third = asyncio.create_task(batcher.submit("r3", "i", "three"))
        with self.assertRaises(RuntimeError):
            await third
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        second.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await second
        release.set()
        await batcher.close()
        self.assertEqual(batcher.pending, [])

    async def test_real_resource_manager_p2_executes_and_buffers_native_batches(self):
        release_first = asyncio.Event()
        first_entered = asyncio.Event()

        class Worker:
            frame_limit = 4096
            def __init__(self): self.calls = []
            async def call(self, operation, **values):
                self.calls.append((operation, values))
                items = values["items"]
                if len(self.calls) == 1:
                    first_entered.set()
                    await release_first.wait()
                return {"outputs": [item["texts"][0] + "-fixed" for item in items],
                    "observation": evidence(len(items))}

        class Provider:
            def __init__(self):
                self.worker = Worker()
                self.batcher = CoEdITBatcher(self.worker, 2, .01, 64)
            async def validate(self, profile): pass
            async def load(self, profile): pass
            async def ready(self): pass
            async def validate_input(self, payload, *, context_size, bucket_identity): pass
            async def execute(self, request_id, payload):
                text = payload.decode()
                result, _ = await self.batcher.submit(request_id, "fix", text)
                return ProviderResponse(result.encode())
            async def cancel(self, request_id): self.batcher.cancel(request_id)
            async def unload(self): await self.batcher.close()
            async def verify_cleanup(self): return True

        profile = CapacityProfile(ModelId.COEDIT, "gpu", "manifest", "model", "runtime", "adapter",
            "p2", 2, 2, 2, 0, (SampleMetadata(2, 0, 2, 1, 1, (1, 1)),),
            bucket_identity="coedit:p2:input128:output64:float16:beams1:nosample")
        rm = ResourceManager(cleanup_timeout=.1, stop_timeout=.1)
        provider = Provider()
        session = await rm.start_session("p2", ModelId.COEDIT, profile, provider, idempotency_key="start")
        try:
            for number in (1, 2):
                accepted = await rm.submit(session.session_token, f"r{number}", f"a{number}", str(number).encode(),
                    idempotency_key=f"k{number}", bucket_identity=profile.bucket_identity)
                self.assertTrue(accepted.accepted)
            await first_entered.wait()
            for number in (3, 4):
                accepted = await rm.submit(session.session_token, f"r{number}", f"a{number}", str(number).encode(),
                    idempotency_key=f"k{number}", bucket_identity=profile.bucket_identity)
                self.assertTrue(accepted.accepted)
            self.assertTrue((await rm.submit(session.session_token, "r5", "a5", b"5", idempotency_key="k5",
                bucket_identity=profile.bucket_identity)).backpressure)
            release_first.set()
            for _ in range(20):
                if len(provider.worker.calls) == 2: break
                await asyncio.sleep(.01)
            self.assertEqual([[item["texts"][0] for item in values["items"]]
                              for _, values in provider.worker.calls], [["1", "2"], ["3", "4"]])
        finally:
            release_first.set()
            await rm.stop_session(session.session_token, idempotency_key="stop")

    async def test_malformed_evidence_and_output_cardinality_fail_callers(self):
        for value in (
            {"outputs": ["only"], "observation": {"batch_size": 2, "execution_started": 1, "execution_ended": 2, "cuda_synchronized": False}},
            {"outputs": ["one", "two"], "observation": {"batch_size": 1, "execution_started": 1, "execution_ended": 2, "cuda_synchronized": False}},
        ):
            self.worker.reset_mock()
            self.worker.call.return_value = value
            batcher = CoEdITBatcher(self.worker, 2, 0, 64)
            calls = [asyncio.create_task(batcher.submit(str(n), "i", text)) for n, text in enumerate(("one", "two"))]
            for call in calls:
                with self.assertRaises(RuntimeError):
                    await call
            await batcher.close()

    async def test_float_native_observation_fences_are_rejected_at_transport(self):
        self.worker.call.return_value = {"outputs": ["fixed"], "observation": dict(evidence(1), execution_started=1.0, execution_ended=2.0)}
        batcher = CoEdITBatcher(self.worker, 1, 0, 64)
        with self.assertRaisesRegex(RuntimeError, "malformed CoEdIT batch observation"):
            await batcher.submit("r1", "i", "text")
        await batcher.close()

    async def test_close_fences_late_running_result(self):
        release = asyncio.Event()

        async def execute(*args, **kwargs):
            await release.wait()
            return {"outputs": ["late"], "observation": dict(evidence(1), cuda_synchronized=False)}

        self.worker.call.side_effect = execute
        batcher = CoEdITBatcher(self.worker, 1, 0, 64)
        call = asyncio.create_task(batcher.submit("r1", "i", "text"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        closing = asyncio.create_task(batcher.close())
        await asyncio.sleep(0)
        release.set()
        await closing
        with self.assertRaises(RuntimeError):
            await call

    async def test_request_id_cancellation_fences_only_that_active_result(self):
        release = asyncio.Event()

        async def execute(*args, **kwargs):
            await release.wait()
            return {"outputs": ["one-fixed", "two-fixed"], "observation": evidence(2)}

        self.worker.call.side_effect = execute
        batcher = CoEdITBatcher(self.worker, 2, 1, 64)
        first = asyncio.create_task(batcher.submit("r1", "i", "one"))
        second = asyncio.create_task(batcher.submit("r2", "i", "two"))
        await asyncio.sleep(0)
        batcher.cancel("r1")
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertEqual(await second, ("two-fixed", ANY))
        await batcher.close()

    async def test_aggregate_frame_limit_partitions_pending_items_after_active_batch(self):
        class Worker:
            # The production batch envelope includes the fixed max-sized RPC
            # id; 190 admits two items (187 bytes) but not three (250 bytes).
            frame_limit = 190
            def __init__(self): self.calls = []; self.release = asyncio.Event()
            async def call(self, operation, **values):
                self.calls.append(values["items"])
                if len(self.calls) == 1: await self.release.wait()
                return {"outputs": ["fixed"] * len(values["items"]), "observation": evidence(len(values["items"]))}
        worker = Worker()
        batcher = CoEdITBatcher(worker, 3, 0, 64)
        calls = [asyncio.create_task(batcher.submit(str(index), "i", "x" * 30)) for index in range(3)]
        await asyncio.sleep(0)
        worker.release.set()
        await asyncio.gather(*calls)
        self.assertEqual([len(items) for items in worker.calls], [2, 1])
        await batcher.close()

    async def test_batcher_rejects_invalid_direct_bounds_and_drains_observation_ids(self):
        with self.assertRaises(ValueError): CoEdITBatcher(self.worker, 0, 0, 64)
        with self.assertRaises(ValueError): CoEdITBatcher(self.worker, 33, 0, 64)
        with self.assertRaises(ValueError): CoEdITBatcher(self.worker, 1, float("nan"), 64)
        with self.assertRaises(ValueError): CoEdITBatcher(self.worker, 1, 0, True)
        self.worker.call.return_value = {"outputs": ["fixed"], "observation": evidence(1)}
        batcher = CoEdITBatcher(self.worker, 1, 0, 64)
        call = asyncio.create_task(batcher.submit("request-1", "i", "t"))
        await call
        self.assertEqual(batcher.drain_observations()[0]["request_ids"], ("request-1",))
        await batcher.close()

    async def test_observation_window_is_bounded_and_eviction_is_observable(self):
        self.worker.call.return_value = {"outputs": ["fixed"], "observation": evidence(1)}
        batcher = CoEdITBatcher(self.worker, 1, 0, 64)
        for request_id in ("one", "two"):
            await batcher.submit(request_id, "i", "t")
        self.assertEqual(batcher.dropped_observations, 1)
        self.assertEqual(batcher.drain_observations()[0]["request_ids"], ("two",))
        await batcher.close()

    async def test_decoder_workload_rejects_bool_and_out_of_bound_counts(self):
        for steps in ([True], [65], [1, 2]):
            self.worker.call.return_value = {"outputs": ["fixed"], "observation": dict(evidence(1), decoder_steps=steps)}
            batcher = CoEdITBatcher(self.worker, 1, 0, 64)
            with self.assertRaises(RuntimeError): await batcher.submit("r", "i", "t")
            await batcher.close()

    async def test_ordinary_output_bucket_is_not_limited_by_candidate_witness(self):
        self.worker.call.return_value = {"outputs": ["fixed"], "observation": {
            **evidence(1), "decoder_steps": [128], "max_output_tokens": 128}}
        batcher = CoEdITBatcher(self.worker, 1, 0, 128)
        output, observation = await batcher.submit("r", "i", "t")
        self.assertEqual((output, observation.max_output_tokens, observation.decoder_steps),
                         ("fixed", 128, (128,)))
        await batcher.close()

    async def test_observation_maximum_must_bind_to_configured_batcher_maximum(self):
        self.worker.call.return_value = {"outputs": ["fixed"], "observation": {
            **evidence(1), "decoder_steps": [63], "max_output_tokens": 63}}
        batcher = CoEdITBatcher(self.worker, 1, 0, 64)
        with self.assertRaisesRegex(RuntimeError, "decoder workload"):
            await batcher.submit("r", "i", "t")
        await batcher.close()

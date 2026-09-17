import asyncio
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from services.llm.provisioning.artifacts import SPECS
from services.llm.provisioning.volume import provision
from services.llm.providers.coedit import CoEdITProvider
from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import (_ResidencyPending, GPUProofError,
                                        GPUMemoryObservation, ProcessIdentity,
                                        ResidencyEvidence)
from services.llm.providers.python_config import PythonProviderConfig
from services.llm.providers.coedit_batch import CoEdITBatcher
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata


class CoEdITProviderTests(unittest.IsolatedAsyncioTestCase):
    def config(self, root, manifest="0" * 64, model="0" * 64):
        raw=(Path("/proc")/str(os.getpid())/"stat").read_bytes()
        supervisor=ProcessIdentity(os.getpid(),int(raw[raw.rfind(b")")+2:].split()[19]))
        proof=GPUProof(lambda: "GPU-test", AsyncMock(return_value=True), lambda: None, supervisor)
        return PythonProviderConfig(Path(root), manifest, model, "GPU-test", "runtime", "adapter", gpu_proof=proof)

    def profile(self, config):
        return CapacityProfile(ModelId.COEDIT, config.gpu_uuid, config.manifest_sha256,
            config.model_sha256, config.runtime_identity, config.adapter_identity,
            config.bucket_identity, 1, 1, 1, 0, (SampleMetadata(1,0,1,1,1,(1,)),),
            bucket_identity=config.bucket_identity)

    async def test_selected_volume_layout_and_streamed_integrity_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"source"; source.mkdir()
            for name in SPECS["CoEdIT"]: (source/name).write_bytes(name.encode())
            volume=Path(directory)/"volume"; document=provision({"CoEdIT":source},volume)
            model=next(x for x in document["models"]["CoEdIT"]["files"] if x["path"]=="model.safetensors")["sha256"]
            provider=CoEdITProvider(self.config(volume,document["manifest_sha256"],model))
            self.assertEqual(provider._artifact(),volume/document["manifest_sha256"]/"models"/"CoEdIT")
            (volume/document["manifest_sha256"]/"models"/"CoEdIT"/"config.json").write_text("corrupt")
            with self.assertRaises(ValueError): provider._artifact()

    async def test_parent_has_no_torch_import(self):
        import services.llm.providers.python_process as process
        self.assertNotIn("torch", process.__dict__)

    async def test_exact_profile_and_native_batch_one_request_are_required(self):
        config=self.config("/tmp"); provider=CoEdITProvider(config); provider.profile=self.profile(config)
        worker=AsyncMock(); provider.worker=worker
        await provider.validate_input(b'{"instruction":"fix","texts":["text"]}',context_size=None,bucket_identity=config.bucket_identity)
        # Admission performs only bounded envelope validation.  Tokenization is
        # authoritative in the child's execute_batch operation, so concurrent
        # admissions do not serialize one validation RPC per request.
        worker.call.assert_not_awaited()
        with self.assertRaises(ValueError):
            await provider.validate_input(b'{"instruction":"fix","texts":["one","two"]}',context_size=None,bucket_identity=config.bucket_identity)
        # Registry profile identities are measurement identities, not buckets;
        # memory_safe_n may retain diagnostic headroom above p=1.
        accepted=CapacityProfile(ModelId.COEDIT,"GPU-test","0"*64,"0"*64,"runtime","adapter","registry-measurement-9",1,3,1,0,(SampleMetadata(1,0,1,1,1,()),),bucket_identity=config.bucket_identity)
        await provider.validate(accepted)
        wrong=CapacityProfile(ModelId.COEDIT,"GPU-test","0"*64,"0"*64,"runtime","adapter","other",1,1,1,0,(SampleMetadata(1,0,1,1,1,()),),bucket_identity="other")
        with self.assertRaises(ValueError): await provider.validate(wrong)

    async def test_load_runs_cuda_readiness_after_load_and_fences_failure(self):
        config = self.config("/tmp")
        provider = CoEdITProvider(config)
        calls = []
        worker = AsyncMock()
        async def call(operation, **values):
            calls.append(operation)
            if operation == "cuda_ready":
                raise RuntimeError("CUDA readiness operation failed")
            return True
        worker.call.side_effect = call
        with patch("services.llm.providers.coedit.PythonWorker", return_value=worker), \
                patch.object(provider, "_artifact", return_value=Path("/offline")):
            with self.assertRaises(RuntimeError): await provider.load(self.profile(config))
        self.assertEqual(calls, ["load", "cuda_ready"])
        worker.close.assert_awaited_once()
        self.assertIsNone(provider.worker)
        self.assertFalse(provider._ready)
        self.assertTrue(await provider.verify_cleanup())

    async def test_opt_in_batch_profile_is_bound_to_exact_batch_bucket(self):
        config = self.config("/tmp")
        config = PythonProviderConfig(Path("/tmp"), config.manifest_sha256, config.model_sha256,
            config.gpu_uuid, config.runtime_identity, config.adapter_identity,
            bucket_identity="coedit:p2:input128:output64:float16:beams1:nosample",
            max_native_batch_size=2, gpu_proof=config.gpu_proof)
        provider = CoEdITProvider(config)
        profile = CapacityProfile(ModelId.COEDIT, config.gpu_uuid, config.manifest_sha256,
            config.model_sha256, config.runtime_identity, config.adapter_identity,
            "measured", 1, 2, 1, 0, (SampleMetadata(1, 0, 1, 1, 1, ()),),
            bucket_identity=config.bucket_identity)
        await provider.validate(profile)
        self.assertEqual(provider.profile.optimal_parallelism, 1)
        with self.assertRaises(ValueError):
            await provider.validate(profile.__class__(ModelId.COEDIT, config.gpu_uuid,
                config.manifest_sha256, config.model_sha256, config.runtime_identity,
                config.adapter_identity, "too-high", 3, 3, 3, 0, profile.raw_samples,
                bucket_identity=config.bucket_identity))
        # Measurement identity is diagnostic; the exact request bucket is the
        # independently pinned bucket_identity above.
        await provider.validate(profile.__class__(ModelId.COEDIT, config.gpu_uuid,
            config.manifest_sha256, config.model_sha256, config.runtime_identity,
            config.adapter_identity, "wrong", 2, 2, 2, 0, profile.raw_samples,
            bucket_identity=config.bucket_identity))
        with self.assertRaises(ValueError):
            PythonProviderConfig(Path("/tmp"), config.manifest_sha256, config.model_sha256,
                config.gpu_uuid, config.runtime_identity, config.adapter_identity,
                bucket_identity=config.bucket_identity, max_native_batch_size=2,
                native_batch_delay_seconds=.01, gpu_proof=config.gpu_proof)

    async def test_native_batch_configuration_accepts_32_rejects_33_and_defaults_to_one(self):
        config = self.config("/tmp")
        self.assertEqual(config.max_native_batch_size, 1)
        config32 = PythonProviderConfig(
            Path("/tmp"), config.manifest_sha256, config.model_sha256,
            config.gpu_uuid, config.runtime_identity, config.adapter_identity,
            bucket_identity="coedit:p32:input128:output64:float16:beams1:nosample",
            max_native_batch_size=32, gpu_proof=config.gpu_proof)
        self.assertEqual(config32.max_native_batch_size, 32)
        with self.assertRaisesRegex(ValueError, "native batch size"):
            PythonProviderConfig(
                Path("/tmp"), config.manifest_sha256, config.model_sha256,
                config.gpu_uuid, config.runtime_identity, config.adapter_identity,
                bucket_identity="coedit:p33:input128:output64:float16:beams1:nosample",
                max_native_batch_size=33, gpu_proof=config.gpu_proof)

    async def test_execute_preserves_order_and_returns_individual_frames(self):
        config = self.config("/tmp")
        provider = CoEdITProvider(config)
        provider.profile = self.profile(config)
        provider._ready = True
        worker = AsyncMock()
        worker.frame_limit = 4096
        worker.call.return_value = True
        provider.worker = worker
        provider._batcher = CoEdITBatcher(worker, 1, 0, config.max_output_tokens)
        worker.call.side_effect = [{"outputs": ["fixed"], "observation": {
            "batch_size": 1, "execution_started": 1, "execution_ended": 2,
            "cuda_synchronized": True, "allocator": {"baseline_allocated": 10, "baseline_reserved": 20,
            "peak_allocated": 30, "peak_reserved": 40, "final_allocated": 10, "final_reserved": 20}, "decoder_steps": [64], "max_output_tokens": 64}}]
        response = await provider.execute("request", b'{"instruction":"fix","texts":["text"]}')
        self.assertEqual(response.result, b'{"texts":["fixed"]}')
        self.assertEqual(worker.call.await_args_list[0].args[0], "execute_batch")
        await provider._batcher.close()

    async def test_invalid_input_does_not_poison_worker(self):
        config = self.config("/tmp")
        provider = CoEdITProvider(config)
        provider.profile = self.profile(config)
        worker = AsyncMock()
        worker.frame_limit = 4096
        provider.worker = worker
        with self.assertRaises(ValueError):
            await provider.validate_input(b'{"instruction":"fix","texts":["one","two"]}',
                context_size=None, bucket_identity=config.bucket_identity)
        await provider.validate_input(b'{"instruction":"fix","texts":["one"]}',
            context_size=None, bucket_identity=config.bucket_identity)
        self.assertEqual(worker.call.await_count, 0)

    async def test_concurrent_admission_validation_is_local_and_bounded(self):
        config = self.config("/tmp")
        provider = CoEdITProvider(config)
        provider.profile = self.profile(config)
        worker = AsyncMock()
        worker.frame_limit = 4096
        provider.worker = worker
        payloads = [f'{{"instruction":"fix-{n}","texts":["text-{n}"]}}'.encode()
                    for n in range(8)]
        await asyncio.gather(*(provider.validate_input(payload, context_size=None,
            bucket_identity=config.bucket_identity) for payload in payloads))
        worker.call.assert_not_awaited()

    async def test_execute_rechecks_exact_bytes_before_child_batch_execution(self):
        config = self.config("/tmp")
        provider = CoEdITProvider(config)
        provider.profile = self.profile(config)
        provider._ready = True
        worker = AsyncMock()
        worker.frame_limit = 4096
        worker.call.return_value = {"outputs": ["fixed"], "observation": {
            "batch_size": 1, "execution_started": 1, "execution_ended": 2,
            "cuda_synchronized": True, "allocator": {
                "baseline_allocated": 10, "baseline_reserved": 20,
                "peak_allocated": 30, "peak_reserved": 40,
                "final_allocated": 10, "final_reserved": 20}, "decoder_steps": [64], "max_output_tokens": 64}}
        provider.worker = worker
        provider._batcher = CoEdITBatcher(worker, 1, 0, config.max_output_tokens)
        with self.assertRaises(ValueError):
            await provider.execute("bad", b'{"instruction":"fix","texts":["one","two"]}')
        worker.call.assert_not_awaited()
        await provider._batcher.close()

    async def test_ready_requires_exact_child_runner_residency(self):
        config=self.config("/tmp"); provider=CoEdITProvider(config)
        runner=ProcessIdentity(123,456)
        worker=AsyncMock(); worker.child_identity=runner
        worker.call.return_value={"gpu_uuid":"GPU-test","runner_pid":123,"runner_start_time":456,"cuda_nvml_agree":True}
        provider.worker=worker
        object.__setattr__(config.gpu_proof, "residency", AsyncMock(return_value=ResidencyEvidence("GPU-test",config.gpu_proof.expected_supervisor,(runner,))))
        await provider.ready(); self.assertTrue(provider._ready)

    async def test_ready_uses_expected_child_residency_when_available(self):
        config=self.config("/tmp"); provider=CoEdITProvider(config)
        runner=ProcessIdentity(123,456)
        worker=AsyncMock(); worker.child_identity=runner
        worker.call.return_value={"gpu_uuid":"GPU-test","runner_pid":123,"runner_start_time":456,"cuda_nvml_agree":True}
        provider.worker=worker
        expected = AsyncMock(return_value=ResidencyEvidence("GPU-test", config.gpu_proof.expected_supervisor, (runner,)))
        object.__setattr__(config.gpu_proof, "residency_for_runner", expected)
        await provider.ready()
        expected.assert_awaited_once_with(runner)

    def _fallback_fixture(self, *, memory=None, calls=None):
        config = self.config("/tmp")
        provider = CoEdITProvider(config)
        runner = ProcessIdentity(123, 456)
        worker = AsyncMock()
        worker.child_identity = runner
        identity = {"gpu_uuid": "GPU-test", "runner_pid": 123,
                    "runner_start_time": 456, "cuda_nvml_agree": True}
        worker.call.side_effect = list(calls or [])
        provider.worker = worker
        object.__setattr__(config.gpu_proof, "residency_for_runner",
                           AsyncMock(side_effect=_ResidencyPending("absent")))
        baseline = GPUMemoryObservation("GPU-test", config.gpu_proof.expected_supervisor,
                                        1, 2, 1000, 400, 600)
        provider._preload_memory = baseline
        object.__setattr__(config.gpu_proof, "memory", AsyncMock(side_effect=memory or (
            GPUMemoryObservation("GPU-test", config.gpu_proof.expected_supervisor, 3, 4, 1000, 500, 500),
            GPUMemoryObservation("GPU-test", config.gpu_proof.expected_supervisor, 5, 6, 1000, 501, 499))))
        return provider, config, runner

    async def test_coedit_fallback_accepts_ordered_samples_with_fluctuating_memory(self):
        provider, config, runner = self._fallback_fixture(calls=[
            {"gpu_uuid": "GPU-test", "runner_pid": 123, "runner_start_time": 456, "cuda_nvml_agree": True},
            {"model_cuda_device": "cuda:0", "witness_exists": True, "witness_cuda_device": "cuda:0"},
            {"gpu_uuid": "GPU-test", "runner_pid": 123, "runner_start_time": 456, "cuda_nvml_agree": True},
            {"model_cuda_device": "cuda:0", "witness_exists": True, "witness_cuda_device": "cuda:0"},
        ])
        await provider._coedit_fallback(runner)
        # Each proof sample is independently fenced: identity, retained CUDA
        # witness, typed absence, and memory observation.
        self.assertEqual(provider.worker.call.await_count, 4)

    async def test_coedit_fallback_rejects_non_pending_or_invalid_evidence(self):
        base_calls = [
            {"gpu_uuid": "GPU-test", "runner_pid": 123, "runner_start_time": 456, "cuda_nvml_agree": True},
            {"model_cuda_device": "cuda:0", "witness_exists": True, "witness_cuda_device": "cuda:0"},
        ]
        for mutation in ("residency", "witness", "identity", "memory"):
            with self.subTest(mutation=mutation):
                provider, config, runner = self._fallback_fixture(calls=base_calls)
                if mutation == "residency":
                    object.__setattr__(config.gpu_proof, "residency_for_runner",
                                       AsyncMock(side_effect=GPUProofError("changed topology")))
                elif mutation == "witness":
                    provider.worker.call.side_effect = [base_calls[0], {"witness_exists": False}]
                elif mutation == "identity":
                    provider.worker.call.side_effect = [base_calls[0], base_calls[1],
                        {"gpu_uuid": "GPU-test", "runner_pid": 999, "runner_start_time": 456, "cuda_nvml_agree": True}]
                else:
                    object.__setattr__(config.gpu_proof, "memory", AsyncMock(side_effect=[
                        GPUMemoryObservation("GPU-test", config.gpu_proof.expected_supervisor, 3, 4, 1000, 400, 600),
                        GPUMemoryObservation("GPU-test", config.gpu_proof.expected_supervisor, 5, 6, 1000, 401, 599)]))
                with self.assertRaises((RuntimeError, GPUProofError)):
                    await provider._coedit_fallback(runner)

    async def test_coedit_fallback_cancellation_is_not_readiness_authority(self):
        provider, config, runner = self._fallback_fixture(calls=[
            {"gpu_uuid": "GPU-test", "runner_pid": 123, "runner_start_time": 456,
             "cuda_nvml_agree": True},
            {"model_cuda_device": "cuda:0", "witness_exists": True,
             "witness_cuda_device": "cuda:0"},
        ])
        object.__setattr__(config.gpu_proof, "residency_for_runner",
                           AsyncMock(side_effect=asyncio.CancelledError()))
        with self.assertRaises(asyncio.CancelledError):
            await provider._coedit_fallback(runner)
        self.assertFalse(provider.accepted_model_specific_residency())

    async def test_ready_falls_back_after_settlement_timeout_and_grants_model_authority(self):
        config = self.config("/tmp")
        provider = CoEdITProvider(config)
        runner = ProcessIdentity(123, 456)
        identity = {"gpu_uuid": "GPU-test", "runner_pid": 123,
                    "runner_start_time": 456, "cuda_nvml_agree": True}
        witness = {"model_cuda_device": "cuda:0", "witness_exists": True,
                   "witness_cuda_device": "cuda:0"}
        worker = AsyncMock()
        worker.child_identity = runner
        worker.call.side_effect = [identity, identity, witness, identity, witness]
        provider.worker = worker
        object.__setattr__(config.gpu_proof, "residency_for_runner",
                           AsyncMock(side_effect=_ResidencyPending("absent")))
        supervisor = config.gpu_proof.expected_supervisor
        provider._preload_memory = GPUMemoryObservation(
            "GPU-test", supervisor, 1, 2, 1000, 400, 600)
        object.__setattr__(config.gpu_proof, "memory", AsyncMock(side_effect=(
            GPUMemoryObservation("GPU-test", supervisor, 3, 4, 1000, 500, 500),
            GPUMemoryObservation("GPU-test", supervisor, 5, 6, 1000, 501, 499))))
        now = [0.0]
        async def expire_settlement(delay):
            now[0] += 5.0
        provider._residency_clock = lambda: now[0]
        provider._residency_sleep = expire_settlement

        await provider.ready()

        self.assertTrue(provider._ready)
        self.assertTrue(provider._model_specific_ready)
        self.assertTrue(provider.accepted_model_specific_residency())
        self.assertEqual(
            [call.args[0] for call in worker.call.await_args_list],
            ["gpu_identity", "gpu_identity", "cuda_residency",
             "gpu_identity", "cuda_residency"])

    async def test_ready_fallback_failure_leaves_model_authority_fenced(self):
        config = self.config("/tmp")
        provider = CoEdITProvider(config)
        runner = ProcessIdentity(123, 456)
        identity = {"gpu_uuid": "GPU-test", "runner_pid": 123,
                    "runner_start_time": 456, "cuda_nvml_agree": True}
        foreign_identity = {"gpu_uuid": "GPU-test", "runner_pid": 999,
                            "runner_start_time": 456, "cuda_nvml_agree": True}
        witness = {"model_cuda_device": "cuda:0", "witness_exists": True,
                   "witness_cuda_device": "cuda:0"}
        worker = AsyncMock()
        worker.child_identity = runner
        worker.call.side_effect = [identity, identity, witness, foreign_identity]
        provider.worker = worker
        object.__setattr__(config.gpu_proof, "residency_for_runner",
                           AsyncMock(side_effect=_ResidencyPending("absent")))
        supervisor = config.gpu_proof.expected_supervisor
        provider._preload_memory = GPUMemoryObservation(
            "GPU-test", supervisor, 1, 2, 1000, 400, 600)
        object.__setattr__(config.gpu_proof, "memory", AsyncMock(return_value=
            GPUMemoryObservation("GPU-test", supervisor, 3, 4, 1000, 500, 500)))
        now = [0.0]
        async def expire_settlement(delay):
            now[0] += 5.0
        provider._residency_clock = lambda: now[0]
        provider._residency_sleep = expire_settlement

        with self.assertRaises(RuntimeError):
            await provider.ready()

        self.assertFalse(provider._ready)
        self.assertFalse(provider._model_specific_ready)
        self.assertFalse(provider.accepted_model_specific_residency())
        self.assertEqual(
            [call.args[0] for call in worker.call.await_args_list],
            ["gpu_identity", "gpu_identity", "cuda_residency", "gpu_identity"])

    async def test_ready_settles_empty_gpu_runner_but_never_grants_authority_on_persistent_or_uncertain_proof(self):
        config=self.config("/tmp"); provider=CoEdITProvider(config)
        runner=ProcessIdentity(123,456)
        worker=AsyncMock(); worker.child_identity=runner
        worker.call.return_value={"gpu_uuid":"GPU-test","runner_pid":123,"runner_start_time":456,"cuda_nvml_agree":True}
        provider.worker=worker
        now, calls = [0.0], [0]
        async def sleep(delay): now[0] += delay
        async def settles():
            calls[0] += 1
            if calls[0] < 3: raise _ResidencyPending("GPU has no resident runner")
            return ResidencyEvidence("GPU-test",config.gpu_proof.expected_supervisor,(runner,))
        object.__setattr__(config.gpu_proof, "residency", settles)
        provider._residency_clock=lambda: now[0]; provider._residency_sleep=sleep
        await provider.ready(); self.assertTrue(provider._ready)
        provider._ready=False
        async def persistent(): raise _ResidencyPending("GPU has no resident runner")
        object.__setattr__(config.gpu_proof, "residency", persistent)
        with self.assertRaises(GPUProofError): await provider.ready()
        self.assertFalse(provider._ready)

    async def test_ready_settlement_cancellation_propagates_without_session_authority(self):
        config=self.config("/tmp"); provider=CoEdITProvider(config)
        runner=ProcessIdentity(123,456)
        worker=AsyncMock(); worker.child_identity=runner
        worker.call.return_value={"gpu_uuid":"GPU-test","runner_pid":123,"runner_start_time":456,"cuda_nvml_agree":True}
        provider.worker=worker
        async def empty(): raise _ResidencyPending("GPU has no resident runner")
        sleeping=asyncio.Event()
        async def sleep(delay):
            sleeping.set()
            await asyncio.Event().wait()
        object.__setattr__(config.gpu_proof, "residency", empty)
        provider._residency_sleep=sleep
        task=asyncio.create_task(provider.ready())
        await sleeping.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertFalse(provider._ready)
        async def uncertain(): raise GPUProofError("supervisor PID was reused or exited")
        object.__setattr__(config.gpu_proof, "residency", uncertain)
        with self.assertRaises(GPUProofError): await provider.ready()
        self.assertFalse(provider._ready)

    async def test_failed_cleanup_retains_provider_worker(self):
        provider=CoEdITProvider(self.config("/tmp")); worker=AsyncMock(); worker.close.side_effect=RuntimeError("uncertain")
        provider.worker=worker
        with self.assertRaises(RuntimeError): await provider.unload()
        self.assertIs(provider.worker,worker)

    async def test_concurrent_unload_closes_worker_once_after_batcher_fence(self):
        config = self.config("/tmp")
        provider = CoEdITProvider(config)
        worker = AsyncMock()
        worker.frame_limit = 4096
        provider.worker = worker
        provider._batcher = CoEdITBatcher(worker, 1, 0, config.max_output_tokens)
        await asyncio.gather(provider.unload(), provider.unload())
        worker.close.assert_awaited_once()
        self.assertTrue(await provider.verify_cleanup())

    async def test_cancelled_unload_retains_cleanup_that_closes_worker_before_joining_batcher(self):
        provider = CoEdITProvider(self.config("/tmp"))
        worker_closed = asyncio.Event()
        batcher_release = asyncio.Event()
        worker = AsyncMock()
        async def close_worker(): worker_closed.set()
        worker.close.side_effect = close_worker
        batcher = AsyncMock()
        batcher.fence = Mock()
        batcher.join.side_effect = batcher_release.wait
        provider.worker, provider._batcher = worker, batcher
        unload = asyncio.create_task(provider.unload())
        await worker_closed.wait()
        unload.cancel()
        with self.assertRaises(asyncio.CancelledError): await unload
        self.assertFalse(provider._cleanup_task.done())
        batcher_release.set()
        await provider.unload()
        worker.close.assert_awaited_once()
        self.assertTrue(await provider.verify_cleanup())

    async def test_failed_reready_fences_execute(self):
        config=self.config("/tmp"); provider=CoEdITProvider(config); provider._ready=True
        worker=AsyncMock(); worker.call.return_value={"gpu_uuid":"wrong"}; provider.worker=worker
        with self.assertRaises(RuntimeError): await provider.ready()
        self.assertFalse(provider._ready)
        with self.assertRaises(RuntimeError): await provider.execute("request",b'{"instruction":"fix","texts":["text"]}')

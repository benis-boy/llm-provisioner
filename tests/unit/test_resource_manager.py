"""Deterministic precursor tests; no real adapter is used."""

import asyncio
import unittest

from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager, ResourceManagerError
from services.llm.resource_manager.protocol import EventKind, ProviderResponse
from services.llm.providers.python_process import WorkerRequestValidationError


def profile(model=ModelId.SMOLLM, p=1, context=128):
    shape = {"context_size": context} if model is ModelId.SMOLLM else {"bucket_identity": "bucket"}
    return CapacityProfile(model, "gpu", "manifest", "model", "runtime", "adapter", "profile",
                           p, p, p, 20, (SampleMetadata(1, 0, 1, 1, 1, (1,)),), **shape)


class FakeProvider:
    def __init__(self):
        self.release = asyncio.Event(); self.started = asyncio.Event(); self.calls = []
        self.ignore_cancel = False; self.raise_cancelled = False; self.cleanup_ok = True
        self.validation_gate = None; self.validation_started = asyncio.Event()
        self.cancel_gate = asyncio.Event()

    async def validate(self, profile): pass
    async def load(self, profile): self.calls.append("load")
    async def ready(self): pass
    async def validate_input(self, payload, *, context_size, bucket_identity):
        self.validation_started.set()
        if self.validation_gate: await self.validation_gate.wait()
        if payload == b"bad": raise ValueError("bad")
    async def execute(self, request_id, payload):
        self.calls.append(("execute", request_id)); self.started.set()
        if self.raise_cancelled: raise asyncio.CancelledError()
        await self.release.wait()
        return ProviderResponse(b"result", 7, True)
    async def cancel(self, request_id):
        self.calls.append(("cancel", request_id))
        if self.ignore_cancel: await self.cancel_gate.wait()
    async def unload(self): self.calls.append("unload")
    async def verify_cleanup(self): return self.cleanup_ok


class ResourceManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.provider = FakeProvider()
        self.rm = ResourceManager(cleanup_timeout=.03, stop_timeout=.03, max_events=8)
        self.session = await self.rm.start_session("s", ModelId.SMOLLM, profile(), self.provider, idempotency_key="start")

    async def asyncTearDown(self):
        self.provider.release.set()
        self.provider.cancel_gate.set()
        if self.rm._session is not None:
            try:
                await self.rm.stop_session(self.rm._session.session_token, idempotency_key="teardown")
            except ResourceManagerError:
                pass
        await asyncio.sleep(0)

    async def submit(self, rid, aid, payload=b"x", key=None):
        return await self.rm.submit(self.session.session_token, rid, aid, payload,
                                    idempotency_key=key or rid + aid, context_size=128)

    async def test_bound_duplicate_and_conflict(self):
        first = await self.submit("r1", "a1", key="k1")
        self.assertEqual(first, await self.submit("r1", "a1", key="k1"))
        await self.submit("r2", "a2", key="k2")
        self.assertTrue((await self.submit("r3", "a3", key="k3")).backpressure)
        with self.assertRaises(ResourceManagerError): await self.submit("r1", "a1", b"changed", "k1")

    async def test_validation_context_and_response_shape(self):
        with self.assertRaises(ResourceManagerError): await self.submit("r", "a", b"", "bad")
        with self.assertRaises(ResourceManagerError):
            await self.rm.submit(self.session.session_token, "r2", "a2", b"x",
                                 idempotency_key="context", context_size=129)
        with self.assertRaises(ResourceManagerError): await self.rm.submit(self.session.session_token, "r", "a", b"bad", idempotency_key="bad", context_size=128)

    async def test_child_validation_failure_releases_active_slot_and_session(self):
        child_calls = 0
        async def execute(request_id, payload):
            nonlocal child_calls
            child_calls += 1
            if payload == b"too-long":
                raise WorkerRequestValidationError("request validation")
            return ProviderResponse(b"result")
        self.provider.execute = execute
        await self.submit("bad-child", "a", b"too-long")
        await asyncio.sleep(0)
        self.assertEqual(child_calls, 1)
        self.assertEqual(len(self.rm._active), 0)
        self.assertEqual(len(self.rm._buffer), 0)
        accepted = await self.submit("after", "a", b"valid")
        self.assertTrue(accepted.accepted)

    async def test_buffered_event_is_not_admission(self):
        await self.submit("r1", "a1"); await asyncio.sleep(0)
        await self.submit("r2", "a2"); await self.submit("r3", "a3")
        events = list(self.rm._events[self.session.session_token])
        self.assertEqual([e.kind for e in events], [EventKind.ADMISSION, EventKind.BUFFERED])

    async def test_buffer_cancel_and_active_cancel_hold_slot(self):
        await self.submit("r1", "a1"); await asyncio.sleep(0)
        await self.submit("r2", "a2"); self.assertTrue(await self.rm.cancel_request(self.session.session_token, "r2", idempotency_key="c2"))
        self.assertEqual((await self.rm.get_capacity(self.session.session_token)).free_slots, 1)
        self.assertTrue(await self.rm.cancel_request(self.session.session_token, "r1", idempotency_key="c1"))
        self.assertEqual((await self.rm.get_capacity(self.session.session_token)).free_slots, 1)
        self.provider.release.set(); await asyncio.sleep(0.01)

    async def test_cleanup_waits_execute_and_timeout_never_unloads(self):
        await self.submit("r1", "a1"); await self.provider.started.wait()
        with self.assertRaises(ResourceManagerError):
            await self.rm.start_session("new", ModelId.COEDIT, profile(ModelId.COEDIT), self.provider, idempotency_key="new")
        self.assertNotIn("unload", self.provider.calls)

    async def test_timed_out_load_blocks_reload_until_completion_or_fail_closed(self):
        gate = asyncio.Event()
        entered = asyncio.Event()

        class HungLoad(FakeProvider):
            async def load(self, profile):
                self.calls.append("load")
                entered.set()
                await gate.wait()

        first = HungLoad()
        rm = ResourceManager(cleanup_timeout=.03, stop_timeout=.03, load_timeout=.02)
        loading = asyncio.create_task(rm.start_session("s", ModelId.SMOLLM, profile(), first,
                                                        idempotency_key="start"))
        await entered.wait()
        try:
            with self.assertRaises(ResourceManagerError) as failure:
                await loading
            self.assertEqual(failure.exception.failure.code, "cleanup_timeout")
            # Initial-load failure is never an exposed session token.  Cleanup
            # failure closes the manager permanently, even when the owned load
            # eventually returns.
            self.assertIsNone(rm._session)
            self.assertFalse(rm._available)
            replacement = FakeProvider()
            with self.assertRaises(ResourceManagerError):
                await rm.start_session("new", ModelId.COEDIT, profile(ModelId.COEDIT), replacement,
                                       idempotency_key="new")
            self.assertNotIn("load", replacement.calls)
        finally:
            gate.set()
            await asyncio.gather(*tuple(rm._lifecycle_tasks), return_exceptions=True)
            await asyncio.sleep(0)

    async def test_cleanup_failure_blocks_new_load(self):
        self.provider.release.set(); self.provider.cleanup_ok = False
        await self.submit("r", "a"); await asyncio.sleep(.01)
        with self.assertRaises(ResourceManagerError):
            await self.rm.start_session("n", ModelId.COEDIT, profile(ModelId.COEDIT), self.provider, idempotency_key="n")

    async def test_failed_ready_is_cleaned_and_fresh_start_is_allowed(self):
        class FailedReady(FakeProvider):
            async def ready(self):
                self.calls.append("ready")
                raise RuntimeError("not ready")
        failed = FailedReady()
        rm = ResourceManager(cleanup_timeout=.03, load_timeout=.03)
        with self.assertRaises(ResourceManagerError) as result:
            await rm.start_session("failed", ModelId.SMOLLM, profile(), failed, idempotency_key="failed")
        self.assertEqual(result.exception.failure.code, "model_load_failed")
        self.assertIn("unload", failed.calls)
        fresh = FakeProvider()
        session = await rm.start_session("fresh", ModelId.SMOLLM, profile(), fresh, idempotency_key="fresh")
        self.assertEqual(session.scheduler_id, "fresh")

    async def test_stale_and_old_replay_are_rejected(self):
        token = self.session.session_token; self.provider.release.set(); await self.submit("r", "a"); await asyncio.sleep(.01)
        replacement = await self.rm.start_session("n", ModelId.COEDIT, profile(ModelId.COEDIT), self.provider, idempotency_key="n")
        self.assertNotEqual(replacement.session_token, self.session.session_token)
        with self.assertRaises(ResourceManagerError): await self.rm.get_capacity(token)

    async def test_same_pair_different_keys_conflicts(self):
        await self.submit("r", "a", key="one")
        with self.assertRaises(ResourceManagerError): await self.submit("r", "a", b"different", "two")

    async def test_concurrent_validation_replays_exact_pair(self):
        self.provider.validation_gate = asyncio.Event()
        a = asyncio.create_task(self.submit("r", "a", key="one")); await self.provider.validation_started.wait()
        b = asyncio.create_task(self.submit("r", "a", key="two")); await asyncio.sleep(0)
        self.provider.validation_gate.set(); self.assertEqual((await a).attempt, (await b).attempt)

    async def test_session_fence_during_validation(self):
        self.provider.validation_gate = asyncio.Event(); task = asyncio.create_task(self.submit("r", "a")); await self.provider.validation_started.wait()
        self.provider.validation_gate.set(); self.provider.release.set()
        replacement = await self.rm.start_session("n", ModelId.COEDIT, profile(ModelId.COEDIT), self.provider, idempotency_key="n")
        self.assertNotEqual(replacement.session_token, self.session.session_token)
        with self.assertRaises(ResourceManagerError): await task

    async def test_cancel_during_validation_then_release_cannot_execute(self):
        self.provider.validation_gate = asyncio.Event()
        submit = asyncio.create_task(self.submit("r", "a"))
        await self.provider.validation_started.wait()
        self.assertTrue(await self.rm.cancel_request(self.session.session_token, "r", idempotency_key="cancel"))
        self.provider.validation_gate.set()
        with self.assertRaises(ResourceManagerError) as failure:
            await submit
        self.assertEqual(failure.exception.failure.code, "request_cancelled")
        await asyncio.sleep(.01)
        self.assertFalse(any(isinstance(call, tuple) and call[0] == "execute" for call in self.provider.calls))

    async def test_cancelled_error_releases_exact_slot(self):
        self.provider.raise_cancelled = True; await self.submit("r", "a"); await asyncio.sleep(.01)
        self.assertEqual((await self.rm.get_capacity(self.session.session_token)).free_slots, 2)

    async def test_nonretryable_provider_failure_preserved(self):
        class Bad(FakeProvider):
            async def execute(self, request_id, payload):
                raise ResourceManagerError(__import__('services.llm.resource_manager.protocol', fromlist=['Failure']).Failure("bad", "bad", False))
        bad = Bad(); rm = ResourceManager(); s = await rm.start_session("s", ModelId.SMOLLM, profile(), bad, idempotency_key="s")
        await rm.submit(s.session_token, "r", "a", b"x", idempotency_key="r", context_size=128); await asyncio.sleep(.01)
        failure = next(event.failure for event in reversed(rm._events[s.session_token])
                        if event.failure is not None)
        self.assertFalse(failure.retryable)

    async def test_incomplete_timing_is_null(self):
        class Incomplete(FakeProvider):
            async def execute(self, request_id, payload): return ProviderResponse(b"x", 99, False)
        p = Incomplete(); rm = ResourceManager(); s = await rm.start_session("s", ModelId.SMOLLM, profile(), p, idempotency_key="s")
        await rm.submit(s.session_token, "r", "a", b"x", idempotency_key="r", context_size=128); await asyncio.sleep(.01)
        event = rm._events[s.session_token][-1]; self.assertIsNone(event.time_on_gpu_ms); self.assertFalse(event.gpu_timing_complete)

    async def test_stop_idempotent_and_watcher_terminates(self):
        await self.rm.stop_session(self.session.session_token, idempotency_key="stop")
        await self.rm.stop_session(self.session.session_token, idempotency_key="stop")
        events = [e async for e in self.rm.watch_progress(self.session.session_token)]
        self.assertEqual(events[-1].kind, EventKind.SESSION_INVALIDATED)

    async def test_required_mutation_keys_and_conflicts(self):
        with self.assertRaises(TypeError): await self.rm.stop_session(self.session.session_token)
        await self.rm.stop_session(self.session.session_token, idempotency_key="stop")
        with self.assertRaises(ResourceManagerError):
            await self.rm.stop_session(self.session.session_token, reason="x", idempotency_key="stop")

    async def test_invalid_constructor_and_profile(self):
        with self.assertRaises(ValueError): ResourceManager(cleanup_timeout=True)
        with self.assertRaises(ValueError): ResourceManager(max_events=True)
        with self.assertRaises(ResourceManagerError): await self.rm.start_session("x", ModelId.COEDIT, profile(), self.provider, idempotency_key="bad-model")

    async def test_cursor_validation(self):
        with self.assertRaises(ResourceManagerError): await self.rm.watch_progress(self.session.session_token, -1).__anext__()
        with self.assertRaises(ResourceManagerError): await self.rm.watch_progress(self.session.session_token, 99).__anext__()

    async def test_start_replay_returns_only_active_session(self):
        replay = await self.rm.start_session("s", ModelId.SMOLLM, profile(), self.provider, idempotency_key="start")
        self.assertEqual(replay, self.session)

    async def test_stop_key_conflict_is_rejected(self):
        await self.rm.stop_session(self.session.session_token, reason="one", idempotency_key="stop")
        with self.assertRaises(ResourceManagerError):
            await self.rm.stop_session(self.session.session_token, reason="two", idempotency_key="stop")

    async def test_late_old_callback_cannot_pop_new_attempt(self):
        await self.submit("same", "old"); await asyncio.sleep(0)
        old = next(iter(self.rm._active.values()))
        self.rm._active[(self.session.session_token, "same", "new")] = type(
            "InjectedAttempt", (), {"cancelled": False})()
        self.provider.release.set(); await asyncio.sleep(.01)
        self.assertIn((self.session.session_token, "same", "new"), self.rm._active)
        self.rm._active.pop((self.session.session_token, "same", "new"), None)

    async def test_provider_cleanup_verification_failure_is_fail_closed(self):
        self.provider.release.set(); self.provider.cleanup_ok = False
        await self.submit("r", "a"); await asyncio.sleep(.01)
        with self.assertRaises(ResourceManagerError):
            await self.rm.start_session("n", ModelId.COEDIT, profile(ModelId.COEDIT), self.provider, idempotency_key="n")
        with self.assertRaises(ResourceManagerError):
            await self.rm.start_session("again", ModelId.SMOLLM, profile(), self.provider, idempotency_key="again")

    async def test_provider_complete_negative_timing_is_malformed(self):
        class Malformed(FakeProvider):
            async def execute(self, request_id, payload): return ProviderResponse(b"x", -1, True)
        provider = Malformed(); rm = ResourceManager(); session = await rm.start_session("s", ModelId.SMOLLM, profile(), provider, idempotency_key="s")
        await rm.submit(session.session_token, "r", "a", b"x", idempotency_key="r", context_size=128); await asyncio.sleep(.01)
        self.assertEqual(rm._events[session.session_token][-1].failure.code, "malformed_provider_response")


if __name__ == "__main__": unittest.main()

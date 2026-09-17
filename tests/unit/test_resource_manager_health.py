import asyncio
import unittest
from dataclasses import FrozenInstanceError

from services.llm.health import DEPENDENCIES
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.core import ResourceManager, ResourceManagerError
from services.llm.resource_manager.health import create_boundary
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.state import ResourceManagerState
from services.llm.resource_manager.protocol import ProviderResponse


def profile():
    return CapacityProfile(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter",
                           "measured", 1, 1, 1, 10,
                           (SampleMetadata(1, 1, 1, 1, 1, (1,)),), context_size=128)


class Provider:
    async def validate(self, _profile): pass
    async def load(self, _profile): pass
    async def ready(self): pass
    async def unload(self): pass
    async def verify_cleanup(self): return True
    async def cancel(self, _request_id): pass
    async def validate_input(self, payload, *, context_size=None, bucket_identity=None): pass
    async def execute(self, _request_id, _payload): return ProviderResponse(b"ok")


class GatedProvider(Provider):
    def __init__(self):
        self.validate_started = asyncio.Event()
        self.validate_gate = asyncio.Event()
        self.load_started = asyncio.Event()
        self.load_gate = asyncio.Event()
        self.ready_started = asyncio.Event()
        self.ready_gate = asyncio.Event()
        self.unload_started = asyncio.Event()
        self.unload_gate = asyncio.Event()
        self.calls = []

    async def validate(self, _profile):
        self.calls.append("validate")
        self.validate_started.set()
        await self.validate_gate.wait()

    async def load(self, _profile):
        self.calls.append("load")
        self.load_started.set()
        await self.load_gate.wait()

    async def ready(self):
        self.calls.append("ready")
        self.ready_started.set()
        await self.ready_gate.wait()

    async def unload(self):
        self.calls.append("unload")
        self.unload_started.set()
        await self.unload_gate.wait()


class ResourceManagerHealthTests(unittest.IsolatedAsyncioTestCase):
    def external(self, **overrides):
        states = {name: True for name in DEPENDENCIES}
        states.update(overrides)
        return states

    async def test_authoritative_lifecycle_states_are_fenced(self):
        rm = ResourceManager()
        self.assertEqual(rm.snapshot(), ResourceManagerState("startup", True, False, 0))

    async def test_snapshots_are_immutable_and_loading_gates_are_authoritative(self):
        initial = ResourceManager().snapshot()
        with self.assertRaises(FrozenInstanceError):
            initial.phase = "stable"

        rm = ResourceManager()
        provider = GatedProvider()
        task = asyncio.create_task(rm.start_session("s", ModelId.SMOLLM, profile(), provider,
                                                     idempotency_key="start"))
        await provider.validate_started.wait()
        self.assertEqual(rm.snapshot(), ResourceManagerState("loading", False, True, 0))
        provider.validate_gate.set()
        await provider.load_started.wait()
        self.assertEqual(rm.snapshot().phase, "loading")
        provider.load_gate.set()
        await provider.ready_started.wait()
        self.assertEqual(rm.snapshot().phase, "loading")
        provider.ready_gate.set()
        session = await task
        self.assertEqual(rm.snapshot(), ResourceManagerState("stable", True, True, 1))
        stopping = asyncio.create_task(rm.stop_session(session.session_token, idempotency_key="stop"))
        await provider.unload_started.wait()
        self.assertEqual(rm.snapshot().phase, "unloading")
        provider.unload_gate.set()
        await stopping
        self.assertEqual(rm.snapshot(), ResourceManagerState("startup", True, False, 1))

    async def test_replacement_unloads_before_new_session_can_load(self):
        rm = ResourceManager()
        old = Provider()
        session = await rm.start_session("old", ModelId.SMOLLM, profile(), old, idempotency_key="old")
        replacement = GatedProvider()
        replacement.validate_gate.set(); replacement.load_gate.set(); replacement.ready_gate.set()
        old.unload_gate = asyncio.Event()  # type: ignore[attr-defined]
        old.unload_started = asyncio.Event()  # type: ignore[attr-defined]

        async def blocked_unload():
            old.unload_started.set()
            await old.unload_gate.wait()
        old.unload = blocked_unload  # type: ignore[method-assign]

        task = asyncio.create_task(rm.start_session("new", ModelId.SMOLLM, profile(), replacement,
                                                     idempotency_key="new"))
        await old.unload_started.wait()
        self.assertEqual(rm.snapshot(), ResourceManagerState("unloading", False, False, 1))
        self.assertNotIn("load", replacement.calls)
        old.unload_gate.set()
        new_session = await task
        self.assertIn("load", replacement.calls)
        await rm.stop_session(new_session.session_token, idempotency_key="stop")

    async def test_failed_initial_load_is_cleaned_and_returns_to_startup(self):
        class Failed(Provider):
            async def load(self, _profile):
                raise RuntimeError("load failed")
        provider = Failed()
        rm = ResourceManager()
        with self.assertRaises(Exception):
            await rm.start_session("s", ModelId.SMOLLM, profile(), provider, idempotency_key="start")
        self.assertEqual(rm.snapshot(), ResourceManagerState("startup", True, False, 0))
        loading = asyncio.create_task(rm.start_session("s", ModelId.SMOLLM, profile(), Provider(),
                                                        idempotency_key="start"))
        await asyncio.sleep(0)
        self.assertEqual(rm.snapshot().phase, "loading")
        session = await loading
        self.assertEqual(rm.snapshot(), ResourceManagerState("stable", True, True, 1))
        stopping = asyncio.create_task(rm.stop_session(session.session_token, idempotency_key="stop"))
        await asyncio.sleep(0)
        self.assertEqual(rm.snapshot().phase, "unloading")
        await stopping
        self.assertEqual(rm.snapshot(), ResourceManagerState("startup", True, False, 1))

    async def test_external_proof_cannot_override_lifecycle_or_profile(self):
        rm = ResourceManager()
        boundary = create_boundary(rm, lambda: self.external(profile=False))
        ok, reasons, _ = await boundary.readiness()
        self.assertFalse(ok)
        self.assertEqual(reasons[0], {"phase": "startup"})
        await boundary.close()

        session = await rm.start_session("s", ModelId.SMOLLM, profile(), Provider(), idempotency_key="start")
        boundary = create_boundary(rm, lambda: self.external(profile=False))
        ok, reasons, _ = await boundary.readiness()
        self.assertFalse(ok)
        self.assertIn({"dependency": "profile", "reason": "unavailable"}, reasons)
        await boundary.close()
        await rm.stop_session(session.session_token, idempotency_key="stop")

    async def test_malformed_or_async_external_state_fails_closed(self):
        rm = ResourceManager()
        async def external():
            return self.external()
        boundary = create_boundary(rm, external)
        ok, _, states = await boundary.readiness()
        self.assertFalse(ok)
        self.assertEqual(states["sqlite"].reason, "state_missing")
        await boundary.close()

    async def test_lifecycle_is_read_after_probe_and_stop_wins(self):
        rm = ResourceManager()
        started = asyncio.Event()
        release = asyncio.Event()

        async def probe():
            started.set()
            await release.wait()
            return True

        session = await rm.start_session("s", ModelId.SMOLLM, profile(), Provider(), idempotency_key="start")
        boundary = create_boundary(rm, lambda: self.external(), probes={"gpu": probe})
        check = asyncio.create_task(boundary.readiness())
        await started.wait()
        stopping = asyncio.create_task(rm.stop_session(session.session_token, idempotency_key="stop"))
        await asyncio.sleep(0)
        release.set()
        result = await check
        await stopping
        self.assertFalse(result[0])
        self.assertEqual(result[1][0], {"phase": "unloading"})
        await boundary.close()

    async def test_replacement_during_probe_cannot_inherit_old_session_proofs(self):
        rm = ResourceManager()
        old = await rm.start_session("old", ModelId.SMOLLM, profile(), Provider(), idempotency_key="old")
        started, release = asyncio.Event(), asyncio.Event()

        async def probe():
            started.set()
            await release.wait()
            return True

        boundary = create_boundary(rm, lambda: self.external(), probes={"gpu": probe})
        check = asyncio.create_task(boundary.readiness())
        await started.wait()
        replacement = asyncio.create_task(rm.start_session("new", ModelId.SMOLLM, profile(), Provider(),
                                                            idempotency_key="new"))
        await asyncio.sleep(0)
        release.set()
        ok, reasons, states = await check
        new = await replacement
        self.assertFalse(ok)
        self.assertIn({"phase": "unloading"}, reasons)
        self.assertFalse(states["adapter"].ok)
        await boundary.close()
        await rm.stop_session(new.session_token, idempotency_key="stop")
        self.assertNotEqual(old.session_token, new.session_token)

    async def test_external_proof_rechecked_after_probe_and_false_cannot_be_upgraded(self):
        rm = ResourceManager()
        session = await rm.start_session("s", ModelId.SMOLLM, profile(), Provider(), idempotency_key="start")
        external = self.external()
        started, release = asyncio.Event(), asyncio.Event()

        async def probe():
            started.set()
            await release.wait()
            return True

        boundary = create_boundary(rm, lambda: external, probes={"profile": probe})
        check = asyncio.create_task(boundary.readiness())
        await started.wait()
        external["profile"] = False
        release.set()
        ok, reasons, states = await check
        self.assertFalse(ok)
        self.assertEqual(states["profile"].reason, "unavailable")
        self.assertIn({"dependency": "profile", "reason": "unavailable"}, reasons)
        await boundary.close()
        await rm.stop_session(session.session_token, idempotency_key="stop")

    async def test_malformed_lifecycle_fields_fail_closed(self):
        class Alternate:
            def snapshot(self):
                return ResourceManagerState("not-a-phase", True, True, 1)

        boundary = create_boundary(Alternate(), lambda: self.external())
        ok, reasons, states = await boundary.readiness()
        self.assertFalse(ok)
        self.assertEqual(reasons, [{"phase": "state_malformed"}])
        self.assertTrue(all(state.reason == "state_malformed" for state in states.values()))
        await boundary.close()

    async def test_cancelled_cleanup_is_failed_closed(self):
        class BlockedCleanup(Provider):
            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def unload(self):
                self.started.set()
                await self.release.wait()

        rm = ResourceManager()
        provider = BlockedCleanup()
        session = await rm.start_session("s", ModelId.SMOLLM, profile(), provider, idempotency_key="start")
        stopping = asyncio.create_task(rm.stop_session(session.session_token, idempotency_key="stop"))
        await provider.started.wait()
        stopping.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await stopping
        self.assertEqual(rm.snapshot().phase, "cleanup_failed")
        boundary = create_boundary(rm, lambda: self.external())
        ok, _, _ = await boundary.readiness()
        self.assertFalse(ok)
        provider.release.set()
        await boundary.close()

    async def test_cancelled_load_stays_fenced_after_late_completion(self):
        class CancellableLoad(Provider):
            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def load(self, _profile):
                self.started.set()
                await self.release.wait()

        provider = CancellableLoad()
        rm = ResourceManager(load_timeout=1, cleanup_timeout=.02)
        loading = asyncio.create_task(rm.start_session("s", ModelId.SMOLLM, profile(), provider,
                                                       idempotency_key="start"))
        await provider.started.wait()
        loading.cancel()
        with self.assertRaises(ResourceManagerError) as failure:
            await loading
        self.assertEqual(failure.exception.failure.code, "cleanup_timeout")
        self.assertEqual(rm.snapshot().phase, "cleanup_failed")
        provider.release.set()
        await asyncio.gather(*tuple(rm._lifecycle_tasks), return_exceptions=True)
        self.assertNotEqual(rm.snapshot().phase, "stable")
        boundary = create_boundary(rm, lambda: self.external())
        ready, _, _ = await boundary.readiness()
        self.assertFalse(ready)
        await boundary.close()

    async def test_timed_out_load_late_completion_cannot_revive_readiness(self):
        class TimedOutLoad(Provider):
            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def load(self, _profile):
                self.started.set()
                await self.release.wait()

        provider = TimedOutLoad()
        rm = ResourceManager(load_timeout=.01, cleanup_timeout=.01)
        with self.assertRaises(ResourceManagerError) as failure:
            await rm.start_session("s", ModelId.SMOLLM, profile(), provider,
                                   idempotency_key="start")
        self.assertEqual(failure.exception.failure.code, "cleanup_timeout")
        self.assertEqual(rm.snapshot().phase, "cleanup_failed")
        provider.release.set()
        await asyncio.gather(*tuple(rm._lifecycle_tasks), return_exceptions=True)
        self.assertEqual(rm.snapshot().phase, "cleanup_failed")
        boundary = create_boundary(rm, lambda: self.external())
        ready, reasons, _ = await boundary.readiness()
        self.assertFalse(ready)
        self.assertIn({"phase": "cleanup_failed"}, reasons)
        await boundary.close()

    async def test_malformed_external_boolean_cannot_be_upgraded_by_probe(self):
        rm = ResourceManager()
        session = await rm.start_session("s", ModelId.SMOLLM, profile(), Provider(), idempotency_key="start")
        boundary = create_boundary(rm, lambda: {**self.external(), "profile": {"ok": 1}},
                                   probes={"profile": lambda: True})
        ready, reasons, states = await boundary.readiness()
        self.assertFalse(ready)
        self.assertEqual(states["profile"].reason, "state_malformed")
        self.assertIn({"dependency": "profile", "reason": "state_malformed"}, reasons)
        await boundary.close()
        await rm.stop_session(session.session_token, idempotency_key="stop")


if __name__ == "__main__":
    unittest.main()

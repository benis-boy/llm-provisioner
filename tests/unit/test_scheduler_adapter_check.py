import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from services.llm.resource_manager.protocol import EventKind, ProgressEvent, ProviderResponse, SessionInfo
from services.llm.providers.gpu import ProcessIdentity
from services.llm.provisioning.artifacts import SPECS, inventory, manifest, spec
from services.llm.provisioning.volume import provision
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from tools.compatibility import scheduler_adapter_check as check


class FakeProvider:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.gate = False
        self.unloaded = False

    async def validate(self, profile): pass
    async def load(self, profile): pass
    async def ready(self): pass
    async def validate_input(self, payload, *, context_size, bucket_identity):
        if (payload != check.coedit._request("Short text.") or context_size is not None
                or bucket_identity != check.coedit.BUCKET):
            raise ValueError("unexpected dispatch contract")
    async def execute(self, request_id, payload):
        if self.gate:
            self.entered.set()
            await self.release.wait()
        return ProviderResponse(json.dumps({"texts": ["corrected"]}).encode(), None, False)
    async def cancel(self, request_id): pass
    async def unload(self): self.unloaded = True
    async def verify_cleanup(self): return self.unloaded


def _profile():
    return CapacityProfile(ModelId.COEDIT, "GPU-test", "b" * 64, "a" * 64,
                           "candidate:test", "candidate", "unmeasured-test",
                           1, 1, 1, 0, (SampleMetadata(1, 0, 0, 0, 0, ()),),
                           bucket_identity=check.coedit.BUCKET)


class SchedulerAdapterCheckTests(unittest.IsolatedAsyncioTestCase):
    def test_selected_identity_accepts_provisioned_runtime_root_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "CoEdIT"
            source.mkdir()
            for name in SPECS["CoEdIT"]:
                (source / name).write_bytes(name.encode())
            selected = manifest({"CoEdIT": inventory(spec("CoEdIT", source))})
            volume = provision({"CoEdIT": source}, root / "volume")
            self.assertNotEqual(selected["models"]["CoEdIT"]["root"],
                                volume["models"]["CoEdIT"]["root"])
            self.assertEqual(check._selected_file_identity(selected["models"]["CoEdIT"]),
                             check._selected_file_identity(volume["models"]["CoEdIT"]))

    async def test_run_uses_real_scheduler_path_and_observes_late_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = FakeProvider()
            source = root / "CoEdIT"
            source.mkdir()
            selected = {"models": {"CoEdIT": {"model_id": "CoEdIT", "files": []}}}
            args = SimpleNamespace(models_root=root, manifest=root / "manifest.json",
                                   target_gpu_uuid="GPU-test", host_pid_namespace=True)
            async def cleanup():
                return True
            proof = SimpleNamespace(identity="GPU-test", supervisor_identity=ProcessIdentity(1, 1),
                                    cleanup=cleanup,
                                    residency=lambda: SimpleNamespace(runners=()))
            volume = {"models": selected["models"], "manifest_sha256": "b" * 64}
            with patch.object(check, "_candidate_manifest", return_value=(selected, "a" * 64)), \
                 patch.object(check, "provision", return_value=volume), \
                 patch.object(check.LinuxGPUProof, "capture", return_value=proof), \
                 patch.object(check, "CoEdITProvider", return_value=provider), \
                 patch.object(check.coedit, "_runtime_identity", return_value="candidate:test"):
                result = await check.run(args)
            self.assertEqual(result["normal_status"], "done")
            self.assertEqual(result["cancelled_status"], "cancelled")
            self.assertEqual(result["receipt_count"], 1)
            self.assertEqual(result["cancelled_receipt_count"], 1)
            self.assertTrue(provider.unloaded)

    async def test_terminal_wait_ignores_cancelled_until_late_response_finished(self):
        session = SessionInfo("scheduler", "session", ModelId.COEDIT, 3)
        cancelled = ProgressEvent(1, 0, EventKind.CANCELLED, "request", "attempt", "session", 3)
        finished = ProgressEvent(2, 1, EventKind.RESPONSE_FINISHED, "request", "attempt", "session", 3,
                                 b"response")

        class RM:
            async def watch_progress(self, token):
                yield cancelled
                await asyncio.sleep(.02)
                yield finished

        event = await check._terminal_event(RM(), session, "request", "attempt")
        self.assertIs(event, finished)

    async def test_cleanup_is_a_conjunction_and_retains_unfinished_work(self):
        pending = set()

        class Scheduler:
            async def stop(self, reason):
                raise RuntimeError("stop failed")

        class RM:
            def snapshot(self):
                return SimpleNamespace(phase="cleanup_failed", available=False, session_present=False)

        class Provider:
            def __init__(self): self.release = asyncio.Event()
            async def unload(self): return None
            async def verify_cleanup(self): return False

        class Proof:
            async def cleanup(self): return False

        session = SimpleNamespace(session_token="session")
        cleanup = await check._cleanup_owned(Scheduler(), RM(), session, Provider(), Proof(), pending, .1)
        self.assertFalse(all(cleanup.values()))
        gate = asyncio.Event()
        self.assertFalse(await check._bounded_cleanup(gate.wait(), .01, pending))
        self.assertEqual(len(pending), 1)
        gate.set()
        await self._wait(lambda: not pending)

    async def test_bounded_cleanup_observes_late_failure(self):
        pending = set()
        gate = asyncio.Event()

        async def late_failure():
            await gate.wait()
            raise RuntimeError("late cleanup failure")

        self.assertFalse(await check._bounded_cleanup(late_failure(), .01, pending))
        gate.set()
        await self._wait(lambda: not pending)

    async def test_run_rejects_corrupt_publication_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = FakeProvider()
            source = root / "CoEdIT"
            source.mkdir()
            selected = {"models": {"CoEdIT": {"model_id": "CoEdIT", "files": []}}}
            args = SimpleNamespace(models_root=root, manifest=root / "manifest.json",
                                   target_gpu_uuid="GPU-test", host_pid_namespace=True)

            async def cleanup(): return True
            proof = SimpleNamespace(identity="GPU-test", supervisor_identity=ProcessIdentity(1, 1),
                                    cleanup=cleanup, residency=lambda: SimpleNamespace(runners=()))
            volume = {"models": selected["models"], "manifest_sha256": "b" * 64}
            original_publish = check.LocalPublisher.publish

            def corrupt_publish(publisher, request_id, attempt, reference, key):
                original_publish(publisher, request_id, attempt, reference, key)
                publisher.db.execute("UPDATE publication_receipts SET result_reference=?", ("0" * 64,))
                return True

            with patch.object(check, "_candidate_manifest", return_value=(selected, "a" * 64)), \
                 patch.object(check, "provision", return_value=volume), \
                 patch.object(check.LinuxGPUProof, "capture", return_value=proof), \
                 patch.object(check, "CoEdITProvider", return_value=provider), \
                 patch.object(check.coedit, "_runtime_identity", return_value="candidate:test"), \
                 patch.object(check.LocalPublisher, "publish", new=corrupt_publish):
                with self.assertRaises(check.CandidateFailure) as raised:
                    await check.run(args)
                self.assertEqual(raised.exception.stage, "verify_normal")
                self.assertEqual(raised.exception.code, "receipt_mismatch")

    async def test_run_rejects_invalid_late_response(self):
        class InvalidProvider(FakeProvider):
            def __init__(self):
                super().__init__()
                self.calls = 0

            async def execute(self, request_id, payload):
                self.calls += 1
                if self.gate:
                    self.entered.set()
                    await self.release.wait()
                if self.calls == 1:
                    return ProviderResponse(json.dumps({"texts": ["corrected"]}).encode(), None, False)
                return ProviderResponse(b"not-json", None, False)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = InvalidProvider()
            source = root / "CoEdIT"
            source.mkdir()
            selected = {"models": {"CoEdIT": {"model_id": "CoEdIT", "files": []}}}
            args = SimpleNamespace(models_root=root, manifest=root / "manifest.json",
                                   target_gpu_uuid="GPU-test", host_pid_namespace=True)
            async def cleanup(): return True
            proof = SimpleNamespace(identity="GPU-test", supervisor_identity=ProcessIdentity(1, 1),
                                    cleanup=cleanup, residency=lambda: SimpleNamespace(runners=()))
            volume = {"models": selected["models"], "manifest_sha256": "b" * 64}
            with patch.object(check, "_candidate_manifest", return_value=(selected, "a" * 64)), \
                 patch.object(check, "provision", return_value=volume), \
                 patch.object(check.LinuxGPUProof, "capture", return_value=proof), \
                 patch.object(check, "CoEdITProvider", return_value=provider), \
                 patch.object(check.coedit, "_runtime_identity", return_value="candidate:test"):
                with self.assertRaises(check.CandidateFailure) as raised:
                    await check.run(args)
                self.assertEqual(raised.exception.stage, "verify_cancel")
                self.assertEqual(raised.exception.code, "late_response_shape")

    async def test_run_rejects_invalid_normal_response(self):
        class InvalidNormalProvider(FakeProvider):
            async def execute(self, request_id, payload):
                return ProviderResponse(b"not-json", None, False)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = InvalidNormalProvider()
            source = root / "CoEdIT"
            source.mkdir()
            selected = {"models": {"CoEdIT": {"model_id": "CoEdIT", "files": []}}}
            args = SimpleNamespace(models_root=root, manifest=root / "manifest.json",
                                   target_gpu_uuid="GPU-test", host_pid_namespace=True)
            async def cleanup(): return True
            proof = SimpleNamespace(identity="GPU-test", supervisor_identity=ProcessIdentity(1, 1),
                                    cleanup=cleanup, residency=lambda: SimpleNamespace(runners=()))
            volume = {"models": selected["models"], "manifest_sha256": "b" * 64}
            with patch.object(check, "_candidate_manifest", return_value=(selected, "a" * 64)), \
                 patch.object(check, "provision", return_value=volume), \
                 patch.object(check.LinuxGPUProof, "capture", return_value=proof), \
                 patch.object(check, "CoEdITProvider", return_value=provider), \
                 patch.object(check.coedit, "_runtime_identity", return_value="candidate:test"):
                with self.assertRaises(check.CandidateFailure) as raised:
                    await check.run(args)
            self.assertEqual((raised.exception.stage, raised.exception.code),
                             ("verify_normal", "normal_response_shape"))
            self.assertEqual(set(raised.exception.cleanup),
                             {"scheduler", "rm_snapshot", "provider_verify", "gpu"})
            self.assertTrue(all(raised.exception.cleanup.values()))

    async def test_run_rejects_persisted_normal_result_that_differs_from_delegate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = FakeProvider()
            source = root / "CoEdIT"
            source.mkdir()
            selected = {"models": {"CoEdIT": {"model_id": "CoEdIT", "files": []}}}
            args = SimpleNamespace(models_root=root, manifest=root / "manifest.json",
                                   target_gpu_uuid="GPU-test", host_pid_namespace=True)
            async def cleanup(): return True
            proof = SimpleNamespace(identity="GPU-test", supervisor_identity=ProcessIdentity(1, 1),
                                    cleanup=cleanup, residency=lambda: SimpleNamespace(runners=()))
            volume = {"models": selected["models"], "manifest_sha256": "b" * 64}
            with patch.object(check, "_candidate_manifest", return_value=(selected, "a" * 64)), \
                 patch.object(check, "provision", return_value=volume), \
                 patch.object(check.LinuxGPUProof, "capture", return_value=proof), \
                 patch.object(check, "CoEdITProvider", return_value=provider), \
                 patch.object(check.coedit, "_runtime_identity", return_value="candidate:test"), \
                 patch.object(check.ResultStore, "read", return_value=b'{"texts":["other"]}'):
                with self.assertRaises(check.CandidateFailure) as raised:
                    await check.run(args)
            self.assertEqual((raised.exception.stage, raised.exception.code),
                             ("verify_normal", "normal_result_delegate_mismatch"))

    async def test_scheduler_owned_cleanup_observes_authoritative_rm_snapshot(self):
        pending = set()

        class Scheduler:
            async def stop(self, reason): return None

        class RM:
            def snapshot(self):
                return SimpleNamespace(phase="startup", available=True, session_present=False)

        class Provider:
            def __init__(self): self.release = asyncio.Event()
            async def verify_cleanup(self): return True

        class Proof:
            async def cleanup(self): return True

        cleanup = await check._cleanup_owned(Scheduler(), RM(), SimpleNamespace(session_token="session"),
                                             Provider(), Proof(), pending, .1)
        self.assertTrue(all(cleanup.values()))

    async def test_cleanup_does_not_probe_dependents_while_scheduler_stop_is_pending(self):
        pending = set()
        gate = asyncio.Event()

        class Scheduler:
            async def stop(self, reason): await gate.wait()

        class RM:
            def snapshot(self): self.fail("RM snapshot must not run")

        class Provider:
            def __init__(self): self.release = asyncio.Event()
            async def verify_cleanup(self): self.fail("provider probe must not run")

        class Proof:
            async def cleanup(self): self.fail("GPU probe must not run")

        cleanup = await check._cleanup_owned(Scheduler(), RM(), SimpleNamespace(session_token="session"),
                                             Provider(), Proof(), pending, .01)
        self.assertEqual(cleanup, {"scheduler": False, "rm_snapshot": False,
                                   "provider_verify": False, "gpu": False})
        self.assertFalse(await check._settle_cleanup(pending, .01))
        gate.set()
        await self._wait(lambda: not pending)

    async def _wait(self, predicate):
        for _ in range(100):
            if predicate():
                return
            await asyncio.sleep(.01)
        self.fail("condition did not become true")

    def test_candidate_profile_is_not_runtime_eligible(self):
        profile = _profile()
        self.assertEqual(profile.profile_identity, "unmeasured-test")
        self.assertEqual((profile.optimal_parallelism, profile.buffer_capacity), (1, 1))

    def test_flat_image_keeps_scheduler_harness(self):
        dockerfile = Path("tools/compatibility/Dockerfile.adapter").read_text()
        ignore = Path("tools/compatibility/Dockerfile.adapter.dockerignore").read_text()
        self.assertIn("scheduler_adapter_check.py", dockerfile)
        self.assertIn("scheduler_adapter_check.py", ignore)


if __name__ == "__main__":
    unittest.main()

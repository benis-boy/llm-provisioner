import asyncio
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import (_ResidencyPending, GPUMemoryObservation, ProcessIdentity,
                                        ResidencyEvidence)
from services.llm.resource_manager.protocol import ProviderResponse
from tools.compatibility import three_model_adapter_check as check


class _Proof:
    supervisor_identity = ProcessIdentity(100, 1)
    def __init__(self): self.clean = True
    async def identity(self): return "GPU-test"
    async def cleanup(self): return self.clean
    async def residency(self): return ResidencyEvidence("GPU-test", self.supervisor_identity, (ProcessIdentity(101, 2),))
    async def residency_for_runner(self, expected):
        return ResidencyEvidence("GPU-test", self.supervisor_identity, (expected,))
    async def memory(self):
        return GPUMemoryObservation("GPU-test", self.supervisor_identity, 10, 11,
                                     1000, 400, 600)


class _Provider:
    def __init__(self, model): self.model, self.cleaned, self.load_calls = model, False, 0
    async def validate(self, profile): pass
    async def load(self, profile): self.load_calls += 1
    async def ready(self): pass
    async def validate_input(self, payload, *, context_size, bucket_identity): pass
    async def execute(self, request_id, payload):
        return ProviderResponse(b"hello" if self.model == "SmolLM" else b'{"texts":["ok"]}')
    async def cancel(self, request_id): pass
    async def unload(self): self.cleaned = True
    async def verify_cleanup(self): return self.cleaned


class _OwnedOllama:
    instances = []
    def __init__(self, config, proof):
        self.config, self.gpu_proof, self.closed = config, proof, False
        # Production exposes the ownership probe; this test double does not
        # model daemon membership, so leave the optional probe absent.
        self.ownership_snapshot = lambda: None
        self.__class__.instances.append(self)
    async def start(self): return "0.11.6"
    async def close(self): self.closed = True


class ThreeModelAdapterCheckTests(unittest.TestCase):
    def test_post_load_residency_settles_transient_empty_observation(self):
        class SettlingProof:
            def __init__(self): self.calls = 0
            async def residency(self):
                self.calls += 1
                if self.calls < 3:
                    raise _ResidencyPending("GPU has no resident runner")
                return ResidencyEvidence("GPU-test", ProcessIdentity(100, 1),
                                         (ProcessIdentity(101, 2),))

        proof = SettlingProof()
        now = [0.0]

        async def sleep(duration):
            now[0] += duration

        async def exercise():
            await check._post_load_residency(proof, sleep=sleep, monotonic=lambda: now[0])

        asyncio.run(exercise())
        self.assertEqual(3, proof.calls)
        self.assertEqual(.4, now[0])

    def test_post_load_residency_cancellation_propagates(self):
        class PersistentProof:
            async def residency(self):
                raise _ResidencyPending("GPU has no resident runner")

        started = asyncio.Event()
        release = asyncio.Event()

        async def sleep(duration):
            started.set()
            await release.wait()

        async def exercise():
            task = asyncio.create_task(check._post_load_residency(
                PersistentProof(), sleep=sleep))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(task.done())
            self.assertFalse(release.is_set())

        asyncio.run(exercise())

    def test_gate_settles_transient_owned_runner_before_load(self):
        class SettlingProof:
            cleanup_reason = "owned_runner_present"
            def __init__(self): self.calls = 0
            async def cleanup(self):
                self.calls += 1
                return self.calls == 3

        provider = _Provider("CoEdIT")
        proof = SettlingProof()
        now = [0.0]

        async def sleep(duration):
            now[0] += duration

        async def exercise():
            await check._GateProvider(provider, proof, cleanup_interval=.2,
                                       cleanup_timeout=1.0, sleep=sleep,
                                       monotonic=lambda: now[0]).load(None)

        asyncio.run(exercise())
        self.assertEqual(3, proof.calls)
        self.assertEqual(1, provider.load_calls)

    def test_gate_persistent_cleanup_timeout_does_not_load(self):
        class PersistentProof:
            cleanup_reason = "owned_runner_present"
            async def cleanup(self): return False

        provider = _Provider("CoEdIT")
        now = [0.0]

        async def sleep(duration): now[0] += duration

        async def exercise():
            with self.assertRaisesRegex(RuntimeError, "reason=owned_runner_present"):
                await check._GateProvider(provider, PersistentProof(), cleanup_interval=.2,
                                           cleanup_timeout=.5, sleep=sleep,
                                           monotonic=lambda: now[0]).load(None)

        asyncio.run(exercise())
        self.assertEqual(0, provider.load_calls)

    def test_gate_cleanup_exception_fails_closed_without_load(self):
        class BrokenProof:
            async def cleanup(self): raise RuntimeError("unbounded detail")

        provider = _Provider("CoEdIT")

        async def exercise():
            with self.assertRaisesRegex(RuntimeError, "reason=gpu_cleanup_probe_failed"):
                await check._GateProvider(provider, BrokenProof()).load(None)

        asyncio.run(exercise())
        self.assertEqual(0, provider.load_calls)

    def test_gate_cancellation_during_settling_propagates(self):
        class PersistentProof:
            cleanup_reason = "owned_runner_present"
            async def cleanup(self): return False

        started = asyncio.Event()
        release = asyncio.Event()

        async def sleep(duration):
            started.set()
            await release.wait()

        async def exercise():
            task = asyncio.create_task(check._GateProvider(
                _Provider("CoEdIT"), PersistentProof(), sleep=sleep).load(None))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(task.done())
            self.assertFalse(release.is_set())

        asyncio.run(exercise())

    def test_transition_diagnostics_preserve_control_flow_and_bound_messages(self):
        with self.assertRaises(asyncio.CancelledError):
            raise check._transition_failure(asyncio.CancelledError(), 1, "SmolLM", "load")
        failure = check.ResourceManagerError(check.Failure("model_load_failed",
            "prompt={secret}" * 1000, True))
        wrapped = check._transition_failure(failure, 2, "CoEdIT", "load")
        self.assertEqual(wrapped.failure.code, "model_load_failed")
        self.assertTrue(wrapped.failure.retryable)
        self.assertLessEqual(len(wrapped.failure.message), 220)
        self.assertNotIn("secret", wrapped.failure.message)

        class Hostile:
            def __str__(self):
                raise AssertionError("diagnostic stringification must not run")
        self.assertEqual("<unavailable>", check._safe_message(Hostile()))
        self.assertEqual("<redacted>", check._safe_message("Bearer secret-token"))

    def test_provider_cleanup_verification_obeys_lifecycle_timeout(self):
        class HangingProvider:
            async def verify_cleanup(self):
                await asyncio.Event().wait()
        self.assertFalse(asyncio.run(check._providers_clean([HangingProvider()], .01)))

    def test_temporary_root_deletion_failure_is_bounded(self):
        errors = []
        with patch.object(check.shutil, "rmtree", side_effect=OSError("path leak")):
            check._remove_root(Path("/tmp/owned-root"), errors)
        self.assertEqual(errors, ["temporary_storage_cleanup_failed"])

    def test_profiles_use_provisioned_digest_and_exact_model_file(self):
        document = {"manifest_sha256": "a" * 64, "models": {}}
        for model, files in check.SPECS.items():
            document["models"][model] = {"model_id": model, "files": [
                {"path": name, "size": 1, "sha256": ("b" if name in (check.GGUF, "model.safetensors") else "c") * 64}
                for name in files]}
        profile = check._profile("SmolLM", document, "GPU-test", "ollama:1")
        self.assertEqual(profile.artifact_manifest_hash, "a" * 64)
        self.assertEqual(profile.model_hash, "b" * 64)
        self.assertEqual(profile.profile_identity, "unmeasured-smollm-adapter-check")

    def test_real_rm_roundtrip_and_cancellation_fence(self):
        proof = _Proof()
        document = {"manifest_sha256": "a" * 64, "models": {}}
        for model, files in check.SPECS.items():
            document["models"][model] = {"model_id": model, "files": [
                {"path": name, "size": 1, "sha256": "b" * 64} for name in files]}
        args = SimpleNamespace(models_root=Path("/source"), manifest=Path("/manifest"), target_gpu_uuid="GPU-test", port=12345, host_pid_namespace=True)
        async def exercise():
            with patch.object(check, "_candidate_manifest", return_value=document), patch.object(check, "provision", return_value=document), patch.object(check.LinuxGPUProof, "capture", return_value=proof) as capture, patch.object(check, "OwnedOllama", _OwnedOllama), patch.object(check, "_runtime_identity", return_value="python:ok"), patch.object(check, "_provider", side_effect=lambda model, *rest: _Provider(model)):
                result = await check.run(args)
            self.assertEqual(capture.call_args.args[1], __import__("os").getpid())
            return result
        result = asyncio.run(exercise())
        self.assertEqual(result["model_sequence"], list(check.SEQUENCE))
        self.assertTrue(result["cancel_fenced"])
        self.assertTrue(_OwnedOllama.instances[-1].closed)
        self.assertEqual(result["profile"], "unmeasured")
        self.assertEqual(result["gpu_uuid"], "GPU-test")
        observations = result["memory_observations"]
        self.assertEqual([item["sequence_index"] for item in observations], list(range(6)))
        self.assertEqual([item["label"] for item in observations],
                         ["baseline", *check.SEQUENCE, "final_cleanup"])
        self.assertEqual({item["total_bytes"] for item in observations}, {1000})
        self.assertEqual({"label", "sequence_index", "start_ns", "end_ns",
                           "total_bytes", "used_bytes", "free_bytes"}, set(observations[0]))
        self.assertIsNotNone(_OwnedOllama.instances[-1].gpu_proof.residency_for_runner)

    def test_memory_samples_are_chronological_and_unmeasured_after_lifecycle_boundaries(self):
        """Whole-device points are telemetry, not capacity or peak evidence."""
        document = {"manifest_sha256": "a" * 64, "models": {}}
        for model, files in check.SPECS.items():
            document["models"][model] = {"model_id": model, "files": [
                {"path": name, "size": 1, "sha256": "b" * 64} for name in files]}
        args = SimpleNamespace(models_root=Path("/source"), manifest=Path("/manifest"),
                               target_gpu_uuid="GPU-test", port=12345, host_pid_namespace=True)
        events = []
        class OrderedProof(_Proof):
            async def memory(self):
                events.append("memory")
                return await super().memory()
            async def residency(self):
                events.append("residency")
                return await super().residency()
            async def cleanup(self):
                clean = await super().cleanup()
                events.append(("gpu_cleanup", clean))
                return clean
        class OrderedDaemon(_OwnedOllama):
            def __init__(self, config, proof):
                events.append("daemon_construct")
                super().__init__(config, proof)
            async def start(self):
                events.append("daemon_start")
                return await super().start()
            async def close(self):
                events.append("daemon_close")
                return await super().close()
        class OrderedProvider(_Provider):
            instances = []
            def __init__(self, model, proof):
                super().__init__(model)
                self.gpu_proof = proof
                self.__class__.instances.append(self)
            async def execute(self, request_id, payload):
                events.append(("inference", self.model))
                return await super().execute(request_id, payload)
            async def unload(self):
                events.append("provider_unload")
                return await super().unload()
        proof = OrderedProof()
        owned_root = None
        async def exercise():
            nonlocal owned_root
            with tempfile.TemporaryDirectory() as temporary:
                owned_root = Path(temporary) / "owned-check-root"
                owned_root.mkdir()
                with patch.object(check, "_candidate_manifest", return_value=document), \
                     patch.object(check, "provision", return_value=document), \
                     patch.object(check.LinuxGPUProof, "capture", return_value=proof), \
                     patch.object(check, "OwnedOllama", OrderedDaemon), \
                     patch.object(check, "_runtime_identity", return_value="python:ok"), \
                     patch.object(check, "_provider", side_effect=lambda model, *rest: OrderedProvider(model, rest[3])), \
                     patch.object(check.tempfile, "mkdtemp", return_value=str(owned_root)):
                    return await check.run(args)
        result = asyncio.run(exercise())
        observations = result["memory_observations"]
        self.assertEqual(6, len(observations))
        self.assertEqual(["baseline", *check.SEQUENCE, "final_cleanup"],
                         [item["label"] for item in observations])
        self.assertTrue(all(set(item) == {"label", "sequence_index", "start_ns", "end_ns",
                                         "total_bytes", "used_bytes", "free_bytes"}
                            for item in observations))
        self.assertEqual("unmeasured", result["profile"])
        self.assertLess(events.index("memory"), events.index("daemon_construct"))
        self.assertLess(events.index("daemon_construct"), events.index("daemon_start"))
        self.assertEqual(4, events.count("residency"))
        self.assertEqual(6, events.count("memory"))
        memory_events = [index for index, event in enumerate(events) if event == "memory"]
        residency_events = [index for index, event in enumerate(events) if event == "residency"]
        self.assertTrue(all(residency < memory for residency, memory
                            in zip(residency_events, memory_events[1:5])))
        final_memory = [index for index, event in enumerate(events) if event == "memory"][-1]
        for model, residency, sample in zip(check.SEQUENCE, residency_events, memory_events[1:5]):
            with self.subTest(model=model):
                inference = events.index(("inference", model))
                self.assertLess(inference, residency)
                self.assertLess(residency, sample)
        self.assertLess(events.index("provider_unload"), events.index("daemon_close"))
        daemon_close = events.index("daemon_close")
        final_cleanup = next(index for index, event in enumerate(events)
                             if daemon_close < index < final_memory and event == ("gpu_cleanup", True))
        self.assertLess(daemon_close, final_cleanup)
        self.assertLess(final_cleanup, final_memory)
        daemon_proof = OrderedDaemon.instances[-1].gpu_proof
        self.assertIsInstance(daemon_proof, GPUProof)
        self.assertTrue(all(provider.gpu_proof is daemon_proof
                            for provider in OrderedProvider.instances))
        self.assertFalse(owned_root.exists())

    def test_cleanup_failure_prevents_next_load_and_stale_cancel_is_fenced(self):
        from services.llm.resource_manager.core import ResourceManager, ResourceManagerError
        async def exercise():
            class CleanupFailure(_Provider):
                async def verify_cleanup(self): return False
            rm, provider = ResourceManager(), CleanupFailure("SmolLM")
            profile = check._profile("SmolLM", {"manifest_sha256": "a" * 64, "models": {"SmolLM": {"files": [{"path": check.GGUF, "sha256": "b" * 64}]}}}, "GPU-test", "x")
            session = await rm.start_session("s", check.ModelId.SMOLLM, profile, provider, idempotency_key="one")
            with self.assertRaises(ResourceManagerError):
                await rm.start_session("s", check.ModelId.SMOLLM, profile, _Provider("SmolLM"), idempotency_key="two")
            with self.assertRaises(ResourceManagerError) as raised:
                await rm.cancel_request(session.session_token, "old", idempotency_key="old")
            self.assertEqual(raised.exception.failure.code, "scheduler_superseded")
        asyncio.run(exercise())

    def test_config_construction_and_flat_image_import_are_explicit(self):
        proof = _Proof()
        document = {"manifest_sha256": "a" * 64, "models": {}}
        for model, files in check.SPECS.items():
            document["models"][model] = {"model_id": model, "files": [{"path": x, "size": 1, "sha256": "b" * 64} for x in files]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for model in check.SPECS:
                with patch.object(check, {"SmolLM": "SmolLMProvider", "CoEdIT": "CoEdITProvider", "GECToR": "GECToRProvider"}[model]):
                    check._provider(model, root, document, "GPU-test", proof, "runtime", 12345, root / "daemon")
        source = Path(check.__file__).read_text()
        self.assertIn("import adapter_check", source)
        self.assertNotIn("import torch", source)
        self.assertEqual(inspect.signature(check.adapter_check._start_server).parameters.keys(), {"port", "model_store"})

    def test_failed_initial_load_deletes_owned_temp_after_positive_cleanup(self):
        proof = _Proof()
        document = {"manifest_sha256": "a" * 64, "models": {}}
        for model, files in check.SPECS.items():
            document["models"][model] = {"model_id": model, "files": [
                {"path": name, "size": 1, "sha256": "b" * 64} for name in files]}
        args = SimpleNamespace(models_root=Path("/source"), manifest=Path("/manifest"), target_gpu_uuid="GPU-test", port=12345, host_pid_namespace=True)
        class Failing(_Provider):
            async def load(self, profile): raise RuntimeError("load failed")
        async def exercise():
            with patch.object(check, "_candidate_manifest", return_value=document), patch.object(check, "provision", return_value=document), patch.object(check.LinuxGPUProof, "capture", return_value=proof), patch.object(check, "OwnedOllama", _OwnedOllama), patch.object(check, "_runtime_identity", return_value="python:ok"), patch.object(check, "_provider", side_effect=lambda model, *rest: Failing(model)):
                with self.assertRaises(Exception) as raised:
                    await check.run(args)
            self.assertEqual(raised.exception.failure.code, "model_load_failed")
        asyncio.run(exercise())

    def test_memory_harness_failures_fail_closed_and_clean_owned_resources(self):
        """Harness telemetry is a gate, not a best-effort diagnostic.

        This deliberately uses only synthetic proof/provider objects.  The
        temporary-root patch also makes the ownership assertion independent of
        the host filesystem and proves the daemon and providers received one
        shared, typed proof object.
        """
        document = {"manifest_sha256": "a" * 64, "models": {}}
        for model, files in check.SPECS.items():
            document["models"][model] = {"model_id": model, "files": [
                {"path": name, "size": 1, "sha256": "b" * 64} for name in files]}
        args = SimpleNamespace(models_root=Path("/source"), manifest=Path("/manifest"),
                               target_gpu_uuid="GPU-test", port=12345,
                               host_pid_namespace=True)

        class HarnessProof(_Proof):
            def __init__(self, *, failure_call=None, changed_call=None,
                         changed_field=None, wrong_baseline=False):
                super().__init__()
                self.memory_calls = 0
                self.cleanup_calls = 0
                self.failure_call = failure_call
                self.changed_call = changed_call
                self.changed_field = changed_field
                self.wrong_baseline = wrong_baseline

            async def memory(self):
                self.memory_calls += 1
                if self.memory_calls == self.failure_call:
                    raise RuntimeError("synthetic memory read failure")
                uuid = "GPU-wrong" if self.wrong_baseline else "GPU-test"
                total = 2000 if self.memory_calls == self.changed_call and self.changed_field == "total" else 1000
                if self.memory_calls == self.changed_call and self.changed_field == "uuid":
                    uuid = "GPU-other"
                return GPUMemoryObservation(uuid, self.supervisor_identity,
                                            self.memory_calls, self.memory_calls + 1,
                                            total, 400, total - 400)

            async def cleanup(self):
                self.cleanup_calls += 1
                return True

        class HarnessProvider(_Provider):
            instances = []
            def __init__(self, model, proof=None):
                super().__init__(model)
                self.gpu_proof = proof
                self.loaded = False
                self.unloads = 0
                self.__class__.instances.append(self)

            async def load(self, profile):
                self.loaded = True

            async def unload(self):
                if self.loaded:
                    self.unloads += 1
                self.cleaned = True

            async def verify_cleanup(self):
                # A provider which never loaded has no resource cleanup to
                # exercise; this avoids asserting cleanup for an unbegun load.
                return (not self.loaded) or (self.unloads == 1)

        class TypedOwnedOllama(_OwnedOllama):
            instances = []
            def __init__(self, config, proof):
                super().__init__(config, proof)
                self.assertIsTyped = isinstance(proof, GPUProof)

        scenarios = (
            ({"failure_call": 1}, "synthetic memory read failure"),
            ({"failure_call": 2},
             "transition=0 model=SmolLM stage=memory: exception=RuntimeError; message=<unavailable>"),
            ({"failure_call": 6}, "synthetic memory read failure"),
            ({"changed_call": 2, "changed_field": "total"},
             "transition=0 model=SmolLM stage=memory: exception=RuntimeError; message=<unavailable>"),
            ({"changed_call": 2, "changed_field": "uuid"},
             "transition=0 model=SmolLM stage=memory: exception=RuntimeError; message=<unavailable>"),
            ({"changed_call": 6, "changed_field": "total"}, "GPU memory identity or total capacity changed"),
            ({"changed_call": 6, "changed_field": "uuid"}, "GPU memory UUID changed or mismatched"),
            ({"wrong_baseline": True}, "GPU memory UUID changed or mismatched"),
        )
        for scenario, message in scenarios:
            with self.subTest(scenario=scenario):
                HarnessProvider.instances = []
                TypedOwnedOllama.instances = []
                proof = HarnessProof(**scenario)
                with tempfile.TemporaryDirectory() as owner:
                    owned_root = Path(owner) / "owned-check-root"
                    owned_root.mkdir()

                    async def exercise():
                        with patch.object(check, "_candidate_manifest", return_value=document), \
                             patch.object(check, "provision", return_value=document), \
                             patch.object(check.LinuxGPUProof, "capture", return_value=proof), \
                             patch.object(check, "OwnedOllama", TypedOwnedOllama), \
                             patch.object(check, "_runtime_identity", return_value="python:ok"), \
                             patch.object(check, "_provider",
                                          side_effect=lambda model, *rest: HarnessProvider(model, rest[3])), \
                             patch.object(check.tempfile, "mkdtemp", return_value=str(owned_root)):
                            with self.assertRaisesRegex(RuntimeError, message) as raised:
                                await check.run(args)

                    asyncio.run(exercise())
                    self.assertFalse(owned_root.exists())
                    self.assertTrue(proof.cleanup_calls >= 1)
                    if scenario.get("failure_call", 0) >= 2 or scenario.get("changed_call", 0) >= 2:
                        self.assertTrue(TypedOwnedOllama.instances[0].closed)
                    for provider in HarnessProvider.instances:
                        self.assertTrue(provider.cleaned)
                        if provider.loaded:
                            self.assertEqual(1, provider.unloads)
                    if TypedOwnedOllama.instances:
                        daemon_proof = TypedOwnedOllama.instances[0].gpu_proof
                        self.assertIsInstance(daemon_proof, GPUProof)
                        self.assertTrue(all(provider.gpu_proof is daemon_proof
                                            for provider in HarnessProvider.instances))

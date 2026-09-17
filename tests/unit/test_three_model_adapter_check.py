import asyncio
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from services.llm.providers.gpu import ProcessIdentity, ResidencyEvidence
from services.llm.resource_manager.protocol import ProviderResponse
from tools.compatibility import three_model_adapter_check as check


class _Proof:
    supervisor_identity = ProcessIdentity(100, 1)
    def __init__(self): self.clean = True
    async def identity(self): return "GPU-test"
    async def cleanup(self): return self.clean
    async def residency(self): return ResidencyEvidence("GPU-test", self.supervisor_identity, (ProcessIdentity(101, 2),))


class _Provider:
    def __init__(self, model): self.model, self.cleaned = model, False
    async def validate(self, profile): pass
    async def load(self, profile): pass
    async def ready(self): pass
    async def validate_input(self, payload, *, context_size, bucket_identity): pass
    async def execute(self, request_id, payload):
        return ProviderResponse(b"hello" if self.model == "SmolLM" else b'{"texts":["ok"]}')
    async def cancel(self, request_id): pass
    async def unload(self): self.cleaned = True
    async def verify_cleanup(self): return self.cleaned


class ThreeModelAdapterCheckTests(unittest.TestCase):
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
            with patch.object(check, "_candidate_manifest", return_value=document), patch.object(check, "provision", return_value=document), patch.object(check.LinuxGPUProof, "capture", return_value=proof) as capture, patch.object(check.adapter_check, "_start_server", return_value=(object(), 1, ())), patch.object(check.adapter_check, "_health", return_value="0.11.6"), patch.object(check.adapter_check, "_cleanup_group", return_value=True), patch.object(check, "_runtime_identity", return_value="python:ok"), patch.object(check, "_provider", side_effect=lambda model, *rest: _Provider(model)):
                result = await check.run(args)
            self.assertEqual(capture.call_args.args[1], __import__("os").getpid())
            return result
        result = asyncio.run(exercise())
        self.assertEqual(result["model_sequence"], list(check.SEQUENCE))
        self.assertTrue(result["cancel_fenced"])

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
            with patch.object(check, "_candidate_manifest", return_value=document), patch.object(check, "provision", return_value=document), patch.object(check.LinuxGPUProof, "capture", return_value=proof), patch.object(check.adapter_check, "_start_server", return_value=(object(), 1, ())), patch.object(check.adapter_check, "_health", return_value="0.11.6"), patch.object(check.adapter_check, "_cleanup_group", return_value=True), patch.object(check, "_runtime_identity", return_value="python:ok"), patch.object(check, "_provider", side_effect=lambda model, *rest: Failing(model)):
                with self.assertRaises(Exception) as raised:
                    await check.run(args)
            self.assertEqual(raised.exception.failure.code, "model_load_failed")
        asyncio.run(exercise())

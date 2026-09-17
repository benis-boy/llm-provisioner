import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.llm.provisioning.artifacts import SPECS, inventory, manifest, spec
from services.llm.provisioning.volume import provision
from tools.compatibility import gector_adapter_check as check
from services.llm.providers.gpu import ProcessIdentity


def _source(root):
    source = root / "GECToR"; source.mkdir()
    for name in SPECS["GECToR"]: (source / name).write_bytes(name.encode())
    return source


class GECToRAdapterCheckTests(unittest.TestCase):
    def test_fixed_unmeasured_profile_and_runtime_identity(self):
        with patch.object(check, "_runtime_identity", return_value="candidate:test"):
            profile = check._profile("a" * 64, "b" * 64, "GPU-test")
        self.assertEqual(profile.profile_identity, "unmeasured-gector-adapter-check")
        self.assertEqual(profile.bucket_identity, check.BUCKET)

    def test_manifest_identity_is_exact_selected_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = _source(root)
            candidate = root / "manifest.json"; candidate.write_text(json.dumps(manifest({"GECToR": inventory(spec("GECToR", source))})))
            selected, digest = check._candidate_manifest(candidate, source)
            volume = provision({"GECToR": source}, root / "volume")
            self.assertEqual(check._selected(selected["models"]["GECToR"]), check._selected(volume["models"]["GECToR"]))
            self.assertEqual(digest, next(x["sha256"] for x in volume["models"]["GECToR"]["files"] if x["path"] == "model.safetensors"))

    def test_attestation_is_required_before_artifact_work(self):
        args = SimpleNamespace(models_root=Path("/none"), manifest=Path("/none"), target_gpu_uuid="GPU-test", host_pid_namespace=False, inject_process_failure=False)
        with self.assertRaisesRegex(ValueError, "host-pid-namespace"): asyncio.run(check.run(args))

    def test_schema_only_response_validation(self):
        self.assertTrue(check._valid_response(SimpleNamespace(result=b'{"texts":["ok"]}')))
        self.assertFalse(check._valid_response(SimpleNamespace(result=b'{"texts":[]}')))

    def test_real_rm_fake_provider_normal_and_process_loss_cleanup(self):
        seen = {}
        class Proof:
            supervisor_identity = ProcessIdentity(7, 1); identity = lambda self: "GPU-test"; cleanup = lambda self: True
            async def residency(self): return SimpleNamespace(runners=(ProcessIdentity(9, 1),))
        class Provider:
            def __init__(self, config): seen["config"] = config
            async def validate(self, profile): pass
            async def load(self, profile): self.worker = SimpleNamespace(child_identity=ProcessIdentity(9, 1), killed=False, _signal_owned=lambda sig: setattr(self.worker, "killed", True))
            async def ready(self): pass
            async def validate_input(self, *a, **k):
                if b"word word word word word" in a[0] and len(a[0]) > 700: raise ValueError("overlong")
            async def execute(self, *a):
                if self.worker.killed: raise RuntimeError("owned worker lost")
                from services.llm.resource_manager.protocol import ProviderResponse
                return ProviderResponse(b'{"texts":["ok"]}')
            async def cancel(self, request_id): pass
            async def unload(self): self.worker = None
            async def verify_cleanup(self): return self.worker is None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = _source(root); candidate = root / "candidate.json"; candidate.write_text(json.dumps(manifest({"GECToR": inventory(spec("GECToR", source))})))
            with patch.object(check.LinuxGPUProof, "capture", return_value=Proof()), patch.object(check, "GECToRProvider", Provider), patch.object(check, "_runtime_identity", return_value="candidate:test"), patch.object(check, "_runtime_versions", return_value={"gector":"1.2.0", "torch":"2.7.1+cu128", "transformers":"4.49.0", "tokenizers":"0.21.0", "safetensors":"0.5.3", "cuda":"12.8"}):
                for injected in (False, True):
                    result = asyncio.run(check.run(SimpleNamespace(models_root=root, manifest=candidate, target_gpu_uuid="GPU-test", host_pid_namespace=True, inject_process_failure=injected)))
                    self.assertTrue(result["cleanup"])
                    self.assertEqual(result["profile"], "unmeasured")

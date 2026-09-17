import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.llm.provisioning.artifacts import SPECS, inventory, manifest, spec
from services.llm.provisioning.volume import provision
from tools.compatibility import coedit_adapter_check as check
from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import ProcessIdentity


def _source(root):
    source = root / "CoEdIT"
    source.mkdir()
    for name in SPECS["CoEdIT"]:
        (source / name).write_bytes(name.encode())
    return source


class CoEdITAdapterCheckTests(unittest.TestCase):
    def test_profile_is_unmeasured_single_p1_bucket(self):
        # The base test environment intentionally has no adapter runtime
        # distributions.  Unit-test the profile contract with an explicit
        # stable identity; the real harness remains fail-closed and obtains
        # installed runtime metadata in the candidate container.
        with patch.object(check, "_runtime_identity", return_value="candidate:test-runtime"):
            profile = check._profile("a" * 64, "b" * 64, "GPU-test")
        self.assertEqual(profile.profile_identity, "unmeasured-coedit-adapter-check")
        self.assertEqual((profile.optimal_parallelism, profile.memory_safe_n, profile.buffer_capacity), (1, 1, 1))
        self.assertEqual(profile.bucket_identity, check.BUCKET)

    def test_candidate_manifest_selects_exact_coedit_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _source(root)
            source_manifest = manifest({"CoEdIT": inventory(spec("CoEdIT", source))})
            volume = provision({"CoEdIT": source}, root / "volume")
            candidate = root / "manifest.json"
            candidate.write_text(json.dumps(source_manifest), encoding="utf-8")
            selected, digest = check._candidate_manifest(candidate, source)
            self.assertEqual(set(selected["models"]), {"CoEdIT"})
            self.assertEqual(digest, volume["models"]["CoEdIT"]["files"][0]["sha256"])
            self.assertNotEqual(selected, volume)
            self.assertEqual(check._selected_file_identity(selected["models"]["CoEdIT"]),
                             check._selected_file_identity(volume["models"]["CoEdIT"]))

    def test_attestation_and_uuid_fail_before_gpu_or_volume_side_effect(self):
        args = SimpleNamespace(models_root=Path("/does/not/exist"), manifest=Path("/none"),
                               target_gpu_uuid="GPU-test", host_pid_namespace=False,
                               inject_process_failure=False)
        with self.assertRaisesRegex(ValueError, "host-pid-namespace"):
            asyncio.run(check.run(args))

    def test_response_validation_is_aligned_and_nonempty_without_dumping_text(self):
        event = SimpleNamespace(result=b'{"texts":["ok"]}')
        self.assertTrue(check._check_response(event))
        self.assertFalse(check._check_response(SimpleNamespace(result=b'{"texts":[]}')))

    def test_run_uses_rm_lifecycle_for_normal_and_process_loss_paths(self):
        captured = {}
        class Proof:
            supervisor_identity = ProcessIdentity(7, 1)
            identity = lambda self: "GPU-test"
            cleanup = lambda self: True
            async def residency(self):
                return SimpleNamespace(runners=(ProcessIdentity(9, 1),))
        class Provider:
            def __init__(self, config): captured["config"] = config
            async def validate(self, profile): pass
            async def load(self, profile):
                self.worker = SimpleNamespace(child_identity=ProcessIdentity(9, 1), killed=False,
                    _signal_owned=lambda signal: setattr(self.worker, "killed", True))
            async def ready(self): pass
            async def validate_input(self, *args, **kwargs): pass
            async def execute(self, *args):
                if self.worker.killed:
                    raise RuntimeError("owned worker lost")
                from services.llm.resource_manager.protocol import ProviderResponse
                return ProviderResponse(b'{"texts":["ok"]}', None, False)
            async def cancel(self, request_id): pass
            async def unload(self): self.worker = None
            async def verify_cleanup(self): return self.worker is None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = _source(root)
            candidate = root / "candidate.json"
            candidate.write_text(json.dumps(manifest({"CoEdIT": inventory(spec("CoEdIT", source))})))
            with patch.object(check.LinuxGPUProof, "capture", return_value=Proof()) as capture, \
                  patch.object(check, "CoEdITProvider", Provider), \
                  patch.object(check, "_runtime_identity", return_value="candidate:test-runtime"), \
                  patch.object(check, "_runtime_versions", return_value={"torch":"2.7.1+cu128", "transformers":"4.49.0", "tokenizers":"0.21.0", "safetensors":"0.5.3", "cuda":"12.8"}):
                for injected in (False, True):
                    args = SimpleNamespace(models_root=root, manifest=candidate, target_gpu_uuid="GPU-test",
                                           host_pid_namespace=True, inject_process_failure=injected)
                    result = asyncio.run(check.run(args))
                    self.assertTrue(result["cleanup"])
                    self.assertEqual(result["profile"], "unmeasured")
                    self.assertIn("manifest_sha256", result)
            self.assertEqual(capture.call_count, 2)
            self.assertEqual(capture.call_args.args[1], __import__("os").getpid())

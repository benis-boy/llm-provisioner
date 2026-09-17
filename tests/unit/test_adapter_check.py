"""Behavioral contract tests for the candidate-only adapter harness."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.llm.provisioning.artifacts import SPECS
from services.llm.provisioning.volume import provision
from services.llm.providers.config import SmolLMProviderConfig
from services.llm.queue.contracts import ModelId
from tools.compatibility import adapter_check


def _smollm_root(root: Path) -> Path:
    root = root / "SmolLM"
    root.mkdir()
    for name in SPECS["SmolLM"]:
        (root / name).write_bytes(name.encode())
    return root


class AdapterCheckTests(unittest.TestCase):
    def test_selected_volume_identity_configures_provider(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _smollm_root(root)
            destination = root / "volume"
            document = provision({"SmolLM": source}, destination)
            gguf = next(item for item in document["models"]["SmolLM"]["files"] if item["path"] == adapter_check.GGUF)
            config = SmolLMProviderConfig(destination, document["manifest_sha256"], gguf["sha256"], "GPU-test", "candidate-adapter-check", "candidate-smollm-provider", allowed_context_sizes=(512,), ollama_port=65535)
            self.assertEqual(config.artifact_root, destination.resolve())
            self.assertEqual(ModelId.SMOLLM.value, "SmolLM")

    def test_port_and_run_arguments_fail_before_side_effects(self):
        with self.assertRaises(ValueError): adapter_check._validate_port(0)
        with self.assertRaises(ValueError): adapter_check._validate_port(65536)
        args = type("Args", (), {"port": 0, "host_pid_namespace": True, "target_gpu_uuid": "GPU-test"})()
        with self.assertRaises(ValueError): asyncio.run(adapter_check.run(args))

    def test_profile_is_explicitly_unmeasured_and_single_capacity(self):
        profile = adapter_check._profile("a" * 64, "b" * 64, "GPU-test")
        self.assertEqual(profile.profile_identity, "unmeasured-adapter-check")
        self.assertEqual((profile.optimal_parallelism, profile.memory_safe_n, profile.buffer_capacity), (1, 1, 1))
        self.assertEqual(profile.context_size, 512)

    def test_candidate_manifest_requires_exact_selected_smollm_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _smollm_root(root)
            document = provision({"SmolLM": source}, root / "volume")
            candidate = root / "manifest.json"
            candidate.write_text(json.dumps(document), encoding="utf-8")
            _, digest = adapter_check._candidate_manifest(candidate, source)
            self.assertEqual(digest, adapter_check._digest(source / adapter_check.GGUF))
            document["models"]["SmolLM"]["files"] = []
            candidate.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises((RuntimeError, ValueError)):
                adapter_check._candidate_manifest(candidate, source)

    def test_malformed_handshake_stat_is_rejected(self):
        for raw in (b"", b"1 bad", b"1 (x) S 0"):
            with self.assertRaises(RuntimeError):
                adapter_check._stat_identity(raw)

    def test_start_server_reports_bounded_launcher_error_and_cleans_up(self):
        launcher = 'import sys; sys.stdout.write("HANDSHAKE_ERROR procfs_missing\\n"); sys.stdout.flush()'
        with tempfile.TemporaryDirectory() as temporary, patch.object(adapter_check, "LAUNCHER", launcher):
            with self.assertRaisesRegex(RuntimeError, "procfs_missing"):
                asyncio.run(adapter_check._start_server(12345, Path(temporary)))

    def test_start_server_cancellation_waits_for_cleanup(self):
        launcher = 'import time; time.sleep(30)'
        async def exercise():
            with tempfile.TemporaryDirectory() as temporary, patch.object(adapter_check, "LAUNCHER", launcher):
                task = asyncio.create_task(adapter_check._start_server(12345, Path(temporary)))
                await asyncio.sleep(.05)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        asyncio.run(exercise())

    def test_run_uses_actual_resource_manager_lifecycle(self):
        class Proof:
            identity = lambda self: "GPU-test"
            cleanup = lambda self: True
            async def residency(self):
                from services.llm.providers.gpu import ProcessIdentity, ResidencyEvidence
                return ResidencyEvidence("GPU-test", ProcessIdentity(7, 1), (ProcessIdentity(8, 1),))
            supervisor_identity = __import__("services.llm.providers.gpu", fromlist=["ProcessIdentity"]).ProcessIdentity(7, 1)
        class Provider:
            def __init__(self, config): self.config = config
            async def validate(self, profile): return None
            async def load(self, profile): return None
            async def ready(self): return None
            async def validate_input(self, payload, *, context_size, bucket_identity): return None
            async def execute(self, request_id, payload):
                from services.llm.resource_manager.protocol import ProviderResponse
                return ProviderResponse(b"ok", None, False)
            async def cancel(self, request_id): return None
            async def unload(self): return None
            async def verify_cleanup(self): return True
        async def exercise(root):
            source = _smollm_root(root)
            original = provision({"SmolLM": source}, root / "candidate-volume")
            manifest_path = root / "candidate.json"
            manifest_path.write_text(json.dumps(original), encoding="utf-8")
            args = SimpleNamespace(models_root=root, manifest=manifest_path, target_gpu_uuid="GPU-test", port=12345, host_pid_namespace=True)
            daemon = SimpleNamespace(pid=123, returncode=0)
            with patch.object(adapter_check, "_start_server", return_value=(daemon, 7, ())), patch.object(adapter_check, "_health", return_value=adapter_check.OLLAMA_VERSION), patch.object(adapter_check.LinuxGPUProof, "capture", return_value=Proof()), patch.object(adapter_check, "SmolLMProvider", Provider), patch.object(adapter_check, "_cleanup_group", return_value=True):
                result = await adapter_check.run(args)
            self.assertEqual(result["status"], "passed-candidate")
            self.assertTrue(result["small_completion"])
        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(exercise(Path(temporary)))

    def test_group_cleanup_handles_leader_exit_with_grandchild(self):
        async def exercise():
            proc = await asyncio.create_subprocess_exec(
                __import__("sys").executable, "-c", "import os,time; os.fork() or time.sleep(30)",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
            drains = (asyncio.create_task(adapter_check._drain(proc.stdout)), asyncio.create_task(adapter_check._drain(proc.stderr)))
            self.assertTrue(await adapter_check._stop_group(proc, drains))
        asyncio.run(exercise())

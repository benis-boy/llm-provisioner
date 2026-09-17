"""Behavioral contract tests for the candidate-only adapter harness."""
import asyncio
import inspect
import json
import stat
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

    def test_host_pid_namespace_attestation_remains_mandatory(self):
        args = SimpleNamespace(port=12345, host_pid_namespace=False,
                               target_gpu_uuid="GPU-test")
        with self.assertRaisesRegex(ValueError, "host-pid-namespace"):
            asyncio.run(adapter_check.run(args))

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

    def test_start_server_supplies_private_home_when_caller_home_is_absent(self):
        captured = {}

        async def launch(*args, **kwargs):
            captured.update(kwargs)
            raise OSError("test launcher")

        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(adapter_check.os.environ, {
                    "HOME": "/caller-home",
                    "UNRELATED_TEST_ENV": "must-not-be-forwarded",
                }, clear=True), \
                patch.object(adapter_check, "_enable_subreaper"), \
                patch.object(adapter_check.asyncio, "create_subprocess_exec", side_effect=launch):
            with self.assertRaisesRegex(OSError, "test launcher"):
                asyncio.run(adapter_check._start_server(12345, Path(temporary) / "models"))

            home = Path(captured["env"]["HOME"])
            self.assertTrue(home.is_absolute())
            self.assertNotEqual(home, Path("/caller-home"))
            self.assertEqual(home.parent, (Path(temporary) / "models").resolve())
            self.assertTrue(home.is_dir())
            self.assertTrue(home.stat().st_mode & stat.S_IWUSR)
            self.assertEqual(stat.S_IMODE(home.stat().st_mode), 0o700)
            self.assertNotIn("UNRELATED_TEST_ENV", captured["env"])

    def test_unpatched_launcher_reaches_handshake_before_exec_failure(self):
        async def exercise():
            with tempfile.TemporaryDirectory() as temporary:
                proc = await asyncio.create_subprocess_exec(
                    __import__("sys").executable, "-c", adapter_check.LAUNCHER,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    start_new_session=True)
                assert proc.stdout is not None
                line = await asyncio.wait_for(proc.stdout.readline(), 2)
                self.assertRegex(line.decode(), r"^HANDSHAKE \d+ \d+\n$")
                self.assertEqual(await asyncio.wait_for(proc.wait(), 2), 1)
        asyncio.run(exercise())

    def test_launcher_and_capture_use_authoritative_proc_root(self):
        self.assertEqual(adapter_check.PROC_ROOT, Path("/proc"))
        self.assertIn('open("/proc/self/stat"', adapter_check.LAUNCHER)
        self.assertNotIn("/host/proc", inspect.getsource(adapter_check))

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
        provider_configs = []
        class Proof:
            identity = lambda self: "GPU-test"
            cleanup = lambda self: True
            async def residency(self):
                from services.llm.providers.gpu import ProcessIdentity, ResidencyEvidence
                return ResidencyEvidence("GPU-test", ProcessIdentity(7, 1), (ProcessIdentity(8, 1),))
            supervisor_identity = __import__("services.llm.providers.gpu", fromlist=["ProcessIdentity"]).ProcessIdentity(7, 1)
        class Provider:
            def __init__(self, config):
                self.config = config
                provider_configs.append(config)
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
            with patch.object(adapter_check, "_start_server", return_value=(daemon, 7, ())) as start_server, patch.object(adapter_check, "_health", return_value=adapter_check.OLLAMA_VERSION), patch.object(adapter_check.LinuxGPUProof, "capture", return_value=Proof()) as capture, patch.object(adapter_check, "SmolLMProvider", Provider), patch.object(adapter_check, "_cleanup_group", return_value=True):
                result = await adapter_check.run(args)
            self.assertEqual(capture.call_args.args[2], adapter_check.PROC_ROOT)
            self.assertTrue(capture.call_args.kwargs["host_pid_namespace"])
            self.assertEqual(result["status"], "passed-candidate")
            self.assertTrue(result["small_completion"])
            self.assertTrue(result["cleanup_model_absent"])
            self.assertIn("cleanup_baseline", result)
            self.assertNotIn("cleanup_owned_residency", result)
            self.assertIn("foreign_baseline_process_count", result)
            daemon_store = start_server.call_args.args[1]
            self.assertEqual(provider_configs[0].ollama_home,
                             adapter_check._runtime_home(daemon_store))
        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(exercise(Path(temporary)))

    def test_group_cleanup_terminates_owned_group(self):
        async def exercise():
            proc = await asyncio.create_subprocess_exec(
                __import__("sys").executable, "-c", "import os,time; os.fork() or time.sleep(30)",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
            drains = (asyncio.create_task(adapter_check._drain(proc.stdout)), asyncio.create_task(adapter_check._drain(proc.stderr)))
            self.assertTrue(await adapter_check._stop_group(proc, drains))
        asyncio.run(exercise())

    def test_group_cleanup_does_not_prove_absence_from_leader_exit(self):
        async def exercise():
            proc = await asyncio.create_subprocess_exec(
                __import__("sys").executable, "-c",
                "import os,time; child=os.fork(); "
                "(os.close(1), os.close(2), time.sleep(30)) if child == 0 else None",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=True)
            raw = (adapter_check.PROC_ROOT / str(proc.pid) / "stat").read_bytes()
            opening, closing = raw.find(b"("), raw.rfind(b")")
            fields = raw[closing + 2:].split()
            start = int(fields[19])
            proc._adapter_group_fence = (proc.pid, start, proc.pid, proc.pid)
            drains = (asyncio.create_task(adapter_check._drain(proc.stdout)),
                      asyncio.create_task(adapter_check._drain(proc.stderr)))
            await proc.wait()
            try:
                self.assertFalse(await adapter_check._stop_group(proc, drains))
            finally:
                # The saved group is intentionally still alive for this
                # assertion; clean up only this test's known owned group.
                try:
                    adapter_check.os.killpg(proc.pid, adapter_check.signal.SIGKILL)
                except ProcessLookupError:
                    pass
        asyncio.run(exercise())

    def test_group_cleanup_never_signals_reused_leader_group(self):
        async def exercise():
            proc = SimpleNamespace(pid=41, returncode=None,
                                   _adapter_group_fence=(41, 1, 41, 41))
            drains = ()
            stat = b"99 (ollama) S 1 41 41 " + b"0 " * 18 + b"8 0\n"
            with patch("pathlib.Path.read_bytes", return_value=stat), \
                 patch.object(adapter_check.os, "killpg") as killpg:
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    await adapter_check._stop_group(proc, drains)
            killpg.assert_not_called()
        asyncio.run(exercise())

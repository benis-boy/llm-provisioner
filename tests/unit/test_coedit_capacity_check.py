import unittest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch
import importlib.util
import tempfile
import sys
import hashlib

from tools.compatibility import coedit_capacity_check
from services.llm.provisioning.capacity import DiscoveryResult
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager, ResourceManagerError
from services.llm.resource_manager.protocol import ProviderResponse


class CoeditCapacityCommandTests(unittest.TestCase):
    def test_without_child_tokenizer_fails_closed(self):
        with patch("sys.argv", ["coedit_capacity_check", "--models-root", "/missing",
                                  "--manifest", "/missing", "--target-gpu-uuid", "bad",
                                  "--native-batch-size", "2", "--configured-fixture"]):
            self.assertEqual(coedit_capacity_check.main(), 2)

    def test_input_rejects_duplicate_keys_and_wrong_shape(self):
        with self.assertRaises(ValueError):
            coedit_capacity_check._input(Path("/does/not/exist"), True)
        self.assertEqual(coedit_capacity_check._input(None, True),
                         ("Improve the grammar.", "word " * 100))

    def test_raw_candidate_never_claims_measured_or_eligible(self):
        sample = coedit_capacity_check.MemorySample(1, 100, 0, 100)
        allocator = coedit_capacity_check.AllocatorObservation(10, 20, 30, 40, 10, 20)
        wave = coedit_capacity_check.Wave(1, 1, ("id",), 1, True, 1, 1, 2, True, (sample,), allocator=allocator)
        result = type("R", (), {"baseline": (wave,), "warmups": (wave,), "points": (wave,), "max_output_verified": False})()
        raw = coedit_capacity_check._raw(result)
        self.assertNotIn("profile_eligible", raw)
        self.assertEqual(set(raw), {"status", "reason", "max_output_verified", "failure_phase", "failure_kind", "baseline", "warmups", "measured"})
        self.assertNotIn("ids", raw["baseline"][0])
        self.assertEqual(raw["baseline"][0]["request_count"], 1)
        self.assertEqual(raw["baseline"][0]["allocator"]["peak_reserved"], 40)
        self.assertEqual(raw["baseline"][0]["sample_summary"]["overlap_count"], 1)
        self.assertIsNotNone(raw["baseline"][0]["sample_summary"]["first_in_window"])
        self.assertNotIn("request_ids", raw["baseline"][0]["native_observation"])
        self.assertEqual(raw["baseline"][0]["decoder_workload"], {"steps": [], "max_output_tokens": None})

    def test_raw_discovery_has_observation_only_contract(self):
        wave = coedit_capacity_check.Wave(1, 1, ("secret",), 1, True, 1)
        result = DiscoveryResult("complete", 4, 4, None, "observed_through_ceiling",
                                 baseline=(wave,), points=(wave,), expected_max_output_tokens=64)
        raw = coedit_capacity_check._raw(result)
        self.assertEqual((raw["observed_safe_through"], raw["candidate_ceiling"], raw["stop_p"]), (4, 4, None))
        self.assertIsNone(raw["memory_safe_n"])
        self.assertFalse(raw["profile_eligible"])
        self.assertNotIn("secret", str(raw))

    def test_raw_discovery_does_not_require_capacity_warmups(self):
        result = DiscoveryResult("incomplete", None, 2, 1, "invalid_baseline",
                                 failure_phase="baseline", failure_kind="native_batch_correlation")
        raw = coedit_capacity_check._raw(result)
        self.assertEqual((raw["warmups"], raw["measured"], raw["failure_kind"]),
                         ([], [], "native_batch_correlation"))

    def test_discovery_flag_is_opt_in_and_default_parser_remains_compatible(self):
        with patch("sys.argv", ["coedit_capacity_check", "--models-root", "/missing",
                                 "--manifest", "/missing", "--target-gpu-uuid", "bad",
                                 "--native-batch-size", "2", "--configured-fixture"]):
            self.assertEqual(coedit_capacity_check.main(), 2)

    def test_main_exit_status_is_mode_aware_for_terminal_results(self):
        cases = (
            (True, "complete", 0),
            (True, "incomplete", 2),
            (False, "candidate", 0),
            (False, "incomplete", 2),
        )
        def no_wait(awaitable, _timeout):
            awaitable.close()
            return None
        class Awaitable:
            def __await__(self):
                yield
                return None
            def close(self):
                pass
        for discovery, status, expected in cases:
            argv = ["coedit_capacity_check", "--models-root", "/models",
                    "--manifest", "/manifest", "--target-gpu-uuid", "GPU-test",
                    "--native-batch-size", "2", "--configured-fixture"]
            if discovery:
                argv.append("--discover-memory")
            with self.subTest(discovery=discovery, status=status), \
                 patch("sys.argv", argv), \
                 patch.object(coedit_capacity_check.asyncio, "wait_for", side_effect=no_wait), \
                 patch.object(coedit_capacity_check, "run", side_effect=lambda _args: Awaitable()), \
                 patch.object(coedit_capacity_check.asyncio, "run",
                              return_value={"status": status}):
                self.assertEqual(coedit_capacity_check.main(), expected)
        with patch("sys.argv", ["coedit_capacity_check", "--models-root", "/missing",
                                 "--manifest", "/missing", "--target-gpu-uuid", "bad",
                                 "--native-batch-size", "2", "--configured-fixture",
                                 "--discover-memory"]):
            self.assertEqual(coedit_capacity_check.main(), 2)

    def test_raw_failed_wave_is_numeric_identity_only(self):
        sample = coedit_capacity_check.MemorySample(1, 100, 0, 100)
        wave = coedit_capacity_check.Wave(1, 1, ("id",), 0, False, 0, samples=(sample,),
                                          failed=True, failure_kind="runner_error")
        result = type("R", (), {"baseline": (wave,), "warmups": (), "points": ()})()
        item = coedit_capacity_check._raw(result)["baseline"][0]
        self.assertEqual((item["failed"], item["failure_kind"], item["outputs_valid"]),
                          (True, "runner_error", False))
        self.assertEqual(item["sample_summary"]["count"], 1)

    def test_cli_batch_bounds_are_mode_aware_before_provisioning_or_proof(self):
        common = ["coedit_capacity_check", "--models-root", "/missing",
                  "--manifest", "/missing", "--target-gpu-uuid", "bad",
                  "--configured-fixture"]
        for size, discovery in ((16, False), (32, True)):
            argv = common + ["--native-batch-size", str(size)]
            if discovery:
                argv.append("--discover-memory")
            with patch("sys.argv", argv), patch.object(coedit_capacity_check, "provision") as provision, \
                 patch.object(coedit_capacity_check.LinuxGPUProof, "capture") as proof, \
                 patch.object(coedit_capacity_check, "CoEdITProvider") as provider:
                self.assertEqual(coedit_capacity_check.main(), 2)
                provision.assert_not_called()
                proof.assert_not_called()
                provider.assert_not_called()
        for size, discovery in ((17, False), (33, True)):
            argv = common + ["--native-batch-size", str(size)]
            if discovery:
                argv.append("--discover-memory")
            with patch("sys.argv", argv):
                with self.assertRaises(SystemExit) as error:
                    coedit_capacity_check.main()
                self.assertEqual(error.exception.code, 2)

    def test_raw_summary_propagates_incomplete_maximum_witness_without_identity_leakage(self):
        wave = coedit_capacity_check.Wave(1, 1, ("private-id",), 1, True, 1,
            decoder_steps=(1,), max_output_tokens=64)
        result = type("R", (), {"status": "candidate", "reason": "probe_limit_not_memory_bound",
            "failure_phase": None, "failure_kind": None, "max_output_verified": False,
            "baseline": (wave,), "warmups": (), "points": ()})()
        raw = coedit_capacity_check._raw(result)
        self.assertEqual((raw["status"], raw["max_output_verified"],
                          raw["baseline"][0]["decoder_workload"]),
                         ("candidate", False, {"steps": [1], "max_output_tokens": 64}))
        self.assertNotIn("private-id", str(raw))

    def test_raw_retains_derived_category_and_failed_flag_consistently(self):
        sample = coedit_capacity_check.MemorySample(1, 100, 90, 10)
        wave = coedit_capacity_check.Wave(1, 0, ("id",), 1, True, 1, 1, 2, True,
                                          (sample,), allocator=coedit_capacity_check.AllocatorObservation(10, 20, 30, 40, 10, 20),
                                          failed=True, failure_kind="reserve", phase="warmup")
        result = type("R", (), {"status": "incomplete", "reason": "reserve_breached",
                                  "failure_phase": "warmup", "failure_kind": "reserve",
                                  "baseline": (), "warmups": (wave,), "points": ()})()
        raw = coedit_capacity_check._raw(result)
        self.assertEqual((raw["failure_phase"], raw["failure_kind"]), ("warmup", "reserve"))
        self.assertEqual((raw["warmups"][0]["failed"], raw["warmups"][0]["failure_kind"]), (True, "reserve"))

    def test_raw_retains_no_execution_and_chronology_categories_consistently(self):
        sample = coedit_capacity_check.MemorySample(10, 100, 0, 100)
        allocator = coedit_capacity_check.AllocatorObservation(10, 20, 30, 40, 10, 20)
        for kind in ("no_execution_sample", "chronology"):
            wave = coedit_capacity_check.Wave(1, 1, ("id",), 1, True, 1, 1, 2, True,
                                              (sample,), allocator=allocator, failed=True,
                                              failure_kind=kind, phase="baseline")
            result = type("R", (), {"status": "incomplete", "reason": "invalid_serial_baseline",
                                      "failure_phase": "baseline", "failure_kind": kind,
                                      "baseline": (wave,), "warmups": (), "points": ()})()
            raw = coedit_capacity_check._raw(result)
            self.assertEqual((raw["failure_kind"], raw["baseline"][0]["failed"],
                              raw["baseline"][0]["failure_kind"]), (kind, True, kind))

    def test_generated_witness_is_one_child_operation_and_payload_is_canonical(self):
        class Worker:
            def __init__(self): self.calls = []
            async def call(self, operation, **values):
                self.calls.append((operation, values))
                return {"count": 128, "max": 128, "fingerprint": __import__("hashlib").sha256(
                    b'{"instruction":"fix","texts":["word word"]}').hexdigest(), "text": "word word"}
        provider = type("P", (), {"worker": Worker()})()
        result = asyncio.run(coedit_capacity_check._configured_witness(provider, "fix", "ignored", 128))
        self.assertEqual(result[1], "word word")
        self.assertEqual(len(provider.worker.calls), 1)
        self.assertTrue(provider.worker.calls[0][1]["generate"])

    def test_raw_output_refuses_existing_path_and_cleans_temp(self):
        sample = coedit_capacity_check.MemorySample(1, 100, 0, 100)
        wave = coedit_capacity_check.Wave(1, 1, ("id",), 1, True, 1, 1, 2, True, (sample,), allocator=coedit_capacity_check.AllocatorObservation(10, 20, 30, 40, 10, 20))
        result = type("R", (), {"baseline": (wave,), "warmups": (), "points": ()})()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            coedit_capacity_check._write_raw(path, result)
            with self.assertRaises(FileExistsError): coedit_capacity_check._write_raw(path, result)
            self.assertEqual(list(Path(directory).glob(".*.raw.json.*")), [])

    def test_raw_output_rejects_artifacts_over_hard_bound(self):
        sample = coedit_capacity_check.MemorySample(1, 100, 0, 100)
        wave = coedit_capacity_check.Wave(1, 1, ("id",), 1, True, 1, 1, 2, True, (sample,), allocator=coedit_capacity_check.AllocatorObservation(10, 20, 30, 40, 10, 20))
        result = type("R", (), {"baseline": (wave,), "warmups": (), "points": ()})()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(coedit_capacity_check, "MAX_JSON_BYTES", 1):
                with self.assertRaises(ValueError): coedit_capacity_check._write_raw(Path(directory) / "raw.json", result)

    def test_full_discovery_raw_artifact_preserves_schema_and_fits_bound(self):
        sample = coedit_capacity_check.MemorySample(1, 100, 0, 100)
        allocator = coedit_capacity_check.AllocatorObservation(10, 20, 30, 40, 10, 20)
        waves = tuple(coedit_capacity_check.Wave(p, wave_number, tuple(f"id-{p}-{wave_number}-{n}" for n in range(p)), 1,
            True, p, 1, 2, True, (sample,), allocator=allocator, decoder_steps=(64,) * p,
            max_output_tokens=64, observation_count=1, observed_native_batch_sizes=(p,),
            native_request_correlation=True) for p, wave_number in
            [(1, n) for n in range(1, 5)] + [(p, n) for p in range(2, 33) for n in range(1, 5)])
        result = type("R", (), {"status": "complete", "reason": "observed", "max_output_verified": True,
            "failure_phase": None, "failure_kind": None, "baseline": (), "warmups": (), "points": waves})()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            coedit_capacity_check._write_raw(path, result)
            self.assertLessEqual(path.stat().st_size, coedit_capacity_check.MAX_JSON_BYTES)
            value = __import__("json").loads(path.read_bytes())
            self.assertEqual(len(value["measured"]), 128)
            self.assertEqual(len(value["measured"]), 128)
            self.assertEqual(value["measured"][0]["decoder_workload"],
                             {"steps": [64], "max_output_tokens": 64})
            self.assertEqual(sum(item["request_count"] for item in value["measured"]), 2112)

    def test_main_preserves_classified_failure_when_diagnostic_persistence_fails(self):
        args = type("Args", (), {"input": None, "configured_fixture": True, "raw_output": Path("raw.json"), "_result": None,
                                  "_cleanup_proved": True, "discover_memory": False, "native_batch_size": 1,
                                  "models_root": Path("/missing"), "manifest": Path("/missing"),
                                  "target_gpu_uuid": "bad", "host_pid_namespace": False})()
        def reject_wait_for(awaitable, _timeout):
            awaitable.close()
            raise ValueError("original")
        with patch("sys.argv", ["coedit_capacity_check", "--models-root", "/missing", "--manifest", "/missing",
                                 "--target-gpu-uuid", "bad", "--native-batch-size", "1", "--configured-fixture"]), \
             patch.object(coedit_capacity_check.asyncio, "wait_for", side_effect=reject_wait_for), \
             patch.object(coedit_capacity_check, "_write_raw", side_effect=ValueError("diagnostic")), \
             patch.object(coedit_capacity_check, "argparse") as argparse:
            parser = argparse.ArgumentParser.return_value
            parser.parse_args.return_value = args
            self.assertEqual(coedit_capacity_check.main(), 2)

    def test_discovery_one_is_rejected_by_main_before_async_run(self):
        with patch("sys.argv", ["coedit_capacity_check", "--models-root", "/missing", "--manifest", "/missing",
                                 "--target-gpu-uuid", "bad", "--native-batch-size", "1", "--configured-fixture",
                                 "--discover-memory"]), patch.object(coedit_capacity_check, "run") as run:
            with self.assertRaises(SystemExit):
                coedit_capacity_check.main()
            run.assert_not_called()

    def test_discovery_one_is_rejected_by_run_before_input_or_gpu_work(self):
        args = type("Args", (), {"host_pid_namespace": True, "target_gpu_uuid": "bad",
            "native_batch_size": 1, "discover_memory": True, "input": None,
            "configured_fixture": True})()
        with patch.object(coedit_capacity_check, "_input") as input_value, \
             patch.object(coedit_capacity_check, "_candidate_manifest") as manifest, \
             patch.object(coedit_capacity_check, "provision") as provision:
            with self.assertRaisesRegex(ValueError, "native batch must be 2..32"):
                asyncio.run(coedit_capacity_check.run(args))
            input_value.assert_not_called()
            manifest.assert_not_called()
            provision.assert_not_called()

    def test_failed_harness_cleanup_is_attempted_and_failure_is_not_success(self):
        class Provider:
            async def unload(self): self.unloaded = True
            async def verify_cleanup(self): return False
        provider = Provider()
        self.assertFalse(asyncio.run(provider.verify_cleanup()))
        asyncio.run(provider.unload())
        self.assertTrue(provider.unloaded)

    def test_run_propagates_discovery_observation_and_proves_cleanup(self):
        class Proof:
            target_uuid = "GPU-01234567-0123-0123-0123-0123456789ab"
            identity = object()
            supervisor_identity = object()
            async def residency(self): return type("R", (), {"runners": (object(),)})()
            async def cleanup(self): return True
        class Provider:
            config = type("C", (), {"max_output_tokens": 64})()
            async def unload(self): pass
            async def verify_cleanup(self): return True
        proof, provider = Proof(), Provider()
        session = type("S", (), {"session_token": "session"})()
        rm = type("RM", (), {"start_session": AsyncMock(return_value=session),
                               "stop_session": AsyncMock()})()
        result = DiscoveryResult("incomplete", 1, 2, 2, "reserve_breached",
                                 failure_phase="discovery", failure_kind="reserve",
                                 expected_max_output_tokens=64)
        args = type("Args", (), {"host_pid_namespace": True, "target_gpu_uuid": proof.target_uuid,
            "native_batch_size": 2, "input": None, "configured_fixture": True,
            "models_root": Path("/models"), "manifest": Path("/manifest"), "raw_output": None,
            "discover_memory": True})()
        with patch.object(coedit_capacity_check, "ResourceManager", return_value=rm), \
             patch.object(coedit_capacity_check, "provision", return_value={"manifest_sha256": "manifest"}), \
             patch.object(coedit_capacity_check.LinuxGPUProof, "capture", return_value=proof), \
             patch.object(coedit_capacity_check, "_runtime_identity", return_value="runtime"), \
             patch.object(coedit_capacity_check, "PythonProviderConfig", return_value=object()), \
             patch.object(coedit_capacity_check, "GPUProof", return_value=object()), \
             patch.object(coedit_capacity_check, "_profile", return_value=object()), \
             patch.object(coedit_capacity_check, "CoEdITProvider", return_value=provider), \
             patch.object(coedit_capacity_check, "_candidate_manifest", return_value=({}, "model")), \
             patch.object(coedit_capacity_check, "_configured_witness", AsyncMock(return_value=("fix", "text", {"count": 128, "max": 128, "fingerprint": hashlib.sha256(b'{"instruction":"fix","texts":["text"]}').hexdigest()}))), \
             patch.object(coedit_capacity_check, "discover_memory", AsyncMock(return_value=result)):
            value = asyncio.run(coedit_capacity_check.run(args))
        self.assertEqual((value["status"], value["observed_safe_through"], value["profile_eligible"], value["cleanup"]),
                         ("incomplete", 1, False, True))
        rm.stop_session.assert_awaited_once()
        self.assertTrue(args._cleanup_proved)

    def test_terminal_cursor_keeps_full_discovery_with_default_event_retention(self):
        class Provider:
            async def validate(self, profile): pass
            async def load(self, profile): pass
            async def ready(self): pass
            async def validate_input(self, payload, *, context_size, bucket_identity): pass
            async def execute(self, request_id, payload): return ProviderResponse(b'{"texts":["ok"]}')
            async def cancel(self, request_id): pass
            async def unload(self): pass
            async def verify_cleanup(self): return True
        async def exercise():
            manager = ResourceManager()
            profile = CapacityProfile(ModelId.COEDIT, "gpu", "manifest", "model", "runtime", "adapter", "candidate",
                1, 1, 1, 0, (SampleMetadata(1, 0, 0, 0, 0, ()),), bucket_identity="bucket")
            session = await manager.start_session("capacity", ModelId.COEDIT, profile, Provider(), idempotency_key="start")
            cursor = 0
            try:
                for index in range(544):
                    request_id = f"r{index}"
                    await manager.submit(session.session_token, request_id, "capacity-1", b"x",
                                         idempotency_key=f"k{index}", bucket_identity="bucket")
                    event = await coedit_capacity_check._terminal(manager, session, request_id,
                                                                  after_sequence=cursor)
                    cursor = event.sequence
                with self.assertRaises(ResourceManagerError):
                    await anext(manager.watch_progress(session.session_token))
                self.assertGreater(cursor, manager.max_events)
                self.assertEqual(manager.max_events, 1024)
            finally:
                await manager.stop_session(session.session_token, idempotency_key="stop")
        asyncio.run(exercise())

    def test_flat_image_import_uses_flat_adapter_helper(self):
        """Docker copies both scripts to /opt/llm rather than a tools package."""
        root = Path(__file__).resolve().parents[2]
        source = (root / "tools/compatibility/coedit_capacity_check.py").read_text(encoding="utf-8")
        adapter = (root / "tools/compatibility/coedit_adapter_check.py").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            flat = Path(directory)
            (flat / "coedit_capacity_check.py").write_text(source, encoding="utf-8")
            (flat / "coedit_adapter_check.py").write_text(adapter, encoding="utf-8")
            blocked = {name: value for name, value in sys.modules.items()
                       if name == "tools" or name.startswith("tools.")}
            with patch.object(sys, "path", [str(flat), str(root)]), patch.dict(sys.modules, {"tools": None}, clear=False):
                for name in blocked: sys.modules.pop(name, None)
                try:
                    spec = importlib.util.spec_from_file_location("flat_capacity", flat / "coedit_capacity_check.py")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.assertEqual(module.ADAPTER, coedit_capacity_check.ADAPTER)
                finally:
                    sys.modules.update(blocked)

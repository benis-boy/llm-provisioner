import argparse
import json
import os
import subprocess
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tools.compatibility import run_measurement_matrix as subject
from tools.compatibility import measure_profiles


class MeasurementMatrixRunnerTests(unittest.TestCase):
    def test_direct_script_help_resolves_local_debug_trace_import(self):
        completed = subprocess.run(
            [str(Path(".venv/bin/python")), str(Path(subject.__file__)), "--help"],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_main_argument_failure_is_one_sanitized_json_line(self):
        with patch("sys.stdout", new_callable=StringIO) as output:
            status = subject.main(["--image", "image", "--bundle", "/missing",
                                   "--gpu-uuid", "not-gpu", "--ollama-version", "v",
                                   "--output", "/tmp/out"])
        self.assertEqual(status, 2)
        value = json.loads(output.getvalue())
        self.assertEqual(value["status"], "incomplete")
        self.assertNotIn("/missing", output.getvalue())
    def test_create_constructs_network_gpu_pid_and_readonly_mounts(self):
        with patch.object(subject, "run") as runner:
            subject._create("c", "image", [("v", "/opt/measurement", True)],
                            ["--bundle", "/opt/measurement"], gpu="GPU-1", pid=True)
        command = runner.call_args.args[0]
        self.assertEqual(command[:5], ["docker", "create", "--name", "c", "--network"])
        self.assertIn("none", command)
        self.assertIn("--pid=host", command)
        self.assertIn("device=GPU-1", command)
        self.assertIn("type=volume,src=v,dst=/opt/measurement,readonly", command)
        self.assertEqual(command[-3:], ["image", "--bundle", "/opt/measurement"])

    def test_result_rejects_missing_malformed_and_partial_matrix(self):
        for value in ("", "not-json", json.dumps({"status": "incomplete"}),
                      json.dumps({"status": "complete", "matrix": [], "profiles": []})):
            with self.subTest(value=value), self.assertRaises((ValueError, json.JSONDecodeError)):
                subject._result(value)

    def test_result_rejects_duplicate_missing_and_wrong_identity_summaries(self):
        matrix = [[model.value, selector] for model, selector in subject.measurement_matrix()]
        valid = {"status": "complete", "matrix": matrix,
                 "profiles": [{"model": model, "profile_identity": model + "-identity"}
                              for model, _ in matrix]}
        for profiles in (valid["profiles"][:2],
                         [*valid["profiles"][:2], valid["profiles"][0]],
                         [{**item, "model": "wrong"} for item in valid["profiles"]]):
            with self.subTest(profiles=profiles), self.assertRaises(ValueError):
                subject._result(json.dumps({**valid, "profiles": profiles}))

    def test_two_preflights_are_required_before_measurement(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        output = root / "profiles.sqlite"
        args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                                  ollama_version="0.11.6", ceiling=2, output=output)
        calls = []
        probes = 0

        def runner(command, *, timeout=60, check=True):
            nonlocal probes
            calls.append(command)
            if command[:3] == ["docker", "run", "--rm"]:
                probes += 1
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-verify"):
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 1, "", "")
            return subprocess.CompletedProcess(command, 0, "", "")

        # The runner seam keeps Docker entirely out of the test.
        with patch.object(subject, "run", side_effect=runner):
            with self.assertRaises(RuntimeError):
                subject.orchestrate(args)
        self.assertEqual(probes, 2)
        measure_index = next(i for i, call in enumerate(calls) if call[-1].endswith("-measure"))
        self.assertTrue(all(i < measure_index for i, call in enumerate(calls)
                            if call[:3] == ["docker", "run", "--rm"]))

    def test_failure_never_exports_and_owned_state_is_cleaned(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                                  ollama_version="0.11.6", ceiling=2,
                                  output=root / "profiles.sqlite")
        calls = []

        def runner(command, *, timeout=60, check=True):
            calls.append(command)
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-verify"):
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 1, "", "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(subject, "run", side_effect=runner), self.assertRaises(RuntimeError):
            subject.orchestrate(args)
        self.assertFalse(args.output.exists())
        self.assertFalse(any(call[-1].endswith("-export") for call in calls
                             if call[:3] == ["docker", "start", "-a"]))
        self.assertTrue(any(call[:3] == ["docker", "rm", "-f"] for call in calls))
        self.assertTrue(any(call[:3] == ["docker", "volume", "rm"] for call in calls))

    def test_primary_failure_proves_owned_docker_cleanup(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        output = root / "profiles.sqlite"
        args = ["--image", "image", "--bundle", str(bundle), "--gpu-uuid", "GPU-1",
                "--ollama-version", "0.11.6", "--ceiling", "2", "--output", str(output)]

        def runner(command, *, timeout=60, check=True):
            if command[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 1, "", "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(subject, "run", side_effect=runner), patch("sys.stdout", new_callable=StringIO) as output_stream:
            self.assertEqual(subject.main(args), 2)
        self.assertEqual(json.loads(output_stream.getvalue())["docker_cleanup"], "proved")

    def test_failed_owned_cleanup_remains_unproved(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                                  ollama_version="0.11.6", ceiling=1,
                                  output=root / "profiles.sqlite")

        def runner(command, *, timeout=60, check=True):
            if command[:3] == ["docker", "volume", "rm"]:
                return subprocess.CompletedProcess(command, 1, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 1, "", "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(subject, "run", side_effect=runner), self.assertRaises(subject.OperatorFailure) as raised:
            subject.orchestrate(args)
        self.assertEqual(raised.exception.code, "cleanup_failed")

    def test_nonzero_measurement_reads_result_and_emits_only_safe_inner_fields(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        args = ["--image", "image", "--bundle", str(bundle), "--gpu-uuid", "GPU-1",
                "--ollama-version", "0.11.6", "--ceiling", "2", "--output",
                str(root / "profiles.sqlite")]
        calls = []
        inner = {"stage": "measure", "failure_kind": "provider_failed",
                 "exception_type": "RuntimeError", "reason": "secret reason",
                 "message": "secret message", "path": "/secret/path",
                  "model": "SmolLM", "measurement_failure_code": "oom",
                  "measurement_failure_detail": "failed_output"}

        def runner(command, *, timeout=60, check=True):
            calls.append(command)
            if command[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 1, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-result"):
                return subprocess.CompletedProcess(command, 0, json.dumps(inner), "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(subject, "run", side_effect=runner), patch("sys.stdout", new_callable=StringIO) as output:
            status = subject.main(args)

        self.assertEqual(status, 2)
        emitted = json.loads(output.getvalue())
        self.assertEqual(emitted["inner_stage"], "measure")
        self.assertEqual(emitted["inner_failure_kind"], "provider_failed")
        self.assertEqual(emitted["inner_exception_type"], "RuntimeError")
        self.assertEqual(emitted["inner_model"], "SmolLM")
        self.assertEqual(emitted["inner_measurement_failure_code"], "oom")
        self.assertEqual(emitted["inner_measurement_failure_detail"], "failed_output")
        self.assertNotIn("reason", emitted)
        self.assertNotIn("message", emitted)
        self.assertNotIn("path", emitted)

    def test_debug_flag_propagates_and_failure_trace_is_retained_before_cleanup(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        output = root / "profiles.sqlite"
        calls = []
        inner = {"status": "incomplete", "stage": "measurement", "failure_kind": "runner_error"}
        trace_text = '{"schema":"llm.debug-trace.v1","sequence":1,"component":"unknown","event":"wave","state":"failure"}\n'

        def runner(command, *, timeout=60, check=True):
            calls.append(command)
            if command[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 1, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-result"):
                return subprocess.CompletedProcess(command, 0, json.dumps(inner), "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-trace"):
                return subprocess.CompletedProcess(command, 0, trace_text, "")
            return subprocess.CompletedProcess(command, 0, "", "")

        args = ["--image", "image", "--bundle", str(bundle), "--gpu-uuid", "GPU-1",
                "--ollama-version", "0.11.6", "--ceiling", "2", "--output", str(output), "--debug"]
        with patch.object(subject, "run", side_effect=runner), patch("sys.stderr", new_callable=StringIO) as stderr:
            self.assertEqual(subject.main(args), 2)
        measure_create = next(call for call in calls if call[:3] == ["docker", "create", "--name"] and "--debug" in call)
        self.assertIn("--debug", measure_create)
        self.assertLess(next(i for i, call in enumerate(calls) if call[-1].endswith("-trace")),
                        next(i for i, call in enumerate(calls) if call[:3] == ["docker", "volume", "rm"]))
        sidecar = Path(str(output) + ".debug.jsonl")
        self.assertEqual(sidecar.read_text(), trace_text)
        self.assertIn(trace_text.strip(), stderr.getvalue())

    def test_debug_sidecar_never_clobbers_existing_file(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        output = root / "profiles.sqlite"
        sidecar = Path(str(output) + ".debug.jsonl")
        sidecar.write_text("foreign\n")
        args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                                  ollama_version="0.11.6", ceiling=1, output=output, debug=True)
        with patch.object(subject, "run", side_effect=RuntimeError("stop")), self.assertRaises(subject.OperatorFailure):
            subject.orchestrate(args)
        self.assertEqual(sidecar.read_text(), "foreign\n")

    def test_sidecar_short_write_is_atomic_and_retryable(self):
        root = Path(tempfile.mkdtemp()); self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        sidecar = root / "profiles.sqlite.debug.jsonl"
        text = '{"schema":"llm.debug-trace.v1","sequence":1,"component":"unknown","event":"wave","state":"failure"}\n'
        original = subject.os.write
        calls = [0]
        def short(fd, data):
            calls[0] += 1
            if calls[0] == 1:
                return 0
            return original(fd, data)
        with patch.object(subject.os, "write", side_effect=short):
            subject._publish_sidecar(sidecar, text)
        self.assertFalse(sidecar.exists())
        subject._publish_sidecar(sidecar, text)
        self.assertEqual(sidecar.read_text(), text)

    def test_debug_does_not_change_stdout_result_contract(self):
        with patch("sys.stdout", new_callable=StringIO) as stdout, patch("sys.stderr", new_callable=StringIO):
            status = subject.main(["--image", "image", "--bundle", "/missing", "--gpu-uuid", "bad",
                                  "--ollama-version", "v", "--output", "/tmp/out", "--debug"])
        self.assertEqual(status, 2)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "incomplete")
        self.assertNotIn("schema", result)

    def test_trace_is_retrieved_on_result_reader_exception_before_cleanup(self):
        root = Path(tempfile.mkdtemp()); self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir(); output = root / "profiles.sqlite"
        calls = []
        def runner(command, *, timeout=60, check=True):
            calls.append(command)
            if command[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-trace"):
                return subprocess.CompletedProcess(command, 0, "trace\n", "")
            return subprocess.CompletedProcess(command, 0, "", "")
        with patch.object(subject, "run", side_effect=runner), patch.object(subject, "_read_runtime_result", side_effect=RuntimeError("reader")):
            with self.assertRaises(subject.OperatorFailure):
                subject.orchestrate(argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                    ollama_version="0.11.6", ceiling=1, output=output, debug=True))
        trace_index = next(i for i, call in enumerate(calls) if call[-1].endswith("-trace"))
        cleanup_index = next(i for i, call in enumerate(calls) if call[:3] == ["docker", "volume", "rm"])
        self.assertLess(trace_index, cleanup_index)

    def test_trace_file_rejects_wrong_path_and_symlink(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); db = root / "profiles.sqlite"
            config = SimpleNamespace(profile_db=db)
            with self.assertRaises(ValueError):
                measure_profiles._trace_file(root / "other.jsonl", config)
            link = root / "debug-trace.jsonl"; target = root / "target"; target.write_text("")
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                measure_profiles._trace_file(link, config)

    def test_unknown_inner_measurement_failure_detail_is_omitted(self):
        result = subject._failure("measurement_failed", "measurement", inner={
            "measurement_failure_detail": "provider-secret"})
        self.assertNotIn("inner_measurement_failure_detail", result)

    def test_smollm_stage_category_is_relayed_from_measurement_failure_detail(self):
        result = subject._failure("measurement_failed", "measurement", inner={
            "measurement_failure_detail": "smollm_input_validation"})
        self.assertEqual(result["inner_measurement_failure_detail"], "smollm_input_validation")
        self.assertEqual(result["inner_failure_code"], "smollm_input_validation")

    def test_persistence_failure_code_is_transportable_without_detail(self):
        result = subject._failure("measurement_failed", "measurement", inner={
            "model": "GECToR",
            "measurement_failure_code": "persistence_validation_failed",
            "measurement_failure_detail": "/private/prompt=secret",
        })
        self.assertEqual(result["inner_model"], "GECToR")
        self.assertEqual(result["inner_measurement_failure_code"],
                         "persistence_validation_failed")
        self.assertNotIn("inner_measurement_failure_detail", result)
        self.assertNotIn("secret", json.dumps(result))

    def test_maximum_witness_detail_is_relayed_from_closed_vocabulary(self):
        result = subject._failure("measurement_failed", "measurement", inner={
            "model": "CoEdIT",
            "measurement_failure_code": "maximum_witness_failed",
            "measurement_failure_detail": "coedit_token_count",
        })
        self.assertEqual(result["inner_model"], "CoEdIT")
        self.assertEqual(result["inner_measurement_failure_code"],
                         "maximum_witness_failed")
        self.assertEqual(result["inner_measurement_failure_detail"],
                         "coedit_token_count")

    def test_outer_failure_envelope_surfaces_only_safe_fields(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        args = ["--image", "image", "--bundle", str(bundle), "--gpu-uuid", "GPU-1",
                "--ollama-version", "0.11.6", "--output", str(root / "profiles.sqlite")]
        calls = []
        inner = {"stage": "measure", "failure_kind": "provider_failed",
                 "exception_type": "RuntimeError", "reason": "/private/secret",
                 "message": "token=secret", "path": "/private/path"}

        def runner(command, *, timeout=60, check=True):
            calls.append(command)
            if command[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 1, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-result"):
                return subprocess.CompletedProcess(command, 0, json.dumps(inner), "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(subject, "run", side_effect=runner), \
             patch("sys.stdout", new_callable=StringIO) as output:
            status = subject.main(args)

        self.assertEqual(status, 2)
        emitted = json.loads(output.getvalue())
        self.assertEqual(set(emitted), {"status", "failure_code", "stage", "db_retained",
                                        "docker_cleanup", "inner_stage", "inner_failure_kind",
                                        "inner_exception_type"})
        self.assertEqual(emitted["inner_stage"], "measure")
        self.assertEqual(emitted["inner_failure_kind"], "provider_failed")
        self.assertEqual(emitted["inner_exception_type"], "RuntimeError")
        self.assertNotIn("private", output.getvalue())
        self.assertNotIn("secret", output.getvalue())

    def test_output_reservation_is_no_clobber(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        destination = root / "profiles.sqlite"
        destination.write_bytes(b"foreign")
        args = argparse.Namespace(image="image", bundle=root, gpu_uuid="GPU-1",
                                  ollama_version="0.11.6", ceiling=1, output=destination)
        with patch.object(subject, "run", side_effect=RuntimeError("stop")), self.assertRaises(subject.OperatorFailure) as failure:
            subject.orchestrate(args)
        self.assertEqual(failure.exception.code, "output_reserved")
        self.assertEqual(destination.read_bytes(), b"foreign")

    def test_export_audit_rejects_wal_sidecar_and_tampered_profile_store(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        exported = root / "profiles.sqlite"
        exported.write_bytes(b"sqlite")
        exported.with_name(exported.name + "-wal").write_bytes(b"tampered")
        result = {"profiles": [{"model": "SmolLM", "profile_identity": "a"},
                                {"model": "CoEdIT", "profile_identity": "b"},
                                {"model": "GECToR", "profile_identity": "c"}]}
        with self.assertRaises(ValueError):
            subject._audit_export(exported, result)


if __name__ == "__main__":
    unittest.main()

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
import subprocess
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.compatibility import run_smollm_p2_diagnostic as subject


class DiagnosticOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        self.bundle = root / "bundle"; self.bundle.mkdir()
        (self.bundle / "current").symlink_to("selected")
        self.request = root / "request.json"; self.request.write_text('"request"')
        self.args = argparse.Namespace(image="example@sha256:abc", bundle=self.bundle,
                                       request=self.request, gpu_uuid="GPU-exact",
                                       ollama_version="0.11.6")
        self.owned = subject._Owned("owned", "owned-bundle", "owned-runtime", "owned-request",
                                    "owned-preflight", "owned-bundle-transfer", "owned-request-transfer",
                                    "owned-verify", "owned-diagnostic", "owned-result-reader")

    def tearDown(self):
        self.tmp.cleanup()

    def _runner(self, *, preflight="", diagnostic=0, diagnostic_stdout=None,
                diagnostic_stderr="", inspect_state=None, result_file=None, fail=None):
        calls = []
        valid = (json.dumps({
            "model":"SmolLM", "provider_class":"SmolLMProvider", "provider_category":"ollama",
            "stage":"p2_wave", "concurrency":2, "wave":1, "status":"complete",
            "failure_kind":None, "failure_code":None, "failure_message":None,
            "native_overlap":True, "native_request_correlation":True,
            "native_observation_count":2, "native_batch_sizes":[1, 1],
            "observation_drops":0, "cleanup":"proved"})
                 if diagnostic_stdout is None else diagnostic_stdout)
        def run(command, check=True, timeout=60):
            calls.append(command)
            text = ""
            code = 0
            if command[:3] == ["docker", "start", "-a"] and command[3] == self.owned.preflight:
                text = preflight
            if command[:3] == ["docker", "start", "-a"] and command[3] == self.owned.diagnostic:
                code = diagnostic
                text = valid
            if command[:3] == ["docker", "start", "-a"] and command[3] == self.owned.result_reader:
                code = 0 if result_file is not None else 1
                text = result_file or ""
            if command[:3] == ["docker", "container", "inspect"] and command[-1] == self.owned.diagnostic:
                text = inspect_state if inspect_state is not None else json.dumps({
                    "Status": "exited", "ExitCode": diagnostic, "OOMKilled": False, "Error": ""})
            if ((command[:3] == ["docker", "container", "inspect"]
                 and "--format" not in command)
                    or command[:3] == ["docker", "volume", "inspect"]):
                code = 1
            if fail is not None and fail(command):
                raise RuntimeError("hostile /prompt/result detail")
            stderr = diagnostic_stderr if (command[:3] == ["docker", "start", "-a"]
                                            and command[3] == self.owned.diagnostic) else ""
            return subprocess.CompletedProcess(command, code, text, stderr)
        return calls, run

    def test_constructs_isolated_volumes_verifies_pristine_bundle_and_returns_diagnostic_status(self):
        calls, runner = self._runner()
        self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 0)
        copied = [call for call in calls if call[:2] == ["docker", "cp"]]
        self.assertEqual(copied[0][2], str(self.bundle) + "/.")
        self.assertEqual(copied[1][2], str(self.request))
        diagnostic = next(call for call in calls if call[:3] == ["docker", "create", "--name"]
                          and call[3] == self.owned.diagnostic)
        self.assertIn("--pid=host", diagnostic)
        self.assertIn("device=GPU-exact", diagnostic)
        self.assertIn("/opt/diagnostic-request/request.json", diagnostic)
        self.assertIn("--result-file", diagnostic)
        self.assertIn("type=volume,src=owned-bundle,dst=/opt/measurement,readonly", diagnostic)
        self.assertLess(next(i for i, c in enumerate(calls) if self.owned.verifier in c),
                        next(i for i, c in enumerate(calls) if self.owned.diagnostic in c))

    def test_foreign_compute_preflight_aborts_before_transfer(self):
        calls, runner = self._runner(preflight="123\n")
        self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)
        self.assertFalse(any(call[:2] == ["docker", "cp"] for call in calls))
        self.assertFalse(any(call[:3] == ["docker", "volume", "create"] for call in calls))

    def test_second_preflight_abort(self):
        calls = []
        probes = 0
        def runner(command, check=True, timeout=60):
            nonlocal probes
            calls.append(command)
            text = ""
            if command[:3] == ["docker", "start", "-a"] and command[3] == self.owned.preflight:
                probes += 1
                text = "foreign\n" if probes == 2 else ""
            return subprocess.CompletedProcess(command, 0, text, "")
        self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)
        self.assertFalse(any(command[3:] == [self.owned.diagnostic] for command in calls
                             if command[:3] == ["docker", "start", "-a"]))

    def test_failure_still_removes_and_verifies_only_owned_resources(self):
        calls, runner = self._runner(fail=lambda command: command[:2] == ["docker", "cp"])
        self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)
        self.assertIn(["docker", "rm", "-f", self.owned.bundle_transfer], calls)
        self.assertIn(["docker", "volume", "rm", self.owned.bundle_volume], calls)
        self.assertIn(["docker", "container", "inspect", self.owned.bundle_transfer], calls)
        self.assertIn(["docker", "volume", "inspect", self.owned.bundle_volume], calls)

    def test_capture_is_bounded_and_not_emitted(self):
        calls, runner = self._runner(diagnostic=2)
        self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)
        self.assertFalse(any(call[:2] == ["docker", "logs"] for call in calls))
        self.assertTrue(any(call[:3] == ["docker", "container", "inspect"] for call in calls))

    def test_accepts_single_diagnostic_json_relayed_on_stderr(self):
        valid = self._runner()[1](["docker", "start", "-a", self.owned.diagnostic]).stdout
        _, runner = self._runner(diagnostic_stdout="", diagnostic_stderr=valid)
        self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 0)

    def test_rejects_missing_malformed_and_multiple_json(self):
        for text in ("", "not-json", '{"status":"complete"} {}'):
            calls, runner = self._runner(diagnostic_stdout=text)
            self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)

    def test_rejects_native_invariant_failures(self):
        valid = json.loads(self._runner()[1](["docker", "start", "-a", self.owned.diagnostic]).stdout)
        for field, value in (("native_overlap", False), ("native_request_correlation", False),
                             ("native_observation_count", 1), ("native_batch_sizes", [2]),
                             ("observation_drops", 1), ("cleanup", "unproved")):
            result = dict(valid, **{field: value})
            _, runner = self._runner(diagnostic_stdout=json.dumps(result))
            self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)

    def test_invalid_native_evidence_returns_safe_failure_without_escaping(self):
        valid = json.loads(self._runner()[1](
            ["docker", "start", "-a", self.owned.diagnostic]).stdout)
        _, runner = self._runner(diagnostic_stdout=json.dumps(
            dict(valid, native_request_correlation=False)))
        self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)
        self.assertEqual(subject.orchestrate.last_result, {
            "status": "incomplete", "failure_code": "diagnostic_native_proof_failed",
            "application_cleanup": "not_proved", "docker_cleanup": "proved",
        })

    def test_main_emits_one_sanitized_json_result_for_invalid_native_evidence(self):
        output = StringIO()
        original = subject.orchestrate

        def invalid_native_evidence(_args):
            invalid_native_evidence.last_result = {
                "status": "incomplete", "failure_code": "diagnostic_native_proof_failed",
                "application_cleanup": "not_proved", "docker_cleanup": "proved",
            }
            return 2

        try:
            subject.orchestrate = invalid_native_evidence
            with redirect_stdout(output):
                self.assertEqual(subject.main([
                    "--image", self.args.image, "--bundle", str(self.bundle),
                    "--request", str(self.request), "--gpu-uuid", self.args.gpu_uuid,
                    "--ollama-version", self.args.ollama_version]), 2)
        finally:
            subject.orchestrate = original
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), {
            "status": "incomplete", "failure_code": "diagnostic_native_proof_failed",
            "application_cleanup": "not_proved", "docker_cleanup": "proved",
        })

    def test_missing_and_malformed_output_have_distinct_safe_codes(self):
        for text, code in (("not-json", "diagnostic_result_malformed"),
                           ('{"status":"complete"} {}', "diagnostic_result_multiple")):
            with self.subTest(code=code):
                _, runner = self._runner(diagnostic_stdout=text)
                self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)
                self.assertEqual(subject.orchestrate.last_result["failure_code"], code)
                self.assertEqual(subject.orchestrate.last_result["docker_cleanup"], "proved")

    def test_stdout_absent_uses_owned_result_reader(self):
        valid = self._runner()[1](["docker", "start", "-a", self.owned.diagnostic]).stdout
        calls, runner = self._runner(diagnostic_stdout="", result_file=valid)
        self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 0)
        self.assertTrue(any(self.owned.result_reader in call for call in calls))
        self.assertIn(["docker", "rm", "-f", self.owned.result_reader], calls)

    def test_stdout_absent_and_missing_result_file_uses_inspect_classification(self):
        _, runner = self._runner(diagnostic_stdout="", result_file=None)
        self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)
        self.assertEqual(subject.orchestrate.last_result["failure_code"],
                         "result_file_missing")

    def test_malformed_result_file_is_classified_without_contents(self):
        _, runner = self._runner(diagnostic_stdout="", result_file="/private not-json")
        self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)
        result = subject.orchestrate.last_result
        self.assertEqual(result["failure_code"], "diagnostic_result_malformed")
        self.assertNotIn("private", json.dumps(result))

    def test_missing_output_uses_safe_inspect_state_classification(self):
        states = (
            ({"Status": "exited", "ExitCode": 0, "OOMKilled": False, "Error": ""},
             "diagnostic_result_missing_after_success"),
            ({"Status": "exited", "ExitCode": 137, "OOMKilled": True, "Error": "private error"},
             "diagnostic_oom_killed"),
            ({"Status": "dead", "ExitCode": 125, "OOMKilled": False, "Error": "private daemon failure"},
             "diagnostic_docker_runtime_failed"),
            ({"Status": "exited", "ExitCode": 2, "OOMKilled": False, "Error": ""},
             "diagnostic_process_exit_exited"),
        )
        for state, code in states:
            with self.subTest(code=code):
                self.assertEqual(subject._missing_result_code(json.dumps(state)), code)

    def test_invalid_inspect_state_is_safe_runtime_failure(self):
        _, runner = self._runner(diagnostic_stdout="", inspect_state='{"Error":"/private"}')
        self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)
        self.assertEqual(subject.orchestrate.last_result["failure_code"],
                         "result_file_missing")

    def test_run_waits_for_delayed_pipe_drain_after_child_exit(self):
        class DelayedStream:
            def __init__(self, payload):
                self.payload = payload
                self.sent = False
            def read(self, _size):
                if not self.sent:
                    self.sent = True
                    time.sleep(.25)
                    return self.payload
                return b""
            def close(self):
                pass

        class Process:
            pid = 1
            returncode = 0
            def __init__(self):
                self.stdout = DelayedStream(b'{"late":"complete"}')
                self.stderr = DelayedStream(b"")
            def wait(self, timeout=None):
                return 0

        result = subject._run(["docker", "start"], popen=lambda *_args, **_kwargs: Process())
        self.assertEqual(result.stdout, '{"late":"complete"}')

    def test_hostile_incomplete_inner_message_is_rejected_without_escape(self):
        incomplete = {
            "model": "SmolLM", "provider_class": "SmolLMProvider",
            "provider_category": "ollama", "stage": "p2_wave", "concurrency": 2,
            "wave": 1, "status": "incomplete", "cleanup": "proved",
            "failure_kind": "provider_execution_failed", "failure_code": "runner_error",
            "failure_message": "/private prompt must not escape",
        }
        _, runner = self._runner(diagnostic=2, diagnostic_stdout=json.dumps(incomplete))
        self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)
        result = subject.orchestrate.last_result
        self.assertEqual(result["failure_code"], "diagnostic_result_schema")
        self.assertEqual(result["application_cleanup"], "not_proved")
        self.assertEqual(result["docker_cleanup"], "proved")
        self.assertNotIn("failure_message", result)

    def test_minimal_incomplete_inner_failure_is_preserved_without_success_evidence(self):
        incomplete = {
            "model": "SmolLM", "provider_class": "SmolLMProvider",
            "provider_category": "ollama", "stage": "p2_wave", "concurrency": 2,
            "wave": 1, "status": "incomplete", "failure_kind": "runner_error",
            "failure_code": "ollama_response_contract", "failure_message": "request rejected",
            "cleanup": "proved",
        }
        _, runner = self._runner(diagnostic=2, diagnostic_stdout=json.dumps(incomplete))
        self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)
        result = subject.orchestrate.last_result
        self.assertEqual(result["failure_code"], "diagnostic_incomplete")
        self.assertEqual(result["inner_failure_kind"], "runner_error")
        self.assertEqual(result["inner_failure_code"], "ollama_response_contract")
        self.assertEqual(result["application_cleanup"], "proved")

    def test_smollm_stage_category_is_accepted_without_provider_text(self):
        incomplete = {
            "model": "SmolLM", "provider_class": "SmolLMProvider",
            "provider_category": "ollama", "stage": "p2_wave", "concurrency": 2,
            "wave": 1, "status": "incomplete", "failure_kind": "runner_error",
            "failure_code": "smollm_observation_contract",
            "failure_message": "provider execution failed", "cleanup": "proved",
        }
        self.assertEqual(subject._parse_diagnostic(json.dumps(incomplete)), incomplete)

    def test_incomplete_failure_requires_bounded_classification_and_rejects_hostile_payloads(self):
        base = {
            "model": "SmolLM", "provider_class": "SmolLMProvider",
            "provider_category": "ollama", "stage": "p2_wave", "concurrency": 2,
            "wave": 1, "status": "incomplete", "failure_kind": "runner_error",
            "failure_code": "ollama_response_contract", "failure_message": "request rejected",
            "cleanup": "proved",
        }
        for change in ({"failure_code": ""}, {"failure_message": "/private/provider"},
                       {"cleanup": "unproved"}, {"native_overlap": True}):
            with self.subTest(change=change):
                with self.assertRaises(subject._DiagnosticFailure):
                    subject._parse_diagnostic(json.dumps({**base, **change}))

    def test_preflight_and_transfer_failures_use_fixed_stage_codes(self):
        cases = (
            ("foreign", self._runner(preflight="123\n")[1], "foreign_compute_present"),
            ("transfer", self._runner(fail=lambda command: command[:2] == ["docker", "cp"])[1],
             "bundle_transfer_failed"),
        )
        for name, runner, code in cases:
            with self.subTest(name=name):
                self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)
                result = subject.orchestrate.last_result
                self.assertEqual(result["failure_code"], code)
                self.assertNotIn("failure_message", result)

    def test_failure_force_removes_owned_diagnostic(self):
        calls, runner = self._runner(fail=lambda command: command[:3] == ["docker", "start", "-a"] and
                                     command[3] == self.owned.diagnostic)
        self.assertEqual(subject.orchestrate(self.args, runner=runner, owned=self.owned), 2)
        self.assertIn(["docker", "rm", "-f", self.owned.diagnostic], calls)

    def test_symlinked_parent_rejected(self):
        target = Path(self.tmp.name) / "target"; target.mkdir()
        parent = Path(self.tmp.name) / "link"
        parent.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            subject._reject_symlink_components(parent / "request.json", "request")


if __name__ == "__main__":
    unittest.main()

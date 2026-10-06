"""Run the one exact-GPU SmolLM p=2 diagnostic without host bind mounts.

This operator entry point deliberately transfers the immutable bundle and the
standalone diagnostic request through separate, uniquely owned Docker volumes.
It is not a measurement-matrix runner and never writes a profile database.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from typing import Callable


MAX_CAPTURE_BYTES = 8192
MAX_DIAGNOSTIC_JSON_BYTES = 8192
DOCKER_TIMEOUT = 60
SAFE_INNER_FAILURES = frozenset({
    "provider_execution_failed", "runner_error", "cleanup_failed", "inner_request_invalid",
    "ollama_http_status", "ollama_transport", "ollama_json_response",
    "ollama_response_contract", "ollama_telemetry_contract", "wave_timeout",
    "smollm_input_decode", "smollm_input_validation",
    "smollm_observation_contract", "smollm_internal",
    "malformed_provider_response",
    "provider_process_exit", "oom",
})
SAFE_FAILURE_STAGE_DETAILS = frozenset({
    "gpu_proof_capture",
    "validation_session_start", "maximum_witness", "validation_session_stop",
    "measurement_session_start", "p2_wave", "runner_close", "provider_unload",
    "daemon_close",
})

_DIAGNOSTIC_IDENTITY = {
    "model", "provider_class", "provider_category", "stage", "concurrency", "wave",
}
_DIAGNOSTIC_FAILURE = {"status", "failure_kind", "failure_code", "failure_message", "cleanup"}
_DIAGNOSTIC_LIFECYCLE = {"lifecycle_phase", "lifecycle_subreason"}
_DIAGNOSTIC_INCOMPLETE = _DIAGNOSTIC_FAILURE | {"failure_stage_detail"}
_DIAGNOSTIC_COMPLETE_EVIDENCE = {
    "native_overlap", "native_request_correlation", "native_observation_count",
    "native_batch_sizes", "observation_drops",
}
_DIAGNOSTIC_ALLOWED = _DIAGNOSTIC_IDENTITY | _DIAGNOSTIC_INCOMPLETE | _DIAGNOSTIC_LIFECYCLE | _DIAGNOSTIC_COMPLETE_EVIDENCE


class _DiagnosticFailure(ValueError):
    """A fixed public classification; never carries Docker/provider text."""

    def __init__(self, code: str, inner: dict[str, object] | None = None):
        super().__init__(code)
        self.code = code
        self.inner = inner


def _run(command: list[str], *, check: bool = True,
         timeout: int = DOCKER_TIMEOUT, popen=subprocess.Popen) -> subprocess.CompletedProcess[str]:
    """Run one Docker command with bounded, non-emitted output capture."""
    captured = [bytearray(), bytearray()]

    def drain(stream, index: int) -> None:
        try:
            while chunk := stream.read(4096):
                remaining = MAX_CAPTURE_BYTES - len(captured[index])
                if remaining > 0:
                    captured[index].extend(chunk[:remaining])
        except (OSError, ValueError):
            pass

    process = popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    start_new_session=True)
    assert process.stdout is not None and process.stderr is not None
    threads = [threading.Thread(target=drain, args=(stream, index))
               for index, stream in enumerate((process.stdout, process.stderr))]
    for thread in threads:
        thread.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # A timed-out child must be reaped before waiting for pipe EOF.  Do not
        # truncate a late final JSON line by closing its streams from the parent.
        process.wait()
        raise RuntimeError("Docker command timed out")
    finally:
        for thread in threads:
            thread.join()
        for stream in (process.stdout, process.stderr):
            stream.close()
    result = subprocess.CompletedProcess(
        command, process.returncode, captured[0].decode(errors="replace"),
        captured[1].decode(errors="replace"))
    if check and result.returncode:
        raise RuntimeError("Docker command failed")
    return result


@dataclass(frozen=True)
class _Owned:
    prefix: str
    bundle_volume: str
    runtime_volume: str
    request_volume: str
    preflight: str
    bundle_transfer: str
    request_transfer: str
    verifier: str
    diagnostic: str
    result_reader: str

    @classmethod
    def create(cls) -> "_Owned":
        prefix = "llm-smollm-p2-" + uuid.uuid4().hex
        return cls(prefix, prefix + "-bundle", prefix + "-runtime", prefix + "-request",
                   prefix + "-preflight", prefix + "-bundle-transfer",
                   prefix + "-request-transfer", prefix + "-verify", prefix + "-diagnostic",
                   prefix + "-result-reader")


def _create_container(runner, name: str, image: str, mounts: list[tuple[str, str, bool]],
                      arguments: list[str], *, gpu_uuid: str | None = None,
                      pid_host: bool = False, entrypoint: str | None = None) -> None:
    command = ["docker", "create", "--name", name, "--network", "none"]
    if pid_host:
        command.append("--pid=host")
    if gpu_uuid is not None:
        command.extend(["--gpus", "device=" + gpu_uuid])
    for volume, target, readonly in mounts:
        mount = "type=volume,src=" + volume + ",dst=" + target
        if readonly:
            mount += ",readonly"
        command.extend(["--mount", mount])
    if entrypoint is not None:
        command.extend(["--entrypoint", entrypoint])
    runner(command + [image] + arguments)


def _start(runner, name: str) -> subprocess.CompletedProcess[str]:
    return runner(["docker", "start", "-a", name], check=False)


def _parse_diagnostic(stdout: str) -> dict[str, object]:
    """Parse exactly one small, public diagnostic result object."""
    if not isinstance(stdout, str) or len(stdout.encode()) > MAX_DIAGNOSTIC_JSON_BYTES:
        raise _DiagnosticFailure("diagnostic_result_malformed")
    if not stdout.strip():
        raise _DiagnosticFailure("diagnostic_result_missing")
    decoder = json.JSONDecoder()
    text = stdout.lstrip()
    try:
        value, end = decoder.raw_decode(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise _DiagnosticFailure("diagnostic_result_malformed") from exc
    if text[end:].strip() or not isinstance(value, dict):
        raise _DiagnosticFailure("diagnostic_result_multiple")
    if not set(value) <= _DIAGNOSTIC_ALLOWED:
        raise _DiagnosticFailure("diagnostic_result_schema")
    if not _DIAGNOSTIC_IDENTITY | {"status"} <= set(value):
        raise _DiagnosticFailure("diagnostic_result_schema")
    if value.get("status") == "incomplete":
        # Failure results intentionally have no native evidence.  Requiring the
        # success-only fields here turns a classified provider failure into a
        # misleading schema failure.
        if not (_DIAGNOSTIC_IDENTITY | _DIAGNOSTIC_FAILURE) <= set(value) \
                or not set(value) <= (_DIAGNOSTIC_IDENTITY | _DIAGNOSTIC_INCOMPLETE | _DIAGNOSTIC_LIFECYCLE):
            raise _DiagnosticFailure("diagnostic_result_schema")
        if (value.get("model") != "SmolLM" or value.get("provider_category") != "ollama"
                or value.get("provider_class") != "SmolLMProvider"
                or value.get("stage") not in {"p2_wave", "cleanup"}
                or value.get("concurrency") != 2 or value.get("wave") != 1
                or value.get("cleanup") != "proved"):
            raise _DiagnosticFailure("diagnostic_result_schema")
        failure_kind = value.get("failure_kind")
        failure_code = value.get("failure_code")
        if (not isinstance(failure_kind, str) or len(failure_kind) > 64
                or failure_kind not in SAFE_INNER_FAILURES
                or not isinstance(failure_code, str) or len(failure_code) > 64
                or failure_code not in SAFE_INNER_FAILURES):
            raise _DiagnosticFailure("diagnostic_result_schema")
        if "failure_stage_detail" in value and value["failure_stage_detail"] not in SAFE_FAILURE_STAGE_DETAILS:
            raise _DiagnosticFailure("diagnostic_result_schema")
        if "lifecycle_phase" in value and value["lifecycle_phase"] not in {"validate", "load", "ready", "cleanup"}:
            raise _DiagnosticFailure("diagnostic_result_schema")
        if "lifecycle_subreason" in value and value["lifecycle_subreason"] not in {
                "profile_shape_identity", "gpu_identity_proof", "artifact_verification",
                "gpu_memory_proof", "ollama_create", "readiness_request",
                "endpoint_residency", "gpu_residency", "fallback_ownership_memory",
                "timeout", "cleanup_verification"}:
            raise _DiagnosticFailure("diagnostic_result_schema")
        message = value.get("failure_message")
        if (not isinstance(message, str) or not 1 <= len(message) <= 160
                or any(ord(char) < 32 or ord(char) == 127 for char in message)
                or "/" in message or "\\" in message):
            raise _DiagnosticFailure("diagnostic_result_schema")
        return value
    if set(value) != _DIAGNOSTIC_IDENTITY | _DIAGNOSTIC_FAILURE | _DIAGNOSTIC_COMPLETE_EVIDENCE:
        raise _DiagnosticFailure("diagnostic_result_schema")
    if (value.get("status") != "complete" or value.get("model") != "SmolLM"
            or value.get("provider_category") != "ollama"
            or value.get("provider_class") != "SmolLMProvider"
            or value.get("stage") != "p2_wave"):
        raise _DiagnosticFailure("diagnostic_result_schema")
    if value.get("native_overlap") is not True or value.get("native_request_correlation") is not True:
        raise _DiagnosticFailure("diagnostic_native_proof_failed")
    if value.get("concurrency") != 2 or value.get("wave") != 1:
        raise _DiagnosticFailure("diagnostic_result_schema")
    if value.get("native_observation_count") != 2 or value.get("native_batch_sizes") != [1, 1]:
        raise _DiagnosticFailure("diagnostic_result_schema")
    if value.get("observation_drops", 0) != 0:
        raise _DiagnosticFailure("diagnostic_native_proof_failed")
    if value.get("cleanup") != "proved":
        raise _DiagnosticFailure("application_cleanup_not_proved")
    if value.get("failure_kind") is not None or value.get("failure_code") is not None \
            or value.get("failure_message") is not None:
        raise _DiagnosticFailure("diagnostic_result_schema")
    for key in ("model", "provider_class", "provider_category", "stage", "status", "cleanup"):
        if not isinstance(value.get(key), str) or len(value[key]) > 64:
            raise _DiagnosticFailure("diagnostic_result_schema")
    return value


def _capture(runner, name: str) -> str:
    # Intentionally no docker logs: retained output can contain model bodies.
    result = runner(
        ["docker", "container", "inspect", "--format", "{{json .State}}", name],
        check=False,
    )
    return result.stdout


def _missing_result_code(inspect_stdout: str) -> str:
    """Classify only Docker's bounded state facts; never emit its Error text."""
    try:
        value = json.loads(inspect_stdout)
    except (TypeError, json.JSONDecodeError):
        return "diagnostic_docker_runtime_failed"
    required = {"Status", "ExitCode", "OOMKilled", "Error"}
    # Docker's State object has additional version-dependent fields. Consume
    # only this fixed bounded subset and never expose the remaining values.
    if not isinstance(value, dict) or not required <= set(value):
        return "diagnostic_docker_runtime_failed"
    status, exit_code, oom_killed, error = (value["Status"], value["ExitCode"],
                                             value["OOMKilled"], value["Error"])
    if (not isinstance(status, str) or status not in {"created", "running", "restarting", "paused", "exited", "dead"}
            or type(exit_code) is not int or exit_code < 0 or type(oom_killed) is not bool
            or not isinstance(error, str)):
        return "diagnostic_docker_runtime_failed"
    if oom_killed:
        return "diagnostic_oom_killed"
    if error:
        return "diagnostic_docker_runtime_failed"
    if exit_code == 0:
        return "diagnostic_result_missing_after_success"
    return "diagnostic_process_exit_" + status


def _read_result_file(runner, owned: _Owned, image: str) -> str:
    """Read only the fixed result transport from an owned read-only mount."""
    _create_container(runner, owned.result_reader, image,
                      [(owned.runtime_volume, "/var/lib/llm-measurement", True)],
                      ["/var/lib/llm-measurement/diagnostic-result.json"], entrypoint="/bin/cat")
    result = _start(runner, owned.result_reader)
    if result.returncode:
        raise _DiagnosticFailure("result_file_missing")
    if len(result.stdout.encode()) > MAX_DIAGNOSTIC_JSON_BYTES:
        raise _DiagnosticFailure("result_file_malformed")
    return result.stdout


def _preflight(runner, owned: _Owned, args) -> bool:
    # Reuse the exclusively owned probe name for the immediate second check.
    # The old stopped probe is removed, never a foreign container.
    runner(["docker", "rm", "-f", owned.preflight], check=False)
    _create_container(runner, owned.preflight, args.image, [],
                      ["--query-compute-apps=pid", "--format=csv,noheader"],
                      gpu_uuid=args.gpu_uuid, entrypoint="nvidia-smi")
    result = _start(runner, owned.preflight)
    _capture(runner, owned.preflight)
    if result.returncode:
        raise RuntimeError("GPU preflight failed")
    return bool(result.stdout.strip())


def _reject_symlink_components(path: Path, name: str) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        if part in ("", "."):
            continue
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{name} must be a non-symlink path")


def _copy_volume(runner, container: str, volume: str, image: str,
                 source: Path, destination: str, *, contents: bool = False) -> None:
    _create_container(runner, container, image, [(volume, "/transfer", False)], [],
                      entrypoint="/bin/true")
    # Path normalizes a trailing "/." away, but Docker needs that suffix to copy
    # directory contents rather than nesting the source directory in the volume.
    # docker cp archives the source and preserves its relative current symlink.
    copy_source = str(source) + "/." if contents else str(source)
    runner(["docker", "cp", copy_source, container + ":" + destination])


def _cleanup(runner, owned: _Owned) -> bool:
    ok = True

    def attempt(command: list[str]) -> subprocess.CompletedProcess[str] | None:
        nonlocal ok
        try:
            return runner(command, check=False)
        except (OSError, RuntimeError, subprocess.SubprocessError):
            ok = False
            return None

    for name in (owned.result_reader, owned.diagnostic, owned.verifier, owned.request_transfer,
                 owned.bundle_transfer, owned.preflight):
        attempt(["docker", "rm", "-f", name])
        inspected = attempt(["docker", "container", "inspect", name])
        if inspected is None or inspected.returncode == 0:
            ok = False
    for volume in (owned.request_volume, owned.runtime_volume, owned.bundle_volume):
        attempt(["docker", "volume", "rm", volume])
        inspected = attempt(["docker", "volume", "inspect", volume])
        if inspected is None or inspected.returncode == 0:
            ok = False
    return ok


def _public_incomplete(code: str, *, application_cleanup: str = "not_proved",
                       inner: dict[str, object] | None = None) -> dict[str, object]:
    """Create the fixed outer contract without copying untrusted inner text."""
    result: dict[str, object] = {
        "status": "incomplete", "failure_code": code,
        "application_cleanup": application_cleanup,
    }
    if inner is not None:
        kind, inner_code = inner.get("failure_kind"), inner.get("failure_code")
        if isinstance(kind, str) and kind in SAFE_INNER_FAILURES:
            result["inner_failure_kind"] = kind
        if isinstance(inner_code, str) and inner_code in SAFE_INNER_FAILURES:
            result["inner_failure_code"] = inner_code
        for key, allowed in (("lifecycle_phase", {"validate", "load", "ready", "cleanup"}),
                             ("lifecycle_subreason", {
                                 "profile_shape_identity", "gpu_identity_proof", "artifact_verification",
                                 "gpu_memory_proof", "ollama_create", "readiness_request",
                                 "endpoint_residency", "gpu_residency", "fallback_ownership_memory",
                                 "timeout", "cleanup_verification"})):
            value = inner.get(key)
            if value in allowed:
                result[key] = value
        detail = inner.get("failure_stage_detail")
        if isinstance(detail, str) and detail in SAFE_FAILURE_STAGE_DETAILS:
            result["inner_failure_stage_detail"] = detail
    return result


def orchestrate(args, *, runner: Callable[..., subprocess.CompletedProcess[str]] = _run,
                owned: _Owned | None = None) -> int:
    """Execute the bounded operator workflow and return the diagnostic status."""
    owned = _Owned.create() if owned is None else owned
    diagnostic_status = 2
    diagnostic_result: dict[str, object] | None = None
    cleanup_ok = True
    failure_code = "preflight_failed"
    try:
        failure_code = "preflight_failed"
        if _preflight(runner, owned, args):
            # This is only a point-in-time check; the inner ownership proof is
            # authoritative and must still reject foreign work.
            failure_code = "foreign_compute_present"
            return 2
        failure_code = "volume_setup_failed"
        for volume in (owned.bundle_volume, owned.runtime_volume, owned.request_volume):
            runner(["docker", "volume", "create", volume])
        failure_code = "bundle_transfer_failed"
        _copy_volume(runner, owned.bundle_transfer, owned.bundle_volume, args.image,
                     args.bundle, "/transfer", contents=True)
        failure_code = "request_transfer_failed"
        _copy_volume(runner, owned.request_transfer, owned.request_volume, args.image,
                     args.request, "/transfer/request.json")
        failure_code = "bundle_verify_failed"
        _create_container(runner, owned.verifier, args.image,
                          [(owned.bundle_volume, "/opt/measurement", True)],
                          ["/opt/llm/prepare_measurement.py", "verify", "--output", "/opt/measurement"])
        verified = _start(runner, owned.verifier)
        _capture(runner, owned.verifier)
        if verified.returncode:
            return 2
        failure_code = "diagnostic_start_failed"
        _create_container(runner, owned.diagnostic, args.image,
                          [(owned.bundle_volume, "/opt/measurement", True),
                           (owned.runtime_volume, "/var/lib/llm-measurement", False),
                           (owned.request_volume, "/opt/diagnostic-request", True)],
                          ["/opt/llm/measure_profiles.py", "--config", "/opt/measurement/config.json",
                            "--diagnostic-smollm-p2", "--request",
                            "/opt/diagnostic-request/request.json", "--selector",
                            "smollm:context512", "--ollama-version", args.ollama_version,
                            "--result-file", "/var/lib/llm-measurement/diagnostic-result.json"],
                          gpu_uuid=args.gpu_uuid, pid_host=True)
        # Recheck immediately before start.  This point-in-time preflight does
        # not replace the diagnostic's ownership proof.
        failure_code = "second_preflight_failed"
        if _preflight(runner, owned, args):
            failure_code = "foreign_compute_started"
            return 2
        failure_code = "diagnostic_start_failed"
        started = _start(runner, owned.diagnostic)
        diagnostic_status = started.returncode
        inspect_state = _capture(runner, owned.diagnostic)
        # Docker Desktop may relay attached container stdout through the client
        # stderr stream. Parse exactly one bounded stream, never combine them.
        diagnostic_output = started.stdout if started.stdout.strip() else started.stderr
        if not diagnostic_output.strip():
            try:
                diagnostic_output = _read_result_file(runner, owned, args.image)
            except _DiagnosticFailure:
                raise
        diagnostic_result = _parse_diagnostic(diagnostic_output)
        if diagnostic_status != 0:
            failure_code = "diagnostic_incomplete"
        elif diagnostic_result.get("status") == "incomplete":
            failure_code = "diagnostic_incomplete"
        else:
            failure_code = "complete"
    except _DiagnosticFailure as exc:
        failure_code = exc.code
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        pass
    finally:
        cleanup_ok = _cleanup(runner, owned)
        if failure_code == "complete" and cleanup_ok and diagnostic_result is not None:
            orchestrate.last_result = diagnostic_result
        else:
            application_cleanup = "not_proved"
            if diagnostic_result is not None and diagnostic_result.get("cleanup") == "proved":
                application_cleanup = "proved"
            orchestrate.last_result = _public_incomplete(
                "docker_cleanup_failed" if not cleanup_ok else failure_code,
                application_cleanup=application_cleanup, inner=diagnostic_result)
            orchestrate.last_result["docker_cleanup"] = "proved" if cleanup_ok else "unproved"
    return 0 if failure_code == "complete" and cleanup_ok else 2


def _arguments(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--ollama-version", required=True)
    args = parser.parse_args(argv)
    if not args.gpu_uuid.startswith("GPU-"):
        parser.error("--gpu-uuid must be an exact GPU UUID")
    _reject_symlink_components(args.bundle, "bundle")
    _reject_symlink_components(args.request, "request")
    if args.bundle.is_symlink() or not args.bundle.is_dir():
        parser.error("--bundle must be a real directory")
    if args.request.is_symlink() or not args.request.is_file():
        parser.error("--request must be a regular file")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = _arguments(argv)
        status = orchestrate(args)
    except BaseException:
        # Never expose Docker, prompts, paths, or exception text at this
        # boundary.  ArgumentParser's stderr remains its conventional UI.
        orchestrate.last_result = {
            "status": "incomplete", "failure_code": "orchestration_failed",
            "application_cleanup": "not_proved", "docker_cleanup": "unproved",
        }
        status = 2
    # One bounded, sanitized audit object is the only stdout contract.
    print(json.dumps(orchestrate.last_result, sort_keys=True, separators=(",", ":")))
    return 0 if status == 0 and orchestrate.last_result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

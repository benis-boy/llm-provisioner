"""Run the complete authoritative measurement matrix in owned Docker state."""
from __future__ import annotations
import argparse, json, os, subprocess, uuid, sys
from pathlib import Path
from services.llm.bootstrap.measurement_matrix import measurement_matrix
from services.llm.resource_manager.profiles import ProfileStore
from services.llm.provisioning.measurement import (
    MEASUREMENT_FAILURE_CODES, MEASUREMENT_FAILURE_DETAILS,
)
from services.llm.provisioning.measured_profiles import PERSISTENCE_VALIDATION_DETAILS
try:
    from tools.compatibility.debug_trace import configure as configure_trace
    from tools.compatibility.debug_trace import record as trace
    from tools.compatibility.debug_trace import sanitize_jsonl
except ModuleNotFoundError:  # Direct script execution sets sys.path to this directory.
    from debug_trace import configure as configure_trace
    from debug_trace import record as trace
    from debug_trace import sanitize_jsonl

TIMEOUT = 10_980
MAX_RESULT = 8192
FAILURE_CODES = frozenset({"argument_invalid", "output_reserved", "volume_setup_failed",
    "bundle_transfer_failed", "bundle_verify_failed", "preflight_failed",
    "measurement_failed", "result_missing", "result_malformed", "export_failed",
    "export_audit_failed", "cleanup_failed", "lock_ownership_changed", "timeout"})
# Keep the outer runner's accepted vocabulary explicit for classified failures
# produced by the measurement image.
MEASUREMENT_FAILURE_CODES = MEASUREMENT_FAILURE_CODES | {
    "persistence_validation_failed",
    "maximum_witness_failed",
}
MEASUREMENT_FAILURE_DETAILS = MEASUREMENT_FAILURE_DETAILS | PERSISTENCE_VALIDATION_DETAILS | frozenset({
    "smollm_token_count", "coedit_token_count", "coedit_configured_maximum",
    "coedit_payload_fingerprint", "gector_token_count", "gector_configured_maximum",
    "gector_payload_fingerprint",
})
PROVIDER_FAILURE_CODES = frozenset({
    "ollama_http_status", "ollama_transport", "ollama_json_response",
    "ollama_response_contract", "ollama_telemetry_contract",
    "smollm_input_decode", "smollm_input_validation",
    "smollm_observation_contract", "smollm_internal",
    "malformed_provider_response",
    "wave_timeout", "provider_process_exit", "oom",
})


class OperatorFailure(RuntimeError):
    def __init__(self, code, stage, *, db_retained=False, inner=None):
        self.code, self.stage, self.db_retained = code, stage, db_retained
        self.inner = inner
        self.docker_cleanup = "unproved"
        super().__init__(code)


def _failure(code, stage, *, db_retained=False, inner=None, docker_cleanup="unproved"):
    result = {"status": "incomplete", "failure_code": code, "stage": stage,
              "db_retained": db_retained, "docker_cleanup": docker_cleanup
              if docker_cleanup in {"proved", "unproved"} else "unproved"}
    if isinstance(inner, dict):
        for source, target in (("stage", "inner_stage"),
                               ("failure_kind", "inner_failure_kind"),
                               ("exception_type", "inner_exception_type"),
                               ("model", "inner_model"),
                                ("measurement_failure_code", "inner_measurement_failure_code")):
            value = inner.get(source)
            if source == "measurement_failure_code" and value not in MEASUREMENT_FAILURE_CODES:
                continue
            if isinstance(value, str) and value and len(value) <= 80 \
                    and all(character.isalnum() or character in "_-" for character in value):
                 result[target] = value
        # Relay only the provider's closed category vocabulary.  Do not copy
        # generic failure messages or exception text into the matrix envelope.
        provider_code = inner.get("failure_code")
        if provider_code in PROVIDER_FAILURE_CODES:
            result["inner_failure_code"] = provider_code
        detail = inner.get("measurement_failure_detail")
        if detail in MEASUREMENT_FAILURE_DETAILS:
            result["inner_measurement_failure_detail"] = detail
            if detail in PROVIDER_FAILURE_CODES:
                result["inner_failure_code"] = detail
        telemetry = inner.get("measurement_telemetry")
        if isinstance(telemetry, dict) and set(telemetry) == {
                "total_vram_bytes", "baseline_pre_used_bytes",
                "peak_incremental_request_bytes", "derived_ceiling"} and all(
                    value is None or type(value) is int for value in telemetry.values()):
            result["inner_measurement_telemetry"] = telemetry
    return result


def _read_runtime_result(container, image, volume):
    _create(container, image, [(volume, "/runtime", True)],
            ["/runtime/measurement-result.json"], entrypoint="/bin/cat")
    reader = run(["docker", "start", "-a", container], check=False)
    if reader.returncode or not reader.stdout or len(reader.stdout.encode()) > MAX_RESULT:
        return None
    try:
        value = json.loads(reader.stdout)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None

def _read_runtime_trace(container, image, volume):
    """Retrieve the bounded inner trace before runtime volume cleanup."""
    try:
        _create(container, image, [(volume, "/runtime", True)],
                ["/runtime/debug-trace.jsonl"], entrypoint="/bin/cat")
        reader = run(["docker", "start", "-a", container], check=False)
        return sanitize_jsonl(reader.stdout) if reader.returncode == 0 else ""
    except Exception:
        return ""

def _publish_sidecar(path: Path, text: str) -> None:
    """Best-effort exclusive sidecar publication; never follows/clobbers."""
    if not text or path.exists() or path.is_symlink():
        return
    data = sanitize_jsonl(text).encode("utf-8")
    if not data:
        return
    temporary = path.with_name("." + path.name + ".tmp-" + uuid.uuid4().hex)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(temporary, flags, 0o600)
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short sidecar write")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(temporary, path)
        os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except (FileExistsError, OSError):
        try:
            temporary.unlink()
        except OSError:
            pass
        return

def run(command, *, timeout=60, check=True):
    value = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if check and value.returncode:
        raise RuntimeError("docker command failed")
    return value

def _create(name, image, mounts=(), args=(), gpu=None, pid=False, entrypoint=None):
    command = ["docker", "create", "--name", name, "--network", "none"]
    if pid: command.append("--pid=host")
    if gpu: command += ["--gpus", "device=" + gpu]
    for volume, target, readonly in mounts:
        spec = f"type=volume,src={volume},dst={target}" + (",readonly" if readonly else "")
        command += ["--mount", spec]
    if entrypoint: command += ["--entrypoint", entrypoint]
    run(command + [image] + list(args))

def _copy(name, image, volume, source, destination, contents=False):
    _create(name, image, [(volume, "/transfer", False)], entrypoint="/bin/true")
    run(["docker", "cp", str(source) + ("/." if contents else ""), name + ":" + destination])

def _result(text):
    if not text or len(text.encode()) > MAX_RESULT: raise ValueError("result missing or oversized")
    value = json.loads(text)
    if not isinstance(value, dict) or value.get("status") != "complete": raise ValueError("result incomplete")
    matrix = value.get("matrix")
    profiles = value.get("profiles")
    expected_matrix = [[model.value, selector] for model, selector in measurement_matrix()]
    if matrix != expected_matrix or not isinstance(profiles, list):
        raise ValueError("result matrix identity mismatch")
    expected_models = [model for model, _ in expected_matrix]
    if len(profiles) != 3 or any(not isinstance(item, dict) for item in profiles):
        raise ValueError("result profile summaries are incomplete")
    if [item.get("model") for item in profiles] != expected_models:
        raise ValueError("result model identities mismatch")
    identities = [item.get("profile_identity") for item in profiles]
    if any(not isinstance(identity, str) or not identity for identity in identities) or len(set(identities)) != 3:
        raise ValueError("result profile identities are not exact")
    return value


def _audit_export(path: Path, result: dict) -> None:
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm")):
        raise ValueError("export contains WAL sidecar state")
    expected = {item["profile_identity"] for item in result["profiles"]}
    with ProfileStore.open_readonly(path) as store:
        profiles = store.validate_all_measured(expected)
    actual = {profile.profile_identity for profile in profiles}
    if actual != expected or len(profiles) != 3:
        raise ValueError("export profile audit does not match result")
    models = {profile.model_id.value for profile in profiles}
    if models != {item["model"] for item in result["profiles"]}:
        raise ValueError("export model audit does not match result")

def orchestrate(args):
    debug = bool(getattr(args, "debug", False))
    configure_trace(debug)
    prefix = "llm-measurement-" + uuid.uuid4().hex
    volumes = [prefix + suffix for suffix in ("-bundle", "-runtime", "-result", "-export")]
    containers = [prefix + suffix for suffix in ("-bundle-copy", "-verify", "-measure", "-result", "-trace", "-export")]
    destination = Path(args.output).absolute()
    if destination.exists():
        raise OperatorFailure("output_reserved", "reserve_output")
    lock = destination.with_name("." + destination.name + ".measurement.lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    except FileExistsError as exc:
        raise OperatorFailure("output_reserved", "reserve_output") from exc
    token = os.urandom(32)
    os.write(lock_fd, token); os.fsync(lock_fd)
    promoted = False
    primary = None
    inner_trace = ""
    trace_retrieved = False
    cleanup_error = None
    temporary = None
    try:
        trace("outer", "reservation", "enter", matrix_index=0)
        for volume in volumes: run(["docker", "volume", "create", volume])
        _copy(containers[0], args.image, volumes[0], args.bundle, "/transfer", True)
        _create(containers[1], args.image, [(volumes[0], "/opt/measurement", True)],
                ["/opt/llm/prepare_measurement.py", "verify", "--output", "/opt/measurement"])
        if run(["docker", "start", "-a", containers[1]], check=False).returncode: raise OperatorFailure("bundle_verify_failed", "bundle_verify")
        _create(containers[2], args.image,
                [(volumes[0], "/opt/measurement", True), (volumes[1], "/var/lib/llm-measurement", False)],
                ["/opt/llm/measure_profiles.py", "--bundle", "/opt/measurement",
                 "--db", "/var/lib/llm-measurement/profiles.sqlite",
                  "--ollama-version", args.ollama_version, "--ceiling", str(args.ceiling),
                  "--result-file", "/var/lib/llm-measurement/measurement-result.json"]
                 + (["--debug", "--trace-file", "/var/lib/llm-measurement/debug-trace.jsonl"] if debug else []),
                gpu=args.gpu_uuid, pid=True)
        # Two checks fence foreign work immediately before compute starts.
        for _ in range(2):
            probe = run(["docker", "run", "--rm", "--network", "none", "--gpus", "device=" + args.gpu_uuid,
                         "--entrypoint", "nvidia-smi", args.image,
                         "--query-compute-apps=pid", "--format=csv,noheader"])
            if probe.returncode or probe.stdout.strip(): raise OperatorFailure("preflight_failed", "preflight")
        try:
            result = run(["docker", "start", "-a", containers[2]], timeout=TIMEOUT, check=False)
        except subprocess.TimeoutExpired as exc:
            raise OperatorFailure("timeout", "measurement") from exc
        inner_result = _read_runtime_result(containers[3], args.image, volumes[1])
        if debug:
            inner_trace = _read_runtime_trace(containers[4], args.image, volumes[1])
            trace_retrieved = True
        if result.returncode:
            trace("outer", "container_exit", "failure", return_code=result.returncode)
            raise OperatorFailure("measurement_failed", "measurement", inner=inner_result)
        if inner_result is None:
            raise OperatorFailure("result_missing", "result_read")
        try:
            parsed = _result(json.dumps(inner_result, separators=(",", ":")))
        except (ValueError, json.JSONDecodeError) as exc:
            raise OperatorFailure("result_malformed", "result_read") from exc
        if destination.exists():
            raise FileExistsError("output already exists")
        _create(containers[5], args.image, [(volumes[1], "/runtime", True), (volumes[3], "/export", False)],
                ["/runtime/profiles.sqlite", "/export/profiles.sqlite"], entrypoint="/bin/cp")
        exporter = run(["docker", "start", "-a", containers[5]], check=False)
        if exporter.returncode:
            raise OperatorFailure("export_failed", "export")
        temporary = destination.with_name("." + destination.name + ".tmp")
        if os.path.lexists(temporary):
            raise OperatorFailure("export_failed", "export")
        run(["docker", "cp", containers[5] + ":/export/profiles.sqlite", str(temporary)])
        if temporary.is_symlink() or not temporary.is_file() or temporary.stat().st_size == 0:
            raise OperatorFailure("export_failed", "export")
        _audit_export(temporary, parsed)
        os.chmod(temporary, 0o444)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, destination); temporary.unlink()
        if (destination.stat().st_mode & 0o777) != 0o444: raise RuntimeError("export is not readonly")
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        promoted = True
        trace("outer", "promotion", "success", matrix_index=0)
    except OperatorFailure as exc:
        primary = exc
    except subprocess.TimeoutExpired as exc:
        primary = OperatorFailure("timeout", "operator")
    except Exception as exc:
        primary = OperatorFailure("export_audit_failed", "operator", db_retained=promoted)
    finally:
        trace("outer", "cleanup", "enter")
        if debug and not trace_retrieved:
            try:
                inner_trace = _read_runtime_trace(containers[4], args.image, volumes[1])
            except Exception:
                inner_trace = ""
            trace_retrieved = True
        if debug and inner_trace:
            try:
                if not promoted:
                    _publish_sidecar(Path(str(destination) + ".debug.jsonl"), inner_trace)
                sys.stderr.write(inner_trace)
                sys.stderr.flush()
            except Exception:
                pass
        cleanup_failures = []
        try:
            if temporary is not None and temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
        except Exception as exc:
            cleanup_failures.append(exc)
        for container in containers:
            try:
                result = run(["docker", "rm", "-f", container], check=False)
                if result.returncode != 0:
                    cleanup_failures.append(RuntimeError("owned container cleanup failed"))
            except Exception as exc:
                cleanup_failures.append(exc)
        for volume in volumes:
            try:
                result = run(["docker", "volume", "rm", volume], check=False)
                if result.returncode != 0:
                    cleanup_failures.append(RuntimeError("owned volume cleanup failed"))
            except Exception as exc:
                cleanup_failures.append(exc)
        if cleanup_failures:
            cleanup_error = cleanup_failures[0]
        lock_released = False
        try:
            lock_stat = lock.stat()
            owner_stat = os.fstat(lock_fd)
            if (lock_stat.st_dev, lock_stat.st_ino) != (owner_stat.st_dev, owner_stat.st_ino) or lock.read_bytes() != token:
                raise OperatorFailure("lock_ownership_changed", "release_lock", db_retained=promoted)
            lock.unlink()
            lock_released = True
        except OperatorFailure as exc:
            cleanup_error = cleanup_error or exc
        except Exception as exc:
            cleanup_error = cleanup_error or exc
        try:
            os.close(lock_fd)
        except Exception as exc:
            cleanup_error = cleanup_error or exc
            lock_released = False
    if cleanup_error:
        raise OperatorFailure("cleanup_failed", "cleanup", db_retained=promoted) from cleanup_error
    if primary:
        primary.db_retained = promoted
        if lock_released:
            primary.docker_cleanup = "proved"
        raise primary
    return {**parsed, "status": "complete", "db_retained": True, "docker_cleanup": "proved"}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True); parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True); parser.add_argument("--ollama-version", required=True)
    parser.add_argument("--ceiling", type=int, choices=range(1, 33), default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug", action="store_true", help="emit bounded structured tracing to stderr")
    try:
        try:
            args = parser.parse_args(argv)
        except SystemExit as exc:
            if exc.code == 0:
                raise
            print(json.dumps(_failure("argument_invalid", "arguments"),
                             sort_keys=True, separators=(",", ":")))
            return 2
        if not args.gpu_uuid.startswith("GPU-") or args.bundle.is_symlink() or not args.bundle.is_dir():
            raise OperatorFailure("argument_invalid", "arguments")
        print(json.dumps(orchestrate(args), sort_keys=True, separators=(",", ":")))
        return 0
    except OperatorFailure as exc:
        print(json.dumps(_failure(exc.code if exc.code in FAILURE_CODES else "export_failed",
                                  exc.stage, db_retained=exc.db_retained, inner=exc.inner,
                                  docker_cleanup=getattr(exc, "docker_cleanup", "unproved")),
                         sort_keys=True, separators=(",", ":")))
        return 2
    except subprocess.TimeoutExpired:
        print(json.dumps(_failure("timeout", "operator"), sort_keys=True, separators=(",", ":")))
        return 2
    except Exception:
        print(json.dumps(_failure("export_failed", "operator"), sort_keys=True, separators=(",", ":")))
        return 2
if __name__ == "__main__": raise SystemExit(main())

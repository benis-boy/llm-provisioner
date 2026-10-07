"""Run the complete authoritative measurement matrix in owned Docker state."""
from __future__ import annotations
import argparse, json, os, subprocess, uuid, sys, stat
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
    "runtime_generated_request_failed",
}
MEASUREMENT_FAILURE_DETAILS = MEASUREMENT_FAILURE_DETAILS | PERSISTENCE_VALIDATION_DETAILS | frozenset({
    "smollm_token_count", "coedit_token_count", "coedit_configured_maximum",
    "coedit_payload_fingerprint", "gector_token_count", "gector_configured_maximum",
    "gector_payload_fingerprint", "coedit_generated_schema", "coedit_generated_count",
    "coedit_generated_max", "coedit_generated_fingerprint", "coedit_generated_bounds",
})
PROVIDER_FAILURE_CODES = frozenset({
    "ollama_http_status", "ollama_transport", "ollama_json_response",
    "ollama_response_contract", "ollama_telemetry_contract",
    "smollm_input_decode", "smollm_input_validation",
    "smollm_observation_contract", "smollm_internal",
    "malformed_provider_response",
    "wave_timeout", "provider_process_exit", "oom",
})
OWNER_LABEL = "com.llm-provider.measurement-owner"


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


def _read_runtime_result(container, image, volume, *, resources=None):
    _create(container, image, [(volume, "/runtime", True)],
            ["/runtime/measurement-result.json"], entrypoint="/bin/cat", resources=resources)
    reader = run(["docker", "start", "-a", resources.container_reference(container)
                  if resources is not None else container], check=False)
    if reader.returncode or not reader.stdout or len(reader.stdout.encode()) > MAX_RESULT:
        return None
    try:
        value = json.loads(reader.stdout)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None

def _read_runtime_trace(container, image, volume, *, resources=None):
    """Retrieve the bounded inner trace before runtime volume cleanup."""
    try:
        _create(container, image, [(volume, "/runtime", True)],
                ["/runtime/debug-trace.jsonl"], entrypoint="/bin/cat", resources=resources)
        reader = run(["docker", "start", "-a", resources.container_reference(container)
                      if resources is not None else container], check=False)
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

class _OwnedDockerResources:
    """Require positive ownership and fence use/cleanup against resource replacement."""

    def __init__(self, owner):
        self.owner = owner
        self.containers: dict[str, str | None] = {}
        self.volumes: dict[str, tuple[str, str] | None] = {}
        self.mounts: dict[str, tuple[str, ...]] = {}

    @staticmethod
    def _valid_container_id(value):
        return isinstance(value, str) and len(value) == 64 and all(
            character in "0123456789abcdef" for character in value)

    def create_container(self, name, command):
        self.containers.setdefault(name, None)  # A failed CLI may follow daemon-side creation.
        result = run(command)
        identity = result.stdout.strip()
        if not self._valid_container_id(identity):
            raise RuntimeError("container creation did not return an immutable ID")
        self.containers[name] = identity
        self.container_reference(name)
        return result

    def create_volume(self, name):
        self.volumes.setdefault(name, None)
        result = run(["docker", "volume", "create", "--label", f"{OWNER_LABEL}={self.owner}", name])
        ownership, identity = self._inspect("volume", name)
        if ownership != "owned":
            raise RuntimeError("volume creation ownership is unproved")
        self.volumes[name] = identity
        return result

    def _inspect(self, kind, name):
        if kind == "container":
            command = ["docker", "inspect", "--type", "container", "--format",
                       "{{json .}}", name]
        else:
            command = ["docker", "volume", "inspect", "--format", "{{json .}}", name]
        result = run(command, check=False)
        if result.returncode:
            message = (result.stderr or "").lower().strip()
            absent_messages = ((f"no such container: {name}", f"no such object: {name}")
                               if kind == "container" else
                               (f"get {name}: no such volume", f"no such volume: {name}"))
            if any(message.endswith(ending.lower()) for ending in absent_messages):
                return "absent", None
            return "unknown", None
        try:
            value = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError):
            return "unknown", None
        if not isinstance(value, dict):
            return "unknown", None
        if kind == "container":
            config = value.get("Config")
            labels = config.get("Labels") if isinstance(config, dict) else None
            identity = value.get("Id")
            valid_identity = self._valid_container_id(identity) and (
                not self._valid_container_id(name) or name == identity)
        else:
            labels = value.get("Labels")
            identity = (value.get("Name"), value.get("CreatedAt"))
            valid_identity = identity[0] == name and isinstance(identity[1], str) and bool(identity[1])
        if labels is None:
            return "foreign", None
        if not isinstance(labels, dict):
            return "unknown", None
        if labels.get(OWNER_LABEL) != self.owner:
            return "foreign", None
        return ("owned", identity) if valid_identity else ("unknown", None)

    def require_volume(self, name):
        expected = self.volumes.get(name)
        ownership, identity = self._inspect("volume", name)
        if expected is None or ownership != "owned" or identity != expected:
            raise RuntimeError("mounted volume ownership changed or is unproved")

    def container_reference(self, name):
        expected = self.containers.get(name)
        if expected is None:
            raise RuntimeError("container identity is unproved")
        ownership, identity = self._inspect("container", expected)
        if ownership != "owned" or identity != expected:
            raise RuntimeError("container ownership changed or is unproved")
        for volume in self.mounts.get(name, ()):
            self.require_volume(volume)
        return expected

    def _cleanup_one(self, kind, name, expected):
        reference = expected if kind == "container" and expected is not None else name
        try:
            ownership, identity = self._inspect(kind, reference)
        except Exception:
            ownership, identity = "unknown", None
        if ownership == "absent" or (ownership == "foreign" and expected is None):
            return None
        if ownership != "owned" or (expected is not None and identity != expected):
            return RuntimeError("Docker cleanup ownership changed or is unproved")
        # Even uncertain daemon-side container creation is removed only by its
        # positively inspected immutable ID, never by its reusable name.
        reference = identity if kind == "container" else name
        command = (["docker", "rm", "-f", reference] if kind == "container"
                   else ["docker", "volume", "rm", name])
        try:
            result = run(command, check=False)
        except Exception as exc:
            return exc
        if result.returncode == 0:
            return None
        try:
            ownership, _ = self._inspect(kind, reference)
        except Exception:
            ownership = "unknown"
        if ownership == "absent":
            return None
        return RuntimeError(f"owned {kind} cleanup failed")

    def cleanup(self):
        failures = []
        for kind, resources in (("container", self.containers), ("volume", self.volumes)):
            for name, expected in resources.items():
                try:
                    failure = self._cleanup_one(kind, name, expected)
                except BaseException as exc:
                    # Retain interruptions, but still attempt every remaining
                    # owned resource. The caller must re-raise, never commit.
                    failure = exc
                if failure is not None:
                    failures.append(failure)
        return failures


def _create(name, image, mounts=(), args=(), gpu=None, pid=False, entrypoint=None, resources=None):
    if resources is not None:
        for volume, _, _ in mounts:
            resources.require_volume(volume)
        resources.mounts[name] = tuple(volume for volume, _, _ in mounts)
    command = ["docker", "create", "--name", name, "--network", "none"]
    if resources is not None:
        command += ["--label", f"{OWNER_LABEL}={resources.owner}"]
    if pid: command.append("--pid=host")
    if gpu: command += ["--gpus", "device=" + gpu]
    for volume, target, readonly in mounts:
        spec = f"type=volume,src={volume},dst={target}" + (",readonly" if readonly else "")
        command += ["--mount", spec]
    if entrypoint: command += ["--entrypoint", entrypoint]
    command += [image] + list(args)
    if resources is not None:
        resources.create_container(name, command)
    else:
        run(command)

def _copy(name, image, volume, source, destination, contents=False, *, resources=None):
    _create(name, image, [(volume, "/transfer", False)], entrypoint="/bin/true", resources=resources)
    reference = resources.container_reference(name) if resources is not None else name
    run(["docker", "cp", str(source) + ("/." if contents else ""), reference + ":" + destination])

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
    if any(os.path.lexists(Path(str(path) + suffix)) for suffix in ("-wal", "-shm")):
        raise ValueError("export contains WAL sidecar state")
    expected = {item["profile_identity"] for item in result["profiles"]}
    readers = []
    class ExportAuditStore(ProfileStore):
        def __new__(cls):
            reader = super().__new__(cls)
            readers.append(reader)
            return reader
    pending = None
    try:
        store = ExportAuditStore.open_readonly(path)
        profiles = store.validate_all_measured(expected)
    except BaseException as exc:
        pending = exc
        raise
    finally:
        # Retain the instance during readonly construction too: an interruption
        # in schema inspection must close the SQLite connection before the
        # private-stage guard captures/removes its generated sidecars.
        for reader in readers:
            if hasattr(reader, "_db"):
                try:
                    reader.close()
                except BaseException as exc:
                    if pending is None or (isinstance(pending, Exception) and not isinstance(exc, Exception)):
                        raise
    actual = {profile.profile_identity for profile in profiles}
    if actual != expected or len(profiles) != 3:
        raise ValueError("export profile audit does not match result")
    models = {profile.model_id.value for profile in profiles}
    if models != {item["model"] for item in result["profiles"]}:
        raise ValueError("export model audit does not match result")


def _owns_lock(path: Path, fd: int, token: bytes) -> bool:
    """Prove that the named reservation is still the regular file we created."""
    try:
        named = os.lstat(path)
        opened = os.fstat(fd)
        if not stat.S_ISREG(named.st_mode) or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
            return False
        os.lseek(fd, 0, os.SEEK_SET)
        if os.read(fd, len(token) + 1) != token:
            return False
        current = os.lstat(path)
        return (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)
    except OSError:
        return False

def _stage_directory_owned(host):
    named = host["stage"].lstat()
    opened = os.fstat(host["stage_fd"]) if host["stage_fd"] is not None else named
    return stat.S_ISDIR(named.st_mode) and named.st_mode & 0o077 == 0 and \
        (named.st_dev, named.st_ino) == host["stage_owner"] == (opened.st_dev, opened.st_ino)


def _stage_namespace_clean(host):
    return _stage_directory_owned(host) and set(os.listdir(host["stage_fd"])) == {host["temporary"].name}


def _capture_stage_sidecars(host):
    """After the reader closes, prove only its two expected SQLite sidecars."""
    if not _stage_directory_owned(host):
        raise RuntimeError("audit staging directory ownership changed")
    for suffix in ("-wal", "-shm"):
        path = Path(str(host["temporary"]) + suffix)
        if not os.path.lexists(path):
            continue
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        host["stage_file_fds"].append(os.open(path.name, flags, dir_fd=host["stage_fd"]))
        opened = os.fstat(host["stage_file_fds"][-1])
        named = path.lstat()
        if not stat.S_ISREG(opened.st_mode) or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino) \
                or not _stage_directory_owned(host):
            raise RuntimeError("audit sidecar ownership is unproved")
        host["stage_sidecars"][path] = (opened.st_dev, opened.st_ino)


def _cleanup_stage_sidecars(host):
    failures = []
    for path, owner in tuple(host.get("stage_sidecars", {}).items()):
        try:
            if not _stage_directory_owned(host):
                raise RuntimeError("audit staging directory ownership changed")
            current = path.lstat()
            if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != owner:
                raise RuntimeError("audit sidecar ownership changed")
            path.unlink()
            del host["stage_sidecars"][path]
        except FileNotFoundError:
            del host["stage_sidecars"][path]
        except BaseException as exc:
            failures.append(exc)
    return failures


def _audit_staged_export(host, result):
    path = host["temporary"]
    if any(os.path.lexists(Path(str(path) + suffix)) for suffix in ("-wal", "-shm")):
        raise ValueError("export contains WAL sidecar state")
    if not _stage_namespace_clean(host):
        raise ValueError("export stage contains unknown state")
    pending = None
    try:
        _audit_export(path, result)
    except BaseException as exc:
        pending = exc
        raise
    finally:
        # mode=ro can still create WAL/SHM in a writable directory. Absence
        # before this reader plus custody of the private namespace establishes
        # their origin; held FDs then fence removal against inode replacement.
        failures = []
        try:
            _capture_stage_sidecars(host)
        except BaseException as exc:
            failures.append(exc)
        failures.extend(_cleanup_stage_sidecars(host))
        try:
            if not _stage_namespace_clean(host):
                raise RuntimeError("audit left unknown staging state")
        except BaseException as exc:
            failures.append(exc)
        host["stage_sidecar_failures"].extend(failures)
        interruption = next((failure for failure in failures if not isinstance(failure, Exception)), None)
        if interruption is not None and (pending is None or isinstance(pending, Exception)):
            raise interruption
        if pending is None and failures:
            raise failures[0]


def _prepare_export_stage(host, destination):
    """Own a fresh private namespace and the initial file before Docker writes."""
    stage = destination.with_name("." + destination.name + ".stage-" + uuid.uuid4().hex)
    os.mkdir(stage, 0o700)
    host["stage"] = stage
    created = stage.lstat()
    host["stage_owner"] = (created.st_dev, created.st_ino)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    host["stage_fd"] = os.open(stage, flags)
    if not _stage_directory_owned(host):
        raise RuntimeError("export staging directory ownership is unproved")
    temporary = stage / "profiles.sqlite"
    host["temporary"] = temporary
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    host["stage_file_fds"].append(os.open(temporary.name, flags, 0o600, dir_fd=host["stage_fd"]))
    opened = os.fstat(host["stage_file_fds"][-1])
    host["temporary_owner"] = (opened.st_dev, opened.st_ino)
    return temporary


def _capture_completed_stage_file(host):
    # Only a successful copy into our unchanged private namespace can establish
    # a replacement file's identity. Never infer it from a failed copy's path.
    if not _stage_directory_owned(host):
        raise RuntimeError("export staging directory ownership changed")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    host["stage_file_fds"].append(os.open(host["temporary"].name, flags, dir_fd=host["stage_fd"]))
    opened = os.fstat(host["stage_file_fds"][-1])
    named = host["temporary"].lstat()
    if not stat.S_ISREG(opened.st_mode) or opened.st_size == 0 or \
            (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino) or \
            not _stage_directory_owned(host):
        raise RuntimeError("completed export staging file is unproved")
    host["temporary_owner"] = (opened.st_dev, opened.st_ino)
    return host["temporary_owner"]


def _cleanup_export_stage(host):
    """Remove only proved file/directory inodes; always attempt descriptor closure."""
    failures = list(host.get("stage_sidecar_failures", ()))
    failures.extend(_cleanup_stage_sidecars(host))
    if host.get("stage_owner") is not None:
        try:
            if not _stage_directory_owned(host):
                raise RuntimeError("export staging directory ownership changed")
            temporary, owner = host.get("temporary"), host.get("temporary_owner")
            if owner is None and host["stage_file_fds"]:
                # Interrupted initial fstat: the exclusively created file FD
                # itself proves the inode, not a post-failure named path.
                opened = os.fstat(host["stage_file_fds"][0])
                owner = (opened.st_dev, opened.st_ino)
            if temporary is not None:
                current = temporary.lstat()
                if owner is None or (current.st_dev, current.st_ino) != owner:
                    raise RuntimeError("export staging file ownership is unproved")
                temporary.unlink()
        except FileNotFoundError:
            pass
        except BaseException as exc:
            failures.append(exc)
        try:
            if _stage_directory_owned(host):
                host["stage"].rmdir()  # Never recursively delete unknown contents.
                host["stage_owner"] = None
        except FileNotFoundError:
            host["stage_owner"] = None
        except BaseException as exc:
            failures.append(exc)
    for fd in tuple(host.get("stage_file_fds", ())):
        try:
            os.close(fd)
            host["stage_file_fds"].remove(fd)
        except BaseException as exc:
            failures.append(exc)
    if host.get("stage_fd") is not None:
        try:
            os.close(host["stage_fd"])
            host["stage_fd"] = None
            # Retire namespace authority with its FD; retries must not mistake
            # a subsequently recreated directory for our original inode.
            host["stage_owner"] = None
        except BaseException as exc:
            failures.append(exc)
    return failures


def _cleanup_host_after_escape(host):
    """Last-resort host guard; each stage runs even if another is interrupted."""
    failures = _cleanup_export_stage(host)
    if host["closed"]:
        return failures
    directory_fd = None
    try:
        owned = _owns_lock(host["lock"], host["fd"], host["token"])
        candidate = host.get("token_candidate")
        if not owned and not host["initialized"] and candidate is not None:
            # A signal can arrive after write(2) changed our inode but before
            # Python records its return count. During initial construction,
            # prove the inode and the actually written candidate prefix.
            os.lseek(host["fd"], 0, os.SEEK_SET)
            written = os.read(host["fd"], len(candidate) + 1)
            if candidate.startswith(written):
                owned = _owns_lock(host["lock"], host["fd"], written)
        if owned:
            host["lock"].unlink()
            directory_fd = os.open(host["lock"].parent, os.O_RDONLY)
            os.fsync(directory_fd)
    except BaseException as exc:
        failures.append(exc)
    finally:
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except BaseException as exc:
                failures.append(exc)
    try:
        os.close(host["fd"])
        host["closed"] = True
    except BaseException as exc:
        failures.append(exc)
    return failures


def orchestrate(args):
    host = {}
    pending = None
    try:
        return _orchestrate_reserved(args, host)
    except BaseException as exc:
        pending = exc
        raise
    finally:
        # Enclose the complete reserved lifetime, including body, Docker
        # teardown, trace retrieval and commit. Never replace its interruption
        # with a secondary host-cleanup failure.
        if host and (not host["closed"] or host.get("stage_fd") is not None or host.get("stage_file_fds")):
            failures = _cleanup_host_after_escape(host)
            interruption = next((failure for failure in failures
                                 if not isinstance(failure, Exception)), None)
            if interruption is not None and (pending is None or isinstance(pending, Exception)):
                raise interruption
            if pending is None and failures:
                raise failures[0]


def _orchestrate_reserved(args, host):
    debug = bool(getattr(args, "debug", False))
    configure_trace(debug)
    prefix = "llm-measurement-" + uuid.uuid4().hex
    volumes = [prefix + suffix for suffix in ("-bundle", "-runtime", "-result", "-export")]
    containers = [prefix + suffix for suffix in ("-bundle-copy", "-verify", "-measure", "-result", "-trace", "-export")]
    resources = _OwnedDockerResources(prefix)
    destination = Path(args.output).absolute()
    if destination.exists():
        raise OperatorFailure("output_reserved", "reserve_output")
    lock = destination.with_name("." + destination.name + ".measurement.lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    except FileExistsError as exc:
        raise OperatorFailure("output_reserved", "reserve_output") from exc
    host.update(lock=lock, fd=lock_fd, token=b"", temporary=None,
                temporary_owner=None, closed=False, initialized=False, token_candidate=None,
                stage=None, stage_owner=None, stage_fd=None, stage_file_fds=[],
                stage_sidecars={}, stage_sidecar_failures=[])
    token = b""  # Also proves our empty inode if token generation fails.
    reservation_ready = False
    promoted = False
    primary = None
    inner_trace = ""
    trace_retrieved = False
    cleanup_errors = []
    docker_cleanup_errors = []
    temporary = None
    temporary_owner = None
    parsed = None
    try:
        try:
            generated = os.urandom(32)
            host["token_candidate"] = generated
            written = os.write(lock_fd, generated)
            token = generated[:written]
            host["token"] = token
            if written != len(generated):
                raise OSError("short reservation write")
            os.fsync(lock_fd)
            reservation_ready = True
            host["initialized"] = True
        except Exception as exc:
            raise OperatorFailure("export_failed", "reserve_output") from exc
        trace("outer", "reservation", "enter", matrix_index=0)
        try:
            for volume in volumes:
                resources.create_volume(volume)
        except Exception as exc:
            raise OperatorFailure("volume_setup_failed", "volume_setup") from exc
        try:
            _copy(containers[0], args.image, volumes[0], args.bundle, "/transfer", True,
                  resources=resources)
        except Exception as exc:
            raise OperatorFailure("bundle_transfer_failed", "bundle_transfer") from exc
        _create(containers[1], args.image, [(volumes[0], "/opt/measurement", True)],
                ["/opt/llm/prepare_measurement.py", "verify", "--output", "/opt/measurement"],
                resources=resources)
        if run(["docker", "start", "-a", resources.container_reference(containers[1])], check=False).returncode: raise OperatorFailure("bundle_verify_failed", "bundle_verify")
        _create(containers[2], args.image,
                [(volumes[0], "/opt/measurement", True), (volumes[1], "/var/lib/llm-measurement", False)],
                ["/opt/llm/measure_profiles.py", "--bundle", "/opt/measurement",
                 "--db", "/var/lib/llm-measurement/profiles.sqlite",
                  "--ollama-version", args.ollama_version, "--ceiling", str(args.ceiling),
                  "--result-file", "/var/lib/llm-measurement/measurement-result.json"]
                  + (["--debug", "--trace-file", "/var/lib/llm-measurement/debug-trace.jsonl"] if debug else []),
                 gpu=args.gpu_uuid, pid=True, resources=resources)
        # Two checks fence foreign work immediately before compute starts.
        for probe_index in range(2):
            probe_name = prefix + f"-probe-{probe_index}"
            # A CLI timeout can leave an auto-remove container still running.
            # Track the uncertain name so teardown must prove absence or owner
            # label/immutable ID before it can authorize the final commit.
            resources.containers.setdefault(probe_name, None)
            probe = run(["docker", "run", "--rm", "--name", probe_name,
                         "--label", f"{OWNER_LABEL}={resources.owner}",
                         "--network", "none", "--gpus", "device=" + args.gpu_uuid,
                         "--entrypoint", "nvidia-smi", args.image,
                         "--query-compute-apps=pid", "--format=csv,noheader"])
            if probe.returncode or probe.stdout.strip(): raise OperatorFailure("preflight_failed", "preflight")
        try:
            result = run(["docker", "start", "-a", resources.container_reference(containers[2])], timeout=TIMEOUT, check=False)
        except subprocess.TimeoutExpired as exc:
            raise OperatorFailure("timeout", "measurement") from exc
        inner_result = _read_runtime_result(containers[3], args.image, volumes[1], resources=resources)
        if debug:
            inner_trace = _read_runtime_trace(containers[4], args.image, volumes[1], resources=resources)
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
                ["/runtime/profiles.sqlite", "/export/profiles.sqlite"], entrypoint="/bin/cp",
                resources=resources)
        exporter = run(["docker", "start", "-a", resources.container_reference(containers[5])], check=False)
        if exporter.returncode:
            raise OperatorFailure("export_failed", "export")
        temporary = _prepare_export_stage(host, destination)
        temporary_owner = host["temporary_owner"]
        run(["docker", "cp", resources.container_reference(containers[5]) + ":/export/profiles.sqlite", str(temporary)])
        temporary_owner = _capture_completed_stage_file(host)
        _audit_staged_export(host, parsed)
        os.chmod(temporary, 0o444)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
    except OperatorFailure as exc:
        primary = exc
    except subprocess.TimeoutExpired as exc:
        primary = OperatorFailure("timeout", "operator")
    except Exception as exc:
        primary = OperatorFailure("export_audit_failed", "operator", db_retained=promoted)
        primary.__cause__ = exc
    finally:
        pending = sys.exception()
        cleanup_interrupt = pending if pending is not None and not isinstance(pending, Exception) else None
        try:
            trace("outer", "cleanup", "enter")
        except BaseException as exc:
            if not isinstance(exc, Exception) and cleanup_interrupt is None:
                cleanup_interrupt = exc
        if debug and reservation_ready and not trace_retrieved:
            try:
                inner_trace = _read_runtime_trace(containers[4], args.image, volumes[1], resources=resources)
            except BaseException as exc:
                inner_trace = ""
                if not isinstance(exc, Exception) and cleanup_interrupt is None:
                    cleanup_interrupt = exc
            trace_retrieved = True
        if debug and inner_trace:
            try:
                sys.stderr.write(inner_trace)
                sys.stderr.flush()
            except BaseException as exc:
                if not isinstance(exc, Exception) and cleanup_interrupt is None:
                    cleanup_interrupt = exc
        try:
            docker_cleanup_errors.extend(resources.cleanup())
            cleanup_errors.extend(docker_cleanup_errors)
        except BaseException as exc:
            cleanup_errors.append(exc)
            docker_cleanup_errors.append(exc)
        for failure in docker_cleanup_errors:
            if not isinstance(failure, Exception) and cleanup_interrupt is None:
                cleanup_interrupt = failure
        if cleanup_interrupt is not None:
            raise cleanup_interrupt

    # The destination is a commit record: complete Docker teardown and validate
    # our reservation before its no-clobber link can make the database visible.
    lock_failure = None
    if primary is None and not cleanup_errors:
        try:
            if not _owns_lock(lock, lock_fd, token):
                lock_failure = OperatorFailure("lock_ownership_changed", "commit")
            elif destination.exists() or destination.is_symlink():
                primary = OperatorFailure("output_reserved", "commit")
            elif temporary_owner is None or not _stage_namespace_clean(host) or not temporary.exists() or temporary.is_symlink() \
                    or (temporary.stat().st_dev, temporary.stat().st_ino) != temporary_owner:
                primary = OperatorFailure("export_audit_failed", "commit")
            else:
                os.link(temporary, destination, follow_symlinks=False)
                promoted = True
                committed = destination.lstat()
                if not stat.S_ISREG(committed.st_mode) or \
                        (committed.st_dev, committed.st_ino) != temporary_owner:
                    raise RuntimeError("committed export is not the audited staged file")
                staged_now = temporary.lstat()
                if (staged_now.st_dev, staged_now.st_ino) != temporary_owner:
                    raise RuntimeError("staged export ownership changed after commit")
                temporary.unlink()
                if (destination.stat().st_mode & 0o777) != 0o444:
                    raise RuntimeError("export is not readonly")
                directory_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                trace("outer", "promotion", "success", matrix_index=0)
        except FileExistsError:
            primary = OperatorFailure("output_reserved", "commit")
        except Exception as exc:
            primary = OperatorFailure("export_audit_failed", "commit", db_retained=promoted)
            primary.__cause__ = exc

    if debug and inner_trace and (primary is not None or lock_failure is not None or cleanup_errors) and not promoted:
        try:
            _publish_sidecar(Path(str(destination) + ".debug.jsonl"), inner_trace)
        except Exception:
            pass

    cleanup_errors.extend(_cleanup_export_stage(host))

    lock_released = False
    if _owns_lock(lock, lock_fd, token):
        try:
            lock.unlink()
            lock_released = True
            directory_fd = os.open(lock.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception as exc:
            cleanup_errors.append(exc)
    elif lock_failure is None:
        lock_failure = OperatorFailure("lock_ownership_changed", "release_lock", db_retained=promoted)
    try:
        os.close(lock_fd)
        host["closed"] = True
    except Exception as exc:
        cleanup_errors.append(exc)

    interruption = next((failure for failure in cleanup_errors
                         if not isinstance(failure, Exception)), None)
    if interruption is not None:
        raise interruption
    if lock_failure is not None:
        lock_failure.docker_cleanup = "proved" if not docker_cleanup_errors else "unproved"
        raise lock_failure
    if cleanup_errors:
        failure = OperatorFailure("cleanup_failed", "cleanup", db_retained=promoted)
        failure.docker_cleanup = "unproved" if docker_cleanup_errors else "proved"
        raise failure from cleanup_errors[0]
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

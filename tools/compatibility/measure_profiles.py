"""Production measurement entry point.

This command is deliberately a thin seam: deployments provide the normal
bootstrap binding factory and GPU proof, while this tool owns only selector
enumeration, bounded persistence, and the close/reopen audit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import re
import stat
import sqlite3
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from services.llm.bootstrap.config import load_config
from services.llm.provisioning.benchmark_requests import (
    BenchmarkRequestError, prepare_benchmark_request,
    replace_coedit_with_runtime_generated,
)
from services.llm.provisioning.measurement import (MEASUREMENT_FAILURE_CODES as _BASE_MEASUREMENT_FAILURE_CODES,
    MEASUREMENT_FAILURE_DETAILS, bounded_exception_text, measure_authoritative)
from services.llm.provisioning.measured_profiles import (
    MeasuredProfileIdentity, PersistenceValidationError,
    PERSISTENCE_VALIDATION_DETAILS, persist_measured_profile,
)
from services.llm.provisioning.rm_runner import ResourceManagerWaveRunner
from services.llm.provisioning.rm_runner import safe_provider_failure_code
from services.llm.provisioning.evidence import (smollm_evidence_extractor,
    coedit_evidence_extractor, gector_evidence_extractor)
from services.llm.provisioning.capacity import MemorySample, CapacityEvidenceError
from services.llm.resource_manager.core import ResourceManager
from services.llm.resource_manager.profiles import ProfileStore
from services.llm.providers.gpu import LinuxGPUProof, _ResidencyPending
from services.llm.providers.config import GPUProof
from services.llm.bootstrap.supervisor import OwnedOllama
from services.llm.bootstrap.bindings import observe_runtime_identities
from services.llm.bootstrap.measurement_bindings import prepare_measurement_bindings
from services.llm.queue.contracts import ModelId
from services.llm.bootstrap.measurement_matrix import MATRIX, measurement_matrix
from tools.compatibility.prepare_measurement import verify_bundle_inputs
try:
    from tools.compatibility.debug_trace import configure as configure_trace, record as trace
except ImportError:  # flat adapter image layout
    from debug_trace import configure as configure_trace, record as trace


# Persistence validation is a distinct, closed classification.  Keep it local
# to this entry point: the underlying persistence layer deliberately continues
# to expose its existing ValueError contract to its other callers.
PERSISTENCE_VALIDATION_FAILED = "persistence_validation_failed"
MEASUREMENT_FAILURE_CODES = _BASE_MEASUREMENT_FAILURE_CODES | {
    PERSISTENCE_VALIDATION_FAILED,
    "maximum_witness_failed",
    "runtime_generated_request_failed",
}
MEASUREMENT_FAILURE_DETAILS = MEASUREMENT_FAILURE_DETAILS | PERSISTENCE_VALIDATION_DETAILS | frozenset({
    "smollm_token_count", "coedit_token_count", "coedit_configured_maximum",
    "coedit_payload_fingerprint", "gector_token_count",
    "gector_configured_maximum", "gector_payload_fingerprint",
    "coedit_generated_schema", "coedit_generated_count", "coedit_generated_max",
    "coedit_generated_fingerprint", "coedit_generated_bounds",
})

DIAGNOSTIC_FAILURE_STAGE_DETAILS = frozenset({
    "gpu_proof_capture",
    "validation_session_start", "maximum_witness", "validation_session_stop",
    "measurement_session_start", "p2_wave", "runner_close", "provider_unload",
    "daemon_close",
})
DIAGNOSTIC_LIFECYCLE_PHASES = frozenset({"validate", "load", "ready", "cleanup"})
DIAGNOSTIC_LIFECYCLE_SUBREASONS = frozenset({
    "profile_shape_identity", "gpu_identity_proof", "artifact_verification",
    "gpu_memory_proof", "ollama_create", "readiness_request",
    "endpoint_residency", "gpu_residency", "fallback_ownership_memory",
    "timeout", "cleanup_verification",
})
DIAGNOSTIC_FAILURE_KINDS = frozenset({
    "provider_execution_failed", "provider_rejection", "runner_error", "oom",
    "runtime_setup_failed", "resident_warmup_failed", "residency_fence_failed",
    "evidence_error", "timeout",
})


class _MeasuredRun(list):
    """Profiles plus the matrix identity used by the completed production run."""

    def __init__(self, profiles, matrix):
        super().__init__(profiles)
        self.matrix = tuple((model.value, selector) for model, selector in matrix)


def enumerate_matrix(config_path: str | Path) -> tuple[tuple[str, str], ...]:
    load_config(config_path) # config is identity only; selectors are production-owned.
    return tuple((model.value, selector) for model, selector in measurement_matrix())


def _production_matrix_identity() -> tuple[tuple[str, str], ...]:
    """Return matrix identity without reopening operator configuration."""
    try:
        return tuple((model.value, selector) for model, selector in measurement_matrix())
    except Exception:
        # Failure reporting must not replace the failure that it is reporting.
        return ()


def _main_failure(exc: BaseException, stage: str) -> dict[str, object]:
    """Build the bounded CLI envelope without loading config or user paths."""
    messages = {
        "cli_argument_parse": "CLI argument parsing failed",
        "config_load": "configuration loading failed",
        "application": "application failure",
    }
    reason = messages.get(stage, "application failure")
    # Public envelopes are classifications, never exception renderings.  Even
    # slash-free provider text can contain prompts, secrets, or model output.
    result = {
        "status": "incomplete",
        "stage": stage,
        "failure_kind": f"{stage}_failed",
        "exception_type": type(exc).__name__,
        "reason": reason,
        "message": reason,
        "matrix": _production_matrix_identity(),
    }
    model = getattr(exc, "measurement_model", None)
    code = getattr(exc, "measurement_failure_code", None)
    if code is not None:
        result["reason"] = result["message"] = "classified measurement failure"
    if isinstance(model, str) and model in {item.value for item in ModelId}:
        result["model"] = model
    if isinstance(code, str) and code in MEASUREMENT_FAILURE_CODES:
        result["measurement_failure_code"] = code
    detail = getattr(exc, "measurement_failure_detail", None)
    if isinstance(detail, str) and detail in MEASUREMENT_FAILURE_DETAILS:
        result["measurement_failure_detail"] = detail
    telemetry = getattr(exc, "measurement_telemetry", None)
    if isinstance(telemetry, dict) and set(telemetry) == {
            "total_vram_bytes", "baseline_pre_used_bytes",
            "peak_incremental_request_bytes", "derived_ceiling"} and all(
                value is None or type(value) is int for value in telemetry.values()):
        result["measurement_telemetry"] = telemetry
    return result


def _trace_main_failure(failure: dict[str, object]) -> None:
    """Publish the terminal application classification without exception data."""
    fields = {key: failure[key] for key in (
        "stage", "failure_kind", "model", "measurement_failure_code",
        "measurement_failure_detail") if key in failure}
    trace("measurement", "output", "failure", **fields)


def _classified_measurement_failure(model: ModelId, measurement) -> CapacityEvidenceError:
    code = getattr(measurement, "failure_code", None)
    if code not in MEASUREMENT_FAILURE_CODES:
        code = "unknown_evidence_failure"
    error = CapacityEvidenceError(code)
    error.measurement_model = model.value
    error.measurement_failure_code = code
    detail = getattr(measurement, "failure_detail", None)
    if detail in MEASUREMENT_FAILURE_DETAILS:
        error.measurement_failure_detail = detail
    error.measurement_telemetry = {
        "total_vram_bytes": measurement.total_vram_bytes,
        "baseline_pre_used_bytes": measurement.baseline_pre_used_bytes,
        "peak_incremental_request_bytes": measurement.peak_incremental_request_bytes,
        "derived_ceiling": measurement.derived_ceiling,
    }
    return error


def _persist_profile_or_classify(model: ModelId, request, measurement,
                                 identity, writer):
    """Persist one measured profile without leaking validation diagnostics."""
    try:
        return persist_measured_profile(
            request, measurement, identity, writer,
            latency_extractor=lambda ids, wave: wave.request_latency_ms,
        )
    except PersistenceValidationError as exc:
        error = CapacityEvidenceError(PERSISTENCE_VALIDATION_FAILED)
        error.measurement_model = model.value
        error.measurement_failure_code = PERSISTENCE_VALIDATION_FAILED
        error.measurement_failure_detail = exc.measurement_failure_detail
        raise error from exc
    except ValueError as exc:
        error = CapacityEvidenceError(PERSISTENCE_VALIDATION_FAILED)
        error.measurement_model = model.value
        error.measurement_failure_code = PERSISTENCE_VALIDATION_FAILED
        raise error from exc


class _Sampler:
    def __init__(self, proof): self.proof = proof
    async def sample(self):
        point = await self.proof.memory()
        if point.gpu_uuid != self.proof.target_uuid: raise CapacityEvidenceError("GPU UUID changed")
        return MemorySample(point.end_ns, point.total_bytes, point.used_bytes, point.free_bytes, point.start_ns, point.end_ns)

def _reserve_destination(destination: Path) -> tuple[int, Path, bytes]:
    """Fence a destination for the complete run without touching an existing DB."""
    if destination.exists():
        raise ValueError("refusing to overwrite an existing profile registry")
    lock = Path(str(destination) + ".measurement.lock")
    try:
        token = os.urandom(32)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.write(fd, token)
        os.fsync(fd)
        return fd, lock, token
    except FileExistsError as exc:
        raise ValueError("profile registry destination is already reserved") from exc


def _release_destination(fd: int, lock: Path, token: bytes) -> None:
    """Release only the lock whose inode and owner token are still ours."""
    try:
        lock_stat = lock.stat()
        owner_stat = os.fstat(fd)
        if (lock_stat.st_dev, lock_stat.st_ino) != (owner_stat.st_dev, owner_stat.st_ino):
            raise RuntimeError("profile registry lock inode changed")
        with lock.open("rb") as stream:
            current = stream.read()
        if current != token:
            raise RuntimeError("profile registry lock ownership changed")
        os.unlink(lock)
    finally:
        os.close(fd)


def _assert_reservation(fd: int, lock: Path, token: bytes) -> None:
    lock_stat = lock.stat()
    owner_stat = os.fstat(fd)
    if (lock_stat.st_dev, lock_stat.st_ino) != (owner_stat.st_dev, owner_stat.st_ino):
        raise RuntimeError("profile registry lock inode changed")
    if lock.read_bytes() != token:
        raise RuntimeError("profile registry lock ownership changed")


def _checkpoint_audited_database(path: Path) -> None:
    db = sqlite3.connect(path, timeout=0)
    try:
        result = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if result is None or tuple(result) != (0, 0, 0):
            raise RuntimeError("profile registry WAL checkpoint was not successful")
    finally:
        db.close()
    for suffix in ("-wal", "-shm"):
        if Path(str(path) + suffix).exists():
            raise RuntimeError("profile registry has residual WAL state")


def _install_atomic(staged: Path, destination: Path, *, reservation_fd: int | None = None,
                    reservation_lock: Path | None = None,
                    reservation_token: bytes | None = None) -> None:
    """Install an audited database with an atomic, no-clobber commit."""
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.chmod(staged, 0o444)
        with staged.open("rb") as stream: os.fsync(stream.fileno())
        os.fsync(directory_fd)
        if reservation_fd is not None:
            if reservation_lock is None or reservation_token is None:
                raise RuntimeError("profile registry reservation is incomplete")
            # This check is deliberately inside the commit critical section:
            # no-clobber and foreign-lock protection cover the install itself.
            _assert_reservation(reservation_fd, reservation_lock, reservation_token)
        # link(2), unlike replace(2), cannot overwrite a foreign destination.
        os.link(staged, destination)
        os.unlink(staged)
        os.fsync(directory_fd)
        mode = stat.S_IMODE(destination.stat().st_mode)
        if mode != 0o444:
            raise RuntimeError("installed profile registry is not readonly")
    finally:
        os.close(directory_fd)

def _requests(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    if not raw or len(raw) > 256 * 1024: raise ValueError("requests file is empty or exceeds 256 KiB")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"SmolLM", "CoEdIT", "GECToR"}:
        raise ValueError("requests must contain exactly one bounded request per configured model")
    return value


def _diagnostic_request(path: Path) -> object:
    """Read one exact, bounded SmolLM request without retaining its contents."""
    raw = path.read_bytes()
    if not raw or len(raw) > 256 * 1024:
        raise ValueError("diagnostic request is empty or exceeds 256 KiB")
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("diagnostic request is not valid JSON") from exc


def _diagnostic_result_file(path: Path, config) -> Path:
    """Accept only the fixed, writable runtime result transport location."""
    expected = Path(config.profile_db).parent / "diagnostic-result.json"
    if not path.is_absolute() or path != expected:
        raise ValueError("diagnostic result file must use the fixed runtime location")
    _reject_symlink_components(path, "result file")
    if path.exists() or path.is_symlink():
        raise ValueError("diagnostic result file is unsafe")
    return path


def _trace_file(path: Path, config) -> Path:
    """Accept only the fixed trace transport beside the configured profile DB."""
    expected = Path(config.profile_db).parent / "debug-trace.jsonl"
    if not path.is_absolute() or path != expected:
        raise ValueError("trace file must use the fixed runtime location")
    _reject_symlink_components(path, "trace file")
    if path.exists() or path.is_symlink():
        raise ValueError("trace file is unsafe")
    if not path.parent.is_dir():
        raise ValueError("trace file parent is unavailable")
    return path


def _write_diagnostic_result(path: Path, result: dict[str, object]) -> None:
    """Atomically publish the already-finalized public JSON line to runtime storage."""
    encoded = (json.dumps(result, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > 8192:
        raise ValueError("diagnostic result is oversized")
    if path.exists() or path.is_symlink():
        raise FileExistsError("result transport already exists")
    fd = os.open(path.parent, os.O_RDONLY)
    temporary = path.parent / ("." + path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("result temporary transport already exists")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        os.fsync(fd)
    finally:
        temporary.unlink(missing_ok=True)
        os.close(fd)


def _profile_summary(identity) -> dict[str, str]:
    model = getattr(identity, "model_id", None)
    model = model.value if hasattr(model, "value") else str(model or "unknown")
    return {"model": model,
            "profile_identity": str(getattr(identity, "profile_identity", identity))}


def _write_measurement_result(path: Path, profiles: _MeasuredRun) -> None:
    """Publish only bounded selector/profile identity summaries."""
    result = {"status": "complete", "matrix": list(profiles.matrix),
              "profiles": [_profile_summary(identity) for identity in profiles]}
    encoded = (json.dumps(result, sort_keys=True) + "\n").encode()
    if len(encoded) > 8192:
        raise ValueError("measurement result is oversized")
    if path.exists() or path.is_symlink():
        raise FileExistsError("result transport already exists")
    fd = os.open(path.parent, os.O_RDONLY)
    temporary = path.parent / ("." + path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("result temporary transport already exists")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path); temporary.unlink(); os.fsync(fd)
    finally:
        temporary.unlink(missing_ok=True); os.close(fd)


def _lexical_absolute(path: Path) -> Path:
    """Make an absolute path without consulting or creating its target."""
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError("path is not usable") from exc


def _reject_symlink_components(path: Path, name: str) -> None:
    """Reject a supplied path that traverses any existing symlink component.

    Inspect the unnormalised spelling as well as the later lexical absolute
    spelling: a ``link/..`` component must not hide a traversal through link.
    ``lexists`` deliberately includes a dangling symlink.
    """
    supplied = Path(path)
    current = supplied.anchor and Path(supplied.anchor) or Path.cwd()
    parts = supplied.parts[1:] if supplied.is_absolute() else supplied.parts
    for part in parts:
        if part in ("", "."):
            continue
        current /= part
        try:
            exists = os.path.lexists(current)
            symlink = current.is_symlink() if exists else False
        except OSError as exc:
            raise ValueError(f"{name} path cannot be safely inspected") from exc
        if symlink:
            raise ValueError(f"{name} must be a non-symlink path")


def _resolve_without_target_creation(path: Path, name: str) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{name} path cannot be resolved safely") from exc


def _validate_db_location(db: Path, bundle: Path, config) -> None:
    """Fence the output registry away from the immutable verified bundle."""
    lexical_db = _lexical_absolute(db)
    lexical_bundle = _lexical_absolute(bundle)
    _reject_symlink_components(Path(db), "db")
    _reject_symlink_components(Path(bundle), "bundle")
    # Check normalized spellings too, since their existing parent components are
    # the ones later used by database creation and atomic installation.
    _reject_symlink_components(lexical_db, "db")
    _reject_symlink_components(lexical_bundle, "bundle")
    resolved_db = _resolve_without_target_creation(lexical_db, "db")
    resolved_bundle = _resolve_without_target_creation(lexical_bundle, "bundle")
    try:
        resolved_db.relative_to(resolved_bundle)
    except ValueError:
        pass
    else:
        raise ValueError("db must be outside the resolved bundle")
    profile_db = _resolve_without_target_creation(
        _lexical_absolute(Path(config.profile_db)), "generated config.profile_db")
    if resolved_db != profile_db:
        raise ValueError("db must equal generated config.profile_db")


def _safe_diagnostic_text(value, fallback: str, *, classification: bool = False) -> str:
    """Return bounded diagnostic text without paths or control characters."""
    if not isinstance(value, str):
        return fallback
    text = bounded_exception_text(RuntimeError(value))
    if not text or "/" in text or "\\" in text:
        return fallback
    if classification:
        # Provider classifications are useful only as compact, opaque names;
        # do not let arbitrary metadata become an output channel.
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", text):
            return fallback
    return text


def _diagnostic_failure(exc: BaseException, stage: str) -> dict[str, object]:
    """Expose only the bounded failure contract useful to an operator."""

    def metadata(error: BaseException, name: str) -> object:
        try:
            return getattr(error, name, None)
        except BaseException:
            return None

    # Walk producer-owned metadata, not rendered exception text.  Cause/context
    # links matter because RM wraps provider failures, while the small bound
    # prevents hostile exception graphs from becoming an unbounded diagnostic.
    pending = [exc]
    seen: set[int] = set()
    nodes: list[BaseException] = []
    while pending and len(seen) < 32:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current)); nodes.append(current)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(item for item in current.exceptions if isinstance(item, BaseException))
        for related in (current.__cause__, current.__context__):
            if isinstance(related, BaseException):
                pending.append(related)

    def first(name: str):
        # Cleanup metadata is authoritative when an RM exception group carries
        # both the original startup fault and a later teardown fault.  If the
        # cleanup node does not carry a particular field, the bounded startup
        # nodes remain available as useful secondary metadata.
        ordered = [node for node in nodes
                   if metadata(node, "lifecycle_phase") == "cleanup"]
        ordered.extend(node for node in nodes if node not in ordered)
        for node in ordered:
            value = metadata(node, name)
            if value is not None:
                yield value

    kind = next((value for value in first("failure_kind")
                 if isinstance(value, str) and value in DIAGNOSTIC_FAILURE_KINDS),
                "provider_execution_failed")
    code = next((value for value in (safe_provider_failure_code(item)
                                     for item in first("failure_code")
                                     if isinstance(item, str)) if value),
                "provider_execution_failed")
    detail = next((value for value in first("failure_stage_detail")
                   if isinstance(value, str) and value in DIAGNOSTIC_FAILURE_STAGE_DETAILS), None)
    phase = None; subreason = None
    ordered_nodes = [node for node in nodes
                     if metadata(node, "lifecycle_phase") == "cleanup"]
    ordered_nodes.extend(node for node in nodes if node not in ordered_nodes)
    for node in ordered_nodes:
        candidate_phase = metadata(node, "lifecycle_phase")
        candidate_subreason = metadata(node, "lifecycle_subreason")
        if (isinstance(candidate_phase, str) and candidate_phase in DIAGNOSTIC_LIFECYCLE_PHASES and
                isinstance(candidate_subreason, str) and candidate_subreason in DIAGNOSTIC_LIFECYCLE_SUBREASONS):
            phase, subreason = candidate_phase, candidate_subreason
            if phase == "cleanup":
                break
    if detail is None and stage in DIAGNOSTIC_FAILURE_STAGE_DETAILS:
        detail = stage
    # Messages are deliberately non-authoritative and may contain provider
    # bodies, prompts, paths, or exception text.  Keep the public envelope
    # fixed rather than attempting to sanitize arbitrary text.
    message = "provider execution failed"
    result = {
        "model": "SmolLM", "provider_class": "SmolLMProvider", "provider_category": "ollama",
        "stage": stage, "concurrency": 2, "wave": 1, "status": "incomplete",
        "failure_kind": kind,
        "failure_code": code,
        "failure_message": message,
    }
    if phase in DIAGNOSTIC_LIFECYCLE_PHASES:
        result["lifecycle_phase"] = phase
    if subreason in DIAGNOSTIC_LIFECYCLE_SUBREASONS:
        result["lifecycle_subreason"] = subreason
    if detail is not None:
        result["failure_stage_detail"] = detail
    return result


async def _diagnostic_run(args, config) -> dict[str, object]:
    """Run exactly one SmolLM p=2 wave; this path has no persistence seam."""
    if args.selector != "smollm:context512":
        raise ValueError("diagnostic selector must be exactly smollm:context512")
    request_value = _diagnostic_request(args.request)
    daemon = None
    bindings = None
    runner = None
    daemon_closed = False
    cleanup_error = None
    cleanup_stage_detail = None
    cancellation = None
    result: dict[str, object] | None = None
    started = False

    async def cleanup() -> None:
        nonlocal daemon_closed, cleanup_stage_detail
        errors: list[BaseException] = []
        providers = []
        if bindings is not None:
            for binding in bindings.bindings.values():
                if all(binding.provider is not provider for provider in providers):
                    providers.append(binding.provider)
        # The runner owns the RM session and must close before providers/daemon.
        if runner is not None:
            try:
                await runner.close()
            except BaseException as exc:
                cleanup_stage_detail = "runner_close"
                errors.append(exc)
        for provider in providers:
            unload = getattr(provider, "unload", None)
            if unload is not None:
                try:
                    await unload()
                except BaseException as exc:
                    cleanup_stage_detail = "provider_unload"
                    errors.append(exc)
        if daemon is not None and not daemon_closed:
            try:
                await daemon.close()
            except BaseException as exc:
                cleanup_stage_detail = "daemon_close"
                errors.append(exc)
            else:
                daemon_closed = True
        if errors:
            raise BaseExceptionGroup("diagnostic lifecycle cleanup failed", errors)

    stage_detail = "gpu_proof_capture"
    try:
        proof = await asyncio.to_thread(LinuxGPUProof.capture, config.gpu_uuid, os.getpid(),
                                        Path("/proc"), None, host_pid_namespace=True)
        stage_detail = "measurement_session_start"
        def ownership_snapshot():
            if daemon is None or not started:
                raise RuntimeError("Ollama ownership snapshot is unavailable before daemon startup")
            return daemon.ownership_snapshot()

        typed_proof = GPUProof(proof.identity, proof.cleanup, proof.residency,
                               expected_supervisor=proof.supervisor_identity,
                               residency_for_runner=proof.residency_for_runner,
                               memory=proof.memory, ollama_ownership=ownership_snapshot)
        daemon = OwnedOllama(config, typed_proof, num_parallel=2)
        version = await daemon.start()
        started = True
        if version != args.ollama_version:
            raise ValueError("started Ollama version differs from --ollama-version")
        bindings = await prepare_measurement_bindings(
            config, typed_proof, observe_runtime_identities(version), ceiling=2)
        binding = bindings.bindings[(ModelId.SMOLLM, args.selector)]
        request = prepare_benchmark_request(
            MATRIX[ModelId.SMOLLM], request_bucket=args.selector,
            identity_witnesses={"dtype": "q8_0",
                                "generation_parameters": {"num_predict": 64, "temperature": 0},
                                "native_batch_shape": [1]},
            configured_request=request_value,
            validate_request=lambda candidate, *_: isinstance(candidate, str))
        validation_rm = ResourceManager()
        stage_detail = "validation_session_start"
        session = await validation_rm.start_session(
            "diagnostic-validation", ModelId.SMOLLM, binding.profile, binding.provider,
            idempotency_key="diagnostic-validate-start")
        try:
            stage_detail = "maximum_witness"
            await _maximum_witness(binding.provider, ModelId.SMOLLM, request,
                                   context=512, bucket=None)
        finally:
            stage_detail = "validation_session_stop"
            await validation_rm.stop_session(session.session_token, reason="diagnostic_validation",
                                             idempotency_key="diagnostic-validate-stop")
        if validation_rm.snapshot().phase == "cleanup_failed":
            raise CapacityEvidenceError("maximum request cleanup failed")
        stage_detail = "measurement_session_start"
        runner = ResourceManagerWaveRunner(
            ResourceManager(), binding, scheduler_id="profile-diagnostic",
            model_id=ModelId.SMOLLM, context_size=512, provisioning_profile=binding.profile,
            configured_ceiling=2, run_identity="smollm-p2")
        stage_detail = "p2_wave"
        wave = await runner(2, 1, ("diagnostic-smollm-p2-0", "diagnostic-smollm-p2-1"),
                            request.payload, smollm_evidence_extractor)
        # The SmolLM evidence extractor is authoritative, but completion is a
        # separate operator gate: do not label a wave complete unless every
        # native overlap/correlation/observation invariant is visible here.
        if (wave.native_request_correlation is not True
                or wave.observation_count != 2
                or wave.observed_native_batch_sizes != (1, 1)
                or wave.observation_drops != 0
                or getattr(wave, "evidence_kind", "ollama_native") != "ollama_native"
                or not (type(wave.execution_started) is int
                        and type(wave.execution_ended) is int
                        and wave.execution_ended > wave.execution_started)):
            raise CapacityEvidenceError("diagnostic evidence did not prove native p2 execution")
        result = {"model": "SmolLM", "provider_class": type(binding.provider).__name__,
                  "provider_category": "ollama", "stage": "p2_wave", "concurrency": 2,
                  "wave": 1, "status": "complete", "failure_kind": None,
                  "failure_code": None, "failure_message": None,
                   "native_overlap": True,
                   "native_request_correlation": wave.native_request_correlation,
                   "native_observation_count": wave.observation_count,
                   "native_batch_sizes": wave.observed_native_batch_sizes,
                   "observation_drops": wave.observation_drops,
                   "cleanup": "pending"}
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        try:
            exc.failure_stage_detail = stage_detail
        except BaseException:
            pass
        result = _diagnostic_failure(exc, "p2_wave")
    finally:
        try:
            # Joining cleanup prevents cancellation from abandoning resident work.
            task = asyncio.create_task(cleanup())
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancellation = exc
                await asyncio.shield(task)
        except BaseException as exc:
            cleanup_error = exc
        if result is None:
            result = _diagnostic_failure(RuntimeError("diagnostic did not produce a result"), "lifecycle")
        if cleanup_error is not None:
            result.update(status="incomplete", stage="cleanup",
                          failure_kind="cleanup_failed", failure_code="cleanup_failed",
                          # Cleanup aggregates independently failing owned
                          # resources.  Its group rendering includes a nested
                          # exception count, which is neither stable nor safe
                          # diagnostic metadata.
                           failure_message="diagnostic lifecycle cleanup failed")
            result["lifecycle_phase"] = "cleanup"
            result["lifecycle_subreason"] = "cleanup_verification"
            if cleanup_stage_detail is not None:
                result["failure_stage_detail"] = cleanup_stage_detail
        else:
            result["cleanup"] = "proved"
        if cancellation is not None:
            raise cancellation
    return result

async def _maximum_witness(provider, model, request, *, context, bucket):
    """Ask the loaded normal adapter/worker to prove the configured maximum."""
    def witness_failure(detail: str) -> None:
        # Only producer-owned classifications cross the application boundary;
        # never put returned values, request bodies, or provider text there.
        error = CapacityEvidenceError("maximum_witness_failed")
        error.measurement_model = model.value
        error.measurement_failure_code = "maximum_witness_failed"
        error.measurement_failure_detail = detail
        raise error

    await provider.validate_input(request.payload, context_size=context, bucket_identity=bucket)
    if model is ModelId.COEDIT:
        body = json.loads(request.payload)
        value = await provider.worker.call("benchmark_input", instruction=body["instruction"], text=body["texts"][0], generate=False)
        if value.get("count") != 128:
            witness_failure("coedit_token_count")
        if value.get("max") != 128:
            witness_failure("coedit_configured_maximum")
        if value.get("fingerprint") != __import__("hashlib").sha256(request.payload).hexdigest():
            witness_failure("coedit_payload_fingerprint")
    elif model is ModelId.GECTOR:
        body = json.loads(request.payload)
        value = await provider.worker.call("benchmark_input", **body)
        if value.get("count") != 128:
            witness_failure("gector_token_count")
        if value.get("max") != 128:
            witness_failure("gector_configured_maximum")
        if value.get("fingerprint") != __import__("hashlib").sha256(request.payload).hexdigest():
            witness_failure("gector_payload_fingerprint")
    else:
        # The selected GGUF metadata proves a 448-token prompt bound and a 64
        # token reserve; actual response telemetry later must prove eval_count.
        if len(request.payload) != 256:
            witness_failure("smollm_token_count")


async def _runtime_generated_coedit_request(provider, seed, *, bucket):
    """Replace the authenticated CoEdIT seed with one worker-proved payload."""
    body = json.loads(seed.payload)
    try:
        generated = await provider.worker.call(
            "benchmark_input", instruction=body["instruction"],
            text=body["texts"][0], generate=True)
        request = replace_coedit_with_runtime_generated(seed, generated)
        await provider.validate_input(request.payload, context_size=None,
                                      bucket_identity=bucket)
        return request
    except BenchmarkRequestError as exc:
        message = str(exc)
        detail = ("coedit_generated_count" if "token count" in message else
                  "coedit_generated_max" if "maximum" in message else
                  "coedit_generated_fingerprint" if "fingerprint" in message else
                  "coedit_generated_bounds" if "bound" in message else
                  "coedit_generated_schema")
    except Exception:
        detail = "coedit_generated_schema"
    error = CapacityEvidenceError("runtime_generated_request_failed")
    error.measurement_model = ModelId.COEDIT.value
    error.measurement_failure_code = "runtime_generated_request_failed"
    error.measurement_failure_detail = detail
    raise error


async def _smollm_residency_fence(typed_proof, provider, _p: int) -> None:
    """Require generic residency, with only the established SmolLM fallback."""
    try:
        await typed_proof.residency()
    except _ResidencyPending:
        accepted = getattr(provider, "accepted_model_specific_residency", None)
        if not callable(accepted) or accepted() is not True:
            raise

async def _run(args, config):
    production_matrix = tuple(measurement_matrix())
    requests = _requests(args.requests)
    # A measurement run is an all-or-nothing operator audit.  Never append to
    # an unknown, draft, partial, or previously completed registry: profiles
    # include the fresh request/provenance identity and a mixed database cannot
    # truthfully represent this matrix run.
    if not args.db.parent.is_dir(): raise ValueError("profile registry parent is unavailable")
    reservation_fd, reservation_lock, reservation_token = _reserve_destination(args.db)
    daemon = None
    bindings = None
    writer = None
    temporary = None
    daemon_closed = False
    persisted = []
    primary = None
    close_lifecycle = None
    try:
        proof = await asyncio.to_thread(LinuxGPUProof.capture, config.gpu_uuid, os.getpid(), Path("/proc"), None, host_pid_namespace=True)
        started = False

        def daemon_ownership_snapshot():
            if daemon is None or not started:
                raise RuntimeError("Ollama ownership snapshot is unavailable before daemon startup")
            return daemon.ownership_snapshot()

        typed_proof = GPUProof(proof.identity, proof.cleanup, proof.residency,
                               expected_supervisor=proof.supervisor_identity,
                               residency_for_runner=proof.residency_for_runner,
                               memory=proof.memory,
                               ollama_ownership=daemon_ownership_snapshot)
        # The benchmark ceiling is a search bound, not an Ollama allocation.
        # Start SmolLM at p=1 and replace this owned daemon at every measured
        # concurrency below.  In particular, never preallocate a 32-slot KV
        # cache merely because the operator requested a ceiling of 32.
        daemon = OwnedOllama(config, typed_proof, num_parallel=1)
        version = await daemon.start()
        started = True
        if version != args.ollama_version:
            raise ValueError("started Ollama version differs from --ollama-version")
        observed = observe_runtime_identities(version)
        bindings = await prepare_measurement_bindings(config, typed_proof, observed, ceiling=args.ceiling)
        fd, temporary_name = tempfile.mkstemp(prefix=".measure-", suffix=".sqlite", dir=args.db.parent)
        os.close(fd); temporary = Path(temporary_name); temporary.unlink()
        writer = ProfileStore(temporary)

        async def close_lifecycle() -> None:
            errors = []
            providers = []
            for binding in bindings.bindings.values():
                if all(binding.provider is not provider for provider in providers):
                    providers.append(binding.provider)
            for provider in providers:
                unload = getattr(provider, "unload", None)
                if unload is not None:
                    try:
                        await unload()
                    except BaseException as exc:
                        errors.append(exc)
            if daemon is not None and not daemon_closed:
                try:
                    await daemon.close()
                except BaseException as exc:
                    errors.append(exc)
                else:
                    nonlocal_daemon_closed[0] = True
            if errors:
                raise BaseExceptionGroup("measurement lifecycle cleanup failed", errors)

        nonlocal_daemon_closed = [False]
        for mid, selector in production_matrix:
            name = mid.value; binding = bindings.bindings[(mid, selector)]
            trace("matrix", "selector", "enter", model=name, selector=selector,
                  matrix_index=production_matrix.index((mid, selector)), configured_ceiling=args.ceiling)
            # Request preparation consumes the production selector and then the
            # loaded adapter validates the exact maximum below.
            if mid is ModelId.SMOLLM:
                cfg = MATRIX[mid]
                witnesses = {"dtype":"q8_0", "generation_parameters":{"num_predict":64,"temperature":0}, "native_batch_shape":[1]}
                expected_output, extractor, context, bucket = 64, smollm_evidence_extractor, 512, None
            elif mid is ModelId.COEDIT:
                cfg = MATRIX[mid]
                witnesses = {"adapter":"loaded"}; expected_output, extractor, context, bucket = 64, coedit_evidence_extractor, None, selector
            else:
                cfg = MATRIX[mid]
                witnesses = {"adapter":"loaded"}; expected_output, extractor, context, bucket = 1, gector_evidence_extractor, None, selector
            def validate(candidate, _cfg, _bucket, _selector):
                # Construction is intentionally separated from loaded witness;
                # _maximum_witness below is the authoritative attestation.
                return isinstance(candidate, (str, dict))
            request = prepare_benchmark_request(cfg, request_bucket=selector, identity_witnesses=witnesses, configured_request=requests[name], validate_request=validate)
            # Prove the exact canonical maximum against the loaded normal
            # adapter before measurement. This session is independently fenced
            # and cleaned; a structural JSON check is never sufficient.
            validation_rm = ResourceManager()
            session = await validation_rm.start_session("profile-validation", mid, binding.profile, binding.provider, idempotency_key="measurement-validate-" + name)
            try:
                if mid is ModelId.COEDIT:
                    # The verified bundle authenticates only the deterministic
                    # seed.  Generate it while the normal validation session
                    # has the owned worker loaded, then keep this exact
                    # immutable request for witness, measurement, and storage.
                    request = await _runtime_generated_coedit_request(
                        binding.provider, request, bucket=selector)
                await _maximum_witness(binding.provider, mid, request, context=context, bucket=bucket)
            finally:
                await validation_rm.stop_session(session.session_token, reason="measurement_validation", idempotency_key="measurement-validate-stop-" + name)
                if validation_rm.snapshot().phase == "cleanup_failed": raise CapacityEvidenceError("maximum request cleanup failed")
            effective_ceiling = min(args.ceiling, binding.provider_max_parallelism,
                                    binding.identity_derived_max_parallelism)
            runner = ResourceManagerWaveRunner(ResourceManager(), binding, scheduler_id="profile-measurement", model_id=mid,
                context_size=context, bucket_identity=bucket, provisioning_profile=binding.profile,
                configured_ceiling=effective_ceiling, run_identity=name)
            runner_for_concurrency = None
            residency_fence = None
            if mid is ModelId.SMOLLM:
                active_slots = 1

                async def smollm_runner_for_concurrency(p: int):
                    """Restart only the owned Ollama runtime with exactly p slots.

                    The measurement engine closes its RM session before asking
                    for a replacement.  Closing the daemon and proving GPU
                    cleanup before its replacement prevents a stale resident
                    runner from contaminating the next configuration.
                    """
                    nonlocal daemon, daemon_closed, active_slots, started
                    if active_slots != p:
                        if daemon is not None and not daemon_closed:
                            await daemon.close()
                            daemon_closed = True
                            started = False
                        if not await typed_proof.cleanup():
                            raise CapacityEvidenceError("cleanup_failed before SmolLM runtime restart")
                        daemon = OwnedOllama(config, typed_proof, num_parallel=p)
                        daemon_closed = False
                        version = await daemon.start()
                        started = True
                        if version != args.ollama_version:
                            raise ValueError("restarted Ollama version differs from --ollama-version")
                        active_slots = p
                    return ResourceManagerWaveRunner(
                        ResourceManager(), binding, scheduler_id="profile-measurement",
                        model_id=mid, context_size=context, bucket_identity=bucket,
                        provisioning_profile=binding.profile,
                         configured_ceiling=effective_ceiling, run_identity=name)

                async def smollm_residency_fence(_p: int) -> None:
                    await _smollm_residency_fence(typed_proof, binding.provider, _p)

                runner_for_concurrency = smollm_runner_for_concurrency
                residency_fence = smollm_residency_fence
            measurement = await measure_authoritative(
                request, runner, _Sampler(proof), configured_ceiling=effective_ceiling,
                identity_derived_max_parallelism=binding.identity_derived_max_parallelism,
                identity_derived_capability_reason=binding.identity_derived_capability_reason,
                provider_max_parallelism=binding.provider_max_parallelism,
                expected_max_output_tokens=expected_output, evidence_extractor=extractor,
                run_identity=name, runner_for_concurrency=runner_for_concurrency,
                residency_fence=residency_fence)
            trace("matrix", "measurement", "success" if measurement.profile_eligible else "failure",
                  model=name, selector=selector, concurrency=measurement.n,
                   configured_ceiling=effective_ceiling,
                   search_ceiling=args.ceiling,
                  failure_code=getattr(measurement, "failure_code", None),
                  failure_detail=getattr(measurement, "failure_detail", None),
                  last_phase=(getattr(measurement.measured[-1], "phase", None)
                              if getattr(measurement, "measured", ()) else None))
            if not measurement.profile_eligible:
                raise _classified_measurement_failure(mid, measurement)
            identity = MeasuredProfileIdentity(mid, config.gpu_uuid, config.manifest_sha256, bindings.hashes[name], config.models[name].runtime_identity, config.models[name].adapter_identity, args.provenance, datetime.now(timezone.utc).isoformat(), context, bucket)
            persisted.append(_persist_profile_or_classify(mid, request, measurement, identity, writer).profile)
            trace("matrix", "persistence", "success", model=name, selector=selector)
        writer.close(); writer = None
        with ProfileStore.open_readonly(temporary) as audit:
            audit.validate_all_measured({profile.profile_identity for profile in persisted})
        trace("matrix", "audit", "success", count=len(persisted))
        # Do not install until the owned daemon and all providers have cleaned up.
        import sqlite3
        _checkpoint_audited_database(temporary)
        await close_lifecycle()
        trace("matrix", "lifecycle_close", "success", count=len(persisted))
        daemon_closed = nonlocal_daemon_closed[0]
        # Commit while the reservation is held.  Release is post-commit and
        # best-effort, so a lock unlink/close fault cannot contradict success.
        _install_atomic(temporary, args.db, reservation_fd=reservation_fd,
                        reservation_lock=reservation_lock,
                        reservation_token=reservation_token)
        try:
            _release_destination(reservation_fd, reservation_lock, reservation_token)
        except BaseException:
            try:
                os.close(reservation_fd)
            except OSError:
                pass
        reservation_fd = None
        return _MeasuredRun(persisted, production_matrix)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        cleanup_error = None
        try:
            if writer is not None:
                writer.close()
            if bindings is not None and not daemon_closed and close_lifecycle is not None:
                await close_lifecycle()
                daemon_closed = nonlocal_daemon_closed[0]
            elif daemon is not None and not daemon_closed:
                await daemon.close()
                daemon_closed = True
        except BaseException as exc:
            cleanup_error = exc
        if temporary is not None:
            temporary.unlink(missing_ok=True)
            for suffix in ("-wal", "-shm"):
                Path(str(temporary) + suffix).unlink(missing_ok=True)
        try:
            if cleanup_error is not None:
                if primary is not None:
                    raise BaseExceptionGroup("measurement and lifecycle cleanup failed", [primary, cleanup_error])
                raise cleanup_error
        finally:
            if reservation_fd is not None:
                try:
                    _release_destination(reservation_fd, reservation_lock, reservation_token)
                except BaseException:
                    # A failed post-failure release must not hide the original
                    # failure; ownership was already checked before commit.
                    try:
                        os.close(reservation_fd)
                    except OSError:
                        pass

def main() -> int:
    parser = argparse.ArgumentParser(description="run all three actual-GPU authoritative measurements; use --pid=host --network none and an exact GPU UUID")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bundle", type=Path,
                        help="verified prepared bundle; atomically binds config, requests, and provenance")
    parser.add_argument("--db", type=Path,
                        help="profile DB for the all-model measurement (never used by diagnostics)")
    parser.add_argument("--requests", type=Path, help="legacy bounded request mapping (full matrix only with an equivalent verified bundle)")
    parser.add_argument("--provenance", help="legacy immutable provenance (not accepted without --bundle)")
    parser.add_argument("--ollama-version", required=True)
    parser.add_argument("--ceiling", type=int, default=32, choices=range(1, 33))
    parser.add_argument("--diagnostic-smollm-p2", action="store_true",
                        help="run one exact SmolLM p=2 diagnostic without writing a profile DB")
    parser.add_argument("--request", type=Path,
                        help="exact configured SmolLM request JSON for diagnostic mode")
    parser.add_argument("--result-file", type=Path,
                         help="fixed writable runtime JSON result transport")
    parser.add_argument("--debug", action="store_true", help="emit bounded structured tracing")
    parser.add_argument("--trace-file", type=Path,
                        help="fixed runtime-volume JSONL trace transport")
    parser.add_argument("--selector", default="smollm:context512",
                        help="diagnostic selector; must remain smollm:context512")
    failure_result_file = None
    try:
        try:
            args = parser.parse_args()
        except SystemExit as exc:
            # argparse's normal usage/error text remains on stderr, but its
            # status-2 exit must still cross the JSON boundary.  --help keeps
            # argparse's conventional successful exit behavior.
            if exc.code == 0:
                raise
            print(json.dumps(_main_failure(exc, "cli_argument_parse"), sort_keys=True))
            return 2
        configure_trace(args.debug)
        trace("measurement", "cli", "entered", matrix_index=0, configured_ceiling=args.ceiling)
        # Preserve the operator's original mode selection before bundle
        # verification replaces legacy values with immutable bundle inputs.
        supplied_config = args.config is not None
        supplied_requests = args.requests is not None
        supplied_provenance = args.provenance is not None
        supplied_bundle = args.bundle is not None
        supplied_db = args.db is not None
        if args.diagnostic_smollm_p2:
            if any((supplied_bundle, supplied_requests, supplied_provenance, supplied_db)):
                raise ValueError(
                    "--diagnostic-smollm-p2 cannot be combined with --bundle, --requests, --provenance, or --db")
            if args.config is None:
                raise ValueError("--config is required in diagnostic mode")
            if args.request is None:
                raise ValueError("--request is required in diagnostic mode")
            config_path = args.config
        else:
            if not supplied_bundle:
                raise ValueError("--bundle is required outside diagnostic mode; it binds config, requests, and provenance")
            if any((supplied_config, supplied_requests, supplied_provenance)):
                raise ValueError("--bundle cannot be combined with --config, --requests, or --provenance")
        try:
            if not args.diagnostic_smollm_p2:
                config_path, requests_path, provenance_path, provenance_digest = verify_bundle_inputs(
                    args.bundle, require_runtime_mount=True)
                args.requests = requests_path
                args.provenance = provenance_digest
                args.config = config_path
            config = load_config(config_path)
        except Exception as exc:
            print(json.dumps(_main_failure(exc, "config_load"), sort_keys=True))
            return 2
        if args.diagnostic_smollm_p2:
            if args.request is None:
                raise ValueError("--request is required in diagnostic mode")
            if args.result_file is not None:
                args.result_file = _diagnostic_result_file(args.result_file, config)
            if args.trace_file is not None:
                args.trace_file = _trace_file(args.trace_file, config)
            configure_trace(args.debug, args.trace_file)
            diagnostic = asyncio.run(_diagnostic_run(args, config))
            if args.result_file is not None:
                _write_diagnostic_result(args.result_file, diagnostic)
            print(json.dumps(diagnostic, sort_keys=True))
            return 0 if diagnostic["status"] == "complete" else 2
        if args.request is not None or args.selector != "smollm:context512":
            raise ValueError("--request and --selector require --diagnostic-smollm-p2")
        if args.db is None:
            raise ValueError("--db is required outside diagnostic mode")
        _validate_db_location(args.db, args.bundle, config)
        if args.result_file is not None:
            expected = Path(config.profile_db).parent / "measurement-result.json"
            if args.result_file != expected or args.result_file.is_symlink():
                raise ValueError("full measurement result file must use the fixed runtime location")
            _reject_symlink_components(args.result_file, "result file")
            if args.result_file.exists():
                raise ValueError("full measurement result file already exists")
            failure_result_file = args.result_file
        if args.trace_file is not None:
            args.trace_file = _trace_file(args.trace_file, config)
        configure_trace(args.debug, args.trace_file)
        if args.requests is None or args.provenance is None:
            raise ValueError("verified bundle inputs were not established")
        profiles = asyncio.run(_run(args, config))
        if args.result_file is not None:
            _write_measurement_result(args.result_file, profiles)
    except Exception as exc:
        failure = _main_failure(exc, "application")
        _trace_main_failure(failure)
        if failure_result_file is not None:
            _write_diagnostic_result(failure_result_file, failure)
        print(json.dumps(failure, sort_keys=True))
        return 2
    print(json.dumps({"status":"complete","matrix":profiles.matrix,
                      "profiles":[_profile_summary(identity) for identity in profiles]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

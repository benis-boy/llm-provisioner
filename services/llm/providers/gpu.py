"""Positive Linux NVML/procfs ownership evidence for one GPU supervisor."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
from pathlib import Path
import re
import threading
import time
from typing import Any


class GPUProofError(RuntimeError):
    """GPU or process ownership could not be proved."""


class _ResidencyPending(GPUProofError):
    """Ownership may become provable when a runner appears; never authority."""


class _ProcessUnconnected(Exception):
    """A non-supervisor NVML PID could not be connected to our supervisor."""


class _SupervisorUnconnected(GPUProofError):
    """The captured supervisor could not be re-read during an ancestry fence."""


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_time: int


@dataclass(frozen=True)
class ResidencyEvidence:
    gpu_uuid: str
    supervisor: ProcessIdentity
    runners: tuple[ProcessIdentity, ...]


@dataclass(frozen=True)
class GPUMemoryObservation:
    """One fenced, device-wide NVML memory point observation."""

    gpu_uuid: str
    supervisor: ProcessIdentity
    start_ns: int
    end_ns: int
    total_bytes: int
    used_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class OwnedOllamaSnapshot:
    """Current, fenced daemon topology; never GPU/NVML authority.

    The private loopback endpoint and this stable owned topology together
    attribute a model load.  Device memory remains supplemental evidence of a
    device effect, not per-process attribution.
    """

    supervisor: ProcessIdentity
    daemon: ProcessIdentity
    descendants: tuple[ProcessIdentity, ...]
    # A root broker is an additional, positively attested hop between the
    # unprivileged supervisor and daemon.  Kept last for old positional callers.
    broker: ProcessIdentity | None = None


@dataclass(frozen=True)
class _ProcessRecord:
    identity: ProcessIdentity
    parent: int


class _ForeignProcess(Exception):
    """A readable process was proved not to belong to the supervisor."""


_NVML_LOCK = threading.RLock()
_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_COMPUTE_APIS = ("nvmlDeviceGetComputeRunningProcesses_v3",
                 "nvmlDeviceGetComputeRunningProcesses_v2",
                 "nvmlDeviceGetComputeRunningProcesses")
_GRAPHICS_APIS = ("nvmlDeviceGetGraphicsRunningProcesses_v3",
                  "nvmlDeviceGetGraphicsRunningProcesses_v2",
                  "nvmlDeviceGetGraphicsRunningProcesses_v1",
                    "nvmlDeviceGetGraphicsRunningProcesses")
RESIDENCY_SETTLE_TIMEOUT_SECONDS = 5.0
RESIDENCY_SETTLE_INTERVAL_SECONDS = 0.2
_NO_RESIDENT_RUNNER = "GPU has no resident runner"
_RESIDENCY_DIAGNOSTIC_CATEGORIES = (
    "strict_supervisor_descendant", "foreign_or_baseline",
    "unreadable_or_unconnectable", "identity_or_ancestry_instability",
)


async def settle_residency(probe, *, timeout: float = RESIDENCY_SETTLE_TIMEOUT_SECONDS,
                           interval: float = RESIDENCY_SETTLE_INTERVAL_SECONDS,
                           sleep=asyncio.sleep, monotonic=time.monotonic) -> ResidencyEvidence:
    """Return a stable positive residency proof after bounded NVML appearance lag.

    Each successful probe still supplies the two-sample, nonempty ownership proof.
    Only typed absence/pending observations may settle: malformed, uncertain, or
    changed ownership evidence remains an immediate failure.
    """
    deadline = monotonic() + timeout
    while True:
        try:
            value = probe()
            return await value if inspect.isawaitable(value) else value
        except asyncio.CancelledError:
            raise
        except _ResidencyPending as exc:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise
            await sleep(min(interval, remaining))
        except GPUProofError:
            raise


def _uuid(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise GPUProofError("NVML returned a non-ASCII UUID") from exc
    if not isinstance(value, str) or not _UUID.fullmatch(value):
        raise GPUProofError("NVML returned an invalid GPU UUID")
    return value


def _unsupported(exc: Exception, backend: Any) -> bool:
    """Recognize only NVML's explicit missing/not-supported API errors."""
    types = tuple(x for name in ("NVMLError_FunctionNotFound", "NVMLError_NotSupported")
                  if isinstance(x := getattr(backend, name, None), type))
    return bool(types) and isinstance(exc, types)


class LinuxGPUProof:
    """Ownership proof for strict descendants of one supervisor PID.

    ``proc_root`` must be authoritative procfs in the same host PID namespace
    used by NVML. No host/container PID translation is attempted.
    """

    def __init__(self, target_uuid: str, supervisor_pid: int,
                 proc_root: Path = Path("/proc"), nvml: Any | None = None,
                 *, _supervisor: ProcessIdentity | None = None,
                 _baseline_empty: bool = False) -> None:
        self.target_uuid = _uuid(target_uuid)
        if type(supervisor_pid) is not int or supervisor_pid <= 0:
            raise ValueError("supervisor PID must be positive")
        root = Path(proc_root)
        if not root.is_absolute():
            raise ValueError("proc_root must be an absolute authoritative procfs")
        if not root.is_dir():
            raise GPUProofError("authoritative procfs is unavailable")
        if nvml is None:
            raise ValueError("an NVML backend is required")
        self.supervisor_pid, self.proc_root, self._nvml = supervisor_pid, root, nvml
        self._supervisor = _supervisor
        self._baseline_empty = _baseline_empty
        self._baseline_process_count = 0
        self._owned_identities: set[ProcessIdentity] = set()
        self._state_lock = threading.RLock()
        self._cleanup_reason = "not_checked"
        self._last_residency_diagnostic = {"exact_child": 0,
            "strict_supervisor_descendant": 0, "foreign_or_baseline": 0,
            "unreadable_or_unconnectable": 0, "identity_mismatch": 0}

    @classmethod
    def capture(cls, target_uuid: str, supervisor_pid: int,
                proc_root: Path = Path("/proc"), nvml: Any | None = None,
                *, host_pid_namespace: bool = False) -> "LinuxGPUProof":
        # Python cannot reliably discover host-PID provenance from an
        # arbitrary container. This explicit operator/bootstrap attestation is
        # required only for the real pynvml path; injected backends are a test
        # seam and must not be treated as production evidence.
        if nvml is None and host_pid_namespace is not True:
            raise GPUProofError("real NVML capture requires host PID namespace attestation")
        backend = nvml
        if backend is None:
            try:
                import pynvml as backend  # type: ignore[import-not-found]
            except ImportError as exc:
                raise GPUProofError("pynvml is required to capture GPU proof") from exc
        # Constructor validation is deliberately performed before any proc read.
        proof = cls(target_uuid, supervisor_pid, proc_root, backend)
        with proof._state_lock:
            # Install the exact supervisor identity before classifying a shared
            # baseline: _owned() must fence every ancestry observation against
            # that identity. Keep it installed through the final recheck so a
            # PID reuse cannot turn a baseline classification into a capture.
            supervisor = proof._read_supervisor_record(supervisor_pid).identity
            proof._supervisor = supervisor
            with proof._nvml_session() as device:
                proof._check_device(device)
                proof._assert_identity(supervisor)
                baseline = proof._process_pids(device)
                owned = proof._classify_pids(baseline)
                if owned:
                    raise GPUProofError("service-owned runner is already using the GPU")
                proof._assert_identity(supervisor)
                proof._baseline_process_count = min(len(baseline), 1024)
            proof._assert_identity(supervisor)
            proof._baseline_empty = True
        return proof

    class _Session:
        def __init__(self, owner: "LinuxGPUProof") -> None:
            self.owner = owner
        def __enter__(self):
            _NVML_LOCK.acquire()
            initialized = False
            try:
                self.owner._nvml.nvmlInit()
                initialized = True
                # Do not acquire a device handle when the physical exposure is
                # already outside the proof's exact-one-device contract.
                if self.owner._nvml.nvmlDeviceGetCount() != 1:
                    raise GPUProofError("exactly one physical GPU must be exposed")
                return self.owner._nvml.nvmlDeviceGetHandleByIndex(0)
            except BaseException:
                if initialized:
                    try:
                        self.owner._nvml.nvmlShutdown()
                    finally:
                        _NVML_LOCK.release()
                else:
                    _NVML_LOCK.release()
                raise
        def __exit__(self, typ, value, tb):
            try:
                self.owner._nvml.nvmlShutdown()
            finally:
                _NVML_LOCK.release()

    def _nvml_session(self) -> "_Session":
        return self._Session(self)

    def _check_device(self, device: Any) -> None:
        try:
            count = self._nvml.nvmlDeviceGetCount()
            if type(count) is not int or count != 1:
                raise GPUProofError("exactly one physical GPU must be exposed")
            if _uuid(self._nvml.nvmlDeviceGetUUID(device)) != self.target_uuid:
                raise GPUProofError("target GPU UUID changed or mismatched")
            mig = getattr(self._nvml, "nvmlDeviceGetMigMode", None)
            if not callable(mig):
                raise GPUProofError("NVML MIG status is unavailable")
            try:
                mode = mig(device)
            except Exception as exc:
                unsupported = getattr(self._nvml, "NVMLError_NotSupported", None)
                if not isinstance(unsupported, type) or not isinstance(exc, unsupported):
                    raise GPUProofError("NVML MIG status is unavailable") from exc
            else:
                if not isinstance(mode, (tuple, list)) or len(mode) < 2 or mode[0] != 0 or mode[1] != 0:
                    raise GPUProofError("MIG is enabled or its state is invalid")
        except GPUProofError:
            raise
        except Exception as exc:
            raise GPUProofError("NVML device identity is unavailable") from exc

    def _choose_process_api(self, names: tuple[str, ...], device: Any) -> list[Any]:
        for name in names:
            fn = getattr(self._nvml, name, None)
            if not callable(fn):
                continue
            try:
                values = fn(device)
            except Exception as exc:
                if _unsupported(exc, self._nvml):
                    continue
                raise GPUProofError(f"NVML process list {name} unavailable") from exc
            if not isinstance(values, (list, tuple)):
                raise GPUProofError(f"NVML process list {name} is malformed")
            return list(values)
        raise GPUProofError("required NVML process-list API is unavailable")

    def _process_pids(self, device: Any) -> set[int]:
        values = self._choose_process_api(_COMPUTE_APIS, device)
        values += self._choose_process_api(_GRAPHICS_APIS, device)
        result: set[int] = set()
        for item in values:
            pid = getattr(item, "pid", item)
            if type(pid) is not int or pid <= 0:
                raise GPUProofError("NVML returned an invalid process PID")
            result.add(pid)
        return result

    def _read_record(self, pid: int) -> _ProcessRecord:
        if type(pid) is not int or pid <= 0:
            raise GPUProofError("invalid process PID")
        try:
            with (self.proc_root / str(pid) / "stat").open("rb") as stream:
                raw = stream.read(4097)
        except FileNotFoundError as exc:
            raise _ProcessUnconnected from exc
        except OSError as exc:
            raise _ProcessUnconnected from exc
        if len(raw) > 4096:
            raise GPUProofError("process stat record exceeds bound")
        try:
            opening = raw.index(b"(")
            closing = raw.rfind(b")")
            actual_pid = int(raw[:opening].strip())
            fields = raw[closing + 2:].split()
            state = fields[0].decode("ascii")
            parent = int(fields[1])
            start = int(fields[19])
        except (ValueError, IndexError, UnicodeError) as exc:
            raise GPUProofError("malformed process stat") from exc
        if actual_pid != pid or state in {"Z", "X", "x"} or parent < 0 or start < 0:
            raise GPUProofError("invalid or dead process stat")
        return _ProcessRecord(ProcessIdentity(pid, start), parent)

    def _read_supervisor_record(self, pid: int) -> _ProcessRecord:
        """Read the supervisor through the public proof-error contract."""
        try:
            return self._read_record(pid)
        except _ProcessUnconnected as exc:
            raise GPUProofError("supervisor process record is unavailable") from exc

    def _assert_identity(self, expected: ProcessIdentity) -> None:
        if self._read_supervisor_record(expected.pid).identity != expected:
            raise GPUProofError("supervisor PID was reused or exited")

    def _candidate_record(self, pid: int, *, strict: bool = False) -> _ProcessRecord:
        try:
            return self._read_record(pid)
        except (GPUProofError, _ProcessUnconnected) as exc:
            if not strict and isinstance(exc, GPUProofError):
                raise _ProcessUnconnected from exc
            raise

    def _owned(self, leaf: int, *, strict: bool = False) -> ProcessIdentity:
        supervisor = self._supervisor
        if supervisor is None or leaf == self.supervisor_pid:
            raise GPUProofError("supervisor proof is unavailable or PID is not a runner")
        first: list[_ProcessRecord] = []
        current, seen = leaf, set()
        for _ in range(64):
            if current in seen:
                raise GPUProofError("process ancestry cycle")
            seen.add(current)
            record = self._candidate_record(current, strict=strict)
            first.append(record)
            if record.parent == self.supervisor_pid:
                break
            if record.parent <= 1:
                # A foreign result is useful only if the complete path which
                # led to it stayed foreign while it was being observed.  Do
                # not return at the ancestry root: a reused PID, a changed
                # parent, or a supervisor transition could otherwise make a
                # resident service descendant look foreign.
                try:
                    self._fence_ancestry(supervisor, first)
                except _ProcessUnconnected as exc:
                    raise _ProcessUnconnected from exc
                raise _ForeignProcess
            current = record.parent
        else:
            raise GPUProofError("process ancestry exceeds depth bound")
        self._fence_ancestry(supervisor, first)
        return first[0].identity

    def _fence_ancestry(self, supervisor: ProcessIdentity,
                        records: list[_ProcessRecord]) -> None:
        """Confirm the supervisor and every sampled ancestry record.

        Foreign classification needs the same snapshot fence as owned
        classification.  In particular, never let an ancestry root be the
        last read before declaring a process irrelevant to cleanup.
        """
        try:
            current_supervisor = self._read_record(self.supervisor_pid)
        except _ProcessUnconnected as exc:
            # Unlike a candidate record, uncertainty about the captured
            # supervisor invalidates the proof even if a later fence succeeds.
            raise _SupervisorUnconnected from exc
        if current_supervisor.identity != supervisor:
            raise GPUProofError("supervisor PID was reused or exited")
        for record in records:
            try:
                current = self._read_record(record.identity.pid)
            except (GPUProofError, _ProcessUnconnected) as exc:
                raise _ProcessUnconnected from exc
            if current != record:
                raise GPUProofError("process ancestry changed during observation")

    def _owned_or_foreign(self, leaf: int) -> ProcessIdentity | None:
        if leaf == self.supervisor_pid:
            return None
        try:
            return self._owned(leaf)
        except (_ForeignProcess, _ProcessUnconnected):
            return None

    def _classify_pids(self, pids: set[int]) -> set[ProcessIdentity]:
        """Return positively proven owned identities and bounded diagnostics.

        Diagnostics are deliberately category-only.  They are not used to
        accept a runner or to alter the ownership fence.
        """
        categories = {name: 0 for name in _RESIDENCY_DIAGNOSTIC_CATEGORIES}
        owned: set[ProcessIdentity] = set()
        instability: GPUProofError | None = None
        for pid in pids:
            if pid == self.supervisor_pid:
                raise GPUProofError("supervisor is using the GPU")
            try:
                # Generic residency keeps malformed/unreadable candidate
                # records non-terminal, as it did before diagnostics were
                # added; only a failed ancestry fence is instability.
                identity = self._owned(pid)
            except _ForeignProcess:
                categories["foreign_or_baseline"] = min(1024, categories["foreign_or_baseline"] + 1)
            except _ProcessUnconnected:
                categories["unreadable_or_unconnectable"] = min(1024, categories["unreadable_or_unconnectable"] + 1)
            except GPUProofError as exc:
                categories["identity_or_ancestry_instability"] = min(
                    1024, categories["identity_or_ancestry_instability"] + 1)
                instability = exc
            else:
                categories["strict_supervisor_descendant"] = min(
                    1024, categories["strict_supervisor_descendant"] + 1)
                owned.add(identity)
        self._last_residency_diagnostic = categories
        if instability is not None:
            raise instability
        return owned

    def _classify_expected_runner(self, pids: set[int], expected: ProcessIdentity) -> set[ProcessIdentity]:
        """Return only a fenced NVML record exactly equal to ``expected``.

        The category counters are failure diagnostics, never an authorization
        mechanism, and deliberately carry no PID or process metadata.
        """
        categories = {"exact_child": 0, "strict_supervisor_descendant": 0,
                      "foreign_or_baseline": 0, "unreadable_or_unconnectable": 0,
                      "identity_mismatch": 0}
        runners: set[ProcessIdentity] = set()
        for pid in pids:
            if pid == self.supervisor_pid:
                raise GPUProofError("supervisor is using the GPU")
            try:
                identity = self._owned(pid, strict=True)
            except _ForeignProcess:
                categories["foreign_or_baseline"] += 1
            except _ProcessUnconnected:
                categories["unreadable_or_unconnectable"] += 1
            else:
                if identity == expected:
                    categories["exact_child"] += 1
                    runners.add(identity)
                elif identity.pid == expected.pid:
                    categories["identity_mismatch"] += 1
                else:
                    categories["strict_supervisor_descendant"] += 1
        self._last_residency_diagnostic = categories
        return runners

    @staticmethod
    def _expected_runner_message(categories: dict[str, int]) -> str:
        return "GPU expected runner was not proved; " + ";".join(
            f"{name}={categories[name]}" for name in
            ("exact_child", "strict_supervisor_descendant", "foreign_or_baseline",
              "unreadable_or_unconnectable", "identity_mismatch"))

    @staticmethod
    def _residency_message(categories: dict[str, int]) -> str:
        return _NO_RESIDENT_RUNNER + "; " + ";".join(
            f"{name}={categories[name]}" for name in _RESIDENCY_DIAGNOSTIC_CATEGORIES)

    def _assert_expected_runner_categories(self) -> None:
        categories = self._last_residency_diagnostic
        if categories["strict_supervisor_descendant"] or categories["identity_mismatch"]:
            raise GPUProofError(self._expected_runner_message(categories))

    def _identity_sync(self) -> str:
        if self._supervisor is None:
            raise GPUProofError("GPU proof was not captured")
        self._assert_identity(self._supervisor)
        with self._nvml_session() as device:
            self._check_device(device)
        self._assert_identity(self._supervisor)
        return self.target_uuid

    @property
    def supervisor_identity(self) -> ProcessIdentity:
        """Immutable supervisor identity captured by this proof."""
        if self._supervisor is None or not self._baseline_empty:
            raise GPUProofError("GPU proof was not captured")
        return self._supervisor

    @property
    def baseline_process_count(self) -> int:
        """A bounded diagnostic count; baseline PIDs are intentionally not exposed."""
        if self._supervisor is None or not self._baseline_empty:
            raise GPUProofError("GPU proof was not captured")
        return self._baseline_process_count

    def _residency_sync(self) -> ResidencyEvidence:
        return self._residency_for_runner_sync(None)

    def _residency_for_runner_sync(self, expected_runner: ProcessIdentity | None) -> ResidencyEvidence:
        with self._state_lock:
            if self._supervisor is None or not self._baseline_empty:
                raise GPUProofError("GPU proof was not captured")
            self._assert_identity(self._supervisor)
            with self._nvml_session() as device:
                self._check_device(device)
                first_pids = self._process_pids(device)
                first_owned = (self._classify_pids(first_pids) if expected_runner is None
                               else self._classify_expected_runner(first_pids, expected_runner))
                if expected_runner is not None:
                    self._assert_expected_runner_categories()
                runners = tuple(sorted(first_owned, key=lambda x: x.pid))
                if not runners:
                    if expected_runner is not None:
                        categories = self._last_residency_diagnostic
                        raise _ResidencyPending(self._expected_runner_message(categories))
                    # A shared GPU commonly contains baseline processes. They
                    # are deliberately not authority, but neither do they
                    # make an owned runner's appearance terminal.
                    raise _ResidencyPending(self._residency_message(
                        self._last_residency_diagnostic))
                second_pids = self._process_pids(device)
                second_owned = (self._classify_pids(second_pids) if expected_runner is None
                                else self._classify_expected_runner(second_pids, expected_runner))
                if expected_runner is not None:
                    self._assert_expected_runner_categories()
                if second_owned != first_owned:
                    raise GPUProofError("owned GPU process set changed during ownership proof")
            self._assert_identity(self._supervisor)
            self._owned_identities.update(runners)
            return ResidencyEvidence(self.target_uuid, self._supervisor, runners)

    def _cleanup_sync(self) -> bool:
        with self._state_lock:
            if self._supervisor is None or not self._baseline_empty:
                self._cleanup_reason = "proof_not_captured"
                return False
            try:
                self._assert_identity(self._supervisor)
            except (GPUProofError, _ProcessUnconnected):
                # CleanupProbe is a boolean callback. A missing,
                # unreadable, malformed, or reused supervisor is an inability
                # to prove cleanup, never an exceptional positive result.
                self._cleanup_reason = "supervisor_identity_unavailable"
                return False
            try:
                with self._nvml_session() as device:
                    self._check_device(device)
                    current = self._process_pids(device)
                    first_owned = self._classify_pids(current)
                    second = self._process_pids(device)
                    second_owned = self._classify_pids(second)
                    if first_owned or second_owned:
                        self._cleanup_reason = "owned_runner_present"
                        return False
            except BaseException:
                # This is a boolean ownership callback.  In particular,
                # nvmlShutdown() runs when the context exits and may itself
                # fail after an otherwise clean sample.  That leaves cleanup
                # unproved, rather than making a caller treat an exception as
                # authority to proceed with replacement.
                self._cleanup_reason = "gpu_cleanup_probe_failed"
                return False
            try:
                # Fence the clean NVML sample as well as the initial one. In
                # particular, an empty/foreign-only sample after supervisor PID
                # reuse must not be reported as service cleanup.
                self._assert_identity(self._supervisor)
            except (GPUProofError, _ProcessUnconnected):
                self._cleanup_reason = "supervisor_identity_changed_after_probe"
                return False
            # Both observations completed with no positively-owned runner.
            self._cleanup_reason = "clean"
            return True

    @property
    def cleanup_reason(self) -> str:
        """Return a bounded, non-sensitive reason from the last cleanup probe."""
        with self._state_lock:
            return self._cleanup_reason

    def _memory_sync(self) -> GPUMemoryObservation:
        with self._state_lock:
            supervisor = self._supervisor
            if supervisor is None or not self._baseline_empty:
                raise GPUProofError("GPU proof was not captured")
            start_ns = time.monotonic_ns()
            try:
                self._assert_identity(supervisor)
                with self._nvml_session() as device:
                    self._check_device(device)
                    memory_fn = getattr(self._nvml, "nvmlDeviceGetMemoryInfo", None)
                    if not callable(memory_fn):
                        raise GPUProofError("NVML memory information API is unavailable")
                    try:
                        info = memory_fn(device)
                        total = getattr(info, "total")
                        used = getattr(info, "used")
                        free = getattr(info, "free")
                    except Exception as exc:
                        raise GPUProofError("NVML memory information is unavailable or malformed") from exc
                    values = (total, used, free)
                    if any(type(value) is not int for value in values):
                        raise GPUProofError("NVML memory fields must be integers")
                    if total <= 0 or used < 0 or free < 0 or used > total or free > total:
                        raise GPUProofError("NVML memory fields are out of range")
                    if used + free > total:
                        raise GPUProofError("NVML memory fields have an inconsistent sum")
                    self._check_device(device)
            except GPUProofError:
                raise
            except Exception as exc:
                raise GPUProofError("GPU memory observation failed") from exc
            self._assert_identity(supervisor)
            end_ns = time.monotonic_ns()
            return GPUMemoryObservation(self.target_uuid, supervisor, start_ns, end_ns,
                                        total, used, free)

    async def identity(self) -> str:
        return await asyncio.to_thread(self._identity_sync)

    async def residency(self) -> ResidencyEvidence:
        return await asyncio.to_thread(self._residency_sync)

    async def residency_for_runner(self, expected_runner: ProcessIdentity) -> ResidencyEvidence:
        if (type(expected_runner) is not ProcessIdentity or
                type(expected_runner.pid) is not int or expected_runner.pid <= 0 or
                type(expected_runner.start_time) is not int or expected_runner.start_time < 0):
            raise GPUProofError("expected runner identity is invalid")
        return await asyncio.to_thread(self._residency_for_runner_sync, expected_runner)

    @property
    def last_residency_diagnostic(self) -> dict[str, int]:
        """Bounded category counts from the latest residency probe."""
        with self._state_lock:
            return dict(self._last_residency_diagnostic)

    async def cleanup(self) -> bool:
        return await asyncio.to_thread(self._cleanup_sync)

    async def memory(self) -> GPUMemoryObservation:
        return await asyncio.to_thread(self._memory_sync)

"""Positive Linux NVML/procfs ownership evidence for one GPU supervisor."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
import threading
from typing import Any


class GPUProofError(RuntimeError):
    """GPU or process ownership could not be proved."""


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

    def _candidate_record(self, pid: int) -> _ProcessRecord:
        try:
            return self._read_record(pid)
        except (GPUProofError, _ProcessUnconnected) as exc:
            raise _ProcessUnconnected from exc

    def _owned(self, leaf: int) -> ProcessIdentity:
        supervisor = self._supervisor
        if supervisor is None or leaf == self.supervisor_pid:
            raise GPUProofError("supervisor proof is unavailable or PID is not a runner")
        first: list[_ProcessRecord] = []
        current, seen = leaf, set()
        for _ in range(64):
            if current in seen:
                raise GPUProofError("process ancestry cycle")
            seen.add(current)
            record = self._candidate_record(current)
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
        """Return only positively proven owned identities; unknown is ignored."""
        owned: set[ProcessIdentity] = set()
        for pid in pids:
            if pid == self.supervisor_pid:
                raise GPUProofError("supervisor is using the GPU")
            identity = self._owned_or_foreign(pid)
            if identity is not None:
                owned.add(identity)
        return owned

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
        with self._state_lock:
            if self._supervisor is None or not self._baseline_empty:
                raise GPUProofError("GPU proof was not captured")
            self._assert_identity(self._supervisor)
            with self._nvml_session() as device:
                self._check_device(device)
                first_owned = self._classify_pids(self._process_pids(device))
                runners = tuple(sorted(first_owned, key=lambda x: x.pid))
                if not runners:
                    raise GPUProofError("GPU has no resident runner")
                second_owned = self._classify_pids(self._process_pids(device))
                if second_owned != first_owned:
                    raise GPUProofError("owned GPU process set changed during ownership proof")
            self._assert_identity(self._supervisor)
            self._owned_identities.update(runners)
            return ResidencyEvidence(self.target_uuid, self._supervisor, runners)

    def _cleanup_sync(self) -> bool:
        with self._state_lock:
            if self._supervisor is None or not self._baseline_empty:
                return False
            try:
                self._assert_identity(self._supervisor)
            except (GPUProofError, _ProcessUnconnected):
                # CleanupProbe is a boolean callback. A missing,
                # unreadable, malformed, or reused supervisor is an inability
                # to prove cleanup, never an exceptional positive result.
                return False
            try:
                with self._nvml_session() as device:
                    self._check_device(device)
                    current = self._process_pids(device)
                    first_owned = self._classify_pids(current)
                    second = self._process_pids(device)
                    second_owned = self._classify_pids(second)
                    if first_owned or second_owned:
                        return False
            except GPUProofError:
                return False
            try:
                # Fence the clean NVML sample as well as the initial one. In
                # particular, an empty/foreign-only sample after supervisor PID
                # reuse must not be reported as service cleanup.
                self._assert_identity(self._supervisor)
            except (GPUProofError, _ProcessUnconnected):
                return False
            # Both observations completed with no positively-owned runner.
            return True

    async def identity(self) -> str:
        return await asyncio.to_thread(self._identity_sync)

    async def residency(self) -> ResidencyEvidence:
        return await asyncio.to_thread(self._residency_sync)

    async def cleanup(self) -> bool:
        return await asyncio.to_thread(self._cleanup_sync)

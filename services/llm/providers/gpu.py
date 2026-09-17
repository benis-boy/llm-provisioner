"""Fail-closed Linux NVML/procfs ownership evidence for one GPU supervisor."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
import threading
from typing import Any


class GPUProofError(RuntimeError):
    """GPU or process ownership could not be proved."""


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
        supervisor = proof._read_record(supervisor_pid).identity
        with proof._nvml_session() as device:
            proof._check_device(device)
            if proof._process_pids(device):
                raise GPUProofError("GPU baseline is not empty")
        proof._assert_identity(supervisor)
        proof._supervisor = supervisor
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
        except OSError as exc:
            raise GPUProofError("process identity is unreadable") from exc
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

    def _assert_identity(self, expected: ProcessIdentity) -> None:
        if self._read_record(expected.pid).identity != expected:
            raise GPUProofError("supervisor PID was reused or exited")

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
            record = self._read_record(current)
            first.append(record)
            if record.parent == self.supervisor_pid:
                break
            if record.parent <= 0:
                raise GPUProofError("runner is not a supervisor descendant")
            current = record.parent
        else:
            raise GPUProofError("process ancestry exceeds depth bound")
        if self._read_record(self.supervisor_pid).identity != supervisor:
            raise GPUProofError("supervisor PID was reused or exited")
        for record in first:
            if self._read_record(record.identity.pid) != record:
                raise GPUProofError("process ancestry changed during observation")
        return first[0].identity

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

    def _residency_sync(self) -> ResidencyEvidence:
        if self._supervisor is None or not self._baseline_empty:
            raise GPUProofError("GPU proof was not captured")
        self._assert_identity(self._supervisor)
        with self._nvml_session() as device:
            self._check_device(device)
            first = self._process_pids(device)
            if not first:
                raise GPUProofError("GPU has no resident runner")
            runners = tuple(sorted((self._owned(pid) for pid in first), key=lambda x: x.pid))
            second = self._process_pids(device)
        if second != first:
            raise GPUProofError("GPU process set changed during ownership proof")
        self._assert_identity(self._supervisor)
        return ResidencyEvidence(self.target_uuid, self._supervisor, runners)

    def _cleanup_sync(self) -> bool:
        if self._supervisor is None or not self._baseline_empty:
            return False
        self._assert_identity(self._supervisor)
        with self._nvml_session() as device:
            self._check_device(device)
            empty = not self._process_pids(device)
        self._assert_identity(self._supervisor)
        return empty

    async def identity(self) -> str:
        return await asyncio.to_thread(self._identity_sync)

    async def residency(self) -> ResidencyEvidence:
        return await asyncio.to_thread(self._residency_sync)

    async def cleanup(self) -> bool:
        return await asyncio.to_thread(self._cleanup_sync)

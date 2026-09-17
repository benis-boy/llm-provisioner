"""Synthetic procfs/NVML coverage for LinuxGPUProof (not run in this task)."""
from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from services.llm.providers.gpu import GPUProofError, LinuxGPUProof


def write_stat(root: Path, pid: int, parent: int, start: int = 1,
               state: str = "S", comm: bytes = b"runner") -> None:
    # comm may contain ')' and arbitrary bytes; stat parsing must use the last ')'.
    fields = [state.encode(), str(parent).encode()] + [b"0"] * 17 + [str(start).encode(), b"0"]
    directory = root / str(pid)
    directory.mkdir(exist_ok=True)
    (directory / "stat").write_bytes(str(pid).encode() + b" (" + comm + b") " + b" ".join(fields) + b"\n")


class NvmlNotSupported(Exception): pass
class NvmlFunctionNotFound(Exception): pass


class Process:
    def __init__(self, pid: int): self.pid = pid


class FakeNVML:
    NVMLError_NotSupported = NvmlNotSupported
    NVMLError_FunctionNotFound = NvmlFunctionNotFound

    def __init__(self):
        self.compute: list[int] = []
        self.graphics: list[int] = []
        self.uuid = b"GPU-test"
        self.count = 1
        self.mig = [0, 0]
        self.shutdowns = 0
        self.init_error = self.handle_error = None
        self.compute_v3_error = None

    def nvmlInit(self):
        if self.init_error: raise self.init_error
    def nvmlShutdown(self): self.shutdowns += 1
    def nvmlDeviceGetCount(self): return self.count
    def nvmlDeviceGetHandleByIndex(self, index):
        if self.handle_error: raise self.handle_error
        return object()
    def nvmlDeviceGetUUID(self, device): return self.uuid
    def nvmlDeviceGetMigMode(self, device):
        if isinstance(self.mig, Exception): raise self.mig
        return self.mig
    def nvmlDeviceGetComputeRunningProcesses_v3(self, device):
        if self.compute_v3_error: raise self.compute_v3_error
        return [Process(pid) for pid in self.compute]
    def nvmlDeviceGetComputeRunningProcesses_v2(self, device): raise NvmlFunctionNotFound()
    def nvmlDeviceGetComputeRunningProcesses(self, device): return [Process(pid) for pid in self.compute]
    def nvmlDeviceGetGraphicsRunningProcesses_v1(self, device): return [Process(pid) for pid in self.graphics]


class GPUProofTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_stat(self.root, 10, 0, 42)
        self.nvml = FakeNVML()
        self.proof = LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)

    def test_real_capture_requires_explicit_attestation_before_import(self):
        with mock.patch.dict("sys.modules", {"pynvml": None}):
            with self.assertRaises(GPUProofError):
                LinuxGPUProof.capture("GPU-test", 10, self.root)
        with self.assertRaises(GPUProofError):
            LinuxGPUProof.capture("GPU-test", 10, self.root, host_pid_namespace="true")

    def test_supervisor_identity_is_read_only_and_stable(self):
        self.assertEqual((10, 42), (self.proof.supervisor_identity.pid,
                                    self.proof.supervisor_identity.start_time))
        with self.assertRaises(AttributeError):
            self.proof.supervisor_identity = self.proof.supervisor_identity

    def tearDown(self): self.tmp.cleanup()

    async def test_deep_descendant_and_async_wrapper(self):
        write_stat(self.root, 11, 10, 2, comm=b"a)b")
        write_stat(self.root, 12, 11, 3)
        self.nvml.compute = [12]
        with mock.patch("services.llm.providers.gpu.asyncio.to_thread", wraps=asyncio.to_thread) as offload:
            evidence = await self.proof.residency()
        self.assertEqual((12,), tuple(x.pid for x in evidence.runners))
        self.assertTrue(offload.called)

    async def test_empty_residency_and_cleanup_active_then_empty(self):
        with self.assertRaises(GPUProofError): await self.proof.residency()
        write_stat(self.root, 11, 10)
        self.nvml.compute = [11]
        self.assertFalse(await self.proof.cleanup())
        self.nvml.compute = []
        self.assertTrue(await self.proof.cleanup())

    def test_constructor_and_stat_fail_closed(self):
        with self.assertRaises(ValueError): LinuxGPUProof("GPU-test", 0, self.root, self.nvml)
        with self.assertRaises(ValueError): LinuxGPUProof("GPU-test", 10, Path("relative"), self.nvml)
        write_stat(self.root, 10, 0, 42)
        (self.root / "10" / "stat").write_bytes(b"11 (x) S 0 " + b"0 " * 18 + b"1")
        with self.assertRaises(GPUProofError): LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)
        write_stat(self.root, 10, 0, 42, comm=b"bad\xff")
        # Non-ASCII comm is opaque and must not itself invalidate a record.
        self.assertEqual("GPU-test", LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml).target_uuid)

    def test_device_mig_and_balanced_handle_failures(self):
        for count, mig in ((2, [0, 0]), (1, [1, 0]), (1, [0, 1])):
            self.nvml.count, self.nvml.mig = count, mig
            with self.assertRaises(GPUProofError): LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)
            self.nvml.count, self.nvml.mig = 1, [0, 0]
        self.nvml.mig = NvmlNotSupported()
        LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)
        self.nvml.mig = NvmlFunctionNotFound()
        with self.assertRaises(GPUProofError): LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)
        self.nvml.mig = [0, 0]
        self.nvml.handle_error = RuntimeError("handle")
        with self.assertRaises(RuntimeError): LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)
        self.assertGreaterEqual(self.nvml.shutdowns, 1)

    def test_alias_and_graphics_errors(self):
        self.nvml.compute_v3_error = NvmlFunctionNotFound()
        LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)
        self.nvml.compute_v3_error = PermissionError("denied")
        with self.assertRaises(GPUProofError): LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)
        with mock.patch.object(self.nvml, "nvmlDeviceGetGraphicsRunningProcesses_v1", None):
            with self.assertRaises(GPUProofError): LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)

    async def test_foreign_pid_reuse_cycle_depth_and_races(self):
        write_stat(self.root, 20, 1)
        self.nvml.compute = [20]
        with self.assertRaises(GPUProofError): await self.proof.residency()
        write_stat(self.root, 20, 21)
        write_stat(self.root, 21, 20)
        with self.assertRaises(GPUProofError): await self.proof.residency()
        write_stat(self.root, 11, 10)
        self.nvml.compute = [11]
        original = self.proof._read_record
        calls = 0
        def race(pid):
            nonlocal calls
            calls += 1
            record = original(pid)
            if calls == 3: write_stat(self.root, 11, 10, 99)
            return record
        with mock.patch.object(self.proof, "_read_record", side_effect=race):
            with self.assertRaises(GPUProofError): await self.proof.residency()
        self.nvml.compute = []
        for pid in range(30, 96): write_stat(self.root, pid, pid - 1)
        write_stat(self.root, 30, 10)
        self.nvml.compute = [95]
        with self.assertRaises(GPUProofError): await self.proof.residency()

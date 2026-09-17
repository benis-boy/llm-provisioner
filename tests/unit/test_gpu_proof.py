"""Synthetic procfs/NVML coverage for LinuxGPUProof (not run in this task)."""
from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from services.llm.providers.gpu import (GPUProofError, LinuxGPUProof,
                                        ProcessIdentity, ResidencyEvidence,
                                        _ResidencyPending, settle_residency)


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
        self.init_error = self.handle_error = self.shutdown_error = None
        self.compute_v3_error = None

    def nvmlInit(self):
        if self.init_error: raise self.init_error
    def nvmlShutdown(self):
        self.shutdowns += 1
        if self.shutdown_error: raise self.shutdown_error
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

    async def test_residency_settlement_retries_pending_runner_observation(self):
        runner = ProcessIdentity(11, 2)
        calls, now = [], [0.0]
        async def probe():
            calls.append(None)
            if len(calls) < 3:
                raise _ResidencyPending("GPU has no resident runner")
            return ResidencyEvidence("GPU-test", self.proof.supervisor_identity, (runner,))
        async def sleep(delay): now[0] += delay
        evidence = await settle_residency(probe, timeout=1, interval=.2, sleep=sleep,
                                          monotonic=lambda: now[0])
        self.assertEqual((runner,), evidence.runners)
        self.assertEqual(3, len(calls))

    async def test_residency_settlement_persistent_pending_and_other_proof_errors_fail_closed(self):
        now = [0.0]
        async def empty(): raise _ResidencyPending("GPU has no resident runner")
        async def sleep(delay): now[0] += delay
        with self.assertRaisesRegex(GPUProofError, "no resident runner"):
            await settle_residency(empty, timeout=.4, interval=.2, sleep=sleep,
                                   monotonic=lambda: now[0])
        async def uncertain(): raise GPUProofError("supervisor PID was reused or exited")
        with self.assertRaisesRegex(GPUProofError, "supervisor PID"):
            await settle_residency(uncertain, timeout=1, sleep=sleep,
                                   monotonic=lambda: now[0])

    async def test_residency_settlement_propagates_cancellation(self):
        sleeping = asyncio.Event()
        async def empty(): raise _ResidencyPending("GPU has no resident runner")
        async def sleep(delay):
            sleeping.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(settle_residency(empty, sleep=sleep))
        await sleeping.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task

    async def test_shared_gpu_foreign_processes_are_filtered_and_churn_is_tolerated(self):
        write_stat(self.root, 20, 1)
        self.nvml.compute = [20]
        # Re-capture with a readable foreign baseline.
        proof = LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)
        self.assertEqual(1, proof.baseline_process_count)
        write_stat(self.root, 11, 10)
        write_stat(self.root, 12, 11)
        self.nvml.compute = [12, 30]
        write_stat(self.root, 30, 1)
        evidence = await proof.residency()
        self.assertEqual((12,), tuple(item.pid for item in evidence.runners))
        self.nvml.compute = [12]
        self.assertFalse(await proof.cleanup())
        self.nvml.compute = [30]
        self.assertTrue(await proof.cleanup())

    async def test_supervisor_nvml_pid_is_not_a_runner_and_blocks_cleanup(self):
        self.nvml.compute = [10]
        with self.assertRaises(GPUProofError):
            await self.proof.residency()
        self.assertFalse(await self.proof.cleanup())

    def test_capture_rejects_a_supervisor_gpu_baseline_pid(self):
        self.nvml.compute = [10]
        with self.assertRaises(GPUProofError):
            LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)

    def test_capture_reports_missing_supervisor_as_public_error(self):
        (self.root / "10" / "stat").unlink()
        with self.assertRaises(GPUProofError):
            LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)

    def test_capture_reports_unreadable_supervisor_as_public_error(self):
        (self.root / "10" / "stat").unlink()
        (self.root / "10" / "stat").mkdir()
        with self.assertRaises(GPUProofError):
            LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)

    def test_capture_accepts_persistent_absent_graphics_pid(self):
        self.nvml.graphics = [30]
        proof = LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)
        self.assertEqual(1, proof.baseline_process_count)

    def test_stable_readable_foreign_baseline_does_not_delay(self):
        write_stat(self.root, 20, 1)
        self.nvml.compute = [20]
        LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)

    async def test_stable_readable_foreign_residency_does_not_delay(self):
        write_stat(self.root, 11, 10)
        write_stat(self.root, 20, 1)
        self.nvml.compute = [11, 20]
        evidence = await self.proof.residency()
        self.assertEqual((11,), tuple(item.pid for item in evidence.runners))

    async def test_residency_pending_for_empty_or_foreign_only_nvml(self):
        with self.assertRaisesRegex(GPUProofError, "no resident runner"):
            await self.proof.residency()
        write_stat(self.root, 20, 1)
        self.nvml.compute = [20]
        with self.assertRaisesRegex(GPUProofError, "foreign_or_baseline=1"):
            await self.proof.residency()

    async def test_persistent_foreign_residency_reports_category_only_diagnostic(self):
        write_stat(self.root, 20, 1)
        self.nvml.compute = [20]
        now = [0.0]
        async def sleep(delay): now[0] += delay
        with self.assertRaisesRegex(GPUProofError,
                "GPU has no resident runner; strict_supervisor_descendant=0;"
                "foreign_or_baseline=1;unreadable_or_unconnectable=0;"
                "identity_or_ancestry_instability=0") as raised:
            await settle_residency(self.proof.residency, timeout=.4, interval=.2,
                                   sleep=sleep, monotonic=lambda: now[0])
        self.assertNotIn("20", str(raised.exception))
        self.assertEqual({"strict_supervisor_descendant": 0, "foreign_or_baseline": 1,
                          "unreadable_or_unconnectable": 0,
                          "identity_or_ancestry_instability": 0},
                         self.proof.last_residency_diagnostic)

    async def test_persistent_unreadable_residency_reports_category_only_diagnostic(self):
        self.nvml.compute = [30]
        now = [0.0]
        async def sleep(delay): now[0] += delay
        with self.assertRaisesRegex(GPUProofError,
                "GPU has no resident runner; strict_supervisor_descendant=0;"
                "foreign_or_baseline=0;unreadable_or_unconnectable=1;"
                "identity_or_ancestry_instability=0") as raised:
            await settle_residency(self.proof.residency, timeout=.4, interval=.2,
                                   sleep=sleep, monotonic=lambda: now[0])
        self.assertNotIn("30", str(raised.exception))

    async def test_expected_runner_requires_exact_identity_and_reports_only_categories(self):
        write_stat(self.root, 11, 10, 2)
        write_stat(self.root, 12, 10, 3)
        write_stat(self.root, 20, 1, 4)
        self.nvml.compute = [11]
        evidence = await self.proof.residency_for_runner(ProcessIdentity(11, 2))
        self.assertEqual((ProcessIdentity(11, 2),), evidence.runners)
        self.nvml.compute = [11, 12, 20, 30]
        with self.assertRaisesRegex(GPUProofError, "exact_child=0;strict_supervisor_descendant=1;foreign_or_baseline=1;unreadable_or_unconnectable=1;identity_mismatch=1") as raised:
            await self.proof.residency_for_runner(ProcessIdentity(11, 99))
        self.assertNotIn("11", str(raised.exception))
        self.assertEqual({"exact_child": 0, "strict_supervisor_descendant": 1,
                           "foreign_or_baseline": 1, "unreadable_or_unconnectable": 1,
                           "identity_mismatch": 1}, self.proof.last_residency_diagnostic)

    async def test_expected_runner_rejects_exact_child_with_strict_extra_immediately(self):
        write_stat(self.root, 11, 10, 2)
        write_stat(self.root, 12, 10, 3)
        calls = 0

        def compute(device):
            nonlocal calls
            calls += 1
            return [Process(11), Process(12)]

        with mock.patch.object(self.nvml, "nvmlDeviceGetComputeRunningProcesses_v3",
                               side_effect=compute):
            with self.assertRaisesRegex(GPUProofError, "exact_child=1;strict_supervisor_descendant=1"):
                await self.proof.residency_for_runner(ProcessIdentity(11, 2))
        self.assertEqual(1, calls)

    async def test_expected_runner_rejects_strict_extra_on_second_sample(self):
        write_stat(self.root, 11, 10, 2)
        write_stat(self.root, 12, 10, 3)
        samples = iter(([Process(11)], [Process(11), Process(12)]))

        with mock.patch.object(self.nvml, "nvmlDeviceGetComputeRunningProcesses_v3",
                               side_effect=lambda device: next(samples)):
            with self.assertRaisesRegex(GPUProofError, "exact_child=1;strict_supervisor_descendant=1"):
                await self.proof.residency_for_runner(ProcessIdentity(11, 2))

    async def test_expected_runner_empty_nvml_settles_then_proves_exact_child(self):
        write_stat(self.root, 11, 10, 2)
        runner = ProcessIdentity(11, 2)
        now, sleeps = [0.0], []

        async def sleep(delay):
            sleeps.append(delay)
            now[0] += delay

        with mock.patch.object(self.nvml, "nvmlDeviceGetComputeRunningProcesses_v3",
                               side_effect=([], [Process(11)], [Process(11)])):
            evidence = await settle_residency(
                lambda: self.proof.residency_for_runner(runner),
                timeout=1, interval=.2, sleep=sleep, monotonic=lambda: now[0])

        self.assertEqual((runner,), evidence.runners)
        self.assertEqual([.2], sleeps)

    async def test_expected_runner_foreign_only_is_retryable_until_timeout(self):
        write_stat(self.root, 20, 1)
        self.nvml.compute = [20]
        calls = []

        async def probe():
            calls.append(None)
            return await self.proof.residency_for_runner(ProcessIdentity(11, 2))

        now = [0.0]
        async def sleep(delay): now[0] += delay
        with self.assertRaisesRegex(GPUProofError, "expected runner was not proved"):
            await settle_residency(probe, timeout=.4, interval=.2, sleep=sleep,
                                   monotonic=lambda: now[0])
        self.assertEqual(3, len(calls))

    async def test_foreign_baseline_does_not_block_delayed_owned_runner(self):
        write_stat(self.root, 20, 1)
        write_stat(self.root, 11, 10, 2)
        self.nvml.compute = [20]
        samples = iter(([Process(20)], [Process(20)], [Process(11), Process(20)],
                        [Process(11), Process(20)]))
        with mock.patch.object(self.nvml, "nvmlDeviceGetComputeRunningProcesses_v3",
                               side_effect=lambda device: next(samples)):
            now = [0.0]
            async def sleep(delay): now[0] += delay
            evidence = await settle_residency(self.proof.residency, timeout=1, interval=.2,
                                               sleep=sleep, monotonic=lambda: now[0])
        self.assertEqual((11,), tuple(item.pid for item in evidence.runners))

    async def test_expected_runner_rejects_malformed_identity_without_sampling(self):
        with self.assertRaisesRegex(GPUProofError, "expected runner identity is invalid"):
            await self.proof.residency_for_runner(ProcessIdentity(0, 1))

    async def test_residency_ignores_absent_unreadable_and_malformed_extra_pids(self):
        write_stat(self.root, 11, 10)
        (self.root / "31").mkdir()
        (self.root / "31" / "stat").write_bytes(b"bad")
        self.nvml.compute = [11, 30, 31, 32]
        evidence = await self.proof.residency()
        self.assertEqual((11,), tuple(item.pid for item in evidence.runners))

    async def test_cleanup_true_with_unknown_only_pid(self):
        self.nvml.compute = [30]
        self.assertTrue(await self.proof.cleanup())

    async def test_cleanup_false_with_proven_runner(self):
        write_stat(self.root, 11, 10)
        self.nvml.compute = [11]
        self.assertFalse(await self.proof.cleanup())

    async def test_cleanup_fails_closed_when_nvml_shutdown_is_uncertain(self):
        self.nvml.shutdown_error = RuntimeError("backend shutdown failed")
        self.assertFalse(await self.proof.cleanup())
        self.assertEqual("gpu_cleanup_probe_failed", self.proof.cleanup_reason)

    async def test_residency_rejects_unknown_to_owned_transition(self):
        write_stat(self.root, 11, 10)
        samples = iter(([Process(11), Process(30)], [Process(11), Process(30)]))
        # The extra PID is initially absent, then becomes our strict descendant.
        def compute(device):
            values = next(samples)
            if values[0].pid == 11 and len(values) == 2 and not hasattr(compute, "changed"):
                compute.changed = True
                return values
            write_stat(self.root, 30, 10)
            return values
        with mock.patch.object(self.nvml, "nvmlDeviceGetComputeRunningProcesses_v3",
                               side_effect=compute):
            with self.assertRaisesRegex(GPUProofError, "owned GPU process set changed"):
                await self.proof.residency()

    async def test_supervisor_unreadable_or_malformed_does_not_mean_gone(self):
        (self.root / "10" / "stat").unlink()
        self.assertFalse(await self.proof.cleanup())
        write_stat(self.root, 10, 0)
        (self.root / "10" / "stat").write_bytes(b"malformed")
        self.assertFalse(await self.proof.cleanup())

    async def test_identity_reports_missing_supervisor_as_public_error(self):
        (self.root / "10" / "stat").unlink()
        with self.assertRaises(GPUProofError):
            await self.proof.identity()

    async def test_identity_reports_unreadable_supervisor_as_public_error(self):
        (self.root / "10" / "stat").unlink()
        (self.root / "10" / "stat").mkdir()
        with self.assertRaises(GPUProofError):
            await self.proof.identity()

    async def test_residency_reports_missing_supervisor_as_public_error(self):
        (self.root / "10" / "stat").unlink()
        with self.assertRaises(GPUProofError):
            await self.proof.residency()

    async def test_residency_reports_unreadable_supervisor_as_public_error(self):
        (self.root / "10" / "stat").unlink()
        (self.root / "10" / "stat").mkdir()
        with self.assertRaises(GPUProofError):
            await self.proof.residency()

    async def test_cleanup_revalidates_supervisor_after_clean_nvml_sample(self):
        with mock.patch.object(self.proof, "_assert_identity",
                               side_effect=(None, GPUProofError("reused"))):
            self.assertFalse(await self.proof.cleanup())

    async def test_supervisor_loss_fails_closed_for_unrecorded_current_pid(self):
        write_stat(self.root, 11, 10)
        self.nvml.compute = [11]
        await self.proof.residency()
        (self.root / "10" / "stat").unlink()
        write_stat(self.root, 30, 1)
        self.nvml.compute = [30]
        self.assertFalse(await self.proof.cleanup())

    async def test_residency_tolerates_new_foreign_pid_that_disappears(self):
        write_stat(self.root, 11, 10)
        write_stat(self.root, 20, 1)
        self.nvml.compute = [11, 20]
        calls = 0
        def compute(device):
            nonlocal calls
            calls += 1
            if calls == 1:
                (self.root / "20" / "stat").unlink()
                return [Process(11), Process(20)]
            return [Process(11)]
        with mock.patch.object(self.nvml, "nvmlDeviceGetComputeRunningProcesses_v3",
                               side_effect=compute):
            evidence = await self.proof.residency()
        self.assertEqual((11,), tuple(item.pid for item in evidence.runners))

    async def test_foreign_numeric_churn_does_not_change_owned_residency(self):
        write_stat(self.root, 11, 10)
        write_stat(self.root, 20, 1)
        write_stat(self.root, 30, 1)
        samples = iter(([Process(11), Process(20)], [Process(11), Process(30)]))
        with mock.patch.object(self.nvml, "nvmlDeviceGetComputeRunningProcesses_v3",
                               side_effect=lambda device: next(samples)):
            evidence = await self.proof.residency()
        self.assertEqual((11,), tuple(item.pid for item in evidence.runners))

    async def test_residency_rejects_a_previously_owned_pid_that_disappears(self):
        write_stat(self.root, 11, 10)
        self.nvml.compute = [11]
        await self.proof.residency()
        (self.root / "11" / "stat").unlink()
        with self.assertRaises(GPUProofError):
            await self.proof.residency()

    async def test_cleanup_tolerates_new_foreign_pid_that_disappears(self):
        write_stat(self.root, 20, 1)
        calls = 0
        def compute(device):
            nonlocal calls
            calls += 1
            if calls == 1:
                (self.root / "20" / "stat").unlink()
                return [Process(20)]
            return []
        with mock.patch.object(self.nvml, "nvmlDeviceGetComputeRunningProcesses_v3",
                               side_effect=compute):
            self.assertTrue(await self.proof.cleanup())

    async def test_foreign_numeric_churn_does_not_block_cleanup(self):
        write_stat(self.root, 20, 1)
        write_stat(self.root, 30, 1)
        samples = iter(([Process(20)], [Process(30)]))
        with mock.patch.object(self.nvml, "nvmlDeviceGetComputeRunningProcesses_v3",
                               side_effect=lambda device: next(samples)):
            self.assertTrue(await self.proof.cleanup())

    async def test_foreign_leaf_start_time_reuse_fails_cleanup_closed(self):
        write_stat(self.root, 20, 1, 1)
        self.nvml.compute = [20]
        original = self.proof._read_record
        leaf_reads = 0

        def reread_reused_leaf(pid):
            nonlocal leaf_reads
            if pid == 20:
                leaf_reads += 1
                if leaf_reads == 2:
                    write_stat(self.root, 20, 1, 99)
            return original(pid)

        with mock.patch.object(self.proof, "_read_record", side_effect=reread_reused_leaf):
            self.assertFalse(await self.proof.cleanup())

    async def test_foreign_ancestry_mutation_into_supervisor_descendant_fails_cleanup_closed(self):
        write_stat(self.root, 20, 21, 2)
        write_stat(self.root, 21, 1, 3)
        self.nvml.compute = [20]
        original = self.proof._read_record
        ancestor_reads = 0

        def reread_mutated_ancestor(pid):
            nonlocal ancestor_reads
            if pid == 21:
                ancestor_reads += 1
                if ancestor_reads == 2:
                    write_stat(self.root, 21, 10, 3)
            return original(pid)

        with mock.patch.object(self.proof, "_read_record", side_effect=reread_mutated_ancestor):
            self.assertFalse(await self.proof.cleanup())

    async def test_supervisor_unreadable_during_foreign_fence_fails_cleanup_closed(self):
        write_stat(self.root, 20, 1)
        self.nvml.compute = [20]
        original = self.proof._read_record
        supervisor_reads = 0

        def unreadable_supervisor_during_fence(pid):
            nonlocal supervisor_reads
            if pid == 10:
                supervisor_reads += 1
                if supervisor_reads == 2:
                    stat = self.root / "10" / "stat"
                    stat.unlink()
                    try:
                        return original(pid)
                    finally:
                        write_stat(self.root, 10, 0, 42)
            return original(pid)

        with mock.patch.object(self.proof, "_read_record",
                               side_effect=unreadable_supervisor_during_fence):
            self.assertFalse(await self.proof.cleanup())

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

    async def test_residency_reclassifies_an_unchanged_pid_set(self):
        write_stat(self.root, 11, 10, 2)
        self.nvml.compute = [11]
        original = self.proof._process_pids
        calls = 0
        def same_pid_set(device):
            nonlocal calls
            calls += 1
            if calls == 2:
                write_stat(self.root, 11, 10, 99)
            return original(device)
        with mock.patch.object(self.proof, "_process_pids", side_effect=same_pid_set):
            with self.assertRaises(GPUProofError):
                await self.proof.residency()

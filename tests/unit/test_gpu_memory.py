"""Bounded, identity-fenced GPU memory observation coverage."""
from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.llm.providers.gpu import GPUProofError, LinuxGPUProof
from tests.unit.test_gpu_proof import FakeNVML, write_stat


class GPUMemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_stat(self.root, 10, 0, 42)
        self.nvml = FakeNVML()
        self.nvml.memory = SimpleNamespace(total=1000, used=400, free=600)
        self.nvml.nvmlDeviceGetMemoryInfo = lambda device: self.nvml.memory
        self.proof = LinuxGPUProof.capture("GPU-test", 10, self.root, self.nvml)

    def tearDown(self):
        self.tmp.cleanup()

    async def test_valid_zero_full_and_reserved_observations(self):
        for values in ((1000, 0, 1000), (1000, 1000, 0), (1000, 400, 500)):
            with self.subTest(values=values):
                self.nvml.memory = SimpleNamespace(total=values[0], used=values[1], free=values[2])
                observation = await self.proof.memory()
                self.assertEqual(values, (observation.total_bytes, observation.used_bytes,
                                          observation.free_bytes))
                self.assertLessEqual(observation.start_ns, observation.end_ns)
                self.assertEqual("GPU-test", observation.gpu_uuid)

    async def test_all_malformed_memory_fields_fail_closed_with_one_shutdown(self):
        cases = ((True, 1, 1), (1000, True, 1), (1000, 1, True),
                 (1000.0, 1, 1), (1000, 1.0, 1), (1000, 1, 1.0),
                 (-1, 0, 0), (0, 0, 0), (1000, -1, 0), (1000, 0, -1),
                 (1000, 1001, 0), (1000, 0, 1001), (1000, 1, 1000))
        for values in cases:
            with self.subTest(values=values):
                self.nvml.memory = SimpleNamespace(total=values[0], used=values[1], free=values[2])
                before = self.nvml.shutdowns
                with self.assertRaises(GPUProofError):
                    await self.proof.memory()
                self.assertEqual(1, self.nvml.shutdowns - before)

    async def test_memory_api_missing_field_and_read_errors_are_typed_and_balanced(self):
        cases = (
            lambda: SimpleNamespace(total=1000, used=1),
            lambda: (_ for _ in ()).throw(RuntimeError("read")),
        )
        for read in cases:
            with self.subTest(read=read):
                self.nvml.nvmlDeviceGetMemoryInfo = lambda device, read=read: read()
                before = self.nvml.shutdowns
                with self.assertRaises(GPUProofError):
                    await self.proof.memory()
                self.assertEqual(1, self.nvml.shutdowns - before)
        del self.nvml.nvmlDeviceGetMemoryInfo
        before = self.nvml.shutdowns
        with self.assertRaises(GPUProofError):
            await self.proof.memory()
        self.assertEqual(1, self.nvml.shutdowns - before)

    async def test_init_handle_and_memory_read_failures_are_typed_with_exact_shutdown_deltas(self):
        for attribute, value, expected_delta in (
                ("init_error", RuntimeError("init"), 0),
                ("handle_error", RuntimeError("handle"), 1)):
            with self.subTest(attribute=attribute):
                before = self.nvml.shutdowns
                setattr(self.nvml, attribute, value)
                with self.assertRaises(GPUProofError):
                    await self.proof.memory()
                self.assertEqual(expected_delta, self.nvml.shutdowns - before)
                setattr(self.nvml, attribute, None)

    async def test_memory_read_is_off_loop(self):
        caller_thread, worker_threads = threading.get_ident(), []
        def read(device):
            worker_threads.append(threading.get_ident())
            return self.nvml.memory
        with patch("services.llm.providers.gpu.asyncio.to_thread", wraps=asyncio.to_thread) as offload:
            self.nvml.nvmlDeviceGetMemoryInfo = read
            await self.proof.memory()
        self.assertTrue(offload.called)
        self.assertEqual(1, len(worker_threads))
        self.assertNotEqual(caller_thread, worker_threads[0])

    async def test_uncaptured_and_pre_read_identity_changes_do_not_read_memory(self):
        uncaptured = LinuxGPUProof("GPU-test", 10, self.root, self.nvml)
        cases = (("uncaptured", lambda: None),
                 ("missing_supervisor", lambda: (self.root / "10" / "stat").unlink()),
                 ("reused_supervisor", lambda: write_stat(self.root, 10, 0, 99)),
                 ("uuid", lambda: setattr(self.nvml, "uuid", b"GPU-other")),
                 ("count", lambda: setattr(self.nvml, "count", 2)),
                 ("mig", lambda: setattr(self.nvml, "mig", [1, 0])))
        for name, change in cases:
            with self.subTest(change=name):
                if name == "uncaptured":
                    proof = uncaptured
                else:
                    change()
                    proof = self.proof
                with patch.object(self.nvml, "nvmlDeviceGetMemoryInfo", side_effect=AssertionError()) as read:
                    with self.assertRaises(GPUProofError):
                        await proof.memory()
                    read.assert_not_called()
                # Restore the fixture without globally stopping unrelated patches.
                write_stat(self.root, 10, 0, 42)
                self.nvml.uuid, self.nvml.count, self.nvml.mig = b"GPU-test", 1, [0, 0]

    async def test_post_read_identity_changes_fail_closed_with_one_shutdown(self):
        changes = (
            ("missing_supervisor", lambda root, nvml: (root / "10" / "stat").unlink()),
            ("reused_supervisor", lambda root, nvml: write_stat(root, 10, 0, 99)),
            ("uuid", lambda root, nvml: setattr(nvml, "uuid", b"GPU-other")),
            ("count", lambda root, nvml: setattr(nvml, "count", 2)),
            ("mig", lambda root, nvml: setattr(nvml, "mig", [1, 0])),
        )
        for name, change in changes:
            with self.subTest(change=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_stat(root, 10, 0, 42)
                nvml = FakeNVML()
                nvml.memory = SimpleNamespace(total=1000, used=400, free=600)
                proof = LinuxGPUProof.capture("GPU-test", 10, root, nvml)
                nvml.nvmlDeviceGetMemoryInfo = lambda device: (change(root, nvml), nvml.memory)[1]
                before = nvml.shutdowns
                with self.assertRaises(GPUProofError):
                    await proof.memory()
                self.assertEqual(1, nvml.shutdowns - before)

    async def test_timestamp_interval_orders_both_identity_fences(self):
        events, ticks = [], iter((10, 20))
        original_fence = self.proof._assert_identity
        def fenced(identity):
            events.append("fence")
            original_fence(identity)
        def clock():
            events.append("clock")
            return next(ticks)
        self.proof._assert_identity = fenced
        with patch("services.llm.providers.gpu.time.monotonic_ns", side_effect=clock):
            observation = await self.proof.memory()
        self.assertEqual((10, 20), (observation.start_ns, observation.end_ns))
        self.assertEqual(["clock", "fence", "fence", "clock"], events)

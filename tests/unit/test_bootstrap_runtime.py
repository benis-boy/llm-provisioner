"""Focused composition seams; all proofs and lifecycle objects are test-owned."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from services.llm.bootstrap.runtime import BootstrapRuntime, RuntimeOptions
from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import ProcessIdentity


class _LinuxProof:
    def __init__(self):
        self.supervisor_identity = ProcessIdentity(os.getpid(), 1)
        self.residency_for_runner = self._residency_for_runner
        self.memory = None

    async def identity(self):
        return "GPU-test"

    async def cleanup(self):
        return True

    async def residency(self):
        raise AssertionError("health must not make a provider resident")

    async def _residency_for_runner(self, runner):
        raise AssertionError("test proof callback must not be invoked here")


class RuntimeOptionsTests(unittest.TestCase):
    def test_requires_explicit_bounded_operator_options(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            with self.assertRaises(ValueError):
                RuntimeOptions(Path("relative"), path)
            with self.assertRaises(ValueError):
                RuntimeOptions(path, path, port=0)
            with self.assertRaises(ValueError):
                RuntimeOptions(path, path, shutdown_grace_seconds=0)
            self.assertFalse(RuntimeOptions(path, path).host_pid_namespace)


class RuntimeProofTests(unittest.IsolatedAsyncioTestCase):
    async def test_linux_capture_is_converted_to_typed_provider_proof(self):
        runtime = object.__new__(BootstrapRuntime)
        runtime.proof = _LinuxProof()
        typed = runtime._typed_proof()
        self.assertIs(type(typed), GPUProof)
        self.assertEqual(typed.expected_supervisor, runtime.proof.supervisor_identity)
        self.assertIs(typed.residency_for_runner, runtime.proof.residency_for_runner)
        self.assertEqual(await typed.identity(), "GPU-test")
        self.assertTrue(await typed.cleanup())

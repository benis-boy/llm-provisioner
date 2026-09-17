"""Unit coverage for the fixed root broker's public configuration constants."""
from __future__ import annotations

import importlib.util
import asyncio
import os
from pathlib import Path
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).parents[2]
MODULE = ROOT / "services/llm/bootstrap/ollama_broker.py"


class FixedBrokerConfigTests(unittest.TestCase):
    def load_broker(self):
        bootstrap = types.ModuleType("services.llm.bootstrap.config")
        providers_config = types.ModuleType("services.llm.providers.config")
        providers_gpu = types.ModuleType("services.llm.providers.gpu")
        supervisor = types.ModuleType("services.llm.bootstrap.supervisor")

        class ModelConfig:
            def __init__(self, runtime, adapter):
                self.runtime, self.adapter = runtime, adapter

        class BootstrapConfig:
            def __init__(self, *args):
                (self.profile, self.artifact_root, self.manifest_digest,
                 self.profiles_db, self.ollama_binary, self.ollama_data,
                 self.ollama_port, self.models) = args

        bootstrap.BootstrapConfig = BootstrapConfig
        bootstrap.ModelConfig = ModelConfig
        providers_config.GPUProof = type("GPUProof", (), {})
        providers_gpu.ProcessIdentity = type("ProcessIdentity", (), {})
        supervisor.OwnedOllama = type("OwnedOllama", (), {})
        modules = {
            "services.llm.bootstrap.config": bootstrap,
            "services.llm.providers.config": providers_config,
            "services.llm.providers.gpu": providers_gpu,
            "services.llm.bootstrap.supervisor": supervisor,
        }
        old_path = sys.path[:]
        try:
            with patch.dict(sys.modules, modules):
                spec = importlib.util.spec_from_file_location("test_ollama_broker", MODULE)
                broker = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(broker)
                return broker
        finally:
            sys.path[:] = old_path

    def test_config_is_fixed_to_pinned_version_and_paths(self):
        broker = self.load_broker()
        config = broker._config()
        self.assertEqual("GPU-broker", config.profile)
        self.assertEqual(Path("/var/lib/ollama"), config.artifact_root)
        self.assertEqual(Path("/var/lib/ollama/profiles.sqlite"), config.profiles_db)
        self.assertEqual(Path("/usr/local/bin/ollama"), config.ollama_binary)
        self.assertEqual(Path("/var/lib/ollama"), config.ollama_data)
        self.assertEqual(11434, config.ollama_port)
        self.assertEqual("ollama:0.11.6", config.models["SmolLM"].runtime)
        self.assertEqual("broker:none", config.models["CoEdIT"].runtime)
        self.assertEqual("broker:none", config.models["GECToR"].runtime)

    def test_protocol_line_limit_is_fixed(self):
        broker = self.load_broker()
        self.assertEqual(4096, broker.MAX_LINE)

    def test_protocol_parser_rejects_partial_duplicates_and_unproved_inputs(self):
        broker = self.load_broker()
        self.assertEqual("start", broker.parse_command(b'{"command":"start"}\n'))
        for frame in (b'{"command":"start"}', b'{"command":"start","command":"ping"}\n',
                      b'{"command":"health"}\n', b'{}\n'):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                broker.parse_command(frame)

    def test_read_command_returns_eof_after_a_fragment(self):
        broker = self.load_broker()
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b'{"command":"sta')
            os.close(write_fd)
            write_fd = -1
            self.assertIsNone(asyncio.run(broker.read_command(read_fd, asyncio.Event())))
        finally:
            os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)

    def test_serve_eof_during_gated_start_cancels_operation(self):
        asyncio.run(self._serve_eof_during_gated_start_cancels_operation())

    async def _serve_eof_during_gated_start_cancels_operation(self):
        broker = self.load_broker()

        class Daemon:
            def __init__(self):
                self.started = asyncio.Event()
                self.cancelled = False
                self.closed = False
                identity = SimpleNamespace(pid=2, start_time=2)
                self._fence = SimpleNamespace(identity=identity)

            async def start(self):
                self.started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise

            async def close(self):
                self.closed = True

            async def alive(self):
                return True

            def _members(self, allow_live):
                return []

        daemon = Daemon()
        input_read, input_write = os.pipe()
        output_read, output_write = os.pipe()
        stopping = asyncio.Event()
        try:
            with patch.object(broker, "_identity", return_value=SimpleNamespace(pid=1, start_time=1)):
                task = asyncio.create_task(broker.serve(daemon, input_read, output_write, stopping))
                os.write(input_write, b'{"command":"start"}\n')
                # Let the gated start and its EOF watcher reach the event loop.
                while not daemon.started.is_set():
                    await asyncio.sleep(0)
                os.close(input_write)
                input_write = -1
                await asyncio.wait_for(task, 1)
            self.assertTrue(daemon.cancelled)
            self.assertTrue(daemon.closed)
        finally:
            os.close(input_read)
            os.close(output_read)
            os.close(output_write)
            if input_write >= 0:
                os.close(input_write)

    def test_serve_detects_idle_daemon_loss(self):
        asyncio.run(self._serve_detects_idle_daemon_loss())

    async def _serve_detects_idle_daemon_loss(self):
        broker = self.load_broker()

        class Daemon:
            def __init__(self):
                self.calls = 0
                self.closed = False
                identity = SimpleNamespace(pid=2, start_time=2)
                self._fence = SimpleNamespace(identity=identity)

            async def start(self):
                return "0.11.6"

            async def alive(self):
                self.calls += 1
                return self.calls == 1

            async def health(self):
                return "0.11.6"

            def _members(self, allow_live):
                return []

            async def close(self):
                self.closed = True

        daemon = Daemon()
        input_read, input_write = os.pipe()
        output_read, output_write = os.pipe()
        stopping = asyncio.Event()
        try:
            with patch.object(broker, "_identity", return_value=SimpleNamespace(pid=1, start_time=1)):
                task = asyncio.create_task(broker.serve(daemon, input_read, output_write, stopping))
                os.write(input_write, b'{"command":"start"}\n')
                response = await asyncio.to_thread(os.read, output_read, 4096)
                self.assertIn(b'"ok":true', response)
                with self.assertRaises(RuntimeError):
                    await asyncio.wait_for(task, 1)
            self.assertTrue(daemon.closed)
        finally:
            for fd in (input_read, input_write, output_read, output_write):
                try:
                    os.close(fd)
                except OSError:
                    pass

"""Test-owned loopback evidence for the private Ollama lifecycle."""
import asyncio
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

from services.llm.bootstrap.supervisor import OllamaSupervisorError, OwnedOllama
from services.llm.bootstrap.config import BootstrapConfig, ModelConfig
from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import ProcessIdentity


FAKE = r'''
import os, signal, socket, sys, time
port = int(os.environ["OLLAMA_HOST"].rsplit(":", 1)[1])
mode = os.environ.get("FAKE_OLLAMA_MODE", "normal")
if mode == "exit": raise SystemExit(23)
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", port)); s.listen(8); s.settimeout(.1)
while True:
    try: c, _ = s.accept()
    except socket.timeout: continue
    with c:
        request = c.recv(4096)
        if mode in ("orphan", "nested", "escape"):
            child = os.fork()
            if child == 0:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                if mode == "nested":
                    grandchild = os.fork()
                    if grandchild == 0:
                        signal.signal(signal.SIGTERM, signal.SIG_IGN)
                        time.sleep(5)
                        raise SystemExit(0)
                if mode == "escape":
                    time.sleep(.15)
                    os.setsid()
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                time.sleep(5)
                raise SystemExit(0)
        version = os.environ.get("FAKE_OLLAMA_VERSION", "0.1.0")
        body = ('{"version":"' + version + '"}').encode()
        c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
'''


def _identity() -> ProcessIdentity:
    raw = (Path("/proc") / str(os.getpid()) / "stat").read_bytes()
    return ProcessIdentity(os.getpid(), int(raw[raw.rfind(b")") + 2:].split()[19]))


def _config(root: Path, port: int, version: str = "0.1.0"):
    models = {name: ModelConfig(("ollama:" + version if name == "SmolLM" else "runtime:" + name),
                                 "adapter:" + name)
              for name in ("SmolLM", "CoEdIT", "GECToR")}
    return BootstrapConfig("GPU-test", root / "artifacts", "0" * 64, root / "profiles.sqlite",
                           Path(sys.executable), root / "home", port, models)


def _port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class OwnedOllamaTests(unittest.IsolatedAsyncioTestCase):
    def supervisor(self, root, *, mode="normal", version="0.1.0", timeout=.8, port=None):
        config = _config(Path(root), port or _port(), version)
        proof = GPUProof(lambda: "GPU-test", lambda: True, expected_supervisor=_identity())
        env = {"FAKE_OLLAMA_MODE": mode, "FAKE_OLLAMA_VERSION": version}
        # The process inherits only supervisor-approved variables.  The fake
        # communicates its fault mode through an explicit command wrapper.
        script = "import os; os.environ.update(" + repr(env) + "); exec(" + repr(FAKE) + ")"
        return OwnedOllama(config, proof, startup_timeout=timeout,
                           command=[sys.executable, "-c", script])

    async def test_start_version_alive_close_and_repeated_concurrent_close(self):
        with tempfile.TemporaryDirectory() as root:
            daemon = self.supervisor(root)
            self.assertEqual(await daemon.start(), "0.1.0")
            self.assertTrue(await daemon.alive())
            await asyncio.gather(*(daemon.close() for _ in range(4)))
            await daemon.close()
            self.assertFalse(await daemon.alive())
            self.assertIsNone(daemon.process)

    async def test_duplicate_malformed_oversized_and_version_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            for response in ("duplicate", "malformed", "oversized"):
                config = _config(Path(root) / response, _port())
                proof = GPUProof(lambda: "GPU-test", lambda: True, expected_supervisor=_identity())
                body = {"duplicate": b'{"version":"0.1.0","version":"0.1.0"}',
                        "malformed": b"not-json", "oversized": b"{" + b"x" * (300 * 1024)}
                payload = "b'{' + b'x' * (300 * 1024)" if response == "oversized" else repr(body[response])
                script = ("import socket,time; s=socket.socket(); s.bind(('127.0.0.1',%d)); s.listen(1); "
                          "c,_=s.accept(); c.recv(4096); b=%s; c.sendall(b'HTTP/1.1 200 OK\\r\\n\\r\\n'+b); time.sleep(5)" %
                          (config.ollama_port, payload))
                daemon = OwnedOllama(config, proof, startup_timeout=.15,
                                     command=[sys.executable, "-c", script], output_limit=1024)
                with self.assertRaises(OllamaSupervisorError):
                    await daemon.start()
                self.assertIsNone(daemon.process)
            # Construct a mismatch without mutating the frozen production config.
            daemon = self.supervisor(root, version="9.9.9")
            daemon.config = _config(Path(root) / "mismatch", daemon.port, "0.1.0")
            with self.assertRaises(OllamaSupervisorError):
                await daemon.start()

    async def test_occupied_foreign_port_is_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            sock = socket.socket(); sock.bind(("127.0.0.1", 0)); sock.listen(1)
            daemon = self.supervisor(root, port=sock.getsockname()[1])
            with self.assertRaises(OllamaSupervisorError):
                await daemon.start()
            self.assertIsNone(daemon.process)
            self.assertGreater(sock.fileno(), 0)
            sock.close()

    async def test_timeout_and_pid_reuse_refusal(self):
        with tempfile.TemporaryDirectory() as root:
            daemon = self.supervisor(root, mode="exit", timeout=.1)
            with self.assertRaises(OllamaSupervisorError):
                await daemon.start()
            daemon = self.supervisor(root)
            await daemon.start()
            with patch("services.llm.bootstrap.supervisor._fence",
                       side_effect=OllamaSupervisorError("PID was reused")):
                self.assertFalse(await daemon.alive())
                with self.assertRaises(OllamaSupervisorError):
                    await daemon.close()
            await daemon.close()

    async def test_wait_cleans_unexpected_exit_and_orphan_group(self):
        with tempfile.TemporaryDirectory() as root:
            daemon = self.supervisor(root, mode="orphan")
            await daemon.start()
            os.kill(daemon.process.pid, 15)
            self.assertEqual(await daemon.wait(), -15)
            self.assertIsNone(daemon.process)

            daemon = self.supervisor(root)
            await daemon.start()
            os.kill(daemon.process.pid, 15)
            self.assertEqual(await daemon.wait(), -15)
            self.assertIsNone(daemon.process)
            self.assertFalse(await daemon.alive())

    async def test_close_kills_live_nested_term_ignoring_descendants(self):
        with tempfile.TemporaryDirectory() as root:
            daemon = self.supervisor(root, mode="nested")
            await daemon.start()
            await daemon.close()
            self.assertIsNone(daemon.process)

    async def test_close_kills_captured_descendant_after_setsid(self):
        with tempfile.TemporaryDirectory() as root:
            daemon = self.supervisor(root, mode="escape")
            await daemon.start()
            await asyncio.sleep(.2)
            await daemon.close()
            self.assertIsNone(daemon.process)

    async def test_health_failure_cancellation_waits_for_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            daemon = self.supervisor(root, timeout=10)
            entered, release = asyncio.Event(), asyncio.Event()
            original = daemon._close_owned
            async def blocked():
                entered.set(); await release.wait(); await original()
            # RuntimeError bypasses the normal retryable-not-ready branch, so it
            # deterministically exercises the outer BaseException cleanup after
            # the verified launcher gate, rather than waiting for a deadline.
            with patch.object(daemon, "_health", side_effect=RuntimeError("forced health failure")), \
                 patch.object(daemon, "_close_owned", blocked):
                task = asyncio.create_task(daemon.start())
                try:
                    await asyncio.wait_for(entered.wait(), 1)
                    task.cancel(); task.cancel()
                finally:
                    # A failed assertion or timeout must never strand the
                    # cancellation-shielded cleanup task in unittest teardown.
                    release.set()
                with self.assertRaises(RuntimeError):
                    await asyncio.wait_for(task, 3)
            self.assertIsNone(daemon.process)

    async def test_start_cancellation_and_cancelled_spawn_are_collected(self):
        with tempfile.TemporaryDirectory() as root:
            daemon = self.supervisor(root, timeout=10)
            task = asyncio.create_task(daemon.start())
            await asyncio.sleep(.05)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertIsNone(daemon.process)

    async def test_repeated_cancellation_before_gate_capture_denies_launcher(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / "executed"
            daemon = self.supervisor(root, timeout=10)
            daemon.command = (sys.executable, "-c", "from pathlib import Path; Path(%r).touch()" % str(marker))
            entered = asyncio.Event()
            original = daemon._gate_launcher
            async def stalled(proc, deadline):
                entered.set()
                await asyncio.Event().wait()
                return await original(proc, deadline)
            with patch.object(daemon, "_gate_launcher", stalled):
                task = asyncio.create_task(daemon.start())
                await entered.wait()
                task.cancel(); task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertIsNone(daemon.process)
            self.assertFalse(marker.exists())

    async def test_withheld_spawn_timeout_retains_then_collects_without_exec(self):
        with tempfile.TemporaryDirectory() as root:
            real, entered, release = asyncio.create_subprocess_exec, asyncio.Event(), asyncio.Event()
            marker = Path(root) / "executed"
            async def delayed(*args, **kwargs):
                proc = await real(*args, **kwargs); entered.set(); await release.wait(); return proc
            daemon = self.supervisor(root, timeout=.01)
            daemon.command = (sys.executable, "-c", "from pathlib import Path; Path(%r).touch()" % str(marker))
            with patch("asyncio.create_subprocess_exec", delayed):
                with self.assertRaises(OllamaSupervisorError):
                    await daemon.start()
                self.assertIsNotNone(daemon._spawn)
                self.assertFalse(marker.exists())
                release.set()
                await daemon.close()
            self.assertIsNone(daemon.process)
            self.assertFalse(marker.exists())

    async def test_concurrent_close_collects_one_late_spawn(self):
        with tempfile.TemporaryDirectory() as root:
            real, entered, release = asyncio.create_subprocess_exec, asyncio.Event(), asyncio.Event()
            async def delayed(*args, **kwargs):
                proc = await real(*args, **kwargs); entered.set(); await release.wait(); return proc
            daemon = self.supervisor(root, timeout=.01)
            with patch("asyncio.create_subprocess_exec", delayed):
                with self.assertRaises(OllamaSupervisorError): await daemon.start()
                release.set()
                await asyncio.gather(daemon.close(), daemon.close())
            self.assertIsNone(daemon.process)

    async def test_mismatched_saved_leader_never_pins_or_signals(self):
        with tempfile.TemporaryDirectory() as root:
            daemon = self.supervisor(root)
            await daemon.start()
            saved = daemon._fence
            daemon._fence = type(saved)(ProcessIdentity(saved.identity.pid, saved.identity.start_time + 1), saved.pgrp, saved.session)
            with patch.object(daemon, "_open_pidfd") as pin, patch.object(daemon, "_signal") as signal_member:
                with self.assertRaises(OllamaSupervisorError): daemon._members(allow_live=True)
                pin.assert_not_called(); signal_member.assert_not_called()
            daemon._fence = saved
            await daemon.close()

            real = asyncio.create_subprocess_exec
            entered, release = asyncio.Event(), asyncio.Event()
            async def delayed(*args, **kwargs):
                process = await real(*args, **kwargs); entered.set(); await release.wait(); return process
            daemon = self.supervisor(root, timeout=10)
            with patch("asyncio.create_subprocess_exec", delayed):
                task = asyncio.create_task(daemon.start())
                await entered.wait(); task.cancel(); release.set()
                with self.assertRaises(asyncio.CancelledError): await task
            self.assertIsNone(daemon.process)

    async def test_fragmented_version_and_bounded_delayed_body(self):
        """Readiness consumes HTTP chunks, but never waits past startup_timeout."""
        with tempfile.TemporaryDirectory() as root:
            config = _config(Path(root), _port())
            proof = GPUProof(lambda: "GPU-test", lambda: True, expected_supervisor=_identity())
            script = """
import socket, time
s=socket.socket(); s.bind(('127.0.0.1', %d)); s.listen(4)
while True:
 c,_=s.accept(); c.recv(4096)
 body=b'{"version":"0.1.0"}'
 c.sendall(b'HTTP/1.1 200 OK\\r\\nContent-Length: '+str(len(body)).encode()+b'\\r\\n\\r\\n{"ver')
 c.sendall(b'sion":"0.1.0"}')
 c.close()
""" % config.ollama_port
            daemon = OwnedOllama(config, proof, startup_timeout=.5,
                                 command=[sys.executable, "-c", script])
            self.assertEqual(await daemon.start(), "0.1.0")
            await daemon.close()

            config = _config(Path(root) / "delayed", _port())
            script = """
import socket, time
s=socket.socket(); s.bind(('127.0.0.1', %d)); s.listen(4)
while True:
 c,_=s.accept(); c.recv(4096)
 body=b'{"version":"0.1.0"}'
 c.sendall(b'HTTP/1.1 200 OK\\r\\nContent-Length: '+str(len(body)).encode()+b'\\r\\n\\r\\n{"ver')
 time.sleep(5)
""" % config.ollama_port
            daemon = OwnedOllama(config, proof, startup_timeout=.15,
                                 command=[sys.executable, "-c", script])
            with self.assertRaises(OllamaSupervisorError):
                await daemon.start()
            self.assertIsNone(daemon.process)

    async def test_close_during_withheld_spawn_waits_and_cleans(self):
        with tempfile.TemporaryDirectory() as root:
            real = asyncio.create_subprocess_exec
            entered, release = asyncio.Event(), asyncio.Event()

            async def delayed(*args, **kwargs):
                process = await real(*args, **kwargs)
                entered.set()
                await release.wait()
                return process

            daemon = self.supervisor(root, timeout=.2)
            with patch("asyncio.create_subprocess_exec", delayed):
                starting = asyncio.create_task(daemon.start())
                await entered.wait()
                closing = asyncio.create_task(daemon.close())
                await asyncio.sleep(.02)
                self.assertFalse(closing.done())
                release.set()
                self.assertEqual(await starting, "0.1.0")
                await closing
            self.assertIsNone(daemon.process)

    async def test_spawn_timeout_retains_pending_ownership_until_release(self):
        """A late spawn cannot execute, be replaced, or leave a daemon marker."""
        with tempfile.TemporaryDirectory() as root:
            real = asyncio.create_subprocess_exec
            entered, release = asyncio.Event(), asyncio.Event()
            marker = Path(root) / "executed"

            async def delayed(*args, **kwargs):
                process = await real(*args, **kwargs)
                entered.set()
                await release.wait()
                return process

            daemon = self.supervisor(root, timeout=.05)
            daemon.command = (sys.executable, "-c",
                              "from pathlib import Path; Path(%r).touch()" % str(marker))
            with patch("asyncio.create_subprocess_exec", delayed):
                with self.assertRaises(OllamaSupervisorError):
                    await daemon.start()
                self.assertIsNotNone(daemon._spawn)
                with self.assertRaises(OllamaSupervisorError):
                    await daemon.start()
                release.set()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    await daemon.close()
                    await asyncio.gather(*(daemon.close() for _ in range(3)))
                self.assertFalse([w for w in caught if issubclass(w.category, RuntimeWarning)])
            self.assertIsNone(daemon.process)
            self.assertIsNone(daemon._spawn)
            self.assertFalse(marker.exists())

    async def test_identity_capture_failure_keeps_handle_without_leaking_child(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / "executed"
            daemon = self.supervisor(root, timeout=.2)
            daemon.command = (sys.executable, "-c", "from pathlib import Path; Path(%r).touch()" % str(marker))
            with patch("services.llm.bootstrap.supervisor._fence",
                       side_effect=OllamaSupervisorError("capture failed")):
                with self.assertRaises(OllamaSupervisorError):
                    await daemon.start()
            self.assertIsNone(daemon.process)
            self.assertFalse(marker.exists())

    def test_pidfd_is_closed_when_fence_verification_raises(self):
        with tempfile.TemporaryDirectory() as root:
            daemon = self.supervisor(root)
            fence = type("Fence", (), {
                "identity": ProcessIdentity(4242, 7),
            })()
            with patch("services.llm.bootstrap.supervisor.os.pidfd_open", return_value=91), \
                 patch("services.llm.bootstrap.supervisor.os.close") as close, \
                 patch("services.llm.bootstrap.supervisor._fence",
                       side_effect=OllamaSupervisorError("verification failed")), \
                 patch("services.llm.bootstrap.supervisor.signal.pidfd_send_signal") as send:
                with self.assertRaises(OllamaSupervisorError):
                    daemon._open_pidfd(fence)
            close.assert_called_once_with(91)
            send.assert_not_called()
            self.assertEqual(daemon._pidfds, {})

    async def test_repeated_cancelled_close_waits_for_one_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            daemon = self.supervisor(root)
            await daemon.start()
            original = daemon._close_owned
            entered, release = asyncio.Event(), asyncio.Event()

            async def delayed_close():
                entered.set()
                await release.wait()
                await original()

            with patch.object(daemon, "_close_owned", delayed_close):
                tasks = [asyncio.create_task(daemon.close()) for _ in range(3)]
                await entered.wait()
                for task in tasks:
                    task.cancel()
                release.set()
                for task in tasks:
                    with self.assertRaises(asyncio.CancelledError):
                        await task
            await daemon.close()
            self.assertIsNone(daemon.process)

"""Bounded protocol and ownership tests for the root Ollama broker client."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.llm.bootstrap.ollama_broker_client import BrokerClient, OllamaBrokerError
from services.llm.providers.gpu import ProcessIdentity


BROKER = r'''#!/usr/bin/env python3
import json, os, sys, time
mode = os.environ.get("BROKER_TEST_MODE", "normal")
for line in sys.stdin:
    if mode == "silent":
        time.sleep(10)
        continue
    if mode == "partial":
        sys.stdout.write('{"ok":true')
        sys.stdout.flush()
        time.sleep(10)
        continue
    if mode == "oversized":
        sys.stdout.write("x" * 4097 + "\n")
    elif mode == "invalid":
        sys.stdout.write("not-json\n")
    elif mode == "duplicate":
        sys.stdout.write('{"ok":true}\n{"ok":true}\n')
    elif mode == "wrongok":
        sys.stdout.write('{"ok":"yes"}\n')
    elif mode == "eof":
        break
    elif mode == "delay":
        time.sleep(10)
    else:
        request = json.loads(line)
        if request["command"] == "stop":
            sys.stdout.write('{"ok":true}\n')
            sys.stdout.flush()
            break
        sys.stdout.write('{"ok":true,"version":"0.11.6"}\n')
    sys.stdout.flush()
'''


class BrokerClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gateway = Path(self.tmp.name) / "gateway"
        self.gateway.write_text(BROKER)
        self.gateway.chmod(0o755)
        self.supervisor = ProcessIdentity(os.getpid(), self._start(os.getpid()))

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _start(pid: int) -> int:
        raw = (Path("/proc") / str(pid) / "stat").read_bytes()
        return int(raw[raw.rfind(b")") + 2:].split()[19])

    def client(self, mode="normal", timeout=.1):
        client = BrokerClient(self.supervisor, timeout=timeout, gateway=str(self.gateway))
        real_popen = subprocess.Popen

        def owned_popen(args, **kwargs):
            env = dict(kwargs.get("env", {}))
            env["BROKER_TEST_MODE"] = mode
            kwargs["env"] = env
            # The production cwd is deliberately fixed; the test executable is not.
            kwargs["cwd"] = self.tmp.name
            return real_popen(args, **kwargs)

        return client, owned_popen

    async def test_start_and_health_pin_version_and_default_configuration(self):
        client, popen = self.client()
        with patch("services.llm.bootstrap.ollama_broker_client.subprocess.Popen", popen), \
             patch.object(client, "_verify"):
            self.assertEqual("0.11.6", await client.start())
            self.assertEqual("0.11.6", await client.health())
        await client.close()

    async def test_malformed_or_incomplete_responses_fail_boundedly_and_reap(self):
        for mode in ("partial", "silent", "oversized", "invalid", "duplicate", "wrongok", "eof"):
            with self.subTest(mode=mode):
                client, popen = self.client(mode)
                with patch("services.llm.bootstrap.ollama_broker_client.subprocess.Popen", popen):
                    started = time.monotonic()
                    with self.assertRaises(OllamaBrokerError):
                        await client.start()
                    self.assertLess(time.monotonic() - started, 2)
                    proc = client._proc
                    if mode in ("partial", "silent"):
                        with self.assertRaises(OllamaBrokerError):
                            await client.close()
                    else:
                        await client.close()
                self.assertIsNotNone(proc)
                self.assertIsNotNone(proc.returncode)
                self.assertTrue(proc.stdin.closed)
                self.assertTrue(proc.stdout.closed)
                with self.assertRaises(OllamaBrokerError):
                    await client.health()

    async def test_close_on_broken_client_is_one_shot_and_reaps_child(self):
        client, popen = self.client("silent")
        with patch("services.llm.bootstrap.ollama_broker_client.subprocess.Popen", popen):
            with self.assertRaises(OllamaBrokerError):
                await client.start()
            proc = client._proc
            with self.assertRaises(OllamaBrokerError):
                await client.close()
            with self.assertRaises(OllamaBrokerError):
                await client.close()
        self.assertIsNotNone(proc)
        self.assertIsNotNone(proc.returncode)
        self.assertTrue(client._closed)
        with self.assertRaises(OllamaBrokerError):
            await client.start()

    async def test_lock_contention_is_bounded(self):
        client, _ = self.client()
        self.assertTrue(client._lock.acquire())
        try:
            started = time.monotonic()
            with self.assertRaises(OllamaBrokerError):
                await client.health()
            self.assertLess(time.monotonic() - started, 1)
        finally:
            client._lock.release()
        await client.close()

    async def test_cancelled_start_then_close_does_not_leave_transport(self):
        client, popen = self.client("delay", timeout=.1)
        with patch("services.llm.bootstrap.ollama_broker_client.subprocess.Popen", popen):
            task = asyncio.create_task(client.start())
            await asyncio.sleep(.02)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            with self.assertRaises(OllamaBrokerError):
                await asyncio.wait_for(client.close(), 1)
        self.assertIsNone(client._proc)
        self.assertTrue(client._closed)

    def _verified_client(self):
        client, _ = self.client()
        client._proc = SimpleNamespace(pid=20)
        return client

    def _patch_topology(self, identities, parents):
        return patch.multiple(
            "services.llm.bootstrap.ollama_broker_client",
            _identity=lambda pid: identities[pid],
            _parent=lambda pid: parents[pid],
            _descends=lambda pid, ancestor: pid == 30 or (pid == 40 and ancestor == 30),
        )

    def test_snapshot_verifier_requires_current_supervisor_and_strict_descendants(self):
        client = self._verified_client()
        client.expected_supervisor = ProcessIdentity(10, 100)
        identities = {10: ProcessIdentity(10, 100), 20: ProcessIdentity(20, 200),
                      30: ProcessIdentity(30, 300), 40: ProcessIdentity(40, 400)}
        parents = {10: 1, 20: 10, 30: 20, 40: 30}
        value = {"broker": {"pid": 20, "start": 200},
                 "daemon": {"pid": 30, "start": 300},
                 "descendants": [{"pid": 40, "start": 400}]}
        with self._patch_topology(identities, parents):
            client._verify(value)
        self.assertEqual(ProcessIdentity(30, 300), client._snapshot.daemon)

        identities[10] = ProcessIdentity(10, 999)  # stale supervisor identity
        with self._patch_topology(identities, parents), self.assertRaises(OllamaBrokerError):
            client._verify(value)

    def test_snapshot_verifier_rejects_duplicate_pid_with_changed_start_identity(self):
        client = self._verified_client()
        client.expected_supervisor = ProcessIdentity(10, 100)
        identities = {10: ProcessIdentity(10, 100), 20: ProcessIdentity(20, 200),
                      30: ProcessIdentity(30, 300), 40: ProcessIdentity(40, 400)}
        parents = {10: 1, 20: 10, 30: 20, 40: 30}
        value = {"broker": {"pid": 20, "start": 200},
                 "daemon": {"pid": 30, "start": 300},
                 "descendants": [{"pid": 40, "start": 400}, {"pid": 40, "start": 401}]}
        with self._patch_topology(identities, parents), self.assertRaises(OllamaBrokerError):
            client._verify(value)

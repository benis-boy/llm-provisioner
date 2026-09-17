import asyncio
import json
import threading
import unittest
from pathlib import Path
import tempfile

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from services.llm.provisioning.http import create_app
from services.llm.provisioning.volume import provision
from services.llm.provisioning.artifacts import SPECS


class ProvisioningHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_verify_endpoint_is_bounded_and_minimal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"; source.mkdir()
            model = source / "SmolLM"; model.mkdir()
            for name in SPECS["SmolLM"]: (model / name).write_bytes(name.encode())
            volume = root / "volume"; expected = provision({"SmolLM": model}, volume)
            server = TestServer(create_app({"offline": volume}))
            await server.start_server()
            try:
                async with ClientSession() as client:
                    response = await client.post(server.make_url("/provisioning/verify-artifacts"),
                        headers={"Idempotency-Key": "correlation"}, json={"volumeId": "offline"})
                    self.assertEqual(response.status, 200)
                    payload = await response.json()
                    self.assertEqual(payload["manifestSha256"], expected["manifest_sha256"])
                    self.assertNotIn("ready", payload); self.assertNotIn("capacity", payload)
                    bad = await client.post(server.make_url("/provisioning/verify-artifacts"),
                        headers={"Idempotency-Key": "x"}, json={"volumeId": "missing"})
                    self.assertEqual(bad.status, 404)
            finally:
                await server.close()

    async def test_request_validation_is_strict_and_unknown_volume_does_not_invoke_verifier(self):
        calls = []
        def verifier(_root):
            calls.append(True)
            return {"manifestSha256": "0" * 64, "models": [], "verified": True,
                    "verificationScope": "selected-file-integrity"}

        server = TestServer(create_app({"offline": Path("/tmp/offline")}, verifier=verifier))
        await server.start_server()
        try:
            async with ClientSession() as client:
                url = server.make_url("/provisioning/verify-artifacts")
                async def post(body, headers=None):
                    request_headers = {"Idempotency-Key": "x", "Content-Type": "application/json"}
                    if headers is not None:
                        request_headers.update(headers)
                    return await client.post(url, headers=request_headers, data=body)
                for body in (b"{}", b"[]", b'{"volumeId":"offline","extra":1}',
                              b'{"volumeId":1}', b'{"volumeId":"offline","expectedManifestSha256":true}',
                              b'{"volumeId":"offline","expectedManifestSha256":null}',
                              b'{"volumeId":"offline","volumeId":"offline"}',
                              b'{"volumeId":"offline","x":NaN}',
                              b"not-json"):
                    response = await post(body)
                    self.assertEqual(response.status, 400, body)
                response = await post(b'{"volumeId":"missing"}')
                self.assertEqual(response.status, 404)
                self.assertEqual(calls, [])
                response = await post(b'{"volumeId":"offline"}', {"Idempotency-Key": ""})
                self.assertEqual(response.status, 400)
                response = await post(b'{"volumeId":"offline"}', {"Content-Type": "text/plain"})
                self.assertEqual(response.status, 400)
                response = await client.post(url, headers={"Idempotency-Key": "missing-content-type"},
                    data=b'{"volumeId":"offline"}')
                self.assertEqual(response.status, 400)
        finally:
            await server.close()

    async def test_chunked_json_is_complete_and_chunked_oversize_is_rejected(self):
        calls = []
        def verifier(_root):
            calls.append(True)
            return {"manifestSha256": "0" * 64, "models": [], "verified": True,
                    "verificationScope": "selected-file-integrity"}

        server = TestServer(create_app({"offline": Path("/tmp/offline")}, verifier=verifier,
                                       max_body_bytes=32))
        await server.start_server()
        try:
            async with ClientSession() as client:
                url = server.make_url("/provisioning/verify-artifacts")
                async def chunks(parts):
                    for part in parts:
                        await asyncio.sleep(0)
                        yield part
                response = await client.post(url, data=chunks([b'{"volume', b'Id":"offline"}']),
                    headers={"Idempotency-Key": "x", "Content-Type": "application/json"})
                self.assertEqual(response.status, 200)
                response = await client.post(url, data=chunks([b"x" * 20, b"y" * 20]),
                    headers={"Idempotency-Key": "x", "Content-Type": "application/json"})
                self.assertEqual(response.status, 413)
                self.assertEqual(len(calls), 1)
        finally:
            await server.close()

    async def test_injected_verifier_cannot_expand_success_response(self):
        def verifier(_root):
            return {"manifestSha256": "0" * 64, "models": [], "verified": True,
                    "verificationScope": "selected-file-integrity", "secret": "no"}

        server = TestServer(create_app({"offline": Path("/tmp/offline")}, verifier=verifier))
        await server.start_server()
        try:
            async with ClientSession() as client:
                response = await client.post(server.make_url("/provisioning/verify-artifacts"),
                    headers={"Idempotency-Key": "x"}, json={"volumeId": "offline"})
                self.assertEqual(response.status, 422)
        finally:
            await server.close()

    async def test_injected_verifier_cannot_return_unsupported_model_summary(self):
        def verifier(_root):
            return {"manifestSha256": "0" * 64,
                    "models": [{"modelId": "unsupported", "fileCount": 1, "totalBytes": 1}],
                    "verified": True, "verificationScope": "selected-file-integrity"}

        server = TestServer(create_app({"offline": Path("/tmp/offline")}, verifier=verifier))
        await server.start_server()
        try:
            async with ClientSession() as client:
                response = await client.post(server.make_url("/provisioning/verify-artifacts"),
                    headers={"Idempotency-Key": "x"}, json={"volumeId": "offline"})
                self.assertEqual(response.status, 422)
        finally:
            await server.close()

    async def test_timeout_retains_worker_slot_until_actual_thread_exit(self):
        started = threading.Event()
        release = threading.Event()
        calls = 0
        def verifier(_root):
            nonlocal calls
            calls += 1
            started.set()
            release.wait(5)
            return {"manifestSha256": "0" * 64, "models": [], "verified": True,
                    "verificationScope": "selected-file-integrity"}

        server = TestServer(create_app({"offline": Path("/tmp/offline")}, verifier=verifier,
                                       max_concurrent=1, timeout_seconds=0.05))
        await server.start_server()
        try:
            async with ClientSession() as client:
                url = server.make_url("/provisioning/verify-artifacts")
                first = asyncio.create_task(client.post(url, headers={"Idempotency-Key": "one"},
                    json={"volumeId": "offline"}))
                await asyncio.to_thread(started.wait, 2)
                timeout = await first
                self.assertEqual(timeout.status, 504)
                busy = await client.post(url, headers={"Idempotency-Key": "two"},
                    json={"volumeId": "offline"})
                self.assertEqual(busy.status, 429)
                release.set()
                await asyncio.sleep(0.05)
                available = await client.post(url, headers={"Idempotency-Key": "three"},
                    json={"volumeId": "offline"})
                self.assertEqual(available.status, 200)
                self.assertEqual(calls, 2)
        finally:
            release.set()
            await server.close()

    async def test_simultaneous_max_plus_one_returns_overload(self):
        entered = threading.Event()
        release = threading.Event()
        active = 0
        lock = threading.Lock()
        def verifier(_root):
            nonlocal active
            with lock:
                active += 1
                if active == 2:
                    entered.set()
            release.wait(5)
            return {"manifestSha256": "0" * 64, "models": [], "verified": True,
                    "verificationScope": "selected-file-integrity"}

        server = TestServer(create_app({"offline": Path("/tmp/offline")}, verifier=verifier,
                                       max_concurrent=2, timeout_seconds=2))
        await server.start_server()
        try:
            async with ClientSession() as client:
                url = server.make_url("/provisioning/verify-artifacts")
                requests = [asyncio.create_task(client.post(url, headers={"Idempotency-Key": str(i)},
                    json={"volumeId": "offline"})) for i in range(3)]
                await asyncio.to_thread(entered.wait, 2)
                await asyncio.sleep(0.02)
                self.assertEqual(requests[2].done(), True)
                self.assertEqual((await requests[2]).status, 429)
                release.set()
                self.assertEqual((await requests[0]).status, 200)
                self.assertEqual((await requests[1]).status, 200)
        finally:
            release.set()
            await server.close()

    async def test_timeout_then_worker_exception_is_consumed_without_loop_diagnostic(self):
        started = threading.Event()
        release = threading.Event()
        def verifier(_root):
            started.set()
            release.wait(5)
            raise RuntimeError("controlled verifier failure")

        server = TestServer(create_app({"offline": Path("/tmp/offline")}, verifier=verifier,
                                       max_concurrent=1, timeout_seconds=0.03))
        await server.start_server()
        diagnostics = []
        loop = asyncio.get_running_loop()
        old_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
        try:
            async with ClientSession() as client:
                url = server.make_url("/provisioning/verify-artifacts")
                response = await client.post(url, headers={"Idempotency-Key": "x"},
                    json={"volumeId": "offline"})
                self.assertEqual(response.status, 504)
                await asyncio.to_thread(started.wait, 2)
                release.set()
                await asyncio.sleep(0.1)
                self.assertEqual(diagnostics, [])
        finally:
            release.set()
            loop.set_exception_handler(old_handler)
            await server.close()

    async def test_disconnect_keeps_worker_slot_until_cleanup(self):
        started = threading.Event()
        release = threading.Event()
        def verifier(_root):
            started.set()
            release.wait(5)
            return {"manifestSha256": "0" * 64, "models": [], "verified": True,
                    "verificationScope": "selected-file-integrity"}

        server = TestServer(create_app({"offline": Path("/tmp/offline")}, verifier=verifier,
                                       max_concurrent=1, timeout_seconds=2))
        await server.start_server()
        try:
            async with ClientSession() as client:
                url = server.make_url("/provisioning/verify-artifacts")
                disconnected = asyncio.create_task(client.post(url, headers={"Idempotency-Key": "one"},
                    json={"volumeId": "offline"}))
                await asyncio.to_thread(started.wait, 2)
                disconnected.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await disconnected
                busy = await client.post(url, headers={"Idempotency-Key": "two"},
                    json={"volumeId": "offline"})
                self.assertEqual(busy.status, 429)
                release.set()
                await asyncio.sleep(0.05)
                available = await client.post(url, headers={"Idempotency-Key": "three"},
                    json={"volumeId": "offline"})
                self.assertEqual(available.status, 200)
        finally:
            release.set()
            await server.close()

    async def test_unimplemented_measure_capacity_and_ready_are_not_exposed(self):
        server = TestServer(create_app({"offline": Path("/tmp/offline")}))
        await server.start_server()
        try:
            async with ClientSession() as client:
                capacity = await client.post(server.make_url("/provisioning/measure-capacity"))
                ready = await client.get(server.make_url("/health/ready"))
                self.assertEqual(capacity.status, 404)
                self.assertEqual(ready.status, 404)
        finally:
            await server.close()


if __name__ == "__main__":
    unittest.main()

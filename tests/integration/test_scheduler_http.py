"""Loopback contract tests for the scheduler HTTP adapter.

These deliberately use the production store, scheduler, resource manager and
publisher.  The provider is only a deterministic test double at the execution
boundary; no frontend or browser behavior is covered here.
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from aiohttp import ClientSession, web

from services.llm.queue.http import SchedulerHttpClient, SchedulerHttpError, SchedulerHttpServer
from services.llm.queue.results import LocalPublisher, ResultStore
from services.llm.queue.scheduler import DecodedPayload, DispatchContext, QueueScheduler
from services.llm.queue.store import QueueStore
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager
from services.llm.resource_manager.protocol import ProviderResponse


def _profile():
    return CapacityProfile(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", "profile",
                           1, 1, 1, 20, (SampleMetadata(1, 0, 1, 1, 1, (1,)),), context_size=128)


class FakeProvider:
    def __init__(self):
        self.release = asyncio.Event()
        self.calls = []

    async def validate(self, profile): pass
    async def load(self, profile): pass
    async def ready(self): pass
    async def validate_input(self, payload, *, context_size, bucket_identity): pass
    async def execute(self, request_id, payload):
        self.calls.append((request_id, payload))
        await self.release.wait()
        return ProviderResponse(b"result:" + payload, None, False)
    async def cancel(self, request_id): pass
    async def unload(self): pass
    async def verify_cleanup(self): return True


class SchedulerHttpIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.provider = FakeProvider()
        self.store = QueueStore(root / "queue.sqlite", "sched", ModelId.SMOLLM)
        self.results = ResultStore(root / "results")
        self.publisher = LocalPublisher(self.results, root / "publisher.sqlite")
        self.rm = ResourceManager(cleanup_timeout=.05, stop_timeout=.05)
        self.scheduler = QueueScheduler(
            self.store, self.rm, _profile(), self.provider,
            decoder=lambda ref: DecodedPayload(ref.encode(), DispatchContext(128)), result_store=self.results,
            publisher=self.publisher, loop_interval=.01, stop_timeout=.05,
        )
        self.http = SchedulerHttpServer({"sched": self.scheduler}, heartbeat=.02, write_timeout=.2)
        self.runner = web.AppRunner(self.http.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.client = SchedulerHttpClient(f"http://127.0.0.1:{port}", timeout=2)

    async def asyncTearDown(self):
        if self.scheduler._started:
            await self.scheduler.stop("test_done")
        await self.runner.cleanup()
        self.publisher.close()
        self.store.close()
        self.tmp.cleanup()

    async def wait_for(self, predicate, timeout=.8):
        end = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() >= end:
                self.fail("condition did not become true")
            await asyncio.sleep(.005)

    async def test_all_operations_and_done_after_local_publisher_ack(self):
        started = await self.client.start("sched", "SmolLM", idempotency_key="start")
        self.assertEqual(started["schedulerId"], "sched")
        created = await self.client.enqueue("sched", {
            "schedulerId": "sched", "requestId": "r", "modelId": "SmolLM",
            "payloadReference": "payload", "dependencies": [],
            "insertionMode": "append", "resultTarget": "local",
        }, idempotency_key="enqueue")
        self.assertEqual(created["status"], "scheduled")
        await self.wait_for(lambda: bool(self.provider.calls))
        self.provider.release.set()
        await self.wait_for(lambda: self.store.get("r")["status"] == "done")
        current = await self.client.get("sched", "r")
        self.assertEqual(current["status"], "done")
        self.assertIsNotNone(current["doneAt"])
        self.assertEqual(self.results.read(current["resultReference"]), b"result:payload")
        self.assertIn("snapshot", json.loads(self.store.events(0)[-1]["data"]))
        await self.client.stop("sched", idempotency_key="stop")

    async def test_cancel_stop_replay_and_strict_request_validation(self):
        await self.client.start("sched", "SmolLM", idempotency_key="start")
        await self.client.enqueue("sched", {
            "schedulerId": "sched", "requestId": "cancel-me", "modelId": "SmolLM",
            "payloadReference": "p", "dependencies": [],
        }, idempotency_key="e")
        cancelled = await self.client.cancel("sched", "cancel-me", idempotency_key="cancel")
        self.assertEqual(cancelled["status"], "cancelled")
        replay = [event async for event in self.client.watch("sched", "cancel-me")]
        self.assertTrue(replay)
        self.assertEqual(replay[-1]["data"]["status"], "cancelled")
        with self.assertRaises(SchedulerHttpError) as mismatch:
            await self.client.start("sched", "CoEdIT", idempotency_key="other")
        self.assertEqual(mismatch.exception.code, "model_mismatch")
        async with ClientSession() as session:
            url = self.client.base_url + "/schedulers/sched/start"
            for raw in (b'{"schedulerId":"sched","modelId":"SmolLM","modelId":"SmolLM"}',
                        b'{"schedulerId":"sched","modelId":NaN}',
                        b'{"schedulerId":"sched","modelId":"SmolLM","extra":1}'):
                async with session.post(url, data=raw, headers={"Idempotency-Key": "raw"}) as response:
                    self.assertEqual(response.status, 400)
        await self.client.stop("sched", outcome="cancelled", reason="client", idempotency_key="stop")

    async def test_future_cursor_and_immutable_sse_snapshots(self):
        await self.client.start("sched", "SmolLM", idempotency_key="start")
        await self.client.enqueue("sched", {
            "schedulerId": "sched", "requestId": "r", "modelId": "SmolLM",
            "payloadReference": "p", "dependencies": [],
        }, idempotency_key="e")
        await self.client.cancel("sched", "r", idempotency_key="c")
        async with ClientSession() as session:
            url = self.client.base_url + "/schedulers/sched/requests/r/watch"
            async with session.get(url, headers={"Last-Event-ID": "999"}) as response:
                self.assertEqual(response.status, 409)
                self.assertEqual((await response.json())["code"], "cursor_future")
        snapshots = [event async for event in self.client.watch("sched", "r")]
        self.assertEqual(snapshots[0]["data"]["status"], "scheduled")
        self.assertEqual(snapshots[-1]["data"]["status"], "cancelled")
        self.assertEqual(len({event["id"] for event in snapshots}), len(snapshots))
        self.assertLess(int(snapshots[0]["id"]), int(snapshots[1]["id"]))

    async def test_request_history_is_bounded_ordered_and_never_replayed_live(self):
        await self.client.start("sched", "SmolLM", idempotency_key="start")
        await self.client.enqueue("sched", {"schedulerId": "sched", "requestId": "r", "modelId": "SmolLM", "payloadReference": "p"}, idempotency_key="e")
        for number in range(70):
            self.store.db.execute("INSERT INTO events(request_id,kind,data,created) VALUES(?,?,?,?)", ("other", "status", "{}", number))
        for _ in range(70):
            self.store._event("r", "status", {"new": "scheduled"})
        await self.client.cancel("sched", "r", idempotency_key="c")
        rows = self.store.request_events("r", 0, 64)
        self.assertEqual(len(rows), 64)
        self.assertEqual([item["cursor"] for item in rows], sorted(item["cursor"] for item in rows))
        events = [event async for event in self.client.watch("sched", "r", after=0)]
        # The request has two durable events before the synthetic history and
        # one terminal event after it; the batch boundary must not drop any.
        self.assertEqual(len(events), 74)
        self.assertEqual(len({event["id"] for event in events}), 74)

    async def test_live_replay_then_target_cancel_is_exactly_once_past_global_noise(self):
        await self.client.start("sched", "SmolLM", idempotency_key="start")
        await self.client.enqueue("sched", {"schedulerId": "sched", "requestId": "live", "modelId": "SmolLM", "payloadReference": "p"}, idempotency_key="e")
        stream = self.client.watch("sched", "live")
        received = []
        task = asyncio.create_task(stream.__anext__())
        first = await asyncio.wait_for(task, 1)
        received.append(first)
        self.assertEqual(first["data"]["status"], "scheduled")
        for number in range(70):
            self.store.db.execute("INSERT INTO events(request_id,kind,data,created) VALUES(?,?,?,?)", ("unrelated", "status", json.dumps({"noise": number}), number))
        self.store._event("live", "status", {"snapshot": self.store._projection("live")})
        await self.client.cancel("sched", "live", idempotency_key="c")
        async for event in stream:
            received.append(event)
        self.assertEqual(received[-1]["data"]["status"], "cancelled")
        self.assertEqual(len({event["id"] for event in received}), len(received))

    async def test_legacy_event_after_valid_matching_row_is_rejected_before_sse_headers(self):
        await self.client.start("sched", "SmolLM", idempotency_key="start")
        await self.client.enqueue("sched", {"schedulerId": "sched", "requestId": "legacy", "modelId": "SmolLM", "payloadReference": "p"}, idempotency_key="e")
        self.store._event("legacy", "status", {"snapshot": self.store._projection("legacy")})
        self.store.db.execute("INSERT INTO events(request_id,kind,data,created) VALUES(?,?,?,?)", ("legacy", "status", "{}", 9999))
        async with ClientSession() as session:
            url = self.client.base_url + "/schedulers/sched/requests/legacy/watch"
            async with session.get(url) as response:
                self.assertEqual(response.status, 409)
                self.assertNotEqual(response.headers.get("Content-Type", "").split(";", 1)[0], "text/event-stream")
                self.assertEqual((await response.json())["code"], "legacy_event_unavailable")

    async def test_heartbeat_and_disconnected_watcher_release_slot(self):
        await self.client.start("sched", "SmolLM", idempotency_key="start")
        await self.client.enqueue("sched", {"schedulerId": "sched", "requestId": "watch", "modelId": "SmolLM", "payloadReference": "p"}, idempotency_key="e")
        url = self.client.base_url + "/schedulers/sched/requests/watch/watch"
        async with ClientSession() as session:
            response = await session.get(url)
            self.assertEqual(response.status, 200)
            data = b""
            while b"heartbeat" not in data:
                data += await asyncio.wait_for(response.content.read(64), 1)
            response.close()
            async with session.get(url) as replacement:
                self.assertEqual(replacement.status, 200)
                replacement.close()

    async def _hostile_client(self, response_body: bytes, *, status=200, max_frame=256):
        async def handler(request):
            return web.Response(body=response_body, status=status, content_type="text/event-stream" if status == 200 else "application/json")
        app = web.Application()
        app.router.add_get("/schedulers/s/requests/r/watch", handler)
        app.router.add_get("/schedulers/s/requests/r", handler)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        port = site._server.sockets[0].getsockname()[1]
        return runner, SchedulerHttpClient(f"http://127.0.0.1:{port}", timeout=1, max_frame=max_frame)

    async def test_bounded_client_strict_json_error_and_success_bodies(self):
        runner, client = await self._hostile_client(b'{"ok":true}')
        try:
            self.assertEqual(await client.get("s", "r"), {"ok": True})
        finally:
            await runner.cleanup()
        runner, client = await self._hostile_client(b'{"code":"x","message":"m","retryable":false}', status=409)
        try:
            with self.assertRaises(SchedulerHttpError) as caught:
                await client.get("s", "r")
            self.assertEqual((caught.exception.status, caught.exception.code), (409, "x"))
        finally:
            await runner.cleanup()

    async def test_bounded_client_rejects_malformed_sse_and_accepts_coalesced_frames(self):
        valid = lambda n: f"id: {n}\nevent: status\ndata: {{\"sequence\":{n},\"requestId\":\"r\",\"schedulerId\":\"s\"}}\n\n".encode()
        cases = (valid(1).replace(b"id: 1\n", b"id: 1\nid: 1\n"), valid(2) + valid(1),
                 valid(1).replace(b'"requestId":"r"', b'"requestId":"wrong"'), valid(1)[:-1])
        for body in cases:
            runner, client = await self._hostile_client(body)
            try:
                with self.assertRaises(SchedulerHttpError) as caught:
                    [event async for event in client.watch("s", "r")]
                self.assertEqual(caught.exception.status, 502)
            finally:
                await runner.cleanup()
        runner, client = await self._hostile_client(valid(1) + valid(2))
        try:
            self.assertEqual([event["id"] async for event in client.watch("s", "r")], ["1", "2"])
        finally:
            await runner.cleanup()

    async def test_terminal_cursor_at_or_beyond_history_closes_without_heartbeat(self):
        await self.client.start("sched", "SmolLM", idempotency_key="start")
        await self.client.enqueue("sched", {"schedulerId": "sched", "requestId": "r", "modelId": "SmolLM", "payloadReference": "p"}, idempotency_key="e")
        await self.client.cancel("sched", "r", idempotency_key="c")
        terminal = self.store.events(0)[-1]["cursor"]
        self.assertEqual([event async for event in self.client.watch("sched", "r", after=terminal)], [])
        self.assertEqual([event async for event in self.client.watch("sched", "r", after=self.store.event_highwater())], [])

    async def test_malformed_enqueue_has_no_durable_side_effect(self):
        await self.client.start("sched", "SmolLM", idempotency_key="start")
        async with ClientSession() as session:
            url = self.client.base_url + "/schedulers/sched/requests"
            body = {"schedulerId": "sched", "requestId": "bad", "modelId": "SmolLM", "payloadReference": "p", "dependencies": "not-an-array"}
            async with session.post(url, json=body, headers={"Idempotency-Key": "bad"}) as response:
                self.assertEqual(response.status, 400)
                self.assertEqual(set((await response.json())), {"code", "message", "retryable"})
        self.assertIsNone(self.store.get("bad"))

    async def test_stop_requires_documented_outcome(self):
        await self.client.start("sched", "SmolLM", idempotency_key="start")
        async with ClientSession() as session:
            url = self.client.base_url + "/schedulers/sched/stop"
            async with session.post(url, json={}, headers={"Idempotency-Key": "stop"}) as response:
                self.assertEqual(response.status, 400)


if __name__ == "__main__":
    unittest.main()

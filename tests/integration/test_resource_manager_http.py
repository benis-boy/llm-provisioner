"""Loopback integration coverage for the ResourceManager HTTP boundary.

These tests deliberately keep provider/profile resolution on the server side.
The bytes crossing the boundary are content-addressed references in the shared
ResultStore, not provider objects or profile selectors supplied by a caller.
"""

import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import ClientSession, web

from services.llm.queue.contracts import ModelId
from services.llm.queue.results import LocalPublisher, ResultStore
from services.llm.queue.scheduler import DecodedPayload, DispatchContext, QueueScheduler
from services.llm.queue.store import QueueStore
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager, ResourceManagerError
from services.llm.resource_manager.http import ModelBinding, ResourceManagerHTTPClient, ResourceManagerHttpServer
from services.llm.resource_manager.profiles import BenchmarkMetadata, ProfileStore
from services.llm.resource_manager.protocol import EventKind, Failure, ProviderResponse


def measured_profile(model=ModelId.SMOLLM, context=128, bucket=None):
    def sample(concurrency, wave):
        return SampleMetadata(concurrency, wave, concurrency, 10, 100, tuple([2] * concurrency))

    baseline = tuple(sample(1, wave) for wave in range(4))
    warmup = (sample(1, 0), sample(2, 0))
    measured = tuple(sample(concurrency, wave) for concurrency in (1, 2) for wave in range(1, 5))
    profile_id = hashlib.sha256(json.dumps({
        "model_id": model.value, "gpu_uuid": "gpu", "artifact_manifest_hash": "manifest",
        "model_hash": "model", "runtime_identity": "runtime", "adapter_identity": "adapter",
        "context_size": context if bucket is None else None, "bucket_identity": bucket,
        "fingerprint": f"http-test-{model.value}-{bucket or context}",
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    profile = CapacityProfile(model, "gpu", "manifest", "model", "runtime", "adapter",
                              profile_id, 2, 2, 2, 20, baseline + warmup + measured,
                              context if bucket is None else None, bucket)
    fingerprint = f"http-test-{model.value}-{bucket or context}"
    metadata = BenchmarkMetadata(fingerprint, "2026-09-16T00:00:00Z", "integration-test",
                                 baseline, warmup, measured,
                                 f"context:{context}" if bucket is None else bucket)
    return profile, metadata


class FakeProvider:
    def __init__(self):
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.calls = []

    async def validate(self, profile):
        self.calls.append("validate")

    async def load(self, profile):
        self.calls.append("load")

    async def ready(self):
        self.calls.append("ready")

    async def validate_input(self, payload, *, context_size, bucket_identity):
        if payload == b"bad":
            raise ValueError("bad payload")

    async def execute(self, request_id, payload):
        self.calls.append(("execute", request_id))
        self.started.set()
        await self.release.wait()
        return ProviderResponse(payload.upper(), 3, True)

    async def cancel(self, request_id):
        self.calls.append(("cancel", request_id))

    async def unload(self):
        self.calls.append("unload")

    async def verify_cleanup(self):
        return True


class ResourceManagerHttpIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.results = ResultStore(root / "results")
        self.profiles = ProfileStore(root / "profiles.sqlite")
        self.profile, metadata = measured_profile()
        self.profiles.save_measured(self.profile, metadata)
        self.provider = FakeProvider()
        self.core = ResourceManager(cleanup_timeout=.05, stop_timeout=.05, max_events=16)
        binding = ModelBinding(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter",
                               self.profiles, self.provider)
        self.server = ResourceManagerHttpServer(self.core, bindings={ModelId.SMOLLM: binding},
                                                 result_store=self.results, max_body=256, max_frame=4096,
                                                 watch_timeout=.05, max_watches=1)
        self.runner = web.AppRunner(self.server.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.client = ResourceManagerHTTPClient(f"http://127.0.0.1:{port}", result_store=self.results,
                                                 timeout=2, max_frame=1024 * 1024)
        self.session = await self.client.start_session("scheduler", ModelId.SMOLLM, self.profile, self.provider,
                                                       idempotency_key="start")

    async def asyncTearDown(self):
        self.provider.release.set()
        if self.core._session is not None:
            try:
                await self.core.stop_session(self.core._session.session_token, idempotency_key="teardown")
            except ResourceManagerError:
                pass
        await self.runner.cleanup()
        self.profiles.close()
        self.tmp.cleanup()

    async def test_typed_start_capacity_submit_watch_resume_cancel_and_stop(self):
        capacity = await self.client.get_capacity(self.session.session_token)
        self.assertEqual(capacity.execution_slots, 2)
        submission = await self.client.submit(self.session.session_token, "r1", "a1", b"hello",
                                              context_size=128, idempotency_key="submit-1")
        self.assertTrue(submission.accepted)
        await self.provider.started.wait()
        self.provider.release.set()
        events = []
        initial_watcher = self.client.watch_progress(self.session.session_token)
        try:
            async for event in initial_watcher:
                events.append(event)
                if event.kind is EventKind.RESPONSE_FINISHED:
                    break
        finally:
            await initial_watcher.aclose()
        self.assertEqual(events[-1].result, b"HELLO")
        self.assertEqual(events[-1].generation, self.session.generation)
        await asyncio.sleep(.1)
        resumed = []
        watcher = self.client.watch_progress(self.session.session_token, events[0].sequence)
        try:
            async for event in watcher:
                resumed.append(event)
                if event.sequence == events[-1].sequence:
                    break
        finally:
            await watcher.aclose()
        self.assertEqual([e.sequence for e in resumed], [e.sequence for e in events[1:]])
        await self.client.stop_session(self.session.session_token, idempotency_key="stop")
        with self.assertRaises(ResourceManagerError):
            await self.client.get_capacity(self.session.session_token)

    async def test_server_only_selectors_and_invalid_selector_hints_fail_closed(self):
        async with ClientSession() as raw:
            url = self.client.base_url + "/resource-manager/sessions"
            headers = {"Idempotency-Key": "bad-start"}
            async with raw.post(url, json={"schedulerId": "s", "modelId": "SmolLM",
                                           "contextSizeEstimate": 128, "provider": "forbidden",
                                           "profile": "forbidden"}, headers=headers) as response:
                self.assertEqual(response.status, 400)
        async with ClientSession() as raw:
            url = self.client.base_url + f"/resource-manager/sessions/{self.session.session_token}/submissions"
            reference = "sha256:" + "0" * 64
            for selector_body in (
                {"contextSizeEstimate": 128, "bucketIdentity": "also-forbidden"},
                {},
                {"contextSizeEstimate": True},
            ):
                body = {"requestId": "r", "attemptToken": "a", "inputReference": reference, **selector_body}
                async with raw.post(url, json=body, headers={"Idempotency-Key": "ref"}) as response:
                    self.assertEqual(response.status, 400)
        # The client must let QueueScheduler's selector-less submit cross the
        # wire; server-side binding supplies the measured profile selector.
        submission = await self.client.submit(self.session.session_token, "plain", "a", b"x",
                                              idempotency_key="plain")
        self.assertTrue(submission.accepted)
        with self.assertRaises(ResourceManagerError) as wrong_hint:
            await self.client.submit(self.session.session_token, "wrong-hint", "a", b"x",
                                     bucket_identity="not-a-smollm-profile", idempotency_key="wrong-hint")
        self.assertIn(wrong_hint.exception.failure.code, {"profile_unavailable", "invalid_input"})

    async def test_missing_corrupt_symlink_oversized_references_and_stale_pre_read(self):
        url = self.client.base_url + f"/resource-manager/sessions/{self.session.session_token}/submissions"

        async def submit_reference(reference, key):
            async with ClientSession() as raw:
                return await raw.post(url, json={"requestId": key, "attemptToken": "a",
                    "inputReference": reference, "contextSizeEstimate": 128},
                    headers={"Idempotency-Key": key})

        missing = "sha256:" + "1" * 64
        response = await submit_reference(missing, "missing")
        self.assertEqual(response.status, 400); await response.release()
        corrupt_digest = "2" * 64
        (self.results.root / corrupt_digest).write_bytes(b"wrong")
        response = await submit_reference("sha256:" + corrupt_digest, "corrupt")
        self.assertEqual(response.status, 400); await response.release()
        target = self.results.root / "target"
        target.write_bytes(b"x")
        symlink_digest = "3" * 64
        (self.results.root / symlink_digest).symlink_to(target)
        response = await submit_reference("sha256:" + symlink_digest, "symlink")
        self.assertEqual(response.status, 400); await response.release()
        oversized = b"x" * 257
        oversized_digest = hashlib.sha256(oversized).hexdigest()
        (self.results.root / oversized_digest).write_bytes(oversized)
        response = await submit_reference("sha256:" + oversized_digest, "oversized")
        self.assertEqual(response.status, 400); await response.release()

        await self.client.stop_session(self.session.session_token, idempotency_key="stale-stop")
        with patch.object(self.server, "_read_reference", side_effect=AssertionError("stale read")):
            response = await submit_reference(missing, "stale")
            self.assertIn(response.status, (400, 409)); await response.release()

    async def test_replay_start_submit_cancel_and_stop_returns_same_outcomes(self):
        replay = await self.client.start_session("scheduler", ModelId.SMOLLM, self.profile, self.provider,
                                                 idempotency_key="start")
        self.assertEqual(replay, self.session)
        first = await self.client.submit(self.session.session_token, "replay", "a", b"x",
                                          context_size=128, idempotency_key="replay-submit")
        second = await self.client.submit(self.session.session_token, "replay", "a", b"x",
                                          context_size=128, idempotency_key="replay-submit")
        self.assertEqual(first, second)
        self.assertEqual(await self.client.cancel_request(self.session.session_token, "replay", idempotency_key="replay-cancel"),
                         await self.client.cancel_request(self.session.session_token, "replay", idempotency_key="replay-cancel"))
        await self.client.stop_session(self.session.session_token, idempotency_key="replay-stop")
        await self.client.stop_session(self.session.session_token, idempotency_key="replay-stop")
        with self.assertRaises(ResourceManagerError):
            await self.client.get_capacity(self.session.session_token)

    async def test_chunked_duplicate_and_nonfinite_json_are_rejected(self):
        url = self.client.base_url + "/resource-manager/sessions"
        cases = (b'{"schedulerId":"s","schedulerId":"s","modelId":"SmolLM","contextSizeEstimate":128}',
                 b'{"schedulerId":"s","modelId":"SmolLM","contextSizeEstimate":NaN}',
                 b'{"schedulerId":"s","modelId":"SmolLM","contextSizeEstimate":128}')
        async with ClientSession() as raw:
            for index, body in enumerate(cases):
                async def chunks():
                    for part in (body[:7], body[7:19], body[19:]):
                        yield part
                async with raw.post(url, data=chunks(), headers={"Content-Type": "application/json",
                                                                   "Idempotency-Key": f"json-{index}"}) as response:
                    self.assertEqual(response.status, 400 if index < 2 else 201)

    async def test_watch_delayed_event_after_heartbeats_expiry_and_released_limit(self):
        async with ClientSession() as raw:
            response = await raw.get(self.client.base_url + f"/resource-manager/sessions/{self.session.session_token}/watch",
                                     headers={"Last-Event-ID": "0"})
            first = await asyncio.wait_for(response.content.read(32), 1)
            self.assertIn(b"heartbeat", first)
            await self.client.submit(self.session.session_token, "delayed", "a", b"x",
                                     context_size=128, idempotency_key="delayed")
            self.provider.release.set()
            rest = await asyncio.wait_for(response.content.read(4096), 1)
            self.assertIn(b"data:", rest)
            response.close()
        async with ClientSession() as raw:
            for index in range(10):
                await self.client.submit(self.session.session_token, f"expire-{index}", "a", b"x",
                                         context_size=128, idempotency_key=f"expire-{index}")
                await asyncio.sleep(.01)
            async with raw.get(self.client.base_url + f"/resource-manager/sessions/{self.session.session_token}/watch",
                               headers={"Last-Event-ID": "0"}) as expired:
                self.assertEqual(expired.status, 409)
                body = await asyncio.wait_for(expired.json(), 1)
                self.assertEqual(body["error"]["code"], "cursor_expired")
            async with raw.get(self.client.base_url + f"/resource-manager/sessions/{self.session.session_token}/watch",
                               headers={"Last-Event-ID": "-1"}) as invalid:
                self.assertEqual(invalid.status, 400)

    async def test_watch_limit_rejects_and_disconnect_releases_permit(self):
        async with ClientSession() as raw:
            first = await raw.get(self.client.base_url + f"/resource-manager/sessions/{self.session.session_token}/watch")
            try:
                second = await raw.get(self.client.base_url + f"/resource-manager/sessions/{self.session.session_token}/watch",
                                       timeout=.2)
                self.assertIn(second.status, (409, 429)); await second.release()
            finally:
                first.close()
            await asyncio.sleep(.1)
            async with raw.get(self.client.base_url + f"/resource-manager/sessions/{self.session.session_token}/watch",
                               headers={"Last-Event-ID": "0"}) as replacement:
                self.assertEqual(replacement.status, 200)

    async def test_watch_limit_flood_rejects_without_semaphore_waiters(self):
        async with ClientSession() as raw:
            responses = await asyncio.gather(*(
                raw.get(self.client.base_url + f"/resource-manager/sessions/{self.session.session_token}/watch")
                for _ in range(8)
            ))
            try:
                statuses = [response.status for response in responses]
                self.assertEqual(statuses.count(200), 1)
                self.assertEqual(statuses.count(429), 7)
                waiters = getattr(self.server._watches, "_waiters", None)
                self.assertFalse(waiters)
            finally:
                for response in responses:
                    response.close()
            await asyncio.sleep(.1)

    async def test_backpressure_is_typed_and_same_key_retries_after_capacity(self):
        submissions = [await self.client.submit(self.session.session_token, str(i), "a", b"x",
                       context_size=128, idempotency_key=str(i)) for i in range(4)]
        rejected = await self.client.submit(self.session.session_token, "blocked", "a", b"x",
                                            context_size=128, idempotency_key="blocked")
        self.assertFalse(rejected.accepted); self.assertTrue(rejected.backpressure)
        self.assertEqual(rejected.request_id, "blocked")
        self.assertTrue(await self.client.cancel_request(self.session.session_token, "0", idempotency_key="cancel-0"))
        self.provider.release.set()
        await asyncio.sleep(.05)
        retry = await self.client.submit(self.session.session_token, "blocked", "a", b"x",
                                         idempotency_key="blocked")
        self.assertTrue(retry.accepted); self.assertFalse(retry.backpressure)

    async def test_queue_scheduler_over_http_submits_plain_bytes_and_publishes(self):
        await self.client.stop_session(self.session.session_token, idempotency_key="replace")
        store = QueueStore(Path(self.tmp.name) / "queue.sqlite", "scheduler-http", ModelId.SMOLLM)
        publisher = LocalPublisher(self.results, Path(self.tmp.name) / "publisher.sqlite")
        scheduler = QueueScheduler(store, self.client, self.profile, self.provider,
                                    decoder=lambda reference: reference.encode(),
                                    result_store=self.results, publisher=publisher, loop_interval=.01,
                                    stop_timeout=.05)
        try:
            await scheduler.start()
            await scheduler.enqueue("plain", "body", idempotency_key="plain-enqueue")
            await self.provider.started.wait()
            self.provider.release.set()
            for _ in range(100):
                if store.get("plain")["status"] == "done": break
                await asyncio.sleep(.01)
            self.assertEqual(store.get("plain")["status"], "done")
            reference = store.db.execute(
                "SELECT result_reference FROM handoffs WHERE request_id=?", ("plain",)
            ).fetchone()[0]
            self.assertEqual(self.results.read(reference), b"BODY")
            self.assertEqual(publisher.db.execute(
                "SELECT result_reference FROM publication_receipts"
            ).fetchone()[0], reference)
        finally:
            if scheduler._started: await scheduler.stop("test_done")
            publisher.close(); store.close()

    async def test_body_chunked_overflow_is_rejected_with_local_bound(self):
        async def chunks():
            yield b"{" + b"x" * 128
            yield b"x" * 128 + b"}"
        async with ClientSession() as raw:
            async with raw.post(self.client.base_url + "/resource-manager/sessions",
                                data=chunks(),
                                headers={"Content-Type": "application/json", "Idempotency-Key": "large"}) as response:
                self.assertEqual(response.status, 413)

    async def test_client_decodes_multiple_sse_frames_in_one_chunk(self):
        async def watch(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            event = {"sequence": 1, "completionSequence": 0, "kind": "buffered", "requestId": "r",
                     "attemptToken": "a", "sessionToken": "s", "residencyGeneration": 1,
                     "resultBase64": None, "failure": None, "timeOnGpuMs": None, "gpuTimingComplete": False}
            event2 = {**event, "sequence": 2, "kind": "cancelled"}
            await response.write((f"id: 1\ndata: {json.dumps(event)}\n\n" +
                                  f"id: 2\ndata: {json.dumps(event2)}\n\n").encode())
            await response.write_eof()
            return response

        app = web.Application(); app.router.add_get("/resource-manager/sessions/{sessionToken}/watch", watch)
        runner = web.AppRunner(app); await runner.setup(); site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        port = site._server.sockets[0].getsockname()[1]
        client = ResourceManagerHTTPClient(f"http://127.0.0.1:{port}", result_store=self.results,
                                           timeout=1, max_frame=500)
        try:
            events = [event async for event in client.watch_progress("s")]
            self.assertEqual([event.sequence for event in events], [1, 2])
        finally:
            await runner.cleanup()

    async def test_watch_first_http_error_uses_one_get_and_parses_error(self):
        calls = 0

        async def error_watch(request):
            nonlocal calls
            calls += 1
            return web.json_response({"error": {"code": "cursor_expired", "message": "expired",
                                                   "retryable": False}}, status=409)

        app = web.Application()
        app.router.add_get("/resource-manager/sessions/{sessionToken}/watch", error_watch)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        port = site._server.sockets[0].getsockname()[1]
        client = ResourceManagerHTTPClient(f"http://127.0.0.1:{port}", result_store=self.results, timeout=1)
        try:
            with self.assertRaises(ResourceManagerError) as failure:
                async for _ in client.watch_progress("arbitrary/request"):
                    pass
            self.assertEqual(failure.exception.failure.code, "cursor_expired")
            self.assertEqual(calls, 1)
        finally:
            await runner.cleanup()

    async def test_partial_chunked_json_response_is_reassembled(self):
        async def partial(request):
            response = web.StreamResponse(headers={"Content-Type": "application/json"})
            await response.prepare(request)
            await response.write(b'{"value":')
            await response.write(b' 7}')
            await response.write_eof()
            return response

        app = web.Application(); app.router.add_get("/partial", partial)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        port = site._server.sockets[0].getsockname()[1]
        client = ResourceManagerHTTPClient(f"http://127.0.0.1:{port}", result_store=self.results, timeout=1)
        try:
            self.assertEqual(await client._json("GET", "/partial"), {"value": 7})
        finally:
            await runner.cleanup()

    async def test_arbitrary_request_id_is_escaped_for_cancel_path(self):
        request_id = "request/with spaces?and#symbols"
        submission = await self.client.submit(self.session.session_token, request_id, "a", b"x",
                                              context_size=128, idempotency_key="escaped-submit")
        self.assertTrue(submission.accepted)
        self.assertTrue(await self.client.cancel_request(self.session.session_token, request_id,
                                                         idempotency_key="escaped-cancel"))

    async def test_stop_accepts_empty_body(self):
        async with ClientSession() as raw:
            async with raw.post(self.client.base_url + f"/resource-manager/sessions/{self.session.session_token}/stop",
                                data=b"", headers={"Idempotency-Key": "empty-stop"}) as response:
                self.assertEqual(response.status, 202)

    async def test_oversized_and_truncated_sse_frames_and_bad_event_id_fail_closed(self):
        event = {"sequence": 1, "completionSequence": 0, "kind": "buffered", "requestId": "r",
                 "attemptToken": "a", "sessionToken": "s", "residencyGeneration": 1,
                 "resultBase64": None, "failure": None, "timeOnGpuMs": None, "gpuTimingComplete": False}

        mode = "oversized"

        async def stream(request):
            nonlocal mode
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            if mode == "oversized":
                await response.write(b"data: " + b"x" * 600 + b"\n\n")
            elif mode == "truncated":
                await response.write(b"id: 1\ndata: " + json.dumps(event).encode())
            else:
                await response.write(b"id: 2\ndata: " + json.dumps(event).encode() + b"\n\n")
            await response.write_eof(); return response

        app = web.Application(); app.router.add_get("/resource-manager/sessions/{sessionToken}/watch", stream)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        port = site._server.sockets[0].getsockname()[1]
        client = ResourceManagerHTTPClient(f"http://127.0.0.1:{port}", result_store=self.results,
                                           timeout=1, max_frame=300)
        try:
            for next_mode, code in (("oversized", "frame_too_large"), ("truncated", "invalid_response"),
                                    ("bad-id", "invalid_response")):
                mode = next_mode
                with self.assertRaises(ResourceManagerError) as failure:
                    async for _ in client.watch_progress("s"):
                        pass
                # The standard path is intentionally exercised separately by
                # the local multi-frame server; these modes document the
                # expected fail-closed classifications.
                self.assertIn(failure.exception.failure.code, {code, "invalid_response"})
        finally:
            await runner.cleanup()

    async def test_oversized_core_watch_failure_after_headers_writes_no_error_frame(self):
        self.server.max_frame = 128

        async def failing_watch(session_token, after_sequence=0):
            del session_token, after_sequence
            await asyncio.sleep(.1)  # headers are prepared before first event is requested
            raise ResourceManagerError(Failure("test_error", "x" * 1024, False))
            yield  # pragma: no cover

        with patch.object(self.core, "watch_progress", failing_watch):
            async with ClientSession() as raw:
                async with raw.get(self.client.base_url + f"/resource-manager/sessions/{self.session.session_token}/watch",
                                   headers={"Last-Event-ID": "0"}) as response:
                    payload = await asyncio.wait_for(response.content.read(4096), 2)
                    self.assertEqual(response.status, 200)
                    self.assertNotIn(b"event: error", payload)

    async def test_sse_error_frames_are_individually_bounded_including_separator(self):
        self.server.max_frame = 128
        short_message = "x" * 10

        async def failing_watch(session_token, after_sequence=0):
            del session_token, after_sequence
            await asyncio.sleep(.1)
            raise ResourceManagerError(Failure("test_error", short_message, False))
            yield  # pragma: no cover

        with patch.object(self.core, "watch_progress", failing_watch):
            async with ClientSession() as raw:
                async with raw.get(self.client.base_url + f"/resource-manager/sessions/{self.session.session_token}/watch",
                                   headers={"Last-Event-ID": "0"}) as response:
                    payload = await asyncio.wait_for(response.read(), 2)
                    frames = [frame for frame in payload.split(b"\n\n") if frame]
                    self.assertTrue(frames)
                    self.assertTrue(all(len(frame) + 2 <= self.server.max_frame for frame in frames))
    async def test_bucket_profiles_start_for_coedit_and_gector_and_publish_response(self):
        root = Path(self.tmp.name)
        profiles = ProfileStore(root / "bucket-profiles.sqlite")
        provider = FakeProvider()
        bindings = {}
        for model in (ModelId.COEDIT, ModelId.GECTOR):
            profile, metadata = measured_profile(model=model, context=128, bucket="bucket-a")
            profiles.save_measured(profile, metadata)
            bindings[model] = ModelBinding(model, "gpu", "manifest", "model", "runtime", "adapter",
                                           profiles, provider)
        core = ResourceManager(cleanup_timeout=.05, stop_timeout=.05, max_events=16)
        server = ResourceManagerHttpServer(core, bindings=bindings, result_store=self.results,
                                           max_body=256, max_frame=4096)
        runner = web.AppRunner(server.app); await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        port = site._server.sockets[0].getsockname()[1]
        client = ResourceManagerHTTPClient(f"http://127.0.0.1:{port}", result_store=self.results, timeout=2)
        try:
            for model in (ModelId.COEDIT, ModelId.GECTOR):
                profile = bindings[model].profile_store.lookup(model, "gpu", "manifest", "model",
                                                                "runtime", "adapter", bucket_identity="bucket-a")
                session = await client.start_session(f"bucket-{model.value}", model, profile, provider,
                                                     idempotency_key=f"start-{model.value}")
                submission = await client.submit(session.session_token, f"request-{model.value}", "attempt",
                                                 b"bucket-body", idempotency_key=f"submit-{model.value}")
                self.assertTrue(submission.accepted)
                await asyncio.sleep(.01)
                provider.release.set()
                observed = []
                watcher = client.watch_progress(session.session_token)
                try:
                    async for event in watcher:
                        observed.append(event)
                        if event.kind is EventKind.RESPONSE_FINISHED:
                            break
                finally:
                    await watcher.aclose()
                self.assertEqual(observed[-1].result, b"BUCKET-BODY")
                await core.stop_session(session.session_token, idempotency_key=f"stop-{model.value}")
                provider = FakeProvider()
        finally:
            if core._session is not None:
                try:
                    await core.stop_session(core._session.session_token, idempotency_key="bucket-teardown")
                except ResourceManagerError:
                    pass
            await runner.cleanup()
            profiles.close()


if __name__ == "__main__":
    unittest.main()

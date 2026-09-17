"""Bounded Phase 2 acceptance evidence.

This module deliberately stops at the local queue/RM boundary.  It does not
claim GPU or production deployment evidence.
"""
import asyncio
import multiprocessing
import tempfile
import time
import unittest
import hashlib
import json
import sqlite3
from pathlib import Path

from aiohttp import web

from services.llm.queue.contracts import ModelId, RequestStatus
from services.llm.queue.results import LocalPublisher, ResultStore
from services.llm.queue.scheduler import DecodedPayload, DispatchContext, QueueScheduler
from services.llm.queue.store import (
    IdempotencyConflict, InvalidTransition, QueueStore, SessionError, StaleCallback,
)
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager, ResourceManagerError
from services.llm.resource_manager.http import (
    ModelBinding, ResourceManagerHTTPClient, ResourceManagerHttpServer,
)
from services.llm.resource_manager.profiles import BenchmarkMetadata, ProfileStore
from services.llm.resource_manager.protocol import ProviderResponse


def _profile():
    baseline = tuple(SampleMetadata(1, wave, 1, 10, 100, (2,)) for wave in range(4))
    warmup = tuple(SampleMetadata(concurrency, 0, concurrency, 10, 100,
                                 (2,) * concurrency) for concurrency in (1, 2))
    measured = tuple(SampleMetadata(concurrency, wave, concurrency,
                                   10 if concurrency == 1 else 20, 100,
                                   (2,) * concurrency)
                     for concurrency in (1, 2) for wave in range(1, 5))
    samples = baseline + warmup + measured
    return CapacityProfile(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime",
                           "adapter", "phase2", 1, 2, 1, 20, samples, context_size=128)


class _Provider:
    def __init__(self):
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.calls = []

    async def validate(self, profile): pass
    async def load(self, profile): pass
    async def ready(self): pass
    async def validate_input(self, payload, *, context_size, bucket_identity): pass

    async def execute(self, request_id, payload):
        self.calls.append((request_id, payload))
        self.started.set()
        await self.release.wait()
        return ProviderResponse(b"published:" + payload, None, False)

    async def cancel(self, request_id): pass
    async def unload(self): pass
    async def verify_cleanup(self): return True


def _decode(reference):
    return DecodedPayload(reference.encode(), DispatchContext(128))


def _crash_after_submit(queue_path, results_path, publisher_path, base_url):
    """Run in a child which is killed while the parent holds the HTTP reply."""
    async def run():
        queue = QueueStore(queue_path, "phase2", ModelId.SMOLLM)
        results = ResultStore(results_path)
        publisher = LocalPublisher(results, publisher_path)
        client = ResourceManagerHTTPClient(base_url, result_store=results, timeout=10)
        scheduler = QueueScheduler(queue, client, _profile(), None, decoder=_decode,
                                    result_store=results, publisher=publisher,
                                    loop_interval=.01, stop_timeout=.2)
        await scheduler.start()
        await scheduler.enqueue("request", "payload", idempotency_key="request")
        await asyncio.Event().wait()

    asyncio.run(run())


def _crash_after_receipt(queue_path, results_path, publisher_path):
    queue = QueueStore(queue_path, "phase2", ModelId.SMOLLM)
    queue.recover_session()
    publisher = LocalPublisher(ResultStore(results_path), publisher_path)
    handoff = queue.db.execute(
        "SELECT request_id,token,result_reference,idempotency_key FROM handoffs WHERE request_id='request'"
    ).fetchone()
    if handoff is None:
        raise RuntimeError("persisted handoff required")
    publisher.publish(handoff["request_id"], handoff["token"], handoff["result_reference"],
                      handoff["idempotency_key"])
    # Simulate process death after the external/local receipt commit but before
    # the queue acknowledgement transaction.
    import os
    os._exit(23)


class Phase2HTTPRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.queue_path = root / "queue.sqlite"
        self.results = ResultStore(root / "results")
        self.publisher_path = root / "publisher.sqlite"
        self.profiles = ProfileStore(root / "profiles.sqlite")
        self.addCleanup(self.profiles.close)
        fingerprint = "phase2-http"
        profile = _profile()
        identity = {"model_id": profile.model_id.value, "gpu_uuid": profile.gpu_uuid,
                    "artifact_manifest_hash": profile.artifact_manifest_hash,
                    "model_hash": profile.model_hash, "runtime_identity": profile.runtime_identity,
                    "adapter_identity": profile.adapter_identity, "context_size": profile.context_size,
                    "bucket_identity": profile.bucket_identity, "fingerprint": fingerprint}
        profile = profile.__class__(profile.model_id, profile.gpu_uuid, profile.artifact_manifest_hash,
                                    profile.model_hash, profile.runtime_identity, profile.adapter_identity,
                                    hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                                    profile.optimal_parallelism, profile.memory_safe_n, profile.buffer_capacity,
                                    profile.safety_reserve_percent, profile.raw_samples, profile.context_size)
        baseline = tuple(SampleMetadata(1, wave, 1, 10, 100, (2,))
                         for wave in range(4))
        metadata = BenchmarkMetadata(fingerprint, "2026-09-17T00:00:00Z", "test",
                                     baseline, profile.raw_samples[4:6], profile.raw_samples[6:],
                                     "context:128")
        self.profiles.save_measured(profile, metadata)
        self.provider = _Provider()
        self.core = ResourceManager(cleanup_timeout=.5, stop_timeout=.2, max_events=32)
        binding = ModelBinding(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime",
                               "adapter", self.profiles, self.provider)
        self.server = ResourceManagerHttpServer(self.core, bindings={ModelId.SMOLLM: binding},
                                                result_store=self.results, max_body=256,
                                                max_frame=4096, watch_timeout=.05,
                                                max_watches=2)
        self.runner = web.AppRunner(self.server.app)
        await self.runner.setup()
        self.addAsyncCleanup(self.runner.cleanup)
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.submit_accepted = asyncio.Event()
        self.release_submit_reply = asyncio.Event()
        # These barriers must always be released before server cleanup, even
        # when a test assertion or wait timeout aborts the test body.
        self.addCleanup(self.provider.release.set)
        self.addCleanup(self.release_submit_reply.set)
        original_submit = self.core.submit

        async def delayed_submit(*args, **kwargs):
            result = await original_submit(*args, **kwargs)
            if result.accepted and not self.submit_accepted.is_set():
                self.submit_accepted.set()
                await self.release_submit_reply.wait()
            return result

        self.core.submit = delayed_submit

    async def asyncTearDown(self):
        self.release_submit_reply.set()
        self.provider.release.set()
        if self.core._session is not None:
            await self.core.stop_session(self.core._session.session_token,
                                         idempotency_key="phase2-teardown")

    async def test_process_death_after_http_acceptance_reconciles_new_attempt(self):
        process = multiprocessing.Process(target=_crash_after_submit,
                                           args=(str(self.queue_path), str(self.results.root),
                                                 str(self.publisher_path), self.base_url))
        process.start()
        self.addCleanup(self._terminate_child, process)
        await asyncio.wait_for(self.submit_accepted.wait(), 2)
        child_queue = sqlite3.connect(f"file:{self.queue_path}?mode=ro", uri=True)
        child_queue.row_factory = sqlite3.Row
        try:
            deadline = time.monotonic() + 2
            child_row = child_queue.execute(
                "SELECT * FROM requests WHERE request_id='request'").fetchone()
            while child_row["status"] != RequestStatus.RUNNING.value:
                if time.monotonic() > deadline:
                    self.fail("child did not reach running state before crash point")
                await asyncio.sleep(.01)
                child_row = child_queue.execute(
                    "SELECT * FROM requests WHERE request_id='request'").fetchone()
            old_attempt = child_queue.execute(
                "SELECT token,session,generation FROM attempts WHERE request_id='request'"
            ).fetchone()
            old_submit = child_queue.execute(
                "SELECT acknowledged,cancelled FROM outbox WHERE kind='submit' AND token=?",
                (old_attempt[0],),
            ).fetchone()
            self.assertEqual(child_row["status"], RequestStatus.RUNNING.value)
            self.assertEqual(tuple(old_submit), (0, 0))
        finally:
            child_queue.close()
        old_token = self.core._session.session_token
        process.terminate()
        process.join(2)
        self.assertFalse(process.is_alive())

        # Let the retained RM finish old provider work only after the child is
        # gone.  Its result must be fenced by the replacement RM session.
        self.release_submit_reply.set()
        self.provider.release.set()
        await asyncio.sleep(.05)

        queue = QueueStore(self.queue_path, "phase2", ModelId.SMOLLM)
        self.addCleanup(queue.close)
        publisher = LocalPublisher(self.results, self.publisher_path)
        self.addCleanup(publisher.close)
        client = ResourceManagerHTTPClient(self.base_url, result_store=self.results, timeout=10)
        scheduler = QueueScheduler(queue, client, _profile(), None, decoder=_decode,
                                   result_store=self.results, publisher=publisher,
                                   loop_interval=.01, stop_timeout=.2)
        self.addAsyncCleanup(scheduler.stop, "test_cleanup")
        await scheduler.start()
        with self.assertRaises(ResourceManagerError) as capacity_error:
            await client.get_capacity(old_token)
        self.assertEqual(capacity_error.exception.failure.code, "scheduler_superseded")
        with self.assertRaises(ResourceManagerError) as submit_error:
            await client.submit(old_token, "request", "old", b"payload",
                                 idempotency_key="old-submit", context_size=128)
        self.assertEqual(submit_error.exception.failure.code, "scheduler_superseded")
        end = time.monotonic() + 2
        while queue.get("request")["status"] != RequestStatus.DONE.value:
            if time.monotonic() > end:
                self.fail("recovered request did not publish")
            await asyncio.sleep(.01)

        attempts = queue.db.execute(
            "SELECT token,session,generation FROM attempts WHERE request_id='request' "
            "ORDER BY started"
        ).fetchall()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(tuple(attempts[0]), tuple(old_attempt))
        self.assertNotEqual(attempts[0][0], attempts[1][0])
        handoff = queue.db.execute(
            "SELECT token,idempotency_key,result_reference FROM handoffs WHERE request_id='request'"
        ).fetchone()
        self.assertEqual(handoff[0], attempts[1][0])
        self.assertEqual(handoff[1], f"handoff:request:{attempts[1][0]}")
        self.assertEqual(handoff[2], hashlib.sha256(b"published:payload").hexdigest())
        self.assertEqual(self.results.read(handoff[2]), b"published:payload")
        self.assertEqual(publisher.db.execute(
            "SELECT idempotency_key,result_reference FROM publication_receipts"
        ).fetchall(), [(handoff[1], handoff[2])])
        old_submit = queue.db.execute(
            "SELECT acknowledged,cancelled FROM outbox WHERE kind='submit' AND token=?",
            (attempts[0][0],),
        ).fetchone()
        self.assertEqual(tuple(old_submit), (1, 1))
        self.assertEqual(len(self.provider.calls), 2)

    async def test_receipt_before_queue_ack_replays_without_provider_execution(self):
        queue = QueueStore(self.queue_path, "phase2", ModelId.SMOLLM)
        self.addCleanup(queue.close)
        session = queue.start_session()
        queue.enqueue("request", "payload", idempotency_key="request")
        token = queue.claim("request", session.token, session.generation)
        result_ref = self.results.write(b"published:payload")
        queue.stage_handoff("request", token, session.token, session.generation,
                            result_ref, "handoff:request")
        handoff = queue.db.execute(
            "SELECT token,result_reference,idempotency_key,acknowledged FROM handoffs WHERE request_id='request'"
        ).fetchone()
        self.assertEqual(tuple(handoff), (token, result_ref, "handoff:request", 0))
        self.assertEqual(result_ref, hashlib.sha256(b"published:payload").hexdigest())
        queue.close()

        process = multiprocessing.Process(target=_crash_after_receipt,
                                           args=(str(self.queue_path), str(self.results.root),
                                                 str(self.publisher_path)))
        process.start()
        self.addCleanup(self._terminate_child, process)
        process.join(2)
        self.assertEqual(process.exitcode, 23)
        queue = QueueStore(self.queue_path, "phase2", ModelId.SMOLLM)
        self.addCleanup(queue.close)
        publisher = LocalPublisher(self.results, self.publisher_path)
        self.addCleanup(publisher.close)
        persisted = queue.db.execute(
            "SELECT token,result_reference,idempotency_key,acknowledged FROM handoffs WHERE request_id='request'"
        ).fetchone()
        self.assertEqual(tuple(persisted), (token, result_ref, "handoff:request", 0))
        self.assertEqual(publisher.db.execute(
            "SELECT idempotency_key,result_reference FROM publication_receipts"
        ).fetchall(), [("handoff:request", result_ref)])
        client = ResourceManagerHTTPClient(self.base_url, result_store=self.results, timeout=10)
        scheduler = QueueScheduler(queue, client, _profile(), None, decoder=_decode,
                                   result_store=self.results, publisher=publisher,
                                   loop_interval=.01, stop_timeout=.2)
        self.addAsyncCleanup(scheduler.stop, "test_cleanup")
        await scheduler.start()
        end = time.monotonic() + 2
        while queue.get("request")["status"] != RequestStatus.DONE.value:
            if time.monotonic() > end:
                self.fail("recovered handoff did not acknowledge")
            await asyncio.sleep(.01)
        self.assertEqual(queue.get("request")["status"], RequestStatus.DONE.value)
        self.assertEqual(len(self.provider.calls), 0)
        self.assertEqual(publisher.db.execute(
            "SELECT idempotency_key,result_reference FROM publication_receipts"
        ).fetchall(), [("handoff:request", result_ref)])
        self.assertEqual(tuple(queue.db.execute(
            "SELECT token,result_reference,idempotency_key,acknowledged FROM handoffs WHERE request_id='request'"
        ).fetchone()), (token, result_ref, "handoff:request", 1))

    @staticmethod
    def _terminate_child(process):
        if process.is_alive():
            process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join(2)
        if process.is_alive():
            raise RuntimeError("child process did not terminate")
        process.close()


class Phase2OperationGuardMatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = QueueStore(Path(self.tmp.name) / "queue.sqlite", "s", ModelId.SMOLLM)
        self.session = self.store.start_session()

    def tearDown(self):
        self.store.close(); self.tmp.cleanup()

    def attempt(self, request_id="r"):
        self.store.enqueue(request_id, "payload", idempotency_key=request_id)
        return self.store.claim(request_id, self.session.token, self.session.generation)

    def test_guard_matrix_rejects_stale_claim_and_attempt_operations(self):
        self.store.enqueue("r", "payload", idempotency_key="r")
        with self.assertRaises(StaleCallback):
            self.store.claim("r", "bad", self.session.generation)
        token = self.store.claim("r", self.session.token, self.session.generation)
        operations = (
            ("mark_on_gpu", lambda t, s, g: self.store.mark_on_gpu("r", t, s, g)),
            ("finish_attempt", lambda t, s, g: self.store.finish_attempt("r", t, s, g)),
            ("retry", lambda t, s, g: self.store.retry("r", t, s, g)),
            ("stage_handoff", lambda t, s, g: self.store.stage_handoff("r", t, s, g, "0" * 64, "h")),
            ("renew_lease", lambda t, s, g: self.store.renew_lease("r", t, s, g)),
            ("transition", lambda t, s, g: self.store.transition("r", RequestStatus.ON_GPU, t, s, g)),
        )
        wrong_fences = (("token", "wrong", self.session.token, self.session.generation),
                        ("session", token, "wrong", self.session.generation),
                        ("generation", token, self.session.token, self.session.generation + 1))
        for operation_name, operation in operations:
            for fence_name, bad_token, bad_session, bad_generation in wrong_fences:
                with self.subTest(operation=operation_name, fence=fence_name):
                    with self.assertRaises(StaleCallback):
                        operation(bad_token, bad_session, bad_generation)

        with self.assertRaises(StaleCallback):
            self.store.persist_dispatch_metadata("wrong", "0" * 64)

        self.assertEqual(self.store.get("r")["status"], RequestStatus.RUNNING.value)

    def test_metadata_replay_is_immutable_and_cancelled_submit_ack_cannot_reopen(self):
        token = self.attempt()
        self.store.persist_dispatch_metadata(token, "0" * 64, 128, "bucket")
        self.store.persist_dispatch_metadata(token, "0" * 64, 128, "bucket")
        with self.assertRaises(IdempotencyConflict):
            self.store.persist_dispatch_metadata(token, "1" * 64, 128, "bucket")

        self.store.retry("r", token, self.session.token, self.session.generation)
        self.assertTrue(self.store.acknowledge_submit(token))
        submit = self.store.db.execute(
            "SELECT acknowledged,cancelled FROM outbox WHERE idempotency_key=?", (token,)
        ).fetchone()
        self.assertEqual(tuple(submit), (1, 1))
        self.assertEqual(self.store.get("r")["status"], RequestStatus.SCHEDULED.value)

    def test_persist_metadata_is_owner_fenced_after_recovery(self):
        token = self.attempt()
        recovered = QueueStore(Path(self.tmp.name) / "queue.sqlite", "s", ModelId.SMOLLM)
        self.addCleanup(recovered.close)
        recovered.recover_session()
        with self.assertRaises(SessionError):
            self.store.persist_dispatch_metadata(token, "0" * 64)
        with self.assertRaises(StaleCallback):
            recovered.persist_dispatch_metadata(token, "0" * 64)

    def test_owner_fences_acknowledgements_and_recovered_owner_replays_handoff(self):
        token = self.attempt()
        self.store.stage_handoff("r", token, self.session.token, self.session.generation,
                                 "0" * 64, "handoff")
        recovered = QueueStore(Path(self.tmp.name) / "queue.sqlite", "s", ModelId.SMOLLM)
        self.addCleanup(recovered.close)
        recovered.recover_session()

        for name, operation in (
            ("handoff", lambda: self.store.acknowledge_handoff("r", "handoff")),
            ("outbox", lambda: self.store.acknowledge_outbox(token)),
            ("submit", lambda: self.store.acknowledge_submit(token)),
        ):
            with self.subTest(operation=name):
                with self.assertRaises(SessionError):
                    operation()
        with self.assertRaises(Exception):
            recovered.acknowledge_handoff("r", "wrong")
        with self.assertRaises(Exception):
            recovered.acknowledge_outbox("wrong")
        with self.assertRaises(Exception):
            recovered.acknowledge_submit("wrong")
        recovered.acknowledge_handoff("r", "handoff")
        self.assertEqual(recovered.get("r")["status"], RequestStatus.DONE.value)

    def test_guard_matrix_allows_write_once_duplicates_and_rejects_conflicts(self):
        token = self.attempt()
        self.store.mark_on_gpu("r", token, self.session.token, self.session.generation)
        self.store.mark_on_gpu("r", token, self.session.token, self.session.generation)
        self.store.finish_attempt("r", token, self.session.token, self.session.generation)
        self.store.finish_attempt("r", token, self.session.token, self.session.generation)
        self.store.stage_handoff("r", token, self.session.token, self.session.generation, "0" * 64, "h")
        self.store.stage_handoff("r", token, self.session.token, self.session.generation, "0" * 64, "h")
        with self.assertRaises(IdempotencyConflict):
            self.store.stage_handoff("r", token, self.session.token, self.session.generation, "1" * 64, "h")
        with self.assertRaises(InvalidTransition):
            self.store.acknowledge_outbox("h")
        self.store.acknowledge_handoff("r", "h")
        self.store.acknowledge_handoff("r", "h")
        self.assertEqual(self.store.get("r")["status"], RequestStatus.DONE.value)

    def test_guard_matrix_terminal_cancel_stop_and_generic_transition(self):
        token = self.attempt("cancelled")
        self.store.cancel("cancelled", idempotency_key="cancel")
        self.store.cancel("cancelled", idempotency_key="cancel")
        with self.assertRaises(StaleCallback):
            self.store.finish_attempt("cancelled", token, self.session.token, self.session.generation)

        self.store.enqueue("queued", "payload", idempotency_key="queued")
        self.store.stop("shutdown", idempotency_key="stop")
        self.assertEqual(self.store.stop("shutdown", idempotency_key="stop"), 1)
        with self.assertRaises(IdempotencyConflict):
            self.store.stop("different", idempotency_key="stop")
        with self.assertRaises(StaleCallback):
            self.store.transition("queued", RequestStatus.RUNNING, "bad", self.session.token, self.session.generation)
        self.assertEqual(self.store.get("queued")["status"], RequestStatus.ERROR.value)


if __name__ == "__main__":
    unittest.main()

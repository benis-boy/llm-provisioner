"""Phase 3 scheduler acceptance at the local queue/ResourceManager boundary."""
import asyncio
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from services.llm.queue.contracts import FunctionDescriptor, FunctionRegistry, ModelId, RequestStatus
from services.llm.queue.eligibility import EligibilityEvaluator
from services.llm.queue.results import LocalPublisher, ResultStore
from services.llm.queue.scheduler import DecodedPayload, DispatchContext, QueueScheduler
from services.llm.queue.store import QueueStore
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager
from services.llm.resource_manager.protocol import (
    EventKind, Failure, ProgressEvent, ProviderResponse, Submission,
)


def _profile():
    return CapacityProfile(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", "phase3", 1, 1, 1, 20, (SampleMetadata(1, 0, 1, 1, 1, (1,)),), context_size=128)


class _Provider:
    def __init__(self):
        self.entered, self.release, self.finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.failure, self.failures, self.calls = None, [], []
        self.timing_complete, self.gpu_ms = True, 7
    async def validate(self, profile): pass
    async def load(self, profile): pass
    async def ready(self): pass
    async def validate_input(self, payload, *, context_size, bucket_identity): pass
    async def execute(self, request_id, payload):
        self.calls.append(request_id); self.entered.set()
        try:
            await self.release.wait()
            failure = self.failures.pop(0) if self.failures else self.failure
            if failure: raise failure
            return ProviderResponse(b"result:" + payload, self.gpu_ms, self.timing_complete)
        finally:
            self.finished.set()
    async def cancel(self, request_id): pass
    async def unload(self): pass
    async def verify_cleanup(self): return True


class Phase3SchedulerAcceptance(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(); root = Path(self.tmp.name)
        self.provider = _Provider(); self.store = QueueStore(root / "queue.sqlite", "phase3", ModelId.SMOLLM)
        self.results = ResultStore(root / "results"); self.publisher = LocalPublisher(self.results, root / "publisher.sqlite")
        self.rm = ResourceManager(cleanup_timeout=.05, stop_timeout=.05)
        self.registry = FunctionRegistry()
        self.registry.register("false", lambda **_: False)
        self.evaluator = EligibilityEvaluator(self.store, self.registry, slice_size=1)
        self.scheduler = QueueScheduler(self.store, self.rm, _profile(), self.provider,
            decoder=lambda ref: DecodedPayload(ref.encode(), DispatchContext(128)),
            result_store=self.results, publisher=self.publisher, evaluator=self.evaluator,
            loop_interval=.01, stop_timeout=.05)
        await self.scheduler.start()

    async def asyncTearDown(self):
        self.provider.release.set()
        if self.scheduler._started: await self.scheduler.stop("test_done")
        self.publisher.close(); self.store.close(); self.tmp.cleanup()

    async def wait_until(self, predicate):
        async def wait():
            while not predicate(): await asyncio.sleep(0)
        await asyncio.wait_for(wait(), .5)

    async def enqueue(self, request_id, **kwargs):
        return await self.scheduler.enqueue(request_id, request_id, idempotency_key=request_id, **kwargs)

    async def test_integrated_status_path_and_complete_gpu_timing(self):
        await self.enqueue("work")
        # scheduled is the durable enqueue event; running/on_gpu are observed
        # through the real scheduler/RM progress loops before completion.
        self.assertEqual(self.store.events(0)[-1]["kind"], "enqueue")
        await asyncio.wait_for(self.provider.entered.wait(), .5)
        await self.wait_until(lambda: self.store.get("work")["status"] == "on_gpu")
        self.provider.release.set(); await self.wait_until(lambda: self.store.get("work")["status"] == "done")
        timing = self.store.db.execute("SELECT gpu_ms,gpu_complete FROM attempts WHERE request_id='work'").fetchone()
        self.assertEqual(tuple(timing), (7, 1))

    async def test_incomplete_timing_and_progress_handler_fences_duplicate_or_stale_events(self):
        self.provider.timing_complete, self.provider.gpu_ms = False, None
        await self.enqueue("work")
        await asyncio.wait_for(self.provider.entered.wait(), .5)
        await self.wait_until(lambda: self.store.get("work")["status"] == "on_gpu")
        attempt = self.store.db.execute("SELECT token FROM attempts WHERE request_id='work'").fetchone()[0]
        session = self.scheduler.rm_session
        # The real RM admission established the attempt above.  The following
        # direct handler calls are narrower callback-fence evidence, not an
        # integrated RM-stream assertion.
        await self.scheduler._handle_event(ProgressEvent(900, 0, EventKind.ADMISSION, "work", attempt,
                                                          "stale", session.generation))
        self.provider.release.set()
        await self.wait_until(lambda: self.store.get("work")["status"] == "done")
        before = self.store.db.execute("SELECT gpu_ms,gpu_complete FROM attempts WHERE token=?", (attempt,)).fetchone()
        await self.scheduler._handle_event(ProgressEvent(901, 99, EventKind.RESPONSE_FINISHED, "work", attempt,
                                                          session.session_token, session.generation, b"late", None, 99, True))
        await self.scheduler._handle_event(ProgressEvent(901, 99, EventKind.RESPONSE_FINISHED, "work", attempt,
                                                          session.session_token, session.generation, b"late", None, 99, True))
        self.assertEqual(tuple(self.store.db.execute("SELECT gpu_ms,gpu_complete FROM attempts WHERE token=?", (attempt,)).fetchone()), tuple(before))
        self.assertEqual(self.store.get("work")["status"], RequestStatus.DONE.value)

    async def test_cancel_decoder_uncertain_submit_and_on_gpu_fence_late_work(self):
        # Decoder stage: no RM execution can begin after cancellation.
        await self.scheduler.stop("replace_decoder")
        decode_entered, decode_release = asyncio.Event(), asyncio.Event()
        async def decode(reference):
            decode_entered.set(); await decode_release.wait()
            return DecodedPayload(reference.encode(), DispatchContext(128))
        self.scheduler = QueueScheduler(self.store, self.rm, _profile(), self.provider, decoder=decode,
            result_store=self.results, publisher=self.publisher, evaluator=self.evaluator, loop_interval=.01, stop_timeout=.05)
        await self.scheduler.start(); await self.enqueue("decoder")
        await asyncio.wait_for(decode_entered.wait(), .5)
        await self.scheduler.cancel("decoder", idempotency_key="cancel-decoder"); decode_release.set()
        await self.wait_until(lambda: self.store.get("decoder")["status"] == "cancelled")
        self.assertNotIn("decoder", self.provider.calls)

        # An accepted RM submission with a withheld acknowledgement is an
        # uncertain durable submit.  Cancellation wins before that reply returns.
        original_submit, submit_accepted, release_reply = self.rm.submit, asyncio.Event(), asyncio.Event()
        async def delayed_submit(*args, **kwargs):
            reply = await original_submit(*args, **kwargs); submit_accepted.set(); await release_reply.wait(); return reply
        self.rm.submit = delayed_submit
        await self.enqueue("uncertain"); await asyncio.wait_for(submit_accepted.wait(), .5)
        await self.scheduler.cancel("uncertain", idempotency_key="cancel-uncertain"); release_reply.set()
        await self.wait_until(lambda: self.store.get("uncertain")["status"] == "cancelled")
        # RM cancellation is advisory: release its already-started provider
        # task and wait for capacity before asserting the next admission.
        self.provider.release.set()
        await asyncio.wait_for(self.provider.finished.wait(), .5)
        self.provider.release, self.provider.finished = asyncio.Event(), asyncio.Event()
        self.rm.submit = original_submit

        # An active admission remains terminal after its late provider work is
        # released.  Reset the execution barrier; uncertain has already used it.
        self.provider.entered, self.provider.finished = asyncio.Event(), asyncio.Event()
        await self.enqueue("gpu"); await asyncio.wait_for(self.provider.entered.wait(), .5)
        await self.scheduler.cancel("gpu", idempotency_key="cancel-gpu"); self.provider.release.set()
        await self.wait_until(lambda: self.store.get("gpu")["status"] == "cancelled")
        self.assertIsNone(self.store.db.execute("SELECT 1 FROM handoffs WHERE request_id='gpu'").fetchone())

    async def test_cancel_active_fences_late_completion_and_replay(self):
        await self.enqueue("work"); await asyncio.wait_for(self.provider.entered.wait(), .5)
        await self.scheduler.cancel("work", idempotency_key="cancel-work"); self.provider.release.set()
        await self.wait_until(lambda: self.store.get("work")["status"] == "cancelled")
        self.assertIsNone(self.store.db.execute("SELECT 1 FROM handoffs WHERE request_id='work' AND acknowledged=1").fetchone())
        await self.scheduler.cancel("work", idempotency_key="cancel-work")

    async def test_nonretryable_failure_stops_all_work(self):
        self.provider.failure = type("FailureError", (Exception,), {"failure": Failure("invalid_input", "invalid", False)})()
        await self.enqueue("first"); await self.enqueue("second"); await asyncio.wait_for(self.provider.entered.wait(), .5)
        self.provider.release.set(); await self.wait_until(lambda: not self.scheduler._started)
        self.assertEqual(self.store.get("first")["status"], "error"); self.assertEqual(self.store.get("second")["status"], "error")

    async def test_idle_abort_terminalizes_active_blocked_and_retry_delayed_work(self):
        await self.scheduler.stop("replace_watchdog")
        self.provider.entered = asyncio.Event()
        self.scheduler = QueueScheduler(self.store, self.rm, _profile(), self.provider,
            decoder=lambda ref: DecodedPayload(ref.encode(), DispatchContext(128)), result_store=self.results, publisher=self.publisher,
            evaluator=self.evaluator, watchdog_seconds=.03, loop_interval=.005, stop_timeout=.05)
        await self.scheduler.start()
        self.provider.entered = asyncio.Event()
        self.store.enqueue("retry", "retry", idempotency_key="retry")
        retry_token = self.store.claim("retry", self.scheduler.local_session.token,
                                       self.scheduler.local_session.generation, now=time.time())
        self.store.retry("retry", retry_token, self.scheduler.local_session.token,
                         self.scheduler.local_session.generation, now=time.time())
        await self.enqueue("blocked", ready=FunctionDescriptor("false"))
        await self.enqueue("active")
        await asyncio.wait_for(self.provider.entered.wait(), .5)
        await self.wait_until(lambda: not self.scheduler._started)
        self.assertEqual([self.store.get(request_id)["status"] for request_id in ("active", "blocked", "retry")],
                         ["error", "error", "error"])
        self.provider.release.set()
        await asyncio.sleep(.02)
        self.assertIsNone(self.publisher.db.execute("SELECT 1 FROM publication_receipts").fetchone())

    async def test_session_invalidated_aborts_active_blocked_and_retry_delayed_work(self):
        self.store.enqueue("retry", "retry", idempotency_key="retry")
        retry_token = self.store.claim("retry", self.scheduler.local_session.token,
                                       self.scheduler.local_session.generation, now=time.time())
        self.store.retry("retry", retry_token, self.scheduler.local_session.token,
                         self.scheduler.local_session.generation, now=time.time())
        await self.enqueue("blocked", ready=FunctionDescriptor("false"))
        await self.enqueue("active")
        await asyncio.wait_for(self.provider.entered.wait(), .5)
        session = self.scheduler.rm_session
        await self.rm.stop_session(session.session_token, reason="replacement",
                                   idempotency_key="external-replacement")
        await self.wait_until(lambda: not self.scheduler._started)
        self.assertEqual([self.store.get(request_id)["status"] for request_id in ("active", "blocked", "retry")],
                         ["error", "error", "error"])

    async def test_cancel_buffered_handoff_and_terminal_noops_do_not_publish_late_work(self):
        # p=1 admits one active and one buffered request through the real RM.
        await self.enqueue("active")
        await asyncio.wait_for(self.provider.entered.wait(), .5)
        await self.enqueue("buffered")
        await self.wait_until(lambda: len(self.store.active_attempts()) == 2)
        await self.scheduler.cancel("buffered", idempotency_key="cancel-buffered")
        self.provider.release.set()
        await self.wait_until(lambda: self.store.get("active")["status"] == "done")
        self.assertEqual(self.store.get("buffered")["status"], "cancelled")
        self.assertNotIn("buffered", self.provider.calls)

        # Pause a real publication worker after a durable handoff exists.
        entered, release = threading.Event(), threading.Event()
        original_publish = self.scheduler._publish_guarded
        def paused_publish(row, session):
            entered.set(); release.wait(.5)
            return original_publish(row, session)
        self.scheduler._publish_guarded = paused_publish
        self.provider.entered = asyncio.Event()
        await self.enqueue("handoff")
        await asyncio.wait_for(self.provider.entered.wait(), .5)
        await self.wait_until(lambda: self.store.db.execute("SELECT 1 FROM handoffs WHERE request_id='handoff'").fetchone())
        await asyncio.to_thread(entered.wait, .5)
        await self.scheduler.cancel("handoff", idempotency_key="cancel-handoff")
        release.set()
        await self.wait_until(lambda: self.store.get("handoff")["status"] == "cancelled")
        await asyncio.sleep(.02)
        self.assertIsNone(self.publisher.db.execute(
            "SELECT 1 FROM publication_receipts WHERE idempotency_key LIKE 'handoff:handoff:%'").fetchone())
        # Replays against cancelled and completed terminal records are no-ops.
        await self.scheduler.cancel("handoff", idempotency_key="cancel-handoff")
        await self.scheduler.cancel("active", idempotency_key="cancel-done")
        self.assertEqual(self.store.get("active")["status"], "done")

    async def test_fifo_skips_blocked_head_and_unavailable_function_is_local(self):
        await self.enqueue("blocked", ready=FunctionDescriptor("false")); await self.enqueue("ready")
        await asyncio.wait_for(self.provider.entered.wait(), .5); self.assertEqual(self.provider.calls, ["ready"])
        await self.enqueue("missing", ready=FunctionDescriptor("not-registered"))
        await self.wait_until(lambda: self.store.get("missing")["status"] == "error")
        self.assertEqual(self.store.get("missing")["error_code"], "function_unavailable")

    async def test_integrated_skip_line_dispatches_before_its_anchor(self):
        # Synchronous durable insertion prevents dispatch from observing a
        # partial queue; execution order is then observed from real RM calls.
        self.store.enqueue("anchor", "anchor", idempotency_key="anchor")
        self.store.enqueue("tail", "tail", idempotency_key="tail")
        self.store.enqueue("skip-a", "skip-a", insertion_mode="skip-line", idempotency_key="skip-a")
        self.store.enqueue("skip-b", "skip-b", insertion_mode="skip-line", idempotency_key="skip-b")
        self.scheduler._wake.set()
        await asyncio.wait_for(self.provider.entered.wait(), .5)
        self.assertEqual(self.provider.calls[0], "skip-a")
        self.provider.release.set()
        await self.wait_until(lambda: self.store.get("tail")["status"] == "done")
        self.assertEqual(self.provider.calls, ["skip-a", "skip-b", "anchor", "tail"])

    async def test_retry_backoff_budget_and_submit_backpressure_preserve_intent(self):
        """Provider failures use durable wall time; RM watchdog time is separate."""
        retryable = type("Retryable", (Exception,), {
            "failure": Failure("provider_busy", "try again", True),
        })()
        self.provider.failures = [retryable] * 5
        wall = [1000.0]
        with patch("services.llm.queue.store.time.time", side_effect=lambda: wall[0]):
            await self.enqueue("retry")
            for count, delay in enumerate((5, 10, 20, 30), 1):
                self.provider.release.set()
                await self.wait_until(lambda count=count: self.store.get("retry")["retry_count"] == count)
                row = self.store.get("retry")
                self.assertEqual(row["next_attempt_at"], wall[0] + delay)
                self.provider.release = asyncio.Event()
                wall[0] = row["next_attempt_at"]
                self.scheduler._wake.set()
            # The fifth provider failure is recorded after the five-minute
            # budget, without a sixth execution or a new durable intent.
            await self.wait_until(lambda: len(self.provider.calls) == 5)
            wall[0] = 1301.0
            self.provider.release.set()
            await self.wait_until(lambda: self.store.get("retry")["status"] == "error")
        self.assertEqual(self.store.get("retry")["error_code"], "retry_exhausted")
        self.assertEqual(len(self.provider.calls), 5)

        # A rejected public submit is backpressure, not a retry.  The same
        # attempt/outbox row is eventually accepted and then completes.
        original_submit = self.rm.submit
        rejected = asyncio.Event()
        reject_once = {"value": True}
        async def backpressure_once(*args, **kwargs):
            if reject_once["value"]:
                reject_once["value"] = False
                rejected.set()
                return Submission(False, args[1], args[2], self.scheduler.rm_session.session_token,
                                  self.scheduler.rm_session.generation, True)
            return await original_submit(*args, **kwargs)
        self.rm.submit = backpressure_once
        self.provider.failures = []
        self.provider.failure = None
        self.provider.entered = asyncio.Event()
        await self.enqueue("backpressure")
        await asyncio.wait_for(rejected.wait(), .5)
        row = self.store.get("backpressure")
        token = self.store.db.execute(
            "SELECT token FROM attempts WHERE request_id='backpressure'"
        ).fetchone()[0]
        self.assertEqual(row["retry_count"], 0)
        self.assertIsNone(row["next_attempt_at"])
        self.assertIsNone(row["error_code"])
        self.assertEqual(self.store.db.execute(
            "SELECT COUNT(*) FROM outbox WHERE request_id='backpressure' AND kind='submit'"
        ).fetchone()[0], 1)
        self.provider.release.set()
        await asyncio.wait_for(self.provider.entered.wait(), .5)
        await self.wait_until(lambda: self.store.get("backpressure")["status"] == "done")
        self.assertEqual(self.store.db.execute(
            "SELECT COUNT(*) FROM attempts WHERE request_id='backpressure' AND token=?", (token,)
        ).fetchone()[0], 1)

    async def test_dependency_template_and_awaited_readiness_invalidation_order(self):
        """An invalidated awaited scan restarts at the now-ready FIFO head."""
        ready_started, ready_release, head_ready = asyncio.Event(), asyncio.Event(), {"value": False}
        self.registry.register("head", lambda **_: head_ready["value"])
        async def later_ready(**_):
            ready_started.set(); await ready_release.wait()
            return True
        self.registry.register("later-ready", later_ready)
        self.registry.register("ready", lambda **_: True)
        self.registry.register("template", lambda **_: "rendered")

        await self.enqueue("dependency")
        self.provider.release.set()
        await self.wait_until(lambda: self.store.get("dependency")["status"] == "done")
        await self.enqueue("gated", dependencies=("dependency",),
                           ready=FunctionDescriptor("ready"),
                           template=FunctionDescriptor("template", dependency_result_ids=("dependency",)))
        await self.enqueue("head", ready=FunctionDescriptor("head"))
        await self.enqueue("later", ready=FunctionDescriptor("later-ready"))
        await asyncio.wait_for(ready_started.wait(), .5)
        head_ready["value"] = True
        self.evaluator.invalidate()
        ready_release.set()
        await self.wait_until(lambda: self.store.get("gated")["status"] == "done")
        await self.wait_until(lambda: self.store.get("later")["status"] == "done")
        self.assertEqual(self.provider.calls[:4], ["dependency", "gated", "head", "later"])

    async def test_recovered_missing_descriptor_is_request_local_and_valid_work_completes(self):
        # Simulate abrupt coordinator loss after accepting durable descriptor
        # intent. Public stop is deliberately not used because it terminalizes
        # outstanding work rather than exercising recovery.
        await self.scheduler._evaluation_lock.acquire()
        try:
            self.store.enqueue("recovered-missing", "recovered-missing", idempotency_key="recovered-missing",
                               ready=FunctionDescriptor("removed-at-restart"))
            self.store.enqueue("recovered-valid", "recovered-valid", idempotency_key="recovered-valid",
                               ready=FunctionDescriptor("ready"))
        finally:
            # The old scheduler tasks are cancelled while the evaluation fence
            # is held, so accepted intent has not been evaluated pre-restart.
            for task in tuple(self.scheduler._tasks):
                task.cancel()
            self.scheduler._evaluation_lock.release()
        for task in tuple(self.scheduler._tasks):
            task.cancel()
        await asyncio.gather(*self.scheduler._tasks, return_exceptions=True)
        self.scheduler._tasks.clear()
        self.scheduler._started, self.scheduler._stopping = False, True
        self.store.close()
        self.store = QueueStore(Path(self.tmp.name) / "queue.sqlite", "phase3", ModelId.SMOLLM)
        registry = FunctionRegistry()
        registry.register("ready", lambda **_: True)
        evaluator = EligibilityEvaluator(self.store, registry, slice_size=1, poll_interval=.01)
        self.scheduler = QueueScheduler(
            self.store, self.rm, _profile(), self.provider,
            decoder=lambda ref: DecodedPayload(ref.encode(), DispatchContext(128)),
            result_store=self.results, publisher=self.publisher, evaluator=evaluator,
            loop_interval=.01, stop_timeout=.05,
        )
        await self.scheduler.start()
        self.provider.release.set()
        await self.wait_until(lambda: self.store.get("recovered-missing")["status"] == "error")
        await self.wait_until(lambda: self.store.get("recovered-valid")["status"] == "done")
        self.assertEqual(self.store.get("recovered-missing")["error_code"], "function_unavailable")

    async def test_concurrent_independent_skip_line_inserts_dispatch_by_recorded_sequence(self):
        # Keep the scheduler's public dispatch path alive while an independent
        # SQLite owner records two concurrent append transactions.  The
        # durable insertion sequence, rather than task completion timing, is
        # the ordering contract.
        await self.enqueue("gate", ready=FunctionDescriptor("false"))
        barrier = threading.Barrier(2)
        session, path = self.store.session, self.store.path
        def insert(request_id):
            writer = QueueStore(path, "phase3", ModelId.SMOLLM)
            writer.session = session
            try:
                barrier.wait()
                writer.enqueue(request_id, request_id, insertion_mode="skip-line", idempotency_key=request_id)
            finally:
                writer.close()
        await asyncio.gather(asyncio.to_thread(insert, "concurrent-a"),
                             asyncio.to_thread(insert, "concurrent-b"))
        positions = {
            row["request_id"]: row["insertion_seq"]
            for row in self.store.list_positions()
            if row["request_id"] in {"concurrent-a", "concurrent-b"}
        }
        self.provider.release.set()
        self.scheduler._wake.set()
        await self.wait_until(lambda: self.store.get("concurrent-a")["status"] == "done")
        await self.wait_until(lambda: self.store.get("concurrent-b")["status"] == "done")
        observed = [request_id for request_id in self.provider.calls
                    if request_id in positions]
        self.assertEqual(observed, sorted(positions, key=positions.get))


if __name__ == "__main__": unittest.main()

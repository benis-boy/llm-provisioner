"""Deterministic scheduler integration tests over SQLite, real RM, and publisher."""
import asyncio
import tempfile
import time
import unittest
import warnings
from pathlib import Path

from services.llm.queue.contracts import FunctionDescriptor, FunctionRegistry, ModelId
from services.llm.queue.eligibility import EligibilityEvaluator
from services.llm.queue.results import LocalPublisher, ResultStore
from services.llm.queue.scheduler import DecodedPayload, DispatchContext, QueueScheduler
from services.llm.queue.store import QueueStore
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager
from services.llm.resource_manager.protocol import ProviderResponse


def profile(p=1):
    return CapacityProfile(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", "profile",
                           p, p, p, 20, (SampleMetadata(1, 0, 1, 1, 1, (1,)),), context_size=128)


class Provider:
    def __init__(self):
        self.release = asyncio.Event()
        self.calls = []
        self.fail = {}
        self.ignore_cancel = False

    async def validate(self, profile): pass
    async def load(self, profile): pass
    async def ready(self): pass
    async def validate_input(self, payload, *, context_size, bucket_identity): pass
    async def execute(self, request_id, payload):
        self.calls.append(("execute", request_id, payload))
        failure = self.fail.pop(request_id, None)
        if failure: raise failure
        await self.release.wait()
        return ProviderResponse(b"result:" + payload, None, False)
    async def cancel(self, request_id):
        self.calls.append(("cancel", request_id))
        if self.ignore_cancel: await asyncio.Event().wait()
    async def unload(self): self.calls.append(("unload",))
    async def verify_cleanup(self): return True


class SchedulerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.provider = Provider()
        self.store = QueueStore(root / "queue.sqlite", "scheduler", ModelId.SMOLLM)
        self.results = ResultStore(root / "results")
        self.publisher = LocalPublisher(self.results, root / "publisher.sqlite")
        self.rm = ResourceManager(cleanup_timeout=.05, stop_timeout=.05)
        self.scheduler = QueueScheduler(self.store, self.rm, profile(), self.provider,
                                        decoder=lambda ref: DecodedPayload(ref.encode(), DispatchContext(128)),
                                        result_store=self.results, publisher=self.publisher, loop_interval=.01,
                                        stop_timeout=.05)
        await self.scheduler.start()

    async def asyncTearDown(self):
        if self.scheduler._started:
            await self.scheduler.stop("test_done")
        self.publisher.close(); self.store.close(); self.directory.cleanup()

    async def wait_for(self, predicate, timeout=.5):
        end = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() > end: self.fail("condition did not become true")
            await asyncio.sleep(.005)

    async def enqueue(self, request_id="r", payload="payload", **kwargs):
        await self.scheduler.enqueue(request_id, payload, idempotency_key=request_id, **kwargs)

    async def test_end_to_end_done_and_local_receipt(self):
        await self.enqueue(); await self.wait_for(lambda: self.provider.calls)
        self.provider.release.set(); await self.wait_for(lambda: self.store.get("r")["status"] == "done")
        self.assertEqual(self.results.read(self.store.db.execute("SELECT result_reference FROM handoffs WHERE request_id='r'").fetchone()[0]), b"result:payload")
        self.assertIsNotNone(self.publisher.db.execute(
            "SELECT 1 FROM publication_receipts WHERE idempotency_key LIKE 'handoff:%'"
        ).fetchone())

    async def test_same_database_publication_commits_receipt_and_done_together(self):
        await self.scheduler.stop("replace_publisher")
        self.publisher.close()
        self.publisher = LocalPublisher(self.results, Path(self.directory.name) / "queue.sqlite")
        self.scheduler = QueueScheduler(self.store, self.rm, profile(), self.provider,
                                        decoder=lambda ref: DecodedPayload(ref.encode(), DispatchContext(128)),
                                        result_store=self.results, publisher=self.publisher, loop_interval=.01,
                                        stop_timeout=.05)
        await self.scheduler.start()
        await self.enqueue(); await self.wait_for(lambda: self.provider.calls)
        self.provider.release.set(); await self.wait_for(lambda: self.store.get("r")["status"] == "done")
        self.assertIsNotNone(self.store.db.execute(
            "SELECT 1 FROM publication_receipts WHERE idempotency_key LIKE 'handoff:%'"
        ).fetchone())

    async def test_incomplete_gpu_timing_stays_null(self):
        await self.enqueue(); await self.wait_for(lambda: self.provider.calls); self.provider.release.set()
        await self.wait_for(lambda: self.store.get("r")["status"] == "done")
        self.assertIsNone(self.store.db.execute("SELECT gpu_ms FROM attempts WHERE request_id='r'").fetchone()[0])

    async def test_capacity_occupancy_prevents_extra_attempt(self):
        await self.enqueue("a"); await self.enqueue("b"); await self.enqueue("c")
        await self.wait_for(lambda: len([x for x in self.provider.calls if x[0] == "execute"]) == 1)
        self.assertEqual(len([x for x in self.provider.calls if x[0] == "execute"]), 1)

    async def test_fifo_blocked_by_dependency(self):
        await self.enqueue("parent"); await self.enqueue("child", dependencies=("parent",))
        await self.wait_for(lambda: self.provider.calls); self.assertEqual(self.provider.calls[0][1], "parent")

    async def test_completion_arms_and_resets_watchdog_only(self):
        await self.enqueue(); await self.wait_for(lambda: self.scheduler._watchdog_deadline is not None)
        first = self.scheduler._watchdog_deadline; self.provider.release.set()
        await self.wait_for(lambda: self.store.get("r")["status"] == "done")
        # Completion resets the deadline, and the watchdog then disarms once
        # there is no eligible or in-flight work left.
        self.assertEqual(self.scheduler._completion_sequence, 1)
        await self.wait_for(lambda: self.scheduler._watchdog_deadline is None)
        self.assertIsNone(self.scheduler._watchdog_deadline)
        self.assertIsNotNone(first)

    async def test_cancel_scheduled_has_no_rm_call(self):
        await self.enqueue(); await self.scheduler.cancel("r")
        await asyncio.sleep(.03)
        self.assertEqual(self.store.get("r")["status"], "cancelled")
        self.assertFalse(any(call[0] == "execute" for call in self.provider.calls))

    async def test_cancel_active_calls_rm_and_ignores_late_result(self):
        await self.enqueue(); await self.wait_for(lambda: self.provider.calls)
        await self.scheduler.cancel("r"); self.provider.release.set(); await asyncio.sleep(.05)
        self.assertEqual(self.store.get("r")["status"], "cancelled")
        self.assertFalse(self.store.db.execute("SELECT 1 FROM handoffs WHERE request_id='r'").fetchone())

    async def test_cancel_is_idempotent(self):
        await self.enqueue(); await self.scheduler.cancel("r"); await self.scheduler.cancel("r")
        self.assertEqual(self.store.get("r")["status"], "cancelled")

    async def test_retry_only_failing_request(self):
        from services.llm.resource_manager.protocol import Failure
        self.provider.fail["a"] = type("E", (Exception,), {"failure": Failure("temporary", "temporary", True)})()
        await self.enqueue("a"); await self.enqueue("b"); await self.wait_for(lambda: any(x[1] == "a" for x in self.provider.calls))
        self.assertIn(self.store.get("b")["status"], {"scheduled", "running"})

    async def test_nonretryable_failure_stops_whole_queue(self):
        from services.llm.resource_manager.protocol import Failure
        self.provider.fail["a"] = type("E", (Exception,), {"failure": Failure("fatal", "fatal", False)})()
        await self.enqueue("a"); await self.enqueue("b")
        await self.wait_for(lambda: not self.scheduler._started)
        self.assertEqual(self.store.get("b")["status"], "error")

    async def test_lost_submit_ack_replays_same_attempt_key(self):
        original = self.rm.submit
        calls = 0

        async def accept_then_lose_ack(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = await original(*args, **kwargs)
            if calls == 1:
                raise OSError("ack lost after acceptance")
            return result

        self.rm.submit = accept_then_lose_ack
        await self.enqueue(); await self.wait_for(lambda: self.provider.calls)
        # An accepted submission with a lost acknowledgement is still an
        # actual in-flight attempt and must remain eligible for lease renewal.
        self.assertEqual(len(self.store.active_attempts()), 1)
        self.provider.release.set()
        await self.wait_for(lambda: self.store.get("r")["status"] == "done")
        attempt = self.store.db.execute("SELECT token FROM attempts WHERE request_id='r'").fetchone()[0]
        submit = self.store.db.execute(
            "SELECT o.idempotency_key,d.payload_digest,d.context_size,d.bucket_identity "
            "FROM outbox o JOIN dispatch_metadata d ON d.token=o.token "
            "WHERE o.kind='submit' AND o.request_id='r'"
        ).fetchone()
        self.assertEqual(submit[0], attempt)
        self.assertEqual(submit[1], self.results.write(b"payload"))
        self.assertEqual(tuple(submit[2:]), (128, None))
        self.assertEqual(calls, 2)
        self.assertEqual(len([x for x in self.provider.calls if x[0] == "execute"]), 1)

    async def test_decode_cancel_race_never_submits(self):
        await self.scheduler.stop("replace_decoder")
        entered, release = asyncio.Event(), asyncio.Event()

        async def decode(reference):
            entered.set()
            await release.wait()
            return reference.encode()

        self.scheduler = QueueScheduler(self.store, self.rm, profile(), self.provider,
                                        decoder=decode, result_store=self.results,
                                        publisher=self.publisher, loop_interval=.01,
                                        stop_timeout=.05)
        await self.scheduler.start()
        await self.enqueue(); await asyncio.wait_for(entered.wait(), .2)
        await self.scheduler.cancel("r"); release.set(); await asyncio.sleep(.03)
        self.assertEqual(self.store.get("r")["status"], "cancelled")
        self.assertFalse(any(call[0] == "execute" for call in self.provider.calls))

    async def test_cancelled_decoder_then_release_disarms_watchdog_with_blocked_work(self):
        await self.scheduler.stop("replace_decoder")
        entered, release = asyncio.Event(), asyncio.Event()

        async def decode(reference):
            entered.set()
            await release.wait()
            return reference.encode()

        self.scheduler = QueueScheduler(self.store, self.rm, profile(), self.provider,
                                        decoder=decode, result_store=self.results,
                                        publisher=self.publisher, loop_interval=.01,
                                        stop_timeout=.05, watchdog_seconds=.05)
        await self.scheduler.start()
        await self.enqueue("r")
        await asyncio.wait_for(entered.wait(), .2)
        await self.scheduler.cancel("r")
        # This request is permanently blocked by the cancelled dependency;
        # it must not keep the watchdog armed after the decoder is released.
        await self.enqueue("blocked", dependencies=("r",))
        release.set()
        await self.wait_for(lambda: self.store.get("r")["status"] == "cancelled")
        await self.wait_for(lambda: self.scheduler._watchdog_deadline is None)
        self.assertFalse(any(call[0] == "execute" for call in self.provider.calls))

    async def test_result_and_handoff_transient_failures_publish_once(self):
        original_write = self.results.write
        original_stage = self.store.stage_handoff
        write_failures = 0
        stage_failures = 0

        def flaky_write(value):
            nonlocal write_failures
            if value == b"result:payload" and write_failures == 0:
                write_failures += 1
                raise OSError("temporary result storage failure")
            return original_write(value)

        def flaky_stage(*args, **kwargs):
            nonlocal stage_failures
            if stage_failures == 0:
                stage_failures += 1
                raise __import__("sqlite3").OperationalError("temporary handoff failure")
            return original_stage(*args, **kwargs)

        self.results.write = flaky_write
        self.store.stage_handoff = flaky_stage
        await self.enqueue()
        await self.wait_for(lambda: self.provider.calls)
        self.provider.release.set()
        await self.wait_for(lambda: self.store.get("r")["status"] == "done")
        self.assertEqual(write_failures, 1)
        self.assertEqual(stage_failures, 1)
        self.assertEqual(len([call for call in self.provider.calls if call[0] == "execute"]), 1)

    async def test_stop_uses_one_deadline_without_unhandled_warnings(self):
        await self.enqueue()
        await self.wait_for(lambda: self.provider.calls)
        self.provider.ignore_cancel = True
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await self.scheduler.stop("shutdown")
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, .12)
        self.assertFalse([warning for warning in caught if issubclass(warning.category, RuntimeWarning)])
        self.assertIn(self.store.get("r")["status"], {"error", "cancelled"})

    async def test_reopen_pending_handoff_does_not_execute_provider(self):
        # Install the deterministic publication pause before completion can
        # race the handoff observer.
        self.scheduler._publish_guarded = lambda row, *_: (_ for _ in ()).throw(OSError("publication paused"))
        await self.enqueue(); await self.wait_for(lambda: self.provider.calls); self.provider.release.set()
        await self.wait_for(lambda: self.store.db.execute("SELECT 1 FROM handoffs WHERE request_id='r'").fetchone())
        # Simulate a crash before publication: do not call public stop(),
        # because explicit stop terminalizes every nonterminal request.
        await asyncio.sleep(.02)
        self.assertEqual(self.store.get("r")["status"], "running")
        calls = len([x for x in self.provider.calls if x[0] == "execute"])
        for task in tuple(self.scheduler._tasks): task.cancel()
        await asyncio.gather(*self.scheduler._tasks, return_exceptions=True)
        self.scheduler._tasks.clear(); self.scheduler._started = False; self.scheduler._stopping = True
        self.store.close(); self.store = QueueStore(Path(self.directory.name) / "queue.sqlite", "scheduler", ModelId.SMOLLM)
        self.publisher.close(); self.publisher = LocalPublisher(self.results, Path(self.directory.name) / "publisher.sqlite")
        self.scheduler = QueueScheduler(self.store, self.rm, profile(), self.provider, decoder=lambda ref: ref.encode(), result_store=self.results, publisher=self.publisher, loop_interval=.01)
        await self.scheduler.start(); await self.wait_for(lambda: self.store.get("r")["status"] == "done")
        self.assertEqual(len([x for x in self.provider.calls if x[0] == "execute"]), calls)

    async def test_explicit_stop_publishing_terminalizes_and_cannot_publish(self):
        # Pause publication before completion can make a worker runnable. The
        # prior ordering patched after observing handoff and therefore allowed
        # a legitimate publication win before stop started.
        self.scheduler._publish_guarded = lambda row, *_: (_ for _ in ()).throw(OSError("publication paused"))
        await self.enqueue(); await self.wait_for(lambda: self.provider.calls); self.provider.release.set()
        await self.wait_for(lambda: self.store.db.execute("SELECT 1 FROM handoffs WHERE request_id='r'").fetchone())
        await self.scheduler.stop("explicit_stop")
        self.assertEqual(self.store.get("r")["status"], "error")
        receipt = self.store.db.execute("SELECT acknowledged,cancelled FROM outbox WHERE kind='handoff'").fetchone()
        self.assertEqual((receipt["acknowledged"], receipt["cancelled"]), (1, 1))
        self.assertIsNone(self.publisher.db.execute(
            "SELECT 1 FROM publication_receipts WHERE idempotency_key LIKE 'handoff:%'"
        ).fetchone())

    async def test_restart_gets_independent_local_session(self):
        old = self.scheduler.local_session; await self.scheduler.stop("restart")
        self.scheduler = QueueScheduler(self.store, self.rm, profile(), self.provider, decoder=lambda ref: ref.encode(), result_store=self.results, publisher=self.publisher, loop_interval=.01)
        await self.scheduler.start(); self.assertNotEqual(old.token, self.scheduler.local_session.token)

    async def test_watch_replays_durable_cursor(self):
        await self.enqueue()
        stream = self.scheduler.watch(0); event = await asyncio.wait_for(stream.__anext__(), .2)
        self.assertEqual(event["cursor"], 1)
        await stream.aclose()

    async def test_stop_marks_all_and_is_bounded(self):
        await self.enqueue(); await self.wait_for(lambda: self.provider.calls); self.provider.ignore_cancel = True
        await asyncio.wait_for(self.scheduler.stop("shutdown"), .2)
        self.assertIn(self.store.get("r")["status"], {"error", "cancelled"})

    async def test_false_readiness_does_not_arm_watchdog(self):
        await self.scheduler.stop("replace_evaluator")
        registry = FunctionRegistry(); registry.register("never", lambda **_: False)
        self.scheduler = QueueScheduler(self.store, self.rm, profile(), self.provider,
                                        decoder=lambda ref: ref.encode(), result_store=self.results,
                                        publisher=self.publisher,
                                        evaluator=EligibilityEvaluator(self.store, registry),
                                        loop_interval=.01, stop_timeout=.05)
        await self.scheduler.start()
        await self.scheduler.enqueue("r", "p", idempotency_key="r", ready=FunctionDescriptor("never"))
        await asyncio.sleep(.03)
        self.assertIsNone(self.scheduler._watchdog_deadline)


if __name__ == "__main__":
    unittest.main()

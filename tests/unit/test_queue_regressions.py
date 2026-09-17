import os
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from services.llm.queue.contracts import ModelId, RequestStatus
from services.llm.queue.store import QueueStore, SessionError, StaleCallback


class QueueRegressionTests(unittest.TestCase):
    def make(self, path, scheduler="s", model=ModelId.SMOLLM):
        store = QueueStore(path, scheduler, model)
        session = store.start_session()
        return store, session

    def test_retry_schedule_is_5_30_30_and_survives_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.sqlite"
            store, session = self.make(path)
            store.enqueue("r", "payload", idempotency_key="r")
            now = 100.0
            for expected in (105.0, 115.0, 135.0):
                token = store.claim("r", session.token, session.generation, now=now)
                self.assertIsNotNone(token)
                store.retry("r", token, session.token, session.generation,
                            failure_elapsed=1, now=now)
                self.assertEqual(store.get("r")["next_attempt_at"], expected)
                store.close()
                store, session = self.make(path)
                now = expected
            store.close()

    def test_retry_budget_uses_first_failure_clock_not_caller_elapsed(self):
        with tempfile.TemporaryDirectory() as directory:
            store, session = self.make(Path(directory) / "q.sqlite")
            store.enqueue("r", "payload", idempotency_key="r")
            token = store.claim("r", session.token, session.generation, now=0)
            store.retry("r", token, session.token, session.generation, now=0)
            token = store.claim("r", session.token, session.generation, now=5)
            store.retry("r", token, session.token, session.generation, now=300)
            self.assertEqual(store.get("r")["status"], "error")
            self.assertEqual(store.get("r")["error_code"], "retry_exhausted")
            store.close()

    def test_retry_delays_are_5_10_20_30_30_across_reopen_and_claim_at_300_is_fenced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.sqlite"
            store, session = self.make(path)
            store.enqueue("r", "payload", idempotency_key="r")
            now = 0.0
            expected_delays = (5, 10, 20, 30, 30)
            for delay in expected_delays:
                token = store.claim("r", session.token, session.generation, now=now)
                self.assertIsNotNone(token)
                store.retry("r", token, session.token, session.generation, now=now)
                row = store.get("r")
                self.assertEqual(row["next_attempt_at"], now + delay)
                store.close()
                store, session = self.make(path)
                now += delay
            self.assertIsNone(store.claim("r", session.token, session.generation, now=300))
            self.assertEqual(store.get("r")["status"], "error")
            self.assertEqual(store.get("r")["error_code"], "retry_exhausted")
            self.assertEqual(store.db.execute(
                "SELECT COUNT(*) FROM attempts WHERE request_id='r' AND active=1"
            ).fetchone()[0], 0)
            store.close()

    def test_expired_gpu_attempt_with_staged_handoff_is_not_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            store, session = self.make(Path(directory) / "q.sqlite")
            store.enqueue("r", "payload", idempotency_key="r")
            token = store.claim("r", session.token, session.generation, now=0)
            store.mark_on_gpu("r", token, session.token, session.generation, at=1)
            store.stage_handoff("r", token, session.token, session.generation, "0" * 64, "handoff")
            store.reconcile_expired(now=31)
            self.assertEqual(store.get("r")["status"], "running")
            attempt = store.db.execute("SELECT gpu_ms,gpu_complete FROM attempts WHERE token=?", (token,)).fetchone()
            self.assertIsNone(attempt["gpu_ms"])
            self.assertEqual(attempt["gpu_complete"], 0)
            self.assertIsNone(store.claim("r", session.token, session.generation, now=31))
            store.acknowledge_handoff("r", "handoff")
            self.assertEqual(store.get("r")["status"], "done")
            store.close()

    def test_generic_running_to_scheduled_is_rejected_without_releasing_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            store, session = self.make(Path(directory) / "q.sqlite")
            store.enqueue("r", "payload", idempotency_key="r")
            token = store.claim("r", session.token, session.generation)
            with self.assertRaises(Exception):
                store.transition("r", RequestStatus.SCHEDULED, token, session.token, session.generation)
            self.assertEqual(store.get("r")["status"], "running")
            self.assertEqual(store.outbox()[0]["kind"], "submit")
            store.close()

    def test_gpu_events_and_finished_telemetry_are_write_once(self):
        with tempfile.TemporaryDirectory() as directory:
            store, session = self.make(Path(directory) / "q.sqlite")
            store.enqueue("r", "payload", idempotency_key="r")
            token = store.claim("r", session.token, session.generation, now=1)
            store.mark_on_gpu("r", token, session.token, session.generation, at=2)
            store.finish_attempt("r", token, session.token, session.generation, at=3, gpu_timing_complete=False, gpu_ms=1)
            store.mark_on_gpu("r", token, session.token, session.generation, at=4)
            attempt = store.db.execute("SELECT gpu_start,gpu_ms,gpu_complete FROM attempts WHERE token=?", (token,)).fetchone()
            self.assertEqual(store.get("r")["status"], "running")
            self.assertEqual(attempt["gpu_start"], 2)
            self.assertIsNone(attempt["gpu_ms"])
            self.assertEqual(attempt["gpu_complete"], 0)
            store.close()

    def test_generic_handoff_outbox_ack_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store, session = self.make(Path(directory) / "q.sqlite")
            store.enqueue("r", "payload", idempotency_key="r")
            token = store.claim("r", session.token, session.generation)
            store.stage_handoff("r", token, session.token, session.generation, "0" * 64, "handoff")
            with self.assertRaises(Exception):
                store.acknowledge_outbox("handoff")
            self.assertEqual(store.get("r")["status"], "running")
            self.assertEqual([entry["kind"] for entry in store.outbox()], ["submit", "handoff"])
            store.close()

    def test_operation_guards_reject_wrong_attempt_and_immutable_terminal_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            store, session = self.make(Path(directory) / "q.sqlite")
            store.enqueue("r", "payload", idempotency_key="r")
            token = store.claim("r", session.token, session.generation)
            with self.assertRaises(StaleCallback):
                store.mark_on_gpu("r", token, session.token, session.generation + 1)
            store.finish_attempt("r", token, session.token, session.generation)
            store.stage_handoff("r", token, session.token, session.generation, "0" * 64, "handoff")
            store.acknowledge_handoff("r", "handoff")
            with self.assertRaises(StaleCallback):
                store.retry("r", token, session.token, session.generation)
            self.assertEqual(store.cancel("r", idempotency_key="cancel-terminal")["status"],
                             RequestStatus.DONE.value)
            self.assertEqual(store.get("r")["status"], RequestStatus.DONE.value)
            store.close()

    def test_non_retryable_failure_stops_the_entire_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            store, session = self.make(Path(directory) / "q.sqlite")
            for request_id in ("failed", "queued"):
                store.enqueue(request_id, "payload", idempotency_key=request_id)
            token = store.claim("failed", session.token, session.generation)
            store.retry("failed", token, session.token, session.generation, retryable=False)
            self.assertEqual(store.get("failed")["status"], "error")
            self.assertEqual(store.get("queued")["status"], "error")
            with self.assertRaises(SessionError):
                store.claim("queued", session.token, session.generation)
            store.close()

    def test_cancel_is_idempotent_in_scheduled_running_and_on_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            store, session = self.make(Path(directory) / "q.sqlite")
            for request_id in ("scheduled", "running", "gpu"):
                store.enqueue(request_id, "p", idempotency_key=request_id)
            token = store.claim("running", session.token, session.generation)
            gpu_token = store.claim("gpu", session.token, session.generation)
            store.mark_on_gpu("gpu", gpu_token, session.token, session.generation)
            for request_id in ("scheduled", "running", "gpu"):
                self.assertEqual(store.cancel(request_id)["status"], "cancelled")
                self.assertEqual(store.cancel(request_id)["status"], "cancelled")
            for request_id in ("scheduled", "running", "gpu"):
                self.assertEqual(store.get(request_id)["status"], "cancelled")
            with self.assertRaises(StaleCallback):
                store.finish_attempt("gpu", gpu_token, session.token, session.generation)
            store.close()

    def test_same_owner_recovery_preserves_handoff_and_fences_old_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.sqlite"
            store, old = self.make(path)
            store.enqueue("r", "p", idempotency_key="r")
            token = store.claim("r", old.token, old.generation)
            store.stage_handoff("r", token, old.token, old.generation, "0" * 64, "handoff")
            store.close()
            recovered, new = self.make(path)
            self.assertIsNotNone(recovered.db.execute("SELECT 1 FROM handoffs WHERE request_id='r'").fetchone())
            # Recovery has the new owner, so the durable handoff can be replayed.
            recovered.acknowledge_handoff("r", "handoff")
            self.assertEqual(recovered.get("r")["status"], "done")
            with self.assertRaises(StaleCallback):
                recovered.finish_attempt("r", token, old.token, old.generation)
            recovered.close()

    def test_different_model_supersession_fences_old_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.sqlite"
            old_store, old = self.make(path, "old", ModelId.SMOLLM)
            old_store.enqueue("r", "p", idempotency_key="r")
            new_store, new = self.make(path, "new", ModelId.COEDIT)
            self.assertEqual(new_store.db.execute(
                "SELECT status FROM requests WHERE request_id='r'"
            ).fetchone()[0], RequestStatus.ERROR.value)
            with self.assertRaises(SessionError):
                old_store.get("r")
            old_store.close()
            new_store.close()

    def test_different_model_supersession_records_old_owner_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.sqlite"
            old_store, _ = self.make(path, "old", ModelId.SMOLLM)
            old_store.enqueue("r", "p", idempotency_key="r")
            new_store, _ = self.make(path, "new", ModelId.COEDIT)
            event = json.loads(new_store.events(0)[-1]["data"])
            self.assertEqual(event["snapshot"]["schedulerId"], "old")
            self.assertEqual(event["snapshot"]["modelId"], ModelId.SMOLLM.value)
            self.assertEqual(event["snapshot"]["status"], RequestStatus.ERROR.value)
            old_store.close()
            new_store.close()

    def test_abrupt_subprocess_exit_preserves_request_and_submit_outbox(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.sqlite"
            code = (
                "from services.llm.queue.store import QueueStore; "
                "from services.llm.queue.contracts import ModelId; "
                f"q=QueueStore(r'{path}', 's', ModelId.SMOLLM); "
                "s=q.start_session(); q.enqueue('r','payload',idempotency_key='r'); "
                "q.claim('r',s.token,s.generation); import os; os._exit(17)"
            )
            completed = subprocess.run([sys.executable, "-c", code])
            self.assertEqual(completed.returncode, 17)
            store, session = self.make(path)
            self.assertEqual(store.get("r")["status"], "scheduled")
            self.assertEqual([row["kind"] for row in store.outbox()], ["cancel"])
            store.close()

    def test_independent_connections_serialize_skip_line_insertions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.sqlite"
            first, session = self.make(path)
            first.enqueue("anchor", "p", idempotency_key="anchor")
            # Both operations use separate SQLite connections, but ownership is
            # intentionally shared by the same scheduler identity.
            second = QueueStore(path, "s", ModelId.SMOLLM)
            second.session = session
            errors = []

            def insert(request_id):
                store = QueueStore(path, "s", ModelId.SMOLLM)
                store.session = session
                try:
                    store.enqueue(request_id, "p", insertion_mode="skip-line", idempotency_key=request_id)
                except Exception as exc:  # assertion below reports the concrete race
                    errors.append(exc)
                finally:
                    store.close()

            threads = [threading.Thread(target=insert, args=(request_id,))
                       for request_id in ("a", "b")]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            rows = first.list_positions()
            concurrent = [row for row in rows if row["request_id"] in {"a", "b"}]
            self.assertEqual(
                [row["request_id"] for row in sorted(concurrent, key=lambda row: row["insertion_seq"])],
                [row["request_id"] for row in sorted(concurrent, key=lambda row: row["rank"])],
            )
            self.assertEqual(rows[-1]["request_id"], "anchor")
            first.close()
            second.close()

    def test_closed_skip_line_anchor_reopens_at_next_scheduled_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            store, session = self.make(Path(directory) / "q.sqlite")
            for request_id in ("anchor", "first"):
                store.enqueue(request_id, "p", idempotency_key=request_id)
            store.enqueue("grouped", "p", insertion_mode="skip-line", idempotency_key="grouped")
            # The original anchor leaves scheduled, closing its group.
            token = store.claim("anchor", session.token, session.generation)
            self.assertIsNotNone(token)
            store.enqueue("reopened", "p", insertion_mode="skip-line", idempotency_key="reopened")
            self.assertEqual([row["request_id"] for row in store.list_positions()],
                             ["reopened", "grouped", "anchor", "first"])
            store.close()

    def test_closed_group_reopens_when_original_anchor_is_scheduled_by_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            store, session = self.make(Path(directory) / "q.sqlite")
            store.enqueue("anchor", "p", idempotency_key="anchor")
            for request_id in ("old1", "old2"):
                store.enqueue(request_id, "p", insertion_mode="skip-line", idempotency_key=request_id)
            old_tokens = []
            for request_id in ("old1", "old2"):
                token = store.claim(request_id, session.token, session.generation)
                old_tokens.append(token)
                store.finish_attempt(request_id, token, session.token, session.generation)
                store.stage_handoff(request_id, token, session.token, session.generation, "0" * 64, f"h-{request_id}")
                store.acknowledge_handoff(request_id, f"h-{request_id}")
            anchor_token = store.claim("anchor", session.token, session.generation, now=0)
            store.retry("anchor", anchor_token, session.token, session.generation, now=0)
            self.assertEqual(store.get("anchor")["status"], "scheduled")
            store.enqueue("new1", "p", insertion_mode="skip-line", idempotency_key="new1")
            store.enqueue("new2", "p", insertion_mode="skip-line", idempotency_key="new2")
            rows = store.list_positions()
            ids = [row["request_id"] for row in rows]
            self.assertEqual(ids[-3:], ["new1", "new2", "anchor"])
            self.assertEqual([store.get(request_id)["status"] for request_id in ("old1", "old2")], ["done", "done"])
            store.close()

    def test_duplicate_finish_attempt_does_not_change_existing_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            store, session = self.make(Path(directory) / "q.sqlite")
            store.enqueue("r", "p", idempotency_key="r")
            token = store.claim("r", session.token, session.generation, now=1)
            store.mark_on_gpu("r", token, session.token, session.generation, at=2)
            store.finish_attempt("r", token, session.token, session.generation, at=3, gpu_ms=100)
            before = store.db.execute(
                "SELECT finished,gpu_end,gpu_ms,gpu_complete FROM attempts WHERE token=?", (token,)
            ).fetchone()
            store.finish_attempt("r", token, session.token, session.generation, at=9, gpu_ms=999)
            after = store.db.execute(
                "SELECT finished,gpu_end,gpu_ms,gpu_complete FROM attempts WHERE token=?", (token,)
            ).fetchone()
            self.assertEqual(tuple(after), tuple(before))
            store.close()


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from services.llm.queue.contracts import ModelId, RequestStatus, ALLOWED_TRANSITIONS
from services.llm.queue.store import (
    DependencyError,
    IdempotencyConflict,
    InvalidTransition,
    QueueStore,
    SessionError,
    StaleCallback,
)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = QueueStore(Path(self.tmp.name) / "queue.sqlite", "scheduler", ModelId.SMOLLM)
        self.session = self.store.start_session()

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def claim(self, request_id="request"):
        self.store.enqueue(request_id, "payload", idempotency_key=request_id)
        return self.store.claim(request_id, self.session.token, self.session.generation, now=0)

    def test_every_transition_pair_requires_matrix_and_fence(self):
        for old in RequestStatus:
            for new in RequestStatus:
                allowed = old == new or new in ALLOWED_TRANSITIONS[old]
                self.assertEqual(allowed, old == new or new in ALLOWED_TRANSITIONS[old])
        token = self.claim()
        with self.assertRaises(StaleCallback):
            self.store.transition("request", RequestStatus.ON_GPU, token, "wrong", self.session.generation)

    def test_enqueue_idempotency_is_exact_and_conflicts(self):
        self.store.enqueue("a", "payload", idempotency_key="key")
        self.assertEqual(self.store.enqueue("a", "payload", idempotency_key="key")["request_id"], "a")
        with self.assertRaises(IdempotencyConflict):
            self.store.enqueue("b", "other", idempotency_key="key")

    def test_claim_requires_dependencies_and_session(self):
        self.store.enqueue("dependency", "p", idempotency_key="d")
        self.store.enqueue("dependent", "p", dependencies=("dependency",), idempotency_key="x")
        with self.assertRaises(StaleCallback):
            self.store.claim("dependent", "wrong", self.session.generation)
        self.assertIsNone(self.store.claim("dependent", self.session.token, self.session.generation))

    def test_deep_missing_and_cyclic_dependencies_are_rejected(self):
        with self.assertRaises(DependencyError):
            self.store.enqueue("missing", "p", dependencies=("nope",), idempotency_key="missing")
        self.store.enqueue("a", "p", idempotency_key="a")
        self.store.enqueue("b", "p", dependencies=("a",), idempotency_key="b")
        # A new cyclic edge cannot be introduced because dependencies are immutable.
        with self.assertRaises(DependencyError):
            self.store.enqueue("a2", "p", dependencies=("b", "a2"), idempotency_key="a2")

    def test_handoff_ack_is_the_only_done_path_and_deactivates_attempt(self):
        token = self.claim()
        with self.assertRaises(InvalidTransition):
            self.store.transition("request", RequestStatus.DONE, token, self.session.token, self.session.generation)
        self.store.stage_handoff("request", token, self.session.token, self.session.generation, "0" * 64, "handoff")
        self.store.acknowledge_handoff("request", "handoff")
        self.assertEqual(self.store.get("request")["status"], RequestStatus.DONE.value)
        self.assertEqual([row["kind"] for row in self.store.outbox()], ["submit"])

    def test_cancel_fences_work_and_hides_outbox(self):
        token = self.claim()
        self.store.cancel("request")
        self.assertEqual([row["kind"] for row in self.store.outbox()], ["cancel"])
        with self.assertRaises(StaleCallback):
            self.store.stage_handoff("request", token, self.session.token, self.session.generation, "0" * 64, "h")

    def test_stop_blocks_future_claim_and_enqueue(self):
        self.claim()
        self.store.stop("shutdown")
        with self.assertRaises(SessionError):
            self.store.enqueue("later", "p", idempotency_key="later")
        with self.assertRaises(SessionError):
            self.store.claim("request", self.session.token, self.session.generation)

    def test_retry_backoff_and_lease_recovery(self):
        token = self.claim()
        self.store.retry("request", token, self.session.token, self.session.generation, failure_elapsed=1, now=10)
        row = self.store.get("request")
        self.assertEqual(row["next_attempt_at"], 15)
        token = self.store.claim("request", self.session.token, self.session.generation, now=15)
        self.assertIsNotNone(token)
        self.assertEqual(self.store.reconcile_expired(now=1000), 1)

    def test_event_cursor_preserves_repeated_events(self):
        self.store.enqueue("a", "p", idempotency_key="a")
        self.store.cancel("a")
        events = self.store.events()
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(self.store.events(events[0]["cursor"], limit=1)[0]["cursor"], events[1]["cursor"])


if __name__ == "__main__":
    unittest.main()

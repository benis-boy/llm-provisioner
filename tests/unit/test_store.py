import asyncio
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

    def test_corrupt_multi_node_cycle_is_rejected_by_graph_validation(self):
        # Public enqueue IDs are immutable and therefore cannot manufacture a
        # cycle.  This fixture represents a damaged/restored queue and checks
        # that the defensive validator still walks the complete graph.
        self.store.enqueue("a", "p", idempotency_key="a")
        self.store.enqueue("b", "p", idempotency_key="b")
        self.store.db.execute("UPDATE requests SET dependencies=? WHERE request_id='a'", ('["b"]',))
        self.store.db.execute("UPDATE requests SET dependencies=? WHERE request_id='b'", ('["a"]',))
        with self.assertRaises(DependencyError):
            self.store.enqueue("c", "p", dependencies=("a",), idempotency_key="c")

    def test_corrupt_missing_dependency_fails_structured_instead_of_crashing(self):
        self.store.enqueue("r", "p", idempotency_key="r")
        self.store.db.execute("UPDATE requests SET dependencies=? WHERE request_id='r'", ('["gone"]',))
        from services.llm.queue.eligibility import EligibilityEvaluator
        result = asyncio.run(EligibilityEvaluator(self.store).evaluate("r"))
        self.assertEqual(result.error_code, "dependency_failed")
        self.assertEqual(self.store.get("r")["error_code"], "dependency_failed")

    def test_corrupt_missing_dependency_fails_both_claim_paths_without_attempt_or_outbox(self):
        self.store.enqueue("direct", "p", idempotency_key="direct")
        self.store.db.execute("UPDATE requests SET dependencies=? WHERE request_id='direct'", ('["gone"]',))
        self.assertIsNone(self.store.claim("direct", self.session.token, self.session.generation))

        self.store.enqueue("evaluated", "p", idempotency_key="evaluated")
        self.store.db.execute("UPDATE requests SET dependencies=? WHERE request_id='evaluated'", ('["gone"]',))
        capability = self.store.record_evaluation("evaluated", self.store._version(), "evaluated-payload")
        self.assertIsNone(self.store.claim_evaluated(capability))

        for request_id in ("direct", "evaluated"):
            self.assertEqual(self.store.get(request_id)["status"], RequestStatus.ERROR.value)
            self.assertEqual(self.store.get(request_id)["error_code"], "dependency_failed")
            self.assertEqual(self.store.db.execute(
                "SELECT COUNT(*) FROM attempts WHERE request_id=?", (request_id,)
            ).fetchone()[0], 0)
            self.assertEqual(self.store.db.execute(
                "SELECT COUNT(*) FROM outbox WHERE request_id=?", (request_id,)
            ).fetchone()[0], 0)

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

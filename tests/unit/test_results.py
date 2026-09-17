import tempfile
import unittest
from pathlib import Path

from services.llm.queue.results import LocalPublisher, ResultError, ResultStore
from services.llm.queue.store import QueueStore
from services.llm.queue.contracts import ModelId


class ResultTests(unittest.TestCase):
    def test_content_addressing_verification_and_publisher_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            state = Path(directory) / "publisher.sqlite"
            results = ResultStore(root)
            reference = results.write(b"result")
            publisher = LocalPublisher(results, state)
            self.assertTrue(publisher.publish("request", "attempt", reference, "key"))
            publisher.close()
            publisher = LocalPublisher(results, state)
            self.assertFalse(publisher.publish("request", "attempt", reference, "key"))
            other_reference = results.write(b"other")
            with self.assertRaises(ResultError):
                publisher.publish("request", "attempt", other_reference, "key")
            with self.assertRaises(ResultError):
                results.read("../publisher.sqlite")
            (root / reference).write_bytes(b"corrupt")
            with self.assertRaises(ResultError):
                results.verify(reference)
            publisher.close()

    def test_stage_result_writes_content_before_durable_handoff_and_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            results = ResultStore(root)
            queue = QueueStore(Path(directory) / "queue.sqlite", "s", ModelId.SMOLLM)
            session = queue.start_session()
            queue.enqueue("request", "payload", idempotency_key="request")
            token = queue.claim("request", session.token, session.generation)
            handoff = queue.stage_result("request", token, session.token, session.generation,
                                         results, b"result", "handoff")
            publisher = LocalPublisher(results, Path(directory) / "publisher.sqlite")
            self.assertTrue(publisher.publish("request", token, handoff["result_reference"], "handoff"))
            queue.acknowledge_handoff("request", "handoff")
            self.assertEqual(queue.get("request")["status"], "done")
            publisher.close()
            queue.close()

    def test_connection_scoped_publish_joins_queue_transaction_and_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            state = Path(directory) / "shared.sqlite"
            results = ResultStore(root)
            reference = results.write(b"result")
            publisher = LocalPublisher(results, state)
            queue = QueueStore(state, "s", ModelId.SMOLLM)
            queue.db.execute("BEGIN IMMEDIATE")
            self.assertTrue(publisher.publish_on_connection(queue.db, reference, "key"))
            queue.db.execute("COMMIT")
            queue.db.execute("BEGIN IMMEDIATE")
            self.assertFalse(publisher.publish_on_connection(queue.db, reference, "key"))
            queue.db.execute("COMMIT")
            publisher.close()
            queue.close()


if __name__ == "__main__":
    unittest.main()

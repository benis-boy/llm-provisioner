import tempfile
import unittest
from pathlib import Path
import hashlib
import json
import sqlite3

from services.llm.queue.contracts import FunctionDescriptor, ModelId
from services.llm.queue.store import DependencyError, DuplicateRequest, QueueStore, SessionError, StaleCallback


class FunctionIntentTests(unittest.TestCase):
    def test_ready_and_template_round_trip_across_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.sqlite"
            ready = FunctionDescriptor("prepare", {"value": 1}, ("dependency",))
            template = FunctionDescriptor("render", {"format": "json"}, ("dependency",))
            with QueueStore(path, "scheduler", ModelId.SMOLLM) as queue:
                queue.start_session()
                queue.enqueue("dependency", "payload", idempotency_key="dependency")
                queue.enqueue("request", "payload", ("dependency",), idempotency_key="request",
                              ready=ready, template=template)
                row = queue.get("request")
                self.assertEqual(queue.descriptor(row, "ready"), ready)
                self.assertEqual(queue.descriptor(row, "template"), template)
            with QueueStore(path, "scheduler", ModelId.SMOLLM) as queue:
                queue.recover_session()
                row = queue.get("request")
                self.assertEqual(queue.descriptor(row, "ready"), ready)
                self.assertEqual(queue.descriptor(row, "template"), template)

    def test_descriptor_claim_is_fail_closed_without_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            with QueueStore(Path(directory) / "queue.sqlite", "scheduler", ModelId.SMOLLM) as queue:
                session = queue.start_session()
                queue.enqueue("dependency", "payload", idempotency_key="dependency")
                queue.enqueue("function", "payload", ("dependency",), idempotency_key="function",
                              ready=FunctionDescriptor("prepare", {}, ("dependency",)))
                queue.enqueue("plain", "payload", idempotency_key="plain")
                dependency_token = queue.claim("dependency", session.token, session.generation)
                queue.stage_handoff("dependency", dependency_token, session.token, session.generation,
                                    "0" * 64, "dependency-handoff")
                queue.acknowledge_handoff("dependency", "dependency-handoff")
                event_count = len(queue.events())
                self.assertIsNone(queue.claim("function", session.token, session.generation))
                self.assertEqual(queue.get("function")["status"], "scheduled")
                self.assertEqual(len(queue.events()), event_count)
                self.assertEqual([row["kind"] for row in queue.outbox()], ["submit"])
                self.assertIsNotNone(queue.claim("plain", session.token, session.generation))

    def test_ready_template_and_both_are_fail_closed_without_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            with QueueStore(Path(directory) / "queue.sqlite", "scheduler", ModelId.SMOLLM) as queue:
                session = queue.start_session()
                descriptor = FunctionDescriptor("prepare", {})
                for request_id, kwargs in (
                    ("ready", {"ready": descriptor}),
                    ("template", {"template": descriptor}),
                    ("both", {"ready": descriptor, "template": descriptor}),
                ):
                    queue.enqueue(request_id, "payload", idempotency_key=request_id, **kwargs)
                before = len(queue.events())
                for request_id in ("ready", "template", "both"):
                    self.assertIsNone(queue.claim(request_id, session.token, session.generation))
                self.assertEqual(len(queue.events()), before)
                self.assertEqual(queue.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 0)
                self.assertEqual(queue.outbox(), [])

    def test_canonical_descriptor_replay_is_noop_but_changes_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            with QueueStore(Path(directory) / "queue.sqlite", "scheduler", ModelId.SMOLLM) as queue:
                queue.start_session()
                descriptor = FunctionDescriptor("prepare", {"b": 2, "a": 1}, ("dep",))
                queue.enqueue("dep", "payload", idempotency_key="dep")
                queue.enqueue("other-dep", "payload", idempotency_key="other-dep")
                original = queue.enqueue("request", "payload", ("dep", "other-dep"), idempotency_key="key", ready=descriptor)
                replay = queue.enqueue(
                    "request", "payload", ("dep", "other-dep"), idempotency_key="key",
                    ready=FunctionDescriptor("prepare", {"a": 1, "b": 2}, ("dep",)),
                )
                self.assertEqual(original["fingerprint"], replay["fingerprint"])
                for changed in (
                    FunctionDescriptor("other", {"a": 1, "b": 2}, ("dep",)),
                    FunctionDescriptor("prepare", {"a": 9, "b": 2}, ("dep",)),
                    FunctionDescriptor("prepare", {"a": 1, "b": 2}, ("other-dep",)),
                ):
                    with self.assertRaises(DuplicateRequest):
                        queue.enqueue("request", "payload", ("dep", "other-dep"), idempotency_key="key", ready=changed)

    def test_descriptor_idempotency_addition_removal_and_template_changes_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            with QueueStore(Path(directory) / "queue.sqlite", "scheduler", ModelId.SMOLLM) as queue:
                queue.start_session()
                queue.enqueue("dep", "payload", idempotency_key="dep")
                queue.enqueue("other", "payload", idempotency_key="other")
                ready = FunctionDescriptor("ready", {}, ("dep",))
                template = FunctionDescriptor("template", {}, ("other",))
                queue.enqueue("plain", "payload", ("dep", "other"), idempotency_key="plain",
                              ready=ready, template=template)
                for kwargs in (
                    {"ready": None, "template": template},
                    {"ready": ready, "template": None},
                    {"ready": FunctionDescriptor("changed", {}, ("dep",)), "template": template},
                    {"ready": ready, "template": FunctionDescriptor("changed", {}, ("other",))},
                ):
                    with self.assertRaises(DuplicateRequest):
                        queue.enqueue("plain", "payload", ("dep", "other"), idempotency_key="plain", **kwargs)

    def test_post_accept_mutation_and_invalid_descriptor_cannot_change_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            with QueueStore(Path(directory) / "queue.sqlite", "scheduler", ModelId.SMOLLM) as queue:
                queue.start_session()
                args = {"nested": {"value": 1}}
                descriptor = FunctionDescriptor("prepare", args)
                queue.enqueue("request", "payload", idempotency_key="request", ready=descriptor)
                args["nested"]["value"] = 7
                descriptor.args["nested"]["value"] = 8
                self.assertEqual(queue.descriptor(queue.get("request"), "ready").args["nested"]["value"], 1)
                args["nested"]["value"] = float("nan")
                with self.assertRaises(ValueError):
                    queue.enqueue("nan", "payload", idempotency_key="nan", ready=descriptor)
                self.assertIsNone(queue.get("nan"))
                with self.assertRaises(ValueError):
                    queue.enqueue("invalid", "payload", idempotency_key="invalid", ready="not-a-descriptor")
                self.assertIsNone(queue.get("invalid"))

    def test_descriptor_cancel_stop_and_stale_session_are_fenced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.sqlite"
            with QueueStore(path, "scheduler", ModelId.SMOLLM) as queue:
                old = queue.start_session()
                queue.enqueue("cancelled", "payload", idempotency_key="cancelled",
                              ready=FunctionDescriptor("ready", {}))
                queue.cancel("cancelled")
                self.assertEqual(queue.get("cancelled")["status"], "cancelled")
                self.assertEqual(queue.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 0)
                queue.enqueue("stopped", "payload", idempotency_key="stopped",
                              template=FunctionDescriptor("template", {}))
                queue.stop("test-stop")
            with QueueStore(path, "scheduler", ModelId.SMOLLM) as recovered:
                new = recovered.recover_session()
                with self.assertRaises(StaleCallback):
                    recovered.claim("stopped", old.token, old.generation)
                self.assertEqual(recovered.get("stopped")["status"], "error")
                self.assertNotEqual(new.token, old.token)

    def test_legacy_schema_upgrade_preserves_rows_and_old_fingerprint_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite"
            db = sqlite3.connect(path)
            db.executescript("""
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE requests (
                    request_id TEXT PRIMARY KEY, scheduler_id TEXT NOT NULL, model_id TEXT NOT NULL,
                    payload_reference TEXT NOT NULL, dependencies TEXT NOT NULL, insertion_mode TEXT NOT NULL,
                    result_target TEXT NOT NULL, fingerprint TEXT NOT NULL, idempotency_key TEXT,
                    status TEXT NOT NULL, cancellation INTEGER NOT NULL DEFAULT 0, running_at REAL,
                    done_at REAL, next_attempt_at REAL, retry_elapsed REAL NOT NULL DEFAULT 0,
                    first_retry_at REAL, retry_count INTEGER NOT NULL DEFAULT 0, error_code TEXT);
                CREATE TABLE positions (request_id TEXT PRIMARY KEY, rank INTEGER UNIQUE, insertion_seq INTEGER UNIQUE NOT NULL,
                    mode TEXT NOT NULL, anchor TEXT, group_tail TEXT);
                CREATE TABLE attempts (request_id TEXT NOT NULL, token TEXT PRIMARY KEY, session TEXT NOT NULL,
                    generation INTEGER NOT NULL, provider_id TEXT, started REAL, finished REAL, lease_until REAL,
                    gpu_start REAL, gpu_end REAL, gpu_ms INTEGER, gpu_complete INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE handoffs (request_id TEXT PRIMARY KEY, token TEXT NOT NULL, idempotency_key TEXT UNIQUE NOT NULL,
                    result_reference TEXT NOT NULL, acknowledged INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE outbox (operation_id TEXT PRIMARY KEY, kind TEXT NOT NULL, idempotency_key TEXT UNIQUE NOT NULL,
                    request_id TEXT, token TEXT, payload_reference TEXT NOT NULL, acknowledged INTEGER NOT NULL DEFAULT 0,
                    cancelled INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL);
                CREATE TABLE events (cursor INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, kind TEXT NOT NULL,
                    data TEXT NOT NULL, created REAL NOT NULL);
            """)
            old_fp = hashlib.sha256(json.dumps(
                ["legacy", "payload", (), "append", "local"], separators=(",", ":")
            ).encode()).hexdigest()
            db.executemany("INSERT INTO meta VALUES (?,?)", [("model_id", "SmolLM"), ("scheduler_id", "scheduler"),
                                                               ("session_token", "old-session"), ("generation", "1"), ("accepting", "1")])
            db.execute("INSERT INTO requests(request_id,scheduler_id,model_id,payload_reference,dependencies,insertion_mode,result_target,fingerprint,idempotency_key,status) VALUES(?,?,?,?,?,?,?,?,?,?)",
                       ("legacy", "scheduler", "SmolLM", "payload", "[]", "append", "local", old_fp, "legacy-key", "running"))
            db.execute("INSERT INTO positions VALUES (?,?,?,?,?,?)", ("legacy", 0, 1, "append", None, "legacy"))
            db.execute("INSERT INTO attempts(request_id,token,session,generation,started,active) VALUES(?,?,?,?,?,?)",
                       ("legacy", "old-token", "old-session", 1, 1.0, 1))
            db.execute("INSERT INTO handoffs VALUES (?,?,?,?,?)", ("legacy", "old-token", "handoff", "0" * 64, 0))
            db.execute("INSERT INTO outbox VALUES (?,?,?,?,?,?,?,?,?)", ("op", "handoff", "handoff", "legacy", "old-token", "0" * 64, 0, 0, 1.0))
            db.execute("INSERT INTO events VALUES (?,?,?,?,?)", (1, "legacy", "claim", "{}", 1.0))
            before = {table: db.execute(f"SELECT * FROM {table}").fetchall() for table in ("requests", "positions", "attempts", "handoffs", "outbox", "events")}
            db.commit(); db.close()
            with QueueStore(path, "scheduler", ModelId.SMOLLM) as queue:
                after_open = {table: queue.db.execute(f"SELECT * FROM {table}").fetchall() for table in before}
                for table, rows in before.items():
                    # Schema opening may append descriptor columns, but must not
                    # rewrite any pre-Slice-E value or historical row.
                    old_width = len(tuple(rows[0]))
                    self.assertEqual([tuple(row)[:old_width] for row in after_open[table]],
                                     [tuple(row) for row in rows])
                queue.start_session()
                self.assertEqual(queue.enqueue("legacy", "payload", idempotency_key="legacy-key")["fingerprint"], old_fp)

    def test_snapshot_and_dependency_boundary_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            with QueueStore(Path(directory) / "queue.sqlite", "scheduler", ModelId.SMOLLM) as queue:
                queue.start_session()
                args = {"nested": {"value": 1}}
                descriptor = FunctionDescriptor("prepare", args)
                queue.enqueue("request", "payload", idempotency_key="request", ready=descriptor)
                args["nested"]["value"] = 99
                self.assertEqual(queue.get("request")["ready"],
                                 '{"args":{"nested":{"value":1}},"dependency_result_ids":[],"name":"prepare"}')
                with self.assertRaises(DependencyError):
                    queue.enqueue("bad", "payload", idempotency_key="bad",
                                  ready=FunctionDescriptor("prepare", {}, ("missing",)))


if __name__ == "__main__":
    unittest.main()

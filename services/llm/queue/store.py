"""Durable queue primitives.

The store is deliberately synchronous.  The session and attempt values are
capabilities: every mutation which could be caused by a delayed callback must
present both of them (and the current owner) to SQLite in one transaction.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import Iterable

from .contracts import (
    FunctionDescriptor, InsertionMode, ModelId, RequestStatus, TERMINAL, can_transition,
    deserialize_function_descriptor, serialize_function_descriptor,
)


class QueueError(Exception):
    """Base class for durable queue errors."""


class DuplicateRequest(QueueError):
    pass


class IdempotencyConflict(QueueError):
    pass


class StaleCallback(QueueError):
    pass


class InvalidTransition(QueueError):
    pass


class DependencyError(QueueError):
    pass


class SessionError(QueueError):
    pass


class OperationStale(QueueError):
    """An operation key names a lifecycle from an older local generation."""
    pass


@dataclass(frozen=True)
class Session:
    scheduler_id: str
    model_id: ModelId
    token: str
    generation: int


@dataclass(frozen=True)
class EvaluatedClaim:
    request_id: str
    version: int
    session_token: str
    generation: int
    payload_reference: str
    intent_fingerprint: str
    nonce: str


def _fingerprint(payload_reference: str, dependencies: tuple[str, ...],
                   insertion_mode: InsertionMode, result_target: str,
                   request_id: str, ready: str | None = None,
                   template: str | None = None) -> str:
    # Keep the original pre-Slice-E identity byte-for-byte when both optional
    # intents are absent, so replaying an old enqueue remains compatible.
    values = [request_id, payload_reference, dependencies, insertion_mode.value, result_target]
    if ready is not None or template is not None:
        values.extend([ready, template])
    value = json.dumps(
        values,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode()).hexdigest()


class QueueStore:
    def __init__(self, path: str | os.PathLike[str], scheduler_id: str,
                 model_id: ModelId):
        self.path = str(path)
        self.scheduler_id = scheduler_id
        self.model_id = ModelId(model_id)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=30000")
        self._schema()
        self.session: Session | None = None

    def _schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY, scheduler_id TEXT NOT NULL,
                model_id TEXT NOT NULL, payload_reference TEXT NOT NULL,
                dependencies TEXT NOT NULL, insertion_mode TEXT NOT NULL,
                result_target TEXT NOT NULL, fingerprint TEXT NOT NULL,
                ready TEXT, template TEXT,
                idempotency_key TEXT, status TEXT NOT NULL,
                cancellation INTEGER NOT NULL DEFAULT 0,
                running_at REAL, done_at REAL, next_attempt_at REAL,
                retry_elapsed REAL NOT NULL DEFAULT 0,
                first_retry_at REAL, retry_count INTEGER NOT NULL DEFAULT 0,
                error_code TEXT, evaluated_payload_reference TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS request_idempotency
                ON requests(scheduler_id, idempotency_key)
                WHERE idempotency_key IS NOT NULL;
            CREATE TABLE IF NOT EXISTS positions (
                request_id TEXT PRIMARY KEY REFERENCES requests ON DELETE CASCADE,
                rank INTEGER UNIQUE, insertion_seq INTEGER UNIQUE NOT NULL,
                mode TEXT NOT NULL, anchor TEXT, group_tail TEXT
            );
            CREATE TABLE IF NOT EXISTS skip_groups (
                anchor TEXT PRIMARY KEY, tail TEXT NOT NULL, sequence INTEGER NOT NULL,
                closed INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS attempts (
                request_id TEXT NOT NULL REFERENCES requests ON DELETE CASCADE,
                token TEXT PRIMARY KEY, session TEXT NOT NULL,
                generation INTEGER NOT NULL, provider_id TEXT,
                started REAL, finished REAL, lease_until REAL,
                gpu_start REAL, gpu_end REAL, gpu_ms INTEGER,
                gpu_complete INTEGER NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1, payload_reference TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_attempt
                ON attempts(request_id) WHERE active=1;
            CREATE TABLE IF NOT EXISTS handoffs (
                request_id TEXT PRIMARY KEY REFERENCES requests ON DELETE CASCADE,
                token TEXT NOT NULL, idempotency_key TEXT UNIQUE NOT NULL,
                result_reference TEXT NOT NULL, acknowledged INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS outbox (
                operation_id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                idempotency_key TEXT UNIQUE NOT NULL, request_id TEXT,
                token TEXT, payload_reference TEXT NOT NULL,
                acknowledged INTEGER NOT NULL DEFAULT 0,
                cancelled INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                cursor INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT,
                kind TEXT NOT NULL, data TEXT NOT NULL, created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evaluated_capabilities (
                nonce TEXT PRIMARY KEY, request_id TEXT NOT NULL, version INTEGER NOT NULL,
                session_token TEXT NOT NULL, generation INTEGER NOT NULL,
                payload_reference TEXT NOT NULL, intent_fingerprint TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dispatch_metadata (
                token TEXT PRIMARY KEY REFERENCES attempts(token) ON DELETE CASCADE,
                payload_digest TEXT NOT NULL, context_size INTEGER,
                bucket_identity TEXT, created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scheduler_operations (
                scheduler_id TEXT NOT NULL, operation TEXT NOT NULL,
                idempotency_key TEXT NOT NULL, fingerprint TEXT NOT NULL,
                target_generation INTEGER, outcome TEXT NOT NULL,
                created REAL NOT NULL,
                PRIMARY KEY(scheduler_id, operation, idempotency_key)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS scheduler_operation_keys
                ON scheduler_operations(scheduler_id, idempotency_key);
            """
        )
        # This is intentionally an additive, transactional upgrade rather than
        # a recreate: positions, attempts, events, and outbox rows are sacred.
        self.db.execute("BEGIN IMMEDIATE")
        try:
            columns = {row[1] for row in self.db.execute("PRAGMA table_info(requests)")}
            for name in ("ready", "template"):
                if name not in columns:
                    self.db.execute(f"ALTER TABLE requests ADD COLUMN {name} TEXT")
            columns = {row[1] for row in self.db.execute("PRAGMA table_info(requests)")}
            if "evaluated_payload_reference" not in columns:
                self.db.execute("ALTER TABLE requests ADD COLUMN evaluated_payload_reference TEXT")
            attempt_columns = {row[1] for row in self.db.execute("PRAGMA table_info(attempts)")}
            if "payload_reference" not in attempt_columns:
                self.db.execute("ALTER TABLE attempts ADD COLUMN payload_reference TEXT")
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        model = self.db.execute("SELECT value FROM meta WHERE key='model_id'").fetchone()
        if not model:
            self.db.execute(
                "INSERT INTO meta(key,value) VALUES('model_id',?)",
                (self.model_id.value,),
            )
        self.db.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('eligibility_version','0')")

    def _version(self) -> int:
        row = self.db.execute("SELECT value FROM meta WHERE key='eligibility_version'").fetchone()
        return int(row[0]) if row else 0

    def _bump_version(self) -> int:
        value = self._version() + 1
        self.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('eligibility_version',?)", (str(value),))
        # Snapshot capabilities cannot survive a durable queue mutation.
        self.db.execute("DELETE FROM evaluated_capabilities")
        return value

    def eligibility_version(self) -> int:
        self._owner()
        return self._version()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "QueueStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _begin(self) -> None:
        self.db.execute("BEGIN IMMEDIATE")

    def _owner(self) -> None:
        if self.session is None:
            raise SessionError("start_session or recover_session is required")
        row = self.db.execute(
            "SELECT value FROM meta WHERE key='scheduler_id'"
        ).fetchone()
        active = self.db.execute(
            "SELECT value FROM meta WHERE key='session_token'"
        ).fetchone()
        generation = self.db.execute(
            "SELECT value FROM meta WHERE key='generation'"
        ).fetchone()
        if (not row or row[0] != self.scheduler_id or not active
                or active[0] != self.session.token or not generation
                or int(generation[0]) != self.session.generation):
            raise SessionError("store session is no longer the owner")

    def start_session(self, idempotency_key: str | None = None) -> Session:
        """Atomically acquire ownership, superseding any prior scheduler."""
        if idempotency_key is not None and (not isinstance(idempotency_key, str) or not idempotency_key):
            raise ValueError("idempotency_key must be non-empty text")
        if self.session is not None:
            try:
                self._owner()
                if idempotency_key is None:
                    return self.session
            except SessionError:
                self.session = None
        self._begin()
        try:
            if idempotency_key is not None:
                conflict = self.db.execute("SELECT operation FROM scheduler_operations WHERE scheduler_id=? AND idempotency_key=?",
                                           (self.scheduler_id, idempotency_key)).fetchone()
                if conflict and conflict[0] != "start":
                    raise IdempotencyConflict(idempotency_key)
                old = self.db.execute(
                    "SELECT * FROM scheduler_operations WHERE scheduler_id=? AND operation='start' AND idempotency_key=?",
                    (self.scheduler_id, idempotency_key)).fetchone()
                fingerprint = self.model_id.value
                if old:
                    if old["fingerprint"] != fingerprint:
                        raise IdempotencyConflict(idempotency_key)
                    accepting = self.db.execute("SELECT value FROM meta WHERE key='accepting'").fetchone()
                    if (self.session is None or (old["target_generation"] is not None and old["target_generation"] != self.session.generation)
                            or not accepting or accepting[0] != "1"):
                        raise OperationStale(idempotency_key)
                    self.db.execute("COMMIT")
                    return self.session
                self.db.execute(
                    "INSERT INTO scheduler_operations VALUES(?,?,?,?,?,?,?)",
                    (self.scheduler_id, "start", idempotency_key, fingerprint, None, "pending", time.time()))
            owner = self.db.execute(
                "SELECT value FROM meta WHERE key='scheduler_id'"
            ).fetchone()
            old_model = self.db.execute(
                "SELECT value FROM meta WHERE key='model_id'"
            ).fetchone()
            if (owner and owner[0] == self.scheduler_id and old_model
                    and old_model[0] != self.model_id.value):
                raise SessionError("model mismatch; owner was not changed")
            generation = int(self.db.execute(
                "SELECT value FROM meta WHERE key='generation'"
            ).fetchone()[0]) + 1 if self.db.execute(
                "SELECT value FROM meta WHERE key='generation'"
            ).fetchone() else 1
            token = secrets.token_urlsafe(24)
            if owner and owner[0] != self.scheduler_id:
                self._terminalize_all("scheduler_superseded")
                self.db.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('model_id',?)",
                    (self.model_id.value,),
                )
            elif owner:
                # A replacement RM session fences provider execution immediately.
                # Publication is already durable work, so its handoff is retained.
                self._recover_prior_attempts()
            self.db.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('scheduler_id',?)",
                (self.scheduler_id,),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('session_token',?)",
                (token,),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('generation',?)",
                (str(generation),),
            )
            self.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('accepting','1')")
            # A generation/token replacement invalidates evaluations held by
            # other store handles even when recovery had no active attempts.
            self._bump_version()
            if idempotency_key is not None:
                self.db.execute(
                    "UPDATE scheduler_operations SET target_generation=?,outcome=? WHERE scheduler_id=? AND operation='start' AND idempotency_key=?",
                    (generation, json.dumps({"generation": generation, "token": token}), self.scheduler_id, idempotency_key))
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        self.session = Session(self.scheduler_id, self.model_id, token, generation)
        return self.session

    def complete_start(self, idempotency_key: str, session: Session) -> None:
        """Commit the RM acknowledgement for a keyed local start."""
        self._begin()
        try:
            self._owner()
            row = self.db.execute(
                "SELECT * FROM scheduler_operations WHERE scheduler_id=? AND operation='start' AND idempotency_key=?",
                (self.scheduler_id, idempotency_key)).fetchone()
            accepting = self.db.execute("SELECT value FROM meta WHERE key='accepting'").fetchone()
            current = self.db.execute("SELECT value FROM meta WHERE key='session_token'").fetchone()
            generation = self.db.execute("SELECT value FROM meta WHERE key='generation'").fetchone()
            owner = self.db.execute("SELECT value FROM meta WHERE key='scheduler_id'").fetchone()
            model = self.db.execute("SELECT value FROM meta WHERE key='model_id'").fetchone()
            if (not row or row["target_generation"] != session.generation
                    or session.scheduler_id != self.scheduler_id or session.model_id != self.model_id
                    or not accepting or accepting[0] != "1" or not current or current[0] != session.token
                    or not generation or int(generation[0]) != session.generation
                    or not owner or owner[0] != session.scheduler_id
                    or not model or model[0] != session.model_id.value):
                raise OperationStale(idempotency_key)
            self.db.execute("UPDATE scheduler_operations SET outcome=? WHERE scheduler_id=? AND operation='start' AND idempotency_key=?",
                             (json.dumps({"generation": session.generation, "token": session.token}), self.scheduler_id, idempotency_key))
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def recover_session(self) -> Session:
        """Acquire a new fenced session for the same durable scheduler."""
        owner = self.db.execute(
            "SELECT value FROM meta WHERE key='scheduler_id'"
        ).fetchone()
        if owner and owner[0] != self.scheduler_id:
            return self.start_session()
        # Recovery deliberately differs from idempotent start_session(): it
        # must replace even this handle's currently valid local fence.
        self.session = None
        return self.start_session()

    def _event(self, request_id: str | None, kind: str, data: dict) -> None:
        self._bump_version()
        # Events are the durable history, not a notification hint.  Insert first
        # to obtain the real global cursor, then attach the projection in the
        # same transaction so replay cannot stamp an old event with a newer row.
        self.db.execute(
            "INSERT INTO events(request_id,kind,data,created) VALUES(?,?,?,?)",
            (request_id, kind, json.dumps(data, sort_keys=True), time.time()),
        )
        if request_id is not None:
            cursor = self.db.execute("SELECT last_insert_rowid()").fetchone()[0]
            data = dict(data)
            data["snapshot"] = self._projection(request_id, sequence=cursor)
            self.db.execute("UPDATE events SET data=? WHERE cursor=?",
                            (json.dumps(data, sort_keys=True), cursor))

    def _projection(self, request_id: str, sequence: int | None = None) -> dict:
        # request_id is globally unique in this database.  Do not qualify this
        # historical lookup with the *current* store owner: a different-model
        # supersession must still be able to finish the old owner's event with
        # the old owner's immutable identity.
        row = self.db.execute(
            "SELECT * FROM requests WHERE request_id=?", (request_id,),
        ).fetchone()
        if not row:
            raise QueueError("unknown request")
        attempt = self.db.execute(
            "SELECT started,finished,gpu_ms,gpu_complete FROM attempts "
            "WHERE request_id=? ORDER BY COALESCE(finished,started) DESC LIMIT 1",
            (request_id,),
        ).fetchone()
        handoff = self.db.execute(
            "SELECT result_reference FROM handoffs WHERE request_id=? AND acknowledged=1",
            (request_id,),
        ).fetchone()
        seq = sequence if sequence is not None else self.db.execute(
            "SELECT COALESCE(MAX(cursor),0) FROM events WHERE request_id=?", (request_id,)
        ).fetchone()[0]
        running_done = None
        if row["running_at"] is not None and row["done_at"] is not None:
            running_done = max(0, round((row["done_at"] - row["running_at"]) * 1000))
        return {
            "requestId": row["request_id"], "schedulerId": row["scheduler_id"],
            "modelId": row["model_id"], "status": row["status"], "sequence": int(seq),
            "runningAt": row["running_at"], "doneAt": row["done_at"],
            "runningToDoneMs": running_done,
            "timeOnGpuMs": None if not attempt else attempt["gpu_ms"],
            "gpuTimingComplete": bool(attempt and attempt["gpu_complete"]),
            "errorCode": row["error_code"],
            "resultReference": None if not handoff else handoff["result_reference"],
            "cancellation": bool(row["cancellation"]),
        }

    def projection(self, request_id: str) -> dict:
        """Return the bounded public request projection."""
        self._owner()
        return self._projection(request_id)

    def event_highwater(self) -> int:
        self._owner()
        return int(self.db.execute("SELECT COALESCE(MAX(cursor),0) FROM events").fetchone()[0])

    def events(self, after: int = 0, limit: int | None = None):
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError("cursor must be a non-negative integer")
        if after > self.event_highwater():
            raise ValueError("cursor is ahead of durable history")
        sql = "SELECT * FROM events WHERE cursor>? ORDER BY cursor"
        args: list[object] = [after]
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        return self.db.execute(sql, args).fetchall()

    def request_events(self, request_id: str, after: int = 0, limit: int = 64):
        """Return one bounded request-specific cursor slice.

        Filtering in SQLite is essential: a global cursor slice can otherwise
        repeatedly contain unrelated events and permanently starve this request.
        """
        if (not isinstance(request_id, str) or not request_id or isinstance(after, bool)
                or not isinstance(after, int) or after < 0
                or isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError("invalid request event cursor or limit")
        self._owner()
        return self.db.execute(
            "SELECT * FROM events WHERE request_id=? AND cursor>? ORDER BY cursor LIMIT ?",
            (request_id, after, limit),
        ).fetchall()

    def request_history_has_legacy_event(self, request_id: str, after: int = 0) -> bool:
        """Boundedly test all replay rows for a usable immutable snapshot."""
        if (not isinstance(request_id, str) or not request_id or isinstance(after, bool)
                or not isinstance(after, int) or after < 0):
            raise ValueError("invalid request event cursor")
        self._owner()
        # CASE prevents json_type from being evaluated against malformed legacy
        # data. EXISTS returns one scalar rather than materializing history.
        row = self.db.execute(
            "SELECT EXISTS(SELECT 1 FROM events WHERE request_id=? AND cursor>? "
            "AND CASE WHEN json_valid(data) THEN json_type(data,'$.snapshot') "
            "ELSE NULL END IS NULL)",
            (request_id, after),
        ).fetchone()
        return bool(row[0])

    def _terminalize_all(self, reason: str) -> None:
        rows = self.db.execute(
            "SELECT request_id FROM requests WHERE status NOT IN ('done','error','cancelled')"
        ).fetchall()
        for row in rows:
            self._cancel_request_rows(row[0], "error", reason)

    def _cancel_request_rows(self, request_id: str, status: str, reason: str) -> None:
        self.db.execute(
            "UPDATE requests SET status=?, cancellation=?, error_code=? WHERE request_id=?",
            (status, int(status == "cancelled"), reason, request_id),
        )
        attempts = self.db.execute(
            "SELECT token FROM attempts WHERE request_id=? AND active=1", (request_id,)
        ).fetchall()
        self.db.execute("UPDATE attempts SET active=0 WHERE request_id=?", (request_id,))
        self._close_skip_groups(request_id)
        self.db.execute(
            "UPDATE outbox SET cancelled=1, acknowledged=1 WHERE request_id=? "
            "AND kind IN ('submit','handoff') AND acknowledged=0",
            (request_id,),
        )
        for attempt in attempts:
            key = f"cancel:{request_id}:{attempt[0]}"
            self.db.execute(
                "INSERT OR IGNORE INTO outbox(operation_id,kind,idempotency_key,"
                "request_id,token,payload_reference,created) VALUES(?,?,?,?,?,?,?)",
                (secrets.token_urlsafe(16), "cancel", key, request_id,
                 attempt[0], request_id, time.time()),
            )
        self._event(request_id, "status", {"status": status, "reason": reason})

    def _close_skip_groups(self, request_id: str) -> None:
        self.db.execute("UPDATE skip_groups SET closed=1 WHERE anchor=?", (request_id,))

    def _recover_prior_attempts(self) -> None:
        """Fence old execution without discarding an already staged handoff."""
        rows = self.db.execute(
            "SELECT a.request_id,a.token,r.status FROM attempts a JOIN requests r USING(request_id) "
            "WHERE a.active=1"
        ).fetchall()
        for row in rows:
            self.db.execute("UPDATE attempts SET active=0 WHERE token=?", (row["token"],))
            self.db.execute("UPDATE outbox SET cancelled=1,acknowledged=1 WHERE token=? AND kind='submit'", (row["token"],))
            handoff = self.db.execute(
                "SELECT 1 FROM handoffs WHERE request_id=? AND acknowledged=0", (row["request_id"],)
            ).fetchone()
            if not handoff and row["status"] in ("running", "on_gpu"):
                self.db.execute("UPDATE requests SET status='scheduled',next_attempt_at=? WHERE request_id=?", (time.time(), row["request_id"]))
                self._close_skip_groups(row["request_id"])
            key = f"cancel:{row['request_id']}:{row['token']}"
            self.db.execute(
                "INSERT OR IGNORE INTO outbox(operation_id,kind,idempotency_key,request_id,token,payload_reference,created) VALUES(?,?,?,?,?,?,?)",
                (secrets.token_urlsafe(16), "cancel", key, row["request_id"], row["token"], row["request_id"], time.time()),
            )

    def _require_accepting(self) -> None:
        self._owner()
        row = self.db.execute(
            "SELECT value FROM meta WHERE key='accepting'"
        ).fetchone()
        if not row or row[0] != "1":
            raise SessionError("scheduler is stopped")

    def get(self, request_id: str):
        self._owner()
        return self.db.execute(
            "SELECT * FROM requests WHERE scheduler_id=? AND request_id=?",
            (self.scheduler_id, request_id),
        ).fetchone()

    @staticmethod
    def descriptor(row, field: str) -> FunctionDescriptor | None:
        """Decode a descriptor from a raw request row without changing the row API."""
        if field not in ("ready", "template"):
            raise ValueError("descriptor field must be ready or template")
        value = row[field]
        return None if value is None else deserialize_function_descriptor(value)

    def enqueue(self, request_id: str, payload_reference: str,
                 dependencies: Iterable[str] = (),
                 insertion_mode: InsertionMode = InsertionMode.APPEND,
                 result_target: str = "local", idempotency_key: str | None = None,
                 *, ready: FunctionDescriptor | None = None,
                 template: FunctionDescriptor | None = None):
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key is required")
        mode = InsertionMode(insertion_mode)
        deps = tuple(dependencies)
        for dependency in deps:
            if not isinstance(dependency, str) or not dependency:
                raise DependencyError("dependencies must be non-empty text")
        ready_json = None if ready is None else serialize_function_descriptor(ready)
        template_json = None if template is None else serialize_function_descriptor(template)
        descriptor_ids = set()
        for descriptor_json in (ready_json, template_json):
            if descriptor_json is not None:
                descriptor_ids.update(deserialize_function_descriptor(descriptor_json).dependency_result_ids)
        if not descriptor_ids.issubset(set(deps)):
            raise DependencyError("function descriptor references undeclared dependency")
        self._begin()
        try:
            self._require_accepting()
            fingerprint = _fingerprint(payload_reference, deps, mode, result_target, request_id, ready_json, template_json)
            existing = self.db.execute(
                "SELECT * FROM requests WHERE scheduler_id=? AND request_id=?",
                (self.scheduler_id, request_id),
            ).fetchone()
            if existing:
                if (idempotency_key and existing["idempotency_key"] == idempotency_key
                        and existing["fingerprint"] == fingerprint):
                    self.db.execute("COMMIT")
                    return existing
                raise DuplicateRequest(request_id)
            if idempotency_key:
                key_row = self.db.execute(
                    "SELECT * FROM requests WHERE scheduler_id=? AND idempotency_key=?",
                    (self.scheduler_id, idempotency_key),
                ).fetchone()
                if key_row:
                    if key_row["fingerprint"] != fingerprint:
                        raise IdempotencyConflict(idempotency_key)
                    self.db.execute("COMMIT")
                    return key_row
            graph = {
                row[0]: tuple(json.loads(row[1])) for row in self.db.execute(
                    "SELECT request_id,dependencies FROM requests WHERE scheduler_id=?",
                    (self.scheduler_id,),
                )
            }
            if request_id in deps or any(dep not in graph for dep in deps):
                raise DependencyError("missing or self dependency")
            graph[request_id] = deps
            self._validate_graph(graph)
            sequence = self.db.execute(
                "SELECT COALESCE(MAX(insertion_seq),0)+1 FROM positions"
            ).fetchone()[0]
            rank, anchor = self._insert_rank(mode)
            self.db.execute(
                "INSERT INTO requests(request_id,scheduler_id,model_id,payload_reference,"
                "dependencies,insertion_mode,result_target,fingerprint,ready,template,idempotency_key,status) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (request_id, self.scheduler_id, self.model_id.value, payload_reference,
                  json.dumps(deps), mode.value, result_target, fingerprint,
                  ready_json, template_json, idempotency_key, RequestStatus.SCHEDULED.value),
            )
            self.db.execute(
                "INSERT INTO positions(request_id,rank,insertion_seq,mode,anchor,group_tail) "
                "VALUES(?,?,?,?,?,?)",
                (request_id, rank, sequence, mode.value, anchor, request_id),
            )
            if anchor:
                self.db.execute("UPDATE skip_groups SET tail=? WHERE anchor=?", (request_id, anchor))
            self._event(request_id, "enqueue", {"idempotency_key": idempotency_key})
            self._bump_version()
            self.db.execute("COMMIT")
            return self.get(request_id)
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    @staticmethod
    def _validate_graph(graph: dict[str, tuple[str, ...]]) -> None:
        colour: dict[str, int] = {}
        for root in graph:
            if colour.get(root):
                continue
            stack = [(root, False)]
            while stack:
                node, leaving = stack.pop()
                if leaving:
                    colour[node] = 2
                elif colour.get(node) == 1:
                    raise DependencyError("cyclic dependency")
                elif colour.get(node) != 2:
                    colour[node] = 1
                    stack.append((node, True))
                    stack.extend((dep, False) for dep in graph.get(node, ()))

    def _insert_rank(self, mode: InsertionMode) -> tuple[int, str | None]:
        if mode is InsertionMode.APPEND:
            return self.db.execute(
                "SELECT COALESCE(MAX(rank),-1)+1 FROM positions"
            ).fetchone()[0], None
        row = self.db.execute(
            "SELECT p.request_id,p.rank,p.anchor FROM positions p JOIN requests r "
            "ON r.request_id=p.request_id WHERE r.status='scheduled' ORDER BY p.rank LIMIT 1"
        ).fetchone()
        if not row:
            return self.db.execute(
                "SELECT COALESCE(MAX(rank),-1)+1 FROM positions"
            ).fetchone()[0], None
        anchor = row["request_id"]
        if row["anchor"]:
            original = self.db.execute(
                "SELECT r.status,g.closed FROM requests r LEFT JOIN skip_groups g ON g.anchor=r.request_id WHERE r.request_id=?", (row["anchor"],)
            ).fetchone()
            if original and original["status"] == RequestStatus.SCHEDULED.value and not original["closed"]:
                anchor = row["anchor"]
                tail = self.db.execute(
                    "SELECT p.rank FROM skip_groups g JOIN positions p ON p.request_id=g.tail "
                    "WHERE g.anchor=?", (anchor,)
                ).fetchone()
                rank = tail[0] + 1
            else:
                anchor = row["request_id"]
                rank = row["rank"]
        else:
            rank = row["rank"]
        # First make every rank negative, then assign the non-negative ranks.
        self.db.execute("UPDATE positions SET rank=-rank-1")
        self.db.execute(
            "UPDATE positions SET rank=-rank-1+CASE WHEN -rank-1>=? THEN 1 ELSE 0 END",
            (rank,),
        )
        if anchor:
            group = self.db.execute("SELECT closed FROM skip_groups WHERE anchor=?", (anchor,)).fetchone()
            if not group or group["closed"]:
                sequence = self.db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM skip_groups").fetchone()[0]
                self.db.execute(
                    "INSERT INTO skip_groups(anchor,tail,sequence,closed) VALUES(?,?,?,0) "
                    "ON CONFLICT(anchor) DO UPDATE SET tail=excluded.tail,sequence=excluded.sequence,closed=0",
                    (anchor, anchor, sequence),
                )
        return rank, anchor

    def list_positions(self):
        self._owner()
        return self.db.execute(
            "SELECT p.*,r.status FROM positions p JOIN requests r USING(request_id) "
            "WHERE r.scheduler_id=? ORDER BY p.rank", (self.scheduler_id,)
        ).fetchall()

    def eligible_candidates(self, limit: int = 32, after_rank: int | None = None):
        """Return one bounded FIFO slice; the evaluator applies readiness gates."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be positive")
        self._owner()
        sql = ("SELECT r.*,p.rank FROM requests r JOIN positions p USING(request_id) "
               "WHERE r.scheduler_id=? AND r.status='scheduled' ")
        args: list[object] = [self.scheduler_id]
        if after_rank is not None:
            sql += "AND p.rank>? "
            args.append(after_rank)
        sql += "ORDER BY p.rank LIMIT ?"
        args.append(limit)
        return self.db.execute(sql, args).fetchall()

    def acknowledged_dependency_results(self, request_id: str) -> dict[str, str] | None:
        self._owner()
        row = self.db.execute("SELECT dependencies FROM requests WHERE request_id=? AND scheduler_id=?", (request_id, self.scheduler_id)).fetchone()
        if not row:
            return None
        result = {}
        for dependency in json.loads(row[0]):
            handoff = self.db.execute("SELECT result_reference FROM handoffs WHERE request_id=? AND acknowledged=1", (dependency,)).fetchone()
            if not handoff:
                return None
            result[dependency] = handoff[0]
        return result

    def dependency_failed(self, request_id: str) -> bool:
        self._owner()
        row = self.db.execute("SELECT dependencies FROM requests WHERE request_id=? AND scheduler_id=?", (request_id, self.scheduler_id)).fetchone()
        if not row:
            return False
        # Enqueue rejects missing dependencies, but a damaged/restored database
        # must fail closed rather than turning a blocked request into an
        # evaluator task exception.  Missing graph nodes have the same public
        # outcome as a failed dependency: the request cannot ever become
        # eligible and must receive a durable structured error.
        for item in json.loads(row[0]):
            dependency = self.db.execute(
                "SELECT status FROM requests WHERE scheduler_id=? AND request_id=?",
                (self.scheduler_id, item),
            ).fetchone()
            if dependency is None or dependency[0] in ("error", "cancelled"):
                return True
        return False

    def _dependency_statuses(self, dependencies: list[str]) -> list[str | None]:
        """Read dependency state in this scheduler's graph.

        Corrupt/restored queues can contain a dependency whose request row is
        gone.  Callers treat that unresolvable dependency as failed rather than
        dereferencing a missing SQLite row or creating an attempt that can
        never become eligible.
        """
        return [
            dependency[0] if dependency is not None else None
            for item in dependencies
            for dependency in (self.db.execute(
                "SELECT status FROM requests WHERE scheduler_id=? AND request_id=?",
                (self.scheduler_id, item),
            ).fetchone(),)
        ]

    def claim_evaluated(self, capability: EvaluatedClaim, now: float | None = None,
                        lease_seconds: float = 30) -> str | None:
        """Atomically consume an evaluator capability and create submit intent."""
        now = time.time() if now is None else now
        if not isinstance(capability, EvaluatedClaim):
            raise ValueError("evaluated capability required")
        self._begin()
        try:
            self._require_accepting()
            if (self.session is None or capability.session_token != self.session.token or
                    capability.generation != self.session.generation or capability.version != self._version()):
                raise StaleCallback("evaluation is stale")
            row = self.db.execute("SELECT * FROM requests WHERE scheduler_id=? AND request_id=?", (self.scheduler_id, capability.request_id)).fetchone()
            proof = self.db.execute("SELECT * FROM evaluated_capabilities WHERE nonce=?", (capability.nonce,)).fetchone()
            if (not proof or proof["request_id"] != capability.request_id or proof["version"] != capability.version or
                    proof["session_token"] != capability.session_token or proof["generation"] != capability.generation or
                    proof["payload_reference"] != capability.payload_reference or proof["intent_fingerprint"] != capability.intent_fingerprint or
                    not row or row["status"] != "scheduled" or row["fingerprint"] != capability.intent_fingerprint or
                    row["evaluated_payload_reference"] != capability.payload_reference):
                self.db.execute("ROLLBACK")
                return None
            if row["next_attempt_at"] and row["next_attempt_at"] > now:
                self.db.execute("ROLLBACK"); return None
            if row["first_retry_at"] is not None and now - row["first_retry_at"] >= 300:
                self._cancel_request_rows(capability.request_id, "error", "retry_exhausted")
                self.db.execute("COMMIT"); return None
            dependencies = json.loads(row["dependencies"])
            statuses = self._dependency_statuses(dependencies)
            if any(s is None or s in ("error", "cancelled") for s in statuses):
                self._cancel_request_rows(capability.request_id, "error", "dependency_failed")
                self.db.execute("COMMIT"); return None
            if any(s != "done" for s in statuses):
                self.db.execute("ROLLBACK"); return None
            used = self.db.execute("DELETE FROM evaluated_capabilities WHERE nonce=?", (capability.nonce,)).rowcount
            if used != 1:
                self.db.execute("ROLLBACK"); return None
            token = secrets.token_urlsafe(24)
            self.db.execute("UPDATE requests SET status='running',running_at=COALESCE(running_at,?) WHERE request_id=?", (now, capability.request_id))
            self._close_skip_groups(capability.request_id)
            self.db.execute("INSERT INTO attempts(request_id,token,session,generation,started,lease_until,payload_reference) VALUES(?,?,?,?,?,?,?)", (capability.request_id, token, capability.session_token, capability.generation, now, now + lease_seconds, capability.payload_reference))
            self.db.execute("INSERT INTO outbox(operation_id,kind,idempotency_key,request_id,token,payload_reference,created) VALUES(?,?,?,?,?,?,?)", (secrets.token_urlsafe(16), "submit", token, capability.request_id, token, capability.payload_reference, now))
            self._event(capability.request_id, "claim", {"token": token, "evaluated": True})
            self._bump_version()
            self.db.execute("COMMIT")
            return token
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def record_evaluation(self, request_id: str, version: int, payload_reference: str) -> EvaluatedClaim:
        """Persist evaluated input only if the queue has not changed."""
        self._begin()
        try:
            self._owner()
            if version != self._version():
                raise StaleCallback("evaluation version changed")
            if not isinstance(payload_reference, str) or not payload_reference:
                raise ValueError("payload reference must be non-empty text")
            row = self.db.execute("SELECT * FROM requests WHERE request_id=? AND scheduler_id=? AND status='scheduled'", (request_id, self.scheduler_id)).fetchone()
            if not row:
                raise StaleCallback("request is no longer scheduled")
            self.db.execute("UPDATE requests SET evaluated_payload_reference=? WHERE request_id=?", (payload_reference, request_id))
            nonce = secrets.token_urlsafe(32)
            self.db.execute("INSERT INTO evaluated_capabilities VALUES(?,?,?,?,?,?,?)", (nonce, request_id, version, self.session.token, self.session.generation, payload_reference, row["fingerprint"]))
            capability = EvaluatedClaim(request_id, version, self.session.token, self.session.generation, payload_reference, row["fingerprint"], nonce)
            self.db.execute("COMMIT")
            return capability
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def fail_scheduled(self, request_id: str, error_code: str, expected_version: int,
                       expected_session: str, expected_generation: int) -> bool:
        self._begin()
        try:
            self._owner()
            if (self._version() != expected_version or self.session is None or
                    self.session.token != expected_session or self.session.generation != expected_generation):
                self.db.execute("ROLLBACK"); return False
            row = self.db.execute("SELECT status FROM requests WHERE request_id=? AND scheduler_id=?", (request_id, self.scheduler_id)).fetchone()
            if not row or row[0] != "scheduled":
                self.db.execute("ROLLBACK"); return False
            self._cancel_request_rows(request_id, "error", error_code)
            self.db.execute("COMMIT"); return True
        except Exception:
            self.db.execute("ROLLBACK"); raise

    def claim(self, request_id: str, session_token: str | None = None,
              generation: int | None = None, now: float | None = None,
              lease_seconds: float = 30) -> str | None:
        now = time.time() if now is None else now
        self._begin()
        try:
            self._require_accepting()
            if session_token != self.session.token or generation != self.session.generation:
                raise StaleCallback("not the current session")
            row = self.db.execute(
                "SELECT * FROM requests WHERE scheduler_id=? AND request_id=?",
                (self.scheduler_id, request_id),
            ).fetchone()
            if not row or row["status"] != RequestStatus.SCHEDULED.value:
                self.db.execute("ROLLBACK")
                return None
            if row["next_attempt_at"] and row["next_attempt_at"] > now:
                self.db.execute("ROLLBACK")
                return None
            if row["first_retry_at"] is not None and now - row["first_retry_at"] >= 300:
                self._cancel_request_rows(request_id, "error", "retry_exhausted")
                self.db.execute("COMMIT")
                return None
            dependencies = json.loads(row["dependencies"])
            dependency_statuses = self._dependency_statuses(dependencies)
            if any(status is None or status in (RequestStatus.ERROR.value, RequestStatus.CANCELLED.value)
                   for status in dependency_statuses):
                self._cancel_request_rows(request_id, "error", "dependency_failed")
                self.db.execute("COMMIT")
                return None
            if any(status != RequestStatus.DONE.value for status in dependency_statuses):
                self.db.execute("ROLLBACK")
                return None
            # Slice E persists intent but does not let a generic claim path
            # execute it. A future scheduler-owned evaluator must own this gate.
            if row["ready"] is not None or row["template"] is not None:
                self.db.execute("ROLLBACK")
                return None
            token = secrets.token_urlsafe(24)
            self.db.execute(
                "UPDATE requests SET status='running',running_at=COALESCE(running_at,?) "
                "WHERE scheduler_id=? AND request_id=?",
                (now, self.scheduler_id, request_id),
            )
            self._close_skip_groups(request_id)
            self.db.execute(
                "INSERT INTO attempts(request_id,token,session,generation,started,lease_until,payload_reference) "
                "VALUES(?,?,?,?,?,?,?)",
                (request_id, token, session_token, generation, now, now + lease_seconds,
                 row["payload_reference"]),
            )
            self.db.execute(
                "INSERT INTO outbox(operation_id,kind,idempotency_key,request_id,token,"
                "payload_reference,created) VALUES(?,?,?,?,?,?,?)",
                (secrets.token_urlsafe(16), "submit", token, request_id, token,
                 row["payload_reference"], now),
            )
            self._event(request_id, "claim", {"token": token})
            self.db.execute("COMMIT")
            return token
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def _guard(self, request_id: str, token: str, session: str,
               generation: int):
        self._owner()
        if self.session is None or session != self.session.token or generation != self.session.generation:
            raise StaleCallback("session fence mismatch")
        row = self.db.execute(
            "SELECT * FROM attempts WHERE request_id=? AND token=? AND session=? "
            "AND generation=? AND active=1", (request_id, token, session, generation)
        ).fetchone()
        if not row:
            raise StaleCallback("attempt fence mismatch")
        return row

    def renew_lease(self, request_id: str, token: str, session: str,
                    generation: int, lease_seconds: float = 30,
                    now: float | None = None) -> bool:
        now = time.time() if now is None else now
        self._begin()
        try:
            self._guard(request_id, token, session, generation)
            self.db.execute("UPDATE attempts SET lease_until=? WHERE token=?",
                            (now + lease_seconds, token))
            self.db.execute("COMMIT")
            return True
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def reconcile_expired(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        self._begin()
        try:
            self._owner()
            rows = self.db.execute(
                "SELECT request_id,token FROM attempts WHERE active=1 AND lease_until<=?",
                (now,),
            ).fetchall()
            for row in rows:
                self.db.execute("UPDATE attempts SET active=0 WHERE token=?", (row["token"],))
                handoff = self.db.execute(
                    "SELECT 1 FROM handoffs WHERE request_id=? AND acknowledged=0", (row["request_id"],)
                ).fetchone()
                if not handoff:
                    self.db.execute("UPDATE requests SET status='scheduled',next_attempt_at=? WHERE request_id=? AND status IN ('running','on_gpu')", (now, row["request_id"]))
                    self._close_skip_groups(row["request_id"])
                self.db.execute(
                    "UPDATE outbox SET cancelled=1,acknowledged=1 WHERE token=? AND kind='submit'",
                    (row["token"],),
                )
                self._event(row["request_id"], "lease_expired", {})
            if rows:
                self._bump_version()
            self.db.execute("COMMIT")
            return len(rows)
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def transition(self, request_id: str, new_status: RequestStatus,
                   token: str, session: str, generation: int, **fields):
        new_status = RequestStatus(new_status)
        self._begin()
        try:
            self._guard(request_id, token, session, generation)
            row = self.db.execute(
                "SELECT * FROM requests WHERE scheduler_id=? AND request_id=?",
                (self.scheduler_id, request_id),
            ).fetchone()
            old = RequestStatus(row["status"])
            if old == new_status:
                self.db.execute("COMMIT")
                return row
            if not can_transition(old, new_status):
                raise InvalidTransition(f"{old.value}->{new_status.value}")
            if new_status is RequestStatus.DONE:
                raise InvalidTransition("done is only set by acknowledge_handoff")
            if new_status in (RequestStatus.ERROR, RequestStatus.CANCELLED):
                self._cancel_request_rows(request_id, new_status.value, fields.get("error_code", new_status.value))
                self.db.execute("COMMIT")
                return self.get(request_id)
            raise InvalidTransition("use claim, retry, mark_on_gpu, or finish_attempt")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def mark_on_gpu(self, request_id: str, token: str, session: str,
                    generation: int, at: float | None = None) -> None:
        at = time.time() if at is None else at
        if not isinstance(at, (int, float)) or isinstance(at, bool) or not __import__('math').isfinite(at):
            raise ValueError("GPU admission time must be finite")
        self._begin()
        try:
            self._guard(request_id, token, session, generation)
            attempt = self.db.execute("SELECT started,finished,gpu_start FROM attempts WHERE token=?", (token,)).fetchone()
            if attempt["finished"] is not None:
                self.db.execute("COMMIT")
                return
            if at < attempt["started"]:
                raise ValueError("GPU admission cannot precede attempt start")
            request = self.db.execute("SELECT status FROM requests WHERE request_id=?", (request_id,)).fetchone()
            if request[0] == "running":
                self.db.execute("UPDATE requests SET status='on_gpu' WHERE request_id=?", (request_id,))
            elif request[0] != "on_gpu":
                raise InvalidTransition(f"{request[0]}->on_gpu")
            self.db.execute("UPDATE attempts SET gpu_start=COALESCE(gpu_start,?) WHERE token=?", (at, token))
            self._event(request_id, "status", {"new": "on_gpu"})
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def finish_attempt(self, request_id: str, token: str, session: str,
                       generation: int, at: float | None = None,
                       gpu_timing_complete: bool = True,
                       gpu_ms: int | None = None) -> None:
        at = time.time() if at is None else at
        self._begin()
        try:
            attempt = self._guard(request_id, token, session, generation)
            if attempt["finished"] is not None:
                self.db.execute("COMMIT")
                return
            if not isinstance(at, (int, float)) or isinstance(at, bool) or not __import__('math').isfinite(at) or at < attempt["started"]:
                raise ValueError("finished time must be finite and not precede start")
            if gpu_ms is not None and (not isinstance(gpu_ms, int) or isinstance(gpu_ms, bool) or gpu_ms < 0):
                raise ValueError("gpu_ms must be a non-negative integer")
            if attempt["gpu_start"] is not None and at < attempt["gpu_start"]:
                raise ValueError("finished time cannot precede GPU admission")
            if attempt["gpu_start"] is None:
                gpu_timing_complete = False
                gpu_ms = None
            elif not gpu_timing_complete:
                gpu_ms = None
            if gpu_ms is None and gpu_timing_complete and attempt["gpu_start"] is not None:
                gpu_ms = max(0, round((at - attempt["gpu_start"]) * 1000))
            self.db.execute(
                "UPDATE attempts SET finished=?,gpu_end=?,gpu_ms=?,gpu_complete=? WHERE token=?",
                (at, at if attempt["gpu_start"] is not None else None, gpu_ms,
                 int(gpu_timing_complete and gpu_ms is not None), token),
            )
            request = self.db.execute("SELECT status FROM requests WHERE request_id=?", (request_id,)).fetchone()
            if request["status"] == "on_gpu":
                self.db.execute("UPDATE requests SET status='running' WHERE request_id=?", (request_id,))
                self._event(request_id, "status", {"new": "running"})
            # A durable result means provider execution is over even if the
            # provider could not supply GPU telemetry.
            self.db.execute(
                "UPDATE attempts SET finished=COALESCE(finished,?),gpu_ms=CASE WHEN finished IS NULL THEN NULL ELSE gpu_ms END,gpu_complete=CASE WHEN finished IS NULL THEN 0 ELSE gpu_complete END WHERE token=?",
                (time.time(), token),
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def stage_handoff(self, request_id: str, token: str, session: str,
                      generation: int, result_reference: str,
                      idempotency_key: str):
        if len(result_reference) != 64 or any(c not in "0123456789abcdef" for c in result_reference):
            raise ValueError("result_reference must be a SHA-256 digest")
        self._begin()
        try:
            self._guard(request_id, token, session, generation)
            old = self.db.execute(
                "SELECT * FROM handoffs WHERE request_id=?", (request_id,)
            ).fetchone()
            if old:
                if old["idempotency_key"] != idempotency_key or old["result_reference"] != result_reference:
                    raise IdempotencyConflict(idempotency_key)
                self.db.execute("COMMIT")
                return old
            request = self.db.execute("SELECT status FROM requests WHERE request_id=?", (request_id,)).fetchone()
            if request["status"] == "on_gpu":
                self.db.execute("UPDATE requests SET status='running' WHERE request_id=?", (request_id,))
                self._event(request_id, "status", {"new": "running"})
            self.db.execute(
                "INSERT INTO handoffs(request_id,token,idempotency_key,result_reference) VALUES(?,?,?,?)",
                (request_id, token, idempotency_key, result_reference),
            )
            self.db.execute(
                "INSERT INTO outbox(operation_id,kind,idempotency_key,request_id,token,payload_reference,created) "
                "VALUES(?,?,?,?,?,?,?)",
                (secrets.token_urlsafe(16), "handoff", idempotency_key, request_id,
                 token, result_reference, time.time()),
            )
            self._bump_version()
            self.db.execute("COMMIT")
            return self.db.execute("SELECT * FROM handoffs WHERE request_id=?", (request_id,)).fetchone()
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def stage_result(self, request_id: str, token: str, session: str, generation: int,
                     result_store, content: bytes, idempotency_key: str):
        """Durably write result bytes before recording the handoff transaction."""
        return self.stage_handoff(request_id, token, session, generation,
                                  result_store.write(content), idempotency_key)

    def acknowledge_handoff(self, request_id: str, idempotency_key: str):
        self._begin()
        try:
            self._owner()
            row = self.db.execute(
                "SELECT h.*,r.status FROM handoffs h JOIN requests r USING(request_id) "
                "WHERE h.request_id=? AND h.idempotency_key=? AND r.scheduler_id=?",
                (request_id, idempotency_key, self.scheduler_id),
            ).fetchone()
            if not row:
                raise QueueError("unknown handoff")
            delivery = self.db.execute(
                "SELECT acknowledged,cancelled,token FROM outbox WHERE kind='handoff' AND idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if row["acknowledged"]:
                self.db.execute("COMMIT")
                return self.get(request_id)
            if not delivery or delivery["cancelled"] or delivery["token"] != row["token"]:
                raise StaleCallback("handoff is no longer pending")
            if row["status"] in (RequestStatus.CANCELLED.value, RequestStatus.ERROR.value):
                self.db.execute("UPDATE outbox SET cancelled=1,acknowledged=1 WHERE idempotency_key=?", (idempotency_key,))
            else:
                self.db.execute("UPDATE handoffs SET acknowledged=1 WHERE request_id=?", (request_id,))
                self.db.execute("UPDATE outbox SET acknowledged=1 WHERE idempotency_key=?", (idempotency_key,))
                self.db.execute("UPDATE attempts SET active=0 WHERE request_id=?", (request_id,))
                if row["status"] != RequestStatus.DONE.value:
                    self.db.execute("UPDATE requests SET status='done',done_at=? WHERE request_id=?", (time.time(), request_id))
                    self._event(request_id, "status", {"new": "done"})
                self._bump_version()
            self.db.execute("COMMIT")
            return self.get(request_id)
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def acknowledge_outbox(self, idempotency_key: str) -> bool:
        """Acknowledge delivery only; submit acknowledgement is not execution."""
        self._begin()
        try:
            self._owner()
            row = self.db.execute(
                "SELECT acknowledged,cancelled FROM outbox WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if not row:
                raise QueueError("unknown outbox operation")
            kind = self.db.execute("SELECT kind FROM outbox WHERE idempotency_key=?", (idempotency_key,)).fetchone()[0]
            if kind == "handoff":
                raise InvalidTransition("use acknowledge_handoff for handoff delivery")
            if not row["cancelled"]:
                self.db.execute(
                    "UPDATE outbox SET acknowledged=1 WHERE idempotency_key=?",
                    (idempotency_key,),
                )
                self._bump_version()
            self.db.execute("COMMIT")
            return bool(row["acknowledged"])
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def retry(self, request_id: str, token: str, session: str,
              generation: int, failure_elapsed: float = 0,
              now: float | None = None, retryable: bool = True):
        now = time.time() if now is None else now
        self._begin()
        try:
            self._guard(request_id, token, session, generation)
            row = self.get(request_id)
            if self.db.execute("SELECT 1 FROM handoffs WHERE request_id=? AND acknowledged=0", (request_id,)).fetchone():
                raise InvalidTransition("a publishing attempt cannot be retried")
            if not retryable:
                self._terminalize_all("non_retryable")
                self.db.execute("UPDATE meta SET value='0' WHERE key='accepting'")
            else:
                first = now if row["first_retry_at"] is None else row["first_retry_at"]
                elapsed = max(0, now - first)
                count = row["retry_count"] + 1
                if elapsed >= 300:
                    self._cancel_request_rows(request_id, "error", "retry_exhausted")
                    self.db.execute("UPDATE requests SET retry_elapsed=?,first_retry_at=?,retry_count=? WHERE request_id=?", (elapsed, first, count, request_id))
                else:
                    delay = min(30, 5 * (2 ** (count - 1)))
                    self.db.execute(
                        "UPDATE requests SET status='scheduled',retry_elapsed=?,first_retry_at=?,"
                        "retry_count=?,next_attempt_at=? WHERE request_id=?",
                        (elapsed, first, count, now + delay, request_id),
                    )
                    self.db.execute("UPDATE attempts SET active=0 WHERE token=?", (token,))
                    self.db.execute("UPDATE outbox SET cancelled=1,acknowledged=1 WHERE token=? AND kind='submit'", (token,))
                    key = f"cancel:{request_id}:{token}"
                    self.db.execute("INSERT OR IGNORE INTO outbox(operation_id,kind,idempotency_key,request_id,token,payload_reference,created) VALUES(?,?,?,?,?,?,?)", (secrets.token_urlsafe(16), "cancel", key, request_id, token, request_id, now))
            self._event(request_id, "retry", {"retryable": retryable})
            self._bump_version()
            self.db.execute("COMMIT")
            return self.get(request_id)
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def cancel(self, request_id: str, idempotency_key: str | None = None):
        self._begin()
        try:
            if idempotency_key is not None:
                prior = self.db.execute("SELECT fingerprint FROM scheduler_operations WHERE scheduler_id=? AND operation='cancel' AND idempotency_key=?",
                                        (self.scheduler_id, idempotency_key)).fetchone()
                if prior:
                    if prior[0] != request_id:
                        raise IdempotencyConflict(idempotency_key)
                    current = self.db.execute("SELECT * FROM requests WHERE scheduler_id=? AND request_id=?",
                                              (self.scheduler_id, request_id)).fetchone()
                    self.db.execute("COMMIT")
                    return current
            self._owner()
            row = self.db.execute(
                "SELECT status FROM requests WHERE scheduler_id=? AND request_id=?",
                (self.scheduler_id, request_id),
            ).fetchone()
            if not row:
                raise QueueError("unknown request")
            generation = self.session.generation
            fingerprint = request_id
            if idempotency_key is not None:
                if not isinstance(idempotency_key, str) or not idempotency_key:
                    raise ValueError("idempotency_key must be non-empty text")
                conflict = self.db.execute("SELECT operation FROM scheduler_operations WHERE scheduler_id=? AND idempotency_key=?",
                                           (self.scheduler_id, idempotency_key)).fetchone()
                if conflict and conflict[0] != "cancel":
                    raise IdempotencyConflict(idempotency_key)
                old = self.db.execute(
                    "SELECT * FROM scheduler_operations WHERE scheduler_id=? AND operation='cancel' AND idempotency_key=?",
                    (self.scheduler_id, idempotency_key)).fetchone()
                if old:
                    if old["fingerprint"] != fingerprint:
                        raise IdempotencyConflict(idempotency_key)
                    result = json.loads(old["outcome"])
                    if old["target_generation"] == generation:
                        current = self.db.execute("SELECT * FROM requests WHERE scheduler_id=? AND request_id=?", (self.scheduler_id, request_id)).fetchone()
                        self.db.execute("COMMIT")
                        return current
                    self.db.execute("COMMIT")
                    return self.db.execute("SELECT * FROM requests WHERE scheduler_id=? AND request_id=?", (self.scheduler_id, request_id)).fetchone()
                self.db.execute("INSERT INTO scheduler_operations VALUES(?,?,?,?,?,?,?)",
                                 (self.scheduler_id, "cancel", idempotency_key, fingerprint, generation, "pending", time.time()))
            if row[0] not in {x.value for x in TERMINAL}:
                self._cancel_request_rows(request_id, "cancelled", "client_cancelled")
                self._bump_version()
            if idempotency_key is not None:
                self.db.execute("UPDATE scheduler_operations SET outcome=? WHERE scheduler_id=? AND operation='cancel' AND idempotency_key=?",
                                 (json.dumps({"status": "cancelled" if row[0] not in {x.value for x in TERMINAL} else row[0]}), self.scheduler_id, idempotency_key))
            self.db.execute("COMMIT")
            return self.get(request_id)
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def stop(self, reason: str = "stopped", cancelled: bool = False,
             idempotency_key: str | None = None) -> int:
        self._begin()
        try:
            if idempotency_key is not None:
                fingerprint = json.dumps([reason, cancelled], separators=(",", ":"))
                prior = self.db.execute("SELECT fingerprint,outcome FROM scheduler_operations WHERE scheduler_id=? AND operation='stop' AND idempotency_key=?",
                                        (self.scheduler_id, idempotency_key)).fetchone()
                if prior:
                    if prior[0] != fingerprint:
                        raise IdempotencyConflict(idempotency_key)
                    result = int(json.loads(prior[1])["count"])
                    self.db.execute("COMMIT")
                    return result
            self._owner()
            status = "cancelled" if cancelled else "error"
            generation = self.session.generation
            if idempotency_key is not None:
                if not isinstance(idempotency_key, str) or not idempotency_key:
                    raise ValueError("idempotency_key must be non-empty text")
                fingerprint = json.dumps([reason, cancelled], separators=(",", ":"))
                conflict = self.db.execute("SELECT operation FROM scheduler_operations WHERE scheduler_id=? AND idempotency_key=?",
                                           (self.scheduler_id, idempotency_key)).fetchone()
                if conflict and conflict[0] != "stop":
                    raise IdempotencyConflict(idempotency_key)
                old = self.db.execute("SELECT * FROM scheduler_operations WHERE scheduler_id=? AND operation='stop' AND idempotency_key=?",
                                      (self.scheduler_id, idempotency_key)).fetchone()
                if old:
                    if old["fingerprint"] != fingerprint:
                        raise IdempotencyConflict(idempotency_key)
                    result = int(json.loads(old["outcome"])["count"])
                    self.db.execute("COMMIT")
                    return result
                self.db.execute("INSERT INTO scheduler_operations VALUES(?,?,?,?,?,?,?)",
                                 (self.scheduler_id, "stop", idempotency_key, fingerprint, generation, "pending", time.time()))
            rows = self.db.execute(
                "SELECT request_id FROM requests WHERE scheduler_id=? AND status NOT IN ('done','error','cancelled')",
                (self.scheduler_id,),
            ).fetchall()
            for row in rows:
                self._cancel_request_rows(row[0], status, reason)
            self.db.execute("UPDATE meta SET value='0' WHERE key='accepting'")
            if idempotency_key is not None:
                self.db.execute("UPDATE scheduler_operations SET outcome=? WHERE scheduler_id=? AND operation='stop' AND idempotency_key=?",
                                 (json.dumps({"count": len(rows), "status": status}), self.scheduler_id, idempotency_key))
            self.db.execute("COMMIT")
            return len(rows)
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def outbox(self, include_acknowledged: bool = False):
        self._owner()
        condition = "" if include_acknowledged else "WHERE acknowledged=0 AND cancelled=0"
        return self.db.execute(
            f"SELECT * FROM outbox {condition} ORDER BY created,operation_id"
        ).fetchall()

    def pending_outbox(self, kind: str | None = None, limit: int = 64):
        """Return a bounded immutable view of undelivered durable work."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be positive")
        self._owner()
        if kind is None:
            return self.db.execute(
                "SELECT * FROM outbox WHERE acknowledged=0 AND cancelled=0 "
                "ORDER BY created,operation_id LIMIT ?", (limit,)).fetchall()
        return self.db.execute(
            "SELECT * FROM outbox WHERE acknowledged=0 AND cancelled=0 AND kind=? "
            "ORDER BY created,operation_id LIMIT ?", (kind, limit)).fetchall()

    def attempt(self, request_id: str, token: str):
        self._owner()
        return self.db.execute(
            "SELECT a.*,d.payload_digest,d.context_size,d.bucket_identity "
            "FROM attempts a LEFT JOIN dispatch_metadata d ON d.token=a.token "
            "WHERE a.request_id=? AND a.token=?", (request_id, token)).fetchone()

    def pending_submissions(self, limit: int = 16):
        """Return replayable submit intents with their persisted decoded input."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be positive")
        self._owner()
        return self.db.execute(
            "SELECT o.*,d.payload_digest,d.context_size,d.bucket_identity FROM outbox o "
            "JOIN dispatch_metadata d ON d.token=o.token JOIN requests r ON r.request_id=o.request_id "
            "WHERE o.kind='submit' AND o.acknowledged=0 AND o.cancelled=0 "
            "AND r.status IN ('running','on_gpu') ORDER BY o.created,o.operation_id LIMIT ?",
            (limit,),
        ).fetchall()

    def active_attempts(self, limit: int = 64, after_token: str | None = None):
        """Bounded view of every live attempt used only for lease renewal."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be positive")
        self._owner()
        return self.db.execute(
            "SELECT a.request_id,a.token FROM attempts a WHERE a.active=1 "
            "AND a.token>? ORDER BY a.token LIMIT ?", (after_token or "", limit)
        ).fetchall()

    def persist_dispatch_metadata(self, token: str, payload_digest: str,
                                  context_size: int | None = None,
                                  bucket_identity: str | None = None) -> None:
        """Persist the exact decoded dispatch identity before network submit."""
        if len(payload_digest) != 64 or any(c not in "0123456789abcdef" for c in payload_digest):
            raise ValueError("payload_digest must be a SHA-256 digest")
        self._begin()
        try:
            self._owner()
            attempt = self.db.execute(
                "SELECT a.request_id,a.session,a.generation,a.active,r.status FROM attempts a "
                "JOIN requests r USING(request_id) WHERE a.token=?", (token,)
            ).fetchone()
            if (not attempt or not attempt["active"] or attempt["session"] != self.session.token
                    or attempt["generation"] != self.session.generation
                    or attempt["status"] not in ("running", "on_gpu")):
                raise StaleCallback("dispatch metadata attempt is stale")
            old = self.db.execute("SELECT * FROM dispatch_metadata WHERE token=?", (token,)).fetchone()
            values = (payload_digest, context_size, bucket_identity)
            if old:
                if (old["payload_digest"], old["context_size"], old["bucket_identity"]) != values:
                    raise IdempotencyConflict("dispatch metadata is immutable")
            else:
                self.db.execute(
                    "INSERT INTO dispatch_metadata(token,payload_digest,context_size,bucket_identity,created) VALUES(?,?,?,?,?)",
                    (token, *values, time.time()),
                )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def fail_attempt(self, request_id: str, token: str, session: str, generation: int,
                     error_code: str) -> None:
        """Terminalize one fenced attempt without aborting unrelated queue work."""
        self._begin()
        try:
            self._guard(request_id, token, session, generation)
            self._cancel_request_rows(request_id, "error", error_code)
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def acknowledge_submit(self, idempotency_key: str) -> bool:
        """Record RM acceptance; replay remains safe because the key is stable."""
        self._begin()
        try:
            self._owner()
            row = self.db.execute(
                "SELECT acknowledged,cancelled,kind FROM outbox WHERE idempotency_key=?",
                (idempotency_key,)).fetchone()
            if not row or row["kind"] != "submit":
                raise QueueError("unknown submit outbox operation")
            if not row["cancelled"] and not row["acknowledged"]:
                self.db.execute("UPDATE outbox SET acknowledged=1 WHERE idempotency_key=?", (idempotency_key,))
            self.db.execute("COMMIT")
            return bool(row["acknowledged"])
        except Exception:
            self.db.execute("ROLLBACK")
            raise

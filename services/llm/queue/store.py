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

from .contracts import InsertionMode, ModelId, RequestStatus, TERMINAL, can_transition


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


@dataclass(frozen=True)
class Session:
    scheduler_id: str
    model_id: ModelId
    token: str
    generation: int


def _fingerprint(payload_reference: str, dependencies: tuple[str, ...],
                  insertion_mode: InsertionMode, result_target: str,
                  request_id: str) -> str:
    value = json.dumps(
        [request_id, payload_reference, dependencies, insertion_mode.value, result_target],
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
                idempotency_key TEXT, status TEXT NOT NULL,
                cancellation INTEGER NOT NULL DEFAULT 0,
                running_at REAL, done_at REAL, next_attempt_at REAL,
                retry_elapsed REAL NOT NULL DEFAULT 0,
                first_retry_at REAL, retry_count INTEGER NOT NULL DEFAULT 0,
                error_code TEXT
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
                active INTEGER NOT NULL DEFAULT 1
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
            """
        )
        model = self.db.execute("SELECT value FROM meta WHERE key='model_id'").fetchone()
        if not model:
            self.db.execute(
                "INSERT INTO meta(key,value) VALUES('model_id',?)",
                (self.model_id.value,),
            )

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

    def start_session(self) -> Session:
        """Atomically acquire ownership, superseding any prior scheduler."""
        if self.session is not None:
            try:
                self._owner()
                return self.session
            except SessionError:
                self.session = None
        self._begin()
        try:
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
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        self.session = Session(self.scheduler_id, self.model_id, token, generation)
        return self.session

    def recover_session(self) -> Session:
        """Acquire a new fenced session for the same durable scheduler."""
        owner = self.db.execute(
            "SELECT value FROM meta WHERE key='scheduler_id'"
        ).fetchone()
        if owner and owner[0] != self.scheduler_id:
            return self.start_session()
        return self.start_session()

    def _event(self, request_id: str | None, kind: str, data: dict) -> None:
        self.db.execute(
            "INSERT INTO events(request_id,kind,data,created) VALUES(?,?,?,?)",
            (request_id, kind, json.dumps(data, sort_keys=True), time.time()),
        )

    def events(self, after: int = 0, limit: int | None = None):
        sql = "SELECT * FROM events WHERE cursor>? ORDER BY cursor"
        args: list[object] = [after]
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        return self.db.execute(sql, args).fetchall()

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

    def enqueue(self, request_id: str, payload_reference: str,
                dependencies: Iterable[str] = (),
                insertion_mode: InsertionMode = InsertionMode.APPEND,
                result_target: str = "local", idempotency_key: str | None = None):
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key is required")
        mode = InsertionMode(insertion_mode)
        deps = tuple(dependencies)
        self._begin()
        try:
            self._require_accepting()
            fingerprint = _fingerprint(payload_reference, deps, mode, result_target, request_id)
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
                "dependencies,insertion_mode,result_target,fingerprint,idempotency_key,status) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (request_id, self.scheduler_id, self.model_id.value, payload_reference,
                 json.dumps(deps), mode.value, result_target, fingerprint,
                 idempotency_key, RequestStatus.SCHEDULED.value),
            )
            self.db.execute(
                "INSERT INTO positions(request_id,rank,insertion_seq,mode,anchor,group_tail) "
                "VALUES(?,?,?,?,?,?)",
                (request_id, rank, sequence, mode.value, anchor, request_id),
            )
            if anchor:
                self.db.execute("UPDATE skip_groups SET tail=? WHERE anchor=?", (request_id, anchor))
            self._event(request_id, "enqueue", {"idempotency_key": idempotency_key})
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
            dependency_statuses = [self.db.execute(
                "SELECT status FROM requests WHERE scheduler_id=? AND request_id=?",
                (self.scheduler_id, dep),
            ).fetchone()[0] for dep in dependencies]
            if any(status in (RequestStatus.ERROR.value, RequestStatus.CANCELLED.value)
                   for status in dependency_statuses):
                self._cancel_request_rows(request_id, "error", "dependency_failed")
                self.db.execute("COMMIT")
                return None
            if any(status != RequestStatus.DONE.value for status in dependency_statuses):
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
                "INSERT INTO attempts(request_id,token,session,generation,started,lease_until) "
                "VALUES(?,?,?,?,?,?)",
                (request_id, token, session_token, generation, now, now + lease_seconds),
            )
            self.db.execute(
                "INSERT INTO outbox(operation_id,kind,idempotency_key,request_id,token,"
                "payload_reference,created) VALUES(?,?,?,?,?,?,?)",
                (secrets.token_urlsafe(16), "submit", token, request_id, token,
                 request_id, now),
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
            self.db.execute("COMMIT")
            return self.get(request_id)
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def cancel(self, request_id: str):
        self._begin()
        try:
            self._owner()
            row = self.db.execute(
                "SELECT status FROM requests WHERE scheduler_id=? AND request_id=?",
                (self.scheduler_id, request_id),
            ).fetchone()
            if not row:
                raise QueueError("unknown request")
            if row[0] not in {x.value for x in TERMINAL}:
                self._cancel_request_rows(request_id, "cancelled", "client_cancelled")
            self.db.execute("COMMIT")
            return self.get(request_id)
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def stop(self, reason: str = "stopped", cancelled: bool = False) -> int:
        self._begin()
        try:
            self._owner()
            status = "cancelled" if cancelled else "error"
            rows = self.db.execute(
                "SELECT request_id FROM requests WHERE scheduler_id=? AND status NOT IN ('done','error','cancelled')",
                (self.scheduler_id,),
            ).fetchall()
            for row in rows:
                self._cancel_request_rows(row[0], status, reason)
            self.db.execute("UPDATE meta SET value='0' WHERE key='accepting'")
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

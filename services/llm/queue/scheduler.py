"""Bounded asynchronous coordinator for the durable queue."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from .contracts import ModelId
from .eligibility import EligibilityEvaluator
from .results import LocalPublisher, ResultStore
from .store import QueueStore, Session, SessionError, StaleCallback
from ..resource_manager.contracts import CapacityProfile
from ..resource_manager.protocol import EventKind, Failure, ProgressEvent, ResourceManagerClient


@dataclass(frozen=True)
class DispatchContext:
    context_size: int | None = None
    bucket_identity: str | None = None


@dataclass(frozen=True)
class DecodedPayload:
    payload: bytes
    context: DispatchContext = DispatchContext()


class QueueScheduler:
    """One owner, one local fence, and one freshly fenced RM session."""

    def __init__(self, store: QueueStore, resource_manager: ResourceManagerClient,
                 profile: CapacityProfile, provider: Any, *, decoder: Callable[[str], Any],
                 result_store: ResultStore, publisher: LocalPublisher,
                 evaluator: EligibilityEvaluator | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 watchdog_seconds: float = 60.0, loop_interval: float = .05,
                 stop_timeout: float = 10.0):
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
               for value in (watchdog_seconds, loop_interval, stop_timeout)):
            raise ValueError("scheduler timeouts must be positive")
        self.store, self.rm, self.profile, self.provider = store, resource_manager, profile, provider
        self.decoder, self.result_store, self.publisher = decoder, result_store, publisher
        # Capture connection metadata on its owning thread. SQLite connections
        # are deliberately never touched from worker threads.
        self._publisher_state_path = self.publisher.db.execute("PRAGMA database_list").fetchone()[2]
        self.evaluator = evaluator or EligibilityEvaluator(store)
        self.clock, self.watchdog_seconds, self.loop_interval, self.stop_timeout = clock, watchdog_seconds, loop_interval, stop_timeout
        self.local_session = self.rm_session = None
        self._tasks: set[asyncio.Task[Any]] = set()
        # A cancelled asyncio wrapper does not stop its executor thread. Keep
        # publication workers separately so no new fence starts while an old
        # publisher may still be inside its transaction.
        self._publisher_workers: set[asyncio.Task[Any]] = set()
        self._wake = asyncio.Event()
        self._started = False
        self._stopping = False
        self._pending: dict[tuple[str, str], tuple[bytes, DispatchContext]] = {}
        self._active: set[tuple[str, str]] = set()
        self._seen_sequences: set[int] = set()
        self._cursor = 0
        self._watchdog_deadline: float | None = None
        self._eligible = False
        self._lease_cursor = None
        self._completion_sequence = 0
        self._result_pending: dict[tuple[str, str], ProgressEvent] = {}
        self._lifecycle_lock = asyncio.Lock()
        self._stop_result = 0
        self._lifecycle_epoch = 0

    async def start(self, *, idempotency_key: str | None = None):
        async with self._lifecycle_lock:
            if self._started:
                return self.rm_session
            if any(not task.done() for task in self._tasks | self._publisher_workers):
                raise RuntimeError("previous coordinator tasks have not stopped")
            self._reset_process_state()
            epoch = self._lifecycle_epoch
            if idempotency_key is None:
                self.local_session = self.store.recover_session()
                key = f"scheduler-start:{self.store.scheduler_id}:{self.local_session.generation}:{time.time_ns()}"
            else:
                self.local_session = self.store.start_session(idempotency_key)
                key = idempotency_key
            rm_session = await self.rm.start_session(
                self.store.scheduler_id, ModelId(self.store.model_id), self.profile, self.provider,
                idempotency_key=key)
            valid = epoch == self._lifecycle_epoch and not self._stopping
            try:
                if idempotency_key is not None:
                    if valid:
                        self.store.complete_start(idempotency_key, self.local_session)
                    else:
                        valid = False
                else:
                    # complete_start performs the same durable fence check for
                    # keyed starts; the owner check here closes the legacy path.
                    self.store._owner()
                    accepting = self.store.db.execute("SELECT value FROM meta WHERE key='accepting'").fetchone()
                    valid = valid and accepting is not None and accepting[0] == "1"
            except SessionError:
                valid = False
            if not valid:
                await self._stop_late_rm(rm_session, reason="start_fenced")
                raise SessionError("start completed after scheduler stop")
            self.rm_session = rm_session
            self._started, self._stopping = True, False
            self._spawn(self._dispatch_loop())
            self._spawn(self._eligibility_loop())
            self._spawn(self._progress_loop(rm_session.session_token))
            self._spawn(self._publication_loop())
            self._spawn(self._watchdog_loop())
            self._wake.set()
            return self.rm_session

    async def _stop_late_rm(self, session, reason: str) -> None:
        """Best-effort bounded cleanup for an RM reply arriving after a stop."""
        try:
            task = asyncio.create_task(self.rm.stop_session(
                session.session_token, reason=reason,
                idempotency_key=f"late-stop:{self.store.scheduler_id}:{session.generation}"))
            done, _ = await asyncio.wait([task], timeout=self.stop_timeout)
            if done:
                task.result()
            else:
                task.cancel()
                task.add_done_callback(self._consume_task)
        except Exception:
            pass

    def _reset_process_state(self):
        self._pending.clear(); self._active.clear(); self._seen_sequences.clear()
        self._cursor = 0; self._watchdog_deadline = None; self._eligible = False; self._wake.clear()
        self._lease_cursor = None; self._completion_sequence = 0; self._result_pending.clear()

    def _spawn(self, awaitable: Any):
        task = asyncio.create_task(awaitable); self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[Any]):
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError: return
        # Detached shutdown work must still have its exception retrieved, but
        # it cannot initiate another stop after the durable fence is set.
        if self._stopping:
            return
        if error is not None:
            detail = str(error).replace("\n", " ")[:160]
            asyncio.create_task(self._fatal(f"coordinator_failure:{type(error).__name__}:{detail}"))

    async def _fatal(self, reason: str):
        if not self._stopping: await self.stop(reason)

    async def enqueue(self, *args, **kwargs):
        row = self.store.enqueue(*args, **kwargs); self._wake.set(); return row

    async def get(self, request_id: str):
        return self.store.get(request_id)

    async def watch(self, after: int = 0) -> AsyncIterator[Any]:
        cursor = after
        while not self._stopping:
            rows = self.store.events(cursor, limit=64)
            if rows:
                for row in rows:
                    cursor = row["cursor"]; yield row
            else:
                await asyncio.sleep(self.loop_interval)

    async def cancel(self, request_id: str, *, idempotency_key: str | None = None):
        row = self.store.cancel(request_id, idempotency_key=idempotency_key)
        for key in tuple(self._pending):
            if key[0] == request_id: self._pending.pop(key, None)
        await self._drain_cancellations()
        self._wake.set(); return row

    async def stop(self, reason: str = "stopped", *, cancelled: bool = False,
                   idempotency_key: str | None = None):
        if self._stopping:
            if idempotency_key is not None:
                return self.store.stop(reason, cancelled=cancelled, idempotency_key=idempotency_key)
            return self._stop_result
        self._stopping = True
        self._lifecycle_epoch += 1
        deadline = self.clock() + self.stop_timeout
        # Durable stop is first: no cancelled task can create a new attempt.
        try: self._stop_result = self.store.stop(reason, cancelled=cancelled, idempotency_key=idempotency_key)
        except SessionError: pass
        token = self.rm_session.session_token if self.rm_session else None
        if token:
            try:
                remote_stop = asyncio.create_task(self.rm.stop_session(token, reason=reason,
                    idempotency_key=idempotency_key or f"stop:{self.store.scheduler_id}:{self.local_session.generation}"))
                done, pending = await asyncio.wait([remote_stop], timeout=max(0, deadline - self.clock()))
                if done:
                    remote_stop.result()  # retrieve exceptions even on shutdown
                else:
                    remote_stop.cancel()
                    self._tasks.add(remote_stop)
                    remote_stop.add_done_callback(self._task_done)
            except Exception: pass
        current = [task for task in self._tasks if task is not asyncio.current_task()]
        for task in current: task.cancel()
        if current:
            _, pending = await asyncio.wait(current, timeout=max(0, deadline - self.clock()))
            # Cancellation-suppressing work cannot be allowed to delay a
            # terminal durable fence. Consume its eventual outcome detached.
            for task in pending:
                task.add_done_callback(self._consume_task)
        self._tasks = {task for task in self._tasks if not task.done() and task is not asyncio.current_task()}
        self.rm_session = None; self._started = False
        return self._stop_result

    @staticmethod
    def _consume_task(task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            try:
                task.exception()
            except BaseException:
                pass

    async def _capacity_allows(self) -> bool:
        if not self.rm_session: return False
        capacity = await self.rm.get_capacity(self.rm_session.session_token)
        # RM free_slots already accounts for both active and buffered work.
        # The coordinator has no second persistent reservation counter.
        return capacity.free_slots > 0

    async def _eligibility_loop(self):
        while not self._stopping:
            try:
                candidate = await self.evaluator.next_eligible(claim=False)
                self._eligible = bool(candidate and candidate.eligible)
                if self._eligible:
                    self._arm_watchdog()
                    self._wake.set()
            except Exception:
                # Evaluator records request-local errors; an unexpected failure
                # is handled by the task boundary rather than silently skipped.
                raise
            await asyncio.sleep(self.loop_interval)

    async def _dispatch_loop(self):
        while not self._stopping:
            worked = await self._dispatch_one()
            if worked:
                # RM fakes and immediate backpressure paths need not suspend.
                await asyncio.sleep(0)
            else:
                try: await asyncio.wait_for(self._wake.wait(), self.loop_interval)
                except asyncio.TimeoutError: pass
                self._wake.clear()

    async def _dispatch_one(self) -> bool:
        if await self._drain_cancellations(): return True
        if not self.rm_session: return False
        replay = self.store.pending_submissions(1)
        if replay:
            row = replay[0]
            self._active.add((row["request_id"], row["token"]))
            try:
                payload = await asyncio.to_thread(self.result_store.read, row["payload_digest"])
            except Exception as exc:
                await self._fatal(f"submit_payload_unavailable:{type(exc).__name__}")
                return True
            return await self._submit(row["request_id"], row["token"], payload,
                                      DispatchContext(row["context_size"], row["bucket_identity"]))
        if not await self._capacity_allows(): return False
        if self._pending:
            (request_id, token), (payload, context) = next(iter(self._pending.items()))
            if self.store.get(request_id)["status"] in {"cancelled", "error", "done"}:
                self._pending.pop((request_id, token), None); return True
            return await self._submit(request_id, token, payload, context)
        evaluation = await self.evaluator.next_eligible(claim=False)
        if not evaluation or not evaluation.eligible or not evaluation.capability: return False
        self._arm_watchdog()
        token = self.store.claim_evaluated(evaluation.capability)
        if not token: return True
        self._active.add((evaluation.request_id, token))
        row = self.store.get(evaluation.request_id)
        if row["status"] != "running": return True
        try:
            reference = row["evaluated_payload_reference"] or row["payload_reference"]
            decoded = (self.decoder(reference) if inspect.iscoroutinefunction(self.decoder)
                       else await asyncio.to_thread(self.decoder, reference))
            if inspect.isawaitable(decoded): decoded = await decoded
            decoded = self._decode(decoded)
            # Decoder results are persisted by digest before RM sees them.
            digest = await asyncio.to_thread(self.result_store.write, decoded.payload)
            self.store.persist_dispatch_metadata(token, digest, decoded.context.context_size, decoded.context.bucket_identity)
            if self.store.get(row["request_id"])["status"] == "cancelled": return True
            return await self._submit(row["request_id"], token, decoded.payload, decoded.context)
        except StaleCallback: return True
        except Exception as exc:
            failure = getattr(exc, "failure", None)
            if failure is not None:
                self._retry_one(row["request_id"], token, bool(getattr(failure, "retryable", False)))
                if not getattr(failure, "retryable", False):
                    await self.stop(failure.code)
                return True
            try:
                self.store.fail_attempt(row["request_id"], token, self.local_session.token,
                                        self.local_session.generation, "decode_failed")
            except StaleCallback:
                pass
            return True
        finally:
            # A claim belongs to this dispatch attempt even when a callback is
            # stale or an early terminal return wins the race.
            try:
                if self.store.get(row["request_id"])["status"] != "running":
                    self._active.discard((row["request_id"], token))
            except SessionError:
                self._active.discard((row["request_id"], token))

    async def _submit(self, request_id: str, token: str, payload: bytes, context: DispatchContext) -> bool:
        if self._stopping or self.store.get(request_id)["status"] in {"cancelled", "error", "done"}:
            self._pending.pop((request_id, token), None); return True
        try:
            result = await self.rm.submit(self.rm_session.session_token, request_id, token, payload,
                idempotency_key=token, context_size=context.context_size, bucket_identity=context.bucket_identity)
            if result.accepted:
                if self.store.get(request_id)["status"] in {"cancelled", "error", "done"}:
                    await self.rm.cancel_request(self.rm_session.session_token, request_id,
                        idempotency_key=f"late-submit-cancel:{token}")
                    self._active.discard((request_id, token))
                    return True
                self._pending.pop((request_id, token), None); self._active.add((request_id, token))
                self.store.acknowledge_submit(token)
            else:
                self._pending[(request_id, token)] = (payload, context)
            return True
        except Exception as exc:
            failure = getattr(exc, "failure", None)
            # Unknown submit exceptions have uncertain delivery; preserve their
            # durable intent for idempotent transport replay, not provider retry.
            if failure is not None:
                self._retry_one(request_id, token, bool(getattr(failure, "retryable", False)))
                if not failure.retryable:
                    await self.stop(failure.code)
            return True

    async def _drain_cancellations(self) -> bool:
        """Send durable cancels before any newer submit intent."""
        rows = self.store.pending_outbox("cancel", 1)
        if not rows or not self.rm_session:
            return False
        row = rows[0]
        try:
            await self.rm.cancel_request(self.rm_session.session_token, row["request_id"],
                                         idempotency_key=row["idempotency_key"])
            self.store.acknowledge_outbox(row["idempotency_key"])
        except Exception:
            pass
        return True

    @staticmethod
    def _decode(value: Any) -> DecodedPayload:
        if isinstance(value, DecodedPayload): result = value
        elif isinstance(value, bytes): result = DecodedPayload(value)
        elif isinstance(value, tuple) and len(value) == 2:
            context = value[1] if isinstance(value[1], DispatchContext) else DispatchContext(**value[1])
            result = DecodedPayload(value[0], context)
        else: raise ValueError("decoder must return bytes or DecodedPayload")
        if not isinstance(result.payload, bytes) or not result.payload: raise ValueError("decoded payload must be non-empty bytes")
        return result

    def _retry_one(self, request_id: str, token: str, retryable: bool):
        try: self.store.retry(request_id, token, self.local_session.token, self.local_session.generation, now=time.time(), retryable=retryable)
        except (StaleCallback, SessionError): pass
        self._pending.pop((request_id, token), None); self._active.discard((request_id, token)); self._wake.set()

    async def _progress_loop(self, token: str):
        async for event in self.rm.watch_progress(token, self._cursor):
            await self._handle_event(event)
        if not self._stopping:
            raise RuntimeError("progress_stream_ended")

    async def _handle_event(self, event: ProgressEvent):
        if (not self.rm_session or event.session_token != self.rm_session.session_token
                or event.generation != self.rm_session.generation):
            return
        if event.sequence in self._seen_sequences:
            return
        self._seen_sequences.add(event.sequence)
        self._cursor = max(self._cursor, event.sequence)
        key = (event.request_id, event.attempt) if event.request_id and event.attempt else None
        if event.kind is EventKind.RESPONSE_FINISHED:
            # Only a new completion sequence can reset the watchdog.
            if event.completion_sequence > self._completion_sequence:
                self._completion_sequence = event.completion_sequence
                self._reset_watchdog()
            if key and event.result is not None:
                self._result_pending[key] = event
                await self._persist_result(key)
                self._wake.set()
            elif key:
                self._active.discard(key)
                self._pending.pop(key, None)
        elif event.kind is EventKind.ADMISSION and key:
            try: self.store.mark_on_gpu(event.request_id, event.attempt, self.local_session.token, self.local_session.generation)
            except StaleCallback: pass
        elif event.kind is EventKind.CANCELLED and key:
            self._active.discard(key); self._pending.pop(key, None); self._wake.set()
        elif event.kind is EventKind.FAILURE and key:
            self._active.discard(key); self._pending.pop(key, None)
            try:
                attempt = self.store.attempt(event.request_id, event.attempt)
                if not attempt or not attempt["active"] or self.store.get(event.request_id)["status"] in {"cancelled", "error", "done"}:
                    return
            except SessionError:
                return
            failure = event.failure or Failure("provider_execution_failed", "provider failure", True)
            if failure.retryable: self._retry_one(event.request_id, event.attempt, True)
            else:
                self._retry_one(event.request_id, event.attempt, False)
                await self.stop(failure.code)
        elif event.kind is EventKind.SESSION_INVALIDATED and not self._stopping:
            await self.stop(event.failure.code if event.failure else "session_invalidated")

    async def _publication_loop(self):
        while not self._stopping:
            rows = self.store.pending_outbox("handoff", 1)
            if not rows:
                await asyncio.sleep(self.loop_interval); continue
            row = rows[0]
            try:
                session = self.local_session
                if session is None:
                    return
                worker = asyncio.create_task(
                    asyncio.to_thread(self._publish_guarded, dict(row), session))
                self._publisher_workers.add(worker)
                worker.add_done_callback(self._publisher_worker_done)
                # Do not let cancellation cancel the executor wrapper: the
                # captured session remains its only permitted lifecycle.
                await asyncio.shield(worker)
            except (OSError, IOError):
                await asyncio.sleep(self.loop_interval)

    def _publisher_worker_done(self, task: asyncio.Task[Any]) -> None:
        self._publisher_workers.discard(task)
        self._consume_task(task)

    def _publish_guarded(self, row: dict[str, Any], session: Session) -> None:
        """Publish under an independent SQLite transaction/fence.

        The scheduler's connection is never passed to a worker.  Cancellation
        and explicit stop serialize with this transaction: whichever commits
        first deterministically wins, and a terminal request cannot receive a
        receipt afterwards.
        """
        db = sqlite3.connect(self.store.path, timeout=30, isolation_level=None)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout=30000")
            db.execute("BEGIN IMMEDIATE")
            owner = db.execute("SELECT value FROM meta WHERE key='session_token'").fetchone()
            generation = db.execute("SELECT value FROM meta WHERE key='generation'").fetchone()
            pending = db.execute(
                "SELECT o.cancelled,o.acknowledged,r.status,h.acknowledged FROM outbox o "
                "JOIN requests r ON r.request_id=o.request_id JOIN handoffs h ON h.request_id=o.request_id "
                "WHERE o.kind='handoff' AND o.idempotency_key=? AND o.token=?",
                (row["idempotency_key"], row["token"]),
            ).fetchone()
            if (not owner or not generation
                    or owner[0] != session.token or int(generation[0]) != session.generation
                    or not pending or pending[0] or pending[1] or pending[2] in ("cancelled", "error", "done") or pending[3]):
                db.execute("ROLLBACK")
                return
            # For a shared database, retain this queue transaction and insert
            # the receipt through it.  For a distinct receipt database retain
            # the queue fence while its independent, idempotent publish commits.
            # In both cases stop/cancel cannot win after receipt publication.
            if os.path.samefile(self.store.path, self._publisher_state_path):
                self.publisher.publish_on_connection(db, row["payload_reference"], row["idempotency_key"])
            else:
                publisher = LocalPublisher(self.result_store, self._publisher_state_path)
                try:
                    publisher.publish(row["request_id"], row["token"], row["payload_reference"], row["idempotency_key"])
                finally:
                    publisher.close()
            db.execute("UPDATE handoffs SET acknowledged=1 WHERE request_id=?", (row["request_id"],))
            db.execute("UPDATE outbox SET acknowledged=1 WHERE kind='handoff' AND idempotency_key=?", (row["idempotency_key"],))
            db.execute("UPDATE attempts SET active=0 WHERE request_id=?", (row["request_id"],))
            db.execute("UPDATE requests SET status='done',done_at=? WHERE request_id=? AND status NOT IN ('cancelled','error')",
                        (time.time(), row["request_id"]))
            db.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='eligibility_version'")
            db.execute("DELETE FROM evaluated_capabilities")
            cursor = db.execute("INSERT INTO events(request_id,kind,data,created) VALUES(?,?,?,?) RETURNING cursor",
                                (row["request_id"], "status", "{}", time.time())).fetchone()[0]
            request = db.execute("SELECT * FROM requests WHERE request_id=?", (row["request_id"],)).fetchone()
            attempt = db.execute("SELECT gpu_ms,gpu_complete FROM attempts WHERE request_id=? ORDER BY COALESCE(finished,started) DESC LIMIT 1", (row["request_id"],)).fetchone()
            snapshot = {"requestId": request["request_id"], "schedulerId": request["scheduler_id"], "modelId": request["model_id"], "status": request["status"], "sequence": cursor, "runningAt": request["running_at"], "doneAt": request["done_at"], "runningToDoneMs": None if request["running_at"] is None else max(0, round((request["done_at"] - request["running_at"]) * 1000)), "timeOnGpuMs": None if attempt is None else attempt["gpu_ms"], "gpuTimingComplete": bool(attempt and attempt["gpu_complete"]), "errorCode": request["error_code"], "resultReference": row["payload_reference"], "cancellation": bool(request["cancellation"])}
            db.execute("UPDATE events SET data=? WHERE cursor=?", (json.dumps({"new": "done", "snapshot": snapshot}, sort_keys=True), cursor))
            db.execute("COMMIT")
        except Exception:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def _arm_watchdog(self):
        if self._watchdog_deadline is None: self._watchdog_deadline = self.clock() + self.watchdog_seconds

    def _reset_watchdog(self): self._watchdog_deadline = self.clock() + self.watchdog_seconds

    async def _watchdog_loop(self):
        while not self._stopping:
            await asyncio.sleep(self.loop_interval)
            # Renew decoder and uncertain-submit attempts too. Restricting
            # renewal to acknowledged submits permits the 30s lease to expire
            # before the 60s completion watchdog.
            attempts = self.store.active_attempts(16, after_token=self._lease_cursor)
            if not attempts:
                self._lease_cursor = None
            for attempt in attempts:
                self._lease_cursor = attempt["token"]
                try:
                    self.store.renew_lease(attempt["request_id"], attempt["token"],
                                           self.local_session.token, self.local_session.generation)
                except StaleCallback:
                    pass
            if self._watchdog_deadline is not None and not self._active and not self._pending and not self._eligible:
                self._watchdog_deadline = None
            if self._watchdog_deadline is not None and self.clock() >= self._watchdog_deadline:
                await self.stop("idle_timeout")
                return

    async def _persist_result(self, key: tuple[str, str]) -> None:
        """Retry durable result handoff without rerunning provider execution."""
        while not self._stopping:
            event = self._result_pending.get(key)
            if event is None:
                return
            try:
                self.store.finish_attempt(event.request_id, event.attempt, self.local_session.token,
                                         self.local_session.generation, gpu_ms=event.time_on_gpu_ms,
                                         gpu_timing_complete=event.gpu_timing_complete)
                reference = await asyncio.to_thread(self.result_store.write, event.result)
                self.store.stage_handoff(event.request_id, event.attempt, self.local_session.token,
                                         self.local_session.generation, reference,
                                         f"handoff:{event.request_id}:{event.attempt}")
            except StaleCallback:
                self._result_pending.pop(key, None)
                return
            except (OSError, IOError, sqlite3.Error):
                await asyncio.sleep(self.loop_interval)
                continue
            self._result_pending.pop(key, None)
            self._active.discard(key)
            self._pending.pop(key, None)
            return

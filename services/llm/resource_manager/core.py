"""Single-GPU, fenced, bounded asynchronous ResourceManager core."""

from __future__ import annotations

import asyncio
import hashlib
import math
import secrets
import time
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator

from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile
from services.llm.resource_manager.protocol import (
    Capacity, EventKind, Failure, ProgressEvent, Provider, ProviderResponse,
    ResourceManagerClient, SessionInfo, Submission,
)
from services.llm.resource_manager.state import ResourceManagerState
try:
    from tools.compatibility.debug_trace import record as trace
except ImportError:
    def trace(*args, **kwargs):
        return None


class ResourceManagerError(RuntimeError):
    def __init__(self, failure: Failure):
        super().__init__(failure.message)
        self.failure = failure


@dataclass
class _Work:
    session: SessionInfo
    request_id: str
    attempt: str
    identity: str
    payload: bytes
    task: asyncio.Task[None] | None = None
    cancelled: bool = False
    slot_started: float | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.session.session_token, self.request_id, self.attempt


class ResourceManager(ResourceManagerClient):
    """Provider injection is only for this in-process precursor boundary."""

    def __init__(self, *, cleanup_timeout: float = 10.0, stop_timeout: float = 10.0,
                 load_timeout: float = 60.0, max_events: int = 1024, max_sessions: int = 16):
        for value, name in ((cleanup_timeout, "cleanup_timeout"), (stop_timeout, "stop_timeout"),
                            (load_timeout, "load_timeout")):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        self.cleanup_timeout, self.stop_timeout, self.load_timeout, self.max_events, self.max_sessions = cleanup_timeout, stop_timeout, load_timeout, max_events, max_sessions
        self._lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._session: SessionInfo | None = None
        self._profile: CapacityProfile | None = None
        self._provider: Provider | None = None
        self._available = True
        self._buffer: deque[_Work] = deque()
        self._active: dict[tuple[str, str, str], _Work] = {}
        self._records: dict[str, dict[str, tuple[str, Submission]]] = {}
        self._request_attempts: dict[str, dict[tuple[str, str], str]] = {}
        self._pair_records: dict[str, dict[tuple[str, str], Submission]] = {}
        # A request may have more than one same-attempt validation in flight
        # (for example, an idempotent retry racing the first submit).  Keep a
        # reference count rather than one shared marker: one validator must not
        # erase the cancellation fence owned by another.
        self._validating: dict[str, dict[str, tuple[str, str, int]]] = {}
        self._start_records: dict[str, tuple[tuple[object, ...], SessionInfo]] = {}
        self._stop_records: dict[str, tuple[tuple[object, ...], str]] = {}
        self._cancel_records: dict[str, tuple[tuple[object, ...], bool]] = {}
        self._events: dict[str, deque[ProgressEvent]] = {}
        self._event_number: dict[str, int] = {}
        self._completion_number: dict[str, int] = {}
        self._waiters: dict[str, list[asyncio.Future[None]]] = {}
        self._terminal: set[str] = set()
        self._session_order: deque[str] = deque()
        self._expired_sessions: set[str] = set()
        self._generation = 0
        self._abandoned: set[asyncio.Task[object]] = set()
        # Lifecycle calls (not just executions) remain residency fences when a
        # timeout detaches them from their caller.  In particular, unload must
        # never race a timed-out load.
        self._lifecycle_tasks: set[asyncio.Task[object]] = set()
        self._cancelled_requests: set[tuple[str, str]] = set()
        self._phase = "startup"
        self._permanently_closed = False
        self._shutdown_task: asyncio.Task[None] | None = None

    def fence_shutdown(self, *, reason: str = "runtime_stopped") -> None:
        """Synchronously close admission and fence the current generation.

        Runtime shutdown calls this before its first await.  Keeping this as a
        public, synchronous operation prevents a callback which is already
        running from publishing into a generation which shutdown has retired.
        """
        if self._permanently_closed:
            return
        self._permanently_closed = True
        self._available = False
        self._generation += 1
        session = self._session
        self._session = None
        self._buffer.clear()
        self._phase = "unloading" if session or self._provider else "startup"
        for work in self._active.values():
            work.cancelled = True
        if session:
            self._invalidate_locked(session, Failure(reason, reason, False))

    def snapshot(self) -> ResourceManagerState:
        """Return the authoritative synchronous lifecycle observation.

        Lifecycle mutations happen synchronously between awaits on the event
        loop, so this read is atomic without exposing the manager's locks or
        tasks.
        """
        return ResourceManagerState(self._phase, self._available,
                                    self._session is not None, self._generation,
                                    self._permanently_closed)

    async def shutdown(self, *, reason: str = "runtime_stopped") -> None:
        """Fence admission and clean the current provider, without private callers.

        Bootstrap uses this when the process is stopping or its daemon has died.
        The synchronous fence is deliberately published before any provider
        cleanup await; a cleanup failure therefore remains visible forever.
        """
        self.fence_shutdown(reason=reason)
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown_cleanup())
        await asyncio.shield(self._shutdown_task)

    async def _shutdown_cleanup(self) -> None:
        async with self._lifecycle_lock:
            async with self._lock:
                provider = self._provider
            succeeded = False
            try:
                if provider:
                    await self._cleanup(provider, timeout=self.stop_timeout)
                succeeded = True
            finally:
                async with self._lock:
                    if succeeded and self._phase == "unloading":
                        self._provider = None
                        self._profile = None
                        self._phase = "startup"

    async def probe_active_dependency(self, expected_profile: CapacityProfile | None = None
                                      ) -> tuple[ResourceManagerState, int, bool]:
        """Probe the active adapter without exposing provider internals.

        The result is valid only when the same stable generation remains
        active after ``ready`` returns; lifecycle changes fail closed.
        """
        async with self._lock:
            state = self.snapshot()
            provider, profile, revision = self._provider, self._profile, self._generation
            if (state.phase != "stable" or not state.available or provider is None
                    or profile is None or (expected_profile is not None and profile != expected_profile)):
                return state, revision, False
        try:
            await provider.ready()
        except Exception:
            return self.snapshot(), revision, False
        async with self._lock:
            final = self.snapshot()
            return final, revision, bool(
                final == state and self._generation == revision and
                self._provider is provider and self._profile == profile)

    @staticmethod
    def _key(value: str, name: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be non-empty")

    def _abandon(self, task: asyncio.Task[object]) -> None:
        self._abandoned.add(task)
        def consume(done: asyncio.Task[object]) -> None:
            self._abandoned.discard(done)
            try: done.exception()
            except BaseException: pass
        task.add_done_callback(consume)

    async def _bounded(self, operation, deadline: float) -> object:
        task = asyncio.create_task(operation)
        self._lifecycle_tasks.add(task)
        task.add_done_callback(self._lifecycle_tasks.discard)
        remaining = max(0.0, deadline - time.monotonic())
        try:
            done, pending = await asyncio.wait((task,), timeout=remaining)
        except asyncio.CancelledError:
            self._abandon(task)
            raise
        if pending:
            self._abandon(task)
            raise self._error("lifecycle_timeout", "provider lifecycle operation timed out")
        return task.result()

    @staticmethod
    def _error(code: str, message: str, retryable: bool = False) -> ResourceManagerError:
        return ResourceManagerError(Failure(code, message, retryable))

    @staticmethod
    def _identity(request_id: str, attempt: str, payload: bytes, context_size, bucket_identity) -> str:
        return hashlib.sha256(payload + repr((request_id, attempt, context_size, bucket_identity)).encode()).hexdigest()

    def _validate_profile(self, model_id: ModelId, profile: CapacityProfile) -> None:
        if profile.model_id != ModelId(model_id):
            raise self._error("invalid_profile", "profile model does not match requested model")

    async def start_session(self, scheduler_id: str, model_id: ModelId,
                            profile: CapacityProfile, provider: Provider, *,
                            idempotency_key: str) -> SessionInfo:
        trace("resource_manager", "load", "enter", model=ModelId(model_id).value)
        self._key(idempotency_key, "idempotency_key")
        args = (scheduler_id, ModelId(model_id), profile, provider)
        # A known start replay must be rejected promptly once its session has
        # been retired.  In particular it must not queue behind a replacement's
        # cleanup while holding no authority to revive the old session.
        async with self._lock:
            previous = self._start_records.get(idempotency_key)
            if previous and previous[0] == args and self._session != previous[1]:
                raise self._error("scheduler_superseded", "start replay belongs to an old session")
        async with self._lifecycle_lock:
            if self._permanently_closed or not self._available:
                raise self._error("resource_manager_unavailable", "cleanup has failed")
            previous = self._start_records.get(idempotency_key)
            if previous:
                if previous[0] != args:
                    raise self._error("idempotency_conflict", "start key arguments differ")
                if self._session == previous[1]:
                    return previous[1]
                raise self._error("scheduler_superseded", "start replay belongs to an old session")
            self._validate_profile(model_id, profile)
            async with self._lock:
                if self._permanently_closed or not self._available:
                    raise self._error("resource_manager_unavailable", "cleanup has failed")
                old = self._session
                old_provider = self._provider
                if old:
                    # This is the synchronous residency fence, before cleanup awaits.
                    self._session = None
                    self._available = False
                    self._phase = "unloading"
                    self._buffer.clear()
                    for work in self._active.values():
                        work.cancelled = True
                    self._invalidate_locked(old, Failure("scheduler_superseded", "session was replaced", False))
                generation = max(self._generation + 1, old.generation + 1 if old else 1)
            if old and old_provider:
                await self._cleanup(old_provider, timeout=self.cleanup_timeout)
            async with self._lock:
                if self._permanently_closed or not self._available:
                    raise self._error("resource_manager_unavailable", "cleanup has failed")
                info = SessionInfo(scheduler_id, secrets.token_urlsafe(24), ModelId(model_id), generation)
                self._session, self._profile, self._provider = info, profile, provider
                self._available = False  # remains closed through validate/load/ready
                self._phase = "loading"
            deadline = time.monotonic() + self.load_timeout
            lifecycle_phase = "validate"
            try:
                await self._bounded(provider.validate(profile), deadline)
                lifecycle_phase = "load"
                await self._bounded(provider.load(profile), deadline)
                lifecycle_phase = "ready"
                await self._bounded(provider.ready(), deadline)
            except BaseException as exc:
                async with self._lock:
                    if self._session == info: self._available = False
                    if self._session == info:
                        self._session = self._profile = self._provider = None
                        # Keep the lifecycle fenced until cleanup has proven
                        # that a timed-out/failed load has released residency.
                        self._phase = "unloading"
                        self._terminal.add(info.session_token)
                # A failed lifecycle may already own a loaded daemon model.  The
                # same bounded cleanup fence used for replacement is mandatory
                # before another session can be admitted.
                # A synchronous shutdown fence owns cleanup of a load which
                # was still in flight.  Do not race that shared cleanup task.
                cleanup_error = None
                if not (self._permanently_closed and self._session != info):
                    try:
                        await self._cleanup(provider, timeout=self.cleanup_timeout)
                    except BaseException as cleanup_exc:
                        cleanup_error = cleanup_exc
                if isinstance(exc, ResourceManagerError):
                    if exc.failure.code == "lifecycle_timeout":
                        exc.lifecycle_phase = lifecycle_phase
                        exc.lifecycle_subreason = "timeout"
                else:
                    phase = getattr(exc, "lifecycle_phase", None)
                    subreason = getattr(exc, "lifecycle_subreason", None)
                    error = self._error("model_load_failed", "provider model load failed")
                    error.lifecycle_phase = phase
                    error.lifecycle_subreason = subreason
                    exc = error
                if cleanup_error is not None:
                    # Cleanup is authoritative for admission, but the startup
                    # classification must remain inspectable rather than being
                    # replaced by a generic cleanup exception.
                    setattr(cleanup_error, "startup_lifecycle_phase",
                            getattr(exc, "lifecycle_phase", None))
                    setattr(cleanup_error, "startup_lifecycle_subreason",
                            getattr(exc, "lifecycle_subreason", None))
                    cleanup_phase = getattr(cleanup_error, "lifecycle_phase", "cleanup")
                    cleanup_subreason = getattr(cleanup_error, "lifecycle_subreason", "cleanup_verification")
                    setattr(cleanup_error, "lifecycle_phase", cleanup_phase)
                    setattr(cleanup_error, "lifecycle_subreason", cleanup_subreason)
                    raise BaseExceptionGroup("startup and cleanup lifecycle failed",
                                             [exc, cleanup_error])
                if isinstance(exc, ResourceManagerError):
                    raise exc
                raise exc
            async with self._lock:
                if self._session != info:
                    raise self._error("scheduler_superseded", "session was replaced")
                if self._permanently_closed:
                    raise self._error("resource_manager_unavailable", "resource manager is permanently stopped")
                self._available = True
                self._phase = "stable"
                self._generation = info.generation
                self._records[info.session_token] = {}
                self._request_attempts[info.session_token] = {}
                self._pair_records[info.session_token] = {}
                self._validating[info.session_token] = {}
                self._cancelled_requests = {item for item in self._cancelled_requests
                                            if item[0] != info.session_token}
                self._events[info.session_token] = deque(maxlen=self.max_events)
                self._session_order.append(info.session_token)
                while len(self._session_order) > self.max_sessions:
                    expired = self._session_order.popleft()
                    self._events.pop(expired, None)
                    self._expired_sessions.add(expired)
                self._event_number[info.session_token] = 0
                self._completion_number[info.session_token] = 0
                self._start_records[idempotency_key] = (args, info)
                trace("resource_manager", "ready", "success", model=info.model_id.value)
                return info

    async def _cleanup(self, provider: Provider, *, timeout: float | None = None) -> None:
        trace("resource_manager", "cleanup", "enter")
        timeout = self.cleanup_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        try:
            async with self._lock:
                works = tuple(self._active.values())
                execute_tasks = tuple(work.task for work in works if work.task and not work.task.done())
                lifecycle_tasks = tuple(task for task in self._lifecycle_tasks if not task.done())
            # Cancellation is advisory and may itself hang or suppress task
            # cancellation.  Never await it unboundedly, and never unload
            # while tracked provider execution remains unfinished.
            cancel_tasks = tuple(asyncio.create_task(provider.cancel(work.request_id)) for work in works)
            if cancel_tasks:
                _, cancel_pending = await asyncio.wait(cancel_tasks, timeout=max(0, deadline-time.monotonic()))
                for task in cancel_pending: self._abandon(task)
                for task in cancel_tasks:
                    if task not in cancel_pending: self._abandon(task)
            tracked_tasks = tuple(dict.fromkeys((*execute_tasks, *lifecycle_tasks)))
            if tracked_tasks:
                _, execute_pending = await asyncio.wait(tracked_tasks, timeout=max(0, deadline-time.monotonic()))
                if execute_pending:
                    for task in execute_pending: self._abandon(task)
                    raise self._error("cleanup_timeout", "provider execution did not finish")
            await self._bounded(provider.unload(), deadline)
            if not await self._bounded(provider.verify_cleanup(), deadline):
                raise self._error("cleanup_failed", "provider cleanup verification failed")
            async with self._lock:
                self._active.clear()
                self._available = not self._permanently_closed
                if self._session is None and not self._permanently_closed:
                    self._phase = "startup"
        except ResourceManagerError as exc:
            if exc.failure.code in {"cleanup_timeout", "cleanup_failed"}:
                exc.lifecycle_phase = "cleanup"
                exc.lifecycle_subreason = "timeout" if exc.failure.code == "cleanup_timeout" else "cleanup_verification"
            async with self._lock:
                self._available = False
                self._phase = "cleanup_failed"
            trace("resource_manager", "cleanup", "failure", failure_code=exc.failure.code)
            raise
        except asyncio.CancelledError:
            async with self._lock:
                self._available = False
                self._phase = "cleanup_failed"
            raise
        except Exception as exc:
            async with self._lock:
                self._available = False
                self._phase = "cleanup_failed"
            error = self._error("cleanup_failed", "provider cleanup failed")
            error.lifecycle_phase = "cleanup"
            error.lifecycle_subreason = "timeout" if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) else "cleanup_verification"
            trace("resource_manager", "cleanup", "failure", failure_code="cleanup_failed")
            raise error from exc

    def _emit_locked(self, session: SessionInfo, kind: EventKind, *, request_id=None,
                     attempt=None, result=None, failure=None, timing=None, complete=False,
                     observation=None) -> None:
        token = session.session_token
        self._event_number[token] = self._event_number.get(token, 0) + 1
        if kind is EventKind.RESPONSE_FINISHED:
            self._completion_number[token] = self._completion_number.get(token, 0) + 1
        event = ProgressEvent(self._event_number[token], self._completion_number.get(token, 0), kind,
                              request_id, attempt, token, session.generation, result, failure,
                               timing, complete, observation)
        self._events.setdefault(token, deque(maxlen=self.max_events)).append(event)
        for waiter in self._waiters.pop(token, []):
            if not waiter.done(): waiter.set_result(None)

    def _invalidate_locked(self, session: SessionInfo, failure: Failure) -> None:
        """Invalidate a session at most once, including during loading."""
        if session.session_token in self._terminal:
            return
        self._terminal.add(session.session_token)
        self._emit_locked(session, EventKind.SESSION_INVALIDATED, failure=failure)

    def _check_session_locked(self, token: str) -> tuple[SessionInfo, CapacityProfile, Provider]:
        if not self._session or self._session.session_token != token:
            raise self._error("scheduler_superseded", "unknown or stale session")
        if not self._profile or not self._provider:
            raise self._error("resource_manager_unavailable", "session is not ready")
        return self._session, self._profile, self._provider

    def _release_validation_locked(self, session_token: str, request_id: str,
                                   attempt: str, identity: str) -> None:
        marker = self._validating.get(session_token, {}).get(request_id)
        if marker and marker[:2] == (attempt, identity):
            if marker[2] > 1:
                self._validating[session_token][request_id] = (*marker[:2], marker[2] - 1)
            else:
                self._validating[session_token].pop(request_id, None)

    async def submit(self, session_token: str, request_id: str, attempt: str, payload: bytes,
                     *, idempotency_key: str, context_size=None, bucket_identity=None) -> Submission:
        self._key(idempotency_key, "idempotency_key")
        if not isinstance(payload, bytes) or not payload:
            raise self._error("invalid_input", "payload must be non-empty bytes")
        identity = self._identity(request_id, attempt, payload, context_size, bucket_identity)
        async with self._lock:
            session, profile, provider = self._check_session_locked(session_token)
            if (session_token, request_id) in self._cancelled_requests:
                raise self._error("request_cancelled", "request was cancelled")
            known = self._records[session_token].get(idempotency_key)
            if known:
                if known[0] != identity: raise self._error("idempotency_conflict", "payload differs")
                return known[1]
            pair = (request_id, attempt)
            for work in (*self._active.values(), *self._buffer):
                if work.request_id == request_id and work.attempt != attempt:
                    raise self._error("request_in_flight", "another attempt for request is active")
            validating = self._validating[session_token].get(request_id)
            if validating and validating[0] != attempt:
                raise self._error("request_in_flight", "another attempt is being validated")
            if validating and validating[1] != identity:
                # Validators for one request/attempt share a reference-counted
                # reservation.  A different immutable input must not replace
                # that marker: doing so strands the first validator's release.
                raise self._error("idempotency_conflict", "request and attempt identity differs")
            old_identity = self._request_attempts[session_token].get(pair)
            if old_identity and old_identity != identity:
                raise self._error("idempotency_conflict", "request and attempt identity differs")
            if old_identity == identity:
                replay = self._pair_records[session_token][pair]
                self._records[session_token][idempotency_key] = (identity, replay)
                return replay
            if not self._available:
                raise self._error("resource_manager_unavailable", "admission is closed")
            # Reserve validation only after all synchronous replay, conflict,
            # and availability exits.  Replays must not leave a phantom
            # reservation that changes a later cancellation outcome.
            count = validating[2] + 1 if validating else 1
            self._validating[session_token][request_id] = (attempt, identity, count)
        reservation_released = False
        def release() -> None:
            nonlocal reservation_released
            if not reservation_released:
                self._release_validation_locked(session_token, request_id, attempt, identity)
                reservation_released = True
        try:
            if not profile.accepts_request(context_size, bucket_identity):
                raise self._error("invalid_input", "request does not match exact capacity profile")
            await provider.validate_input(payload, context_size=context_size, bucket_identity=bucket_identity)
        except ResourceManagerError:
            async with self._lock:
                release()
            trace("resource_manager", "admission", "failure")
            raise
        except Exception as exc:
            async with self._lock:
                release()
            trace("resource_manager", "admission", "failure")
            raise self._error("invalid_input", str(exc)) from exc
        except BaseException:
            async with self._lock:
                release()
            trace("resource_manager", "admission", "failure")
            raise
        try:
            async with self._lock:
                # Fence and replay are both repeated after the await to close duplicate races.
                session, profile, provider = self._check_session_locked(session_token)
                # Keep this reservation until this admission lock is held.  A
                # cancel between validation and admission therefore sees an
                # owned request and fences it rather than returning a false
                # no-op which would let this submit execute.
                release()
                if (session_token, request_id) in self._cancelled_requests:
                    self._emit_locked(session, EventKind.CANCELLED, request_id=request_id, attempt=attempt)
                    raise self._error("request_cancelled", "request was cancelled")
                known = self._records[session_token].get(idempotency_key)
                if known:
                    if known[0] != identity: raise self._error("idempotency_conflict", "payload differs")
                    return known[1]
                pair = (request_id, attempt)
                for work in (*self._active.values(), *self._buffer):
                    if work.request_id == request_id and work.attempt != attempt:
                        raise self._error("request_in_flight", "another attempt for request is active")
                old_identity = self._request_attempts[session_token].get(pair)
                if old_identity and old_identity != identity:
                    raise self._error("idempotency_conflict", "request and attempt identity differs")
                if old_identity == identity:
                    replay = self._pair_records[session_token][pair]
                    self._records[session_token][idempotency_key] = (identity, replay)
                    return replay
                p = profile.optimal_parallelism
                if len(self._active) + len(self._buffer) >= 2 * p:
                    return Submission(False, request_id, attempt, session_token, session.generation, True)
                result = Submission(True, request_id, attempt, session_token, session.generation)
                self._records[session_token][idempotency_key] = (identity, result)
                self._request_attempts[session_token][pair] = identity
                self._pair_records[session_token][pair] = result
                work = _Work(session, request_id, attempt, identity, bytes(payload))
                if len(self._active) < p:
                    self._active[work.key] = work
                    self._emit_locked(session, EventKind.ADMISSION, request_id=request_id, attempt=attempt)
                    work.slot_started = time.monotonic()
                    work.task = asyncio.create_task(self._run(work, provider))
                else:
                    self._buffer.append(work)
                    self._emit_locked(session, EventKind.BUFFERED, request_id=request_id, attempt=attempt)
                return result
        finally:
            async with self._lock:
                release()

    @staticmethod
    def _provider_failure(exc: BaseException) -> Failure:
        """Keep a producer's typed failure through async wrapper exceptions.

        ``execute`` failures normally arrive directly, but task/gather cleanup
        can retain the producer exception as a cause or an exception-group
        member.  Looking only at the outer exception discarded the closed
        provider category at that boundary.
        """
        pending = [exc]
        seen: set[int] = set()
        while pending and len(seen) < 16:
            current = pending.pop(0)
            if id(current) in seen:
                continue
            seen.add(id(current))
            if isinstance(current, ResourceManagerError):
                return current.failure
            failure = getattr(current, "failure", None)
            if isinstance(failure, Failure):
                return failure
            if isinstance(current, BaseExceptionGroup):
                pending.extend(item for item in current.exceptions
                               if isinstance(item, BaseException))
            cause = current.__cause__
            context = current.__context__
            if isinstance(cause, BaseException):
                pending.append(cause)
            if isinstance(context, BaseException):
                pending.append(context)
        # Exception text can contain a provider body, prompt, or local path.
        # The event remains useful through its stable category without making
        # that text a transport channel.
        return Failure("provider_execution_failed", "provider execution failed", True)

    @staticmethod
    def _valid_provider_response(response: object) -> bool:
        """Validate all response fields before publishing provider evidence."""
        if not isinstance(response, ProviderResponse) or not isinstance(response.result, bytes) or not response.result:
            return False
        if type(response.gpu_timing_complete) is not bool:
            return False
        if response.gpu_timing_complete:
            return (type(response.time_on_gpu_ms) is int and response.time_on_gpu_ms >= 0)
        return response.time_on_gpu_ms is None

    async def _run(self, work: _Work, provider: Provider) -> None:
        response: ProviderResponse | None = None
        failure: Failure | None = None
        cancelled = False
        try:
            response = await provider.execute(work.request_id, work.payload)
            if not self._valid_provider_response(response):
                failure = Failure("malformed_provider_response", "provider returned invalid result", False)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException as exc:
            failure = self._provider_failure(exc)
        finally:
            async with self._lock:
                self._active.pop(work.key, None)  # exact session/request/attempt key
                if cancelled:
                    self._emit_locked(work.session, EventKind.CANCELLED, request_id=work.request_id, attempt=work.attempt)
                elif failure:
                    if isinstance(response, ProviderResponse):
                        response_timing = (response.time_on_gpu_ms
                                           if response.gpu_timing_complete
                                           and isinstance(response.time_on_gpu_ms, int)
                                           and not isinstance(response.time_on_gpu_ms, bool)
                                           and response.time_on_gpu_ms >= 0 else None)
                        self._emit_locked(work.session, EventKind.RESPONSE_FINISHED,
                                          request_id=work.request_id, attempt=work.attempt,
                                          timing=response_timing, complete=response_timing is not None)
                    self._emit_locked(work.session, EventKind.FAILURE, request_id=work.request_id,
                                      attempt=work.attempt, failure=failure)
                else:
                    current = self._session == work.session and not work.cancelled
                    # Incomplete provider timing is explicitly null, never inferred.
                    timing = (response.time_on_gpu_ms if response and response.gpu_timing_complete
                              and isinstance(response.time_on_gpu_ms, int)
                              and not isinstance(response.time_on_gpu_ms, bool)
                              and response.time_on_gpu_ms >= 0 else None)
                    self._emit_locked(work.session, EventKind.RESPONSE_FINISHED, request_id=work.request_id,
                                      attempt=work.attempt, result=response.result if current else None,
                                      timing=timing, complete=timing is not None,
                                      observation=response.observation if response else None)
                if self._session == work.session and self._profile and self._available:
                    while self._buffer and len(self._active) < self._profile.optimal_parallelism:
                        nxt = self._buffer.popleft()
                        if nxt.cancelled: continue
                        self._active[nxt.key] = nxt
                        self._emit_locked(nxt.session, EventKind.ADMISSION, request_id=nxt.request_id, attempt=nxt.attempt)
                        nxt.slot_started = time.monotonic()
                        nxt.task = asyncio.create_task(self._run(nxt, provider))
                if failure is not None or cancelled:
                    trace("resource_manager", "execute", "failure")

    async def cancel_request(self, session_token: str, request_id: str, *, idempotency_key: str) -> bool:
        self._key(idempotency_key, "idempotency_key")
        args = (session_token, request_id)
        async with self._lock:
            old = self._cancel_records.get(idempotency_key)
            if old:
                if old[0] != args: raise self._error("idempotency_conflict", "cancel arguments differ")
                self._check_session_locked(session_token)
                return old[1]
            session, _, provider = self._check_session_locked(session_token)
            validating = self._validating[session_token].get(request_id)
            active = tuple(work for work in self._active.values()
                           if work.request_id == request_id)
            buffered = tuple(work for work in self._buffer
                             if work.request_id == request_id)
            # Validation, buffering, and execution can overlap while duplicate
            # submissions race.  Cancellation is a request fence, not a phase
            # shortcut: fence every ownership record before any advisory call.
            if validating:
                self._cancelled_requests.add((session_token, request_id))
            for work in buffered:
                self._buffer.remove(work)
                work.cancelled = True
            for work in active:
                work.cancelled = True
            if validating or buffered or active:
                # An active execution still emits its finished response (with
                # no result) as watchdog evidence.  Do not terminalize it
                # early; validation/buffer-only work has no such completion.
                if not active:
                    self._emit_locked(session, EventKind.CANCELLED, request_id=request_id,
                                      attempt=buffered[0].attempt if buffered else None)
                self._cancel_records[idempotency_key] = (args, True)
                should_cancel = bool(active)
            else:
                self._cancel_records[idempotency_key] = (args, False)
                should_cancel = False
            cancelled = validating or buffered or active
        if not cancelled:
            return False
        if not should_cancel:
            return True
        try:
            await self._bounded(provider.cancel(request_id), time.monotonic() + self.cleanup_timeout)
        except Exception:
            pass
        return True

    async def stop_session(self, session_token: str, *, reason: str = "stopped",
                           idempotency_key: str) -> None:
        self._key(idempotency_key, "idempotency_key")
        args = (session_token, reason)
        async with self._lifecycle_lock:
            old = self._stop_records.get(idempotency_key)
            if old:
                if old[0] != args: raise self._error("idempotency_conflict", "stop arguments differ")
                return
            async with self._lock:
                if not self._session or self._session.session_token != session_token:
                    raise self._error("scheduler_superseded", "unknown or stale session")
                session, provider = self._session, self._provider
                self._session = None; self._available = False; self._buffer.clear()
                self._phase = "unloading"
                for work in self._active.values(): work.cancelled = True
                self._invalidate_locked(session, Failure(reason, reason, False))
            if provider:
                try: await self._cleanup(provider, timeout=self.stop_timeout)
                except Exception: pass
            async with self._lock:
                self._provider = None; self._profile = None
                if self._available:
                    self._phase = "startup"
                self._stop_records[idempotency_key] = (args, session_token)

    async def get_capacity(self, session_token: str) -> Capacity:
        async with self._lock:
            session, profile, _ = self._check_session_locked(session_token)
            p = profile.optimal_parallelism
            free = 2 * p - len(self._active) - len(self._buffer) if self._available else 0
            return Capacity(profile, p, p, free)

    async def watch_progress(self, session_token: str, after_sequence: int = 0) -> AsyncIterator[ProgressEvent]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:
            raise self._error("invalid_cursor", "cursor must be a nonnegative integer")
        while True:
            async with self._lock:
                events = self._events.get(session_token)
                if events is None:
                    if session_token in self._expired_sessions:
                        raise self._error("cursor_expired", "session history is no longer retained")
                    raise self._error("scheduler_superseded", "unknown session")
                if after_sequence > self._event_number.get(session_token, 0):
                    raise self._error("invalid_cursor", "cursor is in the future")
                if events and after_sequence < events[0].sequence - 1:
                    raise self._error("cursor_expired", "progress cursor is no longer retained")
                pending = [event for event in events if event.sequence > after_sequence]
                if not pending:
                    if session_token in self._terminal: return
                    waiter = asyncio.get_running_loop().create_future()
                    self._waiters.setdefault(session_token, []).append(waiter)
            if not pending:
                try:
                    await waiter
                finally:
                    async with self._lock:
                        waiters = self._waiters.get(session_token, [])
                        if waiter in waiters: waiters.remove(waiter)
                        if not waiters: self._waiters.pop(session_token, None)
                continue
            for event in pending:
                yield event
            after_sequence = pending[-1].sequence
            if session_token in self._terminal and any(e.kind is EventKind.SESSION_INVALIDATED for e in pending):
                return

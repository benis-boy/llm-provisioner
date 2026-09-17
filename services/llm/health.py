"""Read-only, injected health state and bounded aiohttp transport."""
from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from aiohttp import web

DEPENDENCIES = ("sqlite", "gpu", "artifacts", "adapter", "ollama", "profile", "cleanup")
PHASES = ("stable", "startup", "provisioning", "loading", "unloading", "profile_mismatch",
          "profile_unproved", "cleanup_failed")
REASONS = ("ok", "unavailable", "probe_timeout", "probe_failed", "state_missing",
           "state_malformed", "overloaded")
MAX_RESPONSE_BYTES = 16 * 1024


@dataclass(frozen=True)
class DependencyState:
    ok: bool
    reason: str = "ok"


@dataclass(frozen=True)
class HealthSnapshot:
    """Atomic injected state; callers must keep it immutable while returned."""
    phase: str = "startup"
    dependencies: Mapping[str, Any] | None = None


def _safe_state(value: Any) -> DependencyState:
    """Convert untrusted injected state into a fixed, serializable state."""
    if isinstance(value, DependencyState):
        ok, reason = value.ok, value.reason
    elif type(value) is bool:
        ok, reason = value, "ok" if value else "unavailable"
    elif isinstance(value, Mapping) and type(value.get("ok")) is bool:
        ok = value["ok"]
        reason = value.get("reason", "ok" if ok else "unavailable")
    else:
        return DependencyState(False, "state_missing" if value is None else "state_malformed")
    if type(ok) is not bool or not isinstance(reason, str) or reason not in REASONS:
        return DependencyState(False, "state_malformed")
    if ok and reason != "ok":
        return DependencyState(False, "state_malformed")
    if not ok and reason == "ok":
        return DependencyState(False, "unavailable")
    return DependencyState(ok, reason)


class HealthBoundary:
    def __init__(self, snapshot: Callable[[], Any] | Any, *, probes: Mapping[str, Callable[[], Any]] | None = None,
                 timeout_seconds: float = 5.0, max_concurrent: int = 4, cleanup_timeout_seconds: float = .25):
        if (not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or
                not math.isfinite(cleanup_timeout_seconds) or cleanup_timeout_seconds <= 0 or
                type(max_concurrent) is not int or max_concurrent < 1):
            raise ValueError("invalid health bounds")
        self.snapshot, self.probes = snapshot, dict(probes or {})
        self.timeout_seconds, self.cleanup_timeout_seconds = timeout_seconds, cleanup_timeout_seconds
        self._slots = asyncio.BoundedSemaphore(max_concurrent)
        self._workers: dict[asyncio.Task[Any], bool] = {}  # task -> async probe (sync workers must not be cancelled)

    def _snapshot(self) -> tuple[str, dict[str, DependencyState]]:
        raw = self.snapshot() if callable(self.snapshot) else self.snapshot
        if inspect.isawaitable(raw):
            raise TypeError("health snapshots must be synchronous")
        if isinstance(raw, HealthSnapshot): phase, values = raw.phase, raw.dependencies
        elif isinstance(raw, Mapping): phase, values = raw.get("phase"), raw.get("dependencies")
        else: phase, values = getattr(raw, "phase", None), getattr(raw, "dependencies", None)
        if phase not in PHASES or not isinstance(values, Mapping):
            raise ValueError("malformed health snapshot")
        return phase, {name: _safe_state(values.get(name)) for name in DEPENDENCIES}

    async def _probe(self, name: str, fn: Callable[[], Any]) -> DependencyState:
        if self._slots.locked():
            return DependencyState(False, "overloaded")
        await self._slots.acquire()
        is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None))

        async def invoke():
            result = fn() if is_async else await asyncio.to_thread(fn)
            return await result if inspect.isawaitable(result) else result

        task = asyncio.create_task(invoke())
        self._workers[task] = is_async

        def finished(done: asyncio.Task[Any]) -> None:
            self._workers.pop(done, None)
            try: done.exception()
            except asyncio.CancelledError: pass
            self._slots.release()
        task.add_done_callback(finished)
        try:
            return _safe_state(await asyncio.wait_for(asyncio.shield(task), self.timeout_seconds))
        except asyncio.TimeoutError:
            return DependencyState(False, "probe_timeout")
        except asyncio.CancelledError:
            raise
        except Exception:
            return DependencyState(False, "probe_failed")

    async def _states(self, states: dict[str, DependencyState]) -> dict[str, DependencyState]:
        jobs = [self._probe(name, fn) for name, fn in self.probes.items()
                if name in states and callable(fn)]
        results = await asyncio.gather(*jobs, return_exceptions=True)
        for (name, _), result in zip(((n, f) for n, f in self.probes.items() if n in states and callable(f)), results):
            states[name] = result if isinstance(result, DependencyState) else DependencyState(False, "probe_failed")
        return states

    async def dependencies(self) -> dict[str, DependencyState]:
        try:
            _, states = self._snapshot()
        except Exception:
            return {name: DependencyState(False, "state_malformed") for name in DEPENDENCIES}
        return await self._states(states)

    async def readiness(self) -> tuple[bool, list[dict[str, str]], dict[str, DependencyState]]:
        try:
            phase, states = self._snapshot()
        except Exception:
            states = {name: DependencyState(False, "state_malformed") for name in DEPENDENCIES}
            return False, [{"phase": "state_malformed"}], states
        states = await self._states(states)
        reasons: list[dict[str, str]] = []
        if phase != "stable": reasons.append({"phase": phase})
        reasons.extend({"dependency": name, "reason": state.reason}
                       for name, state in states.items() if not state.ok)
        return not reasons, reasons[:8], states

    async def close(self) -> None:
        if not self._workers:
            return
        tasks = tuple(self._workers)
        done, _ = await asyncio.wait(tasks, timeout=self.cleanup_timeout_seconds)
        if len(done) != len(tasks):
            # Do not cancel the wrapper around a sync to_thread call: its slot
            # remains owned until the underlying worker actually returns.
            for task, is_async in tuple(self._workers.items()):
                if is_async:
                    task.cancel()


def _wire_states(states: Mapping[str, DependencyState]) -> dict[str, dict[str, Any]]:
    return {name: {"ok": bool(state.ok), "reason": state.reason if state.reason in REASONS else "state_malformed"}
            for name in DEPENDENCIES for state in [states.get(name, DependencyState(False, "state_missing"))]}


def create_app(boundary: HealthBoundary) -> web.Application:
    if not isinstance(boundary, HealthBoundary): raise TypeError("boundary must be a HealthBoundary")

    def emit(value: dict[str, Any], status: int = 200) -> web.Response:
        raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > MAX_RESPONSE_BYTES:  # fixed-cardinality sanitized values make this defensive only
            raise RuntimeError("health response exceeded bound")
        return web.json_response(value, status=status,
                                 dumps=lambda item: json.dumps(item, separators=(",", ":"), allow_nan=False))

    async def live(_request): return emit({"live": True})
    async def ready(_request):
        ok, reasons, _ = await boundary.readiness()
        return emit({"ready": ok, "reasons": reasons}, 200 if ok else 503)
    async def dependencies(_request): return emit({"dependencies": _wire_states(await boundary.dependencies())})

    app = web.Application(client_max_size=1)
    app.router.add_get("/health/live", live)
    app.router.add_get("/health/ready", ready)
    app.router.add_get("/health/dependencies", dependencies)
    app.on_cleanup.append(lambda _app: boundary.close())
    return app


HealthHttpServer = create_app

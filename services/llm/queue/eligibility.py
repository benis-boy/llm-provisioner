"""Bounded asynchronous eligibility evaluation."""
from __future__ import annotations
import asyncio, copy, inspect, math, time
from dataclasses import dataclass
from typing import Any, Callable
from .contracts import FunctionDescriptor, FunctionRegistry, RequestStatus
from .store import EvaluatedClaim, QueueStore, StaleCallback

@dataclass(frozen=True)
class Evaluation:
    request_id: str
    eligible: bool
    capability: EvaluatedClaim | None = None
    error_code: str | None = None

class EligibilityEvaluator:
    def __init__(self, store: QueueStore, registry: FunctionRegistry | None = None,
                 provider_input_validator: Callable[[str], Any] | None = None,
                 slice_size: int = 32, poll_interval: float = 1.0):
        if not isinstance(slice_size, int) or isinstance(slice_size, bool) or slice_size <= 0:
            raise ValueError("slice_size must be positive")
        if not isinstance(poll_interval, (int, float)) or isinstance(poll_interval, bool) or not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be positive and finite")
        self.store, self.registry = store, registry or FunctionRegistry()
        self.validator, self.slice_size, self.poll_interval = provider_input_validator or (lambda _: True), slice_size, poll_interval
        self._invalidated = 0

    def invalidate(self):
        self._invalidated += 1

    @property
    def invalidation_epoch(self) -> int:
        """Monotonic local invalidation identity for scheduler cache fencing."""
        return self._invalidated

    async def _call(self, descriptor: FunctionDescriptor, deps: dict[str, str]):
        try: function = self.registry.resolve(descriptor.name)
        except LookupError as exc: raise RuntimeError("function_unavailable") from exc
        # Callbacks must not mutate nested accepted descriptor intent.
        args = copy.deepcopy(descriptor.args)
        selected = {key: deps[key] for key in descriptor.dependency_result_ids}
        try:
            value = await asyncio.to_thread(function, args=args, dependency_results=dict(selected))
            return await value if inspect.isawaitable(value) else value
        except RuntimeError as exc:
            if str(exc) in {"function_unavailable", "function_failed", "template_invalid"}: raise
            raise RuntimeError("function_failed") from exc
        except Exception as exc: raise RuntimeError("function_failed") from exc

    async def evaluate(self, request_id: str) -> Evaluation:
        row = self.store.get(request_id)
        if not row or row["status"] != RequestStatus.SCHEDULED.value: return Evaluation(request_id, False)
        version, session, invalidation = self.store.eligibility_version(), self.store.session, self._invalidated
        if row["next_attempt_at"] is not None and row["next_attempt_at"] > time.time():
            return Evaluation(request_id, False)
        deps = self.store.acknowledged_dependency_results(request_id)
        if deps is None:
            if self.store.dependency_failed(request_id):
                self.store.fail_scheduled(request_id, "dependency_failed", version, session.token if session else "", session.generation if session else -1)
                return Evaluation(request_id, False, error_code="dependency_failed")
            return Evaluation(request_id, False)
        try:
            ready, template = self.store.descriptor(row, "ready"), self.store.descriptor(row, "template")
            if ready is not None:
                ready_result = await self._call(ready, deps)
                if type(ready_result) is not bool: raise RuntimeError("function_failed")
                if not ready_result:
                    current = self.store.session
                    if (self.store.eligibility_version() != version or self._invalidated != invalidation or
                            session is None or current is None or (session.token, session.generation) != (current.token, current.generation)):
                        raise StaleCallback("stale evaluation")
                    return Evaluation(request_id, False)
            payload = row["payload_reference"]
            if template is not None:
                payload = await self._call(template, deps)
                if not isinstance(payload, str) or not payload: raise RuntimeError("template_invalid")
                try:
                    valid = await asyncio.to_thread(self.validator, payload)
                    if inspect.isawaitable(valid): valid = await valid
                except Exception as exc:
                    raise RuntimeError("template_invalid") from exc
                if type(valid) is not bool or not valid: raise RuntimeError("template_invalid")
            current = self.store.session
            if (self.store.eligibility_version() != version or self._invalidated != invalidation or
                session is None or current is None or (session.token, session.generation) != (current.token, current.generation)):
                raise StaleCallback("stale evaluation")
            return Evaluation(request_id, True, self.store.record_evaluation(request_id, version, payload))
        except StaleCallback: return Evaluation(request_id, False, error_code="stale_evaluation")
        except RuntimeError as exc:
            code = str(exc) if str(exc) in {"function_unavailable", "function_failed", "template_invalid"} else "function_failed"
            if not self.store.fail_scheduled(request_id, code, version, session.token if session else "", session.generation if session else -1):
                return Evaluation(request_id, False, error_code="stale_evaluation")
            return Evaluation(request_id, False, error_code=code)

    async def next_eligible(self, claim=False):
        after, version, invalidation = None, self.store.eligibility_version(), self._invalidated
        while True:
            if self._invalidated != invalidation:
                after, version, invalidation = None, self.store.eligibility_version(), self._invalidated
            rows = self.store.eligible_candidates(self.slice_size, after)
            if not rows: return None
            for row in rows:
                await asyncio.sleep(0)
                result = await self.evaluate(row["request_id"])
                if result.eligible:
                    if self._invalidated != invalidation or self.store.eligibility_version() != version:
                        after, version, invalidation = None, self.store.eligibility_version(), self._invalidated; break
                    if claim: self.store.claim_evaluated(result.capability)
                    return result
                if self._invalidated != invalidation or self.store.eligibility_version() != version:
                    after, version, invalidation = None, self.store.eligibility_version(), self._invalidated; break
                after = row["rank"]
            else: continue

    async def poll(self, stop=None):
        stop = stop or asyncio.Event()
        while not stop.is_set():
            self.invalidate()
            try: await asyncio.wait_for(stop.wait(), self.poll_interval)
            except asyncio.TimeoutError: pass

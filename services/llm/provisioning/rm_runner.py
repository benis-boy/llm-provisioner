"""Bounded, offline provisioning orchestration through the real ResourceManager.

This is deliberately a thin operator/benchmark entry point.  It does not
execute an adapter itself: the server-owned binding supplies the provider to
the ResourceManager, which remains the sole owner of residency, admission,
generation fencing, timing events, and cleanup.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile
from services.llm.resource_manager.core import ResourceManager
from services.llm.resource_manager.protocol import (
    EventKind, ProgressEvent, SessionInfo, Submission,
)
from services.llm.resource_manager.http import ModelBinding
from .benchmark_requests import MAX_REQUEST_BYTES
from .evidence import ClassifiedCapacityEvidenceError
try:
    from tools.compatibility.debug_trace import record as trace
except ImportError:
    def trace(*args, **kwargs):
        return None

MAX_WAVE_EVENTS = 4096
MAX_WAVE_EVIDENCE_BYTES = 8 * 1024 * 1024

# Provider failures cross this boundary only as compact, producer-owned
# categories.  In particular, never copy Failure.message into benchmark
# evidence: it may contain transport or model data.
SAFE_PROVIDER_FAILURE_CODES = frozenset({
    "ollama_http_status", "ollama_transport", "ollama_json_response",
    "ollama_response_contract", "ollama_telemetry_contract",
    "smollm_input_decode", "smollm_input_validation",
    "smollm_observation_contract", "smollm_internal",
    "malformed_provider_response",
    "provider_process_exit", "worker_operation_failed", "oom", "allocator_observation_failed", "decoder_metadata_workload_failed", "output_contract_failed", "rpc_request_bound_failed", "rpc_response_bound_failed", "request_validation_failed",
})


def safe_provider_failure_code(value: object) -> str | None:
    if value in {"out_of_memory", "oom"}:
        return "oom"
    return value if value in SAFE_PROVIDER_FAILURE_CODES else None

@dataclass(frozen=True)
class ProvisioningEvidence:
    """Observed RM evidence; it is not a measured-capacity result."""

    session: SessionInfo
    submission: Submission
    events: tuple[ProgressEvent, ...]
    result: bytes
    time_on_gpu_ms: int | None
    gpu_timing_complete: bool
    elapsed_ms: int
    measured_capacity_claim: bool = False


class ProvisioningError(RuntimeError):
    """A benchmark request was not completed through the RM lifecycle."""


def _wave_contract_error() -> ClassifiedCapacityEvidenceError:
    return ClassifiedCapacityEvidenceError("wave_event_contract_failed")


class ResourceManagerWaveRunner:
    """Reusable one-session wave boundary for authoritative measurement.

    The runner owns no provider calls.  It submits through the real manager,
    fences every event by session/generation/request identity, and exposes only
    the bounded event bundle to the evidence extractor.
    """

    def __init__(self, resource_manager: ResourceManager, binding: ModelBinding, *,
                  scheduler_id: str, model_id: ModelId, context_size: int | None = None,
                  bucket_identity: str | None = None, timeout: float = 120.0,
                  run_identity: str = "run", provisioning_profile: CapacityProfile,
                  configured_ceiling: int):
        if not isinstance(resource_manager, ResourceManager):
            raise TypeError("an actual ResourceManager instance is required")
        if not isinstance(binding, ModelBinding):
            raise TypeError("a server-owned ModelBinding is required")
        self.rm, self.binding = resource_manager, binding
        self.scheduler_id, self.model_id = scheduler_id, ModelId(model_id)
        if not isinstance(run_identity, str) or not run_identity:
            raise ValueError("run_identity must be non-empty")
        if binding.model_id != self.model_id:
            raise ValueError("binding model does not match runner model")
        if not isinstance(provisioning_profile, CapacityProfile):
            raise TypeError("an ephemeral server-owned provisioning profile is required")
        if type(configured_ceiling) is not int or not 1 <= configured_ceiling <= 32:
            raise ValueError("configured_ceiling must be between 1 and 32")
        if provisioning_profile.model_id != self.model_id:
            raise ValueError("provisioning profile model does not match runner model")
        binding_identity = (binding.gpu_uuid, binding.artifact_manifest_hash,
                            binding.model_hash, binding.runtime_identity,
                            binding.adapter_identity)
        profile_identity = (provisioning_profile.gpu_uuid, provisioning_profile.artifact_manifest_hash,
                            provisioning_profile.model_hash, provisioning_profile.runtime_identity,
                            provisioning_profile.adapter_identity)
        if binding_identity != profile_identity:
            raise ValueError("provisioning profile identity does not match binding")
        if ((context_size is not None and (provisioning_profile.context_size != context_size or bucket_identity is not None))
                or (bucket_identity is not None and (provisioning_profile.bucket_identity != bucket_identity or context_size is not None))
                or (context_size is None and bucket_identity is None)):
            raise ValueError("provisioning profile selector does not exactly match runner selector")
        if (provisioning_profile.optimal_parallelism < configured_ceiling
                or provisioning_profile.buffer_capacity < configured_ceiling):
            raise ValueError("provisioning profile cannot admit configured discovery ceiling")
        self.context_size, self.bucket_identity = context_size, bucket_identity
        self.timeout = _positive_timeout(timeout)
        self.run_identity = run_identity
        self.provisioning_profile = provisioning_profile
        self.configured_ceiling = configured_ceiling
        self.session: SessionInfo | None = None
        self._closed = False
        self._request_keys: set[tuple[str, str]] = set()
        self._cursor = 0

    async def _ensure_session(self) -> SessionInfo:
        if self.session is not None:
            return self.session
        profile, provider = self.binding.resolve(context_size=self.context_size,
                                                   bucket_identity=self.bucket_identity)
        if not isinstance(profile, CapacityProfile) or profile.model_id != self.model_id:
            raise ProvisioningError("binding returned a mismatched capacity profile")
        if (profile.gpu_uuid, profile.artifact_manifest_hash, profile.model_hash,
                profile.runtime_identity, profile.adapter_identity) != (
                    self.provisioning_profile.gpu_uuid, self.provisioning_profile.artifact_manifest_hash,
                    self.provisioning_profile.model_hash, self.provisioning_profile.runtime_identity,
                    self.provisioning_profile.adapter_identity):
            raise ProvisioningError("binding resolved an identity mismatched to provisioning profile")
        if (profile.context_size != self.provisioning_profile.context_size
                or profile.bucket_identity != self.provisioning_profile.bucket_identity):
            raise ProvisioningError("binding resolved a selector mismatched to provisioning profile")
        trace("rm_runner", "session", "enter", model=self.model_id.value,
              context=self.context_size, bucket=self.bucket_identity)
        self.session = await self.rm.start_session(
            self.scheduler_id, self.model_id, self.provisioning_profile, provider,
            idempotency_key=f"measurement-start:{self.scheduler_id}:{self.model_id.value}")
        return self.session

    async def __call__(self, concurrency: int, wave: int,
                       request_ids: tuple[str, ...], payload: bytes,
                       extractor: Callable[..., Any]):
        trace("rm_runner", "submit", "enter", model=self.model_id.value,
              concurrency=concurrency, wave=wave)
        session = await self._ensure_session()
        if len(request_ids) != concurrency or len(set(request_ids)) != concurrency:
            raise ProvisioningError("wave request identity is not exact")
        if concurrency > self.configured_ceiling:
            raise ProvisioningError("wave exceeds configured discovery ceiling")
        attempts = {request_id: f"{self.run_identity}-attempt-{request_id}" for request_id in request_ids}
        started = time.monotonic()
        try:
          async with asyncio.timeout(self.timeout):
            submissions = await asyncio.gather(*(
                self.rm.submit(session.session_token, request_id, attempts[request_id], payload,
                               idempotency_key=f"measurement-submit:{self.run_identity}:{request_id}:{attempts[request_id]}",
                               context_size=self.context_size, bucket_identity=self.bucket_identity)
                for request_id in request_ids))
            if any((not item.accepted or item.session_token != session.session_token or
                    item.generation != session.generation or item.request_id != request_id or
                    item.attempt != attempts[request_id])
                   for item, request_id in zip(submissions, request_ids)):
                 raise _wave_contract_error()
            expected = {(item.request_id, item.attempt) for item in submissions}
            self._request_keys.update(expected)
            events: list[ProgressEvent] = []
            finished: set[tuple[str, str]] = set()
            evidence_bytes = 0
            async for event in self.rm.watch_progress(session.session_token, self._cursor):
                self._cursor = event.sequence
                if len(events) >= MAX_WAVE_EVENTS:
                     raise _wave_contract_error()
                if event.session_token != session.session_token or event.generation != session.generation:
                     raise _wave_contract_error()
                if event.request_id is None or event.attempt is None:
                    # Session-level events are allowed, but never terminal events.
                    if (event.kind in {EventKind.RESPONSE_FINISHED, EventKind.CANCELLED}
                            or (event.kind is EventKind.FAILURE and event.failure is None)):
                         raise _wave_contract_error()
                elif (event.request_id, event.attempt) not in expected:
                     raise _wave_contract_error()
                if event.result is not None:
                    evidence_bytes += len(event.result)
                    if evidence_bytes > MAX_WAVE_EVIDENCE_BYTES:
                         raise _wave_contract_error()
                events.append(event)
                if event.kind is EventKind.FAILURE or event.kind is EventKind.CANCELLED:
                    error = ProvisioningError("wave request failed or was cancelled")
                    failure = event.failure
                    code = safe_provider_failure_code(failure.code) if failure else None
                    kind = "oom" if code == "oom" else "runner_error"
                    setattr(error, "failure_kind", kind)
                    if code is not None:
                        setattr(error, "failure_code", code)
                    raise error
                if event.kind is EventKind.RESPONSE_FINISHED:
                    key = (event.request_id, event.attempt)
                    if key in expected:
                        if key in finished:
                             raise _wave_contract_error()
                        if not event.result:
                             raise _wave_contract_error()
                        finished.add(key)
                if finished == expected:
                    break
            if finished != expected or len(finished) != len(expected):
                 raise _wave_contract_error()
        except asyncio.TimeoutError:
            raise ClassifiedCapacityEvidenceError("wave_timeout") from None
        elapsed_ms = max(1, int((time.monotonic() - started) * 1000))
        trace("rm_runner", "watch", "success", model=self.model_id.value,
              concurrency=concurrency, wave=wave, event_count=len(events), elapsed_ms=elapsed_ms)
        return extractor(concurrency, wave, request_ids, tuple(events), elapsed_ms)

    async def close(self) -> None:
        if self._closed or self.session is None:
            return
        session = self.session
        try:
            async def cleanup() -> None:
                failures: list[BaseException] = []
                for request_id, _attempt in tuple(self._request_keys):
                    try:
                        await self.rm.cancel_request(
                            session.session_token, request_id,
                            idempotency_key=f"measurement-cancel:{self.run_identity}:{request_id}")
                    except Exception as exc:
                        failures.append(exc)
                await self.rm.stop_session(
                    session.session_token, reason="measurement_complete",
                    idempotency_key=f"measurement-stop:{self.run_identity}:{session.session_token}")
                if failures:
                    raise ProvisioningError("not all admitted benchmark requests were cancelled") from failures[0]
            task = asyncio.create_task(cleanup())
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await asyncio.shield(task)
                raise
        except Exception as exc:
            raise ProvisioningError("ResourceManager cleanup was not proved") from exc
        if self.rm.snapshot().phase == "cleanup_failed":
            raise ProvisioningError("ResourceManager cleanup was not proved")
        self._closed = True
        trace("rm_runner", "close", "success", model=self.model_id.value)


def _positive_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout must be positive")
    try:
        finite_value = float(value)
    except (OverflowError, ValueError):
        raise ValueError("timeout must be positive") from None
    if not math.isfinite(finite_value) or finite_value <= 0:
        raise ValueError("timeout must be positive")
    return finite_value


async def provision_request(
    resource_manager: ResourceManager,
    binding: ModelBinding,
    *,
    scheduler_id: str,
    model_id: ModelId,
    payload: bytes,
    request_id: str,
    attempt: str,
    context_size: int | None = None,
    bucket_identity: str | None = None,
    timeout: float = 300.0,
) -> ProvisioningEvidence:
    """Run one bounded benchmark request using the normal RM session path.

    ``resource_manager`` must be the actual in-process ResourceManager (or a
    test-owned subclass).  A binding is resolved once, then its real provider
    is passed to ``start_session``; this function never invokes provider
    methods.  The session is stopped in all paths after it is acquired.
    """
    if not isinstance(resource_manager, ResourceManager):
        raise TypeError("an actual ResourceManager instance is required")
    if not isinstance(binding, ModelBinding):
        raise TypeError("a server-owned ModelBinding is required")
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("payload must be non-empty bytes")
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("payload exceeds request size bound")
    try:
        model = ModelId(model_id)
    except ValueError as exc:
        raise ValueError("unknown model") from exc
    if binding.model_id != model:
        raise ValueError("binding model does not match request model")
    for value, name in ((scheduler_id, "scheduler_id"), (request_id, "request_id"), (attempt, "attempt")):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be non-empty")
    timeout = _positive_timeout(timeout)
    profile, provider = binding.resolve(context_size=context_size, bucket_identity=bucket_identity)
    if not isinstance(profile, CapacityProfile) or profile.model_id != model:
        raise ValueError("binding returned a mismatched capacity profile")

    started = time.monotonic()
    session: SessionInfo | None = None
    submitted = False
    cancel_key = f"provision-cancel:{scheduler_id}:{request_id}:{attempt}"
    stop_key = f"provision-stop:{scheduler_id}:{request_id}:{attempt}"
    try:
      try:
        async with asyncio.timeout(timeout):
            session = await resource_manager.start_session(
                scheduler_id, model, profile, provider,
                idempotency_key=f"provision-start:{scheduler_id}:{model.value}",
            )
            if session.model_id != model or session.scheduler_id != scheduler_id:
                raise _wave_contract_error()
            submission = await resource_manager.submit(
                session.session_token, request_id, attempt, payload,
                idempotency_key=f"provision-submit:{request_id}:{attempt}",
                context_size=context_size, bucket_identity=bucket_identity,
            )
            submitted = True
            if (submission.session_token != session.session_token or
                    submission.generation != session.generation or
                    submission.request_id != request_id or submission.attempt != attempt):
                raise _wave_contract_error()
            if not submission.accepted:
                raise _wave_contract_error()

            events: list[ProgressEvent] = []
            result: bytes | None = None
            evidence_bytes = 0
            timing: int | None = None
            timing_complete = False
            async for event in resource_manager.watch_progress(session.session_token):
                if len(events) >= MAX_WAVE_EVENTS:
                    raise _wave_contract_error()
                if event.session_token != session.session_token or event.generation != session.generation:
                    raise _wave_contract_error()
                if event.request_id not in (None, request_id) or event.attempt not in (None, attempt):
                    raise _wave_contract_error()
                if event.result is not None:
                    evidence_bytes += len(event.result)
                    if evidence_bytes > MAX_WAVE_EVIDENCE_BYTES:
                        raise _wave_contract_error()
                events.append(event)
                if event.kind is EventKind.RESPONSE_FINISHED:
                    if (event.request_id != submission.request_id or
                            event.attempt != submission.attempt):
                        raise _wave_contract_error()
                    if event.result is not None:
                        result = event.result
                    timing, timing_complete = event.time_on_gpu_ms, event.gpu_timing_complete
                if event.kind is EventKind.FAILURE:
                    error = ProvisioningError("benchmark request failed")
                    failure = event.failure
                    code = safe_provider_failure_code(failure.code) if failure else None
                    setattr(error, "failure_kind", "oom" if code == "oom" else "runner_error")
                    if code is not None:
                        setattr(error, "failure_code", code)
                    raise error
                if event.kind is EventKind.CANCELLED:
                    raise ProvisioningError("benchmark request was cancelled")
                if event.kind is EventKind.RESPONSE_FINISHED:
                    break
            if not result or len(result) > MAX_WAVE_EVIDENCE_BYTES:
                raise _wave_contract_error()
            return ProvisioningEvidence(session, submission, tuple(events), result, timing,
                                        timing_complete, int((time.monotonic() - started) * 1000))
      except asyncio.TimeoutError:
          raise ClassifiedCapacityEvidenceError("wave_timeout") from None
    finally:
        if session is not None:
            # Cleanup owns the lifecycle fence.  Shield a single cleanup task
            # so caller cancellation cannot abandon a resident provider.
            async def cleanup() -> None:
                if submitted:
                    try:
                        await resource_manager.cancel_request(
                            session.session_token, request_id, idempotency_key=cancel_key)
                    except Exception as exc:
                        raise ProvisioningError("ResourceManager request cleanup was not proved") from exc
                await resource_manager.stop_session(
                    session.session_token, reason="provisioning_complete",
                    idempotency_key=stop_key)
                if resource_manager.snapshot().phase == "cleanup_failed":
                    raise ProvisioningError("ResourceManager cleanup was not proved")

            task = asyncio.create_task(cleanup())
            cancelled = False
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                try:
                    await asyncio.shield(task)
                except Exception as exc:
                    raise ProvisioningError("ResourceManager cleanup was not proved") from exc
                raise

"""Authoritative, bounded capacity measurement.

Unlike :func:`capacity.measure_capacity`, this module is the provisioning
boundary: every request is admitted by the server-owned ResourceManager and
the adapter/native evidence extractor is the only way a wave becomes a
measurement.  This module deliberately does not persist a profile.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Iterable, Protocol

from .benchmark_requests import BenchmarkRequest
from .capacity import (CapacityEvidenceError, MemorySample, Wave, bounded_exception_text,
                       choose_optimum, sample_during, sample_overlaps_execution,
                       MAX_DIAGNOSTIC_LENGTH)
from .rm_runner import ResourceManagerWaveRunner, SAFE_PROVIDER_FAILURE_CODES, safe_provider_failure_code
try:
    from tools.compatibility.debug_trace import record as trace
except ImportError:
    def trace(*args, **kwargs):
        return None


EVIDENCE_ERROR_PREFIX = "evidence_error:"

# This is deliberately a closed public vocabulary.  ``reason`` remains bounded
# private diagnostic context; callers must use this field when crossing the
# measurement/CLI boundary rather than trying to reconstruct provenance from
# provider text.
MEASUREMENT_FAILURE_CODES = frozenset({
    "allocator", "baseline_memory_unsafe", "chronology", "cleanup_failed",
    "configured_ceiling_unproved", "decoder_workload", "evidence_error",
    "identity", "inconsistent_telemetry", "invalid_completion", "invalid_sample",
    "invalid_telemetry", "memory_telemetry_invalid", "missing_native_evidence",
    "native_batch_correlation", "native_overlap_missing", "no_execution_sample",
    "no_reserve_safe_request", "oom", "output", "reserve_breached",
    "runner_error", "runtime_setup_failed", "resident_warmup_failed",
    "residency_fence_failed", "sample_bound", "sampler_error",
    "telemetry_chronology_invalid", "telemetry_contract_failed",
    "telemetry_correlation_failed", "timing", "timing_domain_invalid",
    "timeout", "undercovered_decoder", "wave_event_contract_failed",
    "wave_timeout", "unknown_evidence_failure",
    *SAFE_PROVIDER_FAILURE_CODES,
})

MEASUREMENT_FAILURE_DETAILS = frozenset({
    "runner_error", "failed_output", "identity_mismatch",
    "ollama_native_batch_contract", "non_ollama_native_batch_contract",
    "missing_correlation", "observation_drops_or_count", "oom",
    "wave_timeout", "wave_event_contract_failed", "timeout",
    "sample_bound", "sampler_error", "no_execution_sample",
    *SAFE_PROVIDER_FAILURE_CODES,
})

_FAILURE_CODE_ALIASES = {
    "missing_timing_or_native_evidence": "missing_native_evidence",
    "missing_overlap_or_native_evidence": "missing_native_evidence",
}


def _failure_code(value: str | None) -> str:
    """Normalize producer categories without ever inspecting diagnostic text."""
    value = _FAILURE_CODE_ALIASES.get(value, value)
    return value if value in MEASUREMENT_FAILURE_CODES else "unknown_evidence_failure"


def _evidence_error_reason(exc: BaseException) -> str:
    return EVIDENCE_ERROR_PREFIX + bounded_exception_text(
        exc, limit=MAX_DIAGNOSTIC_LENGTH - len(EVIDENCE_ERROR_PREFIX))


_RESIDENT_RUNNER_CATEGORIES = frozenset({
    "oom", "runner_error", "wave_timeout", "wave_event_contract_failed",
    "timeout", "sample_bound", "sampler_error",
})
_RESIDENT_RUNNER_CATEGORIES |= SAFE_PROVIDER_FAILURE_CODES


def _resident_warmup_category(
    wave: Wave | None, exc: BaseException | None, p: int,
    request_ids: tuple[str, ...] | None = None,
) -> str | None:
    """Return a closed, non-provider diagnostic for a failed resident probe."""
    explicit = getattr(exc, "capacity_failure_kind", None) if exc is not None else None
    if explicit is None and exc is not None:
        explicit = getattr(exc, "failure_kind", None)
    if explicit in _RESIDENT_RUNNER_CATEGORIES:
        return explicit
    if wave is None:
        return "runner_error"
    if wave.failed or not wave.outputs_valid:
        return "failed_output"
    if (wave.concurrency != p or request_ids is None
            or wave.request_ids != request_ids or len(wave.request_ids) != p
            or len(set(wave.request_ids)) != p):
        return "identity_mismatch"
    expected_batch = 1 if wave.evidence_kind == "ollama_native" else p
    if wave.native_batch_size != expected_batch:
        return ("ollama_native_batch_contract" if wave.evidence_kind == "ollama_native"
                else "non_ollama_native_batch_contract")
    if p > 1 and wave.native_request_correlation is not True:
        return "missing_correlation"
    if p > 1 and wave.observation_drops != 0:
        return "observation_drops_or_count"
    if p > 1:
        expected_observations = p if wave.evidence_kind == "ollama_native" else 1
        expected_sizes = ((1,) * p if wave.evidence_kind == "ollama_native" else (p,))
        if wave.observed_native_batch_sizes != expected_sizes:
            return ("ollama_native_batch_contract" if wave.evidence_kind == "ollama_native"
                    else "non_ollama_native_batch_contract")
        if wave.observation_count != expected_observations:
            return "observation_drops_or_count"
    return None


class EvidenceExtractor(Protocol):
    def __call__(self, concurrency: int, wave: int, request_ids: tuple[str, ...],
                 events: tuple[object, ...], elapsed_ms: int) -> Wave: ...


class MeasurementSampler(Protocol):
    async def sample(self) -> MemorySample: ...


@dataclass(frozen=True)
class AuthoritativeMeasurement:
    """Immutable evidence returned to the profile-producing caller."""

    status: str
    n: int | None
    optimum: int | None
    m: int | None
    reserve_percent: int
    baseline: tuple[Wave, ...]
    warmups: tuple[Wave, ...]
    measured: tuple[Wave, ...]
    benchmark_fingerprint: str
    baseline_mean_ms: float | None
    baseline_pre_used_bytes: int | None
    peak_incremental_request_bytes: int | None
    total_vram_bytes: int | None
    derived_ceiling: int | None
    configured_ceiling: int
    failed_discovery: tuple[Wave, ...] = ()
    resource_bound_failure: Wave | None = None
    reason: str = ""
    profile_eligible: bool = False
    # This is an identity/configuration-derived provider capability, not the
    # operator's benchmark ceiling and not a memory estimate.
    supported_parallelism: int | None = None
    parallelism_bound_reason: str | None = None
    # A finite, authoritative stage/result classification for public reporting.
    # Defaulting preserves callers that construct historical evidence objects.
    failure_code: str | None = None
    # Discovery is not profile throughput evidence, but successful discovery
    # waves are retained so the persistence boundary can reconstruct the
    # producer's memory proof (including late-derived SmolLM bounds).
    successful_discovery: tuple[Wave, ...] = ()
    failure_detail: str | None = None

    @property
    def points(self) -> tuple[Wave, ...]:
        return self.measured


def _measurement_failure_detail(*waves: tuple[Wave, ...]) -> str | None:
    """Return only a structurally produced, closed failure category."""
    for group in waves:
        for wave in reversed(group):
            detail = wave.failure_detail
            if detail in MEASUREMENT_FAILURE_DETAILS:
                return detail
    return None


def _ceiling(total: int, pre_used: int, increment: int, configured: int) -> int:
    if min(total, increment) <= 0 or pre_used < 0 or pre_used > total:
        raise CapacityEvidenceError("memory telemetry cannot establish a ceiling")
    usable = total * 80 // 100 - pre_used
    if usable < 0:
        raise CapacityEvidenceError("baseline memory exceeds reserve-safe capacity")
    return usable // increment


def _generic_memory_summary(waves: Iterable[Wave], baseline_pre: int) -> int:
    """Return the replayable generic peak relative to the final baseline.

    This deliberately has no online/provisional state: both measurement
    finalization and persistence must apply the same formula to the same
    retained successful evidence.
    """
    retained = tuple(waves)
    if not retained:
        raise CapacityEvidenceError("memory proof evidence is empty")
    return max(max(sample.used_bytes for sample in wave.samples) - baseline_pre
               for wave in retained)


def _resident_ceiling(total: int, baseline_used: int, p2_delta: int) -> int:
    """Derive SmolLM's resident bound from the exact p=2 witness.

    ``p2_delta`` is the cumulative additional allocation over the p=1
    resident baseline, not a per-slot cost.  The first slot is therefore
    already paid for in ``baseline_used``.
    """
    if (type(total) is not int or type(baseline_used) is not int
            or type(p2_delta) is not int or total <= 0
            or baseline_used < 0 or baseline_used > total or p2_delta <= 0):
        raise CapacityEvidenceError("resident memory telemetry is invalid")
    usable = total * 80 // 100 - baseline_used
    if usable < 0:
        raise CapacityEvidenceError("baseline memory exceeds reserve-safe capacity")
    derived = 1 + usable // p2_delta
    if derived < 2:
        raise CapacityEvidenceError("resident p=2 evidence cannot prove N=2")
    return derived


def _resource_bound_witness(wave: Wave | None, n: int | None) -> bool:
    """Return whether retained discovery evidence proves the next slot unsafe.

    A runner error is not a capacity boundary.  In particular, the fallback
    used after resident SmolLM warmup removes the only request-attributable
    allocator increment, so it must retain a real resource outcome and its
    telemetry rather than promoting the configured probe limit.
    """
    if (wave is None or n is None or not wave.failed or wave.phase != "discovery"
            or wave.concurrency != n + 1 or wave.failure_kind not in {"reserve_breached", "oom"}
            or not wave.samples or any(not sample.valid() for sample in wave.samples)):
        return False
    return ((wave.failure_kind == "oom" or any(not sample.reserve_ok for sample in wave.samples))
            and any(sample_overlaps_execution(sample, wave.execution_started, wave.execution_ended)
                    for sample in wave.samples))


def _sweep_points(n: int) -> tuple[int, ...]:
    # Include endpoints and deterministic rounded tenths.  The set is bounded
    # to p=1 plus ten points, even for a large configured ceiling.
    if n < 1:
        return ()
    values = {1, n, *(round(n * i / 10) for i in range(1, 10))}
    if n >= 2:
        values.add(2)
    values = {value for value in values if 1 <= value <= n}
    # Preserve N while bounding the number of points beyond p=1.
    if n not in values:
        values.add(n)
    ordered = sorted(values - {1})
    if len(ordered) > 10:
        ordered = ordered[:9] + [n]
        ordered = sorted(set(ordered))
    return (1, *ordered)


def _valid_wave(wave: Wave, p: int, ordinal: int, request_ids: tuple[str, ...], expected_output: int) -> str | None:
    expected_native_batch = 1 if wave.evidence_kind == "ollama_native" else p
    if (wave.failed or not wave.outputs_valid or wave.concurrency != p or
            wave.wave != ordinal or wave.native_batch_size != expected_native_batch or
            wave.max_output_tokens != expected_output or
            (wave.evidence_kind == "torch_native" and wave.decoder_steps != (expected_output,) * p) or
            wave.request_ids != request_ids or len(wave.request_ids) != p or len(set(wave.request_ids)) != p):
        return "invalid_completion"
    if (wave.elapsed_ms <= 0 or not wave.cuda_synchronized or
            (wave.evidence_kind != "ollama_native" and (wave.allocator is None or not wave.allocator.valid()))):
        return "missing_timing_or_native_evidence"
    if not (type(wave.execution_started) is int and type(wave.execution_ended) is int
            and wave.execution_started >= 0 and wave.execution_ended >= wave.execution_started):
        return "timing_domain_invalid"
    if p > 1 and (wave.native_request_correlation is not True or wave.observation_drops != 0 or
                    (wave.evidence_kind == "ollama_native" and
                     (wave.observation_count != p or wave.observed_native_batch_sizes != (1,) * p)) or
                   (wave.evidence_kind != "ollama_native" and
                    (wave.observation_count != 1 or wave.observed_native_batch_sizes != (p,)))):
        return "missing_overlap_or_native_evidence"
    if not wave.samples or not all(sample.valid() for sample in wave.samples):
        return "invalid_telemetry"
    starts = [sample.start_ns if sample.start_ns is not None else sample.timestamp_ns for sample in wave.samples]
    ends = [sample.end_ns if sample.end_ns is not None else sample.timestamp_ns for sample in wave.samples]
    if any(start < prior_end for start, prior_end in zip(starts[1:], ends)):
        return "telemetry_chronology_invalid"
    if not any(sample_overlaps_execution(sample, wave.execution_started, wave.execution_ended)
               for sample in wave.samples):
        return "timing_domain_invalid"
    return None


async def measure_authoritative(
    request: BenchmarkRequest,
    runner: ResourceManagerWaveRunner,
    sampler: MeasurementSampler,
    *,
    configured_ceiling: int,
    identity_derived_max_parallelism: int | None = None,
    identity_derived_capability_reason: str | None = None,
    provider_max_parallelism: int | None = None,
    expected_max_output_tokens: int,
    evidence_extractor: EvidenceExtractor,
    runner_for_concurrency: Callable[[int], Awaitable[ResourceManagerWaveRunner]] | None = None,
    residency_fence: Callable[[int], Awaitable[None]] | None = None,
    timeout: float = 120.0,
    run_identity: str = "run",
) -> AuthoritativeMeasurement:
    """Run the four-baseline, bounded discovery and throughput protocol.

    ``runner`` must be a ``ResourceManagerWaveRunner``.  Thus this seam can be
    unit tested with a test provider and binding, while production cannot
    replace the RM with a provider-direct callback.
    """
    trace("measurement", "input_validation", "enter", configured_ceiling=configured_ceiling)
    if not isinstance(request, BenchmarkRequest):
        raise TypeError("a prepared immutable BenchmarkRequest is required")
    if not isinstance(runner, ResourceManagerWaveRunner):
        raise TypeError("an actual ResourceManager wave runner is required")
    if runner_for_concurrency is not None and not callable(runner_for_concurrency):
        raise TypeError("runner_for_concurrency must be callable")
    if residency_fence is not None and not callable(residency_fence):
        raise TypeError("residency_fence must be callable")
    if type(configured_ceiling) is not int or not 1 <= configured_ceiling <= 32:
        raise ValueError("configured_ceiling must be between 1 and 32")
    if (type(identity_derived_max_parallelism) is not int
            or identity_derived_max_parallelism < 1):
        raise ValueError("identity-derived maximum supported parallelism is required")
    if provider_max_parallelism is None:
        provider_max_parallelism = identity_derived_max_parallelism
    if (type(provider_max_parallelism) is not int or not 1 <= provider_max_parallelism <= 32
            or configured_ceiling > provider_max_parallelism
            or identity_derived_max_parallelism > provider_max_parallelism):
        raise ValueError("operator and identity parallelism must not exceed provider capability")
    if identity_derived_capability_reason not in {
        "identity_configured_provider_capability",
        "identity_configured_operator_ceiling",
    }:
        raise ValueError("identity-derived parallelism capability reason is required")
    if type(expected_max_output_tokens) is not int or expected_max_output_tokens < 1:
        raise ValueError("expected_max_output_tokens must be positive")
    if not callable(evidence_extractor):
        raise TypeError("a bounded evidence extractor is required")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    import math
    if not math.isfinite(timeout):
        raise ValueError("timeout must be finite and positive")

    baseline: list[Wave] = []
    warmups: list[Wave] = []
    measured: list[Wave] = []
    failed: list[Wave] = []
    discovery: list[Wave] = []
    used_ids: set[str] = set()
    counter = 0
    total: int | None = None
    pre_used: int | None = None
    peak_increment = 0
    resident_delta: int | None = None
    # Non-resident native batches deliberately have no memory-derived
    # parallelism formula.  Initialize this before the resident-only branch so
    # capability exhaustion and an N+1 resource witness remain usable proofs.
    derived: int | None = None
    last_sample_end = -1
    last_execution_end = -1
    discovery_integrity_failure: str | None = None
    discovery_failure_detail: str | None = None
    evidence_failure_reason: str | None = None
    failure_code: str | None = None
    active_runner: ResourceManagerWaveRunner | None = None
    active_concurrency: int | None = None

    def record_failure(code: str | None) -> str:
        nonlocal failure_code
        failure_code = _failure_code(code)
        return failure_code

    def safe_attr(value, name, default=None):
        try:
            return getattr(value, name, default)
        except BaseException:
            return default

    def failure_event(event, phase, p, wave_no, code, detail=None):
        trace("measurement", event, "failure", model=safe_attr(safe_attr(request, "model_id"), "value"),
              concurrency=p, wave=wave_no, wave_type=phase,
              failure_code=_failure_code(code), failure_detail=detail)

    def ids(p: int, phase: str, wave: int) -> tuple[str, ...]:
        nonlocal counter
        counter += 1
        result = tuple(f"measurement-{run_identity}-{counter}-{phase}-{wave}-{i}" for i in range(p))
        if used_ids.intersection(result):
            raise CapacityEvidenceError("request identity collision")
        used_ids.update(result)
        return result

    async def run(p: int, wave_no: int, phase: str) -> tuple[Wave, tuple[str, ...]]:
        nonlocal evidence_failure_reason, active_runner, active_concurrency
        if runner_for_concurrency is None:
            active_runner = runner
        elif active_concurrency != p:
            try:
                if active_runner is not None:
                    await active_runner.close()
                active_runner = await runner_for_concurrency(p)
                if not isinstance(active_runner, ResourceManagerWaveRunner):
                    raise CapacityEvidenceError("concurrency runtime did not provide an actual wave runner")
                active_concurrency = p
            except (CapacityEvidenceError, asyncio.TimeoutError, RuntimeError, ValueError):
                record_failure("runtime_setup_failed")
                failure_event("runner_switch", phase, p, wave_no, "runtime_setup_failed")
                request_ids = ids(p, phase, wave_no)
                return Wave(p, wave_no, request_ids, 0, False, 0, failed=True,
                            failure_kind="runtime_setup_failed", phase=phase), request_ids

            # The runtime-specific daemon is intentionally primed before memory
            # collection.  This makes model load a resident configuration cost,
            # rather than a per-request increment in the first sampled wave.
            prime: Wave | None = None
            prime_ids: tuple[str, ...] | None = None
            try:
                prime_ids = ids(p, "resident", 0)
                prime = await active_runner(p, 0, prime_ids, request.payload, evidence_extractor)
                if (prime.failed or not prime.outputs_valid or prime.concurrency != p
                        or prime.request_ids != prime_ids
                        or prime.native_batch_size != (1 if prime.evidence_kind == "ollama_native" else p)
                        or (p > 1 and (
                            prime.native_request_correlation is not True
                            or prime.observation_drops != 0
                             or (prime.evidence_kind == "ollama_native" and prime.observation_count != p)
                             or (prime.evidence_kind == "ollama_native" and
                                 prime.observed_native_batch_sizes != (1,) * p)
                            or (prime.evidence_kind != "ollama_native" and
                                (prime.observation_count != 1
                                 or prime.observed_native_batch_sizes != (p,)))))):
                    raise CapacityEvidenceError("runtime residency warmup did not prove native execution")
            except (CapacityEvidenceError, asyncio.TimeoutError, RuntimeError, ValueError) as exc:
                record_failure("resident_warmup_failed")
                request_ids = ids(p, phase, wave_no)
                detail = _resident_warmup_category(prime, exc, p, prime_ids)
                failure_event("runner_prime", phase, p, wave_no, "resident_warmup_failed", detail)
                return Wave(p, wave_no, request_ids, 0, False, 0, failed=True,
                            failure_kind="resident_warmup_failed", failure_detail=detail,
                            phase=phase), request_ids

            if residency_fence is not None:
                try:
                    await residency_fence(p)
                except (CapacityEvidenceError, asyncio.TimeoutError, RuntimeError, ValueError):
                    record_failure("residency_fence_failed")
                    failure_event("runner_residency", phase, p, wave_no, "residency_fence_failed")
                    request_ids = ids(p, phase, wave_no)
                    return Wave(p, wave_no, request_ids, 0, False, 0, failed=True,
                                failure_kind="residency_fence_failed", phase=phase), request_ids
        request_ids = ids(p, phase, wave_no)
        trace("measurement", "wave", "enter", model=request.model_id.value,
              concurrency=p, wave=wave_no, wave_type=phase)
        try:
            item = await sample_during(
                sampler,
                lambda: active_runner(p, wave_no, request_ids, request.payload,
                               evidence_extractor),
                timeout=timeout,
            )
            # Extractors intentionally describe native evidence only and may
            # therefore return Wave's measured default.  The producer owns
            # the protocol schedule; stamp it without changing any evidence
            # fields (or failure objects).
            item = replace(item, phase=phase)
            trace("measurement", "wave", "success", model=request.model_id.value,
                  concurrency=p, wave=wave_no, wave_type=phase, elapsed_ms=item.elapsed_ms)
            return item, request_ids
        except ValueError as exc:
            # Native extractors use ValueError for shape/workload rejection
            # (for example GECToR's unsupported p=2 path).  Treat that as a
            # bounded evidence failure, not as a runner/resource failure and
            # never let it escape the measurement boundary.
            evidence_failure_reason = _evidence_error_reason(exc)
            record_failure("evidence_error")
            failure_event("sample", phase, p, wave_no, "evidence_error")
            return Wave(p, wave_no, request_ids, 0, False, 0,
                        failed=True, failure_kind="evidence_error",
                        phase=phase), request_ids
        except (CapacityEvidenceError, asyncio.TimeoutError, RuntimeError) as exc:
            code = safe_attr(exc, "capacity_failure_kind", safe_attr(exc, "failure_kind", "runner_error"))
            detail = safe_provider_failure_code(safe_attr(exc, "failure_code"))
            if detail is None and isinstance(code, str) and code in MEASUREMENT_FAILURE_DETAILS:
                detail = code
            record_failure(code)
            failure_event("sample", phase, p, wave_no, code, detail)
            return Wave(p, wave_no, request_ids, 0, False, 0,
                        samples=tuple(getattr(exc, "capacity_samples", ())),
                        failed=True,
                        failure_kind=getattr(exc, "capacity_failure_kind",
                                             getattr(exc, "failure_kind", "runner_error")),
                        failure_detail=safe_provider_failure_code(
                            getattr(exc, "failure_code", None)),
                         phase=phase), request_ids

    def account(item: Wave, p: int, wave_no: int, request_ids: tuple[str, ...]) -> str | None:
        nonlocal total, pre_used, peak_increment, last_sample_end, last_execution_end
        if item.failed and item.failure_kind:
            failure_event("point", item.phase, p, wave_no, item.failure_kind, safe_attr(item, "failure_detail"))
            return item.failure_kind
        failure = _valid_wave(item, p, wave_no, request_ids, expected_max_output_tokens)
        if failure:
            failure_event("point", item.phase, p, wave_no, failure, None)
            return failure
        totals = {sample.total_bytes for sample in item.samples}
        if len(totals) != 1 or (total is not None and next(iter(totals)) != total):
            return "inconsistent_telemetry"
        total = next(iter(totals))
        first_used = item.samples[0].used_bytes
        if item.phase == "baseline":
            pre_used = first_used if pre_used is None else max(pre_used, first_used)
        starts = [(sample.start_ns if sample.start_ns is not None else sample.timestamp_ns)
                  for sample in item.samples]
        ends = [(sample.end_ns if sample.end_ns is not None else sample.timestamp_ns)
                for sample in item.samples]
        if (starts[0] < last_sample_end or item.execution_started < last_execution_end
                or any(a < b for a, b in zip(starts[1:], ends))):
            return "telemetry_chronology_invalid"
        last_sample_end = ends[-1]
        last_execution_end = item.execution_ended
        if any(not s.reserve_ok for s in item.samples):
            return "reserve_breached"
        # Only successful, reserve-safe telemetry may contribute to the
        # request-attributable denominator.  In particular, a reserve-breached
        # point must not be allowed to manufacture the formula used to bound
        # later discovery.
        peak_increment = max(peak_increment,
                             max(s.used_bytes for s in item.samples) - pre_used)
        return None

    answer: AuthoritativeMeasurement | None = None
    cancelled = False
    n: int | None = None
    try:
        for wave_no in range(1, 5):
            item, request_ids = await run(1, wave_no, "baseline")
            baseline.append(item)
            failure = account(item, 1, wave_no, request_ids)
            if failure:
                baseline[-1] = replace(item, failed=True, failure_kind=failure)
                record_failure(failure)
                detail = f":{item.failure_detail}" if item.failure_detail else ""
                raise CapacityEvidenceError(bounded_exception_text(
                    RuntimeError(f"{failure}{detail}"), limit=MAX_DIAGNOSTIC_LENGTH))
        # The online denominator was necessarily provisional while the baseline
        # was still being established.  Freeze the baseline witness now and
        # derive the first discovery bound from the same replayable evidence
        # used during finalization; otherwise a rising baseline can make an
        # early, stale peak truncate discovery before the final bound is known.
        pre_used = max(w.samples[0].used_bytes for w in baseline)
        # A resident SmolLM runtime can legitimately report no incremental
        # allocation: its one-time model allocation was deliberately fenced
        # before collection.  Zero is therefore not a division input and never
        # proves N.  Exhaustively exercise the bounded, exact-slot runtime
        # instead, and require an observed N+1 resource boundary below.
        resident_mode = request.model_id.value == "SmolLM"
        # Native non-resident batches report one aggregate device-wide wave.
        # Its delta is not a per-slot denominator: extrapolating it would make
        # a variable batch cost look linear and can truncate discovery early.
        # Retain the summary for replay, but prove the bound only by exhausting
        # the fixed capability or observing a real N+1 resource boundary.
        if not resident_mode:
            try:
                peak_increment = _generic_memory_summary(tuple(baseline), pre_used)
            except CapacityEvidenceError:
                valid_baseline = (total is not None and pre_used is not None and total > 0
                                  and 0 <= pre_used <= total)
                record_failure("baseline_memory_unsafe" if valid_baseline
                               else "memory_telemetry_invalid")
                raise
        # A native p=1-only adapter cannot turn a shape rejection into a memory
        # result.  Continue only when it has an independently admitted p>1 path.
        # Otherwise N=1 is deliberately unproved absent a real resource failure.
        discovery_limit = min(configured_ceiling, provider_max_parallelism)
        if discovery_limit < 1:
            record_failure("no_reserve_safe_request")
            raise CapacityEvidenceError("no reserve-safe representative request")

        # Discovery is exhaustive over concurrently admitted native waves;
        # each p is a separate point and a failed point never proves p-1.
        n = 1
        # A structurally bounded provider is measured at p=1 only.  Its
        # capability is already established by the binding identity; a
        # rejected p=2 is neither submitted nor misclassified as corruption.
        capability_bound = identity_derived_capability_reason == "identity_configured_provider_capability"
        p = 2
        while p <= discovery_limit:
            point_ok = True
            for wave_no in range(1, 5):
                item, request_ids = await run(p, wave_no, "discovery")
                if (failure := account(item, p, wave_no, request_ids)) is not None:
                    item = replace(item, failed=True, failure_kind=failure)
                    record_failure(failure)
                    failed.append(item)
                    point_ok = False
                    if failure not in {"reserve_breached", "oom"}:
                        discovery_integrity_failure = failure
                        discovery_failure_detail = item.failure_detail
                    break
                # Discovery evidence is retained in measured only when swept.
                discovery.append(item)
            if not point_ok:
                break
            n = p
            if resident_mode and p == 2:
                # This is the sole resident witness: four fully valid p=2
                # waves after the zero-increment p=1 baseline.  A zero p=2
                # increment is not a resident formula witness: continue the
                # exhaustive slot search so an actual N+1 resource boundary
                # can prove the bound instead.
                resident_delta = max(
                    max(s.used_bytes for s in w.samples) - (pre_used or 0)
                    for w in discovery[-4:]
                )
                if resident_delta > 0:
                    # Later sweep samples must not change this witness's
                    # denominator meaning.
                    try:
                        candidate = _resident_ceiling(total or 0, pre_used or 0,
                                                      resident_delta)
                    except CapacityEvidenceError:
                        record_failure("memory_telemetry_invalid")
                        raise
                    if candidate < p:
                        record_failure("configured_ceiling_unproved")
                        raise CapacityEvidenceError("configured_ceiling_unproved")
                    # Freeze the resident denominator at the complete p=2
                    # witness.  The resulting bound controls how far
                    # discovery runs, but must not truncate discovery at p=2:
                    # persistence requires every point through the frozen
                    # bound to have been exhaustively exercised.
                    # The immutable identity-derived capability is the
                    # persisted hard boundary.  Retain it here as well, so
                    # the effective formula result is replayable from the
                    # authoritative result rather than depending on the
                    # transient runner-only provider limit.
                    derived = min(candidate, identity_derived_max_parallelism)
                    discovery_limit = min(configured_ceiling, derived,
                                          provider_max_parallelism)
            p += 1
        if discovery_integrity_failure is not None:
            detail = (f":{discovery_failure_detail}" if discovery_failure_detail else "")
            reason = evidence_failure_reason or f"invalid discovery: {discovery_integrity_failure}{detail}"
            raise CapacityEvidenceError(bounded_exception_text(RuntimeError(reason),
                                                                limit=MAX_DIAGNOSTIC_LENGTH))
        # A resource/reserve failure at N+1 is a bound.  Merely reaching the
        # operator cap is not: that would promote an arbitrary benchmark cap.
        bounded_by_failure = _resource_bound_witness(failed[-1] if failed else None, n)
        capability_exhaustive = (identity_derived_capability_reason ==
                                 "identity_configured_provider_capability"
                                 and configured_ceiling == provider_max_parallelism
                                 and n == provider_max_parallelism
                                 and tuple((w.concurrency, w.wave) for w in discovery)
                                 == tuple((p, wave) for p in range(2, provider_max_parallelism + 1)
                                          for wave in range(1, 5)))
        if derived is None and not bounded_by_failure and not capability_exhaustive:
            record_failure("configured_ceiling_unproved")
            raise CapacityEvidenceError("configured_ceiling_unproved")
        if (not resident_mode and not bounded_by_failure and not capability_exhaustive):
            record_failure("configured_ceiling_unproved")
            raise CapacityEvidenceError("native aggregate memory requires capability exhaustion or a resource boundary")
        if (derived is not None and configured_ceiling < derived and not bounded_by_failure
                and not capability_bound):
            record_failure("configured_ceiling_unproved")
            raise CapacityEvidenceError(
                "configured ceiling prevents establishing the memory-safe N; "
                "a cap is not a capacity proof")
        points = _sweep_points(n)
        grouped: dict[int, list[Wave]] = {}
        for p in points:
            warm, request_ids = await run(p, 0, "warmup")
            warmups.append(warm)
            if (failure := account(warm, p, 0, request_ids)) is not None:
                warmups[-1] = replace(warm, failed=True, failure_kind=failure)
                record_failure(failure)
                detail = f":{warm.failure_detail}" if warm.failure_detail else ""
                raise CapacityEvidenceError(bounded_exception_text(
                    RuntimeError(f"{failure}{detail}"), limit=MAX_DIAGNOSTIC_LENGTH))
            rows: list[Wave] = []
            for wave_no in range(1, 5):
                item, request_ids = await run(p, wave_no, "measured")
                measured.append(item)
                if (failure := account(item, p, wave_no, request_ids)) is not None:
                    measured[-1] = replace(item, failed=True, failure_kind=failure)
                    record_failure(failure)
                    detail = f":{item.failure_detail}" if item.failure_detail else ""
                    raise CapacityEvidenceError(bounded_exception_text(
                        RuntimeError(f"{failure}{detail}"), limit=MAX_DIAGNOSTIC_LENGTH))
                rows.append(item)
            grouped[p] = rows
        # Persistence must describe the final retained telemetry.  Native
        # aggregate batches retain this summary for replay, but never turn it
        # into a per-slot denominator.
        retained = (*baseline, *discovery, *warmups, *measured)
        pre_used = max(w.samples[0].used_bytes for w in baseline)
        authoritative_peak = _generic_memory_summary(retained, pre_used)
        if resident_mode and resident_delta is not None:
            p2_waves = tuple(w for w in discovery
                             if w.phase == "discovery" and w.concurrency == 2)
            resident_delta = _generic_memory_summary(p2_waves, pre_used)
        if resident_mode and derived is not None and resident_delta is None:
            peak_increment = authoritative_peak
            if peak_increment <= 0:
                record_failure("memory_telemetry_invalid")
                raise CapacityEvidenceError("positive request-attributable telemetry is required")
            try:
                final_derived = _ceiling(total or 0, pre_used or 0, peak_increment,
                                         configured_ceiling)
            except CapacityEvidenceError:
                record_failure("memory_telemetry_invalid")
                raise
            if derived != final_derived or final_derived < n:
                record_failure("configured_ceiling_unproved")
                raise CapacityEvidenceError("final telemetry changed the proved ceiling")
            derived = final_derived
        if resident_mode:
            peak_increment = resident_delta or 0
        optimum = choose_optimum(grouped)
        answer = AuthoritativeMeasurement(
            "complete", n, optimum, optimum, 20, tuple(baseline), tuple(warmups),
            tuple(measured), request.fingerprint, sum(w.elapsed_ms for w in baseline) / 4,
            pre_used, peak_increment, total, derived, configured_ceiling,
             tuple(failed), failed[-1] if bounded_by_failure else None, "complete", True,
             identity_derived_max_parallelism,
             identity_derived_capability_reason, None, tuple(discovery))
    except asyncio.CancelledError:
        cancelled = True
        raise
    except (CapacityEvidenceError, asyncio.TimeoutError, RuntimeError) as exc:
        answer = AuthoritativeMeasurement(
            "incomplete", n if n else None, None, None, 20, tuple(baseline),
            tuple(warmups), tuple(measured), request.fingerprint,
            sum(w.elapsed_ms for w in baseline) / len(baseline) if baseline else None,
             pre_used, peak_increment or None, total, None, configured_ceiling,
              tuple(failed), None, evidence_failure_reason or bounded_exception_text(exc), False,
              identity_derived_max_parallelism, identity_derived_capability_reason,
              _failure_code(failure_code),
               failure_detail=_measurement_failure_detail(tuple(baseline), tuple(failed),
                                                          tuple(warmups), tuple(measured)))
    except Exception as exc:
        # Keep unexpected producer faults fail-closed and non-diagnostic at the
        # public boundary while preserving a bounded private reason locally.
        answer = AuthoritativeMeasurement(
            "incomplete", n if n else None, None, None, 20, tuple(baseline),
            tuple(warmups), tuple(measured), request.fingerprint,
            sum(w.elapsed_ms for w in baseline) / len(baseline) if baseline else None,
            pre_used, peak_increment or None, total, None, configured_ceiling,
            tuple(failed), None, bounded_exception_text(exc), False,
            identity_derived_max_parallelism, identity_derived_capability_reason,
            "unknown_evidence_failure")
    finally:
        try:
            if active_runner is not None:
                await active_runner.close()
            trace("measurement", "runner_close", "success")
        except asyncio.CancelledError:
            raise
        except Exception as cleanup_error:
            if cancelled:
                # Let the pending original cancellation escape after joined cleanup.
                pass
            else:
                answer = AuthoritativeMeasurement(
                    "incomplete", answer.n if answer else None, None, None, 20,
                    answer.baseline if answer else tuple(baseline),
                    answer.warmups if answer else tuple(warmups),
                    answer.measured if answer else tuple(measured), request.fingerprint,
                    answer.baseline_mean_ms if answer else None,
                    answer.baseline_pre_used_bytes if answer else pre_used,
                    answer.peak_incremental_request_bytes if answer else (peak_increment or None),
                    answer.total_vram_bytes if answer else total, answer.derived_ceiling if answer else None,
                     configured_ceiling, answer.failed_discovery if answer else tuple(failed),
                       None, f"cleanup_failed: {bounded_exception_text(cleanup_error)}", False,
                       failure_code="cleanup_failed", failure_detail=None)
    return answer

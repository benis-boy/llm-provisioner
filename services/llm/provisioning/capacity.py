"""Fail-closed, candidate-only capacity measurement primitives.

These primitives retain observations; they never manufacture an approved
capacity/profile from a bounded probe.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, Sequence
from services.llm.providers.coedit_batch import AllocatorObservation

RESERVE_PERCENT = 20
MAX_INPUT_FILE_BYTES = 256 * 1024
MAX_POINTS = 11                 # p=1 plus at most ten points above it
THROUGHPUT_MAX_PARALLELISM = 16
DISCOVERY_MAX_PARALLELISM = 32
SAMPLE_INTERVAL_SECONDS = 0.01
MAX_SAMPLES_PER_WAVE = 16384
SAMPLE_TIMEOUT_SECONDS = 120.0
MAX_DIAGNOSTIC_LENGTH = 160


class CapacityEvidenceError(RuntimeError):
    """Evidence was insufficient or contradicted the measurement contract."""


def bounded_exception_text(exc: BaseException, *, limit: int = MAX_DIAGNOSTIC_LENGTH) -> str:
    """Return sanitized, single-line operator text without importing measurement."""
    if type(limit) is not int or limit < 1:
        raise ValueError("diagnostic limit must be a positive integer")
    text = str(exc).strip()
    text = "".join(char if char >= " " and char != "\x7f" else " " for char in text)
    return " ".join(text.split())[:limit]


def bounded_failure_text(code: object, message: object, *, limit: int = MAX_DIAGNOSTIC_LENGTH) -> str:
    """Format only the bounded RM failure fields for an operator diagnostic."""
    if type(limit) is not int or limit < 1:
        raise ValueError("diagnostic limit must be a positive integer")
    prefix = bounded_exception_text(RuntimeError(str(code)), limit=limit)
    separator = ":"
    remaining = max(0, limit - len(prefix) - len(separator))
    return prefix + separator + bounded_exception_text(RuntimeError(str(message)), limit=remaining) if remaining else prefix


def _retain_samples(exc: BaseException, samples: list["MemorySample"], kind: str) -> BaseException:
    """Attach bounded numeric evidence without incorporating exception text."""
    setattr(exc, "capacity_samples", tuple(samples[:MAX_SAMPLES_PER_WAVE]))
    setattr(exc, "capacity_failure_kind", kind)
    return exc


@dataclass(frozen=True)
class MemorySample:
    timestamp_ns: int
    total_bytes: int
    used_bytes: int
    free_bytes: int
    start_ns: int | None = None
    end_ns: int | None = None
    def valid(self) -> bool:
        start = self.timestamp_ns if self.start_ns is None else self.start_ns
        end = self.timestamp_ns if self.end_ns is None else self.end_ns
        return (all(type(x) is int for x in (self.timestamp_ns, self.total_bytes, self.used_bytes, self.free_bytes, start, end))
                and self.timestamp_ns >= 0 and self.total_bytes > 0 and 0 <= self.used_bytes <= self.total_bytes
                and 0 <= self.free_bytes <= self.total_bytes and self.used_bytes + self.free_bytes <= self.total_bytes
                and start >= 0 and end >= start and self.timestamp_ns == end)
    @property
    def reserve_ok(self) -> bool:
        return self.valid() and self.free_bytes * 100 >= self.total_bytes * RESERVE_PERCENT


def sample_overlaps_execution(sample: MemorySample, execution_started: object,
                               execution_ended: object) -> bool:
    """Return whether a valid fenced sample intersects a valid execution interval."""
    if (not sample.valid() or type(execution_started) is not int
            or type(execution_ended) is not int or execution_started < 0
            or execution_ended < execution_started):
        return False
    start = sample.timestamp_ns if sample.start_ns is None else sample.start_ns
    end = sample.timestamp_ns if sample.end_ns is None else sample.end_ns
    return start <= execution_ended and end >= execution_started


@dataclass(frozen=True)
class Wave:
    concurrency: int; wave: int; request_ids: tuple[str, ...]; elapsed_ms: int
    outputs_valid: bool; native_batch_size: int
    execution_started: int | float | None = None; execution_ended: int | float | None = None
    cuda_synchronized: bool = False; samples: tuple[MemorySample, ...] = ()
    failed: bool = False; failure_kind: str | None = None
    phase: str = "measured"
    allocator: AllocatorObservation | None = None
    observation_count: int = 0
    observed_native_batch_sizes: tuple[int, ...] = ()
    native_request_correlation: bool = False
    observation_drops: int = 0
    decoder_steps: tuple[int, ...] = ()
    max_output_tokens: int | None = None
    workload_kind: str = "decoder_tokens"
    workload_witness: tuple[int, ...] = ()
    evidence_kind: str = "torch_native"
    # These are authoritative provider/native intervals, keyed in request_ids
    # order.  They are retained so persistence cannot substitute wall time for
    # per-request latency.
    request_latency_ms: tuple[int, ...] = ()
    failure_detail: str | None = None
    @property
    def successful_requests(self) -> int:
        return self.concurrency if not self.failed and self.outputs_valid else 0
    @property
    def peak_used_bytes(self) -> int:
        return max((sample.used_bytes for sample in self.samples), default=0)


@dataclass(frozen=True)
class CapacityResult:
    status: str; candidate_n: int | None; optimal_parallelism: int | None; reason: str
    points: tuple[Wave, ...] = (); baseline: tuple[Wave, ...] = (); warmups: tuple[Wave, ...] = ()
    profile_eligible: bool = False
    failure_phase: str | None = None
    failure_kind: str | None = None
    expected_max_output_tokens: int | None = None
    @property
    def max_output_verified(self) -> bool:
        if (self.status != "candidate" or type(self.expected_max_output_tokens) is not int
                or self.expected_max_output_tokens <= 0):
            return False
        if isinstance(self.expected_max_output_tokens, bool):
            return False
        if type(self.candidate_n) is not int or self.candidate_n < 1 or self.candidate_n > 16:
            return False
        points = _points(self.candidate_n)
        if (tuple(wave.concurrency for wave in self.warmups) != points
                or tuple((wave.phase, wave.concurrency, wave.wave) for wave in self.baseline)
                   != tuple(("baseline", 1, wave) for wave in range(1, 5))
                or tuple((wave.phase, wave.concurrency, wave.wave) for wave in self.warmups)
                   != tuple(("warmup", point, 0) for point in points)
                or tuple((wave.phase, wave.concurrency, wave.wave) for wave in self.points)
                   != tuple(("measured", point, wave) for point in points for wave in range(1, 5))):
            return False
        waves = (*self.baseline, *self.warmups, *self.points)
        return all(type(wave.outputs_valid) is bool and type(wave.failed) is bool
            and not wave.failed and wave.outputs_valid
            and type(wave.max_output_tokens) is int
            and wave.max_output_tokens == self.expected_max_output_tokens
            and len(wave.decoder_steps) == wave.concurrency
            and all(type(value) is int and value == self.expected_max_output_tokens for value in wave.decoder_steps)
            for wave in waves)


@dataclass(frozen=True)
class DiscoveryResult:
    """Observations from the deliberately serial, incrementing discovery pass.

    ``observed_safe_through`` is an observation, not an approved memory bound:
    NVML and allocator samples do not constitute an exhaustive process peak.
    In particular, a non-resource failure never makes the preceding point a
    proved point.  Keeping this type separate prevents discovery from being
    mistaken for throughput optimization or profile approval.
    """
    status: str
    observed_safe_through: int | None
    candidate_ceiling: int | None
    stop_p: int | None
    reason: str
    failure_phase: str | None = None
    failure_kind: str | None = None
    baseline: tuple[Wave, ...] = ()
    points: tuple[Wave, ...] = ()
    expected_max_output_tokens: int | None = None
    memory_safe_n: None = None
    profile_eligible: bool = False

    @property
    def max_output_verified(self) -> bool:
        if (self.status != "complete" or type(self.expected_max_output_tokens) is not int
                or isinstance(self.expected_max_output_tokens, bool)
                or self.expected_max_output_tokens < 1
                or type(self.candidate_ceiling) is not int
                or not 2 <= self.candidate_ceiling <= DISCOVERY_MAX_PARALLELISM
                or self.observed_safe_through != self.candidate_ceiling
                or self.stop_p is not None):
            return False
        expected_baseline = tuple(("baseline", 1, wave) for wave in range(1, 5))
        expected_points = tuple(("discovery", p, wave)
                                for p in range(2, self.candidate_ceiling + 1)
                                for wave in range(1, 5))
        if (tuple((wave.phase, wave.concurrency, wave.wave) for wave in self.baseline) != expected_baseline
                or tuple((wave.phase, wave.concurrency, wave.wave) for wave in self.points) != expected_points):
            return False
        waves = (*self.baseline, *self.points)
        return all(
            not w.failed and w.outputs_valid and type(w.max_output_tokens) is int
            and w.max_output_tokens == self.expected_max_output_tokens
            and len(w.decoder_steps) == w.concurrency
            and all(type(step) is int and step == self.expected_max_output_tokens
                    for step in w.decoder_steps)
            for w in waves)


class WaveRunner(Protocol):
    async def __call__(self, concurrency: int, wave: int, request_ids: tuple[str, ...]) -> Wave: ...
class Sampler(Protocol):
    async def sample(self) -> MemorySample: ...

def throughput(waves: Sequence[Wave]) -> float:
    elapsed, requests = sum(w.elapsed_ms for w in waves), sum(w.successful_requests for w in waves)
    return requests / elapsed if elapsed > 0 else 0.0

def choose_optimum(points: dict[int, Sequence[Wave]]) -> int:
    if not points or 1 not in points: raise CapacityEvidenceError("serial point is missing")
    selected = 1
    for p in sorted(points):
        if p != 1 and throughput(points[p]) >= throughput(points[selected]) * 1.02: selected = p
    return selected

def _native_failure(wave: Wave, p: int, request_ids: tuple[str, ...], expected_wave: int,
                    expected_max_output_tokens: int | None = None) -> str | None:
    # A runner's explicit closed category is primary evidence.  Do not replace
    # it with a derived identity mismatch caused by deliberately empty fields.
    if wave.failed and wave.failure_kind in {"native_batch_correlation", "runner_error", "sampler_error", "timeout", "sample_bound"}:
        return wave.failure_kind
    if (wave.concurrency != p or wave.native_batch_size != p or len(wave.request_ids) != p
            or wave.request_ids != request_ids or wave.wave != expected_wave or len(set(wave.request_ids)) != p):
        return "identity"
    if type(wave.outputs_valid) is not bool or not wave.outputs_valid or type(wave.failed) is not bool or wave.failed:
        return "output"
    if (type(wave.max_output_tokens) is not int or isinstance(wave.max_output_tokens, bool)
            or wave.max_output_tokens < 1
            or (expected_max_output_tokens is not None and wave.max_output_tokens != expected_max_output_tokens)
            or len(wave.decoder_steps) != p
            or any(type(value) is not int or value < 0 or value > wave.max_output_tokens for value in wave.decoder_steps)):
        return "decoder_workload"
    if type(wave.cuda_synchronized) is not bool or not wave.cuda_synchronized or type(wave.elapsed_ms) is not int or wave.elapsed_ms <= 0:
        return "timing"
    if not (isinstance(wave.allocator, AllocatorObservation) and wave.allocator.valid()):
        return "allocator"
    if not (type(wave.execution_started) is int and type(wave.execution_ended) is int
            # Child timing and NVML fences are monotonic nanoseconds, so a
            # value in another unit/domain cannot prove overlap.
            and wave.execution_started >= 0 and wave.execution_ended >= wave.execution_started):
        return "timing"
    if not all(sample.valid() for sample in wave.samples): return "invalid_sample"
    if any((sample.start_ns if sample.start_ns is not None else sample.timestamp_ns) < (previous.end_ns if previous.end_ns is not None else previous.timestamp_ns)
           for previous, sample in zip(wave.samples, wave.samples[1:])): return "chronology"
    if not any(sample_overlaps_execution(sample, wave.execution_started, wave.execution_ended)
               for sample in wave.samples):
        return "no_execution_sample"
    return None

async def sample_during(sampler: Sampler, action: Callable[[], Awaitable[Wave]], *, interval: float = SAMPLE_INTERVAL_SECONDS,
                        max_samples: int = MAX_SAMPLES_PER_WAVE, timeout: float | None = None) -> Wave:
    """Fence pre/during/post samples around an action and join the owned sampler."""
    if interval <= 0 or type(max_samples) is not int or not 3 <= max_samples <= MAX_SAMPLES_PER_WAVE:
        raise ValueError("invalid sampling bound")
    deadline = asyncio.get_running_loop().time() + timeout if timeout is not None else None
    async def read() -> MemorySample:
        remaining = None if deadline is None else deadline - asyncio.get_running_loop().time()
        if remaining is not None and remaining <= 0: raise asyncio.TimeoutError
        return await asyncio.wait_for(sampler.sample(), remaining)
    try:
        pre = await read()                       # do not create an action task before this succeeds
    except Exception as exc:
        raise _retain_samples(exc, [], "sampler_error")
    samples, stop = [pre], asyncio.Event()
    async def collect() -> None:
        while not stop.is_set() and len(samples) < max_samples - 1:
            # The action task is created first below so this candidate gets a
            # chance to observe the provider after it has started.  Sleeping
            # before the first candidate systematically misses short waves.
            samples.append(await read())
            if not stop.is_set():
                # Keep subsequent reads bounded and interval-spaced; the
                # immediate candidate must not turn this into a busy loop.
                await asyncio.sleep(interval)
    action_task = asyncio.create_task(action())
    collector = asyncio.create_task(collect())
    try:
        remaining = None if deadline is None else max(0, deadline - asyncio.get_running_loop().time())
        done, _ = await asyncio.wait((action_task, collector), timeout=remaining,
                                     return_when=asyncio.FIRST_COMPLETED)
        if not done:
            raise _retain_samples(CapacityEvidenceError("wave timeout"), samples, "timeout")
        if collector in done:
            # Sampling is a safety gate: do not let provider execution continue
            # after an unobservable interval.
            exc = collector.exception()
            if not action_task.done():
                action_task.cancel()
                try: await action_task
                except asyncio.CancelledError: pass
            if exc is not None: raise _retain_samples(exc, samples, "sampler_error")
            raise _retain_samples(CapacityEvidenceError("sample bound reached"), samples, "sample_bound")
        try:
            result = action_task.result()
        except Exception as exc:
            # The RM runner may already have classified a bounded failure (for
            # example OOM versus a non-resource provider error).  Sampling
            # adds telemetry; it must not erase that authoritative category.
            kind = getattr(exc, "capacity_failure_kind",
                           getattr(exc, "failure_kind", "runner_error"))
            raise _retain_samples(exc, samples, kind)
    finally:
        stop.set(); collector.cancel()
        await asyncio.gather(collector, return_exceptions=True)
        if not action_task.done():
            action_task.cancel()
            try: await action_task
            except asyncio.CancelledError: pass
    try:
        post = await read()
    except Exception as exc:
        raise _retain_samples(exc, samples, "sampler_error")
    samples.append(post)
    if len(samples) < 3:
        raise _retain_samples(CapacityEvidenceError("no during-execution memory sample"),
                              samples, "no_execution_sample")
    return replace(result, samples=tuple(samples))

def _points(limit: int) -> tuple[int, ...]:
    if type(limit) is not int or not 1 <= limit <= THROUGHPUT_MAX_PARALLELISM: raise ValueError("max_parallelism must be between 1 and 16")
    # Deterministic bounded discovery: p=1, p=2, then increasing points and ceiling.
    values = {1, limit, *((limit * i + 9) // 10 for i in range(2, 10))}
    if limit >= 2: values.add(2)
    return tuple(sorted(values))

async def measure_capacity(runner: WaveRunner, sampler: Sampler, *, max_parallelism: int,
                           request_ids: Callable[[int, int], tuple[str, ...]] | None = None,
                           timeout: float = SAMPLE_TIMEOUT_SECONDS,
                            expected_max_output_tokens: int) -> CapacityResult:
    """Collect bounded candidate evidence; *never* return ``measured``.

    The ceiling is a probe limit, not a memory/resource bound.  Thus even an
    entirely successful sweep is deliberately profile-ineligible.
    """
    points_to_probe = _points(max_parallelism)
    if type(expected_max_output_tokens) is not int or expected_max_output_tokens < 1:
        raise ValueError("expected_max_output_tokens must be a positive integer")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    supplied, used_ids, serial, warmups, measured, grouped = request_ids, set(), [], [], [], {}
    ordinal = 0
    last_observation_ns = -1
    observed_total: int | None = None
    def ids(p: int, w: int, phase: str) -> tuple[str, ...]:
        nonlocal ordinal
        ordinal += 1
        value = supplied(p, w) if supplied else tuple(f"{phase}-p{p}-w{w}-{ordinal}-{i}" for i in range(p))
        if len(value) != p or len(set(value)) != p or any(type(x) is not str or not x for x in value) or used_ids.intersection(value):
            raise CapacityEvidenceError("request identities are not globally unique")
        used_ids.update(value); return value
    def result(reason: str, candidate: int | None, item: Wave | None = None) -> CapacityResult:
        return CapacityResult("incomplete", candidate, None, reason, tuple(measured), tuple(serial), tuple(warmups),
                              False, item.phase if item else None, item.failure_kind if item else None,
                              expected_max_output_tokens)
    def failed(collection: list[Wave], item: Wave, kind: str) -> Wave:
        """Retain the exact categorized wave, including native diagnostics."""
        categorized = replace(item, failed=True, failure_kind=kind)
        if collection and collection[-1] is item:
            collection[-1] = categorized
        return categorized
    def total_consistent(item: Wave) -> bool:
        nonlocal observed_total
        totals = {sample.total_bytes for sample in item.samples}
        if len(totals) != 1: return False
        total = next(iter(totals))
        if observed_total is None:
            observed_total = total
            return True
        return total == observed_total
    async def run(p: int, w: int, phase: str) -> tuple[Wave, tuple[str, ...]]:
        request_ids_for_wave = ids(p, w, phase)
        try:
            item = await sample_during(sampler, lambda: runner(p, w, request_ids_for_wave), timeout=timeout)
            return replace(item, phase=phase), request_ids_for_wave
        except (CapacityEvidenceError, asyncio.TimeoutError, RuntimeError) as exc:
            samples = getattr(exc, "capacity_samples", ())
            kind = getattr(exc, "capacity_failure_kind", "runner_error")
            return Wave(p, w, request_ids_for_wave, 0, False, 0, samples=tuple(samples),
                        failed=True, failure_kind=kind, phase=phase,
                        failure_detail=(bounded_failure_text(getattr(exc, "failure_code", ""),
                                                             getattr(exc, "failure_message", ""))
                                        if hasattr(exc, "failure_code") else None)), request_ids_for_wave
    def chronological(item: Wave) -> bool:
        nonlocal last_observation_ns
        start = item.samples[0].start_ns if item.samples and item.samples[0].start_ns is not None else (item.samples[0].timestamp_ns if item.samples else -1)
        end = item.samples[-1].end_ns if item.samples and item.samples[-1].end_ns is not None else (item.samples[-1].timestamp_ns if item.samples else -1)
        if start < last_observation_ns: return False
        last_observation_ns = end
        return True
    for w in range(1, 5):
        item, expected_ids = await run(1, w, "baseline"); serial.append(item)
        failure = _native_failure(item, 1, expected_ids, w, expected_max_output_tokens)
        if failure: return result("invalid_serial_baseline", None, failed(serial, item, failure))
        if not total_consistent(item): return result("telemetry_identity_changed", None, failed(serial, item, "identity"))
        if not chronological(item): return result("telemetry_chronology_invalid", None, failed(serial, item, "chronology"))
        if any(not s.reserve_ok for s in item.samples): return result("reserve_breached", None, failed(serial, item, "reserve"))
    for p in points_to_probe:
        warm, expected_ids = await run(p, 0, "warmup"); warmups.append(warm)
        failure = _native_failure(warm, p, expected_ids, 0, expected_max_output_tokens)
        if failure: return result("invalid_warmup", max(grouped, default=None), failed(warmups, warm, failure))
        if not total_consistent(warm): return result("telemetry_identity_changed", max(grouped, default=None), failed(warmups, warm, "identity"))
        if not chronological(warm): return result("telemetry_chronology_invalid", max(grouped, default=None), failed(warmups, warm, "chronology"))
        if any(not s.reserve_ok for s in warm.samples): return result("reserve_breached", max(grouped, default=None), failed(warmups, warm, "reserve"))
        waves = []
        for w in range(1, 5):
            item, expected_ids = await run(p, w, "measured"); waves.append(item); measured.append(item)
            failure = _native_failure(item, p, expected_ids, w, expected_max_output_tokens)
            if failure: return result("invalid_wave", max(grouped, default=None), failed(measured, item, failure))
            if not total_consistent(item): return result("telemetry_identity_changed", max(grouped, default=None), failed(measured, item, "identity"))
            if not chronological(item): return result("telemetry_chronology_invalid", max(grouped, default=None), failed(measured, item, "chronology"))
            if any(not s.reserve_ok for s in item.samples): return result("reserve_breached", max(grouped, default=None), failed(measured, item, "reserve"))
        grouped[p] = waves
    optimum = choose_optimum(grouped)
    return CapacityResult("candidate", max(grouped), optimum, "probe_limit_not_memory_bound",
                          tuple(measured), tuple(serial), tuple(warmups), False,
                           expected_max_output_tokens=expected_max_output_tokens)


async def discover_memory(runner: WaveRunner, sampler: Sampler, *, max_parallelism: int,
                          request_ids: Callable[[int, int], tuple[str, ...]] | None = None,
                          timeout: float = SAMPLE_TIMEOUT_SECONDS,
                          expected_max_output_tokens: int) -> DiscoveryResult:
    """Run the bounded, no-skip incremental memory observation protocol.

    This intentionally does not call ``measure_capacity``: discovery has no
    warmups, no throughput choice, and no implied safe/profile bound.
    """
    if type(max_parallelism) is not int or not 2 <= max_parallelism <= DISCOVERY_MAX_PARALLELISM:
        raise ValueError(f"max_parallelism must be between 2 and {DISCOVERY_MAX_PARALLELISM}")
    if type(expected_max_output_tokens) is not int or expected_max_output_tokens < 1:
        raise ValueError("expected_max_output_tokens must be a positive integer")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    baseline: list[Wave] = []; points: list[Wave] = []; used: set[str] = set()
    ordinal = 0; last_ns = -1; total: int | None = None; safe: int | None = 1

    def ids(p: int, w: int, phase: str) -> tuple[str, ...]:
        nonlocal ordinal
        ordinal += 1
        value = request_ids(p, w) if request_ids else tuple(f"{phase}-p{p}-w{w}-{ordinal}-{i}" for i in range(p))
        if (len(value) != p or len(set(value)) != p or any(type(x) is not str or not x for x in value)
                or used.intersection(value)):
            raise CapacityEvidenceError("request identities are not globally unique")
        used.update(value); return value

    def finish(reason: str, stop: int | None, phase: str | None, kind: str | None) -> DiscoveryResult:
        # A reserve breach proves the preceding completed point was observed
        # with reserve intact; other current failures invalidate the historical
        # observation rather than presenting it as a current safe point.
        observed = safe if kind == "reserve" and phase == "discovery" else None
        return DiscoveryResult("incomplete", observed, max_parallelism, stop, reason, phase, kind,
                               tuple(baseline), tuple(points), expected_max_output_tokens)

    async def run(p: int, w: int, phase: str) -> tuple[Wave, tuple[str, ...]]:
        wave_ids = ids(p, w, phase)
        try:
            item = await sample_during(sampler, lambda: runner(p, w, wave_ids), timeout=timeout)
            return replace(item, phase=phase), wave_ids
        except (CapacityEvidenceError, asyncio.TimeoutError, RuntimeError) as exc:
            return Wave(p, w, wave_ids, 0, False, 0, samples=tuple(getattr(exc, "capacity_samples", ())),
                        failed=True, failure_kind=getattr(exc, "capacity_failure_kind", "runner_error"),
                        failure_detail=(bounded_failure_text(getattr(exc, "failure_code", ""),
                                                             getattr(exc, "failure_message", ""))
                                        if hasattr(exc, "failure_code") else None),
                        phase=phase), wave_ids

    def check(item: Wave, expected_p: int, expected: tuple[str, ...], wave_no: int) -> str | None:
        nonlocal last_ns, total
        if item.failed and isinstance(item.failure_kind, str) and item.failure_kind:
            return item.failure_kind
        if (item.observation_count != 1 or item.observed_native_batch_sizes != (expected_p,)
                or item.native_request_correlation is not True or item.observation_drops != 0):
            return "native_batch_correlation"
        failure = _native_failure(item, expected_p, expected, wave_no, expected_max_output_tokens)
        if failure: return failure
        if len(item.decoder_steps) != expected_p or any(type(step) is not int or step < 0
                                                        for step in item.decoder_steps):
            return "decoder_workload"
        if any(step != expected_max_output_tokens for step in item.decoder_steps):
            return "undercovered_decoder"
        totals = {s.total_bytes for s in item.samples}
        if len(totals) != 1 or (total is not None and next(iter(totals)) != total): return "identity"
        total = next(iter(totals))
        start = item.samples[0].start_ns if item.samples[0].start_ns is not None else item.samples[0].timestamp_ns
        end = item.samples[-1].end_ns if item.samples[-1].end_ns is not None else item.samples[-1].timestamp_ns
        if start < last_ns: return "chronology"
        last_ns = end
        if any(not s.reserve_ok for s in item.samples): return "reserve"
        return None

    for w in range(1, 5):
        item, expected = await run(1, w, "baseline"); baseline.append(item)
        failure = check(item, 1, expected, w)
        if failure:
            reason = "reserve_breached" if failure == "reserve" else (failure if failure == "undercovered_decoder" else "invalid_baseline")
            baseline[-1] = replace(item, failed=True, failure_kind=failure)
            return finish(reason, 1, "baseline", failure)
    for p in range(2, max_parallelism + 1):
        for w in range(1, 5):
            item, expected = await run(p, w, "discovery"); points.append(item)
            failure = check(item, p, expected, w)
            if failure:
                reason = "reserve_breached" if failure == "reserve" else (failure if failure == "undercovered_decoder" else "invalid_discovery")
                points[-1] = replace(item, failed=True, failure_kind=failure)
                if reason == "invalid_discovery" and item.failure_detail:
                    reason = bounded_exception_text(RuntimeError(f"{reason}: {failure}:{item.failure_detail}"))
                return finish(reason, p, "discovery", failure)
        safe = p
    return DiscoveryResult("complete", safe, max_parallelism, None, "observed_through_ceiling",
                           baseline=tuple(baseline), points=tuple(points),
                           expected_max_output_tokens=expected_max_output_tokens)

def canonical_benchmark_payload(path: str | Path, *, instruction: str) -> tuple[bytes, str]:
    """Strict UTF-8 input bytes; tokenization is exclusively a child obligation."""
    with Path(path).open("rb") as stream: raw = stream.read(MAX_INPUT_FILE_BYTES + 1)
    if not raw or len(raw) > MAX_INPUT_FILE_BYTES: raise ValueError("benchmark input is empty or exceeds 256 KiB")
    text = raw.decode("utf-8")
    if not isinstance(instruction, str) or not instruction: raise ValueError("instruction is required")
    payload = json.dumps({"instruction": instruction, "texts": [text]}, ensure_ascii=True, separators=(",", ":")).encode()
    return payload, hashlib.sha256(payload).hexdigest()

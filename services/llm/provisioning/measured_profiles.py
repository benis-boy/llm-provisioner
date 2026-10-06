"""The fail-closed boundary between authoritative measurement and storage.

This module is intentionally boring: it does not measure, estimate, or repair
evidence.  It only translates a complete runner result into the existing
immutable profile-store contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import inspect
import json
from typing import Callable, Any, Mapping

from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.profiles import BenchmarkMetadata, ProfileStore
from services.llm.providers.coedit_batch import AllocatorObservation
from services.llm.providers.config import SMOLLM_MAX_PARALLELISM

from .benchmark_requests import BenchmarkRequest
from .capacity import Wave, sample_overlaps_execution
from .measurement import (AuthoritativeMeasurement, _resident_ceiling,
                           _generic_memory_summary, _resource_bound_witness,
                           _sweep_points)
try:
    from tools.compatibility.debug_trace import record as trace
except ImportError:
    def trace(*args, **kwargs):
        return None


MAX_PROVENANCE_BYTES = 64 * 1024

PERSISTENCE_VALIDATION_DETAILS = frozenset({
    "eligibility", "failure_witness", "capability_metadata", "bounds",
    "memory_summary", "resident_p2_witness_formula", "generic_formula",
    "schedule", "telemetry_evidence_chronology", "latency",
    "store_save_readback_identity",
})


class PersistenceValidationError(ValueError):
    """A safe, closed classification for a persistence invariant failure."""

    def __init__(self, detail: str):
        if detail not in PERSISTENCE_VALIDATION_DETAILS:
            raise ValueError("unknown persistence validation detail")
        self.measurement_failure_detail = detail
        super().__init__(detail)


def _invalid(detail: str, message: str) -> PersistenceValidationError:
    # ``message`` is intentionally retained only as the local traceback context;
    # the public exception text is the closed category, never provider evidence.
    error = PersistenceValidationError(detail)
    error.validation_message = message
    return error


@dataclass(frozen=True)
class MeasuredProfileIdentity:
    """Identity witnessed by provisioning, not derived from measured output."""

    model_id: ModelId
    gpu_uuid: str
    artifact_manifest_hash: str
    model_hash: str
    runtime_identity: str
    adapter_identity: str
    provenance: str
    created_at: str
    context_size: int | None = None
    bucket_identity: str | None = None

    def __post_init__(self) -> None:
        model = ModelId(self.model_id)
        values = (self.gpu_uuid, self.artifact_manifest_hash, self.model_hash,
                  self.runtime_identity, self.adapter_identity, self.provenance)
        if any(type(value) is not str or not value or len(value.encode()) > MAX_PROVENANCE_BYTES
               for value in values):
            raise _invalid("store_save_readback_identity", "identity and provenance fields must be bounded non-empty text")
        if type(self.created_at) is not str or not self.created_at:
            raise _invalid("store_save_readback_identity", "created_at is required")
        try:
            parsed = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise _invalid("store_save_readback_identity", "created_at must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise _invalid("store_save_readback_identity", "created_at must include an explicit timezone")
        if (model is ModelId.SMOLLM) != (self.context_size is not None):
            raise _invalid("store_save_readback_identity", "selector must match model")
        if (self.context_size is None) == (self.bucket_identity is None):
            raise _invalid("store_save_readback_identity", "exactly one selector is required")
        if self.context_size is not None and (type(self.context_size) is not int or self.context_size < 1):
            raise _invalid("store_save_readback_identity", "context_size must be positive")
        if self.bucket_identity is not None and (type(self.bucket_identity) is not str or not self.bucket_identity):
            raise _invalid("store_save_readback_identity", "bucket_identity must be non-empty text")


@dataclass(frozen=True)
class MeasuredProfilePersistence:
    profile: CapacityProfile
    metadata: BenchmarkMetadata
    evidence_digest: str


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _identity(request: BenchmarkRequest, identity: MeasuredProfileIdentity) -> tuple[dict[str, Any], str]:
    if ModelId(identity.model_id) is not request.model_id:
        raise _invalid("store_save_readback_identity", "model identity does not match benchmark request")
    if request.model_id is ModelId.SMOLLM:
        expected = f"smollm:context{identity.context_size}"
        if request.request_bucket != expected:
            raise _invalid("store_save_readback_identity", "context selector does not match request bucket")
    elif identity.bucket_identity != request.request_bucket:
        raise _invalid("store_save_readback_identity", "bucket selector does not match request bucket")
    data = {"model_id": ModelId(identity.model_id).value, "gpu_uuid": identity.gpu_uuid,
            "artifact_manifest_hash": identity.artifact_manifest_hash,
            "model_hash": identity.model_hash, "runtime_identity": identity.runtime_identity,
            "adapter_identity": identity.adapter_identity, "context_size": identity.context_size,
            "bucket_identity": identity.bucket_identity, "fingerprint": request.fingerprint}
    return data, hashlib.sha256(_canonical(data).encode()).hexdigest()


def _latencies(extractor: Callable[..., tuple[int, ...]], wave: Wave) -> tuple[int, ...]:
    try:
        signature = inspect.signature(extractor)
        if len(signature.parameters) >= 2:
            values = extractor(wave.request_ids, wave)
        elif len(signature.parameters) == 1:
            values = extractor(wave.request_ids)
        else:
            raise _invalid("latency", "latency extractor must be keyed by request IDs")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("latency extractor"):
            raise _invalid("latency", "latency extractor signature is not supported") from exc
        raise _invalid("latency", "latency extractor signature is not supported") from exc
    if isinstance(values, Mapping):
        if set(values) != set(wave.request_ids):
            raise _invalid("latency", "latency extractor returned the wrong request IDs")
        values = tuple(values[request_id] for request_id in wave.request_ids)
    if not isinstance(values, tuple) or len(values) != wave.successful_requests:
        raise _invalid("latency", "latency extractor returned the wrong number of values")
    if any(type(value) is not int or value < 0 for value in values):
        raise _invalid("latency", "latencies must be non-negative integers")
    return values


def _sample(wave: Wave, extractor: Callable[..., tuple[int, ...]]) -> SampleMetadata:
    if (wave.failed or not wave.outputs_valid or type(wave.elapsed_ms) is not int
            or wave.elapsed_ms <= 0 or wave.successful_requests != wave.concurrency
            or type(wave.concurrency) is not int or wave.concurrency < 1
            or not isinstance(wave.request_ids, tuple)
            or len(wave.request_ids) != wave.concurrency
            or any(type(item) is not str or not item for item in wave.request_ids)
            or len(set(wave.request_ids)) != len(wave.request_ids)
            or not wave.samples or any(not item.valid() for item in wave.samples)):
        raise _invalid("telemetry_evidence_chronology", "wave is not complete authoritative evidence")
    return SampleMetadata(wave.concurrency, wave.wave, wave.successful_requests,
                          wave.elapsed_ms, max(item.used_bytes for item in wave.samples),
                           _latencies(extractor, wave))


def _validate_evidence(
    measurement: AuthoritativeMeasurement,
    expected_output: int,
    points: tuple[int, ...],
) -> None:
    """Independently validate evidence that the measurement boundary consumed."""
    seen: set[str] = set()
    last_sample_end = last_execution_end = -1
    warmups = dict(zip(points, measurement.warmups))
    measured = {
        point: measurement.measured[index * 4:(index + 1) * 4]
        for index, point in enumerate(points)
    }
    execution_order = (*measurement.baseline, *measurement.successful_discovery,
                       *(wave for point in points
                                                 for wave in (warmups[point], *measured[point])))
    for wave in execution_order:
        if (wave.failed or not wave.outputs_valid or wave.max_output_tokens != expected_output
                or (wave.evidence_kind != "ollama_native" and wave.decoder_steps != (expected_output,) * wave.concurrency)
                 or wave.native_batch_size != (1 if wave.evidence_kind == "ollama_native" else wave.concurrency)
                or (wave.evidence_kind != "ollama_native" and
                    (not isinstance(wave.allocator, AllocatorObservation) or not wave.allocator.valid()))
                or not wave.cuda_synchronized or not isinstance(wave.request_ids, tuple)
                or len(wave.request_ids) != wave.concurrency or len(set(wave.request_ids)) != wave.concurrency
                or any(type(request_id) is not str or not request_id for request_id in wave.request_ids)
                or seen.intersection(wave.request_ids)):
            raise _invalid("telemetry_evidence_chronology", "measurement wave identity or decoder evidence is invalid")
        seen.update(wave.request_ids)
        if not (type(wave.execution_started) is int and type(wave.execution_ended) is int
                and wave.execution_started >= 0 and wave.execution_ended >= wave.execution_started):
            raise _invalid("telemetry_evidence_chronology", "measurement execution timing is invalid")
        if wave.workload_kind == "iterations" and wave.workload_witness != (1,) * wave.concurrency:
            raise _invalid("telemetry_evidence_chronology", "GECToR iteration workload evidence is invalid")
        if wave.evidence_kind == "ollama_native":
            if (wave.observation_count != wave.concurrency or
                    wave.observed_native_batch_sizes != (1,) * wave.concurrency):
                raise _invalid("telemetry_evidence_chronology", "Ollama evidence is not one native observation per request")
        elif wave.concurrency > 1 and (wave.native_request_correlation is not True
                or wave.observation_count != 1
                or wave.observed_native_batch_sizes != (wave.concurrency,) or wave.observation_drops != 0):
            raise _invalid("telemetry_evidence_chronology", "measurement native batch correlation is invalid")
        if not wave.samples or any(not sample.valid() for sample in wave.samples):
            raise _invalid("telemetry_evidence_chronology", "measurement telemetry is invalid")
        starts = [sample.start_ns if sample.start_ns is not None else sample.timestamp_ns for sample in wave.samples]
        ends = [sample.end_ns if sample.end_ns is not None else sample.timestamp_ns for sample in wave.samples]
        if (starts[0] < last_sample_end or wave.execution_started < last_execution_end
                or any(start < previous_end for start, previous_end in zip(starts[1:], ends))
                 or not any(sample_overlaps_execution(sample, wave.execution_started,
                                                       wave.execution_ended)
                            for sample in wave.samples)):
            raise _invalid("telemetry_evidence_chronology", "measurement telemetry chronology is invalid")
        last_sample_end, last_execution_end = ends[-1], wave.execution_ended


def _validate_memory_proof(measurement: AuthoritativeMeasurement, n: int,
                           *, resident_smollm: bool = False,
                           capability_exhaustive: bool = False) -> None:
    """Rebuild producer memory summaries from every retained successful wave."""
    successful = (*measurement.baseline, *measurement.successful_discovery,
                  *measurement.warmups, *measurement.measured)
    if not successful or any(w.failed or not w.samples or any(not s.valid() or not s.reserve_ok
                                                               for s in w.samples)
                             for w in successful):
        raise _invalid("memory_summary", "memory proof evidence is incomplete")
    totals = {s.total_bytes for w in successful for s in w.samples}
    if len(totals) != 1 or measurement.total_vram_bytes != next(iter(totals)):
        raise _invalid("memory_summary", "total VRAM is not reconstructed from retained evidence")
    baseline_pre = max(w.samples[0].used_bytes for w in measurement.baseline)
    if measurement.baseline_pre_used_bytes != baseline_pre:
        raise _invalid("memory_summary", "baseline memory is not reconstructed from retained evidence")
    p2_discovery = tuple(w for w in measurement.successful_discovery
                         if w.phase == "discovery" and w.concurrency == 2)
    resident_p2 = (resident_smollm and measurement.n is not None and measurement.n >= 2 and
                   len(p2_discovery) == 4)
    p2_delta = (max(max(s.used_bytes for s in w.samples) - baseline_pre
                    for w in p2_discovery) if resident_p2 else None)
    resident_formula = resident_p2 and p2_delta > 0
    if resident_formula:
        # Resident SmolLM's denominator is an immutable witness: the first
        # complete p=2 discovery point, not an arbitrary later sweep peak.
        try:
            formula_ceiling = _resident_ceiling(
                measurement.total_vram_bytes, baseline_pre, p2_delta)
        except Exception as exc:
            raise _invalid("resident_p2_witness_formula", "resident memory formula is invalid") from exc
        # The producer persists the operational formula bound after applying
        # the immutable identity-derived capability.  This is distinct from
        # an operator ceiling and is reconstructable from the result; rejecting
        # an otherwise valid capability-32 exhaustive run because the raw
        # memory formula is larger would make producer and persistence disagree.
        expected_resident = min(formula_ceiling, measurement.supported_parallelism)
        if measurement.derived_ceiling != expected_resident or n != expected_resident:
            raise _invalid("resident_p2_witness_formula", "resident memory bound is not reconstructed from p=2 evidence")
        peak = p2_delta
    elif resident_p2:
        # Resident p=1 request/sweep overhead is retained evidence, but it is
        # not a per-slot denominator.  Reconstruct the producer's frozen p=2
        # witness independently.
        peak = p2_delta
    else:
        peak = _generic_memory_summary(successful, baseline_pre)
    expected_peak = measurement.peak_incremental_request_bytes
    if expected_peak is None:
        if peak != 0:
            raise _invalid("memory_summary", "incremental memory is not reconstructed from retained evidence")
    elif expected_peak != peak:
        raise _invalid("memory_summary", "incremental memory is not reconstructed from retained evidence")

    witness = measurement.resource_bound_failure
    if (resident_p2 and not resident_formula and not capability_exhaustive
            and not _resource_bound_witness(witness, n)):
        raise _invalid("resident_p2_witness_formula", "zero-increment resident path requires a resource-bound witness")
    if resident_formula and witness is not None:
        raise _invalid("resident_p2_witness_formula", "resident formula and resource-bound paths are mutually exclusive")
    if witness is not None:
        if not _resource_bound_witness(witness, n):
            raise _invalid("resident_p2_witness_formula", "resource-bound witness is invalid")
        if {s.total_bytes for s in witness.samples} != totals:
            raise _invalid("telemetry_evidence_chronology", "resource-bound witness has inconsistent device telemetry")
        if any(s.used_bytes < 0 or s.used_bytes > next(iter(totals)) for s in witness.samples):
            raise _invalid("telemetry_evidence_chronology", "resource-bound witness has invalid device telemetry")
        retained_end = max(w.execution_ended for w in (*measurement.baseline,
                                                        *measurement.successful_discovery))
        sweep_start = min(w.execution_started for w in (*measurement.warmups,
                                                         *measurement.measured))
        if witness.execution_started <= retained_end or witness.execution_ended >= sweep_start:
            raise _invalid("telemetry_evidence_chronology", "resource-bound witness chronology is invalid")
        witness_starts = [s.start_ns if s.start_ns is not None else s.timestamp_ns
                          for s in witness.samples]
        witness_ends = [s.end_ns if s.end_ns is not None else s.timestamp_ns
                        for s in witness.samples]
        if any(a < b for a, b in zip(witness_starts[1:], witness_ends)) or not any(
                sample_overlaps_execution(sample, witness.execution_started,
                                          witness.execution_ended)
                for sample in witness.samples):
            raise _invalid("telemetry_evidence_chronology", "resource-bound witness telemetry chronology is invalid")


def persist_measured_profile(
    request: BenchmarkRequest,
    measurement: AuthoritativeMeasurement,
    identity: MeasuredProfileIdentity,
    store: ProfileStore,
    *,
    latency_extractor: Callable[..., tuple[int, ...]],
) -> MeasuredProfilePersistence:
    """Validate all evidence, save once, then prove exact durable readback."""
    trace("persistence", "validation", "enter", model=request.model_id.value if isinstance(request, BenchmarkRequest) else None)
    if not isinstance(request, BenchmarkRequest) or not isinstance(measurement, AuthoritativeMeasurement):
        raise TypeError("request and measurement must be immutable provisioning contracts")
    if not isinstance(store, ProfileStore) or not callable(latency_extractor):
        raise TypeError("a ProfileStore and authoritative latency extractor are required")
    if (type(measurement.status) is not str or measurement.status != "complete"
            or measurement.profile_eligible is not True
            or measurement.benchmark_fingerprint != request.fingerprint
            or measurement.reserve_percent != 20):
        raise _invalid("eligibility", "measurement is not eligible for immutable persistence")
    if measurement.failed_discovery or measurement.resource_bound_failure is not None:
        failure = measurement.resource_bound_failure
        if (failure is None or measurement.failed_discovery != (failure,) or
                not _resource_bound_witness(failure, measurement.n)):
            raise _invalid("failure_witness", "only an explicit resource-bound N+1 failure may accompany persistence")
    if (measurement.supported_parallelism is None
            or measurement.supported_parallelism < 1
            or measurement.supported_parallelism > SMOLLM_MAX_PARALLELISM):
        raise _invalid("capability_metadata", "identity-derived parallelism capability is missing")
    if measurement.parallelism_bound_reason not in {
            "identity_configured_provider_capability",
            "identity_configured_operator_ceiling",
        }:
        raise _invalid("capability_metadata", "parallelism-one profile lacks its identity-derived capability reason")
    if any(type(value) is not int for value in (measurement.n, measurement.optimum, measurement.m)):
        raise _invalid("bounds", "measurement bounds are incomplete")
    n, optimum, buffer = measurement.n, measurement.optimum, measurement.m
    if not 1 <= n <= 32 or not 1 <= optimum <= n or buffer != optimum:
        raise _invalid("bounds", "measurement bounds are inconsistent")
    if measurement.total_vram_bytes is None or measurement.baseline_pre_used_bytes is None:
        raise _invalid("memory_summary", "memory ceiling telemetry is incomplete")
    capability_bound = measurement.parallelism_bound_reason == "identity_configured_provider_capability"
    capability_exhaustive = (capability_bound
        and measurement.configured_ceiling == measurement.supported_parallelism
        and measurement.n == measurement.supported_parallelism
        and tuple((w.phase, w.concurrency, w.wave) for w in measurement.successful_discovery)
        == tuple(("discovery", p, wave) for p in range(2, measurement.supported_parallelism + 1)
                 for wave in range(1, 5)))
    resource_bound = _resource_bound_witness(measurement.resource_bound_failure, n)
    p2_discovery = tuple(w for w in measurement.successful_discovery
                         if w.phase == "discovery" and w.concurrency == 2)
    exact_baseline = (tuple((w.phase, w.concurrency, w.wave)
                            for w in measurement.baseline)
                      == tuple(("baseline", 1, wave) for wave in range(1, 5)))
    exact_p2 = (tuple((w.phase, w.concurrency, w.wave) for w in p2_discovery)
                == tuple(("discovery", 2, wave) for wave in range(1, 5)))
    resident_mode = request.model_id is ModelId.SMOLLM
    resident_p2 = (resident_mode and n >= 2 and
                   exact_p2 and exact_baseline)
    resident_p2_delta = (max(max(s.used_bytes for s in w.samples) - measurement.baseline_pre_used_bytes
                             for w in p2_discovery) if resident_p2 else None)
    resident_formula = resident_p2 and resident_p2_delta > 0
    if resident_p2 and resident_formula and resource_bound:
        raise _invalid("resident_p2_witness_formula", "resident formula and resource-bound paths are mutually exclusive")
    if resident_p2 and not resident_formula and not resource_bound and not capability_exhaustive:
        raise _invalid("resident_p2_witness_formula", "zero-increment resident path requires a resource-bound witness")
    if resident_mode and not resident_formula:
        # This exception is deliberately narrow: only the resident SmolLM
        # exhaustive discovery path may persist without a formula, and it must
        # retain the failed N+1 wave that proves the boundary.
        if (not (resource_bound or capability_exhaustive)
                or measurement.derived_ceiling is not None
                or (not capability_exhaustive and
                    measurement.parallelism_bound_reason != "identity_configured_operator_ceiling")):
            raise _invalid("memory_summary", "memory ceiling telemetry is incomplete")
    elif not resident_mode:
        # Native non-resident batches have aggregate, variable cost.  Their
        # retained memory summary is evidence for replay only; it is never a
        # per-slot formula.  N is established solely by capability exhaustion
        # or a retained typed N+1 resource witness.  GECToR's fixed p=1
        # capability is the degenerate exhaustive case with no p=2 wave.
        if measurement.derived_ceiling is not None or not (capability_exhaustive or resource_bound):
            raise _invalid("generic_formula", "native aggregate capacity proof is not exhaustive")
        if n != measurement.supported_parallelism and not resource_bound:
            raise _invalid("generic_formula", "native capability was not exhausted")
    elif resident_mode and n == 1:
        derived = (measurement.total_vram_bytes * 80 // 100 - measurement.baseline_pre_used_bytes) // measurement.peak_incremental_request_bytes
        if resident_formula:
            p2 = measurement.successful_discovery[:4]
            if (tuple(w.concurrency for w in p2) != (2,) * 4 or
                    tuple(w.phase for w in p2) != ("discovery",) * 4 or
                    any(max(s.used_bytes for s in w.samples) <= measurement.baseline_pre_used_bytes
                        for w in p2)):
                raise _invalid("resident_p2_witness_formula", "exact resident p=2 discovery witness is required")
            derived = min(_resident_ceiling(
                measurement.total_vram_bytes,
                measurement.baseline_pre_used_bytes,
                resident_p2_delta), measurement.supported_parallelism)
        if (derived < 1 or measurement.derived_ceiling != derived or
                (n != derived and measurement.resource_bound_failure is None and not capability_bound)):
            raise _invalid("generic_formula", "memory-safe N is not independently established from telemetry")
    # Provider capability remains an exact identity-derived bound.  An
    # operator ceiling, however, may persist the memory-safe formula result
    # below the configured/support capability.
    if (resident_mode and not resident_formula and capability_bound and not capability_exhaustive
            and n != measurement.supported_parallelism):
        raise _invalid("capability_metadata", "parallelism-one profile lacks its identity-derived capability reason")
    if n > measurement.supported_parallelism:
        raise _invalid("generic_formula", "memory-safe N is not independently established from telemetry")
    points = _sweep_points(n)
    expected_baseline = tuple((1, wave) for wave in range(1, 5))
    expected_warmups = tuple((point, 0) for point in points)
    expected_measured = tuple((point, wave) for point in points for wave in range(1, 5))
    expected_discovery = tuple(("discovery", point, wave)
                               for point in range(2, n + 1) for wave in range(1, 5))
    if (tuple((w.phase, w.concurrency, w.wave) for w in measurement.baseline)
            != tuple(("baseline", p, wave) for p, wave in expected_baseline)
            or tuple((w.phase, w.concurrency, w.wave) for w in measurement.warmups)
            != tuple(("warmup", p, wave) for p, wave in expected_warmups)
             or tuple((w.phase, w.concurrency, w.wave) for w in measurement.measured)
             != tuple(("measured", p, wave) for p, wave in expected_measured)
             or tuple((w.phase, w.concurrency, w.wave) for w in measurement.successful_discovery)
             != expected_discovery):
        raise _invalid("schedule", "measurement schedules are not the exact bounded schedules")
    _validate_memory_proof(measurement, n, resident_smollm=request.model_id is ModelId.SMOLLM,
                           capability_exhaustive=capability_exhaustive)
    _validate_evidence(measurement, max(w.max_output_tokens or 0 for w in measurement.baseline), points)
    extractor = latency_extractor
    baseline = tuple(_sample(w, extractor) for w in measurement.baseline)
    warmups = tuple(_sample(w, extractor) for w in measurement.warmups)
    measured = tuple(_sample(w, extractor) for w in measurement.measured)
    _, profile_id = _identity(request, identity)
    profile = CapacityProfile(ModelId(identity.model_id), identity.gpu_uuid,
                              identity.artifact_manifest_hash, identity.model_hash,
                              identity.runtime_identity, identity.adapter_identity,
                              profile_id, optimum, n, buffer, 20,
                              baseline + warmups + measured, identity.context_size,
                              identity.bucket_identity)
    representative = (f"context:{identity.context_size}" if identity.context_size is not None
                      else identity.bucket_identity)
    metadata = BenchmarkMetadata(request.fingerprint, identity.created_at,
                                  identity.provenance, baseline, warmups, measured,
                                  representative)
    try:
        store.save_measured(profile, metadata)
    except ValueError as exc:
        raise _invalid("store_save_readback_identity", "profile store save failed") from exc
    kwargs = ({"context_size": identity.context_size} if identity.context_size is not None
              else {"bucket_identity": identity.bucket_identity})
    try:
        saved = store.lookup(profile.model_id, identity.gpu_uuid, identity.artifact_manifest_hash,
                             identity.model_hash, identity.runtime_identity, identity.adapter_identity,
                             **kwargs)
    except ValueError as exc:
        raise _invalid("store_save_readback_identity", "profile store readback failed") from exc
    if saved is None or saved.profile_identity != profile.profile_identity or saved != profile:
        raise _invalid("store_save_readback_identity", "profile store readback did not equal saved profile")
    # ProfileStore exposes exact profile rehydration but no public metadata lookup.
    # The digest makes the returned profile+metadata evidence auditable without
    # claiming metadata readback that this contract cannot provide.
    # Discovery is deliberately not operational profile metadata, but it is
    # part of the durable attestation: without it a late-derived memory proof
    # could not be independently replayed from the returned persistence
    # boundary.
    evidence_digest = hashlib.sha256(_canonical({"profile": asdict(profile),
                                                  "metadata": asdict(metadata),
                                                  "memory_proof": {
                                                      "total_vram_bytes": measurement.total_vram_bytes,
                                                      "baseline_pre_used_bytes": measurement.baseline_pre_used_bytes,
                                                      "peak_incremental_request_bytes": measurement.peak_incremental_request_bytes,
                                                      "discovery": [asdict(w) for w in measurement.successful_discovery],
                                                  }}).encode()).hexdigest()
    return MeasuredProfilePersistence(saved, metadata, evidence_digest)

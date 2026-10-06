"""Fail-closed translation of provider observations into :class:`Wave`.

The ResourceManager remains the execution authority.  These functions only
consume observations attached to the bounded RM completion events; they never
call a provider or manufacture a serial observation.
"""
from __future__ import annotations

import json
from typing import Any

from services.llm.providers.coedit_batch import AllocatorObservation
from .capacity import CapacityEvidenceError, MemorySample, Wave


class ClassifiedCapacityEvidenceError(CapacityEvidenceError):
    """A bounded, machine-readable rejection of provider evidence."""

    _allowed = {
        "native_overlap_missing", "telemetry_correlation_failed",
        "telemetry_contract_failed", "wave_event_contract_failed", "wave_timeout",
    }

    def __init__(self, classification: str):
        if classification not in self._allowed:
            raise ValueError("unsupported evidence classification")
        super().__init__(classification)
        self.failure_kind = classification
        self.failure_code = ""
        self.failure_message = classification


def _value(event: Any, name: str) -> Any:
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def _observation(event: Any) -> Any:
    value = _value(event, "observation")
    if value is None:
        value = _value(event, "native_observation")
    return value


def _allocator(value: Any) -> AllocatorObservation:
    if isinstance(value, AllocatorObservation):
        result = value
    elif isinstance(value, dict) and set(value) == {
            "baseline_allocated", "baseline_reserved", "peak_allocated",
            "peak_reserved", "final_allocated", "final_reserved"}:
        result = AllocatorObservation(*(value[key] for key in (
            "baseline_allocated", "baseline_reserved", "peak_allocated",
            "peak_reserved", "final_allocated", "final_reserved")))
    else:
        raise ValueError("native allocator observation is missing")
    if not result.valid():
        raise ValueError("native allocator observation is invalid")
    return result


def _payload_valid(event: Any, request_id: str) -> bool:
    result = _value(event, "result")
    if not isinstance(result, bytes) or not result:
        return False
    # Results are intentionally bounded and must be parseable for transformer
    # workers.  SmolLM is opaque text, but still requires non-empty bytes.
    if result[:1] in (b"{", b"["):
        try:
            json.loads(result)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
    observed_ids = _value(event, "request_ids")
    return observed_ids is None or request_id in tuple(observed_ids)


def extract_native_wave(model: str, concurrency: int, wave: int,
                        request_ids: tuple[str, ...], events: tuple[object, ...],
                        elapsed_ms: int, *, expected_output_tokens: int) -> Wave:
    """Extract one native adapter wave, rejecting serial/provider-less evidence."""
    if type(concurrency) is not int or concurrency < 1 or len(request_ids) != concurrency:
        raise ValueError("wave concurrency and request IDs are not exact")
    completions = [event for event in events
                   if _value(event, "kind") in ("response_finished", "RESPONSE_FINISHED")]
    if len(completions) != concurrency:
        raise ValueError("wave did not contain every terminal completion")
    observations = [_observation(event) for event in events if _observation(event) is not None]
    # One native batch observation is attached to each response carrying that
    # batch's result.  Collapse those transport copies, but reject conflicting
    # observations rather than allowing one request to hide another.
    unique = []
    for observation in observations:
        if observation not in unique:
            unique.append(observation)
    observations = unique
    if len(observations) != 1 or not isinstance(observations[0], dict):
        raise ValueError("exactly one native execution observation is required")
    obs = observations[0]
    required = {"batch_size", "execution_started", "execution_ended",
                "cuda_synchronized", "allocator", "decoder_steps",
                "max_output_tokens", "request_ids"}
    if set(obs) != required or obs["batch_size"] != concurrency or tuple(obs["request_ids"]) != request_ids:
        raise ValueError("native observation does not correlate to the wave")
    steps = tuple(obs["decoder_steps"])
    if (obs["max_output_tokens"] != expected_output_tokens or len(steps) != concurrency
            or steps != (expected_output_tokens,) * concurrency):
        raise ValueError("maximum decoder workload was not observed")
    if (obs["cuda_synchronized"] is not True or type(obs["execution_started"]) is not int
            or type(obs["execution_ended"]) is not int
            or obs["execution_ended"] < obs["execution_started"]):
        raise ValueError("native execution timing is invalid")
    for request_id in request_ids:
        matching = [event for event in completions if _value(event, "request_id") == request_id]
        if len(matching) != 1 or not _payload_valid(matching[0], request_id):
            raise ValueError("completion output or request identity is invalid")
    latency = max(0, (obs["execution_ended"] - obs["execution_started"]) // 1_000_000)
    return Wave(concurrency, wave, request_ids, elapsed_ms, True, concurrency,
                obs["execution_started"], obs["execution_ended"], True, allocator=_allocator(obs["allocator"]),
                observation_count=1, observed_native_batch_sizes=(concurrency,),
                native_request_correlation=True, observation_drops=0,
                decoder_steps=steps, max_output_tokens=expected_output_tokens,
                request_latency_ms=(latency,) * concurrency)


def smollm_evidence_extractor(concurrency, wave, request_ids, events, elapsed_ms):
    completions = [event for event in events if _value(event, "kind") in ("response_finished", "RESPONSE_FINISHED")]
    observations = [_observation(event) for event in completions if _observation(event) is not None]
    required = {"kind", "request_id", "execution_started", "execution_ended",
                "configured_num_predict", "configured_num_ctx",
                "configured_temperature", "prompt_eval_count", "eval_count",
                "prompt_eval_duration", "eval_duration", "load_duration",
                "total_duration", "done_reason"}
    if len(observations) != concurrency or any(
            not isinstance(item, dict) or set(item) != required or
            item.get("kind") != "ollama_generate" for item in observations):
        raise ClassifiedCapacityEvidenceError("telemetry_contract_failed")
    if any(_value(event, "request_id") != observation.get("request_id")
           for event in completions
           for observation in (_observation(event),)
           if isinstance(observation, dict)):
        raise ClassifiedCapacityEvidenceError("telemetry_correlation_failed")
    by_id = {item.get("request_id"): item for item in observations}
    if len(by_id) != concurrency or any(type(key) is not str for key in by_id) or set(by_id) != set(request_ids):
        raise ClassifiedCapacityEvidenceError("telemetry_correlation_failed")
    intervals = [(item["execution_started"], item["execution_ended"]) for item in observations]
    if any(type(start) is not int or type(end) is not int or end < start for start, end in intervals):
        raise ClassifiedCapacityEvidenceError("telemetry_contract_failed")
    if concurrency > 1 and not any(a < d and c < b for a, b in intervals for c, d in intervals if (a, b) != (c, d)):
        raise ClassifiedCapacityEvidenceError("native_overlap_missing")
    if any(type(item.get(key)) is not int or item[key] < 0 for item in observations
           for key in ("prompt_eval_count", "eval_count", "prompt_eval_duration",
                       "eval_duration", "load_duration", "total_duration")):
        raise ClassifiedCapacityEvidenceError("telemetry_contract_failed")
    if any(type(item["configured_num_predict"]) is not int or
           item["configured_num_predict"] != 64 or
           type(item["configured_num_ctx"]) is not int or
           item["configured_num_ctx"] != 512 or
           type(item["configured_temperature"]) not in (int, float) or
           isinstance(item["configured_temperature"], bool) or
           item["configured_temperature"] != 0 or
           type(item["eval_count"]) is not int or
           not 1 <= item["eval_count"] <= item["configured_num_predict"]
           for item in observations):
        raise ClassifiedCapacityEvidenceError("telemetry_contract_failed")
    known_done_reasons = {"stop", "length", "load", "unload"}
    if any(item["done_reason"] is not None and
           (not isinstance(item["done_reason"], str) or
            item["done_reason"] not in known_done_reasons)
           for item in observations):
        raise ClassifiedCapacityEvidenceError("telemetry_contract_failed")
    # Ollama has no allocator witness.  Memory/NVML overlap is supplied by the
    # measurement sampler and validated by the authoritative measurement.
    latency_by_id = {item["request_id"]: max(0, (item["execution_ended"] - item["execution_started"]) // 1_000_000) for item in observations}
    return Wave(concurrency, wave, request_ids, elapsed_ms, True, 1,
                min(start for start, _ in intervals), max(end for _, end in intervals),
                True, observation_count=concurrency,
                observed_native_batch_sizes=(1,) * concurrency,
                native_request_correlation=True, observation_drops=0,
                decoder_steps=tuple(item["eval_count"] for item in observations),
                max_output_tokens=64, evidence_kind="ollama_native",
                workload_kind="ollama_tokens",
                workload_witness=tuple(item["eval_count"] for item in observations),
                request_latency_ms=tuple(latency_by_id[item] for item in request_ids))


def coedit_evidence_extractor(concurrency, wave, request_ids, events, elapsed_ms):
    return extract_native_wave("CoEdIT", concurrency, wave, request_ids, events, elapsed_ms,
                               expected_output_tokens=64)


def gector_evidence_extractor(concurrency, wave, request_ids, events, elapsed_ms):
    # The configured GECToR bucket is a genuine native batch of one.
    if concurrency != 1:
        raise ValueError("GECToR evidence is only proved for native batch one")
    result = extract_native_wave("GECToR", concurrency, wave, request_ids, events, elapsed_ms,
                                 expected_output_tokens=1)
    return Wave(**{**result.__dict__, "workload_kind": "iterations", "workload_witness": (1,)})

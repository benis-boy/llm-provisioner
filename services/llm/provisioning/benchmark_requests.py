"""Prepare bounded, identity-bound requests for offline capacity work.

This module deliberately does not execute an adapter.  A caller may supply a
safe adapter generator and validator; when either is unavailable the only
permitted alternative is an explicitly configured request.  The returned
object contains a compact canonical request and identity digest, not model
outputs or an unbounded JSON document.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping

from services.llm.queue.contracts import ModelId
from .contracts import CapacityBucket, ModelConfig

MAX_REQUEST_BYTES = 256 * 1024
SMOLLM_MAX_RAW_BYTES = 256
SMOLLM_PROMPT_BYTES = 448
_SMOLLM_PREFIX = "<|im_start|>user\n"
_SMOLLM_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def canonical_measurement_fixtures() -> dict[str, Any]:
    """Return the repository-owned, non-secret Phase 5 request fixtures.

    Keep this data boring on purpose: it is an identity-bearing workload, not a
    prompt corpus.  Its character lengths do *not* attest tokenizer lengths;
    adapter-specific maximum attestation remains the job of the loaded adapter
    during measurement.
    """
    return {
        "SmolLM": "A" * SMOLLM_MAX_RAW_BYTES,
        "CoEdIT": {"instruction": "F" * 16, "texts": ["T" * 128]},
        "GECToR": {"texts": ["T" * 128], "keep_confidence": 0.0,
                   "min_error_prob": 0.0, "n_iteration": 1, "batch_size": 1},
    }


class BenchmarkRequestError(ValueError):
    """The request or its exact workload identity cannot be proved."""


def replace_coedit_with_runtime_generated(
    seed: "BenchmarkRequest", generated: Any,
) -> "BenchmarkRequest":
    """Build the immutable exact-token request returned by the loaded worker.

    The prepared request remains the authenticated seed.  Only the bounded
    worker result may replace its measurement payload, and the replacement's
    identity retains both the seed fingerprint and the generation provenance.
    """
    if seed.model_id is not ModelId.COEDIT or seed.source != "configured":
        raise BenchmarkRequestError("CoEdIT runtime generation requires a configured seed")
    body = _mapping(generated, {"text", "count", "max", "fingerprint"},
                    "CoEdIT generated benchmark response")
    if type(body["count"]) is not int or body["count"] != 128:
        raise BenchmarkRequestError("CoEdIT generated token count is not exact")
    if type(body["max"]) is not int or body["max"] != 128:
        raise BenchmarkRequestError("CoEdIT generated maximum is not exact")
    text = _text(body["text"], "CoEdIT generated text")
    if len(text.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise BenchmarkRequestError("CoEdIT generated text exceeds request bound")
    if (not isinstance(body["fingerprint"], str)
            or len(body["fingerprint"]) != 64
            or any(char not in "0123456789abcdef" for char in body["fingerprint"])):
        raise BenchmarkRequestError("CoEdIT generated fingerprint is malformed")
    try:
        seed_body = json.loads(seed.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkRequestError("CoEdIT seed payload is not canonical JSON") from exc
    if (not isinstance(seed_body, dict)
            or set(seed_body) != {"instruction", "texts"}
            or not isinstance(seed_body["texts"], list)
            or len(seed_body["texts"]) != 1):
        raise BenchmarkRequestError("CoEdIT seed payload is malformed")
    payload = _canonical({"instruction": seed_body["instruction"], "texts": [text]})
    digest = hashlib.sha256(payload).hexdigest()
    if body["fingerprint"] != digest:
        raise BenchmarkRequestError("CoEdIT generated fingerprint is inconsistent")
    identity = dict(_json_value(seed.identity))
    identity["runtime_generation"] = {
        "source": "coedit_worker_benchmark_input",
        "seed_request_fingerprint": seed.fingerprint,
        "generated_payload_sha256": digest,
    }
    return BenchmarkRequest(ModelId.COEDIT, seed.request_bucket, payload,
                            _request_fingerprint(identity, payload), identity,
                            "runtime_generated")


@dataclass(frozen=True)
class BenchmarkRequest:
    model_id: ModelId
    request_bucket: str
    payload: bytes
    fingerprint: str
    identity: Mapping[str, Any]
    source: str

    def __post_init__(self) -> None:
        # ``frozen=True`` only protects the dataclass attributes.  Snapshot the
        # identity before checking its attestation so a caller constructing a
        # BenchmarkRequest directly cannot mutate nested witnesses afterwards.
        frozen_identity = _freeze(self.identity)
        object.__setattr__(self, "identity", frozen_identity)
        expected = _request_fingerprint(frozen_identity, self.payload)
        if self.fingerprint != expected:
            raise BenchmarkRequestError("benchmark request fingerprint is inconsistent")


def _finite_number(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BenchmarkRequestError(f"{name} must be numeric and non-boolean")
    if isinstance(value, float) and not math.isfinite(value):
        raise BenchmarkRequestError(f"{name} must be finite")


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BenchmarkRequestError("identity is not canonical JSON") from exc


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _mapping(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise BenchmarkRequestError(f"{name} has unknown or missing fields")
    return dict(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkRequestError(f"{name} must be non-empty text")
    return value


def _request_payload(model: ModelId, request: Any, bucket: CapacityBucket | None) -> bytes:
    if model is ModelId.SMOLLM:
        text = _text(request, "SmolLM representative request")
        raw = text.encode("utf-8")
        framed = (_SMOLLM_PREFIX + text + _SMOLLM_SUFFIX).encode("utf-8")
        if (len(raw) > SMOLLM_MAX_RAW_BYTES or len(framed) > SMOLLM_PROMPT_BYTES
                or any(ord(char) < 0x20 or ord(char) > 0x7e for char in text)):
            raise BenchmarkRequestError("SmolLM request is over the proved text bucket")
        # Ollama's normal adapter boundary accepts the bounded printable text,
        # not the transformer JSON envelope.  Keep the exact bytes in the
        # fingerprint so the RM submits the same maximum request that was
        # validated.
        return text.encode("ascii")

    if not isinstance(request, (str, Mapping)):
        raise BenchmarkRequestError("transformer representative request must be JSON or an object")
    if isinstance(request, str):
        if len(request.encode("utf-8")) > MAX_REQUEST_BYTES:
            raise BenchmarkRequestError("representative request exceeds JSON size bound")
        try:
            request = json.loads(request, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise BenchmarkRequestError("representative request is not finite JSON") from exc

    if model is ModelId.COEDIT:
        body = _mapping(request, {"instruction", "texts"}, "CoEdIT request")
        _text(body["instruction"], "instruction")
        if not isinstance(body["texts"], list) or len(body["texts"]) != 1:
            raise BenchmarkRequestError("CoEdIT requires exactly one text")
        _text(body["texts"][0], "text")
    else:
        body = _mapping(request, {"texts", "keep_confidence", "min_error_prob", "n_iteration", "batch_size"}, "GECToR request")
        if not isinstance(body["texts"], list) or len(body["texts"]) != 1:
            raise BenchmarkRequestError("GECToR requires exactly one text")
        _text(body["texts"][0], "text")
        _finite_number(body["keep_confidence"], "keep_confidence")
        _finite_number(body["min_error_prob"], "min_error_prob")
        if bucket is None or type(body["n_iteration"]) is not int or body["n_iteration"] != bucket.max_iterations:
            raise BenchmarkRequestError("GECToR iteration identity does not match the bucket")
        if type(body["batch_size"]) is not int or tuple(bucket.native_batch_shape) != (body["batch_size"],):
            raise BenchmarkRequestError("GECToR native batch identity does not match the bucket")
    payload = _canonical(body)
    if len(payload) > MAX_REQUEST_BYTES:
        raise BenchmarkRequestError("representative request exceeds JSON size bound")
    return payload


def _invoke_once(callback: Callable[..., Any], *available: Any) -> Any:
    """Select a supported callback shape without retrying callback execution."""
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError) as exc:
        raise BenchmarkRequestError("adapter callback signature cannot be inspected") from exc
    for count in range(len(available), 0, -1):
        try:
            signature.bind(*available[:count])
        except TypeError:
            continue
        return callback(*available[:count])
    raise BenchmarkRequestError("adapter callback has an unsupported signature")


def _invoke_validator(callback: Callable[..., Any], candidate: Any, config: ModelConfig,
                      bucket: CapacityBucket | None, request_bucket: str) -> Any:
    """Invoke the validator once with the selected bucket identity attached.

    Unlike generation, validation must not be allowed to silently fall back to
    a callback shape that cannot observe the selected request bucket.  That
    would permit a SmolLM validator to attest one context while preparation
    returned another.
    """
    try:
        signature = inspect.signature(callback)
        signature.bind(candidate, config, bucket, request_bucket)
    except (TypeError, ValueError) as exc:
        raise BenchmarkRequestError(
            "adapter validator must accept request, config, bucket, and request_bucket"
        ) from exc
    return callback(candidate, config, bucket, request_bucket)


def prepare_benchmark_request(
    config: ModelConfig,
    *,
    request_bucket: str,
    identity_witnesses: Mapping[str, Any],
    configured_request: Any = None,
    generate_request: Callable[..., Any] | None = None,
    validate_request: Callable[..., Any] | None = None,
) -> BenchmarkRequest:
    """Prepare one maximum representative request without running inference.

    ``generate_request`` is trusted only after ``validate_request`` proves the
    adapter-specific maximum.  A configured request is the explicit fallback;
    absent both paths, preparation fails closed.
    """
    model = ModelId(config.model_id)
    bucket = next((item for item in config.buckets if request_bucket == _bucket_id(model, item)), None)
    if model is ModelId.SMOLLM:
        if request_bucket not in {f"smollm:context{size}" for size in config.context_size_estimates}:
            raise BenchmarkRequestError("unknown SmolLM request bucket")
    elif bucket is None:
        raise BenchmarkRequestError("request bucket is not configured")
    if validate_request is None:
        raise BenchmarkRequestError("maximum request validation is required")
    if not isinstance(identity_witnesses, Mapping) or not identity_witnesses:
        raise BenchmarkRequestError("identity witnesses are required")
    identity = _identity(config, request_bucket, bucket, identity_witnesses)

    source = "generated"
    if generate_request is not None:
        try:
            candidate = _invoke_once(generate_request, config, bucket)
        except Exception as exc:
            raise BenchmarkRequestError("maximum request generation failed") from exc
    elif configured_request is not None:
        candidate = configured_request
        source = "configured"
    else:
        raise BenchmarkRequestError("safe generation unavailable and configured fallback is missing")

    payload = _request_payload(model, candidate, bucket)
    # Structural checks above are only a gate.  The adapter must attest exact
    # maximum validity (including token/context work) for this config/bucket.
    try:
        checked = _invoke_validator(validate_request, candidate, config, bucket, request_bucket)
    except Exception as exc:
        raise BenchmarkRequestError("request maximum validation failed") from exc
    if checked is not True:
        raise BenchmarkRequestError("request does not prove the configured maximum")
    fingerprint = _request_fingerprint(identity, payload)
    return BenchmarkRequest(model, request_bucket, payload, fingerprint, _freeze(identity), source)


def _bucket_id(model: ModelId, bucket: CapacityBucket) -> str:
    if not bucket.native_batch_shape:
        raise BenchmarkRequestError("native batch shape must not be empty")
    if model is ModelId.COEDIT:
        parameters = bucket.generation.parameters
        return (f"coedit:p{bucket.native_batch_shape[0]}:input{bucket.max_input_tokens}:"
                f"output{bucket.max_output_tokens}:{bucket.generation.dtype}:"
                f"beams{parameters.get('num_beams', '')}:nosample")
    parameters = bucket.generation.parameters
    return (f"gector:p{bucket.native_batch_shape[0]}:tokens{bucket.max_input_tokens}:"
            f"keep{_compact_number(parameters.get('keep_confidence', ''))}:min{_compact_number(parameters.get('min_error_prob', ''))}:"
            f"iterations{bucket.max_iterations}:batch{bucket.native_batch_shape[0]}:{bucket.generation.dtype}")


def configured_request_buckets(config: ModelConfig) -> tuple[str, ...]:
    """Return every selector from the production-owned configuration exactly once."""
    model = ModelId(config.model_id)
    if model is ModelId.SMOLLM:
        values = tuple(f"smollm:context{size}" for size in config.context_size_estimates)
    else:
        values = tuple(_bucket_id(model, bucket) for bucket in config.buckets)
    if len(set(values)) != len(values):
        raise BenchmarkRequestError("production configuration contains duplicate request selectors")
    return values


def _compact_number(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _identity(config: ModelConfig, request_bucket: str, bucket: CapacityBucket | None,
             witnesses: Mapping[str, Any]) -> dict[str, Any]:
    if any(not isinstance(k, str) or not k for k in witnesses):
        raise BenchmarkRequestError("identity witness names must be text")
    result: dict[str, Any] = {"model": ModelId(config.model_id).value, "request_bucket": request_bucket,
                              "witnesses": dict(witnesses)}
    if bucket is not None:
        result.update({"dtype": bucket.generation.dtype,
                       "generation_parameters": dict(bucket.generation.parameters),
                       "native_batch_shape": list(bucket.native_batch_shape),
                       "token_limits": [bucket.max_input_tokens, bucket.max_output_tokens],
                       "iterations": bucket.max_iterations})
    else:
        result.update({"dtype": witnesses.get("dtype"), "generation_parameters": witnesses.get("generation_parameters"),
                       "native_batch_shape": witnesses.get("native_batch_shape"),
                       "context_size": int(request_bucket.removeprefix("smollm:context"))})
        if result["dtype"] is None or result["generation_parameters"] is None or result["native_batch_shape"] is None:
            raise BenchmarkRequestError("SmolLM dtype, generation, and native batch identity are required")
    _canonical(result)
    return result


def _request_fingerprint(identity: Mapping[str, Any], payload: bytes) -> str:
    document = {"identity": identity, "request_sha256": hashlib.sha256(payload).hexdigest()}
    return hashlib.sha256(_canonical(document)).hexdigest()

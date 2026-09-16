"""Offline artifact and model-specific provisioning contracts."""

from dataclasses import dataclass
import math
from typing import Mapping

from services.llm.queue.contracts import ModelId, _text


RESOURCE_MANAGER_TYPES = {
    ModelId.SMOLLM: "ollama",
    ModelId.COEDIT: "transformers-coedit",
    ModelId.GECTOR: "gector",
}


@dataclass(frozen=True)
class GenerationConfig:
    parameters: Mapping[str, int | float | bool | str]
    dtype: str
    def __post_init__(self) -> None:
        _text(self.dtype, "dtype")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("unsupported dtype")
        if not self.parameters:
            raise ValueError("generation parameters are required")
        for key, value in self.parameters.items():
            if not isinstance(key, str) or not key:
                raise ValueError("generation parameter names must be text")
            if not isinstance(value, (int, float, bool, str)):
                raise ValueError("generation parameters must be scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("generation parameters must be finite")


@dataclass(frozen=True)
class CapacityBucket:
    max_input_tokens: int
    max_output_tokens: int
    generation: GenerationConfig
    native_batch_shape: tuple[int, ...]
    max_iterations: int | None = None
    thresholds: tuple[float, ...] = ()
    def __post_init__(self) -> None:
        if self.max_input_tokens < 1 or self.max_output_tokens < 1:
            raise ValueError("token limits must be positive")
        if not self.native_batch_shape or any(item < 1 for item in self.native_batch_shape):
            raise ValueError("native batch shape must be positive")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError("iteration limit must be positive")
        if any(item < 0 or item > 1 for item in self.thresholds):
            raise ValueError("thresholds must be between zero and one")


@dataclass(frozen=True)
class ModelConfig:
    model_id: ModelId
    parent_path: str
    model_file: str
    resource_manager_type: str
    context_size_estimates: tuple[int, ...] = ()
    benchmark_requests: tuple[str, ...] = ()
    buckets: tuple[CapacityBucket, ...] = ()

    def __post_init__(self) -> None:
        model = ModelId(self.model_id)
        _text(self.parent_path, "parent path")
        _text(self.model_file, "model file")
        if self.resource_manager_type != RESOURCE_MANAGER_TYPES[model]:
            raise ValueError("resource manager type does not match model")
        if model is ModelId.SMOLLM:
            if not self.context_size_estimates or any(item < 1 for item in self.context_size_estimates):
                raise ValueError("SmolLM needs positive context estimates")
            if self.buckets:
                raise ValueError("SmolLM uses context estimates, not buckets")
        else:
            if self.context_size_estimates:
                raise ValueError("non-Ollama models do not use context estimates")
            if not self.buckets:
                raise ValueError("non-Ollama models require capacity buckets")
            if model is ModelId.GECTOR:
                if any(item.max_iterations is None or not item.thresholds for item in self.buckets):
                    raise ValueError("GECToR buckets require iterations and thresholds")
            if model is ModelId.COEDIT:
                if any(item.max_iterations is not None or item.thresholds for item in self.buckets):
                    raise ValueError("CoEdIT buckets do not accept GECToR iteration or threshold fields")
        if any(not isinstance(item, str) or not item for item in self.benchmark_requests):
            raise ValueError("benchmark requests must be non-empty strings")
        if not isinstance(self.buckets, tuple) or any(not isinstance(item, CapacityBucket) for item in self.buckets):
            raise ValueError("buckets must be typed capacity buckets")


@dataclass(frozen=True)
class ArtifactIdentity:
    manifest_hash: str
    model_hash: str
    transitive_hashes: tuple[str, ...]
    def __post_init__(self) -> None:
        _text(self.manifest_hash, "manifest hash")
        _text(self.model_hash, "model hash")
        if not self.transitive_hashes or any(not item for item in self.transitive_hashes):
            raise ValueError("complete transitive artifact identity is required")

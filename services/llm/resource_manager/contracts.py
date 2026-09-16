"""Exact-identity capacity profile contracts."""

from dataclasses import dataclass
from typing import Mapping

from services.llm.queue.contracts import ModelId, _text


@dataclass(frozen=True)
class SampleMetadata:
    concurrency: int
    wave: int
    successful_requests: int
    wall_time_ms: int
    peak_vram_bytes: int
    latency_ms: tuple[int, ...]
    def __post_init__(self) -> None:
        if self.concurrency < 1 or self.wave < 0 or self.successful_requests < 0:
            raise ValueError("invalid sample counters")
        if self.wall_time_ms < 0 or self.peak_vram_bytes < 0 or any(item < 0 for item in self.latency_ms):
            raise ValueError("sample measurements must be non-negative")


@dataclass(frozen=True)
class CapacityProfile:
    model_id: ModelId
    gpu_uuid: str
    artifact_manifest_hash: str
    model_hash: str
    runtime_identity: str
    adapter_identity: str
    profile_identity: str
    optimal_parallelism: int
    memory_safe_n: int
    buffer_capacity: int
    safety_reserve_percent: int
    raw_samples: tuple[SampleMetadata, ...]
    context_size: int | None = None
    bucket_identity: str | None = None
    def __post_init__(self) -> None:
        ModelId(self.model_id)
        for value, name in (
            (self.gpu_uuid, "GPU UUID"), (self.artifact_manifest_hash, "manifest hash"),
            (self.model_hash, "model hash"), (self.runtime_identity, "runtime identity"),
            (self.adapter_identity, "adapter identity"), (self.profile_identity, "profile identity"),
        ):
            _text(value, name)
        integer_fields = (self.optimal_parallelism, self.memory_safe_n, self.buffer_capacity, self.safety_reserve_percent)
        if any(not isinstance(item, int) or isinstance(item, bool) for item in integer_fields):
            raise ValueError("profile numeric fields must be integers")
        if self.optimal_parallelism < 1 or self.memory_safe_n < self.optimal_parallelism:
            raise ValueError("invalid parallelism bounds")
        if self.buffer_capacity != self.optimal_parallelism:
            raise ValueError("buffer capacity must equal optimal parallelism")
        if not 0 <= self.safety_reserve_percent <= 100:
            raise ValueError("invalid safety reserve")
        if not self.raw_samples:
            raise ValueError("raw capacity samples are required")
        if self.context_size is not None and self.context_size < 1:
            raise ValueError("context size must be positive")
        if self.context_size is None and not self.bucket_identity:
            raise ValueError("a non-Ollama profile requires a bucket identity")

    @property
    def admission_limit(self) -> int:
        return self.optimal_parallelism + self.buffer_capacity

    def matches(self, model_id: ModelId, gpu_uuid: str, artifact_manifest_hash: str,
                model_hash: str, runtime_identity: str, adapter_identity: str) -> bool:
        return (ModelId(model_id), gpu_uuid, artifact_manifest_hash, model_hash,
                runtime_identity, adapter_identity) == (
                    self.model_id, self.gpu_uuid, self.artifact_manifest_hash,
                    self.model_hash, self.runtime_identity, self.adapter_identity)

    def accepts_request(self, context_size: int | None = None,
                        bucket_identity: str | None = None) -> bool:
        """Require the exact profiled context or bucket; never fall back."""
        if self.context_size is not None:
            return context_size is not None and context_size <= self.context_size and bucket_identity is None
        return bucket_identity is not None and bucket_identity == self.bucket_identity and context_size is None

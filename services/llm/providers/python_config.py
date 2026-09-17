"""Configuration for optional, isolated Python model providers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from .config import GPUProof
from .gpu import ProcessIdentity


@dataclass(frozen=True)
class PythonProviderConfig:
    NATIVE_BATCH_DELAY_SECONDS = 0.005
    MAX_NATIVE_BATCH_SIZE = 32
    artifact_root: Path
    manifest_sha256: str
    model_sha256: str
    gpu_uuid: str
    runtime_identity: str
    adapter_identity: str
    bucket_identity: str = "coedit:p1:input128:output64:float16:beams1:nosample"
    max_native_batch_size: int = 1
    native_batch_delay_seconds: float = NATIVE_BATCH_DELAY_SECONDS
    max_input_tokens: int = 128
    max_output_tokens: int = 64
    dtype: str = "float16"
    generation_parameters: tuple[tuple[str, object], ...] = (("num_beams", 1), ("do_sample", False))
    request_timeout_seconds: float = 120.0
    rpc_frame_limit: int = 256 * 1024
    gpu_proof: GPUProof | None = None

    def __post_init__(self) -> None:
        for value, name in ((self.manifest_sha256, "manifest digest"), (self.model_sha256, "model digest")):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be an exact SHA-256 digest")
        if not isinstance(self.artifact_root, Path) or not self.artifact_root.is_absolute():
            raise ValueError("artifact root must be an absolute Path")
        if any(not isinstance(x, str) or not x for x in (self.gpu_uuid, self.runtime_identity, self.adapter_identity, self.bucket_identity)):
            raise ValueError("runtime identities are required")
        if type(self.max_input_tokens) is not int or self.max_input_tokens < 1 or type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("token buckets must be positive integers")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("unsupported dtype")
        parameters = dict(self.generation_parameters)
        if (not isinstance(self.generation_parameters, tuple) or len(parameters) != len(self.generation_parameters)
                or set(parameters) != {"num_beams", "do_sample"}
                or type(parameters["num_beams"]) is not int or parameters["num_beams"] < 1
                or type(parameters["do_sample"]) is not bool or parameters["do_sample"] is not False):
            raise ValueError("CoEdIT generation must be deterministic")
        if type(self.max_native_batch_size) is not int or not 1 <= self.max_native_batch_size <= self.MAX_NATIVE_BATCH_SIZE:
            raise ValueError(f"native batch size must be between 1 and {self.MAX_NATIVE_BATCH_SIZE}")
        if (not isinstance(self.native_batch_delay_seconds, (int, float)) or isinstance(self.native_batch_delay_seconds, bool)
                or not math.isfinite(self.native_batch_delay_seconds) or not 0 <= self.native_batch_delay_seconds <= 1):
            raise ValueError("native batch delay must be between zero and one second")
        if self.native_batch_delay_seconds != self.NATIVE_BATCH_DELAY_SECONDS:
            raise ValueError("native batch delay is fixed by the CoEdIT batch identity")
        expected_bucket = (f"coedit:p{self.max_native_batch_size}:input{self.max_input_tokens}:output{self.max_output_tokens}:"
                           f"{self.dtype}:beams{parameters['num_beams']}:nosample")
        if self.bucket_identity != expected_bucket:
            raise ValueError("bucket identity must bind CoEdIT shape, dtype, generation, and native batch")
        if not isinstance(self.request_timeout_seconds, (int, float)) or isinstance(self.request_timeout_seconds, bool) or not math.isfinite(self.request_timeout_seconds) or self.request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if type(self.rpc_frame_limit) is not int or not 1024 <= self.rpc_frame_limit <= 256 * 1024:
            raise ValueError("invalid RPC frame limit")
        if type(self.gpu_proof) is not GPUProof or type(self.gpu_proof.expected_supervisor) is not ProcessIdentity:
            raise ValueError("a captured typed GPU proof and supervisor identity are required")
        object.__setattr__(self, "artifact_root", self.artifact_root.resolve())

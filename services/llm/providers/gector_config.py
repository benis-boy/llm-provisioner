"""Strict configuration for the isolated, offline GECToR adapter."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from .config import GPUProof
from .gpu import ProcessIdentity


@dataclass(frozen=True)
class GECToRProviderConfig:
    artifact_root: Path
    manifest_sha256: str
    model_sha256: str
    gpu_uuid: str
    runtime_identity: str
    adapter_identity: str
    bucket_identity: str = "gector:p1:tokens128:keep0:min0:iterations1:batch1:float32"
    max_subword_tokens: int = 128
    keep_confidence: float = 0.0
    min_error_prob: float = 0.0
    max_iterations: int = 1
    dtype: str = "float32"
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
        if self.max_subword_tokens != 128 or type(self.max_subword_tokens) is not int:
            raise ValueError("the initial GECToR bucket is fixed at 128 subword tokens")
        for value, name in ((self.keep_confidence, "keep_confidence"), (self.min_error_prob, "min_error_prob")):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value != 0):
                raise ValueError(f"{name} must be exactly numeric 0")
        if type(self.max_iterations) is not int or self.max_iterations != 1:
            raise ValueError("only the proved single GECToR iteration is supported")
        if self.dtype != "float32":
            raise ValueError("the GECToR bucket requires float32")
        expected = "gector:p1:tokens128:keep0:min0:iterations1:batch1:float32"
        if self.bucket_identity != expected:
            raise ValueError("bucket identity must bind every GECToR execution parameter")
        if not isinstance(self.request_timeout_seconds, (int, float)) or isinstance(self.request_timeout_seconds, bool) or not math.isfinite(self.request_timeout_seconds) or self.request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if type(self.rpc_frame_limit) is not int or not 1024 <= self.rpc_frame_limit <= 256 * 1024:
            raise ValueError("invalid RPC frame limit")
        if type(self.gpu_proof) is not GPUProof or type(self.gpu_proof.expected_supervisor) is not ProcessIdentity:
            raise ValueError("a captured typed GPU proof and supervisor identity are required")
        object.__setattr__(self, "artifact_root", self.artifact_root.resolve())

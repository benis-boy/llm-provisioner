"""Typed, server-owned configuration for the SmolLM provider."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from pathlib import Path
from typing import Awaitable, Callable

from .gpu import ResidencyEvidence, ProcessIdentity


GPUProbe = Callable[[], str | Awaitable[str]]
CleanupProbe = Callable[[], bool | Awaitable[bool]]
ResidencyProbe = Callable[[], ResidencyEvidence | Awaitable[ResidencyEvidence]]


@dataclass(frozen=True)
class GPUProof:
    identity: GPUProbe
    cleanup: CleanupProbe
    residency: ResidencyProbe | None = None
    expected_supervisor: ProcessIdentity | None = None

    def __post_init__(self) -> None:
        if self.expected_supervisor is not None:
            if (type(self.expected_supervisor) is not ProcessIdentity or
                    type(self.expected_supervisor.pid) is not int or self.expected_supervisor.pid <= 0 or
                    type(self.expected_supervisor.start_time) is not int or self.expected_supervisor.start_time < 0):
                raise ValueError("expected supervisor identity is invalid")


@dataclass(frozen=True)
class SmolLMProviderConfig:
    artifact_root: Path
    manifest_sha256: str
    model_sha256: str
    gpu_uuid: str
    runtime_identity: str
    adapter_identity: str
    allowed_context_sizes: tuple[int, ...] = (512,)
    parallelism: int = 1
    request_timeout_seconds: float = 60.0
    ollama_port: int = 11434
    gpu_proof: GPUProof | None = None
    ollama_binary: str = "ollama"

    def __post_init__(self) -> None:
        for value, name in ((self.manifest_sha256, "manifest digest"), (self.model_sha256, "model digest")):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} must be an exact SHA-256 digest")
        if not all(isinstance(value, str) and value for value in
                   (self.gpu_uuid, self.runtime_identity, self.adapter_identity)):
            raise ValueError("runtime identities are required")
        if not isinstance(self.artifact_root, Path):
            raise ValueError("artifact root must be a Path")
        object.__setattr__(self, "artifact_root", self.artifact_root.resolve())
        contexts = tuple(self.allowed_context_sizes)
        object.__setattr__(self, "allowed_context_sizes", contexts)
        if contexts != (512,):
            raise ValueError("only the proved 512-token SmolLM context is supported")
        if not contexts or any(type(x) is not int or x < 1 for x in contexts):
            raise ValueError("allowed context sizes must be positive integers")
        if type(self.parallelism) is not int or self.parallelism < 1:
            raise ValueError("parallelism must be positive")
        if type(self.ollama_port) is not int or self.ollama_port < 1 or self.ollama_port > 65535:
            raise ValueError("invalid loopback port")
        if (isinstance(self.request_timeout_seconds, bool) or
                not isinstance(self.request_timeout_seconds, (int, float)) or
                not math.isfinite(self.request_timeout_seconds) or self.request_timeout_seconds <= 0):
            raise ValueError("request timeout must be positive")
        if not isinstance(self.ollama_binary, str) or not self.ollama_binary:
            raise ValueError("Ollama binary is required")

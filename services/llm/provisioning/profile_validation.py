"""Strict, read-only validation of operator-supplied measured profiles."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import sqlite3

from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.profiles import CorruptProfileStore, ProfileStore

@dataclass(frozen=True)
class ProfileValidationBinding:
    registry_path: str | Path
    gpu_uuid: str
    artifact_manifest_hash: str
    model_hash: str
    runtime_identity: str
    adapter_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.registry_path, (str, Path)):
            raise ValueError("profile registry paths must be absolute")
        path = Path(self.registry_path)
        if not path.is_absolute():
            raise ValueError("profile registry paths must be absolute")
        for value in (self.gpu_uuid, self.artifact_manifest_hash, self.model_hash,
                      self.runtime_identity, self.adapter_identity):
            if not isinstance(value, str) or not value:
                raise ValueError("profile identities must be non-empty strings")

class ProfileSyntaxError(ValueError):
    pass

def parse_profile(body: Any) -> CapacityProfile:
    if not isinstance(body, dict): raise ProfileSyntaxError("invalid profile")
    common = {"modelId", "gpuUuid", "artifactManifestHash", "modelHash", "runtimeIdentity", "adapterIdentity", "profileIdentity", "optimalParallelism", "memorySafeN", "bufferCapacity", "safetyReservePercent", "rawSamples"}
    try: model = ModelId(body.get("modelId"))
    except (TypeError, ValueError) as exc: raise ProfileSyntaxError("invalid profile") from exc
    fields = common | ({"contextSize"} if model is ModelId.SMOLLM else {"bucketIdentity"})
    if set(body) != fields or not isinstance(body["rawSamples"], list): raise ProfileSyntaxError("invalid profile")
    samples = []
    keys = {"concurrency", "wave", "successfulRequests", "wallTimeMs", "peakVramBytes", "latencyMs"}
    for item in body["rawSamples"]:
        if not isinstance(item, dict) or set(item) != keys or not isinstance(item["latencyMs"], list): raise ProfileSyntaxError("invalid samples")
        nums = keys - {"latencyMs"}
        if not all(type(item[k]) is int for k in nums) or not all(type(x) is int for x in item["latencyMs"]): raise ProfileSyntaxError("invalid samples")
        try: samples.append(SampleMetadata(item["concurrency"], item["wave"], item["successfulRequests"], item["wallTimeMs"], item["peakVramBytes"], tuple(item["latencyMs"])))
        except ValueError as exc: raise ProfileSyntaxError("invalid samples") from exc
    try:
        return CapacityProfile(model, body["gpuUuid"], body["artifactManifestHash"], body["modelHash"], body["runtimeIdentity"], body["adapterIdentity"], body["profileIdentity"], body["optimalParallelism"], body["memorySafeN"], body["bufferCapacity"], body["safetyReservePercent"], tuple(samples), body.get("contextSize"), body.get("bucketIdentity"))
    except (KeyError, TypeError, ValueError) as exc: raise ProfileSyntaxError("invalid profile") from exc

def validate_profile(body: Any, bindings: Mapping[str | ModelId, ProfileValidationBinding]) -> str:
    profile = parse_profile(body)
    binding = bindings.get(profile.model_id) or bindings.get(profile.model_id.value)
    if binding is None or not profile.matches(profile.model_id, binding.gpu_uuid, binding.artifact_manifest_hash, binding.model_hash, binding.runtime_identity, binding.adapter_identity): raise ValueError("profile validation failed")
    try:
        with ProfileStore.open_readonly(binding.registry_path) as store:
            found = store.lookup(profile.model_id, binding.gpu_uuid, binding.artifact_manifest_hash, binding.model_hash, binding.runtime_identity, binding.adapter_identity, context_size=profile.context_size, bucket_identity=profile.bucket_identity)
    except (OSError, sqlite3.DatabaseError, CorruptProfileStore, ValueError, OverflowError) as exc: raise ValueError("profile validation failed") from exc
    if found is None or found != profile: raise ValueError("profile validation failed")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()

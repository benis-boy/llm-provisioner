"""Offline artifact/profile preflight and unloaded real-adapter assembly."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping

from services.llm.provisioning.artifacts import SPECS, manifest
from services.llm.provisioning.volume import verify_current
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.http import ModelBinding
from services.llm.resource_manager.profiles import ProfileStore
from services.llm.providers.config import GPUProof, SmolLMProviderConfig
from services.llm.providers.python_config import PythonProviderConfig
from services.llm.providers.gector_config import GECToRProviderConfig
from services.llm.providers.smollm import SmolLMProvider
from services.llm.providers.coedit import CoEdITProvider
from services.llm.providers.gector import GECToRProvider
from .config import BootstrapConfig, MODEL_IDS

_OLLAMA = "ollama"
# These are distribution names only.  In particular, observing GECToR must not
# import its ML package (or torch) during bootstrap.
_PYTHON_RUNTIME = ("torch", "transformers", "tokenizers", "safetensors", "gector")
_BUCKETS = {
    ModelId.COEDIT: "coedit:p1:input128:output64:float16:beams1:nosample",
    ModelId.GECTOR: "gector:p1:tokens128:keep0:min0:iterations1:batch1:float32",
}
_MODEL_FILES = {"SmolLM": "SmolLM2-1.7B-Instruct-Q8_0.gguf",
                "CoEdIT": "model.safetensors", "GECToR": "model.safetensors"}


def observe_runtime_identities(ollama_version: str) -> dict[str, str]:
    """Return identities from installed metadata without importing ML runtimes."""
    def version_text(value: str) -> str:
        if (not isinstance(value, str) or not value or len(value) > 64
                or not value.isascii() or any(ord(c) <= 0x20 or ord(c) == 0x7f or c in "|=" for c in value)):
            raise ValueError("runtime version observation is invalid")
        return value

    ollama_version = version_text(ollama_version)
    versions = {}
    try:
        for package in _PYTHON_RUNTIME:
            versions[package] = version_text(importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError(f"required installed runtime metadata is missing: {exc}") from exc
    python = f"python:{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    def runtime(packages):
        return python + "|" + "|".join(f"{name}={versions[name]}" for name in sorted(packages))

    return {"SmolLM": f"ollama:{ollama_version}",
            "CoEdIT": runtime(name for name in versions if name != "gector"),
            "GECToR": runtime(versions)}


def _read_manifest(root: Path, digest: str) -> dict[str, Any]:
    path = root / digest / "manifest.json"
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    try:
        fd = os.open(path, flags)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("selected manifest is not a regular file")
        chunks: list[bytes] = []
        remaining = 64 * 1024 + 1
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    except OSError as exc:
        raise ValueError("selected manifest is unavailable") from exc
    finally:
        if fd is not None:
            os.close(fd)
    if len(raw) > 64 * 1024:
        raise ValueError("selected manifest is too large")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate manifest field")
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode(), object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite manifest")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("invalid selected manifest") from exc
    if not isinstance(value, dict) or set(value) != {"schema", "models", "manifest_sha256"}:
        raise ValueError("selected manifest schema is not exact")
    if value["manifest_sha256"] != digest or manifest(value["models"]) != value:
        raise ValueError("selected manifest canonical digest mismatch")
    return value


def _verify_and_hashes(config: BootstrapConfig) -> dict[str, str]:
    verified = verify_current(config.artifact_root)
    if verified.get("manifestSha256") != config.manifest_sha256:
        raise ValueError("current artifact manifest mismatch")
    document = _read_manifest(config.artifact_root, config.manifest_sha256)
    models = document["models"]
    if set(models) != set(MODEL_IDS):
        raise ValueError("selected manifest does not contain exactly the supported models")
    result = {}
    for model in MODEL_IDS:
        entry = models[model]
        if (not isinstance(entry, Mapping) or set(entry) != {"model_id", "root", "required_count", "present_count", "missing", "files"}
                or entry["model_id"] != model or entry["root"] != f"models/{model}"
                or not isinstance(entry["files"], list)
                or entry["required_count"] != len(SPECS[model])
                or entry["present_count"] != len(SPECS[model])
                or entry["missing"] != []
                or any(not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}
                       or type(item["size"]) is not int or item["size"] < 0
                       or not isinstance(item["sha256"], str) or len(item["sha256"]) != 64
                       or any(char not in "0123456789abcdef" for char in item["sha256"])
                       for item in entry["files"])
                or [item["path"] for item in entry["files"]] != list(SPECS[model])):
            raise ValueError(f"selected manifest entry mismatch for {model}")
        selected = [item for item in entry["files"] if isinstance(item, Mapping) and item.get("path") == _MODEL_FILES[model]]
        if len(selected) != 1 or set(selected[0]) != {"path", "size", "sha256"}:
            raise ValueError(f"missing exact selected model hash for {model}")
        result[model] = selected[0]["sha256"]
    # ``verify_current`` is deliberately point-in-time.  Read the small,
    # canonical manifest only after it has checked every selected file, then
    # check it again: a changed current selection or changed selected contents
    # observed during this extraction is a bootstrap failure, not a new input.
    reverified = verify_current(config.artifact_root)
    if reverified.get("manifestSha256") != config.manifest_sha256:
        raise ValueError("current artifact manifest changed during extraction")
    return result


class PinnedModelBinding(ModelBinding):
    """Binding that cannot admit a larger context or a different bucket later."""
    def resolve(self, *, context_size: int | None, bucket_identity: str | None):
        expected = 512 if self.model_id is ModelId.SMOLLM else _BUCKETS[self.model_id]
        if self.model_id is ModelId.SMOLLM:
            if context_size != expected or bucket_identity is not None:
                raise ValueError("bootstrap binding only admits context 512")
        elif context_size is not None or bucket_identity != expected:
            raise ValueError("bootstrap binding only admits its measured bucket")
        profile, provider = super().resolve(context_size=context_size, bucket_identity=bucket_identity)
        # ProfileStore's general context lookup intentionally chooses the
        # smallest adequate context.  Bootstrap's p=1 admission is stricter:
        # the returned measured shape itself must be the proved shape.
        if self.model_id is ModelId.SMOLLM:
            valid = profile.context_size == expected and profile.bucket_identity is None
        else:
            valid = profile.context_size is None and profile.bucket_identity == expected
        if (not valid or profile.optimal_parallelism != 1 or profile.buffer_capacity != 1
                or profile.safety_reserve_percent != 20):
            raise ValueError("profile lookup did not return the pinned measured shape")
        return profile, provider


@dataclass(frozen=True)
class PreparedBindings:
    bindings: Mapping[ModelId, ModelBinding]
    profiles: Mapping[ModelId, Any]
    _store: ProfileStore

    def close(self) -> None:
        self._store.close()


async def prepare_bindings(config: BootstrapConfig, gpu_proof: GPUProof,
                           observed_runtime_identities: Mapping[str, str]) -> PreparedBindings:
    """Startup-only API: proof capture is async; blocking verification is off-loop."""
    if not isinstance(config, BootstrapConfig) or type(gpu_proof) is not GPUProof or gpu_proof.expected_supervisor is None:
        raise ValueError("typed captured GPU proof with expected supervisor is required")
    if set(observed_runtime_identities) != set(MODEL_IDS):
        raise ValueError("observed runtime identities must cover exactly the supported models")
    if any(not isinstance(value, str) or not value for value in observed_runtime_identities.values()):
        raise ValueError("observed runtime identities must be non-empty text")
    identity = gpu_proof.identity()
    if inspect.isawaitable(identity):
        identity = await identity
    if identity != config.gpu_uuid:
        raise ValueError("GPU identity proof mismatch")
    for model in MODEL_IDS:
        if observed_runtime_identities[model] != config.models[model].runtime_identity:
            raise ValueError(f"installed runtime identity mismatch for {model}")
    hashes = await asyncio.to_thread(_verify_and_hashes, config)
    store = ProfileStore.open_readonly(config.profile_db)
    try:
        profiles = {}
        for model in MODEL_IDS:
            mid = ModelId(model)
            kwargs = {"context_size": 512} if mid is ModelId.SMOLLM else {"bucket_identity": _BUCKETS[mid]}
            profile = store.lookup(mid, config.gpu_uuid, config.manifest_sha256, hashes[model],
                                   config.models[model].runtime_identity,
                                   config.models[model].adapter_identity, **kwargs)
            if (profile is None or profile.optimal_parallelism != 1
                    or profile.buffer_capacity != 1 or profile.safety_reserve_percent != 20):
                raise ValueError(f"no exact measured p=1 profile for {model}")
            if ((mid is ModelId.SMOLLM and (profile.context_size != 512 or profile.bucket_identity is not None)) or
                    (mid is not ModelId.SMOLLM and
                     (profile.context_size is not None or profile.bucket_identity != _BUCKETS[mid]))):
                raise ValueError(f"no exact measured shape profile for {model}")
            profiles[mid] = profile
        common = dict(artifact_root=config.artifact_root, manifest_sha256=config.manifest_sha256,
                      gpu_uuid=config.gpu_uuid, gpu_proof=gpu_proof)
        providers = {
            ModelId.SMOLLM: SmolLMProvider(SmolLMProviderConfig(**common, model_sha256=hashes["SmolLM"], runtime_identity=config.models["SmolLM"].runtime_identity, adapter_identity=config.models["SmolLM"].adapter_identity, ollama_binary=str(config.ollama_binary), ollama_home=config.ollama_home, ollama_port=config.ollama_port)),
            ModelId.COEDIT: CoEdITProvider(PythonProviderConfig(**common, model_sha256=hashes["CoEdIT"], runtime_identity=config.models["CoEdIT"].runtime_identity, adapter_identity=config.models["CoEdIT"].adapter_identity)),
            ModelId.GECTOR: GECToRProvider(GECToRProviderConfig(**common, model_sha256=hashes["GECToR"], runtime_identity=config.models["GECToR"].runtime_identity, adapter_identity=config.models["GECToR"].adapter_identity)),
        }
        bindings = {mid: PinnedModelBinding(mid, config.gpu_uuid, config.manifest_sha256,
                    hashes[mid.value], config.models[mid.value].runtime_identity,
                    config.models[mid.value].adapter_identity, store, providers[mid]) for mid in ModelId}
        return PreparedBindings(MappingProxyType(bindings), MappingProxyType(profiles), store)
    except BaseException:
        store.close()
        raise

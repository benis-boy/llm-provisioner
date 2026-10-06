"""Unloaded real-adapter bindings used only by the offline measurer.

This deliberately does not call ``prepare_bindings``: profiles do not exist
until this boundary has completed its measurements.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.http import ModelBinding
from services.llm.providers.config import GPUProof, SMOLLM_MAX_PARALLELISM, SmolLMProviderConfig
from services.llm.providers.python_config import PythonProviderConfig
from services.llm.providers.gector_config import GECToRProviderConfig
from services.llm.providers.smollm import SmolLMProvider
from services.llm.providers.coedit import CoEdITProvider
from services.llm.providers.gector import GECToRProvider
from .bindings import MODEL_IDS, _verify_and_hashes
from .measurement_matrix import measurement_matrix
from .config import BootstrapConfig


@dataclass(frozen=True)
class MeasurementBinding(ModelBinding):
    profile: CapacityProfile = None  # type: ignore[assignment]
    # Bound by the exact provider/configuration identity, independently of the
    # operator's benchmark ceiling.
    identity_derived_max_parallelism: int = 1
    identity_derived_capability_reason: str = "operator_ceiling"
    provider_max_parallelism: int = 1

    def resolve(self, *, context_size: int | None, bucket_identity: str | None):
        if ((self.model_id is ModelId.SMOLLM and context_size != self.profile.context_size)
                or (self.model_id is not ModelId.SMOLLM and bucket_identity != self.profile.bucket_identity)):
            raise ValueError("measurement selector is not exact")
        return self.profile, self.provider


@dataclass(frozen=True)
class MeasurementBindings:
    bindings: Mapping[tuple[ModelId, str], MeasurementBinding]
    hashes: Mapping[str, str]


async def prepare_measurement_bindings(config: BootstrapConfig, proof: Any,
                                       runtime_identities: Mapping[str, str], *,
                                       ceiling: int) -> MeasurementBindings:
    """Verify current inputs and construct normal, unloaded providers."""
    if type(ceiling) is not int or not 1 <= ceiling <= 32:
        raise ValueError("measurement ceiling must be 1..32")
    if set(runtime_identities) != set(MODEL_IDS):
        raise ValueError("runtime identities must cover the configured matrix")
    if await proof.identity() != config.gpu_uuid:
        raise ValueError("captured GPU UUID differs from configuration")
    if any(runtime_identities[name] != config.models[name].runtime_identity for name in MODEL_IDS):
        raise ValueError("installed runtime identity differs from configuration")
    hashes = await asyncio.to_thread(_verify_and_hashes, config)
    expected_supervisor = getattr(proof, "expected_supervisor", None)
    if expected_supervisor is None:
        expected_supervisor = proof.supervisor_identity
    gpu_proof = GPUProof(proof.identity, proof.cleanup, proof.residency,
                         expected_supervisor=expected_supervisor,
                         residency_for_runner=proof.residency_for_runner,
                         memory=proof.memory,
                         ollama_ownership=proof.ollama_ownership)
    common = dict(artifact_root=config.artifact_root, manifest_sha256=config.manifest_sha256,
                  gpu_uuid=config.gpu_uuid, gpu_proof=gpu_proof)
    providers = {
        ModelId.SMOLLM: SmolLMProvider(SmolLMProviderConfig(**common, model_sha256=hashes["SmolLM"],
            runtime_identity=config.models["SmolLM"].runtime_identity, adapter_identity=config.models["SmolLM"].adapter_identity,
            parallelism=ceiling, ollama_binary=str(config.ollama_binary), ollama_home=config.ollama_home,
            ollama_port=config.ollama_port)),
        # The configured CoEdIT selector is p=1.  Its adapter's normal batcher
        # has that fixed identity; a p>1 attempt is therefore evidence only if
        # the adapter itself emits one correlated native batch.  The discovery
        # ceiling remains the operator's requested search bound.
        ModelId.COEDIT: CoEdITProvider(PythonProviderConfig(**common, model_sha256=hashes["CoEdIT"],
            runtime_identity=config.models["CoEdIT"].runtime_identity, adapter_identity=config.models["CoEdIT"].adapter_identity,
            measurement_max_native_batch_size=ceiling)),
        ModelId.GECTOR: GECToRProvider(GECToRProviderConfig(**common, model_sha256=hashes["GECToR"],
            runtime_identity=config.models["GECToR"].runtime_identity, adapter_identity=config.models["GECToR"].adapter_identity)),
    }
    # Derive the provisioning bound from the provider/configuration identity
    # before making the ephemeral profile.  In particular, GECToR is a fenced
    # p1 worker even when the operator asks the other adapters to search to p32.
    capabilities = {
        ModelId.GECTOR: (1, 1, "identity_configured_provider_capability"),
        ModelId.COEDIT: (PythonProviderConfig.MAX_NATIVE_BATCH_SIZE,
                         PythonProviderConfig.MAX_NATIVE_BATCH_SIZE,
                         "identity_configured_provider_capability"),
        ModelId.SMOLLM: (SMOLLM_MAX_PARALLELISM, SMOLLM_MAX_PARALLELISM,
                         "identity_configured_provider_capability"),
    }
    bindings = {}
    for mid, request_selector in measurement_matrix():
        selector = ({"context_size": int(request_selector.removeprefix("smollm:context"))}
                    if mid is ModelId.SMOLLM else {"bucket_identity": request_selector})
        identity_max, provider_max, capability_reason = capabilities[mid]
        effective_ceiling = min(ceiling, provider_max, identity_max)
        profile = CapacityProfile(mid, config.gpu_uuid, config.manifest_sha256, hashes[mid.value],
            config.models[mid.value].runtime_identity, config.models[mid.value].adapter_identity,
            "measurement-only-" + mid.value, effective_ceiling, effective_ceiling,
            effective_ceiling, 20,
            (SampleMetadata(1, 0, 1, 1, 0, (0,)),), **selector)
        # profile_store is never read: this subtype resolves only its ephemeral profile.
        bindings[(mid, request_selector)] = MeasurementBinding(
            mid, config.gpu_uuid, config.manifest_sha256, hashes[mid.value],
            config.models[mid.value].runtime_identity, config.models[mid.value].adapter_identity,
            None, providers[mid], profile, identity_max, capability_reason, provider_max)
    if set(bindings) != set(measurement_matrix()):
        raise ValueError("measurement binding matrix differs from production matrix")
    return MeasurementBindings(bindings, hashes)

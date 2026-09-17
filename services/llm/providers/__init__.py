"""Offline, server-bound model providers.

Keep this package importable in the flat compatibility image: input-bound proof
does not require the optional HTTP client used by the runtime adapter.
"""

__all__ = ["SmolLMProvider", "SmolLMProviderConfig", "GPUProof", "ProcessIdentity", "ResidencyEvidence"]


def __getattr__(name: str):
    if name == "SmolLMProvider":
        from .smollm import SmolLMProvider
        return SmolLMProvider
    if name in {"SmolLMProviderConfig", "GPUProof"}:
        from .config import GPUProof, SmolLMProviderConfig
        return {"SmolLMProviderConfig": SmolLMProviderConfig, "GPUProof": GPUProof}[name]
    if name in {"ProcessIdentity", "ResidencyEvidence"}:
        from .gpu import ProcessIdentity, ResidencyEvidence
        return {"ProcessIdentity": ProcessIdentity, "ResidencyEvidence": ResidencyEvidence}[name]
    raise AttributeError(name)

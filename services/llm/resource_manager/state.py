"""Public, deliberately small ResourceManager lifecycle observation types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceManagerState:
    """An immutable point-in-time lifecycle observation.

    This contains no provider, model, request, or failure detail.  It is
    intended for adapters such as health, not for admission or control.
    """

    phase: str = "startup"
    available: bool = False
    session_present: bool = False
    revision: int = 0

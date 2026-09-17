"""Health adapter for the ResourceManager's authoritative lifecycle state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from services.llm.health import DEPENDENCIES, PHASES, DependencyState, HealthBoundary, HealthSnapshot
from services.llm.resource_manager.state import ResourceManagerState


class ResourceManagerHealthBoundary(HealthBoundary):
    """HealthBoundary with post-probe lifecycle and conjunctive proofs.

    The generic boundary intentionally permits probes to replace injected
    values.  That is useful for ordinary health, but unsafe here: an external
    false/missing proof is never upgraded by a probe.  Lifecycle is read only
    after all probes complete, which is the linearization point for each call.
    """

    def __init__(self, resource_manager: Any, external: Callable[[], Mapping[str, Any]], **kwargs: Any):
        self._resource_manager = resource_manager
        self._external = external
        super().__init__(self._external_snapshot, **kwargs)

    def _external_snapshot(self) -> HealthSnapshot:
        try:
            values = self._external()
            if hasattr(values, "__await__"):
                close = getattr(values, "close", None)
                if callable(close):
                    close()
                raise TypeError("external health state must be synchronous")
            if not isinstance(values, Mapping):
                raise TypeError("external health state must be a mapping")
        except Exception:
            values = None
        return HealthSnapshot("stable", {
            name: values.get(name) if values is not None else None
            for name in DEPENDENCIES
        })

    async def _probed_conjunctive(self) -> tuple[ResourceManagerState, dict[str, DependencyState]]:
        initial = self._lifecycle()
        if initial is None:
            raise ValueError("malformed lifecycle snapshot")
        _, original = self._snapshot()
        probed = await self._states(dict(original))
        _, final = self._snapshot()
        return initial, {name: (final[name] if not final[name].ok else
                               original[name] if not original[name].ok else probed[name])
                         for name in DEPENDENCIES}

    def _lifecycle(self) -> ResourceManagerState | None:
        state = self._resource_manager.snapshot()
        if not isinstance(state, ResourceManagerState):
            return None
        if (state.phase not in PHASES or type(state.available) is not bool
                or type(state.session_present) is not bool
                or type(state.revision) is not int or state.revision < 0):
            return None
        return state

    @staticmethod
    def _lifecycle_gates(state: ResourceManagerState, dependencies: dict[str, DependencyState],
                         *, changed: bool = False) -> None:
        if changed:
            for name in DEPENDENCIES:
                dependencies[name] = DependencyState(False, "unavailable")
        if state.phase != "stable" or state.available is not True or state.session_present is not True:
            dependencies["adapter"] = DependencyState(False, "unavailable")
        if state.phase != "stable":
            dependencies["cleanup"] = DependencyState(False, "unavailable")

    async def dependencies(self) -> dict[str, DependencyState]:
        try:
            initial, dependencies = await self._probed_conjunctive()
            lifecycle = self._lifecycle()  # final authoritative read
            if lifecycle is None:
                raise ValueError("malformed lifecycle snapshot")
            self._lifecycle_gates(lifecycle, dependencies, changed=lifecycle.revision != initial.revision)
            return dependencies
        except Exception:
            return {name: DependencyState(False, "state_malformed") for name in DEPENDENCIES}

    async def readiness(self) -> tuple[bool, list[dict[str, str]], dict[str, DependencyState]]:
        try:
            initial, dependencies = await self._probed_conjunctive()
            lifecycle = self._lifecycle()  # final authoritative read
            if lifecycle is None:
                raise ValueError("malformed lifecycle snapshot")
            self._lifecycle_gates(lifecycle, dependencies, changed=lifecycle.revision != initial.revision)
            reasons: list[dict[str, str]] = []
            if (lifecycle.phase != "stable" or lifecycle.available is not True
                    or lifecycle.session_present is not True):
                reasons.append({"phase": lifecycle.phase})
            reasons.extend({"dependency": name, "reason": state.reason}
                           for name, state in dependencies.items() if not state.ok)
            return not reasons, reasons[:8], dependencies
        except Exception:
            states = {name: DependencyState(False, "state_malformed") for name in DEPENDENCIES}
            return False, [{"phase": "state_malformed"}], states


def create_boundary(resource_manager: Any, external: Callable[[], Mapping[str, Any]], **kwargs: Any) -> HealthBoundary:
    """Build a fail-closed ResourceManager-specific health boundary."""
    return ResourceManagerHealthBoundary(resource_manager, external, **kwargs)


ResourceManagerHealth = create_boundary

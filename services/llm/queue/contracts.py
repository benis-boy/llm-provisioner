"""Phase 0 queue contracts.

These value objects validate boundaries only.  A store must still perform
transactional idempotency, leasing, fencing, and transition guards.
"""

from dataclasses import dataclass, field
from enum import Enum
import inspect
import json
import math
import numbers
from typing import Any, Callable, Mapping


class ModelId(str, Enum):
    SMOLLM = "SmolLM"
    COEDIT = "CoEdIT"
    GECTOR = "GECToR"


class RequestStatus(str, Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    ON_GPU = "on_gpu"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class InsertionMode(str, Enum):
    APPEND = "append"
    SKIP_LINE = "skip-line"


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _json(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _json(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json(item)
        return
    raise ValueError("value must be JSON-compatible")


def _descriptor_value(descriptor: "FunctionDescriptor") -> dict[str, Any]:
    """Validate and return the wire shape; serialization detaches nested values."""
    if not isinstance(descriptor, FunctionDescriptor):
        raise ValueError("function descriptor must be a FunctionDescriptor")
    _text(descriptor.name, "function name")
    if not isinstance(descriptor.args, Mapping):
        raise ValueError("function arguments must be a mapping")
    args = dict(descriptor.args)
    if any(not isinstance(key, str) for key in args):
        raise ValueError("function argument keys must be text")
    _json(args)
    ids = descriptor.dependency_result_ids
    if not isinstance(ids, (list, tuple)) or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("dependency result IDs must be a list or tuple of non-empty text")
    return {"name": descriptor.name, "args": args, "dependency_result_ids": list(ids)}


def serialize_function_descriptor(descriptor: "FunctionDescriptor") -> str:
    """Canonicalize and snapshot an optional function intent."""
    return json.dumps(_descriptor_value(descriptor), sort_keys=True, separators=(",", ":"), allow_nan=False)


def deserialize_function_descriptor(value: str) -> "FunctionDescriptor":
    """Decode only the exact durable descriptor shape."""
    try:
        raw = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed function descriptor") from exc
    if not isinstance(raw, dict) or set(raw) != {"name", "args", "dependency_result_ids"}:
        raise ValueError("malformed function descriptor")
    if not isinstance(raw["args"], dict) or not isinstance(raw["dependency_result_ids"], list):
        raise ValueError("malformed function descriptor")
    descriptor = FunctionDescriptor(raw["name"], raw["args"], tuple(raw["dependency_result_ids"]))
    # Re-serialize to reject any shape that the constructor did not fully cover.
    _descriptor_value(descriptor)
    return descriptor


@dataclass(frozen=True)
class FunctionDescriptor:
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)
    dependency_result_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.name, "function name")
        if not isinstance(self.args, Mapping):
            raise ValueError("function arguments must be a mapping")
        args = dict(self.args)
        if any(not isinstance(key, str) for key in args):
            raise ValueError("function argument keys must be text")
        _json(args)
        if not isinstance(self.dependency_result_ids, (list, tuple)) or any(
            not isinstance(item, str) or not item for item in self.dependency_result_ids
        ):
            raise ValueError("dependency result IDs must be a list or tuple of non-empty text")


class FunctionRegistry:
    """Application-startup registry for sync and async functions."""

    def __init__(self) -> None:
        self._functions: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, function: Callable[..., Any]) -> None:
        _text(name, "function name")
        if not callable(function):
            raise ValueError("function must be callable")
        if name in self._functions:
            raise ValueError(f"function already registered: {name}")
        self._functions[name] = function

    def resolve(self, name: str) -> Callable[..., Any]:
        try:
            return self._functions[name]
        except KeyError as exc:
            raise LookupError("function_unavailable") from exc

    @staticmethod
    def is_async(function: Callable[..., Any]) -> bool:
        return inspect.iscoroutinefunction(function)


@dataclass(frozen=True)
class RequestRecord:
    scheduler_id: str
    request_id: str
    idempotency_key: str
    model_id: ModelId
    payload_reference: str
    dependencies: tuple[str, ...] = ()
    insertion_mode: InsertionMode = InsertionMode.APPEND
    ready: FunctionDescriptor | None = None
    template: FunctionDescriptor | None = None
    result_target: str = "local"
    cancellation_requested: bool = False
    status: RequestStatus = RequestStatus.SCHEDULED
    running_at: float | None = None
    done_at: float | None = None
    next_attempt_at: float | None = None

    def __post_init__(self) -> None:
        _text(self.scheduler_id, "scheduler_id")
        _text(self.request_id, "request_id")
        _text(self.idempotency_key, "idempotency_key")
        _text(self.payload_reference, "payload_reference")
        _text(self.result_target, "result_target")
        ModelId(self.model_id)
        InsertionMode(self.insertion_mode)
        RequestStatus(self.status)
        if any(not isinstance(item, str) or not item for item in self.dependencies):
            raise ValueError("dependencies must be non-empty text")
        if self.request_id in self.dependencies:
            raise ValueError("request cannot depend on itself")
        for descriptor in (self.ready, self.template):
            if descriptor is not None:
                value = _descriptor_value(descriptor)
                if not set(value["dependency_result_ids"]).issubset(self.dependencies):
                    raise ValueError("function descriptor references undeclared dependency")
        for value, name in ((self.running_at, "running_at"), (self.done_at, "done_at"), (self.next_attempt_at, "next_attempt_at")):
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"{name} must be a finite number")
        if self.running_at is not None and self.done_at is not None and self.done_at < self.running_at:
            raise ValueError("done_at cannot precede running_at")


@dataclass(frozen=True)
class QueuePosition:
    request_id: str
    rank: int
    insertion_sequence: int
    insertion_mode: InsertionMode = InsertionMode.APPEND
    group_anchor: str | None = None
    group_sequence: int | None = None

    def __post_init__(self) -> None:
        _text(self.request_id, "request_id")
        InsertionMode(self.insertion_mode)
        if self.rank < 0 or self.insertion_sequence < 0:
            raise ValueError("rank and insertion sequence must be non-negative")
        if self.insertion_mode is InsertionMode.SKIP_LINE:
            if self.group_anchor is not None and not self.group_anchor:
                raise ValueError("group anchor must not be empty")
            if self.group_sequence is None or self.group_sequence < 0:
                raise ValueError("skip-line group sequence is required")
        elif self.group_anchor is not None or self.group_sequence is not None:
            raise ValueError("append positions cannot carry group metadata")


@dataclass(frozen=True)
class Attempt:
    request_id: str
    attempt_token: str
    session_token: str
    residency_generation: int
    provider_correlation_id: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    time_on_gpu_ms: int | None = None
    gpu_timing_complete: bool = False

    def __post_init__(self) -> None:
        _text(self.request_id, "request_id")
        _text(self.attempt_token, "attempt_token")
        _text(self.session_token, "session_token")
        if not isinstance(self.residency_generation, int) or isinstance(self.residency_generation, bool) or self.residency_generation < 0:
            raise ValueError("residency generation must be a non-negative integer")
        if self.time_on_gpu_ms is not None and (
            not isinstance(self.time_on_gpu_ms, int) or self.time_on_gpu_ms < 0
        ):
            raise ValueError("GPU time must be a non-negative integer or null")
        if self.gpu_timing_complete and self.time_on_gpu_ms is None:
            raise ValueError("complete GPU timing needs a value")
        for value, name in ((self.started_at, "started_at"), (self.finished_at, "finished_at")):
            if value is not None and (not isinstance(value, numbers.Real) or not math.isfinite(value)):
                raise ValueError(f"{name} must be a finite timestamp")
        if self.started_at is not None and self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")


@dataclass(frozen=True)
class ResultHandoff:
    request_id: str
    attempt_token: str
    result_reference: str
    idempotency_key: str
    acknowledged: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.request_id, "request_id"),
            (self.attempt_token, "attempt_token"),
            (self.result_reference, "result_reference"),
            (self.idempotency_key, "idempotency_key"),
        ):
            _text(value, name)


@dataclass(frozen=True)
class OutboxEntry:
    operation_id: str
    kind: str
    idempotency_key: str
    payload_reference: str
    acknowledged: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.operation_id, "operation_id"),
            (self.kind, "kind"),
            (self.idempotency_key, "idempotency_key"),
            (self.payload_reference, "payload_reference"),
        ):
            _text(value, name)


TERMINAL = frozenset(
    {RequestStatus.DONE, RequestStatus.ERROR, RequestStatus.CANCELLED}
)

# Adjacency only.  Stores must enforce publication, lease, retry, cancellation,
# session, attempt, and generation guards transactionally before using an edge.
ALLOWED_TRANSITIONS = {
    RequestStatus.SCHEDULED: frozenset(
        {RequestStatus.RUNNING, RequestStatus.ERROR, RequestStatus.CANCELLED}
    ),
    RequestStatus.RUNNING: frozenset(
        {
            RequestStatus.SCHEDULED,
            RequestStatus.ON_GPU,
            RequestStatus.DONE,
            RequestStatus.ERROR,
            RequestStatus.CANCELLED,
        }
    ),
    RequestStatus.ON_GPU: frozenset(
        {
            RequestStatus.RUNNING,
            RequestStatus.DONE,
            RequestStatus.ERROR,
            RequestStatus.CANCELLED,
        }
    ),
    RequestStatus.DONE: frozenset(),
    RequestStatus.ERROR: frozenset(),
    RequestStatus.CANCELLED: frozenset(),
}


def can_transition(old: RequestStatus, new: RequestStatus) -> bool:
    """Return status adjacency; duplicate events are replay no-ops."""
    old_status = RequestStatus(old)
    new_status = RequestStatus(new)
    return old_status == new_status or new_status in ALLOWED_TRANSITIONS[old_status]

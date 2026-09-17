"""Transport-neutral ResourceManager contracts.

The injected provider is a precursor-only seam.  A future HTTP binding will
resolve the server-configured provider and profile; clients will not supply
either object over the wire.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import AsyncIterator, Protocol

from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile


class EventKind(StrEnum):
    BUFFERED = "buffered"
    ADMISSION = "admission"  # execution-slot acquisition, not buffer acceptance
    RESPONSE_FINISHED = "response_finished"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    SESSION_INVALIDATED = "session-invalidated"


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True)
class SessionInfo:
    scheduler_id: str
    session_token: str
    model_id: ModelId
    generation: int


@dataclass(frozen=True)
class Capacity:
    profile: CapacityProfile
    execution_slots: int
    buffer_slots: int
    free_slots: int


@dataclass(frozen=True)
class Submission:
    accepted: bool
    request_id: str
    attempt: str
    session_token: str
    generation: int
    backpressure: bool = False
    failure: Failure | None = None


@dataclass(frozen=True)
class ProgressEvent:
    sequence: int
    completion_sequence: int
    kind: EventKind
    request_id: str | None
    attempt: str | None
    session_token: str
    generation: int
    result: bytes | None = None
    failure: Failure | None = None
    time_on_gpu_ms: int | None = None
    gpu_timing_complete: bool = False


@dataclass(frozen=True)
class ProviderResponse:
    result: bytes
    time_on_gpu_ms: int | None = None
    gpu_timing_complete: bool = False


class Provider(Protocol):
    async def validate(self, profile: CapacityProfile) -> None: ...
    async def load(self, profile: CapacityProfile) -> None: ...
    async def ready(self) -> None: ...
    async def execute(self, request_id: str, payload: bytes) -> ProviderResponse: ...
    async def cancel(self, request_id: str) -> None: ...
    async def unload(self) -> None: ...
    async def verify_cleanup(self) -> bool: ...
    async def validate_input(self, payload: bytes, *, context_size: int | None,
                             bucket_identity: str | None) -> None: ...


class ResourceManagerClient(Protocol):
    async def start_session(self, scheduler_id: str, model_id: ModelId,
                            profile: CapacityProfile, provider: Provider, *,
                            idempotency_key: str) -> SessionInfo: ...
    async def submit(self, session_token: str, request_id: str, attempt: str,
                     payload: bytes, *, idempotency_key: str,
                     context_size: int | None = None,
                     bucket_identity: str | None = None) -> Submission: ...
    async def cancel_request(self, session_token: str, request_id: str, *,
                             idempotency_key: str) -> bool: ...
    async def stop_session(self, session_token: str, *, reason: str = "stopped",
                           idempotency_key: str) -> None: ...
    async def watch_progress(self, session_token: str, after_sequence: int = 0) -> AsyncIterator[ProgressEvent]: ...
    async def get_capacity(self, session_token: str) -> Capacity: ...

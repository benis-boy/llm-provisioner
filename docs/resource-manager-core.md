# ResourceManager core precursor

`services.llm.resource_manager.core.ResourceManager` is the transport-neutral
single-GPU authority. The injected `Provider` and `CapacityProfile` are
**in-process precursor seams only**; the separate HTTP binding selects them
from server configuration and will not accept them from a scheduler client.

## Exact API for the next scheduler

`ResourceManagerClient` exposes:

* `start_session(scheduler_id, model_id, profile, provider, *,
  idempotency_key) -> SessionInfo` — validates the model/profile identity,
  synchronously fences the prior residency, asks the old provider to cancel
  and waits for tracked execution to finish before unload/verification, then
  validates and loads the new provider. A cleanup timeout/failure leaves the
  manager unavailable and no new load is attempted. A start replay returns
  the same active session only; replay after supersession is
  `scheduler_superseded`.
* `submit(session_token, request_id, attempt, payload, *, idempotency_key,
  context_size=None, bucket_identity=None) -> Submission` — validates exact
  profile limits and provider input, rechecks the fence after every await, and
  admits at most `2p`. Duplicate idempotency keys replay before capacity
  checks; changed payloads conflict. The same `(request_id, attempt)` with an
  equivalent identity is an exact no-op even under another key; changed
  identity conflicts. Full capacity returns `backpressure`, not a failure.
* `cancel_request(session_token, request_id, *, idempotency_key) -> bool` —
  removes buffered work or requests best-effort provider cancellation. Active
  work retains its slot until provider execution returns; cancellation is
  fenced and emits `cancelled` when the execution task exits.
* `stop_session(session_token, *, reason="stopped", idempotency_key) -> None`
  — idempotently invalidates the active session before cleanup, emits a
  terminal session invalidation event, and uses the same lifecycle lock as
  start. It never clears a newer session.
* `watch_progress(session_token, after_sequence=0) -> AsyncIterator[ProgressEvent]`
  — validates nonnegative, non-future cursors, replays immutable events in
  sequence order, reports `cursor_expired` instead of silently losing history,
  and terminates after a stopped session's retained terminal event.
* `get_capacity(session_token) -> Capacity` — reports execution slots, equal
  buffer slots, and free accepted slots. While unavailable, `free_slots` is
  zero even if stale tracked work has not exited.

## Provider and event semantics

The provider abstraction is asynchronous: `validate`, `load`, `ready`,
`execute`, `cancel`, `unload`, `verify_cleanup`, and `validate_input`.
Cleanup uses one absolute lifecycle deadline covering cancellation requests,
tracked execution, unload, and cleanup verification. It uses bounded
`asyncio.wait` and never treats `Task.cancel()` as proof of cleanup. Any
unfinished lifecycle task is retained in an abandoned-task registry with its
exception consumed; the manager remains permanently unavailable and no new
model load overlaps it. Provider `validate`, `load`, and `ready` share a
separate configurable `load_timeout` (default 60 seconds). A cancelled start
also leaves the manager unavailable.

Execution concurrency is exactly `p = optimal_parallelism`; the input buffer is
exactly `p`, so accepted work is bounded at `2p` and provider tasks are never
unbounded. `buffered` means accepted into the input buffer. `admission` means
an execution slot was acquired. `response_finished` is emitted when a
`ProviderResponse` returns, even when its result or timing metadata is
malformed; a following non-retryable `failure` records that contract error.
It advances `completion_sequence` exactly once, including for fenced work,
but fenced output is never exposed. A non-response exception or cancellation
does not advance completion.

Every callback removes only its exact `(session_token, request_id, attempt)`
work object, so a late old callback cannot release a newer attempt's slot.
Provider execution `CancelledError` is handled in `finally`; the exact slot is
released and a `cancelled` event is emitted. Typed provider `Failure` values
retain their retryable classification for the scheduler. Malformed/non-empty
result and complete timing contracts are non-retryable; incomplete timing is
explicitly `null` and `gpu_timing_complete=false`.

Event retention is bounded by `max_events` per session. At most
`max_sessions` histories are retained; a pruned session reports explicit
`cursor_expired`. Watcher waiters are removed when their consumer is
cancelled. Session idempotency state and event history remain associated with
the session token, while old callbacks are fenced. No real adapters, HTTP
transport, queue edits, profile measurement, or compatibility logic is
included in the core. The separate [HTTP adapter](resource-manager-http.md)
now supplies tested local transport and server-owned profile lookup.

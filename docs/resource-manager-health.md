# ResourceManager health wiring

The resource manager exposes `snapshot()` as its only lifecycle observation
seam.  It returns an immutable `ResourceManagerState` and never includes
provider, model, request, prompt, or exception text.

The lifecycle is `startup` until a session is ready, `loading` while provider
validation/loading/readiness is in progress, `unloading` while replacement or
stop cleanup is fenced, `stable` only with the currently owned available
session, and `cleanup_failed` when cleanup is not proven complete.  A timed-out
operation therefore cannot later make readiness stable: the caller's state
remains fenced until cleanup has completed, or remains failed.

The observation includes a monotonic stable-session revision, without exposing
provider or model identity. It identifies a replacement that completed while a
health check was waiting on probes.

`resource_manager.health.create_boundary` combines that observation with an
external mapping of the seven dependency proofs.  Probes are conjunctive:
successful probes cannot replace a missing, malformed, or false external
proof. External proofs are sampled before and after probes, so a proof that
becomes false while a probe waits is not admitted. Lifecycle is sampled before
probes and again afterwards; a changed revision rejects all earlier
measurements, including stable-to-stable replacement. Those final reads are the
readiness linearization point; no callback is awaited afterwards. Stable
readiness additionally requires
`available is True` and `session_present is True`.  Adapter and cleanup gates
are owned by the lifecycle and cannot be overridden by external `true` values.
Profile measurement is external; no profile name or fixture is interpreted as
proof.

Cancellation of cleanup is recorded as `cleanup_failed`, since cancellation
does not prove residency was released.  A late worker completion cannot change
that state or make a prior readiness result current; callers must invoke a new
health operation for a new linearization point.

Unknown lifecycle phases, non-boolean lifecycle fields, and invalid revisions
are sanitized to the fixed `state_malformed` response. An asynchronous external
callback is not run; its coroutine is closed when possible and its proof fails
closed, avoiding an unawaited-coroutine warning.

This is bootstrap precursor wiring only.  It does not load manifests/images,
start servers, or establish production dependency probes.

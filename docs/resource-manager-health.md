# ResourceManager health wiring

The resource manager exposes `snapshot()` as its immutable lifecycle observation
seam.  It returns an immutable `ResourceManagerState` and never includes
provider, model, request, prompt, or exception text.

Runtime shutdown publishes a permanent admission fence before waiting for an
in-progress lifecycle operation.  This prevents a blocked load from reopening
admission after shutdown; later starts remain rejected even if provider cleanup
eventually returns. The revision changes with that fence, so in-flight health
observations cannot be reused across it.

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

This remains bootstrap precursor wiring, not deployed readiness proof. The runtime
composition now registers this boundary on the same aiohttp application as the
ResourceManager routes.  Its external proofs remain conjunctive with the
ResourceManager snapshot: idle startup is intentionally unready, and health
does not create a session or load a model.  Daemon loss and runtime stop fence
admission before cleanup; uncertain cleanup remains unready and is not silently
upgraded by a later probe. Runtime external proof collection rechecks SQLite
and free space, GPU identity, owned daemon/version/listener health, selected
artifact manifest identity (full hashes are verified at startup),
the exact pinned read-only measured profile, and installed runtime identities.
SQLite/artifact work is bounded off the aiohttp loop; a health probe never loads
or makes a provider resident.

`probe_active_dependency()` calls the current provider's read-only readiness
operation and rejects its observation if the lifecycle, revision, provider, or
profile changes while it awaits. Initial idle readiness stays false; session
start admission instead requires the startup dependencies and an available RM.

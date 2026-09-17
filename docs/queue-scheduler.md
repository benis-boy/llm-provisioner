# Queue scheduler precursor

`QueueScheduler` is the bounded coordinator for one `QueueStore`, one
`ResourceManagerClient` session, and one fixed capacity profile.  It is an
in-process seam or the tested RM HTTP client. The HTTP server selects its own
provider and measured profile using client selectors, not client-supplied objects.
The scheduler's own public HTTP endpoints remain unimplemented.

## Fences and recovery

The durable store session (`token`, `generation`) is deliberately independent
from the RM session token and generation.  A scheduler start acquires a new
local store fence and starts a new RM session.  Recovery fences old provider
work; it never hands old work to the new RM session.  Scheduled work is
re-evaluated, while an unacknowledged handoff is published without rerunning
the provider.

The store already persists request, attempt, submit-outbox, handoff, result
reference, event cursor, local session, attempt token, and attempt payload
reference.  The coordinator always decodes the durable evaluated payload (or
the original reference) and never recomputes a template during submit replay.

## Dispatch and publication

Capacity is read from RM before a durable claim; RM's `free_slots` is the sole
occupancy authority, so the scheduler does not double-reserve it. Dispatch and
outbox scans are bounded. A submit first persists the exact decoded bytes
(content-addressed) and dispatch context. If its acknowledgement is lost, a
the live session replays the same attempt key, content, and context;
unstructured transport failure remains an outbox replay rather than consuming
provider retry budget. Durable cancellation intents are sent before later
submits.

Provider completion is persisted as result bytes, then staged as a handoff,
published one handoff at a time, and acknowledged atomically by the store. A
staged but unacknowledged handoff survives a crash/reopen and is
published without rerunning the provider. Explicit stop instead terminalizes
every unfinished request, including publishing work, and fences its handoff.
Only a new completion event resets
the completion-only 60-second watchdog; eligibility (including capacity-blocked
eligible work) can arm it, while polling, dispatch, leases, and publication do
not count as progress. Active leases are renewed by bounded maintenance scans
for every live local attempt, including claimed/decoding work and submissions
with an uncertain or lost acknowledgement. This prevents the 30-second lease
from expiring before the 60-second completion watchdog. The watchdog disarms
when no work is eligible or in flight.

Publication uses an isolated SQLite connection in a worker thread and the
immutable local session captured when that worker starts. A scheduler refuses
restart while an abandoned publication worker remains active; wait for it to
exit or use a fresh scheduler instance/process. This intentionally fail-closed
rule prevents an old worker from publishing under a new lifecycle.

The public methods are `start`, `enqueue`, `get`, `watch`, `cancel`, and
`stop`. `watch` yields durable store event rows with their cursor, so a caller
can reconnect from the last cursor.  Cancellation is durable first and RM
cancellation is best effort; late fenced results cannot publish.

`start`, `cancel`, and `stop` accept an optional keyword `idempotency_key`.
The store journals these operations transactionally, scoped to the scheduler
and with a fingerprint of the operation arguments. Reusing a key with changed
arguments raises `IdempotencyConflict`; a key cannot be reused across
operations. Cancel and stop retries are terminal no-ops and return the
durable/current outcome. `stop` also accepts `cancelled=True`, making its
terminal outcome `cancelled` rather than `error`.

Keyed start is stable only within the live local generation. After a process
restart, replaying an old start key raises the typed `OperationStale` error;
callers must choose a new key to recover a new RM session. A live start key
is passed unchanged to the RM, so a lost RM acknowledgement can be retried
without stopping the accepted queue or creating a second local session.
Unkeyed callers retain the legacy recovery behavior. This is a durable local
operation journal, not exactly-once behavior across an RM restart.

Stop fences the durable accepting state and lifecycle epoch before awaiting
any RM operation. A start that was already waiting on RM startup therefore
cannot revive the coordinator when its response arrives: it checks the exact
epoch, local session fence, and accepting state, then best-effort stops the
late RM session within the configured stop bound. Stop never waits
indefinitely for RM startup or cleanup.

## Store surface and remaining boundary

The store exposes bounded pending-outbox, replayable-submit, and active-lease
views for this in-process coordinator. For a production HTTP implementation,
the primary should consider these explicit methods (or equivalent rows in the
existing schema):

* a typed `pending_outbox(kind, limit)` API that returns immutable request,
  attempt, payload-reference, and idempotency fields;
* a typed event record/API instead of exposing SQLite rows from `events()`;
* a transactional `claim` operation accepting a scheduler-owned capacity
  reservation, if capacity must be coordinated across multiple scheduler
  processes;
* durable cross-process capacity reservation if multiple scheduler processes
  share one RM capacity authority.

One coordinator owns a store connection and `LocalPublisher` is local SQLite
publication, not an external publication protocol. These do not constitute a
multi-process capacity lock or production-boundary proof.

# Queue lifecycle contract

## Complete status adjacency matrix

Rows are current status and columns are target status. `Y` means the status
edge is structurally adjacent; `N` means rejected. Same-status entries are
replay no-ops, not transitions.

| from \\ to | scheduled | running | on_gpu | done | error | cancelled |
|---|---:|---:|---:|---:|---:|---:|
| scheduled | Y | Y | N | N | Y | Y |
| running | Y | Y | Y | Y | Y | Y |
| on_gpu | N | Y | Y | Y | Y | Y |
| done | N | N | N | Y | N | N |
| error | N | N | N | N | Y | N |
| cancelled | N | N | N | N | N | Y |

`can_transition` implements only this adjacency table. The generic store
`transition` is deliberately restricted: `done` is set only by
`acknowledge_handoff`, and error/cancelled terminalization always fences active
attempts and cancels pending submit/handoff deliveries. The store must enforce
the operation guards transactionally: `done` requires durable content-addressed
result availability and acknowledged idempotent publication; retries require a
retryable failure and persisted budget; `on_gpu` requires matching session,
attempt, residency generation, and admission; cancellation and stop fence late
tokens; terminal records cannot be mutated. Duplicate operations with the same
identity are successful no-ops. A conflicting idempotency key is
`idempotency_conflict`.

Buffer-full is backpressure, not an attempt, retry, error, or status transition.
Retryable failures use 5-second exponential backoff capped at 30 seconds and a
5-minute budget measured from the first retryable failure (`now - first_retry_at`),
not caller-provided elapsed time. A retry invalidates its old submit and emits a
best-effort cancel. A pending durable result handoff owns the request and cannot
be retried; recovery replays it without rerunning the provider. Non-retryable
failures and idle timeout stop the
whole scheduler. GPU timing may be incomplete and never blocks publication.

Dispatch is eligible FIFO. Dependency completion, registered function
availability, readiness, and template validity are gates. In the current
boundary, any request carrying a durable `ready` or `template` descriptor is
fail-closed at claim: it remains scheduled and emits no attempt, submit, or
claim event. A future scheduler-owned evaluator may open that gate; descriptor
persistence is not execution. Skip-line groups
extend at the group tail before their scheduled anchor; with no scheduled
anchor, insertion falls back to append. Rank, insertion sequence, anchor, and
group sequence are distinct durable fields. Optional functions poll every one
 second by default, configurable.

## Slice F evaluation boundary

`EligibilityEvaluator.next_eligible()` reads bounded candidate slices and skips
blocked nodes without fetching the full queue. Awaited callbacks execute outside
SQLite transactions. An opaque capability contains queue version, session,
generation, fingerprint, and evaluated payload reference; only
`QueueStore.claim_evaluated()` can consume it. Claim creates the attempt and
submit outbox atomically. Watchdog and ResourceManager admission remain future
scheduler work.

Evaluation does not make retry-delayed work eligible or arm scheduler watchdog
work. Both success and callback errors are discarded if the durable version,
session/generation, cancellation/stop state, or replacement session changed
since the evaluation snapshot. Evaluated capabilities are single-use internal
claims rather than a public direct-claim convenience; a durable version change
removes all unconsumed capabilities. Template payload references are persisted
with the created attempt and submit outbox record.

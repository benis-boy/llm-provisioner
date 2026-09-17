# Durable queue core: Phase 2 complete

This synchronous Python 3.12 storage foundation is **not yet an inference
service**. Run from the repository root using the project virtual environment;
the storage core is standard-library-only, while HTTP acceptance tests use the
project's installed dependencies.

## Boundaries

- `services/llm/queue/contracts.py`: status adjacency and typed boundary records.
  Strict canonical function-descriptor encoding/decoding validates durable intent.
- `services/llm/queue/store.py`: authoritative SQLite schema and transactional
  request, position, attempt, lease, outbox, event and local session primitives.
- `services/llm/queue/eligibility.py`: bounded async FIFO eligibility evaluation
  and guarded evaluated claims.
- `services/llm/queue/results.py`: content-addressed bytes and local publisher
  receipts. Files are fsynced before handoff state is committed.
- `services/llm/queue/transition_table.md`: adjacency versus operation guards.
- `docs/openapi.yaml`: validated external HTTP/SSE contract. RM and scheduler
  bindings have local loopback coverage; future operations remain target contracts.

The local store session is a persistence fence, **not** a ResourceManager
session or evidence of GPU ownership. `QueueScheduler` establishes the replacement
ResourceManager session before dispatching recovered work; a new local fence alone
does not prove that prior provider execution is fenced remotely.

## Publication contract

Payload references must already identify durable input bytes. The queue does
not decode or verify input content, load models, or execute provider requests.

`QueueStore.stage_result` writes result bytes through `ResultStore` before
transactionally recording the handoff. The local publisher implements
`publish(request_id, attempt_token, result_reference, idempotency_key)` and
verifies those bytes before recording its durable receipt. External sinks must
enforce the same idempotency contract.

`acknowledge_handoff` is a **trusted publisher acknowledgement boundary**, not a
verifier of arbitrary external side effects. Call it only after publisher
success, never merely because provider execution finished. Generic outbox
acknowledgement rejects handoffs so they cannot disappear before completion.
The result tests include an executable storage-to-publisher-to-acknowledgement
example and receipt replay across restart.

## Recovery, ordering and delivery

- Open the same database with the same scheduler/model identity and acquire a
  replacement session. Previous execution is fenced and rescheduled in place;
  pending durable publication is retained without rerunning the provider.
- Old submit records remain historical evidence, marked undeliverable. Cancel
  records request best-effort cleanup; the scheduler delivers them through its RM
  client, including the tested HTTP binding.
- Starting a different identity supersedes old nonterminal work in this database.
  Cross-database/process execution is fenced by ResourceManager session replacement.
- `claim` gates dependencies and retry eligibility but does not select the next
  request. Any non-null `ready` or `template` descriptor additionally blocks a
  claim without creating an attempt, submit record or claim event. The bounded
  scheduler chooses eligible FIFO and evaluates readiness/templates;
  arbitrary direct claims are not a FIFO scheduler. Other ungated requests can
  still be claimed; no optional functions execute inside SQLite transactions.
- Missing/self/cyclic dependencies are rejected before enqueue acknowledgement
  with `DependencyError`. Failed dependencies, including missing nodes in damaged
  stored graphs, become `dependency_failed` when evaluated or claimed. Recursive
  propagation requires scheduler scans.
- Grouped skip-line insertions are serialized by SQLite. Concurrent order follows
  durable insertion sequence, not thread start order.
- Lease expiration is reconciled explicitly; no background loop runs inside the
  store. Events have increasing cursors; the scheduler HTTP binding supplies
  resumable SSE replay.
- Stop/cancel fence pending deliveries transactionally. An external delivery
  loop must coordinate publication and cancellation; a caller must not publish
  a previously fetched outbox record without reconciling current ownership.

## Durable optional-function intent

`enqueue(..., ready=FunctionDescriptor(...), template=FunctionDescriptor(...))`
accepts keyword-only descriptors. Each contains a registered name, finite JSON
arguments and dependency result IDs. Those IDs must be among the request's
declared same-scheduler dependency request IDs; evaluation resolves them to
acknowledged durable result references. Function implementations are never stored.

Arguments are revalidated and serialized at enqueue, so later caller mutations
cannot change accepted intent. Equivalent object-key ordering replays as a no-op;
changed descriptor content conflicts with the accepted request's identity. Raw
request rows expose nullable canonical JSON `ready`/`template` columns;
`QueueStore.descriptor(row, "ready")` (or `"template"`) returns the typed value.

Opening a pre-Slice-E database transactionally adds nullable columns under a
SQLite write lock without recreating the database or rewriting existing records.
Absent descriptors retain the original fingerprint format and enqueue replay.
Recovery retains descriptor intent. Cancellation, stop and session fencing still
apply to descriptor-bearing work.

Slice F evaluates bounded FIFO slices outside transactions. Dependency IDs resolve
only to acknowledged durable handoff references. Sync functions run in
`asyncio.to_thread`; async results are awaited. Readiness requires strict `bool`,
and templates require a nonempty reference accepted by the injected validator.
Failures are request-local. Retry-delayed rows are not eligible and do not run
functions. An opaque, single-use version/session/fingerprint capability is
rechecked with ordinary retry, dependency and accepting fences before durable
claim; durable version changes delete unconsumed capabilities. Evaluated template
payload references are retained on the attempt and submit outbox record for
recovery/replay. Mutations and explicit/poll invalidation restart scanning at the
front using monotonic invalidation epochs. This is not a full scheduler or
ResourceManager.

## Verification and remaining work

```sh
.venv/bin/python -W error -m unittest -v tests.integration.test_phase2_acceptance tests.unit.test_contracts tests.unit.test_store tests.unit.test_queue_regressions tests.unit.test_results tests.unit.test_scheduler_operations tests.unit.test_scheduler tests.unit.test_eligibility tests.unit.test_function_intent tests.integration.test_queue_recovery tests.integration.test_resource_manager_http
```

These **130 task-related tests** pass, including actual RM HTTP retained across
scheduler process death and receipt-before-queue-ack replay. The eight-test Phase 2
acceptance module also passed three repeated runs. Fixtures own temporary state,
subprocesses and loopback services. The [completion record](queue-scheduler-resource-manager-plan-partial_completed.md#phase-2-complete--durable-queue-core)
maps the phase exit and limits. The store's acknowledgment method remains a trusted
publisher boundary; real publication verifies bytes before invoking it. Storage
guard tests may supply synthetic digests and are not proof of result availability.
Phase 3 scheduler interaction acceptance is now also complete; see its
[completion record](queue-scheduler-resource-manager-plan-partial_completed.md#phase-3-complete--queuescheduler-behavior).
Production provider/deployment proof remains open; no product goal is promoted
by these local phase exits.

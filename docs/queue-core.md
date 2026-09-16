# Durable queue core: implemented slice

This synchronous Python 3.12 storage foundation is **not yet an inference
service**. Run from the repository root; no third-party packages are needed for
the core or its current tests.

## Boundaries

- `services/llm/queue/contracts.py`: status adjacency and typed boundary records.
  Strict canonical function-descriptor encoding/decoding validates durable intent.
- `services/llm/queue/store.py`: authoritative SQLite schema and transactional
  request, position, attempt, lease, outbox, event and local session primitives.
- `services/llm/queue/results.py`: content-addressed bytes and local publisher
  receipts. Files are fsynced before handoff state is committed.
- `services/llm/queue/transition_table.md`: adjacency versus operation guards.
- `docs/openapi.yaml`: proposed external HTTP/SSE contract, not an implemented
  endpoint or generated binding. Parser/schema validation remains outstanding.

The local store session is a persistence fence, **not** a ResourceManager
session or evidence of GPU ownership. A future coordinator must establish the
replacement ResourceManager session before relying on recovery's assumption
that prior provider execution is fenced.

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
  records request best-effort cleanup; actual HTTP delivery is not implemented.
- Starting a different identity supersedes old nonterminal work in this database.
  Cross-database/process coordination needs the future ResourceManager.
- `claim` gates dependencies and retry eligibility but does not select the next
  request. Any non-null `ready` or `template` descriptor additionally blocks a
  claim without creating an attempt, submit record or claim event. The future
  bounded scheduler must choose eligible FIFO and evaluate readiness/templates;
  arbitrary direct claims are not a FIFO scheduler. Other ungated requests can
  still be claimed; no optional functions execute inside SQLite transactions.
- Missing/self/cyclic dependencies are rejected before enqueue acknowledgement
  with `DependencyError`. Failed dependencies become `dependency_failed` when
  evaluated. Recursive propagation requires scheduler scans.
- Grouped skip-line insertions are serialized by SQLite. Concurrent order follows
  durable insertion sequence, not thread start order.
- Lease expiration is reconciled explicitly; no background loop runs inside the
  store. Events have increasing cursors, but no resumable SSE transport exists.
- Stop/cancel fence pending deliveries transactionally. An external delivery
  loop must coordinate publication and cancellation; a caller must not publish
  a previously fetched outbox record without reconciling current ownership.

## Durable optional-function intent

`enqueue(..., ready=FunctionDescriptor(...), template=FunctionDescriptor(...))`
accepts keyword-only descriptors. Each contains a registered name, finite JSON
arguments and dependency result IDs. Those IDs must be among the request's
declared same-scheduler dependency request IDs; future execution resolves them to
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

This is persistence and safe blocking only: even a registered, always-true ready
function cannot currently enable a claim. Function execution, missing-name
`function_unavailable` classification, result resolution, template validation,
polling and eligibility invalidation belong to the future async scheduler.

## Verification and remaining work

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m compileall -q services tests
```

The 47 current tests use temporary SQLite databases, independent connections,
an abruptly exiting subprocess, and local result publication. Test-owned temporary
directories clean up prerequisites. They do not establish all operation replay
cases, every guarded lifecycle transition, or a production provider boundary.
Optional-function execution, async outbox delivery, scheduler
watchdog, ResourceManager admission/residency, adapters, profile persistence,
provisioning and deployment remain unfinished. All production proof remains open.

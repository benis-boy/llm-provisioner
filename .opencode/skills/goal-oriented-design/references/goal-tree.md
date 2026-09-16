# Product Goal Tree

Status meanings: **target** is intended but not implemented, **partial** has
only part of the outcome implemented, **untested** is implemented without the
required E2E proof, and **done** is implemented with the required E2E proof.
Parents with mixed or missing evidence are assessed conservatively.

## G0 — Reliable, capacity-aware queued LLM work

**partial —** Clients and operators can complete durable asynchronous LLM work
reliably and efficiently on fixed GPU resources without losing accepted work,
misrepresenting progress, or exceeding proved capacity.

### G1 — Accepted work remains durable and correct

**partial —** Accepted LLM work survives failures across scheduling, decoding,
execution, and result publication and reaches a truthful terminal outcome.

- **partial — G1.1 — Durable asynchronous work:** Once accepted, queued LLM work
  remains recoverable across scheduler, consumer, decoding, execution, and
  result-publication failures because durable intent is independent of transient
  attempts.
- **partial — G1.2 — Correct terminal outcomes:** Requests become done only after
  their required result is published; retries, dependency failures,
  cancellations, scheduler aborts, and stale executions cannot silently lose
  work or publish an invalid result. Duplicate operations are harmless no-ops.

### G2 — Progress and intervention are truthful

**partial —** Clients and operators can understand request progress and safely
intervene when individual work is unwanted or a scheduler stops progressing.

- **partial — G2.1 — Truthful request progress:** Clients and operators can
  observe scheduled, running, on-GPU, done, error, and cancelled progress and can
  distinguish total running-to-done time from ResourceManager-supplied time on
  the GPU, including when GPU timing is incomplete.
- **partial — G2.2 — Actionable cancellation and stalls:** Clients can cancel
  work in any status, stop a scheduler completely, and rely on idle or
  non-retryable failures to abort that scheduler without publishing late work.

### G3 — Eligible work is scheduled predictably

**partial —** Each model-specific queue dispatches eligible work predictably
while allowing dependent work to be submitted before it is ready.

- **partial — G3.1 — Predictable eligible ordering:** Each queue normally serves
  eligible work FIFO while durable dependency, readiness, and templating gates
  prevent premature execution without letting blocked work stop ready work.
- **partial — G3.2 — Deterministic priority insertion:** Clients can append work
  normally or insert it ahead of the first scheduled node with deterministic,
  idempotent ordering under serial and concurrent insertion.

### G4 — GPU model service is safe and capacity-aware

**target —** The available GPU remains productively supplied while model
residency, concurrency, and buffering stay within proved operational limits.

- **target — G4.1 — Safe exclusive model service:** The GPU serves SmolLM via
  Ollama, CoEdIT via Transformers/PyTorch, or GECToR via gector under one fenced
  residency authority, without stale execution crossing a model switch.
- **target — G4.2 — Bounded useful admission:** Each resident model accepts no
  more than its measured throughput-optimal parallelism plus a bounded input
  buffer; competing schedulers are rejected so the one GPU maximizes throughput
  for its active model.
- **target — G4.3 — Offline reproducible readiness:** Operators can provision
  useful parent-folder artifacts into the runtime, bootstrap ResourceManager,
  and reproduce a model/GPU/context throughput profile without runtime internet
  access; incompatible or unproved profiles fail closed.

Detailed target contracts and sequencing:
[`docs/queue-scheduler-resource-manager-plan.md`](../../../../docs/queue-scheduler-resource-manager-plan.md).

Implementation boundary: G1–G3 currently have synchronous SQLite/result primitives
and unit/local integration evidence only. Async scheduling, provider/server
boundaries and qualifying E2E remain absent. G4 has contract definitions, not
implemented GPU service, and remains target.

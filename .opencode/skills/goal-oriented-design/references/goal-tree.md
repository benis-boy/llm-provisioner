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

**partial —** The available GPU remains productively supplied while model
residency, concurrency, and buffering stay within proved operational limits.

- **partial — G4.1 — Safe exclusive model service:** The GPU serves SmolLM via
  Ollama, CoEdIT via Transformers/PyTorch, or GECToR via gector under one fenced
  residency authority, without stale execution crossing a model switch.
- **partial — G4.2 — Bounded useful admission:** Each resident model accepts no
  more than its measured throughput-optimal parallelism plus a bounded input
  buffer; competing schedulers are rejected so the one GPU maximizes throughput
  for its active model.
- **partial — G4.3 — Offline reproducible readiness:** Operators can provision
  useful parent-folder artifacts into the runtime, bootstrap ResourceManager,
  and reproduce a model/GPU/context throughput profile without runtime internet
  access; incompatible or unproved profiles fail closed.

Detailed target contracts and sequencing:
[`docs/queue-scheduler-resource-manager-plan.md`](../../../../docs/queue-scheduler-resource-manager-plan.md).

Implementation boundary: G1–G3 have SQLite/result primitives, durable optional
functions, bounded eligibility evaluation, and an async scheduler with local
integration evidence. G4 has a transport-neutral ResourceManager, artifact
verification, atomic content-addressed artifact-volume provisioning and candidate
offline tooling. Three-model RM-mediated fixture/cancellation/cleanup and injected
child-process-loss experiments passed, including owned-group disappearance and
NVML baseline restoration. These are candidate experiments, not production proof.
Durable profile storage validates supplied measurement evidence and exact identity;
it does not perform benchmarking. The RM HTTP/JSON/SSE server/client now has
loopback evidence, including scheduler publication, server-owned measured-profile
lookup, bounded admission and resumable progress. Scheduler HTTP now has local
loopback evidence for durable keyed lifecycle operations, strict JSON, immutable
bounded request replay and fenced publication with shared or separate receipt
databases. Read-only artifact verification HTTP has local tests for selected-file
identity, strict bounded requests and hash-worker lifetime across timeout/disconnect.
It does not assert runtime readiness. Read-only profile-validation HTTP now checks
full submitted profiles against server-owned current identities and exact measured
registry records with bounded worker lifetimes. Candidate GPU scenarios passed
again with independent bounded ASCII input evidence and exact raw SmolLM framing.
An installable SmolLM adapter now uses bounded local Ollama import, exact raw
framing, typed supervisor/runner residency evidence and cleanup checks. Linux
NVML/procfs proof and initial RM failure cleanup have local regression coverage;
real host-PID namespace wiring and real-adapter GPU execution remain unproved.
A network-disabled actual-SmolLM-adapter candidate check now exists, but its
real run failed closed with `procfs_missing` at `/host/proc/self/stat`; operator
repair of authoritative host procfs availability is required before further GPU
verification. Local harness tests do not establish that namespace boundary.
Capacity-measurement and health HTTP bindings, complete production adapters/image, actual measured capacity profiles and
qualifying production-boundary E2E remain absent. All leaves remain partial.

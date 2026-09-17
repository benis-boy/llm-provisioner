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

- **partial — G4.1 — Safe owned model service:** On a shared physical GPU,
  SmolLM via Ollama, CoEdIT via Transformers/PyTorch, or GECToR via gector runs
  under one fenced service-residency authority without claiming control of
  foreign workloads or allowing stale execution to cross a model switch.
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
real-adapter GPU execution remains unproved.
A network-disabled actual-SmolLM-adapter candidate check now exists, but its
real run failed closed with `procfs_missing` at `/host/proc/self/stat`; operator
repair of authoritative host procfs availability was required. A retry after adding the devcontainer procfs mount failed with the
same `procfs_missing` before readiness; its separate adapter container was removed
and verified absent. The devcontainer mount alone does not establish the required
daemon-host bind or NVML alignment. The harness now uses authoritative `/proc`
with `--pid=host`, and the ownership proof accepts changing provably foreign
workloads while returning only strict supervisor descendants. Seventy-four
focused tests pass. The latest actual run advanced through the old procfs mount
failure but could not classify at least one current NVML PID, so it failed closed
before readiness. Bounded lag retries, complete confirmation classification and
stable foreign-ancestry fencing now pass 83 focused tests. A rebuilt actual run
still failed closed; a minimal diagnostic isolated one persistent graphics NVML
PID absent from Linux procfs while Ollama ran. Exhaustive foreign correlation was
later rejected as beyond G4.1: unconnected non-supervisor NVML PIDs are now
ignored and never controlled, while residency still requires a stable nonempty
set of positively proved supervisor descendants. Eighty-eight focused tests pass.
Real inference, residency and cleanup remain unproved pending a new actual run.
The new network-disabled actual run passed with real SmolLM/Ollama inference,
one positively proved supervisor-descendant GPU runner, old-model absence after
unload, stale-session rejection, and identity-fenced daemon-group/container
cleanup. Two baseline NVML PIDs remained unrelated/unknown and untouched.
Focused verification passes 94 tests. G4.1 now has candidate production-boundary
evidence for this SmolLM path; CoEdIT, GECToR, full switching/cancellation, and
production deployment coverage remain incomplete.
An unwired read-only health HTTP precursor now has local contract evidence for
bounded probes, safe diagnostics and fail-closed injected readiness snapshots;
it does not establish production bootstrap or GPU readiness. Capacity-measurement
HTTP bindings, complete production adapters/image, actual measured capacity profiles and
qualifying production-boundary E2E remain absent. All leaves remain partial.

The installable CoEdIT isolated adapter now also has candidate real-boundary
evidence: offline small/near-bucket inference, exact worker residency, stale
session rejection and injected process-loss cleanup passed through ResourceManager.
Its strict batch-one p=1 profile is unmeasured. Full discovery passes 369 tests.
GECToR's installable isolated adapter now also passes offline real-boundary
normal and injected-process-loss candidate checks: exact package preprocessing,
overlong rejection without poisoning the worker, aligned responses, positive
residency, stale-token rejection and owned cleanup. Full discovery passes 391
tests. Its fixed float32/128-subword/one-iteration batch-one profile is unmeasured.
A single parent-rooted GPU proof and ResourceManager now also pass actual
installed-adapter SmolLM → CoEdIT → GECToR → SmolLM replacement: three cleanup-gated
switches, four successful responses, stale submit/cancel rejection, CoEdIT
execution-entry cancellation result fencing and final provider/daemon/GPU cleanup.
The offline candidate container was verified absent; full discovery passes 396
tests. G4.1/G4.2/G4.3 remain partial: this does not prove active kernel interruption,
measured capacity, production bootstrap/deployment or scheduler-to-GPU E2E.

ResourceManager now exposes immutable lifecycle/revision observations used by
health to fence loading, replacement, cleanup failure and late completion.
Readiness rechecks external dependency proofs and lifecycle after asynchronous
probes; probes cannot upgrade missing/false proofs or reuse a replaced session's
evidence. Offline bootstrap preflight now validates bounded operator configuration,
selected artifacts, metadata-only runtime identities and exact measured p=1/m=1
profiles before constructing all three unloaded real providers. Later resolution
cannot expand the supported selector or admission capacity. Full discovery passes
427 tests. These are unit/local-loopback proofs, not a supervised production
bootstrap, actual measured capacity or deployment readiness; G4 remains partial.

Private Ollama supervision now has focused local regression evidence for a
pre-exec identity gate, pidfd-targeted process control, listener ownership,
bounded health, cancellation and observed descendant cleanup. The focused
supervisor/bindings/process/GPU-proof/SmolLM suites pass 98 tests. This does not
establish arbitrary daemon-descendant containment, runtime composition, a
production image, or scheduler-to-GPU E2E; G4.1/G4.3 remain partial.

Supervised runtime composition now has local evidence for pre-child parent GPU
capture, explicit namespace attestation, exact binding preflight, combined RM/
health HTTP, live dependency admission gates and permanent shutdown with bounded
retained cleanup. Focused verification passes 141 tests. The installed three-model
offline candidate also passed with the production `OwnedOllama` supervisor and
shared parent proof, including final owned cleanup (29 focused harness tests).
Its profiles remain unmeasured. Actual composed-server GPU readiness, capacity
measurement and production packaging remain gaps; G4.1/G4.2/G4.3 stay partial.

Identity-fenced whole-device NVML memory observations now have 70 focused tests
and an offline installed-adapter candidate with six ordered points and verified
owned cleanup. They include foreign allocations and are not execution peaks,
incremental per-request VRAM, or measured capacity. Concurrent provider execution,
memory-safe bounds and throughput-optimal profiles remain unproved; G4.1/G4.2/
G4.3 remain **partial**.

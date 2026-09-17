# QueueScheduler and ResourceManager implementation plan

## 1. Purpose and current status

Build a durable asynchronous LLM-work platform in which model-specific client
schedulers can keep useful work flowing through one GPU-owning resource manager
without losing accepted requests or reporting misleading progress.

This checkout now contains a Python contract foundation, synchronous SQLite WAL
queue primitives, content-addressed result storage, a durable local publisher,
durable optional-function evaluation, an async QueueScheduler, a transport-neutral
ResourceManager core, RM and scheduler HTTP/JSON/SSE bindings, artifact verification
and experimental compatibility tools. A read-only selected-artifact verification
HTTP endpoint and read-only exact measured-profile validation also exist. It does
**not** yet contain capacity-measurement HTTP bindings, production health-state
wiring, a production image,
complete three-model production adapters, or measured provisioning runtime. The
installable SmolLM adapter and Linux GPU-ownership proof now have local regression
coverage, but no real-adapter GPU/deployment verification. Unchecked work remains target-state design. Historical
capacity notes are not implementation evidence.

### Implementation progress (2026-09-17)

Work is tracked in logical slices, not whole-phase completion claims. See
[implementation decisions](implementation-decisions.md) and
[queue-core boundaries](queue-core.md).

- [x] **Slice A — Contract foundation:** OpenAPI draft, typed request/attempt,
  function, provisioning-bucket and profile contracts; status adjacency table;
  decision owners and operational blockers recorded.
- [ ] **Phase 0 exit:** fully validate OpenAPI and client/server bindings; finish
  profile runtime integration after compatibility review. Durable profile SQLite
  storage and strict selected-artifact manifests now exist as precursor slices.
  Parsed OpenAPI validation and RM/scheduler loopback binding tests now pass;
   future capacity-measurement and health operations retain
  their full target contracts.
- [ ] **Phase 1 — Compatibility spike and production image:** Docker/NVIDIA and
  candidate three-model offline small and configured upper-fixture inference,
  RM cancellation fencing and process cleanup passed with verified selected
  artifacts. Full compatibility exit and production runtime
  remain unproved; candidate pins are not approved production pins. See
  [environment evidence](implementation-decisions.md#environment-reassessment-2026-09-16).
- [x] **Slice B — Durable queue primitives:** SQLite WAL transactions, exact
  enqueue idempotency, immutable intent references, separate ordered positions,
  attempts, leases, handoffs, cancellation/submission outbox and event cursors.
- [x] **Slice C — Ordering and local recovery:** dependency-gated claims,
  iterative cycle validation, transactional append/grouped skip-line insertion,
  lease/session recovery, stale-owner fencing, retry persistence and abort-all
  primitives; subprocess abrupt-exit and independent-connection tests.
- [x] **Slice D — Durable local results:** fsynced content-addressed files,
  verified idempotent local publication receipts, pending handoff replay, and
  acknowledged-only `done` transitions.
- [x] **Slice E — Durable optional-function intent:** canonical ready/template
  descriptors, dependency-reference validation, mutation-safe enqueue snapshots,
  descriptor-sensitive idempotency and additive legacy database upgrades.
  Direct claims of descriptor-bearing work fail closed without dispatch side
  effects. Function execution and polling are not implemented by this slice.
- [ ] **Phase 2 exit:** integrate real ResourceManager session reconciliation
  and outbox delivery; finish operation-replay and guarded-transition behavioral
  coverage.
- [x] **Slice F — Evaluated eligibility:** bounded FIFO scans, sync/async
  ready/template execution, dependency result resolution and single-use durable
  version/session/intent-fenced claim capabilities.
- [ ] **Phase 3 — Async QueueScheduler:** coordinator, dispatch replay, watch,
  completion watchdog, cancellation and crash publication recovery implemented
  locally. Review-driven race fixes passed local verification; production-boundary
  integration and complete transition/insertion coverage remain open.
- [ ] **ResourceManager precursor:** typed in-process protocol and fake-provider
  lifecycle, exclusive residency fences, p+p admission, bounded cleanup and event
  replay implemented. Real experimental RM-driven lifecycle harness passed;
   RM HTTP binding has local loopback proof; full production adapter integration remains open.
- [x] **RM transport slice:** typed HTTP client, server-owned profile/provider
  lookup, trusted shared content references, bounded JSON/SSE, request-level
  cancellation, typed non-failure backpressure and cursor replay/expiry.
  Scheduler-over-HTTP exact publication and all three model selector shapes
  pass with fake providers. See [HTTP boundary](resource-manager-http.md).
- [ ] **Phase 4 — ResourceManager and real adapters.**
- [x] **SmolLM adapter slice:** existing supervisor-owned loopback Ollama client,
   selected manifest/GGUF identity, bounded local import and raw ASCII framing,
   configured parallelism, typed supervisor/runner readiness, and verified cleanup.
   Failed initial RM lifecycle now cleans up before permitting another start;
   unproved cleanup remains fenced. This is not production packaging or measured capacity.
- [x] **Linux GPU ownership proof slice:** lazy NVML access, empty compute/graphics
   baseline, exact physical GPU, positive bounded procfs ancestry and PID-reuse checks,
   and cleanup absence. Real capture requires explicit host-PID-namespace
   attestation; deployment must supply authoritative procfs and supervisor identity.
   Current evidence uses synthetic NVML/procfs, not real GPU execution.
- [x] **Scheduler transport and control replay slice:** application-configured
  scheduler/function registries, six HTTP operations, durable keyed start/cancel/stop,
  strict bounded JSON and immutable request-specific SSE history. Live replay uses
  bounded batches without starvation from unrelated events; terminal reconnects
  close and unavailable legacy history fails before headers. Publication retains
  its cancellation/session fence for both shared and separate receipt databases.
  These are local fake-provider tests, not production restart or GPU proof.
  See [scheduler HTTP boundary](scheduler-http.md).
- [ ] **Phase 5 — Offline provisioner and measured capacity profiles.**
- [x] **Artifact-volume slice:** serialized, selected-file-only offline copying,
  source-independent content identities, verified digest reuse and atomic `current`
  selection. Corruption, unsafe paths and interrupted copying fail closed; prior
  complete volumes are retained. See [artifact-volume boundary](artifact-volume.md).
- [x] **Read-only artifact verification HTTP slice:** configured volume IDs,
  strict bounded JSON, exact selected-file/hash verification and minimal identity
  summaries. Hashing runs off the event loop with bounded concurrency; timeout
  and disconnect retain worker slots until hashing exits. No copying, download,
  model loading, readiness or measured-capacity claim is made by this endpoint.
- [x] **Profile registry precursor:** SQLite WAL/FULL immutable records, draft
  exclusion, exact identity/closest context lookup, evidence hashes and strict
  schema checks. Supplied successful waves must support the selected throughput
  concurrency and 2% tie rule. No actual capacity measurements or automatic
  production runtime profile integration are claimed. The HTTP binding resolves
   supplied measured registry records; see [profile boundary](capacity-profiles.md).
- [x] **Read-only profile-validation HTTP slice:** strict full-profile input,
   server-owned current identities and registry paths, exact measured-record and
   sample comparison, read-only SQLite access, bounded body/worker admission and
   retained worker slots across timeout/disconnect. This validates supplied
   evidence; it does not measure capacity or assert runtime readiness.
- [x] **Read-only health HTTP precursor:** injected atomic snapshots, bounded
  dependency probes, strict safe diagnostics and fail-closed liveness/readiness
  routes. Production bootstrap wiring remains absent, so this is not deployed
  readiness or GPU proof. See [health HTTP boundary](health-http.md).
- [ ] **Phase 6 — Production E2E, observability, deployment and runbooks.**

Verification: **294 tests passed** in 31.593 seconds using
`.venv/bin/python -W error -m unittest discover -s tests -p 'test_*.py'`; compilation of services,
tests and tools passed. The focused scheduler/HTTP/publication suites passed
**86 tests**; the stop-before-publication regression passed five repeated runs.
The profile/artifact HTTP/profile-store/OpenAPI focused suites passed **50 tests**.
The compatibility input/runtime/packaging suites passed **36 tests**.
The latest GPU-proof/provider/RM/packaging/input-bound focused suites passed
**60 tests**. CLI timeout/overflow cleanup passed three repeated runs with forced
garbage collection and an unraisable-exception hook, without pipe/transport warnings.
The separate actual-adapter harness passed **9 local tests**, including owned
orphan-group and cancelled-launch cleanup. Its hash-locked dependency layer built
with networking disabled. Real GPU execution was initially **environment-blocked**:
the read-only Docker-host procfs binding did not provide `/host/proc/self/stat`
(`procfs_missing`). The owned container was removed and verified absent. No
real-adapter inference/residency/baseline success is claimed; further GPU testing
requires operator repair of host procfs availability and NVML namespace alignment.
A bounded retry after adding the devcontainer's read-only `/proc` -> `/host/proc`
mount failed with the same `procfs_missing` in **5.685 seconds**, before model
readiness. The separate adapter container retained its own explicit read-only
bind, exact GPU UUID and disabled networking; it was removed and verified absent.
The devcontainer configuration alone did not establish procfs availability in
that separately launched Docker runtime. The adapter now uses authoritative
container `/proc` with required `--pid=host`; focused adapter/GPU-proof verification
passed **21 tests in 0.148 seconds** and compilation passed. A rebuilt offline
image (`sha256:b9e6ae2aa0440e3f4d4dd9b7f33ad9c06e914538dbca465493c44d88ddf36954`)
advanced beyond `procfs_missing`, then exposed the dedicated-GPU assumption.
The ownership proof now tolerates continuously changing, provably foreign GPU
processes and compares only positively proved supervisor descendants. Focused
GPU-proof/adapter/provider/RM verification passes **74 tests** with warnings as
errors, and compilation passes. The latest offline image
(`sha256:8bc37a2e60cdbb14c4ed067eeb602b58540af7214bc300dfc257d62ff5293c53`)
built successfully. Its actual run failed closed before readiness with `GPU
baseline ownership could not be observed`: at least one current NVML PID could
not be classified from authoritative procfs. The owned container was removed and
verified absent. Foreign competition is no longer itself a blocker, but actual
inference, residency, cleanup or capacity/readiness success is not yet proved.
Bounded retry/backoff, complete confirmation classification and stable foreign
ancestry fencing now have **83-test** focused regression evidence. A rebuilt
network-disabled image
(`sha256:c1ddec1ae3ab6cb43cf5081423c111da493281aebb9be65c5a101651aacd9a10`)
still failed closed at baseline ownership. A minimal real diagnostic isolated a
persistent graphics NVML PID absent from Linux procfs while Ollama was running;
compute was empty, the other graphics PID was procfs-readable, and pre-Ollama
samples were empty. Exhaustively proving that PID's foreign provenance was later
rejected as beyond G4.1. The proof now ignores unconnected non-supervisor NVML
PIDs and requires strict, stable positive evidence only for service-owned
descendants. Focused GPU-proof/adapter/provider/RM verification passes **88
tests** with warnings as errors, and compilation passes. A new actual run is
required before inference, residency or cleanup advances beyond unproved.
The subsequent goal-focused actual run passed with Ollama 0.11.6: two bounded
inferences completed, one supervisor-descendant GPU runner was positively proved,
the old model became absent after unload, stale-session submission was rejected,
and the identity-fenced daemon group and named container were gone. Two baseline
NVML PIDs were left unrelated/unknown and untouched. Focused verification passes
94 tests. This advances G4.1 candidate evidence only; G4.2 remains unproved.
The focused health/OpenAPI suites passed **15 tests** in **0.144 seconds** with
warnings treated as errors; service/test/tool compilation passed. This proves
only the unwired read-only health boundary with injected state, not production
bootstrap readiness. See `docs/adapter-gpu-check.md` for the exact GPU runtime
boundary and commands.
Review regressions were repaired.
These are unit/local integration results, **not** qualifying provider/GPU E2E.
The expanded three-model RM GPU candidate check passed separately, including
normal lifecycle (41 seconds) and injected child-process loss (36 seconds).
Both scenarios passed again after the bounded raw-input and image-packaging
repairs. SmolLM now proves a conservative no-truncation bound for its configured
printable-ASCII bucket, not arbitrary UTF-8 or model-maximum inputs.
All three failed attempts were fenced and each owned process group disappeared
with the NVML process baseline restored before the next load. Execution-entry
notification is not proof of GPU-kernel entry or interruptibility. This is not
the full compatibility matrix or measured capacity acceptance.
See the [proof inventory](../.opencode/skills/goal-oriented-design/references/e2e-proof.md).
No leaf goal is `done`.

## 2. Scope and fixed decisions

- One `QueueScheduler` instance owns one logical durable queue and exactly one
  model ID: `SmolLM`, `CoEdIT`, or `GECToR`.
- `QueueScheduler` is the client-side scheduling stub. It owns queue ordering,
  eligibility, dependency handling, cancellation intent, and idle-completion
  supervision. It does not load models or manage GPU memory.
- `ResourceManager` is an asynchronous singleton server for the one available
  GPU and is the sole GPU-residency authority. Exactly one QueueScheduler
  session and one model may be active. Model co-residency and fair service among
  competing schedulers are out of scope.
- Queue persistence is independent from consumer-specific input decoding,
  template expansion, provider execution, and result handoff.
- The runtime performs no downloads. Provisioning validates model artifacts in
  the mounted volume and produces reviewed, exact-identity capacity profiles.
- Capacity is static at runtime: the profile's throughput-optimal execution
  concurrency plus a 100% input buffer (`m = optimal_parallelism`), for a total
  ResourceManager admission bound of `2 * optimal_parallelism`.
- There are no absolute request deadlines. A scheduler-scoped idle watchdog
  calls `stop()` after one minute without defined progress in active work.

## 3. Contracts to settle before coding

### 3.1 Durable request and attempt records

Persist an immutable request identity and payload reference before enqueue is
acknowledged. Keep these concepts separate:

1. **Request** — durable client intent, model ID, dependencies, insertion mode,
   readiness/template descriptors, cancellation intent, and result target.
2. **Queue position** — monotonically ordered placement plus insertion metadata.
3. **Attempt** — leased decoding/execution attempt with a residency generation,
   timestamps, and provider correlation ID.
4. **Result handoff** — idempotency key, output reference, and consumer-facing
   handoff outcome. A provider response alone does not make a request `done`.

Keep storage, scheduling, provider execution, and result handoff behind separate
interfaces (separation of concerns). Use atomic state transitions and an opaque
attempt token so delayed asynchronous callbacks cannot complete a newer
attempt. Duplicate enqueue, dispatch, completion, cancellation, or result events
for the same idempotency/attempt identity are successful no-ops. Store large
payloads/results outside the queue row and refer to them by durable identity.

Use SQLite in WAL mode as the v1 durable QueueStore as well as for profiling.
One logical `scheduler_id` owns its queue. Restarting that scheduler identity
recovers every non-terminal request, reconciles attempts with ResourceManager,
and resumes work; starting a different scheduler identity supersedes and aborts
the old queue. A transactional outbox records ResourceManager submissions and
result handoffs. Recovery replays unacknowledged outbox entries idempotently,
expires attempts that no longer match the active ResourceManager session, and
never assumes a missing acknowledgement means an operation did not happen.

The implementer must document the settled transition table beside the code and
test every allowed and rejected transition. Concrete persistence schemas need
not be duplicated in design documentation; their typed implementation is the
authority.

### 3.2 Public lifecycle

The required public statuses are:

| Status | Meaning |
|---|---|
| `scheduled` | Durably accepted, but no consumer currently owns an attempt. This includes dependency-, readiness-, or template-blocked work. |
| `running` | A consumer owns a valid lease and is decoding, templating, waiting for ResourceManager admission/model load, executing, or publishing. |
| `on_gpu` | ResourceManager has admitted this request into an execution slot for the matching resident-model generation. Buffered requests are still `running`. |
| `done` | Required result handoff completed idempotently. |
| `error` | Work terminated without a publishable result because of idle abort, scheduler supersession, retry exhaustion, dependency/function failure, or provisioning/admission failure. Explicit client cancellation is not an error. |
| `cancelled` | QueueScheduler stopped handling the request and fenced any late result. ResourceManager cleanup is best effort and underlying provider work may still finish. |

`running_at` is set when work first leaves the scheduled queue and `done_at`
after successful result handoff. Expose `running_to_done_ms`. ResourceManager
measures `time_on_gpu_ms` from execution-slot admission until provider execution
leaves the GPU slot and returns it with the attempt result. If cancellation,
process loss, or provider limitations prevent a complete measurement, return
`time_on_gpu_ms: null` and `gpu_timing_complete: false`; missing telemetry never
blocks a valid result from becoming `done`. Retries retain per-attempt timings.

Cancellation is accepted idempotently in every status. For terminal requests it
is a no-op returning the existing record. For non-terminal requests,
QueueScheduler stops handling the request, marks it `cancelled`, asks
ResourceManager to clean it up, and rejects all later output for its attempt
token. Cancellation is best effort after provider execution starts: underlying
work may finish, but it cannot publish or change the terminal status.

ResourceManager classifies every failure as retryable or non-retryable.
QueueScheduler retries retryable failures with persisted elapsed retry time and
exponential backoff beginning at five seconds and capped at thirty seconds.
After five cumulative minutes from the first retryable failure, the request
becomes permanently failed with non-retryable `retry_exhausted`. Waiting because
the ResourceManager buffer is full is backpressure—not a failed attempt, error,
or retry—and does not consume the retry budget. Any non-retryable
ResourceManager failure is an outer-boundary failure: QueueScheduler invokes
`stop()`, aborts its complete queue, cancels dispatched work best effort, and
stops submitting data. During backoff a request returns to `scheduled` with its
next-attempt time and keeps its original queue position.

### 3.3 Completed-response watchdog and idle abort

The watchdog answers one operational question: work is eligible or in flight,
but no provider request is finishing, so the scheduler must stop and preserve
evidence for investigation. Provider progress has one uniform definition for
SmolLM, CoEdIT, and GECToR: **a provider response finished**. A finished response
increments a completion sequence on the active QueueScheduler session and
resets the one-minute watchdog.

No intermediate activity resets it. In particular, enqueue/dispatch, lease
renewal, model load/unload, buffer or GPU-slot movement, streamed tokens,
microbatches, provider heartbeats, polling, and result-handoff activity are not
watchdog progress. This deliberately stops a scheduler when no complete response
has been observed for one minute, even if internal work appears active; traces
and durable state are then available to investigate why nothing finished.

Arm the watchdog when at least one request is eligible or dispatched. If the
completion sequence does not advance for one minute, QueueScheduler calls
`stop()` with reason `idle_timeout`, marks unfinished local requests
`error(idle_timeout)`, asks ResourceManager to cancel dispatched work best
effort, and stops submitting. An empty queue, or a queue containing only work
waiting on dependencies or readiness logic, does not arm the watchdog. A
provider response fenced out by cancellation, supersession, or attempt/session
mismatch is still evidence that execution finished and may reset the watchdog,
but it cannot publish a result or change the request's terminal state.

### 3.4 Ordering, dependencies, readiness, and templates

The durable structure is an ordered list with graph references:

- A request is eligible only when every dependency is `done`, its ready
  function evaluates true, and its template function can produce valid provider
  input from durable dependency results.
- Dependencies are request IDs within the same QueueScheduler. Cross-scheduler
  coordination is not built in; consumers may implement it in optional logic.
- A dependency ending in `error` or `cancelled`, or a missing/cyclic dependency,
  produces a structured request error; it must not wait forever. Detect cycles,
  but impose no graph-depth limit.
- Optional ready/template logic uses a consumer-friendly descriptor containing
  a registered function name, JSON-compatible arguments, and referenced
  dependency result IDs. These IDs are declared same-scheduler dependency request
  IDs, to be resolved to their acknowledged durable result references, not file
  paths or arbitrary result identities. The consumer owns registration and execution.
  Templates must support programmatic composition comparable to Go templates.
  Function errors are request errors; a function may intentionally consult
  external state, so purity is not required. QueueScheduler reevaluates blocked
  optional logic on relevant local changes and a configurable poll interval;
  false readiness does not count as progress or arm the idle watchdog.
  Registration is application startup configuration. A recovered request whose
  named function is unavailable terminates with `function_unavailable`; changing
  a function implementation while its durable requests exist is unsupported in
  v1 and requires draining or cancelling that scheduler first.
- Dispatch scans from the front and chooses the first eligible `scheduled`
  request. Ineligible nodes are skipped without being reordered, so FIFO holds
  among requests that are eligible at a dispatch decision.
- Use dependency-cycle detection on enqueue and bounded scan slices so many
  blocked nodes cannot monopolize scheduler work. A continuation is valid only
  while the queue/eligibility version is unchanged. Every dependency, readiness,
  template, cancellation, or insertion event that can affect eligibility
  invalidates it and restarts from the earliest affected position; resumption
  must never bypass an earlier node that became eligible.

Insertion modes are defined as:

- **append**: place the new node at the durable tail.
- **skip-line**: place the new node immediately before the first node whose
  current status is `scheduled`; if none exists, append it. Existing nodes keep
  their relative order. “First” is resolved transactionally at insertion time.

Skip-line changes queue priority but never bypasses eligibility rules. Repeated
skip-line requests use FIFO insertion order: the first request inserted for a
given scheduled anchor remains first. The effective insertion point is the tail
of an existing skip-line group immediately before its original anchor, not the
first inserted member. Persist the group's original anchor, current tail, and a
monotonic insertion sequence, and serialize conflicting insertions. If the
anchor leaves `scheduled`, close that group; the next insertion resolves a new
first-scheduled anchor transactionally. Thus serial and concurrent operations
produce the same order. Idempotency keys prevent a retry from inserting the
same request twice.

## 4. Component design

### 4.1 QueueScheduler

Expose asynchronous operations to start, enqueue, inspect/watch, cancel, and
stop a
model-scoped scheduler. Internally run bounded loops for:

1. eligibility evaluation and lease claiming;
2. ResourceManager submission, never exceeding the server's advertised buffer;
3. lifecycle/progress persistence and result handoff;
4. cancellation reconciliation and expired-lease recovery;
5. the scheduler-session idle watchdog.

`start()` creates a new opaque ResourceManager session for this scheduler and
model. Restarting the same durable `scheduler_id` creates a new ResourceManager
session and reconciles its persisted queue. Starting a different scheduler ID
supersedes the old scheduler and aborts the old durable queue. ResourceManager
invalidates and discards all buffered state from the old session, best-effort
cancels its active work, unloads its model, and accepts only the new session
token. Calls carrying the old token are synchronously rejected as non-retryable
`scheduler_superseded`; the old QueueScheduler must `stop()` and abort
completely. Clients must not run competing QueueSchedulers against one GPU.
Throughput—not fairness between competing schedulers—is the objective.

`stop()` is idempotent: atomically prevent new dispatch, terminalize every
non-terminal request in the scheduler's durable queue (including blocked,
scheduled, retry-delayed, buffered, and executing work) with the supplied
`error` or `cancelled` reason, request best-effort cancellation for every
dispatched attempt, fence late results, and release the session. It does not wait
indefinitely for provider work that is too late to interrupt.

### 4.2 ResourceManager

Use a server-side state machine for the one active session:

`idle -> draining -> unloading -> loading -> ready -> draining ...`

When a new scheduler starts, stop old admission, discard its buffer, cancel
active work best effort, unload its model as far as the provider permits, load
the new scheduler's model, then open admission. Cleanup verification must pass
before loading another model or opening admission; failure leaves
ResourceManager unavailable and returns a non-retryable error to the new
scheduler. The opaque session and attempt tokens fence asynchronous callbacks;
old callbacks can never publish.

Admission has two distinct bounds for the resident model:

- execution semaphore: profile `optimal_parallelism`, obtained only when status
  becomes `on_gpu`;
- input buffer: `m`, containing decoded requests ready to acquire an execution
  slot. Total accepted by ResourceManager is therefore at most
  `optimal_parallelism + m`.

Additional work remains durably scheduled/running at QueueScheduler, creating
backpressure rather than an unbounded second server queue. Validate request
context/input size against profile limits before buffering.

### 4.3 Provider adapters

Define one lifecycle interface: artifact validation, load, readiness check,
execute, best-effort cancel, completed-response callback, unload, and cleanup
verification.

- **SmolLM** — exact artifact
  `LLMs/HuggingFaceTB/SmolLM2-1.7B-Instruct/SmolLM2-1.7B-Instruct-Q8_0.gguf`,
  served through a locally imported Ollama model. Streaming chunks do not count
  as watchdog progress; only the finished provider response does. Close the
  stream on cancellation and use `keep_alive: 0` for
  unload; neither guarantees immediate interruption, so session/attempt fencing
  remains authoritative. Include the scheduler's context-size estimate in
  profile selection and admission.
- **CoEdIT** — exact artifact
  `LLMs/grammarly/coedit-large/model.safetensors`, run with
  `AutoTokenizer` and `T5ForConditionalGeneration` in local-files-only mode.
  Inputs are instruction plus text; outputs are aligned edited strings. Native
  tensor batching is the primary concurrency mechanism. Cancellation is checked
  between batches; in-flight CUDA generation may finish.
- **GECToR** — exact artifact
  `LLMs/gotutiyan/gector-deberta-large-5k/model.safetensors`, run with the
  `gector` package using local tokenizer files and the required local
  `verb-form-vocab.txt`. Inputs are source strings plus explicit thresholds,
  iteration count, and batch size; outputs are aligned corrected strings.
  Cancellation is checked between batches/iterations where the package allows;
  in-flight CUDA work may finish.

Adapters must not download fallback assets. Tokenizers/configuration and all
transitive files needed to load must also exist in the volume manifest. Pin the
production Ollama, Python, PyTorch, Transformers, CUDA, and gector dependencies.
Set `HF_HUB_OFFLINE=1`, use `local_files_only=True`, select safetensors
explicitly, use one explicit CUDA device, and reject silent truncation. The
profile fixes dtype, generation parameters, request-size bucket, and whether a
capacity unit is one request or one native batch.

### 4.4 Offline provisioner, bootstrap, and capacity profiles

Provisioner configuration contains one or more model entries:

```yaml
models:
  - id: SmolLM
    parentPath: LLMs/HuggingFaceTB/SmolLM2-1.7B-Instruct
    modelFile: SmolLM2-1.7B-Instruct-Q8_0.gguf
    resourceManagerType: ollama
    contextSizeEstimates: [2048, 4096, 6144, 8192]
    benchmarkRequests: [] # required only when generation is unavailable
```

`resourceManagerType` selects `ollama`, `transformers-coedit`, or `gector`.
Non-Ollama entries omit `contextSizeEstimates` and instead require benchmark
buckets. CoEdIT buckets specify maximum input tokens, maximum output tokens,
generation parameters, dtype, and native batch shape. GECToR buckets specify
maximum subword tokens, iteration count, thresholds, dtype, and native batch
shape. A request is admitted only under an exact compatible bucket whose limits
cover it. Paths are host inputs to the provisioner, not runtime download
locations. The provisioner inventories the parent folder, copies only adapter-
required files into a deterministic Docker image or volume layer, records
hashes, and rejects missing transitive artifacts.
For Ollama it creates/imports a deterministic local model name from the GGUF and
Modelfile. It never downloads.

Provisioning bootstraps the actual ResourceManager and uses its normal adapter,
residency, admission, timing, and cleanup paths. For each model, and for every
configured Ollama context-size estimate, it performs:

1. Build or validate a maximum-sized valid benchmark request. Prefer an adapter
   generator that creates locally valid input and verifies a valid non-empty
   response. If safe generation is unavailable, require one or more configured
   `benchmarkRequests`; fail rather than inventing unsuitable input.
2. Load the model and run one request at a time four times. Record every sample,
   then calculate mean service time and peak incremental VRAM as the baseline.
3. Increase simultaneous requests by one, repeating representative max-sized
   work and measuring actual peak VRAM. Derive the memory-safe upper bound `N`
   for this exact model, GPU, and optional context size while retaining a 20%
   VRAM safety reserve. Stop on OOM, invalid output, cleanup failure, or
   the derived memory bound; the failed point is not admissible.
4. Measure throughput at concurrency values from 2 through `N`, always including
   2 and `N`, with at most ten values spaced by approximately ten percent of the
   range and rounded to distinct full integers. If `N < 2`, record only `N=1`.
   At each concurrency `n`, run four measured waves of `n` max-sized requests
   after one unmeasured warmup wave, so all `n` slots are populated. Throughput
   is total successful requests divided by wall-clock measured-wave duration.
   Any invalid response, failed request, OOM, or cleanup failure invalidates that
   point. Retain raw latency, completion, error, and VRAM samples. The runner
   must prove real simultaneous provider execution rather than queued
   submissions.
5. Select `optimal_parallelism` as the successful concurrency with the highest
   aggregate handled-requests/second across its four measured waves. A point is
   successful only if every request in all four waves returns a valid response.
   To avoid selecting noise, another point replaces the lower concurrency only
   when its aggregate throughput is at least 2% higher; otherwise choose the
   lower concurrency. `N` remains the memory-safe upper bound and is retained
   for diagnostics; runtime executes at `optimal_parallelism`, not
   automatically at `N`.

Profiles are stored durably in SQLite and looked up by model ID, GPU identity,
and optional context size. They also record model/artifact hashes, adapter and
runtime identity, test-request fingerprint, baseline samples, memory-safe `N`,
selected `optimal_parallelism`, buffer `m`, safety reserve, raw sweep samples,
and creation time. No migration framework is required for v1: incompatible
profile schema changes recreate the profiling database.

For Ollama, every request must provide a positive context-size estimate.
Profile selection uses the smallest configured context size greater than or
equal to that estimate. Estimates above the largest configured size are
rejected; a smaller profile is never used. Non-Ollama adapters use their
profile's explicit maximum input/output or iteration bucket.

Runtime fails closed when no matching profile exists or validation metadata no
longer matches. It does not benchmark, adapt capacity, or download artifacts.

## 5. API boundary

Choose a concrete transport during implementation, but preserve these protocol
operations:

- scheduler: `start`, `enqueue`, `get`, `watch`, `cancel`, `stop`;
- ResourceManager: `start_session`, `submit`, `cancel_request`, `stop_session`,
  `watch_progress`, `get_capacity`;
- provisioner: `verify_artifacts`, `measure_capacity`, `validate_profile`.

Use asynchronous HTTP/1.1 with JSON request/response bodies and Server-Sent
Events for resumable status/progress watches. Bind ResourceManager on one
configured application port; Ollama remains loopback-only. Define the complete
external API in a checked-in OpenAPI YAML document. Every
mutating operation needs an idempotency key. Session and attempt tokens fence
stale asynchronous calls. Progress/status events need sequence numbers and a
resume cursor so reconnect cannot omit or reorder observable state. Keep v1
schemas simple; do not add migration or multi-version negotiation machinery.

Result handoff is a QueueScheduler plug-in interface with
`publish(request_id, attempt_token, result_reference, idempotency_key)`. Before
invoking it, write result bytes to a temporary file in `/var/lib/llm/results`,
`fsync`, atomically rename to a content-addressed final path, then transactionally
store that immutable reference and an outbox item in SQLite. Replay the outbox
until the plug-in acknowledges the idempotency key; only then mark `done`.
Plug-ins must make duplicate keys no-ops. V1 ships a local-result publisher;
other sinks implement the same contract.

## 6. Production Docker image specification

The existing `.devcontainer/Dockerfile` is a development image and must not be
extended into the inference image. Add a separate production Dockerfile, for
example `deploy/docker/Dockerfile`, plus `.dockerignore`, a hash-locked Python
dependency file, an entrypoint, and a small process-supervisor configuration.

### 6.1 Compatibility spike and pinned inputs

Before fixing the base image, run one bounded compatibility spike against the
target machine and record this matrix:

- GPU model/UUID, compute capability, VRAM, host driver, and NVIDIA Container
  Toolkit;
- CUDA runtime image and digest;
- Python, CUDA PyTorch wheel, Transformers, tokenizers, safetensors, and gector;
- Ollama release and archive digest;
- all three exact model artifacts, including the GECToR verb vocabulary.

The chosen combination must load, infer, cancel best effort, unload, switch all
three models, report VRAM through NVML, and run with external networking
disabled. Pin the final base image by digest, Python dependencies by hashes,
Ollama by release plus SHA-256, and model inputs by the provisioner manifest.
Do not install Ollama with `curl | sh`.

### 6.2 Multi-stage image

Use an official `nvidia/cuda:<selected>-runtime-ubuntu<selected>` final base;
use a matching `devel` image only in a builder if a dependency must compile.
Never install a host NVIDIA driver in the image. The stages are:

1. **python-deps:** build an application wheel and offline wheelhouse/virtual
   environment from the hash-locked dependency set. Build access may use the
   internet; the result must install without it.
2. **ollama-fetch:** fetch the pinned official Linux archive as a discrete build
   input, verify its digest, and extract only runtime files.
3. **application:** copy source and install from the local wheelhouse. Generate
   an SBOM and dependency/model provenance metadata.
4. **runtime:** copy only the application, Python environment, Ollama runtime,
   entrypoint/supervisor, and static configuration into the pinned CUDA runtime
   image. Exclude compilers, package indexes, download caches, source-control
   data, credentials, dev tooling, and unrelated model files.

Large models are provision-time inputs, not ordinary Docker build context. V1
uses a content-addressed model volume rather than embedding weights in the image.
The provisioner reads configured parent-folder mounts, copies selected files
into a fresh staging directory on the same filesystem, verifies hashes and
transitive requirements, writes and `fsync`s the manifest, then atomically
renames the directory to its manifest digest. An atomic `current` symlink selects
the completed digest. Interrupted staging directories are never eligible and
are cleaned on the next run. The source GGUF/Modelfile and their hashes are
authoritative; the derived Ollama store is disposable and reconstructed when
its recorded source digest differs. GPU profiling must never run during
`docker build`; it runs from the built image on the target GPU.

### 6.3 Users, paths, and volumes

Run neither long-lived process as root:

- `llm` owns the application and `/var/lib/llm`;
- `ollama` owns `/var/lib/ollama`;
- both can read `/opt/llm/models` and the generated artifact manifest.

Required paths:

```text
/opt/llm/app                 immutable application
/opt/llm/config              immutable config and artifact manifest
/opt/llm/models              provisioned model volume, read-only at runtime
/var/lib/llm/queue.sqlite3   durable QueueStore
/var/lib/llm/profiles.sqlite3 durable capacity profiles
/var/lib/llm/results         durable large result/outbox payload references
/var/lib/ollama              Ollama's derived local model store
```

`/var/lib/llm` and `/var/lib/ollama` must be durable writable volumes. The
default deployment creates Docker-managed named volumes `llm-provider-state`
and `llm-provider-ollama`, mounted at those paths respectively; operators do not
select or configure host filesystem paths. V1 supports Docker local volumes
backed by local ext4 or XFS only. SQLite storage
must provide POSIX locking, atomic rename, `fsync`, and WAL semantics; network
filesystems are unsupported. Startup runs a WAL/read-write probe, checks free
space, and fails readiness on checkpoint or space exhaustion. Set ownership
during image build and fail startup on incorrect permissions. Models are
read-only after provisioning.

### 6.4 Entrypoint and process supervision

Use `tini` as PID 1 plus a pinned minimal supervisor, or an equivalently small
supervisor that correctly reaps children and forwards signals. It manages:

- Ollama bound only to `127.0.0.1:11434` as user `ollama`;
- the application/ResourceManager as user `llm`.

Startup order:

1. Validate configuration, one visible GPU, NVML access, directory ownership,
   artifact hashes, and SQLite WAL operation. Resolve the one visible NVML
   device to its physical GPU UUID and require an exact profile match. Use the
   UUID in deployment configuration; ordinal `0` is only the in-container
   alias. MIG is unsupported in v1 and is rejected.
2. Start Ollama and wait with a bounded timeout for its local API.
3. Import or validate the deterministic local SmolLM name from the provisioned
   GGUF and Modelfile without registry access.
4. Start ResourceManager and validate adapter metadata plus the profile matching
   the exact GPU/model/context identity.
5. Report ready only when admission is safe. Provisioning mode instead invokes
   the same ResourceManager to generate profiles before readiness.

On `SIGTERM` or `SIGINT`: stop API admission, invalidate the active ResourceManager
session so its external QueueScheduler receives a non-retryable stop signal,
flush ResourceManager state, request best-effort cancellation, wait only for
`LLM_SHUTDOWN_GRACE_SECONDS`, stop Ollama, flush logs/traces, and exit. The
external QueueScheduler then applies its documented full `stop()` behavior.
Do not restart Ollama independently: any unexpected Ollama exit invalidates the
session, makes readiness false, and terminates the container. The orchestrator
may restart the whole container, which revalidates artifacts, profiles, and
residency before accepting a new session.

### 6.5 Runtime environment and networking

Support these environment variables, with configuration-file equivalents:

```text
LLM_ARTIFACT_ROOT=/opt/llm/models
LLM_MANIFEST=/opt/llm/config/manifest.yaml
LLM_STATE_DIR=/var/lib/llm
LLM_QUEUE_DB=/var/lib/llm/queue.sqlite3
LLM_PROFILE_DB=/var/lib/llm/profiles.sqlite3
LLM_GPU_UUID=GPU-...
LLM_PROFILE_REQUIRED=1
LLM_RUNTIME_OFFLINE=1
LLM_IDLE_TIMEOUT_SECONDS=60
LLM_RETRY_INITIAL_SECONDS=5
LLM_RETRY_MAX_SECONDS=30
LLM_RETRY_BUDGET_SECONDS=300
LLM_VRAM_SAFETY_PERCENT=20
LLM_SQLITE_MIN_FREE_BYTES=5368709120
LLM_HEALTH_TIMEOUT_SECONDS=5
LLM_SHUTDOWN_GRACE_SECONDS=60
LLM_TRACE_RETENTION_SECONDS=86400
LLM_LOG_LEVEL=INFO
LLM_LOG_FORMAT=json
LLM_TRACE_ENABLED=0
OLLAMA_HOST=127.0.0.1:11434
OLLAMA_MODELS=/var/lib/ollama
OLLAMA_KEEP_ALIVE=0
HF_HOME=/opt/llm/hf-cache
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=GPU-...
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

Reject zero or multiple NVML-visible devices, MIG devices, and UUID/profile
mismatches. The host supplies GPU access with NVIDIA Container Toolkit and one
UUID-selected device; the image supplies CUDA user-space runtime only.

The runtime needs no outbound network. Keep Ollama private and never expose port
11434. If QueueScheduler is in another container/process, expose only the
ResourceManager application port on an isolated network while denying egress.
Do not bake proxy settings, credentials, or registry tokens into the image.

### 6.6 Health, logs, and image hygiene

Provide:

- `GET /health/live`: PID 1, application process, and event loop are alive;
- `GET /health/ready`: SQLite, GPU, artifacts, active adapter, cleanup state,
  and matching profile permit admission;
- `GET /health/dependencies`: diagnostic Ollama/CUDA/artifact/profile state;
- `GET /metrics`: QueueScheduler and ResourceManager metrics already specified.

Readiness must be false during startup, provisioning, model loading/unloading,
profile mismatch, failed cleanup, or Ollama failure. Health responses must not
trigger model downloads or mutate residency. Every health dependency check has
a five-second timeout and should normally complete immediately from cached local
state rather than running inference.

Emit structured JSON logs to stdout/stderr and optional local performance traces
to the durable state volume. Include scheduler/session/request/attempt IDs,
status transitions, model ID, occupancy, profile identity, timings, VRAM,
cleanup, and error retryability. Do not log prompt/result content or secrets.
Rotate or bound local traces so they cannot fill the SQLite/model filesystem.
Retain traces for one day, then delete them automatically. Development agents
and operators must collect needed traces within that window. Refuse new durable
work and become unready when the state volume has less than 5 GiB free; do not
delete durable queue/results to recover space.

Use `.dockerignore` to exclude `.git`, `.opencode`, credentials, local databases,
caches, and model directories not intentionally supplied to an artifact stage.
Clean OS/package caches, scan the final image, produce an SBOM, and record base,
Ollama, dependency-lock, application, and artifact-manifest digests as image
labels/provenance. Licensing review and enforcement are out of scope because the
operator has obtained the required rights.

### 6.7 Container acceptance criteria

- Build is reproducible from pinned inputs and the final image contains no
  compiler, package cache, credential, or undeclared model artifact.
- The image starts with one selected GPU and fails clearly with zero/multiple or
  profile-mismatched GPUs.
- Provisioning and inference for all three adapters succeed with outbound
  networking disabled.
- Removing or changing any required artifact—including GECToR's verb
  vocabulary—prevents readiness.
- Ollama is reachable only over container loopback.
- Abrupt termination preserves recoverable SQLite queue/outbox/profile state;
  graceful termination observes the configured bound and fences late output.
- Starting a second scheduler supersedes the first without starting another
  ResourceManager or exposing stale results.
- SmolLM -> CoEdIT -> GECToR switching verifies cleanup before each load and
  fails closed when cleanup cannot be proved.
- Liveness, readiness, dependency diagnostics, metrics, logs, and traces reflect
  actual process, GPU, profile, and residency state.

## 7. Compatibility spike

Run this before implementing production adapters or fixing dependency pins. It
is a short executable investigation, not a capacity profile and not product
proof. Use the target GPU, host driver, and NVIDIA Container Toolkit.

### Inputs

- Candidate CUDA runtime image digest and matching PyTorch CUDA wheel.
- Candidate Python, Transformers, tokenizers, safetensors, gector, and Ollama
  versions.
- Exact SmolLM GGUF/Modelfile, CoEdIT safetensors/tokenizer/config, and GECToR
  safetensors/tokenizer/config/`verb-form-vocab.txt` artifacts.
- One smallest valid and one maximum-sized valid request per adapter. Use a
  configured request where safe generation is unavailable.

### Procedure

1. Build the candidate production stages and verify installation from the local
   wheelhouse and verified Ollama archive.
2. Run the image with one GPU selected by UUID and outbound networking disabled.
3. Verify NVML identity/VRAM reporting and CUDA/PyTorch device agreement.
4. For each adapter independently: validate all artifacts, load once, run the
   small request, run the maximum-sized request, validate output, exercise
   best-effort cancellation, unload, and prove cleanup.
5. Run SmolLM -> CoEdIT -> GECToR sequentially through ResourceManager and prove
   no stale callback or allocation crosses a session/model switch.
6. Terminate Ollama and each Python provider path during active work to confirm
   session invalidation, late-result fencing, bounded shutdown, and truthful
   health/readiness behavior.
7. Restart the whole container and prove SQLite queue/profile visibility plus
   deterministic reconstruction of the derived Ollama store.
8. Record exact versions, digests, hashes, observed cancellation limitations,
   load/unload behavior, minimum driver, and any adapter-specific request limits.

### Exit criteria

- One complete compatibility row passes every step for all three models.
- No runtime network request or fallback download occurs.
- GECToR loads the newly supplied local `verb-form-vocab.txt`.
- The selected pins and minimum host-driver requirement are committed to the
  dependency lock, Dockerfile inputs, and artifact manifest.
- Any unsupported cancellation behavior is captured as best-effort behavior and
  proven safe through attempt/session fencing.
- Failure means pins/base image or adapter strategy must change before later
  implementation phases;
  it must not be papered over with per-model untracked environments.

## 8. Implementation phases and exit criteria

### Phase 0 — Resolve contracts

- Select optional-function polling/progress granularity and finalize the local
  result-handoff plug-in details. Provider watchdog progress is already fixed as
  completed provider responses only.
- Write OpenAPI YAML first, before server or client implementation. Then add
  generated or validated client/server bindings, typed persistence schemas, the
  documented transition table,
  optional-function descriptor, provisioner configuration, SQLite profile
  schema, and exact model artifact manifests.
- Exit: transition/invariant review passes and every ambiguity in section 11 has
  an owner and decision.

### Phase 1 — Compatibility spike and image skeleton

- Execute section 7, pin the passing dependency/runtime matrix, and create the
  production multi-stage Dockerfile, entrypoint, supervisor, volumes, and health
  skeleton.
- Exit: all three adapters load and run small/max requests sequentially offline
  through the ResourceManager skeleton on the target GPU.

### Phase 2 — Durable queue core

- Implement request/attempt/result persistence, append and skip-line
  transactions, dependency validation, leases, idempotency, and recovery.
- Exit: crash/restart, duplicate enqueue/delivery, cycle, blocked-head, and
  result-handoff retry tests pass.

### Phase 3 — QueueScheduler behavior

- Implement model scoping, eligibility scans, registered ready/template
  functions, status/timing events, duplicate no-ops, retry classification,
  all-status cancellation, `stop()`, supersession, and idle abort-all.
- Exit: deterministic scheduler tests cover every transition and insertion/
  eligibility interaction.

### Phase 4 — ResourceManager and adapters

- Implement server protocol, session/attempt fencing,
  `optimal_parallelism + m` admission, and the three real adapters with
  cancellation and cleanup fencing.
- Exit: local integration tests prove stale-generation rejection, bounded
  backpressure, and adapter lifecycle behavior.

### Phase 5 — Offline provisioning

- Implement configured parent-folder ingestion, Docker artifact assembly,
  ResourceManager bootstrap, request generation/configuration, baseline and
  scaling algorithms, SQLite profiles, and runtime fail-closed validation.
- Exit: all three models load from the volume with network disabled and an
  identity mismatch prevents startup/admission.

### Phase 6 — Production-boundary proof and operations

- Add metrics for queue depth by eligibility/status, oldest eligible age,
  completed-response age/sequence, cancellation latency, model load/unload, residency
  session, buffer/execution occupancy, timing distributions, throughput, VRAM,
  and cleanup. Add verbose structured debug logs and per-request performance
  traces; v1 has no audit subsystem.
- Run real scheduler-to-server GPU E2E tests, failure injection, restart, and
  capacity sweeps. Add operator runbooks for stuck work and failed cleanup.
- Exit: every leaf goal has independent tests including at least one qualifying
  E2E test with real required service/provider boundaries.

## 9. Verification matrix

| Outcome | Required unit/contract evidence | Required production-boundary E2E evidence |
|---|---|---|
| Durable accepted work | transactional enqueue/outbox, duplicate delivery, retry recovery, and restart reconciliation | restart scheduler/consumer during real work and recover without loss or duplicate result handoff |
| Truthful lifecycle | complete transition matrix, duplicate-event no-ops, retry classifications, and complete/incomplete GPU timing | observe all six statuses and ResourceManager-supplied `time_on_gpu` without allowing missing timing to block completion |
| Cancellation and idle safety | cancel in every status; late-result fencing; `stop()`; progress/no-progress clocks | cancel queued, buffered, loading, and on-GPU work; prove late completion cannot publish; prove idle/non-retryable errors stop the whole scheduler |
| Fair eligible ordering | append/skip-line including concurrent grouped insertions and anchor status change, blocked head becoming ready during a resumed scan, dependency failure/cycle, readiness/template behavior | dispatch a mixed real queue and assert eligible execution order |
| Singleton ownership | new-session invalidation and stale-token rejection | start a competing scheduler, prove synchronous rejection and complete abort of the superseded client |
| Exclusive residency | residency state/generation and stale callback tests | execute SmolLM -> CoEdIT -> GECToR on the target GPU and prove unload/cleanup between them |
| Bounded useful capacity | profile lookup and exact `optimal_parallelism + m` admission/backpressure tests | baseline four-run measurements, incremental memory-safe `N`, required integer sweep points, and throughput-optimal selection on the target GPU |
| Offline readiness | config, artifact selection/hash, missing benchmark request, transitive-file, and SQLite profile tests | network-disabled Docker bootstrap and execution for all three models through ResourceManager |
| Reproducible container | Dockerfile policy, dependency-lock, manifest, permissions, entrypoint, and health contract tests | pinned image on the target NVIDIA runtime starts offline, survives restart, shuts down boundedly, and exposes only the application boundary |

No leaf becomes `done` from unit tests alone.

## 10. Suggested initial repository layout

The queue/contracts source layout is established in this checkout. Retain one
canonical layout rather than creating duplicate service/library implementations.
The following includes both existing and planned boundaries:

```text
services/llm/queue/                 # contracts, store, scheduler
services/llm/resource_manager/      # server, residency, admission, profiles
services/llm/provisioning/          # manifests and offline profile tooling
libraries/managed-llm/adapters/     # provider-neutral adapter interface + 3 adapters
deploy/docker/                      # production Dockerfile, entrypoint, supervisor
deploy/manifest.yaml                # selected artifacts and hashes
.dockerignore
tests/unit/
tests/integration/
tests/e2e/
```

## 11. Decision ownership and remaining confirmation

1. Backup policy and durable request/result retention for the v1 SQLite state
   volume. Its deployment location is settled as Docker named volume
   `llm-provider-state`; no host path is configured.
2. Settled: asynchronous HTTP/1.1 JSON with resumable SSE, as specified in section
   5. Authentication/authorization are out of scope for trusted v1 deployment.
   Platform implementation owns remaining validated bindings.
3. Settled: consumer-registered Python sync/async functions; descriptors contain
   a name, JSON-compatible arguments and dependency result IDs. Application
     implementation owns runtime execution; durable descriptor integration is
     implemented in Slice E and fenced evaluation in Slice F.
4. Settled: `publish(request_id, attempt_token, result_reference, idempotency_key)`;
   local publication durably acknowledges verified content-addressed bytes.
   External publishers own duplicate-key no-ops; queue implementation owns delivery.
5. Settled: configurable optional-function polling, default one second.
6. Exact target GPU/container runtime and pinned Ollama, CUDA, PyTorch,
   Transformers, and gector versions.
7. Hash/provenance for the supplied GECToR `verb-form-vocab.txt` and
   representative configured benchmark requests where generation cannot
   preserve validity. Licensing is explicitly out of scope.
8. Settled: content-addressed model volume, as specified in section 6.2. Operations
   owns the remaining image registry and SBOM/signing policy.

Operations owns decisions 1 and 6; provisioning owns decision 7. Resolution
criteria and the conservative no-automatic-deletion default are recorded in
[implementation decisions](implementation-decisions.md). The target compatibility
matrix must be demonstrated before production adapters or pins are accepted.

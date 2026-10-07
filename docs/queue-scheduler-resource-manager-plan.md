# QueueScheduler and ResourceManager open-work plan

## Purpose and current truth

Build a durable asynchronous LLM-work platform that lets clients and operators
complete accepted work reliably and efficiently on fixed GPU resources without
lost work, misleading progress, or unproved capacity. Bounded phase completion
is not product-goal completion.

Phases 0, 2, 3, 4, and 5 have completed their bounded contract, local-acceptance,
or candidate exits. Phase 5's provisioning scripts and current three-model
measurement matrix were accepted on 2026-10-07, including GECToR at p=1.
Phases 1 and 6 remain open. G0 and every G1.1–G4.3
leaf remain `partial`; there is no qualifying production-boundary E2E. The
[evidence ledger](queue-scheduler-resource-manager-plan-partial_completed.md)
contains the completed-phase records, detailed evidence, historical candidates,
and limitations. See also [implementation decisions](implementation-decisions.md)
and the [goal/evidence inventory](../.opencode/skills/goal-oriented-design/references/e2e-proof.md).

- [x] Phase 0 — contracts and validated bindings — [ledger record](queue-scheduler-resource-manager-plan-partial_completed.md#phase-0-complete--contracts-and-validated-bindings)
- [x] Phase 2 — durable queue core — [ledger record](queue-scheduler-resource-manager-plan-partial_completed.md#phase-2-complete--durable-queue-core)
- [x] Phase 3 — QueueScheduler behavior — [ledger record](queue-scheduler-resource-manager-plan-partial_completed.md#phase-3-complete--queuescheduler-behavior)
- [x] Phase 4 — ResourceManager and adapter local/candidate exit — [ledger record](queue-scheduler-resource-manager-plan-partial_completed.md#phase-4-continuation--local-exit-and-exact-gpu-candidate-complete)
- [ ] Phase 1 — compatibility and production image
- [x] Phase 5 — offline provisioning scripts and current measured profile matrix — [bounded completion record](queue-scheduler-resource-manager-plan-partial_completed.md#phase-5-candidate-provisioning-complete--2026-10-07)
- [ ] Phase 6 — production-boundary proof and operations

## Fixed constraints for the remaining work

- One visible GPU, one active QueueScheduler session, and one resident model;
  ResourceManager is the sole residency authority. Model co-residency and
  competing-scheduler fairness are out of scope.
- Runtime is offline and fail closed: no downloads or fallback assets;
  artifacts, runtime, GPU UUID, adapter, request bucket, and profile identity
  must match exactly. The selected GPU UUID, not ordinal `0`, is authoritative.
  GECToR's `verb-form-vocab.txt` is a required artifact.
- Session and attempt generations fence stale callbacks. Durable queue/outbox
  handoff is authoritative: accepted work, results, and acknowledgements survive
  restart and are replayed idempotently; a provider response is not `done` until
  result handoff is acknowledged.
- Progress means a completed provider response. `running_to_done_ms` and
  `time_on_gpu_ms` are truthful; incomplete GPU timing is `null` with
  `gpu_timing_complete: false` and never blocks a valid result. Cancellation is
  idempotent and best effort after provider execution starts; late output cannot
  publish. The one-minute watchdog resets only on completed responses and stops
  the scheduler with `idle_timeout` when eligible/in-flight work makes no such
  progress.
- Admission is `optimal_parallelism + m`, with `m = optimal_parallelism`.
  Execution and input-buffer bounds are separate; excess work remains durable
  scheduler backpressure, not an unbounded RM queue.
- Operational invariants retained by deployment and verification are private
  loopback Ollama, durable named volumes, a 5 GiB free-space gate, five-second
  health checks, one-day trace retention, and clean shutdown/session fencing.

Canonical contract and implementation evidence is in the [RM boundary](resource-manager-http.md),
[scheduler boundary](scheduler-http.md), [artifact boundary](artifact-volume.md),
[profile boundary](capacity-profiles.md), and [health boundary](health-http.md).
Decision ownership and settled policy remain in [implementation decisions](implementation-decisions.md).

## Phase 1 — compatibility spike and production image

Complete the target-machine compatibility row and the image/runtime needed by
Phases 5 and 6. Candidate images and candidate pins are evidence only, not
approved production inputs.

- [ ] Record GPU model/selected UUID, compute capability, VRAM, driver, NVIDIA
  Container Toolkit, CUDA image digest, Python/PyTorch/Transformers/tokenizer/
  safetensors/gector versions, Ollama release/archive digest, and exact
  SmolLM, CoEdIT, and GECToR artifacts including the GECToR vocabulary.
- [ ] Approve exact pins and provenance in the dependency lock, image inputs,
  artifact manifest, and SBOM/provenance record. Use offline wheel installation,
  `HF_HUB_OFFLINE=1`, local-only model loading, explicit CUDA device selection,
  and no `curl | sh` or runtime download.
- [ ] Build and accept the actual production image (not a candidate image):
  reproducible digest-pinned CUDA runtime, clean `.dockerignore`, no compilers,
  caches, credentials, or undeclared models, and durable named volumes for
  `/var/lib/llm` and `/var/lib/ollama`.
- [ ] Run long-lived processes as distinct non-root `llm` and `ollama` users;
  keep Ollama on `127.0.0.1:11434`, use `tini`/a pinned supervisor, and verify
  ownership, WAL/fsync/space checks, selected UUID, and offline network policy.
- [ ] Add and verify the privileged broker's measured-parallelism contract so
  the production image can apply the accepted SmolLM p=32 profile. Until then,
  broker startup above p=1 must fail closed; do not silently downgrade capacity.
- [ ] Verify supervisor bootstrap, artifact/profile preflight, Ollama import,
  ResourceManager startup, five-second liveness/readiness/dependency checks,
  and readiness failure on missing/mismatched artifacts, profile, GPU, cleanup,
  or dependency state.
- [ ] With networking disabled, run each adapter's smallest and maximum valid
  input, validate output, exercise best-effort cancellation and cleanup, switch
  SmolLM → CoEdIT → GECToR → SmolLM, and check provider failure, session fencing,
  restart, and reconstruction of durable state.

**Exit:** one complete compatibility row passes for all three adapters through
ResourceManager with exact approved pins/provenance, distinct non-root users,
actual production image/bootstrap/readiness, no network access, valid small and
maximum inputs, and failure/switch/restart evidence. Unsupported interruption
remains explicitly best effort and is safe only through fencing. Failure changes
the pins, image, or adapter strategy before Phase 5/6; it is not papered over by
an untracked environment. The [environment evidence](implementation-decisions.md#environment-reassessment-2026-09-16)
and ledger retain prior candidate limitations.

## Phase 5 — offline provisioning and measured capacity profiles

**Complete for the accepted current matrix (2026-10-07).** This closes the
provisioning scripts, exact offline measurements and audited immutable export,
not production-image approval or deployment. Accepted optimal parallelism is
SmolLM 32, CoEdIT 32 and GECToR 1, with equal buffers and a 20% reserve. GECToR's
fixed p=1 capability is sufficient for this scope; no p>1 extension is required.
The limits apply only to the recorded GPU, artifacts, runtime and request
selectors. New identities or shapes require new measurements.

- [x] Ingest the selected artifact set deterministically from configured
  parent-folder inputs. Copy only required files into a fresh content-addressed
  volume layer, verify hashes/transitive files, require GECToR vocabulary, and
  reject interrupted, incomplete, or changed selections. Never download.
- [x] Bootstrap the **actual** ResourceManager and its normal adapter, residency,
  admission, timing, cleanup, and fencing paths in provisioning mode; do not use
  a profiling-only substitute.
- [x] Generate or validate maximum-sized, adapter-valid benchmark requests and
  configure exact request buckets, dtype, generation parameters, native batch
  shape, context/iteration limits, and identity witnesses. Fail if safe
  generation is unavailable and no configured benchmark request exists.
- [x] For every model and configured Ollama context size, run the exact
  measurement algorithm below.
- [x] Persist an immutable, exact-identity profile in SQLite, including artifact
  hashes, GPU UUID, adapter/runtime identity, request fingerprint and bucket,
  baseline samples, memory-safe `N`, selected `optimal_parallelism`, `m`, 20%
  reserve, raw sweep samples, and provenance. Validate it before runtime use.
- [x] Make runtime fail closed when no matching profile exists or any identity,
  schema, evidence, or artifact metadata changes. Runtime must not benchmark,
  adapt capacity, or download.

### Required measurement algorithm

1. Run four serial baseline requests, retaining every sample and calculating
   mean service time and peak incremental VRAM.
2. Increase simultaneous requests one at a time using maximum representative
   work. Derive the memory-safe upper bound `N` for the exact GPU/model/context
   while retaining a 20% VRAM reserve. Stop at OOM, invalid output, cleanup
   failure, or the derived bound; the failed point is not admissible.
3. Measure `p=1`, then sweep concurrency from 2 through `N`. Include `p=2` when
   applicable and `N`, with at most ten additional points beyond mandatory
   `p=1`; choose optional points as distinct, approximately 10%-spaced integers.
   At every point, run one unmeasured warmup wave followed by four measured
   waves of `n` requests. Measure successful requests divided by the measured-
   wave wall-clock duration, retain latency/completion/error/VRAM samples,
   invalidate any point with an invalid response/failure/OOM/cleanup failure,
   and prove that provider execution was genuinely simultaneous rather than
   merely queued submissions. If `N=1`, measure only `p=1`.
4. Select the highest aggregate handled-requests/second across the four waves;
   a higher concurrency replaces the lower one only when it is at least 2%
   faster, otherwise retain the lower point. Keep `N` as the memory-safe bound,
   but run at `optimal_parallelism`, with `m = optimal_parallelism`.

The bounded CoEdIT p≤32 observations are precursor evidence only: they did not
establish `N`, a production throughput optimum, or profile eligibility. See the
[memory-observation ledger](queue-scheduler-resource-manager-plan-partial_completed.md#memory-observation-precursor)
and [CoEdIT capacity check](coedit-capacity-check.md).

**Bounded exit met:** selected artifacts and the actual RM provisioning bootstrap
pass offline; every
required exact measurement and identity validation produces a durable eligible
profile with a proved memory-safe `N`, 20% reserve, tie-rule selection, and
`m = optimal_parallelism`; runtime accepts only that exact profile and otherwise
fails closed under local regression coverage. The complete actual-GPU matrix,
not merely a candidate image, configured ceiling or synthetic p=1 result,
establishes the measured records. Production-image approval and privileged-broker
capacity integration remain Phase 1 work; canonical production profile selection,
installation and deployed acceptance remain Phase 6 work. This accepted phase
exit does not promote any product goal to `done`.

### Historical status (2026-09-21)

Artifact ingestion is the only Phase 5 checkbox complete. The artifact root
`LLMs/g4-profile-measurement/current` currently selects manifest digest
`7abbd93bd3e4ec01ba01f8e4581821ae1d2f35cab720695c1309596df5614a19`.
The canonical selectors remain `smollm:context512`,
`coedit:p1:input128:output64:float16:beams1:nosample`, and
`gector:p1:tokens128:keep0:min0:iterations1:batch1:float32`. No current
`profiles.sqlite` exists.

The latest actual-GPU continuation again targeted exactly
`GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963`, an RTX 4070 Ti with 12,282 MiB;
bounded preflight found no compute applications. Docker was 29.7.2. The offline
current-source measurement image was built from verified local base digest
`sha256:ceb56141eb754ee162f6861cd13c70b574861cb2ba729ed122332c938e46b2b4`;
the latest image manifest is
`sha256:8483f0c6db1c884005da1805b904409afa636d110166b968bbf5b267201352db`.

The attempt found and fixed a narrow CLI defect: global argparse
`--provenance` had incorrectly blocked non-persistent
`--diagnostic-smollm-p2`; provenance is now required only outside diagnostic
mode. Focused `tests.unit.test_measure_profiles` passed 10 tests in 0.035s and
`py_compile` passed. This is local regression evidence only. The first operator
transfer also omitted the required `current` symlink; that transfer error was
corrected without product changes. Two independently fresh named artifact
volumes then contained the exact relative symlink
`current -> 7abbd93bd3e4ec01ba01f8e4581821ae1d2f35cab720695c1309596df5614a19`
and passed `verify_current` before execution. Diagnostic requests were exact
retained 256 printable-ASCII SmolLM values.

Two fresh offline, host-PID, exact-GPU diagnostic containers ran the required
p=2 command and exited 2 after 25s and 28s, respectively, without bounded
stdout JSON. Owned containers and volumes were verified absent. Since no
bounded model/stage failure envelope was emitted, the cause remains unresolved
at the process/CLI/runtime boundary; this is neither an environment block nor
a proven provider defect. The missing bounded failure envelope is itself the
next investigation target. The complete matrix was correctly not started: no
`profiles.sqlite`, install, export, or promotion exists. All remaining Phase 5
checkboxes stay open and G4.1/G4.2/G4.3 remain partial.

The CLI boundary has since been hardened so argument, configuration, ordinary
application, provider, and cleanup failures retain one deterministic bounded
JSON classification without exposing request data, paths, control text, or
tracebacks. The focused suite now passes 19 tests, and compilation and
`git diff --check` pass. A rebuilt exact-GPU retry nevertheless again exited 2
with zero captured stdout and stderr (`OOMKilled=false`, empty Docker runtime
error), so no provider or capacity conclusion can be drawn from it. The next
retained-container run was stopped before execution because the exact rebuilt
image manifest `sha256:c80f95d331c69df7f54b67895df979f7d166a7f6fe038136b2100258f1eec9fa`
was no longer available locally. This is now an operator image-availability
block: restore or reproducibly rebuild and verify that exact current-source
measurement image, then compare host attachment output with `docker logs` and
`docker inspect` for the same retained container ID before changing provider
behavior. Phase 5 remains open; no profile database or export exists.

The production machinery now exists for the canonical three-entry matrix: the
normal ResourceManager wave runner, exact benchmark preparation and native
evidence extraction, authoritative baseline/discovery/sweep/2%-selection,
immutable SQLite persistence and audit, and owned Ollama lifecycle with atomic
install. Focused local tests verify those paths, but local tests cannot approve
capacity. The actual target for the run is GPU UUID
`GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963`, an RTX 4070 Ti with 12,282 MiB,
driver 591.86, offline networking, and the host PID namespace. The easy
256-byte SmolLM fixture, SHA-256
`34118b6093639020921ad2e46e06332c7bf24c3d70c9e165fa3b4840367d3d70`, has
been proven to produce native `eval_count == 64` with `context512/predict64/temp0`.

The repeated matrix run remains fail closed during SmolLM p>1 discovery:
`invalid discovery: runner_error:provider_execution_failed:worker operation
failed: worker_operation_failed`. It did not create, export, or promote a
profile database, and its all-or-nothing sequencing did not reach the later
models. A subsequent bounded exact-GPU CoEdIT diagnostic through the normal
provider and RM passed provider-class, start/readiness, `validate_input`, exact
benchmark-count/max `128`, one p=1 execution, and native-evidence checks. It did
not reproduce the generic failure; this narrows the blocker but does not prove
CoEdIT capacity.

The remaining investigation is deliberately strict. It must prove native
correlation and workload execution, not merely semantic output grading; fixtures
may be easy while telemetry remains exact. Current errors lose model, stage, and
inner-provider specificity, so `worker_operation_failed` is not actionable and
does not fit the direct SmolLM/Ollama observations. Add only safe, bounded
tracing before changing behavior. Concurrency must be genuine native overlap:
the configured ceiling is not `N`, and unsupported or failing p>1 remains
ineligible. The 20% reserve requires a genuine `N` or resource-bound `N+1`; an
operator ceiling is never promoted. Lifecycle cleanup and all-or-nothing
registry installation also mean later-model evidence may intentionally be
absent. On this Docker Desktop setup, bind mounts are unreliable: use unique
`llm-g4-*` named volumes and `docker cp`, and never touch foreign GPU,
container, or process work.

The matrix runner now has opt-in bounded diagnostic tracing. With `--debug`,
the outer and inner lifecycle records are emitted as JSONL on stderr while
stdout remains the existing single sanitized JSON result. The inner trace is
created exclusively in the fresh runtime volume and retrieved before cleanup;
failed runs retain `<output>.debug.jsonl` without overwriting an existing
sidecar. Both production and retrieval boundaries enforce 4 KiB per record,
20,000 records, and 8 MiB total with a truncation event. The closed schema
rejects request/prompt/output/response/exception/traceback/token/environment/path
content and raw request/GPU identities, and trace failures cannot fail the
measurement. Focused tracing, matrix-runner, and measurement-entry tests pass
91 tests; compilation and whitespace checks pass. This is diagnostic
infrastructure only: no new GPU matrix was run, no profile was produced, every
remaining Phase 5 checkbox stays open, and G4.1/G4.2/G4.3 remain partial.

The exact current-source image was then rebuilt and verified with adapter
identity `a820adef4ccb674a95771733524c5f3a2f341c75a59b9a0509786195395513d0`.
The requested debug matrix completed SmolLM observations through the configured
ceiling of 32, then failed closed after CoEdIT load/readiness but before its first
measurement wave with `maximum_witness_failed`. Owned Docker cleanup was proved,
no profile database was retained, and the pre-existing debug sidecar was
correctly not overwritten; the new trace remained available on stderr. The
failure boundary now distinguishes the closed CoEdIT token-count, configured-
maximum, and payload-fingerprint witness categories and relays them safely on a
subsequent run. Per-request successful trace amplification was removed while all
failures and lifecycle/wave boundaries remain, preventing the earlier 20,000-
record exhaustion. Focused verification now passes 195 tests. The specific
CoEdIT witness predicate still requires one fresh rebuilt run to identify, so
Phase 5 remains open and no capacity result is eligible.

### Provisioning continuation (2026-10-07)

An identity-tracked offline current-source image
(`86539ea6ef25ed8c3a826e2a1b091a62216f095c06ecc69e35112884a78b509b`)
and the prepared bundle passed their verifiers. The fresh full GPU matrix then
completed measurement and per-model persistence for SmolLM, CoEdIT and GECToR,
but failed its final read-only store audit with `CorruptProfileStore`. No database
was promoted; owned Docker cleanup was proved. The bounded trace was not
truncated. This supersedes the earlier CoEdIT maximum-witness blocker, but is
still failure evidence, not an eligible profile or Phase 5 exit.

The audit exposed incorrect SQL parameter ordering in exact-context/profile-
identity lookup. That defect is now corrected with a closed-writer/read-only
three-model audit regression. The outer runner now stages and audits the export
before owned Docker cleanup and reservation checks, then performs no-clobber
promotion; precommit cleanup failures cannot leave a success-path database.
Closed runtime-generated CoEdIT request failures are relayed without exposing
payloads. Ordinary bootstrap now configures supported measured optima and equal
buffers rather than rejecting all `p > 1` profiles, and pins the complete
prepared profile on later lookup. The fixed privileged image broker still
rejects SmolLM `p > 1` because its protocol cannot apply that setting; this is an
explicit production-image integration gap, not a reason to downgrade capacity.
The hardened runner positively checks resource ownership before use/removal,
uses immutable container IDs, and cleans reservation initialization faults.
Independent local verification passes **384 tests in 27.169s**, with changed-
source compilation and whitespace checks passing. This is regression evidence,
not a production deployment or profile approval.
The successive failures below are historical; the successful third run now
establishes an eligible candidate export. The user accepted this bounded Phase 5
completion, including GECToR p=1. Approved production profile selection,
canonical installation and production-image/runtime evidence remain open in
Phases 1 and 6; product goal statuses remain `partial`.

The second fresh run (image identity
`8ed3a0610e4decc965660ff8c49fafb1053c387dc9ddd260845b79f2b82e1ffb`)
completed all three models and the inner store audit, then failed outer result
validation with `result_malformed` in `result_read` after 1,002.547 seconds.
No database was retained and owned Docker cleanup was proved. The producer had
reduced persisted profile objects to identity strings, causing its bounded
summaries to report model `unknown`. It now retains the objects, audits only
their exact identity strings, and emits only model/identity fields. Composed
result-file/CLI-to-outer-validator regressions cover this boundary without
weakening the exact model contract. Interruptions now attempt remaining owned
teardown and release identity-proved host staging/reservations/descriptors while
preserving the original interruption. Neither failed run supplies a usable
profile database; the subsequent fresh run below supersedes these blockers.

The final hardened source snapshot built offline as
`llm-compatibility-adapter:phase5-20261007-131614-final`, with image identity
`8929e8c0655d07f2c4b709b1f71a3a32d376984232e48d3709e57e080f7bb590`.
The handoff image passed the current-source/package-identity verifier.
Independent review found no remaining issues in the scoped fixes. The successful
third measurement operation used its earlier immutable snapshot
`95096d0186b37a1974be9b090c13ba0bb9446f3a5ad7f5550f2e368c62877a7d`;
the subsequent private-staging and SQLite-sidecar hardening has local regression/review
evidence, not execution evidence from that snapshot. Neither image is approved
as the Phase 1 production image.

**Candidate provisioning completed:** the third fresh full matrix exited 0 in
**1,018.468 seconds** (16m 58.468s), with exactly three measured profiles,
`db_retained: true` and `docker_cleanup: proved`. The readonly export is
`.compatibility/profiles-phase5-20261007-123939.sqlite`, mode `0444`.
An additional readonly store audit confirmed the exact matrix and selected GPU;
no WAL, SHM, export staging, reservation lock or failure sidecar remained after
the final owned-sidecar cleanup. The
2,691-record bounded trace was not truncated.

The final workspace check found two historical staging companions left by the
successful snapshot's readonly SQLite audit (0-byte WAL and 32,768-byte SHM).
The operator removed only those verified unique files after readers closed;
profile bytes were unchanged. The final runner now closes the reader, captures
and removes identity-proved audit-created sidecars inside its private stage,
and refuses commit on foreign substitutions or cleanup failure. Real SQLite
regressions exercise this path rather than mocking the audit.

| Model | Exact selector | Memory-safe N | Optimal p | Buffer m | Reserve |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolLM | `smollm:context512` | 32 | 32 | 32 | 20% |
| CoEdIT | `coedit:p1:input128:output64:float16:beams1:nosample` | 32 | 32 | 32 | 20% |
| GECToR | `gector:p1:tokens128:keep0:min0:iterations1:batch1:float32` | 1 | 1 | 1 | 20% |

These are exact candidate records for the selected GPU/artifacts/runtime, not
universal bounds or production approval. They have not been installed as the
canonical production profile set. Exact profile identities and operation
commands are in the [proof inventory](../.opencode/skills/goal-oriented-design/references/e2e-proof.md#phase-5-provisioning-continuation-2026-10-07).

### Next concentration

1. Complete Phase 1 approval of exact pins, provenance and the production image;
   the working provisioning scripts and candidate database do not waive it.
2. Resolve the privileged broker's measured-parallelism contract before using
   the SmolLM p=32 profile through that image. Until then, that path must fail
   closed, never silently downgrade to an unmeasured p=1 configuration.
3. Approve and install the exact immutable profile set through the operator-owned
   canonical selection procedure, then establish composed runtime readiness and
   the Phase 6 production-boundary E2E/operations evidence.
4. For new GPU/artifact/runtime/adapter identities or request shapes, reproduce a
   fresh all-or-nothing matrix with the guide below. Never resume or promote
   partial results; retain a uniquely named bounded failure sidecar if it fails.

### Practical continuation guide

1. Before rebuilding, verify the retained compatibility base-image identity and
   the prepared `.compatibility/adapter-deps/requirements.lock` and `wheelhouse/`
   inputs; this candidate Dockerfile does not create or approve those inputs.
    Then rebuild offline with the identity-tracked helper, naming the
    already-verified local base explicitly. Use a unique run name for the image,
    identity file, output database and failure sidecar; do not reuse the old
    `.compatibility/profiles.sqlite.debug.jsonl`:

    ```sh
    RUN=phase5-$(date -u +%Y%m%dT%H%M%SZ)
    BASE='<verified-local-base>'
    IMAGE=llm-compatibility-adapter:$RUN
    IDENTITY=.compatibility/$RUN-image-identity.json
    OUT=.compatibility/profiles-$RUN.sqlite
    .venv/bin/python tools/compatibility/image_build.py identity \
      --root . --base-image "$BASE" --base-context .compatibility/context \
      --output "$IDENTITY"
    .venv/bin/python tools/compatibility/image_build.py build-adapter \
      --root . --base-image "$BASE" --base-context .compatibility/context \
      --tag "$IMAGE" --identity "$IDENTITY"
    .venv/bin/python tools/compatibility/image_build.py verify \
      --root . --base-image "$BASE" --base-context .compatibility/context \
      --image "$IMAGE" --identity "$IDENTITY"
    ```
    Stop if any command fails. The helper verifies locked wheel inputs, stages
    only identity-listed source/dependencies, and binds the resulting image to
    the inspected base and current source. A successful build is not Phase 1
    production-image approval.
2. Preflight the exact GPU UUID and owned-name namespace. If foreign GPU compute
   work is active, record the bounded fact and abort the uncontended measurement
   without inspecting, stopping, or controlling it. Then create fresh uniquely
   named volumes, transfer inputs with `docker cp`, preserving/creating the
   selected artifact tree's relative `current` symlink and verifying it with
   `verify_current`. Run with `--network=none`, host PID, and the exact UUID.
   Do not reuse partial or foreign state. The image entrypoint already invokes
   `/opt/venv/bin/python`, so Docker arguments begin with
   `/opt/llm/measure_profiles.py`, not a second interpreter. Diagnostic mode
   omits `--provenance`; the full matrix requires it.
3. Rerun the complete three-entry matrix sequentially. Do not resume, export,
    or promote partial temporary results. Capacity stays fail closed unless all
    required evidence and all three entries are eligible.

    For the prepared bundle, the operator runner performs the fresh-volume
    transfer, in-image verification, exact-GPU checks and owned cleanup above:

    ```sh
    .venv/bin/python tools/compatibility/prepare_measurement.py verify \
      --output .compatibility/measurement-bundle
    .venv/bin/python tools/compatibility/run_measurement_matrix.py \
      --debug --image "$IMAGE" --bundle .compatibility/measurement-bundle \
      --gpu-uuid GPU-EXACT_UUID --ollama-version 0.11.6 --ceiling 32 \
      --output "$OUT"
    ```
    Stop if bundle verification fails. Replace `GPU-EXACT_UUID` with the selected
    UUID authenticated by the bundle. On failure, inspect the bounded stderr
    and unique `$OUT.debug.jsonl`; no profile is eligible from a partial run.
4. Only after that succeeds, close providers and the owned daemon, checkpoint,
   perform a read-only audit of the exact matrix and invariants, atomically
   install the immutable database, export the audited copy, set and verify mode
   `0444` on that exported copy, and then update the checklist, goal, and proof
   documents. The current atomic installer does not itself set the export mode.
   Until then, do not accept results for a new identity or shape. Production
   selection and installation approval remains a separate Phase 6 gate.

Testing execution is delegated per project process. The delegated operator must
record the exact commands and their results, including failures and retained
bounded evidence, rather than treating this continuation guide as execution
evidence.

## Phase 6 — production-boundary proof and operations

- [ ] Approve and install the exact immutable measured profile set as the
  canonical production selection after Phase 1 image/pin approval. Verify
  selected identities, readonly export and composed runtime readiness without
  treating the accepted candidate measurements as deployment approval.
- [ ] Deploy the approved production image and run a real
  QueueScheduler → ResourceManager → provider GPU E2E with durable named
  volumes, offline networking, selected UUID, exact artifacts/profile, and
  actual readiness gates.
- [ ] Exercise accepted-work recovery, crash/restart, queued/buffered/on-GPU
  cancellation, provider failure, non-retryable failure, idle watchdog,
  supersession, stale callbacks, and SmolLM/CoEdIT/GECToR switching. Verify no
  lost accepted work, duplicate handoff, misleading status/timing, or stale
  publication.
- [ ] Provide structured logs, metrics, and one-day traces covering queue depth
  by status/eligibility, oldest eligible age, completion sequence/age,
  cancellation latency, residency/load/unload, occupancy, timing, throughput,
  VRAM, cleanup, session, request, and attempt identity. Do not log prompts,
  results, or secrets.
- [ ] Establish backup/retention and restore policy for queue, results, outbox,
  profiles, and named volumes; define registry, SBOM, signing, digest and
  provenance policy. Add runbooks for stuck work, failed cleanup, disk pressure,
  GPU/profile mismatch, provider loss, restart, and rollback.
- [ ] Complete production image/container acceptance: reproducible pinned
  inputs, non-root supervision, loopback Ollama, no egress/downloads, health
  behavior, bounded shutdown, restart recovery, and clean failure on zero,
  multiple, or mismatched GPUs/artifacts/profiles.

**Exit:** independent production-boundary evidence covers every open leaf and
includes at least one qualifying deployed scheduler-to-real-provider GPU E2E.
All failure, recovery, observability, backup/retention, supply-chain, runbook,
and container acceptance criteria pass. This exit still does not by itself
change the goal-tree status without the required goal-level proof.

## Implemented versus open API boundary

Implemented bindings include scheduler lifecycle/queue/watch operations,
ResourceManager session/submit/cancel/stop/progress operations, artifact
verification, exact profile validation, and liveness/readiness/dependency
health. Their schemas and evidence are linked from the [RM boundary](resource-manager-http.md),
[scheduler boundary](scheduler-http.md), and [ledger](queue-scheduler-resource-manager-plan-partial_completed.md).

The following are explicitly **unimplemented future operations**, not promises
that Phase 0 or local bindings delivered them:

- `GET /resource-manager/capacity`
- `POST /provisioning/measure-capacity`
- `GET /metrics`

After Phase 5, capacity may come only from a durable exact-identity measured
profile. No eligible measured profile currently exists; bounded/default runtime
behavior remains in effect. Operational metrics likewise require Phase 6
deployment instrumentation. Neither is inferred from a candidate or precursor
run.

## Production-boundary verification matrix

| Outcome | Required boundary verification |
|---|---|
| Durable accepted work | Kill/restart the deployed scheduler or provider during real work; recover the exact queue/outbox and publish once. |
| Truthful lifecycle/timing | Observe all statuses, completed-response watchdog behavior, cancellation fencing, and complete/incomplete GPU timing from the real RM. |
| Safe cancellation/failure | Cancel queued, buffered, loading, and on-GPU work; inject provider/RM loss and idle/non-retryable failure; prove stop, cleanup, and no late publication. |
| Singleton/exclusive residency | Supersede a live scheduler and switch all three models; prove stale rejection and cleanup before each load. |
| Measured bounded capacity | Reproduce four baselines, incremental safe `N`, warmup/four waves, real simultaneity, 2% rule, exact profile, and `optimal_parallelism + m`. |
| Offline readiness/container | Start the signed approved image with networking disabled; verify UUID/artifacts/profile, non-root users, named-volume restart, five-second health, 5 GiB gate, and private Ollama. |
| Operations | Restore from backup, exercise retention/one-day traces, inspect logs/metrics/traces, and execute runbooks for disk, GPU, cleanup, provider, and rollback failures. |

No unit, local integration, candidate-image, bounded-memory, or synthetic-provider
result alone satisfies this matrix. Implementation locations and decision
ownership are the linked boundary documents, [implementation decisions](implementation-decisions.md),
and the [evidence ledger](queue-scheduler-resource-manager-plan-partial_completed.md).

# QueueScheduler and ResourceManager bounded evidence ledger

This ledger records completed phase exits and bounded locally verified topics,
without treating local phase completion as a completed product goal. The canonical open plan
is [queue-scheduler-resource-manager-plan.md](queue-scheduler-resource-manager-plan.md);
return there for requirements, contracts, open checklists, exits, and decision
ownership.

## Interpretation and non-claims

G0 (durable, truthful, capacity-aware asynchronous LLM work) and G1–G4 remain
`partial`. **No leaf goal is done.** The evidence below is unit, local
integration, synthetic, candidate-image, or locally run installed-adapter
evidence. It is not a production deployment, approved production capacity profile, or
qualifying production-boundary E2E. Phases 0, 2, 3, 4 and 5 have met their respective
contract, local acceptance and candidate exits.
The completed candidate matrix does not waive the compatibility procedure,
remeasurement for changed GPU/artifact/runtime/adapter/request identities,
production image acceptance, restart/failure E2E, or operations work in the
[canonical plan](queue-scheduler-resource-manager-plan.md).

## Phase 5 candidate provisioning complete — 2026-10-07

**Accepted bounded phase closure:** the user accepted the provisioning scripts
and current measured three-model matrix, including GECToR at p=1. No higher
GECToR concurrency is required for this scope. Production image/pin approval and
privileged-broker capacity integration remain explicit Phase 1 gates; canonical
production profile installation and deployed acceptance remain Phase 6 gates.

The fresh offline exact-GPU matrix completed all three models, the inner
read-only profile-store audit, lifecycle cleanup, outer result validation,
export audit and no-clobber readonly promotion. It exited 0 in **1,018.468s**
with `docker_cleanup: proved`. The candidate database
`.compatibility/profiles-phase5-20261007-123939.sqlite` contains exactly three
measured profiles at mode `0444`; no WAL/SHM, temporary export, lock or failure
sidecar remained after final audit and owned historical-sidecar cleanup. All
three records retain a 20% reserve:
SmolLM context512 and CoEdIT's canonical input128/output64 bucket have
`N=32, optimal_parallelism=32, m=32`; GECToR's fixed bucket has `N=1, p=1, m=1`.

Two earlier fresh runs exposed and now have regressions for exact-context SQL
parameter ordering and model names lost from identity-only result summaries.
Runner ownership, cleanup-before-commit, interruptions, private staging and
closed diagnostics were hardened. Independent final local verification passed
**384 tests in 27.169s**, compilation and whitespace checks; final scoped review
found no remaining issues. The successful GPU run's immutable image identity is
`95096d0186b37a1974be9b090c13ba0bb9446f3a5ad7f5550f2e368c62877a7d`.
The later final source image includes additional private-staging/real SQLite
sidecar cleanup hardening with
local test/review evidence, not execution evidence from that measurement run.

This completes the working candidate provisioning scripts and candidate export,
not Phase 1 approval, canonical production profile installation, composed
production readiness or qualifying goal E2E. In particular, the privileged
broker still fails closed for SmolLM p>1. G0/G4.1/G4.2/G4.3 remain `partial`;
closing this bounded Phase 5 section does not close those production gates.
Exact commands/profile
identities and cleanup ownership are in the
[proof inventory](../.opencode/skills/goal-oriented-design/references/e2e-proof.md#phase-5-provisioning-continuation-2026-10-07);
continue with the [open plan](queue-scheduler-resource-manager-plan.md#next-concentration).

## Phase 0 complete — Contracts and validated bindings

Closed on 2026-09-17 against the contract-phase exit: settled optional-function
polling/completed-response progress and acknowledged local publication; parsed
OpenAPI and validated handwritten bindings; typed request/attempt/function,
provisioner configuration, selected manifests and SQLite profile contracts;
transition/invariant review; and an explicit owner/decision for all eight
decision items. The [decision record](implementation-decisions.md) links the
canonical source and invariant tests rather than duplicating schemas.

The new ten-test acceptance module validates actual RM capacity/progress/error
serializers and durable scheduler projections, the complete declared operation
inventory, idempotency/body/cursor requirements and explicit future markers.
Actual loopback checks validate scheduler SSE data, RM 429 backpressure,
artifact verification success/hash mismatch, profile validation success/bad
input, and health live/ready/dependencies including unready responses against
OpenAPI. A negative test confirms malformed payload rejection. Independent
contract/profile/artifact/bootstrap and HTTP suites supply behavioral evidence.

Verification: **145 tests passed in 5.562s**, warnings treated as errors, using
only task-related suites. The **10-test** acceptance module passed two additional
runs (**0.310s, 0.311s**); targeted compilation and `git diff --check` passed.
Fixtures own temporary SQLite/results/artifacts/profiles, fake providers,
coordinator tasks and loopback services. Exact commands and stable tests are in
the [proof inventory](../.opencode/skills/goal-oriented-design/references/e2e-proof.md#phase-0-contract-closure-2026-09-17).

This supports G1.1/G1.2/G2.1/G2.2/G3.1/G3.2/G4.1/G4.2/G4.3 under G0 without
promoting any beyond `partial`. Phase 1 still owns complete compatibility and
pin approval; Phase 5 owns measured provisioning; Phase 6 owns operator backup/
retention, deployment and qualifying E2E. No automatic deletion, invented hashes,
synthetic-profile promotion or implementation of future capacity/measurement/
metrics operations is implied by contract closure.

## Phase 4 continuation — Local exit and exact GPU candidate complete

Fresh exact candidate image
`sha256:da9b047a4394cdd91d565ee00c288e244ec5c12655229c9c8465830e9c9b0d1f`
completed SmolLM → CoEdIT → GECToR → SmolLM with three cleanup-gated
switches, cancellation and stale-authority fencing, six stable-total memory
points, final used-memory restoration to baseline, and verified owned-container
cleanup. The profile remains unmeasured; this is candidate evidence, not
production deployment or capacity proof.

The completed candidate above supersedes the intermediate blocked runs below.
Phase 4's local/candidate exit is complete, supporting G2.1/G2.2/G4.1/G4.2 under
G0 without promoting any leaf beyond `partial`. The following is historical
implementation and failure-diagnostic evidence, not a current blocker.

- The bounded CoEdIT transition-1 fallback now treats only typed expected-runner
  absence as eligible for fallback. It retains exact worker PID/start-time and
  CUDA/NVML witness checks, the captured supervisor/device fence, a valid
  pre-load whole-device point, and two ordered post-load points. Point-in-time
  used/free fluctuation is allowed, but GPU/supervisor/total-memory identity must
  be immutable and each post-load used value must exceed baseline. This is local
  implementation and regression coverage only; no GPU candidate run was performed
  or promoted by this change.

- Fixed phantom validation reservations left by accepted-pair replay; concurrent
  same-attempt validators now share identity-checked reference accounting.
  Conflicting inputs cannot replace that identity, and a failed/cancelled
  validator releases only its own reference.
- Kept validation ownership through the final admission lock. Cancellation
  fences all overlapping validating, buffered and active ownership and invokes
  advisory cancellation for active work. Active-only cancellation does not emit
  a premature terminal event: the actual late response retains completed-response
  progress with exact identity and no publishable result.
- Known retired start replay fails promptly without waiting behind replacement
  cleanup. Existing lifecycle serialization still owns new starts.
- Local acceptance invokes real SmolLM/CoEdIT/GECToR constructors and lifecycle
  methods with a test-owned selected-artifact volume, loopback Ollama and
  controlled worker/GPU seams. SmolLM CLI import/input-bound checks are patched
  here and independently covered by provider suites. This is not GPU inference.
  Separate acceptance proves p=2 plus m=2, exact accepted replay, backpressure
  retry and stale controls while replacement cleanup is gated. The two former
  acceptance gaps are closed: one composed case proves failed cleanup rejects
  submit/cancel/capacity controls and grants no replacement authority; another
  proves a real local adapter's late completion after cancellation is collected
  without publishing a result.

Verification: **176 tests passed in 9.248s**, warnings treated as errors, with
only task-related suites. The four-test acceptance module passed three additional
runs in **0.234s, 0.237s and 0.236s**. Targeted compilation and whitespace checks
passed. Cleanup diagnostics preserve the primary transition failure and report
only bounded categories. Cleanup disappearance and post-load residency use
bounded 5-second/200-ms observational settlement; persistent or uncertain proof
still fails closed. Stable commands and test IDs are in the proof inventory.

The final current-source adapter image built with `--network none`; RepoDigest:
`llm-compatibility-adapter@sha256:2a02467c1f1e60e6445109fd65ff1ec813817c5b193103dd51067201c2b76e1c`.
Instrumentation localized the former ambiguous cleanup failure to transition 1,
SmolLM → CoEdIT. The pre-load proof initially observed the prior owned runner;
bounded settlement now handles that teardown lag without weakening ownership.
The final offline GPU run advanced through that gate but failed closed during
CoEdIT `start_session` after the bounded post-load settlement with `GPU has no
resident runner`. No passing JSON summary or later-model evidence was emitted.

The tester verified owned container `llm-phase4-three-model-check` absent after
each bounded run; no foreign GPU workload was controlled. The remaining blocker
is now the real CoEdIT worker's absent positive NVML residency evidence after the
five-second bound, not an unidentified cleanup transition or masked stale-session
error. Operator/runtime investigation is required before another GPU run; a
devcontainer rebuild is still not justified by this evidence. Earlier candidate
passes remain historical evidence, not a substitute for a current passing run.

## Phase 2 complete — Durable queue core

Closed on 2026-09-17 against both the top-level checklist and detailed phase
exit: request/attempt/result persistence, transactional append and grouped
skip-line insertion, dependency validation, leases, idempotency and recovery;
crash/restart, duplicate enqueue/delivery, cycles, blocked-head eligibility and
result-handoff retry; actual ResourceManager reconciliation/outbox delivery and
operation-specific guarded transitions.

- The retained-parent RM HTTP acceptance test kills a scheduler subprocess after
  submission acceptance, before acknowledgment and provider completion. Reopening
  the same durable identity obtains a new RM session and attempt, settles the old
  submit, rejects old RM controls with `scheduler_superseded`, and publishes the
  replacement result. Same-session lost acknowledgment separately replays one
  immutable dispatch key/input without duplicate provider execution.
- A second subprocess creates the first verified publication receipt from the
  persisted handoff, then exits before queue acknowledgment. Actual scheduler/RM
  HTTP recovery acknowledges the retained handoff without provider execution or
  another receipt. Shared-database publication remains atomically covered by the
  scheduler suite; separate-database receipt replay is idempotent, not a claimed
  distributed transaction.
- Store guard coverage includes wrong attempt/session/generation, replaced-owner
  acknowledgments, metadata replay/conflict, cancelled delivery, terminal no-ops,
  retry budget and handoff-only completion. Missing dependencies in damaged state
  now produce durable `dependency_failed` through evaluation and both claim paths,
  without creating an attempt or submit. Public enqueue still rejects missing or
  cyclic dependencies; no forward-reference semantics were added.
- Ordering evidence includes blocked-head scans, dependency completion and
  failure, grouped concurrent skip-line insertion and retrying anchors.

Verification: **130 passed in 5.834s**, warnings treated as errors. The focused
eight-test acceptance module additionally passed three runs in 0.437s, 0.449s
and 0.447s. Targeted compilation and whitespace checks passed; final independent
review found no remaining local Phase 2 defects. Stable IDs and the exact focused
command are in the [proof inventory](../.opencode/skills/goal-oriented-design/references/e2e-proof.md#phase-2-durable-queue-core-completion-2026-09-17).
Fixtures own temporary SQLite/result/receipt/profile state, subprocesses and
loopback HTTP resources, including registered failure-path cleanup.

This is **unit/local integration evidence**, with the real RM HTTP/core but a
synthetic provider and profile. It does not prove RM-process crash persistence,
real GPU/deployed restart, measured
capacity, or Phase 6 production E2E. G1.1/G1.2/G2.1/G2.2/G3.1/G3.2 remain
`partial`; closing this implementation phase does not weaken those targets.

## Phase 3 complete — QueueScheduler behavior

Closed on 2026-09-17 against the deterministic scheduler transition and
insertion/eligibility exit. The new acceptance module exercises the actual
QueueScheduler, SQLite queue, local publisher and in-process ResourceManager
with synthetic providers; independent contract, store, evaluator, HTTP and
Phase 2 recovery suites supply operation guards and exhaustive adjacency/order
cases rather than duplicating them in every scheduler scenario.

- Six-status lifecycle and complete/incomplete GPU timing, duplicate/stale
  callback fences and publication-before-done retain truthful outcomes.
- Cancellation covers decoding, uncertain submission, buffered/on-GPU work and
  pending handoff; terminal replays cannot revive work. Idle, non-retryable
  failure and RM session invalidation abort active, blocked and retry-delayed work.
- Provider-failure retry uses persisted wall time with 5/10/20/30-second backoff
  and exhaustion; backpressure retains the original attempt without spending
  retry budget. Watchdog elapsed time remains monotonic and only a new finished
  provider response resets it, including a fenced late response.
- Bounded scans restart after awaited readiness invalidation. Dependency/template
  gates, missing-function recovery, grouped concurrent skip-line dispatch and
  independent anchor/retry/no-anchor guards establish predictable eligible order.
- Fixed scheduler polling to honor configured reevaluation cadence without
  throttling successive eligible claims. Cached capabilities are fenced by
  durable queue version and evaluator invalidation, and positive readiness
  expires on the poll deadline even without a durable mutation.
- Fixed stale finished-result cleanup to remove exact attempt ownership without
  requiring a separate cancellation event. Cancelled output cannot publish or
  leave an otherwise empty scheduler's watchdog armed indefinitely.

Verification: **150 passed in 9.765s**, warnings treated as errors. The 14-test
acceptance module additionally passed three runs in **1.491s, 1.473s and 1.485s**.
Targeted compilation and whitespace checks passed. Independent review found no
remaining defect for the local phase exit; its recovery-order documentation
correction is incorporated. Stable commands and test IDs are in the
[proof inventory](../.opencode/skills/goal-oriented-design/references/e2e-proof.md#phase-3-queuescheduler-completion-2026-09-17)
and [transition evidence map](../services/llm/queue/transition_table.md#stable-phase-3-evidence-map).
Fixtures own temporary databases/results, threads, tasks and HTTP resources;
some narrow callback/cache tests deliberately drive internal coordination seams.

This completes the implementation/local phase, not production-boundary proof.
No measured profile, GPU/deployed lifecycle, arbitrary RM event permutation or
new process-crash boundary is claimed. Existing Phase 2 subprocess/HTTP recovery
remains independent evidence. G1.1/G1.2/G2.1/G2.2/G3.1/G3.2 remain `partial`.

## Historical contract and queue slices

- **Slice A — Contract foundation:** OpenAPI draft, typed request/attempt,
  function, provisioning-bucket and profile contracts, status adjacency, and
  recorded decision ownership. Binding validation and production compatibility
  remain open.
- **Slice B — Durable queue primitives:** SQLite WAL transactions, enqueue
  idempotency, immutable intent references, ordered positions, attempts, leases,
  handoffs, cancellation/submission outbox, and event cursors.
- **Slice C — Ordering and local recovery:** dependency-gated claims, cycle
  validation, append and grouped skip-line insertion, lease/session recovery,
  stale-owner fencing, retry persistence, abort-all, abrupt-exit, and
  independent-connection evidence.
- **Slice D — Durable local results:** fsynced content-addressed files,
  idempotent publication receipts, pending handoff replay, and acknowledged-only
  `done` transitions.
- **Slice E — Durable optional-function intent:** canonical ready/template
  descriptors, dependency-reference validation, mutation-safe snapshots,
  descriptor-sensitive idempotency, and additive legacy upgrades. Descriptor
  claims fail closed without dispatch side effects; function execution and
  polling are not completed by this slice.
- **Slice F — Evaluated eligibility:** bounded FIFO scans, sync/async
  ready/template execution, dependency result resolution, and single-use
  durable version/session/intent-fenced claim capabilities.

These slices support the completed Phase 0/2/3 exits above. Later compatibility,
measured provisioning and production-boundary proof remain open.

## Transport and lifecycle boundaries

- **RM transport:** typed HTTP client, server-owned profile/provider lookup,
  trusted shared references, bounded JSON/SSE, request cancellation, typed
  non-failure backpressure, cursor replay/expiry, scheduler-over-HTTP exact
  publication, and all three selector shapes with fake providers. See the
  [RM HTTP boundary](resource-manager-http.md). This is local transport
  evidence, not production adapter integration.
- **Scheduler transport and control replay:** configured scheduler/function
  registries, six HTTP operations, durable keyed start/cancel/stop, bounded JSON,
  immutable request-specific SSE history, starvation-safe replay, terminal
  reconnect closure, and pre-header failure for unavailable legacy history.
  Publication retains cancellation/session fencing across receipt databases.
  See the [scheduler HTTP boundary](scheduler-http.md). Restart and GPU E2E
  remain open.
- **Linux GPU ownership proof:** lazy NVML, physical-GPU identity, baseline,
  bounded procfs ancestry and PID-reuse checks, positive owned-descendant
  evidence, and cleanup absence. Host PID namespace attestation and
  authoritative deployment identity are still required; synthetic proof is not
  real capacity or runtime readiness proof.

## Artifact, profile, and adapter slices

- **Artifact volume:** serialized selected-file copying, source-independent
  identities, digest reuse, atomic `current` selection, and fail-closed
  corruption, unsafe-path, and interrupted-copy handling. Prior complete volumes
  are retained. See the [artifact boundary](artifact-volume.md).
- **Read-only artifact verification:** configured volume IDs, exact selected
  file/hash verification, bounded JSON, identity summaries, off-loop hashing,
  bounded workers, and retained worker slots after timeout/disconnect. It does
  not copy, download, load models, establish readiness, or measure capacity.
- **Profile registry and validation:** immutable SQLite records, draft exclusion,
  identity/context lookup, evidence hashes, schema checks, supplied-wave/2%
  tie-rule validation, exact current-identity comparison, bounded read-only
  access, and retained workers. These validate supplied evidence; they do not
  measure capacity or integrate automatic runtime profiles. See the
  [profile boundary](capacity-profiles.md).
- **CoEdIT:** isolated offline Transformers worker, exact artifacts, input/output
  checks, deterministic native batch-one execution, bounded RPC, stale-session
  rejection, and process-loss cleanup. Candidate p=1 and switching evidence
  remain explicitly unmeasured and not production packaged. See the
  [CoEdIT GPU check](coedit-adapter-gpu-check.md).
- **SmolLM:** supervisor-owned loopback Ollama client, selected GGUF identity,
  bounded local import, raw ASCII framing, configured parallelism, readiness,
  and cleanup fencing. The configured printable-ASCII bound is not arbitrary
  UTF-8 or model-maximum coverage; packaging and measured capacity remain open.
- **GECToR:** installable offline worker-isolated adapter for the exact float32,
  native-batch-one, one-iteration, 128-subword bucket. Manifest, safetensors,
  vocabulary, preprocessing, output, worker identity, normal/process-loss,
  stale-session, and cleanup checks pass. Candidate image
  `sha256:fed8212118fb3f4309826479cd4fddeaa83701428d1a9f1198b4054e56537389`.
  This is not measured capacity or production packaging. See the
  [GECToR evidence](gector-adapter-gpu-check.md).
- **Installed three-model switching candidate:** SmolLM → CoEdIT → GECToR →
  SmolLM passed offline under one ResourceManager and parent-rooted GPU proof,
  with four responses, cleanup-gated replacements, stale-control rejection,
  cancellation fencing, and final cleanup. Candidate image
  `sha256:6785609cd5c7bfa792f9f839d481b4b0899ce8cdc96cd01ff33d40e7a8829fe2`.
  This is not measured capacity, kernel interruption, production health, or
  scheduler-to-server GPU E2E. See [switching evidence](three-model-adapter-gpu-check.md).

## Supervision and composed runtime

- **Read-only health precursor:** injected snapshots, bounded dependency probes,
  safe diagnostics, and fail-closed liveness/readiness routes. Production
  bootstrap wiring is absent. See the [health boundary](health-http.md).
- **Supervised runtime composition:** parent-rooted GPU capture, host-PID
  attestation, exact binding preflight, combined RM/health HTTP, live dependency
  gates, daemon-loss termination, permanent shutdown fencing, and bounded
  cleanup. Local synthetic-provider/profile race and cancellation evidence is
  not an approved image, measured profile, or deployed readiness proof.
- **OwnedOllama:** local pre-exec identity gate, pidfd-targeted signaling,
  loopback ownership, bounded version responses, cancellation-safe cleanup, and
  installed three-model candidate evidence. It is not a cgroup/container owner;
  children escaping before observation require external containment.

## Memory-observation precursor

Identity-fenced `LinuxGPUProof.memory()` supplies off-loop whole-device NVML
point observations, and the installed-adapter candidate recorded ordered
snapshots before children, during checks, and after cleanup. This remains only a
precursor: sparse post-inference points cannot prove peak incremental VRAM, a
20% reserve, memory-safe `N`, or throughput optimum. GECToR still uses
batch-one worker calls. CoEdIT now has real native-batch and maximum decoder
workload evidence, detailed below, but an exhaustive peak/resource bound is
still required before the complete procedure in section 4.4 of the
[canonical plan](queue-scheduler-resource-manager-plan.md). A successful p=1
check does not justify asserting `N=1`.

### CoEdIT decoder and incremental discovery continuation

The p16 throughput candidate passed after local admission stopped serializing
tokenizer RPCs. Authoritative per-row decoder witnesses then passed all 59
throughput waves at the exact input128/output64 bucket, with candidate optimum16
and owned cleanup. Generic provider output buckets above64 remain supported;
the64 cap belongs only to this candidate configuration.

Incremental discovery now runs every integer p=1..16 with four repeats and
strict native, decoder, allocator, timing, memory, identity and cleanup gates.
An actual run exposed zero-cursor replay expiry at p16 wave3; resumable per-wave
terminal cursors fixed the harness without changing RM retention or admission.
The rerun passed 64 waves/544 requests with maximum workload witnesses and
cleanup. Minimum sampled free memory was 8,192,479,232 of 12,878,610,432 bytes.
No exhaustive peak or memory-safe bound was established: `memory_safe_n` remains
null, profiles remain ineligible, and runtime defaults remain p1.

Latest focused verification: 112 tests in 4.285 seconds, compilation and
whitespace checks passed. Candidate image manifest
`sha256:118f447d79d88207a2fb3e2be28d146c1cd504abd4e064caad469d76cfbcfd6d`.
Commands, reduced artifact identity and cleanup ownership:
[CoEdIT capacity check](coedit-capacity-check.md). G4.1/G4.2/G4.3 remain partial.

Final session verification extended only opt-in discovery through p32 and passed
128 waves/2,112 requests in 212.659s, with every decoder row at64, exact native
correlation, zero drops and owned cleanup. Throughput remains capped at16;
default batch one and all evidence limits are unchanged. Early mode validation,
fail-closed full-schema artifact bounds and successful discovery exit status
have regression coverage. All162 task-related tests passed. No resource limit
was found and no profile was approved. This closes the bounded session slice,
not the remaining capacity-proof or production-readiness goals.

## Evidence summary and open follow-up map

## Phase 5 first-checkbox bounded evidence — 2026-09-18

The operator CLI boundary now has focused subprocess coverage for all three
configured parent-folder roots. Test-owned tiny offline fixtures prove concise
success output, content-addressed `current` selection, exact selected file sets,
deterministic rerun, and nonzero failure for missing GECToR
`verb-form-vocab.txt` while preserving the prior selection. This is unit/
operator-boundary evidence only; it does not claim G4.3 completion, measured
capacity, production image approval, semantic model validation, or production
E2E. The focused artifact-volume suite passed **19 tests** with warnings treated
as errors (acceptance run 0.324s; final confirmation 0.350s), and
`git diff --check` passed.

The consolidated local verification record includes: latest full discovery
**427 tests in 50.677s** with compilation and whitespace checks; **141 focused
runtime-composition tests in 32.034s**; **98 focused OwnedOllama/supervisor
tests**; CoEdIT **369 tests** plus a **36-test** focused check with candidate
image `sha256:e6187ee93a7c9c9a913f983813c6d172eb09ccd8c0d8729f1e146ffef9394582`;
GECToR **391 tests** plus **53 focused tests** with candidate image
`sha256:fed8212118fb3f4309826479cd4fddeaa83701428d1a9f1198b4054e56537389`;
installed switching **396 full / 91 focused tests** with candidate image
`sha256:6785609cd5c7bfa792f9f839d481b4b0899ce8cdc96cd01ff33d40e7a8829fe2`;
and an `OwnedOllama` switching candidate using image
`sha256:ec0af2cb045557602d42eb94c6645b3e330afa220d489bd34bc86bffb54ce108`.
These counts and digests are local evidence only; they do not establish
deployment, measured capacity, or qualifying E2E.

Historical focused suites and commands remain in the linked boundary documents,
including [three-model GPU evidence](three-model-adapter-gpu-check.md) and
[implementation decisions](implementation-decisions.md). They support the
bounded claims above only. Follow-up maps to the canonical plan's open Phases
1, 5 and 6: compatibility and approved production image, deployed RM/scheduler
integration and recovery, concurrent/native-batch measurement and profile
selection, offline bootstrap acceptance, production E2E, observability,
deployment, and runbooks. Contract and local exits for Phases 0/2/3 and the
Phase 4 local/candidate exit do not waive those remaining requirements.

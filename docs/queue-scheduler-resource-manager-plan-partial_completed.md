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
evidence. It is not a production deployment, measured capacity profile,
qualifying production-boundary E2E. Phase 2 alone has met its local phase exit.
The evidence does not replace the compatibility procedure, real concurrent
capacity sweep, production image acceptance, restart/failure E2E, or operations
work in the [canonical plan](queue-scheduler-resource-manager-plan.md).

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
real GPU/deployed restart, complete Phase 3 scheduler interactions, measured
capacity, or Phase 6 production E2E. G1.1/G1.2/G2.1/G2.2/G3.1/G3.2 remain
`partial`; closing this implementation phase does not weaken those targets.

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

These slices support the completed Phase 2 exit above. Remaining Phase 0/3
contracts and scheduler coverage, and production-boundary proof, remain open.

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
bounded claims above only. Follow-up maps to the canonical plan's open Phase 0
through Phase 6 exits: compatibility and production image, real RM/scheduler
integration and recovery, concurrent/native-batch measurement and profile
selection, offline bootstrap acceptance, production E2E, observability,
deployment, and runbooks. No listed evidence changes those open requirements.

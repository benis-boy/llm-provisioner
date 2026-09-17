# Product Proof Inventory

## Current status

G1.1 through G4.3 are **partial**: SQLite queue/result primitives, bounded
optional-function evaluation, an async scheduler, a transport-neutral RM and
candidate offline tooling exist. Real production provider/server boundaries,
measured profiles and production packaging remain incomplete.
No qualifying production-boundary E2E proof exists and no leaf is done.

## Local regression inventory (2026-09-17)

Latest verification: `.venv/bin/python -W error -m unittest discover -s tests -p 'test_*.py'` —
**294 passed** in 31.593 seconds. Adapter-harness verification passed **9 tests**.
Earlier GPU-proof/provider/RM/packaging/input-bound
verification passed **60 tests**. Focused profile/artifact HTTP/profile-store/OpenAPI
verification passed **50 tests**. Focused scheduler/HTTP/publication verification
passed **86 tests**, and the stop-before-publication regression passed five
repeated runs. Earlier focused compatibility verification passed **22 tests**
with warnings treated as errors. Artifact-volume/compatibility-input verification
passed **18 tests** before the later compatibility additions.
Earlier watchdog regressions and a harness live-watch hang were
repaired and reverified. `.venv/bin/python -m compileall -q services tests tools`
passed. All entries below are
**unit** evidence (including local integration tests); none substitutes for
required service/provider E2E. Tests own temporary directories, SQLite databases,
result files, receipts and subprocess cleanup.

| Goals | Stable suite/test IDs | Evidence and limits |
|---|---|---|
| G1.1 | `tests.integration.test_queue_recovery`; `tests.unit.test_queue_regressions.QueueRegressionTests.test_abrupt_subprocess_exit_preserves_request_and_submit_outbox`; `test_same_owner_recovery_preserves_handoff_and_fences_old_attempt` in the same class | WAL crash/reopen, retained handoff and local session replacement; no RM reconciliation |
| G1.2 | `tests.unit.test_results`; `tests.unit.test_store`; `tests.unit.test_queue_regressions.QueueRegressionTests.test_generic_handoff_outbox_ack_is_rejected` | Content-addressed result verification, durable publisher receipt replay, guarded handoff; no external publisher E2E |
| G2.1 | `tests.unit.test_contracts`; `tests.unit.test_queue_regressions.QueueRegressionTests.test_duplicate_finish_attempt_does_not_change_existing_telemetry` | All 36 adjacency pairs, status/timing primitives; not every guarded operation edge and no real GPU telemetry |
| G2.2 | `tests.unit.test_queue_regressions.QueueRegressionTests.test_cancel_is_idempotent_in_scheduled_running_and_on_gpu`; `test_non_retryable_failure_stops_the_entire_queue` in same class | Local cancellation/abort fencing; no production provider cancellation |
| G1.2, G2.2 | `tests.unit.test_queue_regressions.QueueRegressionTests.test_retry_delays_are_5_10_20_30_30_across_reopen_and_claim_at_300_is_fenced` | Exact retry delay/cap and 300-second exhaustion across restart |
| G3.1 | `tests.unit.test_store`; `tests.unit.test_function_intent.FunctionIntentTests.test_ready_template_and_both_are_fail_closed_without_attempts`; `test_descriptor_claim_is_fail_closed_without_side_effects` in the same class; `tests.unit.test_eligibility` | Dependency-gated claims, cycle rejection, bounded FIFO evaluation, optional sync/async functions and readiness polling; direct unevaluated descriptor claims remain fail-closed |
| G1.1, G1.2, G3.1 | `tests.unit.test_function_intent.FunctionIntentTests.test_ready_and_template_round_trip_across_reopen`; `test_legacy_schema_upgrade_preserves_rows_and_old_fingerprint_replay`; `test_canonical_descriptor_replay_is_noop_but_changes_conflict`; `test_descriptor_idempotency_addition_removal_and_template_changes_conflict`; `test_post_accept_mutation_and_invalid_descriptor_cannot_change_intent`; `test_descriptor_cancel_stop_and_stale_session_are_fenced` in the same class; `tests.unit.test_contracts.ContractTests.test_request_record_validates_descriptor_shapes_and_declared_dependencies` | Canonical immutable accepted descriptor intent, recovery, additive legacy upgrade preserving pending work, idempotency conflicts, dependency-reference boundary and cancellation/session fences; local SQLite evidence only |
| G3.2 | `tests.unit.test_queue_regressions.QueueRegressionTests.test_independent_connections_serialize_skip_line_insertions`; `test_closed_group_reopens_when_original_anchor_is_scheduled_by_retry` in same class | SQLite-linearized priority insertion and group reopening; no real dispatch-order E2E |

Additional **unit** suites (fake-provider/local integration, not production E2E):

- `tests.unit.test_scheduler`: durable submit replay after lost acknowledgment,
  crash handoff replay without calling public stop, explicit stop fences
  publication, result persistence retries, watchdog and cancellation. Scheduler
  and RM focused verification passed **45 tests** after review corrections.
- `tests.unit.test_resource_manager`: session/attempt fencing, bounded admission,
  cleanup, cancellation during validation and lifecycle timeout regressions;
  latest isolated run **24 passed**.
- `tests.unit.test_compatibility_inputs` and `tests.unit.test_spike`: selected
  artifacts, safe staging and UUID/cleanup helpers; latest **5 + 5 passed**.
- `tests.unit.test_rm_spike` and `tests.unit.test_model_runtime`: experimental
  RPC/fixture/fencing, private Ollama endpoint, exact execution-entry notification,
  dead-provider unload, saved process-group identity and adopted-grandchild reaping.
  Together with `tests.unit.test_spike`, **22 passed** with
  `.venv/bin/python -W error -m unittest -v tests.unit.test_rm_spike tests.unit.test_model_runtime tests.unit.test_spike`.
  Tests own temporary stores and child/grandchild processes.
- `tests.unit.test_artifact_volume` (**unit**, G4.3): source-independent selected
  artifact identities, atomic selection/reuse, strict manifest/sets, source and
  destination path safety, corruption and missing inputs, marked staging cleanup,
  concurrent reuse and injected copy/rename failures preserving prior selection.
  Tests own temporary source and output directories; no real model/GPU proof.
- `tests.integration.test_provisioning_http` (**unit**, G4.3/G2.1): read-only
  configured-volume verification, minimal identity summaries, expected-digest
  mismatch, strict content type/JSON/body limits, concurrent overload, timeout
  and disconnect slot retention, worker exception cleanup and actual OpenAPI
  response validation. The tiny SPECS-backed artifact files, volumes, worker
  barriers and loopback servers are test-owned. No semantic model validation,
  real GPU capacity, production mount/image or readiness proof is claimed.
  Focused command: `.venv/bin/python -m unittest -v tests.integration.test_provisioning_http tests.unit.test_artifact_volume tests.unit.test_openapi_validation`.
- `tests.unit.test_profiles`: **19 passed** for durable profile replay/conflicts,
  exact closest-context lookup, draft exclusion, throughput evidence consistency,
  content/identity corruption, transaction rollback/reuse and strict schema
  rejection. Temporary SQLite databases are test-owned. This validates supplied
  measurement evidence, not actual GPU measurement or runtime integration.
- `tests.unit.test_profile_contracts`: strict numeric and immutable provisioning
  data, model-specific profile shape and invalid-context rejection. Shared
  contracts still permit explicitly synthetic samples for experiments; only the
  measured registry boundary enforces the production evidence requirements.

- `tests.integration.test_resource_manager_http` (**unit**, local loopback):
  G1.1/G1.2 scheduler-over-HTTP plain-byte dispatch and exact durable publication;
  G2.1 replay, delayed progress, actual cursor expiry, per-frame/error bounds;
  G2.2 request cancellation and idempotent stop; G4.1/G4.2/G4.3 server-owned
  context and CoEdIT/GECToR bucket profiles, stale-before-reference-read fencing,
  strict verified content references, typed p+p backpressure and same-key retry.
  Watch flood/disconnect tests own their HTTP clients, servers and core tasks;
  fixtures own temporary stores and publication receipts. All provider execution
  and measurement evidence here is synthetic, not production or GPU proof.
- `tests.unit.test_openapi_validation` (**unit**): parsed OpenAPI validation,
  actual submission response schema resolution and preservation of future API
  request/response/idempotency contracts. Existing text checks also remain.
- `tests.unit.test_scheduler_operations` (**unit**, G1.1/G1.2/G2.2): durable
  start/cancel/stop journal replay and conflicts, stale generation/session
  rejection, lost start acknowledgment, concurrent same-key start and blocked
  start versus stop fencing. Tests own temporary databases and coordinator tasks.
- `tests.integration.test_scheduler_http` (**unit**, G1.1/G1.2/G2.1/G2.2): all six
  loopback operations, exact local publication, malformed input without durable
  insertion, immutable historical and live request replay across unrelated global
  cursor gaps, bounded batches, terminal reconnect closure, preheader legacy
  rejection, heartbeat/disconnect watcher cleanup and hostile client-wire parsing.
  Clients/servers, temporary stores and fake providers are test-owned. No real
  provider, deployed startup or production restart proof is claimed.
- `tests.unit.test_scheduler.SchedulerIntegrationTests.test_same_database_publication_commits_receipt_and_done_together`,
  `test_end_to_end_done_and_local_receipt`, and
  `test_explicit_stop_publishing_terminalizes_and_cannot_publish` in the same class
  (**unit**, G1.2/G2.2): shared-DB atomic receipt/done, separate-DB idempotent
  receipt and stop-before-publication with no late receipt. Temporary databases
  and result files are fixture-owned. This does not establish every process-crash
  interleaving between separate receipt and queue databases.

Scheduler-focused command: `.venv/bin/python -m unittest -v tests.integration.test_scheduler_http tests.unit.test_openapi_validation tests.unit.test_scheduler_operations tests.unit.test_scheduler tests.unit.test_queue_regressions tests.integration.test_resource_manager_http tests.unit.test_results`.

Stable focused command: `.venv/bin/python -m unittest -v tests.integration.test_resource_manager_http tests.unit.test_openapi_validation tests.unit.test_contracts`.
OpenAPI tests use the public `referencing` registry; full discovery now also
passes with warnings treated as errors.

Docker and NVIDIA GPU access were verified through the host daemon on 2026-09-16
with a disposable network-disabled `nvidia-smi` container: RTX 4070 Ti, 12,282 MiB,
driver 591.86, compute capability 8.9. See
[environment evidence](../../../../docs/implementation-decisions.md#environment-reassessment-2026-09-16).
The subsequent candidate image `llm-compatibility-spike:candidate` passed small
offline inference for SmolLM, CoEdIT and GECToR with selected hashes and local
verb vocabulary, returning `passed-candidate`. This is a **non-qualifying GPU
experiment**, not a full compatibility matrix or product proof. The tester
removed its owned container. The prepared context is retained intentionally.
See [candidate inputs and commands](../../../../docs/compatibility-spike.md).
The expanded RM-mediated network-disabled candidate run also passed for all three
models: small and configured upper fixtures, active cancellation result fencing,
stale-token rejection and restored GPU-process baseline before the next load.
Each emitted `ready`, `cancel_fenced`, and `cleanup_restored`; the tester removed
`llm-compatibility-rm-spike-candidate` and verified absence. This is still a
non-qualifying experiment with synthetic unmeasured p=1 profiles, not the complete
compatibility exit or production E2E. At that earlier stage SmolLM lacked independent
no-truncation proof; the later bounded ASCII proof is recorded below. Active
interruption, production runtime pins and measured capacity remain
unproved. Build-time apt required networking; runtime networking was disabled.

Latest rebuilt candidate RM runs passed on 2026-09-16: normal lifecycle in **41s**
and `--inject-process-failure` in **36s**. Each of SmolLM, CoEdIT and GECToR
produced an exact-attempt failure with no result, typed stale-session rejection,
and `group_gone=True nvml_baseline=True` after explicit RM stop. The tester
verified both disposable containers absent. The injected boundary is entry into
the execution adapter, not proof that a CUDA kernel started. Process-loss proof
does not establish provider interruption, full-container restart, production
health/readiness or a measured profile; all goal statuses remain partial.

Additional latest evidence:

- `tests.integration.test_profile_validation_http` (**unit**, G4.2/G4.3): exact
  submitted measured profiles for all three models, server-owned identity snapshots,
  draft/corrupt/mismatched evidence rejection, missing-database non-creation,
  strict and chunked body bounds, recursive JSON, huge integer handling,
  pre-worker slot release, timeout/overload and disconnected late-worker failure
  cleanup. Profile-store read-only and actual OpenAPI response-schema tests provide
  independent coverage. Tests own temporary registries, HTTP servers and worker
  barriers; synthetic measurements do not prove GPU capacity or readiness.
  Dedicated SQLite OperationalError injection remains a regression coverage gap.
  Command: `.venv/bin/python -W error -m unittest -v tests.integration.test_profile_validation_http tests.integration.test_provisioning_http tests.unit.test_profiles tests.unit.test_openapi_validation`.
- Compatibility focused suites now pass **36 tests**, including
  `tests.unit.test_candidate_input_bounds`, exact raw prompt/finished-response
  checks in `tests.unit.test_model_runtime`, and flat-image import packaging in
  `tests.unit.test_compatibility_inputs`. Rebuilt normal and process-loss GPU
  candidate scenarios passed again for all three models with networking disabled.
  SmolLM has independent conservative no-truncation evidence for the configured
  printable-ASCII bucket only; arbitrary UTF-8 and model maxima remain unproved.
  Both owned containers were absent after verification. These remain experiments,
  not production-boundary E2E or measured profiles.

## Required production-boundary evidence

### Lifecycle health and offline binding preflight (2026-09-17)

G0/G4.1/G4.2/G4.3 **unit** evidence:

- `tests.unit.test_resource_manager_health` and
  `tests.integration.test_resource_manager_health_http`: real RM with fake owned
  provider lifecycle gates, immutable observations, replacement revisions,
  cancelled/timed-out cleanup and late completion, pre/post-probe dependency
  conjunction, safe malformed/async snapshots and loopback readiness responses.
  A successful probe cannot turn an unproved profile into readiness or carry
  stale evidence across model replacement. Tests own tasks, gates and HTTP clients.
- `tests.unit.test_bootstrap_bindings`: bounded immutable operator configuration,
  actual tiny SPECS artifact-volume provisioning, actual SQLite measured-profile
  registry with explicitly synthetic measurement fixtures, all three real unloaded
  provider constructors, exact selectors/runtime/GPU/artifact identities, draft
  rejection, missing database non-creation and diagnostic N>1 with p=1/m=1.
  `test_resolve_rejects_later_profile_that_expands_pinned_capacity` specifically
  substitutes a later registry result to prove per-resolution capacity fencing.
  Runtime metadata fragments are bounded; GECToR version changes affect only its
  identity, without importing ML runtimes. Tests own temporary volumes/databases.
- `tests.unit.test_compatibility_inputs`: staged/refresh allowlists now include
  the RM lifecycle state module and still import offline without Torch.

Final `.venv/bin/python -W error -m unittest discover -v tests`:
**427 passed in 50.677s**. `.venv/bin/python -m compileall -q services tests tools`
and `git diff --check` passed. No new real GPU run was performed for these slices.

Remaining proof gaps: continuously supervised production process/daemon startup,
real dependency-state collection, actual measured capacity through that lifecycle,
approved runtime/image inputs and scheduler-to-GPU deployment E2E. Preflight is a
point-in-time artifact check under the documented immutable-volume precondition,
not a permanent lease on a writable filesystem. Cancellation during artifact
hashing and partial registry-failure handle cleanup lack dedicated fault-injection
tests. All goals remain partial. See
[health contract](../../../../docs/resource-manager-health.md) and
[bootstrap contract](../../../../docs/offline-bootstrap.md).

### Installed three-model switching candidate evidence (2026-09-17)

G4.1/G4.2/G4.3 **unit**: `tests.unit.test_three_model_adapter_check` uses a real
ResourceManager with fake provider/proof/daemon seams to cover replacement order,
cleanup failure blocking the next load, stale controls, cancellation result
fencing, actual provider configuration construction and flat-image imports.
Fixtures own temporary artifacts and lifecycle resources.

Actual **e2e candidate**:
[`docs/three-model-adapter-gpu-check.md`](../../../../docs/three-model-adapter-gpu-check.md).
One ResourceManager and parent-rooted GPU proof ran the installed providers
SmolLM → CoEdIT → GECToR → SmolLM offline. Three replacements gated the next load
on cleanup; four requests returned nonempty aligned results. Old submit/cancel
controls were rejected. An additional CoEdIT execution-entry cancellation allowed
the real delegate to finish but exposed no result. Final installed-provider,
private daemon-group and GPU cleanup were positively checked; the named container
was verified absent. This is not kernel interruption, a measured profile,
production bootstrap or full scheduler-to-GPU proof. Goals remain partial.

Image: `sha256:6785609cd5c7bfa792f9f839d481b4b0899ce8cdc96cd01ff33d40e7a8829fe2`.
Combined selected manifest:
`7abbd93bd3e4ec01ba01f8e4581821ae1d2f35cab720695c1309596df5614a19`.
Focused command: `.venv/bin/python -W error -m unittest -v tests.unit.test_three_model_adapter_check tests.unit.test_adapter_check tests.unit.test_coedit_adapter_check tests.unit.test_gector_adapter_check tests.unit.test_resource_manager tests.unit.test_gpu_proof`
— **91 passed in 5.597s**. Full `.venv/bin/python -W error -m unittest discover -v tests`
— **396 passed in 48.669s**; compilation and whitespace checks passed.

### GECToR isolated adapter candidate evidence (2026-09-17)

G4.1/G4.2/G4.3 **unit** suites: `tests.unit.test_gector_provider`,
`tests.unit.test_gector_worker`, `tests.unit.test_gector_adapter_check` and
`tests.unit.test_python_worker`. Test-owned selected artifact volumes, mocked
package loaders and framed subprocesses cover strict native parameters,
package-style `$START`/split-word tokenization, 127/128/129-token boundaries,
nonfatal expected overlength rejection, local loading/patch restoration and
shared dispatch. These are not GPU or measured-capacity evidence.

Actual **e2e candidate** commands and named-container cleanup ownership:
[`docs/gector-adapter-gpu-check.md`](../../../../docs/gector-adapter-gpu-check.md).
Normal and injected loss passed offline on the selected physical GPU. The
normal run returned two aligned responses, rejected one overlong request and
remained usable, proved one owned runner, rejected the stale token and cleaned
up. Injected execution-entry process loss produced a failure with no result and
proved cleanup. Both containers were verified absent. This is not kernel-entry
or interruption proof, combined switching, production packaging or capacity
measurement. All touched leaves remain partial.

Image: `sha256:fed8212118fb3f4309826479cd4fddeaa83701428d1a9f1198b4054e56537389`.
Selected manifest: `c3468ef6bbd5047d045b38800e33a3fd83f61a06277c9dd6bc8812e422610730`.
Model: `f399c999ba19811601d685016fad4589fd397859b4ac1eca0c42fd8aeb0c9fe4`.
Runtime: GECToR 1.2.0, Torch 2.7.1+cu128, Transformers 4.49.0, tokenizers 0.21.0,
safetensors 0.5.3, CUDA 12.8. Full discovery passed **391 in 40.037s**;
focused **53 in 2.808s**; compilation and `git diff --check` passed.

### CoEdIT isolated adapter candidate evidence (2026-09-17)

G4.1/G4.2/G4.3 **unit** suites: `tests.unit.test_coedit_provider`,
`tests.unit.test_python_process`, `tests.unit.test_python_worker` and
`tests.unit.test_coedit_adapter_check`. Tests own selected-artifact temporary
volumes, fake GPU/runtime evidence and real framed subprocesses. They cover
exact native batch-one buckets, token framing/output bounds, offline loader
options, startup cancellation, orphan-group cleanup, PID reuse refusal,
stderr/transport bounds and failed-readiness/cleanup fencing. Harness tests
exercise real ResourceManager with fake providers, not real GPU evidence.

Actual **e2e candidate** commands and cleanup ownership:
[`docs/coedit-adapter-gpu-check.md`](../../../../docs/coedit-adapter-gpu-check.md).
Both normal and injected process-loss runs passed with networking disabled,
host-PID attestation and exact UUID selection. The normal run proved two aligned
nonempty responses, one positively identified worker, stale-session rejection
and owned GPU/process cleanup. Injected loss produced no result and cleanup
passed. Both named containers were removed. Injection is execution-entry
evidence, not proof of kernel entry or interruption. The profile is explicitly
unmeasured p=1; no throughput capacity or production image claim is made.

Image: `sha256:e6187ee93a7c9c9a913f983813c6d172eb09ccd8c0d8729f1e146ffef9394582`.
Runtime: Torch 2.7.1+cu128, Transformers 4.49.0, tokenizers 0.21.0,
safetensors 0.5.3, CUDA 12.8. Full discovery with warnings as errors passed
**369 tests in 39.766s**; focused verification passed **36 in 2.791s**;
compilation passed. All touched leaves remain partial.

Real-adapter candidate check (2026-09-17, **blocked**, not qualifying E2E):
`tools/compatibility/adapter_check.py` runs the actual SmolLM provider and Linux
GPU proof with an explicitly unmeasured profile. Its separate hash-locked aiohttp
layer built with Docker networking disabled. Runtime with `--network none`, one
UUID-selected GPU and a read-only Docker-host `/proc` bind failed before readiness:
the launcher reported `procfs_missing` while opening `/host/proc/self/stat`.
No inference, supervisor residency or GPU cleanup-baseline proof was reached.
The tester removed `llm-compatibility-adapter-check` and verified its absence.
Further GPU testing stopped pending operator repair of authoritative host procfs
availability and NVML PID-namespace alignment. See
[command and limits](../../../../docs/adapter-gpu-check.md).

One bounded retry on 2026-09-17 after the devcontainer procfs mount was added
failed non-zero in **5.685 seconds** with the same `Ollama launcher procfs_missing`
before readiness. It used existing adapter image
`sha256:30174aecfa41c99ff94228826c93f631618e2370c892426847d06f7ecec2c485`,
the documented explicit Docker-host bind, exact GPU UUID, `--init`,
`--network none` and host-PID attestation. The tester removed its owned
`llm-compatibility-adapter-check-retry` container and verified absence. No further
tests ran after the environment block. G4.1/G4.3 proof gaps remain unchanged:
the devcontainer mount does not establish readable authoritative procfs in the
separate adapter runtime or NVML PID alignment.

The harness now uses authoritative `/proc` with required `--pid=host`. The GPU
proof was redesigned for a shared physical GPU: only positively proved strict
supervisor descendants are service-owned, while readable foreign process churn
is tolerated and never controlled. The focused GPU-proof, adapter, SmolLM and RM
suites pass **74 tests** with warnings treated as errors; compilation passes.
Network-disabled image
`sha256:8bc37a2e60cdbb14c4ed067eeb602b58540af7214bc300dfc257d62ff5293c53`
built successfully. Its actual run failed closed before readiness with `GPU
baseline ownership could not be observed`, meaning a current NVML PID could not
be safely classified from procfs. The owned container was removed and verified
absent. This is non-qualifying failure evidence: inference, service residency,
cleanup and shared-contention capacity remain unproved.

The ownership proof now adds bounded 10–50 ms retry backoff within a 250 ms
post-ambiguity grace, fully classifies confirmation snapshots, and revalidates
readable foreign ancestry before exclusion. Focused GPU-proof, adapter, SmolLM
and RM verification passes **83 tests** with warnings as errors; compilation
passes. Network-disabled image
`sha256:c1ddec1ae3ab6cb43cf5081423c111da493281aebb9be65c5a101651aacd9a10`
built successfully, but its actual run again failed closed at baseline
ownership. A bounded diagnostic observed empty compute and graphics before
Ollama; with Ollama, graphics reported one procfs-readable PID and one PID absent
through ten 200 ms samples. The latter cannot be authoritatively classified from
Linux procfs. Persistence is not accepted as foreign provenance, and all owned
containers were removed and verified absent. This is non-qualifying failure
evidence. The exhaustive-correlation requirement was subsequently removed as
stricter than G4.1: unknown non-supervisor NVML PIDs are ignored and never
controlled, while service ownership requires stable positive supervisor descent.
The focused GPU-proof, adapter, SmolLM and RM suites pass **88 tests** with
warnings as errors and compilation passes. No new production-boundary run has
yet proved G4.1 inference/residency/cleanup; G4.2 contention capacity remains
unproved.

Latest goal-focused SmolLM adapter **e2e candidate evidence**: the exact
network-disabled host-PID run on
`GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963` passed with Ollama 0.11.6. Two bounded
requests produced nonempty completions; readiness positively proved one strict
supervisor-descendant GPU runner; unload reached empty private Ollama model state;
the stale session was rejected; and the identity-fenced daemon group and named
container were verified gone. Two baseline NVML PIDs were recorded only as a
bounded count and were neither exhaustively identified nor controlled. Focused
GPU-proof/adapter/SmolLM/RM verification passes **94 tests** with warnings as
errors. This advances the SmolLM portion of G4.1 only. It does not prove CoEdIT,
GECToR, full production deployment, or G4.2 contention capacity.

`tests.unit.test_adapter_check` (**unit**) covers the actual in-process
RM lifecycle with mocked provider/GPU/daemon seams, selected artifact identities,
unmeasured profile labeling, local orphaned process-group cleanup, launcher
failure categorization and cancellation cleanup. Tests own temporary artifacts
and processes. These do not prove real Ollama execution or host/NVML alignment.

Latest adapter/proof **unit** evidence (G4.1/G4.2/G4.3):

- `tests.unit.test_smollm_provider`: test-owned loopback Ollama and tiny real GGUF
  metadata validate framing, strict response bounds, local import failure paths,
  readiness/cleanup and admission. Mandatory supervisor evidence rejects missing
  or mismatched identity, empty/duplicate runner PIDs and supervisor aliases.
  Failed readiness disables execution; never-loaded cleanup checks actual API
  absence before RM recovery. These are synthetic residency/profile fixtures,
  not real Ollama compatibility or measured GPU admission.
- `tests.unit.test_gpu_proof`: synthetic procfs and injected NVML prove ancestry,
  PID reuse/race rejection (including ancestry mutation), unknown non-supervisor
  PID tolerance, shared-GPU foreign churn,
  positive service ownership,
  MIG/API failures and explicit
  real-capture host-namespace attestation. Actual deployment must make that
  attestation truthful; these tests do not establish a host/container PID mapping.
- `tests.unit.test_resource_manager`: failed initial readiness cleanup permits
  a fresh start only after verification; timed-out unfinished load blocks cleanup
  and leaves no exposed session. Tests release and await their lifecycle gates.
- `tests.unit.test_smollm_provider.SmolLMProviderTests.test_real_cli_timeout_and_output_overflow_are_bounded`:
  real test-owned CLI processes exercise timeout and output-overflow cleanup.
  A temporary unraisable hook, forced GC and event-loop turns assert no leaked
  subprocess transports. Three repeated runs passed without warnings after a
  reproducible stderr transport leak was repaired.

Focused command: `.venv/bin/python -W error -m unittest -v tests.unit.test_gpu_proof tests.unit.test_smollm_provider tests.unit.test_resource_manager tests.unit.test_compatibility_inputs tests.unit.test_candidate_input_bounds`.
Tests own temporary artifacts, procfs fixtures, subprocesses, HTTP servers and
event-loop resources. Candidate flat-image imports are independently covered by
the prepare/refresh regression. No production-boundary E2E status changes.

`tests.unit.test_health_http` and the health assertions in
`tests.unit.test_openapi_validation` (**unit**, G4.3) cover exact read-only
liveness, readiness and dependency routes; atomic injected snapshots; safe
structured diagnostics; bounded sync/async probes; timeout, exception and
overload behavior; retained worker slots; bounded cleanup; and representative
OpenAPI response validation. **15 tests passed in 0.144 seconds** with warnings
treated as errors, and service/test/tool compilation passed. The health boundary
is not wired to production bootstrap state and does not prove actual GPU,
artifact, adapter or measured-profile readiness.

- **G1.1 — Durable asynchronous work (`e2e`):** Accepted work in the SQLite
  QueueStore survives real scheduler or consumer restart and reaches an
  idempotent result handoff through transactional outbox recovery.
- **G1.2 — Correct terminal outcomes (`e2e`):** Result-handoff failure, retry,
  cancellation, dependency failure, duplicate operations, and stale execution
  produce the documented outcome without duplicate or stale publication.
- **G2.1 — Truthful request progress (`e2e`):** A real request exposes all six
  statuses and truthful timing, including
  ResourceManager-supplied complete or explicitly incomplete `time_on_gpu`.
- **G2.2 — Actionable cancellation and stalls (`e2e`):** Cancellation works in
  every status, `stop()` fences late work, and idle or non-retryable failure
  aborts the complete QueueScheduler. The one-minute idle watchdog resets only
  when a complete provider response is observed; tokens, heartbeats, model
  lifecycle activity, dispatch, buffering, and result handoff do not reset it.
- **G3.1 — Predictable eligible ordering (`e2e`):** A mixed dependency,
  readiness, and template queue proves eligible FIFO while blocked work remains
  durably queued.
- **G3.2 — Deterministic priority insertion (`e2e`):** Append and skip-line
  insertion remain deterministic and idempotent across eligibility changes and
  concurrent insertion.
- **G4.1 — Safe exclusive model service (`e2e`):** The target GPU executes
  SmolLM, CoEdIT, and GECToR through their real adapters with exclusive fenced
  model switching, cancellation, and cleanup.
- **G4.2 — Bounded useful admission (`e2e`):** Exact-profile `optimal_parallelism` execution plus
  `m` buffering is measured on the target GPU; runtime uses the highest measured
  successful requests/second concurrency, and an outdated competing scheduler
  is synchronously rejected and aborts.
- **G4.3 — Offline reproducible readiness (`e2e`):** Provisioning and service
  bootstrap succeed with networking disabled using configured parent folders,
  real ResourceManager adapters, generated or configured maximum-sized
  requests, four serial baseline samples, one warmup plus four measured waves of
  `n` requests at each required concurrency point, and durable SQLite profiles
  keyed by model, GPU, and optional context size. Artifact or runtime identity
  mismatch fails closed.

The G4.3 production-boundary proof must use the production image—not the
devcontainer—and one GPU supplied through NVIDIA Container Toolkit. It must also
prove pinned/offline dependency installation, non-root processes, private
loopback-only Ollama, Docker-managed `llm-provider-state` and
`llm-provider-ollama` volumes, bounded signal shutdown, truthful
health/readiness, missing-artifact failure, and no undeclared model downloads.
Before implementation, a non-qualifying compatibility spike must establish one
pinned GPU/driver/CUDA/PyTorch/Transformers/gector/Ollama matrix that loads,
executes, cancels best effort, unloads, and switches all three models offline,
including the supplied GECToR `verb-form-vocab.txt`.

Operational contract coverage must prove the one-minute idle stop, 100% input
buffer, 20% VRAM reserve, five-second-to-thirty-second retry backoff with a
five-minute request retry budget, one-minute shutdown grace, five-second health
timeouts, 5 GiB minimum free-space admission threshold, and one-day trace
retention. ResourceManager-buffer-full backpressure must not consume an attempt,
retry budget, or error transition.

Every leaf also requires multiple independent tests, including focused unit or
contract coverage. Unit evidence alone cannot advance a leaf beyond
**untested**.

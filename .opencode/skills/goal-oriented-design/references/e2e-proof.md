# Product Proof Inventory

## Current status

Fresh exact candidate image
`sha256:da9b047a4394cdd91d565ee00c288e244ec5c12655229c9c8465830e9c9b0d1f`
completed SmolLM → CoEdIT → GECToR → SmolLM with three switches,
`cancel_fenced=true`, `stale_rejected=true`, six stable-total memory points,
final used-memory restoration to baseline, and verified owned-container
cleanup. The profile remains unmeasured and this is candidate evidence, not
qualifying production deployment or capacity proof.

G1.1 through G4.3 are **partial**: SQLite queue/result primitives, bounded
optional-function evaluation, an async scheduler, a transport-neutral RM and
candidate offline tooling exist. Real production provider/server boundaries,
measured profiles and production packaging remain incomplete.
No qualifying production-boundary E2E proof exists and no leaf is done.

## Phase 0 contract closure (2026-09-17)

**Unit/local integration**, supporting G1.1/G1.2/G2.1/G2.2/G3.1/G3.2/G4.1/G4.2/
G4.3 under G0: durable accepted work, acknowledged fenced results, truthful
progress/intervention, predictable eligible ordering, safe owned service and
bounded offline exact-identity admission. Phase 0's contract exit is complete;
all goal leaves remain `partial`, with qualifying production-boundary proof open.

`tests.integration.test_phase0_contract_acceptance.Phase0ContractAcceptanceTests`:

- `test_production_rm_serializers_validate_nullable_progress_and_capacity` and
  `test_actual_scheduler_projection_and_rm_error_validate`: actual production
  serializer/projection output matches declared schemas, including null timing.
- `test_complete_operation_inventory_future_set_and_post_contracts` and
  `test_error_and_sse_response_media_types_are_declared`: complete declared
  inventory, idempotency/body/cursor/media contracts and exact future markers.
- `test_actual_scheduler_sse_frame_validates_with_durable_cursor`: real loopback
  SSE data uses the durable request schema and numeric event IDs.
- `test_actual_rm_backpressure_response_validates_as_submission`: full p+p
  admission returns typed HTTP 429, not a malformed-reference error.
- `test_actual_provisioning_success_and_failure_responses_validate` and
  `test_actual_profile_validation_success_and_failure_responses_validate`:
  actual integrity and exact-profile endpoints return schema-conformant
  success and rejection envelopes.
- `test_actual_health_success_failure_and_dependency_responses_validate`:
  loopback liveness/readiness/dependencies, including startup HTTP 503.
- `test_schema_validator_rejects_invalid_payload`: negative validation control.

Independent contract, profile, artifact-volume, bootstrap-binding and HTTP
suites cover behavioral guards; the new module is not exhaustive status/error
or production E2E proof. Fixtures own temporary SQLite/results/artifacts/profiles,
fake providers, tasks and loopback server cleanup. No GPU or deployment run is
required for this contract-only change, and none is claimed.

Focused command (no full discovery):

```sh
.venv/bin/python -W error -m unittest -v tests.integration.test_phase0_contract_acceptance tests.unit.test_openapi_validation tests.unit.test_contracts tests.unit.test_profile_contracts tests.unit.test_profiles tests.unit.test_artifact_volume tests.unit.test_bootstrap_bindings tests.integration.test_profile_validation_http tests.integration.test_provisioning_http tests.integration.test_resource_manager_http tests.integration.test_scheduler_http tests.unit.test_health_http
```

**145 passed in 5.562s**. Command
`.venv/bin/python -W error -m unittest -v tests.integration.test_phase0_contract_acceptance`
also passed **10 tests** twice (**0.310s, 0.311s**). Targeted compilation and
`git diff --check` passed. Phase 1 compatibility/pin approval, Phase 5 measured
provisioning, and Phase 6 backup/deployment/E2E remain gates; no synthetic
profile, future endpoint or candidate image is promoted by this closure.

## Historical Phase 4 local continuation and blocked GPU verification (2026-09-17)

The successful exact candidate in Current status supersedes the blocked GPU
runs recorded in this section; they are not current blockers.

**Unit/local integration**, G2.1/G2.2/G4.1/G4.2 under G0: truthful completion,
safe cancellation, fenced owned service and bounded admission. Phase 4's local
exit is complete; all touched product goals remain `partial`.

- `tests.integration.test_phase4_acceptance.Phase4ResourceManagerAcceptance.test_real_adapter_lifecycles_switch_only_after_cleanup`:
  real adapter orchestration with selected artifacts, loopback Ollama and
  controlled worker/GPU seams; successful responses retain request/attempt
  identity, completion sequence and explicit null/incomplete GPU timing.
- `test_p2_plus_p2_retry_and_stale_replacement_controls` in that class: two active
  plus two buffered, exact accepted replay, rejected work does not execute,
  same-key retry after capacity release, prompt stale start/submit/cancel/capacity
  rejection during gated replacement.
- `test_failed_cleanup_fences_old_controls_without_replacement_authority` composes
  failed cleanup with rejected retired submit/cancel/capacity controls and no new
  authority. `test_actual_adapter_late_completion_after_cancel_is_not_published`
  collects a real local CoEdIT adapter completion after cancellation while exposing
  only the cancelled terminal event and no result.
- `tests.unit.test_resource_manager.ResourceManagerTests`:
  `test_replay_does_not_leave_validation_reservation`,
  `test_concurrent_same_attempt_validation_keeps_cancellation_fenced`,
  `test_conflicting_duplicate_validator_cannot_replace_marker`,
  `test_failed_duplicate_validator_preserves_survivor_reservation_for_cancel`,
  `test_cancelled_wait_for_final_admission_releases_reservation`, and
  `test_cancel_validation_and_active_work_fences_late_result_and_calls_provider`
  cover replay/immutable identity, overlapping validators, exactly-once release
  after asymmetric validation failure, final-lock task cancellation and concurrent
  validating/active ownership with advisory cancellation and suppressed results.
- Existing provider, RM shutdown/HTTP, three-model harness and Phase 3 suites
  independently cover lifecycle errors, cleanup, progress and publication fences.
  Fixtures own temporary artifact roots, loopback servers, tasks and gates;
  synthetic GPU/worker evidence is not production-boundary proof.

The local transition-1 CoEdIT fallback contract is also covered by focused
regressions: typed expected-runner absence only, exact worker identity and
retained CUDA witness, fenced supervisor/device identity, valid baseline, and
two ordered post-load observations with stable immutable identity and positive
used-memory effect. These tests are unverified pending independent tester
execution; they are not GPU candidate evidence and do not advance a goal.

Focused command (no full discovery):

```sh
.venv/bin/python -W error -m unittest -v tests.integration.test_phase4_acceptance tests.unit.test_resource_manager tests.unit.test_resource_manager_shutdown tests.unit.test_smollm_provider tests.unit.test_coedit_provider tests.unit.test_gector_provider tests.unit.test_three_model_adapter_check tests.unit.test_gpu_proof tests.integration.test_phase3_acceptance tests.integration.test_resource_manager_http
```

**176 passed in 9.248s**. Additional command
`.venv/bin/python -W error -m unittest -v tests.integration.test_phase4_acceptance`
passed **4 tests** three times (0.234s, 0.237s, 0.236s). Targeted compilation and
`git diff --check` passed. No final warnings reported.

**E2E candidate — failed/blocked**, not qualifying proof: current source built
offline into `llm-compatibility-adapter:phase4`, RepoDigest
`llm-compatibility-adapter@sha256:2a02467c1f1e60e6445109fd65ff1ec813817c5b193103dd51067201c2b76e1c`.
The documented three-model command (tag `:phase4`, owned container
`llm-phase4-three-model-check`) on GPU
`GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963` now identifies transition 1,
SmolLM → CoEdIT. Bounded pre-load settlement clears the former owned-runner
cleanup lag, but the final run fails during CoEdIT `start_session` with `GPU has
no resident runner` after bounded post-load settlement. No passing result/manifest
projection was emitted. The tester verified its container absent. The temporary
CUDA allocation used by that build was not retained after `cuda_ready` returned;
the source now retains a bounded worker-lifetime allocation and requires rebuild.
The retained-witness rebuild instead observed runners but no generic owned
identity (image `sha256:8fc0ab1008fc84878e4f28b426b1a3ea1639745db671d8b26a7601bc11605222`);
the source now sends the exact worker identity through an expected-runner,
category-only proof diagnostic. The precise failed relation remains unestablished
pending the next candidate run. See the
[bounded ledger](../../../../docs/queue-scheduler-resource-manager-plan-partial_completed.md#phase-4-continuation--local-fences-verified-gpu-check-blocked).

**Latest superseding initial-load evidence:** this platform's NVML process API
reported only two unchanged procfs-unmappable baseline graphics IDs while the
private Ollama endpoint reported the exact SmolLM model fully in VRAM and procfs
proved the fenced Python supervisor → owned daemon → owned runner topology. A
narrow SmolLM-only fallback now requires that topology and exact private model
state to remain stable across two observations, plus matching fenced pre/post GPU
memory identity and a positive used-memory increase. It grants no authority over
the baseline NVML IDs. Focused verification passed **183 tests**; compilation and
whitespace checks passed. Fresh offline image manifest-list digest:
`sha256:7c8015b8e989414e6af74254211c4247377b48d6674a8f7e529e719d6f660d8e`
(image manifest `sha256:661783d3baf6b3c0202dc98e98549cf957217edbf5e0d3acf68f7fcc9700e0f9`).
The exact candidate completed transition 0 SmolLM start/readiness, a valid response,
model-specific residency acceptance, and memory observation before failing at
transition 1 CoEdIT exact-worker proof (`exact_child=0`,
`strict_supervisor_descendant=0`, `foreign_or_baseline=1`,
`unreadable_or_unconnectable=1`, `identity_mismatch=0`). Its owned container was
verified absent. This establishes initial SmolLM loading for this candidate only;
the full switch and production-boundary proof remain incomplete, so G0/G4.1/G4.3
remain **partial**.

## Phase 2 durable queue core completion (2026-09-17)

**Unit/local integration**, supporting G1.1/G1.2/G2.1/G2.2/G3.1/G3.2 under G0:
accepted queued work survives scheduler crashes, publication stays acknowledged
and fenced, cancellation cannot revive work, and eligible ordering remains
deterministic. Phase 2's implementation/local exit is complete; all goal leaves
remain **partial** with production-boundary proof outstanding.

- `tests.integration.test_phase2_acceptance.Phase2HTTPRecoveryTests.test_process_death_after_http_acceptance_reconciles_new_attempt`:
  scheduler child killed after real RM HTTP accepts submission but before ack or
  provider completion, retained server, new session/attempt after reopen, exact
  `scheduler_superseded` stale-control failures and settled old submit.
- `tests.integration.test_phase2_acceptance.Phase2HTTPRecoveryTests.test_receipt_before_queue_ack_replays_without_provider_execution`:
  first receipt committed in an abruptly exiting child from durable handoff
  identity; actual scheduler/RM HTTP recovery reaches done without executing the
  provider again or creating another receipt.
- `tests.integration.test_phase2_acceptance.Phase2OperationGuardMatrixTests`:
  stale attempt/session/generation and owner fences, dispatch metadata replay and
  conflicts, cancelled-submit acknowledgment, pending handoff recovery, and
  terminal/idempotency guards. `tests.unit.test_contracts`, `test_store`,
  `test_queue_regressions`, `test_scheduler_operations`, `test_results`,
  `test_eligibility`, and `test_function_intent` independently cover adjacency,
  leases/retry, publication verification, concurrent skip-line and blocked FIFO.
- `tests.unit.test_scheduler.SchedulerIntegrationTests.test_lost_submit_ack_replays_same_attempt_key`,
  `test_reopen_pending_handoff_does_not_execute_provider`,
  `test_result_and_handoff_transient_failures_publish_once`, and
  `test_same_database_publication_commits_receipt_and_done_together` independently
  cover same-session replay and result publication using the real in-process RM.
- `tests.unit.test_store.StoreTests` covers defensive multi-node cycle detection
  and missing dependency errors through evaluator/direct/evaluated claims without
  creating attempts or submits.

Fixtures own temporary SQLite/results/receipt/profile state, child processes and
loopback HTTP resources, including failure cleanup. Provider and measurement
fixtures are synthetic: this does **not** prove RM-process persistence, GPU
execution, deployment restart or the complete Phase 3 scheduler interaction matrix.

Focused command (no full discovery):

```sh
.venv/bin/python -W error -m unittest -v tests.integration.test_phase2_acceptance tests.unit.test_contracts tests.unit.test_store tests.unit.test_queue_regressions tests.unit.test_results tests.unit.test_scheduler_operations tests.unit.test_scheduler tests.unit.test_eligibility tests.unit.test_function_intent tests.integration.test_queue_recovery tests.integration.test_resource_manager_http
```

**130 passed in 5.834s**. Additional command
`.venv/bin/python -W error -m unittest -v tests.integration.test_phase2_acceptance`
passed all **8 tests** three times (0.437s, 0.449s, 0.447s). Targeted compilation
and whitespace checks passed. Independent final review found no remaining local
Phase 2 defect. These results supersede the historical queue-core claims of no
actual RM reconciliation; they do not supersede the production-proof gaps.

## Phase 3 QueueScheduler completion (2026-09-17)

**Unit/local integration**, G1.1/G1.2/G2.1/G2.2/G3.1/G3.2 under G0: accepted
work stays recoverable, terminal results are acknowledged and fenced, progress
and intervention remain truthful, and eligible append/skip-line work dispatches
predictably. Phase 3's implementation/local exit is complete; all leaves remain
**partial**, without qualifying production-boundary E2E.

- `tests.integration.test_phase3_acceptance.Phase3SchedulerAcceptance` contains
  14 scenarios: complete/incomplete timing, callback fences, cancellation across
  dispatch/publication stages, idle/non-retryable/session-invalidated abort,
  retry/backpressure, resumed readiness scans, missing-function recovery and
  concurrent grouped skip-line execution order. The
  [stable evidence map](../../../../services/llm/queue/transition_table.md#stable-phase-3-evidence-map)
  maps exact test titles to the independent adjacency/store/recovery tests.
- `tests.unit.test_scheduler.SchedulerIntegrationTests.test_optional_function_poll_cadence_and_local_enqueue_wake`,
  `test_large_function_poll_interval_does_not_delay_armed_watchdog`,
  `test_large_poll_dispatches_two_ready_requests_without_per_request_delay`,
  `test_independent_mutation_discards_cached_capability_for_new_head`, and
  `test_positive_cache_expires_and_rescans_external_readiness` establish bounded
  polling, version/invalidation fencing and external readiness expiry without
  throttling successive eligible requests.
- `test_late_finished_result_without_cancel_event_cleans_watchdog_ownership` in
  that same scheduler class verifies exact stale-attempt cleanup and the required
  finished-response reset even when cancelled output cannot publish.
- Phase 2 acceptance and independent contract/store/evaluator/regression suites
  retain exhaustive adjacency, operation guards, anchor retry/cancellation and
  no-anchor fallback, recovery, and durable publication evidence. Scheduler/RM
  HTTP suites independently exercise lifecycle and replay through loopback.

Fixtures own temporary SQLite/results/receipts, worker threads, coordinator
tasks and loopback services. Providers/profiles are synthetic; internal cache
and callback tests are narrow unit evidence, not production transport E2E.

Focused command (no full discovery):

```sh
.venv/bin/python -W error -m unittest -v tests.integration.test_phase3_acceptance tests.integration.test_phase2_acceptance tests.unit.test_scheduler tests.unit.test_scheduler_operations tests.unit.test_eligibility tests.unit.test_queue_regressions tests.unit.test_store tests.unit.test_contracts tests.integration.test_scheduler_http tests.integration.test_resource_manager_http
```

**150 passed in 9.765s**. Command
`.venv/bin/python -W error -m unittest -v tests.integration.test_phase3_acceptance`
also passed **14 tests** three times (1.491s, 1.473s, 1.485s). Targeted compilation
and whitespace checks passed. Production GPU/deployment, measured capacity and
additional process-crash boundaries remain outside this local completion.

## Historical local regression inventory (2026-09-17)

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

## Private Ollama supervision precursor (G0, G4.1, G4.3)

`tests.unit.test_bootstrap_supervisor` (**unit**) uses test-owned local Python
daemon processes, ephemeral loopback ports, temporary homes and synthetic typed
GPU proof. It covers the pre-exec identity gate, occupied foreign listener
refusal, strict bounded/fragmented version responses, total readiness deadline,
pending-spawn ownership, concurrent/repeated cancellation, nested descendant
cleanup, and previously pinned descendants changing session. PIDfds, not bare
process groups, authorize destructive signals; identity mismatch tests assert
foreign processes are not pinned or signalled. Tests release their own gates and
collect child processes and transports.

Focused command:
`.venv/bin/python -W error -m unittest -v tests.unit.test_bootstrap_supervisor tests.unit.test_bootstrap_bindings tests.unit.test_python_process tests.unit.test_gpu_proof tests.unit.test_smollm_provider`.
Latest run: **98 tests passed**, including pidfd closure when post-open identity
verification raises, without emitted resource warnings;
focused compilation and whitespace checks passed. No full discovery or real GPU
run was performed for this slice.

This is not production-boundary E2E. The common Python parent must capture GPU
proof before Ollama and sibling Python workers start. Runtime composition, live
dependency collection, approved image/deployment, and scheduler-to-GPU proof
remain absent. Unobserved children that daemonize outside the tracked session
require external containment; procfs/pidfd discovery is not a cgroup guarantee.
Touched leaves remain **partial**.

## Supervised runtime composition and live daemon candidate (2026-09-17)

G0/G2.2/G4.1/G4.2/G4.3 **unit** evidence:
`tests.unit.test_bootstrap_runtime`, `tests.integration.test_bootstrap_http`, and
`tests.unit.test_resource_manager_shutdown` cover the composed loopback service,
idle-unready then session-ready, exact result bytes through SSE, artifact/profile/
free-space admission rejection, no repeated artifact hashing/provider creation
from health, daemon-loss fencing, concurrent startup/stop, repeated cancellation,
cleanup-stage failures and retained grace-timeout tasks. RM regressions preserve
provider failure events and completed cancelled responses with no result, so
shutdown fences do not erase watchdog progress. Tests own temporary selected
artifact volumes, synthetic measured registries, result stores, fake daemon/GPU/
providers, HTTP clients/listeners and task gates. This is not measured GPU proof.

Command: `.venv/bin/python -W error -m unittest -v tests.unit.test_bootstrap_runtime tests.integration.test_bootstrap_http tests.unit.test_resource_manager_shutdown tests.unit.test_resource_manager tests.unit.test_resource_manager_health tests.integration.test_resource_manager_health_http tests.unit.test_bootstrap_supervisor tests.unit.test_bootstrap_bindings tests.unit.test_health_http tests.integration.test_resource_manager_http tests.unit.test_scheduler`
— **141 passed in 32.034s**. Focused compilation and whitespace checks passed.
Run only task-related suites going forward, not full discovery.

G4.1/G4.3 **e2e candidate**: installed SmolLM → CoEdIT → GECToR → SmolLM
switching passed again offline, now using `services.llm.bootstrap.supervisor.OwnedOllama`
instead of the old harness launcher. Four successful responses, three replacements,
stale submit/cancel rejection, CoEdIT cancelled-result fencing, and final
provider/daemon/GPU cleanup passed. The tester verified its owned container
`llm-three-model-adapter-check-supervised` absent. Foreign GPU processes were not
manipulated. Image: `sha256:ec0af2cb045557602d42eb94c6645b3e330afa220d489bd34bc86bffb54ce108`;
manifest: `7abbd93bd3e4ec01ba01f8e4581821ae1d2f35cab720695c1309596df5614a19`.
The network-disabled build used the existing hash-locked wheelhouse.
Command/environment/cleanup contract: [three-model check](../../../../docs/three-model-adapter-gpu-check.md).
Focused `.venv/bin/python -W error -m unittest -v tests.unit.test_three_model_adapter_check tests.unit.test_bootstrap_supervisor tests.unit.test_resource_manager_shutdown`
— **29 passed in 17.285s**.

Profiles remain explicitly **unmeasured**. This is real supervisor/installed-adapter
topology evidence, not composed HTTP runtime GPU E2E, memory-safe capacity,
throughput-optimal concurrency, external containment or production image approval.
All touched leaves remain **partial**.

## Identity-fenced memory observation precursor (2026-09-17)

G4.1/G4.2/G4.3 **unit** evidence: `tests.unit.test_gpu_memory` checks typed
integer ranges, reserved bytes, unavailable/error telemetry, balanced NVML
shutdown, worker-thread execution, supervisor loss/reuse and device UUID/count/
MIG fences before and after reads, and full timestamp fence ordering.
`tests.unit.test_three_model_adapter_check` checks six-point lifecycle ordering,
minimal output schema, shared typed proof, explicitly unmeasured results,
baseline/inference/final telemetry failures and changed UUID/total rejection.
Tests own synthetic procfs directories, injected NVML and providers, and candidate
temporary roots; failure cases assert owned cleanup and root removal.

Command: `.venv/bin/python -W error -m unittest -v tests.unit.test_gpu_memory tests.unit.test_gpu_proof tests.unit.test_three_model_adapter_check tests.unit.test_bootstrap_supervisor`
— **70 passed in 17.335s**. Focused compilation and whitespace checks passed.
An intermediate test leak and a first-event chronology assertion were corrected
before this final run; ordinary RM cleanup probes legitimately precede shutdown.

G4.1/G4.3 **e2e candidate**: network-disabled build and real installed-adapter
run passed with exact GPU `GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963`, six ordered
memory points and stable total **12,878,610,432 bytes**. Four successful responses,
three model switches, one cancelled terminal event without a result, stale-result
fencing and final owned cleanup passed. The tester owned
`llm-three-model-memory-check` and verified it absent. RepoDigest (not image ID):
`llm-compatibility-adapter@sha256:f53c0463a2c38de4a48fa139fee740214077709fa152455ccc840413763562a2`.
Commands, output schema and ownership limits:
[three-model check](../../../../docs/three-model-adapter-gpu-check.md).

Memory points include foreign allocations, are not sampled execution peaks, and
do not prove per-request incremental VRAM, a safety reserve, memory-safe `N` or
optimal parallelism. No measured profile was produced. No full discovery was
run. G4.1/G4.2/G4.3 remain **partial**.

### CoEdIT bounded native-batch capacity candidate (2026-09-17)

G4.1/G4.2/G4.3 remain **partial**. The latest independent **unit** command,
`.venv/bin/python -W error -m unittest -v tests.unit.test_python_worker
tests.unit.test_coedit_batch tests.unit.test_capacity_measurement
tests.unit.test_coedit_capacity_check tests.unit.test_coedit_adapter_check
tests.unit.test_coedit_provider tests.unit.test_coedit_benchmark_witness
tests.unit.test_gpu_memory tests.unit.test_python_process
tests.unit.test_gector_worker`, passed **101 tests in 3.816 seconds**.
The offline network-disabled adapter build passed with image manifest
`sha256:dc7a0903f0b86734f373316537766b18262fee90cc554dd44a431066b7612f9d`.

The latest actual **e2e candidate** p=16 run failed closed in warmup with
`native_batch_correlation` after a native 14+2 split; its minimum sampled
whole-device free memory was 7,125,778,432 bytes of 12,878,610,432 (not a
between-sample guarantee). Four
waves were measured at each of p=1,2,4,5,7,8,10,12,13,15. This is scheduling /
batch-correlation failure evidence, not a memory-bound result or proof of
useful native concurrency. The owned container was removed and verified absent;
foreign GPU work was untouched. No profile was written or approved, and the
runtime/default remains p=1. The remaining design boundary must preserve
admission/direct-input safety and the 5 ms bucket contract; it must not use a
benchmark barrier or private execution bypass.

The tester owned `llm-coedit-capacity-p16-diag-v2`, extracted only a bounded
numeric projection of its diagnostic artifact, removed the container and
verified exact-name absence. Earlier p=2 candidate execution passed with the
exact 128-token witness, synchronized Torch allocator and fenced NVML evidence,
candidate optimum 2, profile ineligibility and owned cleanup. Reproducible
invocation and evidence schema: [CoEdIT capacity check](../../../../docs/coedit-capacity-check.md).
These experiments do not prove a memory-safe bound, worst-case decoder workload,
approved profiles or production readiness. No full discovery was run.

The remaining validation-arrival implementation now performs CoEdIT admission's
bounded envelope and exact bucket checks locally instead of issuing one
serialized child tokenizer RPC per request. The child still validates every
native-batch item before allocator measurement or model execution. Its exact
`request_validation_failed` response is recoverable only for `execute_batch`;
the same response for another operation and all uncertain worker/protocol errors
remain fatal. Focused tests cover concurrent local admission, exact execution
rechecks, mixed-batch pre-execution rejection, worker reuse, non-execute fatal
classification, and ResourceManager slot release/reuse after child rejection.
The final command was `.venv/bin/python -W error -m unittest -v
tests.unit.test_python_worker tests.unit.test_coedit_batch
tests.unit.test_capacity_measurement tests.unit.test_coedit_capacity_check
tests.unit.test_coedit_adapter_check tests.unit.test_coedit_provider
tests.unit.test_coedit_benchmark_witness tests.unit.test_resource_manager
tests.unit.test_python_process`: **122 tests passed in 4.255 seconds** with no
warnings. Compilation and `git diff --check` passed. No GPU/Docker rerun was
performed, so this is unit evidence only: the 14+2 observation remains current,
no p=16 success or memory-safe bound is proved, no profile is approved, default
admission remains p=1, and G4.1/G4.2/G4.3 remain **partial**.

Final diagnostic-guard verification used the same focused command: **106 tests
passed in 3.828 seconds**. Derived failure categories now remain on the retained
failed wave, and changed GPU total-memory evidence stops before the next wave.
That run exposed an unawaited coroutine in a CLI test's mocked async runner;
the tester corrected the fixture to close its intercepted coroutine. The exact
worker cancellation test plus `tests.unit.test_capacity_measurement` and
`tests.unit.test_coedit_capacity_check` then passed **33 tests in 1.069 seconds**
without unraisable/coroutine warnings. `git diff --check` passed. These final
failure-path guards were not followed by another GPU run; the image evidence
above precedes them.

### CoEdIT decoder coverage and incremental discovery continuation

G4.1/G4.2/G4.3 **unit** evidence: decoder-start/EOS/PAD handling, strict
configured maximum binding through native IPC, generic output128 support,
short-output noncoverage, complete aggregate schedules, exact sequential
four-repeat discovery, identity/native/allocator/telemetry/reserve failures,
failed-wave retention, and bounded sanitized output. A real default-RM
544-request regression verifies terminal cursors survive event-history eviction.
Tests own synthetic providers, temporary artifacts and their RM/task cleanup.

Command: `.venv/bin/python -W error -m unittest -v
tests.unit.test_capacity_measurement tests.unit.test_coedit_capacity_check
tests.unit.test_coedit_benchmark_witness tests.unit.test_coedit_batch
tests.unit.test_python_worker tests.unit.test_coedit_provider
tests.unit.test_python_process` — **112 passed in4.285s**. Focused compilation
and `git diff --check` passed. No unrelated/full discovery was run.

G4.1/G4.2/G4.3 **e2e candidate**: offline rebuilt p16 throughput passed all59
waves with exact128input/64decoder-step witnesses and cleanup. Subsequently,
`--discover-memory` passed64 waves/544 requests: four serial baselines and four
repeats at every p2..16, exact native correlation and no observation drops.
Minimum sampled free memory was8,192,479,232 of12,878,610,432 bytes. Watchers
resume after each completed wave; an initial zero-cursor history-expiry failure
was not a memory limit. Latest image manifest:
`sha256:118f447d79d88207a2fb3e2be28d146c1cd504abd4e064caad469d76cfbcfd6d`.
The tester removed its `llm-coedit-memory-discovery-v2` container and verified
absence; foreign workloads were untouched. Artifact schema, digest and command:
[CoEdIT capacity check](../../../../docs/coedit-capacity-check.md).

Full workload observation does not establish exhaustive peak memory, a
resource-limitedN or approved profiles. `memory_safe_n:null`,
`profile_eligible:false`; all touched goals remain **partial**.

### Final bounded p=32 discovery closeout

G4.1/G4.2/G4.3 **unit**: config/batcher32 acceptance and33 rejection, default1,
actual32-row result alignment/correlation, mode-aware early CLI rejection,
128-wave/2,112-request discovery schedule, full-schema256KiB artifact bounds
and atomic failure cleanup, discovery/throughput terminal exit statuses.
Testers own fake providers, temporary artifacts, RM tasks and cleanup.

Command: `.venv/bin/python -W error -m unittest -v tests.unit.test_capacity_measurement tests.unit.test_coedit_capacity_check tests.unit.test_coedit_benchmark_witness tests.unit.test_coedit_batch tests.unit.test_python_worker tests.unit.test_coedit_provider tests.unit.test_python_process tests.unit.test_resource_manager tests.unit.test_coedit_adapter_check tests.unit.test_gector_worker`
— **162 passed in9.86s**; focused compilation and whitespace checks passed.
No unrelated/full discovery was run.

G4.1/G4.2/G4.3 **e2e candidate**: rebuilt offline discovery passed every p1..32
with four repeats,128 waves/2,112 requests in212.659s, exit0. Input128 and every
decoder row64 verified; exact native correlation and zero drops throughout.
Minimum sampled free memory was7,862,497,280 bytes with stable total readings.
The full sanitized artifact was127,166 bytes, below the unchanged256KiB cap.
The tester removed its `llm-coedit-memory-discovery-p32-final` container and
verified absence; foreign workloads were untouched. Reproducible command,
image and reduced artifact identity:
[CoEdIT capacity check](../../../../docs/coedit-capacity-check.md#session-closeout-discovery-through-p32).

Result: `observed_through_ceiling`, `observed_safe_through:32`,
`memory_safe_n:null`, `profile_eligible:false`. No resource bound was found;
exhaustive peaks, approved capacity and production readiness remain unproved.
All touched goals remain **partial**.

## QueueScheduler-to-CoEdIT and non-root candidate continuation

G1.1/G1.2/G2.1/G2.2/G4.1/G4.3 **unit**:
`tests.unit.test_scheduler_adapter_check` exercises the actual harness with real
QueueScheduler/SQLite/LocalPublisher/ResourceManager and contract-faithful external
doubles. It covers exact receipt/result correspondence, malformed normal/late
responses, selected-file identity across actual tiny SPECS provisioning, terminal
watching past cancellation, and dependent cleanup ordering/timeout retention.
Tests own temporary artifacts, stores, tasks and fake provider/proof resources.

Command: `.venv/bin/python -W error -m unittest -v tests.unit.test_scheduler_adapter_check tests.unit.test_scheduler tests.unit.test_store tests.unit.test_results`
— **47 passed in 3.842s**. Compilation and `git diff --check` passed; no full
discovery was run.

**e2e candidate**, not qualifying production proof: fresh offline image
`sha256:9ce2cfbe09fce26d955e5cf1ee9841beaa7907a4b672b93749197ee25d59ee55`
passed the scheduler/installed-CoEdIT check as root and `65534:65534` on
`GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963`. Both proved pre-dispatch accepted
SQLite reopen, normal `done` with one exact publication receipt, cancelled late
delegate completion without a handoff/additional receipt, and owned cleanup.
`gpu_ms:null`, `gpu_timing_complete:false`, `profile:unmeasured`,
`profile_eligible:false` remained explicit. Test-owned containers
`llm-scheduler-adapter-check` and `llm-scheduler-adapter-check-nonroot` were removed
and verified absent. The wrapper intentionally declines advisory cancellation
while its gate is held; this is result fencing, not provider interruption proof.
Commands and diagnostic schema: [scheduler check](../../../../docs/scheduler-adapter-gpu-check.md).

The same image also passed existing installed supervised three-model switching
as UID/GID 65534: four responses, three switches, cancellation/stale fencing,
six bounded memory observations and final owned cleanup. Its test-owned
`llm-three-model-adapter-check-nonroot` container was verified absent. This proves
the existing same-UID topology can run non-root, not the distinct `llm`/`ollama`
deployment contract. All touched goals remain **partial**; process-crash/in-flight
recovery, HTTP deployment, measured profiles and production readiness remain gaps.

## Deployment foundation closeout (2026-09-17)

G4.1/G4.3 **unit**: `.venv/bin/python -W error -m unittest -v tests.unit.test_deployment_inputs tests.unit.test_deployment_image`
— **14 passed in 0.012s**. Tests own temporary ZIP wheels, locks, archives,
staging directories and shell subprocesses. Coverage includes canonical wheel
identity, top-level versus vendored metadata, copied-byte verification, rollback,
path safety, Docker policy and fail-closed serve refusal. Targeted compilation
and `git diff --check` passed.

**e2e candidate / packaging smoke only**: actual retained inputs yielded 54
locked wheels and 57 staged files; the clean foundation Docker build passed in
185.0s with image ID
`sha256:11216cfed17487c5ff12a20fe5ee815dd158d389236bcd50c78b51b6e9b87c25`.
Offline help exited 0, unsupported serve exited 78, and non-root bootstrap import
plus ownership/checked-path hygiene passed. Test-owned containers
`llm-provider-foundation-help`, `llm-provider-foundation-serve`, and
`llm-provider-foundation-import` were all verified absent. Generated staging is
retained as gitignored build input; no models or source JSON were copied.

Commands and boundaries: [foundation](../../../../docs/production-image-foundation.md).
This is not provider/GPU serving or qualifying deployment E2E. OS packages still
come from live apt repositories, and distinct-user supervision remains absent.
G4.1/G4.3 and parent G0 remain **partial**.

## Distinct-user broker boundary continuation (2026-09-17)

G0/G2.2/G4.1/G4.3 **unit/local integration**: focused broker/client/supervisor,
deployment/runtime/HTTP and retained-harness suites passed **58 in 20.367s**.
Tests own fake brokers, temporary procfs/socket/stat seams and child processes;
the parent-loss regression subreaps only its own adopted daemon leader.
Stateful tests assert paired filesystem UID/GID restoration order after success
and denied access, and fail-closed transition/restoration failures. Bounded protocol/EOF handling and
leader parent-death fencing are local evidence, not arbitrary runner containment.

**e2e candidate boundary**: `tests/integration/test_image_broker_boundary.py`
is executed as a standalone script via stdin to the real image (not unittest
discovery). Final CPU and selected-GPU-exposed runs passed with network disabled,
default app `llm` UID1001, broker UID tuple `(1001,0,0,0)`, daemon `ollama`
UID/GID1002, version0.11.6, loopback-only listener, duplicate rejection,
hostile environment/cwd isolation, checked path ownership and EOF-driven
broker/daemon disappearance. The harness closes its clients in `finally`;
runner cleanup owns only `llm-phase1-broker-boundary-cpu` and
`llm-phase1-broker-boundary-gpu`. The final credential-assertion correction was
verified by these real runs; the subsequent 58-test run strengthened restoration
coverage and verified both owned container names absent, without an image rerun.

Image manifest:
`sha256:58c319fe8d5362d1982f3a3c8cb9fec4e295faecfddee1574c3be8d4b593be3f`.
Exact commands and release gates:
[broker boundary](../../../../docs/production-image-foundation.md#distinct-user-broker-continuation-2026-09-17).

This supersedes historical absence of distinct-user startup evidence, not the
open deployment proof. No NVML query, inference, model residency, host-PID
Compose run, broker-loss descendant containment or adversarial import-tree
immutability proof was performed. Complete privileged path hardening, reviewed
Section7 compatibility, measured profiles and OS snapshot pins remain gates.
All touched goals stay **partial**; no full suite was run.

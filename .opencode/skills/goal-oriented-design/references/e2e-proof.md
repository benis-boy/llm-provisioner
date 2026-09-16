# Product Proof Inventory

## Current status

G1.1, G1.2, G2.1, G2.2, G3.1 and G3.2 are **partial**: synchronous SQLite
queue/result primitives exist, but async scheduler and real provider boundaries
do not. G4.1, G4.2 and G4.3 remain **target** with contract definitions only.
No qualifying production-boundary E2E proof exists and no leaf is done.

## Local regression inventory (2026-09-16)

Command: `python3 -m unittest discover -s tests -p 'test_*.py'` — **47 passed**.
`python3 -m compileall -q services tests` also passed. All entries below are
**unit** evidence (including local integration tests); none substitutes for
required service/provider E2E. Tests own temporary directories, SQLite databases,
result files, receipts and subprocess cleanup.

| Goals | Stable suite/test IDs | Evidence and limits |
|---|---|---|
| G1.1 | `tests.integration.test_queue_recovery`; `tests.unit.test_queue_regressions.QueueRegressionTests.test_abrupt_subprocess_exit_preserves_request_and_submit_outbox`; `test_same_owner_recovery_preserves_handoff_and_fences_old_attempt` in the same class | WAL crash/reopen, retained handoff and local session replacement; no RM reconciliation |
| G1.2 | `tests.unit.test_results`; `tests.unit.test_store`; `tests.unit.test_queue_regressions.QueueRegressionTests.test_generic_handoff_outbox_ack_is_rejected` | Content-addressed result verification, durable publisher receipt replay, guarded handoff; external publication/cancellation races await delivery coordinator |
| G2.1 | `tests.unit.test_contracts`; `tests.unit.test_queue_regressions.QueueRegressionTests.test_duplicate_finish_attempt_does_not_change_existing_telemetry` | All 36 adjacency pairs, status/timing primitives; not every guarded operation edge and no real GPU telemetry |
| G2.2 | `tests.unit.test_queue_regressions.QueueRegressionTests.test_cancel_is_idempotent_in_scheduled_running_and_on_gpu`; `test_non_retryable_failure_stops_the_entire_queue` in same class | Local cancellation/abort fencing; no completed-response watchdog or provider cancellation |
| G1.2, G2.2 | `tests.unit.test_queue_regressions.QueueRegressionTests.test_retry_delays_are_5_10_20_30_30_across_reopen_and_claim_at_300_is_fenced` | Exact retry delay/cap and 300-second exhaustion across restart |
| G3.1 | `tests.unit.test_store`; `tests.unit.test_function_intent.FunctionIntentTests.test_ready_template_and_both_are_fail_closed_without_attempts`; `test_descriptor_claim_is_fail_closed_without_side_effects` in the same class | Dependency-gated claims, cycle rejection and fail-closed descriptor-bearing claims while ungated work remains claimable; no bounded FIFO scan, function execution or readiness polling |
| G1.1, G1.2, G3.1 | `tests.unit.test_function_intent.FunctionIntentTests.test_ready_and_template_round_trip_across_reopen`; `test_legacy_schema_upgrade_preserves_rows_and_old_fingerprint_replay`; `test_canonical_descriptor_replay_is_noop_but_changes_conflict`; `test_descriptor_idempotency_addition_removal_and_template_changes_conflict`; `test_post_accept_mutation_and_invalid_descriptor_cannot_change_intent`; `test_descriptor_cancel_stop_and_stale_session_are_fenced` in the same class; `tests.unit.test_contracts.ContractTests.test_request_record_validates_descriptor_shapes_and_declared_dependencies` | Canonical immutable accepted descriptor intent, recovery, additive legacy upgrade preserving pending work, idempotency conflicts, dependency-reference boundary and cancellation/session fences; local SQLite evidence only |
| G3.2 | `tests.unit.test_queue_regressions.QueueRegressionTests.test_independent_connections_serialize_skip_line_insertions`; `test_closed_group_reopens_when_original_anchor_is_scheduled_by_retry` in same class | SQLite-linearized priority insertion and group reopening; no real dispatch-order E2E |

OpenAPI tests are dependency-free text checks, **not** parser/schema validation.
Docker and NVIDIA GPU access were verified through the host daemon on 2026-09-16
with a disposable network-disabled `nvidia-smi` container: RTX 4070 Ti, 12,282 MiB,
driver 591.86, compute capability 8.9. See
[environment evidence](../../../../docs/implementation-decisions.md#environment-reassessment-2026-09-16).
This is not compatibility-spike or product proof. Artifact completeness,
production runtime compatibility and three-model offline inference remain
unverified; no runtime pins, measured GPU capacity or offline model execution
are claimed.

## Required production-boundary evidence

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

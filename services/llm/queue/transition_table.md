# Queue lifecycle contract

## Complete status adjacency matrix

Rows are current status and columns are target status. `Y` means the status
edge is structurally adjacent; `N` means rejected. Same-status entries are
replay no-ops, not transitions.

| from \\ to | scheduled | running | on_gpu | done | error | cancelled |
|---|---:|---:|---:|---:|---:|---:|
| scheduled | Y | Y | N | N | Y | Y |
| running | Y | Y | Y | Y | Y | Y |
| on_gpu | N | Y | Y | Y | Y | Y |
| done | N | N | N | Y | N | N |
| error | N | N | N | N | Y | N |
| cancelled | N | N | N | N | N | Y |

`can_transition` implements only this adjacency table. The generic store
`transition` is deliberately restricted: `done` is set only by
`acknowledge_handoff`, and error/cancelled terminalization always fences active
attempts and cancels pending submit/handoff deliveries. The store must enforce
the operation guards transactionally: `done` requires durable content-addressed
result availability and acknowledged idempotent publication; retries require a
retryable failure and persisted budget; `on_gpu` requires matching session,
attempt, residency generation, and admission; cancellation and stop fence late
tokens; terminal records cannot be mutated. Duplicate operations with the same
identity are successful no-ops. A conflicting idempotency key is
`idempotency_conflict`.

Buffer-full is backpressure, not an attempt, retry, error, or status transition.
Retryable failures use 5-second exponential backoff capped at 30 seconds and a
5-minute budget measured from the first retryable failure (`now - first_retry_at`),
not caller-provided elapsed time. A retry invalidates its old submit and emits a
best-effort cancel. A pending durable result handoff owns the request and cannot
be retried; recovery replays it without rerunning the provider. Non-retryable
failures and idle timeout stop the
whole scheduler. GPU timing may be incomplete and never blocks publication.

Dispatch is eligible FIFO. Dependency completion, registered function
availability, readiness, and template validity are gates. In the current
boundary, a request carrying a durable `ready` or `template` descriptor is
fail-closed at direct `claim`: it remains scheduled and emits no attempt, submit,
or claim event. The scheduler-owned evaluator opens that gate only through a
guarded `claim_evaluated` capability; descriptor persistence is not execution. Skip-line groups
extend at the group tail before their scheduled anchor; with no scheduled
anchor, insertion falls back to append. Rank, insertion sequence, anchor, and
group sequence are distinct durable fields. Optional functions poll every one
 second by default, configurable.

## Slice F evaluation boundary

`EligibilityEvaluator.next_eligible()` reads bounded candidate slices and skips
blocked nodes without fetching the full queue. Awaited callbacks execute outside
SQLite transactions. An opaque capability contains queue version, session,
generation, fingerprint, and evaluated payload reference; only
`QueueStore.claim_evaluated()` can consume it. Claim creates the attempt and
submit outbox atomically. The separate QueueScheduler owns watchdog and
ResourceManager admission; these are not store/evaluator responsibilities.

Evaluation does not make retry-delayed work eligible or arm scheduler watchdog
work. Both success and callback errors are discarded if the durable version,
session/generation, cancellation/stop state, or replacement session changed
since the evaluation snapshot. Evaluated capabilities are single-use internal
claims rather than a public direct-claim convenience; a durable version change
removes all unconsumed capabilities. Template payload references are persisted
with the created attempt and submit outbox record.

## Phase 3 acceptance boundary

The scheduler acceptance slice is local to the queue and in-process
ResourceManager. It covers the six-status lifecycle, admission and completion
timing, duplicate and fenced callbacks, cancellation at each dispatch stage,
stop/idle behavior, retry/backpressure, and deterministic append/skip-line
insertion with bounded eligible FIFO evaluation. Existing Phase 2 operation
guard and recovery suites remain independent durability and publication
evidence; this phase does not claim GPU, deployment, capacity, or production
end-to-end evidence.

### Stable Phase 3 evidence map

| Requirement | Local scheduler acceptance evidence | Independent guard/recovery evidence |
|---|---|---|
| Six-status projection, complete/incomplete GPU timing, duplicate/stale progress | `test_integrated_status_path_and_complete_gpu_timing`; `test_incomplete_timing_and_progress_handler_fences_duplicate_or_stale_events` (direct handler fence only) | `ContractTests.test_transition_matrix_checks_each_of_36_pairs`; `QueueRegressionTests.test_gpu_events_and_finished_telemetry_are_write_once` |
| Cancellation through decoder, uncertain submit, admission, buffered work, pending handoff, terminal no-op and no late publication | `test_cancel_decoder_uncertain_submit_and_on_gpu_fence_late_work`; `test_cancel_buffered_handoff_and_terminal_noops_do_not_publish_late_work`; `test_cancel_active_fences_late_completion_and_replay` | `Phase2OperationGuardMatrixTests.test_guard_matrix_terminal_cancel_stop_and_generic_transition` |
| Idle/non-retryable abort and late-result fence; replacement-session supersession | `test_idle_abort_terminalizes_active_blocked_and_retry_delayed_work`; `test_session_invalidated_aborts_active_blocked_and_retry_delayed_work`; `test_nonretryable_failure_stops_all_work` | `Phase2HTTPRecoveryTests.test_process_death_after_http_acceptance_reconciles_new_attempt` |
| Retry backoff/exhaustion and backpressure consumes no budget | `test_retry_backoff_budget_and_submit_backpressure_preserve_intent`; `SchedulerIntegrationTests.test_retry_uses_store_wall_clock_while_watchdog_keeps_monotonic_clock`; `test_lost_submit_ack_replays_same_attempt_key` | `QueueRegressionTests.test_retry_delays_are_5_10_20_30_30_across_reopen_and_claim_at_300_is_fenced` |
| Eligible FIFO with dependency/readiness/template gates, bounded function polling, invalidated scan, and recovered unavailable functions | `test_dependency_template_and_awaited_readiness_invalidation_order`; `test_recovered_missing_descriptor_is_request_local_and_valid_work_completes`; `test_fifo_skips_blocked_head_and_unavailable_function_is_local`; `SchedulerIntegrationTests.test_optional_function_poll_cadence_and_local_enqueue_wake`; `test_large_function_poll_interval_does_not_delay_armed_watchdog`; `test_large_poll_dispatches_two_ready_requests_without_per_request_delay`; `test_independent_mutation_discards_cached_capability_for_new_head` | `EligibilityTests.test_early_node_change_restarts_scan_during_later_await`; `test_sync_async_functions_receive_detached_selected_dependencies` |
| Append/skip-line ordering, grouped anchors, retry reopening, concurrent insertion | `test_integrated_skip_line_dispatches_before_its_anchor`; `test_concurrent_independent_skip_line_inserts_dispatch_by_recorded_sequence` | `QueueRegressionTests.test_independent_connections_serialize_skip_line_insertions`; `test_closed_group_reopens_when_original_anchor_is_scheduled_by_retry` |

This map is local integration/unit evidence only.  It neither adds
`reconcile_expired` to the live scheduler nor claims GPU/deployment E2E: the
live watchdog renews owned decoder and uncertain-submit leases, while recovery
fences the old RM session before durable replay. Renewal is deliberately a live
owner operation: it only extends attempts fenced by the current local session
and generation. Startup first establishes the local durable recovery fence,
then obtains the replacement RM session. Dispatch and outbox replay start only
after both fences are established, so live lease renewal cannot authorize
post-restart execution. The watchdog's
monotonic elapsed clock is never persisted; QueueStore exclusively assigns its
restart-stable wall-clock retry and lease timestamps. Optional-function
evaluation is bounded by its configured poll interval; local enqueue/cancel
mutations invalidate cached evaluation and wake it immediately.
Positive cached readiness also expires at the poll deadline. Direct cache
regression coverage verifies that external-state changes replace an obsolete
eligible candidate on the next poll
(`SchedulerIntegrationTests.test_positive_cache_expires_and_rescans_external_readiness`).

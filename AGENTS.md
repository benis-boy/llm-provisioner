# Repository Agent Context

## Product outcome and durable limits

G0 is **partial**: clients and operators should complete reliable,
capacity-aware queued LLM work without losing accepted work, misrepresenting
progress, or exceeding proved capacity. Existing goal IDs, statuses, and proof
evidence in `.opencode/skills/goal-oriented-design/references/` are the
authoritative record; do not initialize, promote, or rewrite them during
ordinary implementation. Current queue contracts cover durable intent,
fenced attempts/publication, truthful six-status progress, cancellation and
stop behavior, eligible FIFO/gated dispatch, and deterministic insertion.
GPU, measured-capacity, deployment, and production-boundary proof remain
partial unless the proof inventory says otherwise.

## Sources of truth and invariants

- `services/llm/queue/transition_table.md` is authoritative for queue status
  adjacency, operation guards, retry/backpressure, evaluation capabilities,
  and scheduler ownership boundaries.
- `docs/queue-scheduler-resource-manager-plan.md` and its partial-completion
  record describe sequencing and evidence; the goal tree and proof inventory
  preserve statuses and stable evidence.
- `docs/openapi.yaml` and the service serializers are the API contract.
  Preserve durable identity, idempotency, content-addressed result
  publication, stale-session/generation fences, and fail-closed capacity or
  artifact checks.

## Workflows and boundaries

The project targets Python >=3.12 (`pyproject.toml`). The canonical focused
test form is `.venv/bin/python -W error -m unittest -v <tests>`; use exact
test modules or test names when possible. Generated caches, model weights,
large JSON/JSONL measurements, Docker images, and compatibility artifacts are
not source edits: keep them out of commits and use the bounded tooling under
`tools/` with the documented artifact boundaries. Do not claim GPU, browser,
deployment, or production readiness from copied guidance or local unit tests.

Product goals and proof inventories are living summaries: when product
direction, capability boundaries, or system-of-record decisions change, update
the summary in the same approved change. Goal initialization requires a fresh
session, discovery, human correction, and explicit approval before any
authoritative write; never run `/init-goals` silently.

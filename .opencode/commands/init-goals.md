---
description: Infer and seed project goal documentation with human approval
agent: design
---

Initialize or rebaseline goal documentation in two approval-gated phases.
Complete Phase 1, present its result, and stop. Apply Phase 2 only after the
human explicitly approves the current proposal for writing.

## Phase 1 — Discover and propose

1. Load `goal-oriented-design`, its reference index, the existing goal tree,
   and proof inventory. Treat an initialized tree as authoritative context,
   not permission to change statuses.
2. Inspect `AGENTS.md`, `pyproject.toml`, `services/llm/queue/transition_table.md`,
   the queue/resource-manager docs, implemented behavior, and tests. Keep
   product direction, implemented contracts, and proof evidence distinct.
3. Use bounded read-only scouts where useful. Record concrete paths/symbols,
   observed behavior, gaps, contradictions, and aspirational ideas. For
   candidate proof, record exact test IDs/commands, unit versus e2e type,
   real versus replaced dependencies, prerequisites, and cleanup ownership.
4. Propose a concise hierarchy of durable outcomes, not services, packages,
   APIs, migrations, screens, or tasks. Preserve existing G0 and goal IDs;
   assign only `target`, `partial`, `untested`, or `done` from evidence.
5. Record a content fingerprint for every destination document the proposal
   would edit, including the goal tree, proof inventory, and index (or its
   absence if it would be created), so approval is tied to that baseline.

Present the candidate outcome tree, statuses and rationale, evidence links,
contradictions, omitted aspirations, proof gaps, and exact files Phase 2
would change. Ask for approval or corrections. Corrections require a revised
proposal and another approval; Phase 1 never edits goal, proof, index, or
product-direction files.

## Phase 2 — Seed approved documentation

After explicit approval, recheck the named evidence and the recorded content
fingerprints of all destination documents. If either changed, return to
Phase 1 and obtain approval for a revised proposal. Otherwise write only the
approved scope to the goal tree, proof
inventory, index, and any approved product guidance. Preserve all accurate
existing goal IDs, statuses, and evidence. Create proof rows for approved
leaves, recording gaps explicitly; do not claim test execution or production
proof merely because initialization completed.

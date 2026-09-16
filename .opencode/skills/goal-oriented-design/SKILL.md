---
name: goal-oriented-design
description: Use broadly for design work involving product goals, product-code, behavior, verification, or goal documentation; provides the goal workflow and proof inventory for the primary Design agent.
---

# Goal-Oriented Design

Before mapping work, read the [reference index](references/index.md) and the
relevant goal tree. Before choosing verification, read the relevant proof
inventory. Read only the documents needed for the touched branch.

Goals describe product outcomes across implementation boundaries; they are not a decomposition of services, packages, APIs, tables, or tasks.

Use [references/goal-tree.md](references/goal-tree.md) as the current target-state hierarchy. Follow its links only when the touched branch needs deeper context.

## Initial Setup And Rebaselining

When this skill is first installed in a project, or when its goal tree is known to be stale, delegate repository and documentation discovery to a read-only scout. Require it to return:

- the overarching goal stated by product direction;
- a high-level outcome tree inferred separately from documentation and implemented source behavior;
- evidence links and implementation status for each proposed goal;
- candidate leaf-to-E2E mappings and proof gaps;
- contradictions and aspirational examples that should not be treated as current capabilities.

Present the extracted goals to the user for correction before treating inferred target-state additions as authoritative. Do not organize the result by service. Store only the concise hierarchy in the goal-tree reference and link deeper descendants to product documentation.

## Required Design Workflow

For every prompt where this skill is useful:

1. Reject designs that advance a child outcome while undermining a parent outcome; goals describe target state rather than prescribing architecture.
2. Whenever Design communicates goals to any subagent, write every relevant goal and necessary parent outcome in plain text in that assignment. IDs may accompany the text for traceability, but never substitute for it; links and references must never be required for the subagent to understand the outcome.
3. If implementation establishes, removes, or materially changes a target, update `references/goal-tree.md` and [references/e2e-proof.md](references/e2e-proof.md) in the same change. Update linked product documentation when the user requests it or when the implemented behavior makes that documentation inaccurate.
4. Read the relevant proof inventory and select suites that cover the touched goals.
5. In the final result, name touched goal IDs and summarize verification evidence or proof gaps.

When the prompt does not map cleanly to the tree, do not force it under a nearby goal. Determine whether it is an implementation detail that supports an existing goal or evidence of a genuinely missing target-state branch. Add a goal only when it expresses an enduring user or product outcome across implementation boundaries.

## Goal Quality Rules

- A high-level goal states a durable product outcome and remains meaningful if services, storage, or APIs are reorganized.
- A child narrows its parent's outcome. It does not merely list the mechanism used to fulfill it.
- Keep this skill compact. Put deeper child goals in linked product documentation rather than copying their detail here.
- Use exactly these statuses: `target` means intended but not implemented; `partial` means only part of the outcome is implemented; `untested` means the outcome is implemented but required E2E proof is missing, regardless of unit tests; `done` means implemented and required E2E proof exists. Assess mixed branches conservatively: a parent remains partial when any essential child outcome is incomplete or unproven.
- Examples such as arbitrary input, arbitrary analysis output, editable drafts, exports, or intent-aware spellchecking are candidate outcomes, not automatically current goals. Adopt them only when repository evidence or explicit product direction supports them.

## Leaf-Goal Proof Contract

Every leaf goal requires multiple independent tests, including at least one
e2e test in a production environment. Tests create and clean up their own
prerequisites using the fastest suitable method. A qualifying inventory entry:

- labels each test as `e2e` or `unit`;
- records cleanup ownership and stable commands where useful;
- proves a product outcome, not merely that an endpoint returned or a component rendered.

For this contract, a **production environment** exercises production
implementations and the real required service boundaries. It may be local, CI,
staging, or deployed production; it does not mean customer production. A
e2e test may not replace a required production dependency with a fake or mock.
Other tests may use fakes and mocks where appropriate; there is no blanket mock
ban.

There is no blanket prohibition on mocks or fakes and no direct-database
mandate. Unit tests are regression evidence but can never advance a leaf beyond
`untested` by themselves. Split long processes into short independently
verifiable transitions; A=>C may be established by A=>B and B=>C when B's
identity and contract are asserted. When relying on that decomposition, inventory
the transition suites and intermediate identity and contract. UI-only outcomes are proof of product
goals, not product goals themselves. Migrations are implementation mechanisms,
never product goals.

Libraries have independent lifecycles: each library has its own root goal, leaf
hierarchy, test environments, and proof inventory. Map library outcomes to
product goals only where appropriate; do not nest technical library details in
the product tree.

## Required Design checklist

- Read the relevant tree before mapping and the relevant proof inventory before verification selection.
- State outcomes in plain text in every subagent assignment.
- Keep goals durable and concise; keep stable test IDs in proof inventories only.
- Reconcile mixed statuses conservatively and report proof gaps honestly.

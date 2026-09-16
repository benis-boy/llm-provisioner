---
name: react-composition-patterns
description: React component architecture, compound components, context providers, and reusable APIs. Use when boolean props proliferate, shared state crosses component boundaries, or a reusable component API needs redesign.
license: MIT
metadata:
  source: vercel-labs/agent-skills
  adapted-for: book-data-analyzer
---

# React Composition Patterns

Use composition to make component variants explicit without turning every component into a framework.

## Apply When

- A component has several boolean props that produce interacting modes.
- Sibling or surrounding controls need access to the same state and actions.
- A reusable component needs flexible structure without many render callbacks.
- State management details prevent otherwise reusable UI from being shared.

## Core Patterns

- Prefer explicit variant components over boolean mode combinations. Each variant should compose the pieces it needs.
- Use `children` for static structural composition. Use render props only when the parent must pass data or behavior into the rendered child.
- For complex reusable widgets, use compound components backed by a narrowly scoped context.
- Define context as a stable contract of state, actions, and necessary metadata. Providers own the concrete state implementation; UI consumes the contract.
- Lift state to the smallest provider boundary that contains every consumer. Visual nesting does not need to match state ownership.
- In React 19, accept `ref` as a normal prop rather than adding `forwardRef` to new components. Use `use(Context)` where conditional context consumption is specifically useful; ordinary `useContext` remains valid.

## Restraint

- Keep a simple component simple. Do not introduce compound components for one caller or a single independent option.
- Do not hide important behavior in a generic context. Keep business operations explicit and typed.
- Avoid one giant provider that causes unrelated state and actions to share a lifecycle.
- Follow the existing UI's architecture and naming before introducing a new composition convention.

## Review Questions

- Are invalid prop combinations representable?
- Is each variant clear from its call site?
- Does the provider expose an implementation-independent contract?
- Can state be placed closer to its actual consumers?
- Would plain props and children be simpler than context?

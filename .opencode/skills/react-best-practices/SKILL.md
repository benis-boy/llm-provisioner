---
name: react-best-practices
description: React, TSX, JSX, hooks, effects, rendering, and frontend performance. Use when creating, editing, or reviewing React UI code; apply React 19 patterns while respecting the repository's Vite architecture.
license: MIT
metadata:
  source: vercel-labs/agent-skills
  adapted-for: book-data-analyzer
---

# React Best Practices

Apply these rules to `services/ui`. This project uses React 19 and Vite, not Next.js. Do not introduce Next.js, React Server Components, SWR, or framework-specific caching patterns.

## Correctness First

- Derive values from current props and state during render. Do not mirror derived values into state with an effect.
- Put interaction-driven work in the event handler that caused it, not in state-plus-effect machinery.
- Use functional state updates whenever the next state depends on the previous state.
- Use lazy `useState` initialization for expensive initial values.
- Do not define components inside components; doing so remounts them on each parent render.
- Keep effect dependencies narrow and accurate. Split effects that synchronize unrelated systems.
- Use `useEffectEvent` for non-reactive callbacks called by an effect. Do not include the returned Effect Event in dependency arrays.
- Use refs for transient values that should not trigger rendering, not for visible application state.
- Avoid mutable module-level user or request state. Immutable constants and deliberate keyed caches are acceptable.

## Concurrency And Responsiveness

- Use `startTransition` or `useTransition` for non-urgent rendering updates, not for every asynchronous operation.
- Use `useDeferredValue` when an urgent value such as text input drives an observably expensive render.
- Keep network requests independently concurrent when there is no dependency between them; avoid sequential `await` waterfalls.
- Start dependent work as soon as its prerequisite is available instead of waiting for unrelated work.
- Use `Suspense` only where the data or component integration supports it and the loading boundary improves the user experience.

## Rendering And Browser Work

- Use explicit conditionals when a numeric or otherwise renderable falsy value could leak through `&&`.
- Prefer CSS classes over repeated imperative style mutation. Batch DOM reads and writes when imperative layout work is necessary.
- Use passive listeners for scroll, wheel, and touch observation only when the handler never calls `preventDefault`.
- Use `content-visibility` or virtualization for genuinely long lists after confirming rendering is a bottleneck.
- Keep persisted browser data minimal, version its schema, and handle storage API failures.
- Preserve immutable props and state. Prefer `toSorted()` or a copied array over mutating `sort()`.

## Optimization Discipline

- Do not add `useMemo`, `useCallback`, or `memo` by default. Use them only for measured expensive work, required referential stability, or an established local pattern.
- Prefer direct, readable code over low-impact micro-optimizations. Optimize repeated lookups with `Map` or `Set` when data size or frequency justifies it.
- Lazy-load truly heavy, optional UI and third-party code. Do not split small components merely to create more chunks.
- Respect generated-code boundaries under `services/ui/src/generated`; change OpenAPI sources and regenerate instead of hand-editing clients.

## Verification

Run the narrowest relevant checks first. Typical UI checks are `npm run typecheck`, `npm run lint`, and focused Playwright tests from `services/ui`; use the repository's canonical end-to-end runner when behavior crosses services.

---
name: golang-best-practices
description: Go, Golang, go.mod, go.sum, *.go, internal/, cmd/, package design, tests. Use when creating or editing Go code, planning Go project layout, or reviewing Go changes for idiomatic structure and error handling.
---

# Golang Best Practices

Use this skill when the task involves Go code, Go module structure, or code review of Go changes.

## Source Priorities

- Prefer the repo's existing patterns first.
- For language idioms, follow `Effective Go` and current module guidance from `go.dev`.
- Keep changes minimal. Do not introduce framework-style directory trees unless the repo already needs them.

## Default Go Rules

- Run `gofmt` on any edited Go files.
- Use clear package names: short, lowercase, no underscores.
- Avoid Java-style naming such as `GetX`; prefer `Owner()` over `GetOwner()`.
- Keep the successful path flowing downward. Handle errors early and return quickly.
- Prefer concrete types and small interfaces defined where they are consumed.
- Use `internal/` for non-public packages by default.
- For multi-command repos, prefer `cmd/<name>/main.go`.
- Keep related code in the smallest reasonable number of packages.

## Project Layout Heuristics

- If the repo is a single library, keep the main package at the module root until complexity forces extraction.
- If the repo is a server or application, prefer `cmd/` for binaries and `internal/` for app packages.
- Add public subpackages only when there is a real reuse boundary.
- Do not create `pkg/` by default just to look conventional.

## Implementation Guidance

- Use constructors only when zero values are not useful.
- Prefer slices over arrays for general-purpose collections.
- Use `defer` for cleanup close to resource acquisition.
- Return wrapped errors with actionable context.
- Keep functions focused; split only when a helper improves readability or reuse.
- Avoid speculative abstractions, especially generic interfaces that only have one implementation.

## Testing Guidance

- Prefer table-driven tests when multiple inputs map to the same behavior.
- Test exported behavior, not private implementation details.
- Keep test fixtures local and small.

## Review Checklist

- Is the package layout proportional to the current size of the codebase?
- Did the change keep names idiomatic and concise?
- Are errors handled immediately and consistently?
- Are `context.Context` and cancellation propagated where appropriate?
- Are public APIs minimal and stable-looking?
- Did any new `.go` files get formatted?

## References

- `https://go.dev/doc/effective_go`
- `https://go.dev/doc/modules/layout`

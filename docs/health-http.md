# Health HTTP precursor

`services.llm.health` supplies a transport-neutral, injected snapshot/probe
boundary and an aiohttp application for `GET /health/live`, `GET
/health/ready`, and `GET /health/dependencies`.

This is an unwired, read-only precursor. It is **not** proof of production GPU
readiness: a future bootstrapper must inject cached state proving the exact
artifact, adapter, GPU, and measured profile. The boundary never loads models,
runs inference, downloads artifacts, benchmarks, changes residency, admits
work, or writes durable state. Missing, malformed, transitional, failed, or
unproved state fails readiness closed. Liveness only reports that this
application handler can run.

Each evaluation captures one injected snapshot atomically and probes that
captured state. Snapshot providers must return an immutable value, or keep the
returned value immutable for the duration of the evaluation; mutable state
should be replaced atomically rather than changed in place.

Injected dependency probes are bounded to five seconds by default and execute
blocking callables in worker threads. Concurrency is capped; a timed-out
worker is retained until it actually finishes so capacity is not leaked.
Readiness reasons are bounded objects (`{"phase": ...}` or
`{"dependency": ..., "reason": ...}`), while dependency diagnostics contain
only fixed dependency names, booleans, and allow-listed reason codes. Responses contain only bounded, allow-listed booleans and reason codes; probe
exceptions and paths, secrets, prompts, results, and tokens are never exposed.

Probes are started concurrently and each has the configured five-second
deadline. A fail-fast semaphore reports overload without waiting for a slot.
Cleanup has its own short bound: asynchronous probe tasks are cancelled, while
blocking thread work is detached until completion and continues to own its
worker slot.

# Resource-manager HTTP adapter

The adapter exposes only the RM session lifecycle, bounded submission,
capacity, cancellation, stop, and resumable progress stream. It does not
implement scheduler, provisioning, health, or compatibility endpoints.

Profiles and providers are selected exclusively by server configuration and
`ProfileStore.lookup` exact measured identity. Client `profile` and `provider`
arguments are local compatibility selectors; they are never serialized or
trusted. A submission references bytes already persisted in the same trusted
shared `ResultStore` as `sha256:<64 lowercase hex>`. This is a precursor for
shared storage, not a general network upload API. Inputs and serialized SSE
frames are bounded to 8 MiB by default; base64 and JSON overhead reduce the
largest result that fits a frame.

Every mutation requires `Idempotency-Key`. Backpressure is a `429` response
whose body has `accepted: false, backpressure: true`; it is not a failed
attempt and must not be treated as an error retry classification. Cancellation
is request-level and therefore never accepts an attempt token; stop accepts an
optional reason. HTTP failures use `{error: {code,message,retryable}}` envelopes.
Submit may omit both selectors so the active measured profile validates the
normal scheduler path; supplied hints are exclusive and must match that profile.

SSE uses `id: <sequence>`, `event: progress`, and strict camelCase JSON data.
`Last-Event-ID` resumes only events with sequence greater than the cursor;
future and expired cursors are rejected before headers where possible. Result
bytes are currently base64 on the wire and remain bounded; a future durable
reference protocol may replace this representation. Incomplete GPU timing is
preserved as `null` with `gpuTimingComplete: false`.

Session capability tokens are opaque and must never be logged. The server
closes the core iterator on client disconnect. Finite client timeouts and
frame/body bounds are mandatory. The server verifies referenced files with a
no-follow regular-file, size, and SHA-256 check off the event loop before it
submits them to the core. Concurrent watches are fail-fast bounded and SSE
event IDs must agree with their JSON sequence values.

Local verification exercises the real loopback HTTP boundary with fake providers
and synthetic measurement fixtures, including exact scheduler result publication
and all three model selector shapes. This is not measured GPU capacity or
production-boundary E2E proof. Future API operations in `openapi.yaml` are marked
`x-implementation: future`; they are not routes served by this adapter.

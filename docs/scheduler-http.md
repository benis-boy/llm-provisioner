# Scheduler HTTP

`SchedulerHttpServer` exposes the durable queue through a caller-supplied
`dict[str, QueueScheduler]` registry.  The registry is the trust boundary:
there is no dynamic import, path construction, or payload-reference decoding
in this adapter.  Applications construct schedulers, providers, profiles,
and payload consumers before starting aiohttp.

The six operations are `start`, `enqueue`, `get`, request-specific `watch`,
`cancel`, and `stop`.  Mutation requests require `Idempotency-Key`; the key is
passed to the scheduler's durable operation journal.  `stop` accepts `error` or
`cancelled`, and cancellation is forwarded as `cancelled=True`.

Watch uses the global durable event cursor. Events are selected in bounded,
request-specific cursor batches but retain global IDs, so gaps are expected.
Each event carries
the immutable request projection captured in the transaction that emitted it.
Future cursors are rejected before SSE headers; legacy events without a
snapshot fail with `legacy_event_unavailable`.  Streams send bounded frames,
heartbeats at the configured interval, and stop after delivering a terminal snapshot. This slice does
not implement client reconnect automation; callers resume with
`Last-Event-ID` using `SchedulerHttpClient` or their own client.

The adapter returns the documented top-level `Error` object (`code`, `message`,
and `retryable`) with bounded generic messages; it does not expose raw
exceptions. Result and payload references remain opaque to HTTP consumers.
Applicable scheduler responses are `400` for malformed input/cursors, `404`
for unknown scheduler/request, `409` for durable conflicts, future cursors, or
legacy replay, `413` for bounded request/SSE frames, `429` for watcher limits,
and `500` for bounded internal failures.

`SchedulerHttpClient.watch` yields validated wire envelopes with `id`, `event`,
and `data`; callers resume from the numeric `id`.  A terminal request whose
cursor is at or beyond its last event closes immediately rather than emitting
heartbeats indefinitely.

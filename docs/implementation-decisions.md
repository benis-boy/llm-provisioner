# Phase 0 implementation decisions

This contract slice supports G0 and G1.1, G1.2, G2.1, G2.2, G3.1, G3.2,
G4.1, G4.2, and G4.3. Local queue/result primitives now accompany the contracts;
G1–G3 outcomes are partial and no leaf is done without production E2E proof.

## Settled decisions and blockers from plan section 11

1. **Backup and retention — owner: operations; blocker.** Durable request and
   result retention defaults to no automatic deletion. Operator backup and
   retention policy must be supplied before production acceptance.
2. **Async transport — owner: platform; settled.** HTTP/1.1 JSON plus SSE is
   used. Trusted v1 has no authentication or authorization.
3. **Function runtime — owner: application; settled.** Startup-registered sync
   or async Python functions receive finite JSON arguments and declared result
   IDs. Missing names produce `function_unavailable`; polling defaults to one
   second and is configurable.
4. **Result handoff — owner: queue worker; settled.** Local publication is an
   fsynced, atomically renamed content-addressed result plus durable outbox;
   `done` follows publisher acknowledgment. External sinks require idempotency.
5. **Function polling — owner: application; settled.** Default is one second.
6. **GPU and pins — owner: operations; blocker.** Target GPU, CUDA/Python/
   PyTorch/Transformers/gector/Ollama versions require compatibility-spike
   evidence and are intentionally not guessed.
7. **Hashes and benchmark inputs — owner: provisioning; blocker.** Exact
   manifest, transitive hashes, GECToR vocabulary hash, and configured benchmark
   requests require real mounted artifacts and profile measurements.
8. **Model volume and registry — owner: operations; settled.** Content-addressed
   model volume is fixed by the plan. Signing and registry policy are operator
   owned; no image-layer or registry decision is reopened here.

OpenAPI HTTP bindings, exact manifest/profile SQLite persistence, async scheduler
and all provider behavior remain unimplemented. Queue/result persistence is
implemented as a separate synchronous core slice; see [its boundaries](queue-core.md).

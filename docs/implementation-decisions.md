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
    second and is configurable. Slice E persists descriptors and blocks their
    claims; execution and polling remain target behavior. Descriptor result IDs
    identify declared same-scheduler dependency requests, resolved in future
    execution to acknowledged durable result references.
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

## Environment reassessment (2026-09-16)

Docker-outside-of-Docker access is available. `docker version` reported client
29.8.1-1 and daemon 29.7.2 (Docker Desktop); the daemon advertises the `nvidia`
runtime. A temporary, read-only, network-disabled container from an already local
image successfully queried the GPU:

```sh
docker run --rm --pull=never --network none --read-only --cap-drop ALL --security-opt no-new-privileges --gpus all --entrypoint nvidia-smi nvidia/cuda:12.4.1-base-ubuntu22.04 --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
```

- NVIDIA GeForce RTX 4070 Ti, UUID `GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963`
- Driver 591.86; total VRAM 12,282 MiB; compute capability 8.9

The probe removed its container and neither downloaded nor loaded model artifacts.
This replaces the historical claim that Docker/NVIDIA access is absent. It is
environment discovery only—not a production image choice, dependency pin,
capacity measurement, compatibility-spike pass or provider E2E proof. Exact
artifact/transitive-file availability, NVML/PyTorch device agreement and the
three-model load/infer/cancel/unload/switch matrix still need verification.

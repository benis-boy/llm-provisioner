# Phase 0 implementation decisions

This contract slice supports G0 and G1.1, G1.2, G2.1, G2.2, G3.1, G3.2,
G4.1, G4.2, and G4.3. Local queue/result primitives now accompany the contracts;
G1–G3 outcomes are partial and no leaf is done without production E2E proof.

## Settled decisions and later evidence gates from plan section 11

1. **Backup and retention — owner: operations; contract decision settled.**
   Durable request and result retention defaults to no automatic deletion. An
   operator backup and retention policy is a Phase 6 production-evidence gate;
   this phase neither selects that policy nor waives it.
2. **Async transport — owner: platform; settled.** HTTP/1.1 JSON plus SSE is
   used. Trusted v1 has no authentication or authorization.
3. **Function runtime — owner: application; settled.** Startup-registered sync
    or async Python functions receive finite JSON arguments and declared result
    IDs. Missing names produce `function_unavailable`; polling defaults to one
     second and is configurable. Slice E persists descriptors; Slice F evaluates
     them and issues single-use fenced claims. Descriptor result IDs identify
     declared same-scheduler dependency requests and resolve only to acknowledged
     durable result references. Sync user callbacks run off the event loop.
4. **Result handoff — owner: queue worker; settled.** Local publication is an
   fsynced, atomically renamed content-addressed result plus durable outbox;
   `done` follows publisher acknowledgment. External sinks require idempotency.
5. **Function polling — owner: application; settled.** Default is one second.
6. **GPU and pins — owner: operations; contract decision settled.** Target
   GPU, CUDA/Python/PyTorch/Transformers/gector/Ollama versions require
   compatibility-spike evidence and are intentionally not guessed. Exact
   compatible-row approval remains a Phase 1 evidence gate.
7. **Hashes and benchmark inputs — owner: provisioning; contract decision settled.**
   Selected manifests must hash every required transitive file, including
   GECToR vocabulary. Generate maximum valid requests only when the adapter can
   verify validity; otherwise require configured benchmark requests and fail
   closed when absent. Existing selected-file candidate evidence is not invented
   or promoted here: approved provenance and measured benchmark evidence remain
   Phase 1/5 prerequisites.
8. **Model volume and registry — owner: operations; settled.** Content-addressed
   model volume is fixed by the plan. Signing and registry policy are operator
   owned; no image-layer or registry decision is reopened here.

The scheduler, Resource Manager, artifact-verification, profile-validation,
health HTTP bindings, and all three installable provider adapters are
implemented bounded local components. Server-owned bootstrap binding performs
offline exact-identity artifact/profile preflight and assembles those adapters.
Measured-capacity provisioning, approved deployment packaging and production
provider acceptance remain incomplete. Three operations remain explicitly
unimplemented: `GET /resource-manager/capacity`,
`POST /provisioning/measure-capacity`, and `GET /metrics`.
The RM HTTP/JSON/SSE binding has local loopback coverage with server-owned
profile lookup and fake providers. The OpenAPI document is the contract
authority for these implemented operations and preserves the future schemas
without implying runtime support. Counts from older full-suite runs are
historical and are not a Phase 0 exit criterion.
The async scheduler and transport-neutral RM accompany queue/result primitives; see
[scheduler boundaries](queue-scheduler.md) and [RM boundaries](resource-manager-core.md).
The [HTTP boundary](resource-manager-http.md) requires trusted shared content
storage and is not a general network upload API or production E2E evidence.
Explicit scheduler stop terminalizes all unfinished work, including publication;
crash recovery instead preserves staged unacknowledged handoffs without rerunning
the provider. Same-session lost acknowledgments replay immutable dispatch bytes,
context and attempt key. New scheduler starts replace local and RM session fences.

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
three-model load/infer/cancel/unload/switch matrix required subsequent verification.

## Candidate compatibility progress (2026-09-16)

The later selected-artifact, hash-locked candidate image built and passed small
offline inference for all three models, including local GECToR verb vocabulary.
The candidate matrix and exact reproducible commands are recorded in
[compatibility spike](compatibility-spike.md). This is not acceptance of production
pins, cancellation behavior, maximum request limits, measured capacity or the
production image. The expanded RM-mediated lifecycle experiment subsequently
passed small/configured upper fixtures, active cancellation-result fencing,
stale-session rejection and child/NVML cleanup for all three models. Its p=1
profiles are synthetic, not measurements. SmolLM uses normal Ollama templating
with a fixed candidate context of 512; the lack of an independent local tokenizer
means no-truncation proof remains open. No devcontainer rebuild blocker has been
established.

## Durable profile registry precursor

The [profile store](capacity-profiles.md) now persists immutable caller-supplied
evidence with WAL/FULL durability, strict corruption/schema rejection and
exact-identity lookup. Drafts cannot be selected for runtime. Measured records
must contain successful baseline/warmup/wave evidence consistent with the claimed
throughput-optimal concurrency, but these checks do not prove actual GPU
execution. The candidate harness's synthetic profiles are never registered.
Benchmark execution and measured capacity provisioning/deployment integration
remain open. Server-owned exact profile lookup, bootstrap identity/hash
preflight, and adapter binding are implemented; the profile-validation HTTP
check is implemented and read-only, but it validates
caller-attested evidence against server-owned exact identity; it does not
measure capacity or make a draft selectable. Latest local verification counts
in this document are historical and are not repeated by this bounded contract
task.

## Contract authority and invariant review map

`docs/openapi.yaml` is the wire-contract authority. The implementation map is:

* scheduler routes and projection serializer: `services/llm/queue/http.py` and
  `services/llm/queue/store.py` (`_projection`); transition invariants:
  `services/llm/queue/contracts.py` and `services/llm/queue/transition_table.md`;
* Resource Manager JSON/SSE serializers and route set:
  `services/llm/resource_manager/http.py` (`_capacity_wire`, `_event_wire`);
* artifact and exact-profile validation routes:
  `services/llm/provisioning/http.py`;
  provisioning contracts, configured maximums, and artifact manifest rules:
  `services/llm/provisioning/contracts.py`, `services/llm/provisioning/artifacts.py`,
  `services/llm/bootstrap/config.py`, and `services/llm/bootstrap/bindings.py`;
* health serializers and routes: `services/llm/health.py`;
* durable profile schema and exact measured-row lookup:
  `services/llm/resource_manager/profiles.py` and
  `services/llm/resource_manager/contracts.py`;
* typed lifecycle, fencing, timing, nullable failure, and capacity invariants:
  `services/llm/resource_manager/protocol.py` and `core.py`.

Focused contract evidence is maintained by
`tests/unit/test_openapi_validation.py`, `tests/unit/test_contracts.py`,
`tests/unit/test_scheduler_operations.py`, `tests/unit/test_profile_contracts.py`,
`tests/unit/test_capacity_measurement.py`, `tests/unit/test_bootstrap_bindings.py`,
the RM/scheduler/provisioning/health loopback suites, and the Phase 0 contract
acceptance suite. This map reviews
body shapes, required idempotency keys, status outcomes, SSE nullability and
cursor identity, profile exactness, and error envelopes; it does not waive the
later production acceptance gates.

## Phase 0 closure review

The eight section-11 decisions above have an explicit owner and disposition:
operations owns retention and model-volume/registry policy; platform owns the
trusted async transport; application owns function execution and polling;
queue workers own durable result handoff; provisioning owns exact artifact and
benchmark evidence; and operations owns compatibility/pin evidence. Decisions
are locally closed for this contract phase, while retention backup policy,
compatibility pins, measured inputs, and production deployment remain required
later acceptance prerequisites. No item is waived by this review.

Phase 0 verification passed **145 task-related tests** with warnings treated as
errors; its **10-test** acceptance module passed two additional runs. See the
[completion record](queue-scheduler-resource-manager-plan-partial_completed.md#phase-0-complete--contracts-and-validated-bindings)
for scope and the proof inventory for exact commands. This is contract/local
integration evidence, not measured capacity or production deployment proof.

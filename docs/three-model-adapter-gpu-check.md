# Three-model adapter GPU check

This is a bounded offline candidate check, not a benchmark. It runs one
parent-owned Linux GPU proof and one `ResourceManager` through the sequence
`SmolLM -> CoEdIT -> GECToR -> SmolLM`. Every transition is a real RM
replacement, so cleanup failure prevents the next load. The capacity profile
is explicitly `UNMEASURED`; the output contains identities and fence results,
not prompts, completions, process tables, or throughput claims. It also emits
six bounded device-wide memory point observations; these are not peak memory or
capacity measurements.

The image expects the exact selected artifact volume at `/opt/llm/models` and
the canonical combined manifest at `/opt/llm/manifest.json`. It is entered as:

```text
/opt/venv/bin/python /opt/llm/three_model_adapter_check.py
```

Build the existing adapter image offline (`--network none`) and run the named
check:

```bash
docker build --network none --progress=plain --file tools/compatibility/Dockerfile.adapter \
  --tag llm-compatibility-adapter:candidate .
docker run --rm --name llm-three-model-adapter-check --init --network none --pid=host \
  --gpus '"device=GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963"' \
  --entrypoint /opt/venv/bin/python llm-compatibility-adapter:candidate \
  /opt/llm/three_model_adapter_check.py --models-root /opt/llm/models \
  --manifest /opt/llm/manifest.json --target-gpu-uuid GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963 \
   --port 11434 --host-pid-namespace
```

`--host-pid-namespace` is required because the command uses `--pid=host`; do
not add a procfs bind. The selected port must be unused on the host.

The harness constructs the production `BootstrapConfig` from the exact
provisioned manifest and uses `OwnedOllama` for the private, loopback-only
daemon. The daemon and all three providers receive the same typed GPU proof;
the proof is captured before any child is started. The pinned daemon identity
is `ollama:0.11.6`, with `/usr/bin/ollama`, the same private home, and the
same port used by the SmolLM provider. Identity-fenced cleanup therefore
targets only this owned process group and leaves foreign GPU processes alone.
Its cancellation check fences publication at execution entry;
it does not claim to interrupt an active GPU kernel. Temporary artifacts are
retained only when owned provider, daemon, or GPU cleanup remains uncertain;
positively cleaned failed and successful runs delete them. No completeness
beyond these checks is claimed. Profiles remain explicitly `UNMEASURED`:
this check does not call `prepare_bindings`, does not consume measured rows,
and is not evidence of capacity or an approved production image.

## Memory observation boundary

`LinuxGPUProof.memory()` reads NVML's whole-device total, used and free bytes
off the event loop. Each immutable observation carries the captured supervisor,
exact GPU UUID, and monotonic start/end nanoseconds covering both identity
fences. The physical device, UUID and MIG state are checked before and after
the read. Missing telemetry, invalid integer ranges, or lost identity fail
closed; ownership APIs do not otherwise require the memory API.

The candidate emits only `label`, `sequence_index`, `start_ns`, `end_ns`,
`total_bytes`, `used_bytes`, and `free_bytes` per point, with the common UUID at
top level. Labels are `baseline`, `SmolLM`, `CoEdIT`, `GECToR`, `SmolLM`,
`final_cleanup`, indexed 0–5. Baseline precedes all children; the four model
points follow inference/residency checks; the final point follows owned daemon
shutdown and successful GPU cleanup. UUID and total bytes must remain stable.
Telemetry failure prevents a passing result and still runs owned cleanup.

These observations include foreign allocations and may leave reserved memory
unaccounted for (`used + free <= total`). They neither attribute allocations to
a model nor sample during inference. Their maximum is not an execution peak;
differences are not per-request incremental VRAM. Final used memory need not
equal baseline: cleanup is proved by ownership, not by total device usage.
No profile, reserve-derived `N`, or optimal parallelism is generated.

Latest offline candidate passed with six points, stable total **12,878,610,432
bytes**, four valid responses, three switches, and cancellation/stale-result
fencing. The cancellation adds a terminal event without a published result; it
is not a fifth successful response. Owned container
`llm-three-model-memory-check` was verified absent. Candidate RepoDigest:
`llm-compatibility-adapter@sha256:f53c0463a2c38de4a48fa139fee740214077709fa152455ccc840413763562a2`.
Focused memory/proof/harness/supervisor verification passed **70 tests** with
warnings as errors. Profiles remain **UNMEASURED**.

# GECToR adapter GPU check

This is a bounded, offline candidate check. It uses the installed `GECToRProvider`
and one ResourceManager-owned worker on GPU UUID
`GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963`. Its fixed bucket is
`gector:p1:tokens128:keep0:min0:iterations1:batch1:float32`; capacity is reported
as **UNMEASURED**, not as a performance or concurrency claim.

## Commands

Run from the repository root. The adapter image inherits the selected artifacts
and manifest at `/opt/llm/models` and `/opt/llm/manifest.json` from the prepared
candidate base; no repository-root `models` bind mount is required.
The command IDs are `gector-normal` and `gector-injected`.

```sh
docker build --network none --progress=plain --file tools/compatibility/Dockerfile.adapter --tag llm-compatibility-adapter:candidate .
```

```sh
docker run --rm --name llm-gector-adapter-check --init --network none --pid=host --gpus '"device=GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963"' \
  --entrypoint /opt/venv/bin/python llm-compatibility-adapter:candidate /opt/llm/gector_adapter_check.py \
  --models-root /opt/llm/models --manifest /opt/llm/manifest.json --target-gpu-uuid GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963 --host-pid-namespace
```

Expected status: `passed-candidate`, with `successful_count=2`,
`rejected_count=1`, `cleanup=true`, and `stale_rejected=true`.

The injected process-loss command ID is `gector-injected`; use the same command
with `--inject-process-failure`:

```sh
docker run --rm --name llm-gector-adapter-failure-check --init --network none --pid=host --gpus '"device=GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963"' \
  --entrypoint /opt/venv/bin/python llm-compatibility-adapter:candidate /opt/llm/gector_adapter_check.py \
  --models-root /opt/llm/models --manifest /opt/llm/manifest.json --target-gpu-uuid GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963 --host-pid-namespace --inject-process-failure
```

Expected status is
`passed-candidate-failure-cleanup`, with `failure_count=1` and `cleanup=true`.
The wrapper kills only the installed worker's identity-fenced process group;
the failure must contain no result.

The harness captures GPU ownership before worker spawn, compares every selected
file by normalized path, size, and SHA-256 against the provisioned manifest,
and retains temporary artifacts if cleanup cannot be positively proved. The
runtime identity includes `gector=1.2.0`, Torch, Transformers, Tokenizers,
Safetensors, and the packaged CUDA suffix. It does not measure throughput,
VRAM, latency, or production capacity.

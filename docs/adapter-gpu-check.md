# Offline SmolLM GPU adapter check

This is a bounded candidate check of the actual
`services.llm.providers.SmolLMProvider` and `LinuxGPUProof`. It is **not** a
measured capacity profile: the profile identity is
`unmeasured-adapter-check`, and `SampleMetadata` is deliberately dummy data.
No profile or provenance file is written.

The check requires one physically exposed, non-MIG GPU and its exact UUID. It
fails closed for a missing NVML API, unknown process namespace, foreign GPU
work, non-empty baseline, artifact mismatch, unexpected Ollama version, or
anything it cannot prove. It starts one supervisor-owned daemon; the provider
never starts a daemon. The daemon uses a private temporary model store and is
stopped only after ResourceManager cleanup has proved that its runner is gone.

## Prepare dependencies (outside the image build)

Use the repository environment with its declared aiohttp version range already
installed and a clean, networked preparation directory. Generate an exact lock
from the installed aiohttp version and resolved binary dependencies:

```sh
.venv/bin/python tools/compatibility/prepare_adapter.py \
  --wheelhouse .compatibility/adapter-deps/wheelhouse \
  --lock .compatibility/adapter-deps/requirements.lock
```

The preparation tool writes the deterministic minimal wheel lock; do not copy
manual `pip hash` output. Do not run downloads from the Docker build.

## Build and run

Build from the repository root, using the already prepared candidate image and
the supplied artifact tree. The GPU selection must be the same exact physical
UUID supplied to the harness:

```sh
docker build --network none --progress=plain \
  --build-arg BASE_IMAGE=llm-compatibility-spike:candidate \
  -f tools/compatibility/Dockerfile.adapter -t llm-compatibility-spike:adapter .

docker run --rm --init --network none --gpus '"device=GPU-UUID"' \
  --mount type=bind,src=/proc,dst=/host/proc,readonly \
  llm-compatibility-spike:adapter \
  --models-root /opt/llm/models \
  --manifest /opt/llm/manifest.json \
  --target-gpu-uuid GPU-UUID --port 11434 --host-pid-namespace
```

The launcher reads `/host/proc/self/stat`, emits one bounded handshake, and
execs `/usr/bin/ollama`; that host PID is the only supervisor identity passed to
`LinuxGPUProof.capture`. Results contain only the fixed, small evidence schema:
version, completion booleans, supervisor host PID, runner count, and cleanup
status. Generated text and full NVML output are never emitted.

`--host-pid-namespace` is explicit operator attestation that Docker's read-only
`/proc` bind is the PID namespace used by NVML. It does not require (and this
command deliberately does not use) `--pid=host`. The handshake PID is verified
in that procfs before capture. The daemon preserves CUDA/NVIDIA visibility,
driver capability, loader, home, and temporary-path environment, forces
loopback, and ignores proxies.

`--init` and `--network none` are required fences. `ownedgroupgone` proves only
the saved Ollama process group is gone, not arbitrary descendants that escaped
it with a new session. `docker rm -f` after a non-`--rm` diagnostic run is the
final all-container boundary; verify the owned container is absent.

## Verification status — 2026-09-17

- Local harness suite: **9 tests passed**; full backend discovery: **294 tests
  passed in 31.593 seconds**. Compilation passed. No unraisable subprocess
  diagnostics were reported; the full suite emitted asyncio slow-task messages.
- The separate aiohttp **3.14.3** dependency layer built successfully using
  `--network none`, a local wheelhouse and exact hash lock. The Dockerfile-specific
  ignore file excludes unrelated artifacts and repository state from its context.
- The actual GPU run used UUID
  `GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963`, networking disabled and the documented
  read-only procfs bind. It failed before model readiness with
  `Ollama launcher procfs_missing` while opening `/host/proc/self/stat`.
- The tester removed the owned `llm-compatibility-adapter-check` container and
  verified absence. Runtime inference, GPU residency, and baseline restoration
  remain unproved. Earlier candidate harness results do not substitute for this
  actual-provider check.

**Operator unblock required:** expose readable authoritative host procfs in the
Docker runtime, aligned with the PID namespace used by NVML. The observed failure
does not establish whether a devcontainer rebuild is needed. Do not bypass the
attestation or substitute container-local PIDs merely to make readiness pass.

# Offline SmolLM GPU adapter check

This is a bounded candidate check of the actual
`services.llm.providers.SmolLMProvider` and `LinuxGPUProof`. It is **not** a
measured capacity profile: the profile identity is
`unmeasured-adapter-check`, and `SampleMetadata` is deliberately dummy data.
No profile or provenance file is written.

The check requires one physically exposed, non-MIG GPU and its exact UUID. It
fails for a missing NVML API, invalid process-namespace attestation, artifact
mismatch, unexpected Ollama version, or inability to prove its own runner.
Non-empty compute/graphics baselines and changing unrelated or unknown workloads
are accepted and represented only by bounded counts. Only a positively proved
strict descendant of the captured supervisor is service-owned; other NVML PIDs
are never controlled. It starts one supervisor-owned daemon; the provider
never starts a daemon. The daemon uses a private temporary model store and is
stopped after model-switch cleanup has proved that no model remains in private
Ollama. Ollama may retain a reusable owned GPU process; that does not block a
model switch. Final shutdown separately proves the fenced owned daemon process
group is gone. This does not prove uncontended throughput, VRAM availability,
measured capacity, or production E2E.

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

docker run --rm --init --network none --pid=host --gpus '"device=GPU-UUID"' \
  llm-compatibility-spike:adapter \
  --models-root /opt/llm/models \
  --manifest /opt/llm/manifest.json \
  --target-gpu-uuid GPU-UUID --port 11434 --host-pid-namespace
```

The launcher reads `/proc/self/stat`, emits one bounded handshake, and
execs `/usr/bin/ollama`; that host PID is the only supervisor identity passed to
`LinuxGPUProof.capture`. Results contain only the fixed, small evidence schema:
version, completion booleans, supervisor host PID, runner count, and cleanup
status. Generated text and full NVML output are never emitted.

`--host-pid-namespace` remains explicit operator attestation that the
container's `/proc` is the PID namespace used by NVML. `--pid=host` is required:
Docker Desktop/WSL2 can expose a broken `/host/proc/self` magic symlink through a
proc bind, while a host-PID container's own `/proc/self/stat` is readable. This
is deliberate read-only process observation across Docker Desktop's security
boundary; no extra capabilities or mounts are granted, and cleanup remains
restricted to the owned Ollama process group. The handshake PID is verified in
that procfs before capture. The daemon preserves CUDA/NVIDIA visibility,
driver capability, loader, home, and temporary-path environment, forces
loopback, and ignores proxies.

The flag is an operator claim, not proof that Docker was invoked with
`--pid=host`; it cannot be self-attested by the container. Real readiness still
fails closed through `LinuxGPUProof`'s NVML process lists and procfs supervisor
identity/start-time matching in the same namespace. No portable, privilege-free
namespace invariant is assumed here because Docker Desktop/WSL2 may present
different procfs and namespace arrangements.

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
- After the devcontainer configuration gained
  `source=/proc,target=/host/proc,type=bind,readonly`, one bounded retry used the
  same documented runtime arguments and existing adapter image
  `sha256:30174aecfa41c99ff94228826c93f631618e2370c892426847d06f7ecec2c485`.
  It failed non-zero in **5.685 seconds** with `Ollama launcher procfs_missing`
  and `invalid Ollama launcher handshake`, before readiness. The tester removed
  `llm-compatibility-adapter-check-retry` and verified absence. Testing stopped
  at this environment block; no new unit-test results are claimed.

The Docker Desktop/WSL2 procfs failure is fixed: the harness now uses one
authoritative `/proc` root under the documented `--pid=host` runtime. The focused
adapter/GPU-proof suites passed **21 tests in 0.148 seconds**, including launcher
syntax and PID-reuse-safe host process-group cleanup; compilation passed. A fresh
network-disabled image was built as
`llm-compatibility-spike@sha256:b9e6ae2aa0440e3f4d4dd9b7f33ad9c06e914538dbca465493c44d88ddf36954`.
The actual post-fix run advanced beyond the old `procfs_missing` failure and
exposed the old dedicated-GPU assumption. The proof now identifies ownership
positively: only stable strict descendants of the captured Ollama supervisor are
service runners; readable foreign workloads may continuously enter or leave.
Focused GPU-proof, adapter, provider and ResourceManager verification passes
**74 tests** with warnings as errors, and compilation passes. Network-disabled
image `sha256:8bc37a2e60cdbb14c4ed067eeb602b58540af7214bc300dfc257d62ff5293c53`
built successfully. Its latest actual run failed closed before readiness with
`GPU baseline ownership could not be observed`, meaning a current NVML PID could
not be classified safely from procfs. The owned container was removed and
verified absent. Do not bypass host-PID attestation or treat unreadable ownership
as foreign. Inference, residency and service-owned cleanup remain unproved; this
check does not establish capacity under contention.

A subsequent bounded fix added real elapsed-time backoff for transient
NVML-after-procfs teardown lag, complete classification of confirmation
snapshots, and stable foreign-ancestry revalidation. The focused GPU-proof,
adapter, provider and ResourceManager suites now pass **83 tests** with warnings
as errors; compilation passes. Network-disabled image
`sha256:c1ddec1ae3ab6cb43cf5081423c111da493281aebb9be65c5a101651aacd9a10`
built successfully, but its actual run still failed closed at baseline ownership.
A minimal diagnostic found compute empty and graphics empty before Ollama; while
Ollama ran, graphics consistently reported two PIDs, one readable in authoritative
Linux procfs and one absent throughout ten 200 ms samples. Repeated absence is
not authoritative foreign-process correlation, so the proof deliberately does
not exclude that PID. The owned diagnostic and adapter containers were removed
and verified absent. This Docker Desktop/WSL2 NVML-to-Linux-procfs mismatch now
previously appeared to require an authoritative platform-specific correlation
mechanism. That requirement exceeded G4.1: the service needs positive proof of
its own runner, not exhaustive proof of every unrelated GPU client.

The proof now ignores unconnected non-supervisor NVML PIDs while retaining strict
positive descendant ownership, stable two-sample residency, supervisor fencing,
and no signaling of NVML PIDs. Focused GPU-proof, adapter, provider and
ResourceManager verification passes **88 tests** with warnings as errors, and
compilation passes. A new actual adapter run is still required to prove inference,
residency and service-owned cleanup. G4.2 capacity under contention remains
unproved.

The goal-focused run now passes on the target GPU with Ollama **0.11.6**. Both
bounded completion cases succeeded, readiness positively proved one runner under
the captured supervisor, model-switch cleanup proved the old model absent, the
stale session was rejected, and final shutdown proved the identity-fenced owned
daemon group gone. The baseline contained two unrelated/unknown NVML PIDs; they
were counted but neither classified exhaustively nor controlled. The named
container was removed and verified absent. Focused GPU-proof, adapter, provider
and ResourceManager verification passes **94 tests** with warnings as errors.
This is candidate G4.1 evidence only; G4.2 contention capacity remains unproved.

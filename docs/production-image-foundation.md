# Production image foundation (candidate)

This slice assembles an offline, non-root image foundation. It is **not** a
production deployment and does not claim OS-package reproducibility: apt
packages are currently installed from the CUDA image's configured Ubuntu
repositories, without a snapshot or approved package pin set.

## Deterministic inputs

Input assembly and the Python installation layer are offline. The Docker build
still needs network access for the unpinned Ubuntu `apt` repositories in the
foundation image. Reviewed inputs are retained at the exact paths below: the
two lock files and wheelhouses, plus
`.compatibility/context/ollama-linux-amd64.tgz`. Do not substitute source JSON,
model caches, or a downloaded wheelhouse.

```sh
.venv/bin/python -m tools.deployment.prepare_inputs \
  --lock .compatibility/context/requirements.lock \
  --lock .compatibility/adapter-deps/requirements.lock \
  --wheelhouse .compatibility/context/wheelhouse \
  --wheelhouse .compatibility/adapter-deps/wheelhouse \
  --ollama .compatibility/context/ollama-linux-amd64.tgz \
  --output deploy/docker/.inputs
docker build --tag llm-provider-foundation:deployment-foundation -f deploy/docker/Dockerfile .
docker run --rm --network none llm-provider-foundation:deployment-foundation --help
# Current entrypoint accepts serve CONFIG with required state/result arguments.
# Do not deploy this candidate: the security/compatibility gates below remain open.
```

The assembler fails closed on lock conflicts, missing wheels, wheel hash
mismatches, and an unverified Ollama archive. It stages only the locked
wheel files, the verified archive, the generated union lock, and minimal
provenance. Models, compatibility JSON, source dumps, pip cache, and the
original wheelhouses are not copied.

The CUDA base is pinned to runtime `12.8.1-ubuntu24.04` digest
`ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867`.
The retained Ollama archive is hash-verified, extracted, then checked for an
executable `bin/ollama` and directory `lib/ollama`; they remain in their
corresponding `/usr/local` paths.

Replacement uses same-parent renames with rollback if installation fails before
commit. This is not crash-atomic directory exchange or concurrent-writer support.
A post-commit backup-removal failure explicitly reports that output was committed
and retains the uniquely owned backup; it does not imply the old output was
restored. Staged bytes are rehashed before installation. Generated `.inputs/`
is gitignored and contains no model artifacts.

## Verified closeout (2026-09-17)

The following focused command passed **14 tests in 0.012s**, including real ZIP
metadata (with nested vendored metadata), dotted/local versions and build tags,
canonical lock identities, staged corruption, rollback and path-safety checks:

```sh
.venv/bin/python -W error -m unittest -v tests.unit.test_deployment_inputs tests.unit.test_deployment_image
```

Targeted compilation and whitespace checks passed. Actual retained-input assembly
produced **54 locked wheels / 57 files**, with no models or source JSON. The union
lock SHA-256 is
`2216a61caa45a77c5c2c42e3b19de215758062cd9891b6ea3451764c5bc2b087`.
The tagged build passed in **185.0s**, producing image ID
`sha256:11216cfed17487c5ff12a20fe5ee815dd158d389236bcd50c78b51b6e9b87c25`.

Network-disabled help exited 0; serving exited 78. A separate offline import and
filesystem check verified non-root execution, root-owned code/venv not writable
by the application user, writable LLM state/results, Ollama-owned home/models,
and absence of staged archives, wheelhouse, model assets and caches at the checked
final-image paths. These are filesystem checks, not an exhaustive image-layer
or supply-chain audit. All three named test-owned containers were verified absent.
No GPU/provider serving was tested in this foundation image.

## Distinct-user broker continuation (2026-09-17)

The earlier exit-78 smoke result above describes the historical foundation,
not the current entrypoint. The current CLI dispatches to `BootstrapRuntime`;
measured-profile admission remains mandatory. The fixed argument-free SUID
gateway starts a root broker, which starts Ollama as the separate service user.
Cross-UID listener attestation temporarily matches **both** filesystem UID and
GID on the probing thread and restores both. Speculative dumpability/ptracer
changes were removed; no broad ptrace capability was added.

Final focused **unit/local integration** verification passed **58 tests in 20.367s**:

```sh
.venv/bin/python -W error -m unittest -v tests.unit.test_image_broker_boundary_harness tests.unit.test_ollama_broker tests.unit.test_ollama_broker_client tests.unit.test_bootstrap_supervisor tests.unit.test_deployment_image tests.integration.test_bootstrap_http tests.unit.test_bootstrap_runtime
```

Stateful regressions verify exact filesystem-credential restoration ordering
after success and denied access, plus fail-closed transition/restoration failure.
The final SUID assertion correction was verified by these real
**e2e candidate boundary** checks (the retained file is a standalone script,
not a unittest suite):

```sh
docker build --progress=plain --tag llm-provider-foundation:phase1-broker -f deploy/docker/Dockerfile .
docker run --rm -i --network none --name llm-phase1-broker-boundary-cpu --entrypoint /opt/venv/bin/python llm-provider-foundation:phase1-broker - < tests/integration/test_image_broker_boundary.py
docker run --rm -i --network none --gpus device=GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963 --name llm-phase1-broker-boundary-gpu --entrypoint /opt/venv/bin/python llm-provider-foundation:phase1-broker - < tests/integration/test_image_broker_boundary.py
```

Tested image manifest:
`sha256:58c319fe8d5362d1982f3a3c8cb9fec4e295faecfddee1574c3be8d4b593be3f`.
Both runs passed: app UID/GID 1001, broker UID tuple `(1001,0,0,0)` with
GID tuple `(1001,1001,1001,1001)`, daemon UID/GID fields all 1002,
Ollama `0.11.6`, private loopback listener, duplicate-broker rejection,
hostile caller environment/cwd isolation, checked root-owned paths, and
EOF cleanup with broker/daemon disappearance. The harness owns clients and
closes them in `finally`; the runner owns and removes the named containers.
No foreign processes were signalled. The GPU run proves startup with the
selected device exposed, **not** an NVML query, model inference or residency.

## Deployment gates — do not promote this candidate

- The gateway checks final interpreter/broker files but not the complete
  trusted ancestor/import tree. Passing ownership checks on this image is not
  adversarial proof that substituted writable imports are rejected. Privileged
  import-path hardening and mutation/refusal tests remain required.
- The parent-death signal protects the exec'd daemon **leader**, not arbitrary
  detached/reparented runners. Unit evidence includes test-owned subreaping and
  leader/listener disappearance. Broker-loss descendant containment and durable
  replacement fencing remain unproved; EOF cleanup does not establish them.
- The Compose skeleton's host-PID topology, immutable model/config mounts and
  persistent volume restart behavior were not exercised by these checks.
- Section 7 compatibility, reviewed version approval, actual measured profiles,
  full runtime/model execution and reproducible OS-package pins remain open.

These are release-blocking gaps, not a production-readiness claim. G0/G2.2/
G4.1/G4.3 remain **partial**. Existing direct injected-command/same-UID behavior
and runtime admission fences are not replaced by this bounded broker proof.

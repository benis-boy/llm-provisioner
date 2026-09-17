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
# Expected exit 78: serving is deliberately unavailable.
docker run --rm --network none llm-provider-foundation:deployment-foundation serve
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

## Deliberate boundary

The image creates separate `llm` and `ollama` users and owned volume layout,
but does not pretend to supervise the second user. The entrypoint therefore
allows only the existing bootstrap `--help` smoke check and refuses serving
with a clear exit. It does not weaken `BootstrapRuntime`, profile gates, or
the proven same-UID candidate. Implementing the distinct-user launch contract,
config injection, and qualifying runtime checks is a later slice.

# Compatibility image build and verification

This workflow produces an **equivalent current-source rebuild**, not an exact
historical manifest recreation. Docker build metadata and apt state are not
retained well enough to make the latter claim. It makes no capacity-proof claim.

## Prerequisites and ownership

Install Docker with BuildKit/buildx and a Linux/amd64 builder. The workflow
inspects and rejects a base that is not exactly Linux/amd64, and binds that
platform to the identity. Prepare the base
context with `tools/compatibility/prepare.py`; that generated directory owns
`provenance.json`, `manifest.json`, the selected models, Ollama archive, and
hash-locked base wheelhouse. Prepare adapter dependencies with
`tools/compatibility/prepare_adapter.py`; it owns
`.compatibility/adapter-deps/requirements.lock` and `wheelhouse/`. Do not edit
generated inputs while building. The base preparation may use the network; the
image builds and runtime probe must not.

## Copyable workflow

```sh
.venv/bin/python tools/compatibility/prepare.py --output .compatibility/context \
  --smollm-root /path/SmolLM --coedit-root /path/CoEdIT --gector-root /path/GECToR --download
.venv/bin/python tools/compatibility/prepare_adapter.py \
  --wheelhouse .compatibility/adapter-deps/wheelhouse \
  --lock .compatibility/adapter-deps/requirements.lock

# The base image must already be loaded locally. The workflow resolves a digest
# when available, otherwise creates and removes only a unique owned tag after
# re-inspecting it immediately before build.
.venv/bin/python tools/compatibility/image_build.py identity --root . \
  --base-image llm-compatibility-spike:candidate \
  --base-context .compatibility/context --output .compatibility/adapter.identity.json
.venv/bin/python tools/compatibility/image_build.py build-adapter --root . \
  --base-image llm-compatibility-spike:candidate \
  --base-context .compatibility/context --tag llm-compatibility-adapter:current \
  --identity .compatibility/adapter.identity.json
.venv/bin/python tools/compatibility/image_build.py verify --root . \
  --base-image llm-compatibility-spike:candidate \
  --base-context .compatibility/context --image llm-compatibility-adapter:current \
  --identity .compatibility/adapter.identity.json
```

If an image is missing, repeat the identity/build/verify steps. The identity is
atomic and idempotent, so an equivalent rebuild can replace a missing tag.
Missing locks, wheel hashes, base provenance/artifact contracts, source files,
labels, entrypoint, or package probe results fail closed. Locks, the wheelhouse,
and every wheel must be regular non-symlink paths under the selected root.
No GPU is required for verification. The commands use explicit platform, network
isolation, disabled provenance/SBOM, and local loading.

Verification is current-source verification: pass `--root`, `--base-image`, and
the same optional `--base-context` so source, lock, and base identity are
recomputed before the image is checked. It proves an equivalent rebuild from
today's inputs, not byte-identical images or exact historical recreation.

The adapter image preserves the complete `tools.compatibility` package beneath
`/opt/llm/tools/compatibility`, including `__init__.py`. The canonical
measurement entrypoint therefore resolves its
`tools.compatibility.prepare_measurement` import inside the image. The required
executable path remains `/opt/llm/measure_profiles.py` through an image-local
symlink to the package module (and the preparation helper is exposed the same
way), so there are no duplicate implementations and Docker arguments still
begin with `/opt/llm/measure_profiles.py`. Running that path with `--help` is a
useful import-contract smoke check before Phase 5 work.

For the build step, the workflow revalidates the admitted Dockerfile `COPY`
inputs and adapter dependency inputs, then copies only those regular files into
a temporary owned build context. BuildKit consumes that staged context rather
than the live repository; changes to the checkout after staging cannot alter the
image. The stage is removed in all outcomes. The identity inventory describes
those staged source bytes (and therefore the verified source bytes at staging),
not model assets or unrelated repository data.

# Deterministic measurement inputs

Phase 5 preparation owns a small bundle of requests, identities, strict
bootstrap configuration, and the selected artifact volume. It is not capacity
evidence and does not authorize a measured profile.

The retained CoEdIT and GECToR strings are stable request identities, not
character-count proofs of a 128-token maximum. The full measurement path must
validate each loaded adapter's no-truncation/tokenizer maximum before it admits
the request; failure rejects the measurement and cannot create a profile.

From the repository root:

```sh
.venv/bin/python tools/compatibility/prepare_measurement.py prepare \
  --output .compatibility/measurement-bundle --context .compatibility/context \
  --gpu-uuid GPU-EXACT_UUID --ollama-version 0.11.6 \
  --container-mount /opt/measurement
.venv/bin/python tools/compatibility/prepare_measurement.py verify --output .compatibility/measurement-bundle
```

Runtime identities and adapter identities may be supplied explicitly with
`--smollm-runtime`, `--coedit-runtime`, `--gector-runtime` and corresponding
`--*-adapter` options. Defaults are conservative identities, not proof.

For the exact-GPU SmolLM p=2 diagnostic, use the bounded transfer entry point
instead of composing Docker binds/copies manually. It creates unique owned
bundle, runtime, and request volumes; transfers with `docker cp` only (so the
relative `current` symlink is retained); verifies the pristine in-image bundle
before adding the separate request volume; aborts if compute work is already
present; captures bounded same-container facts without printing request/result
bodies; and removes/verifies only its own resources. It runs one diagnostic,
not the full matrix:

```sh
.venv/bin/python tools/compatibility/run_smollm_p2_diagnostic.py \
  --image IMAGE_REFERENCE --bundle .compatibility/measurement-bundle \
  --request path/to/exact-smollm-request.json \
  --gpu-uuid GPU-EXACT_UUID --ollama-version 0.11.6
```

Docker Desktop host bind mounts are not an acceptable substitute for this
workflow. For full-matrix work, use the all-or-nothing runner. It creates fresh
owned bundle, runtime, result, and export volumes, verifies the pristine bundle
in-image, performs two bounded foreign-compute checks, and uses `--network none`
and `--pid=host` with the exact GPU. There is no resume or partial promotion;
success requires all three selectors. Measurement, audit, reservation, and owned
Docker cleanup failures before commit export nothing:

```sh
.venv/bin/python tools/compatibility/run_measurement_matrix.py \
  --debug \
  --image IMAGE_REFERENCE --bundle .compatibility/measurement-bundle \
  --gpu-uuid GPU-EXACT_UUID --ollama-version 0.11.6 --ceiling 32 \
  --output .compatibility/profiles.sqlite
```

`--debug` is opt-in and does not change the stdout contract: stdout remains one
sanitized JSON result, while bounded JSONL trace records are written to stderr.
The inner process exclusively creates its trace in the fresh runtime volume;
the runner retrieves and revalidates it before removing owned Docker state. On
failure, it also publishes `<output>.debug.jsonl` without replacing an existing
sidecar. Trace transport is limited to 4 KiB per record, 20,000 records, and
8 MiB total, with a truncation record when a limit is reached. Trace failures
are best effort and cannot change the measurement result. The closed trace
schema excludes request content, model output, response bodies, exception text,
tracebacks, tokens, environment values, and filesystem paths; identifiers are
reported only as bounded classifications, counts, or short digests.

The runner reserves the host output before any Docker or GPU work and retains
that owner lock through export and cleanup. It allows 10,980 seconds (three
hours plus margin) and removes only its owned Docker state. Its fixed bounded
result is checked against the exact canonical matrix and all three model and
profile identities. After copy, the host reopens the SQLite database through
the readonly `ProfileStore` validator, rejects WAL sidecars or tampering, then
fsyncs, audits mode `0444`, and performs no-clobber promotion. The profile
installer similarly requires a successful `TRUNCATE` checkpoint with no WAL or
SHM residue before installation. Existing lock files are treated as live or
ambiguous ownership: the runner fails closed and never applies an automatic
stale-lock timeout or removes another owner’s lock. The operator emits exactly
one sanitized JSON status line on both success and failure; a zero exit status
is possible only after export audit, owned Docker cleanup, no-clobber promotion,
and lock release have all succeeded. The audited host copy is staged before
Docker cleanup but is not promoted until cleanup and reservation ownership are
proved. UUID-named Docker resources require positive owner-label proof; container
operations use verified immutable IDs, and volume metadata is rechecked before
use/removal. Export copying uses a fresh private staging directory with tracked
file/directory identities, so a proved partial copy is cleaned without blocking
a retry or deleting substituted foreign paths. Interruptions attempt remaining
owned teardown and host cleanup before propagating. These checks assume no
hostile concurrent same-UID/root filesystem or Docker-daemon mutation; Docker
volumes do not provide an atomic inspect-and-remove operation.
A postcommit durability or lock-release failure reports an incomplete
result with `db_retained: true`; it does not silently delete a committed audited
database. Inspect that explicit result and retained ownership evidence before
retrying, and never overwrite the destination or another owner's lock.
Therefore a real invocation cannot claim success with no output under this
contract.

For manual transfer/debugging only, use the lower-level immutable bundle
workflow with exactly one fresh writable runtime volume:

```sh
V=llm-measurement-$(date +%s)
docker volume create "$V"
C=llm-measurement-transfer-$(date +%s)
docker create --name "$C" \
  --mount "type=volume,src=$V,dst=/var/lib/llm-measurement" \
  --mount "type=bind,src=$PWD/.compatibility/measurement-bundle,dst=/opt/measurement,readonly" \
  /opt/llm/prepare_measurement.py verify --output /opt/measurement
docker start -a "$C"
docker rm "$C"
```

Run `verify_current` on the mounted `artifacts` directory before starting the
runtime. The diagnostic CLI maps to `requests.json`. Full measurement uses one
atomic bundle argument; the verifier binds `config.json`, `requests.json`, and
the exact `provenance.sha256` before any service is loaded. The profile DB is a
fresh caller-selected output (and is never part of bundle verification):

```sh
.venv/bin/python tools/compatibility/measure_profiles.py \
  --bundle /opt/measurement \
  --db /var/lib/llm-measurement/profiles.sqlite \
  --ollama-version 0.11.6 --ceiling 32
```

The bundle is mounted read-only at `/opt/measurement`; the single fresh writable
volume is mounted at `/var/lib/llm-measurement` (do not use nested mounts).
The generated config always uses `/var/lib/llm-measurement/ollama` for Ollama
state, `/usr/bin/ollama` for the image binary, and
`/var/lib/llm-measurement/profiles.sqlite` for the external profile database.
The full CLI `--db` must equal that generated external path and must remain
outside the bundle. The bundle's `config.json` must point `artifact_root` at the
mounted `/opt/measurement/artifacts`. Do not pass separate `--config`, `--requests`,
or `--provenance` arguments; they are rejected with `--bundle` so inputs cannot
be mixed. Preparation only makes exact inputs;
throughput, optimal parallelism, bounded buffering, and capacity evidence are
separate measurement results.

Preparation consumes the real retained `.compatibility/context` directly; do
not create an ad hoc strict staging directory. Its measurement inputs are only
`manifest.json` and the exact selected files under `models/`. The root may also
contain the explicit, non-measurement build outputs emitted by
`tools/compatibility/prepare.py` (`requirements-candidate.txt`, `Dockerfile`,
the compatibility harness files, `services/`, `wheelhouse/`, the pinned
archive/lock/provenance files, and `.compatibility-spike-owned`). Those outputs
are checked for symlinks but are ignored by bundle identity. Python-generated
`__pycache__/` directories and `.pyc` files anywhere under the context are
also ignored by bundle identity, after checking that they contain only regular
files and directories (no symlinks or special entries). Unknown entries,
extra model files, and symlinks anywhere else are rejected. Output is staged in a
temporary sibling and atomically selected. Repeating the same request is an
idempotent verified no-op; a changed request is rejected unless `--replace` is
explicit. The context manifest verifies the retained selected source files,
but its digest can legitimately differ from the provisioned volume manifest:
provisioning normalizes source roots to the runtime `models/<model>` layout.
The selected artifact-volume manifest returned by `verify_current` is the
authoritative `manifest_sha256` in configuration, request fingerprints, and
bundle identity. Provenance schema 2 separately retains the bounded
`source_context_manifest_sha256` and per-model `source_model_sha256` binding;
verification requires those model hashes to equal the selected runtime hashes.
It never treats distinct manifest digests as interchangeable. Verification
rechecks artifact and model hashes, request fingerprints, configuration,
provenance, and exact bundle identity.
Diagnostics print only the provenance digest, never fixture or model contents.

The image entrypoint is Python, so the transfer verifier is passed as the exact
script path shown above; do not wrap it in `sh -c`. The image contains that
script specifically for this verification step.

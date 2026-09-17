# Offline bootstrap preflight

`services.llm.bootstrap` is a bounded precursor to production bootstrap.  It
does **not** supervise a daemon, start children, load models, benchmark, probe
an HTTP endpoint, or provide an approved image.  A later Linux supervisor must
capture `LinuxGPUProof` and pass its typed `GPUProof` to this API before it
creates any provider children.

## Configuration

The operator-owned JSON file is at most 32 KiB.  It has schema `1`, rejects
duplicate, unknown, and missing fields, and requires all paths to be absolute.
All three model entries are mandatory.  Hashes are lowercase SHA-256 values;
model hashes are selected from the small, verified artifact manifest rather
than from an arbitrary model JSON document.

Small example (identities are placeholders, not approved production values):

```json
{
  "schema": 1,
  "gpu_uuid": "GPU-operator-selected",
  "artifact_root": "/srv/llm/artifacts",
  "manifest_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "profile_db": "/srv/llm/profiles.sqlite",
  "ollama_binary": "/usr/local/bin/ollama",
  "ollama_home": "/srv/llm/ollama-home",
  "ollama_port": 11434,
  "models": {
    "SmolLM": {"runtime_identity": "operator-approved-runtime", "adapter_identity": "operator-approved-adapter"},
    "CoEdIT": {"runtime_identity": "operator-approved-runtime", "adapter_identity": "operator-approved-adapter"},
    "GECToR": {"runtime_identity": "operator-approved-runtime", "adapter_identity": "operator-approved-adapter"}
  }
}
```

## Exact API

```python
from services.llm.bootstrap import load_config, observe_runtime_identities, prepare_bindings

config = load_config("/etc/llm/bootstrap.json")
observed = observe_runtime_identities(ollama_version_from_supervisor_handshake)
prepared = await prepare_bindings(config, captured_gpu_proof, observed)
try:
    bindings = prepared.bindings
finally:
    prepared.close()
```

`observe_runtime_identities` uses `importlib.metadata` for installed Python
runtime packages (`torch`, `transformers`, `tokenizers`, and `safetensors`, plus
`gector` for GECToR) without importing them. It combines their sorted versions with
the Python version; the Ollama version comes from the supervisor health handshake.
Version fragments are bounded to 64 ASCII characters, excluding whitespace,
control characters and identity delimiters. These
observations are not HTTP-submitted identities and this module does not import
Torch/CUDA or invent an identity. `prepare_bindings` verifies the selected current artifact
volume and exact selected model files, requires measured (never draft) p=1
profiles with a 20% safety reserve, and opens the existing profile database
read-only.  The initial profiles are SmolLM context 512, CoEdIT's fixed
`input128/output64/float16/beams1/nosample` bucket, and GECToR's fixed
`tokens128/float32/iterations1` bucket.  `N > 1` remains diagnostic evidence;
it is not confused with profile identity or used to synthesize capacity.

The returned mapping contains immutable `ModelBinding` values and real,
unloaded provider instances. Resolution is pinned to exactly context 512 for
SmolLM and the fixed bucket for the other models, so later larger measured rows
cannot be admitted accidentally.  This is an exact measured-profile admission:
the selected row must itself carry context 512 or the exact fixed bucket, p=1,
m=1, and the 20% reserve, including on every later resolution. A row measured at context 1024 never satisfies this
bootstrap, even though the general profile registry can select a larger context
for other callers.  `N > 1` samples are permitted as diagnostic evidence, but
are not admission capacity and do not alter the p=1 identity. `close()` releases the owned read-only
`ProfileStore`; all-or-nothing failures close it before propagating. Artifact
verification and hashing run in `asyncio.to_thread`; that thread produces no
owned resources. The thread-affine SQLite store is opened only afterward on the
event-loop thread. This is startup-only work, not request-path work, and
cancellation during hashing therefore cannot leak a store.

Artifact selection is checked before and after extracting the canonical selected
manifest and selected-file hashes.  A `current` change observed in that window
fails closed.  This is a preflight check, not a permanent filesystem snapshot:
the provision volume must be an immutable mounted volume while it is in use and
providers must revalidate their own inputs before load. Concurrent operator
mutation is not supported, and this module does not attempt a hostile-root
filesystem defense. Configuration and returned dataclasses are frozen only at
their own fields (the bindings mapping is read-only); this is shallow Python
immutability, not a promise that provider internals cannot later mutate.

Remaining production work is the single fenced supervisor: capture/recheck
Linux GPU ownership, perform the Ollama health handshake, start and fence
children, load/ready/unload providers, expose the HTTP server, and retain
cleanup ownership on cancellation or failure.

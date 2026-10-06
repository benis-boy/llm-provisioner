# Offline bootstrap preflight

`prepare_bindings` is the read-only preflight API: it does not start children or
measure capacity. `BootstrapRuntime` composes that API with parent GPU capture,
private Ollama supervision, ResourceManager, and one HTTP/health listener.
Neither API supplies an approved production image or measured capacity runner.

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

## Runtime composition boundary

`BootstrapRuntime` is the next composition slice.  It captures
`LinuxGPUProof` in the common Python parent, with explicit host-PID-namespace
attestation (`--host-pid-namespace`), before creating `OwnedOllama` or any
provider.  The attestation is intentionally not inferred or defaulted.  It
converts the captured Linux proof into the typed `GPUProof` required by the
supervisor and providers. It then performs
the exact state SQLite/free-space preflight, starts the one private daemon,
observes the installed runtime identities, performs the read-only artifact and
profile preflight, and exposes the existing ResourceManager and health routes
from one aiohttp listener.  The daemon monitor is private: daemon loss fences
ResourceManager admission and there is no independent daemon restart.

Stop fences admission and the active session first, stops the listener, joins
ResourceManager cleanup, closes HTTP/health, closes Ollama, and releases profiles.
Concurrent and repeatedly-cancelled stop calls share one cleanup task. Every
cleanup stage is attempted within one configured positive, at-most-300-second
total grace (60 seconds by default). Timed-out tasks remain retained until they
finish. A timeout or failure is recorded as unproved cleanup rather than claimed
as clean. Daemon normal exit and exceptional monitor loss both fence admission;
there is no daemon restart. The CLI also treats SIGINT/SIGTERM as a bounded
stop request.

The initial selected-file hashing is a full point-in-time check under the
immutable mounted-volume precondition. Health subsequently rechecks the
selected artifact, exact profile shape, runtime identities, GPU identity,
daemon ownership, and SQLite/free-space in bounded workers. It never rehashes
artifacts per request, and health probes do not load a provider, alter residency,
or create a session. Startup idle is intentionally unready while a session may
still be started through the normal ResourceManager API.

Focused runtime/RM/health/scheduler verification passes **141 tests** with warnings
as errors. The separate offline installed-adapter candidate now also exercises
the real `OwnedOllama` with the shared parent proof; it still uses explicitly
unmeasured profiles and does not start this composed HTTP service.

## ResourceManager provisioning-mode precursor

`services.llm.provisioning.rm_runner.provision_request` is the bounded offline
benchmark entry point. It accepts the actual `ResourceManager` owned by the
runtime and a server-owned `ModelBinding`, resolves the real adapter once, and
then uses only `start_session`, `submit`, `watch_progress`, `cancel_request`,
and `stop_session`. Thus admission, residency generation, provider lifecycle,
fencing, cleanup, and GPU timing remain ResourceManager authority rather than a
profiling-only substitute. The returned `ProvisioningEvidence` records the
session/submission identities, progress events, result, and optional completed
GPU timing; `measured_capacity_claim` is always false.

The entry point is bounded, offline, performs no download, rejects mismatched
model/session/request/generation identity, and shields cancellation cleanup.
Failure, backpressure, malformed output, and caller cancellation all attempt
request cancellation followed by session stop. A cleanup failure remains an
operator-visible failure; it is never reported as successful evidence. This is
an implementation precursor only: without GPU production evidence it does not
prove capacity and must not check the production-evidence checkbox.

This is still composition and local lifecycle evidence, not a production image
or qualifying deployment proof.  The artifact check is point-in-time and
requires the selected volume to remain immutable/read-only while providers use
it.  Unobserved daemon descendants that escape before observation require
external cgroup/container containment; procfs and pidfds are not a cgroup
guarantee.  The runtime does not measure capacity, synthesize parallelism, run
inference from health, download artifacts, or claim a production image.

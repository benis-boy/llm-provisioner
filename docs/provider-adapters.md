# Provider adapters

The Phase 4 SmolLM adapter is intentionally a client of the fixed supervisor-owned
loopback Ollama daemon. It never starts Ollama, downloads a model, benchmarks, or
selects a capacity profile. Startup verifies the operator-selected content-addressed
volume, including the selected GGUF's configured SHA-256 identity, and the exact
server-bound manifest, model, GPU, runtime, adapter, context,
and concurrency identities. Missing proof fails closed.

Ollama local import uses a deterministic digest-derived name and a temporary,
quoted Modelfile whose `FROM` is the absolute selected GGUF path. The selected
manifest digest also binds its required `Modelfile`; the configured GGUF digest
binds the runtime weights. The adapter hashes selected files off the event loop,
never starts `ollama serve`, and only
uses the configured IPv4 loopback endpoint without redirects. Generation is a
raw framed request with explicit `num_ctx=512`, `num_predict=64`, temperature
zero, and keep-alive settings; ordinary Ollama metadata is tolerated while the
required completion and prompt-count proof is strict.

The local `ollama create` subprocess has a bounded aggregate stdout/stderr
drain, timeout, and process-group reap. A failed import remains adapter-owned
until Resource Manager cleanup has made `/api/ps` model absence available. It is
not safe to execute before
readiness, and direct execution additionally enforces the configured measured
parallelism; Resource Manager remains the normal admission authority.

Readiness preloads the model and requires `/api/ps` to show exactly the canonical
`<digest-name>:latest` model,
fully resident in VRAM, plus concrete Linux shared-GPU residency evidence: the configured
supervisor and non-empty, unique runner identities must be exact descendants;
missing or unsafe ownership evidence fails closed. Model-switch cleanup drains
owned work, unloads, and verifies that no model remains resident in private
Ollama; reusable GPU processes may remain. Final adapter shutdown separately
fences and terminates the owned daemon process group. Foreign GPU processes are
tolerated and never controlled. GPU timing remains incomplete for
Resource Manager accounting; authoritative host procfs and NVML are required by
the real proof, while production image/bootstrap wiring remains an explicit
deployment seam. Only the explicitly unmeasured 512-token
profile is accepted. Candidate aggregate identities and synthetic profiles are
not production profile identities; the adapter does not claim arbitrary contexts
or registry-origin provenance.

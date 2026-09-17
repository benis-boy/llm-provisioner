# Provider adapters

## CoEdIT

CoEdIT uses one fresh, process-group-owned offline Python worker per load. The
parent imports no Torch/CUDA and requires a captured Linux GPU proof for the
parent supervisor before the child is launched. The strict request is
`{"instruction": string, "texts": [string]}` with one configured native batch
item, and the aligned result is `{"texts": [string]}`. Tokenization enables
special tokens and disables truncation before admission. The child uses local
Transformers, explicit safetensors, no remote code, configured dtype and
generation parameters, and `cuda:0`. Framed RPC, stderr, EOF, malformed
frames, timeouts, child death, PID reuse, and uncertain process-group cleanup
fail closed. Only serial execution is supported; no measured capacity is
claimed.

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

## GECToR

GECToR uses the common owned Python worker through the child command
`services.llm.providers.gector_worker`; the parent imports no Torch or GECToR.
Its selected offline manifest contains the exact GECToR file set, including
`verb-form-vocab.txt`, and binds `model.safetensors` to the configured model
digest. Loading is local-only, safetensors-only, no remote code, explicit
float32 on `cuda:0`, and rejects missing, unexpected, or mismatched weights.
The child supplies the reviewed DeBERTa-large configuration only to satisfy
GECToR 1.2.0's upstream encoder lookup and restores its temporary patch.

The sole supported bucket is native batch one, one iteration, exactly 128
tokenizer subwords, float32, numeric JSON zero `keep_confidence` and
`min_error_prob` (integer or float, never boolean or non-finite), with identity
`gector:p1:tokens128:keep0:min0:iterations1:batch1:float32`. Requests contain
exactly those five fields and must match the bucket. Tokenization disables
truncation before execution and returns the exact `{"accepted": boolean}`
validation result: an overlong input returns `false` so the parent rejects it
without poisoning an otherwise healthy worker; tokenizer/runtime faults remain
fatal. Loading sets `config.max_length=128` before model creation, overriding
artifact or Transformers generation defaults while separately requiring the
DeBERTa architecture's 512 positions. Execution checks the aligned output again. The response
is `{"texts":[non-empty string]}`. No measured capacity is claimed; RM
admission is bounded at p=1 with one buffered request.

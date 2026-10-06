# Provider adapters

## CoEdIT

CoEdIT uses one fresh, process-group-owned offline Python worker per load. The
parent imports no Torch/CUDA and requires a captured Linux GPU proof for the
parent supervisor before the child is launched. The strict request is
`{"instruction": string, "texts": [string]}` and the aligned result is
`{"texts": [string]}`. The default bucket remains native batch one. An explicit
`max_native_batch_size` opt-in (maximum 32) uses an exact batch-bound bucket and
the fixed 5 ms collection delay. The capacity tool limits throughput candidates
to 16 and permits discovery ceilings of 2–32 only with `--discover-memory`.
Neither mode approves a profile; runtime defaults remain batch one. Candidate
waves retain one exact configured bucket and bounded `p+p` admission.
Concurrent calls coalesce into one child `execute_batch` RPC. Admission
performs bounded envelope and bucket checks without a serialized child RPC;
immediately before native execution, the child validates every item without
truncation and retains request order. An expected child request-validation
rejection fails the whole native batch before model execution and is recoverable
only for `execute_batch`; uncertain worker errors still fail the worker closed.
Tokenization enables special tokens and disables truncation. The child uses local
Transformers, explicit safetensors, no remote code, configured dtype and
generation parameters, and `cuda:0`. After load it performs one bounded
child-owned `cuda_ready` operation: a one-element CUDA allocation followed by
`torch.cuda.synchronize()`, retained by the loaded worker until it exits. A
temporary CUDA operation is not residency evidence because its tensor may be
released as the RPC returns. The retained allocation creates actual continuing
CUDA state without operator input, tokenization, generation, or model output;
readiness still requires exact CUDA/NVML identity and nonempty owned-runner
proof. If the bounded expected-runner probe remains `_ResidencyPending` with no
strict descendant, CoEdIT alone uses a bounded fail-closed fallback: two fresh child
RPC observations prove the exact PID/start-time, model and retained witness on
`cuda:0`, while the captured supervisor ancestry fence and same-device/total
memory observations show a strictly positive post-load effect over the
pre-load baseline. Observation timestamps are ordered within each point; used/free
bytes may fluctuate, while GPU UUID, supervisor identity, and total memory remain
immutable. Any other proof error, topology/identity change, foreign descendant,
malformed witness, or nonpositive memory fails immediately.
Memory is supplemental effect evidence and never becomes cleanup authority;
NVML IDs remain excluded from cleanup. CoEdIT passes its exact child PID/start-time identity to the optional
expected-runner proof path: it accepts only a matching NVML PID whose fenced
procfs ancestry reaches the captured supervisor. On failure, the path emits
only counts for `exact_child`, `strict_supervisor_descendant`,
`foreign_or_baseline`, `unreadable_or_unconnectable`, and
`identity_mismatch`; no PID, command, or ancestry data is exposed, and every
non-exact category remains excluded. An empty NVML observation is reported
separately from a bounded observed-but-excluded runner set. Framed RPC, stderr, EOF, malformed
frames, timeouts, child death, PID reuse, and uncertain process-group cleanup
fail closed. The child returns private bounded batch cardinality and monotonic
execution timing evidence, accepted only when CUDA synchronization is confirmed;
this is an observation for future measurement, not a capacity claim. The retained
observation window is bounded by the configured native batch size and its explicit
drop count makes eviction observable. Execution RPC serialization, ownership
fencing, and cleanup remain unchanged. The batching window remains the ordinary
fixed 5 ms collection policy; admission does not introduce a wave barrier.

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
required completion and prompt-count proof is strict. Evidence binds the
configured options to the exact request body and retains actual `eval_count` as
the workload witness. Counts from 1 through 64 are valid because natural EOS
may stop below the configured maximum-output ceiling; an optional `done_reason`
is accepted only when it is a known bounded value.

The local `ollama create` subprocess has a bounded aggregate stdout/stderr
drain, timeout, and process-group reap. A failed import remains adapter-owned
until Resource Manager cleanup has made `/api/ps` model absence available. It is
not safe to execute before
readiness, and direct execution additionally enforces the configured measured
parallelism; Resource Manager remains the normal admission authority.

Readiness preloads the model and requires `/api/ps` to show exactly the canonical
`<digest-name>:latest` model,
fully resident in VRAM. Ordinary Linux uses concrete shared-GPU residency evidence: the configured
supervisor and non-empty, unique runner identities must be exact descendants;
missing or unsafe ownership evidence fails closed. On Docker Desktop/WSL, SmolLM may instead
use a bounded fallback requiring unchanged GPU/supervisor identities, a current fenced Ollama
daemon with a strict owned descendant, the exact private `/api/ps` fully-VRAM model, matching
fenced pre/post memory observations on the same GPU and total capacity, and a strictly positive
used-memory delta. Every conjunct is required; malformed, changed, missing, nonpositive, or
non-pending evidence fails closed. This grants readiness only: it never adds NVML IDs to cleanup
authority or controls foreign workloads. Model-switch cleanup drains
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

The new readiness contract retains a worker `cuda_ready` witness, prefers
exact expected-runner proof, and permits a bounded local fallback only after
typed `_ResidencyPending`: exact child identity, retained `cuda:0` witness,
and two valid post-load whole-device points above a validated pre-load baseline
are required, with stable GPU/supervisor/total-memory identity. Used/free
values may fluctuate; other proof errors fail closed. This is local and
unverified pending tester execution, not candidate success.

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
CoEdIT native batch observations include a typed allocator witness with
baseline/peak/final allocated and reserved bytes.  Adapter and capacity
checks must preserve and validate this record, rather than treating a
successful response as memory evidence.  Whole-device NVML sampling remains
the authoritative reserve/foreign-allocation guard.

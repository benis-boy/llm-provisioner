# Bounded ResourceManager compatibility experiment

This is candidate-only evidence for G4.1 and G4.3, not a production adapter,
dependency pin, capacity claim, or service implementation. Profiles are
synthetic `p=1`, `unmeasured-harness` contracts; they must not be used for
admission sizing.

## Candidate flow and truthful limits

The RM harness verifies the selected manifest and every canonical selected
artifact hash before loading. It selects one NVML physical GPU by the supplied
UUID, records its compute-process baseline, and each CUDA-owning child reports
the same NVML UUID. The parent does not import Torch.

For `SmolLM -> CoEdIT -> GECToR`, one persistent child validates, loads, becomes
ready, validates and executes an exact small fixture and an exact configured
upper-fixture. The latter is **not** a model maximum. CoEdIT and GECToR use the
`upper-fixture` bucket; SmolLM uses positive `context_size=512`. Transformers
fixtures are tokenized without truncation and fail if over 128 tokens. Before
Ollama starts, SmolLM streams the selected GGUF metadata and token array. It
requires `gpt2`/`smollm`, every printable-ASCII singleton fallback, disabled
implicit BOS, and a GGUF context of at least 512. It then admits only printable ASCII in
a 256-raw-byte fixture bucket. The exact raw request frame is
`<|im_start|>user\n` + text + `<|im_end|>\n<|im_start|>assistant\n`; `raw:true`
prevents Ollama from applying another template. This establishes the conservative
upper bound `framed ASCII bytes <= 448`, leaving a 64-token output reserve
in `num_ctx=512`. Ollama uses fixed `num_predict=64` and `temperature=0`, and a
response must be `done: true` with `prompt_eval_count <= 448`. This is neither an
exact token count nor a model maximum; non-ASCII input fails closed because its
byte fallback coverage was not independently established.

Every model also has an active execution cancellation: the child must report
that it has entered `LoadedRuntime.execute` for the exact request before RM
cancellation is sent, then the late `RESPONSE_FINISHED`
event must have no result. This proves RM's stale-result fence, not guaranteed
GPU-kernel entry or interruption. The notification is separate from the final
RPC response so a short execution cannot race the polling RPC. The child can
receive cancel concurrently with execute, but
the real adapters currently have no claimed safe in-flight interruption hook.
Any failed event, rejected submission, missing result, manifest mismatch, UUID
mismatch, timeout, or cleanup mismatch fails the candidate run.

`stop_session` performs provider unload and cleanup before the next model is
loaded. The harness always terminates and verifies the child process group
(including descendants), even if the child leader is already dead, **before**
the next load. On Linux the parent enables `PR_SET_CHILD_SUBREAPER` before
spawning the child and reaps adopted group descendants after a group kill, so a
zombie descendant cannot leave `killpg(..., 0)` falsely holding the residency
fence. The saved child PGID is used after its leader exits; cleanup never
derives a PGID from a reaped leader. A cleanup mismatch still fails closed.
After a process-loss stop the harness logs group-gone and NVML-baseline booleans
only. Child RPC reads, writes, shutdown, TERM, and KILL have
bounded waits. Runtime phase logs contain lifecycle/identity information only,
never prompt text.

## Context refresh and build

The context must already have been created with `prepare.py --download` and
must retain its manifest, model inputs, wheelhouse, and provenance. Refresh
updates only executable harness copies; it does not recopy models or wheels.

```sh
python3 tools/compatibility/refresh_context.py --output .compatibility/context
docker build --progress=plain --network=default \
  --build-arg CUDA_RUNTIME='nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867' \
  -t llm-compatibility-spike:candidate .compatibility/context
```

`CUDA_RUNTIME` intentionally has no Dockerfile default. The explicit build
argument is required so the candidate base image is auditable.
Build-time networking is required for this experimental Dockerfile's apt OS
packages; Python installs use the retained hash-locked local wheelhouse. The
inference container itself has no network. A `--network=none` build failed at
apt on 2026-09-16; that command mismatch is not a devcontainer/GPU blocker or
proof of a fully offline reproducible OS build.

## Exact candidate run commands and bounds

Set `TARGET_GPU_UUID` to the physical `GPU-...` identity selected for this
experiment. The owned container name, no network, dropped capabilities and
no-new-privileges are deliberate. Do not run the two commands concurrently.

```sh
TARGET_GPU_UUID='GPU-REPLACE-WITH-TARGET-UUID'
timeout --signal=TERM --kill-after=25s 660s docker run --rm \
  --name llm-compatibility-spike-candidate --network=none --cap-drop=ALL \
  --security-opt=no-new-privileges --gpus "device=${TARGET_GPU_UUID}" \
  -e NVIDIA_VISIBLE_DEVICES="${TARGET_GPU_UUID}" \
  llm-compatibility-spike:candidate
```

```sh
TARGET_GPU_UUID='GPU-REPLACE-WITH-TARGET-UUID'
timeout --signal=TERM --kill-after=25s 660s docker run --rm \
  --name llm-compatibility-rm-spike-candidate --network=none --cap-drop=ALL \
  --security-opt=no-new-privileges --gpus "device=${TARGET_GPU_UUID}" \
  -e NVIDIA_VISIBLE_DEVICES="${TARGET_GPU_UUID}" \
  --entrypoint /opt/venv/bin/python llm-compatibility-spike:candidate \
  /opt/llm/rm_spike.py --models-root /opt/llm/models \
  --manifest /opt/llm/manifest.json --target-gpu-uuid "${TARGET_GPU_UUID}" \
  --timeout 120 --rpc-timeout 30 --cleanup-timeout 20
```

The 660-second outer deadline bounds the complete container. Each RM lifecycle
or fixture wait is 120 seconds, ordinary RPC is 30 seconds (cold load/execute
RPC is 240 seconds), and cleanup is 20
seconds; local child close uses TERM for two seconds then KILL for two seconds.
Both RM commands were rerun successfully after the raw-input and flat-image
packaging repairs described below.

## Observed verification (2026-09-16)

The expanded RM candidate image built with the explicit CUDA digest above and
passed with networking disabled on RTX 4070 Ti,
`GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963`, driver 591.86. SmolLM, CoEdIT and
GECToR each emitted `ready`, `cancel_fenced`, and `cleanup_restored`, and the run
returned `passed-candidate`. The latest rebuilt image includes the private
Ollama endpoint, independent bounded ASCII input evidence, exact raw framing,
finished-response checks and the packaged input-bound helper. Both normal and
injected-process-loss runs passed for all three models. The normal run exercised
small and configured upper fixtures; the failure run restored the NVML process
baseline and proved every owned process group gone. The tester verified both
owned containers absent. Focused harness tests passed **36** with warnings
treated as errors. Candidate Python
dependencies remain hash-locked in the retained context; the base is CUDA 12.8.1,
Python 3.12, Torch 2.7.1+cu128, Transformers 4.49.0, tokenizers 0.21.0,
safetensors 0.5.3, gector 1.2.0 and Ollama v0.11.6. These are candidate inputs,
not approved production pins or a measured profile.

## Process-loss candidate scenario

The ordinary command above is unchanged. A separate, opt-in failure run can be
requested with `--inject-process-failure`; it waits for observed active
execution, kills the child process group, requires a `FAILURE` for that exact
request/attempt with no publishable result, then explicitly stops/fences the
session through RM authority before checking the NVML baseline. This scenario
does not claim that every provider exception automatically invalidates a
session. It is bounded candidate evidence only:

```sh
timeout --signal=TERM --kill-after=25s 660s docker run --rm --network=none \
  --cap-drop=ALL --security-opt=no-new-privileges \
  --gpus "device=${TARGET_GPU_UUID}" -e NVIDIA_VISIBLE_DEVICES="${TARGET_GPU_UUID}" \
  --entrypoint /opt/venv/bin/python llm-compatibility-spike:candidate \
  /opt/llm/rm_spike.py --models-root /opt/llm/models --manifest /opt/llm/manifest.json \
  --target-gpu-uuid "${TARGET_GPU_UUID}" --timeout 120 --rpc-timeout 30 \
  --cleanup-timeout 20 --inject-process-failure
```

SmolLM uses a child-owned temporary `OLLAMA_MODELS` store reconstructed from the
selected GGUF/Modelfile and a source-hash-derived local model name. The store is
removed only after the child process group is proved gone. Every Ollama CLI call,
including `ps` and `rm`, receives the same private `OLLAMA_HOST` as its server.
Child snapshots expose
only lifecycle/readiness, PIDs, active state, model/source identity, store state,
and GPU identity; prompts and results are excluded. The SmolLM bucket is 256
printable-ASCII raw bytes with exact raw framing, a proved 448-token conservative
prompt ceiling and `num_ctx=512`/`num_predict=64`. It is a conservative
no-truncation proof only for that bucket, not a model maximum or an exact token
count. Transformers remain explicitly bounded to 128 tokenizer tokens without
truncation.

## Remaining evidence gaps

A passing run establishes only an experimental offline candidate matrix. It
does not measure throughput, VRAM reserve, useful parallelism, production
provider cancellation, service health, artifact provisioning from scratch, or
qualifying production-boundary E2E proof. SmolLM rejects selected artifacts
lacking the required tokenizer/context metadata or printable-ASCII fallback
coverage, and rejects a reply that is not finished, lacks `prompt_eval_count`, or
exceeds the conservative ceiling. Extending this proof to arbitrary UTF-8 requires
independent coverage evidence for all byte fallbacks; the harness does not claim it.

# CoEdIT adapter GPU check

This is a bounded candidate check for G4.1. It runs the installed
`CoEdITProvider` (not a tools-runtime substitute) through the in-process
ResourceManager. The profile is labelled `unmeasured`, uses only `p=1/m=1`,
and makes no capacity, production-readiness, or model-maximum claim. The
near-bucket fixture is validated by the child tokenizer with truncation
disabled; failure is the safe result if it does not fit.

The default `--native-batch-size 1` path is the legacy serialized scenario.
Opt in to the bounded real p2 check with `--native-batch-size 2`: it uses the
exact `coedit:p2:input128:output64:float16:beams1:nosample` bucket and submits
two distinct one-text requests concurrently through the real ResourceManager.
For both the small and near-bucket waves, success requires exactly one
synchronized native batch-2 observation with both request identities, aligned
nonempty individual responses, and zero lost observations. A serialized
batch-1 response, malformed observation, or dropped observation fails the check;
the cleanup path still runs. P2 remains explicitly `unmeasured` and is not a
capacity measurement.

The harness verifies the exact selected CoEdIT manifest/model digest, captures
`LinuxGPUProof` for the parent before the worker is spawned, checks the exact
worker PID residency, validates aligned nonempty output internally, stops the
RM session, rejects a stale token, and proves worker/GPU cleanup. It never
prints prompts, completions, or process tables. `--inject-process-failure`
kills only the identity-fenced owned worker group and requires failure without
a result followed by RM cleanup.

## Offline image and run

The candidate inputs are deliberately not final pins: base
`llm-compatibility-spike:candidate`, Torch 2.7.1+cu128, Transformers 4.49.0,
safetensors 0.5.3, gector 1.2.0, and Ollama 0.11.6 are investigation values.
CoEdIT uses the installed Python dependency layer and does not use Ollama.

The adapter image now copies the harness and allowlists it in
`Dockerfile.adapter.dockerignore`. The candidate base image supplies the
existing `/opt/llm/models` tree and candidate manifest; do not invent host
`.compatibility/models` or manifest mounts. The harness provisions a temporary
selected volume under `/tmp` and therefore needs a writable container layer (or
an explicitly supplied writable temporary volume). Do not use `--read-only`
unless `/tmp` is separately mounted writable with several GB available.

From the repository root, with the candidate image and wheelhouse already
available locally:

```sh
docker build --network none \
  --file tools/compatibility/Dockerfile.adapter \
  --tag llm-compatibility-adapter:candidate .

docker run --rm --init --network none --pid=host --gpus '"device=GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963"' \
  --entrypoint /opt/venv/bin/python llm-compatibility-adapter:candidate \
  /opt/llm/coedit_adapter_check.py --models-root /opt/llm/models \
   --manifest /opt/llm/manifest.json \
   --target-gpu-uuid GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963 \
   --host-pid-namespace
```

The exact focused GPU invocation (using the existing UUID, network isolation,
and host PID namespace) is:

```sh
docker run --rm --init --network none --pid=host --gpus '"device=GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963"' \
  --entrypoint /opt/venv/bin/python llm-compatibility-adapter:candidate \
  /opt/llm/coedit_adapter_check.py --models-root /opt/llm/models \
  --manifest /opt/llm/manifest.json \
  --target-gpu-uuid GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963 \
  --host-pid-namespace --native-batch-size 2
```

Use a separate run with `--inject-process-failure` for the bounded cleanup
case. Suggested overall timeout is 360 seconds; allow at least 240 seconds for
model load and 120 seconds per request. The operator must replace `GPU-UUID`
with the exact physical NVML UUID and attest that `/proc` under `--pid=host`
is authoritative for that NVML namespace. Networking remains disabled.

The entrypoint override above is intentional and prevents accidentally running
the SmolLM check. The exact installed Torch, Transformers, tokenizers, and
safetensors versions plus CUDA runtime identity are emitted in the fixed
result schema together with the exact provisioned manifest and CoEdIT model
SHA-256 values; the `candidate:` profile identity is derived from those values,
not from a hand-written version label. The parent obtains this metadata through
`importlib.metadata` and never imports Torch.

After the run, verify the `--rm` container is absent (`docker ps -a --filter
ancestor=llm-compatibility-adapter:candidate`), and verify no owned worker remains in the authoritative GPU
residency proof. Do not treat unrelated/unknown NVML processes as owned or
signal them. Remove any temporary diagnostic mount/container only after the
fixed-schema result has been collected; the harness itself removes its
temporary artifact volume on success.

No operational GPU result is claimed by the unit tests. Backend-tester must
run the commands above and report exact image digest, GPU/runtime identity,
manifest and model hashes, worker residency, failure-cleanup outcome, and any
environment blocker. Candidate versions must not be promoted to final pins.

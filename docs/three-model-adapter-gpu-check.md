# Three-model adapter GPU check

This is a bounded offline candidate check, not a benchmark. It runs one
parent-owned Linux GPU proof and one `ResourceManager` through the sequence
`SmolLM -> CoEdIT -> GECToR -> SmolLM`. Every transition is a real RM
replacement, so cleanup failure prevents the next load. The capacity profile
is explicitly `UNMEASURED`; the output contains identities and fence results,
not prompts, completions, process tables, or timing claims.

The image expects the exact selected artifact volume at `/opt/llm/models` and
the canonical combined manifest at `/opt/llm/manifest.json`. It is entered as:

```text
/opt/venv/bin/python /opt/llm/three_model_adapter_check.py
```

Build the existing adapter image offline (`--network none`) and run the named
check:

```bash
docker build --network none --progress=plain --file tools/compatibility/Dockerfile.adapter \
  --tag llm-compatibility-adapter:candidate .
docker run --rm --name llm-three-model-adapter-check --init --network none --pid=host \
  --gpus '"device=GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963"' \
  --entrypoint /opt/venv/bin/python llm-compatibility-adapter:candidate \
  /opt/llm/three_model_adapter_check.py --models-root /opt/llm/models \
  --manifest /opt/llm/manifest.json --target-gpu-uuid GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963 \
   --port 11434 --host-pid-namespace
```

`--host-pid-namespace` is required because the command uses `--pid=host`; do
not add a procfs bind. The selected port must be unused on the host.

The harness uses the existing private Ollama launcher and identity-fenced
cleanup helper. Its cancellation check fences publication at execution entry;
it does not claim to interrupt an active GPU kernel. Temporary artifacts are
retained only when owned provider, daemon, or GPU cleanup remains uncertain;
positively cleaned failed and successful runs delete them. No completeness
beyond these checks is claimed.

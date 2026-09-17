# Candidate QueueScheduler-to-CoEdIT GPU check

## Verified continuation (2026-09-17)

The focused scheduler-adapter/scheduler/store/results suites pass **47 tests**
with warnings treated as errors. Fresh offline image
`sha256:9ce2cfbe09fce26d955e5cf1ee9841beaa7907a4b672b93749197ee25d59ee55`
passed the command below both with its default user and with
`--user 65534:65534 --name llm-scheduler-adapter-check-nonroot` in place of the
default name. Both returned `done`, one total receipt, `cancelled`, no cancelled
publication, and proved cleanup. Timing was explicitly `gpu_ms:null` /
`gpu_timing_complete:false`; profiles remained unmeasured/ineligible.
Both owned containers were removed and verified absent.

The existing `three_model_adapter_check.py` also passed with this image and
`--user 65534:65534`, using name `llm-three-model-adapter-check-nonroot` and its
documented `--port 11434`. Four model responses, three switches, cancellation
and stale fences, six memory observations and owned cleanup passed; its
container was verified absent. This is same-UID non-root compatibility, not
distinct `llm` and `ollama` service users or an approved production image.

This slice adds `tools/compatibility/scheduler_adapter_check.py`, a bounded
candidate-only in-process composition:

```
QueueScheduler -> SQLite QueueStore/ResultStore/LocalPublisher -> ResourceManager -> installed CoEdIT
```

It uses the real installed CoEdIT provider and the selected artifact manifest,
the parent-process GPU proof, and an explicit p1 profile whose identity is
`unmeasured-scheduler-adapter-check`. It does **not** use BootstrapRuntime or
HTTP ModelBinding and does not alter their measured-profile fail-closed rules.
The result is always `candidate: true`, `profile: unmeasured`, and
`profile_eligible: false`; it is not production E2E or capacity evidence.

## Checks and bounded observations

The harness performs three bounded checks with one owned lifecycle:

1. It enqueues accepted intent, closes and reopens the actual QueueStore using
   the same identity before scheduler start, then proves `done` only after the
    content-addressed result is read/verified, its exact one-text CoEdIT JSON
    contract is validated against the persisted bytes (which must also equal
    the delegate result), and exactly one local publication receipt exists.
    This is labelled **pre-dispatch reopen**; it is not process crash or
    in-flight recovery evidence.

   Provisioning may replace source-root metadata with its stable runtime root.
   The harness therefore compares only the model ID and ordered selected
   `(path, size, sha256)` records; selected-set validation and source-byte
   `verify_manifest` remain mandatory before this comparison.
2. It wraps the installed provider with an observable execution-entry gate,
   cancels after entry, releases the delegate, and waits for that delegate's
   later `response_finished` observation before requiring terminal `cancelled`,
   no handoff, and no additional receipt. The gate proves result fencing; it
   does not claim CUDA kernel interruption.
3. It reports the exact selected manifest/model hashes and GPU UUID, receipt
   counts, status/timing facts, and cleanup. It emits no prompts, results, or
   request IDs.

`QueueScheduler.stop()` is the sole owner of RM session stop and provider
unload. Because that API contains its own best-effort error handling, the
harness verifies the authoritative public RM lifecycle snapshot (`startup`,
available, no session), provider cleanup verification, and GPU-proof cleanup
only after it returns successfully. A timed-out or failed scheduler stop, or a
direct RM stop, prevents all dependent lifecycle/GPU probes. It never issues a stale second `stop_session`. Timed-out
cleanup gets one final bounded settlement window; unresolved work retains its
temporary evidence, emits only `cleanup_retained=true` (not a private path),
and fails rather than producing candidate success. Late detached cleanup
exceptions are observed. The named Docker container is the external containment
boundary; cancellation-suppressing work cannot guarantee a bounded
`asyncio.run()` shutdown. The harness never controls foreign workloads.

Failure output is sanitized JSON containing only a closed `stage`, closed
`code`, and independent cleanup flags (`scheduler`, `rm_snapshot`,
`provider_verify`, `gpu`) plus top-level `cleanup_unresolved`; it never nests
the complete diagnostic record. The cancellation gate deliberately
declines the provider's advisory cancellation before it calls the real delegate;
this proves scheduler/RM result fencing after a real response, not provider or
CUDA interruption.

## Exact offline Docker command

Build from the repository root using the already prepared offline base image
and wheelhouse, then run with networking disabled and the selected GPU:

```sh
docker build --network=none -f tools/compatibility/Dockerfile.adapter \
  --build-arg BASE_IMAGE=llm-compatibility-spike:candidate \
  -t llm-compatibility-adapter:candidate .
docker run --name llm-scheduler-adapter-check --init --network=none \
  --gpus '"device=GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963"' \
  --pid=host \
  --entrypoint /opt/venv/bin/python \
  llm-compatibility-adapter:candidate /opt/llm/scheduler_adapter_check.py \
  --models-root /opt/llm/models \
  --manifest /opt/llm/manifest.json \
  --target-gpu-uuid GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963 \
  --host-pid-namespace
docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' llm-scheduler-adapter-check
docker rm llm-scheduler-adapter-check
```

The candidate image must package `/opt/llm/models` and `/opt/llm/manifest.json`;
there are no host bind-mount guesses in this command. The named container is
intentionally retained until its exit status is inspected, then explicitly
removed. No network or runtime download is permitted.

## Focused verification (delegated; do not run here)

Backend-tester should run only:

```sh
.venv/bin/python -W error -m unittest tests.unit.test_scheduler_adapter_check
.venv/bin/python -W error -m unittest tests.unit.test_scheduler tests.unit.test_store tests.unit.test_results
```

The Docker command above is the only compatibility execution command. This
implementation intentionally does not claim full production E2E, measured
capacity, CUDA interruption, or crash/in-flight recovery.

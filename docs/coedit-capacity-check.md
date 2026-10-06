# Offline CoEdIT capacity check

`coedit_capacity_check.py` is a fail-closed candidate boundary. It accepts an
operator-selected strict UTF-8 JSON input file (exactly `instruction` and
`text`, duplicate keys rejected, maximum 256 KiB), or the explicitly labelled
`--configured-fixture` input. Input JSON is canonicalized to the provider
request shape (`instruction` plus one `texts` item); source JSON whitespace is
not part of identity. After the owned worker loads, the worker's loaded
tokenizer is authoritative: it tokenizes exactly `instruction + "\n" + text`
with `add_special_tokens=True`, `truncation=False`. A configured fixture is
extended only with deterministic explicit `word` tokens for at most 256
attempts; it is never padded. A supplied input must already be exactly 128
tokens or the run fails with `insufficient_max_input`.

The worker's private `benchmark_input` RPC returns only token count, configured
maximum, SHA-256 identity, and (only for configured generation) the generated
text. The harness retains the exact canonical
request bytes privately, checks their identity before every wave, and uses
those same bytes for every native request. It does not log the payload. An
exact witness sets `max_input_verified:true`, but `profile_eligible` remains
false: these observations are candidate evidence and do not prove a memory
resource bound.

The candidate runner uses one real ResourceManager session, selected artifact
provisioning, the installed CoEdIT provider, and native CoEdIT batch
observations. In default throughput mode it performs four serial baselines, then wave zero and four
ordered waves at at most eleven deterministic probe points (p=1 plus at most
ten larger points, including the configured ceiling). Every wave must have exact request IDs,
native cardinality, valid aligned outputs, CUDA synchronization, and bounded
whole-device NVML samples. Samples are device-wide high-water observations,
not VRAM attribution to requests. Identity changes, missing samples,
serialization, timeout, failed cleanup, or an unclassified failure makes the
result incomplete.

The successful p=2 run is candidate evidence only: its `candidate_n:2` and
`optimal_parallelism:2` are a bounded-probe candidate optimum, not measured
capacity or an approved profile. Each memory observation retains its fenced
read start/end interval; a read must fall wholly within the synchronized native
execution interval, and observation chronology is strict across the run.

The 20% free-memory reserve is enforced for every successful wave; it uses
whole-device free/total and does not attribute foreign or reserved memory. A failed
point leaves only the preceding successful point as a candidate; a max
supported value or operator ceiling is never promoted. Throughput selection
uses the existing deterministic 2% rule. Reaching the requested ceiling has
reason `probe_limit_not_memory_bound` and is only a `candidate`, never
`measured`; `profile_eligible` is always false and this tool never opens or
writes a profile database. It retains four serial baselines, one warmup and
four measured waves per bounded point, with globally unique IDs and exact
per-wave batch observations. Optional raw JSON is a sanitized per-wave summary
bounded to 256 KiB, published atomically without overwriting an existing file.
It omits request IDs, prompts, outputs and full sample arrays; it retains native
observation counts, batch sizes, correlation/drop counts, sample validity and
overlap counts, and one actual fully in-window fenced memory observation.
A failed or unproved cleanup
retains its owned temporary resources and reports no success.

Cleanup requires both the provider's cleanup witness and the identity-fenced
`LinuxGPUProof.cleanup()` check after unload. Expected provider, RM, sampling,
evidence, and timeout failures return a minimal structured `incomplete` summary
with a sanitized exception category and cleanup outcome; exception text and
payloads are never emitted. Failed waves retain bounded numeric summaries and
closed failure phase/category diagnostics. A secondary diagnostic-write failure
does not replace the original sanitized terminal failure.

Native execution fences are integer monotonic nanoseconds, as are NVML read
fences. NVML sampling uses a 10 ms interval, at most 16,384 samples per wave,
and a 120-second wave deadline. This sampling interval is independent of the
unchanged 5 ms native request-collection delay.

## Candidate command

The adapter image copies compatibility scripts flat at `/opt/llm`; override its
default adapter entrypoint explicitly. The following remains an offline,
candidate-only command and writes no profile:

```bash
docker run --rm --name llm-coedit-capacity-candidate --init --gpus '"device=GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963"' \
  --pid=host --network none --entrypoint /opt/venv/bin/python \
  llm-compatibility-adapter:candidate /opt/llm/coedit_capacity_check.py \
  --models-root /opt/llm/models --manifest /opt/llm/manifest.json \
  --target-gpu-uuid GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963 \
  --host-pid-namespace --native-batch-size 2 --configured-fixture
```
## Allocator evidence

An earlier bounded p=16 candidate run failed closed during warmup with
`native_batch_correlation` after a native 14+2 split; the minimum sampled free
memory was 7,125,778,432 bytes of 12,878,610,432 whole-device bytes. This is not
a lower bound on free memory between samples.
Four waves at p=1,2,4,5,7,8,10,12,13,15 were measured; p=16 was then
incomplete. This is historical scheduling/batch-correlation evidence, not an OOM or
memory-bound result. The owned container was removed and verified absent.

The production CoEdIT path now avoids one serialized child tokenizer RPC per
request during ResourceManager admission. Admission retains bounded envelope and
bucket checks, while `execute_batch` authoritatively validates every item before
any model operation. Expected request-validation rejection is recoverable only
for that operation; uncertain worker failures remain fatal. This preserves the
fixed 5 ms collection policy and does not add a benchmark barrier, force p=16,
or bypass normal ResourceManager admission. Focused unit verification passed 122
tests with warnings treated as errors; compilation and whitespace checks passed.
The subsequent rebuilt offline p=16 run passed. The earlier 14+2 result remains
historical failure evidence; no profile is approved.

Each synchronized native batch records the child worker's Torch allocator
baseline, peak, and final allocated/reserved counters.  The counters are
bounded integer witnesses owned by the worker; missing or contradictory
records fail the measurement closed.  They are not total-process CUDA peaks.
The device-wide GPU proof remains authoritative for the sampled NVML
used/free/reserve guard, including foreign allocations.

Peak allocator values are reset immediately before the tokenizer/tensor/
generation operation and captured after CUDA synchronization.  Consequently,
the incremental high-water evidence is relative to the loaded baseline and
excludes unrelated allocations, while CUDA caching and allocator ownership
caveats remain.  Torch reserved deltas and NVML peaks are intentionally not
combined into an exhaustive peak.

The probe through p16 is exploratory candidate evidence only: it does not
certify a safe N or make a profile eligible.  Exact input, runtime, artifact,
GPU, and profile identities are checked and mismatches fail closed.

## Decoder workload and incremental discovery (2026-09-17)

Native execution reports bounded per-row decoder-step counts against the
provider's configured output maximum. The model's generation configuration owns
decoder-start, EOS and PAD identity. Counts exclude decoder-start, include the
first EOS, and exclude validated trailing padding. Ordinary valid short outputs
remain valid inference results, but cannot verify maximum decoder coverage.
No generation settings are changed to manufacture the witness. The candidate
summary's `max_output_verified` requires the complete expected wave schedule
and exact configured maximum for every row.

The rebuilt throughput candidate passed all 59 waves with input128/output64,
candidate optimum16 and owned cleanup (135 focused tests). Image manifest:
`sha256:ed7ddf7abe705a3a17daf755001d102ada7ca68c555fb3aca5f86858b17a4474`.

Add `--discover-memory` to the candidate command to collect four serial
baselines followed by four repeats at **every** p=2..configured ceiling.
There are no throughput warmups or optimum selection in this mode. Invalid
native correlation, decoder undercoverage, allocator/timing/telemetry failures
or a sampled reserve breach stop further waves and retain the categorized
failed wave. A reserve failure can retain the preceding fully observed point
as historical `observed_safe_through`; it never establishes current capacity.
Every result keeps `memory_safe_n:null` and `profile_eligible:false`.

The first actual discovery stopped at p16 wave3 because its terminal watchers
replayed from zero beyond the default 1,024-event retained history. This was not
a memory limit. Watchers now share the previous completed wave's cursor and
advance only after all current terminal sequences are validated; ResourceManager
retention and p+p admission are unchanged. A real-RM 544-request regression
proves resumable reads across history eviction.

The corrected offline run passed **64 waves/544 requests**: four baseline waves
and 60 discovery waves, with one exact correlated native observation and no
drops per wave. Maximum input/output witnesses and owned cleanup passed.
Sampled total memory stayed 12,878,610,432 bytes; minimum sampled free memory
was 8,192,479,232 bytes. This is not a between-sample guarantee or attribution
of foreign allocations. The ceiling16 remains unproved as a resource bound.
Image manifest:
`sha256:118f447d79d88207a2fb3e2be28d146c1cd504abd4e064caad469d76cfbcfd6d`.
The 62,110-byte sanitized artifact had SHA-256
`87fc05dd8e81b2c7800975a1923f05b7b295f9567a938c6325d13fb599d4f14e`.
It was reduced programmatically through a streamed Docker tar archive; no full
JSON was copied or dumped. Container `llm-coedit-memory-discovery-v2` was removed
and verified absent. **112 task-related tests**, focused compilation and
whitespace validation passed. No full test discovery was run.

## Session closeout: discovery through p=32

Native batching now permits an explicit maximum of 32, with unchanged default
batch one, 5 ms collection delay, request/frame limits and generation settings.
The CLI accepts throughput ceilings 1–16 and discovery ceilings 2–32; invalid
mode/ceiling combinations are rejected before provisioning or GPU capture.
Successful discovery (`complete`) and throughput (`candidate`) exit zero;
incomplete runs exit two. These exit statuses do not imply profile approval.
Raw artifacts preserve the full sanitized per-wave schema; exceeding 256 KiB
fails closed rather than silently removing measurement evidence.

The final offline discovery passed **128 waves / 2,112 requests** in **212.659s**:
four baseline waves and four repeats at every p=2..32. Every native batch was
correlated with zero drops, and every row verified exactly 64 decoder steps.
The exact 128-token input and owned cleanup also passed. Minimum sampled free
memory was **7,862,497,280 bytes**, with stable total-memory observations.
The result was `complete`, `observed_through_ceiling`, `observed_safe_through:32`,
`memory_safe_n:null`, `profile_eligible:false`. No resource limit was found;
this finite probe is not exhaustive peak-memory or approved capacity proof.

Reproducible final command (retain the owned container until diagnostics are
projected, then remove it and verify exact-name absence):

```bash
docker build --network none --file tools/compatibility/Dockerfile.adapter --tag llm-compatibility-adapter:candidate .
docker run --name llm-coedit-memory-discovery-p32-final --init \
  --gpus '"device=GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963"' \
  --pid=host --network none --entrypoint /opt/venv/bin/python \
  llm-compatibility-adapter:candidate /opt/llm/coedit_capacity_check.py \
  --models-root /opt/llm/models --manifest /opt/llm/manifest.json \
  --target-gpu-uuid GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963 \
  --host-pid-namespace --native-batch-size 32 --configured-fixture \
  --discover-memory --raw-output /tmp/discovery-p32-final.json
```

Image RepoDigest:
`llm-compatibility-adapter@sha256:137dfb5b6db92e22865bbc7d9ea4193a4e3b60d1882510af35cc250a4c5f0836`.
The **127,166-byte** artifact had SHA-256
`a5c98639ccd51e04e85304f78cf90400431a36e14a2de8e25fdb0971216d3111`.
Only bounded numeric evidence was projected programmatically from a streamed
Docker tar archive; no full JSON was dumped or copied. The tester removed the
named container and verified absence; foreign workloads were untouched.
**162 task-related tests**, focused compilation, whitespace validation and
offline build passed. Independent review found no remaining code defects.
For the provisioning proof (as distinct from this candidate-only tool),
CoEdIT's whole-device samples are aggregate native-batch observations. They
are retained for replay, but are not treated as bytes per slot. The owned
discovery protocol therefore tests every p=2..fixed-provider-capability point,
or accepts only a typed OOM/reserve-breach witness at N+1. Reaching a lower
operator ceiling without either boundary fails closed as
`configured_ceiling_unproved`; unknown and contract failures remain integrity
failures. Persistence requires the exact retained discovery sequence and
replayed memory summary, rejecting missing or tampered evidence.

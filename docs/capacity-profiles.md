# Capacity-profile benchmark preparation

`services.llm.provisioning.benchmark_requests` is the Phase 5 adapter-valid
benchmark-request precursor.  It binds a compact canonical fingerprint to the
model, request bucket, dtype, generation parameters, native batch shape, and
the model-specific context or iteration/token limits, plus caller-supplied
artifact/runtime/adapter identity witnesses.  Requests are schema-checked,
bounded, deterministic, and reject unknown fields, non-finite or boolean
numeric values, shape mismatches, and unsafe JSON/text sizes.

An adapter may provide a generator only when it also provides a validator that
proves the maximum representative request.  If safe generation cannot be
established, an explicitly configured request is required; no fallback means
fail closed.  The module stores only the bounded canonical request and hashes,
never model outputs or large diagnostic JSON.

This is a precursor to measured capacity profiles, not a profile approval or
production evidence claim.  Real-adapter offline execution, tokenizer/native
batch witnesses, GPU measurements, exact artifact identity integration, and
production deployment remain required before a profile can be selected.

`services.llm.provisioning.measurement` now implements the bounded Phase 5
measurement protocol as a local integration precursor. It uses an actual
`ResourceManagerWaveRunner`, an explicit ephemeral provisioning profile for
admission (never a durable measured profile), exact request/event fencing,
bounded telemetry, four serial baselines, incremental reserve-safe discovery,
the mandatory bounded sweep, and the 2% throughput-selection rule. A result is
eligible for persistence only after ResourceManager cleanup succeeds.

`services.llm.provisioning.measured_profiles` independently revalidates the
measurement schedule, exact identities, decoder/native/allocator/timing and
telemetry evidence, converts retained waves to `SampleMetadata`, constructs the
deterministic profile identity, calls `ProfileStore.save_measured()`, and
requires exact profile/raw-sample lookup afterward. Identical replay remains a
no-op and changed content under the same identity remains a conflict.

These integrations do not make a synthetic test measurement production
eligible. Every configured model and Ollama context still requires a real
offline adapter/GPU run with authoritative native evidence and exact selected
artifacts before its measured profile can be approved.

Resident SmolLM uses
`1 + floor((0.8V-B)/D)`: `D` is the cumulative additional-slot cost from the
exact fully valid p=2 discovery after the resident baseline. p=1 request or
sweep deltas are retained and validated, but are not authoritative capacity
evidence.
For non-resident native adapters (CoEdIT and GECToR), whole-device samples
describe an aggregate native batch and are retained only as replayable memory
summary evidence. They are never divided or extrapolated into a per-slot
formula. Discovery therefore runs exhaustively from p=2 through the fixed
provider capability, or stops only at a typed OOM/reserve-breach N+1 witness.
Each NVML sample is a fenced interval and correlates to execution when the
intervals intersect (`sample_start <= execution_end` and
`sample_end >= execution_start`); monotonic chronology and at least one
overlap are still required. A sample need not be wholly contained by execution.
Unknown and contract failures remain integrity failures. A lower operator cap
that reaches neither boundary is `configured_ceiling_unproved`.

It is not an arbitrary configured ceiling, and successful observation alone is
not a maximum. The retained native discovery sequence, memory summary,
chronology, reserve, and capability/resource witness are independently
revalidated before persistence. GECToR's fixed p=1 capability is the
degenerate exhaustive case and does not require a p=2 probe.

SmolLM has a fixed structural provider capability of 32, configured by the
adapter/runtime and separate from the operator's measurement ceiling. When the
memory formula is unavailable, only exhaustive direct evidence for every
reserve-safe p=2..32 wave with the operator ceiling set to 32 may establish
N=32. A lower ceiling remains unproved; a formula or resource-bound N below 32
continues to determine N.

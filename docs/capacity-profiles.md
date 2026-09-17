# Capacity profiles (precursor)

This document describes the bounded precursor implemented by
`ProfileStore`. It is **not G4**: it does not run a provider, measure a GPU,
prove concurrency, select artifacts, or produce production profiles.

## Boundary and API

`ProfileStore(path)` opens a SQLite database in WAL mode with `synchronous=FULL`.
The public operations are:

* `save_draft(profile, metadata)` stores a caller-labelled draft with only
  structural/bounds validation. Drafts are never returned by `lookup`; a later
  measured revision with changed evidence needs a new fingerprint. Explicit
  `save_measured` can promote byte-identical qualifying draft content after
  validating its complete evidence; draft writes never downgrade a measured row.
* `save_measured(profile, metadata)` stores an approved, caller-labelled
  measured record only when the evidence has four successful serial baseline
  samples, wave-zero warmups for every measured point, and four ordered
  successful measured waves at every sweep point. Points include 1, 2 when
  applicable, and N; no more than ten additional points beyond 1 are accepted. Throughput is
  computed from successful requests / aggregate wall time. A higher point must
  be at least 2% better to replace the lower point, so ties and near-ties
  deterministically prefer lower concurrency. The raw profile samples must be
  exactly baseline, warmups, then measured waves in that order.
* `lookup(...)` performs exact identity lookup. For SmolLM it selects the
  smallest configured positive context at least as large as the request. For
  CoEdIT and GECToR it requires the exact bucket identity. There is no fallback
  for another artifact, runtime, adapter, bucket, model, or larger-than-highest
  request.

`BenchmarkMetadata` carries the test-request fingerprint, ISO-8601 creation
time, provenance, raw baseline/warmup/measured evidence, and representative
bucket/context identity. The fingerprint is included in the deterministic
`profile_identity`. Identity replay is a no-op; changed content under an
existing identity is a conflict. Revisions are immutable rather than updated.

The store retains raw samples as individually ordered JSON rows. It stores no
pickle and no full benchmark/config dump. Canonical JSON uses sorted keys,
compact separators, UTF-8, and rejects non-finite numbers. Boolean values are
not accepted as numeric fields. Runtime selection only sees `measured` rows.

## Trust and fail-closed invariants

`POST /provisioning/validate-profile` is a read-only exact measured-profile
check. The server owns registry paths and current model/GPU/artifact/runtime/
adapter identities; the client supplies none of those trusted selections.
Every submitted field and raw sample must equal the measured row selected for
the requested context or bucket. Missing, draft, corrupt, or mismatched rows
fail closed. It requires `Idempotency-Key` and returns the existing `Ack`
shape; its deterministic operation ID is SHA-256 of canonical compact sorted
JSON of the validated request. Repeated requests revalidate without journaling.
Validation bodies default to 4 MiB because evidence can be large; deployments
may configure a smaller safe bound.

Read-only validation opens SQLite with `mode=ro` and never initializes or
promotes a registry. A WAL registry may require its `-wal` companion file to
remain readable for a consistent view; deployments must mount the database and
its SQLite sidecars read-only together.

The caller explicitly attests benchmark provenance. Provenance is metadata, not
cryptographic proof; SQLite cannot prove that a claimed GPU, runtime, or
measurement actually occurred. This precursor therefore does not claim
measured capacity or actual-GPU readiness.

The offline CoEdIT candidate slice is documented in
`coedit-capacity-check.md`. It never promotes its bounded probe to a measured
profile: exact 128-token input and 64-step decoder witnesses now have real
candidate evidence, but exhaustive peak/resource-bound proof remains absent,
so no runtime profile is saved. Incremental discovery reports only historical
`observed_safe_through`, never an approved `memory_safe_n`.

Profiles require `p <= N`, `buffer_capacity == p`, and a 20% reserve. SmolLM
requires positive context and no bucket. Other models require a bucket and no
context. Missing rows, malformed schema/data, identity mismatch, or incompatible
schema version raises/fails closed; the store never deletes or recreates an
existing runtime database. Creating a fresh database is an explicit caller or
provisioning action outside this module.

Schema version 1 consists of a metadata table, a canonical profile table, and
ordered raw-sample rows. Multi-connection writers use an immediate transaction;
duplicate identical writes are harmless and conflicting writes fail.
The schema is inspected exactly at startup, including columns, keys, foreign
keys, and enabled foreign-key enforcement. A non-empty unversioned or malformed
database is rejected. Creation is transactional. Lookup uses one read
transaction, verifies raw-row ordinals, metadata, identity, content hash, and
all evidence before returning. Ambiguous same-context records fail closed.

Evidence fields are bounded: ten additional sweep points beyond mandatory p=1,
16,384 latencies per sample,
64 KiB text fields, and a 4 MiB canonical evidence payload. Stored text lengths
are checked before JSON parsing.

## Deliberate non-goals and dependent issues

No benchmark runner, throughput sweep, GPU execution, production profile,
artifact-manifest producer, or scheduler/ResourceManager measurement
integration is included. The read-only profile-validation HTTP route is a
server integration of exact lookup, not a benchmark or runtime admission
integration. The existing `CapacityProfile` contract does not yet carry
benchmark metadata or distinguish draft/measured state; this module keeps that
evidence at its storage boundary rather than changing the contract.
Shared contracts now validate strict sample numerics, immutable descriptors and
model-specific context/bucket shapes. The registry adds measurement completeness,
20% reserve and evidence-integrity requirements. Synthetic experimental profiles
remain valid shared contracts but are not approved registry measurements.

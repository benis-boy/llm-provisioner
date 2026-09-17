# Offline artifact volume

`services.llm.provisioning.volume.provision` creates the bounded, offline input
volume used by a future runtime.  It copies only the exact file sets in
`artifacts.SPECS` for one or more known models; it does not copy an image,
download anything, parse weights, or claim semantic tokenizer validation.

The destination contains immutable, content-addressed directories named by the
schema-1 manifest digest.  Each has `manifest.json` and stable roots such as
`models/CoEdIT`, independent of source and host paths.  `current` is a relative
symlink selected atomically.  Provisioning is serialized with a local lock,
uses same-filesystem staging and atomic rename, fsyncs files and directories,
and verifies sizes, hashes, and exact transitive file sets before selection.
Equal inputs reuse a verified digest.  A corrupt or changed existing digest
fails closed; it is never overwritten.  Only tool-owned incomplete staging
directories are removed.  Older complete digests and unrelated files are not
removed.

Example:

```sh
python tools/provision_artifacts.py --output /srv/llm-artifacts \
  --model SmolLM=/offline/SmolLM \
  --model CoEdIT=/offline/CoEdIT
```

The command emits the digest, concise counts, and runtime paths, never the full
manifest or source JSON.  Inputs must be ordinary files beneath non-overlapping
source roots; unsafe paths, symlink escapes, duplicate model arguments, unknown
models, missing files, and unsafe existing selections fail closed.

Deployment should mount the completed destination read-only for runtime use.
Runtime code must not download or mutate it.  This artifact identity does not
claim approved dependency pins, measured capacity profiles, full transitive
semantic validation, or production GPU proof; those remain separate gates.

## Read-only verification HTTP slice

`services.llm.provisioning.http.create_app` exposes only `POST
/provisioning/verify-artifacts`. Startup receives an operator-controlled map
of nonempty volume IDs to absolute roots; clients cannot provide filesystem
paths or imports. The request is `{volumeId, expectedManifestSha256?}` and
requires a nonempty `Idempotency-Key`. The key is correlation only: every
request rechecks bytes and no readiness/result is cached.

The bounded response contains the selected digest, sorted model summaries,
`verified: true`, and `verificationScope: selected-file-integrity`. It does
not claim semantic model validity, readiness, GPU capacity, or production
proof. Manifest and file errors are sanitized into `{code,message,retryable}`.
Hashing runs off the event loop with bounded workers and request body size,
overload `429`, and timeout `504`. A timeout or disconnect does not release a
worker slot until hashing exits; filesystem hashing is not interruptible at
arbitrary calls. Shutdown awaits workers under the runtime supervisor's
cleanup policy.

The destination is an operator-controlled trusted filesystem: provisioning
serializes cooperating invocations and rejects observed corrupt selections, but
does not support an adversary concurrently changing destination files or
directory entries.  Source files are opened with no-follow descriptors and
verified before selection, so source symlinks and copied-content changes fail
closed.

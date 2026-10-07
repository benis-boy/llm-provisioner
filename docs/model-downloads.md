# On-demand core model downloads

`tools/download_models.py` provisions the exact offline adapter inputs for
SmolLM, CoEdIT, and GECToR only. It does not run at application startup and
does not import model adapters. Downloads are streamed from immutable
source revisions and checked against the selected compatibility
manifest's byte size and SHA-256 before an atomic file commit. The small
SmolLM `Modelfile` is generated locally as exactly
`FROM ./SmolLM2-1.7B-Instruct-Q8_0.gguf\n`; it contains no registry template
metadata.

From the repository root, explicitly start the online download when the roughly
6.6 GB (decimal) of selected artifacts is wanted:

```sh
.venv/bin/python tools/download_models.py
```

The default destination is ignored `LLMs/project-models/`, laid out as
`SmolLM/`, `CoEdIT/`, and `GECToR/`. To select a different destination:

```sh
.venv/bin/python tools/download_models.py --output /path/to/project-models
```

The pinned GECToR model revision does not include `verb-form-vocab.txt`. That
selected file is supplied separately from the official `grammarly/gector`
repository at immutable commit
`3d41d2841512d2690cffce1b5ac6795fe9a0a5dd` (`data/verb-form-vocab.txt`). Its
4,390,076 bytes match the selected SHA-256 exactly. This source override is
pinned per file; all selected bytes are still verified before commit.

The downloader skips files already matching their pinned hash and size. It
preflights only remote files that are missing or being explicitly replaced, so
a complete verified local set can be re-run offline. It fails rather than
overwriting an invalid or unrelated existing file; inspect
and remove that file yourself, or explicitly opt in to replacement with
`--replace`. Temporary files are private siblings and are removed after
catchable interruption or verification failure. A force-kill can leave a
hidden `.part` sibling; it is never treated as a completed artifact and may be
removed manually. Destination roots and files may not be symlinks. The script
never deletes old files. When hard links are unavailable, no-replace commits
use Linux `renameat2(RENAME_NOREPLACE)` or Windows `os.rename`; if the platform
or filesystem cannot provide atomic no-replace behavior, the downloader fails
closed and asks for a supporting output filesystem. Explicit `--replace` uses
atomic replacement and should be serialized for a given destination. The default
is under `/LLMs` in `.gitignore`; if
`--output` points elsewhere, the caller is responsible for keeping downloaded
weights and other generated work out of source control.

These are exact artifact inputs, not capacity, GPU, deployment, or production
readiness evidence. To prepare the existing compatibility context after the
models are present, invoke its normal explicit preparation command (which can
also download its separate build dependencies and Ollama assets):

```sh
.venv/bin/python tools/compatibility/prepare.py \
  --smollm-root LLMs/project-models/SmolLM \
  --coedit-root LLMs/project-models/CoEdIT \
  --gector-root LLMs/project-models/GECToR \
  --output .compatibility/context --download
```

For direct artifact-volume provisioning instead, use the existing
`tools.provision_artifacts` entry point with the same roots:

```sh
.venv/bin/python tools/provision_artifacts.py \
  --output /path/to/artifact-volume \
  --model SmolLM=LLMs/project-models/SmolLM \
  --model CoEdIT=LLMs/project-models/CoEdIT \
  --model GECToR=LLMs/project-models/GECToR
```

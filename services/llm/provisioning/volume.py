"""Serialized, content-addressed provisioning of the small offline artifact set.

This is intentionally a file copier, not a model downloader or validator.  The
selected file lists in :mod:`artifacts` are the production boundary.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import uuid
import re
from typing import Mapping

from .artifacts import SPECS, manifest, verify_manifest


_STAGING_PREFIX = ".artifact-volume-staging-"
_LOCK_NAME = ".artifact-volume.lock"
_STAGING_MARKER = ".artifact-volume-staging"
_MANIFEST_LIMIT = 64 * 1024
_DIGEST_NAME = re.compile(r"^[0-9a-f]{64}$")


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _safe_relative(value: str) -> None:
    p = Path(value)
    if not value or p.is_absolute() or ".." in p.parts or "\\" in value:
        raise ValueError(f"unsafe artifact path: {value!r}")


def _safe_destination(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("destination must be an absolute, non-traversing path")
    for parent in (path, *path.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError("destination contains a symlink")


def _digest(document: Mapping[str, object]) -> str:
    value = document.get("manifest_sha256")
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid manifest digest")
    return value


def _source_ancestors(path: Path) -> None:
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError("source root has a symlink ancestor")
        current = current.parent


def _source_file(root: Path, relative: str) -> tuple[int, os.stat_result]:
    """Open a selected flat file relative to an O_NOFOLLOW source directory."""
    _safe_relative(relative)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        fd = os.open(relative, flags, dir_fd=root_fd)
    finally:
        os.close(root_fd)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"source artifact is not a regular file: {relative}")
        return fd, metadata
    except Exception:
        os.close(fd)
        raise


def _read_source(root: Path, relative: str) -> tuple[int, str]:
    fd, initial = _source_file(root, relative)
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "rb", closefd=True) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        final = os.stat(root / relative, follow_symlinks=False)
        if final.st_ino != initial.st_ino or final.st_dev != initial.st_dev or final.st_size != initial.st_size:
            raise ValueError(f"source changed during read: {relative}")
        return initial.st_size, digest.hexdigest()
    except Exception:
        raise


def _validate_selected(document: Mapping[str, object]) -> None:
    """Stricter production validation than the legacy compatibility helper."""
    if (set(document) != {"schema", "models", "manifest_sha256"}
            or type(document.get("schema")) is not int or document.get("schema") != 1):
        raise ValueError("invalid artifact manifest")
    models = document.get("models")
    if not isinstance(models, Mapping) or not models or set(models) - set(SPECS):
        raise ValueError("manifest contains an unknown model")
    for model_id, entry in models.items():
        if not isinstance(model_id, str) or not isinstance(entry, Mapping):
            raise ValueError("invalid model manifest entry")
        if entry.get("model_id") != model_id or entry.get("root") != f"models/{model_id}":
            raise ValueError("manifest has an unstable runtime root")
        files = entry.get("files")
        if not isinstance(files, list) or [x.get("path") for x in files if isinstance(x, Mapping)] != list(SPECS[model_id]):
            raise ValueError(f"manifest file set is not exact for {model_id}")
        if entry.get("required_count") != len(SPECS[model_id]) or entry.get("present_count") != len(SPECS[model_id]) or entry.get("missing") != []:
            raise ValueError("manifest counts are not exact")
    expected = manifest(models)  # also checks canonical digest and entry schema
    if expected != dict(document):
        raise ValueError("manifest is not canonical")


def _verify_volume(directory: Path, document: Mapping[str, object], directory_name: str | None = None) -> None:
    _validate_selected(document)
    if directory_name is None:
        directory_name = directory.name
    if directory_name != _digest(document):
        raise ValueError("artifact directory does not match manifest digest")
    if {p.name for p in directory.iterdir()} != {"manifest.json", "models"}:
        raise ValueError("artifact volume contains unexpected top-level entries")
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("unsafe artifact manifest file")
    models_dir = directory / "models"
    if models_dir.is_symlink() or not models_dir.is_dir():
        raise ValueError("unsafe artifact models directory")
    roots = {model: directory / str(entry["root"]) for model, entry in document["models"].items()}  # type: ignore[index]
    if {p.name for p in models_dir.iterdir()} != set(roots):
        raise ValueError("artifact models directory contains unexpected entries")
    for model, root in roots.items():
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"invalid artifact model directory: {model}")
        selected = set(SPECS[model])
        actual = {p.name for p in root.iterdir()}
        if actual != selected or any(p.is_symlink() or not p.is_file() for p in root.iterdir()):
            raise ValueError(f"artifact file set is not exact for {model}")
    verify_manifest(document, {m: str(p) for m, p in roots.items()})


def _read_selected_manifest(path: Path) -> dict:
    """Read the small selected manifest without following its final symlink."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("unsafe artifact manifest file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        data = os.read(fd, _MANIFEST_LIMIT + 1)
    finally:
        os.close(fd)
    if len(data) > _MANIFEST_LIMIT:
        raise ValueError("artifact manifest is too large")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate manifest key")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-finite manifest value")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid artifact manifest") from exc
    if not isinstance(value, dict):
        raise ValueError("invalid artifact manifest")
    return value


def verify_current(destination: str | Path) -> dict:
    """Verify the operator-selected volume and return only deterministic counts.

    This is deliberately read-only.  It snapshots the relative digest selected
    by ``current`` before hashing and rejects a selection change observed after
    hashing, rather than claiming the result is still current.
    """
    dest = Path(destination)
    _safe_destination(dest)
    if not dest.exists() or dest.is_symlink() or not dest.is_dir():
        raise ValueError("unsafe artifact volume")
    current = dest / "current"
    if not current.is_symlink():
        raise ValueError("missing current artifact selection")
    selected = os.readlink(current)
    if selected != Path(selected).name or not _DIGEST_NAME.fullmatch(selected):
        raise ValueError("unsafe current target")
    selected_dir = dest / selected
    if selected_dir.is_symlink() or not selected_dir.is_dir():
        raise ValueError("current target is not a verified directory")
    document = _read_selected_manifest(selected_dir / "manifest.json")
    _verify_volume(selected_dir, document, selected)
    if not current.is_symlink() or os.readlink(current) != selected:
        raise ValueError("current selection changed during verification")
    models = []
    for model_id in sorted(document["models"]):
        entry = document["models"][model_id]
        models.append({"modelId": model_id, "fileCount": len(entry["files"]),
                       "totalBytes": sum(item["size"] for item in entry["files"])})
    return {"volumeId": None, "manifestSha256": selected, "models": models,
            "verified": True, "verificationScope": "selected-file-integrity"}


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    data = (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _owned_staging(directory: Path) -> bool:
    """Return whether a staging directory has this tool's durable ownership mark."""
    marker = directory / _STAGING_MARKER
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")) == {"owned": True}
    except (OSError, json.JSONDecodeError):
        return False


def provision(model_roots: Mapping[str, str | Path], destination: str | Path) -> dict:
    """Provision all ``model_roots`` into ``destination`` and select them.

    The returned document is the consumer manifest.  The operation is serialized
    by a lock in the destination and never replaces an existing digest directory.
    """
    if not model_roots or any(model not in SPECS for model in model_roots):
        raise ValueError("one or more known model roots are required")
    dest = Path(destination)
    _safe_destination(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or not dest.is_dir():
        raise ValueError("unsafe destination")
    lock_path = dest / _LOCK_NAME
    if lock_path.exists() or os.path.lexists(lock_path):
        if lock_path.is_symlink() or not lock_path.is_file():
            raise ValueError("unsafe provisioning lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            for item in dest.iterdir():
                if item.is_dir() and not item.is_symlink() and item.name.startswith(_STAGING_PREFIX):
                    if _owned_staging(item):
                        shutil.rmtree(item)
            entries = {}
            source_paths = {}
            for model in sorted(model_roots):
                source_input = Path(model_roots[model])
                _source_ancestors(source_input)
                if source_input.is_symlink():
                    raise ValueError(f"invalid source root: {model}")
                source = source_input.resolve()
                if not source.is_dir():
                    raise ValueError(f"invalid source root: {model}")
                if source == dest.resolve() or dest.resolve() in source.parents or source in dest.resolve().parents:
                    raise ValueError("source and destination overlap")
                if any(source == other or source in other.parents or other in source.parents for other in source_paths.values()):
                    raise ValueError("source roots overlap")
                source_paths[model] = source
                files = []
                for relative in SPECS[model]:
                    try:
                        size, file_digest = _read_source(source, relative)
                    except OSError as exc:
                        raise ValueError(f"unable to read source artifact: {model}/{relative}") from exc
                    files.append({"path": relative, "size": size, "sha256": file_digest})
                entry = {
                    "model_id": model,
                    "root": f"models/{model}",
                    "required_count": len(SPECS[model]),
                    "present_count": len(SPECS[model]),
                    "missing": [],
                    "files": files,
                }
                entries[model] = entry
            document = manifest(entries)
            _validate_selected(document)
            digest = _digest(document)
            final = dest / digest
            if final.exists() or final.is_symlink():
                if final.is_symlink() or not final.is_dir():
                    raise ValueError("existing artifact digest is corrupt")
                try:
                    existing = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
                    if existing != document:
                        raise ValueError("existing artifact digest differs from requested manifest")
                    _verify_volume(final, existing)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError("existing artifact digest is corrupt") from exc
            else:
                staging = dest / (_STAGING_PREFIX + uuid.uuid4().hex)
                staging.mkdir()
                try:
                    _write_json(staging / _STAGING_MARKER, {"owned": True})
                    for model in sorted(source_paths):
                        target = staging / "models" / model
                        target.mkdir(parents=True)
                        for relative in SPECS[model]:
                            source = source_paths[model]
                            out = target / relative
                            fd, _ = _source_file(source, relative)
                            try:
                                with os.fdopen(fd, "rb") as stream, out.open("xb") as written:
                                    shutil.copyfileobj(stream, written)
                                    written.flush()
                                    os.fsync(written.fileno())
                            except Exception:
                                try:
                                    os.close(fd)
                                except OSError:
                                    pass
                                raise
                        _fsync_directory(target)
                    _write_json(staging / "manifest.json", document)
                    (staging / _STAGING_MARKER).unlink()
                    _fsync_directory(staging / "models")
                    _fsync_directory(staging)
                    _verify_volume(staging, document, digest)
                    os.replace(staging, final)
                    _fsync_directory(dest)
                    _verify_volume(final, document)
                except Exception:
                    if staging.exists():
                        shutil.rmtree(staging)
                    raise
            current = dest / "current"
            if os.path.lexists(current):
                if not current.is_symlink():
                    raise ValueError("unsafe current target")
                prior = os.readlink(current)
                if Path(prior).name != prior or len(prior) != 64 or any(c not in "0123456789abcdef" for c in prior):
                    raise ValueError("unsafe current target")
                prior_dir = dest / prior
                if not prior_dir.is_dir() or prior_dir.is_symlink():
                    raise ValueError("current target is not a verified directory")
                try:
                    prior_manifest = json.loads((prior_dir / "manifest.json").read_text(encoding="utf-8"))
                    _verify_volume(prior_dir, prior_manifest)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError("current target is corrupt") from exc
            temporary = dest / (".current-" + uuid.uuid4().hex)
            try:
                os.symlink(digest, temporary)
                os.replace(temporary, current)
                _fsync_directory(dest)
                return document
            except Exception:
                if os.path.lexists(temporary):
                    temporary.unlink()
                raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

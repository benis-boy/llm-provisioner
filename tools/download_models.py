#!/usr/bin/env python3
"""Download the small, pinned offline model set on explicit operator request."""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import BinaryIO, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


CATALOG_PATH = Path(__file__).with_name("model_downloads.json")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "LLMs" / "project-models"
CHUNK_SIZE = 1024 * 1024
MODELFILE_CONTENT = b"FROM ./SmolLM2-1.7B-Instruct-Q8_0.gguf\n"
MODELFILE_NAME = "Modelfile"

# Keep this tool's input boundary in sync with the light-weight artifact
# contract. Tests compare this set with services.llm.provisioning.artifacts.SPECS.
EXPECTED_FILES = {
    "SmolLM": {"SmolLM2-1.7B-Instruct-Q8_0.gguf", MODELFILE_NAME},
    "CoEdIT": {
        "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "spiece.model", "generation_config.json",
    },
    "GECToR": {
        "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "verb-form-vocab.txt",
    },
}


class DownloadError(RuntimeError):
    """A pinned download or destination validation failed."""


class AtomicNoReplaceUnsupported(RuntimeError):
    """The host filesystem cannot atomically rename without replacing a target."""


def _validate_catalog(catalog: Mapping[str, object]) -> dict[str, dict[str, object]]:
    if set(catalog) != {"models"} or not isinstance(catalog["models"], dict):
        raise DownloadError("invalid model download catalog")
    models = catalog["models"]
    if set(models) != set(EXPECTED_FILES):
        raise DownloadError("catalog must contain exactly SmolLM, CoEdIT, and GECToR")
    result: dict[str, dict[str, object]] = {}
    for model, raw in models.items():
        if not isinstance(raw, dict) or set(raw) != {"repo", "revision", "files"}:
            raise DownloadError(f"invalid catalog entry for {model}")
        repo, revision, files = raw["repo"], raw["revision"], raw["files"]
        if (not isinstance(repo, str) or not repo or not isinstance(revision, str)
                or len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision)
                or not isinstance(files, list)):
            raise DownloadError(f"invalid source or file list for {model}")
        names: set[str] = set()
        for item in files:
            if not isinstance(item, dict) or set(item) not in (
                    {"path", "size", "sha256"}, {"path", "size", "sha256", "source"}):
                raise DownloadError(f"invalid file entry for {model}")
            name, size, digest = item["path"], item["size"], item["sha256"]
            if (not isinstance(name, str) or not name or Path(name).name != name
                    or name in {".", ".."}):
                raise DownloadError(f"invalid filename in {model} catalog")
            if (not isinstance(size, int) or isinstance(size, bool) or size < 0
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(ch not in "0123456789abcdef" for ch in digest)):
                raise DownloadError(f"invalid pinned size or digest for {model}/{name}")
            if "source" in item:
                source = item["source"]
                if not isinstance(source, dict) or set(source) != {"url"}:
                    raise DownloadError(f"invalid source override for {model}/{name}")
                source_url = source["url"]
                parsed = urlsplit(source_url) if isinstance(source_url, str) else None
                parts = parsed.path.split("/") if parsed is not None else []
                if (parsed is None or parsed.scheme != "https"
                        or parsed.hostname != "raw.githubusercontent.com"
                        or parsed.query or parsed.fragment or len(parts) != 6
                        or parts[1:3] != ["grammarly", "gector"]
                        or len(parts[3]) != 40
                        or any(ch not in "0123456789abcdef" for ch in parts[3])
                        or parts[4:] != ["data", "verb-form-vocab.txt"]
                        or model != "GECToR" or name != "verb-form-vocab.txt"):
                    raise DownloadError(f"invalid source override for {model}/{name}")
            if name in names:
                raise DownloadError(f"duplicate catalog file {model}/{name}")
            names.add(name)
        if names != EXPECTED_FILES[model]:
            raise DownloadError(f"catalog files do not match the supported {model} artifact set")
        result[model] = raw
    return result


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, dict[str, object]]:
    try:
        with path.open(encoding="utf-8") as source:
            catalog = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise DownloadError(f"cannot read model catalog {path}: {exc}") from exc
    if not isinstance(catalog, dict):
        raise DownloadError("invalid model download catalog")
    return _validate_catalog(catalog)


def _ensure_directory(path: Path) -> Path:
    """Create path components without following any symlink in the destination."""
    requested = path if path.is_absolute() else Path.cwd() / path
    current = Path(requested.anchor)
    for part in requested.parts[1:]:
        if part in {"", "."}:
            continue
        if part == "..":
            current = current.parent
            continue
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o755)
            except FileExistsError:
                info = current.lstat()
            else:
                continue
        if stat.S_ISLNK(info.st_mode):
            raise DownloadError(f"destination contains a symlink: {current}")
        if not stat.S_ISDIR(info.st_mode):
            raise DownloadError(f"destination component is not a directory: {current}")
    return current


def _regular_destination(path: Path) -> os.stat_result | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode):
        raise DownloadError(f"refusing symlink destination: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise DownloadError(f"destination is not a regular file: {path}")
    return info


def _matches(path: Path, expected_size: int, expected_hash: str) -> bool:
    info = _regular_destination(path)
    if info is None or info.st_size != expected_size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_hash


def _copy_response(response: BinaryIO, temporary: Path, size: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    written = 0
    with temporary.open("wb") as output:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            written += len(chunk)
            if written > size:
                raise DownloadError(f"download exceeded pinned size ({size} bytes)")
            digest.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    return written, digest.hexdigest()


def _hardlink_unsupported(exc: OSError) -> bool:
    unsupported = {errno.EACCES, errno.EPERM, errno.ENOSYS, errno.EXDEV}
    for name in ("ENOTSUP", "EOPNOTSUPP"):
        value = getattr(errno, name, None)
        if value is not None:
            unsupported.add(value)
    return exc.errno in unsupported or getattr(exc, "winerror", None) in {1, 50}


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically rename source only if target does not exist, or fail closed."""
    if os.name == "nt":
        # Windows os.rename does not replace an existing destination.
        os.rename(source, target)
        return
    if not sys.platform.startswith("linux"):
        raise AtomicNoReplaceUnsupported("atomic no-replace rename is unavailable on this platform")

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise AtomicNoReplaceUnsupported("libc does not provide renameat2") from exc
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)  # RENAME_NOREPLACE
    if result == 0:
        return

    code = ctypes.get_errno()
    if code == errno.EEXIST:
        raise FileExistsError(code, os.strerror(code), target)
    unsupported = {errno.ENOSYS, errno.EINVAL}
    for name in ("ENOTSUP", "EOPNOTSUPP"):
        value = getattr(errno, name, None)
        if value is not None:
            unsupported.add(value)
    if code in unsupported:
        raise AtomicNoReplaceUnsupported(
            f"filesystem does not support renameat2(RENAME_NOREPLACE): {os.strerror(code)}"
        )
    raise OSError(code, os.strerror(code), target)


def download_file(
    target: Path,
    spec: Mapping[str, object],
    *,
    url: str | None = None,
    content: bytes | None = None,
    replace: bool = False,
) -> bool:
    """Fetch and atomically install one pinned regular file; return True if fetched."""
    parent = _ensure_directory(target.parent)
    target = parent / target.name
    size, expected_hash = spec["size"], spec["sha256"]
    assert isinstance(size, int) and isinstance(expected_hash, str)

    existing = _regular_destination(target)
    if existing is not None:
        if _matches(target, size, expected_hash):
            print(f"verified existing {target}")
            return False
        if not replace:
            raise DownloadError(f"existing file does not match pinned artifact; use --replace: {target}")

    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        if content is not None:
            with temporary.open("wb") as output_stream:
                output_stream.write(content)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            actual_size, actual_hash = len(content), hashlib.sha256(content).hexdigest()
        else:
            if url is None:
                raise DownloadError("a URL is required for remote artifacts")
            try:
                request = Request(url, headers={"User-Agent": "llm-provider-model-download/1"})
                with urlopen(request, timeout=60) as response:
                    actual_size, actual_hash = _copy_response(response, temporary, size)
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                raise DownloadError(f"download failed for {target.name} ({type(exc).__name__})") from exc
        if actual_size != size or actual_hash != expected_hash:
            raise DownloadError(
                f"pinned artifact mismatch for {target.name}: expected {size} bytes/{expected_hash}, "
                f"received {actual_size} bytes/{actual_hash}"
            )

        os.chmod(temporary, 0o644)
        # A hard link is an atomic no-clobber commit. --replace is the only path
        # that is allowed to replace a pre-existing invalid regular file.
        try:
            os.link(temporary, target)
        except FileExistsError:
            current = _regular_destination(target)
            if current is not None and _matches(target, size, expected_hash):
                return False
            if not replace:
                raise DownloadError(f"destination appeared during download; refusing overwrite: {target}")
            _regular_destination(target)
            os.replace(temporary, target)
        except OSError as exc:
            if not _hardlink_unsupported(exc):
                raise DownloadError(f"cannot commit {target.name} ({type(exc).__name__})") from exc
            if replace:
                current = _regular_destination(target)
                if current is not None and _matches(target, size, expected_hash):
                    return False
                _regular_destination(target)
                # --replace explicitly permits atomic replacement; callers must
                # serialize replacement runs for a destination.
                os.replace(temporary, target)
            else:
                try:
                    _rename_noreplace(temporary, target)
                except FileExistsError:
                    current = _regular_destination(target)
                    if current is not None and _matches(target, size, expected_hash):
                        return False
                    raise DownloadError(
                        f"destination appeared during download; refusing overwrite: {target}"
                    ) from None
                except AtomicNoReplaceUnsupported as commit_error:
                    raise DownloadError(
                        "hard links are unavailable and this platform/filesystem cannot commit "
                        "without replacement; choose an output filesystem supporting hard links "
                        "or atomic no-replace rename, or use --replace only if replacement is intended"
                    ) from commit_error
                except OSError as commit_error:
                    raise DownloadError(
                        f"cannot atomically commit {target.name} ({type(commit_error).__name__})"
                    ) from commit_error
        print(f"downloaded {target}")
        return True
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _file_url(repo: str, revision: str, filename: str) -> str:
    encoded_repo = "/".join(quote(component, safe="") for component in repo.split("/"))
    encoded_name = "/".join(quote(component, safe="") for component in filename.split("/"))
    return f"https://huggingface.co/{encoded_repo}/resolve/{revision}/{encoded_name}?download=true"


def _artifact_url(entry: Mapping[str, object], spec: Mapping[str, object]) -> str:
    override = spec.get("source")
    if isinstance(override, dict):
        return override["url"]
    repo, revision, filename = entry["repo"], entry["revision"], spec["path"]
    assert isinstance(repo, str) and isinstance(revision, str) and isinstance(filename, str)
    return _file_url(repo, revision, filename)


def _preflight_sources(pending: list[tuple[str, Mapping[str, object], str]]) -> None:
    """Check missing/replacement sources before transferring large artifacts."""
    for model, spec, url in pending:
        try:
            request = Request(url, method="HEAD", headers={"User-Agent": "llm-provider-model-download/1"})
            with urlopen(request, timeout=30) as response:
                content_length = response.headers.get("Content-Length")
        except HTTPError as exc:
            raise DownloadError(f"pinned source for {model}/{spec['path']} returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise DownloadError(
                f"cannot verify pinned source for {model}/{spec['path']} ({type(exc).__name__})"
            ) from exc
        if content_length is not None:
            try:
                remote_size = int(content_length)
            except ValueError as exc:
                raise DownloadError(f"invalid Content-Length for {model}/{spec['path']}") from exc
            if remote_size != spec["size"]:
                raise DownloadError(
                    f"pinned upstream size mismatch for {model}/{spec['path']}: "
                    f"expected {spec['size']}, source reports {remote_size}"
                )


def download_models(output: Path = DEFAULT_OUTPUT, *, replace: bool = False) -> None:
    catalog = load_catalog()
    output = _ensure_directory(output)
    pending: list[tuple[Path, Mapping[str, object], str | None, bytes | None, str]] = []
    preflight: list[tuple[str, Mapping[str, object], str]] = []
    for model in ("SmolLM", "CoEdIT", "GECToR"):
        entry = catalog[model]
        model_root = _ensure_directory(output / model)
        for spec in entry["files"]:  # type: ignore[union-attr]
            assert isinstance(spec, dict)
            filename = spec["path"]
            assert isinstance(filename, str)
            target = model_root / filename
            content = MODELFILE_CONTENT if model == "SmolLM" and filename == MODELFILE_NAME else None
            if _matches(target, spec["size"], spec["sha256"]):
                continue
            if _regular_destination(target) is not None and not replace:
                raise DownloadError(f"existing file does not match pinned artifact; use --replace: {target}")
            if content is not None:
                pending.append((target, spec, None, content, model))
            else:
                url = _artifact_url(entry, spec)
                pending.append((target, spec, url, None, model))
                preflight.append((model, spec, url))

    _preflight_sources(preflight)
    for target, spec, url, content, _model in pending:
        download_file(target, spec, url=url, content=content, replace=replace)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"model root (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--replace", action="store_true",
                        help="atomically replace existing files that fail pinned verification")
    args = parser.parse_args(argv)
    try:
        download_models(args.output, replace=args.replace)
    except DownloadError as exc:
        parser.exit(1, f"download_models: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Assemble the small, verified offline Docker build input set.

This intentionally treats the two compatibility lock files as independent
contracts: a version or hash disagreement is an error, never something to
resolve opportunistically.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import tempfile
import zipfile
from email.parser import Parser
from dataclasses import dataclass
from pathlib import Path

WHEEL = re.compile(
    r"^(?P<distribution>[A-Za-z0-9_.]+)-(?P<version>[0-9][^-]*)(?:-(?P<build>[0-9][0-9A-Za-z_]*))?"
    r"-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^-]+)\.whl$"
)
DEFAULT_OLLAMA_SHA256 = "fa9608a428c7d6bd46fce69f7016242feedf60fba1a57d3ee767257b071216f6"
CUDA_DIGEST = "ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867"


@dataclass(frozen=True)
class LockedWheel:
    name: str
    version: str
    digest: str


def _identity(value: str) -> str:
    """PEP 503 project identity (also used for wheel distribution names)."""
    return re.sub(r"[-_.]+", "-", value).lower()


def _wheel_metadata(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        # A wheel's own metadata is at archive root.  Some legitimate wheels
        # (notably setuptools) vendor other distributions beneath package
        # directories; those nested METADATA files are not this wheel's
        # identity contract.
        names = [name for name in archive.namelist()
                 if re.fullmatch(r"[^/]+\.dist-info/METADATA", name)]
        if len(names) != 1:
            raise ValueError(f"wheel must contain one dist-info METADATA: {path.name}")
        metadata = Parser().parsestr(archive.read(names[0]).decode("utf-8"))
    name, version = metadata.get("Name"), metadata.get("Version")
    if not name or not version:
        raise ValueError(f"wheel metadata lacks Name/Version: {path.name}")
    return name, version


def _wheel_identity(path: Path) -> tuple[str, str]:
    match = WHEEL.fullmatch(path.name)
    if not match:
        raise ValueError(f"invalid wheel filename: {path.name}")
    metadata_name, metadata_version = _wheel_metadata(path)
    distribution = match.group("distribution")
    filename_version = match.group("version")
    # Wheel filenames escape distribution punctuation as underscores.  Do not
    # split the escaped prefix: the version is already a distinct grammar
    # field, and can legitimately contain dots or a local-version suffix.
    if _identity(distribution) != _identity(metadata_name):
        raise ValueError(f"wheel filename/metadata mismatch: {path.name}")
    if filename_version != metadata_version:
        raise ValueError(f"wheel filename/metadata mismatch: {path.name}")
    return _identity(metadata_name), metadata_version


def _lock(path: Path) -> dict[str, LockedWheel]:
    result: dict[str, LockedWheel] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or not re.fullmatch(r"--hash=sha256:[0-9a-fA-F]{64}", parts[1]):
            raise ValueError(f"unsupported lock line in {path}: {line}")
        requirement = parts[0]
        if requirement.count("==") != 1:
            raise ValueError(f"unlocked requirement in {path}: {line}")
        name, version = requirement.split("==", 1)
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9!+._-]*", version)):
            raise ValueError(f"invalid locked requirement in {path}: {line}")
        key = _identity(name)
        wheel = LockedWheel(key, version, parts[1].split(":", 1)[1].lower())
        prior = result.get(key)
        if prior is not None and prior != wheel:
            raise ValueError(f"duplicate lock conflict for {name}")
        result[key] = wheel
    return result


def union_locks(paths: tuple[Path, ...]) -> dict[str, LockedWheel]:
    merged: dict[str, LockedWheel] = {}
    for path in paths:
        for key, wheel in _lock(path).items():
            if key in merged and merged[key] != wheel:
                raise ValueError(f"lock conflict for {wheel.name}")
            merged[key] = wheel
    return merged


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _select_wheels(wheelhouses: tuple[Path, ...], locks: dict[str, LockedWheel]) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for wheelhouse in wheelhouses:
        for candidate in wheelhouse.glob("*.whl"):
            filename = WHEEL.fullmatch(candidate.name)
            # Filter by filename fields first.  Wheelhouses may contain
            # unrelated or malformed payloads, which must not be opened.
            if filename is None:
                continue
            locked = locks.get(_identity(filename.group("distribution")))
            if locked is None or filename.group("version") != locked.version:
                continue
            if _digest(candidate) != locked.digest:
                raise ValueError(f"hash mismatch for locked wheel {candidate.name}")
            try:
                key, version = _wheel_identity(candidate)
            except ValueError:
                raise ValueError(f"invalid locked wheel: {candidate.name}") from None
            if key != _identity(filename.group("distribution")) or version != locked.version:
                raise ValueError(f"wheel filename/metadata mismatch: {candidate.name}")
            if key in selected and selected[key].name != candidate.name:
                raise ValueError(f"ambiguous wheel for locked package {locked.name}")
            selected[key] = candidate
    missing = sorted(set(locks) - set(selected))
    if missing:
        raise FileNotFoundError("missing locked wheels: " + ", ".join(missing))
    return selected


def assemble(*, locks: tuple[Path, ...], wheelhouses: tuple[Path, ...], ollama: Path,
             output: Path, ollama_sha256: str = DEFAULT_OLLAMA_SHA256) -> None:
    """Atomically replace *output* with selected wheels, archive, and metadata."""
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("output must be a directory, not a symlink or file")
    output_resolved = output.resolve()
    sources = [ollama.resolve(), *[p.resolve() for p in locks], *[p.resolve() for p in wheelhouses]]
    if any(source == output_resolved or source in output_resolved.parents or output_resolved in source.parents
           for source in sources):
        raise ValueError("output overlaps an input source")
    merged = union_locks(locks)
    selected = _select_wheels(wheelhouses, merged)
    if _digest(ollama) != ollama_sha256:
        raise ValueError("Ollama archive hash mismatch")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=parent))
    try:
        wheel_dir = temporary / "wheelhouse"
        wheel_dir.mkdir()
        for key, path in sorted(selected.items(), key=lambda item: item[1].name):
            destination = wheel_dir / path.name
            shutil.copy2(path, destination)
            if _digest(destination) != merged[key].digest:
                raise ValueError(f"staged wheel changed during copy: {path.name}")
            staged_key, version = _wheel_identity(destination)
            if staged_key != key or version != merged[key].version:
                raise ValueError(f"staged wheel changed during copy: {path.name}")
        archive_destination = temporary / "ollama-linux-amd64.tgz"
        shutil.copy2(ollama, archive_destination)
        if _digest(archive_destination) != ollama_sha256:
            raise ValueError("staged Ollama archive changed during copy")
        lock_text = "\n".join(
            f"{wheel.name}=={wheel.version} --hash=sha256:{wheel.digest}"
            for wheel in sorted(merged.values(), key=lambda item: item.name.lower())
        ) + "\n"
        (temporary / "requirements.lock").write_text(lock_text, encoding="utf-8")
        (temporary / "provenance.txt").write_text(
            "format=llm-provider-deployment-inputs-v1\n"
            f"wheel_count={len(selected)}\n"
            f"ollama_sha256={ollama_sha256}\n"
            f"cuda_runtime_digest={CUDA_DIGEST}\n"
            "models=excluded\nsource_json=excluded\n", encoding="utf-8")
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=parent))
            backup.rmdir()
            os.replace(output, backup)
            try:
                os.replace(temporary, output)
            except BaseException:
                os.replace(backup, output)
                raise
            try:
                shutil.rmtree(backup)
            except BaseException as exc:
                raise RuntimeError("output committed; owned backup cleanup failed") from exc
        else:
            os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", action="append", type=Path, required=True)
    parser.add_argument("--wheelhouse", action="append", type=Path, required=True)
    parser.add_argument("--ollama", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(locks=tuple(args.lock), wheelhouses=tuple(args.wheelhouse),
             ollama=args.ollama, output=args.output)


if __name__ == "__main__":
    main()

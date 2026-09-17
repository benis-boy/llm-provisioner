"""Small, deterministic inventory helpers for the compatibility spike.

This module deliberately hashes only adapter inputs.  It never walks an entire
model checkout and it never parses weight files.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ArtifactSpec:
    model_id: str
    root: Path
    required: tuple[str, ...]


SPECS: Mapping[str, tuple[str, ...]] = {
    "SmolLM": ("SmolLM2-1.7B-Instruct-Q8_0.gguf", "Modelfile"),
    "CoEdIT": (
        "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "spiece.model", "generation_config.json",
    ),
    "GECToR": (
        "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "verb-form-vocab.txt",
    ),
}
def spec(model_id: str, root: str | Path) -> ArtifactSpec:
    if model_id not in SPECS:
        raise ValueError(f"unknown model: {model_id}")
    return ArtifactSpec(model_id, Path(root), SPECS[model_id])


def _safe(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError(f"path escapes input root: {relative}")
    return path


def inventory(s: ArtifactSpec) -> dict[str, Any]:
    root = s.root.resolve()
    missing: list[str] = []
    files: list[dict[str, Any]] = []
    for relative in s.required:
        path = _safe(root, relative)
        if not path.is_file():
            missing.append(relative)
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append({"path": relative, "size": path.stat().st_size, "sha256": digest.hexdigest()})
    return {"model_id": s.model_id, "root": str(root), "required_count": len(s.required),
            "present_count": len(files), "missing": missing, "files": files}


def validate(result: Mapping[str, Any]) -> None:
    required_keys = {"model_id", "root", "required_count", "present_count", "missing", "files"}
    if set(result) != required_keys or not isinstance(result["model_id"], str) or not result["model_id"]:
        raise ValueError("invalid artifact entry schema")
    if not isinstance(result["root"], str) or not result["root"]:
        raise ValueError("invalid artifact root")
    if (not isinstance(result["required_count"], int) or isinstance(result["required_count"], bool)
            or not isinstance(result["present_count"], int) or isinstance(result["present_count"], bool)):
        raise ValueError("invalid artifact count")
    if not isinstance(result["missing"], list) or not all(isinstance(path, str) and path for path in result["missing"]):
        raise ValueError("invalid missing artifact list")
    if not isinstance(result["files"], list):
        raise ValueError("invalid artifact file list")
    if result.get("missing"):
        raise ValueError("missing required artifacts: " + ", ".join(result["missing"]))
    if result.get("present_count") != result.get("required_count"):
        raise ValueError("artifact count invariant failed")
    paths = []
    for item in result["files"]:
        if (not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}
                or not isinstance(item["path"], str) or not item["path"]):
            raise ValueError("invalid artifact file schema")
        if (not isinstance(item["size"], int) or isinstance(item["size"], bool) or item["size"] < 0
                or not isinstance(item["sha256"], str)):
            raise ValueError("invalid artifact file fields")
        paths.append(item["path"])
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate artifact path")
    for item in result["files"]:
        if len(item["sha256"]) != 64 or any(char not in "0123456789abcdef" for char in item["sha256"]):
            raise ValueError(f"invalid hash for {item['path']}")


def verify_manifest(document: Mapping[str, Any], roots: Mapping[str, str | Path] | None = None) -> None:
    """Verify the manifest digest and every selected file against its root."""
    if (set(document) != {"schema", "models", "manifest_sha256"}
            or type(document.get("schema")) is not int or document.get("schema") != 1
            or not isinstance(document.get("models"), Mapping)):
        raise ValueError("unsupported artifact manifest")
    supplied = document.get("manifest_sha256")
    body = {"schema": document["schema"], "models": document["models"]}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    if supplied != hashlib.sha256(canonical).hexdigest():
        raise ValueError("artifact manifest digest mismatch")
    for model_id, entry in document["models"].items():
        if not isinstance(model_id, str) or not isinstance(entry, Mapping) or entry.get("model_id") != model_id:
            raise ValueError("invalid model manifest entry")
        validate(entry)
        root = Path((roots or {}).get(entry["model_id"], entry["root"])).resolve()
        for item in entry["files"]:
            path = _safe(root, item["path"])
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if path.stat().st_size != item["size"] or digest.hexdigest() != item["sha256"]:
                raise ValueError(f"artifact changed: {entry['model_id']}/{item['path']}")


def manifest(entries: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    if not entries or any(not isinstance(model_id, str) or entry.get("model_id") != model_id for model_id, entry in entries.items()):
        raise ValueError("invalid model manifest entries")
    for entry in entries.values():
        validate(entry)
    body = {"schema": 1, "models": dict(sorted(entries.items()))}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return {**body, "manifest_sha256": hashlib.sha256(canonical).hexdigest()}

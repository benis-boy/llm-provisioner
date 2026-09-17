"""Strict reader for the small, offline bootstrap configuration.

The file is operator input, not an HTTP contract.  Keeping its shape here makes
unknown fields and accidental truthy values fail before any provider is made.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat as stat_module
from types import MappingProxyType
from typing import Any, Mapping

MAX_CONFIG_BYTES = 32 * 1024
MODEL_IDS = ("SmolLM", "CoEdIT", "GECToR")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_GPU = re.compile(r"^GPU-[A-Za-z0-9-]+$")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate configuration field")
        result[key] = value
    return result


def _text(value: Any, name: str, limit: int = 512) -> str:
    if (not isinstance(value, str) or not value or len(value.encode()) > limit or
            any(ord(char) < 0x20 or ord(char) == 0x7f for char in value)):
        raise ValueError(f"{name} must be bounded non-empty text")
    return value


def _path(value: Any, name: str) -> Path:
    value = _text(value, name, 4096)
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be an absolute non-traversing path")
    return path


@dataclass(frozen=True)
class ModelConfig:
    runtime_identity: str
    adapter_identity: str

    def __post_init__(self) -> None:
        _text(self.runtime_identity, "runtime_identity")
        _text(self.adapter_identity, "adapter_identity")


@dataclass(frozen=True)
class BootstrapConfig:
    gpu_uuid: str
    artifact_root: Path
    manifest_sha256: str
    profile_db: Path
    ollama_binary: Path
    ollama_home: Path
    ollama_port: int
    models: Mapping[str, ModelConfig]

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))
        if set(self.models) != set(MODEL_IDS) or any(
                type(self.models.get(model)) is not ModelConfig for model in MODEL_IDS):
            raise ValueError("models must contain exactly the supported model configurations")
        if not _GPU.fullmatch(self.gpu_uuid) or not _SHA.fullmatch(self.manifest_sha256):
            raise ValueError("invalid GPU or manifest identity")
        for value, name in ((self.artifact_root, "artifact_root"), (self.profile_db, "profile_db"),
                            (self.ollama_binary, "ollama_binary"), (self.ollama_home, "ollama_home")):
            if (not isinstance(value, Path) or not value.is_absolute() or ".." in value.parts or
                    any(ord(char) < 0x20 or ord(char) == 0x7f for char in str(value))):
                raise ValueError(f"{name} must be an absolute path")
        if type(self.ollama_port) is not int or not 1 <= self.ollama_port <= 65535:
            raise ValueError("invalid Ollama port")


def _document(data: Any) -> BootstrapConfig:
    expected = {"schema", "gpu_uuid", "artifact_root", "manifest_sha256", "profile_db",
                "ollama_binary", "ollama_home", "ollama_port", "models"}
    if not isinstance(data, dict) or set(data) != expected or type(data["schema"]) is not int or data["schema"] != 1:
        raise ValueError("configuration schema must be exactly version 1")
    gpu = _text(data["gpu_uuid"], "gpu_uuid", 128)
    if not _GPU.fullmatch(gpu):
        raise ValueError("gpu_uuid is not a physical GPU UUID")
    digest = _text(data["manifest_sha256"], "manifest_sha256", 64)
    if not _SHA.fullmatch(digest):
        raise ValueError("manifest_sha256 must be lowercase SHA-256")
    models = data["models"]
    if not isinstance(models, dict) or set(models) != set(MODEL_IDS):
        raise ValueError("models must contain exactly the three supported model IDs")
    parsed: dict[str, ModelConfig] = {}
    for model in MODEL_IDS:
        entry = models[model]
        if not isinstance(entry, dict) or set(entry) != {"runtime_identity", "adapter_identity"}:
            raise ValueError(f"invalid configuration for {model}")
        parsed[model] = ModelConfig(_text(entry["runtime_identity"], f"{model}.runtime_identity"),
                                    _text(entry["adapter_identity"], f"{model}.adapter_identity"))
    port = data["ollama_port"]
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("ollama_port must be an integer in the TCP port range")
    return BootstrapConfig(gpu, _path(data["artifact_root"], "artifact_root"), digest,
                           _path(data["profile_db"], "profile_db"),
                           _path(data["ollama_binary"], "ollama_binary"),
                           _path(data["ollama_home"], "ollama_home"), port,
                           MappingProxyType(parsed))


def load_config(path: str | Path) -> BootstrapConfig:
    target = Path(path)
    if not target.is_absolute() or target.is_symlink() or not target.is_file():
        raise ValueError("configuration file is unavailable or unsafe")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    try:
        fd = os.open(target, flags)
        stat = os.fstat(fd)
        if not stat_module.S_ISREG(stat.st_mode):
            raise ValueError("configuration file is not a regular file")
        chunks: list[bytes] = []
        remaining = MAX_CONFIG_BYTES + 1
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    except OSError as exc:
        raise ValueError("configuration file is unavailable or unsafe") from exc
    finally:
        if fd is not None:
            os.close(fd)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("configuration file exceeds 32 KiB")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite number")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("invalid configuration JSON") from exc
    return _document(value)

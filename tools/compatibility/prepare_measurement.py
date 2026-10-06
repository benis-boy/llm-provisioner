"""Prepare and verify a deterministic, transferable Phase 5 input bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from services.llm.bootstrap.config import load_config
from services.llm.bootstrap.measurement_matrix import MATRIX
from services.llm.provisioning.artifacts import SPECS, verify_manifest
from services.llm.provisioning.benchmark_requests import (
    canonical_measurement_fixtures, prepare_benchmark_request,
)
from services.llm.provisioning.volume import provision, verify_current
from services.llm.queue.contracts import ModelId

SCHEMA = 1
MAX_JSON_BYTES = 256 * 1024
RUNTIME_ROOT = Path("/var/lib/llm-measurement")
RUNTIME_PROFILE_DB = RUNTIME_ROOT / "profiles.sqlite"
RUNTIME_OLLAMA_HOME = RUNTIME_ROOT / "ollama"
CANONICAL_OLLAMA_BINARY = Path("/usr/bin/ollama")
MATRIX_IDS = (
    "smollm:context512",
    "coedit:p1:input128:output64:float16:beams1:nosample",
    "gector:p1:tokens128:keep0:min0:iterations1:batch1:float32",
)
MODELS = ("SmolLM", "CoEdIT", "GECToR")

# ``prepare.py`` emits a complete compatibility build context.  Measurement
# preparation intentionally consumes only manifest.json and models/, but the
# retained context is also used by the image build.  Keep this list explicit:
# accepting arbitrary context entries would make it too easy to accidentally
# bind build-only material to a measurement input.
PREPARE_CONTEXT_ENTRIES = frozenset({
    "requirements-candidate.txt", "Dockerfile", "spike.py", "rm_spike.py",
    "model_runtime.py", "input_bounds.py", "artifacts.py",
    "services", "wheelhouse", "requirements.lock", "ollama-linux-amd64.tgz",
    "provenance.json", ".compatibility-spike-owned",
})
PREPARE_CONTEXT_FILES = frozenset({
    "requirements-candidate.txt", "Dockerfile", "spike.py", "rm_spike.py",
    "model_runtime.py", "input_bounds.py", "artifacts.py", "requirements.lock",
    "ollama-linux-amd64.tgz", "provenance.json", ".compatibility-spike-owned",
})
PREPARE_HARNESS_FILES = frozenset({
    "services/__init__.py", "services/llm/__init__.py",
    "services/llm/providers/__init__.py", "services/llm/providers/input_bounds.py",
    "services/llm/queue/__init__.py", "services/llm/queue/contracts.py",
    "services/llm/resource_manager/__init__.py",
    "services/llm/resource_manager/contracts.py",
    "services/llm/resource_manager/protocol.py",
    "services/llm/resource_manager/state.py", "services/llm/resource_manager/core.py",
})


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path, *, limit: int = MAX_JSON_BYTES) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError("JSON input is unsafe")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("JSON input exceeds size limit")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def _is_python_cache(path: Path) -> bool:
    return path.name == "__pycache__" or path.suffix == ".pyc"


def _validate_python_cache(path: Path) -> None:
    """Validate ignored Python caches without making them preparation inputs."""
    if path.is_symlink():
        raise ValueError("measurement context Python cache contains a symlink")
    if path.name == "__pycache__" and not path.is_dir():
        raise ValueError("measurement context Python cache has unexpected shape")
    if path.suffix == ".pyc" and not path.is_file():
        raise ValueError("measurement context Python cache has unexpected shape")
    if path.is_file():
        return
    if path.is_dir():
        for child in path.iterdir():
            _validate_python_cache(child)
        return
    raise ValueError("measurement context Python cache contains a non-regular entry")


def _manifest(context: Path) -> dict:
    context = context.absolute()
    if context.is_symlink() or not context.is_dir():
        raise ValueError("measurement context must be a real directory")
    manifest_path = context / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("measurement context manifest is unsafe")
    value = read_json(manifest_path)
    if (not isinstance(value, dict) or set(value) != {"schema", "models", "manifest_sha256"}
            or type(value.get("schema")) is not int or value["schema"] != 1
            or not isinstance(value.get("models"), dict)):
        raise ValueError("context manifest schema is invalid")
    if set(value["models"]) != set(MODELS):
        raise ValueError("context manifest must select all three models")
    roots = {model: context / "models" / model for model in MODELS}
    entries = {item.name for item in context.iterdir()}
    unknown = {name for name in entries if not _is_python_cache(context / name)} \
        - {"manifest.json", "models"} - PREPARE_CONTEXT_ENTRIES
    if unknown:
        raise ValueError("measurement context contains unexpected entries")
    # The compatibility context is retained for two consumers.  Validate the
    # ignored build inputs as well, so a symlink cannot smuggle data into the
    # preparation environment even though those inputs are not identity-bound.
    def reject_symlinks(path: Path) -> None:
        if _is_python_cache(path):
            _validate_python_cache(path)
            return
        if path.is_symlink():
            raise ValueError("measurement context contains a symlink")
        if path.is_dir():
            for child in path.iterdir():
                reject_symlinks(child)

    for child in context.iterdir():
        reject_symlinks(child)
    for name in entries:
        if name not in {"manifest.json", "models"}:
            path = context / name
            if name in PREPARE_CONTEXT_FILES and not path.is_file():
                raise ValueError("measurement context build input has unexpected shape")
            if name == "services":
                actual = {item.relative_to(context).as_posix() for item in path.rglob("*")
                          if item.is_file() and not _is_python_cache(item) and not any(
                              _is_python_cache(parent) for parent in item.relative_to(context).parents)}
                if actual != PREPARE_HARNESS_FILES:
                    raise ValueError("measurement context harness files are not exact")
            if name == "wheelhouse" and (not path.is_dir() or any(
                    not _is_python_cache(item) and (item.is_symlink() or not item.is_file())
                    for item in path.iterdir())):
                raise ValueError("measurement context wheelhouse is unsafe")
    models_root = context / "models"
    if models_root.is_symlink() or not models_root.is_dir() or {
            item.name for item in models_root.iterdir() if not _is_python_cache(item)} != set(MODELS):
        raise ValueError("measurement context model set is not exact")
    for model in MODELS:
        root = roots[model]
        if (root.is_symlink() or not root.is_dir()
                or {item.name for item in root.iterdir() if not _is_python_cache(item)} != set(SPECS[model])
                or any(not _is_python_cache(item) and (item.is_symlink() or not item.is_file())
                       for item in root.iterdir())):
            raise ValueError(f"measurement context files are not exact for {model}")
    # ``verify_manifest`` is the sole authority for artifact-manifest bytes.
    # In particular, its compact JSON deliberately has no trailing newline.
    verify_manifest(value, roots)
    return value


def _config_document(gpu: str, manifest_hash: str, mount: str, runtimes: dict, adapters: dict) -> dict:
    root = Path(mount)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("container mount must be absolute and non-traversing")
    return {"schema": 1, "gpu_uuid": gpu, "artifact_root": str(root / "artifacts"),
            "manifest_sha256": manifest_hash, "profile_db": str(RUNTIME_PROFILE_DB),
            "ollama_binary": str(CANONICAL_OLLAMA_BINARY),
            "ollama_home": str(RUNTIME_OLLAMA_HOME), "ollama_port": 11434,
            "models": {m: {"runtime_identity": runtimes[m], "adapter_identity": adapters[m]}
                        for m in MODELS}}


def _model_hashes(document: dict) -> dict[str, str]:
    """Return the runtime model-file identities from one verified manifest."""
    return {model: next(item["sha256"] for item in document["models"][model]["files"]
                        if item["path"] == SPECS[model][0])
            for model in MODELS}


def _selected_manifest(artifacts: Path) -> tuple[dict, str]:
    """Read the manifest selected by the volume verifier, not its source input."""
    result = verify_current(artifacts)
    selected = artifacts / "current"
    target = os.readlink(selected)
    document = read_json(artifacts / target / "manifest.json")
    if document.get("manifest_sha256") != result["manifestSha256"]:
        raise ValueError("selected artifact manifest changed during verification")
    return document, result["manifestSha256"]


def _prepare(output: Path, context: Path, gpu: str, ollama: str, mount: str,
              runtimes: dict, adapters: dict, replace: bool) -> str:
    source_manifest = _manifest(context)
    source_model_hashes = _model_hashes(source_manifest)
    output = output.absolute()
    if output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise ValueError("output path contains a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temp:
        stage = Path(temp)
        # Provisioning normalizes source-only roots into the runtime volume's
        # models/<model> roots. Its manifest is therefore the authoritative
        # runtime identity and may intentionally differ from the context one.
        provision({m: context / "models" / m for m in MODELS}, stage / "artifacts")
        manifest, manifest_hash = _selected_manifest(stage / "artifacts")
        model_hashes = _model_hashes(manifest)
        if model_hashes != source_model_hashes:
            raise ValueError("selected artifact model hashes do not match verified context")
        fixtures = canonical_measurement_fixtures()
        configs = [MATRIX[ModelId.SMOLLM], MATRIX[ModelId.COEDIT], MATRIX[ModelId.GECTOR]]
        witnesses = ({"dtype": "q8_0", "generation_parameters": {"num_predict": 64, "temperature": 0}, "native_batch_shape": [1]},
                     {"adapter": adapters["CoEdIT"]}, {"adapter": adapters["GECToR"]})
        requests = {}
        fingerprints = {}
        for model, cfg, selector, witness in zip(MODELS, configs, MATRIX_IDS, witnesses):
            item = prepare_benchmark_request(cfg, request_bucket=selector,
                identity_witnesses={**witness, "manifest": manifest_hash, "runtime": runtimes[model]},
                configured_request=fixtures[model], validate_request=lambda *_: True)
            requests[model] = fixtures[model]
            fingerprints[model] = item.fingerprint
        config = _config_document(gpu, manifest_hash, mount, runtimes, adapters)
        config_bytes = canonical(config)
        requests_bytes = canonical(requests)
        provenance = {"schema": 2, "tool": "prepare_measurement.py", "matrix": list(MATRIX_IDS),
                       "gpu_uuid": gpu, "ollama_version": ollama,
                       "container_mount": mount,
                       "manifest_sha256": manifest_hash, "model_sha256": model_hashes,
                      "source_context_manifest_sha256": source_manifest["manifest_sha256"],
                      "source_model_sha256": source_model_hashes,
                      "runtime_identities": runtimes, "adapter_identities": adapters,
                      "request_fingerprints": fingerprints}
        identity_body = {"config_sha256": digest(config_bytes), "requests_sha256": digest(requests_bytes),
                         "manifest_sha256": manifest_hash,
                         "source_context_manifest_sha256": source_manifest["manifest_sha256"],
                         "source_model_sha256": source_model_hashes, "matrix": list(MATRIX_IDS),
                         "request_fingerprints": fingerprints}
        provenance["bundle_identity"] = digest(canonical(identity_body))
        provenance_bytes = canonical(provenance)
        files = {"config.json": config_bytes, "requests.json": requests_bytes,
                 "provenance.json": provenance_bytes,
                 "provenance.sha256": (digest(provenance_bytes) + "\n").encode("ascii")}
        if output.exists() and not replace:
        # Idempotence must not turn a tampered bundle into a successful no-op.
        # Verify the complete existing bundle before accepting its identity.
            if (output / "provenance.sha256").is_file() and (output / "provenance.sha256").read_bytes() == files["provenance.sha256"]:
                if verify(output) != digest(provenance_bytes):
                    raise ValueError("existing bundle identity mismatch")
                return digest(provenance_bytes)
            raise ValueError("output exists; use --replace for a changed bundle")
        for name, data in files.items():
            (stage / name).write_bytes(data)
        # Verify all generated documents and the selected volume before the
        # stage can be atomically made visible.
        if verify(stage) != digest(provenance_bytes):
            raise ValueError("staged bundle identity mismatch")
        if output.exists():
            if not output.is_dir() or output.is_symlink():
                raise ValueError("cannot safely replace output")
            # Never remove the last known-good bundle before the new staged
            # bundle has become selected.  All names are siblings, therefore
            # these are same-filesystem renames.
            backup = output.parent / f".{output.name}.backup-{next(tempfile._get_candidate_names())}"
            os.replace(output, backup)
            try:
                os.replace(stage, output)
            except Exception:
                try:
                    os.replace(backup, output)
                except Exception as rollback_error:
                    raise ValueError("bundle replacement and rollback failed") from rollback_error
                raise
            shutil.rmtree(backup)
        else:
            os.replace(stage, output)
    return digest(provenance_bytes)


def verify(output: Path, *, require_runtime_mount: bool = False) -> str:
    """Verify immutable bundle contents, optionally enforcing its runtime mount.

    Host-side preparation deliberately verifies an arbitrary output directory.
    The full measurement runtime additionally requires the bundle to be mounted
    at the location bound into its provenance and configuration.
    """
    output = output.absolute()
    if output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise ValueError("measurement bundle path contains a symlink")
    expected = {"config.json", "requests.json", "provenance.json", "provenance.sha256", "artifacts"}
    if not output.is_dir() or {p.name for p in output.iterdir()} != expected or any(
            p.is_symlink() for p in output.iterdir()):
        raise ValueError("measurement bundle has unexpected entries")
    config = read_json(output / "config.json"); load_config(output / "config.json")
    if (config["profile_db"] != str(RUNTIME_PROFILE_DB) or
            config["ollama_binary"] != str(CANONICAL_OLLAMA_BINARY) or
            config["ollama_home"] != str(RUNTIME_OLLAMA_HOME)):
        raise ValueError("bundle runtime mount contract mismatch")
    requests = read_json(output / "requests.json")
    provenance = read_json(output / "provenance.json")
    if canonical(config) != (output / "config.json").read_bytes() or canonical(requests) != (output / "requests.json").read_bytes() or canonical(provenance) != (output / "provenance.json").read_bytes():
        raise ValueError("bundle JSON is not canonical")
    pd = digest((output / "provenance.json").read_bytes())
    if (output / "provenance.sha256").read_text() != pd + "\n": raise ValueError("provenance digest mismatch")
    required_provenance = {"schema", "tool", "matrix", "gpu_uuid", "ollama_version",
                             "container_mount", "manifest_sha256", "model_sha256", "runtime_identities",
                            "source_context_manifest_sha256", "source_model_sha256",
                            "adapter_identities", "request_fingerprints", "bundle_identity"}
    if (set(provenance) != required_provenance or provenance["schema"] != 2
            or provenance["tool"] != "prepare_measurement.py"
            or not isinstance(provenance["gpu_uuid"], str)
            or not isinstance(provenance["ollama_version"], str)
            or not isinstance(provenance["container_mount"], str)
            or not isinstance(provenance["manifest_sha256"], str)
            or not isinstance(provenance["source_context_manifest_sha256"], str)
            or not isinstance(provenance["bundle_identity"], str)
            or not isinstance(provenance["matrix"], list)
            or not isinstance(provenance["request_fingerprints"], dict)
            or set(provenance["request_fingerprints"]) != set(MODELS)
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in (provenance["manifest_sha256"],
                                 provenance["source_context_manifest_sha256"],
                                 *provenance["request_fingerprints"].values()))):
        raise ValueError("provenance schema mismatch")
    container_mount = Path(provenance["container_mount"])
    if (not container_mount.is_absolute() or ".." in container_mount.parts
            or Path(config["artifact_root"]) != container_mount / "artifacts"):
        raise ValueError("bundle runtime mount contract mismatch")
    if tuple(provenance["matrix"]) != MATRIX_IDS or set(requests) != set(MODELS) or requests != canonical_measurement_fixtures():
        raise ValueError("matrix or requests mismatch")
    if provenance["gpu_uuid"] != config["gpu_uuid"] or provenance["manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError("configuration provenance mismatch")
    if (not isinstance(provenance["model_sha256"], dict) or not isinstance(provenance["source_model_sha256"], dict)
            or set(provenance["model_sha256"]) != set(MODELS)
            or set(provenance["source_model_sha256"]) != set(MODELS)
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for hashes in (provenance["model_sha256"], provenance["source_model_sha256"])
                   for value in hashes.values())):
        raise ValueError("model provenance schema mismatch")
    if provenance["source_model_sha256"] != provenance["model_sha256"]:
        raise ValueError("source/runtime model provenance mismatch")
    if provenance["runtime_identities"] != {m: config["models"][m]["runtime_identity"] for m in MODELS} or provenance["adapter_identities"] != {m: config["models"][m]["adapter_identity"] for m in MODELS}:
        raise ValueError("runtime identity mismatch")
    configs = [MATRIX[ModelId.SMOLLM], MATRIX[ModelId.COEDIT], MATRIX[ModelId.GECTOR]]
    for model, cfg, selector in zip(MODELS, configs, MATRIX_IDS):
        witness = {"dtype": "q8_0", "generation_parameters": {"num_predict": 64, "temperature": 0}, "native_batch_shape": [1]} if model == "SmolLM" else {"adapter": provenance["adapter_identities"][model]}
        checked = prepare_benchmark_request(cfg, request_bucket=selector,
            identity_witnesses={**witness, "manifest": provenance["manifest_sha256"], "runtime": provenance["runtime_identities"][model]},
            configured_request=requests[model], validate_request=lambda *_: True)
        if checked.fingerprint != provenance["request_fingerprints"][model]:
            raise ValueError("request fingerprint mismatch")
    identity_body = {"config_sha256": digest((output / "config.json").read_bytes()),
                      "requests_sha256": digest((output / "requests.json").read_bytes()),
                      "manifest_sha256": provenance["manifest_sha256"], "matrix": list(MATRIX_IDS),
                      "source_context_manifest_sha256": provenance["source_context_manifest_sha256"],
                      "source_model_sha256": provenance["source_model_sha256"],
                      "request_fingerprints": provenance["request_fingerprints"]}
    if provenance["bundle_identity"] != digest(canonical(identity_body)):
        raise ValueError("bundle identity mismatch")
    artifact_result = verify_current(output / "artifacts")
    if artifact_result["manifestSha256"] != provenance["manifest_sha256"]:
        raise ValueError("artifact manifest provenance mismatch")
    selected_manifest = read_json(output / "artifacts" / "current" / "manifest.json")
    expected_hashes = {
        model: next(item["sha256"] for item in selected_manifest["models"][model]["files"]
                    if item["path"] == SPECS[model][0])
        for model in MODELS
    }
    if provenance["model_sha256"] != expected_hashes:
        raise ValueError("model hash provenance mismatch")
    if require_runtime_mount and output.resolve() != container_mount.resolve():
        raise ValueError("bundle is not mounted at its recorded container mount")
    return pd


def verify_bundle_inputs(bundle: Path, *, require_runtime_mount: bool = False) -> tuple[Path, Path, Path, str]:
    """Verify a complete bundle and return its bound input paths and digest.

    This deliberately performs no database access: a bundle is immutable input,
    while the profile database is a fresh caller-owned output.
    """
    bundle = bundle.absolute()
    provenance_digest = verify(bundle, require_runtime_mount=require_runtime_mount)
    return (bundle / "config.json", bundle / "requests.json",
            bundle / "provenance.json", provenance_digest)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "verify")); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--context", type=Path, default=ROOT / ".compatibility/context")
    p.add_argument("--gpu-uuid"); p.add_argument("--ollama-version", default="0.11.6")
    p.add_argument("--container-mount", default="/opt/measurement"); p.add_argument("--replace", action="store_true")
    for model in MODELS:
        p.add_argument(f"--{model.lower()}-runtime"); p.add_argument(f"--{model.lower()}-adapter")
    try:
        a = p.parse_args()
        if a.mode == "verify": print(f"provenance={verify(a.output)}"); return 0
        if not a.gpu_uuid or not a.gpu_uuid.startswith("GPU-"): p.error("--gpu-uuid is required")
        runtimes = {m: getattr(a, f"{m.lower()}_runtime") or ("ollama:" + a.ollama_version if m == "SmolLM" else "runtime:" + m) for m in MODELS}
        adapters = {m: getattr(a, f"{m.lower()}_adapter") or ("adapter:" + m) for m in MODELS}
        print(f"provenance={_prepare(a.output, a.context, a.gpu_uuid, a.ollama_version, a.container_mount, runtimes, adapters, a.replace)}")
        return 0
    except Exception as exc:
        message = str(exc).replace("\n", " ")[:240]
        print(f"prepare_measurement: {message or 'invalid input'}", file=sys.stderr)
        return 2


if __name__ == "__main__": raise SystemExit(main())

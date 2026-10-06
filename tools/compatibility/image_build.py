"""Fail-closed, offline compatibility image build and verification workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import uuid
import zipfile
from email.parser import BytesParser
from typing import Any, Callable

SCHEMA = "compatibility-image-identity/v1"
ROLE = "adapter"
ENTRYPOINT = ["/opt/venv/bin/python"]
MAX_OUTPUT = 8192
INSPECT_TIMEOUT = 30
BUILD_TIMEOUT = 1800
RUN_TIMEOUT = 30
LABELS = {"schema": "org.llm-provider.compatibility.schema", "role": "org.llm-provider.compatibility.role", "identity": "org.llm-provider.compatibility.identity", "base": "org.llm-provider.compatibility.base", "source": "org.llm-provider.compatibility.source-sha256", "inputs": "org.llm-provider.compatibility.inputs-sha256"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _bounded(value: Any) -> str:
    text = value.decode(errors="replace") if isinstance(value, bytes) else str(value or "")
    return text if len(text) <= MAX_OUTPUT else text[:MAX_OUTPUT] + "…"


def _run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run a child while retaining bounded diagnostics without blocking its pipes."""
    check = kwargs.pop("check", True)
    timeout = kwargs.pop("timeout", None)
    kwargs.pop("text", None)
    kwargs.pop("capture_output", None)
    if kwargs:
        raise TypeError(f"unsupported command options: {', '.join(sorted(kwargs))}")
    captured = [bytearray(), bytearray()]

    def drain(stream: Any, index: int) -> None:
        try:
            while chunk := stream.read(4096):
                remaining = MAX_OUTPUT - len(captured[index])
                if remaining > 0:
                    captured[index].extend(chunk[:remaining])
        except (OSError, ValueError):
            # The parent closes the descriptors after a bounded join.  A child
            # retaining an inherited descriptor must not strand this workflow.
            pass

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    assert process.stdout is not None and process.stderr is not None
    drains = [threading.Thread(target=drain, args=(stream, index))
              for index, stream in enumerate((process.stdout, process.stderr))]
    for thread in drains:
        thread.start()
    timed_out = None
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        timed_out = exc
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    finally:
        for thread in drains:
            thread.join(timeout=.2)
        for stream in (process.stdout, process.stderr):
            stream.close()
        for thread in drains:
            thread.join(timeout=.2)
    if timed_out is not None:
        raise RuntimeError(
            f"command timed out: {' '.join(command[:4])}; "
            f"stdout: {_bounded(bytes(captured[0]))}; stderr: {_bounded(bytes(captured[1]))}"
        ) from timed_out
    stdout, stderr = _bounded(bytes(captured[0])), _bounded(bytes(captured[1]))
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command, output=stdout, stderr=stderr)
    return result


def inspect_image(ref: str, runner: Callable[..., subprocess.CompletedProcess[str]] | None = None) -> dict[str, Any]:
    runner = _run if runner is None else runner
    result = runner(["docker", "image", "inspect", ref, "--format", "{{json .}}"], timeout=INSPECT_TIMEOUT)
    try:
        data = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"docker inspect returned malformed JSON for {ref}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("Id"), str) or not data["Id"]:
        raise ValueError(f"docker inspect did not resolve immutable image ID for {ref}")
    return data


def _wheel_identity(name: str) -> tuple[str, str] | None:
    """Return PEP 427 distribution/version, accepting its optional build tag."""
    if not name.endswith(".whl"):
        return None
    parts = name[:-4].split("-")
    if len(parts) not in (5, 6) or any(not part for part in parts):
        return None
    distribution, version = parts[0], parts[1]
    if len(parts) == 6 and not re.fullmatch(r"\d[0-9A-Za-z_]*", parts[2]):
        return None
    return distribution.lower().replace("_", "-"), version


def _lock_inputs(context: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    lock = context / ".compatibility/adapter-deps/requirements.lock"
    wheelhouse = context / ".compatibility/adapter-deps/wheelhouse"
    root = context.resolve()
    def under_root(path: Path, kind: str) -> None:
        if path.is_symlink() or not path.exists():
            raise ValueError(f"adapter {kind} must be a non-symlink under root")
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"adapter {kind} resolves outside root") from exc
    under_root(lock, "requirements.lock")
    under_root(wheelhouse, "wheelhouse")
    if not lock.is_file() or not wheelhouse.is_dir():
        raise ValueError("adapter lock and wheelhouse are required under .compatibility/adapter-deps")
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in lock.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^ ]+) --hash=sha256:([0-9a-f]{64})", line)
        if not match:
            raise ValueError(f"malformed or unhashed adapter lock entry: {line}")
        name, version, digest = match.groups()
        key = name.lower().replace("_", "-")
        if (key, version) in seen:
            raise ValueError(f"duplicate adapter lock record: {key}=={version}")
        seen.add((key, version))
        records.append({"name": key, "version": version, "sha256": digest})
    if not records:
        raise ValueError("adapter lock is empty")
    entries = sorted(wheelhouse.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("wheelhouse must contain only regular wheels")
    files = entries
    if any(p.suffix != ".whl" for p in files):
        raise ValueError("wheelhouse contains a non-wheel file")
    by_hash: dict[str, Path] = {}
    for path in files:
        identity = _wheel_identity(path.name)
        if identity is None:
            raise ValueError(f"malformed wheel filename: {path.name}")
        try:
            with zipfile.ZipFile(path) as archive:
                metadata = [name for name in archive.namelist() if re.fullmatch(r"[^/]+\.dist-info/METADATA", name)]
                if len(metadata) != 1:
                    raise ValueError("wheel has no unique METADATA")
                fields = BytesParser().parsebytes(archive.read(metadata[0]))
                wheel_name, wheel_version = fields.get("Name"), fields.get("Version")
        except (OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
            raise ValueError(f"malformed wheel archive: {path.name}") from exc
        if not isinstance(wheel_name, str) or not isinstance(wheel_version, str) or (
            wheel_name.lower().replace("_", "-"), wheel_version
        ) != identity:
            raise ValueError(f"wheel METADATA does not match filename: {path.name}")
        digest = sha256(path)
        if digest in by_hash:
            raise ValueError("duplicate wheel content")
        by_hash[digest] = path
    lock_hashes = {item["sha256"] for item in records}
    if set(by_hash) != lock_hashes:
        raise ValueError("wheelhouse and lock are not an exact match")
    for item in records:
        wheel = by_hash[item["sha256"]]
        identity = _wheel_identity(wheel.name)
        if identity and identity != (item["name"], item["version"]):
            raise ValueError(f"wheel filename does not match lock: {wheel.name}")
    records.sort(key=lambda x: (x["name"], x["version"], x["sha256"]))
    return records, {item["name"]: item["version"] for item in records}


def _base_contract(base_context: Path | None) -> dict[str, Any]:
    if base_context is None:
        return {}
    provenance, manifest = base_context / "provenance.json", base_context / "manifest.json"
    if not provenance.is_file() or not manifest.is_file():
        raise ValueError("base context requires provenance.json and manifest.json")
    return {"provenance_sha256": sha256(provenance), "manifest_sha256": sha256(manifest)}


def _ignored(relative: str, rules: list[tuple[bool, str]]) -> bool:
    path = PurePosixPath(relative)
    ignored = False
    for negate, pattern in rules:
        pattern = pattern.lstrip("/")
        directory = pattern.endswith("/")
        pattern = pattern.rstrip("/")
        matched = path.match(pattern) or PurePosixPath(relative).match(pattern.lstrip("**/"))
        if directory:
            matched = matched or relative.startswith(pattern + "/")
        if matched:
            ignored = not negate
    return ignored


def _ignore_rules(root: Path) -> list[tuple[bool, str]]:
    ignore = root / "tools/compatibility/Dockerfile.adapter.dockerignore"
    if ignore.is_symlink():
        raise ValueError("adapter dockerignore is a symlink")
    rules = []
    for raw in ignore.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            rules.append((line.startswith("!"), line[1:] if line.startswith("!") else line))
    return rules


def _copy_sources(root: Path) -> set[str]:
    dockerfile = root / "tools/compatibility/Dockerfile.adapter"
    if dockerfile.is_symlink() or not dockerfile.is_file():
        raise ValueError("adapter Dockerfile is missing")
    try:
        dockerfile.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("adapter Dockerfile resolves outside root") from exc
    rules = _ignore_rules(root)
    result = {"tools/compatibility/Dockerfile.adapter", "tools/compatibility/Dockerfile.adapter.dockerignore"}
    lines = dockerfile.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        instruction = stripped.split(None, 1)[0].upper()
        if instruction != "COPY":
            # A lower-case COPY is valid Docker syntax, but any other
            # instruction is not relevant to the source inventory.
            if stripped.split(None, 1)[0].lower() == "copy":
                stripped = "COPY" + stripped.split(None, 1)[1]
            continue
        if stripped.endswith("\\"):
            raise ValueError("unsupported continued Dockerfile COPY syntax")
        try:
            fields = shlex.split(stripped)
        except ValueError as exc:
            raise ValueError("unsupported Dockerfile COPY syntax") from exc
        if fields[0] != "COPY" or len(fields) != 3 or fields[1].startswith("--") or fields[1].startswith("[") or any(x.startswith("[") for x in fields):
            raise ValueError("unsupported Dockerfile COPY options or multiple sources")
        source = fields[1].rstrip("/")
        if not source or source.startswith("/") or ".." in PurePosixPath(source).parts or any(c in source for c in "*?["):
            raise ValueError(f"unsupported Dockerfile COPY source: {source}")
        candidate = root / source
        components = [candidate, *candidate.parents]
        if any(path.is_symlink() for path in components[:components.index(root) + 1] if path != root):
            raise ValueError(f"Dockerfile source is a symlink: {source}")
        if not candidate.exists():
            raise ValueError(f"Dockerfile source is missing: {source}")
        try:
            candidate.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"Dockerfile source resolves outside root: {source}") from exc
        if candidate.is_dir():
            descendants = list(candidate.rglob("*"))
            for path in descendants:
                relative = str(path.relative_to(root)).replace(os.sep, "/")
                if path.is_symlink():
                    raise ValueError(f"Dockerfile source contains a symlink: {relative}")
                try:
                    path.resolve().relative_to(root.resolve())
                except ValueError as exc:
                    raise ValueError(f"Dockerfile source resolves outside root: {relative}") from exc
            copied = [p for p in descendants if p.is_file()]
        else:
            copied = [candidate]
        admitted = 0
        for path in copied:
            relative = str(path.relative_to(root)).replace(os.sep, "/")
            if _ignored(relative, rules):
                continue
            result.add(relative)
            admitted += 1
        if not copied:
            raise ValueError(f"Dockerfile source is empty: {source}")
        if not admitted:
            raise ValueError(f"Dockerfile source is excluded by dockerignore: {source}")
    return result


def _source_files(root: Path) -> list[dict[str, str]]:
    result = []
    for relative in sorted(_copy_sources(root)):
        path = root / relative
        if not path.is_file() or (_ignored(relative, _ignore_rules(root)) and not relative.startswith("tools/compatibility/Dockerfile")):
            raise ValueError(f"Dockerfile source is missing or ignored: {relative}")
        result.append({"path": relative, "sha256": sha256(path)})
    return result


def calculate_identity(root: Path, base: dict[str, Any], base_context: Path | None = None) -> dict[str, Any]:
    sources, wheels = _source_files(root), _lock_inputs(root)[0]
    repo_digests = base.get("RepoDigests") or []
    if not isinstance(repo_digests, list) or any(not isinstance(x, str) for x in repo_digests):
        raise ValueError("base image has malformed RepoDigests")
    if base.get("Os") != "linux" or base.get("Architecture") != "amd64":
        raise ValueError("base image platform must be exactly linux/amd64")
    base_identity = {"id": base["Id"], "platform": "linux/amd64", "repo_digest": sorted(repo_digests), **_base_contract(base_context)}
    source_hash = hashlib.sha256(_canonical(sources)).hexdigest()
    input_hash = hashlib.sha256(_canonical(wheels)).hexdigest()
    payload = {"schema": SCHEMA, "role": ROLE, "platform": "linux/amd64", "base": base_identity, "sources": sources, "adapter_inputs": wheels}
    return {**payload, "identity": hashlib.sha256(_canonical(payload)).hexdigest(), "source_sha256": source_hash, "inputs_sha256": input_hash}


def _validate_identity(document: dict[str, Any]) -> None:
    required = {"schema", "role", "platform", "base", "sources", "adapter_inputs", "identity", "source_sha256", "inputs_sha256"}
    if not isinstance(document, dict) or set(document) != required or document["schema"] != SCHEMA or document["role"] != ROLE or document["platform"] != "linux/amd64":
        raise ValueError("malformed identity file")
    if not all(isinstance(document[k], str) and HEX64.fullmatch(document[k]) for k in ("identity", "source_sha256", "inputs_sha256")):
        raise ValueError("malformed identity digest")
    base = document["base"]
    if not isinstance(base, dict) or set(base) - {"id", "platform", "repo_digest", "provenance_sha256", "manifest_sha256"} or base.get("platform") != "linux/amd64" or not isinstance(base.get("id"), str) or not isinstance(base.get("repo_digest"), list) or base["repo_digest"] != sorted(base["repo_digest"]):
        raise ValueError("malformed base identity")
    if any(not isinstance(x, str) for x in base["repo_digest"]):
        raise ValueError("malformed base repository digest")
    for key in ("provenance_sha256", "manifest_sha256"):
        if key in base and (not isinstance(base[key], str) or not HEX64.fullmatch(base[key])):
            raise ValueError("malformed base contract digest")
    sources, inputs = document["sources"], document["adapter_inputs"]
    if not isinstance(sources, list) or sources != sorted(sources, key=lambda x: x.get("path", "") if isinstance(x, dict) else "") or any(not isinstance(x, dict) or set(x) != {"path", "sha256"} or not isinstance(x["path"], str) or not x["path"] or x["path"].startswith("/") or ".." in PurePosixPath(x["path"]).parts or not HEX64.fullmatch(x["sha256"]) for x in sources):
        raise ValueError("malformed source inventory")
    if not isinstance(inputs, list) or inputs != sorted(inputs, key=lambda x: (x.get("name", ""), x.get("version", ""), x.get("sha256", "")) if isinstance(x, dict) else ("", "", "")) or any(not isinstance(x, dict) or set(x) != {"name", "version", "sha256"} or not all(isinstance(x[k], str) and x[k] for k in x) or not HEX64.fullmatch(x["sha256"]) for x in inputs):
        raise ValueError("malformed adapter inventory")
    if len({item["path"] for item in sources}) != len(sources) or len({(item["name"], item["version"]) for item in inputs}) != len(inputs):
        raise ValueError("duplicate identity inventory entry")
    if hashlib.sha256(_canonical(sources)).hexdigest() != document["source_sha256"] or hashlib.sha256(_canonical(inputs)).hexdigest() != document["inputs_sha256"]:
        raise ValueError("identity inventory digest mismatch")
    payload = {k: document[k] for k in ("schema", "role", "platform", "base", "sources", "adapter_inputs")}
    if hashlib.sha256(_canonical(payload)).hexdigest() != document["identity"]:
        raise ValueError("identity file digest mismatch")


def write_identity(path: Path, document: dict[str, Any]) -> None:
    _validate_identity(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == encoded:
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _stage_context(root: Path, identity: dict[str, Any]) -> tempfile.TemporaryDirectory[str]:
    """Make BuildKit consume immutable regular files, never the live checkout."""
    stage = tempfile.TemporaryDirectory(prefix="compatibility-adapter-stage-")
    destination = Path(stage.name)
    try:
        # Recompute immediately before copying: the stage identity is the
        # admitted source byte inventory and can never include unrelated assets.
        inputs, _ = _lock_inputs(root)
        if _source_files(root) != identity["sources"] or inputs != identity["adapter_inputs"]:
            raise ValueError("identity does not match current source or dependency inputs before staging")
        for item in identity["sources"]:
            source = root / item["path"]
            target = destination / item["path"]
            if source.is_symlink() or not source.is_file() or sha256(source) != item["sha256"]:
                raise ValueError(f"staged source changed or is unsafe: {item['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            os.chmod(target, 0o644)
            os.utime(target, (0, 0))
        # Dependency inputs are identity-listed separately from Dockerfile
        # sources, but Docker COPY needs them in their canonical context paths.
        # Rebuild that small tree solely from the validated lock inventory so a
        # live checkout cannot contribute an unlisted wheel to the build.
        source_deps = root / ".compatibility/adapter-deps"
        staged_deps = destination / ".compatibility/adapter-deps"
        source_lock = source_deps / "requirements.lock"
        staged_lock = staged_deps / "requirements.lock"
        staged_lock.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_lock, staged_lock)
        os.chmod(staged_lock, 0o644)
        os.utime(staged_lock, (0, 0))
        source_wheelhouse = source_deps / "wheelhouse"
        staged_wheelhouse = staged_deps / "wheelhouse"
        staged_wheelhouse.mkdir(exist_ok=True)
        accepted = {item["sha256"] for item in inputs}
        for wheel in sorted(source_wheelhouse.iterdir()):
            if wheel.is_file() and not wheel.is_symlink() and sha256(wheel) in accepted:
                target = staged_wheelhouse / wheel.name
                shutil.copyfile(wheel, target)
                os.chmod(target, 0o644)
                os.utime(target, (0, 0))
        if _source_files(destination) != identity["sources"] or _lock_inputs(destination)[0] != identity["adapter_inputs"]:
            raise ValueError("staged context does not match identity")
        return stage
    except Exception:
        stage.cleanup()
        raise


def build_command(context: Path, tag: str, base_ref: str, identity: dict[str, Any]) -> list[str]:
    labels = {LABELS["schema"]: SCHEMA, LABELS["role"]: ROLE, LABELS["identity"]: identity["identity"], LABELS["base"]: identity["base"]["id"], LABELS["source"]: identity["source_sha256"], LABELS["inputs"]: identity["inputs_sha256"]}
    command = ["docker", "buildx", "build", "--platform=linux/amd64", "--network=none", "--provenance=false", "--sbom=false", "--load", "-f", str(context / "tools/compatibility/Dockerfile.adapter")]
    command += ["--build-arg", f"BASE_IMAGE={base_ref}", "--build-arg", f"COMPATIBILITY_IDENTITY={identity['identity']}", "--build-arg", f"COMPATIBILITY_BASE_ID={identity['base']['id']}", "--build-arg", f"COMPATIBILITY_SOURCE_SHA256={identity['source_sha256']}", "--build-arg", f"COMPATIBILITY_INPUTS_SHA256={identity['inputs_sha256']}"]
    for key, value in labels.items(): command += ["--label", f"{key}={value}"]
    return command + ["-t", tag, str(context)]


def _owned_base_ref(ref: str, inspected: dict[str, Any], runner: Callable[..., subprocess.CompletedProcess[str]]) -> tuple[str, str | None]:
    repo = sorted(inspected.get("RepoDigests") or [])
    if repo:
        immutable = repo[0]
        if inspect_image(immutable, runner).get("Id") != inspected["Id"]:
            raise ValueError("base repository digest changed before build")
        return immutable, None
    owned = f"compatibility-owned-base-{uuid.uuid4().hex}:build"
    runner(["docker", "tag", inspected["Id"], owned], timeout=INSPECT_TIMEOUT)
    try:
        rechecked = inspect_image(owned, runner)
        if rechecked.get("Id") != inspected["Id"]:
            raise ValueError("owned immutable base tag changed before build")
    except Exception:
        runner(["docker", "image", "rm", owned], timeout=INSPECT_TIMEOUT, check=False)
        raise
    return owned, owned


def _verify_current(root: Path, base_image: str, base_context: Path | None, identity: dict[str, Any], runner: Callable[..., subprocess.CompletedProcess[str]]) -> None:
    current = calculate_identity(root.resolve(), inspect_image(base_image, runner), base_context)
    if current != identity:
        raise ValueError("identity does not match current source, lock, or base inputs")


def verify(image: str, identity_path: Path, root: Path, base_image: str, base_context: Path | None, runner: Callable[..., subprocess.CompletedProcess[str]] | None = None) -> None:
    runner = _run if runner is None else runner
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    _validate_identity(identity)
    _verify_current(root, base_image, base_context, identity, runner)
    inspected = inspect_image(image, runner)
    labels = inspected.get("Config", {}).get("Labels") or {}
    expected = {LABELS["schema"]: SCHEMA, LABELS["role"]: ROLE, LABELS["identity"]: identity["identity"], LABELS["base"]: identity["base"]["id"], LABELS["source"]: identity["source_sha256"], LABELS["inputs"]: identity["inputs_sha256"]}
    for key, value in expected.items():
        if labels.get(key) != value: raise ValueError(f"image label mismatch: {key}")
    if inspected.get("Config", {}).get("Entrypoint") != ENTRYPOINT: raise ValueError("image entrypoint mismatch")
    names = [item["name"] for item in identity["adapter_inputs"]]
    script = "import importlib.metadata as m; " + "; ".join(f"print(m.version({name!r}))" for name in names)
    result = runner(["docker", "run", "--rm", "--network=none", "--entrypoint", "/opt/venv/bin/python", image, "-c", script], timeout=RUN_TIMEOUT)
    lines = result.stdout.splitlines()
    expected_versions = [item["version"] for item in identity["adapter_inputs"]]
    if lines != expected_versions or result.stderr.strip():
        raise ValueError("runtime package identity probe returned unexpected output")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    identity_parser = sub.add_parser("identity"); identity_parser.add_argument("--root", type=Path, default=Path(".")); identity_parser.add_argument("--base-image", required=True); identity_parser.add_argument("--base-context", type=Path); identity_parser.add_argument("--output", type=Path, required=True)
    build = sub.add_parser("build-adapter"); build.add_argument("--root", type=Path, default=Path(".")); build.add_argument("--base-image", required=True); build.add_argument("--tag", required=True); build.add_argument("--identity", type=Path, required=True); build.add_argument("--base-context", type=Path)
    check = sub.add_parser("verify"); check.add_argument("--root", type=Path, default=Path(".")); check.add_argument("--base-image", required=True); check.add_argument("--base-context", type=Path); check.add_argument("--image", required=True); check.add_argument("--identity", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        if args.command == "verify": verify(args.image, args.identity, root, args.base_image, args.base_context); print(f"verified equivalent current-source image: {args.image}")
        else:
            base = inspect_image(args.base_image); document = calculate_identity(root, base, args.base_context)
            if args.command == "identity": write_identity(args.output, document); print(document["identity"])
            else:
                supplied = json.loads(args.identity.read_text(encoding="utf-8")); _validate_identity(supplied)
                if supplied != document: raise ValueError("identity file does not match current source, lock, or base inputs; recalculate it")
                # Resolve the mutable user reference again after identity
                # inspection.  A tag retarget between these operations must
                # never be silently turned into a build from a different base.
                latest_base = inspect_image(args.base_image)
                if latest_base.get("Id") != base.get("Id") or latest_base.get("Os") != base.get("Os") or latest_base.get("Architecture") != base.get("Architecture") or sorted(latest_base.get("RepoDigests") or []) != sorted(base.get("RepoDigests") or []):
                    raise ValueError("base image changed between inspection and build")
                base_ref, owned = _owned_base_ref(args.base_image, base, _run)
                try:
                    _run(["docker", "buildx", "version"], timeout=INSPECT_TIMEOUT)
                    stage = _stage_context(root, document)
                    try:
                        _run(build_command(Path(stage.name), args.tag, base_ref, document), timeout=BUILD_TIMEOUT)
                    finally:
                        stage.cleanup()
                finally:
                    if owned: _run(["docker", "image", "rm", owned], timeout=INSPECT_TIMEOUT, check=False)
                write_identity(args.identity, document); print(f"built equivalent current-source image: {args.tag}")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"compatibility image workflow refused: {_bounded(exc)}", file=__import__("sys").stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())

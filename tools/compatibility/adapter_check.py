"""Offline, fail-closed GPU check for the real SmolLM provider.

This is deliberately a harness, rather than another provider implementation.  The
Ollama daemon is started by this process and is the only supervisor whose GPU
ownership is accepted by :class:`LinuxGPUProof`.
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import sys
import tempfile
import time
import re

import aiohttp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.llm.provisioning.volume import provision
from services.llm.providers.config import GPUProof, SmolLMProviderConfig
from services.llm.providers.gpu import LinuxGPUProof
from services.llm.providers.smollm import SmolLMProvider
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager, ResourceManagerError
from services.llm.resource_manager.protocol import EventKind

GGUF = "SmolLM2-1.7B-Instruct-Q8_0.gguf"
OLLAMA_VERSION = "0.11.6"
LOG_LIMIT = 64 * 1024
_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")
PROC_ROOT = Path("/proc")


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _stat_identity(raw: bytes) -> tuple[int, int]:
    opening, closing = raw.find(b"("), raw.rfind(b")")
    if opening <= 0 or closing <= opening:
        raise RuntimeError("launcher returned malformed host proc stat")
    pid = int(raw[:opening].strip())
    if pid <= 0:
        raise RuntimeError("launcher returned invalid host PID")
    fields = raw[closing + 2:].split()
    try:
        start_time = int(fields[19])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("launcher returned malformed host proc stat") from exc
    if start_time < 0:
        raise RuntimeError("launcher returned invalid host process start time")
    return pid, start_time


def _validate_port(port: int) -> None:
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be in range 1..65535")


def _runtime_home(model_store: Path) -> Path:
    """Return the private HOME shared by the daemon and its import CLI."""
    return model_store.resolve() / ".ollama-home"


def _enable_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")


LAUNCHER = rf'''import os, sys
def fail(code):
    sys.stdout.write("HANDSHAKE_ERROR " + code + "\n")
    sys.stdout.flush()
    raise SystemExit(1)
try:
    raw = open("{PROC_ROOT}/self/stat", "rb").read(4096)
except FileNotFoundError: fail("procfs_missing")
except PermissionError: fail("procfs_permission")
except OSError: fail("procfs_unavailable")
opening, closing = raw.find(b"("), raw.rfind(b")")
if opening <= 0 or closing <= opening: fail("procfs_malformed")
try:
    pid = int(raw[:opening].strip())
    fields = raw[closing + 2:].split()
    start = int(fields[19])
except (ValueError, IndexError): fail("procfs_malformed")
if pid <= 0 or start < 0: fail("procfs_malformed")
sys.stdout.write("HANDSHAKE " + str(pid) + " " + str(start) + "\n")
sys.stdout.flush()
try:
    os.execve("/usr/bin/ollama", ["/usr/bin/ollama", "serve"], os.environ)
except OSError: fail("ollama_exec_failed")
'''


async def _drain(stream: asyncio.StreamReader, limit: int = LOG_LIMIT) -> bytes:
    kept = bytearray()
    while chunk := await stream.read(8192):
        kept.extend(chunk)
        if len(kept) > limit:
            del kept[:-limit]
    return bytes(kept)


async def _start_server(port: int, model_store: Path) -> tuple[asyncio.subprocess.Process, int, tuple[asyncio.Task[bytes], ...]]:
    _validate_port(port)
    _enable_subreaper()
    model_store = model_store.resolve()
    model_store.mkdir(mode=0o700, parents=True, exist_ok=True)
    runtime_home = _runtime_home(model_store)
    runtime_home.mkdir(mode=0o700, exist_ok=True)
    runtime_home.chmod(0o700)
    env = {key: os.environ[key] for key in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
           "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "TMPDIR", "TEMP", "TMP")
           if key in os.environ}
    env.update({"PATH": "/usr/bin:/bin", "OLLAMA_HOST": f"127.0.0.1:{port}",
            "OLLAMA_MODELS": str(model_store), "HOME": str(runtime_home), "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "1"})
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", LAUNCHER, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True)
    assert proc.stdout is not None and proc.stderr is not None
    stderr_drain = asyncio.create_task(_drain(proc.stderr, 2048))
    drains: tuple[asyncio.Task[bytes], ...] = (stderr_drain,)
    try:
        line = await asyncio.wait_for(proc.stdout.readline(), 10)
        if len(line) > 256:
            raise RuntimeError("Ollama launcher handshake exceeded bound")
        fields = line.split()
        if len(fields) == 2 and fields[0] == b"HANDSHAKE_ERROR" and re.fullmatch(rb"[a-z_]+", fields[1]):
            raise RuntimeError("Ollama launcher " + fields[1].decode("ascii", "replace"))
        if len(fields) != 3 or fields[0] != b"HANDSHAKE":
            raise RuntimeError("Ollama launcher handshake missing")
        host_pid = int(fields[1])
        host_start = int(fields[2])
        if host_pid <= 0 or host_start < 0: raise ValueError
        # The attested host PID must name the running launcher in authoritative
        # procfs.  This rejects malformed namespace handshakes and PID reuse.
        stat_pid, stat_start = _stat_identity((PROC_ROOT / str(host_pid) / "stat").read_bytes())
        if (stat_pid, stat_start) != (host_pid, host_start):
            raise RuntimeError("host PID handshake does not match procfs")
        # The direct child must still be the launched process in this namespace;
        # this is a liveness/reuse check, not an attempted host PID translation.
        local_pid, local_start = _stat_identity((PROC_ROOT / str(proc.pid) / "stat").read_bytes())
        if local_pid != proc.pid or local_start != host_start:
            raise RuntimeError("launcher PID does not match local procfs")
        # Save the leader's identity and session fence before handing it to
        # cleanup.  --pid=host makes an unchecked killpg an unacceptable
        # container-wide blast radius.
        proc._adapter_group_fence = (host_pid, host_start, host_pid, host_pid)  # noqa: SLF001
    except BaseException as exc:
        try:
            await _cleanup_group(proc, drains)
        except BaseException as cleanup_exc:
            raise RuntimeError("launcher cleanup failed") from cleanup_exc
        diagnostic = b""
        if stderr_drain.done():
            try: diagnostic = stderr_drain.result()[-2048:]
            except BaseException: pass
        detail = diagnostic.decode("utf-8", "replace").replace("\n", " ")[:2048]
        if isinstance(exc, asyncio.CancelledError):
            raise
        safe_exc = str(exc) if str(exc).startswith("Ollama launcher ") else ""
        suffix = ": " + (safe_exc or detail) if (safe_exc or detail) else ""
        raise RuntimeError("invalid Ollama launcher handshake" + suffix) from exc
    drains = (asyncio.create_task(_drain(proc.stdout)), stderr_drain)
    return proc, host_pid, drains


async def _health(port: int, timeout: float = 60) -> str:
    _validate_port(port)
    deadline = time.monotonic() + timeout
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=5)) as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(f"http://127.0.0.1:{port}/api/version", allow_redirects=False) as response:
                    if response.status == 200 and not response.history:
                        body = await response.content.read(4097)
                        if len(body) > 4096:
                            raise RuntimeError("health response exceeded bound")
                        def pairs(items):
                            result = {}
                            for key, value in items:
                                if key in result: raise ValueError("duplicate JSON key")
                                result[key] = value
                            return result
                        value = json.loads(body, object_pairs_hook=pairs,
                                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))
                        version = value.get("version") if isinstance(value, dict) else None
                        if version != OLLAMA_VERSION:
                            raise RuntimeError("unexpected Ollama version")
                        return version
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
                pass
            await asyncio.sleep(.2)
    raise RuntimeError("Ollama health check timed out")


async def _events(rm, token: str, request_id: str, timeout: float) -> list:
    async def collect():
        found = []
        async for event in rm.watch_progress(token):
            found.append(event)
            if event.request_id == request_id and event.kind in (EventKind.RESPONSE_FINISHED, EventKind.FAILURE):
                return found
        raise RuntimeError("resource manager progress stream ended")
    return await asyncio.wait_for(collect(), timeout)


def _profile(manifest: str, model: str, gpu: str) -> CapacityProfile:
    return CapacityProfile(ModelId.SMOLLM, gpu, manifest, model,
        "candidate-adapter-check", "candidate-smollm-provider",
        "unmeasured-adapter-check", 1, 1, 1, 0,
        (SampleMetadata(1, 0, 0, 0, 0, ()),), context_size=512)


async def _stop_group(proc: asyncio.subprocess.Process, drains: tuple[asyncio.Task[bytes], ...]) -> bool:
    def group_fence() -> tuple[int, int, int, int]:
        try:
            raw = (PROC_ROOT / str(proc.pid) / "stat").read_bytes()
            opening, closing = raw.find(b"("), raw.rfind(b")")
            fields = raw[closing + 2:].split()
            return (int(raw[:opening]), int(fields[19]), int(fields[2]), int(fields[3]))
        except (OSError, ValueError, IndexError) as exc:
            raise RuntimeError("owned Ollama leader identity is unavailable") from exc

    expected = getattr(proc, "_adapter_group_fence", None)  # noqa: SLF001
    if expected is None:
        expected = group_fence()
    expected_pid, expected_start, expected_pgrp, expected_session = expected
    if expected_pid != proc.pid or expected_pgrp != proc.pid or expected_session != proc.pid:
        raise RuntimeError("owned Ollama process group fence is invalid")

    def signal_owned(sig: signal.Signals) -> bool | None:
        try:
            current = group_fence()
        except RuntimeError:
            # The leader may have exited before the child watcher observed it.
            # The saved pgrp is checked below; this is not permission to signal.
            return None
        if current != expected:
            return False
        try:
            os.killpg(expected_pgrp, sig)
        except ProcessLookupError:
            return None
        return True

    for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        signaled = signal_owned(sig)
        if signaled is False:
            # Never trade a failed identity proof for a broader signal. The
            # caller/container boundary must handle any remaining processes.
            if proc.returncode is None:
                raise RuntimeError("owned Ollama process group identity changed")
        try:
            await asyncio.wait_for(proc.wait(), wait)
        except asyncio.TimeoutError:
            continue
        # Do not reap adopted children until asyncio's child watcher has observed
        # its direct child.  Reap only this saved process group, never unrelated
        # children owned by the harness.
        if proc.returncode is not None:
            while True:
                try:
                    pid, _ = os.waitpid(-proc.pid, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
    deadline = time.monotonic() + 5
    gone = False
    while time.monotonic() < deadline:
        try:
            os.killpg(expected_pgrp, 0)
        except ProcessLookupError:
            gone = True
            break
        except PermissionError:
            # Existence could not be disproved.  A missing/reused leader is
            # not evidence that the saved process group is gone.
            break
        await asyncio.sleep(.05)
    # Drains are bounded: an inherited pipe held by a surviving descendant is
    # evidence of a failed group fence, not a reason to wait forever.
    # Continue readers to EOF after the group is dead.  A finite inherited-pipe
    # timeout closes only our asyncio transports and fails closed rather than
    # leaving unraisable subprocess transports behind.
    try:
        await asyncio.wait_for(asyncio.gather(*drains), 5)
    except asyncio.TimeoutError:
        for fd in (1, 2):
            pipe = proc._transport.get_pipe_transport(fd)  # noqa: SLF001
            if pipe is not None: pipe.close()
        await asyncio.gather(*drains, return_exceptions=True)
        raise RuntimeError("Ollama daemon pipes did not close after group termination")
    return gone


async def _cleanup_group(proc, drains) -> bool:
    """Finish owned cleanup even if the caller is repeatedly cancelled."""
    task = asyncio.create_task(_stop_group(proc, drains))
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


def _candidate_manifest(path: Path, source: Path) -> tuple[dict, str]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError("duplicate manifest key")
            result[key] = value
        return result
    document = json.loads(path.read_bytes(), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite manifest value")))
    if not isinstance(document, dict) or not isinstance(document.get("models"), dict):
        raise RuntimeError("candidate manifest is invalid")
    from services.llm.provisioning.artifacts import SPECS, manifest, verify_manifest
    # The retained candidate manifest may contain all three model records. Its
    # aggregate digest is authentic, while the adapter hashes only SmolLM.
    if manifest(document["models"]) != document:
        raise RuntimeError("candidate manifest digest is not canonical")
    if "SmolLM" not in document["models"]:
        raise RuntimeError("candidate manifest has no SmolLM record")
    entry = document["models"]["SmolLM"]
    if [item.get("path") for item in entry.get("files", ()) if isinstance(item, dict)] != list(SPECS["SmolLM"]):
        raise RuntimeError("candidate SmolLM manifest file set is not exact")
    selected_document = manifest({"SmolLM": entry})
    verify_manifest(selected_document, {"SmolLM": source})
    gguf = next(item for item in entry["files"] if item["path"] == GGUF)
    return selected_document, gguf["sha256"]


async def run(args: argparse.Namespace) -> dict:
    _validate_port(args.port)
    if not getattr(args, "host_pid_namespace", False):
        raise ValueError("--host-pid-namespace operator attestation is required")
    if not isinstance(args.target_gpu_uuid, str) or not _UUID.fullmatch(args.target_gpu_uuid):
        raise ValueError("target GPU UUID is invalid")
    source = args.models_root / "SmolLM"
    if not source.is_dir() or not (source / GGUF).is_file():
        raise RuntimeError("selected SmolLM production artifact is missing")
    manifest_task = asyncio.create_task(asyncio.to_thread(_candidate_manifest, args.manifest, source))
    cancelled = False
    while not manifest_task.done():
        try:
            await asyncio.shield(manifest_task)
        except asyncio.CancelledError:
            cancelled = True
    _, model_hash = manifest_task.result()
    if cancelled:
        raise asyncio.CancelledError
    root = Path(tempfile.mkdtemp(prefix="adapter-check-"))
    retain = True
    try:
        volume = root / "artifacts"
        document = await asyncio.to_thread(provision, {"SmolLM": source}, volume)
        daemon_store = root / "ollama-models"
        daemon, host_pid, drains = await _start_server(args.port, daemon_store)
        runtime_home = _runtime_home(daemon_store)
        rm = ResourceManager(cleanup_timeout=60, stop_timeout=60, load_timeout=240)
        provider = None
        session = None
        proof = None
        evidence = None
        normal_cleanup = False
        version = None
        try:
            version = await _health(args.port)
            # Capture only after the daemon is healthy and before provider.load.
            proof = await asyncio.to_thread(LinuxGPUProof.capture, args.target_gpu_uuid,
                host_pid, PROC_ROOT, None, host_pid_namespace=True)
            gpu = GPUProof(proof.identity, proof.cleanup, proof.residency,
                           expected_supervisor=proof.supervisor_identity)
            config = SmolLMProviderConfig(volume, document["manifest_sha256"], model_hash,
                args.target_gpu_uuid, "candidate-adapter-check", "candidate-smollm-provider",
                allowed_context_sizes=(512,),
                 parallelism=1, request_timeout_seconds=60, ollama_port=args.port,
                 gpu_proof=gpu, ollama_binary="/usr/bin/ollama",
                 ollama_home=runtime_home)
            provider = SmolLMProvider(config)
            profile = _profile(document["manifest_sha256"], model_hash, args.target_gpu_uuid)
            session = await rm.start_session("candidate-adapter-check", ModelId.SMOLLM, profile, provider,
                                             idempotency_key="adapter-check-start")
            checks = {}
            for request_id, payload in (("small", b"Say hello."), ("upper", b"A" * 256)):
                submitted = await rm.submit(session.session_token, request_id, "attempt-1", payload,
                    idempotency_key="adapter-check-" + request_id, context_size=512)
                events = await _events(rm, session.session_token, request_id, 60)
                checks[request_id] = submitted.accepted and any(
                    e.kind is EventKind.RESPONSE_FINISHED and bool(e.result) and not e.gpu_timing_complete
                    for e in events)
                if not checks[request_id]:
                    raise RuntimeError("candidate request did not produce a nonempty incomplete-timing result")
            residency = await proof.residency()
            await rm.stop_session(session.session_token, reason="completed", idempotency_key="adapter-check-stop")
            normal_cleanup = await provider.verify_cleanup()
            if not normal_cleanup:
                raise RuntimeError("model cleanup proof did not remove the resident model")
            try:
                await rm.submit(session.session_token, "stale", "attempt-1", b"Say hello.",
                    idempotency_key="adapter-check-stale", context_size=512)
            except ResourceManagerError as exc:
                if exc.failure.code != "scheduler_superseded":
                    raise
            else:
                raise RuntimeError("stale session token was accepted")
            evidence = {"status": "passed-candidate", "uuid": args.target_gpu_uuid,
                "requested_version": OLLAMA_VERSION, "actual_version": version,
                "small_completion": checks["small"], "upper_completion": checks["upper"],
                "residency": {"supervisor_hostpid": residency.supervisor.pid,
                               "runner_hostpids": len(residency.runners)},
                "foreign_baseline_process_count": getattr(proof, "baseline_process_count", 0),
                "cleanup_baseline": normal_cleanup,
                "cleanup_model_absent": normal_cleanup, "ownedgroupgone": False}
            return evidence
        finally:
            if session is not None and not normal_cleanup:
                try:
                    await rm.stop_session(session.session_token, reason="finally", idempotency_key="adapter-check-finally")
                except BaseException:
                    pass
            if provider is not None and not normal_cleanup:
                try:
                    await provider.unload()
                except BaseException:
                    pass
            gone = await _cleanup_group(daemon, drains)
            if not gone:
                raise RuntimeError("owned Ollama process group did not terminate")
            if evidence is not None:
                evidence["ownedgroupgone"] = gone
            retain = False
    finally:
        if not retain:
            import shutil
            shutil.rmtree(root)
        else:
            # Keep evidence needed to diagnose a failed cleanup or offloaded work.
            print(f"adapter check retained temporary storage: {root}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-gpu-uuid", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host-pid-namespace", action="store_true",
                        help="operator attests /proc is the NVML host PID namespace")
    args = parser.parse_args()
    result = asyncio.run(asyncio.wait_for(run(args), 360))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

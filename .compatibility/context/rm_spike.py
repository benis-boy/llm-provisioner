"""Bounded, candidate-only ResourceManager compatibility experiment.

The parent deliberately never imports torch.  The persistent child owns every
CUDA import and is killed as a process group if its RPC boundary stops making
progress.
"""
from __future__ import annotations

import argparse, asyncio, ctypes, hashlib, json, logging, os, queue, shutil, signal, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager, ResourceManagerError
from services.llm.resource_manager.protocol import EventKind, Provider, ProviderResponse
try:
    from artifacts import verify_manifest
except ModuleNotFoundError:
    from services.llm.provisioning.artifacts import verify_manifest
try:
    from tools.compatibility.model_runtime import MAX_FIXTURES, SMALL_FIXTURES, LoadedRuntime
except ModuleNotFoundError:
    from model_runtime import MAX_FIXTURES, SMALL_FIXTURES, LoadedRuntime

RPC_TIMEOUT = 30.0
LOG = logging.getLogger("compatibility.rm")
_SUBREAPER_LOCK = threading.Lock()
_SUBREAPER_ENABLED = False

def _enable_child_subreaper() -> None:
    """Make this Linux harness reap grandchildren orphaned by a killed child."""
    global _SUBREAPER_ENABLED
    with _SUBREAPER_LOCK:
        if _SUBREAPER_ENABLED:
            return
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
        _SUBREAPER_ENABLED = True

def _physical_uuid(value: object) -> str:
    value = value.decode() if isinstance(value, bytes) else str(value)
    if value.startswith("GPU-"): value = value[4:]
    if value.startswith("MIG-"): raise ValueError("MIG is not an exclusive physical GPU")
    import uuid
    return str(uuid.UUID(value))

def _nvml_baseline(target_uuid: str) -> tuple[object, set[int], str]:
    from pynvml import nvmlDeviceGetComputeRunningProcesses, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUUID, nvmlInit
    nvmlInit(); wanted = _physical_uuid(target_uuid)
    if nvmlDeviceGetCount() != 1: raise RuntimeError("exactly one NVML GPU must be exposed to the candidate")
    matches = [nvmlDeviceGetHandleByIndex(i) for i in range(nvmlDeviceGetCount())
               if _physical_uuid(nvmlDeviceGetUUID(nvmlDeviceGetHandleByIndex(i))) == wanted]
    if len(matches) != 1: raise RuntimeError("target GPU UUID is not exactly one NVML physical device")
    handle = matches[0]
    return handle, {int(p.pid) for p in nvmlDeviceGetComputeRunningProcesses(handle)}, wanted

def _nvml_processes(handle: object) -> set[int]:
    from pynvml import nvmlDeviceGetComputeRunningProcesses
    return {int(p.pid) for p in nvmlDeviceGetComputeRunningProcesses(handle)}

class ChildRPCProvider(Provider):
    def __init__(self, model: str, root: Path, *, command: list[str] | None = None, timeout: float = RPC_TIMEOUT, cleanup_check=None):
        self.model, self.root, self.timeout, self.cleanup_check = model, root, timeout, cleanup_check
        _enable_child_subreaper()
        self.process = subprocess.Popen(command or [sys.executable, __file__, "--child", model, "--root", str(root)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1, start_new_session=True)
        self._write_lock, self._pending, self._serial = threading.Lock(), {}, 0
        self._dead, self._closed = threading.Event(), False
        self._pgid = self.process.pid
        self._executing: set[str] = set()
        self._store_path: Path | None = None
        self._reader = threading.Thread(target=self._read, daemon=True); self._reader.start()
    def _read(self) -> None:
        assert self.process.stdout
        for line in self.process.stdout:
            try: response = json.loads(line)
            except (ValueError, TypeError): continue
            if response.get("event") == "execution_started":
                request_id = response.get("request_id")
                if isinstance(request_id, str): self._executing.add(request_id)
                continue
            waiter = self._pending.get(response.get("id"))
            if waiter:
                # A malformed/duplicate late reply must never wedge the one
                # reader responsible for every pending RPC.
                try: waiter.put_nowait(response)
                except queue.Full: pass
        self._dead.set()
        error = RuntimeError("model child RPC EOF/process death")
        for waiter in list(self._pending.values()):
            try: waiter.put_nowait(error)
            except queue.Full: pass
    def _call(self, operation: str, *, timeout: float | None = None, **arguments):
        with self._write_lock:
            if self._dead.is_set() or self.process.poll() is not None:
                self._dead.set()
                raise RuntimeError("model child RPC child is dead")
            self._serial += 1; ident = self._serial; waiter: queue.Queue = queue.Queue(1); self._pending[ident] = waiter
            try:
                assert self.process.stdin; self.process.stdin.write(json.dumps({"id": ident, "op": operation, **arguments}) + "\n"); self.process.stdin.flush()
            except Exception:
                self._pending.pop(ident, None); raise RuntimeError("model child RPC write failed")
        timeout = self.timeout if timeout is None else timeout
        try: response = waiter.get(timeout=timeout)
        except queue.Empty: raise RuntimeError(f"model child RPC {operation} timed out after {timeout}s")
        finally: self._pending.pop(ident, None)
        if isinstance(response, Exception): raise response
        if not response.get("ok"): raise RuntimeError(response.get("error", "child RPC failed"))
        return response.get("value")
    async def _async(self, operation: str, *, timeout: float | None = None, **arguments): return await asyncio.to_thread(self._call, operation, timeout=timeout, **arguments)
    async def validate(self, profile): await self._async("validate")
    async def load(self, profile): await self._async("load", timeout=240)
    async def ready(self): return await self._async("ready", timeout=30)
    async def snapshot(self):
        value = await self._async("snapshot", timeout=5)
        path = value.get("store_path") if isinstance(value, dict) else None
        if path: self._store_path = Path(path)
        return value
    async def execute(self, request_id, payload): return ProviderResponse((await self._async("execute", timeout=240, request_id=request_id, payload=payload.decode())).encode())
    async def cancel(self, request_id): await self._async("cancel", request_id=request_id)
    async def unload(self): await asyncio.to_thread(self._unload_and_close)
    def _unload_and_close(self):
        rpc_error = None
        if not self._dead.is_set() and self.process.poll() is None:
            try: self._call("unload", timeout=30)
            except RuntimeError as exc: rpc_error = exc
        self.close()
        if not self.process_group_gone():
            raise RuntimeError("model child process group survived unload")
        # A dead RPC boundary is expected after injected process loss. Group and
        # NVML verification remain authoritative; a dead advisory unload RPC
        # must not turn proved cleanup into a synthetic failure.
        if rpc_error and not self._dead.is_set():
            raise rpc_error
    async def verify_cleanup(self):
        return self.process.poll() is not None and self.process_group_gone() and (self.cleanup_check is None or self.cleanup_check())
    async def executing(self, request_id: str | None = None):
        return request_id in self._executing if request_id is not None else bool(self._executing)
    async def validate_input(self, payload, *, context_size, bucket_identity): await self._async("validate_input", payload=payload.decode(), context_size=context_size, bucket=bucket_identity)
    def close(self) -> None:
        if self._closed: return
        self._closed = True
        if self.process.poll() is None:
            try: self._call("shutdown", timeout=min(self.timeout, 2))
            except Exception: pass
        # The child is a session leader.  Its death alone is insufficient: an
        # Ollama descendant can retain GPU residency after an RPC/process loss.
        # Fence the whole owned group before a later model can be started.
        if not self.process_group_gone():
            try: os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            deadline = time.monotonic() + 2
            while not self.process_group_gone() and time.monotonic() < deadline:
                time.sleep(.02)
            if not self.process_group_gone():
                try: os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                deadline = time.monotonic() + 2
                while not self.process_group_gone() and time.monotonic() < deadline:
                    time.sleep(.02)
        try: self.process.wait(timeout=2)
        except subprocess.TimeoutExpired: pass
        # A killed leader can orphan an Ollama descendant. Reap only this owned
        # group after Popen has retained the direct child's exit status.
        deadline = time.monotonic() + 2
        while True:
            self._reap_adopted_group()
            if self.process_group_gone() or time.monotonic() >= deadline: break
            time.sleep(.02)
        if self._reader.is_alive(): self._reader.join(timeout=1)
        self._close_streams()
        if self.process_group_gone() and self._store_path:
            shutil.rmtree(self._store_path, ignore_errors=True)

    def process_group_gone(self) -> bool:
        try:
            os.killpg(self._pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def _reap_adopted_group(self) -> None:
        while True:
            try: pid, _ = os.waitpid(-self._pgid, os.WNOHANG)
            except ChildProcessError: return
            if pid == 0: return

    def inject_process_failure(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)

    def _close_streams(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        if not self._reader.is_alive() and self.process.stdout and not self.process.stdout.closed:
            self.process.stdout.close()

def _profile(model: ModelId, gpu_uuid="candidate-gpu", manifest_hash="candidate-manifest", model_hash="candidate-model") -> CapacityProfile:
    # Ollama is context-bound; Transformers/GECToR are explicit fixture buckets.
    return CapacityProfile(model, gpu_uuid, manifest_hash, model_hash, "candidate-runtime", "experimental-rm-spike", "unmeasured-harness", 1, 1, 1, 0, (SampleMetadata(1, 0, 0, 0, 0, ()),), context_size=512 if model is ModelId.SMOLLM else None, bucket_identity=None if model is ModelId.SMOLLM else "upper-fixture")

async def _events_until(rm, token, expected: set[EventKind], timeout: float, request_id: str | None = None, *, fail_fast: bool = True):
    async def collect():
        found = []
        async for event in rm.watch_progress(token):
            found.append(event)
            if event.kind is EventKind.FAILURE and fail_fast:
                raise RuntimeError(f"provider failure for {event.request_id}: {event.failure}")
            relevant = [e for e in found if request_id is None or e.request_id == request_id]
            if expected.issubset({e.kind for e in relevant}): return found
    return await asyncio.wait_for(collect(), timeout)

async def _run(args) -> dict:
    roots = {m: args.models_root / m for m in ("SmolLM", "CoEdIT", "GECToR")}
    document = json.loads(args.manifest.read_text(encoding="utf-8")); verify_manifest(document, roots)
    handle, baseline, target_uuid = _nvml_baseline(args.target_gpu_uuid)
    rm, evidence, providers = ResourceManager(cleanup_timeout=args.cleanup_timeout, stop_timeout=args.cleanup_timeout, load_timeout=args.timeout), [], []
    try:
        for model in ("SmolLM", "CoEdIT", "GECToR"):
            LOG.info("phase=load model=%s target_gpu=%s", model, target_uuid)
            entry = document["models"][model]; model_hash = hashlib.sha256(json.dumps(entry["files"], sort_keys=True).encode()).hexdigest()
            provider = ChildRPCProvider(model, roots[model], timeout=args.rpc_timeout,
                                        cleanup_check=lambda: _nvml_processes(handle) == baseline); providers.append(provider)
            session = await rm.start_session("compatibility-rm-spike", ModelId(model), _profile(ModelId(model), target_uuid, document["manifest_sha256"], model_hash), provider, idempotency_key=f"start-{model}")
            ready = await provider.ready()
            await provider.snapshot()
            if not ready.get("cuda_nvml_agree") or _physical_uuid(ready["gpu_uuid"]) != target_uuid: raise RuntimeError(f"{model} child CUDA/NVML UUID disagrees with selected GPU")
            LOG.info("phase=ready model=%s child_gpu=%s", model, ready["gpu_uuid"])
            fixture, bucket = SMALL_FIXTURES[model], None if model == "SmolLM" else "upper-fixture"
            submitted = await rm.submit(session.session_token, f"{model}-small", "attempt-1", fixture.encode(), idempotency_key=f"small-{model}", context_size=512 if model == "SmolLM" else None, bucket_identity=bucket)
            events = await _events_until(rm, session.session_token, {EventKind.RESPONSE_FINISHED}, args.timeout, f"{model}-small")
            if not submitted.accepted or any(e.kind is EventKind.FAILURE for e in events) or not any(e.result for e in events if e.kind is EventKind.RESPONSE_FINISHED): raise RuntimeError(f"{model} small fixture did not finish successfully")
            # The configured upper bucket is exercised, not represented as a model maximum.
            await rm.submit(session.session_token, f"{model}-upper", "attempt-1", MAX_FIXTURES[model].encode(), idempotency_key=f"upper-{model}", context_size=512 if model == "SmolLM" else None, bucket_identity=bucket)
            upper_events = await _events_until(rm, session.session_token, {EventKind.RESPONSE_FINISHED}, args.timeout, f"{model}-upper")
            if not any(e.kind is EventKind.RESPONSE_FINISHED and e.request_id == f"{model}-upper" and e.result for e in upper_events):
                raise RuntimeError(f"{model} upper fixture did not finish successfully")
            # Cancellation is issued only after the child reports execution, so
            # this checks the RM's late-result fence rather than queued cancel.
            await rm.submit(session.session_token, f"{model}-cancel", "attempt-1", fixture.encode(), idempotency_key=f"cancel-{model}", context_size=512 if model == "SmolLM" else None, bucket_identity=bucket)
            deadline = time.monotonic() + args.timeout
            while not await provider.executing(f"{model}-cancel"):
                if time.monotonic() >= deadline: raise RuntimeError(f"{model} child never began cancellable execution")
                await asyncio.sleep(.01)
            if args.inject_process_failure:
                failed_request = f"{model}-cancel"
                provider.inject_process_failure()
                failure_events = await _events_until(rm, session.session_token, {EventKind.FAILURE}, args.timeout, failed_request, fail_fast=False)
                if not any(e.kind is EventKind.FAILURE and e.request_id == failed_request and e.attempt == "attempt-1" for e in failure_events):
                    raise RuntimeError("process failure did not fail the exact active attempt")
                if any(e.kind is EventKind.RESPONSE_FINISHED and e.request_id == failed_request and e.result is not None for e in failure_events):
                    raise RuntimeError("process failure published a result for the failed attempt")
                # Core stop_session deliberately suppresses provider cleanup
                # failures while leaving RM unavailable. Record the observable
                # ownership checks immediately rather than claiming its hidden
                # internal exception is available at this boundary.
                await rm.stop_session(session.session_token, reason="process-failure", idempotency_key=f"stop-failure-{model}")
                LOG.info("phase=process_failure_post_stop model=%s group_gone=%s nvml_baseline=%s",
                         model, provider.process_group_gone(), _nvml_processes(handle) == baseline)
                try:
                    await rm.submit(session.session_token, f"{model}-stale-failure", "attempt-1", fixture.encode(),
                                    idempotency_key=f"stale-failure-{model}", context_size=512 if model == "SmolLM" else None,
                                    bucket_identity=bucket)
                except ResourceManagerError as exc:
                    if exc.failure.code != "scheduler_superseded": raise
                else:
                    raise RuntimeError("failed session token was accepted after process failure stop")
                if _nvml_processes(handle) != baseline:
                    raise RuntimeError(f"{model} process-failure cleanup did not restore target NVML baseline")
                evidence.append({"model": model, "process_failure_fenced": True, "profile": "p=1 synthetic/unmeasured"})
                continue
            if not await rm.cancel_request(session.session_token, f"{model}-cancel", idempotency_key=f"cancel-active-{model}"): raise RuntimeError("active cancellation was rejected")
            cancelled_events = await _events_until(rm, session.session_token, {EventKind.RESPONSE_FINISHED}, args.timeout, f"{model}-cancel")
            late = [e for e in cancelled_events if e.request_id == f"{model}-cancel" and e.kind is EventKind.RESPONSE_FINISHED]
            if not late or any(e.result is not None for e in late): raise RuntimeError(f"{model} late result crossed cancellation fence")
            LOG.info("phase=cancel_fenced model=%s", model)
            await rm.stop_session(session.session_token, reason="switch", idempotency_key=f"stop-{model}")
            try:
                await rm.submit(session.session_token, f"{model}-stale", "attempt-1", fixture.encode(),
                                idempotency_key=f"stale-{model}", context_size=512 if model == "SmolLM" else None,
                                bucket_identity=bucket)
            except ResourceManagerError as exc:
                if exc.failure.code != "scheduler_superseded": raise
            else:
                raise RuntimeError(f"{model} stale session token was accepted after switch")
            if _nvml_processes(handle) != baseline: raise RuntimeError(f"{model} cleanup did not restore target NVML baseline before switch")
            LOG.info("phase=cleanup_restored model=%s", model)
            evidence.append({"model": model, "small_and_upper_fixture": True, "profile": "p=1 synthetic/unmeasured"})
        return {"status": "passed-candidate", "models": evidence, "limitations": ["best-effort active interruption and capacity are unmeasured"]}
    finally:
        for provider in providers: await asyncio.to_thread(provider.close)

def _child(model: str, root: Path) -> int:
    runtime, lock = LoadedRuntime(model, root), threading.Lock()
    output_lock = threading.Lock()
    def reply(request, ok=True, value=None, error=None):
        with output_lock: print(json.dumps({"id": request["id"], "ok": ok, "value": value} if ok else {"id": request["id"], "ok": False, "error": error}), flush=True)
    def started(request_id):
        with output_lock: print(json.dumps({"event": "execution_started", "request_id": request_id}), flush=True)
    runtime.execution_started = started
    def execute(request):
        try: reply(request, value=runtime.execute(request["payload"], request["request_id"]))
        except Exception as exc: reply(request, False, error=str(exc))
    try:
        for line in sys.stdin:
            request = json.loads(line); op = request["op"]
            try:
                if op == "execute": threading.Thread(target=execute, args=(request,), daemon=True).start(); continue
                with lock:
                    if op == "validate": value = True
                    elif op == "load": runtime.load(); value = True
                    elif op == "ready": value = runtime.gpu_identity()
                    elif op == "validate_input": runtime.validate(request["payload"], request.get("context_size"), request.get("bucket")); value = True
                    elif op == "cancel": value = runtime.cancel(request["request_id"])
                    elif op == "executing": value = runtime.active_request is not None
                    elif op == "unload": runtime.close(); value = True
                    elif op == "verify_cleanup": value = runtime.verify_cleanup()
                    elif op == "shutdown": runtime.close(); value = True
                    elif op == "snapshot": value = runtime.snapshot()
                    else: raise ValueError(f"unknown operation: {op}")
                reply(request, value=value)
            except Exception as exc: reply(request, False, error=str(exc))
    finally: runtime.close()
    return 0

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--child", choices=("SmolLM", "CoEdIT", "GECToR")); p.add_argument("--root", type=Path); p.add_argument("--models-root", type=Path); p.add_argument("--manifest", type=Path); p.add_argument("--target-gpu-uuid"); p.add_argument("--timeout", type=float, default=120); p.add_argument("--rpc-timeout", type=float, default=30); p.add_argument("--cleanup-timeout", type=float, default=20); p.add_argument("--inject-process-failure", action="store_true"); a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    if a.child: return _child(a.child, a.root)
    if not a.models_root or not a.manifest or not a.target_gpu_uuid: p.error("--models-root, --manifest, and --target-gpu-uuid are required")
    print(json.dumps(asyncio.run(_run(a)), sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())

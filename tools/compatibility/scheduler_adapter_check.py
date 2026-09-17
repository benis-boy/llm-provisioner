"""Candidate-only QueueScheduler -> ResourceManager -> CoEdIT check.

This is deliberately separate from the production bootstrap path.  It uses an
explicitly unmeasured p1 profile, and therefore can report compatibility only;
it must never be used as readiness or capacity evidence.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from typing import Any
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.llm.provisioning.artifacts import SPECS, manifest, verify_manifest
from services.llm.provisioning.volume import provision
from services.llm.providers.coedit import CoEdITProvider
from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import LinuxGPUProof
from services.llm.providers.python_config import PythonProviderConfig
from services.llm.queue.contracts import ModelId
from services.llm.queue.results import LocalPublisher, ResultStore
from services.llm.queue.scheduler import DecodedPayload, DispatchContext, QueueScheduler
from services.llm.queue.store import QueueStore
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager
from services.llm.resource_manager.protocol import EventKind

try:
    from tools.compatibility import coedit_adapter_check as coedit
except ModuleNotFoundError:  # flat image imports
    import coedit_adapter_check as coedit  # type: ignore[no-redef]


def _profile(manifest_hash: str, model_hash: str, gpu: str) -> CapacityProfile:
    return CapacityProfile(
        ModelId.COEDIT, gpu, manifest_hash, model_hash, coedit._runtime_identity(),
        "candidate-coedit-provider", "unmeasured-scheduler-adapter-check",
        1, 1, 1, 0, (SampleMetadata(1, 0, 0, 0, 0, ()),),
        bucket_identity=coedit.BUCKET,
    )


class ObservableGateProvider:
    """Delegate all provider behavior while exposing an execution-entry gate."""

    def __init__(self, provider):
        self.provider = provider
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.gate = False
        self.completed = asyncio.Event()
        self.response = None

    def __getattr__(self, name):
        return getattr(self.provider, name)

    async def execute(self, request_id, payload):
        if self.gate:
            self.entered.set()
            await self.release.wait()
        self.response = await self.provider.execute(request_id, payload)
        self.completed.set()
        return self.response

    async def cancel(self, request_id):
        # This deliberately declines advisory provider cancellation only while
        # the gate holds execution before the real provider call.  It lets the
        # real delegate run after release so RM/scheduler result fencing, not
        # a claimed kernel interruption, is what this candidate observes.
        if not self.gate:
            await self.provider.cancel(request_id)


class CandidateFailure(RuntimeError):
    def __init__(self, stage: str, code: str, cleanup: dict[str, bool], unresolved: bool):
        super().__init__(code)
        self.stage, self.code, self.cleanup, self.unresolved = stage, code, cleanup, unresolved


def _require(condition: bool, diagnostics: dict[str, Any], code: str) -> None:
    if not condition:
        diagnostics["code"] = code
        raise RuntimeError(code)


def _candidate_manifest(path: Path, source: Path) -> tuple[dict, str]:
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict) or set(document) != {"schema", "models", "manifest_sha256"}:
        raise RuntimeError("candidate manifest is invalid")
    if manifest(document["models"]) != document or "CoEdIT" not in document["models"]:
        raise RuntimeError("candidate manifest digest or model is invalid")
    entry = document["models"]["CoEdIT"]
    if [item.get("path") for item in entry.get("files", ())] != list(SPECS["CoEdIT"]):
        raise RuntimeError("candidate CoEdIT file set is not exact")
    selected = manifest({"CoEdIT": entry})
    verify_manifest(selected, {"CoEdIT": source})
    model_hash = next(item["sha256"] for item in entry["files"] if item["path"] == "model.safetensors")
    return selected, model_hash


def _selected_file_identity(entry: dict) -> tuple[str, tuple[tuple[str, int, str], ...]]:
    """Compare immutable selected bytes, not source/runtime-root metadata."""
    return entry["model_id"], tuple((item["path"], item["size"], item["sha256"])
                                     for item in entry["files"])


async def _wait_for(predicate, timeout: float = 30.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("scheduler compatibility check timed out")
        await asyncio.sleep(.01)


async def _terminal_event(rm, session, request_id: str, attempt: str):
    async def collect():
        async for event in rm.watch_progress(session.session_token):
            if (event.session_token == session.session_token and event.generation == session.generation
                    and event.request_id == request_id and event.attempt == attempt
                    and event.kind in (EventKind.RESPONSE_FINISHED, EventKind.FAILURE)):
                return event
        raise RuntimeError("resource manager progress stream ended")
    return await asyncio.wait_for(collect(), 30)


async def _bounded_cleanup(awaitable: Any, timeout: float, pending: set[asyncio.Task[Any]], *,
                           require_true: bool = False) -> bool:
    task = asyncio.create_task(awaitable)
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout)
        return not task.cancelled() and task.exception() is None and (not require_true or task.result() is True)
    except BaseException:
        if not task.done():
            pending.add(task)
            def observe(completed: asyncio.Task[Any]) -> None:
                pending.discard(completed)
                if not completed.cancelled():
                    try:
                        completed.exception()
                    except BaseException:
                        pass
            task.add_done_callback(observe)
        elif not task.cancelled():
            try:
                task.exception()
            except BaseException:
                pass
        return False


async def _settle_cleanup(pending: set[asyncio.Task[Any]], timeout: float) -> bool:
    """Give retained cleanup a final bounded observation window before retention."""
    if not pending:
        return True
    _, unfinished = await asyncio.wait(tuple(pending), timeout=timeout)
    return not unfinished


def _aligned_response(response) -> bool:
    try:
        return coedit._check_response(type("Event", (), {"result": response.result})()) is True
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


async def _cleanup_owned(scheduler, rm, session, provider, proof, pending: set[asyncio.Task[Any]],
                         timeout: float = 10.0) -> dict[str, bool]:
    """Use QueueScheduler as the sole RM-stop owner, then observe its outcome."""
    if provider is not None:
        provider.release.set()
    scheduler_ok = await _bounded_cleanup(scheduler.stop("finally"), timeout, pending) if scheduler else True
    if not scheduler_ok:
        return {"scheduler": False, "rm_snapshot": False, "provider_verify": False, "gpu": False}
    if scheduler is None and rm is not None and session is not None:
        rm_ok = await _bounded_cleanup(
            rm.stop_session(session.session_token, reason="finally", idempotency_key="scheduler-adapter-finally"),
            timeout, pending)
    elif rm is not None:
        state = rm.snapshot()
        rm_ok = state.phase == "startup" and state.available and not state.session_present
    else:
        rm_ok = True
    if not rm_ok:
        return {"scheduler": True, "rm_snapshot": False, "provider_verify": False, "gpu": False}
    provider_ok = True
    if provider is not None:
        verify_ok = await _bounded_cleanup(provider.verify_cleanup(), timeout, pending, require_true=True)
        provider_ok = verify_ok
    gpu_ok = True
    if proof is not None:
        gpu_ok = await _bounded_cleanup(proof.cleanup(), timeout, pending, require_true=True)
    return {"scheduler": scheduler_ok, "rm_snapshot": rm_ok,
            "provider_verify": provider_ok, "gpu": gpu_ok}


async def _run(args: argparse.Namespace, diagnostics: dict[str, Any]) -> dict:
    stage = "manifest"
    if not args.host_pid_namespace:
        diagnostics["code"] = "host_pid_namespace_required"
        raise ValueError("host-pid-namespace")
    source = args.models_root / "CoEdIT"
    selected, model_hash = await asyncio.to_thread(_candidate_manifest, args.manifest, source)
    root = Path(tempfile.mkdtemp(prefix="scheduler-adapter-check-"))
    store = results = publisher = scheduler = rm = provider = session = proof = None
    cleanup_ok = False
    cleanup_pending: set[asyncio.Task[Any]] = set()
    try:
        stage = "provision"
        volume = await asyncio.to_thread(provision, {"CoEdIT": source}, root / "artifacts")
        _require(_selected_file_identity(selected["models"]["CoEdIT"])
                 == _selected_file_identity(volume["models"]["CoEdIT"]), diagnostics,
                 "provision_identity_mismatch")
        stage = "capture"
        proof = await asyncio.to_thread(LinuxGPUProof.capture, args.target_gpu_uuid, os.getpid(),
                                        Path("/proc"), None, host_pid_namespace=True)
        typed_proof = GPUProof(proof.identity, proof.cleanup, proof.residency,
                               expected_supervisor=proof.supervisor_identity)
        stage = "provider_construct"
        provider = ObservableGateProvider(CoEdITProvider(PythonProviderConfig(
            root / "artifacts", volume["manifest_sha256"], model_hash, args.target_gpu_uuid,
            coedit._runtime_identity(), "candidate-coedit-provider",
            bucket_identity=coedit.BUCKET, max_native_batch_size=1,
            max_input_tokens=128, max_output_tokens=64, gpu_proof=typed_proof)))
        rm = ResourceManager(cleanup_timeout=60, stop_timeout=60, load_timeout=240)
        profile = _profile(volume["manifest_sha256"], model_hash, args.target_gpu_uuid)
        store_path = root / "queue.sqlite"
        store = QueueStore(store_path, "scheduler-adapter-check", ModelId.COEDIT)
        results = ResultStore(root / "results")
        publisher = LocalPublisher(results, root / "publisher.sqlite")

        # The owner is established before acceptance; reopening retains the
        # same durable session identity and the request is still pre-dispatch.
        store.start_session("scheduler-adapter-pre-dispatch")
        request_payload = coedit._request("Short text.").decode("utf-8")
        store.enqueue("durable", request_payload, idempotency_key="durable")
        store.close()
        store = QueueStore(store_path, "scheduler-adapter-check", ModelId.COEDIT)
        scheduler = QueueScheduler(
            store, rm, profile, provider,
            decoder=lambda reference: DecodedPayload(
                reference.encode(), DispatchContext(context_size=None, bucket_identity=coedit.BUCKET)),
            result_store=results, publisher=publisher, loop_interval=.02, stop_timeout=5,
        )
        started = asyncio.get_running_loop().time()
        stage = "start"
        session = await scheduler.start()
        stage = "wait_normal"
        await _wait_for(lambda: store.get("durable")["status"] == "done")
        normal_elapsed_ms = round((asyncio.get_running_loop().time() - started) * 1000)
        handoff = store.db.execute("SELECT result_reference FROM handoffs WHERE request_id='durable'").fetchone()
        stage = "verify_normal"
        durable_result = results.read(handoff[0]) if handoff else b""
        _require(bool(durable_result), diagnostics, "normal_result_missing")
        _require(_aligned_response(type("Response", (), {"result": durable_result})()), diagnostics,
                 "normal_response_shape")
        _require(provider.response is not None and durable_result == provider.response.result, diagnostics,
                 "normal_result_delegate_mismatch")
        handoff = store.db.execute(
            "SELECT request_id,token,result_reference,idempotency_key FROM handoffs WHERE request_id='durable'"
        ).fetchone()
        receipt = publisher.db.execute(
            "SELECT idempotency_key,result_reference FROM publication_receipts"
        ).fetchone()
        expected_key = f"handoff:{handoff['request_id']}:{handoff['token']}"
        _require(receipt is not None and tuple(receipt) == (expected_key, handoff["result_reference"])
                 and handoff["idempotency_key"] == expected_key, diagnostics, "receipt_mismatch")
        receipt_count = publisher.db.execute("SELECT COUNT(*) FROM publication_receipts").fetchone()[0]
        _require(receipt_count == 1, diagnostics, "receipt_count_mismatch")

        provider.gate = True
        provider.entered.clear()
        provider.completed.clear()
        provider.response = None
        stage = "cancel_gate"
        await scheduler.enqueue("cancelled", request_payload, idempotency_key="cancelled")
        await asyncio.wait_for(provider.entered.wait(), 30)
        cancelled_attempt = store.db.execute(
            "SELECT token FROM attempts WHERE request_id='cancelled' AND active=1 ORDER BY started DESC LIMIT 1"
        ).fetchone()[0]
        stage = "cancel"
        await scheduler.cancel("cancelled")
        provider.release.set()
        stage = "late_delegate"
        await asyncio.wait_for(provider.completed.wait(), 30)
        stage = "late_rm_event"
        late_event = await _terminal_event(rm, session, "cancelled", cancelled_attempt)
        stage = "verify_cancel"
        _require(late_event.kind is EventKind.RESPONSE_FINISHED and provider.response is not None,
                 diagnostics, "late_delegate_missing")
        _require(_aligned_response(provider.response), diagnostics, "late_response_shape")
        await _wait_for(lambda: store.get("cancelled")["status"] == "cancelled")
        cancelled_handoff = store.db.execute("SELECT 1 FROM handoffs WHERE request_id='cancelled'").fetchone()
        cancelled_receipts = publisher.db.execute("SELECT COUNT(*) FROM publication_receipts").fetchone()[0]
        _require(not cancelled_handoff and cancelled_receipts == 1, diagnostics, "cancel_fence_broken")
        timing = store.db.execute("SELECT gpu_ms,gpu_complete FROM attempts WHERE request_id='durable'").fetchone()
        return {
            "status": "passed-candidate", "candidate": True, "profile": "unmeasured",
            "profile_eligible": False, "model": "CoEdIT", "manifest_sha256": volume["manifest_sha256"],
            "model_sha256": model_hash, "gpu_uuid": args.target_gpu_uuid,
            "normal_status": "done", "normal_elapsed_ms": normal_elapsed_ms,
            "receipt_count": receipt_count, "cancelled_status": "cancelled",
            "cancelled_receipt_count": cancelled_receipts, "cancelled_result_published": False,
            "gpu_ms": timing[0], "gpu_timing_complete": bool(timing[1]), "cleanup": True,
        }
    finally:
        cleanup_stage = stage
        cleanup = {"scheduler": False, "rm_snapshot": False, "provider_verify": False, "gpu": False}
        try:
            cleanup = await _cleanup_owned(scheduler, rm, session, provider, proof, cleanup_pending)
            settled = await _settle_cleanup(cleanup_pending, 1.0)
            unresolved_cleanup = bool(cleanup_pending)
            cleanup_ok = all(cleanup.values()) and settled and not unresolved_cleanup
        except BaseException:
            cleanup_ok = False
            unresolved_cleanup = bool(cleanup_pending)
        diagnostics["stage"] = stage
        diagnostics["cleanup"] = cleanup
        diagnostics["cleanup_unresolved"] = unresolved_cleanup
        # Do not close SQLite handles while a timed-out lifecycle task could
        # still be using them.  Retain all evidence until ownership is settled.
        if cleanup_ok:
            if publisher is not None: publisher.close()
            if store is not None: store.close()
        if not cleanup_ok:
            print("scheduler adapter check cleanup_retained=true", file=sys.stderr)
            if sys.exc_info()[0] is None:
                diagnostics["code"] = "cleanup_unproved"
                raise CandidateFailure(cleanup_stage, "cleanup_unproved", cleanup, unresolved_cleanup)
        else:
            shutil.rmtree(root, ignore_errors=True)


async def run(args: argparse.Namespace) -> dict:
    diagnostics: dict[str, Any] = {"stage": "manifest", "cleanup": {}, "cleanup_unresolved": False}
    try:
        return await _run(args, diagnostics)
    except CandidateFailure:
        raise
    except asyncio.TimeoutError:
        raise CandidateFailure(diagnostics["stage"], diagnostics.get("code", "stage_timeout"),
                               diagnostics["cleanup"], diagnostics["cleanup_unresolved"]) from None
    except BaseException:
        raise CandidateFailure(diagnostics["stage"], diagnostics.get("code", "unexpected"),
                               diagnostics["cleanup"], diagnostics["cleanup_unresolved"]) from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-gpu-uuid", required=True)
    parser.add_argument("--host-pid-namespace", action="store_true")
    try:
        result = asyncio.run(asyncio.wait_for(run(parser.parse_args()), 360))
    except CandidateFailure as exc:
        # Keep operator output bounded and free of prompts, results, paths, and
        # exception payloads from provider libraries.
        print(json.dumps({"status": "failed-candidate", "stage": exc.stage, "code": exc.code,
                           "cleanup": exc.cleanup, "cleanup_unresolved": exc.unresolved}, separators=(",", ":")))
        return 1
    except BaseException:
        print(json.dumps({"status": "failed-candidate", "stage": "runner", "code": "runner_timeout",
                          "cleanup": {}}, separators=(",", ":")))
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

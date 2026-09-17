"""Bounded, candidate-only real GECToR adapter check.

This is an offline smoke harness, not a benchmark.  It runs the installed
GECToR provider behind ResourceManager and reports only identity and check
counts; capacity remains explicitly ``unmeasured``.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.llm.provisioning.artifacts import SPECS, manifest, verify_manifest
from services.llm.provisioning.volume import provision
from services.llm.providers.config import GPUProof
from services.llm.providers.gector import GECToRProvider
from services.llm.providers.gector_config import GECToRProviderConfig
from services.llm.providers.gpu import LinuxGPUProof
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager, ResourceManagerError
from services.llm.resource_manager.protocol import EventKind

UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")
ADAPTER = "candidate-gector-provider"
BUCKET = "gector:p1:tokens128:keep0:min0:iterations1:batch1:float32"


def _cuda_runtime(version: str) -> str:
    match = re.search(r"\+cu(\d{2,3})(?:\D|$)", version)
    return "none" if not match else f"{match.group(1)[:-1]}.{match.group(1)[-1]}"


def _runtime_versions() -> dict[str, str]:
    names = ("gector", "torch", "transformers", "tokenizers", "safetensors")
    versions = {name: importlib.metadata.version(name) for name in names}
    versions["cuda"] = _cuda_runtime(versions["torch"])
    return versions


def _runtime_identity() -> str:
    versions = _runtime_versions()
    return "candidate:" + ";".join(f"{name}={versions[name]}" for name in
        ("gector", "torch", "transformers", "tokenizers", "safetensors")) + f";cuda={versions['cuda']}"


def _profile(manifest_hash: str, model_hash: str, gpu: str) -> CapacityProfile:
    return CapacityProfile(ModelId.GECTOR, gpu, manifest_hash, model_hash,
        _runtime_identity(), ADAPTER, "unmeasured-gector-adapter-check", 1, 1, 1, 0,
        (SampleMetadata(1, 0, 0, 0, 0, ()),), bucket_identity=BUCKET)


def _selected(entry: dict) -> tuple[str, tuple[tuple[str, int, str], ...]]:
    files = entry.get("files")
    if not isinstance(files, list):
        raise RuntimeError("candidate GECToR files are invalid")
    return entry.get("model_id"), tuple((Path(x.get("path")).as_posix(), x.get("size"), x.get("sha256")) for x in files)


def _candidate_manifest(path: Path, source: Path) -> tuple[dict, str]:
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict) or set(document) != {"schema", "models", "manifest_sha256"}:
        raise RuntimeError("candidate manifest is invalid")
    if manifest(document["models"]) != document:
        raise RuntimeError("candidate manifest digest is invalid")
    entry = document["models"].get("GECToR")
    if entry is None or [x.get("path") for x in entry.get("files", ())] != list(SPECS["GECToR"]):
        raise RuntimeError("candidate GECToR file set is not exact")
    selected = manifest({"GECToR": entry})
    verify_manifest(selected, {"GECToR": source})
    model_hash = next(x["sha256"] for x in entry["files"] if x["path"] == "model.safetensors")
    return selected, model_hash


def _request(text: str) -> bytes:
    return json.dumps({"texts": [text], "keep_confidence": 0.0,
                       "min_error_prob": 0.0, "n_iteration": 1, "batch_size": 1},
                      separators=(",", ":")).encode()


def _valid_response(event) -> bool:
    if not event.result:
        return False
    value = json.loads(event.result)
    return isinstance(value, dict) and set(value) == {"texts"} and isinstance(value["texts"], list) and len(value["texts"]) == 1 and isinstance(value["texts"][0], str) and bool(value["texts"][0])


async def _events(rm, session, request_id: str, attempt: str) -> list:
    result = []
    async for event in rm.watch_progress(session.session_token):
        result.append(event)
        if (event.session_token == session.session_token and event.generation == session.generation
                and event.request_id == request_id and event.attempt == attempt
                and event.kind in (EventKind.RESPONSE_FINISHED, EventKind.FAILURE)):
            return result
    raise RuntimeError("resource manager progress stream ended")


async def run(args: argparse.Namespace) -> dict:
    if not args.host_pid_namespace:
        raise ValueError("--host-pid-namespace operator attestation is required")
    if not isinstance(args.target_gpu_uuid, str) or not UUID.fullmatch(args.target_gpu_uuid):
        raise ValueError("target GPU UUID is invalid")
    source = args.models_root / "GECToR"
    if not source.is_dir():
        raise RuntimeError("selected GECToR artifact is missing")
    selected, model_hash = await asyncio.to_thread(_candidate_manifest, args.manifest, source)
    root = Path(tempfile.mkdtemp(prefix="gector-adapter-check-"))
    keep, worker_may_exist, cleaned = True, False, False
    rm, provider, session = ResourceManager(cleanup_timeout=60, stop_timeout=60, load_timeout=240), None, None
    try:
        volume = root / "artifacts"
        document = await asyncio.to_thread(provision, {"GECToR": source}, volume)
        if _selected(selected["models"]["GECToR"]) != _selected(document["models"]["GECToR"]):
            raise RuntimeError("candidate selected manifest differs from provisioned manifest")
        proof = await asyncio.to_thread(LinuxGPUProof.capture, args.target_gpu_uuid, os.getpid(), Path("/proc"), None, host_pid_namespace=True)
        gpu_proof = GPUProof(proof.identity, proof.cleanup, proof.residency, expected_supervisor=proof.supervisor_identity)
        config = GECToRProviderConfig(volume, document["manifest_sha256"], model_hash, args.target_gpu_uuid,
            _runtime_identity(), ADAPTER, gpu_proof=gpu_proof)
        provider = GECToRProvider(config)
        profile = _profile(document["manifest_sha256"], model_hash, args.target_gpu_uuid)
        worker_may_exist = True
        session = await rm.start_session("candidate-gector-adapter-check", ModelId.GECTOR, profile, provider, idempotency_key="gector-adapter-start")

        if args.inject_process_failure:
            entered = asyncio.Event()
            real = provider
            class ProcessLossProvider:
                def __getattr__(self, name): return getattr(real, name)
                async def execute(self, request_id, payload):
                    entered.set()
                    real.worker._signal_owned(signal.SIGKILL)
                    return await real.execute(request_id, payload)
            rm._provider = ProcessLossProvider()  # noqa: SLF001
            await rm.submit(session.session_token, "failure", "attempt-1", _request("Short text."), idempotency_key="gector-adapter-failure", bucket_identity=BUCKET)
            await asyncio.wait_for(entered.wait(), 10)
            events = await asyncio.wait_for(_events(rm, session, "failure", "attempt-1"), 120)
            if not any(e.kind is EventKind.FAILURE and e.result is None for e in events):
                raise RuntimeError("owned worker loss produced a result")
            await rm.stop_session(session.session_token, reason="injected-process-failure", idempotency_key="gector-adapter-stop")
            await provider.unload(); cleaned = await provider.verify_cleanup()
            if not cleaned: raise RuntimeError("failure cleanup was not proved")
            keep = False
            return {"status": "passed-candidate-failure-cleanup", "model": "GECToR", "profile": "unmeasured", "failure_count": 1, "cleanup": True, "gpu_uuid": args.target_gpu_uuid, "manifest_sha256": document["manifest_sha256"], "model_sha256": model_hash, "runtime": _runtime_versions()}

        counts = {"successful": 0, "rejected": 0}
        await rm.submit(session.session_token, "near-bucket", "attempt-1", _request("word " * 110), idempotency_key="gector-adapter-near-bucket", bucket_identity=BUCKET)
        events = await asyncio.wait_for(_events(rm, session, "near-bucket", "attempt-1"), 120)
        finished = next((e for e in events if e.kind is EventKind.RESPONSE_FINISHED), None)
        if not finished or not _valid_response(finished): raise RuntimeError("GECToR response was not aligned")
        counts["successful"] += 1
        try:
            await rm.submit(session.session_token, "overlong", "attempt-1", _request("word " * 160), idempotency_key="gector-adapter-overlong", bucket_identity=BUCKET)
        except ResourceManagerError as exc:
            if exc.failure.code != "invalid_input": raise
            counts["rejected"] += 1
        else: raise RuntimeError("overlong GECToR request was accepted")
        for request_id, text in (("valid-after-rejection", "Short text."),):
            await rm.submit(session.session_token, request_id, "attempt-1", _request(text), idempotency_key="gector-adapter-" + request_id, bucket_identity=BUCKET)
            events = await asyncio.wait_for(_events(rm, session, request_id, "attempt-1"), 120)
            finished = next((e for e in events if e.kind is EventKind.RESPONSE_FINISHED), None)
            if not finished or not _valid_response(finished): raise RuntimeError("GECToR response was not aligned")
            counts["successful"] += 1
        residency = await proof.residency()
        if len(residency.runners) != 1 or provider.worker is None or provider.worker.child_identity not in residency.runners:
            raise RuntimeError("exactly one owned worker GPU resident was not proved")
        await rm.stop_session(session.session_token, reason="completed", idempotency_key="gector-adapter-stop")
        await provider.unload(); cleaned = await provider.verify_cleanup()
        if not cleaned: raise RuntimeError("GECToR cleanup was not proved")
        try:
            await rm.submit(session.session_token, "stale", "attempt-1", _request("Short text."), idempotency_key="gector-adapter-stale", bucket_identity=BUCKET)
        except ResourceManagerError as exc:
            if exc.failure.code != "scheduler_superseded": raise
        else: raise RuntimeError("stale session token was accepted")
        keep = False
        return {"status": "passed-candidate", "model": "GECToR", "profile": "unmeasured", "successful_count": counts["successful"], "rejected_count": counts["rejected"], "runner_count": 1, "cleanup": True, "stale_rejected": True, "gpu_uuid": args.target_gpu_uuid, "manifest_sha256": document["manifest_sha256"], "model_sha256": model_hash, "runtime": _runtime_versions()}
    finally:
        if session is not None and not cleaned:
            try: await rm.stop_session(session.session_token, reason="finally", idempotency_key="gector-adapter-finally")
            except BaseException: pass
        if provider is not None and not cleaned:
            try: await provider.unload(); cleaned = await provider.verify_cleanup()
            except BaseException: pass
        if not worker_may_exist or not keep or cleaned: shutil.rmtree(root, ignore_errors=True)
        elif root.exists(): print("gector adapter check retained temporary storage", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-gpu-uuid", required=True); parser.add_argument("--host-pid-namespace", action="store_true"); parser.add_argument("--inject-process-failure", action="store_true")
    print(json.dumps(asyncio.run(asyncio.wait_for(run(parser.parse_args()), 360)), sort_keys=True, separators=(",", ":")))
    return 0

if __name__ == "__main__": raise SystemExit(main())

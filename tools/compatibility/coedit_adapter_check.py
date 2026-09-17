"""Bounded, candidate-only real CoEdIT adapter check.

This harness intentionally exercises the installed Python adapter through the
ResourceManager.  It is not a benchmark: its profile is explicitly
``unmeasured`` and its output contains no prompt, completion, or process table.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import shutil
import sys
import tempfile

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
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager, ResourceManagerError
from services.llm.resource_manager.protocol import EventKind

UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")
ADAPTER = "candidate-coedit-provider"
BUCKET = "coedit:p1:input128:output64:float16:beams1:nosample"


def _bucket(native_batch_size: int) -> str:
    return f"coedit:p{native_batch_size}:input128:output64:float16:beams1:nosample"


def _profile(manifest_hash: str, model_hash: str, gpu: str, native_batch_size: int = 1) -> CapacityProfile:
    return CapacityProfile(ModelId.COEDIT, gpu, manifest_hash, model_hash,
        _runtime_identity(), ADAPTER, "unmeasured-coedit-adapter-check", native_batch_size,
        native_batch_size, native_batch_size, 0,
        (SampleMetadata(native_batch_size, 0, 0, 0, 0, ()),), bucket_identity=_bucket(native_batch_size))


def _runtime_identity() -> str:
    versions = _runtime_versions()
    return "candidate:" + ";".join(
        f"{name}={versions[name]}" for name in ("torch", "transformers", "tokenizers", "safetensors")
    ) + f";cuda={versions['cuda']}"


def _cuda_runtime(torch_version: str) -> str:
    """Derive the packaged CUDA runtime without importing Torch in the parent."""
    match = re.search(r"\+cu(\d{2,3})(?:\D|$)", torch_version)
    if not match:
        return "none"
    digits = match.group(1)
    return f"{digits[:-1]}.{digits[-1]}"


def _runtime_versions() -> dict[str, str]:
    names = ("torch", "transformers", "tokenizers", "safetensors")
    versions = {name: importlib.metadata.version(name) for name in names}
    return {**versions, "cuda": _cuda_runtime(versions["torch"])}


def _selected_file_identity(entry: dict) -> tuple[str, tuple[tuple[str, int, str], ...]]:
    """The selected files, not provenance roots, are the artifact contract."""
    files = entry.get("files")
    if not isinstance(files, list):
        raise RuntimeError("candidate CoEdIT files are invalid")
    return (entry.get("model_id"), tuple(
        (item.get("path"), item.get("size"), item.get("sha256")) for item in files
    ))


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
    model_hash = next(item["sha256"] for item in entry["files"]
                       if item["path"] == "model.safetensors")
    return selected, model_hash


async def _events(rm, session, request_id: str, attempt: str) -> list:
    async def collect():
        result = []
        async for event in rm.watch_progress(session.session_token):
            result.append(event)
            if (event.session_token == session.session_token and event.generation == session.generation
                    and event.request_id == request_id and event.attempt == attempt
                    and event.kind in (EventKind.RESPONSE_FINISHED, EventKind.FAILURE)):
                return result
        raise RuntimeError("resource manager progress stream ended")
    return await asyncio.wait_for(collect(), 120)


def _request(text: str) -> bytes:
    return json.dumps({"instruction": "Improve the grammar.", "texts": [text]},
                      separators=(",", ":"), ensure_ascii=True).encode()


def _check_response(event) -> bool:
    if not event.result:
        return False
    value = json.loads(event.result)
    return (isinstance(value, dict) and set(value) == {"texts"} and
            isinstance(value["texts"], list) and len(value["texts"]) == 1 and
            isinstance(value["texts"][0], str) and bool(value["texts"][0]))


def _check_batch_observation(observations, drops: int, request_ids: tuple[str, ...], batch_size: int) -> bool:
    """Accept only the evidence that proves this particular RM wave batched."""
    if drops != 0 or len(observations) != 1:
        return False
    observation = observations[0]
    required = {"batch_size", "execution_started", "execution_ended", "cuda_synchronized", "allocator", "request_ids"}
    if not isinstance(observation, dict) or set(observation) != required:
        return False
    allocator = observation["allocator"]
    allocator_keys = {"baseline_allocated", "baseline_reserved", "peak_allocated", "peak_reserved", "final_allocated", "final_reserved"}
    values = ({key: allocator[key] for key in allocator_keys} if isinstance(allocator, dict)
              else {key: getattr(allocator, key, None) for key in allocator_keys})
    allocator_valid = ((isinstance(allocator, dict) and set(allocator) == allocator_keys) or type(allocator).__name__ == "AllocatorObservation") and all(
        type(values[key]) is int and values[key] >= 0 for key in allocator_keys)
    allocator_valid = allocator_valid and values["baseline_allocated"] <= values["baseline_reserved"]
    allocator_valid = allocator_valid and values["peak_allocated"] <= values["peak_reserved"]
    allocator_valid = allocator_valid and values["final_allocated"] <= values["final_reserved"]
    allocator_valid = allocator_valid and values["peak_allocated"] >= values["baseline_allocated"]
    allocator_valid = allocator_valid and values["peak_reserved"] >= values["baseline_reserved"]
    return (observation["batch_size"] == batch_size
            and observation["request_ids"] == request_ids
            and observation["cuda_synchronized"] is True and allocator_valid)


async def _wave(rm, session, requests: tuple[tuple[str, str], ...], bucket: str) -> list:
    """Submit independent requests concurrently through the real RM boundary."""
    await asyncio.gather(*(rm.submit(session.session_token, request_id, "attempt-1", _request(text),
                                     idempotency_key="coedit-adapter-" + request_id,
                                     bucket_identity=bucket)
                           for request_id, text in requests))
    events = []
    for request_id, _ in requests:
        events.append(await _events(rm, session, request_id, "attempt-1"))
    return events


async def run(args: argparse.Namespace) -> dict:
    if not args.host_pid_namespace:
        raise ValueError("--host-pid-namespace operator attestation is required")
    if not isinstance(args.target_gpu_uuid, str) or not UUID.fullmatch(args.target_gpu_uuid):
        raise ValueError("target GPU UUID is invalid")
    native_batch_size = getattr(args, "native_batch_size", 1)
    if type(native_batch_size) is not int or native_batch_size not in (1, 2):
        raise ValueError("--native-batch-size must be 1 or 2")
    if native_batch_size == 2 and args.inject_process_failure:
        raise ValueError("--inject-process-failure cannot be combined with --native-batch-size 2")
    bucket = _bucket(native_batch_size)
    source = args.models_root / "CoEdIT"
    if not source.is_dir():
        raise RuntimeError("selected CoEdIT artifact is missing")

    selected, model_hash = await asyncio.to_thread(_candidate_manifest, args.manifest, source)
    root = Path(tempfile.mkdtemp(prefix="coedit-adapter-check-"))
    keep = True
    rm = ResourceManager(cleanup_timeout=60, stop_timeout=60, load_timeout=240)
    provider = None
    session = None
    normal_cleanup = False
    worker_may_exist = False
    try:
        volume = root / "artifacts"
        document = await asyncio.to_thread(provision, {"CoEdIT": source}, volume)
        if _selected_file_identity(selected["models"]["CoEdIT"]) != _selected_file_identity(document["models"]["CoEdIT"]):
            raise RuntimeError("candidate selected manifest differs from provisioned manifest")
        # Capture before PythonWorker.start: no worker GPU activity is admitted
        # before the baseline and parent supervisor identity are fenced.
        proof = await asyncio.to_thread(LinuxGPUProof.capture, args.target_gpu_uuid,
            os.getpid(), Path("/proc"), None, host_pid_namespace=True)
        gpu_proof = GPUProof(proof.identity, proof.cleanup, proof.residency,
                             expected_supervisor=proof.supervisor_identity)
        config = PythonProviderConfig(volume, document["manifest_sha256"], model_hash,
            args.target_gpu_uuid, _runtime_identity(), ADAPTER, bucket_identity=bucket,
            max_native_batch_size=native_batch_size, max_input_tokens=128, max_output_tokens=64,
            gpu_proof=gpu_proof)
        provider = CoEdITProvider(config)
        profile = _profile(document["manifest_sha256"], model_hash, args.target_gpu_uuid, native_batch_size)
        # Loading can create a child before it reports failure, so retain the
        # temporary volume unless cleanup is subsequently positively proved.
        worker_may_exist = True
        session = await rm.start_session("candidate-coedit-adapter-check", ModelId.COEDIT,
            profile, provider, idempotency_key="coedit-adapter-start")

        if args.inject_process_failure:
            # Enter the installed provider path first, then kill only its
            # identity-fenced worker group. This is not a synthetic response.
            entered = asyncio.Event()
            real_provider = provider
            class ProcessLossProvider:
                def __init__(self, wrapped): self.wrapped = wrapped
                def __getattr__(self, name): return getattr(self.wrapped, name)
                async def execute(self, request_id, payload):
                    entered.set()
                    assert self.wrapped.worker is not None
                    self.wrapped.worker._signal_owned(signal.SIGKILL)  # noqa: SLF001
                    return await self.wrapped.execute(request_id, payload)
            rm._provider = ProcessLossProvider(real_provider)  # noqa: SLF001
            await rm.submit(session.session_token, "failure", "attempt-1", _request("Short text."),
                            idempotency_key="coedit-adapter-failure", bucket_identity=bucket)
            await asyncio.wait_for(entered.wait(), 10)
            events = await _events(rm, session, "failure", "attempt-1")
            if not any(event.kind is EventKind.FAILURE for event in events):
                raise RuntimeError("owned worker loss produced a result")
            failure = next(event for event in events if event.kind is EventKind.FAILURE)
            if (failure.session_token != session.session_token or failure.generation != session.generation
                    or failure.request_id != "failure" or failure.attempt != "attempt-1"
                    or failure.result is not None):
                raise RuntimeError("owned worker loss was not exactly fenced")
            await rm.stop_session(session.session_token, reason="injected-process-failure",
                                  idempotency_key="coedit-adapter-stop")
            await provider.unload()
            normal_cleanup = await provider.verify_cleanup()
            if not normal_cleanup:
                raise RuntimeError("failure cleanup was not proved")
            keep = False
            return {"status": "passed-candidate-failure-cleanup", "model": "CoEdIT",
                    "profile": "unmeasured", "failure_result": False,
                    "cleanup": True, "gpu_uuid": args.target_gpu_uuid,
                    "manifest_sha256": document["manifest_sha256"], "model_sha256": model_hash,
                    "runtime": _runtime_versions()}

        checks = {}
        waves = (("small", "This sentence needs improvement."), ("near-bucket", "word " * 100))
        if native_batch_size == 2:
            wave_results = []
            for prefix, text in waves:
                requests = ((prefix + "-a", text), (prefix + "-b", text + " now."))
                event_sets = await _wave(rm, session, requests, bucket)
                for (request_id, _), events in zip(requests, event_sets):
                    finished = next((event for event in events if event.session_token == session.session_token
                                      and event.generation == session.generation and event.request_id == request_id
                                      and event.attempt == "attempt-1" and event.kind is EventKind.RESPONSE_FINISHED), None)
                    checks[request_id] = finished is not None and _check_response(finished)
                    if not checks[request_id]:
                        raise RuntimeError("CoEdIT response was not nonempty and aligned")
                observations = tuple(provider.drain_batch_observations())
                drops = provider.batch_observation_drops()
                ids = tuple(request_id for request_id, _ in requests)
                if not _check_batch_observation(observations, drops, ids, 2):
                    raise RuntimeError("independent RM requests did not produce one synchronized native batch2")
                wave_results.append(observations[0])
            batch_sizes = [item["batch_size"] for item in wave_results]
            synchronized = all(item["cuda_synchronized"] is True for item in wave_results)
        else:
            # Legacy p1 mode intentionally retains the original serialized scenario.
            for request_id, text in waves:
                await rm.submit(session.session_token, request_id, "attempt-1", _request(text),
                                idempotency_key="coedit-adapter-" + request_id, bucket_identity=bucket)
                events = await _events(rm, session, request_id, "attempt-1")
                finished = next((event for event in events if event.session_token == session.session_token
                                  and event.generation == session.generation and event.request_id == request_id
                                  and event.attempt == "attempt-1" and event.kind is EventKind.RESPONSE_FINISHED), None)
                checks[request_id] = finished is not None and _check_response(finished)
                if not checks[request_id]:
                    raise RuntimeError("CoEdIT response was not nonempty and aligned")
                if hasattr(provider, "drain_batch_observations"):
                    provider.drain_batch_observations()
            batch_sizes, synchronized = [1, 1], True
        residency = await proof.residency()
        if len(residency.runners) != 1 or provider.worker is None or provider.worker.child_identity not in residency.runners:
            raise RuntimeError("exactly one owned worker GPU resident was not proved")
        await rm.stop_session(session.session_token, reason="completed", idempotency_key="coedit-adapter-stop")
        await provider.unload()
        normal_cleanup = await provider.verify_cleanup()
        if not normal_cleanup:
            raise RuntimeError("CoEdIT cleanup was not proved")
        try:
            await rm.submit(session.session_token, "stale", "attempt-1", _request("Short text."),
                            idempotency_key="coedit-adapter-stale", bucket_identity=bucket)
        except ResourceManagerError as exc:
            if exc.failure.code != "scheduler_superseded":
                raise
        else:
            raise RuntimeError("stale session token was accepted")
        keep = False
        return {"status": "passed-candidate", "model": "CoEdIT", "profile": "unmeasured",
                 "bucket": bucket, "native_batch_size": native_batch_size, "batch_sizes": batch_sizes,
                 "request_count": len(checks), "synchronized": synchronized,
                 "small": checks.get("small", True), "near_bucket": checks.get("near-bucket", True),
                 "gpu_uuid": args.target_gpu_uuid, "runner_count": 1, "cleanup": True,
                 "stale_rejected": True, "manifest_sha256": document["manifest_sha256"],
                 "model_sha256": model_hash, "runtime": _runtime_versions()}
    finally:
        cleanup_errors = []
        if session is not None and not normal_cleanup:
            try:
                await rm.stop_session(session.session_token, reason="finally", idempotency_key="coedit-adapter-finally")
            except BaseException as exc:
                cleanup_errors.append("rm_stop_failed:" + type(exc).__name__)
        if provider is not None and not normal_cleanup:
            try:
                await provider.unload()
            except BaseException as exc:
                cleanup_errors.append("provider_unload_failed:" + type(exc).__name__)
            else:
                try:
                    if await provider.verify_cleanup() is not True:
                        cleanup_errors.append("provider_cleanup_unproved")
                except BaseException as exc:
                    cleanup_errors.append("cleanup_probe_failed:" + type(exc).__name__)
        if not worker_may_exist or not keep or normal_cleanup:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print("coedit adapter check retained temporary storage", file=sys.stderr)
        if cleanup_errors:
            raise RuntimeError("cleanup failed: " + ",".join(cleanup_errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-gpu-uuid", required=True)
    parser.add_argument("--host-pid-namespace", action="store_true")
    parser.add_argument("--inject-process-failure", action="store_true")
    parser.add_argument("--native-batch-size", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    result = asyncio.run(asyncio.wait_for(run(args), 360))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

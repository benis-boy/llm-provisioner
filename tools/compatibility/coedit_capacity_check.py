"""Offline, real-RM CoEdIT capacity candidate (never writes a profile)."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil
import hashlib
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from services.llm.provisioning.capacity import (CapacityEvidenceError, MemorySample, Wave, measure_capacity,
    discover_memory, sample_overlaps_execution, THROUGHPUT_MAX_PARALLELISM, DISCOVERY_MAX_PARALLELISM)
from services.llm.providers.coedit_batch import AllocatorObservation
from services.llm.provisioning.volume import provision
from services.llm.providers.coedit import CoEdITProvider
from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import LinuxGPUProof
from services.llm.providers.python_config import PythonProviderConfig
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager
from services.llm.resource_manager.protocol import EventKind
try:
    from tools.compatibility.coedit_adapter_check import ADAPTER, UUID, _bucket, _candidate_manifest, _runtime_identity
except ImportError:  # Dockerfile.adapter deliberately copies compatibility CLIs flat.
    from coedit_adapter_check import ADAPTER, UUID, _bucket, _candidate_manifest, _runtime_identity

MAX_JSON_BYTES = 256 * 1024

def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError("duplicate JSON key")
        result[key] = value
    return result

def _input(path: Path, configured_fixture: bool) -> tuple[str, str]:
    if configured_fixture:
        if path is not None: raise ValueError("choose --input or --configured-fixture")
        return "Improve the grammar.", "word " * 100
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if not raw or len(raw) > MAX_JSON_BYTES: raise ValueError("input JSON is empty or too large")
    value = json.loads(raw, object_pairs_hook=_no_duplicates)
    if not isinstance(value, dict) or set(value) != {"instruction", "text"} or not all(isinstance(value[k], str) and value[k] for k in value):
        raise ValueError("input JSON must contain exactly nonempty instruction and text")
    return value["instruction"], value["text"]

async def _witness(provider, instruction, text, *, generate=False):
    """Ask the loaded tokenizer for the only accepted input identity."""
    value = await provider.worker.call("benchmark_input", instruction=instruction,
                                       text=text, generate=generate)
    expected = {"count", "max", "fingerprint"} | ({"text"} if generate else set())
    if (not isinstance(value, dict) or set(value) != expected
            or type(value["count"]) is not int or type(value["max"]) is not int
            or not isinstance(value["fingerprint"], str)
            or (generate and (not isinstance(value["text"], str) or not value["text"]))):
        raise ValueError("malformed benchmark witness")
    return value

async def _configured_witness(provider, instruction, text, maximum):
    # Keep the bytes in this harness.  Each candidate is re-tokenized by the
    # loaded worker; no padding and no external corpus can manufacture a witness.
    witness = await _witness(provider, instruction, text, generate=True)
    if witness["max"] != maximum or witness["count"] != maximum:
        raise ValueError("insufficient_max_input")
    candidate = witness["text"]
    payload = json.dumps({"instruction": instruction, "texts": [candidate]},
                         ensure_ascii=True, separators=(",", ":")).encode()
    if hashlib.sha256(payload).hexdigest() != witness["fingerprint"]:
        raise ValueError("benchmark payload identity mismatch")
    return instruction, candidate, {key: witness[key] for key in ("count", "max", "fingerprint")}

def _profile(manifest_hash, model_hash, gpu, batch):
    return CapacityProfile(ModelId.COEDIT, gpu, manifest_hash, model_hash, _runtime_identity(), ADAPTER,
        "unmeasured-coedit-capacity-candidate", batch, batch, batch, 0,
        (SampleMetadata(batch, 0, 0, 0, 0, ()),), bucket_identity=_bucket(batch))

async def _terminal(rm, session, request_id, *, after_sequence=0):
    async def collect():
        async for event in rm.watch_progress(session.session_token, after_sequence):
            if (event.session_token == session.session_token and event.generation == session.generation
                and event.request_id == request_id and event.attempt == "capacity-1"
                and event.kind in (EventKind.RESPONSE_FINISHED, EventKind.FAILURE)):
                if type(event.sequence) is not int or event.sequence <= after_sequence:
                    raise CapacityEvidenceError("invalid progress sequence")
                return event
        raise CapacityEvidenceError("progress stream ended")
    return await asyncio.wait_for(collect(), 120)

def _response(event):
    try: value = json.loads(event.result)
    except (TypeError, ValueError): return False
    return isinstance(value, dict) and set(value) == {"texts"} and isinstance(value["texts"], list) and len(value["texts"]) == 1 and isinstance(value["texts"][0], str) and bool(value["texts"][0])

class _ProofSampler:
    def __init__(self, proof): self.proof = proof
    async def sample(self):
        point = await self.proof.memory()
        if point.gpu_uuid != self.proof.target_uuid:
            raise CapacityEvidenceError("GPU UUID changed during memory observation")
        return MemorySample(point.end_ns, point.total_bytes, point.used_bytes, point.free_bytes,
                            point.start_ns, point.end_ns)

class _RealWaveRunner:
    def __init__(self, rm, session, provider, payload, bucket, witness):
        self.rm, self.session, self.provider, self.payload, self.bucket, self.witness = rm, session, provider, payload, bucket, witness
        self._progress_cursor = 0
    async def __call__(self, p, wave, ids):
        body = json.loads(self.payload.decode("utf-8"))
        current = await _witness(self.provider, body["instruction"], body["texts"][0])
        if (current["count"], current["max"], current["fingerprint"]) != (self.witness["count"], self.witness["max"], self.witness["fingerprint"]):
            raise CapacityEvidenceError("benchmark witness identity changed")
        start = time.monotonic_ns()
        await asyncio.gather(*(self.rm.submit(self.session.session_token, request_id, "capacity-1", self.payload,
            idempotency_key="coedit-capacity-" + request_id, bucket_identity=self.bucket) for request_id in ids))
        events = await asyncio.gather(*(_terminal(self.rm, self.session, request_id,
                                                   after_sequence=self._progress_cursor)
                                        for request_id in ids))
        sequences = tuple(event.sequence for event in events)
        if len(set(sequences)) != len(sequences) or any(type(sequence) is not int or sequence <= self._progress_cursor
                                                         for sequence in sequences):
            raise CapacityEvidenceError("invalid progress sequence")
        self._progress_cursor = max(sequences)
        end = time.monotonic_ns()
        observations = tuple(self.provider.drain_batch_observations())
        drops = self.provider.batch_observation_drops()
        valid_observation = (drops == 0 and len(observations) == 1 and isinstance(observations[0], dict)
            and set(observations[0]) == {"batch_size", "execution_started", "execution_ended", "cuda_synchronized", "allocator", "decoder_steps", "max_output_tokens", "request_ids"}
            and observations[0]["batch_size"] == p and observations[0]["request_ids"] == ids
            and observations[0]["cuda_synchronized"] is True
            and type(observations[0].get("max_output_tokens")) is int
            and isinstance(observations[0].get("decoder_steps"), tuple)
            and len(observations[0]["decoder_steps"]) == p
            and all(type(value) is int and 0 <= value <= observations[0]["max_output_tokens"] for value in observations[0]["decoder_steps"])
            and observations[0]["max_output_tokens"] == self.provider.config.max_output_tokens
            and isinstance(observations[0]["allocator"], AllocatorObservation)
            and observations[0]["allocator"].valid())
        observation = observations[0] if valid_observation else {}
        return Wave(p, wave, ids, max(1, (end - start) // 1_000_000), all(_response(event) for event in events),
            observation.get("batch_size", 0), observation.get("execution_started"), observation.get("execution_ended"),
            observation.get("cuda_synchronized", False), allocator=observation.get("allocator"), failed=not valid_observation,
            failure_kind=None if valid_observation else "native_batch_correlation",
            observation_count=len(observations),
            observed_native_batch_sizes=tuple(value.get("batch_size") for value in observations
                                              if isinstance(value, dict) and type(value.get("batch_size")) is int),
            native_request_correlation=any(isinstance(value, dict) and value.get("request_ids") == ids
                                           for value in observations), observation_drops=drops,
            decoder_steps=observation.get("decoder_steps", ()), max_output_tokens=observation.get("max_output_tokens"))

def _raw(result):
    def wave(w):
        allocator = None if w.allocator is None else {key: getattr(w.allocator, key) for key in ("baseline_allocated", "baseline_reserved", "peak_allocated", "peak_reserved", "final_allocated", "final_reserved")}
        samples = w.samples
        valid = [s for s in samples if s.valid()]
        overlap = [s for s in valid if sample_overlaps_execution(s, w.execution_started, w.execution_ended)]
        witness = overlap[0] if overlap else None
        def point(s): return None if s is None else {"start_ns": s.start_ns, "end_ns": s.end_ns, "timestamp_ns": s.timestamp_ns, "total_bytes": s.total_bytes, "used_bytes": s.used_bytes, "free_bytes": s.free_bytes}
        return {"p": w.concurrency, "wave": w.wave, "phase": w.phase, "request_count": len(w.request_ids), "elapsed_ms": w.elapsed_ms, "native_batch_size": w.native_batch_size, "outputs_valid": w.outputs_valid, "failed": w.failed, "failure_kind": w.failure_kind, "execution_started_ns": w.execution_started, "execution_ended_ns": w.execution_ended, "allocator": allocator, "decoder_workload": {"steps": list(w.decoder_steps), "max_output_tokens": w.max_output_tokens}, "native_observation": {"count": w.observation_count, "batch_sizes": list(w.observed_native_batch_sizes), "request_correlation": w.native_request_correlation, "drops": w.observation_drops}, "sample_summary": {"count": len(samples), "valid_count": len(valid), "overlap_count": len(overlap), "first_start_ns": samples[0].start_ns if samples else None, "last_end_ns": samples[-1].end_ns if samples else None, "total_bytes": samples[0].total_bytes if samples else None, "min_free_bytes": min((s.free_bytes for s in samples), default=None), "max_used_bytes": max((s.used_bytes for s in samples), default=None), "first_in_window": point(witness)}}
    baseline = getattr(result, "baseline", ())
    warmups = getattr(result, "warmups", ())
    points = getattr(result, "points", ())
    output = {"status": getattr(result, "status", None), "reason": getattr(result, "reason", None),
              "max_output_verified": bool(getattr(result, "max_output_verified", False)),
             "failure_phase": getattr(result, "failure_phase", None), "failure_kind": getattr(result, "failure_kind", None), "baseline": [wave(w) for w in baseline],
             "warmups": [wave(w) for w in warmups], "measured": [wave(w) for w in points]}
    if hasattr(result, "observed_safe_through"):
        output.update({"observed_safe_through": result.observed_safe_through,
                       "candidate_ceiling": result.candidate_ceiling,
                       "stop_p": result.stop_p, "memory_safe_n": None,
                       "profile_eligible": False})
    return output

def _write_raw(path, result):
    if path.exists(): raise FileExistsError("refusing to overwrite raw output")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            encoded = json.dumps(_raw(result), sort_keys=True, separators=(",", ":")).encode()
            if len(encoded) > MAX_JSON_BYTES:
                raise ValueError("raw capacity artifact exceeds 256 KiB bound")
            stream.write(encoded.decode())
            stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise
    finally:
        try: temporary.unlink()
        except FileNotFoundError: pass

def _safe_reason(exc):
    text = str(exc)
    safe = {"insufficient_max_input", "benchmark_tokenization_failed", "benchmark_input_over_bound",
            "worker_operation_failed", "model worker is dead", "worker stderr transport failed",
            "malformed worker response", "model worker transport failed"}
    if text in safe: return text
    if text.startswith("worker operation failed: ") and text.rsplit(": ", 1)[-1] in safe:
        return text.rsplit(": ", 1)[-1]
    return type(exc).__name__

async def run(args):
    minimum = 2 if getattr(args, "discover_memory", False) else 1
    maximum = DISCOVERY_MAX_PARALLELISM if getattr(args, "discover_memory", False) else THROUGHPUT_MAX_PARALLELISM
    if type(args.native_batch_size) is not int or not minimum <= args.native_batch_size <= maximum: raise ValueError(f"native batch must be {minimum}..{maximum}")
    if not args.host_pid_namespace or not UUID.fullmatch(args.target_gpu_uuid): raise ValueError("host PID attestation and exact GPU UUID are required")
    instruction, text = _input(args.input, args.configured_fixture)
    source = args.models_root / "CoEdIT"; selected, model_hash = await asyncio.to_thread(_candidate_manifest, args.manifest, source)
    root, provider, session, proof, cleaned = Path(tempfile.mkdtemp(prefix="coedit-capacity-")), None, None, None, False
    rm = ResourceManager(cleanup_timeout=60, stop_timeout=60, load_timeout=240)
    try:
        document = await asyncio.to_thread(provision, {"CoEdIT": source}, root / "artifacts")
        proof = await asyncio.to_thread(LinuxGPUProof.capture, args.target_gpu_uuid, os.getpid(), Path("/proc"), None, host_pid_namespace=True)
        config = PythonProviderConfig(root / "artifacts", document["manifest_sha256"], model_hash, args.target_gpu_uuid, _runtime_identity(), ADAPTER, bucket_identity=_bucket(args.native_batch_size), max_native_batch_size=args.native_batch_size, max_input_tokens=128, max_output_tokens=64, gpu_proof=GPUProof(proof.identity, proof.cleanup, proof.residency, expected_supervisor=proof.supervisor_identity))
        provider = CoEdITProvider(config); session = await rm.start_session("coedit-capacity-candidate", ModelId.COEDIT, _profile(document["manifest_sha256"], model_hash, args.target_gpu_uuid, args.native_batch_size), provider, idempotency_key="coedit-capacity-start")
        if args.configured_fixture:
            instruction, text, witness = await _configured_witness(provider, instruction, text, 128)
        else:
            witness = await _witness(provider, instruction, text)
            if witness["count"] != 128: raise ValueError("insufficient_max_input")
        payload = json.dumps({"instruction": instruction, "texts": [text]}, ensure_ascii=True, separators=(",", ":")).encode()
        import hashlib
        digest = hashlib.sha256(payload).hexdigest()
        if digest != witness["fingerprint"]: raise ValueError("benchmark payload identity mismatch")
        runner = _RealWaveRunner(rm, session, provider, payload, _bucket(args.native_batch_size), witness)
        if getattr(args, "discover_memory", False):
            result = await discover_memory(runner, _ProofSampler(proof), max_parallelism=args.native_batch_size,
                                           expected_max_output_tokens=provider.config.max_output_tokens)
        else:
            result = await measure_capacity(runner, _ProofSampler(proof), max_parallelism=args.native_batch_size,
                                            expected_max_output_tokens=provider.config.max_output_tokens)
        args._result = result
        resident = await proof.residency()
        if len(resident.runners) != 1: raise CapacityEvidenceError("owned GPU residency was not proved")
        await rm.stop_session(session.session_token, reason="candidate-complete", idempotency_key="coedit-capacity-stop"); await provider.unload()
        provider_clean = await provider.verify_cleanup()
        proof_clean = await proof.cleanup()
        cleaned = provider_clean and proof_clean
        if not cleaned: raise CapacityEvidenceError("owned provider/GPU cleanup was not proved")
        if args.raw_output:
            _write_raw(args.raw_output, result)
        if getattr(args, "discover_memory", False):
            value = {"status": result.status, "reason": result.reason, "failure_phase": result.failure_phase,
                     "failure_kind": result.failure_kind, "profile_eligible": False,
                     "observed_safe_through": result.observed_safe_through,
                     "candidate_ceiling": result.candidate_ceiling, "stop_p": result.stop_p,
                     "memory_safe_n": None, "max_output_verified": result.max_output_verified,
                     "native_batch_size": args.native_batch_size, "payload_sha256": digest,
                     "max_input_verified": witness["count"] == 128,
                     "configured_fixture": args.configured_fixture, "cleanup": True}
        else:
            value = {"status": result.status, "reason": result.reason, "failure_phase": result.failure_phase,
                     "failure_kind": result.failure_kind, "profile_eligible": False, "candidate_n": result.candidate_n,
                     "optimal_parallelism": result.optimal_parallelism, "native_batch_size": args.native_batch_size,
                     "max_output_verified": result.max_output_verified,
                     "payload_sha256": digest, "max_input_verified": witness["count"] == 128,
                     "configured_fixture": args.configured_fixture, "cleanup": True}
        return value
    finally:
        if session is not None and not cleaned:
            try: await rm.stop_session(session.session_token, reason="finally", idempotency_key="coedit-capacity-finally")
            except BaseException: pass
        if provider is not None and not cleaned:
            try:
                await provider.unload()
                provider_clean = await provider.verify_cleanup()
                proof_clean = proof is not None and await proof.cleanup()
                cleaned = provider_clean and proof_clean
            except BaseException: cleaned = False
        args._cleanup_proved = cleaned
        if cleaned: shutil.rmtree(root, ignore_errors=True)
        else: print("coedit capacity candidate retained temporary storage", file=sys.stderr)

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--models-root", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--target-gpu-uuid", required=True); p.add_argument("--host-pid-namespace", action="store_true"); p.add_argument("--native-batch-size", type=int, required=True); p.add_argument("--input", type=Path); p.add_argument("--configured-fixture", action="store_true"); p.add_argument("--raw-output", type=Path); p.add_argument("--discover-memory", action="store_true")
    args = p.parse_args()
    if bool(args.input) == args.configured_fixture: p.error("provide exactly one of --input or --configured-fixture")
    minimum = 2 if args.discover_memory else 1
    maximum = DISCOVERY_MAX_PARALLELISM if args.discover_memory else THROUGHPUT_MAX_PARALLELISM
    if type(args.native_batch_size) is not int or not minimum <= args.native_batch_size <= maximum:
        p.error(f"--native-batch-size must be between {minimum} and {maximum}")
    try: value = asyncio.run(asyncio.wait_for(run(args), 900))
    except (ValueError, OSError, CapacityEvidenceError, asyncio.TimeoutError, RuntimeError) as exc:
        # Only a type category is exposed: provider/RM exceptions may include
        # request payloads or unbounded remote diagnostics.
        result = getattr(args, "_result", None)
        if args.raw_output and result is not None and not args.raw_output.exists():
            try: _write_raw(args.raw_output, result)
            # Persistence is diagnostic only: a bounded-artifact failure must
            # not conceal the already-classified terminal candidate result.
            except (OSError, ValueError, TypeError): pass
        reason = _safe_reason(exc)
        print(json.dumps({"status":"incomplete", "reason":reason,
                           "failure_phase":getattr(result, "failure_phase", None),
                           "failure_kind":getattr(result, "failure_kind", None),
                           "profile_eligible":False,
                          "cleanup":getattr(args, "_cleanup_proved", False)}, separators=(",", ":")))
        return 2
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    success_status = "complete" if args.discover_memory else "candidate"
    return 0 if value["status"] == success_status else 2
if __name__ == "__main__": raise SystemExit(main())

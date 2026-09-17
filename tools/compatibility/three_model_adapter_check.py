"""Bounded, offline ResourceManager switching check for installed adapters.

The harness process is the single GPU-proof supervisor.  Model runtimes are
only imported by the providers' owned children.
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
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.llm.provisioning.artifacts import SPECS, manifest, verify_manifest
from services.llm.provisioning.volume import provision
from services.llm.bootstrap.config import BootstrapConfig, ModelConfig
from services.llm.bootstrap.supervisor import OwnedOllama
from services.llm.providers.config import GPUProof, SmolLMProviderConfig
from services.llm.providers.coedit import CoEdITProvider
from services.llm.providers.gector import GECToRProvider
from services.llm.providers.gector_config import GECToRProviderConfig
from services.llm.providers.gpu import (LinuxGPUProof, RESIDENCY_SETTLE_INTERVAL_SECONDS,
                                         RESIDENCY_SETTLE_TIMEOUT_SECONDS, _ResidencyPending,
                                         settle_residency)
from services.llm.providers.python_config import PythonProviderConfig
from services.llm.providers.smollm import SmolLMProvider
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager, ResourceManagerError
from services.llm.resource_manager.protocol import EventKind, Failure

# The adapter image copies this helper as /opt/llm/adapter_check.py, whereas
# source-tree unit tests import it as a package member.
try:
    from tools.compatibility import adapter_check
except ModuleNotFoundError:
    import adapter_check  # type: ignore[no-redef]

UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")
GGUF = SPECS["SmolLM"][0]
BUCKETS = {
    "CoEdIT": "coedit:p1:input128:output64:float16:beams1:nosample",
    "GECToR": "gector:p1:tokens128:keep0:min0:iterations1:batch1:float32",
}
MODEL_IDS = {"SmolLM": ModelId.SMOLLM, "CoEdIT": ModelId.COEDIT, "GECToR": ModelId.GECTOR}
SEQUENCE = ("SmolLM", "CoEdIT", "GECToR", "SmolLM")
_CLEANUP_REASONS = frozenset({
    "not_checked", "proof_not_captured", "supervisor_identity_unavailable",
    "owned_runner_present", "gpu_cleanup_probe_failed",
    "supervisor_identity_changed_after_probe", "clean",
})
CLEANUP_SETTLE_INTERVAL_SECONDS = 0.2
CLEANUP_SETTLE_TIMEOUT_SECONDS = 5.0


def _runtime_identity(model: str) -> str:
    names = ("torch", "transformers", "tokenizers", "safetensors")
    if model == "GECToR":
        names = ("gector",) + names
    values = {name: importlib.metadata.version(name) for name in names}
    return "candidate:" + ";".join(f"{name}={values[name]}" for name in sorted(values))


def _model_hash(document: dict, model: str) -> str:
    name = GGUF if model == "SmolLM" else "model.safetensors"
    return next(item["sha256"] for item in document["models"][model]["files"] if item["path"] == name)


def _selected(entry: dict) -> tuple[str, tuple[tuple[str, int, str], ...]]:
    return entry["model_id"], tuple((item["path"], item["size"], item["sha256"])
                                    for item in entry["files"])


def _candidate_manifest(path: Path, models_root: Path) -> dict:
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict) or set(document) != {"schema", "models", "manifest_sha256"}:
        raise RuntimeError("candidate manifest is invalid")
    if manifest(document["models"]) != document or set(document["models"]) != set(SPECS):
        raise RuntimeError("candidate manifest is not the exact combined selection")
    for model, required in SPECS.items():
        entry = document["models"].get(model)
        if not isinstance(entry, dict) or [item.get("path") for item in entry.get("files", ())] != list(required):
            raise RuntimeError(f"candidate {model} file set is not exact")
    verify_manifest(document, {model: models_root / model for model in SPECS})
    return document


def _profile(model: str, provisioned: dict, gpu: str, runtime: str) -> CapacityProfile:
    return CapacityProfile(MODEL_IDS[model], gpu, provisioned["manifest_sha256"],
        _model_hash(provisioned, model), runtime, f"candidate-{model.lower()}-provider",
        f"unmeasured-{model.lower()}-adapter-check", 1, 1, 1, 0,
        (SampleMetadata(1, 0, 0, 0, 0, ()),),
        context_size=512 if model == "SmolLM" else None,
        bucket_identity=None if model == "SmolLM" else BUCKETS[model])


def _request(model: str) -> bytes:
    if model == "SmolLM":
        return b"Say hello."
    if model == "CoEdIT":
        return b'{"instruction":"Improve the grammar.","texts":["Short text."]}'
    return b'{"texts":["Short text."],"keep_confidence":0.0,"min_error_prob":0.0,"n_iteration":1,"batch_size":1}'


def _response_ok(model: str, event) -> bool:
    if not event.result:
        return False
    if model == "SmolLM":
        return bool(event.result.strip())
    try:
        value = json.loads(event.result)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (isinstance(value, dict) and set(value) == {"texts"} and
            isinstance(value["texts"], list) and len(value["texts"]) == 1 and
            isinstance(value["texts"][0], str) and bool(value["texts"][0].strip()))


async def _terminal_events(rm, session, request_id: str, attempt: str) -> list:
    found = []
    async for event in rm.watch_progress(session.session_token):
        if (event.session_token, event.generation, event.request_id, event.attempt) != (
                session.session_token, session.generation, request_id, attempt):
            continue
        found.append(event)
        if event.kind in (EventKind.RESPONSE_FINISHED, EventKind.FAILURE, EventKind.CANCELLED):
            return found
    raise RuntimeError("resource manager progress stream ended")


class _GateProvider:
    """Delegating provider whose load gate proves prior GPU cleanup."""
    def __init__(self, provider, proof, *, cleanup_interval=CLEANUP_SETTLE_INTERVAL_SECONDS,
                 cleanup_timeout=CLEANUP_SETTLE_TIMEOUT_SECONDS, sleep=asyncio.sleep,
                 monotonic=time.monotonic):
        self.provider, self.proof = provider, proof
        self.cleanup_interval = cleanup_interval
        self.cleanup_timeout = cleanup_timeout
        self._sleep = sleep
        self._monotonic = monotonic
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.gate = False

    def __getattr__(self, name):
        return getattr(self.provider, name)

    async def _settle_cleanup(self) -> str | None:
        """Observe owned cleanup until it settles, without widening authority."""
        deadline = self._monotonic() + self.cleanup_timeout
        while True:
            try:
                clean = await self.proof.cleanup()
            except asyncio.CancelledError:
                raise
            except Exception:
                return "gpu_cleanup_probe_failed"
            if clean:
                return None
            reason = _cleanup_reason(self.proof)
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return reason
            await self._sleep(min(self.cleanup_interval, remaining))

    async def load(self, profile):
        reason = await self._settle_cleanup()
        if reason is not None:
            raise RuntimeError(f"shared GPU cleanup proof is false before load; reason={reason}")
        await self.provider.load(profile)

    async def execute(self, request_id, payload):
        if self.gate:
            self.entered.set()
            await self.release.wait()
        return await self.provider.execute(request_id, payload)


def _provider(model: str, root: Path, provisioned: dict, gpu: str, proof, runtime: str,
               port: int, daemon_store: Path):
    if type(proof) is not GPUProof:
        proof = GPUProof(proof.identity, proof.cleanup, proof.residency,
                          expected_supervisor=proof.supervisor_identity,
                           residency_for_runner=getattr(proof, "residency_for_runner", None),
                           memory=getattr(proof, "memory", None))
    common = (root, provisioned["manifest_sha256"], _model_hash(provisioned, model), gpu, runtime,
              f"candidate-{model.lower()}-provider")
    if model == "SmolLM":
        return SmolLMProvider(SmolLMProviderConfig(*common, allowed_context_sizes=(512,), parallelism=1,
            request_timeout_seconds=60, ollama_port=port, gpu_proof=proof, ollama_binary="/usr/bin/ollama",
            ollama_home=adapter_check._runtime_home(daemon_store)))
    if model == "CoEdIT":
        return CoEdITProvider(PythonProviderConfig(*common, bucket_identity=BUCKETS[model], gpu_proof=proof))
    return GECToRProvider(GECToRProviderConfig(*common, bucket_identity=BUCKETS[model], gpu_proof=proof))


def _bootstrap_config(root: Path, provisioned: dict, gpu: str, runtimes: dict[str, str],
                      port: int, daemon_store: Path) -> BootstrapConfig:
    return BootstrapConfig(
        gpu, root / "artifacts", provisioned["manifest_sha256"], root / "profiles.sqlite",
        Path("/usr/bin/ollama"), adapter_check._runtime_home(daemon_store), port,
        {model: ModelConfig(runtimes[model], f"candidate-{model.lower()}-provider")
         for model in MODEL_IDS},
    )


async def _reject_stale(rm, token: str, model: str, number: int) -> None:
    for operation in ("submit", "cancel"):
        try:
            if operation == "submit":
                await rm.submit(token, f"stale-{number}", "attempt-1", _request(model),
                    idempotency_key=f"three-model-stale-{number}",
                    context_size=512 if model == "SmolLM" else None,
                    bucket_identity=None if model == "SmolLM" else BUCKETS[model])
            else:
                await rm.cancel_request(token, f"stale-{number}",
                                        idempotency_key=f"three-model-stale-cancel-{number}")
        except ResourceManagerError as exc:
            if exc.failure.code != "scheduler_superseded":
                raise
        else:
            raise RuntimeError(f"stale session {operation} was accepted")


async def _providers_clean(providers: list[_GateProvider], timeout: float) -> bool:
    """Require every begun installed provider to release its own resources."""
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(provider.verify_cleanup() for provider in providers),
                           return_exceptions=True), timeout)
    except TimeoutError:
        return False
    return bool(results) and all(result is True for result in results)


def _safe_message(message: object) -> str:
    """Keep diagnostics bounded and redact likely payload/process-table content."""
    # Do not invoke arbitrary exception/object __str__: diagnostics are on a
    # failure path and such a method can be hostile, blocking, or disclose a
    # secret.  Failure.message is normally a plain string; anything else has
    # no safe text representation at this boundary.
    if type(message) is not str:
        return "<unavailable>"
    value = message[:160].replace("\r", " ").replace("\n", " ")
    lowered = value.lower()
    if ("{" in value or "[" in value or any(marker in lowered for marker in
            ("prompt", "completion", "process", "token", "bearer", "authorization"))):
        return "<redacted>"
    value = re.sub(r"[^A-Za-z0-9 .,;:_=/()-]", "?", value)
    return value[:160]


def _cleanup_reason(proof) -> str:
    reason = getattr(proof, "cleanup_reason", "cleanup_not_proved")
    return reason if reason in _CLEANUP_REASONS else "cleanup_not_proved"


def _remove_root(root: Path, errors: list[str]) -> None:
    """Best-effort bounded storage cleanup; callers retain the root on failure."""
    try:
        shutil.rmtree(root)
    except OSError:
        errors.append("temporary_storage_cleanup_failed")


def _transition_failure(exc: BaseException, transition: int, model: str,
                        stage: str) -> BaseException:
    """Add bounded lifecycle coordinates without exposing payload or runtime state."""
    # Cancellation and other BaseException control flow must reach the caller
    # unchanged.  Converting it to a diagnostic failure would make cancelled
    # lifecycle work look like an ordinary failed transition.
    if not isinstance(exc, Exception):
        return exc
    detail = f"transition={transition} model={model} stage={stage}"
    if isinstance(exc, ResourceManagerError):
        failure = exc.failure
        return ResourceManagerError(Failure(failure.code, f"{detail}: {_safe_message(failure.message)}", failure.retryable))
    return RuntimeError(f"{detail}: exception={type(exc).__name__}; message={_safe_message(exc)}")


def _memory_record(observation, sequence_index: int, label: str) -> dict:
    return {"label": label, "sequence_index": sequence_index,
            "start_ns": observation.start_ns, "end_ns": observation.end_ns,
            "total_bytes": observation.total_bytes, "used_bytes": observation.used_bytes,
            "free_bytes": observation.free_bytes}


async def _observe_memory(proof, observations: list[dict], sequence_index: int,
                          label: str, expected_gpu_uuid: str,
                          expected: dict | None = None) -> dict:
    observation = await proof.memory()
    if observation.gpu_uuid != expected_gpu_uuid:
        raise RuntimeError("GPU memory UUID changed or mismatched")
    identity = {"gpu_uuid": observation.gpu_uuid, "total_bytes": observation.total_bytes}
    if expected is not None and identity != expected:
        raise RuntimeError("GPU memory identity or total capacity changed")
    observations.append(_memory_record(observation, sequence_index, label))
    return identity


async def _post_load_residency(proof, model=None, provider=None, *, sleep=asyncio.sleep,
                               monotonic=time.monotonic):
    """Require stable, positive ownership after a model load."""
    try:
        residency = await settle_residency(
            proof.residency,
            timeout=RESIDENCY_SETTLE_TIMEOUT_SECONDS,
            interval=RESIDENCY_SETTLE_INTERVAL_SECONDS,
            sleep=sleep,
            monotonic=monotonic,
        )
    except _ResidencyPending:
        accepted = getattr(provider, "accepted_model_specific_residency", None) if provider is not None else None
        if model not in {"SmolLM", "CoEdIT", "GECToR"} or not callable(accepted) or not accepted():
            raise
        return None
    if len(residency.runners) != 1:
        raise RuntimeError("did not prove exactly one resident runner")
    return residency


async def run(args: argparse.Namespace) -> dict:
    if not getattr(args, "host_pid_namespace", False):
        raise ValueError("--host-pid-namespace operator attestation is required")
    if not isinstance(args.target_gpu_uuid, str) or not UUID.fullmatch(args.target_gpu_uuid):
        raise ValueError("target GPU UUID is invalid")
    source = await asyncio.to_thread(_candidate_manifest, args.manifest, args.models_root)
    root = Path(tempfile.mkdtemp(prefix="three-model-adapter-check-"))
    retain, daemon, session, provider, proof = False, None, None, None, None
    daemon_started = False
    cleanup_ok = False
    begun: list[_GateProvider] = []
    try:
        # This mandatory capture occurs before starting any daemon or worker.
        proof = await asyncio.to_thread(LinuxGPUProof.capture, args.target_gpu_uuid, os.getpid(),
                                        Path("/proc"), None, host_pid_namespace=True)
        memory_observations: list[dict] = []
        memory_identity = await _observe_memory(proof, memory_observations, 0, "baseline",
                                                args.target_gpu_uuid)
        def daemon_ownership_snapshot():
            # The proof is intentionally assembled before daemon construction so
            # the daemon and every provider receive one object.  Do not let the
            # callback become an authority before that daemon has started.
            if daemon is None or not daemon_started:
                raise RuntimeError("Ollama ownership snapshot is unavailable before daemon startup")
            return daemon.ownership_snapshot()

        typed_proof = GPUProof(proof.identity, proof.cleanup, proof.residency,
                                expected_supervisor=proof.supervisor_identity,
                                residency_for_runner=proof.residency_for_runner,
                                memory=proof.memory,
                                ollama_ownership=daemon_ownership_snapshot)
        provisioned = await asyncio.to_thread(provision, {m: args.models_root / m for m in SPECS}, root / "artifacts")
        if any(_selected(source["models"][m]) != _selected(provisioned["models"][m]) for m in SPECS):
            raise RuntimeError("provisioned selected files differ from source selection")
        daemon_store = root / "ollama-models"
        # The daemon receives the exact same typed proof as every provider.  It
        # is created only after capture, so foreign GPU processes remain outside
        # this ownership fence.
        runtimes = {model: (_runtime_identity(model) if model != "SmolLM" else "ollama:0.11.6")
                    for model in MODEL_IDS}
        config = _bootstrap_config(root, provisioned, args.target_gpu_uuid, runtimes,
                                   args.port, daemon_store)
        daemon = OwnedOllama(config, typed_proof)
        ollama_version = await daemon.start()
        daemon_started = True
        if ollama_version != "0.11.6":
            raise RuntimeError("unexpected pinned Ollama version")
        rm = ResourceManager(cleanup_timeout=60, stop_timeout=60, load_timeout=240)
        checks, transitions, cancel_fenced = [], 0, False
        for index, model in enumerate(SEQUENCE):
            stage = "construct"
            try:
                runtime = runtimes[model]
                wrapped = _GateProvider(_provider(model, root / "artifacts", provisioned, args.target_gpu_uuid,
                                                   typed_proof, runtime, args.port, daemon_store), proof)
                begun.append(wrapped)
                previous = session.session_token if session else None
                stage = "start_session"
                session = await rm.start_session("three-model-adapter-check", MODEL_IDS[model],
                    _profile(model, provisioned, args.target_gpu_uuid, runtime), wrapped,
                    idempotency_key=f"three-model-start-{index}")
                if previous:
                    transitions += 1
                    stage = "stale_fence"
                    await _reject_stale(rm, previous, model, index)
                request_id, attempt = f"{model.lower()}-{index}", "attempt-1"
                stage = "submit"
                await rm.submit(session.session_token, request_id, attempt, _request(model),
                    idempotency_key=f"three-model-request-{index}",
                    context_size=512 if model == "SmolLM" else None,
                    bucket_identity=None if model == "SmolLM" else BUCKETS[model])
                stage = "complete"
                events = await _terminal_events(rm, session, request_id, attempt)
                if not any(event.kind is EventKind.RESPONSE_FINISHED and _response_ok(model, event) for event in events):
                    raise RuntimeError("response was not aligned and nonempty")
                # Cancellation is a separate CoEdIT execution-entry fence; it does
                # not substitute for that model's genuine successful response.
                if model == "CoEdIT":
                    wrapped.gate = True
                    cancelled_id = "coedit-cancelled"
                    await rm.submit(session.session_token, cancelled_id, attempt, _request(model),
                        idempotency_key="three-model-cancelled-request", bucket_identity=BUCKETS[model])
                    await asyncio.wait_for(wrapped.entered.wait(), 10)
                    if not await rm.cancel_request(session.session_token, cancelled_id,
                                                    idempotency_key="three-model-cancel"):
                        raise RuntimeError("CoEdIT cancellation was not accepted")
                    wrapped.release.set()
                    cancelled = await _terminal_events(rm, session, cancelled_id, attempt)
                    if any(event.kind is EventKind.RESPONSE_FINISHED and event.result for event in cancelled):
                        raise RuntimeError("cancelled CoEdIT execution published a result")
                    cancel_fenced = any(event.kind is EventKind.RESPONSE_FINISHED for event in cancelled)
                    if not cancel_fenced:
                        raise RuntimeError("CoEdIT delegate completion was not observed")
                stage = "residency"
                await _post_load_residency(proof, model, wrapped.provider)
                stage = "memory"
                memory_identity = await _observe_memory(proof, memory_observations, index + 1,
                                                        model, args.target_gpu_uuid, memory_identity)
                checks.append(model)
                provider = wrapped
            except BaseException as exc:
                failure = _transition_failure(exc, index, model, stage)
                if failure is exc:
                    raise
                raise failure from exc
        old = session.session_token
        await rm.stop_session(old, reason="completed", idempotency_key="three-model-stop")
        if not await _providers_clean(begun, rm.cleanup_timeout):
            raise RuntimeError("an installed provider did not prove cleanup")
        await _reject_stale(rm, old, "SmolLM", len(SEQUENCE))
        await daemon.close()
        daemon = None
        daemon_started = False
        cleanup_ok = await proof.cleanup()
        if not cleanup_ok:
            raise RuntimeError("final shared GPU cleanup proof is false")
        memory_identity = await _observe_memory(proof, memory_observations, 5,
                                                "final_cleanup", args.target_gpu_uuid, memory_identity)
        return {"status": "passed-candidate", "model_sequence": list(SEQUENCE), "count": len(checks),
                "switch_count": transitions, "profile": "unmeasured", "cleanup": cleanup_ok,
                "stale_rejected": True, "cancel_fenced": cancel_fenced,
                "memory_observations": memory_observations,
                "gpu_uuid": memory_identity["gpu_uuid"],
                "manifest_sha256": provisioned["manifest_sha256"],
                "model_sha256": {m: _model_hash(provisioned, m) for m in SPECS}, "runtime": runtimes}
    except BaseException as exc:
        # A failed first load can have started an owned worker even without a
        # session. Retention is decided only after every provider/daemon/GPU
        # cleanup proof below; GPU absence alone is not sufficient.
        retain = bool(begun)
        primary_error = exc
        raise
    finally:
        errors = []
        if session is not None and not cleanup_ok:
            try:
                await rm.stop_session(session.session_token, reason="finally", idempotency_key="three-model-finally")
                cleanup_ok = proof is not None and await proof.cleanup()
                retain = not cleanup_ok
            except ResourceManagerError as exc:
                if exc.failure.code != "scheduler_superseded":
                    errors.append("session_cleanup_" + exc.failure.code)
            except BaseException as exc:
                errors.append("session_cleanup_" + type(exc).__name__)
        providers_clean = not begun
        if begun:
            try:
                providers_clean = await _providers_clean(begun, rm.cleanup_timeout)
                if not providers_clean:
                    errors.append("provider_cleanup_not_proved")
            except BaseException as exc:
                errors.append("provider_cleanup_" + type(exc).__name__)
        daemon_gone = daemon is None
        if daemon is not None:
            try:
                daemon_started = False
                await daemon.close()
                daemon_gone = True
            except BaseException as exc:
                errors.append("daemon_cleanup_" + type(exc).__name__)
        gpu_clean = proof is None and not begun and daemon is None
        if proof is not None:
            try:
                gpu_clean = await proof.cleanup()
                if not gpu_clean: errors.append("proof_cleanup_not_proved")
            except BaseException as exc:
                errors.append("proof_cleanup_" + type(exc).__name__)
        cleanup_ok = providers_clean and daemon_gone and gpu_clean
        retain = not cleanup_ok
        if not retain and not errors:
            _remove_root(root, errors)
        if retain or errors:
            print(f"three-model adapter check retained temporary storage: {root}", file=sys.stderr)
        if errors and "primary_error" not in locals():
            raise RuntimeError("cleanup failed: " + ",".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-gpu-uuid", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host-pid-namespace", action="store_true")
    print(json.dumps(asyncio.run(asyncio.wait_for(run(parser.parse_args()), 900)), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

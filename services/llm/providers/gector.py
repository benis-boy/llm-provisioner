"""Offline GECToR provider using the common owned Python worker transport."""
from __future__ import annotations
import asyncio, inspect, json, math, time
from services.llm.provisioning.volume import verify_current
from services.llm.provisioning.artifacts import SPECS
from services.llm.resource_manager.contracts import CapacityProfile
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.protocol import ProviderResponse
from .gpu import (GPUMemoryObservation, ProcessIdentity, ResidencyEvidence,
                  _ResidencyPending, settle_residency)
from .python_process import PythonWorker
from .gector_config import GECToRProviderConfig
try:
    from tools.compatibility.debug_trace import lifecycle
except ImportError:
    def lifecycle(*args, **kwargs): return lambda function: function


class GECToRProvider:
    def __init__(self, config: GECToRProviderConfig):
        self.config, self.worker, self.profile = config, None, None
        self._ready = False
        self._cleanup = True
        self._preload_memory = None
        self._model_specific_ready = False
        self._residency_sleep = asyncio.sleep
        self._residency_clock = time.monotonic

    def _artifact(self):
        evidence = verify_current(self.config.artifact_root)
        if evidence.get("manifestSha256") != self.config.manifest_sha256:
            raise ValueError("current artifact manifest mismatch")
        root = self.config.artifact_root / self.config.manifest_sha256
        manifest = json.loads((root / "manifest.json").read_bytes())
        entry = manifest.get("models", {}).get("GECToR")
        files = entry.get("files", []) if isinstance(entry, dict) else []
        if [x.get("path") for x in files if isinstance(x, dict)] != list(SPECS["GECToR"]):
            raise ValueError("GECToR manifest file set mismatch")
        model = next((x for x in files if x.get("path") == "model.safetensors"), None)
        vocab = next((x for x in files if x.get("path") == "verb-form-vocab.txt"), None)
        if not model or model.get("sha256") != self.config.model_sha256 or not vocab:
            raise ValueError("GECToR artifact identity or vocabulary mismatch")
        return root / "models" / "GECToR"

    @lifecycle("provider.gector")
    async def validate(self, profile: CapacityProfile):
        if (profile.model_id is not ModelId.GECTOR or profile.bucket_identity != self.config.bucket_identity
                or profile.context_size is not None or profile.optimal_parallelism != 1
                or profile.buffer_capacity != 1 or not profile.matches(ModelId.GECTOR, self.config.gpu_uuid,
                self.config.manifest_sha256, self.config.model_sha256, self.config.runtime_identity,
                self.config.adapter_identity)):
            raise ValueError("unsupported GECToR profile")
        self.profile = profile

    @lifecycle("provider.gector")
    async def load(self, profile):
        self._ready = False
        self._model_specific_ready = False
        self._preload_memory = None
        await self.validate(profile)
        root = await asyncio.to_thread(self._artifact)
        if self.worker is not None:
            raise RuntimeError("previous GECToR worker has not been cleaned up")
        self._cleanup = False
        if self.config.gpu_proof is not None and self.config.gpu_proof.memory is not None:
            self._preload_memory = await self._memory()
        self.worker = PythonWorker(root, {"dtype": self.config.dtype, "gpu_uuid": self.config.gpu_uuid,
            "max_subword_tokens": self.config.max_subword_tokens, "keep_confidence": self.config.keep_confidence,
            "min_error_prob": self.config.min_error_prob, "max_iterations": self.config.max_iterations,
            "batch_size": 1}, command=[__import__('sys').executable, "-m", "services.llm.providers.gector_worker"],
            timeout=self.config.request_timeout_seconds, frame_limit=self.config.rpc_frame_limit, gpu_proof=self.config.gpu_proof)
        try:
            await self.worker.start(); await self.worker.call("load")
            await self.worker.call("cuda_ready")
        except BaseException:
            await self.worker.close()
            self.worker = None
            self._cleanup = True
            raise

    @lifecycle("provider.gector")
    async def ready(self):
        self._ready = False
        self._model_specific_ready = False
        if self.worker is None: raise RuntimeError("provider is not loaded")
        value = await self.worker.call("gpu_identity")
        if not isinstance(value, dict) or set(value) != {"gpu_uuid", "runner_pid", "runner_start_time", "cuda_nvml_agree"} or value.get("gpu_uuid") != self.config.gpu_uuid or value.get("cuda_nvml_agree") is not True:
            raise RuntimeError("CUDA/NVML GPU identity mismatch")
        if self.config.gpu_proof is None or self.worker.child_identity is None: raise RuntimeError("GPU ownership proof is required")
        runner = ProcessIdentity(value["runner_pid"], value["runner_start_time"])
        if runner != self.worker.child_identity:
            raise RuntimeError("worker GPU identity does not match child identity")
        proof = self.config.gpu_proof
        probe = proof.residency_for_runner or proof.residency
        if probe is None: raise RuntimeError("GPU expected-runner residency proof is required")
        try:
            evidence = await settle_residency(
                lambda: probe(runner) if proof.residency_for_runner else probe(),
                sleep=self._residency_sleep, monotonic=self._residency_clock)
            self._validate_residency(evidence, runner)
        except _ResidencyPending:
            if proof.residency_for_runner is None or proof.memory is None:
                raise
            await self._gector_fallback(runner)
            self._model_specific_ready = True
        self._ready = True

    async def _memory(self):
        value = self.config.gpu_proof.memory()
        if inspect.isawaitable(value): value = await value
        self._validate_memory(value)
        return value

    def _validate_memory(self, value):
        if type(value) is not GPUMemoryObservation:
            raise RuntimeError("GPU memory proof is invalid")
        expected = self.config.gpu_proof.expected_supervisor
        fields = (value.start_ns, value.end_ns, value.total_bytes, value.used_bytes, value.free_bytes)
        if (value.gpu_uuid != self.config.gpu_uuid or value.supervisor != expected or
                any(type(x) is not int for x in fields) or value.start_ns > value.end_ns or
                value.total_bytes <= 0 or value.used_bytes < 0 or value.free_bytes < 0 or
                value.used_bytes > value.total_bytes or value.free_bytes > value.total_bytes or
                value.used_bytes + value.free_bytes > value.total_bytes):
            raise RuntimeError("GPU memory proof is malformed or changed")

    def _validate_residency(self, value, runner):
        if (type(value) is not ResidencyEvidence or value.gpu_uuid != self.config.gpu_uuid or
                value.supervisor != self.config.gpu_proof.expected_supervisor or
                runner != self.worker.child_identity or runner not in value.runners or
                len(value.runners) != 1):
            raise RuntimeError("worker is not a proved GPU runner")

    def _validate_child_identity(self, value, runner):
        if (not isinstance(value, dict) or set(value) != {"gpu_uuid", "runner_pid", "runner_start_time", "cuda_nvml_agree"} or
                type(value.get("runner_pid")) is not int or type(value.get("runner_start_time")) is not int or
                value.get("gpu_uuid") != self.config.gpu_uuid or value.get("cuda_nvml_agree") is not True or
                ProcessIdentity(value["runner_pid"], value["runner_start_time"]) != runner or
                self.worker.child_identity != runner):
            raise RuntimeError("worker GPU identity does not match child identity")

    async def _gector_fallback(self, runner):
        proof, pre = self.config.gpu_proof, self._preload_memory
        if type(pre) is not GPUMemoryObservation:
            raise RuntimeError("GECToR fallback baseline memory proof is unavailable")
        self._validate_memory(pre)
        async def sample():
            self._validate_child_identity(await self.worker.call("gpu_identity"), runner)
            witness = await self.worker.call("cuda_residency")
            if witness != {"model_cuda_device": "cuda:0", "witness_exists": True,
                           "witness_cuda_device": "cuda:0"}:
                raise RuntimeError("GECToR CUDA witness is invalid")
            try:
                pending = proof.residency_for_runner(runner)
                if inspect.isawaitable(pending): pending = await pending
            except _ResidencyPending:
                pass
            else:
                raise RuntimeError("GECToR fallback expected typed pending residency")
            post = await self._memory()
            if (post.gpu_uuid != pre.gpu_uuid or post.supervisor != pre.supervisor or
                    post.total_bytes != pre.total_bytes or post.used_bytes <= pre.used_bytes):
                raise RuntimeError("GECToR GPU memory residency effect is invalid")
            return post
        first, second = await sample(), await sample()
        if (first.gpu_uuid, first.supervisor, first.total_bytes) != (second.gpu_uuid, second.supervisor, second.total_bytes):
            raise RuntimeError("GECToR fallback observation identity changed")

    def accepted_model_specific_residency(self):
        return self._model_specific_ready

    @staticmethod
    def _decode(payload):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result: raise ValueError("duplicate JSON key")
                result[key] = value
            return result
        try: return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc: raise ValueError("invalid GECToR request") from exc

    @lifecycle("provider.gector", failures_only=True)
    async def validate_input(self, payload, *, context_size, bucket_identity):
        if self.profile is None or context_size is not None or bucket_identity != self.config.bucket_identity or not self.profile.accepts_request(context_size, bucket_identity) or self.worker is None:
            raise ValueError("request does not match GECToR bucket")
        if not isinstance(payload, bytes) or len(payload) > self.config.rpc_frame_limit: raise ValueError("GECToR request exceeds bound")
        body = self._decode(payload)
        if (not isinstance(body, dict) or set(body) != {"texts", "keep_confidence", "min_error_prob", "n_iteration", "batch_size"} or
            not isinstance(body["texts"], list) or len(body["texts"]) != 1 or not isinstance(body["texts"][0], str) or not body["texts"][0].strip() or
            not isinstance(body["keep_confidence"], (int, float)) or isinstance(body["keep_confidence"], bool) or not math.isfinite(body["keep_confidence"]) or
            not isinstance(body["min_error_prob"], (int, float)) or isinstance(body["min_error_prob"], bool) or not math.isfinite(body["min_error_prob"]) or
            body["keep_confidence"] != self.config.keep_confidence or body["min_error_prob"] != self.config.min_error_prob or
            type(body["n_iteration"]) is not int or body["n_iteration"] != self.config.max_iterations or
            type(body["batch_size"]) is not int or body["batch_size"] != 1):
            raise ValueError("invalid GECToR request")
        result = await self.worker.call("validate", texts=body["texts"], keep_confidence=body["keep_confidence"], min_error_prob=body["min_error_prob"], n_iteration=body["n_iteration"], batch_size=body["batch_size"])
        if type(result) is not dict or set(result) != {"accepted"} or type(result["accepted"]) is not bool:
            raise RuntimeError("malformed GECToR validation response")
        if not result["accepted"]:
            raise ValueError("GECToR request exceeds no-truncation bucket")

    @lifecycle("provider.gector", failures_only=True)
    async def execute(self, request_id, payload):
        if not self._ready or self.worker is None: raise RuntimeError("provider is not ready")
        await self.validate_input(payload, context_size=None, bucket_identity=self.config.bucket_identity)
        body = self._decode(payload)
        result = await self.worker.call("execute", **{key: body[key] for key in ("texts", "keep_confidence", "min_error_prob", "n_iteration", "batch_size")})
        if (not isinstance(result, dict) or set(result) != {"outputs", "observation"}
                or not isinstance(result["outputs"], list) or len(result["outputs"]) != 1
                or not isinstance(result["outputs"][0], str) or not result["outputs"][0].strip()):
            raise RuntimeError("invalid aligned GECToR response")
        observation = result["observation"]
        if not isinstance(observation, dict): raise RuntimeError("invalid GECToR observation")
        observation = {**observation, "request_ids": (request_id,)}
        encoded = json.dumps({"texts": result["outputs"]}, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > self.config.rpc_frame_limit: raise RuntimeError("GECToR response exceeds bound")
        return ProviderResponse(encoded, observation=observation)

    async def cancel(self, request_id): return None
    @lifecycle("provider.gector")
    async def unload(self):
        self._ready = False
        if self.worker is not None: await self.worker.close()
        self.worker = self.profile = None; self._preload_memory = None
        self._model_specific_ready = False; self._cleanup = True
    @lifecycle("provider.gector")
    async def verify_cleanup(self): return self._cleanup and self.worker is None

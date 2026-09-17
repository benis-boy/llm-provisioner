"""Offline GECToR provider using the common owned Python worker transport."""
from __future__ import annotations
import asyncio, json, math
from services.llm.provisioning.volume import verify_current
from services.llm.provisioning.artifacts import SPECS
from services.llm.resource_manager.contracts import CapacityProfile
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.protocol import ProviderResponse
from .gpu import ProcessIdentity, ResidencyEvidence
from .python_process import PythonWorker
from .gector_config import GECToRProviderConfig


class GECToRProvider:
    def __init__(self, config: GECToRProviderConfig):
        self.config, self.worker, self.profile = config, None, None
        self._ready = False
        self._cleanup = True

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

    async def validate(self, profile: CapacityProfile):
        if (profile.model_id is not ModelId.GECTOR or profile.bucket_identity != self.config.bucket_identity
                or profile.context_size is not None or profile.optimal_parallelism != 1
                or profile.buffer_capacity != 1 or not profile.matches(ModelId.GECTOR, self.config.gpu_uuid,
                self.config.manifest_sha256, self.config.model_sha256, self.config.runtime_identity,
                self.config.adapter_identity)):
            raise ValueError("unsupported GECToR profile")
        self.profile = profile

    async def load(self, profile):
        await self.validate(profile)
        root = await asyncio.to_thread(self._artifact)
        if self.worker is not None:
            raise RuntimeError("previous GECToR worker has not been cleaned up")
        self._cleanup = False
        self.worker = PythonWorker(root, {"dtype": self.config.dtype, "gpu_uuid": self.config.gpu_uuid,
            "max_subword_tokens": self.config.max_subword_tokens, "keep_confidence": self.config.keep_confidence,
            "min_error_prob": self.config.min_error_prob, "max_iterations": self.config.max_iterations,
            "batch_size": 1}, command=[__import__('sys').executable, "-m", "services.llm.providers.gector_worker"],
            timeout=self.config.request_timeout_seconds, frame_limit=self.config.rpc_frame_limit, gpu_proof=self.config.gpu_proof)
        try:
            await self.worker.start(); await self.worker.call("load")
        except BaseException:
            await self.worker.close(); raise

    async def ready(self):
        self._ready = False
        if self.worker is None: raise RuntimeError("provider is not loaded")
        value = await self.worker.call("gpu_identity")
        if not isinstance(value, dict) or set(value) != {"gpu_uuid", "runner_pid", "runner_start_time", "cuda_nvml_agree"} or value.get("gpu_uuid") != self.config.gpu_uuid or value.get("cuda_nvml_agree") is not True:
            raise RuntimeError("CUDA/NVML GPU identity mismatch")
        if self.config.gpu_proof is None or self.worker.child_identity is None: raise RuntimeError("GPU ownership proof is required")
        evidence = await self.config.gpu_proof.residency(); runner = ProcessIdentity(value["runner_pid"], value["runner_start_time"])
        if type(evidence) is not ResidencyEvidence or evidence.gpu_uuid != self.config.gpu_uuid or evidence.supervisor != self.config.gpu_proof.expected_supervisor or runner != self.worker.child_identity or runner not in evidence.runners:
            raise RuntimeError("worker is not a proved GPU runner")
        self._ready = True

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

    async def execute(self, request_id, payload):
        if not self._ready or self.worker is None: raise RuntimeError("provider is not ready")
        await self.validate_input(payload, context_size=None, bucket_identity=self.config.bucket_identity)
        body = self._decode(payload)
        result = await self.worker.call("execute", **{key: body[key] for key in ("texts", "keep_confidence", "min_error_prob", "n_iteration", "batch_size")})
        if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], str) or not result[0].strip(): raise RuntimeError("invalid aligned GECToR response")
        encoded = json.dumps({"texts": result}, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > self.config.rpc_frame_limit: raise RuntimeError("GECToR response exceeds bound")
        return ProviderResponse(encoded)

    async def cancel(self, request_id): return None
    async def unload(self):
        self._ready = False
        if self.worker is not None: await self.worker.close()
        self.worker = self.profile = None; self._cleanup = True
    async def verify_cleanup(self): return self._cleanup and self.worker is None

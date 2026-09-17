"""Offline CoEdIT adapter using one bounded child worker per load."""
from __future__ import annotations
import asyncio, json
from services.llm.provisioning.volume import verify_current
from services.llm.provisioning.artifacts import SPECS
from services.llm.resource_manager.contracts import CapacityProfile
from .gpu import ProcessIdentity, ResidencyEvidence
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.protocol import ProviderResponse
from .python_process import PythonWorker
from .coedit_batch import CoEdITBatcher
class CoEdITProvider:
    def __init__(self, config):
        self.config = config
        self.worker = None
        self.profile = None
        self._ready = False
        self._cleanup = True
        self._batcher = None
        self._last_batch_observation = None
        self._cleanup_task = None
    def _artifact(self):
        evidence = verify_current(self.config.artifact_root)
        if evidence.get("manifestSha256") != self.config.manifest_sha256:
            raise ValueError("current artifact manifest mismatch")
        root = self.config.artifact_root / self.config.manifest_sha256
        manifest = json.loads((root / "manifest.json").read_bytes())
        entry = manifest.get("models", {}).get("CoEdIT")
        files = entry.get("files", []) if isinstance(entry, dict) else []
        if [x.get("path") for x in files if isinstance(x, dict)] != list(SPECS["CoEdIT"]):
            raise ValueError("CoEdIT manifest file set mismatch")
        # verify_current already performs selected-manifest canonical validation and
        # streamed hashing for every selected file; do not reread weights here.
        if next(x for x in files if x["path"] == "model.safetensors")["sha256"] != self.config.model_sha256:
            raise ValueError("CoEdIT model identity mismatch")
        return root / "models" / "CoEdIT"
    async def validate(self,profile: CapacityProfile):
        if (profile.model_id is not ModelId.COEDIT or profile.bucket_identity != self.config.bucket_identity
                or profile.context_size is not None or not 1 <= profile.optimal_parallelism <= self.config.max_native_batch_size
                or profile.buffer_capacity != profile.optimal_parallelism or not profile.matches(ModelId.COEDIT,
                self.config.gpu_uuid, self.config.manifest_sha256, self.config.model_sha256,
                self.config.runtime_identity, self.config.adapter_identity)):
            raise ValueError("unsupported CoEdIT profile")
        self.profile=profile
    async def load(self,profile):
        await self.validate(profile); root=await asyncio.to_thread(self._artifact); self._cleanup=False
        if self.worker is not None: raise RuntimeError("previous CoEdIT worker has not been cleaned up")
        self.worker=PythonWorker(root,{"dtype":self.config.dtype,"gpu_uuid":self.config.gpu_uuid,"max_input_tokens":self.config.max_input_tokens,"max_output_tokens":self.config.max_output_tokens,"generation_parameters":dict(self.config.generation_parameters),"max_native_batch_size":self.config.max_native_batch_size,"rpc_frame_limit":self.config.rpc_frame_limit,"cuda_timing":True},timeout=self.config.request_timeout_seconds,frame_limit=self.config.rpc_frame_limit,gpu_proof=self.config.gpu_proof)
        try: await self.worker.start(); await self.worker.call("load")
        except BaseException:
            await self.worker.close()
            self.worker = None
            self._cleanup = True
            raise
        self._batcher = CoEdITBatcher(self.worker, self.config.max_native_batch_size,
            self.config.native_batch_delay_seconds, self.config.max_output_tokens)
    async def ready(self):
        self._ready = False
        if self.worker is None: raise RuntimeError("provider is not loaded")
        value=await self.worker.call("gpu_identity")
        if not isinstance(value,dict) or set(value)!={"gpu_uuid","runner_pid","runner_start_time","cuda_nvml_agree"} or value.get("gpu_uuid")!=self.config.gpu_uuid or value.get("cuda_nvml_agree") is not True: raise RuntimeError("CUDA/NVML GPU identity mismatch")
        if self.config.gpu_proof is None or self.worker.child_identity is None: raise RuntimeError("GPU ownership proof is required")
        evidence=await self.config.gpu_proof.residency()
        expected=self.config.gpu_proof.expected_supervisor
        runner=ProcessIdentity(value["runner_pid"],value["runner_start_time"])
        if type(evidence) is not ResidencyEvidence or evidence.gpu_uuid!=self.config.gpu_uuid or evidence.supervisor!=expected or runner!=self.worker.child_identity or runner not in evidence.runners: raise RuntimeError("worker is not a proved GPU runner")
        self._ready=True
    @staticmethod
    def _decode_input(payload):
        """Decode the immutable request envelope without doing model work.

        Resource-manager admission must not serialize on the owned worker.  The
        worker remains the authority for tokenization, however: ``execute_batch``
        calls its no-truncation validator for every item immediately before
        encoding/model execution.  This local check is only a bounded envelope
        check and is never a provenance claim for a different payload.
        """
        def pairs(items):
            result={}
            for key,value in items:
                if key in result: raise ValueError("duplicate JSON key")
                result[key]=value
            return result
        try:
            body=json.loads(payload.decode("utf-8"),object_pairs_hook=pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))
        except (UnicodeDecodeError,json.JSONDecodeError,ValueError,RecursionError) as exc:
            raise ValueError("invalid CoEdIT request") from exc
        if (set(body) != {"instruction", "texts"} or not isinstance(body["instruction"], str)
                or not body["instruction"].strip() or not isinstance(body["texts"], list)
                or len(body["texts"]) != 1 or not isinstance(body["texts"][0], str)
                or not body["texts"][0].strip()):
            raise ValueError("invalid CoEdIT request")
        return body

    async def validate_input(self,payload,*,context_size,bucket_identity):
        if (self.profile is None or context_size is not None or bucket_identity!=self.config.bucket_identity
                or not self.profile.accepts_request(context_size, bucket_identity) or self.worker is None): raise ValueError("request does not match CoEdIT bucket")
        if not isinstance(payload,bytes) or len(payload)>self.config.rpc_frame_limit: raise ValueError("CoEdIT request exceeds bound")
        self._decode_input(payload)
    async def execute(self,request_id,payload):
        if not self._ready or self.worker is None: raise RuntimeError("provider is not ready")
        # Do not trust a prior admission result (or a caller-held mutable
        # object).  Recheck the exact bytes and bucket here; the child then
        # performs the authoritative tokenizer/no-truncation check inside
        # execute_batch.
        await self.validate_input(payload,context_size=None,
            bucket_identity=self.config.bucket_identity)
        body=self._decode_input(payload)
        if self._batcher is None: raise RuntimeError("provider batcher is unavailable")
        result, observation = await self._batcher.submit(request_id, body["instruction"], body["texts"][0])
        self._last_batch_observation = observation
        if not isinstance(result,str) or not result: raise RuntimeError("invalid aligned CoEdIT response")
        encoded=json.dumps({"texts":[result]},separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
        if len(encoded)>self.config.rpc_frame_limit: raise RuntimeError("CoEdIT response exceeds bound")
        return ProviderResponse(encoded)
    async def cancel(self,request_id):
        if self._batcher is not None: self._batcher.cancel(request_id)

    def drain_batch_observations(self):
        return self._batcher.drain_observations() if self._batcher is not None else ()
    def batch_observation_drops(self):
        return self._batcher.dropped_observations if self._batcher is not None else 0
    async def unload(self):
        self._ready = False
        if self._cleanup_task is None or self._cleanup_task.done():
            batcher, worker = self._batcher, self.worker
            if batcher is not None:
                batcher.fence()
            self._cleanup_task = asyncio.create_task(self._cleanup_owned(batcher, worker))
        await asyncio.shield(self._cleanup_task)

    async def _cleanup_owned(self, batcher, worker):
        # Closing the worker first interrupts its serialized, potentially long RPC;
        # the fenced batcher can then join without publishing a late result.
        if worker is not None:
            await worker.close()
        if batcher is not None:
            await batcher.join()
        if self.worker is worker and self._batcher is batcher:
            self.worker=self.profile=self._batcher=None
            self._last_batch_observation=None
            self._ready=False
            self._cleanup=True
    async def verify_cleanup(self): return self._cleanup and self.worker is None

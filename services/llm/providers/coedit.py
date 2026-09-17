"""Offline CoEdIT adapter using one bounded child worker per load."""
from __future__ import annotations
import asyncio, inspect, json, time
from services.llm.provisioning.volume import verify_current
from services.llm.provisioning.artifacts import SPECS
from services.llm.resource_manager.contracts import CapacityProfile
from .gpu import (GPUMemoryObservation, ProcessIdentity, ResidencyEvidence,
                  _ResidencyPending, settle_residency)
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
        self._model_specific_ready = False
        self._preload_memory = None
        self._cleanup = True
        self._batcher = None
        self._last_batch_observation = None
        self._cleanup_task = None
        self._residency_sleep = asyncio.sleep
        self._residency_clock = time.monotonic
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
        self._ready = False
        self._model_specific_ready = False
        await self.validate(profile); root=await asyncio.to_thread(self._artifact); self._cleanup=False
        self._preload_memory = await self._memory() if self.config.gpu_proof.memory is not None else None
        if self.worker is not None: raise RuntimeError("previous CoEdIT worker has not been cleaned up")
        self.worker=PythonWorker(root,{"dtype":self.config.dtype,"gpu_uuid":self.config.gpu_uuid,"max_input_tokens":self.config.max_input_tokens,"max_output_tokens":self.config.max_output_tokens,"generation_parameters":dict(self.config.generation_parameters),"max_native_batch_size":self.config.max_native_batch_size,"rpc_frame_limit":self.config.rpc_frame_limit,"cuda_timing":True},timeout=self.config.request_timeout_seconds,frame_limit=self.config.rpc_frame_limit,gpu_proof=self.config.gpu_proof)
        try:
            await self.worker.start()
            await self.worker.call("load")
            # Loading alone may not create an NVML compute-process record.
            # Synchronize a deterministic, input-free CUDA witness first.
            await self.worker.call("cuda_ready")
        except BaseException:
            await self.worker.close()
            self.worker = None
            self._cleanup = True
            raise
        self._batcher = CoEdITBatcher(self.worker, self.config.max_native_batch_size,
            self.config.native_batch_delay_seconds, self.config.max_output_tokens)
    async def ready(self):
        self._ready = False
        self._model_specific_ready = False
        if self.worker is None: raise RuntimeError("provider is not loaded")
        value=await self.worker.call("gpu_identity")
        if not isinstance(value,dict) or set(value)!={"gpu_uuid","runner_pid","runner_start_time","cuda_nvml_agree"} or value.get("gpu_uuid")!=self.config.gpu_uuid or value.get("cuda_nvml_agree") is not True: raise RuntimeError("CUDA/NVML GPU identity mismatch")
        if self.config.gpu_proof is None or self.worker.child_identity is None: raise RuntimeError("GPU ownership proof is required")
        runner=ProcessIdentity(value["runner_pid"],value["runner_start_time"])
        if runner != self.worker.child_identity: raise RuntimeError("worker GPU identity does not match child identity")
        proof = self.config.gpu_proof
        probe = proof.residency_for_runner or proof.residency
        if probe is None: raise RuntimeError("GPU expected-runner residency proof is required")
        try:
            evidence=await settle_residency(lambda: probe(runner) if proof.residency_for_runner else probe(),
                sleep=self._residency_sleep, monotonic=self._residency_clock)
            self._validate_residency(evidence, runner)
        except _ResidencyPending:
            if proof.residency_for_runner is None or proof.memory is None:
                raise
            await self._coedit_fallback(runner)
        self._ready=True
        self._model_specific_ready=True

    async def _memory(self):
        value = self.config.gpu_proof.memory()
        if inspect.isawaitable(value):
            value = await value
        self._validate_memory_observation(value)
        return value

    def _validate_memory_observation(self, value):
        if type(value) is not GPUMemoryObservation:
            raise RuntimeError("GPU memory proof is invalid")
        expected = self.config.gpu_proof.expected_supervisor
        fields = (value.start_ns, value.end_ns, value.total_bytes, value.used_bytes, value.free_bytes)
        if (value.gpu_uuid != self.config.gpu_uuid or value.supervisor != expected or
                any(type(item) is not int for item in fields) or value.start_ns > value.end_ns or
                value.total_bytes <= 0 or value.used_bytes < 0 or value.free_bytes < 0 or
                value.used_bytes > value.total_bytes or value.free_bytes > value.total_bytes or
                value.used_bytes + value.free_bytes > value.total_bytes):
            raise RuntimeError("GPU memory proof is malformed or changed")

    def _validate_residency(self, value, runner):
        expected = self.config.gpu_proof.expected_supervisor
        if (type(value) is not ResidencyEvidence or value.gpu_uuid != self.config.gpu_uuid or
                value.supervisor != expected or runner != self.worker.child_identity or
                runner not in value.runners or len(value.runners) != 1):
            raise RuntimeError("worker is not a proved GPU runner")

    async def _coedit_fallback(self, runner):
        """Use only the bounded CoEdIT composition when NVML has no child PID."""
        proof = self.config.gpu_proof
        expected = proof.expected_supervisor
        pre = self._preload_memory
        if type(pre) is not GPUMemoryObservation or pre.supervisor != expected:
            raise RuntimeError("CoEdIT fallback baseline memory proof is unavailable")
        self._validate_memory_observation(pre)
        async def sample():
            identity = await self.worker.call("gpu_identity")
            self._validate_child_identity(identity, runner)
            witness = await self.worker.call("cuda_residency")
            if (type(witness) is not dict or set(witness) != {"model_cuda_device","witness_exists","witness_cuda_device"}
                    or witness != {"model_cuda_device":"cuda:0","witness_exists":True,"witness_cuda_device":"cuda:0"}):
                raise RuntimeError("CoEdIT CUDA witness is invalid")
            # This call is the public ancestry/device fence; its pending result is
            # expected here, while every other proof error remains terminal.
            try:
                pending = proof.residency_for_runner(runner)
                if inspect.isawaitable(pending):
                    pending = await pending
            except _ResidencyPending:
                # Absence is the only reason to use this model-specific path.
                # In particular, do not turn a malformed or newly positive
                # topology result into memory-based ownership.
                pass
            else:
                raise RuntimeError("CoEdIT fallback expected typed pending residency")
            post = await self._memory()
            if (post.total_bytes != pre.total_bytes or post.gpu_uuid != pre.gpu_uuid or
                    post.supervisor != expected or post.used_bytes <= pre.used_bytes):
                raise RuntimeError("CoEdIT GPU memory residency effect is invalid")
            return post
        first = await sample()
        second = await sample()
        # Samples are points, not snapshots.  Allocator/NVML readings may
        # fluctuate; only the immutable device/fence identity must remain
        # stable, and each point has already been independently validated.
        if ((first.gpu_uuid, first.supervisor, first.total_bytes) !=
                (second.gpu_uuid, second.supervisor, second.total_bytes)):
            raise RuntimeError("CoEdIT fallback observation identity changed")

    def _validate_child_identity(self, value, runner):
        try:
            identity = ProcessIdentity(value["runner_pid"], value["runner_start_time"])
        except (KeyError, TypeError):
            identity = None
        if (not isinstance(value, dict) or set(value) != {"gpu_uuid","runner_pid","runner_start_time","cuda_nvml_agree"}
                or type(value.get("runner_pid")) is not int or type(value.get("runner_start_time")) is not int
                or value.get("gpu_uuid") != self.config.gpu_uuid or value.get("cuda_nvml_agree") is not True
                or identity != runner or self.worker.child_identity != runner):
            raise RuntimeError("worker GPU identity does not match child identity")

    def accepted_model_specific_residency(self) -> bool:
        return self._model_specific_ready
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
            self._model_specific_ready=False
            self._preload_memory=None
            self._cleanup=True
    async def verify_cleanup(self): return self._cleanup and self.worker is None

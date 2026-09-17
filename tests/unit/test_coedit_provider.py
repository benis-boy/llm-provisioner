import asyncio
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from services.llm.provisioning.artifacts import SPECS
from services.llm.provisioning.volume import provision
from services.llm.providers.coedit import CoEdITProvider
from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import ProcessIdentity, ResidencyEvidence
from services.llm.providers.python_config import PythonProviderConfig
from services.llm.providers.coedit_batch import CoEdITBatcher
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata


class CoEdITProviderTests(unittest.IsolatedAsyncioTestCase):
    def config(self, root, manifest="0" * 64, model="0" * 64):
        raw=(Path("/proc")/str(os.getpid())/"stat").read_bytes()
        supervisor=ProcessIdentity(os.getpid(),int(raw[raw.rfind(b")")+2:].split()[19]))
        proof=GPUProof(lambda: "GPU-test", AsyncMock(return_value=True), lambda: None, supervisor)
        return PythonProviderConfig(Path(root), manifest, model, "GPU-test", "runtime", "adapter", gpu_proof=proof)

    def profile(self, config):
        return CapacityProfile(ModelId.COEDIT, config.gpu_uuid, config.manifest_sha256,
            config.model_sha256, config.runtime_identity, config.adapter_identity,
            config.bucket_identity, 1, 1, 1, 0, (SampleMetadata(1,0,1,1,1,(1,)),),
            bucket_identity=config.bucket_identity)

    async def test_selected_volume_layout_and_streamed_integrity_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"source"; source.mkdir()
            for name in SPECS["CoEdIT"]: (source/name).write_bytes(name.encode())
            volume=Path(directory)/"volume"; document=provision({"CoEdIT":source},volume)
            model=next(x for x in document["models"]["CoEdIT"]["files"] if x["path"]=="model.safetensors")["sha256"]
            provider=CoEdITProvider(self.config(volume,document["manifest_sha256"],model))
            self.assertEqual(provider._artifact(),volume/document["manifest_sha256"]/"models"/"CoEdIT")
            (volume/document["manifest_sha256"]/"models"/"CoEdIT"/"config.json").write_text("corrupt")
            with self.assertRaises(ValueError): provider._artifact()

    async def test_parent_has_no_torch_import(self):
        import services.llm.providers.python_process as process
        self.assertNotIn("torch", process.__dict__)

    async def test_exact_profile_and_native_batch_one_request_are_required(self):
        config=self.config("/tmp"); provider=CoEdITProvider(config); provider.profile=self.profile(config)
        worker=AsyncMock(); provider.worker=worker
        await provider.validate_input(b'{"instruction":"fix","texts":["text"]}',context_size=None,bucket_identity=config.bucket_identity)
        # Admission performs only bounded envelope validation.  Tokenization is
        # authoritative in the child's execute_batch operation, so concurrent
        # admissions do not serialize one validation RPC per request.
        worker.call.assert_not_awaited()
        with self.assertRaises(ValueError):
            await provider.validate_input(b'{"instruction":"fix","texts":["one","two"]}',context_size=None,bucket_identity=config.bucket_identity)
        # Registry profile identities are measurement identities, not buckets;
        # memory_safe_n may retain diagnostic headroom above p=1.
        accepted=CapacityProfile(ModelId.COEDIT,"GPU-test","0"*64,"0"*64,"runtime","adapter","registry-measurement-9",1,3,1,0,(SampleMetadata(1,0,1,1,1,()),),bucket_identity=config.bucket_identity)
        await provider.validate(accepted)
        wrong=CapacityProfile(ModelId.COEDIT,"GPU-test","0"*64,"0"*64,"runtime","adapter","other",1,1,1,0,(SampleMetadata(1,0,1,1,1,()),),bucket_identity="other")
        with self.assertRaises(ValueError): await provider.validate(wrong)

    async def test_opt_in_batch_profile_is_bound_to_exact_batch_bucket(self):
        config = self.config("/tmp")
        config = PythonProviderConfig(Path("/tmp"), config.manifest_sha256, config.model_sha256,
            config.gpu_uuid, config.runtime_identity, config.adapter_identity,
            bucket_identity="coedit:p2:input128:output64:float16:beams1:nosample",
            max_native_batch_size=2, gpu_proof=config.gpu_proof)
        provider = CoEdITProvider(config)
        profile = CapacityProfile(ModelId.COEDIT, config.gpu_uuid, config.manifest_sha256,
            config.model_sha256, config.runtime_identity, config.adapter_identity,
            "measured", 1, 2, 1, 0, (SampleMetadata(1, 0, 1, 1, 1, ()),),
            bucket_identity=config.bucket_identity)
        await provider.validate(profile)
        self.assertEqual(provider.profile.optimal_parallelism, 1)
        with self.assertRaises(ValueError):
            await provider.validate(profile.__class__(ModelId.COEDIT, config.gpu_uuid,
                config.manifest_sha256, config.model_sha256, config.runtime_identity,
                config.adapter_identity, "too-high", 3, 3, 3, 0, profile.raw_samples,
                bucket_identity=config.bucket_identity))
        # Measurement identity is diagnostic; the exact request bucket is the
        # independently pinned bucket_identity above.
        await provider.validate(profile.__class__(ModelId.COEDIT, config.gpu_uuid,
            config.manifest_sha256, config.model_sha256, config.runtime_identity,
            config.adapter_identity, "wrong", 2, 2, 2, 0, profile.raw_samples,
            bucket_identity=config.bucket_identity))
        with self.assertRaises(ValueError):
            PythonProviderConfig(Path("/tmp"), config.manifest_sha256, config.model_sha256,
                config.gpu_uuid, config.runtime_identity, config.adapter_identity,
                bucket_identity=config.bucket_identity, max_native_batch_size=2,
                native_batch_delay_seconds=.01, gpu_proof=config.gpu_proof)

    async def test_native_batch_configuration_accepts_32_rejects_33_and_defaults_to_one(self):
        config = self.config("/tmp")
        self.assertEqual(config.max_native_batch_size, 1)
        config32 = PythonProviderConfig(
            Path("/tmp"), config.manifest_sha256, config.model_sha256,
            config.gpu_uuid, config.runtime_identity, config.adapter_identity,
            bucket_identity="coedit:p32:input128:output64:float16:beams1:nosample",
            max_native_batch_size=32, gpu_proof=config.gpu_proof)
        self.assertEqual(config32.max_native_batch_size, 32)
        with self.assertRaisesRegex(ValueError, "native batch size"):
            PythonProviderConfig(
                Path("/tmp"), config.manifest_sha256, config.model_sha256,
                config.gpu_uuid, config.runtime_identity, config.adapter_identity,
                bucket_identity="coedit:p33:input128:output64:float16:beams1:nosample",
                max_native_batch_size=33, gpu_proof=config.gpu_proof)

    async def test_execute_preserves_order_and_returns_individual_frames(self):
        config = self.config("/tmp")
        provider = CoEdITProvider(config)
        provider.profile = self.profile(config)
        provider._ready = True
        worker = AsyncMock()
        worker.frame_limit = 4096
        worker.call.return_value = True
        provider.worker = worker
        provider._batcher = CoEdITBatcher(worker, 1, 0, config.max_output_tokens)
        worker.call.side_effect = [{"outputs": ["fixed"], "observation": {
            "batch_size": 1, "execution_started": 1, "execution_ended": 2,
            "cuda_synchronized": True, "allocator": {"baseline_allocated": 10, "baseline_reserved": 20,
            "peak_allocated": 30, "peak_reserved": 40, "final_allocated": 10, "final_reserved": 20}, "decoder_steps": [64], "max_output_tokens": 64}}]
        response = await provider.execute("request", b'{"instruction":"fix","texts":["text"]}')
        self.assertEqual(response.result, b'{"texts":["fixed"]}')
        self.assertEqual(worker.call.await_args_list[0].args[0], "execute_batch")
        await provider._batcher.close()

    async def test_invalid_input_does_not_poison_worker(self):
        config = self.config("/tmp")
        provider = CoEdITProvider(config)
        provider.profile = self.profile(config)
        worker = AsyncMock()
        worker.frame_limit = 4096
        provider.worker = worker
        with self.assertRaises(ValueError):
            await provider.validate_input(b'{"instruction":"fix","texts":["one","two"]}',
                context_size=None, bucket_identity=config.bucket_identity)
        await provider.validate_input(b'{"instruction":"fix","texts":["one"]}',
            context_size=None, bucket_identity=config.bucket_identity)
        self.assertEqual(worker.call.await_count, 0)

    async def test_concurrent_admission_validation_is_local_and_bounded(self):
        config = self.config("/tmp")
        provider = CoEdITProvider(config)
        provider.profile = self.profile(config)
        worker = AsyncMock()
        worker.frame_limit = 4096
        provider.worker = worker
        payloads = [f'{{"instruction":"fix-{n}","texts":["text-{n}"]}}'.encode()
                    for n in range(8)]
        await asyncio.gather(*(provider.validate_input(payload, context_size=None,
            bucket_identity=config.bucket_identity) for payload in payloads))
        worker.call.assert_not_awaited()

    async def test_execute_rechecks_exact_bytes_before_child_batch_execution(self):
        config = self.config("/tmp")
        provider = CoEdITProvider(config)
        provider.profile = self.profile(config)
        provider._ready = True
        worker = AsyncMock()
        worker.frame_limit = 4096
        worker.call.return_value = {"outputs": ["fixed"], "observation": {
            "batch_size": 1, "execution_started": 1, "execution_ended": 2,
            "cuda_synchronized": True, "allocator": {
                "baseline_allocated": 10, "baseline_reserved": 20,
                "peak_allocated": 30, "peak_reserved": 40,
                "final_allocated": 10, "final_reserved": 20}, "decoder_steps": [64], "max_output_tokens": 64}}
        provider.worker = worker
        provider._batcher = CoEdITBatcher(worker, 1, 0, config.max_output_tokens)
        with self.assertRaises(ValueError):
            await provider.execute("bad", b'{"instruction":"fix","texts":["one","two"]}')
        worker.call.assert_not_awaited()
        await provider._batcher.close()

    async def test_ready_requires_exact_child_runner_residency(self):
        config=self.config("/tmp"); provider=CoEdITProvider(config)
        runner=ProcessIdentity(123,456)
        worker=AsyncMock(); worker.child_identity=runner
        worker.call.return_value={"gpu_uuid":"GPU-test","runner_pid":123,"runner_start_time":456,"cuda_nvml_agree":True}
        provider.worker=worker
        object.__setattr__(config.gpu_proof, "residency", AsyncMock(return_value=ResidencyEvidence("GPU-test",config.gpu_proof.expected_supervisor,(runner,))))
        await provider.ready(); self.assertTrue(provider._ready)

    async def test_failed_cleanup_retains_provider_worker(self):
        provider=CoEdITProvider(self.config("/tmp")); worker=AsyncMock(); worker.close.side_effect=RuntimeError("uncertain")
        provider.worker=worker
        with self.assertRaises(RuntimeError): await provider.unload()
        self.assertIs(provider.worker,worker)

    async def test_concurrent_unload_closes_worker_once_after_batcher_fence(self):
        config = self.config("/tmp")
        provider = CoEdITProvider(config)
        worker = AsyncMock()
        worker.frame_limit = 4096
        provider.worker = worker
        provider._batcher = CoEdITBatcher(worker, 1, 0, config.max_output_tokens)
        await asyncio.gather(provider.unload(), provider.unload())
        worker.close.assert_awaited_once()
        self.assertTrue(await provider.verify_cleanup())

    async def test_cancelled_unload_retains_cleanup_that_closes_worker_before_joining_batcher(self):
        provider = CoEdITProvider(self.config("/tmp"))
        worker_closed = asyncio.Event()
        batcher_release = asyncio.Event()
        worker = AsyncMock()
        async def close_worker(): worker_closed.set()
        worker.close.side_effect = close_worker
        batcher = AsyncMock()
        batcher.fence = Mock()
        batcher.join.side_effect = batcher_release.wait
        provider.worker, provider._batcher = worker, batcher
        unload = asyncio.create_task(provider.unload())
        await worker_closed.wait()
        unload.cancel()
        with self.assertRaises(asyncio.CancelledError): await unload
        self.assertFalse(provider._cleanup_task.done())
        batcher_release.set()
        await provider.unload()
        worker.close.assert_awaited_once()
        self.assertTrue(await provider.verify_cleanup())

    async def test_failed_reready_fences_execute(self):
        config=self.config("/tmp"); provider=CoEdITProvider(config); provider._ready=True
        worker=AsyncMock(); worker.call.return_value={"gpu_uuid":"wrong"}; provider.worker=worker
        with self.assertRaises(RuntimeError): await provider.ready()
        self.assertFalse(provider._ready)
        with self.assertRaises(RuntimeError): await provider.execute("request",b'{"instruction":"fix","texts":["text"]}')

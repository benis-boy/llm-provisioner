import asyncio
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from services.llm.provisioning.artifacts import SPECS
from services.llm.provisioning.volume import provision
from services.llm.providers.coedit import CoEdITProvider
from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import ProcessIdentity, ResidencyEvidence
from services.llm.providers.python_config import PythonProviderConfig
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
        worker.call.assert_awaited_once()
        with self.assertRaises(ValueError):
            await provider.validate_input(b'{"instruction":"fix","texts":["one","two"]}',context_size=None,bucket_identity=config.bucket_identity)
        # Registry profile identities are measurement identities, not buckets;
        # memory_safe_n may retain diagnostic headroom above p=1.
        accepted=CapacityProfile(ModelId.COEDIT,"GPU-test","0"*64,"0"*64,"runtime","adapter","registry-measurement-9",1,3,1,0,(SampleMetadata(1,0,1,1,1,()),),bucket_identity=config.bucket_identity)
        await provider.validate(accepted)
        wrong=CapacityProfile(ModelId.COEDIT,"GPU-test","0"*64,"0"*64,"runtime","adapter","other",1,1,1,0,(SampleMetadata(1,0,1,1,1,()),),bucket_identity="other")
        with self.assertRaises(ValueError): await provider.validate(wrong)

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

    async def test_failed_reready_fences_execute(self):
        config=self.config("/tmp"); provider=CoEdITProvider(config); provider._ready=True
        worker=AsyncMock(); worker.call.return_value={"gpu_uuid":"wrong"}; provider.worker=worker
        with self.assertRaises(RuntimeError): await provider.ready()
        self.assertFalse(provider._ready)
        with self.assertRaises(RuntimeError): await provider.execute("request",b'{"instruction":"fix","texts":["text"]}')

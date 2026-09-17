import unittest
import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import GPUMemoryObservation, ProcessIdentity, ResidencyEvidence, _ResidencyPending
from services.llm.providers.gector_config import GECToRProviderConfig
from services.llm.providers.gector import GECToRProvider
from services.llm.provisioning.artifacts import SPECS
from services.llm.provisioning.volume import provision
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata


class GECToRConfigTests(unittest.TestCase):
    def proof(self):
        return GPUProof(lambda: "GPU-test", AsyncMock(return_value=True), AsyncMock(), ProcessIdentity(1, 1))

    def kwargs(self):
        return dict(artifact_root=Path("/tmp/gector"), manifest_sha256="0" * 64,
                    model_sha256="1" * 64, gpu_uuid="GPU-test", runtime_identity="runtime",
                    adapter_identity="adapter", gpu_proof=self.proof())

    def test_exact_bucket_is_bound(self):
        config = GECToRProviderConfig(**self.kwargs())
        self.assertEqual(config.bucket_identity, "gector:p1:tokens128:keep0:min0:iterations1:batch1:float32")

    def test_non_single_iteration_is_rejected(self):
        with self.assertRaises(ValueError):
            GECToRProviderConfig(**self.kwargs(), max_iterations=2)

    def test_parent_module_has_no_optional_runtime_import(self):
        import services.llm.providers.gector as provider
        self.assertNotIn("torch", provider.__dict__)
        self.assertNotIn("gector", provider.__dict__)

    def test_request_shape_is_strict_and_scalar_types_do_not_coerce(self):
        config = GECToRProviderConfig(**self.kwargs())
        provider = GECToRProvider(config)
        provider.worker = AsyncMock()
        provider.profile = CapacityProfile(ModelId.GECTOR, "GPU-test", "0" * 64, "1" * 64, "runtime", "adapter", config.bucket_identity, 1, 3, 1, 0, (SampleMetadata(1, 0, 1, 1, 1, ()),), bucket_identity=config.bucket_identity)
        for value in ([], None, True, 1):
            payload = json.dumps(value).encode()
            with self.assertRaises(ValueError): asyncio.run(provider.validate_input(payload, context_size=None, bucket_identity=config.bucket_identity))
        body = {"texts":["hello"],"keep_confidence":False,"min_error_prob":0.0,"n_iteration":1,"batch_size":1}
        with self.assertRaises(ValueError): asyncio.run(provider.validate_input(json.dumps(body).encode(), context_size=None, bucket_identity=config.bucket_identity))

    def test_numeric_zero_json_values_are_accepted_but_boolean_and_nonfinite_are_not(self):
        config = GECToRProviderConfig(**self.kwargs(), keep_confidence=0, min_error_prob=0)
        provider = GECToRProvider(config)
        provider.worker = AsyncMock(); provider.worker.call.return_value = {"accepted": True}
        provider.profile = CapacityProfile(ModelId.GECTOR, "GPU-test", "0" * 64, "1" * 64, "runtime", "adapter", config.bucket_identity, 1, 2, 1, 0, (SampleMetadata(1, 0, 1, 1, 1, ()),), bucket_identity=config.bucket_identity)
        body = {"texts":["hello"], "keep_confidence":0, "min_error_prob":0.0, "n_iteration":1, "batch_size":1}
        asyncio.run(provider.validate_input(json.dumps(body).encode(), context_size=None, bucket_identity=config.bucket_identity))
        body["keep_confidence"] = True
        with self.assertRaises(ValueError): asyncio.run(provider.validate_input(json.dumps(body).encode(), context_size=None, bucket_identity=config.bucket_identity))

    def test_expected_overlong_validation_is_a_value_error_without_closing_worker(self):
        config = GECToRProviderConfig(**self.kwargs())
        provider = GECToRProvider(config); worker = AsyncMock(); worker.call.return_value = {"accepted": False}
        provider.worker = worker
        provider.profile = CapacityProfile(ModelId.GECTOR, "GPU-test", "0" * 64, "1" * 64, "runtime", "adapter", config.bucket_identity, 1, 3, 1, 0, (SampleMetadata(1, 0, 1, 1, 1, ()),), bucket_identity=config.bucket_identity)
        body = {"texts":["too long"], "keep_confidence":0.0, "min_error_prob":0.0, "n_iteration":1, "batch_size":1}
        with self.assertRaises(ValueError): asyncio.run(provider.validate_input(json.dumps(body).encode(), context_size=None, bucket_identity=config.bucket_identity))
        self.assertIs(provider.worker, worker)

    def test_profile_identity_is_independent_from_memory_safe_capacity(self):
        config = GECToRProviderConfig(**self.kwargs())
        profile = CapacityProfile(ModelId.GECTOR, "GPU-test", "0" * 64, "1" * 64, "runtime", "adapter", config.bucket_identity, 1, 8, 1, 0, (SampleMetadata(1, 0, 1, 1, 1, ()),), bucket_identity=config.bucket_identity)
        asyncio.run(GECToRProvider(config).validate(profile))

    def test_failed_load_closes_worker_and_allows_reload_after_cleanup(self):
        config = GECToRProviderConfig(**self.kwargs())
        provider = GECToRProvider(config)
        profile = CapacityProfile(ModelId.GECTOR, "GPU-test", "0" * 64, "1" * 64, "runtime", "adapter", config.bucket_identity, 1, 2, 1, 0, (SampleMetadata(1, 0, 1, 1, 1, ()),), bucket_identity=config.bucket_identity)
        worker = AsyncMock(); worker.start.side_effect = RuntimeError("load failed")
        with patch("services.llm.providers.gector.PythonWorker", return_value=worker), patch.object(provider, "_artifact", return_value=Path("/offline")):
            with self.assertRaisesRegex(RuntimeError, "load failed"): asyncio.run(provider.load(profile))
            self.assertIsNone(provider.worker)
            self.assertTrue(provider._cleanup)
        asyncio.run(provider.unload())
        self.assertTrue(asyncio.run(provider.verify_cleanup()))

    def test_ready_uses_typed_pending_fallback_with_identity_witness_and_stable_memory(self):
        supervisor = ProcessIdentity(1, 1)
        runner = ProcessIdentity(2, 3)
        memory = iter((
            GPUMemoryObservation("GPU-test", supervisor, 1, 2, 1000, 400, 600),
            GPUMemoryObservation("GPU-test", supervisor, 3, 4, 1000, 500, 500),
            GPUMemoryObservation("GPU-test", supervisor, 5, 6, 1000, 600, 400),
        ))
        async def pending(_runner):
            raise _ResidencyPending("runner observation is pending")
        proof = GPUProof(lambda: "GPU-test", AsyncMock(return_value=True),
                         expected_supervisor=supervisor, residency_for_runner=pending,
                         memory=AsyncMock(side_effect=lambda: next(memory)))
        config = GECToRProviderConfig(**self.kwargs())
        object.__setattr__(config, "gpu_proof", proof)
        provider = GECToRProvider(config)
        worker = AsyncMock(); worker.child_identity = runner
        worker.call.side_effect = lambda operation, **kwargs: {
            "gpu_identity": {"gpu_uuid": "GPU-test", "runner_pid": 2,
                              "runner_start_time": 3, "cuda_nvml_agree": True},
            "cuda_residency": {"model_cuda_device": "cuda:0", "witness_exists": True,
                                "witness_cuda_device": "cuda:0"},
        }[operation]
        provider.worker = worker
        provider._preload_memory = asyncio.run(provider._memory())
        # The first residency settlement is typed-pending; fallback then requires
        # two positive, identity-stable post-load memory observations.
        asyncio.run(provider.ready())
        self.assertTrue(provider._ready)
        self.assertTrue(provider.accepted_model_specific_residency())

    def test_selected_volume_requires_real_vocab_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"; source.mkdir()
            for name in SPECS["GECToR"]: (source / name).write_bytes(name.encode())
            volume = Path(directory) / "volume"; document = provision({"GECToR": source}, volume)
            config = GECToRProviderConfig(artifact_root=volume, manifest_sha256=document["manifest_sha256"], model_sha256=next(item["sha256"] for item in document["models"]["GECToR"]["files"] if item["path"] == "model.safetensors"), gpu_uuid="GPU-test", runtime_identity="runtime", adapter_identity="adapter", gpu_proof=self.proof())
            provider = GECToRProvider(config)
            self.assertTrue(provider._artifact().joinpath("verb-form-vocab.txt").is_file())
            vocab = provider._artifact() / "verb-form-vocab.txt"; vocab.write_text("corrupt")
            with self.assertRaises(ValueError): provider._artifact()

    def test_ready_uses_the_same_bounded_empty_runner_settlement(self):
        config = GECToRProviderConfig(**self.kwargs())
        provider = GECToRProvider(config)
        runner, now, calls = ProcessIdentity(2, 3), [0.0], [0]
        worker = AsyncMock(); worker.child_identity = runner
        worker.call.return_value = {"gpu_uuid": "GPU-test", "runner_pid": 2,
                                    "runner_start_time": 3, "cuda_nvml_agree": True}
        provider.worker = worker
        async def residency():
            calls[0] += 1
            if calls[0] < 2: raise _ResidencyPending("GPU has no resident runner")
            return ResidencyEvidence("GPU-test", config.gpu_proof.expected_supervisor, (runner,))
        async def sleep(delay): now[0] += delay
        object.__setattr__(config.gpu_proof, "residency", residency)
        provider._residency_clock = lambda: now[0]
        provider._residency_sleep = sleep
        asyncio.run(provider.ready())
        self.assertTrue(provider._ready)

"""Phase 4 local acceptance using real adapter orchestration and owned seams."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import web

from services.llm.provisioning.artifacts import SPECS
from services.llm.provisioning.volume import provision
from services.llm.providers.coedit import CoEdITProvider
from services.llm.providers.config import GPUProof, SmolLMProviderConfig
from services.llm.providers.gector import GECToRProvider
from services.llm.providers.gector_config import GECToRProviderConfig
from services.llm.providers.gpu import GPUMemoryObservation, ProcessIdentity, ResidencyEvidence
from services.llm.providers.python_config import PythonProviderConfig
from services.llm.providers.smollm import SmolLMProvider
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager, ResourceManagerError
from services.llm.resource_manager.protocol import EventKind, ProviderResponse


def _profile(model, manifest, model_hash, bucket=None, p=1):
    shape = {"context_size": 512} if model is ModelId.SMOLLM else {"bucket_identity": bucket}
    return CapacityProfile(model, "GPU-test", manifest, model_hash, "runtime", "adapter", "phase4",
                           p, p, p, 20, (SampleMetadata(1, 0, 1, 1, 1, (1,)),), **shape)


class _Worker:
    """Controlled worker transport seam; adapter lifecycle remains real."""
    def __init__(self, *args, **kwargs):
        self.child_identity = ProcessIdentity(123, 456)
        self.closed = False
        self.frame_limit = 256 * 1024
    async def start(self): pass
    async def close(self): self.closed = True
    async def call(self, operation, **kwargs):
        if operation == "load": return None
        if operation == "gpu_identity":
            return {"gpu_uuid": "GPU-test", "runner_pid": 123, "runner_start_time": 456,
                    "cuda_nvml_agree": True}
        if operation == "cuda_ready": return {"retained": True}
        if operation == "validate": return {"accepted": True}
        if operation == "execute": return ["fixed"]
        if operation == "execute_batch":
            return {"outputs": ["fixed"], "observation": {"batch_size": 1,
                "execution_started": 1, "execution_ended": 2, "cuda_synchronized": True,
                "allocator": {"baseline_allocated": 1, "baseline_reserved": 1,
                              "peak_allocated": 1, "peak_reserved": 1,
                              "final_allocated": 1, "final_reserved": 1},
                "decoder_steps": [1], "max_output_tokens": 64}}
        raise AssertionError(operation)


class _BlockingProvider:
    """Explicit RM-only fake, never presented as an adapter."""
    def __init__(self, cleanup_ok=True):
        self.release, self.started, self.calls = asyncio.Event(), asyncio.Event(), []
        self.unload_started, self.unload_gate = asyncio.Event(), None
        self.cleanup_ok = cleanup_ok
    async def validate(self, profile): pass
    async def load(self, profile): pass
    async def ready(self): pass
    async def validate_input(self, payload, **kwargs): pass
    async def execute(self, request_id, payload):
        self.calls.append(("execute", request_id)); self.started.set(); await self.release.wait()
        return ProviderResponse(b"result", None, False)
    async def cancel(self, request_id): self.calls.append(("cancel", request_id))
    async def unload(self):
        self.unload_started.set()
        if self.unload_gate is not None: await self.unload_gate.wait()
    async def verify_cleanup(self): return self.cleanup_ok


class Phase4ResourceManagerAcceptance(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name); source = root / "source"; source.mkdir()
        for model, names in SPECS.items():
            folder = source / model; folder.mkdir()
            for name in names: (folder / name).write_bytes((model + name).encode())
        self.volume = root / "volume"
        self.document = provision({model: source / model for model in SPECS}, self.volume)
        raw = (Path("/proc") / str(os.getpid()) / "stat").read_bytes()
        self.supervisor = ProcessIdentity(os.getpid(), int(raw[raw.rfind(b")") + 2:].split()[19]))
        self.proof = GPUProof(lambda: "GPU-test", AsyncMock(return_value=True),
            AsyncMock(return_value=ResidencyEvidence("GPU-test", self.supervisor,
                                                     (ProcessIdentity(123, 456),))), self.supervisor)
        memory = iter((GPUMemoryObservation("GPU-test", self.supervisor, 1, 2, 1000, 400, 600),
                       GPUMemoryObservation("GPU-test", self.supervisor, 3, 4, 1000, 500, 500),
                       GPUMemoryObservation("GPU-test", self.supervisor, 5, 6, 1000, 600, 400),
                       GPUMemoryObservation("GPU-test", self.supervisor, 7, 8, 1000, 700, 300)))
        self.proof = GPUProof(lambda: "GPU-test", AsyncMock(return_value=True),
            AsyncMock(return_value=ResidencyEvidence("GPU-test", self.supervisor,
                                                     (ProcessIdentity(123, 456),))), self.supervisor,
            memory=AsyncMock(side_effect=lambda: next(memory)))
        self.rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        app = web.Application()
        self.loaded = False
        async def generate(request):
            body = await request.json()
            self.loaded = body.get("keep_alive") != 0
            return web.json_response({"response": "ok", "done": True, "prompt_eval_count": 4})
        async def ps(request):
            return web.json_response({"models": ([{"name": self.name + ":latest", "size": 1, "size_vram": 1}]
                                                if self.loaded else [])})
        app.router.add_post("/api/generate", generate); app.router.add_get("/api/ps", ps)
        self.runner = web.AppRunner(app); await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0); await self.site.start()
        self.port = self.site._server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        if self.rm._session is not None:
            try: await self.rm.stop_session(self.rm._session.session_token, idempotency_key="teardown")
            except ResourceManagerError: pass
        await self.runner.cleanup(); self.tmp.cleanup()

    def _hash(self, model):
        return next(item["sha256"] for item in self.document["models"][model]["files"]
                    if item["path"] == "model.safetensors")

    def adapter(self, model):
        manifest = self.document["manifest_sha256"]
        if model is ModelId.SMOLLM:
            digest = next(item["sha256"] for item in self.document["models"]["SmolLM"]["files"]
                          if item["path"].endswith(".gguf"))
            self.name = "smollm-" + digest[:16]; self.loaded = True
            provider = SmolLMProvider(SmolLMProviderConfig(self.volume, manifest, digest, "GPU-test", "runtime", "adapter",
                request_timeout_seconds=.2, gpu_proof=self.proof, ollama_port=self.port))
            return provider, _profile(model, manifest, digest), b"hello", {"context_size": 512}
        if model is ModelId.COEDIT:
            digest = self._hash("CoEdIT"); config = PythonProviderConfig(self.volume, manifest, digest, "GPU-test", "runtime", "adapter", gpu_proof=self.proof)
            return CoEdITProvider(config), _profile(model, manifest, digest, config.bucket_identity), b'{"instruction":"fix","texts":["text"]}', {"bucket_identity": config.bucket_identity}
        digest = self._hash("GECToR"); config = GECToRProviderConfig(self.volume, manifest, digest, "GPU-test", "runtime", "adapter", gpu_proof=self.proof)
        return GECToRProvider(config), _profile(model, manifest, digest, config.bucket_identity), json.dumps({"texts":["text"], "keep_confidence":0.0, "min_error_prob":0.0, "n_iteration":1, "batch_size":1}).encode(), {"bucket_identity": config.bucket_identity}

    async def test_real_adapter_lifecycles_switch_only_after_cleanup(self):
        """Real constructors/methods run; loopback and worker transport are owned seams."""
        patches = (patch("services.llm.providers.smollm.validate_smollm_input", return_value={}),
                   patch.object(SmolLMProvider, "_run_create", new=AsyncMock()),
                   patch("services.llm.providers.coedit.PythonWorker", _Worker),
                   patch("services.llm.providers.gector.PythonWorker", _Worker))
        with patches[0], patches[1], patches[2], patches[3]:
            old = None
            for index, model in enumerate((ModelId.SMOLLM, ModelId.COEDIT, ModelId.GECTOR)):
                provider, profile, payload, kwargs = self.adapter(model)
                session = await self.rm.start_session("scheduler", model, profile, provider, idempotency_key=f"start-{index}")
                if old is not None: self.assertTrue(await old.verify_cleanup())
                submission = await self.rm.submit(session.session_token, f"r{index}", "a", payload,
                    idempotency_key=f"r{index}", **kwargs)
                self.assertTrue(submission.accepted)
                while self.rm._active:
                    await asyncio.sleep(0)
                finished = [event for event in self.rm._events[session.session_token]
                            if event.kind is EventKind.RESPONSE_FINISHED]
                self.assertEqual(len(finished), 1)
                self.assertEqual((finished[0].request_id, finished[0].attempt), (f"r{index}", "a"))
                self.assertIsInstance(finished[0].result, bytes)
                self.assertEqual(finished[0].completion_sequence, 1)
                self.assertIsNone(finished[0].time_on_gpu_ms)
                self.assertFalse(finished[0].gpu_timing_complete)
                old = provider

    async def test_p2_plus_p2_retry_and_stale_replacement_controls(self):
        provider = _BlockingProvider(); manifest = "m"; model = "d"
        session = await self.rm.start_session("s", ModelId.SMOLLM, _profile(ModelId.SMOLLM, manifest, model, p=2), provider, idempotency_key="old")
        accepted = await asyncio.gather(*(self.rm.submit(session.session_token, f"r{i}", "a", b"x", idempotency_key=f"k{i}", context_size=512) for i in range(4)))
        self.assertTrue(all(item.accepted for item in accepted)); self.assertEqual(len(self.rm._active), 2); self.assertEqual(len(self.rm._buffer), 2)
        before = len(provider.calls)
        self.assertEqual(accepted[0], await self.rm.submit(session.session_token, "r0", "a", b"x", idempotency_key="k0", context_size=512))
        self.assertEqual(before, len(provider.calls))
        pressure = await self.rm.submit(session.session_token, "retry", "a", b"x", idempotency_key="retry", context_size=512)
        self.assertTrue(pressure.backpressure); self.assertNotIn(("execute", "retry"), provider.calls)
        provider.release.set()
        await asyncio.wait_for(provider.started.wait(), .2)
        while self.rm._active or self.rm._buffer:
            await asyncio.sleep(0)
        retry = await self.rm.submit(session.session_token, "retry", "a", b"x", idempotency_key="retry", context_size=512)
        self.assertTrue(retry.accepted)
        await asyncio.sleep(.02)
        provider.unload_gate = asyncio.Event()
        replacement_task = asyncio.create_task(self.rm.start_session("new", ModelId.COEDIT, _profile(ModelId.COEDIT, manifest, model, "bucket"), _BlockingProvider(), idempotency_key="new"))
        try:
            await asyncio.wait_for(provider.unload_started.wait(), .2)
            with self.assertRaises(ResourceManagerError) as stale:
                await asyncio.wait_for(self.rm.start_session("s", ModelId.SMOLLM, _profile(ModelId.SMOLLM, manifest, model, p=2), provider, idempotency_key="old"), .2)
            self.assertEqual(stale.exception.failure.code, "scheduler_superseded")
            for operation in (self.rm.submit(session.session_token, "old", "a", b"x", idempotency_key="old-submit", context_size=512),
                              self.rm.cancel_request(session.session_token, "old", idempotency_key="old-cancel"),
                              self.rm.get_capacity(session.session_token)):
                with self.assertRaises(ResourceManagerError) as rejected:
                    await asyncio.wait_for(operation, .2)
                self.assertEqual(rejected.exception.failure.code, "scheduler_superseded")
        finally:
            provider.unload_gate.set()
            await asyncio.wait_for(replacement_task, .2)

    async def test_failed_cleanup_fences_old_controls_without_replacement_authority(self):
        provider = _BlockingProvider(cleanup_ok=False)
        session = await self.rm.start_session("s", ModelId.SMOLLM,
            _profile(ModelId.SMOLLM, "m", "d"), provider, idempotency_key="old")
        replacement = asyncio.create_task(self.rm.start_session(
            "new", ModelId.COEDIT, _profile(ModelId.COEDIT, "m", "d", "bucket"),
            _BlockingProvider(), idempotency_key="new"))
        with self.assertRaises(ResourceManagerError) as failed:
            await replacement
        self.assertEqual(failed.exception.failure.code, "cleanup_failed")
        self.assertIsNone(self.rm._session)
        for operation in (
            self.rm.submit(session.session_token, "late", "a", b"x",
                           idempotency_key="late", context_size=512),
            self.rm.cancel_request(session.session_token, "late", idempotency_key="late-cancel"),
            self.rm.get_capacity(session.session_token),
        ):
            with self.assertRaises(ResourceManagerError) as rejected:
                await operation
            self.assertEqual(rejected.exception.failure.code, "scheduler_superseded")
        with self.assertRaises(ResourceManagerError) as no_authority:
            await self.rm.start_session("newer", ModelId.SMOLLM,
                _profile(ModelId.SMOLLM, "m", "d"), _BlockingProvider(),
                idempotency_key="newer")
        self.assertEqual(no_authority.exception.failure.code, "resource_manager_unavailable")

    async def test_actual_adapter_late_completion_after_cancel_is_not_published(self):
        class LateWorker(_Worker):
            started = asyncio.Event()
            release = asyncio.Event()
            completed = asyncio.Event()

            async def call(self, operation, **kwargs):
                if operation == "execute_batch":
                    self.__class__.started.set()
                    await self.__class__.release.wait()
                    try:
                        return await super().call(operation, **kwargs)
                    finally:
                        self.__class__.completed.set()
                return await super().call(operation, **kwargs)

        provider, profile, payload, kwargs = self.adapter(ModelId.COEDIT)
        LateWorker.started = asyncio.Event()
        LateWorker.release = asyncio.Event()
        LateWorker.completed = asyncio.Event()
        with patch("services.llm.providers.coedit.PythonWorker", LateWorker):
            session = await self.rm.start_session("s", ModelId.COEDIT, profile, provider,
                                                  idempotency_key="adapter")
            await self.rm.submit(session.session_token, "request", "a", payload,
                                 idempotency_key="request", **kwargs)
            await asyncio.wait_for(LateWorker.started.wait(), .2)
            self.assertTrue(await self.rm.cancel_request(session.session_token, "request",
                                                         idempotency_key="cancel"))
            LateWorker.release.set()
            await asyncio.wait_for(LateWorker.completed.wait(), .2)
            while self.rm._active:
                await asyncio.sleep(0)
            await asyncio.sleep(0)
        events = self.rm._events[session.session_token]
        # ResourceManager's cancellation contract suppresses publication.  The
        # adapter completion is nevertheless collected above, and cancellation
        # remains truthfully observable as the terminal lifecycle event.
        self.assertFalse(any(event.kind is EventKind.RESPONSE_FINISHED for event in events))
        cancelled = [event for event in events if event.kind is EventKind.CANCELLED]
        self.assertEqual(len(cancelled), 1)
        self.assertEqual((cancelled[0].request_id, cancelled[0].attempt), ("request", "a"))

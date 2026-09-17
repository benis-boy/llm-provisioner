"""Focused loopback regressions for the supervisor-owned SmolLM adapter."""
from __future__ import annotations

import hashlib
import asyncio
import gc
import os
from pathlib import Path
import tempfile
import sys
import textwrap
import unittest
from dataclasses import replace
from unittest import mock

from aiohttp import web

from services.llm.provisioning.volume import provision
from services.llm.providers.config import GPUProof, SmolLMProviderConfig
from services.llm.providers.gpu import ProcessIdentity, ResidencyEvidence
from services.llm.providers.smollm import SmolLMProvider
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager, ResourceManagerError


def _profile(manifest: str, model: str) -> CapacityProfile:
    return CapacityProfile(ModelId.SMOLLM, "GPU-1", manifest, model, "runtime", "adapter",
        "measured", 1, 1, 1, 20, (SampleMetadata(1, 1, 1, 1, 1, (1,)),), 512)


def _gpu_proof(*, cleanup=True) -> GPUProof:
    supervisor = ProcessIdentity(10, 42)
    return GPUProof(lambda: "GPU-1", lambda: cleanup,
        lambda: ResidencyEvidence("GPU-1", supervisor, (ProcessIdentity(11, 43),)),
        expected_supervisor=supervisor)


class SmolLMProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.unloaded = False
        self.name = "smollm-test"
        self.mode = None
        self.unload_attempts = 0
        self.foreign = False
        self.ps_calls = 0
        self.ps_models = None
        self.block_started = asyncio.Event()
        self.block_release = asyncio.Event()
        async def generate(request):
            if self.mode == "redirect":
                raise web.HTTPFound("/api/generate-target")
            body = await request.json(); self.requests.append(body)
            if self.mode == "blocking":
                self.block_started.set()
                await self.block_release.wait()
            if body.get("keep_alive") == 0:
                self.unload_attempts += 1
                if self.mode == "failed-unload" and self.unload_attempts == 1:
                    return web.Response(status=500)
                self.unloaded = True
            else:
                self.unloaded = False
            if self.mode == "malformed":
                return web.Response(text="{not-json", content_type="application/json")
            if self.mode == "oversize":
                return web.Response(body=b"{" + b"x" * (128 * 1024) + b"}",
                                    content_type="application/json")
            if self.mode == "chunked":
                response = web.StreamResponse(status=200, headers={"Content-Type": "application/json"})
                await response.prepare(request)
                await response.write(b'{"response":"o')
                await response.write(b'k","done":true,"prompt_eval_count":4}')
                await response.write_eof()
                return response
            if self.mode == "done-false":
                return web.json_response({"response": "", "done": False})
            count = 0 if self.mode == "bad-p-count" else 4
            return web.json_response({"response": "ok", "done": True, "prompt_eval_count": count})
        async def ps(request):
            self.ps_calls += 1
            if self.ps_models is not None:
                models = self.ps_models[0]
                if len(self.ps_models) > 1:
                    self.ps_models.pop(0)
                return web.json_response({"models": models})
            if self.foreign:
                return web.json_response({"models": [{"name": "foreign:latest", "size": 1, "size_vram": 1}]})
            if self.unloaded:
                return web.json_response({"models": []})
            return web.json_response({"models": [{"name": self.name + ":latest", "size": 2, "size_vram": 2}]})
        self.app = web.Application()
        self.app.router.add_post("/api/generate", generate)
        self.app.router.add_get("/api/ps", ps)
        self.runner = web.AppRunner(self.app); await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0); await self.site.start()
        self.port = self.site._server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_happy_raw_wire_lifecycle_and_strict_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, volume = Path(tmp) / "source", Path(tmp) / "volume"
            source.mkdir()
            gguf = source / "SmolLM2-1.7B-Instruct-Q8_0.gguf"; gguf.write_bytes(b"tiny-gguf")
            (source / "Modelfile").write_text("ignored", encoding="ascii")
            document = provision({"SmolLM": source}, volume)
            digest = hashlib.sha256(b"tiny-gguf").hexdigest()
            config = SmolLMProviderConfig(volume, document["manifest_sha256"], digest, "GPU-1", "runtime", "adapter",
                request_timeout_seconds=.05,
                 gpu_proof=_gpu_proof(), ollama_port=self.port)
            provider = SmolLMProvider(config); self.name = "smollm-" + digest[:16]
            profile = _profile(document["manifest_sha256"], digest)
            # The GGUF byte-BPE proof is independently tested; this test owns a
            # deliberately tiny provisioned artifact and exercises adapter I/O.
            with mock.patch("services.llm.providers.smollm.validate_smollm_input", return_value={}), \
                 mock.patch.object(provider, "_run_create") as create:
                await provider.load(profile)
                create.assert_awaited_once()
                await provider.ready()
                response = await provider.execute("request", b"hello")
            self.assertEqual(response.result, b"ok")
            self.assertEqual(self.requests[-1]["prompt"], "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n")
            self.assertEqual(self.requests[-1]["options"], {"num_ctx": 512, "num_predict": 64, "temperature": 0})
            await provider.unload()
            self.assertTrue(await provider.verify_cleanup())

    async def test_identity_mismatch_fails_before_local_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = SmolLMProviderConfig(root, "0" * 64, "1" * 64, "GPU-1", "runtime", "adapter",
                gpu_proof=GPUProof(lambda: "GPU-1", lambda: True))
            provider = SmolLMProvider(config)
            with mock.patch("services.llm.providers.smollm.verify_current", return_value={"manifestSha256": "1" * 64}), \
                 mock.patch.object(provider, "_run_create") as create:
                with self.assertRaises(ValueError):
                    await provider.load(_profile("0" * 64, "1" * 64))
            create.assert_not_awaited()

    async def test_cleanup_verification_is_model_state_only_and_does_not_probe_gpu(self):
        cleanup = mock.Mock(return_value=False)
        config = SmolLMProviderConfig(Path("/tmp"), "0" * 64, "1" * 64, "GPU-1", "runtime", "adapter",
            gpu_proof=GPUProof(lambda: "GPU-1", cleanup))
        provider = SmolLMProvider(config)
        provider._cleanup_verified = True
        self.assertTrue(await provider.verify_cleanup())
        cleanup.assert_not_called()
        provider._model = "stale"
        self.assertFalse(await provider.verify_cleanup())

    async def test_model_state_cleanup_proves_absence_and_rejects_foreign_state(self):
        config = SmolLMProviderConfig(Path("/tmp"), "0" * 64, "1" * 64, "GPU-1", "runtime", "adapter",
            gpu_proof=_gpu_proof(), ollama_port=self.port)
        provider = SmolLMProvider(config)
        self.unloaded = True
        await provider.unload()
        self.assertGreaterEqual(self.ps_calls, 1)
        self.assertTrue(await provider.verify_cleanup())

        foreign = SmolLMProvider(replace(config, gpu_proof=_gpu_proof()))
        self.foreign = True
        with self.assertRaises(RuntimeError): await foreign.unload()
        self.foreign = False
        false_cleanup = mock.Mock(return_value=False)
        false_probe = SmolLMProvider(replace(
            config, gpu_proof=GPUProof(lambda: "GPU-1", false_cleanup)))
        await false_probe.unload()
        self.assertTrue(await false_probe.verify_cleanup())
        false_cleanup.assert_not_called()

    async def test_absence_check_returns_without_sleep_for_empty_models(self):
        provider, tmp = await self._loaded_provider()
        try:
            self.ps_models = [[]]
            sleeps = []
            provider._absence_sleep = lambda delay: sleeps.append(delay)
            await provider._check_absent()
            self.assertEqual([], sleeps)
        finally:
            self.ps_models = None
            await provider.unload()

    async def test_absence_check_polls_until_models_are_gone(self):
        provider, tmp = await self._loaded_provider()
        try:
            resident = [{"name": "smollm-test:latest", "size": 2, "size_vram": 2}]
            self.ps_models = [resident, []]
            now = [0.0]
            sleeps = []
            provider._absence_clock = lambda: now[0]
            async def sleep(delay):
                sleeps.append(delay)
                now[0] += delay
            provider._absence_sleep = sleep
            await provider._check_absent()
            self.assertEqual([0.2], sleeps)
            self.assertEqual(2, self.ps_calls)
        finally:
            self.ps_models = None
            await provider.unload()

    async def test_absence_check_bounded_failure_for_persistent_models(self):
        provider, tmp = await self._loaded_provider()
        try:
            resident = [{"name": "smollm-test:latest", "size": 2, "size_vram": 2}]
            self.ps_models = [resident]
            now = [0.0]
            sleeps = []
            provider._absence_clock = lambda: now[0]
            async def sleep(delay):
                sleeps.append(delay)
                now[0] += delay
            provider._absence_sleep = sleep
            with self.assertRaisesRegex(RuntimeError, "Ollama still has resident models"):
                await provider._check_absent()
            self.assertEqual(5.0, now[0])
            self.assertEqual(25, len(sleeps))
        finally:
            # The test intentionally leaves the daemon resident; make the
            # provider state safe for the fixture teardown without polling.
            await provider._session.close()
            provider._session = None

    async def test_readiness_requires_exact_typed_supervisor_owned_residency(self):
        provider, tmp = await self._loaded_provider()
        try:
            supervisor = ProcessIdentity(10, 42)
            cases = (
                ("missing expected supervisor", replace(provider.config, gpu_proof=GPUProof(
                    lambda: "GPU-1", lambda: True,
                    lambda: ResidencyEvidence("GPU-1", supervisor, (ProcessIdentity(11, 43),))))),
                ("missing residency", replace(provider.config, gpu_proof=GPUProof(
                    lambda: "GPU-1", lambda: True, expected_supervisor=supervisor))),
                ("wrong uuid", replace(provider.config, gpu_proof=GPUProof(
                    lambda: "GPU-1", lambda: True,
                    lambda: ResidencyEvidence("GPU-2", supervisor, (ProcessIdentity(11, 43),)), supervisor))),
                ("wrong supervisor", replace(provider.config, gpu_proof=GPUProof(
                    lambda: "GPU-1", lambda: True,
                    lambda: ResidencyEvidence("GPU-1", ProcessIdentity(12, 42), (ProcessIdentity(11, 43),)), supervisor))),
                ("empty runners", replace(provider.config, gpu_proof=GPUProof(
                    lambda: "GPU-1", lambda: True,
                    lambda: ResidencyEvidence("GPU-1", supervisor, ()), supervisor))),
                ("duplicate runner pid", replace(provider.config, gpu_proof=GPUProof(
                    lambda: "GPU-1", lambda: True,
                    lambda: ResidencyEvidence("GPU-1", supervisor, (ProcessIdentity(11, 43), ProcessIdentity(11, 44))), supervisor))),
                ("supervisor pid runner alias", replace(provider.config, gpu_proof=GPUProof(
                    lambda: "GPU-1", lambda: True,
                    lambda: ResidencyEvidence("GPU-1", supervisor, (ProcessIdentity(10, 99),)), supervisor))),
            )
            for label, config in cases:
                with self.subTest(label=label):
                    candidate = SmolLMProvider(config)
                    candidate._session, candidate._model = provider._session, provider._model
                    with self.assertRaises(RuntimeError): await candidate.ready()
                    self.assertFalse(candidate._ready)
        finally:
            await provider.unload()

    async def test_repeated_ready_rechecks_proof_and_blocks_execute(self):
        provider, tmp = await self._loaded_provider()
        try:
            await provider.ready()
            good = provider.config.gpu_proof
            bad = replace(good, residency=lambda: ResidencyEvidence(
                "GPU-1", ProcessIdentity(99, 42), (ProcessIdentity(11, 43),)))
            provider.config = replace(provider.config, gpu_proof=bad)
            with self.assertRaises(RuntimeError): await provider.ready()
            with self.assertRaises(RuntimeError): await provider.execute("after-bad-ready", b"hello")
        finally:
            await provider.unload()

    async def test_resource_manager_validation_failure_proves_absence_before_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, volume = Path(tmp) / "source", Path(tmp) / "volume"
            source.mkdir()
            gguf = source / "SmolLM2-1.7B-Instruct-Q8_0.gguf"; gguf.write_bytes(b"tiny-gguf")
            (source / "Modelfile").write_text("ignored", encoding="ascii")
            document = provision({"SmolLM": source}, volume)
            digest = hashlib.sha256(b"tiny-gguf").hexdigest()
            config = SmolLMProviderConfig(volume, document["manifest_sha256"], digest, "GPU-1", "runtime", "adapter",
                request_timeout_seconds=.2, gpu_proof=_gpu_proof(), ollama_port=self.port)
            provider = SmolLMProvider(config)
            good = _profile(document["manifest_sha256"], digest)
            # Validation fails before acquisition; cleanup must nevertheless
            # query actual daemon absence before admitting a good retry.
            self.unloaded = True
            rm = ResourceManager(cleanup_timeout=.2, load_timeout=.2)
            original_validate = provider.validate
            failed_once = True
            async def fail_once(profile):
                nonlocal failed_once
                if failed_once:
                    failed_once = False
                    raise ValueError("synthetic validation failure")
                await original_validate(profile)
            with mock.patch.object(provider, "validate", side_effect=fail_once):
                with self.assertRaises(ResourceManagerError):
                    await rm.start_session("bad", ModelId.SMOLLM, good, provider, idempotency_key="bad")
            self.assertGreaterEqual(self.ps_calls, 1)
            with mock.patch.object(provider, "_run_create"), \
                 mock.patch("services.llm.providers.smollm.validate_smollm_input", return_value={}):
                self.name = "smollm-" + digest[:16]
                self.unloaded = False
                session = await rm.start_session("good", ModelId.SMOLLM, good, provider, idempotency_key="good")
            self.assertEqual(session.scheduler_id, "good")
            await rm.stop_session(session.session_token, idempotency_key="stop")

    async def _loaded_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, volume = Path(tmp) / "source", Path(tmp) / "volume"
            source.mkdir()
            gguf = source / "SmolLM2-1.7B-Instruct-Q8_0.gguf"; gguf.write_bytes(b"tiny-gguf")
            (source / "Modelfile").write_text("ignored", encoding="ascii")
            document = provision({"SmolLM": source}, volume)
            digest = hashlib.sha256(b"tiny-gguf").hexdigest()
            config = SmolLMProviderConfig(volume, document["manifest_sha256"], digest, "GPU-1", "runtime", "adapter",
                request_timeout_seconds=.2,
                 gpu_proof=_gpu_proof(), ollama_port=self.port)
            provider = SmolLMProvider(config); self.name = "smollm-" + digest[:16]
            with mock.patch("services.llm.providers.smollm.validate_smollm_input", return_value={}), \
                 mock.patch.object(provider, "_run_create"):
                await provider.load(_profile(document["manifest_sha256"], digest))
            return provider, tmp

    async def test_chunked_json_is_consumed_and_malformed_or_oversize_fails_closed(self):
        provider, tmp = await self._loaded_provider()
        try:
            await provider.ready()
            self.mode = "chunked"
            self.assertEqual((await provider.execute("chunk", b"hello")).result, b"ok")
            self.mode = "malformed"
            with self.assertRaises(RuntimeError): await provider.execute("bad", b"hello")
            self.mode = "oversize"
            with self.assertRaises(RuntimeError): await provider.execute("large", b"hello")
        finally:
            self.mode = None
            await provider.unload()

    async def test_redirect_foreign_residency_and_gpu_identity_fail_closed(self):
        provider, tmp = await self._loaded_provider()
        try:
            self.mode = "redirect"
            with self.assertRaises(RuntimeError): await provider.execute("redirect", b"hello")
            self.mode = None
            self.foreign = True
            with self.assertRaises(RuntimeError): await provider._check_residency()
            self.assertEqual(await provider._gpu(), "GPU-1")
        finally:
            self.foreign = False
            await provider.unload()

    async def test_foreign_api_ps_and_prompt_count_admission_are_rejected(self):
        provider, tmp = await self._loaded_provider()
        try:
            self.mode = "bad-p-count"
            with self.assertRaises(RuntimeError): await provider.execute("count", b"hello")
            self.mode = None
            self.foreign = True
            with self.assertRaises(RuntimeError): await provider.unload()
            self.foreign = False
            await provider.unload()
        finally:
            if provider._session is not None:
                await provider.unload()

    async def test_failed_unload_is_retryable_and_cancel_waits_for_owned_request(self):
        provider, tmp = await self._loaded_provider()
        try:
            self.mode = "failed-unload"
            with self.assertRaises(RuntimeError): await provider.unload()
            self.assertIsNotNone(provider._session)
            await provider.unload()
            self.assertTrue(await provider.verify_cleanup())
        finally:
            if provider._session is not None:
                await provider.unload()

    async def test_live_duplicate_request_id_and_repeated_caller_cancel_retain_ownership(self):
        provider, tmp = await self._loaded_provider()
        try:
            await provider.ready(); self.mode = "blocking"
            first = asyncio.create_task(provider.execute("same", b"hello"))
            await self.block_started.wait()
            with self.assertRaises(RuntimeError):
                await provider.execute("same", b"hello")
            async def release_later():
                await asyncio.sleep(.01)
                self.block_release.set()
            asyncio.create_task(release_later())
            caller = first
            caller.cancel()
            with self.assertRaises(asyncio.CancelledError): await caller
            self.assertNotIn("same", provider._tasks)
            await provider.unload()
        finally:
            self.block_release.set()
            if provider._session is not None:
                await provider.unload()

    async def test_done_false_unload_is_rejected_and_failed_create_can_retry(self):
        provider, tmp = await self._loaded_provider()
        try:
            await provider.ready(); self.mode = "done-false"
            with self.assertRaises(RuntimeError): await provider.unload()
            self.assertIsNotNone(provider._session)
        finally:
            self.mode = None
            if provider._session is not None:
                await provider.unload()

        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "ollama_stub.py"
            script.write_text("#!/usr/bin/env python3\n" + textwrap.dedent("""
                import sys
                if 'fail' in sys.argv:
                    raise SystemExit(7)
                raise SystemExit(0)
            """), encoding="ascii")
            script.chmod(0o755)
            config = SmolLMProviderConfig(Path(root), "0" * 64, "1" * 64, "GPU-1", "runtime", "adapter",
                gpu_proof=_gpu_proof(), ollama_binary=str(script))
            retry = SmolLMProvider(config)
            with mock.patch.object(retry, "_prove_artifact", return_value=({}, Path(root), {})):
                with self.assertRaises(RuntimeError): await retry._run_create("fail", Path(root) / "x.gguf")
                await retry._run_create("ok", Path(root) / "x.gguf")

    async def test_create_cli_receives_configured_home_without_ambient_environment(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            home = root / "private-home"
            observed = root / "home.txt"
            script = root / "ollama_stub.py"
            script.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(f"""
                import os
                from pathlib import Path
                Path({str(observed)!r}).write_text(os.environ.get("HOME", "missing"))
            """), encoding="ascii")
            script.chmod(0o755)
            config = SmolLMProviderConfig(root, "0" * 64, "1" * 64, "GPU-1", "runtime", "adapter",
                gpu_proof=_gpu_proof(), ollama_binary=str(script), ollama_home=home)
            await SmolLMProvider(config)._run_create("ok", root / "x.gguf")
            self.assertEqual(observed.read_text(), str(home))

    async def test_real_cli_timeout_and_output_overflow_are_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "ollama_stub.py"
            script.write_text("#!/usr/bin/env python3\n" + textwrap.dedent("""
                import os, sys, time
                if 'overflow' in sys.argv:
                    os.write(2, b'x' * (4 * 1024 * 1024))
                    raise SystemExit(9)
                child = os.fork()
                if child == 0:
                    time.sleep(30)
                time.sleep(30)
            """), encoding="ascii")
            script.chmod(0o755)
            config = SmolLMProviderConfig(Path(root), "0" * 64, "1" * 64, "GPU-1", "runtime", "adapter",
                 request_timeout_seconds=.05, gpu_proof=_gpu_proof(),
                ollama_binary=str(script))
            provider = SmolLMProvider(config)
            with self.assertRaises(Exception):
                unraisable = []
                previous_hook = sys.unraisablehook
                sys.unraisablehook = unraisable.append
                try:
                    await provider._run_create("timeout", Path(root) / "x.gguf")
                    await asyncio.sleep(0)
                    gc.collect()
                    await asyncio.sleep(0)
                    gc.collect()
                finally:
                    sys.unraisablehook = previous_hook
                self.assertEqual([], unraisable,
                                 [(item.exc_type, str(item.exc_value), item.object)
                                  for item in unraisable])
            with self.assertRaises(RuntimeError):
                unraisable = []
                previous_hook = sys.unraisablehook
                sys.unraisablehook = unraisable.append
                try:
                    await provider._run_create("overflow", Path(root) / "x.gguf")
                    await asyncio.sleep(0)
                    gc.collect()
                    await asyncio.sleep(0)
                    gc.collect()
                finally:
                    sys.unraisablehook = previous_hook
                self.assertEqual([], unraisable,
                                 [(item.exc_type, str(item.exc_value), item.object)
                                  for item in unraisable])

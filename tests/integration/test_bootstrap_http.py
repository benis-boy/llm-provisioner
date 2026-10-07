"""Composed loopback tests for the owned bootstrap runtime.

These deliberately keep the real artifact volume, profile store, HTTP server,
resource manager, and result store.  Only the machine-bound daemon/GPU and the
three model implementations are replaced.
"""

import asyncio
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from aiohttp import ClientSession, web
from services.llm.health import HealthBoundary, register_routes

from services.llm.bootstrap.runtime import BootstrapRuntime, RuntimeOptions
from services.llm.bootstrap.config import ModelConfig
from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import ProcessIdentity
from services.llm.resource_manager.protocol import ProviderResponse
from services.llm.resource_manager.profiles import ProfileStore
from services.llm.queue.contracts import ModelId
from services.llm.bootstrap.bindings import _BUCKETS

from tests.unit.test_bootstrap_bindings import _provisioned_config, _profile


class FakeProvider:
    constructed = 0

    def __init__(self, config):
        type(self).constructed += 1
        self.config = config
        self.loaded = False
        self.unloaded = False

    async def validate(self, profile): pass
    async def load(self, profile): self.loaded = True
    async def ready(self): pass
    async def execute(self, request_id, payload):
        return ProviderResponse(b"exact:" + payload)
    async def cancel(self, request_id): pass
    async def unload(self): self.unloaded = True
    async def verify_cleanup(self): return True
    async def validate_input(self, payload, *, context_size, bucket_identity): pass


class FakeDaemon:
    def __init__(self, config, proof, *, num_parallel=1):
        self.alive_now = True
        self.closed = False
        self.num_parallel = num_parallel
    async def start(self): return "fake-ollama"
    async def health(self): return None
    async def alive(self): return self.alive_now
    async def close(self): self.closed = True


class FakeCapture:
    def __call__(self, gpu_uuid, supervisor_pid, *, host_pid_namespace):
        class Proof:
            supervisor_identity = ProcessIdentity(supervisor_pid, 1)
            residency_for_runner = None
            memory = None
            async def identity(self): return gpu_uuid
            async def cleanup(self): return True
            async def residency(self): raise AssertionError("fake provider must not ask residency")
        return Proof()


def _runtime(root, *, port, free_space=None, capacities=None):
    config, document, hashes, runtime = _provisioned_config(root)
    runtime = dict(runtime)
    runtime["SmolLM"] = "ollama:fake-ollama"
    config = replace(config, models={name: ModelConfig(runtime[name], config.models[name].adapter_identity)
                                    for name in runtime})
    with ProfileStore(config.profile_db) as store:
        for model in ModelId:
            name = model.value
            profile, metadata = _profile(model, config.manifest_sha256, hashes[name], runtime[name],
                                        f"adapter-{name}", context=512 if model is ModelId.SMOLLM else None,
                                        bucket=None if model is ModelId.SMOLLM else _BUCKETS[model],
                                        optimum=(capacities or {}).get(model, 1))
            store.save_measured(profile, metadata)
    proof_capture = FakeCapture()
    options = RuntimeOptions(Path(root) / "state", Path(root) / "results", port=port,
                             sqlite_min_free_bytes=free_space if free_space is not None else 0,
                             host_pid_namespace=True)
    return config, options, proof_capture, runtime


async def _request(session, method, url, **kwargs):
    async with session.request(method, url, **kwargs) as response:
        return response.status, await response.json()


class BootstrapHttpTests(unittest.TestCase):
    def _patch_runtime(self, identities):
        return patch("services.llm.bootstrap.runtime.observe_runtime_identities",
                     return_value=identities), \
               patch("services.llm.bootstrap.bindings.SmolLMProvider", FakeProvider), \
               patch("services.llm.bootstrap.bindings.CoEdITProvider", FakeProvider), \
               patch("services.llm.bootstrap.bindings.GECToRProvider", FakeProvider)

    def test_health_routes_share_an_existing_resource_manager_application(self):
        app = web.Application()
        async def resource_manager(_request):
            return web.Response()
        app.router.add_get("/v1/resource-manager", resource_manager)
        boundary = HealthBoundary({"phase": "startup", "dependencies": {
            "sqlite": True, "gpu": True, "artifacts": True, "adapter": True,
            "ollama": True, "profile": True, "cleanup": True,
        }})
        self.assertIs(register_routes(app, boundary), app)
        self.assertEqual({route.resource.canonical for route in app.router.routes()}, {
            "/v1/resource-manager", "/health/live", "/health/ready", "/health/dependencies"})

    def test_real_start_submit_readiness_and_exact_result(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                config, options, capture, identities = _runtime(Path(directory), port=18991)
                # RuntimeOptions validates a concrete port; reserve an unused
                # deterministic test port only for this short-lived loopback.
                options = RuntimeOptions(options.state_dir, options.result_dir, port=18991,
                                         sqlite_min_free_bytes=0, host_pid_namespace=True)
                FakeProvider.constructed = 0
                with patch("services.llm.bootstrap.runtime.observe_runtime_identities",
                           return_value=identities), \
                     patch("services.llm.bootstrap.bindings.SmolLMProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.CoEdITProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.GECToRProvider", FakeProvider):
                    runtime = BootstrapRuntime(config, options, proof_capture=capture,
                                               daemon_factory=FakeDaemon)
                    await runtime.start()
                    try:
                        async with ClientSession() as session:
                            base = "http://127.0.0.1:18991"
                            status, value = await _request(session, "GET", base + "/health/live")
                            self.assertEqual((status, value), (200, {"live": True}))
                            status, _ = await _request(session, "GET", base + "/health/ready")
                            self.assertEqual(status, 503)
                            status, value = await _request(session, "POST", base + "/resource-manager/sessions",
                                json={"schedulerId":"s", "modelId":"SmolLM", "contextSizeEstimate":512},
                                headers={"Idempotency-Key":"start-1"})
                            self.assertEqual(status, 201)
                            token = value["sessionToken"]
                            status, _ = await _request(session, "GET", base + "/health/ready")
                            self.assertEqual(status, 200)
                            digest = await asyncio.to_thread(runtime.result_store.write, b"hello")
                            status, submitted = await _request(session, "POST", base + f"/resource-manager/sessions/{token}/submissions",
                                json={"requestId":"r", "attemptToken":"a", "inputReference":"sha256:" + digest},
                                headers={"Idempotency-Key":"submit-1"})
                            self.assertEqual(status, 202)
                            self.assertTrue(submitted["accepted"])
                            async with session.get(base + f"/resource-manager/sessions/{token}/watch",
                                                    timeout=5) as response:
                                self.assertEqual(response.status, 200)
                                frame = bytearray()
                                async for chunk in response.content.iter_chunked(4096):
                                    frame.extend(chunk)
                                    if b'"kind":"response_finished"' in frame:
                                        break
                                self.assertIn(b'"kind":"response_finished"', frame)
                                self.assertIn(b"ZXhhY3Q6aGVsbG8=", frame)
                    finally:
                        await runtime.stop()

        # The watch endpoint is streaming; the HTTP assertions above stop
        # before consuming it.  The submission itself proves admission, while
        # the provider contract is covered by the focused RM watch tests.
        asyncio.run(scenario())

    def test_measured_capacity_drives_daemon_provider_and_http_admission(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                capacities = {ModelId.SMOLLM: 2, ModelId.COEDIT: 2, ModelId.GECTOR: 1}
                config, options, capture, identities = _runtime(
                    Path(directory), port=19001, capacities=capacities)
                with patch("services.llm.bootstrap.runtime.observe_runtime_identities",
                           return_value=identities), \
                     patch("services.llm.bootstrap.bindings.SmolLMProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.CoEdITProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.GECToRProvider", FakeProvider):
                    runtime = BootstrapRuntime(config, options, proof_capture=capture,
                                               daemon_factory=FakeDaemon)
                    await runtime.start()
                    try:
                        self.assertEqual(runtime.daemon.num_parallel, 2)
                        self.assertEqual(runtime.prepared.bindings[ModelId.SMOLLM].provider.config.parallelism, 2)
                        coedit = runtime.prepared.bindings[ModelId.COEDIT].provider.config
                        self.assertEqual(coedit.max_native_batch_size, 2)
                        self.assertEqual(coedit.bucket_identity, _BUCKETS[ModelId.COEDIT])
                        self.assertIsNone(coedit.measurement_max_native_batch_size)
                        async with ClientSession() as session:
                            base = "http://127.0.0.1:19001"
                            status, value = await _request(session, "POST", base + "/resource-manager/sessions",
                                json={"schedulerId":"s", "modelId":"SmolLM", "contextSizeEstimate":512},
                                headers={"Idempotency-Key":"measured-p2"})
                            self.assertEqual(status, 201)
                            token = value["sessionToken"]
                            status, capacity = await _request(session, "GET", base + f"/resource-manager/sessions/{token}/capacity")
                            self.assertEqual(status, 200)
                            self.assertEqual(capacity["profile"]["optimalParallelism"], 2)
                            self.assertEqual(capacity["executionSlots"], 2)
                            self.assertEqual(capacity["bufferSlots"], 2)
                            status, coedit_session = await _request(
                                session, "POST", base + "/resource-manager/sessions",
                                json={"schedulerId":"coedit-s", "modelId":"CoEdIT",
                                      "bucketIdentity":_BUCKETS[ModelId.COEDIT]},
                                headers={"Idempotency-Key":"coedit-measured-p2"})
                            self.assertEqual(status, 201)
                            status, coedit_capacity = await _request(
                                session, "GET", base + f"/resource-manager/sessions/{coedit_session['sessionToken']}/capacity")
                            self.assertEqual(status, 200)
                            self.assertEqual(coedit_capacity["executionSlots"], 2)
                            self.assertEqual(coedit_capacity["bufferSlots"], 2)
                    finally:
                        await runtime.stop()
        asyncio.run(scenario())

    def test_artifact_profile_and_free_space_gate_admission(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                config, options, capture, identities = _runtime(Path(directory), port=18992, free_space=1)
                with patch("services.llm.bootstrap.runtime.observe_runtime_identities", return_value=identities), \
                     patch("services.llm.bootstrap.bindings.SmolLMProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.CoEdITProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.GECToRProvider", FakeProvider):
                    runtime = BootstrapRuntime(config, options, proof_capture=capture, daemon_factory=FakeDaemon)
                    await runtime.start()
                    try:
                        async with ClientSession() as session:
                            url = "http://127.0.0.1:18992/resource-manager/sessions"
                            config.artifact_root.joinpath("current").unlink()
                            config.artifact_root.joinpath("current").symlink_to("wrong")
                            status, _ = await _request(session, "POST", url, json={"schedulerId":"s", "modelId":"SmolLM", "contextSizeEstimate":512}, headers={"Idempotency-Key":"a"})
                            self.assertEqual(status, 503)
                            config.artifact_root.joinpath("current").unlink()
                            config.artifact_root.joinpath("current").symlink_to(config.manifest_sha256)
                            with patch("services.llm.bootstrap.runtime.shutil.disk_usage", return_value=type("U", (), {"free": 0})()):
                                status, _ = await _request(session, "POST", url, json={"schedulerId":"s", "modelId":"SmolLM", "contextSizeEstimate":512}, headers={"Idempotency-Key":"b"})
                                self.assertEqual(status, 503)
                    finally:
                        await runtime.stop()
        asyncio.run(scenario())

    def test_normal_daemon_exit_fences_and_stops_runtime(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                config, options, capture, identities = _runtime(Path(directory), port=18993)
                daemon = None
                class ExitingDaemon(FakeDaemon):
                    def __init__(self, config, proof, *, num_parallel=1):
                        super().__init__(config, proof, num_parallel=num_parallel)
                        nonlocal daemon
                        daemon = self
                    async def alive(self): return False
                with patch("services.llm.bootstrap.runtime.observe_runtime_identities", return_value=identities), \
                     patch("services.llm.bootstrap.bindings.SmolLMProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.CoEdITProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.GECToRProvider", FakeProvider):
                    runtime = BootstrapRuntime(config, options, proof_capture=capture,
                                               daemon_factory=ExitingDaemon)
                    await runtime.start()
                    await runtime._monitor
                    with self.assertRaises(Exception):
                        await runtime.core.start_session("s", ModelId.SMOLLM, runtime.prepared.profiles[ModelId.SMOLLM],
                                                        runtime.prepared.bindings[ModelId.SMOLLM].provider,
                                                        idempotency_key="after-loss")
                    await runtime.stop()
                    self.assertTrue(runtime._fenced)
                    self.assertTrue(daemon.closed)
                    self.assertFalse(runtime.core.snapshot().available)
        asyncio.run(scenario())

    def test_concurrent_start_stop_cancels_blocked_start_without_deadlock(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                config, options, capture, identities = _runtime(Path(directory), port=18998)
                entered = asyncio.Event()
                release = asyncio.Event()
                class BlockedDaemon(FakeDaemon):
                    async def start(self):
                        entered.set()
                        await release.wait()
                        return "late-daemon"
                with patch("services.llm.bootstrap.runtime.observe_runtime_identities", return_value=identities), \
                     patch("services.llm.bootstrap.bindings.SmolLMProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.CoEdITProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.GECToRProvider", FakeProvider):
                    runtime = BootstrapRuntime(config, options, proof_capture=capture, daemon_factory=BlockedDaemon)
                    starting = asyncio.create_task(runtime.start())
                    await entered.wait()
                    stopping = asyncio.create_task(runtime.stop())
                    with self.assertRaises(asyncio.CancelledError):
                        await starting
                    await stopping
                    self.assertIsNone(runtime.prepared)
                    self.assertFalse(runtime.core.snapshot().available)
        asyncio.run(scenario())

    def test_cancelled_stop_callers_share_blocked_daemon_close_and_join(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                config, options, capture, identities = _runtime(Path(directory), port=18999)
                entered = asyncio.Event()
                release = asyncio.Event()
                class BlockedCloseDaemon(FakeDaemon):
                    async def close(self):
                        entered.set()
                        await release.wait()
                        self.closed = True
                with patch("services.llm.bootstrap.runtime.observe_runtime_identities", return_value=identities), \
                     patch("services.llm.bootstrap.bindings.SmolLMProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.CoEdITProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.GECToRProvider", FakeProvider):
                    runtime = BootstrapRuntime(config, options, proof_capture=capture, daemon_factory=BlockedCloseDaemon)
                    await runtime.start()
                    callers = [asyncio.create_task(runtime.stop()) for _ in range(3)]
                    await entered.wait()
                    for caller in callers:
                        caller.cancel()
                    # A cancelled caller still joins the shared cleanup before
                    # re-raising cancellation.  Releasing close must therefore
                    # precede joining those callers.
                    await asyncio.sleep(0)
                    self.assertTrue(all(not caller.done() for caller in callers))
                    release.set()
                    results = await asyncio.gather(*callers, return_exceptions=True)
                    await asyncio.shield(runtime._stop_task)
                    self.assertTrue(all(isinstance(result, asyncio.CancelledError) for result in results))
                    self.assertTrue(runtime.daemon.closed)
        asyncio.run(scenario())

    def test_cleanup_grace_timeout_retains_close_task_until_release(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                config, options, capture, identities = _runtime(Path(directory), port=19000)
                options = RuntimeOptions(options.state_dir, options.result_dir, port=19000,
                                         sqlite_min_free_bytes=0, shutdown_grace_seconds=.02,
                                         host_pid_namespace=True)
                entered = asyncio.Event()
                release = asyncio.Event()
                class SlowCloseDaemon(FakeDaemon):
                    async def close(self):
                        entered.set()
                        await release.wait()
                        self.closed = True
                with patch("services.llm.bootstrap.runtime.observe_runtime_identities", return_value=identities), \
                     patch("services.llm.bootstrap.bindings.SmolLMProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.CoEdITProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.GECToRProvider", FakeProvider):
                    runtime = BootstrapRuntime(config, options, proof_capture=capture, daemon_factory=SlowCloseDaemon)
                    await runtime.start()
                    stopping = asyncio.create_task(runtime.stop())
                    await entered.wait()
                    with self.assertRaises(RuntimeError):
                        await stopping
                    self.assertTrue(runtime._retained)
                    release.set()
                    await asyncio.gather(*tuple(runtime._retained), return_exceptions=True)
                    self.assertTrue(runtime.daemon.closed)
        asyncio.run(scenario())

    def test_repeated_cancel_stop_is_shared_and_cleanup_stage_failure_does_not_skip_closure(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                config, options, capture, identities = _runtime(Path(directory), port=18994)
                with patch("services.llm.bootstrap.runtime.observe_runtime_identities", return_value=identities), \
                     patch("services.llm.bootstrap.bindings.SmolLMProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.CoEdITProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.GECToRProvider", FakeProvider):
                    runtime = BootstrapRuntime(config, options, proof_capture=capture, daemon_factory=FakeDaemon)
                    await runtime.start()
                    daemon = runtime.daemon
                    original_stop = type(runtime.site).stop
                    async def bad_stop(site):
                        await original_stop(site)
                        raise RuntimeError("site cleanup failure")
                    with patch.object(type(runtime.site), "stop", bad_stop):
                        calls = [asyncio.create_task(runtime.stop()) for _ in range(4)]
                        results = await asyncio.gather(*calls, return_exceptions=True)
                    self.assertTrue(all(isinstance(result, RuntimeError) for result in results))
                    self.assertTrue(daemon.closed)
                    self.assertIsNone(runtime.prepared)
        asyncio.run(scenario())

    def test_profile_database_mutation_fails_readiness_and_session_admission(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                config, options, capture, identities = _runtime(Path(directory), port=18995)
                with patch("services.llm.bootstrap.runtime.observe_runtime_identities", return_value=identities), \
                     patch("services.llm.bootstrap.bindings.SmolLMProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.CoEdITProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.GECToRProvider", FakeProvider):
                    runtime = BootstrapRuntime(config, options, proof_capture=capture, daemon_factory=FakeDaemon)
                    await runtime.start()
                    try:
                        with sqlite3.connect(config.profile_db) as db:
                            db.execute("UPDATE profiles SET runtime_identity='mutated' WHERE model_id='SmolLM'")
                            db.commit()
                        async with ClientSession() as session:
                            base = "http://127.0.0.1:18995"
                            status, _ = await _request(session, "GET", base + "/health/ready")
                            self.assertEqual(status, 503)
                            status, _ = await _request(session, "POST", base + "/resource-manager/sessions",
                                json={"schedulerId":"s", "modelId":"SmolLM", "contextSizeEstimate":512},
                                headers={"Idempotency-Key":"mutated"})
                            self.assertEqual(status, 503)
                    finally:
                        await runtime.stop()
        asyncio.run(scenario())

    def test_repeated_health_does_not_reprepare_hash_or_construct_providers(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                config, options, capture, identities = _runtime(Path(directory), port=18996)
                FakeProvider.constructed = 0
                with patch("services.llm.bootstrap.runtime.observe_runtime_identities", return_value=identities), \
                     patch("services.llm.bootstrap.bindings.SmolLMProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.CoEdITProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.GECToRProvider", FakeProvider):
                    runtime = BootstrapRuntime(config, options, proof_capture=capture, daemon_factory=FakeDaemon)
                    await runtime.start()
                    try:
                        constructed = FakeProvider.constructed
                        with patch("services.llm.bootstrap.runtime.prepare_bindings", side_effect=AssertionError("reprepare")), \
                             patch("services.llm.bootstrap.bindings._verify_and_hashes", side_effect=AssertionError("rehash")):
                            async with ClientSession() as session:
                                for _ in range(5):
                                    status, _ = await _request(session, "GET", "http://127.0.0.1:18996/health/ready")
                                    self.assertIn(status, (200, 503))
                        self.assertEqual(FakeProvider.constructed, constructed)
                    finally:
                        await runtime.stop()
        asyncio.run(scenario())

    def test_daemon_loss_fences_and_rejects_new_session(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                config, options, capture, identities = _runtime(Path(directory), port=18997)
                daemon = None
                class LosingDaemon(FakeDaemon):
                    def __init__(self, config, proof, *, num_parallel=1):
                        super().__init__(config, proof, num_parallel=num_parallel)
                        nonlocal daemon
                        daemon = self
                    async def alive(self):
                        return not self.closed and self.alive_now
                with patch("services.llm.bootstrap.runtime.observe_runtime_identities", return_value=identities), \
                     patch("services.llm.bootstrap.bindings.SmolLMProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.CoEdITProvider", FakeProvider), \
                     patch("services.llm.bootstrap.bindings.GECToRProvider", FakeProvider):
                    runtime = BootstrapRuntime(config, options, proof_capture=capture, daemon_factory=LosingDaemon)
                    await runtime.start()
                    daemon.alive_now = False
                    await runtime._monitor
                    with self.assertRaises(Exception):
                        await runtime.core.start_session("s", ModelId.SMOLLM,
                                                        runtime.prepared.profiles[ModelId.SMOLLM],
                                                        runtime.prepared.bindings[ModelId.SMOLLM].provider,
                                                        idempotency_key="after-loss")
                    await runtime.stop()
        asyncio.run(scenario())

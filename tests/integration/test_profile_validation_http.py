import asyncio
import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path

from aiohttp import ClientSession
from aiohttp.test_utils import TestServer

from services.llm.provisioning.http import create_app
from services.llm.provisioning.profile_validation import ProfileValidationBinding
from services.llm.provisioning.profile_validation import validate_profile
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile
from services.llm.resource_manager.profiles import ProfileStore
from tests.unit.test_profiles import fixture


def wire(profile):
    result = {
        "modelId": profile.model_id.value,
        "gpuUuid": profile.gpu_uuid,
        "artifactManifestHash": profile.artifact_manifest_hash,
        "modelHash": profile.model_hash,
        "runtimeIdentity": profile.runtime_identity,
        "adapterIdentity": profile.adapter_identity,
        "profileIdentity": profile.profile_identity,
        "optimalParallelism": profile.optimal_parallelism,
        "memorySafeN": profile.memory_safe_n,
        "bufferCapacity": profile.buffer_capacity,
        "safetyReservePercent": profile.safety_reserve_percent,
        "rawSamples": [
            {"concurrency": s.concurrency, "wave": s.wave,
             "successfulRequests": s.successful_requests, "wallTimeMs": s.wall_time_ms,
             "peakVramBytes": s.peak_vram_bytes, "latencyMs": list(s.latency_ms)}
            for s in profile.raw_samples
        ],
    }
    if profile.model_id is ModelId.SMOLLM:
        result["contextSize"] = profile.context_size
    else:
        result["bucketIdentity"] = profile.bucket_identity
    return result


class ProfileValidationHTTPTests(unittest.IsolatedAsyncioTestCase):
    def stored_profile(self, tmp):
        profile, metadata = fixture()
        path = Path(tmp) / "profiles.sqlite"
        with ProfileStore(path) as store:
            store.save_measured(profile, metadata)
        return profile, metadata, path, ProfileValidationBinding(
            path, "gpu", "manifest", "model", "runtime", "adapter")

    async def test_all_models_accept_exact_measured_profiles_and_are_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            bindings = {}
            bodies = {}
            for model in ModelId:
                base, metadata = fixture()
                identity_data = {
                    "model_id": model.value, "gpu_uuid": "gpu", "artifact_manifest_hash": "manifest",
                    "model_hash": "model", "runtime_identity": "runtime", "adapter_identity": "adapter",
                    "context_size": 2048 if model is ModelId.SMOLLM else None,
                    "bucket_identity": None if model is ModelId.SMOLLM else "default",
                    "fingerprint": metadata.fingerprint,
                }
                identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                profile = replace(base, model_id=model, profile_identity=identity,
                                  context_size=identity_data["context_size"], bucket_identity=identity_data["bucket_identity"])
                if model is not ModelId.SMOLLM:
                    metadata = replace(metadata, representative_config="default")
                path = Path(tmp) / f"{model.value}.sqlite"
                with ProfileStore(path) as store:
                    store.save_measured(profile, metadata)
                binding = ProfileValidationBinding(path, "gpu", "manifest", "model", "runtime", "adapter")
                bindings[model.value] = binding
                bodies[model] = wire(profile)
            server = TestServer(create_app({}, profile_registries=bindings))
            await server.start_server()
            try:
                async with ClientSession() as client:
                    for model in ModelId:
                        response = await client.post(server.make_url("/provisioning/validate-profile"),
                            headers={"Idempotency-Key": "key-" + model.value}, json=bodies[model])
                        self.assertEqual(response.status, 200)
                        payload = await response.json()
                        self.assertEqual(payload["accepted"], True)
                        self.assertEqual(payload["operationId"], hashlib.sha256(
                            json.dumps(bodies[model], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest())
            finally:
                await server.close()

    async def test_malformed_json_is_a_client_error_and_validation_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, binding = self.stored_profile(tmp)
            server = TestServer(create_app({}, profile_registries={"SmolLM": binding}))
            await server.start_server()
            try:
                async with ClientSession() as client:
                    url = server.make_url("/provisioning/validate-profile")
                    headers = {"Idempotency-Key": "x", "Content-Type": "application/json"}
                    for raw in (b"not-json", b'{"modelId":"SmolLM","modelId":"SmolLM"}', b'{"x":NaN}', b"null", b"[]"):
                        response = await client.post(url, headers=headers, data=raw)
                        expected = 422 if b'modelId":"SmolLM","modelId' in raw or b'NaN' in raw else 400
                        self.assertEqual(response.status, expected, raw)
                    response = await client.post(url, headers=headers, json={"modelId": "SmolLM"})
                    self.assertEqual(response.status, 400)
            finally:
                await server.close()

    async def test_timeout_does_not_release_validation_worker_before_thread_exit(self):
        started = threading.Event()
        release = threading.Event()

        def blocked(_body, _bindings):
            started.set()
            release.wait(5)
            return "operation"

        import services.llm.provisioning.http as http_module
        original = http_module.validate_profile
        http_module.validate_profile = blocked
        server = TestServer(create_app({}, profile_registries={"SmolLM": ProfileValidationBinding(
            Path("/tmp/nonexistent-profile-registry"), "gpu", "manifest", "model", "runtime", "adapter")}, timeout_seconds=0.03))
        await server.start_server()
        try:
            async with ClientSession() as client:
                url = server.make_url("/provisioning/validate-profile")
                first = asyncio.create_task(client.post(url, headers={"Idempotency-Key": "x"}, json={}))
                await asyncio.to_thread(started.wait, 2)
                self.assertEqual((await first).status, 504)
                release.set()
        finally:
            release.set()
            http_module.validate_profile = original
            await server.close()

    async def test_valid_profile_can_be_sent_in_many_chunks_and_oversize_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile, metadata = fixture()
            path = Path(tmp) / "profiles.sqlite"
            with ProfileStore(path) as store:
                store.save_measured(profile, metadata)
            body = json.dumps(wire(profile), separators=(",", ":")).encode()
            binding = ProfileValidationBinding(path, "gpu", "manifest", "model", "runtime", "adapter")

            async def chunks():
                for index in range(0, len(body), 3):
                    yield body[index:index + 3]
                    await asyncio.sleep(0)

            server = TestServer(create_app({}, profile_registries={"SmolLM": binding},
                                           profile_max_body_bytes=len(body) + 1))
            await server.start_server()
            try:
                async with ClientSession() as client:
                    response = await client.post(server.make_url("/provisioning/validate-profile"),
                        headers={"Idempotency-Key": "chunked", "Content-Type": "application/json"}, data=chunks())
                    self.assertEqual(response.status, 200)
                    oversized = await client.post(server.make_url("/provisioning/validate-profile"),
                        headers={"Idempotency-Key": "large", "Content-Type": "application/json"},
                        data=body + b"xx")
                    self.assertEqual(oversized.status, 413)
                    reusable = await client.post(server.make_url("/provisioning/validate-profile"),
                        headers={"Idempotency-Key": "after-large", "Content-Type": "application/json"},
                        data=body)
                    self.assertEqual(reusable.status, 200)
            finally:
                await server.close()

    async def test_pre_worker_rejections_release_reserved_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, binding = self.stored_profile(tmp)
            server = TestServer(create_app({}, profile_registries={"SmolLM": binding}, max_concurrent=1,
                                           profile_max_body_bytes=8))
            await server.start_server()
            try:
                async with ClientSession() as client:
                    url = server.make_url("/provisioning/validate-profile")
                    headers = {"Idempotency-Key": "x", "Content-Type": "application/json"}
                    oversized = await client.post(url, headers=headers, data=b"012345678")
                    self.assertEqual(oversized.status, 413)
                    malformed = await client.post(url, headers=headers, data=b"{")
                    self.assertEqual(malformed.status, 400)
                    # A leaked pre-worker reservation would make this request 429.
                    retry = await client.post(url, headers=headers, data=b"{}")
                    self.assertEqual(retry.status, 400)
            finally:
                await server.close()

    async def test_recursive_json_is_strict_400_and_huge_integer_is_validation_422(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile, _, _, binding = self.stored_profile(tmp)
            server = TestServer(create_app({}, profile_registries={"SmolLM": binding}))
            await server.start_server()
            try:
                async with ClientSession() as client:
                    url = server.make_url("/provisioning/validate-profile")
                    recursive = (b"[" * 1200) + (b"]" * 1200)
                    response = await client.post(url, headers={"Idempotency-Key": "recursive",
                        "Content-Type": "application/json"}, data=recursive)
                    self.assertEqual(response.status, 400)
                    payload = wire(profile)
                    huge = json.dumps(payload, separators=(",", ":")).replace(
                        str(payload["contextSize"]), "9" * 10000).encode()
                    response = await client.post(url, headers={"Idempotency-Key": "huge",
                        "Content-Type": "application/json"}, data=huge)
                    self.assertEqual(response.status, 422)
            finally:
                await server.close()

    async def test_disconnected_worker_late_value_error_is_consumed_and_retains_capacity(self):
        started, release = threading.Event(), threading.Event()
        diagnostics = []

        def late_failure(_body, _bindings):
            started.set()
            release.wait(5)
            raise ValueError("late controlled failure")

        import services.llm.provisioning.http as http_module
        original = http_module.validate_profile
        http_module.validate_profile = late_failure
        with tempfile.TemporaryDirectory() as tmp:
            profile, _, _, binding = self.stored_profile(tmp)
            server = TestServer(create_app({}, profile_registries={"SmolLM": binding},
                                           max_concurrent=1, timeout_seconds=2))
            loop = asyncio.get_running_loop()
            old_handler = loop.get_exception_handler()
            loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
            await server.start_server()
            try:
                async with ClientSession() as client:
                    url = server.make_url("/provisioning/validate-profile")
                    disconnected = asyncio.create_task(client.post(url,
                        headers={"Idempotency-Key": "disconnect"}, json=wire(profile)))
                    await asyncio.to_thread(started.wait, 2)
                    disconnected.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await disconnected
                    busy = await client.post(url, headers={"Idempotency-Key": "busy"}, json=wire(profile))
                    self.assertEqual(busy.status, 429)
                    release.set()
                    await asyncio.sleep(.1)
                    self.assertEqual(diagnostics, [])
                    # The callback releases the slot only after the thread exits.
                    http_module.validate_profile = validate_profile
                    available = await client.post(url, headers={"Idempotency-Key": "available"}, json=wire(profile))
                    self.assertEqual(available.status, 200)
            finally:
                release.set()
                http_module.validate_profile = original
                loop.set_exception_handler(old_handler)
                await server.close()

    async def test_sqlite_error_is_redacted_and_draft_or_corrupt_hash_is_422(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile, metadata, path, binding = self.stored_profile(tmp)
            with ProfileStore(path) as store:
                store._db.execute("UPDATE profiles SET content_hash=?", ("corrupt",))
            server = TestServer(create_app({}, profile_registries={"SmolLM": binding}))
            await server.start_server()
            try:
                async with ClientSession() as client:
                    url = server.make_url("/provisioning/validate-profile")
                    corrupt = await client.post(url, headers={"Idempotency-Key": "corrupt"}, json=wire(profile))
                    self.assertEqual(corrupt.status, 422)
                    text = await corrupt.text()
                    self.assertNotIn("corrupt", text)
            finally:
                await server.close()

            draft_path = Path(tmp) / "draft.sqlite"
            with ProfileStore(draft_path) as store:
                store.save_draft(profile, metadata)
            draft_binding = ProfileValidationBinding(draft_path, "gpu", "manifest", "model", "runtime", "adapter")
            server = TestServer(create_app({}, profile_registries={"SmolLM": draft_binding}))
            await server.start_server()
            try:
                async with ClientSession() as client:
                    response = await client.post(server.make_url("/provisioning/validate-profile"),
                        headers={"Idempotency-Key": "draft"}, json=wire(profile))
                    self.assertEqual(response.status, 422)
            finally:
                await server.close()

    async def test_caller_mapping_mutation_cannot_change_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile, _, _, binding = self.stored_profile(tmp)
            authority = {"SmolLM": binding}
            server = TestServer(create_app({}, profile_registries=authority))
            authority.clear()
            await server.start_server()
            try:
                async with ClientSession() as client:
                    response = await client.post(server.make_url("/provisioning/validate-profile"),
                        headers={"Idempotency-Key": "snapshot"}, json=wire(profile))
                    self.assertEqual(response.status, 200)
            finally:
                await server.close()

    async def test_timeout_overload_and_slot_reuse_are_independent_of_body_reading(self):
        started, release = threading.Event(), threading.Event()
        calls = 0

        def blocked(body, bindings):
            nonlocal calls
            calls += 1
            started.set()
            release.wait(5)
            return validate_profile(body, bindings)

        import services.llm.provisioning.http as http_module
        original = http_module.validate_profile
        http_module.validate_profile = blocked
        with tempfile.TemporaryDirectory() as tmp:
            profile, metadata = fixture()
            path = Path(tmp) / "profiles.sqlite"
            with ProfileStore(path) as store:
                store.save_measured(profile, metadata)
            binding = ProfileValidationBinding(path, "gpu", "manifest", "model", "runtime", "adapter")
            server = TestServer(create_app({}, profile_registries={"SmolLM": binding},
                                           max_concurrent=1, timeout_seconds=.03))
            await server.start_server()
            try:
                async with ClientSession() as client:
                    url = server.make_url("/provisioning/validate-profile")
                    first = asyncio.create_task(client.post(url, headers={"Idempotency-Key": "one"}, json=wire(profile)))
                    await asyncio.to_thread(started.wait, 2)
                    self.assertEqual((await first).status, 504)
                    busy = await client.post(url, headers={"Idempotency-Key": "two"}, json=wire(profile))
                    self.assertEqual(busy.status, 429)
                    release.set()
                    await asyncio.sleep(.05)
                    valid = await client.post(url, headers={"Idempotency-Key": "three"}, json=wire(profile))
                    self.assertEqual(valid.status, 200)
            finally:
                release.set()
                http_module.validate_profile = original
                await server.close()

    async def test_identity_selector_and_measurement_mutations_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile, metadata = fixture()
            path = Path(tmp) / "profiles.sqlite"
            with ProfileStore(path) as store:
                store.save_measured(profile, metadata)
            binding = ProfileValidationBinding(path, "gpu", "manifest", "model", "runtime", "adapter")
            server = TestServer(create_app({}, profile_registries={"SmolLM": binding}))
            await server.start_server()
            try:
                async with ClientSession() as client:
                    for field, value in (
                        ("gpuUuid", "other"), ("artifactManifestHash", "other"),
                        ("runtimeIdentity", "other"), ("adapterIdentity", "other"),
                        ("profileIdentity", "other"), ("contextSize", 4096),
                        ("optimalParallelism", 1), ("rawSamples", []),
                    ):
                        payload = wire(profile)
                        payload[field] = value
                        response = await client.post(server.make_url("/provisioning/validate-profile"),
                            headers={"Idempotency-Key": field}, json=payload)
                        expected = 400 if field in {"optimalParallelism", "rawSamples"} else 422
                        self.assertEqual(response.status, expected, field)
            finally:
                await server.close()

    def test_invalid_binding_and_missing_registry_are_rejected_without_creation(self):
        with self.assertRaises(ValueError):
            ProfileValidationBinding("relative.sqlite", "gpu", "manifest", "model", "runtime", "adapter")
        with self.assertRaises(ValueError):
            ProfileValidationBinding("/tmp/profile.sqlite", "", "manifest", "model", "runtime", "adapter")
        missing = Path(tempfile.mkdtemp()) / "never-created.sqlite"
        with self.assertRaises(ValueError):
            validate_profile(wire(fixture()[0]), {"SmolLM": ProfileValidationBinding(
                missing, "gpu", "manifest", "model", "runtime", "adapter")})
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()

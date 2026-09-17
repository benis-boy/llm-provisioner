import asyncio
import unittest

from aiohttp import ClientSession
from aiohttp.test_utils import TestServer

from services.llm.health import create_app
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager
from services.llm.resource_manager.health import create_boundary
from services.llm.resource_manager.protocol import ProviderResponse


def measured_profile():
    return CapacityProfile(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter",
                           "measured", 1, 1, 1, 10,
                           (SampleMetadata(1, 1, 1, 1, 1, (1,)),), context_size=128)


class Provider:
    async def validate(self, _profile): pass
    async def load(self, _profile): pass
    async def ready(self): pass
    async def unload(self): pass
    async def verify_cleanup(self): return True
    async def cancel(self, _request_id): pass
    async def validate_input(self, payload, *, context_size=None, bucket_identity=None): pass
    async def execute(self, _request_id, _payload): return ProviderResponse(b"ok")


class ResourceManagerHealthHTTPTests(unittest.IsolatedAsyncioTestCase):
    def external(self, **overrides):
        values = {name: True for name in (
            "sqlite", "gpu", "artifacts", "adapter", "ollama", "profile", "cleanup")}
        values.update(overrides)
        return values

    async def test_live_and_readiness_follow_real_resource_manager(self):
        rm = ResourceManager()
        boundary = create_boundary(rm, lambda: self.external())
        server = TestServer(create_app(boundary))
        await server.start_server()
        try:
            async with ClientSession() as client:
                response = await client.get(server.make_url("/health/live"))
                self.assertEqual(response.status, 200)
                response = await client.get(server.make_url("/health/ready"))
                self.assertEqual(response.status, 503)
                self.assertEqual((await response.json())["reasons"], [
                    {"phase": "startup"},
                    {"dependency": "adapter", "reason": "unavailable"},
                    {"dependency": "cleanup", "reason": "unavailable"}])

                session = await rm.start_session("scheduler", ModelId.SMOLLM, measured_profile(),
                                                 Provider(), idempotency_key="start")
                response = await client.get(server.make_url("/health/ready"))
                self.assertEqual(response.status, 200)
                self.assertEqual((await response.json())["reasons"], [])

                await rm.stop_session(session.session_token, idempotency_key="stop")
                response = await client.get(server.make_url("/health/ready"))
                self.assertEqual(response.status, 503)
                self.assertEqual((await response.json())["reasons"], [
                    {"phase": "startup"},
                    {"dependency": "adapter", "reason": "unavailable"},
                    {"dependency": "cleanup", "reason": "unavailable"}])
        finally:
            await server.close()

    async def test_profile_dependency_false_cannot_be_overridden_by_measured_lifecycle(self):
        rm = ResourceManager()
        session = await rm.start_session("scheduler", ModelId.SMOLLM, measured_profile(),
                                         Provider(), idempotency_key="start")
        boundary = create_boundary(rm, lambda: self.external(profile=False))
        server = TestServer(create_app(boundary))
        await server.start_server()
        try:
            async with ClientSession() as client:
                response = await client.get(server.make_url("/health/ready"))
                self.assertEqual(response.status, 503)
                body = await response.json()
                self.assertIn({"dependency": "profile", "reason": "unavailable"}, body["reasons"])
                dependencies = await client.get(server.make_url("/health/dependencies"))
                self.assertEqual(dependencies.status, 200)
                self.assertFalse((await dependencies.json())["dependencies"]["profile"]["ok"])
        finally:
            await server.close()
            await rm.stop_session(session.session_token, idempotency_key="stop")

    async def test_readiness_rechecks_external_proof_after_a_blocked_probe(self):
        rm = ResourceManager()
        session = await rm.start_session("scheduler", ModelId.SMOLLM, measured_profile(),
                                         Provider(), idempotency_key="start")
        values = self.external()
        started, release = asyncio.Event(), asyncio.Event()

        async def probe():
            started.set()
            await release.wait()
            return True

        boundary = create_boundary(rm, lambda: values, probes={"profile": probe})
        server = TestServer(create_app(boundary))
        await server.start_server()
        try:
            async with ClientSession() as client:
                request = asyncio.create_task(client.get(server.make_url("/health/ready")))
                await started.wait()
                values["profile"] = False
                release.set()
                response = await request
                self.assertEqual(response.status, 503)
                self.assertIn({"dependency": "profile", "reason": "unavailable"},
                              (await response.json())["reasons"])
        finally:
            release.set()
            await server.close()
            await rm.stop_session(session.session_token, idempotency_key="stop")


if __name__ == "__main__":
    unittest.main()

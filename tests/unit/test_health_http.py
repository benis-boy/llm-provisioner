import asyncio
import threading
import unittest

from aiohttp import ClientSession
from aiohttp.test_utils import TestServer

from services.llm.health import HealthBoundary, HealthSnapshot, create_app


class HealthHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_and_fail_closed_snapshot(self):
        boundary = HealthBoundary(HealthSnapshot("startup", {name: True for name in (
            "sqlite", "gpu", "artifacts", "adapter", "ollama", "profile", "cleanup")}))
        server = TestServer(create_app(boundary)); await server.start_server()
        try:
            async with ClientSession() as client:
                live = await client.get(server.make_url("/health/live")); self.assertEqual(live.status, 200)
                self.assertEqual(await live.json(), {"live": True})
                ready = await client.get(server.make_url("/health/ready")); self.assertEqual(ready.status, 503)
                self.assertEqual((await ready.json())["reasons"], [{"phase": "startup"}])
                deps = await client.get(server.make_url("/health/dependencies")); self.assertEqual(deps.status, 200)
                self.assertEqual(set((await deps.json())["dependencies"]), {"sqlite", "gpu", "artifacts", "adapter", "ollama", "profile", "cleanup"})
        finally: await server.close()

    async def test_healthy_readiness_and_unsafe_phases(self):
        states = {name: True for name in ("sqlite", "gpu", "artifacts", "adapter", "ollama", "profile", "cleanup")}
        for phase in ("provisioning", "loading", "unloading", "profile_mismatch", "profile_unproved", "cleanup_failed"):
            boundary = HealthBoundary(HealthSnapshot(phase, states))
            ok, reasons, _ = await boundary.readiness()
            self.assertFalse(ok); self.assertEqual(reasons, [{"phase": phase}])
        boundary = HealthBoundary(HealthSnapshot("stable", states))
        ok, reasons, _ = await boundary.readiness()
        self.assertTrue(ok); self.assertEqual(reasons, [])

    async def test_readiness_captures_callable_snapshot_once(self):
        states = {name: True for name in ("sqlite", "gpu", "artifacts", "adapter", "ollama", "profile", "cleanup")}
        calls = []
        def changing_snapshot():
            calls.append(True)
            return HealthSnapshot("stable" if len(calls) == 1 else "loading", states)
        boundary = HealthBoundary(changing_snapshot)
        ok, reasons, _ = await boundary.readiness()
        self.assertTrue(ok)
        self.assertEqual(reasons, [])
        self.assertEqual(len(calls), 1)

    async def test_malformed_missing_and_sensitive_states_fail_closed(self):
        boundary = HealthBoundary({"phase": "stable", "dependencies": {"gpu": {"ok": False, "reason": "secret /tmp"}}})
        ok, reasons, states = await boundary.readiness()
        self.assertFalse(ok); self.assertEqual(states["gpu"].reason, "state_malformed")
        self.assertEqual(reasons[0], {"dependency": "sqlite", "reason": "state_missing"})
        self.assertNotIn("secret", str(reasons)); self.assertNotIn("tmp", str(reasons))

    async def test_probe_timeout_exception_overload_and_no_leak(self):
        gate = asyncio.Event()
        async def slow(): await gate.wait()
        def broken(): raise RuntimeError("secret /path prompt")
        boundary = HealthBoundary(HealthSnapshot("stable", {name: True for name in (
            "sqlite", "gpu", "artifacts", "adapter", "ollama", "profile", "cleanup")}),
            probes={"gpu": slow}, timeout_seconds=.01, max_concurrent=1)
        result = await boundary.dependencies()
        self.assertEqual(result["gpu"].reason, "probe_timeout")
        self.assertEqual(result["sqlite"].reason, "ok")
        separate = HealthBoundary(HealthSnapshot("stable", {name: True for name in (
            "sqlite", "gpu", "artifacts", "adapter", "ollama", "profile", "cleanup")}), probes={"sqlite": broken})
        self.assertEqual((await separate.dependencies())["sqlite"].reason, "probe_failed")
        gate.set(); await boundary.close()

    async def test_sync_async_and_callable_probes_and_bounded_cleanup(self):
        release = threading.Event()
        class Probe:
            def __call__(self): return True
        async def async_probe(): return {"ok": True}
        def blocked(): release.wait()
        states = {name: True for name in ("sqlite", "gpu", "artifacts", "adapter", "ollama", "profile", "cleanup")}
        boundary = HealthBoundary(HealthSnapshot("stable", states), probes={"sqlite": Probe(), "gpu": async_probe, "adapter": blocked}, timeout_seconds=.01, max_concurrent=3, cleanup_timeout_seconds=.01)
        result = await boundary.dependencies()
        self.assertEqual(result["sqlite"].reason, "ok"); self.assertEqual(result["gpu"].reason, "ok")
        await asyncio.wait_for(boundary.close(), .1)
        release.set()

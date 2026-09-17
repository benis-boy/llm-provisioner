"""Fast harness tests; no model, CUDA runtime, Docker build, or GPU is used."""
import asyncio
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.core import ResourceManager
from services.llm.resource_manager.protocol import EventKind, ProviderResponse
from tools.compatibility.model_runtime import MAX_FIXTURES, SMALL_FIXTURES
from tools.compatibility.rm_spike import ChildRPCProvider, _profile


class FixtureTests(unittest.TestCase):
    def test_fixtures_are_deterministic_and_bounded(self):
        for model in ("SmolLM", "CoEdIT", "GECToR"):
            self.assertNotEqual(SMALL_FIXTURES[model], MAX_FIXTURES[model])
            self.assertLessEqual(len(MAX_FIXTURES[model]), 512)

    def test_smollm_fixtures_are_within_the_explicit_raw_byte_bucket(self):
        for fixture in (SMALL_FIXTURES["SmolLM"], MAX_FIXTURES["SmolLM"]):
            self.assertLessEqual(len(fixture.encode("utf-8")), 256)

    def test_candidate_profiles_are_explicitly_unmeasured_p1(self):
        for model in ModelId:
            profile = _profile(model)
            self.assertEqual(profile.optimal_parallelism, 1)
            self.assertEqual(profile.memory_safe_n, 1)
            self.assertEqual(profile.raw_samples[0].successful_requests, 0)
            self.assertEqual(profile.profile_identity, "unmeasured-harness")
            if model is ModelId.SMOLLM:
                self.assertEqual(profile.context_size, 512)
                self.assertIsNone(profile.bucket_identity)
            else:
                self.assertIsNone(profile.context_size)
                self.assertEqual(profile.bucket_identity, "upper-fixture")


class RPCBoundsTests(unittest.TestCase):
    def test_execution_started_notification_is_not_blocked_by_concurrent_response(self):
        program = (
            "import json, sys; "
            "\nfor line in sys.stdin:"
            "\n r=json.loads(line); "
            "\n if r['op']=='execute': print(json.dumps({'event':'execution_started','request_id':r['request_id']}), flush=True); print(json.dumps({'id':r['id'],'ok':True,'value':'ok'}), flush=True)"
        )
        provider = ChildRPCProvider("SmolLM", Path("."), command=[sys.executable, "-u", "-c", program], timeout=2)
        try:
            response = asyncio.run(provider.execute("request", b"fixture"))
            self.assertEqual(response.result, b"ok")
            self.assertTrue(asyncio.run(provider.executing("request")))
        finally:
            provider.close()

    def test_eof_fails_pending_calls_and_rejects_new_calls(self):
        provider = ChildRPCProvider("SmolLM", __import__("pathlib").Path("."),
                                    command=[sys.executable, "-u", "-c", "import sys; sys.stdin.close()"],
                                    timeout=5)
        try:
            with self.assertRaisesRegex(RuntimeError, "EOF/process death"):
                provider._call("ready")
            with self.assertRaisesRegex(RuntimeError, "dead"):
                provider._call("ready")
        finally:
            provider.close()

    def test_unresponsive_child_rpc_and_cleanup_are_bounded(self):
        provider = ChildRPCProvider("SmolLM", __import__("pathlib").Path("."),
                                    command=[sys.executable, "-u", "-c", "import time; time.sleep(30)"],
                                    timeout=.05)
        try:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                provider._call("ready")
        finally:
            provider.close()

    def test_killed_child_group_reaps_orphaned_grandchild(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "grandchild.pid"
            program = (
                "import pathlib, subprocess, sys, time; "
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); time.sleep(30)"
            )
            provider = ChildRPCProvider("SmolLM", Path("."), command=[sys.executable, "-u", "-c", program], timeout=.1)
            try:
                deadline = time.monotonic() + 2
                while not pid_file.exists() and time.monotonic() < deadline: time.sleep(.01)
                self.assertTrue(pid_file.exists(), "child did not create grandchild")
                grandchild = int(pid_file.read_text())
                provider.inject_process_failure()
                provider.close()
                self.assertTrue(provider.process_group_gone())
                with self.assertRaises(ProcessLookupError): os.kill(grandchild, 0)
            finally:
                provider.close()

    def test_dead_provider_unload_succeeds_only_after_group_cleanup(self):
        provider = ChildRPCProvider("SmolLM", Path("."),
            command=[sys.executable, "-u", "-c", "import time; time.sleep(30)"], timeout=.1)
        try:
            provider.inject_process_failure()
            asyncio.run(provider.unload())
            self.assertTrue(provider.process_group_gone())
            self.assertTrue(asyncio.run(provider.verify_cleanup()))
        finally:
            provider.close()

    def test_saved_group_id_detects_orphan_after_leader_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "orphan.pid"
            program = (
                "import pathlib, subprocess, sys; "
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))"
            )
            provider = ChildRPCProvider("SmolLM", Path("."), command=[sys.executable, "-c", program])
            try:
                deadline = time.monotonic() + 2
                while provider.process.poll() is None and time.monotonic() < deadline: time.sleep(.01)
                self.assertIsNotNone(provider.process.poll())
                self.assertFalse(provider.process_group_gone())
            finally:
                provider.close()


class FenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_result_is_fenced_after_cancel(self):
        class Slow:
            async def validate(self, profile): pass
            async def load(self, profile): pass
            async def ready(self): pass
            async def validate_input(self, payload, *, context_size, bucket_identity): pass
            async def execute(self, request_id, payload):
                await asyncio.sleep(.02)
                return ProviderResponse(b"late")
            async def cancel(self, request_id): pass
            async def unload(self): pass
            async def verify_cleanup(self): return True

        from tools.compatibility.rm_spike import _profile
        rm = ResourceManager(cleanup_timeout=.2)
        provider = Slow()
        session = await rm.start_session("test", ModelId.SMOLLM, _profile(ModelId.SMOLLM), provider,
                                         idempotency_key="start")
        try:
            await rm.submit(session.session_token, "r", "a", b"x", idempotency_key="submit", context_size=512)
            self.assertTrue(await rm.cancel_request(session.session_token, "r", idempotency_key="cancel"))
            async def finished():
                async for event in rm.watch_progress(session.session_token):
                    if event.kind is EventKind.RESPONSE_FINISHED:
                        return event
            event = await asyncio.wait_for(finished(), .2)
            self.assertIsNone(event.result)
        finally:
            await rm.stop_session(session.session_token, idempotency_key="stop")

    async def test_failure_event_fails_fast_in_harness_collector(self):
        from tools.compatibility.rm_spike import _events_until
        class Failed:
            async def validate(self, profile): pass
            async def load(self, profile): pass
            async def ready(self): pass
            async def validate_input(self, *args, **kwargs): pass
            async def execute(self, *args): raise RuntimeError("broken")
            async def cancel(self, *args): pass
            async def unload(self): pass
            async def verify_cleanup(self): return True
        rm = ResourceManager(cleanup_timeout=.2)
        session = await rm.start_session("test", ModelId.SMOLLM, _profile(ModelId.SMOLLM), Failed(), idempotency_key="failed-start")
        try:
            await rm.submit(session.session_token, "failed", "a", b"x", idempotency_key="failed-submit", context_size=512)
            with self.assertRaisesRegex(RuntimeError, "provider failure"):
                await _events_until(rm, session.session_token, {EventKind.RESPONSE_FINISHED}, .2, "failed")
        finally:
            await rm.stop_session(session.session_token, idempotency_key="failed-stop")

    async def test_stale_token_is_rejected_after_stop(self):
        class Empty:
            async def validate(self, profile): pass
            async def load(self, profile): pass
            async def ready(self): pass
            async def validate_input(self, *args, **kwargs): pass
            async def execute(self, *args): return ProviderResponse(b"ok")
            async def cancel(self, *args): pass
            async def unload(self): pass
            async def verify_cleanup(self): return True
        rm = ResourceManager(cleanup_timeout=.2)
        session = await rm.start_session("test", ModelId.SMOLLM, _profile(ModelId.SMOLLM), Empty(), idempotency_key="stale-start")
        await rm.stop_session(session.session_token, idempotency_key="stale-stop")
        with self.assertRaises(Exception):
            await rm.submit(session.session_token, "stale", "a", b"x", idempotency_key="stale-submit", context_size=512)


if __name__ == "__main__":
    unittest.main()

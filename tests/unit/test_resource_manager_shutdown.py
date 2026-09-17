"""Shutdown fencing tests (provider operations are deliberately fake)."""

import asyncio
import unittest

from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager, ResourceManagerError
from services.llm.resource_manager.protocol import EventKind, ProviderResponse


def make_profile():
    return CapacityProfile(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime",
                           "adapter", "profile", 1, 1, 1, 10,
                           (SampleMetadata(1, 0, 1, 1, 1, (1,)),), context_size=128)


class Provider:
    def __init__(self):
        self.done = asyncio.Event()
        self.unload_started = asyncio.Event()
        self.unload_gate = asyncio.Event()
        self.cleanup_ok = True

    async def validate(self, _): pass
    async def load(self, _): pass
    async def ready(self): pass
    async def validate_input(self, *_args, **_kwargs): pass
    async def execute(self, *_args):
        await self.done.wait()
        return ProviderResponse(b"late")
    async def cancel(self, _): pass
    async def unload(self):
        self.unload_started.set()
        await self.unload_gate.wait()
    async def verify_cleanup(self): return self.cleanup_ok


class ShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_exception_emits_failure_without_response_completion(self):
        provider = Provider()
        async def execute(_request_id, _payload):
            raise RuntimeError("provider exploded")
        provider.execute = execute
        rm = ResourceManager()
        session = await rm.start_session("s", ModelId.SMOLLM, make_profile(), provider,
                                         idempotency_key="start")
        await rm.submit(session.session_token, "r", "a", b"x", idempotency_key="r", context_size=128)
        await asyncio.sleep(0)
        events = rm._events[session.session_token]
        self.assertTrue(any(event.kind is EventKind.FAILURE for event in events))
        self.assertFalse(any(event.kind is EventKind.RESPONSE_FINISHED for event in events))

    async def test_cancelled_response_completes_without_result_and_cancelled_task_emits_cancelled(self):
        provider = Provider()
        entered = asyncio.Event()
        async def execute(_request_id, _payload):
            entered.set()
            await provider.done.wait()
            return ProviderResponse(b"late")
        provider.execute = execute
        rm = ResourceManager()
        session = await rm.start_session("s", ModelId.SMOLLM, make_profile(), provider,
                                         idempotency_key="start")
        await rm.submit(session.session_token, "r", "a", b"x", idempotency_key="r", context_size=128)
        await entered.wait()
        await rm.cancel_request(session.session_token, "r", idempotency_key="cancel")
        provider.done.set()
        await asyncio.sleep(0)
        events = rm._events[session.session_token]
        completed = [event for event in events if event.kind is EventKind.RESPONSE_FINISHED]
        self.assertEqual(len(completed), 1)
        self.assertIsNone(completed[0].result)

        # A directly cancelled owned execution must publish CANCELLED, rather
        # than losing its terminal progress event to asyncio cancellation.
        provider2 = Provider()
        entered2 = asyncio.Event()
        async def execute2(_request_id, _payload):
            entered2.set()
            await provider2.done.wait()
            return ProviderResponse(b"late")
        provider2.execute = execute2
        rm2 = ResourceManager()
        session2 = await rm2.start_session("s2", ModelId.SMOLLM, make_profile(), provider2,
                                            idempotency_key="start-2")
        await rm2.submit(session2.session_token, "r2", "a", b"x", idempotency_key="r2", context_size=128)
        # Cancelling an unstarted coroutine legitimately executes none of its
        # try/finally body, so explicitly wait for execution entry first.
        await entered2.wait()
        task = next(iter(rm2._active.values())).task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        events = rm2._events[session2.session_token]
        self.assertTrue(any(event.kind is EventKind.CANCELLED for event in events))
    async def test_shutdown_fences_late_result_and_replay(self):
        provider = Provider()
        rm = ResourceManager(stop_timeout=.05)
        session = await rm.start_session("s", ModelId.SMOLLM, make_profile(), provider,
                                         idempotency_key="start")
        await rm.submit(session.session_token, "r", "a", b"x", idempotency_key="r", context_size=128)
        stopping = asyncio.create_task(rm.shutdown())
        await asyncio.sleep(.01)
        self.assertFalse(stopping.done(), "shutdown must wait for provider execution")
        with self.assertRaises(ResourceManagerError) as failure:
            await stopping
        self.assertEqual(failure.exception.failure.code, "cleanup_timeout")
        provider.done.set()
        await asyncio.sleep(0)
        self.assertTrue(all(event.result is None for event in rm._events[session.session_token]
                            if event.kind is EventKind.RESPONSE_FINISHED))
        with self.assertRaises(ResourceManagerError):
            await rm.start_session("s", ModelId.SMOLLM, make_profile(), provider,
                                   idempotency_key="start")

    async def test_concurrent_shutdown_shares_cleanup_and_failure_is_sticky(self):
        provider = Provider()
        provider.cleanup_ok = False
        rm = ResourceManager(stop_timeout=.05)
        await rm.start_session("s", ModelId.SMOLLM, make_profile(), provider, idempotency_key="start")
        results = await asyncio.gather(rm.shutdown(), rm.shutdown(), return_exceptions=True)
        self.assertTrue(all(isinstance(result, ResourceManagerError) for result in results))
        self.assertEqual(rm.snapshot().phase, "cleanup_failed")

    async def test_shutdown_during_load_retains_provider_fence(self):
        provider = Provider()
        entered = asyncio.Event()
        gate = asyncio.Event()

        async def load(_):
            entered.set()
            await gate.wait()
        provider.load = load
        rm = ResourceManager(load_timeout=1, stop_timeout=.02)
        starting = asyncio.create_task(rm.start_session("s", ModelId.SMOLLM, make_profile(), provider,
                                                        idempotency_key="start"))
        await entered.wait()
        stopping = asyncio.create_task(rm.shutdown())
        await asyncio.sleep(0)
        self.assertTrue(rm.snapshot().permanently_closed)
        gate.set()
        await asyncio.gather(starting, stopping, return_exceptions=True)
        self.assertFalse(rm.snapshot().available)


if __name__ == "__main__":
    unittest.main()

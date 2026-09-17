import tempfile
import unittest
import asyncio
from pathlib import Path

from services.llm.queue.contracts import ModelId
from services.llm.queue.store import IdempotencyConflict, OperationStale, QueueStore
from services.llm.queue.scheduler import QueueScheduler
from services.llm.queue.results import LocalPublisher, ResultStore
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata


def _profile():
    return CapacityProfile(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", "profile",
                           1, 1, 1, 20, (SampleMetadata(1, 0, 1, 1, 1, (1,)),), context_size=128)


class BlockedRM:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.sessions = []
        self.stops = []

    async def start_session(self, scheduler_id, model_id, profile, provider, *, idempotency_key):
        self.entered.set()
        await self.release.wait()
        from services.llm.resource_manager.protocol import SessionInfo
        session = SessionInfo(scheduler_id, "rm-" + idempotency_key, model_id, len(self.sessions) + 1)
        self.sessions.append(session)
        return session

    async def stop_session(self, token, *, reason, idempotency_key):
        self.stops.append((token, reason, idempotency_key))

    async def get_capacity(self, token):
        from types import SimpleNamespace
        return SimpleNamespace(free_slots=0)

    async def watch_progress(self, token, after=0):
        await asyncio.Event().wait()
        yield None


class _Provider:
    async def validate(self, profile): pass
    async def load(self, profile): pass
    async def ready(self): pass
    async def validate_input(self, payload, *, context_size, bucket_identity): pass
    async def execute(self, request_id, payload): return None
    async def cancel(self, request_id): pass
    async def unload(self): pass
    async def verify_cleanup(self): return True


class SchedulerOperationJournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "queue.sqlite"
        self.store = QueueStore(self.path, "scheduler", ModelId.SMOLLM)
        self.session = self.store.start_session("start-1")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_cancel_replay_and_changed_key_are_durable(self):
        self.store.enqueue("request", "payload", idempotency_key="request")
        self.assertEqual(self.store.cancel("request", idempotency_key="cancel-1")["status"], "cancelled")
        self.assertEqual(self.store.cancel("request", idempotency_key="cancel-1")["status"], "cancelled")
        with self.assertRaises(IdempotencyConflict):
            self.store.cancel("other", idempotency_key="cancel-1")

    def test_stop_cancelled_outcome_and_reopen(self):
        self.store.enqueue("request", "payload", idempotency_key="request")
        self.assertEqual(self.store.stop("client", cancelled=True, idempotency_key="stop-1"), 1)
        self.store.close()
        self.store = QueueStore(self.path, "scheduler", ModelId.SMOLLM)
        self.store.session = self.session
        self.assertEqual(self.store.stop("client", cancelled=True, idempotency_key="stop-1"), 1)
        self.assertEqual(self.store.get("request")["status"], "cancelled")

    def test_old_start_key_is_stale_after_new_generation(self):
        self.store.recover_session()
        with self.assertRaises(OperationStale):
            self.store.start_session("start-1")

    def test_altered_same_generation_session_is_rejected(self):
        from services.llm.queue.store import Session, OperationStale
        with self.assertRaises(OperationStale):
            self.store.complete_start("start-1", Session("scheduler", "wrong", "wrong", self.session.generation))

    def test_blocked_start_stop_fences_late_rm_release(self):
        async def scenario():
            root = Path(self.tmp.name)
            rm = BlockedRM()
            results = ResultStore(root / "results")
            publisher = LocalPublisher(results, root / "publisher.sqlite")
            scheduler = QueueScheduler(self.store, rm, _profile(), _Provider(), decoder=lambda x: x,
                                        result_store=results, publisher=publisher, stop_timeout=.02)
            starting = asyncio.create_task(scheduler.start(idempotency_key="start-race"))
            await rm.entered.wait()
            stopped = asyncio.create_task(scheduler.stop("race", cancelled=True, idempotency_key="stop-race"))
            await asyncio.sleep(0)
            rm.release.set()
            with self.assertRaises(Exception): await starting
            await stopped
            self.assertFalse(scheduler._started)
            self.assertFalse(scheduler._tasks)
            self.assertTrue(rm.stops)
            publisher.close()
        asyncio.run(scenario())

    def test_concurrent_keyed_starts_create_one_local_and_rm_session(self):
        async def scenario():
            root = Path(self.tmp.name)
            rm = BlockedRM(); rm.release.set()
            results = ResultStore(root / "results")
            publisher = LocalPublisher(results, root / "publisher.sqlite")
            scheduler = QueueScheduler(self.store, rm, _profile(), _Provider(), decoder=lambda x: x,
                                        result_store=results, publisher=publisher)
            first, second = await asyncio.gather(scheduler.start(idempotency_key="same"),
                                                 scheduler.start(idempotency_key="same"))
            self.assertIs(first, second)
            self.assertEqual(len(rm.sessions), 1)
            await scheduler.stop("done")
            publisher.close()
        asyncio.run(scenario())

    def test_lost_start_ack_retries_same_key_without_local_recovery(self):
        async def scenario():
            class LostAckRM(BlockedRM):
                def __init__(self): super().__init__(); self.calls = 0
                async def start_session(self, *args, **kwargs):
                    self.calls += 1
                    if self.calls == 1:
                        self.entered.set(); await self.release.wait()
                        raise OSError("ack lost")
                    return await super().start_session(*args, **kwargs)
            root = Path(self.tmp.name); rm = LostAckRM(); rm.release.set()
            results = ResultStore(root / "results")
            publisher = LocalPublisher(results, root / "publisher.sqlite")
            scheduler = QueueScheduler(self.store, rm, _profile(), _Provider(), decoder=lambda x: x,
                                        result_store=results, publisher=publisher)
            with self.assertRaises(OSError): await scheduler.start(idempotency_key="retry-start")
            accepted_generation = self.store.session.generation
            await scheduler.start(idempotency_key="retry-start")
            self.assertEqual(self.store.session.generation, accepted_generation)
            self.assertEqual(rm.calls, 2)
            await scheduler.stop("done")
            publisher.close()
        asyncio.run(scenario())

    def test_old_keyed_stop_does_not_abort_new_generation(self):
        self.store.enqueue("request", "payload", idempotency_key="request")
        self.store.stop("first", idempotency_key="stop-old")
        new_session = self.store.recover_session()
        self.assertEqual(self.store.stop("first", idempotency_key="stop-old"), 1)
        self.assertEqual(self.store.session.generation, new_session.generation)
        self.assertEqual(self.store.get("request")["status"], "error")
        self.assertEqual(self.store.db.execute("SELECT value FROM meta WHERE key='accepting'").fetchone()[0], "1")


if __name__ == "__main__":
    unittest.main()

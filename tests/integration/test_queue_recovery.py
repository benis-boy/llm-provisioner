import sqlite3, tempfile, unittest
from pathlib import Path
from services.llm.queue.store import QueueStore
from services.llm.queue.contracts import ModelId
class RecoveryTests(unittest.TestCase):
    def test_wal_state_survives_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'q.sqlite'
            with QueueStore(p, 's', ModelId.SMOLLM) as q:
                session = q.start_session()
                q.enqueue('r', 'external', idempotency_key='r')
                q.claim('r', session.token, session.generation)
            with QueueStore(p, 's', ModelId.SMOLLM) as q:
                session = q.recover_session()
                self.assertEqual(q.get('r')['status'], 'scheduled')
                self.assertEqual([row['kind'] for row in q.outbox()], ['cancel'])
                self.assertNotEqual(session.token, '')

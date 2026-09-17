import asyncio
import os
import signal
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import ProcessIdentity
from services.llm.providers.python_process import PythonWorker, WorkerRequestValidationError


def proof(clean=True):
    identity = PythonWorker._identity(os.getpid())
    return GPUProof(lambda: "GPU-test", AsyncMock(return_value=clean), lambda: None, identity)


class PythonProcessTests(unittest.IsolatedAsyncioTestCase):
    def worker(self, script, **kwargs):
        return PythonWorker(Path("/tmp"), {}, timeout=.25, frame_limit=1024, gpu_proof=proof(),
            command=[sys.executable, "-c", script], **kwargs)

    async def test_start_requires_captured_typed_gpu_proof(self):
        with self.assertRaises(RuntimeError):
            await PythonWorker(Path("/tmp"), {}, command=["false"]).start()

    async def test_real_binary_framing_round_trip(self):
        script = '''import json,struct,sys
h=sys.stdin.buffer.read(4); n=struct.unpack(">I",h)[0]; q=json.loads(sys.stdin.buffer.read(n)); b=json.dumps({"id":q["id"],"ok":True,"value":"ok"}).encode(); sys.stdout.buffer.write(struct.pack(">I",len(b))+b);sys.stdout.buffer.flush()'''
        worker = self.worker(script)
        await worker.start()
        self.assertEqual(await worker.call("load"), "ok")
        await worker.close()

    async def test_timeout_poisons_transport_and_retains_no_clean_replacement(self):
        worker = self.worker("import time; time.sleep(5)")
        await worker.start()
        with self.assertRaises(RuntimeError): await worker.call("load")
        self.assertIsNone(worker.process)  # successful forced cleanup permits no stale reuse
        with self.assertRaises(RuntimeError): await worker.call("load")

    async def test_stderr_overflow_interrupts_outstanding_call(self):
        worker = self.worker("import sys,time;sys.stderr.write('x'*2048);sys.stderr.flush();time.sleep(5)")
        await worker.start()
        with self.assertRaises(RuntimeError): await worker.call("load")
        self.assertIsNone(worker.process)

    async def test_allowlisted_worker_error_is_typed_not_malformed(self):
        script = '''import json,struct,sys
h=sys.stdin.buffer.read(4);q=json.loads(sys.stdin.buffer.read(struct.unpack(">I",h)[0]));b=json.dumps({"id":q["id"],"ok":False,"error":"gpu_mig_api_unavailable"}).encode();sys.stdout.buffer.write(struct.pack(">I",len(b))+b);sys.stdout.buffer.flush()'''
        worker=self.worker(script)
        await worker.start()
        with self.assertRaisesRegex(RuntimeError,"gpu_mig_api_unavailable"): await worker.call("gpu_identity")

    async def test_execute_batch_validation_error_is_recoverable(self):
        script = '''import json,struct,sys
for _ in range(2):
 h=sys.stdin.buffer.read(4);q=json.loads(sys.stdin.buffer.read(struct.unpack(">I",h)[0]))
 value={"id":q["id"],"ok":False,"error":"request_validation_failed"} if q["op"]=="execute_batch" else {"id":q["id"],"ok":True,"value":"ok"}
 b=json.dumps(value).encode();sys.stdout.buffer.write(struct.pack(">I",len(b))+b);sys.stdout.buffer.flush()'''
        worker = self.worker(script)
        await worker.start()
        with self.assertRaises(WorkerRequestValidationError):
            await worker.call("execute_batch", items=[])
        self.assertIsNotNone(worker.process)
        self.assertEqual(await worker.call("load"), "ok")
        await worker.close()

    async def test_request_validation_error_on_non_execute_operation_is_fatal(self):
        script = '''import json,struct,sys
h=sys.stdin.buffer.read(4);q=json.loads(sys.stdin.buffer.read(struct.unpack(">I",h)[0]))
b=json.dumps({"id":q["id"],"ok":False,"error":"request_validation_failed"}).encode()
sys.stdout.buffer.write(struct.pack(">I",len(b))+b);sys.stdout.buffer.flush()'''
        worker = self.worker(script)
        await worker.start()
        with self.assertRaisesRegex(RuntimeError, "worker operation failed"):
            await worker.call("load")
        self.assertIsNone(worker.process)

    async def test_cleanup_failure_retains_ownership_fences(self):
        worker = PythonWorker(Path("/tmp"), {}, timeout=.25, frame_limit=1024, gpu_proof=proof(False),
            command=[sys.executable, "-c", "import time; time.sleep(5)"])
        await worker.start()
        with self.assertRaises(RuntimeError): await worker.close()
        self.assertIsNotNone(worker.process)
        with self.assertRaises(RuntimeError): await worker.start()

    async def test_pid_reuse_refusal_never_signals_group(self):
        worker = self.worker("import time; time.sleep(5)")
        await worker.start()
        worker.child_identity = ProcessIdentity(worker.process.pid, worker.child_identity.start_time + 1)
        with self.assertRaises(RuntimeError): worker._signal_owned(signal.SIGTERM)
        # Restore the test-owned identity solely to perform cleanup.
        worker.child_identity = PythonWorker._identity(worker.process.pid)
        await worker.close()

    async def test_cancellation_after_spawn_is_cleaned(self):
        worker = self.worker("import time; time.sleep(5)")
        task = asyncio.create_task(worker.start())
        await asyncio.sleep(0)
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
        await worker.close()

    async def test_cancellation_while_spawn_result_is_withheld_reaps_child(self):
        real = asyncio.create_subprocess_exec
        entered, release = asyncio.Event(), asyncio.Event()
        async def delayed(*args, **kwargs):
            process = await real(*args, **kwargs)
            entered.set()
            await release.wait()
            return process
        worker = self.worker("import time; time.sleep(5)")
        with patch("asyncio.create_subprocess_exec", delayed):
            task = asyncio.create_task(worker.start())
            await entered.wait()
            task.cancel(); release.set()
            with self.assertRaises(asyncio.CancelledError): await task
        self.assertIsNone(worker.process)

    async def test_owned_group_orphan_is_reaped(self):
        # The grandchild survives its leader unless the saved session group is killed.
        worker = self.worker("import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)']); time.sleep(5)")
        await worker.start()
        pgid = worker._pgid
        await worker.close()
        with self.assertRaises(ProcessLookupError): os.killpg(pgid, 0)

    async def test_exited_leader_owned_orphan_is_reaped(self):
        # The subreaper adopts this child after its session leader exits; it
        # ignores TERM so close must perform the positively-fenced KILL phase.
        worker = self.worker("import os,signal,subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(5)']); time.sleep(.1); os._exit(0)")
        await worker.start()
        await asyncio.sleep(.05)
        await worker.close()

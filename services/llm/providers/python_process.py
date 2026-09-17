"""Fail-closed, identity-fenced transport for one isolated Python worker."""
from __future__ import annotations

import asyncio
import ctypes
import inspect
import json
import os
from pathlib import Path
import signal
import struct
import sys
import time

from .config import GPUProof
from .gpu import ProcessIdentity


class PythonWorker:
    def __init__(self, root: Path, config: dict, *, timeout: float = 120.0,
                 frame_limit: int = 256 * 1024, command=None, gpu_proof: GPUProof | None = None,
                 supervisor_pid: int | None = None) -> None:
        self.root, self.config, self.timeout, self.frame_limit = root, config, timeout, frame_limit
        self.command = command or [sys.executable, "-m", "services.llm.providers.python_worker"]
        self.gpu_proof, self.supervisor_pid = gpu_proof, supervisor_pid or os.getpid()
        self.process: asyncio.subprocess.Process | None = None
        self.child_identity: ProcessIdentity | None = None
        self.supervisor_identity: ProcessIdentity | None = None
        self._pgid: int | None = None
        self._serial = 0
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None
        self._failed: BaseException | None = None
        self._closing = False

    @staticmethod
    def _identity(pid: int) -> ProcessIdentity:
        raw = (Path("/proc") / str(pid) / "stat").read_bytes()
        fields = raw[raw.rfind(b")") + 2:].split()
        if len(fields) <= 19: raise RuntimeError("process identity unavailable")
        return ProcessIdentity(pid, int(fields[19]))

    def _assert_leader(self) -> None:
        if self.process is None or self.child_identity is None or self._pgid is None:
            raise RuntimeError("worker is unavailable")
        current = self._identity(self.process.pid)
        if current != self.child_identity: raise RuntimeError("worker PID was reused")
        fields = ((Path("/proc") / str(current.pid) / "stat").read_bytes().split(b")", 1)[1].strip().split())
        if len(fields) <= 3 or int(fields[2]) != self._pgid or int(fields[3]) != self._pgid:
            raise RuntimeError("worker process group identity changed")

    async def start(self) -> None:
        if self.process is not None or self._pgid is not None: raise RuntimeError("worker ownership is not clean")
        if type(self.gpu_proof) is not GPUProof or type(self.gpu_proof.expected_supervisor) is not ProcessIdentity:
            raise RuntimeError("captured GPU ownership proof is required")
        self.supervisor_identity = self._identity(self.supervisor_pid)
        if self.gpu_proof.expected_supervisor != self.supervisor_identity:
            raise RuntimeError("GPU proof supervisor identity mismatch")
        if sys.platform != "linux": raise RuntimeError("isolated worker ownership requires Linux")
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0: raise RuntimeError("could not establish child subreaper")
        env = {key: os.environ[key] for key in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
               "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "TMPDIR", "TEMP", "TMP")
               if key in os.environ}
        env.update({"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "LLM_PYTHON_MODEL_ROOT": str(self.root), "LLM_PYTHON_WORKER_CONFIG": json.dumps(self.config, separators=(",", ":"), allow_nan=False)})
        proc = None
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(*self.command,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=env, start_new_session=True))
        try:
            proc = await asyncio.shield(spawn)
            # Registration and identity capture are deliberately adjacent.  A cancellation here
            # is still cleaned through the local proc reference.
            self.process, self._pgid, self.child_identity = proc, proc.pid, self._identity(proc.pid)
            self._assert_leader()
            self._stderr_task = asyncio.create_task(self._drain_stderr(proc))
        except asyncio.CancelledError:
            # create_subprocess_exec can have spawned the OS child before its await
            # returns.  Collect the shielded result despite repeated cancellation,
            # register it, then complete the same identity-fenced cleanup path.
            while not spawn.done():
                try: await asyncio.shield(spawn)
                except asyncio.CancelledError: continue
            proc = spawn.result()
            self.process, self._pgid, self.child_identity = proc, proc.pid, self._identity(proc.pid)
            self._stderr_task = asyncio.create_task(self._drain_stderr(proc))
            await self.close()
            raise
        except BaseException:
            if not spawn.done():
                spawn.cancel()
                await asyncio.gather(spawn, return_exceptions=True)
            if proc is not None and self.process is None:
                self.process, self._pgid = proc, proc.pid
                try: self.child_identity = self._identity(proc.pid)
                except RuntimeError: pass
            await self.close()
            raise

    async def _drain_stderr(self, proc) -> None:
        total = 0
        try:
            while chunk := await proc.stderr.read(8192):
                total += len(chunk)
                if total > self.frame_limit: raise RuntimeError("worker stderr exceeds bound")
        except BaseException as exc:
            self._failed = exc
            raise

    async def _wait(self, awaitable):
        return await asyncio.wait_for(awaitable, self.timeout)

    async def _response(self, p):
        header = await p.stdout.readexactly(4)
        size = struct.unpack(">I", header)[0]
        if not 0 < size <= self.frame_limit: raise RuntimeError("RPC response exceeds bound")
        return await p.stdout.readexactly(size)

    async def _await_response_or_stderr_failure(self, p) -> bytes:
        response = asyncio.create_task(self._response(p))
        deadline = time.monotonic() + self.timeout
        try:
            while not response.done():
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise asyncio.TimeoutError
                watchers = {response}
                if self._stderr_task is not None and not self._stderr_task.done(): watchers.add(self._stderr_task)
                done, _ = await asyncio.wait(watchers, timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED)
                if not done: raise asyncio.TimeoutError
                if self._stderr_task in done:
                    # EOF is harmless, but a drain error is an immediate poisoned transport.
                    self._stderr_task.result()
                if self._failed is not None: raise RuntimeError("worker stderr transport failed") from self._failed
            return response.result()
        finally:
            if not response.done():
                response.cancel()
                await asyncio.gather(response, return_exceptions=True)

    async def call(self, operation: str, **values):
        async with self._lock:
            if self._failed is not None: await self.close(); raise RuntimeError("worker stderr transport failed") from self._failed
            p = self.process
            if p is None or p.returncode is not None: await self.close(); raise RuntimeError("model worker is dead")
            self._assert_leader()
            self._serial += 1
            ident = self._serial
            body = json.dumps({"id": ident, "op": operation, **values}, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
            if len(body) > self.frame_limit: raise ValueError("RPC request exceeds bound")
            try:
                p.stdin.write(struct.pack(">I", len(body)) + body)
                await self._wait(p.stdin.drain())
                raw = await self._await_response_or_stderr_failure(p)
                response = json.loads(raw.decode("utf-8"), object_pairs_hook=self._pairs,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))
            except BaseException as exc:
                self._failed = exc
                await self.close()
                if isinstance(exc, asyncio.CancelledError): raise
                raise RuntimeError("model worker transport failed") from exc
            if not isinstance(response, dict) or response.get("id") != ident:
                self._failed = RuntimeError("malformed worker response")
                await self.close()
                raise self._failed
            if response.get("ok") is False and set(response)=={"id","ok","error"} and response["error"] in {"worker_operation_failed","gpu_mig_api_unavailable","gpu_mig_api_failed","gpu_identity_mismatch"}:
                self._failed=RuntimeError("worker operation failed: "+response["error"])
                await self.close()
                raise self._failed
            if set(response) != {"id", "ok", "value"} or response.get("ok") is not True:
                self._failed = RuntimeError("malformed worker response")
                await self.close()
                raise self._failed
            return response["value"]

    @staticmethod
    def _pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def _signal_owned(self, sig: signal.Signals) -> None:
        self._assert_leader()
        try: os.killpg(self._pgid, sig)  # type: ignore[arg-type]
        except ProcessLookupError: pass

    def _orphan_group_members(self) -> tuple[ProcessIdentity, ...]:
        """Prove a surviving saved group is still adopted service work.

        This is only used after the session leader has exited.  We never signal a
        bare recycled PGID: every observed member must still be in the saved
        session/group and must be an immediate child of our subreaper.
        """
        if self._pgid is None: raise RuntimeError("worker group is unavailable")
        members = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal(): continue
            try:
                raw = (entry / "stat").read_bytes()
                fields = raw[raw.rfind(b")") + 2:].split()
                pid, parent, pgrp, session, start = int(raw[:raw.index(b"(")]), int(fields[1]), int(fields[2]), int(fields[3]), int(fields[19])
            except (OSError, ValueError, IndexError): continue
            if pgrp == self._pgid or session == self._pgid:
                if pgrp != self._pgid or session != self._pgid or parent != os.getpid():
                    raise RuntimeError("orphan worker group ownership cannot be proved")
                members.append(ProcessIdentity(pid, start))
        if not members: raise RuntimeError("orphan worker group disappeared without proof")
        # Re-read the complete membership snapshot immediately before signalling.
        for member in members:
            if self._identity(member.pid) != member: raise RuntimeError("orphan worker PID was reused")
        return tuple(members)

    async def _group_gone(self) -> bool:
        if self._pgid is None: return False
        try: os.killpg(self._pgid, 0)
        except ProcessLookupError: return True
        except PermissionError: return False
        return False

    async def _settle_group(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if await self._group_gone(): return True
            if self.process is not None and self.process.returncode is not None:
                while True:
                    try: pid, _ = os.waitpid(-self._pgid, os.WNOHANG)  # type: ignore[arg-type]
                    except ChildProcessError: break
                    if pid == 0: break
            await asyncio.sleep(.02)
        return await self._group_gone()

    async def close(self) -> None:
        if self.process is None: return
        if self._closing: return
        self._closing = True
        clean = False
        try:
            p = self.process
            if p.returncode is None:
                try: self._signal_owned(signal.SIGTERM)
                except RuntimeError: raise RuntimeError("cannot safely signal worker process group")
            elif not await self._group_gone():
                self._orphan_group_members()
                os.killpg(self._pgid, signal.SIGTERM)
            if not await self._settle_group(2):
                if p.returncode is None:
                    try: self._signal_owned(signal.SIGKILL)
                    except RuntimeError: raise RuntimeError("cannot safely kill worker process group")
                else:
                    self._orphan_group_members()
                    os.killpg(self._pgid, signal.SIGKILL)
                if not await self._settle_group(2): raise RuntimeError("worker process group survived cleanup")
            try: await asyncio.wait_for(p.wait(), 2)
            except asyncio.TimeoutError: raise RuntimeError("worker leader did not exit")
            if self._stderr_task is not None:
                try: await asyncio.wait_for(self._stderr_task, 2)
                except asyncio.TimeoutError: raise RuntimeError("worker stderr did not close")
                except BaseException:
                    if self._failed is None: raise
            cleanup = self.gpu_proof.cleanup()
            if inspect.isawaitable(cleanup): cleanup = await asyncio.wait_for(cleanup, self.timeout)
            if cleanup is not True: raise RuntimeError("GPU cleanup could not be proved")
            clean = True
        finally:
            self._closing = False
            # Retain all fences after any uncertainty: a subsequent load must not replace them.
            if clean:
                self.process = None; self.child_identity = None; self._pgid = None; self._stderr_task = None

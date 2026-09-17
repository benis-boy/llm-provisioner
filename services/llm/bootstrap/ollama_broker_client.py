"""Bounded capability-free client for the fixed root Ollama broker."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import select
import subprocess
import threading
import time

from services.llm.providers.gpu import OwnedOllamaSnapshot, ProcessIdentity

class OllamaBrokerError(RuntimeError): pass

@dataclass
class BrokerClient:
    expected_supervisor: ProcessIdentity
    timeout: float = 5.0
    gateway: str = "/usr/local/bin/llm-ollama-launch"

    def __post_init__(self) -> None:
        if (type(self.expected_supervisor) is not ProcessIdentity or isinstance(self.timeout, bool)
                or not isinstance(self.timeout, (int, float)) or not math.isfinite(self.timeout)
                or not 0 < self.timeout <= 300):
            raise TypeError("valid broker authority is required")
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock, self._closed, self._broken = threading.Lock(), False, False
        self._snapshot: OwnedOllamaSnapshot | None = None
        self._close_task = None
        self._broker_identity = None
        self._daemon_identity = None

    def _break(self) -> None:
        self._broken = True
        proc = self._proc  # retain the child until its exit is collected
        if proc is not None:
            for stream in (proc.stdin, proc.stdout):
                if stream is not None: stream.close()

    def _rpc(self, command: str) -> dict:
        if command not in {"start", "snapshot", "stop", "ping"}: raise OllamaBrokerError("unsupported broker command")
        limit = max(self.timeout, 35) if command == "start" and self.timeout == 5 else self.timeout
        if not self._lock.acquire(timeout=limit): raise OllamaBrokerError("broker is busy")
        try:
            if self._closed or self._broken: raise OllamaBrokerError("broker is unavailable")
            if self._proc is None:
                if command != "start" or not Path(self.gateway).is_file() or not os.access(self.gateway, os.X_OK): raise OllamaBrokerError("fixed Ollama gateway is unavailable")
                try:
                    self._proc = subprocess.Popen([self.gateway], cwd="/opt/llm", env={"PATH":"/usr/bin:/bin","HOME":"/var/empty"}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, close_fds=True)
                    self._broker_identity = _identity(self._proc.pid)
                    assert self._proc.stdin is not None and self._proc.stdout is not None
                    os.set_blocking(self._proc.stdin.fileno(), False)
                    os.set_blocking(self._proc.stdout.fileno(), False)
                except Exception as exc:
                    # A partially initialized gateway is still our child.  Keep
                    # it retained for close(), even when procfs/fd setup fails.
                    self._break()
                    raise OllamaBrokerError("broker startup failed") from exc
            proc = self._proc
            assert proc.stdin is not None and proc.stdout is not None
            try:
                deadline, raw = time.monotonic() + limit, bytearray()
                message = (json.dumps({"command":command}, separators=(",", ":"))+"\n").encode("ascii")
                if not select.select([], [proc.stdin.fileno()], [], limit)[1]:
                    raise OllamaBrokerError("broker write timed out")
                if os.write(proc.stdin.fileno(), message) != len(message):
                    raise OllamaBrokerError("broker write incomplete")
                fd = proc.stdout.fileno()
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0: raise OllamaBrokerError("broker response timed out")
                    readable, _, _ = select.select([fd], [], [], remaining)
                    if not readable: raise OllamaBrokerError("broker response timed out")
                    part = os.read(fd, min(4097 - len(raw), 512))
                    if not part: raise OllamaBrokerError("broker closed unexpectedly")
                    raw.extend(part)
                    if len(raw) > 4096 or raw.count(b"\n") > 1: raise OllamaBrokerError("broker response is malformed")
                    if raw.endswith(b"\n"): break
                value = json.loads(bytes(raw[:-1]).decode("ascii"), object_pairs_hook=_pairs)
                if type(value) is not dict or type(value.get("ok")) is not bool or not value["ok"]: raise OllamaBrokerError("broker rejected request")
                if command in {"start", "snapshot"}: self._verify(value)
                if command in {"start", "ping"} and value.get("version") != "0.11.6":
                    raise OllamaBrokerError("unexpected pinned Ollama version")
                return value
            except Exception as exc:
                self._break()
                if isinstance(exc, OllamaBrokerError): raise
                raise OllamaBrokerError("broker protocol failure") from exc
        finally: self._lock.release()

    @staticmethod
    def _identity(value: object) -> ProcessIdentity:
        if not isinstance(value, dict) or set(value) != {"pid", "start"} or type(value["pid"]) is not int or type(value["start"]) is not int or value["pid"] <= 0 or value["start"] <= 0: raise ValueError
        return ProcessIdentity(value["pid"], value["start"])

    def _verify(self, value: dict) -> None:
        broker, daemon = self._identity(value.get("broker")), self._identity(value.get("daemon"))
        items = value.get("descendants")
        if type(items) is not list: raise OllamaBrokerError("broker descendants are invalid")
        descendants = tuple(sorted((self._identity(item) for item in items), key=lambda x:(x.pid,x.start_time)))
        proc = self._proc
        if (proc is None or broker.pid != proc.pid or _identity(proc.pid) != broker
                or (self._broker_identity is not None and broker != self._broker_identity)
                or _parent(broker.pid) != self.expected_supervisor.pid
                or _identity(self.expected_supervisor.pid) != self.expected_supervisor
                or _parent(daemon.pid) != broker.pid or _identity(daemon.pid) != daemon
                or (self._daemon_identity is not None and daemon != self._daemon_identity)):
            raise OllamaBrokerError("broker ancestry changed")
        identities = (self.expected_supervisor, broker, daemon, *descendants)
        if (len({item.pid for item in identities}) != len(identities)
                or any(_identity(item.pid) != item or not _descends(item.pid, daemon.pid)
                       for item in descendants)):
            raise OllamaBrokerError("daemon descendants are invalid")
        # Re-read both sides after walking ancestry to fence PID reuse/races.
        if any(_identity(item.pid) != item for item in identities):
            raise OllamaBrokerError("broker identities changed")
        self._daemon_identity = daemon
        self._snapshot = OwnedOllamaSnapshot(self.expected_supervisor, daemon, descendants, broker)

    async def start(self) -> str:
        value = await asyncio.to_thread(self._rpc, "start"); version=value.get("version")
        if version != "0.11.6": raise OllamaBrokerError("unexpected pinned Ollama version")
        return version
    async def health(self) -> str:
        value=await asyncio.to_thread(self._rpc,"ping"); version=value.get("version")
        if version != "0.11.6": raise OllamaBrokerError("broker health is invalid")
        return version
    async def alive(self) -> bool:
        try: await asyncio.to_thread(self._rpc,"snapshot"); return True
        except OllamaBrokerError: return False
    def ownership_snapshot(self) -> OwnedOllamaSnapshot:
        self._rpc("snapshot")
        if self._snapshot is None: raise OllamaBrokerError("ownership snapshot unavailable")
        return self._snapshot
    def _close(self) -> None:
        # Serialize with a retained start/RPC worker, including cancelled callers.
        if not self._lock.acquire(timeout=max(self.timeout, 40)):
            raise OllamaBrokerError("broker worker did not settle")
        try:
            proc = self._proc
            self._closed = True
            if proc is None:
                return
            # EOF is authoritative revocation, even after a broken response.
            self._break()
            try:
                code = proc.wait(timeout=max(self.timeout, 8) if self.timeout == 5 else self.timeout)
            except subprocess.TimeoutExpired as exc:
                # Collect an unresponsive broker but never call this proved
                # daemon cleanup: root may have died before stopping its child.
                try:
                    proc.kill()
                    proc.wait(timeout=self.timeout)
                except (PermissionError, subprocess.TimeoutExpired, OSError) as kill_exc:
                    # A same-UID client may be unable to signal the privileged
                    # broker.  It remains retained and cleanup is unproved.
                    raise OllamaBrokerError("broker cleanup ownership is uncertain") from kill_exc
                self._proc = None
                raise OllamaBrokerError("broker cleanup exceeded its bound") from exc
            self._proc = None
            if code != 0:
                raise OllamaBrokerError("broker cleanup failed")
        finally:
            self._lock.release()

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(asyncio.to_thread(self._close))
        cancelled = False
        while True:
            try:
                await asyncio.shield(self._close_task)
                break
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

def _identity(pid: int) -> ProcessIdentity:
    raw=(Path("/proc")/str(pid)/"stat").read_bytes(); return ProcessIdentity(pid,int(raw[raw.rfind(b")")+2:].split()[19]))
def _parent(pid: int) -> int:
    raw=(Path("/proc")/str(pid)/"stat").read_bytes(); return int(raw[raw.rfind(b")")+2:].split()[1])
def _descends(pid: int, ancestor: int) -> bool:
    seen = set()
    for _ in range(64):
        if pid == ancestor: return True
        if pid <= 1 or pid in seen:
            return False
        seen.add(pid)
        identity = _identity(pid)
        parent = _parent(pid)
        if _identity(pid) != identity:
            return False
        pid = parent
    return False

def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate broker field")
        result[key] = value
    return result

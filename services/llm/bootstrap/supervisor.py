"""Fail-closed ownership of one private, loopback-only Ollama daemon."""
from __future__ import annotations

import asyncio
import ctypes
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import pwd
import signal
import stat
import sys
import time
from typing import Any

import aiohttp

from .config import BootstrapConfig
from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import ProcessIdentity, OwnedOllamaSnapshot
from .ollama_broker_client import BrokerClient


class OllamaSupervisorError(RuntimeError):
    """Ownership, readiness, or cleanup could not be proved."""


def _identity(pid: int) -> ProcessIdentity:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_bytes()
        opening, closing = raw.index(b"("), raw.rfind(b")")
        fields = raw[closing + 2:].split()
        if int(raw[:opening]) != pid or len(fields) <= 19 or int(fields[19]) < 0:
            raise ValueError
        return ProcessIdentity(pid, int(fields[19]))
    except (OSError, ValueError, IndexError) as exc:
        raise OllamaSupervisorError("process identity is unavailable") from exc


@dataclass(frozen=True)
class _Fence:
    identity: ProcessIdentity
    pgrp: int
    session: int


def _fence(pid: int) -> _Fence:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_bytes()
        closing = raw.rfind(b")")
        fields = raw[closing + 2:].split()
        return _Fence(_identity(pid), int(fields[2]), int(fields[3]))
    except (OSError, ValueError, IndexError) as exc:
        raise OllamaSupervisorError("process group identity is unavailable") from exc


class OwnedOllama:
    """Own one Ollama process group.

    ``await start()`` returns the verified version. ``await alive()`` performs a
    current identity/group check. ``await wait()`` waits for leader exit without
    waiting on inherited pipes, then cleans owned descendants. ``await close()``
    is idempotent and concurrent-safe; uncertain cleanup retains every fence and
    permits a later retry, never a replacement start.
    """

    def __init__(self, config: BootstrapConfig, gpu_proof: GPUProof, *,
                 startup_timeout: float = 30.0, command: list[str] | None = None,
                 output_limit: int = 256 * 1024, launch_user: bool = False) -> None:
        if type(config) is not BootstrapConfig:
            raise TypeError("validated BootstrapConfig is required")
        if type(gpu_proof) is not GPUProof or type(gpu_proof.expected_supervisor) is not ProcessIdentity:
            raise ValueError("captured GPU proof with current parent identity is required")
        if gpu_proof.expected_supervisor != _identity(os.getpid()):
            raise ValueError("GPU proof supervisor identity is not the current parent")
        if sys.platform != "linux":
            raise ValueError("private Ollama ownership requires Linux")
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise ValueError("private Ollama ownership requires Linux pidfds")
        if (isinstance(startup_timeout, bool) or not isinstance(startup_timeout, (int, float)) or
                not math.isfinite(startup_timeout) or startup_timeout <= 0 or startup_timeout > 300):
            raise ValueError("startup timeout is out of bounds")
        if type(output_limit) is not int or not 1024 <= output_limit <= 4 * 1024 * 1024:
            raise ValueError("output limit is out of bounds")
        self.config, self.gpu_proof = config, gpu_proof
        self.startup_timeout, self.output_limit = float(startup_timeout), output_limit
        if type(launch_user) is not bool:
            raise ValueError("launch_user must be boolean")
        self.launch_user = launch_user
        self.command = tuple(command or (("/usr/local/bin/llm-ollama-launch",)
                                         if Path("/usr/local/bin/llm-ollama-launch").exists()
                                         else (str(config.ollama_binary), "serve")))
        if not self.command or any(not isinstance(x, str) or not x for x in self.command):
            raise ValueError("invalid Ollama command")
        # The image gateway is a complete, fixed root broker (not an Ollama
        # executable).  Keep the established direct supervisor for tools and
        # tests that provide a command, while production uses the broker.
        self._broker = (BrokerClient(gpu_proof.expected_supervisor)
                         if command is None and Path("/usr/local/bin/llm-ollama-launch").exists()
                         else None)
        if self._broker is not None and (
                config.ollama_binary != Path("/usr/local/bin/ollama") or
                config.ollama_home != Path("/var/lib/ollama") or config.ollama_port != 11434 or
                config.models["SmolLM"].runtime_identity != "ollama:0.11.6"):
            raise ValueError("image Ollama configuration must match the fixed broker")
        self.process: asyncio.subprocess.Process | None = None
        self._fence: _Fence | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._output_failed: BaseException | None = None
        self._state = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._starting = False
        self._started = False
        self._spawn: asyncio.Task[asyncio.subprocess.Process] | None = None
        self._late_cleanup: asyncio.Task[None] | None = None
        self._cleanup_grace = 2.0
        self._pidfds: dict[ProcessIdentity, int] = {}

    @property
    def port(self) -> int:
        return self.config.ollama_port

    def _assert_owned(self) -> _Fence:
        fence = self._fence
        if self.process is None or fence is None:
            raise OllamaSupervisorError("Ollama ownership is unavailable")
        current = _fence(self.process.pid)
        if current != fence or current.identity != fence.identity:
            raise OllamaSupervisorError("Ollama PID or process-group identity changed")
        return current

    def _env(self) -> dict[str, str]:
        env = {key: os.environ[key] for key in
               ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES", "TMPDIR", "TEMP", "TMP") if key in os.environ}
        home = self.config.ollama_home
        models = home / "models"
        if self.launch_user or self.command[0] == "/usr/local/bin/llm-ollama-launch":
            try:
                owner = pwd.getpwnam("ollama").pw_uid
                for path in (home, models):
                    info = path.stat()
                    if info.st_uid != owner or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                        raise OllamaSupervisorError("Ollama state ownership or mode is unsafe")
            except (KeyError, OSError) as exc:
                raise OllamaSupervisorError("Ollama state is unavailable or unsafe") from exc
        else:
            home.mkdir(mode=0o700, parents=True, exist_ok=True)
            models.mkdir(mode=0o700, parents=True, exist_ok=True)
            home.chmod(0o700)
        env.update({"PATH": env.get("PATH", "/usr/bin:/bin"), "HOME": str(home),
                    "OLLAMA_MODELS": str(models), "OLLAMA_HOST": f"127.0.0.1:{self.port}",
                    "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1"})
        return env

    async def _drain(self, stream: asyncio.StreamReader) -> None:
        # Drain forever. Keep no prompt/log content and do not impose a lifetime quota.
        try:
            while await stream.read(8192):
                pass
        except BaseException as exc:
            self._output_failed = exc
            raise

    @staticmethod
    def _launcher_record(raw: bytes) -> _Fence:
        try:
            value = json.loads(raw.decode("ascii"))
            if (not isinstance(value, dict) or set(value) != {"pid", "start", "pgrp", "session"}
                    or any(type(value[key]) is not int or value[key] < 0 for key in value)):
                raise ValueError
            return _Fence(ProcessIdentity(value["pid"], value["start"]), value["pgrp"], value["session"])
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise OllamaSupervisorError("Ollama launcher handshake is malformed") from exc

    async def _gate_launcher(self, proc: asyncio.subprocess.Process, deadline: float) -> _Fence:
        try:
            raw = await asyncio.wait_for(proc.stdout.readline(), max(.001, deadline - time.monotonic()))
        except asyncio.TimeoutError as exc:
            raise OllamaSupervisorError("Ollama launcher handshake timed out") from exc
        if not raw or len(raw) > 256:
            raise OllamaSupervisorError("Ollama launcher handshake is unavailable")
        announced = self._launcher_record(raw.rstrip(b"\n"))
        actual = _fence(proc.pid)
        if announced != actual or actual.pgrp != proc.pid or actual.session != proc.pid:
            raise OllamaSupervisorError("Ollama launcher identity is not private")
        # Publish the verified fence before opening the gate: cancellation after
        # this point must use normal fenced group cleanup, never gate denial.
        self._fence = actual
        self._open_pidfd(actual)
        if proc.stdin is None:
            raise OllamaSupervisorError("Ollama launcher gate is unavailable")
        proc.stdin.write(b"G")
        await asyncio.wait_for(proc.stdin.drain(), max(.001, deadline - time.monotonic()))
        proc.stdin.close()
        return actual

    async def _deny_launcher(self, proc: asyncio.subprocess.Process) -> None:
        """Close the trusted pre-exec gate and fully collect its transports."""
        if proc.stdin is not None:
            proc.stdin.close()
            try:
                await asyncio.wait_for(proc.stdin.wait_closed(), 1)
            except (asyncio.TimeoutError, ConnectionError):
                pass
        try:
            await asyncio.wait_for(proc.wait(), 2)
        except asyncio.TimeoutError as exc:
            raise OllamaSupervisorError("Ollama launcher did not exit after gate denial") from exc
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                await stream.read()

    async def _complete_cleanup(self, task: asyncio.Task[None]) -> bool:
        """Wait through repeated cancellation; report it only after cleanup."""
        cancelled = False
        while True:
            try:
                await asyncio.shield(task)
                return cancelled
            except asyncio.CancelledError:
                cancelled = True

    def _retain_gate_denial(self, proc: asyncio.subprocess.Process) -> asyncio.Task[None]:
        """Create exactly one cancellation-shielded pre-exec cleanup owner."""
        if self._late_cleanup is None:
            self._late_cleanup = asyncio.create_task(self._deny_launcher(proc))
        return self._late_cleanup

    def _retain_owned_cleanup(self) -> asyncio.Task[None]:
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._close_owned())
        return self._close_task

    def _open_pidfd(self, fence: _Fence) -> None:
        """Pin an identity before it is eligible for a destructive signal."""
        try:
            fd = os.pidfd_open(fence.identity.pid, 0)
        except OSError as exc:
            raise OllamaSupervisorError("Ollama pidfd is unavailable") from exc
        try:
            if _fence(fence.identity.pid) != fence:
                raise OllamaSupervisorError("Ollama process identity changed")
        except BaseException:
            os.close(fd)
            raise
        old = self._pidfds.setdefault(fence.identity, fd)
        if old != fd:
            os.close(fd)

    def _signal(self, identities: tuple[ProcessIdentity, ...], sig: signal.Signals) -> None:
        for identity in identities:
            fd = self._pidfds.get(identity)
            if fd is None or _identity(identity.pid) != identity:
                raise OllamaSupervisorError("Ollama process identity changed")
            try:
                signal.pidfd_send_signal(fd, sig)
            except ProcessLookupError:
                continue
            except OSError as exc:
                raise OllamaSupervisorError("Ollama pidfd signal failed") from exc

    def _close_pidfds(self) -> None:
        for fd in self._pidfds.values():
            try: os.close(fd)
            except OSError: pass
        self._pidfds.clear()

    @staticmethod
    def _proc_net_inode(port: int, *, loopback_only: bool = True) -> set[int]:
        wanted = f"{port:04X}"
        result: set[int] = set()
        try:
            lines = (Path("/proc/net/tcp").read_text().splitlines()[1:] +
                     Path("/proc/net/tcp6").read_text().splitlines()[1:])
        except OSError as exc:
            raise OllamaSupervisorError("authoritative socket table is unavailable") from exc
        for line in lines:
            fields = line.split()
            local = fields[1].split(":", 1)[0]
            if (len(fields) >= 10 and fields[1].rsplit(":", 1)[-1] == wanted and fields[3] == "0A"
                    and (not loopback_only or local in {"0100007F", "00000000000000000000000000000001"})):
                try: result.add(int(fields[9]))
                except ValueError: raise OllamaSupervisorError("malformed socket table")
        return result

    def _owned_listener(self) -> bool:
        if self.process is None or self._fence is None:
            return False
        inodes = self._proc_net_inode(self.port)
        if not inodes:
            return False
        try:
            fd_dir = Path("/proc") / str(self.process.pid) / "fd"
            # Linux gates /proc/PID/fd with filesystem credentials.  The check
            # compares both fsuid and fsgid, so change the fixed child's pair on
            # this thread only; no application capability is retained.
            libc = ctypes.CDLL(None, use_errno=True) if self.launch_user else None
            previous_uid = previous_gid = None
            try:
                if libc is not None:
                    account = pwd.getpwnam("ollama")
                    previous_gid = libc.setfsgid(account.pw_gid)
                    if libc.setfsgid(-1) != account.pw_gid:
                        raise OllamaSupervisorError("Ollama filesystem group identity unavailable")
                    previous_uid = libc.setfsuid(account.pw_uid)
                    if libc.setfsuid(-1) != account.pw_uid:
                        raise OllamaSupervisorError("Ollama filesystem identity unavailable")
                found = {os.readlink(fd) for fd in fd_dir.iterdir() if fd.is_symlink()}
            finally:
                if libc is not None:
                    restore_error = False
                    if previous_uid is not None:
                        libc.setfsuid(previous_uid)
                        restore_error |= libc.setfsuid(-1) != previous_uid
                    if previous_gid is not None:
                        libc.setfsgid(previous_gid)
                        restore_error |= libc.setfsgid(-1) != previous_gid
                    if restore_error:
                        raise OllamaSupervisorError("broker filesystem identity restoration failed")
        except OSError:
            return False
        owned = {"socket:[%d]" % inode for inode in inodes}
        return bool(owned & found) and self._assert_owned() == self._fence

    async def _health(self, remaining: float) -> str:
        expected = self.config.models["SmolLM"].runtime_identity
        if not expected.startswith("ollama:"):
            raise OllamaSupervisorError("SmolLM runtime identity is not an Ollama identity")
        if not self._owned_listener():
            raise OllamaSupervisorError("loopback listener is not proved to be owned")
        timeout = aiohttp.ClientTimeout(total=max(.001, remaining), connect=max(.001, remaining))
        try:
            async with aiohttp.ClientSession(trust_env=False, timeout=timeout) as session:
                async with session.get(f"http://127.0.0.1:{self.port}/api/version",
                                       allow_redirects=False) as response:
                    if response.status != 200 or response.history:
                        raise OllamaSupervisorError("Ollama health endpoint is not ready")
                    if response.content_length is not None and response.content_length > 4096:
                        raise OllamaSupervisorError("Ollama health response exceeds bound")
                    data = bytearray()
                    async for chunk in response.content.iter_chunked(1024):
                        data.extend(chunk)
                        if len(data) > 4096:
                            raise OllamaSupervisorError("Ollama health response exceeds bound")
                    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
                        result: dict[str, Any] = {}
                        for key, item in items:
                            if key in result:
                                raise ValueError("duplicate health field")
                            result[key] = item
                        return result
                    value = json.loads(bytes(data).decode("utf-8"), object_pairs_hook=pairs)
        except (aiohttp.ClientError, asyncio.TimeoutError, UnicodeError, ValueError) as exc:
            raise OllamaSupervisorError("Ollama health request failed") from exc
        if not isinstance(value, dict) or set(value) != {"version"} or value["version"] != expected[7:]:
            raise OllamaSupervisorError("unexpected Ollama version")
        if not self._owned_listener():
            raise OllamaSupervisorError("loopback listener ownership changed")
        self._members(allow_live=True)
        return expected[7:]

    async def health(self, remaining: float = 5.0) -> str:
        """Return the currently observed, owned daemon version.

        This is deliberately a readiness-only seam: it does not load a model
        or otherwise change daemon residency.
        """
        if self._broker is not None:
            return await self._broker.health()
        return await self._health(remaining)

    async def start(self) -> str:
        if self._broker is not None:
            return await self._broker.start()
        async with self._state:
            if self._started or self.process is not None or self._starting or self._spawn is not None:
                raise OllamaSupervisorError("Ollama ownership is not clean")
            # Do not race a foreign listener.  In particular, never start an
            # executable merely to discover that it could not bind this port.
            if self._proc_net_inode(self.port, loopback_only=False):
                raise OllamaSupervisorError("Ollama port is already occupied")
            self._starting = True
            self._started = True
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(36, 1, 0, 0, 0) != 0:
                self._starting = False
                raise OllamaSupervisorError("could not establish child subreaper")
            deadline = time.monotonic() + self.startup_timeout
            kwargs = {}
            if self.launch_user:
                try:
                    account = pwd.getpwnam("ollama")
                except KeyError as exc:
                    raise OllamaSupervisorError("ollama service identity is unavailable") from exc
                kwargs.update(user=account.pw_uid, group=account.pw_gid,
                              extra_groups=os.getgrouplist(account.pw_name, account.pw_gid))
            launcher_args = ()
            if self.launch_user:
                # This is an internal root-to-ollama handoff, not caller input.
                # The gate rechecks this exact broker identity around PDEATHSIG.
                parent = _identity(os.getpid())
                launcher_args = ("--expected-parent", str(parent.pid),
                                 str(parent.start_time), "--")
            spawn = asyncio.create_task(asyncio.create_subprocess_exec(
                sys.executable, "-m", "services.llm.bootstrap.ollama_launcher", *launcher_args,
                *self.command,
                env=self._env(), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, start_new_session=True, **kwargs))
            self._spawn = spawn
            try:
                proc = await asyncio.wait_for(asyncio.shield(spawn), max(.001, deadline - time.monotonic()))
                self.process = proc
                try:
                    await self._gate_launcher(proc, deadline)
                finally:
                    if self._fence is not None:
                        self._stdout_task = asyncio.create_task(self._drain(proc.stdout))
                        self._stderr_task = asyncio.create_task(self._drain(proc.stderr))
                while time.monotonic() < deadline:
                    if proc.returncode is not None:
                        raise OllamaSupervisorError("Ollama exited during startup")
                    if self._output_failed is not None:
                        raise OllamaSupervisorError("Ollama output transport failed")
                    self._assert_owned()
                    try:
                        return await self._health(deadline - time.monotonic())
                    except OllamaSupervisorError as exc:
                        if time.monotonic() >= deadline:
                            raise OllamaSupervisorError("Ollama readiness timed out") from exc
                        await asyncio.sleep(min(.05, deadline - time.monotonic()))
                raise OllamaSupervisorError("Ollama readiness timed out")
            except asyncio.CancelledError:
                await self._finish_spawn(spawn, bounded=True)
                try:
                    if self._fence is None and self.process is not None:
                        task = self._retain_gate_denial(self.process)
                        await self._complete_cleanup(task)
                        self.process = None
                    else:
                        await self._complete_cleanup(self._retain_owned_cleanup())
                finally:
                    self._starting = False
                raise
            except BaseException:
                await self._finish_spawn(spawn, bounded=True)
                try:
                    if self._fence is None and self.process is not None:
                        task = self._retain_gate_denial(self.process)
                        await self._complete_cleanup(task)
                        self.process = None
                    else:
                        await self._complete_cleanup(self._retain_owned_cleanup())
                finally:
                    self._starting = False
                raise
            finally:
                self._starting = False

    async def _finish_spawn(self, spawn: asyncio.Task, *, bounded: bool) -> None:
        if not spawn.done():
            if bounded:
                # Retain this explicit ownership task.  Its completion performs
                # gate denial; close can later collect it, and replacement is
                # permanently refused by one-shot lifecycle state.
                try:
                    await asyncio.wait_for(asyncio.shield(spawn), self._cleanup_grace)
                except asyncio.TimeoutError:
                    self._ensure_late_collector(spawn)
                    return
            while True:
                try:
                    await asyncio.shield(spawn)
                    break
                except asyncio.CancelledError:
                    continue
        if self.process is None and not spawn.cancelled():
            self.process = spawn.result()
        self._spawn = None

    def _ensure_late_collector(self, spawn: asyncio.Task[asyncio.subprocess.Process]) -> None:
        if self._late_cleanup is not None:
            return
        def collect(done: asyncio.Task[asyncio.subprocess.Process]) -> None:
            if self._late_cleanup is None:
                self._late_cleanup = asyncio.create_task(self._collect_late_spawn(done))
        if spawn.done():
            collect(spawn)
        else:
            spawn.add_done_callback(collect)

    async def _collect_late_spawn(self, spawn: asyncio.Task[asyncio.subprocess.Process]) -> None:
        if spawn.cancelled():
            self._spawn = None
            return
        proc = spawn.result()
        self.process = proc
        await self._deny_launcher(proc)
        self.process = None
        self._spawn = None

    def _members(self, *, allow_live: bool = False) -> tuple[tuple[_Fence, str], ...]:
        if self._fence is None:
            raise OllamaSupervisorError("saved process fence is unavailable")
        records: dict[int, tuple[_Fence, int, str]] = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                raw = (entry / "stat").read_bytes()
                fields = raw[raw.rfind(b")") + 2:].split()
                pid = int(raw[:raw.index(b"(")]); parent, state = int(fields[1]), fields[0].decode("ascii")
                fence = _Fence(ProcessIdentity(pid, int(fields[19])), int(fields[2]), int(fields[3]))
            except (OSError, ValueError, IndexError, OllamaSupervisorError):
                continue
            records[pid] = (fence, parent, state)
        leader = self._fence.identity.pid
        saved = records.get(leader)
        if saved is not None and saved[0] != self._fence:
            raise OllamaSupervisorError("saved Ollama leader identity changed")
        known = {identity.pid for identity in self._pidfds}
        if saved is None and not known:
            raise OllamaSupervisorError("Ollama session continuity is unavailable")
        # Start only from the exact saved leader and already pin-held identities.
        # A pin remains an exact process capability even when its process calls
        # setsid; a numerical old session never grants new authority.
        accepted: set[int] = {leader} if saved is not None else set()
        for identity in self._pidfds:
            record = records.get(identity.pid)
            if record is not None:
                if record[0].identity != identity:
                    raise OllamaSupervisorError("pinned Ollama identity changed")
                accepted.add(identity.pid)
        changed = True
        while changed:
            changed = False
            for pid, (_, parent, _) in records.items():
                # Before a process is captured, it must remain in the original
                # private session.  Captured descendants may subsequently setsid.
                fence = records[pid][0]
                if (parent in accepted and pid not in accepted and
                        (fence.pgrp == self._fence.pgrp and fence.session == self._fence.session)):
                    accepted.add(pid); changed = True
        result = []
        matching = {pid for pid, (fence, _, _) in records.items()
                    if fence.pgrp == self._fence.pgrp and fence.session == self._fence.session}
        if matching - accepted:
            raise OllamaSupervisorError("Ollama session contains an unproved process")
        for pid in accepted:
            if pid not in records: continue
            fence, _, state = records[pid]
            if _fence(pid) != fence: raise OllamaSupervisorError("orphan process identity changed")
            self._open_pidfd(fence)
            result.append((fence, state))
        if not allow_live and self.process is not None and self.process.returncode is not None and not result:
            raise OllamaSupervisorError("owned orphan group disappeared without proof")
        return tuple(result)

    def ownership_snapshot(self) -> OwnedOllamaSnapshot:
        """Return a read-only current daemon/descendant ownership snapshot."""
        if self._broker is not None:
            return self._broker.ownership_snapshot()
        self._assert_owned()
        members = self._members(allow_live=True)
        daemon = self._fence.identity if self._fence is not None else None
        if daemon is None:
            raise OllamaSupervisorError("Ollama ownership is unavailable")
        supervisor = self.gpu_proof.expected_supervisor
        if type(supervisor) is not ProcessIdentity or daemon == supervisor:
            raise OllamaSupervisorError("Ollama ownership supervisor identity is invalid")
        descendants = tuple(sorted((fence.identity for fence, _ in members
                                    if fence.identity != daemon), key=lambda item: (item.pid, item.start_time)))
        if (len({item.pid for item in descendants}) != len(descendants) or
                any(item.pid in {daemon.pid, supervisor.pid} for item in descendants)):
            raise OllamaSupervisorError("Ollama descendant identities are not unique")
        self._assert_owned()
        return OwnedOllamaSnapshot(supervisor, daemon, descendants)

    def _group_gone(self) -> bool:
        if self._fence is None:
            return False
        try:
            os.killpg(self._fence.pgrp, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def _reap_adopted(self, members: tuple[tuple[_Fence, str], ...]) -> None:
        """Reap only positively proved direct children, never a whole PGID."""
        leader = self.process.pid if self.process is not None else None
        for member, state in members:
            if state != "Z" or member.identity.pid == leader:
                continue
            try:
                raw = (Path("/proc") / str(member.identity.pid) / "stat").read_bytes()
                fields = raw[raw.rfind(b")") + 2:].split()
                if int(fields[1]) != os.getpid() or _fence(member.identity.pid) != member:
                    raise OllamaSupervisorError("adopted Ollama child identity changed")
                os.waitpid(member.identity.pid, os.WNOHANG)
            except (ChildProcessError, ProcessLookupError, OSError):
                pass

    async def _settle(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            members = self._members(allow_live=True)
            if not members:
                return True
            # A descendant may still belong to its live leader.  Only after it
            # exits can this subreaper own an adopted zombie; never make that
            # transitional parentage a reason to signal or reject the group.
            if self.process is not None and self.process.returncode is not None:
                self._reap_adopted(members)
            await asyncio.sleep(.02)
        return not self._members(allow_live=True)

    async def _close_owned(self) -> None:
        if self.process is None:
            if self._spawn is not None:
                self._ensure_late_collector(self._spawn)
                if self._late_cleanup is None:
                    raise OllamaSupervisorError("Ollama spawn ownership remains pending")
                try:
                    await asyncio.wait_for(asyncio.shield(self._late_cleanup), self._cleanup_grace)
                except asyncio.TimeoutError as exc:
                    raise OllamaSupervisorError("Ollama spawn ownership remains pending") from exc
            return
        proc, fence = self.process, self._fence
        if fence is None:
            await self._deny_launcher(proc)
            self.process = None
            return
        clean = False
        try:
            members = self._members(allow_live=True)
            self._signal(tuple(member.identity for member, _ in members), signal.SIGTERM)
            if not await self._settle(2):
                members = self._members(allow_live=True)
                self._signal(tuple(member.identity for member, _ in members), signal.SIGKILL)
                if not await self._settle(2):
                    raise OllamaSupervisorError("Ollama process group survived cleanup")
            # Wait only for the leader; descendants may have owned pipes and are
            # accounted for by the group fence, not by EOF on a StreamReader.
            await asyncio.wait_for(proc.wait(), 2)
            for task in (self._stdout_task, self._stderr_task):
                if task is not None and not task.done():
                    task.cancel()
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)
            clean = True
        finally:
            if clean:
                self.process = None
                self._fence = None
                self._close_pidfds()
                self._stdout_task = self._stderr_task = None
                self._output_failed = None

    async def _cancel_safe_cleanup(self) -> None:
        task = asyncio.create_task(self._close_owned())
        while True:
            try:
                await asyncio.shield(task)
                return
            except asyncio.CancelledError:
                continue

    async def close(self) -> None:
        if self._broker is not None:
            await self._broker.close()
            return
        async with self._state:
            task = self._retain_owned_cleanup()
        cancelled = False
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                cancelled = True
                continue
        if task.done() and not task.cancelled() and task.exception() is not None:
            # A failed close is retryable, not a permanently cached failure.
            self._close_task = None
            task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def alive(self) -> bool:
        if self._broker is not None:
            return await self._broker.alive()
        if self.process is None or self._output_failed is not None:
            return False
        try:
            self._assert_owned()
        except OllamaSupervisorError:
            return False
        return self.process.returncode is None

    async def wait(self) -> int:
        proc = self.process
        if proc is None:
            raise OllamaSupervisorError("Ollama is not started")
        while proc.returncode is None:
            await asyncio.sleep(.02)
        result = proc.returncode
        await self.close()
        return result

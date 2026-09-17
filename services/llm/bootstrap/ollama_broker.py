"""Fixed privileged Ollama coordinator, entered only by the image gateway.

No protocol field selects an executable, path, PID, environment or identity.
The application owns the pipe; EOF or a failed channel permanently stops service.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
from pathlib import Path
import pwd
import signal
import stat
import sys
import time

# -I supplies trusted stdlib/site paths. Add only the immutable application.
if __name__ == "__main__":
    sys.path.insert(0, "/opt/llm")

from services.llm.bootstrap.config import BootstrapConfig, ModelConfig
from services.llm.bootstrap.supervisor import OwnedOllama
from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import ProcessIdentity

MAX_LINE = 4096
FRAME_TIMEOUT = 5.0


def _identity(pid: int) -> ProcessIdentity:
    raw = (Path("/proc") / str(pid) / "stat").read_bytes()
    return ProcessIdentity(pid, int(raw[raw.rfind(b")") + 2:].split()[19]))


def _config() -> BootstrapConfig:
    models = {name: ModelConfig("ollama:0.11.6" if name == "SmolLM" else "broker:none",
                                "broker:none") for name in ("SmolLM", "CoEdIT", "GECToR")}
    return BootstrapConfig("GPU-broker", Path("/var/lib/ollama"), "0" * 64,
                           Path("/var/lib/ollama/profiles.sqlite"),
                           Path("/usr/local/bin/ollama"), Path("/var/lib/ollama"), 11434, models)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate protocol field")
        result[key] = value
    return result


def parse_command(raw: bytes) -> str:
    if len(raw) > MAX_LINE or not raw.endswith(b"\n"):
        raise ValueError("invalid protocol frame")
    value = json.loads(raw.decode("ascii"), object_pairs_hook=_pairs)
    if (type(value) is not dict or set(value) != {"command"} or
            value["command"] not in ("start", "snapshot", "stop", "ping")):
        raise ValueError("invalid protocol command")
    return value["command"]


async def read_command(fd: int, stopping: asyncio.Event) -> str | None:
    raw = bytearray()
    deadline = None
    while not stopping.is_set():
        try:
            chunk = os.read(fd, MAX_LINE + 1 - len(raw))
        except BlockingIOError:
            chunk = None
        if chunk == b"":
            return None
        if chunk:
            raw.extend(chunk)
            if deadline is None:
                deadline = time.monotonic() + FRAME_TIMEOUT
            if len(raw) > MAX_LINE:
                raise ValueError("oversized protocol frame")
            if b"\n" in raw:
                return parse_command(bytes(raw))
        if deadline is not None and time.monotonic() >= deadline:
            raise ValueError("incomplete protocol frame")
        await asyncio.sleep(.02)
    return None


async def startup_revocation(fd: int, stopping: asyncio.Event) -> bool:
    """Wait for EOF or the first unexpected byte while a start is in flight.

    It intentionally never buffers a second frame: any byte before the start
    reply is a protocol violation, so no partial frame can be silently dropped.
    """
    while not stopping.is_set():
        try:
            chunk = os.read(fd, 1)
        except BlockingIOError:
            chunk = None
        if chunk == b"":
            return False
        if chunk:
            return True
        await asyncio.sleep(.02)
    return False


async def reply(fd: int, value: dict) -> None:
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("ascii") + b"\n"
    if len(raw) > MAX_LINE:
        raise ValueError("oversized broker response")
    deadline = time.monotonic() + FRAME_TIMEOUT
    while raw:
        try:
            count = os.write(fd, raw)
            raw = raw[count:]
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError("blocked broker response")
            await asyncio.sleep(.02)


def _unavailable(*args, **kwargs):
    raise RuntimeError("broker is not a GPU proof authority")


async def serve(daemon: OwnedOllama, input_fd: int, output_fd: int,
                stopping: asyncio.Event) -> None:
    """Own one lifecycle; internal injectable seam is not part of root protocol."""
    os.set_blocking(input_fd, False)
    os.set_blocking(output_fd, False)
    owner = _identity(os.getpid())
    started = False
    operation = None
    stop_waiter = asyncio.create_task(stopping.wait())
    pending: set[asyncio.Task] = {stop_waiter}

    def watch(coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        pending.add(task)
        task.add_done_callback(pending.discard)
        return task

    try:
        while not stopping.is_set():
            reader = watch(read_command(input_fd, stopping))
            watchers = (reader, stop_waiter)
            lost = None
            if started:
                async def daemon_lost() -> None:
                    while await daemon.alive():
                        await asyncio.sleep(.02)
                lost = watch(daemon_lost())
                watchers += (lost,)
            done, _ = await asyncio.wait(watchers, return_when=asyncio.FIRST_COMPLETED)
            if reader not in done:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
                if lost is not None and not lost.done():
                    lost.cancel()
                    await asyncio.gather(lost, return_exceptions=True)
                if stop_waiter in done:
                    break
                raise RuntimeError("owned daemon lost")
            if lost is not None:
                lost.cancel()
                await asyncio.gather(lost, return_exceptions=True)
            command = reader.result()
            if command is None:
                break
            if command == "stop":
                await daemon.close()
                await reply(output_fd, {"ok": True})
                return
            if command == "start":
                if started:
                    raise ValueError("broker is one-shot")
                started = True
                operation = watch(daemon.start())
                # Do not leave an accepted start running after the application
                # revokes its pipe.  EOF is as authoritative as a signal.
                eof_waiter = watch(startup_revocation(input_fd, stopping))
                done, _ = await asyncio.wait((operation, stop_waiter, eof_waiter),
                                             return_when=asyncio.FIRST_COMPLETED)
                if stop_waiter in done:
                    eof_waiter.cancel()
                    await asyncio.gather(eof_waiter, return_exceptions=True)
                    operation.cancel()
                    await asyncio.gather(operation, return_exceptions=True)
                    break
                if eof_waiter in done:
                    # The protocol is one request at a time.  A second frame
                    # during startup is a violation, and EOF revokes service.
                    if eof_waiter.result():
                        raise ValueError("broker command during startup")
                    operation.cancel()
                    await asyncio.gather(operation, return_exceptions=True)
                    break
                eof_waiter.cancel()
                await asyncio.gather(eof_waiter, return_exceptions=True)
                version = await operation
            else:
                if not started or not await daemon.alive():
                    raise RuntimeError("owned daemon lost")
                version = await daemon.health()
            if command == "ping":
                await reply(output_fd, {"ok": True, "version": version})
                continue
            members = daemon._members(allow_live=True)
            leader = daemon._fence.identity if daemon._fence is not None else None
            if leader is None or not await daemon.alive():
                raise RuntimeError("owned daemon lost")
            def record(identity):
                return {"pid": identity.pid, "start": identity.start_time}
            result = {"ok": True, "broker": record(owner), "daemon": record(leader),
                      "descendants": [record(f.identity) for f, state in members
                                      if f.identity != leader and state != "Z"]}
            if command == "start":
                result["version"] = version
            await reply(output_fd, result)
    finally:
        for task in tuple(pending):
            if not task.done():
                task.cancel()
        await asyncio.gather(*tuple(pending), return_exceptions=True)
        stop_waiter.cancel()
        await asyncio.gather(stop_waiter, return_exceptions=True)
        if operation is not None and not operation.done():
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
        await daemon.close()


async def main() -> None:
    if (os.geteuid() != 0 or os.getuid() != pwd.getpwnam("llm").pw_uid or
            os.getcwd() != "/opt/llm"):
        raise SystemExit(126)
    lock = os.open("/run/llm-ollama-broker.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    info = os.fstat(lock)
    if info.st_uid != 0 or not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        os.close(lock)
        raise SystemExit(126)
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopping.set)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proof = GPUProof(_unavailable, _unavailable, expected_supervisor=_identity(os.getpid()))
        daemon = OwnedOllama(_config(), proof, command=["/usr/local/bin/ollama", "serve"], launch_user=True)
        await serve(daemon, sys.stdin.fileno(), sys.stdout.fileno(), stopping)
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)
        os.close(lock)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        raise SystemExit(1) from None

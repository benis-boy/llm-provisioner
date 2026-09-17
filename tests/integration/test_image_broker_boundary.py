"""Standalone real-image broker boundary check.

Host invocation (the file is piped into the image, not imported as a test)::

  docker run --rm -i --network none --name llm-phase1-broker-boundary-cpu \
    --entrypoint /opt/venv/bin/python llm-provider-foundation:phase1-broker - \
    < tests/integration/test_image_broker_boundary.py

The image's default USER llm is required; this check deliberately does not
use mocks, root, or a replacement daemon.
"""
from __future__ import annotations

import asyncio
import ipaddress
import os
from pathlib import Path
import pwd
import stat
import traceback
from collections.abc import Callable, Iterable

from services.llm.bootstrap.ollama_broker_client import (
    BrokerClient, OllamaBrokerError, _identity,
)


def _credentials(pid: int) -> dict[str, tuple[int, int, int, int]]:
    result: dict[str, tuple[int, int, int, int]] = {}
    for line in Path("/proc", str(pid), "status").read_text().splitlines():
        if line.startswith(("Uid:", "Gid:")):
            fields = line.split()
            result[line[:3].strip()] = tuple(map(int, fields[1:5]))  # type: ignore[assignment]
    if set(result) != {"Uid", "Gid"}:
        raise AssertionError("incomplete process credentials")
    return result


def _listening_addresses(rows: Iterable[str], *, ipv6: bool) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for line in rows:
        fields = line.split()
        if len(fields) < 4 or fields[3] != "0A":  # TCP_LISTEN
            continue
        try:
            address, raw_port = fields[1].split(":", 1)
            if int(raw_port, 16) != 11434:
                continue
            raw_address = bytes.fromhex(address)
            if ipv6:
                if len(raw_address) != 16:
                    continue
                # procfs stores each IPv6 32-bit word in host byte order.
                raw_address = b"".join(
                    raw_address[index:index + 4][::-1]
                    for index in range(0, len(raw_address), 4))
            elif len(raw_address) != 4:
                continue
            addresses.add(ipaddress.ip_address(raw_address[::-1] if not ipv6 else raw_address))
        except (ValueError, ipaddress.AddressValueError):
            continue
    return addresses


def _private_listen(
    tcp_rows: Iterable[str] | None = None,
    tcp6_rows: Iterable[str] | None = None,
) -> bool:
    """Require a loopback LISTEN on 11434 and reject public same-port binds."""
    if tcp_rows is None:
        tcp_rows = Path("/proc/net/tcp").read_text().splitlines()[1:]
    if tcp6_rows is None:
        tcp6_rows = Path("/proc/net/tcp6").read_text().splitlines()[1:]
    found_private = False
    for address in (*_listening_addresses(tcp_rows, ipv6=False),
                    *_listening_addresses(tcp6_rows, ipv6=True)):
        if address.is_loopback:
            found_private = True
        else:
            raise AssertionError("public listener on Ollama port")
    return found_private


def _assert_immutable(
    path: str,
    *,
    lstat: Callable[[str], os.stat_result] = os.lstat,
) -> None:
    current = Path(path)
    for item in (current, *current.parents):
        info = lstat(str(item))
        if stat.S_ISLNK(info.st_mode):
            raise AssertionError(f"symlink in immutable path: {item}")
        if info.st_uid != 0 or info.st_gid != 0:
            raise AssertionError(f"non-root immutable path: {item}")
        if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise AssertionError(f"mutable path: {item}")


async def check() -> None:
    app_uid = os.getuid()
    app_gid = os.getgid()
    app_euid = os.geteuid()
    llm_uid = pwd.getpwnam("llm").pw_uid
    llm_gid = pwd.getpwnam("llm").pw_gid
    ollama_uid = pwd.getpwnam("ollama").pw_uid
    ollama_gid = pwd.getpwnam("ollama").pw_gid
    assert (app_uid, app_gid, app_euid) == (llm_uid, llm_gid, llm_uid) and app_uid != 0

    # Establish the hostile caller context before the first gateway execution.
    os.environ.clear()
    os.environ.update(PATH="/tmp", HOME="/tmp", OLLAMA_HOST="public.invalid:9",
                      OLLAMA_MODELS="/tmp/redirected", PYTHONPATH="/tmp")
    os.chdir("/tmp")

    first = BrokerClient(_identity(os.getpid()))
    snapshot = None
    try:
        version = await first.start()
        health = await first.health()
        snapshot = first.ownership_snapshot()
        broker = _credentials(snapshot.broker.pid)
        daemon = _credentials(snapshot.daemon.pid)
        assert version == health == "0.11.6"
        assert broker["Uid"] == (app_uid, 0, 0, 0), (
            f"broker Uid={broker['Uid']} expected={(app_uid, 0, 0, 0)}")
        assert broker["Gid"] == (app_gid, app_gid, app_gid, app_gid), (
            f"broker Gid={broker['Gid']} expected={(app_gid, app_gid, app_gid, app_gid)}")
        assert daemon["Uid"] == (ollama_uid, ollama_uid, ollama_uid, ollama_uid), (
            f"daemon Uid={daemon['Uid']} expected={(ollama_uid,) * 4}")
        assert daemon["Gid"] == (ollama_gid, ollama_gid, ollama_gid, ollama_gid), (
            f"daemon Gid={daemon['Gid']} expected={(ollama_gid,) * 4}")
        assert _private_listen()
        print("summary app_uid=%d app_gid=%d broker_uid=%s broker_gid=%s "
              "daemon_uid=%s daemon_gid=%s version=%s private_listener=true" %
              (app_uid, app_gid, broker["Uid"], broker["Gid"], daemon["Uid"],
               daemon["Gid"], version), flush=True)

        second = BrokerClient(_identity(os.getpid()), timeout=2)
        rejected = False
        try:
            await second.start()
        except OllamaBrokerError:
            rejected = True
        finally:
            try:
                await second.close()
            except OllamaBrokerError:
                pass
        assert rejected and await first.alive()

        for immutable in ("/opt/llm", "/opt/venv",
                          "/usr/local/bin/llm-ollama-launch",
                          "/usr/local/bin/ollama"):
            _assert_immutable(immutable)
        assert first.gateway == "/usr/local/bin/llm-ollama-launch"
        daemon_pid, broker_pid = snapshot.daemon.pid, snapshot.broker.pid
    finally:
        # Cleanup is mandatory even when startup, identity, or protocol checks fail.
        await first.close()

    assert not Path("/proc", str(daemon_pid)).exists()
    assert not Path("/proc", str(broker_pid)).exists()
    print("summary duplicate_rejected=true cleanup=ok daemon_gone=true broker_gone=true",
          flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(check())
    except Exception:
        traceback.print_exc()
        raise

"""One-shot launch gate for the private Ollama supervisor.

This process deliberately does not execute the configured daemon until its
parent has independently checked the reported procfs fence and writes ``G``.
EOF or any other byte exits without executing arbitrary configured code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import select
import ctypes
import signal


def _record() -> dict[str, int]:
    pid = os.getpid()
    raw = (Path("/proc") / str(pid) / "stat").read_bytes()
    fields = raw[raw.rfind(b")") + 2:].split()
    return {"pid": pid, "start": int(fields[19]), "pgrp": int(fields[2]),
            "session": int(fields[3])}


def _identity(pid: int) -> tuple[int, int]:
    raw = (Path("/proc") / str(pid) / "stat").read_bytes()
    fields = raw[raw.rfind(b")") + 2:].split()
    return pid, int(fields[19])


def _parent_death_guard(expected_pid: int, expected_start: int) -> None:
    """Arm a parent-death signal only for the fixed privileged handoff.

    PDEATHSIG follows this launcher/daemon only.  It does not contain daemon
    grandchildren that subsequently detach or change their parent; supervisor
    cleanup still requires independently proved pidfd-owned descendants.
    """
    if os.getppid() != expected_pid or _identity(expected_pid) != (expected_pid, expected_start):
        raise RuntimeError("expected broker parent is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError("could not arm parent-death signal")
    # prctl and the preceding procfs read are not atomic with parent exit.
    if os.getppid() != expected_pid or _identity(expected_pid) != (expected_pid, expected_start):
        raise RuntimeError("broker parent changed while arming death signal")


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["--expected-parent"]:
        if len(args) < 5 or args[3] != "--":
            raise SystemExit(126)
        try:
            expected_pid, expected_start = int(args[1]), int(args[2])
            # A container application's fixed broker parent may legitimately
            # be PID 1.  Its procfs start time still fences PID reuse.
            if expected_pid <= 0 or expected_start < 0:
                raise ValueError
            _parent_death_guard(expected_pid, expected_start)
        except (OSError, ValueError, RuntimeError):
            raise SystemExit(126) from None
        args = args[4:]
    if not args:
        raise SystemExit(126)
    record = json.dumps(_record(), separators=(",", ":"), allow_nan=False).encode()
    sys.stdout.buffer.write(record + b"\n")
    sys.stdout.buffer.flush()
    # A parent which dies or wedges after spawn must not retain a pre-exec
    # launcher indefinitely.  This is deliberately shorter than supervisor
    # readiness and is only a gate, not daemon startup time.
    ready, _, _ = select.select([sys.stdin.buffer], [], [], 5.0)
    if ready and sys.stdin.buffer.read(1) == b"G":
        os.execvp(args[0], args)


if __name__ == "__main__":
    main()

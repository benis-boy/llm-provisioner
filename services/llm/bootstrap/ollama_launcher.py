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


def _record() -> dict[str, int]:
    pid = os.getpid()
    raw = (Path("/proc") / str(pid) / "stat").read_bytes()
    fields = raw[raw.rfind(b")") + 2:].split()
    return {"pid": pid, "start": int(fields[19]), "pgrp": int(fields[2]),
            "session": int(fields[3])}


def main() -> None:
    record = json.dumps(_record(), separators=(",", ":"), allow_nan=False).encode()
    sys.stdout.buffer.write(record + b"\n")
    sys.stdout.buffer.flush()
    # A parent which dies or wedges after spawn must not retain a pre-exec
    # launcher indefinitely.  This is deliberately shorter than supervisor
    # readiness and is only a gate, not daemon startup time.
    ready, _, _ = select.select([sys.stdin.buffer], [], [], 5.0)
    if ready and sys.stdin.buffer.read(1) == b"G":
        os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()

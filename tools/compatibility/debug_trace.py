"""Opt-in, bounded, secret-free JSONL tracing for compatibility runs."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from functools import wraps
from pathlib import Path

SCHEMA = "llm.debug-trace.v1"
MAX_RECORD_BYTES = 4096
MAX_RECORDS = 20_000
MAX_BYTES = 8 * 1024 * 1024

# This is deliberately closed: tracing is an observability boundary, not a
# general-purpose serialization hook.  Values which identify a request or a
# machine are represented by a digest/count before they reach this module.
SAFE_FIELDS = frozenset({
    "model", "selector", "context", "bucket", "concurrency", "wave", "count",
    "byte_count", "payload_bytes", "configured_ceiling", "return_code",
     "failure_code", "measurement_failure_code", "failure_detail", "failure_kind",
     "stage", "wave_type",
    "matrix_index", "total_vram_bytes", "safe", "elapsed_ms", "event_count",
    "baseline_pre_used_bytes", "peak_incremental_request_bytes", "derived_ceiling",
})
SAFE_STRING_FIELDS = frozenset({"model", "selector", "context", "bucket", "failure_code",
                                 "measurement_failure_code", "failure_detail", "failure_kind",
                                 "stage", "wave_type"})
SAFE_STRING = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
SAFE_COMPONENTS = frozenset({
    "matrix", "measurement", "outer", "persistence", "profile_store",
    "provider.coedit", "provider.gector", "provider.smollm", "resource_manager",
    "rm_runner", "supervisor", "trace",
})
SAFE_EVENTS = frozenset({
    "admission", "audit", "cleanup", "cli", "close", "container_exit", "execute",
    "input_validation", "lifecycle_close", "load", "measurement", "output",
    "persistence", "promotion", "readiness", "ready", "reservation",
    "runner_close", "runner_prime", "runner_residency", "runner_switch", "save",
    "selector", "session", "start", "submit", "truncated", "unload", "validate",
    "validate_input", "verify_cleanup", "watch", "wave",
})
SAFE_STATES = frozenset({"enter", "entered", "failure", "success", "truncated"})
FORBIDDEN_FIELD_PARTS = frozenset({
    "prompt", "payload", "output", "response", "body", "exception", "traceback",
    "token", "session", "lock", "environment", "path", "file", "request_id",
    "gpu_uuid", "gpu", "identity", "secret", "password",
})

_tls = threading.local()


def _short(value: object) -> object:
    """Keep only operator-safe scalar diagnostics; never stringify objects."""
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, str) and len(value) <= 80 and "/" not in value and "\\" not in value:
        return value if all(ord(c) >= 32 for c in value) else None
    return None


def _safe_field(key: str, value: object) -> tuple[str, object] | None:
    if not isinstance(key, str) or key not in SAFE_FIELDS:
        return None
    # Explicitly allowed numeric counts describe sizes, never their contents.
    if (any(part in key.lower() for part in FORBIDDEN_FIELD_PARTS)
            and key not in {"payload_bytes"}):
        return None
    safe = _short(value)
    if key in SAFE_STRING_FIELDS:
        if (not isinstance(safe, str) or not SAFE_STRING.fullmatch(safe)
                or any(part in safe.lower() for part in FORBIDDEN_FIELD_PARTS)):
            return None
    return (key, safe) if safe is not None else None


def _safe_label(value: object, vocabulary: frozenset[str]) -> str:
    safe = _short(value)
    if not isinstance(safe, str) or safe not in vocabulary:
        return "unknown"
    return safe


def _json_line(record: dict[str, object]) -> bytes:
    encoded = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True) + "\n").encode("utf-8")
    if len(encoded) <= MAX_RECORD_BYTES:
        return encoded
    # A record supplied by a caller can never make the transport unbounded.
    return _marker(record.get("sequence", 0), record.get("monotonic_ns", time.monotonic_ns()))


def _marker(sequence: object, monotonic_ns: object) -> bytes:
    return (json.dumps({"schema": SCHEMA, "sequence": sequence,
                        "monotonic_ns": monotonic_ns, "component": "trace",
                        "event": "truncated", "state": "truncated"},
                       separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sanitize_jsonl(text: str) -> str:
    """Return only bounded, valid trace records from untrusted runtime text."""
    if not isinstance(text, str):
        return ""
    output: list[bytes] = []
    total = 0
    truncated = False
    # Leave room for a marker, so the marker itself is always within both caps.
    marker = _marker(0, time.monotonic_ns())
    for line in text.splitlines():
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                continue
            # Re-apply the same boundary policy to data read from a container;
            # the inner process is not trusted merely because it is ours.
            clean: dict[str, object] = {}
            for key in ("schema", "sequence", "monotonic_ns"):
                safe = _short(value.get(key))
                if safe is not None:
                    clean[key] = safe
            for key in ("component", "event", "state"):
                vocabulary = {"component": SAFE_COMPONENTS, "event": SAFE_EVENTS,
                              "state": SAFE_STATES}[key]
                clean[key] = _safe_label(value.get(key), vocabulary)
            if clean.get("schema") != SCHEMA:
                continue
            for key, item in value.items():
                safe = _safe_field(key, item)
                if safe is not None:
                    clean[safe[0]] = safe[1]
            encoded = (json.dumps(clean, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
            if len(encoded) > MAX_RECORD_BYTES:
                encoded = _marker(clean.get("sequence", len(output) + 1),
                                  clean.get("monotonic_ns", time.monotonic_ns()))
            if len(output) >= MAX_RECORDS - 1 or total + len(encoded) > MAX_BYTES - len(marker):
                truncated = True
                break
            output.append(encoded)
            total += len(encoded)
        except Exception:
            continue
    if truncated:
        sequence = len(output) + 1
        marker = _marker(sequence, time.monotonic_ns())
        if len(output) < MAX_RECORDS and total + len(marker) <= MAX_BYTES:
            output.append(marker)
    return b"".join(output).decode("utf-8")


def digest_ids(ids: object) -> str | None:
    if not isinstance(ids, (tuple, list, set, frozenset)) or not all(isinstance(x, str) for x in ids):
        return None
    canonical = json.dumps(sorted(ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


class DebugTrace:
    def __init__(self, enabled: bool = False, path: str | os.PathLike[str] | None = None):
        self.enabled = bool(enabled)
        self.path = Path(path) if path is not None else None
        # record() and _emit() share the boundary lock.  RLock also protects
        # future helpers which may emit while holding the accounting lock.
        self._lock = threading.RLock()
        self._sequence = 0
        self._bytes = 0
        self._truncated = False
        self._records = 0
        self._fd = None
        if self.enabled and self.path is not None:
            try:
                if not self.path.parent.is_dir() or self.path.exists():
                    raise FileExistsError(os.fspath(self.path))
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                self._fd = os.open(self.path, flags, 0o600)
            except Exception:
                self._fd = None

    def _emit(self, record: dict[str, object]) -> None:
        if not self.enabled:
            return
        try:
            encoded = _json_line(record)
            with self._lock:
                marker = _marker(self._sequence + 1, time.monotonic_ns())
                if (self._sequence >= MAX_RECORDS - 1 or
                        self._bytes + len(encoded) > MAX_BYTES - len(marker)):
                    if not self._truncated:
                        self._truncated = True
                        if self._records < MAX_RECORDS and self._bytes + len(marker) <= MAX_BYTES:
                            self._write(marker)
                            self._records += 1
                    return
                self._write(encoded)
                self._records += 1
        except Exception:
            # Diagnostics are strictly non-interfering.
            return

    def _write(self, data: bytes) -> None:
        self._bytes += len(data)
        try:
            stream = getattr(sys.stderr, "buffer", sys.stderr)
            try:
                stream.write(data if stream is not sys.stderr else data.decode("utf-8"))
            except TypeError:
                stream.write(data)
            stream.flush()
        except Exception:
            pass
        if self.path is not None:
            try:
                if self._fd is None:
                    return
                view = memoryview(data)
                while view:
                    written = os.write(self._fd, view)
                    if written <= 0:
                        break
                    view = view[written:]
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None

    def record(self, component: str, event: str, state: str, **fields: object) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            record = {"schema": SCHEMA, "sequence": sequence, "monotonic_ns": time.monotonic_ns(),
                      "component": _safe_label(component, SAFE_COMPONENTS),
                      "event": _safe_label(event, SAFE_EVENTS),
                      "state": _safe_label(state, SAFE_STATES)}
            for key, value in fields.items():
                safe = _safe_field(key, value)
                if safe is not None:
                    record[safe[0]] = safe[1]
            # Keep sequence allocation and emission in one critical section so
            # concurrent lifecycle tasks cannot reorder the JSONL stream.
            self._emit(record)


def configure(enabled: bool = False, path: str | os.PathLike[str] | None = None) -> DebugTrace:
    previous = getattr(_tls, "trace", None)
    if previous is not None:
        try:
            previous.close()
        except Exception:
            pass
    trace = DebugTrace(enabled, path)
    _tls.trace = trace
    return trace


def current() -> DebugTrace:
    trace = getattr(_tls, "trace", None)
    if trace is None:
        trace = DebugTrace(False)
        _tls.trace = trace
    return trace


def record(component: str, event: str, state: str, **fields: object) -> None:
    current().record(component, event, state, **fields)


def lifecycle(component: str, event: str | None = None, *, failures_only: bool = False):
    """Trace async lifecycle milestones without inspecting arguments/results."""
    def decorate(function):
        @wraps(function)
        async def wrapped(*args, **kwargs):
            name = event or function.__name__
            if not failures_only:
                record(component, name, "enter")
            try:
                value = await function(*args, **kwargs)
            except BaseException:
                record(component, name, "failure")
                raise
            else:
                if not failures_only:
                    record(component, name, "success")
                return value
        return wrapped
    return decorate

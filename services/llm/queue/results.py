"""Crash-safe content-addressed results and durable publication receipts."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from pathlib import Path


class ResultError(Exception):
    pass


class ResultStore:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()
        target = self.root / digest
        if target.exists():
            if not target.is_file() or self._digest(target) != digest:
                raise ResultError("content-addressed result is corrupted")
            return digest
        fd, temporary = tempfile.mkstemp(dir=self.root, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            directory = os.open(self.root, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return digest

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def verify(self, reference: str) -> Path:
        if len(reference) != 64 or any(char not in "0123456789abcdef" for char in reference):
            raise ResultError("invalid result reference")
        path = (self.root / reference).resolve()
        if path.parent != self.root or not path.is_file() or self._digest(path) != reference:
            raise ResultError("result reference is unavailable or corrupted")
        return path

    def read(self, reference: str) -> bytes:
        return self.verify(reference).read_bytes()


class LocalPublisher:
    """A publisher which only records verified local result availability."""

    def __init__(self, result_store: ResultStore, state_path: str | os.PathLike[str]):
        self.result_store = result_store
        self.db = sqlite3.connect(str(state_path), timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS publication_receipts "
            "(idempotency_key TEXT PRIMARY KEY, result_reference TEXT NOT NULL)"
        )

    def publish(self, request_id: str, attempt_token: str,
                result_reference: str, idempotency_key: str) -> bool:
        del request_id, attempt_token
        self.result_store.verify(result_reference)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            receipt = self.db.execute(
                "SELECT result_reference FROM publication_receipts WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if receipt:
                if receipt[0] != result_reference:
                    raise ResultError("publication key conflict")
                self.db.execute("COMMIT")
                return False
            self.db.execute(
                "INSERT INTO publication_receipts VALUES(?,?)",
                (idempotency_key, result_reference),
            )
            self.db.execute("COMMIT")
            return True
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self.db.close()

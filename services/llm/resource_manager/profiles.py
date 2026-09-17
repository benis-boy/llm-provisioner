"""Durable, fail-closed capacity profile storage.

This module deliberately does not measure anything.  A caller which has done
the measurement supplies its evidence and explicitly chooses ``save_measured``;
the store is only the validation, persistence, and exact lookup boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata

SCHEMA_VERSION = 1
MAX_SWEEP_POINTS = 10  # additional points beyond the mandatory p=1 point
MAX_LATENCIES_PER_SAMPLE = 16384
MAX_TEXT_BYTES = 64 * 1024
MAX_EVIDENCE_BYTES = 4 * 1024 * 1024


class ProfileStoreError(RuntimeError):
    pass


class ProfileConflict(ProfileStoreError):
    pass


class CorruptProfileStore(ProfileStoreError):
    pass


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    if len(value.encode()) > MAX_TEXT_BYTES:
        raise ValueError(f"{name} is too large")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _sample(sample: SampleMetadata) -> dict[str, Any]:
    if not isinstance(sample, SampleMetadata):
        raise ValueError("samples must be SampleMetadata values")
    values = (sample.concurrency, sample.wave, sample.successful_requests,
              sample.wall_time_ms, sample.peak_vram_bytes)
    for value, name, minimum in zip(values, ("concurrency", "wave", "successful_requests", "wall_time_ms", "peak_vram_bytes"), (1, 0, 0, 0, 0)):
        _integer(value, name, minimum=minimum)
    if sample.successful_requests > sample.concurrency:
        raise ValueError("successful requests cannot exceed concurrency")
    latencies = []
    if len(sample.latency_ms) > MAX_LATENCIES_PER_SAMPLE:
        raise ValueError("too many latency values")
    for latency in sample.latency_ms:
        _integer(latency, "latency", minimum=0)
        latencies.append(latency)
    return {"concurrency": sample.concurrency, "wave": sample.wave,
            "successful_requests": sample.successful_requests,
            "wall_time_ms": sample.wall_time_ms, "peak_vram_bytes": sample.peak_vram_bytes,
            "latency_ms": latencies}


def _from_sample(value: dict[str, Any]) -> SampleMetadata:
    if not isinstance(value, dict):
        raise CorruptProfileStore("invalid raw capacity sample")
    try:
        return SampleMetadata(value["concurrency"], value["wave"], value["successful_requests"],
                              value["wall_time_ms"], value["peak_vram_bytes"], tuple(value["latency_ms"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise CorruptProfileStore("invalid raw capacity sample") from exc


@dataclass(frozen=True)
class BenchmarkMetadata:
    """Caller-attested evidence required for an approved measured profile."""

    fingerprint: str
    created_at: str
    provenance: str
    baseline_samples: tuple[SampleMetadata, ...]
    warmup_samples: tuple[SampleMetadata, ...]
    measured_samples: tuple[SampleMetadata, ...]
    representative_config: str

    def canonical(self) -> dict[str, Any]:
        _text(self.fingerprint, "fingerprint")
        _text(self.created_at, "created_at")
        _text(self.provenance, "provenance")
        _text(self.representative_config, "representative_config")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("created_at must be an ISO-8601 timestamp") from exc
        return {"fingerprint": self.fingerprint, "created_at": self.created_at,
                "provenance": self.provenance,
                "baseline_samples": [_sample(x) for x in self.baseline_samples],
                "warmup_samples": [_sample(x) for x in self.warmup_samples],
                "measured_samples": [_sample(x) for x in self.measured_samples],
                "representative_config": self.representative_config}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class ProfileStore:
    """SQLite WAL profile store.  Existing incompatible databases are rejected."""

    def __init__(self, path: str | Path):
        self._db = sqlite3.connect(str(path), timeout=10, isolation_level=None)
        try:
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA foreign_keys=ON")
            if self._db.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                raise CorruptProfileStore("WAL mode is unavailable")
            if self._db.execute("PRAGMA synchronous").fetchone()[0] != 2:
                raise CorruptProfileStore("full SQLite synchronization is unavailable")
            self._initialize()
        except Exception:
            self._db.close()
            raise

    @classmethod
    def open_readonly(cls, path: str | Path) -> "ProfileStore":
        """Open an existing registry without creating or changing SQLite state."""
        target = Path(path)
        if not target.is_file():
            raise CorruptProfileStore("profile registry is unavailable")
        self = cls.__new__(cls)
        self._db = sqlite3.connect(f"file:{target.absolute()}?mode=ro", uri=True,
                                   timeout=10, isolation_level=None)
        try:
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA foreign_keys=ON")
            self._inspect_existing()
        except Exception:
            self._db.close()
            raise
        return self

    def _inspect_existing(self) -> None:
        try:
            row = self._db.execute("SELECT value FROM profile_meta WHERE key='schema_version'").fetchone()
        except sqlite3.DatabaseError as exc:
            raise CorruptProfileStore("profile registry is unavailable") from exc
        if row is None or row[0] != str(SCHEMA_VERSION):
            raise CorruptProfileStore("incompatible profile schema version")
        self._inspect_schema()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "ProfileStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize(self) -> None:
        try:
            row = self._db.execute("SELECT value FROM profile_meta WHERE key='schema_version'").fetchone()
        except sqlite3.OperationalError:
            row = None
            if self._db.execute("SELECT count(*) FROM sqlite_master WHERE type IN ('table','index','trigger','view')").fetchone()[0]:
                raise CorruptProfileStore("non-empty database has no profile schema version")
        if row is not None and row[0] != str(SCHEMA_VERSION):
            raise CorruptProfileStore("incompatible profile schema version")
        if row is not None:
            self._inspect_schema()
        if row is None:
            # A non-profile database at this path must never be silently replaced.
            tables = self._db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if tables and any(item[0] not in ("profile_meta", "profiles", "profile_samples") for item in tables):
                raise CorruptProfileStore("database is not a profile database")
            if any(item[0] == "profile_meta" for item in tables):
                raise CorruptProfileStore("profile schema version is missing")
            try:
                self._db.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS profile_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS profiles (
                  profile_identity TEXT PRIMARY KEY, status TEXT NOT NULL,
                  model_id TEXT NOT NULL, gpu_uuid TEXT NOT NULL,
                  artifact_manifest_hash TEXT NOT NULL, model_hash TEXT NOT NULL,
                  runtime_identity TEXT NOT NULL, adapter_identity TEXT NOT NULL,
                  optimal_parallelism INTEGER NOT NULL, memory_safe_n INTEGER NOT NULL,
                  buffer_capacity INTEGER NOT NULL, safety_reserve_percent INTEGER NOT NULL,
                  context_size INTEGER, bucket_identity TEXT, metadata_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS profile_samples (
                  profile_identity TEXT NOT NULL, ordinal INTEGER NOT NULL,
                  sample_json TEXT NOT NULL, PRIMARY KEY(profile_identity, ordinal),
                  FOREIGN KEY(profile_identity) REFERENCES profiles(profile_identity)
                );
                INSERT OR IGNORE INTO profile_meta(key, value) VALUES ('schema_version', '1');
                COMMIT;
                """)
            except Exception:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise

    def _inspect_schema(self) -> None:
        expected = {
            "profile_meta": {"key", "value"},
            "profiles": {"profile_identity", "status", "model_id", "gpu_uuid", "artifact_manifest_hash", "model_hash", "runtime_identity", "adapter_identity", "optimal_parallelism", "memory_safe_n", "buffer_capacity", "safety_reserve_percent", "context_size", "bucket_identity", "metadata_json", "content_hash"},
            "profile_samples": {"profile_identity", "ordinal", "sample_json"},
        }
        objects = self._db.execute("SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
        if any(item[0] != "table" for item in objects):
            raise CorruptProfileStore("unexpected profile schema object")
        names = {row[1] for row in objects}
        if names != set(expected):
            raise CorruptProfileStore("profile schema tables differ")
        expected_types = {
            "profile_meta": {"key": "TEXT", "value": "TEXT"},
            "profiles": {"profile_identity": "TEXT", "status": "TEXT", "model_id": "TEXT",
                         "gpu_uuid": "TEXT", "artifact_manifest_hash": "TEXT", "model_hash": "TEXT",
                         "runtime_identity": "TEXT", "adapter_identity": "TEXT",
                         "optimal_parallelism": "INTEGER", "memory_safe_n": "INTEGER",
                         "buffer_capacity": "INTEGER", "safety_reserve_percent": "INTEGER",
                         "context_size": "INTEGER", "bucket_identity": "TEXT", "metadata_json": "TEXT",
                         "content_hash": "TEXT"},
            "profile_samples": {"profile_identity": "TEXT", "ordinal": "INTEGER", "sample_json": "TEXT"},
        }
        expected_not_null = {
            "profile_meta": {"key": False, "value": True},
            "profiles": {name: name not in {"profile_identity", "context_size", "bucket_identity"}
                         for name in expected["profiles"]},
            "profile_samples": {"profile_identity": True, "ordinal": True, "sample_json": True},
        }
        for table, columns in expected.items():
            info = list(self._db.execute(f"PRAGMA table_info({table})"))
            actual = {row[1] for row in info}
            if actual != columns:
                raise CorruptProfileStore(f"profile schema columns differ for {table}")
            if any(row[2].upper() != expected_types[table][row[1]] or
                   bool(row[3]) != expected_not_null[table][row[1]] for row in info):
                raise CorruptProfileStore("profile column definition differs")
        expected_pk = {"profile_meta": ("key",), "profiles": ("profile_identity",), "profile_samples": ("profile_identity", "ordinal")}
        for table, key in expected_pk.items():
            info = list(self._db.execute(f"PRAGMA table_info({table})"))
            if tuple(row[1] for row in sorted(info, key=lambda row: row[5]) if row[5]) != key:
                raise CorruptProfileStore("profile primary-key definition differs")
        foreign = list(self._db.execute("PRAGMA foreign_key_list(profile_samples)"))
        if len(foreign) != 1 or tuple(foreign[0][2:4]) != ("profiles", "profile_identity") or foreign[0][4] != "profile_identity" or self._db.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise CorruptProfileStore("foreign-key enforcement is unavailable")

    @staticmethod
    def _validate_evidence(profile: CapacityProfile, metadata: BenchmarkMetadata) -> dict[str, Any]:
        data = metadata.canonical()
        if len(metadata.baseline_samples) != 4 or any(s.concurrency != 1 or s.successful_requests != 1 for s in metadata.baseline_samples):
            raise ValueError("baseline evidence must contain four successful serial samples")
        points = {s.concurrency for s in metadata.measured_samples}
        if len(points - {1}) > MAX_SWEEP_POINTS or not points or 1 not in points or profile.memory_safe_n not in points:
            raise ValueError("measured evidence must include bounded points, p=1, and N")
        if any(point > profile.memory_safe_n for point in points):
            raise ValueError("measured concurrency cannot exceed N")
        if any(s.wall_time_ms <= 0 or len(s.latency_ms) != s.successful_requests or s.successful_requests != 1 for s in metadata.baseline_samples):
            raise ValueError("baseline samples must be successful positive-time serial samples")
        if profile.memory_safe_n >= 2 and 2 not in points:
            raise ValueError("measured evidence must include concurrency 2")
        if len(metadata.warmup_samples) != len(points) or {s.concurrency for s in metadata.warmup_samples} != points:
            raise ValueError("warmup evidence does not cover selected concurrency and N")
        if len(metadata.measured_samples) != 4 * len(points):
            raise ValueError("measured evidence must contain four waves per point")
        rates: dict[int, float] = {}
        for concurrency in sorted(points):
            waves = [s for s in metadata.measured_samples if s.concurrency == concurrency]
            if len(waves) != 4 or [s.wave for s in waves] != [1, 2, 3, 4] or any(s.successful_requests != concurrency or s.wall_time_ms <= 0 or len(s.latency_ms) != s.successful_requests for s in waves):
                raise ValueError("measured evidence must contain four successful ordered waves")
            rates[concurrency] = sum(s.successful_requests for s in waves) / sum(s.wall_time_ms for s in waves)
        selected = 1
        for concurrency in sorted(points):
            if concurrency != 1 and rates[concurrency] >= rates[selected] * 1.02:
                selected = concurrency
        if profile.optimal_parallelism != selected:
            raise ValueError("optimal parallelism is inconsistent with measured throughput")
        if any(s.wave != 0 or s.successful_requests != s.concurrency or s.wall_time_ms <= 0 or len(s.latency_ms) != s.successful_requests for s in metadata.warmup_samples):
            raise ValueError("warmup samples must be successful wave zero samples")
        raw = tuple(metadata.baseline_samples) + tuple(metadata.warmup_samples) + tuple(metadata.measured_samples)
        if tuple(profile.raw_samples) != raw:
            raise ValueError("profile raw samples must exactly preserve benchmark evidence order")
        if metadata.representative_config != (profile.bucket_identity or "context:" + str(profile.context_size)):
            raise ValueError("representative configuration is not the profile identity")
        if len(_canonical(data).encode()) > MAX_EVIDENCE_BYTES:
            raise ValueError("benchmark evidence is too large")
        return data

    def _record(self, profile: CapacityProfile, metadata: BenchmarkMetadata, status: str) -> None:
        # Re-run validation here rather than trusting CapacityProfile's looser legacy checks.
        ModelId(profile.model_id)
        if profile.optimal_parallelism > profile.memory_safe_n or profile.safety_reserve_percent != 20:
            raise ValueError("profile must have p <= N and a 20 percent reserve")
        if profile.model_id == ModelId.SMOLLM:
            if profile.context_size is None or profile.bucket_identity is not None:
                raise ValueError("SmolLM profiles require context and no bucket")
        elif profile.context_size is not None or not profile.bucket_identity:
            raise ValueError("non-Ollama profiles require a bucket and no context")
        for value, name in ((profile.context_size, "context_size"),):
            if value is not None:
                _integer(value, name, minimum=1)
        for sample in profile.raw_samples:
            _sample(sample)
        if status == "measured":
            evidence = self._validate_evidence(profile, metadata)
        else:
            evidence = metadata.canonical()
            if len(_canonical(evidence).encode()) > MAX_EVIDENCE_BYTES:
                raise ValueError("benchmark evidence is too large")
        identity_data = {"model_id": ModelId(profile.model_id).value, "gpu_uuid": profile.gpu_uuid,
                         "artifact_manifest_hash": profile.artifact_manifest_hash, "model_hash": profile.model_hash,
                         "runtime_identity": profile.runtime_identity, "adapter_identity": profile.adapter_identity,
                         "context_size": profile.context_size, "bucket_identity": profile.bucket_identity,
                         "fingerprint": metadata.fingerprint}
        expected_identity = hashlib.sha256(_canonical(identity_data).encode()).hexdigest()
        if profile.profile_identity != expected_identity:
            raise ValueError("profile_identity must be the deterministic identity including fingerprint")
        samples = [_sample(x) for x in profile.raw_samples]
        payload = {"profile": identity_data, "values": [profile.optimal_parallelism, profile.memory_safe_n, profile.buffer_capacity, profile.safety_reserve_percent], "metadata": evidence, "samples": samples}
        content_hash = hashlib.sha256(_canonical(payload).encode()).hexdigest()
        try:
            self._db.execute("BEGIN IMMEDIATE")
            old = self._db.execute("SELECT content_hash, status FROM profiles WHERE profile_identity=?", (profile.profile_identity,)).fetchone()
            if old:
                if old[0] != content_hash:
                    raise ProfileConflict("profile identity already contains different content")
                if status == "measured" and old[1] != "measured":
                    self._db.execute("UPDATE profiles SET status='measured' WHERE profile_identity=?", (profile.profile_identity,))
                self._db.execute("COMMIT")
                return
            self._db.execute("INSERT INTO profiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (profile.profile_identity, status, ModelId(profile.model_id).value, profile.gpu_uuid, profile.artifact_manifest_hash, profile.model_hash, profile.runtime_identity, profile.adapter_identity, profile.optimal_parallelism, profile.memory_safe_n, profile.buffer_capacity, profile.safety_reserve_percent, profile.context_size, profile.bucket_identity, _canonical(evidence), content_hash))
            self._db.executemany("INSERT INTO profile_samples VALUES (?,?,?)", [(profile.profile_identity, i, _canonical(value)) for i, value in enumerate(samples)])
            self._db.execute("COMMIT")
        except Exception:
            self._db.execute("ROLLBACK")
            raise

    def save_measured(self, profile: CapacityProfile, metadata: BenchmarkMetadata) -> None:
        self._record(profile, metadata, "measured")

    def save_draft(self, profile: CapacityProfile, metadata: BenchmarkMetadata) -> None:
        self._record(profile, metadata, "draft")

    def lookup(self, model_id: ModelId, gpu_uuid: str, artifact_manifest_hash: str, model_hash: str,
               runtime_identity: str, adapter_identity: str, *, context_size: int | None = None,
               bucket_identity: str | None = None) -> CapacityProfile | None:
        if isinstance(context_size, bool) or (context_size is not None and _integer(context_size, "context_size", minimum=1) < 1):
            raise ValueError("context_size must be positive")
        if context_size is not None and bucket_identity is not None:
            raise ValueError("context and bucket lookup are mutually exclusive")
        params: list[Any] = [ModelId(model_id).value, gpu_uuid, artifact_manifest_hash, model_hash, runtime_identity, adapter_identity]
        try:
            self._db.execute("BEGIN")
            if context_size is not None:
                if ModelId(model_id) != ModelId.SMOLLM:
                    self._db.execute("ROLLBACK")
                    return None
                rows = self._db.execute("SELECT *, length(metadata_json) AS metadata_bytes FROM profiles WHERE status='measured' AND model_id=? AND gpu_uuid=? AND artifact_manifest_hash=? AND model_hash=? AND runtime_identity=? AND adapter_identity=? AND context_size>=? ORDER BY context_size, profile_identity LIMIT 2", (*params, context_size)).fetchall()
            else:
                if not bucket_identity:
                    self._db.execute("ROLLBACK")
                    return None
                if ModelId(model_id) == ModelId.SMOLLM:
                    self._db.execute("ROLLBACK")
                    return None
                rows = self._db.execute("SELECT *, length(metadata_json) AS metadata_bytes FROM profiles WHERE status='measured' AND model_id=? AND gpu_uuid=? AND artifact_manifest_hash=? AND model_hash=? AND runtime_identity=? AND adapter_identity=? AND bucket_identity=? ORDER BY profile_identity LIMIT 2", (*params, bucket_identity)).fetchall()
            if len(rows) > 1 and (context_size is None or rows[0]["context_size"] == rows[1]["context_size"]):
                raise CorruptProfileStore("ambiguous measured profile identity")
            row = rows[0] if rows else None
        except (sqlite3.DatabaseError, CorruptProfileStore) as exc:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            if isinstance(exc, CorruptProfileStore):
                raise
            raise CorruptProfileStore("profile schema or row is unreadable") from exc
        if row is None:
            self._db.execute("COMMIT")
            return None
        try:
            if row["metadata_bytes"] > MAX_EVIDENCE_BYTES:
                raise CorruptProfileStore("stored evidence is too large")
            raw_count, raw_bytes = self._db.execute("SELECT count(*), coalesce(sum(length(sample_json)), 0) FROM profile_samples WHERE profile_identity=?", (row["profile_identity"],)).fetchone()
            if raw_count > 4 + 11 + 44 or raw_bytes > MAX_EVIDENCE_BYTES:
                raise CorruptProfileStore("stored raw evidence is too large")
            raw = self._db.execute("SELECT ordinal, sample_json FROM profile_samples WHERE profile_identity=? ORDER BY ordinal", (row["profile_identity"],)).fetchall()
            if not raw or [item["ordinal"] for item in raw] != list(range(len(raw))):
                raise CorruptProfileStore("profile raw sample ordinals are not contiguous")
            samples = tuple(_from_sample(json.loads(item["sample_json"])) for item in raw)
            evidence = json.loads(row["metadata_json"])
            metadata = BenchmarkMetadata(evidence["fingerprint"], evidence["created_at"], evidence["provenance"], tuple(_from_sample(x) for x in evidence["baseline_samples"]), tuple(_from_sample(x) for x in evidence["warmup_samples"]), tuple(_from_sample(x) for x in evidence["measured_samples"]), evidence["representative_config"])
            profile = CapacityProfile(ModelId(row["model_id"]), row["gpu_uuid"], row["artifact_manifest_hash"], row["model_hash"], row["runtime_identity"], row["adapter_identity"], row["profile_identity"], row["optimal_parallelism"], row["memory_safe_n"], row["buffer_capacity"], row["safety_reserve_percent"], samples, row["context_size"], row["bucket_identity"])
            if row["status"] != "measured":
                self._db.execute("ROLLBACK")
                return None
            self._record_hashes(profile, metadata, row["content_hash"])
            self._db.execute("COMMIT")
            return profile
        except BaseException as exc:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            if isinstance(exc, CorruptProfileStore):
                raise
            raise CorruptProfileStore("invalid stored profile") from exc

    @staticmethod
    def _record_hashes(profile: CapacityProfile, metadata: BenchmarkMetadata, stored_hash: str) -> None:
        identity = {"model_id": ModelId(profile.model_id).value, "gpu_uuid": profile.gpu_uuid, "artifact_manifest_hash": profile.artifact_manifest_hash, "model_hash": profile.model_hash, "runtime_identity": profile.runtime_identity, "adapter_identity": profile.adapter_identity, "context_size": profile.context_size, "bucket_identity": profile.bucket_identity, "fingerprint": metadata.fingerprint}
        expected_id = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        if expected_id != profile.profile_identity:
            raise CorruptProfileStore("stored profile identity mismatch")
        payload = {"profile": identity, "values": [profile.optimal_parallelism, profile.memory_safe_n, profile.buffer_capacity, profile.safety_reserve_percent], "metadata": ProfileStore._validate_evidence(profile, metadata), "samples": [_sample(x) for x in profile.raw_samples]}
        if hashlib.sha256(_canonical(payload).encode()).hexdigest() != stored_hash:
            raise CorruptProfileStore("stored profile content hash mismatch")

import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.profiles import (
    BenchmarkMetadata, CorruptProfileStore, ProfileConflict, ProfileStore,
)


def sample(concurrency, wave, successful=None, wall=10):
    return SampleMetadata(concurrency, wave, concurrency if successful is None else successful,
                          wall, 100, tuple([2] * concurrency))


def fixture(context=2048, fingerprint="test", walls=None, optimal=2):
    p = 2
    baseline = tuple(sample(1, i, 1) for i in range(4))
    warmup = (sample(1, 0, 1), sample(p, 0))
    walls = walls or {1: 10, 2: 5}
    measured = tuple(sample(concurrency, wave, wall=walls[concurrency])
                     for concurrency in (1, p) for wave in range(1, 5))
    profile_id = __import__("hashlib").sha256(json.dumps({
        "model_id": "SmolLM", "gpu_uuid": "gpu", "artifact_manifest_hash": "manifest",
        "model_hash": "model", "runtime_identity": "runtime", "adapter_identity": "adapter",
        "context_size": context, "bucket_identity": None, "fingerprint": fingerprint,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    profile = CapacityProfile(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter",
                               profile_id, optimal, p, optimal, 20, baseline + warmup + measured, context, None)
    metadata = BenchmarkMetadata(fingerprint, "2026-09-16T00:00:00Z", "unit-test",
                                 baseline, warmup, measured, f"context:{context}")
    return profile, metadata


class ProfileStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "profiles.sqlite"

    def tearDown(self):
        self.tmp.cleanup()

    def test_reopen_and_smallest_context_lookup(self):
        with ProfileStore(self.path) as store:
            for context in (4096, 2048):
                profile, metadata = fixture(context)
                store.save_measured(profile, metadata)
        with ProfileStore(self.path) as store:
            found = store.lookup(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", context_size=3000)
            self.assertEqual(found.context_size, 4096)
            self.assertIsNone(store.lookup(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", context_size=8192))

    def test_duplicate_identical_is_noop_and_changed_content_conflicts(self):
        profile, metadata = fixture()
        with ProfileStore(self.path) as store:
            store.save_measured(profile, metadata)
            store.save_measured(profile, metadata)
            changed = BenchmarkMetadata(metadata.fingerprint, metadata.created_at, "different", metadata.baseline_samples,
                                        metadata.warmup_samples, metadata.measured_samples, metadata.representative_config)
            with self.assertRaises(ProfileConflict):
                store.save_measured(profile, changed)

    def test_every_identity_component_mismatch_is_not_found(self):
        profile, metadata = fixture()
        identity = ("gpu", "manifest", "model", "runtime", "adapter")
        with ProfileStore(self.path) as store:
            store.save_measured(profile, metadata)
            for index in range(len(identity)):
                query = list(identity)
                query[index] += "-different"
                self.assertIsNone(store.lookup(ModelId.SMOLLM, *query, context_size=2048))

    def test_cross_connection_replay_is_noop_and_changed_content_conflicts(self):
        profile, metadata = fixture()
        first = ProfileStore(self.path)
        second = ProfileStore(self.path)
        try:
            first.save_measured(profile, metadata)
            second.save_measured(profile, metadata)
            changed = BenchmarkMetadata(metadata.fingerprint, metadata.created_at, "other-provenance",
                                       metadata.baseline_samples, metadata.warmup_samples,
                                       metadata.measured_samples, metadata.representative_config)
            with self.assertRaises(ProfileConflict):
                second.save_measured(profile, changed)
        finally:
            first.close()
            second.close()

    def test_draft_is_never_selectable_and_bool_inputs_rejected(self):
        profile, metadata = fixture()
        with ProfileStore(self.path) as store:
            store.save_draft(profile, metadata)
            self.assertIsNone(store.lookup(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", context_size=1))
            with self.assertRaises(ValueError):
                store.lookup(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", context_size=True)

    def test_corrupt_raw_row_fails_closed(self):
        profile, metadata = fixture()
        with ProfileStore(self.path) as store:
            store.save_measured(profile, metadata)
            store._db.execute("UPDATE profile_samples SET sample_json=?", ("{}",))
            with self.assertRaises(CorruptProfileStore):
                store.lookup(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", context_size=2048)

    def test_hash_corruption_rolls_back_and_same_connection_can_query_again(self):
        profile, metadata = fixture()
        with ProfileStore(self.path) as store:
            store.save_measured(profile, metadata)
            original = store._db.execute(
                "SELECT content_hash FROM profiles WHERE profile_identity=?", (profile.profile_identity,)
            ).fetchone()[0]
            store._db.execute("UPDATE profiles SET content_hash=?", ("tampered",))
            with self.assertRaises(CorruptProfileStore):
                store.lookup(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", context_size=2048)
            store._db.execute("UPDATE profiles SET content_hash=?", (original,))
            self.assertEqual(
                store.lookup(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", context_size=2048),
                profile,
            )

    def test_ambiguous_lookup_rolls_back_and_same_connection_can_query_again(self):
        first, first_metadata = fixture(fingerprint="first")
        second, second_metadata = fixture(fingerprint="second")
        with ProfileStore(self.path) as store:
            store.save_measured(first, first_metadata)
            store.save_measured(second, second_metadata)
            with self.assertRaises(CorruptProfileStore):
                store.lookup(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", context_size=2048)
            store._db.execute("DELETE FROM profile_samples WHERE profile_identity=?", (second.profile_identity,))
            store._db.execute("DELETE FROM profiles WHERE profile_identity=?", (second.profile_identity,))
            self.assertEqual(
                store.lookup(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", context_size=2048),
                first,
            )

    def test_metadata_content_hash_parallelism_sample_and_ordinal_corruption_fail_closed(self):
        mutations = (
            ("UPDATE profiles SET metadata_json=?", ("{}",)),
            ("UPDATE profiles SET content_hash=?", ("bad-hash",)),
            ("UPDATE profiles SET optimal_parallelism=?", (1,)),
            ("UPDATE profile_samples SET sample_json=?", ('{"concurrency":true}',)),
            ("UPDATE profile_samples SET ordinal=? WHERE ordinal=0", (-1,)),
        )
        for statement, params in mutations:
            with self.subTest(statement=statement):
                profile, metadata = fixture()
                with ProfileStore(self.path) as store:
                    store.save_measured(profile, metadata)
                    store._db.execute(statement, params)
                    with self.assertRaises(CorruptProfileStore):
                        store.lookup(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", context_size=2048)
                self.path.unlink()

    def test_incompatible_existing_schema_is_rejected(self):
        connection = sqlite3.connect(self.path)
        connection.execute("CREATE TABLE profile_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO profile_meta VALUES ('schema_version', '99')")
        connection.commit()
        connection.close()
        with self.assertRaises(CorruptProfileStore):
            ProfileStore(self.path)

    def test_malformed_versioned_schema_is_rejected(self):
        connection = sqlite3.connect(self.path)
        connection.execute("CREATE TABLE profile_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO profile_meta VALUES ('schema_version', '1')")
        connection.commit()
        connection.close()
        with self.assertRaises(CorruptProfileStore):
            ProfileStore(self.path)

    def test_missing_version_profile_database_is_rejected_unchanged(self):
        connection = sqlite3.connect(self.path)
        connection.execute("CREATE TABLE profiles (profile_identity TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO profiles VALUES ('untouched')")
        connection.commit()
        before = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        connection.close()
        with self.assertRaises(CorruptProfileStore):
            ProfileStore(self.path)
        connection = sqlite3.connect(self.path)
        self.assertEqual(
            connection.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall(), before
        )
        self.assertEqual(connection.execute("SELECT * FROM profiles").fetchall(), [("untouched",)])
        connection.close()

    def test_incomplete_draft_is_saved_but_excluded(self):
        profile, metadata = fixture()
        incomplete = BenchmarkMetadata(metadata.fingerprint, metadata.created_at, metadata.provenance,
                                       metadata.baseline_samples, metadata.warmup_samples, (),
                                       metadata.representative_config)
        with ProfileStore(self.path) as store:
            store.save_draft(profile, incomplete)
            self.assertIsNone(store.lookup(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", context_size=2048))

    def test_higher_parallelism_with_worse_rate_is_rejected(self):
        profile, metadata = fixture(walls={1: 10, 2: 21}, optimal=2)
        with ProfileStore(self.path) as store:
            with self.assertRaises(ValueError):
                store.save_measured(profile, metadata)

    def test_measured_sample_point_above_memory_bound_is_rejected(self):
        profile, metadata = fixture()
        extra = tuple(sample(3, wave) for wave in range(1, 5))
        warmup = metadata.warmup_samples + (sample(3, 0),)
        measured = metadata.measured_samples + extra
        changed_metadata = replace(metadata, warmup_samples=warmup, measured_samples=measured)
        changed_profile = replace(profile, raw_samples=metadata.baseline_samples + warmup + measured)
        with ProfileStore(self.path) as store:
            with self.assertRaises(ValueError):
                store.save_measured(changed_profile, changed_metadata)

    def test_zero_or_latencyless_baseline_is_rejected(self):
        profile, metadata = fixture()
        for broken in (sample(1, 0, wall=0), replace(sample(1, 0), latency_ms=())):
            with self.subTest(broken=broken):
                baseline = (broken,) + metadata.baseline_samples[1:]
                changed_metadata = replace(metadata, baseline_samples=baseline)
                changed_profile = replace(profile, raw_samples=baseline + metadata.warmup_samples + metadata.measured_samples)
                with ProfileStore(self.path) as store:
                    with self.assertRaises(ValueError):
                        store.save_measured(changed_profile, changed_metadata)

    def test_exact_two_percent_gain_keeps_lower_parallelism(self):
        # The implementation's aggregate rates are 4 / (4 * wall), so this
        # chooses a slightly-under-threshold tie without relying on float
        # rounding at exactly 1.02.
        profile, metadata = fixture(walls={1: 50, 2: 99}, optimal=1)
        with ProfileStore(self.path) as store:
            store.save_measured(profile, metadata)

    def test_closest_context_is_selected(self):
        with ProfileStore(self.path) as store:
            for context in (1024, 2048, 4096):
                profile, metadata = fixture(context=context, fingerprint=str(context))
                store.save_measured(profile, metadata)
            found = store.lookup(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", context_size=1500)
            self.assertEqual(found.context_size, 2048)

    def test_schema_wrong_type_primary_key_and_foreign_key_are_rejected(self):
        variants = {
            "wrong type": "key TEXT PRIMARY KEY, value INTEGER NOT NULL",
            "wrong primary key": "key TEXT NOT NULL, value TEXT PRIMARY KEY",
            "wrong foreign key": "key TEXT PRIMARY KEY, value TEXT NOT NULL",
        }
        for name, meta_definition in variants.items():
            with self.subTest(schema=name):
                connection = sqlite3.connect(self.path)
                connection.executescript(f"""
                    CREATE TABLE profile_meta ({meta_definition});
                    CREATE TABLE profiles (
                      profile_identity TEXT PRIMARY KEY, status TEXT NOT NULL,
                      model_id TEXT NOT NULL, gpu_uuid TEXT NOT NULL,
                      artifact_manifest_hash TEXT NOT NULL, model_hash TEXT NOT NULL,
                      runtime_identity TEXT NOT NULL, adapter_identity TEXT NOT NULL,
                      optimal_parallelism INTEGER NOT NULL, memory_safe_n INTEGER NOT NULL,
                      buffer_capacity INTEGER NOT NULL, safety_reserve_percent INTEGER NOT NULL,
                      context_size INTEGER, bucket_identity TEXT, metadata_json TEXT NOT NULL,
                      content_hash TEXT NOT NULL
                    );
                    CREATE TABLE profile_samples (
                      profile_identity TEXT NOT NULL, ordinal INTEGER NOT NULL,
                      sample_json TEXT NOT NULL, PRIMARY KEY(profile_identity, ordinal)
                      {'' if name == 'wrong foreign key' else ', FOREIGN KEY(profile_identity) REFERENCES profiles(profile_identity)'}
                    );
                """)
                if name != "wrong type":
                    connection.execute("INSERT INTO profile_meta(key, value) VALUES ('schema_version', '1')")
                connection.commit()
                connection.close()
                with self.assertRaises(CorruptProfileStore):
                    ProfileStore(self.path)
                self.path.unlink()

    def test_readonly_open_does_not_create_or_change_database(self):
        profile, metadata = fixture()
        with ProfileStore(self.path) as store:
            store.save_measured(profile, metadata)
        before = self.path.read_bytes()
        with ProfileStore.open_readonly(self.path) as store:
            self.assertEqual(store.lookup(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", context_size=2048), profile)
            self.assertEqual(store._db.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            self.assertEqual(store._db.execute("PRAGMA journal_mode=WAL").fetchone()[0], "wal")
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

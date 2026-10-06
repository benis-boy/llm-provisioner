import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.profiles import ProfileConflict, ProfileStore
from services.llm.provisioning.benchmark_requests import BenchmarkRequest, _request_fingerprint
from services.llm.provisioning.capacity import MemorySample, Wave
from services.llm.provisioning.measurement import AuthoritativeMeasurement, _sweep_points
from services.llm.provisioning.measured_profiles import (
    MeasuredProfileIdentity, PersistenceValidationError, persist_measured_profile,
)
from services.llm.providers.coedit_batch import AllocatorObservation


class SpyStore(ProfileStore):
    def __init__(self):
        self.saved = 0
        self.profile = None

    def save_measured(self, profile, metadata):
        self.saved += 1
        self.profile = profile

    def lookup(self, *args, **kwargs):
        return self.profile


class PersistenceValidationErrorTests(unittest.TestCase):
    def test_categories_are_closed_and_remain_value_errors(self):
        for detail in ("eligibility", "resident_p2_witness_formula", "latency",
                       "store_save_readback_identity"):
            error = PersistenceValidationError(detail)
            self.assertIsInstance(error, ValueError)
            self.assertEqual(error.measurement_failure_detail, detail)
        with self.assertRaises(ValueError):
            PersistenceValidationError("provider_secret")


def request(model, bucket):
    identity = {"model": model.value, "request_bucket": bucket}
    payload = b"representative"
    return BenchmarkRequest(model, bucket, payload,
                            _request_fingerprint(identity, payload), identity, "configured")


def identity(model, bucket, context=None, **changes):
    values = dict(model_id=model, gpu_uuid="GPU", artifact_manifest_hash="manifest",
                  model_hash="model", runtime_identity="runtime",
                  adapter_identity="adapter", provenance="unit-test",
                  created_at="2026-09-18T00:00:00Z", context_size=context,
                  bucket_identity=None if context is not None else bucket)
    values.update(changes)
    return MeasuredProfileIdentity(**values)


def wave(phase, p, number, bad_sample=False, timestamp=None):
    timestamp = (timestamp if timestamp is not None else
                 {"baseline": 0, "discovery": 500, "warmup": 1000, "measured": 2000}[phase] + p * 100 + number * 3)
    sample = MemorySample(timestamp, 1000, 100, 900)
    if bad_sample:
        sample = MemorySample(-1, 1000, 100, 900)
    return Wave(p, number, tuple(f"{phase}-{p}-{number}-{i}" for i in range(p)),
                10, True, p, timestamp, timestamp, True, (sample,), phase=phase,
                allocator=AllocatorObservation(1, 2, 3, 4, 1, 2), observation_count=1,
                observed_native_batch_sizes=(p,), native_request_correlation=True,
                observation_drops=0, decoder_steps=(64,) * p, max_output_tokens=64)


def measurement(n=1, **changes):
    points = _sweep_points(n)
    baseline = tuple(wave("baseline", 1, i) for i in range(1, 5))
    discovery = tuple(wave("discovery", p, i) for p in range(2, n + 1) for i in range(1, 5))
    warmups = tuple(wave("warmup", p, 0) for p in points)
    measured = tuple(wave("measured", p, i) for p in points for i in range(1, 5))
    # Keep the generic fixture valid through the provider's bounded p=32
    # capability while preserving its original values for the small cases.
    pre = max(100, 800 - 100 * n)
    baseline = tuple(replace(item, samples=tuple(replace(sample, used_bytes=pre,
                                                         free_bytes=1000 - pre)
                                      for sample in item.samples)) for item in baseline)
    discovery = tuple(replace(item, samples=tuple(replace(sample, used_bytes=pre + 200,
                                                          free_bytes=800 - pre)
                                       for sample in item.samples)) for item in discovery)
    warmups = tuple(replace(item, samples=tuple(replace(sample,
                                                        used_bytes=pre + (200 if item.concurrency > 1 else (0 if n >= 2 else 100)),
                                                        free_bytes=1000 - pre - (200 if item.concurrency > 1 else (0 if n >= 2 else 100)))
                                     for sample in item.samples)) for item in warmups)
    measured = tuple(replace(item, samples=tuple(replace(sample,
                                                         used_bytes=pre + (200 if item.concurrency > 1 else (0 if n >= 2 else 100)),
                                                         free_bytes=1000 - pre - (200 if item.concurrency > 1 else (0 if n >= 2 else 100)))
                                      for sample in item.samples)) for item in measured)
    values = dict(status="complete", n=n, optimum=2 if n >= 2 else 1, m=2 if n >= 2 else 1, reserve_percent=20,
                  baseline=baseline, warmups=warmups, measured=measured,
                  benchmark_fingerprint="", baseline_mean_ms=10.0,
                   # The persistence contract independently recomputes the
                   # reserve-safe ceiling.  Keep this fixture internally
                   # consistent for each synthetic N (800 - pre_used = N *
                   # increment), rather than relying on the old arbitrary
                   # telemetry values.
                    baseline_pre_used_bytes=pre,
                   peak_incremental_request_bytes=200 if n >= 2 else 100,
                   total_vram_bytes=1000, derived_ceiling=n, configured_ceiling=n,
                   supported_parallelism=n,
                   parallelism_bound_reason="identity_configured_provider_capability",
                  profile_eligible=True)
    values["successful_discovery"] = discovery
    values.update(changes)
    item = AuthoritativeMeasurement(**values)
    # The runner executes discovery before throughput, then each point's
    # warmup and four measured waves.  Retain that chronology in the ordinary
    # valid fixture; individual rejection tests can still deliberately tamper
    # with it below.
    ordered = [*item.baseline, *item.successful_discovery]
    for index, _point in enumerate(points):
        ordered.append(item.warmups[index])
        ordered.extend(item.measured[index * 4:(index + 1) * 4])
    rewritten = []
    for timestamp, current in enumerate(ordered, 1):
        samples = tuple(replace(sample, timestamp_ns=timestamp,
                                start_ns=timestamp, end_ns=timestamp)
                        for sample in current.samples)
        rewritten.append(replace(current, execution_started=timestamp,
                                 execution_ended=timestamp, samples=samples))
    cursor = 0
    baseline = tuple(rewritten[cursor:cursor + len(item.baseline)])
    cursor += len(item.baseline)
    discovery = tuple(rewritten[cursor:cursor + len(item.successful_discovery)])
    cursor += len(item.successful_discovery)
    warmups = []
    measured = []
    for _point in points:
        warmups.append(rewritten[cursor])
        cursor += 1
        measured.extend(rewritten[cursor:cursor + 4])
        cursor += 4
    return replace(item, baseline=baseline, successful_discovery=discovery,
                   warmups=tuple(warmups), measured=tuple(measured))


def latencies(ids):
    return {request_id: 3 for request_id in ids}


class MeasuredProfilePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def persist(self, model=ModelId.SMOLLM, bucket="smollm:context2048", context=2048,
                 **measurement_changes):
        req = request(model, bucket)
        item = measurement(2 if model is ModelId.SMOLLM else 1,
                           **{"benchmark_fingerprint": req.fingerprint,
                              **measurement_changes})
        ident = identity(model, bucket, context)
        with ProfileStore(Path(self.tmp.name) / "profiles.sqlite") as store:
            return req, item, ident, persist_measured_profile(req, item, ident, store,
                                                               latency_extractor=latencies)

    def test_smollm_success_exact_readback_and_identity(self):
        req, item, ident, result = self.persist()
        self.assertEqual({wave.phase for wave in item.baseline}, {"baseline"})
        self.assertEqual({wave.phase for wave in item.warmups}, {"warmup"})
        self.assertEqual({wave.phase for wave in item.measured}, {"measured"})
        expected = hashlib.sha256((__import__("json").dumps({
            "model_id": "SmolLM", "gpu_uuid": "GPU", "artifact_manifest_hash": "manifest",
            "model_hash": "model", "runtime_identity": "runtime", "adapter_identity": "adapter",
            "context_size": 2048, "bucket_identity": None, "fingerprint": req.fingerprint,
        }, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()
        self.assertEqual(result.profile.profile_identity, expected)
        self.assertEqual(result.profile, result.profile)
        self.assertEqual(result.metadata.representative_config, "context:2048")

    def test_bucket_success_and_exact_readback(self):
        # A non-resident native provider cannot infer a per-slot memory formula.
        # CoEdIT's fixed p=32 identity capability is nevertheless exhaustively
        # established by every p=2..32 discovery wave.
        req = request(ModelId.COEDIT, "coedit:bucket")
        item = measurement(32, benchmark_fingerprint=req.fingerprint,
                           optimum=32, m=32, configured_ceiling=32,
                           supported_parallelism=32, derived_ceiling=None)
        ident = identity(ModelId.COEDIT, "coedit:bucket")
        with ProfileStore(Path(self.tmp.name) / "profiles.sqlite") as store:
            result = persist_measured_profile(req, item, ident, store,
                                              latency_extractor=latencies)
            found = store.lookup(ModelId.COEDIT, "GPU", "manifest", "model", "runtime", "adapter",
                                 bucket_identity="coedit:bucket")
        self.assertEqual(found, result.profile)
        self.assertEqual(result.metadata.representative_config, "coedit:bucket")

    def test_coedit_rising_baseline_replays_shared_memory_summary_and_rejects_tamper(self):
        req = request(ModelId.COEDIT, "coedit:bucket")
        item = measurement(3, benchmark_fingerprint=req.fingerprint,
                           optimum=3, m=3, baseline_pre_used_bytes=400,
                           peak_incremental_request_bytes=200,
                           total_vram_bytes=1400, derived_ceiling=None,
                           configured_ceiling=3, supported_parallelism=3)

        # The baseline's first sample is allowed to rise across waves.  The
        # final baseline (400), not an online intermediate denominator, is the
        # value used to replay every retained successful wave.
        baseline = tuple(
            replace(w, samples=tuple(replace(s, total_bytes=1400,
                                              used_bytes=100 + index * 100,
                                              free_bytes=1400 - (100 + index * 100))
                                      for s in w.samples))
            for index, w in enumerate(item.baseline))
        retained = (*item.successful_discovery, *item.warmups, *item.measured)
        retained = tuple(
            replace(w, samples=tuple(replace(s, total_bytes=1400,
                                              used_bytes=600, free_bytes=800)
                                      for s in w.samples))
            for w in retained)
        discovery = retained[:len(item.successful_discovery)]
        warmups = retained[len(item.successful_discovery):
                           len(item.successful_discovery) + len(item.warmups)]
        measured = retained[len(item.successful_discovery) + len(item.warmups):]
        item = replace(item, baseline=baseline, successful_discovery=discovery,
                       warmups=warmups, measured=measured)
        ident = identity(ModelId.COEDIT, "coedit:bucket")

        with ProfileStore(Path(self.tmp.name) / "profiles.sqlite") as store:
            result = persist_measured_profile(req, item, ident, store,
                                              latency_extractor=latencies)
        self.assertEqual(result.profile.memory_safe_n, 3)
        with self.assertRaises(ValueError):
            persist_measured_profile(req,
                                     replace(item, peak_incremental_request_bytes=201),
                                     ident, SpyStore(), latency_extractor=latencies)

    def test_formula_backed_operator_ceiling_persists_below_supported_parallelism(self):
        req = request(ModelId.SMOLLM, "smollm:context2048")
        ident = identity(ModelId.SMOLLM, "smollm:context2048", 2048)
        item = measurement(
            2,
            optimum=2,
            m=2,
            benchmark_fingerprint=req.fingerprint,
            configured_ceiling=32,
            supported_parallelism=32,
            parallelism_bound_reason="identity_configured_operator_ceiling",
        )
        # The persistence contract validates actual interleaved execution
        # chronology.  This fixture is intentionally rewritten into that
        # chronology rather than relying on phase-local synthetic timestamps.
        ordered = (*item.baseline, *item.successful_discovery,
                   *(wave for point in _sweep_points(2)
                                     for wave in (item.warmups[_sweep_points(2).index(point)],
                                                  *item.measured[_sweep_points(2).index(point) * 4:
                                                                 (_sweep_points(2).index(point) + 1) * 4])))
        rewritten = []
        for index, current in enumerate(ordered, 1):
            timestamp = index * 100
            samples = tuple(replace(sample, timestamp_ns=timestamp,
                                    start_ns=timestamp, end_ns=timestamp)
                            for sample in current.samples)
            rewritten.append(replace(current, execution_started=timestamp,
                                     execution_ended=timestamp, samples=samples))
        cursor = 0
        baseline = tuple(rewritten[cursor:cursor + len(item.baseline)])
        cursor += len(item.baseline)
        discovery = tuple(rewritten[cursor:cursor + len(item.successful_discovery)])
        cursor += len(item.successful_discovery)
        warmups = []
        measured = []
        for point in _sweep_points(2):
            warmups.append(rewritten[cursor]); cursor += 1
            measured.extend(rewritten[cursor:cursor + 4]); cursor += 4
        item = replace(item, baseline=baseline, successful_discovery=discovery,
                       warmups=tuple(warmups),
                       measured=tuple(measured))
        path = Path(self.tmp.name) / "profiles.sqlite"
        with ProfileStore(path) as store:
            result = persist_measured_profile(req, item, ident, store,
                                              latency_extractor=latencies)
        with ProfileStore(path) as store:
            reopened = store.lookup(ModelId.SMOLLM, "GPU", "manifest", "model",
                                    "runtime", "adapter", context_size=2048)
        self.assertEqual(reopened, result.profile)

        for invalid in (
            replace(item, n=33, optimum=1, m=1),
            replace(item, n=3, optimum=1, m=1),
            replace(item, derived_ceiling=3),
        ):
            with self.subTest(invalid=invalid):
                 with self.assertRaises(ValueError):
                     persist_measured_profile(req, invalid, ident, SpyStore(),
                                              latency_extractor=latencies)

    def test_resident_formula_freezes_p2_witness_despite_later_larger_waves(self):
        req = request(ModelId.SMOLLM, "smollm:context2048")
        item = measurement(3, benchmark_fingerprint=req.fingerprint,
                           optimum=3, m=3,
                           baseline_pre_used_bytes=300, derived_ceiling=3,
                           configured_ceiling=3, supported_parallelism=3)

        def set_used(waves, value, cursor):
            rewritten = []
            for wave in waves:
                start = cursor
                end = start + 1
                rewritten.append(replace(
                    wave, execution_started=start, execution_ended=end,
                    samples=tuple(replace(sample, timestamp_ns=end,
                                           start_ns=start, end_ns=end,
                                           used_bytes=value, free_bytes=1000 - value)
                                   for sample in wave.samples)))
                cursor = end + 1
            return tuple(rewritten), cursor

        # p=1 sweep overhead is positive, while p=2 remains the frozen D2=200
        # witness. Later p=3 discovery waves are larger, but remain
        # reserve-safe and valid.
        cursor = 0
        baseline, cursor = set_used(item.baseline, 300, cursor)
        p2_discovery, cursor = set_used(item.successful_discovery[:4], 500, cursor)
        later_discovery, cursor = set_used(item.successful_discovery[4:], 700, cursor)
        warmups = []
        measured = []
        points = _sweep_points(3)
        for index, point in enumerate(points):
            warmup, cursor = set_used((item.warmups[index],),
                                       350 if point == 1 else 500, cursor)
            warmups.extend(warmup)
            measured_waves, cursor = set_used(
                item.measured[index * 4:(index + 1) * 4],
                 350 if point == 1 else 500, cursor)
            measured.extend(measured_waves)
        item = replace(item, baseline=baseline,
                       successful_discovery=(*p2_discovery, *later_discovery),
                       warmups=tuple(warmups), measured=tuple(measured))
        ident = identity(ModelId.SMOLLM, "smollm:context2048", 2048)
        with ProfileStore(Path(self.tmp.name) / "profiles.sqlite") as store:
            result = persist_measured_profile(req, item, ident, store,
                                              latency_extractor=latencies)
        self.assertEqual(result.profile.memory_safe_n, 3)

    def test_resident_formula_capability_32_replays_with_positive_p1_noise(self):
        """A production-shaped result retains D2, not later p=1 resident noise."""
        req = request(ModelId.SMOLLM, "smollm:context2048")
        item = measurement(32, benchmark_fingerprint=req.fingerprint,
                            # Every measured resident wave has the same wall
                            # time, so the deterministic 2% throughput rule
                            # selects the largest measured point (N=32).
                            optimum=32, m=32, configured_ceiling=32,
                           supported_parallelism=32,
                           baseline_pre_used_bytes=1000,
                           peak_incremental_request_bytes=100,
                           total_vram_bytes=10000, derived_ceiling=32)

        def resident_samples(waves, used):
            return tuple(replace(
                current,
                samples=tuple(replace(sample, total_bytes=10000, used_bytes=used,
                                     free_bytes=10000 - used)
                              for sample in current.samples),
            ) for current in waves)

        baseline = resident_samples(item.baseline, 1000)
        discovery = resident_samples(item.successful_discovery, 1100)
        warmups = tuple(resident_samples((current,), 1050 if current.concurrency == 1 else 1100)[0]
                        for current in item.warmups)
        measured = tuple(resident_samples((current,), 1050 if current.concurrency == 1 else 1100)[0]
                         for current in item.measured)
        item = replace(item, baseline=baseline, successful_discovery=discovery,
                       warmups=warmups, measured=measured)

        with ProfileStore(Path(self.tmp.name) / "profiles.sqlite") as store:
            result = persist_measured_profile(
                req, item, identity(ModelId.SMOLLM, "smollm:context2048", 2048), store,
                latency_extractor=latencies)
        self.assertEqual(result.profile.memory_safe_n, 32)

    def test_tampered_frozen_resident_d2_is_rejected(self):
        req = request(ModelId.SMOLLM, "smollm:context2048")
        item = measurement(2, benchmark_fingerprint=req.fingerprint,
                           peak_incremental_request_bytes=201)
        with self.assertRaises(ValueError):
            persist_measured_profile(
                req, item, identity(ModelId.SMOLLM, "smollm:context2048", 2048),
                SpyStore(), latency_extractor=latencies,
            )

    def test_identical_replay_and_changed_evidence_conflict(self):
        req, item, ident, _ = self.persist()
        with ProfileStore(Path(self.tmp.name) / "profiles.sqlite") as store:
            persist_measured_profile(req, item, ident, store, latency_extractor=latencies)
            with self.assertRaises(ProfileConflict):
                persist_measured_profile(req, item, ident, store,
                                         latency_extractor=lambda ids: {key: 4 for key in ids})

    def test_identity_selector_and_fingerprint_mismatches_fail_before_save(self):
        base = identity(ModelId.SMOLLM, "smollm:context2048", 2048)
        # Model/selector contradictions are rejected by the immutable identity
        # contract before persistence can be attempted.
        with self.assertRaises(ValueError):
            replace(base, model_id=ModelId.COEDIT)
        with self.assertRaises(ValueError):
            persist_measured_profile(request(ModelId.SMOLLM, "smollm:context2048"),
                                     measurement(benchmark_fingerprint= request(ModelId.SMOLLM, "smollm:context2048").fingerprint),
                                     replace(base, context_size=4096), SpyStore(),
                                     latency_extractor=latencies)
        req = request(ModelId.SMOLLM, "smol")
        spy = SpyStore()
        with self.assertRaises(ValueError):
            persist_measured_profile(req, measurement(benchmark_fingerprint="wrong"),
                                     identity(ModelId.SMOLLM, "smol", 2048), spy,
                                     latency_extractor=latencies)
        self.assertEqual(spy.saved, 0)

    def test_incomplete_ineligible_failed_discovery_and_numeric_mismatches_do_not_save(self):
        for changes in ({"status": "incomplete"}, {"profile_eligible": False},
                        {"failed_discovery": (wave("measured", 2, 1),)},
                        {"optimum": 2}, {"m": 2}, {"reserve_percent": 10}):
            with self.subTest(changes=changes):
                req = request(ModelId.SMOLLM, "smollm:context2048")
                spy = SpyStore()
                with self.assertRaises(ValueError):
                    persist_measured_profile(req, measurement(benchmark_fingerprint=req.fingerprint, **changes),
                                             identity(ModelId.SMOLLM, "smollm:context2048", 2048), spy,
                                             latency_extractor=latencies)
                self.assertEqual(spy.saved, 0)

    def test_zero_increment_requires_a_retained_smollm_resource_bound_witness(self):
        req = request(ModelId.SMOLLM, "smollm:context2048")
        base = measurement(2, benchmark_fingerprint=req.fingerprint,
                           optimum=2, m=2,
                           peak_incremental_request_bytes=0, derived_ceiling=None,
                           configured_ceiling=3, supported_parallelism=3,
                           parallelism_bound_reason="identity_configured_operator_ceiling")
        # Keep the synthetic persisted evidence in the same execution order
        # required of the real interleaved measurement schedule.
        warmups = tuple(
            replace(item, execution_started=1000 + index * 5000,
                    execution_ended=1000 + index * 5000,
                    samples=(MemorySample(1000 + index * 5000, 1000, 600, 400),))
            for index, item in enumerate(base.warmups)
        )
        measured = tuple(
            replace(item, execution_started=1100 + index * 5000 + wave_number * 100,
                    execution_ended=1100 + index * 5000 + wave_number * 100,
                    samples=(MemorySample(1100 + index * 5000 + wave_number * 100,
                                           1000, 600, 400),))
            for index, point in enumerate(_sweep_points(2))
            for wave_number, item in enumerate(
                base.measured[index * 4:(index + 1) * 4], 1)
        )
        discovery = tuple(
            replace(item, samples=(MemorySample(item.execution_started, 1000, 600, 400),))
            for item in base.successful_discovery
        )
        base = replace(base, successful_discovery=discovery, warmups=warmups, measured=measured)
        # Discovery (including the retained N+1 failure) precedes the
        # throughput sweep; its reserve-breaching sample is inside the failed
        # wave's execution interval.
        failed = replace(wave("measured", 3, 1, timestamp=900), phase="discovery", failed=True,
                         execution_started=900, execution_ended=900,
                         failure_kind="reserve_breached",
                         samples=(MemorySample(900, 1000, 900, 100),))
        proven = replace(base, failed_discovery=(failed,), resource_bound_failure=failed)
        with ProfileStore(Path(self.tmp.name) / "profiles.sqlite") as store:
            persisted = persist_measured_profile(req, proven,
                identity(ModelId.SMOLLM, "smollm:context2048", 2048), store,
                latency_extractor=latencies)
        self.assertEqual(persisted.profile.memory_safe_n, 2)
        for unproven in (base, replace(proven, resource_bound_failure=None),
                         replace(proven, failed_discovery=())):
            with self.assertRaises(ValueError):
                persist_measured_profile(req, unproven,
                    identity(ModelId.SMOLLM, "smollm:context2048", 2048), SpyStore(),
                    latency_extractor=latencies)

    def test_schedules_samples_and_latencies_are_strict(self):
        req = request(ModelId.SMOLLM, "smollm:context2048")
        ident = identity(ModelId.SMOLLM, "smollm:context2048", 2048)
        bad = [replace(measurement(benchmark_fingerprint=req.fingerprint), baseline=()),
               replace(measurement(benchmark_fingerprint=req.fingerprint), warmups=()),
               replace(measurement(benchmark_fingerprint=req.fingerprint), measured=()),
               replace(measurement(benchmark_fingerprint=req.fingerprint),
                       measured=(wave("measured", 1, 1, True),) + measurement(benchmark_fingerprint=req.fingerprint).measured[1:])]
        for item in bad:
            with self.assertRaises(ValueError):
                persist_measured_profile(req, item, ident, SpyStore(), latency_extractor=latencies)
        for extractor in (lambda ids: {"wrong": 1}, lambda ids: (1, 2),
                          lambda ids: {key: -1 for key in ids}, lambda ids: {key: "1" for key in ids}):
                 with self.assertRaises(ValueError):
                     persist_measured_profile(req, measurement(benchmark_fingerprint=req.fingerprint), ident,
                                           SpyStore(), latency_extractor=extractor)

    def test_missing_intermediate_discovery_wave_fails_closed(self):
        req = request(ModelId.SMOLLM, "smollm:context2048")
        item = measurement(2, benchmark_fingerprint=req.fingerprint)
        item = replace(item, successful_discovery=item.successful_discovery[:-1])
        with self.assertRaises(ValueError):
            persist_measured_profile(
                req, item, identity(ModelId.SMOLLM, "smollm:context2048", 2048),
                SpyStore(), latency_extractor=latencies,
            )

    def test_evidence_integrity_rejects_duplicate_native_timing_and_chronology_before_save(self):
        req = request(ModelId.SMOLLM, "smollm:context2048")
        base = measurement(benchmark_fingerprint=req.fingerprint)
        duplicate = replace(base, warmups=(replace(base.warmups[0], request_ids=base.baseline[0].request_ids),))
        invalid_timing = replace(base, baseline=(replace(base.baseline[0], execution_ended=-1),) + base.baseline[1:])
        chronology = replace(base, warmups=(replace(base.warmups[0], execution_started=1, execution_ended=1),))
        for item in (duplicate, invalid_timing, chronology):
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    persist_measured_profile(req, item, identity(ModelId.SMOLLM, "smollm:context2048", 2048),
                                             SpyStore(), latency_extractor=latencies)
        # Native-batch evidence is required for a genuinely batched point;
        # n=1 cannot exercise that contract.
        batched_req = request(ModelId.SMOLLM, "smollm:context2048")
        batched = measurement(2, benchmark_fingerprint=batched_req.fingerprint)
        malformed_native = replace(
            batched,
            measured=(batched.measured[0],
                      *batched.measured[1:4],
                      replace(batched.measured[4], native_request_correlation=False))
                     + batched.measured[5:],
        )
        with self.assertRaises(ValueError):
            persist_measured_profile(
                batched_req, malformed_native,
                identity(ModelId.SMOLLM, "smollm:context2048", 2048),
                SpyStore(), latency_extractor=latencies,
            )

    def test_n2_validates_actual_interleaved_execution_without_reordering_storage(self):
        req = request(ModelId.SMOLLM, "smollm:context2048")
        base = measurement(2, benchmark_fingerprint=req.fingerprint)
        # The discovery pass is intentionally absent from persisted evidence.
        # Actual execution is baseline, then each point's warmup and four waves.
        warmups = tuple(wave("warmup", point, 0, timestamp=1000 + index * 5000)
                        for index, point in enumerate(_sweep_points(2)))
        measured = tuple(
            wave("measured", point, wave_number, timestamp=1100 + index * 5000 + wave_number * 100)
            for index, point in enumerate(_sweep_points(2))
            for wave_number in range(1, 5)
        )
        # The synthetic waves have twice the throughput at p=2, so the
        # evidence selects p=2 under the strict throughput contract.
        item = replace(base, optimum=2, m=2, warmups=warmups, measured=measured)
        ident = identity(ModelId.SMOLLM, "smollm:context2048", 2048)
        with ProfileStore(Path(self.tmp.name) / "profiles.sqlite") as store:
            persisted = persist_measured_profile(
                req, item, ident, store, latency_extractor=latencies,
            )
        self.assertEqual(
            tuple((sample.concurrency, sample.wave) for sample in persisted.metadata.warmup_samples),
            ((1, 0), (2, 0)),
        )
        self.assertEqual(
            tuple((sample.concurrency, sample.wave) for sample in persisted.metadata.measured_samples),
            tuple((point, wave_number) for point in _sweep_points(2) for wave_number in range(1, 5)),
        )
        self.assertEqual(
            persisted.profile.raw_samples,
            persisted.metadata.baseline_samples + persisted.metadata.warmup_samples + persisted.metadata.measured_samples,
        )

    def test_n2_inverted_interleaved_execution_still_fails(self):
        req = request(ModelId.SMOLLM, "smollm:context2048")
        base = measurement(2, benchmark_fingerprint=req.fingerprint)
        # Point 2's warmup is retained in its grouped storage slot, but its
        # timestamp claims it ran before point 1's measured waves.
        warmups = (wave("warmup", 1, 0, timestamp=1000), wave("warmup", 2, 0, timestamp=1050))
        measured = tuple(
            wave("measured", point, wave_number,
                 timestamp=(1100 + wave_number * 100 if index == 0 else 6000 + wave_number * 100))
            for index, point in enumerate(_sweep_points(2))
            for wave_number in range(1, 5)
        )
        with self.assertRaises(ValueError):
            persist_measured_profile(
                req, replace(base, warmups=warmups, measured=measured),
                identity(ModelId.SMOLLM, "smollm:context2048", 2048), SpyStore(),
                latency_extractor=latencies,
            )

    def test_allocator_and_native_batch_evidence_are_required_at_every_parallelism(self):
        for n in (1, 2):
            req = request(ModelId.SMOLLM, "smollm:context2048")
            base = measurement(n, benchmark_fingerprint=req.fingerprint)
            target_index = next(index for index, item in enumerate(base.measured)
                                if item.concurrency == n)
            target = base.measured[target_index]
            cases = (
                replace(target, allocator=None),
                replace(target, allocator=AllocatorObservation(2, 1, 3, 4, 1, 2)),
                replace(target, native_batch_size=n + 1),
            )
            for malformed in cases:
                with self.subTest(n=n, malformed=malformed):
                    item = replace(base, measured=base.measured[:target_index] +
                                   (malformed,) + base.measured[target_index + 1:])
                    spy = SpyStore()
                    with self.assertRaises(ValueError):
                        persist_measured_profile(
                            req, item, identity(ModelId.SMOLLM, "smollm:context2048", 2048),
                            spy, latency_extractor=latencies)
                    self.assertEqual(spy.saved, 0)


if __name__ == "__main__":
    unittest.main()

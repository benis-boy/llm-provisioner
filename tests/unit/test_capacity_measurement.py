import asyncio
import tempfile
import unittest
import itertools
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from services.llm.provisioning.capacity import (
    MemorySample, Wave, choose_optimum, measure_capacity, discover_memory,
    CapacityEvidenceError, canonical_benchmark_payload, sample_during,
)
from services.llm.providers.coedit_batch import AllocatorObservation
from services.llm.provisioning import capacity as capacity_module

ALLOCATOR = AllocatorObservation(10, 20, 30, 40, 10, 20)


def wave(p, n, ms=10, *, free=100):
    return Wave(p, n, tuple(f"r{i}" for i in range(p)), ms, True, p,
                   execution_started=1_100_000_000, execution_ended=2_000_000_000, cuda_synchronized=True, allocator=ALLOCATOR,
                 samples=(MemorySample(1_500_000_000, 100, 100 - free, free,
                                       1_500_000_000, 1_500_001_000),), decoder_steps=(64,) * p,
                 max_output_tokens=64, observation_count=1,
                 observed_native_batch_sizes=(p,), native_request_correlation=True,
                 observation_drops=0)


_sample_clock = itertools.count()


def sampled(total=100, used=0, free=None):
    start = 1_100_000_000 + next(_sample_clock) * 1_000_000
    free = total - used if free is None else free
    return MemorySample(start + 1_000, total, used, free, start, start + 1_000)


class CapacityMeasurementTests(unittest.TestCase):
    def _discover_with_fenced_samples(self, runner, *, mutate=None, error_kind=None):
        clock = itertools.count(1)
        async def stub(_sampler, action, **_kwargs):
            item = await action()
            start = 1_200_000_000 + next(clock) * 10_000
            samples = tuple(MemorySample(start + offset + 1, 100, 0, 100,
                                         start + offset, start + offset + 1)
                            for offset in (0, 2_000, 4_000))
            if error_kind and item.concurrency == 2 and item.wave == 2:
                exc = CapacityEvidenceError("synthetic failure")
                setattr(exc, "capacity_samples", samples)
                setattr(exc, "capacity_failure_kind", error_kind)
                raise exc
            item = replace(item, samples=samples)
            return mutate(item, samples) if mutate else item
        with patch.object(capacity_module, "sample_during", new=stub):
            return asyncio.run(discover_memory(runner, object(), max_parallelism=3,
                                                expected_max_output_tokens=64))
    def test_discovery_is_exactly_incremental_four_repeat_and_globally_unique(self):
        calls = []
        async def run(p, n, ids):
            calls.append((p, n, ids)); await asyncio.sleep(.02)
            return Wave(p, n, ids, 10, True, p, 1_100_000_000, 2_000_000_000, True,
                        allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64,
                        observation_count=1, observed_native_batch_sizes=(p,), native_request_correlation=True)
        class Sampler:
            calls = 0
            async def sample(self):
                self.calls += 1
                start = 1_500_000_000 + self.calls * 1_000_000
                return MemorySample(start + 1_000, 100, 0, 100, start, start + 1_000)
        result = asyncio.run(discover_memory(run, Sampler(), max_parallelism=3,
                                             expected_max_output_tokens=64))
        self.assertEqual(result.status, "complete")
        self.assertEqual([(p, n) for p, n, _ in calls],
                         [(1, n) for n in range(1, 5)] +
                         [(p, n) for p in (2, 3) for n in range(1, 5)])
        identities = [item for _, _, values in calls for item in values]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(result.observed_safe_through, 3)
        self.assertIsNone(result.memory_safe_n)
        self.assertFalse(result.profile_eligible)
        self.assertTrue(result.max_output_verified)

    def test_discovery_supports_full_memory_ceiling_without_changing_throughput_points(self):
        calls = []
        async def run(p, n, ids):
            calls.append((p, n, len(ids)))
            await asyncio.sleep(.02)
            return replace(wave(p, n), request_ids=ids,
                           execution_started=1_100_000_000, execution_ended=2_000_000_000)
        class Sampler:
            calls = 0
            async def sample(self):
                self.calls += 1
                start = 1_100_000_000 + self.calls * 10_000
                return MemorySample(start + 1, 100, 0, 100, start, start + 1)
        result = asyncio.run(discover_memory(run, Sampler(), max_parallelism=32,
                                             expected_max_output_tokens=64))
        self.assertEqual(result.observed_safe_through, 32)
        self.assertEqual(len(calls), 4 + 31 * 4)
        self.assertEqual(calls[-1], (32, 4, 32))
        self.assertTrue(result.max_output_verified)
        self.assertIsNone(result.memory_safe_n)
        self.assertFalse(result.profile_eligible)

    def test_full_synthetic_discovery_has_128_waves_2112_requests_and_exact_decoder_witness(self):
        calls = []
        async def run(p, n, ids):
            calls.append((p, n, ids))
            await asyncio.sleep(.02)
            return replace(wave(p, n), request_ids=ids, decoder_steps=(64,) * p,
                           max_output_tokens=64, execution_started=1_100_000_000,
                           execution_ended=2_000_000_000)
        class Sampler:
            calls = 0
            async def sample(self):
                self.calls += 1
                start = 1_100_000_000 + self.calls * 10_000
                return MemorySample(start + 1, 100, 0, 100, start, start + 1)
        result = asyncio.run(discover_memory(run, Sampler(), max_parallelism=32,
                                             expected_max_output_tokens=64))
        self.assertEqual(result.status, "complete")
        self.assertEqual(len(calls), 128)
        self.assertEqual(sum(len(ids) for _, _, ids in calls), 2112)
        self.assertTrue(result.max_output_verified)
        self.assertIsNone(result.memory_safe_n)
        self.assertFalse(result.profile_eligible)

    def test_discovery_stops_at_failed_point_and_never_proves_preceding_point(self):
        for failure_kind, expected_reason in (("reserve", "reserve_breached"),
                                               ("decoder_workload", "undercovered_decoder"),
                                               ("identity", "invalid_discovery"),
                                               ("chronology", "invalid_discovery"),
                                               ("allocator", "invalid_discovery"),
                                               ("timing", "invalid_discovery"),
                                               ("sampler_error", "invalid_discovery"),
                                               ("timeout", "invalid_discovery"),
                                               ("native_batch_correlation", "invalid_discovery")):
            calls = []
            async def run(p, n, ids, kind=failure_kind):
                calls.append((p, n))
                if p == 2 and n == 2:
                    if kind == "native_batch_correlation":
                        return Wave(p, n, ids, 0, False, 0, failed=True, failure_kind=kind,
                                    observation_count=1, observed_native_batch_sizes=(p,), native_request_correlation=False)
                    if kind == "timing":
                        return Wave(p, n, ids, 10, True, p, 1.0, 2.0, True,
                                    allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64,
                                    observation_count=1, observed_native_batch_sizes=(p,), native_request_correlation=True)
                    if kind == "allocator":
                        return Wave(p, n, ids, 10, True, p, 1_100_000_000, 2_000_000_000, True,
                                    allocator=None, decoder_steps=(64,) * p, max_output_tokens=64,
                                    observation_count=1, observed_native_batch_sizes=(p,), native_request_correlation=True)
                    if kind == "decoder_workload":
                        return Wave(p, n, ids, 10, True, p, 1_100_000_000, 2_000_000_000, True,
                                    allocator=ALLOCATOR, decoder_steps=(63,) * p, max_output_tokens=64,
                                    observation_count=1, observed_native_batch_sizes=(p,), native_request_correlation=True)
                return replace(wave(p, n), request_ids=ids)
            def mutate(item, samples):
                if item.concurrency != 2 or item.wave != 2: return item
                if failure_kind == "reserve":
                    return replace(item, samples=tuple(replace(s, used_bytes=90, free_bytes=10) for s in samples))
                if failure_kind == "identity": return replace(item, request_ids=("foreign-1", "foreign-2"))
                if failure_kind == "chronology": return replace(item, samples=(samples[0], samples[0], samples[2]))
                if failure_kind == "allocator": return replace(item, allocator=None)
                if failure_kind == "timing": return replace(item, execution_started=1.0, execution_ended=2.0)
                if failure_kind == "decoder_workload": return replace(item, decoder_steps=(63,) * 2)
                if failure_kind == "native_batch_correlation": return replace(item, native_request_correlation=False)
                return item
            result = self._discover_with_fenced_samples(run, mutate=mutate,
                error_kind=failure_kind if failure_kind in {"sampler_error", "timeout"} else None)
            expected_kind = "undercovered_decoder" if failure_kind == "decoder_workload" else failure_kind
            self.assertEqual((result.reason, result.failure_kind), (expected_reason, expected_kind))
            self.assertEqual(calls[-1], (2, 2))
            self.assertNotIn((2, 3), calls)
            self.assertEqual(result.observed_safe_through, 1 if failure_kind == "reserve" else None)
            self.assertFalse(result.profile_eligible)
            self.assertIsNone(result.memory_safe_n)

    def test_discovery_rejects_native_observation_count_or_batch_mismatch_and_retains_failed_wave(self):
        for changes in ({"observation_count": 2}, {"observed_native_batch_sizes": (1,)},
                        {"native_request_correlation": False}, {"observation_drops": 1}):
            calls = []
            async def run(p, n, ids, changes=changes):
                calls.append((p, n)); item = replace(wave(p, n), request_ids=ids)
                return replace(item, **changes) if p == 2 and n == 1 else item
            result = self._discover_with_fenced_samples(run)
            self.assertEqual(result.failure_kind, "native_batch_correlation")
            self.assertTrue(result.points[-1].failed)
            self.assertEqual(result.points[-1].failure_kind, "native_batch_correlation")
            self.assertEqual(calls[-1], (2, 1))
            self.assertNotIn((2, 2), calls)

    def test_discovery_missing_native_observation_fields_fails_closed_after_authoritative_baseline(self):
        async def run(p, n, ids):
            if p == 2:
                # Defaults model an incomplete real observation, rather than a
                # synthetic baseline fixture accidentally omitting evidence.
                return Wave(p, n, ids, 10, True, p, 1_100_000_000, 2_000_000_000, True,
                            allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64)
            return Wave(p, n, ids, 10, True, p, 1_100_000_000, 2_000_000_000, True,
                        allocator=ALLOCATOR, decoder_steps=(64,), max_output_tokens=64,
                        observation_count=1, observed_native_batch_sizes=(1,),
                        native_request_correlation=True)
        result = self._discover_with_fenced_samples(run)
        self.assertEqual((result.failure_phase, result.failure_kind, result.observed_safe_through),
                         ("discovery", "native_batch_correlation", None))
        self.assertTrue(result.points[-1].failed)

    def test_discovery_rejects_changed_total_and_request_identity_without_next_point(self):
        for mode in ("total", "request"):
            calls = []
            async def run(p, n, ids):
                calls.append((p, n))
                if mode == "request" and p == 2 and n == 1:
                    ids = ("foreign",) * p
                return Wave(p, n, ids, 10, True, p, 1_100_000_000, 2_000_000_000, True,
                            allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64,
                            observation_count=1, observed_native_batch_sizes=(p,), native_request_correlation=True)
            def mutate(item, samples):
                if mode == "total" and item.concurrency == 2 and item.wave == 1:
                    return replace(item, samples=tuple(replace(s, total_bytes=101, free_bytes=101)
                                                       for s in samples))
                return item
            result = self._discover_with_fenced_samples(run, mutate=mutate)
            self.assertEqual(result.failure_kind, "identity")
            self.assertNotIn((2, 2), calls)

    def test_discovery_maximum_verification_requires_exact_complete_schedule_and_type(self):
        def item(p, w, phase):
            return Wave(p, w, tuple(f"{phase}-{p}-{w}-{i}" for i in range(p)), 1, True, p,
                1, 2, True, allocator=ALLOCATOR, phase=phase,
                decoder_steps=(64,) * p, max_output_tokens=64,
                observation_count=1, observed_native_batch_sizes=(p,), native_request_correlation=True)
        result = capacity_module.DiscoveryResult("complete", 2, 2, None, "observed_through_ceiling",
            baseline=tuple(item(1, w, "baseline") for w in range(1, 5)),
            points=tuple(item(p, w, "discovery") for p in (2,) for w in range(1, 5)),
            expected_max_output_tokens=64)
        self.assertTrue(result.max_output_verified)
        self.assertFalse(result.__class__(**{**result.__dict__, "points": result.points[:-1]}).max_output_verified)
        self.assertFalse(result.__class__(**{**result.__dict__, "expected_max_output_tokens": True}).max_output_verified)
    def test_max_output_verification_requires_complete_exact_phase_schedule(self):
        def item(p, w, phase):
            return Wave(p, w, tuple(f"{phase}-{w}-{i}" for i in range(p)), 1, True, p,
                1_100_000_000, 2_000_000_000, True,
                (MemorySample(1_500_000_000, 100, 0, 100),), allocator=ALLOCATOR,
                phase=phase, decoder_steps=(64,) * p, max_output_tokens=64)
        result = capacity_module.CapacityResult("candidate", 2, 2, "candidate",
            tuple(item(p, w, "measured") for p in (1, 2) for w in range(1, 5)),
            tuple(item(1, w, "baseline") for w in range(1, 5)),
            tuple(item(p, 0, "warmup") for p in (1, 2)), expected_max_output_tokens=64)
        self.assertTrue(result.max_output_verified)
        self.assertFalse(result.__class__(**{**result.__dict__, "points": result.points[:-1]}).max_output_verified)
        self.assertFalse(result.__class__(**{**result.__dict__, "points": result.points[:-1] + (item(2, 3, "measured"),)}).max_output_verified)

    def test_expected_maximum_is_required_strict_and_mismatch_is_incomplete(self):
        class Sampler:
            async def sample(self): return sampled()
        async def run(p, n, ids):
            await asyncio.sleep(.02)
            return Wave(p, n, ids, 1, True, p, 1_100_000_000, 2_000_000_000, True,
                allocator=ALLOCATOR, decoder_steps=(63,) * p, max_output_tokens=63)
        with self.assertRaises(ValueError): asyncio.run(measure_capacity(run, Sampler(), max_parallelism=1, expected_max_output_tokens=True))
        result = asyncio.run(measure_capacity(run, Sampler(), max_parallelism=1, expected_max_output_tokens=64))
        self.assertEqual((result.status, result.failure_kind), ("incomplete", "decoder_workload"))

    def test_valid_short_decoder_rows_remain_candidate_but_never_verify_maximum(self):
        class Sampler:
            async def sample(self): return sampled()
        async def run(p, n, ids):
            await asyncio.sleep(.02)
            return Wave(p, n, ids, 1, True, p, 1_100_000_000, 2_000_000_000, True,
                allocator=ALLOCATOR, decoder_steps=(1,) * p, max_output_tokens=64)
        result = asyncio.run(measure_capacity(run, Sampler(), max_parallelism=1, expected_max_output_tokens=64))
        self.assertEqual(result.status, "candidate")
        self.assertFalse(result.max_output_verified)
    def test_sampling_bound_is_16384_at_independent_ten_ms_with_explicit_120_second_timeout(self):
        self.assertEqual(capacity_module.MAX_SAMPLES_PER_WAVE, 16384)
        self.assertEqual(capacity_module.SAMPLE_INTERVAL_SECONDS, .01)
        self.assertEqual(capacity_module.SAMPLE_TIMEOUT_SECONDS, 120.0)

    def test_two_percent_tie_prefers_lower_point(self):
        self.assertEqual(choose_optimum({1: [wave(1, 1, 10)] * 4,
                                          2: [wave(2, 1, 20)] * 4}), 1)

    def test_failed_point_returns_previous_candidate_without_approval(self):
        async def run(p, n, ids):
            await asyncio.sleep(.02)
            return Wave(p, n, ids, 10, True, p, 1_100_000_000, 2_000_000_000, True, (MemorySample(1_500_000_000, 100, 0, 100),), allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64) if p < 2 else Wave(p, n, ids, 10, False, p,
                                                   cuda_synchronized=True, allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64,
                                                   samples=(MemorySample(2, 100, 90, 10),), failed=True)
        class Sampler:
            async def sample(self): return sampled()
        result = asyncio.run(measure_capacity(run, Sampler(), max_parallelism=2, expected_max_output_tokens=64))
        self.assertEqual((result.status, result.candidate_n), ("incomplete", 1))

    def test_probe_ceiling_is_candidate_not_measured_or_profile_eligible(self):
        async def run(p, n, ids):
            await asyncio.sleep(.02)
            return Wave(p, n, ids, 10, True, p, 1_100_000_000, 2_000_000_000, True, (MemorySample(1_500_000_000, 100, 0, 100),), allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64)
        class Sampler:
            async def sample(self): return sampled()
        result = asyncio.run(measure_capacity(run, Sampler(), max_parallelism=2, expected_max_output_tokens=64))
        self.assertEqual((result.status, result.reason, result.profile_eligible), ("candidate", "probe_limit_not_memory_bound", False))

    def test_all_phases_receive_exactly_unique_request_ids(self):
        seen = []
        async def run(p, n, ids):
            seen.append((p, n, ids))
            await asyncio.sleep(.02)
            return Wave(p, n, ids, 10, True, p, 1_100_000_000, 2_000_000_000, True,
                         (MemorySample(1_500_000_000, 100, 0, 100),), allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64)
        class Sampler:
            async def sample(self): return sampled()
        result = asyncio.run(measure_capacity(run, Sampler(), max_parallelism=2, expected_max_output_tokens=64))
        self.assertEqual(result.status, "candidate")
        ids = [request_id for _, _, wave_ids in seen for request_id in wave_ids]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(seen), 4 + 2 * 5)
        self.assertTrue(all(item[2] for item in seen))

    def test_duplicate_supplied_identity_fails_closed(self):
        async def run(p, n, ids):
            await asyncio.sleep(.02)
            return Wave(p, n, ids, 10, True, p, 1_100_000_000, 2_000_000_000, True,
                        (MemorySample(1_500_000_000, 100, 0, 100,
                                       1_500_000_000, 1_500_001_000),), allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64)
        class Sampler:
            async def sample(self): return sampled()
        with self.assertRaisesRegex(RuntimeError, "request identities are not globally unique"):
            asyncio.run(measure_capacity(run, Sampler(), max_parallelism=1, expected_max_output_tokens=64,
                                         request_ids=lambda p, n: tuple("same" for _ in range(p))))

    def test_sample_timing_and_identity_are_strict(self):
        async def run(p, n, ids):
            await asyncio.sleep(.02)
            return Wave(p, n, ids, 10, True, p, 1_100_000_000, 2_000_000_000, True,
                        (MemorySample(1_500_000_000, 100, 0, 100),), allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64)
        class Sampler:
            calls = 0
            async def sample(self):
                self.calls += 1
                total = 100 if self.calls == 1 else 101
                return sampled(total)
        result = asyncio.run(measure_capacity(run, Sampler(), max_parallelism=1, expected_max_output_tokens=64))
        self.assertEqual(result.reason, "telemetry_identity_changed")
        self.assertFalse(result.profile_eligible)

    def test_sample_outside_execution_window_is_rejected(self):
        async def run(p, n, ids):
            await asyncio.sleep(.02)
            return Wave(p, n, ids, 10, True, p, 1, 2, True,
                        (MemorySample(1_500_000_000, 100, 0, 100),
                         MemorySample(1_500_000_000, 100, 0, 100)))
        class Sampler:
            async def sample(self): return MemorySample(3_000_000_000, 100, 0, 100,
                                                        3_000_000_000, 3_000_001_000)
        result = asyncio.run(measure_capacity(run, Sampler(), max_parallelism=1, expected_max_output_tokens=64))
        self.assertEqual(result.reason, "invalid_serial_baseline")

    def test_non_ns_execution_fence_is_rejected_with_phase_and_failure_category(self):
        async def run(p, n, ids):
            await asyncio.sleep(.02)
            return Wave(p, n, ids, 10, True, p, 1.0, 2.0, True,
                        (MemorySample(1_500_000_000, 100, 0, 100),), allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64)
        class Sampler:
            async def sample(self): return sampled()
        result = asyncio.run(measure_capacity(run, Sampler(), max_parallelism=1, expected_max_output_tokens=64))
        self.assertEqual((result.reason, result.failure_phase, result.failure_kind),
                         ("invalid_serial_baseline", "baseline", "timing"))

    def test_invalid_sample_and_reserve_are_categorized_with_phase(self):
        async def invalid(p, n, ids):
            await asyncio.sleep(.02)
            return Wave(p, n, ids, 10, True, p, 1_100_000_000, 2_000_000_000, True,
                        allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64)
        class Sampler:
            calls = 0
            async def sample(self):
                self.calls += 1
                if self.calls == 2:
                    return MemorySample(1_500_000_000, 100, 101, 0,
                                        1_500_000_000, 1_500_001_000)
                return sampled()
        result = asyncio.run(measure_capacity(invalid, Sampler(), max_parallelism=1, expected_max_output_tokens=64))
        self.assertEqual((result.failure_phase, result.failure_kind), ("baseline", "invalid_sample"))

    def test_explicit_runner_correlation_failure_is_not_reclassified_as_identity(self):
        async def run(p, n, ids):
            await asyncio.sleep(.02)
            return Wave(p, n, ids, 0, False, 0, failed=True, failure_kind="native_batch_correlation")
        class Sampler:
            async def sample(self): return sampled()
        result = asyncio.run(measure_capacity(run, Sampler(), max_parallelism=1, expected_max_output_tokens=64))
        self.assertEqual((result.reason, result.failure_phase, result.failure_kind),
                         ("invalid_serial_baseline", "baseline", "native_batch_correlation"))

    def test_derived_failure_replaces_retained_wave_with_category_and_failed_flag(self):
        async def run(p, n, ids):
            await asyncio.sleep(.02)
            return Wave(p, n, ids, 10, True, p, 1_100_000_000, 2_000_000_000, True,
                        allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64)
        class Sampler:
            async def sample(self): return sampled(free=10)
        result = asyncio.run(measure_capacity(run, Sampler(), max_parallelism=1, expected_max_output_tokens=64))
        retained = result.baseline[0]
        self.assertEqual((result.failure_phase, result.failure_kind), ("baseline", "reserve"))
        self.assertEqual((retained.failed, retained.failure_kind), (True, "reserve"))

    def test_total_change_stops_after_the_first_changed_baseline_wave(self):
        calls = []
        state = {"wave": 0}
        async def run(p, n, ids):
            state["wave"] += 1; calls.append((p, n))
            await asyncio.sleep(.02)
            return Wave(p, n, ids, 10, True, p, 1_100_000_000, 2_000_000_000, True, allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64)
        class Sampler:
            async def sample(self):
                return sampled(total=100 if state["wave"] < 2 else 101)
        result = asyncio.run(measure_capacity(run, Sampler(), max_parallelism=1, expected_max_output_tokens=64))
        self.assertEqual((result.reason, result.failure_phase, result.failure_kind),
                         ("telemetry_identity_changed", "baseline", "identity"))
        self.assertEqual(calls, [(1, 1), (1, 2)])

    def test_no_execution_sample_and_chronology_replace_retained_wave(self):
        async def no_overlap(p, n, ids):
            await asyncio.sleep(.02)
            return Wave(p, n, ids, 10, True, p, 1, 2, True, allocator=ALLOCATOR, decoder_steps=(64,) * p, max_output_tokens=64)
        class Outside:
            async def sample(self): return MemorySample(10, 100, 0, 100)
        no_execution = asyncio.run(measure_capacity(no_overlap, Outside(), max_parallelism=1, expected_max_output_tokens=64))
        self.assertEqual((no_execution.failure_kind, no_execution.baseline[0].failed,
                          no_execution.baseline[0].failure_kind), ("no_execution_sample", True, "no_execution_sample"))

        values = iter((MemorySample(100, 100, 0, 100, 100, 100),
                       MemorySample(50, 100, 0, 100, 50, 50),
                       *(MemorySample(200, 100, 0, 100, 200, 200) for _ in range(8))))
        class Backward:
            async def sample(self): return next(values)
        chronological = asyncio.run(measure_capacity(
            lambda p, n, ids: no_overlap(p, n, ids), Backward(), max_parallelism=1, expected_max_output_tokens=64))
        self.assertEqual((chronological.failure_kind, chronological.baseline[0].failed,
                          chronological.baseline[0].failure_kind), ("chronology", True, "chronology"))

    def test_pre_sample_failure_does_not_start_action(self):
        called = False
        class Sampler:
            async def sample(self): raise RuntimeError("unavailable")
        async def action():
            nonlocal called
            called = True
            return wave(1, 1)
        with self.assertRaises(RuntimeError): asyncio.run(sample_during(Sampler(), action))
        self.assertFalse(called)

    def test_sampler_error_cancels_owned_action(self):
        cancelled = False
        class Sampler:
            count = 0
            async def sample(self):
                self.count += 1
                if self.count == 2: raise RuntimeError("sampler failed")
                return MemorySample(1_500_000_000, 100, 0, 100)
        async def action():
            nonlocal cancelled
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                cancelled = True
                raise
            return wave(1, 1)
        with self.assertRaises(RuntimeError):
            asyncio.run(sample_during(Sampler(), action, interval=.001))
        self.assertTrue(cancelled)

    def test_runner_failure_retains_failed_wave_identity_and_samples(self):
        async def run(p, n, ids):
            raise RuntimeError("payload must not escape")
        class Sampler:
            async def sample(self): return sampled()
        result = asyncio.run(measure_capacity(run, Sampler(), max_parallelism=1, expected_max_output_tokens=64))
        failed = result.baseline[0]
        self.assertTrue(failed.failed)
        self.assertEqual(failed.failure_kind, "runner_error")
        self.assertEqual(len(failed.request_ids), 1)
        self.assertTrue(failed.samples)

    def test_sampler_failure_retains_partial_wave_samples(self):
        async def run(p, n, ids):
            await asyncio.sleep(.02)
            return wave(p, n)
        class Sampler:
            calls = 0
            async def sample(self):
                self.calls += 1
                if self.calls == 2: raise RuntimeError("not telemetry")
                return MemorySample(1_500_000_000, 100, 0, 100)
        result = asyncio.run(measure_capacity(run, Sampler(), max_parallelism=1, expected_max_output_tokens=64))
        self.assertTrue(result.baseline[0].failed)
        self.assertEqual(result.baseline[0].failure_kind, "sampler_error")
        self.assertEqual(len(result.baseline[0].samples), 1)

    def test_caller_cancellation_cleans_sampler_and_action(self):
        cancelled = False
        class Sampler:
            async def sample(self): return sampled()
        async def action():
            nonlocal cancelled
            try: await asyncio.sleep(1)
            except asyncio.CancelledError:
                cancelled = True
                raise
        async def invoke():
            task = asyncio.create_task(sample_during(Sampler(), action))
            await asyncio.sleep(.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
        asyncio.run(invoke())
        self.assertTrue(cancelled)

    def test_input_digest_uses_canonical_payload_and_child_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_text("hello", encoding="utf-8")
            payload, digest = canonical_benchmark_payload(path, instruction="Improve")
        self.assertEqual(len(digest), 64)
        self.assertIn(b'"instruction":"Improve"', payload)

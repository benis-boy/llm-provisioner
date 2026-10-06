"""Phase 5 provisioning-boundary tests.

These tests use an actual ResourceManager and server binding.  The extractor
only translates bounded RM events into native evidence; it never executes the
provider itself.
"""
import asyncio
import dataclasses
import time
import unittest
from unittest.mock import AsyncMock, patch

from services.llm.provisioning.benchmark_requests import BenchmarkRequest, _request_fingerprint
from services.llm.provisioning.capacity import MemorySample, Wave, _points
from services.llm.provisioning.measurement import (_ceiling, _resident_ceiling, _sweep_points,
                                                   _failure_code,
                                                   _resident_warmup_category,
                                                   bounded_exception_text,
                                                   measure_authoritative)
from services.llm.provisioning.rm_runner import ProvisioningError, ResourceManagerWaveRunner
from services.llm.provisioning.evidence import ClassifiedCapacityEvidenceError
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager
from services.llm.resource_manager.http import ModelBinding
from services.llm.resource_manager.protocol import EventKind, Failure, ProviderResponse
from services.llm.providers.coedit_batch import AllocatorObservation
from services.llm.providers.python_process import WorkerFailure


ALLOCATOR = AllocatorObservation(1, 2, 3, 4, 1, 2)


class ProviderFailure(RuntimeError):
    def __init__(self, failure):
        super().__init__(failure.message)
        self.failure = failure


def profile(parallelism=4, *, identity="provisioning-candidate"):
    return CapacityProfile(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter",
                           identity, parallelism, parallelism, parallelism, 20,
                           (SampleMetadata(1, 0, 1, 1, 1, (1,)),), context_size=128)


def request():
    identity = {"model": "smollm", "request_bucket": "smollm:context128",
                "witnesses": {"maximum": True}}
    payload = b'{"text":"maximum"}'
    return BenchmarkRequest(ModelId.SMOLLM, "smollm:context128", payload,
                            _request_fingerprint(identity, payload), identity, "configured")


class Provider:
    def __init__(self, *, delay=0.002, fail=False, failure_after=None):
        self.delay, self.fail, self.failure_after, self.payloads = delay, fail, failure_after, []
        self.calls = []
        self.executions = 0

    async def validate(self, profile): self.calls.append("validate")
    async def load(self, profile): self.calls.append("load")
    async def ready(self): self.calls.append("ready")
    async def validate_input(self, payload, *, context_size, bucket_identity): self.payloads.append(payload)
    async def execute(self, request_id, payload):
        self.calls.append(("execute", request_id))
        self.executions += 1
        await asyncio.sleep(self.delay)
        if self.failure_after is not None and self.executions > self.failure_after:
            raise ProviderFailure(Failure("provider_capacity", "GPU admission refused; retry later", False))
        if self.fail:
            raise RuntimeError("oom")
        return ProviderResponse(b"ok", 1, True)
    async def cancel(self, request_id): self.calls.append(("cancel", request_id))
    async def unload(self): self.calls.append("unload")
    async def verify_cleanup(self): return True


class Binding(ModelBinding):
    def __init__(self, provider, capacity=4):
        super().__init__(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", None, provider)
        self.capacity = capacity

    def resolve(self, *, context_size, bucket_identity):
        return profile(self.capacity), self.provider


class AlteredWatch(ResourceManager):
    async def watch_progress(self, token, after_sequence=0):
        async for event in super().watch_progress(token, after_sequence):
            if event.kind is EventKind.RESPONSE_FINISHED:
                yield dataclasses.replace(event, request_id="foreign", attempt="foreign")
            else:
                yield event


class MeasurementRunnerTests(unittest.IsolatedAsyncioTestCase):
    def runner(self, rm, provider, *, candidate_capacity=1, ceiling=4, **kwargs):
        return ResourceManagerWaveRunner(
            rm, Binding(provider, candidate_capacity), scheduler_id="s", model_id=ModelId.SMOLLM,
            context_size=128, provisioning_profile=profile(ceiling, identity="ephemeral-provisioning"),
            configured_ceiling=ceiling, **kwargs)

    async def test_actual_rm_submits_exact_payload_and_cleans(self):
        provider = Provider()
        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, provider, run_identity="fixed")
        seen = []
        async def invoke():
            return await runner(1, 1, ("fixed-r1",), request().payload,
                                lambda p, w, ids, events, elapsed: (seen.append((ids, events)) or
                                    Wave(p, w, ids, elapsed, True, p, 0, 10, True,
                                         allocator=ALLOCATOR, decoder_steps=(64,), max_output_tokens=64)))
        wave = await invoke()
        await runner.close()
        self.assertEqual(provider.payloads, [request().payload])
        self.assertEqual(wave.request_ids, ("fixed-r1",))
        self.assertEqual(rm.snapshot().phase, "startup")
        self.assertTrue(seen[0][1])

    async def test_foreign_completion_is_rejected_and_cleanup_is_fenced(self):
        rm = AlteredWatch(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, Provider())
        with self.assertRaises(ClassifiedCapacityEvidenceError):
            await runner(1, 1, ("r",), request().payload, lambda *args: None)
        await runner.close()
        self.assertEqual(rm.snapshot().phase, "startup")

    async def test_extractor_value_error_is_incomplete_and_cleanup_runs(self):
        class Sampler:
            async def sample(self):
                return MemorySample(1, 1000, 100, 900)

        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, Provider(), candidate_capacity=1, ceiling=1)

        def reject(*_args):
            raise ValueError("unsupported native batch\nshape\x00; " + "x" * 300)

        result = await measure_authoritative(
            request(), runner, Sampler(), configured_ceiling=1,
            identity_derived_max_parallelism=1,
            identity_derived_capability_reason="identity_configured_operator_ceiling",
            expected_max_output_tokens=64, evidence_extractor=reject)

        self.assertEqual((result.status, result.profile_eligible), ("incomplete", False))
        self.assertEqual(result.failure_code, "evidence_error")
        self.assertTrue(result.reason.startswith("evidence_error:"))
        self.assertLessEqual(len(result.reason), 160 + len("evidence_error:"))
        self.assertNotIn("\n", result.reason)
        self.assertNotIn("\x00", result.reason)
        self.assertEqual(rm.snapshot().phase, "startup")

    async def test_rm_failure_code_and_message_reach_bounded_invalid_discovery_reason(self):
        # This exceeds the sampler interval, producing a required
        # during-execution sample.  The former default delay completed before
        # collection and therefore failed the first baseline as runner_error.
        provider = Provider(delay=0.02, failure_after=4)
        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, provider, candidate_capacity=2, ceiling=2)
        class Clock:
            def __init__(self):
                self.value = time.monotonic_ns()

            def sample(self):
                # Preserve strict ordering even if consecutive test samples
                # occur within the same monotonic-clock tick.
                self.value = max(time.monotonic_ns(), self.value + 1)
                return self.value

        clock = Clock()
        class Sampler:
            calls = 0
            async def sample(self):
                self.calls += 1
                used = 100 if self.calls == 1 else 200
                return MemorySample(clock.sample(), 1000, used, 1000 - used)

        last_execution_end = -1
        def extract(p, w, ids, _events, elapsed):
            nonlocal last_execution_end
            # The action runs between the pre and post samples.  At extractor
            # invocation the collector has already recorded a during-action
            # sample, so this shared monotonic interval overlaps it and stays
            # ordered with every preceding wave.
            execution_started = last_execution_end + 1
            execution_ended = clock.value
            last_execution_end = execution_ended
            return Wave(
                p, w, ids, elapsed, True, p, execution_started, execution_ended,
                True, allocator=ALLOCATOR, decoder_steps=(64,) * p,
                max_output_tokens=64, observation_count=1,
                observed_native_batch_sizes=(p,), native_request_correlation=True)

        result = await measure_authoritative(
            request(), runner, Sampler(),
            configured_ceiling=2, expected_max_output_tokens=64,
            identity_derived_max_parallelism=2,
            identity_derived_capability_reason="identity_configured_operator_ceiling",
            evidence_extractor=extract)

        self.assertEqual(result.status, "incomplete")
        self.assertFalse(result.profile_eligible)
        self.assertEqual(result.failure_code, "runner_error")
        self.assertEqual(result.reason, "invalid discovery: runner_error")
        self.assertEqual(len(result.baseline), 4)
        self.assertEqual(len(result.failed_discovery), 1)
        self.assertEqual(result.failed_discovery[0].phase, "discovery")
        self.assertEqual(result.failed_discovery[0].failure_kind, "runner_error")
        self.assertIsNone(result.failed_discovery[0].failure_detail)
        self.assertLessEqual(len(result.reason), 160)
        self.assertNotIn(request().payload.decode(), result.reason)
        self.assertNotIn("measurement-run", result.reason)
        self.assertNotIn("response", result.reason)
        self.assertEqual(rm.snapshot().phase, "startup")

    async def test_typed_worker_failure_traverses_rm_and_measurement_closed_vocabulary(self):
        for code, expected_kind in (("output_contract_failed", "runner_error"), ("oom", "oom")):
            with self.subTest(code=code):
                class TypedFailureProvider(Provider):
                    async def execute(self, request_id, payload):
                        raise WorkerFailure(code)
                rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
                runner = self.runner(rm, TypedFailureProvider(), candidate_capacity=1, ceiling=1)
                with self.assertRaises(ProvisioningError) as raised:
                    await runner(1, 1, ("typed",), request().payload, lambda *args: None)
                self.assertEqual(raised.exception.failure_code, code)
                self.assertEqual(raised.exception.failure_kind, expected_kind)
                self.assertEqual(_failure_code(raised.exception.failure_code), code)
                await runner.close()
                self.assertEqual(rm.snapshot().phase, "startup")

    async def test_authoritative_schedule_stamps_extractor_successes(self):
        class Sampler:
            async def sample(self):
                return MemorySample(0, 1000, 700, 300)

        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        # A one-slot operator ceiling cannot establish SmolLM resident
        # capacity: the p=2 D2 witness is required.  Give this schedule-only
        # fixture the smallest formula-backed, fully discoverable bound.
        runner = self.runner(rm, Provider(), candidate_capacity=2, ceiling=2)
        counter = 0

        async def sample_during(_sampler, action, **_kwargs):
            return await action()

        async def extractor_runner(self, concurrency, wave, request_ids, payload, extractor):
            nonlocal counter
            counter += 1
            start = counter * 10
            return Wave(concurrency, wave, request_ids, 10, True, concurrency,
                        start, start + 1, True,
                        (MemorySample(start, 1000, 700, 300),
                         MemorySample(start + 1, 1000, 800, 200)),
                        allocator=ALLOCATOR, observation_count=1,
                        observed_native_batch_sizes=(concurrency,),
                        native_request_correlation=True,
                        decoder_steps=(64,) * concurrency, max_output_tokens=64)

        with patch("services.llm.provisioning.measurement.sample_during", sample_during):
            with patch.object(ResourceManagerWaveRunner, "__call__", extractor_runner):
                result = await measure_authoritative(
                    request(), runner, Sampler(), configured_ceiling=2,
                    identity_derived_max_parallelism=2,
                    identity_derived_capability_reason="identity_configured_operator_ceiling",
                    expected_max_output_tokens=64,
                    evidence_extractor=lambda *args: args[0],
                )

        self.assertTrue(result.profile_eligible, (result.status, result.reason, result.failure_code,
                                                  [(w.phase, w.concurrency, w.wave, w.failure_kind) for w in result.failed_discovery]))
        self.assertEqual({wave.phase for wave in result.baseline}, {"baseline"})
        self.assertEqual({wave.phase for wave in result.warmups}, {"warmup"})
        self.assertEqual({wave.phase for wave in result.measured}, {"measured"})

    async def test_rising_baseline_uses_final_summary_for_initial_nonresident_discovery(self):
        """A provisional online peak must not fence discovery below replayable proof."""
        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, Provider(), candidate_capacity=12, ceiling=12)
        coedit_request = dataclasses.replace(request(), model_id=ModelId.COEDIT)
        calls = []
        clock = 0

        async def sample_during(_sampler, action, **_kwargs):
            return await action()

        async def fake_call(self, p, wave_no, request_ids, _payload, _extractor):
            nonlocal clock
            calls.append((p, wave_no))
            clock += 10
            # The first sample rises across the baseline.  The first baseline
            # wave also has a transient peak, so online accounting is larger
            # than the final baseline-relative replay summary.
            if len(calls) == 1:
                used = (100, 1000)
            elif len(calls) <= 4:
                used = (100 * len(calls), 100 * len(calls))
            else:
                used = (1000, 1000)
            samples = tuple(MemorySample(clock + index, 10_000, value,
                                         10_000 - value, clock + index, clock + index)
                            for index, value in enumerate(used))
            return Wave(p, wave_no, request_ids, 10, True, p, clock, clock + 1, True,
                        samples, phase="baseline" if len(calls) <= 4 else "discovery",
                        allocator=ALLOCATOR, observation_count=1,
                        observed_native_batch_sizes=(p,), native_request_correlation=True,
                        decoder_steps=(64,) * p, max_output_tokens=64)

        with patch("services.llm.provisioning.measurement.sample_during", sample_during), \
             patch.object(ResourceManagerWaveRunner, "__call__", fake_call):
            result = await measure_authoritative(
                coedit_request, runner, object(), configured_ceiling=12,
                identity_derived_max_parallelism=12,
                identity_derived_capability_reason="identity_configured_provider_capability",
                provider_max_parallelism=12, expected_max_output_tokens=64,
                evidence_extractor=lambda *args: args[0])

        # Final baseline pre-used is 400 and its replayable peak is 600, so the
        # initial formula permits 12.  The old provisional peak of 900 fenced
        # the same run at 8 and could never establish the configured bound.
        self.assertEqual((result.status, result.n, result.derived_ceiling),
                         ("complete", 12, None))
        self.assertEqual(len(result.successful_discovery), 11 * 4)
        self.assertEqual(result.baseline_pre_used_bytes, 400)
        self.assertEqual(result.peak_incremental_request_bytes, 600)
        self.assertEqual(tuple((wave.concurrency, wave.wave) for wave in result.successful_discovery),
                         tuple((p, wave) for p in range(2, 13) for wave in range(1, 5)))

    async def test_coedit_capability_32_exhaustion_succeeds_without_a_memory_formula(self):
        """Aggregate native-batch memory is replay evidence, not a slot formula."""
        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, Provider(), candidate_capacity=32, ceiling=32)
        coedit_request = dataclasses.replace(request(), model_id=ModelId.COEDIT)
        clock = 0

        async def sample_during(_sampler, action, **_kwargs):
            return await action()

        async def fake_call(self, p, wave_no, request_ids, _payload, _extractor):
            nonlocal clock
            clock += 10
            sample = MemorySample(clock, 10_000, 1_000 + p, 9_000 - p, clock, clock)
            return Wave(p, wave_no, request_ids, 10, True, p, clock, clock, True,
                        (sample,), allocator=ALLOCATOR, observation_count=1,
                        observed_native_batch_sizes=(p,), native_request_correlation=True,
                        decoder_steps=(64,) * p, max_output_tokens=64)

        with patch("services.llm.provisioning.measurement.sample_during", sample_during), \
             patch.object(ResourceManagerWaveRunner, "__call__", fake_call):
            result = await measure_authoritative(
                coedit_request, runner, object(), configured_ceiling=32,
                identity_derived_max_parallelism=32,
                identity_derived_capability_reason="identity_configured_provider_capability",
                expected_max_output_tokens=64, evidence_extractor=lambda *args: args[0])

        self.assertEqual((result.status, result.n, result.derived_ceiling,
                          result.profile_eligible), ("complete", 32, None, True))
        self.assertEqual(len(result.successful_discovery), 31 * 4)

    async def test_coedit_lower_operator_ceiling_does_not_prove_provider_capacity(self):
        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, Provider(), candidate_capacity=2, ceiling=2)
        coedit_request = dataclasses.replace(request(), model_id=ModelId.COEDIT)
        clock = 0

        async def sample_during(_sampler, action, **_kwargs):
            return await action()

        async def fake_call(self, p, wave_no, request_ids, _payload, _extractor):
            nonlocal clock
            clock += 10
            sample = MemorySample(clock, 1_000, 100, 900, clock, clock)
            return Wave(p, wave_no, request_ids, 10, True, p, clock, clock, True, (sample,),
                        allocator=ALLOCATOR, observation_count=1,
                        observed_native_batch_sizes=(p,), native_request_correlation=True,
                        decoder_steps=(64,) * p, max_output_tokens=64)

        with patch("services.llm.provisioning.measurement.sample_during", sample_during), \
             patch.object(ResourceManagerWaveRunner, "__call__", fake_call):
            result = await measure_authoritative(
                coedit_request, runner, object(), configured_ceiling=2,
                identity_derived_max_parallelism=32,
                identity_derived_capability_reason="identity_configured_provider_capability",
                provider_max_parallelism=32, expected_max_output_tokens=64,
                evidence_extractor=lambda *args: args[0])

        self.assertEqual((result.status, result.reason, result.failure_code,
                          result.profile_eligible),
                         ("incomplete", "configured_ceiling_unproved",
                          "configured_ceiling_unproved", False))

    async def test_smollm_zero_increment_exhausts_slots_until_resource_failure(self):
        """Resident warmup may make the formula unavailable without proving N."""
        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, Provider(), candidate_capacity=1, ceiling=3)
        calls, clock = [], 0

        async def sample_during(_sampler, action, **_kwargs):
            return await action()

        async def fake_call(self, p, wave_no, request_ids, _payload, _extractor):
            nonlocal clock
            calls.append((p, wave_no))
            clock += 10
            sample = MemorySample(clock, 1000, 100 if p < 3 else 900,
                                  900 if p < 3 else 100, clock, clock)
            return Wave(p, wave_no, request_ids, 10, True, p, clock, clock, True,
                        (sample,), failed=p == 3, failure_kind="reserve_breached" if p == 3 else None,
                        phase=("baseline" if len(calls) <= 4 else
                               "discovery" if len(calls) <= 13 else
                               "warmup" if wave_no == 0 else "measured"),
                        allocator=ALLOCATOR, observation_count=1,
                        observed_native_batch_sizes=(p,), native_request_correlation=True,
                        decoder_steps=(64,) * p, max_output_tokens=64)

        with patch("services.llm.provisioning.measurement.sample_during", sample_during), \
             patch.object(ResourceManagerWaveRunner, "__call__", fake_call):
            result = await measure_authoritative(
                request(), runner, object(), configured_ceiling=3,
                identity_derived_max_parallelism=3,
                identity_derived_capability_reason="identity_configured_operator_ceiling",
                expected_max_output_tokens=64, evidence_extractor=lambda *args: args[0])

        self.assertTrue(result.profile_eligible, (result.status, result.reason, result.failure_code,
                                                  [(w.phase, w.concurrency, w.wave, w.failure_kind) for w in result.failed_discovery]))
        self.assertEqual((result.n, result.derived_ceiling, result.peak_incremental_request_bytes),
                         (2, None, 0))
        self.assertEqual((result.resource_bound_failure.concurrency,
                          result.resource_bound_failure.failure_kind), (3, "reserve_breached"))
        self.assertEqual(calls[:9], [(1, wave) for wave in range(1, 5)] +
                         [(2, wave) for wave in range(1, 5)] + [(3, 1)])
        self.assertEqual(calls[9:12], [(1, 0), (1, 1), (1, 2)])

    async def test_smollm_zero_increment_at_ceiling_is_unproved(self):
        """A lower operator cap does not prove the fixed provider capability."""
        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, Provider(), candidate_capacity=1, ceiling=2)
        clock = 0

        async def sample_during(_sampler, action, **_kwargs):
            return await action()

        async def fake_call(self, p, wave_no, request_ids, _payload, _extractor):
            nonlocal clock
            clock += 10
            sample = MemorySample(clock, 1000, 100, 900, clock, clock)
            return Wave(p, wave_no, request_ids, 10, True, p, clock, clock, True, (sample,),
                        phase="baseline" if p == 1 and wave_no <= 4 else "discovery",
                        allocator=ALLOCATOR, observation_count=1,
                        observed_native_batch_sizes=(p,), native_request_correlation=True,
                        decoder_steps=(64,) * p, max_output_tokens=64)

        with patch("services.llm.provisioning.measurement.sample_during", sample_during), \
             patch.object(ResourceManagerWaveRunner, "__call__", fake_call):
            result = await measure_authoritative(
                request(), runner, object(), configured_ceiling=2,
                # Model the CoEdIT binding: the provider capability is fixed at
                # 32 even though this run deliberately searches only through 2.
                identity_derived_max_parallelism=32,
                identity_derived_capability_reason="identity_configured_provider_capability",
                provider_max_parallelism=32,
                expected_max_output_tokens=64, evidence_extractor=lambda *args: args[0])
        self.assertEqual((result.status, result.reason, result.profile_eligible),
                          ("incomplete", "configured_ceiling_unproved", False))
        self.assertEqual(result.failure_code, "configured_ceiling_unproved")

    async def test_smollm_zero_increment_exhaustive_provider_capability_succeeds(self):
        """Only the fixed provider maximum, not an operator cap, proves zero-D2 N."""
        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, Provider(), candidate_capacity=1, ceiling=32)
        clock = 0

        async def sample_during(_sampler, action, **_kwargs):
            return await action()

        async def fake_call(self, p, wave_no, request_ids, _payload, _extractor):
            nonlocal clock
            clock += 10
            sample = MemorySample(clock, 1000, 100, 900, clock, clock)
            return Wave(p, wave_no, request_ids, 10, True, p, clock, clock, True, (sample,),
                        allocator=ALLOCATOR, observation_count=1,
                        observed_native_batch_sizes=(p,), native_request_correlation=True,
                        decoder_steps=(64,) * p, max_output_tokens=64)

        with patch("services.llm.provisioning.measurement.sample_during", sample_during), \
             patch.object(ResourceManagerWaveRunner, "__call__", fake_call):
            result = await measure_authoritative(
                request(), runner, object(), configured_ceiling=32,
                identity_derived_max_parallelism=32,
                identity_derived_capability_reason="identity_configured_provider_capability",
                expected_max_output_tokens=64, evidence_extractor=lambda *args: args[0])

        self.assertEqual((result.status, result.n, result.derived_ceiling,
                          result.peak_incremental_request_bytes),
                         ("complete", 32, None, 0))
        self.assertEqual(len(result.successful_discovery), 31 * 4)

    async def test_smollm_formula_is_capped_by_persisted_capability_despite_p1_noise(self):
        """Positive resident p=1 sweep noise never becomes D2 or raises N past capability."""
        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, Provider(), candidate_capacity=1, ceiling=32)
        clock = calls = 0

        async def sample_during(_sampler, action, **_kwargs):
            return await action()

        async def fake_call(self, p, wave_no, request_ids, _payload, _extractor):
            nonlocal clock, calls
            calls += 1
            clock += 10
            # Baselines establish B=1000.  The p=1 sweep has ordinary positive
            # resident noise, while the immutable p=2 witness remains D2=100.
            used = 1000 if calls <= 4 else (1050 if p == 1 else 1100)
            sample = MemorySample(clock, 10000, used, 10000 - used, clock, clock)
            return Wave(p, wave_no, request_ids, 10, True, p, clock, clock, True,
                        (sample,), allocator=ALLOCATOR, observation_count=1,
                        observed_native_batch_sizes=(p,), native_request_correlation=True,
                        decoder_steps=(64,) * p, max_output_tokens=64)

        with patch("services.llm.provisioning.measurement.sample_during", sample_during), \
             patch.object(ResourceManagerWaveRunner, "__call__", fake_call):
            result = await measure_authoritative(
                request(), runner, object(), configured_ceiling=32,
                identity_derived_max_parallelism=32,
                identity_derived_capability_reason="identity_configured_provider_capability",
                expected_max_output_tokens=64, evidence_extractor=lambda *args: args[0])

        # Raw D2 mathematics permits 71 slots, but the identity capability is
        # the exact persisted operational limit.
        self.assertEqual((result.status, result.n, result.derived_ceiling,
                          result.peak_incremental_request_bytes),
                         ("complete", 32, 32, 100))
        self.assertEqual(len(result.successful_discovery), 31 * 4)

    async def test_smollm_late_increment_freezes_bound_and_discovers_through_it(self):
        """A p=2 resident witness freezes N, then discovery remains exhaustive."""
        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, Provider(), candidate_capacity=1, ceiling=3)
        clock = 0
        calls = []

        async def sample_during(_sampler, action, **_kwargs):
            return await action()

        async def fake_call(self, p, wave_no, request_ids, _payload, _extractor):
            nonlocal clock
            calls.append((p, wave_no))
            clock += 10
            # Resident p=1 request overhead is non-authoritative noise; p=2
            # remains the frozen resident witness.
            used = (150 if len(calls) > 4 else 100) if p == 1 else 400
            sample = MemorySample(clock, 1000, used, 1000 - used, clock, clock)
            return Wave(p, wave_no, request_ids, 10, True, p, clock, clock, True,
                        (sample,), allocator=ALLOCATOR, observation_count=1,
                        observed_native_batch_sizes=(p,), native_request_correlation=True,
                        decoder_steps=(64,) * p, max_output_tokens=64)

        with patch("services.llm.provisioning.measurement.sample_during", sample_during), \
             patch.object(ResourceManagerWaveRunner, "__call__", fake_call):
            result = await measure_authoritative(
                request(), runner, object(), configured_ceiling=3,
                identity_derived_max_parallelism=3,
                identity_derived_capability_reason="identity_configured_operator_ceiling",
                expected_max_output_tokens=64, evidence_extractor=lambda *args: args[0])

        self.assertEqual((result.status, result.n, result.derived_ceiling,
                           result.peak_incremental_request_bytes),
                         ("complete", 3, 3, 300))
        self.assertEqual({wave.concurrency for wave in result.successful_discovery}, {2, 3})
        self.assertEqual(
            tuple((wave.concurrency, wave.wave) for wave in result.successful_discovery),
            tuple((p, wave) for p in (2, 3) for wave in range(1, 5)),
        )
        self.assertEqual({wave.concurrency for wave in result.measured}, {1, 2, 3})

    async def test_smollm_late_increment_below_observed_point_fails_closed(self):
        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, Provider(), candidate_capacity=1, ceiling=3)
        clock = 0

        async def sample_during(_sampler, action, **_kwargs):
            return await action()

        async def fake_call(self, p, wave_no, request_ids, _payload, _extractor):
            nonlocal clock
            clock += 10
            used = 100 if p == 1 else 800
            sample = MemorySample(clock, 1000, used, 1000 - used, clock, clock)
            return Wave(p, wave_no, request_ids, 10, True, p, clock, clock, True,
                        (sample,), allocator=ALLOCATOR, observation_count=1,
                        observed_native_batch_sizes=(p,), native_request_correlation=True,
                        decoder_steps=(64,) * p, max_output_tokens=64)

        with patch("services.llm.provisioning.measurement.sample_during", sample_during), \
             patch.object(ResourceManagerWaveRunner, "__call__", fake_call):
            result = await measure_authoritative(
                request(), runner, object(), configured_ceiling=3,
                identity_derived_max_parallelism=3,
                identity_derived_capability_reason="identity_configured_operator_ceiling",
                expected_max_output_tokens=64, evidence_extractor=lambda *args: args[0])

        self.assertEqual((result.status, result.reason, result.failure_code),
                         ("complete", "complete", None))

    async def test_smollm_invalid_positive_discovery_telemetry_is_not_formula_evidence(self):
        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, Provider(), candidate_capacity=1, ceiling=2)
        clock = 0

        async def sample_during(_sampler, action, **_kwargs):
            return await action()

        async def fake_call(self, p, wave_no, request_ids, _payload, _extractor):
            nonlocal clock
            clock += 10
            used = 100 if p == 1 else 900
            sample = MemorySample(clock, 1000, used, 1000 - used, clock, clock)
            if p == 2:
                sample = dataclasses.replace(sample, free_bytes=-1)
            return Wave(p, wave_no, request_ids, 10, True, p, clock, clock, True,
                        (sample,), allocator=ALLOCATOR, observation_count=1,
                        observed_native_batch_sizes=(p,), native_request_correlation=True,
                        decoder_steps=(64,) * p, max_output_tokens=64)

        with patch("services.llm.provisioning.measurement.sample_during", sample_during), \
             patch.object(ResourceManagerWaveRunner, "__call__", fake_call):
            result = await measure_authoritative(
                request(), runner, object(), configured_ceiling=2,
                identity_derived_max_parallelism=2,
                identity_derived_capability_reason="identity_configured_operator_ceiling",
                expected_max_output_tokens=64, evidence_extractor=lambda *args: args[0])

        self.assertEqual((result.status, result.failure_code),
                         ("incomplete", "invalid_telemetry"))
        self.assertIsNone(result.derived_ceiling)
        self.assertEqual(result.peak_incremental_request_bytes, None)

    async def test_runtime_residency_seams_are_finite_and_incomplete(self):
        class Sampler:
            async def sample(self):
                return MemorySample(1, 1000, 100, 900)

        async def no_sample(_sampler, action, **_kwargs):
            return await action()

        async def measure(expected_code, factory=None, fence=None, call_error=None):
            rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
            base = self.runner(rm, Provider(), candidate_capacity=1, ceiling=1)

            async def make(_p):
                if factory:
                    raise RuntimeError("foreign daemon secret")
                return base

            async def residency(_p):
                if fence:
                    raise RuntimeError("GPU proof secret")

            async def invoke(self, p, wave_no, request_ids, _payload, _extractor):
                if call_error:
                    raise RuntimeError("warmup secret")
                return Wave(p, wave_no, request_ids, 1, True, p, 1, 2, True,
                            (MemorySample(2, 1000, 100, 900),),
                            allocator=ALLOCATOR, observation_count=1,
                            observed_native_batch_sizes=(p,),
                            native_request_correlation=True,
                            decoder_steps=(64,) * p, max_output_tokens=64)

            with patch("services.llm.provisioning.measurement.sample_during", no_sample), \
                 patch.object(ResourceManagerWaveRunner, "__call__", invoke):
                result = await measure_authoritative(
                    request(), base, Sampler(), configured_ceiling=1,
                    identity_derived_max_parallelism=1,
                    identity_derived_capability_reason="identity_configured_operator_ceiling",
                    expected_max_output_tokens=64, evidence_extractor=lambda *args: args[0],
                    runner_for_concurrency=make if (factory or call_error or fence) else None,
                    residency_fence=residency if fence else None)
            expected_reason = ("resident_warmup_failed:runner_error"
                               if expected_code == "resident_warmup_failed" else expected_code)
            self.assertEqual((result.status, result.profile_eligible, result.reason,
                               result.failure_code),
                              ("incomplete", False, expected_reason, expected_code))
            await base.close()

        await measure("runtime_setup_failed", factory=True)
        await measure("resident_warmup_failed", call_error=True)
        await measure("residency_fence_failed", fence=True)

    async def test_baseline_warmup_measured_and_cleanup_failures_have_structured_codes(self):
        """Stage classifications survive private bounded reasons unchanged."""
        class Sampler:
            async def sample(self):
                return MemorySample(1, 1000, 100, 900)

        async def no_sample(_sampler, action, **_kwargs):
            return await action()

        for phase, expected in (("baseline", "invalid_completion"),
                                ("warmup", "invalid_completion"),
                                ("measured", "invalid_completion")):
            with self.subTest(phase=phase):
                rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
                runner = self.runner(rm, Provider(), candidate_capacity=1, ceiling=1)
                calls = 0

                async def invoke(self, p, wave_no, request_ids, _payload, _extractor):
                    nonlocal calls
                    calls += 1
                    start = calls * 10
                    current_phase = ("baseline" if calls <= 4 else
                                     "warmup" if calls == 5 else "measured")
                    valid = current_phase != phase
                    return Wave(p, wave_no, request_ids, 1, valid, p, start, start + 1, True,
                                (MemorySample(start, 1000, 100, 900, start, start),
                                 MemorySample(start + 1, 1000, 200, 800, start + 1, start + 1)),
                                allocator=ALLOCATOR,
                                observation_count=1, observed_native_batch_sizes=(p,),
                                native_request_correlation=True,
                                decoder_steps=(64,) * p, max_output_tokens=64)

                with patch("services.llm.provisioning.measurement.sample_during", no_sample), \
                     patch.object(ResourceManagerWaveRunner, "__call__", invoke):
                    result = await measure_authoritative(
                        request(), runner, Sampler(), configured_ceiling=1,
                        identity_derived_max_parallelism=1,
                        identity_derived_capability_reason="identity_configured_provider_capability",
                        expected_max_output_tokens=64, evidence_extractor=lambda *args: args[0])
                self.assertEqual(result.failure_code, expected)

        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        runner = self.runner(rm, Provider(), candidate_capacity=1, ceiling=1)
        runner.close = AsyncMock(side_effect=RuntimeError("private cleanup detail"))
        result = await measure_authoritative(
            request(), runner, Sampler(), configured_ceiling=1,
            identity_derived_max_parallelism=1,
            identity_derived_capability_reason="identity_configured_provider_capability",
            expected_max_output_tokens=64, evidence_extractor=lambda *args: args[0])
        self.assertEqual(result.failure_code, "cleanup_failed")

    def test_exception_diagnostic_is_bounded_and_control_safe(self):
        value = bounded_exception_text(ValueError("line\r\nbreak\x00\x1f" + "z" * 300))
        self.assertEqual(value[:10], "line break")
        self.assertLessEqual(len(value), 160)
        self.assertNotRegex(value, r"[\x00-\x1f\x7f]")

    async def test_backpressure_and_cleanup_cancel_are_not_silent(self):
        # A capacity-one RM admits at most two requests; a wave of three is
        # rejected by the authoritative admission boundary.
        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        # The candidate profile is deliberately too small, but ephemeral
        # provisioning admission reaches the configured ceiling.
        runner = self.runner(rm, Provider(), candidate_capacity=1, ceiling=3)
        with self.assertRaises(ProvisioningError):
            await runner(4, 1, ("a", "b", "c", "d"), request().payload, lambda *args: None)
        await runner.close()

    def test_ephemeral_profile_preflight_rejects_insufficient_capacity(self):
        rm = ResourceManager(cleanup_timeout=.2, stop_timeout=.2)
        with self.assertRaises(ValueError):
            ResourceManagerWaveRunner(
                rm, Binding(Provider()), scheduler_id="s", model_id=ModelId.SMOLLM,
                context_size=128, provisioning_profile=profile(3), configured_ceiling=4)

    def test_schedule_ceiling_and_conservative_memory_rules(self):
        self.assertEqual(_sweep_points(1), (1,))
        points = _sweep_points(100)
        self.assertEqual(points[-1], 100)
        self.assertLessEqual(len(points), 11)
        self.assertEqual(_ceiling(1000, 100, 100, 32), 7)
        self.assertEqual(_resident_ceiling(12_878_610_432, 4_127_797_248,
                                           3_162_267_648), 2)
        for values in ((1000, 900, 100), (1000, 100, 0), (0, 0, 1)):
            with self.assertRaises(Exception):
                _resident_ceiling(*values)
        self.assertEqual(_points(16)[0], 1)

    def test_measurement_failure_code_aliases_are_authoritative(self):
        for producer_code in ("missing_timing_or_native_evidence",
                              "missing_overlap_or_native_evidence"):
            with self.subTest(producer_code=producer_code):
                 self.assertEqual(_failure_code(producer_code), "missing_native_evidence")

    def test_resident_warmup_diagnostics_are_bounded_and_closed(self):
        ids = ("r1", "r2")
        valid = Wave(2, 0, ids, 1, True, 1, 1, 2, True,
                     native_request_correlation=True, observation_count=2,
                     observed_native_batch_sizes=(1, 1), evidence_kind="ollama_native")
        self.assertIsNone(_resident_warmup_category(valid, None, 2, ids))
        runner_error = RuntimeError("provider secret prompt/output/path")
        runner_error.failure_kind = "runner_error"

        cases = (
            (Wave(2, 0, ids, 1, False, 1, failed=True), None, "failed_output"),
            (valid, runner_error, "runner_error"),
            (dataclasses.replace(valid, request_ids=("foreign", "r2")), None, "identity_mismatch"),
            (dataclasses.replace(valid, native_batch_size=2), None, "ollama_native_batch_contract"),
            (dataclasses.replace(valid, native_request_correlation=False), None, "missing_correlation"),
            (dataclasses.replace(valid, observation_count=1), None, "observation_drops_or_count"),
            (dataclasses.replace(valid, evidence_kind="torch_native", native_batch_size=1,
                                 observed_native_batch_sizes=(2,), observation_count=1),
             None, "non_ollama_native_batch_contract"),
        )
        for wave, exc, expected in cases:
            with self.subTest(expected=expected):
                detail = _resident_warmup_category(wave, exc, 2, ids)
                self.assertEqual(detail, expected)
                self.assertNotIn("secret", detail)
                self.assertNotIn("prompt", detail)

    def test_two_percent_tie_and_evidence_failures_are_fail_closed(self):
        from services.llm.provisioning.capacity import choose_optimum
        self.assertEqual(choose_optimum({1: [Wave(1, 1, ("a",), 10, True, 1)] * 4,
                                         2: [Wave(2, 1, ("b", "c"), 20, True, 2)] * 4}), 1)
        invalid = Wave(2, 1, ("a", "b"), 1, True, 2, native_request_correlation=False)
        self.assertFalse(invalid.native_request_correlation)

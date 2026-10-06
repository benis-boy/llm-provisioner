"""Focused proof that provisioning composes the real ResourceManager path."""

import asyncio
import dataclasses
import math
import unittest

from services.llm.provisioning import (
    BenchmarkRequest,
    BenchmarkRequestError,
    prepare_benchmark_request,
)
from services.llm.provisioning.benchmark_requests import MAX_REQUEST_BYTES
from services.llm.provisioning.rm_runner import (ProvisioningError, provision_request,
                                                 safe_provider_failure_code)
from services.llm.provisioning.evidence import ClassifiedCapacityEvidenceError
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager
from services.llm.resource_manager.http import ModelBinding
from services.llm.resource_manager.protocol import EventKind, ProviderResponse


def _profile() -> CapacityProfile:
    return CapacityProfile(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", "profile",
                           1, 1, 1, 20, (SampleMetadata(1, 0, 1, 1, 1, (1,)),), context_size=128)


class Provider:
    def __init__(self, result=b"ok", *, block_execute=False):
        self.result = result
        self.block_execute = block_execute
        self.calls = []
        self.release = asyncio.Event()

    async def validate(self, profile): self.calls.append("validate")
    async def load(self, profile): self.calls.append("load")
    async def ready(self): self.calls.append("ready")
    async def validate_input(self, payload, *, context_size, bucket_identity): self.calls.append("validate_input")
    async def execute(self, request_id, payload):
        self.calls.append("execute")
        if self.block_execute:
            await self.release.wait()
        return ProviderResponse(self.result, 9, True)
    async def cancel(self, request_id): self.calls.append("cancel"); self.release.set()
    async def unload(self): self.calls.append("unload")
    async def verify_cleanup(self): self.calls.append("verify_cleanup"); return True


class Binding(ModelBinding):
    def __init__(self, provider, profile=None):
        super().__init__(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime", "adapter", None, provider)
        self._profile = profile or _profile()

    def resolve(self, *, context_size, bucket_identity):
        return self._profile, self.provider


class AlteredCompletionResourceManager(ResourceManager):
    def __init__(self, request_id, attempt):
        super().__init__(cleanup_timeout=.1, stop_timeout=.1)
        self.completion_request_id = request_id
        self.completion_attempt = attempt

    async def watch_progress(self, session_token, after_sequence=0):
        async for event in super().watch_progress(session_token, after_sequence):
            if event.kind is EventKind.RESPONSE_FINISHED:
                event = dataclasses.replace(event, request_id=self.completion_request_id,
                                             attempt=self.completion_attempt)
            yield event


class ProvisioningRunnerTests(unittest.IsolatedAsyncioTestCase):
    def test_closed_smollm_and_malformed_response_categories_are_transportable(self):
        for code in ("smollm_input_decode", "smollm_input_validation",
                     "smollm_observation_contract", "smollm_internal",
                     "malformed_provider_response"):
            with self.subTest(code=code):
                self.assertEqual(safe_provider_failure_code(code), code)

    async def test_timeout_must_be_finite_non_boolean_and_positive(self):
        for timeout in (True, False, 0, -1, math.nan, math.inf, -math.inf):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                await provision_request(ResourceManager(), Binding(Provider()), scheduler_id="scheduler",
                                        model_id=ModelId.SMOLLM, payload=b"input", request_id="r1",
                                        attempt="a1", timeout=timeout)

    async def test_response_finished_requires_exact_request_and_attempt_identity(self):
        for completion_identity in ((None, None), ("other", "other-attempt")):
            with self.subTest(completion_identity=completion_identity):
                provider = Provider()
                rm = AlteredCompletionResourceManager(*completion_identity)
                with self.assertRaises(ClassifiedCapacityEvidenceError):
                    await provision_request(rm, Binding(provider), scheduler_id="scheduler",
                                            model_id=ModelId.SMOLLM, payload=b"input", request_id="r1",
                                            attempt="a1", context_size=128)
                self.assertEqual(rm.snapshot().phase, "startup")
                self.assertIn("unload", provider.calls)

    def test_benchmark_api_is_exported(self):
        from services.llm.provisioning import benchmark_requests

        self.assertIs(BenchmarkRequest, benchmark_requests.BenchmarkRequest)
        self.assertIs(BenchmarkRequestError, benchmark_requests.BenchmarkRequestError)
        self.assertIs(prepare_benchmark_request, benchmark_requests.prepare_benchmark_request)

    async def test_success_uses_actual_rm_and_returns_evidence_then_cleans(self):
        provider = Provider()
        rm = ResourceManager(cleanup_timeout=.1, stop_timeout=.1)
        evidence = await provision_request(rm, Binding(provider), scheduler_id="scheduler",
                                           model_id=ModelId.SMOLLM, payload=b"input", request_id="r1",
                                           attempt="a1", context_size=128)
        self.assertIs(type(rm), ResourceManager)
        self.assertEqual(evidence.result, b"ok")
        self.assertEqual(evidence.time_on_gpu_ms, 9)
        self.assertFalse(evidence.measured_capacity_claim)
        self.assertIn("execute", provider.calls)
        self.assertEqual(rm.snapshot().phase, "startup")
        self.assertIn("unload", provider.calls)

    async def test_invalid_provider_output_is_failure_and_cleanup(self):
        provider = Provider(result=b"")
        rm = ResourceManager(cleanup_timeout=.1, stop_timeout=.1)
        with self.assertRaises(ClassifiedCapacityEvidenceError):
            await provision_request(rm, Binding(provider), scheduler_id="scheduler",
                                    model_id=ModelId.SMOLLM, payload=b"input", request_id="r1",
                                    attempt="a1", context_size=128)
        self.assertEqual(rm.snapshot().phase, "startup")
        self.assertIn("unload", provider.calls)

    async def test_cancellation_fences_and_cleans_active_request(self):
        provider = Provider(block_execute=True)
        rm = ResourceManager(cleanup_timeout=.1, stop_timeout=.1)
        task = asyncio.create_task(provision_request(rm, Binding(provider), scheduler_id="scheduler",
                                                     model_id=ModelId.SMOLLM, payload=b"input", request_id="r1",
                                                     attempt="a1", context_size=128))
        for _ in range(20):
            if "execute" in provider.calls:
                break
            await asyncio.sleep(0)
        self.assertIn("execute", provider.calls)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(rm.snapshot().phase, "startup")
        self.assertIn("cancel", provider.calls)
        self.assertIn("unload", provider.calls)

    async def test_identity_mismatch_is_rejected_without_provider_bypass(self):
        provider = Provider()
        rm = ResourceManager()
        with self.assertRaises(ValueError):
            await provision_request(rm, Binding(provider), scheduler_id="scheduler",
                                    model_id=ModelId.COEDIT, payload=b"input", request_id="r1", attempt="a1")
        self.assertEqual(provider.calls, [])

    async def test_oversized_payload_is_rejected_before_binding_resolution_or_admission(self):
        provider = Provider()
        binding = Binding(provider)
        rm = ResourceManager()
        with self.assertRaises(ValueError):
            await provision_request(rm, binding, scheduler_id="scheduler",
                                    model_id=ModelId.SMOLLM,
                                    payload=b"x" * (MAX_REQUEST_BYTES + 1),
                                    request_id="r1", attempt="a1", context_size=128)
        self.assertEqual(provider.calls, [])
        self.assertEqual(rm.snapshot().phase, "startup")

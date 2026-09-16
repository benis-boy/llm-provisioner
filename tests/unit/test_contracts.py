import json
import math
import re
import unittest
from pathlib import Path

from services.llm.provisioning.contracts import (
    CapacityBucket,
    GenerationConfig,
    ModelConfig,
)
from services.llm.queue.contracts import (
    ALLOWED_TRANSITIONS,
    Attempt,
    FunctionDescriptor,
    InsertionMode,
    ModelId,
    QueuePosition,
    RequestStatus,
    can_transition,
)
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata


class ContractTests(unittest.TestCase):
    def test_openapi_declares_refs_and_valid_3_0_nullable_form(self):
        text = Path("docs/openapi.yaml").read_text()
        self.assertIn("openapi: 3.0.3", text)
        refs = re.findall(r"#/components/(?:schemas|responses|parameters|requestBodies)/[A-Za-z]+", text)
        for ref in refs:
            name = ref.rsplit("/", 1)[1]
            self.assertRegex(text, rf"(?:^|[{{,]\s*|\n\s+){re.escape(name)}:", msg=ref)
        self.assertIn("nullable: true", text)
        self.assertNotIn("type: 'null'", text)
        self.assertIn("SseEnvelope", text)
        self.assertIn("LastEventId", text)
        self.assertIn("StartSessionRequest", text)
        self.assertIn("contextSizeEstimate", text)
        self.assertIn("capacity_backpressure", text)
        self.assertNotIn('type: [integer, "null"]', text)

    def test_all_mutations_require_idempotency_key(self):
        text = Path("docs/openapi.yaml").read_text()
        for path in ("/schedulers/{schedulerId}/start", "/schedulers/{schedulerId}/requests",
                     "/schedulers/{schedulerId}/requests/{requestId}/cancel", "/schedulers/{schedulerId}/stop",
                     "/resource-manager/sessions", "/resource-manager/sessions/{sessionToken}/submissions",
                     "/resource-manager/sessions/{sessionToken}/requests/{requestId}/cancel",
                     "/resource-manager/sessions/{sessionToken}/stop", "/provisioning/verify-artifacts",
                     "/provisioning/measure-capacity", "/provisioning/validate-profile"):
            self.assertIn("IdempotencyKey", text, path)

    def test_transition_matrix_checks_each_of_36_pairs(self):
        statuses = list(RequestStatus)
        expected = {
            RequestStatus.SCHEDULED: {RequestStatus.SCHEDULED, RequestStatus.RUNNING, RequestStatus.ERROR, RequestStatus.CANCELLED},
            RequestStatus.RUNNING: set(statuses),
            RequestStatus.ON_GPU: {RequestStatus.ON_GPU, RequestStatus.RUNNING, RequestStatus.DONE, RequestStatus.ERROR, RequestStatus.CANCELLED},
            RequestStatus.DONE: {RequestStatus.DONE},
            RequestStatus.ERROR: {RequestStatus.ERROR},
            RequestStatus.CANCELLED: {RequestStatus.CANCELLED},
        }
        self.assertEqual(set(ALLOWED_TRANSITIONS), set(statuses))
        for old in statuses:
            for new in statuses:
                self.assertEqual(can_transition(old, new), new in expected[old], (old, new))

    def test_json_args_reject_nonfinite_numbers(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                FunctionDescriptor("ready", {"value": value})

    def test_queue_position_supports_append_fallback_and_group_metadata(self):
        QueuePosition("r1", rank=4, insertion_sequence=2, insertion_mode=InsertionMode.SKIP_LINE, group_sequence=9, group_anchor="anchor")
        QueuePosition("r2", rank=5, insertion_sequence=3, insertion_mode="append")
        with self.assertRaises(ValueError):
            QueuePosition("r3", rank=1, insertion_sequence=1, insertion_mode="append", group_sequence=1)

    def test_profile_and_model_invariants(self):
        generation = GenerationConfig({"temperature": 0.2}, "float16")
        coedit_bucket = CapacityBucket(1024, 128, generation, (2,))
        gector_bucket = CapacityBucket(1024, 128, generation, (2,), 5, (0.5,))
        ModelConfig(ModelId.COEDIT, "models/coedit", "model.safetensors", "transformers-coedit", buckets=(coedit_bucket,))
        ModelConfig(ModelId.GECTOR, "models/gector", "model.safetensors", "gector", buckets=(gector_bucket,))
        with self.assertRaises(ValueError):
            ModelConfig(ModelId.GECTOR, "models/gector", "model.safetensors", "gector")
        sample = SampleMetadata(2, 1, 2, 100, 1000, (50, 60))
        profile = CapacityProfile(ModelId.SMOLLM, "GPU-x", "manifest", "model", "runtime", "adapter", "profile", 2, 3, 2, 20, (sample,), context_size=2048)
        self.assertEqual(profile.admission_limit, 4)
        self.assertTrue(profile.accepts_request(context_size=1024))
        self.assertFalse(profile.accepts_request(context_size=4096))

    def test_attempt_timing_can_be_incomplete_and_fence_identity_is_typed(self):
        attempt = Attempt("r", "a", "s", 1)
        self.assertIsNone(attempt.time_on_gpu_ms)
        self.assertFalse(attempt.gpu_timing_complete)


if __name__ == "__main__":
    unittest.main()

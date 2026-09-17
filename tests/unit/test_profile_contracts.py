import math
import unittest

from services.llm.provisioning.contracts import CapacityBucket, GenerationConfig, ModelConfig
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata


class ProfileContractTests(unittest.TestCase):
    def test_sample_metadata_is_strict_and_keeps_zero_candidate_values(self):
        SampleMetadata(0, 0, 0, 0, 0, ())
        for value in (True, 1.0):
            with self.assertRaises(ValueError):
                SampleMetadata(value, 0, 0, 0, 0, ())
        with self.assertRaises(ValueError):
            SampleMetadata(1, 0, 2, 0, 0, ())
        with self.assertRaises(ValueError):
            SampleMetadata(1, 0, 0, 0, 0, [1])

    def test_profile_shape_and_request_inputs_are_fail_closed(self):
        sample = SampleMetadata(1, 0, 0, 0, 0, ())
        profile = CapacityProfile(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime",
                                  "adapter", "profile", 1, 1, 1, 0, (sample,), context_size=512)
        self.assertTrue(profile.accepts_request(512))
        for context in (None, 0, -1, True, 1.5):
            self.assertFalse(profile.accepts_request(context))
        self.assertFalse(profile.accepts_request(1, "bucket"))
        with self.assertRaises(ValueError):
            CapacityProfile(ModelId.SMOLLM, "gpu", "manifest", "model", "runtime",
                             "adapter", "profile", 1, 1, 1, 0, (sample,), bucket_identity="b")
        with self.assertRaises(ValueError):
            CapacityProfile(ModelId.COEDIT, "gpu", "manifest", "model", "runtime",
                             "adapter", "profile", 1, 1, 1, 0, (sample,), context_size=512)

    def test_provisioning_values_are_immutable_and_numeric_checks_are_strict(self):
        parameters = {"temperature": 0.2}
        generation = GenerationConfig(parameters, "float16")
        parameters["temperature"] = math.inf
        self.assertEqual(generation.parameters["temperature"], 0.2)
        with self.assertRaises(TypeError):
            generation.parameters["top_p"] = 0.9
        with self.assertRaises(ValueError):
            CapacityBucket(True, 1, generation, (1,))
        with self.assertRaises(ValueError):
            CapacityBucket(1, 1, generation, (True,))
        with self.assertRaises(ValueError):
            CapacityBucket(1, 1, generation, (1,), thresholds=(math.nan,))

    def test_model_config_requires_typed_immutable_shapes(self):
        generation = GenerationConfig({"temperature": 0.2}, "float16")
        bucket = CapacityBucket(1, 1, generation, (1,))
        with self.assertRaises(ValueError):
            ModelConfig(ModelId.COEDIT, "models/coedit", "model.safetensors",
                        "transformers-coedit", context_size_estimates=[1], buckets=(bucket,))
        with self.assertRaises(ValueError):
            ModelConfig(ModelId.SMOLLM, "models/smol", "model.gguf", "ollama",
                        context_size_estimates=(True,))


if __name__ == "__main__":
    unittest.main()

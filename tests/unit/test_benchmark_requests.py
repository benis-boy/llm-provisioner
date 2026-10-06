import hashlib
import json
import unittest

from services.llm.provisioning.benchmark_requests import (
    BenchmarkRequest,
    BenchmarkRequestError,
    _request_fingerprint,
    prepare_benchmark_request,
    canonical_measurement_fixtures,
)
from services.llm.provisioning.contracts import CapacityBucket, GenerationConfig, ModelConfig
from services.llm.queue.contracts import ModelId


def configs():
    generation = GenerationConfig({"num_beams": 1, "do_sample": False}, "float16")
    return (
        ModelConfig(ModelId.SMOLLM, "/m", "x", "ollama", context_size_estimates=(512,)),
        ModelConfig(ModelId.COEDIT, "/m", "x", "transformers-coedit", buckets=(CapacityBucket(128, 64, generation, (1,)),)),
        ModelConfig(ModelId.GECTOR, "/m", "x", "gector", buckets=(CapacityBucket(128, 1, GenerationConfig({"keep_confidence": 0.0, "min_error_prob": 0.0}, "float32"), (1,), 1, (0.0,)),)),
    )


class BenchmarkRequestTests(unittest.TestCase):
    def _coedit_seed(self):
        return prepare_benchmark_request(
            configs()[1],
            request_bucket="coedit:p1:input128:output64:float16:beams1:nosample",
            identity_witnesses={"m": "x"},
            configured_request={"instruction": "fix", "texts": ["seed"]},
            validate_request=lambda value, *_: True,
        )

    def test_runtime_generated_coedit_response_is_strict_and_seed_stays_unchanged(self):
        from services.llm.provisioning.benchmark_requests import replace_coedit_with_runtime_generated

        seed = self._coedit_seed()
        generated_text = "generated"
        payload = json.dumps({"instruction": "fix", "texts": [generated_text]},
                             sort_keys=True, separators=(",", ":")).encode()
        valid = {"text": generated_text, "count": 128, "max": 128,
                 "fingerprint": hashlib.sha256(payload).hexdigest()}
        result = replace_coedit_with_runtime_generated(seed, valid)
        self.assertIsNot(result, seed)
        self.assertEqual(seed.payload, b'{"instruction":"fix","texts":["seed"]}')
        self.assertEqual(json.loads(result.payload)["texts"], [generated_text])
        self.assertEqual(result.identity["runtime_generation"]["seed_request_fingerprint"],
                         seed.fingerprint)
        self.assertEqual(result.identity["runtime_generation"]["generated_payload_sha256"],
                         hashlib.sha256(result.payload).hexdigest())

        malformed = (
            {"text": generated_text, "count": True, "max": 128,
             "fingerprint": valid["fingerprint"]},
            {"text": generated_text, "count": 128, "max": 127,
             "fingerprint": valid["fingerprint"]},
            {"text": generated_text, "count": 128, "max": 128,
             "fingerprint": "0" * 64},
            {"text": "x" * (256 * 1024 + 1), "count": 128, "max": 128,
             "fingerprint": "0" * 64},
            {"text": generated_text, "count": 128, "max": 128},
            {"text": generated_text, "count": 128, "max": 128,
             "fingerprint": "not-a-fingerprint"},
        )
        for candidate in malformed:
            with self.subTest(candidate=list(candidate)):
                with self.assertRaises(BenchmarkRequestError):
                    replace_coedit_with_runtime_generated(seed, candidate)
        self.assertEqual(seed.payload, b'{"instruction":"fix","texts":["seed"]}')

    def test_all_models_have_deterministic_identity(self):
        requests = ("x", {"instruction": "fix", "texts": ["text"]},
                    {"texts": ["text"], "keep_confidence": 0.0, "min_error_prob": 0.0, "n_iteration": 1, "batch_size": 1})
        buckets = ("smollm:context512", "coedit:p1:input128:output64:float16:beams1:nosample", "gector:p1:tokens128:keep0:min0:iterations1:batch1:float32")
        witnesses = {"manifest": "m", "model": "h", "runtime": "r", "adapter": "a", "dtype": "float32", "generation_parameters": {"temperature": 0}, "native_batch_shape": [1]}
        for config, request, bucket in zip(configs(), requests, buckets):
            first = prepare_benchmark_request(config, request_bucket=bucket, identity_witnesses=witnesses, configured_request=request, validate_request=lambda value, *_: True)
            second = prepare_benchmark_request(config, request_bucket=bucket, identity_witnesses=witnesses, configured_request=request, validate_request=lambda value, *_: True)
            self.assertEqual(first.fingerprint, second.fingerprint)

    def test_identity_is_deeply_immutable(self):
        witnesses = {"nested": {"values": [1]}, "dtype": "float32",
                     "generation_parameters": {"temperature": 0},
                     "native_batch_shape": [1]}
        request = prepare_benchmark_request(configs()[0], request_bucket="smollm:context512",
                                            identity_witnesses=witnesses, configured_request="x",
                                            validate_request=lambda value, *_: True)
        witnesses["nested"]["values"].append(2)
        with self.assertRaises(TypeError):
            request.identity["witnesses"]["nested"]["values"] += (3,)

    def test_direct_construction_freezes_nested_identity_before_attestation(self):
        identity = {"witnesses": {"nested": {"values": [1]}}}
        payload = b"{}"
        request = BenchmarkRequest(ModelId.SMOLLM, "smol", payload,
                                   _request_fingerprint(identity, payload), identity, "configured")
        identity["witnesses"]["nested"]["values"].append(2)
        with self.assertRaises(TypeError):
            request.identity["witnesses"]["nested"]["values"] += (2,)

    def test_validator_is_required_and_called_once(self):
        calls = []
        config = configs()[1]
        request = {"instruction": "fix", "texts": ["text"]}
        def validator(value, config, bucket, request_bucket):
            calls.append((value, config, bucket, request_bucket))
            return True
        prepare_benchmark_request(config, request_bucket="coedit:p1:input128:output64:float16:beams1:nosample",
                                  identity_witnesses={"m": "x"}, configured_request=request,
                                  validate_request=validator)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][3], "coedit:p1:input128:output64:float16:beams1:nosample")
        with self.assertRaises(BenchmarkRequestError):
            prepare_benchmark_request(config, request_bucket="coedit:p1:input128:output64:float16:beams1:nosample",
                                      identity_witnesses={"m": "x"}, configured_request=request)

    def test_generated_requires_positive_structural_proof_and_fallback_is_explicit(self):
        config = configs()[1]; request = {"instruction": "fix", "texts": ["text"]}
        made = prepare_benchmark_request(config, request_bucket="coedit:p1:input128:output64:float16:beams1:nosample", identity_witnesses={"m": "x"}, generate_request=lambda *_: request, validate_request=lambda value, *_: True)
        self.assertEqual(made.source, "generated")
        with self.assertRaises(BenchmarkRequestError):
            prepare_benchmark_request(config, request_bucket="coedit:p1:input128:output64:float16:beams1:nosample", identity_witnesses={"m": "x"})

    def test_schema_identity_and_bounds_fail_closed(self):
        config = configs()[2]; base = {"texts": ["text"], "keep_confidence": 0.0, "min_error_prob": 0.0, "n_iteration": 1, "batch_size": 1}
        args = dict(request_bucket="gector:p1:tokens128:keep0:min0:iterations1:batch1:float32", identity_witnesses={"m": "x"})
        for bad in ({**base, "extra": 1}, {**base, "n_iteration": 2}, {**base, "keep_confidence": True}, {**base, "keep_confidence": float("nan")}):
            with self.subTest(bad=bad), self.assertRaises(BenchmarkRequestError):
                prepare_benchmark_request(config, configured_request=bad, validate_request=lambda value, *_: True, **args)
        with self.assertRaises(BenchmarkRequestError):
            prepare_benchmark_request(config, configured_request=base, request_bucket="wrong", identity_witnesses={"m": "x"}, validate_request=lambda value, *_: True)

    def test_validator_rejects_structurally_valid_over_limit_candidates(self):
        config = configs()[1]
        with self.assertRaises(BenchmarkRequestError):
            prepare_benchmark_request(config, request_bucket="coedit:p1:input128:output64:float16:beams1:nosample",
                                      identity_witnesses={"m": "x"}, configured_request={"instruction": "fix", "texts": ["text"]},
                                       validate_request=lambda value, *_: False)

    def test_validator_observes_selected_smol_context(self):
        observed = []
        bucket = "smollm:context512"

        def validator(value, config, bucket, request_bucket):
            observed.append(request_bucket)
            return request_bucket == "smollm:context512"

        prepare_benchmark_request(
            configs()[0], request_bucket=bucket,
            identity_witnesses={"m": "x", "dtype": "float32",
                                "generation_parameters": {"temperature": 0},
                                "native_batch_shape": [1]},
            configured_request="x", validate_request=validator)
        self.assertEqual(observed, [bucket])

    def test_empty_native_batch_shape_is_a_benchmark_request_error(self):
        config = ModelConfig(
            ModelId.COEDIT, "/m", "x", "transformers-coedit",
            buckets=(CapacityBucket(128, 64, GenerationConfig({"num_beams": 1}, "float16"), ()),),
        )
        with self.assertRaises(BenchmarkRequestError):
            prepare_benchmark_request(config, request_bucket="anything",
                                      identity_witnesses={"m": "x"}, configured_request={"instruction": "fix", "texts": ["text"]},
                                      validate_request=lambda value, *_: True)

    def test_canonical_gector_bucket_accepts_structurally_valid_request(self):
        config = configs()[2]
        request = {"texts": ["text"], "keep_confidence": 0.0, "min_error_prob": 0.0,
                   "n_iteration": 1, "batch_size": 1}
        result = prepare_benchmark_request(
            config, request_bucket="gector:p1:tokens128:keep0:min0:iterations1:batch1:float32",
            identity_witnesses={"m": "x"}, configured_request=request,
            validate_request=lambda value, *_: True)
        self.assertEqual(result.source, "configured")

    def test_canonical_fixtures_pass_each_authoritative_preparation_contract(self):
        selectors = ("smollm:context512",
                     "coedit:p1:input128:output64:float16:beams1:nosample",
                     "gector:p1:tokens128:keep0:min0:iterations1:batch1:float32")
        witnesses = ({"dtype": "q8_0", "generation_parameters": {"num_predict": 64,
                     "temperature": 0}, "native_batch_shape": [1]},
                     {"adapter": "coedit"}, {"adapter": "gector"})
        for config, selector, model, witness in zip(configs(), selectors,
                                                     ("SmolLM", "CoEdIT", "GECToR"), witnesses):
            with self.subTest(model=model):
                result = prepare_benchmark_request(
                    config, request_bucket=selector,
                    identity_witnesses={**witness, "manifest": "m", "runtime": "r"},
                    configured_request=canonical_measurement_fixtures()[model],
                    validate_request=lambda value, config, bucket, request_bucket: True)
                self.assertEqual(result.source, "configured")

    def test_fixture_character_length_is_not_a_tokenizer_maximum_claim(self):
        fixtures = canonical_measurement_fixtures()
        self.assertEqual(len(fixtures["CoEdIT"]["texts"][0]), 128)
        self.assertEqual(len(fixtures["GECToR"]["texts"][0]), 128)
        # The preparation contract requires an adapter validator: character
        # length alone cannot substitute for its production tokenizer witness.
        with self.assertRaises(BenchmarkRequestError):
            prepare_benchmark_request(
                configs()[1],
                request_bucket="coedit:p1:input128:output64:float16:beams1:nosample",
                identity_witnesses={"m": "x"}, configured_request=fixtures["CoEdIT"],
                validate_request=lambda *_: False)


if __name__ == "__main__":
    unittest.main()

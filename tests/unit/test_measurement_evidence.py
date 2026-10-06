import unittest
from services.llm.provisioning.evidence import (
    ClassifiedCapacityEvidenceError, smollm_evidence_extractor,
)

class MeasurementEvidenceTests(unittest.TestCase):
    def test_smollm_short_generation_is_execution_under_ceiling(self):
        for count in (1, 32, 63, 64):
            observation = {"kind":"ollama_generate", "request_id":"r", "execution_started":1,
                "execution_ended":2, "configured_num_predict":64, "configured_num_ctx":512,
                "configured_temperature":0, "prompt_eval_count":1, "eval_count":count,
                "prompt_eval_duration":1, "eval_duration":1, "load_duration":0,
                "total_duration":1, "done_reason":"stop"}
            event = {"kind":"response_finished", "request_id":"r", "result":b"ok", "observation": observation}
            wave = smollm_evidence_extractor(1, 1, ("r",), (event,), 1)
            self.assertEqual(wave.decoder_steps, (count,))

    def test_smollm_invalid_output_or_configured_witness_fails_closed(self):
        base = {"kind":"ollama_generate", "request_id":"r", "execution_started":1,
                "execution_ended":2, "configured_num_predict":64, "configured_num_ctx":512,
                "configured_temperature":0, "prompt_eval_count":1, "eval_count":1,
                "prompt_eval_duration":1, "eval_duration":1, "load_duration":0,
                "total_duration":1, "done_reason":None}
        for field, value in (("eval_count", 0), ("eval_count", True), ("eval_count", 65),
                             ("eval_count", None), ("configured_num_predict", 63),
                             ("configured_num_ctx", 256)):
            event = {"kind":"response_finished", "request_id":"r", "result":b"ok",
                     "observation": {**base, field: value}}
            with self.assertRaises(ClassifiedCapacityEvidenceError) as raised:
                smollm_evidence_extractor(1, 1, ("r",), (event,), 1)
            self.assertEqual(raised.exception.failure_kind, "telemetry_contract_failed")

    def test_smollm_load_duration_requires_nonnegative_non_bool_integer(self):
        base = {"kind":"ollama_generate", "request_id":"r", "execution_started":1,
                "execution_ended":2, "configured_num_predict":64, "configured_num_ctx":512,
                "configured_temperature":0, "prompt_eval_count":1, "eval_count":1,
                "prompt_eval_duration":1, "eval_duration":1, "load_duration":0,
                "total_duration":1, "done_reason":None}
        for value in (-1, True, 1.0, "1", None):
            event = {"kind":"response_finished", "request_id":"r", "result":b"ok",
                     "observation": {**base, "load_duration": value}}
            with self.assertRaises(ClassifiedCapacityEvidenceError) as raised:
                smollm_evidence_extractor(1, 1, ("r",), (event,), 1)
            self.assertEqual(raised.exception.failure_kind, "telemetry_contract_failed")

    def test_smollm_correlation_and_overlap_are_distinct_and_bounded(self):
        observation = {"kind": "ollama_generate", "request_id": "r", "execution_started": 1,
                       "execution_ended": 2, "configured_num_predict": 64, "configured_num_ctx": 512,
                       "configured_temperature": 0, "prompt_eval_count": 1, "eval_count": 1,
                       "prompt_eval_duration": 1, "eval_duration": 1, "load_duration": 0,
                       "total_duration": 1, "done_reason": None}
        event = {"kind": "response_finished", "request_id": "wrong", "result": b"ok",
                 "observation": observation}
        with self.assertRaises(ClassifiedCapacityEvidenceError) as raised:
            smollm_evidence_extractor(1, 1, ("r",), (event,), 1)
        self.assertEqual(raised.exception.failure_kind, "telemetry_correlation_failed")

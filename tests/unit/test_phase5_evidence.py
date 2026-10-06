import unittest

from services.llm.provisioning.evidence import ClassifiedCapacityEvidenceError

from services.llm.bootstrap.measurement_matrix import measurement_matrix
from services.llm.provisioning.evidence import (
    coedit_evidence_extractor,
    gector_evidence_extractor,
    smollm_evidence_extractor,
)


def allocator():
    return {
        "baseline_allocated": 1, "baseline_reserved": 2,
        "peak_allocated": 3, "peak_reserved": 4,
        "final_allocated": 1, "final_reserved": 2,
    }


def native_event(request_id, *, output=64, steps=None):
    ids = (request_id,)
    observation = {
        "batch_size": 1, "execution_started": 1_000_000_000,
        "execution_ended": 1_010_000_000, "cuda_synchronized": True,
        "allocator": allocator(), "decoder_steps": (output,) if steps is None else steps,
        "max_output_tokens": output, "request_ids": ids,
    }
    return {"kind": "response_finished", "request_id": request_id,
            "result": b"ok", "observation": observation}


class Phase5EvidenceTests(unittest.TestCase):
    def test_production_matrix_is_canonical_and_enumerable(self):
        self.assertEqual(tuple(model.value for model, _ in measurement_matrix()),
                         ("SmolLM", "CoEdIT", "GECToR"))
        self.assertEqual(len(measurement_matrix()), 3)
        self.assertEqual(len({selector for _, selector in measurement_matrix()}), 3)

    def test_coedit_and_gector_extractors_require_exact_native_workload(self):
        event = native_event("r")
        coedit = coedit_evidence_extractor(1, 1, ("r",), (event,), 10)
        self.assertTrue(coedit.native_request_correlation)
        gector_event = native_event("r", output=1)
        gector = gector_evidence_extractor(1, 1, ("r",), (gector_event,), 10)
        self.assertEqual((gector.workload_kind, gector.workload_witness), ("iterations", (1,)))
        with self.assertRaises(ValueError):
            coedit_evidence_extractor(1, 1, ("r",),
                                      (native_event("r", steps=(63,)),), 10)
        with self.assertRaises(ValueError):
            gector_evidence_extractor(2, 1, ("r", "s"), (gector_event,), 10)

    def test_smollm_extractor_requires_correlated_native_telemetry(self):
        common = {
            "kind": "ollama_generate", "execution_started": 1_000_000_000,
            "execution_ended": 1_010_000_000, "prompt_eval_count": 4,
            "configured_num_predict": 64, "configured_num_ctx": 512,
            "configured_temperature": 0,
            "eval_count": 64, "prompt_eval_duration": 1,
            "eval_duration": 2, "load_duration": 3, "total_duration": 4,
            "done_reason": "stop",
        }
        event = {"kind": "response_finished", "request_id": "r",
                 "result": b"text", "observation": {**common, "request_id": "r"}}
        wave = smollm_evidence_extractor(1, 1, ("r",), (event,), 10)
        self.assertEqual(wave.evidence_kind, "ollama_native")
        broken = {**event, "observation": {**common, "request_id": "other"}}
        with self.assertRaises(ClassifiedCapacityEvidenceError):
            smollm_evidence_extractor(1, 1, ("r",), (broken,), 10)

    def test_smollm_p2_wave_matches_full_native_contract(self):
        common = {
            "kind": "ollama_generate", "execution_started": 1_000_000_000,
            "execution_ended": 1_010_000_000, "prompt_eval_count": 4,
            "configured_num_predict": 64, "configured_num_ctx": 512,
            "configured_temperature": 0, "eval_count": 64,
            "prompt_eval_duration": 1, "eval_duration": 2,
            "load_duration": 3, "total_duration": 4, "done_reason": "stop",
        }
        events = tuple(
            {"kind": "response_finished", "request_id": request_id,
             "result": b"text", "observation": {**common, "request_id": request_id,
             "execution_started": 1_000_000_000 + index * 5_000_000,
             "execution_ended": 1_010_000_000 + index * 5_000_000}}
            for index, request_id in enumerate(("r", "s"))
        )

        wave = smollm_evidence_extractor(2, 1, ("r", "s"), events, 10)

        self.assertEqual(wave.concurrency, 2)
        self.assertEqual(wave.wave, 1)
        self.assertEqual(wave.request_ids, ("r", "s"))
        self.assertEqual(wave.elapsed_ms, 10)
        self.assertTrue(wave.outputs_valid)
        self.assertEqual(wave.native_batch_size, 1)
        self.assertEqual(wave.observation_count, 2)
        self.assertEqual(wave.observed_native_batch_sizes, (1, 1))
        self.assertTrue(wave.native_request_correlation)
        self.assertEqual(wave.observation_drops, 0)
        self.assertEqual(wave.decoder_steps, (64, 64))
        self.assertEqual(wave.max_output_tokens, 64)
        self.assertEqual(wave.evidence_kind, "ollama_native")
        self.assertEqual(wave.workload_kind, "ollama_tokens")
        self.assertEqual(wave.workload_witness, (64, 64))

    def test_smollm_rejects_swapped_observations_even_when_id_sets_match(self):
        common = {
            "kind": "ollama_generate", "execution_started": 1_000_000_000,
            "execution_ended": 1_010_000_000, "prompt_eval_count": 4,
            "configured_num_predict": 64, "configured_num_ctx": 512,
            "configured_temperature": 0, "eval_count": 64,
            "prompt_eval_duration": 1, "eval_duration": 2,
            "load_duration": 3, "total_duration": 4, "done_reason": "stop",
        }
        event_r = {"kind": "response_finished", "request_id": "r", "result": b"r",
                   "observation": {**common, "request_id": "s"}}
        event_s = {"kind": "response_finished", "request_id": "s", "result": b"s",
                   "observation": {**common, "request_id": "r"}}
        with self.assertRaises(ClassifiedCapacityEvidenceError):
            smollm_evidence_extractor(2, 1, ("r", "s"), (event_r, event_s), 10)


if __name__ == "__main__":
    unittest.main()

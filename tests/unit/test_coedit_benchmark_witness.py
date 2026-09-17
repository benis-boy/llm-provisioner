import asyncio
import hashlib
import io
import json
import struct
import unittest
from unittest.mock import AsyncMock, patch

from services.llm.providers import python_worker
from tools.compatibility import coedit_capacity_check as check


class Tokenizer:
    def __init__(self, counts):
        self.counts = counts

    def __call__(self, value, **kwargs):
        self.last = (value, kwargs)
        return {"input_ids": list(range(self.counts(value)))}


class BenchmarkWitnessTests(unittest.TestCase):
    def runtime(self, tokenizer, maximum=128):
        runtime = python_worker.Runtime.__new__(python_worker.Runtime)
        runtime.tokenizer = tokenizer
        runtime.config = {"max_input_tokens": maximum}
        return runtime

    def test_exact_count_uses_framing_and_special_tokens(self):
        tokenizer = Tokenizer(lambda value: 128 if value == "fix\ntext" else 1)
        result = self.runtime(tokenizer).benchmark_input("fix", "text", False)
        payload = b'{"instruction":"fix","texts":["text"]}'
        self.assertEqual(result, {"count": 128, "max": 128,
                                  "fingerprint": hashlib.sha256(payload).hexdigest()})
        self.assertEqual(tokenizer.last[0], "fix\ntext")
        self.assertEqual(tokenizer.last[1], {"add_special_tokens": True, "truncation": False})

    def test_over_bound_is_rejected(self):
        with self.assertRaises(ValueError):
            self.runtime(Tokenizer(lambda _: 129)).benchmark_input("fix", "text", False)

    def test_generation_is_bounded_and_fails_without_exact_count(self):
        with self.assertRaises(python_worker.InsufficientMaxInput):
            self.runtime(Tokenizer(lambda _: 1)).benchmark_input("fix", "text", True)

    def test_gector_request_set_rejects_benchmark_operation(self):
        body = json.dumps({"id": 1, "op": "benchmark_input", "instruction": "x",
                           "text": "y", "generate": False}, separators=(",", ":")).encode()
        with self.assertRaises(RuntimeError):
            python_worker._read(io.BytesIO(struct.pack(">I", len(body)) + body), True)

    def test_harness_rejects_switching_witness_identity(self):
        provider = type("Provider", (), {"worker": type("Worker", (), {
            "call": AsyncMock(side_effect=[{"count": 128, "max": 128, "fingerprint": "a"},
                                             {"count": 127, "max": 128, "fingerprint": "b"}])})()})()
        async def exercise():
            first = await check._witness(provider, "fix", "text")
            self.assertEqual(first["fingerprint"], "a")
            with self.assertRaises(check.CapacityEvidenceError):
                await check._RealWaveRunner(None, None, provider,
                    b'{"instruction":"fix","texts":["text"]}', "bucket", first)(1, 1, ("id",))
        asyncio.run(exercise())

    def test_harness_marks_worker_output_maximum_different_from_provider_configuration(self):
        provider = type("Provider", (), {"config": type("C", (), {"max_output_tokens": 64})(),
            "worker": type("Worker", (), {"call": AsyncMock(return_value={
                "count": 128, "max": 128, "fingerprint": "same"})})(),
            "drain_batch_observations": lambda self: ({"batch_size": 1,
                "request_ids": ("id",), "cuda_synchronized": True,
                "execution_started": 1, "execution_ended": 2,
                "decoder_steps": (63,), "max_output_tokens": 63,
                "allocator": check.AllocatorObservation(1, 2, 3, 4, 1, 2)},),
            "batch_observation_drops": lambda self: 0})()
        rm = type("RM", (), {"submit": AsyncMock()})()
        runner = check._RealWaveRunner(rm, type("S", (), {"session_token": "session"})(), provider,
            b'{"instruction":"fix","texts":["text"]}', "bucket",
            {"count": 128, "max": 128, "fingerprint": "same"})
        async def exercise():
            with patch.object(check, "_terminal", AsyncMock(return_value=type("E", (), {"result": b'{"texts":["ok"]}', "sequence": 1})())):
                return await runner(1, 1, ("id",))
        wave = asyncio.run(exercise())
        self.assertEqual((wave.failed, wave.failure_kind), (True, "native_batch_correlation"))


if __name__ == "__main__":
    unittest.main()

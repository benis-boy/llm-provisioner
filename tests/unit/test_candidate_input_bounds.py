import struct
import tempfile
import unittest
from pathlib import Path

from tools.compatibility.input_bounds import (MAX_METADATA_BYTES, SMOLLM_CONTEXT,
                                              SMOLLM_MAX_RAW_BYTES, SMOLLM_OUTPUT_RESERVE,
                                              frame_smollm_prompt, smollm_evidence,
                                              validate_smollm_input, _exact)
from tools.compatibility.model_runtime import MAX_FIXTURES, SMALL_FIXTURES, LoadedRuntime


def _gguf(root: Path, model="gpt2", *, pre="smollm", duplicate=False):
    values = [("tokenizer.ggml.model", 8, model.encode()), ("tokenizer.ggml.pre", 8, b"smollm"),
              ("tokenizer.ggml.tokens", 9, (8, ["Ġ", "Ċ"] + [chr(value) for value in range(0x21, 0x7f)])),
              ("tokenizer.ggml.add_bos_token", 7, False), ("llama.context_length", 4, 512)]
    values[1] = ("tokenizer.ggml.pre", 8, pre.encode())
    if duplicate:
        values.append(("tokenizer.ggml.pre", 8, pre.encode()))
    data = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(values)))
    for key, kind, value in values:
        encoded = key.encode(); data += struct.pack("<Q", len(encoded)) + encoded + struct.pack("<I", kind)
        if kind == 8: data += struct.pack("<Q", len(value)) + value
        elif kind == 9:
            data += struct.pack("<IQ", value[0], len(value[1]))
            for token in value[1]:
                encoded = token.encode(); data += struct.pack("<Q", len(encoded)) + encoded
        elif kind == 7: data += b"\1" if value else b"\0"
        elif kind == 4: data += struct.pack("<I", value)
        else: data += struct.pack("<Q", value)
    (root / "model.gguf").write_bytes(data)


class CandidateInputBoundTests(unittest.TestCase):
    def test_metadata_exact_read_budget_rejects_before_read(self):
        class Probe:
            def __init__(self):
                self.reads = 0

            def tell(self):
                return MAX_METADATA_BYTES

            def read(self, _size):
                self.reads += 1
                return b""

        stream = Probe()
        with self.assertRaisesRegex(ValueError, "aggregate byte budget"):
            _exact(stream, 1)
        self.assertEqual(stream.reads, 0)

    def test_raw_framing_bound_leaves_output_reserve(self):
        framed = frame_smollm_prompt("x" * SMOLLM_MAX_RAW_BYTES).encode("utf-8")
        self.assertLessEqual(len(framed) + SMOLLM_OUTPUT_RESERVE, SMOLLM_CONTEXT)
        self.assertLessEqual(len(framed), SMOLLM_CONTEXT - SMOLLM_OUTPUT_RESERVE)

    def test_metadata_proves_byte_bound_without_loading_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _gguf(root)
            evidence = smollm_evidence(root)
            self.assertEqual(evidence["algorithm"], "streamed-byte-BPE-upper-bound+explicit-raw-framing")
            # The conservative ASCII bucket is printable ASCII (94 bytes),
            # plus the two explicit whitespace markers used by this fixture.
            self.assertEqual(evidence["metadata"]["vocabulary_size"], 96)
            self.assertEqual(evidence["output_token_reserve"], 64)
            self.assertLessEqual(len(SMALL_FIXTURES["SmolLM"].encode()), SMOLLM_MAX_RAW_BYTES)

    def test_unknown_tokenizer_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _gguf(root, model="unknown")
            with self.assertRaisesRegex(ValueError, "does not prove"):
                smollm_evidence(root)

    def test_oversize_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _gguf(root)
            with self.assertRaisesRegex(ValueError, "exceeds configured raw-byte bucket"):
                validate_smollm_input(root, "x" * (SMOLLM_MAX_RAW_BYTES + 1))

    def test_non_ascii_and_duplicate_proof_key_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _gguf(root)
            with self.assertRaisesRegex(ValueError, "printable-ASCII"):
                validate_smollm_input(root, "é")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _gguf(root, duplicate=True)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                smollm_evidence(root)

    def test_all_configured_fixtures_are_nonempty_and_admissible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _gguf(root)
            runtime = LoadedRuntime("SmolLM", root)
            for fixture in (SMALL_FIXTURES["SmolLM"], MAX_FIXTURES["SmolLM"]):
                runtime.validate(fixture, 512, None); self.assertTrue(fixture.strip())


if __name__ == "__main__":
    unittest.main()

"""Pure model-runtime identity and redaction tests; no model or CUDA required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

from tools.compatibility.model_runtime import (LoadedRuntime, SMOLLM_OUTPUT_RESERVE,
                                               SMOLLM_PROMPT_TOKEN_LIMIT,
                                               smollm_model_name, smollm_source_identity)
from tools.compatibility.input_bounds import frame_smollm_prompt
from tools.compatibility.rm_spike import _physical_uuid


class ModelRuntimeTests(unittest.TestCase):
    def test_nvml_bytes_uuid_is_decoded(self):
        self.assertEqual(_physical_uuid(b"GPU-12345678-1234-1234-1234-123456789abc"),
                         "12345678-1234-1234-1234-123456789abc")
    def test_smollm_identity_is_deterministic_and_source_based(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Modelfile").write_text("FROM model.gguf\n", encoding="utf-8")
            (root / "model.gguf").write_bytes(b"weights")
            first = smollm_source_identity(root), smollm_model_name(root)
            second = smollm_source_identity(root), smollm_model_name(root)
            self.assertEqual(first, second)
            (root / "model.gguf").write_bytes(b"changed")
            self.assertNotEqual(first[0], smollm_source_identity(root))

    def test_store_is_fresh_and_snapshot_redacts_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Modelfile").write_text("FROM model.gguf\n", encoding="utf-8")
            (root / "model.gguf").write_bytes(b"weights")
            runtime = LoadedRuntime("SmolLM", root)
            with patch("tools.compatibility.model_runtime.tempfile.TemporaryDirectory", wraps=tempfile.TemporaryDirectory):
                first, name = runtime._allocate_smollm_store()
                runtime.store, runtime.model_name = first, name
                snapshot = runtime.snapshot()
                self.assertTrue(first.is_dir())
                self.assertNotIn("prompt", snapshot)
                self.assertNotIn("result", snapshot)
                runtime.close()
                # Store deletion is parent-owned after the child process group
                # has terminated; this direct runtime test must model that
                # ownership before asserting filesystem cleanup.
                runtime._store_temp.cleanup()
                runtime._store_temp = None
                self.assertFalse(first.exists())
                second, _ = runtime._allocate_smollm_store()
                self.assertNotEqual(first, second)
                runtime.store = second
                runtime.close()
                runtime._store_temp.cleanup()
                runtime._store_temp = None

    def test_smollm_residency_check_uses_its_private_ollama_endpoint(self):
        runtime = LoadedRuntime("SmolLM", Path("."))
        runtime.model_name = "private-model"
        runtime._ollama_env = {"OLLAMA_HOST": "127.0.0.1:43123"}
        nvml = SimpleNamespace(nvmlInit=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda _: object(),
            nvmlDeviceGetUUID=lambda _: b"GPU-12345678-1234-1234-1234-123456789abc")
        with patch.dict("sys.modules", {"pynvml": nvml}), \
             patch("tools.compatibility.model_runtime.subprocess.run",
                   return_value=SimpleNamespace(stdout="private-model GPU", stderr="")) as run:
            evidence = runtime.gpu_identity()
        self.assertTrue(evidence["ollama_gpu_resident"])
        self.assertEqual(run.call_args.kwargs["env"], runtime._ollama_env)

    def test_smollm_cleanup_uses_its_private_ollama_endpoint(self):
        runtime = LoadedRuntime("SmolLM", Path("."))
        runtime.model_name = "private-model"
        runtime._ollama_env = {"OLLAMA_HOST": "127.0.0.1:43123"}
        server = SimpleNamespace(poll=lambda: 0)
        runtime.server = server
        with patch("tools.compatibility.model_runtime.subprocess.run") as run:
            runtime.close()
        self.assertEqual(run.call_args.kwargs["env"], {"OLLAMA_HOST": "127.0.0.1:43123"})

    def test_smollm_execution_uses_exact_raw_frame_and_rejects_incomplete_or_over_budget_reply(self):
        runtime = LoadedRuntime("SmolLM", Path(".")); runtime.model_name = "private-model"; runtime._ollama_url = "http://private"
        response = Mock()
        response.read.return_value = json.dumps({"done": True, "prompt_eval_count": SMOLLM_PROMPT_TOKEN_LIMIT, "response": "ok"}).encode()
        response.__enter__ = Mock(return_value=response); response.__exit__ = Mock(return_value=False)
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            self.assertEqual(runtime._execute("x", object()), "ok")
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertTrue(body["raw"]); self.assertEqual(body["options"]["num_predict"], SMOLLM_OUTPUT_RESERVE)
        self.assertEqual(body["prompt"], frame_smollm_prompt("x"))
        for invalid in ({"done": False, "prompt_eval_count": 1, "response": "ok"}, {"done": True, "prompt_eval_count": SMOLLM_PROMPT_TOKEN_LIMIT + 1, "response": "ok"}):
            response.read.return_value = json.dumps(invalid).encode()
            with patch("urllib.request.urlopen", return_value=response), self.assertRaises(RuntimeError):
                runtime._execute("x", object())


if __name__ == "__main__":
    unittest.main()

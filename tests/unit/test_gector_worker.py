import contextlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from services.llm.providers import gector_worker


class FakeIds:
    def __init__(self, width=128): self.shape = (1, width); self.width = width
    def to(self, device): return self
    def word_ids(self, batch_index=0): return [0] + [None] * (self.width - 1)
    def get(self, key): return self if key == "input_ids" else None


class FakeTokenizer:
    def __init__(self, width=128): self.calls = []; self.width = width
    def __call__(self, value, **kwargs): self.calls.append((value, kwargs)); return FakeIds(self.width)


class GECToRWorkerTests(unittest.TestCase):
    def runtime(self):
        runtime = object.__new__(gector_worker.Runtime)
        runtime.config = {"keep_confidence": 0.0, "min_error_prob": 0.0, "max_iterations": 1, "max_subword_tokens": 128}
        runtime.tokenizer = FakeTokenizer()
        runtime.model = types.SimpleNamespace(config=types.SimpleNamespace(is_official_model=False, max_length=128))
        return runtime

    def test_actual_entrypoint_delegates_shared_main(self):
        self.assertIs(gector_worker.python_worker.Runtime, __import__("services.llm.providers.python_worker", fromlist=["Runtime"]).Runtime)
        self.assertTrue(callable(gector_worker.python_worker.main))

    def test_source_preprocessing_matches_package_and_disables_truncation(self):
        runtime = self.runtime()
        runtime._check(["one two"], 0.0, 0.0, 1, 1)
        value, options = runtime.tokenizer.calls[-1]
        self.assertEqual(value, [["$START", "one", "two"]])
        self.assertEqual(options, {"return_tensors":"pt", "max_length":128, "padding":"max_length", "truncation":False, "is_split_into_words":True, "add_special_tokens":True})

    def test_official_model_disables_extra_special_tokens(self):
        runtime = self.runtime(); runtime.model.config.is_official_model = True
        runtime._check(["one"], 0.0, 0.0, 1, 1)
        self.assertFalse(runtime.tokenizer.calls[-1][1]["add_special_tokens"])

    def test_malformed_ids_and_boolean_parameter_types_rejected(self):
        runtime = self.runtime(); runtime.tokenizer = lambda *a, **k: {"input_ids": [1]}
        with self.assertRaises(RuntimeError): runtime._check(["one"], 0.0, 0.0, 1, 1)
        runtime = self.runtime()
        with self.assertRaises(ValueError): runtime._check(["one"], True, 0.0, 1, 1)
        with self.assertRaises(ValueError): runtime._check(["one"], 0.0, 0.0, True, 1)

    def test_preflight_distinguishes_128_token_boundary_from_runtime_failure(self):
        for count, accepted in ((127, True), (128, True), (129, False)):
            runtime = self.runtime(); runtime.tokenizer = FakeTokenizer(128 if accepted else 129)
            self.assertEqual(runtime._check([" ".join(["x"] * count)], 0, 0.0, 1, 1), accepted)
        runtime = self.runtime(); runtime.tokenizer = FakeTokenizer(127)
        with self.assertRaises(RuntimeError): runtime._check(["x"], 0, 0, 1, 1)

    def test_output_is_preflighted_with_same_bound(self):
        runtime = self.runtime()
        runtime.model.config.is_official_model = False
        runtime.encode, runtime.decode = object(), object()
        runtime.tokenizer.calls.clear()
        fake_gector = types.SimpleNamespace(predict=lambda *a, **k: ["corrected"])
        torch = types.SimpleNamespace(
            inference_mode=lambda: contextlib.nullcontext(),
            cuda=types.SimpleNamespace(
                synchronize=lambda: None,
                memory_allocated=lambda _: 10,
                memory_reserved=lambda _: 20,
                reset_peak_memory_stats=lambda _: None,
                max_memory_allocated=lambda _: 30,
                max_memory_reserved=lambda _: 40,
            ),
        )
        with patch.dict(__import__("sys").modules, {"torch": torch, "gector": fake_gector}):
            result = runtime.execute(texts=["one"], keep_confidence=0.0, min_error_prob=0.0, n_iteration=1, batch_size=1)
            self.assertEqual(result["outputs"], ["corrected"])
            self.assertEqual(result["observation"]["batch_size"], 1)
        self.assertGreaterEqual(len(runtime.tokenizer.calls), 2)

    def test_load_sets_bucket_before_model_load_and_restores_patches_on_success_and_failure(self):
        calls = {}; tokenizer = FakeTokenizer()
        class Model:
            config = types.SimpleNamespace(max_length=128)
            # This mirrors GECToR 1.2.0, whose local encoder is ``bert``.
            bert = types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=512))
            def to(self, device, dtype): calls["device"]=(device,dtype); return self
            def eval(self): calls["eval"]=True
        def run(fail=False):
            runtime = object.__new__(gector_worker.Runtime); runtime.root=Path("/offline"); runtime.config={"max_subword_tokens":128}
            modeling = types.ModuleType("gector.modeling")
            original_model, original_tokenizer = object(), object()
            modeling.AutoModel=types.SimpleNamespace(from_pretrained=original_model)
            modeling.AutoTokenizer=types.SimpleNamespace(from_pretrained=original_tokenizer)
            gector = types.ModuleType("gector"); gector.__path__=[]; gector.GECToRConfig=types.SimpleNamespace(from_pretrained=lambda *a,**k: types.SimpleNamespace(model_id="artifact", max_length=20))
            def loaded(root, config, **kw):
                calls.update(load_options=kw, configured_length=config.max_length, model_id=config.model_id)
                if fail: raise RuntimeError("boom")
                return Model(), {"missing_keys":[],"unexpected_keys":[],"mismatched_keys":[],"error_msgs":[]}
            gector.GECToR=types.SimpleNamespace(from_pretrained=loaded); gector.load_verb_dict=lambda path:("encode","decode")
            torch=types.SimpleNamespace(device=lambda value:value,float32="float32")
            transformers=types.SimpleNamespace(AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a,**k:"base-config"), AutoModel=types.SimpleNamespace(from_config=lambda value:"encoder"), AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:tokenizer))
            with patch.dict(sys.modules, {"torch":torch,"transformers":transformers,"gector":gector,"gector.modeling":modeling}):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "boom"): runtime.load()
                else: runtime.load()
            self.assertIs(modeling.AutoModel.from_pretrained, original_model); self.assertIs(modeling.AutoTokenizer.from_pretrained, original_tokenizer)
        run(); self.assertEqual(calls["configured_length"], 128); self.assertEqual(calls["load_options"], {"local_files_only":True,"use_safetensors":True,"output_loading_info":True}); self.assertEqual(calls["device"], ("cuda:0","float32")); self.assertTrue(calls["eval"])
        run(fail=True)

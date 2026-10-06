import io
import json
import os
import struct
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from services.llm.providers.python_worker import (MAX_FRAME, InputValidationError, Runtime,
                                                  WorkerContractFailure, _execute_batch_failure_code,
                                                  _read)


class PythonWorkerTests(unittest.TestCase):
    def frame(self, value):
        raw=json.dumps(value).encode(); return io.BytesIO(struct.pack(">I",len(raw))+raw)
    def test_eof_and_strict_wire_schema(self):
        self.assertIsNone(_read(io.BytesIO(b"")))
        for value in ({"id":1,"op":"x"},{"id":1,"op":"load","extra":1},{"id":1,"id":2,"op":"load"}):
            raw = b'{"id":1,"id":2,"op":"load"}' if value.get("op")=="load" and len(value)==2 else json.dumps(value).encode()
            with self.assertRaises(RuntimeError): _read(io.BytesIO(struct.pack(">I",len(raw))+raw))
    def test_oversize_and_truncation_are_rejected(self):
        with self.assertRaises(RuntimeError): _read(io.BytesIO(struct.pack(">I",MAX_FRAME+1)))
        with self.assertRaises(RuntimeError): _read(io.BytesIO(struct.pack(">I",3)+b"{}"))

    def test_fragmented_payload_is_read_exactly(self):
        class Fragmented(io.BytesIO):
            def read(self, size=-1): return super().read(1 if size > 4 else size)
        raw=json.dumps({"id":1,"op":"load"}).encode()
        # Header is deliberately normal; payload reads are one byte each.
        self.assertEqual(_read(Fragmented(struct.pack(">I",len(raw))+raw))["op"],"load")

    def test_native_batch_rpc_schema_is_explicit_and_bounded(self):
        request = {"id": 1, "op": "execute_batch", "items": [
            {"instruction": "fix", "texts": ["text"]}]}
        self.assertEqual(_read(self.frame(request)), request)
        malformed = {"id": 1, "op": "execute_batch", "items": [], "texts": []}
        with self.assertRaises(RuntimeError):
            _read(self.frame(malformed))
        with self.assertRaises(RuntimeError):
            _read(self.frame(request), gector=True)

    def test_execute_batch_classification_is_closed_and_uses_typed_cuda_oom(self):
        class TypedOOM(Exception): pass
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(OutOfMemoryError=TypedOOM))
        self.assertEqual(_execute_batch_failure_code(TypedOOM(), torch), "oom")
        self.assertEqual(_execute_batch_failure_code(
            WorkerContractFailure("output_contract_failed"), torch), "output_contract_failed")
        self.assertEqual(_execute_batch_failure_code(RuntimeError("CUDA out of memory"), torch),
                         "worker_operation_failed")

    def test_cuda_ready_rpc_schema_is_strict(self):
        request = {"id": 1, "op": "cuda_ready"}
        self.assertEqual(_read(self.frame(request)), request)
        with self.assertRaises(RuntimeError):
            _read(self.frame({**request, "input": "operator data"}))

    def test_cuda_ready_retains_one_synchronized_device_allocation_for_worker_lifetime(self):
        calls = []
        witness = object()
        class Torch:
            class cuda:
                @staticmethod
                def synchronize(): calls.append("synchronize")
            @staticmethod
            def zeros(shape, *, device): calls.append(("zeros", shape, device)); return witness
        runtime = Runtime(Path("/tmp"), {})
        runtime.model = object()
        with patch.dict(sys.modules, {"torch": Torch}):
            self.assertTrue(runtime.cuda_ready())
        self.assertEqual(calls, [("zeros", (1,), "cuda:0"), "synchronize"])
        self.assertIs(runtime._cuda_residency_witness, witness)

    def test_cuda_ready_rejects_an_unretained_allocation_result(self):
        class Torch:
            class cuda:
                @staticmethod
                def synchronize(): raise AssertionError("must not synchronize an absent allocation")
            @staticmethod
            def zeros(shape, *, device): return None
        runtime = Runtime(Path("/tmp"), {})
        runtime.model = object()
        with patch.dict(sys.modules, {"torch": Torch}):
            with self.assertRaisesRegex(RuntimeError, "allocation is unavailable"):
                runtime.cuda_ready()
        self.assertIsNone(runtime._cuda_residency_witness)

    def test_runtime_uses_full_single_string_framing_and_output_bound(self):
        seen=[]
        class Tokens(dict):
            def to(self, device): return self
            def get(self, key, default=None): return [[1], [2]] if key == "input_ids" else [[1], [1]]
        class Tokenizer:
            def __call__(self, text, **kwargs): seen.append((text,kwargs)); return Tokens(input_ids=[1,2])
            def batch_decode(self, output, **kwargs): return ["fixed"]
        class Output:
            shape=(1,3)
            def __getitem__(self, key): return [[0, 4, 4]][key]
        class Model:
            generation_config = type("G", (), {"eos_token_id": 4, "pad_token_id": 0, "decoder_start_token_id": 0})()
            def generate(self, **kwargs): return Output()
        runtime=Runtime(Path("/tmp"),{"max_input_tokens":2,"max_output_tokens":2,"generation_parameters":{"num_beams":1,"do_sample":False}, "max_native_batch_size": 1})
        runtime.tokenizer, runtime.model=Tokenizer(),Model()
        runtime.tokenizer.eos_token_id = 4; runtime.tokenizer.pad_token_id = 0
        fake_torch=types.SimpleNamespace(inference_mode=lambda: __import__("contextlib").nullcontext())
        with patch.dict(sys.modules,{"torch":fake_torch}):
            self.assertEqual(runtime.execute("Fix:", ["bad text"]), ["fixed"])
        self.assertEqual(seen[0][0], "Fix:\nbad text")
        self.assertEqual(seen[-1][0], ["Fix:\nbad text"])

    def test_runtime_native_batch_proves_cardinality_and_timing_without_prompts(self):
        seen = []
        class Tokens(dict):
            def to(self, device): return self
        class Tokenizer:
            def __call__(self, text, **kwargs):
                seen.append((text, kwargs)); return Tokens(input_ids=[[1], [2]], attention_mask=[[1], [1]])
            def batch_decode(self, output, **kwargs): return ["one-fixed", "two-fixed"]
        class Output:
            shape = (2, 3)
            def __getitem__(self, key): return [[0, 1, 2], [0, 1, 2]][key]
        class Model:
            generation_config = type("G", (), {"eos_token_id": 99, "pad_token_id": 0, "decoder_start_token_id": 0})()
            def generate(self, **kwargs):
                self.encoded = kwargs
                return Output()
        model = Model()
        runtime = Runtime(Path("/tmp"), {"max_input_tokens": 2, "max_output_tokens": 2,
            "max_native_batch_size": 2, "generation_parameters": {"num_beams": 1, "do_sample": False, "eos_token_id": 99, "pad_token_id": 0}})
        runtime.tokenizer, runtime.model = Tokenizer(), model
        fake_torch = types.SimpleNamespace(inference_mode=lambda: __import__("contextlib").nullcontext(), cuda=types.SimpleNamespace(
            synchronize=lambda: None, memory_allocated=lambda device: 10, memory_reserved=lambda device: 20,
            reset_peak_memory_stats=lambda device: None, max_memory_allocated=lambda device: 30,
            max_memory_reserved=lambda device: 40))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            result = runtime.execute_batch([
                {"instruction": "Fix one", "texts": ["one"]},
                {"instruction": "Fix two", "texts": ["two"]},
            ])
        self.assertEqual(result["outputs"], ["one-fixed", "two-fixed"])
        self.assertEqual(result["observation"]["batch_size"], 2)
        self.assertLessEqual(result["observation"]["execution_started"], result["observation"]["execution_ended"])
        self.assertNotIn("outputs", result["observation"])
        self.assertEqual(seen[-1][0], ["Fix one\none", "Fix two\ntwo"])
        self.assertEqual((len(model.encoded["input_ids"]), len(model.encoded["attention_mask"])), (2, 2))
        self.assertEqual(result["observation"]["decoder_steps"], [2, 2])

    def test_native_batch_emits_integer_monotonic_nanosecond_fences(self):
        # Capacity collection may correlate these only with monotonic-ns NVML
        # fences, never loop-clock floats.
        class Tokens(dict):
            def to(self, device): return self
        runtime = Runtime(Path("/tmp"), {"max_input_tokens": 2, "max_output_tokens": 2,
            "max_native_batch_size": 1, "generation_parameters": {"eos_token_id": 9, "pad_token_id": 0}})
        runtime.tokenizer = type("T", (), {"__call__": lambda s, *a, **k: Tokens(input_ids=[[1]], attention_mask=[[1]]),
            "batch_decode": lambda s, o, **k: ["x"]})()
        runtime.model = type("M", (), {"generation_config": type("G", (), {"eos_token_id": 9, "pad_token_id": 0, "decoder_start_token_id": 1})(), "generate": lambda s, **k: type("O", (), {"shape": (1, 1), "__getitem__": lambda s, key: [[1]][key]})()})()
        cuda = types.SimpleNamespace(synchronize=lambda: None, memory_allocated=lambda _: 1,
            memory_reserved=lambda _: 2, reset_peak_memory_stats=lambda _: None,
            max_memory_allocated=lambda _: 1, max_memory_reserved=lambda _: 2)
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(inference_mode=lambda: __import__("contextlib").nullcontext(), cuda=cuda)}):
            observation = runtime.execute_batch([{"instruction": "i", "texts": ["t"]}])["observation"]
        self.assertIs(type(observation["execution_started"]), int)
        self.assertIs(type(observation["execution_ended"]), int)
        self.assertGreaterEqual(observation["execution_ended"], observation["execution_started"])

    def test_decoder_workload_counts_eos_and_ignores_padding(self):
        runtime = Runtime(Path("/tmp"), {"max_output_tokens": 64, "generation_parameters": {"eos_token_id": 9, "pad_token_id": 0}})
        runtime.model = type("M", (), {"generation_config": type("G", (), {"eos_token_id": 9, "pad_token_id": 0, "decoder_start_token_id": 1})()})()
        class Output:
            shape = (2, 5)
            def __getitem__(self, key): return [[1, 2, 9, 0, 0], [1, 2, 3, 4, 5]][key]
        runtime.tokenizer = type("T", (), {})()
        self.assertEqual(runtime._decoder_workload(Output(), 2), [2, 4])

    def test_decoder_workload_early_eos_with_full_candidate_width_counts_only_model_steps(self):
        runtime = Runtime(Path("/tmp"), {"max_output_tokens": 64})
        runtime.model = type("M", (), {"generation_config": type("G", (), {
            "eos_token_id": 9, "pad_token_id": 0, "decoder_start_token_id": 1})()})()
        class Output:
            shape = (1, 65)
            def __getitem__(self, key): return [[1, 2, 9] + [0] * 62][key]
        self.assertEqual(runtime._decoder_workload(Output(), 1), [2])

    def test_decoder_workload_rejects_malformed_and_out_of_bound_witness(self):
        runtime = Runtime(Path("/tmp"), {"max_output_tokens": 2, "generation_parameters": {"eos_token_id": 9, "pad_token_id": 0}})
        class Output:
            shape = (1, 4)
            def __getitem__(self, key): return [[1, 2, 3, 4]][key]
        with self.assertRaises(RuntimeError): runtime._decoder_workload(Output(), 1)

    def test_decoder_workload_requires_authoritative_metadata_and_structured_padding(self):
        runtime = Runtime(Path("/tmp"), {"max_output_tokens": 64})
        class Output:
            shape = (1, 4)
            def __getitem__(self, key): return [[1, 9, 0, 0]][key]
        runtime.model = type("M", (), {"generation_config": type("G", (), {
            "eos_token_id": [9, 10], "pad_token_id": 0, "decoder_start_token_id": 1})()})()
        self.assertEqual(runtime._decoder_workload(Output(), 1), [1])
        runtime.model.generation_config.decoder_start_token_id = 2
        with self.assertRaises(RuntimeError): runtime._decoder_workload(Output(), 1)
        runtime.model.generation_config.decoder_start_token_id = 1
        runtime.model.generation_config.eos_token_id = [9, -1]
        with self.assertRaises(RuntimeError): runtime._decoder_workload(Output(), 1)
        runtime.model.generation_config.eos_token_id = 9
        class BadPadding(Output):
            def __getitem__(self, key): return [[1, 9, 7, 0]][key]
        with self.assertRaises(RuntimeError): runtime._decoder_workload(BadPadding(), 1)

    def test_decoder_workload_supports_ordinary_output_bucket_above_candidate_witness(self):
        runtime = Runtime(Path("/tmp"), {"max_output_tokens": 128})
        runtime.model = type("M", (), {"generation_config": type("G", (), {
            "eos_token_id": 9, "pad_token_id": 0, "decoder_start_token_id": 1})()})()
        class Output:
            shape = (1, 2)
            def __getitem__(self, key): return [[1, 2]][key]
        self.assertEqual(runtime._decoder_workload(Output(), 1), [1])

    def test_native_batch_rejects_over_bound_and_wrong_tensor_cardinality(self):
        runtime = Runtime(Path("/tmp"), {"max_input_tokens": 2, "max_output_tokens": 2,
            "max_native_batch_size": 2, "generation_parameters": {"num_beams": 1, "do_sample": False}})
        runtime.tokenizer = type("T", (), {
            "__call__": lambda s, *a, **k: type("K", (dict,), {
                "to": lambda s, d: s,
                "get": lambda s, key, default=None: [1],
            })(input_ids=[1], attention_mask=[1]),
            "batch_decode": lambda s, o, **k: ["x"],
        })()
        runtime.model = type("M", (), {"generate": lambda s, **k: type("O", (), {"shape": (1, 2)})()})()
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(inference_mode=lambda: __import__("contextlib").nullcontext(), cuda=types.SimpleNamespace(synchronize=lambda: None))}):
            with self.assertRaises(ValueError):
                runtime.execute_batch([{"instruction": "i", "texts": ["t"]}] * 3)
            with self.assertRaises(RuntimeError):
                runtime.execute_batch([{"instruction": "i", "texts": ["t"]}] * 2)

    def test_native_batch_requires_cuda_synchronization(self):
        runtime = Runtime(Path("/tmp"), {"max_input_tokens": 2, "max_output_tokens": 2,
            "max_native_batch_size": 1, "generation_parameters": {"num_beams": 1, "do_sample": False}})
        runtime.tokenizer = type("T", (), {"__call__": lambda s, *a, **k: {"input_ids": [1], "attention_mask": [1]},
            "batch_decode": lambda s, o, **k: ["x"]})()
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(inference_mode=lambda: __import__("contextlib").nullcontext())}):
            with self.assertRaises(RuntimeError):
                runtime.execute_batch([{"instruction": "i", "texts": ["t"]}])

    def test_native_batch_validates_all_items_before_model_execution(self):
        calls = []
        class Tokens(dict):
            def to(self, device): return self
        class Tokenizer:
            def __call__(self, value, **kwargs):
                calls.append(value)
                return Tokens(input_ids=[1, 2, 3] if isinstance(value, str) and "too-long" in value else [1])
        class Model:
            def generate(self, **kwargs): raise AssertionError("model execution must be fenced")
        runtime = Runtime(Path("/tmp"), {"max_input_tokens": 2, "max_output_tokens": 2,
            "max_native_batch_size": 2, "generation_parameters": {}})
        runtime.tokenizer, runtime.model = Tokenizer(), Model()
        # execute_batch imports torch before entering its validation fence;
        # provide the same strict module seam used by the neighboring runtime
        # tests without changing the production import or execution path.
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(
                inference_mode=lambda: __import__("contextlib").nullcontext())}):
            with self.assertRaises(InputValidationError):
                runtime.execute_batch([{"instruction": "ok", "texts": ["text"]},
                                      {"instruction": "too-long", "texts": ["text"]}])
        self.assertEqual(calls, ["ok\ntext", "too-long\ntext"])

    def test_gpu_identity_requires_cuda_uuid_nvml_and_runner_identity(self):
        gpu_uuid="GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963"
        class Props: uuid=gpu_uuid
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True,device_count=lambda:1,current_device=lambda:0,get_device_properties=lambda _:Props()))
        nvml=types.SimpleNamespace(nvmlInit=lambda:None,nvmlShutdown=lambda:None,nvmlDeviceGetCount=lambda:1,nvmlDeviceGetHandleByIndex=lambda _:1,nvmlDeviceGetUUID=lambda _:gpu_uuid,nvmlDeviceGetMigMode=lambda _:(0,0))
        runtime=Runtime(Path("/tmp"),{"gpu_uuid":gpu_uuid})
        with patch.dict(sys.modules,{"torch":torch,"pynvml":nvml}):
            value=runtime.gpu_identity()
        self.assertEqual(value["runner_pid"],os.getpid()); self.assertTrue(value["cuda_nvml_agree"])

    def test_gpu_identity_allows_only_explicit_mig_not_supported(self):
        class NotSupported(Exception): pass
        class Props: uuid=bytes.fromhex("d15a7ff9a19b3ece7510759a0bca1963")
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True,device_count=lambda:1,current_device=lambda:0,get_device_properties=lambda _:Props()))
        nvml=types.SimpleNamespace(NVMLError_NotSupported=NotSupported,nvmlInit=lambda:None,nvmlShutdown=lambda:None,nvmlDeviceGetCount=lambda:1,nvmlDeviceGetHandleByIndex=lambda _:1,nvmlDeviceGetUUID=lambda _:b"GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963",nvmlDeviceGetMigMode=lambda _:(_ for _ in ()).throw(NotSupported()))
        runtime=Runtime(Path("/tmp"),{"gpu_uuid":"GPU-d15a7ff9-a19b-3ece-7510-759a0bca1963"})
        with patch.dict(sys.modules,{"torch":torch,"pynvml":nvml}): self.assertTrue(runtime.gpu_identity()["cuda_nvml_agree"])
        nvml.nvmlDeviceGetMigMode=lambda _: (1,1)
        with patch.dict(sys.modules,{"torch":torch,"pynvml":nvml}):
            with self.assertRaises(RuntimeError): runtime.gpu_identity()

    def test_load_uses_strict_offline_safetensors_cuda_and_dtype(self):
        calls = {}
        class Model:
            def to(self, device): calls["device"] = device; return self
            def eval(self): calls["eval"] = True
        class Torch:
            float16="fp16"; bfloat16="bf16"; float32="fp32"
            @staticmethod
            def device(value): return value
        transformers=types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda root, **kw: calls.setdefault("tokenizer",kw)),
            T5ForConditionalGeneration=types.SimpleNamespace(from_pretrained=lambda root, **kw: calls.setdefault("model",kw) and Model()))
        runtime=Runtime(Path("/offline"),{"dtype":"float16"})
        with patch.dict(sys.modules,{"torch":Torch,"transformers":transformers}): runtime.load()
        self.assertEqual(calls["tokenizer"],{"local_files_only":True,"trust_remote_code":False})
        self.assertEqual(calls["model"],{"local_files_only":True,"trust_remote_code":False,"use_safetensors":True,"torch_dtype":"fp16"})
        self.assertEqual(calls["device"],"cuda:0")

    def test_output_over_decoder_start_bound_is_rejected(self):
        runtime=Runtime(Path("/tmp"),{"max_input_tokens":2,"max_output_tokens":2,"generation_parameters":{"num_beams":1,"do_sample":False}})
        class Tokens(dict):
            def to(self, device): return self
        runtime.tokenizer=type("T",(),{"__call__":lambda s,*a,**k:Tokens(input_ids=[1]),"batch_decode":lambda s,o,**k:["x"]})()
        runtime.model=type("M",(),{"generate":lambda s,**k:type("O",(),{"shape":(1,4)})()})()
        with patch.dict(sys.modules,{"torch":types.SimpleNamespace(inference_mode=lambda:__import__("contextlib").nullcontext())}):
            with self.assertRaises(RuntimeError): runtime.execute("i",["t"])

    def test_gector_validate_dispatch_preserves_bounded_false_without_cuda(self):
        source = '''
from services.llm.providers import python_worker
class FakeRuntime:
    def __init__(self, root, config): pass
    def validate(self, **values): return {"accepted": False}
raise SystemExit(python_worker.main(FakeRuntime, True))
'''
        request = {"id": 1, "op": "validate", "texts": ["too long"],
                   "keep_confidence": 0, "min_error_prob": 0.0,
                   "n_iteration": 1, "batch_size": 1}
        raw = json.dumps(request, separators=(",", ":")).encode()
        env = {**os.environ, "LLM_PYTHON_MODEL_ROOT": "/offline",
               "LLM_PYTHON_WORKER_CONFIG": "{}"}
        completed = subprocess.run([sys.executable, "-c", source],
            input=struct.pack(">I", len(raw)) + raw, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, check=True)
        size = struct.unpack(">I", completed.stdout[:4])[0]
        self.assertEqual(json.loads(completed.stdout[4:4 + size]),
                         {"id": 1, "ok": True, "value": {"accepted": False}})

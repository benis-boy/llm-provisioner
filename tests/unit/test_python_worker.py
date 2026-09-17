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

from services.llm.providers.python_worker import MAX_FRAME, Runtime, _read


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

    def test_runtime_uses_full_single_string_framing_and_output_bound(self):
        seen=[]
        class Tokens(dict):
            def to(self, device): return self
        class Tokenizer:
            def __call__(self, text, **kwargs): seen.append((text,kwargs)); return Tokens(input_ids=[1,2])
            def batch_decode(self, output, **kwargs): return ["fixed"]
        class Output:
            shape=(1,3)
        class Model:
            def generate(self, **kwargs): return Output()
        runtime=Runtime(Path("/tmp"),{"max_input_tokens":2,"max_output_tokens":2,"generation_parameters":{"num_beams":1,"do_sample":False}})
        runtime.tokenizer, runtime.model=Tokenizer(),Model()
        fake_torch=types.SimpleNamespace(inference_mode=lambda: __import__("contextlib").nullcontext())
        with patch.dict(sys.modules,{"torch":fake_torch}):
            self.assertEqual(runtime.execute("Fix:", ["bad text"]), ["fixed"])
        self.assertEqual(seen[0][0], "Fix:\nbad text")
        self.assertEqual(seen[-1][0], ["Fix:\nbad text"])

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

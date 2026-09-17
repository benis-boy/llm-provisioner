"""Child-only CoEdIT runtime and strict binary-framed JSON RPC."""
from __future__ import annotations
import hashlib, json, math, os, struct, sys, uuid
from pathlib import Path
MAX_FRAME = 256 * 1024
MAX_RPC_ID = 2 ** 63 - 1
class InsufficientMaxInput(ValueError): pass
class BenchmarkInputError(ValueError): pass
class InputValidationError(ValueError): pass
def _pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError("duplicate JSON key")
        result[key] = value
    return result
def _read(stream, gector=False):
    header=stream.read(4)
    if not header: return None
    if len(header)!=4: raise RuntimeError("truncated RPC frame")
    size=struct.unpack(">I",header)[0]
    if not 0<size<=MAX_FRAME: raise RuntimeError("RPC frame exceeds bound")
    data=bytearray()
    while len(data)<size:
        chunk=stream.read(size-len(data))
        if not chunk: raise RuntimeError("truncated RPC payload")
        data.extend(chunk)
    try: value=json.loads(bytes(data).decode("utf-8"), object_pairs_hook=_pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc: raise RuntimeError("malformed RPC request") from exc
    if not isinstance(value,dict) or type(value.get("id")) is not int or not 1 <= value["id"] <= MAX_RPC_ID or not isinstance(value.get("op"),str): raise RuntimeError("malformed RPC request")
    allowed={"load":{"id","op"},"cuda_ready":{"id","op"},"cuda_residency":{"id","op"},"gpu_identity":{"id","op"},"shutdown":{"id","op"},"validate":{"id","op","instruction","texts"},"execute":{"id","op","instruction","texts"},"execute_batch":{"id","op","items"},"benchmark_input":{"id","op","instruction","text","generate"}}
    if gector: allowed.update({"validate":{"id","op","texts","keep_confidence","min_error_prob","n_iteration","batch_size"},"execute":{"id","op","texts","keep_confidence","min_error_prob","n_iteration","batch_size"}}); allowed.pop("execute_batch", None); allowed.pop("benchmark_input", None)
    if value["op"] not in allowed or set(value)!=allowed[value["op"]]: raise RuntimeError("malformed RPC request")
    return value
def _write(value):
    data=json.dumps(value,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
    if len(data)>MAX_FRAME: raise RuntimeError("RPC response exceeds bound")
    _PROTOCOL.write(struct.pack(">I",len(data))+data); _PROTOCOL.flush()
class Runtime:
    def __init__(self,root,config): self.root,self.config,self.tokenizer,self.model,self._cuda_residency_witness=root,config,None,None,None
    def load(self):
        import torch
        from transformers import AutoTokenizer,T5ForConditionalGeneration
        self.tokenizer=AutoTokenizer.from_pretrained(self.root,local_files_only=True,trust_remote_code=False)
        dtype={"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[self.config["dtype"]]
        self.model=T5ForConditionalGeneration.from_pretrained(self.root,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=dtype).to(torch.device("cuda:0")); self.model.eval()
    def cuda_ready(self):
        """Retain a synchronized CUDA allocation for this loaded worker's lifetime."""
        import torch
        if self.model is None: raise RuntimeError("model is not loaded")
        cuda = getattr(torch, "cuda", None)
        synchronize = getattr(cuda, "synchronize", None)
        zeros = getattr(torch, "zeros", None)
        if not callable(zeros) or not callable(synchronize):
            raise RuntimeError("CUDA readiness operation is unavailable")
        witness = zeros((1,), device="cuda:0")
        if witness is None:
            raise RuntimeError("CUDA readiness allocation is unavailable")
        synchronize()
        # A temporary tensor is decref'd when this RPC returns and consequently
        # cannot be evidence of continuing model-worker residency. Keep one
        # bounded device allocation owned by this loaded Runtime until worker
        # exit; the resident model and this witness therefore share exactly the
        # worker process which GPU ownership later proves.
        self._cuda_residency_witness = witness
        return True
    def cuda_residency(self):
        """Return a scalar witness that is checked on every readiness sample."""
        import torch
        witness = self._cuda_residency_witness
        if self.model is None or witness is None:
            raise RuntimeError("CUDA residency witness is unavailable")
        device = getattr(witness, "device", None)
        model_device = next(self.model.parameters()).device
        if str(device) != "cuda:0" or str(model_device) != "cuda:0":
            raise RuntimeError("CUDA residency device mismatch")
        return {"model_cuda_device":"cuda:0", "witness_exists":True,
                "witness_cuda_device":"cuda:0"}
    @staticmethod
    def _physical_uuid(value):
        if isinstance(value,bytes):
            if len(value)==16: return str(uuid.UUID(bytes=value))
            value=value.decode("ascii")
        elif not isinstance(value,str): value=str(value)
        if value.startswith("MIG-"): raise ValueError("MIG UUID is not physical")
        if value.startswith("GPU-"): value=value[4:]
        return str(uuid.UUID(value))
    def gpu_identity(self):
        import torch,pynvml
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.current_device()!=0: raise RuntimeError("CUDA device contract failed")
        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount()!=1: raise RuntimeError("NVML device contract failed")
            handle=pynvml.nvmlDeviceGetHandleByIndex(0); raw=pynvml.nvmlDeviceGetUUID(handle); uuid=raw.decode("ascii") if isinstance(raw,bytes) else raw
            mig_fn=getattr(pynvml,"nvmlDeviceGetMigMode",None)
            if not callable(mig_fn): raise RuntimeError("gpu_mig_api_unavailable")
            try: mig=mig_fn(handle)
            except Exception as exc:
                unsupported=getattr(pynvml,"NVMLError_NotSupported",None)
                if not isinstance(unsupported,type) or not isinstance(exc,unsupported): raise RuntimeError("gpu_mig_api_failed") from exc
                mig=(0,0)
            cuda_uuid=getattr(torch.cuda.get_device_properties(0),"uuid",None)
            if (not isinstance(uuid,str) or self._physical_uuid(uuid)!=self._physical_uuid(self.config["gpu_uuid"])
                    or cuda_uuid is None or self._physical_uuid(cuda_uuid)!=self._physical_uuid(uuid)
                    or not isinstance(mig,(tuple,list)) or len(mig)<2 or mig[0]!=0 or mig[1]!=0): raise RuntimeError("gpu_identity_mismatch")
            return {"gpu_uuid":uuid,"runner_pid":os.getpid(),"runner_start_time":_identity(os.getpid()),"cuda_nvml_agree":True}
        finally: pynvml.nvmlShutdown()
    @staticmethod
    def frame(instruction,text): return instruction + "\n" + text
    def validate(self,instruction,texts):
        if not isinstance(instruction,str) or not instruction.strip() or not isinstance(texts,list) or len(texts)!=1 or not isinstance(texts[0],str) or not texts[0].strip(): raise InputValidationError("invalid request shape")
        ids=self.tokenizer(self.frame(instruction,texts[0]),add_special_tokens=True,truncation=False).get("input_ids")
        if not isinstance(ids,list): raise RuntimeError("tokenizer returned invalid input ids")
        if len(ids)>self.config["max_input_tokens"]: raise InputValidationError("request exceeds no-truncation bucket")
    def _benchmark_once(self,instruction,text):
        if not isinstance(instruction,str) or not instruction.strip() or not isinstance(text,str) or not text.strip():
            raise ValueError("invalid benchmark input")
        payload=json.dumps({"instruction":instruction,"texts":[text]},ensure_ascii=True,separators=(",", ":")).encode()
        ids=self.tokenizer(self.frame(instruction,text),add_special_tokens=True,truncation=False).get("input_ids")
        if not isinstance(ids,list): raise BenchmarkInputError("benchmark_tokenization_failed")
        count=len(ids); maximum=self.config["max_input_tokens"]
        if count>maximum: raise BenchmarkInputError("benchmark_input_over_bound")
        return {"count":count,"max":maximum,"fingerprint":hashlib.sha256(payload).hexdigest()}
    def benchmark_input(self,instruction,text,generate):
        if type(generate) is not bool: raise ValueError("invalid benchmark generation flag")
        if not generate: return self._benchmark_once(instruction,text)
        # This path is deliberately bounded and deterministic.  It is useful to
        # probe a fixture without ever padding or inventing an unbounded corpus.
        candidate=text
        for attempt in range(256):
            try:
                witness=self._benchmark_once(instruction,candidate)
            except BenchmarkInputError as exc:
                if str(exc) == "benchmark_input_over_bound": break
                raise
            if witness["count"]==witness["max"]:
                witness["text"] = candidate
                return witness
            candidate += (" " if candidate else "") + "word"
        raise InsufficientMaxInput("configured fixture cannot reach maximum")
    def execute(self,instruction,texts):
        import torch
        self.validate(instruction,texts)
        with torch.inference_mode():
            encoded=self.tokenizer([self.frame(instruction,texts[0])],return_tensors="pt",padding=True,truncation=False).to("cuda:0")
            output=self.model.generate(**encoded,max_new_tokens=self.config["max_output_tokens"],**self.config["generation_parameters"])
        if output.shape[0] != 1 or output.shape[1] > self.config["max_output_tokens"] + 1: raise RuntimeError("generation exceeded decoder-start output bound")
        result=self.tokenizer.batch_decode(output,skip_special_tokens=True)
        if len(result)!=len(texts) or any(not isinstance(x,str) or not x for x in result): raise RuntimeError("invalid output")
        return result
    def execute_batch(self, items):
        import time, torch
        limit = int(self.config.get("max_native_batch_size", 1))
        if not isinstance(items, list) or not 1 <= len(items) <= limit:
            raise ValueError("invalid native batch size")
        frames = []
        for item in items:
            if (not isinstance(item, dict) or set(item) != {"instruction", "texts"}
                    or not isinstance(item["instruction"], str) or not isinstance(item["texts"], list)
                    or len(item["texts"]) != 1):
                raise InputValidationError("invalid native batch item")
            self.validate(item["instruction"], item["texts"])
            frames.append(self.frame(item["instruction"], item["texts"][0]))
        envelope = {"id": MAX_RPC_ID, "op": "execute_batch", "items": items}
        if len(json.dumps(envelope, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()) + 4 > int(self.config.get("rpc_frame_limit", MAX_FRAME)):
            raise RuntimeError("native batch request exceeds bound")
        cuda = getattr(torch, "cuda", None)
        if self.config.get("cuda_timing", True) is not True or not callable(getattr(cuda, "synchronize", None)):
            raise RuntimeError("CUDA synchronization is required for native batch timing")
        synchronized = True
        torch.cuda.synchronize()
        # These are deliberately the child-owned Torch allocator counters.  They
        # are not a process-wide CUDA peak and must not be presented as one.
        allocator = getattr(torch.cuda, "memory_allocated", None), getattr(torch.cuda, "memory_reserved", None)
        reset_peak = getattr(torch.cuda, "reset_peak_memory_stats", None)
        peak_allocated = getattr(torch.cuda, "max_memory_allocated", None)
        peak_reserved = getattr(torch.cuda, "max_memory_reserved", None)
        if not all(callable(value) for value in (allocator[0], allocator[1], reset_peak, peak_allocated, peak_reserved)):
            raise RuntimeError("Torch CUDA allocator measurement is unavailable")
        baseline_allocated, baseline_reserved = allocator[0](0), allocator[1](0)
        if any(type(value) is not int for value in (baseline_allocated, baseline_reserved)):
            raise RuntimeError("invalid Torch CUDA allocator baseline")
        reset_peak(0)
        started = time.monotonic_ns()
        with torch.inference_mode():
            encoded = self.tokenizer(frames, return_tensors="pt", padding=True, truncation=False).to("cuda:0")
            output = self.model.generate(**encoded, max_new_tokens=self.config["max_output_tokens"], **self.config["generation_parameters"])
        torch.cuda.synchronize()
        ended = time.monotonic_ns()
        peak_allocated_value, peak_reserved_value = peak_allocated(0), peak_reserved(0)
        final_allocated, final_reserved = allocator[0](0), allocator[1](0)
        values = (peak_allocated_value, peak_reserved_value, final_allocated, final_reserved)
        if any(type(value) is not int for value in values) or not (
                baseline_allocated <= peak_allocated_value and baseline_allocated <= final_allocated
                and baseline_reserved <= peak_reserved_value and baseline_reserved <= final_reserved
                and peak_allocated_value >= final_allocated >= 0
                and peak_reserved_value >= final_reserved >= 0
                and baseline_allocated <= baseline_reserved
                and peak_allocated_value <= peak_reserved_value
                and final_allocated <= final_reserved):
            raise RuntimeError("inconsistent Torch CUDA allocator measurement")
        encoded_values = encoded if hasattr(encoded, "get") else None
        if encoded_values is None or not self._batch_dimension_matches(encoded_values.get("input_ids"), len(items)) or not self._batch_dimension_matches(encoded_values.get("attention_mask"), len(items)):
            raise RuntimeError("invalid encoded native batch cardinality")
        if output.shape[0] != len(items) or output.shape[1] > self.config["max_output_tokens"] + 1:
            raise RuntimeError("generation exceeded decoder-start output bound")
        decoder_steps = self._decoder_workload(output, len(items))
        result = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        if len(result) != len(items) or any(not isinstance(x, str) or not x for x in result):
            raise RuntimeError("invalid output")
        response = {"outputs": result, "observation": {"batch_size": len(items), "execution_started": started, "execution_ended": ended, "cuda_synchronized": synchronized,
            "decoder_steps": decoder_steps, "max_output_tokens": self.config["max_output_tokens"],
            "allocator": {"baseline_allocated": baseline_allocated, "baseline_reserved": baseline_reserved,
                          "peak_allocated": peak_allocated_value, "peak_reserved": peak_reserved_value,
                          "final_allocated": final_allocated, "final_reserved": final_reserved}}}
        if len(json.dumps(response, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()) + 4 > int(self.config.get("rpc_frame_limit", MAX_FRAME)):
            raise RuntimeError("native batch response exceeds bound")
        return response

    def _decoder_workload(self, output, expected_batch):
        """Return the bounded, per-row decoder witness for a real generation.

        ``generate`` includes T5's decoder-start token in column zero.  Only
        generated IDs are inspected; the IDs themselves never cross the RPC
        boundary.  A row ends at its first configured EOS, including EOS, so
        the padding emitted by generation after an early stop is not workload.
        """
        maximum = self.config.get("max_output_tokens")
        if type(maximum) is not int or maximum < 1:
            raise RuntimeError("invalid decoder workload bound")
        shape = getattr(output, "shape", None)
        if shape is None or len(shape) != 2 or type(shape[0]) is not int or type(shape[1]) is not int:
            raise RuntimeError("invalid generated output dimensionality")
        if shape[0] != expected_batch or shape[1] < 1 or shape[1] > maximum + 1:
            raise RuntimeError("generated output cardinality exceeds bound")
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is None:
            raise RuntimeError("missing generation token metadata")
        eos = getattr(generation_config, "eos_token_id", None)
        pad = getattr(generation_config, "pad_token_id", None)
        decoder_start = getattr(generation_config, "decoder_start_token_id", None)
        eos_ids = (eos,) if type(eos) is int else tuple(eos) if isinstance(eos, (list, tuple)) else ()
        # These are model metadata, not request inputs.  Still bound a list so a
        # malformed object cannot turn per-token membership checks into work.
        if (not eos_ids or len(eos_ids) > 64 or any(type(value) is not int or value < 0 for value in eos_ids)
                or type(pad) is not int or pad < 0
                or type(decoder_start) is not int or decoder_start < 0):
            raise RuntimeError("missing generation token metadata")
        counts = []
        for row in range(expected_batch):
            count = 0
            try: first = output[row][0]
            except (IndexError, KeyError, TypeError): raise RuntimeError("malformed generated output")
            item = getattr(first, "item", None)
            if callable(item): first = item()
            if type(first) is not int or first != decoder_start:
                raise RuntimeError("invalid decoder-start token")
            finished = False
            for column in range(1, shape[1]):
                try: token = output[row][column]
                except (IndexError, KeyError, TypeError): raise RuntimeError("malformed generated output")
                item = getattr(token, "item", None)
                if callable(item): token = item()
                if type(token) is not int or token < 0:
                    raise RuntimeError("malformed generated token ID")
                if finished:
                    if token != pad:
                        raise RuntimeError("invalid generated padding")
                    continue
                count += 1
                if token in eos_ids:
                    finished = True
            if count > maximum: raise RuntimeError("decoder workload exceeds configured bound")
            counts.append(count)
        if len(counts) != expected_batch or any(type(value) is not int for value in counts):
            raise RuntimeError("malformed decoder workload witness")
        return counts
    @staticmethod
    def _batch_dimension_matches(value, expected):
        try: return len(value) == expected
        except (TypeError, AttributeError): return False
def _identity(pid):
    raw=(Path("/proc")/str(pid)/"stat").read_bytes(); fields=raw[raw.rfind(b")")+2:].split(); return int(fields[19])
_PROTOCOL = sys.stdout.buffer
def main(runtime_class=Runtime, gector=False):
    # Keep protocol on a duplicated FD; native libraries may write directly to fd 1.
    global _PROTOCOL
    _PROTOCOL=os.fdopen(os.dup(sys.stdout.fileno()),"wb",closefd=True)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    runtime=runtime_class(Path(os.environ["LLM_PYTHON_MODEL_ROOT"]),json.loads(os.environ["LLM_PYTHON_WORKER_CONFIG"]))
    for request in iter(lambda:_read(sys.stdin.buffer, gector),None):
        try:
            op=request["op"]
            if op == "load": value=runtime.load() or True
            elif op == "cuda_ready": value=runtime.cuda_ready()
            elif op == "cuda_residency": value=runtime.cuda_residency()
            elif op == "gpu_identity": value=runtime.gpu_identity()
            elif op == "validate":
                value = (runtime.validate(**{k:request[k] for k in request if k not in {"id","op"}}) if gector else runtime.validate(request.get("instruction"),request.get("texts")))
                if not gector: value = value or True
            elif op == "execute": value=runtime.execute(**{k:request[k] for k in request if k not in {"id","op"}}) if gector else runtime.execute(request.get("instruction"),request.get("texts"))
            elif op == "execute_batch": value=runtime.execute_batch(request["items"])
            elif op == "benchmark_input": value=runtime.benchmark_input(request["instruction"],request["text"],request["generate"])
            else: value=None
            if op=="shutdown": break
            _write({"id":request.get("id"),"ok":True,"value":value})
        except Exception as exc:
            code = "worker_operation_failed"
            if isinstance(exc, InputValidationError): code = "request_validation_failed"
            if isinstance(exc, InsufficientMaxInput): code = "insufficient_max_input"
            if isinstance(exc, BenchmarkInputError): code = str(exc)
            if isinstance(exc, RuntimeError) and str(exc) in {"gpu_mig_api_unavailable", "gpu_mig_api_failed", "gpu_identity_mismatch"}:
                code = str(exc)
            _write({"id":request.get("id"),"ok":False,"error":code})
    return 0
if __name__=="__main__": raise SystemExit(main())

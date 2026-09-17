"""Child-only CoEdIT runtime and strict binary-framed JSON RPC."""
from __future__ import annotations
import json, os, struct, sys, uuid
from pathlib import Path
MAX_FRAME = 256 * 1024
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
    if not isinstance(value,dict) or type(value.get("id")) is not int or value["id"] < 1 or not isinstance(value.get("op"),str): raise RuntimeError("malformed RPC request")
    allowed={"load":{"id","op"},"gpu_identity":{"id","op"},"shutdown":{"id","op"},"validate":{"id","op","instruction","texts"},"execute":{"id","op","instruction","texts"}}
    if gector: allowed.update({"validate":{"id","op","texts","keep_confidence","min_error_prob","n_iteration","batch_size"},"execute":{"id","op","texts","keep_confidence","min_error_prob","n_iteration","batch_size"}})
    if value["op"] not in allowed or set(value)!=allowed[value["op"]]: raise RuntimeError("malformed RPC request")
    return value
def _write(value):
    data=json.dumps(value,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
    if len(data)>MAX_FRAME: raise RuntimeError("RPC response exceeds bound")
    _PROTOCOL.write(struct.pack(">I",len(data))+data); _PROTOCOL.flush()
class Runtime:
    def __init__(self,root,config): self.root,self.config,self.tokenizer,self.model=root,config,None,None
    def load(self):
        import torch
        from transformers import AutoTokenizer,T5ForConditionalGeneration
        self.tokenizer=AutoTokenizer.from_pretrained(self.root,local_files_only=True,trust_remote_code=False)
        dtype={"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[self.config["dtype"]]
        self.model=T5ForConditionalGeneration.from_pretrained(self.root,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=dtype).to(torch.device("cuda:0")); self.model.eval()
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
        if not isinstance(instruction,str) or not instruction.strip() or not isinstance(texts,list) or len(texts)!=1 or not isinstance(texts[0],str) or not texts[0].strip(): raise ValueError("invalid request shape")
        ids=self.tokenizer(self.frame(instruction,texts[0]),add_special_tokens=True,truncation=False).get("input_ids")
        if not isinstance(ids,list) or len(ids)>self.config["max_input_tokens"]: raise ValueError("request exceeds no-truncation bucket")
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
            elif op == "gpu_identity": value=runtime.gpu_identity()
            elif op == "validate":
                value = (runtime.validate(**{k:request[k] for k in request if k not in {"id","op"}}) if gector else runtime.validate(request.get("instruction"),request.get("texts")))
                if not gector: value = value or True
            elif op == "execute": value=runtime.execute(**{k:request[k] for k in request if k not in {"id","op"}}) if gector else runtime.execute(request.get("instruction"),request.get("texts"))
            else: value=None
            if op=="shutdown": break
            _write({"id":request.get("id"),"ok":True,"value":value})
        except Exception as exc:
            code = "worker_operation_failed"
            if isinstance(exc, RuntimeError) and str(exc) in {"gpu_mig_api_unavailable", "gpu_mig_api_failed", "gpu_identity_mismatch"}:
                code = str(exc)
            _write({"id":request.get("id"),"ok":False,"error":code})
    return 0
if __name__=="__main__": raise SystemExit(main())

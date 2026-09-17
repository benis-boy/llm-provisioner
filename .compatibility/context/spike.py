"""Fail-closed, sequential real-model compatibility spike."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from urllib.request import Request, urlopen

sys.path.insert(0, "/opt/llm")
from artifacts import verify_manifest
try:
    from .model_runtime import MAX_FIXTURES, SMALL_FIXTURES
    from .input_bounds import SMOLLM_CONTEXT, SMOLLM_OUTPUT_RESERVE, SMOLLM_PROMPT_TOKEN_LIMIT, frame_smollm_prompt, validate_smollm_input
except ImportError:  # copied flat into the candidate image
    from model_runtime import MAX_FIXTURES, SMALL_FIXTURES
    from input_bounds import SMOLLM_CONTEXT, SMOLLM_OUTPUT_RESERVE, SMOLLM_PROMPT_TOKEN_LIMIT, frame_smollm_prompt, validate_smollm_input


class Deadline(Exception):
    pass


def _text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _physical_uuid(value: str | bytes | object) -> str:
    """Normalize only CUDA's bare UUID and NVML's ``GPU-`` UUID forms."""
    if isinstance(value, bytes):
        if len(value) == 16:
            return str(uuid.UUID(bytes=value))
        value = value.decode("ascii")
    elif not isinstance(value, str):
        value = str(value)
    if value.startswith("MIG-"):
        raise ValueError("MIG UUID is not an exclusive physical GPU UUID")
    if value.startswith("GPU-"):
        value = value[4:]
    return str(uuid.UUID(value))


def _same_physical_uuid(*values: str | bytes | object) -> bool:
    try:
        return len({_physical_uuid(value) for value in values}) == 1
    except (UnicodeDecodeError, ValueError, AttributeError):
        return False


def _alarm(_signum: int, _frame: object) -> None:
    raise Deadline("spike hard timeout exceeded")


def _request(path: str, payload: dict | None = None, timeout: int = 120) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request("http://127.0.0.1:11434" + path, data=data,
                      headers={"Content-Type": "application/json"} if data else {},
                      method="GET" if data is None else "POST")
    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise RuntimeError(f"invalid Ollama response from {path}")
    return result


def _nvml_processes(handle: object) -> set[int]:
    from pynvml import nvmlDeviceGetComputeRunningProcesses
    try:
        return {int(process.pid) for process in nvmlDeviceGetComputeRunningProcesses(handle)}
    except Exception as error:
        raise RuntimeError(f"NVML cannot prove compute-process cleanup: {error}") from error


def _nvml_baseline() -> tuple[object, set[int]]:
    from pynvml import nvmlDeviceGetHandleByIndex, nvmlInit
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    return handle, _nvml_processes(handle)


def _gpu_check() -> dict:
    import torch
    from pynvml import nvmlDeviceGetName, nvmlDeviceGetUUID, nvmlInit
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device is required")
    torch.cuda.init()
    nvmlInit()
    from pynvml import nvmlDeviceGetHandleByIndex
    handle = nvmlDeviceGetHandleByIndex(0)
    nvml_uuid = _text(nvmlDeviceGetUUID(handle))
    name = _text(nvmlDeviceGetName(handle))
    smi_uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    properties = torch.cuda.get_device_properties(0)
    torch_uuid = getattr(properties, "uuid", None)
    if not smi_uuid or "\n" in smi_uuid or not _same_physical_uuid(smi_uuid, nvml_uuid):
        raise RuntimeError(f"nvidia-smi/NVML GPU UUID identity mismatch: smi={smi_uuid!r}, nvml={nvml_uuid!r}")
    if torch_uuid is not None and not _same_physical_uuid(torch_uuid, nvml_uuid):
        raise RuntimeError(f"CUDA/NVML GPU UUID identity mismatch: torch={str(torch_uuid)!r}, nvml={nvml_uuid!r}")
    return {"torch_device": torch.cuda.get_device_name(0), "nvml_name": name,
            "uuid": nvml_uuid, "memory_total": int(properties.total_memory)}


def _terminate_group(process: subprocess.Popen[bytes], timeout: int = 20) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _smollm(root: Path) -> None:
    # Evidence is collected before Ollama starts: no GPU/provider execution can
    # occur for an unproved selected artifact or an oversized fixture.
    validate_smollm_input(root, SMALL_FIXTURES["SmolLM"])
    validate_smollm_input(root, MAX_FIXTURES["SmolLM"])
    env = {**os.environ, "OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_MODELS": "/tmp/ollama-models"}
    server = subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        for _ in range(40):
            try:
                _request("/api/version")
                break
            except Exception:
                time.sleep(.25)
        else:
            raise RuntimeError("Ollama loopback server did not become ready")
        name = "compatibility-spike-smollm"
        subprocess.run(["ollama", "create", name, "-f", str(root / "Modelfile")], check=True,
                       timeout=60, env=env, cwd=root)
        for fixture in (SMALL_FIXTURES["SmolLM"], MAX_FIXTURES["SmolLM"]):
            result = _request("/api/generate", {"model": name, "prompt": frame_smollm_prompt(fixture),
                                                  "raw": True, "stream": False, "keep_alive": 0,
                                                  "options": {"num_ctx": SMOLLM_CONTEXT, "num_predict": SMOLLM_OUTPUT_RESERVE, "temperature": 0}}, timeout=240)
            if (result.get("done") is not True or not isinstance(result.get("prompt_eval_count"), int)
                    or result["prompt_eval_count"] > SMOLLM_PROMPT_TOKEN_LIMIT
                    or not isinstance(result.get("response"), str) or not result["response"].strip()):
                raise RuntimeError("SmolLM returned incomplete, truncated, or empty output")
        subprocess.run(["ollama", "rm", name], check=True, timeout=30, env=env)
    finally:
        _terminate_group(server)


def _coedit(root: Path) -> None:
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    for fixture in (SMALL_FIXTURES["CoEdIT"], MAX_FIXTURES["CoEdIT"]):
        if len(tokenizer(fixture, truncation=False, add_special_tokens=True)["input_ids"]) > 128:
            raise RuntimeError("CoEdIT fixture exceeds configured upper-fixture bucket")
    model = T5ForConditionalGeneration.from_pretrained(root, local_files_only=True, use_safetensors=True).to("cuda")
    with torch.inference_mode():
        encoded = tokenizer("Fix grammatical errors: This are valid.", return_tensors="pt").to("cuda")
        output = model.generate(**encoded, max_new_tokens=32)
    if not tokenizer.decode(output[0], skip_special_tokens=True).strip():
        raise RuntimeError("CoEdIT returned an empty response")


def _gector(root: Path) -> None:
    import torch
    from gector import GECToR, GECToRConfig, load_verb_dict, predict
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    import gector.modeling as gector_modeling
    config = GECToRConfig.from_pretrained(root, local_files_only=True)
    # Published microsoft/deberta-large config, checked against selected checkpoint
    # header: 24 layers, 1024 hidden size, 50266 resized embeddings, and 5001 labels.
    base_config = {"model_type": "deberta", "attention_probs_dropout_prob": 0.1, "hidden_act": "gelu",
                   "hidden_dropout_prob": 0.1, "hidden_size": 1024, "initializer_range": 0.02,
                   "intermediate_size": 4096, "max_position_embeddings": 512, "relative_attention": True,
                   "pos_att_type": "c2p|p2c", "layer_norm_eps": 1e-7, "max_relative_positions": -1,
                   "position_biased_input": False, "num_attention_heads": 16, "num_hidden_layers": 24,
                   "type_vocab_size": 0, "vocab_size": 50265}
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        (base / "config.json").write_text(json.dumps(base_config, sort_keys=True), encoding="utf-8")
        config.model_id = str(base)
        model_loader = gector_modeling.AutoModel.from_pretrained
        tokenizer_loader = gector_modeling.AutoTokenizer.from_pretrained
        local_tokenizer_loader = AutoTokenizer.from_pretrained
        gector_modeling.AutoModel.from_pretrained = lambda _path, **_kwargs: AutoModel.from_config(AutoConfig.from_pretrained(base, local_files_only=True))
        gector_modeling.AutoTokenizer.from_pretrained = lambda _path, **_kwargs: local_tokenizer_loader(root, local_files_only=True)
        try:
            model, loading = GECToR.from_pretrained(root, config=config, local_files_only=True,
                                                     output_loading_info=True)
        finally:
            gector_modeling.AutoModel.from_pretrained = model_loader
            gector_modeling.AutoTokenizer.from_pretrained = tokenizer_loader
        if loading["missing_keys"] or loading["unexpected_keys"] or loading["mismatched_keys"]:
            raise RuntimeError(f"GECToR weight load is incomplete: {loading}")
        model = model.to("cuda")
    tokenizer = local_tokenizer_loader(root, local_files_only=True)
    encode, decode = load_verb_dict(str(root / "verb-form-vocab.txt"))
    for fixture in (SMALL_FIXTURES["GECToR"], MAX_FIXTURES["GECToR"]):
        if len(tokenizer(fixture, truncation=False, add_special_tokens=True)["input_ids"]) > 128:
            raise RuntimeError("GECToR fixture exceeds configured upper-fixture bucket")
        result = predict(model, tokenizer, [fixture], encode, decode,
                         keep_confidence=0.0, min_error_prob=0.0, n_iteration=1, batch_size=1)
        if not result or not str(result[0]).strip():
            raise RuntimeError("GECToR returned an empty response")


def _child(model: str, root: str) -> int:
    _gpu_check()
    {"SmolLM": _smollm, "CoEdIT": _coedit, "GECToR": _gector}[model](Path(root))
    return 0


def _run_model(model: str, root: Path, baseline: set[int], handle: object) -> None:
    child = subprocess.Popen([sys.executable, __file__, "--child", model, "--root", str(root)], start_new_session=True)
    try:
        child.wait(timeout=540)
    except subprocess.TimeoutExpired:
        _terminate_group(child, timeout=10)
        raise RuntimeError(f"{model} child timed out")
    if child.returncode:
        raise RuntimeError(f"{model} child failed with exit status {child.returncode}")
    if child.pid in _nvml_processes(handle):
        raise RuntimeError(f"{model} child PID remains on GPU after exit")
    if _nvml_processes(handle) != baseline:
        raise RuntimeError(f"{model} changed global NVML compute processes from baseline")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", choices=("SmolLM", "CoEdIT", "GECToR"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--child", choices=("SmolLM", "CoEdIT", "GECToR"))
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.child:
        if not args.root:
            parser.error("--child requires --root")
        return _child(args.child, str(args.root))
    if not (args.all or args.model) or not args.manifest or not args.models_root:
        parser.error("--all or --model with --manifest and --models-root is required")
    if not 1 <= args.timeout_seconds <= 900:
        parser.error("timeout must be between 1 and 900 seconds")
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    roots = {model: args.models_root / model for model in ("SmolLM", "CoEdIT", "GECToR")}
    verify_manifest(document, roots)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(args.timeout_seconds)
    try:
        handle, baseline = _nvml_baseline()
        models = ("SmolLM", "CoEdIT", "GECToR") if args.all else (args.model,)
        for model in models:
            _run_model(model, roots[model], baseline, handle)
        print(json.dumps({"status": "passed-candidate", "models": list(models)}, sort_keys=True))
        return 0
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Deadline, Exception) as error:
        raise SystemExit(f"compatibility spike failed closed: {error}")

"""Child-process model runtimes for the experimental RM compatibility harness.

This module is deliberately not an adapter package.  Its only supported caller
is :mod:`rm_spike`, and all imports which initialise CUDA happen in the child.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import socket
from typing import Any
try:
    from .input_bounds import (SMOLLM_CONTEXT, SMOLLM_OUTPUT_RESERVE,
                                SMOLLM_PROMPT_TOKEN_LIMIT, TRANSFORMER_BUCKET_TOKENS,
                                frame_smollm_prompt, validate_smollm_input)
except ImportError:  # copied into the flat candidate image
    from input_bounds import (SMOLLM_CONTEXT, SMOLLM_OUTPUT_RESERVE,
                              SMOLLM_PROMPT_TOKEN_LIMIT, TRANSFORMER_BUCKET_TOKENS,
                              frame_smollm_prompt, validate_smollm_input)


SMALL_FIXTURES = {
    "SmolLM": "Correct: This are valid.",
    "CoEdIT": "Fix grammatical errors: This are valid.",
    "GECToR": "This are valid.",
}
# These are candidate request buckets, not model maxima or measured capacity.
MAX_FIXTURES = {
    "SmolLM": ("Correct: " + "This are valid. " * 11).ljust(192),
    "CoEdIT": "Fix grammatical errors: " + "This are valid. " * 6,
    "GECToR": "This are valid. " * 6,
}


def smollm_source_identity(root: Path) -> str:
    """Return a stable identity for the selected local GGUF and Modelfile."""
    digest = hashlib.sha256()
    for path in sorted((p for p in root.iterdir() if p.is_file()), key=lambda p: p.name):
        if path.name == "Modelfile" or path.suffix.lower() in {".gguf", ".bin"}:
            digest.update(path.name.encode("utf-8"))
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def smollm_model_name(root: Path) -> str:
    return "compatibility-rm-smollm-" + smollm_source_identity(root)[:16]


def _terminate_group(process: subprocess.Popen[Any], timeout: float = 10.0) -> None:
    """Stop only the directly owned server; the parent owns group cleanup."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


class LoadedRuntime:
    def __init__(self, model: str, root: Path) -> None:
        self.model, self.root = model, root
        self.tokenizer = None
        self.model_object = None
        self.server = None
        self.active_request: str | None = None
        self.store: Path | None = None
        self.model_name: str | None = None
        self._store_temp: tempfile.TemporaryDirectory[str] | None = None
        self._gpu_uuid: str | None = None
        self._ollama_url: str | None = None
        self._ollama_env: dict[str, str] | None = None
        self.execution_started = None

    def snapshot(self) -> dict[str, object]:
        """Safe lifecycle evidence; deliberately excludes prompts and results."""
        return {"lifecycle": "loaded" if self.server or self.model_object is not None else "stopped",
                "ready": self.server is not None or self.model_object is not None,
                "server_pid": self.server.pid if self.server else None,
                "active": self.active_request is not None, "model": self.model,
                "source_identity": smollm_source_identity(self.root) if self.model == "SmolLM" else str(self.root),
                "model_name": self.model_name, "store_path": str(self.store) if self.store else None,
                "store_state": "allocated" if self.store else "none",
                "gpu_uuid": self._gpu_uuid}

    def _allocate_smollm_store(self) -> tuple[Path, str]:
        self._store_temp = tempfile.TemporaryDirectory(prefix="compatibility-ollama-")
        store = Path(self._store_temp.name)
        for source in self.root.iterdir():
            if source.name == "Modelfile" or source.suffix.lower() in {".gguf", ".bin"}:
                (store / source.name).write_bytes(source.read_bytes())
        return store, smollm_model_name(self.root)

    def load(self) -> None:
        if self.model == "SmolLM":
            self.store, self.model_name = self._allocate_smollm_store()
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            self._ollama_url = f"http://127.0.0.1:{port}"
            self._ollama_env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{port}",
                                "OLLAMA_MODELS": str(self.store)}
            self.server = subprocess.Popen(["ollama", "serve"], env=self._ollama_env,
                                            start_new_session=False,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
            import urllib.request
            for _ in range(80):
                if self.server.poll() is not None:
                    raise RuntimeError("private Ollama server exited before readiness")
                try:
                    urllib.request.urlopen(self._ollama_url + "/api/version", timeout=1).close()
                    break
                except Exception:
                    time.sleep(.25)
            else:
                raise RuntimeError("Ollama loopback server did not become ready")
            subprocess.run(["ollama", "create", self.model_name, "-f",
                            str(self.root / "Modelfile")], check=True, timeout=60, env=self._ollama_env,
                           stdout=sys.stderr, stderr=sys.stderr, cwd=self.root)
            # Import does not establish residency. Explicitly preload before
            # ready() checks the private Ollama server's GPU residency.
            body = json.dumps({"model": self.model_name, "prompt": "", "raw": True,
                                "stream": False, "keep_alive": -1,
                                "options": {"num_ctx": 512, "num_predict": SMOLLM_OUTPUT_RESERVE,
                                            "temperature": 0}}).encode()
            request = urllib.request.Request(self._ollama_url + "/api/generate",
                data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                if not json.loads(response.read()).get("done"):
                    raise RuntimeError("Ollama preload did not finish")
            return
        import torch
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.root, local_files_only=True)
        if self.model == "CoEdIT":
            from transformers import T5ForConditionalGeneration
            self.model_object = T5ForConditionalGeneration.from_pretrained(
                self.root, local_files_only=True, use_safetensors=True).to("cuda")
        else:
            from gector import GECToR, GECToRConfig
            import gector.modeling as modeling
            from transformers import AutoConfig, AutoModel
            config = GECToRConfig.from_pretrained(self.root, local_files_only=True)
            base_config = {"model_type": "deberta", "attention_probs_dropout_prob": 0.1,
                           "hidden_act": "gelu", "hidden_dropout_prob": 0.1,
                           "hidden_size": 1024, "initializer_range": 0.02,
                           "intermediate_size": 4096, "max_position_embeddings": 512,
                           "relative_attention": True, "pos_att_type": "c2p|p2c",
                           "layer_norm_eps": 1e-7, "max_relative_positions": -1,
                           "position_biased_input": False, "num_attention_heads": 16,
                           "num_hidden_layers": 24, "type_vocab_size": 0,
                           "vocab_size": 50265}
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                (base / "config.json").write_text(json.dumps(base_config), encoding="utf-8")
                config.model_id = str(base)
                old_model, old_tokenizer = modeling.AutoModel.from_pretrained, modeling.AutoTokenizer.from_pretrained
                modeling.AutoModel.from_pretrained = lambda _path, **_: AutoModel.from_config(AutoConfig.from_pretrained(base, local_files_only=True))
                modeling.AutoTokenizer.from_pretrained = lambda _path, **_: self.tokenizer
                try:
                    self.model_object, loading = GECToR.from_pretrained(
                        self.root, config=config, local_files_only=True, use_safetensors=True,
                        output_loading_info=True)
                    if loading["missing_keys"] or loading["unexpected_keys"] or loading["mismatched_keys"]:
                        raise RuntimeError("GECToR weight load is incomplete")
                finally:
                    modeling.AutoModel.from_pretrained, modeling.AutoTokenizer.from_pretrained = old_model, old_tokenizer
            self.model_object = self.model_object.to("cuda")

    def gpu_identity(self) -> dict[str, object]:
        """Prove CUDA/NVML identity for Torch; Ollama is checked through ps."""
        # NVML is the common identity source for Ollama and Torch, and it runs
        # only in this CUDA-owning child.
        from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetUUID, nvmlInit
        nvmlInit()
        raw_uuid = nvmlDeviceGetUUID(nvmlDeviceGetHandleByIndex(0))
        nvml_uuid = raw_uuid.decode() if isinstance(raw_uuid, bytes) else str(raw_uuid)
        self._gpu_uuid = nvml_uuid
        if self.model == "SmolLM":
            result = subprocess.run(["ollama", "ps"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, timeout=10, check=True, env=self._ollama_env)
            if self.model_name not in result.stdout or "GPU" not in result.stdout.upper():
                raise RuntimeError("Ollama ps does not prove SmolLM GPU residency")
            return {"gpu_uuid": nvml_uuid, "cuda_nvml_agree": True, "ollama_gpu_resident": True}
        # Reuse the existing child-side CUDA/NVML/smi agreement helper.  It is
        # intentionally never imported by the parent RM process.
        try:
            from spike import _gpu_check
        except ModuleNotFoundError:
            from tools.compatibility.spike import _gpu_check
        evidence = _gpu_check()
        return {"gpu_uuid": nvml_uuid,
                "cuda_nvml_agree": evidence["uuid"] == nvml_uuid}

    def validate(self, text: str, context_size: int | None, bucket: str | None) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("invalid bounded fixture")
        expected = SMALL_FIXTURES[self.model] if text == SMALL_FIXTURES[self.model] else MAX_FIXTURES[self.model]
        if text != expected:
            raise ValueError("payload is not an exact harness fixture")
        if self.model == "SmolLM":
            if context_size != SMOLLM_CONTEXT or bucket is not None:
                raise ValueError("SmolLM requires the configured 512-token context")
            validate_smollm_input(self.root, text)
        else:
            if context_size is not None or bucket != "upper-fixture":
                raise ValueError("Transformers fixtures require upper-fixture bucket")
            encoded = self.tokenizer(text, truncation=False, add_special_tokens=True)
            if len(encoded["input_ids"]) > TRANSFORMER_BUCKET_TOKENS:
                raise ValueError("fixture exceeds configured candidate bucket")

    def execute(self, text: str, request_id: str | None = None) -> str:
        import torch
        self.active_request = request_id or "active"
        if self.execution_started is not None:
            self.execution_started(self.active_request)
        try:
            return self._execute(text, torch)
        finally:
            self.active_request = None

    def _execute(self, text: str, torch: Any) -> str:
        if self.model == "SmolLM":
            from urllib.request import Request, urlopen
            body = json.dumps({"model": self.model_name, "prompt": frame_smollm_prompt(text),
                                "raw": True, "stream": False, "keep_alive": -1,
                                "options": {"num_ctx": SMOLLM_CONTEXT, "num_predict": SMOLLM_OUTPUT_RESERVE,
                                            "temperature": 0}}).encode()
            with urlopen(Request(self._ollama_url + "/api/generate", data=body,
                                headers={"Content-Type": "application/json"}), timeout=240) as response:
                result = json.loads(response.read())
            answer = result.get("response")
            if result.get("done") is not True:
                raise RuntimeError("SmolLM response did not finish")
            if (not isinstance(result.get("prompt_eval_count"), int)
                    or result["prompt_eval_count"] > SMOLLM_PROMPT_TOKEN_LIMIT):
                raise RuntimeError("SmolLM context/truncation evidence is missing or exceeds configured context")
        elif self.model == "CoEdIT":
            with torch.inference_mode():
                encoded = self.tokenizer(text, return_tensors="pt").to("cuda")
                output = self.model_object.generate(**encoded, max_new_tokens=32)
            answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            from gector import load_verb_dict, predict
            encode, decode = load_verb_dict(str(self.root / "verb-form-vocab.txt"))
            with torch.inference_mode():
                result = predict(self.model_object, self.tokenizer, [text], encode, decode,
                                 keep_confidence=0.0, min_error_prob=0.0, n_iteration=1, batch_size=1)
            answer = result[0] if result else ""
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(f"{self.model} returned an empty response")
        return answer

    def cancel(self, request_id: str) -> bool:
        # In-process Transformers/GECToR generation has no safe adapter-level
        # interruption hook in this experiment. RM fencing remains authoritative.
        return self.active_request is not None

    def verify_cleanup(self) -> bool:
        return self.model_object is None and self.tokenizer is None and self.server is None

    def close(self) -> None:
        if self.server is not None:
            try:
                subprocess.run(["ollama", "rm", self.model_name or smollm_model_name(self.root)], timeout=20,
                               check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               env=self._ollama_env)
                _terminate_group(self.server)
            finally:
                self.server = None
        self.model_object = None
        self.tokenizer = None
        self._ollama_env = None
        # The RPC child must not delete its store until the parent has proved
        # that the entire owning process group is gone.

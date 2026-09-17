"""Fail-closed printable-ASCII bound proof for a selected SmolLM GGUF."""
from __future__ import annotations
from functools import lru_cache
import hashlib, struct
from pathlib import Path

SMOLLM_CONTEXT = 512
SMOLLM_OUTPUT_RESERVE = 64
SMOLLM_PROMPT_TOKEN_LIMIT = 448
SMOLLM_MAX_RAW_BYTES = 256
SMOLLM_FRAME_PREFIX = "<|im_start|>user\n"
SMOLLM_FRAME_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
TRANSFORMER_BUCKET_TOKENS = 128
MAX_METADATA_BYTES = 32 * 1024 * 1024

def frame_smollm_prompt(text: str) -> str:
    return SMOLLM_FRAME_PREFIX + text + SMOLLM_FRAME_SUFFIX

def _exact(stream, size: int) -> bytes:
    if size < 0 or stream.tell() + size > MAX_METADATA_BYTES:
        raise ValueError("GGUF metadata exceeds aggregate byte budget")
    data = stream.read(size)
    if len(data) != size: raise ValueError("truncated GGUF metadata")
    return data
def _u32(s): return struct.unpack("<I", _exact(s, 4))[0]
def _u64(s): return struct.unpack("<Q", _exact(s, 8))[0]
def _string(s):
    n = _u64(s)
    if n > 4096: raise ValueError("GGUF metadata string is unexpectedly large")
    try: return _exact(s, n).decode()
    except UnicodeDecodeError as e: raise ValueError("GGUF metadata string is not UTF-8") from e

_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_keys = {"tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens", "tokenizer.ggml.add_bos_token", "llama.context_length"}
def _skip(s, kind, symbols=None):
    if kind == 8: return _string(s)
    if kind == 9:
        element, n = _u32(s), _u64(s)
        if n > 10_000_000: raise ValueError("GGUF metadata array is unexpectedly large")
        if element == 8:
            for _ in range(n):
                value = _string(s)
                if symbols is not None and len(value) == 1: symbols.add(value)
            return n
        if element not in _sizes: raise ValueError("unsupported GGUF metadata array type")
        _exact(s, _sizes[element] * n); return n
    if kind not in _sizes: raise ValueError(f"unsupported GGUF metadata type {kind}")
    raw = _exact(s, _sizes[kind])
    if kind == 7:
        if raw not in (b"\0", b"\1"): raise ValueError("invalid GGUF boolean")
        return raw == b"\1"
    if kind == 4: return struct.unpack("<I", raw)[0]
    if kind in (10, 11, 12): return struct.unpack("<Q", raw)[0]
    return None
def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()
@lru_cache(maxsize=4)
def _read(path_text, size, mtime):
    values, symbols = {}, set()
    with Path(path_text).open("rb") as s:
        if _exact(s, 4) != b"GGUF": raise ValueError("selected SmolLM input is not GGUF")
        version, _, count = struct.unpack("<IQQ", _exact(s, 20))
        if version not in (2, 3) or count > 100_000: raise ValueError("unsupported GGUF header")
        for _ in range(count):
            key, kind = _string(s), _u32(s)
            if key in values and key in _keys: raise ValueError(f"duplicate GGUF proof key: {key}")
            value = _skip(s, kind, symbols if key == "tokenizer.ggml.tokens" else None)
            if key in _keys: values[key] = value
    if values.get("tokenizer.ggml.model") != "gpt2" or values.get("tokenizer.ggml.pre") != "smollm": raise ValueError("GGUF tokenizer metadata does not prove the selected SmolLM byte-BPE tokenizer")
    if not isinstance(values.get("tokenizer.ggml.tokens"), int) or not {"Ġ", "Ċ"} | {chr(x) for x in range(0x21, 0x7f)} <= symbols: raise ValueError("GGUF vocabulary does not prove ASCII byte coverage")
    if values.get("tokenizer.ggml.add_bos_token") is not False: raise ValueError("GGUF metadata does not prove disabled implicit BOS handling")
    if not isinstance(values.get("llama.context_length"), int) or values["llama.context_length"] < 512: raise ValueError("GGUF context metadata does not prove configured context")
    return {"path": Path(path_text).name, "sha256": _sha(Path(path_text)), "model": "gpt2", "pre": "smollm", "vocabulary_size": values["tokenizer.ggml.tokens"], "implicit_bos_tokens": 0, "context_length": values["llama.context_length"]}
def read_smollm_tokenizer_metadata(root):
    files = sorted(Path(root).glob("*.gguf"))
    if len(files) != 1: raise ValueError("SmolLM requires exactly one selected GGUF for input evidence")
    p = files[0].resolve(); st = p.stat(); return _read(str(p), st.st_size, st.st_mtime_ns)
def smollm_evidence(root):
    metadata = read_smollm_tokenizer_metadata(root)
    return {"algorithm": "streamed-byte-BPE-upper-bound+explicit-raw-framing", "metadata": metadata, "max_raw_bytes": 256, "prompt_token_limit": 448, "output_token_reserve": 64}
def validate_smollm_input(root, text):
    if not isinstance(text, str): raise ValueError("SmolLM input must be text")
    evidence = smollm_evidence(root); raw = text.encode("utf-8")
    if len(raw) > 256: raise ValueError("SmolLM input exceeds configured raw-byte bucket")
    if any(ord(c) < 0x20 or ord(c) > 0x7e for c in text): raise ValueError("SmolLM input must use the proved printable-ASCII bucket")
    if len(frame_smollm_prompt(text).encode()) > 448: raise ValueError("SmolLM input exceeds proved no-truncation token bound")
    return evidence


def validate_transformer_input(payload, *, max_bytes=SMOLLM_MAX_RAW_BYTES):
    """Legacy compatibility validator retained for the candidate tooling.

    Transformer adapters have their own token proof; this helper only preserves
    the historical finite byte/text boundary and deliberately performs no
    truncation or tokenisation.
    """
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if not isinstance(payload, bytes) or len(payload) > max_bytes:
        raise ValueError("transformer input exceeds configured byte bound")
    return payload

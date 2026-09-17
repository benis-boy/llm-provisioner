"""Fail-closed, conservative SmolLM input-bound evidence.

This is intentionally not a tokenizer.  It streams just enough GGUF metadata
to establish the byte-BPE upper bound for the selected artifact.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import struct
from pathlib import Path

SMOLLM_CONTEXT = 512
SMOLLM_OUTPUT_RESERVE = 64
SMOLLM_PROMPT_TOKEN_LIMIT = SMOLLM_CONTEXT - SMOLLM_OUTPUT_RESERVE
# This is a harness request bucket, deliberately smaller than the proved limit.
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
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("truncated GGUF metadata")
    return value


def _u32(stream) -> int:
    return struct.unpack("<I", _exact(stream, 4))[0]


def _u64(stream) -> int:
    return struct.unpack("<Q", _exact(stream, 8))[0]


def _read_string(stream) -> str:
    size = _u64(stream)
    if size > 4096:
        raise ValueError("GGUF metadata string is unexpectedly large")
    try:
        return _exact(stream, size).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("GGUF metadata string is not UTF-8") from error


def _proved_byte_symbols() -> set[str]:
    # The actual selected vocabulary supplies these printable ASCII singleton
    # tokens. The harness admits ASCII only: absent proof of all non-ASCII byte
    # fallbacks is not silently generalized into a UTF-8 claim.
    # GPT-2 maps bytes 0x20 and 0x0a to U+0120/U+010a respectively.
    return {"Ġ", "Ċ"} | {chr(value) for value in range(0x21, 0x7F)}


_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_PROOF_KEYS = {"tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens", "tokenizer.ggml.add_bos_token", "llama.context_length"}


def _skip_value(stream, kind: int, *, token_symbols: set[str] | None = None) -> object | None:
    if kind == 8:
        return _read_string(stream)
    if kind == 9:
        element_kind, length = _u32(stream), _u64(stream)
        if length > 10_000_000:
            raise ValueError("GGUF metadata array is unexpectedly large")
        if element_kind == 8:
            for _ in range(length):
                token = _read_string(stream)
                if token_symbols is not None and len(token) == 1:
                    token_symbols.add(token)
            return length
        size = _SCALAR_SIZES.get(element_kind)
        if size is None:
            raise ValueError("unsupported GGUF metadata array type")
        _exact(stream, size * length)
        return length
    size = _SCALAR_SIZES.get(kind)
    if size is None:
        raise ValueError(f"unsupported GGUF metadata type {kind}")
    raw = _exact(stream, size)
    if kind == 7:
        if raw not in (b"\0", b"\1"):
            raise ValueError("invalid GGUF boolean")
        return raw == b"\1"
    if kind == 4:
        return struct.unpack("<I", raw)[0]
    if kind in (10, 11, 12):
        return struct.unpack("<Q", raw)[0]
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _read_selected(path_text: str, size: int, mtime_ns: int) -> dict[str, object]:
    path = Path(path_text)
    symbols: set[str] = set()
    values: dict[str, object] = {}
    with path.open("rb") as stream:
        if _exact(stream, 4) != b"GGUF":
            raise ValueError("selected SmolLM input is not GGUF")
        version, _, count = struct.unpack("<IQQ", _exact(stream, 20))
        if version not in (2, 3) or count > 100_000:
            raise ValueError("unsupported GGUF header")
        for _ in range(count):
            key, kind = _read_string(stream), _u32(stream)
            if key in values and key in _PROOF_KEYS:
                raise ValueError(f"duplicate GGUF proof key: {key}")
            value = _skip_value(stream, kind, token_symbols=symbols if key == "tokenizer.ggml.tokens" else None)
            if key in _PROOF_KEYS:
                values[key] = value
    if values.get("tokenizer.ggml.model") != "gpt2" or values.get("tokenizer.ggml.pre") != "smollm":
        raise ValueError("GGUF tokenizer metadata does not prove the selected SmolLM byte-BPE tokenizer")
    if not isinstance(values.get("tokenizer.ggml.tokens"), int) or values["tokenizer.ggml.tokens"] <= 0:
        raise ValueError("GGUF tokenizer vocabulary metadata is missing")
    if not _proved_byte_symbols().issubset(symbols):
        raise ValueError("GGUF vocabulary does not prove ASCII byte coverage")
    if values.get("tokenizer.ggml.add_bos_token") is not False:
        raise ValueError("GGUF metadata does not prove disabled implicit BOS handling")
    if not isinstance(values.get("llama.context_length"), int) or values["llama.context_length"] < SMOLLM_CONTEXT:
        raise ValueError("GGUF context metadata does not prove configured context")
    return {"path": path.name, "sha256": _sha256(path), "model": values["tokenizer.ggml.model"],
            "pre": values["tokenizer.ggml.pre"], "vocabulary_size": values["tokenizer.ggml.tokens"],
            "implicit_bos_tokens": 0, "context_length": values["llama.context_length"]}


def read_smollm_tokenizer_metadata(root: Path) -> dict[str, object]:
    files = sorted(root.glob("*.gguf"))
    if len(files) != 1:
        raise ValueError("SmolLM requires exactly one selected GGUF for input evidence")
    path = files[0].resolve()
    stat = path.stat()
    return _read_selected(str(path), stat.st_size, stat.st_mtime_ns)


def smollm_evidence(root: Path) -> dict[str, object]:
    metadata = read_smollm_tokenizer_metadata(root)
    return {"algorithm": "streamed-byte-BPE-upper-bound+explicit-raw-framing",
            "metadata": metadata, "max_raw_bytes": SMOLLM_MAX_RAW_BYTES,
            "prompt_token_limit": SMOLLM_PROMPT_TOKEN_LIMIT, "output_token_reserve": SMOLLM_OUTPUT_RESERVE}


def validate_smollm_input(root: Path, text: str) -> dict[str, object]:
    if not isinstance(text, str):
        raise ValueError("SmolLM input must be text")
    evidence = smollm_evidence(root)
    raw = text.encode("utf-8")
    if len(raw) > SMOLLM_MAX_RAW_BYTES:
        raise ValueError("SmolLM input exceeds configured raw-byte bucket")
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in text):
        raise ValueError("SmolLM input must use the proved printable-ASCII bucket")
    # Every admitted byte has a one-token fallback; BPE merges can only reduce
    # the count. Raw mode adds no template, and metadata proves no implicit BOS.
    upper_tokens = len(frame_smollm_prompt(text).encode("utf-8")) + int(evidence["metadata"]["implicit_bos_tokens"])
    if upper_tokens > int(evidence["prompt_token_limit"]):
        raise ValueError("SmolLM input exceeds proved no-truncation token bound")
    return evidence

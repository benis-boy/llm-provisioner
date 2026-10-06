"""Canonical, typed production measurement selectors and request contracts."""
from __future__ import annotations
from services.llm.queue.contracts import ModelId
from services.llm.provisioning.contracts import ModelConfig, CapacityBucket, GenerationConfig
from services.llm.provisioning.benchmark_requests import configured_request_buckets

MATRIX = {
 ModelId.SMOLLM: ModelConfig(ModelId.SMOLLM, "models/SmolLM", "SmolLM2-1.7B-Instruct-Q8_0.gguf", "ollama", (512,)),
 ModelId.COEDIT: ModelConfig(ModelId.COEDIT, "models/CoEdIT", "model.safetensors", "transformers-coedit", buckets=(CapacityBucket(128,64,GenerationConfig({"num_beams":1,"do_sample":False},"float16"),(1,)),)),
 ModelId.GECTOR: ModelConfig(ModelId.GECTOR, "models/GECToR", "model.safetensors", "gector", buckets=(CapacityBucket(128,1,GenerationConfig({"keep_confidence":0,"min_error_prob":0},"float32"),(1,),1,(0.,0.)),)),
}

def measurement_matrix() -> tuple[tuple[ModelId, str], ...]:
    return tuple((model, selector) for model in ModelId for selector in configured_request_buckets(MATRIX[model]))

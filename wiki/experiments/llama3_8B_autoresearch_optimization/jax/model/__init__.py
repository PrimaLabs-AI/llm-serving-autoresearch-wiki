"""Llama 3 8B native-JAX (Flax NNX) model package."""
from .modeling_llama3 import (
    LlamaForCausalLM,
    LlamaForCausalLMScan,
    LlamaModel,
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaMLP,
    LlamaRotaryEmbedding,
    LlamaEmbedding,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    set_splash_mesh,
)
from .weight_loader import load_hf_weights
from .sharding import (
    AXIS_FSDP, AXIS_TP,
    SHARDING_PLAN, SCAN_SHARDING_PLAN,
    ShardingPlan, build_plan, apply_sharding,
    get_mesh, get_fsdp_mesh, get_tp_mesh,
    input_sharding, replicated,
)

__all__ = [
    "LlamaForCausalLM",
    "LlamaForCausalLMScan",
    "LlamaModel",
    "LlamaDecoderLayer",
    "LlamaAttention",
    "LlamaMLP",
    "LlamaRotaryEmbedding",
    "LlamaEmbedding",
    "LlamaRMSNorm",
    "apply_rotary_pos_emb",
    "set_splash_mesh",
    "load_hf_weights",
    "AXIS_FSDP", "AXIS_TP",
    "SHARDING_PLAN", "SCAN_SHARDING_PLAN",
    "ShardingPlan", "build_plan", "apply_sharding",
    "get_mesh", "get_fsdp_mesh", "get_tp_mesh",
    "input_sharding", "replicated",
]

"""Gemma 4 native-JAX (Flax NNX) model package."""
from .modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4TextModel,
    Gemma4TextDecoderLayer,
    Gemma4TextAttention,
    Gemma4TextMLP,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
    Gemma4RMSNorm,
    apply_rotary_pos_emb,
)
from .weight_loader import load_hf_weights
from .sharding import (
    get_mesh, get_fsdp_mesh, get_tp_mesh,
    get_param_sharding, apply_sharding,
    plan_fsdp_shardings, plan_tp_shardings,
    input_sharding, logits_sharding, replicated,
    ShardingPlan,
    AXIS_FSDP, AXIS_DP, AXIS_TP,
)

__all__ = [
    "Gemma4ForCausalLM",
    "Gemma4TextModel",
    "Gemma4TextDecoderLayer",
    "Gemma4TextAttention",
    "Gemma4TextMLP",
    "Gemma4TextRotaryEmbedding",
    "Gemma4TextScaledWordEmbedding",
    "Gemma4RMSNorm",
    "apply_rotary_pos_emb",
    "load_hf_weights",
    "get_mesh", "get_fsdp_mesh", "get_tp_mesh",
    "get_param_sharding", "apply_sharding",
    "plan_fsdp_shardings", "plan_tp_shardings",
    "input_sharding", "logits_sharding", "replicated",
    "ShardingPlan",
    "AXIS_FSDP", "AXIS_DP", "AXIS_TP",
]

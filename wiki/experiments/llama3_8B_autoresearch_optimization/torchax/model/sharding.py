"""Sharding plan for HuggingFace Llama-3-8B on TPU under torchax + JAX.

Wildcard-by-layer-index map keyed on `_process_sharding_name(name)` so all 32
transformer layers share the same spec. The patterns ('fsdp', 'tp') etc. resolve
to `NamedSharding(mesh, P(...))` against the trainer's mesh
(typically `Mesh((fsdp, tp), ('fsdp', 'tp'))`).

For TP=1 we get pure FSDP (the default on v6e-4 / v6e-8). For TP>1 the same map
also encodes Megatron-style tensor parallelism on attention/FFN projections.

HF Llama parameter naming (different from torchtitan):
- `model.embed_tokens.weight`  → torchtitan `tok_embeddings.weight`
- `model.layers.{i}.self_attn.q_proj.weight`  → torchtitan `layers.{i}.attention.wq.weight`
- `model.layers.{i}.self_attn.k_proj.weight`  → torchtitan `layers.{i}.attention.wk.weight`
- `model.layers.{i}.self_attn.v_proj.weight`  → torchtitan `layers.{i}.attention.wv.weight`
- `model.layers.{i}.self_attn.o_proj.weight`  → torchtitan `layers.{i}.attention.wo.weight`
- `model.layers.{i}.mlp.gate_proj.weight`     → torchtitan `layers.{i}.feed_forward.w1.weight`
- `model.layers.{i}.mlp.up_proj.weight`       → torchtitan `layers.{i}.feed_forward.w3.weight`
- `model.layers.{i}.mlp.down_proj.weight`     → torchtitan `layers.{i}.feed_forward.w2.weight`
- `model.layers.{i}.input_layernorm.weight`   → torchtitan `layers.{i}.attention_norm.weight`
- `model.layers.{i}.post_attention_layernorm.weight` → torchtitan `layers.{i}.ffn_norm.weight`
- `model.norm.weight`  → torchtitan `norm.weight`
- `lm_head.weight`     → torchtitan `output.weight`
"""

import jax
from jax.sharding import NamedSharding, PartitionSpec as P


# FSDP-only spec. For each row, the tuple is the PartitionSpec across mesh axes
# in the canonical mesh order ('fsdp', 'tp'). Empty tuple () = replicated.
SHARDING_MAP = {
    "model.embed_tokens.weight": ("fsdp", "tp"),  # (vocab, hidden)
    # Attention QKV/O.
    "model.layers.*.self_attn.q_proj.weight": ("tp", "fsdp"),  # (num_heads*head_dim, hidden)
    "model.layers.*.self_attn.k_proj.weight": ("tp", "fsdp"),  # (num_kv_heads*head_dim, hidden)
    "model.layers.*.self_attn.v_proj.weight": ("tp", "fsdp"),  # (num_kv_heads*head_dim, hidden)
    "model.layers.*.self_attn.o_proj.weight": ("fsdp", "tp"),  # (hidden, num_heads*head_dim)
    # FFN (SwiGLU-style: gate, up, down).
    "model.layers.*.mlp.gate_proj.weight": ("tp", "fsdp"),  # (intermediate, hidden)
    "model.layers.*.mlp.up_proj.weight":   ("tp", "fsdp"),
    "model.layers.*.mlp.down_proj.weight": ("fsdp", "tp"),
    # RMSNorm (1-D).
    "model.layers.*.input_layernorm.weight":          ("fsdp",),
    "model.layers.*.post_attention_layernorm.weight": ("fsdp",),
    "model.norm.weight": ("fsdp",),
    # LM head.
    "lm_head.weight": ("tp", "fsdp"),
}


def _process_sharding_name(name: str) -> str:
    """Replace integer tokens (layer indices) with `*` for wildcard matching."""
    def is_int(t):
        try:
            int(t)
            return True
        except ValueError:
            return False
    return ".".join("*" if is_int(t) else t for t in name.split("."))


def get_sharding(mesh, param_name: str) -> NamedSharding | None:
    """Return the canonical sharding for a HF Llama parameter, or None if unknown."""
    spec = SHARDING_MAP.get(_process_sharding_name(param_name))
    if spec is None:
        return None
    return NamedSharding(mesh, P(*spec))

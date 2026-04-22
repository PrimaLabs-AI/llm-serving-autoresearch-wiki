"""Mesh + per-parameter sharding rules for Gemma 4 on TPU.

Adapted from the jax-huggingface Part 2 NeMo-Megatron tensor-parallel recipe
(see `raw/code/learning-machine/jax-huggingface/jax_hg_02.py`), generalized
for a 2D (dp, tp) mesh and extended to GQA (Gemma 4's `num_kv_heads=2` does
not divide a typical TP size of 8 — we replicate K/V by default).

Not exercised — see the UNTESTED warning in `train.py`.

Shardings returned are `jax.sharding.NamedSharding` objects built on the
mesh returned by `get_mesh(dp, tp)`. The matcher keys off substrings in the
parameter's fully-qualified name (`model.layers.0.self_attn.q_proj.weight`,
etc.) — the same strategy Han Qi used in jax-huggingface Part 2. This is
fragile to upstream rename; re-check on every transformers bump.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# -----------------------------------------------------------------------------
# Mesh construction
# -----------------------------------------------------------------------------

AXIS_DP = "dp"
AXIS_TP = "tp"


def get_mesh(dp: int = 1, tp: Optional[int] = None) -> Mesh:
    """Build a (dp, tp) 2D device mesh.

    Defaults `tp` to `jax.device_count()` so a v6e-8 host runs as TP=8, DP=1.
    If `dp * tp != jax.device_count()` we raise — refuse to silently drop
    chips.
    """
    if tp is None:
        tp = jax.device_count() // max(dp, 1)
    total = dp * tp
    avail = jax.device_count()
    if total != avail:
        raise ValueError(
            f"Mesh (dp={dp}, tp={tp}) needs {total} devices, "
            f"but jax.device_count()={avail}."
        )
    devices = mesh_utils.create_device_mesh((dp, tp))
    return Mesh(devices, axis_names=(AXIS_DP, AXIS_TP))


# -----------------------------------------------------------------------------
# Parameter sharding rules
# -----------------------------------------------------------------------------

# The Gemma-family HF param-name convention (as of transformers main, 2026-04):
#
#   model.embed_tokens.weight                                       (vocab, hidden)
#   model.layers.<i>.self_attn.q_proj.weight                        (num_heads*head_dim, hidden)
#   model.layers.<i>.self_attn.k_proj.weight                        (num_kv_heads*head_dim, hidden)
#   model.layers.<i>.self_attn.v_proj.weight                        (num_kv_heads*head_dim, hidden)
#   model.layers.<i>.self_attn.o_proj.weight                        (hidden, num_heads*head_dim)
#   model.layers.<i>.mlp.gate_proj.weight                           (intermediate, hidden)
#   model.layers.<i>.mlp.up_proj.weight                             (intermediate, hidden)
#   model.layers.<i>.mlp.down_proj.weight                           (hidden, intermediate)
#   model.layers.<i>.input_layernorm.weight                         (hidden,)
#   model.layers.<i>.post_attention_layernorm.weight                (hidden,)
#   model.norm.weight                                               (hidden,)
#   lm_head.weight                                                  (vocab, hidden)  # often tied with embed_tokens
#
# Gemma 3 also carried `pre_feedforward_layernorm`, `post_feedforward_layernorm`
# and rotary `inv_freq` buffers; Gemma 4 likely keeps those. Unknown names
# default to replicated (conservative).


# HF matmul weights are stored with shape (out, in). So column-sharding is
# sharding on the **first** (row) axis of the weight, and row-sharding is
# sharding on the **second** (column) axis. This mirrors jax-huggingface
# Part 2's recipe.
_COL_SHARD_SUBSTR = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
_ROW_SHARD_SUBSTR = ("o_proj", "down_proj")
_ROW_SHARD_EMBED = ("embed_tokens", "lm_head")


@dataclass
class ShardingPlan:
    """What to shard and how. Returned by `plan_shardings`."""

    shardings: Dict[str, NamedSharding]
    # For reporting: bucket name -> list of param names that fell in it.
    buckets: Dict[str, list]
    notes: list  # strings, flags for the report / logging


def _gqa_can_shard_kv(num_kv_heads: int, tp: int) -> bool:
    """Can we shard K/V heads over `tp` without replication?

    Requires tp to divide num_kv_heads. For Gemma 4 E4B (kv=2, tp=8) this is
    False: we would need a broadcasted GQA sharding, not the basic row/column
    pattern. Future hypothesis.
    """
    return num_kv_heads % tp == 0 and num_kv_heads >= tp


def plan_shardings(
    param_names: Iterable[str],
    mesh: Mesh,
    *,
    num_kv_heads: int,
    num_attention_heads: int,
) -> ShardingPlan:
    """Decide `PartitionSpec` for every parameter name.

    `param_names` is what `model.state_dict().keys()` yields after the model
    is loaded. We don't need the shapes — the column/row convention is
    positional.
    """
    tp = mesh.shape[AXIS_TP]
    dp = mesh.shape[AXIS_DP]  # unused for parameter sharding; reserved for activations

    notes = []
    if num_attention_heads % tp != 0:
        notes.append(
            f"WARNING: num_attention_heads={num_attention_heads} not divisible by tp={tp}; "
            "Q/O sharding will be wrong. Consider a different tp."
        )
    kv_head_shardable = _gqa_can_shard_kv(num_kv_heads, tp)
    if not kv_head_shardable:
        notes.append(
            f"GQA: num_kv_heads={num_kv_heads} does not divide tp={tp}. "
            "K/V projections will be REPLICATED over the tp axis (correct but "
            "suboptimal). Future hypothesis: broadcast KV groups across the "
            "Q-head partitions."
        )

    # A bookkeeping bucket for the final report.
    buckets: Dict[str, list] = {
        "col_shard": [],
        "row_shard": [],
        "row_shard_embed": [],
        "kv_replicated": [],
        "replicated": [],
    }
    shardings: Dict[str, NamedSharding] = {}

    def _assign(name: str, spec: P, bucket: str) -> None:
        shardings[name] = NamedSharding(mesh, spec)
        buckets[bucket].append(name)

    for name in param_names:
        # K/V projections: shard over tp if num_kv_heads allows; else replicate.
        if any(s in name for s in ("k_proj", "v_proj")):
            if kv_head_shardable:
                _assign(name, P(AXIS_TP, None), "col_shard")
            else:
                _assign(name, P(None, None), "kv_replicated")
            continue

        # Q + MLP input projections: columnwise (shard the output "row" dim).
        if any(s in name for s in ("q_proj", "gate_proj", "up_proj")):
            _assign(name, P(AXIS_TP, None), "col_shard")
            continue

        # Attention O + MLP down projections: rowwise (shard the input "col" dim).
        if any(s in name for s in _ROW_SHARD_SUBSTR):
            _assign(name, P(None, AXIS_TP), "row_shard")
            continue

        # Embedding / LM head: shard the vocab (row) dim over tp. Vocab is
        # 262144 on Gemma 4 — divides cleanly by tp<=8.
        if any(s in name for s in _ROW_SHARD_EMBED):
            _assign(name, P(AXIS_TP, None), "row_shard_embed")
            continue

        # Everything else (norms, rotary, biases, small buffers): replicate.
        _assign(name, P(None), "replicated")

    return ShardingPlan(shardings=shardings, buckets=buckets, notes=notes)


# -----------------------------------------------------------------------------
# Convenience: take a state dict (param-name -> tensor) and apply shardings
# -----------------------------------------------------------------------------

def get_param_sharding(
    model: Any,  # torch.nn.Module or HF config + named_parameters iterable
    mesh: Mesh,
) -> ShardingPlan:
    """Build the plan off a live model's state_dict keys and its config.

    Lightly-duck-typed: we just need `model.config.num_key_value_heads` /
    `num_attention_heads` and the state_dict keys. Works on
    `Gemma4ForCausalLM` and on `Gemma4ForConditionalGeneration` (which
    exposes `model.config.text_config` — we fall back to that).
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Model has no .config attribute; cannot infer head counts.")
    text_cfg = getattr(cfg, "text_config", cfg)
    num_kv = getattr(text_cfg, "num_key_value_heads", None)
    num_q = getattr(text_cfg, "num_attention_heads", None)
    if num_kv is None or num_q is None:
        raise ValueError(
            "Model config missing num_key_value_heads / num_attention_heads. "
            f"Got: text_cfg={text_cfg!r}"
        )
    names = list(model.state_dict().keys())
    return plan_shardings(
        names, mesh, num_kv_heads=num_kv, num_attention_heads=num_q
    )


# -----------------------------------------------------------------------------
# Activation-sharding helpers — called inside the compiled train step.
# -----------------------------------------------------------------------------

def input_sharding(mesh: Mesh) -> NamedSharding:
    """input_ids / labels sharding: DP across batch, replicated within TP."""
    return NamedSharding(mesh, P(AXIS_DP, None))


def logits_sharding(mesh: Mesh) -> NamedSharding:
    """logits sharding after lm_head: DP on batch, TP on vocab."""
    return NamedSharding(mesh, P(AXIS_DP, None, AXIS_TP))


def replicated(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


__all__ = [
    "AXIS_DP",
    "AXIS_TP",
    "ShardingPlan",
    "get_mesh",
    "get_param_sharding",
    "plan_shardings",
    "input_sharding",
    "logits_sharding",
    "replicated",
]

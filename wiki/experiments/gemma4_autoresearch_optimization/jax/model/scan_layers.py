"""Scan-over-layers for Gemma 4 E4B native-JAX port (exp 49).

Replaces the 42-iteration Python for-loop in ``Gemma4TextModel.__call__``
with two nested ``jax.lax.scan``s: outer over 7 super-blocks, inner over
the 5 sliding-attention layers inside each super-block, plus one
full-attention layer applied explicitly per super-block.

Why this structure. Gemma 4 E4B's 42 layers are interleaved:
``[S S S S S F] x 7`` where S = sliding_attention (head_dim=256) and
F = full_attention (head_dim=512). The two attention variants have
**different projection shapes**, so a single flat scan over all 42
layers is not possible (stacked weight trees must be homogeneous).
Splitting into two groups preserves the interleave: each super-block
runs 5 sliding layers then 1 full layer.

KV sharing. The last 18 layers (indices 24..41) are
``is_kv_shared_layer=True`` and borrow K/V from the last non-shared
layer of the matching type. That is:
  - shared sliding layers (24-28, 30-34, 36-40) borrow from layer 22.
  - shared full layers (29, 35, 41) borrow from layer 23.
Handled here by:
  - Zero-stubs for the missing ``k_proj`` / ``v_proj`` / ``k_norm``
    weights on shared layers (the stack of weights is homogeneous).
  - A carry containing ``(k_shared_sliding, v_shared_sliding)`` and
    ``(k_shared_full, v_shared_full)`` tensors. The store-layer writes
    (gated on a traced ``is_store`` flag computed from the super-block
    and inner counters), the borrow-layer reads (gated on
    ``is_kv_shared``). Shared layers still do the wasted matmul with
    zero weights, but ``jnp.where`` selects the borrowed K/V, so
    semantics are preserved.

Env gate: set ``JAX_SCAN_LAYERS=1`` to select this path.
Default off — exp 36 (Python for-loop) remains the baseline.

Compile-time target: drop step-0 compile from ~180 s → ≤ 30 s on v6e-4.
Steady-state TPS: expected within ±0.5 % of exp 36 (shared activation
buffers may give 2-5 %, or small overhead from extra matmuls on shared
layers may take a little back).
"""
from __future__ import annotations

import os
from typing import Optional

import jax
import jax.numpy as jnp

from .modeling_gemma4 import (
    Gemma4RMSNorm,
    apply_rotary_pos_emb,
    _gelu_pytorch_tanh,
    _attn_xla_sdpa,
    _repeat_kv,
)


# ---------------------------------------------------------------------------
# Env gate
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    return os.environ.get("JAX_SCAN_LAYERS") == "1"


# ---------------------------------------------------------------------------
# Gemma 4 E4B structural constants (computed once from config)
# ---------------------------------------------------------------------------


def _partition_layers(layer_types: list[str]) -> tuple[int, int, list[int], list[int]]:
    """Validate the E4B pattern and return (num_super_blocks, block_len,
    sliding_indices_per_block, full_indices_per_block)."""
    assert len(layer_types) == 42, f"expected 42 layers, got {len(layer_types)}"
    # Pattern: [S S S S S F] x 7
    n_blocks = 7
    block_len = 6
    sliding_per_block = 5
    for b in range(n_blocks):
        for i in range(sliding_per_block):
            idx = b * block_len + i
            assert layer_types[idx] == "sliding_attention", (
                f"layer {idx} expected sliding, got {layer_types[idx]}"
            )
        full_idx = b * block_len + sliding_per_block
        assert layer_types[full_idx] == "full_attention", (
            f"layer {full_idx} expected full, got {layer_types[full_idx]}"
        )
    return n_blocks, block_len, sliding_per_block


# ---------------------------------------------------------------------------
# Weight extraction — lift NNX Param values into plain arrays
# ---------------------------------------------------------------------------


def _layer_weights(layer) -> dict:
    """Extract all trainable arrays from a Gemma4TextDecoderLayer into a flat
    dict-of-arrays.  For shared-KV layers, returns zero-stubs for k_proj,
    v_proj, k_norm so the stack is homogeneous."""
    out = {}
    out["input_layernorm_w"] = layer.input_layernorm.weight[...]
    out["post_attention_layernorm_w"] = layer.post_attention_layernorm.weight[...]
    out["pre_feedforward_layernorm_w"] = layer.pre_feedforward_layernorm.weight[...]
    out["post_feedforward_layernorm_w"] = layer.post_feedforward_layernorm.weight[...]
    out["layer_scalar"] = layer.layer_scalar[...]
    out["q_proj_w"] = layer.self_attn.q_proj.weight[...]
    out["q_norm_w"] = layer.self_attn.q_norm.weight[...]
    out["o_proj_w"] = layer.self_attn.o_proj.weight[...]
    attn = layer.self_attn
    head_dim = attn.head_dim
    num_kv_heads = attn.num_kv_heads
    hidden = layer.hidden_size
    if attn.k_proj is not None:
        out["k_proj_w"] = attn.k_proj.weight[...]
        out["v_proj_w"] = attn.v_proj.weight[...]
        out["k_norm_w"] = attn.k_norm.weight[...]
    else:
        # Zero-stubs so all layers in a group have the same pytree.
        dtype = out["q_proj_w"].dtype
        f32 = out["q_norm_w"].dtype
        out["k_proj_w"] = jnp.zeros((num_kv_heads * head_dim, hidden), dtype=dtype)
        out["v_proj_w"] = jnp.zeros((num_kv_heads * head_dim, hidden), dtype=dtype)
        # k_norm weight uses ones (not zeros) so (x * 1) produces x — matters
        # for the matmul producing junk K but the shared path ignores it; we
        # still match what the "real" layer would output on the dropped path.
        out["k_norm_w"] = jnp.ones((head_dim,), dtype=f32)
    out["mlp_gate_w"] = layer.mlp.gate_proj.weight[...]
    out["mlp_up_w"] = layer.mlp.up_proj.weight[...]
    out["mlp_down_w"] = layer.mlp.down_proj.weight[...]
    out["ple_gate_w"] = layer.per_layer_input_gate.weight[...]
    out["ple_proj_w"] = layer.per_layer_projection.weight[...]
    out["ple_post_norm_w"] = layer.post_per_layer_input_norm.weight[...]
    return out


def _stack_group(layers_subset: list) -> dict:
    """Stack a list of same-shape layer-weight dicts along a new leading axis.

    ``layers_subset[i]`` must be a dict with identical keys and matching
    per-key shapes. Returns a dict with the same keys where every value
    has shape ``[len(layers_subset), ...]``."""
    weights = [_layer_weights(layer) for layer in layers_subset]
    keys = weights[0].keys()
    return {k: jnp.stack([w[k] for w in weights], axis=0) for k in keys}


def collect_stacked_weights(layers: list, layer_types: list[str]) -> dict:
    """Given the 42-layer list, produce two stacked trees:
      - sliding: every key is shape [7, 5, ...] (super-block x inner)
      - full:    every key is shape [7, ...] (one full layer per block)
    """
    n_blocks, block_len, sliding_per_block = _partition_layers(layer_types)
    # Per super-block: sliding = layers[b*6 : b*6+5], full = layers[b*6+5].
    sliding_per_blocks = []  # len=7, each item len=5
    full_per_blocks = []     # len=7
    for b in range(n_blocks):
        sliding_per_blocks.append([
            layers[b * block_len + i] for i in range(sliding_per_block)
        ])
        full_per_blocks.append(layers[b * block_len + sliding_per_block])
    # Stack inner (per-block) first -> gives each block a [5, ...] tree.
    per_block_sliding_stacked = [_stack_group(x) for x in sliding_per_blocks]
    # Now stack over the 7 blocks -> [7, 5, ...] tree.
    keys = per_block_sliding_stacked[0].keys()
    sliding_stacked = {
        k: jnp.stack([pb[k] for pb in per_block_sliding_stacked], axis=0)
        for k in keys
    }
    # Full: just stack the 7 full layers -> [7, ...] tree.
    full_per_block_weights = [_layer_weights(l) for l in full_per_blocks]
    full_keys = full_per_block_weights[0].keys()
    full_stacked = {
        k: jnp.stack([w[k] for w in full_per_block_weights], axis=0)
        for k in full_keys
    }
    return {"sliding": sliding_stacked, "full": full_stacked}


# ---------------------------------------------------------------------------
# Functional layer body (pure JAX, no NNX)
# ---------------------------------------------------------------------------


def _rms_norm(x, weight, eps: float, with_scale: bool = True):
    """Pure-JAX RMSNorm matching Gemma4RMSNorm."""
    in_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    mean_sq = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps
    normed = x_f32 * jnp.pow(mean_sq, jnp.float32(-0.5))
    if with_scale:
        normed = normed * weight.astype(jnp.float32)
    return normed.astype(in_dtype)


def _attention_body(
    hidden_states: jax.Array,       # (B, T, hidden)
    position_embeddings: tuple[jax.Array, jax.Array],  # (cos, sin)
    attention_mask: Optional[jax.Array],
    *,
    weights: dict,                  # per-layer weights (sliced from stack)
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_key_value_groups: int,
    sliding_window: Optional[int],
    is_sliding: bool,               # static: full vs sliding group
    is_kv_shared: jax.Array,        # traced 0/1 scalar bool
    borrowed_k: jax.Array,          # carry: (B, Hkv, T, D) — used when is_kv_shared
    borrowed_v: jax.Array,          # carry: (B, Hkv, T, D)
    rms_eps: float,
    attn_impl: str,
):
    """Pure-JAX Gemma4TextAttention body. Returns (attn_out, (k, v))."""
    B, T, _ = hidden_states.shape
    cos, sin = position_embeddings

    # Q path.
    q = hidden_states @ weights["q_proj_w"].T  # (B, T, Hq*D)
    q = q.reshape(B, T, num_heads, head_dim)
    q = _rms_norm(q, weights["q_norm_w"], rms_eps, with_scale=True)
    q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
    q = jnp.transpose(q, (0, 2, 1, 3))  # (B, Hq, T, D)

    # KV path: always compute (from possibly-zero weights for shared), then
    # mux with borrowed K/V via is_kv_shared.
    k_local = hidden_states @ weights["k_proj_w"].T
    v_local = hidden_states @ weights["v_proj_w"].T
    k_local = k_local.reshape(B, T, num_kv_heads, head_dim)
    v_local = v_local.reshape(B, T, num_kv_heads, head_dim)
    k_local = _rms_norm(k_local, weights["k_norm_w"], rms_eps, with_scale=True)
    k_local = apply_rotary_pos_emb(k_local, cos, sin, unsqueeze_dim=2)
    k_local = jnp.transpose(k_local, (0, 2, 1, 3))  # (B, Hkv, T, D)
    # v_norm has with_scale=False (no param).
    # We keep v_local as-is and apply the norm below.
    v_local = _rms_norm(v_local, None, rms_eps, with_scale=False)
    v_local = jnp.transpose(v_local, (0, 2, 1, 3))  # (B, Hkv, T, D)

    # Select borrowed vs local.
    # is_kv_shared is a traced scalar (0 or 1) — use jnp.where for broadcasted pick.
    sel = is_kv_shared.astype(jnp.bool_)
    k = jnp.where(sel, borrowed_k, k_local)
    v = jnp.where(sel, borrowed_v, v_local)

    # Attention kernel.
    if attn_impl == "splash":
        from .pallas_attention import splash_attention
        attn_out = splash_attention(q, k, v, sliding_window=sliding_window)  # (B,T,Hq,D)
    else:
        attn_out = _attn_xla_sdpa(
            q, k, v, attention_mask,
            num_key_value_groups=num_key_value_groups,
            scaling=1.0,
            is_causal=True,
        )  # (B, T, Hq, D)

    attn_out = attn_out.reshape(B, T, num_heads * head_dim)
    attn_out = attn_out @ weights["o_proj_w"].T
    return attn_out, (k_local, v_local)  # return the local (unborrowed) KV so caller can store


def _decoder_layer_body(
    hidden_states: jax.Array,
    per_layer_input: Optional[jax.Array],
    position_embeddings: tuple[jax.Array, jax.Array],
    attention_mask: Optional[jax.Array],
    *,
    weights: dict,
    cfg: dict,                      # dict of attention/layer static hyperparams
    is_kv_shared: jax.Array,
    is_store_kv: jax.Array,         # traced 0/1: should this layer's K/V be stored?
    borrowed_k: jax.Array,
    borrowed_v: jax.Array,
    stored_k: jax.Array,            # carry to update if is_store_kv
    stored_v: jax.Array,
    attn_impl: str,
):
    """One full Gemma4TextDecoderLayer body, pure JAX. Returns
    (new_hidden, new_stored_k, new_stored_v)."""
    residual = hidden_states
    x = _rms_norm(hidden_states, weights["input_layernorm_w"], cfg["rms_eps"])
    attn_out, (k_local, v_local) = _attention_body(
        x, position_embeddings, attention_mask,
        weights=weights,
        num_heads=cfg["num_heads"],
        num_kv_heads=cfg["num_kv_heads"],
        head_dim=cfg["head_dim"],
        num_key_value_groups=cfg["num_key_value_groups"],
        sliding_window=cfg["sliding_window"],
        is_sliding=cfg["is_sliding"],
        is_kv_shared=is_kv_shared,
        borrowed_k=borrowed_k,
        borrowed_v=borrowed_v,
        rms_eps=cfg["rms_eps"],
        attn_impl=attn_impl,
    )
    # Update stored_k/v if is_store_kv (this layer is the store point).
    sel_store = is_store_kv.astype(jnp.bool_)
    new_stored_k = jnp.where(sel_store, k_local, stored_k)
    new_stored_v = jnp.where(sel_store, v_local, stored_v)

    x = _rms_norm(attn_out, weights["post_attention_layernorm_w"], cfg["rms_eps"])
    hidden_states = residual + x

    # MLP.
    residual = hidden_states
    x = _rms_norm(hidden_states, weights["pre_feedforward_layernorm_w"], cfg["rms_eps"])
    gate = x @ weights["mlp_gate_w"].T
    up = x @ weights["mlp_up_w"].T
    x = _gelu_pytorch_tanh(gate) * up
    x = x @ weights["mlp_down_w"].T
    x = _rms_norm(x, weights["post_feedforward_layernorm_w"], cfg["rms_eps"])
    hidden_states = residual + x

    # PLE residual (E4B always has it).
    residual = hidden_states
    x = hidden_states @ weights["ple_gate_w"].T
    x = _gelu_pytorch_tanh(x)
    if per_layer_input is not None:
        x = x * per_layer_input
    x = x @ weights["ple_proj_w"].T
    x = _rms_norm(x, weights["ple_post_norm_w"], cfg["rms_eps"])
    hidden_states = residual + x

    hidden_states = hidden_states * weights["layer_scalar"]
    return hidden_states, new_stored_k, new_stored_v


# ---------------------------------------------------------------------------
# Scan dispatch
# ---------------------------------------------------------------------------


def scan_layers(
    stacked_weights: dict,      # {"sliding": ..., "full": ...}
    hidden_states: jax.Array,   # (B, T, hidden)
    per_layer_inputs: Optional[jax.Array],  # (B, T, L, D_ple) or None
    position_embeddings: dict,  # {layer_type: (cos, sin)}
    masks: dict,                # {layer_type: mask_or_None}
    config,                     # Gemma4TextConfig
    attn_impl: str,
) -> jax.Array:
    """Apply all 42 decoder layers via two-level scan.

    Returns new hidden_states shape (B, T, hidden).
    """
    n_blocks, block_len, sliding_per_block = _partition_layers(config.layer_types)
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_key_value_groups = num_heads // num_kv_heads
    sliding_hd = config.head_dim
    full_hd = config.global_head_dim or config.head_dim
    sliding_window = config.sliding_window
    rms_eps = config.rms_norm_eps
    first_shared = config.num_hidden_layers - config.num_kv_shared_layers

    B, T, _ = hidden_states.shape

    # Precompute per-layer scalars: for each of the 42 slots, whether it's
    # kv-shared and whether it's the KV-store layer.
    is_shared_flags_sliding = jnp.zeros((n_blocks, sliding_per_block), dtype=jnp.int32)
    is_store_flags_sliding = jnp.zeros((n_blocks, sliding_per_block), dtype=jnp.int32)
    is_shared_flags_full = jnp.zeros((n_blocks,), dtype=jnp.int32)
    is_store_flags_full = jnp.zeros((n_blocks,), dtype=jnp.int32)

    # Find "last non-shared layer of matching type" = the store layer.
    # For E4B: layer 22 is last non-shared sliding (store for sliding),
    # layer 23 is last non-shared full (store for full).
    prev_layers = config.layer_types[:first_shared]
    sliding_store_idx = (
        len(prev_layers) - 1 - prev_layers[::-1].index("sliding_attention")
    )
    full_store_idx = (
        len(prev_layers) - 1 - prev_layers[::-1].index("full_attention")
    )
    # Map global layer idx -> (block, inner) for sliding; block for full.
    sliding_block = sliding_store_idx // block_len
    sliding_inner = sliding_store_idx % block_len
    full_block = full_store_idx // block_len

    # Build flat numpy arrays then cast to jnp.
    import numpy as _np
    is_shared_sliding = _np.zeros((n_blocks, sliding_per_block), dtype=_np.int32)
    is_store_sliding = _np.zeros((n_blocks, sliding_per_block), dtype=_np.int32)
    is_shared_full = _np.zeros((n_blocks,), dtype=_np.int32)
    is_store_full = _np.zeros((n_blocks,), dtype=_np.int32)
    for b in range(n_blocks):
        for i in range(sliding_per_block):
            idx = b * block_len + i
            if idx >= first_shared:
                is_shared_sliding[b, i] = 1
            if idx == sliding_store_idx:
                is_store_sliding[b, i] = 1
        full_idx = b * block_len + sliding_per_block
        if full_idx >= first_shared:
            is_shared_full[b] = 1
        if full_idx == full_store_idx:
            is_store_full[b] = 1
    is_shared_sliding = jnp.asarray(is_shared_sliding)
    is_store_sliding = jnp.asarray(is_store_sliding)
    is_shared_full = jnp.asarray(is_shared_full)
    is_store_full = jnp.asarray(is_store_full)

    # Initial stored KV: zeros with the right shape. The store-layer overwrites
    # them; before that, is_kv_shared is always 0 so they're never read.
    stored_k_sliding = jnp.zeros(
        (B, num_kv_heads, T, sliding_hd), dtype=hidden_states.dtype
    )
    stored_v_sliding = jnp.zeros_like(stored_k_sliding)
    stored_k_full = jnp.zeros(
        (B, num_kv_heads, T, full_hd), dtype=hidden_states.dtype
    )
    stored_v_full = jnp.zeros_like(stored_k_full)

    # Static per-group cfgs.
    sliding_cfg = {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": sliding_hd,
        "num_key_value_groups": num_key_value_groups,
        "sliding_window": sliding_window,
        "is_sliding": True,
        "rms_eps": rms_eps,
    }
    full_cfg = {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": full_hd,
        "num_key_value_groups": num_key_value_groups,
        "sliding_window": None,
        "is_sliding": False,
        "rms_eps": rms_eps,
    }
    # Position embeddings + masks per group.
    sliding_pos = position_embeddings["sliding_attention"]
    full_pos = position_embeddings["full_attention"]
    sliding_mask = masks["sliding_attention"]
    full_mask = masks["full_attention"]

    # Preslice per-layer-input tokens by layer index. per_layer_inputs shape is
    # (B, T, L, D_ple); we reshape to (B, T, 7, 6, D_ple) so inner scans index
    # by [block, inner].
    if per_layer_inputs is not None:
        D_ple = per_layer_inputs.shape[-1]
        ple_reshaped = per_layer_inputs.reshape(B, T, n_blocks, block_len, D_ple)
        # Split into sliding (indices 0..4 of each block) and full (index 5).
        ple_sliding = ple_reshaped[:, :, :, :sliding_per_block, :]  # (B,T,7,5,D)
        ple_full = ple_reshaped[:, :, :, sliding_per_block, :]      # (B,T,7,D)
    else:
        ple_sliding = None
        ple_full = None

    # Inner scan body over the 5 sliding layers within one super-block.
    # Wrap in `jax.checkpoint` so scan doesn't materialize a [7*5, B, T, hidden]
    # stack of intermediate activations for the backward pass — without this
    # the scan OOMs at b=3 seq=1024 (1 GiB per layer * 35 = 35 GiB heap).
    from jax import checkpoint_policies as _ckpt_policies
    _REMAT_POLICY = _ckpt_policies.checkpoint_dots_with_no_batch_dims

    def _inner_body(carry, scan_input):
        hidden, k_store, v_store = carry
        weights_i = scan_input["weights"]
        is_shared_i = scan_input["is_shared"]
        is_store_i = scan_input["is_store"]
        ple_i = scan_input["ple"]  # (B, T, D_ple) or None
        @jax.checkpoint  # remat per layer; policy inherited from outer forward_loss wrap
        def _step(hidden, ple_i, is_shared_i, is_store_i, k_store, v_store, weights_i):
            return _decoder_layer_body(
                hidden, ple_i, sliding_pos, sliding_mask,
                weights=weights_i,
                cfg=sliding_cfg,
                is_kv_shared=is_shared_i,
                is_store_kv=is_store_i,
                borrowed_k=k_store,
                borrowed_v=v_store,
                stored_k=k_store,
                stored_v=v_store,
                attn_impl=attn_impl,
            )
        new_hidden, new_k, new_v = _step(hidden, ple_i, is_shared_i, is_store_i, k_store, v_store, weights_i)
        return (new_hidden, new_k, new_v), None

    # Outer scan body over the 7 super-blocks.
    def _outer_body(carry, scan_input):
        hidden, ks, vs, kf, vf = carry  # sliding/full stored KV
        inner_weights = scan_input["inner_sliding_weights"]  # pytree w/ leading 5
        full_weights = scan_input["full_weights"]            # pytree leaf
        inner_is_shared = scan_input["inner_is_shared"]      # (5,)
        inner_is_store = scan_input["inner_is_store"]        # (5,)
        inner_ple = scan_input["inner_ple"]                  # (B, T, 5, D_ple) or None
        full_is_shared = scan_input["full_is_shared"]        # scalar
        full_is_store = scan_input["full_is_store"]          # scalar
        full_ple = scan_input["full_ple"]                    # (B, T, D_ple) or None

        # Inner scan input is per-layer along axis 0 (length 5).
        # Rearrange ple_sliding: inner_ple shape (B, T, 5, D_ple) -> scan expects leading=5.
        if inner_ple is not None:
            # Move axis 2 (len=5) to axis 0.
            inner_ple_scan = jnp.transpose(inner_ple, (2, 0, 1, 3))  # (5, B, T, D_ple)
        else:
            inner_ple_scan = None
        inner_scan_in = {
            "weights": inner_weights,  # already leading-5
            "is_shared": inner_is_shared,
            "is_store": inner_is_store,
            "ple": inner_ple_scan,
        }
        (hidden, ks, vs), _ = jax.lax.scan(_inner_body, (hidden, ks, vs), inner_scan_in)

        # Now apply the one full-attention layer of this super-block, also
        # per-layer rematerialized to match the inner scan's memory behavior.
        @jax.checkpoint
        def _full_step(hidden, full_ple, full_is_shared, full_is_store, kf, vf, full_weights):
            return _decoder_layer_body(
                hidden, full_ple, full_pos, full_mask,
                weights=full_weights,
                cfg=full_cfg,
                is_kv_shared=full_is_shared,
                is_store_kv=full_is_store,
                borrowed_k=kf,
                borrowed_v=vf,
                stored_k=kf,
                stored_v=vf,
                attn_impl=attn_impl,
            )
        new_hidden, new_kf, new_vf = _full_step(hidden, full_ple, full_is_shared, full_is_store, kf, vf, full_weights)
        return (new_hidden, ks, vs, new_kf, new_vf), None

    # Outer scan input: slice across axis 0 (length 7).
    outer_scan_in = {
        "inner_sliding_weights": stacked_weights["sliding"],  # leading [7, 5, ...]
        "full_weights": stacked_weights["full"],              # leading [7, ...]
        "inner_is_shared": is_shared_sliding,                 # (7, 5)
        "inner_is_store": is_store_sliding,                   # (7, 5)
        "inner_ple": (
            jnp.transpose(ple_sliding, (2, 0, 1, 3, 4))  # (7, B, T, 5, D_ple)
            if ple_sliding is not None else None
        ),
        "full_is_shared": is_shared_full,                     # (7,)
        "full_is_store": is_store_full,                       # (7,)
        "full_ple": (
            jnp.transpose(ple_full, (2, 0, 1, 3))  # (7, B, T, D_ple)
            if ple_full is not None else None
        ),
    }
    init = (hidden_states, stored_k_sliding, stored_v_sliding, stored_k_full, stored_v_full)
    (hidden_states, *_), _ = jax.lax.scan(_outer_body, init, outer_scan_in)
    return hidden_states


__all__ = ["is_enabled", "collect_stacked_weights", "scan_layers"]

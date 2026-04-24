"""Pallas `splash_attention` wiring for Gemma 4 under the native-JAX (Flax NNX) port.

JAX-side analog of `../../torchax/model/pallas_attention.py`. Exposes a single
function `splash_attention(q, k, v, sliding_window)` that the attention module
in `modeling_gemma4.py` calls instead of `_attn_xla_sdpa` when
``JAX_ATTENTION_IMPL=splash`` is set.

Differences from the torchax wiring:
  - No `torchax.interop.call_jax` — we're already in JAX land; call the kernel
    directly.
  - No `ALL_ATTENTION_FUNCTIONS` registration — the model's attention module
    dispatches directly on the env var.
  - **No Q-scaling**. The torchax module applies `q * 1/sqrt(head_dim)` before
    the kernel because HF Gemma 4's torch path expects `scaling=1/sqrt(D)`
    semantics wired through SDPA. The native-JAX port sets
    `Gemma4TextAttention.scaling = 1.0` (q_norm / k_norm already normalize per
    head), so splash's internal "no 1/sqrt(D)" behavior matches the baseline
    SDPA path without any pre-kernel scaling. Confirmed against
    `modeling_gemma4.py` line ~374 (`self.scaling = 1.0`) and the exp-34
    writeup.
  - Supports both sliding (head_dim=256) and full-attention (head_dim=512) —
    the kernel is specialized per `(seq_len, num_q_heads, sliding_window,
    head_dim)` tuple and cached.

Config chosen to match the torchax exp-25 best (the +9.2 % over torchax
baseline):
  - block_q = block_kv = block_kv_compute = min(1024, seq_len)
  - block_*_dkv = min(1024, seq_len); block_*_dq omitted (fused bwd kernel)
  - use_fused_bwd_kernel = True
  - QKVLayout.SEQ_MINOR for q/k/v
  - sliding-window masks use LocalMask(window_size=(W, 0))
  - causal full layers use CausalMask
"""
from __future__ import annotations

import functools
import os
from typing import Any, Optional

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Mesh handle — shard_map needs a concrete Mesh (Mosaic custom-calls cannot be
# auto-partitioned). train.py registers the mesh via `set_mesh(mesh)` before
# the first forward.
# ---------------------------------------------------------------------------
_MESH: Optional[Any] = None


def set_mesh(mesh: Any) -> None:
    """Register the active Mesh. Must be called once after `get_mesh(...)`."""
    global _MESH
    _MESH = mesh


def get_mesh() -> Optional[Any]:
    return _MESH


def is_enabled() -> bool:
    """True iff the env var selects the splash path."""
    return os.environ.get("JAX_ATTENTION_IMPL", "xla").lower() == "splash"


# ---------------------------------------------------------------------------
# Kernel builder (LRU cached)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _build_splash_kernel(
    seq_len: int,
    num_q_heads: int,
    sliding_window: Optional[int],
    head_dim: int,
):
    """Build a splash-attention MHA kernel specialized for this shape set.

    The cache key includes head_dim because Gemma 4 has two variants
    (sliding: 256, full: 512). The kernel's BlockSizes struct is otherwise
    head_dim-agnostic.

    IMPORTANT: `make_splash_mha_single_device` internally materializes the
    MaskInfo via `jnp.array(...)`. If this runs inside a traced jit context
    (which happens when the first forward is called under the top-level
    `jitted_step`), those arrays become tracers and the returned closure
    leaks them — any subsequent trace that hits the lru_cache sees a
    tracer from a dead trace (UnexpectedTracerError). Fix: wrap the build
    in `jax.ensure_compile_time_eval()` so everything materializes as
    concrete `jax.Array` values regardless of caller trace context.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sa_kernel,
        splash_attention_mask as sa_mask,
    )

    block_q = min(1024, seq_len)
    block_kv = min(1024, seq_len)
    block_kv_compute = min(1024, seq_len)

    # Fused bwd kernel omits block_q_dq / block_kv_dq (see torchax
    # pallas_attention.py for the exp-16 → exp-17 history).
    block_sizes = sa_kernel.BlockSizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=min(1024, seq_len),
        block_kv_dkv=min(1024, seq_len),
        block_kv_dkv_compute=min(1024, seq_len),
        use_fused_bwd_kernel=True,
        q_layout=sa_kernel.QKVLayout.SEQ_MINOR,
        k_layout=sa_kernel.QKVLayout.SEQ_MINOR,
        v_layout=sa_kernel.QKVLayout.SEQ_MINOR,
    )

    if sliding_window is not None and sliding_window > 0:
        head_mask = sa_mask.LocalMask(
            shape=(seq_len, seq_len),
            window_size=(int(sliding_window), 0),
            offset=0,
        )
    else:
        head_mask = sa_mask.CausalMask(shape=(seq_len, seq_len))

    multi_head_mask = sa_mask.MultiHeadMask(masks=(head_mask,) * num_q_heads)

    # Materialize outside any enclosing trace — see docstring.
    with jax.ensure_compile_time_eval():
        kernel = sa_kernel.make_splash_mha_single_device(
            mask=multi_head_mask,
            block_sizes=block_sizes,
            attn_logits_soft_cap=None,
        )
    return kernel


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def splash_attention(
    q: jax.Array,   # (B, Hq, T, D)
    k: jax.Array,   # (B, Hkv, T, D)
    v: jax.Array,   # (B, Hkv, T, D)
    sliding_window: Optional[int],
) -> jax.Array:
    """Drop-in replacement for the SDPA path in the JAX attention module.

    Returns `out` with shape `(B, T, Hq, D)` — the layout the caller expects
    (it then reshapes to (B, T, Hq*D) for o_proj).

    Notes:
      - GQA is native to splash: `Hq > Hkv` is fine, **do not** repeat K/V.
      - No Q scaling — the caller's `scaling=1.0` is preserved.
      - Wrapped in `jax.shard_map(P('fsdp', None, None, None))` so the Mosaic
        custom-call sees per-chip (batch-local) tensors.
    """
    from jax.sharding import PartitionSpec as P

    b, num_q_heads, seq_len, head_dim = q.shape

    kernel = _build_splash_kernel(seq_len, num_q_heads, sliding_window, head_dim)

    def _inner(qj, kj, vj):
        # kernel expects per-example (rank-3); vmap over the batch axis.
        return jax.vmap(kernel, in_axes=(0, 0, 0))(qj, kj, vj)

    mesh = _MESH
    if mesh is None:
        # Fallback for single-device / unit tests.
        out = _inner(q, k, v)
    else:
        in_specs = (
            P("fsdp", None, None, None),  # Q: [B, H, S, D]
            P("fsdp", None, None, None),  # K: [B, Hkv, S, D]
            P("fsdp", None, None, None),  # V: [B, Hkv, S, D]
        )
        out_specs = P("fsdp", None, None, None)
        sharded = jax.shard_map(
            _inner,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )
        out = sharded(q, k, v)

    # out is (B, Hq, T, D); caller expects (B, T, Hq, D).
    return jnp.transpose(out, (0, 2, 1, 3))


__all__ = ["splash_attention", "set_mesh", "get_mesh", "is_enabled"]

"""Pallas `rmsnorm` wiring for Gemma 4 under torchax.

Replaces `Gemma4RMSNorm.forward` with a hand-written TPU Pallas kernel +
hand-rolled `jax.custom_vjp` so `jax.grad` / `jax.vjp` can differentiate
through it. The forward kernel fuses the row reduction, rsqrt, and the
optional gain multiply into a single VMEM-resident pass; the backward is
plain JAX (memory-bound, XLA handles it fine) — both compute in fp32 and
cast back to bf16 at kernel boundaries.

Why this exists:
  - exp 8's splash kernel collapsed the attention portion of `loop fusion`
    but left ~57 ms/step in that bucket — see [exp 8 writeup](`../../2026-04-23-exp8-splash-attention.md`).
  - Gemma 4 has many RMSNorms per layer (input / post_attn / pre_mlp /
    post_mlp / q_norm / k_norm) × 42 layers + per_layer_input_norm on each
    layer + a top-level final norm, plus a couple more inside the LM head
    path. Each is one `loop fusion` op in the XLA graph. Replacing them
    with a single Pallas custom-call removes bf16 ↔ fp32 casts and the
    per-op reduce + multiply + cast trip through HBM.
  - exp 20 parked because `pallas-forge.rmsnorm` lacks a `custom_vjp`, so
    `jax.grad` couldn't differentiate through it at train time. Exp 33
    solves that directly: write the bwd rule by hand.

References:
  - Pallas-equivalent in our stack: `torchax/model/pallas_attention.py` —
    mesh handle + cached kernel factory + `call_jax`-wrapped torch-side
    function + `register_*` monkey-patch pattern.
  - `jax.experimental.pallas.tpu` (`pltpu`) — TPU VMEM memory space and
    tile semantics.
  - `jax.custom_vjp` — the JAX decorator that lets us ship our own
    (fwd, bwd) pair for `jax.vjp` to find.
  - Reference forward:
        y = x * rsqrt(mean(x^2) + eps) * (weight if with_scale else 1.0)
    with the RMS computed in fp32, weight multiplied in fp32, then cast
    back to `x.dtype` (bf16 in training).

Gemma 4's `Gemma4RMSNorm` (transformers `modeling_gemma4.py:168`) is:

    class Gemma4RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6, with_scale=True):
            self.eps = eps
            self.with_scale = with_scale
            if with_scale:
                self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            mean_sq = x.float().pow(2).mean(-1, keepdim=True) + self.eps
            normed = x.float() * torch.pow(mean_sq, -0.5)
            if self.with_scale:
                normed = normed * self.weight.float()
            return normed.type_as(x)

Note this Gemma 4 variant initializes `weight = ones` and applies `normed
* weight` directly — it does NOT use the zero-init + `(1.0 + weight)`
convention from the original Gemma / Gemma 2 code. We preserve that.

Shape surfaces in Gemma 4 E4B at seq=1024 batch=3 fsdp=4 (per-chip):
  - `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`,
    `post_feedforward_layernorm`: `[1, 1024, 2560]` — D = hidden = 2560.
  - `q_norm`, `k_norm`, `v_norm` inside attention: `[1, 1024, H, 256]` —
    D = head_dim = 256, with a non-trivial middle axis.
  - `norm` at the top of the language model before LM head:
    `[1, 1024, 2560]`.
  - Final `embedding_pre_projection_norm` and others with `with_scale=False`:
    same shape space.

We flatten all leading dims to a single `N` for the kernel grid and let
Pallas iterate one row-of-length-D at a time. D is allowed to be either
256 or 2560 (or anything divisible by 128 for TPU lane alignment); the
kernel is parameterized on D via `functools.lru_cache`.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch


# Imports that pull in JAX / Pallas are deferred inside functions so this
# module can be imported without a TPU present.


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Like `pallas_attention._MESH`, stashed at registration time. We do NOT
# `shard_map` the RMSNorm kernel here — Mosaic kernels technically cannot
# be auto-partitioned, but the Pallas TPU lowering for simple row-wise
# reductions is benign: with per-chip FSDP layouts (leading axis on
# 'fsdp') the batch axis is already local to each chip, and D is fully
# replicated, so JAX's default partitioning is the identity. We keep the
# handle available in case we need to explicitly shard_map in the future
# (e.g. if a shape ever crosses an FSDP seam unexpectedly).
_MESH: "Optional[Any]" = None


# ---------------------------------------------------------------------------
# JAX-side kernel factory. Cached by (D, dtype_str).
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def _build_rmsnorm_kernel(D: int, dtype_str: str, block_rows: int):
    """Build a Pallas TPU forward kernel normalizing the last dim of length D.

    The kernel runs one invocation per group of `block_rows` rows. Each
    invocation:
      1. Loads `[block_rows, D]` into VMEM (implicit via BlockSpec).
      2. Casts to fp32.
      3. Computes `sq = x * x`, then `mean_sq = sum(sq, axis=-1) / D` per row.
      4. Computes `rstd = rsqrt(mean_sq + eps)` per row.
      5. `y = x * rstd`, then `y = y * w` if with_scale.
      6. Casts back to the input dtype and stores.

    Pallas TPU's lowering requires the second-to-last block dim be
    divisible by 8 (sublane dim) and the last by 128 (lane dim), OR equal
    to the overall array dim. So we pick `block_rows = min(32, N)` —
    the last block of a non-multiple-of-32 N uses the full-dim-equal
    fallback (we size N via padding below, see `_pallas_rmsnorm_primal`).

    We also emit the `rstd` scalar as a side output so the backward can
    avoid recomputing it. Shape `[N, 1]` where N = total rows.

    Args:
        D: hidden-dim length (e.g. 2560 or 256).
        dtype_str: "bfloat16" or "float32" — the input/output dtype.
        block_rows: rows per kernel invocation (must divide N; must be a
            multiple of 8 unless block_rows == N).
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    dtype = jnp.dtype(dtype_str)

    def kernel(x_ref, w_ref, use_scale_ref, eps_ref, y_ref, rstd_ref):
        # x: [block_rows, D] fp32
        x = x_ref[...].astype(jnp.float32)
        sq = x * x
        # per-row mean over D: [block_rows, 1]
        mean_sq = jnp.sum(sq, axis=-1, keepdims=True) / jnp.float32(D)
        eps = eps_ref[...].astype(jnp.float32)  # [1]
        rstd = jax.lax.rsqrt(mean_sq + eps)     # [block_rows, 1]
        y = x * rstd                            # [block_rows, D]
        w = w_ref[...].astype(jnp.float32)      # [D]
        use_scale = use_scale_ref[...].astype(jnp.float32)   # [1]
        gain = 1.0 + use_scale[None] * (w[None, :] - 1.0)    # [1, D]
        y = y * gain
        y_ref[...] = y.astype(dtype)
        rstd_ref[...] = rstd.astype(jnp.float32)

    def fwd(x, w, use_scale, eps):
        # x: [N, D] (already reshape-collapsed by caller; N assumed
        # divisible by block_rows, enforced by padding in the caller).
        # w: [D]; use_scale / eps: [1] scalars in fp32.
        N = x.shape[0]
        assert N % block_rows == 0, f"N={N} not divisible by block_rows={block_rows}"
        grid = (N // block_rows,)
        return pl.pallas_call(
            kernel,
            grid=grid,
            in_specs=[
                pl.BlockSpec((block_rows, D), lambda i: (i, 0)),
                pl.BlockSpec((D,), lambda i: (0,)),       # broadcast across grid
                pl.BlockSpec((1,), lambda i: (0,)),       # scalar use_scale
                pl.BlockSpec((1,), lambda i: (0,)),       # scalar eps
            ],
            out_specs=[
                pl.BlockSpec((block_rows, D), lambda i: (i, 0)),   # y
                pl.BlockSpec((block_rows, 1), lambda i: (i, 0)),   # rstd
            ],
            out_shape=[
                jax.ShapeDtypeStruct((N, D), dtype),
                jax.ShapeDtypeStruct((N, 1), jnp.float32),
            ],
        )(x, w, use_scale, eps)

    return fwd


def _pick_block_rows(N: int) -> int:
    """Pick a valid `block_rows` that divides N and is a multiple of 8.

    TPU sublane dim is 8, so Pallas requires that the first block dim is
    divisible by 8 (otherwise it refuses to lower the kernel). Try a few
    sizes in decreasing order; fall back to equal-to-N if none divide it.
    """
    for br in (32, 16, 8):
        if N % br == 0:
            return br
    return N  # the "block_rows == full dim" escape hatch


# ---------------------------------------------------------------------------
# custom_vjp-wrapped JAX function
# ---------------------------------------------------------------------------

def _rmsnorm_ref_fwd(x, w, use_scale: float, eps: float):
    """Plain-JAX fallback forward. Used when the Pallas kernel can't run
    (e.g. D not divisible by 128, or this module is imported without a TPU).
    Also used as the reference for unit tests."""
    import jax
    import jax.numpy as jnp
    x_fp32 = x.astype(jnp.float32)
    mean_sq = jnp.mean(x_fp32 * x_fp32, axis=-1, keepdims=True)
    rstd = jax.lax.rsqrt(mean_sq + jnp.float32(eps))
    y = x_fp32 * rstd
    if use_scale:
        y = y * w.astype(jnp.float32)
    return y.astype(x.dtype), rstd


def _pallas_rmsnorm_primal(x, w, use_scale: float, eps: float):
    """JAX-visible primal: collapse leading dims, call kernel, reshape back.

    Returns (y, (x, rstd_collapsed, w)) — the residuals tuple is consumed
    by the backward rule below. Stays in jax-space throughout.

    Under FSDP the input `x` is sharded on the leading (batch) axis and
    the Pallas custom-call can't be auto-partitioned — JAX raises
    `NotImplementedError: Mosaic kernels cannot be automatically
    partitioned.`. We solve this exactly like pallas_attention does:
    wrap the inner (kernel + reshape) in `jax.shard_map` with a
    batch-sharded in/out spec. Because this module's `_MESH` is set at
    registration time (see `register_pallas_rmsnorm`), the shard_map
    path is available during train. If `_MESH` is None (unit tests,
    single-device runs), we skip shard_map and just call the kernel.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    D = x.shape[-1]
    orig_shape = x.shape
    dtype_str = str(x.dtype)

    # Build a per-chip body that takes x at ORIGINAL rank (with the
    # fsdp-sharded leading axis kept separate from the rest) and returns
    # y at the same shape plus rstd at [N_flat, 1]. We do the reshape
    # INSIDE shard_map so the Pallas call sees a purely-local tensor.
    # Spec pattern: batch (axis 0) is the fsdp-sharded axis; everything
    # else is replicated. shard_map with P('fsdp', None, ...) matches.
    n_axes = len(orig_shape)
    # Shard only on the leading axis. All intermediate axes are
    # replicated per chip, and the last axis D is replicated too.
    in_spec_x = P(*(("fsdp",) + (None,) * (n_axes - 1)))
    in_spec_w = P(None)
    out_spec_y = in_spec_x

    def _kernel_body(xj, wj):
        xf = xj.reshape(-1, D)
        N_local = xf.shape[0]
        if D % 128 != 0 or N_local == 0 or dtype_str not in ("bfloat16", "float32"):
            y, rstd = _rmsnorm_ref_fwd(xf, wj, use_scale, eps)
            rstd = rstd.reshape(N_local, 1)
            return y.reshape(xj.shape), rstd
        block_rows = _pick_block_rows(N_local)
        fwd = _build_rmsnorm_kernel(D, dtype_str, block_rows)
        use_scale_arr = jnp.array([1.0 if use_scale else 0.0], dtype=jnp.float32)
        eps_arr = jnp.array([eps], dtype=jnp.float32)
        y, rstd = fwd(xf, wj, use_scale_arr, eps_arr)
        return y.reshape(xj.shape), rstd

    if _MESH is not None:
        # rstd is [N_flat_local, 1]. Since N_flat_local = fsdp_local_batch *
        # S * ... (product of leading axes), the first axis of rstd is
        # NOT the 'fsdp' axis of the global mesh — it's just a local
        # tensor. Use P() (fully replicated from shard_map's POV) for the
        # rstd output; the result is a concatenated [N_flat_total, 1]
        # fake-replicated across chips. We use the rstd only in the bwd
        # rule which itself goes back through this same shard_map, so
        # the layout never leaves the kernel's per-chip world.
        # Actually: simpler = leave rstd entirely inside shard_map scope
        # and don't expose it as a multi-chip array; instead return only
        # y from shard_map and have the bwd recompute the small rstd
        # value (mean(x*x, -1) is cheap — one pass over x).
        try:
            sharded = jax.shard_map(
                lambda xj, wj: _kernel_body(xj, wj)[0],
                mesh=_MESH,
                in_specs=(in_spec_x, in_spec_w),
                out_specs=out_spec_y,
                check_vma=False,
            )
            y = sharded(x, w)
            # rstd recomputed outside the kernel — cheap, and stays in
            # global-sharding-land so residuals are transparent to jit.
            x_fp32 = x.astype(jnp.float32)
            mean_sq = jnp.mean(x_fp32 * x_fp32, axis=-1, keepdims=True)
            rstd = jax.lax.rsqrt(mean_sq + jnp.float32(eps))
        except Exception as e:  # pragma: no cover - defensive
            print(f"[pallas_rmsnorm] shard_map call failed, falling back: "
                  f"{type(e).__name__}: {e}")
            x_flat = x.reshape(-1, D)
            y_flat, rstd_flat = _rmsnorm_ref_fwd(x_flat, w, use_scale, eps)
            y = y_flat.reshape(orig_shape)
            rstd = rstd_flat.reshape(orig_shape[:-1] + (1,))
    else:
        try:
            y, _rstd_local = _kernel_body(x, w)
            x_fp32 = x.astype(jnp.float32)
            mean_sq = jnp.mean(x_fp32 * x_fp32, axis=-1, keepdims=True)
            rstd = jax.lax.rsqrt(mean_sq + jnp.float32(eps))
        except Exception as e:  # pragma: no cover - defensive
            print(f"[pallas_rmsnorm] kernel call failed, falling back: "
                  f"{type(e).__name__}: {e}")
            x_flat = x.reshape(-1, D)
            y_flat, rstd_flat = _rmsnorm_ref_fwd(x_flat, w, use_scale, eps)
            y = y_flat.reshape(orig_shape)
            rstd = rstd_flat.reshape(orig_shape[:-1] + (1,))

    # Residuals live in original (potentially sharded) space; the bwd
    # rule below re-shards as needed.
    return y, (x, rstd, w)


def _pallas_rmsnorm_bwd(use_scale: float, eps: float, residuals, dy):
    """Backward for RMSNorm.

    Given `y = x * rstd * gain` (gain = w if use_scale else 1):

        dL/dw_i = sum_over_rows (dy_i * x_i * rstd)        (when use_scale)
        g_i    = dy_i * gain_i
        dL/dx_i = rstd * (g_i - x_i * rstd * (sum(g * x) / D) * rstd)
               = rstd * g_i - x_i * rstd^3 * (sum(g * x) / D)

    All reductions in fp32; cast dx back to x's dtype at the end.

    Residuals: (x, rstd, w) at the ORIGINAL shape:
        x:     [..., D]
        rstd:  [..., 1]  (keepdims=True pattern — broadcasts over D)
        w:     [D]
    dy also has shape [..., D]. Reductions over axis=-1 only.
    """
    import jax.numpy as jnp

    x, rstd, w = residuals
    D = x.shape[-1]
    orig_dtype = x.dtype

    dy_fp32 = dy.astype(jnp.float32)
    x_fp32 = x.astype(jnp.float32)
    rstd_fp32 = rstd.astype(jnp.float32)
    w_fp32 = w.astype(jnp.float32)

    # gain: broadcast w across leading dims, or 1.0 if no scale.
    if use_scale:
        g = dy_fp32 * w_fp32  # [..., D]
    else:
        g = dy_fp32

    # sum(g * x) over last dim — keeps leading dims. [..., 1]
    gx_sum = jnp.sum(g * x_fp32, axis=-1, keepdims=True)
    dx = rstd_fp32 * g - x_fp32 * (rstd_fp32 ** 3) * (gx_sum / jnp.float32(D))
    dx = dx.astype(orig_dtype)

    if use_scale:
        # dw_i = sum_over_all_leading_dims(dy * x * rstd) at each D.
        y_normed = x_fp32 * rstd_fp32  # broadcasts rstd over D
        # sum over everything except the last axis
        axes = tuple(range(dy_fp32.ndim - 1))
        dw = jnp.sum(dy_fp32 * y_normed, axis=axes).astype(w.dtype)
    else:
        dw = jnp.zeros_like(w)

    return dx, dw


# Build a `custom_vjp` JAX function. `use_scale` and `eps` are static
# (nondiff) args passed by the caller; in the module wrapper below we
# bake them into the jax function's closure.

def _make_custom_vjp_fn(use_scale: bool, eps: float):
    """Return a `jax.custom_vjp` function specialized to (use_scale, eps)."""
    import jax

    use_scale_flag = 1.0 if use_scale else 0.0

    @jax.custom_vjp
    def rmsnorm_fn(x, w):
        # Primary path used in fwd-only (e.g. inference): discard residuals.
        y, _residuals = _pallas_rmsnorm_primal(x, w, use_scale_flag, eps)
        return y

    def _fwd(x, w):
        y, residuals = _pallas_rmsnorm_primal(x, w, use_scale_flag, eps)
        return y, residuals

    def _bwd(residuals, dy):
        return _pallas_rmsnorm_bwd(use_scale_flag, eps, residuals, dy)

    rmsnorm_fn.defvjp(_fwd, _bwd)
    return rmsnorm_fn


# Cache the custom_vjp callables per (use_scale, eps). eps is typically
# 1e-6 across all Gemma 4 norms so this cache is tiny.
@functools.lru_cache(maxsize=8)
def _get_custom_vjp(use_scale: bool, eps: float):
    return _make_custom_vjp_fn(use_scale, eps)


# ---------------------------------------------------------------------------
# Torch-side forward replacement
# ---------------------------------------------------------------------------

def pallas_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for `Gemma4RMSNorm.forward`.

    Behaviour parity with the HF reference:
      - Input cast to fp32, reduction in fp32, rsqrt in fp32, gain in fp32,
        cast back to input dtype at the very end.
      - `with_scale=False` path (q_norm's v_norm / embedding_pre_projection_norm
        / router norm): skip the multiply by self.weight entirely; self.weight
        is not allocated in that case.

    Side note on the weight convention: this Gemma 4 variant uses
    `weight = ones` + `normed * weight`, NOT the Gemma-2 `zeros + (1+weight)`
    convention mentioned in exp 33's prompt — we verified against
    transformers' `modeling_gemma4.py:168`. We preserve the exact semantics.
    """
    from torchax import interop

    use_scale = self.with_scale
    eps = self.eps

    fn = _get_custom_vjp(use_scale, eps)

    if use_scale:
        def _jax_fn(xj, wj):
            return fn(xj, wj)
        return interop.call_jax(_jax_fn, hidden_states, self.weight)
    else:
        # No weight parameter. We pass a dummy [D] ones array; the kernel
        # ignores it because use_scale=0.0.
        def _jax_fn(xj):
            import jax.numpy as jnp
            D = xj.shape[-1]
            w_dummy = jnp.ones((D,), dtype=xj.dtype)
            return fn(xj, w_dummy)
        return interop.call_jax(_jax_fn, hidden_states)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_PATCHED = False


def register_pallas_rmsnorm(mesh: "Any") -> bool:
    """Monkey-patch `Gemma4RMSNorm.forward` with the Pallas implementation.

    Must be called BEFORE `interop.JittableModule(model, …)` — the wrapper
    captures the module's forward at construction time via `functional_call`,
    so later patches may be invisible.

    Returns True on success, False (and logs a fallback message) on any
    exception. Safe to call multiple times; the second call is a no-op.

    Args:
        mesh: the active `jax.sharding.Mesh`. Stashed for future use; at
            present the kernel path runs without a shard_map because the
            RMSNorm op is row-wise and fully lane-parallel.
    """
    global _MESH, _PATCHED
    _MESH = mesh
    if _PATCHED:
        return True
    try:
        from transformers.models.gemma4 import modeling_gemma4
        modeling_gemma4.Gemma4RMSNorm.forward = pallas_rmsnorm_forward
        _PATCHED = True
        print("[pallas_rmsnorm] registered — Gemma4RMSNorm.forward patched")
        return True
    except Exception as e:
        print(f"[pallas_rmsnorm] fallback: registration failed — "
              f"{type(e).__name__}: {e}")
        return False

"""Pallas RMSNorm wiring for Gemma 4 under torchax.

Swaps HuggingFace's default ``Gemma4RMSNorm.forward`` (stock PyTorch -> torchax
aten lowering -> XLA) for a fused Pallas-TPU RMSNorm kernel, routed via
``torchax.interop.call_jax`` + ``jax.shard_map``.

Gemma 4 invokes RMSNorm ~5x per layer x 42 layers = ~210 calls per forward,
so even a modest per-call win stacks.

References:
  - pallas-forge source: https://github.com/linhkid/pallas-forge
  - pallas-forge README-published benchmark: 3.44x fused RMSNorm+residual vs
    XLA on TPU v5e (``fused_rmsnorm_residual`` in pallas_forge/kernels/rmsnorm.py).
  - ``wiki/experiments/gemma4_autoresearch_optimization/program.md`` -- build
    candidates #5 (Pallas RMSNorm) and #7 (fused residual+RMSNorm).
  - ``wiki/experiments/.../torchax/model/pallas_attention.py`` -- reference
    template for the torch -> JAX -> Pallas -> torch bridge and the
    ``jax.shard_map`` mesh-wrapping trick for Mosaic custom-calls.

HF ``Gemma4RMSNorm`` surface (transformers modeling_gemma4.py line 168):
    ``__init__(dim, eps=1e-6, with_scale=True)`` -- stores ``self.eps``,
    ``self.with_scale``, and (if ``with_scale``) ``self.weight`` of shape
    ``[dim]``. Gemma 4's ``v_norm`` (line 1181) passes ``with_scale=False``, so
    the module has no ``self.weight`` attribute in that case -- we must
    synthesize a ones-vector before calling the kernel.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch


# Imports that pull in JAX / Pallas are deferred inside functions so this
# module can be imported in environments without a TPU present.


# ---------------------------------------------------------------------------
# Module-level mesh handle + fallback state
# ---------------------------------------------------------------------------
#
# Same reasoning as pallas_attention._MESH: Pallas / Mosaic custom-calls can't
# be auto-partitioned inside a sharded-jit context, so we wrap the kernel body
# in ``jax.shard_map`` with explicit ``PartitionSpec``s. shard_map requires a
# concrete ``Mesh`` (not ``AbstractMesh``), which ``train.py`` passes through
# ``register_pallas_rmsnorm``.
_MESH: "Optional[Any]" = None

# Original HF forward, captured before we monkey-patch. Fallback path uses it.
_ORIG_FORWARD: "Optional[Any]" = None

# (shape, dtype, with_scale) -> True once we've logged a fallback for that key.
# Keeps the log at "at most one line per unique input signature".
_FALLBACK_LOGGED: "dict[tuple, bool]" = {}


# ---------------------------------------------------------------------------
# Vendored pallas-forge bits
# ---------------------------------------------------------------------------
#
# We vendor (rather than import) the small pallas-forge pieces we need so this
# experiment doesn't require pip-installing the package into the env. Chunks
# are annotated with the pallas-forge source file:line of origin.

# pallas-forge/pallas_forge/kernels/rmsnorm.py:31 -- sublane-alignment constant.
TOKENS_PER_TILE = 8


# pallas-forge/pallas_forge/_compat.py:53 -- pallas_call compat shim. We keep
# only the pieces we actually need (auto interpret-mode on CPU; drop
# ``num_stages`` because we don't use it).
def _pallas_call_compat(kernel_fn, *, grid, in_specs, out_specs, out_shape, **kwargs):
    import jax
    from jax.experimental import pallas as pl

    # Detect TPU once. On CPU fall back to interpret mode for unit tests.
    try:
        interpret = not any(d.platform == "tpu" for d in jax.devices())
    except RuntimeError:
        interpret = True
    return pl.pallas_call(
        kernel_fn,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        interpret=interpret,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Pallas kernel: plain RMSNorm (no residual)
# ---------------------------------------------------------------------------
#
# pallas-forge ships ``fused_rmsnorm_residual`` (rmsnorm.py:68) which fuses
# (x + residual) -> rmsnorm. Gemma 4's ``Gemma4RMSNorm.forward`` does NOT do a
# residual add -- the residual connection lives outside the norm in the model
# (e.g. ``hidden_states = residual + hidden_states`` before the next
# ``input_layernorm``). So we strip the residual input from the kernel and
# keep just the RMSNorm body; numerics stay identical to the HF ``_norm`` +
# optional ``*weight`` flow.
#
# Block sizes on TPU v6e (Trillium; MXU 256x256): we keep TOKENS_PER_TILE=8
# (sublane alignment) and pass the full hidden dim as the lane dim per tile.
# For Gemma 4 ``hidden_size=2560`` and ``head_dim=256``, both are multiples of
# 128 (TPU lane alignment), so no extra padding on the last axis is needed.
#
# pallas-forge/pallas_forge/kernels/rmsnorm.py:34-65 -- kernel body, with the
# residual inputs removed.
def _rmsnorm_kernel(
    x_ref,       # [TOKENS_PER_TILE, dim]
    weight_ref,  # [dim]
    out_ref,     # [TOKENS_PER_TILE, dim]
    *,
    eps: float,
    apply_scale: bool,
):
    import jax.numpy as jnp

    x = x_ref[...].astype(jnp.float32)
    variance = jnp.mean(x * x, axis=-1, keepdims=True)
    rms = jnp.sqrt(variance + eps)
    normed = x / rms
    if apply_scale:
        w = weight_ref[...].astype(jnp.float32)
        normed = normed * w[None, :]
    out_ref[...] = normed.astype(out_ref.dtype)


# ---------------------------------------------------------------------------
# JAX-side entry point (jit + shard_map wrapped)
# ---------------------------------------------------------------------------

@functools.partial(
    # eps is static so the kernel can fold it into the constant pool; the
    # apply_scale flag flips the weight-multiply path on/off without retracing
    # every call from a Python-side branch.
    # Bookkeeping: we re-jit per (eps, apply_scale, shape) combo via jax's
    # tracing cache.
    lambda fn: fn,  # no-op; we jit from _pallas_rmsnorm below so we can close
                    # over the mesh without confusing static_argnames.
)
def _noop():
    pass


def _build_pallas_rmsnorm(dim: int, eps: float, apply_scale: bool, out_dtype):
    """Return a jittable function ``(x_2d, weight) -> out_2d`` specialized
    for this (dim, eps, apply_scale, dtype) combo.

    pallas-forge/pallas_forge/kernels/rmsnorm.py:68-149 -- tracks the
    ``fused_rmsnorm_residual`` wrapper: flatten to 2D, pad tokens to a
    TOKENS_PER_TILE multiple, call pallas, slice back.
    """
    import jax
    from jax.experimental import pallas as pl

    def _fn(x_2d, weight):
        import jax.numpy as jnp

        num_tokens = x_2d.shape[0]
        # pallas-forge/pallas_forge/_utils.py:22 -- pad_to_multiple idea,
        # inlined because we only need it on one axis.
        pad = (-num_tokens) % TOKENS_PER_TILE
        if pad:
            x_2d = jnp.pad(x_2d, ((0, pad), (0, 0)))
        padded_tokens = x_2d.shape[0]
        n_tiles = padded_tokens // TOKENS_PER_TILE

        kernel = functools.partial(
            _rmsnorm_kernel, eps=eps, apply_scale=apply_scale
        )

        (out_padded,) = _pallas_call_compat(
            kernel,
            grid=(n_tiles,),
            in_specs=[
                pl.BlockSpec((TOKENS_PER_TILE, dim), lambda i: (i, 0)),
                pl.BlockSpec((dim,), lambda i: (0,)),
            ],
            out_specs=[
                pl.BlockSpec((TOKENS_PER_TILE, dim), lambda i: (i, 0)),
            ],
            out_shape=[
                jax.ShapeDtypeStruct((padded_tokens, dim), out_dtype),
            ],
        )(x_2d, weight)

        # pallas-forge/pallas_forge/_utils.py:37 -- unpad: slice back.
        return out_padded[:num_tokens]

    return _fn


def _jax_rmsnorm(x_jax, weight_jax, eps: float, apply_scale: bool):
    """JAX-side forward. ``x_jax`` is [B, S, D] (3D) or [B, S, H, D] (4D);
    weight is [D]. We flatten all but the last dim before the kernel.

    shard_map specs: the FSDP layout batch-shards on axis 0 (``'fsdp'``). The
    RMSNorm reduction is over the last axis, so there's no cross-device
    collective needed -- each chip can normalize its own batch slice
    independently. Weight is replicated (tiny, [D]), and eps/apply_scale are
    Python-side static.
    """
    import jax
    from jax.sharding import PartitionSpec as P

    original_shape = x_jax.shape
    dim = original_shape[-1]
    out_dtype = x_jax.dtype

    # Flatten leading dims -> [M, D]
    x_2d = x_jax.reshape(-1, dim)

    fn_2d = _build_pallas_rmsnorm(dim, eps, apply_scale, out_dtype)

    if _MESH is None:
        # No mesh registered (single-device / unit tests). Run directly.
        out_2d = fn_2d(x_2d, weight_jax)
        return out_2d.reshape(original_shape)

    # Re-shape back to the original rank before shard_map so we can pin the
    # FSDP axis to the batch dim (axis 0). Inside the shard_map body we
    # flatten again.
    x_r = x_2d.reshape(original_shape)

    # Build per-rank in_specs. Only the batch axis (0) is sharded on 'fsdp';
    # everything else (seq, heads, head_dim) is replicated. Weight is
    # replicated (P()).
    batch_spec = P("fsdp", *([None] * (x_r.ndim - 1)))

    def _inner(xr, w):
        import jax.numpy as jnp

        local_shape = xr.shape
        local_2d = xr.reshape(-1, local_shape[-1])
        local_out = fn_2d(local_2d, w)
        return local_out.reshape(local_shape)

    sharded = jax.shard_map(
        _inner,
        mesh=_MESH,
        in_specs=(batch_spec, P()),
        out_specs=batch_spec,
        check_vma=False,
    )
    return sharded(x_r, weight_jax)


# ---------------------------------------------------------------------------
# Torch-facing forward (monkey-patched onto Gemma4RMSNorm)
# ---------------------------------------------------------------------------

def _rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for ``Gemma4RMSNorm.forward``.

    Bridges torch -> JAX -> Pallas kernel -> torch via
    ``torchax.interop.call_jax``. Handles both ``with_scale=True`` (uses
    ``self.weight``) and ``with_scale=False`` (Gemma 4 ``v_norm``; synthesize
    a ones vector -- but we also pass ``apply_scale=False`` to the kernel so
    we skip the redundant multiply entirely).

    Input rank: the kernel expects 2D ``[M, D]``. We flatten all leading dims
    and reshape back afterwards inside ``_jax_rmsnorm``. This covers:
      - ``[B, S, H]`` (input_layernorm, post_attention_layernorm,
        pre_feedforward_layernorm, post_feedforward_layernorm, self.norm),
      - ``[B, S, n_heads, head_dim]`` (q_norm, k_norm, v_norm).
    """
    from torchax import interop

    eps = float(self.eps)
    apply_scale = bool(getattr(self, "with_scale", True))

    # Build / fetch the weight tensor. When with_scale=False, HF doesn't
    # create self.weight at all -- synthesize a ones vector matching the
    # last dim so the kernel's weight BlockSpec is always satisfied (we skip
    # the multiply via apply_scale=False, so its exact value doesn't matter;
    # but the kernel still needs *a* ref of the right shape).
    if apply_scale:
        weight = self.weight
    else:
        dim = hidden_states.shape[-1]
        weight = torch.ones(dim, dtype=hidden_states.dtype, device=hidden_states.device)

    shape_key = (tuple(hidden_states.shape), hidden_states.dtype, apply_scale)

    try:
        def _jax_fn(x_j, w_j):
            return _jax_rmsnorm(x_j, w_j, eps=eps, apply_scale=apply_scale)

        out = interop.call_jax(_jax_fn, hidden_states, weight)
        # Cast back to the input dtype in case the kernel bumped to f32.
        if out.dtype != hidden_states.dtype:
            out = out.to(hidden_states.dtype)
        return out
    except Exception as e:  # pragma: no cover - defensive fallback
        if shape_key not in _FALLBACK_LOGGED:
            _FALLBACK_LOGGED[shape_key] = True
            print(
                f"[rmsnorm] fallback to torch: shape={shape_key[0]} "
                f"dtype={shape_key[1]} with_scale={apply_scale}: "
                f"{type(e).__name__}: {e}"
            )
        assert _ORIG_FORWARD is not None, (
            "register_pallas_rmsnorm was not called before the first forward."
        )
        return _ORIG_FORWARD(self, hidden_states)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_pallas_rmsnorm(mesh: "Any") -> None:
    """Monkey-patch ``Gemma4RMSNorm.forward`` to route through the pallas-forge
    Pallas kernel.

    Args:
        mesh: the active ``jax.sharding.Mesh``; must carry an ``'fsdp'`` axis.
            Stashed module-globally so ``_jax_rmsnorm`` can wrap the Pallas
            call in ``jax.shard_map`` (Mosaic custom-calls cannot be
            auto-partitioned).
    """
    global _MESH, _ORIG_FORWARD
    _MESH = mesh

    from transformers.models.gemma4 import modeling_gemma4

    # Capture the original forward exactly once so we can restore it in the
    # fallback path. Subsequent calls to register_pallas_rmsnorm (e.g. from a
    # test harness) don't clobber the reference.
    if _ORIG_FORWARD is None:
        _ORIG_FORWARD = modeling_gemma4.Gemma4RMSNorm.forward

    modeling_gemma4.Gemma4RMSNorm.forward = _rmsnorm_forward
    print("[rmsnorm] using pallas-forge Pallas kernel")

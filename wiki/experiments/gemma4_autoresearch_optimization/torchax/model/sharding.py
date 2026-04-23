"""Mesh + per-parameter sharding rules for Gemma 4 on TPU.

Two strategies are supported:

* ``fsdp`` (**default**) — every ≥2D parameter is sharded along its largest
  divisible dim over a 1D ``'fsdp'`` mesh axis that spans all devices.
  GSPMD inserts the all-gather (fwd) / reduce-scatter (bwd) automatically.
  Matches PyTorch FSDP's ``FULL_SHARD`` semantics. Default for fine-tuning.
* ``tp`` — NeMo-Megatron tensor-parallel recipe (columnwise Q / MLP
  up/gate, rowwise O / MLP down, vocab-sharded embed). Adapted from the
  jax-huggingface Part 2 recipe for an 8-way TP on v6e-8. Use when a
  model doesn't fit on one chip's HBM or when decode latency is the
  headline metric.

For Gemma 4 E4B (~8B params with embeddings, ~4.5B effective via
Per-Layer Embeddings) FSDP is the right default — weights fit on one
v6e chip but optimizer state (AdamW fp32 ≈ 12 bytes/param = 96 GB) does
not. FSDP drops optimizer-state-per-chip to ~12 GB.

Not exercised — see the UNTESTED warning in ``train.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# -----------------------------------------------------------------------------
# Mesh axis names — only the axes actually present in a given mesh are used.
# -----------------------------------------------------------------------------

AXIS_FSDP = "fsdp"
AXIS_DP = "dp"
AXIS_TP = "tp"


# -----------------------------------------------------------------------------
# Mesh construction
# -----------------------------------------------------------------------------


def get_fsdp_mesh(fsdp: Optional[int] = None) -> Mesh:
    """1D FSDP mesh spanning all visible devices.

    If ``fsdp`` is unset, uses ``jax.device_count()``. Refuses to silently
    drop devices.
    """
    n = jax.device_count()
    if fsdp is None:
        fsdp = n
    if fsdp != n:
        raise ValueError(
            f"FSDP mesh size {fsdp} != jax.device_count()={n}. "
            f"Pass --fsdp {n} or unset the flag."
        )
    devices = mesh_utils.create_device_mesh((fsdp,))
    return Mesh(devices, axis_names=(AXIS_FSDP,))


def get_tp_mesh(dp: int = 1, tp: Optional[int] = None) -> Mesh:
    """2D (dp, tp) mesh for the tensor-parallel strategy.

    Defaults ``tp`` to ``jax.device_count() // dp``. Raises if
    ``dp * tp != jax.device_count()``.
    """
    if tp is None:
        tp = jax.device_count() // max(dp, 1)
    total = dp * tp
    avail = jax.device_count()
    if total != avail:
        raise ValueError(
            f"TP mesh (dp={dp}, tp={tp}) needs {total} devices, "
            f"but jax.device_count()={avail}."
        )
    devices = mesh_utils.create_device_mesh((dp, tp))
    return Mesh(devices, axis_names=(AXIS_DP, AXIS_TP))


def get_mesh(strategy: str = "fsdp", *, dp: int = 1, tp: Optional[int] = None,
             fsdp: Optional[int] = None) -> Mesh:
    """Dispatch to the mesh matching ``strategy``."""
    if strategy == "fsdp":
        return get_fsdp_mesh(fsdp)
    if strategy == "tp":
        return get_tp_mesh(dp=dp, tp=tp)
    raise ValueError(f"unknown strategy: {strategy!r}")


def _is_fsdp_mesh(mesh: Mesh) -> bool:
    return AXIS_FSDP in mesh.axis_names


# -----------------------------------------------------------------------------
# Param sharding — FSDP strategy
# -----------------------------------------------------------------------------


@dataclass
class ShardingPlan:
    """What to shard and how. Returned by the plan_* functions."""

    shardings: Dict[str, NamedSharding]
    buckets: Dict[str, list]  # bucket name -> param names (for reporting)
    notes: list  # strings for the report / logging


def plan_fsdp_shardings(
    param_names: Iterable[str],
    param_shapes: Mapping[str, Tuple[int, ...]],
    mesh: Mesh,
) -> ShardingPlan:
    """FSDP: shard every ≥2D param on its largest divisible dim over ``'fsdp'``.

    1D params (norms, biases, rotary buffers) are sharded if the dim is
    divisible by ``fsdp``, else replicated (conservative).
    Params with no dim divisible by ``fsdp`` are replicated with a note.
    """
    fsdp_size = mesh.shape[AXIS_FSDP]
    buckets: Dict[str, list] = {"fsdp_shard": [], "replicated": [], "undivisible": []}
    shardings: Dict[str, NamedSharding] = {}
    notes: list = []

    for name in param_names:
        shape = param_shapes.get(name)
        if shape is None or len(shape) == 0:
            shardings[name] = NamedSharding(mesh, P())
            buckets["replicated"].append(name)
            continue

        # Find largest dim divisible by fsdp_size.
        ordered = sorted(range(len(shape)), key=lambda i: -shape[i])
        shard_dim: Optional[int] = None
        for d in ordered:
            if shape[d] % fsdp_size == 0:
                shard_dim = d
                break

        if shard_dim is None:
            shardings[name] = NamedSharding(mesh, P())
            buckets["undivisible"].append(name)
            continue

        spec_axes = [None] * len(shape)
        spec_axes[shard_dim] = AXIS_FSDP
        shardings[name] = NamedSharding(mesh, P(*spec_axes))
        buckets["fsdp_shard"].append(name)

    if buckets["undivisible"]:
        notes.append(
            f"FSDP: {len(buckets['undivisible'])} param(s) have no dim divisible "
            f"by fsdp={fsdp_size}; they are replicated. Example: "
            f"{buckets['undivisible'][:3]}"
        )

    return ShardingPlan(shardings=shardings, buckets=buckets, notes=notes)


# -----------------------------------------------------------------------------
# Param sharding — TP strategy (NeMo-Megatron)
# -----------------------------------------------------------------------------

# Gemma-family HF param-name convention (as of transformers main, 2026-04).
_COL_SHARD_SUBSTR = ("q_proj", "gate_proj", "up_proj")
_ROW_SHARD_SUBSTR = ("o_proj", "down_proj")
_ROW_SHARD_EMBED = ("embed_tokens", "lm_head")


def _gqa_can_shard_kv(num_kv_heads: int, tp: int) -> bool:
    """Can we shard K/V heads over ``tp`` without replication?

    Requires tp to divide num_kv_heads. For Gemma 4 E4B (kv=2, tp=8) this is
    False: we would need a broadcasted GQA sharding, not the basic row/column
    pattern. Future hypothesis.
    """
    return num_kv_heads % tp == 0 and num_kv_heads >= tp


def plan_tp_shardings(
    param_names: Iterable[str],
    mesh: Mesh,
    *,
    num_kv_heads: int,
    num_attention_heads: int,
    param_shapes: Optional[Mapping[str, Tuple[int, ...]]] = None,
) -> ShardingPlan:
    """NeMo-Megatron TP recipe adapted for GQA.

    When ``param_shapes`` is given, rank-0 / rank-1 params that happen to
    match a TP substring (e.g., a per-layer scalar named like a proj) are
    skipped and fall through to the replicated bucket — splitting a rank-<2
    tensor on a 2D ``P(axis, None)`` spec is a JAX error (exp 32 hit this
    before ``param_shapes`` was threaded through).

    For 2D (dp, tp) meshes (exp 32+), params that do NOT match a TP
    substring are dp-sharded FSDP-style on the largest dim divisible by dp
    when shapes are available. This is a one-sided hybrid — TP where the
    NeMo-Megatron recipe applies, FSDP-like sharding for everything else,
    so opt-state per chip still shrinks on the dp axis.
    """
    tp = mesh.shape[AXIS_TP]
    dp_size = mesh.shape.get(AXIS_DP, 1) if AXIS_DP in mesh.axis_names else 1

    notes: list = []
    if num_attention_heads % tp != 0:
        notes.append(
            f"WARNING: num_attention_heads={num_attention_heads} not divisible "
            f"by tp={tp}; Q/O sharding will be wrong. Consider a different tp."
        )
    kv_head_shardable = _gqa_can_shard_kv(num_kv_heads, tp)
    if not kv_head_shardable:
        notes.append(
            f"GQA: num_kv_heads={num_kv_heads} does not divide tp={tp}. "
            "K/V projections will be REPLICATED over the tp axis."
        )

    buckets: Dict[str, list] = {
        "col_shard": [], "row_shard": [], "row_shard_embed": [],
        "kv_replicated": [], "replicated": [], "dp_shard": [], "undivisible": [],
    }
    shardings: Dict[str, NamedSharding] = {}

    def _assign(name: str, spec: P, bucket: str) -> None:
        shardings[name] = NamedSharding(mesh, spec)
        buckets[bucket].append(name)

    def _rank(name: str) -> int:
        if param_shapes is None:
            return 2  # assume 2D (original behavior)
        shape = param_shapes.get(name)
        return 0 if shape is None else len(shape)

    def _fallback(name: str) -> None:
        """Replicate over tp; dp-shard FSDP-style if 2D mesh and divisible."""
        shape = param_shapes.get(name) if param_shapes else None
        if shape is None or len(shape) == 0 or dp_size == 1:
            _assign(name, P(), "replicated")
            return
        # Find largest dim divisible by dp.
        ordered = sorted(range(len(shape)), key=lambda i: -shape[i])
        shard_dim: Optional[int] = None
        for d in ordered:
            if shape[d] % dp_size == 0:
                shard_dim = d
                break
        if shard_dim is None:
            _assign(name, P(), "undivisible")
            return
        # Replicate over all dims except shard_dim which takes dp.
        spec_axes = [None] * len(shape)
        spec_axes[shard_dim] = AXIS_DP
        _assign(name, P(*spec_axes), "dp_shard")

    # For 2D (dp, tp) meshes: also shard the OTHER dim of TP-matched weights
    # on the dp axis so opt-state is 2-way × 2-way = 4-way sharded per chip,
    # matching what 1D fsdp=4 gave. Without this, tp-only sharding keeps
    # opt-state 2x larger per chip than 1D fsdp and compile-time OOMs.
    use_hybrid_2d = dp_size > 1 and AXIS_DP in mesh.axis_names

    def _shape_div(name: str, axis_idx: int, divisor: int) -> bool:
        if param_shapes is None:
            return True  # optimistic
        shape = param_shapes.get(name)
        if shape is None or axis_idx >= len(shape):
            return False
        return shape[axis_idx] % divisor == 0

    for name in param_names:
        rank = _rank(name)
        if any(s in name for s in ("k_proj", "v_proj")) and rank >= 2:
            if kv_head_shardable:
                # col-shard: out_dim on tp. Also dp-shard in_dim if possible.
                if use_hybrid_2d and _shape_div(name, 1, dp_size):
                    _assign(name, P(AXIS_TP, AXIS_DP), "col_shard")
                else:
                    _assign(name, P(AXIS_TP, None), "col_shard")
            else:
                # Can't TP-shard kv; dp-shard via fallback.
                _fallback(name)
            continue
        if any(s in name for s in _COL_SHARD_SUBSTR) and rank >= 2:
            if use_hybrid_2d and _shape_div(name, 1, dp_size):
                _assign(name, P(AXIS_TP, AXIS_DP), "col_shard")
            else:
                _assign(name, P(AXIS_TP, None), "col_shard")
            continue
        if any(s in name for s in _ROW_SHARD_SUBSTR) and rank >= 2:
            if use_hybrid_2d and _shape_div(name, 0, dp_size):
                _assign(name, P(AXIS_DP, AXIS_TP), "row_shard")
            else:
                _assign(name, P(None, AXIS_TP), "row_shard")
            continue
        if any(s in name for s in _ROW_SHARD_EMBED) and rank >= 2:
            if use_hybrid_2d and _shape_div(name, 1, dp_size):
                _assign(name, P(AXIS_TP, AXIS_DP), "row_shard_embed")
            else:
                _assign(name, P(AXIS_TP, None), "row_shard_embed")
            continue
        _fallback(name)

    if buckets["undivisible"]:
        notes.append(
            f"TP-mesh fallback: {len(buckets['undivisible'])} non-TP param(s) "
            f"could not be dp-sharded on dp={dp_size}; replicated. "
            f"Example: {buckets['undivisible'][:3]}"
        )
    if dp_size > 1 and buckets["dp_shard"]:
        notes.append(
            f"TP-mesh hybrid: dp-sharded {len(buckets['dp_shard'])} non-TP "
            f"param(s) FSDP-style on dp={dp_size}."
        )

    return ShardingPlan(shardings=shardings, buckets=buckets, notes=notes)


# -----------------------------------------------------------------------------
# Dispatch: inspect the mesh, pick the strategy.
# -----------------------------------------------------------------------------


def get_param_sharding(model: Any, mesh: Mesh) -> ShardingPlan:
    """Build a ShardingPlan for a loaded HF model, picking the strategy from
    the mesh's axis names (FSDP if the mesh carries a ``'fsdp'`` axis, TP
    otherwise).
    """
    state = model.state_dict()
    names = list(state.keys())

    if _is_fsdp_mesh(mesh):
        shapes = {k: tuple(v.shape) for k, v in state.items()}
        return plan_fsdp_shardings(names, shapes, mesh)

    # TP strategy — needs head counts.
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
    shapes = {k: tuple(v.shape) for k, v in state.items()}
    return plan_tp_shardings(
        names, mesh, num_kv_heads=num_kv, num_attention_heads=num_q,
        param_shapes=shapes,
    )


# Back-compat alias.
plan_shardings = plan_tp_shardings


# -----------------------------------------------------------------------------
# Activation-sharding helpers — called inside the compiled train step.
# -----------------------------------------------------------------------------


def input_sharding(mesh: Mesh) -> NamedSharding:
    """``input_ids`` / ``labels`` sharding: shard batch dim on the data axis.

    - FSDP mesh: batch shards on ``'fsdp'`` (each chip sees batch/fsdp rows).
    - TP mesh:  batch shards on ``'dp'``.
    """
    if _is_fsdp_mesh(mesh):
        return NamedSharding(mesh, P(AXIS_FSDP, None))
    return NamedSharding(mesh, P(AXIS_DP, None))


def logits_sharding(mesh: Mesh) -> NamedSharding:
    """Logits sharding after lm_head.

    - FSDP mesh: batch on ``'fsdp'``, vocab replicated (GSPMD will schedule).
    - TP mesh:  batch on ``'dp'``, vocab on ``'tp'``.
    """
    if _is_fsdp_mesh(mesh):
        return NamedSharding(mesh, P(AXIS_FSDP, None, None))
    return NamedSharding(mesh, P(AXIS_DP, None, AXIS_TP))


def replicated(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


__all__ = [
    "AXIS_FSDP", "AXIS_DP", "AXIS_TP",
    "ShardingPlan",
    "get_mesh", "get_fsdp_mesh", "get_tp_mesh",
    "get_param_sharding",
    "plan_fsdp_shardings", "plan_tp_shardings", "plan_shardings",
    "input_sharding", "logits_sharding", "replicated",
]

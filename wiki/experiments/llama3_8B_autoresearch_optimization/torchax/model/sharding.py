"""Mesh + per-parameter sharding rules for Llama 3 8B on TPU."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

AXIS_FSDP = "fsdp"
AXIS_DP = "dp"
AXIS_TP = "tp"

def get_mesh(strategy: str = "fsdp", *, dp: int = 1, tp: Optional[int] = None,
             fsdp: Optional[int] = None) -> Mesh:
    n = jax.device_count()
    if strategy == "fsdp":
        fsdp = fsdp or n
        devices = mesh_utils.create_device_mesh((fsdp,))
        return Mesh(devices, axis_names=(AXIS_FSDP,))
    elif strategy == "tp":
        tp = tp or (n // dp)
        devices = mesh_utils.create_device_mesh((dp, tp))
        return Mesh(devices, axis_names=(AXIS_DP, AXIS_TP))
    raise ValueError(f"unknown strategy: {strategy}")

@dataclass
class ShardingPlan:
    shardings: Dict[str, NamedSharding]
    buckets: Dict[str, list]
    notes: list

def plan_fsdp_shardings(param_names: Iterable[str], param_shapes: Mapping[str, Tuple[int, ...]], mesh: Mesh) -> ShardingPlan:
    fsdp_size = mesh.shape[AXIS_FSDP]
    shardings = {}
    buckets = {"fsdp_shard": [], "replicated": []}
    for name in param_names:
        shape = param_shapes[name]
        shard_dim = None
        for i, d in enumerate(reversed(shape)):
            if d % fsdp_size == 0:
                shard_dim = len(shape) - 1 - i
                break
        if shard_dim is not None:
            spec = [None] * len(shape)
            spec[shard_dim] = AXIS_FSDP
            shardings[name] = NamedSharding(mesh, P(*spec))
            buckets["fsdp_shard"].append(name)
        else:
            shardings[name] = NamedSharding(mesh, P())
            buckets["replicated"].append(name)
    return ShardingPlan(shardings, buckets, [])

def get_param_sharding(model: Any, mesh: Mesh) -> ShardingPlan:
    state = model.state_dict()
    names = list(state.keys())
    shapes = {k: tuple(v.shape) for k, v in state.items()}
    if AXIS_FSDP in mesh.axis_names:
        return plan_fsdp_shardings(names, shapes, mesh)
    # Basic TP plan can be added here if needed
    return ShardingPlan({n: NamedSharding(mesh, P()) for n in names}, {"replicated": names}, [])

def input_sharding(mesh: Mesh) -> NamedSharding:
    if AXIS_FSDP in mesh.axis_names:
        return NamedSharding(mesh, P(AXIS_FSDP, None))
    return NamedSharding(mesh, P(AXIS_DP, None))

def replicated(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())

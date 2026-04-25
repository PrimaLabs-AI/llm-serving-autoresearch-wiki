import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

# Set JAX memory fraction to 90%
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import optax

# Multi-host JAX init: required on multi-slice / multi-host GKE setups (e.g.
# v6e-8 = 2 hosts × 4 chips). On a single-host VM this no-ops cleanly.
# TPU_WORKER_ID is set by XPK; if absent we're running stand-alone.
if "TPU_WORKER_ID" in os.environ:
    jax.distributed.initialize()
    print(f"[dist] jax.distributed initialized; "
          f"process {jax.process_index()}/{jax.process_count()}, "
          f"local devices={jax.local_device_count()}, "
          f"global devices={jax.device_count()}")
import torch
import torchax
from torchax import interop
from torchax.tensor import Tensor as TorchaxTensor


def _torch_to_jax(t: torch.Tensor):
    """Convert a CPU torch tensor to a JAX array, with bf16 bit-cast via uint16
    (numpy has no bf16 dtype, so `t.numpy()` errors for bfloat16 tensors)."""
    t = t.detach().contiguous()
    if t.dtype == torch.bfloat16:
        return lax.bitcast_convert_type(jnp.asarray(t.view(torch.uint16).numpy()),
                                        jnp.bfloat16)
    return jnp.asarray(t.numpy())

# Local imports
this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(this_dir))
from model import LlamaForCausalLM, AutoTokenizer
from model.sharding import get_mesh, get_param_sharding, input_sharding, replicated as replicated_sharding
from data import IGNORE_INDEX, make_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--strategy", default="fsdp")
    parser.add_argument("--weights_dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--compute_dtype", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--profile_dir", default=None)
    parser.add_argument("--profile_steps", type=int, nargs="*", default=[])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
            for k, v in cfg.items():
                setattr(args, k, v)

    # 1. Initialize tokenizer and Data (CPU torch)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print(f"[data] Initializing dataloader for {args.model_id}...")
    global_batch_size = args.batch_size * jax.device_count()
    data_iter = make_dataloader(
        seq_len=args.seq_len, 
        batch_size=global_batch_size, 
        tokenizer=tokenizer
    )

    # 2. Setup torchax compute mode
    if args.compute_dtype == "bf16":
        torchax.enable_performance_mode()
    else:
        torchax.enable_accuracy_mode()

    mesh = get_mesh(args.strategy)
    print(f"[mesh] strategy={args.strategy} devices={jax.device_count()}")

    # 3. Load model on CPU
    torch_weights_dtype = torch.float32 if args.weights_dtype == "fp32" else torch.bfloat16
    print(f"[load] {args.model_id} weights_dtype={args.weights_dtype} on CPU")
    model = LlamaForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch_weights_dtype, 
        device_map="cpu"
    )

    # 4. Sharding and torchax wrapping
    print("[shard] Creating sharded torchax.Tensors from CPU...")
    plan = get_param_sharding(model, mesh)
    env = torchax.default_env()
    
    with env:
        target_state = model.state_dict()
        sharded_state = {}
        for k, v in target_state.items():
            sh = plan.shardings.get(k, replicated_sharding(mesh))
            jax_arr = jax.device_put(_torch_to_jax(v), sh)
            sharded_state[k] = TorchaxTensor(jax_arr, env, requires_grad=v.requires_grad)
            
        print("[shard] Loading sharded state into model...")
        model.load_state_dict(sharded_state, assign=True, strict=False)
        
        print("[shard] Exhaustive conversion pass...")
        def _to_torchax(m):
            for name, p in m.named_parameters(recurse=False):
                if not isinstance(p, TorchaxTensor):
                    jax_arr = jax.device_put(_torch_to_jax(p), replicated_sharding(mesh))
                    setattr(m, name, TorchaxTensor(jax_arr, env, requires_grad=p.requires_grad))
            for name, b in m.named_buffers(recurse=False):
                if not isinstance(b, TorchaxTensor):
                    jax_arr = jax.device_put(_torch_to_jax(b), replicated_sharding(mesh))
                    setattr(m, name, TorchaxTensor(jax_arr, env, requires_grad=False))
            for child in m.children():
                _to_torchax(child)

        _to_torchax(model)
        jmodel = interop.JittableModule(model)

        # 5. Optimizer (AdamW)
        weights = interop.jax_view(jmodel.params)
        buffers = interop.jax_view(jmodel.buffers)
        optimizer = optax.adamw(args.learning_rate)
        opt_state = optimizer.init(weights)

    in_shard = input_sharding(mesh)

    # 6. Training Step with Mixed Precision Gradients
    def forward_loss(weights, buffers, input_ids, labels):
        with torchax.default_env():
            # Cast weights to bf16 for the forward pass to save memory
            tw = {k: interop.torch_view(v).to(torch.bfloat16) for k, v in weights.items()}
            tb = {k: interop.torch_view(v) for k, v in buffers.items()}
            tin = interop.torch_view(input_ids)
            tlabels = interop.torch_view(labels)
            
            out = jmodel.functional_call("forward", tw, tb, tin)
            logits = out.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tlabels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                ignore_index=IGNORE_INDEX
            )
        return interop.jax_view(loss)

    from jax import checkpoint_policies
    checkpointed_forward = jax.checkpoint(
        forward_loss, 
        policy=checkpoint_policies.checkpoint_dots_with_no_batch_dims
    )
    grad_fn = jax.value_and_grad(checkpointed_forward)

    @jax.jit
    def train_step(weights, buffers, opt_state, input_ids, labels):
        loss, grads = grad_fn(weights, buffers, input_ids, labels)
        
        # Grads back to fp32 for Adam
        grads = jax.tree_util.tree_map(lambda g, w: g.astype(w.dtype), grads, weights)
        
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)
        return loss, weights, opt_state

    # 7. Loop
    print(f"[train] starting loop, seq_len={args.seq_len}, batch_size={global_batch_size}")
    step_times = []
    t_start = time.perf_counter()
    
    for step in range(args.steps):
        try:
            input_ids_np, labels_np = next(data_iter)
        except StopIteration:
            break

        input_ids = jax.device_put(input_ids_np, in_shard)
        labels = jax.device_put(labels_np, in_shard)

        if args.profile_dir and args.profile_steps and step in args.profile_steps:
            jax.profiler.start_trace(args.profile_dir)

        t0 = time.perf_counter()
        loss, weights, opt_state = train_step(weights, buffers, opt_state, input_ids, labels)
        jax.block_until_ready(loss)
        dt = time.perf_counter() - t0

        if step > 0:
            step_times.append(dt)

        print(f"[step {step:2d}] loss={float(loss):.4f} dt={dt*1000:.1f}ms")

        if args.profile_dir and args.profile_steps and step in args.profile_steps:
            jax.profiler.stop_trace()

    wall = time.perf_counter() - t_start

    # 8. Report
    tokens_per_step = global_batch_size * args.seq_len
    if step_times:
        mean_dt = sum(step_times) / len(step_times)
        tps = tokens_per_step / mean_dt
    else:
        mean_dt = float("nan")
        tps = float("nan")

    n_params = 8.03e9
    flops_per_token = 6 * n_params
    peak_flops_per_chip = 183.5e12
    mfu = (tps * flops_per_token) / (jax.device_count() * peak_flops_per_chip) if not np.isnan(tps) else 0

    print("\n================ summary ================")
    print(f"steps measured        : {len(step_times)}")
    print(f"mean step time        : {mean_dt*1000:.1f} ms")
    print(f"tokens / sec          : {tps:.0f}")
    print(f"MFU                   : {mfu*100:.1f}%")
    print(f"wall clock            : {wall:.1f}s")
    print("==========================================")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Gemma 4 E4B fine-tune trainer — **native JAX** (Flax NNX).

Mirrors the torchax trainer's CLI, summary block, sharding strategy, and
baseline behaviour so `awk '/step  2/,/step 19/'` comparisons keep working.
The model is the port in `model/modeling_gemma4.py`; weights are loaded
from HuggingFace safetensors into the NNX tree via `model/weight_loader.py`.

Differences from the torchax path:
  - no torch / torchax dependency at run time; only jax + flax + optax.
  - attention is a plain XLA SDPA matmul path. Splash Pallas is a
    follow-up — see the `_attn_xla_sdpa` comment in the model file.
  - weights are loaded in bf16 directly into sharded jax.Arrays.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional


DEFAULTS = {
    "model_id": "google/gemma-4-E4B",
    "dataset": "wikitext-2-raw-v1",
    "seq_len": 2048,
    "batch_size": 4,
    "steps": 20,
    "learning_rate": 1e-5,
    "warmup_steps": 2,
    "strategy": "fsdp",
    "dp": 1,
    "tp": 1,
    "fsdp": 0,
    "dtype": "bf16",
    "log_every": 1,
    "grad_accum": 1,
    "seed": 42,
}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gemma4-jax-trainer",
        description="Gemma 4 E4B fine-tune (native JAX / Flax NNX).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_id", default=DEFAULTS["model_id"])
    p.add_argument("--dataset", default=DEFAULTS["dataset"])
    p.add_argument("--seq_len", type=int, default=DEFAULTS["seq_len"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    p.add_argument("--learning_rate", type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--warmup_steps", type=int, default=DEFAULTS["warmup_steps"])
    p.add_argument("--strategy", choices=["fsdp", "tp"], default=DEFAULTS["strategy"])
    p.add_argument("--fsdp", type=int, default=DEFAULTS["fsdp"])
    p.add_argument("--dp", type=int, default=DEFAULTS["dp"])
    p.add_argument("--tp", type=int, default=DEFAULTS["tp"])
    p.add_argument("--dtype", choices=["bf16", "fp32"], default=DEFAULTS["dtype"])
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--profile_dir", default=None)
    p.add_argument("--profile_steps", type=int, nargs="*", default=[])
    p.add_argument("--log_every", type=int, default=DEFAULTS["log_every"])
    p.add_argument("--grad_accum", type=int, default=DEFAULTS["grad_accum"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--config", default=None,
                   help="YAML file with default values (CLI overrides file).")
    return p


def _load_yaml_defaults(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main(argv: Optional[list] = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.config:
        cfg = _load_yaml_defaults(args.config)
        for k, v in cfg.items():
            if getattr(args, k, None) == DEFAULTS.get(k):
                setattr(args, k, v)

    random.seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    import numpy as np
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from jax.sharding import NamedSharding, PartitionSpec as P
    from transformers import AutoConfig, AutoTokenizer

    # Module-local imports.
    this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(this_dir))
    from model.modeling_gemma4 import Gemma4ForCausalLM
    from model.weight_loader import load_hf_weights
    from model.sharding import (
        get_mesh, get_param_sharding, apply_sharding, input_sharding,
        replicated as replicated_sharding, _iter_params,
    )
    from data import IGNORE_INDEX, make_dataloader

    np.random.seed(args.seed)

    jnp_dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32

    # Mesh --------------------------------------------------------------------
    if args.strategy == "fsdp":
        fsdp_size = args.fsdp or jax.device_count()
        mesh = get_mesh("fsdp", fsdp=fsdp_size)
        print(f"[mesh] strategy=fsdp fsdp={fsdp_size} devices={jax.device_count()}")
    else:
        mesh = get_mesh("tp", dp=args.dp, tp=args.tp)
        print(f"[mesh] strategy=tp dp={args.dp} tp={args.tp} devices={jax.device_count()}")

    # Load HF config (text-only sub-config). ----------------------------------
    print(f"[load] {args.model_id} ({args.dtype})")
    t0 = time.perf_counter()
    hf_config = AutoConfig.from_pretrained(args.model_id)
    text_cfg = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Build the NNX model with random init (will be overwritten by loader).
    rngs = nnx.Rngs(args.seed)
    model = Gemma4ForCausalLM(text_cfg, dtype=jnp_dtype, rngs=rngs)
    stats = load_hf_weights(model, args.model_id, dtype=jnp_dtype, verbose=True)
    print(f"[load] weights: assigned={stats['assigned']} "
          f"skipped_modality={stats['skipped_modality']} "
          f"skipped_shared_kv={stats['skipped_shared_kv']} "
          f"missing={stats['missing']} "
          f"(missing should be 0 or 2 — vision/audio embed connectors)")
    print(f"[load] done in {time.perf_counter() - t0:.1f}s")

    # Sharding ---------------------------------------------------------------
    plan = get_param_sharding(model, mesh)
    for note in plan.notes:
        print(f"[sharding] {note}")
    counts = {k: len(v) for k, v in plan.buckets.items()}
    print(f"[sharding] bucket counts: {counts}")
    apply_sharding(model, plan)

    # Split the NNX model into (graphdef, state) so jit sees a pytree.
    # graphdef is static; state carries the Param values.
    graphdef, state = nnx.split(model)
    # Keep a reference to the text config for helpful prints.
    vocab_size = text_cfg.vocab_size

    # Optimizer ---------------------------------------------------------------
    lr_schedule = optax.warmup_constant_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_steps,
    )
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=0.01)
    opt_state = optimizer.init(state)

    # Data --------------------------------------------------------------------
    if args.strategy == "fsdp":
        fsdp_size = args.fsdp or jax.device_count()
        global_bs = args.batch_size * fsdp_size
        print(f"[data] dataset=wikitext/{args.dataset} seq_len={args.seq_len} "
              f"per_chip_batch={args.batch_size} global_batch={global_bs} (fsdp={fsdp_size})")
    else:
        global_bs = args.batch_size * args.dp
        print(f"[data] dataset=wikitext/{args.dataset} seq_len={args.seq_len} "
              f"per_chip_batch={args.batch_size} global_batch={global_bs} (dp={args.dp})")
    data_iter = make_dataloader(
        seq_len=args.seq_len,
        batch_size=global_bs,
        tokenizer=tokenizer,
        dataset_config=args.dataset,
    )

    in_shard = input_sharding(mesh)

    # Pure-jax forward+loss, closing over graphdef (static). -----------------
    def forward_loss(state, input_ids, labels):
        model = nnx.merge(graphdef, state)
        logits = model(input_ids)  # (B, T, V)
        # CE with ignore_index=-100; bf16 log_softmax to drop the ~4 GiB
        # fp32 logits tensor. Gemma 4 softcap=30 keeps bf16 stable.
        vocab = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab)
        flat_labels = labels.reshape(-1)
        mask = (flat_labels != IGNORE_INDEX).astype(flat_logits.dtype)
        log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
        safe_labels = jnp.where(
            flat_labels == IGNORE_INDEX, jnp.zeros_like(flat_labels), flat_labels
        )
        picked = jnp.take_along_axis(log_probs, safe_labels[:, None], axis=-1).squeeze(-1)
        loss = -(picked * mask).sum() / jnp.maximum(mask.sum(), 1.0)
        return loss.astype(jnp.float32)

    from jax import checkpoint_policies as _ckpt_policies
    checkpointed = jax.checkpoint(
        forward_loss,
        policy=_ckpt_policies.checkpoint_dots_with_no_batch_dims,
    )
    grad_fn = jax.value_and_grad(checkpointed)

    def train_step(state, opt_state, input_ids, labels):
        with jax.named_scope("train_step"):
            with jax.named_scope("forward_backward"):
                loss, grads = grad_fn(state, input_ids, labels)
            with jax.named_scope("optimizer"):
                updates, opt_state = optimizer.update(grads, opt_state, state)
                state = optax.apply_updates(state, updates)
        return loss, state, opt_state

    jitted_step = jax.jit(train_step, donate_argnums=(0, 1))

    # Training loop -----------------------------------------------------------
    print(f"[train] steps={args.steps} grad_accum={args.grad_accum}")
    prof_steps = set(args.profile_steps)
    prof_start = min(prof_steps) if prof_steps else None
    prof_stop = max(prof_steps) if prof_steps else None
    profile_dir = args.profile_dir
    if profile_dir:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)

    step_times = []
    compile_time = None
    profile_active = False

    # Enter the mesh so that jit sees the right sharding environment.
    if hasattr(jax.sharding, "use_mesh"):
        _mesh_cm = jax.sharding.use_mesh(mesh)
        _mesh_cm.__enter__()
    else:
        mesh.__enter__()
    t_start = time.perf_counter()
    for step in range(args.steps):
        try:
            input_ids_np, labels_np = next(data_iter)
        except StopIteration:
            print(f"[data] exhausted at step {step}; stopping.")
            break
        input_ids = jax.device_put(input_ids_np, in_shard)
        labels = jax.device_put(labels_np, in_shard)

        if profile_dir and prof_start is not None and step == prof_start and not profile_active:
            print(f"[profile] start_trace -> {profile_dir}")
            jax.profiler.start_trace(profile_dir)
            profile_active = True

        with jax.profiler.TraceAnnotation(f"train_step_{step}"):
            t_step = time.perf_counter()
            loss, state, opt_state = jitted_step(state, opt_state, input_ids, labels)
            jax.block_until_ready(loss)
            dt = time.perf_counter() - t_step

        if step == 0:
            compile_time = dt
            print(f"[compile] step 0 (compile + first exec): {dt:.2f}s")
        else:
            step_times.append(dt)

        if step % args.log_every == 0:
            loss_f = float(jax.device_get(loss))
            print(f"[step {step:4d}] loss={loss_f:.4f}  dt={dt*1000:.1f}ms")

        if profile_dir and prof_stop is not None and step == prof_stop and profile_active:
            jax.block_until_ready(loss)
            jax.profiler.stop_trace()
            profile_active = False
            print(f"[profile] stop_trace (captured steps {prof_start}..{prof_stop})")

    if profile_active:
        jax.profiler.stop_trace()

    wall = time.perf_counter() - t_start

    # Report ------------------------------------------------------------------
    tokens_per_step = global_bs * args.seq_len
    if step_times:
        mean_dt = sum(step_times) / len(step_times)
        tps = tokens_per_step / mean_dt
    else:
        mean_dt = float("nan")
        tps = float("nan")

    print("\n================ summary ================")
    print(f"compile time (step 0) : {compile_time:.2f}s"
          if compile_time is not None else "compile time          : (no step ran)")
    print(f"steps measured        : {len(step_times)}")
    print(f"mean step time        : {mean_dt*1000:.1f} ms")
    print(f"tokens per step       : {tokens_per_step}")
    print(f"tokens / sec          : {tps:.0f}")
    print(f"wall clock            : {wall:.1f}s")
    print("==========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

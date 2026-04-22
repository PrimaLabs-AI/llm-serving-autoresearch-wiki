# torchax code — Gemma 4 E4B autoresearch

Working scripts, notebooks, and configs for running **Gemma 4 E4B via torchax** (PyTorch-on-JAX) on TPU. This is the **primary** execution path for the program — Gemma 4 ships as a PyTorch model, and torchax carries it to TPU with minimal code changes.

Companion folder: [`../jax/`](../jax/README.md) — native-JAX port (secondary, lit-up once the torchax baseline is stable and a native port becomes a hypothesis).

## Convention

- Code is tracked directly in the main wiki git repo (not a submodule).
- File types permitted: `.py`, `.ipynb`, `.sh`, `.toml`, `.yaml`, `.json`. Binary artifacts (profiles, HLO dumps, checkpoints) go under `raw/profiles/<experiment>/`.
- Every dated experiment (`../<YYYY-MM-DD>-<slug>.md`) references the exact script + config used by relative path into this folder.
- When a script diverges materially between experiments, save a dated copy (`run_v2026-04-22.py`) rather than overwriting — experiment pages must stay reproducible.

## Expected layout (fill as we go)

```
torchax/
  README.md              this file
  train.py               Gemma 4 E4B fine-tune trainer (UNTESTED scaffold)
  run.sh                 wrapper that sets XLA_FLAGS + LIBTPU_INIT_ARGS and
                         hands `--profile_dir` / `--profile_steps` to train.py
  config.yaml            default args for train.py (CLI overrides)
  requirements.txt       pip install targets (jax[tpu], transformers @ main, ...)
  data.py                wikitext loader + fixed-length packer
  model/                 re-exports HF `Gemma4*` classes + the sharding plan
    __init__.py          thin re-exports
    README.md            config + sharding assumptions; canonical source of
                         truth about "where the model code lives"
    sharding.py          get_mesh / plan_shardings — NeMo-Megatron recipe
                         adapted for Gemma 4's GQA (kv=2) and TP=8
  <YYYY-MM-DD>-<slug>/   per-experiment script+config when materially divergent
```

## Running the trainer

**Status:** scaffold only. Written but not executed. See the header of
`train.py` and `UNTESTED` markers for the concrete risks.

### First-time setup

```bash
# Accept the Gemma license on HF (the model card is Apache-2.0 / not gated
# as of 2026-04-22, but you still need to log in).
pip install -U huggingface_hub
huggingface-cli login

# Install deps (from this folder).
pip install -r requirements.txt
```

The trainer does **not** download weights itself — `transformers.from_pretrained`
handles that on first run, into the usual `~/.cache/huggingface/hub/`.

### Baseline smoke test

```bash
# Default: 20 steps, bf16, TP=8 across v6e-8, profile steps 5..7.
bash run.sh
```

`run.sh` sets the XLA / LIBTPU env vars that the
[xprof-mcp TPU optimization guide](../../../sources/2026-xprof-mcp-tpu-optimization.md)
calls out, then forwards any extra arguments to `train.py`. Profile dumps to
`raw/profiles/<YYYY-MM-DD-HHMMSS>-gemma4-baseline/`; HLO dumps next to it
under `hlo/`.

### Direct invocation

```bash
python train.py \
  --model_id google/gemma-4-E4B \
  --dataset wikitext-2-raw-v1 \
  --seq_len 2048 \
  --batch_size 4 \
  --steps 20 \
  --dp 1 --tp 8 \
  --dtype bf16 \
  --profile_dir /tmp/gemma4-profile \
  --profile_steps 5 6 7
```

Key flags (full list: `python train.py --help`):

| Flag | Default | Notes |
|---|---|---|
| `--model_id` | `google/gemma-4-E4B` | HF repo. Use `google/gemma-4-E4B-it` for the instruction-tuned variant. |
| `--dataset` | `wikitext-2-raw-v1` | Pass `wikitext-103-raw-v1` for a real dataset. |
| `--seq_len` | `2048` | Model supports up to 131072 — memory permitting. |
| `--batch_size` | `4` | Per-chip. Global = `batch_size * dp`. |
| `--steps` | `20` | Step 0 is compile + first exec; step-time stats drop it. |
| `--dp` / `--tp` | `1` / `8` | 2D mesh. Require `dp * tp == jax.device_count()`. |
| `--dtype` | `bf16` | `bf16` flips torchax `enable_performance_mode()`; `fp32` flips `enable_accuracy_mode()`. |
| `--profile_dir` + `--profile_steps` | off | Captures xprof trace for the listed step indices. |

### Reporting output

Exit prints a summary table:

```
================ summary ================
compile time (step 0) : 45.32s
steps measured        : 19
mean step time        : 182.4 ms
tokens per step       : 8192
tokens / sec          : 44912
wall clock            : 48.8s
==========================================
```

The baseline experiment page fills these in and references the profile
directory. Numbers above are placeholders — until the scaffold actually
runs, all targets are TBD.

### Known limitations of the scaffold

See the header of `train.py` and `model/README.md`. Most consequential:

- **Untested end-to-end.** Written from the codebase + source page cookbook;
  no step has run.
- **`num_kv_heads = 2` < `tp = 8`** → K/V projections replicated (correct,
  suboptimal). See `model/sharding.py`.
- **Pytree registration** may need patching to the shipped HF version
  (`CausalLMOutputWithPast`, `DynamicCache`, `StaticCache`).
- **Captured constants**: if JAX logs a multi-GB captured-constants warning,
  follow the `functional_call` pattern from
  [jax-huggingface part 3](../../../sources/2026-jax-huggingface-part-3.md)
  — the scaffold already threads weights through `JittableModule.functional_call`,
  so this should be fine, but verify.

## See also

- [torchax codebase page](../../../codebases/torchax.md) — framework architecture, op-lowering, compile boundary.
- [jax-huggingface codebase page](../../../codebases/jax-huggingface.md) — prior-art patterns for running HuggingFace PyTorch models on TPU via torchax.
- [jax-huggingface part 3 — StaticCache + jax.jit decode](../../../sources/2026-jax-huggingface-part-3.md) — captured-constants and `functional_call` patterns relevant to Gemma 4.

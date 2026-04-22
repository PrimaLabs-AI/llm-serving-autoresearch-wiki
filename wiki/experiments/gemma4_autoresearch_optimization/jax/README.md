# jax code — Gemma 4 E4B autoresearch

Working scripts, notebooks, and configs for running **Gemma 4 E4B as a native JAX model** on TPU. This path is secondary — it's lit up once the torchax baseline is stable and converting to native JAX becomes a hypothesis (e.g., to drop torchax dispatch overhead, use JAX primitives directly, or integrate JAX-only tooling like `tokamax`/Pallas kernels with less friction).

Companion folder: [`../torchax/`](../torchax/README.md) — primary PyTorch-on-JAX execution path.

## Convention

- Code is tracked directly in the main wiki git repo (not a submodule).
- File types permitted: `.py`, `.ipynb`, `.sh`, `.toml`, `.yaml`, `.json`. Binary artifacts (profiles, HLO dumps, checkpoints) go under `raw/profiles/<experiment>/`.
- Every dated experiment (`../<YYYY-MM-DD>-<slug>.md`) references the exact script + config used by relative path into this folder.
- When a script diverges materially between experiments, save a dated copy (`run_v2026-04-22.py`) rather than overwriting — experiment pages must stay reproducible.
- **Port-equivalence discipline**: any JAX port must reproduce the torchax baseline's outputs within bf16 tolerance before its performance numbers count as a valid comparison. A JAX version that subtly changes the model is an [invalid](../../../../SCHEMA.md) experiment, not a win.

## Expected layout (fill as we go)

```
jax/
  README.md              this file
  model/                 native-JAX Gemma 4 definition (flax.nnx or equinox or raw jax)
    gemma4.py            model module
    weights.py           HF → JAX weight loader
  baseline/              first native-JAX baseline
    run.py
    config.yaml
  utils/                 shared helpers
  <YYYY-MM-DD>-<slug>/   per-experiment script+config when materially divergent
```

Nothing is filed yet.

## See also

- [scaling-book](../../../codebases/scaling-book.md) — native-JAX scaling idioms (chapters will be sourced in Wave 3).
- [tokamax codebase page](../../../codebases/tokamax.md) — native-JAX kernel library; integration is cheaper from this path than from torchax.
- [torchax companion folder](../torchax/README.md) — the reference implementation whose outputs a JAX port must match.

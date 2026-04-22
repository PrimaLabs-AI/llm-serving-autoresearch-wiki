# tpu_performance_autoresearch_wiki

An LLM-maintained knowledge base for **autoresearch-style optimization of TPU model performance**.

The loop:

```
sources + codebases + profiles  →  concepts + models  →  ranked hypotheses
                                                               ↓
                             observations  ←  experiments (config + run + profile)
                                 ↓
                      (priors revised, new hypotheses formulated)
```

Humans curate sources, set optimization targets, and approve experiments. The LLM writes and maintains every file in `wiki/` — page summaries, cross-references, rankings, experiment logs.

## Scope

- **In:** step time, MFU, tokens/sec, memory, and everything that affects them on TPU — compiler flags, parallelism, rematerialization, attention kernels, layout, fusion, scheduling, precision.
- **Out:** model quality/convergence. An optimization that changes model semantics is invalid — noted, not counted.

## Layout

```
SCHEMA.md           single source of truth — read this to understand how the wiki works
CLAUDE.md           @SCHEMA.md pointer (Claude Code)
GEMINI.md           @SCHEMA.md pointer (Gemini CLI)
wiki/               LLM-owned markdown (index, log, page types per schema)
raw/                immutable inputs — sources, code, profiles, assets
  code/             ingested repos (git submodules)
```

## Working with the repo

Clone with submodules:

```bash
git clone --recurse-submodules <url>
```

Or after a plain clone:

```bash
git submodule update --init --recursive
```

Add a new codebase:

```bash
git submodule add <repo-url> raw/code/<slug>
```

Then ask the agent to ingest it — see `SCHEMA.md` → `INGEST-CODEBASE`.

## Ingested codebases

- [xprof](raw/code/xprof) — XProf profiler + TensorBoard plugin (OpenXLA)
- [xprof-mcp](raw/code/xprof-mcp) — MCP server wrapping xprof for agent-driven profile analysis
- [torchax](raw/code/torchax) — PyTorch-on-JAX interop layer (Google)
- [tokamax](raw/code/tokamax) — custom TPU/GPU kernels on JAX + Pallas (OpenXLA)
- [stablehlo](raw/code/stablehlo) — StableHLO operation set + MLIR dialect (OpenXLA)
- [scaling-book](raw/code/scaling-book) — "How To Scale Your Model": TPU scaling / parallelism reference (JAX ML / Google DeepMind)
- [autoresearch](raw/code/autoresearch) — Karpathy's autoresearch reference implementation (the methodology this wiki adapts to TPU perf)
- [learning-machine](raw/code/learning-machine) — Qi Huang's JAX/ML experiments repo; the `jax-huggingface/` subfolder is ingested as the [jax-huggingface](wiki/codebases/jax-huggingface.md) codebase

## Authoritative contract

[SCHEMA.md](SCHEMA.md) defines page types, operations, frontmatter, naming, and behavioral rules. If anything here conflicts with `SCHEMA.md`, the schema wins.

# Log

## [2026-04-22] lint | link hygiene + cross-link + 4 missing stubs

**Op**: lint (automated pass, 2 parallel subagents + main-thread checks)
**Pages created**: `wiki/concepts/{hbm-bandwidth,ridge-point,reduce-scatter,trace-me}.md` (4 stubs).
**Pages updated**:
- `wiki/codebases/tokamax.md` — 2 broken links fixed (`../scaling-book.md` / `../stablehlo.md` → `./scaling-book.md` / `./stablehlo.md`).
- `wiki/codebases/xprof.md` — 5 broken links fixed (missing `2026-` year prefix on xprof source slugs; also dropped stale `(stub — fill once source pages land)` annotations now that the targets exist).
- `wiki/concepts/{mark-step-sync,tensor-parallelism,sharding,ici,collective-communication,all-reduce,kv-cache,static-cache,jax-trace,decode-profile-signature,dcn,attention-block-sizes}.md` — 12 stubs had jax-huggingface and/or ultrascale-playbook sources added to their `## Sources` section.
- `wiki/index.md` — totals (77→81 concepts, 118→122 pages); Performance metrics & roofline (11→13), Parallelism & collectives (11→12), Profiling (10→11).

**Key result**: All broken markdown links in wiki/ eliminated. 0 orphan pages. Wave-2 ↔ human-ingest cross-link asymmetry closed for the known-relevant concept stubs.

**Notes**:
- **Broken-link residue**: 10 intentional placeholder links in `SCHEMA.md` itself (e.g., `../sources/2022-flash-attention.md`, `relative/path.md`, index-template `<slug>` placeholders). Left as-is — they are documentation examples.
- **Orphans**: 0 under `wiki/sources/` and `wiki/concepts/`. Codebase pages are expected to be lightly-linked (mostly from `index.md` and a few sources) and were not scanned as orphan candidates.
- **Unlinked-mention candidates reported but NOT auto-fixed** (judgment required): top 20 opportunities to wrap concept names in prose as markdown links. The main offenders (by raw mention count in prose without a link) are `megascale` (24× in megascale-viewer source), `hbm` (19× in TPU_OPTIMIZATION), `ici` (16× in TPU_OPTIMIZATION), `splash-attention` (12× in its own source page — expected), and `hlo` (12× in hlo-op-stats). Most of these are "first-mention" linking candidates rather than blanket-wrap-every-mention — deferred to per-page editing rather than bulk automation.
- **Cross-link held-backs** (content-unjustified, flagged for future reconsideration):
  - `jax-huggingface-part-1` → `xla-flags` or `hlo-dumping-and-diffing`: source does not discuss either.
  - `jax-huggingface-part-4` → `custom-trace-annotations`: hardware is A100 GPU and the source only mentions profiling as a gap.
  - `ultrascale-playbook` → `int8-quantization`: playbook covers FP8 (DeepSeek tile-scaled), not int8/AQT. Concept mismatch.
- **Submodule commit freshness**: all 8 codebase pages' `commit:` frontmatter matches current `git submodule status`. No drift.
- **Frontmatter `sources:` field**: all 81 concept stubs consistently carry `sources: N`. Schema doesn't require it, but the vault convention is now established and consistent — no reconciliation needed. If desired, a future edit to `SCHEMA.md` could codify the convention.

## [2026-04-22] ingest-source | Ultra-Scale Playbook (Tazi et al., HF, 2025-02-19)

**Op**: ingest-source
**Pages created**:
- `wiki/sources/2025-ultrascale-playbook.md` (primary source page — **first non-2026 slug**: playbook is dated Feb 2025).
- `wiki/concepts/{ring-attention,context-parallelism,sequence-parallelism,pipeline-parallelism,expert-parallelism}.md` — 5 stubs for parallelism concepts not present after Wave 2.

**Pages updated**:
- `wiki/concepts/{rematerialization,flash-attention,splash-attention,fsdp,tensor-parallelism,sharding,async-collectives,dtype-strategy}.md` — appended the new source to `## Sources` with a one-line GPU↔TPU claim.
- `wiki/codebases/{tokamax,torchax,scaling-book,autoresearch}.md` — appended the source under `## Sources` with a per-codebase connection note.
- `wiki/index.md` — added the source under Sources and the 5 new stubs under Concepts.

**Raw artifacts**:
- `raw/sources/2025-ultrascale-playbook.html` (788 KB full HTML capture of the static asset URL).
- `raw/assets/ultrascale-playbook/` — **90 figures, 5.2 MB**. Every `<img>` referenced by the playbook, downloaded from `nanotron-ultrascale-playbook.static.hf.space`. Referenced inline in the source page; the full inventory is tabulated at the bottom of that page.

**Key result**: —

**Notes**:
- Emphasis directed by human: **GPU/PyTorch ↔ TPU/JAX differences in scaling/optimization mechanics**. The source page's centrepiece is a 20-row translation matrix ("Axis | GPU/PyTorch (playbook) | TPU / JAX / XLA | Tuning surface that actually matters on TPU"). Every playbook claim in the Key-claims table is annotated with "Transfers to TPU?" and, where not, the TPU delta.
- GPU-specific sections explicitly flagged as **not transferring**: Section 10 (memory coalescing, tiling, thread coarsening, control divergence, `torch.compile`+Triton) — different programming model; Pallas Mosaic-TPU is our analogue with its own tuning vocabulary.
- The HF Space URL is dynamically rendered; `WebFetch` on the public URL returned a loading shell. The static-asset URL (`nanotron-ultrascale-playbook.static.hf.space/index.html`) served the complete document and all figures.
- **6 hypothesis candidates** surfaced but **not filed** as `hypotheses/*.md` — schema requires a `model:` slug; no model page exists yet. Listed on the source page under `## Gaps & caveats`, to be promoted when the first model page is filed:
  1. Selective activation recomputation via `jax.checkpoint_policies` (Korthikanti 70% / 2.7% claim).
  2. Wire tokamax `ring_attention_kernel` through `dot_product_attention` dispatch (kernel exists, API gap only — Wave 1 finding + this source confirms it).
  3. Zig-Zag Ring Attention on TPU — no implementation found in any ingested codebase; open algorithmic port from Brandon et al. 2023.
  4. TPU-native Pallas kernels for `gated_linear_unit` and `layer_norm` in tokamax — Wave 1 finding that these fall back to XLA reference; this source quantifies the upside category (fused kernels pay off on memory-bound ops).
  5. DeepSeek-V3 tile-scaled FP8 (1×128 activations, 128×128 weights+scales) on v6e MXU.
  6. Expose tokamax splash-attention **backward** block sizes to the autotuner — Wave 1 hidden-tuning-surface finding cross-referenced.
- Concept-page convention in the vault includes a `sources: N` frontmatter integer alongside the `## Sources` H2 section. SCHEMA.md prescribes only the section. My 5 new stubs follow the vault convention (both). Noting the coexistence; not reconciling here.
- Index entry for this source is added alongside the existing Wave 2 rebuild; Concepts section in the index now lists my 5 new stubs explicitly.

## [2026-04-22] ingest-source + stub | Wave 2 — profiling & optimization docs

**Op**: ingest-source (batch) + stub (concept stubs)
**Pages created**: 28 source pages + 72 concept stubs.
- Sources: `wiki/sources/2026-xprof-mcp-tpu-optimization.md`; xprof docs `2026-xprof-{overview-page,trace-viewer,memory-profile,memory-viewer,graph-viewer,utilization-viewer,terminology,hlo-op-stats,hlo-op-profile,framework-op-stats,perf-counters,custom-call-profiling,capturing-profiles,jax-profiling,pytorch-xla-profiling,tensorflow-profiling,docker-deployment,kubernetes-deployment,roofline-model,megascale-stats,megascale-viewer,megascale-viewer-sql}.md`; tokamax docs `2026-tokamax-{supported-ops,basic-usage,splash-attention,autotuning,benchmarking}.md`.
- Concepts: 72 stubs under `wiki/concepts/` grouped as Architecture & hardware (12), Performance metrics & roofline (11), Compiler & HLO (12), Kernels (8), Optimization techniques (7), Parallelism & collectives (7), Inference (5), Profiling (10).
**Pages updated**: `wiki/index.md` (Sources/Codebases/Concepts sections rebuilt; merged with the concurrent `jax-huggingface` ingest).
**Key result**: ~3,475 lines of source content + 72 concept stubs. Wiki now has a working vocabulary — hypothesis candidates can be stated in terms of existing concepts with citations.
**Notes**:
- Six subagents ran in parallel for source ingestion (one per content group); a seventh consolidated concept stubs from their deduplicated recommendations.
- Subagent reports surfaced additional hypothesis candidates beyond Wave 1 findings (not yet filed):
  - **xprof-mcp TPU_OPTIMIZATION**: per-matmul fp32 cast ~17% overhead; int8 shifts v5e critical batch 240→~120→~240; Llama-2-7B decode 8.8× from static-cache; flash attention saves ~32 MB/layer/request at N=4096; selective AC ~+2.7% compute for ~70% activation memory. Most claims are v5e-anchored — generalization to v6e is not pinned down in the source.
  - **tokamax-supported-ops**: `docs/supported_ops.md` lists `dot_product_attention` as GPU-only, but the code ships a TPU Pallas/Mosaic backend — doc is stale. Flagged on the source page.
  - **tokamax-splash-attention / autotuning**: raw docs are 2–3 line stubs; source pages were written from code + basic-usage doc. The autotune backward-pass block-size coverage gap (Wave 1) is now cross-linked from `autotuning.md` and `attention-block-sizes.md`.
  - **xprof-roofline**: the "bytes accessed" definition in arithmetic intensity isn't scoped to a memory tier in the doc; peak FLOPs / bandwidth per TPU generation are pulled from Device Information at runtime and not listed.
  - **xprof-megascale**: `megascale_viewer_sql` has a minor inconsistency between `pt.name` vs `ppt.name` as `device` column across the two example queries — could mislabel rows.
  - **xprof-perf-counters**: doc lists filters/columns but not individual counter semantics — deeper docs/source needed before counter-level hypotheses.
  - **xprof-hlo-op-profile** "wasted time" sort is computed against FLOPs utilization only; will underweight memory-bound ops.
- Concept stubs: 72 created; subagent flagged 4 more worth adding in a follow-up (`hbm-bandwidth`, `ridge-point`, `reduce-scatter`, `trace-me`/TraceMe). Deferred — will add if Wave 3 scaling-book references them in depth.
- No broken markdown links introduced: grep confirmed `reduce-scatter` only appears as prose in `fsdp.md` / `collective-communication.md`, not as a link.
- No schema deviations this wave.
- Concurrency note: the `jax-huggingface` codebase + 4 source pages were ingested by the human (next log entry) while Wave 2 subagents were running; index.md was merged to reflect both. Wave 2 subagents did not see or cross-link to the jax-huggingface pages; Wave 3 (scaling-book) or a dedicated LINT pass should add cross-links from `jax-huggingface-part-{2,3}.md` into the new concept stubs (sharding, tensor-parallelism, static-cache, kv-cache, splash-attention, etc.).

## [2026-04-22] ingest-codebase + ingest-source | jax-huggingface (learning_machine subfolder)

**Op**: ingest-codebase + ingest-source (combined, user-directed option B)
**Pages created**: wiki/codebases/jax-huggingface.md; wiki/sources/2026-jax-huggingface-part-{1,2,3,4}.md
**Pages updated**: wiki/index.md; .gitmodules (added `raw/code/learning-machine` submodule, commit `93328b2`)
**Key result**: 5 pages written. Ingestion scoped to `jax-huggingface/` subfolder of `qihqi/learning_machine`; sibling subprojects (llama_ref, spmd_sharding, torch_pallas, flash_attn_speed, jax_perf, etc.) deferred. First `source/` pages in the wiki — exercised the source template alongside the codebase template.
**Notes**:
- Per-post data points captured in source-page tables: Part 1 v6e 1-chip Llama-2-7B forward 4.37s→13ms; Part 2 8-chip TP 13ms→3.4ms (3.8× cached, blog rounds to 4.3×); Part 3 50-tok decode 130.9s DynamicCache eager → 88.4s StaticCache eager → 14.77s StaticCache+jit (8.9×); Part 4 **A100 GPU, not TPU**: 5.9s→1.07s/image after VAE `methods_to_compile=['decode']` fix.
- **HF API drift flagged:** Part 3 post text's `StaticCache` pytree flattener (`cache.key_cache`/`cache.value_cache`) does NOT match the companion `jax_hg_03.py` (`cache.layers[i].keys`/`.values`). Script is current-HF-API version. Noted on codebase page and Part 3 source page. Candidate observation once a `model/` page exists.
- **Hardware ambiguity in Part 3:** post does not state device for the 130/88/14.77s numbers. Flagged in source-page "Gaps & caveats". Resolving this is a prerequisite before using those numbers as a baseline.
- **Part 4 hardware is A100 GPU.** Kept ingested because the `torchax.compile` / `CompileOptions` / `methods_to_compile` / scheduler-move patterns are TPU-portable, but reported numbers are not. Explicitly flagged on both the codebase page and Part 4 source page.
- Performance-relevant surfaces on the codebase page enumerate 10 concrete anchors (sharding recipe, KV-cache post-prefill sharding, functional_call escape from captured-constants, methods_to_compile override, static_argnames routing, scheduler tensor-move, pytree cookbook, static-arg strategies, default_matmul_precision, profile-capture idiom) — each grounded in a specific file:line.
- No hypotheses / concept stubs filed — no `model/` page yet to attach them to. Per-page "Future hypothesis anchors" sections carry the candidates forward.

## [2026-04-22] ingest-codebase | Wave 1 — seven repos

**Op**: ingest-codebase (batch)
**Pages created**: wiki/codebases/{xprof,xprof-mcp,torchax,tokamax,stablehlo,scaling-book,autoresearch}.md
**Pages updated**: wiki/index.md
**Key result**: 7 codebase parent pages written in parallel. Total 935 lines. Commit SHAs recorded in each page's frontmatter.
**Notes**:
- Per-repo "discuss before writing" step was batched into a single up-front categorization with the human (A=direct ingest, B=methodology, C=book-as-sources) rather than seven separate discussions.
- Scope discipline held: codebase pages map structure; the docs under each repo's `docs/` were deferred to Wave 2 (profiling/optimization) and Wave 3 (scaling-book chapters as sources). No source/concept/hypothesis pages created.
- Noteworthy findings surfaced during ingestion, flagged for follow-up when hypotheses are formulated:
  - **tokamax**: TPU `gated_linear_unit` and `layer_norm` have NO TPU-specific kernel — they silently fall back to the XLA reference. Candidate hypothesis source.
  - **tokamax**: splash-attention backward-pass block sizes (`block_q_dkv`, `block_kv_dkv`, `block_kv_dkv_compute`, `block_q_dq`, `block_kv_dq`) are exposed via `SplashConfig` but NOT surfaced to the autotuner (hidden tuning surface). Note: a Splash Attention kernel is also available upstream in JAX at `jax.experimental.pallas.ops.tpu.splash_attention` — the tokamax copy under `_src/ops/experimental/tpu/splash_attention/` mirrors that implementation, so hypothesis-writers can target either entry point.
  - **tokamax**: `ring_attention_kernel` exists but isn't reachable from `tokamax.dot_product_attention`.
  - **torchax**: `torchax.compile()` modes `dynamo` and `export` raise; only `jax` mode is functional.
  - **scaling-book**: book dated 2025-02-04 → Wave 3 source slugs will use `2025-` prefix, not `2026-`.
- **autoresearch** page uses the reframed H2 title **"Structural surfaces we borrow"** in place of "Performance-relevant surfaces" (the repo is methodology, not TPU-perf content). This is an intentional schema deviation for this single page.

## [2026-04-22] manual | wiki bootstrap

**Op**: manual
**Pages created**: SCHEMA.md, CLAUDE.md, wiki/index.md, wiki/log.md
**Pages updated**: —
**Key result**: —
**Notes**: Bootstrapped autoresearch-oriented schema from scratch. Independent of sibling `tpu_wiki` by design. Loop: sources + codebases + profiles → concepts + models → ranked hypotheses → experiments → observations → revised priors. Next: ingest first codebase and/or file the first model page.

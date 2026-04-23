# Log

## [2026-04-23] ingest-codebase + ingest-source | pallas-forge + Karpathy LLM-wiki backfill

**Op**: ingest-codebase (pallas-forge) + ingest-source (Karpathy LLM-wiki).
**Pages created**:
- `wiki/codebases/pallas-forge.md` — codebase parent page for `linhkid/pallas-forge` at commit `090510b7`. Already used (and found wanting) in gemma4 exp 20; the page makes that context first-class and captures the methodological lessons (isolated-microbench vs in-graph, no-custom_vjp blocker, v5e-only numbers, canonical 3D-grid matmul template).
- `wiki/sources/2026-karpathy-llm-wiki.md` — source page for Karpathy's "LLM Wiki" idea file, the methodological ancestor of this wiki's `SCHEMA.md`. Already locally saved as `raw/sources/2026-karpathy-llm-wiki.md`; backfilling the wiki-side page makes the design lineage explicit.

**Pages updated**:
- `wiki/index.md` — Codebases 8→9 (added pallas-forge at top of list); Sources 33→34 (new "Methodology (1)" subsection); header count bumped to 126 pages.

**Key result**: Two ingestion gaps closed. The pallas-forge page consolidates what the gemma4 program has already learned about the library (exp 20 + exp 33) rather than letting it sit only in experiment pages; the Karpathy-wiki source page documents where this wiki's operating protocol comes from, which matters for future schema edits.

**Notes**:
- **pallas-forge status recap (as captured on the codebase page)**: 3 reference kernels, tuner, roofline, XProf-trace integration — *but* forward-only (no `jax.custom_vjp`), so any training hypothesis that swaps a pallas-forge kernel into a module's `forward` crashes at `jax.value_and_grad` ("Linearization failed to produce known values"). Gemma4 exp 20 is the direct confirmation. Alternative for kernel swaps in training: [tokamax](codebases/tokamax.md) (exposes autodiff-capable `rms_norm` / `layer_norm` / splash-attention paths).
- **Generalizable lesson already in the wiki, now cross-linked**: from the 2026-04-23 Gemma 4 optimization-ceiling analysis (exp 33) — *"Pallas kernels are a net win only when XLA wasn't already exploiting the pattern via fusion"*. pallas-forge's v5e README numbers (3.44× RMSNorm isolated-microbench) do not predict in-graph wins on v6e-4; exp 33 showed an 8.1% regression. The codebase page flags this explicitly under "Performance-relevant surfaces §3 Isolated-microbench ≠ in-graph gain".
- **pallas-forge `TPU_SPECS` gap**: the hardware-preset table for `roofline_chart` covers v4/v5e/v5p only; no v6e entry. Anyone running `roofline_chart` on v6e must supply peak-TFLOPS and peak-HBM-bandwidth directly.
- **Karpathy-wiki source is methodology, not TPU-perf content**: ingested to make the schema's lineage traceable; no hypotheses generated. Its "Connections" section explicitly maps the source's pattern to this wiki's `SCHEMA.md` / `CLAUDE.md` / `index.md` / `log.md` implementations.
- **Ingestion-audit findings recorded for completeness**: (1) the `notes.md` TODO `using git submodule add https://github.com/linhkid/pallas-forge under raw/code, update readme` was already satisfied (`.gitmodules` entry exists, `raw/code/pallas-forge/` is checked out at commit `090510b`); the only missing piece was the wiki page, now filed. (2) `raw/sources/2026-karpathy-llm-wiki.md` was on disk but had no wiki counterpart, now filed. (3) No other unprocessed raw sources or un-documented submodules found.
- **Out of scope for this ingestion** (per `notes.md` lines 28–37, the gemma-4-E4B import + torchax trainer work): that is implementation, not ingestion; the Gemma 4 program already has its own `experiments/gemma4_autoresearch_optimization/` tree with 33 experiments completed and a ceiling analysis filed. Not touched.

## [2026-04-23] analyze | Gemma 4 E4B on v6e-4 — optimization ceiling reached at exp 25

**Op**: analyze (session ceiling synthesis).

**Pages created**:
- [`wiki/analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md`](analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md) — synthesis of the 33-experiment loop. Documents the trajectory (baseline 30,570 → exp 25 33,372, +9.2%), what worked vs what didn't, the step-time decomposition at exp 25, the Pallas-fuses-into-matmul lesson from exp 33, and what would actually move the needle next (hardware scale-up, scan-over-layers Option B, or accepting ceiling).

**Pages updated**:
- `wiki/index.md` — Analyses section populated.

**Key result**: Optimization loop has reached diminishing returns on this hardware/model combo. Trunk stays at exp 25 (33,372 TPS, seq=1024 batch=3 fsdp=4 bf16, splash_pallas + SEQ_MINOR + block=1024 + fused_bwd + bf16 CE + selective remat). Eight experiments since exp 25 (exp 26–33) produced zero further wins: scan-over-layers blocked (5 Gemma-specific issues), tokamax DPA blocked (mosaic_tpu no sliding-window), 2D mesh 2.4× slower at this chip count, Pallas RMSNorm −8.1% due to XLA already fusing it with neighbor matmuls, long-seq and XLA-flag-isolation experiments neutral or dominated.

**Notes**:
- **Generalizable lesson (exp 33)**: Pallas kernels are a net win only when XLA wasn't already exploiting the pattern via fusion. Splash won (exp 8) because XLA can't express online-softmax. RMSNorm loses because XLA already fuses it. Likely Pallas SwiGLU would too — don't build it.
- **Remaining viable paths** (none attempted here): (a) 300-500 LOC scan-over-layers Option B for compile-time compression, (b) hardware scale-up to v6e-8 or v5p-4 where 2D mesh and collective overlap become economic, (c) persistent compile cache (infrastructure only, no TPS).
- **Exp 28 kept but not merged**: +0.9% at seq=2048 b=1 strictly dominated by exp 25's higher-batch throughput. Preserved for long-seq reference.

---

## [2026-04-23] protocol + experiments | Gemma 4 E4B program — program.md formalization + exp2 crash

**Op**: manual (protocol formalization) + run-experiment (exp2, crashed).
**Pages created**:
- `wiki/experiments/gemma4_autoresearch_optimization/program.md` — the agent-facing protocol for this program, adapted from the sibling-wiki TorchTitan autoresearch template but specialized to Gemma 4 E4B / torchax / v6e-4.
- `wiki/experiments/gemma4_autoresearch_optimization/RESULTS.tsv` — tab-separated ledger (gitignored via `wiki/experiments/*/RESULTS.tsv`).
- `wiki/experiments/gemma4_autoresearch_optimization/OBSERVATIONS.md` — skim-and-reason aggregation log; backfilled with baseline + exp1 + exp2 blocks.
- `wiki/experiments/gemma4_autoresearch_optimization/2026-04-23-exp2-pin-out-shardings-rejected.md` — Exp 2 experiment page (crashed/invalid, reverted).

**Pages updated**:
- Program `README.md` — stripped the in-README "Optimization loop procedure" section; replaced with a short "How to start the optimization loop" pointer to `program.md`. History extended for exp2 + the protocol formalization.

**Key result**: **Exp 2 CRASHED** pre-trace with `ValueError: Received incompatible devices` on `weights['lm_head.weight']`. Attempt to pin `out_shardings=(loss_ns, weights_ns, opt_state_ns)` on `jax.jit` to fix the ~150 s step-1 recompile hit a tied-weight-dedup plumbing issue in torchax's `JittableModule` — the tied `lm_head.weight` ↔ `embed_tokens.weight` emits an `out_shardings` leaf whose device list collapses to `[0]` even though the input sharding is full-mesh `[0, 1, 3, 2]`. Reverted the `train.py` change; step-1 recompile remains open. Hypothesis marked `verdict: invalid` (pre-trace crash, nothing measured) — still viable under a different implementation (`dedup_parameters=False` + manual tying, or `with_sharding_constraint` inside the train step).

**Notes**:
- **Protocol formalization**: Karpathy-autoresearch-style discipline now lives in a dedicated `program.md` that defines the fixed bindings (hardware, conda env, trainer path, baseline command, libtpu version, profile/HLO conventions, xprof_mcp server), the architectural invariants (can/cannot-change tables for Gemma 4), the measurement protocol (TPS primary, MFU secondary, median steps 6–15), the ledger + observations schema, the full experiment loop, and accumulated heuristics. Future experiments start from reading `program.md` and tail-ing `OBSERVATIONS.md` / `RESULTS.tsv`.
- **Retroactive backfill**: baseline + exp1 + exp2 are transcribed into `RESULTS.tsv` and `OBSERVATIONS.md` for continuity — these experiments predate the formal protocol but provide the foundation the protocol was written against.
- **libtpu is at 0.0.40, the latest** (verified across PyPI, `libtpu-lts-releases`, `libtpu-nightly-releases`). The exp1 "unknown flag" failure was a name error (real name is `--xla_tpu_overlap_compute_collective_tc`, not `_comms`), not a version issue. Documented the symbol-dump-as-source-of-truth trick in `program.md` heuristics.
- **Discovered gotcha**: tied `lm_head` ↔ `embed_tokens` + torchax `JittableModule.dedup_parameters=True` + HF `load_state_dict(assign=True)` interact in a way that confuses `out_shardings` construction. Logged as a `program.md` heuristic; a future experiment can route around it via `dedup_parameters=False`.
- Step-1 recompile mitigation deferred; next experiment focuses on a structural memory-saving change (per the memory-ceiling rule — baseline hit 95% HBM).

## [2026-04-23] experiment | Gemma 4 E4B — Exp 1: async-collective XLA flags (REFUTED)

**Op**: run-experiment (first optimization-loop cycle on the Gemma 4 program).
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/2026-04-23-exp1-async-collective-flags-rejected.md`.
**Pages updated**: program `README.md` history (baseline-seq-1024 + Exp 1).
**Raw artifacts**: `raw/profiles/2026-04-23-gemma4-loss-confirm/` (20-step seq=1024 baseline, clean loss), `raw/profiles/2026-04-23-gemma4-exp1-async-collectives/` (exp1 trace). Both symlinked into the live xprof instance as `gemma4_baseline_seq1024_20260423` and `gemma4_exp1_async_collectives_20260423`.

**Key result**: **Hypothesis REFUTED.** Enabling `--xla_tpu_enable_latency_hiding_scheduler` + `--xla_tpu_enable_async_collective_fusion` (+ `fuse_all_gather`, `multiple_steps`) via `LIBTPU_INIT_ARGS` regressed steady-state step time **134.4 → 168.3 ms (+25 %)** at seq=1024, batch=1, FSDP=4, v6e-4. Collectives got faster (all-gather −5 ms, all-reduce-scatter −9 ms) but compute-fusion memory traffic blew up (convolution fusion +2.5× bytes, loop fusion +1.9×). Scheduler gained freedom to reorder; it fused collectives but broke compute-order locality. Loss trajectory preserved (bf16-reorder noise only).

**Notes**:
- 20-step loss-descent confirmation (pre-experiment): loss 3.93 → 1.97 over 20 steps at seq=1024; step time stable at 134.4 ± 0.5 ms. Confirms the baseline scaffold is training correctly.
- Ran full autoresearch loop: baseline profile → hypothesis pick (#9) → experiment → profile diff → verdict → file page → pick next hypothesis. First complete cycle on this program.
- **Flag-placement gotcha**: `--xla_tpu_*` flags go in `LIBTPU_INIT_ARGS`, **not** `XLA_FLAGS`. First attempt bombed with `parse_flags_from_env.cc:234 Unknown flags in XLA_FLAGS`. Logged for future experiments — hypothesis-writers should not trust the "XLA flag" name.
- **Libtpu 0.0.40 does not know `--xla_tpu_overlap_compute_collective_comms`.** Dropped before the successful run. Good instance of a flag catalog drifting vs. the installed runtime.
- **xprof-mcp tools used in the loop**: `list_runs`, `get_overview`, `get_top_hlo_ops`, `get_op_profile`, `get_memory_profile`, `get_device_information`. The HLO-op-level before/after diff was the decisive evidence for "refuted" — wall-clock alone would have been ambiguous, but the bytes-accessed explosion made the mechanism clear.
- Hypothesis parked, not retired — at larger effective batch (via remat or NaN-fix at seq=2048) the collective-overlap win may swamp the compute-locality loss. Revisit after hypothesis #6.
- Next natural hypothesis: **fix the step-1 recompile** (explicit `in_shardings` / `out_shardings` on `jax.jit`) — saves ~150 s per experiment run, pure iteration-speed win. Then **#6 selective remat**.

## [2026-04-22] experiment | Gemma 4 E4B baseline (v6e-4, FSDP)

**Op**: run-experiment (baseline infrastructure check)
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/2026-04-22-baseline.md`.
**Pages updated**: program `README.md` (History section updated with first baseline numbers); `wiki/experiments/gemma4_autoresearch_optimization/torchax/{train.py,model/sharding.py,requirements.txt}` (8 targeted fixes to get the scaffold to actually run — enumerated on the baseline page).
**Raw artifacts**: `raw/profiles/2026-04-22-gemma4-baseline/` — xprof trace, steps 5–7 at seq=2048.

**Key result**: First working run. Steady-state **249 ms/step at seq=2048, batch=1, FSDP=4 → ~33k tokens/sec, ~26% MFU** (corrected from an initial `6PT` overestimate of 44% — `P=8B` double-counts Gemma 4's Per-Layer-Embedding lookup tables, which don't participate in matmul FLOPs). Seq-length sweep: **13% MFU @ seq=512, 23% @ seq=1024, 26% @ seq=2048**. Verdict: **supported** on the infrastructure-check hypothesis; **NOT a quality-valid baseline at seq=2048** until the NaN loss is fixed.

**Notes**:
- Hardware was **v6e-4**, not the v6e-8 the scaffold README assumed. FSDP default auto-picked fsdp=4 from `jax.device_count()` — no config change needed.
- **Default sharding strategy flipped TP=8 → FSDP** per user directive. Both paths now available; FSDP is the default. FSDP sharding rule: every ≥2D param shards on its largest dim divisible by `fsdp_size` over a 1D `'fsdp'` axis.
- **Python 3.13.13** env (`gemma4_py313`) works with jax 0.10.0, torch 2.11.0+cpu, torchax 0.0.12 (editable install from the wiki submodule), transformers 5.7.0.dev0 (Gemma 4 only on main), datasets, optax, accelerate.
- **8 scaffold fixes** applied to get from written-but-untested to running; all enumerated in the baseline page's "Scaffold changes applied" table. Most consequential:
  1. `interop._jax_view` → `interop.jax_view` (pytree-map variant; single-value left torch tensors unconverted).
  2. Load `Gemma4ForConditionalGeneration` (not `Gemma4ForCausalLM`) — HF checkpoint is multimodal-only and `ForCausalLM` silently re-inits every weight (name-prefix mismatch against `model.language_model.*`).
  3. Monkey-patch `model.forward` with a text-only path (bypass the multimodal orchestrator's `input_ids[mask] = pad` which is not JIT-traceable).
  4. Apply `final_logit_softcapping=30.0` in the text-only forward (did not fix the seq=2048 NaN, but is semantically required).
  5. Requirements.txt pointed torchax at `pytorch/xla` subdirectory; actual repo is `google/torchax`. Switched to an editable install from the wiki's own submodule (commit `8f957d1`).
- **Correctness issues found by the baseline** (flagged for the next experiment):
  - **NaN loss at seq≥2048** — loss is clean at seq∈{512, 1024}. Likely bf16 attention overflow or a Gemma 4 hybrid-attention mask edge case. Prerequisite for any seq=2048 perf work.
  - **OOM at batch=4, seq=2048** — attention N×N materialized (no flash/splash). Directly motivates hypothesis #1 (Splash Attention).
  - **Step 1 recompiles** — both step 0 and step 1 take ~155 s, step 2+ hits the cache. Likely a sharding-spec / donation mismatch on step-1 inputs. Low-effort follow-up: pass explicit `in_shardings` to `jax.jit`.
- **Arithmetic error noted publicly**: initial MFU claim of 44% was wrong — used `6PT` with `P = headline 8B` instead of `P ≈ non-embedding-matmul params`. Gemma 4's "E4B" headline includes PLE lookups which add params but no matmul FLOPs. User caught this; corrected to ~26% MFU via detailed per-matmul FLOP counting. Keep this in mind for future model-size-related FLOP estimates.
- **Compile time dominates** for short runs: 2 × ~155 s compile vs 2.5 s of useful work at seq=2048 × 8 steady-state steps. Hypothesis #7 (scan-over-layers) should collapse this.

## [2026-04-22] scaffold | Gemma 4 E4B torchax trainer (untested)

**Op**: manual (scaffold code for first execution path of an optimization program)
**Pages created**: 9 files under `wiki/experiments/gemma4_autoresearch_optimization/torchax/` (1,215 lines total):
- `train.py` (439 lines) — fine-tune trainer with profile-step capture.
- `model/sharding.py` (245 lines) — 2D `(dp, tp)` mesh + NeMo-Megatron sharding adapted for Gemma 4 GQA.
- `model/README.md` (118 lines) — config + sharding assumptions, upstream source of truth.
- `data.py` (118 lines) — wikitext loader + fixed-length packer.
- `README.md` (134 lines, includes augmented "Running the trainer" section) — runbook.
- `model/__init__.py` (46 lines) — re-exports `Gemma4Config`, `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, `Gemma4Model` from `transformers`.
- `run.sh` (50 lines) — wrapper setting `XLA_FLAGS` + `LIBTPU_INIT_ARGS`, forwards to `train.py`.
- `config.yaml` (22 lines) — default args.
- `requirements.txt` (43 lines) — deps pinned against torchax commit `8f957d1`.

**Pages updated**: none (wiki markdown was already in place; only trainer code added).

**Key result**: First execution path of the Gemma 4 program is scaffolded. Status marked **UNTESTED** in multiple places — scaffold written from ingested source pages (jax-huggingface part 2 sharding recipe, torchax codebase architecture, xprof capture docs) without running a single step.

**Research findings worth capturing as wiki content later (not yet pages)**:
- **Gemma 4 E4B is public Apache-2.0**, not gated (login still required for HF hub). `config.json` readable at `https://huggingface.co/google/gemma-4-E4B/raw/main/config.json`.
- **Architecture specifics**: 42 layers, hidden=2560, `num_attention_heads=8`, `num_key_value_heads=2`, `head_dim=256`, `intermediate_size=10240`, vocab=262144, sliding_window=512, max_position=131072. "E4B" = effective-4B via Per-Layer Embeddings; **~8B with embeddings**.
- **Novelties vs Gemma 3**: hybrid attention (local SW 512 + global), `num_kv_shared_layers=18` (cross-layer KV sharing), `rope_type=proportional` with `partial_rotary_factor=0.25` on full-attention layers, `final_logit_softcapping=30.0`, `gelu_pytorch_tanh` MLP, `tie_word_embeddings=true`.
- **Multimodal**: E4B ships vision + audio branches. Trainer targets text-only (`Gemma4ForCausalLM` with fallback to `Gemma4ForConditionalGeneration`).
- **Sharding corner case**: `num_kv_heads=2` does NOT divide `tp=8`. Default partitioning therefore **replicates K/V projections** rather than silently dropping parallelism — flagged as a future hypothesis.
- **Canonical class names** (per HF `transformers` main, `transformers_version: 5.5.0.dev0`): `Gemma4Config`, `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, `Gemma4Model`. Transformers ships Gemma 4 in `src/transformers/models/gemma4/`.
- **DeepMind gemma repo** exposes `gm.nn.Gemma4_E4B()` + `gm.ckpts.CheckpointPath.GEMMA4_E4B_IT` — native-JAX reference for the `../jax/` folder when that path is activated.

**Assumptions flagged for baseline-run verification**:
1. `Gemma4ForCausalLM` import works; falls back to `ForConditionalGeneration` if not.
2. HF state-dict key naming matches Gemma-family convention (`q_proj`, `k_proj`, …, `embed_tokens`) — regex-based sharder in `sharding.py` is fragile and should be verified with `print(list(model.state_dict())[:20])`.
3. `num_kv_shared_layers=18` may surface as extra/renamed params not covered by the regex — they default to replicated (conservative).
4. torchax API at commit `8f957d1` matches the calls in `train.py` (`JittableModule`, `interop.jax_view/torch_view`, `enable_performance_mode`, `apply_jax_`, `save_checkpoint`).
5. `wikitext-2-raw-v1` chosen as default (small, fast smoke-test). `wikitext-103-raw-v1` available via flag.
6. HF pytree registration targets `CausalLMOutputWithPast` + `DynamicCache`; `StaticCache` registration deferred to a future decode hypothesis.
7. `final_logit_softcapping=30.0` assumed to be implemented inside HF's forward (Gemma 2/3 precedent).

**Known gaps in the scaffold**:
- `--grad_accum` is parsed but not threaded through the training loop.
- No `with_sharding_constraint` activation annotations inside the forward (relies on GSPMD propagation from weight shardings).
- No checkpoint *load* path — `--checkpoint_dir` only saves.
- Splash-attention swap (program hypothesis #1) not wired yet.
- Optimizer states inherit gradient dtype (bf16 if forward is bf16); no explicit fp32 promotion — a baseline concern.

**Next steps for the human** (runbook):
1. `pip install -r wiki/experiments/gemma4_autoresearch_optimization/torchax/requirements.txt` on a v6e-8 host.
2. `huggingface-cli login` with a token that's accepted the Gemma license.
3. `bash wiki/experiments/gemma4_autoresearch_optimization/torchax/run.sh --steps 5 --profile_steps 3`.
4. File the baseline numbers into `wiki/experiments/gemma4_autoresearch_optimization/<YYYY-MM-DD>-baseline.md` (the first dated experiment page).

## [2026-04-22] file-program | Gemma 4 E4B — TPU autoresearch optimization

**Op**: manual (file a new optimization program)
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/README.md` — program page for `google/gemma-4-E4B` on TPU v6e via torchax.
**Pages updated**: `wiki/index.md` — Models section (0 → 1).

**Key result**: First `model/` analogue filed. 16 open hypotheses consolidated from Wave 1/2 findings, the xprof-mcp TPU_OPTIMIZATION guide, and the Ultra-Scale Playbook — now have a place to attach. Baseline not yet captured; hypothesis #0 in the ranked list is "capture baseline profile."

**Notes**:
- **Intentional schema deviation**: SCHEMA.md specifies `wiki/models/<slug>.md` for model-under-optimization pages and `wiki/experiments/<YYYY-MM-DD>-<slug>.md` (flat) for experiments. This program uses a **nested folder** `wiki/experiments/gemma4_autoresearch_optimization/` that co-locates the program README (functions as the model page), the dated experiment files (schema-conformant names inside the folder), and optionally local scripts/code. Rationale: a long-running optimization program generates many related files and benefits from being namespaced together; the flat experiments/ directory would make it hard to find "everything about Gemma 4" vs. "everything about the next model."
- This deviation is the **second** intentional one in the wiki (first was the `autoresearch` codebase page's reframed "Structural surfaces we borrow" H2). If it works, the next SCHEMA.md edit should codify `wiki/experiments/<program-slug>/` as a permitted layout for multi-experiment programs, with the README.md inside doubling as the `model` page.
- **Code location decision (2026-04-22)**: option (b) — inside the program folder — selected by human. Further refined: **split into two sibling subfolders by execution path** rather than one `code/` folder:
  - `wiki/experiments/gemma4_autoresearch_optimization/torchax/` — primary, Gemma 4 via torchax.
  - `wiki/experiments/gemma4_autoresearch_optimization/jax/` — secondary, native-JAX port (port-equivalence discipline required: must reproduce torchax-baseline outputs within bf16 tolerance before perf numbers count).
  Each subfolder has its own README documenting conventions (dated copies for divergent scripts, relative-path references from experiment pages, binaries go to `raw/profiles/`, not these folders). The program's top-level README links both. Next SCHEMA.md update should codify `wiki/experiments/<program-slug>/` folders with execution-path-named subfolders for code as a permitted layout.
- Hypotheses are listed on the program page but **not filed as `wiki/hypotheses/*.md` individually** — the program page serves as the consolidated ranked list. Once individual hypotheses become in-flight experiments, each will be promoted to `wiki/hypotheses/<slug>.md` per schema.
- **Gemma 4 E4B**: user confirmed the identifier is correct (per https://huggingface.co/google/gemma-4-E4B). Claude's training cutoff predates this model's release; treating it as a black-box target with Gemma-family architecture (GQA, SwiGLU, RMSNorm) until the baseline ingest confirms exact config.

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

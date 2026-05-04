# Autoresearch driver — optimize a model on a hardware tier

**How to use:** open Claude Code in this repo on your Mac and paste the prompt below (everything between the rules). Replace the angle-bracket placeholders. Claude will take it from there.

The prompt is generic over `<model>`, `<hardware>`, `<target metric>`, `<workload>`, and `<budget>`. Every other piece of state — the wiki, the bench tooling, the host registry, the per-turn contracts — Claude reads from the repo.

Examples this is designed to handle:
- `<model>=openai/gpt-oss-20b`, `<hardware>=h100`, `<target metric>=ShareGPT output_tok_s at c=256`, `<workload>=multi-turn-agentic`, `<budget>=2 hours`
- `<model>=Qwen/Qwen2.5-32B-Instruct`, `<hardware>=h200`, `<target metric>=decode tok/s at c=128`, `<workload>=chain-of-thought`, `<budget>=4 hours`

---

You are the autoresearch driver for our serving-optimization wiki.

# Goal

Find the best vLLM configuration for **`<model>`** on **`<hardware>`** that maximizes **`<target metric>`** under the **`<workload>`** profile, while preserving accuracy (lossless gate ≤0.1% deviation on GSM8K / HellaSwag / MMLU). Run experiments autonomously, commit each as a wiki experiment page, stop when the frontier hasn't moved for 3 rounds or budget hits **`<budget>`**.

# Where state lives

- `SCHEMA.md` — page types, frontmatter, operations. The contract.
- `wiki/index.md` — ranked open hypotheses (read first), supported list, retired list.
- `wiki/codebases/vllm-tune.md` — the bench infrastructure surface (knobs, scripts, entry points).
- `wiki/sources/2026-04-gptoss-*.md` — load-bearing priors. Read before proposing anything new.
- `wiki/experiments/2026-05-04-gptoss20b-h100-opt.md` — canonical example of a schema-compliant experiment writeup. Use as the writing template.
- `bench/scripts/experiments/<study>/run_matrix.sh` — config-matrix orchestrator. Already self-contained; rides along with the rsync.
- `bench/scripts/benchmark/sweep_api_providers_evalscope.py` — load generator (evalscope-driven).
- `prompts/{pick,run}.md` — the per-turn contracts. Don't deviate from the line-based stdout markers.
- `.hosts.toml` — registered GPU hosts with SSH details (gitignored, your local file).

# How the loop works (don't re-implement any of this)

1. `./run_loop.sh --rounds N --tag <yourname> --hosts <host-name>`
2. Each round: **PICK** turn (read index, choose top schedulable hypothesis) → **RUN** turn (rsync wiki to box, drive `bench/`, rsync results back, write experiment page) → lint + git commit.
3. Mac is the only authoritative source. The GPU box is a stateless worker — every round rsyncs a fresh wiki copy to it.
4. Each round commits one experiment page to `wiki/experiments/`. Already-committed rounds aren't repeated on rerun, so the loop is safely idempotent.

# Your job

1. **Audit the open ranked list.** Read `wiki/index.md`. If a hypothesis exists that already targets `<model>` + `<hardware>` + `<workload>`, run it. If not, write one before launching the loop.
2. **Write hypotheses by reading priors first** — `wiki/sources/`, prior experiments, the codebase page. Each hypothesis must be a single falsifiable claim with a quantitative pass threshold tied to a named baseline (e.g., "OPT round 7 measured 16,192 sharegpt tok/s; LEAN should beat that by ≥5%"). Schema-compliant frontmatter (model, engine, workload, hardware, status=open, expected_gain, confidence, effort, origin).
3. **Drive the loop** with `./run_loop.sh --rounds N --tag <yourname> --hosts <host>`. After each round, read the experiment page that just committed. Extract observations worth filing as `wiki/observations/<slug>.md` if a finding may recur.
4. **Adapt as you learn.** If a surprising result lands (a config wins for an unexpected reason, or a hypothesis is decisively refuted), write a new hypothesis testing the surprise and bump it in the ranked list before the next round.
5. **Stop when** any of: (a) 3 consecutive rounds without a frontier improvement on `<target metric>`, (b) `<budget>` exhausted, (c) the user interrupts.
6. **At the end**, write `wiki/analyses/<date>-<topic>.md` summarizing the frontier, the configurations explored, what won, and what to ship. If confidence is high, propose a `wiki/releases/<model>-<hardware>-<date>.md` page with the winning Dockerfile, image tag, and pinned config.

# Constraints

- **Never invent infrastructure.** The `bench/` scripts and `run_matrix.sh` are authoritative — do not write a new benchmark harness or skirt the orchestrator.
- **Never modify model weights or fine-tune anything.** Configuration knobs only.
- **Every `supported` verdict requires** (a) measurable improvement beyond noise, (b) no regression on tracked metrics, (c) lossless gate pass.
- **Never commit to `wiki/` while a round is running** — the loop holds the lock and will halt on conflict.
- **Don't push to GitHub** unless explicitly asked.
- **Don't touch `~/vllm-tune/` on the box** — we deliberately self-contained the bench tooling under `bench/`. The wiki ships everything the box needs.
- **One round at a time per host.** No parallel dispatch yet (slice 9+ work).

# What a teammate sees when this works

After the run terminates:
- Several new commits on the branch, one per round (`round N: <hypothesis> on <host> (<verdict>)`).
- `wiki/index.md` ranked list updated: hypotheses moved to `### Supported` / `### Refuted`.
- `wiki/experiments/<date>-*.md` pages with full results tables, metrics, observations, next-hypothesis suggestions.
- `raw/benchmarks/<run_slug>/` (gitignored) with the per-cell evalscope dumps and Prometheus metrics.
- A final `wiki/analyses/<date>-<topic>.md` ranking the configs explored and naming the winner.
- (Optional) a `wiki/releases/<model>-<hardware>-<date>.md` page with a ready-to-build Dockerfile.

# Now

Tell me the goal in one line if you haven't already (`<model>` + `<hardware>` + `<target metric>` + `<workload>` + `<budget>`). I'll audit the wiki, propose any missing hypotheses, and start the loop.

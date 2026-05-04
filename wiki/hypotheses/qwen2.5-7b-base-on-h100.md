---
title: "Qwen2.5-7B BASE establishes per-replica baseline on 1× H100 80 GB"
type: hypothesis
tags: [serving, qwen, h100, baseline, generalization-test]
model: qwen2.5-7b
engine: vllm
workload: multi-turn-agentic
hardware: h100
status: inconclusive
expected_gain: "establish per-replica floor (decode/prefill/sharegpt at 256/1024/512); no claim about which knob will help"
confidence: high
effort: S
origin: human
created: 2026-05-04
updated: 2026-05-04
---

## Statement

Stock vLLM 0.19.0 (no extra flags except `--gpu-memory-utilization 0.9`) serves [Qwen2.5-7B-Instruct](../models/qwen2.5-7b.md) on a single H100 80 GB SXM and completes a per-replica concurrency sweep across the three reference workloads (decode @ c=256, prefill @ c=1024, sharegpt @ c=512) with **100% request success** and non-zero `output_tok_s` on each.

This is a calibration round, not a knob test — its job is to establish the floor against which all subsequent ablations measure their delta. Verdict is `supported` if all three cells return `success_pct=100.0`, `refuted` if any cell returns `success_pct<99` or `output_tok_s=0`.

## Rationale

This is the **first non-gpt-oss target** in the wiki — the autoresearch program needs to validate that the BASE-first → ablate → synthesize → validate pattern generalizes beyond a single model family.

Per-replica concurrency choices vs the gpt-oss-20B baseline (round 7 OPT measured c=128/512/256):

| Workload | gpt-oss-20B | Qwen 2.5 7B | Why |
|---|---|---|---|
| decode | 128 | **256** | Qwen-7B's smaller weights (14 GB vs 39 GB) leave ~62 GB free for KV vs ~40 GB; 2× the in-flight sequences fit |
| prefill | 512 | **1024** | Same — input prefill is bound by HBM not compute on this rig at <2K context |
| sharegpt | 256 | **512** | Same — bandwidth-headroom doubles |

`gpu_memory_utilization=0.9` (vLLM 0.19 default) is safe on this combination: weights ~14 GB + CUDA context + graph pool ~5 GB + KV cache filling the remainder. No need for the 0.85 dropdown that gpt-oss-20B required to dodge the v0.18 sampler-warmup OOM (see [gpt-oss-20B H100 OOM finding](../sources/2026-04-gptoss-20b-h100-oom.md)) — that bug is fixed in v0.19, and Qwen's KV per-token is much smaller than gpt-oss-20B's anyway.

## Proposed experiment

**Important**: `bench/scripts/experiments/gptoss20b_config_search/run_matrix.sh` is **hardcoded to gpt-oss-20B** and is NOT usable here. The RUN turn must take the fallback path documented in [`prompts/run.md`](../../prompts/run.md):

1. SSH to the host:
   ```bash
   sudo docker run -d --name qwen7b-bench --gpus all --ipc=host --shm-size 16g \
       -p 8000:8000 -v /srv/models:/models:ro \
       --entrypoint python3 \
       vllm/vllm-openai:v0.19.0 \
       -m vllm.entrypoints.openai.api_server \
       --model /models/Qwen/Qwen2.5-7B-Instruct \
       --served-model-name qwen \
       --host 0.0.0.0 --port 8000 \
       --tensor-parallel-size 1 \
       --trust-remote-code \
       --gpu-memory-utilization 0.9
   ```
2. Wait for `/health` (≤5 minutes; cold start should be ~30-60s for this size).
3. Run three evalscope cells via `bench/scripts/benchmark/sweep_api_providers_evalscope.py`:
   - decode @ c=256 (256 in / 4096 out, 256 prompts)
   - prefill @ c=1024 (4096 in / 512 out, 1024 prompts)
   - sharegpt @ c=512 (~175 in / ~2K out from sharegpt dataset, 512 prompts)
4. Snapshot `/metrics` to capture KV cache utilization and any spec decode counters (will be 0 in BASE — useful as a control row).
5. Tear down container, rsync `raw/benchmarks/<run_slug>/` back to Mac, write the experiment page.

## Risks

- **Tokenizer**: Qwen uses tiktoken-style BPE, vocab 152K. The evalscope sweep uses the model's own tokenizer at the URL, so this should be transparent — but flag if `output_tok_s` numbers look wildly out of band (Qwen's vocab is smaller than gpt-oss's 201K).
- **Trust-remote-code**: Qwen requires `--trust-remote-code` for some chat templates. Container launch must include it.
- **No Eagle3 draft**: There's no public Qwen2.5-7B Eagle3 speculator. So all spec-related ablations are NOT viable for this model unless we either (a) train one, (b) use ngram speculation (`--speculative-config '{"method":"ngram","prompt_lookup_max":4,"prompt_lookup_min":1,"num_speculative_tokens":3}'`), or (c) use an off-the-shelf small-model draft (Qwen2.5-1.5B-Instruct or 0.5B).
- **Lossless gate**: not yet wired into the bench (slice 10). For this round, mark "lossless gate not yet validated" in the verdict reasoning.

## Result

[2026-05-04 experiment](../experiments/2026-05-04-qwen2.5-7b-base-on-h100.md) — verdict `inconclusive`. Decode @ c=256 (7,293 output tok/s) and sharegpt @ c=512 (8,279 output tok/s) ran cleanly with 100% success. Prefill @ c=1024 failed with a client-side `Errno 24: Too many open files` ulimit cap; the BASE config itself was never actually stress-tested at that concurrency. Re-run with `ulimit -n 65536` (or drop concurrency to c=512) to resolve.

## See also

- [Qwen2.5-7B model page](../models/qwen2.5-7b.md)
- [vLLM engine page](../engines/vllm.md)
- [H100 hardware page](../hardware/h100.md)
- [Round 7 OPT experiment](../experiments/2026-05-04-gptoss20b-h100-opt.md) — methodology template

## Sources

- [Qwen2.5 model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [vLLM CLI flags](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

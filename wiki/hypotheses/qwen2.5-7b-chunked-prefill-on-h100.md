---
title: "Qwen2.5-7B + chunked-prefill on 1× H100 — prefill TTFT reduction"
type: hypothesis
tags: [serving, qwen, h100, chunked-prefill, ablation]
model: qwen2.5-7b
engine: vllm
workload: multi-turn-agentic
hardware: h100
status: open
expected_gain: "−30 to −50% prefill TTFT p50 vs BASE; ≤noise change in throughput"
confidence: medium
effort: S
origin: human
created: 2026-05-04
updated: 2026-05-04
---

## Statement

Adding `--enable-chunked-prefill` to the [Qwen2.5-7B BASE config](qwen2.5-7b-base-on-h100.md) on 1× H100 80 GB SXM delivers **≥30% reduction in prefill TTFT p50 at c=1024** vs BASE (78.7 s). The trade-off is at most a 5% regression in decode `output_tok_s` (interleaving prefill chunks with decode steps adds scheduler overhead).

Pass: prefill TTFT p50 ≤ 55 s (78.7 × 0.70) AND decode output_tok_s ≥ 5,674 (5,973 × 0.95).

## Rationale

[BASE rerun](../experiments/2026-05-04-qwen2.5-7b-base-rerun.md) measured prefill TTFT p50 = 78.7 s at c=1024. With 4096-input prompts and a batch ceiling of `max_num_batched_tokens=8192` (vLLM 0.19 default), the scheduler can only prefill 2 prompts per iteration. 1024 prompts → ~512 iterations to clear the prefill queue → most requests sit idle in the queue.

`--enable-chunked-prefill` lets vLLM split a long prefill across multiple iterations and **interleave it with ongoing decode steps**. This means new requests start producing tokens (TTFT measured) while their prompt is still being prefilled. The expected TTFT improvement is roughly `1 / num_prefill_chunks`, so at default chunk size we'd expect ≥30% reduction.

Decode-only workloads see no benefit (no prefill interleaving), and may see minor regression from scheduler overhead.

## Proposed experiment

```bash
sudo docker run --rm --name qwen7b-bench --gpus all --ipc=host --shm-size 16g \
    -p 8000:8000 -v /srv/models:/models:ro \
    --entrypoint python3 vllm/vllm-openai:v0.19.0 \
      -m vllm.entrypoints.openai.api_server \
      --model /models/Qwen/Qwen2.5-7B-Instruct \
      --served-model-name qwen \
      --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size 1 \
      --trust-remote-code \
      --gpu-memory-utilization 0.9 \
      --enable-chunked-prefill
```

3-cell sweep at the same concurrencies as BASE (decode @ c=256, prefill @ c=1024, sharegpt @ c=512). Compare prefill TTFT p50 against BASE's 78.7s, decode tok/s against 5,973, sharegpt tok/s against 8,282.

## Risks

- **vLLM 0.19.0 may already enable chunked prefill by default for some configs.** Verify in the boot log: `Chunked prefill is enabled` should appear if active. If BASE was already running with it, this hypothesis is a no-op and should be marked `inconclusive`.
- **Prefill scheduler overhead at c=1024 is real** — chunking adds scheduling decisions per iteration. The decode regression upper bound (5%) is a guess; actual could be more.
- **TTFT measurement subtlety**: TTFT is measured from request submission to first generated token. With chunked prefill and many concurrent prefills, the "first token" event arrives sooner per request — but inter-token latency may rise during the prefill-decode mixing phase.

## See also

- [BASE baseline](qwen2.5-7b-base-on-h100.md) (supported)
- [vLLM engine page](../engines/vllm.md) (chunked prefill description in tunable knobs)

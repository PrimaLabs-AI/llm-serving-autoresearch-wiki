#!/usr/bin/env python3
"""Benchmark harness for LLM serving engines.

Launches a serving engine with a given config, runs a workload against it,
and writes structured JSON results to raw/benchmarks/<date>-<slug>/.

Usage:
  python benchmark_harness.py \
    --engine vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --workload multi-turn-agentic \
    --config '{"max_num_seqs": 128, "enable_prefix_caching": true}' \
    --output-dir raw/benchmarks/2026-04-29-prefix-cache-vllm

Supported engines: vllm, sglang, tensorrt-llm
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Engine launch commands
# ---------------------------------------------------------------------------

ENGINE_COMMANDS = {
    "vllm": {
        "base": [
            "python", "-m", "vllm.entrypoints.openai.api_server",
        ],
        "flag_map": {
            "max_num_seqs": "--max-num-seqs",
            "max_num_batched_tokens": "--max-num-batched-tokens",
            "gpu_memory_utilization": "--gpu-memory-utilization",
            "enable_prefix_caching": "--enable-prefix-caching",
            "enable_chunked_prefill": "--enable-chunked-prefill",
            "block_size": "--block-size",
            "tensor_parallel_size": "--tensor-parallel-size",
            "pipeline_parallel_size": "--pipeline-parallel-size",
            "data_parallel_size": "--data-parallel-size",
            "quantization": "--quantization",
            "dtype": "--dtype",
            "enforce_eager": "--enforce-eager",
            "swap_space": "--swap-space",
        },
        "health_url": "http://localhost:8000/health",
    },
    "sglang": {
        "base": [
            "python", "-m", "sglang.launch_server",
        ],
        "flag_map": {
            "tp": "--tp",
            "dp": "--dp",
            "pp": "--pp",
            "mem_fraction_static": "--mem-fraction-static",
            "chunk_prefill_size": "--chunk-prefill-size",
            "enable_overlap_schedule": "--enable-overlap-schedule",
            "enable_dp_attention": "--enable-dp-attention",
            "disable_cuda_graph": "--disable-cuda-graph",
            "quantization": "--quantization",
            "dtype": "--dtype",
        },
        "health_url": "http://localhost:30000/health",
    },
    "tensorrt-llm": {
        "base": [
            "python", "-m", "tensorrt_llm.commands.run_server",
        ],
        "flag_map": {
            "tensor_parallel": "--tensor-parallel",
            "pipeline_parallel": "--pipeline-parallel",
            "dtype": "--dtype",
            "max_batch_size": "--max-batch-size",
            "max_seq_len": "--max-seq-len",
            "tokens_per_block": "--tokens-per-block",
        },
        "health_url": "http://localhost:8000/health",
    },
}


def build_launch_command(engine: str, model: str, config: dict) -> list[str]:
    """Build the engine launch command from engine name, model, and config dict."""
    spec = ENGINE_COMMANDS[engine]
    cmd = spec["base"] + ["--model", model]

    for key, value in config.items():
        flag = spec["flag_map"].get(key)
        if flag is None:
            print(f"WARNING: unknown config key '{key}' for engine '{engine}', skipping")
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    return cmd


# ---------------------------------------------------------------------------
# Workload definitions — map workload slug to benchmark parameters
# ---------------------------------------------------------------------------

WORKLOADS = {
    "multi-turn-agentic": {
        "description": "Multi-turn agent loop with growing context",
        "params": {
            "num_turns": 5,
            "input_length_turn1": 512,
            "input_growth_per_turn": 300,
            "output_length_per_turn": 200,
            "concurrency_levels": [16, 32, 64, 128],
        },
    },
    "parallel-tool-use": {
        "description": "Burst of parallel requests sharing long prefix",
        "params": {
            "shared_prefix_length": 7500,
            "output_length": 64,
            "parallel_per_batch": 32,
            "concurrency_levels": [32, 64, 128, 256],
        },
    },
    "long-context-rag": {
        "description": "Long input, short output (prefill-heavy)",
        "params": {
            "input_length": 32000,
            "output_length": 256,
            "concurrency_levels": [4, 8, 16, 32, 64],
        },
    },
    "chain-of-thought": {
        "description": "Short input, long output (decode-heavy)",
        "params": {
            "input_length": 512,
            "output_length": 4096,
            "concurrency_levels": [8, 16, 32, 64, 128],
        },
    },
    "structured-output": {
        "description": "JSON-constrained decoding",
        "params": {
            "input_length": 1024,
            "output_length": 256,
            "constrained_decoding": True,
            "concurrency_levels": [32, 64, 128, 256],
        },
    },
}


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def wait_for_server(health_url: str, timeout: int = 300):
    """Wait for the serving engine to become healthy."""
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=5):
                print(f"Server healthy at {health_url}")
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(2)
    raise TimeoutError(f"Server did not become healthy within {timeout}s at {health_url}")


def run_vllm_benchmark(model: str, workload: str, params: dict, concurrency: int,
                       base_url: str = "http://localhost:8000") -> dict:
    """Run vLLM's benchmark_serving.py against the live server."""
    cmd = [
        "python", "-m", "vllm.benchmark.benchmark_serving",
        "--backend", "vllm",
        "--base-url", base_url,
        "--model", model,
        "--num-prompts", str(max(100, concurrency * 4)),
        "--request-rate", "inf",
        "--concurrency", str(concurrency),
    ]

    # Add synthetic workload parameters
    if "input_length" in params:
        cmd.extend(["--input-len", str(params["input_length"])])
    if "output_length" in params:
        cmd.extend(["--output-len", str(params["output_length"])])

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    return {
        "command": cmd,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "concurrency": concurrency,
    }


def run_sglang_benchmark(model: str, workload: str, params: dict, concurrency: int,
                         base_url: str = "http://localhost:30000") -> dict:
    """Run SGLang's bench_serving against the live server."""
    cmd = [
        "python", "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--base-url", base_url,
        "--model", model,
        "--num-prompts", str(max(100, concurrency * 4)),
        "--request-rate", "inf",
        "--concurrency", str(concurrency),
    ]

    if "input_length" in params:
        cmd.extend(["--input-len", str(params["input_length"])])
    if "output_length" in params:
        cmd.extend(["--output-len", str(params["output_length"])])

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    return {
        "command": cmd,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "concurrency": concurrency,
    }


BENCHMARK_RUNNERS = {
    "vllm": run_vllm_benchmark,
    "sglang": run_sglang_benchmark,
    # tensorrt-llm uses Triton's perf_analyzer — add when needed
}


# ---------------------------------------------------------------------------
# Parse benchmark output into structured metrics
# ---------------------------------------------------------------------------

def parse_metrics(raw_output: str, engine: str) -> dict:
    """Extract key metrics from benchmark tool stdout.

    This is a best-effort parser — engine benchmark tools print tables
    to stdout in slightly different formats. Returns whatever it can extract.
    """
    metrics = {}

    for line in raw_output.splitlines():
        line = line.strip()
        # Common patterns across engines
        if "Throughput" in line and "requests/s" in line.lower():
            try:
                metrics["throughput_req_s"] = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "Throughput" in line and "tokens/s" in line.lower():
            try:
                metrics["throughput_tokens_s"] = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "TTFT" in line and "mean" in line.lower():
            try:
                metrics["ttft_mean_ms"] = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "TTFT" in line and "p99" in line.lower():
            try:
                metrics["ttft_p99_ms"] = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "TPOT" in line or "Time per output token" in line:
            try:
                metrics["tpot_mean_ms"] = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "E2E" in line or "End-to-end" in line or "Total latency" in line:
            try:
                metrics["e2e_mean_ms"] = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM serving engine benchmark harness")
    parser.add_argument("--engine", required=True, choices=ENGINE_COMMANDS.keys(),
                        help="Serving engine to benchmark")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--workload", required=True, choices=WORKLOADS.keys(),
                        help="Workload profile to run")
    parser.add_argument("--config", default="{}",
                        help="Engine config as JSON string (e.g. '{\"max_num_seqs\": 128}')")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for results (e.g. raw/benchmarks/2026-04-29-slug)")
    parser.add_argument("--launch-server", action="store_true",
                        help="Launch the server before benchmarking (otherwise connect to existing)")
    parser.add_argument("--skip-server", action="store_true",
                        help="Skip server launch and health check (server already running)")
    parser.add_argument("--concurrency-levels", type=int, nargs="*",
                        help="Override workload's default concurrency levels")
    args = parser.parse_args()

    config = json.loads(args.config)
    workload_def = WORKLOADS[args.workload]
    params = workload_def["params"]

    concurrency_levels = args.concurrency_levels or params.get("concurrency_levels", [64])

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    server_proc = None

    # Launch server if requested
    if args.launch_server and not args.skip_server:
        launch_cmd = build_launch_command(args.engine, args.model, config)
        print(f"Launching server: {' '.join(launch_cmd)}")
        server_proc = subprocess.Popen(launch_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        spec = ENGINE_COMMANDS[args.engine]
        wait_for_server(spec["health_url"])
    elif not args.skip_server:
        print("Connecting to existing server (use --launch-server to auto-launch)")

    # Save benchmark config
    benchmark_config = {
        "engine": args.engine,
        "model": args.model,
        "workload": args.workload,
        "engine_config": config,
        "workload_params": params,
        "concurrency_levels": concurrency_levels,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(benchmark_config, f, indent=2)

    # Run benchmarks at each concurrency level
    runner = BENCHMARK_RUNNERS.get(args.engine)
    if runner is None:
        print(f"ERROR: no benchmark runner for engine '{args.engine}'")
        print("Supported engines with runners:", list(BENCHMARK_RUNNERS.keys()))
        sys.exit(1)

    all_results = []
    all_metrics = []

    for conc in concurrency_levels:
        print(f"\n--- Concurrency level: {conc} ---")
        result = runner(args.model, args.workload, params, conc)
        all_results.append(result)

        metrics = parse_metrics(result["stdout"], args.engine)
        metrics["concurrency"] = conc
        all_metrics.append(metrics)

        print(f"  Extracted metrics: {json.dumps(metrics, indent=2)}")

    # Save raw results
    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save parsed metrics summary
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Engine:    {args.engine}")
    print(f"Model:     {args.model}")
    print(f"Workload:  {args.workload}")
    print(f"Config:    {json.dumps(config)}")
    print()
    print(f"{'Conc':>6} | {'Req/s':>10} | {'Tok/s':>10} | {'TTFT mean':>10} | {'TTFT p99':>10} | {'TPOT mean':>10}")
    print("-" * 70)
    for m in all_metrics:
        print(f"{m.get('concurrency', '?'):>6} | "
              f"{m.get('throughput_req_s', 'N/A'):>10} | "
              f"{m.get('throughput_tokens_s', 'N/A'):>10} | "
              f"{m.get('ttft_mean_ms', 'N/A'):>10} | "
              f"{m.get('ttft_p99_ms', 'N/A'):>10} | "
              f"{m.get('tpot_mean_ms', 'N/A'):>10}")

    print(f"\nResults saved to: {output_dir}")

    # Shut down server if we launched it
    if server_proc is not None:
        print("Shutting down server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    main()

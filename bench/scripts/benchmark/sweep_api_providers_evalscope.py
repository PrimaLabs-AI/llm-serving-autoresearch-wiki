#!/usr/bin/env python3
"""Client-side concurrency sweep — evalscope-driven.

Drop-in replacement for sweep_api_providers.py that uses `evalscope perf` as
the load generator instead of a hand-rolled aiohttp client. This matches the
tool the rest of the repo uses for every other benchmark
(scripts/benchmark/run_max_throughput_dual.sh, run_sharegpt_specdec.sh, etc.)
and is the tool the published API-provider numbers in
docs/gptoss-120b-api-provider-benchmark.md were produced with.

Per cell we shell out to:

    evalscope perf --api openai --url <chat_url> --model <model>
                   --tokenizer-path <tokenizer>
                   --dataset {random,custom} ...
                   -n <n> --parallel <concurrency>
                   --outputs-dir <per_cell_dir> --no-timestamp --stream

then read benchmark_summary.json + benchmark_percentile.json from the cell's
output directory and append one row matching the existing CSV schema:

    provider,workload,concurrency,n,succeed,total,success_pct,
    output_tok_s,req_s,
    ttft_{avg,p50,p90,p95,p99},
    tpot_{avg,p50,p90,p95,p99},
    itl_{avg,p50,p90,p95,p99},
    e2e_{avg,p50,p90,p95,p99},
    avg_input_tokens,avg_output_tokens,per_req_tok_s_p50

Typical run:

    export FIREWORKS_API_KEY=...
    export TOGETHER_API_KEY=...
    python scripts/benchmark/sweep_api_providers_evalscope.py \
        --self-urls http://MI300X_HOST:8080 \
        --providers self fireworks together \
        --workloads decode prefill sharegpt \
        --concurrency 1 2 4 8 16 32 64 128 256 512 1024 2048 \
        --out results/api_sweep/sweep_$(date +%Y%m%d_%H%M%S).csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_SHAREGPT = PROJECT_ROOT / "data" / "datasets" / "sharegpt_prompts.txt"
DEFAULT_TOKENIZER = PROJECT_ROOT / "data" / "models" / "openai" / "gpt-oss-120b"

PROVIDER_DEFAULTS = {
    "fireworks": {
        "chat_url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "model": "accounts/fireworks/models/gpt-oss-120b",
        "env_key": "FIREWORKS_API_KEY",
    },
    "together": {
        "chat_url": "https://api.together.xyz/v1/chat/completions",
        "model": "openai/gpt-oss-120b",
        "env_key": "TOGETHER_API_KEY",
    },
    "self": {
        "model": "gptoss",
        "env_key": None,
    },
}

WORKLOADS = {
    "decode":   {"in": 256,  "out": 4096, "dataset": "random"},
    "prefill":  {"in": 4096, "out": 512,  "dataset": "random"},
    "sharegpt": {"in": None, "out": 4096, "dataset": "custom"},
}

DEFAULT_CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

CSV_HEADER = [
    "provider", "workload", "concurrency", "n", "succeed", "total", "success_pct",
    "output_tok_s", "req_s",
    "ttft_avg", "ttft_p50", "ttft_p90", "ttft_p95", "ttft_p99",
    "tpot_avg", "tpot_p50", "tpot_p90", "tpot_p95", "tpot_p99",
    "itl_avg", "itl_p50", "itl_p90", "itl_p95", "itl_p99",
    "e2e_avg", "e2e_p50", "e2e_p90", "e2e_p95", "e2e_p99",
    "avg_input_tokens", "avg_output_tokens", "per_req_tok_s_p50",
]

# Sentinel values for failed cells (match the schema convention of the old sweep).
FAILED_THROUGHPUT = "-1"
FAILED_LATENCY = ""
FAILED_E2E = "-1"


@dataclass
class CellResult:
    provider: str
    workload: str
    concurrency: int
    n: int
    succeed: int
    total: int
    summary: dict
    percentiles: list[dict]

    def to_csv_row(self) -> list[str]:
        total = self.total or 1
        success_pct = round(100 * self.succeed / total, 1)

        if self.succeed == 0 or not self.summary:
            row = [
                self.provider, self.workload, self.concurrency, self.n,
                self.succeed, self.total, success_pct,
                FAILED_THROUGHPUT, FAILED_THROUGHPUT,
                FAILED_THROUGHPUT, FAILED_LATENCY, FAILED_LATENCY, FAILED_LATENCY, FAILED_LATENCY,
                FAILED_THROUGHPUT, FAILED_LATENCY, FAILED_LATENCY, FAILED_LATENCY, FAILED_LATENCY,
                FAILED_THROUGHPUT, FAILED_LATENCY, FAILED_LATENCY, FAILED_LATENCY, FAILED_LATENCY,
                FAILED_E2E, FAILED_LATENCY, FAILED_LATENCY, FAILED_LATENCY, FAILED_LATENCY,
                FAILED_THROUGHPUT, FAILED_THROUGHPUT, FAILED_LATENCY,
            ]
            return [str(c) for c in row]

        s = self.summary
        # Percentile rows are keyed by "10%", "50%", "90%", ...; build a lookup.
        by_pct = {p["Percentiles"]: p for p in self.percentiles}

        def pct(label: str, key: str, default="") -> str:
            row = by_pct.get(label)
            if not row:
                return default
            v = row.get(key)
            return "" if v is None else str(round(float(v), 4))

        row = [
            self.provider, self.workload, self.concurrency, self.n,
            self.succeed, self.total, success_pct,
            round(s.get("Output token throughput (tok/s)", 0.0), 4),
            round(s.get("Request throughput (req/s)", 0.0), 4),
            round(s.get("Average time to first token (s)", 0.0), 4),
            pct("50%", "TTFT (s)"), pct("90%", "TTFT (s)"),
            pct("95%", "TTFT (s)"), pct("99%", "TTFT (s)"),
            round(s.get("Average time per output token (s)", 0.0), 4),
            pct("50%", "TPOT (s)"), pct("90%", "TPOT (s)"),
            pct("95%", "TPOT (s)"), pct("99%", "TPOT (s)"),
            round(s.get("Average inter-token latency (s)", 0.0), 4),
            pct("50%", "ITL (s)"), pct("90%", "ITL (s)"),
            pct("95%", "ITL (s)"), pct("99%", "ITL (s)"),
            round(s.get("Average latency (s)", 0.0), 4),
            pct("50%", "Latency (s)"), pct("90%", "Latency (s)"),
            pct("95%", "Latency (s)"), pct("99%", "Latency (s)"),
            round(s.get("Average input tokens per request", 0.0), 4),
            round(s.get("Average output tokens per request", 0.0), 4),
            pct("50%", "Output (tok/s)"),
        ]
        return [str(c) for c in row]


def build_evalscope_cmd(
    *,
    provider: str,
    workload: str,
    concurrency: int,
    n: int,
    chat_url: str,
    model: str,
    api_key: Optional[str],
    tokenizer_path: Path,
    outputs_dir: Path,
    sharegpt_path: Path,
    timeout_s: float,
) -> list[str]:
    spec = WORKLOADS[workload]
    cmd = [
        "evalscope", "perf",
        "--api", "openai",
        "--url", chat_url,
        "--model", model,
        "--tokenizer-path", str(tokenizer_path),
        "-n", str(n),
        "--parallel", str(concurrency),
        "--outputs-dir", str(outputs_dir),
        "--no-timestamp",
        "--stream",
        "--temperature", "0.0",
        "--connect-timeout", "60",
        "--read-timeout", str(int(timeout_s)),
        "--total-timeout", str(int(timeout_s) + 60),
    ]

    if api_key:
        cmd += ["--api-key", api_key]

    if spec["dataset"] == "random":
        in_toks = spec["in"]
        out_toks = spec["out"]
        cmd += [
            "--dataset", "random",
            "--min-prompt-length", str(in_toks),
            "--max-prompt-length", str(in_toks),
            "--max-tokens", str(out_toks),
            "--min-tokens", str(out_toks),
        ]
    elif spec["dataset"] == "custom":
        cmd += [
            "--dataset", "custom",
            "--dataset-path", str(sharegpt_path),
            "--max-tokens", str(spec["out"]),
        ]
    else:
        raise SystemExit(f"unknown dataset for workload {workload}")

    # ignore_eos only works against our vLLM server; Fireworks/Together reject it.
    if provider == "self" and spec["dataset"] == "random":
        cmd += ["--extra-args", json.dumps({"ignore_eos": True})]

    return cmd


def parse_cell_outputs(
    outputs_dir: Path, model: str, concurrency: int, n: int
) -> tuple[dict, list[dict], int, int]:
    """Read benchmark_summary.json + benchmark_percentile.json from evalscope's
    nested output dir. evalscope writes to `<outputs-dir>/<model>/parallel_<c>_number_<n>/`,
    but safe-names the model (`/` -> `_`). We search for the first matching subdir.
    """
    # evalscope's "model" subdir uses the --model value, possibly with /'s. Find it.
    root = outputs_dir
    if not root.exists():
        return {}, [], 0, 0
    target_leaf = f"parallel_{concurrency}_number_{n}"
    match = None
    for p in root.rglob(target_leaf):
        if p.is_dir():
            match = p
            break
    if match is None:
        return {}, [], 0, 0

    summary_path = match / "benchmark_summary.json"
    pct_path = match / "benchmark_percentile.json"
    summary: dict = {}
    percentiles: list[dict] = []
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = {}
    if pct_path.exists():
        try:
            percentiles = json.loads(pct_path.read_text())
        except json.JSONDecodeError:
            percentiles = []

    total = int(summary.get("Total requests", 0))
    succeed = int(summary.get("Succeed requests", 0))
    return summary, percentiles, succeed, total


def run_cell(
    *,
    provider: str,
    workload: str,
    concurrency: int,
    n: int,
    chat_url: str,
    model: str,
    api_key: Optional[str],
    tokenizer_path: Path,
    outputs_root: Path,
    sharegpt_path: Path,
    timeout_s: float,
    log_file: Path,
) -> CellResult:
    cell_dir = outputs_root / provider / workload / f"c{concurrency}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    # evalscope refuses to overwrite a dir that already has a matching run.
    for sub in cell_dir.iterdir():
        if sub.is_dir():
            shutil.rmtree(sub, ignore_errors=True)

    cmd = build_evalscope_cmd(
        provider=provider,
        workload=workload,
        concurrency=concurrency,
        n=n,
        chat_url=chat_url,
        model=model,
        api_key=api_key,
        tokenizer_path=tokenizer_path,
        outputs_dir=cell_dir,
        sharegpt_path=sharegpt_path,
        timeout_s=timeout_s,
    )

    t0 = time.monotonic()
    # Don't leak the API key in the printed command.
    safe_cmd = [("***" if (api_key and a == api_key) else a) for a in cmd]
    print(f"  $ {' '.join(safe_cmd)}", flush=True)

    with log_file.open("ab") as lf:
        lf.write(f"\n\n===== {provider}/{workload}/c={concurrency} n={n} =====\n".encode())
        lf.flush()
        try:
            proc = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=timeout_s + 300,
                check=False,
            )
        except subprocess.TimeoutExpired:
            lf.write(f"!! evalscope subprocess timed out after {timeout_s + 300}s\n".encode())
            proc = None

    wall = time.monotonic() - t0
    print(f"    wall={wall:.1f}s rc={proc.returncode if proc else 'TIMEOUT'}", flush=True)

    summary, percentiles, succeed, total = parse_cell_outputs(cell_dir, model, concurrency, n)
    if total == 0:
        total = n  # we asked for n; 0 parsed means evalscope didn't even start a request
    return CellResult(provider, workload, concurrency, n, succeed, total, summary, percentiles)


def resolve_chat_url(provider: str, self_urls: list[str]) -> str:
    if provider == "self":
        if not self_urls:
            raise SystemExit("--self-urls is required when 'self' is in --providers")
        # evalscope takes one URL. If the user passed multiple, use the first — the
        # nginx LB on 8080 should already be round-robining to the backends.
        return self_urls[0].rstrip("/") + "/v1/chat/completions"
    return PROVIDER_DEFAULTS[provider]["chat_url"]


def resolve_api_key(provider: str, cli_key: Optional[str]) -> Optional[str]:
    if provider == "self":
        return None
    if cli_key:
        return cli_key
    env_key = PROVIDER_DEFAULTS[provider]["env_key"]
    return os.environ.get(env_key) if env_key else None


def maybe_write_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)


def append_row(path: Path, row: list[str]) -> None:
    with path.open("a", newline="") as f:
        csv.writer(f).writerow(row)


def load_completed(path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not path.exists():
        return done
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add((row["provider"], row["workload"], int(row["concurrency"])))
            except (KeyError, ValueError):
                continue
    return done


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Three-provider concurrency sweep (evalscope-driven)")
    p.add_argument("--providers", nargs="+",
                   choices=["self", "fireworks", "together"],
                   default=["self", "fireworks", "together"])
    p.add_argument("--workloads", nargs="+",
                   choices=list(WORKLOADS.keys()),
                   default=list(WORKLOADS.keys()))
    p.add_argument("--concurrency", nargs="+", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--self-urls", type=str, default="",
                   help="Comma-separated base URLs for self-hosted servers "
                        "(e.g. http://MI300X_HOST:8080). Only the first is used; the "
                        "server-side nginx LB does the round-robining.")
    p.add_argument("--self-model", type=str, default="gptoss",
                   help="--served-model-name used when launching vLLM (default: gptoss)")
    p.add_argument("--fireworks-key", default=None)
    p.add_argument("--together-key", default=None)
    p.add_argument("--fireworks-model", default=None,
                   help="Override the Fireworks model string. Use this to target a "
                        "Dedicated deployment (e.g. "
                        "accounts/<acct>/deployments/<id>) instead of the shared-tier "
                        "default accounts/fireworks/models/gpt-oss-120b.")
    p.add_argument("--together-model", default=None,
                   help="Override the Together model string (same pattern as "
                        "--fireworks-model for Together dedicated endpoints).")
    p.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER,
                   help="Local path to the GPT-OSS-120B HF tokenizer (needed for "
                        "--dataset random length accuracy).")
    p.add_argument("--n-mode", choices=["conc", "fixed"], default="conc",
                   help="'conc' sets n=concurrency (matches the old sweep); "
                        "'fixed' uses --n-fixed for every cell.")
    p.add_argument("--n-fixed", type=int, default=64)
    p.add_argument("--sharegpt-path", type=Path, default=DEFAULT_SHAREGPT)
    p.add_argument("--timeout", type=float, default=900,
                   help="Per-cell evalscope timeout in seconds.")
    p.add_argument("--out", type=Path, required=True,
                   help="CSV path; rows are appended as each cell finishes.")
    p.add_argument("--outputs-root", type=Path, default=None,
                   help="Directory for per-cell evalscope outputs. Defaults to "
                        "<out_parent>/<out_stem>_evalscope/.")
    p.add_argument("--skip-completed", action="store_true",
                   help="Skip (provider,workload,concurrency) cells already present in --out.")
    p.add_argument("--cooldown", type=float, default=5.0,
                   help="Seconds to sleep between cells.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.tokenizer_path.exists():
        raise SystemExit(f"tokenizer path not found: {args.tokenizer_path}. "
                         f"Run setup/install_05_gptoss_models.sh first.")
    if shutil.which("evalscope") is None:
        raise SystemExit("evalscope not on PATH. pip install 'evalscope[perf]'")

    self_urls = [u.strip() for u in args.self_urls.split(",") if u.strip()]
    maybe_write_header(args.out)
    already = load_completed(args.out) if args.skip_completed else set()

    outputs_root = args.outputs_root
    if outputs_root is None:
        outputs_root = args.out.parent / f"{args.out.stem}_evalscope"
    outputs_root.mkdir(parents=True, exist_ok=True)
    log_file = outputs_root / "sweep.log"

    cells: list[tuple[str, str, int]] = [
        (p, w, c)
        for p in args.providers
        for w in args.workloads
        for c in args.concurrency
    ]

    for provider, workload, conc in cells:
        if (provider, workload, conc) in already:
            print(f"[skip-completed] {provider}/{workload}/c={conc} already in CSV")
            continue

        chat_url = resolve_chat_url(provider, self_urls)
        if provider == "self":
            model = args.self_model
        elif provider == "fireworks" and args.fireworks_model:
            model = args.fireworks_model
        elif provider == "together" and args.together_model:
            model = args.together_model
        else:
            model = PROVIDER_DEFAULTS[provider]["model"]
        api_key = resolve_api_key(
            provider,
            args.fireworks_key if provider == "fireworks"
            else args.together_key if provider == "together" else None,
        )
        if provider != "self" and not api_key:
            print(f"[skip] {provider}: no API key (env {PROVIDER_DEFAULTS[provider]['env_key']})")
            continue

        n = conc if args.n_mode == "conc" else args.n_fixed
        spec = WORKLOADS[workload]
        print(f"\n=== {provider:<9} / {workload:<8} / c={conc:<4} n={n} "
              f"(in={spec['in']} out={spec['out']}) ===", flush=True)

        try:
            result = run_cell(
                provider=provider,
                workload=workload,
                concurrency=conc,
                n=n,
                chat_url=chat_url,
                model=model,
                api_key=api_key,
                tokenizer_path=args.tokenizer_path,
                outputs_root=outputs_root,
                sharegpt_path=args.sharegpt_path,
                timeout_s=args.timeout,
                log_file=log_file,
            )
        except KeyboardInterrupt:
            print("interrupted; partial row not written", file=sys.stderr)
            sys.exit(130)

        row = result.to_csv_row()
        append_row(args.out, row)
        print(f"    wrote: succeed={result.succeed}/{result.total} "
              f"out_tok/s={row[7]} req/s={row[8]} ttft_p50={row[10]} e2e_p50={row[25]}",
              flush=True)

        if args.cooldown > 0 and (provider, workload, conc) != cells[-1]:
            time.sleep(args.cooldown)

    print(f"\nAll cells done. CSV: {args.out}")


if __name__ == "__main__":
    main()

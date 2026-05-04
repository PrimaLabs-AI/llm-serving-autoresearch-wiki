#!/usr/bin/env python3
"""Aggregate per-config CSVs into one ranked summary.

Reads results/gptoss20b_config_search_<ts>/per_config/<CID>.csv, pulls the
spec acceptance metrics out of metrics/<CID>.txt where present, and writes:

  - all.csv     : per_config CSVs concatenated with a config_id column
  - summary.md  : per-workload ranked tables + an overall pick

Usage:
  aggregate.py --results-dir results/gptoss20b_config_search_<ts>
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

WORKLOADS = ("decode", "prefill", "sharegpt")

# Prometheus metric names vLLM exposes for spec decoding (vLLM 0.18+).
SPEC_METRICS = {
    "accept_rate":         re.compile(r"^vllm:spec_decode_draft_acceptance_rate\s+([0-9.eE+-]+)", re.M),
    "accept_pos0":         re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos\{[^}]*position="0"[^}]*\}\s+([0-9.eE+-]+)', re.M),
    "accept_pos1":         re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos\{[^}]*position="1"[^}]*\}\s+([0-9.eE+-]+)', re.M),
    "accept_pos2":         re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos\{[^}]*position="2"[^}]*\}\s+([0-9.eE+-]+)', re.M),
    "mean_accept_length":  re.compile(r"^vllm:spec_decode_mean_acceptance_length\s+([0-9.eE+-]+)", re.M),
}


def parse_metrics(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    out: dict[str, str] = {}
    for k, rx in SPEC_METRICS.items():
        m = rx.search(text)
        if m:
            out[k] = m.group(1)
    return out


def load_per_config_csvs(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    pc_dir = results_dir / "per_config"
    for csv_path in sorted(pc_dir.glob("*.csv")):
        cid = csv_path.stem
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                row["config_id"] = cid
                rows.append(row)
    return rows


def write_all_csv(rows: list[dict], out: Path) -> None:
    if not rows:
        out.write_text("")
        return
    fields = ["config_id"] + [k for k in rows[0].keys() if k != "config_id"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt_num(s: str | None, fmt: str = "{:.0f}") -> str:
    if not s:
        return "—"
    try:
        return fmt.format(float(s))
    except (TypeError, ValueError):
        return s or "—"


def write_summary(rows: list[dict], metrics: dict[str, dict[str, str]], out: Path) -> None:
    lines: list[str] = []
    lines.append("# GPT-OSS-20B config-search results\n")
    lines.append("Per-replica throughput on 1× H100. Multiply by ~4 for a 4-GPU projection.\n")

    for w in WORKLOADS:
        wrows = [r for r in rows if r.get("workload") == w]
        # Sort descending by output_tok_s (string → float; missing → 0).
        def _tps(r: dict) -> float:
            try:
                return float(r.get("output_tok_s") or 0)
            except ValueError:
                return 0.0
        wrows.sort(key=_tps, reverse=True)

        lines.append(f"## {w}\n")
        lines.append("| rank | config | conc | output_tok_s | ttft_p50 | tpot_p50 | e2e_p99 |")
        lines.append("|-----:|--------|-----:|-------------:|---------:|---------:|--------:|")
        for i, r in enumerate(wrows, 1):
            lines.append("| {rank} | `{cid}` | {c} | {tps} | {ttft} | {tpot} | {e2e} |".format(
                rank=i,
                cid=r.get("config_id", "?"),
                c=r.get("concurrency", "?"),
                tps=fmt_num(r.get("output_tok_s")),
                ttft=fmt_num(r.get("ttft_p50"), "{:.0f}"),
                tpot=fmt_num(r.get("tpot_p50"), "{:.1f}"),
                e2e=fmt_num(r.get("e2e_p99"), "{:.0f}"),
            ))
        lines.append("")

    # Spec-decode acceptance summary across configs that ran with spec.
    spec_rows = [(cid, m) for cid, m in metrics.items() if m]
    if spec_rows:
        lines.append("## Spec-decode acceptance (from /metrics)\n")
        lines.append("| config | accept_rate | mean_accept_length | pos0 | pos1 | pos2 |")
        lines.append("|--------|------------:|-------------------:|-----:|-----:|-----:|")
        for cid, m in sorted(spec_rows):
            lines.append("| `{cid}` | {ar} | {mal} | {p0} | {p1} | {p2} |".format(
                cid=cid,
                ar=fmt_num(m.get("accept_rate"), "{:.3f}"),
                mal=fmt_num(m.get("mean_accept_length"), "{:.2f}"),
                p0=fmt_num(m.get("accept_pos0"), "{:.3f}"),
                p1=fmt_num(m.get("accept_pos1"), "{:.3f}"),
                p2=fmt_num(m.get("accept_pos2"), "{:.3f}"),
            ))
        lines.append("")

    # Overall pick: config with the best worst-case ratio vs `BASE`.
    base = {(r["workload"]): _safe_float(r.get("output_tok_s")) for r in rows if r.get("config_id") == "BASE"}
    if all(base.get(w) for w in WORKLOADS):
        scores: list[tuple[str, float, dict[str, float]]] = []
        cids = sorted({r["config_id"] for r in rows if r.get("config_id") != "BASE"})
        for cid in cids:
            ratios: dict[str, float] = {}
            for w in WORKLOADS:
                tps = next((_safe_float(r.get("output_tok_s"))
                            for r in rows if r["config_id"] == cid and r.get("workload") == w), 0.0)
                ratios[w] = tps / base[w] if base[w] else 0.0
            worst = min(ratios.values())
            scores.append((cid, worst, ratios))
        scores.sort(key=lambda x: x[1], reverse=True)

        lines.append("## Overall pick (sorted by worst-workload ratio vs `BASE`)\n")
        lines.append("| rank | config | worst-vs-BASE | decode | prefill | sharegpt |")
        lines.append("|-----:|--------|--------------:|-------:|--------:|---------:|")
        for i, (cid, worst, ratios) in enumerate(scores, 1):
            lines.append(f"| {i} | `{cid}` | {worst:.2f}× | "
                         f"{ratios['decode']:.2f}× | {ratios['prefill']:.2f}× | {ratios['sharegpt']:.2f}× |")
        lines.append("")
        if scores and scores[0][1] >= 1.0:
            lines.append(f"**Winner:** `{scores[0][0]}` — the only config that doesn't lose on any workload.\n")
        else:
            lines.append("**No config beats BASE on every workload.** Best worst-case "
                         f"is `{scores[0][0]}` at {scores[0][1]:.2f}×.\n")

    out.write_text("\n".join(lines))


def _safe_float(s: str | None) -> float:
    try:
        return float(s) if s is not None else 0.0
    except ValueError:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()

    rd = args.results_dir
    rows = load_per_config_csvs(rd)
    metrics = {p.stem: parse_metrics(p) for p in (rd / "metrics").glob("*.txt")}

    write_all_csv(rows, rd / "all.csv")
    write_summary(rows, metrics, rd / "summary.md")
    print(f"wrote {rd/'all.csv'} ({len(rows)} rows)")
    print(f"wrote {rd/'summary.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Regenerate wiki/concepts/engine-hardware-compatibility.md from
engine pages' `supported_hardware:` frontmatter and the set of hardware
pages under wiki/hardware/.

Output is committed; the LINT operation calls this script and refuses
to commit if the file would change.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

FRONTMATTER_DELIM = "---"


def parse_frontmatter(text: str) -> dict | None:
    if not text.startswith(FRONTMATTER_DELIM + "\n"):
        return None
    rest = text[len(FRONTMATTER_DELIM) + 1:]
    end = rest.find("\n" + FRONTMATTER_DELIM + "\n")
    if end == -1:
        return None
    out = {}
    for line in rest[:end].splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def parse_list(s: str) -> list[str]:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        return [x.strip() for x in s[1:-1].split(",") if x.strip()]
    return [x.strip() for x in s.split(",") if x.strip()]


def collect_engines(root: Path) -> dict[str, list[str]]:
    out = {}
    for md in sorted((root / "wiki" / "engines").glob("*.md")):
        fm = parse_frontmatter(md.read_text())
        if not fm or fm.get("type") != "engine":
            continue
        title = fm.get("title", md.stem).strip().strip('"')
        supported = parse_list(fm.get("supported_hardware", "[]"))
        out[title] = supported
    return out


def collect_hardware(root: Path) -> list[str]:
    # Order by `display_order:` frontmatter (defaulting to 999 → alphabetical
    # tiebreak by stem). mtime-based ordering would reset on `git clone`.
    rows = []
    for md in (root / "wiki" / "hardware").glob("*.md"):
        fm = parse_frontmatter(md.read_text())
        if not fm or fm.get("type") != "hardware":
            continue
        try:
            order = int(fm.get("display_order", "999"))
        except ValueError:
            order = 999
        rows.append((order, md.stem))
    rows.sort()
    return [stem for _, stem in rows]


def render_table(engines: dict[str, list[str]], hardware: list[str]) -> str:
    head = "| Engine | " + " | ".join(hardware) + " |"
    sep = "|---|" + "|".join("---" for _ in hardware) + "|"
    rows = [head, sep]
    for engine in sorted(engines):
        cells = []
        for hw in hardware:
            cells.append("✓" if hw in engines[engine] else "✗")
        rows.append(f"| {engine} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


HEADER = """---
title: "Engine × Hardware Compatibility"
type: concept
tags: [generated, compatibility]
created: 2026-05-01
updated: 2026-05-01
---

> **Generated** by `scripts/regenerate-compat-table.py` from `wiki/engines/*.md`
> `supported_hardware:` frontmatter. **Do not edit by hand** — edits will be
> overwritten on the next `LINT`.

The orchestration loop's scheduler reads each engine's `supported_hardware`
field directly. This page is a human-readable readout of that data.

"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    args = p.parse_args()
    root = Path(args.root)

    engines = collect_engines(root)
    hardware = collect_hardware(root)
    if not engines or not hardware:
        print("no engines or hardware pages found", file=sys.stderr)
        sys.exit(2)

    out_dir = root / "wiki" / "concepts"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "engine-hardware-compatibility.md").write_text(HEADER + render_table(engines, hardware))

    print(f"wrote {out_dir / 'engine-hardware-compatibility.md'}")


if __name__ == "__main__":
    main()

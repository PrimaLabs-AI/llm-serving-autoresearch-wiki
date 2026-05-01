#!/usr/bin/env python3
"""Add new required frontmatter fields to existing wiki pages.

Idempotent: pages that already carry the new fields are left unchanged.

Usage:
  python3 scripts/migrate-frontmatter.py [--root <repo-root>] [--dry-run]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

FRONTMATTER_DELIM = "---"


def split_frontmatter(text: str) -> tuple[list[str] | None, str]:
    """Return (frontmatter-lines, body). frontmatter-lines is None if absent."""
    if not text.startswith(FRONTMATTER_DELIM + "\n"):
        return None, text
    rest = text[len(FRONTMATTER_DELIM) + 1:]
    end = rest.find("\n" + FRONTMATTER_DELIM + "\n")
    if end == -1:
        return None, text
    fm = rest[:end].splitlines()
    body = rest[end + len(FRONTMATTER_DELIM) + 2:]
    return fm, body


def serialize(fm: list[str], body: str) -> str:
    return FRONTMATTER_DELIM + "\n" + "\n".join(fm) + "\n" + FRONTMATTER_DELIM + "\n" + body


def has_field(fm: list[str], key: str) -> bool:
    prefix = f"{key}:"
    return any(line.strip().startswith(prefix) for line in fm)


def add_field(fm: list[str], key: str, value: str) -> list[str]:
    if has_field(fm, key):
        return fm
    return fm + [f"{key}: {value}"]


def migrate_hypothesis(fm: list[str]) -> list[str]:
    return add_field(fm, "hardware", "any")


def migrate_experiment(fm: list[str]) -> list[str]:
    fm = add_field(fm, "hardware", "tpu-v6e")
    fm = add_field(fm, "host", "legacy-tpu")
    return fm


def migrate_model(fm: list[str]) -> list[str]:
    return add_field(fm, "target_hardware", "[tpu-v6e]")


MIGRATIONS = {
    "hypotheses": migrate_hypothesis,
    "experiments": migrate_experiment,
    "models": migrate_model,
}


def migrate_file(path: Path, migrate, dry_run: bool) -> bool:
    text = path.read_text()
    fm, body = split_frontmatter(text)
    if fm is None:
        return False
    new_fm = migrate(fm)
    if new_fm == fm:
        return False
    if not dry_run:
        path.write_text(serialize(new_fm, body))
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = Path(args.root) / "wiki"
    if not root.exists():
        print(f"no wiki dir at {root}", file=sys.stderr)
        sys.exit(2)

    changed = 0
    for subdir, fn in MIGRATIONS.items():
        target = root / subdir
        if not target.exists():
            continue
        for md in sorted(target.rglob("*.md")):
            if migrate_file(md, fn, args.dry_run):
                changed += 1
                print(f"migrated: {md.relative_to(args.root)}")

    print(f"{changed} files {'would be ' if args.dry_run else ''}migrated")


if __name__ == "__main__":
    main()

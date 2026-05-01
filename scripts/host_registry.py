#!/usr/bin/env python3
"""Host registry helper for the autoresearch loop.

Reads:
  - .hosts.toml         user-edited list of provisioned hosts
  - .host-state.toml    driver-written setup/dispatch state

Writes:
  - .host-state.toml    via `state` and `state-error` subcommands

All TOML I/O is centralized here. Bash callers only see plain text
(one host name per line, or "none", or a single value).
"""
from __future__ import annotations
import argparse
import datetime as dt
import sys
import tomllib
from pathlib import Path

HOSTS_FILE = ".hosts.toml"
STATE_FILE = ".host-state.toml"


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text())


def load_hosts(root: Path) -> dict:
    return _read(root / HOSTS_FILE).get("hosts", {})


def load_state(root: Path) -> dict:
    return _read(root / STATE_FILE).get("hosts", {})


def write_state(root: Path, state: dict) -> None:
    """TOML write — minimal, no external dep."""
    lines = []
    for name in sorted(state):
        lines.append(f"[hosts.{name}]")
        for k, v in state[name].items():
            if isinstance(v, str):
                escaped = v.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{k} = "{escaped}"')
            else:
                lines.append(f"{k} = {v}")
        lines.append("")
    (root / STATE_FILE).write_text("\n".join(lines))


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def cmd_list(args, root: Path) -> int:
    hosts = load_hosts(root)
    state = load_state(root)
    if args.summary:
        for name in sorted(hosts):
            h = hosts[name]
            s = state.get(name, {}).get("setup_state", "pending")
            print(f"{name}\t{h.get('vendor', '?')}\t{h.get('hardware', '?')}\t{s}")
    else:
        for name in sorted(hosts):
            print(name)
    return 0


def synthetic(host: dict, key: str) -> str | None:
    if key == "ssh_target":
        return f"{host['user']}@{host['ip']}"
    return None


def cmd_get(args, root: Path) -> int:
    hosts = load_hosts(root)
    if args.host not in hosts:
        print(f"unknown host: {args.host}", file=sys.stderr)
        return 2
    h = hosts[args.host]
    syn = synthetic(h, args.field)
    if syn is not None:
        print(syn)
        return 0
    if args.field not in h:
        print(f"unknown field: {args.field}", file=sys.stderr)
        return 2
    print(h[args.field])
    return 0


def matches_hypothesis_hardware(host: dict, hyp_hw: str) -> bool:
    if hyp_hw == "any":
        return True
    if hyp_hw in {"nvidia", "amd"}:
        return host.get("vendor") == hyp_hw
    return host.get("hardware") == hyp_hw


def cmd_match(args, root: Path) -> int:
    hosts = load_hosts(root)
    matched = []
    for name in sorted(hosts):
        h = hosts[name]
        if args.hardware is not None:
            if matches_hypothesis_hardware(h, args.hardware):
                matched.append(name)
        elif args.vendor is not None:
            if h.get("vendor") == args.vendor:
                matched.append(name)
    for n in matched:
        print(n)
    return 0


def cmd_state(args, root: Path) -> int:
    state = load_state(root)
    if args.host not in state:
        state[args.host] = {}
    if args.set is not None:
        state[args.host]["setup_state"] = args.set
        if args.set == "running":
            state[args.host]["setup_started"] = now_iso()
        elif args.set == "ready":
            state[args.host]["setup_finished"] = now_iso()
            state[args.host].pop("last_error", None)
        else:  # failed | unreachable | pending
            state[args.host]["last_state_change"] = now_iso()
    if args.set_error is not None:
        state[args.host]["last_error"] = args.set_error
    write_state(root, state)
    return 0


def cmd_schedule(args, root: Path) -> int:
    hosts = load_hosts(root)
    state = load_state(root)
    engine_supported = set(args.engine_supported.split(","))
    excluded = set(args.exclude or [])
    hyp_hw = args.hypothesis_hardware

    for name in sorted(hosts):
        if name in excluded:
            continue
        if state.get(name, {}).get("setup_state") != "ready":
            continue
        h = hosts[name]
        if h.get("hardware") not in engine_supported:
            continue
        if not matches_hypothesis_hardware(h, hyp_hw):
            continue
        print(name)
        return 0

    print("none")
    return 0


def cmd_reachable(args, root: Path) -> int:
    """Real ssh ping. Used at runtime by run_loop.sh."""
    import subprocess
    hosts = load_hosts(root)
    if args.host not in hosts:
        return 2
    h = hosts[args.host]
    target = f"{h['user']}@{h['ip']}"
    key = Path(h["ssh_key"]).expanduser()
    res = subprocess.run(
        [
            "ssh", "-i", str(key),
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            target, "echo ok",
        ],
        capture_output=True, text=True, timeout=15,
    )
    return 0 if res.returncode == 0 and "ok" in res.stdout else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list")
    sp.add_argument("--summary", action="store_true")
    sp.set_defaults(fn=cmd_list)

    sp = sub.add_parser("get")
    sp.add_argument("host")
    sp.add_argument("field")
    sp.set_defaults(fn=cmd_get)

    sp = sub.add_parser("match")
    grp = sp.add_mutually_exclusive_group(required=True)
    grp.add_argument("--hardware")
    grp.add_argument("--vendor")
    sp.set_defaults(fn=cmd_match)

    sp = sub.add_parser("state")
    sp.add_argument("host")
    sp.add_argument("--set", choices=["pending", "running", "ready", "failed", "unreachable"])
    sp.add_argument("--set-error")
    sp.set_defaults(fn=cmd_state)

    sp = sub.add_parser("schedule")
    sp.add_argument("--hypothesis-hardware", required=True)
    sp.add_argument("--engine-supported", required=True)
    sp.add_argument("--exclude", action="append")
    sp.set_defaults(fn=cmd_schedule)

    sp = sub.add_parser("reachable")
    sp.add_argument("host")
    sp.set_defaults(fn=cmd_reachable)

    return p


def main():
    args = build_parser().parse_args()
    root = Path(args.root)
    sys.exit(args.fn(args, root) or 0)


if __name__ == "__main__":
    main()

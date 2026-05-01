#!/usr/bin/env bash
# Top-level setup dispatcher. Detects vendor and delegates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

detect_vendor() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
        echo nvidia; return
    fi
    if command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showid 2>/dev/null | grep -q GPU; then
        echo amd; return
    fi
    echo "ERROR: no NVIDIA or AMD GPU detected (no nvidia-smi or rocm-smi)" >&2
    exit 1
}

vendor="$(detect_vendor)"
echo "Detected vendor: $vendor"

case "$vendor" in
    nvidia) exec bash "$SCRIPT_DIR/scripts/setup-cuda.sh" "$@" ;;
    amd)    exec bash "$SCRIPT_DIR/scripts/setup-rocm.sh" "$@" ;;
    *)      echo "ERROR: unknown vendor: $vendor" >&2; exit 1 ;;
esac

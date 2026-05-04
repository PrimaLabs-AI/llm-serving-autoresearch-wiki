#!/bin/bash
# GPT-OSS-20B serving-config search — orchestrator.
#
# For each config in the matrix:
#   1. tear down any existing experiment container
#   2. start one vllm replica with that config's flags (no nginx LB)
#   3. wait for /health
#   4. run evalscope at the per-replica peak concurrency for each workload
#   5. snapshot /metrics (spec accept rates, KV stats)
#   6. tear down
# Then aggregate.py turns per-config CSVs into all.csv + summary.md.
#
# Usage:
#   run_matrix.sh [--vllm-version v0.18.0] [--gpu-mem 0.85] [--configs OPT,K2,...]
#
# Prerequisites (host):
#   - Docker + NVIDIA Container Toolkit
#   - 1× H100 visible to docker
#   - Models pre-downloaded under $MODELS_DIR (default: /srv/gptoss-models):
#       /srv/gptoss-models/openai/gpt-oss-20b/
#       /srv/gptoss-models/RedHatAI/gpt-oss-20b-speculator.eagle3/
#   - evalscope + tokenizer for sweep_api_providers_evalscope.py.
#     The sweep script defaults its tokenizer-path to data/models/openai/gpt-oss-120b.
#     Override with TOKENIZER_PATH env var (point at a 20B tokenizer dir).
#
# See docs/gptoss-20b-config-search-plan.md for the full plan.

set -euo pipefail

VLLM_VERSION="v0.18.0"
GPU_MEM="0.85"
CONFIGS_FILTER=""
DOCKER="${DOCKER:-docker}"   # set DOCKER="sudo docker" if your user isn't in the docker group

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vllm-version) VLLM_VERSION="$2"; shift 2 ;;
        --gpu-mem)      GPU_MEM="$2"; shift 2 ;;
        --configs)      CONFIGS_FILTER="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,30p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-/srv/gptoss-models}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODELS_DIR}/openai/gpt-oss-20b}"
SWEEP="${PROJECT_ROOT}/scripts/benchmark/sweep_api_providers_evalscope.py"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/results/gptoss20b_config_search_${TS}}"
CONTAINER="vllm-cfgsearch"
PORT=8000
URL="http://localhost:${PORT}"
READY_TIMEOUT=1800
COOLDOWN=15

EAGLE_PATH="/models/RedHatAI/gpt-oss-20b-speculator.eagle3"
SPEC_K3='{"method":"eagle3","model":"'"${EAGLE_PATH}"'","num_speculative_tokens":3}'
SPEC_K2='{"method":"eagle3","model":"'"${EAGLE_PATH}"'","num_speculative_tokens":2}'

# --- matrix --------------------------------------------------------
# Each set_flags_<ID> populates the EXTRA_FLAGS bash array. Keep entries
# in sync with docs/gptoss-20b-config-search-plan.md.

ALL_CONFIG_IDS=(BASE OPT K2 NOSPEC BLK64 BATCH8K NOFP8KV NOBF16 LEAN)

set_config_flags() {
    EXTRA_FLAGS=()
    case "$1" in
        BASE)
            ;;
        OPT)
            EXTRA_FLAGS=(
                --dtype bfloat16 --kv-cache-dtype fp8
                --block-size 128 --max-num-batched-tokens 16384
                --speculative-config "$SPEC_K3"
            ) ;;
        K2)
            EXTRA_FLAGS=(
                --dtype bfloat16 --kv-cache-dtype fp8
                --block-size 128 --max-num-batched-tokens 16384
                --speculative-config "$SPEC_K2"
            ) ;;
        NOSPEC)
            EXTRA_FLAGS=(
                --dtype bfloat16 --kv-cache-dtype fp8
                --block-size 128 --max-num-batched-tokens 16384
            ) ;;
        BLK64)
            EXTRA_FLAGS=(
                --dtype bfloat16 --kv-cache-dtype fp8
                --block-size 64 --max-num-batched-tokens 16384
                --speculative-config "$SPEC_K3"
            ) ;;
        BATCH8K)
            EXTRA_FLAGS=(
                --dtype bfloat16 --kv-cache-dtype fp8
                --block-size 128 --max-num-batched-tokens 8192
                --speculative-config "$SPEC_K3"
            ) ;;
        NOFP8KV)
            EXTRA_FLAGS=(
                --dtype bfloat16
                --block-size 128 --max-num-batched-tokens 16384
                --speculative-config "$SPEC_K3"
            ) ;;
        NOBF16)
            EXTRA_FLAGS=(
                --kv-cache-dtype fp8
                --block-size 128 --max-num-batched-tokens 16384
                --speculative-config "$SPEC_K3"
            ) ;;
        LEAN)
            EXTRA_FLAGS=(
                --dtype bfloat16 --kv-cache-dtype fp8
                --block-size 64 --max-num-batched-tokens 8192
                --speculative-config "$SPEC_K2"
            ) ;;
        *)
            echo "FATAL: unknown config id: $1" >&2; exit 2 ;;
    esac
}

# Per-workload concurrency (per-replica equivalent of the 4-GPU peaks).
declare -A WORKLOAD_CONC=(
    [decode]=128
    [prefill]=512
    [sharegpt]=256
)

# --- preflight -----------------------------------------------------

if [[ ! -f "$SWEEP" ]]; then
    echo "FATAL: sweep script not found at $SWEEP" >&2; exit 1
fi
if [[ ! -d "$TOKENIZER_PATH" ]]; then
    echo "FATAL: tokenizer dir not found at $TOKENIZER_PATH" >&2
    echo "Set TOKENIZER_PATH env var to a 20B tokenizer dir, or download one with:" >&2
    echo "  hf download openai/gpt-oss-20b --local-dir <path> \\" >&2
    echo "      --include 'config.json' 'tokenizer*' '*.jinja' 'special_tokens_map.json'" >&2
    exit 1
fi
if [[ ! -d "${MODELS_DIR}/openai/gpt-oss-20b" ]]; then
    echo "FATAL: weights not found at ${MODELS_DIR}/openai/gpt-oss-20b" >&2
    echo "Download with: hf download openai/gpt-oss-20b --local-dir ${MODELS_DIR}/openai/gpt-oss-20b" >&2
    exit 1
fi
if [[ ! -d "${MODELS_DIR}/RedHatAI/gpt-oss-20b-speculator.eagle3" ]]; then
    echo "WARN: Eagle3 draft not found at ${MODELS_DIR}/RedHatAI/gpt-oss-20b-speculator.eagle3" >&2
    echo "      Spec configs (OPT/K2/BLK64/BATCH8K/NOFP8KV/NOBF16/LEAN) will fail at boot." >&2
fi

mkdir -p "${OUT_DIR}"/{per_config,boot_logs,metrics,evalscope}
echo "[run_matrix] OUT_DIR=${OUT_DIR}"
echo "[run_matrix] vLLM=${VLLM_VERSION} GPU_MEM=${GPU_MEM}"

# --- helpers -------------------------------------------------------

teardown() {
    # rm -f is idempotent: stops if running, removes if exited, no-ops if absent.
    # Avoids the docker-run-with-rm async-cleanup race we hit between back-to-back
    # invocations of this script in canary_v019.sh.
    $DOCKER rm -f "$CONTAINER" >/dev/null 2>&1 || true
    # Drain any background "docker logs -f" subprocess from a prior launch in
    # this same shell.
    wait 2>/dev/null || true
}

launch() {
    local cid="$1" log="${OUT_DIR}/boot_logs/${cid}.log"
    teardown
    set_config_flags "$cid"
    echo "[run_matrix] launching ${cid}: vllm/vllm-openai:${VLLM_VERSION} ${EXTRA_FLAGS[*]:-}"
    $DOCKER run -d --name "$CONTAINER" \
        --gpus all --ipc=host --shm-size 16g \
        -p "${PORT}:${PORT}" \
        -v "${MODELS_DIR}:/models:ro" \
        --entrypoint python3 \
        "vllm/vllm-openai:${VLLM_VERSION}" \
        -m vllm.entrypoints.openai.api_server \
        --model /models/openai/gpt-oss-20b \
        --served-model-name gptoss \
        --host 0.0.0.0 --port "$PORT" \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        --gpu-memory-utilization "$GPU_MEM" \
        "${EXTRA_FLAGS[@]}" \
        >"${log}.cid" 2>&1 || { echo "FATAL: docker run failed; see ${log}"; cat "${log}.cid"; return 1; }

    # Stream logs to the file in the background so we can tail on failure.
    $DOCKER logs -f "$CONTAINER" >"$log" 2>&1 &
    local logger_pid=$!

    local deadline=$(( SECONDS + READY_TIMEOUT ))
    while (( SECONDS < deadline )); do
        if ! $DOCKER ps --filter "name=^${CONTAINER}$" --format '{{.Names}}' | grep -q .; then
            echo "[run_matrix] FATAL: container ${CONTAINER} died during startup; tail of log:" >&2
            tail -n 80 "$log" >&2
            kill "$logger_pid" 2>/dev/null || true
            return 1
        fi
        if curl -fsS "${URL}/health" >/dev/null 2>&1; then
            echo "[run_matrix] ${cid} ready after $((SECONDS))s of this attempt"
            kill "$logger_pid" 2>/dev/null || true
            return 0
        fi
        sleep 5
    done
    echo "[run_matrix] FATAL: ${cid} not ready after ${READY_TIMEOUT}s" >&2
    tail -n 80 "$log" >&2
    kill "$logger_pid" 2>/dev/null || true
    return 1
}

run_workload() {
    local cid="$1" workload="$2" conc="$3"
    local csv="${OUT_DIR}/per_config/${cid}.csv"
    local outputs="${OUT_DIR}/evalscope/${cid}/${workload}_c${conc}"
    mkdir -p "$(dirname "$outputs")"
    echo "[run_matrix] ${cid}/${workload} c=${conc}"
    python3 "$SWEEP" \
        --self-urls "$URL" \
        --providers self \
        --self-model gptoss \
        --workloads "$workload" \
        --concurrency "$conc" \
        --n-mode conc \
        --tokenizer-path "$TOKENIZER_PATH" \
        --cooldown "$COOLDOWN" \
        --outputs-root "${OUT_DIR}/evalscope/${cid}" \
        --out "$csv"
}

snapshot_metrics() {
    local cid="$1"
    curl -sS "${URL}/metrics" >"${OUT_DIR}/metrics/${cid}.txt" 2>/dev/null || true
}

# --- main loop -----------------------------------------------------

# Optional filter (--configs ID1,ID2,...). Otherwise run all.
SELECTED=()
if [[ -n "$CONFIGS_FILTER" ]]; then
    IFS=',' read -ra SELECTED <<<"$CONFIGS_FILTER"
else
    SELECTED=("${ALL_CONFIG_IDS[@]}")
fi

trap 'echo "[run_matrix] interrupted; tearing down"; teardown; exit 130' SIGINT SIGTERM

OVERALL_T0=$SECONDS
FAILED=()

for cid in "${SELECTED[@]}"; do
    echo
    echo "============================================================"
    echo "[run_matrix] config ${cid} (elapsed: $(( (SECONDS - OVERALL_T0) / 60 )) min)"
    echo "============================================================"
    if ! launch "$cid"; then
        FAILED+=("${cid}:launch")
        continue
    fi
    for workload in decode prefill sharegpt; do
        if ! run_workload "$cid" "$workload" "${WORKLOAD_CONC[$workload]}"; then
            FAILED+=("${cid}:${workload}")
        fi
    done
    snapshot_metrics "$cid"
    teardown
done

echo
echo "[run_matrix] all configs done in $(( (SECONDS - OVERALL_T0) / 60 )) min"
if (( ${#FAILED[@]} )); then
    echo "[run_matrix] FAILURES: ${FAILED[*]}" >&2
fi

echo "[run_matrix] aggregating to ${OUT_DIR}/summary.md"
python3 "$(dirname "$0")/aggregate.py" --results-dir "${OUT_DIR}" || {
    echo "[run_matrix] aggregate.py failed; per-config CSVs are in ${OUT_DIR}/per_config/" >&2
}

echo "[run_matrix] done. Results: ${OUT_DIR}"

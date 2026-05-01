#!/usr/bin/env bash
# Automated evaluation pipeline for LLM serving engines.
#
# End-to-end: start engine → sweep workloads → sweep concurrency levels →
# collect metrics → write results → stop engine → repeat for next engine.
#
# Usage:
#   ./run_eval.sh                                          # full sweep, all engines
#   ./run_eval.sh --engines vllm,sglang                    # specific engines
#   ./run_eval.sh --model meta-llama/Meta-Llama-3-8B-Instruct
#   ./run_eval.sh --workloads multi-turn-agentic,chain-of-thought  # specific workloads
#   ./run_eval.sh --concurrency 16,32,64,128               # specific concurrency levels
#   ./run_eval.sh --setup                                  # just bootstrap, don't run
#   ./run_eval.sh --dry-run                                # print what would run
#
# On a fresh GPU instance:
#   git clone https://github.com/PrimaLabs-AI/llm-serving-autoresearch-wiki
#   cd llm-serving-autoresearch-wiki
#   ./run_eval.sh --setup && ./run_eval.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"
RESULTS_DIR="${REPO_DIR}/raw/benchmarks"
DATE=$(date +%Y-%m-%d)

# Defaults
ENGINES=("vllm" "sglang" "trt-llm")
WORKLOADS=("multi-turn-agentic" "parallel-tool-use" "long-context-rag" "chain-of-thought" "structured-output")
CONCURRENCY_LEVELS=(16 32 64 128)
MODEL=""
DRY_RUN=false
SETUP_ONLY=false
GPU_WAIT_TIMEOUT=300

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()   { echo -e "${GREEN}[EVAL]${NC} $(date +%H:%M:%S) $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $(date +%H:%M:%S) $*"; }
error() { echo -e "${RED}[ERROR]${NC} $(date +%H:%M:%S) $*" >&2; }
info()  { echo -e "${CYAN}[INFO]${NC} $(date +%H:%M:%S) $*"; }
header() { echo -e "\n${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n${BOLD}$*${NC}\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --engines <list>        Comma-separated engines (vllm,sglang,trt-llm)"
    echo "  --workloads <list>      Comma-separated workloads"
    echo "  --concurrency <list>    Comma-separated concurrency levels"
    echo "  --model <name>          Model to benchmark (required for running)"
    echo "  --setup                 Bootstrap instance only (Docker + images)"
    echo "  --dry-run               Print what would run, don't execute"
    echo "  -h, --help              Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --setup                                              # bootstrap only"
    echo "  $0 --model meta-llama/Meta-Llama-3-8B-Instruct          # full sweep"
    echo "  $0 --engines vllm --workloads multi-turn-agentic        # targeted"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --engines)
            shift; IFS=',' read -ra ENGINES <<< "$1" ;;
        --workloads)
            shift; IFS=',' read -ra WORKLOADS <<< "$1" ;;
        --concurrency)
            shift; IFS=',' read -ra CONCURRENCY_LEVELS <<< "$1" ;;
        --model)
            shift; MODEL="$1" ;;
        --setup) SETUP_ONLY=true ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

do_setup() {
    header "Bootstrapping instance"

    # Run setup.sh with Docker mode
    if [[ ! -f "${REPO_DIR}/.setup-done" ]]; then
        log "Running setup.sh --docker..."
        cd "${REPO_DIR}"
        bash setup.sh --docker
        touch "${REPO_DIR}/.setup-done"
        log "Setup complete."
    else
        log "Setup already done (.setup-done exists). Skipping."
        log "To re-run setup: rm .setup-done && ./run_eval.sh --setup"
    fi

    # Verify Docker is working
    if ! docker info &>/dev/null; then
        error "Docker is not running. Start it and retry."
        exit 1
    fi

    # Verify GPU access from Docker
    if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        error "Docker cannot access GPUs. Check NVIDIA Container Toolkit."
        exit 1
    fi

    log "GPU access from Docker: OK"
}

# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------

# Engine → docker compose service name mapping
engine_to_service() {
    case $1 in
        vllm) echo "vllm" ;;
        sglang) echo "sglang" ;;
        trt-llm) echo "trt-llm" ;;
    esac
}

# Engine → health check URL (from container's perspective, accessed from host)
engine_to_health_url() {
    case $1 in
        vllm) echo "http://localhost:8000/health" ;;
        sglang) echo "http://localhost:30000/health" ;;
        trt-llm) echo "http://localhost:8001/health" ;;
    esac
}

# Engine → port flag for benchmark harness
engine_to_url() {
    case $1 in
        vllm) echo "http://localhost:8000" ;;
        sglang) echo "http://localhost:30000" ;;
        trt-llm) echo "http://localhost:8001" ;;
    esac
}

start_engine() {
    local engine=$1
    local service
    service=$(engine_to_service "$engine")

    header "Starting ${engine}"

    # Set model in .env if specified
    if [[ -n "${MODEL}" ]]; then
        log "Model: ${MODEL}"
        export MODEL
    fi

    log "Starting ${service} container..."

    if $DRY_RUN; then
        echo "  docker compose up -d ${service}"
        return 0
    fi

    if [[ "${engine}" == "trt-llm" ]]; then
        docker compose --profile trt-llm up -d "${service}"
    else
        docker compose up -d "${service}"
    fi

    # Wait for health
    local health_url
    health_url=$(engine_to_health_url "$engine")
    log "Waiting for ${engine} to be healthy at ${health_url} (timeout: ${GPU_WAIT_TIMEOUT}s)..."

    local elapsed=0
    while [[ $elapsed -lt $GPU_WAIT_TIMEOUT ]]; do
        if curl -sf "${health_url}" &>/dev/null; then
            log "${engine} is healthy (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        printf "  waiting... %ds\r" "$elapsed"
    done

    error "${engine} did not become healthy within ${GPU_WAIT_TIMEOUT}s"
    docker compose logs --tail=50 "${service}"
    return 1
}

stop_engine() {
    local engine=$1
    local service
    service=$(engine_to_service "$engine")

    log "Stopping ${service}..."
    if ! $DRY_RUN; then
        docker compose stop "${service}" 2>/dev/null || true
        docker compose rm -f "${service}" 2>/dev/null || true
    fi
    log "${engine} stopped."
}

# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

run_benchmark() {
    local engine=$1
    local workload=$2
    local model=$3
    local base_url
    base_url=$(engine_to_url "$engine")

    local slug="${engine}-$(echo "${workload}" | tr '_' '-')"
    local output_dir="${RESULTS_DIR}/${DATE}-${slug}"

    log "Benchmark: engine=${engine} workload=${workload} model=${model}"
    log "Output: ${output_dir}"
    log "Concurrency levels: ${CONCURRENCY_LEVELS[*]}"

    if $DRY_RUN; then
        echo "  python benchmark_harness.py \\"
        echo "    --engine ${engine} \\"
        echo "    --model ${model} \\"
        echo "    --workload ${workload} \\"
        echo "    --skip-server \\"
        echo "    --output-dir ${output_dir}"
        return 0
    fi

    # Create output dir
    mkdir -p "${output_dir}"

    # Run benchmark harness (connects to already-running engine)
    cd "${REPO_DIR}"
    python benchmark_harness.py \
        --engine "${engine}" \
        --model "${model}" \
        --workload "${workload}" \
        --skip-server \
        --output-dir "${output_dir}" \
        --concurrency-levels "${CONCURRENCY_LEVELS[@]}" \
        2>&1 | tee "${output_dir}/console.log"

    local exit_code=${PIPESTATUS[0]}

    if [[ $exit_code -eq 0 ]]; then
        log "Benchmark complete: ${output_dir}"
    else
        error "Benchmark failed (exit ${exit_code}): ${output_dir}"
    fi

    return $exit_code
}

# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------

aggregate_results() {
    header "Aggregating results"

    local summary_file="${RESULTS_DIR}/${DATE}-summary.md"

    {
        echo "# Benchmark Summary — ${DATE}"
        echo ""
        echo "Model: ${MODEL:-<not set>}"
        echo "Engines: ${ENGINES[*]}"
        echo "Workloads: ${WORKLOADS[*]}"
        echo "Concurrency levels: ${CONCURRENCY_LEVELS[*]}"
        echo ""
        echo "---"
        echo ""

        for engine in "${ENGINES[@]}"; do
            for workload in "${WORKLOADS[@]}"; do
                local slug="${engine}-$(echo "${workload}" | tr '_' '-')"
                local metrics_file="${RESULTS_DIR}/${DATE}-${slug}/metrics.json"

                if [[ -f "${metrics_file}" ]]; then
                    echo "## ${engine} / ${workload}"
                    echo ""
                    echo '| Conc | Req/s | Tok/s | TTFT mean | TTFT p99 | TPOT mean |'
                    echo '|------|-------|-------|-----------|----------|-----------|'

                    # Parse metrics.json — each line is one concurrency level
                    python3 -c "
import json, sys
with open('${metrics_file}') as f:
    data = json.load(f)
for m in data:
    print(f\"| {m.get('concurrency','?')} \"
          f\"| {m.get('throughput_req_s','N/A')} \"
          f\"| {m.get('throughput_tokens_s','N/A')} \"
          f\"| {m.get('ttft_mean_ms','N/A')} \"
          f\"| {m.get('ttft_p99_ms','N/A')} \"
          f\"| {m.get('tpot_mean_ms','N/A')} |\")
" 2>/dev/null || echo "(metrics parse failed)"

                    echo ""
                    echo "Results: [\`${slug}\`](${DATE}-${slug}/)"
                    echo ""
                else
                    echo "## ${engine} / ${workload}"
                    echo ""
                    echo "*(no results)*"
                    echo ""
                fi
            done
        done
    } > "${summary_file}"

    log "Summary written to: ${summary_file}"

    # Print summary to console
    if ! $DRY_RUN; then
        header "Results summary"
        cat "${summary_file}"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    header "LLM Serving Evaluation Pipeline"
    echo "  Engines:       ${ENGINES[*]}"
    echo "  Workloads:     ${WORKLOADS[*]}"
    echo "  Concurrency:   ${CONCURRENCY_LEVELS[*]}"
    echo "  Model:         ${MODEL:-<not set — use --model>}"
    echo "  Dry run:       ${DRY_RUN}"
    echo ""

    # Phase 1: Setup
    do_setup

    if $SETUP_ONLY; then
        log "Setup-only mode. Exiting."
        exit 0
    fi

    # Validate model is set
    if [[ -z "${MODEL}" ]] && ! $DRY_RUN; then
        error "No model specified. Use --model <model-name>"
        echo ""
        echo "Example models:"
        echo "  meta-llama/Meta-Llama-3-8B-Instruct"
        echo "  meta-llama/Meta-Llama-3.1-70B-Instruct"
        echo "  Qwen/Qwen2.5-72B-Instruct"
        echo "  mistralai/Mixtral-8x7B-Instruct-v0.1"
        exit 1
    fi

    # Phase 2: Run benchmarks per engine
    local failed=0
    local succeeded=0

    for engine in "${ENGINES[@]}"; do
        header "Engine: ${engine}"

        # Start engine
        if ! start_engine "${engine}"; then
            error "Failed to start ${engine}. Skipping."
            failed=$((failed + 1))
            continue
        fi

        # Run each workload
        for workload in "${WORKLOADS[@]}"; do
            if run_benchmark "${engine}" "${workload}" "${MODEL}"; then
                succeeded=$((succeeded + 1))
            else
                failed=$((failed + 1))
                warn "Benchmark failed: ${engine}/${workload}"
            fi
        done

        # Stop engine
        stop_engine "${engine}"

        # Brief pause between engines
        if ! $DRY_RUN; then
            log "Waiting 10s before next engine..."
            sleep 10
        fi
    done

    # Phase 3: Aggregate
    aggregate_results

    # Final summary
    header "Done"
    echo "  Succeeded: ${succeeded}"
    echo "  Failed:    ${failed}"
    echo "  Results:   ${RESULTS_DIR}/${DATE}-summary.md"
    echo ""
    echo "  Next steps:"
    echo "    1. Review: cat ${RESULTS_DIR}/${DATE}-summary.md"
    echo "    2. Commit: git add raw/benchmarks/${DATE}-* && git commit -m 'benchmarks ${DATE}'"
    echo "    3. Agent:  claude   → 'Ingest benchmark results from ${RESULTS_DIR}/${DATE}-summary.md'"
    echo ""
}

main

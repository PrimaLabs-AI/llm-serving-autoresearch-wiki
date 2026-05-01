#!/usr/bin/env bash
# Bootstrap script for LLM Serving Autoresearch Wiki on a fresh GPU instance.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh                          # install everything
#   ./setup.sh --engines vllm,sglang    # install only specific engines
#   ./setup.sh --skip-claude            # skip Claude Code installation
#   ./setup.sh --docker                 # use Docker containers instead of pip install
#   ./setup.sh --skip-tensorrt          # skip TensorRT-LLM (requires NVIDIA repo)
#
# Tested on: Ubuntu 22.04 with NVIDIA A100/H100 GPUs
# Prerequisites: CUDA toolkit, NVIDIA drivers (pre-installed on most GPU clouds)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_URL="https://github.com/PrimaLabs-AI/llm-serving-autoresearch-wiki"
REPO_DIR="llm-serving-autoresearch-wiki"
PYTHON="python3"
PIP="pip3"

# Defaults
INSTALL_VLLM=true
INSTALL_SGLANG=true
INSTALL_TRTLLM=true
INSTALL_CLAUDE=true
USE_DOCKER=false
SKIP_REPO=false

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --engines <list>     Comma-separated list of engines to install (vllm,sglang,tensorrt)"
    echo "  --skip-claude        Skip Claude Code installation"
    echo "  --skip-repo          Skip repo clone (already cloned)"
    echo "  --skip-tensorrt      Skip TensorRT-LLM installation"
    echo "  --docker             Use Docker containers instead of pip install"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                # install everything"
    echo "  $0 --engines vllm,sglang          # vLLM + SGLang only"
    echo "  $0 --skip-claude --skip-tensorrt  # no Claude Code, no TensorRT-LLM"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --engines)
            shift
            IFS=',' read -ra ENGINES <<< "$1"
            INSTALL_VLLM=false
            INSTALL_SGLANG=false
            INSTALL_TRTLLM=false
            for e in "${ENGINES[@]}"; do
                case $e in
                    vllm) INSTALL_VLLM=true ;;
                    sglang) INSTALL_SGLANG=true ;;
                    tensorrt|trt) INSTALL_TRTLLM=true ;;
                    *) echo "WARNING: unknown engine '$e'" ;;
                esac
            done
            ;;
        --skip-claude) INSTALL_CLAUDE=false ;;
        --docker) USE_DOCKER=true ;;
        --skip-repo) SKIP_REPO=true ;;
        --skip-tensorrt) INSTALL_TRTLLM=false ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()   { echo -e "${GREEN}[SETUP]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
info()  { echo -e "${CYAN}[INFO]${NC} $*"; }

check_gpu() {
    if nvidia-smi &>/dev/null; then
        local gpu_count
        gpu_count=$(nvidia-smi --list-gpus | wc -l)
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        local gpu_mem
        gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
        log "GPU detected: ${gpu_name} (${gpu_mem} VRAM) x ${gpu_count}"
        return 0
    else
        error "No NVIDIA GPU detected. This setup requires a GPU instance."
        return 1
    fi
}

check_cuda() {
    if nvcc --version &>/dev/null; then
        local cuda_version
        cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
        log "CUDA version: ${cuda_version}"
        return 0
    else
        warn "nvcc not found. CUDA toolkit may not be installed."
        warn "Most GPU cloud instances have CUDA pre-installed."
        warn "Continuing anyway — engine install will fail if CUDA is truly missing."
        return 0
    fi
}

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------

install_system_deps() {
    log "Installing system dependencies..."

    sudo apt-get update -qq

    sudo apt-get install -y -qq \
        build-essential \
        git \
        curl \
        wget \
        "${PYTHON}" \
        "${PYTHON}-pip" \
        "${PYTHON}-venv" \
        nvtop \
        htop \
        tmux \
        2>/dev/null

    log "System dependencies installed."
}

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------

setup_venv() {
    log "Setting up Python virtual environment..."

    # Create venv in the repo directory
    if [[ ! -d "venv" ]]; then
        "${PYTHON}" -m venv venv
    fi

    source venv/bin/activate
    pip install --upgrade pip setuptools wheel

    log "Virtual environment ready at venv/"
}

# ---------------------------------------------------------------------------
# Docker setup
# ---------------------------------------------------------------------------

install_docker() {
    if command -v docker &>/dev/null && docker compose version &>/dev/null; then
        log "Docker + Docker Compose already installed."
        return 0
    fi

    log "Installing Docker + Docker Compose..."

    # Install Docker
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "${USER}"

    # Install Docker Compose plugin
    sudo apt-get install -y -qq docker-compose-plugin 2>/dev/null || \
        sudo mkdir -p /usr/local/lib/docker/cli-plugins && \
        sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
            -o /usr/local/lib/docker/cli-plugins/docker-compose && \
        sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

    log "Docker installed: $(docker --version)"
    log "Docker Compose installed: $(docker compose version)"

    # NVIDIA Container Toolkit (required for GPU passthrough)
    if ! dpkg -l nvidia-container-toolkit &>/dev/null; then
        log "Installing NVIDIA Container Toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update -qq
        sudo apt-get install -y -qq nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        log "NVIDIA Container Toolkit installed."
    fi
}

pull_engine_images() {
    log "Pulling engine Docker images..."

    $INSTALL_VLLM && {
        log "Pulling vLLM image..."
        docker pull vllm/vllm-openai:latest
    }

    $INSTALL_SGLANG && {
        log "Pulling SGLang image..."
        docker pull sglang/sglang:latest
    }

    $INSTALL_TRTLLM && {
        log "Pulling TensorRT-LLM image..."
        docker pull nvcr.io/nvidia/tensorrt-llm:latest
    }

    log "Building benchmark orchestrator image..."
    docker compose build benchmark

    log "All images ready."
}

# ---------------------------------------------------------------------------
# Serving engines
# ---------------------------------------------------------------------------

install_vllm() {
    log "Installing vLLM..."
    pip install vllm
    log "vLLM installed: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'version check failed')"
}

install_sglang() {
    log "Installing SGLang..."
    pip install "sglang[all]"
    log "SGLang installed: $(python -c 'import sglang; print(sglang.__version__)' 2>/dev/null || echo 'version check failed')"
}

install_trtllm() {
    log "Installing TensorRT-LLM..."
    warn "TensorRT-LLM requires specific CUDA/Python versions."
    warn "See: https://github.com/NVIDIA/TensorRT-LLM/releases"
    warn "Attempting pip install..."

    # TensorRT-LLM requires specific wheel per CUDA version
    local cuda_major
    cuda_major=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1 || echo "12")

    if [[ "${cuda_major}" == "12" ]]; then
        pip install tensorrt-llm 2>/dev/null || \
            warn "TensorRT-LLM pip install failed. Manual install may be needed:"
            warn "  pip3 install tensorrt-llm --extra-index-url https://pypi.nvidia.com"
    else
        warn "TensorRT-LLM requires CUDA 12+. Skipping."
        INSTALL_TRTLLM=false
        return
    fi

    log "TensorRT-LLM installed."
}

# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------

install_claude_code() {
    log "Installing Claude Code..."

    # Check if Node.js is available
    if ! command -v node &>/dev/null; then
        log "Installing Node.js..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y -qq nodejs
    fi

    log "Node.js version: $(node --version)"

    if command -v claude &>/dev/null; then
        log "Claude Code already installed: $(claude --version 2>/dev/null || echo 'unknown')"
    else
        npm install -g @anthropic-ai/claude-code
        log "Claude Code installed."
    fi
}

# ---------------------------------------------------------------------------
# Repo
# ---------------------------------------------------------------------------

clone_repo() {
    if [[ "${SKIP_REPO}" == true ]]; then
        log "Skipping repo clone (--skip-repo)."
        if [[ -d "${REPO_DIR}" ]]; then
            cd "${REPO_DIR}"
            log "Working in existing repo: $(pwd)"
        fi
        return
    fi

    log "Cloning repo..."
    git clone "${REPO_URL}" "${REPO_DIR}"
    cd "${REPO_DIR}"
    log "Repo cloned to: $(pwd)"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print_summary() {
    echo ""
    echo "========================================================"
    log "Setup complete!"
    echo "========================================================"
    echo ""
    echo "  Repo:       $(pwd)"
    if $USE_DOCKER; then
        echo "  Mode:       Docker"
        echo "  Engines:"
        $INSTALL_VLLM  && echo "    - vLLM:     docker compose up vllm"
        $INSTALL_SGLANG && echo "    - SGLang:   docker compose up sglang"
        $INSTALL_TRTLLM && echo "    - TRT-LLM:  docker compose --profile trt-llm up trt-llm"
    else
        echo "  Mode:       Native (pip)"
        echo "  Engines:"
        $INSTALL_VLLM  && echo "    - vLLM:     $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'installed')"
        $INSTALL_SGLANG && echo "    - SGLang:   $(python -c 'import sglang; print(sglang.__version__)' 2>/dev/null || echo 'installed')"
        $INSTALL_TRTLLM && echo "    - TRT-LLM:  installed"
    fi
    $INSTALL_CLAUDE && echo "    - Claude:   $(claude --version 2>/dev/null || echo 'installed')"
    echo ""
    if $USE_DOCKER; then
        echo "  Quick start (Docker):"
        echo "    # 1. Configure model"
        echo "    cp .env.example .env && vi .env"
        echo ""
        echo "    # 2. Start an engine"
        echo "    docker compose up -d vllm"
        echo ""
        echo "    # 3. Run benchmark"
        echo "    docker compose run --rm benchmark \\"
        echo "      --engine vllm --model \$MODEL \\"
        echo "      --workload multi-turn-agentic --skip-server \\"
        echo "      --output-dir raw/benchmarks/\$(date +%Y-%m-%d)-vllm-test"
        echo ""
        echo "    # Or: interactive shell with Claude Code"
        echo "    docker compose run --rm -it benchmark-shell"
        echo "    # Then inside: claude"
    else
        echo "  Quick start:"
        echo "    source venv/bin/activate"
        echo "    claude"
    fi
    echo ""
    echo "  Then tell the agent:"
    echo "    'Run hypothesis #1 — prefix caching in vLLM for multi-turn"
    echo "     agentic workloads with model <MODEL> on this GPU.'"
    echo ""
    echo "  Run a manual benchmark:"
    echo "    python benchmark_harness.py \\"
    echo "      --engine vllm \\"
    echo "      --model <MODEL> \\"
    echo "      --workload multi-turn-agentic \\"
    echo "      --skip-server \\"
    echo "      --output-dir raw/benchmarks/\$(date +%Y-%m-%d)-test"
    echo ""
    echo "========================================================"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    echo ""
    log "LLM Serving Autoresearch Wiki — Instance Setup"
    log "================================================"
    echo ""

    check_gpu
    check_cuda

    install_system_deps
    clone_repo

    if $USE_DOCKER; then
        # Docker mode: engines run in containers, only need Docker + Claude Code
        log "Using Docker mode — engines will run in containers."
        install_docker
        setup_venv  # lightweight venv for the benchmark harness only
        pip install httpx requests pydantic rich  # minimal client deps
        pull_engine_images
    else
        # Native mode: install engines directly via pip
        setup_venv

        if $INSTALL_VLLM; then
            install_vllm
        else
            info "Skipping vLLM (--engines filter)"
        fi

        if $INSTALL_SGLANG; then
            install_sglang
        else
            info "Skipping SGLang (--engines filter)"
        fi

        if $INSTALL_TRTLLM; then
            install_trtllm
        else
            info "Skipping TensorRT-LLM (--skip-tensorrt or --engines filter)"
        fi
    fi

    # Install Claude Code
    if $INSTALL_CLAUDE; then
        install_claude_code
    else
        info "Skipping Claude Code (--skip-claude)"
    fi

    # Save environment info
    {
        echo "# Auto-generated by setup.sh on $(date)"
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
        echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | sed 's/,//' || echo 'unknown')"
        echo "Setup date: $(date -I)"
    } > .env/.setup-info 2>/dev/null || true

    print_summary
}

main

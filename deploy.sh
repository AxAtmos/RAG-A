#!/usr/bin/env bash
# ============================================================
# RAG Enterprise - Linux One-Click Deploy Script
# ============================================================
# Usage: sudo bash deploy.sh
# Target: Ubuntu 22.04+ / CentOS 8+ / Debian 12+
# ============================================================

set -euo pipefail

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $*"; }

# ── Configuration ──
RAG_DIR="/opt/rag-enterprise"
DATA_DIR="/data"
KNOWLEDGE_DIR="/knowledge_base"
PYTHON_VERSION="3.11"
OLLAMA_MODEL="qwen3:8b"

# ── Pre-check ──
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

check_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
        log_info "GPU detected: ${GPU_INFO}"
        HAS_GPU=true
    else
        log_warn "No NVIDIA GPU detected. Will use CPU for inference (slower)."
        HAS_GPU=false
    fi
}

# ── Step 1: Install system dependencies ──
install_system_deps() {
    log_step "Installing system dependencies..."

    if command -v apt-get &>/dev/null; then
        apt-get update -qq
        apt-get install -y -qq \
            curl wget git \
            python3 python3-pip python3-venv \
            build-essential libpq-dev \
            docker.io docker-compose-plugin \
            jq
    elif command -v yum &>/dev/null; then
        yum install -y -q \
            curl wget git \
            python3 python3-pip \
            gcc postgresql-devel \
            docker docker-compose \
            jq
    elif command -v dnf &>/dev/null; then
        dnf install -y -q \
            curl wget git \
            python3 python3-pip \
            gcc postgresql-devel \
            docker docker-compose \
            jq
    else
        log_error "Unsupported package manager. Install manually: curl python3 docker docker-compose jq"
        exit 1
    fi

    # Start Docker
    systemctl enable docker
    systemctl start docker

    log_info "System dependencies installed"
}

# ── Step 2: Auto-detect and install Ollama ──
setup_ollama() {
    log_step "Setting up Ollama..."

    if command -v ollama &>/dev/null; then
        OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
        log_info "Ollama already installed: ${OLLAMA_VERSION}"
    else
        log_info "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        log_info "Ollama installed"
    fi

    # Ensure Ollama service is running
    if systemctl is-active --quiet ollama 2>/dev/null; then
        log_info "Ollama service is running"
    else
        log_info "Starting Ollama service..."
        systemctl enable ollama 2>/dev/null || true
        systemctl start ollama 2>/dev/null || ollama serve &
        sleep 3
    fi

    # Check if model exists, pull if not
    if ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL}"; then
        log_info "Model ${OLLAMA_MODEL} already available"
    else
        log_info "Pulling model ${OLLAMA_MODEL} (this may take a while)..."
        ollama pull "${OLLAMA_MODEL}"
        log_info "Model ${OLLAMA_MODEL} ready"
    fi
}

# ── Step 3: Setup project files ──
setup_project() {
    log_step "Setting up project..."

    mkdir -p "${RAG_DIR}"
    mkdir -p "${DATA_DIR}/images" "${DATA_DIR}/logs"
    mkdir -p "${KNOWLEDGE_DIR}/_processed"
    mkdir -p "${KNOWLEDGE_DIR}/公开"
    mkdir -p "${KNOWLEDGE_DIR}/软件部/通用"
    mkdir -p "${KNOWLEDGE_DIR}/硬件部/通用"

    # Copy project files (script is run from project dir)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cp -r "${SCRIPT_DIR}"/* "${RAG_DIR}/" 2>/dev/null || true

    log_info "Project files at ${RAG_DIR}"
}

# ── Step 4: Setup Python virtual environment ──
setup_python() {
    log_step "Setting up Python environment..."

    cd "${RAG_DIR}"

    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
    fi

    source venv/bin/activate

    pip install --upgrade pip -q

    # Install PyTorch (with or without CUDA)
    if [[ "${HAS_GPU}" == "true" ]]; then
        log_info "Installing PyTorch with CUDA support..."
        pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
    else
        log_info "Installing PyTorch (CPU only)..."
        pip install torch --index-url https://download.pytorch.org/whl/cpu -q
    fi

    # Install other dependencies
    pip install -r requirements.txt -q

    log_info "Python environment ready"
}

# ── Step 5: Start Docker services (Qdrant + PostgreSQL + Nginx) ──
start_docker_services() {
    log_step "Starting Docker services..."

    cd "${RAG_DIR}/docker"

    # Create docker-compose override for data volumes
    cat > docker-compose.override.yml <<'EOF'
version: "3.8"
services:
  qdrant:
    volumes:
      - /data/qdrant:/qdrant/storage
  postgres:
    volumes:
      - /data/postgres:/var/lib/postgresql/data
EOF

    docker compose up -d

    # Wait for services to be healthy
    log_info "Waiting for Qdrant..."
    for i in $(seq 1 30); do
        if curl -sf http://localhost:6333/healthz >/dev/null 2>&1; then
            log_info "Qdrant is ready"
            break
        fi
        sleep 2
    done

    log_info "Waiting for PostgreSQL..."
    for i in $(seq 1 30); do
        if docker exec rag-postgres pg_isready -U rag -d rag_enterprise >/dev/null 2>&1; then
            log_info "PostgreSQL is ready"
            break
        fi
        sleep 2
    done

    log_info "Docker services started"
}

# ── Step 6: Initialize database ──
init_database() {
    log_step "Initializing database..."

    cd "${RAG_DIR}"
    source venv/bin/activate

    python scripts/init_db.py

    log_info "Database initialized"
}

# ── Step 7: Create systemd services ──
create_systemd_services() {
    log_step "Creating systemd services..."

    # RAG API service
    cat > /etc/systemd/system/rag-api.service <<EOF
[Unit]
Description=RAG Enterprise API
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=${RAG_DIR}
Environment=PATH=${RAG_DIR}/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${RAG_DIR}/venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5
StandardOutput=append:${DATA_DIR}/logs/api.log
StandardError=append:${DATA_DIR}/logs/api.log

[Install]
WantedBy=multi-user.target
EOF

    # Folder watcher service
    cat > /etc/systemd/system/rag-watcher.service <<EOF
[Unit]
Description=RAG Folder Watcher
After=network.target rag-api.service

[Service]
Type=simple
User=root
WorkingDirectory=${RAG_DIR}
Environment=PATH=${RAG_DIR}/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${RAG_DIR}/venv/bin/python scripts/run_watcher.py
Restart=always
RestartSec=10
StandardOutput=append:${DATA_DIR}/logs/watcher.log
StandardError=append:${DATA_DIR}/logs/watcher.log

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable rag-api rag-watcher

    log_info "Systemd services created"
}

# ── Step 8: Start services ──
start_services() {
    log_step "Starting RAG services..."

    systemctl start rag-api
    sleep 3
    systemctl start rag-watcher

    log_info "Services started"
}

# ── Step 9: Verify deployment ──
verify_deployment() {
    log_step "Verifying deployment..."

    local all_ok=true

    # Check API
    if curl -sf http://localhost:8000/health | jq -e '.api == "ok"' >/dev/null 2>&1; then
        log_info "✓ API is healthy"
    else
        log_warn "✗ API health check failed"
        all_ok=false
    fi

    # Check Qdrant
    if curl -sf http://localhost:6333/healthz >/dev/null 2>&1; then
        log_info "✓ Qdrant is healthy"
    else
        log_warn "✗ Qdrant health check failed"
        all_ok=false
    fi

    # Check PostgreSQL
    if docker exec rag-postgres pg_isready -U rag -d rag_enterprise >/dev/null 2>&1; then
        log_info "✓ PostgreSQL is healthy"
    else
        log_warn "✗ PostgreSQL health check failed"
        all_ok=false
    fi

    # Check Ollama
    if ollama list >/dev/null 2>&1; then
        log_info "✓ Ollama is healthy"
    else
        log_warn "✗ Ollama not available"
        all_ok=false
    fi

    if [[ "${all_ok}" == "true" ]]; then
        echo ""
        log_info "========================================="
        log_info "  RAG Enterprise deployed successfully!"
        log_info "========================================="
        echo ""
        log_info "API endpoint: http://localhost:8000"
        log_info "API docs:     http://localhost:8000/docs"
        log_info "Health check: http://localhost:8000/health"
        echo ""
        log_info "Knowledge base: ${KNOWLEDGE_DIR}"
        log_info "  Drop documents into subfolders to auto-ingest"
        log_info "  Folder structure: /knowledge_base/{部门}/{项目}/"
        echo ""
        log_info "Useful commands:"
        log_info "  systemctl status rag-api      # Check API status"
        log_info "  systemctl status rag-watcher   # Check watcher status"
        log_info "  journalctl -u rag-api -f       # View API logs"
        log_info "  docker logs rag-qdrant          # View Qdrant logs"
        log_info "  docker logs rag-postgres         # View PostgreSQL logs"
    else
        log_warn "Some services failed health checks. Check logs for details."
    fi
}

# ── Main ──
main() {
    echo ""
    echo "============================================"
    echo "  RAG Enterprise - Linux Deployment Script"
    echo "============================================"
    echo ""

    check_root
    check_gpu
    install_system_deps
    setup_ollama
    setup_project
    setup_python
    start_docker_services
    init_database
    create_systemd_services
    start_services
    verify_deployment
}

main "$@"

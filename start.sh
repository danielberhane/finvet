#!/usr/bin/env bash
set -euo pipefail

# FinVet — Start all services
# Usage: ./start.sh
# Stop:  Ctrl+C (kills all background processes)

ROOT="$(cd "$(dirname "$0")" && pwd)"
SEC_MCP_DIR="${SEC_MCP_DIR:-$ROOT/../sec-edgar-mcp}"
PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "All processes stopped."
}
trap cleanup EXIT

wait_for_port() {
    local port=$1 name=$2 max=$3
    local i=0
    while ! python3 -c "import socket; s=socket.create_connection(('localhost',$port),1); s.close()" 2>/dev/null; do
        i=$((i + 1))
        if [ "$i" -ge "$max" ]; then
            echo "  FAIL: $name on port $port did not start"
            exit 1
        fi
        sleep 1
    done
    echo "  OK: $name (port $port)"
}

echo "=== FinVet Dev Environment ==="
echo ""

# --- 0. Ollama + Llama Guard (optional) ---
echo "[0/6] Ollama (Llama Guard)"
if command -v ollama &>/dev/null; then
    if ! pgrep -x ollama &>/dev/null; then
        ollama serve &>/dev/null &
        PIDS+=($!)
        sleep 2
    fi
    ollama pull llama-guard3:8b 2>/dev/null || true
    wait_for_port 11434 "Ollama" 10 2>/dev/null && echo "  OK: Ollama (port 11434)" || echo "  SKIP: Ollama not available (Llama Guard disabled)"
else
    echo "  SKIP: ollama not installed (Llama Guard disabled)"
fi

# --- 1. PostgreSQL (Docker) ---
echo "[1/6] PostgreSQL"
if docker compose ps postgres 2>/dev/null | grep -q "running"; then
    echo "  OK: already running (port 5432)"
else
    docker compose up -d postgres
    wait_for_port 5432 "PostgreSQL" 15
fi

# --- 2. DeepSeek API ---
echo "[2/6] DeepSeek API"
set -a; source "$ROOT/.env" 2>/dev/null || true; set +a
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "  FAIL: DEEPSEEK_API_KEY is not set. Add it to .env"
    exit 1
fi
if "$ROOT/.venv/bin/python" -c "
import httpx, os
r = httpx.post(
    'https://api.deepseek.com/chat/completions',
    headers={'Authorization': f'Bearer {os.environ[\"DEEPSEEK_API_KEY\"]}'},
    json={'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 1},
    timeout=10,
)
r.raise_for_status()
" 2>/dev/null; then
    echo "  OK: DeepSeek API reachable"
else
    echo "  FAIL: DeepSeek API unreachable or key invalid"
    exit 1
fi

# --- 3. SEC EDGAR MCP Server ---
echo "[3/6] SEC EDGAR MCP Server"
if python3 -c "import socket; s=socket.create_connection(('localhost',9870),1); s.close()" 2>/dev/null; then
    echo "  OK: already running (port 9870)"
else
    if [ ! -d "$SEC_MCP_DIR" ]; then
        echo "  FAIL: SEC MCP server not found at $SEC_MCP_DIR"
        exit 1
    fi
    (cd "$SEC_MCP_DIR" && python -m sec_edgar_mcp.server --transport streamable-http --port 9870) &
    PIDS+=($!)
    wait_for_port 9870 "SEC EDGAR MCP" 15
fi

# --- 4. FastAPI ---
echo "[4/6] FastAPI"
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
"$ROOT/.venv/bin/uvicorn" finvet.main:app --host 0.0.0.0 --port 8000 --app-dir "$ROOT/src" &
PIDS+=($!)
wait_for_port 8000 "FastAPI" 10

# --- 5. Streamlit UI ---
echo "[5/6] Streamlit UI"
lsof -ti :8501 | xargs kill -9 2>/dev/null || true
"$ROOT/.venv/bin/streamlit" run "$ROOT/ui/app.py" --server.port 8501 --server.address localhost --server.headless true &
PIDS+=($!)
wait_for_port 8501 "Streamlit" 10

echo ""
echo "=== All services running ==="
echo "  SEC EDGAR MCP:  http://localhost:9870"
echo "  API:            http://localhost:8000"
echo "  API docs:       http://localhost:8000/docs"
echo "  UI:             http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services."
wait

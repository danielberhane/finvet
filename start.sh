#!/usr/bin/env bash
set -euo pipefail

# FinVet — Start all services (maintainer script)
# Usage: ./start.sh
# Stop:  Ctrl+C (kills all background processes)
#
# Expects a checkout of the SEC EDGAR MCP server as a sibling directory
# (../sec-edgar-mcp) or SEC_MCP_DIR pointing at one, kills whatever holds
# ports 8000/8501, and exits if the configured LLM provider does not answer.
# For a fresh clone, `docker compose --profile sec up --build` (README) needs
# none of that.

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

# --- 2. LLM provider ---
# Checks whichever provider the three roles are configured for, rather than
# assuming DeepSeek: any role can be pointed elsewhere with LLM_<ROLE>__MODEL,
# __BASE_URL and __API_KEY_ENV, and this used to fail a MiniMax run for a
# missing DeepSeek key nothing would have used.
echo "[2/6] LLM provider"
set -a; source "$ROOT/.env" 2>/dev/null || true; set +a
if "$ROOT/.venv/bin/python" - <<'PY'
import os, sys
sys.path.insert(0, "src")
import httpx
from finvet.config.settings import settings

# settings is the single source of truth for which variable each role reads.
# Re-deriving it here from the env would hardcode the default a second time,
# and the two would drift the first time it changed.
roles = {r: getattr(settings, f"llm_{r}") for r in ("parser", "agent", "verdict")}
for name, cfg in roles.items():
    print(f"  {name:<8}: {cfg.model} @ {cfg.base_url}  (key: {cfg.api_key_env})")

missing = sorted({c.api_key_env for c in roles.values()
                  if not os.environ.get(c.api_key_env)})
if missing:
    print(f"  FAIL: not set in .env: {', '.join(missing)}")
    sys.exit(1)

# Reachability and auth for the agent role, which is the one that does the
# work. GET /models rather than a completion: it proves the same two things
# without generating a token, so a warm-up delay on the provider does not fail
# startup. A 15s completion probe timed out against a gateway that was fine.
agent = roles["agent"]
try:
    r = httpx.get(
        agent.base_url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + os.environ[agent.api_key_env]},
        timeout=15,
    )
    r.raise_for_status()
except Exception as exc:
    # The reason, not a traceback. This runs at startup, where the useful
    # output is one line naming what could not be reached.
    print(f"  FAIL: {agent.base_url} did not answer: "
          f"{type(exc).__name__}: {exc}")
    sys.exit(1)
PY
then
    echo "  OK: LLM provider reachable"
else
    echo "  Fix the above, or point the roles elsewhere with LLM_<ROLE>__BASE_URL."
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

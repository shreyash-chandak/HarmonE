#!/usr/bin/env bash
# harmone_start_wsl.sh — HarmonE launcher for WSL / headless Linux environments.
#
# Unlike harmone_start.sh (Arch), this script does NOT spawn GUI terminal windows.
# All three processes run in the background; PIDs are written to logs/harmone.pids.
# Logs: logs/acp.log, logs/dashboard.log, logs/wrapper.log
#
# Usage:
#   cd tool/
#   ./harmone_start_wsl.sh [--config <dataset>]
#     --config: dataset name used for preflight validation (default: pems_node1)
#
# Stop:   ./harmone_stop.sh
# Status: cat logs/harmone.pids | xargs ps -p

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Parse optional --config argument
CONFIG_NAME="pems_node1"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_NAME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "=============================="
echo "  HarmonE WSL Launcher"
echo "  Config: $CONFIG_NAME"
echo "=============================="

# Source shared setup (venv creation, pip install, preflight)
# shellcheck source=scripts/_launch_common.sh
source "$SCRIPT_DIR/scripts/_launch_common.sh"

cd "$PROJECT_DIR"

# ── Preflight ──────────────────────────────────────────────────────────────────

_preflight_checks "$CONFIG_NAME" || true   # non-fatal; print only

# ── Launch processes in background ────────────────────────────────────────────

PID_FILE="$LOG_DIR/harmone.pids"
> "$PID_FILE"   # truncate any stale pidfile

echo ""
echo "[WSL] Starting ACP server  → logs/acp.log"
"$PYTHON_CMD" app.py > "$LOG_DIR/acp.log" 2>&1 &
ACP_PID=$!
echo "$ACP_PID" >> "$PID_FILE"

echo "[WSL] Starting dashboard   → logs/dashboard.log"
(cd frontend && "$PYTHON_CMD" -m http.server 8000) > "$LOG_DIR/dashboard.log" 2>&1 &
DASH_PID=$!
echo "$DASH_PID" >> "$PID_FILE"

echo "[WSL] Starting managed system → logs/wrapper.log"
"$PYTHON_CMD" run_managed_system.py > "$LOG_DIR/wrapper.log" 2>&1 &
WRAPPER_PID=$!
echo "$WRAPPER_PID" >> "$PID_FILE"

echo "[WSL] PIDs written to $PID_FILE (acp=$ACP_PID, dash=$DASH_PID, wrapper=$WRAPPER_PID)"

# ── Health wait (poll :5000 and :8080 for up to 30 s) ────────────────────────

echo ""
echo "[WSL] Waiting for services to come up (timeout 30 s) ..."

_wait_http() {
    local url="$1"
    local label="$2"
    local deadline=$(( $(date +%s) + 30 ))
    while [[ $(date +%s) -lt $deadline ]]; do
        if "$PYTHON_CMD" -c "
import urllib.request, sys
try:
    urllib.request.urlopen('$url', timeout=2)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
            echo "[WSL] ✔ $label is up at $url"
            return 0
        fi
        sleep 1
    done
    return 1
}

HEALTH_FAIL=0

if ! _wait_http "http://localhost:5000/" "ACP server"; then
    echo "[WSL] ✗ ACP server did not respond within 30 s. Last 30 lines of acp.log:" >&2
    tail -n 30 "$LOG_DIR/acp.log" >&2
    HEALTH_FAIL=1
fi

if ! _wait_http "http://localhost:8080/adaptor/health" "Inference adaptor"; then
    echo "[WSL] ✗ Inference adaptor did not respond within 30 s. Last 30 lines of wrapper.log:" >&2
    tail -n 30 "$LOG_DIR/wrapper.log" >&2
    HEALTH_FAIL=1
fi

if [[ $HEALTH_FAIL -eq 1 ]]; then
    echo "" >&2
    echo "[WSL] Health check failed. Run ./harmone_stop.sh to clean up." >&2
    exit 1
fi

# ── Success summary ────────────────────────────────────────────────────────────

echo ""
echo "=============================="
echo "  HarmonE running in WSL"
echo "  Dashboard:  http://localhost:8000/dashboard.html"
echo "  ACP server: http://localhost:5000/"
echo "  Logs:       $LOG_DIR/"
echo "  Stop:       ./harmone_stop.sh"
echo "=============================="

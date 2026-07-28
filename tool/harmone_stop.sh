#!/usr/bin/env bash
# harmone_stop.sh — Stop all HarmonE background processes.
#
# Kills PIDs recorded in logs/harmone.pids by harmone_start_wsl.sh,
# then runs a psutil sweep to catch any orphan inference/app processes.
#
# Usage (from tool/ directory):
#   ./harmone_stop.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/logs/harmone.pids"

echo "[stop] Stopping HarmonE processes ..."

# ── 1. Kill from pidfile ───────────────────────────────────────────────────────

if [[ -f "$PID_FILE" ]]; then
    while IFS= read -r pid; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && echo "[stop] Killed PID $pid"
        else
            echo "[stop] PID $pid not running (already gone)."
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    echo "[stop] Removed $PID_FILE"
else
    echo "[stop] No pidfile at $PID_FILE — using psutil sweep only."
fi

# ── 2. psutil sweep (catches any orphans not in pidfile) ──────────────────────

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [[ -n "$PYTHON_CMD" ]]; then
    "$PYTHON_CMD" -c "
import sys, signal
try:
    import psutil
except ImportError:
    print('[stop] psutil not installed — skipping sweep.')
    sys.exit(0)

targets = ['inference.py', 'app.py', 'run_managed_system.py', 'http.server']
killed = []
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmdline = ' '.join(proc.info['cmdline'] or [])
        if any(t in cmdline for t in targets):
            proc.send_signal(signal.SIGTERM)
            killed.append((proc.pid, cmdline[:80]))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
if killed:
    for pid, cmd in killed:
        print(f'[stop] psutil: terminated PID {pid} ({cmd})')
else:
    print('[stop] psutil sweep: no matching processes found.')
"
else
    echo "[stop] No Python found — skipping psutil sweep."
fi

echo "[stop] Done."

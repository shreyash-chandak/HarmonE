#!/usr/bin/env bash
# harmone_start.sh — HarmonE launcher for Arch Linux (GUI terminal emulator).
#
# Spawns three GUI terminal windows (gnome-terminal / konsole / xterm / etc.).
# For WSL or headless use, use harmone_start_wsl.sh instead.
#
# Usage:
#   cd tool/
#   ./harmone_start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

echo "=============================="
echo "  HarmonE: Setup + Launch (Arch)"
echo "=============================="

# Source shared setup (venv creation, pip install, preflight)
# shellcheck source=scripts/_launch_common.sh
source "$SCRIPT_DIR/scripts/_launch_common.sh"

cd "$PROJECT_DIR"

# ── RAPL permissions (Arch; skip silently if not available) ───────────────────

echo "[arch] Setting pyJoules RAPL energy permissions ..."
sudo chmod -R a+r /sys/class/powercap/intel-rapl/ 2>/dev/null \
    && echo "[arch]  RAPL readable." \
    || echo "[arch]  RAPL not available — EnergyMeter will use null meters."

# ── Detect terminal emulator ───────────────────────────────────────────────────

_detect_terminal() {
    for term in gnome-terminal konsole xfce4-terminal tilix xterm; do
        if command -v "$term" &>/dev/null; then
            echo "$term"
            return
        fi
    done
    echo ""
}

TERMINAL=$(_detect_terminal)

if [[ -z "$TERMINAL" ]]; then
    echo "ERROR: No supported terminal emulator found (tried gnome-terminal, konsole, xfce4-terminal, tilix, xterm)." >&2
    echo "       Install one, or use ./harmone_start_wsl.sh for headless use." >&2
    exit 1
fi

echo "[arch] Using terminal: $TERMINAL"

_launch_terminal() {
    local cmd="$1"
    case "$TERMINAL" in
        gnome-terminal) gnome-terminal -- bash -c "$cmd; exec bash" ;;
        konsole)        konsole -e bash -c "$cmd; exec bash" ;;
        xfce4-terminal) xfce4-terminal --hold -e "bash -c '$cmd; exec bash'" ;;
        tilix)          tilix -e "bash -c '$cmd; exec bash'" ;;
        xterm)          xterm -hold -e "bash -c '$cmd; exec bash'" ;;
    esac
}

# ── Launch terminals ───────────────────────────────────────────────────────────

echo "[arch] Launching ACP server ..."
_launch_terminal "cd $PROJECT_DIR; source $VENV_PATH; $PYTHON_CMD app.py"

echo "[arch] Launching dashboard ..."
_launch_terminal "cd $PROJECT_DIR/frontend; $PYTHON_CMD -m http.server 8000"

echo "[arch] Opening managed system console ..."
_launch_terminal "cd $PROJECT_DIR; source $VENV_PATH; echo 'Run: $PYTHON_CMD run_managed_system.py'; exec bash"

echo ""
echo "=============================="
echo "  All systems launched!"
echo "  Dashboard: http://localhost:8000/dashboard.html"
echo "  ACP server: http://localhost:5000/"
echo "=============================="

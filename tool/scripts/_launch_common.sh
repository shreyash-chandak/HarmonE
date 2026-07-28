#!/usr/bin/env bash
# scripts/_launch_common.sh — Shared setup steps sourced by harmone_start.sh (Arch)
# and harmone_start_wsl.sh (WSL/Linux headless).
#
# Exports after sourcing: PYTHON_CMD, PIP_CMD, PROJECT_DIR, VENV_PATH
# Requires: PROJECT_DIR must be set by the caller before sourcing.

set -euo pipefail

VENV_NAME="harmone_env"
VENV_PATH="$PROJECT_DIR/$VENV_NAME/bin/activate"
LOG_DIR="$PROJECT_DIR/logs"

# ── Detect Python ──────────────────────────────────────────────────────────────

_detect_python() {
    for cmd in python3 python python3.12 python3.11 python3.10; do
        if command -v "$cmd" &>/dev/null; then
            echo "$cmd"
            return
        fi
    done
    echo ""
}

PYTHON_CMD=$(_detect_python)

if [[ -z "$PYTHON_CMD" ]]; then
    echo "ERROR: No Python 3 interpreter found. Install Python 3 first." >&2
    exit 1
fi

echo "[common] Python: $($PYTHON_CMD --version 2>&1)"
PIP_CMD="$PYTHON_CMD -m pip"

# ── Virtual environment ────────────────────────────────────────────────────────

if [[ ! -d "$PROJECT_DIR/$VENV_NAME" ]]; then
    echo "[common] Creating virtual environment '$VENV_NAME' ..."
    "$PYTHON_CMD" -m venv "$PROJECT_DIR/$VENV_NAME"
    echo "[common] venv created."
else
    echo "[common] venv already exists."
fi

source "$VENV_PATH"
echo "[common] venv activated: $VIRTUAL_ENV"

# ── Install dependencies ───────────────────────────────────────────────────────

REQ_FILE="$PROJECT_DIR/requirements.txt"
if [[ -f "$REQ_FILE" ]]; then
    echo "[common] Installing dependencies from requirements.txt ..."
    pip install -q -r "$REQ_FILE" \
        --extra-index-url https://download.pytorch.org/whl/cpu
    echo "[common] Dependencies installed."
else
    echo "[common] WARNING: requirements.txt not found at $REQ_FILE" >&2
fi

# ── Log directory ──────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"

# ── Preflight checks ───────────────────────────────────────────────────────────

_preflight_checks() {
    local config_name="${1:-pems_node1}"
    echo "[common] Running preflight checks (config=$config_name) ..."

    # Dataset validator
    if "$PYTHON_CMD" scripts/validate_dataset.py --config "$config_name" 2>&1 | grep -q "FAIL"; then
        echo "[common] WARNING: Dataset validation found errors — check above output." >&2
    else
        echo "[common] Dataset validation OK."
    fi

    # Energy probe check (expected: all unavailable in WSL → null meters; not an error)
    "$PYTHON_CMD" -c "
from core.energy import get_backend_status
st = get_backend_status()
cpu_ok = st.get('cpu_backend') not in (None, 'none')
gpu_ok = st.get('gpu_backend') not in (None, 'none')
if not cpu_ok and not gpu_ok:
    print('[common]  Energy: all backends unavailable (expected in WSL) — EnergyMeter will use null meters; valid=false rows emitted.')
else:
    print(f'[common]  Energy: cpu_backend={st.get(\"cpu_backend\")}, gpu_backend={st.get(\"gpu_backend\")}')
" 2>/dev/null || echo "[common]  Energy probe check skipped (import error)."
}

export PYTHON_CMD PIP_CMD PROJECT_DIR VENV_PATH LOG_DIR
export -f _preflight_checks 2>/dev/null || true

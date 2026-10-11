#!/usr/bin/env bash
# scripts/stage1_diagnostics.sh — stage 1 remote diagnostics (2026-10-12):
#   1. power diagnostic: RAPL package power idle / harness paused / LSTM / ridge
#   2. SVR C sweep: 1,200-row SVRs, R², mJ per prediction, support vectors, size
#   3. retrain timing: one retrain per dataset/model, torch default vs 1 thread
# Everything goes under one folder, tool/diagnostics/stage1_<timestamp>/, whose
# path is printed at the end. Takes ~20 minutes. Run it with NOTHING else
# running on the machine (the power numbers depend on it).
#
# Usage (from tool/, after git pull):
#   bash scripts/stage1_diagnostics.sh

set -uo pipefail
TOOL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TOOL_DIR" || exit 1
# shellcheck disable=SC1091
source "$TOOL_DIR/../.venv/bin/activate" || { echo "venv not found"; exit 1; }

if pgrep -f "run_concurrent.py|overnight_regression.sh" >/dev/null; then
    echo "A harness run is still active — stop it first (power readings would be wrong)."
    exit 1
fi
if ! cat /sys/class/powercap/intel-rapl:0/energy_uj >/dev/null 2>&1; then
    echo "RAPL not readable — running the permission setup (needs sudo):"
    sudo bash scripts/setup_energy_permissions.sh || exit 1
fi

OUT="diagnostics/stage1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
{
    echo "git: $(git rev-parse --short HEAD)"
    echo "date: $(date)"
    echo "nproc: $(nproc)"
    lscpu 2>/dev/null | grep -E "Model name|Socket|Core|Thread|MHz" || true
    python3 -c "import sys, torch, sklearn; print('python', sys.version.split()[0], 'torch', torch.__version__, 'sklearn', sklearn.__version__)"
    echo "load: $(cat /proc/loadavg)"
} > "$OUT/system.txt" 2>&1

echo "[1/3] power diagnostic (~12 min)"
python3 experiments/power_diagnostic.py --out "$OUT" 2>&1 | tee "$OUT/power.log"
echo "[2/3] SVR C sweep (~5-10 min)"
python3 experiments/svr_c_sweep.py --out "$OUT" 2>&1 | grep -v "^INFO" | tee "$OUT/svr_c_sweep.log"
echo "[3/3] retrain timing (~2-5 min)"
for th in default 1; do
    python3 experiments/retrain_timing.py --out "$OUT" --threads "$th" 2>&1 | grep -v "^INFO" | tee -a "$OUT/retrain_timing.log"
done

rm -rf "$OUT/power_runs"/*/models "$OUT/svr_models"   # weights are not needed back
echo
echo "done. outputs in: $TOOL_DIR/$OUT"

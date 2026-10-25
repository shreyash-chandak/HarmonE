#!/usr/bin/env bash
# scripts/smoke_check.sh — the full regression grid as a fast smoke check
# (2026-10-18): every run with run_concurrent.py --smoke (no pacing, weights
# loaded once and reloaded only after a retrain/restore, fixed per-model
# energies from the real naive runs in --energy-from, retrains waited for),
# then scripts/check_runs.py over the results. Checks the plumbing — schedules,
# retrains reaching the stream, switching, VMR, file contents — before hours
# of real runs; its energy and timing are not results. Smoke runs overwrite
# earlier smoke runs of the same name in --runs-dir.
#
# Usage (from tool/; ~20-30 min on the remote for all 39 runs):
#   bash scripts/smoke_check.sh
#   bash scripts/smoke_check.sh --datasets "pems_driftinduced" --planners "naive_prt harmone_original"
# Options: --runs-dir runs_smoke  --energy-from runs_calib
# Result: <runs-dir>/check.txt (also printed); exit 1 if a run failed a check.

set -uo pipefail
TOOL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TOOL_DIR" || exit 1
VENV="$TOOL_DIR/../.venv/bin/activate"
# shellcheck disable=SC1090
[[ -f "$VENV" ]] && source "$VENV"

RUNS_DIR=runs_smoke
ENERGY_FROM=runs_calib
DATASETS="pems_driftinduced spot_prices_driftinduced uci_electricity_driftinduced"
PLANNERS="naive naive_prt random_switch random_switch_prt greedy_switch harmone_original violation_aware violation_aware_gated pareto bandit"
PY="${PYTHON:-python3}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs-dir) RUNS_DIR="$2"; shift 2 ;;
        --energy-from) ENERGY_FROM="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --planners) PLANNERS="$2"; shift 2 ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done
mkdir -p "$RUNS_DIR/logs"
echo "smoke check: $RUNS_DIR (energies from $ENERGY_FROM), git $(git rev-parse --short HEAD)"

t_start=$(date +%s)
for ds in $DATASETS; do
    for planner in $PLANNERS; do
        if [[ "$planner" == naive || "$planner" == naive_prt ]]; then
            runs=("ridge" "svr" "lstm")
        else
            runs=("")
        fi
        for m in "${runs[@]}"; do
            id="${ds}_${planner}${m:+_$m}_conc"
            t0=$(date +%s)
            "$PY" concurrent_harness/run_concurrent.py --dataset "$ds" --planner "$planner" ${m:+--pin-model "$m"} \
                --run-id "$id" --runs-dir "$RUNS_DIR" --seed 1 --smoke --smoke-energy-from "$ENERGY_FROM" \
                > "$RUNS_DIR/logs/$id.log" 2>&1
            echo "  $id: exit $? ($(( $(date +%s) - t0 )) s)"
        done
    done
done
echo "all runs: $(( ($(date +%s) - t_start) / 60 )) min"

"$PY" scripts/check_runs.py --runs "$RUNS_DIR" --naive-from "$RUNS_DIR" "$ENERGY_FROM" | tee "$RUNS_DIR/check.txt"
exit "${PIPESTATUS[0]}"

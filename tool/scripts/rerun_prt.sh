#!/usr/bin/env bash
# scripts/rerun_prt.sh — re-run the PRT runs after the 2026-10-14 trigger fix
# (with monitor_interval 260, `step % 3200 == 0` first held at step 41 600, so
# PeMS never retrained and Spot/UCI retrained 2-3 times instead of ~27/39).
#
# Moves the affected runs to old/prt_<timestamp>/ (nothing is deleted), then
# re-runs them in place, seed 1:
#   runs_calib/<ds>_naive_prt_<model>_conc   (9 runs)
#   runs_seed1/<ds>_random_switch_prt_conc   (3 runs)
# The calibration and the other planners do not use PRT and are kept.
# Needs the calibrated configs from the seed-1 pipeline (do not git checkout configs/).
#
# Usage (from tool/, after git pull; ~2.5 h):
#   tmux new -s prt 'bash scripts/rerun_prt.sh; read'
#   log: overnight/rerun_prt.log

set -uo pipefail
TOOL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TOOL_DIR" || exit 1
# shellcheck disable=SC1091
source "$TOOL_DIR/../.venv/bin/activate"
mkdir -p overnight
LOG=overnight/rerun_prt.log
log() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

python3 - <<'PY' || { log "ABORT: configs do not hold the seed-1 calibration (E_m should be ~20000 uJ at 7.5 ms pacing)"; exit 1; }
import json, sys
for ds in ["pems_driftinduced", "spot_prices_driftinduced", "uci_electricity_driftinduced"]:
    c = json.load(open(f"configs/datasets/{ds}.json"))
    print(ds, "E_m", c["E_m"], "E_M", c["E_M"], "min_accuracy", c["min_accuracy"])
    if c["E_m"] < 15000:
        sys.exit(1)
PY

ts="$(date +%Y%m%d_%H%M%S)"
mkdir -p "old/prt_$ts/runs_calib" "old/prt_$ts/runs_seed1"
for d in runs_calib/*_naive_prt_*_conc; do [[ -e "$d" ]] && mv "$d" "old/prt_$ts/runs_calib/"; done
for d in runs_seed1/*_random_switch_prt_conc; do [[ -e "$d" ]] && mv "$d" "old/prt_$ts/runs_seed1/"; done
log "=== rerun_prt start (git $(git rev-parse --short HEAD)); old runs in old/prt_$ts ==="

FAIL=0
run() {  # run <runs_dir> <run_id> <args...>
    local dir="$1" id="$2"; shift 2
    log "START $dir/$id"
    if python3 concurrent_harness/run_concurrent.py "$@" --run-id "$id" --runs-dir "$dir" --seed 1 \
            > "overnight/rerun_prt_$id.log" 2>&1; then
        log "DONE  $dir/$id  retrains=$(python3 -c "import json;print(json.load(open('$dir/$id/run_manifest.json'))['event_counters'].get('retrains'))")"
    else
        log "FAIL  $dir/$id (see overnight/rerun_prt_$id.log)"; FAIL=$((FAIL + 1))
    fi
}

for ds in pems_driftinduced spot_prices_driftinduced uci_electricity_driftinduced; do
    for m in ridge svr lstm; do
        run runs_calib "${ds}_naive_prt_${m}_conc" --dataset "$ds" --planner naive_prt --pin-model "$m"
    done
    run runs_seed1 "${ds}_random_switch_prt_conc" --dataset "$ds" --planner random_switch_prt
done

python3 scripts/aggregate_seeds.py --runs runs_calib runs_seed1 > overnight/summary_seed1.txt 2>&1
log "=== rerun_prt done, FAIL=$FAIL; summary in overnight/summary_seed1.txt ==="

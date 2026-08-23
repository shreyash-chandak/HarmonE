#!/usr/bin/env bash
# run_regression.sh — Run all HarmonE regression experiments sequentially.
#
# Datasets : pems, uci_electricity, spot_prices
# Planners : naive (×3 models), naive_prt (×3 models), random_switch,
#            random_switch_prt, greedy_switch, harmone_original,
#            violation_aware, pareto, bandit
#
# Naive and naive_prt baselines are run once per model (lstm, ridge, svr)
# so each model's fixed-model baseline is recorded separately.
# All other planners run once as a multi-model pool.
#
# Per-run artifacts (predictions.csv, mape_events.csv, mape_info.json,
# thresholds.json, run_manifest.json) are written to:
#   runs/<dataset>_<planner>_s<seed>/         (non-naive)
#   runs/<dataset>_naive_<model>_s<seed>/     (naive per-model)
#
# Full stdout+stderr is tee'd to:
#   runs/logs/<run_id>.log
#
# A master timeline is appended to:
#   runs/logs/master_regression.log
#
# Usage (from inside tool/):
#   bash scripts/run_regression.sh            # seed=1
#   bash scripts/run_regression.sh --seed 42  # explicit seed
#
# To resume after interruption, re-run the same script.
# Runs with an existing run_manifest.json are skipped automatically.

set -uo pipefail

SEED=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

TOOL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$TOOL_DIR/runs/logs"
mkdir -p "$LOG_DIR"

DATASETS=(pems uci_electricity spot_prices)
ADAPTIVE_PLANNERS=(random_switch random_switch_prt greedy_switch harmone_original violation_aware pareto bandit)
REGRESSION_MODELS=(lstm ridge svr)

MASTER_LOG="$LOG_DIR/master_regression.log"
FAILED_LOG="$LOG_DIR/failed_regression.log"
PASS=0
FAIL=0

log() {
    local msg
    msg="$(date -u +%Y-%m-%dT%H:%M:%SZ) | $*"
    echo "$msg"
    echo "$msg" >> "$MASTER_LOG"
}

_run_one() {
    local run_id="$1"; shift
    local run_dir="$TOOL_DIR/runs/$run_id"
    local run_log="$LOG_DIR/${run_id}.log"

    if [ -f "$run_dir/run_manifest.json" ]; then
        log "SKIP  | $run_id  (manifest exists)"
        return 0
    fi

    log "START | $run_id"
    if (
        cd "$TOOL_DIR"
        python experiments/run_experiment.py "$@" --run-dir "$run_dir" --verbose
    ) 2>&1 | tee "$run_log"; then
        log "DONE  | $run_id"
        PASS=$((PASS + 1))
    else
        log "FAIL  | $run_id  (see $run_log)"
        echo "$run_id" >> "$FAILED_LOG"
        FAIL=$((FAIL + 1))
    fi
}

log "=== Regression grid START ==="
log "TOOL_DIR : $TOOL_DIR"
log "Datasets : ${DATASETS[*]}"
log "Seed     : $SEED"

for dataset in "${DATASETS[@]}"; do

    # Naive: one run per model (pinned, never switches, no retraining)
    for model in "${REGRESSION_MODELS[@]}"; do
        _run_one "${dataset}_naive_${model}_s${SEED}" \
            --dataset "$dataset" --planner naive \
            --pin-model "$model" --seed "$SEED"
    done

    # Naive+PRT: one run per model (pinned, retrains every 3 200 steps)
    for model in "${REGRESSION_MODELS[@]}"; do
        _run_one "${dataset}_naive_prt_${model}_s${SEED}" \
            --dataset "$dataset" --planner naive_prt \
            --pin-model "$model" --seed "$SEED"
    done

    # All adaptive planners: one run each (full model pool)
    for planner in "${ADAPTIVE_PLANNERS[@]}"; do
        _run_one "${dataset}_${planner}_s${SEED}" \
            --dataset "$dataset" --planner "$planner" --seed "$SEED"
    done

done

log "=== Regression grid DONE | passed=$PASS failed=$FAIL ==="
if [ "$FAIL" -gt 0 ]; then
    log "Failed runs listed in: $FAILED_LOG"
    exit 1
fi

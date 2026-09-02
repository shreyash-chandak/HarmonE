#!/usr/bin/env bash
# run_regression.sh — Run all HarmonE regression experiments sequentially.
#
# Datasets : pems_driftinduced, uci_electricity_driftinduced, spot_prices_driftinduced
# Planners : naive (×3 models), naive_prt (×3 models), random_switch,
#            random_switch_prt, greedy_switch, harmone_original,
#            violation_aware, pareto, bandit
#
# Naive and naive_prt baselines are run once per model (lstm, ridge, svr)
# so each model's fixed-model baseline is recorded separately.
# All other planners run once as a multi-model pool.
#
# Drift induction (HarmonE.pdf §4.2-4.3): models are always TRAINED on the
# unaltered dataset — the *_driftinduced configs' train_path/data_path
# training-portion rows are byte-identical to the clean configs' (pems.json,
# uci_electricity.json, spot_prices.json), and weights_path is shared between
# the clean and drift-induced config for each dataset, so whichever run
# happens to execute first trains once and every other run just loads the
# saved weights. ALL streaming (naive included, matching the paper's own
# Table 2 methodology, which evaluates every baseline against the same
# drift-induced test set) reads from the drift-injected stream file instead
# — see managed_system_regression/utility/drift/induce.py and the
# *_driftinduced.json configs' own _comment fields for the full mechanism.
#
# Prerequisite: the three *_driftInduced.csv files must already exist (this
# script does not generate them). Build them first, e.g.:
#   python managed_system_regression/utility/drift/induce.py --dataset pems \
#       --region START END SCALE SHIFT [--region START END SCALE SHIFT ...]
#   (repeat for uci_electricity, spot_prices)
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

DATASETS=(pems_driftinduced uci_electricity_driftinduced spot_prices_driftinduced)
ADAPTIVE_PLANNERS=(random_switch random_switch_prt greedy_switch harmone_original violation_aware pareto bandit)
REGRESSION_MODELS=(lstm ridge svr)

# Prerequisite: the drift-induced CSVs referenced by each *_driftinduced.json
# config must already exist — nothing in this script generates them.
declare -A DRIFT_CSV
DRIFT_CSV["pems_driftinduced"]="$TOOL_DIR/data/pems/flow_data_test_driftInduced.csv"
DRIFT_CSV["uci_electricity_driftinduced"]="$TOOL_DIR/data/uci_electricity/uci_electricity_driftInduced.csv"
DRIFT_CSV["spot_prices_driftinduced"]="$TOOL_DIR/data/spot_prices/spot_prices_driftInduced.csv"

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

MISSING=0
for dataset in "${DATASETS[@]}"; do
    csv="${DRIFT_CSV[$dataset]}"
    if [ ! -f "$csv" ]; then
        log "MISSING | $dataset  (expected $csv — run induce.py first, see this script's header)"
        MISSING=$((MISSING + 1))
    fi
done
if [ "$MISSING" -gt 0 ]; then
    log "=== Regression grid ABORTED | $MISSING drift-induced CSV(s) not found ==="
    exit 1
fi

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

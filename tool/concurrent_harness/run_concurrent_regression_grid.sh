#!/usr/bin/env bash
# concurrent_harness/run_concurrent_regression_grid.sh — Run the full
# regression grid through the concurrent (multi-process/multi-threaded)
# harness instead of the single-threaded experiments/run_experiment.py.
# CV equivalent: run_concurrent_cv_grid.sh.
#
# Mirrors scripts/run_regression.sh's structure and coverage exactly (same
# datasets, same planner/model matrix) so results are directly comparable —
# but invokes concurrent_harness/run_concurrent.py per combination instead.
# Nothing under experiments/, core/, adapters/, or scripts/run_regression.sh
# itself is read or modified by this script.
#
# Per-run artifacts (predictions.csv, mape_info.json, run_manifest.json) are
# written to:
#   concurrent_harness/runs/<dataset>_<planner>_conc/
#   concurrent_harness/runs/<dataset>_<planner>_<model>_conc/   (naive/naive_prt per-model)
#
# Full stdout+stderr is tee'd to:
#   concurrent_harness/runs/logs/<run_id>.log
# A master timeline is appended to:
#   concurrent_harness/runs/logs/master_regression_concurrent.log
#
# Usage (from inside tool/):
#   bash concurrent_harness/run_concurrent_regression_grid.sh
#
# To resume after interruption, re-run the same script — runs with an
# existing run_manifest.json are skipped automatically (run_concurrent.py's
# run_id is fixed/non-timestamped, same convention as run_regression.sh's
# own run_dir naming, specifically so this resume check works).
#
# Note: unlike run_experiment.py --seed, the concurrent harness has no
# reproducibility guarantee (see the approved plan's "Open risks" — real
# thread/process scheduling means two runs of the same config won't produce
# bit-identical output). --seed is accepted and recorded in the manifest for
# schema parity only.

set -uo pipefail

SEED=1
ONLY_NAIVE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        --only-naive) ONLY_NAIVE=1; shift ;;   # naive runs only, for experiments/calibrate_from_naive_runs.py
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

HARNESS_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOL_DIR="$(cd "$HARNESS_DIR/.." && pwd)"
RUNS_DIR="$HARNESS_DIR/runs"
LOG_DIR="$RUNS_DIR/logs"
mkdir -p "$LOG_DIR"

DATASETS=(pems_driftinduced uci_electricity_driftinduced spot_prices_driftinduced)
ADAPTIVE_PLANNERS=(random_switch random_switch_prt greedy_switch harmone_original violation_aware pareto bandit)
REGRESSION_MODELS=(lstm ridge svr)

declare -A DRIFT_CSV
DRIFT_CSV["pems_driftinduced"]="$TOOL_DIR/data/pems/flow_data_test_driftInduced.csv"
DRIFT_CSV["uci_electricity_driftinduced"]="$TOOL_DIR/data/uci_electricity/uci_electricity_driftInduced.csv"
DRIFT_CSV["spot_prices_driftinduced"]="$TOOL_DIR/data/spot_prices/spot_prices_driftInduced.csv"

# No per-dataset --stream-delay-s override anymore. plan_thread.py/
# drift_thread.py (t1/t2) now trigger on accumulated row count
# (monitor_interval, default 50 — matching run_experiment.py's own
# per-monitor_interval cycle exactly) rather than on a fixed wall-clock
# schedule, so decision cadence no longer depends on how fast the stream
# runs. Every run uses each dataset config's own stream_delay_s (0.0 for all
# *_driftinduced configs) — i.e. full speed, matching run_experiment.py.

MASTER_LOG="$LOG_DIR/master_regression_concurrent.log"
FAILED_LOG="$LOG_DIR/failed_regression_concurrent.log"
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
    local run_dir="$RUNS_DIR/$run_id"
    local run_log="$LOG_DIR/${run_id}.log"

    if [ -f "$run_dir/run_manifest.json" ]; then
        log "SKIP  | $run_id  (manifest exists)"
        return 0
    fi

    log "START | $run_id"
    if (
        cd "$TOOL_DIR"
        python3 concurrent_harness/run_concurrent.py "$@" --run-id "$run_id" --runs-dir "$RUNS_DIR" --seed "$SEED"
    ) 2>&1 | tee "$run_log"; then
        log "DONE  | $run_id"
        PASS=$((PASS + 1))
    else
        log "FAIL  | $run_id  (see $run_log)"
        echo "$run_id" >> "$FAILED_LOG"
        FAIL=$((FAIL + 1))
    fi
}

log "=== Concurrent regression grid START ==="
log "TOOL_DIR : $TOOL_DIR"
log "Datasets : ${DATASETS[*]}"
log "Seed     : $SEED"

MISSING=0
for dataset in "${DATASETS[@]}"; do
    csv="${DRIFT_CSV[$dataset]}"
    if [ ! -f "$csv" ]; then
        log "MISSING | $dataset  (expected $csv — run induce.py first, see scripts/run_regression.sh's header)"
        MISSING=$((MISSING + 1))
    fi
done
if [ "$MISSING" -gt 0 ]; then
    log "=== Concurrent regression grid ABORTED | $MISSING drift-induced CSV(s) not found ==="
    exit 1
fi

for dataset in "${DATASETS[@]}"; do

    # Naive: one run per model (pinned). t1 still runs for naive (see
    # mape/manage.py — it starts unconditionally so the periodic Monitor/
    # Analyse/Log cycle happens even though naive's own decisions are always
    # noop, matching run_experiment.py's unconditional per-monitor_interval
    # cycle).
    for model in "${REGRESSION_MODELS[@]}"; do
        _run_one "${dataset}_naive_${model}_conc" \
            --dataset "$dataset" --planner naive --pin-model "$model"
    done

    # Naive+PRT: one run per model (pinned, t1 only, retrains on schedule).
    for model in "${REGRESSION_MODELS[@]}"; do
        _run_one "${dataset}_naive_prt_${model}_conc" \
            --dataset "$dataset" --planner naive_prt --pin-model "$model"
    done

    # All adaptive planners: one run each (full model pool).
    [ "$ONLY_NAIVE" -eq 1 ] && continue
    for planner in "${ADAPTIVE_PLANNERS[@]}"; do
        _run_one "${dataset}_${planner}_conc" \
            --dataset "$dataset" --planner "$planner"
    done

done

log "=== Concurrent regression grid DONE | PASS=$PASS FAIL=$FAIL ==="
[ "$FAIL" -eq 0 ]

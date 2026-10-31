#!/usr/bin/env bash
# concurrent_harness/run_concurrent_regression_grid.sh — one seed of the regression
# grid through the concurrent harness: 3 datasets x (6 single-model baselines + 8
# adaptive planners) = 42 runs, in an order shuffled deterministically by the seed
# across all datasets and planners (2026-10-31), so no planner always runs first
# or right after a heavy run. The order is logged to <runs-dir>/run_order.txt.
# CV equivalent: run_concurrent_cv_grid.sh.
#
# Per-run artifacts (predictions.csv, mape_info.json, run_manifest.json):
#   <runs-dir>/<dataset>_<planner>_conc/
#   <runs-dir>/<dataset>_<planner>_<model>_conc/   (naive/naive_prt per-model)
# Stdout+stderr of each run: <runs-dir>/logs/<run_id>.log
# Timeline: <runs-dir>/logs/master_regression_concurrent.log
#
# Usage (from inside tool/):
#   bash concurrent_harness/run_concurrent_regression_grid.sh --seed 1
# Options:
#   --static-naive-only  of the baselines, run only naive ridge/svr/lstm (no naive_prt)
#   --only-naive         baselines only (with --static-naive-only: the calibration runs)
#   --skip-naive         adaptive planners only
#   --run-gap-s N        pause between runs, default 60
#   --runs-dir DIR       default runs/
#   --configs-dir DIR    default configs/
#   --smoke-from DIR     smoke check (run_concurrent.py --smoke) with energies from DIR; no gap
#
# To resume after interruption, re-run the same command: runs whose
# run_manifest.json has status ok are skipped; failed or unfinished ones are
# moved to <runs-dir>/failed/ and rerun (run_id is fixed/non-timestamped,
# specifically so this check works).
#
# --seed seeds random/numpy in both processes and LSTM retrains (train_seed + seed).
# Thread and process timing still differs between runs, so results are close but
# not bit-identical.

set -uo pipefail

SEED=1
ONLY_NAIVE=0
SKIP_NAIVE=0
STATIC_NAIVE=0
RUN_GAP_S=60
RUNS_DIR=""
CONFIGS_DIR=""
SMOKE_FROM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        --only-naive) ONLY_NAIVE=1; shift ;;
        --skip-naive) SKIP_NAIVE=1; shift ;;
        --static-naive-only) STATIC_NAIVE=1; shift ;;
        --run-gap-s) RUN_GAP_S="$2"; shift 2 ;;
        --runs-dir) RUNS_DIR="$2"; shift 2 ;;
        --configs-dir) CONFIGS_DIR="$2"; shift 2 ;;
        --smoke-from) SMOKE_FROM="$2"; RUN_GAP_S=0; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

HARNESS_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOL_DIR="$(cd "$HARNESS_DIR/.." && pwd)"
RUNS_DIR="${RUNS_DIR:-$TOOL_DIR/runs}"
mkdir -p "$RUNS_DIR"
RUNS_DIR="$(cd "$RUNS_DIR" && pwd)"
LOG_DIR="$RUNS_DIR/logs"
mkdir -p "$LOG_DIR"

DATASETS=(pems_driftinduced uci_electricity_driftinduced spot_prices_driftinduced)
ADAPTIVE_PLANNERS=(switch switch_prt greedy_switch harmone_original violation_aware violation_aware_gated pareto bandit)
REGRESSION_MODELS=(lstm ridge svr)

declare -A DRIFT_CSV
DRIFT_CSV["pems_driftinduced"]="$TOOL_DIR/data/pems/flow_data_test_driftInduced.csv"
DRIFT_CSV["uci_electricity_driftinduced"]="$TOOL_DIR/data/uci_electricity/uci_electricity_driftInduced.csv"
DRIFT_CSV["spot_prices_driftinduced"]="$TOOL_DIR/data/spot_prices/spot_prices_driftInduced.csv"

# Each run uses its dataset config's own stream_delay_s (the pacing sleep between
# predictions; 7.5 ms for the *_driftinduced configs since 2026-10-13). MAPE
# cadence is row-count based, so it does not depend on stream speed.

MASTER_LOG="$LOG_DIR/master_regression_concurrent.log"
FAILED_LOG="$LOG_DIR/failed_regression_concurrent.log"
PASS=0
FAIL=0
RAN=0

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

    # Only a run that finished ok is skipped (2026-10-31): a failed manifest used
    # to count as done, so the pipeline's retry did nothing. Anything else is kept
    # under failed/ and rerun.
    if [ -f "$run_dir/run_manifest.json" ] && \
       python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("status") != "ok")' "$run_dir/run_manifest.json"; then
        log "SKIP  | $run_id  (finished ok)"
        return 0
    fi
    if [ -e "$run_dir" ]; then
        mkdir -p "$RUNS_DIR/failed"
        mv "$run_dir" "$RUNS_DIR/failed/${run_id}_$(date +%Y%m%d_%H%M%S)"
        log "MOVED | $run_id  (unfinished or failed run kept in failed/)"
    fi

    # Pause between runs so the package settles (2026-10-31; paper: 20 min).
    if [ "$RAN" -gt 0 ] && [ "$RUN_GAP_S" -gt 0 ]; then sleep "$RUN_GAP_S"; fi
    RAN=$((RAN + 1))

    log "START | $run_id"
    if (
        cd "$TOOL_DIR"
        python3 concurrent_harness/run_concurrent.py "$@" --run-id "$run_id" --runs-dir "$RUNS_DIR" --seed "$SEED" \
            ${CONFIGS_DIR:+--configs-dir "$CONFIGS_DIR"} \
            ${SMOKE_FROM:+--smoke --smoke-energy-from "$SMOKE_FROM"}
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

# The job list: "dataset planner model" (model "-" = full pool).
JOBS=()
for dataset in "${DATASETS[@]}"; do
    if [ "$SKIP_NAIVE" -eq 0 ]; then
        for model in "${REGRESSION_MODELS[@]}"; do
            JOBS+=("$dataset naive $model")
            [ "$STATIC_NAIVE" -eq 1 ] || JOBS+=("$dataset naive_prt $model")
        done
    fi
    if [ "$ONLY_NAIVE" -eq 0 ]; then
        for planner in "${ADAPTIVE_PLANNERS[@]}"; do
            JOBS+=("$dataset $planner -")
        done
    fi
done

# Deterministic shuffle by seed; the same command gives the same order on resume.
mapfile -t ORDER < <(printf '%s\n' "${JOBS[@]}" | python3 -c '
import random, sys
jobs = sorted(l.strip() for l in sys.stdin if l.strip())
random.Random(int(sys.argv[1])).shuffle(jobs)
print("\n".join(jobs))' "$SEED")
printf '%s\n' "${ORDER[@]}" > "$RUNS_DIR/run_order.txt"
log "Order    : ${#ORDER[@]} runs, see $RUNS_DIR/run_order.txt"

for job in "${ORDER[@]}"; do
    read -r dataset planner model <<< "$job"
    if [ "$model" = "-" ]; then
        _run_one "${dataset}_${planner}_conc" --dataset "$dataset" --planner "$planner"
    else
        _run_one "${dataset}_${planner}_${model}_conc" --dataset "$dataset" --planner "$planner" --pin-model "$model"
    fi
done

log "=== Concurrent regression grid DONE | PASS=$PASS FAIL=$FAIL ==="
[ "$FAIL" -eq 0 ]

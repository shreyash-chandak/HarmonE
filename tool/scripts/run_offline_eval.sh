#!/usr/bin/env bash
# run_offline_eval.sh — Offline GT-accuracy evaluation for all completed CV runs.
#
# Iterates over CV run directories (bdd100k, iwildcam, acdc) produced by
# run_cv.sh and calls experiments/offline_eval.py for each completed run.
#
# Outputs: {run_dir}/offline_eval.json for each evaluated run.
# Logs:    runs/logs/<run_id>_offline_eval.log per run
#          runs/logs/master_offline_eval.log   overall timeline
#
# Skips:
#   - Runs without run_manifest.json (not yet completed)
#   - Runs that already have offline_eval.json (already evaluated; delete to re-run)
#
# Prerequisites:
#   - run_cv.sh must have completed for the target datasets/seed
#   - Ground-truth label files must be present (column "label_path" in manifests)
#
# Usage (from inside tool/):
#   bash scripts/run_offline_eval.sh                         # seed=1, all CV datasets
#   bash scripts/run_offline_eval.sh --seed 42               # different seed
#   bash scripts/run_offline_eval.sh --dataset bdd100k       # one dataset only
#   bash scripts/run_offline_eval.sh --interval 500          # smaller reporting window

set -uo pipefail

SEED=1
FILTER_DATASET=""
INTERVAL=1000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)     SEED="$2"; shift 2 ;;
        --dataset)  FILTER_DATASET="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

TOOL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$TOOL_DIR/runs/logs"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/master_offline_eval.log"
PASS=0
FAIL=0
SKIP=0

log() {
    local msg
    msg="$(date -u +%Y-%m-%dT%H:%M:%SZ) | $*"
    echo "$msg"
    echo "$msg" >> "$MASTER_LOG"
}

_eval_one() {
    local run_id="$1"
    local dataset="$2"
    local run_dir="$TOOL_DIR/runs/$run_id"
    local eval_log="$LOG_DIR/${run_id}_offline_eval.log"

    if [ ! -f "$run_dir/run_manifest.json" ]; then
        log "SKIP  | $run_id  (no run_manifest.json — run not completed)"
        SKIP=$((SKIP + 1))
        return 0
    fi

    if [ -f "$run_dir/offline_eval.json" ]; then
        log "SKIP  | $run_id  (offline_eval.json exists — delete to re-evaluate)"
        SKIP=$((SKIP + 1))
        return 0
    fi

    log "START | $run_id"
    if (
        cd "$TOOL_DIR"
        python experiments/offline_eval.py \
            --run-dir "$run_dir" \
            --dataset "$dataset" \
            --interval "$INTERVAL"
    ) 2>&1 | tee "$eval_log"; then
        log "DONE  | $run_id"
        PASS=$((PASS + 1))
    else
        log "FAIL  | $run_id  (see $eval_log)"
        FAIL=$((FAIL + 1))
    fi
}

log "=== Offline eval START ==="
log "TOOL_DIR : $TOOL_DIR"
log "Seed     : $SEED"
log "Interval : $INTERVAL"
[ -n "$FILTER_DATASET" ] && log "Dataset  : $FILTER_DATASET (filtered)"

# ── bdd100k — detection ───────────────────────────────────────────────────────

if [ -z "$FILTER_DATASET" ] || [ "$FILTER_DATASET" = "bdd100k" ]; then
    for model in yolo_n yolo_s yolo_m; do
        _eval_one "bdd100k_naive_${model}_s${SEED}" "bdd100k"
    done
    for planner in random_switch greedy_switch harmone_original violation_aware pareto bandit; do
        _eval_one "bdd100k_${planner}_s${SEED}" "bdd100k"
    done
fi

# ── iwildcam — classification ─────────────────────────────────────────────────

if [ -z "$FILTER_DATASET" ] || [ "$FILTER_DATASET" = "iwildcam" ]; then
    for model in efficientnet_b0 resnet50 resnet101; do
        _eval_one "iwildcam_naive_${model}_s${SEED}" "iwildcam"
    done
    for planner in random_switch greedy_switch harmone_original violation_aware pareto bandit; do
        _eval_one "iwildcam_${planner}_s${SEED}" "iwildcam"
    done
fi

# ── acdc — segmentation ───────────────────────────────────────────────────────

if [ -z "$FILTER_DATASET" ] || [ "$FILTER_DATASET" = "acdc" ]; then
    for model in segformer_b0 segformer_b1 segformer_b2; do
        _eval_one "acdc_naive_${model}_s${SEED}" "acdc"
    done
    for planner in random_switch greedy_switch harmone_original violation_aware pareto bandit; do
        _eval_one "acdc_${planner}_s${SEED}" "acdc"
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────────

log "=== Offline eval DONE | passed=$PASS failed=$FAIL skipped=$SKIP ==="
if [ "$FAIL" -gt 0 ]; then
    log "Check individual logs in $LOG_DIR for details."
    exit 1
fi

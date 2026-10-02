#!/usr/bin/env bash
# concurrent_harness/run_concurrent_cv_grid.sh — Run the CV grid through the
# concurrent (multi-process/multi-threaded) harness instead of the
# single-threaded experiments/run_experiment.py.
# Regression equivalent: run_concurrent_regression_grid.sh.
#
# Mirrors scripts/run_cv.sh's structure and coverage — same datasets, same
# per-dataset model lists, same sentinel-file gate before attempting anything
# (so this script itself can never accidentally kick off the multi-hour
# initial-training bootstrap for a dataset whose weights aren't ready — that
# bootstrap does exist and does work in concurrent_harness (see
# manage.py -> _train_cv_models_if_missing), it's just deliberately not
# something a grid loop should trigger unattended). Nothing under
# experiments/, core/, adapters/, or scripts/run_cv.sh itself is read or
# modified by this script.
#
# iwildcam is EXCLUDED from this grid, same as scripts/run_cv.sh's own
# DATASETS array — its weights were never produced and it's been superseded
# by imagenet (C4) as the trial classification-drift dataset. iwildcam.json
# is untouched; nothing stops it being run directly via run_concurrent.py if
# ever wanted, it's just not part of the grid.
#
# Per-run artifacts (predictions.csv, mape_info.json, run_manifest.json,
# predictions/ raw CV outputs) are written directly to:
#   concurrent_harness/runs/<dataset>_<planner>_conc/
#   concurrent_harness/runs/<dataset>_<planner>_<model>_conc/   (naive/naive_prt per-model)
#
# Full stdout+stderr is tee'd to:
#   concurrent_harness/runs/logs/<run_id>.log
# A master timeline is appended to:
#   concurrent_harness/runs/logs/master_cv_concurrent.log
#
# Usage (from inside tool/):
#   bash concurrent_harness/run_concurrent_cv_grid.sh
#
# To resume after interruption, re-run the same script — runs with an
# existing run_manifest.json are skipped automatically (run_concurrent.py's
# run_id is fixed/non-timestamped, same convention as run_cv.sh's own
# run_dir naming, specifically so this resume check works).
#
# Note: unlike run_experiment.py --seed, the concurrent harness has no
# reproducibility guarantee (see run_concurrent_regression_grid.sh's own
# note on this) — --seed is accepted and recorded in the manifest for schema
# parity only.

set -uo pipefail

SEED=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

HARNESS_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOL_DIR="$(cd "$HARNESS_DIR/.." && pwd)"
RUNS_DIR="$HARNESS_DIR/runs"
LOG_DIR="$RUNS_DIR/logs"
mkdir -p "$LOG_DIR"

# Dataset sentinels — same paths and same meaning as scripts/run_cv.sh's own
# gate: a dataset only runs if its first model's weights already exist.
declare -A DATASET_SENTINEL
DATASET_SENTINEL["bdd100k"]="$TOOL_DIR/yolov8n.pt"
DATASET_SENTINEL["acdc"]="$TOOL_DIR/managed_system_cv/models/segformer_b0_acdc.pt"
DATASET_SENTINEL["imagenet"]="$TOOL_DIR/managed_system_cv/models/efficientnet_b0_imagenet.pth"
DATASET_SENTINEL["imagenet_c"]="$TOOL_DIR/managed_system_cv/models/efficientnet_b0_imagenet.pth"

declare -A DATASET_NAIVE_MODELS
DATASET_NAIVE_MODELS["bdd100k"]="yolo_n yolo_s yolo_m"
DATASET_NAIVE_MODELS["acdc"]="segformer_b0 segformer_b1 segformer_b2"
DATASET_NAIVE_MODELS["imagenet"]="efficientnet_b0 resnet50 resnet101"
DATASET_NAIVE_MODELS["imagenet_c"]="efficientnet_b0 resnet50 resnet101"

# All four support the CV inline PRT+VMR fine-tune path
# (_do_cv_inline_finetune) — classification, segmentation (SegFormer
# pseudo-labeled masks), and detection (YOLO pseudo-labeled boxes) are all
# covered. Verified for real this session: acdc (segmentation) and bdd100k
# (detection, including an actual YOLO retrain) both ran clean end to end.
declare -A DATASET_SUPPORTS_PRT
DATASET_SUPPORTS_PRT["bdd100k"]=1
DATASET_SUPPORTS_PRT["acdc"]=1
DATASET_SUPPORTS_PRT["imagenet"]=1
DATASET_SUPPORTS_PRT["imagenet_c"]=1

# No per-dataset --stream-delay-s anymore. plan_thread.py/drift_thread.py
# (t1/t2) now trigger on accumulated row count (monitor_interval, default
# 50 — matching run_experiment.py's own per-monitor_interval cycle exactly)
# rather than on a fixed wall-clock schedule, so decision cadence no longer
# depends on how fast the stream runs. Every run uses each dataset config's
# own stream_delay_s (i.e. full speed, matching run_experiment.py).

DATASETS=(bdd100k acdc imagenet imagenet_c)  # iwildcam excluded — see header comment
ADAPTIVE_PLANNERS=(random_switch random_switch_prt greedy_switch harmone_original violation_aware pareto bandit)

MASTER_LOG="$LOG_DIR/master_cv_concurrent.log"
FAILED_LOG="$LOG_DIR/failed_cv_concurrent.log"
PASS=0
FAIL=0
SKIP_WEIGHTS=0

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
        python concurrent_harness/run_concurrent.py "$@" --run-id "$run_id" --runs-dir "$RUNS_DIR" --seed "$SEED"
    ) 2>&1 | tee "$run_log"; then
        log "DONE  | $run_id"
        PASS=$((PASS + 1))
    else
        log "FAIL  | $run_id  (see $run_log)"
        echo "$run_id" >> "$FAILED_LOG"
        FAIL=$((FAIL + 1))
    fi
}

log "=== CV concurrent grid START ==="
log "TOOL_DIR : $TOOL_DIR"
log "Datasets : ${DATASETS[*]}"
log "Seed     : $SEED"

for dataset in "${DATASETS[@]}"; do
    sentinel="${DATASET_SENTINEL[$dataset]}"
    if [ ! -f "$sentinel" ]; then
        log "SKIP_DATASET | $dataset — weight sentinel not found: $sentinel"
        log "             | This dataset's initial-training bootstrap has not been run —"
        log "             | see context/cv_concurrent_plan.md. Not triggering it from this"
        log "             | grid loop (it can take hours); run it deliberately first."
        SKIP_WEIGHTS=$((SKIP_WEIGHTS + 1))
        continue
    fi

    # imagenet_c additionally needs its own synthetic-drift manifest (built by
    # managed_system_cv/utility/drift/induce_imagenet_c.py) — same gate as
    # scripts/run_cv.sh's own check.
    if [ "$dataset" = "imagenet_c" ] && [ ! -f "$TOOL_DIR/data/imagenet_c/imagenet_c_manifest.csv" ]; then
        log "SKIP_DATASET | imagenet_c — manifest not found: $TOOL_DIR/data/imagenet_c/imagenet_c_manifest.csv"
        log "             | Run managed_system_cv/utility/drift/induce_imagenet_c.py first."
        SKIP_WEIGHTS=$((SKIP_WEIGHTS + 1))
        continue
    fi

    read -ra naive_models <<< "${DATASET_NAIVE_MODELS[$dataset]}"

    # Naive: one run per model (pinned). t1 still runs for naive (see
    # mape/manage.py — starts unconditionally so periodic Monitor/Analyse/Log
    # cycles happen even though naive's own decisions are always noop).
    for model in "${naive_models[@]}"; do
        _run_one "${dataset}_naive_${model}_conc" \
            --dataset "$dataset" --planner naive --pin-model "$model"
    done

    # Naive+PRT: one run per model (pinned, t1 only, periodic pseudo-label
    # fine-tune + VMR every prt_interval steps — 500 for CV configs).
    if [ -n "${DATASET_SUPPORTS_PRT[$dataset]:-}" ]; then
        for model in "${naive_models[@]}"; do
            _run_one "${dataset}_naive_prt_${model}_conc" \
                --dataset "$dataset" --planner naive_prt --pin-model "$model"
        done
    fi

    # All adaptive planners: one run each (full model pool).
    for planner in "${ADAPTIVE_PLANNERS[@]}"; do
        _run_one "${dataset}_${planner}_conc" \
            --dataset "$dataset" --planner "$planner"
    done

done

log "=== CV concurrent grid DONE | PASS=$PASS FAIL=$FAIL SKIPPED_NO_WEIGHTS=$SKIP_WEIGHTS ==="
[ "$FAIL" -eq 0 ]

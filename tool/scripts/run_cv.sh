#!/usr/bin/env bash
# run_cv.sh — Run all HarmonE CV experiments sequentially.
#
# Dataset readiness
# -----------------
# C1 bdd100k      READY — yolov8n.pt + yolov8s.pt + yolov8m.pt present in tool/
# C2 iwildcam     NEEDS WEIGHTS — see "Getting weights" below
# C3 acdc         NEEDS WEIGHTS — see "Getting weights" below
#
# Getting weights
# ---------------
# iWildCam (EfficientNet-B0 / ResNet-50 / ResNet-101 fine-tuned on iWildCam-2020):
#
#   # Option 1 — WILDS benchmark pre-trained models
#   pip install wilds
#   python - <<'EOF'
#   from wilds import get_dataset
#   # This downloads the dataset AND official model checkpoints if available.
#   # Alternatively, use the WILDS model zoo:
#   # https://github.com/p-lambda/wilds#pretrained-models
#   EOF
#
#   # Option 2 — torchvision fine-tuned checkpoint (manual)
#   #   Train or download a checkpoint, then save the state dict as:
#   mkdir -p managed_system_cv/models
#   # efficientnet_b0_iwildcam.pt  (182-class classifier state dict)
#   # resnet50_iwildcam.pt
#   # resnet101_iwildcam.pt
#
# ACDC (SegFormer-B0 / SegFormer-B1 / SegFormer-B2 fine-tuned on Cityscapes/ACDC, 19 classes):
#
#   pip install transformers accelerate
#   python - <<'EOF'
#   import torch
#   from transformers import SegformerForSemanticSegmentation

#   # B0 — nvidia/segformer-b0-finetuned-cityscapes-1024-1024
#   m = SegformerForSemanticSegmentation.from_pretrained(
#       "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
#   )
#   torch.save(m.state_dict(), "managed_system_cv/models/segformer_b0_acdc.pt")

#   # B1 — nvidia/segformer-b1-finetuned-cityscapes-1024-1024
#   m1 = SegformerForSemanticSegmentation.from_pretrained(
#       "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
#   )
#   torch.save(m1.state_dict(), "managed_system_cv/models/segformer_b1_acdc.pt")

#   # B2 — nvidia/segformer-b2-finetuned-cityscapes-1024-1024
#   m2 = SegformerForSemanticSegmentation.from_pretrained(
#       "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
#   )
#   torch.save(m2.state_dict(), "managed_system_cv/models/segformer_b2_acdc.pt")
#   EOF
#
# Run these from inside tool/ before executing this script for iwildcam/acdc.
#
# Usage (from inside tool/):
#   bash scripts/run_cv.sh            # seed=1
#   bash scripts/run_cv.sh --seed 42  # explicit seed
#
# To skip datasets whose weights are not yet present, the script checks for the
# weight files and prints a warning instead of failing.

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

# Dataset sentinels and per-dataset naive model lists
declare -A DATASET_SENTINEL
DATASET_SENTINEL["bdd100k"]="$TOOL_DIR/yolov8n.pt"
DATASET_SENTINEL["iwildcam"]="$TOOL_DIR/managed_system_cv/models/efficientnet_b0_iwildcam.pt"
DATASET_SENTINEL["acdc"]="$TOOL_DIR/managed_system_cv/models/segformer_b0_acdc.pt"

declare -A DATASET_NAIVE_MODELS
DATASET_NAIVE_MODELS["bdd100k"]="yolo_n yolo_s yolo_m"
DATASET_NAIVE_MODELS["iwildcam"]="efficientnet_b0 resnet50 resnet101"
DATASET_NAIVE_MODELS["acdc"]="segformer_b0 segformer_b1 segformer_b2"

DATASETS=(bdd100k iwildcam acdc)
ADAPTIVE_PLANNERS=(random_switch greedy_switch harmone_original violation_aware pareto bandit)

MASTER_LOG="$LOG_DIR/master_cv.log"
FAILED_LOG="$LOG_DIR/failed_cv.log"
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

log "=== CV grid START ==="
log "TOOL_DIR : $TOOL_DIR"
log "Datasets : ${DATASETS[*]}"
log "Seed     : $SEED"

for dataset in "${DATASETS[@]}"; do
    sentinel="${DATASET_SENTINEL[$dataset]}"
    if [ ! -f "$sentinel" ]; then
        log "SKIP_DATASET | $dataset — weight sentinel not found: $sentinel"
        log "             | Run the weight download commands at the top of this script first."
        SKIP_WEIGHTS=$((SKIP_WEIGHTS + 1))
        continue
    fi

    # Naive: one run per model (pinned, never switches)
    read -ra naive_models <<< "${DATASET_NAIVE_MODELS[$dataset]}"
    for model in "${naive_models[@]}"; do
        _run_one "${dataset}_naive_${model}_s${SEED}" \
            --dataset "$dataset" --planner naive \
            --pin-model "$model" --seed "$SEED"
    done

    # All adaptive planners: one run each (full model pool)
    for planner in "${ADAPTIVE_PLANNERS[@]}"; do
        _run_one "${dataset}_${planner}_s${SEED}" \
            --dataset "$dataset" --planner "$planner" --seed "$SEED"
    done

done

log "=== CV grid DONE | passed=$PASS failed=$FAIL skipped_no_weights=$SKIP_WEIGHTS ==="
if [ "$FAIL" -gt 0 ]; then
    log "Failed runs listed in: $FAILED_LOG"
    exit 1
fi

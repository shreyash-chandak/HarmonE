#!/usr/bin/env bash
# run_cv.sh — Run all HarmonE CV experiments sequentially.
#
# Dataset readiness (checked against disk 2026-09-02 — see DATASET_SENTINEL below,
# which is the actual gate; this block is a human-readable summary of the same facts)
# -----------------
# C1 bdd100k      READY — yolov8n.pt + yolov8s.pt + yolov8m.pt present in tool/
# C3 acdc         READY — segformer_b0/b1/b2_acdc.pt present in
#                 managed_system_cv/models/ (already bootstrapped; no action needed —
#                 see "Getting weights" below if these are ever missing again)
# C4 imagenet     NEEDS WEIGHTS (2026-08-30) — the old efficientnet_b0/resnet50/
#                 resnet101_imagenet.pth files were DELETED: they had 1000-class
#                 heads, incompatible with the 100-class trim (DP26). Will
#                 auto-bootstrap on first real run (see "Getting weights" below)
#                 — needs torchvision + network access for the pretrained-weight
#                 download, neither available in every dev environment.
# C4 imagenet_c   NEEDS WEIGHTS (shares imagenet's, so same gate) AND its own
#                 manifest — data/imagenet_c/imagenet_c_manifest.csv doesn't
#                 exist until managed_system_cv/utility/drift/induce_imagenet_c.py
#                 is run (see that script's own usage docstring). The sentinel
#                 check below now verifies both. Manifest already exists as of
#                 2026-09-02 (1800 rows) — only the weight gate is still open.
# C2 iwildcam     EXCLUDED from this grid (2026-08-30) — weights were never
#                 produced (see DECISIONS_PENDING.md DP21's "CV initial train"
#                 note) and ImageNet (C4) has since been assimilated as its
#                 trial replacement per DP18. iwildcam.json is NOT deleted —
#                 experiments/run_experiment.py::_train_cv_models_if_missing()
#                 (added 2026-08-30) would now auto-fine-tune it from an
#                 ImageNet-pretrained backbone on its own train_split() if
#                 ever run directly; it's just not part of the active grid.
#
# Getting weights
# ---------------
# As of 2026-08-30, a dataset with missing weights_path files no longer needs
# manual sourcing — experiments/run_experiment.py auto-bootstraps: an
# ImageNet/COCO-pretrained backbone is fine-tuned (last finetune_n_layers
# layers, real labels from train_split()) and saved to weights_path before
# the run proceeds. The DATASET_SENTINEL check below still gates this
# script's grid (so a first real classification/segmentation/detection run
# isn't accidentally kicked off by every `naive`/`naive_prt`/planner
# invocation in the loop) — remove a dataset's sentinel file, or point it at
# a path that doesn't exist yet, to let auto-train populate it once, then
# re-run this script.
#
# ACDC's SegFormer weights were sourced manually (not via the auto-bootstrap
# above, which fine-tunes from a torchvision/COCO backbone — SegFormer's
# Cityscapes-pretrained checkpoints come from HuggingFace instead) and are
# already present on disk as of 2026-09-02. Kept here for reference only, in
# case they ever need regenerating — do NOT re-run this to "fix" anything;
# it would overwrite already-working weights for no reason:
#
#   pip install transformers accelerate
#   python3 - <<'EOF'
#   import torch
#   from transformers import SegformerForSemanticSegmentation
#   for size in ("b0", "b1", "b2"):
#       m = SegformerForSemanticSegmentation.from_pretrained(
#           f"nvidia/segformer-{size}-finetuned-cityscapes-1024-1024"
#       )
#       torch.save(m.state_dict(), f"managed_system_cv/models/segformer_{size}_acdc.pt")
#   EOF
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
DATASET_SENTINEL["imagenet"]="$TOOL_DIR/managed_system_cv/models/efficientnet_b0_imagenet.pth"
DATASET_SENTINEL["imagenet_c"]="$TOOL_DIR/managed_system_cv/models/efficientnet_b0_imagenet.pth"

declare -A DATASET_NAIVE_MODELS
DATASET_NAIVE_MODELS["bdd100k"]="yolo_n yolo_s yolo_m"
DATASET_NAIVE_MODELS["iwildcam"]="efficientnet_b0 resnet50 resnet101"
DATASET_NAIVE_MODELS["acdc"]="segformer_b0 segformer_b1 segformer_b2"
DATASET_NAIVE_MODELS["imagenet"]="efficientnet_b0 resnet50 resnet101"
DATASET_NAIVE_MODELS["imagenet_c"]="efficientnet_b0 resnet50 resnet101"

# Datasets where the CV inline periodic-retrain (PRT) + VMR fine-tune path
# (experiments/run_experiment.py::_do_cv_inline_finetune) does something.
# All three CV task families are covered as of the follow-up that extended
# PRT+VMR beyond classification-only (see context/DECISIONS_PENDING.md DP21):
# classification (iwildcam/imagenet/imagenet_c), segmentation (acdc, via
# SegFormer pseudo-labeled masks), detection (bdd100k, via YOLO pseudo-labeled
# boxes, adapted from managed_system_cv/retrain_tactics/pseudo_label.py).
declare -A DATASET_SUPPORTS_PRT
DATASET_SUPPORTS_PRT["bdd100k"]=1
DATASET_SUPPORTS_PRT["iwildcam"]=1
DATASET_SUPPORTS_PRT["acdc"]=1
DATASET_SUPPORTS_PRT["imagenet"]=1
DATASET_SUPPORTS_PRT["imagenet_c"]=1

DATASETS=(bdd100k acdc imagenet imagenet_c)  # iwildcam excluded — see header comment
ADAPTIVE_PLANNERS=(random_switch random_switch_prt greedy_switch harmone_original violation_aware pareto bandit)

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
        python3 experiments/run_experiment.py "$@" --run-dir "$run_dir" --verbose
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

    # imagenet_c additionally needs its own synthetic-drift manifest (built by
    # managed_system_cv/utility/drift/induce_imagenet_c.py) — the weight
    # sentinel alone doesn't cover it, and without it CVImageDirAdapter has
    # nothing to stream.
    if [ "$dataset" = "imagenet_c" ] && [ ! -f "$TOOL_DIR/data/imagenet_c/imagenet_c_manifest.csv" ]; then
        log "SKIP_DATASET | imagenet_c — manifest not found: $TOOL_DIR/data/imagenet_c/imagenet_c_manifest.csv"
        log "             | Run managed_system_cv/utility/drift/induce_imagenet_c.py first."
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

    # Naive+PRT: one run per model (pinned, periodic pseudo-label fine-tune +
    # VMR every 3200 steps). Classification datasets only — see
    # DATASET_SUPPORTS_PRT above.
    if [ -n "${DATASET_SUPPORTS_PRT[$dataset]:-}" ]; then
        for model in "${naive_models[@]}"; do
            _run_one "${dataset}_naive_prt_${model}_s${SEED}" \
                --dataset "$dataset" --planner naive_prt \
                --pin-model "$model" --seed "$SEED"
        done
    fi

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

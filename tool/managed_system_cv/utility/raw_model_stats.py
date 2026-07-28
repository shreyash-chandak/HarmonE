import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from core.energy import EnergyMeter

# -----------------------
# CONFIG
# -----------------------
IMAGES_DIR = Path("data/bdd100k/images/train")
LABELS_DIR = Path("data/bdd100k/labels/train")
MODELS = {
    # "yolo_m": "models/yolo_m_retrained.pt",
    # "yolo_s": "models/yolo_s_retrained.pt",
    "yolo_n": "models/yolo_n_retrained.pt"
}
STEP = 500     # step size
WINDOW = 50    # consecutive images per window
MAX_IMAGES = 70000  # stop when exceeding 70k
SAVE_DIR = Path("pilot")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Collect all image paths
image_files = sorted([p for p in IMAGES_DIR.glob("*.jpg")])
print(f"Found {len(image_files)} images in {IMAGES_DIR}")

# -----------------------
# Helper: IoU calculation & AP computation
# -----------------------
def box_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0


def yolo_to_xyxy(label, w, h):
    cls, cx, cy, bw, bh = label
    cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2
    return int(cls), [x1, y1, x2, y2]


def compute_ap(preds, n_gt):
    if len(preds) == 0 or n_gt == 0:
        return 0.0
    preds = sorted(preds, key=lambda x: -x[0])  # sort by confidence
    tp = np.array([p[1] for p in preds])
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    # VOC-style AP: interpolate precision
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    ap = np.trapz(precisions, recalls)
    return ap

# -----------------------
# Run Inference & Collect Stats
# -----------------------
block_results = []
aggregate_results = []
IOU_THRESHOLDS = [0.5, 0.75, 0.9]

for model_name, model_path in MODELS.items():
    print(f"\n🚀 Evaluating {model_name.upper()} from {model_path}")
    model = YOLO(model_path)

    model_confidences = []
    model_times = []
    model_energies = []
    model_maps = {thr: [] for thr in IOU_THRESHOLDS}

    for start_idx in range(0, MAX_IMAGES, STEP):
        end_idx = start_idx + WINDOW
        if end_idx >= len(image_files) or end_idx > MAX_IMAGES:
            break

        subset = image_files[start_idx:end_idx]
        if not subset:
            continue

        with EnergyMeter("block_eval", backend="auto") as _em:
            start_time = time.time()
            preds = model(subset, verbose=False, save=False)
            total_time = time.time() - start_time
        energy_usage = _em.total_uJ or 0.0

        confidences = []
        all_preds_thr = {thr: [] for thr in IOU_THRESHOLDS}
        n_gt_thr = {thr: 0 for thr in IOU_THRESHOLDS}

        for img_path, r in zip(subset, preds):
            h, w = r.orig_shape

            # Predictions
            pred_boxes = []
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    pred_boxes.append((int(b.cls), [x1, y1, x2, y2], float(b.conf)))
                    confidences.append(float(b.conf))

            # Ground truth
            label_file = LABELS_DIR / (img_path.stem + ".txt")
            gt_boxes = []
            if label_file.exists():
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            gt_boxes.append(yolo_to_xyxy(parts, w, h))

            for thr in IOU_THRESHOLDS:
                matched = set()
                for pred_cls, pred_box, conf in pred_boxes:
                    best_iou, best_gt = 0, None
                    for i, (gt_cls, gt_box) in enumerate(gt_boxes):
                        if gt_cls == pred_cls and i not in matched:
                            iou = box_iou(pred_box, gt_box)
                            if iou > best_iou:
                                best_iou, best_gt = iou, i
                    if best_iou >= thr:
                        all_preds_thr[thr].append((conf, 1))  # TP
                        matched.add(best_gt)
                    else:
                        all_preds_thr[thr].append((conf, 0))  # FP
                n_gt_thr[thr] += len(gt_boxes)

        # Compute AP for each threshold
        ap_results = {}
        for thr in IOU_THRESHOLDS:
            ap = compute_ap(all_preds_thr[thr], n_gt_thr[thr])
            ap_results[f"mAP@{thr}"] = ap
            model_maps[thr].append(ap)

        avg_conf = np.mean(confidences) if confidences else 0.0
        avg_time = total_time / len(subset)
        avg_energy = energy_usage / len(subset)

        print(f"{model_name} | {start_idx}-{end_idx-1} → Conf={avg_conf:.4f}, " + ", ".join([f"mAP@{thr}={ap_results[f'mAP@{thr}']:.3f}" for thr in IOU_THRESHOLDS]) + f", Time={avg_time:.4f}s, Energy={avg_energy:.1f} µJ")

        block_results.append({
            "model": model_name,
            "start_idx": start_idx,
            "end_idx": end_idx - 1,
            "avg_conf": avg_conf,
            **ap_results,
            "avg_time_sec": avg_time,
            "avg_energy_uJ": avg_energy,
            "num_detections": len(confidences)
        })

        model_confidences.extend(confidences)
        model_times.append(avg_time)
        model_energies.append(avg_energy)

    aggregate_results.append({
        "model": model_name,
        "overall_avg_conf": np.mean(model_confidences) if model_confidences else 0.0,
        **{f"overall_mAP@{thr}": np.mean(model_maps[thr]) if model_maps[thr] else 0.0 for thr in IOU_THRESHOLDS},
        "overall_avg_time_sec": np.mean(model_times) if model_times else 0.0,
        "overall_avg_energy_uJ": np.mean(model_energies) if model_energies else 0.0,
        "total_detections": len(model_confidences)
    })

# -----------------------
# Save Results
# -----------------------
df_blocks = pd.DataFrame(block_results)
df_agg = pd.DataFrame(aggregate_results)

blocks_path = SAVE_DIR / "avg_conf_time_energy_acc_by_block.csv"
agg_path = SAVE_DIR / "avg_conf_time_energy_acc_overall.csv"

df_blocks.to_csv(blocks_path, index=False)
df_agg.to_csv(agg_path, index=False)

print(f"\n✔ Block results saved to {blocks_path}")
print(f"✔ Aggregate results saved to {agg_path}")
print(df_agg)

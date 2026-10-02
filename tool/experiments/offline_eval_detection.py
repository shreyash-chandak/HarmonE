"""experiments/offline_eval_detection.py — Offline detection evaluation.

Part 1 of the 3-way offline eval split (2026-08-30 audit; see
offline_eval_classification.py / offline_eval_segmentation.py for the other
two). Predictions are logged during the stream by run_experiment.py
(predictions.csv rows: step, active_model, planner, proxy_acc/confidence,
energy; per-step boxes/scores/classes saved to predictions/*.npz) — this
module reads those back and checks them against ground truth to compute
mAP@0.5, mAP@0.75, and mAP@0.90.

Design (Option 2 — replay, matching the other two task modules): predictions
are NOT re-generated unless predictions/*.npz is missing (legacy runs);
GT boxes are read straight from the manifest's label_path files.

Fast path:  predictions/*.npz (boxes, scores, classes) saved during the run —
            no model loading needed.
Slow path:  re-runs YOLO.predict() per image using the local .pt weights —
            only for runs that pre-date the predictions/ directory.

"Overall" is a single pooled mAP computed over every image in the stream at
once (the standard corpus-level definition) — not an average of per-interval
means, which the previous single-file implementation used and which is
sensitive to how many images happen to fall in each interval.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.offline_eval_common import BDD_TO_COCO, has_saved_predictions, resolve_weights

IOU_THRESHOLDS: tuple[float, ...] = (0.50, 0.75, 0.90)


def _metric_key(iou_threshold: float) -> str:
    return f"map{round(iou_threshold * 100)}"


# ── GT loading ────────────────────────────────────────────────────────────────

def load_gt_boxes(
    label_path: str | None,
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (boxes_xyxy_abs, class_ids) for one image's GT.

    Supports two label formats (auto-detected by extension):
    - YOLO .txt: each line "class cx cy w h" (normalised coords)
    - BDD per-image .json: {"frames": [{"objects": [{"category":…, "box2d":{x1,y1,x2,y2}}]}]}

    Returns empty arrays when label_path is None, missing, or has no valid boxes.
    """
    _empty = (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32))
    if not label_path:
        return _empty
    p = Path(label_path)
    if not p.exists():
        return _empty

    if p.suffix.lower() == ".txt":
        boxes, classes = [], []
        with open(p) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                boxes.append([x1, y1, x2, y2])
                classes.append(cls)
        if not boxes:
            return _empty
        return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)

    if p.suffix.lower() == ".json":
        with open(p) as f:
            data = json.load(f)
        boxes, classes = [], []
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                box2d = obj.get("box2d")
                if not box2d:
                    continue
                cat = obj.get("category", "").lower()
                cls = BDD_TO_COCO.get(cat, -1)
                if cls < 0:
                    continue
                boxes.append([box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]])
                classes.append(cls)
        if not boxes:
            return _empty
        return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)

    return _empty


# ── mAP core ────────────────────────────────────────────────────────────────

def _compute_map(
    pred_boxes_list: list[np.ndarray],
    pred_scores_list: list[np.ndarray],
    pred_classes_list: list[np.ndarray],
    gt_boxes_list: list[np.ndarray],
    gt_classes_list: list[np.ndarray],
    num_classes: int,
    iou_threshold: float,
) -> float:
    """Class-aware mAP@iou_threshold over a list of images.

    Uses 11-point interpolated AP per class, then macro-averages over classes
    that have at least one GT instance (same convention as COCO/VOC).
    """
    import torch
    from torchvision.ops import box_iou as tv_box_iou

    per_class_aps: list[float] = []

    for cls in range(num_classes):
        img_preds: list[tuple[np.ndarray, np.ndarray]] = []
        img_gts: list[np.ndarray] = []

        for i in range(len(pred_boxes_list)):
            p_mask = pred_classes_list[i] == cls
            g_mask = gt_classes_list[i] == cls
            img_preds.append((pred_boxes_list[i][p_mask], pred_scores_list[i][p_mask]))
            img_gts.append(gt_boxes_list[i][g_mask])

        n_gt = sum(len(g) for g in img_gts)
        if n_gt == 0:
            continue  # class absent from GT → skip (not penalised)

        all_preds = [
            (img_i, float(s), pred_i)
            for img_i, (boxes, scores) in enumerate(img_preds)
            for pred_i, s in enumerate(scores)
        ]
        all_preds.sort(key=lambda x: x[1], reverse=True)

        matched: list[set] = [set() for _ in range(len(pred_boxes_list))]
        tp_arr, fp_arr = [], []

        for img_i, _score, pred_i in all_preds:
            p_box = img_preds[img_i][0][pred_i : pred_i + 1]
            g_boxes = img_gts[img_i]

            if len(g_boxes) == 0:
                tp_arr.append(0)
                fp_arr.append(1)
                continue

            ious = tv_box_iou(
                torch.tensor(p_box, dtype=torch.float32),
                torch.tensor(g_boxes, dtype=torch.float32),
            ).numpy()[0]

            best = int(ious.argmax())
            if ious[best] >= iou_threshold and best not in matched[img_i]:
                matched[img_i].add(best)
                tp_arr.append(1)
                fp_arr.append(0)
            else:
                tp_arr.append(0)
                fp_arr.append(1)

        tp_cum = np.cumsum(tp_arr).astype(float)
        fp_cum = np.cumsum(fp_arr).astype(float)
        recalls = tp_cum / n_gt
        precisions = tp_cum / (tp_cum + fp_cum)

        ap = sum(
            (precisions[recalls >= t].max() if (recalls >= t).any() else 0.0)
            for t in np.linspace(0.0, 1.0, 11)
        ) / 11.0
        per_class_aps.append(ap)

    return float(np.mean(per_class_aps)) if per_class_aps else 0.0


# ── Stream collection ─────────────────────────────────────────────────────────

def _collect_stream_predictions(
    config: dict,
    pred_df: pd.DataFrame,
    stream_images: list[str],
    stream_labels: list[str | None],
    run_path: Path | None,
) -> list[dict]:
    """One record per usable step: {step, model, pred_boxes, pred_scores,
    pred_classes, gt_boxes, gt_classes}. Fast path (predictions/*.npz)
    preferred; falls back to YOLO.predict() re-inference for legacy runs."""
    from PIL import Image

    records: list[dict] = []
    use_saved = run_path is not None and has_saved_predictions(run_path, "detection")
    pred_dir = (run_path / "predictions") if use_saved else None
    if use_saved:
        print(f"[EVAL detection] using saved predictions from {pred_dir}")

    _model_cache: dict = {}
    if not use_saved:
        from ultralytics import YOLO

    n_total = len(pred_df)
    for i, row in pred_df.iterrows():
        step = int(row["step"])
        model_name = str(row["active_model"])
        if step >= len(stream_images):
            continue
        img_path = stream_images[step]
        if not Path(img_path).exists():
            continue
        try:
            img = Image.open(img_path)
            img_w, img_h = img.width, img.height
        except Exception:
            continue

        label_path = stream_labels[step] if step < len(stream_labels) else None
        gt_boxes, gt_cls = load_gt_boxes(label_path, img_w, img_h)

        if use_saved:
            pred_file = pred_dir / f"step_{step:06d}.npz"
            if not pred_file.exists():
                continue
            try:
                data = np.load(pred_file)
            except Exception as exc:
                print(f"[EVAL detection] step {step}: {exc}")
                continue
            pred_boxes = data["boxes"].astype(np.float32)
            pred_scores = data["scores"].astype(np.float32)
            pred_classes = data["classes"].astype(np.int32)
        else:
            wp = resolve_weights(config, model_name)
            if not Path(wp).exists():
                continue
            if model_name not in _model_cache:
                _model_cache[model_name] = YOLO(wp)
                print(f"[EVAL detection] loaded {model_name}")
            yolo = _model_cache[model_name]
            try:
                from core.device import yolo_device
                results = yolo.predict(img_path, verbose=False, device=yolo_device())
            except Exception as exc:
                print(f"[EVAL detection] predict failed ({img_path}): {exc}")
                continue
            if results and results[0].boxes is not None:
                pred_boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
                pred_scores = results[0].boxes.conf.cpu().numpy().astype(np.float32)
                pred_classes = results[0].boxes.cls.cpu().numpy().astype(np.int32)
            else:
                pred_boxes = np.zeros((0, 4), dtype=np.float32)
                pred_scores = np.zeros(0, dtype=np.float32)
                pred_classes = np.zeros(0, dtype=np.int32)

        records.append({
            "step": step, "model": model_name,
            "pred_boxes": pred_boxes, "pred_scores": pred_scores, "pred_classes": pred_classes,
            "gt_boxes": gt_boxes, "gt_classes": gt_cls,
        })
        if (i + 1) % 500 == 0:
            print(f"[EVAL detection] {i + 1}/{n_total} steps processed ...")

    return records


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate_detection(
    config: dict,
    pred_df: pd.DataFrame,
    stream_images: list[str],
    stream_labels: list[str | None],
    interval_size: int,
    run_path: Path | None = None,
) -> dict:
    """mAP@0.5 / mAP@0.75 / mAP@0.90, overall (pooled) + per monitoring interval."""
    records = _collect_stream_predictions(config, pred_df, stream_images, stream_labels, run_path)
    if not records:
        return {"metric_names": [_metric_key(t) for t in IOU_THRESHOLDS], "overall": None,
                "interval_results": [], "n_intervals": 0}

    num_classes = config.get("num_classes", 80)

    overall: dict = {}
    for thr in IOU_THRESHOLDS:
        overall[_metric_key(thr)] = round(_compute_map(
            [r["pred_boxes"] for r in records], [r["pred_scores"] for r in records],
            [r["pred_classes"] for r in records], [r["gt_boxes"] for r in records],
            [r["gt_classes"] for r in records], num_classes=num_classes, iou_threshold=thr,
        ), 6)
    overall["n_images"] = len(records)

    n_steps = pred_df["step"].max() + 1 if len(pred_df) > 0 else 0
    n_intervals = max(1, (n_steps + interval_size - 1) // interval_size)
    by_interval: dict[int, list[dict]] = {i: [] for i in range(n_intervals)}
    for r in records:
        idx = r["step"] // interval_size
        if idx < n_intervals:
            by_interval[idx].append(r)

    interval_results = []
    for idx in range(n_intervals):
        items = by_interval[idx]
        if not items:
            continue
        models = [r["model"] for r in items]
        dominant_model = max(set(models), key=models.count)
        entry: dict = {"interval_idx": idx, "n_images": len(items), "model": dominant_model}
        for thr in IOU_THRESHOLDS:
            entry[_metric_key(thr)] = round(_compute_map(
                [r["pred_boxes"] for r in items], [r["pred_scores"] for r in items],
                [r["pred_classes"] for r in items], [r["gt_boxes"] for r in items],
                [r["gt_classes"] for r in items], num_classes=num_classes, iou_threshold=thr,
            ), 6)
        interval_results.append(entry)
        print(f"[EVAL detection] interval {idx}: "
              + "  ".join(f"{_metric_key(t)}={entry[_metric_key(t)]:.4f}" for t in IOU_THRESHOLDS)
              + f"  ({dominant_model})")

    return {
        "metric_names": [_metric_key(t) for t in IOU_THRESHOLDS],
        "overall": overall,
        "interval_results": interval_results,
        "n_intervals": len(interval_results),
    }

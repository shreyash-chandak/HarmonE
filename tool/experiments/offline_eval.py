"""
experiments/offline_eval.py — Offline GT-accuracy evaluation for CV runs (Option 2: replay).

Reads run_experiment.py output (runs/{run_id}/) and replays the recorded model-switch
decisions against ground-truth labels to compute true accuracy metrics:

  detection    (bdd100k) :  mAP@0.5  — YOLO.predict() per image + IoU matching vs GT
  segmentation (acdc)    :  mIoU     — via SegmentationAdapter.offline_accuracy() per image
  classification (iwildcam): top-1   — via ClassificationAdapter.offline_accuracy() per image

Design (Option 2 — replay):
  The run already logged which model was active at each step in predictions.csv.
  We re-read the dataset manifest to get the GT label path for each step and
  run the appropriate task adapter to compute the ground-truth metric.  No
  predictions are re-generated from scratch; model loading is cached so each
  model is loaded at most once per evaluation.

Limitation: uses the original model weights from the dataset config, not VMR
snapshots.  Post-retrain weight changes are not reproduced exactly — this gives
the accuracy of the pre-retrain model at retrain-triggered steps.

Usage (from inside tool/):
    python experiments/offline_eval.py \\
        --run-dir runs/bdd100k_harmone_original_s1 \\
        --dataset bdd100k \\
        [--interval 1000] \\
        [--output runs/bdd100k_harmone_original_s1/offline_eval.json]

Output JSON:
    {
      "dataset": "bdd100k",
      "task": "detection",
      "run_dir": "...",
      "interval_size": 1000,
      "metric_name": "map50",
      "interval_results": [
        {"interval_idx": 0, "metric": 0.42, "n_images": 1000, "model": "yolo_n"}
      ],
      "overall_metric": 0.44,
      "n_intervals": 5
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# BDD100K category names → COCO class IDs (YOLOv8 COCO-pretrained label space)
_BDD_TO_COCO: dict[str, int] = {
    "car":           2,
    "truck":         7,
    "bus":           5,
    "person":        0,
    "rider":         0,
    "bicycle":       1,
    "motor":         3,
    "motorcycle":    3,
    "traffic light": 9,
    "traffic sign":  11,
    "train":         6,
}

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))


# ── Config / manifest helpers ──────────────────────────────────────────────────

def _load_config(dataset: str) -> dict:
    path = _TOOL_DIR / "configs" / "datasets" / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with open(path) as f:
        return json.load(f)


def _build_stream_index(
    config: dict,
) -> tuple[list[str], list[str | None], list[int | None]]:
    """Return (image_paths, label_paths, inline_labels) for the stream portion only.

    Mirrors CVImageDirAdapter's split arithmetic so step=0 in predictions.csv
    corresponds to index 0 of the returned lists.

    label_paths:   path to a GT file (mask PNG for segmentation, YOLO .txt for
                   detection).  None when the manifest has no label_path column
                   or the cell is empty.
    inline_labels: integer GT label embedded directly in the manifest (iWildCam
                   `label` column).  None for detection / segmentation datasets.
    """
    manifest_csv = config.get("manifest_csv")
    data_root_str = config.get("data_root", ".")
    data_root = (
        Path(data_root_str)
        if Path(data_root_str).is_absolute()
        else _TOOL_DIR / data_root_str
    )

    image_paths: list[str] = []
    label_paths: list[str | None] = []
    inline_labels: list[int | None] = []

    if manifest_csv:
        manifest_path = (
            Path(manifest_csv)
            if Path(manifest_csv).is_absolute()
            else _TOOL_DIR / manifest_csv
        )
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row.get("input_path", row.get("image_path", ""))
                image_paths.append(str(data_root / img) if img else "")

                label_file = row.get("label_path", "")
                label_paths.append(str(data_root / label_file) if label_file else None)

                # iWildCam stores the GT class index directly in a 'label' column
                raw_label = row.get("label", "")
                try:
                    inline_labels.append(int(raw_label))
                except (ValueError, TypeError):
                    inline_labels.append(None)
    else:
        image_dir = config.get("image_dir", "")
        img_dir = (
            Path(image_dir) if Path(image_dir).is_absolute() else _TOOL_DIR / image_dir
        )
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(
            str(p) for p in img_dir.iterdir() if p.suffix.lower() in extensions
        )
        label_paths = [None] * len(image_paths)
        inline_labels = [None] * len(image_paths)

    n = len(image_paths)
    train_frac = float(config.get("train_frac", 0.8))
    val_frac = float(config.get("val_frac", 0.0))
    val_end = int(n * train_frac) + int(n * val_frac)
    return image_paths[val_end:], label_paths[val_end:], inline_labels[val_end:]


def _load_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"predictions.csv not found in {run_dir}")
    df = pd.read_csv(path)
    if "proxy_acc" not in df.columns:
        raise ValueError(
            f"Run at {run_dir} has no 'proxy_acc' column — "
            "this appears to be a regression run. offline_eval.py is for CV only."
        )
    return df


def _resolve_weights(config: dict, model_name: str) -> str:
    spec = config.get("models", {}).get(model_name, {})
    wp = spec.get("weights_path", "")
    if not Path(wp).is_absolute():
        wp = str(_TOOL_DIR / wp)
    return wp


# ── Detection GT loading + mAP helpers ────────────────────────────────────────

def _load_gt_boxes(
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
                cls = _BDD_TO_COCO.get(cat, -1)
                if cls < 0:
                    continue
                boxes.append([box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]])
                classes.append(cls)
        if not boxes:
            return _empty
        return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)

    return _empty


def _compute_map50(
    pred_boxes_list: list[np.ndarray],
    pred_scores_list: list[np.ndarray],
    pred_classes_list: list[np.ndarray],
    gt_boxes_list: list[np.ndarray],
    gt_classes_list: list[np.ndarray],
    num_classes: int = 80,
) -> float:
    """Class-aware mAP@IoU=0.5 over a list of images.

    Uses 11-point interpolated AP per class, then macro-averages over classes
    that have at least one GT instance (same convention as COCO/VOC).
    """
    import torch
    from torchvision.ops import box_iou as tv_box_iou

    per_class_aps: list[float] = []

    for cls in range(num_classes):
        # --- collect per-image predictions and GT for this class ---
        img_preds: list[tuple[np.ndarray, np.ndarray]] = []  # (boxes, scores) per image
        img_gts: list[np.ndarray] = []

        for i in range(len(pred_boxes_list)):
            p_mask = pred_classes_list[i] == cls
            g_mask = gt_classes_list[i] == cls
            img_preds.append((pred_boxes_list[i][p_mask], pred_scores_list[i][p_mask]))
            img_gts.append(gt_boxes_list[i][g_mask])

        n_gt = sum(len(g) for g in img_gts)
        if n_gt == 0:
            continue  # class absent from GT → skip (not penalised)

        # sort all predictions by confidence descending
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
            if ious[best] >= 0.5 and best not in matched[img_i]:
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

        # 11-point interpolated AP
        ap = sum(
            (precisions[recalls >= t].max() if (recalls >= t).any() else 0.0)
            for t in np.linspace(0.0, 1.0, 11)
        ) / 11.0
        per_class_aps.append(ap)

    return float(np.mean(per_class_aps)) if per_class_aps else 0.0


# ── Saved-prediction fast paths ───────────────────────────────────────────────

def _miou_from_saved_mask(
    pred_mask: np.ndarray,
    label_path: str,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """Compute mIoU from a saved uint8 class mask vs a GT mask PNG.  No model needed."""
    from PIL import Image

    gt = np.array(Image.open(label_path), dtype=np.int64)
    h, w = gt.shape[:2]

    pred = pred_mask.astype(np.int64)
    if pred.shape != (h, w):
        pred_img = Image.fromarray(pred_mask.astype(np.uint8), mode="L")
        pred_img = pred_img.resize((w, h), resample=Image.NEAREST)
        pred = np.array(pred_img, dtype=np.int64)

    iou_list = []
    for cls in range(num_classes):
        valid = gt != ignore_index
        tp = int(((pred == cls) & (gt == cls) & valid).sum())
        fp = int(((pred == cls) & (gt != cls) & valid).sum())
        fn = int(((pred != cls) & (gt == cls) & valid).sum())
        denom = tp + fp + fn
        if denom > 0:
            iou_list.append(tp / denom)
    return float(np.mean(iou_list)) if iou_list else 0.0


def _has_saved_predictions(run_path: Path, task: str) -> bool:
    """Return True if the run has a non-empty predictions/ directory."""
    pred_dir = run_path / "predictions"
    if not pred_dir.exists():
        return False
    ext = {"segmentation": "*.png", "classification": "*.txt", "detection": "*.npz"}.get(task, "")
    return ext != "" and any(pred_dir.glob(ext))


# ── Task-specific evaluation ───────────────────────────────────────────────────

def _eval_per_image(
    task: str,
    config: dict,
    pred_df: pd.DataFrame,
    stream_images: list[str],
    stream_labels: list[str | None],
    stream_inline_labels: list[int | None],
    interval_size: int,
    run_path: Path | None = None,
) -> tuple[list[dict], str]:
    """Per-image metric for segmentation (mIoU) and classification (top-1 accuracy).

    Fast path (preferred): if run_path/predictions/ contains saved per-step outputs
    from run_experiment.py, metrics are computed by GT comparison only — no model
    loading or GPU inference needed.

    Slow path (fallback): loads models and re-runs inference for runs that pre-date
    the predictions/ directory feature.
    """
    from PIL import Image as _PILImage

    metric_name = "miou" if task == "segmentation" else "accuracy"
    num_classes: int = config.get("num_classes", 19)
    ignore_index: int = config.get("ignore_index", 255)

    step_metrics: list[tuple[int, float, str]] = []
    n_total = len(pred_df)

    use_saved = run_path is not None and _has_saved_predictions(run_path, task)
    pred_dir = (run_path / "predictions") if use_saved else None
    if use_saved:
        print(f"[EVAL] using saved predictions from {pred_dir}")

    # Slow-path infrastructure — only initialised when no saved predictions exist
    _task_adapter: Any = None
    _model_cache: dict[str, Any] = {}
    if not use_saved:
        import torch as _torch
        from adapters.tasks.base import get_task_adapter
        _task_adapter = get_task_adapter(task, config)

    for i, row in pred_df.iterrows():
        step = int(row["step"])
        model_name = str(row["active_model"])

        if step >= len(stream_images):
            continue

        if task == "classification":
            inline_label = stream_inline_labels[step] if step < len(stream_inline_labels) else None
            label_path = stream_labels[step] if step < len(stream_labels) else None
            has_gt = (inline_label is not None) or (label_path and Path(label_path).exists())
            if not has_gt:
                continue
        else:
            label_path = stream_labels[step] if step < len(stream_labels) else None
            if not label_path or not Path(label_path).exists():
                continue
            inline_label = None

        if use_saved:
            # ── Fast path: load saved prediction, compare to GT ───────────
            try:
                if task == "segmentation":
                    pred_file = pred_dir / f"step_{step:06d}.png"
                    if not pred_file.exists():
                        continue
                    pred_mask = np.array(_PILImage.open(pred_file), dtype=np.uint8)
                    metric = _miou_from_saved_mask(pred_mask, label_path, num_classes, ignore_index)
                elif task == "classification":
                    pred_file = pred_dir / f"step_{step:06d}.txt"
                    if not pred_file.exists():
                        continue
                    pred_class = int(Path(pred_file).read_text().strip())
                    gt_class = inline_label
                    if gt_class is None:
                        with open(label_path) as _f:
                            gt_class = int(_f.read().strip())
                    metric = 1.0 if pred_class == gt_class else 0.0
                else:
                    continue
                step_metrics.append((step, metric, model_name))
            except Exception as exc:
                print(f"[EVAL] step {step} (saved pred): {exc}")
        else:
            # ── Slow path: load model, re-run inference ───────────────────
            image_path = stream_images[step]
            if not Path(image_path).exists():
                continue

            if model_name not in _model_cache:
                wp = _resolve_weights(config, model_name)
                if not Path(wp).exists():
                    print(f"[EVAL] weights not found for {model_name}: {wp} — skipping step {step}")
                    continue
                try:
                    _model_cache[model_name] = _task_adapter.load_model(model_name, wp)
                    print(f"[EVAL] loaded {model_name}")
                except Exception as exc:
                    print(f"[EVAL] could not load {model_name}: {exc}")
                    continue

            loaded_model = _model_cache[model_name]
            try:
                result = _task_adapter.infer(loaded_model, image_path)
                if task == "classification" and inline_label is not None:
                    pred_class = int(_torch.argmax(result, dim=-1).item())
                    metric = 1.0 if pred_class == inline_label else 0.0
                else:
                    metric = _task_adapter.offline_accuracy(result, label_path)
                step_metrics.append((step, metric, model_name))
            except Exception as exc:
                print(f"[EVAL] step {step} ({model_name}): {exc}")

        if (i + 1) % 500 == 0:
            print(f"[EVAL] {i + 1}/{n_total} steps processed ...")

    if not step_metrics:
        return [], metric_name

    n_steps = pred_df["step"].max() + 1 if len(pred_df) > 0 else 0
    n_intervals = max(1, (n_steps + interval_size - 1) // interval_size)

    per_interval: dict[int, list[tuple[float, str]]] = {i: [] for i in range(n_intervals)}
    for step, metric, model_name in step_metrics:
        idx = step // interval_size
        if idx < n_intervals:
            per_interval[idx].append((metric, model_name))

    interval_results = []
    for idx in range(n_intervals):
        items = per_interval[idx]
        if not items:
            continue
        vals = [v for v, _ in items]
        dominant_model = max(
            set(m for _, m in items),
            key=lambda m: sum(1 for _, mm in items if mm == m),
        )
        interval_results.append({
            "interval_idx": idx,
            "metric": round(float(np.mean(vals)), 6),
            "n_images": len(vals),
            "model": dominant_model,
        })

    return interval_results, metric_name


def _eval_detection_per_interval(
    config: dict,
    pred_df: pd.DataFrame,
    stream_images: list[str],
    stream_labels: list[str | None],
    _stream_inline_labels: list[int | None],
    interval_size: int,
    run_path: Path | None = None,
) -> tuple[list[dict], str]:
    """mAP@0.5 per monitoring interval.

    Fast path: if run_path/predictions/*.npz exist, reads saved boxes/scores/classes
    directly — no YOLO model loading needed.

    Slow path: runs YOLO.predict() per image using the local .pt weights.
    """
    from PIL import Image

    n_steps = pred_df["step"].max() + 1 if len(pred_df) > 0 else 0
    n_intervals = max(1, (n_steps + interval_size - 1) // interval_size)

    interval_data: dict[int, dict] = {
        i: {"model": None, "images": [], "labels": [], "steps": []} for i in range(n_intervals)
    }
    for _, row in pred_df.iterrows():
        step = int(row["step"])
        model_name = str(row["active_model"])
        if step >= len(stream_images):
            continue
        img_path = stream_images[step]
        if not Path(img_path).exists():
            continue
        idx = step // interval_size
        if idx >= n_intervals:
            continue
        if interval_data[idx]["model"] is None:
            interval_data[idx]["model"] = model_name
        lbl = stream_labels[step] if step < len(stream_labels) else None
        interval_data[idx]["images"].append(img_path)
        interval_data[idx]["labels"].append(lbl)
        interval_data[idx]["steps"].append(step)

    nc = config.get("num_classes", 80)
    interval_results = []
    use_saved = run_path is not None and _has_saved_predictions(run_path, "detection")
    if use_saved:
        pred_dir = run_path / "predictions"
        print(f"[EVAL] detection: using saved predictions from {pred_dir}")

    for idx in range(n_intervals):
        info = interval_data[idx]
        if not info["images"] or info["model"] is None:
            continue

        model_name = info["model"]
        print(f"[EVAL] interval {idx}: evaluating {len(info['images'])} images ({model_name}) ...")

        pred_boxes_list: list[np.ndarray] = []
        pred_scores_list: list[np.ndarray] = []
        pred_classes_list: list[np.ndarray] = []
        gt_boxes_list: list[np.ndarray] = []
        gt_classes_list: list[np.ndarray] = []
        n_valid = 0

        if use_saved:
            # ── Fast path: load npz predictions ──────────────────────────
            for img_path, lbl_path, step in zip(info["images"], info["labels"], info["steps"]):
                pred_file = pred_dir / f"step_{step:06d}.npz"
                if not pred_file.exists():
                    continue
                try:
                    data = np.load(pred_file)
                    img = Image.open(img_path)
                    img_w, img_h = img.width, img.height
                except Exception:
                    continue
                gt_boxes, gt_cls = _load_gt_boxes(lbl_path, img_w, img_h)
                pred_boxes_list.append(data["boxes"].astype(np.float32))
                pred_scores_list.append(data["scores"].astype(np.float32))
                pred_classes_list.append(data["classes"].astype(np.int32))
                gt_boxes_list.append(gt_boxes)
                gt_classes_list.append(gt_cls)
                n_valid += 1
        else:
            # ── Slow path: YOLO.predict() per image ──────────────────────
            from ultralytics import YOLO
            wp = _resolve_weights(config, model_name)
            if not Path(wp).exists():
                print(f"[EVAL] interval {idx}: weights not found for {model_name}: {wp}")
                continue
            if not hasattr(_eval_detection_per_interval, "_model_cache"):
                _eval_detection_per_interval._model_cache = {}
            if model_name not in _eval_detection_per_interval._model_cache:
                _eval_detection_per_interval._model_cache[model_name] = YOLO(wp)
                print(f"[EVAL] loaded {model_name}")
            yolo = _eval_detection_per_interval._model_cache[model_name]

            for img_path, lbl_path in zip(info["images"], info["labels"]):
                try:
                    img = Image.open(img_path)
                    img_w, img_h = img.width, img.height
                except Exception:
                    continue
                gt_boxes, gt_cls = _load_gt_boxes(lbl_path, img_w, img_h)
                try:
                    results = yolo.predict(img_path, verbose=False)
                except Exception as exc:
                    print(f"[EVAL] predict failed ({img_path}): {exc}")
                    pred_boxes_list.append(np.zeros((0, 4), dtype=np.float32))
                    pred_scores_list.append(np.zeros(0, dtype=np.float32))
                    pred_classes_list.append(np.zeros(0, dtype=np.int32))
                    gt_boxes_list.append(gt_boxes)
                    gt_classes_list.append(gt_cls)
                    n_valid += 1
                    continue
                if results and results[0].boxes is not None:
                    p_boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
                    p_scores = results[0].boxes.conf.cpu().numpy().astype(np.float32)
                    p_classes = results[0].boxes.cls.cpu().numpy().astype(np.int32)
                else:
                    p_boxes = np.zeros((0, 4), dtype=np.float32)
                    p_scores = np.zeros(0, dtype=np.float32)
                    p_classes = np.zeros(0, dtype=np.int32)
                pred_boxes_list.append(p_boxes)
                pred_scores_list.append(p_scores)
                pred_classes_list.append(p_classes)
                gt_boxes_list.append(gt_boxes)
                gt_classes_list.append(gt_cls)
                n_valid += 1

        if n_valid == 0:
            print(f"[EVAL] interval {idx}: no valid images — skipping")
            continue

        map50 = _compute_map50(
            pred_boxes_list, pred_scores_list, pred_classes_list,
            gt_boxes_list, gt_classes_list,
            num_classes=nc,
        )
        interval_results.append({
            "interval_idx": idx,
            "metric": round(map50, 6),
            "n_images": n_valid,
            "model": model_name,
        })
        print(f"[EVAL] interval {idx}: map50={map50:.4f}  ({model_name})")

    return interval_results, "map50"


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate_run(
    run_dir: str,
    dataset: str,
    interval_size: int = 1000,
    output_path: str | None = None,
) -> dict:
    """Evaluate a CV run against ground-truth labels.

    Args:
        run_dir:       Path to runs/{run_id}/ produced by run_experiment.py.
        dataset:       Dataset name matching configs/datasets/<name>.json.
        interval_size: Steps per reporting interval (default 1000).
        output_path:   Where to write JSON results. Defaults to {run_dir}/offline_eval.json.

    Returns:
        Results dict (same content written to output_path).
    """
    run_path = Path(run_dir)
    config = _load_config(dataset)
    task = config.get("task", "detection")

    print(f"[EVAL] run_dir={run_path}  dataset={dataset}  task={task}")
    pred_df = _load_predictions(run_path)
    stream_images, stream_labels, stream_inline_labels = _build_stream_index(config)
    print(
        f"[EVAL] predictions={len(pred_df)}  stream_images={len(stream_images)}"
        f"  interval_size={interval_size}"
    )

    if task == "detection":
        interval_results, metric_name = _eval_detection_per_interval(
            config, pred_df, stream_images, stream_labels, stream_inline_labels, interval_size,
            run_path=run_path,
        )
    elif task in ("segmentation", "classification"):
        interval_results, metric_name = _eval_per_image(
            task, config, pred_df, stream_images, stream_labels, stream_inline_labels, interval_size,
            run_path=run_path,
        )
    else:
        raise ValueError(
            f"Unsupported task '{task}'. Must be detection, segmentation, or classification."
        )

    overall = (
        float(np.mean([r["metric"] for r in interval_results]))
        if interval_results
        else None
    )

    output = {
        "dataset": dataset,
        "task": task,
        "run_dir": str(run_path.resolve()),
        "interval_size": interval_size,
        "metric_name": metric_name,
        "interval_results": interval_results,
        "overall_metric": round(overall, 6) if overall is not None else None,
        "n_intervals": len(interval_results),
    }

    if output_path is None:
        output_path = str(run_path / "offline_eval.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[EVAL] results written to {output_path}")
    if overall is not None:
        print(f"[EVAL] overall {metric_name}: {overall:.4f}")
    else:
        print("[EVAL] no valid intervals — check that label files exist")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline GT-accuracy evaluation for CV runs")
    parser.add_argument("--run-dir", required=True, help="Path to runs/{run_id}/ directory")
    parser.add_argument("--dataset", required=True, help="Dataset key (bdd100k | acdc | iwildcam)")
    parser.add_argument(
        "--interval", type=int, default=1000,
        help="Steps per reporting interval (default: 1000)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: {run-dir}/offline_eval.json)",
    )
    args = parser.parse_args()

    results = evaluate_run(
        run_dir=args.run_dir,
        dataset=args.dataset,
        interval_size=args.interval,
        output_path=args.output,
    )
    print(json.dumps(results, indent=2))

"""experiments/offline_eval_segmentation.py — Offline segmentation evaluation.

Part 3 of the 3-way offline eval split (2026-08-30 audit; see
offline_eval_detection.py / offline_eval_classification.py for the other
two). Computes mIoU from per-pixel predicted-class vs. ground-truth-class
masks.

Fast path:  predictions/*.png (saved uint8 class masks) — no model loading
            needed. Per-class TP/FP/FN pixel counts are accumulated across
            EVERY image in the stream (or interval) before dividing — the
            standard corpus-level ("dataset") mIoU definition — rather than
            averaging each image's own mIoU, which the previous single-file
            implementation effectively did via per-interval means of
            per-image values.
Slow path:  re-runs inference per image via the segmentation task adapter's
            offline_accuracy() (only for runs that pre-date the predictions/
            directory) — that method returns an already-divided per-image
            IoU, not raw counts, so this fallback keeps the old per-image
            mean-of-means approximation rather than the exact pooled metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.offline_eval_common import has_saved_predictions, resolve_weights


def _class_confusion_counts(
    pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int, ignore_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class (tp, fp, fn) arrays for ONE image (resizes pred to gt's shape
    via nearest-neighbour if the two don't already match)."""
    gt = gt_mask.astype(np.int64)
    pred = pred_mask.astype(np.int64)
    if pred.shape != gt.shape:
        from PIL import Image
        pred_img = Image.fromarray(pred_mask.astype(np.uint8), mode="L").resize(
            (gt.shape[1], gt.shape[0]), resample=Image.NEAREST,
        )
        pred = np.array(pred_img, dtype=np.int64)

    valid = gt != ignore_index
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    for cls in range(num_classes):
        tp[cls] = int(((pred == cls) & (gt == cls) & valid).sum())
        fp[cls] = int(((pred == cls) & (gt != cls) & valid).sum())
        fn[cls] = int(((pred != cls) & (gt == cls) & valid).sum())
    return tp, fp, fn


def _miou_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    denom = tp + fp + fn
    present = denom > 0
    if not present.any():
        return 0.0
    return float((tp[present] / denom[present]).mean())


def evaluate_segmentation(
    config: dict,
    pred_df: pd.DataFrame,
    stream_images: list[str],
    stream_labels: list[str | None],
    interval_size: int,
    run_path: Path | None = None,
) -> dict:
    """mIoU, overall (pooled across the whole stream) + per monitoring interval."""
    from PIL import Image as PILImage

    num_classes = int(config.get("num_classes", 19))
    ignore_index = int(config.get("ignore_index", 255))

    n_steps = pred_df["step"].max() + 1 if len(pred_df) > 0 else 0
    n_intervals = max(1, (n_steps + interval_size - 1) // interval_size)

    use_saved = run_path is not None and has_saved_predictions(run_path, "segmentation")
    pred_dir = (run_path / "predictions") if use_saved else None
    if use_saved:
        print(f"[EVAL segmentation] using saved predictions from {pred_dir}")

    _task_adapter: Any = None
    _model_cache: dict[str, Any] = {}
    if not use_saved:
        from adapters.tasks.base import get_task_adapter
        _task_adapter = get_task_adapter("segmentation", config)

    global_tp = np.zeros(num_classes, dtype=np.int64)
    global_fp = np.zeros(num_classes, dtype=np.int64)
    global_fn = np.zeros(num_classes, dtype=np.int64)
    interval_tp = {i: np.zeros(num_classes, dtype=np.int64) for i in range(n_intervals)}
    interval_fp = {i: np.zeros(num_classes, dtype=np.int64) for i in range(n_intervals)}
    interval_fn = {i: np.zeros(num_classes, dtype=np.int64) for i in range(n_intervals)}
    interval_models: dict[int, list[str]] = {i: [] for i in range(n_intervals)}
    # Slow-path fallback: per-image already-divided IoU (see module docstring).
    interval_slow_ious: dict[int, list[float]] = {i: [] for i in range(n_intervals)}
    global_slow_ious: list[float] = []
    n_images = 0

    n_total = len(pred_df)
    for i, row in pred_df.iterrows():
        step = int(row["step"])
        model_name = str(row["active_model"])
        if step >= len(stream_images):
            continue
        label_path = stream_labels[step] if step < len(stream_labels) else None
        if not label_path or not Path(label_path).exists():
            continue
        idx = step // interval_size

        if use_saved:
            pred_file = pred_dir / f"step_{step:06d}.png"
            if not pred_file.exists():
                continue
            try:
                pred_mask = np.array(PILImage.open(pred_file), dtype=np.uint8)
                gt_mask = np.array(PILImage.open(label_path), dtype=np.int64)
            except Exception as exc:
                print(f"[EVAL segmentation] step {step}: {exc}")
                continue
            tp, fp, fn = _class_confusion_counts(pred_mask, gt_mask, num_classes, ignore_index)
            global_tp += tp; global_fp += fp; global_fn += fn
            if idx < n_intervals:
                interval_tp[idx] += tp; interval_fp[idx] += fp; interval_fn[idx] += fn
                interval_models[idx].append(model_name)
            n_images += 1
        else:
            image_path = stream_images[step]
            if not Path(image_path).exists():
                continue
            if model_name not in _model_cache:
                wp = resolve_weights(config, model_name)
                if not Path(wp).exists():
                    print(f"[EVAL segmentation] weights not found for {model_name}: {wp} — skipping step {step}")
                    continue
                try:
                    _model_cache[model_name] = _task_adapter.load_model(model_name, wp)
                    print(f"[EVAL segmentation] loaded {model_name}")
                except Exception as exc:
                    print(f"[EVAL segmentation] could not load {model_name}: {exc}")
                    continue
            try:
                result = _task_adapter.infer(_model_cache[model_name], image_path)
                metric = _task_adapter.offline_accuracy(result, label_path)
            except Exception as exc:
                print(f"[EVAL segmentation] step {step} ({model_name}): {exc}")
                continue
            global_slow_ious.append(metric)
            if idx < n_intervals:
                interval_slow_ious[idx].append(metric)
                interval_models[idx].append(model_name)
            n_images += 1

        if (i + 1) % 500 == 0:
            print(f"[EVAL segmentation] {i + 1}/{n_total} steps processed ...")

    if n_images == 0:
        return {"overall": None, "interval_results": [], "n_intervals": 0}

    if use_saved:
        overall_miou = _miou_from_counts(global_tp, global_fp, global_fn)
    else:
        overall_miou = float(np.mean(global_slow_ious)) if global_slow_ious else 0.0

    overall = {"miou": round(overall_miou, 6), "n_images": n_images, "pooled": use_saved}

    interval_results = []
    for idx in range(n_intervals):
        if not interval_models[idx]:
            continue
        models = interval_models[idx]
        dominant_model = max(set(models), key=models.count)
        if use_saved:
            miou = _miou_from_counts(interval_tp[idx], interval_fp[idx], interval_fn[idx])
        else:
            vals = interval_slow_ious[idx]
            miou = float(np.mean(vals)) if vals else 0.0
        interval_results.append({
            "interval_idx": idx,
            "miou": round(miou, 6),
            "n_images": len(models),
            "model": dominant_model,
        })

    print(f"[EVAL segmentation] overall: mIoU={overall_miou:.4f}  n={n_images}"
          + ("" if use_saved else "  (slow-path approximation: mean of per-image mIoU)"))

    return {
        "overall": overall,
        "interval_results": interval_results,
        "n_intervals": len(interval_results),
    }

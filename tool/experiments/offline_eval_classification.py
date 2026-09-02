"""experiments/offline_eval_classification.py — Offline classification evaluation.

Part 2 of the 3-way offline eval split (2026-08-30 audit; see
offline_eval_detection.py / offline_eval_segmentation.py for the other two).
The simplest of the three: per-image predicted class vs. ground truth class,
reduced to a confusion matrix, from which accuracy / precision / recall / F1
and TP / FP / FN / TN are derived.

Fast path:  predictions/*.txt (predicted class index, saved during the run) —
            no model loading needed.
Slow path:  re-runs inference per image via the classification task adapter —
            only for runs that pre-date the predictions/ directory.

Multiclass TP/FP/FN/TN convention: reported as MICRO sums (aggregate counts
across all classes) — for single-label multiclass, micro-precision =
micro-recall = micro-F1 = accuracy exactly, so this is the well-defined,
standard choice (a per-class MEAN of small counts would not be). accuracy/
precision/recall/f1 are reported MACRO-averaged (the conventional "balanced"
multiclass summary, not dominated by majority classes). Per-class TP/FP/FN/TN
and support are included in full for anyone who wants the raw breakdown.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.offline_eval_common import has_saved_predictions, load_class_names, resolve_weights


def _collect_stream_predictions(
    config: dict,
    pred_df: pd.DataFrame,
    stream_images: list[str],
    stream_labels: list[str | None],
    stream_inline_labels: list[int | None],
    run_path: Path | None,
) -> list[dict]:
    """One record per usable step: {step, model, pred_class, gt_class}."""
    records: list[dict] = []
    use_saved = run_path is not None and has_saved_predictions(run_path, "classification")
    pred_dir = (run_path / "predictions") if use_saved else None
    if use_saved:
        print(f"[EVAL classification] using saved predictions from {pred_dir}")

    _task_adapter: Any = None
    _model_cache: dict[str, Any] = {}
    if not use_saved:
        import torch as _torch
        from adapters.tasks.base import get_task_adapter
        _task_adapter = get_task_adapter("classification", config)

    n_total = len(pred_df)
    for i, row in pred_df.iterrows():
        step = int(row["step"])
        model_name = str(row["active_model"])
        if step >= len(stream_images):
            continue

        inline_label = stream_inline_labels[step] if step < len(stream_inline_labels) else None
        label_path = stream_labels[step] if step < len(stream_labels) else None
        gt_class: int | None = inline_label
        if gt_class is None and label_path and Path(label_path).exists():
            try:
                with open(label_path) as f:
                    gt_class = int(f.read().strip())
            except Exception:
                gt_class = None
        if gt_class is None:
            continue

        if use_saved:
            pred_file = pred_dir / f"step_{step:06d}.txt"
            if not pred_file.exists():
                continue
            try:
                pred_class = int(pred_file.read_text().strip())
            except Exception as exc:
                print(f"[EVAL classification] step {step}: {exc}")
                continue
        else:
            image_path = stream_images[step]
            if not Path(image_path).exists():
                continue
            if model_name not in _model_cache:
                wp = resolve_weights(config, model_name)
                if not Path(wp).exists():
                    print(f"[EVAL classification] weights not found for {model_name}: {wp} — skipping step {step}")
                    continue
                try:
                    _model_cache[model_name] = _task_adapter.load_model(model_name, wp)
                    print(f"[EVAL classification] loaded {model_name}")
                except Exception as exc:
                    print(f"[EVAL classification] could not load {model_name}: {exc}")
                    continue
            try:
                result = _task_adapter.infer(_model_cache[model_name], image_path)
                pred_class = int(_torch.argmax(result, dim=-1).item())
            except Exception as exc:
                print(f"[EVAL classification] step {step} ({model_name}): {exc}")
                continue

        records.append({"step": step, "model": model_name, "pred_class": pred_class, "gt_class": gt_class})
        if (i + 1) % 500 == 0:
            print(f"[EVAL classification] {i + 1}/{n_total} steps processed ...")

    return records


def evaluate_classification(
    config: dict,
    pred_df: pd.DataFrame,
    stream_images: list[str],
    stream_labels: list[str | None],
    stream_inline_labels: list[int | None],
    interval_size: int,
    run_path: Path | None = None,
) -> dict:
    """Confusion-matrix-derived accuracy/precision/recall/F1 + TP/FP/FN/TN,
    overall (pooled) + per-interval accuracy."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    records = _collect_stream_predictions(
        config, pred_df, stream_images, stream_labels, stream_inline_labels, run_path,
    )
    if not records:
        return {"overall": None, "per_class": [], "interval_results": [], "n_intervals": 0}

    y_true = np.array([r["gt_class"] for r in records])
    y_pred = np.array([r["pred_class"] for r in records])
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0,
    )

    total = int(cm.sum())
    class_names = load_class_names(config)
    per_class: list[dict] = []
    tp_sum = fp_sum = fn_sum = tn_sum = 0
    for i, lbl in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        tp_sum += tp; fp_sum += fp; fn_sum += fn; tn_sum += tn
        per_class.append({
            "label": int(lbl),
            "class_name": class_names.get(int(lbl)) if class_names else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support": int(cm[i, :].sum()),
        })

    overall = {
        "accuracy": round(accuracy, 6),
        "precision_macro": round(float(precision), 6),
        "recall_macro": round(float(recall), 6),
        "f1_macro": round(float(f1), 6),
        "tp": tp_sum, "fp": fp_sum, "fn": fn_sum, "tn": tn_sum,
        "n_samples": total,
        "n_classes": len(labels),
    }

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
        correct = sum(1 for r in items if r["pred_class"] == r["gt_class"])
        interval_results.append({
            "interval_idx": idx,
            "accuracy": round(correct / len(items), 6),
            "n_images": len(items),
            "model": dominant_model,
        })

    print(f"[EVAL classification] overall: acc={accuracy:.4f}  precision={precision:.4f}  "
          f"recall={recall:.4f}  f1={f1:.4f}  n={total}")

    return {
        "overall": overall,
        "per_class": per_class,
        "interval_results": interval_results,
        "n_intervals": len(interval_results),
    }

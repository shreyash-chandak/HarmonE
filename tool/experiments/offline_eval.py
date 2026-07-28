"""
experiments/offline_eval.py — Offline ground-truth evaluation for CV.

Computes true mAP@0.5 per monitoring interval (aligned to interval boundaries
in predictions.csv) using ground-truth labels from the dataset adapter's
offline_labels() directory.

Usage:
    python experiments/offline_eval.py \
        --run-dir managed_system_cv/knowledge \
        --labels-dir data/bdd100k/labels/val \
        --interval 1000 \
        --output results/offline_eval.json

Output JSON schema:
    {
      "interval_results": [
        {"interval_idx": 0, "map50": 0.42, "n_images": 1000,
         "model": "yolo_n", "timestamp_start": ..., "timestamp_end": ...}
      ],
      "overall_map50": 0.44,
      "n_intervals": 5
    }

The mAP@0.5 computation uses the Ultralytics val() pipeline, which requires
YOLO-format label files (.txt) adjacent to the images. Only invoked for
intervals where offline_labels_available is True in the adapter config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate_run(
    run_dir: str,
    labels_dir: str,
    images_dir: str,
    interval_size: int = 1000,
    output_path: str | None = None,
    model_weights_dir: str = "managed_system_cv/models",
) -> dict:
    """Evaluate offline mAP@0.5 across all monitoring intervals in run_dir.

    Args:
        run_dir:           Path to knowledge/ directory containing predictions.csv.
        labels_dir:        Path to YOLO ground-truth label directory (.txt files).
        images_dir:        Path to corresponding images directory.
        interval_size:     Number of predictions per monitoring interval.
        output_path:       Optional path to write JSON results.
        model_weights_dir: Directory containing active model weights.

    Returns:
        Dict with interval_results and overall_map50.
    """
    predictions_path = Path(run_dir) / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions.csv not found in {run_dir}")

    df = pd.read_csv(predictions_path)
    n_rows = len(df)
    n_intervals = max(1, n_rows // interval_size)

    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    # Discover model weights for each interval from model.csv history
    # (Simplified: use the model that was active at each interval boundary)
    model_log = _load_model_log(run_dir)

    interval_results = []
    map50_values = []

    for i in range(n_intervals):
        start_row = i * interval_size
        end_row = min(start_row + interval_size, n_rows)
        interval_df = df.iloc[start_row:end_row]

        model_name = _model_at_row(model_log, start_row)
        weights_path = _find_weights(model_weights_dir, model_name)

        try:
            map50 = _compute_map50_for_interval(
                interval_df, images_path, labels_path, weights_path
            )
        except Exception as exc:
            print(f"[EVAL] Interval {i}: mAP@0.5 computation failed: {exc}")
            map50 = None

        result = {
            "interval_idx": i,
            "map50": map50,
            "n_images": end_row - start_row,
            "model": model_name,
            "row_start": start_row,
            "row_end": end_row,
        }
        interval_results.append(result)
        if map50 is not None:
            map50_values.append(map50)

    overall_map50 = float(np.mean(map50_values)) if map50_values else None

    output = {
        "interval_results": interval_results,
        "overall_map50": overall_map50,
        "n_intervals": n_intervals,
        "interval_size": interval_size,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[EVAL] Results written to {output_path}")

    return output


def _compute_map50_for_interval(
    interval_df: pd.DataFrame,
    images_path: Path,
    labels_path: Path,
    weights_path: str | None,
) -> float | None:
    """Run Ultralytics val() on the images in this interval."""
    if weights_path is None or not Path(weights_path).exists():
        print(f"[EVAL] Weights not found: {weights_path}")
        return None

    # Get image filenames from predictions.csv if present
    image_filenames = None
    if "image_path" in interval_df.columns:
        image_filenames = interval_df["image_path"].dropna().tolist()
    elif "filename" in interval_df.columns:
        image_filenames = interval_df["filename"].dropna().tolist()

    if not image_filenames:
        # Fall back: use all images in the directory (order by name)
        all_images = sorted(list(images_path.glob("*.jpg")) + list(images_path.glob("*.png")))
        n = len(interval_df)
        image_filenames = [str(p) for p in all_images[:n]]

    # Build a temporary YAML pointing to a symlinked subset
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_img_dir = Path(tmpdir) / "images"
        tmp_lbl_dir = Path(tmpdir) / "labels"
        tmp_img_dir.mkdir()
        tmp_lbl_dir.mkdir()

        for img_name in image_filenames:
            src_img = images_path / Path(img_name).name
            src_lbl = labels_path / (Path(img_name).stem + ".txt")
            if src_img.exists():
                shutil.copy2(src_img, tmp_img_dir / src_img.name)
            if src_lbl.exists():
                shutil.copy2(src_lbl, tmp_lbl_dir / src_lbl.name)

        yaml_path = Path(tmpdir) / "eval.yaml"
        yaml_path.write_text(f"""path: {tmpdir}
val: images
nc: 80
names: {{ {', '.join([f'{j}: class{j}' for j in range(80)])} }}
""")

        from ultralytics import YOLO
        model = YOLO(weights_path)
        metrics = model.val(data=str(yaml_path), imgsz=640, verbose=False)
        return float(metrics.box.map50)


def _load_model_log(run_dir: str) -> list[tuple[int, str]]:
    """Load event log from mape_info to reconstruct which model was active at each row."""
    mape_info_path = Path(run_dir) / "mape_info.json"
    if not mape_info_path.exists():
        return []
    with open(mape_info_path) as f:
        info = json.load(f)
    # event_log format: [{"row": int, "model": str, ...}] — if present
    return [(e.get("row", 0), e.get("model", "")) for e in info.get("event_log", [])]


def _model_at_row(model_log: list[tuple[int, str]], row: int) -> str:
    """Return the model name active at the given row index."""
    current = "yolo_n"
    for log_row, model_name in sorted(model_log):
        if log_row <= row:
            current = model_name
        else:
            break
    return current


def _find_weights(weights_dir: str, model_name: str) -> str | None:
    weights_path = Path(weights_dir) / f"{model_name}.pt"
    if weights_path.exists():
        return str(weights_path)
    # Also try base name without version suffix
    base = model_name.split("_v")[0]
    weights_path = Path(weights_dir) / f"{base}.pt"
    return str(weights_path) if weights_path.exists() else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline mAP@0.5 evaluation per interval")
    parser.add_argument("--run-dir", required=True, help="Path to knowledge/ directory")
    parser.add_argument("--labels-dir", required=True, help="Path to ground-truth YOLO labels dir")
    parser.add_argument("--images-dir", required=True, help="Path to images dir")
    parser.add_argument("--interval", type=int, default=1000, help="Monitoring interval size (rows)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--weights-dir", default="managed_system_cv/models")
    args = parser.parse_args()

    results = evaluate_run(
        run_dir=args.run_dir,
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        interval_size=args.interval,
        output_path=args.output,
        model_weights_dir=args.weights_dir,
    )
    print(json.dumps(results, indent=2))

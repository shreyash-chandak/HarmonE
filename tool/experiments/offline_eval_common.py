"""experiments/offline_eval_common.py — Shared helpers for offline_eval_*.py.

Split out of a single offline_eval.py (2026-08-30 audit, per request to
cleanly divide detection/classification/segmentation offline evaluation into
three parts) — config/manifest/predictions loading is identical across all
three tasks, so it lives here once instead of being copied three ways.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

# BDD100K category names → COCO class IDs (YOLOv8 COCO-pretrained label space).
BDD_TO_COCO: dict[str, int] = {
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


def load_config(dataset: str) -> dict:
    path = _TOOL_DIR / "configs" / "datasets" / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with open(path) as f:
        return json.load(f)


def build_stream_index(
    config: dict,
) -> tuple[list[str], list[str | None], list[int | None]]:
    """Return (image_paths, label_paths, inline_labels) for the stream portion only.

    Mirrors CVImageDirAdapter's split arithmetic so step=0 in predictions.csv
    corresponds to index 0 of the returned lists.

    label_paths:   path to a GT file (mask PNG for segmentation, detection
                   annotation file). None when the manifest has no
                   "label_path" column or the cell is empty.
    inline_labels: integer GT label embedded directly in the manifest
                   (classification datasets — iWildCam/ImageNet — store GT
                   this way via a "label" column). None for detection/
                   segmentation datasets.
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


def load_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"predictions.csv not found in {run_dir}")
    df = pd.read_csv(path)
    if "proxy_acc" not in df.columns:
        raise ValueError(
            f"Run at {run_dir} has no 'proxy_acc' column — "
            "this appears to be a regression run. offline_eval is for CV only."
        )
    return df


def resolve_weights(config: dict, model_name: str) -> str:
    spec = config.get("models", {}).get(model_name, {})
    wp = spec.get("weights_path", "")
    if not Path(wp).is_absolute():
        wp = str(_TOOL_DIR / wp)
    return wp


def has_saved_predictions(run_path: Path, task: str) -> bool:
    """Return True if the run has a non-empty predictions/ directory."""
    pred_dir = run_path / "predictions"
    if not pred_dir.exists():
        return False
    ext = {"segmentation": "*.png", "classification": "*.txt", "detection": "*.npz"}.get(task, "")
    return ext != "" and any(pred_dir.glob(ext))


def load_class_names(config: dict) -> dict[int, str] | None:
    """Optional label -> class_name lookup for human-readable reports.

    NOT used as a ground-truth source: the numeric class index actually
    compared against predictions always comes from the manifest's own inline
    "label" column (see build_stream_index). This is purely a display-name
    lookup, checked in two places next to the dataset's manifest_csv:

    1. class_index_trimmed.json (written by scripts/trim_manifest.py's
       imagenet strategy, added 2026-08-30) — preferred when present, since
       a trim REMAPS labels to a fresh contiguous 0..N-1 range over just the
       selected classes; class_mapping.csv's original 0..999 labels no
       longer correspond to the trimmed manifest's label column at all.
    2. class_mapping.csv (data/imagenet/class_mapping.csv, added
       2026-08-30 — "easy label and id lookup") — used when there's no trim.
       Alphabetically ordered by class_name (confirmed by inspection,
       DECISIONS_PENDING.md DP24) — a DIFFERENT ordering than any pretrained
       model's canonical class order, irrelevant here since this is cosmetic
       only.

    Returns None if the dataset has neither file.
    """
    manifest_csv = config.get("manifest_csv")
    if not manifest_csv:
        return None
    manifest_path = (
        Path(manifest_csv) if Path(manifest_csv).is_absolute() else _TOOL_DIR / manifest_csv
    )

    trimmed_path = manifest_path.parent / "class_index_trimmed.json"
    if trimmed_path.exists():
        try:
            with open(trimmed_path) as f:
                index = json.load(f)
            return {int(k): v for k, v in index["index_to_class"].items()}
        except Exception:
            pass

    mapping_path = manifest_path.parent / "class_mapping.csv"
    if not mapping_path.exists():
        return None
    try:
        with open(mapping_path, newline="") as f:
            reader = csv.DictReader(f)
            return {int(row["label"]): row["class_name"] for row in reader}
    except Exception:
        return None

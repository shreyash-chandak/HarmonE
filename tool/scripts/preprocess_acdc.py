"""
scripts/preprocess_acdc.py — Build condition-ordered manifest for ACDC dataset.

Raw directory structure:
    rgb_anon/{condition}/{split}/{sequence}/{filename}.png
    gt_trainval/gt/{condition}/{split}/{sequence}/{filename}_gt_labelTrainIds.png

Conditions (in drift stream order): fog, rain, night, snow
Splits: train, val, test

Label format: Cityscapes trainIds (0-18 valid, 255 = ignore_index).
              Use _gt_labelTrainIds.png NOT _gt_labelIds.png.

Pairing rule:
    image stem  →  label stem + "_gt_labelTrainIds"
    Example:
      rgb_anon/fog/val/GOPR0475/GOPR0475_frame_000599_rgb_anon.png
      gt_trainval/gt/fog/val/GOPR0475/GOPR0475_frame_000599_gt_labelTrainIds.png

Drift stream ordering: clear (N/A in ACDC) → fog → rain → night → snow
    ACDC has no "clear" condition. If Cityscapes reference images are included
    in a combined dataset, place them first with domain="clear".
    For pure ACDC runs, stream starts with fog.

Output: data/acdc/acdc_manifest.csv with columns:
    sample_id   (string, image filename stem)
    input_path  (string, path to RGB image in rgb_anon/)
    label_path  (string, path to _gt_labelTrainIds.png mask)
    domain      (string, fog/rain/night/snow)
    split       (string, train/val/test)

Usage:
    cd tool/
    python scripts/preprocess_acdc.py \\
        --acdc-root /path/to/acdc/ \\
        --output    data/acdc/acdc_manifest.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


OUTPUT_DEFAULT = "data/acdc/acdc_manifest.csv"

CONDITION_ORDER = ["fog", "rain", "night", "snow"]


def build_manifest(acdc_root: str, output_path: str) -> pd.DataFrame:
    acdc_root = Path(acdc_root)
    rgb_root = acdc_root / "rgb_anon"
    gt_root = acdc_root / "gt_trainval" / "gt"

    if not rgb_root.exists():
        raise FileNotFoundError(
            f"rgb_anon/ not found at {rgb_root}\n"
            "Expected structure: <acdc-root>/rgb_anon/{condition}/{split}/{sequence}/*.png"
        )
    if not gt_root.exists():
        raise FileNotFoundError(
            f"gt_trainval/gt/ not found at {gt_root}\n"
            "Expected structure: <acdc-root>/gt_trainval/gt/{condition}/{split}/{sequence}/*_gt_labelTrainIds.png"
        )

    rows = []
    missing_labels = 0

    for condition in CONDITION_ORDER:
        rgb_cond = rgb_root / condition
        if not rgb_cond.exists():
            print(f"  Condition '{condition}' not found in rgb_anon/ — skipping.")
            continue

        for split in ("train", "val", "test"):
            rgb_split = rgb_cond / split
            if not rgb_split.exists():
                continue

            images = sorted(rgb_split.rglob("*.png"))
            for img_path in images:
                stem = img_path.stem  # e.g. GOPR0475_frame_000599_rgb_anon

                # Derive label filename: replace _rgb_anon suffix with _gt_labelTrainIds
                label_stem = stem.replace("_rgb_anon", "_gt_labelTrainIds")
                if "_rgb_anon" not in stem:
                    # Some ACDC versions use plain stems
                    label_stem = stem + "_gt_labelTrainIds"

                # Label is in same sequence subdirectory
                seq_dir = img_path.parent.name  # sequence folder
                label_path = gt_root / condition / split / seq_dir / f"{label_stem}.png"

                if not label_path.exists():
                    missing_labels += 1
                    label_path_str = str(label_path)  # placeholder
                else:
                    label_path_str = str(label_path)

                rows.append({
                    "sample_id": stem,
                    "input_path": str(img_path),
                    "label_path": label_path_str,
                    "domain": condition,
                    "split": split,
                })

    if missing_labels:
        print(f"  WARNING: {missing_labels} label files not found. "
              "Check that gt_trainval/ uses _gt_labelTrainIds.png suffix.")

    df = pd.DataFrame(rows)

    # Apply drift stream ordering
    cond_rank = {c: i for i, c in enumerate(CONDITION_ORDER)}
    df["_cond_rank"] = df["domain"].map(cond_rank).fillna(len(CONDITION_ORDER))
    df = df.sort_values(["split", "_cond_rank"]).drop(columns=["_cond_rank"])
    df = df.reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nManifest summary:")
    print(df.groupby(["split", "domain"]).size().to_string())
    print(f"\nSaved {len(df)} rows to {output_path}")
    print(f"ignore_index = 255 (configured in configs/datasets/acdc.json)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build condition-ordered ACDC manifest.")
    parser.add_argument("--acdc-root", required=True,
                        help="Root directory of ACDC download "
                             "(contains rgb_anon/ and gt_trainval/)")
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="Output manifest CSV")
    args = parser.parse_args()

    build_manifest(args.acdc_root, args.output)


if __name__ == "__main__":
    main()

"""
scripts/preprocess_acdc.py — Build condition+sequence-ordered manifest for ACDC.

Raw directory structure:
    rgb_anon/{condition}/{split}/{sequence}/{filename}_rgb_anon.png
    gt_trainval/gt/{condition}/{split}/{sequence}/{filename}_gt_labelTrainIds.png

Conditions (drift stream order): fog → rain → night → snow
Splits processed: train, val only.
    test has no GT labels — excluded.
    trainref / valref / testref are reference counterparts — excluded.

Key design decision (GoPro sequence reconstruction):
    The ACDC train/val/test split does NOT follow temporal frame order within a
    GoPro recording. A single GoPro (e.g. GOPR0475) can have frames distributed
    across train/ AND val/ directories. Simply iterating splits in order would
    interleave unrelated sequences and destroy temporal continuity.

    This script therefore:
      1. Collects frames from train + val for every GoPro sequence.
      2. Merges them, sorted by the frame number encoded in the filename.
      3. Emits one manifest row per frame in natural sequence order.
      4. Adds 'sequence' and 'frame_number' columns to the manifest.

    The 'split' column records the frame's ORIGINAL split (train or val), which
    is needed to locate its GT label file in gt_trainval/gt/.

Label format: Cityscapes trainIds (0-18 valid, 255 = ignore_index).
              Use _gt_labelTrainIds.png NOT _gt_labelIds.png.

Pairing rule:
    GOPR0475_frame_000599_rgb_anon.png
    → gt_trainval/gt/fog/train/GOPR0475/GOPR0475_frame_000599_gt_labelTrainIds.png

Output: data/acdc/acdc_manifest.csv with columns:
    sample_id    (string, image filename stem)
    input_path   (string, absolute path to RGB image)
    label_path   (string, path to _gt_labelTrainIds.png mask)
    domain       (string, fog/rain/night/snow)
    sequence     (string, GoPro folder name, e.g. GOPR0475)
    frame_number (int, zero-padded frame index extracted from filename)
    split        (string, original split of this frame: train or val)

Usage:
    cd tool/
    python scripts/preprocess_acdc.py \\
        --acdc-root data/acdc \\
        --output    data/acdc/acdc_manifest.csv
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd


OUTPUT_DEFAULT = "data/acdc/acdc_manifest.csv"

CONDITION_ORDER = ["fog", "rain", "night", "snow"]
# Only train + val have GT labels. test has no labels; ref splits are excluded.
VALID_SPLITS = ("train", "val")


def _extract_frame_number(stem: str) -> int:
    """Extract the numeric frame index from an ACDC filename stem.

    ACDC format: GOPR0475_frame_000599_rgb_anon  →  599
    Falls back to 0 if the pattern is absent.
    """
    m = re.search(r"_frame_(\d+)", stem)
    return int(m.group(1)) if m else 0


def build_manifest(acdc_root: str, output_path: str) -> pd.DataFrame:
    acdc_root = Path(acdc_root)
    rgb_root = acdc_root / "rgb_anon"
    gt_root = acdc_root / "gt_trainval" / "gt"

    if not rgb_root.exists():
        raise FileNotFoundError(
            f"rgb_anon/ not found at {rgb_root}\n"
            "Expected: <acdc-root>/rgb_anon/{condition}/{split}/{sequence}/*.png"
        )
    if not gt_root.exists():
        raise FileNotFoundError(
            f"gt_trainval/gt/ not found at {gt_root}\n"
            "Expected: <acdc-root>/gt_trainval/gt/{condition}/{split}/{sequence}/*_gt_labelTrainIds.png"
        )

    rows = []
    missing_labels = 0

    for condition in CONDITION_ORDER:
        rgb_cond = rgb_root / condition
        if not rgb_cond.exists():
            print(f"  Condition '{condition}' not found in rgb_anon/ — skipping.")
            continue

        # ── Reconstruct GoPro sequences across train + val ──────────────────
        # seq_map: sequence_name → list of frame dicts
        seq_map: dict[str, list] = {}

        for split in VALID_SPLITS:
            rgb_split = rgb_cond / split
            if not rgb_split.exists():
                continue

            for img_path in rgb_split.rglob("*.png"):
                stem = img_path.stem
                seq_name = img_path.parent.name  # GoPro folder (e.g. GOPR0475)
                frame_num = _extract_frame_number(stem)

                # Derive the label path (label lives under its own split dir)
                label_stem = stem.replace("_rgb_anon", "_gt_labelTrainIds")
                if "_rgb_anon" not in stem:
                    label_stem = stem + "_gt_labelTrainIds"
                label_path = gt_root / condition / split / seq_name / f"{label_stem}.png"

                if not label_path.exists():
                    missing_labels += 1

                seq_map.setdefault(seq_name, []).append({
                    "sample_id":    stem,
                    "input_path":   img_path.as_posix(),
                    "label_path":   label_path.as_posix(),
                    "domain":       condition,
                    "sequence":     seq_name,
                    "frame_number": frame_num,
                    "split":        split,
                })

        # Emit sequences in alphabetical order; frames within each sequence by
        # frame_number → preserves the original GoPro recording continuity.
        for seq_name in sorted(seq_map.keys()):
            frames = sorted(seq_map[seq_name], key=lambda f: f["frame_number"])
            rows.extend(frames)

    if missing_labels:
        print(
            f"  WARNING: {missing_labels} label files not found. "
            "Verify gt_trainval/ uses _gt_labelTrainIds.png suffix and "
            "sequence folder names match between rgb_anon/ and gt_trainval/gt/."
        )

    df = pd.DataFrame(rows)

    # Primary sort: condition (drift order) → sequence → frame_number.
    # The 'split' column is metadata only — do NOT sort by it here.
    cond_rank = {c: i for i, c in enumerate(CONDITION_ORDER)}
    df["_cond_rank"] = df["domain"].map(cond_rank).fillna(len(CONDITION_ORDER))
    df = df.sort_values(["_cond_rank", "sequence", "frame_number"]).drop(columns=["_cond_rank"])
    df = df.reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nManifest summary (domain × original-split):")
    print(df.groupby(["domain", "split"]).size().to_string())
    print(f"\nTotal frames : {len(df)}")
    print(f"Sequences    : {df['sequence'].nunique()}")
    print(f"Saved to     : {output_path}")
    print("ignore_index = 255 (configs/datasets/acdc.json)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sequence-reconstructed ACDC manifest."
    )
    parser.add_argument(
        "--acdc-root", required=True,
        help="Root directory of ACDC download (contains rgb_anon/ and gt_trainval/)"
    )
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="Output manifest CSV")
    args = parser.parse_args()

    build_manifest(args.acdc_root, args.output)


if __name__ == "__main__":
    main()

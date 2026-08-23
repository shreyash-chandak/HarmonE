"""
scripts/preprocess_iwildcam.py — Build location-ordered manifest for iWildCam (WILDS v2.0).

Raw input (WILDS v2.0 directory):
    metadata.csv      — columns: image_id, y (label), split, location, ...
    categories.csv    — columns: class_id, species_name (optional)
    images/           — flat directory or per-location subdirectory of .jpg

Split encoding used by WILDS:
    0 = train, 1 = id_val, 2 = ood_val, 3 = id_test, 4 = ood_test

The primary inference stream for HarmonE is the OOD test split (split=4):
    unseen camera trap locations → geographic domain shift.

Domain mapping: each unique `location` value is a domain. Samples are
ordered by location to create a controlled geographic drift sequence.
Within each location, original row order is preserved.

Output: data/iwildcam/iwildcam_manifest.csv with columns:
    sample_id   (string, image_id from metadata)
    input_path  (string, absolute path to image file)
    label       (int, species class index 0-181)
    domain      (string, location id)
    split       (string, train/id_val/ood_val/id_test/ood_test)

Usage:
    cd tool/
    python scripts/preprocess_iwildcam.py \\
        --wilds-root /path/to/iwildcam_v2.0/ \\
        --output     data/iwildcam/iwildcam_manifest.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


OUTPUT_DEFAULT = "data/iwildcam/iwildcam_manifest.csv"

SPLIT_MAP = {
    0: "train",
    1: "id_val",
    2: "ood_val",
    3: "id_test",
    4: "ood_test",
}

# WILDS v2.0 exports short string labels. Normalize to the canonical long names
# so the manifest is consistent regardless of which WILDS version supplied the data.
#   "test"   → "ood_test"   (42k geographic OOD images — primary inference stream)
#   "val"    → "ood_val"
#   "id_test"/"id_val"/"train" pass through unchanged.
_NORMALIZE_SPLIT = {
    "train":   "train",
    "val":     "ood_val",
    "test":    "ood_test",
    "id_val":  "id_val",
    "id_test": "id_test",
    # long-form names (already canonical) pass through unchanged
    "ood_val":  "ood_val",
    "ood_test": "ood_test",
}


def _find_image(images_root: Path, image_id: str) -> str | None:
    """Search for image file by id; handles flat and per-location structures."""
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = images_root / f"{image_id}{ext}"
        if candidate.exists():
            return str(candidate)
    # WILDS v2.0 may have per-location subdirectories
    for subdir in images_root.iterdir():
        if subdir.is_dir():
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = subdir / f"{image_id}{ext}"
                if candidate.exists():
                    return str(candidate)
    return None


def build_manifest(wilds_root: str, output_path: str) -> pd.DataFrame:
    wilds_root = Path(wilds_root)
    metadata_path = wilds_root / "metadata.csv"

    # WILDS v2.0 iWildCam ships images under 'train/' not 'images/'
    for _img_dir in ("images", "train"):
        candidate = wilds_root / _img_dir
        if candidate.exists() and candidate.is_dir():
            images_root = candidate
            break
    else:
        raise FileNotFoundError(
            f"Image directory not found under {wilds_root}. "
            "Expected 'images/' or 'train/' subdirectory."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")

    print(f"Loading {metadata_path} ...")
    meta = pd.read_csv(metadata_path)
    print(f"  Columns: {list(meta.columns)}")
    print(f"  Rows: {len(meta)}")

    # Normalise column names (WILDS v2.0 uses lowercase)
    meta.columns = meta.columns.str.strip().str.lower()

    # Resolve label column (may be 'y' or 'label')
    label_col = "y" if "y" in meta.columns else "label"
    if label_col not in meta.columns:
        raise ValueError(
            f"Expected label column 'y' or 'label' not found. Columns: {list(meta.columns)}"
        )

    # Resolve split column
    split_col = "split" if "split" in meta.columns else None
    if split_col is None:
        raise ValueError(f"'split' column not found. Columns: {list(meta.columns)}")

    # Resolve location/domain column
    domain_col = next(
        (c for c in meta.columns if c in ("location", "location_remapped", "site")),
        None,
    )
    if domain_col is None:
        raise ValueError(
            f"Domain column (location/location_remapped/site) not found. "
            f"Columns: {list(meta.columns)}"
        )

    # Resolve image_id column
    id_col = next(
        (c for c in meta.columns if c in ("image_id", "id", "filename")),
        None,
    )
    if id_col is None:
        raise ValueError(
            f"Image ID column not found. Columns: {list(meta.columns)}"
        )

    rows = []
    missing = 0
    for _, row in meta.iterrows():
        image_id = str(row[id_col])
        # image_id in WILDS is a UUID; filename column (if present) may include .jpg
        # strip extension so _find_image() can append it
        if id_col in ("filename",) and image_id.endswith((".jpg", ".jpeg", ".png")):
            image_id = image_id.rsplit(".", 1)[0]

        label = int(row[label_col])

        raw_split = row[split_col]
        # Handle both integer splits (WILDS ≤1.2) and string splits (WILDS ≥2.0).
        # Normalize v2.0 short names ("test"→"ood_test", "val"→"ood_val").
        if isinstance(raw_split, str):
            split_str = _NORMALIZE_SPLIT.get(raw_split, raw_split)
        else:
            split_str = SPLIT_MAP.get(int(raw_split), str(int(raw_split)))

        domain = str(row[domain_col])

        img_path = _find_image(images_root, image_id)
        if img_path is None:
            missing += 1
            img_path = str(images_root / f"{image_id}.jpg")  # placeholder

        rows.append({
            "sample_id": image_id,
            "input_path": img_path,
            "label": label,
            "domain": domain,
            "split": split_str,
        })

    if missing:
        print(f"  WARNING: {missing} image files not found on disk (paths are placeholders).")

    df = pd.DataFrame(rows)

    # Sort by location to create geographic drift sequence
    df = df.sort_values(["split", "domain"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nManifest summary (by split):")
    print(df.groupby("split").size().to_string())
    print(f"\nSplit=ood_test has {len(df[df['split'] == 'ood_test'])} samples "
          f"across {df[df['split'] == 'ood_test']['domain'].nunique()} locations.")
    print(f"\nSaved {len(df)} rows to {output_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build location-ordered iWildCam manifest.")
    parser.add_argument("--wilds-root", required=True,
                        help="Root directory of iWildCam WILDS v2.0 download "
                             "(contains metadata.csv and images/)")
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="Output manifest CSV")
    args = parser.parse_args()

    build_manifest(args.wilds_root, args.output)


if __name__ == "__main__":
    main()

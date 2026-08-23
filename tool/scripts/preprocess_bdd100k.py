"""
scripts/preprocess_bdd100k.py — Build drift-ordered manifest for BDD100K.

Raw input (per-image JSON format — the format used by BDD100K tool releases):
    images/100k/{train,val,test}/*.jpg
    labels/100k/{train,val,test}/{stem}.json

Each per-image JSON file has:
    "name": image stem without .jpg  (e.g. "b1c66a42-6f7d68ca")
    "attributes": {
        "weather": "clear" | "overcast" | "partly cloudy" | "foggy" | "rainy" | "snowy"
        "timeofday": "daytime" | "dawn/dusk" | "night" | "undefined"
    }
    "frames": [...] (bounding box annotations per frame)

Also supports the older two-file bundle format:
    labels/bdd100k_labels_images_train.json   (list of all train entries)
    labels/bdd100k_labels_images_val.json     (list of all val entries)
The script detects which format is present automatically.

Domain mapping:
    clear_day = daytime + clear
    overcast  = daytime + (overcast | partly cloudy)
    foggy     = foggy (daytime)
    dusk      = dawn/dusk (any weather)
    night     = night (any weather)
    rain      = rainy (any timeofday)
    snow      = snowy (any timeofday)

Drift stream order: clear_day → overcast → foggy → dusk → night → rain

Output: data/bdd100k/bdd100k_manifest.csv with columns:
    sample_id   (string, image stem)
    input_path  (string, path to image)
    label_path  (string, YOLO .txt path if --yolo-labels-dir given, else per-image JSON path)
    domain      (string, one of the domains above)
    split       (string, train/val/test)

Usage:
    cd tool/
    python scripts/preprocess_bdd100k.py \\
        --bdd-root data/bdd100k \\
        --output   data/bdd100k/bdd100k_manifest.csv \\
        [--yolo-labels-dir data/bdd100k/labels_yolo/]
        [--no-yolo]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


OUTPUT_DEFAULT = "data/bdd100k/bdd100k_manifest.csv"

DOMAIN_ORDER = ["clear_day", "overcast", "foggy", "dusk", "night", "rain", "snow"]


def _assign_domain(attributes: dict) -> str:
    weather = attributes.get("weather", "").lower()
    timeofday = attributes.get("timeofday", "").lower()

    if weather in ("rainy",):
        return "rain"
    if weather in ("snowy",):
        return "snow"
    if timeofday in ("night",):
        return "night"
    if timeofday in ("dawn/dusk",):
        return "dusk"
    if weather in ("foggy",):
        return "foggy"
    if weather in ("overcast", "partly cloudy"):
        return "overcast"
    if weather in ("clear",) and timeofday in ("daytime",):
        return "clear_day"
    # Fallback for "undefined" or unknown combinations
    return "overcast"


def _load_split_annotations(labels_path: Path, split: str) -> dict[str, dict]:
    """Load annotations for one split. Returns {image_stem: entry_dict}.

    Searches (in order):
      1. labels/100k/{split}/*.json  — per-image JSON files (BDD100K tool format)
      2. labels/*.json               — bundled JSON list (old two-file release)
      3. labels/100k/*.json          — bundled JSON in 100k/ subdirectory
    """
    # Per-image format: labels/100k/{split}/ — one file per image
    per_image_dir = labels_path / "100k" / split
    if per_image_dir.exists():
        json_files = list(per_image_dir.glob("*.json"))
        if json_files:
            annotations: dict[str, dict] = {}
            print(f"  [{split}] Loading {len(json_files)} per-image JSONs ...", flush=True)
            for jf in json_files:
                with open(jf) as f:
                    data = json.load(f)
                # 'name' is the stem without extension (e.g. "b1c66a42-6f7d68ca")
                stem = data.get("name", jf.stem)
                annotations[stem] = data
            return annotations

    # Bundle format: one large JSON list covering a whole split
    for bundle_dir in [labels_path, labels_path / "100k"]:
        if not bundle_dir.exists():
            continue
        # Look for files whose name contains the split name
        for jf in bundle_dir.glob("*.json"):
            if split in jf.name:
                with open(jf) as f:
                    entries = json.load(f)
                if isinstance(entries, list):
                    print(f"  [{split}] Loaded {len(entries)} annotations from {jf.name}")
                    return {e.get("name", ""): e for e in entries if e.get("name")}

    print(f"  [{split}] WARNING: No annotation JSON files found — domain will default to 'overcast'.")
    return {}


def build_manifest(
    bdd_root: str,
    output_path: str,
    yolo_labels_dir: str | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> pd.DataFrame:
    bdd_root = Path(bdd_root)
    images_root = bdd_root / "images" / "100k"
    labels_root = bdd_root / "labels"

    if not images_root.exists():
        raise FileNotFoundError(
            f"Images directory not found: {images_root}\n"
            "Expected structure: <bdd-root>/images/100k/{{train,val,test}}/"
        )

    rows = []
    for split in splits:
        split_dir = images_root / split
        if not split_dir.exists():
            print(f"  Skipping {split} (directory not found).")
            continue

        # Load annotations for this split only (avoids loading 70k train files unnecessarily)
        print(f"Loading annotations for split='{split}' ...")
        split_annotations = _load_split_annotations(labels_root, split)

        images = sorted(split_dir.glob("*.jpg"))
        print(f"  {split}: {len(images)} images, {len(split_annotations)} annotations loaded")
        for img_path in images:
            stem = img_path.stem

            # Annotations keyed by stem (name without .jpg)
            annotation = split_annotations.get(stem, {})
            attributes = annotation.get("attributes", {})
            domain = _assign_domain(attributes)

            # Determine label path (always forward slashes for cross-platform compat)
            if yolo_labels_dir:
                label_path = (Path(yolo_labels_dir) / split / f"{stem}.txt").as_posix()
            else:
                # Point to the per-image JSON file (actual annotation source)
                label_path = (labels_root / "100k" / split / f"{stem}.json").as_posix()

            rows.append({
                "sample_id": stem,
                "input_path": img_path.as_posix(),
                "label_path": label_path,
                "domain": domain,
                "split": split,
            })

    df = pd.DataFrame(rows)

    # Sort inference stream: for train/val, sort by domain then by original order
    domain_rank = {d: i for i, d in enumerate(DOMAIN_ORDER)}
    df["_domain_rank"] = df["domain"].map(domain_rank).fillna(len(DOMAIN_ORDER))
    df = df.sort_values(["split", "_domain_rank"]).drop(columns=["_domain_rank"])
    df = df.reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nManifest summary:")
    print(df.groupby(["split", "domain"]).size().to_string())
    print(f"\nSaved {len(df)} rows to {output_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build drift-ordered BDD100K manifest.")
    parser.add_argument("--bdd-root", required=True,
                        help="Root directory of BDD100K download "
                             "(contains images/ and labels/)")
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="Output manifest CSV")
    parser.add_argument("--yolo-labels-dir", default=None,
                        help="Directory for per-image YOLO .txt label files. "
                             "If omitted, label_path points to the source JSON.")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Skip YOLO label conversion (manifest only)")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Splits to include (default: train val test). "
             "Use '--splits val' to skip train (~5 min) when you only need val for experiments.",
    )
    args = parser.parse_args()

    if args.yolo_labels_dir and not args.no_yolo:
        print("Note: YOLO label conversion requires bdd_to_yolo_labels.py.")
        print("Run: python managed_system_cv/utility/bdd_to_yolo_labels.py --help")

    build_manifest(args.bdd_root, args.output, args.yolo_labels_dir, tuple(args.splits))


if __name__ == "__main__":
    main()

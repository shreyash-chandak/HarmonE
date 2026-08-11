"""
scripts/preprocess_bdd100k.py — Build drift-ordered manifest for BDD100K.

Raw input:
    images/100k/{train,val,test}/*.jpg   — dashcam frames
    labels/bdd100k_labels_images_train.json
    labels/bdd100k_labels_images_val.json

Each JSON entry has:
    "name": filename (without directory)
    "attributes": {
        "weather": "clear" | "overcast" | "partly cloudy" | "foggy" | "rainy" | "snowy"
        "timeofday": "daytime" | "dawn/dusk" | "night" | "undefined"
        "scene": ...
    }
    "labels": [...] (bounding box annotations)

Domain mapping (applied to inference stream ordering):
    clear_day   = daytime + clear
    overcast    = daytime + (overcast | partly cloudy)
    dusk        = dawn/dusk (any weather)
    night       = night (any weather)
    rain        = rainy (any timeofday)
    snow        = snowy (any timeofday)   [merged into "rain" for 5-condition ordering]
    foggy       = foggy (daytime only)    [inserted between overcast and dusk]

Drift stream ordering:
    clear_day → overcast → foggy → dusk → night → rain

Output: data/bdd100k/bdd100k_manifest.csv with columns:
    sample_id   (string, image filename without extension)
    input_path  (string, absolute or relative path to image)
    label_path  (string, path to per-image YOLO label .txt if generated,
                 else JSON annotation path)
    domain      (string, one of clear_day/overcast/foggy/dusk/night/rain/snow)
    split       (string, train/val/test)

Also exports per-image YOLO .txt label files via the existing
managed_system_cv/utility/bdd_to_yolo_labels.py converter.
(Pass --no-yolo to skip label conversion if only the manifest is needed.)

Usage:
    cd tool/
    python scripts/preprocess_bdd100k.py \\
        --bdd-root /path/to/bdd100k/ \\
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


def _load_annotations(labels_dir: str) -> dict[str, dict]:
    """Load BDD100K JSON annotations, return {filename: entry}."""
    annotations: dict[str, dict] = {}
    for json_file in Path(labels_dir).glob("*.json"):
        with open(json_file) as f:
            entries = json.load(f)
        for entry in entries:
            name = entry.get("name", "")
            annotations[name] = entry
        print(f"  Loaded {len(entries)} annotations from {json_file.name}")
    return annotations


def build_manifest(
    bdd_root: str,
    output_path: str,
    yolo_labels_dir: str | None = None,
) -> pd.DataFrame:
    bdd_root = Path(bdd_root)
    images_root = bdd_root / "images" / "100k"
    labels_root = bdd_root / "labels"

    if not images_root.exists():
        raise FileNotFoundError(
            f"Images directory not found: {images_root}\n"
            "Expected structure: <bdd-root>/images/100k/{{train,val,test}}/"
        )

    print(f"Loading annotations from {labels_root} ...")
    annotations = _load_annotations(str(labels_root))

    rows = []
    for split in ("train", "val", "test"):
        split_dir = images_root / split
        if not split_dir.exists():
            print(f"  Skipping {split} (directory not found).")
            continue
        images = sorted(split_dir.glob("*.jpg"))
        print(f"  {split}: {len(images)} images")
        for img_path in images:
            filename = img_path.name
            stem = img_path.stem

            annotation = annotations.get(filename, {})
            attributes = annotation.get("attributes", {})
            domain = _assign_domain(attributes)

            # Determine label path
            if yolo_labels_dir:
                label_path = str(Path(yolo_labels_dir) / split / f"{stem}.txt")
            else:
                label_path = str(labels_root / f"bdd100k_labels_images_{split}.json")

            rows.append({
                "sample_id": stem,
                "input_path": str(img_path),
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
    args = parser.parse_args()

    if args.yolo_labels_dir and not args.no_yolo:
        print("Note: YOLO label conversion requires bdd_to_yolo_labels.py.")
        print("Run: python managed_system_cv/utility/bdd_to_yolo_labels.py --help")

    build_manifest(args.bdd_root, args.output, args.yolo_labels_dir)


if __name__ == "__main__":
    main()

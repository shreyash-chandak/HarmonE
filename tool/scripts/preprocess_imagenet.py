"""
scripts/preprocess_imagenet.py — Build the base manifest for ImageNet-1k (C4).

Raw input, per the layout actually present under data/imagenet/ (2026-08-30):
    data/imagenet/<class_name>/<NNN>.jpg   — 1000 class folders, human-readable
                                              folder names (e.g. "abacus"), no
                                              train/val split on disk, no
                                              weather/location metadata.
    data/imagenet/class_mapping.csv        — "label,class_name" — 1000 rows,
                                              the authoritative label index
                                              for every class.
    data/imagenet/gt.csv                   — "image_id,label" — one row per
                                              image, the authoritative ground
                                              truth. image_id is
                                              "{class_idx:03d}-{img_idx:03d}",
                                              where class_idx is the class's
                                              row position in class_mapping.csv
                                              (== its alphabetical folder rank)
                                              and img_idx is that image's
                                              position in the class folder's
                                              own sorted filename order.
                                              Verified 2026-08-30 across all
                                              1000 classes / 539 826 images:
                                              zero mismatches between this
                                              scheme and gt.csv's actual rows.

Unlike BDD100K/iWildCam/ACDC, ImageNet-1k has no natural drift axis on disk —
there is no weather attribute, no camera-trap location, no adverse-condition
folder. This script therefore builds a single-domain ("clean") base manifest
only. A genuinely drift-ordered variant (corruption-based, following the
Hendrycks & Dietterich ImageNet-C recipe that TENT is benchmarked on) is built
separately by managed_system_cv/utility/drift/induce_imagenet_c.py — see
context/idea.md for the full rationale connecting this to TENT and Augur.

Class-index correctness — CHANGED 2026-08-30 (see DECISIONS_PENDING.md DP24):
    Earlier revisions of this script resolved ground truth by importing
    torchvision and matching folder names against
    `<Weights>.IMAGENET1K_V2.meta["categories"]` (torchvision's canonical
    pretrained-weight class order), with an --allow-alphabetical-fallback
    escape hatch. That entire mechanism is REMOVED: it required torchvision
    to be importable at preprocessing time (broken in this dev environment),
    and it is no longer the authoritative source now that class_mapping.csv/
    gt.csv exist — those files ARE this dataset's ground truth, full stop,
    and (confirmed by inspection) use ALPHABETICAL class order, not
    torchvision's canonical order. Concretely: label 0 is "abacus" here, not
    torchvision's canonical index-0 class ("tench").

    Practical consequence: a torchvision pretrained classification head
    cannot be used as-is against this manifest's label column anymore (their
    class orders don't match) — export_pretrained_weights() is REMOVED for
    the same reason. Bootstrapping model weights for this dataset is instead
    handled by experiments/run_experiment.py::_train_cv_models_if_missing(),
    which fine-tunes a freshly-sized, freshly-trained head on this manifest's
    own labels (same as any other CV dataset with a class count that doesn't
    match an off-the-shelf pretrained head) — see configs/datasets/
    imagenet.json's _comment, updated accordingly.

Output:
    data/imagenet/imagenet_manifest.csv   — columns: sample_id, input_path,
        image_path (duplicate of input_path — see note below), label (int,
        from gt.csv), class_name, domain ("clean"), split (train/val/stream).
    data/imagenet/class_index.json        — index<->class_name mapping + source.

Note on the duplicate `image_path` column: adapters/cv_imagedir.py reads
`input_path` (falling back to `image_path`), which is the schema every other
CV preprocessing script in this repo writes. core/dataset_validator.py's CV
path check, however, looks for a literal `image_path` column — a pre-existing
mismatch that would also affect bdd100k/iwildcam/acdc manifests if they were
re-validated (see context/DECISIONS_PENDING.md DP15). Writing both columns
here keeps `validate_dataset.py` accurate for this dataset without touching
the shared validator or any other dataset's files.

Usage:
    cd tool/
    python3 scripts/preprocess_imagenet.py --imagenet-root data/imagenet
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = "data/imagenet"
DEFAULT_OUTPUT = "data/imagenet/imagenet_manifest.csv"
DEFAULT_CLASS_INDEX = "data/imagenet/class_index.json"
DEFAULT_CLASS_MAPPING = "data/imagenet/class_mapping.csv"
DEFAULT_GT = "data/imagenet/gt.csv"


def _load_class_mapping(path: Path) -> dict[str, int]:
    """Read class_mapping.csv ("label,class_name") -> {class_name: label}."""
    if not path.exists():
        raise FileNotFoundError(
            f"class_mapping.csv not found at {path}. This script requires it "
            "(label <-> class_name lookup) — see this module's docstring."
        )
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        mapping = {row["class_name"]: int(row["label"]) for row in reader}
    if not mapping:
        raise ValueError(f"{path} is empty.")
    return mapping


def _load_ground_truth(path: Path) -> dict[str, int]:
    """Read gt.csv ("image_id,label") -> {image_id: label}."""
    if not path.exists():
        raise FileNotFoundError(
            f"gt.csv not found at {path}. This script requires it "
            "(per-image ground truth) — see this module's docstring."
        )
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        gt = {row["image_id"]: int(row["label"]) for row in reader}
    if not gt:
        raise ValueError(f"{path} is empty.")
    return gt


def build_manifest(
    imagenet_root: str,
    output_path: str,
    class_index_path: str,
    class_mapping_path: str,
    gt_path: str,
    train_frac: float,
    val_frac: float,
) -> pd.DataFrame:
    root = Path(imagenet_root)
    if not root.exists():
        raise FileNotFoundError(f"ImageNet root not found: {root}")

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories found under {root}")
    print(f"Found {len(class_dirs)} class directories under {root}")

    class_to_index = _load_class_mapping(Path(class_mapping_path))
    ground_truth = _load_ground_truth(Path(gt_path))
    print(f"Loaded {len(class_to_index)} classes from {class_mapping_path}, "
          f"{len(ground_truth)} ground-truth rows from {gt_path}")

    missing_classes = [d.name for d in class_dirs if d.name not in class_to_index]
    if missing_classes:
        raise RuntimeError(
            f"{len(missing_classes)} class folder(s) have no entry in "
            f"{class_mapping_path}: {missing_classes[:10]}"
            + (" ..." if len(missing_classes) > 10 else "")
            + ". Every folder under --imagenet-root must have a matching "
            "class_name row in class_mapping.csv."
        )

    rows_train, rows_val, rows_stream = [], [], []
    unmatched_images = 0
    for class_idx, cdir in enumerate(class_dirs):
        class_name = cdir.name
        label = class_to_index[class_name]
        images = sorted(
            p for p in cdir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        n = len(images)
        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)

        for img_idx, img_path in enumerate(images):
            image_id = f"{class_idx:03d}-{img_idx:03d}"
            gt_label = ground_truth.get(image_id)
            if gt_label is None:
                unmatched_images += 1
                continue
            if gt_label != label:
                raise RuntimeError(
                    f"Label mismatch for {image_id} ({img_path}): "
                    f"class_mapping.csv says {label} ('{class_name}'), "
                    f"gt.csv says {gt_label}. Ground-truth sources disagree — "
                    "refusing to silently pick one (invariant I5: no "
                    "fabricated/ambiguous ground truth)."
                )
            row = {
                "sample_id": f"{class_name}/{img_path.stem}",
                "input_path": str(img_path),
                "image_path": str(img_path),  # validator compat — see module docstring
                "label": label,
                "class_name": class_name,
                "domain": "clean",
            }
            if img_idx < train_end:
                row["split"] = "train"
                rows_train.append(row)
            elif img_idx < val_end:
                row["split"] = "val"
                rows_val.append(row)
            else:
                row["split"] = "stream"
                rows_stream.append(row)

    if unmatched_images:
        raise RuntimeError(
            f"{unmatched_images} image(s) on disk have no matching image_id in "
            f"{gt_path}. Ground truth must cover every image under "
            "--imagenet-root — see this module's docstring for the expected "
            "image_id scheme."
        )

    # Row order matters: adapters/cv_imagedir.py slices train_frac/val_frac
    # POSITIONALLY over the whole manifest, not by the `split` column. All
    # train rows must precede all val rows, which must precede all stream rows.
    df = pd.DataFrame(rows_train + rows_val + rows_stream)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    os.makedirs(os.path.dirname(class_index_path) or ".", exist_ok=True)
    with open(class_index_path, "w") as f:
        json.dump(
            {
                "index_to_class": {str(v): k for k, v in class_to_index.items()},
                "class_to_index": class_to_index,
                "source": "provided_lookup_alphabetical",
                "n_classes": len(class_to_index),
            },
            f,
            indent=2,
        )

    print(f"\nManifest summary (by split):")
    print(df.groupby("split").size().to_string())
    print(f"\nSaved {len(df)} rows to {output_path}")
    print(f"Saved class index ({len(class_to_index)} classes) to {class_index_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the base (single-domain) ImageNet-1k manifest for HarmonE."
    )
    parser.add_argument("--imagenet-root", default=DEFAULT_ROOT,
                        help="Root directory containing one subfolder per class.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output manifest CSV.")
    parser.add_argument("--class-index-output", default=DEFAULT_CLASS_INDEX,
                        help="Output class_index.json path.")
    parser.add_argument("--class-mapping", default=DEFAULT_CLASS_MAPPING,
                        help="label,class_name lookup CSV (authoritative).")
    parser.add_argument("--gt", default=DEFAULT_GT,
                        help="image_id,label ground-truth CSV (authoritative).")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    build_manifest(
        args.imagenet_root, args.output, args.class_index_output,
        args.class_mapping, args.gt, args.train_frac, args.val_frac,
    )


if __name__ == "__main__":
    main()

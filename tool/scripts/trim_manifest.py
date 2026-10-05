"""
scripts/trim_manifest.py — Trim a CV dataset manifest to experiment-appropriate size.

The full preprocess scripts produce manifests covering all available images
(10k val for BDD100K, 42k ood_test for iWildCam, 539 826 for ImageNet).
Running experiments on the full set would take many hours per run. This
script trims the manifest CSV to a target size while preserving each
dataset's meaningful structure.

    BDD100K    → sample ~N images per domain from the val split
                 (default: 500/domain × 6 domains = ~3 000)
    iWildCam   → sample ~N ood_test images maintaining location grouping
                 (default: 3 000, distributed proportionally across locations)
    ACDC       → no trimming; total train+val is already ~2 006 frames.
    ImageNet   → sample N classes (default 100) and M images per class
                 (default 100) — flat "N images across 1000 classes" would
                 leave too few images per class for real classification
                 signal, so ImageNet trims on class COUNT, not just a total
                 image target (see --n-classes/--images-per-class below).
                 Deterministic (seed), train/val/stream splits are re-derived
                 per class on the trimmed pool (using the dataset config's
                 own train_frac/val_frac) so every trimmed class appears in
                 all three splits — not inherited from the full manifest's
                 already-fixed split assignment, which would distort each
                 class's train fraction after random subsampling. Labels are
                 remapped to a fresh contiguous 0..N-1 range over just the
                 selected classes (the original manifest's labels are sparse
                 over 0..999 after subsampling — a classifier's head needs
                 contiguous, zero-indexed labels matching its output width);
                 a companion class_index CSV is written recording the new
                 label -> original label + class_name mapping.

Image files are NOT deleted. The trimmed manifest is written alongside the
original (original is kept as <name>_full.csv).

Usage:
    cd tool/

    # BDD100K — trim val to ~500 images per domain:
    python3 scripts/trim_manifest.py \\
        --dataset bdd100k \\
        --manifest data/bdd100k/bdd100k_manifest.csv \\
        --target   3000

    # iWildCam — trim ood_test to ~3 000:
    python3 scripts/trim_manifest.py \\
        --dataset iwildcam \\
        --manifest data/iwildcam/iwildcam_manifest.csv \\
        --target   3000

    # ImageNet — trim to 100 classes x 100 images/class (seed=42):
    python3 scripts/trim_manifest.py \\
        --dataset imagenet \\
        --manifest data/imagenet/imagenet_manifest.csv \\
        --n-classes 100 --images-per-class 100 --seed 42

    # Check size without writing:
    python3 scripts/trim_manifest.py --dataset bdd100k --manifest ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ── Dataset-specific trim strategies ─────────────────────────────────────────

def _trim_bdd100k(df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """
    Select up to target images from the val split, balanced across domains.
    Domain ordering is preserved in the output.
    """
    val = df[df["split"] == "val"].copy()
    if len(val) == 0:
        raise ValueError("No rows with split='val' found. Run preprocess_bdd100k.py first.")

    domains = val["domain"].unique().tolist()
    n_per_domain = max(1, target // len(domains))

    kept = []
    for domain in domains:
        chunk = val[val["domain"] == domain]
        n = min(len(chunk), n_per_domain)
        kept.append(chunk.sample(n=n, random_state=seed))

    result = pd.concat(kept, ignore_index=True)

    # Restore drift order: clear_day → overcast → foggy → dusk → night → rain → snow
    domain_rank = {
        "clear_day": 0, "overcast": 1, "foggy": 2,
        "dusk": 3, "night": 4, "rain": 5, "snow": 6,
    }
    result["_rank"] = result["domain"].map(domain_rank).fillna(99)
    result = result.sort_values("_rank").drop(columns=["_rank"]).reset_index(drop=True)

    print(f"BDD100K val after trim ({len(result)} / {len(val)}):")
    print(result.groupby("domain").size().to_string())
    return result


def _trim_iwildcam(df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """
    Select up to target images from the ood_test split, distributed
    proportionally across locations (geographic drift axis).
    Within each location, original ordering is preserved.
    """
    ood = df[df["split"] == "ood_test"].copy()
    if len(ood) == 0:
        raise ValueError(
            "No rows with split='ood_test' found. "
            "Run preprocess_iwildcam.py (which normalizes 'test' → 'ood_test')."
        )

    locations = ood["domain"].unique().tolist()
    n_locs = len(locations)
    n_per_loc = max(1, target // n_locs)
    remainder = target - n_per_loc * n_locs

    kept = []
    for i, loc in enumerate(locations):
        chunk = ood[ood["domain"] == loc]
        # Give one extra image to the first `remainder` locations
        n = min(len(chunk), n_per_loc + (1 if i < remainder else 0))
        kept.append(chunk.head(n))  # head() preserves location ordering

    result = pd.concat(kept, ignore_index=True)

    print(f"iWildCam ood_test after trim ({len(result)} / {len(ood)}):")
    print(f"  Locations kept: {result['domain'].nunique()} / {n_locs}")
    print(f"  Images/location (approx): {n_per_loc}")
    return result


def _trim_imagenet(
    df: pd.DataFrame, n_classes: int, images_per_class: int, seed: int,
    train_frac: float, val_frac: float,
) -> tuple[pd.DataFrame, dict]:
    """Select n_classes classes and up to images_per_class images per class,
    deterministically (seed=42 by default). Re-derives train/val/stream splits
    fresh on the trimmed per-class pool — see module docstring for why. Labels
    are remapped to a contiguous 0..n_classes-1 range (alphabetical over just
    the selected class names, so the remapping itself is deterministic and
    reproducible from class_name alone, independent of selection order).

    Returns (trimmed_manifest_df, class_index_dict) — the caller writes both.
    """
    all_classes = sorted(df["class_name"].unique())
    rng = np.random.default_rng(seed)

    if n_classes < len(all_classes):
        chosen_idx = rng.choice(len(all_classes), size=n_classes, replace=False)
        selected_classes = sorted(all_classes[i] for i in chosen_idx)
    else:
        selected_classes = all_classes

    new_label = {name: i for i, name in enumerate(selected_classes)}
    original_label = {
        name: int(df.loc[df["class_name"] == name, "label"].iloc[0])
        for name in selected_classes
    }

    rows_train, rows_val, rows_stream = [], [], []
    for class_name in selected_classes:
        class_rows = (
            df[df["class_name"] == class_name]
            .sort_values("input_path")
            .reset_index(drop=True)
        )
        n_avail = len(class_rows)
        k = min(images_per_class, n_avail)
        chosen_idx = np.sort(rng.choice(n_avail, size=k, replace=False))
        chosen = class_rows.iloc[chosen_idx].reset_index(drop=True)
        chosen["label"] = new_label[class_name]

        n = len(chosen)
        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)
        records = chosen.to_dict("records")
        for i, row in enumerate(records):
            if i < train_end:
                row["split"] = "train"
                rows_train.append(row)
            elif i < val_end:
                row["split"] = "val"
                rows_val.append(row)
            else:
                row["split"] = "stream"
                rows_stream.append(row)

    trimmed = pd.DataFrame(rows_train + rows_val + rows_stream)

    class_index = {
        "index_to_class": {str(v): k for k, v in new_label.items()},
        "class_to_index": new_label,
        "index_to_original_label": {str(v): original_label[k] for k, v in new_label.items()},
        "source": "trimmed_subset_of_provided_lookup_alphabetical",
        "n_classes": len(new_label),
        "seed": seed,
        "images_per_class_requested": images_per_class,
    }

    print(f"ImageNet trim: {len(selected_classes)} classes selected "
          f"(of {len(all_classes)}), seed={seed}")
    print(f"ImageNet after trim ({len(trimmed)} / {len(df)}):")
    print(trimmed.groupby("split").size().to_string())
    return trimmed, class_index


# ── Main ──────────────────────────────────────────────────────────────────────

STRATEGIES = {
    "bdd100k":  _trim_bdd100k,
    "iwildcam": _trim_iwildcam,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim a CV dataset manifest.")
    parser.add_argument("--dataset",  required=True, choices=list(STRATEGIES) + ["imagenet"],
                        help="Dataset name (bdd100k, iwildcam, or imagenet)")
    parser.add_argument("--manifest", required=True,
                        help="Path to the full manifest CSV produced by preprocess_*.py")
    parser.add_argument("--target",   type=int, default=3000,
                        help="Target number of images to keep (bdd100k/iwildcam only; default: 3000)")
    parser.add_argument("--n-classes", type=int, default=100,
                        help="ImageNet only: number of classes to keep (default: 100)")
    parser.add_argument("--images-per-class", type=int, default=100,
                        help="ImageNet only: images to keep per class (default: 100)")
    parser.add_argument("--train-frac", type=float, default=0.7,
                        help="ImageNet only: re-derived per-class train fraction (default: 0.7, matches configs/datasets/imagenet.json)")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="ImageNet only: re-derived per-class val fraction (default: 0.1, matches configs/datasets/imagenet.json)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print statistics without writing any files")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(df)} rows from {manifest_path}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    print()

    class_index: dict | None = None
    if args.dataset == "imagenet":
        trimmed, class_index = _trim_imagenet(
            df, args.n_classes, args.images_per_class, args.seed,
            args.train_frac, args.val_frac,
        )
    else:
        strategy = STRATEGIES[args.dataset]
        trimmed = strategy(df, args.target, args.seed)

    print(f"\nTrimmed: {len(df)} -> {len(trimmed)} rows")

    if args.dry_run:
        print("(dry-run — no files written)")
        return

    # Back up original manifest
    backup = manifest_path.with_name(manifest_path.stem + "_full.csv")
    if not backup.exists():
        shutil.copy(manifest_path, backup)
        print(f"Original backed up to: {backup}")
    else:
        print(f"Backup already exists at: {backup} (not overwritten)")

    trimmed.to_csv(manifest_path, index=False)
    print(f"Trimmed manifest written to: {manifest_path}")

    if class_index is not None:
        class_index_path = manifest_path.with_name("class_index_trimmed.json")
        with open(class_index_path, "w") as f:
            json.dump(class_index, f, indent=2)
        print(f"Trimmed class index written to: {class_index_path}")


if __name__ == "__main__":
    main()

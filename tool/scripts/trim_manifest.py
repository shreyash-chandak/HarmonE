"""
scripts/trim_manifest.py — Trim a CV dataset manifest to experiment-appropriate size.

The full preprocess scripts produce manifests covering all available images
(10k val for BDD100K, 42k ood_test for iWildCam).  Running experiments on the
full set would take many hours per run.  This script trims the manifest CSV to a
target size while preserving the drift ordering that each dataset is designed to
represent.

    BDD100K    → sample ~N images per domain from the val split
                 (default: 500/domain × 6 domains = ~3 000)
    iWildCam   → sample ~N ood_test images maintaining location grouping
                 (default: 3 000, distributed proportionally across locations)
    ACDC       → no trimming; total train+val is already ~2 006 frames.

Image files are NOT deleted.  The trimmed manifest is written alongside the
original (original is kept as <name>_full.csv).

Usage:
    cd tool/

    # BDD100K — trim val to ~500 images per domain:
    python scripts/trim_manifest.py \\
        --dataset bdd100k \\
        --manifest data/bdd100k/bdd100k_manifest.csv \\
        --target   3000

    # iWildCam — trim ood_test to ~3 000:
    python scripts/trim_manifest.py \\
        --dataset iwildcam \\
        --manifest data/iwildcam/iwildcam_manifest.csv \\
        --target   3000

    # Check size without writing:
    python scripts/trim_manifest.py --dataset bdd100k --manifest ... --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

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


# ── Main ──────────────────────────────────────────────────────────────────────

STRATEGIES = {
    "bdd100k":  _trim_bdd100k,
    "iwildcam": _trim_iwildcam,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim a CV dataset manifest.")
    parser.add_argument("--dataset",  required=True, choices=list(STRATEGIES),
                        help="Dataset name (bdd100k or iwildcam)")
    parser.add_argument("--manifest", required=True,
                        help="Path to the full manifest CSV produced by preprocess_*.py")
    parser.add_argument("--target",   type=int, default=3000,
                        help="Target number of images to keep (default: 3000)")
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

    strategy = STRATEGIES[args.dataset]
    trimmed = strategy(df, args.target, args.seed)

    print(f"\nTrimmed: {len(df)} → {len(trimmed)} rows")

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


if __name__ == "__main__":
    main()

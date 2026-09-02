"""
managed_system_cv/utility/drift/induce_imagenet_c.py — Build a corruption-ordered
("ImageNet-C-lite") drift manifest for the ImageNet dataset (C4).

Why this exists: ImageNet-1k ships as flat per-class folders with no natural
drift attribute (unlike BDD100K's weather/timeofday, iWildCam's camera-trap
location, or ACDC's adverse-condition folders). Without an induced drift axis,
ImageNet-in-HarmonE would be a static, single-domain stream on which no
adaptive planner ever sees a violation — an inert addition to the experiment
grid. This script gives it one, using the exact benchmark TENT (Wang et al.,
ICLR 2021) is validated against: common image corruptions at increasing
severity (Hendrycks & Dietterich, 2019, "ImageNet-C"). It also follows Augur's
(Lewis et al., 2022) drift-induction methodology in spirit: a deliberately
engineered, documented, reproducible drift schedule built for evaluation
purposes, not a naturally-occurring one — see context/idea.md for the full
discussion connecting this script to both papers.

Corruption implementations here are SIMPLIFIED/APPROXIMATE versions of the
official Hendrycks & Dietterich corruptions (numpy/PIL only, no dependency on
the `imagecorruptions` package). They are not calibrated to the five official
severity levels; treat the `--severity` value as a relative dial (1=mild,
5=severe), not an exact reproduction of the published ImageNet-C benchmark.

Domain schedule (clean -> progressively different corruption families,
mirroring the condition-ordering convention already used for BDD100K/ACDC):
    clean -> gaussian_noise -> defocus_blur -> fog -> brightness_low -> contrast_low

Output:
    data/imagenet_c/<domain>/<sample_id>.jpg
    data/imagenet_c/imagenet_c_manifest.csv  — columns: sample_id, input_path,
        image_path, label, class_name, domain, split ("stream" for every row —
        this manifest is inference-only; pair it with
        configs/datasets/imagenet_c.json, which sets train_frac=val_frac=0.0).

Usage:
    cd tool/
    python managed_system_cv/utility/drift/induce_imagenet_c.py \\
        --manifest data/imagenet/imagenet_manifest.csv \\
        --output-dir data/imagenet_c \\
        --target-per-domain 500 --severity 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

RANDOM_SEED = 1

# ── Corruption functions ────────────────────────────────────────────────────
# severity in [1, 5]; each function returns a new PIL.Image (RGB).

def gaussian_noise(img: Image.Image, severity: int) -> Image.Image:
    std = [0.03, 0.06, 0.09, 0.14, 0.20][severity - 1]
    arr = np.asarray(img).astype(np.float32) / 255.0
    noisy = arr + np.random.normal(0, std, arr.shape)
    return Image.fromarray((np.clip(noisy, 0, 1) * 255).astype(np.uint8))


def defocus_blur(img: Image.Image, severity: int) -> Image.Image:
    radius = [1, 2, 3, 5, 7][severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def fog(img: Image.Image, severity: int) -> Image.Image:
    """Contrast reduction toward the per-channel mean — same construction as
    managed_system_cv/utility/drift/induce.py's apply_fog(), reused here for
    consistency with the rest of the codebase's 'fog' semantics."""
    factor = [0.85, 0.70, 0.55, 0.40, 0.25][severity - 1]
    arr = np.asarray(img).astype(np.float32)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    foggy = np.clip((arr - mean) * factor + mean, 0, 255)
    return Image.fromarray(foggy.astype(np.uint8))


def brightness_low(img: Image.Image, severity: int) -> Image.Image:
    factor = [0.80, 0.65, 0.50, 0.35, 0.20][severity - 1]
    return ImageEnhance.Brightness(img).enhance(factor)


def contrast_low(img: Image.Image, severity: int) -> Image.Image:
    factor = [0.75, 0.60, 0.45, 0.30, 0.15][severity - 1]
    return ImageEnhance.Contrast(img).enhance(factor)


_CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "defocus_blur": defocus_blur,
    "fog": fog,
    "brightness_low": brightness_low,
    "contrast_low": contrast_low,
}

_DOMAIN_ORDER = ["clean"] + list(_CORRUPTIONS.keys())


def _stratified_sample(df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """Sample ~target rows, proportional per class, preserving class balance.

    Iterates the groupby object directly rather than groupby(...).apply(...) —
    pandas >=2.2 excludes the grouping column from the frame handed to an
    .apply() callback (the include_groups=False default, made the only
    behavior in pandas 3.0), which silently dropped "class_name" here.
    """
    if len(df) <= target:
        return df
    frac = target / len(df)
    parts = [g.sample(frac=frac, random_state=seed) if len(g) > 1 else g
             for _, g in df.groupby("class_name")]
    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(n=min(target, len(sampled)), random_state=seed).reset_index(drop=True)


def build_imagenet_c(
    manifest_path: str,
    output_dir: str,
    target_per_domain: int,
    severity: int,
    seed: int,
) -> pd.DataFrame:
    np.random.seed(seed)
    base = pd.read_csv(manifest_path)
    stream = base[base["split"] == "stream"].reset_index(drop=True)
    if stream.empty:
        raise ValueError(
            f"No rows with split=='stream' in {manifest_path}. "
            "Run scripts/preprocess_imagenet.py first."
        )
    print(f"Base manifest: {len(stream)} stream-eligible images across "
          f"{stream['class_name'].nunique()} classes.")

    out_root = Path(output_dir)
    rows = []
    used_ids: set[str] = set()

    for domain in _DOMAIN_ORDER:
        # Sample a fresh, non-overlapping subset per domain so no image
        # appears twice in the drift-ordered stream.
        remaining = stream[~stream["sample_id"].isin(used_ids)]
        subset = _stratified_sample(remaining, target_per_domain, seed=seed + len(used_ids))
        used_ids.update(subset["sample_id"].tolist())

        domain_dir = out_root / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        fn = _CORRUPTIONS.get(domain)  # None for "clean"
        print(f"  domain={domain}: {len(subset)} images"
              + (f" (severity={severity})" if fn else " (no corruption applied)"))

        for _, row in subset.iterrows():
            safe_id = row["sample_id"].replace("/", "__")
            out_path = domain_dir / f"{safe_id}.jpg"
            if not out_path.exists():
                img = Image.open(row["input_path"]).convert("RGB")
                if fn is not None:
                    img = fn(img, severity)
                img.save(out_path, quality=90)

            rows.append({
                "sample_id": f"{domain}/{safe_id}",
                "input_path": str(out_path),
                "image_path": str(out_path),
                "label": int(row["label"]),
                "class_name": row["class_name"],
                "domain": domain,
                "split": "stream",
            })

    df = pd.DataFrame(rows)  # already in domain (clean-first) order
    manifest_out = out_root / "imagenet_c_manifest.csv"
    df.to_csv(manifest_out, index=False)
    print(f"\nSaved {len(df)} rows to {manifest_out}")
    print(df.groupby("domain").size().reindex(_DOMAIN_ORDER).to_string())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a corruption-ordered drift manifest for ImageNet (ImageNet-C-lite)."
    )
    parser.add_argument("--manifest", default="data/imagenet/imagenet_manifest.csv",
                        help="Base manifest from scripts/preprocess_imagenet.py")
    parser.add_argument("--output-dir", default="data/imagenet_c")
    parser.add_argument("--target-per-domain", type=int, default=500,
                        help="Images per domain block (default 500, matching the "
                             "~3000-image CV target-size convention: 500 x 6 domains).")
    parser.add_argument("--severity", type=int, default=3, choices=[1, 2, 3, 4, 5],
                        help="Relative corruption severity dial, 1 (mild) to 5 (severe). "
                             "NOT calibrated to official Hendrycks & Dietterich levels.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    build_imagenet_c(
        args.manifest, args.output_dir, args.target_per_domain, args.severity, args.seed
    )


if __name__ == "__main__":
    main()

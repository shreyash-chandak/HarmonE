"""scripts/make_toy_datasets.py — Generate synthetic toy datasets for conformance tests.

Creates:
  tool/data/toy_regression/dataset.csv     — 5000 rows, sinusoidal + noise
  tool/data/toy_cv/images/frame_NNNN.jpg   — 60 solid-colour images (640×480)
  tool/data/toy_cv/manifest.csv            — image_path column

Both datasets are minimal but schema-compliant so the plug-and-play test
can validate → init → run a short headless session without real data.

Usage:
    cd tool/
    python3 scripts/make_toy_datasets.py [--force]
"""
from __future__ import annotations

import argparse
from pathlib import Path

_TOOL_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _TOOL_DIR / "data"


def make_toy_regression(force: bool = False) -> Path:
    """Create data/toy_regression/dataset.csv (5 000 rows)."""
    import numpy as np
    import pandas as pd

    out_dir = _DATA_DIR / "toy_regression"
    out_csv = out_dir / "dataset.csv"

    if out_csv.exists() and not force:
        print(f"  ✔ {out_csv.relative_to(_TOOL_DIR)} already exists (--force to overwrite)")
        return out_csv

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 5000
    t = np.linspace(0, 4 * np.pi, n)
    flow = 100 + 50 * np.sin(t) + 10 * rng.standard_normal(n)
    df = pd.DataFrame({"flow": flow.round(4)})
    df.to_csv(out_csv, index=False)
    print(f"  ✔ Created {out_csv.relative_to(_TOOL_DIR)} ({n} rows)")
    return out_csv


def make_toy_cv(force: bool = False) -> tuple[Path, Path]:
    """Create data/toy_cv/images/*.jpg (60 images) and manifest.csv."""
    try:
        from PIL import Image
    except ImportError:
        print("  ⚠  Pillow not installed — skipping toy CV dataset. pip install pillow")
        return Path(), Path()

    import numpy as np
    import pandas as pd

    img_dir = _DATA_DIR / "toy_cv" / "images"
    manifest_csv = _DATA_DIR / "toy_cv" / "manifest.csv"

    if manifest_csv.exists() and not force:
        print(f"  ✔ {manifest_csv.relative_to(_TOOL_DIR)} already exists (--force to overwrite)")
        return img_dir, manifest_csv

    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    names: list[str] = []

    for i in range(60):
        # Solid colour images with per-image random RGB so histograms vary
        r, g, b = (int(rng.integers(30, 225)) for _ in range(3))
        arr = np.full((480, 640, 3), [r, g, b], dtype=np.uint8)
        img_name = f"frame_{i:04d}.jpg"
        Image.fromarray(arr).save(img_dir / img_name, quality=85)
        names.append(f"images/{img_name}")

    manifest = pd.DataFrame({"image_path": names})
    manifest.to_csv(manifest_csv, index=False)
    print(f"  ✔ Created {manifest_csv.relative_to(_TOOL_DIR)} (60 images)")
    return img_dir, manifest_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toy datasets for conformance tests.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    print("[make_toy_datasets] Creating toy regression dataset …")
    make_toy_regression(args.force)

    print("[make_toy_datasets] Creating toy CV dataset …")
    make_toy_cv(args.force)

    print("[make_toy_datasets] Done.")


if __name__ == "__main__":
    main()

"""tests/test_preprocess_imagenet.py — scripts/preprocess_imagenet.py (2026-08-30 fix).

Covers build_manifest() against a tiny synthetic 2-class directory layout,
mirroring the real data/imagenet/ structure: class folders + class_mapping.csv
+ gt.csv, with image_id = "{class_idx:03d}-{img_idx:03d}" over sorted class
folder order and sorted per-class filename order (verified against the real
539 826-image dataset before this fix was written — see
context/DECISIONS_PENDING.md DP24).
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.preprocess_imagenet import build_manifest


def _make_dataset(root: Path, classes: dict[str, int]) -> None:
    """classes: {class_name: n_images}, folders created in sorted-name order."""
    for name, n in classes.items():
        cdir = root / name
        cdir.mkdir(parents=True)
        for i in range(n):
            (cdir / f"{i:03d}.jpg").write_bytes(b"")

    with open(root / "class_mapping.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "class_name"])
        w.writeheader()
        for idx, name in enumerate(sorted(classes.keys())):
            w.writerow({"label": idx, "class_name": name})

    with open(root / "gt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "label"])
        w.writeheader()
        for class_idx, name in enumerate(sorted(classes.keys())):
            for img_idx in range(classes[name]):
                w.writerow({"image_id": f"{class_idx:03d}-{img_idx:03d}", "label": class_idx})


class TestBuildManifest:
    def test_labels_and_row_counts(self, tmp_path):
        _make_dataset(tmp_path, {"abacus": 10, "abaya": 4})
        df = build_manifest(
            str(tmp_path), str(tmp_path / "manifest.csv"), str(tmp_path / "class_index.json"),
            str(tmp_path / "class_mapping.csv"), str(tmp_path / "gt.csv"),
            train_frac=0.5, val_frac=0.2,
        )
        assert len(df) == 14
        abacus_rows = df[df["class_name"] == "abacus"]
        abaya_rows = df[df["class_name"] == "abaya"]
        assert (abacus_rows["label"] == 0).all()
        assert (abaya_rows["label"] == 1).all()

    def test_split_boundaries_positional_and_contiguous(self, tmp_path):
        _make_dataset(tmp_path, {"abacus": 10})
        df = build_manifest(
            str(tmp_path), str(tmp_path / "manifest.csv"), str(tmp_path / "class_index.json"),
            str(tmp_path / "class_mapping.csv"), str(tmp_path / "gt.csv"),
            train_frac=0.6, val_frac=0.2,
        )
        assert list(df["split"]) == ["train"] * 6 + ["val"] * 2 + ["stream"] * 2

    def test_class_index_json_source_is_honest(self, tmp_path):
        _make_dataset(tmp_path, {"abacus": 2, "abaya": 2})
        build_manifest(
            str(tmp_path), str(tmp_path / "manifest.csv"), str(tmp_path / "class_index.json"),
            str(tmp_path / "class_mapping.csv"), str(tmp_path / "gt.csv"),
            train_frac=0.5, val_frac=0.0,
        )
        import json
        idx = json.loads((tmp_path / "class_index.json").read_text())
        assert idx["source"] == "provided_lookup_alphabetical"
        assert idx["n_classes"] == 2

    def test_missing_class_mapping_entry_raises(self, tmp_path):
        _make_dataset(tmp_path, {"abacus": 2})
        (tmp_path / "extra_class").mkdir()
        (tmp_path / "extra_class" / "000.jpg").write_bytes(b"")
        with pytest.raises(RuntimeError, match="no entry in"):
            build_manifest(
                str(tmp_path), str(tmp_path / "manifest.csv"), str(tmp_path / "class_index.json"),
                str(tmp_path / "class_mapping.csv"), str(tmp_path / "gt.csv"),
                train_frac=0.5, val_frac=0.0,
            )

    def test_conflicting_gt_and_class_mapping_labels_raise(self, tmp_path):
        _make_dataset(tmp_path, {"abacus": 2})
        # Corrupt gt.csv so its label disagrees with class_mapping.csv's.
        with open(tmp_path / "gt.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image_id", "label"])
            w.writeheader()
            w.writerow({"image_id": "000-000", "label": 99})
            w.writerow({"image_id": "000-001", "label": 99})
        with pytest.raises(RuntimeError, match="disagree"):
            build_manifest(
                str(tmp_path), str(tmp_path / "manifest.csv"), str(tmp_path / "class_index.json"),
                str(tmp_path / "class_mapping.csv"), str(tmp_path / "gt.csv"),
                train_frac=0.5, val_frac=0.0,
            )

    def test_image_with_no_gt_row_raises(self, tmp_path):
        _make_dataset(tmp_path, {"abacus": 3})
        # Drop the last row so one image has no ground truth.
        rows = list(csv.DictReader(open(tmp_path / "gt.csv", newline="")))
        with open(tmp_path / "gt.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image_id", "label"])
            w.writeheader()
            w.writerows(rows[:-1])
        with pytest.raises(RuntimeError, match="no matching image_id"):
            build_manifest(
                str(tmp_path), str(tmp_path / "manifest.csv"), str(tmp_path / "class_index.json"),
                str(tmp_path / "class_mapping.csv"), str(tmp_path / "gt.csv"),
                train_frac=0.5, val_frac=0.0,
            )

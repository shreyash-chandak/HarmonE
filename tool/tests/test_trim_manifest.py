"""tests/test_trim_manifest.py — scripts/trim_manifest.py's ImageNet strategy (2026-08-30).

ImageNet needs a different trim shape than BDD100K/iWildCam's flat image-count
target: N classes x M images/class (a flat total would leave too few images
per class across 1000 classes for real classification signal). Covers
_trim_imagenet()'s class selection, per-class image selection, label
remapping to a contiguous 0..N-1 range, and re-derived train/val/stream
splits — all against a small synthetic manifest.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.trim_manifest import _trim_imagenet


def _make_full_manifest(n_classes: int, images_per_class: int) -> pd.DataFrame:
    class_names = [f"class_{i:03d}" for i in range(n_classes)]
    rows = []
    for label, name in enumerate(class_names):
        for j in range(images_per_class):
            rows.append({
                "sample_id": f"{name}/{j:03d}",
                "input_path": f"data/imagenet/{name}/{j:03d}.jpg",
                "image_path": f"data/imagenet/{name}/{j:03d}.jpg",
                "label": label,
                "class_name": name,
                "domain": "clean",
                "split": "train" if j < images_per_class * 0.7 else "stream",
            })
    return pd.DataFrame(rows)


class TestTrimImagenet:
    def test_selects_requested_class_and_image_counts(self):
        df = _make_full_manifest(n_classes=20, images_per_class=50)
        trimmed, class_index = _trim_imagenet(
            df, n_classes=5, images_per_class=10, seed=42, train_frac=0.7, val_frac=0.1,
        )
        assert trimmed["class_name"].nunique() == 5
        counts = trimmed.groupby("class_name").size()
        assert (counts == 10).all()
        assert class_index["n_classes"] == 5

    def test_deterministic_with_fixed_seed(self):
        df = _make_full_manifest(n_classes=20, images_per_class=50)
        t1, _ = _trim_imagenet(df, 5, 10, seed=42, train_frac=0.7, val_frac=0.1)
        t2, _ = _trim_imagenet(df, 5, 10, seed=42, train_frac=0.7, val_frac=0.1)
        assert sorted(t1["sample_id"]) == sorted(t2["sample_id"])

    def test_different_seed_selects_differently(self):
        df = _make_full_manifest(n_classes=50, images_per_class=50)
        t1, _ = _trim_imagenet(df, 5, 10, seed=1, train_frac=0.7, val_frac=0.1)
        t2, _ = _trim_imagenet(df, 5, 10, seed=2, train_frac=0.7, val_frac=0.1)
        assert set(t1["class_name"]) != set(t2["class_name"])

    def test_labels_remapped_to_contiguous_range(self):
        df = _make_full_manifest(n_classes=10, images_per_class=20)
        trimmed, class_index = _trim_imagenet(
            df, n_classes=3, images_per_class=5, seed=42, train_frac=0.7, val_frac=0.1,
        )
        assert set(trimmed["label"].unique()) == {0, 1, 2}
        assert set(class_index["class_to_index"].values()) == {0, 1, 2}

    def test_splits_rederived_per_class_all_present(self):
        """Every selected class must appear in train, val, AND stream — not
        siloed by whatever split the full manifest happened to assign to the
        randomly-retained rows."""
        df = _make_full_manifest(n_classes=10, images_per_class=100)
        trimmed, _ = _trim_imagenet(
            df, n_classes=3, images_per_class=20, seed=42, train_frac=0.7, val_frac=0.1,
        )
        for class_name in trimmed["class_name"].unique():
            splits = set(trimmed[trimmed["class_name"] == class_name]["split"])
            assert splits == {"train", "val", "stream"}, f"{class_name} missing a split: {splits}"

    def test_split_sizes_match_fractions(self):
        df = _make_full_manifest(n_classes=1, images_per_class=100)
        trimmed, _ = _trim_imagenet(
            df, n_classes=1, images_per_class=100, seed=42, train_frac=0.7, val_frac=0.1,
        )
        counts = trimmed["split"].value_counts().to_dict()
        assert counts == {"train": 70, "val": 10, "stream": 20}

    def test_fewer_images_available_than_requested_takes_all(self):
        df = _make_full_manifest(n_classes=2, images_per_class=5)
        trimmed, _ = _trim_imagenet(
            df, n_classes=2, images_per_class=100, seed=42, train_frac=0.7, val_frac=0.1,
        )
        assert len(trimmed) == 10  # 2 classes x 5 available each, not 100

    def test_n_classes_gte_available_keeps_all_classes(self):
        df = _make_full_manifest(n_classes=3, images_per_class=10)
        trimmed, class_index = _trim_imagenet(
            df, n_classes=100, images_per_class=5, seed=42, train_frac=0.7, val_frac=0.1,
        )
        assert trimmed["class_name"].nunique() == 3
        assert class_index["n_classes"] == 3

    def test_class_index_records_original_label(self):
        df = _make_full_manifest(n_classes=5, images_per_class=10)
        _, class_index = _trim_imagenet(
            df, n_classes=2, images_per_class=5, seed=42, train_frac=0.7, val_frac=0.1,
        )
        for new_label_str, orig_label in class_index["index_to_original_label"].items():
            class_name = class_index["index_to_class"][new_label_str]
            expected_orig = int(class_name.split("_")[1])
            assert orig_label == expected_orig

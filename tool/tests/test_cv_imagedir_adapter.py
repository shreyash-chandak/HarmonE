"""tests/test_cv_imagedir_adapter.py — CVImageDirAdapter manifest parsing.

Covers the train_labels() addition (2026-08-30, added to support CV initial
train — see experiments/run_experiment.py::_train_cv_models_if_missing):
manifests carry ground truth either as an inline "label" int column
(classification) or a "label_path" file column (segmentation/detection),
never both. train_split()/val_split()/stream() positional-split arithmetic
is unchanged; these tests pin it down alongside the new accessor since there
was previously no dedicated adapter test file.
"""

from __future__ import annotations

import csv
from pathlib import Path

from adapters.cv_imagedir import CVImageDirAdapter


def _write_manifest(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _touch_images(tmp_path: Path, names: list[str]) -> None:
    for n in names:
        (tmp_path / n).write_bytes(b"")


class TestInlineLabelClassification:
    """Manifest with a 'label' column (iWildCam/ImageNet-style)."""

    def _make_config(self, tmp_path: Path, n: int) -> dict:
        names = [f"img{i}.jpg" for i in range(n)]
        _touch_images(tmp_path, names)
        rows = [{"input_path": nm, "label": str(i % 3)} for i, nm in enumerate(names)]
        manifest = tmp_path / "manifest.csv"
        _write_manifest(manifest, rows, ["input_path", "label"])
        return {
            "manifest_csv": str(manifest),
            "data_root": str(tmp_path),
            "train_frac": 0.6,
            "val_frac": 0.2,
            "models": {},
        }

    def test_train_labels_returns_inline_ints(self, tmp_path):
        cfg = self._make_config(tmp_path, 10)
        adapter = CVImageDirAdapter(cfg)
        label_paths, inline_labels = adapter.train_labels()
        assert len(label_paths) == len(adapter.train_split()) == 6
        assert all(lp is None for lp in label_paths)
        assert inline_labels == [i % 3 for i in range(6)]

    def test_stream_excludes_train_and_val(self, tmp_path):
        cfg = self._make_config(tmp_path, 10)
        adapter = CVImageDirAdapter(cfg)
        stream_paths = [s.inputs for s in adapter.stream()]
        assert len(stream_paths) == 2  # 10 - 6 (train) - 2 (val)

    def test_offline_labels_none_when_manifest_has_no_label_path_column(self, tmp_path):
        cfg = self._make_config(tmp_path, 10)
        adapter = CVImageDirAdapter(cfg)
        assert adapter.offline_labels() is None


class TestLabelPathSegmentation:
    """Manifest with a 'label_path' column (ACDC/BDD100K-style)."""

    def _make_config(self, tmp_path: Path, n: int) -> dict:
        names = [f"img{i}.png" for i in range(n)]
        mask_names = [f"mask{i}.png" for i in range(n)]
        _touch_images(tmp_path, names + mask_names)
        rows = [
            {"input_path": nm, "label_path": mk}
            for nm, mk in zip(names, mask_names)
        ]
        manifest = tmp_path / "manifest.csv"
        _write_manifest(manifest, rows, ["input_path", "label_path"])
        return {
            "manifest_csv": str(manifest),
            "data_root": str(tmp_path),
            "train_frac": 0.7,
            "val_frac": 0.1,
            "models": {},
        }

    def test_train_labels_returns_label_paths(self, tmp_path):
        cfg = self._make_config(tmp_path, 10)
        adapter = CVImageDirAdapter(cfg)
        label_paths, inline_labels = adapter.train_labels()
        assert len(label_paths) == 7
        assert all(lp is not None and lp.endswith(".png") for lp in label_paths)
        assert all(lbl is None for lbl in inline_labels)

    def test_offline_labels_populated(self, tmp_path):
        cfg = self._make_config(tmp_path, 10)
        adapter = CVImageDirAdapter(cfg)
        stream_labels = adapter.offline_labels()
        assert stream_labels is not None
        assert len(stream_labels) == 2  # 10 - 7 (train) - 1 (val, int(10*0.1)=1)


class TestImageDirOnly:
    """No manifest — plain image_dir mode has no labels at all."""

    def test_train_labels_all_none(self, tmp_path):
        _touch_images(tmp_path, [f"img{i}.jpg" for i in range(5)])
        cfg = {"image_dir": str(tmp_path), "train_frac": 0.6, "val_frac": 0.0, "models": {}}
        adapter = CVImageDirAdapter(cfg)
        label_paths, inline_labels = adapter.train_labels()
        assert all(lp is None for lp in label_paths)
        assert all(lbl is None for lbl in inline_labels)
        assert len(label_paths) == len(adapter.train_split()) == 3

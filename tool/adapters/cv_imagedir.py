"""adapters/cv_imagedir.py — DatasetAdapter for image-directory or manifest streams.

Implements the DatasetAdapter contract for CV datasets where the stream is an
ordered sequence of images from a flat directory or a manifest CSV file.

Stream order:
  - image_dir only: sorted filename order
  - manifest_csv:   row order (the preprocessing step controls ordering)

The adapter never reads labels at runtime (I4).  Label paths (manifest column
"label_path") are available only via offline_labels() for experiments/offline_eval.py.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterator

from .base import DatasetAdapter, Sample, ModelSpec


class CVImageDirAdapter(DatasetAdapter):
    """DatasetAdapter for image directory (sorted filename) or manifest CSV streams.

    Config keys consumed:
        image_dir (str):   Path to image directory (option A — sorting by filename).
        manifest_csv (str | null): Path to manifest CSV (option B — row-order stream).
        data_root (str):   Base directory for resolving image_path in manifest (default ".").
        train_frac (float): Fraction for training split (default 0.8).
        val_frac (float):   Fraction for validation split (default 0.0).
        models (dict):     Model spec catalogue.
        stream_delay_s:    Ignored by adapter (live system uses it externally).
    """

    domain = "cv"
    self_labeling = False

    def __init__(self, config: dict, tool_dir: str | None = None) -> None:
        self._config = config
        self._tool_dir = tool_dir or os.getcwd()

        manifest_csv = config.get("manifest_csv")
        image_dir = config.get("image_dir")
        data_root = config.get("data_root", ".")

        # Resolve paths relative to tool_dir
        def _resolve(p: str) -> Path:
            path = Path(p)
            if not path.is_absolute():
                path = Path(self._tool_dir) / path
            return path

        if manifest_csv:
            self._image_paths, self._label_paths = _load_manifest(
                _resolve(manifest_csv), _resolve(data_root)
            )
        elif image_dir:
            img_dir = _resolve(image_dir)
            extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            self._image_paths = sorted(
                str(p) for p in img_dir.iterdir()
                if p.suffix.lower() in extensions
            )
            self._label_paths = [None] * len(self._image_paths)
        else:
            self._image_paths = []
            self._label_paths = []

        n = len(self._image_paths)
        train_frac = float(config.get("train_frac", 0.8))
        val_frac = float(config.get("val_frac", 0.0))
        self._train_end = int(n * train_frac)
        self._val_end = self._train_end + int(n * val_frac)

        # Model catalogue
        self._models: dict[str, ModelSpec] = {}
        for name, spec in config.get("models", {}).items():
            wp = spec["weights_path"]
            if not os.path.isabs(wp):
                wp = str(Path(self._tool_dir) / wp)
            self._models[name] = ModelSpec(
                name=name,
                weights_path=wp,
                loader=spec.get("loader", "adapters.loaders.yolo_loader"),
                cost_class=spec.get("cost_class", "medium"),
            )

    @classmethod
    def from_config(cls, config: dict) -> "CVImageDirAdapter":
        return cls(config)

    def train_split(self) -> list[str]:
        return self._image_paths[: self._train_end]

    def val_split(self) -> list[str]:
        return self._image_paths[self._train_end : self._val_end]

    def stream(self) -> Iterator[Sample]:
        for idx, path in enumerate(self._image_paths[self._val_end :]):
            yield Sample(index=idx, inputs=path, ground_truth=None)

    def offline_labels(self) -> list[str | None] | None:
        """Return label paths for the stream portion, or None if no labels."""
        stream_labels = self._label_paths[self._val_end :]
        if all(lp is None for lp in stream_labels):
            return None
        return stream_labels

    def models(self) -> dict[str, ModelSpec]:
        return self._models


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_manifest(
    manifest_path: Path, data_root: Path
) -> tuple[list[str], list[str | None]]:
    """Parse a manifest CSV and return (image_paths, label_paths)."""
    image_paths: list[str] = []
    label_paths: list[str | None] = []

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # preprocess scripts write "input_path"; fall back to legacy "image_path"
            img = row.get("input_path", row.get("image_path", ""))
            # pathlib: absolute img overrides data_root, relative is joined
            image_paths.append(str(Path(data_root) / img) if img else "")

            label = row.get("label_path", "")
            label_paths.append(str(Path(data_root) / label) if label else None)

    return image_paths, label_paths

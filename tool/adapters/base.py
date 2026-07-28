"""
adapters/base.py — DatasetAdapter interface.

Every dataset is accessed exclusively through a DatasetAdapter constructed from
a config file in configs/datasets/<name>.json. No dataset-specific path, column
name, class list, or threshold may appear anywhere outside the config + adapter.

The adapter is the single integration point for adding a new dataset: write the
config JSON and optionally subclass an existing generic adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class Sample:
    """A single streaming inference input produced by DatasetAdapter.stream()."""
    index: int                 # 0-based position in the stream
    inputs: Any                # model-ready input (numpy array, tensor, image path, …)
    ground_truth: Any | None   # available only for self-labeling or offline eval


@dataclass
class ModelSpec:
    """Specification for one candidate model in this dataset."""
    name: str                  # short identifier (e.g. "lstm", "yolo_n")
    weights_path: str          # absolute path to weights file
    loader: str                # module path to a load() function, e.g. "adapters.loaders.lstm"
    cost_class: str            # "light" | "medium" | "heavy" (qualitative energy tier)


class DatasetAdapter(ABC):
    """Constructed entirely from a configs/datasets/<name>.json config.

    The adapter owns:
    - dataset I/O (splits, streaming, offline labels)
    - the model catalogue for this dataset
    - stream transforms (drift induction, normalisation)

    It does NOT run inference or maintain state beyond the stream cursor.
    """

    domain: str          # "regression" | "cv" — set on each subclass
    self_labeling: bool  # True if ground truth is available at inference time

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "DatasetAdapter":
        """Construct from a parsed configs/datasets/<name>.json dict."""

    @abstractmethod
    def train_split(self) -> Any:
        """Return training data for initial scaler/reference fitting."""

    @abstractmethod
    def val_split(self) -> Any:
        """Return validation data for proxy calibration."""

    @abstractmethod
    def stream(self) -> Iterator[Sample]:
        """Yield test-set samples in chronological order."""

    def offline_labels(self) -> Any | None:
        """Return ground-truth labels for the full test split, or None if absent.

        Used by experiments/offline_eval.py only; must not be called at runtime.
        Default: None (no labels available).
        """
        return None

    @abstractmethod
    def models(self) -> dict[str, ModelSpec]:
        """Return the candidate model catalogue for this dataset.

        Keys are the model names used in model.csv / ema_scores.
        """

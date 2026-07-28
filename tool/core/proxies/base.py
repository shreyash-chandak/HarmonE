"""
core/proxies/base.py — AccuracyProxy interface.

Accuracy proxies replace ground-truth accuracy signals in CV deployments
where labels are unavailable at inference time. All proxies must implement
this interface so monitor.py can swap them via config without code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AccuracyProxy(ABC):
    """Base class for all accuracy proxy implementations."""

    name: str  # set on each subclass

    def setup(self, models: dict, val_split: Any) -> None:
        """Optional one-time calibration on validation data.

        Args:
            models:    Dict of {name: model_object} for each candidate model.
            val_split: Validation data (adapter-specific type).

        Default: no-op (for proxies that need no calibration).
        """

    @abstractmethod
    def score(self, inference_record: dict) -> float:
        """Compute proxy accuracy A_i from one inference interval's records.

        Args:
            inference_record: Dict with at minimum:
              "confidences": list[float]  — per-detection confidence scores
              "model": str                — name of the model that produced them
              (optionally): "image_paths", "predictions", "embeddings"

        Returns:
            Proxy score in [0, 1]. Higher means better estimated accuracy.
        """

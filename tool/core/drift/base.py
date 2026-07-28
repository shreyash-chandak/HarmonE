"""
core/drift/base.py — Abstract base class for all drift detectors.

Phase 2 version added detect(). Phase 3 extends it with fit_reference / score /
save / load so detectors can be used generically by the experiment harness.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any


class DriftDetector(ABC):
    """All drift detectors must implement this interface."""

    name: str  # set on each subclass

    # ------------------------------------------------------------------
    # Primary interface (Phase 3)
    # ------------------------------------------------------------------

    def fit_reference(self, reference: Any) -> None:
        """Compute and store reference statistics from training data.

        'reference' interpretation is detector-specific:
          - value-histogram detectors: 1-D array of floats
          - embedding detectors: 2-D array (N × D) of feature vectors

        Must be called before score(). Default: no-op (for detectors that
        persist their reference at construction time, e.g. KLFixedRefDetector).
        """

    @abstractmethod
    def score(self, window: Any) -> float | None:
        """Compute a drift score for the given window.

        Args:
            window: Window of current observations (same type as reference).

        Returns:
            Float divergence/distance measure, or None if insufficient data.
        """

    def save(self, path: str) -> None:
        """Persist reference statistics to disk."""

    def load(self, path: str) -> None:
        """Load previously persisted reference statistics."""

    # ------------------------------------------------------------------
    # Legacy interface (Phase 2 — kept for backward compat)
    # ------------------------------------------------------------------

    def detect(self, current_values: list[float]) -> dict[str, Any]:
        """Compute drift signal and apply threshold.

        Wraps score() for detectors that set self.tau_drift. Subclasses that
        do not set tau_drift should override this method directly.

        Returns:
            Dict with at minimum:
              "drift_score": float | None
              "drift_detected": bool
        """
        s = self.score(current_values)
        tau = getattr(self, "tau_drift", None)
        detected = (s is not None and tau is not None and s > tau)
        return {
            "kl_div": s,
            "drift_score": s,
            "drift_detected": detected,
            "detector": getattr(self, "name", type(self).__name__),
        }

"""
core/drift/kl_fixed_ref.py — KL divergence vs. the training reference distribution.

Fixes B3: the paper used rolling (adjacent-window) KL which cannot detect gradual
drift. This module computes KL(current ‖ training_reference) so every cycle is
compared to the same fixed anchor — the distribution seen during training.

The training reference is stored in knowledge/reference_distribution.json as a
histogram over the same bin edges as the current window. Generate it once via
scripts/init_regression.py (regression) or scripts/init_cv.py (CV).
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .base import DriftDetector


_EPSILON = 1e-10  # Laplace smoothing to avoid log(0)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p ‖ q) with Laplace smoothing on both distributions.

    Args:
        p: Current window distribution (unnormalised histogram counts).
        q: Reference distribution (unnormalised histogram counts).

    Returns:
        KL divergence in nats. Returns 0.0 for empty inputs.
    """
    p = np.asarray(p, dtype=float) + _EPSILON
    q = np.asarray(q, dtype=float) + _EPSILON
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


class KLFixedRefDetector(DriftDetector):
    """Detect drift as KL(current window ‖ fixed training reference).

    Args:
        reference_path: Absolute path to reference_distribution.json.
        tau_drift:      Detection threshold. Drift fires when KL score exceeds this.
        window_size:    Number of most-recent values to use per cycle.
        n_bins:         Number of histogram bins. Must match the reference file.
    """

    def __init__(
        self,
        reference_path: str,
        tau_drift: float,
        window_size: int = 1200,
        n_bins: int = 50,
    ) -> None:
        self.reference_path = reference_path
        self.tau_drift = tau_drift
        self.window_size = window_size
        self.n_bins = n_bins
        self._ref_hist: np.ndarray | None = None
        self._bin_edges: np.ndarray | None = None
        self._load_reference()

    def _load_reference(self) -> None:
        if not os.path.exists(self.reference_path):
            return
        with open(self.reference_path, "r") as f:
            data = json.load(f)
        self._ref_hist = np.array(data["histogram"], dtype=float)
        self._bin_edges = np.array(data["bin_edges"], dtype=float)

    def score(self, window: Any) -> float | None:
        """Return KL(window ‖ reference) or None during warmup."""
        if self._ref_hist is None:
            return None
        w = list(window) if not isinstance(window, list) else window
        if len(w) < self.window_size:
            return None
        current_hist, _ = np.histogram(w[-self.window_size:], bins=self._bin_edges)
        return round(_kl_divergence(current_hist, self._ref_hist), 6)

    def detect(self, current_values: list[float]) -> dict[str, Any]:
        """Compute KL(current ‖ reference).

        Returns drift_score=None and drift_detected=False when:
        - Fewer than window_size values are available (warmup period).
        - Reference file does not exist yet (run scripts/init_regression.py or scripts/init_cv.py first).

        Fixes B5: never returns a random placeholder; returns None during warmup.
        """
        if self._ref_hist is None:
            return {"kl_div": None, "drift_detected": False, "detector": "kl_fixed_ref"}

        window = current_values[-self.window_size:]
        if len(window) < self.window_size:
            return {"kl_div": None, "drift_detected": False, "detector": "kl_fixed_ref"}

        current_hist, _ = np.histogram(window, bins=self._bin_edges)
        kl = _kl_divergence(current_hist, self._ref_hist)

        return {
            "kl_div": round(kl, 6),
            "drift_detected": kl > self.tau_drift,
            "detector": "kl_fixed_ref",
        }

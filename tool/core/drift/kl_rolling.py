"""
core/drift/kl_rolling.py — KL divergence between two adjacent sliding windows.

This is the method used in the original Harmonica paper. It detects abrupt local
shifts but misses gradual drift. Kept as secondary telemetry alongside
KLFixedRefDetector (see B3 in CHANGES_FROM_PAPER.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import DriftDetector


_EPSILON = 1e-10


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float) + _EPSILON
    q = np.asarray(q, dtype=float) + _EPSILON
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


class KLRollingDetector(DriftDetector):
    """Detect drift as KL(recent window ‖ preceding window).

    Matches the paper's original detection approach (two adjacent prediction
    windows, each of window_size samples, histogrammed into n_bins bins).

    Args:
        tau_drift:   Detection threshold.
        window_size: Half-window size — total lookback is 2 * window_size.
        n_bins:      Number of histogram bins.
    """

    def __init__(
        self,
        tau_drift: float,
        window_size: int = 1200,
        n_bins: int = 50,
    ) -> None:
        self.tau_drift = tau_drift
        self.window_size = window_size
        self.n_bins = n_bins

    def score(self, window: Any) -> float | None:
        """Return rolling KL or None during warmup."""
        values = list(window) if not isinstance(window, list) else window
        required = 2 * self.window_size
        if len(values) < required:
            return None
        preceding = values[-required:-self.window_size]
        recent = values[-self.window_size:]
        all_vals = list(preceding) + list(recent)
        bins = np.linspace(min(all_vals), max(all_vals), self.n_bins + 1)
        p_hist, _ = np.histogram(recent, bins=bins)
        q_hist, _ = np.histogram(preceding, bins=bins)
        return round(_kl_divergence(p_hist, q_hist), 6)

    def detect(self, current_values: list[float]) -> dict[str, Any]:
        """Compute KL(recent ‖ preceding) from adjacent windows.

        Returns drift_score=None during warmup (fewer than 2 * window_size samples).
        Fixes B5: never returns a random placeholder.
        """
        required = 2 * self.window_size
        if len(current_values) < required:
            return {"kl_div": None, "drift_detected": False, "detector": "kl_rolling"}

        preceding = current_values[-required:-self.window_size]
        recent = current_values[-self.window_size:]

        all_vals = list(preceding) + list(recent)
        bins = np.linspace(min(all_vals), max(all_vals), self.n_bins + 1)

        p_hist, _ = np.histogram(recent, bins=bins)
        q_hist, _ = np.histogram(preceding, bins=bins)
        kl = _kl_divergence(p_hist, q_hist)

        return {
            "kl_div": round(kl, 6),
            "drift_detected": kl > self.tau_drift,
            "detector": "kl_rolling",
        }

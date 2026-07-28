"""
core/drift/luminance_kl.py — KL divergence on luminance histograms (CV legacy detector).

Wraps the existing utility/drift_utils luminance histogram logic in the
DriftDetector interface. Kept as a selectable baseline; embedding-space
detectors (MMD, Fréchet) are the primary signal for Phase 3.

Reference: per-image luminance histogram (Y from Rec.601 RGB, 64 bins).
Current signal: mean of last window_size image histograms.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from scipy.stats import entropy

from .base import DriftDetector


def _luminance_histogram(img_path: str, bins: int = 64) -> np.ndarray | None:
    """Compute normalised luminance histogram for one image."""
    try:
        from PIL import Image
        with Image.open(img_path).convert("RGB") as im:
            arr = np.asarray(im, dtype=np.float32)
    except Exception:
        return None
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    y = np.clip(y, 0, 255)
    hist, _ = np.histogram(y, bins=bins, range=(0, 255))
    hist = hist.astype(np.float64)
    return hist / (hist.sum() + 1e-12)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    return float(entropy(p, q))


class LuminanceKLDetector(DriftDetector):
    """KL divergence between mean luminance histograms of reference and current window.

    Input to score() / detect(): list of either
      - image file paths (str), or
      - pre-computed histogram arrays (np.ndarray of shape (bins,))
    Window mode: uses the last window_size items.

    Args:
        tau_drift:   Detection threshold (from thresholds.json "tau_drift").
        window_size: Number of images per window (default 1000).
        bins:        Luminance histogram bins (default 64; must match reference).
        reference_path: Optional path to a persisted reference JSON.
    """

    name = "luminance_kl"

    def __init__(
        self,
        tau_drift: float = 0.07,
        window_size: int = 1000,
        bins: int = 64,
        reference_path: str | None = None,
    ) -> None:
        self.tau_drift = tau_drift
        self.window_size = window_size
        self.bins = bins
        self._ref_hist: np.ndarray | None = None
        if reference_path and os.path.exists(reference_path):
            self.load(reference_path)

    def fit_reference(self, reference: Any) -> None:
        """Compute mean histogram from a list of image paths or pre-computed histograms."""
        hists = self._collect_hists(reference)
        if hists:
            self._ref_hist = np.mean(hists, axis=0)

    def score(self, window: Any) -> float | None:
        if self._ref_hist is None:
            return None
        items = list(window) if not isinstance(window, list) else window
        items = items[-self.window_size:]
        if len(items) < self.window_size:
            return None
        hists = self._collect_hists(items)
        if not hists:
            return None
        cur_dist = np.mean(hists, axis=0)
        return round(_kl(cur_dist, self._ref_hist), 6)

    def save(self, path: str) -> None:
        if self._ref_hist is None:
            raise RuntimeError("No reference fitted yet; call fit_reference() first.")
        with open(path, "w") as f:
            json.dump({"reference_histogram": self._ref_hist.tolist(), "bins": self.bins}, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self._ref_hist = np.array(data["reference_histogram"])
        self.bins = data.get("bins", self.bins)

    def _collect_hists(self, items: list) -> list[np.ndarray]:
        hists = []
        for item in items:
            if isinstance(item, (str, os.PathLike)):
                h = _luminance_histogram(str(item), self.bins)
                if h is not None:
                    hists.append(h)
            elif isinstance(item, np.ndarray) and item.ndim == 1:
                hists.append(item / (item.sum() + 1e-12))
        return hists

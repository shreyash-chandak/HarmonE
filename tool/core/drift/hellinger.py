"""
core/drift/hellinger.py — Hellinger distance vs. a fixed training-reference
histogram (Augur-inspired detector, Lewis et al. 2022, SE4RAI'22 Section 2.3).

Augur's evaluation (their Table 2 / Section 5) found Hellinger Distance among
the fastest metrics to detect drift (comparable to KL Divergence and Energy
Distance), but — unlike Energy Distance — it decreases again as the
post-training interval grows, because it responds to instantaneous
distributional distance rather than accumulating evidence over time. This
module is offered as an alternative/comparison signal alongside the existing
kl_fixed_ref detector; it is NOT wired as anyone's default (see
context/idea.md for the full discussion of which datasets benefit from
comparing detectors side by side).

Same fixed-reference histogram construction as core/drift/kl_fixed_ref.py
(reference_distribution.json, same bin edges) — the two are directly
comparable on the same data with no extra setup.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .base import DriftDetector

_EPSILON = 1e-10


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two (unnormalised histogram) count arrays.

    H(p, q) = (1/sqrt(2)) * sqrt(sum((sqrt(p_i) - sqrt(q_i))^2))

    Bounded in [0, 1]. 0 = identical distributions, 1 = disjoint support.
    """
    p = np.asarray(p, dtype=float) + _EPSILON
    q = np.asarray(q, dtype=float) + _EPSILON
    p /= p.sum()
    q /= q.sum()
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2.0))


class HellingerFixedRefDetector(DriftDetector):
    """Detect drift as Hellinger(current window, fixed training reference).

    Same reference_distribution.json format as KLFixedRefDetector, so both
    can be run against the same reference file for direct comparison.
    """

    name = "hellinger_fixed_ref"

    def __init__(
        self,
        reference_path: str | None = None,
        tau_drift: float = 0.3,
        window_size: int = 1200,
        n_bins: int = 50,
    ) -> None:
        self.reference_path = reference_path
        self.tau_drift = tau_drift
        self.window_size = window_size
        self.n_bins = n_bins
        self._ref_hist: np.ndarray | None = None
        self._bin_edges: np.ndarray | None = None
        if reference_path is not None:
            self._load_reference()

    def _load_reference(self) -> None:
        if not self.reference_path or not os.path.exists(self.reference_path):
            return
        with open(self.reference_path, "r") as f:
            data = json.load(f)
        self._ref_hist = np.array(data["histogram"], dtype=float)
        self._bin_edges = np.array(data["bin_edges"], dtype=float)

    def fit_reference(self, reference: Any) -> None:
        """Fit on a 1-D array of raw training values (builds the histogram
        directly — an alternative to loading reference_distribution.json)."""
        ref = np.asarray(reference, dtype=float).flatten()
        hist, edges = np.histogram(ref, bins=self.n_bins)
        self._ref_hist = hist.astype(float)
        self._bin_edges = edges

    def score(self, window: Any) -> float | None:
        if self._ref_hist is None or self._bin_edges is None:
            return None
        w = list(window) if not isinstance(window, list) else window
        if len(w) < self.window_size:
            return None
        current_hist, _ = np.histogram(w[-self.window_size:], bins=self._bin_edges)
        return round(hellinger_distance(current_hist, self._ref_hist), 6)

    def save(self, path: str) -> None:
        if self._ref_hist is None or self._bin_edges is None:
            raise RuntimeError("No reference fitted; call fit_reference() first.")
        np.savez_compressed(path, histogram=self._ref_hist, bin_edges=self._bin_edges)

    def load(self, path: str) -> None:
        data = np.load(path)
        self._ref_hist = data["histogram"]
        self._bin_edges = data["bin_edges"]

"""
core/drift/energy_distance.py — Energy Distance vs. a fixed reference sample
(Augur-inspired detector, Lewis et al. 2022, SE4RAI'22 Section 2.3).

Of the six metrics Augur evaluated, Energy Distance was singled out (their
Section 5 / Figure 6) as the one that increases MONOTONICALLY with the
post-training interval — every other tested metric (including KL Divergence
and Hellinger Distance) spikes on drift onset and then decays back toward
zero as prediction uncertainty grows, which can look like "drift resolving
itself" even when it has not. Energy Distance instead accumulates evidence
over time, which Augur's authors recommend pairing with a fast-responding
metric (KL Divergence) for an "immediate signal + accumulating evidence"
ensemble — see context/idea.md for how this maps onto our kl_fixed_ref +
energy_distance pairing.

Reference: Székely & Rizzo (2013), "Energy statistics: A class of statistics
based on distances." Journal of Statistical Planning and Inference.

Unlike the histogram-based detectors (kl_fixed_ref, hellinger), Energy
Distance operates on the RAW SAMPLE, not a binned histogram — it needs a
reference sample of individual values, not counts. Both reference and window
are subsampled (capped, default 500) to keep the O(n*m) pairwise cost bounded,
mirroring core/drift/mmd_embedding.py's _median_bandwidth subsampling pattern.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import DriftDetector


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Energy distance between two 1-D samples.

    E(X, Y) = 2*E|X-Y| - E|X-X'| - E|Y-Y'|

    0 when X and Y are drawn from the same distribution (in expectation);
    grows with the separation between the two distributions. Unlike KL or
    Hellinger this is a true metric (satisfies the triangle inequality) and
    is well-defined even for disjoint supports.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if len(x) < 2 or len(y) < 2:
        return 0.0

    d_xy = np.abs(x - y.T).mean()
    d_xx = np.abs(x - x.T).mean()
    d_yy = np.abs(y - y.T).mean()

    return float(2.0 * d_xy - d_xx - d_yy)


class EnergyDistanceDetector(DriftDetector):
    """Detect drift as EnergyDistance(current window sample, reference sample).

    Args:
        tau_drift:       Detection threshold.
        window_size:      Number of most-recent raw values per detection cycle.
        reference_size:   Max reference sample size retained (subsampled if larger).
        window_subsample: Max window sample size used per score() call.
    """

    name = "energy_distance"

    def __init__(
        self,
        tau_drift: float = 0.05,
        window_size: int = 1200,
        reference_size: int = 500,
        window_subsample: int = 500,
        seed: int = 0,
    ) -> None:
        self.tau_drift = tau_drift
        self.window_size = window_size
        self.reference_size = reference_size
        self.window_subsample = window_subsample
        self._rng = np.random.default_rng(seed)
        self._ref_sample: np.ndarray | None = None

    def fit_reference(self, reference: Any) -> None:
        ref = np.asarray(reference, dtype=float).flatten()
        if len(ref) > self.reference_size:
            idx = self._rng.choice(len(ref), self.reference_size, replace=False)
            ref = ref[idx]
        self._ref_sample = ref

    def score(self, window: Any) -> float | None:
        if self._ref_sample is None:
            return None
        w = np.asarray(window, dtype=float).flatten()
        if len(w) < self.window_size:
            return None
        w = w[-self.window_size:]
        if len(w) > self.window_subsample:
            idx = self._rng.choice(len(w), self.window_subsample, replace=False)
            w = w[idx]
        return round(energy_distance(w, self._ref_sample), 8)

    def save(self, path: str) -> None:
        if self._ref_sample is None:
            raise RuntimeError("No reference fitted; call fit_reference() first.")
        np.savez_compressed(path, ref_sample=self._ref_sample)

    def load(self, path: str) -> None:
        data = np.load(path)
        self._ref_sample = data["ref_sample"]

"""
core/drift/mmd_embedding.py — Unbiased MMD² drift detector on embedding vectors.

MMD (Maximum Mean Discrepancy) with an RBF kernel is a principled two-sample
test for detecting distributional shift in high-dimensional feature spaces.
This makes it suitable for CV embedding drift where luminance histograms miss
semantic changes (same-brightness objects with different class distributions).

Bandwidth σ is set via the median heuristic at fit_reference() time and frozen
for the duration of the experiment (re-estimating at runtime would allow the
reference bandwidth to "track" the drift, defeating the purpose).

Reference: Gretton et al. (2012) "A Kernel Two-Sample Test", JMLR.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .base import DriftDetector


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """RBF kernel matrix K(X, Y) with bandwidth σ."""
    diff = X[:, None, :] - Y[None, :, :]           # (N, M, D)
    sq_dist = np.sum(diff ** 2, axis=-1)             # (N, M)
    return np.exp(-sq_dist / (2.0 * sigma ** 2))


def _median_bandwidth(X: np.ndarray, subsample: int = 500) -> float:
    """Estimate σ via the median pairwise distance heuristic.

    Subsamples up to `subsample` rows to keep O(n²) cost manageable.
    """
    if len(X) > subsample:
        idx = np.random.default_rng(0).choice(len(X), subsample, replace=False)
        X = X[idx]
    diff = X[:, None, :] - X[None, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    median_sq = float(np.median(sq_dists[sq_dists > 0]))
    return max(float(np.sqrt(median_sq / 2.0)), 1e-6)


def unbiased_mmd2(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased MMD² estimator.

    MMD²(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    with the off-diagonal unbiased variant (excludes i=j terms).
    Returns a float that is positive (in expectation) when P ≠ Q.
    """
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return 0.0

    Kxx = _rbf_kernel(X, X, sigma)
    Kyy = _rbf_kernel(Y, Y, sigma)
    Kxy = _rbf_kernel(X, Y, sigma)

    # Unbiased: zero out diagonal for within-set terms
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term_xx = Kxx.sum() / (n * (n - 1))
    term_yy = Kyy.sum() / (m * (m - 1))
    term_xy = Kxy.mean()

    return float(term_xx + term_yy - 2.0 * term_xy)


class MMDEmbeddingDetector(DriftDetector):
    """MMD² drift detector on D-dimensional embedding vectors.

    Input to score(): 2-D array of shape (N_current, D).

    Args:
        tau_drift:     Detection threshold (from thresholds.json "tau_drift").
        window_size:   Number of most-recent embeddings per detection cycle.
        reference_size: Number of reference embeddings to retain.
        bandwidth:     RBF σ. If None, estimated from reference via median heuristic.
    """

    name = "mmd_embedding"

    def __init__(
        self,
        tau_drift: float = 0.01,
        window_size: int = 500,
        reference_size: int = 500,
        bandwidth: float | None = None,
    ) -> None:
        self.tau_drift = tau_drift
        self.window_size = window_size
        self.reference_size = reference_size
        self._bandwidth: float | None = bandwidth
        self._ref_embeddings: np.ndarray | None = None

    def fit_reference(self, reference: Any) -> None:
        """Fit on a 2-D array of training embeddings (N, D).

        Computes and freezes the median-heuristic bandwidth.
        Stores the first reference_size rows as the persistent reference set.
        """
        ref = np.asarray(reference, dtype=np.float32)
        if ref.ndim != 2:
            raise ValueError(f"reference must be 2-D (N, D), got shape {ref.shape}")
        self._ref_embeddings = ref[: self.reference_size]
        if self._bandwidth is None:
            self._bandwidth = _median_bandwidth(self._ref_embeddings)

    def score(self, window: Any) -> float | None:
        """Compute unbiased MMD² between reference and current embedding window.

        Args:
            window: 2-D array (N_current, D) of current embeddings.

        Returns:
            MMD² float, or None if reference not fitted or window too small.
        """
        if self._ref_embeddings is None or self._bandwidth is None:
            return None
        cur = np.asarray(window, dtype=np.float32)
        if cur.ndim != 2 or len(cur) < self.window_size:
            return None
        cur = cur[-self.window_size:]
        return round(unbiased_mmd2(self._ref_embeddings, cur, self._bandwidth), 8)

    def save(self, path: str) -> None:
        if self._ref_embeddings is None:
            raise RuntimeError("No reference fitted; call fit_reference() first.")
        np.savez_compressed(
            path,
            ref_embeddings=self._ref_embeddings,
            bandwidth=np.array([self._bandwidth]),
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        self._ref_embeddings = data["ref_embeddings"]
        self._bandwidth = float(data["bandwidth"][0])

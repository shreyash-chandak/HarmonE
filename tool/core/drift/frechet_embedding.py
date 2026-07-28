"""
core/drift/frechet_embedding.py — Fréchet distance drift detector on embeddings.

Cheaper than MMD (O(D²) vs O(N²) kernel matrix), FID-style: computes mean and
covariance of the reference and current embedding distributions, then returns the
Fréchet/Wasserstein-2 distance between the two Gaussians.

Suitable for large embedding dimensions where MMD kernel computation is slow.
Select via thresholds.json "drift_detectors": ["frechet_embedding"].

Reference: Heusel et al. (2017) "GANs Trained by a Two Time-Scale Update Rule"
(the FID metric). Here applied to detection backbone embeddings, not GAN outputs.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .base import DriftDetector


def _sqrtm_symmetric(A: np.ndarray) -> np.ndarray:
    """Square root of a symmetric positive semi-definite matrix via eigendecomposition."""
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 0.0)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
) -> float:
    """Fréchet distance between N(mu1,sigma1) and N(mu2,sigma2).

    FD = ||mu1 - mu2||² + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))
    """
    diff = mu1 - mu2
    mean_term = float(np.dot(diff, diff))

    # sqrt(sigma1 @ sigma2)  — approximate via symmetric sqrtm
    product = sigma1 @ sigma2
    sqrt_product = _sqrtm_symmetric(product)
    # Tr(sigma1) + Tr(sigma2) - 2*Tr(sqrt(sigma1@sigma2))
    cov_term = float(np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(sqrt_product))

    return max(0.0, mean_term + cov_term)


class FrechetEmbeddingDetector(DriftDetector):
    """Fréchet distance between reference and current embedding Gaussians.

    Input to score(): 2-D array of shape (N_current, D).

    Args:
        tau_drift:    Detection threshold.
        window_size:  Number of most-recent embeddings per detection cycle.
    """

    name = "frechet_embedding"

    def __init__(self, tau_drift: float = 0.5, window_size: int = 500) -> None:
        self.tau_drift = tau_drift
        self.window_size = window_size
        self._ref_mu: np.ndarray | None = None
        self._ref_sigma: np.ndarray | None = None

    def fit_reference(self, reference: Any) -> None:
        """Compute and store mean + covariance from training embeddings (N, D)."""
        ref = np.asarray(reference, dtype=np.float64)
        if ref.ndim != 2:
            raise ValueError(f"reference must be 2-D (N, D), got {ref.shape}")
        self._ref_mu = np.mean(ref, axis=0)
        self._ref_sigma = np.cov(ref, rowvar=False) if len(ref) > 1 else np.zeros((ref.shape[1], ref.shape[1]))

    def score(self, window: Any) -> float | None:
        if self._ref_mu is None:
            return None
        cur = np.asarray(window, dtype=np.float64)
        if cur.ndim != 2 or len(cur) < self.window_size:
            return None
        cur = cur[-self.window_size:]
        cur_mu = np.mean(cur, axis=0)
        cur_sigma = np.cov(cur, rowvar=False) if len(cur) > 1 else np.zeros_like(self._ref_sigma)
        return round(frechet_distance(self._ref_mu, self._ref_sigma, cur_mu, cur_sigma), 6)

    def save(self, path: str) -> None:
        if self._ref_mu is None:
            raise RuntimeError("No reference fitted; call fit_reference() first.")
        np.savez_compressed(path, ref_mu=self._ref_mu, ref_sigma=self._ref_sigma)

    def load(self, path: str) -> None:
        data = np.load(path)
        self._ref_mu = data["ref_mu"]
        self._ref_sigma = data["ref_sigma"]

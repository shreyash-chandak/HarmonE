"""
tests/test_phase3_drift.py — Tests for Phase 3 drift detector implementations.

Tests cover:
  - DriftDetector ABC cannot be instantiated directly
  - MMDEmbeddingDetector: score() returns None during warmup; detects distribution shift
  - FrechetEmbeddingDetector: warmup returns None; detects Gaussian mean shift
  - LuminanceKLDetector: fit_reference + score, save/load round-trip
  - unbiased_mmd2(): zero for identical distributions, positive for shifted
  - frechet_distance(): zero for identical Gaussians, positive for shifted
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.drift.base import DriftDetector
from core.drift.mmd_embedding import MMDEmbeddingDetector, unbiased_mmd2
from core.drift.frechet_embedding import FrechetEmbeddingDetector, frechet_distance
from core.drift.luminance_kl import LuminanceKLDetector


# ── DriftDetector ABC ──────────────────────────────────────────────────────────

class TestDriftDetectorABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            DriftDetector()

    def test_detect_wraps_score(self):
        """detect() should use score() and compare to tau_drift."""

        class FakeDet(DriftDetector):
            name = "fake"
            tau_drift = 0.5

            def score(self, window):
                return 0.9  # above threshold

        det = FakeDet()
        result = det.detect([])
        assert result["drift_detected"] is True
        assert result["kl_div"] == pytest.approx(0.9)

    def test_detect_no_drift_when_below_threshold(self):
        class FakeDet(DriftDetector):
            name = "fake"
            tau_drift = 0.5

            def score(self, window):
                return 0.1  # below threshold

        det = FakeDet()
        result = det.detect([])
        assert result["drift_detected"] is False

    def test_detect_returns_none_score_as_no_drift(self):
        class FakeDet(DriftDetector):
            name = "fake"
            tau_drift = 0.5

            def score(self, window):
                return None  # warmup

        det = FakeDet()
        result = det.detect([])
        assert result["drift_detected"] is False
        assert result["kl_div"] is None


# ── unbiased_mmd2 ──────────────────────────────────────────────────────────────

class TestUnbiasedMMD2:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 2))
        # Use a sigma appropriate for 2-D unit-normal data
        from core.drift.mmd_embedding import _median_bandwidth
        sigma = _median_bandwidth(X)
        mmd = unbiased_mmd2(X, X.copy(), sigma=sigma)
        assert abs(mmd) < 0.1, f"Expected ~0 for identical distributions, got {mmd}"

    def test_shifted_distributions_positive(self):
        rng = np.random.default_rng(42)
        # Use 2-D data so median bandwidth is meaningful
        X = rng.normal(0, 1, (200, 2))
        Y = rng.normal(5, 1, (200, 2))  # large mean shift
        from core.drift.mmd_embedding import _median_bandwidth
        sigma = _median_bandwidth(np.vstack([X, Y]))
        mmd = unbiased_mmd2(X, Y, sigma=sigma)
        assert mmd > 0.1, f"Expected large MMD² for shifted dists, got {mmd}"

    def test_requires_more_than_one_sample(self):
        X = np.array([[1.0, 2.0]])
        Y = np.array([[3.0, 4.0]])
        # Single-sample inputs return 0.0 gracefully (n < 2 guard)
        mmd = unbiased_mmd2(X, Y, sigma=1.0)
        assert isinstance(mmd, float)
        assert mmd == 0.0


# ── MMDEmbeddingDetector ───────────────────────────────────────────────────────

class TestMMDEmbeddingDetector:
    def test_warmup_returns_none(self):
        # window_size=100 means score() returns None for inputs with < 100 rows
        det = MMDEmbeddingDetector(tau_drift=0.1, window_size=100)
        rng = np.random.default_rng(0)
        det.fit_reference(rng.normal(0, 1, (200, 8)))
        score = det.score(rng.normal(0, 1, (20, 8)))  # 20 < window_size=100
        assert score is None

    def test_no_drift_on_same_distribution(self):
        rng = np.random.default_rng(1)
        det = MMDEmbeddingDetector(tau_drift=0.5, window_size=100, reference_size=200)
        ref = rng.normal(0, 1, (200, 2))
        det.fit_reference(ref)
        current = rng.normal(0, 1, (100, 2))
        score = det.score(current)
        assert score is not None
        assert score < 0.3, f"No drift expected, got score={score}"

    def test_detects_distribution_shift(self):
        rng = np.random.default_rng(2)
        det = MMDEmbeddingDetector(tau_drift=0.1, window_size=100, reference_size=200)
        ref = rng.normal(0, 1, (200, 2))
        det.fit_reference(ref)
        shifted = rng.normal(5, 1, (100, 2))  # large shift
        score = det.score(shifted)
        assert score is not None
        assert score > det.tau_drift, f"Drift expected, got score={score}"

    def test_save_load_preserves_bandwidth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.default_rng(3)
            det = MMDEmbeddingDetector(tau_drift=0.1, window_size=100, reference_size=200)
            ref = rng.normal(0, 1, (200, 4))
            det.fit_reference(ref)
            sigma_before = det._bandwidth

            path = os.path.join(tmpdir, "mmd_state.npz")
            det.save(path)

            det2 = MMDEmbeddingDetector(tau_drift=0.1)
            det2.load(path)
            assert det2._bandwidth == pytest.approx(sigma_before, rel=1e-5)

    def test_name_attribute(self):
        assert MMDEmbeddingDetector.name == "mmd_embedding"


# ── frechet_distance ───────────────────────────────────────────────────────────

class TestFrechetDistance:
    def test_identical_gaussians_returns_zero(self):
        mu = np.array([1.0, 2.0])
        sigma = np.eye(2)
        dist = frechet_distance(mu, sigma, mu, sigma)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_shifted_mean_increases_distance(self):
        mu1 = np.array([0.0, 0.0])
        mu2 = np.array([3.0, 4.0])
        sigma = np.eye(2)
        dist = frechet_distance(mu1, sigma, mu2, sigma)
        # ||mu1 - mu2||^2 = 9 + 16 = 25; covariance term = 0 since identical
        assert dist == pytest.approx(25.0, abs=1e-4)


# ── FrechetEmbeddingDetector ───────────────────────────────────────────────────

class TestFrechetEmbeddingDetector:
    def test_warmup_returns_none(self):
        # score() returns None when len(window) < window_size
        det = FrechetEmbeddingDetector(tau_drift=0.5, window_size=100)
        rng = np.random.default_rng(10)
        det.fit_reference(rng.normal(0, 1, (200, 8)))
        assert det.score(rng.normal(0, 1, (20, 8))) is None  # 20 < window_size=100

    def test_no_drift_same_distribution(self):
        rng = np.random.default_rng(11)
        det = FrechetEmbeddingDetector(tau_drift=100.0, window_size=100)
        ref = rng.normal(0, 1, (200, 8))
        det.fit_reference(ref)
        current = rng.normal(0, 1, (100, 8))
        score = det.score(current)
        assert score is not None
        assert score < 50.0, f"Low drift expected for same dist, got {score}"

    def test_detects_large_mean_shift(self):
        rng = np.random.default_rng(12)
        det = FrechetEmbeddingDetector(tau_drift=50.0, window_size=100)
        ref = rng.normal(0, 1, (200, 8))
        det.fit_reference(ref)
        shifted = rng.normal(10, 1, (100, 8))  # large shift → high Fréchet
        score = det.score(shifted)
        assert score is not None
        assert score > 50.0, f"Large drift expected, got score={score}"

    def test_name_attribute(self):
        assert FrechetEmbeddingDetector.name == "frechet_embedding"


# ── LuminanceKLDetector ────────────────────────────────────────────────────────

class TestLuminanceKLDetector:
    def _make_histograms(self, n: int, bias: float = 0.0) -> list[np.ndarray]:
        rng = np.random.default_rng(99)
        hists = []
        for _ in range(n):
            h = rng.dirichlet(np.ones(256) * 0.1 + bias) * 1000
            hists.append(h)
        return hists

    def test_warmup_no_ref_returns_none(self):
        det = LuminanceKLDetector(tau_drift=0.1)
        hists = self._make_histograms(10)
        assert det.score(hists) is None

    def test_same_distribution_low_kl(self):
        rng = np.random.default_rng(7)
        ref_hists = [rng.dirichlet(np.ones(256)) * 1000 for _ in range(100)]
        # window_size=50 so score() accepts 50-item lists
        det = LuminanceKLDetector(tau_drift=0.5, window_size=50)
        det.fit_reference(ref_hists)

        cur_hists = [rng.dirichlet(np.ones(256)) * 1000 for _ in range(50)]
        score = det.score(cur_hists)
        assert score is not None
        assert score < 0.5, f"Low KL expected for similar distributions, got {score}"

    def test_different_distribution_higher_kl(self):
        rng = np.random.default_rng(8)
        # Reference: uniform-ish
        ref_hists = [rng.dirichlet(np.ones(256)) * 1000 for _ in range(100)]
        det = LuminanceKLDetector(tau_drift=0.5, window_size=50)
        det.fit_reference(ref_hists)

        # Current: heavily skewed toward low values (dark images)
        alpha = np.ones(256) * 0.01
        alpha[:20] = 10.0
        cur_hists = [rng.dirichlet(alpha) * 1000 for _ in range(50)]
        score = det.score(cur_hists)
        assert score is not None
        assert score > 0.1, f"Higher KL expected for dark images, got {score}"

    def test_save_load_roundtrip(self):
        rng = np.random.default_rng(9)
        ref_hists = [rng.dirichlet(np.ones(256)) * 1000 for _ in range(50)]
        det = LuminanceKLDetector(tau_drift=0.3)
        det.fit_reference(ref_hists)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lum_kl.json")
            det.save(path)
            det2 = LuminanceKLDetector(tau_drift=0.3)
            det2.load(path)

        cur_hists = [rng.dirichlet(np.ones(256)) * 1000 for _ in range(20)]
        s1 = det.score(cur_hists)
        s2 = det2.score(cur_hists)
        assert s1 == pytest.approx(s2, rel=1e-5)

    def test_name_attribute(self):
        assert LuminanceKLDetector.name == "luminance_kl"

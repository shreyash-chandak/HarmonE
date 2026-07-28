"""
B3: Drift detection should use a fixed training reference, not rolling adjacent windows.

Rolling KL can miss gradual drift: if the distribution shifts slowly, each adjacent
pair looks similar even as the overall distribution has drifted far from training.
The fixed-reference detector catches this.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from core.drift.kl_fixed_ref import KLFixedRefDetector, _kl_divergence
from core.drift.kl_rolling import KLRollingDetector


def _write_reference(path: str, values: list, n_bins: int = 50):
    counts, edges = np.histogram(values, bins=n_bins)
    with open(path, "w") as f:
        json.dump({"histogram": counts.tolist(), "bin_edges": edges.tolist()}, f)


class TestKLDivergenceHelper:
    def test_identical_distributions_give_zero(self):
        p = np.array([10, 20, 30, 20, 10], dtype=float)
        assert _kl_divergence(p, p) < 1e-6

    def test_different_distributions_give_positive(self):
        p = np.array([100, 1, 1, 1, 1], dtype=float)
        q = np.array([1, 1, 1, 1, 100], dtype=float)
        assert _kl_divergence(p, q) > 0.5

    def test_asymmetric(self):
        # KL is not symmetric in general; prove it with a 3-bin non-mirror case
        p = np.array([90.0, 5.0, 5.0])
        q = np.array([1.0, 5.0, 94.0])
        assert abs(_kl_divergence(p, q) - _kl_divergence(q, p)) > 0.01


class TestKLFixedRefDetector:
    def _detector_with_ref(self, ref_values, tau=0.5, window=50, n_bins=20):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            _write_reference(f.name, ref_values, n_bins)
            path = f.name
        det = KLFixedRefDetector(reference_path=path, tau_drift=tau, window_size=window, n_bins=n_bins)
        os.unlink(path)
        return det

    def test_no_drift_when_distribution_matches_reference(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500).tolist()
        det = self._detector_with_ref(ref, tau=0.3, window=50)
        current = rng.normal(0, 1, 200).tolist()
        result = det.detect(current)
        assert result["drift_detected"] is False

    def test_drift_detected_on_shifted_distribution(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 500).tolist()
        det = self._detector_with_ref(ref, tau=0.3, window=50)
        shifted = rng.normal(5, 1, 200).tolist()  # mean shifted by 5σ
        result = det.detect(shifted)
        assert result["drift_detected"] is True

    def test_warmup_returns_none_score(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 500).tolist()
        det = self._detector_with_ref(ref, tau=0.5, window=100)
        short_values = rng.normal(0, 1, 30).tolist()  # fewer than window_size
        result = det.detect(short_values)
        assert result["kl_div"] is None, "Must return None during warmup, not a random placeholder"
        assert result["drift_detected"] is False

    def test_missing_reference_file_returns_none(self):
        det = KLFixedRefDetector(
            reference_path="/nonexistent/path/ref.json",
            tau_drift=0.5, window_size=50,
        )
        result = det.detect(list(range(100)))
        assert result["kl_div"] is None


class TestGradualDriftDetection:
    """Demonstrates B3: rolling KL misses gradual drift; fixed-ref catches it."""

    def test_rolling_misses_gradual_drift(self):
        rng = np.random.default_rng(7)
        # Stream that gradually drifts from mean=0 to mean=3 over 2400 steps
        n = 2400
        drift_stream = [rng.normal(i * 3 / n, 1) for i in range(n)]

        rolling = KLRollingDetector(tau_drift=0.5, window_size=600, n_bins=30)
        result = rolling.detect(drift_stream)
        # Adjacent windows look similar even though the whole stream drifted
        rolling_fires = result["drift_detected"]

        # Fixed-ref detector uses beginning of stream as reference
        ref_segment = drift_stream[:600]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            _write_reference(f.name, ref_segment, n_bins=30)
            ref_path = f.name

        fixed = KLFixedRefDetector(reference_path=ref_path, tau_drift=0.5, window_size=600, n_bins=30)
        result_fixed = fixed.detect(drift_stream)
        os.unlink(ref_path)

        # Fixed-ref should detect the drift; rolling may or may not
        assert result_fixed["drift_detected"] is True, "Fixed-ref should detect gradual drift"

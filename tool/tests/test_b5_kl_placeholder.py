"""
B5: During warmup (fewer than 2400 samples), monitor_drift() returned a random
value in [0.01, 0.15]. The ACP secondary boundary was 0.10, so ~50% of warmup
cycles would generate spurious drift events. Fixed to return None.
"""

import pytest
from core.drift.kl_fixed_ref import KLFixedRefDetector
from core.drift.kl_rolling import KLRollingDetector


ACP_SECONDARY_THRESHOLD = 0.10


class TestWarmupReturnsNone:
    def test_fixed_ref_warmup_returns_none(self):
        det = KLFixedRefDetector(
            reference_path="/does/not/exist.json",
            tau_drift=ACP_SECONDARY_THRESHOLD,
            window_size=1200,
        )
        result = det.detect(list(range(100)))
        assert result["kl_div"] is None, (
            "Warmup kl_div must be None, never a random value that could trigger drift"
        )

    def test_rolling_warmup_returns_none(self):
        det = KLRollingDetector(tau_drift=ACP_SECONDARY_THRESHOLD, window_size=1200)
        short = list(range(500))  # less than 2 * window_size
        result = det.detect(short)
        assert result["kl_div"] is None

    def test_warmup_never_triggers_drift(self):
        det = KLRollingDetector(tau_drift=ACP_SECONDARY_THRESHOLD, window_size=1200)
        for seed in range(20):
            import random
            random.seed(seed)
            values = [random.uniform(0, 1) for _ in range(1000)]
            result = det.detect(values)
            assert result["drift_detected"] is False, (
                f"Warmup must never trigger drift (seed={seed})"
            )

    def test_random_uniform_could_exceed_threshold(self):
        """Confirms the original bug: np.random.uniform(0.01, 0.15) can exceed 0.10."""
        import numpy as np
        rng = np.random.default_rng(999)
        draws = rng.uniform(0.01, 0.15, 1000)
        exceeds = (draws > ACP_SECONDARY_THRESHOLD).sum()
        assert exceeds > 300, (
            f"Old random placeholder exceeded ACP threshold in {exceeds}/1000 draws "
            f"(expected ~500 — confirms the bug was real)"
        )

"""
tests/test_tent_and_augur.py — Tests for the TENT and Augur-inspired additions
(context/idea.md): core/tta/tent.py, core/drift/hellinger.py,
core/drift/energy_distance.py.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.drift.hellinger import HellingerFixedRefDetector, hellinger_distance
from core.drift.energy_distance import EnergyDistanceDetector, energy_distance


# ── Hellinger ────────────────────────────────────────────────────────────────

class TestHellingerDistance:
    def test_identical_distributions_zero(self):
        p = np.array([10.0, 20.0, 30.0, 20.0, 10.0])
        assert hellinger_distance(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_disjoint_distributions_near_one(self):
        p = np.array([100.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 100.0])
        assert hellinger_distance(p, q) == pytest.approx(1.0, abs=1e-3)

    def test_bounded_zero_one(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = rng.uniform(0, 100, size=10)
            q = rng.uniform(0, 100, size=10)
            d = hellinger_distance(p, q)
            assert 0.0 <= d <= 1.0 + 1e-9


class TestHellingerFixedRefDetector:
    def test_warmup_returns_none(self):
        det = HellingerFixedRefDetector(window_size=100)
        det.fit_reference(np.random.default_rng(0).normal(0, 1, 1000))
        assert det.score([1.0, 2.0, 3.0]) is None  # fewer than window_size

    def test_no_reference_returns_none(self):
        det = HellingerFixedRefDetector(window_size=10)
        assert det.score(list(range(50))) is None

    def test_detects_distribution_shift(self):
        rng = np.random.default_rng(1)
        det = HellingerFixedRefDetector(window_size=200, n_bins=30)
        det.fit_reference(rng.normal(0, 1, 2000))

        same = list(rng.normal(0, 1, 200))
        shifted = list(rng.normal(6, 1, 200))

        score_same = det.score(same)
        score_shifted = det.score(shifted)
        assert score_same is not None and score_shifted is not None
        assert score_shifted > score_same

    def test_save_load_roundtrip(self, tmp_path):
        det = HellingerFixedRefDetector(window_size=50, n_bins=20)
        det.fit_reference(np.random.default_rng(0).normal(0, 1, 500))
        path = str(tmp_path / "hellinger_ref.npz")
        det.save(path)

        det2 = HellingerFixedRefDetector(window_size=50, n_bins=20)
        det2.load(path)
        window = list(np.random.default_rng(2).normal(0, 1, 50))
        assert det.score(window) == det2.score(window)


# ── Energy Distance ──────────────────────────────────────────────────────────

class TestEnergyDistance:
    def test_identical_samples_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 300)
        y = rng.normal(0, 1, 300)
        # Same distribution, different draws: should be small but not exactly 0.
        d = energy_distance(x, y)
        assert 0.0 <= d < 0.3

    def test_shifted_samples_larger(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 300)
        y_same = rng.normal(0, 1, 300)
        y_shifted = rng.normal(5, 1, 300)
        assert energy_distance(x, y_shifted) > energy_distance(x, y_same)

    def test_too_few_points_returns_zero(self):
        assert energy_distance([1.0], [2.0]) == 0.0


class TestEnergyDistanceDetector:
    def test_warmup_returns_none(self):
        det = EnergyDistanceDetector(window_size=100)
        det.fit_reference(np.random.default_rng(0).normal(0, 1, 500))
        assert det.score([1.0, 2.0]) is None

    def test_detects_distribution_shift(self):
        rng = np.random.default_rng(1)
        det = EnergyDistanceDetector(window_size=150, reference_size=300)
        det.fit_reference(rng.normal(0, 1, 1000))

        same = list(rng.normal(0, 1, 150))
        shifted = list(rng.normal(4, 1, 150))

        score_same = det.score(same)
        score_shifted = det.score(shifted)
        assert score_same is not None and score_shifted is not None
        assert score_shifted > score_same

    def test_save_load_roundtrip(self, tmp_path):
        det = EnergyDistanceDetector(window_size=50, reference_size=200)
        det.fit_reference(np.random.default_rng(0).normal(0, 1, 300))
        path = str(tmp_path / "energy_ref.npz")
        det.save(path)

        det2 = EnergyDistanceDetector(window_size=50, reference_size=200)
        det2.load(path)
        assert np.array_equal(det._ref_sample, det2._ref_sample)


# ── Tent ─────────────────────────────────────────────────────────────────────

torch = pytest.importorskip("torch")
nn = torch.nn


def _tiny_bn_model():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(8, 5)

        def forward(self, x):
            x = torch.relu(self.bn(self.conv(x)))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    return Tiny()


class TestTentConfigureModel:
    def test_only_bn_affine_params_trainable(self):
        from core.tta.tent import configure_model

        model = configure_model(_tiny_bn_model())
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert trainable == {"bn.weight", "bn.bias"}

    def test_bn_running_stats_disabled(self):
        from core.tta.tent import configure_model

        model = configure_model(_tiny_bn_model())
        assert model.bn.track_running_stats is False
        assert model.bn.running_mean is None
        assert model.bn.running_var is None

    def test_no_batchnorm_raises(self):
        from core.tta.tent import collect_params

        class NoBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        with pytest.raises(ValueError):
            collect_params(NoBN())


class TestTentAdaptation:
    def test_reduces_entropy_over_steps(self):
        from core.tta.tent import configure_model, collect_params, Tent, softmax_entropy

        torch.manual_seed(0)
        model = configure_model(_tiny_bn_model())
        params, names = collect_params(model)
        assert names == ["bn.weight", "bn.bias"]
        optimizer = torch.optim.Adam(params, lr=0.05)
        tented = Tent(model, optimizer, steps=1, episodic=False)

        x = torch.randn(16, 3, 16, 16)
        ent_before = softmax_entropy(model(x)).mean().item()
        for _ in range(15):
            tented(x)
        ent_after = softmax_entropy(model(x)).mean().item()
        assert ent_after < ent_before

    def test_episodic_resets_between_calls(self):
        from core.tta.tent import configure_model, collect_params, Tent

        torch.manual_seed(0)
        model = configure_model(_tiny_bn_model())
        params, _ = collect_params(model)
        optimizer = torch.optim.Adam(params, lr=0.1)
        tented = Tent(model, optimizer, steps=5, episodic=True)

        x1 = torch.randn(8, 3, 16, 16)
        w_after_call1 = model.bn.weight.detach().clone()
        tented(x1)
        w_after_call2_pre_reset = model.bn.weight.detach().clone()
        assert not torch.equal(w_after_call1, w_after_call2_pre_reset)

        # Episodic: a second call must reset to the ORIGINAL snapshot before adapting again,
        # not continue from wherever call 1 left off.
        tented(x1)
        # Manually reset and compare starting point
        tented.reset()
        w_reset = model.bn.weight.detach().clone()
        assert torch.equal(w_reset, w_after_call1)

    def test_raises_on_zero_steps(self):
        from core.tta.tent import configure_model, collect_params, Tent

        model = configure_model(_tiny_bn_model())
        params, _ = collect_params(model)
        optimizer = torch.optim.Adam(params, lr=0.01)
        with pytest.raises(ValueError):
            Tent(model, optimizer, steps=0)

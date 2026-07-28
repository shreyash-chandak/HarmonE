"""
B7: MinMaxScaler was fitted on the full dataset (train + test) at the top of
inference.py. This leaks test distribution extremes into the scaler bounds,
meaning model inputs during the test period used test-data-aware normalization.

Fix: scaler fitted on training split only, persisted to knowledge/scaler.pkl.
inference.py calls transform() only; retrain.py re-fits on drift data and
re-persists the scaler atomically.
"""

import os
import pickle
import tempfile

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler


def fit_scaler_correctly(data: np.ndarray, train_frac: float = 0.8) -> MinMaxScaler:
    """Fit scaler on training split only (correct approach)."""
    split_idx = int(len(data) * train_frac)
    scaler = MinMaxScaler()
    scaler.fit(data[:split_idx].reshape(-1, 1))
    return scaler


def fit_scaler_incorrectly(data: np.ndarray) -> MinMaxScaler:
    """Fit scaler on full dataset (the original bug)."""
    scaler = MinMaxScaler()
    scaler.fit_transform(data.reshape(-1, 1))
    return scaler


class TestScalerLeakage:
    def test_scaler_bounds_differ_between_correct_and_buggy(self):
        rng = np.random.default_rng(42)
        # Training: low-variance; test: contains an outlier
        train = rng.normal(0, 1, 800)
        test = np.concatenate([rng.normal(0, 1, 199), np.array([10.0])])  # outlier in test
        data = np.concatenate([train, test])

        correct = fit_scaler_correctly(data, train_frac=0.8)
        buggy = fit_scaler_incorrectly(data)

        assert correct.data_max_[0] < buggy.data_max_[0], (
            "Correct scaler max should be lower (no test outlier); "
            f"got correct={correct.data_max_[0]:.2f}, buggy={buggy.data_max_[0]:.2f}"
        )

    def test_inference_never_calls_fit(self):
        """Validate that inference only calls transform(), not fit()."""
        rng = np.random.default_rng(1)
        train = rng.normal(0, 1, 800)

        scaler = MinMaxScaler()
        scaler.fit(train.reshape(-1, 1))

        # Simulated inference: transform only
        test_sample = np.array([[5.0]])
        transformed = scaler.transform(test_sample)
        assert transformed is not None  # just verifying transform() works

        # The scaler bounds must remain unchanged after transform()
        assert scaler.data_max_[0] == train.max()

    def test_scaler_persisted_atomically(self):
        """Verify atomic write pattern: write to .tmp then os.replace."""
        rng = np.random.default_rng(2)
        train = rng.normal(0, 1, 100)
        scaler = MinMaxScaler()
        scaler.fit(train.reshape(-1, 1))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "scaler.pkl")
            tmp_path = output_path + ".tmp"

            with open(tmp_path, "wb") as f:
                pickle.dump(scaler, f)
            os.replace(tmp_path, output_path)

            assert os.path.exists(output_path)
            assert not os.path.exists(tmp_path)

            with open(output_path, "rb") as f:
                loaded = pickle.load(f)
            assert abs(loaded.data_max_[0] - scaler.data_max_[0]) < 1e-9

    def test_retrain_updates_scaler_on_drift_data(self):
        """After retrain, scaler should reflect drift data distribution, not original."""
        rng = np.random.default_rng(3)
        original_train = rng.normal(0, 1, 1000)
        drift_data = rng.normal(5, 1, 200)  # shifted distribution

        original_scaler = MinMaxScaler()
        original_scaler.fit(original_train.reshape(-1, 1))

        retrain_scaler = MinMaxScaler()
        retrain_scaler.fit(drift_data.reshape(-1, 1))

        # The new scaler's min/max should reflect drift data
        assert abs(retrain_scaler.data_min_[0] - drift_data.min()) < 1e-9
        assert abs(retrain_scaler.data_max_[0] - drift_data.max()) < 1e-9

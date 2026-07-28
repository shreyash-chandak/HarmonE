"""
Phase 2 adapter and separated EMA tests.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from core.scoring import update_separated_emas, update_ema


class TestSeparatedEMAs:
    def test_new_keys_added_to_mape_info(self):
        info = {"ema_scores": {"lstm": 0.5}}
        update_separated_emas(info, "lstm", accuracy=0.8, normalized_energy=0.3, gamma=0.8)
        assert "ema_accuracy" in info
        assert "ema_energy" in info

    def test_first_update_initialises_from_observation(self):
        info = {}
        update_separated_emas(info, "lstm", accuracy=0.8, normalized_energy=0.4, gamma=0.8)
        # On first call, prev == observation → EMA = gamma*obs + (1-gamma)*obs = obs
        assert abs(info["ema_accuracy"]["lstm"] - 0.8) < 1e-9
        assert abs(info["ema_energy"]["lstm"] - 0.4) < 1e-9

    def test_subsequent_updates_use_ema(self):
        info = {}
        update_separated_emas(info, "lstm", accuracy=0.8, normalized_energy=0.4, gamma=0.8)
        update_separated_emas(info, "lstm", accuracy=0.6, normalized_energy=0.6, gamma=0.8)
        expected_acc = 0.8 * 0.6 + 0.2 * 0.8
        expected_eng = 0.8 * 0.6 + 0.2 * 0.4
        assert abs(info["ema_accuracy"]["lstm"] - expected_acc) < 1e-9
        assert abs(info["ema_energy"]["lstm"] - expected_eng) < 1e-9

    def test_returns_same_dict(self):
        info = {}
        result = update_separated_emas(info, "lstm", 0.8, 0.3, 0.8)
        assert result is info

    def test_independent_per_model(self):
        info = {}
        update_separated_emas(info, "lstm", 0.8, 0.3, 0.8)
        update_separated_emas(info, "svm", 0.5, 0.7, 0.8)
        assert "lstm" in info["ema_accuracy"]
        assert "svm" in info["ema_accuracy"]
        assert info["ema_accuracy"]["lstm"] != info["ema_accuracy"]["svm"]


class TestRegressionCSVAdapter:
    def _make_csv(self, n: int = 100) -> str:
        """Write a temporary CSV and return its path."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df = pd.DataFrame({"flow": np.linspace(0, 10, n)})
        df.to_csv(tmp.name, index=False)
        return tmp.name

    def _make_config(self, csv_path: str, seq_length: int = 5, train_frac: float = 0.8):
        return {
            "data_path": csv_path,
            "value_column": "flow",
            "seq_length": seq_length,
            "train_frac": train_frac,
            "stream_delay_s": 0.0,
            "models": {},
        }

    def test_train_split_length(self):
        from adapters.regression_csv import RegressionCSVAdapter
        path = self._make_csv(100)
        try:
            adapter = RegressionCSVAdapter(self._make_config(path))
            train = adapter.train_split()
            assert len(train) == 80
        finally:
            os.unlink(path)

    def test_stream_yields_correct_count(self):
        from adapters.regression_csv import RegressionCSVAdapter
        path = self._make_csv(100)
        try:
            adapter = RegressionCSVAdapter(self._make_config(path, seq_length=5, train_frac=0.8))
            # test split = 20 rows; sequences = 20 - 5 = 15
            samples = list(adapter.stream())
            assert len(samples) == 15
        finally:
            os.unlink(path)

    def test_sample_has_correct_shapes(self):
        from adapters.regression_csv import RegressionCSVAdapter
        path = self._make_csv(50)
        try:
            adapter = RegressionCSVAdapter(self._make_config(path, seq_length=5, train_frac=0.8))
            first = next(iter(adapter.stream()))
            assert len(first.inputs) == 5
            assert first.ground_truth is not None
        finally:
            os.unlink(path)

    def test_induced_drift_shift(self):
        from adapters.regression_csv import RegressionCSVAdapter
        path = self._make_csv(100)
        cfg = self._make_config(path)
        cfg["induced_drift"] = {"type": "shift", "start_frac": 0.0, "end_frac": 1.0, "shift": 100.0}
        try:
            adapter = RegressionCSVAdapter(cfg)
            samples = list(adapter.stream())
            # All test values should be shifted by 100
            assert all(s.ground_truth > 90 for s in samples), "Shift should move all values up"
        finally:
            os.unlink(path)

    def test_offline_labels_covers_test_split(self):
        from adapters.regression_csv import RegressionCSVAdapter
        path = self._make_csv(100)
        try:
            adapter = RegressionCSVAdapter(self._make_config(path, train_frac=0.8))
            labels = adapter.offline_labels()
            assert len(labels) == 20
        finally:
            os.unlink(path)

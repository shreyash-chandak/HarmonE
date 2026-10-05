"""Model-pool positioning support (2026-10-06): stale-weights guard,
train_window_rows, per-model retrain windows."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.regression_csv import RegressionCSVAdapter  # noqa: E402
from experiments.run_experiment import (  # noqa: E402
    _do_inline_retrain,
    _stale_regression_models,
    _write_weights_meta,
)


def _cfg(tmp_path, alpha=10):
    return {
        "seq_length": 3,
        "train_frac": 0.5,
        "models": {"ridge": {"weights_path": str(tmp_path / "ridge.pkl"),
                             "hyperparams": {"train": {"alpha": alpha}, "retrain": {}}}},
    }


def test_missing_weights_are_stale(tmp_path):
    assert _stale_regression_models(_cfg(tmp_path), np.zeros(100)) == ["ridge"]


def test_weights_without_meta_are_stale(tmp_path):
    (tmp_path / "ridge.pkl").write_bytes(b"x")
    assert _stale_regression_models(_cfg(tmp_path), np.zeros(100)) == ["ridge"]


def test_matching_meta_is_fresh_and_changed_hyperparams_are_stale(tmp_path):
    cfg = _cfg(tmp_path)
    (tmp_path / "ridge.pkl").write_bytes(b"x")
    _write_weights_meta(cfg, "ridge", str(tmp_path / "ridge.pkl"), 100)
    assert _stale_regression_models(cfg, np.zeros(100)) == []
    assert _stale_regression_models(_cfg(tmp_path, alpha=20), np.zeros(100)) == ["ridge"]
    assert _stale_regression_models(cfg, np.zeros(90)) == ["ridge"]  # different training data


def test_train_window_rows_keeps_stream(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"value": np.arange(100, dtype=float)}).to_csv(csv, index=False)
    base = {"data_path": str(csv), "value_column": "value", "train_frac": 0.6, "seq_length": 2}
    full = RegressionCSVAdapter(base)
    win = RegressionCSVAdapter({**base, "train_window_rows": 20})
    assert len(full.train_split()) == 60 and list(win.train_split()) == list(range(40, 60))
    assert [s.ground_truth for s in full.stream()] == [s.ground_truth for s in win.stream()]


def test_retrain_window_rows_used(tmp_path):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import MinMaxScaler
    hist = list(np.sin(np.arange(5000) / 10.0))
    scaler = MinMaxScaler().fit(np.array(hist).reshape(-1, 1))
    seen = {}

    class _Spy(Ridge):
        def fit(self, X, y, sample_weight=None):
            seen["n"] = len(X)
            return super().fit(X, y)

    info = {"type": "sklearn", "model": _Spy(alpha=1.0), "retrain_params": {},
            "retrain_window_rows": 3000, "train_params": {"alpha": 1.0}, "n_train_sequences": 3000}
    assert _do_inline_retrain("ridge", {"ridge": info}, {}, hist, scaler, seq_length=5, drift_window=1200)
    assert seen["n"] == 3000 - 5

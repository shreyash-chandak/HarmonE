"""
adapters/regression_csv.py — Generic sliding-window CSV regression adapter.

Configured entirely by configs/datasets/<name>.json.

Config schema (all keys read from the dataset config JSON):

  Two mutually exclusive data-source modes:

  Mode A — single file with internal split (default):
    data_path         : path to the CSV file (absolute or relative to tool/)
    train_frac        : fraction of rows used for training (default 0.8)
    val_frac          : optional; val_split() returns this slice right after the
                        training rows. It is NOT held out: stream() starts at the
                        end of the training rows, so these rows are also the first
                        stream rows. Do not fit on them without
                        assert_disjoint_from_stream() (2026-10-31).
    train_window_rows : optional; train on only the last N rows before the
                        stream starts (default: all rows before it). The
                        stream is unchanged. Added 2026-10-06 for spot_prices,
                        whose train_frac 0.6 put 15 years (1999-2014) in training.

  Mode B — pre-split files (original HarmonE format):
    train_path        : CSV containing the initial training portion only
    stream_path       : CSV containing the streaming/evaluation portion only
    (train_frac and val_frac are ignored in Mode B)

  Shared keys:
    value_column      : name of the target column (e.g. "flow")
    seq_length        : number of timesteps per input window (e.g. 5)
    stream_delay_s    : not used here; the consumer sleeps between predictions
                        (concurrent_harness/inference.py, run_experiment.py)
    models            : dict of model specs (name → {weights_path, loader, cost_class})
    induced_drift     : optional; {type, start_frac, end_frac, scale, shift}
"""

from __future__ import annotations

import os
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .base import DatasetAdapter, ModelSpec, Sample


class RegressionCSVAdapter(DatasetAdapter):
    """Generic sliding-window regression adapter over a single CSV column."""

    domain = "regression"
    self_labeling = True  # ground truth is always in the CSV

    def __init__(self, config: dict, config_dir: str = ".") -> None:
        self._config = config
        self._config_dir = config_dir
        col = config["value_column"]

        if "train_path" in config and "stream_path" in config:
            # Mode B — pre-split files (original HarmonE format)
            train_path = config["train_path"]
            stream_path = config["stream_path"]
            if not os.path.isabs(train_path):
                train_path = os.path.join(config_dir, train_path)
            if not os.path.isabs(stream_path):
                stream_path = os.path.join(config_dir, stream_path)
            _train = pd.read_csv(train_path)[col].values.astype(float)
            _stream = pd.read_csv(stream_path)[col].values.astype(float)
            self._values = np.concatenate([_train, _stream])
            self._train_end = len(_train)
        else:
            # Mode A — single file with train_frac split
            data_path = config["data_path"]
            if not os.path.isabs(data_path):
                data_path = os.path.join(config_dir, data_path)
            self._values = pd.read_csv(data_path)[col].values.astype(float)
            train_frac = float(config.get("train_frac", 0.8))
            self._train_end = int(len(self._values) * train_frac)

        self._seq_length = int(config.get("seq_length", 5))
        self._induced_drift_cfg = config.get("induced_drift", None)
        self._model_specs = {
            name: ModelSpec(
                name=name,
                weights_path=spec["weights_path"],
                loader=spec["loader"],
                cost_class=spec.get("cost_class", "medium"),
            )
            for name, spec in config.get("models", {}).items()
        }

    @classmethod
    def from_config(cls, config: dict, config_dir: str = ".") -> "RegressionCSVAdapter":
        return cls(config, config_dir)

    def train_split(self) -> np.ndarray:
        """Return raw training values (unscaled). Scaler should be fit on this."""
        window = self._config.get("train_window_rows")
        start = max(0, self._train_end - int(window)) if window else 0
        return self._values[start : self._train_end]

    def stream_row_range(self) -> tuple[int, int]:
        """[start, end) of the stream's source rows in the concatenated values."""
        return self._train_end, len(self._values)

    def val_row_range(self) -> tuple[int, int]:
        """[start, end) of val_split()'s rows; empty when val_frac is 0 or absent."""
        val_frac = float(self._config.get("val_frac", 0.0))
        if val_frac == 0.0:
            return self._train_end, self._train_end
        return self._train_end, int(len(self._values) * (float(self._config.get("train_frac", 0.8)) + val_frac))

    def assert_disjoint_from_stream(self, rows) -> None:
        """Raise AssertionError if any row index is in the stream's source rows.
        For anything that fits on rows and must not see the stream."""
        start, end = self.stream_row_range()
        bad = [int(r) for r in rows if start <= int(r) < end]
        assert not bad, (f"{len(bad)} of these rows are stream rows [{start}, {end}), "
                         f"e.g. {bad[:3]}: fitting on them would see the evaluation stream")

    def val_split(self) -> np.ndarray:
        """The val_frac slice right after the training rows. These rows are also
        the first stream rows (see the module docstring)."""
        val_frac = float(self._config.get("val_frac", 0.0))
        if val_frac == 0.0:
            return self._values[self._train_end : self._train_end]
        val_end = int(len(self._values) * (float(self._config.get("train_frac", 0.8)) + val_frac))
        return self._values[self._train_end : val_end]

    def stream(self) -> Iterator[Sample]:
        """Yield sliding-window samples from the test portion chronologically."""
        test_values = self._apply_induced_drift(self._values[self._train_end :])
        seq = self._seq_length
        for i in range(len(test_values) - seq):
            inputs = test_values[i : i + seq]
            truth = test_values[i + seq]
            # No sleep here (2026-10-18): inference.py also slept stream_delay_s,
            # so every paced run waited twice the configured delay.
            yield Sample(index=i, inputs=inputs, ground_truth=truth)

    def offline_labels(self) -> np.ndarray:
        """Ground-truth values for the full test split."""
        return self._values[self._train_end :]

    def models(self) -> dict[str, ModelSpec]:
        return self._model_specs

    def _apply_induced_drift(self, values: np.ndarray) -> np.ndarray:
        """Apply drift induction transform if configured. No-op by default."""
        cfg = self._induced_drift_cfg
        if cfg is None:
            return values

        n = len(values)
        start = int(cfg.get("start_frac", 0.5) * n)
        end = int(cfg.get("end_frac", 1.0) * n)
        drift_type = cfg.get("type", "shift")

        result = values.copy()
        if drift_type == "shift":
            result[start:end] = result[start:end] + cfg.get("shift", 0.0)
        elif drift_type == "scale":
            result[start:end] = result[start:end] * cfg.get("scale", 1.0)
        elif drift_type == "shift_scale":
            result[start:end] = result[start:end] * cfg.get("scale", 1.0) + cfg.get("shift", 0.0)
        return result

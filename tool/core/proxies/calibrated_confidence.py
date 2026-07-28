"""
core/proxies/calibrated_confidence.py — Temperature-scaled confidence proxy.

Calibrates each model's raw confidence scores using a per-model temperature T
fitted on the validation split. T is the scalar that minimises NLL on the
val set under the softmax-with-temperature distribution.

For Ultralytics YOLO detection, we proxy per-detection calibration via Platt
scaling on confidence-vs-IoU-correctness at IoU 0.5 on the val split, since
raw detection logits before softmax are not cleanly accessible. This fallback
is documented in DECISIONS_PENDING.md DP2.

Config key: "calibration_method": "temperature" | "platt" (default: "platt")
Persisted: knowledge/calibration.json {"model_name": T}
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

from .base import AccuracyProxy


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CalibratedConfidenceProxy(AccuracyProxy):
    """Temperature-scaled (or Platt-scaled) confidence proxy.

    After setup(), score() applies the fitted temperature before averaging.
    When temperature T > 1, the model is over-confident and scores are reduced.
    When T < 1, the model is under-confident and scores are increased.

    Args:
        calibration_path: Path to knowledge/calibration.json.
        method:           "temperature" | "platt" (see DECISIONS_PENDING.md DP2).
    """

    name = "calibrated_confidence"

    def __init__(
        self,
        calibration_path: str = "knowledge/calibration.json",
        method: str = "platt",
    ) -> None:
        self._calibration_path = calibration_path
        self._method = method
        self._temperatures: dict[str, float] = {}  # per-model temperature T
        self._load_calibration()

    def setup(self, models: dict, val_split: Any) -> None:
        """Fit per-model temperature on validation data.

        val_split is expected to provide (confidence, is_correct_at_iou50) pairs
        for the Platt method, or (logit, label) pairs for temperature scaling.
        For now we implement Platt scaling as the default (see DP2).

        If val_split is None or empty, temperatures default to 1.0 (no calibration).
        """
        if val_split is None:
            return
        # Platt scaling: fit logistic regression T on confidence-vs-correctness
        # This requires (confidence, correct_bool) pairs, supplied by the adapter's
        # val_split when self_labeling=True. In label-free deployments, T=1.0.
        try:
            data = list(val_split)  # [(model_name, confidence, correct)]
        except Exception:
            return

        from collections import defaultdict
        per_model: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for record in data:
            m, conf, correct = record["model"], record["confidence"], int(record["correct"])
            per_model[m][0].append(conf)
            per_model[m][1].append(correct)

        for model_name, (confs, labels) in per_model.items():
            if len(confs) < 10:
                self._temperatures[model_name] = 1.0
                continue
            T = self._fit_platt(np.array(confs, dtype=float), np.array(labels, dtype=float))
            self._temperatures[model_name] = T

        self._save_calibration()

    def score(self, inference_record: dict) -> float:
        model = inference_record.get("model", "")
        confs = inference_record.get("confidences", [])
        if not confs:
            return 0.0
        T = self._temperatures.get(model, 1.0)
        calibrated = [_sigmoid((math.log(c / (1 - c + 1e-9))) / T) if 0 < c < 1 else c for c in confs]
        return float(sum(calibrated) / len(calibrated))

    def _fit_platt(self, confs: np.ndarray, labels: np.ndarray) -> float:
        """Fit scalar temperature via gradient descent on NLL (logistic loss)."""
        T = 1.0
        lr = 0.1
        for _ in range(200):
            logit = np.log(confs / (1.0 - confs + 1e-9)) / T
            p = 1.0 / (1.0 + np.exp(-logit))
            nll_grad = np.mean((p - labels) * logit / (-T))
            T = max(0.1, T - lr * nll_grad)
        return float(T)

    def _load_calibration(self) -> None:
        if os.path.exists(self._calibration_path):
            try:
                with open(self._calibration_path) as f:
                    self._temperatures = json.load(f)
            except Exception:
                pass

    def _save_calibration(self) -> None:
        tmp = self._calibration_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._temperatures, f, indent=2)
        os.replace(tmp, self._calibration_path)

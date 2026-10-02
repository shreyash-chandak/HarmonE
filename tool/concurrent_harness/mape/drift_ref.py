"""concurrent_harness/mape/drift_ref.py — per-model drift reference for t2.

Three fixes to the fixed-training-reference KL detector
(core/drift/kl_fixed_ref.py) as used by the concurrent harness (audit E1/E2,
2026-09-28). All three borrow logic from the original HarmonE:

1. Overflow bins (E2). The fixed reference edges span [train_min, train_max];
   np.histogram silently DROPS anything outside them (20% of the pems drift
   stream, 10% of uci). Here the edges get two extra open-ended bins
   (-inf, first_edge) and (last_edge, +inf). The training reference has zero
   mass there by construction, so values that drift past the training range
   now RAISE the KL instead of vanishing. (The original HarmonE never had
   this problem because it binned each window on its own range.)

2. Re-basing after adaptation (E1). The original HarmonE compared the current
   window with the PREVIOUS window (HarmonE/mape/monitor.py::monitor_drift),
   i.e. drift meant "the data changed", not "the data is still far from the
   training set". This repo's B3 fix switched to a fixed training reference,
   so once the stream moved, drift fired on every cycle forever (~850-970
   VMR replaces per run). Here each model starts with the training reference;
   after that model is replaced or retrained, its reference becomes the window
   it was adapted to. Drift then means "different from what the current model
   is fitted to", which keeps B3's ability to catch gradual drift.

3. Cooldown (E1). The original's execute_drift() ends with time.sleep(400):
   no drift checks for a while after acting. Here, after an adaptation of a
   model, drift is suppressed for that model for `cooldown_rows` stream rows
   (row-count, not wall-clock, to match the harness's row-based triggering).

This module has no fcntl dependency so it can be unit-tested on any OS.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from core.drift.kl_fixed_ref import _kl_divergence


class PerModelDriftReference:
    def __init__(
        self,
        reference_path: str,
        tau_drift: float,
        window_size: int,
        cooldown_rows: int,
    ) -> None:
        with open(reference_path) as f:
            data = json.load(f)
        base_edges = np.asarray(data["bin_edges"], dtype=float)
        base_hist = np.asarray(data["histogram"], dtype=float)
        # (-inf, e0), [e0, e1), ..., [e_{n-1}, e_n], (e_n, +inf)
        self.edges = np.concatenate(([-np.inf], base_edges, [np.inf]))
        self._training_hist = np.concatenate(([0.0], base_hist, [0.0]))
        self.base_edges = base_edges
        self.tau_drift = float(tau_drift)
        self.window_size = int(window_size)
        self.cooldown_rows = int(cooldown_rows)
        self._ref: dict[str, np.ndarray] = {}
        self._cooldown_until: dict[str, int] = {}
        self._seen_adaptation: dict[str, int] = {}

    def histogram(self, window) -> np.ndarray:
        hist, _ = np.histogram(np.asarray(window, dtype=float), bins=self.edges)
        return hist.astype(float)

    def reference_for(self, model: str) -> np.ndarray:
        return self._ref.get(model, self._training_hist)

    def note_adaptation(self, model: str, adaptation_step: int, window, current_step: int) -> bool:
        """Re-base `model`'s reference to `window` and start its cooldown, if
        `adaptation_step` is newer than the last one seen. Returns True if a
        re-base happened."""
        if adaptation_step <= self._seen_adaptation.get(model, -1):
            return False
        self._seen_adaptation[model] = adaptation_step
        if len(window) >= self.window_size:
            self._ref[model] = self.histogram(window[-self.window_size:])
        self._cooldown_until[model] = adaptation_step + self.cooldown_rows
        return True

    def in_cooldown(self, model: str, current_step: int) -> bool:
        return current_step < self._cooldown_until.get(model, -1)

    def detect(self, model: str, window, current_step: int) -> dict[str, Any]:
        """Same return contract as KLFixedRefDetector.detect(), plus
        "cooldown": True when drift was suppressed by the cooldown."""
        w = list(window)[-self.window_size:]
        if len(w) < self.window_size:
            return {"kl_div": None, "drift_detected": False, "detector": "kl_per_model_ref"}
        kl = _kl_divergence(self.histogram(w), self.reference_for(model))
        cooldown = self.in_cooldown(model, current_step)
        return {
            "kl_div": round(kl, 6),
            "drift_detected": (kl > self.tau_drift) and not cooldown,
            "detector": "kl_per_model_ref",
            "cooldown": cooldown,
        }


class BoundDetector:
    """Adapter so experiments.run_experiment._analyse_drift() (which calls
    detector.detect(values)) can be reused unchanged with a per-model
    reference: binds the model name and step."""

    def __init__(self, ref: PerModelDriftReference, model: str, current_step: int):
        self._ref = ref
        self._model = model
        self._step = current_step
        self.window_size = ref.window_size

    def detect(self, values) -> dict[str, Any]:
        return self._ref.detect(self._model, values, self._step)

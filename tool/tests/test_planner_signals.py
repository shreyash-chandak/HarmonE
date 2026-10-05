"""Planner input fixes (2026-10-06): audit A1 (accuracy signal), A5 (observed
flag), C1 (batch energy metering), A7 (Pareto explores unknowns)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.planners.base import PlanningContext  # noqa: E402
from core.planners.pareto import ParetoPlanner  # noqa: E402
from core.planners.violation_aware import ViolationAwarePlanner  # noqa: E402
from experiments.run_experiment import (  # noqa: E402
    BatchEnergy,
    _initial_mape_info,
    _monitor_batch,
)

MODELS = ["ridge", "svr", "lstm"]
COST = {"ridge": {"cost_class": "light"}, "svr": {"cost_class": "medium"},
        "lstm": {"cost_class": "heavy"}}


# ── A1: window_r2 accuracy ────────────────────────────────────────────────────

def _calm_batch():
    rng = np.random.default_rng(0)
    y_true = 100.0 + rng.normal(0, 1.0, 50)          # low within-batch variance
    y_pred = y_true + rng.normal(0, 1.0, 50)          # error ~ batch spread
    reference = 100.0 + 50.0 * np.sin(np.linspace(0, 20, 1200))  # long-run spread
    return list(y_true), list(y_pred), list(reference)


def test_batch_r2_collapses_on_calm_window():
    yt, yp, _ = _calm_batch()
    info = _initial_mape_info(MODELS)
    tel = _monitor_batch(yt, yp, [1.0] * 50, "ridge", info, {"accuracy_signal": "batch_r2"})
    assert tel["accuracy"] < 0.2


def test_window_r2_uses_reference_variance():
    yt, yp, ref = _calm_batch()
    info = _initial_mape_info(MODELS)
    tel = _monitor_batch(yt, yp, [1.0] * 50, "ridge", info,
                         {"accuracy_signal": "window_r2"}, reference_values=ref)
    assert tel["accuracy"] > 0.99
    assert tel["r2"] < 0.2  # raw batch R² still reported
    assert abs(info["ema_accuracy"]["ridge"] - tel["accuracy"]) < 1e-6


# ── A5: observed flag ─────────────────────────────────────────────────────────

def test_first_observation_replaces_placeholder():
    info = _initial_mape_info(MODELS)
    assert info["observed"] == {m: False for m in MODELS}
    _monitor_batch([], [], [0.0], "svr", info, {"E_m": 0, "E_M": 1}, accuracy=0.9)
    assert info["observed"]["svr"] is True
    assert info["ema_accuracy"]["svr"] == 0.9  # not 0.8*0.5 + 0.2*0.9
    _monitor_batch([], [], [0.0], "svr", info, {"E_m": 0, "E_M": 1, "gamma": 0.8}, accuracy=0.4)
    # EMA = gamma * new + (1 - gamma) * previous
    assert abs(info["ema_accuracy"]["svr"] - (0.8 * 0.4 + 0.2 * 0.9)) < 1e-9


def _ctx(violation, current, acc, eng, observed, **thr):
    return PlanningContext(
        violation=violation, ema_scores=dict(acc), ema_accuracy=acc, ema_energy=eng,
        current_model=current, available_models=MODELS,
        thresholds={"min_accuracy": 0.7, "models": COST, **thr},
        drift_result=None, observed=observed,
    )


def test_s5_energy_violation_tries_cheaper_unobserved_model():
    ctx = _ctx("energy", "lstm",
               acc={"ridge": 0.5, "svr": 0.5, "lstm": 0.99},
               eng={"ridge": 0.5, "svr": 0.5, "lstm": 0.9},
               observed={"ridge": False, "svr": False, "lstm": True})
    d = ViolationAwarePlanner().plan(ctx)
    assert d.action == "switch" and d.model == "ridge"


def test_s5_ignores_placeholder_estimates():
    # svr observed and accurate+cheaper; ridge unobserved with a 0.5 placeholder
    ctx = _ctx("energy", "lstm",
               acc={"ridge": 0.5, "svr": 0.85, "lstm": 0.99},
               eng={"ridge": 0.5, "svr": 0.1, "lstm": 0.9},
               observed={"ridge": False, "svr": True, "lstm": True})
    d = ViolationAwarePlanner().plan(ctx)
    assert d.action == "switch" and d.model == "svr"


# ── A7: Pareto explores unknown models while violating ───────────────────────

def test_pareto_explores_unknown_when_current_is_known_optimum():
    ctx = _ctx("score", "lstm",
               acc={"ridge": 0.5, "svr": 0.5, "lstm": 0.99},
               eng={"ridge": 0.5, "svr": 0.5, "lstm": 0.2},
               observed={"ridge": False, "svr": False, "lstm": True})
    d = ParetoPlanner().plan(ctx)
    assert d.action == "switch" and d.model in ("ridge", "svr")


# ── C1: batch energy metering ─────────────────────────────────────────────────

class _FakeMeter:
    def __init__(self, total):
        self.total_uJ = total

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_batch_energy_charges_total_over_n(monkeypatch):
    import experiments.run_experiment as rx
    monkeypatch.setattr(rx, "EnergyMeter", lambda *a, **k: _FakeMeter(500.0))
    be = BatchEnergy("auto", enabled=True)
    rows = [{"energy_uJ": 0.0, "energy_valid": True} for _ in range(3)]
    batch_e = [0.0, 0.0]
    be.open("b", first_row=1)            # rows[1:] belong to this batch
    be.close(rows, batch_e)
    assert rows[0]["energy_uJ"] == 0.0
    assert rows[1]["energy_uJ"] == rows[2]["energy_uJ"] == 250.0
    assert batch_e == [250.0, 250.0]


def test_batch_energy_invalid_reading_marks_rows(monkeypatch):
    import experiments.run_experiment as rx
    monkeypatch.setattr(rx, "EnergyMeter", lambda *a, **k: _FakeMeter(None))
    be = BatchEnergy("auto", enabled=True)
    rows = [{"energy_uJ": 0.0, "energy_valid": True}]
    batch_e = [0.0]
    be.open("b", first_row=0)
    be.close(rows, batch_e)
    assert rows[0]["energy_valid"] is False
    assert batch_e == []


def test_step_mode_keeps_per_step_meter():
    be = BatchEnergy("auto", enabled=False)
    be.open("b", first_row=0)  # no-op in step mode
    assert be._em is None

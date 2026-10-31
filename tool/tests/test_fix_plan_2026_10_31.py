"""Tests for the 2026-10-31 concurrent-harness fix plan (context/fix_plan_2026-10-31.md)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from core.planners.base import PlanningContext, get_planner
from core.planners.harmone_original import HarmonEOriginalPlanner

_TOOL = Path(__file__).resolve().parent.parent
MODELS = ["lstm", "ridge", "svr"]
REGRESSION = ["pems", "spot_prices", "uci_electricity",
              "pems_driftinduced", "spot_prices_driftinduced", "uci_electricity_driftinduced"]


def ctx(violation=None, thr=None, current="lstm", step=1000, **kw):
    base = {"alpha": 0.0, "min_score": 0.78, "min_accuracy": 0.78, "explore_prob": 0.0, "monitor_interval": 260}
    base.update(thr or {})
    kw.setdefault("last_observed_step", {"lstm": step, "ridge": step, "svr": step})
    kw.setdefault("observed", {m: True for m in MODELS})
    return PlanningContext(
        violation=violation, ema_scores={"lstm": .9, "ridge": .6, "svr": .7},
        ema_accuracy={"lstm": .95, "ridge": .7, "svr": .75}, ema_energy={"lstm": .9, "ridge": .05, "svr": .3},
        current_model=current, available_models=MODELS, thresholds=base,
        drift_result=None, current_step=step, **kw)


class TestH4CostClass:
    @pytest.mark.parametrize("name", REGRESSION)
    def test_lstm_is_most_expensive(self, name):
        cfg = json.loads((_TOOL / "configs" / "datasets" / f"{name}.json").read_text(encoding="utf-8"))
        assert cfg["models"]["lstm"]["cost_class"] == "heavy"
        assert all(s["cost_class"] != "heavy" for m, s in cfg["models"].items() if m != "lstm")

    def test_s5_can_reach_stale_lstm_from_svr(self):
        cfg = json.loads((_TOOL / "configs" / "datasets" / "uci_electricity_driftinduced.json").read_text(encoding="utf-8"))
        c = ctx("score", thr={**cfg, "alpha": 0, "explore_prob": 0}, current="svr", step=9000,
                staleness_window=2600, last_observed_step={"lstm": 100, "ridge": 8900, "svr": 9000},
                accuracy_credit=0.2)
        c.ema_accuracy = {"lstm": .95, "ridge": .1, "svr": .75}
        d = get_planner("violation_aware").plan(c)
        assert d.action == "switch" and d.model == "lstm"

    def test_calibration_guard(self):
        from experiments.calibrate_from_naive_runs import cost_class_warnings
        pm = {"lstm": {"energy_uJ_per_step": 100.}, "svr": {"energy_uJ_per_step": 50.}, "ridge": {"energy_uJ_per_step": 10.}}
        good = {"models": {"lstm": {"cost_class": "heavy"}, "svr": {"cost_class": "medium"}, "ridge": {"cost_class": "light"}}}
        bad = {"models": {"lstm": {"cost_class": "medium"}, "svr": {"cost_class": "medium"}, "ridge": {"cost_class": "light"}}}
        wrong = {"models": {"lstm": {"cost_class": "light"}, "svr": {"cost_class": "medium"}, "ridge": {"cost_class": "heavy"}}}
        assert cost_class_warnings(good, pm) == []
        assert cost_class_warnings(bad, pm)
        assert len(cost_class_warnings(wrong, pm)) >= 2


class TestH3SwitchBaselines:
    @pytest.mark.parametrize("violation", [None, "energy", "score"])
    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_switch_matches_s4_without_drift(self, violation, alpha):
        for seed in range(30):
            random.seed(seed)
            a = HarmonEOriginalPlanner().plan(ctx(violation, {"alpha": alpha}))
            random.seed(seed)
            b = get_planner("switch").plan(ctx(violation, {"alpha": alpha}))
            assert (a.action, a.model) == (b.action, b.model)

    def test_drift_never_reaches_switch(self):
        from experiments.run_experiment import _initial_mape_info, _plan
        info = _initial_mape_info({m: None for m in MODELS})
        seen = {}
        planner = get_planner("switch")
        orig = planner.plan
        planner.plan = lambda c: (seen.update(v=c.violation, d=c.drift_result), orig(c))[1]
        _plan(None, {"drift_detected": True, "action": "retrain", "version": None}, "lstm", MODELS, info,
              {"alpha": 0.0}, planner)
        assert seen == {"v": None, "d": None}

    def test_switch_prt_fires_on_schedule(self):
        p = get_planner("switch_prt")
        assert p.plan(ctx(step=3380, thr={"prt_interval": 3200})).action == "retrain"
        assert p.plan(ctx(step=3640, thr={"prt_interval": 3200})).action == "noop"

    def test_grid_uses_paper_baselines(self):
        text = (_TOOL / "concurrent_harness" / "run_concurrent_regression_grid.sh").read_text(encoding="utf-8")
        line = next(l for l in text.splitlines() if l.startswith("ADAPTIVE_PLANNERS"))
        assert "random_switch" not in line and "switch switch_prt greedy_switch" in line

    @pytest.mark.parametrize("name", ["random_switch", "random_switch_prt"])
    def test_random_planners_stay_registered(self, name):
        assert get_planner(name).name == name

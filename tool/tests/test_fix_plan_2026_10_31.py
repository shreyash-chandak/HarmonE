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


# ── step 2: decision logic ────────────────────────────────────────────────────

def _harness_modules():
    """concurrent_harness modules (Linux fcntl stubbed), without leaking their
    top-level names to the tests of the managed systems."""
    import sys
    import types
    saved_path, saved = list(sys.path), set(sys.modules)
    sys.path[:0] = [str(_TOOL), str(_TOOL / "concurrent_harness"), str(_TOOL / "concurrent_harness" / "mape")]
    stub = "fcntl" not in sys.modules
    if stub:
        try:
            import fcntl  # noqa: F401
        except ImportError:
            sys.modules["fcntl"] = types.ModuleType("fcntl")
            sys.modules["fcntl"].LOCK_EX = sys.modules["fcntl"].LOCK_UN = sys.modules["fcntl"].LOCK_NB = 0
            sys.modules["fcntl"].flock = lambda *a, **k: None
    import importlib
    mods = {n: importlib.import_module(n) for n in
            ("knowledge_io", "execute", "plan_thread", "mape_cycle", "drift_ref")}
    sys.path[:] = saved_path
    for m in {"execute", "retrain", "knowledge_io", "plan_thread", "mape_cycle", "drift_ref", "drift_thread"} & (set(sys.modules) - saved):
        del sys.modules[m]
    return mods


class _Spy:
    name = "spy"

    def __init__(self):
        self.ctx = None

    def plan(self, c):
        from core.planners.base import PlanDecision
        self.ctx = c
        return PlanDecision(action="noop", reason="spy")


def _info(model="lstm", ema=0.5, baseline=0.9, n=4):
    from experiments.run_experiment import _initial_mape_info
    info = _initial_mape_info({m: None for m in MODELS})
    info["ema_accuracy"][model] = ema
    info.setdefault("acc_since_adapt", {})[model] = [baseline * n, n]
    return info


DRIFT = {"drift_detected": True, "action": "retrain", "version": None, "kl_div": 2.0}
CONFIRM = {"drift_confirmation": "accuracy", "drift_accuracy_drop": 0.05}


class TestH5EnergyBeatsScore:
    def _v(self, score, energy, tau=0.5):
        from experiments.run_experiment import _analyse_violation
        info = {"current_energy_threshold": tau, "event_counters": {}}
        return _analyse_violation({"ema_score": score, "normalized_energy": energy}, info, {"min_score": 0.8}), info

    def test_energy_wins_when_both(self):
        v, info = self._v(0.5, 0.9)
        assert v == "energy" and info["event_counters"]["score_and_energy"] == 1

    def test_each_alone(self):
        assert self._v(0.5, 0.1)[0] == "score"
        assert self._v(0.9, 0.9)[0] == "energy"
        v, info = self._v(0.9, 0.1)
        assert v is None and not info["event_counters"]


class TestH1DriftDeferredOnLag:
    def test_lag_batch_defers_without_confirming(self):
        from experiments.run_experiment import _plan
        spy, info, dr = _Spy(), _info(), dict(DRIFT)
        _plan(None, dr, "lstm", MODELS, info, CONFIRM, spy, served_model="ridge")
        assert spy.ctx.violation is None and spy.ctx.drift_result is None
        assert info["event_counters"]["drift_deferred_lag"] == 1
        assert dr["deferred"] == "lag" and "drift_confirmed" not in dr

    def test_own_batch_adapts(self):
        from experiments.run_experiment import _plan
        spy, info = _Spy(), _info()
        _plan(None, dict(DRIFT), "lstm", MODELS, info, CONFIRM, spy, served_model="lstm")
        assert spy.ctx.violation == "drift"

    def test_frozen_low_ema_cannot_confirm_on_lag(self):
        from experiments.run_experiment import _plan
        spy, info = _Spy(), _info(ema=0.1)
        _plan(None, dict(DRIFT), "lstm", MODELS, info, CONFIRM, spy, served_model="svr")
        assert spy.ctx.violation is None and info["event_counters"].get("drift_unconfirmed", 0) == 0


class TestH2DriftFirst:
    @pytest.mark.parametrize("violation", ["score", "energy"])
    def test_confirmed_drift_beats_violation(self, violation):
        from experiments.run_experiment import _plan
        spy, info = _Spy(), _info()
        d = _plan(violation, dict(DRIFT), "lstm", MODELS, info, CONFIRM, spy, served_model="lstm")
        assert spy.ctx.violation == "drift" and d.effective_violation == "drift"
        assert info["event_counters"]["violation_deferred_drift"] == 1

    def test_unconfirmed_drift_leaves_violation(self):
        from experiments.run_experiment import _plan
        spy, info = _Spy(), _info(ema=0.9)
        _plan("score", dict(DRIFT), "lstm", MODELS, info, CONFIRM, spy, served_model="lstm")
        assert spy.ctx.violation == "score" and spy.ctx.drift_result is None

    def test_energy_with_drift_starts_no_hold(self):
        from experiments.run_experiment import _plan
        spy, info = _Spy(), _info()
        _plan("energy", dict(DRIFT), "lstm", MODELS, info,
              {**CONFIRM, "switch_hold_mode": "after_energy_violation", "switch_hold_cycles": 3}, spy, served_model="lstm")
        assert info.get("recovery_cycles", 0) == 0

    @pytest.mark.parametrize("name", ["harmone_original", "violation_aware", "pareto"])
    def test_planners_adapt_on_drift_first(self, name):
        from experiments.run_experiment import _plan
        info = _info()
        d = _plan("score", dict(DRIFT), "lstm", MODELS, info, {**CONFIRM, "alpha": 1.0, "explore_prob": 1.0},
                  get_planner(name), served_model="lstm")
        assert d.action == "retrain"


class TestBusyGuard:
    def _store(self):
        class S:
            d = {"event_counters": {}}
            def update(self, fn):
                fn(self.d)
        return S()

    def test_suppressed_while_busy(self):
        pt = _harness_modules()["plan_thread"]
        s = self._store()
        out = pt._busy_guard(dict(DRIFT), True, s)
        assert not out["drift_detected"] and out["kl_div"] == 2.0
        assert s.d["event_counters"]["drift_suppressed_busy"] == 1

    def test_passes_when_idle_or_no_drift(self):
        pt = _harness_modules()["plan_thread"]
        s = self._store()
        assert pt._busy_guard(dict(DRIFT), False, s)["drift_detected"]
        pt._busy_guard({"drift_detected": False}, True, s)
        assert not s.d["event_counters"]


class TestSingleVersionRule:
    class _Det:
        def detect(self, _):
            return {"kl_div": 2.0, "drift_detected": True}

    def _vmr(self, tmp_path, n):
        from core.vmr import VMR
        vmr, w = VMR(str(tmp_path / "vmr")), tmp_path / "w.pkl"
        w.write_bytes(b"x")
        for i in range(n):
            vmr.store("ridge", str(w), {"type": "histogram", "data": [1] * 50, "raw": [float(j % 50) for j in range(500)]}, tag=f"v{i}")
        return vmr

    def _go(self, vmr):
        from experiments.run_experiment import _analyse_drift
        cur = {"type": "histogram", "data": [1] * 50, "raw": [float(j % 50) for j in range(500)]}
        return _analyse_drift([0.0], self._Det(), {"tau_drift": 0.5}, vmr=vmr, current_model="ridge", current_distribution=cur)

    def test_one_version_retrains(self, tmp_path):
        out = self._go(self._vmr(tmp_path, 1))
        assert out["action"] == "retrain" and out["single_version"]

    def test_two_versions_replace(self, tmp_path):
        out = self._go(self._vmr(tmp_path, 2))
        assert out["action"] == "replace" and "single_version" not in out


class TestInvalidEnergy:
    def test_no_valid_reading_keeps_estimate(self):
        from experiments.run_experiment import _monitor_batch, _initial_mape_info
        info = _initial_mape_info({m: None for m in MODELS})
        info["observed"]["lstm"] = True
        info["ema_energy"]["lstm"] = 0.9
        t = _monitor_batch([1., 2., 3.], [1., 2., 3.], [], "lstm", info, {"E_m": 0, "E_M": 100, "gamma": 0.8})
        assert t["normalized_energy"] == pytest.approx(0.9)
        assert info["ema_energy"]["lstm"] == pytest.approx(0.9)


# ── step 3: drift reference ───────────────────────────────────────────────────

@pytest.fixture()
def ref_file(tmp_path):
    import numpy as np
    train = np.random.default_rng(2).normal(100, 10, 5000)
    hist, edges = np.histogram(train, bins=50)
    p = tmp_path / "reference_distribution.json"
    p.write_text(json.dumps({"histogram": hist.tolist(), "bin_edges": edges.tolist()}))
    return str(p)


class TestM1SavedWindow:
    def test_return_to_model_uses_window_saved_at_adaptation(self, ref_file):
        import numpy as np
        dr = _harness_modules()["drift_ref"]
        rng = np.random.default_rng(5)
        adapted, later = rng.normal(160, 10, 1200), rng.normal(40, 10, 1200)
        ref = dr.PerModelDriftReference(ref_file, tau_drift=0.5, window_size=1200, cooldown_rows=600)
        ref.note_adaptation("ridge", 3000, adapted, 3000)   # saved window, not the current one
        # Many rows later: the reference is still what the model was adapted to.
        assert not ref.detect("ridge", rng.normal(160, 10, 1200), 9000)["drift_detected"]
        assert ref.detect("ridge", later, 9000)["drift_detected"]

    def test_window_round_trip_and_background_result(self, tmp_path):
        mods = _harness_modules()
        kio, ex = mods["knowledge_io"], mods["execute"]
        assert kio.read_drift_ref_window(tmp_path, "ridge") is None

        class S:
            d = {"event_counters": {"retrains": 0, "retrain_skipped": 0}}
            def update(self, fn):
                fn(self.d)
        ex._bg_results.put(("ridge", True, None, "r", None, [1.0, 2.0]))
        ex.apply_background_results(S(), 3000, tmp_path)
        assert kio.read_drift_ref_window(tmp_path, "ridge") == [1.0, 2.0]
        assert S.d["last_adaptation"]["ridge"] == 3000


class TestM2GlobalCooldown:
    def test_adapting_one_model_suppresses_all(self, ref_file):
        import numpy as np
        dr = _harness_modules()["drift_ref"]
        rng = np.random.default_rng(6)
        ref = dr.PerModelDriftReference(ref_file, tau_drift=0.5, window_size=1200, cooldown_rows=2600)
        w = rng.normal(160, 10, 1200)
        assert ref.detect("svr", w, 3000)["drift_detected"]
        ref.note_adaptation("ridge", 3000, w, 3000)
        r = ref.detect("svr", w, 5000)
        assert r["cooldown"] and not r["drift_detected"]
        assert ref.detect("svr", w, 5600)["drift_detected"]

    def test_t1_ignores_result_older_than_any_adaptation(self, tmp_path):
        mods = _harness_modules()
        kio, pt = mods["knowledge_io"], mods["plan_thread"]
        kp = kio.knowledge_paths(tmp_path)
        kio.write_drift_state(kp["drift_state"], {
            "step": 520, "model": "svr", "adaptation_seen": -1,
            "drift_result": {"drift_detected": True, "action": "retrain", "version": None, "kl_div": 2.0}})
        assert not pt._drift_for_batch(kp, 520, "svr", {"ridge": 520})["drift_detected"]
        assert pt._drift_for_batch(kp, 520, "svr", {})["drift_detected"]


# ── step 4: planners ──────────────────────────────────────────────────────────

COSTS = {"models": {"lstm": {"cost_class": "heavy"}, "svr": {"cost_class": "medium"}, "ridge": {"cost_class": "light"}}}


class TestM6DirectedEscape:
    def _ctx(self, violation, current, unknown):
        step = 9000
        seen = {m: (100 if m in unknown else step) for m in MODELS}
        c = ctx(violation, {**COSTS, "min_accuracy": 0.0, "max_energy": 1.0, "energy_reference": 1.0},
                current=current, step=step, staleness_window=2600, last_observed_step=seen)
        c.ema_accuracy = {"lstm": .99, "svr": .99, "ridge": .99}
        c.ema_energy = {"lstm": .01, "svr": .01, "ridge": .01}
        return c

    def test_score_never_escapes_to_a_cheaper_model(self):
        d = get_planner("pareto").plan(self._ctx("score", "svr", ["ridge"]))
        assert d.action == "noop"

    def test_score_goes_to_costlier_unknown(self):
        d = get_planner("pareto").plan(self._ctx("score", "svr", ["ridge", "lstm"]))
        assert d.action == "switch" and d.model == "lstm"

    def test_energy_goes_to_cheapest_unknown(self):
        d = get_planner("pareto").plan(self._ctx("energy", "lstm", ["svr", "ridge"]))
        assert d.action == "switch" and d.model == "ridge"

    def test_energy_never_escapes_to_a_costlier_model(self):
        d = get_planner("pareto").plan(self._ctx("energy", "ridge", ["lstm"]))
        assert d.action == "noop"

    def test_cost_rank_shared(self):
        from core.planners import base, pareto, violation_aware, violation_aware_gated
        assert violation_aware._cost_rank is base.cost_rank
        assert violation_aware_gated._cost_rank is base.cost_rank
        assert pareto.cost_rank is base.cost_rank


class TestM5BanditContext:
    def _go(self, tmp_path, lag):
        import numpy as np
        from core.planners.bandit import BanditPlanner, LinUCBBandit
        b = LinUCBBandit(MODELS, 4, 1.0, tmp_path / "s.json")
        seen = []
        orig = b.observe_outcome
        b.observe_outcome = lambda x, arm, a, e: (seen.append(np.asarray(x).copy()), orig(x, arm, a, e))
        planner = BanditPlanner(bandit=b)
        for k in range(5):
            c = ctx(thr={"tau_drift": 1.0, **COSTS}, current_energy_threshold=0.1 * (k + 1), telemetry={
                "accuracy": 0.9, "normalized_energy": 0.5, "served_model": "lstm", "actuation_lag": lag})
            planner.plan(c)
        return seen

    def test_outcome_pairs_with_the_decision_that_took_effect(self, tmp_path):
        seen = self._go(tmp_path, 1)
        # cycle k (τ = 0.1·(k+1)) learns with the context of cycle k-2
        assert [round(float(x[3]), 1) for x in seen] == [0.1, 0.2, 0.3]

    def test_no_lag_keeps_previous_cycle(self, tmp_path):
        seen = self._go(tmp_path, 0)
        assert [round(float(x[3]), 1) for x in seen] == [0.1, 0.2, 0.3, 0.4]

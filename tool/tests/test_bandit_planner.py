"""S7 bandit v3 (2026-10-06, audit B1–B6): learns from every monitored batch,
decides every cycle, unit-consistent energy feasibility, Lyapunov energy debt."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.planners.bandit import (  # noqa: E402
    BanditPlanner,
    LinUCBBandit,
    build_context,
    load_or_create_bandit,
    set_bandit_instance,
)
from core.planners.base import REGISTRY, PlanningContext  # noqa: E402

MODELS = ["ridge", "svr", "lstm"]
# true per-arm behaviour of a toy environment: (accuracy, normalised energy)
TRUTH = {"ridge": (0.80, 0.0), "svr": (0.88, 0.1), "lstm": (0.99, 1.0)}


def _bandit(tmp_path, **kw):
    thr = {"bandit_alpha": kw.pop("alpha", 0.5), "E_ref": kw.pop("E_ref", 0.5), **kw}
    return load_or_create_bandit(thr, MODELS, tmp_path, dataset_id="toy")


def _ctx(current, served=None, violation=None, tel=None, threshold=1.0, drift_result=None):
    acc, eng = TRUTH[served or current]
    telemetry = {"accuracy": acc, "normalized_energy": eng, "kl_div": 0.0,
                 "volatility": 0.3, "served_model": served or current}
    if tel is not None:
        telemetry = tel
    return PlanningContext(
        violation=violation, ema_scores={m: 0.5 for m in MODELS},
        ema_accuracy={m: 0.5 for m in MODELS}, ema_energy={m: 0.5 for m in MODELS},
        current_model=current, available_models=MODELS,
        thresholds={"tau_drift": 0.5, "bandit_switch_cost": 0.01},
        drift_result=drift_result, current_energy_threshold=threshold, telemetry=telemetry,
    )


def _run(planner, cycles, start="lstm", threshold=1.0):
    """Toy closed loop: the chosen model serves the next batch."""
    current, served_counts = start, {m: 0 for m in MODELS}
    for _ in range(cycles):
        d = planner.plan(_ctx(current, threshold=threshold))
        if d.action == "switch":
            current = d.model
        served_counts[current] += 1
    return served_counts


def test_registered():
    assert REGISTRY["bandit"] is BanditPlanner


def test_no_instance_raises():
    set_bandit_instance(None)
    with pytest.raises(RuntimeError):
        BanditPlanner()


def test_learns_every_cycle_and_persists_state(tmp_path):
    b = _bandit(tmp_path)
    p = BanditPlanner(bandit=b)
    _run(p, 20)
    assert b.total_updates == 19          # first cycle has no prior decision
    assert sum(b.n_obs.values()) == 19
    state = json.loads((tmp_path / "bandit_state.json").read_text())
    assert state["state_version"] == 3 and state["total_updates"] == 19


def test_tries_each_arm_first(tmp_path):
    p = BanditPlanner(bandit=_bandit(tmp_path))
    seen = []
    current = "lstm"
    for _ in range(4):
        d = p.plan(_ctx(current))
        if d.action == "switch":
            current = d.model
        seen.append(current)
    assert set(seen) == set(MODELS)


def test_estimates_converge_to_arm_behaviour(tmp_path):
    b = _bandit(tmp_path, alpha=0.1)
    p = BanditPlanner(bandit=b)
    _run(p, 200)
    x = build_context(_ctx("svr"))
    for m, (acc, eng) in TRUTH.items():
        if b.n_obs[m] >= 5:
            mean_acc = float(np.linalg.inv(b.A_acc[m]) @ b.b_acc[m] @ x)
            assert abs(mean_acc - acc) < 0.1, (m, mean_acc)


def test_energy_debt_limits_expensive_arm(tmp_path):
    # Budget E_ref 0.5: lstm (energy 1.0) builds debt, so it cannot be served
    # all the time even though it is the most accurate.
    b = _bandit(tmp_path, alpha=0.1, E_ref=0.5, bandit_mu=0.2)
    counts = _run(BanditPlanner(bandit=b), 300)
    assert counts["lstm"] < 300 * 0.9
    assert counts["lstm"] > 0


def test_generous_budget_prefers_accurate_arm(tmp_path):
    b = _bandit(tmp_path, alpha=0.1, E_ref=1.0)
    counts = _run(BanditPlanner(bandit=b), 200)
    assert max(counts, key=counts.get) == "lstm"


def test_feasibility_uses_normalised_energy(tmp_path):
    b = _bandit(tmp_path, alpha=0.0)
    for m, (acc, eng) in TRUTH.items():
        for _ in range(5):
            b.observe_outcome(np.ones(4), m, acc, eng)
    chosen, diag = b.select_action(np.ones(4), "lstm", MODELS, energy_threshold=0.2)
    assert "lstm" not in diag["feasible"]
    assert chosen in ("ridge", "svr")


def test_stay_is_noop_and_decides_without_violation(tmp_path):
    b = _bandit(tmp_path, alpha=0.0, E_ref=1.0)
    for m, (acc, eng) in TRUTH.items():
        b.observe_outcome(np.ones(4), m, acc, eng)
    b.pending_x = None
    d = BanditPlanner(bandit=b).plan(_ctx("lstm", violation=None))
    assert d.action == "noop" and "stay on lstm" in d.reason


def test_drift_routes_to_fixed_rule(tmp_path):
    p = BanditPlanner(bandit=_bandit(tmp_path))
    d = p.plan(_ctx("lstm", violation="drift",
                    drift_result={"drift_detected": True, "action": "retrain"}))
    assert d.action == "retrain"
    d = p.plan(_ctx("lstm", violation="drift",
                    drift_result={"drift_detected": True, "action": "replace", "version": "v.pkl"}))
    assert d.action == "replace" and d.version_path == "v.pkl"


def test_no_learning_without_telemetry(tmp_path):
    b = _bandit(tmp_path)
    p = BanditPlanner(bandit=b)
    p.plan(_ctx("lstm"))
    p.plan(_ctx("lstm", tel={}))
    assert b.total_updates == 0


def test_context_describes_stream_not_arm(tmp_path):
    x_svr = build_context(_ctx("svr"))
    x_lstm = build_context(_ctx("lstm"))
    assert x_svr.shape == (4,)
    assert np.all(x_svr >= 0) and np.all(x_svr <= 1)
    assert np.array_equal(x_svr, x_lstm)  # same stream state, different serving arm

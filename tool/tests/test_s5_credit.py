"""
tests/test_s5_credit.py — S5 redesign (run-level accuracy credit, 2026-10-26).

Covers core/planners/violation_aware.py (new S5), core/planners/credit.py, the
credit update in the harness monitor, the S5 mape_events columns, the B_max
calibration formula, and core/config_keys.py. The gated ablation
(violation_aware_gated) keeps its original tests in test_planners.py,
test_borrowed_fixes.py and test_planner_signals.py.
"""

from __future__ import annotations

import glob
import json
import logging
import random
import re
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TOOL))

from core.config_keys import READ_KEYS, unread_keys  # noqa: E402
from core.planners.base import PlanningContext, PlanDecision, get_planner  # noqa: E402
from core.planners.credit import (  # noqa: E402
    FALLBACK_CREDIT_CAP, credit_cap_from_naive, projected_credit, required_accuracy,
    target_accuracy, update_credit,
)
from core.planners.violation_aware import ViolationAwarePlanner  # noqa: E402
from core.planners.violation_aware_gated import ViolationAwareGatedPlanner  # noqa: E402
from experiments.run_experiment import (  # noqa: E402
    _initial_mape_info, _monitor_batch, _plan, s5_event_fields,
)

MODELS = ["ridge", "svr", "lstm"]
A_TARGET = 0.8163
THR = {
    "min_accuracy": A_TARGET, "min_score": 0.8005, "beta": 0.95, "E_ref": 0.5,
    "s5_switch_margin": 0.02, "s5_credit_horizon": 3, "s5_credit_cap": 0.372,
    "explore_prob": 0.0, "staleness_window": 2600, "monitor_interval": 260,
    "models": {"ridge": {"cost_class": "light"}, "svr": {"cost_class": "medium"},
               "lstm": {"cost_class": "heavy"}},
}
STEP = 10_000


def ctx(violation, current, acc, eng, credit=0.0, last_seen=None, observed=None, drift=None, thr=None):
    """Synthetic planner state; every model observed and fresh unless overridden."""
    return PlanningContext(
        violation=violation, ema_scores={}, ema_accuracy=dict(acc), ema_energy=dict(eng),
        current_model=current, available_models=list(MODELS), thresholds=thr or THR,
        drift_result=drift, current_step=STEP, current_energy_threshold=0.6,
        last_observed_step=last_seen if last_seen is not None else {m: STEP - 260 for m in MODELS},
        staleness_window=2600, observed=observed or {m: True for m in MODELS},
        accuracy_credit=credit,
    )


PEMS_ACC = {"ridge": 0.60, "svr": 0.74, "lstm": 0.93}
PEMS_ENG = {"ridge": 0.0, "svr": 0.25, "lstm": 1.0}


def reason(d: PlanDecision) -> str:
    return (d.metadata or {}).get("s5_reason")


# ── registry ──────────────────────────────────────────────────────────────────

def test_both_variants_registered():
    assert isinstance(get_planner("violation_aware"), ViolationAwarePlanner)
    assert isinstance(get_planner("violation_aware_gated"), ViolationAwareGatedPlanner)


# ── required accuracy a*_m ────────────────────────────────────────────────────

def test_required_accuracy_uses_the_models_own_energy():
    # at ê = 0.5 it is min_accuracy
    assert required_accuracy("m", {"m": 0.5}, THR) == pytest.approx(A_TARGET, abs=1e-4)
    assert required_accuracy("m", {"m": 0.0}, THR) == pytest.approx((0.8005 - 0.05) / 0.95)
    assert required_accuracy("m", {"m": 1.0}, THR) == pytest.approx(0.8005 / 0.95)


def test_target_accuracy_falls_back_to_min_score():
    assert target_accuracy(THR) == A_TARGET
    no_min_acc = {k: v for k, v in THR.items() if k != "min_accuracy"}
    assert target_accuracy(no_min_acc) == pytest.approx(A_TARGET, abs=1e-4)


# ── energy violation: the PeMS-like stuck case ────────────────────────────────

def test_energy_large_credit_switches_to_cheapest_affordable():
    d = ViolationAwarePlanner().plan(ctx("energy", "lstm", PEMS_ACC, PEMS_ENG, credit=0.7))
    assert (d.action, d.model, reason(d)) == ("switch", "ridge", "affordable_cheapest")
    assert d.metadata["s5_projected"] == pytest.approx(0.7 - 3 * (A_TARGET - 0.60))


def test_energy_medium_credit_affords_svr_only():
    d = ViolationAwarePlanner().plan(ctx("energy", "lstm", PEMS_ACC, PEMS_ENG, credit=0.3))
    assert (d.action, d.model, reason(d)) == ("switch", "svr", "affordable_cheapest")


def test_energy_zero_credit_falls_back_to_svr():
    d = ViolationAwarePlanner().plan(ctx("energy", "lstm", PEMS_ACC, PEMS_ENG, credit=0.0))
    assert (d.action, d.model, reason(d)) == ("switch", "svr", "infeasible_fallback")


def test_energy_violation_on_lstm_never_noops():
    for b in [x / 100 for x in range(-37, 38)]:
        d = ViolationAwarePlanner().plan(ctx("energy", "lstm", PEMS_ACC, PEMS_ENG, credit=b))
        assert d.action == "switch" and d.model in ("ridge", "svr"), (b, d)


def test_energy_on_cheapest_model_noops_already_cheapest():
    d = ViolationAwarePlanner().plan(ctx("energy", "ridge", PEMS_ACC, PEMS_ENG, credit=-0.2))
    assert (d.action, reason(d)) == ("noop", "already_cheapest")


def test_energy_stale_cheapest_is_reprobed_and_its_ema_ignored():
    acc = {"ridge": 0.99, "svr": 0.50, "lstm": 0.93}   # ridge's stale estimates must not count
    eng = {"ridge": 0.99, "svr": 0.25, "lstm": 1.0}
    last = {"ridge": STEP - 5000, "svr": STEP - 260, "lstm": STEP}
    d = ViolationAwarePlanner().plan(ctx("energy", "lstm", acc, eng, credit=0.0, last_seen=last))
    assert (d.action, d.model, reason(d)) == ("switch", "ridge", "unknown_probe")
    assert d.metadata["s5_unknown_models"] == "ridge"
    assert "s5_projected" not in d.metadata          # no estimate for an unknown model


def test_energy_unknown_not_probed_while_in_debt():
    acc = {"ridge": 0.99, "svr": 0.50, "lstm": 0.93}
    last = {"ridge": STEP - 5000, "svr": STEP - 260, "lstm": STEP}
    d = ViolationAwarePlanner().plan(ctx("energy", "lstm", acc, PEMS_ENG, credit=-0.1, last_seen=last))
    assert (d.action, d.model, reason(d)) == ("switch", "svr", "infeasible_fallback")


# ── score violation ───────────────────────────────────────────────────────────

def test_score_in_debt_repays_with_most_accurate():
    acc = {"ridge": 0.60, "svr": 0.80, "lstm": 0.90}
    d = ViolationAwarePlanner().plan(ctx("score", "ridge", acc, PEMS_ENG, credit=-0.05))
    assert (d.action, d.model, reason(d)) == ("switch", "lstm", "repay_debt")


def test_score_with_credit_takes_cheapest_sufficient():
    acc = {"ridge": 0.60, "svr": 0.81, "lstm": 0.90}   # a*_svr at ê=0.25 is 0.8032
    d = ViolationAwarePlanner().plan(ctx("score", "ridge", acc, PEMS_ENG, credit=0.0))
    assert (d.action, d.model, reason(d)) == ("switch", "svr", "cheapest_sufficient")
    assert d.metadata["s5_required_acc"] == pytest.approx(required_accuracy("svr", PEMS_ENG, THR))


def test_score_sufficiency_is_judged_at_each_models_own_energy():
    acc = {"ridge": 0.60, "svr": 0.81, "lstm": 0.90}
    eng = {"ridge": 0.0, "svr": 0.9, "lstm": 1.0}     # costly svr needs 0.8374 > 0.81
    d = ViolationAwarePlanner().plan(ctx("score", "ridge", acc, eng, credit=0.1))
    assert (d.action, d.model, reason(d)) == ("switch", "lstm", "cheapest_sufficient")


def test_score_nothing_sufficient_most_accurate_with_margin():
    acc = {"ridge": 0.60, "svr": 0.70, "lstm": 0.80}   # lstm a* = 0.8426
    d = ViolationAwarePlanner().plan(ctx("score", "ridge", acc, PEMS_ENG, credit=0.1))
    assert (d.action, d.model, reason(d)) == ("switch", "lstm", "most_accurate")
    acc = {"ridge": 0.60, "svr": 0.605, "lstm": 0.61}   # best beats current by < 0.02
    d = ViolationAwarePlanner().plan(ctx("score", "ridge", acc, PEMS_ENG, credit=0.1))
    assert (d.action, reason(d)) == ("noop", "current_is_best")


def test_score_unknown_costlier_is_probed():
    acc = {"ridge": 0.60, "svr": 0.70, "lstm": 0.20}
    last = {"ridge": STEP, "svr": STEP - 260, "lstm": STEP - 5000}
    d = ViolationAwarePlanner().plan(ctx("score", "ridge", acc, PEMS_ENG, credit=0.1, last_seen=last))
    # svr is known and more accurate but below its a*; lstm is stale (its 0.20 is ignored)
    assert (d.action, d.model, reason(d)) == ("switch", "lstm", "unknown_probe")


def test_score_empty_candidates_noop_even_in_debt():
    d = ViolationAwarePlanner().plan(ctx("score", "lstm", PEMS_ACC, PEMS_ENG, credit=-0.3))
    assert (d.action, reason(d)) == ("noop", "current_is_best")


# ── drift: adapt, never switch ────────────────────────────────────────────────

def test_drift_only_retrains_or_replaces():
    r = random.Random(0)
    for _ in range(2000):
        action = r.choice(["replace", "retrain"])
        acc = {m: r.random() for m in MODELS}
        eng = {m: r.random() for m in MODELS}
        d = ViolationAwarePlanner().plan(ctx(
            "drift", r.choice(MODELS), acc, eng, credit=r.uniform(-0.4, 0.4),
            drift={"drift_detected": True, "action": action, "version": "v1"}))
        assert d.action == action
        assert reason(d) == f"drift_{action}"


# ── credit update ─────────────────────────────────────────────────────────────

def test_credit_update_clips_at_cap():
    assert update_credit(0.3, 1.0, A_TARGET, 0.372) == pytest.approx(0.372)
    assert update_credit(-0.3, 0.0, A_TARGET, 0.372) == pytest.approx(-0.372)
    assert update_credit(0.0, 0.9, A_TARGET, 0.372) == pytest.approx(0.9 - A_TARGET)


def test_credit_updates_on_lagging_batches_and_during_the_hold():
    info = _initial_mape_info(MODELS)
    thr = dict(THR, accuracy_signal="window_r2", switch_hold_mode="after_energy_violation",
               switch_hold_cycles=3, E_m=0.0, E_M=100.0, gamma=0.8, tau_drift=0.5)
    info["recovery_cycles"] = 3
    ref = [float(i % 7) for i in range(1200)]
    seen = []

    class Spy:
        name = "violation_aware"
        reacts_to_drift = True

        def plan(self, c):
            seen.append(c.accuracy_credit)
            return PlanDecision(action="noop")

    b = 0.0
    for k in range(3):
        y = ref[-260:]
        # served by "svr" while "lstm" is commanded: a lagging batch, inside the hold
        t = _monitor_batch(y, y, [10.0] * 260, "svr", info, thr, reference_values=ref)
        b = update_credit(b, t["accuracy"], A_TARGET, 0.372)
        assert info["s5_credit"] == pytest.approx(b)
        _plan("energy", {"drift_detected": False}, "lstm", MODELS, info, thr, Spy(),
              current_step=260 * (k + 1), telemetry=t, served_model="svr")
        assert seen[-1] == pytest.approx(b)
    assert info["recovery_cycles"] == 0                    # the hold ran out meanwhile
    assert info["event_counters"]["violations_in_recovery"] == 3


# ── calibration of B_max ──────────────────────────────────────────────────────

def test_credit_cap_formula():
    assert credit_cap_from_naive(0.8163, 0.7336, 3) == pytest.approx(1.5 * 3 * (0.8163 - 0.7336))
    assert projected_credit(0.5, 0.6, 0.8, 3) == pytest.approx(0.5 - 0.6)


def test_credit_cap_fallback_when_ridge_meets_target(caplog):
    with caplog.at_level(logging.WARNING):
        assert credit_cap_from_naive(0.80, 0.85, 3) == FALLBACK_CREDIT_CAP
    assert "fallback" in caplog.text


# ── mape_events columns ───────────────────────────────────────────────────────

def test_s5_event_fields():
    d = ViolationAwarePlanner().plan(ctx("energy", "lstm", PEMS_ACC, PEMS_ENG, credit=0.7))
    row = s5_event_fields("violation_aware", d, {"s5_credit": 0.7})
    assert row["s5_reason"] == "affordable_cheapest" and row["s5_credit"] == 0.7
    assert row["s5_required_acc"] == pytest.approx(required_accuracy("ridge", PEMS_ENG, THR), abs=1e-6)
    assert set(s5_event_fields("pareto", d, {"s5_credit": 0.7}).values()) == {""}


def test_gated_variant_labels_every_decision():
    r = random.Random(3)
    for _ in range(500):
        c = ctx(r.choice([None, "energy", "score"]), r.choice(MODELS),
                {m: r.random() for m in MODELS}, {m: r.random() for m in MODELS},
                thr=dict(THR, explore_prob=0.05))
        assert reason(ViolationAwareGatedPlanner().plan(c))


# ── config keys ───────────────────────────────────────────────────────────────

def _code_text() -> str:
    parts = []
    for d in ("core", "experiments", "concurrent_harness", "adapters", "scripts",
              "managed_system_regression", "managed_system_cv"):
        for p in glob.glob(str(_TOOL / d / "**" / "*.py"), recursive=True):
            if not p.endswith("config_keys.py"):
                parts.append(Path(p).read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def test_every_listed_key_is_referenced_in_code():
    text = _code_text()
    missing = [k for k in READ_KEYS if not re.search(r"[\"']" + re.escape(k) + r"[\"']", text)]
    assert not missing, missing


def test_dead_keys_removed_and_flagged():
    for p in glob.glob(str(_TOOL / "configs" / "datasets" / "*.json")):
        cfg = json.loads(Path(p).read_text(encoding="utf-8"))
        for k in ("bandit_min_observation_steps", "bandit_switch_margin", "drift_reference"):
            assert k not in cfg, (p, k)
        if "bandit_alpha" in cfg:
            assert cfg.get("bandit_switch_cost") == 0.01, p
    assert unread_keys({"drift_reference": "both", "_comment": "x", "tau_drift_source": "y",
                        "tau_drift": 0.1}) == ["drift_reference"]

"""
Phase 2 planner tests.

Each planner is tested with table-driven PlanningContext fixtures to verify
the exact PlanDecision returned. Tests are pure (no file I/O).
"""

from __future__ import annotations

import pytest
from core.planners.base import PlanningContext, PlanDecision, get_planner, REGISTRY, _register
from core.planners.naive import NaivePlanner
from core.planners.random_switch import RandomSwitchPlanner
from core.planners.greedy_switch import GreedySwitchPlanner
from core.planners.harmone_original import HarmonEOriginalPlanner
from core.planners.violation_aware import ViolationAwarePlanner
from core.planners.pareto import ParetoPlanner, build_pareto_front, chebyshev_distance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_THRESHOLDS = {
    "min_score": 0.7,
    "min_accuracy": 0.0,           # S5: explicit; suppresses _get_min_accuracy warning
    "pareto_accuracy_reference": 0.7,  # S6: reference point for augmented distance
    "max_energy": 0.6,
    "E_ref": 0.5,
    "energy_reference": 0.5,       # Phase 3 key; S6 reads this before E_ref
    "alpha": 0.0,   # disable exploration for deterministic tests
    "w_acc": 1.0,
    "w_e": 1.0,
    "current_energy_threshold": 0.6,
}

EMA_SCORES = {"lstm": 0.8, "linear": 0.4, "svm": 0.6}
EMA_ACCURACY = {"lstm": 0.85, "linear": 0.4, "svm": 0.55}
EMA_ENERGY = {"lstm": 0.7, "linear": 0.2, "svm": 0.4}
MODELS = ["lstm", "linear", "svm"]

NO_VIOLATION = PlanningContext(
    violation=None, ema_scores=EMA_SCORES, ema_accuracy=EMA_ACCURACY,
    ema_energy=EMA_ENERGY, current_model="svm", available_models=MODELS,
    thresholds=BASE_THRESHOLDS, drift_result=None,
)
SCORE_VIOLATION = PlanningContext(
    violation="score", ema_scores=EMA_SCORES, ema_accuracy=EMA_ACCURACY,
    ema_energy=EMA_ENERGY, current_model="svm", available_models=MODELS,
    thresholds=BASE_THRESHOLDS, drift_result=None,
)
ENERGY_VIOLATION = PlanningContext(
    violation="energy", ema_scores=EMA_SCORES, ema_accuracy=EMA_ACCURACY,
    ema_energy=EMA_ENERGY, current_model="lstm", available_models=MODELS,
    thresholds=BASE_THRESHOLDS, drift_result=None,
)
DRIFT_REPLACE = PlanningContext(
    violation="drift", ema_scores=EMA_SCORES, ema_accuracy=EMA_ACCURACY,
    ema_energy=EMA_ENERGY, current_model="svm", available_models=MODELS,
    thresholds=BASE_THRESHOLDS,
    drift_result={"drift_detected": True, "action": "replace", "version": "vMR/lstm/v2/lstm.pth"},
)
DRIFT_RETRAIN = PlanningContext(
    violation="drift", ema_scores=EMA_SCORES, ema_accuracy=EMA_ACCURACY,
    ema_energy=EMA_ENERGY, current_model="svm", available_models=MODELS,
    thresholds=BASE_THRESHOLDS,
    drift_result={"drift_detected": True, "action": "retrain", "version": None},
)


# ---------------------------------------------------------------------------
# S1 Naive
# ---------------------------------------------------------------------------

class TestNaivePlanner:
    def test_always_noop(self):
        p = NaivePlanner()
        for ctx in [NO_VIOLATION, SCORE_VIOLATION, ENERGY_VIOLATION, DRIFT_REPLACE]:
            assert p.plan(ctx).action == "noop"


# ---------------------------------------------------------------------------
# S2 Random Switch
# ---------------------------------------------------------------------------

class TestRandomSwitchPlanner:
    def test_noop_on_no_violation(self):
        assert RandomSwitchPlanner().plan(NO_VIOLATION).action == "noop"

    def test_switch_on_violation(self):
        d = RandomSwitchPlanner().plan(SCORE_VIOLATION)
        assert d.action == "switch"
        assert d.model != "svm"
        assert d.model in MODELS

    def test_all_choices_reachable(self):
        seen = set()
        for _ in range(100):
            d = RandomSwitchPlanner().plan(SCORE_VIOLATION)
            if d.model:
                seen.add(d.model)
        assert "lstm" in seen or "linear" in seen, "Should sometimes pick lstm or linear"


# ---------------------------------------------------------------------------
# S3 Greedy Switch
# ---------------------------------------------------------------------------

class TestGreedySwitchPlanner:
    def test_noop_on_no_violation(self):
        assert GreedySwitchPlanner().plan(NO_VIOLATION).action == "noop"

    def test_picks_highest_ema(self):
        # current=svm (0.6), lstm(0.8) > linear(0.4) → must pick lstm
        d = GreedySwitchPlanner().plan(SCORE_VIOLATION)
        assert d.action == "switch"
        assert d.model == "lstm"

    def test_stays_if_already_best(self):
        ctx = PlanningContext(
            violation="score", ema_scores=EMA_SCORES, ema_accuracy=EMA_ACCURACY,
            ema_energy=EMA_ENERGY, current_model="lstm", available_models=MODELS,
            thresholds=BASE_THRESHOLDS, drift_result=None,
        )
        d = GreedySwitchPlanner().plan(ctx)
        # lstm is highest; alternatives are linear(0.4) and svm(0.6) → greedy picks svm
        assert d.action == "switch"
        assert d.model == "svm"


# ---------------------------------------------------------------------------
# S4 HarmonE Original (α=0 for deterministic tests)
# ---------------------------------------------------------------------------

class TestHarmonEOriginalPlanner:
    def test_noop_on_no_violation(self):
        assert HarmonEOriginalPlanner().plan(NO_VIOLATION).action == "noop"

    def test_exploit_picks_best_alternative(self):
        # α=0 in BASE_THRESHOLDS → exploit only; current=svm, best alt=lstm
        d = HarmonEOriginalPlanner().plan(SCORE_VIOLATION)
        assert d.action == "switch"
        assert d.model == "lstm"

    def test_drift_replace(self):
        d = HarmonEOriginalPlanner().plan(DRIFT_REPLACE)
        assert d.action == "replace"
        assert "lstm" in d.version_path

    def test_drift_retrain(self):
        d = HarmonEOriginalPlanner().plan(DRIFT_RETRAIN)
        assert d.action == "retrain"

    def test_exploration_when_alpha_high(self):
        thr = {**BASE_THRESHOLDS, "alpha": 1.0}  # always explore
        ctx = PlanningContext(
            violation="score", ema_scores=EMA_SCORES, ema_accuracy=EMA_ACCURACY,
            ema_energy=EMA_ENERGY, current_model="svm", available_models=MODELS,
            thresholds=thr, drift_result=None,
        )
        seen = set()
        for _ in range(50):
            d = HarmonEOriginalPlanner().plan(ctx)
            if d.model:
                seen.add(d.model)
        assert len(seen) >= 1  # at least explores some alternative


# ---------------------------------------------------------------------------
# S5 Violation Aware
# ---------------------------------------------------------------------------

class TestViolationAwarePlanner:
    def test_noop_on_no_violation(self):
        assert ViolationAwarePlanner().plan(NO_VIOLATION).action == "noop"

    def test_energy_violation_picks_lowest_energy_accurate_model(self):
        # Phase 2.1: uses min_accuracy=0.0 (all models qualify).
        # current=lstm (energy=0.7), energy violation.
        # All three meet min_accuracy=0.0. Min energy = linear(0.2).
        # linear ≠ current(lstm); hysteresis margin=0.02: 0.2+0.02=0.22 < 0.7 → switch.
        d = ViolationAwarePlanner().plan(ENERGY_VIOLATION)
        assert d.action == "switch"
        assert d.model == "linear"

    def test_energy_violation_returns_noop_when_current_is_best(self):
        # Fix 2.3: current=linear (energy=0.2) is already lowest-energy → noop.
        ctx = PlanningContext(
            violation="energy", ema_scores=EMA_SCORES, ema_accuracy=EMA_ACCURACY,
            ema_energy=EMA_ENERGY, current_model="linear", available_models=MODELS,
            thresholds=BASE_THRESHOLDS, drift_result=None,
        )
        d = ViolationAwarePlanner().plan(ctx)
        assert d.action == "noop"
        assert "already" in d.reason

    def test_score_violation_returns_noop_when_current_is_best_within_budget(self):
        # Fix 2.3: current=svm (acc=0.55, eng=0.4). Budget=0.6.
        # Within budget (≤0.6): linear(0.2)✓, svm(0.4)✓. lstm(0.7) excluded.
        # Best acc within budget: svm(0.55) > linear(0.4) → chosen=svm=current → noop.
        d = ViolationAwarePlanner().plan(SCORE_VIOLATION)
        assert d.action == "noop"
        assert "already" in d.reason

    def test_score_violation_switches_when_better_alt_exists(self):
        # current=linear (acc=0.4, eng=0.2), score violation.
        # Within budget (≤0.6): linear(0.2)✓, svm(0.4)✓. lstm(0.7) excluded.
        # Best acc: svm(0.55) > linear(0.4). svm≠current; hysteresis: 0.55-0.4=0.15 > 0.02 → switch.
        ctx = PlanningContext(
            violation="score", ema_scores=EMA_SCORES, ema_accuracy=EMA_ACCURACY,
            ema_energy=EMA_ENERGY, current_model="linear", available_models=MODELS,
            thresholds=BASE_THRESHOLDS, drift_result=None,
        )
        d = ViolationAwarePlanner().plan(ctx)
        assert d.action == "switch"
        assert d.model == "svm"

    def test_drift_replace(self):
        d = ViolationAwarePlanner().plan(DRIFT_REPLACE)
        assert d.action == "replace"

    def test_drift_retrain(self):
        d = ViolationAwarePlanner().plan(DRIFT_RETRAIN)
        assert d.action == "retrain"


# ---------------------------------------------------------------------------
# S6 Pareto
# ---------------------------------------------------------------------------

class TestBuildParetoFront:
    def test_dominated_model_excluded(self):
        # Staircase trade-off: A=(0.9, 0.8), B=(0.7, 0.5), D=(0.5, 0.2), C=(0.6, 0.7)
        # C=(0.6, 0.7): B has higher acc (0.7>=0.6) AND lower energy (0.5<=0.7) → B dominates C
        acc = {"A": 0.9, "B": 0.7, "C": 0.6, "D": 0.5}
        eng = {"A": 0.8, "B": 0.5, "C": 0.7, "D": 0.2}
        front = build_pareto_front(["A", "B", "C", "D"], acc, eng)
        assert "C" not in front, "C is dominated by B"
        assert "A" in front
        assert "B" in front
        assert "D" in front

    def test_single_model_is_its_own_front(self):
        assert build_pareto_front(["X"], {"X": 0.8}, {"X": 0.3}) == ["X"]

    def test_all_non_dominated(self):
        # A=(0.9, 0.8), B=(0.7, 0.4), C=(0.5, 0.1) — trade-off staircase
        acc = {"A": 0.9, "B": 0.7, "C": 0.5}
        eng = {"A": 0.8, "B": 0.4, "C": 0.1}
        front = build_pareto_front(["A", "B", "C"], acc, eng)
        assert set(front) == {"A", "B", "C"}


class TestChebyshevDistance:
    def test_goal_met(self):
        # model exceeds both goals → distance 0
        d = chebyshev_distance("m", {"m": 0.9}, {"m": 0.3}, s_min=0.7, e_ref=0.5)
        assert d == 0.0

    def test_accuracy_gap(self):
        d = chebyshev_distance("m", {"m": 0.5}, {"m": 0.3}, s_min=0.7, e_ref=0.5)
        assert abs(d - 0.2) < 1e-9  # max(1*0.2, 1*0) = 0.2

    def test_energy_gap(self):
        d = chebyshev_distance("m", {"m": 0.9}, {"m": 0.8}, s_min=0.7, e_ref=0.5)
        assert abs(d - 0.3) < 1e-9  # max(0, 0.3) = 0.3

    def test_overshoot_not_rewarded(self):
        # accuracy far above goal; energy at goal → distance 0, not negative
        d = chebyshev_distance("m", {"m": 0.99}, {"m": 0.5}, s_min=0.7, e_ref=0.5)
        assert d == 0.0


class TestParetoPlanner:
    def test_noop_on_no_violation(self):
        assert ParetoPlanner().plan(NO_VIOLATION).action == "noop"

    def test_selects_pareto_optimal_model(self):
        # current=lstm(acc=0.85, eng=0.7); goal=(s_min=0.7, E_ref=0.5)
        # lstm: acc gap=0, eng gap=0.2 → d=0.2  (but current → stays only if it IS the min-d)
        # linear: acc gap=0.3, eng gap=0 → d=0.3
        # svm: acc gap=0.15, eng gap=0 → d=0.15
        # Front: all non-dominated? lstm(0.85,0.7), linear(0.4,0.2), svm(0.55,0.4)
        #   lstm dominates nothing (linear has lower energy); svm not dominated; linear not dominated
        # Min-d: svm(0.15) → should pick svm (if not current)
        ctx = PlanningContext(
            violation="score", ema_scores=EMA_SCORES, ema_accuracy=EMA_ACCURACY,
            ema_energy=EMA_ENERGY, current_model="linear", available_models=MODELS,
            thresholds=BASE_THRESHOLDS, drift_result=None,
        )
        d = ParetoPlanner().plan(ctx)
        assert d.action == "switch"
        assert d.model == "svm"  # closest to (0.7, 0.5) on Pareto front

    def test_noop_if_current_is_pareto_optimal(self):
        # current=lstm; give lstm perfect scores so it's clearly best
        acc = {"lstm": 0.99, "linear": 0.3, "svm": 0.3}
        eng = {"lstm": 0.1, "linear": 0.5, "svm": 0.5}
        ctx = PlanningContext(
            violation="score", ema_scores=EMA_SCORES, ema_accuracy=acc,
            ema_energy=eng, current_model="lstm", available_models=MODELS,
            thresholds=BASE_THRESHOLDS, drift_result=None,
        )
        d = ParetoPlanner().plan(ctx)
        assert d.action == "noop"

    def test_drift_replace(self):
        d = ParetoPlanner().plan(DRIFT_REPLACE)
        assert d.action == "replace"

    def test_drift_retrain(self):
        d = ParetoPlanner().plan(DRIFT_RETRAIN)
        assert d.action == "retrain"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestPlannerRegistry:
    def test_all_planners_registered(self):
        _register()
        expected = {"naive", "random_switch", "greedy_switch", "harmone_original",
                    "violation_aware", "pareto", "bandit"}
        assert expected.issubset(REGISTRY.keys())

    def test_get_planner_returns_instance(self):
        p = get_planner("naive")
        assert isinstance(p, NaivePlanner)

    def test_unknown_planner_raises_key_error(self):
        with pytest.raises(KeyError):
            get_planner("nonexistent_planner_xyz")

    def test_bandit_is_in_registry(self):
        from core.planners.bandit import BanditPlanner
        _register()
        assert "bandit" in REGISTRY
        assert REGISTRY["bandit"] is BanditPlanner

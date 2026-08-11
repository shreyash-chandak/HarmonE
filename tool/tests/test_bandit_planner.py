"""tests/test_bandit_planner.py — S7 LinUCB bandit planner tests.

Follows the same patterns as test_planners.py: pure dataclass fixtures,
tmp_path for filesystem, no real knowledge/ files touched.

Groups:
  1. LinUCBBandit unit tests
  2. BanditPlanner integration tests
  3. resolve_pending() lifecycle tests
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from core.planners.base import PlanningContext, PlanDecision, _register, REGISTRY
from core.planners.bandit import (
    LinUCBBandit,
    BanditPlanner,
    build_context,
    load_or_create_bandit,
    resolve_pending,
    set_bandit_instance,
    _atomic_write,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MODELS = ["lstm", "linear", "svm"]
CONTEXT_DIM = 10 + 2 * len(MODELS)  # 16


def _make_bandit(tmp_path: Path, alpha: float = 1.0) -> LinUCBBandit:
    return LinUCBBandit(
        models=MODELS,
        context_dim=CONTEXT_DIM,
        alpha=alpha,
        w_acc=0.7,
        w_energy=0.3,
        state_path=tmp_path / "bandit_state.json",
    )


BASE_THRESHOLDS = {
    "min_score": 0.7,
    "max_energy": 0.6,
    "E_ref": 0.5,
    "beta": 0.95,
    "w_acc": 0.7,
    "w_energy": 0.3,
    "current_normalised_energy": 0.4,
    "monitoring_interval_s": 5.0,
}

EMA_SCORES = {"lstm": 0.8, "linear": 0.4, "svm": 0.6}
EMA_ACCURACY = {"lstm": 0.85, "linear": 0.4, "svm": 0.55}
EMA_ENERGY = {"lstm": 0.7, "linear": 0.2, "svm": 0.4}


def _ctx(violation="score", current_model="svm") -> PlanningContext:
    return PlanningContext(
        violation=violation,
        ema_scores=EMA_SCORES,
        ema_accuracy=EMA_ACCURACY,
        ema_energy=EMA_ENERGY,
        current_model=current_model,
        available_models=MODELS,
        thresholds=BASE_THRESHOLDS,
        drift_result=None,
    )


# ---------------------------------------------------------------------------
# Group 1 — LinUCBBandit unit tests
# ---------------------------------------------------------------------------

class TestLinUCBBanditInit:
    def test_bandit_init_creates_identity_matrices(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        for m in MODELS:
            np.testing.assert_array_almost_equal(bandit.A[m], np.eye(CONTEXT_DIM))
            np.testing.assert_array_almost_equal(bandit.b[m], np.zeros(CONTEXT_DIM))
        assert bandit.total_decisions == 0
        assert bandit.total_updates == 0

    def test_bandit_state_persists_across_instances(self, tmp_path):
        b1 = _make_bandit(tmp_path)
        context = np.ones(CONTEXT_DIM, dtype=np.float32)
        b1._load_state("test_ds")
        b1.observe_outcome(context, "lstm", reward=1.0, dataset_id="test_ds")

        b2 = _make_bandit(tmp_path)
        b2._load_state("test_ds")

        np.testing.assert_array_almost_equal(b2.A["lstm"], b1.A["lstm"])
        np.testing.assert_array_almost_equal(b2.b["lstm"], b1.b["lstm"])
        assert b2.total_updates == 1

    def test_bandit_state_keyed_by_dataset_id(self, tmp_path):
        b = _make_bandit(tmp_path)
        ctx_pems = np.full(CONTEXT_DIM, 0.1, dtype=np.float32)
        ctx_uci = np.full(CONTEXT_DIM, 0.9, dtype=np.float32)

        b._load_state("pems")
        b.observe_outcome(ctx_pems, "lstm", reward=2.0, dataset_id="pems")

        b._load_state("uci")
        # After loading "uci" (no prior state), A should be reset to identity
        np.testing.assert_array_almost_equal(b.A["lstm"], np.eye(CONTEXT_DIM))

        # Load pems back and verify it still has the update
        b._load_state("pems")
        # A[lstm] should differ from identity
        assert not np.allclose(b.A["lstm"], np.eye(CONTEXT_DIM))


class TestSelectAction:
    def test_select_action_excludes_current(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        context = np.ones(CONTEXT_DIM, dtype=np.float32)
        for _ in range(20):
            chosen = bandit.select_action(context, "lstm", ["linear", "svm"])
            assert chosen != "lstm"
            assert chosen in ["linear", "svm"]

    def test_select_action_pure_exploitation_when_alpha_zero(self, tmp_path):
        bandit = _make_bandit(tmp_path, alpha=0.0)
        # Manually set b["linear"] high so theta["linear"] @ x is high
        bandit.b["linear"] = np.ones(CONTEXT_DIM, dtype=np.float64) * 5.0
        context = np.ones(CONTEXT_DIM, dtype=np.float32)
        chosen = bandit.select_action(context, "lstm", ["linear", "svm"])
        assert chosen == "linear"

    def test_select_action_raises_on_empty_candidates(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        context = np.ones(CONTEXT_DIM, dtype=np.float32)
        with pytest.raises(ValueError):
            bandit.select_action(context, "lstm", [])


class TestComputeReward:
    def test_compute_reward_positive_on_improvement(self, tmp_path):
        b = _make_bandit(tmp_path)
        r = b.compute_reward(
            ema_before=0.5, ema_after=0.8,
            energy_before=0.6, energy_after=0.3,
            switch_cost_J=1.0,
        )
        assert r > 0.0

    def test_compute_reward_negative_on_regression(self, tmp_path):
        b = _make_bandit(tmp_path)
        r = b.compute_reward(
            ema_before=0.8, ema_after=0.5,
            energy_before=0.3, energy_after=0.7,
            switch_cost_J=1.0,
        )
        assert r < 0.0

    def test_compute_reward_clipped(self, tmp_path):
        b = _make_bandit(tmp_path)
        # Extremely positive scenario with near-zero cost → clip to 10
        r = b.compute_reward(
            ema_before=0.0, ema_after=1.0,
            energy_before=1.0, energy_after=0.0,
            switch_cost_J=1e-15,
        )
        assert r == pytest.approx(10.0)

        # Extremely negative scenario
        r2 = b.compute_reward(
            ema_before=1.0, ema_after=0.0,
            energy_before=0.0, energy_after=1.0,
            switch_cost_J=1e-15,
        )
        assert r2 == pytest.approx(-10.0)


class TestObserveOutcome:
    def test_observe_outcome_updates_matrices(self, tmp_path):
        b = _make_bandit(tmp_path)
        b._load_state("ds")
        x = np.ones(CONTEXT_DIM, dtype=np.float32)
        A_before = b.A["lstm"].copy()
        b_before = b.b["lstm"].copy()
        b.observe_outcome(x, "lstm", reward=1.0, dataset_id="ds")

        # A_new = A_old + outer(x, x)
        expected_A = A_before + np.outer(x.astype(np.float64), x.astype(np.float64))
        np.testing.assert_array_almost_equal(b.A["lstm"], expected_A)

        # b_new = b_old + r * x
        expected_b = b_before + 1.0 * x.astype(np.float64)
        np.testing.assert_array_almost_equal(b.b["lstm"], expected_b)

    def test_observe_outcome_increments_counter(self, tmp_path):
        b = _make_bandit(tmp_path)
        b._load_state("ds")
        x = np.ones(CONTEXT_DIM, dtype=np.float32)
        b.observe_outcome(x, "lstm", reward=0.5, dataset_id="ds")
        assert b.total_updates == 1

    def test_get_stats_returns_required_keys(self, tmp_path):
        b = _make_bandit(tmp_path)
        stats = b.get_stats()
        assert "total_decisions" in stats
        assert "total_updates" in stats
        assert "has_pending" in stats


# ---------------------------------------------------------------------------
# Group 2 — BanditPlanner integration tests
# ---------------------------------------------------------------------------

class TestBanditPlanner:
    def test_bandit_planner_returns_switch_decision(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        p = BanditPlanner(bandit=bandit)
        d = p.plan(_ctx(violation="score", current_model="lstm"))
        assert d.action == "switch"
        assert d.model in ["linear", "svm"]

    def test_bandit_planner_returns_noop_when_no_candidates(self, tmp_path):
        bandit = LinUCBBandit(
            models=["lstm"],
            context_dim=10 + 2,
            alpha=1.0,
            w_acc=0.7,
            w_energy=0.3,
            state_path=tmp_path / "state.json",
        )
        ctx = PlanningContext(
            violation="score",
            ema_scores={"lstm": 0.8},
            ema_accuracy={"lstm": 0.85},
            ema_energy={"lstm": 0.4},
            current_model="lstm",
            available_models=["lstm"],
            thresholds=BASE_THRESHOLDS,
            drift_result=None,
        )
        p = BanditPlanner(bandit=bandit)
        d = p.plan(ctx)
        assert d.action == "noop"

    def test_bandit_planner_returns_noop_on_no_violation(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        p = BanditPlanner(bandit=bandit)
        d = p.plan(_ctx(violation=None))
        assert d.action == "noop"

    def test_bandit_planner_returns_noop_on_drift(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        p = BanditPlanner(bandit=bandit)
        d = p.plan(_ctx(violation="drift"))
        assert d.action == "noop"

    def test_bandit_planner_decision_contains_pending_metadata(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        p = BanditPlanner(bandit=bandit)
        d = p.plan(_ctx(violation="score", current_model="lstm"))
        assert d.metadata is not None
        pending = d.metadata["bandit_pending"]
        assert "context" in pending
        assert "action" in pending
        assert "timestamp" in pending
        assert "ema_before" in pending
        assert "energy_before" in pending

    def test_bandit_planner_writes_pending_file(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        p = BanditPlanner(bandit=bandit)
        p.plan(_ctx(violation="score", current_model="lstm"))
        pending_path = tmp_path / "bandit_pending.json"
        assert pending_path.exists()

    def test_bandit_planner_fallback_on_numerical_error(self, tmp_path, monkeypatch):
        bandit = _make_bandit(tmp_path)
        monkeypatch.setattr(bandit, "select_action",
                            lambda ctx, cur, cands: (_ for _ in ()).throw(ValueError("injected")))
        p = BanditPlanner(bandit=bandit)
        # Must NOT raise; must fall back to greedy
        d = p.plan(_ctx(violation="score", current_model="svm"))
        assert d.action == "switch"
        assert d.model == "lstm"  # highest EMA among ["lstm", "linear"]

    def test_bandit_planner_increments_decision_counter(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        p = BanditPlanner(bandit=bandit)
        before = bandit.total_decisions
        p.plan(_ctx(violation="score", current_model="svm"))
        assert bandit.total_decisions == before + 1

    def test_bandit_planner_registered_in_registry(self, tmp_path):
        _register()
        assert "bandit" in REGISTRY
        assert REGISTRY["bandit"] is BanditPlanner
        # Confirm it can be instantiated with an explicit bandit (no module-level needed)
        b = _make_bandit(tmp_path)
        planner = BanditPlanner(bandit=b)
        assert isinstance(planner, BanditPlanner)

    def test_bandit_planner_no_instance_raises_runtime_error(self):
        # When module-level _bandit_instance is None and no explicit bandit, must raise
        set_bandit_instance(None)
        with pytest.raises(RuntimeError, match="no bandit instance"):
            BanditPlanner()


# ---------------------------------------------------------------------------
# Group 3 — resolve_pending() lifecycle tests
# ---------------------------------------------------------------------------

class TestResolvePending:
    def _write_pending(self, path: Path, elapsed_s: float = 10.0) -> dict:
        context = np.ones(CONTEXT_DIM, dtype=np.float32)
        pending = {
            "context": context.tolist(),
            "action": "lstm",
            "ema_before": 0.6,
            "energy_before": 0.5,
            "switch_cost_J": 0.0,
            "timestamp": time.time() - elapsed_s,
        }
        _atomic_write(path, pending)
        return pending

    def test_resolve_pending_too_soon_skips(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        bandit._load_state("ds")
        pending_path = tmp_path / "bandit_pending.json"
        self._write_pending(pending_path, elapsed_s=0.1)

        mape_info = {"ema_scores": {"lstm": 0.7}, "current_normalised_energy": 0.4}
        thresholds = {**BASE_THRESHOLDS, "monitoring_interval_s": 5.0}

        result = resolve_pending(bandit, pending_path, thresholds, mape_info)
        assert result is False
        assert pending_path.exists()  # untouched
        assert bandit.total_updates == 0

    def test_resolve_pending_resolves_after_interval(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        bandit._load_state("ds")
        pending_path = tmp_path / "bandit_pending.json"
        self._write_pending(pending_path, elapsed_s=10.0)

        mape_info = {"ema_scores": {"lstm": 0.75}, "current_normalised_energy": 0.35}
        thresholds = {**BASE_THRESHOLDS, "monitoring_interval_s": 5.0}

        result = resolve_pending(bandit, pending_path, thresholds, mape_info)
        assert result is True
        assert not pending_path.exists()  # deleted
        assert bandit.total_updates == 1

    def test_resolve_pending_handles_corrupt_json(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        pending_path = tmp_path / "bandit_pending.json"
        pending_path.write_text("{ not valid json !!!")

        mape_info: dict = {}
        thresholds = {**BASE_THRESHOLDS}

        # Must not raise
        result = resolve_pending(bandit, pending_path, thresholds, mape_info)
        assert result is True
        assert not pending_path.exists()  # deleted after error


# ---------------------------------------------------------------------------
# Group 4 — build_context smoke test
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_build_context_correct_dimension(self):
        ctx = _ctx()
        vec = build_context(ctx, MODELS)
        assert vec.shape == (CONTEXT_DIM,)
        assert vec.dtype == np.float32

    def test_build_context_violation_onehot(self):
        vec_score = build_context(_ctx(violation="score"), MODELS)
        vec_energy = build_context(_ctx(violation="energy"), MODELS)
        vec_none = build_context(_ctx(violation=None), MODELS)
        # score=1 at index 0
        assert vec_score[0] == 1.0
        assert vec_score[1] == 0.0
        # energy=1 at index 1
        assert vec_energy[0] == 0.0
        assert vec_energy[1] == 1.0
        # no violation → all zeros
        assert vec_none[0] == 0.0
        assert vec_none[1] == 0.0
        assert vec_none[2] == 0.0

    def test_build_context_values_bounded(self):
        ctx = _ctx()
        vec = build_context(ctx, MODELS)
        # All values in [-1, 1]
        assert np.all(vec >= -1.0)
        assert np.all(vec <= 1.0)


# ---------------------------------------------------------------------------
# Group 5 — load_or_create_bandit factory
# ---------------------------------------------------------------------------

class TestLoadOrCreateBandit:
    def test_factory_correct_context_dim(self, tmp_path):
        thresholds = {"bandit_alpha": 0.5, "beta": 0.95}
        b = load_or_create_bandit(thresholds, MODELS, tmp_path)
        assert b.context_dim == CONTEXT_DIM
        assert b.alpha == pytest.approx(0.5)

    def test_factory_loads_state_when_dataset_id_given(self, tmp_path):
        thresholds = {"bandit_alpha": 1.0, "beta": 0.95}
        b1 = load_or_create_bandit(thresholds, MODELS, tmp_path)
        b1._load_state("myds")
        x = np.ones(CONTEXT_DIM, dtype=np.float32)
        b1.observe_outcome(x, "lstm", reward=2.0, dataset_id="myds")

        b2 = load_or_create_bandit(thresholds, MODELS, tmp_path, dataset_id="myds")
        assert b2.total_updates == 1

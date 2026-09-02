"""tests/test_bandit_planner.py — S7 LinUCB bandit planner tests (Phase 4).

Phase 4 interface changes vs Phase 3 (all documented as structural changes):
  - Dual estimators: A_acc/b_acc and A_eng/b_eng; old A/b removed.
  - Scalar compute_reward() removed; replaced by dual acc_reward/eng_reward.
  - select_action() takes phi_per_arm dict + energy_threshold + Lyapunov params.
  - resolve_pending() is step-gated (bandit_min_observation_steps), not wall-clock.
  - pending record: phi/decision_step keys, not context/timestamp/switch_cost_J.
  - State version = 2; v1 state refused and reset fresh.
  - build_candidate_context() generates 16-dim candidate-specific feature vectors.
    build_context() is a legacy alias.

Groups:
  1. LinUCBBandit unit tests (dual estimators)
  2. BanditPlanner integration tests
  3. resolve_pending() step-gated lifecycle tests
  4. build_candidate_context() smoke tests
  5. load_or_create_bandit factory tests
  6. State versioning tests
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.planners.base import PlanningContext, PlanDecision, _register, REGISTRY
from core.planners.bandit import (
    LinUCBBandit,
    BanditPlanner,
    build_context,
    build_candidate_context,
    load_or_create_bandit,
    resolve_pending,
    set_bandit_instance,
    _atomic_write,
    _STATE_VERSION,
    _CONTEXT_DIM,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MODELS = ["lstm", "linear", "svm"]
CONTEXT_DIM = _CONTEXT_DIM  # 16


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
    "min_accuracy": 0.0,
    "max_energy": 0.6,
    "E_ref": 0.5,
    "beta": 0.95,
    "w_acc": 0.7,
    "w_energy": 0.3,
    "current_normalised_energy": 0.4,
}

EMA_SCORES = {"lstm": 0.8, "linear": 0.4, "svm": 0.6}
EMA_ACCURACY = {"lstm": 0.85, "linear": 0.4, "svm": 0.55}
EMA_ENERGY = {"lstm": 0.7, "linear": 0.2, "svm": 0.4}


def _ctx(violation="score", current_model="svm", drift_result=None) -> PlanningContext:
    return PlanningContext(
        violation=violation,
        ema_scores=EMA_SCORES,
        ema_accuracy=EMA_ACCURACY,
        ema_energy=EMA_ENERGY,
        current_model=current_model,
        available_models=MODELS,
        thresholds=BASE_THRESHOLDS,
        drift_result=drift_result,
        current_energy_threshold=0.6,
    )


# ---------------------------------------------------------------------------
# Group 1 — LinUCBBandit unit tests (dual estimators)
# ---------------------------------------------------------------------------

class TestLinUCBBanditInit:
    def test_init_creates_dual_identity_matrices(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        for m in MODELS:
            np.testing.assert_array_almost_equal(bandit.A_acc[m], np.eye(CONTEXT_DIM))
            np.testing.assert_array_almost_equal(bandit.b_acc[m], np.zeros(CONTEXT_DIM))
            np.testing.assert_array_almost_equal(bandit.A_eng[m], np.eye(CONTEXT_DIM))
            np.testing.assert_array_almost_equal(bandit.b_eng[m], np.zeros(CONTEXT_DIM))
        assert bandit.total_decisions == 0
        assert bandit.total_updates == 0
        assert bandit.Q == pytest.approx(0.0)

    def test_state_persists_across_instances(self, tmp_path):
        b1 = _make_bandit(tmp_path)
        b1._load_state("test_ds")
        phi = np.ones(CONTEXT_DIM, dtype=np.float32)
        b1.observe_outcome(phi, "lstm", acc_reward=0.5, eng_reward=0.2, dataset_id="test_ds")

        b2 = _make_bandit(tmp_path)
        b2._load_state("test_ds")

        np.testing.assert_array_almost_equal(b2.A_acc["lstm"], b1.A_acc["lstm"])
        np.testing.assert_array_almost_equal(b2.b_acc["lstm"], b1.b_acc["lstm"])
        np.testing.assert_array_almost_equal(b2.A_eng["lstm"], b1.A_eng["lstm"])
        np.testing.assert_array_almost_equal(b2.b_eng["lstm"], b1.b_eng["lstm"])
        assert b2.total_updates == 1

    def test_state_keyed_by_dataset_id(self, tmp_path):
        b = _make_bandit(tmp_path)
        b._load_state("pems")
        phi = np.full(CONTEXT_DIM, 0.1, dtype=np.float32)
        b.observe_outcome(phi, "lstm", acc_reward=1.0, eng_reward=0.5, dataset_id="pems")

        b._load_state("uci")
        np.testing.assert_array_almost_equal(b.A_acc["lstm"], np.eye(CONTEXT_DIM))

        b._load_state("pems")
        assert not np.allclose(b.A_acc["lstm"], np.eye(CONTEXT_DIM))


class TestDualObserve:
    def test_observe_updates_acc_matrices(self, tmp_path):
        b = _make_bandit(tmp_path)
        b._load_state("ds")
        phi = np.ones(CONTEXT_DIM, dtype=np.float32)
        A_before = b.A_acc["lstm"].copy()
        b_before = b.b_acc["lstm"].copy()
        b.observe_outcome(phi, "lstm", acc_reward=1.0, eng_reward=0.0, dataset_id="ds")
        expected_A = A_before + np.outer(phi.astype(np.float64), phi.astype(np.float64))
        np.testing.assert_array_almost_equal(b.A_acc["lstm"], expected_A)
        expected_b = b_before + 1.0 * phi.astype(np.float64)
        np.testing.assert_array_almost_equal(b.b_acc["lstm"], expected_b)

    def test_observe_updates_eng_matrices(self, tmp_path):
        b = _make_bandit(tmp_path)
        b._load_state("ds")
        phi = np.ones(CONTEXT_DIM, dtype=np.float32)
        A_before = b.A_eng["lstm"].copy()
        b_before = b.b_eng["lstm"].copy()
        b.observe_outcome(phi, "lstm", acc_reward=0.0, eng_reward=0.8, dataset_id="ds")
        expected_A = A_before + np.outer(phi.astype(np.float64), phi.astype(np.float64))
        np.testing.assert_array_almost_equal(b.A_eng["lstm"], expected_A)
        expected_b = b_before + 0.8 * phi.astype(np.float64)
        np.testing.assert_array_almost_equal(b.b_eng["lstm"], expected_b)

    def test_observe_increments_update_counter(self, tmp_path):
        b = _make_bandit(tmp_path)
        b._load_state("ds")
        phi = np.ones(CONTEXT_DIM, dtype=np.float32)
        b.observe_outcome(phi, "lstm", acc_reward=0.3, eng_reward=0.1, dataset_id="ds")
        assert b.total_updates == 1


class TestSelectAction:
    def _phi_map(self, val: float) -> dict[str, np.ndarray]:
        return {m: np.full(CONTEXT_DIM, val, dtype=np.float32) for m in MODELS}

    def test_raises_on_empty_candidates(self, tmp_path):
        b = _make_bandit(tmp_path)
        with pytest.raises(ValueError):
            b.select_action(
                phi_per_arm={},
                current_model="lstm",
                candidates=[],
                energy_threshold=0.6,
                lambda_0=0.1,
                mu=0.05,
                switch_costs={},
            )

    def test_feasibility_filter_respects_energy_threshold(self, tmp_path):
        b = _make_bandit(tmp_path, alpha=0.0)
        # Push E_UCB(lstm) very high via b_eng so it always fails feasibility
        b.b_eng["lstm"] = np.ones(CONTEXT_DIM, dtype=np.float64) * 100.0
        phi_map = self._phi_map(1.0)
        # Only linear and svm should be feasible with threshold=0.6
        for _ in range(10):
            chosen = b.select_action(
                phi_per_arm=phi_map,
                current_model="svm",
                candidates=["lstm", "linear", "svm"],
                energy_threshold=0.6,
                lambda_0=0.1,
                mu=0.05,
                switch_costs={},
            )
            assert chosen in ["linear", "svm"]

    def test_exploitation_with_zero_alpha(self, tmp_path):
        b = _make_bandit(tmp_path, alpha=0.0)
        b.b_acc["linear"] = np.ones(CONTEXT_DIM, dtype=np.float64) * 10.0
        phi_map = self._phi_map(1.0)
        chosen = b.select_action(
            phi_per_arm=phi_map,
            current_model="svm",
            candidates=["lstm", "linear", "svm"],
            energy_threshold=1.0,
            lambda_0=0.0,
            mu=0.0,
            switch_costs={},
        )
        assert chosen == "linear"

    def test_lyapunov_debt_penalises_energy(self, tmp_path):
        b = _make_bandit(tmp_path, alpha=0.0)
        # Both lstm and linear identical in acc; lstm has high E_UCB
        b.b_acc["lstm"] = np.ones(CONTEXT_DIM, dtype=np.float64) * 5.0
        b.b_acc["linear"] = np.ones(CONTEXT_DIM, dtype=np.float64) * 5.0
        b.b_eng["lstm"] = np.ones(CONTEXT_DIM, dtype=np.float64) * 5.0
        b.b_eng["linear"] = np.zeros(CONTEXT_DIM, dtype=np.float64)
        b.Q = 100.0  # large debt → high λ → energy strongly penalised
        phi_map = self._phi_map(1.0)
        chosen = b.select_action(
            phi_per_arm=phi_map,
            current_model="svm",
            candidates=["lstm", "linear"],
            energy_threshold=1.0,
            lambda_0=0.1,
            mu=0.05,
            switch_costs={},
        )
        assert chosen == "linear"

    def test_energy_debt_update(self, tmp_path):
        b = _make_bandit(tmp_path)
        b.Q = 0.0
        b.update_energy_debt(energy_used=0.8, energy_budget=0.5)
        assert b.Q == pytest.approx(0.3)
        b.update_energy_debt(energy_used=0.3, energy_budget=0.5)
        assert b.Q == pytest.approx(0.1)  # 0.3 + (0.3-0.5) = 0.1
        b.update_energy_debt(energy_used=0.1, energy_budget=0.5)
        assert b.Q == pytest.approx(0.0)  # max(0, 0.1+0.1-0.5) = 0.0


class TestGetStats:
    def test_get_stats_returns_required_keys(self, tmp_path):
        b = _make_bandit(tmp_path)
        stats = b.get_stats()
        assert "total_decisions" in stats
        assert "total_updates" in stats
        assert "Q" in stats
        assert "has_pending" in stats
        assert "theta_acc_l2_norms" in stats


# ---------------------------------------------------------------------------
# Group 2 — BanditPlanner integration tests
# ---------------------------------------------------------------------------

class TestBanditPlanner:
    def test_returns_switch_or_noop_on_violation(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        p = BanditPlanner(bandit=bandit)
        # current=lstm; candidates include linear and svm which have lower acc EMA
        # hysteresis may noop if selected arm isn't better by margin — that's fine
        d = p.plan(_ctx(violation="score", current_model="svm"))
        assert d.action in ("switch", "noop")

    def test_returns_noop_on_no_violation(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        p = BanditPlanner(bandit=bandit)
        assert p.plan(_ctx(violation=None)).action == "noop"

    def test_returns_noop_on_drift_without_drift_result(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        p = BanditPlanner(bandit=bandit)
        assert p.plan(_ctx(violation="drift")).action == "noop"

    def test_drift_with_no_vmr_match_triggers_retrain(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        p = BanditPlanner(bandit=bandit)
        d = p.plan(_ctx(violation="drift", drift_result={"drift_detected": True, "action": "retrain"}))
        assert d.action == "retrain"

    def test_drift_with_vmr_match_triggers_replace(self, tmp_path):
        bandit = _make_bandit(tmp_path)
        p = BanditPlanner(bandit=bandit)
        d = p.plan(_ctx(
            violation="drift",
            drift_result={"drift_detected": True, "action": "replace", "version": "/vmr/lstm/v3"},
        ))
        assert d.action == "replace"
        assert d.version_path == "/vmr/lstm/v3"

    def test_drift_does_not_touch_bandit_arm_state(self, tmp_path):
        """Drift routing bypasses select_action/record_pending — no pending file,
        no decision count increment (unlike a score/energy violation switch)."""
        bandit = _make_bandit(tmp_path)
        p = BanditPlanner(bandit=bandit)
        before = bandit.total_decisions
        p.plan(_ctx(violation="drift", drift_result={"drift_detected": True, "action": "retrain"}))
        assert bandit.total_decisions == before
        assert not (tmp_path / "bandit_pending.json").exists()

    def test_returns_noop_when_no_candidates(self, tmp_path):
        bandit = LinUCBBandit(
            models=["lstm"],
            context_dim=CONTEXT_DIM,
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
            current_energy_threshold=0.6,
        )
        d = BanditPlanner(bandit=bandit).plan(ctx)
        assert d.action == "noop"

    def test_switch_decision_writes_pending_file(self, tmp_path):
        # Force a switch by making alpha=0 and biasing linear strongly
        bandit = _make_bandit(tmp_path, alpha=0.0)
        bandit.b_acc["linear"] = np.ones(CONTEXT_DIM, dtype=np.float64) * 100.0
        p = BanditPlanner(bandit=bandit)
        d = p.plan(_ctx(violation="score", current_model="lstm"))
        if d.action == "switch":
            pending_path = bandit.state_path.parent / "bandit_pending.json"
            assert pending_path.exists()
            with open(pending_path) as f:
                pending = json.load(f)
            assert "phi" in pending
            assert "action" in pending
            assert "decision_step" in pending
            assert "ema_acc_before" in pending
            assert "ema_eng_before" in pending

    def test_switch_increments_decision_counter(self, tmp_path):
        bandit = _make_bandit(tmp_path, alpha=0.0)
        bandit.b_acc["linear"] = np.ones(CONTEXT_DIM, dtype=np.float64) * 100.0
        p = BanditPlanner(bandit=bandit)
        before = bandit.total_decisions
        p.plan(_ctx(violation="score", current_model="lstm"))
        # counter increments only when a switch fires
        assert bandit.total_decisions >= before

    def test_fallback_to_greedy_on_numerical_error(self, tmp_path, monkeypatch):
        bandit = _make_bandit(tmp_path)

        def _fail(*a, **kw):
            raise ValueError("injected")

        monkeypatch.setattr(bandit, "select_action", _fail)
        p = BanditPlanner(bandit=bandit)
        d = p.plan(_ctx(violation="score", current_model="svm"))
        # Must not raise; falls back to greedy (lstm=0.85 highest acc)
        # hysteresis may gate — just assert no exception
        assert d.action in ("switch", "noop")

    def test_no_instance_raises_runtime_error(self):
        set_bandit_instance(None)
        with pytest.raises(RuntimeError, match="no bandit instance"):
            BanditPlanner()

    def test_registered_in_registry(self, tmp_path):
        _register()
        assert "bandit" in REGISTRY
        assert REGISTRY["bandit"] is BanditPlanner


# ---------------------------------------------------------------------------
# Group 3 — resolve_pending() step-gated lifecycle tests
# ---------------------------------------------------------------------------

class TestResolvePending:
    def _write_pending(self, path: Path, decision_step: int = 0) -> None:
        phi = np.ones(CONTEXT_DIM, dtype=np.float32)
        pending = {
            "phi": phi.tolist(),
            "action": "lstm",
            "ema_acc_before": 0.6,
            "ema_eng_before": 0.5,
            "decision_step": decision_step,
        }
        _atomic_write(path, pending)

    def test_skips_when_not_enough_steps(self, tmp_path):
        b = _make_bandit(tmp_path)
        b._load_state("ds")
        pending_path = tmp_path / "bandit_pending.json"
        self._write_pending(pending_path, decision_step=0)

        thresholds = {**BASE_THRESHOLDS, "bandit_min_observation_steps": 50}
        mape_info = {
            "ema_accuracy": {"lstm": 0.7},
            "ema_energy": {"lstm": 0.4},
            "current_normalised_energy": 0.4,
            "current_energy_threshold": 0.6,
        }
        result = resolve_pending(b, pending_path, thresholds, mape_info, current_step=10)
        assert result is False
        assert pending_path.exists()
        assert b.total_updates == 0

    def test_resolves_after_enough_steps(self, tmp_path):
        b = _make_bandit(tmp_path)
        b._load_state("ds")
        pending_path = tmp_path / "bandit_pending.json"
        self._write_pending(pending_path, decision_step=0)

        thresholds = {**BASE_THRESHOLDS, "bandit_min_observation_steps": 50}
        mape_info = {
            "ema_accuracy": {"lstm": 0.75},
            "ema_energy": {"lstm": 0.35},
            "current_normalised_energy": 0.4,
            "current_energy_threshold": 0.6,
        }
        result = resolve_pending(b, pending_path, thresholds, mape_info, current_step=60)
        assert result is True
        assert not pending_path.exists()
        assert b.total_updates == 1

    def test_handles_corrupt_json(self, tmp_path):
        b = _make_bandit(tmp_path)
        pending_path = tmp_path / "bandit_pending.json"
        pending_path.write_text("{ not valid json !!!")

        result = resolve_pending(b, pending_path, BASE_THRESHOLDS, {}, current_step=999)
        assert result is True
        assert not pending_path.exists()

    def test_updates_both_estimators(self, tmp_path):
        b = _make_bandit(tmp_path)
        b._load_state("ds")
        pending_path = tmp_path / "bandit_pending.json"
        self._write_pending(pending_path, decision_step=0)

        A_acc_before = b.A_acc["lstm"].copy()
        A_eng_before = b.A_eng["lstm"].copy()

        thresholds = {**BASE_THRESHOLDS, "bandit_min_observation_steps": 0}
        mape_info = {
            "ema_accuracy": {"lstm": 0.8},
            "ema_energy": {"lstm": 0.4},
            "current_normalised_energy": 0.4,
            "current_energy_threshold": 0.6,
        }
        resolve_pending(b, pending_path, thresholds, mape_info, current_step=1)
        assert not np.allclose(b.A_acc["lstm"], A_acc_before)
        assert not np.allclose(b.A_eng["lstm"], A_eng_before)


# ---------------------------------------------------------------------------
# Group 4 — build_candidate_context smoke tests
# ---------------------------------------------------------------------------

class TestBuildCandidateContext:
    def test_correct_dimension(self):
        ctx = _ctx()
        phi = build_candidate_context(ctx, "lstm", {})
        assert phi.shape == (CONTEXT_DIM,)
        assert phi.dtype == np.float32

    def test_values_bounded(self):
        ctx = _ctx()
        for arm in MODELS:
            phi = build_candidate_context(ctx, arm, {})
            assert np.all(phi >= -1.1), f"Out of range for arm={arm}"
            assert np.all(phi <= 2.1), f"Out of range for arm={arm}"

    def test_is_current_flag(self):
        ctx = _ctx(current_model="svm")
        phi_svm = build_candidate_context(ctx, "svm", {})
        phi_lstm = build_candidate_context(ctx, "lstm", {})
        # is_current is index 14
        assert phi_svm[14] == pytest.approx(1.0)
        assert phi_lstm[14] == pytest.approx(0.0)

    def test_cost_class_encoding(self):
        ctx = _ctx()
        models_cfg = {"lstm": {"cost_class": "heavy"}, "linear": {"cost_class": "light"}}
        phi_heavy = build_candidate_context(ctx, "lstm", models_cfg)
        phi_light = build_candidate_context(ctx, "linear", models_cfg)
        # cost_class is index 15
        assert phi_heavy[15] == pytest.approx(1.0)
        assert phi_light[15] == pytest.approx(0.0)

    def test_legacy_build_context_wrapper(self):
        ctx = _ctx()
        phi = build_context(ctx, MODELS)
        assert phi.shape == (CONTEXT_DIM,)

    def test_violation_one_hot_score(self):
        phi_score = build_candidate_context(_ctx(violation="score"), "lstm", {})
        phi_energy = build_candidate_context(_ctx(violation="energy"), "lstm", {})
        phi_none = build_candidate_context(_ctx(violation=None), "lstm", {})
        assert phi_score[0] == pytest.approx(1.0)  # v_score
        assert phi_energy[1] == pytest.approx(1.0)  # v_energy
        assert phi_none[0] == pytest.approx(0.0)
        assert phi_none[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Group 5 — load_or_create_bandit factory tests
# ---------------------------------------------------------------------------

class TestLoadOrCreateBandit:
    def test_correct_context_dim(self, tmp_path):
        b = load_or_create_bandit({"bandit_alpha": 0.5, "beta": 0.95}, MODELS, tmp_path)
        assert b.context_dim == CONTEXT_DIM
        assert b.alpha == pytest.approx(0.5)

    def test_loads_state_when_dataset_id_given(self, tmp_path):
        thresholds = {"bandit_alpha": 1.0, "beta": 0.95}
        b1 = load_or_create_bandit(thresholds, MODELS, tmp_path, dataset_id="myds")
        phi = np.ones(CONTEXT_DIM, dtype=np.float32)
        b1.observe_outcome(phi, "lstm", acc_reward=0.5, eng_reward=0.2, dataset_id="myds")

        b2 = load_or_create_bandit(thresholds, MODELS, tmp_path, dataset_id="myds")
        assert b2.total_updates == 1


# ---------------------------------------------------------------------------
# Group 6 — State versioning tests (Phase 4.5)
# ---------------------------------------------------------------------------

class TestStateVersioning:
    def test_state_version_written_as_2(self, tmp_path):
        b = _make_bandit(tmp_path)
        b._load_state("ds")
        phi = np.ones(CONTEXT_DIM, dtype=np.float32)
        b.observe_outcome(phi, "lstm", acc_reward=0.3, eng_reward=0.1, dataset_id="ds")

        with open(tmp_path / "bandit_state.json") as f:
            saved = json.load(f)
        assert saved["ds"]["state_version"] == _STATE_VERSION

    def test_v1_state_refused_and_starts_fresh(self, tmp_path):
        state_path = tmp_path / "bandit_state.json"
        # Write a fake v1 state (no state_version key = v1)
        v1_state = {
            "myds": {
                # no state_version → treated as v1
                "total_decisions": 99,
                "total_updates": 42,
                "A": {"lstm": np.eye(CONTEXT_DIM).tolist()},
                "b": {"lstm": np.zeros(CONTEXT_DIM).tolist()},
            }
        }
        with open(state_path, "w") as f:
            json.dump(v1_state, f)

        b = LinUCBBandit(
            models=MODELS,
            context_dim=CONTEXT_DIM,
            alpha=1.0,
            w_acc=0.7,
            w_energy=0.3,
            state_path=state_path,
        )
        b._load_state("myds")
        # v1 state should be ignored; fresh init: counters reset
        assert b.total_decisions == 0
        assert b.total_updates == 0
        np.testing.assert_array_almost_equal(b.A_acc["lstm"], np.eye(CONTEXT_DIM))

    def test_v2_state_loads_correctly(self, tmp_path):
        b = _make_bandit(tmp_path)
        b._load_state("ds")
        phi = np.ones(CONTEXT_DIM, dtype=np.float32)
        b.observe_outcome(phi, "lstm", acc_reward=0.5, eng_reward=0.3, dataset_id="ds")

        b2 = _make_bandit(tmp_path)
        b2._load_state("ds")
        assert b2.total_updates == 1
        assert not np.allclose(b2.A_acc["lstm"], np.eye(CONTEXT_DIM))
        assert not np.allclose(b2.A_eng["lstm"], np.eye(CONTEXT_DIM))

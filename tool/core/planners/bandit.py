"""
core/planners/bandit.py — S7: LinUCB contextual bandit planner (Phase 4 redesign).

Phase 4 changes from the v1 scalar-reward implementation:

  4.1 Dual estimators: separate A_acc/b_acc and A_eng/b_eng matrices per arm,
      updated independently from per-interval accuracy and energy observations.
      The old scalar reward (acc_delta / switch_cost_J with switch_cost_J=0.0)
      is retired.

  4.2 Candidate-specific feature vectors φ(x, m): each arm is evaluated with
      a 16-dim feature vector that mixes global context with arm-specific
      information (acc_m, eng_m, ratios to current, cost_class, is_current).
      Context dim is fixed at 16 regardless of the number of models.

  4.3 Constrained selection:
        (a) Feasibility filter: E_UCB(m) ≤ current_energy_threshold
        (b) Score: argmax( A_UCB(m) − λ(Q) · E_UCB(m) − switch_penalty(m) )
        where Q is the Lyapunov energy-debt accumulator and
        λ(Q) = bandit_lambda_0 + bandit_mu * Q (grows with debt).

  4.4 Reward attribution fix: pending records carry decision_step; reward is
      resolved when current_step − decision_step ≥ bandit_min_observation_steps
      (config key, not wall-clock).

  4.5 State versioning: bandit_state.json now carries state_version=2.
      Loading v1 state emits a warning and starts fresh for that dataset.

Planners stay pure: all file I/O goes through harness-mediated helpers and
run_experiment.py's bandit_pending.json contract.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from .base import Planner, PlanningContext, PlanDecision, REGISTRY
from .hysteresis import should_switch

logger = logging.getLogger(__name__)

_STATE_VERSION = 2
_CONTEXT_DIM = 16          # fixed; independent of model count
_DEFAULT_ALPHA = 1.0
_DEFAULT_LAMBDA_0 = 0.1
_DEFAULT_MU = 0.05
_DEFAULT_MIN_OBS_STEPS = 50

# Module-level bandit instance — set once by manage.py / run_experiment.py
_bandit_instance: "LinUCBBandit | None" = None


def set_bandit_instance(bandit: "LinUCBBandit | None") -> None:
    global _bandit_instance
    _bandit_instance = bandit


# ── Atomic write ──────────────────────────────────────────────────────────────

def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, str(path))


# ── Dual-estimator LinUCB per arm ─────────────────────────────────────────────

class LinUCBBandit:
    """
    LinUCB bandit with separate accuracy and energy estimators per arm.

    State persists to knowledge/bandit_state.json keyed by dataset_id.
    State version 2 — v1 scalar-reward state is not compatible.
    Learning accumulates across runs (never reset by run_reset.py).
    """

    def __init__(
        self,
        models: list[str],
        context_dim: int,
        alpha: float,
        w_acc: float,
        w_energy: float,
        state_path: str | Path,
    ) -> None:
        self.models = list(models)
        self.context_dim = context_dim
        self.alpha = alpha
        self.w_acc = w_acc
        self.w_energy = w_energy
        self.state_path = Path(state_path)

        # Dual estimators per arm
        self.A_acc: dict[str, np.ndarray] = {}
        self.b_acc: dict[str, np.ndarray] = {}
        self.A_eng: dict[str, np.ndarray] = {}
        self.b_eng: dict[str, np.ndarray] = {}

        # Energy-debt accumulator (Lyapunov)
        self.Q: float = 0.0

        self.total_decisions: int = 0
        self.total_updates: int = 0
        self._current_dataset_id: str = "unknown"

        self._init_arms()

    def _init_arms(self) -> None:
        d = self.context_dim
        for m in self.models:
            self.A_acc[m] = np.eye(d, dtype=np.float64)
            self.b_acc[m] = np.zeros(d, dtype=np.float64)
            self.A_eng[m] = np.eye(d, dtype=np.float64)
            self.b_eng[m] = np.zeros(d, dtype=np.float64)

    def _ensure_arm(self, m: str) -> None:
        d = self.context_dim
        if m not in self.A_acc:
            self.A_acc[m] = np.eye(d, dtype=np.float64)
            self.b_acc[m] = np.zeros(d, dtype=np.float64)
            self.A_eng[m] = np.eye(d, dtype=np.float64)
            self.b_eng[m] = np.zeros(d, dtype=np.float64)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_state(self, dataset_id: str) -> None:
        self._current_dataset_id = dataset_id
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path) as f:
                all_state = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[Bandit] Cannot read state (%s); starting fresh.", exc)
            return

        if dataset_id not in all_state:
            self._init_arms()
            return

        ds = all_state[dataset_id]

        # Version check (Phase 4.5)
        if ds.get("state_version", 1) != _STATE_VERSION:
            logger.warning(
                "[Bandit] State for '%s' is version %d (expected %d); starting fresh.",
                dataset_id, ds.get("state_version", 1), _STATE_VERSION,
            )
            self._init_arms()
            return

        self.total_decisions = ds.get("total_decisions", 0)
        self.total_updates = ds.get("total_updates", 0)
        self.Q = float(ds.get("Q", 0.0))

        for m in self.models:
            if m in ds.get("A_acc", {}):
                self.A_acc[m] = np.array(ds["A_acc"][m], dtype=np.float64)
                self.b_acc[m] = np.array(ds["b_acc"][m], dtype=np.float64)
            if m in ds.get("A_eng", {}):
                self.A_eng[m] = np.array(ds["A_eng"][m], dtype=np.float64)
                self.b_eng[m] = np.array(ds["b_eng"][m], dtype=np.float64)

        logger.info(
            "[Bandit] Loaded v2 state for '%s': %d decisions, %d updates, Q=%.4f.",
            dataset_id, self.total_decisions, self.total_updates, self.Q,
        )

    def _save_state(self, dataset_id: str) -> None:
        try:
            all_state: dict = {}
            if self.state_path.exists():
                with open(self.state_path) as f:
                    all_state = json.load(f)
        except (json.JSONDecodeError, OSError):
            all_state = {}

        arms = [m for m in self.models if m in self.A_acc]
        all_state[dataset_id] = {
            "state_version": _STATE_VERSION,
            "total_decisions": self.total_decisions,
            "total_updates": self.total_updates,
            "Q": float(self.Q),
            "A_acc": {m: self.A_acc[m].tolist() for m in arms},
            "b_acc": {m: self.b_acc[m].tolist() for m in arms},
            "A_eng": {m: self.A_eng[m].tolist() for m in arms},
            "b_eng": {m: self.b_eng[m].tolist() for m in arms},
        }
        _atomic_write(self.state_path, all_state)

    # ── UCB computation ───────────────────────────────────────────────────────

    def _ucb(self, phi: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
        x = phi.astype(np.float64)
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.eye(len(b))
        theta = A_inv @ b
        expected = float(theta @ x)
        uncertainty = self.alpha * float(np.sqrt(max(0.0, x @ A_inv @ x)))
        return expected + uncertainty

    def acc_ucb(self, phi: np.ndarray, arm: str) -> float:
        self._ensure_arm(arm)
        return self._ucb(phi, self.A_acc[arm], self.b_acc[arm])

    def eng_ucb(self, phi: np.ndarray, arm: str) -> float:
        self._ensure_arm(arm)
        return self._ucb(phi, self.A_eng[arm], self.b_eng[arm])

    # ── Constrained selection (Phase 4.3) ─────────────────────────────────────

    def select_action(
        self,
        phi_per_arm: dict[str, np.ndarray],
        current_model: str,
        candidates: list[str],
        energy_threshold: float,
        lambda_0: float,
        mu: float,
        switch_costs: dict[str, float],
    ) -> str:
        if not candidates:
            raise ValueError("No candidates to select from.")

        lam = lambda_0 + mu * max(0.0, self.Q)

        a_ucb = {m: self.acc_ucb(phi_per_arm[m], m) for m in candidates}
        e_ucb = {m: self.eng_ucb(phi_per_arm[m], m) for m in candidates}

        # Feasibility filter
        feasible = [m for m in candidates if e_ucb[m] <= energy_threshold]
        pool = feasible if feasible else sorted(candidates, key=lambda m: e_ucb[m])[:1]

        # Score: maximize acc_ucb - λ·eng_ucb - switch_cost
        scores = {
            m: a_ucb[m] - lam * e_ucb[m] - (switch_costs.get(m, 0.0) if m != current_model else 0.0)
            for m in pool
        }
        logger.debug("[Bandit] UCB scores: %s", scores)
        return max(scores, key=lambda m: scores[m])

    # ── Pending record ────────────────────────────────────────────────────────

    def record_pending(
        self,
        phi: np.ndarray,
        action: str,
        ema_acc_before: float,
        ema_eng_before: float,
        decision_step: int,
    ) -> dict:
        return {
            "phi": phi.tolist(),
            "action": action,
            "ema_acc_before": float(ema_acc_before),
            "ema_eng_before": float(ema_eng_before),
            "decision_step": int(decision_step),
        }

    # ── Dual reward update (Phase 4.1) ────────────────────────────────────────

    def observe_outcome(
        self,
        phi: np.ndarray,
        action: str,
        acc_reward: float,
        eng_reward: float,
        dataset_id: str | None = None,
    ) -> None:
        """Update A_acc/b_acc and A_eng/b_eng for the selected arm."""
        self._ensure_arm(action)
        x = phi.astype(np.float64)
        self.A_acc[action] = self.A_acc[action] + np.outer(x, x)
        self.b_acc[action] = self.b_acc[action] + acc_reward * x
        self.A_eng[action] = self.A_eng[action] + np.outer(x, x)
        self.b_eng[action] = self.b_eng[action] + eng_reward * x
        self.total_updates += 1
        self._save_state(dataset_id or self._current_dataset_id)

    def update_energy_debt(self, energy_used: float, energy_budget: float) -> None:
        """Update Lyapunov energy-debt accumulator Q."""
        self.Q = max(0.0, self.Q + energy_used - energy_budget)

    def get_stats(self) -> dict:
        theta_norms: dict[str, float] = {}
        for m in self.models:
            if m in self.A_acc:
                try:
                    theta = np.linalg.inv(self.A_acc[m]) @ self.b_acc[m]
                    theta_norms[m] = float(np.linalg.norm(theta))
                except np.linalg.LinAlgError:
                    theta_norms[m] = 0.0
        pending_path = self.state_path.parent / "bandit_pending.json"
        return {
            "total_decisions": self.total_decisions,
            "total_updates": self.total_updates,
            "Q": self.Q,
            "has_pending": pending_path.exists(),
            "theta_acc_l2_norms": theta_norms,
        }


# ── Candidate-specific feature vector φ(x, m) ── Phase 4.2 ──────────────────

_COST_CLASS_ENC = {"light": 0.0, "medium": 0.5, "heavy": 1.0}


def build_candidate_context(ctx: PlanningContext, arm: str, models_cfg: dict) -> np.ndarray:
    """Build a 16-dim candidate-specific feature vector φ(x, m).

    Global features (10):
      violation_score, violation_energy, violation_drift,
      current_ema, current_energy_norm, drift_signal,
      ema_slope, steps_since_switch, retrains_norm, vmr_size_norm

    Candidate features (6):
      acc_m, eng_m, acc_ratio (m/current), eng_ratio (m/current),
      is_current, cost_class_encoding
    """
    thr = ctx.thresholds

    # Global (same for all arms)
    v_score  = float(ctx.violation == "score")
    v_energy = float(ctx.violation == "energy")
    v_drift  = float(ctx.violation == "drift")
    curr_ema = float(np.clip(ctx.ema_scores.get(ctx.current_model, 0.5), 0, 1))
    curr_e   = float(np.clip(thr.get("current_normalised_energy", 0.5), 0, 1))
    drift_s  = float(np.clip((thr.get("current_kl_div") or 0.0) / 2.0, 0, 1))
    slope    = float(np.clip(thr.get("ema_slope", 0.0), -1.0, 1.0))
    steps_sw = float(np.clip(thr.get("steps_since_last_switch", 0) / 500.0, 0, 1))
    r_norm   = float(np.clip(thr.get("retrains_this_run", 0) / 10.0, 0, 1))
    vmr_n    = float(np.clip(thr.get("vmr_size", 0) / 20.0, 0, 1))

    # Candidate-specific
    ema_acc = ctx.ema_accuracy if ctx.ema_accuracy else ctx.ema_scores
    ema_eng = ctx.ema_energy if ctx.ema_energy else {}

    acc_m = float(np.clip(ema_acc.get(arm, 0.5), 0, 1))
    eng_m = float(np.clip(ema_eng.get(arm, 0.5), 0, 1))

    curr_acc = float(ema_acc.get(ctx.current_model, 0.5))
    curr_eng = float(ema_eng.get(ctx.current_model, 0.5))
    acc_ratio = float(np.clip(acc_m / max(curr_acc, 1e-6), 0, 2))
    eng_ratio = float(np.clip(eng_m / max(curr_eng, 1e-6), 0, 2))

    is_current = float(arm == ctx.current_model)
    cost_enc = _COST_CLASS_ENC.get(
        models_cfg.get(arm, {}).get("cost_class", "medium"), 0.5
    )

    return np.array([
        v_score, v_energy, v_drift,
        curr_ema, curr_e, drift_s, slope, steps_sw, r_norm, vmr_n,
        acc_m, eng_m, acc_ratio, eng_ratio, is_current, cost_enc,
    ], dtype=np.float32)


# Legacy alias kept for tests
def build_context(ctx: PlanningContext, models: list[str]) -> np.ndarray:
    """Backward-compat wrapper: returns the context vector for the first model."""
    return build_candidate_context(ctx, models[0] if models else ctx.current_model, {})


# ── Factory ───────────────────────────────────────────────────────────────────

def load_or_create_bandit(
    thresholds: dict,
    models: list[str],
    knowledge_dir: str | Path,
    dataset_id: str | None = None,
) -> LinUCBBandit:
    alpha   = float(thresholds.get("bandit_alpha", _DEFAULT_ALPHA))
    beta    = float(thresholds.get("beta", 0.95))
    w_acc   = float(thresholds.get("w_acc", beta))
    w_eng   = float(thresholds.get("w_energy", 1.0 - beta))
    state_path = Path(knowledge_dir) / "bandit_state.json"

    bandit = LinUCBBandit(
        models=models,
        context_dim=_CONTEXT_DIM,
        alpha=alpha,
        w_acc=w_acc,
        w_energy=w_eng,
        state_path=state_path,
    )
    if dataset_id:
        bandit._load_state(dataset_id)
    return bandit


# ── Pending reward resolution (Phase 4.4) ────────────────────────────────────

def resolve_pending(
    bandit: LinUCBBandit,
    pending_path: Path,
    thresholds: dict,
    mape_info: dict,
    current_step: int = 0,
) -> bool:
    """Resolve a pending reward if bandit_min_observation_steps have elapsed.

    Phase 4.4 fix: resolution is step-gated, not wall-clock-gated.
    Returns True if resolved (or deleted on error), False if not ready.
    """
    try:
        with open(pending_path) as f:
            pending = json.load(f)

        min_steps = int(thresholds.get("bandit_min_observation_steps", _DEFAULT_MIN_OBS_STEPS))
        decision_step = int(pending.get("decision_step", 0))
        if (current_step - decision_step) < min_steps:
            return False

        arm = pending["action"]
        phi = np.array(pending["phi"], dtype=np.float32)

        # Accuracy reward: change in EMA accuracy
        acc_after = float(mape_info.get("ema_accuracy", {}).get(arm, 0.5))
        acc_reward = float(np.clip(acc_after - float(pending["ema_acc_before"]), -1.0, 1.0))

        # Energy reward: negative of normalised energy change (lower energy = positive reward)
        eng_after = float(mape_info.get("ema_energy", {}).get(arm, 0.5))
        eng_reward = float(np.clip(float(pending["ema_eng_before"]) - eng_after, -1.0, 1.0))

        # Update energy debt
        e_norm = float(mape_info.get("current_normalised_energy", 0.5))
        e_budget = float(mape_info.get("current_energy_threshold", 0.5))
        bandit.update_energy_debt(e_norm, e_budget)

        bandit.observe_outcome(phi, arm, acc_reward, eng_reward)
        os.unlink(str(pending_path))
        logger.info(
            "[Bandit] Resolved pending: arm=%s, acc_r=%.4f, eng_r=%.4f, Q=%.4f",
            arm, acc_reward, eng_reward, bandit.Q,
        )
        return True

    except Exception as exc:
        logger.warning("[Bandit] Failed to resolve pending (%s). Deleting to avoid stale state.", exc)
        try:
            os.unlink(str(pending_path))
        except OSError:
            pass
        return True


# ── BanditPlanner — Planner ABC adapter ──────────────────────────────────────

class BanditPlanner(Planner):
    """S7 — LinUCB contextual bandit planner with dual estimators (Phase 4).

    Registered as "bandit" in the planner registry.
    """

    name = "bandit"

    def __init__(
        self,
        bandit: "LinUCBBandit | None" = None,
        dataset_id: str = "unknown",
    ) -> None:
        if bandit is None:
            if _bandit_instance is None:
                raise RuntimeError(
                    "BanditPlanner: no bandit instance registered. "
                    "run_experiment.py must call set_bandit_instance() before the MAPE loop."
                )
            self._bandit = _bandit_instance
        else:
            self._bandit = bandit
        self.dataset_id = dataset_id or self._bandit._current_dataset_id

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        if ctx.violation is None:
            return PlanDecision(action="noop", reason="S7 bandit: no violation")

        if ctx.violation == "drift":
            return self._handle_drift(ctx)

        thr = ctx.thresholds
        models_cfg = thr.get("models", {})
        candidates = list(ctx.available_models)

        if not candidates:
            return PlanDecision(action="noop", reason="S7 bandit: no candidates")

        # Build candidate-specific context vectors (Phase 4.2)
        phi_per_arm = {
            m: build_candidate_context(ctx, m, models_cfg)
            for m in candidates
        }

        # Switch costs from cost_class
        switch_costs = {
            m: _COST_CLASS_ENC.get(models_cfg.get(m, {}).get("cost_class", "medium"), 0.5) * 0.1
            for m in candidates
        }

        lambda_0 = float(thr.get("bandit_lambda_0", _DEFAULT_LAMBDA_0))
        mu       = float(thr.get("bandit_mu", _DEFAULT_MU))

        try:
            selected = self._bandit.select_action(
                phi_per_arm=phi_per_arm,
                current_model=ctx.current_model,
                candidates=candidates,
                energy_threshold=ctx.current_energy_threshold,
                lambda_0=lambda_0,
                mu=mu,
                switch_costs=switch_costs,
            )
        except Exception as exc:
            logger.warning("[Bandit] select_action failed (%s); greedy fallback.", exc)
            ema_acc = ctx.ema_accuracy if ctx.ema_accuracy else ctx.ema_scores
            selected = max(candidates, key=lambda m: ema_acc.get(m, 0.0))

        # Hysteresis check (Phase 4.6)
        margin = float(thr.get("bandit_switch_margin", 0.01))
        curr_acc = (ctx.ema_accuracy or ctx.ema_scores).get(ctx.current_model, 0.5)
        sel_acc  = (ctx.ema_accuracy or ctx.ema_scores).get(selected, 0.5)
        if selected == ctx.current_model or not should_switch(
            d_current=-curr_acc, d_candidate=-sel_acc, margin=margin
        ):
            return PlanDecision(
                action="noop",
                reason=(
                    f"S7 bandit: selected arm '{selected}' not better than current "
                    f"'{ctx.current_model}' by margin={margin} — hysteresis guard."
                ),
            )

        phi_selected = phi_per_arm[selected]
        ema_acc_before = float((ctx.ema_accuracy or ctx.ema_scores).get(selected, 0.5))
        ema_eng_before = float((ctx.ema_energy or {}).get(selected, 0.5))

        pending = self._bandit.record_pending(
            phi=phi_selected,
            action=selected,
            ema_acc_before=ema_acc_before,
            ema_eng_before=ema_eng_before,
            decision_step=ctx.current_step,
        )

        pending_path = self._bandit.state_path.parent / "bandit_pending.json"
        _atomic_write(pending_path, pending)
        self._bandit.total_decisions += 1

        stats = self._bandit.get_stats()
        return PlanDecision(
            action="switch",
            model=selected,
            reason=(
                f"S7 bandit: violation={ctx.violation}, selected={selected}, "
                f"decisions={self._bandit.total_decisions}, Q={self._bandit.Q:.4f}"
            ),
            metadata={
                "bandit_pending": pending,
                "dataset_id": self.dataset_id,
                "bandit_stats": stats,
            },
        )

    def _handle_drift(self, ctx: PlanningContext) -> PlanDecision:
        """Drift violation: route to VMR replace or inline retrain.

        Same contract as S4/S5/S6 (harmone_original / violation_aware /
        pareto) — drift is not an arm-selection problem the LinUCB estimators
        model (there's no "retrain" arm with its own context vector), so it
        bypasses select_action()/record_pending() entirely and is not fed
        into the reward update. Previously this always no-op'd (referenced a
        never-implemented plan_drift()), so drift-triggered retrain/replace
        never fired under the bandit planner.
        """
        dr = ctx.drift_result
        if dr is None:
            return PlanDecision(action="noop", reason="S7 bandit: drift signalled but no drift_result")
        if dr.get("action") == "replace":
            path = dr.get("version")
            return PlanDecision(
                action="replace",
                version_path=path,
                reason=f"S7 bandit: VMR replace → {path}",
            )
        return PlanDecision(action="retrain", reason="S7 bandit: no VMR match, retrain")


REGISTRY["bandit"] = BanditPlanner

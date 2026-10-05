"""
core/planners/bandit.py — S7: constrained LinUCB contextual bandit planner (v3).

v3 (2026-10-06, audit B1–B6) replaces the v2 design, which never learned:
its reward-resolution hook was never called by either harness (B1), so every
choice was a fixed function of the features (B2); its energy feasibility
filter compared a raw UCB magnitude with a normalised threshold (B3); 6 of its
10 global features were constants (B4); its reward was a difference of EMAs
(B5); it only acted on violations and an EMA-accuracy hysteresis check vetoed
almost every choice (B6).

Design
------
Arms are the models. Each arm has two disjoint linear estimators over a
shared context vector x (dim 4): one predicts the arm's batch accuracy, one
its normalised batch energy (both in [0, 1], the same quantities the monitor
measures).

Every MAPE cycle (`plan`) does two things:

1. **Learn.** The batch just monitored was served by one model (the
   telemetry's served model). Its measured accuracy and normalised energy
   are that arm's outcome for the context recorded at the previous decision;
   both estimators of the served arm are updated (A += x xᵀ, b += r x). The
   energy-debt queue is updated: Q ← max(0, Q + e_norm − E_ref).

2. **Decide.** Each arm that has never been observed is tried once first.
   Otherwise, for each arm:
       acc_ucb = θ_accᵀx + α·sqrt(xᵀA⁻¹x)        (optimistic accuracy)
       eng_lcb = θ_engᵀx − α·sqrt(xᵀA⁻¹x)        (optimistic energy), clipped to [0, 1]
   Feasible arms: eng_lcb ≤ live energy threshold τ (same normalised units);
   if none, the arm with the lowest eng_lcb. Choose
       argmax  acc_ucb − λ·eng_lcb − switch_cost·[arm ≠ current]
   with λ = bandit_lambda_0 + bandit_mu·Q (Lyapunov drift-plus-penalty: the
   more the run has exceeded the energy budget E_ref, the more energy costs).
   Staying on the current model is a valid choice, so the bandit decides on
   every cycle, not only on violations.

Drift with no score/energy violation still goes to the fixed VMR-replace /
retrain rule shared with S4–S6 (retrain/replace are not arms).

Context x: [1, drift_signal, volatility, τ] — describes the stream, from the
current cycle's telemetry (PlanningContext.telemetry, filled by both
harnesses' _plan()); see build_context().

State (θ statistics, Q, counters) persists to <run_dir>/bandit_state.json
after each update (audit B7: per run, never shared between runs).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from .base import Planner, PlanningContext, PlanDecision, REGISTRY

logger = logging.getLogger(__name__)

_STATE_VERSION = 3
_CONTEXT_DIM = 4
_DEFAULT_ALPHA = 1.0
_DEFAULT_LAMBDA_0 = 0.1
_DEFAULT_MU = 0.05
_DEFAULT_SWITCH_COST = 0.01

# Module-level bandit instance — set once per run by the harness.
_bandit_instance: "LinUCBBandit | None" = None


def set_bandit_instance(bandit: "LinUCBBandit | None") -> None:
    global _bandit_instance
    _bandit_instance = bandit


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, str(path))


# ── Dual-estimator disjoint LinUCB ────────────────────────────────────────────

class LinUCBBandit:
    """Per-arm accuracy and energy ridge-regression estimators with UCB/LCB."""

    def __init__(
        self,
        models: list[str],
        context_dim: int,
        alpha: float,
        state_path: str | Path,
        lambda_0: float = _DEFAULT_LAMBDA_0,
        mu: float = _DEFAULT_MU,
        energy_budget: float = 0.5,
    ) -> None:
        self.models = list(models)
        self.context_dim = context_dim
        self.alpha = alpha
        self.lambda_0 = lambda_0
        self.mu = mu
        self.energy_budget = energy_budget
        self.state_path = Path(state_path)

        self.A_acc: dict[str, np.ndarray] = {}
        self.b_acc: dict[str, np.ndarray] = {}
        self.A_eng: dict[str, np.ndarray] = {}
        self.b_eng: dict[str, np.ndarray] = {}
        self.n_obs: dict[str, int] = {}

        self.Q: float = 0.0
        self.total_decisions: int = 0
        self.total_updates: int = 0
        self._current_dataset_id: str = "unknown"
        # Context of the last decision, consumed by the next observation.
        self.pending_x: np.ndarray | None = None
        for m in self.models:
            self._ensure_arm(m)

    def _ensure_arm(self, m: str) -> None:
        if m not in self.A_acc:
            d = self.context_dim
            self.A_acc[m] = np.eye(d)
            self.b_acc[m] = np.zeros(d)
            self.A_eng[m] = np.eye(d)
            self.b_eng[m] = np.zeros(d)
            self.n_obs[m] = 0

    # ── Estimates ─────────────────────────────────────────────────────────────

    def _mean_and_width(self, x: np.ndarray, A: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        A_inv = np.linalg.inv(A)
        theta = A_inv @ b
        width = self.alpha * float(np.sqrt(max(0.0, x @ A_inv @ x)))
        return float(theta @ x), width

    def acc_ucb(self, x: np.ndarray, arm: str) -> float:
        self._ensure_arm(arm)
        mean, width = self._mean_and_width(x, self.A_acc[arm], self.b_acc[arm])
        return mean + width

    def eng_lcb(self, x: np.ndarray, arm: str) -> float:
        self._ensure_arm(arm)
        mean, width = self._mean_and_width(x, self.A_eng[arm], self.b_eng[arm])
        return float(np.clip(mean - width, 0.0, 1.0))

    @property
    def lam(self) -> float:
        return self.lambda_0 + self.mu * max(0.0, self.Q)

    # ── Learning ──────────────────────────────────────────────────────────────

    def observe_outcome(self, x: np.ndarray, arm: str, accuracy: float, energy_norm: float) -> None:
        """Update the served arm with its measured accuracy and normalised energy."""
        self._ensure_arm(arm)
        x = np.asarray(x, dtype=np.float64)
        outer = np.outer(x, x)
        self.A_acc[arm] = self.A_acc[arm] + outer
        self.b_acc[arm] = self.b_acc[arm] + float(accuracy) * x
        self.A_eng[arm] = self.A_eng[arm] + outer
        self.b_eng[arm] = self.b_eng[arm] + float(energy_norm) * x
        self.n_obs[arm] = self.n_obs.get(arm, 0) + 1
        self.Q = max(0.0, self.Q + float(energy_norm) - self.energy_budget)
        self.total_updates += 1

    # ── Selection ─────────────────────────────────────────────────────────────

    def select_action(
        self,
        x: np.ndarray,
        current_model: str,
        candidates: list[str],
        energy_threshold: float,
        switch_cost: float = _DEFAULT_SWITCH_COST,
    ) -> tuple[str, dict]:
        """Return (chosen arm, per-arm diagnostics)."""
        if not candidates:
            raise ValueError("No candidates to select from.")
        untried = [m for m in candidates if self.n_obs.get(m, 0) == 0]
        if untried:
            pick = current_model if current_model in untried else untried[0]
            return pick, {"untried": untried}

        a_ucb = {m: self.acc_ucb(x, m) for m in candidates}
        e_lcb = {m: self.eng_lcb(x, m) for m in candidates}
        feasible = [m for m in candidates if e_lcb[m] <= energy_threshold]
        pool = feasible if feasible else [min(candidates, key=lambda m: e_lcb[m])]
        lam = self.lam
        scores = {
            m: a_ucb[m] - lam * e_lcb[m] - (switch_cost if m != current_model else 0.0)
            for m in pool
        }
        chosen = max(scores, key=lambda m: scores[m])
        return chosen, {"acc_ucb": a_ucb, "eng_lcb": e_lcb, "feasible": feasible,
                        "scores": scores, "lambda": lam}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        _atomic_write(self.state_path, {
            "state_version": _STATE_VERSION,
            "dataset_id": self._current_dataset_id,
            "total_decisions": self.total_decisions,
            "total_updates": self.total_updates,
            "Q": float(self.Q),
            "n_obs": dict(self.n_obs),
            "A_acc": {m: v.tolist() for m, v in self.A_acc.items()},
            "b_acc": {m: v.tolist() for m, v in self.b_acc.items()},
            "A_eng": {m: v.tolist() for m, v in self.A_eng.items()},
            "b_eng": {m: v.tolist() for m, v in self.b_eng.items()},
        })

    def get_stats(self) -> dict:
        theta_norms = {
            m: float(np.linalg.norm(np.linalg.inv(self.A_acc[m]) @ self.b_acc[m]))
            for m in self.A_acc
        }
        return {
            "total_decisions": self.total_decisions,
            "total_updates": self.total_updates,
            "Q": self.Q,
            "n_obs": dict(self.n_obs),
            "theta_acc_l2_norms": theta_norms,
        }


# ── Context ───────────────────────────────────────────────────────────────────

def build_context(ctx: PlanningContext) -> np.ndarray:
    """Context x for this cycle: describes the STREAM, not the serving arm.

    [1, drift_signal, volatility, τ]
      drift_signal — this cycle's drift statistic relative to tau_drift, in [0, 1]
      volatility   — batch variance / reference-window variance, in [0, 1]
                     (regression with accuracy_signal="window_r2"; 0.5 otherwise)
      τ            — the live adaptive energy threshold
    The served model's own accuracy/energy are its reward, never its context:
    with them in x, each arm's estimator only ever sees contexts it produced
    itself and comparisons across arms become extrapolation.
    """
    tel = ctx.telemetry or {}
    tau_d = float(ctx.thresholds.get("tau_drift", 0.5)) or 0.5
    drift = float(np.clip((tel.get("kl_div") or 0.0) / (2.0 * tau_d), 0.0, 1.0))
    vol = tel.get("volatility")
    vol = 0.5 if vol is None else float(np.clip(vol, 0.0, 1.0))
    return np.array([1.0, drift, vol, float(np.clip(ctx.current_energy_threshold, 0.0, 1.0))],
                    dtype=np.float64)


# ── Factory ───────────────────────────────────────────────────────────────────

def load_or_create_bandit(
    thresholds: dict,
    models: list[str],
    knowledge_dir: str | Path,
    dataset_id: str | None = None,
) -> LinUCBBandit:
    """New bandit for one run; state lives in knowledge_dir (the run directory)."""
    bandit = LinUCBBandit(
        models=models,
        context_dim=_CONTEXT_DIM,
        alpha=float(thresholds.get("bandit_alpha", _DEFAULT_ALPHA)),
        state_path=Path(knowledge_dir) / "bandit_state.json",
        lambda_0=float(thresholds.get("bandit_lambda_0", _DEFAULT_LAMBDA_0)),
        mu=float(thresholds.get("bandit_mu", _DEFAULT_MU)),
        energy_budget=float(thresholds.get("E_ref", 0.5)),
    )
    bandit._current_dataset_id = dataset_id or "unknown"
    return bandit


# ── Planner ───────────────────────────────────────────────────────────────────

class BanditPlanner(Planner):
    """S7 — constrained LinUCB over models; learns from every monitored batch."""

    name = "bandit"

    def __init__(self, bandit: "LinUCBBandit | None" = None, dataset_id: str = "unknown") -> None:
        if bandit is None:
            if _bandit_instance is None:
                raise RuntimeError(
                    "BanditPlanner: no bandit instance registered. "
                    "The harness must call set_bandit_instance() before the MAPE loop."
                )
            bandit = _bandit_instance
        self._bandit = bandit
        self.dataset_id = dataset_id or bandit._current_dataset_id

    def _learn(self, ctx: PlanningContext) -> None:
        tel = ctx.telemetry or {}
        b = self._bandit
        if b.pending_x is None or tel.get("accuracy") is None or tel.get("normalized_energy") is None:
            return
        served = tel.get("served_model") or ctx.current_model
        b.observe_outcome(b.pending_x, served, tel["accuracy"], tel["normalized_energy"])
        b.save()

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        b = self._bandit
        self._learn(ctx)
        x = build_context(ctx)
        b.pending_x = x  # the next monitored batch is this decision's outcome

        if ctx.violation == "drift":
            return self._handle_drift(ctx)

        candidates = list(ctx.available_models)
        if not candidates:
            return PlanDecision(action="noop", reason="S7 bandit: no candidates")

        switch_cost = float(ctx.thresholds.get("bandit_switch_cost", _DEFAULT_SWITCH_COST))
        chosen, diag = b.select_action(
            x, ctx.current_model, candidates, ctx.current_energy_threshold, switch_cost,
        )
        b.total_decisions += 1
        meta = {"bandit_stats": b.get_stats(), "diagnostics": _round(diag)}
        if "untried" in diag:
            why = f"trying unobserved arm (untried={diag['untried']})"
        else:
            why = (f"score={diag['scores'].get(chosen, float('nan')):.4f} "
                   f"lambda={diag['lambda']:.3f} Q={b.Q:.3f}")
        if chosen == ctx.current_model:
            return PlanDecision(action="noop", reason=f"S7 bandit: stay on {chosen} ({why})",
                                metadata=meta)
        return PlanDecision(action="switch", model=chosen,
                            reason=f"S7 bandit: switch to {chosen} ({why})", metadata=meta)

    def _handle_drift(self, ctx: PlanningContext) -> PlanDecision:
        """Drift without a score/energy violation: the fixed VMR-replace /
        retrain rule shared with S4–S6 (retrain/replace are not arms)."""
        dr = ctx.drift_result
        if dr is None:
            return PlanDecision(action="noop", reason="S7 bandit: drift signalled but no drift_result")
        if dr.get("action") == "replace":
            path = dr.get("version")
            return PlanDecision(action="replace", version_path=path,
                                reason=f"S7 bandit: VMR replace → {path}")
        return PlanDecision(action="retrain", reason="S7 bandit: no VMR match, retrain")


def _round(obj):
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


REGISTRY["bandit"] = BanditPlanner

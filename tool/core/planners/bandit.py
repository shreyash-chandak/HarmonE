"""
core/planners/bandit.py — S7: LinUCB contextual bandit planner.

Li et al., "A Contextual-Bandit Approach to Personalised News Article
Recommendation", WWW 2010. Fixed alpha (not decaying) — see DECISIONS.md D12.

Architecture
~~~~~~~~~~~~
- `LinUCBBandit`   — raw algorithm: A/b matrices, selection, reward, persistence.
- `build_context`  — assembles the 10 + 2*|models| context vector.
- `load_or_create_bandit` — factory; called once at startup by manage.py.
- `BanditPlanner`  — Planner ABC adapter; uses module-level bandit instance when
                     called with no args (via get_planner()), or an explicit
                     instance when called directly (used in tests).
- `resolve_pending` — resolves a bandit_pending.json after one monitoring
                      interval; called at the top of each MAPE cycle.
- `set_bandit_instance` — called by manage.py to register the session bandit.

State persistence
~~~~~~~~~~~~~~~~~
knowledge/bandit_state.json  — keyed by dataset_id; accumulates across runs.
knowledge/bandit_pending.json — one pending decision awaiting reward resolution.
Both are written atomically (.tmp → os.replace).

bandit_state.json is intentionally NOT reset by run_reset.py.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from .base import Planner, PlanningContext, PlanDecision, REGISTRY

logger = logging.getLogger(__name__)

# Module-level bandit instance.  manage.py calls set_bandit_instance() once at
# startup so that get_planner("bandit") → BanditPlanner() can find it.
_bandit_instance: "LinUCBBandit | None" = None


def set_bandit_instance(bandit: "LinUCBBandit | None") -> None:
    """Register (or clear) the session bandit.  Called once by manage.py."""
    global _bandit_instance
    _bandit_instance = bandit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON to path atomically via .tmp → os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, str(path))


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

class LinUCBBandit:
    """
    LinUCB contextual bandit.

    Maintains per-arm A matrices and b vectors.
    State persists to knowledge/bandit_state.json keyed by dataset_id.
    Learning accumulates across runs (bandit_state.json is never reset
    by run_reset.py — see experiments/run_reset.py for the explicit comment).

    Thread safety: not thread-safe.  The MAPE loop calls plan() and
    observe_outcome() from the same thread in manage.py — this is correct.
    Do not add locks.
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

        self.A: dict[str, np.ndarray] = {}
        self.b: dict[str, np.ndarray] = {}
        self.total_decisions: int = 0
        self.total_updates: int = 0
        self._current_dataset_id: str = "unknown"

        self._init_arms()

    def _init_arms(self) -> None:
        """Set A[m] = I_d, b[m] = 0_d for all known models."""
        for m in self.models:
            self.A[m] = np.eye(self.context_dim, dtype=np.float64)
            self.b[m] = np.zeros(self.context_dim, dtype=np.float64)

    def _load_state(self, dataset_id: str) -> None:
        """Load A, b, counters from bandit_state.json for this dataset_id."""
        self._current_dataset_id = dataset_id
        if not self.state_path.exists():
            logger.debug("[Bandit] No state file; starting fresh for '%s'.", dataset_id)
            return
        try:
            with open(self.state_path) as f:
                all_state = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[Bandit] Cannot read state (%s); starting fresh.", exc)
            return

        if dataset_id not in all_state:
            logger.debug("[Bandit] No prior state for '%s'; starting fresh.", dataset_id)
            self._init_arms()  # reset to identity so switching datasets doesn't contaminate
            return

        ds = all_state[dataset_id]
        self.total_decisions = ds.get("total_decisions", 0)
        self.total_updates = ds.get("total_updates", 0)
        for m in self.models:
            if m in ds.get("A", {}):
                a_data = ds["A"][m]
                self.A[m] = np.array(a_data, dtype=np.float64)
            if m in ds.get("b", {}):
                self.b[m] = np.array(ds["b"][m], dtype=np.float64)
        logger.info(
            "[Bandit] Loaded state for '%s': %d decisions, %d updates.",
            dataset_id, self.total_decisions, self.total_updates,
        )

    def _save_state(self, dataset_id: str) -> None:
        """Atomic write of current A/b/counters for dataset_id."""
        try:
            all_state: dict = {}
            if self.state_path.exists():
                with open(self.state_path) as f:
                    all_state = json.load(f)
        except (json.JSONDecodeError, OSError):
            all_state = {}

        all_state[dataset_id] = {
            "total_decisions": self.total_decisions,
            "total_updates": self.total_updates,
            "A": {m: self.A[m].tolist() for m in self.models if m in self.A},
            "b": {m: self.b[m].tolist() for m in self.models if m in self.b},
        }
        _atomic_write(self.state_path, all_state)

    def select_action(
        self,
        context: np.ndarray,
        current_model: str,
        candidates: list[str],
    ) -> str:
        """LinUCB arm selection.  Raises ValueError if candidates is empty."""
        if not candidates:
            raise ValueError("No candidates to select from.")

        ucb_scores: dict[str, float] = {}
        for arm in candidates:
            if arm not in self.A:
                self.A[arm] = np.eye(self.context_dim, dtype=np.float64)
                self.b[arm] = np.zeros(self.context_dim, dtype=np.float64)
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            x = context.astype(np.float64)
            expected = float(theta @ x)
            uncertainty = self.alpha * float(np.sqrt(x @ A_inv @ x))
            ucb_scores[arm] = expected + uncertainty

        logger.debug("[Bandit] UCB scores: %s", ucb_scores)
        return max(ucb_scores, key=lambda a: ucb_scores[a])

    def record_pending(
        self,
        context: np.ndarray,
        action: str,
        ema_before: float,
        energy_before: float,
        switch_cost_J: float,
    ) -> dict:
        """Return the pending record dict; caller persists it."""
        return {
            "context": context.tolist(),
            "action": action,
            "ema_before": float(ema_before),
            "energy_before": float(energy_before),
            "switch_cost_J": float(switch_cost_J),
            "timestamp": time.time(),
        }

    def observe_outcome(
        self,
        context: np.ndarray,
        action: str,
        reward: float,
        dataset_id: str | None = None,
    ) -> None:
        """LinUCB matrix update: A += outer(x,x), b += r*x.  Saves state."""
        if action not in self.A:
            self.A[action] = np.eye(self.context_dim, dtype=np.float64)
            self.b[action] = np.zeros(self.context_dim, dtype=np.float64)
        x = context.astype(np.float64)
        self.A[action] = self.A[action] + np.outer(x, x)
        self.b[action] = self.b[action] + reward * x
        self.total_updates += 1
        self._save_state(dataset_id or self._current_dataset_id)

    def compute_reward(
        self,
        ema_before: float,
        ema_after: float,
        energy_before: float,
        energy_after: float,
        switch_cost_J: float,
        w_acc: float | None = None,
        w_energy: float | None = None,
    ) -> float:
        """
        Reward = sustainability gain per joule of switching cost.

        Positive when the new model improves accuracy and/or reduces energy.
        Negative when the switch makes things worse.
        Clipped to [-10, 10] to prevent gradient explosion.
        """
        _w_acc = w_acc if w_acc is not None else self.w_acc
        _w_energy = w_energy if w_energy is not None else self.w_energy

        delta_acc = ema_after - ema_before
        delta_energy = energy_before - energy_after

        delta_acc_norm = float(np.clip(delta_acc / 0.5, -1.0, 1.0))
        delta_energy_norm = float(np.clip(
            delta_energy / max(energy_before, 1e-9), -1.0, 1.0
        ))

        gain = _w_acc * delta_acc_norm + _w_energy * delta_energy_norm
        cost = max(switch_cost_J, 1e-9)
        return float(np.clip(gain / cost, -10.0, 10.0))

    def get_stats(self) -> dict:
        """Return summary stats for logging/dashboard."""
        pending_path = self.state_path.parent / "bandit_pending.json"
        theta_norms: dict[str, float] = {}
        for m in self.models:
            if m in self.A and m in self.b:
                try:
                    theta = np.linalg.inv(self.A[m]) @ self.b[m]
                    theta_norms[m] = float(np.linalg.norm(theta))
                except np.linalg.LinAlgError:
                    theta_norms[m] = 0.0
        return {
            "total_decisions": self.total_decisions,
            "total_updates": self.total_updates,
            "has_pending": pending_path.exists(),
            "theta_l2_norms": theta_norms,
        }


# ---------------------------------------------------------------------------
# Context vector
# ---------------------------------------------------------------------------

def build_context(ctx: PlanningContext, models: list[str]) -> np.ndarray:
    """
    10 + 2*len(models) dimensional context vector from PlanningContext.

    Fixed ordering; all features normalised to approximately [0,1] or [-1,1].
    If ema_accuracy / ema_energy are absent, falls back to ema_scores for both
    (noted limitation: accuracy and energy signals are then indistinguishable).
    """
    violation_score = float(ctx.violation == "score")
    violation_energy = float(ctx.violation == "energy")
    violation_drift = float(ctx.violation == "drift")

    current_ema = float(np.clip(ctx.ema_scores.get(ctx.current_model, 0.5), 0, 1))
    current_energy = float(np.clip(
        ctx.thresholds.get("current_normalised_energy", 0.5), 0, 1
    ))
    drift_signal = float(
        np.clip((ctx.thresholds.get("current_kl_div") or 0.0), 0.0, 2.0) / 2.0
    )
    ema_slope = float(np.clip(ctx.thresholds.get("ema_slope", 0.0), -1.0, 1.0))
    steps_since = float(np.clip(
        ctx.thresholds.get("steps_since_last_switch", 0) / 500.0, 0.0, 1.0
    ))
    retrains_norm = float(np.clip(
        ctx.thresholds.get("retrains_this_run", 0) / 10.0, 0.0, 1.0
    ))
    vmr_size_norm = float(np.clip(
        ctx.thresholds.get("vmr_size", 0) / 20.0, 0.0, 1.0
    ))

    ema_acc = ctx.ema_accuracy if ctx.ema_accuracy else ctx.ema_scores
    ema_eng = ctx.ema_energy if ctx.ema_energy else {m: 0.5 for m in models}

    model_ema_acc = [float(np.clip(ema_acc.get(m, 0.5), 0, 1)) for m in models]
    model_ema_eng = [float(np.clip(ema_eng.get(m, 0.5), 0, 1)) for m in models]

    features = (
        [violation_score, violation_energy, violation_drift,
         current_ema, current_energy, drift_signal, ema_slope,
         steps_since, retrains_norm, vmr_size_norm]
        + model_ema_acc
        + model_ema_eng
    )
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_or_create_bandit(
    thresholds: dict,
    models: list[str],
    knowledge_dir: str | Path,
    dataset_id: str | None = None,
) -> LinUCBBandit:
    """
    Construct LinUCBBandit from thresholds config.

    Args:
        thresholds:    Contents of thresholds.json.
        models:        List of model names (must match what plan.py uses).
        knowledge_dir: Path to knowledge/ directory (state_path lives here).
        dataset_id:    If provided, immediately calls _load_state(dataset_id).
    """
    alpha = float(thresholds.get("bandit_alpha", 1.0))
    beta = float(thresholds.get("beta", 0.95))
    w_acc = float(thresholds.get("w_acc", beta))
    w_energy = float(thresholds.get("w_energy", 1.0 - beta))
    context_dim = 10 + 2 * len(models)
    state_path = Path(knowledge_dir) / "bandit_state.json"

    bandit = LinUCBBandit(
        models=models,
        context_dim=context_dim,
        alpha=alpha,
        w_acc=w_acc,
        w_energy=w_energy,
        state_path=state_path,
    )
    if dataset_id:
        bandit._load_state(dataset_id)
    return bandit


# ---------------------------------------------------------------------------
# Pending reward resolution (called from manage.py each MAPE cycle)
# ---------------------------------------------------------------------------

def resolve_pending(
    bandit: LinUCBBandit,
    pending_path: Path,
    thresholds: dict,
    mape_info: dict,
) -> bool:
    """
    Resolve a pending bandit reward if at least one monitoring interval has passed.

    Reads current EMA state from mape_info, computes the reward, calls
    bandit.observe_outcome(), and deletes the pending file.

    Returns:
        True  if the pending was resolved (or deleted on error).
        False if not enough time has passed yet (file untouched).

    Called at the top of each MAPE cycle from manage.py.  Safe to call even
    when pending_path does not exist (caller should check .exists() first).
    """
    try:
        with open(pending_path) as f:
            pending = json.load(f)

        elapsed = time.time() - float(pending["timestamp"])
        min_interval = float(thresholds.get("monitoring_interval_s", 40.0))

        if elapsed < min_interval:
            return False

        arm = pending["action"]
        ema_after = float(mape_info.get("ema_scores", {}).get(arm, 0.5))
        energy_after = float(mape_info.get("current_normalised_energy", 0.5))
        context = np.array(pending["context"], dtype=np.float32)

        beta = float(thresholds.get("beta", 0.95))
        reward = bandit.compute_reward(
            ema_before=float(pending["ema_before"]),
            ema_after=ema_after,
            energy_before=float(pending["energy_before"]),
            energy_after=energy_after,
            switch_cost_J=float(pending.get("switch_cost_J", 0.0)),
            w_acc=float(thresholds.get("w_acc", beta)),
            w_energy=float(thresholds.get("w_energy", 1.0 - beta)),
        )

        bandit.observe_outcome(context, arm, reward)
        os.unlink(str(pending_path))
        logger.info(
            "[Bandit] Resolved pending: arm=%s, reward=%.4f, ema_delta=%.4f",
            arm, reward, ema_after - float(pending["ema_before"]),
        )
        return True

    except Exception as exc:
        logger.warning(
            "[Bandit] Failed to resolve pending reward (%s). "
            "Deleting pending file to avoid stale state.", exc
        )
        try:
            os.unlink(str(pending_path))
        except OSError:
            pass
        return True


# ---------------------------------------------------------------------------
# BanditPlanner — Planner ABC adapter
# ---------------------------------------------------------------------------

class BanditPlanner(Planner):
    """
    S7 — LinUCB contextual bandit planner.

    Registered as "bandit" in the planner registry.

    When instantiated with no arguments (via get_planner("bandit")), reads the
    module-level _bandit_instance set by manage.py.  Tests pass an explicit
    bandit instance to the constructor instead.

    Pending decision lifecycle:
    - plan() selects the arm, writes bandit_pending.json atomically, and
      returns the pending record in PlanDecision.metadata["bandit_pending"].
    - manage.py checks bandit_pending.json at the top of each MAPE cycle
      and calls resolve_pending() once enough time has passed.

    Decision fallback:
    If select_action() raises for any reason (empty candidates, numerical
    error in matrix inversion), falls back to greedy (highest-EMA alternative)
    and logs a WARNING.  Does NOT raise.
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
                    "manage.py must call set_bandit_instance() before the MAPE loop."
                )
            self._bandit = _bandit_instance
        else:
            self._bandit = bandit
        self.dataset_id = dataset_id or self._bandit._current_dataset_id

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        if ctx.violation is None:
            return PlanDecision(action="noop", reason="S7 bandit: no violation")

        if ctx.violation == "drift":
            # Drift is handled by plan_drift() / execute_drift(), not the bandit.
            return PlanDecision(
                action="noop",
                reason="S7 bandit: drift violations delegated to plan_drift()",
            )

        models = list(ctx.ema_scores.keys()) or list(ctx.available_models)
        context = build_context(ctx, models)

        candidates = [m for m in models if m != ctx.current_model]
        if not candidates:
            return PlanDecision(
                action="noop",
                reason="S7 bandit: no candidates (single model)",
            )

        try:
            selected = self._bandit.select_action(context, ctx.current_model, candidates)
        except Exception as exc:
            logger.warning(
                "[Bandit] select_action failed (%s); falling back to greedy.", exc
            )
            selected = max(
                candidates,
                key=lambda m: ctx.ema_scores.get(m, 0.0),
            )

        ema_before = float(ctx.ema_scores.get(selected, 0.5))
        energy_before = float(ctx.thresholds.get("current_normalised_energy", 0.5))

        pending = self._bandit.record_pending(
            context=context,
            action=selected,
            ema_before=ema_before,
            energy_before=energy_before,
            switch_cost_J=0.0,  # unknown at plan time; reward still meaningful
        )

        # Write pending file so manage.py can resolve the reward next cycle.
        pending_path = self._bandit.state_path.parent / "bandit_pending.json"
        _atomic_write(pending_path, pending)

        self._bandit.total_decisions += 1

        return PlanDecision(
            action="switch",
            model=selected,
            reason=(
                f"S7 bandit: violation={ctx.violation}, selected={selected}, "
                f"decisions={self._bandit.total_decisions}"
            ),
            metadata={"bandit_pending": pending, "dataset_id": self.dataset_id},
        )


REGISTRY["bandit"] = BanditPlanner

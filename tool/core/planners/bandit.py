"""
core/planners/bandit.py — S7: LinUCB contextual bandit (stub).

Full algorithm specification:
  Context vector x_t = [EMA_slope_k, drift_signal, steps_since_switch, budget_remaining]
  Reward r_t = ΔS_i per Joule of switching cost
  LinUCB update: A_a += x_t x_t^T, b_a += r_t x_t
  Action: a* = argmax_a (θ_a^T x_t + α * sqrt(x_t^T A_a^{-1} x_t))

  Implementation cost ~2 days; requires:
  - Feature vector definition per managed system (EMA slope needs k-interval history)
  - Reward signal: ΔS_i = S_i(chosen) - S_i(prev) normalised by switching overhead µJ
  - Per-arm (per-model) A matrix + b vector persisted across MAPE-K cycles
  - α hyperparameter (exploration/exploitation trade-off; tune via grid in Phase 5)

Status: Documented stub. Raises NotImplementedError with a clear message.
See DECISIONS_PENDING.md DP3 for the pending decision on full implementation.
"""

from __future__ import annotations

from .base import Planner, PlanningContext, PlanDecision, REGISTRY


class BanditPlanner(Planner):
    """S7: LinUCB contextual bandit (not yet implemented — see DECISIONS_PENDING.md DP3)."""

    name = "bandit"

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        raise NotImplementedError(
            "S7 LinUCB bandit planner is not yet implemented. "
            "See DECISIONS_PENDING.md DP3 for the full algorithm specification "
            "and the pending decision on implementation timeline. "
            "Use planner='harmone_original' or planner='pareto' for current experiments."
        )


REGISTRY["bandit"] = BanditPlanner

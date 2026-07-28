"""
core/planners/greedy_switch.py — S3: Greedy EMA switch.

On any violation, switches to the non-current model with the highest combined
EMA score. No exploration (alpha=0), no drift-specific tactics.
This is the honest "always pick the historically best model" baseline.
"""

from __future__ import annotations

from .base import Planner, PlanningContext, PlanDecision, REGISTRY


class GreedySwitchPlanner(Planner):
    """S3: Always switch to the highest-EMA non-current model on any violation."""

    name = "greedy_switch"

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        if ctx.violation is None:
            return PlanDecision(action="noop", reason="S3 greedy: no violation")

        alternatives = {
            m: ctx.ema_scores.get(m, 0.0)
            for m in ctx.available_models
            if m != ctx.current_model
        }
        if not alternatives:
            return PlanDecision(action="noop", reason="S3 greedy: no alternatives available")

        chosen = max(alternatives, key=alternatives.__getitem__)
        score = alternatives[chosen]

        return PlanDecision(
            action="switch",
            model=chosen,
            reason=f"S3 greedy: violation={ctx.violation}, best_ema={score:.4f} → {chosen}",
        )


REGISTRY["greedy_switch"] = GreedySwitchPlanner

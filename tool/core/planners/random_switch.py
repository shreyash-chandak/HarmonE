"""
core/planners/random_switch.py — S2: Random model switch.

On any violation, picks uniformly at random from the non-current models.
Our own additional baseline, kept registered but no longer in the regression
grid (2026-10-31). The paper's "Switch" is core/planners/switch.py.
"""

from __future__ import annotations

import random

from .base import Planner, PlanningContext, PlanDecision, REGISTRY


class RandomSwitchPlanner(Planner):
    """S2: Random switch. On violation, choose any other available model."""

    name = "random_switch"
    reacts_to_drift = False  # like the original "switch" baselines (no drift thread)

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        if ctx.violation is None:
            return PlanDecision(action="noop", reason="S2 random: no violation")

        alternatives = [m for m in ctx.available_models if m != ctx.current_model]
        if not alternatives:
            return PlanDecision(action="noop", reason="S2 random: no alternatives available")

        chosen = random.choice(alternatives)
        return PlanDecision(
            action="switch",
            model=chosen,
            reason=f"S2 random: violation={ctx.violation}, picked {chosen} uniformly",
        )


REGISTRY["random_switch"] = RandomSwitchPlanner

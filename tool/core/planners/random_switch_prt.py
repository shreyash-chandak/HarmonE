"""
core/planners/random_switch_prt.py — Random Switch + Periodic Retraining (PRT) baseline.

Replicates the Switch+PRT baseline from the paper: random model switching on
any violation, PLUS blind retraining of the active model every PRT_INTERVAL steps.
PRT takes priority over a same-cycle switch decision.
"""

from __future__ import annotations

import random

from .base import Planner, PlanningContext, PlanDecision, REGISTRY

PRT_INTERVAL = 3200


class RandomSwitchPRTPlanner(Planner):
    """Random switch on violation + periodic retraining every 3 200 steps."""

    name = "random_switch_prt"

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        if ctx.current_step > 0 and ctx.current_step % PRT_INTERVAL == 0:
            return PlanDecision(
                action="retrain",
                reason=f"random_switch_prt: PRT trigger at step {ctx.current_step}",
            )

        if ctx.violation is None:
            return PlanDecision(action="noop", reason="random_switch_prt: no violation")

        alternatives = [m for m in ctx.available_models if m != ctx.current_model]
        if not alternatives:
            return PlanDecision(action="noop", reason="random_switch_prt: no alternatives")

        chosen = random.choice(alternatives)
        return PlanDecision(
            action="switch",
            model=chosen,
            reason=f"random_switch_prt: violation={ctx.violation}, random → {chosen}",
        )


REGISTRY["random_switch_prt"] = RandomSwitchPRTPlanner

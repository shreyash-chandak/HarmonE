"""
core/planners/random_switch_prt.py — Random Switch + Periodic Retraining (PRT) baseline.

Replicates the Switch+PRT baseline from the paper: random model switching on
any violation, PLUS blind retraining of the active model every prt_interval steps.
PRT takes priority over a same-cycle switch decision. Config-driven via
thresholds["prt_interval"] (falls back to PRT_INTERVAL=3200 when absent) — see
naive_prt.py's docstring for why CV needed this to stop being a shared, regression-
sized constant.
"""

from __future__ import annotations

import random

from .base import Planner, PlanningContext, PlanDecision, REGISTRY

PRT_INTERVAL = 3200


class RandomSwitchPRTPlanner(Planner):
    """Random switch on violation + periodic retraining every prt_interval steps."""

    name = "random_switch_prt"

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        prt_interval = int(ctx.thresholds.get("prt_interval", PRT_INTERVAL))
        if ctx.current_step > 0 and ctx.current_step % prt_interval == 0:
            return PlanDecision(
                action="retrain",
                reason=f"random_switch_prt: PRT trigger at step {ctx.current_step} (interval={prt_interval})",
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

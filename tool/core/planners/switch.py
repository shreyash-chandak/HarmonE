"""
core/planners/switch.py — the paper's "Switch" and "Switch+PRT" baselines (2026-10-31).

The original HarmonE's `switch` approach runs the same t1 as `harmone`
(HarmonE/mape/manage.py -> execute_mape -> plan_mape: ε-greedy exploration,
best-EMA selection) with no drift thread and no retraining; `switch+retrain`
adds its periodic retrain thread. So Switch is S4 without drift handling, not a
uniform random pick: `random_switch` is our own additional baseline.
"""

from __future__ import annotations

from .base import PlanningContext, PlanDecision, REGISTRY, prt_due
from .harmone_original import HarmonEOriginalPlanner

PRT_INTERVAL = 3200


class SwitchPlanner(HarmonEOriginalPlanner):
    """Paper "Switch": ε-greedy + best EMA; the harness withholds drift."""

    name = "switch"
    reacts_to_drift = False

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        decision = super().plan(ctx)
        decision.reason = decision.reason.replace("S4 harmone_original", self.name)
        return decision


class SwitchPRTPlanner(SwitchPlanner):
    """Paper "Switch+PRT": Switch plus a blind retrain of the active model
    every prt_interval rows (PRT has priority in its cycle)."""

    name = "switch_prt"

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        if prt_due(ctx, PRT_INTERVAL):
            return PlanDecision(
                action="retrain",
                reason=f"switch_prt: PRT trigger at step {ctx.current_step}",
                metadata={"prt": True},
            )
        return super().plan(ctx)


REGISTRY["switch"] = SwitchPlanner
REGISTRY["switch_prt"] = SwitchPRTPlanner

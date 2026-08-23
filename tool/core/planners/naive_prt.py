"""
core/planners/naive_prt.py — Naive + Periodic Retraining (PRT) baseline.

Replicates the LR+PRT / SVM+PRT / LSTM+PRT baselines from the paper.
Use with --pin-model to lock a single model; this planner never switches.
Every PRT_INTERVAL steps it issues "retrain" to retrain the pinned model
on the most recent data window, mirroring the paper's blind 3 200-step schedule.
"""

from __future__ import annotations

from .base import Planner, PlanningContext, PlanDecision, REGISTRY

PRT_INTERVAL = 3200


class NaivePRTPlanner(Planner):
    """Naive (no switching) + periodic retraining every 3 200 steps."""

    name = "naive_prt"

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        if ctx.current_step > 0 and ctx.current_step % PRT_INTERVAL == 0:
            return PlanDecision(
                action="retrain",
                reason=f"naive_prt: PRT trigger at step {ctx.current_step}",
            )
        return PlanDecision(action="noop", reason="naive_prt: no PRT trigger")


REGISTRY["naive_prt"] = NaivePRTPlanner

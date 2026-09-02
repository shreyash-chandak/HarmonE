"""
core/planners/naive_prt.py — Naive + Periodic Retraining (PRT) baseline.

Replicates the LR+PRT / SVM+PRT / LSTM+PRT baselines from the paper.
Use with --pin-model to lock a single model; this planner never switches.
Every prt_interval steps it issues "retrain" to retrain the pinned model
on the most recent data window, mirroring the paper's blind 3 200-step schedule
for regression. Config-driven via thresholds["prt_interval"] (falls back to
PRT_INTERVAL=3200 when absent) — CV datasets' streams (1600-2500 images, see
context/cv_explained.md §15) are far shorter than 3200, so the shared regression
constant meant PRT never fired for CV at all; CV configs now set prt_interval=500
explicitly instead of inheriting a value sized for regression's much longer runs.
"""

from __future__ import annotations

from .base import Planner, PlanningContext, PlanDecision, REGISTRY

PRT_INTERVAL = 3200


class NaivePRTPlanner(Planner):
    """Naive (no switching) + periodic retraining every prt_interval steps."""

    name = "naive_prt"

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        prt_interval = int(ctx.thresholds.get("prt_interval", PRT_INTERVAL))
        if ctx.current_step > 0 and ctx.current_step % prt_interval == 0:
            return PlanDecision(
                action="retrain",
                reason=f"naive_prt: PRT trigger at step {ctx.current_step} (interval={prt_interval})",
            )
        return PlanDecision(action="noop", reason="naive_prt: no PRT trigger")


REGISTRY["naive_prt"] = NaivePRTPlanner

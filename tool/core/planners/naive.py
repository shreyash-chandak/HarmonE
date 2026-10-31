"""
core/planners/naive.py — S1: Static single-model baseline.

Never switches. Returns noop regardless of violation type. Used as the lower
bound in the experiment grid — pure single-model inference with no adaptation.
"""

from __future__ import annotations

from .base import Planner, PlanningContext, PlanDecision, REGISTRY


class NaivePlanner(Planner):
    """S1: Never adapt. Keeps the model fixed for the entire stream."""

    name = "naive"
    reacts_to_drift = False  # the original's single-model approaches run no drift thread

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        return PlanDecision(action="noop", reason="S1 naive: no adaptation")


REGISTRY["naive"] = NaivePlanner

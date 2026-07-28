"""
core/planners/harmone_original.py — S4: Bug-fixed original HarmonE planner.

Epsilon-greedy selection with exploration probability alpha:
- With probability alpha: pick a random non-current model (exploration).
- Otherwise: pick the highest-EMA non-current model (exploitation).

Drift violations are routed to VMR replace (if drift_result["action"]=="replace")
or retrain. This is the paper's published strategy, bug-fixed (B1-B6).
"""

from __future__ import annotations

import random

from .base import Planner, PlanningContext, PlanDecision, REGISTRY


class HarmonEOriginalPlanner(Planner):
    """S4: Bug-fixed ε-greedy HarmonE (the paper's published strategy)."""

    name = "harmone_original"

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        if ctx.violation is None:
            return PlanDecision(action="noop", reason="S4 harmone_original: no violation")

        # Drift violation: route to VMR or retrain
        if ctx.violation == "drift":
            return self._handle_drift(ctx)

        # Score/energy violation: ε-greedy model selection
        alpha = ctx.thresholds.get("alpha", 0.1)
        alternatives = [m for m in ctx.available_models if m != ctx.current_model]
        if not alternatives:
            return PlanDecision(action="noop", reason="S4 harmone_original: no alternatives")

        # Explore
        if random.random() < alpha:
            chosen = random.choice(alternatives)
            return PlanDecision(
                action="switch",
                model=chosen,
                reason=f"S4 harmone_original: exploration (α={alpha}), picked {chosen}",
            )

        # Exploit
        chosen = max(alternatives, key=lambda m: ctx.ema_scores.get(m, 0.0))
        score = ctx.ema_scores.get(chosen, 0.0)
        return PlanDecision(
            action="switch",
            model=chosen,
            reason=f"S4 harmone_original: exploit, best_ema={score:.4f} → {chosen}",
        )

    def _handle_drift(self, ctx: PlanningContext) -> PlanDecision:
        dr = ctx.drift_result
        if dr is None:
            return PlanDecision(action="noop", reason="S4 harmone_original: drift signalled but no drift_result")

        if dr.get("action") == "replace":
            path = dr.get("version")
            return PlanDecision(
                action="replace",
                version_path=path,
                reason=f"S4 harmone_original: VMR replace → {path}",
            )

        return PlanDecision(action="retrain", reason="S4 harmone_original: no VMR match, retrain")


REGISTRY["harmone_original"] = HarmonEOriginalPlanner

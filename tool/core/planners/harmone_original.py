"""
core/planners/harmone_original.py — S4: Bug-fixed original HarmonE planner.

Epsilon-greedy selection with exploration probability alpha:
- Every cycle (violation or not), with probability alpha: pick a random model
  (exploration). If the draw is the current model, fall through to the normal
  logic below.
- Otherwise, on a score/energy violation: pick the highest-EMA non-current
  model (exploitation).

Drift violations are routed to VMR replace (if drift_result["action"]=="replace")
or retrain, and take priority over exploration. This is the paper's published
strategy, bug-fixed (B1-B6).

2026-09-28 (audit A2): exploration now rolls on EVERY cycle, matching the
original HarmonE/mape/plan.py::plan_mape(), which drew `random() < alpha`
before even analysing thresholds. The previous version only explored during a
violation, so inactive models were almost never re-measured and their EMA
estimates froze.
"""

from __future__ import annotations

import random

from .base import Planner, PlanningContext, PlanDecision, REGISTRY


class HarmonEOriginalPlanner(Planner):
    """S4: Bug-fixed ε-greedy HarmonE (the paper's published strategy)."""

    name = "harmone_original"

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        # Drift violation: route to VMR or retrain (takes priority)
        if ctx.violation == "drift":
            return self._handle_drift(ctx)

        # Explore — every cycle, like the original plan_mape()
        alpha = ctx.thresholds.get("alpha", 0.1)
        if ctx.available_models and random.random() < alpha:
            chosen = random.choice(list(ctx.available_models))
            if chosen != ctx.current_model:
                return PlanDecision(
                    action="switch",
                    model=chosen,
                    reason=f"S4 harmone_original: exploration (α={alpha}), picked {chosen}",
                )

        if ctx.violation is None:
            return PlanDecision(action="noop", reason="S4 harmone_original: no violation")

        # Score/energy violation: exploit, as HarmonE/mape/plan.py::plan_mape():
        #   energy violation -> best EMA among the OTHER models;
        #   score violation  -> best EMA among ALL models, so if the current
        #   model is already the best, no switch (2026-10-08: previously the
        #   current model was excluded for both, always forcing a switch).
        if ctx.violation == "energy":
            pool = [m for m in ctx.available_models if m != ctx.current_model]
        else:
            pool = list(ctx.available_models)
        if not pool:
            return PlanDecision(action="noop", reason="S4 harmone_original: no alternatives")

        chosen = max(pool, key=lambda m: ctx.ema_scores.get(m, 0.0))
        score = ctx.ema_scores.get(chosen, 0.0)
        if chosen == ctx.current_model:
            return PlanDecision(
                action="noop",
                reason=f"S4 harmone_original: score violation, current model is best (ema={score:.4f})",
            )
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

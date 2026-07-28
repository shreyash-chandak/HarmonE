"""
core/planners/violation_aware.py — S5: Violation-type-aware routing.

Routes each violation type to a targeted response:
- energy violation: find an accurate model that fits within the energy budget.
- score violation: find an energy-efficient model that meets accuracy.
- drift violation: same as S4 (VMR replace or retrain).

Uses separated ema_accuracy and ema_energy signals (Phase 2.4).
Falls back gracefully when those signals are missing.
"""

from __future__ import annotations

from .base import Planner, PlanningContext, PlanDecision, REGISTRY


class ViolationAwarePlanner(Planner):
    """S5: Route model selection based on which threshold was violated."""

    name = "violation_aware"

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        if ctx.violation is None:
            return PlanDecision(action="noop", reason="S5 violation_aware: no violation")

        if ctx.violation == "drift":
            return self._handle_drift(ctx)

        if ctx.violation == "energy":
            return self._handle_energy_violation(ctx)

        # "score" or any other violation type
        return self._handle_score_violation(ctx)

    def _handle_energy_violation(self, ctx: PlanningContext) -> PlanDecision:
        """Energy too high: find an accurate model that uses less energy."""
        thr = ctx.thresholds
        min_score = thr.get("min_score", 0.0)
        others = [m for m in ctx.available_models if m != ctx.current_model]

        ema_acc = ctx.ema_accuracy
        ema_eng = ctx.ema_energy

        # Candidates: meeting accuracy requirement
        candidates = [m for m in others if ema_acc.get(m, 0.0) >= min_score]
        pool = candidates if candidates else others  # fallback: any alternative

        # Pick minimum energy from pool
        chosen = min(pool, key=lambda m: ema_eng.get(m, float("inf")))
        eng = ema_eng.get(chosen, float("nan"))
        return PlanDecision(
            action="switch",
            model=chosen,
            reason=f"S5 violation_aware: energy violation, lowest_energy_model={chosen} (ema_E={eng:.4f})",
        )

    def _handle_score_violation(self, ctx: PlanningContext) -> PlanDecision:
        """Score too low: find an energy-compliant model with best accuracy."""
        thr = ctx.thresholds
        e_threshold = thr.get("current_energy_threshold", thr.get("max_energy", 1.0))
        others = [m for m in ctx.available_models if m != ctx.current_model]

        ema_acc = ctx.ema_accuracy
        ema_eng = ctx.ema_energy

        # Candidates: within energy budget
        candidates = [m for m in others if ema_eng.get(m, 1.0) <= e_threshold]
        pool = candidates if candidates else others

        chosen = max(pool, key=lambda m: ema_acc.get(m, 0.0))
        acc = ema_acc.get(chosen, float("nan"))
        return PlanDecision(
            action="switch",
            model=chosen,
            reason=f"S5 violation_aware: score violation, best_acc_model={chosen} (ema_A={acc:.4f})",
        )

    def _handle_drift(self, ctx: PlanningContext) -> PlanDecision:
        dr = ctx.drift_result
        if dr is None:
            return PlanDecision(action="noop", reason="S5 violation_aware: drift signalled but no drift_result")
        if dr.get("action") in ("replace", "switch_version"):
            path = dr.get("version")
            return PlanDecision(
                action="replace",
                version_path=path,
                reason=f"S5 violation_aware: VMR replace → {path}",
            )
        return PlanDecision(action="retrain", reason="S5 violation_aware: no VMR match, retrain")


REGISTRY["violation_aware"] = ViolationAwarePlanner

"""
core/planners/violation_aware.py — S5: Violation-type-aware routing (Phase 2 repair).

Routes each violation type to a targeted response:
- energy violation: find the model that meets min_accuracy AND uses least energy.
- score  violation: find the model that meets energy budget AND has best accuracy.
- drift  violation: VMR replace or retrain (same as S4).

Phase 2 bug fixes applied (all labeled as fixes, not redesigns):
  2.1 Scale bug:   energy handler now reads min_accuracy, not min_score.
  2.2 Threshold bug: score handler reads live current_energy_threshold from ctx.
  2.3 Forced-switch bug: current model included in both candidate pools; returns
      noop (with explicit reason) when current is already the best choice.
  2.4 Thrash guard: hysteresis via shared helper with s5_switch_margin (default 0.02).
  2.5 Fallback semantics: when no candidate meets the constraint, minimise the
      violated quantity (least energy for energy-violations; best accuracy for
      score-violations) across ALL models including current, rather than picking
      any arbitrary alternative.
"""

from __future__ import annotations

import logging

from .base import Planner, PlanningContext, PlanDecision, REGISTRY
from .hysteresis import should_switch

logger = logging.getLogger(__name__)

_DEFAULT_MIN_ACCURACY = 0.0   # conservative: never filter on accuracy if key absent
_DEFAULT_SWITCH_MARGIN = 0.02


def _get_min_accuracy(thr: dict) -> float:
    """Read min_accuracy from thresholds; fall back to 0.0 with a warning (Phase 1.3)."""
    if "min_accuracy" in thr:
        return float(thr["min_accuracy"])
    logger.warning(
        "S5 violation_aware: 'min_accuracy' not in thresholds; "
        "defaulting to 0.0 (accept all models on accuracy). "
        "Add 'min_accuracy' to the dataset config to suppress this warning."
    )
    return _DEFAULT_MIN_ACCURACY


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

    # ------------------------------------------------------------------
    # Energy violation: find model with acceptable accuracy and least energy
    # ------------------------------------------------------------------

    def _handle_energy_violation(self, ctx: PlanningContext) -> PlanDecision:
        """Energy too high: find the most accurate model that uses least energy.

        Fix 2.1: uses min_accuracy, not min_score.
        Fix 2.3: includes current model; returns noop if current is already best.
        Fix 2.4: hysteresis — only switch if energy improvement exceeds margin.
        Fix 2.5: if no model meets min_accuracy, fall back to min-energy over all.
        """
        thr = ctx.thresholds
        min_acc = _get_min_accuracy(thr)
        margin = float(thr.get("s5_switch_margin", _DEFAULT_SWITCH_MARGIN))

        all_models = list(ctx.available_models)
        ema_acc = ctx.ema_accuracy
        ema_eng = ctx.ema_energy

        # Candidates meeting accuracy requirement (Fix 2.1: min_accuracy not min_score)
        candidates = [m for m in all_models if ema_acc.get(m, 0.0) >= min_acc]
        pool = candidates if candidates else all_models  # Fix 2.5: fallback = all models

        # Among pool, find the one with minimum energy
        chosen = min(pool, key=lambda m: ema_eng.get(m, float("inf")))

        # Fix 2.3: if that's the current model, noop
        if chosen == ctx.current_model:
            return PlanDecision(
                action="noop",
                reason=(
                    f"S5 violation_aware: energy violation — current model "
                    f"'{ctx.current_model}' is already the lowest-energy accurate option "
                    f"(ema_E={ema_eng.get(chosen, float('nan')):.4f}). "
                    "No better alternative; violation persists."
                ),
            )

        # Fix 2.4: hysteresis — switch only if candidate energy is meaningfully lower
        curr_eng = ema_eng.get(ctx.current_model, float("inf"))
        cand_eng = ema_eng.get(chosen, float("inf"))
        if not should_switch(d_current=curr_eng, d_candidate=cand_eng, margin=margin):
            return PlanDecision(
                action="noop",
                reason=(
                    f"S5 violation_aware: energy violation — candidate '{chosen}' "
                    f"(ema_E={cand_eng:.4f}) not better than current "
                    f"(ema_E={curr_eng:.4f}) by margin={margin}; staying."
                ),
            )

        return PlanDecision(
            action="switch",
            model=chosen,
            reason=(
                f"S5 violation_aware: energy violation — lowest_energy_model={chosen} "
                f"(ema_E={cand_eng:.4f}, ema_A={ema_acc.get(chosen, float('nan')):.4f})"
            ),
        )

    # ------------------------------------------------------------------
    # Score violation: find model with acceptable energy and best accuracy
    # ------------------------------------------------------------------

    def _handle_score_violation(self, ctx: PlanningContext) -> PlanDecision:
        """Score too low: find the most accurate model within energy budget.

        Fix 2.2: uses live ctx.current_energy_threshold, not stale config value.
        Fix 2.3: includes current model; returns noop if current is already best.
        Fix 2.4: hysteresis — only switch if accuracy improvement exceeds margin.
        Fix 2.5: if no model within energy budget, fall back to best-accuracy over all.
        """
        thr = ctx.thresholds
        e_threshold = ctx.current_energy_threshold  # Fix 2.2: live threshold
        margin = float(thr.get("s5_switch_margin", _DEFAULT_SWITCH_MARGIN))

        all_models = list(ctx.available_models)
        ema_acc = ctx.ema_accuracy
        ema_eng = ctx.ema_energy

        # Candidates within energy budget (Fix 2.2: live threshold)
        candidates = [m for m in all_models if ema_eng.get(m, 1.0) <= e_threshold]
        pool = candidates if candidates else all_models  # Fix 2.5: fallback = all models

        # Among pool, find the one with highest accuracy
        chosen = max(pool, key=lambda m: ema_acc.get(m, 0.0))

        # Fix 2.3: if that's the current model, noop
        if chosen == ctx.current_model:
            return PlanDecision(
                action="noop",
                reason=(
                    f"S5 violation_aware: score violation — current model "
                    f"'{ctx.current_model}' is already the most accurate within "
                    f"energy budget={e_threshold:.4f} "
                    f"(ema_A={ema_acc.get(chosen, float('nan')):.4f}). "
                    "No better alternative; violation persists."
                ),
            )

        # Fix 2.4: hysteresis — switch only if candidate accuracy is meaningfully higher
        curr_acc = ema_acc.get(ctx.current_model, 0.0)
        cand_acc = ema_acc.get(chosen, 0.0)
        # For accuracy: higher is better → d = -acc; should_switch(−curr, −cand, margin)
        if not should_switch(d_current=-curr_acc, d_candidate=-cand_acc, margin=margin):
            return PlanDecision(
                action="noop",
                reason=(
                    f"S5 violation_aware: score violation — candidate '{chosen}' "
                    f"(ema_A={cand_acc:.4f}) not better than current "
                    f"(ema_A={curr_acc:.4f}) by margin={margin}; staying."
                ),
            )

        return PlanDecision(
            action="switch",
            model=chosen,
            reason=(
                f"S5 violation_aware: score violation — best_acc_model={chosen} "
                f"(ema_A={cand_acc:.4f}, ema_E={ema_eng.get(chosen, float('nan')):.4f})"
            ),
        )

    # ------------------------------------------------------------------
    # Drift violation (unchanged from original)
    # ------------------------------------------------------------------

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

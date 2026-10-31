"""
core/planners/violation_aware.py — S5: violation-aware planning with a run-level
accuracy budget (redesign, 2026-10-26).

The previous S5 (now `violation_aware_gated`, S5a — kept as the "hard per-cycle
constraints" ablation) got stuck on the LSTM: its min_accuracy gate was a hard
per-cycle constraint ridge and SVR almost never passed, it no-oped when the gate
left no candidate (121/131 PeMS energy violations), it switched instead of
adapting on drift, and it trusted frozen estimates. This version:

- treats the accuracy target as a run-level budget: a signed, capped credit B
  (core/planners/credit.py) lets the run spend banked accuracy on cheap stints;
- judges each model against its own required accuracy
  a*_m = (min_score − (1−β)(1 − ê_m)) / β instead of the single min_accuracy;
- treats stale estimates as unknown (never observed, or last served more than
  staleness_window = 2,600 rows ago; same rule as pareto) — unknown models'
  EMAs are never compared, they are ordered by cost class;
- always adapts on drift (VMR replace / retrain, as HarmonE), never switches;
- always leaves on an energy violation unless already on the cheapest model
  (no 0.02 energy margin — HarmonE has none, and it produced no-ops).

Decision order each cycle (recovery hold, busy guard and drift-over-violation
precedence are the harness's, run_experiment._plan):
  1. drift   → replace if the VMR has a version within tau_drift, else retrain
  2. probe   → with explore_prob, the stalest other model (as pareto)
  3. no violation → noop
  4. energy violation, candidates C = cheaper others (ema_energy if known,
     cost class if unknown):
       a. C empty → noop "already_cheapest"
       b. known with projected_B >= 0 → lowest ema_energy (tie: higher â)  "affordable_cheapest"
       c. an unknown in C and B >= 0 → cheapest unknown                    "unknown_probe"
       d. else → known with highest â, or the cheapest unknown             "infeasible_fallback"
  5. score violation, C = more accurate others (â if known) or costlier
     (cost class if unknown):
       C empty → noop "current_is_best"
       a. B < 0 → known with highest â, or the costliest unknown           "repay_debt"
       b. known with â_m >= a*_m → the cheapest such                       "cheapest_sufficient"
       c. an unknown in C → cheapest unknown                               "unknown_probe"
       d. known with highest â if it beats the current â by more than
          s5_switch_margin (0.02) → "most_accurate", else noop "current_is_best"

Every decision carries metadata for mape_events.csv: s5_reason, s5_projected
(projected_B of the chosen model), s5_required_acc (a*_m of the chosen model),
s5_unknown_models.
"""

from __future__ import annotations

from .base import Planner, PlanningContext, PlanDecision, REGISTRY, cost_rank as _cost_rank
from .credit import (credit_horizon, projected_credit, required_accuracy,
                     split_known_unknown, target_accuracy)
from .exploration import maybe_explore

_DEFAULT_SWITCH_MARGIN = 0.02
_LABEL = "S5 violation_aware"


class ViolationAwarePlanner(Planner):
    """S5: violation routing under a run-level accuracy credit."""

    name = "violation_aware"

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        known, unknown = split_known_unknown(ctx)

        if ctx.violation == "drift":
            return self._decide(ctx, unknown, self._drift(ctx))

        probe = maybe_explore(ctx, float(ctx.thresholds.get("explore_prob", 0.0)), _LABEL)
        if probe is not None:
            return self._decide(ctx, unknown, (probe, "probe"))

        if ctx.violation is None:
            return self._decide(ctx, unknown, (
                PlanDecision(action="noop", reason=f"{_LABEL}: no violation"), "no_violation"))

        if ctx.violation == "energy":
            return self._decide(ctx, unknown, self._energy(ctx, known, unknown))
        return self._decide(ctx, unknown, self._score(ctx, known, unknown))

    # ------------------------------------------------------------------

    def _energy(self, ctx: PlanningContext, known: list[str], unknown: list[str]):
        thr, cur = ctx.thresholds, ctx.current_model
        acc, eng = ctx.ema_accuracy, ctx.ema_energy
        a_target = target_accuracy(thr)
        h = credit_horizon(thr)
        b = float(ctx.accuracy_credit)

        c_known = [m for m in known if m != cur and eng.get(m, 1.0) < eng.get(cur, 1.0)]
        c_unknown = [m for m in unknown if _cost_rank(thr, m) < _cost_rank(thr, cur)]
        if not c_known and not c_unknown:
            return self._noop(f"energy violation — {cur} is already the cheapest model"), "already_cheapest"

        affordable = [m for m in c_known if projected_credit(b, acc.get(m, 0.0), a_target, h) >= 0]
        if affordable:
            pick = min(affordable, key=lambda m: (eng.get(m, 1.0), -acc.get(m, 0.0)))
            return self._switch(pick, f"energy violation — {pick} is the cheapest affordable model "
                                      f"(B={b:.4f})"), "affordable_cheapest"
        if c_unknown and b >= 0:
            pick = min(c_unknown, key=lambda m: _cost_rank(thr, m))
            return self._switch(pick, f"energy violation — re-measuring unknown cheaper model {pick} "
                                      f"(B={b:.4f})"), "unknown_probe"
        if c_known:
            pick = max(c_known, key=lambda m: (acc.get(m, 0.0), -eng.get(m, 1.0)))
        else:
            pick = min(c_unknown, key=lambda m: _cost_rank(thr, m))
        return self._switch(pick, f"energy violation — nothing affordable (B={b:.4f}); "
                                  f"falling back to {pick}"), "infeasible_fallback"

    def _score(self, ctx: PlanningContext, known: list[str], unknown: list[str]):
        thr, cur = ctx.thresholds, ctx.current_model
        acc, eng = ctx.ema_accuracy, ctx.ema_energy
        b = float(ctx.accuracy_credit)
        margin = float(thr.get("s5_switch_margin", _DEFAULT_SWITCH_MARGIN))

        c_known = [m for m in known if m != cur and acc.get(m, 0.0) > acc.get(cur, 0.0)]
        c_unknown = [m for m in unknown if _cost_rank(thr, m) > _cost_rank(thr, cur)]
        if not c_known and not c_unknown:
            return self._noop(f"score violation — no more accurate or costlier model than {cur}"), \
                "current_is_best"

        if b < 0:
            if c_known:
                pick = max(c_known, key=lambda m: acc.get(m, 0.0))
            else:
                pick = max(c_unknown, key=lambda m: _cost_rank(thr, m))
            return self._switch(pick, f"score violation — run owes accuracy (B={b:.4f}); "
                                      f"switching to {pick}"), "repay_debt"

        sufficient = [m for m in c_known if acc.get(m, 0.0) >= required_accuracy(m, eng, thr)]
        if sufficient:
            pick = min(sufficient, key=lambda m: eng.get(m, 1.0))
            return self._switch(pick, f"score violation — {pick} is the cheapest model meeting its "
                                      f"required accuracy"), "cheapest_sufficient"
        if c_unknown:
            pick = min(c_unknown, key=lambda m: _cost_rank(thr, m))
            return self._switch(pick, f"score violation — re-measuring unknown model {pick}"), "unknown_probe"

        pick = max(c_known, key=lambda m: acc.get(m, 0.0))
        if acc.get(pick, 0.0) > acc.get(cur, 0.0) + margin:
            return self._switch(pick, f"score violation — {pick} is the most accurate model"), "most_accurate"
        return self._noop(f"score violation — {pick} does not beat {cur} by more than {margin}"), \
            "current_is_best"

    def _drift(self, ctx: PlanningContext):
        dr = ctx.drift_result or {}
        if dr.get("action") in ("replace", "switch_version"):
            path = dr.get("version")
            return PlanDecision(action="replace", version_path=path,
                                reason=f"{_LABEL}: drift — VMR replace → {path}"), "drift_replace"
        return PlanDecision(action="retrain", reason=f"{_LABEL}: drift — no VMR match, retrain"), "drift_retrain"

    # ------------------------------------------------------------------

    @staticmethod
    def _switch(model: str, why: str) -> PlanDecision:
        return PlanDecision(action="switch", model=model, reason=f"{_LABEL}: {why}")

    @staticmethod
    def _noop(why: str) -> PlanDecision:
        return PlanDecision(action="noop", reason=f"{_LABEL}: {why}")

    @staticmethod
    def _decide(ctx: PlanningContext, unknown: list[str], result) -> PlanDecision:
        """Attach the logging metadata: reason label, and projected_B / a*_m of
        the chosen model when it is a known model being switched to."""
        decision, label = result
        meta = {"s5_reason": label, "s5_unknown_models": ";".join(unknown)}
        target = decision.model if decision.action == "switch" else None
        if target is not None and target not in unknown:
            meta["s5_projected"] = round(projected_credit(
                float(ctx.accuracy_credit), ctx.ema_accuracy.get(target, 0.0),
                target_accuracy(ctx.thresholds), credit_horizon(ctx.thresholds)), 6)
            meta["s5_required_acc"] = round(required_accuracy(target, ctx.ema_energy, ctx.thresholds), 6)
        decision.metadata = {**(decision.metadata or {}), **meta}
        return decision


REGISTRY["violation_aware"] = ViolationAwarePlanner

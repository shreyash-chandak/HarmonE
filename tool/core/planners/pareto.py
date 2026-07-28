"""
core/planners/pareto.py — S6: Pareto-optimal model selection (primary novel contribution).

Builds an empirical Pareto front over available models in (accuracy, energy)
space using separated EMA signals, then selects the Pareto-optimal model that
minimises the Chebyshev distance to the goal point (S_min, E_ref):

    d(m) = max(w_acc * max(0, S_min - acc[m]), w_e * max(0, energy[m] - E_ref))

Choice of max(0, ·): models that exceed the goal on an axis are not rewarded
for overshoot — overshooting the accuracy target while burning more energy
should not improve the score. This differs from a symmetric Chebyshev distance
and is deliberate (documented in CHANGES_FROM_PAPER.md when the paper compares
S6 to S5/S4).

Tie-break: lower energy. If no Pareto front can be constructed (0 or 1 model),
falls back to greedy_switch logic.

Drift violations delegate to VMR replace or retrain (same as S4/S5).
"""

from __future__ import annotations

from .base import Planner, PlanningContext, PlanDecision, REGISTRY


def _is_dominated(acc_a: float, eng_a: float, acc_b: float, eng_b: float) -> bool:
    """Return True if (acc_a, eng_a) is dominated by (acc_b, eng_b).

    Model b dominates model a when b is at least as accurate AND uses no more
    energy, with strict improvement on at least one axis.
    (Higher accuracy is better; lower energy is better.)
    """
    return acc_b >= acc_a and eng_b <= eng_a and (acc_b > acc_a or eng_b < eng_a)


def build_pareto_front(models: list[str], acc: dict, eng: dict) -> list[str]:
    """Return the non-dominated subset of models.

    A model is on the Pareto front if no other model in the set has both
    ≥ accuracy and ≤ energy with at least one strict improvement.
    """
    front = []
    for m in models:
        dominated = any(
            _is_dominated(acc.get(m, 0.0), eng.get(m, 1.0), acc.get(n, 0.0), eng.get(n, 1.0))
            for n in models if n != m
        )
        if not dominated:
            front.append(m)
    return front


def chebyshev_distance(
    m: str,
    acc: dict,
    eng: dict,
    s_min: float,
    e_ref: float,
    w_acc: float = 1.0,
    w_e: float = 1.0,
) -> float:
    """Asymmetric Chebyshev distance from model m to goal point (s_min, e_ref)."""
    acc_gap = max(0.0, s_min - acc.get(m, 0.0))   # penalty for being below accuracy goal
    eng_gap = max(0.0, eng.get(m, 1.0) - e_ref)   # penalty for exceeding energy goal
    return max(w_acc * acc_gap, w_e * eng_gap)


class ParetoPlanner(Planner):
    """S6: Pareto-optimal model selection via Chebyshev distance to goal point."""

    name = "pareto"

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        if ctx.violation is None:
            return PlanDecision(action="noop", reason="S6 pareto: no violation")

        if ctx.violation == "drift":
            return self._handle_drift(ctx)

        return self._select_pareto(ctx)

    def _select_pareto(self, ctx: PlanningContext) -> PlanDecision:
        thr = ctx.thresholds
        s_min = thr.get("min_score", 0.5)
        e_ref = thr.get("E_ref", 0.7)
        w_acc = thr.get("w_acc", 1.0)
        w_e = thr.get("w_e", 1.0)

        ema_acc = ctx.ema_accuracy
        ema_eng = ctx.ema_energy

        # Include all models (not just non-current) to allow staying if current is best
        all_models = ctx.available_models
        if not all_models:
            return PlanDecision(action="noop", reason="S6 pareto: no models available")

        front = build_pareto_front(all_models, ema_acc, ema_eng)
        if not front:
            front = all_models  # degenerate: use all

        # Select Pareto-optimal model minimising Chebyshev distance; tie-break: lower energy
        chosen = min(
            front,
            key=lambda m: (
                chebyshev_distance(m, ema_acc, ema_eng, s_min, e_ref, w_acc, w_e),
                ema_eng.get(m, 1.0),
            ),
        )

        dist = chebyshev_distance(chosen, ema_acc, ema_eng, s_min, e_ref, w_acc, w_e)

        if chosen == ctx.current_model:
            return PlanDecision(
                action="noop",
                reason=f"S6 pareto: current model is already Pareto-optimal (d={dist:.4f})",
            )

        return PlanDecision(
            action="switch",
            model=chosen,
            reason=(
                f"S6 pareto: violation={ctx.violation}, "
                f"selected {chosen} from front={front} (d={dist:.4f})"
            ),
        )

    def _handle_drift(self, ctx: PlanningContext) -> PlanDecision:
        dr = ctx.drift_result
        if dr is None:
            return PlanDecision(action="noop", reason="S6 pareto: drift but no drift_result")
        if dr.get("action") in ("replace", "switch_version"):
            path = dr.get("version")
            return PlanDecision(
                action="replace",
                version_path=path,
                reason=f"S6 pareto: VMR replace → {path}",
            )
        return PlanDecision(action="retrain", reason="S6 pareto: no VMR match, retrain")


REGISTRY["pareto"] = ParetoPlanner

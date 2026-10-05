"""
core/planners/pareto.py — S6: Pareto-optimal model selection (Phase 3 redesign).

Builds an empirical Pareto front over available models in (accuracy, energy)
space, then selects the model minimising an augmented reference-point
scalarization:

    D(m) = max(w_A * d_A(m), w_E * d_E(m))
           + ε * (w_A * d_A(m) + w_E * d_E(m))

where:
    d_A(m) = max(0, accuracy_reference  - acc[m])   # accuracy deficit
    d_E(m) = max(0, energy[m] - energy_reference)   # energy excess
    ε      = pareto_augmentation (default 0.01)      # linear tie-break term

Phase 3 changes from the old one-sided Chebyshev (documented in CHANGES_FROM_PAPER.md):
  3.1 Scalarization: augmented reference-point D(m) replaces asymmetric Chebyshev.
      Reference point reads from pareto_accuracy_reference / energy_reference
      (not from min_score, which is the combined HarmonE score threshold).
  3.2 Estimate validation: models without observed (non-sentinel) accuracy AND
      energy estimates are flagged as 'unknown' and routed to a controlled
      exploration path; they never enter the Pareto front with fabricated defaults.
  3.3 Violation-conditioned hard-constraint filtering: the Pareto front is
      filtered before selection:
        energy violation → keep only models with acc ≥ min_accuracy
        score  violation → keep only models with eng ≤ max_energy
      If the filtered set is empty, fall back to minimising the violated quantity
      over the full front (constraint relaxation).
  3.4 Hysteresis: switch only when D(candidate) + pareto_switch_margin < D(current).
  3.5 Switching-cost penalty: D'(m) = D(m) + λ_C * C_switch(m) where C_switch is
      derived from the model's cost_class annotation in the config.
"""

from __future__ import annotations

import logging

from .base import Planner, PlanningContext, PlanDecision, REGISTRY, is_observed
from .hysteresis import should_switch
from .exploration import maybe_explore

logger = logging.getLogger(__name__)

_SENTINEL_EMA = 0.5       # initial EMA value for unobserved models
_SENTINEL_TOL = 1e-9      # tolerance for detecting sentinel


# ── Pareto dominance ──────────────────────────────────────────────────────────

def _is_dominated(acc_a: float, eng_a: float, acc_b: float, eng_b: float) -> bool:
    """True if model-b dominates model-a (b ≥ acc and b ≤ eng, strictly better on one)."""
    return acc_b >= acc_a and eng_b <= eng_a and (acc_b > acc_a or eng_b < eng_a)


def build_pareto_front(models: list[str], acc: dict, eng: dict) -> list[str]:
    """Return non-dominated subset of models.  Exported for tests."""
    front = []
    for m in models:
        if not any(
            _is_dominated(acc.get(m, 0.0), eng.get(m, 1.0), acc.get(n, 0.0), eng.get(n, 1.0))
            for n in models if n != m
        ):
            front.append(m)
    return front


# ── Augmented reference-point scalarization ───────────────────────────────────

def augmented_distance(
    m: str,
    acc: dict,
    eng: dict,
    acc_ref: float,
    eng_ref: float,
    w_acc: float = 1.0,
    w_eng: float = 1.0,
    epsilon: float = 0.01,
) -> float:
    """Augmented reference-point distance D(m) (Phase 3.1).

    Lower is better.  Exported for tests.
    """
    d_a = max(0.0, acc_ref - acc.get(m, 0.0))
    d_e = max(0.0, eng.get(m, 1.0) - eng_ref)
    return max(w_acc * d_a, w_eng * d_e) + epsilon * (w_acc * d_a + w_eng * d_e)


# Legacy alias kept for tests that still import chebyshev_distance.
def chebyshev_distance(m, acc, eng, s_min, e_ref, w_acc=1.0, w_e=1.0):
    """Backward-compat alias → augmented_distance with ε=0 and the same parameters."""
    return augmented_distance(m, acc, eng, s_min, e_ref, w_acc, w_e, epsilon=0.0)


# ── Switch-cost from cost_class ───────────────────────────────────────────────

_COST_CLASS_MAP = {"light": 0.0, "medium": 0.05, "heavy": 0.10}


def _switch_cost(model_name: str, thresholds: dict) -> float:
    """Return a normalised switching cost based on cost_class annotation."""
    models_cfg = thresholds.get("models", {})
    cost_class = models_cfg.get(model_name, {}).get("cost_class", "medium")
    return _COST_CLASS_MAP.get(cost_class, 0.05)


# ── Main planner ──────────────────────────────────────────────────────────────

class ParetoPlanner(Planner):
    """S6: Pareto-optimal model selection via augmented reference-point scalarization."""

    name = "pareto"

    def plan(self, ctx: PlanningContext) -> PlanDecision:
        if ctx.violation == "drift":
            return self._handle_drift(ctx)

        # Audit A2/A7 (2026-09-28): periodically re-probe the stalest inactive
        # model (never-observed first), like the original HarmonE's per-cycle
        # exploration. This is what lets "unknown" (sentinel/stale) models
        # acquire estimates and enter the front; previously they were only
        # explored when ALL models were unknown, which never happens. Disabled
        # unless explore_prob > 0.
        probe = maybe_explore(
            ctx, float(ctx.thresholds.get("explore_prob", 0.0)), "S6 pareto"
        )
        if probe is not None:
            return probe

        if ctx.violation is None:
            return PlanDecision(action="noop", reason="S6 pareto: no violation")

        return self._select_pareto(ctx)

    def _select_pareto(self, ctx: PlanningContext) -> PlanDecision:
        thr = ctx.thresholds

        # Phase 3.1: reference-point keys (NOT min_score)
        acc_ref = float(thr.get("pareto_accuracy_reference",
                                thr.get("min_accuracy",
                                        thr.get("min_score", 0.5))))
        eng_ref = float(thr.get("energy_reference",
                                thr.get("E_ref", 0.5)))
        w_acc = float(thr.get("w_acc", 1.0))
        w_eng = float(thr.get("w_e", thr.get("w_eng", 1.0)))
        epsilon = float(thr.get("pareto_augmentation", 0.01))
        margin = float(thr.get("pareto_switch_margin", 0.02))
        lambda_c = float(thr.get("pareto_switch_cost_lambda", 0.1))
        min_acc = float(thr.get("min_accuracy", thr.get("min_score", 0.0)))
        max_eng = float(thr.get("max_energy", ctx.current_energy_threshold))

        ema_acc = ctx.ema_accuracy
        ema_eng = ctx.ema_energy
        all_models = list(ctx.available_models)

        # Phase 3.2: separate known models from unknowns (EMA at sentinel or stale)
        known, unknown = _split_known_unknown(all_models, ema_acc, ema_eng, ctx)

        if not known:
            # All models are unknown — explore the first unknown and return
            chosen = unknown[0] if unknown else ctx.current_model
            return PlanDecision(
                action="switch" if chosen != ctx.current_model else "noop",
                model=chosen if chosen != ctx.current_model else None,
                reason=f"S6 pareto: all estimates unknown, exploring {chosen}",
            )

        # Phase 3.3: build front from known models only
        front = build_pareto_front(known, ema_acc, ema_eng)
        if not front:
            front = known

        # Hard-constraint filtering on the Pareto front
        constrained_front = _apply_hard_constraints(
            front, ema_acc, ema_eng, ctx.violation, min_acc, max_eng
        )

        # Phase 3.5: compute augmented distance + switch-cost penalty for each model
        def score(m: str) -> float:
            d = augmented_distance(m, ema_acc, ema_eng, acc_ref, eng_ref, w_acc, w_eng, epsilon)
            c = _switch_cost(m, thr) * lambda_c if m != ctx.current_model else 0.0
            return d + c

        chosen = min(constrained_front, key=score)
        chosen_score = score(chosen)
        curr_score = score(ctx.current_model) if ctx.current_model in known else float("inf")

        # Audit A7: if the known front would keep the current model while it
        # violates, try an unknown (never-observed or stale) model first, so
        # unknowns can enter the front instead of waiting for a random probe.
        if chosen == ctx.current_model and unknown:
            target = unknown[0]
            return PlanDecision(
                action="switch", model=target,
                reason=f"S6 pareto: violation={ctx.violation}, current model is the known optimum; "
                       f"exploring unknown '{target}'",
            )

        # Phase 3.4: hysteresis guard
        if not should_switch(d_current=curr_score, d_candidate=chosen_score, margin=margin):
            return PlanDecision(
                action="noop",
                reason=(
                    f"S6 pareto: candidate '{chosen}' (D={chosen_score:.4f}) "
                    f"not better than current '{ctx.current_model}' "
                    f"(D={curr_score:.4f}) by margin={margin} — within hysteresis band."
                ),
            )

        if chosen == ctx.current_model:
            return PlanDecision(
                action="noop",
                reason=f"S6 pareto: current model is Pareto-optimal (D={chosen_score:.4f})",
            )

        d_chosen = augmented_distance(chosen, ema_acc, ema_eng, acc_ref, eng_ref, w_acc, w_eng, epsilon)
        return PlanDecision(
            action="switch",
            model=chosen,
            reason=(
                f"S6 pareto: violation={ctx.violation}, "
                f"front={front}, selected {chosen} "
                f"(D={d_chosen:.4f}, score={chosen_score:.4f})"
            ),
            metadata={"pareto_front": front, "pareto_front_size": len(front)},
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_sentinel(val: float) -> bool:
    return abs(val - _SENTINEL_EMA) < _SENTINEL_TOL


def _split_known_unknown(
    models: list[str],
    ema_acc: dict,
    ema_eng: dict,
    ctx: PlanningContext,
) -> tuple[list[str], list[str]]:
    """Separate models into 'known' (observed, non-stale) and 'unknown'.

    A model is unknown if:
      - It has never been observed (ctx.observed, audit A5), OR
      - Its last_observed_step is more than staleness_window steps ago.
    """
    known, unknown = [], []
    staleness_window = ctx.staleness_window
    current_step = ctx.current_step
    last_seen = ctx.last_observed_step

    for m in models:
        # Audit A5: the harness's explicit observed flag (legacy callers fall
        # back to the 0.5-sentinel test inside is_observed).
        sentinel = m != ctx.current_model and not is_observed(ctx, m)

        stale = False
        if m in last_seen and staleness_window > 0:
            stale = (current_step - last_seen[m]) > staleness_window

        if sentinel or stale:
            unknown.append(m)
        else:
            known.append(m)

    return known, unknown


def _apply_hard_constraints(
    front: list[str],
    ema_acc: dict,
    ema_eng: dict,
    violation: str | None,
    min_acc: float,
    max_eng: float,
) -> list[str]:
    """Filter the Pareto front based on the violation type (Phase 3.3).

    Falls back to minimising the violated quantity if the filtered set is empty.
    """
    if violation == "energy":
        # Energy too high: prefer models that meet accuracy requirement
        constrained = [m for m in front if ema_acc.get(m, 0.0) >= min_acc]
        if not constrained:
            # No accurate model on front — pick the one with best accuracy (constraint relaxation)
            constrained = sorted(front, key=lambda m: -ema_acc.get(m, 0.0))
    elif violation == "score":
        # Score too low: prefer models within energy budget
        constrained = [m for m in front if ema_eng.get(m, 1.0) <= max_eng]
        if not constrained:
            # No in-budget model on front — pick the one with lowest energy (constraint relaxation)
            constrained = sorted(front, key=lambda m: ema_eng.get(m, 1.0))
    else:
        constrained = front

    return constrained if constrained else front


REGISTRY["pareto"] = ParetoPlanner

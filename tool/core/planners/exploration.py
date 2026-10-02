"""
core/planners/exploration.py — periodic re-probing of inactive models (audit A2/A7).

A model's EMA estimates only update while it is the active model, so a model
that had one unlucky monitoring window keeps that estimate forever and is
never chosen again (observed: uci ridge frozen at ema_accuracy=0.015 while its
typical batch accuracy is ~0.80). The original HarmonE avoided this by rolling
its exploration probability on EVERY MAPE cycle, whether or not a threshold
was violated (HarmonE/mape/plan.py::plan_mape) — so idle models were
occasionally re-tried and re-measured.

`maybe_explore` gives the violation-driven planners (S5, S6) the same
behaviour, config-driven via `explore_prob` (default 0.0 → disabled, i.e. the
pre-2026-09-28 behaviour). The probe targets the model observed least recently
(never-observed first), so estimates are refreshed systematically rather than
at random. After the probe the planner's normal violation logic decides
whether to stay on the probed model or move away.

Documented randomness: this is the one place S5/S6 call `random`.
"""

from __future__ import annotations

import random

from .base import PlanningContext, PlanDecision


def stalest_model(ctx: PlanningContext) -> str | None:
    """The non-current model observed least recently (never-observed first)."""
    candidates = [m for m in ctx.available_models if m != ctx.current_model]
    if not candidates:
        return None
    last_seen = ctx.last_observed_step or {}
    return min(candidates, key=lambda m: last_seen.get(m, -1))


def maybe_explore(ctx: PlanningContext, prob: float, label: str) -> PlanDecision | None:
    """With probability `prob`, return a switch to the stalest model; else None."""
    if prob <= 0.0 or random.random() >= prob:
        return None
    target = stalest_model(ctx)
    if target is None:
        return None
    last = (ctx.last_observed_step or {}).get(target)
    seen = "never observed" if last is None else f"last observed at step {last}"
    return PlanDecision(
        action="switch",
        model=target,
        reason=f"{label}: exploration probe (explore_prob={prob}) → {target} ({seen})",
    )

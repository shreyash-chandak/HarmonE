"""
core/planners/hysteresis.py — Shared switching hysteresis primitive (Phase 1.5).

Used by S5 (violation_aware), S6 (pareto), and S7 (bandit) to prevent
thrashing: a switch only fires when the candidate is strictly better than the
current model by at least `margin` on the relevant distance/score criterion.

S2 (random) and S3 (greedy) intentionally do NOT use hysteresis — their
naive semantics are the comparison substrate.
"""

from __future__ import annotations


def should_switch(d_current: float, d_candidate: float, margin: float) -> bool:
    """Return True only if the candidate is better than current by at least margin.

    For distance-based planners (S6 Pareto): d is a cost/distance; lower is
    better; switch when d_candidate + margin < d_current.

    For score-based planners (S5): pass negated scores so the same formula
    applies: d = -score, switch when -cand_score + margin < -curr_score
    i.e. cand_score > curr_score + margin.

    Args:
        d_current:   Distance/cost of the current model.
        d_candidate: Distance/cost of the candidate model.
        margin:      Required improvement margin (config: *_switch_margin key).

    Returns:
        True  → switch is warranted.
        False → candidate is not sufficiently better; stay or keep looking.
    """
    return d_candidate + margin < d_current

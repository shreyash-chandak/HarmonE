"""
core/planners/credit.py — run-level accuracy credit for S5 (2026-10-26).

The score / accuracy targets of the planners are really run-level averages:
HarmonE meets them by alternating between an accurate and a cheap model, not
by finding one model that satisfies both every cycle. S5 makes that explicit
with a signed accuracy credit B (the mirror of the bandit's energy debt Q,
but signed and capped):

    B ← clip(B + (acc_batch − A_target), −B_max, +B_max)        every cycle

A_target is the dataset's min_accuracy (the score target at the reference
energy E_ref = 0.5). B is updated from the batch accuracy of whichever model
actually served the monitored batch — lagging batches and batches inside the
recovery hold included — so it tracks the run's realised accuracy, not any
one model's.

Cap (written by experiments/calibrate_from_naive_runs.py as s5_credit_cap):

    B_max = 1.5 × H × max(0, A_target − acc_naive_ridge)

with H = s5_credit_horizon = 3 cycles (= switch_hold_cycles, the minimum stint
after an energy switch). The cap lets the run bank enough credit for about one
and a half held ridge stints, and no more, so credit cannot accumulate
indefinitely during long LSTM stretches. If acc_naive_ridge >= A_target the
cap falls back to 0.5 (with a warning).

A model m is affordable over the horizon if

    projected_B(m) = B − H × max(0, A_target − â_m) >= 0.

All functions here are pure apart from the warnings; the harness keeps B in
mape_info["s5_credit"] and passes it to the planner as ctx.accuracy_credit.
"""

from __future__ import annotations

import logging

from .base import PlanningContext

logger = logging.getLogger(__name__)

S5_CREDIT_FACTOR = 1.5          # stints of credit the cap allows (see module docstring)
DEFAULT_CREDIT_HORIZON = 3      # cycles; = switch_hold_cycles
FALLBACK_CREDIT_CAP = 0.5       # cap when the cheap model already meets A_target / no cap configured

_warned_missing_cap = False


def target_accuracy(thresholds: dict) -> float:
    """A_target = min_accuracy. Configs without it (the live managed system's
    thresholds.json) get the same quantity from min_score, inverting the
    calibration rule min_score = β·min_accuracy + (1−β)(1 − E_ref)."""
    if "min_accuracy" in thresholds:
        return float(thresholds["min_accuracy"])
    beta = float(thresholds.get("beta", 0.95))
    e_ref = float(thresholds.get("E_ref", 0.5))
    return (float(thresholds.get("min_score", 0.78)) - (1.0 - beta) * (1.0 - e_ref)) / beta


def credit_horizon(thresholds: dict) -> int:
    return int(thresholds.get("s5_credit_horizon", DEFAULT_CREDIT_HORIZON))


def credit_cap_from_naive(a_target: float, acc_naive_ridge: float,
                          horizon: int = DEFAULT_CREDIT_HORIZON) -> float:
    """B_max = 1.5 × H × max(0, A_target − acc_naive_ridge); 0.5 (with a
    warning) if the cheap model already meets the target."""
    gap = a_target - acc_naive_ridge
    if gap <= 0:
        logger.warning(
            "S5 credit cap: naive ridge accuracy %.4f >= A_target %.4f — using the "
            "fallback cap %.2f", acc_naive_ridge, a_target, FALLBACK_CREDIT_CAP)
        return FALLBACK_CREDIT_CAP
    return S5_CREDIT_FACTOR * horizon * gap


def credit_cap(thresholds: dict) -> float:
    """The configured cap (s5_credit_cap), or the fallback with a one-time warning."""
    global _warned_missing_cap
    if "s5_credit_cap" in thresholds:
        return float(thresholds["s5_credit_cap"])
    if not _warned_missing_cap:
        logger.warning("S5 credit: 's5_credit_cap' not in the config (run "
                       "experiments/calibrate_from_naive_runs.py) — using %.2f", FALLBACK_CREDIT_CAP)
        _warned_missing_cap = True
    return FALLBACK_CREDIT_CAP


def update_credit(credit: float, accuracy: float, a_target: float, cap: float) -> float:
    """One cycle's update: B + (acc_batch − A_target), clipped to [−cap, +cap]."""
    return max(-cap, min(cap, credit + (accuracy - a_target)))


def update_run_credit(mape_info: dict, accuracy: float | None, thresholds: dict) -> None:
    """Apply update_credit to mape_info["s5_credit"] (Monitor step, every cycle
    with a monitored batch). No-op when there is no accuracy."""
    if accuracy is None:
        return
    mape_info["s5_credit"] = update_credit(
        float(mape_info.get("s5_credit", 0.0)), float(accuracy),
        target_accuracy(thresholds), credit_cap(thresholds))


def projected_credit(credit: float, accuracy_estimate: float, a_target: float, horizon: int) -> float:
    """projected_B(m) = B − H × max(0, A_target − â_m)."""
    return credit - horizon * max(0.0, a_target - accuracy_estimate)


def required_accuracy(model: str, ema_energy: dict, thresholds: dict) -> float:
    """a*_m = (min_score − (1−β)(1 − ê_m)) / β: the accuracy at which model m,
    at its own energy estimate ê_m, would not itself trigger a score violation.
    (min_accuracy is the same formula at ê = E_ref = 0.5.)"""
    beta = float(thresholds.get("beta", 0.95))
    min_score = float(thresholds.get("min_score", 0.78))
    e_hat = float(ema_energy.get(model, 1.0))
    return (min_score - (1.0 - beta) * (1.0 - e_hat)) / beta


def split_known_unknown(ctx: PlanningContext) -> tuple[list[str], list[str]]:
    """Known / unknown models as in pareto (never observed, or last served more
    than staleness_window rows ago → unknown); the current model is always known."""
    from .pareto import _split_known_unknown
    known, unknown = _split_known_unknown(
        list(ctx.available_models), ctx.ema_accuracy, ctx.ema_energy, ctx)
    if ctx.current_model in unknown:
        unknown.remove(ctx.current_model)
        known.append(ctx.current_model)
    return known, unknown

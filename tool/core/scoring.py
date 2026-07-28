"""
core/scoring.py — Pure scoring functions shared across all managed systems.

All functions here are pure (no file I/O) so they can be unit-tested in isolation.
"""

from __future__ import annotations


def update_energy_threshold(
    current: float,
    e_ref: float,
    e_used: float,
    delta: float,
    lo: float = 0.1,
    hi: float = 1.0,
) -> float:
    """Eq. 3 from the paper: adaptive energy boundary update.

    tau_E(i+1) = clamp(tau_E(i) + delta * (E_ref - E_used), lo, hi)

    When E_used < E_ref the threshold rises (system is energy-efficient → relax limit).
    When E_used > E_ref the threshold falls (system is spending too much → tighten limit).

    Args:
        current: Current energy threshold tau_E(i).
        e_ref:   Reference energy level (normalised, from thresholds.json "E_ref").
        e_used:  Observed normalised energy this cycle.
        delta:   Step size (from thresholds.json "delta", typically 0.1).
        lo:      Minimum allowed threshold (hard floor, default 0.1).
        hi:      Maximum allowed threshold (hard ceiling, default 1.0).

    Returns:
        Updated threshold tau_E(i+1), clamped to [lo, hi].
    """
    raw = current + delta * (e_ref - e_used)
    return max(lo, min(hi, raw))


def compute_harmone_score(
    accuracy: float,
    normalized_energy: float,
    beta: float,
) -> float:
    """Instantaneous HarmonE model score.

    S_i = beta * A_i + (1 - beta) * (1 - E_norm_i)

    Args:
        accuracy:          R² (regression) or proxy accuracy (CV), in [0, 1].
        normalized_energy: Normalised energy for this interval, in [0, 1].
        beta:              Accuracy weight (0 = energy-only, 1 = accuracy-only).

    Returns:
        Score in [0, 1].
    """
    return beta * accuracy + (1.0 - beta) * (1.0 - normalized_energy)


def update_ema(
    prev_score: float,
    new_observation: float,
    gamma: float,
) -> float:
    """Exponential Moving Average update.

    EMA(i) = gamma * new_observation + (1 - gamma) * EMA(i-1)

    Args:
        prev_score:      EMA value from the previous cycle.
        new_observation: Newly observed raw score.
        gamma:           Smoothing factor (closer to 1 → more reactive).

    Returns:
        Updated EMA value.
    """
    return gamma * new_observation + (1.0 - gamma) * prev_score


def update_separated_emas(
    mape_info: dict,
    model: str,
    accuracy: float,
    normalized_energy: float,
    gamma: float,
) -> dict:
    """Update separated accuracy and energy EMA signals alongside the legacy combined score.

    Phase 2.4: maintains ema_accuracy[model] and ema_energy[model] in mape_info
    for use by violation_aware (S5) and pareto (S6) planners that need to reason
    about accuracy and energy independently.

    Args:
        mape_info:        The current mape_info dict (mutated in place and returned).
        model:            Name of the currently active model.
        accuracy:         This interval's raw accuracy (R² or proxy, in [0,1]).
        normalized_energy: This interval's normalised energy in [0,1].
        gamma:            EMA smoothing factor.

    Returns:
        The mutated mape_info dict (same object, for convenience).
    """
    if "ema_accuracy" not in mape_info:
        mape_info["ema_accuracy"] = {}
    if "ema_energy" not in mape_info:
        mape_info["ema_energy"] = {}

    prev_acc = mape_info["ema_accuracy"].get(model, accuracy)
    prev_eng = mape_info["ema_energy"].get(model, normalized_energy)

    mape_info["ema_accuracy"][model] = update_ema(prev_acc, accuracy, gamma)
    mape_info["ema_energy"][model] = update_ema(prev_eng, normalized_energy, gamma)
    return mape_info


def normalize_energy(
    avg_energy_uj: float,
    e_min: float,
    e_max: float,
) -> float:
    """Normalize raw energy (µJ) to [0, 1] using calibrated bounds.

    Clamps to [0, 1] — values outside the calibrated range are valid at inference
    time (do not re-fit bounds at runtime; see B7 for scaler analogue rationale).

    Args:
        avg_energy_uj: Mean per-inference energy for this interval (µJ).
        e_min:         Minimum calibrated energy (thresholds.json "E_m").
        e_max:         Maximum calibrated energy (thresholds.json "E_M").

    Returns:
        Normalised energy in [0, 1].
    """
    if e_max <= e_min:
        return 0.0
    raw = (avg_energy_uj - e_min) / (e_max - e_min)
    return max(0.0, min(1.0, raw))

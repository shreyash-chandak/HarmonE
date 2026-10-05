"""concurrent_harness/mape/mape_cycle.py — one Monitor->Analyse->Plan->Execute cycle.

Called ONLY by plan_thread.py (t1) as of 2026-09-28 — t2 (drift_thread.py) no
longer plans or executes; it only detects drift and hands its result to t1
(audit D4/D8). So every planning decision in a run goes through this one
function, exactly like the single cycle in experiments/run_experiment.py.

Order of operations mirrors run_experiment.py's cycle (audit D7/D8 parity):
  1. Monitor: _monitor_batch() on the batch, credited to the model that
     actually SERVED it (`monitor_model`, read from the rows).
  2. Analyse: _analyse_violation() — computed BEFORE the energy boundary is
     updated, as in single-threaded (the previous version updated the
     boundary first). Only raised when monitor_model is the commanded
     current_model: a batch served by the previous model (the one-batch
     actuation lag of the barrier, see inference.py) is evidence about THAT
     model, not about the one now in charge, so it updates EMAs but can't
     trigger a switch away from a model that hasn't served yet.
  3. drift_result comes from t2 (already analysed incl. the VMR lookup).
  4. _update_energy_boundary(); last_observed_step / steps_since_last_switch
     bookkeeping exactly as single-threaded.
  5. Plan via the shared _plan() (includes the switch hold), then the
     ema_head_start forced trial (CV), ported from run_experiment.py.
  6. Execute, then log mape_events.csv + planner_decisions.csv.

Locking (audit D4 fix): the mape_info snapshot is loaded INSIDE the lock
(kio.ENERGY_LOCK + the cross-process energy_lock), so nothing can change it
between read and write-back. Previously it was loaded before taking the lock,
and a waiting thread could later write its stale snapshot back over fresher
EMA values.

Energy: the whole cycle (monitor through execute) is inside one EnergyMeter,
like single-threaded's `_mape_em`, and accumulates into
event_counters.mape_k_energy_uJ (kio.add_mape_energy).
"""

from __future__ import annotations

import logging

from core.energy import EnergyMeter
from core.planners.base import PlanDecision
from core.vmr import VMR

from experiments.run_experiment import (
    _monitor_batch,
    _analyse_violation,
    _update_energy_boundary,
    _plan,
)

import knowledge_io as kio
from execute import execute_decision

logger = logging.getLogger(__name__)

_MONITOR_FIELDS = (
    "ema_scores", "ema_accuracy", "ema_energy", "current_energy_threshold",
    "last_observed_step", "steps_since_last_switch",
)


def _head_start(decision: PlanDecision, is_cv: bool, thresholds: dict,
                available_models: list[str], current_model: str, mape_info: dict) -> PlanDecision:
    """Verbatim port of run_experiment.py's ema_head_start block (audit D7):
    in CV runs, if the planner no-ops while some model has never been
    observed (EMA exactly at the 0.5 initialisation), force a trial of it."""
    if decision.action != "noop" or not is_cv:
        return decision
    head_start = thresholds.get("ema_head_start", 0.0)
    if head_start <= 0.0:
        return decision
    unobserved = [
        m for m in available_models
        if m != current_model and abs(mape_info["ema_scores"].get(m, 0.5) - 0.5) < 1e-9
    ]
    if not unobserved:
        return decision
    trial = unobserved[0]
    logger.info("ema_head_start: forcing trial of %s", trial)
    return PlanDecision(action="switch", model=trial,
                        reason=f"ema_head_start: bootstrap {trial} (never observed)")


def run_cycle(
    *,
    planner,
    current_model: str,
    monitor_model: str | None,
    available_models: list[str],
    mape_store: kio.MapeInfoStore,
    thresholds: dict,
    drift_result: dict,
    current_step: int,
    monitor_rows: list[dict],
    is_cv: bool,
    knowledge_dir,
    local_dataset_config: dict,
    cv_task: str,
    scaler,
    seq_length: int,
    vmr: VMR,
    value_history: list[float] | None,
    image_path_history: list[str] | None,
    luminance_history: list[float] | None,
    energy_backend: str,
) -> str:
    """Returns the (possibly new) commanded model name."""
    kp = kio.knowledge_paths(knowledge_dir)

    with kio.ENERGY_LOCK:  # t1/t2 within-process serialization
        with kio.energy_lock(knowledge_dir, blocking=True):  # cross-process
            em = EnergyMeter("mape_cycle", backend=energy_backend)
            em.__enter__()
            try:
                mape_info = mape_store.load()  # inside the lock (audit D4)

                # ── Monitor ──────────────────────────────────────────────
                telemetry: dict = {
                    "r2": None, "avg_energy_uJ": None,
                    "ema_score": mape_info["ema_scores"].get(current_model),
                }
                violation = None
                if monitor_rows and monitor_model:
                    energies = [float(r["energy_uJ"]) for r in monitor_rows]
                    if is_cv:
                        acc = sum(float(r["proxy_acc"]) for r in monitor_rows) / len(monitor_rows)
                        telemetry = _monitor_batch([], [], energies, monitor_model,
                                                   mape_info, thresholds, accuracy=acc)
                    else:
                        telemetry = _monitor_batch(
                            [float(r["y_true"]) for r in monitor_rows],
                            [float(r["y_pred"]) for r in monitor_rows],
                            energies, monitor_model, mape_info, thresholds,
                        )
                    # ── Analyse (before the boundary update, as single-threaded)
                    if monitor_model == current_model:
                        violation = _analyse_violation(telemetry, mape_info, thresholds)
                    _update_energy_boundary(mape_info, telemetry, thresholds)
                    mape_info["last_observed_step"][monitor_model] = current_step

                mape_info["steps_since_last_switch"] = mape_info.get("steps_since_last_switch", 0) + 1

                # ── Plan ─────────────────────────────────────────────────
                decision = _plan(
                    violation, drift_result, current_model, available_models,
                    mape_info, thresholds, planner, current_step=current_step,
                )
                decision = _head_start(decision, is_cv, thresholds, available_models,
                                       current_model, mape_info)

                if violation is not None and decision.action == "noop":
                    mape_info["event_counters"]["noop_on_violation"] = (
                        mape_info["event_counters"].get("noop_on_violation", 0) + 1)

                decision_row = {
                    "step": current_step,
                    "violation": violation,
                    "drift_detected": drift_result.get("drift_detected", False),
                    "decision_action": decision.action,
                    "decision_model": decision.model,
                    "decision_reason": decision.reason,
                    "current_model": current_model,
                    "monitor_model": monitor_model,
                    **{f"ema_score_{m}": round(mape_info["ema_scores"].get(m, 0.5), 6) for m in available_models},
                    **{f"ema_acc_{m}": round(mape_info["ema_accuracy"].get(m, 0.5), 6) for m in available_models},
                    **{f"ema_eng_{m}": round(mape_info["ema_energy"].get(m, 0.5), 6) for m in available_models},
                }

                # Persist the monitor/analyse fields BEFORE executing: execute
                # does its own counter updates through mape_store, and this
                # must not overwrite them (only fields this cycle owns).
                noop_on_violation = mape_info["event_counters"].get("noop_on_violation", 0)
                owned = {k: mape_info[k] for k in _MONITOR_FIELDS}

                def _merge(d: dict) -> None:
                    d.update(owned)
                    d["event_counters"]["noop_on_violation"] = noop_on_violation
                mape_store.update(_merge)

                # ── Execute ──────────────────────────────────────────────
                new_model = execute_decision(
                    decision,
                    knowledge_dir=knowledge_dir,
                    local_dataset_config=local_dataset_config,
                    is_cv=is_cv, cv_task=cv_task,
                    current_model=current_model,
                    current_step=current_step,
                    mape_store=mape_store,
                    vmr=vmr, drift_result=drift_result,
                    scaler=scaler, seq_length=seq_length,
                    value_history=value_history,
                    image_path_history=image_path_history,
                    luminance_history=luminance_history,
                )
                if new_model != current_model:
                    # Same bookkeeping as run_experiment.py after a switch.
                    def _on_switch(d: dict) -> None:
                        d["steps_since_last_switch"] = 0
                        d.setdefault("last_observed_step", {})[new_model] = current_step
                    mape_store.update(_on_switch)
            finally:
                em.__exit__(None, None, None)
            kio.add_mape_energy(mape_store, em.total_uJ)

        kio.append_row(
            kp["mape_events_file"],
            {
                "step": current_step,
                "violation": violation,
                "drift_detected": drift_result.get("drift_detected", False),
                "kl_div": drift_result.get("kl_div"),
                "decision_action": decision.action,
                "decision_model": decision.model,
                "decision_reason": decision.reason,
                "model_before": current_model,
                "model_after": new_model,
                "r2": telemetry.get("r2"),
                "ema_score": telemetry.get("ema_score"),
                "avg_energy_uJ": telemetry.get("avg_energy_uJ"),
                "energy_threshold": round(mape_info.get("current_energy_threshold", 0.6), 6),
            },
            kio.MAPE_EVENT_FIELDS,
        )
        kio.append_dict_row(kp["planner_decisions_file"], decision_row)

        # Exact same line/format as experiments/run_experiment.py's per-cycle
        # INFO line; r2/energy may be None on a cycle with no monitor rows.
        _action_str = decision.action
        if decision.action == "switch" and decision.model and decision.model != current_model:
            _action_str = f"switch→{decision.model}"
        _flags = ""
        if violation:
            _flags += f"  violation={violation}"
        if drift_result.get("drift_detected"):
            _flags += "  drift=yes"
        if monitor_model and monitor_model != current_model:
            _flags += f"  (batch served by {monitor_model})"
        _metric_label = "conf" if is_cv else "R²  "
        _r2_str = "n/a" if telemetry.get("r2") is None else f"{telemetry['r2']:.4f}"
        _energy_str = "n/a" if telemetry.get("avg_energy_uJ") is None else f"{telemetry['avg_energy_uJ']:.1f}"
        _ema = telemetry.get("ema_score")
        _ema_str = "n/a" if _ema is None else f"{_ema:.4f}"
        cycles = mape_store.update(lambda d: d.__setitem__("cycles", d.get("cycles", 0) + 1))["cycles"]
        logger.info(
            "MAPE[%d] model=%-8s  action=%-16s  %s=%s  energy=%s µJ/step  ema=%s%s",
            cycles, current_model, _action_str, _metric_label,
            _r2_str, _energy_str, _ema_str, _flags,
        )

    return new_model

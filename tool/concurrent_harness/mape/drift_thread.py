"""concurrent_harness/mape/drift_thread.py — t2: drift DETECTION only.

Mirrors the role of HarmonE/mape/manage.py's run_execute_drift() thread (a
periodic KL check of the recent data window), but — as of 2026-09-28 (audit
D4/D8) — t2 no longer plans or executes anything. It publishes its latest
drift result (including the VMR lookup, via the shared _analyse_drift()) to
knowledge_dir/drift_state.json, and t1 (plan_thread.py) folds that result into
its own single Monitor->Analyse->Plan->Execute cycle, exactly the way
experiments/run_experiment.py passes violation AND drift_result into one
_plan() call. Consequences:
  - drift-aware planners see drift and score/energy violations in the same
    decision, with single-threaded's precedence rules (D8);
  - only one thread ever executes decisions, so there are no competing
    cycles, no double-trigger races, and no stale mape_info write-backs (D4);
  - "cycles" counts t1 planning cycles only (one per batch), comparable with
    single-threaded's mape_cycles; t2's work is counted in drift_checks.

t2 now runs for EVERY planner (previously only drift-aware ones; the others
had t1 compute drift itself), so drift semantics are identical across
planners.

Drift semantics come from mape/drift_ref.py (audit E1/E2): overflow bins, a
per-model reference re-based after each replace/retrain, and a row-count
cooldown after adaptation (the original's `time.sleep(400)` after acting).

Triggering is row-count-based (every monitor_interval rows), matching t1.
Its detection work is metered under the same locks as a MAPE cycle and
accumulated into mape_k_energy_uJ (single-threaded meters drift analysis as
part of its MAPE cycle too).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core.energy import EnergyMeter
from experiments.run_experiment import _analyse_drift, _subsample_raw

import knowledge_io as kio
from drift_ref import PerModelDriftReference, BoundDetector

logger = logging.getLogger(__name__)

# Purely a responsiveness tick (how often to check for a full batch / for
# shutdown) — NOT a decision-cadence parameter.
_POLL_TICK_S = 0.02


def run(
    *,
    dataset_config: dict,
    knowledge_dir: Path,
    is_cv: bool,
    vmr,
    mape_store: kio.MapeInfoStore,
    drift_window: int,
    energy_backend: str,
) -> None:
    kp = kio.knowledge_paths(knowledge_dir)
    available_models = list(dataset_config.get("models", {}).keys())
    thresholds = dataset_config
    tau_drift = float(dataset_config.get("tau_drift", 0.5))
    monitor_interval = int(dataset_config.get("monitor_interval", 50))
    cooldown_rows = int(dataset_config.get("drift_cooldown_rows", drift_window))

    ref = PerModelDriftReference(
        reference_path=str(kp["reference_dist_file"]),
        tau_drift=tau_drift, window_size=drift_window, cooldown_rows=cooldown_rows,
    )

    pred_buf = kio.RowBuffer(kp["predictions_file"])
    if is_cv:
        lum_buf = kio.RowBuffer(knowledge_dir / "luminance_history.csv")

    # Trailing window, capped to drift_window, extended by exactly
    # monitor_interval values per batch (see RowBuffer's docstring).
    window: list[float] = []

    running = True
    while running:
        running = kio.interruptible_sleep(_POLL_TICK_S, kp["shutdown"])
        pred_buf.poll()
        if is_cv:
            lum_buf.poll()

        while True:
            batch = pred_buf.drain_chunk(monitor_interval)
            if batch is None:
                break
            try:
                current_step = pred_buf.committed_count
                if is_cv:
                    lum_batch = lum_buf.drain_chunk(monitor_interval) or []
                    window.extend(float(r["luminance"]) for r in lum_batch)
                else:
                    window.extend(float(r["y_true"]) for r in batch)
                if len(window) > drift_window:
                    window = window[-drift_window:]

                current_model = kio.read_current_model(kp["model_file"], default=available_models[0])
                last_adapt = mape_store.load().get("last_adaptation", {})
                if current_model in last_adapt:
                    if ref.note_adaptation(current_model, int(last_adapt[current_model]), window, current_step):
                        logger.info(
                            "drift_thread: re-based drift reference for %s after adaptation at step %s "
                            "(cooldown %d rows)", current_model, last_adapt[current_model], cooldown_rows,
                        )

                with kio.ENERGY_LOCK:
                    with kio.energy_lock(knowledge_dir, blocking=True):
                        em = EnergyMeter("drift_check", backend=energy_backend)
                        em.__enter__()
                        try:
                            current_dist = None
                            if len(window) >= drift_window:
                                hist, _ = np.histogram(window, bins=ref.base_edges)
                                current_dist = {
                                    "type": "histogram",
                                    "data": hist.tolist(),
                                    "raw": _subsample_raw(window),
                                }
                            drift_result = _analyse_drift(
                                window, BoundDetector(ref, current_model, current_step), thresholds,
                                vmr=vmr, current_model=current_model,
                                current_distribution=current_dist,
                            )
                        finally:
                            em.__exit__(None, None, None)
                        kio.add_mape_energy(mape_store, em.total_uJ)

                kio.write_drift_state(kp["drift_state"], {
                    "step": current_step,
                    "model": current_model,
                    # Latest adaptation of this model t2 had already accounted
                    # for (re-based + cooldown) when producing this result —
                    # t1 ignores a result computed before it knew about the
                    # model's latest adaptation (see plan_thread.py).
                    "adaptation_seen": ref._seen_adaptation.get(current_model, -1),
                    "drift_result": drift_result,
                })
                mape_store.update(lambda d: d.__setitem__("drift_checks", d.get("drift_checks", 0) + 1))
            except Exception:
                logger.exception(
                    "drift_thread: drift check at committed_count=%d failed — "
                    "skipping this batch; the next batch is unaffected.",
                    pred_buf.committed_count,
                )

    logger.info("drift_thread: shutdown")

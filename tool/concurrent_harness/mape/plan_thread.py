"""concurrent_harness/mape/plan_thread.py — t1: the one MAPE-K decision loop.

Mirrors HarmonE/mape/manage.py's run_execute_mape() thread. Because this
repo's core/planners/* unify what the original splits across three mechanisms
(ε-greedy switch, drift-triggered replace/retrain, blind periodic retrain)
into ONE Planner.plan(ctx) call, this single thread drives every planner.

As of 2026-09-28 it is also the ONLY thread that plans or executes (audit
D4/D8): t2 (drift_thread.py) just detects drift and publishes the result;
t1 folds it into the same cycle as the score/energy analysis — exactly like
experiments/run_experiment.py's single cycle. See mape_cycle.py.

Per batch (row-count triggered, one cycle per monitor_interval rows, same as
single-threaded's `step % monitor_interval == 0`):
  1. Monitor rows are credited to the model that actually SERVED the batch
     (read from the rows' active_model). inference.py only changes models at
     batch boundaries, so every batch has exactly one serving model (audit
     D1: previously 267/466 batches mixed models and the own-row filter
     scored as few as 1-10 rows).
  2. Wait (briefly) for t2's drift result for this same batch and use it only
     if it is for the commanded model and t2 had already accounted for that
     model's latest adaptation.
  3. Run the cycle (mape_cycle.run_cycle).
  4. Publish the processed-row count (kio.PROCESSED_FILE) — this advances
     the inference/manager barrier (see inference.py). Always written, even
     if the cycle failed, so inference can never deadlock on a bad batch.
"""

from __future__ import annotations

import collections
import logging
import time
from pathlib import Path

from core.planners.base import get_planner

import knowledge_io as kio
import mape_cycle

logger = logging.getLogger(__name__)

_NO_DRIFT = {"drift_detected": False, "action": None, "version": None, "kl_div": None}

# Purely a responsiveness tick (how often to check for a full batch / for
# shutdown) — NOT a decision-cadence parameter.
_POLL_TICK_S = 0.02
# Max wait for t2's drift result for the current batch. t2's per-batch work is
# a histogram + KL (+ a VMR lookup when drift fires), normally milliseconds.
_DRIFT_WAIT_S = 10.0


def _drift_for_batch(kp: dict, current_step: int, current_model: str,
                     last_adaptation: dict) -> dict:
    deadline = time.monotonic() + _DRIFT_WAIT_S
    state = kio.read_drift_state(kp["drift_state"])
    while (state is None or state.get("step", -1) < current_step) and time.monotonic() < deadline:
        time.sleep(0.005)
        state = kio.read_drift_state(kp["drift_state"])
    if state is None or state.get("step", -1) < current_step:
        logger.warning("plan_thread: no drift result from t2 for step %d — assuming no drift.", current_step)
        return dict(_NO_DRIFT)
    if state.get("model") != current_model:
        return dict(_NO_DRIFT, kl_div=state["drift_result"].get("kl_div"))
    if state.get("adaptation_seen", -1) < int(last_adaptation.get(current_model, -1)):
        # t2 computed this before re-basing for the model's latest adaptation.
        return dict(_NO_DRIFT, kl_div=state["drift_result"].get("kl_div"))
    return state["drift_result"]


def run(
    *,
    dataset_config: dict,
    local_dataset_config: dict,
    knowledge_dir: Path,
    planner_name: str,
    pin_model: str | None,
    is_cv: bool,
    cv_task: str,
    scaler,
    vmr,
    mape_store: kio.MapeInfoStore,
    drift_window: int,
    energy_backend: str,
) -> None:
    kp = kio.knowledge_paths(knowledge_dir)
    planner = get_planner(planner_name)
    all_models = list(dataset_config.get("models", {}).keys())
    available_models = [pin_model] if pin_model else all_models
    thresholds = dataset_config
    seq_length = int(dataset_config.get("seq_length", 1))
    monitor_interval = int(dataset_config.get("monitor_interval", 50))

    pred_buf = kio.RowBuffer(kp["predictions_file"])
    if is_cv:
        img_buf = kio.RowBuffer(kp["image_history_file"])
        lum_buf = kio.RowBuffer(knowledge_dir / "luminance_history.csv")

    # Windows used by retrain/fine-tune (capped to drift_window).
    value_history: list[float] = []
    luminance_history: list[float] = []
    image_path_history: list[str] = []

    running = True
    while running:
        running = kio.interruptible_sleep(_POLL_TICK_S, kp["shutdown"])
        pred_buf.poll()
        if is_cv:
            img_buf.poll()
            lum_buf.poll()

        while True:
            batch = pred_buf.drain_chunk(monitor_interval)
            if batch is None:
                break
            current_step = pred_buf.committed_count
            try:
                current_model = kio.read_current_model(kp["model_file"], default=available_models[0])

                served = collections.Counter(r.get("active_model") for r in batch)
                monitor_model = served.most_common(1)[0][0]
                if len(served) > 1:
                    logger.warning("plan_thread: batch ending at %d mixed models %s — "
                                   "scoring only %s's rows.", current_step, dict(served), monitor_model)
                monitor_rows = [r for r in batch if r.get("active_model") == monitor_model]

                if is_cv:
                    image_path_history.extend(r["image_path"] for r in (img_buf.drain_chunk(monitor_interval) or []))
                    luminance_history.extend(float(r["luminance"]) for r in (lum_buf.drain_chunk(monitor_interval) or []))
                    image_path_history = image_path_history[-drift_window:]
                    luminance_history = luminance_history[-drift_window:]
                else:
                    # Retrain window is the raw stream (all rows), like
                    # single-threaded's value_history.
                    value_history.extend(float(r["y_true"]) for r in batch)
                    value_history = value_history[-drift_window:]

                last_adaptation = mape_store.load().get("last_adaptation", {})
                drift_result = _drift_for_batch(kp, current_step, current_model, last_adaptation)

                mape_cycle.run_cycle(
                    planner=planner,
                    current_model=current_model,
                    monitor_model=monitor_model,
                    available_models=available_models,
                    mape_store=mape_store,
                    thresholds=thresholds,
                    drift_result=drift_result,
                    current_step=current_step,
                    monitor_rows=monitor_rows,
                    is_cv=is_cv,
                    knowledge_dir=knowledge_dir,
                    local_dataset_config=local_dataset_config,
                    cv_task=cv_task,
                    scaler=scaler,
                    seq_length=seq_length,
                    vmr=vmr,
                    value_history=(value_history if not is_cv else None),
                    image_path_history=(image_path_history if is_cv else None),
                    luminance_history=(luminance_history if is_cv else None),
                    energy_backend=energy_backend,
                )
            except Exception:
                # A failed cycle must not kill the thread (torn rows can no
                # longer be read, see knowledge_io.RowBuffer.poll).
                logger.exception(
                    "plan_thread: cycle at committed_count=%d failed — skipping this cycle; "
                    "the next batch is unaffected.", current_step,
                )
            finally:
                kio.write_processed(kp["processed"], current_step)

    logger.info("plan_thread: shutdown")

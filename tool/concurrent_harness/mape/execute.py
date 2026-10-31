"""concurrent_harness/mape/execute.py — the "E" stage: apply one PlanDecision.

Mirrors HarmonE/mape/execute.py's role directly (E: switch/replace/retrain
dispatch). Shared by plan_thread.py (t1) and drift_thread.py (t2): whichever
thread's context produced a decision, applying it (switch / replace / retrain
/ noop) is identical file/model I/O either way. New code, reusing
experiments/run_experiment.py's VMR restore helpers read-only rather than
duplicating them.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

from core.energy import Pkg0Meter
from core.planners.base import PlanDecision
from core.vmr import VMR

from experiments.run_experiment import _do_vmr_restore, _do_cv_vmr_restore, note_adaptation

import knowledge_io as kio
import retrain

logger = logging.getLogger(__name__)

# ── Background regression retrains (2026-10-12, item 9a) ─────────────────────
# As in the original (manage.py runs retrain.py via os.system from its own
# thread while inference keeps streaming), a regression retrain runs in a
# background thread: the plan cycle returns at once, so monitoring and the
# inference barrier keep advancing, and the retrain's package energy lands in
# the concurrent prediction readings. The new weights are written atomically
# and picked up by inference on its next prediction (per-prediction reload).
# One retrain at a time: while one is in flight, further retrain/replace
# decisions are skipped (counted as adaptation_skipped_busy) — the original's
# drift thread is likewise blocked during os.system(retrain) (then sleeps).
# The thread only produces results; the plan thread applies the bookkeeping
# (counters, last_adaptation, accuracy baseline) at its next cycle, so it
# cannot race the cycle's own mape_info read-modify-write.
_bg_thread: threading.Thread | None = None
_bg_results: "queue.Queue[tuple]" = queue.Queue()


def background_busy() -> bool:
    return _bg_thread is not None and _bg_thread.is_alive()


def apply_background_results(mape_store: kio.MapeInfoStore, current_step: int,
                             knowledge_dir: Path | None = None) -> None:
    """Apply finished background retrains (call from the plan thread only)."""
    while True:
        try:
            model, ok, archived, reason, energy_uJ, *rest = _bg_results.get_nowait()
        except queue.Empty:
            return
        if ok and rest and knowledge_dir is not None:
            kio.write_drift_ref_window(knowledge_dir, model, rest[0])  # before last_adaptation (M1)
        key = "retrains" if ok else "retrain_skipped"
        mape_store.update(lambda d: d["event_counters"].__setitem__(
            key, d["event_counters"][key] + 1))
        if energy_uJ is not None:
            mape_store.update(lambda d: d["event_counters"].__setitem__(
                "retrain_energy_uJ", d["event_counters"].get("retrain_energy_uJ", 0.0) + energy_uJ))
        if ok:
            def _rec(d: dict) -> None:
                d.setdefault("last_adaptation", {})[model] = current_step
                note_adaptation(d, model, archived)
            mape_store.update(_rec)
        logger.info("retrain %s (background, applied at step %d): %s (%s)",
                    "ok" if ok else "skipped", current_step, model, reason)


def finish_background(mape_store: kio.MapeInfoStore, current_step: int,
                      knowledge_dir: Path | None = None, timeout_s: float = 600.0) -> None:
    """Wait for an in-flight background retrain and apply its result (shutdown)."""
    if _bg_thread is not None:
        _bg_thread.join(timeout_s)
        if _bg_thread.is_alive():
            logger.warning("background retrain still running after %.0f s at shutdown — not counted.", timeout_s)
    apply_background_results(mape_store, current_step, knowledge_dir)


def execute_decision(
    decision: PlanDecision,
    *,
    knowledge_dir: Path,
    local_dataset_config: dict,
    is_cv: bool,
    cv_task: str,
    current_model: str,
    current_step: int,
    mape_store: kio.MapeInfoStore,
    vmr: VMR,
    drift_result: dict,
    scaler=None,
    seq_length: int = 0,
    value_history: list[float] | None = None,
    image_path_history: list[str] | None = None,
    luminance_history: list[float] | None = None,
) -> str:
    """Apply decision; update mape_info counters; return the (possibly new)
    active model name.

    A successful replace/retrain also saves the window it adapted to and records
    mape_info["last_adaptation"][current_model] = current_step, which t2
    (drift_thread.py) uses to re-base that model's drift reference and start
    the cooldown (audit E1 — see mape/drift_ref.py)."""
    global _bg_thread
    kp = kio.knowledge_paths(knowledge_dir)
    new_model = current_model

    if decision.action in ("retrain", "replace") and background_busy():
        mape_store.update(lambda d: d["event_counters"].__setitem__(
            "adaptation_skipped_busy", d["event_counters"].get("adaptation_skipped_busy", 0) + 1))
        logger.info("%s skipped: a background retrain is still running (%s)", decision.action, decision.reason)
        return current_model

    if decision.action == "retrain" and (drift_result or {}).get("single_version"):
        mape_store.update(lambda d: d["event_counters"].__setitem__(
            "vmr_single_version_retrain", d["event_counters"].get("vmr_single_version_retrain", 0) + 1))

    # The window this decision adapts to (2026-10-31, M1): for a retrain, what it
    # trains on; for a restore, what the version was matched against.
    drift_window = int(local_dataset_config.get("drift_window_size", 1200))
    adapted_window = list((luminance_history if is_cv else value_history) or [])[-drift_window:]

    def _record_adaptation(d: dict, version_path: str | None = None) -> None:
        d.setdefault("last_adaptation", {})[current_model] = current_step
        note_adaptation(d, current_model, version_path)

    if decision.action == "switch":
        if decision.model and decision.model != current_model:
            kio.write_current_model(kp["model_file"], decision.model)
            mape_store.update(lambda d: d["event_counters"].__setitem__(
                "model_switches", d["event_counters"]["model_switches"] + 1))
            logger.info("switch: %s -> %s (%s)", current_model, decision.model, decision.reason)
            new_model = decision.model
        else:
            mape_store.update(lambda d: d["event_counters"].__setitem__(
                "noops", d["event_counters"]["noops"] + 1))

    elif (decision.action == "replace" and decision.version_path
          and decision.version_path == mape_store.load().get("serving_version", {}).get(current_model)):
        # No-op restore (2026-10-07): the matched version is the one already
        # serving (e.g. the initial weights of a never-retrained model) —
        # reloading it changes nothing. Only re-base drift and the accuracy
        # baseline, as a restore would.
        mape_store.update(lambda d: d["event_counters"].__setitem__(
            "vmr_noop", d["event_counters"].get("vmr_noop", 0) + 1))
        kio.write_drift_ref_window(knowledge_dir, current_model, adapted_window)
        mape_store.update(_record_adaptation)
        logger.info("replace skipped (already serving): %s <- %s", current_model, decision.version_path)

    elif decision.action == "replace" and decision.version_path:
        models: dict = {}
        if is_cv:
            from experiments.run_experiment import _load_cv_model_store
            model_store = _load_cv_model_store(local_dataset_config, cv_task, only=[current_model])
            ok = _do_cv_vmr_restore(decision.version_path, current_model, models, model_store)
        else:
            model_store: dict = {}
            ok = _do_vmr_restore(decision.version_path, current_model, models, model_store)
        if ok:
            info = model_store[current_model]
            local_path = kio.local_weights_path(
                knowledge_dir, current_model,
                local_dataset_config["models"][current_model]["weights_path"],
            )
            retrain._persist_local(info, local_path)
            kio.write_reload_flag(knowledge_dir, current_model)
            mape_store.update(lambda d: d["event_counters"].__setitem__(
                "vmr_events", d["event_counters"]["vmr_events"] + 1))
            kio.write_drift_ref_window(knowledge_dir, current_model, adapted_window)
            mape_store.update(lambda d: _record_adaptation(d, decision.version_path))
            logger.info("replace: %s <- %s", current_model, decision.version_path)
        else:
            mape_store.update(lambda d: d["event_counters"].__setitem__(
                "retrain_skipped", d["event_counters"]["retrain_skipped"] + 1))

    elif decision.action == "retrain":
        mape_info = mape_store.load()
        proxy_score = mape_info["ema_scores"].get(current_model)
        if is_cv:
            ok, archived = retrain.do_cv_retrain(
                local_dataset_config, knowledge_dir, current_model, cv_task,
                image_path_history or [], luminance_history or [],
                local_dataset_config, vmr, drift_result, proxy_score,
            )
        elif local_dataset_config.get("background_retrain", False):
            history = list(value_history or [])     # snapshot: the plan thread keeps appending
            drift_snapshot = dict(drift_result or {})

            def _job(model=current_model, reason=decision.reason) -> None:
                # Metered like the original's periodic_retrain (pyRAPL pkg[0]
                # around retrain.py), without the MAPE energy lock so monitoring
                # is not blocked; added to the reported total (2026-10-13).
                meter = Pkg0Meter()
                meter.begin()
                try:
                    ok_, archived_ = retrain.do_regression_retrain(
                        local_dataset_config, knowledge_dir, model, scaler,
                        history, seq_length, vmr, drift_snapshot, proxy_score,
                    )
                except Exception:
                    logger.exception("background retrain of %s failed", model)
                    ok_, archived_ = False, None
                _bg_results.put((model, ok_, archived_, reason, meter.end(), adapted_window))

            _bg_thread = threading.Thread(target=_job, daemon=True, name=f"retrain-{current_model}")
            _bg_thread.start()
            logger.info("retrain started in background: %s (%s)", current_model, decision.reason)
            if local_dataset_config.get("smoke_wait_retrain"):
                # Smoke check: without pacing the stream would run thousands of
                # rows past a retrain; waiting keeps the new weights' arrival
                # (in rows) close to a paced run. Applied next cycle as usual.
                _bg_thread.join()
            return new_model
        else:
            ok, archived = retrain.do_regression_retrain(
                local_dataset_config, knowledge_dir, current_model, scaler,
                value_history or [], seq_length, vmr, drift_result, proxy_score,
            )
        key = "retrains" if ok else "retrain_skipped"
        mape_store.update(lambda d: d["event_counters"].__setitem__(
            key, d["event_counters"][key] + 1))
        if ok:
            kio.write_drift_ref_window(knowledge_dir, current_model, adapted_window)
            mape_store.update(lambda d: _record_adaptation(d, archived))
        logger.info("retrain %s: %s (%s)", "ok" if ok else "skipped", current_model, decision.reason)

    else:
        mape_store.update(lambda d: d["event_counters"].__setitem__(
            "noops", d["event_counters"]["noops"] + 1))

    return new_model

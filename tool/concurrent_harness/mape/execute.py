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
from pathlib import Path

from core.planners.base import PlanDecision
from core.vmr import VMR

from experiments.run_experiment import _do_vmr_restore, _do_cv_vmr_restore

import knowledge_io as kio
import retrain

logger = logging.getLogger(__name__)


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

    A successful replace/retrain also records
    mape_info["last_adaptation"][current_model] = current_step, which t2
    (drift_thread.py) uses to re-base that model's drift reference and start
    its cooldown (audit E1 — see mape/drift_ref.py)."""
    kp = kio.knowledge_paths(knowledge_dir)
    new_model = current_model

    def _record_adaptation(d: dict) -> None:
        d.setdefault("last_adaptation", {})[current_model] = current_step

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

    elif decision.action == "replace" and decision.version_path:
        models: dict = {}
        if is_cv:
            from experiments.run_experiment import _load_cv_model_store
            model_store = _load_cv_model_store(local_dataset_config, cv_task)
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
            mape_store.update(_record_adaptation)
            logger.info("replace: %s <- %s", current_model, decision.version_path)
        else:
            mape_store.update(lambda d: d["event_counters"].__setitem__(
                "retrain_skipped", d["event_counters"]["retrain_skipped"] + 1))

    elif decision.action == "retrain":
        mape_info = mape_store.load()
        proxy_score = mape_info["ema_scores"].get(current_model)
        if is_cv:
            ok = retrain.do_cv_retrain(
                local_dataset_config, knowledge_dir, current_model, cv_task,
                image_path_history or [], luminance_history or [],
                local_dataset_config, vmr, drift_result, proxy_score,
            )
        else:
            ok = retrain.do_regression_retrain(
                local_dataset_config, knowledge_dir, current_model, scaler,
                value_history or [], seq_length, vmr, drift_result, proxy_score,
            )
        key = "retrains" if ok else "retrain_skipped"
        mape_store.update(lambda d: d["event_counters"].__setitem__(
            key, d["event_counters"][key] + 1))
        if ok:
            mape_store.update(_record_adaptation)
        logger.info("retrain %s: %s (%s)", "ok" if ok else "skipped", current_model, decision.reason)

    else:
        mape_store.update(lambda d: d["event_counters"].__setitem__(
            "noops", d["event_counters"]["noops"] + 1))

    return new_model

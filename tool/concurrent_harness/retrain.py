"""concurrent_harness/retrain.py — retrain/fine-tune routine, reused from the
single-threaded harness's own reference implementation.

Mirrors HarmonE/retrain.py's role directly (same filename, same role: the
standalone retraining routine invoked by the managing system on a decision) —
though here it's called as an in-process function from mape/execute.py rather
than shelled out to as a separate script, since it's new code with no
existing subprocess-boundary constraint to preserve.

experiments/run_experiment.py's inline-retrain helpers (_do_inline_retrain,
_do_cv_inline_finetune, _load_model_store, _load_cv_model_store,
_archive_in_vmr) are plain module-level functions that take all their state as
explicit arguments — nothing tied to run_experiment()'s own loop. Importing
them here read-only means retraining behaves *identically* to the
single-threaded path (same hyperparams.retrain handling, same 5-epoch LSTM
fine-tune, same pseudo-label/safety-valve CV logic) instead of a second,
independently-drifting reimplementation. This is a deliberate coupling,
documented in the approved plan — no file under experiments/ is modified.

Retrained/fine-tuned weights are never written back to the shared
configs/datasets/*.json weights_path — only to this run's own
knowledge_dir/models/ copy (see knowledge_io.localize_dataset_config). VMR
archives go to the SHARED tool/knowledge/vmr/<dataset>/<planner>/ tree, the
same one the single-threaded harness uses (see mape/manage.py).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from experiments.run_experiment import (
    _load_model_store,
    _load_cv_model_store,
    _do_inline_retrain,
    _do_cv_inline_finetune,
    _archive_in_vmr,
    _DRIFT_WINDOW_SIZE,
)
from core.vmr import VMR

import knowledge_io as kio

logger = logging.getLogger(__name__)


def _persist_local(info: dict, local_path: Path) -> None:
    """Save a retrained/fine-tuned model object to its per-run local copy."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    t = info["type"]
    if t in ("lstm", "torchvision_classifier", "segformer_segmentation"):
        import torch
        torch.save(info["model"].state_dict(), str(local_path))
    elif t == "yolo_detection":
        info["model"].save(str(local_path))
    else:
        with open(local_path, "wb") as f:
            pickle.dump(info["model"], f)


def do_regression_retrain(
    local_dataset_config: dict,
    knowledge_dir: Path,
    model_name: str,
    scaler,
    value_history: list[float],
    seq_length: int,
    vmr: VMR,
    drift_result: dict,
    proxy_score: float | None,
    drift_window: int = _DRIFT_WINDOW_SIZE,
) -> bool:
    """Retrain one regression model on the recent window; persist locally + VMR.

    local_dataset_config must already have weights_path entries localized to
    this run's knowledge_dir/models/ (see knowledge_io.localize_dataset_config)
    — _load_model_store reads spec["weights_path"] directly.
    """
    model_store = _load_model_store(local_dataset_config)
    if model_store.get(model_name) is None:
        logger.warning("retrain: no loadable model_store entry for '%s'", model_name)
        return False

    run_path = knowledge_dir  # _archive_in_vmr only uses this for a scratch tmp file
    _archive_in_vmr(
        model_name, model_store, vmr, value_history, drift_result, run_path,
        drift_window=drift_window, tag="pre_retrain", proxy_score=proxy_score,
    )

    models: dict = {}
    ok = _do_inline_retrain(
        model_name, model_store, models, value_history,
        scaler=scaler, seq_length=seq_length, drift_window=drift_window,
    )
    if not ok:
        return False

    info = model_store[model_name]
    local_path = kio.local_weights_path(
        knowledge_dir, model_name, local_dataset_config["models"][model_name]["weights_path"]
    )
    _persist_local(info, local_path)

    _archive_in_vmr(
        model_name, model_store, vmr, value_history, drift_result, run_path,
        drift_window=drift_window, tag="retrain", proxy_score=proxy_score,
    )
    kio.write_reload_flag(knowledge_dir, model_name)
    return True


def do_cv_retrain(
    local_dataset_config: dict,
    knowledge_dir: Path,
    model_name: str,
    cv_task: str,
    image_path_history: list[str],
    luminance_history: list[float],
    thresholds: dict,
    vmr: VMR,
    drift_result: dict,
    proxy_score: float | None,
    drift_window: int = _DRIFT_WINDOW_SIZE,
) -> bool:
    """Fine-tune one CV model on the recent window; persist locally + VMR."""
    model_store = _load_cv_model_store(local_dataset_config, cv_task)
    if model_store.get(model_name) is None:
        logger.warning("cv retrain: no loadable model_store entry for '%s'", model_name)
        return False

    _archive_in_vmr(
        model_name, model_store, vmr, luminance_history, drift_result, knowledge_dir,
        drift_window=drift_window, tag="pre_retrain", proxy_score=proxy_score,
    )

    models: dict = {}
    ok = _do_cv_inline_finetune(
        model_name, model_store, models, image_path_history,
        thresholds=thresholds, drift_window=drift_window, run_path=knowledge_dir,
    )
    if not ok:
        return False

    info = model_store[model_name]
    local_path = kio.local_weights_path(
        knowledge_dir, model_name, local_dataset_config["models"][model_name]["weights_path"]
    )
    _persist_local(info, local_path)

    _archive_in_vmr(
        model_name, model_store, vmr, luminance_history, drift_result, knowledge_dir,
        drift_window=drift_window, tag="retrain", proxy_score=proxy_score,
    )
    kio.write_reload_flag(knowledge_dir, model_name)
    return True

"""concurrent_harness/mape/manage.py — Process B: the MAPE-K management process.

Mirrors HarmonE/mape/manage.py's role — starts the management threads and
blocks until shutdown — with shutdown coordinated via a sentinel file instead
of a human killing the process.

Threads (2026-09-28 redesign, audit D4/D8 — both start for EVERY planner):
  - t1 (plan_thread.py): the only thread that plans and executes. One
    Monitor->Analyse->Plan->Execute cycle per monitor_interval batch, with
    violation AND drift passed into the same _plan() call, exactly like
    experiments/run_experiment.py. Also drives PRT (naive_prt /
    random_switch_prt check current_step % prt_interval inside plan()).
  - t2 (drift_thread.py): drift detection only (KL + VMR lookup), published to
    drift_state.json for t1. Mirrors the original's separate drift thread in
    role, but never acts on its own.

Why not the original's literal three threads (switch / drift / blind PRT):
this repo's core/planners/* already unify all three mechanisms into one
Planner.plan(ctx) call, and a second executing thread was a source of real
bugs (double-triggered adaptations, stale mape_info write-backs). With a single
executor, core/vmr.py (no file locking) and core/planners/bandit.py's
module-level singleton are only ever used for decisions from one thread.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import threading
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent      # concurrent_harness/mape/
_HARNESS_DIR = _THIS_DIR.parent                    # concurrent_harness/
_TOOL_DIR = _HARNESS_DIR.parent                     # tool/
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))
sys.path.insert(0, str(_HARNESS_DIR))  # knowledge_io.py, retrain.py (harness root)
sys.path.insert(0, str(_THIS_DIR))      # plan_thread, drift_thread, mape_cycle, execute (mape/ siblings)

import knowledge_io as kio
import plan_thread
import drift_thread

from experiments.run_experiment import (
    _load_dataset_config,
    _build_adapter,
    _fit_scaler,
    _build_reference_distribution,
    _build_cv_reference,
    _seed_vmr_initial,
    _subsample_raw,
    _DRIFT_WINDOW_SIZE,
    _DRIFT_N_BINS,
)
from core.vmr import VMR

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--configs-dir", default=str(_TOOL_DIR / "configs"))
    p.add_argument("--knowledge-dir", required=True)
    p.add_argument("--planner", required=True)
    p.add_argument("--pin-model", default=None)
    p.add_argument("--seed", type=int, default=42,
                    help="Seeds random/numpy exactly like run_experiment.py (audit D7).")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    kio.setup_logging(args.verbose)
    random.seed(args.seed)
    np.random.seed(args.seed)

    configs_dir = Path(args.configs_dir)
    knowledge_dir = Path(args.knowledge_dir)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    kp = kio.knowledge_paths(knowledge_dir)

    dataset_config = _load_dataset_config(args.dataset, configs_dir)
    is_cv = dataset_config.get("domain", "regression") == "cv"
    cv_task = dataset_config.get("task", "classification")
    if is_cv:
        from core.device import configure_device
        configure_device(dataset_config)  # CV fine-tune / VMR restore on the GPU (core/device.py)
    local_dataset_config = kio.localize_dataset_config(dataset_config, knowledge_dir)
    model_names = list(dataset_config.get("models", {}).keys())
    energy_backend = dataset_config.get("energy_meter", "auto")

    drift_window = int(dataset_config.get("drift_window_size", _DRIFT_WINDOW_SIZE))
    n_bins = _DRIFT_N_BINS

    # Independently fit scaler / build the drift reference from the SAME
    # deterministic train_split() every process uses — no cross-process file
    # dependency (mirrors the original: inference.py and retrain.py each fit
    # their own scaler locally too).
    adapter = _build_adapter(dataset_config, configs_dir)
    scaler = None
    train_values = None
    if is_cv:
        train_paths = adapter.train_split()
        _build_cv_reference(train_paths, str(kp["reference_dist_file"]), n_bins=n_bins)
    else:
        train_values = adapter.train_split()
        scaler = _fit_scaler(train_values)
        _build_reference_distribution(train_values, str(kp["reference_dist_file"]), n_bins=n_bins)

    # Shared VMR path — identical construction to experiments/run_experiment.py
    # (_TOOL_DIR/knowledge/vmr/<dataset>/<planner>/). Shared with the
    # single-threaded harness; never run both grids for the same
    # dataset+planner at the same moment (core/vmr.py has no file locking).
    _vmr_dir = _TOOL_DIR / "knowledge" / "vmr" / args.dataset / args.planner
    _vmr_dir.mkdir(parents=True, exist_ok=True)
    vmr = VMR(str(_vmr_dir))
    import json
    with open(kp["reference_dist_file"]) as f:
        training_distribution = {"type": "histogram", "data": json.load(f)["histogram"]}
    if train_values is not None:
        # raw training window for raw-vs-raw VMR matching (audit E2)
        training_distribution["raw"] = _subsample_raw(train_values)
    _seed_vmr_initial(model_names, dataset_config, vmr, training_distribution)

    if args.planner == "bandit":
        # Shared tool/knowledge/ path, identical to run_experiment.py. Only t1
        # ever calls the planner, so the module-level singleton is used from
        # one thread.
        from core.planners.bandit import load_or_create_bandit, set_bandit_instance
        _shared_knowledge_dir = _TOOL_DIR / "knowledge"
        _shared_knowledge_dir.mkdir(parents=True, exist_ok=True)
        bandit_obj = load_or_create_bandit(dataset_config, model_names, _shared_knowledge_dir, args.dataset)
        set_bandit_instance(bandit_obj)

    mape_store = kio.MapeInfoStore(kp["mape_info_file"], kio.build_initial_mape_info(model_names))

    # Thread-death detection: threading.Thread.join() returns normally even
    # if the target raised, so record deaths and exit non-zero (run_concurrent
    # then reports "status": "failed").
    errors: list[BaseException] = []

    def _guarded(target, name):
        def _wrapped(**kwargs):
            try:
                target(**kwargs)
            except BaseException as exc:  # noqa: BLE001 - must catch everything to detect death
                logger.exception("manage_proc: thread %s died unexpectedly", name)
                errors.append(exc)
        return _wrapped

    t1 = threading.Thread(
        target=_guarded(plan_thread.run, "t1-plan"),
        kwargs=dict(
            dataset_config=dataset_config,
            local_dataset_config=local_dataset_config,
            knowledge_dir=knowledge_dir,
            planner_name=args.planner,
            pin_model=args.pin_model,
            is_cv=is_cv, cv_task=cv_task,
            scaler=scaler, vmr=vmr, mape_store=mape_store,
            drift_window=drift_window,
            energy_backend=energy_backend,
        ),
        daemon=True, name="t1-plan",
    )
    t2 = threading.Thread(
        target=_guarded(drift_thread.run, "t2-drift"),
        kwargs=dict(
            dataset_config=dataset_config,
            knowledge_dir=knowledge_dir,
            is_cv=is_cv,
            vmr=vmr, mape_store=mape_store,
            drift_window=drift_window,
            energy_backend=energy_backend,
        ),
        daemon=True, name="t2-drift",
    )
    threads = [t1, t2]

    for t in threads:
        t.start()
    logger.info("manage_proc: started %s", [t.name for t in threads])

    for t in threads:
        t.join()

    if errors:
        logger.error("manage_proc: %d thread(s) died — exiting with failure status.", len(errors))
        sys.exit(1)
    logger.info("manage_proc: all threads exited.")


if __name__ == "__main__":
    main()

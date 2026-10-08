"""concurrent_harness/inference.py — Process A: streaming inference (the managed system).

Mirrors HarmonE/inference.py's role and shape directly (same filename, same
role): one bounded loop over the dataset's stream, reading the active model
name from a shared file, measuring its own per-step energy, appending one row
per prediction. Runs as its own OS process (spawned by run_concurrent.py),
entirely separate from mape/manage.py's process — exactly the two-process
split the original uses.

2026-09-28 (audit D1/D2/D3/C3/D9): the active model is re-read only at
monitor_interval batch boundaries, and at each boundary inference waits until
the manager is at most `max_batches_ahead` batches behind (see the barrier
comment in main()). Every step is energy-measured under a blocking
cross-process lock. CV image/luminance rows are written before the matching
prediction row.

Model loading matches experiments/run_experiment.py's own logic, not the
original's: every model in the config is loaded ONCE up front via
_load_models() (same function, imported read-only) into a {name: predict_fn}
dict — switching only changes which entry gets called, same as the
single-threaded harness. The one addition beyond that: since retrain/replace
happens in a *different process* (mape/manage.py) in this design — unlike
single-threaded, where it mutates the same in-memory dict directly — this
process still watches for a per-model reload flag and reloads just that one
model's weights from disk when one appears. That's a new problem this
process split creates, not a deviation from single-threaded loading logic.

Reuses experiments/run_experiment.py's config loading, adapter construction,
scaler-fitting, and CV raw-prediction-saving helpers (read-only import —
nothing there is modified) so this process behaves identically to the
single-threaded harness. Model weights are always loaded from this run's own
knowledge_dir/models/ copies, never from the shared configs/datasets/*.json
weights_path directly, so a concurrent run can never affect the
single-threaded harness's files.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import knowledge_io as kio

from experiments.run_experiment import (
    _load_dataset_config,
    _build_adapter,
    _fit_scaler,
    _compute_image_luminance,
    _extract_cv_proxy,
    _save_cv_prediction,
    _load_models,
)
from core.energy import EnergyMeter, Pkg0Meter
from adapters.loaders import get_loader

logger = logging.getLogger(__name__)


def _seed_local_weights(dataset_config: dict, knowledge_dir: Path) -> None:
    models_dir = knowledge_dir / kio.MODELS_SUBDIR
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in dataset_config.get("models", {}).items():
        wp = spec["weights_path"]
        src = Path(wp) if Path(wp).is_absolute() else _TOOL_DIR / wp
        dst = kio.local_weights_path(knowledge_dir, name, wp)
        if src.exists():
            shutil.copy2(src, dst)


_MAX_RELOAD_ATTEMPTS = 3
_last_reload_error: dict[str, str] = {}


def _reload_one_model(models: dict, name: str, local_config: dict) -> bool:
    """Reload a single model's weights from its (local) weights_path.

    Mirrors _load_models()'s per-entry logic exactly (same loader dispatch,
    same kwargs) so a reloaded model behaves identically to one loaded at
    startup — this only exists because retrain/replace runs in a separate
    process here; keeps the old predict_fn cached on failure rather than
    dropping the model entirely. Returns False on failure, which is logged
    (audit D6: failures used to be swallowed silently, so an adaptation could
    be counted while the old model kept serving).
    """
    spec = local_config["models"].get(name)
    if spec is None:
        return True
    try:
        loader_fn = get_loader(spec["loader"])
        models[name] = loader_fn(
            spec["weights_path"],
            seq_length=local_config.get("seq_length", 5),
            num_classes=local_config.get("num_classes", 1000),
        )
        return True
    except Exception as exc:
        logger.exception("inference_proc: reload of %r from %s failed — old weights keep serving.",
                         name, spec.get("weights_path"))
        _last_reload_error[name] = f"{type(exc).__name__}: {exc}"
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--configs-dir", default=str(_TOOL_DIR / "configs"))
    p.add_argument("--knowledge-dir", required=True)
    p.add_argument("--pin-model", default=None)
    p.add_argument("--stream-delay-s", type=float, default=None)
    p.add_argument("--planner", default="",
                    help="Recorded into CV predictions.csv's planner column only — "
                         "matches run_experiment.py's schema. This process makes no "
                         "planning decisions itself regardless of this value.")
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
        configure_device(dataset_config)  # CV inference on the GPU (core/device.py)

    adapter = _build_adapter(dataset_config, configs_dir)

    _seed_local_weights(dataset_config, knowledge_dir)
    local_config = kio.localize_dataset_config(dataset_config, knowledge_dir)

    model_names = list(dataset_config.get("models", {}).keys())
    initial_model = args.pin_model or model_names[0]
    if not kp["model_file"].exists():
        kio.write_current_model(kp["model_file"], initial_model)

    scaler = None
    if not is_cv:
        train_values = adapter.train_split()
        scaler = _fit_scaler(train_values)
        with open(kp["scaler_file"], "wb") as f:
            pickle.dump(scaler, f)

    pred_dir: Path | None = None
    if is_cv:
        pred_dir = knowledge_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)

    energy_backend = dataset_config.get("energy_meter", "auto")
    stream_delay_s = (
        args.stream_delay_s if args.stream_delay_s is not None
        else float(dataset_config.get("stream_delay_s", 0.0))
    )

    # Load every model up front — same call, same semantics as
    # experiments/run_experiment.py's own inference setup, against this
    # run's local weight copies (local_config) rather than the shared ones.
    models = _load_models(local_config, configs_dir)
    available = [m for m in model_names if models.get(m) is not None]
    if not available:
        raise RuntimeError(
            f"No loadable models for dataset '{args.dataset}' — "
            "see the 'Failed to load model' / 'weights not found' warnings above: "
            "either a weights file is missing or a required package is not installed "
            "(e.g. transformers for SegFormer, ultralytics for YOLO, torchvision)."
        )

    # ── Inference/manager barrier (audit D1/D2/D3, 2026-09-28) ─────────────
    # Borrowed from the single-threaded harness, where inference simply stops
    # at every monitor_interval boundary while the MAPE cycle runs, so a
    # decision always takes effect on the very next row. Here inference may
    # run at most `max_batches_ahead` batches past the last batch t1 has
    # fully processed (kio.PROCESSED_FILE). With the default of 1, inference
    # keeps streaming batch k+1 while the manager decides on batch k (real
    # concurrency), and a decision made on batch k takes effect exactly at
    # the start of batch k+2 — a fixed, one-batch lag instead of an
    # unbounded one (was median 39-86 steps, max 2,821). A long retrain
    # therefore pauses the stream after at most one batch, so a fine-tuned
    # CV model actually serves the rest of the stream (previously every CV
    # fine-tune finished after the stream had ended).
    #   max_batches_ahead = 0  → strict lockstep (single-threaded semantics)
    #   max_batches_ahead < 0  → barrier disabled (pre-2026-09-28 free-running)
    monitor_interval = int(dataset_config.get("monitor_interval", 50))
    max_ahead = int(dataset_config.get("max_batches_ahead", 1))

    def _wait_for_manager(step_: int) -> bool:
        """Block until t1 has processed enough rows. False if the run aborted."""
        need = step_ - monitor_interval * max_ahead
        while kio.read_processed(kp["processed"]) < need:
            if kp["abort"].exists():
                return False
            time.sleep(0.005)
        return True

    # Batch-level energy metering (audit C1, 2026-10-06) — see
    # experiments/run_experiment.BatchEnergy. One EnergyMeter spans each
    # monitor_interval batch, held under the cross-process energy lock so no
    # MAPE-side meter overlaps it; the batch's prediction rows are buffered
    # and written when it closes, each charged total / n. The batch is closed
    # (rows written, lock released) BEFORE waiting on the manager, which
    # needs those rows and the lock to process it.
    batch_mode = str(dataset_config.get("energy_metering", "batch")) == "batch"
    # Original HarmonE metering (2026-10-09; energy_metering "per_prediction",
    # regression): exactly HarmonE/inference.py — after reading the active
    # model's name, a pyRAPL-equivalent meter (Pkg0Meter: socket-0 package
    # counter at begin/end, no lock, no DRAM/GPU) spans ONLY the model load
    # from disk + the prediction; each row's energy_uJ is that reading and
    # inference_time_s the same window's wall time (time.time(), as the
    # original). Whatever else runs on the package during the window (the
    # managing system, a retrain) is in the reading, as in the original.
    # The reported energy is the mean over all predictions (paper Table 2).
    per_prediction = (str(dataset_config.get("energy_metering", "batch")) == "per_prediction"
                      and not is_cv)
    pkg0 = Pkg0Meter() if per_prediction else None
    pred_fields = kio.PREDICTION_FIELDS_CV if is_cv else kio.PREDICTION_FIELDS_REGRESSION
    batch_rows: list[dict] = []
    batch_lock = contextlib.ExitStack()
    batch_em = None

    def _open_batch(step_: int) -> None:
        nonlocal batch_em
        batch_lock.enter_context(kio.energy_lock(knowledge_dir, blocking=True))
        batch_em = EnergyMeter(f"inference_batch_{step_}", backend=energy_backend)
        batch_em.__enter__()

    def _close_batch() -> None:
        nonlocal batch_em
        if batch_em is None:
            return
        em_, batch_em = batch_em, None
        em_.__exit__(None, None, None)
        batch_lock.close()
        valid = em_.total_uJ is not None
        per_step = (em_.total_uJ / len(batch_rows)) if (valid and batch_rows) else 0.0
        for r in batch_rows:
            r["energy_uJ"] = round(per_step, 4)
            r["energy_valid"] = valid
            kio.append_row(kp["predictions_file"], r, pred_fields)
        batch_rows.clear()

    def _emit(row: dict) -> None:
        """Batch mode: buffer until the batch closes; step mode: write now."""
        if batch_mode:
            batch_rows.append(row)
        else:
            kio.append_row(kp["predictions_file"], row, pred_fields)

    # Per-step model loading (2026-10-08; config reload_model_per_step), as
    # in the original HarmonE inference.py: every prediction re-reads which
    # model is active (model.txt, written by the managing system) and loads
    # that model's weights from its file (rewritten by retrain/restore). The
    # loading cost is part of the measured energy. Model switches can then
    # land mid-batch, as in the original; the monitor scores the batch's
    # majority model and only judges batches served by the current model.
    per_step_reload = bool(dataset_config.get("reload_model_per_step", False))

    step = 0
    reload_attempts: dict[str, int] = {}
    current_model = kio.read_current_model(kp["model_file"], default=initial_model)
    for sample in adapter.stream():
        # Model changes (switch, or new weights after retrain/replace) are
        # picked up ONLY at batch boundaries, so every monitor_interval-row
        # batch is served by a single model — t1's batches are never mixed
        # (audit D1: previously 267 of 466 batches mixed 2-3 models).
        if step % monitor_interval == 0:
            _close_batch()
            if step > 0 and max_ahead >= 0 and not _wait_for_manager(step):
                logger.error("inference_proc: manager aborted — stopping stream at step %d.", step)
                break
            current_model = kio.read_current_model(kp["model_file"], default=initial_model)
            # Cross-process retrain/replace signal — see _reload_one_model's docstring.
            if not per_step_reload and kio.consume_reload_flag(knowledge_dir, current_model):
                if _reload_one_model(models, current_model, local_config):
                    reload_attempts.pop(current_model, None)
                else:
                    n = reload_attempts.get(current_model, 0) + 1
                    reload_attempts[current_model] = n
                    kio.append_row(
                        knowledge_dir / kio.RELOAD_FAILURES_FILE,
                        {"step": step, "model": current_model, "attempt": n,
                         "error": _last_reload_error.get(current_model, "")[:300]},
                        ["step", "model", "attempt", "error"],
                    )
                    if n < _MAX_RELOAD_ATTEMPTS:
                        kio.write_reload_flag(knowledge_dir, current_model)  # retry next batch
                    else:
                        reload_attempts.pop(current_model, None)
                        logger.error("inference_proc: giving up reloading %r after %d attempts.",
                                     current_model, n)

        if batch_mode and batch_em is None:
            _open_batch(step)

        if per_step_reload:
            current_model = kio.read_current_model(kp["model_file"], default=initial_model)
            if not per_prediction:  # per_prediction loads inside its meter window
                _reload_one_model(models, current_model, local_config)

        predict_fn = models.get(current_model)
        if predict_fn is None:
            for name in available:
                predict_fn = models[name]
                current_model = name
                break

        if is_cv:
            image_path = str(sample.inputs)
            # Image/luminance rows are written BEFORE the prediction row (audit
            # D9): the manager drains predictions.csv in batches and then the
            # matching image/luminance rows, so they must already exist
            # whenever the corresponding prediction row does.
            kio.append_row(
                kp["image_history_file"],
                {"step": step, "image_path": image_path},
                ["step", "image_path"],
            )
            luminance = _compute_image_luminance(image_path)
            kio.append_row(
                knowledge_dir / "luminance_history.csv",
                {"step": step, "luminance": luminance},
                ["step", "luminance"],
            )
            # Blocking cross-process lock (audit C3): every step is measured,
            # like single-threaded; the lock only keeps this meter's window
            # from overlapping a MAPE-side meter.
            energy_uJ, energy_valid = 0.0, True  # batch mode: set by _close_batch
            if batch_mode:
                try:
                    raw_pred = predict_fn(image_path)
                except Exception as exc:
                    logger.warning("CV prediction failed at step %d: %s", step, exc)
                    raw_pred = None
            else:
                with kio.energy_lock(knowledge_dir, blocking=True):
                    em = EnergyMeter(f"inference_step_{step}", backend=energy_backend)
                    em.__enter__()
                    try:
                        raw_pred = predict_fn(image_path)
                    except Exception as exc:
                        logger.warning("CV prediction failed at step %d: %s", step, exc)
                        raw_pred = None
                    finally:
                        em.__exit__(None, None, None)
                energy_uJ = em.total_uJ if em.total_uJ is not None else 0.0
                energy_valid = em.valid
            proxy_acc = _extract_cv_proxy(raw_pred)
            _emit({
                "step": step, "proxy_acc": round(proxy_acc, 6),
                "active_model": current_model, "planner": args.planner,
                "energy_uJ": round(energy_uJ, 4), "energy_valid": energy_valid,
            })
            if raw_pred is not None and pred_dir is not None:
                _save_cv_prediction(raw_pred, step, pred_dir, cv_task)
            # Exact same line/format as experiments/run_experiment.py:2645-2647 —
            # CV has no periodic MAPE-cycle line of its own driving visibility
            # the way regression's monitor_interval batching does, so
            # single-threaded logs one of these per step.
            logger.info(
                "stream  step=%-5d  model=%-12s  conf=%.4f  energy=%s",
                step, current_model, proxy_acc,
                "per-batch" if batch_mode else f"{energy_uJ:.1f} µJ",
            )
        else:
            scaled_inputs = scaler.transform(sample.inputs.reshape(-1, 1)).flatten()
            # Blocking cross-process lock — see the CV branch above.
            energy_uJ, energy_valid = 0.0, True  # batch mode: set by _close_batch
            inference_time_s = None
            if per_prediction:
                pkg0.begin()
                t0 = time.time()
                try:
                    if per_step_reload:
                        _reload_one_model(models, current_model, local_config)
                        predict_fn = models.get(current_model) or predict_fn
                    raw_pred = predict_fn(scaled_inputs)
                except Exception as exc:
                    logger.warning("Prediction failed at step %d: %s", step, exc)
                    raw_pred = 0.0
                inference_time_s = time.time() - t0
                reading = pkg0.end()
                energy_uJ, energy_valid = (reading or 0.0), reading is not None
            elif batch_mode:
                try:
                    raw_pred = predict_fn(scaled_inputs)
                except Exception as exc:
                    logger.warning("Prediction failed at step %d: %s", step, exc)
                    raw_pred = 0.0
            else:
                with kio.energy_lock(knowledge_dir, blocking=True):
                    em = EnergyMeter(f"inference_step_{step}", backend=energy_backend)
                    em.__enter__()
                    try:
                        raw_pred = predict_fn(scaled_inputs)
                    except Exception as exc:
                        logger.warning("Prediction failed at step %d: %s", step, exc)
                        raw_pred = 0.0
                    finally:
                        em.__exit__(None, None, None)
                energy_uJ = em.total_uJ if em.total_uJ is not None else 0.0
                energy_valid = em.valid
            y_pred = float(scaler.inverse_transform([[raw_pred]])[0][0])
            y_true = float(sample.ground_truth)
            _emit({
                "step": step, "y_true": round(y_true, 6), "y_pred": round(y_pred, 6),
                "active_model": current_model,
                "energy_uJ": round(energy_uJ, 4), "energy_valid": energy_valid,
                "inference_time_s": None if inference_time_s is None else round(inference_time_s, 7),
            })
            # Regression has no per-step log line in single-threaded either
            # (relies on the periodic "MAPE[...]" cycle line instead — see
            # mape/mape_cycle.py) — matching that means NOT adding one here.

        if stream_delay_s > 0:
            time.sleep(stream_delay_s)
        step += 1

    _close_batch()  # trailing partial batch
    kio.touch(kp["stream_done"])
    logger.info("inference_proc: stream complete, %d steps.", step)


if __name__ == "__main__":
    main()

"""concurrent_harness/run_concurrent.py — top-level launcher/orchestrator.

The original HarmonE has no equivalent of this file: a human opens two
terminals, runs inference.py in one and mape/manage.py in the other, and
manually kills manage.py once inference.py's stream finishes. This closes
that gap so a concurrent run is a single command, bounded, and produces a
run_manifest.json compatible with experiments/metrics.py and
scripts/plot_results.py — without modifying either of those files.

Spawns inference.py and mape/manage.py as separate OS processes (not threads,
not multiprocessing) — the same process boundary the original uses. Energy
meters in both processes are serialized by a cross-process flock
(knowledge_io.energy_lock) so their windows never overlap; the manifest
reports inference energy and MAPE-K overhead separately, like
run_experiment.py (audit C3, 2026-09-28).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_TOOL_DIR = _THIS_DIR.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))
sys.path.insert(0, str(_THIS_DIR))

import knowledge_io as kio

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 1.0
# How long to wait for mape/manage.py to exit after shutdown is signaled.
# If t1/t2 is mid-retrain when shutdown fires, it won't even check the
# shutdown flag until execute_decision() returns — regression retrains are
# fast (a sklearn .fit() or a 5-step LSTM backprop), but a real CV fine-tune
# (verified: YOLO's ultralytics .train() call, 3 epochs, ~1.1 it/s x 225
# batches) can legitimately take several minutes. A too-short timeout here
# doesn't just leave a lingering process — it force-kills a working retrain
# mid-training and reports the whole run as FAILED. Sized well above the
# slowest observed real retrain, not just "a bit more than usual."
SHUTDOWN_TIMEOUT_S = 600.0


def build_run_id(dataset: str, planner: str, pin_model: str | None) -> str:
    parts = [dataset, planner]
    if pin_model:
        parts.append(pin_model)
    parts.append("conc")
    return "_".join(parts)


def _read_csv_column(path: Path, col: str) -> list[float]:
    import csv
    if not path.exists():
        return []
    with open(path, "r", newline="") as f:
        return [float(r[col]) for r in csv.DictReader(f)]


def _count_csv_rows(path: Path) -> int:
    """Data rows in a headed CSV; 0 if the file doesn't exist."""
    if not path.exists():
        return 0
    with open(path, newline="") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def _build_task_metrics_and_energy_by_model(predictions_file: Path, is_cv: bool) -> tuple[dict, dict]:
    import csv
    if not predictions_file.exists():
        return {}, {}
    with open(predictions_file, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}, {}

    # energy_per_inference_mJ (2026-09-06) is computed from VALID readings
    # only — n_steps counts every row active_model served, but a step whose
    # EnergyMeter attempt lost the non-blocking cross-process energy_lock
    # (kio.energy_lock; typically because a MAPE cycle held it) is recorded
    # as energy_uJ=0.0, energy_valid=False, not "measured and free". Averaging
    # over all of n_steps (including those zeroed rows) understates true
    # per-inference cost, and understates it MORE for planners that generate
    # more lock contention (more MAPE cycles / more retrains) — exactly the
    # planners you'd most want an honest number for. n_valid_steps and
    # energy_per_inference_mJ use only rows with energy_valid == "True";
    # n_steps (kept for backward compatibility) still counts every row.
    energy_by_model: dict[str, dict] = {}
    for r in rows:
        m = r.get("active_model", "unknown")
        eb = energy_by_model.setdefault(
            m, {"energy_uJ": 0.0, "n_steps": 0, "n_valid_steps": 0})
        eb["n_steps"] += 1
        if r.get("energy_valid") == "True":
            eb["energy_uJ"] += float(r["energy_uJ"])
            eb["n_valid_steps"] += 1
    for v in energy_by_model.values():
        v["energy_uJ"] = round(v["energy_uJ"], 2)
        v["energy_mJ"] = round(v["energy_uJ"] / 1000.0, 4)
        v["energy_per_inference_mJ"] = (
            round(v["energy_mJ"] / v["n_valid_steps"], 6) if v["n_valid_steps"] else None
        )

    # total_inference_energy_* (2026-09-28, audit C3): restored, same field
    # names as run_experiment.py. Every step is now measured (inference.py
    # uses a blocking energy lock), and MAPE-side energy is accumulated
    # separately in event_counters.mape_k_energy_uJ, so the split matches the
    # single-threaded harness.
    _total_e_uJ = sum(float(r["energy_uJ"]) for r in rows if r.get("energy_valid") == "True")
    _energy_fields = {
        "total_inference_energy_uJ": round(_total_e_uJ, 2),
        "total_inference_energy_mJ": round(_total_e_uJ / 1000.0, 4),
        # Steps with no energy reading (audit C6); excluded from the total.
        "energy_invalid_steps": sum(1 for r in rows if r.get("energy_valid") != "True"),
    }

    if is_cv:
        # Exact same schema as experiments/run_experiment.py's own CV
        # task_metrics block — mean/std/min/max proxy accuracy plus a
        # per-model breakdown, plus total inference energy.
        import numpy as np
        proxy_vals = [float(r["proxy_acc"]) for r in rows]
        model_confs: dict[str, list] = {}
        for r in rows:
            model_confs.setdefault(r.get("active_model", "unknown"), []).append(float(r["proxy_acc"]))
        task_metrics = {
            "mean_proxy_acc": round(sum(proxy_vals) / len(proxy_vals), 6),
            "std_proxy_acc": round(float(np.std(proxy_vals)), 6),
            "min_proxy_acc": round(min(proxy_vals), 6),
            "max_proxy_acc": round(max(proxy_vals), 6),
            "n_samples": len(rows),
            "per_model": {
                m: {"mean_proxy_acc": round(sum(v) / len(v), 6), "n_steps": len(v)}
                for m, v in model_confs.items()
            },
            **_energy_fields,
        }
    else:
        # Exact same schema as run_experiment.py's own regression task_metrics
        # block — r2/rmse/mae plus total inference energy.
        from sklearn.metrics import r2_score
        y_true = [float(r["y_true"]) for r in rows]
        y_pred = [float(r["y_pred"]) for r in rows]
        n = len(y_true)
        mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n
        mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n
        r2 = float(r2_score(y_true, y_pred)) if n > 1 else 0.0
        task_metrics = {
            "r2": round(r2, 6),
            "rmse": round(mse ** 0.5, 6),
            "mae": round(mae, 6),
            "n_samples": n,
            **_energy_fields,
        }
    return task_metrics, energy_by_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--planner", required=True)
    p.add_argument("--pin-model", default=None)
    p.add_argument("--seed", type=int, default=42,
                    help="Passed to both subprocesses, which seed random/numpy like "
                         "run_experiment.py. Thread/process timing can still differ "
                         "run to run, so results are close but not bit-identical.")
    p.add_argument("--configs-dir", default=str(_TOOL_DIR / "configs"))
    p.add_argument("--runs-dir", default=str(_THIS_DIR / "runs"))
    p.add_argument("--stream-delay-s", type=float, default=None)
    p.add_argument("--shutdown-timeout-s", type=float, default=SHUTDOWN_TIMEOUT_S,
                    help="How long to wait for mape/manage.py to exit after signaling "
                         "shutdown — must exceed the slowest retrain that could be mid-flight.")
    p.add_argument("--run-id", default=None,
                    help="Override the computed run_id (e.g. from a grid script that "
                         "already decided the name for its own skip-if-exists check).")
    p.add_argument("--verbose", action="store_true",
                    help="DEBUG-level logging, propagated to both inference.py and "
                         "mape/manage.py subprocesses too.")
    args = p.parse_args()

    kio.setup_logging(args.verbose)

    run_id = args.run_id or build_run_id(args.dataset, args.planner, args.pin_model)
    run_dir = Path(args.runs_dir) / run_id
    # knowledge_dir == run_dir (no separate "knowledge/" subdirectory) so
    # every per-run artifact (predictions.csv, predictions/, mape_events.csv,
    # etc.) lands directly under runs/<run_id>/, exactly matching
    # run_experiment.py's own flat layout — offline_eval.py (and
    # scripts/run_offline_eval_conc.sh, which shells out to it) look for
    # predictions.csv/predictions/ directly in run_dir and are not, and
    # should not need to be, aware of a concurrent-only "knowledge/" indirection.
    knowledge_dir = run_dir
    if run_dir.exists():
        # run_id is fixed (not timestamped, unlike run_experiment.py's
        # _build_run_id) so repeated invocations reuse the same directory —
        # start every run from a clean knowledge dir rather than silently
        # appending to a previous run's predictions.csv/mape_info.json.
        import shutil
        shutil.rmtree(run_dir)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    kp = kio.knowledge_paths(knowledge_dir)

    from experiments.run_experiment import (
        _load_dataset_config, _build_adapter, _fit_scaler,
        _train_regression_models, _train_cv_models_if_missing, _stale_regression_models,
    )
    import os as _os

    dataset_config = _load_dataset_config(args.dataset, Path(args.configs_dir))
    is_cv = dataset_config.get("domain", "regression") == "cv"
    compute_device = {"device": "cpu"}
    if is_cv:
        # Initial-training bootstrap runs here (before the subprocesses) — on
        # the GPU like everything else CV. Fails fast if "device": "cuda" is
        # configured but CUDA is unavailable (core/device.py).
        from core.device import configure_device, device_info
        configure_device(dataset_config)
        compute_device = device_info()

    # Initial-training bootstrap — same gate and same functions
    # experiments/run_experiment.py itself calls inline when weights are
    # missing (run_experiment.py:2476-2492), so behavior matches the
    # single-threaded harness exactly. Trained weights are written to the
    # SHARED configs/datasets/*.json weights_path (not a per-run local copy)
    # — this is a one-time bootstrap step, not a mid-run retrain, so it's
    # meant to be reused by every future run (single-threaded or concurrent)
    # against this dataset, same as the original's tools/train_models.py.
    # Run once here, before either subprocess starts, to avoid both
    # processes racing to train+write the same files concurrently.
    adapter = _build_adapter(dataset_config, Path(args.configs_dir))
    if not is_cv:
        train_values = adapter.train_split()
        stale = _stale_regression_models(dataset_config, train_values)
        if stale:
            logger.info("run_concurrent: model weights missing or stale for '%s' %s "
                        "— training inline on training split (matches run_experiment.py) ...",
                        args.dataset, stale)
            bootstrap_scaler = _fit_scaler(train_values)
            _train_regression_models(dataset_config, train_values, bootstrap_scaler, only=stale)
    else:
        train_paths = adapter.train_split()
        train_label_paths, train_inline_labels = adapter.train_labels()
        _train_cv_models_if_missing(
            dataset_config, dataset_config.get("task", "classification"),
            train_paths, train_label_paths, train_inline_labels,
            dataset_config, run_dir,
        )

    inf_cmd = [
        sys.executable, str(_THIS_DIR / "inference.py"),
        "--dataset", args.dataset, "--configs-dir", args.configs_dir,
        "--knowledge-dir", str(knowledge_dir), "--planner", args.planner,
        "--seed", str(args.seed),
    ]
    if args.pin_model:
        inf_cmd += ["--pin-model", args.pin_model]
    if args.stream_delay_s is not None:
        inf_cmd += ["--stream-delay-s", str(args.stream_delay_s)]
    if args.verbose:
        inf_cmd += ["--verbose"]

    mgr_cmd = [
        sys.executable, str(_THIS_DIR / "mape" / "manage.py"),
        "--dataset", args.dataset, "--configs-dir", args.configs_dir,
        "--knowledge-dir", str(knowledge_dir), "--planner", args.planner,
        "--seed", str(args.seed),
    ]
    if args.pin_model:
        mgr_cmd += ["--pin-model", args.pin_model]
    if args.verbose:
        mgr_cmd += ["--verbose"]

    # thresholds.json snapshot — matches run_experiment.py's own thresholds_path
    # write (a copy of the effective config for this run, for provenance/debugging).
    with open(kp["thresholds_file"], "w") as f:
        json.dump(dataset_config, f, indent=2)

    start_ts = datetime.now(timezone.utc).isoformat()
    start_wall = time.monotonic()
    failure_reason: str | None = None

    # Both subprocesses INHERIT this process's own stdout/stderr rather than
    # being redirected to separate per-process files (the previous design:
    # run_dir/inference.log + run_dir/manage.log). That previous split meant
    # the one file the grid scripts actually tee (runs/logs/<run_id>.log)
    # only ever captured this launcher's own ~10 summary lines — none of
    # inference.py's per-step "stream ..." lines or mape/manage.py's
    # "MAPE[...]"/switch/replace/retrain lines, which is the reason those
    # log files looked broken/empty compared to the single-threaded harness's.
    # experiments/run_experiment.py never manages its own log file either —
    # it just logs to the console and scripts/run_{cv,regression}.sh capture
    # that with `... 2>&1 | tee "$run_log"`. Inheriting stdio here reproduces
    # exactly that mechanism for the concurrent case: this launcher and both
    # subprocesses share the same underlying stdout/stderr fd, so whatever
    # wraps THIS process (a grid script's tee, or a human's terminal for a
    # manual run) transparently captures all three processes' output,
    # correctly interleaved by real write order — no new file-plumbing needed,
    # and run_concurrent_{cv,regression}_grid.sh need no changes to benefit.
    logger.info("run_concurrent: starting inference_proc + manage_proc for %s", run_id)
    inf_proc = subprocess.Popen(inf_cmd, cwd=str(_TOOL_DIR))
    mgr_proc = subprocess.Popen(mgr_cmd, cwd=str(_TOOL_DIR))

    while not kp["stream_done"].exists():
        if inf_proc.poll() is not None:
            failure_reason = (
                f"inference.py exited early (code {inf_proc.returncode}) "
                "before writing _stream_done — see this run's log output"
            )
            logger.warning("run_concurrent: %s", failure_reason)
            break
        if mgr_proc.poll() is not None:
            # inference.py waits on the manager at every batch boundary (the
            # barrier, see inference.py) — if the manager is gone it would
            # wait forever, so tell it to stop.
            failure_reason = (
                f"mape/manage.py exited early (code {mgr_proc.returncode}) "
                "before the stream finished — see this run's log output"
            )
            logger.warning("run_concurrent: %s", failure_reason)
            kio.touch(kp["abort"])
            break
        time.sleep(POLL_INTERVAL_S)

    kio.touch(kp["shutdown"])
    logger.info("run_concurrent: stream complete, signalled shutdown to manage_proc.")

    try:
        inf_proc.wait(timeout=args.shutdown_timeout_s)
    except subprocess.TimeoutExpired:
        inf_proc.terminate()
        if failure_reason is None:
            failure_reason = "inference.py did not exit within the shutdown timeout"

    if mgr_proc.poll() is None:
        try:
            mgr_proc.wait(timeout=args.shutdown_timeout_s)
        except subprocess.TimeoutExpired:
            logger.warning("run_concurrent: manage_proc did not exit in time — terminating.")
            mgr_proc.terminate()
            mgr_proc.wait()
            if failure_reason is None:
                failure_reason = "mape/manage.py did not exit within the shutdown timeout"

    # Checked unconditionally (not just in an early-exit branch) — the common
    # case is manage.py still running at the poll() check above and exiting
    # normally inside the wait() just taken, which used to fall through
    # without ever inspecting returncode at all. That silently swallowed
    # mape/manage.py's own exit-status signal (2026-09-06: added so a thread
    # dying mid-run, see mape/manage.py's `errors`/sys.exit(1), is reported
    # as "status": "failed" instead of "ok" — this check is what makes that
    # visible; without it the fix in manage.py would have no effect here).
    if mgr_proc.returncode not in (None, 0) and failure_reason is None:
        failure_reason = f"mape/manage.py exited with code {mgr_proc.returncode} — see this run's log output"

    elapsed_s = time.monotonic() - start_wall
    end_ts = datetime.now(timezone.utc).isoformat()

    # Whole-stream CPU package energy (2026-10-07): RAPL counter read by
    # inference.py just before its first prediction vs now (both processes
    # have exited). One overlap- and gap-free total covering inference,
    # MAPE-K and everything in between; None without RAPL.
    from core.energy import read_package_counter, package_energy_between
    stream_pkg_uJ, stream_pkg_s = None, None
    try:
        _start = json.loads((knowledge_dir / kio.PKG_START_FILE).read_text())
        stream_pkg_uJ = package_energy_between(_start.get("counter"), read_package_counter())
        stream_pkg_s = round(time.time() - float(_start["t"]), 3)
    except (OSError, ValueError, KeyError):
        pass

    mape_info = {}
    if kp["mape_info_file"].exists():
        with open(kp["mape_info_file"]) as f:
            mape_info = json.load(f)
    event_counters = mape_info.get("event_counters", {
        "model_switches": 0, "retrains": 0, "retrain_skipped": 0,
        "vmr_events": 0, "noops": 0, "noop_on_violation": 0, "mape_k_energy_uJ": 0.0,
        "mape_energy_invalid_cycles": 0,
    })

    total_steps = len(_read_csv_column(kp["predictions_file"], "step"))
    task_metrics, energy_by_model = _build_task_metrics_and_energy_by_model(kp["predictions_file"], is_cv)

    # Energy split like run_experiment.py (audit C3, 2026-09-28): inference
    # total in task_metrics.total_inference_energy_uJ, MAPE-K overhead in
    # event_counters.mape_k_energy_uJ. energy_used_uJ/mJ (their sum) is kept
    # for scripts/generate_results_table.py; every EnergyMeter in the run is
    # opened under the blocking cross-process energy_lock, so the two never
    # overlap in time and the sum does not double count.
    event_counters = dict(event_counters)
    _inf_uJ = float(task_metrics.get("total_inference_energy_uJ", 0.0)) if task_metrics else 0.0
    _mape_uJ = float(event_counters.get("mape_k_energy_uJ", 0.0))
    event_counters["energy_used_uJ"] = round(_inf_uJ + _mape_uJ, 2)
    event_counters["energy_used_mJ"] = round((_inf_uJ + _mape_uJ) / 1000.0, 4)

    # A clean exit with zero recorded steps is still a failure (e.g. the
    # stream produced nothing before crashing between polls) — matches
    # run_experiment.py's own RuntimeError when no models load at all.
    if failure_reason is None and total_steps == 0:
        failure_reason = "no prediction rows were written — inference.py produced no output"

    manifest = {
        "run_id": run_id,
        "dataset": args.dataset,
        "planner": args.planner,
        "pin_model": args.pin_model,
        "seed": args.seed,
        "mode": "concurrent",
        "compute_device": compute_device,
        "started_at": start_ts,
        "finished_at": end_ts,
        "elapsed_s": round(elapsed_s, 3),
        "monitor_interval": int(dataset_config.get("monitor_interval", 50)),
        "max_batches_ahead": int(dataset_config.get("max_batches_ahead", 1)),
        "total_steps": total_steps,
        # t1 planning cycles only (one per batch) — same meaning as
        # single-threaded's mape_cycles. t2's detections: drift_checks.
        "mape_cycles": mape_info.get("cycles", 0),
        "drift_checks": mape_info.get("drift_checks", 0),
        "final_model": kio.read_current_model(kp["model_file"]),
        "stream_package_energy_uJ": None if stream_pkg_uJ is None else round(stream_pkg_uJ, 1),
        "stream_package_window_s": stream_pkg_s,
        # Failed cross-process weight reloads (audit D6); > 0 means some
        # counted adaptations never reached the stream.
        "reload_failures": _count_csv_rows(knowledge_dir / kio.RELOAD_FAILURES_FILE),
        "event_counters": event_counters,
        "final_ema_scores": mape_info.get("ema_scores", {}),
        "task_metrics": task_metrics,
        "energy_by_model": energy_by_model,
        "artifacts": {
            "predictions_csv": str(kp["predictions_file"]),
            "mape_events_csv": str(kp["mape_events_file"]),
            "planner_decisions_csv": str(kp["planner_decisions_file"]),
            "mape_info_json": str(kp["mape_info_file"]),
            "thresholds_json": str(kp["thresholds_file"]),
            "cv_predictions_dir": str(knowledge_dir / "predictions") if is_cv else None,
        },
        "status": "ok" if failure_reason is None else "failed",
    }
    if failure_reason is not None:
        manifest["failure_reason"] = failure_reason

    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Exact same summary block as experiments/run_experiment.py:3038-3080 —
    # "run_experiment done | ...", "accuracy summary | ...",
    # "energy summary | ...", "energy by model | ..." — same field names,
    # same order, same %-format specifiers, so a completed concurrent run's
    # tail looks identical to a completed single-threaded one.
    logger.info(
        "run_experiment done | steps=%d cycles=%d switches=%d retrains=%d elapsed=%.1fs",
        total_steps, manifest["mape_cycles"],
        event_counters.get("model_switches", 0), event_counters.get("retrains", 0),
        elapsed_s,
    )
    if task_metrics:
        if is_cv:
            logger.info(
                "accuracy summary  | mean_conf=%.4f  std=%.4f  min=%.4f  max=%.4f  n=%d",
                task_metrics["mean_proxy_acc"], task_metrics["std_proxy_acc"],
                task_metrics["min_proxy_acc"], task_metrics["max_proxy_acc"],
                task_metrics["n_samples"],
            )
            for _m, _ms in sorted(task_metrics.get("per_model", {}).items()):
                logger.info(
                    "conf by model     | %-12s  steps=%d  mean_conf=%.4f",
                    _m, _ms["n_steps"], _ms["mean_proxy_acc"],
                )
        else:
            logger.info(
                "accuracy summary  | R²=%.4f  RMSE=%.4f  MAE=%.4f  n=%d",
                task_metrics.get("r2", float("nan")), task_metrics.get("rmse", float("nan")),
                task_metrics.get("mae", float("nan")), task_metrics.get("n_samples", 0),
            )
        # Same line as run_experiment.py (audit C3, 2026-09-28).
        _inf_mJ = _inf_uJ / 1000.0
        _mape_mJ = _mape_uJ / 1000.0
        logger.info(
            "energy summary    | inference=%.3f mJ  mape_overhead=%.3f mJ  total=%.3f mJ",
            _inf_mJ, _mape_mJ, _inf_mJ + _mape_mJ,
        )
    if energy_by_model:
        for _m, _ev in sorted(energy_by_model.items()):
            logger.info(
                "energy by model   | %-10s  steps=%d  energy=%.3f mJ",
                _m, _ev["n_steps"], _ev["energy_mJ"],
            )

    logger.info("run_concurrent: manifest written to %s", run_dir / "run_manifest.json")

    # Same tail-of-log JSON dump experiments/run_experiment.py's own main()
    # does (run_experiment.py:3166) — the grid scripts tee stdout to each
    # run's .log file, so this gives every concurrent run's log the same
    # full-manifest overview single-threaded logs already end with.
    print(json.dumps(manifest, indent=2))

    if failure_reason is not None:
        logger.error("run_concurrent: FAILED — %s", failure_reason)
        sys.exit(1)


if __name__ == "__main__":
    main()

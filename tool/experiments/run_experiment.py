"""
experiments/run_experiment.py — Headless single-run experiment driver.

Runs one (dataset × planner × seed) combination entirely in-process (no Flask/ACP).
Reads model weights from the paths in the dataset config; scales inputs with a
MinMaxScaler fitted on the training split; runs the MAPE loop inline.

Usage (CLI):
    python experiments/run_experiment.py \\
        --dataset pems_node1 --planner harmone_original --seed 42 \\
        --run-dir runs/my_run

Or from Python:
    from experiments.run_experiment import run_experiment
    manifest = run_experiment("pems_node1", "harmone_original", seed=42,
                              run_dir="runs/my_run")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────

_TOOL_DIR = Path(__file__).resolve().parent.parent  # tool/
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

# ── Local imports ─────────────────────────────────────────────────────────────

from core.planners.base import PlanningContext, PlanDecision, get_planner
from core.scoring import (
    compute_harmone_score,
    normalize_energy,
    update_ema,
    update_energy_threshold,
    update_separated_emas,
)
from core.energy import EnergyMeter
from core.drift.kl_fixed_ref import KLFixedRefDetector

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_MONITOR_INTERVAL = 50
_DRIFT_WINDOW_SIZE = 1200
_DRIFT_N_BINS = 50


# ── Setup helpers ─────────────────────────────────────────────────────────────

def setup_run_dir(run_dir: str) -> Path:
    """Create run directory and return as Path.  Idempotent."""
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_dataset_config(dataset_name: str, configs_dir: Path) -> dict:
    config_path = configs_dir / "datasets" / f"{dataset_name}.json"
    with open(config_path, "r") as f:
        return json.load(f)


def _build_adapter(dataset_config: dict, configs_dir: Path):
    """Instantiate the adapter class named in the config."""
    adapter_ref = dataset_config["adapter"]
    # e.g. "adapters.regression_csv.RegressionCSVAdapter"
    module_path, class_name = adapter_ref.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(dataset_config, config_dir=str(configs_dir / "datasets"))


def _fit_scaler(train_values: np.ndarray):
    """Fit a MinMaxScaler on the training split."""
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_values.reshape(-1, 1))
    return scaler


def _build_reference_distribution(
    train_values: np.ndarray, ref_path: str, n_bins: int = _DRIFT_N_BINS
) -> None:
    """Write reference_distribution.json for the KL drift detector."""
    hist, bin_edges = np.histogram(train_values, bins=n_bins)
    with open(ref_path, "w") as f:
        json.dump(
            {"histogram": hist.tolist(), "bin_edges": bin_edges.tolist()},
            f,
            indent=2,
        )


def _load_models(dataset_config: dict, configs_dir: Path) -> dict[str, Any]:
    """Load all models listed in the config.  Returns {name: predict_fn | None}.

    A None entry means the weights file was not found — that model will be
    treated as unavailable and skipped in planning.
    """
    from adapters.loaders import get_loader

    models: dict[str, Any] = {}
    config_dir = str(configs_dir / "datasets")
    for name, spec in dataset_config.get("models", {}).items():
        weights_path = spec["weights_path"]
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(config_dir, weights_path)
        if not os.path.exists(weights_path):
            logger.warning("Model '%s' weights not found at %s — skipping.", name, weights_path)
            models[name] = None
            continue
        try:
            loader_fn = get_loader(spec["loader"])
            models[name] = loader_fn(weights_path, seq_length=dataset_config.get("seq_length", 5))
        except Exception as exc:
            logger.warning("Failed to load model '%s': %s", name, exc)
            models[name] = None

    return models


def _initial_mape_info(models: dict[str, Any]) -> dict:
    """Build a fresh mape_info dict for a new run."""
    return {
        "current_energy_threshold": 0.6,
        "ema_scores": {m: 0.5 for m in models},
        "ema_accuracy": {m: 0.5 for m in models},
        "ema_energy": {m: 0.5 for m in models},
        "event_counters": {
            "model_switches": 0,
            "retrains": 0,
            "retrain_skipped": 0,
            "vmr_events": 0,
            "noops": 0,
            "mape_k_energy_uJ": 0.0,
        },
    }


# ── MAPE logic (inline) ───────────────────────────────────────────────────────

def _monitor_batch(
    y_true: list[float],
    y_pred: list[float],
    energies_uJ: list[float],
    current_model: str,
    mape_info: dict,
    thresholds: dict,
) -> dict:
    """Inline monitor: compute R², energy, EMA; return telemetry dict."""
    from sklearn.metrics import r2_score as _r2

    r2 = float(_r2(y_true, y_pred)) if len(y_true) > 1 else 0.0

    avg_energy_uJ = float(np.mean(energies_uJ)) if energies_uJ else 0.0
    e_min = thresholds.get("E_m", 0.0)
    e_max = thresholds.get("E_M", 25000.0)
    e_norm = normalize_energy(avg_energy_uJ, e_min, e_max)

    beta = thresholds.get("beta", 0.95)
    gamma = thresholds.get("gamma", 0.8)

    raw_score = compute_harmone_score(r2, e_norm, beta)
    prev_score = mape_info["ema_scores"].get(current_model, 0.5)
    ema = update_ema(prev_score, raw_score, gamma)

    mape_info["ema_scores"][current_model] = ema
    update_separated_emas(mape_info, current_model, r2, e_norm, gamma)

    return {
        "r2": round(r2, 6),
        "avg_energy_uJ": round(avg_energy_uJ, 4),
        "normalized_energy": round(e_norm, 6),
        "ema_score": round(ema, 6),
    }


def _analyse_violation(
    telemetry: dict,
    mape_info: dict,
    thresholds: dict,
) -> str | None:
    """Return 'score' | 'energy' | None."""
    min_score = thresholds.get("min_score", 0.78)
    energy_threshold = mape_info["current_energy_threshold"]

    if telemetry["ema_score"] < min_score:
        return "score"
    if telemetry["normalized_energy"] > energy_threshold:
        return "energy"
    return None


def _update_energy_boundary(
    mape_info: dict, telemetry: dict, thresholds: dict
) -> None:
    """Apply Eq. 3 (B1 fix) to update the adaptive energy threshold."""
    mape_info["current_energy_threshold"] = update_energy_threshold(
        current=mape_info["current_energy_threshold"],
        e_ref=thresholds.get("E_ref", 0.7),
        e_used=telemetry["normalized_energy"],
        delta=thresholds.get("delta", 0.1),
    )


def _analyse_drift(
    value_history: list[float],
    drift_detector: KLFixedRefDetector,
    thresholds: dict,
) -> dict:
    """Run drift detection; return B2-aligned dict."""
    detection = drift_detector.detect(value_history)
    if not detection["drift_detected"]:
        return {"drift_detected": False, "action": None, "version": None}

    # In the harness there is no VMR — always route to retrain
    return {"drift_detected": True, "action": "retrain", "version": None}


def _plan(
    violation: str | None,
    drift_result: dict,
    current_model: str,
    available_models: list[str],
    mape_info: dict,
    thresholds: dict,
    planner,
) -> PlanDecision:
    effective_violation = violation
    if drift_result["drift_detected"] and violation is None:
        effective_violation = "drift"

    ctx = PlanningContext(
        violation=effective_violation,
        ema_scores=dict(mape_info["ema_scores"]),
        ema_accuracy=dict(mape_info["ema_accuracy"]),
        ema_energy=dict(mape_info["ema_energy"]),
        current_model=current_model,
        available_models=available_models,
        thresholds=thresholds,
        drift_result=drift_result if drift_result["drift_detected"] else None,
    )
    return planner.plan(ctx)


def _execute(
    decision: PlanDecision,
    current_model: str,
    mape_info: dict,
    energy_uJ: float,
) -> str:
    """Apply decision; update mape_info counters.  Returns new current_model."""
    counters = mape_info["event_counters"]
    counters["mape_k_energy_uJ"] += energy_uJ

    if decision.action == "switch":
        new_model = decision.model
        if new_model and new_model != current_model:
            counters["model_switches"] += 1
            logger.debug("Switch: %s → %s  (%s)", current_model, new_model, decision.reason)
            return new_model

    elif decision.action in ("retrain", "replace"):
        # No live retrain in the harness — log and stay on current model
        counters["retrain_skipped"] += 1
        logger.debug("Retrain/replace skipped in harness: %s", decision.reason)

    else:
        counters["noops"] += 1

    return current_model


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(
    dataset_name: str,
    planner_name: str,
    seed: int,
    run_dir: str,
    *,
    configs_dir: str | None = None,
    monitor_interval: int = _DEFAULT_MONITOR_INTERVAL,
    stream_delay_s: float = 0.0,
    max_steps: int | None = None,
    extra_thresholds: dict | None = None,
) -> dict:
    """Run one (dataset × planner × seed) experiment and write artifacts.

    Args:
        dataset_name:      Key in configs/datasets/<name>.json.
        planner_name:      Registered planner name (e.g. "harmone_original").
        seed:              RNG seed for reproducibility.
        run_dir:           Directory where artifacts will be written.
        configs_dir:       Path to the configs/ directory; defaults to
                           <tool_dir>/configs.
        monitor_interval:  Predictions per MAPE cycle (default 50).
        stream_delay_s:    Seconds to sleep between predictions (0 for headless).
        max_steps:         Cap the stream at N steps (None = run to exhaustion).
        extra_thresholds:  Dict merged on top of dataset config thresholds
                           (for grid sweeps that override individual keys).

    Returns:
        The manifest dict (same content written to run_dir/run_manifest.json).
    """
    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)

    # ── Paths ─────────────────────────────────────────────────────────────────
    if configs_dir is None:
        configs_dir = str(_TOOL_DIR / "configs")
    configs_path = Path(configs_dir)
    run_path = setup_run_dir(run_dir)

    # ── Config ────────────────────────────────────────────────────────────────
    dataset_config = _load_dataset_config(dataset_name, configs_path)
    thresholds = dict(dataset_config)  # dataset config IS the thresholds dict
    thresholds["planner"] = planner_name
    if extra_thresholds:
        thresholds.update(extra_thresholds)

    energy_backend = thresholds.get("energy_meter", "null")  # null = no hardware needed

    # ── Adapter ───────────────────────────────────────────────────────────────
    adapter = _build_adapter(dataset_config, configs_path)
    # Override the adapter's per-sample sleep so headless runs are not throttled.
    # The dataset config has stream_delay_s=0.15 (for real-time simulation), but
    # experiment mode should run at CPU speed.  extra_thresholds and the caller's
    # stream_delay_s parameter both control this.
    effective_delay = float(thresholds.get("stream_delay_s", stream_delay_s))
    if hasattr(adapter, "_stream_delay"):
        adapter._stream_delay = effective_delay

    # ── Scaler (fit on training split) ────────────────────────────────────────
    train_values = adapter.train_split()
    scaler = _fit_scaler(train_values)

    import pickle
    scaler_path = run_path / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # ── Reference distribution for drift detection ─────────────────────────────
    ref_dist_path = str(run_path / "reference_distribution.json")
    _build_reference_distribution(train_values, ref_dist_path)
    drift_detector = KLFixedRefDetector(
        reference_path=ref_dist_path,
        tau_drift=thresholds.get("tau_drift", 0.5),
        window_size=_DRIFT_WINDOW_SIZE,
        n_bins=_DRIFT_N_BINS,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    models = _load_models(dataset_config, configs_path)
    available_models = [m for m, fn in models.items() if fn is not None]
    if not available_models:
        raise RuntimeError(
            f"No loadable models found for dataset '{dataset_name}'. "
            "Check that weights_path entries in the config point to existing files."
        )

    # ── Planner ───────────────────────────────────────────────────────────────
    planner = get_planner(planner_name)

    # ── State ─────────────────────────────────────────────────────────────────
    current_model = available_models[0]
    mape_info = _initial_mape_info(models)
    value_history: list[float] = []

    # Batch accumulators (reset every monitor_interval steps)
    batch_y_true: list[float] = []
    batch_y_pred: list[float] = []
    batch_energy_uJ: list[float] = []

    # Predictions log (written to CSV at end)
    prediction_rows: list[dict] = []
    mape_events: list[dict] = []

    # ── Run metadata ──────────────────────────────────────────────────────────
    start_ts = datetime.now(timezone.utc).isoformat()
    start_wall = time.monotonic()

    logger.info(
        "run_experiment | dataset=%s planner=%s seed=%d monitor_interval=%d",
        dataset_name, planner_name, seed, monitor_interval,
    )

    # ── Stream loop ───────────────────────────────────────────────────────────
    step = 0
    for sample in adapter.stream():
        if max_steps is not None and step >= max_steps:
            break

        # Scale inputs (B7 fix: scaler fitted on training data only)
        scaled_inputs = scaler.transform(sample.inputs.reshape(-1, 1)).flatten()

        # Predict with active model
        predict_fn = models.get(current_model)
        if predict_fn is None:
            # Fallback: pick first available model
            for m in available_models:
                if models[m] is not None:
                    current_model = m
                    predict_fn = models[m]
                    break

        with EnergyMeter(f"step_{step}", backend=energy_backend) as _em:
            try:
                raw_pred = predict_fn(scaled_inputs)
            except Exception as exc:
                logger.warning("Prediction failed at step %d: %s", step, exc)
                raw_pred = 0.0

        energy_uJ = _em.total_uJ or 0.0

        # Inverse-scale prediction to original units
        y_pred_orig = float(
            scaler.inverse_transform([[raw_pred]])[0][0]
        )
        y_true = float(sample.ground_truth)

        # Accumulate
        value_history.append(y_true)
        batch_y_true.append(y_true)
        batch_y_pred.append(y_pred_orig)
        batch_energy_uJ.append(energy_uJ)

        prediction_rows.append({
            "step": step,
            "y_true": round(y_true, 6),
            "y_pred": round(y_pred_orig, 6),
            "active_model": current_model,
            "energy_uJ": round(energy_uJ, 4),
        })

        if stream_delay_s > 0:
            time.sleep(stream_delay_s)

        # ── MAPE cycle ────────────────────────────────────────────────────────
        step += 1
        if step % monitor_interval == 0 and batch_y_true:
            # Monitor
            telemetry = _monitor_batch(
                batch_y_true, batch_y_pred, batch_energy_uJ,
                current_model, mape_info, thresholds,
            )

            # Analyse: violation?
            violation = _analyse_violation(telemetry, mape_info, thresholds)

            # Analyse: drift?
            drift_result = _analyse_drift(value_history, drift_detector, thresholds)

            # Update energy boundary (B1 fix)
            _update_energy_boundary(mape_info, telemetry, thresholds)

            # Plan
            with EnergyMeter("mape_plan", backend=energy_backend) as _plan_em:
                decision = _plan(
                    violation, drift_result, current_model,
                    available_models, mape_info, thresholds, planner,
                )

            # Execute
            new_model = _execute(
                decision, current_model, mape_info,
                energy_uJ=_plan_em.total_uJ or 0.0,
            )
            old_model = current_model
            current_model = new_model

            mape_events.append({
                "step": step,
                "violation": violation,
                "drift_detected": drift_result["drift_detected"],
                "kl_div": drift_result.get("kl_div"),
                "decision_action": decision.action,
                "decision_model": decision.model,
                "decision_reason": decision.reason,
                "model_before": old_model,
                "model_after": current_model,
                "r2": telemetry["r2"],
                "ema_score": telemetry["ema_score"],
                "avg_energy_uJ": telemetry["avg_energy_uJ"],
                "energy_threshold": round(mape_info["current_energy_threshold"], 6),
            })

            # Reset batch
            batch_y_true = []
            batch_y_pred = []
            batch_energy_uJ = []

    # ── Write artifacts ───────────────────────────────────────────────────────
    elapsed_s = time.monotonic() - start_wall
    end_ts = datetime.now(timezone.utc).isoformat()

    # predictions.csv
    import csv
    predictions_path = run_path / "predictions.csv"
    if prediction_rows:
        fieldnames = list(prediction_rows[0].keys())
        with open(predictions_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prediction_rows)

    # mape_events.csv
    events_path = run_path / "mape_events.csv"
    if mape_events:
        event_fields = list(mape_events[0].keys())
        with open(events_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=event_fields)
            writer.writeheader()
            writer.writerows(mape_events)

    # mape_info.json — final state
    mape_info_path = run_path / "mape_info.json"
    with open(mape_info_path, "w") as f:
        json.dump(mape_info, f, indent=2)

    # thresholds.json — copy of effective thresholds
    thresholds_path = run_path / "thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    # run_manifest.json — top-level metadata + summary
    manifest = {
        "run_id": run_path.name,
        "dataset": dataset_name,
        "planner": planner_name,
        "seed": seed,
        "monitor_interval": monitor_interval,
        "started_at": start_ts,
        "finished_at": end_ts,
        "elapsed_s": round(elapsed_s, 3),
        "total_steps": step,
        "mape_cycles": len(mape_events),
        "final_model": current_model,
        "event_counters": dict(mape_info["event_counters"]),
        "final_ema_scores": dict(mape_info["ema_scores"]),
        "artifacts": {
            "predictions_csv": str(predictions_path),
            "mape_events_csv": str(events_path),
            "mape_info_json": str(mape_info_path),
            "thresholds_json": str(thresholds_path),
            "reference_distribution_json": ref_dist_path,
            "scaler_pkl": str(scaler_path),
        },
        "status": "ok",
    }

    manifest_path = run_path / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "run_experiment done | steps=%d cycles=%d switches=%d elapsed=%.1fs",
        step, len(mape_events),
        mape_info["event_counters"]["model_switches"],
        elapsed_s,
    )

    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_run_id(dataset: str, planner: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{ts}_{dataset}_{planner}_s{seed}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a single HarmonE experiment.")
    parser.add_argument("--dataset", required=True, help="Dataset name (configs/datasets/<name>.json)")
    parser.add_argument("--planner", required=True, help="Planner name (e.g. harmone_original)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-dir", default=None, help="Output directory (auto-generated if omitted)")
    parser.add_argument("--runs-dir", default="runs", help="Parent dir for auto-generated run dirs")
    parser.add_argument("--monitor-interval", type=int, default=_DEFAULT_MONITOR_INTERVAL)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_dir = args.run_dir
    if run_dir is None:
        run_id = _build_run_id(args.dataset, args.planner, args.seed)
        run_dir = str(Path(args.runs_dir) / run_id)

    manifest = run_experiment(
        dataset_name=args.dataset,
        planner_name=args.planner,
        seed=args.seed,
        run_dir=run_dir,
        configs_dir=args.configs_dir,
        monitor_interval=args.monitor_interval,
        max_steps=args.max_steps,
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

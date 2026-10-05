"""
experiments/calibrate_energy_bounds.py — Per-dataset E_m/E_M energy-normalisation
bound calibration (regression datasets).

core/scoring.py::normalize_energy() clamps to [0, 1] correctly, but the E_m/E_M
bounds it's clamped against were hand-set, never validated against real measured
energy, and referenced a calibration script (this one) that never existed —
docs/datasets/uci_electricity.md and spot_prices.md both call
`python3 experiments/calibrate_energy_bounds.py --config <dataset>` as a documented
setup step for a script that was, until now, missing.

Consequence, found by inspecting logged runs (context, 2026-09-02): pems' E_M=12000
sat BELOW the lstm model's own p99 batch-energy (13346 uJ, from pooled
mape_events.csv logs across every pems_driftinduced planner run) — meaning an
ordinary noisy reading, not genuine sustained overuse, was enough to trip an
"energy" violation and force a spurious model switch. This script closes that gap
the same way calibrate_drift_threshold.py closed DP4 for tau_drift: measure a real
distribution instead of guessing a constant.

Method: load each configured model, run it over (a prefix of) the dataset's
stream, average energy over windows of --window-size steps (matching
monitor_interval — the same batching _monitor_batch() uses at runtime), then
recommend:
  - E_M = the --e-M-percentile (default p99.5) of the MOST EXPENSIVE model's
    batch-energy distribution — the ceiling should sit above that model's normal
    operating range, not just the cheap models'.
  - E_m = the p1 of the CHEAPEST model's distribution, excluding exact-zero
    readings (a RAPL counter-resolution artifact at these batch sizes, not a real
    energy floor — see core/energy.py's own "counter resolution low" logging).

CV datasets are out of scope for this pass (a different adapter/batching shape;
tracked as follow-up work, not built here).

Usage:
    cd tool/
    python3 experiments/calibrate_energy_bounds.py --dataset pems
    python3 experiments/calibrate_energy_bounds.py --dataset uci_electricity \\
        --max-steps 8000 --e-M-percentile 99
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

_DEFAULT_WINDOW_SIZE = 50          # matches the harness's monitor_interval default
_DEFAULT_MAX_STEPS = 5000          # cap per-model measurement pass; full stream is much larger
_DEFAULT_E_M_PERCENTILE = 99.5     # mirrors calibrate_drift_threshold.py's percentile approach,
                                    # with extra headroom vs its p99 default — this energy signal
                                    # is noisier than the drift signal that script calibrates.


def _measure_model_energy(
    model_name: str,
    predict_fn,
    adapter,
    scaler,
    dataset_config: dict,
    window_size: int,
    max_steps: int,
) -> np.ndarray:
    """Run predict_fn over a prefix of the stream; return one avg-energy-per-window
    (µJ) per window — exactly mirroring _monitor_batch()'s avg_energy_uJ computation
    so the calibrated bounds are measured the same way they'll be applied.
    """
    from core.energy import EnergyMeter

    backend = dataset_config.get("energy_meter", "auto")
    energies: list[float] = []
    window: list[float] = []
    n = 0
    for sample in adapter.stream():
        if n >= max_steps:
            break
        scaled_inputs = scaler.transform(sample.inputs.reshape(-1, 1)).flatten()
        with EnergyMeter(f"calib_{model_name}_{n}", backend=backend) as em:
            try:
                predict_fn(scaled_inputs)
            except Exception as exc:
                logger.warning("Model '%s' prediction failed at step %d: %s", model_name, n, exc)
        window.append(em.total_uJ if em.total_uJ is not None else 0.0)
        n += 1
        if len(window) >= window_size:
            energies.append(float(np.mean(window)))
            window = []
    if window:
        energies.append(float(np.mean(window)))
    return np.array(energies)


def calibrate(
    dataset_name: str,
    window_size: int,
    max_steps: int,
    e_M_percentile: float,
    tool_dir: Path,
) -> dict:
    from experiments.run_experiment import _build_adapter, _fit_scaler, _load_models

    config_path = tool_dir / "configs" / "datasets" / f"{dataset_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    with open(config_path) as f:
        dataset_config = json.load(f)

    if dataset_config.get("domain") == "cv":
        raise NotImplementedError(
            "calibrate_energy_bounds.py currently supports regression datasets only "
            "(CV energy calibration needs a different adapter/batching shape — "
            "out of scope for this pass)."
        )

    configs_dir = tool_dir / "configs" / "datasets"
    adapter = _build_adapter(dataset_config, configs_dir)
    train_values = np.asarray(adapter.train_split(), dtype=float)
    scaler = _fit_scaler(train_values)
    models = _load_models(dataset_config, configs_dir)

    report: dict = {
        "dataset": dataset_name,
        "window_size": window_size,
        "max_steps": max_steps,
        "e_M_percentile": e_M_percentile,
        "models": {},
    }

    print(f"\n{'Model':<10}{'N windows':>12}{'Min':>12}{'P50':>12}{'Mean':>12}"
          f"{'P95':>12}{'P99':>12}{'P99.5':>12}{'Max':>12}")
    print("-" * 118)

    all_energies: dict[str, np.ndarray] = {}
    for name, predict_fn in models.items():
        if predict_fn is None:
            logger.warning("Model '%s' has no loadable weights — skipping. "
                            "Train/retrain it first, then re-run this script.", name)
            continue
        energies = _measure_model_energy(
            name, predict_fn, adapter, scaler, dataset_config, window_size, max_steps
        )
        if energies.size == 0:
            logger.warning("Model '%s': no windows measured — skipping.", name)
            continue
        all_energies[name] = energies
        entry = {
            "n_windows": int(energies.size),
            "min": round(float(energies.min()), 2),
            "p50": round(float(np.percentile(energies, 50)), 2),
            "mean": round(float(energies.mean()), 2),
            "p95": round(float(np.percentile(energies, 95)), 2),
            "p99": round(float(np.percentile(energies, 99)), 2),
            "p99.5": round(float(np.percentile(energies, 99.5)), 2),
            "max": round(float(energies.max()), 2),
        }
        report["models"][name] = entry
        print(f"{name:<10}{entry['n_windows']:>12d}{entry['min']:>12.1f}{entry['p50']:>12.1f}"
              f"{entry['mean']:>12.1f}{entry['p95']:>12.1f}{entry['p99']:>12.1f}"
              f"{entry['p99.5']:>12.1f}{entry['max']:>12.1f}")

    if not all_energies:
        raise RuntimeError(
            "No model produced measurable energy — check that weights_path files exist "
            "(train the models first) and that the energy backend is available."
        )

    most_expensive = max(all_energies, key=lambda m: float(all_energies[m].mean()))
    recommended_E_M = float(np.percentile(all_energies[most_expensive], e_M_percentile))

    cheapest = min(all_energies, key=lambda m: float(all_energies[m].mean()))
    cheap_nonzero = all_energies[cheapest][all_energies[cheapest] > 0]
    recommended_E_m = float(np.percentile(cheap_nonzero, 1)) if cheap_nonzero.size else 0.0

    report["recommended_E_m"] = round(recommended_E_m, 2)
    report["recommended_E_M"] = round(recommended_E_M, 2)
    report["most_expensive_model"] = most_expensive
    report["cheapest_model"] = cheapest

    print(f"\nRecommended E_m (p1 of '{cheapest}', nonzero readings only): "
          f"{report['recommended_E_m']:.0f}")
    print(f"Recommended E_M (p{e_M_percentile} of '{most_expensive}'): "
          f"{report['recommended_E_M']:.0f}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate per-dataset E_m/E_M energy-normalisation bounds from real "
                     "measured inference energy (regression datasets only)."
    )
    parser.add_argument("--dataset", required=True, help="Dataset config name (no .json).")
    parser.add_argument("--window-size", type=int, default=_DEFAULT_WINDOW_SIZE,
                         help=f"Batch size energy is averaged over, matching monitor_interval "
                              f"(default {_DEFAULT_WINDOW_SIZE}).")
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS,
                         help=f"Cap on stream steps measured per model (default "
                              f"{_DEFAULT_MAX_STEPS}). Pass a larger value to cover more of the "
                              "stream; the full stream is typically much larger than this.")
    parser.add_argument("--e-M-percentile", type=float, default=_DEFAULT_E_M_PERCENTILE,
                         help=f"Percentile of the most expensive model's distribution used as "
                              f"the recommended E_M (default {_DEFAULT_E_M_PERCENTILE}).")
    parser.add_argument("--output", default=None,
                         help="Output JSON path (default: runs/energy_calibration_<dataset>.json).")
    args = parser.parse_args()

    report = calibrate(
        args.dataset, args.window_size, args.max_steps, args.e_M_percentile, _TOOL_DIR
    )

    out_path = (
        Path(args.output) if args.output
        else _TOOL_DIR / "runs" / f"energy_calibration_{args.dataset}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved calibration report to {out_path}")
    print("Copy 'recommended_E_m'/'recommended_E_M' into the dataset config's E_m/E_M keys "
          "(and update energy_bounds_source) — this script never writes config files for you.")


if __name__ == "__main__":
    main()

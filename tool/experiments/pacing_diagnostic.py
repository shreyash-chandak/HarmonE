"""
experiments/pacing_diagnostic.py — effect of pacing (a sleep between
predictions, the original's commented-out time.sleep(0.15) in inference.py)
on the per-prediction energy reading and inference time (2026-10-13).

For each dataset, model and delay, run --n predictions exactly like the
harness's per-prediction metering: a Pkg0Meter (pyRAPL pkg[0] equivalent)
around loading the weights from disk + one prediction, then sleep `delay`
outside the meter. With 150 ms the CPU idles between predictions (C-states,
lower clocks); without it the CPU stays busy. A full 5-seed grid at 150 ms
would take weeks, so this picks the shortest delay whose readings match 150 ms.

Weights that are missing or stale for the current configs are trained first
(same check as the harness). Single torch thread, as the inference process.

Output: <out>/pacing_diagnostic.csv (one row per dataset/model/delay).

Usage (from tool/):
    python3 experiments/pacing_diagnostic.py --out diagnostics/<folder>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

DATASETS = ["pems_driftinduced", "spot_prices_driftinduced", "uci_electricity_driftinduced"]
DELAYS = [0.0, 0.001, 0.005, 0.02, 0.15]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--delays", default=",".join(str(d) for d in DELAYS))
    p.add_argument("--n", type=int, default=1000, help="Predictions per dataset/model/delay.")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)  # as the inference process
    from adapters.loaders import get_loader
    from core.energy import Pkg0Meter
    from experiments.run_experiment import (_build_adapter, _fit_scaler, _stale_regression_models,
                                            _train_regression_models, _abs_weights_path)

    delays = [float(d) for d in args.delays.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meter = Pkg0Meter()
    rows = []

    for ds in args.datasets:
        cfg = json.loads((_TOOL_DIR / "configs" / "datasets" / f"{ds}.json").read_text())
        L = int(cfg.get("seq_length", 5))
        ad = _build_adapter(cfg, _TOOL_DIR / "configs")
        train = np.asarray(ad.train_split(), dtype=float)
        scaler = _fit_scaler(train)
        stale = _stale_regression_models(cfg, train)
        if stale:
            print(f"{ds}: training stale weights {stale}", flush=True)
            _train_regression_models(cfg, train, scaler, only=stale)
        samples = [s for _, s in zip(range(args.n), ad.stream())]
        X = scaler.transform(np.array([s.inputs for s in samples], dtype=float).reshape(-1, 1)).reshape(len(samples), L)

        for model, spec in cfg["models"].items():
            loader = get_loader(spec["loader"])
            path = _abs_weights_path(spec)
            for delay in delays:
                readings, times = [], []
                for x in X:
                    meter.begin()
                    t0 = time.perf_counter()
                    loader(path, seq_length=L, num_classes=1)(x)
                    times.append(time.perf_counter() - t0)
                    r = meter.end()
                    if r is not None:
                        readings.append(r)
                    if delay > 0:
                        time.sleep(delay)
                e = np.array(readings) / 1000.0
                row = {
                    "dataset": ds, "model": model, "delay_s": delay, "n": len(X),
                    "mJ_mean": round(float(e.mean()), 4) if len(e) else None,
                    "mJ_median": round(float(np.median(e)), 4) if len(e) else None,
                    "zero_readings_pct": round(100.0 * float((e == 0).mean()), 1) if len(e) else None,
                    "ms_mean": round(1000.0 * float(np.mean(times)), 4),
                    "ms_median": round(1000.0 * float(np.median(times)), 4),
                }
                rows.append(row)
                print(f"{ds:30s} {model:5s} delay {delay:6.3f}s  mJ mean {row['mJ_mean']}  median {row['mJ_median']}"
                      f"  zero {row['zero_readings_pct']}%  ms mean {row['ms_mean']}  median {row['ms_median']}", flush=True)
                time.sleep(5)  # settle between settings

    with open(out / "pacing_diagnostic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'pacing_diagnostic.csv'}")


if __name__ == "__main__":
    main()

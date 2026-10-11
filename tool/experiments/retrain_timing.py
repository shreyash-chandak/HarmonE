"""
experiments/retrain_timing.py — wall time of one retrain per dataset/model
with the harness's own code (_load_model_store + _do_inline_retrain, current
configs, a 1,200-row window at stream row --at), under a given number of
torch intra-op threads (2026-10-12, stage 1 diagnostics).

Retrains now run in a background thread (item 9a) and further adaptations are
skipped while one is in flight, so a slow retrain changes results. In WSL the
5-step LSTM fine-tune took 25.8 s with torch's default threads and 0.04 s with
one; this measures it on the experiment machine.

Usage (from tool/; torch threads are per process, so one call per setting):
    python3 experiments/retrain_timing.py --out diagnostics/<folder> --threads default
    python3 experiments/retrain_timing.py --out diagnostics/<folder> --threads 1
Appends to <out>/retrain_timing.csv.
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--threads", default="default", help='"default" or an integer')
    p.add_argument("--at", type=int, default=6400, help="Stream row the retrain happens at.")
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    import torch
    if args.threads != "default":
        torch.set_num_threads(int(args.threads))
    from experiments.run_experiment import _build_adapter, _fit_scaler, _load_model_store, _do_inline_retrain

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "retrain_timing.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "torch_threads", "repeat", "load_s", "retrain_s", "ok"])
        if new:
            w.writeheader()
        for ds in DATASETS:
            cfg = json.loads((_TOOL_DIR / "configs" / "datasets" / f"{ds}.json").read_text())
            L = int(cfg.get("seq_length", 5))
            ad = _build_adapter(cfg, _TOOL_DIR / "configs")
            train = np.asarray(ad.train_split(), dtype=float)
            scaler = _fit_scaler(train)
            cfg["_n_train_sequences"] = max(len(train) - L, 0)
            hist = [s.ground_truth for _, s in zip(range(args.at), ad.stream())]
            for model in cfg.get("models", {}):
                for rep in range(1, args.repeats + 1):
                    t0 = time.perf_counter()
                    store = _load_model_store(cfg, only=[model])
                    t1 = time.perf_counter()
                    ok = _do_inline_retrain(model, store, {}, hist, scaler, seq_length=L,
                                            drift_window=int(cfg.get("drift_window_size", 1200)))
                    t2 = time.perf_counter()
                    w.writerow({"dataset": ds, "model": model, "torch_threads": torch.get_num_threads(),
                                "repeat": rep, "load_s": round(t1 - t0, 4), "retrain_s": round(t2 - t1, 4), "ok": ok})
                    print(f"{ds:30s} {model:5s} threads={torch.get_num_threads():3d} rep {rep}: "
                          f"load {t1 - t0:.3f}s retrain {t2 - t1:.3f}s ok={ok}", flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

"""
experiments/svr_c_sweep.py — SVR C sweep for the 1,200-row initial training
(2026-10-12, stage 1 diagnostics, item 4a).

For each dataset and each C, train the SVR exactly like the harness's initial
training with hyperparams.train.train_window_rows = 1200 (most recent 1,200
training rows, the config's kernel/tol/epsilon) and record:
  - stream R² and window R² (mean per-batch R², the planners' signal) of the
    static model over the whole *_driftinduced stream,
  - mean mJ per prediction with per-prediction reload, measured like the
    harness: a Pkg0Meter (pyRAPL pkg[0] equivalent) around loading the pickle
    from disk + one prediction, over --n-energy evenly spaced stream rows,
  - support-vector count and pickle size.
Ridge and LSTM (the current initial weights) are measured the same way as
reference rows, so "SVR between ridge and LSTM" can be judged on one method.
Energy here is standalone (no manager process running), so absolute mJ are
lower than in harness runs; compare rows with each other.

Output: <out>/svr_c_sweep.csv; trained pickles under <out>/svr_models/.

Usage (from tool/):
    python3 experiments/svr_c_sweep.py --out diagnostics/<folder>
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

DATASETS = ["pems_driftinduced", "spot_prices_driftinduced", "uci_electricity_driftinduced"]
C_MULTIPLIERS = [0.3, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--n-energy", type=int, default=3000)
    p.add_argument("--window-rows", type=int, default=1200)
    p.add_argument("--c-values", default=None,
                   help="Comma-separated C values (default: config C x 0.3 .. x300).")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)  # as the inference process
    from sklearn.svm import SVR
    from sklearn.metrics import r2_score
    from adapters.loaders import get_loader
    from core.energy import Pkg0Meter
    from core.scoring import window_r2_mean
    from experiments.run_experiment import _build_adapter, _fit_scaler

    out = Path(args.out)
    (out / "svr_models").mkdir(parents=True, exist_ok=True)
    meter = Pkg0Meter()
    rows = []

    for ds in args.datasets:
        cfg = json.loads((_TOOL_DIR / "configs" / "datasets" / f"{ds}.json").read_text())
        L = int(cfg.get("seq_length", 5))
        interval = int(cfg.get("monitor_interval", 50))
        window = int(cfg.get("drift_window_size", 1200))
        ad = _build_adapter(cfg, _TOOL_DIR / "configs")
        train = np.asarray(ad.train_split(), dtype=float)
        scaler = _fit_scaler(train)
        s_train = scaler.transform(train.reshape(-1, 1)).ravel()
        X_tr = np.lib.stride_tricks.sliding_window_view(s_train[:-1], L)
        y_tr = s_train[L:]
        n_seq = max(1, args.window_rows - L)
        X_fit, y_fit = X_tr[-n_seq:], y_tr[-n_seq:]

        samples = list(ad.stream())
        X_st = scaler.transform(np.array([s.inputs for s in samples], dtype=float).reshape(-1, 1)).reshape(len(samples), L)
        y_st = np.array([s.ground_truth for s in samples], dtype=float)
        inv = lambda z: scaler.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()
        e_idx = np.linspace(0, len(samples) - 1, min(args.n_energy, len(samples))).astype(int)

        def energy_per_prediction(loader_ref: str, path: str) -> tuple[float | None, float]:
            loader = get_loader(loader_ref)
            readings, times = [], []
            for i in e_idx:
                x = X_st[i]
                meter.begin()
                t0 = time.time()
                fn = loader(path, seq_length=L, num_classes=1)
                fn(x)
                times.append(time.time() - t0)
                r = meter.end()
                if r is not None:
                    readings.append(r)
            mj = float(np.mean(readings)) / 1000.0 if readings else None
            return mj, float(np.mean(times)) * 1000.0

        def add_row(model: str, C, n_sv, size, pred, loader_ref, path):
            mj, ms = energy_per_prediction(loader_ref, path)
            rows.append({
                "dataset": ds, "model": model, "C": C, "n_support": n_sv, "pickle_bytes": size,
                "stream_r2": round(float(r2_score(y_st, pred)), 6),
                "window_r2": round(window_r2_mean(y_st, pred, interval, window), 6),
                "mJ_per_prediction": None if mj is None else round(mj, 4),
                "load_predict_ms": round(ms, 4), "n_energy": len(e_idx),
            })
            r = rows[-1]
            print(f"{ds:30s} {model:6s} C={C!s:10s} SV={n_sv!s:6s} {size/1024 if size else 0:8.0f} KB  "
                  f"R2 {r['stream_r2']:+.4f} wR2 {r['window_r2']:.4f}  mJ {r['mJ_per_prediction']}  {ms:.3f} ms",
                  flush=True)

        # reference rows: current ridge and LSTM initial weights
        for ref in ("ridge", "lstm"):
            spec = cfg["models"][ref]
            wp = str(_TOOL_DIR / spec["weights_path"])
            if ref == "ridge":
                m = pickle.load(open(wp, "rb"))
                pred = inv(m.predict(X_st))
            else:
                from adapters.loaders import LSTMModel
                net = LSTMModel()
                net.load_state_dict(torch.load(wp, map_location="cpu", weights_only=False))
                net.eval()
                with torch.no_grad():
                    pred = inv(torch.cat([net(torch.tensor(X_st[i:i + 4096], dtype=torch.float32).unsqueeze(-1))
                                          for i in range(0, len(X_st), 4096)]).numpy().ravel())
            add_row(ref, "", "", Path(wp).stat().st_size, pred, spec["loader"], wp)

        hp = cfg["models"]["svr"]["hyperparams"]["train"]
        c_values = ([float(c) for c in args.c_values.split(",")] if args.c_values
                    else [float(hp.get("C", 0.08)) * k for k in C_MULTIPLIERS])
        for C in c_values:
            m = SVR(kernel=hp.get("kernel", "linear"), C=C, tol=float(hp.get("tol", 0.16)),
                    epsilon=float(hp.get("epsilon", 0.1)))
            m.fit(X_fit, y_fit)
            path = out / "svr_models" / f"{ds}_C{C:.6g}.pkl"
            with open(path, "wb") as f:
                pickle.dump(m, f)
            add_row("svr", f"{C:.6g}", int(len(m.support_)), path.stat().st_size,
                    inv(m.predict(X_st)), cfg["models"]["svr"]["loader"], str(path))

    with open(out / "svr_c_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'svr_c_sweep.csv'}")


if __name__ == "__main__":
    main()

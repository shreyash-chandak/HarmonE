"""
experiments/prt_gain_sweep.py — offline replay of a naive_prt run for a
sklearn model, to pick the retrain regularisation factor k that keeps the
periodic-retrain gain under a cap (2026-10-26).

Replays the harness exactly where it matters for R²: same adapter stream,
scaler and training size, the same retrain code (_load_model_store +
_do_inline_retrain, sample-scaled regularisation with factor k), PRT at the
first 260-row cycle past each 3,200-row multiple (core/planners/base.py
prt_due), new weights serving 2 cycles after the trigger (background retrain
applied at the next cycle while inference is one batch ahead; matches the
"applied at step" lines of the real runs). Prints stream R² per k.

Usage (from tool/):
    python3 experiments/prt_gain_sweep.py --dataset spot_prices_driftinduced --model ridge --k 1,10,30,60,100
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, choices=["ridge", "svr"])
    p.add_argument("--k", default="1", help="Comma-separated regularisation factors.")
    p.add_argument("--apply-delay", type=int, default=None, help="Rows from trigger to new weights (default 2 cycles).")
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    from experiments.run_experiment import _build_adapter, _fit_scaler, _load_model_store, _do_inline_retrain

    cfg = json.loads((_TOOL_DIR / "configs" / "datasets" / f"{args.dataset}.json").read_text(encoding="utf-8"))
    L = int(cfg.get("seq_length", 5))
    mi = int(cfg.get("monitor_interval", 50))
    prt = int(cfg.get("prt_interval", 3200))
    delay = args.apply_delay if args.apply_delay is not None else 2 * mi
    ad = _build_adapter(dict(cfg, stream_delay_s=0.0), _TOOL_DIR / "configs")
    train = np.asarray(ad.train_split(), dtype=float)
    scaler = _fit_scaler(train)
    cfg["_n_train_sequences"] = max(len(train) - L, 0)
    samples = list(ad.stream())
    n = len(samples)
    X = scaler.transform(np.array([s.inputs for s in samples], dtype=float).reshape(-1, 1)).reshape(n, L)
    y = np.array([s.ground_truth for s in samples], dtype=float)
    sched = [s for s in range(mi, n + 1, mi) if s // prt > (s - mi) // prt]
    window = int(cfg.get("drift_window_size", 1200))

    def r2(p_):
        return 1.0 - float(((y - p_) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())

    print(f"{args.dataset} {args.model}: {n} rows, {len(sched)} retrains at {sched[:3]}..., apply delay {delay} rows")
    for k in [float(v) for v in args.k.split(",")]:
        c = json.loads(json.dumps(cfg))
        c["models"][args.model].setdefault("hyperparams", {}).setdefault("retrain", {})["regularisation_factor"] = k
        store = _load_model_store(c, only=[args.model])
        info = store[args.model]
        preds = np.empty(n)
        bounds = [0] + [min(s + delay, n) for s in sched] + [n]
        static = None
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            if b > a:
                preds[a:b] = scaler.inverse_transform(info["model"].predict(X[a:b]).reshape(-1, 1)).ravel()
            if i == 0 and static is None:
                static = scaler.inverse_transform(info["model"].predict(X).reshape(-1, 1)).ravel()
            if i < len(sched):
                s = sched[i]
                _do_inline_retrain(args.model, store, {}, list(y[:s]), scaler, seq_length=L, drift_window=window)
        print(f"  k={k:g}: stream R² {r2(preds):.4f} (static {r2(static):.4f}, gain {r2(preds) - r2(static):+.4f})"
              f"  final params {({kk: v for kk, v in info['model'].get_params().items() if kk in ('alpha', 'C')})}",
              flush=True)


if __name__ == "__main__":
    main()

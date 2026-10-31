"""
scripts/aggregate_seeds.py — multi-seed regression results tables (markdown).

Reads runs_seed*/<dataset>_<planner>[_<model>]_conc/run_manifest.json (+
predictions.csv for model shares, and for window R² on runs made before the
manifest carried it) and prints, per dataset, mean ± sd over seeds of:
stream R², window R² (mean per-batch R², the planners' accuracy signal),
energy (all metered energy per prediction: inference + MAPE-K + retrains, net
of overlaps, and the per-prediction readings alone), inference time,
switches, retrains, VMR restores, skipped adaptations and model shares. Then
the trade-off against the ridge-LSTM line, retraining gains, SVR model sizes
(initial vs after retraining), per-seed greedy energy on UCI and S5's LSTM
share on PeMS.

Usage (from tool/):
    python3 scripts/aggregate_seeds.py --runs runs_seed1 runs_seed2 runs_seed3 runs_seed4 runs_seed5
(single-model baselines run inside every seed since 2026-10-31, so they get mean ±
sd too. Leave runs_calib out: it only calibrates, and would add a sample.)
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))
from core.scoring import window_r2_mean  # noqa: E402

DATASETS = ["pems_driftinduced", "spot_prices_driftinduced", "uci_electricity_driftinduced"]
SHORT = {"pems_driftinduced": "PeMS", "spot_prices_driftinduced": "Spot", "uci_electricity_driftinduced": "UCI"}
ORDER = ["naive_ridge", "naive_svr", "naive_lstm", "naive_prt_ridge", "naive_prt_svr", "naive_prt_lstm",
         "switch", "switch_prt", "random_switch", "random_switch_prt", "greedy_switch", "harmone_original",
         "violation_aware", "violation_aware_gated", "pareto", "bandit"]
MODELS = ("ridge", "svr", "lstm")


def load(run_dirs: list[Path]) -> dict:
    rows: dict = collections.defaultdict(list)
    for rd in run_dirs:
        for mpath in sorted(rd.glob("*_conc/run_manifest.json")):
            j = json.loads(mpath.read_text())
            ds = j["dataset"]
            if ds not in DATASETS:
                continue
            key = j["run_id"][len(ds) + 1:-len("_conc")]
            with open(mpath.parent / "predictions.csv", newline="") as f:
                pr = list(csv.DictReader(f))
            n = len(pr) or 1
            tm = j.get("task_metrics", {})
            wr2 = tm.get("window_r2")
            if wr2 is None:
                thr = json.loads((mpath.parent / "thresholds.json").read_text())
                wr2 = window_r2_mean([float(r["y_true"]) for r in pr], [float(r["y_pred"]) for r in pr],
                                     int(thr.get("monitor_interval", 50)), int(thr.get("drift_window_size", 1200)))
            ec = j.get("event_counters", {})
            share = collections.Counter(r["active_model"] for r in pr)
            rows[(ds, key)].append({
                "seed": rd.name, "status": j.get("status"),
                "r2": tm.get("r2"), "wr2": wr2,
                "e": j.get("energy_mJ"), "it": j.get("inference_time_ms"),
                "eg": j.get("energy_gross_mJ", j.get("energy_mJ")),
                # per-prediction readings only (manifests before 2026-10-13: energy_mJ was this)
                "ei": j.get("inference_energy_mJ", j.get("energy_mJ")),
                "sw": ec.get("model_switches", 0), "rt": ec.get("retrains", 0),
                "vmr": ec.get("vmr_events", 0), "busy": ec.get("adaptation_skipped_busy", 0),
                "share": {m: share.get(m, 0) / n for m in MODELS},
                "files": j.get("model_files", {}),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # R², Δ on a Windows console
    rows = load([Path(r) for r in args.runs])

    def mean(ds, k, key):
        v = [r[key] for r in rows[(ds, k)] if r[key] is not None]
        return float(np.mean(v)) if v else float("nan")

    def sd(ds, k, key):
        v = [r[key] for r in rows[(ds, k)] if r[key] is not None]
        return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

    status = collections.Counter(r["status"] for v in rows.values() for r in v)
    print(f"runs: {sum(status.values())}  status: {dict(status)}")

    for ds in DATASETS:
        keys = [k for k in ORDER if (ds, k) in rows]
        if not keys:
            continue
        n_seeds = max(len(rows[(ds, k)]) for k in keys)
        print(f"\n### {SHORT[ds]} (mean ± sd over up to {n_seeds} seeds; energy net of overlaps)\n")
        print("| Planner | Runs | Stream R² | Window R² | mJ/pred | gross mJ/pred | inference mJ/pred | inf ms | Switches | Retrains | VMR | Skipped (busy) | ridge/svr/lstm % |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k in keys:
            sh = {m: np.mean([r["share"][m] for r in rows[(ds, k)]]) * 100 for m in MODELS}
            print(f"| {k} | {len(rows[(ds, k)])} | {mean(ds,k,'r2'):.4f} ± {sd(ds,k,'r2'):.4f} | {mean(ds,k,'wr2'):.4f} ± {sd(ds,k,'wr2'):.4f} "
                  f"| {mean(ds,k,'e'):.2f} ± {sd(ds,k,'e'):.2f} | {mean(ds,k,'eg'):.2f} | {mean(ds,k,'ei'):.2f} | {mean(ds,k,'it'):.3f} | {mean(ds,k,'sw'):.0f} "
                  f"| {mean(ds,k,'rt'):.0f} | {mean(ds,k,'vmr'):.0f} | {mean(ds,k,'busy'):.0f} "
                  f"| {sh['ridge']:.0f}/{sh['svr']:.0f}/{sh['lstm']:.0f} |")

    print("\n### Trade-off vs the ridge–LSTM line (stream R², mean mJ/pred, net)\n")
    print("| Planner | " + " | ".join(f"{SHORT[d]} R² @ mJ | vs line" for d in DATASETS) + " |")
    print("|---|" + "---|---|" * len(DATASETS))
    for k in ORDER:
        cells = []
        for ds in DATASETS:
            if (ds, k) not in rows or (ds, "naive_ridge") not in rows or (ds, "naive_lstm") not in rows:
                cells.append(" | ")
                continue
            r, e = mean(ds, k, "r2"), mean(ds, k, "e")
            r0, e0 = mean(ds, "naive_ridge", "r2"), mean(ds, "naive_ridge", "e")
            r1, e1 = mean(ds, "naive_lstm", "r2"), mean(ds, "naive_lstm", "e")
            line = r0 + (r1 - r0) * (e - e0) / (e1 - e0) if e1 != e0 else float("nan")
            cells.append(f"{r:.4f} @ {e:.1f} | {r - line:+.4f}")
        print(f"| {k} | " + " | ".join(cells) + " |")

    print("\n### Pool position and retraining (naive vs naive_prt)\n")
    print("| Dataset | Model | naive R² / window R² @ mJ | prt R² / window R² @ mJ | Δ stream R² | Δ window R² |")
    print("|---|---|---|---|---|---|")
    for ds in DATASETS:
        for m in MODELS:
            a, b = f"naive_{m}", f"naive_prt_{m}"
            if (ds, a) not in rows or (ds, b) not in rows:
                continue
            print(f"| {SHORT[ds]} | {m} | {mean(ds,a,'r2'):.4f} / {mean(ds,a,'wr2'):.4f} @ {mean(ds,a,'e'):.1f} "
                  f"| {mean(ds,b,'r2'):.4f} / {mean(ds,b,'wr2'):.4f} @ {mean(ds,b,'e'):.1f} "
                  f"| {mean(ds,b,'r2') - mean(ds,a,'r2'):+.4f} | {mean(ds,b,'wr2') - mean(ds,a,'wr2'):+.4f} |")

    print("\n### SVR model files (support vectors / KB): initial vs after the run's last retrain\n")
    print("| Dataset | initial (naive_svr) | final (naive_prt_svr) |")
    print("|---|---|---|")
    for ds in DATASETS:
        a = (rows.get((ds, "naive_svr")) or [{}])[0].get("files", {}).get("svr", {})
        b = (rows.get((ds, "naive_prt_svr")) or [{}])[0].get("files", {}).get("svr", {})
        fmt = lambda e, lab: (f"{e.get(f'{lab}_n_support')} SV / {e[f'{lab}_bytes'] / 1024:.0f} KB"
                              if e.get(f"{lab}_bytes") else "n/a (manifest predates model_files)")
        print(f"| {SHORT[ds]} | {fmt(a, 'initial')} | {fmt(b, 'final')} |")

    ds = "uci_electricity_driftinduced"
    if (ds, "greedy_switch") in rows:
        print("\n### greedy_switch on UCI, per seed\n")
        print("| Seed | Stream R² | mJ/pred | ridge/svr/lstm % | Switches |")
        print("|---|---|---|---|---|")
        for r in rows[(ds, "greedy_switch")]:
            print(f"| {r['seed']} | {r['r2']:.4f} | {r['e']:.2f} | "
                  f"{r['share']['ridge']*100:.0f}/{r['share']['svr']*100:.0f}/{r['share']['lstm']*100:.0f} | {r['sw']} |")
    ds = "pems_driftinduced"
    if (ds, "violation_aware") in rows:
        v = [r["share"]["lstm"] * 100 for r in rows[(ds, "violation_aware")]]
        print(f"\nS5 (violation_aware) LSTM share on PeMS: {np.mean(v):.1f} % "
              f"(per seed: {', '.join(f'{x:.0f}' for x in v)})")


if __name__ == "__main__":
    main()

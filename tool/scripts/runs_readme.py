"""
scripts/runs_readme.py — per-dataset results tables for one machine's runs
folder, written to <root>/README.md (2026-10-25).

Expects <root>/runs_calib (single-model baselines, run once) and
<root>/runs_seed<N> (adaptive planners). Per planner and dataset:
accuracy (stream R² x 100) min / max / mean over seeds, total energy per
prediction (energy_mJ: inference + MAPE-K + retrains) min / max / mean,
mean inference energy per prediction (inference_energy_mJ), and for the
adaptive planners accuracy / total energy (mean over seeds of the per-run
ratio).

Usage (from tool/):
    python3 scripts/runs_readme.py --root runs_thinkpad_2010 --machine "ThinkPad (Core Ultra 5 125U)"
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

DATASETS = [("pems_driftinduced", "PeMS"), ("spot_prices_driftinduced", "Spot prices"),
            ("uci_electricity_driftinduced", "UCI electricity")]
BASELINES = ["naive_ridge", "naive_svr", "naive_lstm", "naive_prt_ridge", "naive_prt_svr", "naive_prt_lstm"]
ADAPTIVE = ["switch", "switch_prt", "random_switch", "random_switch_prt", "greedy_switch", "harmone_original",
            "violation_aware", "violation_aware_gated", "pareto", "bandit"]
LABEL = {"naive_ridge": "Ridge (naive)", "naive_svr": "SVR (naive)", "naive_lstm": "LSTM (naive)",
         "naive_prt_ridge": "Ridge + PRT", "naive_prt_svr": "SVR + PRT", "naive_prt_lstm": "LSTM + PRT",
         "switch": "Switch", "switch_prt": "Switch + PRT", "random_switch": "Random switch", "random_switch_prt": "Random switch + PRT",
         "greedy_switch": "Greedy switch", "harmone_original": "HarmonE (original)",
         "violation_aware": "Violation-aware (S5)", "violation_aware_gated": "S5a gated",
         "pareto": "Pareto", "bandit": "Bandit"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--machine", required=True)
    args = p.parse_args()
    root = Path(args.root)
    seeds = sorted((d for d in root.glob("runs_seed*") if d.is_dir()),
                   key=lambda d: int(re.sub(r"\D", "", d.name)))
    runs = collections.defaultdict(list)   # (dataset, planner) -> [manifest]
    thr = {}
    for d in [root / "runs_calib", *seeds]:
        for mf in sorted(d.glob("*_conc/run_manifest.json")):
            j = json.loads(mf.read_text())
            name = j["planner"] + (f"_{j['pin_model']}" if j.get("pin_model") else "")
            runs[(j["dataset"], name)].append(j)
            if d.name != "runs_calib":
                thr.setdefault(j["dataset"], json.loads((mf.parent / "thresholds.json").read_text()))

    out = [f"# Regression results — {args.machine}", "",
           f"Generated {date.today()} by `scripts/runs_readme.py` from `{root.name}/`.", "",
           f"- **Seeds:** {', '.join(s.name.replace('runs_seed', '') for s in seeds)} "
           f"(adaptive planners, `runs_seed<N>/`). Single-model baselines run once "
           "(`runs_calib/`), so their min = max = mean.",
           "- **Accuracy:** stream R² × 100 over the whole drift-induced test stream.",
           "- **Total energy:** all metered energy of the run (per-prediction pkg[0] readings + MAPE-K "
           "cycles + background retrains) divided by the number of predictions, in mJ.",
           "- **Inference energy:** the per-prediction readings alone (model load + predict), mean, in mJ.",
           "- **Accuracy / energy:** accuracy (out of 100) ÷ total energy (mJ), computed per run and averaged "
           "over seeds; adaptive planners only. Higher is better.",
           "- Energy is this machine's own RAPL package reading; it is not comparable across machines.", ""]
    out.append("| Dataset | Stream rows | E_m (µJ) | E_M (µJ) | min_accuracy | min_score | Pacing |")
    out.append("|---|---|---|---|---|---|---|")
    for ds, short in DATASETS:
        t = thr.get(ds, {})
        rows = runs[(ds, "naive_ridge")][0]["total_steps"] if runs[(ds, "naive_ridge")] else "?"
        out.append(f"| {short} | {rows:,} | {t.get('E_m')} | {t.get('E_M')} | {t.get('min_accuracy')} "
                   f"| {t.get('min_score')} | {1000 * float(t.get('stream_delay_s', 0)):.1f} ms |")

    for ds, short in DATASETS:
        out += ["", f"## {short}", "",
                "| Planner | Runs | Acc. min | Acc. max | Acc. mean | Total E min (mJ) | Total E max (mJ) "
                "| Total E mean (mJ) | Inference E mean (mJ) | Acc. / total E |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for name in BASELINES + ADAPTIVE:
            js = runs.get((ds, name))
            if not js:
                continue
            acc = np.array([100 * j["task_metrics"]["r2"] for j in js])
            e = np.array([j["energy_mJ"] for j in js])
            ei = np.array([j["inference_energy_mJ"] for j in js])
            ratio = f"{np.mean(acc / e):.3f}" if name in ADAPTIVE else "—"
            out.append(f"| {LABEL[name]} | {len(js)} | {acc.min():.2f} | {acc.max():.2f} | {acc.mean():.2f} "
                       f"| {e.min():.2f} | {e.max():.2f} | {e.mean():.2f} | {ei.mean():.2f} | {ratio} |")
    (root / "README.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

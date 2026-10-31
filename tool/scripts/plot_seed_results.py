"""
scripts/plot_seed_results.py — paper-style plots for one machine's runs folder
(2026-10-26). Reads <root>/runs_calib (single-model baselines) and
<root>/runs_seed<N> (adaptive planners); values are means over seeds.

Writes to <root>/plots/:
  energy_vs_error_<dataset>.png  as the paper's Fig. 7 (left): average total
                              energy per prediction (mJ, left axis) beside MSE
                              (right axis), every planner
  usage_<dataset>.png         share of predictions served by ridge / SVR / LSTM,
                              per adaptive planner
  inference_time_<dataset>.png  mean inference time per prediction, every planner
  adaptations_<dataset>.png   switches / retrains / VMR restores per adaptive
                              planner (one panel each; their scales differ)

Usage (from tool/):
    python3 scripts/plot_seed_results.py --root runs_thinkpad_2610
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = [("pems_driftinduced", "PeMS"), ("spot_prices_driftinduced", "Spot prices"),
            ("uci_electricity_driftinduced", "UCI electricity")]
BASELINES = ["naive_ridge", "naive_svr", "naive_lstm", "naive_prt_ridge", "naive_prt_svr", "naive_prt_lstm"]
ADAPTIVE = ["switch", "switch_prt", "random_switch", "random_switch_prt", "greedy_switch", "harmone_original",
            "violation_aware", "violation_aware_gated", "pareto", "bandit"]
LABEL = {"naive_ridge": "Ridge", "naive_svr": "SVR", "naive_lstm": "LSTM",
         "naive_prt_ridge": "Ridge + PRT", "naive_prt_svr": "SVR + PRT", "naive_prt_lstm": "LSTM + PRT",
         "switch": "Switch", "switch_prt": "Switch + PRT", "random_switch": "Random switch", "random_switch_prt": "Random switch + PRT",
         "greedy_switch": "Greedy switch", "harmone_original": "HarmonE (original)",
         "violation_aware": "Violation-aware (S5)", "violation_aware_gated": "S5a gated",
         "pareto": "Pareto", "bandit": "Bandit"}
# categorical slots in fixed order (validated: dataviz palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR = dict(zip(ADAPTIVE, SERIES))
MODEL_COLOR = {"ridge": SERIES[0], "svr": SERIES[1], "lstm": SERIES[2]}
INK, MUTED, GRID, BASE = "#1f1f1e", "#6b6a64", "#e4e3de", "#8f8e87"
E_COLOR, MSE_COLOR = "#3b4cc0", "#eda100"   # paper Fig. 7: energy blue, loss orange

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def load(root: Path) -> dict:
    """(dataset, planner) -> dict of per-seed-mean metrics."""
    seeds = sorted((d for d in root.glob("runs_seed*") if d.is_dir()),
                   key=lambda d: int(re.sub(r"\D", "", d.name)))
    raw = collections.defaultdict(list)
    for d in [root / "runs_calib", *seeds]:
        for mf in sorted(d.glob("*_conc/run_manifest.json")):
            j = json.loads(mf.read_text())
            name = j["planner"] + (f"_{j['pin_model']}" if j.get("pin_model") else "")
            steps = {m: v.get("n_steps", 0) for m, v in (j.get("energy_by_model") or {}).items()}
            n = max(sum(steps.values()), 1)
            ec = j.get("event_counters", {})
            raw[(j["dataset"], name)].append({
                "acc": 100 * j["task_metrics"]["r2"], "mse": j["task_metrics"]["rmse"] ** 2, "e": j["energy_mJ"], "t": j["inference_time_ms"],
                "switches": ec.get("model_switches", 0), "retrains": ec.get("retrains", 0),
                "vmr": ec.get("vmr_events", 0),
                **{f"use_{m}": 100 * steps.get(m, 0) / n for m in MODEL_COLOR},
            })
    out = {k: {f: float(np.mean([r[f] for r in v])) for f in v[0]} | {"n": len(v)} for k, v in raw.items()}
    return out, len(seeds)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def energy_vs_error(data: dict, out: Path, ds: str, short: str, n_seeds: int) -> None:
    """As the original paper's Fig. 7 (left): per approach, average total energy
    per prediction (left axis) beside prediction error, MSE (right axis); lower
    is better on both."""
    names = [n for n in BASELINES + ADAPTIVE if (ds, n) in data]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax2 = ax.twinx()
    ax.bar(x - w / 2, [data[(ds, n)]["e"] for n in names], w, color=E_COLOR, label="Avg energy (mJ)")
    ax2.bar(x + w / 2, [data[(ds, n)]["mse"] for n in names], w, color=MSE_COLOR, label="Avg loss (MSE)")
    ax.set_ylabel("Avg energy (mJ)", color=E_COLOR)
    ax2.set_ylabel("Avg loss (MSE)", color=MSE_COLOR)
    ax.tick_params(axis="y", colors=E_COLOR)
    ax2.tick_params(axis="y", colors=MSE_COLOR)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x, [LABEL[n] for n in names], rotation=35, ha="right")
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)    # headroom for the legend
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.15)
    ax.axvline(len(BASELINES) - 0.5, color=GRID, linewidth=1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2)
    ax.set_title(f"{short}: energy vs prediction error per approach — lower is better on both axes "
                 f"(adaptive: mean over {n_seeds} seed(s))", loc="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / f"energy_vs_error_{ds}.png")


def usage(data: dict, out: Path, ds: str, short: str) -> None:
    names = [n for n in ADAPTIVE if (ds, n) in data][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1.3))
    left = np.zeros(len(names))
    for m, c in MODEL_COLOR.items():
        vals = np.array([data[(ds, n)][f"use_{m}"] for n in names])
        ax.barh(range(len(names)), vals, left=left, color=c, height=0.62, edgecolor="white", linewidth=2,
                label=m.upper() if m == "lstm" else ("SVR" if m == "svr" else "Ridge"))
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 8:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7.5, color="white",
                        fontweight="bold")
        left += vals
    ax.set_yticks(range(len(names)), [LABEL[n] for n in names])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of predictions served (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{short}: model usage per planner", loc="left", fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, borderaxespad=1.2)
    fig.tight_layout()
    _save(fig, out / f"usage_{ds}.png")


def inference_time(data: dict, out: Path, ds: str, short: str) -> None:
    names = [n for n in BASELINES + ADAPTIVE if (ds, n) in data][::-1]
    vals = [data[(ds, n)]["t"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(names) + 1.1))
    colors = [COLOR.get(n, BASE) for n in names]
    ax.barh(range(len(names)), vals, color=colors, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(names)), [LABEL[n] for n in names])
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_xlabel("Mean inference time per prediction (ms)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{short}: inference time per planner (grey = single-model baselines)",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / f"inference_time_{ds}.png")


def adaptations(data: dict, out: Path, ds: str, short: str) -> None:
    names = [n for n in ADAPTIVE if (ds, n) in data][::-1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.42 * len(names) + 1.4), sharey=True)
    for ax, (f, title) in zip(axes, [("switches", "Model switches"), ("retrains", "Retrains"),
                                      ("vmr", "VMR restores")]):
        vals = [data[(ds, n)][f] for n in names]
        ax.barh(range(len(names)), vals, color=[COLOR[n] for n in names], height=0.62)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:g}", va="center", fontsize=7.5, color=INK)
        ax.set_xlim(0, max(max(vals), 1) * 1.15)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(names)), [LABEL[n] for n in names])
    fig.suptitle(f"{short}: adaptations per planner (count per run, mean over seeds)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save(fig, out / f"adaptations_{ds}.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    out = root / "plots"
    out.mkdir(exist_ok=True)
    data, n_seeds = load(root)
    for ds, short in DATASETS:
        energy_vs_error(data, out, ds, short, n_seeds)
        usage(data, out, ds, short)
        inference_time(data, out, ds, short)
        adaptations(data, out, ds, short)


if __name__ == "__main__":
    main()

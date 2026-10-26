"""
scripts/plot_seed_results.py — paper-style plots for one machine's runs folder
(2026-10-26). Reads <root>/runs_calib (single-model baselines) and
<root>/runs_seed<N> (adaptive planners); values are means over seeds.

Writes to <root>/plots/:
  accuracy_vs_energy.png      stream R² x 100 vs total energy per prediction
                              (log x), one panel per dataset, every planner
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
from matplotlib.ticker import FuncFormatter

DATASETS = [("pems_driftinduced", "PeMS"), ("spot_prices_driftinduced", "Spot prices"),
            ("uci_electricity_driftinduced", "UCI electricity")]
BASELINES = ["naive_ridge", "naive_svr", "naive_lstm", "naive_prt_ridge", "naive_prt_svr", "naive_prt_lstm"]
ADAPTIVE = ["random_switch", "random_switch_prt", "greedy_switch", "harmone_original",
            "violation_aware", "violation_aware_gated", "pareto", "bandit"]
LABEL = {"naive_ridge": "Ridge", "naive_svr": "SVR", "naive_lstm": "LSTM",
         "naive_prt_ridge": "Ridge + PRT", "naive_prt_svr": "SVR + PRT", "naive_prt_lstm": "LSTM + PRT",
         "random_switch": "Random switch", "random_switch_prt": "Random switch + PRT",
         "greedy_switch": "Greedy switch", "harmone_original": "HarmonE (original)",
         "violation_aware": "Violation-aware (S5)", "violation_aware_gated": "S5a gated",
         "pareto": "Pareto", "bandit": "Bandit"}
# categorical slots in fixed order (validated: dataviz palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR = dict(zip(ADAPTIVE, SERIES))
MODEL_COLOR = {"ridge": SERIES[0], "svr": SERIES[1], "lstm": SERIES[2]}
INK, MUTED, GRID, BASE = "#1f1f1e", "#6b6a64", "#e4e3de", "#8f8e87"

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
                "acc": 100 * j["task_metrics"]["r2"], "e": j["energy_mJ"], "t": j["inference_time_ms"],
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


def _place_labels(fig, ax, items: list[tuple]) -> None:
    """Direct labels without collisions: each (x, y, text, colour) takes the
    first candidate offset whose box overlaps no point and no earlier label."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    frame = ax.get_window_extent(rend)
    pts = [ax.transData.transform((x, y)) for x, y, _, _ in items]
    taken = [(px - 5, py - 5, px + 5, py + 5) for px, py in pts]
    cands = [(6, 2), (6, -9), (-6, 2), (-6, -9), (0, 7), (0, -15), (6, 10), (6, -17), (-6, 10), (-6, -17),
             (6, 18), (6, -25), (-6, 18), (-6, -25)]
    for (x, y, text, colour), (px, py) in zip(items, pts):
        for dx, dy in cands:
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                            color=colour, ha=ha, va="bottom")
            bb = t.get_window_extent(rend)
            box = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
            inside = frame.x0 <= box[0] and box[2] <= frame.x1 and frame.y0 <= box[1] and box[3] <= frame.y1
            if inside and not any(box[0] < o[2] and o[0] < box[2] and box[1] < o[3] and o[1] < box[3] for o in taken):
                taken.append(box)
                break
            t.remove()
        else:   # nothing free: keep the first candidate, with a leader line
            ax.annotate(text, (x, y), xytext=(6, 2), textcoords="offset points", fontsize=7, color=colour)


def accuracy_vs_energy(data: dict, out: Path, n_seeds: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    labels = []
    for ax, (ds, short) in zip(axes, DATASETS):
        items = []
        for name in BASELINES:
            r = data.get((ds, name))
            if r is None:
                continue
            ax.scatter(r["e"], r["acc"], s=46, marker="s", color="white", edgecolor=BASE, linewidth=1.4, zorder=3)
            items.append((r["e"], r["acc"], LABEL[name], MUTED))
        for name in ADAPTIVE:
            r = data.get((ds, name))
            if r is None:
                continue
            ax.scatter(r["e"], r["acc"], s=64, color=COLOR[name], edgecolor="white", linewidth=1.5,
                       zorder=4, label=LABEL[name])
            items.append((r["e"], r["acc"], LABEL[name], INK))
        labels.append((ax, items))
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v in (3, 5, 7, 20, 30, 50) else ""))
        ax.set_title(short, loc="left", fontweight="bold")
        ax.set_xlabel("Total energy per prediction (mJ, log scale)")
        ax.set_ylabel("Stream R² × 100")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLOR[n], markersize=7, label=LABEL[n]) for n in ADAPTIVE]
    handles.append(plt.Line2D([], [], marker="s", ls="", color="white", markeredgecolor=BASE,
                              markeredgewidth=1.4, markersize=6, label="Single-model baseline"))
    fig.legend(handles=handles, loc="lower center", ncol=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Accuracy vs energy per planner (mean over {n_seeds} seed(s); baselines run once)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    for ax, items in labels:
        ax.margins(x=0.08, y=0.08)
        _place_labels(fig, ax, items)
    _save(fig, out / "accuracy_vs_energy.png")


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
    accuracy_vs_energy(data, out, n_seeds)
    for ds, short in DATASETS:
        usage(data, out, ds, short)
        inference_time(data, out, ds, short)
        adaptations(data, out, ds, short)


if __name__ == "__main__":
    main()

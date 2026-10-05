"""
plot_results.py — HarmonE experiment dashboard generator.

Generates per-dataset accuracy and energy bar charts (PNG + embedded HTML).
Each pin-model baseline is its own separate bar. Charts are styled for
research-paper readability (white background, clean axes).

Usage:
    python3 scripts/plot_results.py [--runs-dir runs/] [--out runs/dashboard.html]
                                   [--plots-dir runs/plots/]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style — plain white, research-paper ready
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "text.color":        "black",
    "grid.color":        "#dddddd",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Dataset / planner metadata
# ---------------------------------------------------------------------------

DATASET_ORDER = ["acdc", "bdd100k", "pems", "spot_prices", "uci_electricity"]
DATASET_LABELS = {
    "acdc":            "ACDC",
    "bdd100k":         "BDD100K",
    "pems":            "PEMS",
    "spot_prices":     "Spot Prices",
    "uci_electricity": "UCI Electricity",
}
ACCURACY_YLABEL = {
    "acdc":            "Mean Proxy Accuracy",
    "bdd100k":         "Mean Proxy Accuracy",
    "pems":            "R²",
    "spot_prices":     "R²",
    "uci_electricity": "R²",
}

# Canonical order for pin-model (S1) baselines per dataset
PIN_ORDER = {
    "pems":            ["lstm", "lstm_prt", "ridge", "ridge_prt", "svr", "svr_prt"],
    "spot_prices":     ["lstm", "lstm_prt", "ridge", "ridge_prt", "svr", "svr_prt"],
    "uci_electricity": ["lstm", "lstm_prt", "ridge", "ridge_prt", "svr", "svr_prt"],
    "acdc":            ["segformer_b0", "segformer_b1", "segformer_b2"],
    "bdd100k":         ["yolo_n", "yolo_s", "yolo_m"],
}
PIN_LABELS = {
    "lstm":          "LSTM",
    "lstm_prt":      "LSTM+PRT",
    "ridge":         "Ridge",
    "ridge_prt":     "Ridge+PRT",
    "svr":           "SVR",
    "svr_prt":       "SVR+PRT",
    "segformer_b0":  "SegF-B0",
    "segformer_b1":  "SegF-B1",
    "segformer_b2":  "SegF-B2",
    "yolo_n":        "YOLO-n",
    "yolo_s":        "YOLO-s",
    "yolo_m":        "YOLO-m",
}

# S2–S7 planners (non-pin, non-naive)
PLANNER_ORDER = [
    "random_switch",
    "random_switch_prt",
    "greedy_switch",
    "harmone_original",
    "violation_aware",
    "pareto",
    "bandit",
]
PLANNER_LABELS = {
    "random_switch":     "S2 Random",
    "random_switch_prt": "S2b Rnd-PRT",
    "greedy_switch":     "S3 Greedy",
    "harmone_original":  "S4 HarmonE",
    "violation_aware":   "S5 ViolAware",
    "pareto":            "S6 Pareto",
    "bandit":            "S7 Bandit",
}

# Color palette — colorblind-friendly (Okabe-Ito subset)
# Pin baselines: steel-blue family (light → dark)
# S2–S7 planners: warm colors to distinguish from baselines
_GRAY_6 = ["#b0b0b0", "#909090", "#cccccc", "#a0a0a0", "#d8d8d8", "#888888"]
_CV_GRAYS = ["#b0b0b0", "#707070", "#444444"]

PLANNER_COLORS = {
    "random_switch":     "#4878d0",
    "random_switch_prt": "#6fa0e0",
    "greedy_switch":     "#e87d4c",
    "harmone_original":  "#d62728",
    "violation_aware":   "#9467bd",
    "pareto":            "#2ca02c",
    "bandit":            "#17becf",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _pin_key(pin_model: str, planner: str) -> str:
    """Derive canonical pin key from pin_model + planner fields."""
    is_prt = (planner == "naive_prt")
    suffix = "_prt" if is_prt else ""
    return pin_model + suffix


def load_runs(runs_dir: Path) -> list[dict]:
    manifests: list[dict] = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir() or entry.name in ("logs", "plots"):
            continue
        mf_path = entry / "run_manifest.json"
        if not mf_path.exists():
            continue
        with open(mf_path) as f:
            mf = json.load(f)
        mf["_run_dir"] = str(entry)
        pin = mf.get("pin_model")
        if pin:
            mf["_is_pin"] = True
            mf["_pin_key"] = _pin_key(pin, mf.get("planner", "naive"))
        else:
            mf["_is_pin"] = False
            mf["_pin_key"] = None
        manifests.append(mf)
    return manifests


def load_events(runs_dir: Path, manifests: list[dict]) -> dict[str, pd.DataFrame]:
    events: dict[str, pd.DataFrame] = {}
    for mf in manifests:
        ev_path = Path(mf["_run_dir"]) / "mape_events.csv"
        if ev_path.exists():
            try:
                events[mf["run_id"]] = pd.read_csv(ev_path)
            except Exception:
                pass
    return events


def _accuracy(mf: dict) -> Optional[float]:
    tm = mf.get("task_metrics", {})
    return tm.get("mean_proxy_acc") or tm.get("r2")


def _energy_mj(mf: dict) -> Optional[float]:
    tm = mf.get("task_metrics", {})
    v = tm.get("total_inference_energy_mJ")
    if v is not None:
        return v
    return sum(
        v2.get("energy_mJ", 0) for v2 in mf.get("energy_by_model", {}).values()
    ) or None


# ---------------------------------------------------------------------------
# Row assembly: (label, accuracy, energy, color, is_pin)
# ---------------------------------------------------------------------------

def dataset_rows(manifests: list[dict], dataset: str):
    """Ordered list of (label, acc, eng_mj, color, is_pin) for one dataset."""
    ds = [m for m in manifests if m["dataset"] == dataset]
    rows = []

    # ── Pin baselines (S1 group) ──────────────────────────────────────────
    pin_order = PIN_ORDER.get(dataset, [])
    n_pins = len(pin_order)
    # generate a gray gradient across the baseline bars
    gray_vals = [int(180 - 80 * i / max(n_pins - 1, 1)) for i in range(n_pins)]

    for i, pk in enumerate(pin_order):
        match = [m for m in ds if m["_is_pin"] and m["_pin_key"] == pk]
        if not match:
            continue
        m = match[0]
        acc = _accuracy(m)
        eng = _energy_mj(m)
        if acc is None or eng is None:
            continue
        g = gray_vals[i]
        color = f"#{g:02x}{g:02x}{g:02x}"
        rows.append((PIN_LABELS.get(pk, pk), float(acc), float(eng), color, True))

    # ── S2–S7 planners ───────────────────────────────────────────────────
    for p in PLANNER_ORDER:
        match = [m for m in ds if not m["_is_pin"] and m.get("planner") == p]
        if not match:
            continue
        m = match[0]
        acc = _accuracy(m)
        eng = _energy_mj(m)
        if acc is None or eng is None:
            continue
        rows.append((PLANNER_LABELS[p], float(acc), float(eng), PLANNER_COLORS[p], False))

    return rows


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150,
                facecolor="white")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _save_png(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), format="png", bbox_inches="tight", dpi=150,
                facecolor="white")


def _bar_chart(
    labels: list[str],
    values: list[float],
    colors: list[str],
    is_pin: list[bool],
    ylabel: str,
    title: str,
    vline_after: int,
) -> plt.Figure:
    """Vertical bar chart. Draws a dashed separator after the baseline group."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.75), 4))

    x = np.arange(n)
    bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    # separator between baselines and planners
    if 0 < vline_after < n:
        ax.axvline(vline_after - 0.5, color="#888888", linewidth=0.8, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8, fontweight="normal")
    ax.set_ylim(0, max(values) * 1.20)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(values) * 0.015,
            f"{v:.3f}" if v < 10 else f"{v:.0f}",
            ha="center", va="bottom", fontsize=6.5, color="#333333",
        )

    fig.tight_layout()
    return fig


def make_charts(
    manifests: list[dict], dataset: str, plots_dir: Path
) -> tuple[str, str]:
    """Return (acc_b64, eng_b64) and save PNGs for one dataset."""
    rows = dataset_rows(manifests, dataset)
    if not rows:
        return "", ""

    labels  = [r[0] for r in rows]
    accs    = [r[1] for r in rows]
    engs    = [r[2] for r in rows]
    colors  = [r[3] for r in rows]
    is_pin  = [r[4] for r in rows]
    sep     = sum(is_pin)  # number of baseline bars

    ds_label = DATASET_LABELS[dataset]

    fig_acc = _bar_chart(
        labels, accs, colors, is_pin,
        ylabel=ACCURACY_YLABEL.get(dataset, "Accuracy"),
        title=f"{ds_label} — Accuracy by planner",
        vline_after=sep,
    )
    fig_eng = _bar_chart(
        labels, engs, colors, is_pin,
        ylabel="Total Inference Energy (mJ)",
        title=f"{ds_label} — Energy consumption by planner",
        vline_after=sep,
    )

    _save_png(fig_acc, plots_dir / f"{dataset}_accuracy.png")
    _save_png(fig_eng, plots_dir / f"{dataset}_energy.png")

    acc_b64 = _fig_to_b64(fig_acc)
    eng_b64 = _fig_to_b64(fig_eng)

    plt.close(fig_acc)
    plt.close(fig_eng)
    return acc_b64, eng_b64


# ---------------------------------------------------------------------------
# Secondary charts (EMA timeseries, utilization, switch stats)
# ---------------------------------------------------------------------------

def chart_ema_timeseries(
    events: dict[str, pd.DataFrame], manifests: list[dict]
) -> dict[str, str]:
    reg = ["pems", "spot_prices", "uci_electricity"]
    out: dict[str, str] = {}
    for ds in reg:
        runs = [m for m in manifests if m["dataset"] == ds and not m["_is_pin"]]
        if not runs:
            continue
        fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="white")
        ax.set_facecolor("white")
        plotted = False
        for m in runs:
            df = events.get(m["run_id"])
            if df is None or "ema_score" not in df.columns:
                continue
            p = m.get("planner", "?")
            color = PLANNER_COLORS.get(p, "#888888")
            ax.plot(df["step"], df["ema_score"],
                    label=PLANNER_LABELS.get(p, p),
                    color=color, linewidth=1.2, alpha=0.85)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Step")
        ax.set_ylabel("EMA Score (R²)")
        ax.set_title(f"{DATASET_LABELS[ds]} — EMA Score over Time",
                     fontweight="normal")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", ncol=2, fontsize=7)
        fig.tight_layout()
        out[ds] = _fig_to_b64(fig)
        plt.close(fig)
    return out


def chart_model_utilization(manifests: list[dict]) -> str:
    reg = ["pems", "spot_prices", "uci_electricity"]
    fig, axes = plt.subplots(1, len(reg), figsize=(13, 4), facecolor="white")
    accent = ["#4878d0", "#e87d4c", "#2ca02c", "#d62728", "#9467bd", "#17becf"]

    for ax, ds in zip(axes, reg):
        ax.set_facecolor("white")
        runs = [m for m in manifests if m["dataset"] == ds and not m["_is_pin"]]
        if not runs:
            continue
        all_models = sorted({mod for m in runs for mod in m.get("energy_by_model", {})})
        model_colors = {mod: accent[i % len(accent)] for i, mod in enumerate(all_models)}
        util: dict[str, list[float]] = defaultdict(list)
        planner_labels: list[str] = []

        for p in PLANNER_ORDER:
            match = [m for m in runs if m.get("planner") == p]
            if not match:
                continue
            m = match[0]
            total = m.get("total_steps", 1) or 1
            planner_labels.append(PLANNER_LABELS.get(p, p))
            for mod in all_models:
                util[mod].append(
                    m.get("energy_by_model", {}).get(mod, {}).get("n_steps", 0) / total
                )

        x = np.arange(len(planner_labels))
        bottom = np.zeros(len(planner_labels))
        for mod in all_models:
            vals = np.array(util[mod])
            ax.bar(x, vals, bottom=bottom, label=mod, color=model_colors[mod],
                   alpha=0.88, width=0.6, edgecolor="white", linewidth=0.5)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(planner_labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Fraction of steps")
        ax.set_ylim(0, 1.05)
        ax.set_title(DATASET_LABELS[ds], fontweight="normal")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=7)

    fig.suptitle("Model utilization by planner (fraction of steps)", fontsize=10)
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def chart_switch_stats(manifests: list[dict]) -> str:
    non_pin  = [m for m in manifests if not m["_is_pin"]]
    datasets = [d for d in DATASET_ORDER if any(m["dataset"] == d for m in non_pin)]
    planners = [p for p in PLANNER_ORDER
                if any(m.get("planner") == p for m in non_pin)]

    sw = np.full((len(planners), len(datasets)), np.nan)
    nv = np.full((len(planners), len(datasets)), np.nan)

    for i, p in enumerate(planners):
        for j, ds in enumerate(datasets):
            match = [m for m in non_pin if m.get("planner") == p and m["dataset"] == ds]
            if not match:
                continue
            m = match[0]
            cycles = m.get("mape_cycles", 1) or 1
            ec = m.get("event_counters", {})
            sw[i, j] = ec.get("model_switches", 0) / cycles
            nv[i, j] = ec.get("noop_on_violation", 0) / cycles

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="white")
    for ax, matrix, title, cmap in [
        (axes[0], sw, "Switch rate (switches / MAPE cycle)", "Blues"),
        (axes[1], nv, "Noop-on-violation rate (noops / cycle)", "Oranges"),
    ]:
        ax.set_facecolor("white")
        display = np.where(np.isnan(matrix), 0, matrix)
        im = ax.imshow(display, aspect="auto", cmap=cmap, vmin=0)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(
            [DATASET_LABELS[d] for d in datasets], rotation=30, ha="right", fontsize=7
        )
        ax.set_yticks(range(len(planners)))
        ax.set_yticklabels([PLANNER_LABELS.get(p, p) for p in planners], fontsize=8)
        ax.set_title(title, pad=6, fontsize=9, fontweight="normal")
        plt.colorbar(im, ax=ax, pad=0.02)
        for i in range(len(planners)):
            for j in range(len(datasets)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="black")
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Sans:wght@300;400;600&display=swap"
)

CSS = """
<style>
:root {
  --bg: #f5f6fa; --card: #ffffff; --border: #dde1ea;
  --accent: #2563eb; --text: #1a1f36; --muted: #6b7a99;
  --mono: monospace; --sans: 'IBM Plex Sans', sans-serif;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: var(--sans);
       font-size: 14px; line-height: 1.6; padding: 0 0 64px; }
header { background: var(--card); border-bottom: 1px solid var(--border);
         padding: 18px 32px; display: flex; align-items: baseline; gap: 16px; }
header h1 { font-size: 1.25rem; font-weight: 600; color: var(--accent); }
header .sub { font-size: 0.75rem; color: var(--muted); font-family: var(--mono); }
main { max-width: 1280px; margin: 0 auto; padding: 28px 24px 0; }
.sh { display: flex; align-items: center; gap: 10px;
      margin: 36px 0 14px; border-bottom: 1px solid var(--border); padding-bottom: 7px; }
.tag { font-size: 0.65rem; font-weight: 700; background: var(--accent); color: #fff;
       padding: 2px 7px; border-radius: 3px; letter-spacing: 0.07em;
       font-family: var(--mono); }
.stitle { font-size: 0.9rem; font-weight: 600; font-family: var(--mono); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(440px, 1fr)); gap: 16px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.clabel { font-size: 0.68rem; color: var(--muted); padding: 7px 12px 3px;
          letter-spacing: 0.05em; text-transform: uppercase; font-family: var(--mono); }
.card img { width: 100%; display: block; }
.fw { width: 100%; }
.fw img { width: 100%; display: block; }
</style>
"""


def _img(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="">'


def _sec(tag: str, title: str) -> str:
    return (f'<div class="sh"><span class="tag">{tag}</span>'
            f'<span class="stitle">{title}</span></div>')


def build_html(
    acc_charts: dict[str, str],
    eng_charts: dict[str, str],
    ema_charts: dict[str, str],
    util_b64: str,
    switch_b64: str,
    generated_at: str,
    n_runs: int,
) -> str:
    p: list[str] = []
    p.append(f'<link rel="stylesheet" href="{GOOGLE_FONTS}">')
    p.append(CSS)
    p.append(f"""
<header>
  <h1>HarmonE</h1>
  <span class="sub">experiment dashboard &middot; {n_runs} runs &middot; {generated_at}</span>
</header>
<main>
""")

    p.append(_sec("01", "Accuracy by planner"))
    p.append('<div class="grid">')
    for ds in DATASET_ORDER:
        if ds in acc_charts:
            p.append(f'<div class="card"><div class="clabel">{DATASET_LABELS[ds]}</div>'
                     f'{_img(acc_charts[ds])}</div>')
    p.append("</div>")

    p.append(_sec("02", "Energy consumption by planner"))
    p.append('<div class="grid">')
    for ds in DATASET_ORDER:
        if ds in eng_charts:
            p.append(f'<div class="card"><div class="clabel">{DATASET_LABELS[ds]}</div>'
                     f'{_img(eng_charts[ds])}</div>')
    p.append("</div>")

    if ema_charts:
        p.append(_sec("03", "EMA score over time (regression)"))
        p.append('<div class="grid">')
        for ds, b64 in ema_charts.items():
            p.append(f'<div class="card"><div class="clabel">{DATASET_LABELS[ds]}</div>'
                     f'{_img(b64)}</div>')
        p.append("</div>")

    if util_b64:
        p.append(_sec("04", "Model utilization (regression)"))
        p.append(f'<div class="fw card">{_img(util_b64)}</div>')

    if switch_b64:
        p.append(_sec("05", "Switch rate &amp; violation suppression"))
        p.append(f'<div class="fw card">{_img(switch_b64)}</div>')

    p.append("</main>")
    return "\n".join(p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir",  default="runs")
    parser.add_argument("--out",       default=None)
    parser.add_argument("--plots-dir", default=None)
    args = parser.parse_args()

    runs_dir  = Path(args.runs_dir)
    out_path  = Path(args.out) if args.out else runs_dir / "dashboard.html"
    plots_dir = Path(args.plots_dir) if args.plots_dir else runs_dir / "plots"

    print(f"Loading runs from {runs_dir} ...")
    manifests = load_runs(runs_dir)
    events    = load_events(runs_dir, manifests)
    print(f"  {len(manifests)} manifests, {len(events)} event CSVs")

    acc_charts: dict[str, str] = {}
    eng_charts: dict[str, str] = {}
    for ds in DATASET_ORDER:
        rows = dataset_rows(manifests, ds)
        if not rows:
            print(f"  {ds}: no data")
            continue
        n_pins = sum(1 for r in rows if r[4])
        print(f"  {ds}: {n_pins} baselines + {len(rows)-n_pins} planners")
        acc_b64, eng_b64 = make_charts(manifests, ds, plots_dir)
        acc_charts[ds] = acc_b64
        eng_charts[ds] = eng_b64

    print("Secondary charts ...")
    ema_charts = chart_ema_timeseries(events, manifests)
    util_b64   = chart_model_utilization(manifests)
    switch_b64 = chart_switch_stats(manifests)

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = build_html(acc_charts, eng_charts, ema_charts, util_b64, switch_b64,
                      ts, len(manifests))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard: {out_path}")
    print(f"PNGs:      {plots_dir}/")


if __name__ == "__main__":
    main()

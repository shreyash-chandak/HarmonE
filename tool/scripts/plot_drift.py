"""
scripts/plot_drift.py — Drift-metric-vs-step plot, one curve per dataset.

The KL-divergence drift signal (core/drift/kl_fixed_ref.py::KLFixedRefDetector)
is computed purely from the ground-truth/luminance stream + a fixed reference
distribution + per-dataset window/bin config — it never looks at model
predictions, energy, or planner state. Both experiments/run_experiment.py and
concurrent_harness/mape/plan_thread.py call it unconditionally every
monitor_interval cycle, for every planner. So for a given dataset the kl_div
time series is identical no matter which planner produced the log — there is
exactly one curve per dataset, not one per planner.

This script does NOT recompute the signal. It reads it straight out of one
representative completed run's mape_events.csv per dataset, and reads that
same run's own thresholds.json for the tau_drift value that was actually in
effect (not the current, possibly since-recalibrated, dataset config).

Why a *non-drift-aware* planner's run specifically: in concurrent_harness,
mape/drift_thread.py (t2 — the only thread running for harmone_original/
violation_aware/pareto/bandit) only appends a mape_events.csv row once drift
is already detected, so a drift-aware planner's own log is missing every
sub-threshold cycle. mape/plan_thread.py (t1 — naive/naive_prt/random_switch/
random_switch_prt/greedy_switch) logs every cycle unconditionally, giving a
complete trace. Since the signal itself doesn't depend on the planner, picking
one of those five planners' runs is just picking the version with a complete
log — not a different drift curve.

Usage (from inside tool/):
    python3 scripts/plot_drift.py
    python3 scripts/plot_drift.py --runs-dir runs --out-dir runs/plots/drift
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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

# Planners whose mape_events.csv logs a kl_div value every cycle (not just on
# detection) — see module docstring. Order is a preference for which one to
# pick if a dataset has several.
NON_DRIFT_AWARE_PLANNERS = [
    "naive", "naive_prt", "random_switch", "random_switch_prt", "greedy_switch",
]

DATASETS = [
    "pems_driftinduced",
    "spot_prices_driftinduced",
    "uci_electricity_driftinduced",
    "bdd100k",
    "acdc",
    "imagenet",
]

DATASET_LABELS = {
    "pems_driftinduced":            "PEMS (drift-induced)",
    "spot_prices_driftinduced":     "Spot Prices (drift-induced)",
    "uci_electricity_driftinduced": "UCI Electricity (drift-induced)",
    "bdd100k":                      "BDD100K",
    "acdc":                         "ACDC",
    "imagenet":                     "ImageNet",
}


def _locate(run_dir: Path, filename: str) -> Path | None:
    """Find `filename` in run_dir directly, falling back to run_dir/knowledge/.

    Pre-2026-09-07 concurrent runs still have mape_events.csv/thresholds.json
    nested under a knowledge/ subdirectory (fixed for new runs, not migrated
    for old ones — see context/concurrent_harness_setup.md's 2026-09-07 note).
    """
    top = run_dir / filename
    if top.exists():
        return top
    nested = run_dir / "knowledge" / filename
    if nested.exists():
        return nested
    return None


def find_representative_run(runs_dir: Path, dataset: str) -> Path | None:
    """Pick one non-drift-aware-planner run directory for `dataset`.

    Reads each candidate's run_manifest.json (never inferred from the
    directory name alone) and returns the first match, sorted for
    determinism, whose planner is in NON_DRIFT_AWARE_PLANNERS and which has
    a usable mape_events.csv.
    """
    candidates = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        mf_path = entry / "run_manifest.json"
        if not mf_path.exists():
            continue
        try:
            with open(mf_path) as f:
                mf = json.load(f)
        except Exception:
            continue
        if mf.get("dataset") != dataset:
            continue
        if mf.get("planner") not in NON_DRIFT_AWARE_PLANNERS:
            continue
        if _locate(entry, "mape_events.csv") is None:
            continue
        candidates.append((NON_DRIFT_AWARE_PLANNERS.index(mf["planner"]), entry.name, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]))
    return candidates[0][2]


def load_drift_series(run_dir: Path) -> tuple[pd.DataFrame, float | None]:
    """Return (dataframe with step/kl_div/drift_detected, tau_drift used)."""
    events_path = _locate(run_dir, "mape_events.csv")
    df = pd.read_csv(events_path)
    df = df[df["kl_div"].notna()].sort_values("step")

    tau_drift = None
    thresholds_path = _locate(run_dir, "thresholds.json")
    if thresholds_path is not None:
        with open(thresholds_path) as f:
            tau_drift = json.load(f).get("tau_drift")

    return df, tau_drift


def plot_dataset_drift(dataset: str, df: pd.DataFrame, tau_drift: float | None,
                        run_name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.4))

    ax.plot(df["step"], df["kl_div"], drawstyle="steps-post",
            color="#4878d0", linewidth=1.2, label="KL divergence")

    triggered = df[df["drift_detected"] == True]  # noqa: E712 (CSV bool column)
    if not triggered.empty:
        ax.scatter(triggered["step"], triggered["kl_div"], color="#d62728",
                   s=18, zorder=3, label="drift detected")

    if tau_drift is not None:
        ax.axhline(tau_drift, color="#333333", linewidth=1.0, linestyle=(0, (3, 2)),
                   label=f"tau_drift = {tau_drift:g}")

    ax.set_xlabel("Image / step number")
    ax.set_ylabel("Drift metric (KL divergence)")
    ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)} — drift metric over time\n"
                 f"(source run: {run_name})", fontweight="normal")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), format="png", bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else runs_dir / "plots" / "drift"

    for dataset in DATASETS:
        run_dir = find_representative_run(runs_dir, dataset)
        if run_dir is None:
            print(f"  {dataset}: no usable non-drift-aware-planner run found — skipped")
            continue

        df, tau_drift = load_drift_series(run_dir)
        if df.empty:
            print(f"  {dataset}: {run_dir.name} has no logged kl_div values — skipped")
            continue

        out_path = out_dir / f"{dataset}_drift.png"
        plot_dataset_drift(dataset, df, tau_drift, run_dir.name, out_path)
        n_triggers = int((df["drift_detected"] == True).sum())  # noqa: E712
        print(f"  {dataset}: {run_dir.name}  ({len(df)} points, "
              f"{n_triggers} above tau_drift={tau_drift})  -> {out_path}")


if __name__ == "__main__":
    main()

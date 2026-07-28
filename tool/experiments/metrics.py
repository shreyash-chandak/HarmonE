"""
experiments/metrics.py — Post-hoc metrics computation and aggregation.

Functions:
    compute_run_metrics(run_dir)       → dict
    aggregate_grid(grid_dir)           → list[dict]
    to_csv(rows, path)
    to_latex(rows, path, caption, label)
    pareto_efficiency(rows)            → list[dict]  (adds is_pareto column)
    wilcoxon_test(rows, baseline, treatment)  → dict

CLI:
    python experiments/metrics.py aggregate --grid-dir runs/my_grid --out metrics.csv
    python experiments/metrics.py latex     --grid-dir runs/my_grid --out table.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))


# ── Per-run metrics ───────────────────────────────────────────────────────────

def compute_run_metrics(run_dir: str) -> dict[str, Any]:
    """Compute summary metrics from a single completed run directory.

    Reads predictions.csv and run_manifest.json.  Returns a flat dict
    suitable for aggregation across many runs.

    Keys returned:
        run_id, dataset, planner, seed,
        r2_mean, r2_std, r2_final_window,
        energy_uJ_mean, energy_uJ_total,
        model_switches, retrains, retrain_skipped, vmr_events, noops,
        mape_k_energy_uJ,
        elapsed_s, total_steps, mape_cycles.
    """
    run_path = Path(run_dir)

    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No run_manifest.json in {run_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    predictions_path = run_path / "predictions.csv"
    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    energy_all: list[float] = []

    if predictions_path.exists():
        with open(predictions_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                y_true_all.append(float(row["y_true"]))
                y_pred_all.append(float(row["y_pred"]))
                energy_all.append(float(row["energy_uJ"]))

    # Overall R²
    r2_mean: float | None = None
    r2_std: float | None = None
    r2_final_window: float | None = None

    if len(y_true_all) > 1:
        from sklearn.metrics import r2_score
        r2_mean = float(r2_score(y_true_all, y_pred_all))

        # R² per monitor window (50-sample blocks)
        window_r2s: list[float] = []
        w = 50
        for start in range(0, len(y_true_all), w):
            chunk_t = y_true_all[start : start + w]
            chunk_p = y_pred_all[start : start + w]
            if len(chunk_t) > 1:
                window_r2s.append(float(r2_score(chunk_t, chunk_p)))
        if window_r2s:
            r2_std = float(np.std(window_r2s, ddof=1) if len(window_r2s) > 1 else 0.0)
            r2_final_window = window_r2s[-1]

    energy_uJ_mean = float(np.mean(energy_all)) if energy_all else 0.0
    energy_uJ_total = float(np.sum(energy_all)) if energy_all else 0.0

    counters = manifest.get("event_counters", {})

    return {
        "run_id": manifest.get("run_id", run_path.name),
        "dataset": manifest.get("dataset", ""),
        "planner": manifest.get("planner", ""),
        "seed": manifest.get("seed", -1),
        "r2_mean": round(r2_mean, 6) if r2_mean is not None else None,
        "r2_std": round(r2_std, 6) if r2_std is not None else None,
        "r2_final_window": round(r2_final_window, 6) if r2_final_window is not None else None,
        "energy_uJ_mean": round(energy_uJ_mean, 4),
        "energy_uJ_total": round(energy_uJ_total, 4),
        "model_switches": counters.get("model_switches", 0),
        "retrains": counters.get("retrains", 0),
        "retrain_skipped": counters.get("retrain_skipped", 0),
        "vmr_events": counters.get("vmr_events", 0),
        "noops": counters.get("noops", 0),
        "mape_k_energy_uJ": counters.get("mape_k_energy_uJ", 0.0),
        "elapsed_s": manifest.get("elapsed_s", None),
        "total_steps": manifest.get("total_steps", 0),
        "mape_cycles": manifest.get("mape_cycles", 0),
    }


# ── Grid aggregation ──────────────────────────────────────────────────────────

def aggregate_grid(grid_dir: str) -> list[dict[str, Any]]:
    """Iterate all run sub-directories and aggregate metrics.

    A sub-directory is a run if it contains run_manifest.json.

    Returns a list of metric dicts, one per completed run, sorted by
    (dataset, planner, seed).
    """
    grid_path = Path(grid_dir)
    rows: list[dict] = []

    for sub in sorted(grid_path.iterdir()):
        if not sub.is_dir():
            continue
        manifest = sub / "run_manifest.json"
        if not manifest.exists():
            continue
        try:
            metrics = compute_run_metrics(str(sub))
            rows.append(metrics)
        except Exception as exc:
            print(f"[WARNING] Skipping {sub.name}: {exc}", file=sys.stderr)

    rows.sort(key=lambda r: (r["dataset"], r["planner"], r["seed"]))
    return rows


def aggregate_by_planner(rows: list[dict]) -> list[dict]:
    """Collapse seeds: return mean ± std across seeds for each (dataset, planner)."""
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["planner"])].append(r)

    numeric_cols = [
        "r2_mean", "r2_std", "energy_uJ_mean", "energy_uJ_total",
        "model_switches", "mape_k_energy_uJ", "elapsed_s", "total_steps",
    ]
    summaries = []
    for (dataset, planner), group in sorted(groups.items()):
        summary: dict[str, Any] = {"dataset": dataset, "planner": planner, "n_seeds": len(group)}
        for col in numeric_cols:
            vals = [r[col] for r in group if r.get(col) is not None]
            if vals:
                summary[f"{col}_mean"] = round(float(np.mean(vals)), 6)
                summary[f"{col}_std"] = round(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), 6)
        summaries.append(summary)

    return summaries


# ── Pareto efficiency ─────────────────────────────────────────────────────────

def pareto_efficiency(rows: list[dict]) -> list[dict]:
    """Mark each row as Pareto-dominant over all others.

    A planner (dataset, planner) is Pareto-efficient if no other planner on the
    same dataset has both higher r2_mean_mean AND lower energy_uJ_mean_mean.

    Adds column ``is_pareto`` (bool) to each row.  Input rows should be the
    output of ``aggregate_by_planner``.
    """
    per_dataset: dict[str, list[dict]] = {}
    for r in rows:
        per_dataset.setdefault(r["dataset"], []).append(r)

    result = []
    for dataset, group in per_dataset.items():
        for candidate in group:
            c_r2 = candidate.get("r2_mean_mean", 0.0) or 0.0
            c_e = candidate.get("energy_uJ_mean_mean", float("inf")) or float("inf")
            dominated = any(
                (other.get("r2_mean_mean", 0.0) or 0.0) > c_r2
                and (other.get("energy_uJ_mean_mean", float("inf")) or float("inf")) < c_e
                for other in group
                if other is not candidate
            )
            result.append({**candidate, "is_pareto": not dominated})

    return result


# ── Statistical tests ─────────────────────────────────────────────────────────

def wilcoxon_test(
    rows: list[dict],
    baseline: str,
    treatment: str,
    metric: str = "r2_mean",
    dataset: str | None = None,
) -> dict[str, Any]:
    """Wilcoxon signed-rank test comparing treatment vs. baseline across seeds.

    Args:
        rows:      Per-run metric dicts (from aggregate_grid).
        baseline:  Planner name to use as the null hypothesis.
        treatment: Planner name to compare against the baseline.
        metric:    Column to compare (default "r2_mean").
        dataset:   Filter to this dataset; None = use all.

    Returns:
        dict with keys: statistic, p_value, n_pairs, direction, significant.
    """
    from scipy.stats import wilcoxon as _wilcoxon

    def _filter(planner: str) -> list[float]:
        return [
            r[metric]
            for r in rows
            if r["planner"] == planner
            and r.get(metric) is not None
            and (dataset is None or r["dataset"] == dataset)
        ]

    base_vals = _filter(baseline)
    treat_vals = _filter(treatment)

    paired = [
        (b, t) for b, t in zip(base_vals, treat_vals)
        if b is not None and t is not None
    ]

    if len(paired) < 2:
        return {
            "statistic": None,
            "p_value": None,
            "n_pairs": len(paired),
            "direction": None,
            "significant": None,
            "note": "Insufficient paired samples",
        }

    b_arr = np.array([p[0] for p in paired])
    t_arr = np.array([p[1] for p in paired])
    diffs = t_arr - b_arr

    if np.all(diffs == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_pairs": len(paired),
            "direction": "equal",
            "significant": False,
        }

    stat, p = _wilcoxon(diffs)
    direction = "better" if float(np.median(diffs)) > 0 else "worse"
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_pairs": len(paired),
        "direction": direction,
        "significant": float(p) < 0.05,
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def to_csv(rows: list[dict], path: str) -> None:
    """Write rows to a CSV file."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {path}")


def to_latex(
    rows: list[dict],
    path: str,
    caption: str = "Experiment results",
    label: str = "tab:results",
    columns: list[str] | None = None,
) -> None:
    """Write rows as a LaTeX booktabs table."""
    if not rows:
        return

    if columns is None:
        columns = list(rows[0].keys())

    header = " & ".join(c.replace("_", "\\_") for c in columns) + r" \\"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for row in rows:
        vals = []
        for col in columns:
            v = row.get(col, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            elif v is None:
                vals.append("--")
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute and aggregate HarmonE experiment metrics.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # aggregate
    agg_p = sub.add_parser("aggregate", help="Aggregate all runs in a grid dir")
    agg_p.add_argument("--grid-dir", required=True)
    agg_p.add_argument("--out", default="metrics.csv")
    agg_p.add_argument("--by-planner", action="store_true", help="Collapse seeds")
    agg_p.add_argument("--pareto", action="store_true", help="Add Pareto column")

    # latex
    lat_p = sub.add_parser("latex", help="Emit a LaTeX table from a grid dir")
    lat_p.add_argument("--grid-dir", required=True)
    lat_p.add_argument("--out", default="table.tex")
    lat_p.add_argument("--caption", default="HarmonE experiment results")
    lat_p.add_argument("--label", default="tab:harmone_results")
    lat_p.add_argument("--columns", nargs="+", default=None)

    # wilcoxon
    wil_p = sub.add_parser("wilcoxon", help="Wilcoxon test: treatment vs baseline")
    wil_p.add_argument("--grid-dir", required=True)
    wil_p.add_argument("--baseline", required=True)
    wil_p.add_argument("--treatment", required=True)
    wil_p.add_argument("--metric", default="r2_mean")
    wil_p.add_argument("--dataset", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "aggregate":
        rows = aggregate_grid(args.grid_dir)
        if args.by_planner:
            rows = aggregate_by_planner(rows)
        if args.pareto:
            rows = pareto_efficiency(rows)
        to_csv(rows, args.out)

    elif args.cmd == "latex":
        rows = aggregate_grid(args.grid_dir)
        rows = aggregate_by_planner(rows)
        to_latex(rows, args.out, caption=args.caption, label=args.label, columns=args.columns)

    elif args.cmd == "wilcoxon":
        rows = aggregate_grid(args.grid_dir)
        result = wilcoxon_test(
            rows, args.baseline, args.treatment,
            metric=args.metric, dataset=args.dataset,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

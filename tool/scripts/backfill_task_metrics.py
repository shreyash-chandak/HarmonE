"""
scripts/backfill_task_metrics.py — Backfill RMSE/MAE into existing run manifests.

Scans all subdirectories of tool/runs/ that contain both predictions.csv and
run_manifest.json.  For regression runs (predictions.csv has y_true/y_pred
columns), computes RMSE and MAE and patches the manifest with a task_metrics
block.  CV runs (proxy_acc column) are skipped.

Usage (from tool/):
    python3 scripts/backfill_task_metrics.py [--runs-dir runs] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def _compute_energy_by_model(rows: list[dict]) -> dict:
    """Return {model_name: {energy_uJ, energy_mJ, n_steps}} from prediction rows."""
    ebm: dict = {}
    for r in rows:
        m = r.get("active_model", "unknown")
        if m not in ebm:
            ebm[m] = {"energy_uJ": 0.0, "n_steps": 0}
        ebm[m]["energy_uJ"] += float(r.get("energy_uJ", 0.0))
        ebm[m]["n_steps"] += 1
    for v in ebm.values():
        v["energy_uJ"] = round(v["energy_uJ"], 2)
        v["energy_mJ"] = round(v["energy_uJ"] / 1000.0, 4)
    return ebm


def _load_regression_rows(predictions_path: Path) -> tuple[list[dict], list[str]] | tuple[None, None]:
    """Read predictions.csv; return (rows, fieldnames) for regression runs, or (None, None) for CV."""
    with open(predictions_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "y_true" not in fieldnames or "y_pred" not in fieldnames:
            return None, None
        rows = list(reader)
    return rows, fieldnames


def _compute_metrics(rows: list[dict], fieldnames: list[str]) -> dict:
    """Return task_metrics dict from already-loaded regression prediction rows."""
    y_true = [float(r["y_true"]) for r in rows]
    y_pred = [float(r["y_pred"]) for r in rows]
    n = len(y_true)
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n
    mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)

    # R² (coefficient of determination) — canonical HarmonE accuracy metric
    y_mean = sum(y_true) / n
    ss_tot = sum((v - y_mean) ** 2 for v in y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    total_energy_uJ = 0.0
    if "energy_uJ" in fieldnames:
        total_energy_uJ = sum(float(r["energy_uJ"]) for r in rows)

    return {
        "r2": round(r2, 6),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "n_samples": n,
        "total_inference_energy_uJ": round(total_energy_uJ, 2),
        "total_inference_energy_mJ": round(total_energy_uJ / 1000.0, 4),
    }


def backfill(runs_dir: Path, dry_run: bool) -> None:
    updated = skipped_cv = skipped_exists = missing = 0

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "logs":
            continue

        manifest_path = run_dir / "run_manifest.json"
        predictions_path = run_dir / "predictions.csv"

        if not manifest_path.exists() or not predictions_path.exists():
            missing += 1
            continue

        manifest = json.loads(manifest_path.read_text())

        existing = manifest.get("task_metrics", {})
        has_full_metrics = existing and "r2" in existing and "total_inference_energy_uJ" in existing
        has_ebm = bool(manifest.get("energy_by_model"))
        if has_full_metrics and has_ebm:
            skipped_exists += 1
            continue

        rows, fieldnames = _load_regression_rows(predictions_path)
        if rows is None:
            skipped_cv += 1
            continue
        if not rows:
            missing += 1
            continue

        metrics = _compute_metrics(rows, fieldnames)
        ebm = _compute_energy_by_model(rows)

        manifest["task_metrics"] = metrics
        manifest["energy_by_model"] = ebm
        if not dry_run:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        print(
            f"{'[DRY] ' if dry_run else ''}Updated {run_dir.name}: "
            f"R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
            f"n={metrics['n_samples']}  energy={metrics['total_inference_energy_mJ']:.4f}mJ"
        )
        updated += 1

    print(
        f"\nDone. updated={updated}  skipped_already_has_metrics={skipped_exists}  "
        f"skipped_cv={skipped_cv}  missing_files={missing}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill RMSE/MAE into existing run manifests.")
    parser.add_argument("--runs-dir", default="runs", help="Path to the runs directory (default: runs)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be updated without writing")
    args = parser.parse_args(argv)

    tool_dir = Path(__file__).resolve().parent.parent
    runs_dir = Path(args.runs_dir) if Path(args.runs_dir).is_absolute() else tool_dir / args.runs_dir

    if not runs_dir.exists():
        print(f"ERROR: runs directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    backfill(runs_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

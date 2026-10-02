"""
scripts/generate_results_table.py — build context/results.md from the
concurrent harness's own per-run logs.

Reads directly from concurrent_harness/runs/logs/<run_id>_conc.log — NOT from
run_manifest.json on disk — because run_concurrent.py prints the exact same
manifest dict as indented JSON as the last thing it does (mirroring
run_experiment.py:3166's own behavior), so every completed run's full result
already lives at the tail of its own log. This also means a run whose
directory was since cleaned up but whose log survives still contributes a row.

Regression rows: accuracy = task_metrics.r2, energy = event_counters'
combined energy_used_mJ (the single cross-process total — see
concurrent_harness_setup.md's "energy double-counting" resolution note; NOT a
per-model sum, matches the "energy summary | energy_used=..." log line
exactly).

CV rows: confidence = task_metrics.mean_proxy_acc (the live run's own proxy
score — literally "confidence" for classification/detection), energy = same
combined energy_used_mJ field, offline accuracy = read from the matching
<run_id>_conc_offline_eval.log if one exists (mAP50 for detection, accuracy
for classification, mIoU for segmentation — whichever key that task's
offline_eval module reports), left blank if that log doesn't exist (offline
eval not run yet / no ground truth for that run).

A run whose manifest has status != "ok" (mape/manage.py or inference.py
crashed/timed-out — see run_concurrent.py) still gets a row: its task_metrics/
energy are computed from whatever predictions.csv it managed to write before
failing, same as a clean run, but the row is marked FAILED so the numbers
aren't mistaken for a clean result.

Usage (from inside tool/):
    python scripts/generate_results_table.py
    python scripts/generate_results_table.py --logs-dir concurrent_harness/runs/logs --out ../context/results.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Canonical row ordering — mirrors scripts/plot_results.py's PLANNER_ORDER.
PLANNER_ORDER = [
    "naive", "naive_prt", "random_switch", "random_switch_prt", "greedy_switch",
    "harmone_original", "violation_aware", "pareto", "bandit",
]

DATASET_ORDER = [
    "pems_driftinduced", "spot_prices_driftinduced", "uci_electricity_driftinduced",
    "bdd100k", "acdc", "imagenet", "imagenet_c", "iwildcam",
]


def _extract_trailing_json(log_path: Path) -> dict | None:
    """Return the last top-level JSON object printed in a log file, if any.

    Both run_concurrent.py's manifest print and offline_eval.py's results
    print use `print(json.dumps(..., indent=2))`, which always starts with a
    line that is exactly "{" — find the LAST such line and parse from there.
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i] == "{":
            start = i
            break
    if start is None:
        return None
    blob = "\n".join(lines[start:])
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


def _planner_sort_key(planner: str, pin_model: str | None):
    idx = PLANNER_ORDER.index(planner) if planner in PLANNER_ORDER else len(PLANNER_ORDER)
    return (idx, pin_model or "")


def _fmt(value, digits=4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def load_runs(logs_dir: Path) -> list[dict]:
    runs = []
    for log_path in sorted(logs_dir.glob("*.log")):
        name = log_path.name
        if name.startswith("master_") or name.startswith("failed_") or name.endswith("_offline_eval.log"):
            continue
        mf = _extract_trailing_json(log_path)
        if mf is None or "task_metrics" not in mf:
            continue
        mf["_log_path"] = log_path
        runs.append(mf)
    return runs


def load_offline_metric(logs_dir: Path, run_id: str) -> tuple[float | None, str | None]:
    """Return (value, metric_label) from <run_id>_conc_offline_eval.log, or (None, None)."""
    ev_path = logs_dir / f"{run_id}_offline_eval.log"
    if not ev_path.exists():
        return None, None
    result = _extract_trailing_json(ev_path)
    if result is None:
        return None, None
    overall = result.get("overall", {})
    for key, label in (("map50", "mAP50"), ("accuracy", "Accuracy"), ("miou", "mIoU")):
        if key in overall:
            return overall[key], label
    return None, None


def build_regression_table(runs: list[dict]) -> str:
    rows = [m for m in runs if "r2" in m.get("task_metrics", {})]
    rows.sort(key=lambda m: (
        DATASET_ORDER.index(m["dataset"]) if m["dataset"] in DATASET_ORDER else 99,
        _planner_sort_key(m.get("planner", ""), m.get("pin_model")),
    ))

    lines = [
        "| Dataset | Planner | Pin Model | Accuracy (R²) | Energy Usage (mJ) | Status |",
        "|---|---|---|---|---|---|",
    ]
    for m in rows:
        tm = m["task_metrics"]
        energy_mj = m.get("event_counters", {}).get("energy_used_mJ")
        status = "" if m.get("status") == "ok" else f"**FAILED** — {m.get('failure_reason', '')}"
        lines.append(
            f"| {m['dataset']} | {m.get('planner', '')} | {m.get('pin_model') or '-'} "
            f"| {_fmt(tm.get('r2'))} | {_fmt(energy_mj, 3)} | {status} |"
        )
    return "\n".join(lines)


def build_cv_table(runs: list[dict], logs_dir: Path) -> str:
    rows = [m for m in runs if "mean_proxy_acc" in m.get("task_metrics", {})]
    rows.sort(key=lambda m: (
        DATASET_ORDER.index(m["dataset"]) if m["dataset"] in DATASET_ORDER else 99,
        _planner_sort_key(m.get("planner", ""), m.get("pin_model")),
    ))

    lines = [
        "| Dataset | Planner | Pin Model | Offline Accuracy | Offline Metric | Confidence | Energy Usage (mJ) | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in rows:
        tm = m["task_metrics"]
        energy_mj = m.get("event_counters", {}).get("energy_used_mJ")
        status = "" if m.get("status") == "ok" else f"**FAILED** — {m.get('failure_reason', '')}"
        off_val, off_label = load_offline_metric(logs_dir, m["run_id"])
        lines.append(
            f"| {m['dataset']} | {m.get('planner', '')} | {m.get('pin_model') or '-'} "
            f"| {_fmt(off_val)} | {off_label or ''} | {_fmt(tm.get('mean_proxy_acc'))} "
            f"| {_fmt(energy_mj, 3)} | {status} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default="concurrent_harness/runs/logs")
    parser.add_argument("--out", default="../context/results.md")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_path = Path(args.out)

    runs = load_runs(logs_dir)
    print(f"Parsed {len(runs)} run logs from {logs_dir}")

    reg_table = build_regression_table(runs)
    cv_table = build_cv_table(runs, logs_dir)

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    content = f"""# Concurrent Harness Results

Generated {ts} by `scripts/generate_results_table.py` from
`concurrent_harness/runs/logs/*_conc.log` (each run's full manifest is printed
as JSON at the end of its own log — see `run_concurrent.py`). Not sourced
from `run_manifest.json` files on disk.

A row marked **FAILED** still reflects real numbers computed from whatever
predictions that run wrote before `mape/manage.py` or `inference.py`
crashed/timed out — treat those numbers as partial, not representative of a
clean run.

## Regression datasets

Accuracy is R² on the full stream. Energy usage is the run's combined
(inference + MAPE-K) energy total.

{reg_table}

## CV datasets

Confidence is the live run's own proxy-accuracy score (task-appropriate proxy:
detection/classification confidence or segmentation confidence proxy).
Offline accuracy is computed separately against ground truth by
`experiments/offline_eval.py` (mAP50 for detection, accuracy for
classification, mIoU for segmentation) and is blank where that pass hasn't
been run for that specific run yet.

{cv_table}
"""

    out_path.write_text(content, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

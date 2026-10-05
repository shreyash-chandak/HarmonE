"""
scripts/experiment_logger.py — Timestamped experiment logger for HarmonE runs.

Polls the knowledge/ directory at a fixed interval and appends one CSV row per
snapshot, capturing all readings needed for paper reporting:

    Regression:  r2_score, RMSE, energy_per_pred_uJ, model_used, ema_scores,
                 event_counters, drift_kl, current_energy_threshold

    CV:          proxy_score, energy_per_pred_uJ, model_used, ema_scores,
                 event_counters, drift_kl, embedding_drift

Log files are named:  logs/<dataset>_<YYYY-MM-DD_HH-MM-SS>.csv
One row per polling tick (default: every 30 s).

Usage:
    cd tool/managed_system_regression    # or managed_system_cv
    python3 ../../scripts/experiment_logger.py \\
        --knowledge knowledge/ \\
        --dataset pems_node2 \\
        [--domain regression|cv]  \\
        [--interval 30]           \\
        [--log-dir ../../logs/]

Run alongside the managed system; Ctrl-C to stop.
The logger never writes to knowledge/ — read-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _read_csv_tail(path: Path, n: int = 200) -> list[dict]:
    """Read last n rows of a CSV file, return as list of dicts."""
    try:
        import pandas as pd
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df.tail(n).to_dict("records")
    except Exception:
        return []


def _safe_mean(values: list, key: str) -> float | None:
    vals = [v.get(key) for v in values if v.get(key) is not None]
    try:
        nums = [float(v) for v in vals if v == v]  # excludes NaN strings
        return sum(nums) / len(nums) if nums else None
    except Exception:
        return None


# ── Snapshot builders ─────────────────────────────────────────────────────────

def _snapshot_regression(knowledge_dir: Path) -> dict:
    mape = _read_json(knowledge_dir / "mape_info.json")
    thresholds = _read_json(knowledge_dir / "thresholds.json")
    drift_kl = _read_json(knowledge_dir / "drift_kl.json")

    try:
        with open(knowledge_dir / "model.csv") as f:
            current_model = f.read().strip()
    except Exception:
        current_model = ""

    recent = _read_csv_tail(knowledge_dir / "predictions.csv", n=200)

    row: dict = {
        "wall_clock_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "unix_ts": time.time(),
        "current_model": current_model,
        "last_mape_line": mape.get("last_line"),
        "current_energy_threshold": mape.get("current_energy_threshold"),
        # EMA scores
        "ema_score_lstm":   (mape.get("ema_scores") or {}).get("lstm"),
        "ema_score_linear": (mape.get("ema_scores") or {}).get("linear"),
        "ema_score_svm":    (mape.get("ema_scores") or {}).get("svm"),
        "ema_acc_lstm":     (mape.get("ema_accuracy") or {}).get("lstm"),
        "ema_acc_linear":   (mape.get("ema_accuracy") or {}).get("linear"),
        "ema_acc_svm":      (mape.get("ema_accuracy") or {}).get("svm"),
        "ema_energy_lstm":  (mape.get("ema_energy") or {}).get("lstm"),
        "ema_energy_linear":(mape.get("ema_energy") or {}).get("linear"),
        "ema_energy_svm":   (mape.get("ema_energy") or {}).get("svm"),
        # Event counters
        "event_model_switches": (mape.get("event_counters") or {}).get("model_switches"),
        "event_retrains":       (mape.get("event_counters") or {}).get("retrains"),
        "event_vmr":            (mape.get("event_counters") or {}).get("vmr_events"),
        "event_noops":          (mape.get("event_counters") or {}).get("noops"),
        "mape_k_energy_uJ":     (mape.get("event_counters") or {}).get("mape_k_energy_uJ"),
        # Recent prediction stats
        "recent_r2_mean":     _safe_mean(recent, "r2_score") if recent and "r2_score" in (recent[0] if recent else {}) else None,
        "recent_energy_mean_uJ": _safe_mean(recent, "energy_uJ"),
        "recent_n_rows":      len(recent),
        # Drift
        "drift_kl_primary":   drift_kl.get("kl_primary"),
        "drift_kl_rolling":   drift_kl.get("kl_rolling"),
        "drift_threshold":    thresholds.get("tau_drift"),
        # Planner
        "planner": thresholds.get("planner"),
        "dataset_id": thresholds.get("dataset_id"),
    }

    # Compute recent mean r2 from predictions if column exists
    if recent and "true_value" in recent[0] and "predicted_value" in recent[0]:
        try:
            import numpy as np
            tv = [r["true_value"] for r in recent]
            pv = [r["predicted_value"] for r in recent]
            ss_res = sum((t - p) ** 2 for t, p in zip(tv, pv))
            ss_tot = sum((t - sum(tv) / len(tv)) ** 2 for t in tv)
            row["recent_r2_mean"] = 1 - ss_res / ss_tot if ss_tot > 0 else None
            row["recent_rmse"] = (ss_res / len(tv)) ** 0.5
        except Exception:
            pass

    return row


def _snapshot_cv(knowledge_dir: Path) -> dict:
    mape = _read_json(knowledge_dir / "mape_info.json")
    thresholds = _read_json(knowledge_dir / "thresholds.json")
    drift_kl = _read_json(knowledge_dir / "drift_kl.json")

    try:
        with open(knowledge_dir / "model.csv") as f:
            current_model = f.read().strip()
    except Exception:
        current_model = ""

    recent = _read_csv_tail(knowledge_dir / "predictions.csv", n=200)

    row: dict = {
        "wall_clock_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "unix_ts": time.time(),
        "current_model": current_model,
        "last_mape_line": mape.get("last_line"),
        "current_energy_threshold": mape.get("current_energy_threshold"),
        # EMA per model (CV may have yolo_n/yolo_s/yolo_m or similar)
        "ema_scores_json": json.dumps(mape.get("ema_scores", {})),
        "ema_accuracy_json": json.dumps(mape.get("ema_accuracy", {})),
        "ema_energy_json": json.dumps(mape.get("ema_energy", {})),
        # Event counters
        "event_model_switches": (mape.get("event_counters") or {}).get("model_switches"),
        "event_retrains":       (mape.get("event_counters") or {}).get("retrains"),
        "event_vmr":            (mape.get("event_counters") or {}).get("vmr_events"),
        "event_noops":          (mape.get("event_counters") or {}).get("noops"),
        "mape_k_energy_uJ":     (mape.get("event_counters") or {}).get("mape_k_energy_uJ"),
        # Recent proxy scores
        "recent_proxy_mean": _safe_mean(recent, "proxy_score"),
        "recent_energy_mean_uJ": _safe_mean(recent, "energy_uJ"),
        "recent_n_rows": len(recent),
        # Drift
        "drift_kl_primary": drift_kl.get("kl_primary"),
        "drift_kl_rolling": drift_kl.get("kl_rolling"),
        "drift_threshold":  thresholds.get("tau_drift"),
        "drift_detector":   thresholds.get("drift_detector"),
        # Config
        "proxy": thresholds.get("proxy"),
        "planner": thresholds.get("planner"),
    }
    return row


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_logger(
    knowledge_dir: str,
    dataset: str,
    domain: str,
    interval_s: float,
    log_dir: str,
) -> None:
    knowledge_path = Path(knowledge_dir).resolve()
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    start_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_path / f"{dataset}_{start_ts}.csv"

    snapshot_fn = _snapshot_cv if domain == "cv" else _snapshot_regression

    print(f"[logger] Knowledge dir : {knowledge_path}")
    print(f"[logger] Log file      : {log_file}")
    print(f"[logger] Polling every : {interval_s}s   (Ctrl-C to stop)")
    print()

    writer: csv.DictWriter | None = None
    fh = None

    def _shutdown(sig, frame):
        print(f"\n[logger] Interrupted. Log saved: {log_file}")
        if fh:
            fh.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    tick = 0
    while True:
        try:
            snap = snapshot_fn(knowledge_path)
        except Exception as exc:
            print(f"[logger] Snapshot error: {exc}")
            time.sleep(interval_s)
            continue

        if writer is None:
            fh = open(log_file, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(fh, fieldnames=list(snap.keys()), extrasaction="ignore")
            writer.writeheader()

        writer.writerow(snap)
        fh.flush()

        tick += 1
        if tick % 10 == 1:
            print(f"[logger] tick={tick:4d}  {snap['wall_clock_utc']}  "
                  f"model={snap.get('current_model','?')}  "
                  f"switches={snap.get('event_model_switches','?')}")

        time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Timestamped experiment logger for HarmonE runs."
    )
    parser.add_argument(
        "--knowledge", required=True,
        help="Path to knowledge/ directory of the running managed system "
             "(e.g. managed_system_regression/knowledge)"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name used in the log file name (e.g. pems_node2)"
    )
    parser.add_argument(
        "--domain", default="regression", choices=["regression", "cv"],
        help="Domain — determines which fields to log (default: regression)"
    )
    parser.add_argument(
        "--interval", type=float, default=30.0,
        help="Polling interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--log-dir", default="logs/",
        help="Directory to write log files into (default: logs/)"
    )
    args = parser.parse_args()

    run_logger(
        knowledge_dir=args.knowledge,
        dataset=args.dataset,
        domain=args.domain,
        interval_s=args.interval,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()

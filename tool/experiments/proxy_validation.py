"""
experiments/proxy_validation.py — Proxy validation pipeline (RQ3).

Computes Spearman ρ between proxy accuracy signal and true mAP@0.5 across
monitoring intervals in a completed run. Produces per-proxy ρ so Table 3
in the journal paper can compare: confidence vs calibrated_confidence vs agreement.

Usage:
    python experiments/proxy_validation.py --run <run_dir> [--output results/proxy_val.json]

Single-command smoke run (uses bundled predictions and labels):
    python experiments/proxy_validation.py --run managed_system_cv/knowledge \
        --labels-dir data/bdd100k/labels/val \
        --images-dir data/bdd100k/images/val \
        --smoke  # use only first 3 intervals

Output JSON schema:
    {
      "proxies": {
        "confidence":            {"spearman_rho": 0.71, "p_value": 0.03, "n_intervals": 5},
        "calibrated_confidence": {"spearman_rho": 0.84, "p_value": 0.01, "n_intervals": 5},
        "agreement":             {"spearman_rho": 0.79, "p_value": 0.02, "n_intervals": 5}
      },
      "map50_per_interval": [0.42, 0.38, ...],
      "n_intervals": 5
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def validate_proxies(
    run_dir: str,
    labels_dir: str,
    images_dir: str,
    interval_size: int = 1000,
    output_path: str | None = None,
    weights_dir: str = "managed_system_cv/models",
    smoke: bool = False,
) -> dict:
    """Compute Spearman ρ between each proxy and true mAP@0.5.

    Args:
        run_dir:       Path to knowledge/ directory containing predictions.csv.
        labels_dir:    Path to YOLO ground-truth label directory.
        images_dir:    Path to images directory.
        interval_size: Monitoring interval (rows in predictions.csv).
        output_path:   Optional path to write results JSON.
        weights_dir:   Directory containing model .pt files.
        smoke:         If True, only process first 3 intervals.

    Returns:
        Dict with per-proxy Spearman ρ and supporting data.
    """
    from experiments.offline_eval import evaluate_run, _compute_map50_for_interval, _model_at_row, _load_model_log, _find_weights

    # Step 1: compute true mAP@0.5 per interval
    print("[PROXY-VAL] Computing offline mAP@0.5 per interval...")
    eval_results = evaluate_run(
        run_dir=run_dir,
        labels_dir=labels_dir,
        images_dir=images_dir,
        interval_size=interval_size,
        output_path=None,
        model_weights_dir=weights_dir,
    )
    intervals = eval_results["interval_results"]
    if smoke:
        intervals = intervals[:3]

    map50_values = [iv["map50"] for iv in intervals]
    valid_mask = [m is not None for m in map50_values]
    map50_clean = [m for m in map50_values if m is not None]

    if len(map50_clean) < 2:
        print("[PROXY-VAL] Not enough valid intervals to compute Spearman ρ.")
        return {"error": "insufficient_intervals", "n_intervals": len(intervals)}

    # Step 2: compute proxy scores per interval
    predictions_path = Path(run_dir) / "predictions.csv"
    df = pd.read_csv(predictions_path)

    proxies_to_test = ["confidence", "calibrated_confidence", "agreement"]
    proxy_results = {}

    for proxy_name in proxies_to_test:
        print(f"[PROXY-VAL] Evaluating proxy: {proxy_name}")
        proxy = _build_proxy(proxy_name, run_dir)

        proxy_scores = []
        for idx, interval in enumerate(intervals):
            if not valid_mask[idx]:
                continue
            start = interval["row_start"]
            end = interval["row_end"]
            interval_df = df.iloc[start:end]

            model_name = interval.get("model", "yolo_n")
            confs = interval_df["confidence"].dropna().tolist() if "confidence" in interval_df else []
            record = {"model": model_name, "confidences": confs}
            score = proxy.score(record)
            proxy_scores.append(score)

        if len(proxy_scores) != len(map50_clean):
            print(f"[PROXY-VAL] Length mismatch for {proxy_name}: {len(proxy_scores)} vs {len(map50_clean)}")
            continue

        rho, p_value = _spearman(proxy_scores, map50_clean)
        proxy_results[proxy_name] = {
            "spearman_rho": rho,
            "p_value": p_value,
            "n_intervals": len(map50_clean),
            "proxy_scores": proxy_scores,
        }
        print(f"[PROXY-VAL] {proxy_name}: ρ={rho:.4f}, p={p_value:.4f}")

    output = {
        "proxies": proxy_results,
        "map50_per_interval": map50_clean,
        "n_intervals": len(map50_clean),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[PROXY-VAL] Results written to {output_path}")

    return output


def _build_proxy(proxy_name: str, run_dir: str):
    """Instantiate a proxy by name, loading calibration from run_dir if available."""
    calib_path = os.path.join(run_dir, "calibration.json")
    if proxy_name == "calibrated_confidence":
        from core.proxies.calibrated_confidence import CalibratedConfidenceProxy
        return CalibratedConfidenceProxy(calibration_path=calib_path)
    elif proxy_name == "agreement":
        from core.proxies.agreement import AgreementProxy
        return AgreementProxy()
    else:
        from core.proxies.confidence import ConfidenceProxy
        return ConfidenceProxy()


def _spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute Spearman rank correlation coefficient and two-tailed p-value."""
    n = len(x)
    if n < 2:
        return 0.0, 1.0

    rx = _rank(x)
    ry = _rank(y)

    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    rho = 1.0 - 6.0 * d2 / (n * (n ** 2 - 1))

    # Two-tailed t-test for significance
    import math
    if abs(rho) >= 1.0:
        p_value = 0.0
    else:
        t_stat = rho * math.sqrt((n - 2) / (1.0 - rho ** 2))
        # Approximate p-value via normal distribution for large n; use t-dist for small n
        try:
            from scipy.stats import t as t_dist
            p_value = float(2.0 * t_dist.sf(abs(t_stat), df=n - 2))
        except ImportError:
            # Fallback: standard normal approximation (valid for n > 20)
            from math import erfc, sqrt
            z = t_stat / math.sqrt(1 + t_stat ** 2 / (n - 2))
            p_value = float(erfc(abs(z) / math.sqrt(2)))

    return float(rho), float(p_value)


def _rank(x: list[float]) -> list[float]:
    """Return fractional ranks (average ties) for a list."""
    n = len(x)
    indexed = sorted(enumerate(x), key=lambda t: t[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


if __name__ == "__main__":
    # Insert tool/ root onto path so core/ and experiments/ are importable
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    parser = argparse.ArgumentParser(description="Proxy validation: Spearman ρ vs mAP@0.5")
    parser.add_argument("--run", required=True, help="Path to knowledge/ directory")
    parser.add_argument("--labels-dir", required=True, help="Ground-truth YOLO labels directory")
    parser.add_argument("--images-dir", required=True, help="Images directory")
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--weights-dir", default="managed_system_cv/models")
    parser.add_argument("--smoke", action="store_true", help="Only process first 3 intervals")
    args = parser.parse_args()

    results = validate_proxies(
        run_dir=args.run,
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        interval_size=args.interval,
        output_path=args.output,
        weights_dir=args.weights_dir,
        smoke=args.smoke,
    )
    print(json.dumps(results, indent=2))

"""experiments/run_reset.py — Reset live managed-system run state (L2-b / L6 fix).

Zeros event counters, EMA scores, last_line, and clears volatile knowledge files
so that each live experiment session starts clean.

Does NOT touch:
  - model weights (versionedMR/, models/)
  - scaler.pkl, reference_distribution.json   (training-time, expensive to recreate)
  - thresholds.json                            (configuration, not run state)
  - model.csv                                  (user-set active model)

Usage:
    cd tool/
    python3 experiments/run_reset.py --domain regression
    python3 experiments/run_reset.py --domain cv
    python3 experiments/run_reset.py --domain regression --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

_DOMAIN_DIRS = {
    "regression": "managed_system_regression",
    "cv": "managed_system_cv",
}

_EMA_DEFAULTS = {
    "regression": {"lstm": 0.5, "linear": 0.5, "svm": 0.5},
    "cv": {"yolo_n": 0.5, "yolo_s": 0.5, "yolo_m": 0.5},
}

_EMA_ACC_DEFAULTS = {
    "lstm": 0.0, "linear": 0.0, "svm": 0.0,
    "yolo_n": 0.0, "yolo_s": 0.0, "yolo_m": 0.0,
}

_EMA_ENERGY_DEFAULTS = {
    "lstm": 0.0, "linear": 0.0, "svm": 0.0,
    "yolo_n": 0.0, "yolo_s": 0.0, "yolo_m": 0.0,
}


def reset_run_state(domain_dir: str, dry_run: bool = False) -> dict:
    """Zero all per-run state in domain_dir/knowledge/.

    Returns a dict with before/after snapshots of event_counters so the caller
    can detect a missed reset from non-zero deltas.
    """
    knowledge = Path(domain_dir) / "knowledge"
    if not knowledge.exists():
        raise FileNotFoundError(f"Knowledge dir not found: {knowledge}")

    # Determine domain name for default EMA values
    domain_name = "cv" if "cv" in str(knowledge) else "regression"

    result: dict = {"domain_dir": str(domain_dir), "dry_run": dry_run}

    # ── mape_info.json ────────────────────────────────────────────────────────
    mape_info_path = knowledge / "mape_info.json"
    counters_before: dict = {}
    if mape_info_path.exists():
        with open(mape_info_path) as f:
            info = json.load(f)
        counters_before = dict(info.get("event_counters", {}))

        # Zero run state
        info["last_line"] = 0
        info["recovery_cycles"] = 0
        info["last_switch_ts"] = 0.0  # reset cooldown guard; epoch = no prior switch
        info["event_counters"] = {
            "model_switches": 0,
            "retrains": 0,
            "vmr_events": 0,
            "noops": 0,
            "mape_k_energy_uJ": 0.0,
        }
        info["simple_switch_counters"] = {"simple_switches": 0}

        # Reset EMA scores to 0.5 (eliminate corrupted priors from garbage sessions)
        default_emas = _EMA_DEFAULTS.get(domain_name, {})
        current_keys = list(info.get("ema_scores", default_emas).keys())
        info["ema_scores"] = {k: 0.5 for k in (current_keys or default_emas.keys())}

        # Reset separated EMA signals if present
        if "ema_accuracy" in info:
            info["ema_accuracy"] = {k: 0.0 for k in info["ema_accuracy"]}
        if "ema_energy" in info:
            info["ema_energy"] = {k: 0.0 for k in info["ema_energy"]}

        if not dry_run:
            with open(mape_info_path, "w") as f:
                json.dump(info, f, indent=4)
            print(f"✔ Reset mape_info.json (was: {counters_before})")
        else:
            print(f"[dry-run] Would reset mape_info.json (currently: {counters_before})")

    result["counters_before"] = counters_before
    result["counters_after"] = {k: 0 for k in counters_before}

    # ── predictions.csv — truncate to header only ─────────────────────────────
    predictions_path = knowledge / "predictions.csv"
    if predictions_path.exists():
        if not dry_run:
            # Read header then overwrite with header only
            with open(predictions_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    header = []
            with open(predictions_path, "w", newline="", encoding="utf-8") as f:
                if header:
                    csv.writer(f).writerow(header)
            print(f"✔ Truncated predictions.csv to header only")
        else:
            print("[dry-run] Would truncate predictions.csv to header only")

    # ── Volatile files — delete ───────────────────────────────────────────────
    # bandit_state.json is intentionally NOT reset between runs.
    # The LinUCB bandit (S7) accumulates learning across runs.
    # To reset the bandit, delete knowledge/bandit_state.json manually.
    # bandit_pending.json IS cleared so stale pending rewards don't corrupt
    # the next session's reward signal.
    _DELETE_FILES = ["command.txt", "drift.csv", "drift_kl.json", "bandit_pending.json"]
    for fname in _DELETE_FILES:
        fpath = knowledge / fname
        if fpath.exists():
            if not dry_run:
                fpath.unlink()
                print(f"✔ Deleted {fname}")
            else:
                print(f"[dry-run] Would delete {fname}")

    # ── Embedding store (CV domain only) ──────────────────────────────────────
    if domain_name == "cv":
        _reset_embedding_store(knowledge, dry_run)

    return result


def _reset_embedding_store(knowledge: Path, dry_run: bool) -> None:
    """Zero the embedding ring buffer and index if they exist (§5 embedding store)."""
    npy_path = knowledge / "embeddings.f16.npy"
    idx_path = knowledge / "embeddings_index.csv"
    if npy_path.exists() or idx_path.exists():
        if dry_run:
            print("[dry-run] Would reset embedding store (embeddings.f16.npy + index)")
            return
        # Load mape_info to discover dim/capacity; fall back to re-reading npy header
        try:
            sys.path.insert(0, str(knowledge.parent.parent))
            from core.drift.embedding_store import EmbeddingStore
            info_path = knowledge / "mape_info.json"
            dim = 256  # fallback
            window = 500
            if info_path.exists():
                with open(info_path) as f:
                    info = json.load(f)
                dim = info.get("embedding_dim", dim)
                window = info.get("drift_window_size", window)
            elif npy_path.exists():
                arr = np.load(str(npy_path))
                if arr.ndim == 2:
                    dim = arr.shape[1]
                    window = arr.shape[0] // 2
            store = EmbeddingStore(str(knowledge), embedding_dim=dim, drift_window=window)
            store.reset()
            print("✔ Reset embedding store (embeddings.f16.npy + index)")
        except Exception as exc:
            # Graceful degradation: manually zero the files
            if npy_path.exists():
                arr = np.load(str(npy_path))
                arr[:] = 0.0
                np.save(str(npy_path), arr)
            if idx_path.exists():
                idx_path.unlink()
            print(f"✔ Reset embedding store (direct zero; EmbeddingStore unavailable: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset live managed-system run state.")
    parser.add_argument("--domain", required=True, choices=["regression", "cv"],
                        help="Which domain to reset.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without making changes.")
    args = parser.parse_args()

    # Resolve relative to tool/
    tool_dir = Path(__file__).resolve().parent.parent
    domain_dir = tool_dir / _DOMAIN_DIRS[args.domain]

    result = reset_run_state(str(domain_dir), dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\nReset complete. Counter deltas from last session: {result['counters_before']}")


if __name__ == "__main__":
    main()

"""scripts/init_regression.py — One-time initialiser for the regression managed system.

Fits a MinMaxScaler on the chronological training split and saves both the scaler
and the KL reference distribution histogram to knowledge/.

Must be run once before the first inference session and again after any retrain
that changes the training window.

Usage:
    cd tool/
    python scripts/init_regression.py [--config pems_node1] [--force]

Outputs (in managed_system_regression/knowledge/):
    scaler.pkl                    — MinMaxScaler fitted on train split only (B7 fix)
    reference_distribution.json   — histogram + bin_edges for KL drift detection (B3 fix)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Allow importing from tool/
_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

_N_BINS = 50  # must match KLFixedRefDetector default


def init_regression(config_name: str = "pems_node1", force: bool = False) -> None:
    # ── Load dataset config ────────────────────────────────────────────────────
    config_path = _TOOL_DIR / "configs" / "datasets" / f"{config_name}.json"
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    if cfg.get("domain", "regression") != "regression":
        sys.exit(f"Config '{config_name}' is not a regression dataset.")

    # Resolve data path relative to tool/
    data_path = (_TOOL_DIR / cfg["data_path"]).resolve()
    value_col = cfg.get("value_column", "flow")
    train_frac = float(cfg.get("train_frac", 0.8))

    # The managed system knowledge dir is adjacent to the dataset
    knowledge_dir = data_path.parent  # managed_system_regression/knowledge/
    scaler_path = knowledge_dir / "scaler.pkl"
    ref_path = knowledge_dir / "reference_distribution.json"

    if not force and scaler_path.exists() and ref_path.exists():
        print(f"Artifacts already exist in {knowledge_dir}. Use --force to overwrite.")
        return

    # ── Load and split ─────────────────────────────────────────────────────────
    if not data_path.exists():
        sys.exit(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if value_col not in df.columns:
        sys.exit(f"Column '{value_col}' not found in {data_path}. Available: {list(df.columns)}")

    n_total = len(df)
    train_end = int(n_total * train_frac)
    train_values = df[value_col].iloc[:train_end].values.astype(float)

    print(f"Dataset: {n_total} rows. Training split: {train_end} rows ({train_frac*100:.0f}%).")

    # ── Fit MinMaxScaler on training split only (B7 fix) ───────────────────────
    scaler = MinMaxScaler()
    scaler.fit(train_values.reshape(-1, 1))

    tmp_scaler = scaler_path.with_suffix(".pkl.tmp")
    with open(tmp_scaler, "wb") as f:
        pickle.dump(scaler, f)
    os.replace(tmp_scaler, scaler_path)
    print(f"✔ Scaler fitted on {train_end} training samples → {scaler_path}")
    print(f"  feature_range: {scaler.data_min_[0]:.4f} – {scaler.data_max_[0]:.4f}")

    # ── Build reference histogram for KL drift detection (B3 fix) ─────────────
    hist, bin_edges = np.histogram(train_values, bins=_N_BINS)
    ref_data = {
        "histogram": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "n_bins": _N_BINS,
        "train_rows": int(train_end),
        "value_column": value_col,
        "config": config_name,
    }

    tmp_ref = ref_path.with_suffix(".json.tmp")
    with open(tmp_ref, "w") as f:
        json.dump(ref_data, f, indent=2)
    os.replace(tmp_ref, ref_path)
    print(f"✔ Reference distribution histogram ({_N_BINS} bins) → {ref_path}")

    print("\nSetup complete. You can now start the regression managed system.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise regression managed system artifacts.")
    parser.add_argument("--config", default="pems_node1",
                        help="Dataset config name in configs/datasets/ (default: pems_node1)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing artifacts")
    args = parser.parse_args()
    init_regression(args.config, args.force)


if __name__ == "__main__":
    main()

"""
scripts/preprocess_pems.py — Preprocess PeMS (California Performance Measurement System) data.

Supported input formats (auto-detected from column names):
  1. PeMS D03/D04/D07/D11 district downloads (semicolon or comma delimited):
       Timestamp, Station, District, Freeway, Direction, Lane Type, Station Length,
       Samples, % Observed, Total Flow, Avg Occupancy, Avg Speed
     → use "Total Flow" column

  2. Aggregated node CSV (e.g. flow_data_train.csv from original HarmonE):
       flow (single column of 5-minute vehicle counts)
     → use "flow" column directly

  3. Generic: any CSV where you specify --flow-col explicitly.

Output: <output-path> CSV with columns:
    timestamp  (optional — included if input has a recognisable datetime column)
    flow       (int or float, vehicles per 5-minute interval)

The output column is named "flow" so it is compatible with:
  - managed_system_regression/inference.py (hardcoded df["flow"])
  - configs/datasets/pems_node*.json  (value_column: "flow")
  - scripts/init_regression.py        (cfg.get("value_column", "flow"))

Usage:
    cd tool/
    python scripts/preprocess_pems.py \\
        --input  /path/to/pems_raw.csv \\
        --output data/pems_node2/pems_node2.csv \\
        [--flow-col "Total Flow"]   # override detected column name
        [--sep ,]                   # separator (default: auto-detect)
        [--no-timestamp]            # omit timestamp column from output
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


OUTPUT_DEFAULT = "data/pems_node2/pems_node2.csv"

# Candidate column names that contain 5-min flow counts, in preference order
_FLOW_CANDIDATES = [
    "Total Flow", "total_flow", "flow", "Flow",
    "total flow", "TOTAL_FLOW", "Vehicles", "volume",
    "Flow (Veh/5 Minutes)",  # PeMS single-station download format
]

# Candidate timestamp column names
_TS_CANDIDATES = [
    "Timestamp", "timestamp", "Time", "time", "DateTime", "datetime",
    "5 Minutes", "5_Minutes",
]


def _detect_sep(path: str) -> str:
    """Sniff delimiter by reading the first line."""
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline()
    if first.count(";") > first.count(","):
        return ";"
    return ","


def preprocess(
    input_path: str,
    output_path: str,
    flow_col: str | None = None,
    sep: str | None = None,
    include_timestamp: bool = True,
) -> None:
    actual_sep = sep or _detect_sep(input_path)
    print(f"Reading {input_path} (sep={repr(actual_sep)}) ...")

    df = pd.read_csv(input_path, sep=actual_sep, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns (first 12): {list(df.columns)[:12]}")

    # ── Resolve flow column ───────────────────────────────────────────────────
    if flow_col is not None:
        if flow_col not in df.columns:
            sys.exit(
                f"Flow column '{flow_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        resolved_flow = flow_col
    else:
        resolved_flow = next(
            (c for c in _FLOW_CANDIDATES if c in df.columns), None
        )
        if resolved_flow is None:
            sys.exit(
                f"Could not auto-detect flow column. "
                f"Available columns: {list(df.columns)}\n"
                f"Use --flow-col to specify explicitly."
            )
    print(f"  Using flow column: '{resolved_flow}'")

    # ── Resolve timestamp column (optional) ──────────────────────────────────
    ts_col = next((c for c in _TS_CANDIDATES if c in df.columns), None)
    if ts_col:
        print(f"  Using timestamp column: '{ts_col}'")

    # ── Extract and clean ─────────────────────────────────────────────────────
    flow = pd.to_numeric(df[resolved_flow], errors="coerce")

    n_nan = flow.isna().sum()
    if n_nan:
        print(f"  Dropping {n_nan} NaN / unparseable rows.")
    flow = flow.dropna()

    # Build output frame
    if include_timestamp and ts_col is not None:
        ts = df[ts_col].loc[flow.index]
        try:
            ts = pd.to_datetime(ts)
            ts_str = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts_str = ts.astype(str)
        out = pd.DataFrame({"timestamp": ts_str, "flow": flow.values})
    else:
        out = pd.DataFrame({"flow": flow.values})

    # Sanity checks
    if out["flow"].min() < 0:
        n_neg = (out["flow"] < 0).sum()
        print(f"  WARNING: {n_neg} negative flow values detected. Check your data.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"  Output rows: {len(out)}")
    if include_timestamp and "timestamp" in out.columns:
        print(f"  Date range: {out['timestamp'].iloc[0]}  →  {out['timestamp'].iloc[-1]}")
    print(f"  Flow range: {out['flow'].min():.1f} – {out['flow'].max():.1f} vehicles/5min")
    print(f"  Saved to: {output_path}")
    print()
    print("Next steps:")
    print(f"  1. python scripts/init_regression.py --config pems_node2  (fit scaler + KL ref)")
    print(f"  2. python scripts/validate_dataset.py configs/datasets/pems_node2.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PeMS traffic flow data.")
    parser.add_argument("--input",  required=True,
                        help="Path to raw PeMS CSV (district download or aggregated flow CSV)")
    parser.add_argument("--output", default=OUTPUT_DEFAULT,
                        help="Output CSV path (default: data/pems_node2/pems_node2.csv)")
    parser.add_argument("--flow-col", default=None,
                        help="Name of the flow column (auto-detected if omitted)")
    parser.add_argument("--sep", default=None,
                        help="CSV separator (auto-detected if omitted)")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Omit timestamp column from output (flow column only)")
    args = parser.parse_args()

    preprocess(
        args.input,
        args.output,
        flow_col=args.flow_col,
        sep=args.sep,
        include_timestamp=not args.no_timestamp,
    )


if __name__ == "__main__":
    main()

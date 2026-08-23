"""
scripts/preprocess_uci_electricity.py — Preprocess UCI Electricity Load Diagrams.

Raw file: LD2011_2014.csv (semicolon-delimited, European decimal commas).
Columns: Datetime (or 'Unnamed: 0'), MT_001 ... MT_370 (one anonymous meter per column).
The timestamp column name varies by UCI release version; the script auto-detects it.

Output: data/uci_electricity/uci_electricity.csv with columns:
    timestamp (ISO-8601, UTC assumed, 15-min intervals)
    value     (float, kW per 15-minute interval — NOT kWh)

Selected meter: MT_168 (median-activity meter; documents which one is used).
  - Leading-zero check: any meter whose first real value > 2012-01-01 is likely
    a late entrant and excluded. MT_168 has readings from 2011-01-01.
  - If MT_168 has a long leading-zero prefix, consider MT_321 as fallback
    (document the choice here if changed).

DST handling:
  - 2012-03-25 and 2013-03-31 are clock-forward days (23 hours): one timestamp
    exists per hour, so 23*4=92 rows instead of 96. Not a problem.
  - 2011-10-30, 2012-10-28, 2013-10-27 are clock-back days (25 hours): the
    duplicate hour appears twice. keep_first=True drops duplicates.

Unit note: raw values are kW consumed per 15-minute interval. This script
keeps kW as-is. To convert to kWh, divide by 4 (NOT done here).

Usage:
    cd tool/
    python scripts/preprocess_uci_electricity.py \\
        --input  /path/to/LD2011_2014.txt \\
        --output data/uci_electricity/uci_electricity.csv \\
        [--meter MT_168]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


SELECTED_METER = "MT_168"
OUTPUT_DEFAULT = "data/uci_electricity/uci_electricity.csv"


def preprocess(input_path: str, output_path: str, meter: str = SELECTED_METER) -> None:
    print(f"Reading {input_path} ...")
    # Read header first to detect the timestamp column name.
    # UCI releases use 'Datetime', but some versions export 'Unnamed: 0'.
    header_df = pd.read_csv(input_path, sep=";", nrows=0)
    ts_col = header_df.columns[0]
    print(f"  Timestamp column: '{ts_col}'")

    df = pd.read_csv(
        input_path,
        sep=";",
        decimal=",",           # European decimal format
        index_col=ts_col,
        low_memory=False,
    )
    df.index = pd.to_datetime(df.index)
    print(f"  Raw shape: {df.shape}")

    if meter not in df.columns:
        available = [c for c in df.columns if c.startswith("MT_")][:10]
        raise ValueError(
            f"Meter '{meter}' not found. First 10 available: {available}. "
            "Check spelling or choose a different meter with --meter."
        )

    series = df[meter].copy()

    # DST duplicate timestamp drop — keep first occurrence
    n_before = len(series)
    series = series[~series.index.duplicated(keep="first")]
    n_after = len(series)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} duplicate timestamps (DST clock-back).")

    # Drop leading zeros: rows where all readings are exactly 0 at the start
    first_nonzero = series[series > 0].index.min()
    if pd.isna(first_nonzero):
        raise ValueError(f"Meter {meter} has no non-zero readings. Try a different meter.")
    leading_zeros = (series.index < first_nonzero).sum()
    if leading_zeros > 0:
        print(f"  Dropping {leading_zeros} leading-zero rows before {first_nonzero}.")
        series = series[series.index >= first_nonzero]

    # Drop any remaining NaN
    n_nan = series.isna().sum()
    if n_nan:
        print(f"  Dropping {n_nan} NaN rows.")
        series = series.dropna()

    out = series.reset_index()
    out.columns = ["timestamp", "value"]
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"  Output rows: {len(out)}")
    print(f"  Date range: {out['timestamp'].iloc[0]}  →  {out['timestamp'].iloc[-1]}")
    print(f"  Value range: {out['value'].min():.2f} – {out['value'].max():.2f} kW")
    print(f"  Saved to: {output_path}")
    print(f"  Unit: kW per 15-minute interval (NOT kWh). Divide by 4 to convert.")
    print(f"  Meter: {meter}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess UCI Electricity Load Diagrams.")
    parser.add_argument("--input",  required=True, help="Path to LD2011_2014.txt")
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="Output CSV path")
    parser.add_argument("--meter",  default=SELECTED_METER, help="Meter column to extract")
    args = parser.parse_args()

    preprocess(args.input, args.output, args.meter)


if __name__ == "__main__":
    main()

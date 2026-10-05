"""
scripts/preprocess_spot_prices.py — Preprocess electricity spot price data.

Supports two source formats (auto-detected from CSV headers):

  ERCOT (recommended — public domain):
    Download: ERCOT DAM Settlement Point Prices 2019-2022
    Typical columns: "Delivery Date", "Delivery Hour",
                     "Settlement Point", "Settlement Point Price"
    Settlement point: HB_NORTH (default; document in config if changed)

  Nord Pool (restricted licensing):
    Typical columns: "HourUTC", "HourDK", "PriceArea",
                     "SpotPriceDKK", "SpotPriceEUR"
    Bidding area: DK1 (default)

Output: data/spot_prices/spot_prices.csv with columns:
    timestamp (ISO-8601 UTC, hourly)
    value     (float, price in USD/MWh for ERCOT or EUR/MWh for Nord Pool)

Structural break context:
    ERCOT: Winter Storm Uri peak ~2021-02-13. Ensure 2021+ is in the test split.
    Nord Pool: Energy crisis peak 2021-2022.
    The config splits (train_frac=0.60) put 2019-2020 in training and 2021+ in
    test/inference by design — this is the stress-test window.

Usage:
    cd tool/
    python3 scripts/preprocess_spot_prices.py \\
        --input  /path/to/ercot_prices.csv \\
        --output data/spot_prices/spot_prices.csv \\
        [--settlement-point HB_NORTH]     # ERCOT only
        [--area DK1]                       # Nord Pool only
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


OUTPUT_DEFAULT = "data/spot_prices/spot_prices.csv"


def _detect_format(df: pd.DataFrame) -> str:
    cols = {c.lower() for c in df.columns}
    if "hourutc" in cols or "spotpriceeur" in cols:
        return "nordpool"
    if "delivery date" in cols or "delivery hour" in cols:
        return "ercot"
    raise ValueError(
        f"Cannot detect source format. Columns found: {list(df.columns)}\n"
        "Expected ERCOT: 'Delivery Date', 'Delivery Hour', 'Settlement Point Price'\n"
        "Expected Nord Pool: 'HourUTC', 'PriceArea', 'SpotPriceEUR'"
    )


def _preprocess_ercot(df: pd.DataFrame, settlement_point: str) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    print(f"  Detected format: ERCOT")

    # Filter to the chosen settlement point if column present
    if "Settlement Point" in df.columns:
        available = df["Settlement Point"].unique().tolist()
        if settlement_point not in available:
            print(f"  WARNING: '{settlement_point}' not in dataset. Available: {available[:10]}")
            print(f"  Using first available: {available[0]}")
            settlement_point = available[0]
        df = df[df["Settlement Point"] == settlement_point].copy()
        print(f"  Settlement point: {settlement_point}")

    # Build timestamp from date + hour columns
    date_col = next(c for c in df.columns if "delivery date" in c.lower())
    hour_col = next(c for c in df.columns if "delivery hour" in c.lower())
    price_col = next(c for c in df.columns if "settlement point price" in c.lower())

    df["timestamp"] = (
        pd.to_datetime(df[date_col])
        + pd.to_timedelta(df[hour_col].astype(int) - 1, unit="h")
    )
    df = df.rename(columns={price_col: "value"})[["timestamp", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _preprocess_nordpool(df: pd.DataFrame, area: str) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    print(f"  Detected format: Nord Pool")

    # Find area column
    area_col = next((c for c in df.columns if c.lower() in ("pricearea", "area")), None)
    if area_col is not None:
        available = df[area_col].unique().tolist()
        if area not in available:
            print(f"  WARNING: area '{area}' not found. Available: {available}")
            area = available[0]
            print(f"  Using: {area}")
        df = df[df[area_col] == area].copy()
        print(f"  Bidding area: {area}")

    time_col = next(c for c in df.columns if "hourutc" in c.lower())
    price_col = next(c for c in df.columns if "spotpriceeur" in c.lower())

    df = df.rename(columns={time_col: "timestamp", price_col: "value"})[["timestamp", "value"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _read_csv_auto(path: str) -> pd.DataFrame:
    """Read CSV, auto-detecting separator and decimal character.

    Nord Pool exports use semicolons and European decimal commas (e.g. "1,95").
    ERCOT exports use commas and US decimal points.
    Try comma first; if column detection fails, retry with semicolon + decimal=','.
    """
    df = pd.read_csv(path, low_memory=False)
    cols = {c.lower() for c in df.columns}
    known = {"hourutc", "spotpriceeur", "delivery date", "delivery hour",
             "settlement point price"}
    if not cols.intersection(known):
        # No known column found with comma separator — try semicolon + European decimal
        df = pd.read_csv(path, sep=";", decimal=",", low_memory=False)
    return df


def preprocess(
    input_path: str,
    output_path: str,
    settlement_point: str = "HB_NORTH",
    area: str = "DK1",
) -> None:
    print(f"Reading {input_path} ...")
    df = _read_csv_auto(input_path)
    print(f"  Raw columns: {list(df.columns)}")
    print(f"  Raw rows: {len(df)}")

    fmt = _detect_format(df)
    if fmt == "ercot":
        out = _preprocess_ercot(df, settlement_point)
    else:
        out = _preprocess_nordpool(df, area)

    # Sort, deduplicate, drop NaN
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    n_nan = out["value"].isna().sum()
    if n_nan:
        print(f"  Dropping {n_nan} NaN price rows.")
        out = out.dropna(subset=["value"])

    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"  Output rows: {len(out)}")
    print(f"  Date range: {out['timestamp'].iloc[0]}  →  {out['timestamp'].iloc[-1]}")
    print(f"  Value range: {out['value'].min():.2f} – {out['value'].max():.2f}")
    print(f"  Saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess electricity spot price data.")
    parser.add_argument("--input",  required=True, help="Path to raw CSV file")
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="Output CSV path")
    parser.add_argument("--settlement-point", default="HB_NORTH",
                        help="ERCOT settlement point (default: HB_NORTH)")
    parser.add_argument("--area", default="DK1",
                        help="Nord Pool bidding area (default: DK1)")
    args = parser.parse_args()

    preprocess(args.input, args.output, args.settlement_point, args.area)


if __name__ == "__main__":
    main()

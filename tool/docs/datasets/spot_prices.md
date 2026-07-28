# R3 — Electricity Spot Prices (ERCOT / Nord Pool)

## Purpose in the study

Electricity day-ahead spot prices over a crisis window (energy price spike
2021–2022) represent a *structural break* — a permanent regime shift where
the distribution never reverts to pre-crisis levels. HarmonE is EXPECTED to
partially fail here: the VMR will accumulate versions but none will match the
new distribution, causing repeated retrains without lasting recovery. This is
a designed stress test that validates HarmonE's honest failure mode and
demonstrates the boundaries of the VMR mechanism.

Document this outcome explicitly in experiment reports: "HarmonE's VMR cannot
help when drift is structural; it falls back to retrain every N cycles."

## Task & models

**Domain:** regression  
**Target:** hourly day-ahead price in USD/MWh (ERCOT) or EUR/MWh (Nord Pool)  
**Models:** LSTM, Ridge, SVR  
**Adapter:** `adapters.regression_csv.RegressionCSVAdapter`

## Primary source: ERCOT (public)

ERCOT (Electric Reliability Council of Texas) day-ahead price data is freely
downloadable without registration. Recommended download:

1. Go to https://www.ercot.com/gridinfo/load/load_hist
2. Download "DAM Settlement Point Prices" for 2019–2022 (XLS or CSV)
3. Filter for a single settlement point, e.g., `HB_NORTH` (hub average)
4. Period: 2019-01-01 to 2022-12-31 covers pre-crisis, crisis onset (Winter
   Storm Uri Feb 2021), and post-crisis stabilisation

## Secondary source: Nord Pool (restricted)

Nord Pool day-ahead prices (Elspot) are available at
https://www.nordpoolgroup.com/en/Market-data1/Power-system-data/ but require
commercial licensing for systematic use. Use ERCOT as the default; if Nord
Pool data is obtained through institutional access, document the license
reference in your run notes.

## Required PREPROCESSED form

| Column | Type | Notes |
|--------|------|-------|
| `timestamp` | ISO-8601 (hourly) | monotonically increasing |
| `value` | float64 | price in USD/MWh; no NaN |

**Preprocessing recipe (ERCOT):**
```python
import pandas as pd

df = pd.read_csv("ercot_dam_prices_2019_2022.csv", parse_dates=["Delivery Date"])
# Assumes columns: "Delivery Date", "Delivery Hour", "Settlement Point Price"
df["timestamp"] = pd.to_datetime(df["Delivery Date"]) + pd.to_timedelta(df["Delivery Hour"] - 1, unit="h")
df = df.rename(columns={"Settlement Point Price": "value"})
df = df[["timestamp", "value"]].dropna().sort_values("timestamp").reset_index(drop=True)
df.to_csv("data/spot_prices/spot_prices.csv", index=False)
```

## Drift-stream construction

Split chronologically. Recommended: 2019–2020 train (~60%), 2021-H1 val (~15%),
2021-H2+ stream (includes Uri spike and aftermath). The structural break occurs
at approximately 2021-02-13 (Winter Storm Uri).

## Config skeleton

`configs/datasets/spot_prices.json` — `"status": "awaiting_data"`.

Key fields: `data_path`, `value_column: "value"`, `seq_length: 24`
(24-hour look-back).

## Init & calibration

```bash
cd tool/
python scripts/init_regression.py --config spot_prices
python experiments/calibrate_energy_bounds.py --config spot_prices
python experiments/calibrate_drift_threshold.py --config spot_prices
```

## Offline label availability

Not applicable.

## Expected outcome

After the structural break (~2021-02-13), expect:
- Repeated drift detections
- VMR version-match scores above τ_drift for all stored versions
- Repeated retrains without recovery
- HarmonE score chronically below `min_score`

This is the correct behaviour. It validates the fail-honest property.

## License / registration

ERCOT data: public domain, no registration required.
Nord Pool: commercial licensing required for systematic use.

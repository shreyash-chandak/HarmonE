# R1 — PeMS Node 2

## Purpose in the study

Second PeMS traffic-flow sensor to evaluate HarmonE's regression domain
across sensors with different traffic character. While `pems_node1` exhibits
moderate recurring daily/weekly drift, `pems_node2` targets a sensor in a
higher-variance corridor to stress-test the adaptive energy threshold (Eq. 3)
and verify that VMR version-reuse triggers on the same weekly periodicity
observed in node1.

## Task & models

**Domain:** regression  
**Target:** occupancy / flow value  
**Models:** LSTM (baseline), Ridge (light), SVR (medium)  
**Adapter:** `adapters.regression_csv.RegressionCSVAdapter`

## Expected RAW form

Download from PeMS (https://pems.dot.ca.gov/ → Data → Station 5-Minute →
Export). Any station that provides 5-minute interval readings works. The raw
export is a multi-column CSV with headers such as:

```
Timestamp, Station, District, Freeway, Direction, Lane Type, Station Length,
Samples, % Observed, Total Flow, Avg Occupancy, Avg Speed, ...
```

## Required PREPROCESSED form

A single CSV file with at minimum:

| Column | Type | Notes |
|--------|------|-------|
| `timestamp` | string ISO-8601 or parseable by `pd.to_datetime` | monotonically increasing |
| `flow` | float64 | the flow reading (vehicles/5-min); no NaN |

The column is named `flow` (not `value`) so it matches the original training code
in `managed_system_regression/inference.py` and `train.py`. The config key
`"value_column": "flow"` in `configs/datasets/pems_node2.json` reflects this.
Minimum row count: 5 000.

**Preferred: use the preprocess script:**
```bash
cd tool/
python scripts/preprocess_pems.py \
    --input /path/to/pems_raw.csv \
    --output data/pems_node2/pems_node2.csv
```

**Manual recipe (if needed):**
```python
import pandas as pd
df = pd.read_csv("raw_pems_stationXXX.csv", parse_dates=["Timestamp"])
df = df.rename(columns={"Timestamp": "timestamp", "Total Flow": "flow"})
df = df[["timestamp", "flow"]].dropna().sort_values("timestamp").reset_index(drop=True)
df.to_csv("data/pems_node2/pems_node2.csv", index=False)
```

## Drift-stream construction

Split chronologically: first 80% train, next 10% val, remaining 10% stream.
Recurring daily/weekly drift is inherent in 5-minute interval readings.
No shuffling. Set `stream_delay_s: 0.0` for fast experiment mode.

## Config skeleton

`configs/datasets/pems_node2.json` — `"status": "awaiting_data"` until downloaded.

Key fields to fill:
- `data_path`: relative path to the preprocessed CSV
- `value_column`: `"flow"` (matches output of `preprocess_pems.py`)
- `seq_length`: `5` (default)
- `train_frac`: `0.8`

## Init & calibration

After download and preprocessing:
```bash
cd tool/
python scripts/init_regression.py --config pems_node2
python experiments/calibrate_energy_bounds.py --config pems_node2
python experiments/calibrate_drift_threshold.py --config pems_node2
```

## Offline label availability

Not applicable — regression target is the `value` column itself.
Proxy validation uses the lag-1 prediction error.

## License / registration

PeMS data requires a free Caltrans account registration. Academic use is
permitted; redistribution of raw data is not. Preprocessed CSV is a
derived work — check Caltrans terms before sharing.

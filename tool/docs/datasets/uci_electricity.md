# R2 — UCI Electricity Load Diagrams 2011–2014

## Purpose in the study

UCI Electricity Load Diagrams 2011–2014 provides a long, seasonal regression
stream. The recurring seasonal drift (winter peaks / summer troughs) is the
canonical showcase for HarmonE's VMR version-reuse: a model trained on a
summer window should be retrievable and reused the following summer rather
than triggering a fresh retrain. Demonstrates that the VMR signature matching
correctly ranks the prior summer version as the best match.

## Task & models

**Domain:** regression  
**Target:** electricity load in kW  
**Models:** LSTM (baseline), Ridge (light), SVR (medium)  
**Adapter:** `adapters.regression_csv.RegressionCSVAdapter`

## Expected RAW form

Download `LD2011_2014.txt` from UCI ML Repository
(https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014).

Raw format: semicolon-delimited, first column is timestamp, remaining 370
columns are individual Portuguese household client IDs (MT_001 … MT_370).
Values are in kW per 15-minute interval.

```
;MT_001;MT_002;...;MT_370
2011-01-01 00:15:00;0,000;0,000;...;0,000
...
```

Note: comma is the decimal separator (European locale); values must be
converted. Timestamps in UTC+0 / WET — Portuguese DST creates duplicate
timestamps twice per year (last Sunday of March and last Sunday of October).
Handle by dropping the ambiguous duplicate row before resampling.

## Required PREPROCESSED form

Single client, single `value` column CSV:

| Column | Type | Notes |
|--------|------|-------|
| `timestamp` | ISO-8601 string | monotonically increasing after DST dedup |
| `value` | float64 | kW per 15-min interval; no NaN |

**Why MT_168?** MT_168 is the client with the most complete record (fewest
missing intervals) and a clear seasonal pattern suitable for VMR reuse.
Any client may be substituted; document the choice in your run notes.

**Preprocessing recipe:**
```python
import pandas as pd

df = pd.read_csv(
    "LD2011_2014.txt", sep=";", index_col=0,
    parse_dates=True, decimal=","
)
# Pick client MT_168
s = df["MT_168"].dropna()
# Remove DST duplicate timestamps (keep first occurrence)
s = s[~s.index.duplicated(keep="first")]
# Rename and output
out = s.reset_index()
out.columns = ["timestamp", "value"]
out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
out.to_csv("data/uci_electricity/uci_electricity.csv", index=False)
```

## Drift-stream construction

Split chronologically: 2011–2013 train (~80%), 2014-Q1 val (~5%), 2014-Q2+ stream.
Seasonal drift is inherent in the chronological order. No shuffling.

## Config skeleton

`configs/datasets/uci_electricity.json` — `"status": "awaiting_data"`.

Key fields: `data_path`, `value_column: "value"`, `seq_length: 96`
(96 × 15 min = 24 h look-back window).

## Init & calibration

```bash
cd tool/
python scripts/init_regression.py --config uci_electricity
python experiments/calibrate_energy_bounds.py --config uci_electricity
python experiments/calibrate_drift_threshold.py --config uci_electricity
```

## Offline label availability

Not applicable — target is the `value` column.

## License / registration

UCI ML Repository: CC BY 4.0. Attribution required. No registration needed.
Original source: "Electricity Load Diagrams 2011–2014", Trindade (2015).

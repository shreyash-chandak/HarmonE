# HarmonE — Dataset Preparation & Experiment Runbook

Complete instructions for preprocessing all six datasets, running experiments,
computing CV accuracy metrics, and capturing timestamped logs.
All commands run from `tool/` unless stated otherwise.

---

## Quick-reference: order of operations per dataset

| Step | Command | All datasets |
|------|---------|-------------|
| 0 | Activate env | `source harmone_env/bin/activate` |
| 1 | Preprocess raw → canonical CSV / manifest | `python scripts/preprocess_<dataset>.py …` |
| 2 | Fit scaler + KL reference (regression only) | `python scripts/init_regression.py --config <name>` |
| 3 | Fit embedding reference (CV, embedding detectors only) | `python scripts/init_cv.py --config <name>` |
| 4 | Validate placement & schema | `python scripts/validate_dataset.py configs/datasets/<name>.json` |
| 5 | Train models | see per-dataset section |
| 6 | Run experiment | `python experiments/run_experiment.py --dataset <name> …` |
| 7 | Compute CV accuracy (CV only) | `python experiments/offline_eval.py …` |
| 8 | Log readings | `python scripts/experiment_logger.py --knowledge … --dataset …` |

---

## Directory layout (all paths relative to `tool/`)

```
tool/
├── data/
│   ├── pems_node2/           ← R1 preprocessed CSV
│   ├── pems_node1/           ← R1b (same format, second node)
│   ├── uci_electricity/      ← R2 preprocessed CSV
│   └── spot_prices/          ← R3 preprocessed CSV
│
├── managed_system_cv/
│   └── data/
│       ├── bdd100k/
│       │   ├── images/
│       │   │   ├── test/     ← C1 test images (already placed)
│       │   │   ├── train/    ← C1 train images (full download)
│       │   │   └── val/      ← C1 val images (full download)
│       │   ├── labels/       ← C1 BDD100K JSON annotation files
│       │   └── labels_yolo/  ← C1 converted YOLO .txt labels (generated)
│       ├── iwildcam/
│       │   └── images/       ← C2 flat image dir (UUID filenames)
│       └── acdc/             ← C3 (symlink or copy; see below)
│
├── data/
│   ├── bdd100k/
│   │   └── bdd100k_manifest.csv   ← C1 generated manifest
│   ├── iwildcam/
│   │   └── iwildcam_manifest.csv  ← C2 generated manifest
│   └── acdc/
│       └── acdc_manifest.csv      ← C3 generated manifest
│
├── configs/datasets/         ← one .json per dataset (already exist)
├── scripts/                  ← preprocess_*.py, init_*.py, experiment_logger.py
├── experiments/              ← offline_eval.py, run_experiment.py, run_grid.py
└── logs/                     ← timestamped experiment log CSVs (auto-created)
```

---

## R1 — PeMS node2 (traffic flow regression)

### Raw format

Multi-column CSV from the PeMS district download portal. The column you need is
`Total Flow` (vehicles per 5-minute interval). Separator may be comma or
semicolon; the preprocess script detects it automatically.

```
Timestamp, Station, District, Freeway, Direction, Lane Type, Station Length,
Samples, % Observed, Total Flow, Avg Occupancy, Avg Speed, …
```

If your raw export already has only a `flow` column (e.g. the aggregated
`flow_data_train.csv` from the original codebase), the script handles that too.

### Step 1 — Preprocess

```bash
cd tool/
python scripts/preprocess_pems.py \
    --input /path/to/pems_raw.csv \
    --output data/pems_node2/pems_node2.csv
```

Output: `data/pems_node2/pems_node2.csv` with columns `timestamp, flow`.

For the second PeMS node (pems_node1, same format):
```bash
python scripts/preprocess_pems.py \
    --input /path/to/pems_node1_raw.csv \
    --output data/pems_node1/pems_node1.csv
```

Override the detected column if yours is named differently:
```bash
python scripts/preprocess_pems.py \
    --input /path/to/pems_raw.csv \
    --output data/pems_node2/pems_node2.csv \
    --flow-col "Total Flow"      # explicit, or "flow", "volume", etc.
```

### Step 2 — Init (scaler + KL reference)

```bash
python scripts/init_regression.py --config pems_node2
# Also for node1:
python scripts/init_regression.py --config pems_node1
```

Writes to `managed_system_regression/knowledge/`:
- `scaler.pkl` — MinMaxScaler fitted on training split only (B7 fix)
- `reference_distribution.json` — 50-bin histogram for KL drift detection (B3 fix)

### Step 3 — Validate

```bash
python scripts/validate_dataset.py configs/datasets/pems_node2.json
```

Expected: `PASS`. Status field in the config will be updated automatically.

### Step 4 — Train models

```bash
cd managed_system_regression/
python train.py
# Outputs: models/lstm.pth, models/linear.pkl, models/svm.pkl
cd ..
```

> The trained models are for the live managed system. The headless experiment
> harness (`run_experiment.py`) trains its own models per run.

### Step 5 — Run headless experiment

```bash
python experiments/run_experiment.py \
    --dataset pems_node2 \
    --planner harmone_original \
    --seed 42
```

For all 7 planners × 5 seeds (paper grid):
```bash
python experiments/run_grid.py configs/experiments/baseline.yaml
# Edit baseline.yaml first: datasets: [pems_node2], seeds: [1,2,3,4,5]
```

### Live managed system (dashboard mode)

```bash
# Place preprocessed CSV as dataset.csv for the live inference engine:
cp data/pems_node2/pems_node2.csv managed_system_regression/knowledge/dataset.csv

# Start ACP + inference + dashboard (three terminals):
python app.py                                           # terminal 1 — port 5000
python run_managed_system.py                            # terminal 2 — port 8080
python -m http.server 8000 --directory frontend/       # terminal 3 — port 8000
```

Open http://localhost:8000/dashboard.html → HarmonE (Regression) → choose planner.

---

## R2 — UCI Electricity Load Diagrams

### Raw format

`LD2011_2014.txt` — semicolon-delimited, European decimal commas, 370 anonymous
meter columns (`MT_001` … `MT_370`). ~1.4 million rows at 15-minute intervals.

```
"Datetime";"MT_001";"MT_002";…;"MT_370"
"2011-01-01 00:15:00";"0,000";"0,000";…;"0,000"
```

### Step 1 — Preprocess

```bash
python scripts/preprocess_uci_electricity.py \
    --input /path/to/LD2011_2014.txt \
    --output data/uci_electricity/uci_electricity.csv
```

Output: `timestamp, value` (kW per 15-minute interval, meter MT_168 by default).

If MT_168 has a long leading-zero prefix (≥ 6 months of zeros before first
non-zero reading), switch to the fallback meter:
```bash
python scripts/preprocess_uci_electricity.py \
    --input /path/to/LD2011_2014.txt \
    --output data/uci_electricity/uci_electricity.csv \
    --meter MT_321
```
Document which meter was chosen — it affects the VMR reuse showcase (R2's
seasonal periodicity is the key drift signal).

### Step 2 — Init

```bash
python scripts/init_regression.py --config uci_electricity
```

### Step 3 — Validate

```bash
python scripts/validate_dataset.py configs/datasets/uci_electricity.json
```

### Step 4 — Train & run

```bash
cd managed_system_regression/ && python train.py && cd ..

python experiments/run_experiment.py \
    --dataset uci_electricity \
    --planner harmone_original \
    --seed 42
```

### Live managed system

```bash
cp data/uci_electricity/uci_electricity.csv managed_system_regression/knowledge/dataset.csv
# Update knowledge/thresholds.json: set "value_column": "value"
```

> When switching between PeMS (flow) and UCI/spot_prices (value) in the live
> system, update `"value_column"` in `knowledge/thresholds.json` to match the
> column name in `dataset.csv`. The headless harness reads `value_column` from
> the dataset config file automatically.

---

## R3 — Spot Prices (Nord Pool)

### Raw format

Nord Pool semicolon-delimited CSV with European decimal commas. Multiple
bidding areas per row.

```
HourUTC;HourDK;PriceArea;SpotPriceDKK;SpotPriceEUR
2024-02-22 22:00;2024-02-22 23:00;DK1;14,540000;1,950000
2024-02-22 22:00;2024-02-22 23:00;DK2;14,540000;1,950000
2024-02-22 22:00;2024-02-22 23:00;SE3;14,540000;1,950000
```

> **Bug fixed:** the original `preprocess_spot_prices.py` read with default
> comma separator, failing to detect Nord Pool's semicolons. The script now
> auto-detects the separator and handles European decimal commas.

### Step 1 — Preprocess

```bash
python scripts/preprocess_spot_prices.py \
    --input /path/to/nordpool_data.csv \
    --output data/spot_prices/spot_prices.csv \
    --area DK1
```

`--area` accepts any bidding zone present in your file (DK1, DK2, SE3, NO2,
etc.). The script prints available zones if the specified one is not found.
Output: `timestamp, value` (EUR/MWh, hourly).

### Step 2 — Init

```bash
python scripts/init_regression.py --config spot_prices
```

The `train_frac: 0.60` in the config puts 2019–2020 in training and 2021+ in
the test/inference window. This is intentional — Winter Storm Uri (~2021-02-13
for ERCOT) and the Nord Pool energy crisis (2021–2022) are the structural-break
events being tested.

### Step 3 — Validate & run

```bash
python scripts/validate_dataset.py configs/datasets/spot_prices.json

python experiments/run_experiment.py \
    --dataset spot_prices \
    --planner harmone_original \
    --seed 42
```

> **Expected behaviour:** HarmonE will fail to adapt on R3. The structural
> break is permanent — VMR reuse cannot help because no historical model matches
> the post-break distribution. This is the designed stress test (RQ6). Record
> it as a partial failure, not a bug.

---

## C1 — BDD100K (object detection)

### Raw format

Full BDD100K dataset download. Required structure:

```
bdd100k/
├── images/
│   └── 100k/
│       ├── train/   *.jpg   (70 000 images)
│       ├── val/     *.jpg   (10 000 images)
│       └── test/    *.jpg   (20 000 images — no GT labels)
└── labels/
    ├── bdd100k_labels_images_train.json
    └── bdd100k_labels_images_val.json
```

The test images you already have at
`managed_system_cv/data/bdd100k/images/test/` are correctly placed. The
preprocess script reads them from there via the manifest.

### Step 1 — Preprocess (build drift-ordered manifest + YOLO labels)

```bash
python scripts/preprocess_bdd100k.py \
    --bdd-root /path/to/bdd100k/ \
    --output data/bdd100k/bdd100k_manifest.csv \
    --yolo-labels-dir managed_system_cv/data/bdd100k/labels_yolo/
```

This does two things:
1. Reads annotation JSONs to determine `weather` + `timeofday` for each image
2. Assigns domain labels and sorts into drift order:
   `clear_day → overcast → foggy → dusk → night → rain`
3. Converts BDD100K bounding-box JSON annotations to per-image YOLO `.txt`
   files in `--yolo-labels-dir` (needed for `offline_eval.py`)

Output: `data/bdd100k/bdd100k_manifest.csv` with columns:
`sample_id, input_path, label_path, domain, split`

> The test split has no GT labels in BDD100K. For ground-truth evaluation
> (`offline_eval.py`), use the `val` split — it has 10 000 labelled images
> covering all domains.

If you only need the manifest (no YOLO conversion yet):
```bash
python scripts/preprocess_bdd100k.py \
    --bdd-root /path/to/bdd100k/ \
    --output data/bdd100k/bdd100k_manifest.csv \
    --no-yolo
```

### Step 2 — Init CV artifacts

```bash
python scripts/init_cv.py --config bdd100k
```

Writes to `managed_system_cv/knowledge/`:
- `model.csv` — default active model (`yolo_n`)
- `mape_info.json` — initial MAPE-K state
- `versionedMR/` — copies of base model weights for VMR reuse
- `reference_embeddings.npz` — fixed-reference embeddings for MMD/Fréchet
  drift detection (only if `drift_detector` is embedding-based)

For the default `luminance_kl` detector, use:
```bash
python scripts/init_cv.py --config bdd100k --skip-embeddings
```

### Step 3 — Validate

```bash
python scripts/validate_dataset.py configs/datasets/bdd100k.json
```

### Step 4 — Run headless experiment

```bash
python experiments/run_experiment.py \
    --dataset bdd100k \
    --planner harmone_original \
    --seed 42
```

### Step 5 — CV accuracy (mAP@0.5)

After a run completes (or while it runs):

```bash
python experiments/offline_eval.py \
    --run-dir managed_system_cv/knowledge \
    --labels-dir managed_system_cv/data/bdd100k/labels_yolo/val \
    --images-dir managed_system_cv/data/bdd100k/images/val \
    --interval 1000 \
    --output results/bdd100k_eval.json
```

Output JSON structure:
```json
{
  "interval_results": [
    {"interval_idx": 0, "map50": 0.42, "n_images": 1000,
     "model": "yolo_n", "timestamp_start": 1234, "timestamp_end": 1235}
  ],
  "overall_map50": 0.44,
  "n_intervals": 5
}
```

The evaluation runs Ultralytics `val()` per monitoring interval, so it requires
GPU access. Add `--smoke` for a 3-interval sanity check on CPU.

### Live managed system

```bash
# Same three-terminal setup; choose CV HarmonE in dashboard
python app.py
python run_managed_system.py
python -m http.server 8000 --directory frontend/
```

In dashboard: HarmonE (CV) → choose planner → Start.

---

## C2 — iWildCam (classification)

### Raw format

WILDS v2.0 download with:

```
iwildcam_v2.0/
├── metadata.csv     (one row per image; split, location_remapped, y, image_id, filename, …)
├── categories.csv   (y → species name mapping)
└── images/
    ├── 97f407ac-21bc-11ea-a13a-137349068a90.jpg
    ├── 954d7740-21bc-11ea-a13a-137349068a90.jpg
    └── …            (flat directory, UUID filenames)
```

Your metadata excerpt uses string splits (`id_test`, `train`, etc.) from WILDS
≥2.0. Older versions used integer splits (0=train, 1=id_val, etc.).

> **Bug fixed:** the original `preprocess_iwildcam.py` called `int(row["split"])`
> which crashed on WILDS ≥2.0 string splits. Both formats are now handled.

### Step 1 — Preprocess

```bash
python scripts/preprocess_iwildcam.py \
    --wilds-root /path/to/iwildcam_v2.0/ \
    --output data/iwildcam/iwildcam_manifest.csv
```

Output: `data/iwildcam/iwildcam_manifest.csv` with columns:
`sample_id, input_path, label, domain, split`

The script:
- Uses `location_remapped` as the domain column (geographic drift axis)
- Sorts by location to create a controlled drift sequence
- The OOD test split (`id_test` / split=3 in older format) is the primary
  inference stream — unseen camera trap locations → geographic domain shift

Print a split summary to verify:
```bash
python -c "
import pandas as pd
m = pd.read_csv('data/iwildcam/iwildcam_manifest.csv')
print(m.groupby('split').size())
print(f\"OOD test: {len(m[m.split=='id_test'])} images across "
      f\"{m[m.split=='id_test']['domain'].nunique()} locations\")
"
```

### Step 2 — Init CV artifacts

```bash
python scripts/init_cv.py --config iwildcam --skip-embeddings
# Add --force to overwrite existing artifacts
```

For embedding drift (MMD/Fréchet) — needed for RQ2:
```bash
python scripts/init_cv.py --config iwildcam
# Extracts reference embeddings from training split (requires model weights)
```

### Step 3 — Validate

```bash
python scripts/validate_dataset.py configs/datasets/iwildcam.json
```

### Step 4 — Run

```bash
python experiments/run_experiment.py \
    --dataset iwildcam \
    --planner harmone_original \
    --seed 42
```

### Step 5 — CV accuracy (top-1 classification)

iWildCam ground truth is the `label` column in the manifest (species class
index, 0-indexed). The framework logs `proxy_score` (confidence-based) to
`predictions.csv` during inference. True top-1 accuracy requires offline eval:

```bash
python experiments/offline_eval.py \
    --run-dir managed_system_cv/knowledge \
    --manifest data/iwildcam/iwildcam_manifest.csv \
    --split id_test \
    --interval 500 \
    --output results/iwildcam_eval.json
```

The classification adapter's `offline_accuracy()` method computes top-1 against
the `label` column for each monitoring interval.

---

## C3 — ACDC (semantic segmentation)

### Raw format

ACDC dataset download. Required structure:

```
acdc/
├── rgb_anon/
│   ├── fog/
│   │   ├── train/{sequence}/*_rgb_anon.png
│   │   ├── val/{sequence}/*_rgb_anon.png
│   │   ├── test/{sequence}/*_rgb_anon.png
│   │   ├── trainref/{sequence}/*_rgb_anon.png   ← ignored by preprocess
│   │   ├── valref/{sequence}/*_rgb_anon.png     ← ignored
│   │   └── testref/{sequence}/*_rgb_anon.png    ← ignored
│   ├── rain/  (same structure)
│   ├── night/ (same structure)
│   └── snow/  (same structure)
└── gt_trainval/
    └── gt/
        ├── fog/
        │   ├── train/{sequence}/*_gt_labelTrainIds.png
        │   ├── val/{sequence}/*_gt_labelTrainIds.png
        │   ├── trainref/{sequence}/…
        │   └── valref/{sequence}/…
        ├── rain/  (same structure)
        ├── night/ (same structure)
        └── snow/  (same structure)
```

Your download has GT for `train`, `val`, `trainref`, `valref` — the script uses
`train` and `val` only; `trainref`/`valref`/`testref` are automatically skipped.

**Critical:** use `_gt_labelTrainIds.png` files, NOT `_gt_labelIds.png`.
Cityscapes trainId encoding: 0–18 valid classes, 255 = `ignore_index`.

### Step 1 — Preprocess

```bash
python scripts/preprocess_acdc.py \
    --acdc-root /path/to/acdc/ \
    --output data/acdc/acdc_manifest.csv
```

The preprocess script:
- Pairs each RGB image with its `_gt_labelTrainIds.png` mask
- Assigns domain from directory name (`fog`, `rain`, `night`, `snow`)
- Orders drift stream: `fog → rain → night → snow` (no "clear" condition in ACDC)

Output: `data/acdc/acdc_manifest.csv` with columns:
`sample_id, input_path, label_path, domain, split`

Verify label pairing:
```bash
python -c "
import pandas as pd
from pathlib import Path
m = pd.read_csv('data/acdc/acdc_manifest.csv')
missing = m[~m.label_path.apply(lambda p: Path(p).exists())]
print(f'Missing labels: {len(missing)} / {len(m)}')
print(m.groupby([\"split\",\"domain\"]).size())
"
```

### Step 2 — Init CV artifacts

```bash
python scripts/init_cv.py --config acdc --skip-embeddings
```

### Step 3 — Validate

```bash
python scripts/validate_dataset.py configs/datasets/acdc.json
```

### Step 4 — Run

```bash
python experiments/run_experiment.py \
    --dataset acdc \
    --planner harmone_original \
    --seed 42
```

### Step 5 — CV accuracy (mIoU)

GT is the `label_path` column (`_gt_labelTrainIds.png` masks, Cityscapes trainIds,
`ignore_index=255`).

```bash
python experiments/offline_eval.py \
    --run-dir managed_system_cv/knowledge \
    --manifest data/acdc/acdc_manifest.csv \
    --split val \
    --interval 400 \
    --output results/acdc_eval.json
```

The segmentation adapter's `offline_accuracy()` computes per-class IoU using
`ignore_index=255`, then macro-averages over the 19 Cityscapes classes.

---

## CV accuracy metrics — summary

| Dataset | Task | Metric | GT source | Script |
|---------|------|--------|-----------|--------|
| BDD100K | Detection | mAP@0.5 | YOLO `.txt` label files from preprocess | `offline_eval.py --labels-dir` |
| iWildCam | Classification | Top-1 accuracy | `label` column in manifest | `offline_eval.py --manifest` |
| ACDC | Segmentation | mIoU (19 classes, ignore=255) | `_gt_labelTrainIds.png` in manifest | `offline_eval.py --manifest` |

All three share the same `offline_eval.py` entry point. The script auto-detects
task type from the dataset config (`"task": "detection"` / `"classification"` /
`"segmentation"`).

For proxy validation (RQ3 — how well does the runtime proxy track GT accuracy):

```bash
python experiments/proxy_validation.py \
    --eval-results results/bdd100k_eval.json \
    --run-dir managed_system_cv/knowledge \
    --output results/bdd100k_proxy_validation.json
```

Reports Spearman ρ between proxy score and ground-truth accuracy per monitoring
interval. Run once per proxy choice (`confidence`, `calibrated_confidence`,
`agreement`) to pick the best proxy for the paper.

---

## Experiment logger (timestamped readings)

The logger polls the `knowledge/` directory and appends one CSV row per tick
to `logs/<dataset>_<YYYY-MM-DD_HH-MM-SS>.csv`. Run it in a separate terminal
alongside the managed system or headless experiment.

### Regression

```bash
# From tool/
python scripts/experiment_logger.py \
    --knowledge managed_system_regression/knowledge \
    --dataset pems_node2 \
    --domain regression \
    --interval 30 \
    --log-dir logs/
```

Captured columns per tick (every 30 s by default):

| Column | Description |
|--------|-------------|
| `wall_clock_utc` | ISO timestamp of this snapshot |
| `unix_ts` | Unix epoch (for alignment with predictions.csv) |
| `current_model` | Active model name (lstm / linear / svm) |
| `last_mape_line` | Row index in predictions.csv at snapshot time |
| `current_energy_threshold` | Dynamic τ_E value (Eq. 3) |
| `ema_score_lstm/linear/svm` | Per-model EMA composite score |
| `ema_acc_lstm/linear/svm` | Per-model EMA accuracy |
| `ema_energy_lstm/linear/svm` | Per-model EMA energy |
| `event_model_switches` | Cumulative model switch count |
| `event_retrains` | Cumulative retrain count |
| `event_vmr` | Cumulative VMR reuse count |
| `event_noops` | Cumulative noop count |
| `mape_k_energy_uJ` | Cumulative MAPE-K overhead energy (µJ) |
| `recent_r2_mean` | Mean R² over last 200 predictions |
| `recent_rmse` | RMSE over last 200 predictions |
| `recent_energy_mean_uJ` | Mean energy/prediction over last 200 rows |
| `drift_kl_primary` | Current KL divergence (fixed-reference detector) |
| `drift_kl_rolling` | Current KL divergence (rolling detector) |
| `drift_threshold` | Configured τ_drift |
| `planner` | Active planner name |
| `dataset_id` | Bandit dataset ID (if planner=bandit) |

### CV

```bash
python scripts/experiment_logger.py \
    --knowledge managed_system_cv/knowledge \
    --dataset bdd100k \
    --domain cv \
    --interval 30 \
    --log-dir logs/
```

CV-specific columns (replaces per-model columns above):

| Column | Description |
|--------|-------------|
| `ema_scores_json` | JSON dict of per-model EMA scores |
| `ema_accuracy_json` | JSON dict of per-model EMA accuracy |
| `ema_energy_json` | JSON dict of per-model EMA energy |
| `recent_proxy_mean` | Mean proxy score over last 200 predictions |
| `drift_detector` | Active drift detector name |
| `proxy` | Active proxy name |

### Stopping

`Ctrl-C` closes the file cleanly. The log is also flushed every tick so a hard
kill (e.g. machine shutdown) loses at most one row.

### Merging logs after a run

```python
import pandas as pd, glob

logs = pd.concat(
    [pd.read_csv(f) for f in sorted(glob.glob("logs/pems_node2_*.csv"))],
    ignore_index=True,
).sort_values("unix_ts")
logs.to_csv("logs/pems_node2_merged.csv", index=False)
```

---

## Common pitfalls

### `Column 'flow' not found` in inference.py
The live inference engine reads `value_column` from `knowledge/thresholds.json`
(default: `"flow"`). If you switch to UCI electricity or spot_prices (which
output a `value` column), add `"value_column": "value"` to
`managed_system_regression/knowledge/thresholds.json`.

### Nord Pool CSV not parsed correctly
If `preprocess_spot_prices.py` prints `Raw rows: 1` or column names contain
semicolons, the auto-detection failed. Force the separator:
```bash
# The script already handles this via _read_csv_auto, but if it still fails:
# Manually check the separator with:
head -2 /path/to/nordpool_data.csv | cat -A
```

### iWildCam split detection fails (`ValueError: invalid literal for int()`)
This happens with old versions of the script on WILDS ≥2.0 data. Pull the
latest `preprocess_iwildcam.py` — the fix handles both integer and string splits.

### BDD100K manifest: all images assigned `overcast` domain
This means the annotation JSONs were not found. Verify:
- `--bdd-root` points to the directory that contains both `images/` AND `labels/`
- `labels/bdd100k_labels_images_train.json` exists under `<bdd-root>/labels/`

### `YOLO weights not found` on first CV run
Ultralytics downloads weights to `~/.cache/ultralytics/` on first use (requires
internet). Pre-download on a connected machine and copy the cache directory:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt'); YOLO('yolov8m.pt')"
```

### ACDC label mismatch warnings (`WARNING: N label files not found`)
Check that:
1. GT directory is `gt_trainval/gt/` (not `gt_trainval/` directly)
2. Label files end with `_gt_labelTrainIds.png` (not `_gt_labelIds.png`)
3. Sequence subdirectory names match between `rgb_anon/` and `gt_trainval/gt/`

### Energy reads as 0 / null on baremetal
Run the energy probe first:
```bash
python scripts/probe_energy.py
sudo bash scripts/setup_energy_permissions.sh
python scripts/probe_energy.py   # re-check
```
If readings are still 0, set `"energy_meter": "null"` in the dataset config and
note this in the paper's threats section. AMD Zen 2+ exposes RAPL via the same
sysfs path as Intel — the setup script handles both.

---

## Full run sequence for a new dataset (checklist)

```
[ ] Download raw data to a scratch location outside the repo
[ ] python scripts/preprocess_<dataset>.py --input … --output …
[ ] python scripts/validate_dataset.py configs/datasets/<name>.json
[ ] # regression only:
[ ]   python scripts/init_regression.py --config <name>
[ ]   cd managed_system_regression/ && python train.py && cd ..
[ ] # cv only:
[ ]   python scripts/init_cv.py --config <name> [--skip-embeddings]
[ ] python experiments/run_experiment.py --dataset <name> --planner harmone_original --seed 42 --verbose
[ ] # (in parallel terminal) python scripts/experiment_logger.py --knowledge … --dataset … --domain …
[ ] # cv only, after run:
[ ]   python experiments/offline_eval.py --run-dir … --output results/<name>_eval.json
[ ] python experiments/run_grid.py configs/experiments/baseline.yaml   # full 7-planner × 5-seed grid
[ ] python experiments/metrics.py aggregate --grid-dir runs/baseline --out runs/baseline/metrics.csv
```

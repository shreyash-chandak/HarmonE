# HarmonE — Dataset Preparation & Experiment Runbook

Complete instructions for preprocessing all six datasets, running experiments,
computing CV accuracy metrics, and capturing timestamped logs.

**All commands run from `tool/` unless stated otherwise.**
**All relative paths are relative to `tool/`.**

---

## CV target sizes

| Dataset | Task | Target images | Eval split | Interval | # Intervals |
|---------|------|-------------:|-----------|---------|------------|
| BDD100K | Detection (mAP@0.5) | ~3 000 val images | `val` | 1 000 | ~3 |
| iWildCam | Classification (top-1) | ~3 000 ood_test images | `ood_test` | 500 | ~6 |
| ACDC | Segmentation (mIoU) | ~2 000 train+val frames | `val` | 400 | ~5 |
| **Total** | | **~8 000** | | | |

The manifests encode the experimental ordering; do not reorganize the source directories.

---

## Actual data layout (on disk right now)

```
tool/data/
├── pems/
│   ├── flow_data_train.csv  ← R1 initial training portion (~892 obs, 10% of total)
│   └── flow_data_test.csv   ← R1 streaming evaluation portion (~8 036 obs, 90% of total)
├── loadDiagrams/
│   └── LD2011_2014.csv      ← R2 raw UCI Electricity (already downloaded)
├── elspot/
│   └── Elspotprices 2015- 2024.csv   ← R3 raw Nord Pool (already downloaded)
├── bdd100k/
│   ├── images/
│   │   └── 100k/
│   │       ├── train/   *.jpg   (70 000 images)
│   │       ├── val/     *.jpg   (10 000 images — primary eval stream)
│   │       └── test/    *.jpg   (20 000 images — no GT labels)
│   └── labels/
│       └── (100k/ subdirectory OR directly here — script checks both)
│           ├── bdd100k_labels_images_train.json
│           └── bdd100k_labels_images_val.json
├── iwildcam/
│   ├── train/           ← flat dir of UUID.jpg images (WILDS v2.0 ships as train/)
│   ├── metadata.csv
│   ├── categories.csv
│   ├── iwildcam2021_train_annotations_final.json
│   └── RELEASE_v2.0.txt
└── acdc/
    ├── rgb_anon/
    │   ├── fog/{train,trainref,val,valref,test,testref}/{sequence}/*_rgb_anon.png
    │   ├── rain/  (same)
    │   ├── night/ (same)
    │   └── snow/  (same)
    └── gt_trainval/
        └── gt/
            ├── fog/{train,trainref,val,valref}/{sequence}/*_gt_labelTrainIds.png
            ├── rain/  (same)
            ├── night/ (same)
            └── snow/  (same)
```

> ACDC note: the `trainref`, `valref`, `testref` splits are reference counterparts
> of the adverse-condition frames. They are NOT additional adverse-condition images.
> The preprocess script excludes them automatically.

Preprocessed outputs and manifests land in these directories (created by the scripts):

```
tool/data/
├── pems/
│   ├── flow_data_train.csv     ← R1 already ready (no preprocessing needed)
│   └── flow_data_test.csv      ← R1 already ready (no preprocessing needed)
├── uci_electricity/
│   └── uci_electricity.csv     ← R2 preprocessed
├── spot_prices/
│   └── spot_prices.csv         ← R3 preprocessed
├── bdd100k/
│   └── bdd100k_manifest.csv    ← C1 generated manifest (images stay at images/100k/)
├── iwildcam/
│   └── iwildcam_manifest.csv   ← C2 generated manifest
└── acdc/
    └── acdc_manifest.csv       ← C3 generated manifest
```

---

## Quick-reference: order of operations

| Step | Command | Notes |
|------|---------|-------|
| 0 | `source harmone_env/bin/activate` | activate Python env |
| 1 | `python scripts/preprocess_<dataset>.py …` | raw → canonical CSV / manifest |
| 2 | `python scripts/init_regression.py --config <name>` | regression only: fit scaler + KL reference |
| 3 | `python scripts/init_cv.py --config <name>` | CV only: init MAPE-K state (+ optional embeddings) |
| 4 | `python scripts/validate_dataset.py configs/datasets/<name>.json` | verify schema + file existence |
| 5 | train models | see per-dataset section |
| 6 | `python experiments/run_experiment.py --dataset <name> …` | headless run |
| 7 | `python experiments/offline_eval.py …` | CV only: compute GT accuracy |
| 8 | `python scripts/experiment_logger.py …` | parallel terminal: timestamped CSV log |

---

## R1 — PeMS (traffic flow regression)

### Data format

**No preprocessing required.** The data is already in the original HarmonE format:

```
data/pems/
├── flow_data_train.csv   ← 892 flow observations  (~10% of total, initialization portion)
└── flow_data_test.csv    ← 8 036 flow observations (~90% of total, streaming portion)
```

Both files have a single column `flow` (vehicles per 5-minute interval, no timestamp).
The 10%/90% split matches the original HarmonE experimental protocol: a small initial
portion for training the model zoo, and the remainder as the long-running evaluation stream.

The config (`configs/datasets/pems.json`) uses the adapter's pre-split file mode:
`train_path` and `stream_path` point directly to these files — no `train_frac` needed.

### Step 1 — Validate

```bash
python scripts/validate_dataset.py configs/datasets/pems.json
```

Expected: `PASS`.

### Step 2 — Init (scaler + KL reference)

```bash
python scripts/init_regression.py --config pems
```

Writes to `managed_system_regression/knowledge/`:
- `scaler.pkl` — MinMaxScaler fitted on `flow_data_train.csv` only
- `reference_distribution.json` — 50-bin histogram for KL drift detection

### Step 3 — Run headless experiment

The harness trains models inline on first run (when weight files are absent):

```bash
python experiments/run_experiment.py \
    --dataset pems \
    --planner harmone_original \
    --seed 1

# Full paper grid:
python experiments/run_grid.py configs/experiments/baseline.yaml
```

To force retraining (e.g. after changing train_path data):
```bash
rm managed_system_regression/models/lstm_pems.pt
rm managed_system_regression/models/ridge_pems.pkl
rm managed_system_regression/models/svr_pems.pkl
```

### Step 4 — Train models (live managed system only)

```bash
cat data/pems/flow_data_train.csv <(tail -n +2 data/pems/flow_data_test.csv) \
    > managed_system_regression/knowledge/dataset.csv
# (or copy just flow_data_train.csv if you only want the init portion for train.py)
cd managed_system_regression/
python train.py
cd ..
```

### Live managed system (dashboard mode)

```bash
# terminal 1
python app.py
# terminal 2
python run_managed_system.py
# terminal 3
python -m http.server 8000 --directory frontend/
```

Open http://localhost:8000/dashboard.html → Regression → choose planner.

> **value_column note**: PeMS uses column name `flow`.
> `knowledge/thresholds.json` must have `"value_column": "flow"` (the default).
> When switching to UCI or spot_prices (which use `"value"`), update `thresholds.json`.

---

## R2 — UCI Electricity Load Diagrams

### Raw file (already downloaded)

```
data/loadDiagrams/LD2011_2014.csv
```

Semicolon-delimited, European decimal commas, 370 meter columns (`MT_001`–`MT_370`),
~1.4 million rows at 15-minute intervals.

### Step 1 — Preprocess

```bash
python scripts/preprocess_uci_electricity.py \
    --input  data/loadDiagrams/LD2011_2014.csv \
    --output data/uci_electricity/uci_electricity.csv
```

Output: `timestamp, value` (kW per 15-minute interval, meter MT_168 by default).

If MT_168 has a leading-zero gap ≥ 6 months before first non-zero reading:
```bash
python scripts/preprocess_uci_electricity.py \
    --input  data/loadDiagrams/LD2011_2014.csv \
    --output data/uci_electricity/uci_electricity.csv \
    --meter MT_321
```

Document which meter was chosen — it matters for the VMR reuse showcase (R2's
seasonal periodicity is the key drift signal).

### Step 2 — Init

```bash
python scripts/init_regression.py --config uci_electricity
```

### Step 3 — Validate

```bash
python scripts/validate_dataset.py configs/datasets/uci_electricity.json
```

### Step 4 — Run

```bash
python experiments/run_experiment.py \
    --dataset uci_electricity \
    --planner harmone_original \
    --seed 42
```

### Live managed system

```bash
cp data/uci_electricity/uci_electricity.csv managed_system_regression/knowledge/dataset.csv
# Also update knowledge/thresholds.json: "value_column": "value"
```

---

## R3 — Spot Prices (Nord Pool)

### Raw file (already downloaded)

```
data/elspot/Elspotprices 2015- 2024.csv
```

Note: filename contains spaces — quote it in shell. Semicolon-delimited, European decimal commas,
multiple bidding areas per row (DK1, DK2, SE3, NO2, …).

### Step 1 — Preprocess

```bash
python scripts/preprocess_spot_prices.py \
    --input  "data/elspot/Elspotprices 2015- 2024.csv" \
    --output data/spot_prices/spot_prices.csv \
    --area DK1
```

`--area` can be any bidding zone in the file. The script prints available zones if
the specified one is not found.
Output: `timestamp, value` (EUR/MWh, hourly).

### Step 2 — Init

```bash
python scripts/init_regression.py --config spot_prices
```

`train_frac: 0.60` puts the 2019–2020 period in training; 2021+ is the test/inference
window containing the Nord Pool energy crisis — the structural-break stress test.

### Step 3 — Validate & run

```bash
python scripts/validate_dataset.py configs/datasets/spot_prices.json

python experiments/run_experiment.py \
    --dataset spot_prices \
    --planner harmone_original \
    --seed 42
```

> **Expected**: HarmonE will partially fail here. The post-crisis price regime is
> permanently different; no historical model in VMR matches the new distribution.
> This is the designed stress test (RQ6). Record it as a partial failure, not a bug.

---

## C1 — BDD100K (object detection)

### Target: ~3 000 val images → ~3 × 1000-image eval intervals

### Raw files (already downloaded)

```
data/bdd100k/
├── images/100k/
│   ├── train/   *.jpg   (70 000 images)
│   ├── val/     *.jpg   (10 000 images — primary eval stream)
│   └── test/    *.jpg   (20 000 images — no GT labels)
└── labels/
    └── 100k/
        ├── train/   {stem}.json   (70 000 per-image annotation files)
        ├── val/     {stem}.json   (10 000 per-image annotation files)
        └── test/    {stem}.json   (20 000 per-image annotation files)
```

Each per-image JSON contains `"name"` (stem without .jpg), `"attributes"` (weather + timeofday),
and `"frames"` (bounding box annotations). The script reads `labels/100k/{split}/{stem}.json`.

Use the **val split** for the primary evaluated stream — test images have no GT labels.
Do NOT reorganize images into condition subdirectories. The manifest (not the
filesystem) encodes the domain drift order.

**No data movement needed — run from `tool/` with `--bdd-root data/bdd100k`.**

### Step 1 — Preprocess (build manifest)

```bash
cd tool/

# Val + test only (fast — skips 70k train JSONs, ~2 min instead of ~8 min):
# Recommended since only val is used in experiments.
python scripts/preprocess_bdd100k.py \
    --bdd-root data/bdd100k \
    --output   data/bdd100k/bdd100k_manifest.csv \
    --splits val test \
    --no-yolo

# All splits (train + val + test, ~8 min — needed if you want train rows in the manifest):
python scripts/preprocess_bdd100k.py \
    --bdd-root data/bdd100k \
    --output   data/bdd100k/bdd100k_manifest.csv \
    --no-yolo

# With YOLO label conversion (required for offline_eval.py mAP@0.5):
python scripts/preprocess_bdd100k.py \
    --bdd-root        data/bdd100k \
    --output          data/bdd100k/bdd100k_manifest.csv \
    --splits val \
    --yolo-labels-dir data/bdd100k/labels_yolo/
```

Manifest columns: `sample_id, input_path, label_path, domain, split`

Domain drift order: `clear_day → overcast → foggy → dusk → night → rain`

Check domain distribution in the val split:
```bash
python -c "
import pandas as pd
m = pd.read_csv('data/bdd100k/bdd100k_manifest.csv')
print(m[m.split=='val'].groupby('domain').size())
print('Total val:', len(m[m.split=='val']))
"
```

### Step 2 — Init CV artifacts

```bash
# Default luminance_kl drift detector (no GPU needed for init):
python scripts/init_cv.py --config bdd100k --skip-embeddings

# Embedding-based drift (MMD/Fréchet) — needs GPU and model weights:
python scripts/init_cv.py --config bdd100k
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

```bash
python experiments/offline_eval.py \
    --run-dir    managed_system_cv/knowledge \
    --labels-dir data/bdd100k/labels_yolo/val \
    --images-dir data/bdd100k/images/100k/val \
    --interval   1000 \
    --output     results/bdd100k_eval.json
```

Output JSON:
```json
{
  "interval_results": [
    {"interval_idx": 0, "map50": 0.42, "n_images": 1000,
     "model": "yolo_n", "timestamp_start": 1234, "timestamp_end": 1235}
  ],
  "overall_map50": 0.44,
  "n_intervals": 3
}
```

Uses Ultralytics `val()` per interval — requires GPU. Add `--smoke` for a
3-interval CPU sanity check.

### Live managed system

```bash
python app.py
python run_managed_system.py
python -m http.server 8000 --directory frontend/
```

In dashboard: HarmonE (CV) → choose planner → Start.

---

## C2 — iWildCam (classification)

### Target: ~3 000 ood_test images → ~6 × 500-image eval intervals

### Raw files (already downloaded)

```
data/iwildcam/
├── train/           ← flat directory of UUID.jpg images (WILDS v2.0 layout)
├── metadata.csv     ← one row per image: image_id, y, split, location_remapped, …
├── categories.csv   ← class_id → species name
├── iwildcam2021_train_annotations_final.json   ← NOT used by preprocess script
└── RELEASE_v2.0.txt
```

**WILDS v2.0 puts images in `train/` not `images/`** — the preprocess script handles both.
`metadata.csv` is the source of truth. The annotation JSON is not used.

**No data movement needed — run from `tool/` with `--wilds-root data/iwildcam`.**

### Step 1 — Preprocess (build manifest)

```bash
cd tool/

python scripts/preprocess_iwildcam.py \
    --wilds-root data/iwildcam \
    --output     data/iwildcam/iwildcam_manifest.csv
```

Manifest columns: `sample_id, input_path, label, domain, split`

The script uses `location_remapped` as the domain column (geographic drift axis).
Samples are sorted by location — **do not globally shuffle** the OOD test stream.
The drift represents camera-location transitions, so location-to-location ordering
matters. Within a location, order is less critical.

Verify:
```bash
python -c "
import pandas as pd
m = pd.read_csv('data/iwildcam/iwildcam_manifest.csv')
print(m.groupby('split').size())
ood = m[m.split == 'ood_test']
print('OOD test:', len(ood), 'images across', ood['domain'].nunique(), 'locations')
"
```

### Step 2 — Init CV artifacts

```bash
# Fast init without embeddings:
python scripts/init_cv.py --config iwildcam --skip-embeddings

# With embedding reference (MMD drift detector — needs GPU):
python scripts/init_cv.py --config iwildcam
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

```bash
python experiments/offline_eval.py \
    --run-dir  managed_system_cv/knowledge \
    --manifest data/iwildcam/iwildcam_manifest.csv \
    --split    ood_test \
    --interval 500 \
    --output   results/iwildcam_eval.json
```

---

## C3 — ACDC (semantic segmentation)

### Target: ~2 000 frames → ~5 × 400-image eval intervals (~500 per condition)

### Raw files (already downloaded)

```
data/acdc/
├── rgb_anon/
│   ├── fog/
│   │   ├── train/{sequence}/*_rgb_anon.png
│   │   ├── val/{sequence}/*_rgb_anon.png
│   │   ├── test/{sequence}/*_rgb_anon.png      ← excluded (no GT)
│   │   ├── trainref/{sequence}/*_rgb_anon.png  ← excluded (reference counterpart)
│   │   ├── valref/{sequence}/*_rgb_anon.png    ← excluded
│   │   └── testref/{sequence}/*_rgb_anon.png   ← excluded
│   ├── rain/  (same structure)
│   ├── night/ (same structure)
│   └── snow/  (same structure)
└── gt_trainval/
    └── gt/
        ├── fog/{train,val,trainref,valref}/{sequence}/*_gt_labelTrainIds.png
        ├── rain/  (same)
        ├── night/ (same)
        └── snow/  (same)
```

**Critical**: ACDC's train/val split does NOT divide a GoPro recording temporally.
A single GoPro (e.g. `GOPR0475`) can have frames split across BOTH `train/` and
`val/` directories. The preprocess script reconstructs each GoPro's natural frame
order by merging train+val and sorting by frame number encoded in the filename.

Key rules:
- Use `_gt_labelTrainIds.png` — **NOT** `_gt_labelIds.png`
- Only train + val are processed (no GT for test; ref splits excluded)
- The `split` column in the manifest records each frame's original directory
  (train or val) so the correct GT label path can be resolved
- Primary ordering is: condition → sequence → frame_number

**No data movement needed — run from `tool/` with `--acdc-root data/acdc`.**

### Step 1 — Preprocess (build manifest with sequence reconstruction)

```bash
cd tool/

python scripts/preprocess_acdc.py \
    --acdc-root data/acdc \
    --output    data/acdc/acdc_manifest.csv
```

Manifest columns: `sample_id, input_path, label_path, domain, sequence, frame_number, split`

Drift stream order: `fog → rain → night → snow`

Within each condition, frames are ordered by `sequence` (GoPro name) then
`frame_number` — this is the natural GoPro recording order.

Verify sequence reconstruction:
```bash
python -c "
import pandas as pd
from pathlib import Path
m = pd.read_csv('data/acdc/acdc_manifest.csv')
print('=== Count by domain/split ===')
print(m.groupby(['domain', 'split']).size())
print()
print('=== Sequences per condition ===')
print(m.groupby('domain')['sequence'].nunique())
print()
missing = m[~m.label_path.apply(lambda p: Path(p).exists())]
print(f'Missing labels: {len(missing)} / {len(m)}')
"
```

Example of correct sequence reconstruction (frames from different splits merged):
```
domain=fog, sequence=GOPR0475:
  GOPR0475_frame_000001_rgb_anon.png  (split=train)
  GOPR0475_frame_000002_rgb_anon.png  (split=val)
  GOPR0475_frame_000003_rgb_anon.png  (split=train)
  ...
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

GT masks are the `label_path` column (`_gt_labelTrainIds.png`, Cityscapes trainIds,
`ignore_index=255`). The `offline_eval.py --split val` flag selects only frames
whose original split was `val` (these have GT labels; train frames also have GT
but `val` is the standard held-out evaluation partition).

```bash
python experiments/offline_eval.py \
    --run-dir  managed_system_cv/knowledge \
    --manifest data/acdc/acdc_manifest.csv \
    --split    val \
    --interval 400 \
    --output   results/acdc_eval.json
```

Segmentation adapter computes per-class IoU ignoring `ignore_index=255`, then
macro-averages over the 19 Cityscapes classes.

---

## CV accuracy metrics — summary

| Dataset | Task | Metric | GT source | Key arg |
|---------|------|--------|-----------|---------|
| BDD100K | Detection | mAP@0.5 | YOLO `.txt` files from preprocess | `--labels-dir data/bdd100k/labels_yolo/val` |
| iWildCam | Classification | Top-1 accuracy | `label` column in manifest | `--manifest data/iwildcam/iwildcam_manifest.csv` |
| ACDC | Segmentation | mIoU (19 classes, ignore=255) | `_gt_labelTrainIds.png` in manifest | `--manifest data/acdc/acdc_manifest.csv` |

All three use the same `offline_eval.py` entry point. It auto-detects the task from the
dataset config (`"task": "detection"` / `"classification"` / `"segmentation"`).

For proxy validation (RQ3 — does the runtime proxy track GT accuracy):

```bash
python experiments/proxy_validation.py \
    --eval-results results/bdd100k_eval.json \
    --run-dir      managed_system_cv/knowledge \
    --output       results/bdd100k_proxy_validation.json
```

---

## Experiment logger (timestamped readings)

Polls the `knowledge/` directory and appends one CSV row per tick to
`logs/<dataset>_<YYYY-MM-DD_HH-MM-SS>.csv`. Run in a separate terminal alongside
the managed system or headless experiment.

### Regression

```bash
python scripts/experiment_logger.py \
    --knowledge managed_system_regression/knowledge \
    --dataset   pems_node2 \
    --domain    regression \
    --interval  30 \
    --log-dir   logs/
```

Captured columns per tick:

| Column | Description |
|--------|-------------|
| `wall_clock_utc` | ISO timestamp of this snapshot |
| `unix_ts` | Unix epoch (for alignment with predictions.csv) |
| `current_model` | Active model name (lstm / linear / svm) |
| `last_mape_line` | Row index in predictions.csv at snapshot time |
| `current_energy_threshold` | Dynamic τ_E value |
| `ema_score_lstm/linear/svm` | Per-model EMA composite score |
| `ema_acc_lstm/linear/svm` | Per-model EMA accuracy |
| `ema_energy_lstm/linear/svm` | Per-model EMA energy |
| `event_model_switches` | Cumulative switch count |
| `event_retrains` | Cumulative retrain count |
| `event_vmr` | Cumulative VMR reuse count |
| `event_noops` | Cumulative noop count |
| `mape_k_energy_uJ` | Cumulative MAPE-K overhead energy (µJ) |
| `recent_r2_mean` | Mean R² over last 200 predictions |
| `recent_rmse` | RMSE over last 200 predictions |
| `recent_energy_mean_uJ` | Mean energy/prediction over last 200 rows |
| `drift_kl_primary` | KL divergence (fixed-reference detector) |
| `drift_kl_rolling` | KL divergence (rolling detector) |
| `drift_threshold` | Configured τ_drift |
| `planner` | Active planner name |
| `dataset_id` | Bandit dataset ID (if planner=bandit) |

### CV

```bash
python scripts/experiment_logger.py \
    --knowledge managed_system_cv/knowledge \
    --dataset   bdd100k \
    --domain    cv \
    --interval  30 \
    --log-dir   logs/
```

CV-specific columns (per-model fields are JSON dicts instead):

| Column | Description |
|--------|-------------|
| `ema_scores_json` | JSON dict of per-model EMA scores |
| `ema_accuracy_json` | JSON dict of per-model EMA accuracy |
| `ema_energy_json` | JSON dict of per-model EMA energy |
| `recent_proxy_mean` | Mean proxy score over last 200 predictions |
| `drift_detector` | Active drift detector name |
| `proxy` | Active proxy name |

### Stopping & merging

`Ctrl-C` flushes the file cleanly. To merge logs from multiple runs:

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

### `Column 'flow' not found` in live inference.py
Switch between datasets by updating `"value_column"` in
`managed_system_regression/knowledge/thresholds.json`:
- PeMS → `"flow"` (the default)
- UCI electricity / spot_prices → `"value"`

The headless harness reads `value_column` from the dataset config file automatically;
only the live system needs the manual update.

### Spot prices: shell quoting for filename with spaces
```bash
# Quote the path — the filename is "Elspotprices 2015- 2024.csv" with a space:
python scripts/preprocess_spot_prices.py \
    --input "data/elspot/Elspotprices 2015- 2024.csv" \
    --output data/spot_prices/spot_prices.csv \
    --area DK1
```

### BDD100K: all images get domain `overcast` (annotation fallback)
The script falls back to `overcast` when no annotation JSON is found. If this
happens for every image, the JSON files weren't discovered. The expected layout is
**one JSON per image** in `labels/100k/{split}/`:
```bash
ls data/bdd100k/labels/100k/val/    # should list 10 000 *.json files
ls data/bdd100k/labels/100k/train/  # should list 70 000 *.json files
```
The script also falls back to the older bundle format (`labels/bdd100k_labels_images_val.json`).
If neither is found, verify your BDD100K download includes per-image JSON labels.

### iWildCam: `Image directory not found` error
The script tries `images/` then `train/` under `--wilds-root`. If neither exists:
```bash
ls data/iwildcam/    # find the actual image folder name
```
If it's named something else (e.g. `photos/`), create a symlink:
```bash
ln -s photos data/iwildcam/images
```

### ACDC: `WARNING: N label files not found`
Check these three things:
1. GT path must be `gt_trainval/gt/` (there IS a `gt/` subdirectory inside `gt_trainval/`)
2. Label files must end with `_gt_labelTrainIds.png` (not `_gt_labelIds.png`)
3. Sequence folder names must match between `rgb_anon/` and `gt_trainval/gt/`

### ACDC: manifest `split` column looks wrong / mixed
This is expected and correct. The `split` column records each frame's original
directory (train or val), NOT the experiment split. A single GoPro sequence will
have frames from both `train` and `val` interleaved in frame-number order. That is
the intended output of the sequence-reconstruction logic.

### YOLO weights not available offline
Ultralytics downloads weights to `~/.cache/ultralytics/` on first use (needs internet).
Pre-download on a connected machine:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt'); YOLO('yolov8m.pt')"
```
Then copy `~/.cache/ultralytics/` to the experiment machine.

### Energy reads as 0 / null
```bash
python scripts/probe_energy.py
sudo bash scripts/setup_energy_permissions.sh
python scripts/probe_energy.py   # re-check
```
If still 0 (no RAPL support), set `"energy_meter": "null"` in the dataset config.
Note this in the paper's threats-to-validity section.

---

## Full run checklist (new dataset or fresh machine)

```bash
cd tool/
source harmone_env/bin/activate

# ── Preprocess ──────────────────────────────────────────────────────────────
# R1: PeMS data already at data/pems/ (flow_data_train.csv + flow_data_test.csv)
# No preprocessing step needed — files are already in the required format.

# R2: raw file already at data/loadDiagrams/LD2011_2014.csv
python scripts/preprocess_uci_electricity.py \
    --input  data/loadDiagrams/LD2011_2014.csv \
    --output data/uci_electricity/uci_electricity.csv

# R3: raw file already at data/elspot/ — note space in filename, must quote
python scripts/preprocess_spot_prices.py \
    --input  "data/elspot/Elspotprices 2015- 2024.csv" \
    --output data/spot_prices/spot_prices.csv \
    --area   DK1

# C1: data already at data/bdd100k/
# Use --splits val to skip 70k train JSONs (~2 min vs ~8 min); only val is used in experiments.
python scripts/preprocess_bdd100k.py \
    --bdd-root data/bdd100k \
    --output   data/bdd100k/bdd100k_manifest.csv \
    --splits val \
    --no-yolo

# ── Trim manifests to experiment-appropriate size ─────────────────────────────
# BDD100K: keep ~3 000 val images (500/domain × 6 domains)
python scripts/trim_manifest.py \
    --dataset  bdd100k \
    --manifest data/bdd100k/bdd100k_manifest.csv \
    --target   3000

# iWildCam: keep ~3 000 ood_test images distributed across locations
python scripts/trim_manifest.py \
    --dataset  iwildcam \
    --manifest data/iwildcam/iwildcam_manifest.csv \
    --target   3000
# ACDC: no trim needed (~2 006 frames total)

# C2: data already at data/iwildcam/
python scripts/preprocess_iwildcam.py \
    --wilds-root data/iwildcam \
    --output     data/iwildcam/iwildcam_manifest.csv

# C3: data already at data/acdc/
python scripts/preprocess_acdc.py \
    --acdc-root data/acdc \
    --output    data/acdc/acdc_manifest.csv

# ── Validate all ─────────────────────────────────────────────────────────────
for cfg in pems uci_electricity spot_prices bdd100k iwildcam acdc; do
    python scripts/validate_dataset.py configs/datasets/${cfg}.json
done

# ── Init regression ───────────────────────────────────────────────────────────
python scripts/init_regression.py --config pems
python scripts/init_regression.py --config uci_electricity
python scripts/init_regression.py --config spot_prices

# ── Init CV (no embeddings for speed; run without --skip-embeddings later) ───
python scripts/init_cv.py --config bdd100k  --skip-embeddings
python scripts/init_cv.py --config iwildcam --skip-embeddings
python scripts/init_cv.py --config acdc     --skip-embeddings

# ── Smoke tests ───────────────────────────────────────────────────────────────
python experiments/run_experiment.py --dataset pems --planner harmone_original --seed 42 --verbose
python experiments/run_experiment.py --dataset bdd100k    --planner harmone_original --seed 42 --verbose

# ── Full paper grid ───────────────────────────────────────────────────────────
python experiments/run_grid.py configs/experiments/baseline.yaml

# ── CV ground-truth accuracy (after runs complete) ────────────────────────────
python experiments/offline_eval.py \
    --run-dir    managed_system_cv/knowledge \
    --labels-dir data/bdd100k/labels_yolo/val \
    --images-dir data/bdd100k/images/100k/val \
    --interval   1000 \
    --output     results/bdd100k_eval.json

python experiments/offline_eval.py \
    --run-dir  managed_system_cv/knowledge \
    --manifest data/iwildcam/iwildcam_manifest.csv \
    --split    ood_test \
    --interval 500 \
    --output   results/iwildcam_eval.json

python experiments/offline_eval.py \
    --run-dir  managed_system_cv/knowledge \
    --manifest data/acdc/acdc_manifest.csv \
    --split    val \
    --interval 400 \
    --output   results/acdc_eval.json

# ── Aggregate metrics ─────────────────────────────────────────────────────────
python experiments/metrics.py aggregate \
    --grid-dir runs/baseline \
    --out      runs/baseline/metrics.csv
```

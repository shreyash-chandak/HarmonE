# Dataset Row Schemas — What Each Dataset Must Look Like on Disk

> Synthesised from `docs/datasets/`, `docs/DATA_CONTRACT.md`, and `configs/datasets/`.
> This is the ground truth for anyone doing preprocessing. If what you produce does not
> match these tables exactly, the validator will reject it.

---

## Quick reference

| ID | Dataset | Domain | Preprocessed file | Key columns |
|----|---------|--------|------------------|-------------|
| R1 | PeMS Node 2 | regression | single CSV | `timestamp`, `value` (vehicle flow, int-like float) |
| R2 | UCI Electricity | regression | single CSV | `timestamp`, `value` (kW per 15-min, float) |
| R3 | ERCOT Spot Prices | regression | single CSV | `timestamp`, `value` (USD/MWh, float) |
| C1 | BDD100K | cv / detection | manifest CSV + JPEG images + YOLO .txt labels | `image_path`, `label_path`, `weather`, `timeofday` |
| C2 | iWildCam | cv / classification | manifest CSV + JPEG images + single-int .txt labels | `image_path`, `label_path`, `location`, `split` |
| C3 | ACDC | cv / segmentation | manifest CSV + PNG images + PNG mask labels | `image_path`, `label_path`, `condition`, `split` |

---

## Regression datasets — shared contract

All three regression datasets must produce **one UTF-8 CSV** with:

| Column | Type | Constraint |
|--------|------|-----------|
| `timestamp` | string (ISO-8601 or any `pd.to_datetime`-parseable format) | Monotonically increasing. Required for split boundary tooling; not used for inference. |
| `value` | float64 | The prediction target. **Zero NaN allowed.** Drop or impute before delivery. |

Additional columns are ignored but do not cause failure.

Row order = stream order. Do not shuffle. The validator checks that the value column has no NaN and that the row count meets the minimum.

---

## R1 — PeMS Node 2

**Source:** PeMS 5-minute sensor export (any station; `Total Flow` column).

**Raw → Preprocessed mapping:**

| Raw column | Preprocessed column | Notes |
|-----------|-------------------|-------|
| `Timestamp` | `timestamp` | Already ISO-8601 in PeMS exports |
| `Total Flow` | `value` | Integer vehicle count per 5-min interval; safe to cast to float64 |

**Row characteristics:**
- Interval: 5 minutes
- Value range: typically 0–400 vehicles per interval (sensor-dependent)
- Expected volume: ≥ 5 000 rows (practical minimum); a full year of 5-min data = ~105 000 rows
- Recurring daily and weekly patterns — drift detector will fire on the expected weekly periodicity

**Config:**
```
value_column: "value"
seq_length: 5      ← 5 steps × 5 min = 25-min look-back
train_frac: 0.8
tau_drift: 0.5     ← default; calibrate per-dataset before experiments
```

**Preprocessed file location:** `data/pems_node2/pems_node2.csv`

---

## R2 — UCI Electricity Load Diagrams 2011–2014

**Source:** `LD2011_2014.txt` from UCI ML Repository. Semicolon-delimited,
370 client columns (MT_001 … MT_370), European decimal comma, 15-minute intervals.

**Raw → Preprocessed mapping:**

| Raw | Preprocessed | Notes |
|-----|-------------|-------|
| Row index (datetime) | `timestamp` | After DST duplicate removal (keep first occurrence each spring/autumn clock change) |
| `MT_168` (or chosen client) | `value` | European comma decimal (`0,315` → `0.315`); units: kW per 15-min interval |

**Row characteristics:**
- Interval: 15 minutes
- Recommended client: MT_168 (most complete record, clear seasonal signal)
- Value range: 0–~10 kW per 15-min interval per client (varies widely)
- Expected volume: ~140 000 rows (2011-01-01 to 2014-12-31 at 15-min granularity, minus DST dupes)
- **DST quirk:** Two duplicated timestamps per year (spring forward / fall back). Drop second occurrence before writing output.
- Seasonal drift (winter peak, summer trough) is the VMR-reuse showcase — do not filter it out.

**Config:**
```
value_column: "value"
seq_length: 96     ← 96 × 15 min = 24-hour look-back window
train_frac: 0.8    ← roughly 2011–2013 train, 2014 stream
tau_drift: 0.5     ← default; calibrate before experiments
```

**Preprocessed file location:** `data/uci_electricity/uci_electricity.csv`

---

## R3 — ERCOT Day-Ahead Spot Prices

**Source:** ERCOT DAM Settlement Point Prices, settlement point `HB_NORTH`,
period 2019-01-01 to 2022-12-31. Columns in raw export:
`Delivery Date`, `Delivery Hour`, `Settlement Point Price`.

**Raw → Preprocessed mapping:**

| Raw column(s) | Preprocessed column | Notes |
|--------------|-------------------|-------|
| `Delivery Date` + `Delivery Hour` | `timestamp` | Combine: `date + timedelta(hours=hour-1)`. Hour column is 1-indexed. |
| `Settlement Point Price` | `value` | USD/MWh; float64. Occasional negative prices are valid — do not clip. |

**Row characteristics:**
- Interval: hourly
- Value range: typically 20–80 USD/MWh pre-crisis; spikes above 9 000 USD/MWh during Winter Storm Uri (Feb 2021)
- Expected volume: ~35 000 rows (4 years hourly)
- **Structural break:** approximately 2021-02-13. Values before and after are from different regimes and will not revert. HarmonE is expected to fail here — this is the designed stress test.
- Do not filter out the spike. Do not clip. The spike is the scientific signal.

**Config:**
```
value_column: "value"
seq_length: 24     ← 24-hour look-back
train_frac: 0.6    ← 2019–2020 train; 2021-H1 val; 2021-H2+ stream
tau_drift: 0.5     ← default; calibrate before experiments
```

**Preprocessed file location:** `data/spot_prices/spot_prices.csv`

---

## CV datasets — shared contract

All three CV datasets must produce a **manifest CSV** (rows in stream order, no shuffling)
plus the image files and label files the manifest points to.

The manifest's `image_path` column contains paths relative to `data_root` (set in config).
Stream order = row order of the manifest.

---

## C1 — BDD100K (Detection)

**Task:** object detection · **Adapter:** `DetectionAdapter` · **Models:** YOLOv8n/s/m

**Manifest CSV columns:**

| Column | Type | Notes |
|--------|------|-------|
| `image_path` | string | Path to JPEG, relative to `data_root`. E.g. `images/100k/train/00001234.jpg` |
| `label_path` | string | Path to YOLO-format `.txt` for this image. Empty string if no boxes. |
| `weather` | string | From BDD JSON `attributes.weather`: `clear`, `overcast`, `partly cloudy`, `foggy`, `rainy`, `snowy`, `undefined` |
| `timeofday` | string | From BDD JSON `attributes.timeofday`: `daytime`, `dawn/dusk`, `night`, `undefined` |

**Label file format (YOLO .txt):** one line per bounding box:
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates normalised to [0, 1]. Empty file = no objects in frame. Class IDs follow the BDD → YOLO mapping in `managed_system_cv/utility/bdd_to_yolo_labels.py`.

**Drift-stream ordering (manifest row order):**
```
clear/daytime → overcast/daytime → dawn/dusk → nighttime → rainy
```
Sort key: `(weather_severity_rank, timeofday_severity_rank)`.

**Image characteristics:**
- Format: JPEG, 1280×720 (BDD100K native resolution)
- Expected volume: up to 70 000 training images; a filtered subset (e.g. 10 000–20 000) is sufficient

**CV config keys:**
```
task: "detection"
embedding_model: "yolo_n"    ← pinned; do not change between runs
embedding_dim: 256            ← SPPF spatial-mean-pool output dim for yolo_n
drift_detector: "luminance_kl"
proxy: "confidence"           ← mean box confidence per frame
```

**Labels are offline-only.** Runtime inference never opens `label_path`. Only `experiments/offline_eval.py` reads them.

---

## C2 — iWildCam (Classification)

**Task:** image classification · **Adapter:** `ClassificationAdapter` · **Models:** EfficientNet-B0, ResNet-50, ResNet-101

**Manifest CSV columns:**

| Column | Type | Notes |
|--------|------|-------|
| `image_path` | string | Path to JPEG, relative to `data_root`. E.g. `iwildcam_v2.0/train/abc123.jpg` |
| `label_path` | string | Path to single-integer label `.txt` for this image |
| `location` | int | WILDS metadata location ID — used only for drift-stream ordering |
| `split` | string | WILDS split tag: `train`, `val`, `test`, `id_val`, `ood_val`, etc. |

**Label file format (single-integer .txt):**
```
42
```
One integer per file: the species class ID, range 0–181 (182 total classes).

**Drift-stream ordering (manifest row order):**
Sort ascending by `location` ID. This places images from the same geographic location
together. Locations seen during training appear first; geographically distant new locations
appear later, creating progressive geographic drift invisible to luminance-KL.

**Image characteristics:**
- Format: JPEG, variable resolution (camera-trap thumbnails, typically 480×360 or larger)
- Expected volume: ~200 000 training images (~150 GB download via WILDS package)

**CV config keys:**
```
task: "classification"
num_classes: 182
embedding_model: "efficientnet_b0"   ← pinned; dim 1280
embedding_dim: 1280
drift_detector: "mmd_embedding"      ← primary; luminance_kl is logged but not the trigger
proxy: "max_softmax"                 ← max predicted softmax probability per image
```

**Labels are offline-only.** `label_path` is only read by `experiments/offline_eval.py`.

---

## C3 — ACDC (Segmentation)

**Task:** semantic segmentation · **Adapter:** `SegmentationAdapter` · **Models:** SegFormer-B0, SegFormer-B2

**Manifest CSV columns:**

| Column | Type | Notes |
|--------|------|-------|
| `image_path` | string | Path to RGB `.png`, relative to `data_root`. E.g. `rgb_anon/fog/train/GOPR0351_frame_000001_rgb_anon.png` |
| `label_path` | string | Path to Cityscapes-format label PNG (`*_gt_labelTrainIds.png`). Empty string if no label available. |
| `condition` | string | `fog`, `night`, `rain`, `snow`, or `clear` (for correspondence subset) |
| `split` | string | `train` or `val` |

**Label file format (Cityscapes PNG mask):**
- Single-channel PNG
- Each pixel = Cityscapes trainId integer in range [0, 18]
- Pixel value 255 = void / unlabelled — **must be ignored in mIoU** (`ignore_index: 255` in config)
- The SegmentationAdapter upsamples logits to the mask's H×W (bilinear, then argmax) before computing mIoU (R5-b fix)

**Drift-stream ordering (manifest row order):**
```
clear → fog → rain → night → snow
```
Within each condition: ordered by sequence name (chronological within sequence).

**Image characteristics:**
- Format: PNG, 1920×1080 (ACDC native resolution)
- Expected volume: ~4 006 images total (400 fog + 1 006 night + 1 600 rain + 1 000 snow train/val combined; exact counts vary by release)

**CV config keys:**
```
task: "segmentation"
num_classes: 19
ignore_index: 255
embedding_model: "segformer_b0"   ← pinned; dim 256
embedding_dim: 256
drift_detector: "luminance_kl"    ← clear→night is luminance-visible
proxy: "max_logit"                ← max pre-softmax logit, spatial-mean-pooled
```

**Labels are offline-only.** `label_path` is only read by `experiments/offline_eval.py`.

---

## Shared rules for all datasets

1. **No NaN in the signal column.** For regression: the `value` column. For CV: not applicable at row level (NaN in metadata columns is tolerated).
2. **Row order = stream order.** The validator does not sort. The order you write is the order the system streams.
3. **`status: "awaiting_data"` in config skips path checks.** Remove this key once the data is on disk and the validator runs clean.
4. **Labels are never read at runtime.** `label_path` in CV manifests is consumed only by `experiments/offline_eval.py`. The runtime loop is label-free (invariant I4). Do not add any code that reads labels during inference.
5. **Validator runs before init.** `scripts/validate_dataset.py --config <name>` must PASS before you run `init_regression.py` or `init_cv.py`. Nothing else runs until validation passes.

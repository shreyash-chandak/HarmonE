# HarmonE Dataset Plug-and-Play Contract

> Version: 2026-07  
> See `configs/datasets/_template.json` for all config keys with inline comments.

A dataset whose preprocessed form matches this contract can be run by HarmonE with **zero code changes — config file only**.  Violating the schema causes `core/dataset_validator.py` to abort startup with a clear error list.

---

## 1. Regression Domain (`domain: "regression"`)

### 1.1 Data file

| Requirement | Detail |
|---|---|
| Format | UTF-8 CSV, single header row |
| Value column | One numeric column; name declared in config as `value_column` |
| Row order | Strictly chronological (stream order = row order) |
| NaN policy | **Zero NaN allowed** in the value column. Impute or drop beforehand; HarmonE never imputes silently. |
| Minimum size | ≥ `max(100, (seq_length + horizon) × 10)` rows |
| Optional column | `timestamp` (ISO-8601 or unix seconds) — used for plots/split boundaries only; must be monotonically increasing if present |

### 1.2 Config keys

```jsonc
{
  "name":          "REQUIRED — short identifier",
  "domain":        "regression",
  "adapter":       "adapters.regression_csv.RegressionCSVAdapter",
  "data_path":     "REQUIRED — path to CSV (relative to tool/ or absolute)",
  "value_column":  "REQUIRED — column to predict",
  "seq_length":    5,          // optional, default 5
  "train_frac":    0.8,        // optional, default 0.8
  "val_frac":      0.0,        // optional; train_frac + val_frac ≤ 1.0
  "stream_delay_s": 0.15       // optional; seconds between streaming steps
}
```

### 1.3 Init script

```
python scripts/init_regression.py --config <name>
```

Produces (idempotent):
- `managed_system_regression/knowledge/scaler.pkl` — MinMaxScaler fitted on train split only  
- `managed_system_regression/knowledge/reference_distribution.json` — 50-bin histogram of train-split values

These are training-time artifacts and **must not be deleted between runs** (only `experiments/run_reset.py --domain regression` touches run-state files).

---

## 2. CV Domain (`domain: "cv"`)

### 2.1 Data files

**Option A — Image directory** (`image_dir`):  
A flat directory of JPEG/PNG images.  Stream order = sorted filename order.

**Option B — Manifest CSV** (`manifest_csv`):  
Required column: `image_path` (path relative to `data_root`).  Rows in stream order.  
Optional columns:

| Column | Purpose |
|---|---|
| `attributes` | Free-form string (weather, time-of-day) — available for logging |
| `label_path` | YOLO-format `.txt` for offline evaluation only — never read at runtime |
| `split` | `train` / `val` / `stream` — if absent, config must provide `train_frac` |

**Image format:** any format readable by Pillow (JPEG recommended for file size).  
**Minimum size:** ≥ 60 images (validation samples 100 for existence check).

### 2.2 Config keys

```jsonc
{
  "name":         "REQUIRED",
  "domain":       "cv",
  "adapter":      "adapters.cv_imagedir.CVImageDirAdapter",
  "image_dir":    "REQUIRED — path to image directory (or use manifest_csv)",
  "manifest_csv": null,        // optional; overrides image_dir ordering
  "data_root":    ".",         // used to resolve manifest image_path column
  "models": {
    "yolo_n": {"weights_path": "...", "loader": "adapters.loaders.yolo_loader", "cost_class": "light"},
    "yolo_s": {"weights_path": "...", "loader": "adapters.loaders.yolo_loader", "cost_class": "medium"},
    "yolo_m": {"weights_path": "...", "loader": "adapters.loaders.yolo_loader", "cost_class": "heavy"}
  }
}
```

### 2.3 CV task keys (v2, added in prototype sprint)

When `domain: "cv"`, four additional groups of keys are supported:

```jsonc
// Task dispatch (required for non-detection tasks)
"task":         "detection",        // "detection" | "classification" | "segmentation"
"task_adapter": "adapters.tasks.detection.DetectionAdapter",  // resolved from task if omitted
"num_classes":  80,                 // required for classification and segmentation
"ignore_index": 255,                // segmentation only — pixel value to exclude from mIoU

// Embedding drift (required only when drift_detector is mmd_embedding/frechet_embedding)
"embedding_model":        "yolo_n",    // pinned model for drift embedding extraction (R4)
"embedding_dim":          256,         // embedding vector length — validated on first extraction
"embedding_sample_every": 5,           // extract every N frames (default 1)
"drift_window_size":      500,         // ring-buffer size for EmbeddingStore

// Retrain tactics
"pseudo_label_threshold": 0.5,         // confidence threshold for pseudo-label acceptance
"rollback_if_worse":      true,        // rollback new weights if proxy score drops after retrain
```

All these keys are documented with inline comments in
`configs/datasets/_template.json` and validated by `core/dataset_validator.py`.
Per-dataset specifics live in `docs/datasets/`.

### 2.4 Awaiting-data status

```json
"status": "awaiting_data"
```

When a config has this key set, the validator skips all data-path existence checks
and reports SKIPPED (not FAIL). Use this for configs whose data has not yet been
downloaded. Remove the key once data is in place.

### 2.5 Init script

```
python scripts/init_cv.py --config <name> [--skip-embeddings]
```

Produces (idempotent):
- `managed_system_cv/versionedMR/{model}_v1.pt` — copy of base weights (VMR seed)
- `managed_system_cv/versionedMR/{model}_v1_hist.json` — avg luminance histogram over reference images
- `managed_system_cv/knowledge/model.csv` — default active model
- `managed_system_cv/knowledge/mape_info.json` — blank EMA state
- `managed_system_cv/knowledge/reference_embeddings.npz` — fixed reference embeddings for
  embedding-based drift detectors (§5.2); requires embedding_model weights; skipped when
  `--skip-embeddings` is passed or when `drift_detector` is not embedding-based

Per-dataset specs and preprocessing recipes: see `docs/datasets/`.

---

## 2a. Per-Dataset Specs

Detailed dataset documentation (raw form, preprocessing recipe, drift-stream
construction, license) lives in `docs/datasets/`:

| Dataset | Doc | Config |
|---------|-----|--------|
| PeMS node 2 | [pems_node2.md](datasets/pems_node2.md) | `configs/datasets/pems_node2.json` |
| UCI Electricity | [uci_electricity.md](datasets/uci_electricity.md) | `configs/datasets/uci_electricity.json` |
| ERCOT Spot Prices | [spot_prices.md](datasets/spot_prices.md) | `configs/datasets/spot_prices.json` |
| BDD100K | [bdd100k.md](datasets/bdd100k.md) | `configs/datasets/bdd100k.json` |
| iWildCam | [iwildcam.md](datasets/iwildcam.md) | `configs/datasets/iwildcam.json` |
| ACDC | [acdc.md](datasets/acdc.md) | `configs/datasets/acdc.json` |

All schema statements in those docs are cross-checked against `core/dataset_validator.py`.

---

## 3. Per-Run Reset

Before every experiment session:

```
python experiments/run_reset.py --domain regression   # or cv
```

Zeroes: `last_line`, `event_counters`, `ema_scores` → 0.5, `last_switch_ts` → 0.0  
Truncates: `predictions.csv` to header  
Deletes: `command.txt`, `drift.csv`, `drift_kl.json`  
Preserves: `scaler.pkl`, `reference_distribution.json`, `versionedMR/`, model weights

---

## 4. Validation

Run at any time to check a config + data:

```
python scripts/validate_dataset.py --config <name>
```

`run_managed_system.py` calls the validator automatically on startup.  A failed validation aborts the run and prints the error table.  To bypass (not recommended): pass `--skip-validation` to `run_managed_system.py`; this stamps `"validation_skipped": true` into the run manifest.

---

## 5. Adding a New Dataset

1. Preprocess your data to match §1 (regression) or §2 (CV) schema.
2. Copy `configs/datasets/_template.json`, fill in your values, save as `configs/datasets/<name>.json`.
3. Run `python scripts/validate_dataset.py --config <name>` — fix any errors.
4. Run the appropriate init script.
5. Start the managed system: `python run_managed_system.py --system reg` (or `cv`).

No code changes are needed. If you encounter a case where code must change, file an issue — it is a contract violation.

# Claude Code — Dataset Integration Audit & Update Instructions

> **Hand this entire document to Claude Code before it touches any file.**
> Read every section in full before writing a single line of code.
> This is an audit-first, fix-second workflow. Do not change anything
> until you have read the relevant files and confirmed what is actually there.

---

## 0. Project Context

This is the HarmonE Journal Extension project. HarmonE is a self-adaptive MLOps
framework that wraps an ML pipeline in a MAPE-K loop to manage sustainability
trade-offs (accuracy vs energy vs cost) at runtime. The codebase lives at
`sa4s-serc/HarmonE-tool`. The current branch is `fix/bugs-phase1`.

The extension adds six datasets beyond the original single PeMS sensor benchmark.
All six datasets are already downloaded and on disk. Your job is to audit the
current codebase against the dataset specification below, identify every gap, and
close every gap. No dataset-specific logic should exist in MAPE-K core files. All
dataset knowledge lives in adapters and config files.

**Hardware context** (relevant for energy measurement code):
- AMD Ryzen AI 7 350 CPU (Zen 5, RAPL-compatible MSRs but verify)
- NVIDIA RTX 5060 Laptop GPU (sm_120, Blackwell — does NOT support
  `nvmlDeviceGetTotalEnergyConsumption()`, must use power.draw polling)
- Running Linux (Arch)
- Python 3.13, virtual environment at `~/.venvs/harmone`

---

## 1. The Six Datasets — Authoritative Specification

These are the ground-truth specifications. Every adapter, config, and manifest
must match these exactly.

### R1 — PeMS (Traffic Flow Regression)

- **Task**: univariate autoregressive forecasting
- **Input**: previous N timesteps of traffic flow (vehicles/5min) from one sensor
- **Output**: traffic flow at timestep N+1
- **Drift type**: recurring daily/weekly periodicity with incident shocks
- **Expected drift behaviour**: VMR reuse should fire correctly (rush-hour patterns recur)
- **Model spectrum**: LR (Linear Regression), SVR (Support Vector Regression), LSTM
- **Accuracy metric at runtime**: R² (self-labeling — label arrives as next sensor reading)
- **Energy measurement**: CPU only (`EnergyContext("cpu")`)
- **Preprocessing contract**:
  ```
  timestamp (datetime)
  value (float, vehicles per 5 minutes)
  ```
- **Notes**: this is the original HarmonE benchmark dataset. A second sensor node
  beyond the original is being used. Sensor selection should be documented in config.

### R2 — UCI Electricity Load Diagrams (Electricity Demand Regression)

- **Task**: univariate autoregressive forecasting per meter
- **Input**: previous N timesteps of electricity consumption (kW per 15 minutes)
- **Output**: consumption at timestep N+1
- **Drift type**: seasonal recurring (summer/winter cycles, weekday/weekend)
- **Expected drift behaviour**: VMR reuse should fire on seasonal recurrence
  (winter-2012 model reusable in winter-2013)
- **Model spectrum**: LR, SVR, LSTM (same as R1)
- **Accuracy metric at runtime**: R² (self-labeling)
- **Energy measurement**: CPU only (`EnergyContext("cpu")`)
- **Preprocessing contract**:
  ```
  timestamp (datetime)
  value (float, kW — raw dataset is kW per 15 min, NOT kWh)
  ```
- **Raw format**: CSV with semicolon delimiter. Columns: `Datetime`, `MT_001`
  through `MT_370`. Each MT column is one anonymous meter. DST duplicate
  timestamps exist and must be handled (keep first occurrence, drop duplicate).
  Some meters have leading zeros before their first real reading — these are
  customers created after 2011 and should be excluded or handled explicitly.
- **Planned usage**: treat each meter as an independent time series. Select one
  meter (document which in config). Do not mix meters.
- **Unit note**: values are in kW per 15-minute interval. To convert to kWh,
  divide by 4. Document the unit choice explicitly in config and preprocessing.

### R3 — Nord Pool Spot Prices (Electricity Market Price Regression)

- **Task**: univariate autoregressive forecasting
- **Input**: previous N timesteps of day-ahead spot price (EUR/MWh)
- **Output**: price at timestep N+1
- **Drift type**: structural break — long-term regime changes (COVID, energy
  crisis, renewable expansion, policy). VMR reuse expected to fail on regime
  breaks; this is the deliberate stress-test dataset.
- **Expected drift behaviour**: KL / MMD fires. VMR search finds no match.
  System retrains. Retrained model may perform worse than original on the new
  regime. This failure mode should be logged and reported, not hidden.
- **Model spectrum**: LR, SVR, LSTM (same as R1/R2)
- **Accuracy metric at runtime**: R² (self-labeling)
- **Energy measurement**: CPU only (`EnergyContext("cpu")`)
- **Preprocessing contract**:
  ```
  timestamp (datetime, UTC)
  value (float, EUR/MWh spot price)
  ```
- **Raw format**: CSV with columns `HourUTC`, `HourDK`, `PriceArea`,
  `SpotPriceDKK`, `SpotPriceEUR`. Each hour appears once per bidding area
  (DK1, DK2, NO2, SE3, SE4, SYSTEM, etc.). Select a single bidding area
  (DK1 recommended, document in config). Retain `HourUTC` and `SpotPriceEUR`.
  Convert `HourUTC` to a proper datetime index. The crisis period (2021–2022)
  should be within the test/inference split.
- **Note**: this dataset is noted as "Nord Pool" in the documentation. If the
  downloaded data is actually ERCOT (Texas market, USD/MWh), the preprocessing
  contract is the same but the column names differ. Check the actual CSV headers
  and update the adapter accordingly. Do not assume column names.

### C1 — BDD100K (Object Detection)

- **Task**: detect and localise objects (vehicles, pedestrians, traffic lights,
  cyclists, etc.) in dashcam frames
- **Input**: RGB image (1280×720)
- **Output**: bounding boxes with class labels and confidence scores
- **Drift type**: natural photometric/environmental — weather, time of day,
  lighting. Drift is constructed by ordering inference stream using BDD100K's
  attribute metadata: clear-day → overcast → dusk → night → rain.
- **Expected drift behaviour**: luminance histograms detect brightness shift
  (night/rain visible). Embedding drift also fires. Proxy validation experiment
  should show confidence correlates with mAP under these conditions.
- **Model spectrum**: YOLOv8n, YOLOv8s, YOLOv8m
- **Accuracy proxy at runtime**: mean detection confidence across boxes in each
  frame, averaged over the monitoring interval. This is a proxy — not true mAP.
- **Offline ground truth**: bounding box annotations in BDD100K JSON format.
  Used ONLY for proxy validation, never by the runtime MAPE-K loop.
- **Energy measurement**: GPU (`EnergyContext("gpu")`)
- **Preprocessing contract**:
  ```
  sample_id (string, image filename)
  input (string, path to image file)
  label (string, path to annotation file or JSON string of boxes)
  domain (string, BDD100K attribute — "clear_day", "overcast", "night", "rain")
  split (string, "train" / "val" / "test")
  ```
- **Raw format**: images in `images/100k/train/`, `images/100k/val/`,
  `images/100k/test/`. Annotations in `labels/` as JSON. Each JSON entry
  has `attributes` dict containing `weather`, `timeofday`, `scene`.
  Use these attributes to assign the `domain` field.
- **Stream ordering**: sort inference stream by domain in order:
  clear_day → overcast → dusk → night → rain. Within each domain, use
  the original file ordering. This creates a controlled temporal drift sequence.

### C2 — iWildCam via WILDS v2.0 (Image Classification)

- **Task**: classify animal species from camera trap images
- **Input**: full image (no bounding boxes, no crops)
- **Output**: species class label (182 classes)
- **Drift type**: geographic domain shift — each camera trap location is a
  distinct domain. Drift occurs when the inference stream moves from one
  location cluster to another. Background, vegetation, lighting distribution,
  and species prevalence all change. Luminance histograms are largely blind
  to this drift — this is the primary benchmark for embedding-based drift detection.
- **Expected drift behaviour**: luminance KL stays flat or low. Embedding MMD
  fires when location cluster changes. True accuracy drops. Proxy (confidence)
  may or may not track the drop — proxy validation experiment answers this.
- **Model spectrum**: EfficientNet-B0, ResNet-50, ResNet-101
- **Accuracy proxy at runtime**: max softmax probability (top-1 confidence),
  averaged over the monitoring interval
- **Offline ground truth**: species labels from WILDS metadata. Used ONLY
  for proxy validation.
- **Energy measurement**: GPU (`EnergyContext("gpu")`)
- **Preprocessing contract**:
  ```
  sample_id (string, image id from WILDS metadata)
  input (string, path to image file)
  label (int, species class index 0–181)
  domain (string, camera location id)
  split (string, "train" / "id_val" / "ood_val" / "id_test" / "ood_test")
  ```
- **Raw format**: WILDS v2.0 provides `metadata.csv` with columns
  `image_id`, `y` (label), `split`, `location` (domain). Images are in
  a flat directory or subdirectory structure depending on version.
  `categories.csv` maps class indices to species names.
  Use the WILDS Python package's data loader if available, otherwise
  parse `metadata.csv` directly.
- **Stream ordering for drift**: order inference stream by `location` cluster.
  Group locations geographically (use WILDS metadata if it provides coordinates,
  otherwise use location ID ordering). The OOD test split (unseen locations)
  is the primary inference stream.

### C3 — ACDC (Semantic Segmentation)

- **Task**: assign a semantic class to every pixel in a driving scene image
- **Input**: RGB image (adverse condition: fog, night, rain, or snow)
- **Output**: per-pixel class label (19 Cityscapes classes)
- **Drift type**: adverse condition severity — the inference stream is ordered
  by condition: clear → fog → rain → night → snow. Each condition presents
  a qualitatively different visual degradation.
- **Expected drift behaviour**: all drift detectors should fire on condition
  transitions. Embedding drift is most semantically meaningful. True mIoU
  drops significantly under severe conditions (especially night and dense fog).
- **Model spectrum**: SegFormer-B0, SegFormer-B1, SegFormer-B2
- **Accuracy proxy at runtime**: mean max-softmax confidence across all pixels
  in the frame, averaged over the monitoring interval. For segmentation, per-pixel
  confidence is more meaningful than for classification (more pixels = more
  statistical stability per frame).
- **Offline ground truth**: pixel-level semantic masks in `gt_trainval/`.
  Used ONLY for proxy validation (computing true mIoU vs confidence proxy).
  `ignore_index = 255` — pixels with this value must be excluded from mIoU
  computation.
- **Energy measurement**: GPU (`EnergyContext("gpu")`)
- **Preprocessing contract**:
  ```
  sample_id (string, image filename stem)
  input (string, path to RGB image in rgb_anon/)
  label (string, path to ground truth mask in gt_trainval/)
  domain (string, "fog" / "night" / "rain" / "snow")
  split (string, "train" / "val" / "test")
  ```
- **Raw format**: two separate downloads.
  `rgb_anon/` contains adversely-conditioned RGB images organised as
  `rgb_anon/{condition}/{split}/{sequence}/{filename}.png`.
  `gt_trainval/` contains semantic masks organised as
  `gt_trainval/gt/{condition}/{split}/{sequence}/{filename}_gt_labelTrainIds.png`.
  The `_gt_labelTrainIds.png` suffix distinguishes the training-ID mask
  (0–18 + 255) from the raw label mask. Use `labelTrainIds` files.
- **Stream ordering**: clear (from Cityscapes reference images if included)
  → fog → rain → night → snow. Within each condition, use the provided
  sequence ordering.

---

## 2. Unified Internal Contracts

After preprocessing, every dataset feeds into HarmonE via one of two contracts.
These contracts must be enforced by the adapter layer. Core MAPE-K files must
never reference dataset-specific column names or file formats.

### Regression Contract

```python
# Produced by every regression adapter's __iter__ or get_batch() method
{
    "timestamp": datetime,       # UTC datetime of the observation
    "value": float,              # the scalar to predict
    "sensor_id": str,            # which sensor/meter/area (for logging)
}
```

### CV Contract

```python
# Produced by every CV adapter's __iter__ or get_sample() method
{
    "sample_id": str,            # unique identifier
    "input_path": str,           # absolute path to input image/file
    "label": Any,                # ground truth (int for classification,
                                 # str path for detection/segmentation)
    "domain": str,               # which domain/condition this sample belongs to
    "split": str,                # train/val/test
}
```

---

## 3. What to Audit — File-by-File Checklist

Work through every item in this list. For each item, read the relevant file(s)
first, then determine whether a change is needed, then make the change.
Do not assume anything about what the file currently contains.

### 3.1 Dataset Config Files

**Location**: `tool/managed_system_*/configs/` or `tool/configs/datasets/`
(check both, the location may vary)

For each of the six datasets, a config JSON must exist with the following fields:

```json
{
    "dataset_id": "r1_pems",
    "dataset_name": "PeMS Traffic Flow",
    "task": "regression",
    "data_path": "<absolute or relative path to preprocessed data>",
    "raw_data_path": "<path to raw downloaded data>",
    "adapter": "regression",
    "model_spectrum": ["lr", "svr", "lstm"],
    "sensor_id": "<which sensor node is being used>",
    "input_window": 5,
    "thresholds": {
        "min_score": 0.78,
        "max_energy": 0.6,
        "beta": 0.95,
        "gamma": 0.8,
        "alpha": 0.1,
        "tau_drift": 0.5,
        "E_ref": 0.7,
        "delta": 0.1,
        "E_m": 0,
        "E_M": 25000,
        "tau_drift_source": "default_pending_calibration"
    },
    "energy_domain": "cpu",
    "proxy_metric": "r2",
    "drift_detector": "kl_fixed_ref",
    "notes": "<any dataset-specific notes>"
}
```

**Checks to perform**:
1. Does a config exist for each of the six datasets?
2. Do the `data_path` values point to real directories that exist on disk?
3. Are the `model_spectrum` values consistent with what models are actually
   available in `managed_system_*/knowledge/` or `models/`?
4. Are the `thresholds` values clearly marked as pending calibration if they
   are defaults? Add `"tau_drift_source": "default_pending_calibration"` if
   not yet calibrated via pilot run.
5. Is `energy_domain` set to `"cpu"` for regression and `"gpu"` for CV?

**Action**: create or update configs to match the spec above for all six datasets.

### 3.2 Dataset Adapters

**Location**: `tool/managed_system_*/adapters/` or `tool/core/adapters/`

**Expected adapters**:

For regression:
- `regression_adapter.py` — a single adapter that reads the unified
  `timestamp, value` CSV format for all three regression datasets.
  Dataset differences (raw format, sensor selection, unit conversion)
  are handled in preprocessing scripts, NOT in this adapter.

For CV:
- `base.py` — `CVAdapter` abstract base class with methods:
  `load_model`, `run_inference`, `extract_proxy`, `extract_embedding`,
  `compute_offline_accuracy`
- `detection.py` — `DetectionAdapter` for BDD100K / YOLO models
- `classification.py` — `ClassificationAdapter` for iWildCam / EfficientNet / ResNet
- `segmentation.py` — `SegmentationAdapter` for ACDC / SegFormer

**Checks to perform for each adapter**:

1. Does the adapter exist?
2. Does it implement all required methods from the base class?
3. For `extract_proxy()`: does it return a float in [0, 1]? Does it handle the
   case where the model returns no detections / empty output (return 0.0, not error)?
4. For `extract_embedding()`: does it use an intermediate layer, NOT the output
   head? Does it return a 1-D numpy array? Does it handle GPU tensors correctly
   (detach, move to CPU before numpy)?
5. For `compute_offline_accuracy()`: is it clearly gated so it is never called
   during the runtime MAPE-K loop? It should only be called from
   `experiments/proxy_validation.py`.
6. For `detection.py`: is the YOLOv8 backbone hook registered correctly?
   Layer 9 is the backbone output in standard YOLOv8 — verify this is correct
   for n/s/m variants (they share the same architecture up to depth/width scaling).
7. For `classification.py`: is the penultimate layer hook correct for both
   EfficientNet-B0 and ResNet-50/101? These have different architectures —
   the hook registration must handle both.
8. For `segmentation.py`: is SegFormer being loaded from a pretrained checkpoint
   or from scratch? Pretrained on Cityscapes is required — verify the model path
   in the config points to fine-tuned weights, not random init.
9. Does any adapter import dataset-specific paths or hardcoded filenames?
   If so, move those to config. Adapters must be path-agnostic.

### 3.3 Preprocessing Scripts

**Location**: `tool/scripts/preprocess/` or `tool/preprocessing/` or `scripts/`

A preprocessing script must exist for each dataset that converts raw downloaded
data into the unified internal contract format. Check whether each script exists
and whether it correctly handles the raw format described in Section 1.

**R1 — PeMS preprocessing**:
- Input: raw PeMS CSV or pre-processed sensor CSV
- Output: `timestamp, value` CSV, chronologically ordered
- Check: is a specific sensor node selected and documented?
- Check: are missing values handled (PeMS has sensor dropouts)?
- Check: is the output file path consistent with `data_path` in the config?

**R2 — UCI Electricity preprocessing**:
- Input: `LD2011_2014.txt` (semicolon-delimited, European decimal format)
- Output: `timestamp, value` CSV for a single selected meter
- Check: DST duplicate handling — 2012/03/25 and 2013/03/31 (and equivalents)
  have a 25-hour day in October and 23-hour day in March.
  The preprocessing script must handle or document this explicitly.
- Check: leading-zero rows (meters created after 2011) — are they excluded
  or handled?
- Check: is the selected meter documented in config?
- Check: is the kW vs kWh unit documented? The raw values are kW per 15 minutes.
  If converting to kWh, divide by 4. If keeping kW, document it.
- Check: European decimal format — the file uses commas as decimal separators
  in some versions. Verify the parser handles this.

**R3 — Nord Pool / ERCOT preprocessing**:
- FIRST: inspect the actual downloaded file headers. Do not assume column names.
  Run `head -5 <datafile>` and report what you see.
- If Nord Pool: columns include `HourUTC`, `PriceArea`, `SpotPriceEUR`.
  Select one bidding area (DK1) and retain `HourUTC` + `SpotPriceEUR`.
- If ERCOT: columns will differ. Common ERCOT columns include
  `DeliveryDate`, `HourEnding`, `SettlementPointPrice`, `SettlementPoint`.
  Select one settlement point (e.g. `HB_NORTH` for the North hub price)
  and retain timestamp + price.
- Output: `timestamp, value` CSV regardless of source
- Check: is the selected area/hub documented in config?
- Check: is the crisis period (Feb 2021 for ERCOT, 2021-2022 for Nord Pool)
  within the test/inference split?
- Check: are missing hours handled? Both markets have occasional missing data
  for off-peak hours.

**C1 — BDD100K preprocessing**:
- Input: BDD100K images + annotation JSONs
- Output: manifest CSV with `sample_id, input_path, label_path, domain, split`
- Check: is the `domain` field correctly derived from BDD100K attributes?
  The JSON has `attributes.weather` (clear/overcast/partly cloudy/foggy/rainy/snowy)
  and `attributes.timeofday` (daytime/dawn/dusk/night). The domain mapping should be:
  - daytime + clear → `clear_day`
  - daytime + overcast → `overcast`
  - dawn/dusk → `dusk`
  - night → `night`
  - rainy/snowy → `rain` or `snow`
  Document the exact mapping in the script.
- Check: is the manifest sorted by domain in drift order
  (`clear_day → overcast → dusk → night → rain`)?
- Check: do all `input_path` values in the manifest point to files that exist?
  Run a spot check on 10 random rows.

**C2 — iWildCam preprocessing**:
- Input: WILDS v2.0 directory structure with `metadata.csv`
- Output: manifest CSV with `sample_id, input_path, label, domain, split`
- Check: is `metadata.csv` parsed correctly? Columns: check actual CSV headers.
- Check: is `domain` mapped from the `location` field?
- Check: is the OOD test split used as the primary inference stream?
- Check: do all `input_path` values point to real files? WILDS v2.0 uses
  a specific directory structure — verify the path construction is correct.
- Check: is the manifest sorted by `domain` (location) to create the
  geographic drift sequence?

**C3 — ACDC preprocessing**:
- Input: `rgb_anon/` and `gt_trainval/` directories
- Output: manifest CSV with `sample_id, input_path, label_path, domain, split`
- Check: is the path pairing correct? Each image in `rgb_anon/fog/val/seq/`
  must be paired with its mask in `gt_trainval/gt/fog/val/seq/_gt_labelTrainIds.png`.
  The filename stems match; only the suffix and directory differ.
- Check: is `domain` set from the condition directory name (fog/night/rain/snow)?
- Check: is `ignore_index = 255` documented in the config for use during
  mIoU computation?
- Check: is the manifest sorted in condition order: fog → rain → night → snow?
  (The proxy validation should see the full range of conditions.)
- Check: are the `_gt_labelTrainIds.png` files used, not the `_gt_labelIds.png`?
  The trainIds version uses the 0–18 + 255 encoding; the rawIds version uses
  a different encoding not compatible with standard Cityscapes training.

### 3.4 Energy Measurement — `core/energy/`

**Checks to perform**:

1. Does `core/energy/context.py` contain the `EnergyContext` class?
2. Does it probe for RAPL availability at first call (not at import time)?
3. Does it probe for NVML availability at first call?
4. If NVML is unavailable (which it likely is on RTX 5060 — `nvmlDeviceGetTotalEnergyConsumption()` 
   is not supported), does it automatically fall back to `NvidiaPowerPollingMeter`?
5. Does `core/energy/gpu_polling.py` contain `NvidiaPowerPollingMeter` with
   50ms polling interval, `start()`, `stop()`, `energy_joules` property?
6. Does the polling meter use `subprocess.run` with `nvidia-smi --query-gpu=power.draw
   --format=csv,noheader,nounits`?
7. Is `get_backend_status()` exported from `core/energy/context.py`?
8. Is `EnergyContext("cpu")` used in all regression execute.py files?
9. Is `EnergyContext("gpu")` used in CV inference.py?
10. Is `EnergyContext("both")` available and used in retrain.py?
11. Are `cpu_joules`, `gpu_joules`, and `total_joules` all logged separately
    in predictions.csv?
12. Is `mape_k_energy_uJ` reported in telemetry? (This is the overhead energy
    of the MAPE-K loop itself, separate from inference energy.)

**If any of the above are missing**: implement them. The `EnergyContext` class
specification is in the CV extension guide document. Do not use pyRAPL directly
anywhere in the codebase after this pass.

### 3.5 Drift Detection — `core/drift/`

**Checks to perform**:

1. Does `core/drift/kl_fixed_ref.py` exist with the fixed-reference KL implementation?
   (Computes KL between current window and a stored training distribution,
   NOT between two rolling windows.)
2. Does `core/drift/kl_rolling.py` exist as the legacy rolling-window KL?
   (Kept as a secondary signal for comparison.)
3. Does `core/drift/embedding_mmd.py` exist with `mmd_squared()` and `rbf_kernel()`?
4. Does `mmd_squared(X, Y)` implement the unbiased estimator correctly?
   The formula is:
   ```
   MMD² = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]
   ```
   with diagonal terms excluded from the within-set expectations.
   Check the implementation against this formula.
5. Does the bandwidth default to the median heuristic when not specified?
6. Is there a config key `drift_detector` that selects which detector is active
   per dataset? Options should be: `"kl_fixed_ref"`, `"kl_rolling"`,
   `"embedding_mmd"`. Regression datasets default to `"kl_fixed_ref"`.
   CV datasets that need semantic drift detection use `"embedding_mmd"`.
7. Does the monitor correctly route to the selected detector based on config?
8. For the embedding path: are embeddings stored in `predictions.csv` as
   JSON-serialised float lists? This is required for the drift detector to
   compute MMD over a rolling window of stored embeddings.

### 3.6 Proxy Validation — `experiments/proxy_validation.py`

This script is the first experiment that must run before any other CV results
are collected. It measures whether the runtime confidence proxy actually
correlates with true offline accuracy under drift.

**Checks to perform**:

1. Does `experiments/proxy_validation.py` exist?
2. Does it accept a dataset config path as input?
3. Does it loop over frames in drift-ordered sequence?
4. For each monitoring interval does it compute both:
   - `confidence_proxy`: the value returned by `adapter.extract_proxy(result)`
   - `true_accuracy`: the value returned by `adapter.compute_offline_accuracy(result, label_path)`
5. Does it output a CSV with columns: `interval, confidence_proxy, true_accuracy,
   model, dataset, condition`?
6. Does it compute and print Spearman correlation ρ between proxy and true accuracy?
7. Is it clearly marked as an OFFLINE ANALYSIS ONLY script — i.e. it must never
   be called during the runtime MAPE-K loop?
8. Does it support all three CV adapters (detection, classification, segmentation)?

If this script does not exist or is incomplete, implement it. It must be runnable
as a standalone script: `python experiments/proxy_validation.py --config configs/datasets/bdd100k.json`

### 3.7 Experiment Harness — `experiments/run_experiment.py`

**Checks to perform**:

1. Does a harness script exist that can run a full experiment for a given
   dataset config?
2. Before starting each run, does it:
   - Clear `knowledge/command.txt` (prevents stale command replay — bug L2)
   - Reset all counters in `mape_info.json` to zero
   - Reset EMA scores to 0.5 in `mape_info.json`
   - Clear `predictions.csv` to header-only
   - Log `get_backend_status()` output to the run log
3. After each run, does it snapshot:
   - Final adaptation counts (switches S, retrains R, VMR events V)
   - Total energy (cpu_joules, gpu_joules, total_joules)
   - Final EMA score per model
   - Run duration
4. Does it support `--n-runs 5` (5 independent runs per approach)?
5. Does it enforce a 20-minute cooldown between runs?
   (Or at minimum flag that cooldown must be done manually.)
6. Does it support `--approach` flag to select planner strategy?

### 3.8 MAPE-K Core Files — Contamination Check

These files must contain NO dataset-specific logic after this audit.

Files to check:
- `managed_system_regression/mape_logic/monitor.py`
- `managed_system_regression/mape_logic/analyse.py`
- `managed_system_regression/mape_logic/plan.py`
- `managed_system_regression/mape_logic/execute.py`
- `managed_system_cv/mape_logic/monitor.py`
- `managed_system_cv/mape_logic/analyse.py`
- `managed_system_cv/mape_logic/plan.py`
- `managed_system_cv/mape_logic/execute.py`

**For each file, grep for**:
- `bdd100k`, `pems`, `electricity`, `nord`, `ercot`, `iwildcam`, `acdc`
- Any hardcoded file paths (strings starting with `/home/`, `./data/`, etc.)
- Any hardcoded threshold values (numbers like `0.78`, `0.5`, `0.07`) not
  loaded from config
- Any hardcoded model names (`"yolo"`, `"lstm"`, `"resnet"`) not loaded from config
- Any import of dataset-specific libraries at the top of the file

If any of these are found, move them to config and load them at runtime.

### 3.9 The `knowledge/` Directory State

Each managed system has a `knowledge/` directory that holds runtime state.
This state must be cleanable between runs.

**Checks to perform**:

1. Does `knowledge/command.txt` get cleared on managed system startup?
   Check `manage.py` for a startup clear. If missing, add:
   ```python
   # At top of run_mape_loop(), before the command listener starts
   command_file = os.path.join(KNOWLEDGE_DIR, "command.txt")
   if os.path.exists(command_file):
       open(command_file, "w").close()
   ```
2. Does `knowledge/scaler.pkl` exist for regression?
   If not, `scripts/init_scaler.py` must be run. Check the script exists
   and documents what split it fits on (training split only, never test).
3. Does `knowledge/reference_distribution.json` exist?
   This stores the training-set value distribution for fixed-reference KL.
   If not, `scripts/init_reference.py` must be run or `init_scaler.py`
   must also compute and save this.
4. Is there a `scripts/reset_run.py` that performs the full between-run reset
   described in Section 3.7 item 2? If not, create it. It must be idempotent
   (safe to run multiple times).

### 3.10 Planners — `core/planners/`

The codebase should have multiple planner strategies, each as a separate file
implementing the same interface. Check:

1. Does a planner interface exist (function signature or abstract class)?
   Expected: `plan(trigger, knowledge) -> dict or None`
2. Do the following planners exist:
   - `random_switch.py` — random alternative model selection (honest rename
     of the original buggy "switch" baseline)
   - `greedy_switch.py` — select highest-EMA alternative (the correct Switch)
   - `harmone_original.py` — EMA-greedy + ε-exploration + VMR + selective retrain
   - `violation_aware.py` — different response per violation type
     (energy violation → switch to lowest-energy above S_min;
      score violation → switch to highest-EMA within energy budget;
      drift → VMR lookup then retrain if no match)
   - `pareto.py` — Pareto-aware planner maintaining separate EMA_accuracy and
     EMA_energy per model, Chebyshev distance selection
3. Is the active planner selected via config (not hardcoded)?
4. Does `violation_aware.py` correctly distinguish between
   `trigger == "energy"`, `trigger == "score"`, `trigger == "drift"`?

### 3.11 VMR (Versioned Model Repository) — `core/vmr.py`

**Checks to perform**:

1. Does VMR matching use the unified key contract:
   ```python
   {
       "action": "replace" | "retrain" | None,
       "version": str | None,   # path to versioned model weights
   }
   ```
   This was the regression bug B2 — the key mismatch that made VMR reuse
   unreachable. Confirm the fix is in place and consistent across both
   regression and CV managed systems.
2. For regression: does VMR match on value-distribution histograms (fixed-ref KL)?
3. For CV: does VMR match on embedding-space statistics (Fréchet distance
   or MMD on stored feature centroids)?
4. Does each versioned model entry store:
   - model weights path
   - the data distribution signature it was trained on
     (histogram for regression, embedding centroid/covariance for CV)
   - training timestamp
   - training dataset slice (which rows/frames were used)

---

## 4. Specific Bugs to Verify Are Fixed

These are the bugs diagnosed in the Phase 1 audit. Verify each is actually
fixed in the current code, not just documented.

| Bug | Location | Verification method |
|-----|----------|---------------------|
| B1 — Energy threshold monotonically increasing | `analyse.py` regression | Confirm formula: `tau = clamp(tau + delta * (E_ref - E_used), lo, hi)`. E_ref must be a separate value from max_energy. |
| B2 — VMR dict key mismatch | `analyse.py` + `plan.py` regression | Confirm `analyse_drift()` returns `{"action": ..., "version": ..., "drift_detected": ...}` |
| B3 — Rolling drift reference | `monitor.py` regression | Confirm two signals: `kl_fixed_ref` (vs training distribution) and `kl_rolling` (vs previous window). Config selects which triggers adaptation. |
| B4 — Model reload per inference step | `inference.py` regression | Confirm model loaded once at startup, cached in memory, reloaded only when `model_reload.flag` exists |
| B5 — Random KL placeholder | `monitor.py` regression | Confirm returns `None` when fewer than 2*window_size samples exist |
| B6 — Switch counter on no-op | `execute.py` regression | Confirm `model_switches` only incremented when a model switch actually occurs |
| B7 — Scaler leakage | `inference.py` regression | Confirm scaler loaded from `knowledge/scaler.pkl`, never fitted on test data |
| L1 — Missing scaler.pkl | startup | Confirm `scripts/init_scaler.py` exists and `run_managed_system.py` health-checks for scaler before starting |
| L2 — Stale command replay | `manage.py` | Confirm `command.txt` is cleared on startup |
| L3 — PyTorch CUDA sm_120 | environment | Run `python3 -c "import torch; print(torch.cuda.get_device_capability(0))"` — must print `(12, 0)` |
| L4 — None comparison in ACP | `app.py` | Confirm `periodic_secondary_checks` guards against `None` telemetry values before comparison |
| L5 — Thrashing on stale telemetry | downstream of L1/L2 | Fixed by fixing L1+L2 |
| L6 — Counter persistence | `mape_info.json` | Confirm `scripts/reset_run.py` zeros all counters |

For each bug, add a comment in the fixed code: `# BugFix: B1` etc. so it is
traceable.

---

## 5. Output — What to Produce

After the audit and fixes, produce the following:

### 5.1 Audit Report

A file at `AUDIT_REPORT.md` in the repo root containing:
- Every file checked
- For each file: what was found, what was changed (or "no change needed")
- Any bugs newly discovered beyond the list above
- Any dataset-specific files that could not be verified (e.g. raw data not
  at expected path)

### 5.2 Missing File List

If any required file does not exist, list it in `MISSING_FILES.md` with:
- The file path
- What it should contain (one paragraph)
- Whether it blocks experiments from running

### 5.3 Verified Config Files

All six dataset configs must exist and be valid JSON after this pass.
Name them:
```
tool/configs/datasets/r1_pems.json
tool/configs/datasets/r2_uci_electricity.json
tool/configs/datasets/r3_nordpool.json     (or r3_ercot.json if ERCOT)
tool/configs/datasets/c1_bdd100k.json
tool/configs/datasets/c2_iwildcam.json
tool/configs/datasets/c3_acdc.json
```

### 5.4 Verified Preprocessing Manifests

For each dataset, a manifest CSV must exist at the path specified in the config.
If the manifest does not exist, run the preprocessing script to generate it.
If the preprocessing script does not exist, create it first.

After generating, run a validation check:
- Regression: confirm `timestamp` is monotonically increasing, `value` has
  no NaNs or Infs, row count matches expected dataset size
- CV: confirm all `input_path` values point to existing files (spot-check 50
  random rows for large datasets), `domain` values are from the expected set,
  no duplicate `sample_id` values

---

## 6. Do Not Do These Things

- Do not add dataset-specific logic to any file in `mape_logic/`
- Do not hardcode any data paths — all paths come from config
- Do not modify the MAPE-K loop's core logic while doing this audit pass —
  this is an integration pass, not an algorithm change pass
- Do not create separate `managed_system_cv_detection/`,
  `managed_system_cv_classification/`, `managed_system_cv_segmentation/`
  directories — the adapter architecture handles task differences within
  a single `managed_system_cv/`
- Do not change any threshold values to improve results — mark them as
  "pending calibration" and leave calibration for the experiment harness
- Do not remove the `kl_rolling` signal even though it is the legacy method —
  it is kept as a secondary comparison signal for the paper
- Do not call `compute_offline_accuracy()` from anywhere inside the MAPE-K
  loop — this method is for offline proxy validation only
- Do not commit broken state — every commit should leave the regression
  smoke test passing (`python run_managed_system.py` with `reg_harmone` config
  should run without crashing for at least 30 seconds)

---

## 7. Smoke Tests to Run After Changes

Run these in order. If any fails, fix it before proceeding to the next.

```bash
# 1. Confirm PyTorch can use the GPU
python3 -c "import torch; print(torch.cuda.get_device_capability(0))"
# Expected: (12, 0)

# 2. Confirm energy probes
python3 -c "
from core.energy.context import get_backend_status
print(get_backend_status())
"
# Expected: rapl field may be True or False; gpu_backend should be 'nvml' or 'polling_50ms'

# 3. Regression smoke test (30 seconds)
cd tool
python scripts/reset_run.py --domain regression
python scripts/init_scaler.py
python run_managed_system.py
# Expected: no FileNotFoundError, predictions.csv grows, no crash in 30s

# 4. Config validation
python3 -c "
import json, glob
for f in glob.glob('configs/datasets/*.json'):
    with open(f) as fp:
        d = json.load(fp)
    assert 'dataset_id' in d, f
    assert 'task' in d, f
    assert 'thresholds' in d, f
    print(f'OK: {f}')
"

# 5. Manifest validation (runs for all 6 datasets)
python scripts/validate_manifests.py
# Expected: all manifests exist, all paths valid, no NaNs

# 6. Adapter instantiation test
python3 -c "
from managed_system_cv.adapters.detection import DetectionAdapter
from managed_system_cv.adapters.classification import ClassificationAdapter
from managed_system_cv.adapters.segmentation import SegmentationAdapter
print('All adapters import OK')
"

# 7. Proxy validation dry run (BDD100K, first 100 frames only)
python experiments/proxy_validation.py \
    --config configs/datasets/c1_bdd100k.json \
    --max-frames 100
# Expected: outputs CSV with confidence_proxy and true_accuracy columns, prints Spearman ρ
```

---

## 8. Reference — Key File Locations

Use these as a map when navigating the repo. Exact paths may vary — find the
actual location and update this map if it differs.

```
tool/
├── app.py                          ACP engine (Flask, port 5000)
├── run_managed_system.py           MasterWrapper (Flask, port 8080)
├── approach.conf                   Active approach selector
├── core/
│   ├── energy/
│   │   ├── context.py              EnergyContext + get_backend_status()
│   │   └── gpu_polling.py          NvidiaPowerPollingMeter
│   ├── drift/
│   │   ├── kl_fixed_ref.py         Fixed-reference KL divergence
│   │   ├── kl_rolling.py           Rolling-window KL (legacy, kept for comparison)
│   │   └── embedding_mmd.py        MMD² with RBF kernel for CV drift
│   ├── planners/
│   │   ├── random_switch.py        S2 — random alternative
│   │   ├── greedy_switch.py        S3 — highest-EMA alternative
│   │   ├── harmone_original.py     S4 — original paper approach (bug-fixed)
│   │   ├── violation_aware.py      S5 — violation-type-aware response
│   │   └── pareto.py               S6 — Pareto Chebyshev planner
│   └── vmr.py                      VMR match/store logic
├── configs/
│   └── datasets/
│       ├── r1_pems.json
│       ├── r2_uci_electricity.json
│       ├── r3_nordpool.json
│       ├── c1_bdd100k.json
│       ├── c2_iwildcam.json
│       └── c3_acdc.json
├── scripts/
│   ├── init_scaler.py              Fits scaler on training data, saves to knowledge/
│   ├── init_reference.py           Saves training distribution for fixed-ref KL
│   ├── reset_run.py                Clears knowledge state between runs
│   └── validate_manifests.py       Verifies all 6 dataset manifests
├── experiments/
│   ├── proxy_validation.py         Offline proxy vs true accuracy comparison
│   ├── calibrate_drift_threshold.py Calibrates tau_drift per dataset
│   └── run_experiment.py           Full experiment harness with reset/logging
├── managed_system_regression/
│   ├── inference.py                Regression inference loop
│   ├── retrain.py                  Regression retraining
│   ├── adapters/
│   │   └── regression_adapter.py   Reads timestamp,value CSV
│   └── mape_logic/
│       ├── monitor.py
│       ├── analyse.py
│       ├── plan.py
│       └── execute.py
├── managed_system_cv/
│   ├── inference.py                CV inference loop (task-agnostic)
│   ├── retrain.py                  CV retraining (pseudo-label / contrastive)
│   ├── adapters/
│   │   ├── base.py                 CVAdapter interface
│   │   ├── detection.py            BDD100K / YOLO
│   │   ├── classification.py       iWildCam / EfficientNet / ResNet
│   │   └── segmentation.py         ACDC / SegFormer
│   └── mape_logic/
│       ├── monitor.py
│       ├── analyse.py
│       ├── plan.py
│       └── execute.py
└── policies/
    ├── reg_harmone_score.json
    ├── reg_switch_r2.json
    ├── cv_harmone_score.json
    └── cv_switch_confidence.json
```

---

*This document was generated from full project context including paper analysis,
codebase audit findings, live run debugging, dataset specifications, and
architectural design decisions. All six datasets are already downloaded.*
*Last updated: July 2026.*
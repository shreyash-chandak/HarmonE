# Harmonica — Comprehensive Technical Documentation

**Version:** Current (as of July 2026)  
**License:** MIT (Copyright 2025 The Authors)  
**Repository:** `D:/Desktop/HarmonE-tool`

> **Note:** Sections 1–10 below describe the original *Harmonica* tool as published.
> Section 11 documents the v2 architecture extensions added during the HarmonE prototype sprint.

---

## 11. v2 Architecture — HarmonE Prototype Extensions

This section describes structural changes made in the `feat/final-prototype` sprint (July 2026) that extend the original Harmonica architecture. All original MAPE-K semantics are preserved; these are additive extensions.

### 11.1 Unified CV Task-Adapter Layer

The original CV managed system had hard-coded YOLO detection calls throughout `inference.py` and the MAPE logic. The v2 architecture introduces a `TaskAdapter` abstract base class (`tool/adapters/tasks/base.py`) with three concrete implementations:

| Adapter | Task | Key method |
|---------|------|-----------|
| `DetectionAdapter` | Object detection (YOLO) | `extract_proxy()`: mean box confidence; `extract_embedding()`: SPPF hook spatial mean-pool |
| `ClassificationAdapter` | Image classification (torchvision) | `extract_proxy()`: max softmax probability; `weights=None` fix (R5-c) |
| `SegmentationAdapter` | Semantic segmentation (SegFormer) | `extract_proxy()`: max logit; `offline_accuracy()`: bilinear upsample before argmax then mIoU (R5-b) |

Task selection is driven entirely by the `task` key in the dataset config — no code changes are needed to switch between detection, classification, and segmentation.

### 11.2 EmbeddingStore Ring Buffer (§5)

`tool/core/drift/embedding_store.py` implements a float16 ring buffer for storing backbone feature vectors extracted during inference. Key properties:

- **Capacity**: 2 × `drift_window_size` rows (allows both the current window and a rolling-reference window)
- **Storage**: `knowledge/embeddings.f16.npy` + `knowledge/embeddings_index.csv`
- **Cursor persistence**: `mape_info.json["embedding_cursor"]` survives process restarts
- **Thread safety**: atomic cursor update via `os.replace(tmp_path, target_path)`

`EmbeddingStore.last_window(n)` returns the most recent `n` vectors as float32, or `None` if fewer than `n` have been collected (warmup period — no fabricated telemetry, invariant I5).

### 11.3 Fixed-Reference Embedding Drift (§5.2–§5.3)

The paper's luminance-KL detector is preserved as the default and is suitable for luminance-visible drift (clear → night, etc.). For semantic drift invisible to luminance (e.g., geographic domain shift in iWildCam), two embedding-based detectors are now wired:

- **`mmd_embedding`**: unbiased MMD² with RBF kernel; bandwidth frozen at fit time via median heuristic
- **`frechet_embedding`**: Fréchet/Wasserstein-2 distance between Gaussian approximations (cheaper for high dimensions)

Both detectors are fitted against **fixed reference embeddings** extracted at init time by `scripts/init_cv.py` and stored in `knowledge/reference_embeddings.npz`. The fixed-reference design (R3 fix) prevents the detector from tracking gradual drift and going blind, which was a bug in the cv_guide's rolling-reference design.

The pinned `embedding_model` config key (R4) ensures the same backbone is always used for embedding extraction regardless of which model is currently serving inference, preventing embedding-space inconsistency across model switches.

### 11.4 VMR Embedding Signatures (§5.4)

When the active drift detector is embedding-based, model versions stored in `versionedMR/` include an embedding signature file (`{version}_emb_sig.json`) alongside the existing luminance histogram file (`{version}_hist.json`). The signature contains the mean and diagonal covariance of the embedding distribution at retrain time.

Version-matching at drift response time (in `analyse.py`) dispatches based on signature type:
- Luminance config → KL distance between histograms
- Embedding config → Fréchet distance between Gaussian approximations of embedding signatures
- Cross-type mismatch → `ValueError` (invariant I6)

### 11.5 WSL Launcher (§6)

`tool/harmone_start_wsl.sh` launches all three processes (ACP server, dashboard, managed system) in the background with log files and a PID file. A 30-second health poll confirms ports 5000 and 8080 are responsive before reporting success. `tool/harmone_stop.sh` kills from the PID file then runs a psutil sweep for orphan processes.

`tool/harmone_start.sh` (Arch) now sources shared setup from `tool/scripts/_launch_common.sh`, eliminating duplication between the two launchers.

### 11.6 Energy Metering — pyJoules Migration

The original tool used pyRAPL. The v2 implementation uses pyJoules with three backends:
- `_PyJoulesRaplBackend`: Intel RAPL (Linux only; requires `/sys/class/powercap/intel-rapl/` read access)
- `_PyJoulesNvmlBackend`: NVIDIA GPU via NVML
- `_PollingGPUBackend`: polling fallback for GPUs without NVML support
- `_NullBackend`: silent fallback when no probes are available (always used on Windows and WSL)

All nine call sites in the codebase use the unified `core/energy.EnergyMeter` context manager. Energy results include `cpu_valid` and `gpu_valid` flags indicating whether the readings are real (backend available) or zero (null backend).

For full details of all changes from the original paper design, see `CHANGES_FROM_PAPER.md`.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Architecture Overview](#3-architecture-overview)
4. [Component 1: Frontend — `tool/frontend/dashboard.html`](#4-component-1-frontend)
5. [Component 2: Managing System — `tool/app.py`](#5-component-2-managing-system)
6. [Component 3: Master Wrapper — `tool/run_managed_system.py`](#6-component-3-master-wrapper)
7. [Component 4: Managed System — Regression](#7-component-4-managed-system-regression)
8. [Component 5: Managed System — Computer Vision](#8-component-5-managed-system-computer-vision)
9. [MAPE-K Loop Deep Dive](#9-mape-k-loop-deep-dive)
10. [Adaptation Policies](#10-adaptation-policies)
11. [Knowledge Base and Runtime State Files](#11-knowledge-base-and-runtime-state-files)
12. [REST API Reference](#12-rest-api-reference)
13. [Configuration Reference](#13-configuration-reference)
14. [Data Flow Walkthrough](#14-data-flow-walkthrough)
15. [Customization and Extension](#15-customization-and-extension)
16. [Examples Directory](#16-examples-directory)
17. [Dependencies and Environment](#17-dependencies-and-environment)
18. [Setup and Running](#18-setup-and-running)
19. [Artifact Outputs](#19-artifact-outputs)
20. [Troubleshooting](#20-troubleshooting)

---

## 1. Project Overview

**Harmonica** is a research artifact that provides a fully runnable implementation of the **HarmonE** approach for sustainable MLOps. It operationalizes a structured **MAPE-K loop** (Monitor–Analyze–Plan–Execute over a shared Knowledge base) that wraps machine-learning inference pipelines and autonomously maintains sustainability goals at runtime.

### Motivation

Modern Machine-Learning-enabled Systems (MLS) operate under environmental uncertainty: data drift, workload fluctuations, hardware variability, and changing performance expectations. Traditional MLOps practices streamline model development and deployment but offer limited support for detecting and responding to runtime deviations that affect system stability and cost. Harmonica bridges that gap by introducing a **Managing System** that oversees an MLS at runtime and enforces adaptation policies decoupled from system execution.

### Key Capabilities

- **Self-adaptive MAPE-K loop** that autonomously switches models, handles data drift, and enforces energy budgets.
- **Two ML domains supported out of the box:**
  - **Regression** — time-series traffic flow prediction using LSTM, SVM, and Linear models.
  - **Computer Vision (CV)** — object detection using YOLOv8 variants (nano/small/medium).
- **Three operational modes per domain:**
  - **HarmonE (Score/Drift)** — full intelligent adaptive system with composite scoring and drift detection.
  - **Simple Switch** — baseline system that reacts to a single accuracy metric.
  - **Single Model (Monitor-only)** — passively monitors without triggering adaptation.
- **Custom MAPE upload** — researchers can supply their own `monitor.py`, `analyse.py`, `plan.py`, `execute.py`, `manage.py` and dataset to run on top of the provided inference engines.
- **Web dashboard** — real-time visualization of telemetry, model distribution, adaptation events, and policy configuration.
- **Energy monitoring** — uses `pyRAPL` (Intel RAPL interface) to measure inference and MAPE-K loop energy consumption in microjoules.

---

## 2. Repository Structure

```
HarmonE-tool/
├── README.md                               # Setup guide and experiment walkthrough
├── LICENSE.md                              # MIT License
├── documentation.md                        # This file
├── docs/
│   ├── architecture.md                     # Architecture narrative (maps code to paper)
│   ├── architecture-diagram.png            # System architecture diagram
│   ├── rest_endpoints.md                   # Condensed REST API reference
│   └── user-study.xlsx                     # User study data
├── examples/
│   ├── HarmonE/                            # Reference MAPE files (complete HarmonE implementation)
│   │   ├── monitor.py
│   │   ├── analyse.py
│   │   ├── plan.py
│   │   ├── execute.py
│   │   ├── dataset.csv
│   │   └── dataset.csv
│   └── templates/                          # Skeleton MAPE files for building custom systems
│       ├── monitor.py
│       ├── analyse.py
│       ├── plan.py
│       ├── execute.py
│       └── dataset.csv
└── tool/
    ├── app.py                              # Managing System Flask server (port 5000)
    ├── approach.conf                       # Active approach identifier (single-line plaintext)
    ├── harmone_start.sh                    # Unified setup + multi-terminal launch script
    ├── run_managed_system.py               # Master Wrapper / Adaptation Handler (port 8080)
    ├── frontend/
    │   └── dashboard.html                  # Single-page web application (port 8000)
    ├── policies/                           # Persisted JSON adaptation policy files
    │   ├── reg_harmone_score.json
    │   ├── reg_switch_r2.json
    │   ├── reg_single_lstm.json
    │   ├── reg_single_svm.json
    │   ├── reg_single_linear.json
    │   ├── cv_harmone_score.json
    │   ├── cv_switch_confidence.json
    │   ├── cv_single_yolo_s.json
    │   ├── cv_single_yolo_m.json
    │   ├── custom_regression_policy.json
    │   └── custom_cv_policy.json
    ├── managed_system_regression/          # Regression managed system
    │   ├── inference.py                    # Streaming inference engine (LSTM/SVM/Linear)
    │   ├── retrain.py                      # Fine-tune models on drift data
    │   ├── approach.conf                   # Local mode config (written by master wrapper)
    │   ├── knowledge/                      # Runtime state
    │   │   ├── dataset.csv                 # Traffic flow input data
    │   │   ├── model.csv                   # Currently active model name
    │   │   ├── predictions.csv             # Log of all inferences
    │   │   ├── thresholds.json             # Energy/score thresholds and EMA parameters
    │   │   └── mape_info.json              # EMA scores, event counters, recovery state
    │   ├── models/                         # Pre-trained model weights
    │   │   ├── lstm.pth
    │   │   ├── svm.pkl
    │   │   └── linear.pkl
    │   ├── mape_logic/
    │   │   ├── monitor.py
    │   │   ├── analyse.py
    │   │   ├── plan.py
    │   │   ├── execute.py
    │   │   └── manage.py
    │   └── versionedMR/                    # Versioned model archive (created at runtime)
    │       └── <model_name>/version_N/     # Retrained model weights + training data
    └── managed_system_cv/                  # CV managed system
        ├── inference.py                    # YOLO inference engine
        ├── retrain.py                      # YOLO fine-tuning with drift-aware augmentation
        ├── approach.conf                   # Local mode config
        ├── knowledge/                      # Runtime state
        │   ├── model.csv
        │   ├── predictions.csv
        │   ├── thresholds.json
        │   ├── mape_info.json
        │   └── event_log.csv
        ├── models/                         # YOLOv8 model weights
        │   ├── yolo_n.pt
        │   ├── yolo_s.pt
        │   └── yolo_m.pt
        ├── mape_logic/
        │   ├── monitor.py
        │   ├── analyse.py
        │   ├── plan.py
        │   ├── execute.py
        │   └── manage.py
        ├── data/bdd100k/images/test/       # BDD100K test images
        ├── versionedMR/                    # Versioned YOLO models + luminance histograms
        └── utility/
            └── drift_utils.py              # Luminance histogram + KL divergence utilities
```

---

## 3. Architecture Overview

Harmonica consists of **three primary components** operating as separate processes that communicate over localhost HTTP.

```
┌────────────────────────────────────────────────────────────────────┐
│                         Web Browser                                │
│              http://localhost:8000/dashboard.html                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Frontend (dashboard.html)                                    │  │
│  │  • Welcome screen / approach selector                       │  │
│  │  • Policy Management tab                                    │  │
│  │  • Live Dashboard tab (Chart.js, polls every 1 second)      │  │
│  │  • History tab (adaptation event cards)                     │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
└─────────────────────────│──────────────────────────────────────────┘
                          │ HTTP (REST)
                          ▼
┌────────────────────────────────────────────────────────────────────┐
│  Managing System — app.py                    [port 5000]           │
│                                                                    │
│  In-memory KNOWLEDGE_BASE                                          │
│    policies{}  telemetry_data{}  intervention_logs{}               │
│                                                                    │
│  MAPE-K (server-side):                                             │
│    receive_telemetry() → analyze_telemetry() → plan_and_execute()  │
│    periodic_secondary_checks() [background thread, every 30s]      │
│                                                                    │
│  Endpoints:                                                        │
│    POST /api/policy         POST /api/telemetry                    │
│    GET  /api/knowledge/<id> POST /api/write-approach               │
│    POST /api/reset          POST /api/save-policy                  │
│    POST /api/set-model      POST /api/start-managed-system         │
│    POST /api/stop-managed-system                                   │
│    POST /api/upload-custom-mape                                    │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ HTTP (tactic commands)
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│  Master Wrapper — run_managed_system.py      [port 8080]           │
│                                                                    │
│  Reads approach.conf → resolves LOGIC_PATH                         │
│  Imports monitor.py dynamically (importlib)                        │
│  Spawns subprocesses: inference.py + mape_logic/manage.py          │
│  push_telemetry() thread: calls monitor functions → POST /telemetry│
│                                                                    │
│  Handler Endpoints:                                                │
│    POST /adaptor/tactic    → writes knowledge/command.txt          │
│    POST /adaptor/shutdown  → kills all subprocesses                │
│    GET  /adaptor/health                                            │
└──────┬────────────────────────────────────────────────────────────┘
       │ subprocess                          │ subprocess
       ▼                                    ▼
┌──────────────┐                   ┌──────────────────────────┐
│ inference.py │                   │  mape_logic/manage.py    │
│              │                   │                          │
│ Reads:       │                   │  Polls command.txt       │
│  model.csv   │                   │  every 1–5 seconds       │
│  dataset.csv │                   │                          │
│              │                   │  On command received:    │
│ Writes:      │                   │    execute_mape()        │
│  predictions │                   │    execute_drift()       │
│  .csv        │                   │    execute_simple_switch │
└──────────────┘                   └──────────────────────────┘
```

### Approach Determination Chain

1. `dashboard.html` calls `POST /api/write-approach` with a dashboard key (e.g. `"reg_harmone_score"`).
2. `app.py` maps it to a short token (e.g. `"reg_harmone"`) and writes it to `tool/approach.conf`.
3. `run_managed_system.py` reads `approach.conf`, splits on `_` to get `system_type` (`reg`/`cv`) and `run_mode` (`harmone`/`switch`/`single`), resolves `LOGIC_PATH` (`managed_system_regression` or `managed_system_cv`), and writes `{run_mode}_acp` (e.g. `"harmone_acp"`) to the local `<LOGIC_PATH>/approach.conf`.
4. `mape_logic/manage.py` reads its local `approach.conf` to decide which execution loop to run.

---

## 4. Component 1: Frontend

**File:** `tool/frontend/dashboard.html`  
**Served by:** `python3 -m http.server 8000` from the `tool/frontend/` directory  
**Dependencies (CDN):**
- Tailwind CSS JIT — utility-first styling
- Chart.js 4.4.0 — time-series and doughnut charts
- chartjs-adapter-date-fns 3.0.0 — time-axis support
- chartjs-plugin-annotation 3.0.1 — vertical line annotations for adaptation events

### 4.1 JavaScript Constants

| Constant | Value | Purpose |
|---|---|---|
| `ACP_SERVER_URL` | `"http://localhost:5000"` | Base URL for all backend calls |
| `POLLING_INTERVAL` | `1000` ms | How often the dashboard fetches new data |
| `CHART_MAX_DATA_POINTS` | `10` | Maximum data points displayed per chart |

### 4.2 Global JavaScript State

| Variable | Type | Purpose |
|---|---|---|
| `currentPolicyId` | string | Policy ID being actively monitored |
| `pollingIntervalId` | number | Handle returned by `setInterval` |
| `chartInstances` | object | Map of chart names → Chart.js instances |
| `globalKnowledge` | object | Last fetched `/api/knowledge` response |
| `currentSystemType` | string | `"regression"` or `"cv"` (used for model selection modal) |

### 4.3 UI Views

The page has two top-level views toggled via DOM visibility:

**Welcome Screen (`#welcome-screen`):** Approach selection landing page with buttons for all built-in modes and a custom upload modal.

**Dashboard Wrapper (`#dashboard-wrapper`):** The main application containing three tabs:
1. **Policy Management** — view and edit the active policy JSON; register it with the server.
2. **Live Dashboard** — four real-time charts; Start/Stop buttons.
3. **History** — adaptation event cards with mini contextual charts.

### 4.4 Built-in Approach Presets (`HARMONY_PRESETS`)

The dashboard hard-codes policy objects for all standard modes. These are sent to the server on approach selection.

| Preset Key | `policy_id` | Monitored QA | Primary Boundary | Secondary Boundary |
|---|---|---|---|---|
| `reg_harmone_score` | `reg_harmone_score` | `score` | LESS_THAN 0.78 | `kl_div` GREATER_THAN 0.10 → `handle_data_drift` |
| `reg_harmone_drift` | `reg_harmone_drift` | `kl_div` | GREATER_THAN 0.75 | — |
| `reg_switch_r2` | `reg_switch_r2` | `r2_score` | LESS_THAN 0.70 | — |
| `cv_harmone_score` | `cv_harmone_score` | `score` | LESS_THAN 0.54 | `kl_div` GREATER_THAN 0.07 → `handle_data_drift` |
| `cv_harmone_drift` | `cv_harmone_drift` | `kl_div` | GREATER_THAN 0.07 | — |
| `cv_switch_conf` | `cv_switch_confidence` | `confidence` | LESS_THAN 0.70 | — |

Single-model presets (`reg_single_*`, `cv_single_*`) use a threshold of `0.0` with empty tactics and secondary boundaries, effectively monitor-only.

### 4.5 Key JavaScript Functions

#### `initializeCharts()`
Creates four Chart.js instances:
- `chartInstances.main` — primary metric line chart (`#main-metric-chart`).
- `chartInstances.qa1` / `chartInstances.qa2` — associated quality attribute line charts.
- `chartInstances.modelChart` — doughnut chart showing model usage distribution.

All line charts use `type: "time"` on the x-axis (second granularity, `HH:mm:ss` format), `animation: { duration: 0 }` for real-time rendering, and the annotation plugin.

#### `fetchKnowledge()`
Calls `GET /api/knowledge/<currentPolicyId>?t=<Date.now()>`. Includes a **race-condition guard**: captures `requestedPolicyId` before the async call and discards the response if `currentPolicyId` changed while the request was in-flight. On success, calls `updateDashboardCharts`, `updatePolicyForm`, and `updateHistoryFeed`.

#### `updateDashboardCharts(data)`
Main rendering function called on each 1-second poll:
1. Slices the last `CHART_MAX_DATA_POINTS` from `telemetry_history`.
2. Determines a `validModels` whitelist based on `currentPolicyId` prefix (`cv_` → YOLO; `reg_` → LSTM/SVM/Linear) to prevent stale cross-domain data from rendering.
3. Builds **annotation objects** from `intervention_logs` where `status === "CONFIRMED_SUCCESS"` and timestamp falls within the visible chart window. Each annotation is a dashed green vertical line labeled with the tactic ID.
4. Updates the main chart with the primary metric and a red dashed **static threshold reference line**.
5. Updates the doughnut chart with model usage counts (filtered by `validModels`).
6. Updates the two QA charts with `associated_qas` entries (minus `model_used`).

#### `analyze_telemetry()` (server-side counterpart in `app.py`)
Static threshold check + optional dynamic logic. See [Section 5.3](#53-core-helper-functions).

#### `handleTelemetryDownload()`
Exports all `globalKnowledge.telemetry_history` as a CSV by auto-detecting all column keys across all rows, building a `Blob`, and triggering a download via a synthetic `<a>` element click.

#### `handleModelSelection()`
Sequentially executes the single-model startup flow:
1. `resetServerKnowledge()` — `POST /api/reset`
2. `writeApproachConfig(approachConfig)` — `POST /api/write-approach`
3. `setSelectedModel(selectedModel, currentSystemType)` — `POST /api/set-model`
4. `populateFormWithPreset(singlePreset)` — fills the policy form
5. `POST /api/policy` — registers the no-op preset policy
6. `savePolicyToFolder()` — `POST /api/save-policy`
7. `runManagedSystem()` — `POST /api/start-managed-system`
8. Starts polling loop

#### `updateHistoryFeed(data)` / `createHistoryCard()` / `renderHistoryChart()`
Builds the adaptation history tab. Each confirmed adaptation event becomes a card showing:
- The trigger (metric name, value, condition, threshold).
- The tactic taken.
- A mini Chart.js line chart of ±10 telemetry points around the event, with a dashed red vertical annotation marking the exact adaptation moment.

---

## 5. Component 2: Managing System

**File:** `tool/app.py`  
**Process:** `python3 app.py` (port 5000)  
**Framework:** Flask + flask-cors

### 5.1 Global In-Memory State

```python
KNOWLEDGE_BASE = {
    "policies": {},           # policy_id → full policy dict
    "telemetry_data": {},     # policy_id → list of telemetry dicts
    "intervention_logs": {}   # policy_id → list of intervention records
}
```

All state is in-memory and resets on process restart or via `POST /api/reset`. There is no database or persistent backing for the runtime telemetry.

### 5.2 Startup Behavior

On `__main__`, before serving requests, a **daemon background thread** is started:

```python
threading.Thread(target=periodic_secondary_checks, args=(30,), daemon=True).start()
```

This thread runs forever, sleeping 30 seconds between iterations, and checks secondary boundaries (e.g., KL divergence drift) independently of the primary telemetry ingestion path.

### 5.3 Core Helper Functions

#### `get_historical_average(policy_id, metric_key) → float | None`

Computes the mean of all stored values for `metric_key` under `policy_id` in `KNOWLEDGE_BASE["telemetry_data"]`. Returns `None` if no data exists for that key.

**Used by:** `analyze_telemetry()` for optional dynamic threshold logic.

#### `analyze_telemetry(policy, metric_value, metric_key) → bool`

Evaluates a single incoming metric value against the policy's `adaptation_boundary`. Two-stage process:

**Stage 1 — Static Threshold Check:**
- `condition == "GREATER_THAN"`: returns `False` immediately if `metric_value <= threshold`.
- `condition == "LESS_THAN"`: returns `False` immediately if `metric_value >= threshold`.
- Otherwise, sets `violation = True`.

**Stage 2 — Optional Dynamic Logic Check:**
Only reached if stage 1 found a static violation AND `boundary["dynamic_logic"]` is set (e.g., `"historic_avg * 1.2"`).
- Fetches the historical average for the metric.
- Parses the factor from the dynamic logic string: `float(dynamic_logic_str.split('*')[1].strip())`.
- Computes `dynamic_threshold = historical_average * factor`.
- Returns `True` only if the metric value also crosses the dynamic threshold.
- Falls back to the static result if no historical data is available.

#### `plan_and_execute(policy, trigger_value, trigger_metric=None)`

Selects the highest-priority tactic from `policy["tactics"]` (sorted ascending by `priority`) and sends it to the managed system:

1. Creates an `intervention_record` dict with fields: `timestamp`, `policy_id`, `tactic_id`, `trigger_value`, `trigger_metric`, `status: "TRIGGERED"`.
2. Appends it to `KNOWLEDGE_BASE["intervention_logs"][policy_id]`.
3. POSTs `{"tactic_id": ..., "trigger_value": ..., "trigger_timestamp": ...}` to `tactic["tactic_endpoint"]` with a 5-second timeout.
4. Updates `intervention_record["status"]` to `"CONFIRMED_SUCCESS"`, `"CONFIRMED_FAILED"`, or `"REQUEST_FAILED"` based on the response.

#### `periodic_secondary_checks(interval=30)` — Background Thread

Runs forever. Every `interval` seconds:
- Iterates over all registered policies that have `secondary_boundaries`.
- For each secondary boundary, reads the latest telemetry entry.
- If violated, synthesizes a temporary `drift_policy` dict containing only the secondary tactic and calls `plan_and_execute`.
- This is the dedicated path for **KL divergence drift detection**, which must be evaluated on a slower cadence than the primary score boundary.

### 5.4 API Endpoints (app.py)

#### `POST /api/policy`
Registers or replaces a policy.
- **Body:** Full policy JSON with at minimum a `policy_id` field.
- **Response:** `{"message": "Policy added"}` — 201.

#### `POST /api/telemetry`
Core ingestion endpoint. Called by the master wrapper's `push_telemetry` thread every 5 seconds.
- **Body:** Arbitrary JSON with at minimum a `timestamp` key.
- **Behavior:**
  1. For each registered policy, checks if `policy["quality_attribute"]` is present in the telemetry.
  2. Appends matching telemetry to `KNOWLEDGE_BASE["telemetry_data"][policy_id]`.
  3. Calls `analyze_telemetry()`. If violated → `plan_and_execute()`.
  4. Telemetry that matches no registered policy goes to the `"unassigned"` bucket.
- **Response:** `{"message": "Telemetry received"}` — 200.

#### `GET /api/knowledge/<policy_id>`
Returns a JSON snapshot for the dashboard:
```json
{
    "policy": { ... },
    "telemetry_history": [ ... ],
    "intervention_logs": [ ... ]
}
```
The special value `"unassigned"` returns unmatched telemetry with an empty logs list.

#### `POST /api/write-approach`
Changes the active approach configuration.
- **Body:** `{"approach": "<approach_key>"}`.
- **Behavior:**
  1. Calls `stop_managed_system()` internally and waits 2 seconds.
  2. Maps the dashboard key to a short token via a lookup table:

| Dashboard Key | Written to `approach.conf` |
|---|---|
| `reg_harmone_score` | `reg_harmone` |
| `reg_switch_r2` | `reg_switch` |
| `reg_single` | `reg_single` |
| `cv_harmone_score` | `cv_harmone` |
| `cv_switch_conf` | `cv_switch` |
| `cv_single` | `cv_single` |

  3. Writes the token to `approach.conf`.
  4. Clears the in-memory `KNOWLEDGE_BASE` entirely.
- **Response:** `{"message": "Approach written and system cleaned"}` — 200.

#### `POST /api/save-policy`
Persists a policy JSON to `policies/<policy_id>.json`.
- Creates the `policies/` directory if it does not exist.

#### `POST /api/set-model`
Writes a model name to `knowledge/model.csv` in the appropriate managed system directory.
- **Body:** `{"model": "<name>", "system": "regression" | "cv"}`.
- Paths: `managed_system_regression/knowledge/model.csv` or `managed_system_cv/knowledge/model.csv`.

#### `POST /api/reset`
Clears the in-memory `KNOWLEDGE_BASE` (all three sub-dicts) to empty dicts.
- **Response:** `{"message": "Knowledge base reset."}` — 200.

#### `POST /api/start-managed-system`
Launches `run_managed_system.py` as a non-blocking subprocess via `python3 run_managed_system.py`.

#### `POST /api/stop-managed-system`
Two-stage shutdown:
1. POSTs to `http://localhost:8080/adaptor/shutdown` (timeout 10 seconds).
2. Iterates all running processes via `psutil.process_iter` and terminates any Python process whose command line contains `run_managed_system.py`, `inference.py`, `manage.py`, `managed_system_cv`, or `managed_system_regression` — except the server's own PID.

#### `POST /api/upload-custom-mape`
Multipart form endpoint for building a custom managed system. Full sequence:
1. Reads `base_system` field (`"regression"` or `"cv"`).
2. Determines `source_dir` and `approach_conf_content`.
3. Deletes and recreates `managed_system_custom/` by `shutil.copytree` from the base system.
4. Accepts `.py` files in `files[]`; saves allowed ones (`monitor.py`, `analyse.py`, `plan.py`, `execute.py`, `manage.py`) to `managed_system_custom/mape_logic/`.
5. If a `dataset` file is uploaded:
   - Regression: saves as `managed_system_custom/knowledge/dataset.csv`.
   - CV: saves as a `.zip`, extracts to `managed_system_custom/data/bdd100k/images/test/`, deletes the zip.
6. Writes `approach_conf_content` to `approach.conf`.
7. Resets `KNOWLEDGE_BASE`.
- **Response:** `{"message": "Custom system built with N MAPE files.", "approach": "..."}` — 200.

---

## 6. Component 3: Master Wrapper

**File:** `tool/run_managed_system.py`  
**Process:** `python3 run_managed_system.py` (started by `app.py` on `/api/start-managed-system`)  
**Ports used:** 8080 (Adaptation Handler)

The master wrapper **orchestrates the entire managed system lifecycle**: it reads `approach.conf`, resolves the correct logic path, dynamically imports monitor functions, spawns inference and management subprocesses, registers policies with the ACP, and runs the Adaptation Handler Flask server.

### 6.1 Module-Level Globals

```python
ACP_SERVER_URL = "http://localhost:5000"
APPROACH_CONFIG_FILE = "approach.conf"
POLICY_DIR = "policies"
HANDLER_PORT = 8080

LOGIC_PATH = ""          # resolved at startup
COMMAND_FILE_PATH = ""   # resolved at startup
monitor_mape = None      # imported dynamically
monitor_drift = None     # imported dynamically
subprocesses = []        # Popen handles
should_shutdown = False  # shutdown flag
```

### 6.2 Startup Sequence (Main Block)

```
1. Read approach.conf
2. Resolve system_type + run_mode + LOGIC_PATH
   - "custom_*"  → LOGIC_PATH = "managed_system_custom"
   - "cv_*"      → LOGIC_PATH = "managed_system_cv"
   - "reg_*"     → LOGIC_PATH = "managed_system_regression"
3. import_monitor_from_path(LOGIC_PATH)
4. Set KNOWLEDGE_PATH, COMMAND_FILE_PATH, write local approach.conf
   (writes "{run_mode}_acp", e.g., "harmone_acp" or "switch_acp")
5. Start push_telemetry() daemon thread
6. Popen inference.py (always) + mape_logic/manage.py (if not "single" mode)
7. register_policies_with_acp(policy_prefix)
8. Start Adaptation Handler Flask server (if not "single" mode) in daemon thread
9. Block on subprocess.wait() — KeyboardInterrupt triggers cleanup_processes()
```

### 6.3 `import_monitor_from_path(logic_path)`

Uses `importlib.util` to dynamically load `<logic_path>/mape_logic/monitor.py` at runtime. This design allows the same wrapper to serve any managed system that exposes `monitor_mape()` and optionally `monitor_drift()`.

Temporarily inserts `logic_path` into `sys.path` so that the monitor module's own local imports resolve correctly, then removes it.

```python
spec = importlib.util.spec_from_file_location("monitor", monitor_path)
monitor_module = importlib.util.module_from_spec(spec)
sys.path.insert(0, logic_path)
spec.loader.exec_module(monitor_module)
sys.path.pop(0)
monitor_mape = getattr(monitor_module, "monitor_mape", None)
monitor_drift = getattr(monitor_module, "monitor_drift", None)
```

### 6.4 `push_telemetry()` — Background Thread

Runs every 5 seconds until `should_shutdown` is set. On each iteration:
1. Calls `monitor_mape()` → merges resulting dict into payload.
2. If `monitor_drift` exists, calls it → merges `kl_div` into payload.
3. If payload has more than just `timestamp`, POSTs to `http://localhost:5000/api/telemetry`.

### 6.5 `register_policies_with_acp(policy_prefix)`

Scans the `policies/` directory for JSON files whose filename starts with `policy_prefix`. Loads each and POSTs to `/api/policy`. Returns `False` (fatal) if no matching files exist (single mode skips this entirely).

**Example:** `approach.conf = "cv_harmone"` → `policy_prefix = "cv_harmone"` → loads `cv_harmone_score.json`.

### 6.6 Adaptation Handler Flask App (port 8080)

#### `POST /adaptor/tactic`
Receives `{"tactic_id": "..."}` from `app.py`'s `plan_and_execute()`. Writes `tactic_id` string to `COMMAND_FILE_PATH` (`<LOGIC_PATH>/knowledge/command.txt`). Returns 503 if `should_shutdown` is set.

#### `POST /adaptor/shutdown`
Sets `should_shutdown = True`, starts `cleanup_processes()` in a background thread, then calls `os._exit(0)` after a 2-second delay.

#### `GET /adaptor/health`
Returns `{"status": "running", "processes": N}` — 200, or `{"status": "shutting_down"}` — 503.

### 6.7 `cleanup_processes()`

Two-pass process termination:
1. **Direct subprocesses:** `p.terminate()` with a 5-second graceful timeout, then `p.kill()` if unresponsive.
2. **Orphaned processes:** Scans all system processes via `psutil.process_iter` for Python processes whose command line contains `inference.py`, `manage.py`, `managed_system_cv`, or `managed_system_regression`, then kills them.

---

## 7. Component 4: Managed System — Regression

**Directory:** `tool/managed_system_regression/`

The regression managed system performs time-series traffic flow prediction (PEMS dataset) using three interchangeable models:
- **LSTM** (`models/lstm.pth`) — PyTorch LSTM with 1 input, 50 hidden units, 1 output.
- **Linear** (`models/linear.pkl`) — scikit-learn Ridge regression (α=200).
- **SVM** (`models/svm.pkl`) — scikit-learn SVR (kernel=linear, C=0.08, tol=0.16).

### 7.1 `inference.py`

Runs as a subprocess in `cwd=managed_system_regression`.

**Data loading:** Reads `knowledge/dataset.csv`, extracts the `flow` column, normalizes with `MinMaxScaler`, and creates sliding-window sequences of length 5.

**Inference loop (for each sample `i`):**
1. Reads `knowledge/model.csv` to determine the active model (checked on every iteration — enables hot-swapping).
2. Starts a `pyRAPL.Measurement("inference")` block.
3. Loads the model file and runs inference.
4. Stops the energy measurement.
5. Inverse-transforms predictions back to original scale.
6. Appends `[true_value, predicted_value, chosen_model, inference_time, energy_uJ]` to `knowledge/predictions.csv`.
7. Sleeps 0.15 seconds to simulate real-time streaming.

**Model loading (per iteration):**
- `lstm` → loads `models/lstm.pth` via `torch.load(..., weights_only=False)`.
- `linear` → loads `models/linear.pkl` via `pickle.load`.
- `svm` → loads `models/svm.pkl` via `pickle.load`.

**CSV columns:** `true_value, predicted_value, model_used, inference_time, energy_uJ`

### 7.2 `retrain.py`

Triggered by `execute_drift()` when no suitable versioned model is found. Reads `knowledge/drift.csv` (the last 1200 rows saved when drift was detected) and retrains the current model.

**Steps:**
1. Reads `knowledge/model.csv` to determine which model to retrain.
2. Reads `knowledge/drift.csv` for the `true_value` column.
3. Scales with `MinMaxScaler`, creates sequences of length 5.
4. Trains the model:
   - `linear` → `Ridge(alpha=200)`
   - `svm` → `SVR(kernel="linear", C=0.08, tol=0.16)`
   - `lstm` → 50-epoch Adam training with MSELoss, batch size 16
5. Calls `save_model_and_data(model, model_name, train_df)`:
   - Saves weights to `models/<name>.pth` or `.pkl`.
   - Creates `versionedMR/<model_name>/version_N/` and saves weights + `data.csv` (training data distribution) there.

**Versioning logic (`get_next_version`):** Scans `versionedMR/<model_name>/` for existing `version_N` directories and increments the highest `N`.

### 7.3 MAPE Logic — Regression

#### `mape_logic/monitor.py`

**`monitor_mape() → dict | None`**

File-increment reader: uses `mape_info.json["last_line"]` to skip already-processed rows from `predictions.csv`. On each call:
- Reads only new rows (`skiprows=range(1, last_line + 1)`).
- If no new rows, falls back to the last 50 rows for metrics.
- Computes `r2_score(true_value, predicted_value)`.
- Computes `avg_energy` (mean of `energy_uJ` column in µJ).
- Normalizes energy: `(avg_energy − E_m) / (E_M − E_m)`, clamped to [0, 1].
- Computes **HarmonE model score**: `beta * r2 + (1 - beta) * (1 - energy_normalized)`.
- Computes **EMA score**: `gamma * model_score + (1 - gamma) * prev_ema_score`.
- Updates `ema_scores[current_model]` and `last_line` in `mape_info.json`.

**Returns:**
```json
{
  "r2_score": 0.72,
  "energy": 45230.5,
  "normalized_energy": 0.34,
  "score": 0.81,
  "model_used": "lstm",
  "model_switches": 68,
  "retrains": 5,
  "vmr_events": 0,
  "mape_k_energy_uJ": 197140423.0,
  "simple_switches": 0
}
```

**`monitor_drift() → dict | None`**

Requires ≥ 2400 rows in `predictions.csv`. Uses two consecutive 1200-row windows on `true_value`:
- Reference window: rows `[-2400:-1200]`.
- Current window: rows `[-1200:]`.
- Builds 50-bin density histograms, adds epsilon `1e-10`, computes `scipy.stats.entropy(ref_hist, curr_hist)`.
- Returns `{"kl_div": <float>}`.
- If insufficient data, returns a random placeholder `kl_div` in [0.01, 0.15].

#### `mape_logic/analyse.py`

**`analyse_mape() → dict | None`**

Calls `monitor_mape()`. Loads `thresholds.json`. Implements **dynamic energy threshold** adaptation:

```python
new_energy_threshold = current_energy_threshold + 0.95 * (original_max_energy - used_energy)
```

This relaxes the energy budget when the system is operating efficiently and tightens it otherwise.

Recovery mode: if `energy` threshold is violated, sets `recovery_cycles = 3` in `mape_info.json`, blocking further switches for 3 evaluation cycles.

**Returns:** `{"switch_needed": bool, "score": float, "threshold_violated": "score" | "energy" | None}`

**`analyse_drift() → dict | None`**

Calls `monitor_drift()`. If `kl_div > 0.5`:
- Saves last 1200 rows of `predictions.csv` to `knowledge/drift.csv`.
- Reads current model from `model.csv`.
- Calls `get_best_version(model_name)` to find a previous version with lower KL divergence.

**`get_best_version(model_name) → str | None`**

Scans `versionedMR/<model_name>/version_N/data.csv`. For each version:
- Loads the `train_data` column.
- Computes KL divergence between the version's training distribution and `drift.csv`.
- Returns the path with minimum KL, only if that minimum is below 0.75.

**Returns:** `{"drift_detected": bool, "best_version": str | None}`

#### `mape_logic/plan.py`

**`plan_mape(trigger="local") → str | None`**

Epsilon-greedy model selection using `alpha` from `thresholds.json`:
- `trigger="local"`: first runs `analyse_mape()`; returns `None` if no switch is needed.
- `trigger="acp"`: skips analysis (ACP already confirmed a violation).
- With probability `alpha`: **exploration** — randomly picks from available non-current models.
- Otherwise: **exploitation** — picks the model with the highest EMA score that is not the current model.
- Returns the chosen model name string, or `None` if already on the best model.

**`plan_drift(trigger="local") → dict | None`**

- `trigger="local"`: calls `analyse_drift()` and gates on `drift_detected == True`.
- `trigger="acp"`: calls `analyse_drift()` regardless to find the solution (replace or retrain).
- Returns `{"action": "replace", "version": <path>}` or `{"action": "retrain"}`.

**`plan_simple_switch(trigger="local") → str`**

Baseline planner: removes the current model from `["lstm", "linear", "svm"]` and randomly picks from the remainder. No analysis involved.

#### `mape_logic/execute.py`

All execute functions use `pyRAPL.Measurement` to measure the energy cost of the MAPE-K cycle itself.

**`execute_mape(trigger="local")`**
- Wraps `plan_mape(trigger)` in an energy measurement.
- If decision is not `None`: writes model name to `knowledge/model.csv`.
- Calls `record_event("switch", energy_consumed, details)`.

**`execute_drift(trigger="local")`**
- Wraps `plan_drift(trigger)` in an energy measurement.
- If `action == "replace"`: uses `shutil.copy` to copy from `versionedMR/<name>/<version>/<name>.pth|pkl` to `models/<name>.pth|pkl`.
- If `action == "retrain"`: runs `retrain.py` via `os.system("python retrain.py")`.
- Calls `record_event("retrain"|"vmr", energy_consumed, details)`.

**`execute_simple_switch(trigger="local")`**
- Calls `plan_simple_switch(trigger)`.
- Writes model name to `model.csv`.
- Calls `record_simple_switch()` — increments the separate `simple_switch_counters` counter (no energy tracking).

**`record_event(event_type, energy_consumed, details)`**
Loads `mape_info.json`, increments the appropriate counter (`model_switches`, `retrains`, `vmr_events`), accumulates `mape_k_energy_uJ`, and saves back.

#### `mape_logic/manage.py`

Entry point for the local MAPE controller subprocess.

**Reads** its local `approach.conf` (e.g., `"harmone_acp"`) to decide the execution mode.

**`run_mape_loop(approach)`:**
- `"harmone_local"`: Runs `execute_mape(trigger="local")` every 40 seconds on an internal timer (legacy standalone mode).
- `"harmone_acp"` or `"switch_acp"`: Polls `knowledge/command.txt` every 5 seconds. On finding the file, reads and deletes it, then calls `execute_tactic_locally(tactic_id)`.

**`execute_tactic_locally(tactic_id)`:**
- `"execute_mape_plan"` → `execute_mape(trigger="acp")`.
- `"handle_data_drift"` → `execute_drift(trigger="acp")`.
- `"switch_model_r2_baseline"` → `execute_mape(trigger="acp")`.

---

## 8. Component 5: Managed System — Computer Vision

**Directory:** `tool/managed_system_cv/`

The CV managed system performs object detection on BDD100K driving images using three YOLOv8 variants:
- **YOLOv8n** (`models/yolo_n.pt`) — nano, fastest / lowest energy.
- **YOLOv8s** (`models/yolo_s.pt`) — small, balanced.
- **YOLOv8m** (`models/yolo_m.pt`) — medium, highest accuracy / energy.

The CV system uses **luminance histograms** for drift detection rather than raw value distributions, because image data distributions are more naturally captured by visual statistics.

### 8.1 `utility/drift_utils.py`

Three functions shared across the CV MAPE logic and retrain components:

**`luminance_histogram(img, bins=64) → np.ndarray | None`**
Converts an image (file path or PIL Image) to a normalized luminance histogram using the Rec.601 weighting:
```python
y = 0.299 * R + 0.587 * G + 0.114 * B
```
Bins luminance values 0–255 into `bins` buckets, normalizes to sum=1.

**`kl_divergence(p, q) → float`**
Computes KL(p ∥ q) with epsilon 1e-10 added to both distributions to avoid log(0).

**`window_hist_stats(image_paths, bins=64) → (hist, mean, std)`**
Aggregates luminance histograms over a list of image paths and returns the summed histogram plus mean and std dev reconstructed from histogram moments.

### 8.2 `inference.py`

**Initialization:** Before the main inference loop, for each model in `MODEL_PATHS`:
- Saves a version 1 copy of the model weights to `versionedMR/<model_name>_v1.pt` if not already present.
- Computes and saves the **average luminance histogram** of the first 1000 reference images to `versionedMR/<model_name>_v1_hist.json`.

**Inference loop (for each `image_path`):**
1. Reads `knowledge/model.csv` for the active model.
2. Loads the YOLO model with `YOLO(model_path)`.
3. Starts `pyRAPL.Measurement("inference")`.
4. Runs `model(image_path, verbose=False)`.
5. Computes `top_conf = boxes.conf.mean()` (mean confidence across all detected boxes; 0.0 if no detections).
6. Computes **luminance histogram** of the current image.
7. Appends `[image_name, confidence, model_used, inference_time, energy_uJ, histogram_str]` to `knowledge/predictions.csv`.
8. Saves per-image detection results to `knowledge/inferences/<stem>.txt` in YOLO format (normalized bounding boxes + confidence).

**CSV columns:** `image_name, confidence, model_used, inference_time, energy_uJ, histogram`  
The `histogram` column stores the 64-bin luminance histogram as a space-separated float string.

### 8.3 `retrain.py`

Significantly more sophisticated than the regression retrain. Uses **drift-type classification** and **augmentation-based fine-tuning**.

**`deduce_drift_type(ref_hist, current_hist) → "dark" | "fog" | "clear"`**  
Classifies the type of visual distribution shift:
- Computes mean and std dev from both histograms using histogram moments.
- `is_dark = cur_mean / ref_mean < 0.80` (significant luminance drop).
- `is_foggy = cur_std / ref_std < 0.85` (significant contrast reduction).
- Both → "dark"; only foggy → "fog"; only dark → "dark"; neither → "clear".

**`create_augmented_retrain_set(image_paths, label_dir, drift_type)`**  
Creates a synthetic training set matching the drift type:
- `"dark"`: applies `torchvision.transforms.functional.adjust_brightness(img, factor=0.35)`.
- `"fog"`: reduces contrast by `(img - mean) * 0.25 + mean`.
- `"clear"`: no augmentation.
Saves augmented images to `data/bdd100k/images/retrain_augmented/` with corresponding label files.

**Main `retrain_yolo()` function:**
1. Reads `knowledge/model.csv` for the base model name.
2. Loads the model from `base_models/<name>.pt` (original pre-trained weights).
3. **Freezes all layers** except the detection head (final module) to enable efficient fine-tuning.
4. Creates augmented retrain set using deduced drift type.
5. Generates a temporary YAML config for YOLO training.
6. Trains for 5 epochs using `pyRAPL.Measurement("model_training")` to track energy.
7. Saves the new model version to `versionedMR/<model_name>_v{N}.pt` and its histogram to `versionedMR/<model_name>_v{N}_hist.json`.
8. Copies new weights to `models/<name>.pt` as the active model.
9. Cleans up temporary YAML and augmented data directories.

**Batch sizes per model:** yolo_n → 4, yolo_m → 1, yolo_s → 2.

### 8.4 MAPE Logic — CV

The CV MAPE logic mirrors the regression MAPE logic but uses CV-specific metrics and multi-model drift analysis.

#### `mape_logic/monitor.py`

**`monitor_mape() → dict | None`**

Uses the same file-increment pattern as the regression monitor. Reads `predictions.csv` for new rows (skipping `last_line`), computes:
- `avg_conf = df["confidence"].mean()`
- `avg_energy = df["energy_uJ"].mean()`
- `energy_norm = (avg_energy - E_m) / (E_M - E_m)` clamped to [0, 1]
- `score = beta * avg_conf + (1 - beta) * (1 - energy_norm)` (note: `beta=0.95` for CV, weighting confidence heavily)
- `final_score = gamma * score + (1 - gamma) * prev_ema_score` (EMA)

**`monitor_drift() → dict | None`**

Requires ≥ 2000 rows. Uses two 1000-image windows and their **luminance histograms** (stored in the `histogram` column of `predictions.csv`):
- Parses histogram strings with `np.fromstring(h, sep=' ')`.
- Computes mean histogram for each window.
- Returns `kl_divergence(cur_dist, ref_dist)`.

#### `mape_logic/analyse.py`

**`analyse_mape()`**: Same structure as regression but with CV-specific `beta` (typically 0.95 — confidence-focused) and energy threshold update formula: `new_threshold = current + 0.4 * (original - used)` (uses factor 0.4 vs. 0.95 in regression).

**`analyse_drift()`**: More sophisticated than regression — searches across **all three model types** (not just the current model) for the best matching version:

```python
for model_name in ["yolo_n", "yolo_s", "yolo_m"]:
    best_path, min_kl = get_best_version_for_model(model_name, current_drift_dist)
```

**`get_best_version_for_model(model_name, current_drift_dist)`**: Scans `versionedMR/` for files matching `{model_name}_v*.pt`, loads the corresponding `_hist.json` files, and computes KL divergence between each stored histogram and the current drift distribution. Returns the best path and its KL score.

Returns one of:
- `{"drift_detected": True, "best_version": <path>, "action": "switch_version"}` if min KL < 0.07
- `{"drift_detected": True, "best_version": None, "action": "retrain"}`
- `{"drift_detected": False}`

#### `mape_logic/plan.py`

**`plan_mape(trigger="local")`**: Epsilon-greedy with CV-specific logic:
- For `"energy"` violations: picks the best alternative model by EMA score.
- For `"score"` violations or ACP trigger: picks the model with the **absolute highest** EMA score.

**`plan_drift(trigger="local")`**: Calls `analyse_drift()`. Returns `{"action": "switch_version", "version_path": <path>}` or `{"action": "retrain"}`.

#### `mape_logic/execute.py`

**`execute_mape(trigger="local")`**: Same pattern as regression. Writes model name to `model.csv` and logs via both `log_event()` (CSV event log) and `record_event()` (JSON counters).

**`execute_drift(trigger="local")`**: For `"switch_version"`:
1. Validates the version path exists.
2. Extracts base model name (`yolo_n`, `yolo_s`, or `yolo_m`) from the filename using regex.
3. Copies versioned `.pt` to `models/<base_name>.pt`.
4. Writes base name to `model.csv`.
5. **Inflates EMA score** by +0.1 (capped at 1.0) to give the newly-deployed version a head start before it accumulates real-world performance data.
6. Sleeps 20 seconds (reduced simulation pause).

**`execute_simple_switch(trigger="local")`**: Calls `plan_simple_switch()` and logs with both `log_event` and `record_simple_switch`.

#### `mape_logic/manage.py`

**More structured than the regression manager.** Uses threading:

```python
if "acp" in approach:
    t_listener = threading.Thread(target=acp_command_listener, daemon=True)
elif approach in ["harmone", "switch"]:
    t_mape = threading.Thread(target=run_execute_mape_local, daemon=True)
    # + t_drift for "harmone" mode
elif "single" in approach:
    # No threads — monitor-only
```

**`acp_command_listener()`**: Polls `knowledge/command.txt` every **1 second** (faster than regression's 5 seconds). On finding the file, reads and removes it, then calls `execute_tactic_locally(tactic_id)`.

**`execute_tactic_locally(tactic_id)`**: Same dispatch as regression:
- `"execute_mape_plan"` → `execute_mape(trigger="acp")`
- `"handle_data_drift"` → (currently commented out / pass; placeholder for future drift handling)
- `"switch_model_r2_baseline"` → `execute_mape(trigger="acp")`

**Also measures tactic execution energy** using `pyRAPL.Measurement(tactic_id)` and logs to `knowledge/mape_log.csv`.

---

## 9. MAPE-K Loop Deep Dive

### 9.1 Composite HarmonE Score Formula

The HarmonE score is a composite metric that balances model accuracy against energy efficiency:

```
normalized_energy = clip((avg_energy - E_m) / (E_M - E_m), 0, 1)
model_score = beta * accuracy + (1 - beta) * (1 - normalized_energy)
ema_score = gamma * model_score + (1 - gamma) * prev_ema_score
```

**Parameters (from `knowledge/thresholds.json`):**

| Parameter | Regression Typical | CV Typical | Meaning |
|---|---|---|---|
| `E_m` | energy min baseline | energy min baseline | Lower bound for normalization (µJ) |
| `E_M` | energy max baseline | energy max baseline | Upper bound for normalization (µJ) |
| `beta` | 0.5 | 0.95 | Accuracy vs. energy weighting (higher = more accuracy-focused) |
| `gamma` | 0.8 | 0.8 | EMA smoothing factor (higher = more responsive to recent data) |
| `alpha` | 0.1 | 0.1 | Exploration probability in epsilon-greedy model selection |
| `min_score` | 0.78 | 0.54 | Score threshold below which adaptation is triggered |
| `max_energy` | 0.6 (normalized) | 0.6 (normalized) | Energy threshold above which adaptation is triggered |

### 9.2 Dynamic Energy Threshold

The energy threshold adapts over time to prevent thrashing:

**Regression formula:**
```python
new_threshold = current_threshold + 0.95 * (original_max_energy - used_energy)
```

**CV formula:**
```python
new_threshold = current_threshold + 0.4 * (original_max_energy - used_energy)
```

When the system consumes less energy than the original maximum, the threshold rises (becoming more permissive). When the system consumes more energy, the threshold approaches the original maximum (becoming stricter). The coefficient (0.95 vs. 0.4) controls the speed of this adaptation.

### 9.3 Recovery Mode

When an energy threshold violation occurs, the system enters a **recovery mode** that prevents further model switches for `N` evaluation cycles:

```python
recovery_cycles = 3   # set on energy violation
# Each subsequent call to analyse_mape() decrements by 1 until 0
```

This prevents rapid switching caused by a temporarily high energy reading.

### 9.4 Epsilon-Greedy Model Selection

When a model switch is warranted:

```python
if random.random() < alpha:
    # Exploration: randomly pick any other model
    chosen = random.choice([m for m in ema_scores.keys() if m != current_model])
else:
    # Exploitation: pick the model with the highest EMA score (not current)
    best_alternative = sorted(ema_scores.items(), key=lambda x: x[1], reverse=True)
    chosen = next((m for m, score in best_alternative if m != current_model), None)
```

Each model maintains an independent EMA score, updated whenever that model is the active one during inference.

### 9.5 Data Drift Detection (Regression)

Uses KL divergence on the distribution of `true_value` across two consecutive 1200-sample windows:

```python
ref_window = df['true_value'].iloc[-2400:-1200]
cur_window = df['true_value'].iloc[-1200:]
kl_div = scipy.stats.entropy(ref_hist + 1e-10, curr_hist + 1e-10)
```

**Primary drift threshold (local):** 0.5  
**ACP secondary boundary threshold:** 0.10 (per `reg_harmone_score.json`)  
**Secondary check cadence:** every 30 seconds (via `periodic_secondary_checks` in `app.py`)

### 9.6 Data Drift Detection (CV)

Uses KL divergence on **average luminance histograms** of 1000-image windows:

```python
ref_hists_str = df["histogram"].iloc[-2000:-1000]
cur_hists_str = df["histogram"].iloc[-1000:]
ref_dist = np.mean([np.fromstring(h, sep=' ') for h in ref_hists_str], axis=0)
cur_dist = np.mean([np.fromstring(h, sep=' ') for h in cur_hists_str], axis=0)
kl = kl_divergence(cur_dist, ref_dist)
```

**Local drift threshold:** 0.07 (in `analyse.py`, `DRIFT_THRESHOLD = 0.07`)  
**ACP secondary boundary threshold:** 0.07 (per `cv_harmone_score.json`)

### 9.7 Versioned Model Repository (VMR)

When a retrain completes, model weights and the training data distribution are archived:

**Regression VMR layout:**
```
versionedMR/
└── lstm/
    ├── version_1/
    │   ├── lstm.pth        # trained weights
    │   └── data.csv        # column: train_data (original scale values)
    └── version_2/
        ├── lstm.pth
        └── data.csv
```

**CV VMR layout:**
```
versionedMR/
├── yolo_s_v1.pt            # initial version of yolo_s weights
├── yolo_s_v1_hist.json     # {"average_histogram": [...64 floats...]}
├── yolo_s_v2.pt            # retrained version
└── yolo_s_v2_hist.json
```

When drift is detected, the VMR is searched for the version whose training distribution most closely matches the current drift distribution (minimum KL divergence). If a suitable version is found (KL below threshold), it is copied to `models/` without retraining. This **Versioned Model Replacement (VMR)** path is cheaper than retraining.

### 9.8 Two-Tier Boundary Evaluation

The Managing System (`app.py`) evaluates boundaries at two timescales:

**Tier 1 — Per-Telemetry (every 5 seconds):**
Called in `receive_telemetry()`. Checks only the **primary** `adaptation_boundary` of each policy against the incoming metric value.

**Tier 2 — Periodic Secondary (every 30 seconds):**
Called in the background thread `periodic_secondary_checks()`. Checks **secondary boundaries** (e.g., `kl_div`) in the latest stored telemetry for each policy.

This two-tier design prevents drift handling from overwhelming the primary control loop.

---

## 10. Adaptation Policies

Policies are JSON files stored in `tool/policies/`. They define what to monitor, when to act, and how to act.

### 10.1 Policy Schema

```json
{
  "policy_id": "<string>",
  "quality_attribute": "<metric_key>",
  "adaptation_boundary": {
    "type": "STATIC_THRESHOLD",
    "condition": "LESS_THAN | GREATER_THAN",
    "threshold": <number>,
    "dynamic_logic": "historic_avg * <factor>"   // optional
  },
  "tactics": [
    {
      "tactic_id": "<string>",
      "priority": <integer>,
      "tactic_type": "CORE",
      "tactic_endpoint": "http://localhost:8080/adaptor/tactic"
    }
  ],
  "secondary_boundaries": [
    {
      "quality_attribute": "<metric_key>",
      "condition": "LESS_THAN | GREATER_THAN",
      "threshold": <number>,
      "tactic_id": "<string>"
    }
  ],
  "associated_qas": ["<metric_key>", ...]
}
```

### 10.2 Policy Files Reference

| File | `policy_id` | Monitors | Threshold | Tactic |
|---|---|---|---|---|
| `reg_harmone_score.json` | `reg_harmone_score` | `score` | < 0.78 | `execute_mape_plan` |
| `reg_switch_r2.json` | `reg_switch_r2` | `r2_score` | < 0.70 | `switch_model_r2_baseline` |
| `reg_single_lstm.json` | `reg_single_lstm` | `score` | < 0.0 | (none — monitor only) |
| `reg_single_svm.json` | `reg_single_svm` | `score` | < 0.0 | (none) |
| `reg_single_linear.json` | `reg_single_linear` | `score` | < 0.0 | (none) |
| `cv_harmone_score.json` | `cv_harmone_score` | `score` | < 0.54 | `execute_mape_plan` |
| `cv_switch_confidence.json` | `cv_switch_confidence` | `confidence` | < 0.70 | `switch_model_r2_baseline` |
| `cv_single_yolo_s.json` | `cv_single_yolo_s` | `score` | < 0.0 | (none) |
| `cv_single_yolo_m.json` | `cv_single_yolo_m` | `score` | < 0.0 | (none) |

**`reg_harmone_score.json` secondary boundary:** `kl_div > 0.10` → `handle_data_drift`  
**`cv_harmone_score.json` secondary boundary:** `kl_div > 0.07` → `handle_data_drift`

### 10.3 Tactic ID Reference

| `tactic_id` | What happens |
|---|---|
| `execute_mape_plan` | `manage.py` calls `execute_mape(trigger="acp")` — epsilon-greedy model switch |
| `handle_data_drift` | `manage.py` calls `execute_drift(trigger="acp")` — VMR lookup or retrain |
| `switch_model_r2_baseline` | `manage.py` calls `execute_mape(trigger="acp")` — same as MAPE plan (simple switch `execute_simple_switch` is currently commented out) |

---

## 11. Knowledge Base and Runtime State Files

### 11.1 `knowledge/model.csv`

Single-line plaintext file containing the name of the currently active model. Read on every inference iteration, enabling hot-swapping without process restart.

**Values (Regression):** `lstm`, `linear`, `svm`  
**Values (CV):** `yolo_n`, `yolo_s`, `yolo_m`

### 11.2 `knowledge/predictions.csv`

Append-only log of every inference.

**Regression columns:**
```
true_value, predicted_value, model_used, inference_time, energy_uJ
```

**CV columns:**
```
image_name, confidence, model_used, inference_time, energy_uJ, histogram
```
where `histogram` is a space-separated string of 64 floats.

### 11.3 `knowledge/mape_info.json`

Persistent runtime state for the MAPE loop. Updated by monitor and execute functions.

```json
{
  "last_line": 2456,
  "current_energy_threshold": 1.0,
  "ema_scores": {
    "lstm": 0.352,
    "linear": -2.492,
    "svm": -0.115
  },
  "recovery_cycles": 0,
  "event_counters": {
    "model_switches": 68,
    "retrains": 5,
    "vmr_events": 0,
    "mape_k_energy_uJ": 197140423.0
  },
  "simple_switch_counters": {
    "simple_switches": 0
  }
}
```

| Field | Purpose |
|---|---|
| `last_line` | Number of rows already processed; next read skips these |
| `current_energy_threshold` | Dynamically adjusted energy boundary |
| `ema_scores` | Per-model EMA sustainability scores |
| `recovery_cycles` | Remaining cycles before next switch is allowed after energy violation |
| `event_counters` | Cumulative counts for model switches, retrains, VMR events; total MAPE-K energy |
| `simple_switch_counters` | Separate counter for baseline simple switches |

### 11.4 `knowledge/thresholds.json`

Design-time parameters loaded by the MAPE logic.

```json
{
  "min_score": 0.78,
  "max_energy": 0.6,
  "E_m": <min_energy_uJ>,
  "E_M": <max_energy_uJ>,
  "beta": 0.5,
  "gamma": 0.8,
  "alpha": 0.1
}
```

### 11.5 `knowledge/command.txt`

Transient file. Written by the Adaptation Handler (`/adaptor/tactic`) and read+deleted by `manage.py`. Contains the `tactic_id` string. Lifetime: milliseconds.

### 11.6 `knowledge/drift.csv`

Written by `analyse_drift()` when a drift event is detected. Contains the last 1200 rows of `predictions.csv` at the time drift was detected, used by `get_best_version()` for VMR search.

### 11.7 `knowledge/drift_kl.json`

Written by `get_best_version()` (both regression and CV) for debugging. Contains KL divergence scores computed during VMR search.

```json
{
  "best_version": "<path_or_null>",
  "min_kl_div": 0.042,
  "kl_per_model": { "yolo_n": 0.12, "yolo_s": 0.042, "yolo_m": 0.09 }
}
```

### 11.8 `knowledge/event_log.csv` (CV only)

CSV log written by `execute.py` in the CV system for each adaptation event.

**Columns:** `event_type, last_line, model, version, details`

### 11.9 `knowledge/mape_log.csv` (CV only)

Written by `manage.py` in the CV system. Tracks energy consumed by each tactic execution.

**Columns:** `function, energy_uJ`

---

## 12. REST API Reference

### 12.1 Managing System (port 5000)

| Method | Path | Request Body | Response | Description |
|---|---|---|---|---|
| `GET` | `/` | — | `"Welcome to ACP Server!"` 200 | Health check |
| `GET` | `/favicon.ico` | — | 204 | Empty |
| `POST` | `/api/policy` | Policy JSON | `{"message": "Policy added"}` 201 | Register/update a policy |
| `POST` | `/api/telemetry` | Telemetry dict | `{"message": "Telemetry received"}` 200 | Ingest runtime metrics |
| `GET` | `/api/knowledge/<id>` | — | Knowledge snapshot JSON | Retrieve telemetry + logs |
| `POST` | `/api/write-approach` | `{"approach": "..."}` | `{"message": "..."}` 200 | Switch approach config |
| `POST` | `/api/save-policy` | Policy JSON | `{"message": "Policy saved"}` 200 | Persist policy to disk |
| `POST` | `/api/set-model` | `{"model": "...", "system": "..."}` | `{"message": "Model set"}` 200 | Write model.csv directly |
| `POST` | `/api/reset` | — | `{"message": "Knowledge base reset."}` 200 | Clear in-memory KB |
| `POST` | `/api/start-managed-system` | — | `{"status": "ok"}` | Launch run_managed_system.py |
| `POST` | `/api/stop-managed-system` | — | `{"status": "ok"}` | Terminate all managed procs |
| `POST` | `/api/upload-custom-mape` | Multipart form | `{"message": "..."}` 200 | Upload custom MAPE files |

### 12.2 Adaptation Handler / Master Wrapper (port 8080)

| Method | Path | Request Body | Response | Description |
|---|---|---|---|---|
| `POST` | `/adaptor/tactic` | `{"tactic_id": "..."}` | `{"message": "Command queued."}` 200 | Relay tactic to command.txt |
| `POST` | `/adaptor/shutdown` | — | `{"message": "..."}` 200 | Shutdown all processes |
| `GET` | `/adaptor/health` | — | `{"status": "running", "processes": N}` | Health check |

---

## 13. Configuration Reference

### 13.1 `tool/approach.conf`

Single-line plaintext. Controls which managed system and mode are activated when `run_managed_system.py` starts.

| Value in file | System | Mode |
|---|---|---|
| `reg_harmone` | Regression | HarmonE Score + Drift (ACP-driven) |
| `reg_switch` | Regression | Simple R² Switch (ACP-driven) |
| `reg_single` | Regression | Monitor-only, no adaptation |
| `cv_harmone` | CV (YOLO) | HarmonE Score + Drift (ACP-driven) |
| `cv_switch` | CV (YOLO) | Simple Confidence Switch (ACP-driven) |
| `cv_single` | CV (YOLO) | Monitor-only, no adaptation |
| `custom_regression` | Custom | Regression-based custom system |
| `custom_cv` | Custom | CV-based custom system |

### 13.2 Local `approach.conf` (inside `managed_system_*/`)

Written by `run_managed_system.py` at startup. Always ends in `_acp` for ACP-driven modes.

| Master config value | Local config written | Effect in manage.py |
|---|---|---|
| `reg_harmone` | `harmone_acp` | ACP listener mode |
| `reg_switch` | `switch_acp` | ACP listener mode |
| `reg_single` | `single_acp` | No management thread started |
| `cv_harmone` | `harmone_acp` | ACP listener mode |
| `cv_switch` | `switch_acp` | ACP listener mode |
| `cv_single` | `single_acp` | No management thread started |

---

## 14. Data Flow Walkthrough

### Full Adaptation Cycle (HarmonE Mode)

```
T=0s    Inference loop writes row to predictions.csv
        [true_value, predicted_value, lstm, 0.003, 45230]

T=5s    push_telemetry() thread fires:
          monitor_mape() reads new rows, computes:
            r2=0.71, energy=45230µJ, normalized_energy=0.34
            model_score = 0.5*0.71 + 0.5*(1-0.34) = 0.685
            ema_score = 0.8*0.685 + 0.2*0.80 = 0.708
          Returns: {r2_score:0.71, score:0.708, model_used:"lstm", ...}
          monitor_drift() returns {kl_div:0.05}
          Merged payload: {timestamp:.., r2_score:0.71, score:0.708, kl_div:0.05, ...}
          POST http://localhost:5000/api/telemetry

T=5s    app.py receive_telemetry():
          For policy "reg_harmone_score" (quality_attribute="score"):
            Appends telemetry to KNOWLEDGE_BASE["telemetry_data"]["reg_harmone_score"]
            analyze_telemetry(): 0.708 < 0.78 → VIOLATION
            plan_and_execute():
              Creates intervention_record (status="TRIGGERED")
              POST http://localhost:8080/adaptor/tactic
                body: {"tactic_id": "execute_mape_plan", ...}

T=5s    run_managed_system.py /adaptor/tactic:
          Writes "execute_mape_plan" to knowledge/command.txt

T=5s    mape_logic/manage.py ACP listener (polls every 1-5s):
          Reads command.txt → "execute_mape_plan"
          Deletes command.txt
          Calls execute_tactic_locally("execute_mape_plan")
            → execute_mape(trigger="acp")
              energy_meter.begin()
              plan_mape(trigger="acp"):
                Loads ema_scores: {lstm:0.708, svm:0.45, linear:0.32}
                alpha=0.1 → no random exploration
                Best alternative to lstm → svm (0.45)
                Returns "svm"
              Writes "svm" to knowledge/model.csv
              energy_meter.end()
              record_event("switch", energy_consumed, ...)
              Updates mape_info.json: model_switches += 1

T=5.15s inference.py next iteration:
          Reads knowledge/model.csv → "svm"
          Now uses models/svm.pkl for inference

T=5s    app.py intervention_record["status"] → "CONFIRMED_SUCCESS"

T=6s    dashboard.html fetchKnowledge():
          GET /api/knowledge/reg_harmone_score
          Receives updated telemetry + intervention_log entry
          updateDashboardCharts():
            Main chart shows score dropping then recovering
            Doughnut chart updates: LSTM → SVM
            Green vertical annotation marks the switch event
```

### Secondary Drift Check Cycle (every 30 seconds)

```
T=30s   periodic_secondary_checks() wakes:
          For "reg_harmone_score" policy:
            secondary_boundaries: [{kl_div GREATER_THAN 0.10, tactic_id: handle_data_drift}]
            Latest telemetry: kl_div=0.05
            0.05 ≤ 0.10 → no violation, skip

T=2400+ inference rows accumulated:
          kl_div rises to 0.13

T=30s   periodic_secondary_checks():
          kl_div=0.13 > 0.10 → SECONDARY VIOLATION
          Creates drift_policy with tactic "handle_data_drift"
          plan_and_execute(drift_policy, 0.13, "kl_div")
            POST /adaptor/tactic → {"tactic_id": "handle_data_drift"}

          manage.py receives "handle_data_drift":
            execute_drift(trigger="acp"):
              plan_drift(trigger="acp"):
                analyse_drift():
                  monitor_drift() → kl_div=0.13 > 0.5? NO (not local threshold)
                  Actually: analyse_drift is called regardless in ACP mode
                  drift.csv saved; get_best_version("svm") checks versionedMR
                  No suitable version found (min_kl > 0.75)
                  Returns {"drift_detected": True, "best_version": None}
                Returns {"action": "retrain"}
              os.system("python retrain.py")
              retrain.py runs: fine-tunes svm on drift.csv, saves to versionedMR/
              record_event("retrain", energy, ...)
```

---

## 15. Customization and Extension

### 15.1 Building a Custom Managed System

The dashboard's **Build Custom System** flow allows uploading custom MAPE logic and a dataset. The custom system runs on top of the existing inference engines (regression or CV).

**Required files:**
- `monitor.py` — must export `monitor_mape() -> dict`
- `analyse.py` — optional (called by plan.py)
- `plan.py` — must export `plan_mape(trigger: str) -> str`
- `execute.py` — must write model name to `knowledge/model.csv`
- `manage.py` — entry point subprocess that orchestrates the above

**Dataset formats:**
- **Regression:** CSV file with a column named `flow` containing numerical values.
- **CV:** ZIP file containing `.jpg` or `.png` images (extracted to `data/bdd100k/images/test/`).

**Valid model names:**
- Regression: `"lstm"`, `"svm"`, `"linear"`
- CV: `"yolo_n"`, `"yolo_s"`, `"yolo_m"`

### 15.2 Custom Policy JSON

Users can define arbitrary monitoring goals:

```json
{
  "policy_id": "my_custom_policy",
  "quality_attribute": "my_metric",
  "adaptation_boundary": {
    "type": "STATIC_THRESHOLD",
    "condition": "LESS_THAN",
    "threshold": 0.8
  },
  "tactics": [
    {
      "tactic_id": "my_custom_tactic",
      "priority": 1,
      "tactic_type": "CORE",
      "tactic_endpoint": "http://localhost:8080/adaptor/tactic"
    }
  ],
  "secondary_boundaries": [],
  "associated_qas": ["my_metric", "energy"]
}
```

The `tactic_id` in the policy must match a case handled in `manage.py`'s `execute_tactic_locally()`.

### 15.3 Adding a New Approach Mode

To add a new mode:
1. Add an entry to the `mapping` dict in `app.py`'s `write_approach_config()`.
2. Add the dispatch logic to `run_mape_loop()` in `manage.py`.
3. Create a corresponding policy JSON in `tool/policies/`.
4. Add a button and preset to the `HARMONY_PRESETS` object in `dashboard.html`.

### 15.4 Complete Dataset Integration

**Regression (PEMS traffic flow):**
1. Download from `https://pems.dot.ca.gov/`.
2. Convert to CSV with a column `flow` (the "Flow (Veh/5 Minutes)" field).
3. Replace `managed_system_regression/knowledge/dataset.csv`.

**CV (BDD100K):**
1. Download from `https://bair.berkeley.edu/blog/2018/05/30/bdd/`.
2. Place `.jpg` images in `managed_system_cv/data/bdd100k/images/test/`.

---

## 16. Examples Directory

The `examples/` directory contains reference implementations for building custom MAPE systems.

### `examples/HarmonE/`

Complete, ready-to-upload MAPE files that replicate the built-in HarmonE regression behavior. These files are functionally identical to `managed_system_regression/mape_logic/` and serve as reference for understanding the expected function signatures and file interaction patterns.

### `examples/templates/`

Skeleton files with empty function bodies and the correct signatures. Intended as starting points for custom MAPE implementations.

**Expected function signatures:**

```python
# monitor.py
def monitor_mape() -> dict | None:
    """Must return dict with at least the quality attribute key."""

def monitor_drift() -> dict | None:
    """Must return {"kl_div": float} or None."""

# plan.py
def plan_mape(trigger: str) -> str | None:
    """Must return a valid model name string or None."""

# execute.py
def execute_mape(trigger: str):
    """Must write model name to knowledge/model.csv."""
```

---

## 17. Dependencies and Environment

### 17.1 Python Requirements (`tool/requirements.txt`)

| Category | Package | Notes |
|---|---|---|
| Web server | Flask 2.2.5+ / 3.x | 2.x for Python < 3.12; 3.x for ≥ 3.12 |
| Web server | flask-cors, Werkzeug | CORS and WSGI utilities |
| Web server | uvicorn, FastAPI | Alternative async server (optional path) |
| ML / Data | NumPy, SciPy, scikit-learn, pandas | Core numerical stack |
| ML / Data | PyTorch (CPU-only) | 2.1.0 for Python < 3.12; ≥ 2.9.0 for ≥ 3.12 |
| CV | ultralytics (YOLOv8) | Object detection model library |
| CV | opencv-python, Pillow, torchvision | Image processing |
| Energy | pyRAPL 0.2.3.1 | Intel RAPL energy measurement (Linux only) |
| Process | psutil ≥ 5.9.0 | Cross-platform process management |
| Visualization | matplotlib, seaborn | Plotting (used in scripts, not in production) |
| Numerics | scipy, sympy | Statistics (KL divergence) and symbolic math |

**Install command:**
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 17.2 System Requirements

- **OS:** Linux (Ubuntu/Debian recommended). `pyRAPL` requires the Intel RAPL interface, which is Linux-specific.
- **Python:** 3.8 or higher.
- **Privileges:** Sudo access for RAPL energy file permissions.
- **Ports:** 5000 (Managing System), 8000 (Dashboard static server), 8080 (Adaptation Handler).
- **Hardware:** Intel CPU with RAPL support (most modern Intel CPUs).

### 17.3 `harmone_env` Virtual Environment

The `tool/harmone_env/` directory contains the Python virtual environment. It is created by `harmone_start.sh` if absent. Key binary paths:
- `harmone_env/bin/python3`
- `harmone_env/bin/flask`
- `harmone_env/bin/torchrun`

---

## 18. Setup and Running

### 18.1 First-Time Setup

```bash
cd /path/to/HarmonE-tool/tool
chmod +x harmone_start.sh
./harmone_start.sh
```

`harmone_start.sh` performs the following:
1. Detects the Python executable (`python3`, `python`, `python3.12`, `python3.11` in order).
2. Creates `harmone_env` virtual environment if absent.
3. Activates `harmone_env` and runs `pip install -r requirements.txt`.
4. Sets RAPL permissions: `sudo chmod -R 777 /sys/class/powercap/intel-rapl/`.
5. Detects a supported terminal emulator (gnome-terminal, konsole, xfce4-terminal, tilix, xterm).
6. Opens three new terminal windows:
   - **Terminal 1 (ACP Server):** `python3 app.py` — starts the Managing System on port 5000.
   - **Terminal 2 (Dashboard):** `python3 -m http.server 8000` from `tool/frontend/` — serves the dashboard.
   - **Terminal 3 (Managed System Console):** Interactive shell with a hint to run `python3 run_managed_system.py`.

### 18.2 Accessing the Dashboard

Navigate to `http://localhost:8000/dashboard.html` in a browser.

### 18.3 Running an Experiment

1. **Select Approach:** Click one of the approach buttons on the welcome screen (e.g., "HarmonE (Score/Drift)" under Regression).
2. **Review Policy:** The "Policy Management" tab shows the auto-loaded policy. Thresholds can be edited and re-saved.
3. **Start Execution:** On the "Live Dashboard" tab, click the green "Start Managed System" button. This calls `POST /api/start-managed-system`.
4. **Observe:**
   - Watch the "Main Metric" chart for the quality attribute (R²/confidence/HarmonE score).
   - Watch the doughnut chart for model distribution changes.
   - Green vertical lines on the main chart mark successful adaptation events.
   - Terminal 3 shows the inference and MAPE loop logs.
5. **Switch Approaches:** Click the back-arrow to return to the welcome screen. The system stops the current managed system, clears the knowledge base, and lets you select a new approach.

---

## 19. Artifact Outputs

After running an experiment, the following files capture execution history:

### 19.1 `knowledge/predictions.csv`

**Location:** `tool/managed_system_<type>/knowledge/predictions.csv`

Raw log of every inference. Allows verifying model switches by cross-referencing the `model_used` column against timestamps.

### 19.2 Telemetry Download (Dashboard)

The "Download Telemetry (CSV)" button on the Live Dashboard exports all `telemetry_history` for the active policy as a CSV. Contains the primary metric, energy, model used, event counters, and all `associated_qas` for every telemetry push interval.

### 19.3 `knowledge/model.csv`

Current active model. A single-line file that can be read at any time to see the immediate state of the adaptation system.

### 19.4 `knowledge/mape_info.json`

Cumulative event counters (switches, retrains, VMR events) and total MAPE-K energy consumption. Useful for comparing adaptation approach efficiency.

### 19.5 `knowledge/event_log.csv` (CV only)

CSV log of every adaptation event type, the model involved, the version used (for VMR events), and descriptive details.

---

## 20. Troubleshooting

### PyRAPL Permission Errors

```bash
sudo chmod -R 777 /sys/class/powercap/intel-rapl/
```

If `pyRAPL.setup()` fails because `/sys/class/powercap/intel-rapl/` is inaccessible or does not exist (non-Intel CPU or virtualized environment), pyRAPL will raise an exception. The `energy_meter.result.pkg[0]` read will fail; the code handles this by defaulting energy to `0.0`.

### Port Already in Use

```bash
sudo lsof -ti:5000 | xargs kill -9
sudo lsof -ti:8080 | xargs kill -9
sudo lsof -ti:8000 | xargs kill -9
```

### Process Not Terminating

The two-stage shutdown in `stop_managed_system()` should handle most cases. If processes persist:
```bash
pkill -f run_managed_system.py
pkill -f inference.py
pkill -f "mape_logic/manage.py"
```

### Missing Dependencies

```bash
pip install ultralytics opencv-python matplotlib seaborn psutil pyRAPL
```

### Virtual Environment Issues

```bash
deactivate
rm -rf harmone_env
python3 -m venv harmone_env
source harmone_env/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### Dashboard Not Showing Data

- Confirm `app.py` is running on port 5000 (check `http://localhost:5000/`).
- Confirm the policy is registered — check the "Policy Management" tab for the loaded policy.
- Confirm `run_managed_system.py` is running and the inference loop has started generating rows in `predictions.csv`.
- Check the browser console for CORS errors or network failures.

### KL Divergence Always Shows Placeholder Values (Regression)

The regression `monitor_drift()` function returns a random value in [0.01, 0.15] until `predictions.csv` accumulates ≥ 2400 rows. This is expected behavior; wait for sufficient inference data.

### CV Drift Not Triggering

CV drift detection requires ≥ 2000 rows in `predictions.csv` AND a valid `histogram` column. Confirm that `inference.py` is writing histograms (check the file with a CSV reader). The `histogram` column should contain space-separated floats.

### Model Files Missing

If `models/lstm.pth`, `models/svm.pkl`, `models/linear.pkl` (regression) or `models/yolo_n.pt`, `models/yolo_s.pt`, `models/yolo_m.pt` (CV) are absent, the inference engine will fail. These must be pre-trained and placed in the respective `models/` directory before running.

---

*End of Documentation*

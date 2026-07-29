# HarmonE — Codebase File Reference

> **Live document.** Update whenever files are added, moved, or significantly changed.  
> Excludes: `harmone_env/`, `data/`, `models/`, `versionedMR/`, `runs/`, `__pycache__/`, `.git/`.

---

## Folder Tree

```
HarmonE-tool/
├── examples/
│   ├── HarmonE/                    # Reference standalone MAPE implementation
│   │   ├── monitor.py
│   │   ├── analyse.py
│   │   ├── plan.py
│   │   └── execute.py
│   └── templates/                  # Blank templates for custom MAPE components
│       ├── monitor.py
│       ├── analyse.py
│       ├── plan.py
│       └── execute.py
│
└── tool/                           # Main package — all runnable code lives here
    ├── app.py                      # ACP server (Flask)
    ├── approach.conf               # System/mode selector
    ├── harmone_start.sh            # 3-process live launcher (Arch Linux, GUI terminals)
    ├── harmone_start_wsl.sh        # WSL/headless launcher; processes in background
    ├── harmone_stop.sh             # Kill from pidfile + psutil sweep
    ├── run_managed_system.py       # Master wrapper + process manager
    │
    ├── adapters/
    │   ├── base.py
    │   ├── loaders.py
    │   ├── regression_csv.py
    │   ├── cv_imagedir.py            # CVImageDirAdapter — image-dir / manifest-CSV dataset adapter
    │   └── tasks/
    │       ├── __init__.py
    │       ├── base.py               # TaskAdapter ABC + _TASK_REGISTRY + get_task_adapter()
    │       ├── detection.py          # YOLO detection adapter (SPPF hook, spatial mean-pool)
    │       ├── classification.py     # torchvision classification adapter
    │       └── segmentation.py       # SegFormer segmentation adapter (upsample before argmax)
    │
    ├── configs/
    │   ├── datasets/
    │   │   ├── _template.json
    │   │   ├── pems_node1.json
    │   │   ├── bdd100k.json
    │   │   ├── toy_regression.json
    │   │   ├── toy_cv.json
    │   │   ├── pems_node2.json       # awaiting_data
    │   │   ├── uci_electricity.json  # awaiting_data
    │   │   ├── spot_prices.json      # awaiting_data
    │   │   ├── iwildcam.json         # awaiting_data
    │   │   └── acdc.json             # awaiting_data
    │   └── experiments/
    │       ├── _template.yaml
    │       └── baseline.yaml
    │
    ├── core/
    │   ├── dataset_validator.py
    │   ├── energy.py
    │   ├── scoring.py
    │   ├── vmr.py
    │   ├── drift/
    │   │   ├── base.py
    │   │   ├── kl_rolling.py
    │   │   ├── kl_fixed_ref.py
    │   │   ├── luminance_kl.py
    │   │   ├── mmd_embedding.py
    │   │   ├── frechet_embedding.py
    │   │   └── embedding_store.py    # float16 ring buffer for drift embeddings
    │   ├── planners/
    │   │   ├── base.py
    │   │   ├── naive.py
    │   │   ├── random_switch.py
    │   │   ├── greedy_switch.py
    │   │   ├── harmone_original.py
    │   │   ├── violation_aware.py
    │   │   ├── pareto.py
    │   │   └── bandit.py
    │   └── proxies/
    │       ├── base.py
    │       ├── confidence.py
    │       ├── calibrated_confidence.py
    │       └── agreement.py
    │
    ├── docs/
    │   ├── DATA_CONTRACT.md        # Plug-and-play data contract
    │   ├── datasets/               # Per-dataset expectation specs
    │   │   ├── index.md
    │   │   ├── pems_node2.md
    │   │   ├── uci_electricity.md
    │   │   ├── spot_prices.md
    │   │   ├── bdd100k.md
    │   │   ├── iwildcam.md
    │   │   └── acdc.md
    │   └── checkpoints/            # Sprint checkpoint reports
    │       ├── CP0.md
    │       ├── CP1.md
    │       ├── CP2.md
    │       ├── CP3.md
    │       ├── CP4.md
    │       ├── CP5.md
    │       ├── CP6.md
    │       └── CP7_integration_audit.md
    │
    ├── experiments/
    │   ├── metrics.py
    │   ├── offline_eval.py
    │   ├── proxy_validation.py
    │   ├── run_experiment.py
    │   ├── run_grid.py
    │   └── run_reset.py
    │
    ├── legacy/
    │   ├── README.md
    │   └── simulator.py            # Superseded by adapters/regression_csv.py + experiments/run_experiment.py
    │
    ├── frontend/
    │   └── dashboard.html
    │
    ├── managed_system_cv/
    │   ├── approach.conf
    │   ├── cleanup.sh
    │   ├── inference.py
    │   ├── retrain.py
    │   ├── knowledge/
    │   │   ├── thresholds.json
    │   │   ├── mape_info.json
    │   │   └── drift_kl.json
    │   ├── mape_logic/
    │   │   ├── monitor.py
    │   │   ├── analyse.py
    │   │   ├── plan.py
    │   │   ├── execute.py
    │   │   └── manage.py
    │   ├── retrain_tactics/
    │   │   ├── retrain_oracle.py
    │   │   ├── pseudo_label.py
    │   │   └── human_in_loop.py
    │   └── utility/
    │       ├── drift_utils.py
    │       ├── bdd_to_yolo_labels.py
    │       ├── get_models.py
    │       ├── prediction_stats.py
    │       ├── evaluate_run_against_labels.py
    │       ├── raw_model_stats.py
    │       └── drift/
    │           ├── induce.py
    │           ├── plot.py
    │           └── plot_ref.py
    │
    ├── managed_system_regression/
    │   ├── approach.conf
    │   ├── inference.py
    │   ├── retrain.py
    │   ├── retrain_svm_only.py
    │   ├── train.py
    │   ├── knowledge/
    │   │   ├── thresholds.json
    │   │   ├── mape_info.json
    │   │   └── drift_kl.json
    │   └── mape_logic/
    │       ├── monitor.py
    │       ├── analyse.py
    │       ├── plan.py
    │       ├── execute.py
    │       └── manage.py
    │
    ├── policies/
    │   ├── reg_harmone_score.json
    │   ├── reg_switch_r2.json
    │   ├── reg_single_lstm.json
    │   ├── reg_single_linear.json
    │   ├── reg_single_svm.json
    │   ├── cv_harmone_score.json
    │   ├── cv_switch_confidence.json
    │   ├── cv_single_yolo_s.json
    │   ├── cv_single_yolo_m.json
    │   ├── custom_regression_policy.json
    │   └── custom_cv_policy.json
    │
    ├── scripts/
    │   ├── _launch_common.sh       # Shared venv/pip/preflight steps for both launchers
    │   ├── init_regression.py
    │   ├── init_cv.py              # +--skip-embeddings; seeds reference_embeddings.npz (§5.2)
    │   ├── make_toy_datasets.py
    │   ├── probe_energy.py
    │   ├── setup_energy_permissions.sh
    │   └── validate_dataset.py
    │
    └── tests/
        ├── test_b1_energy_threshold.py
        ├── test_b2_vmr_reuse.py
        ├── test_b3_drift_reference.py
        ├── test_b4_model_cache.py
        ├── test_b5_kl_placeholder.py
        ├── test_b6_switch_counter.py
        ├── test_b7_scaler_leakage.py
        ├── test_adapters.py
        ├── test_planners.py
        ├── test_phase3_drift.py
        ├── test_phase3_proxies.py
        ├── test_phase3_vmr.py
        ├── test_phase4_energy.py
        ├── test_phase5_harness.py
        ├── test_live_run_fixes.py
        ├── test_plug_and_play.py
        ├── test_task_adapters.py   # §4 task adapter + EmbeddingStore + plug-and-play CV
        ├── test_api_endpoints.py   # G9 set-planner validation, G13 CORS, G2 policy routing, L4 None-safe (skips if Flask unavailable)
        └── test_dashboard_flow.py  # G2 policy poll routing, G8 startup error surfacing, planner variant flow (skips if Flask unavailable)
```

---

## File Descriptions

### `tool/legacy/` — Superseded files

| File | Purpose |
|---|---|
| `legacy/simulator.py` | Legacy per-step simulator for regression inference. Superseded by `adapters/regression_csv.py` + `experiments/run_experiment.py`. Preserved for git history context. |

### `examples/`

| File | Purpose |
|---|---|
| `examples/HarmonE/monitor.py` | Reference monitor for a standalone MAPE loop — reads predictions and computes quality signals. Shows the intended four-phase contract. |
| `examples/HarmonE/analyse.py` | Reference analyser — evaluates monitor output against thresholds and decides whether adaptation is needed. |
| `examples/HarmonE/plan.py` | Reference planner — selects an adaptation action (switch, retrain, noop) given the analysis verdict. |
| `examples/HarmonE/execute.py` | Reference executor — applies the planned action and records it. |
| `examples/templates/monitor.py` | Blank stub for a custom monitor; documents the required return contract. |
| `examples/templates/analyse.py` | Blank stub for a custom analyser. |
| `examples/templates/plan.py` | Blank stub for a custom planner. |
| `examples/templates/execute.py` | Blank stub for a custom executor. |

---

### `tool/` — top level

| File | Purpose |
|---|---|
| `app.py` | **ACP server.** Flask app (port 5000). Receives telemetry POSTs from `run_managed_system.py`, evaluates quality-attribute boundaries defined in the active policy JSON, fires tactic commands to the adaptation handler at port 8080. Exposes REST endpoints for the dashboard: `/api/telemetry`, `/api/policy`, `/api/knowledge/<policy_id>`, `/api/set-model`, `/api/set-planner` (writes `planner` key to `thresholds.json`; validates against 5-planner allowlist), `/api/write-approach`, `/api/save-policy`, `/api/reset`, `/api/start-managed-system` (waits 4 s; returns HTTP 500 + last 20 log lines on early exit), `/api/stop-managed-system`, `/api/upload-custom-mape`. Handles secondary boundary checks (`kl_div` drift) on a background monitoring thread with heartbeat logging and None-safe boundary evaluation. |
| `approach.conf` | System/mode selector. Single-token file: `reg_harmone`, `cv_switch`, etc. Carries the approach token only — never the planner name (planner lives in `thresholds.json["planner"]`). Read by `run_managed_system.py` at startup to select the managed system directory and policy file prefix. |
| `harmone_start.sh` | Convenience Bash script that opens three terminals (inference, MAPE manage, dashboard) and starts them in the correct order. Arch Linux entry point for a live run. |
| `run_managed_system.py` | **Master wrapper.** Reads `approach.conf`, performs the L1-b startup artifact health check, spawns the three subprocesses (inference, manage, dashboard server), streams telemetry from `predictions.csv` to the ACP, polls subprocess liveness every cycle, and shuts everything down cleanly on any child death or SIGINT. Also hosts the adaptation handler endpoint (port 8080) that receives tactic commands from the ACP and writes timestamped `command.txt`. |

---

### `adapters/`

| File | Purpose |
|---|---|
| `base.py` | `DatasetAdapter` ABC. Defines `Sample` and `ModelSpec` dataclasses and the interface contract (`train_split()`, `val_split()`, `stream()`, `offline_labels()`, `models()`). Every dataset integration subclasses this; no dataset-specific code appears anywhere else. |
| `loaders.py` | Model loader registry. Implements `lstm_loader`, `sklearn_loader`, and `yolo_loader` (CV stub) as `load(weights_path) → model` functions. `get_loader(dotted_path)` resolves a loader by dotted module path from config. Used by the experiment harness to load models without domain-specific code. |
| `regression_csv.py` | Concrete `DatasetAdapter` for CSV regression datasets. Reads config keys (`data_path`, `value_column`, `seq_length`, `train_frac`), builds sliding-window sequences, streams rows in order. The only place that knows PeMS or any CSV regression dataset layout. |

---

### `configs/datasets/`

| File | Purpose |
|---|---|
| `_template.json` | Fully annotated template listing every config key with its type, default, and purpose. Copy and fill to add a new dataset. |
| `pems_node1.json` | Config for the PeMS traffic speed regression dataset (node 1). Sets model paths (LSTM/Linear/SVM), thresholds, and energy bounds calibrated for this dataset. |
| `bdd100k.json` | Config for the BDD100K driving CV dataset. Sets YOLO n/s/m model paths, drift thresholds, and CV-specific energy bounds. |
| `toy_regression.json` | Config for the synthetic 5 000-row regression dataset generated by `make_toy_datasets.py`. Used exclusively by conformance tests. |
| `toy_cv.json` | Config for the synthetic 60-image CV dataset generated by `make_toy_datasets.py`. Used exclusively by conformance tests. |

### `configs/experiments/`

| File | Purpose |
|---|---|
| `_template.yaml` | Annotated grid config template showing all keys (`datasets`, `planners`, `seeds`, `overrides`). |
| `baseline.yaml` | Paper baseline grid: `pems_node1` × 6 planners (naive → pareto) × seeds [1, 2]. Overrides `energy_meter=null` and `stream_delay_s=0` for Windows-safe headless runs. |

---

### `docs/`

| File | Purpose |
|---|---|
| `docs/DATA_CONTRACT.md` | Plug-and-play data contract spec: required fields for regression CSV and CV manifest schemas, validation rules enforced by `core/dataset_validator.py`. |
| `docs/datasets/index.md` | Index of all six dataset specs with links and status (`awaiting_data` for five; toy datasets ready). |
| `docs/datasets/pems_node2.md` | Expected data schema and preprocessing recipe for PeMS node 2 regression dataset. |
| `docs/datasets/uci_electricity.md` | Expected schema for UCI Electricity dataset. |
| `docs/datasets/spot_prices.md` | Expected schema for Spot Prices structural-break dataset. |
| `docs/datasets/bdd100k.md` | Expected schema and YOLO label format for BDD100K. |
| `docs/datasets/iwildcam.md` | Expected schema for iWildCam classification dataset. |
| `docs/datasets/acdc.md` | Expected schema for ACDC adverse-conditions segmentation dataset. |
| `docs/checkpoints/CP0.md` | Phase 0 checkpoint: environment and baseline green. |
| `docs/checkpoints/CP1.md` | Phase 1 checkpoint: B1–B7 fixes, 37/37 tests. |
| `docs/checkpoints/CP2.md` | Phase 2 checkpoint: pluggable interfaces, planner registry, adapter ABC. |
| `docs/checkpoints/CP3.md` | Phase 3 checkpoint: CV generalisation, task adapters, embedding drift, proxy validation. |
| `docs/checkpoints/CP4.md` | WSL smoke test checkpoint: regression smoke verified 2026-07-28; CV smoke pending lab machine. |
| `docs/checkpoints/CP5.md` | Documentation verification checkpoint: endpoint names corrected, context docs consistency audit. |
| `docs/checkpoints/CP6.md` | Pre-push checkpoint: 290/290 tests, hygiene greps clean, runtime files untracked. |
| `docs/checkpoints/CP7_integration_audit.md` | Integration hardening audit: G1–G16 findings, G12 preset matrix (10/10 PASS), endpoint inventory, DoD audit. |

---

### `core/`

| File | Purpose |
|---|---|
| `dataset_validator.py` | `validate(config_path) → ValidationReport`. Enforces the plug-and-play data contract: config keys present and typed, `train_frac + val_frac ≤ 1`, regression CSV has the declared column with no NaNs and enough rows, CV image directory/manifest is readable. Called at startup by `run_managed_system.py` and both init scripts. |
| `energy.py` | **Hardware-agnostic energy measurement.** `EnergyMeter` context manager with three backends: `_PyJoulesRaplBackend` (CPU RAPL), `_PyJoulesNvmlBackend` (GPU cumulative counter), `_PollingGPUBackend` (50 ms nvidia-smi integration fallback). Module-level `_ACTIVE` flag prevents nested contexts from double-counting package-wide RAPL counters. `_probe_rapl()` / `_probe_nvml()` validate backends lazily; results cached to `.energy_backends.json`. Reports `cpu_uJ`, `gpu_uJ`, `total_uJ` in µJ; `valid` flag distinguishes real from null readings. |
| `scoring.py` | Shared scoring math. `update_energy_threshold()` implements Eq. 3 (adaptive threshold clamp). `update_separated_emas()` maintains per-model `ema_accuracy` and `ema_energy` signals alongside the legacy composite `ema_scores`. |
| `vmr.py` | **Versioned Model Repository.** `VMR` class with `store(model, weights, distribution, meta)`, `best_match(model, current_dist, strategy)`, `list_versions()`, and `restore()`. Strategies: `best_score` (highest proxy_score) and `closest_distribution` (KL for histograms, Euclidean for embeddings). Replaces ad-hoc file scanning in the managed systems' `analyse.py` files. |

#### `core/drift/`

| File | Purpose |
|---|---|
| `base.py` | `DriftDetector` ABC with `fit_reference(data)` and `score(window) → float\|None`. `get_detector(name)` registry maps config strings to classes. |
| `kl_rolling.py` | Rolling-window KL divergence between adjacent prediction windows. The paper's original drift signal — detects local distribution changes. |
| `kl_fixed_ref.py` | Fixed-reference KL divergence: `KL(current_window ‖ training_distribution)`. Detects gradual drift that rolling KL misses. Reads reference from `knowledge/reference_distribution.json`. (B3 fix) |
| `luminance_kl.py` | CV wrapper around `kl_fixed_ref` operating on luminance histograms of images rather than prediction values. |
| `mmd_embedding.py` | Unbiased MMD² with RBF kernel. Bandwidth frozen at `fit_reference()` via median pairwise distance heuristic. Used when embedding-based drift detection is configured. |
| `frechet_embedding.py` | FID-style Fréchet distance between Gaussian approximations of reference and current embedding distributions. Higher sensitivity to distribution shape than MMD. |

#### `core/planners/`

| File | Purpose |
|---|---|
| `base.py` | `PlanningContext` dataclass (violation, EMA scores, thresholds, current model). `Planner` ABC with `plan(ctx) → Decision`. `get_planner(name)` registry. |
| `naive.py` | **S1 — Naive.** Always returns noop. Used as the lower-bound baseline (no adaptation). |
| `random_switch.py` | **S2 — Random switch.** Picks uniformly at random from non-current models. Corresponds to the paper's exploration arm in isolation. |
| `greedy_switch.py` | **S3 — Greedy switch.** Always switches to `argmax(ema_scores)` excluding the current model. Pure exploitation without ε. |
| `harmone_original.py` | **S4 — HarmonE original (default).** ε-greedy: with probability α explores randomly; otherwise exploits best EMA alternative. Triggers VMR replace or retrain on drift. Preserves exact paper behaviour. |
| `violation_aware.py` | **S5 — Violation-aware.** Differentiates score violations (switch to best accuracy model) from energy violations (switch to lightest model); does not mix the two decision paths. |
| `pareto.py` | **S6 — Pareto.** Selects via weighted Chebyshev distance on the (accuracy, energy) objective space. Weights `w_acc` and `w_e` are config-driven. |
| `bandit.py` | **S7 — LinUCB stub.** Raises `NotImplementedError`. Placeholder for a contextual bandit planner in future work. |

#### `core/proxies/`

| File | Purpose |
|---|---|
| `base.py` | `AccuracyProxy` ABC with `score(inference_record) → float`. `get_proxy(name)` registry. |
| `confidence.py` | Mean detection confidence per inference record. The paper's implicit default proxy. |
| `calibrated_confidence.py` | Platt/temperature-scaled confidence. Reads calibration parameters from `knowledge/calibration.json`. Enables RQ3 proxy comparison. |
| `agreement.py` | Multi-model detection-level agreement (IoU > threshold, same predicted class). Falls back to raw confidence when companion predictions are unavailable. |

---

### `experiments/`

| File | Purpose |
|---|---|
| `metrics.py` | Post-run analysis. `compute_run_metrics(run_dir)` summarises a single run. `aggregate_grid(grid_dir)` collects all runs in a grid. `aggregate_by_planner(rows)` collapses seeds. `pareto_efficiency(rows)` marks non-dominated solutions. `wilcoxon_test()` for paper statistical claims. `to_csv()` / `to_latex()` output formatters. |
| `offline_eval.py` | Computes true mAP@0.5 per monitoring interval for a completed CV run, aligned to `predictions.csv` boundaries, using ground-truth YOLO label files. Produces per-interval JSON results for proxy validation. |
| `proxy_validation.py` | Spearman ρ pipeline (RQ3). Compares confidence, calibrated_confidence, and agreement proxies against true mAP@0.5 from `offline_eval.py`. Single command produces the ρ table for the paper. |
| `run_experiment.py` | **Headless inline MAPE loop.** Runs a single (dataset × planner × seed) combination in one Python process — no Flask, no subprocesses. Streams from the adapter, calls monitor→analyse→plan→execute inline, writes per-run artifacts (`predictions.csv`, `mape_events.csv`, `run_manifest.json`). Energy backend defaults to null for Windows/CI safety. |
| `run_grid.py` | **Grid driver.** Loads a YAML/JSON grid config, iterates all dataset × planner × seed combinations, calls `run_experiment` for each, resumes incomplete grids (skips dirs with `run_manifest.json`), writes `grid_manifest.json`. |
| `run_reset.py` | **Per-run state reset.** Zeros `event_counters`, `ema_scores` → 0.5, `last_switch_ts` → 0.0, `last_line`, `recovery_cycles`. Truncates `predictions.csv` to header. Deletes `command.txt`, `drift.csv`, `drift_kl.json`. Preserves `scaler.pkl`, `reference_distribution.json`, `versionedMR/`. Must be called before every live session to prevent cross-run counter bleed. |

---

### `frontend/`

| File | Purpose |
|---|---|
| `dashboard.html` | **Live monitoring dashboard.** Single-page Tailwind + Chart.js app served at port 8000. Plots rolling R²/confidence score, normalised energy, per-model EMA scores, and model switches in real time. Welcome screen has buttons for all 10 policy presets across four optgroups; clicking a HarmonE button opens the **Planner Selection Modal** (5 planners: harmone_original, greedy_switch, violation_aware, pareto, random_switch) before launching. Calls `/api/set-planner` to persist the planner choice, then proceeds with the standard approach flow. CV HarmonE is fully enabled. 12 `HARMONY_PRESETS` (was 6) and 5-planner `PLANNERS` array drive the modal and preset form. |

---

### `managed_system_cv/`

| File | Purpose |
|---|---|
| `approach.conf` | CV-specific mode selector. Sets `system=cv`, `run_mode`, and default planner. |
| `cleanup.sh` | **Legacy pre-run reset script.** Archives `knowledge/` to `runs_artifact/`, resets `models/` from `base_models/`, wipes `predictions.csv`, resets `mape_info.json` and `model.csv`. Superseded by `experiments/run_reset.py` but kept for quick manual resets during development. |
| `inference.py` | **CV inference loop.** Reads `knowledge/model.csv` each iteration to pick the active YOLO model. Runs YOLOv8 detection on BDD100K images. Appends to `predictions.csv` (confidence, model, inference time, energy in µJ, luminance histogram). Also seeds `versionedMR/` on first run. Includes a startup GPU arch guard that fast-fails with a remediation message if the installed PyTorch has no kernels for the detected GPU. (L3 fix) |
| `retrain.py` | **CV YOLO fine-tune.** Triggered by drift execute when action == "retrain". Builds an augmented training batch matched to the detected drift type (dark/fog), fine-tunes the last-N layers of the current model for a fixed epoch count, archives the result to `versionedMR/` via `VMR.store()`, and writes `knowledge/model_reload.flag`. |

#### `managed_system_cv/knowledge/` (runtime state)

| File | Purpose |
|---|---|
| `thresholds.json` | All CV adaptation thresholds: `min_score`, `max_energy`, `alpha`, `beta`, `gamma`, `E_m`, `E_M`, `tau_drift`, `ema_head_start`, `energy_meter`, `switch_cooldown_s`. Edited by users to tune the system; never written by runtime code. |
| `mape_info.json` | Live MAPE state: `ema_scores`, `ema_accuracy`, `ema_energy`, `event_counters`, `last_switch_ts`, `last_line`, `recovery_cycles`. Written by `execute.py` and `analyse.py` on every MAPE cycle. |
| `drift_kl.json` | Output of the last drift analysis: best version path per model and overall minimum KL divergence. Populated by `analyse.py`; read by the dashboard. |

#### `managed_system_cv/mape_logic/`

| File | Purpose |
|---|---|
| `monitor.py` | Reads new rows from `predictions.csv` since `last_line`. Computes mean confidence (score), normalised energy, and luminance KL divergence. Returns `{"fresh": False}` when no new rows exist — never re-serves stale data as fresh telemetry. (L1-c fix) |
| `analyse.py` | Evaluates monitor output against thresholds. Updates the adaptive energy threshold (Eq. 3), EMA scores, and `recovery_cycles`. Returns `switch_needed` and `threshold_violated`. Also runs `analyse_drift()` which computes per-model KL divergences against `versionedMR/` histograms. |
| `plan.py` | Selects an adaptation action. Checks `switch_cooldown_s` before any switch (L5 anti-thrash guard). Routes `plan_mape()` (ε-greedy), `plan_drift()` (VMR replace or retrain), `plan_random_switch()`, and `plan_greedy_switch()` based on the trigger and configured planner. |
| `execute.py` | Carries out the plan decision: writes `knowledge/model.csv`, calls `retrain.py`, logs events to `event_log.csv`. Tracks `last_switch_ts` for the cooldown guard. Wraps all execution in `EnergyMeter` and accumulates MAPE-K energy into `event_counters`. |
| `manage.py` | **CV MAPE orchestrator.** On startup: clears stale `command.txt` (L2-a fix). Runs the MAPE loop on a timer. Spawns an ACP command listener thread that reads `command.txt` with timestamp gating — commands older than 30 s are discarded (L2-b fix). |

#### `managed_system_cv/retrain_tactics/`

| File | Purpose |
|---|---|
| `retrain_oracle.py` | **Supervised retrain.** Deduces drift type (dark/fog/clear) from luminance statistics, builds an augmented training set, fine-tunes the last N layers for 5 epochs. Only fires when `offline_labels()` are available; returns `{"status": "skipped", "reason": "no_labels"}` otherwise. |
| `pseudo_label.py` | **Label-free retrain.** Runs the current model over the recent drift window and keeps detections with confidence > `tau_pseudo`. Fine-tunes on this pseudo-labeled set. Safety valve: discards new weights if the proxy score drops more than `tau_regression` versus a held-out subsample. |
| `human_in_loop.py` | **Human-in-the-loop tactic.** When neither oracle nor pseudo-label is viable: emits a `labeling_required` event, switches to the lowest-energy model to reduce cost while waiting, and polls `knowledge/incoming_labels/` on subsequent MAPE ticks. Triggers oracle retrain once YOLO label files arrive. |

#### `managed_system_cv/utility/`

| File | Purpose |
|---|---|
| `drift_utils.py` | Shared image analysis math: `luminance_histogram()` (Y-channel 64-bin histogram), `kl_divergence()`, `window_hist_stats()`. Imported by both `inference.py` and `analyse.py`. |
| `bdd_to_yolo_labels.py` | Converts BDD100K JSON annotation files to per-image YOLO `.txt` label format (class cx cy w h). Run once as a data preparation step before training or offline evaluation. |
| `get_models.py` | Downloads YOLOv8 n/s/m pretrained weights from the Ultralytics GitHub release page into `base_models/`. Run once during environment setup. |
| `prediction_stats.py` | Quick analysis script: reads `predictions.csv` and prints per-model mean confidence, inference time, and energy. Useful for sanity-checking a completed run. |
| `evaluate_run_against_labels.py` | Computes overall P/R/F1 by matching inference `.txt` output files against ground-truth YOLO label files at a configurable confidence threshold. |
| `raw_model_stats.py` | **Pilot energy measurement.** Runs each YOLO model on training images, records raw inference energy with `EnergyMeter`. Used to calibrate `E_m`/`E_M` normalisation bounds in `thresholds.json` for a specific hardware setup. |
| `utility/drift/induce.py` | Applies synthetic luminance drift (dark = reduced mean; fog = reduced std) to BDD100K train images in 7 randomised-size intervals. Fixed random seed ensures reproducible experiment conditions. |
| `utility/drift/plot.py` | Plots rolling KL divergence across the BDD100K stream with sliding windows (adjacent-window comparison). Supports RGB and luminance histogram modes. |
| `utility/drift/plot_ref.py` | Plots fixed-reference KL divergence: each window compared to the first N test images as a static reference. Shows gradual drift that rolling KL misses. |

---

### `managed_system_regression/`

| File | Purpose |
|---|---|
| `approach.conf` | Regression-specific mode selector. |
| `inference.py` | **Regression inference loop.** Loads the active model from `knowledge/model.csv` (with module-level cache to avoid per-step reload). Applies `scaler.pkl` transform to the sliding window input. Runs LSTM/Linear/SVM prediction. Appends to `predictions.csv` (true value, predicted value, model, time, energy). Writes `model_reload.flag`-aware cache invalidation. |
| `retrain.py` | **Regression retrain.** On drift signal: retrains all three models (LSTM 50 epochs, Ridge, SVR) on `knowledge/drift.csv` (recent anomalous data). Archives new weights to `versionedMR/<model>/version_N/`. Writes `model_reload.flag`. Saves new `scaler.pkl` fitted on the drift batch only. |
| `retrain_svm_only.py` | Quick standalone SVM retrain on the full `dataset.csv`. Development/debug utility when only the SVM needs refreshing without triggering the full drift retrain pipeline. |
| *(moved)* `simulator.py` | Moved to `tool/legacy/simulator.py`. Superseded by `adapters/regression_csv.py` + `experiments/run_experiment.py`. |
| `train.py` | **Initial training script.** Trains LSTM (50 epochs), Ridge regression, and SVR on `dataset.csv` (train split only). Writes weights to `models/` and seeds `versionedMR/` with version 1 of each model. Run once during environment setup. |

#### `managed_system_regression/knowledge/` (runtime state)

| File | Purpose |
|---|---|
| `thresholds.json` | Regression adaptation thresholds: `min_score`, `max_energy`, `E_m`, `E_M`, `alpha`, `beta`, `gamma`, `tau_drift`, `energy_meter`, `switch_cooldown_s`. |
| `mape_info.json` | Live MAPE state mirroring the CV equivalent. |
| `drift_kl.json` | Last KL drift analysis output for regression. |

#### `managed_system_regression/mape_logic/`

| File | Purpose |
|---|---|
| `monitor.py` | Reads new rows from `predictions.csv` since `last_line`. Computes R² over the window, normalises energy against `E_m`/`E_M`, runs both fixed-ref and rolling KL detectors. Returns `{"fresh": False}` when no new rows — stale telemetry path removed (L1-c fix). |
| `analyse.py` | Evaluates monitor signals against thresholds. Selects the active KL signal based on `drift_reference` config (`"fixed"`, `"rolling"`, or `"both"`). Updates EMA via `core/scoring`. Returns `switch_needed` and `threshold_violated`. |
| `plan.py` | Regression planning layer. `dispatch_plan()` routes through the planner registry for Phase-2 planners. Legacy `plan_mape()`, `plan_drift()`, `plan_random_switch()`, `plan_greedy_switch()` functions remain for backward compatibility. All paths check `switch_cooldown_s` (L5 guard). |
| `execute.py` | Executes the regression plan: writes `model.csv`, runs `retrain.py` or VMR replace, records events, tracks `last_switch_ts`. |
| `manage.py` | **Regression MAPE orchestrator.** Startup command.txt clear (L2-a), MAPE loop, ACP command listener thread with timestamp gating (L2-b). Identical structure to CV manage.py. |

---

### `policies/`

Policy JSON files consumed by `app.py`. Each defines one `quality_attribute` primary boundary and optional secondary boundaries. `app.py` fires the listed tactic at the adaptation handler when a boundary is violated.

| File | Purpose |
|---|---|
| `reg_harmone_score.json` | Fire `execute_mape_plan` when R² score < 0.78; secondary: `handle_data_drift` when KL > 0.1. The paper's main regression policy. |
| `reg_switch_r2.json` | Fire `execute_random_switch` on R² violation. Random-switch baseline (S2). |
| `reg_single_lstm.json` | No adaptation; LSTM always active. Single-model lower bound (S1). |
| `reg_single_linear.json` | No adaptation; Linear always active. |
| `reg_single_svm.json` | No adaptation; SVM always active. |
| `cv_harmone_score.json` | Fire `execute_mape_plan` when confidence < 0.54; secondary: drift tactic. Main CV policy. |
| `cv_switch_confidence.json` | Fire `execute_random_switch` on confidence violation. CV random-switch baseline. |
| `cv_single_yolo_n.json` | No adaptation; YOLOv8-N always active. Lightest single-model baseline. |
| `cv_single_yolo_s.json` | No adaptation; YOLOv8-S always active. |
| `cv_single_yolo_m.json` | No adaptation; YOLOv8-M always active. |
| `custom_regression_policy.json` | User-editable template for a regression policy. Copy and modify thresholds without touching code. |
| `custom_cv_policy.json` | User-editable template for a CV policy. |

---

### `scripts/` (tool/scripts/)

| File | Purpose |
|---|---|
| `init_regression.py` | **Regression init.** Reads a dataset config, loads the CSV, fits `MinMaxScaler` on the training split only (atomically written), computes a 50-bin reference histogram, saves both to `managed_system_regression/knowledge/`. Idempotent; `--force` overwrites. Supersedes the root-level `scripts/init_scaler.py` + `scripts/init_reference.py`. |
| `init_cv.py` | **CV init.** For each model in the dataset config, copies base weights to `versionedMR/{model}_v1.pt` and computes an average luminance histogram over reference images → `versionedMR/{model}_v1_hist.json`. Also writes a fresh `model.csv` and blank `mape_info.json`. Idempotent; `--force` overwrites. |
| `make_toy_datasets.py` | Generates the synthetic toy datasets used by conformance tests: a 5 000-row sinusoidal CSV (`data/toy_regression/dataset.csv`) and 60 solid-colour JPEG images + manifest (`data/toy_cv/`). |
| `probe_energy.py` | **E1/E2 hardware probe.** Tests whether pyJoules RAPL and NVML backends return non-zero readings on the current machine. Falls back through pynvml power polling → nvidia-smi subprocess for GPU. Caches results to `knowledge/.energy_backends.json` keyed by hostname. Exits 1 if all probes fail. |
| `setup_energy_permissions.sh` | Once-per-boot shell script (run with sudo). Loads `msr`, `intel_rapl_common`, `intel_rapl_msr` kernel modules and sets powercap sysfs permissions to 777. Required before any RAPL energy measurement on Linux. |
| `validate_dataset.py` | CLI wrapper for `core/dataset_validator.py`. Accepts a config name or path, prints the PASS/FAIL table, exits 1 on failure. |

---

### `tests/`

| File | Purpose |
|---|---|
| `test_b1_energy_threshold.py` | Verifies B1 fix: `update_energy_threshold()` tightens as well as loosens; never reaches 1.0 asymptotically. |
| `test_b2_vmr_reuse.py` | Verifies B2 fix: `analyse_drift()` returns the correct `action`/`version` keys so the VMR replace path is reachable. |
| `test_b3_drift_reference.py` | Verifies B3 fix: fixed-reference KL detector detects gradual drift that rolling KL misses; `drift_reference` config key routes correctly. |
| `test_b4_model_cache.py` | Verifies B4 fix: model is loaded from cache on repeated calls with the same `model.csv` content; reloaded only on model change or flag file. |
| `test_b5_kl_placeholder.py` | Verifies B5 fix: warmup period returns `{"kl_div": None}` instead of a random float; downstream consumers skip boundary checks on None. |
| `test_b6_switch_counter.py` | Verifies B6 fix: `event_counters["model_switches"]` increments only when a model switch actually executes, not on noop planning cycles. |
| `test_b7_scaler_leakage.py` | Verifies B7 fix: scaler params equal the train-split min/max, not the full-dataset extremes. |
| `test_adapters.py` | Tests the `DatasetAdapter` interface contract and `RegressionCSVAdapter` stream/split behaviour. |
| `test_planners.py` | Tests all 7 planners via `PlanningContext` mocks: noop returns, switch selection, cooldown guard, energy vs score separation, Pareto distance. |
| `test_phase3_drift.py` | Tests all drift detectors: KL fixed-ref updates, MMD kernel, Fréchet Gaussian approximation. |
| `test_phase3_proxies.py` | Tests all three accuracy proxies: confidence mean, calibrated scaling, agreement fallback. |
| `test_phase3_vmr.py` | Tests `VMR.store()`, `best_match()` with both strategies, and `restore()` path invariants. |
| `test_phase4_energy.py` | Tests `EnergyMeter`: null backend validity, backends dict structure, `cpu_valid`/`gpu_valid` flags, unit conversion consistency. |
| `test_phase5_harness.py` | Tests `run_experiment()` inline loop, `run_grid()` resume logic, `compute_run_metrics()`, `aggregate_by_planner()`, and model loaders. |
| `test_live_run_fixes.py` | Tests covering the July 24 live-run bug fixes plus CP7 additions: L2 command gating, L4 None-safe boundary eval, L1-c stale monitor, L2-b/L6 run reset (incl. `last_switch_ts`), L1-a init_regression, E3 energy abstraction (no pyRAPL, nesting, null backend), L3 GPU arch guard, G4 reset preserves thresholds.json, G7 CV drift tactic dispatch. |
| `test_plug_and_play.py` | 15 conformance tests proving config-only dataset onboarding: validator schema enforcement, NaN detection, CV image checks, init_regression end-to-end, force-overwrite behaviour. |
| `test_api_endpoints.py` | CP7 Phase 3 tests: G9 set-planner validation (valid/invalid/bandit/bad-system), G13 CORS registration, G2 policy routing, G8 startup error surfacing, L4 None kl_div skip. Skips if Flask not installed (`pytest.importorskip`). |
| `test_dashboard_flow.py` | CP7 Phase 3 tests: scripted dashboard flow simulation — set-planner accepted, policy registered under base policy_id, telemetry populates history, planner variants use correct policy_id, startup errors surface as HTTP 500. Skips if Flask not installed. |

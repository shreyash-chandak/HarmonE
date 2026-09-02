# HarmonE — Sustainable Runtime MLOps Adaptation

This repository is a **journal-extension implementation of HarmonE** (ECSA 2025), the
self-adaptive MLOps approach that operationalises the MAPE-K
(Monitor–Analyse–Plan–Execute over a shared Knowledge base) feedback loop for sustainable ML
pipelines: it monitors prediction quality and energy consumption at runtime, adapts model
selection, retrains or restores a versioned model when drift is detected, and records every
adaptation decision in a structured event log.

HarmonE is a single-dataset, regression-only (PeMS traffic
prediction) research experiment this codebase derives from. This repository has
since grown far beyond that single case: it generalises HarmonE's architecture across 6
regression/CV datasets, adds a unified computer-vision task-adapter layer (detection,
classification, segmentation), 5 additional adaptation planners beyond the paper's original
policy, 4 additional drift detectors, test-time adaptation, and synthetic drift injection for
controlled evaluation — a body of work that extends the HarmonE *approach* itself, and also
Harmonica's regression demo.

---

## What's here

- **9 adaptation planners** (`core/planners/`): the paper's original score-threshold policy
  (`harmone_original`), plus `naive`/`naive_prt` pinned baselines, `random_switch`/
  `random_switch_prt`, `greedy_switch`, and three extensions built and bug-fixed this round —
  `violation_aware` (S5: routes each violation type to a targeted response), `pareto` (S6:
  augmented reference-point scalarization over an empirical accuracy/energy Pareto front), and
  `bandit` (S7: LinUCB contextual bandit with dual accuracy/energy estimators and a Lyapunov
  energy-debt penalty). All four non-trivial planners share a common hysteresis guard
  (`core/planners/hysteresis.py`) to avoid switch thrashing.
- **7 drift detectors** (`core/drift/`): the paper's rolling-window KL (`kl_rolling`), a
  fixed-reference KL detector that fixes the paper's gradual-drift blind spot (`kl_fixed_ref`),
  a luminance-histogram CV baseline (`luminance_kl`), two embedding-space detectors for CV
  (`mmd_embedding`, `frechet_embedding`), and two detectors added from Augur (Lewis et al.,
  SE4RAI'22) — `hellinger` and `energy_distance` — plus `experiments/calibrate_drift_threshold.py`,
  which implements Augur's threshold-selection method: a per-dataset drift threshold calibrated
  from a null distribution of scores on held-out in-distribution data, instead of a hand-set
  default.
- **Test-time adaptation** (`core/tta/tent.py`): a from-scratch implementation of TENT (Wang et
  al., ICLR 2021) — label-free entropy-minimization adaptation of BatchNorm affine parameters —
  wired in as a `retrain_tactic` for classification datasets (`imagenet`/`imagenet_c`).
- **A unified CV task-adapter layer** (`adapters/tasks/`): detection (YOLOv8), classification
  (torchvision), segmentation (SegFormer) share one task-agnostic MAPE loop; adding a new CV
  task requires a new adapter, not new MAPE code.
- **Versioned Model Repository (VMR)** (`core/vmr.py`): re-uses archived model weights when
  drift recurs and a close-enough archived distribution exists, instead of always retraining
  from scratch.
- **Synthetic drift induction** for controlled evaluation: region scale/shift injection for the
  three regression datasets (`managed_system_regression/utility/drift/induce.py`) and a
  corruption-ordered ImageNet-C-lite variant for CV (`managed_system_cv/utility/drift/
  induce_imagenet_c.py`), both built for the same reason Augur calls "Drifter" — engineered,
  documented, reproducible drift rather than hoping a dataset happens to contain some.
- **Offline ground-truth evaluation** (`experiments/offline_eval_{classification,detection,
  segmentation}.py`): replays a completed run's logged model-switch decisions against real
  labels to compute true accuracy/mAP/mIoU, merged into that run's `run_manifest.json`
  alongside the live proxy-based metrics.
- **Dataset plug-and-play** via JSON config (`core/dataset_validator.py`) — no code changes to
  add a new dataset, only a config + a preprocessing script that produces a manifest matching
  the contract in `tool/docs/DATA_CONTRACT.md`.

---

## Architecture

Two coexisting entry points share the same `core/` library:

```
┌─────────────────────────────────────────────────────────────────┐
│  Live system  (app.py :5000 + run_managed_system.py)            │
│  ACP telemetry server + wrapper; launches inference + MAPE       │
│  processes per domain; dashboard at frontend/dashboard.html      │
├─────────────────────────────────────────────────────────────────┤
│  Experiment harness  (experiments/run_experiment.py, run_grid.py)│
│  Single-process, seeded, reproducible runs; drives the paper's   │
│  planner/dataset comparison grid (scripts/run_regression.sh,     │
│  run_cv.sh, run_offline_eval.sh, calibrate_drift_threshold.py)   │
└──────────────────────────┬────────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────┐
│  Managed systems                                                    │
│  managed_system_regression/  (RegressionCSVAdapter, LSTM/Ridge/SVR) │
│  managed_system_cv/          (TaskAdapter, EmbeddingStore, VMR)     │
├───────────────────────────────────────────────────────────────────┤
│  Core (tool/core/)                                                  │
│  planners/  (9 policies + shared hysteresis)                        │
│  drift/     (7 detectors: KL/luminance/MMD/Fréchet/Hellinger/energy)│
│  tta/       (TENT test-time adaptation)                             │
│  energy.py  scoring.py  vmr.py  dataset_validator.py                │
└───────────────────────────────────────────────────────────────────┘
```

The live system is the original Harmonica-derived demo path (ACP server + dashboard,
long-running processes). The experiment harness is what this journal extension's comparison
grid actually runs on — seeded, one-shot, and what `scripts/run_*.sh` drive. Both read the
same `configs/datasets/*.json` and share `core/`.

---

## Quickstart — live system (WSL, headless, CPU-only)

Energy probes are unavailable in WSL; `EnergyMeter` falls back to null meters and logs
`valid: false` rows — the pipeline still runs to completion.

```bash
git clone <this-repo>
cd HarmonE-tool/tool

# 1. Generate a toy regression dataset
python scripts/make_toy_datasets.py

# 2. Init a regression managed system
python scripts/init_regression.py --config pems

# 3. Launch (venv created + deps installed automatically)
./harmone_start_wsl.sh --config pems
# → ACP server at http://localhost:5000/
# → Dashboard at http://localhost:8000/dashboard.html

# 4. Let it run a few telemetry cycles, then stop
./harmone_stop.sh
```

## Quickstart — Linux with energy probes (RAPL/NVML)

```bash
cd HarmonE-tool/tool
python scripts/make_toy_datasets.py
sudo chmod -R a+r /sys/class/powercap/intel-rapl/   # one-time per session
python scripts/probe_energy.py                       # verify RAPL/NVML are readable
python scripts/init_regression.py --config pems
./harmone_start.sh                                    # opens three terminal windows
```

---

## Running the experiment grids

This is the primary workflow used for the journal extension's dataset/planner comparison.

```bash
cd tool/

# Single run (regression, S4 planner, seed 1)
python experiments/run_experiment.py --dataset pems --planner harmone_original --seed 1

# Full regression grid: 3 drift-induced datasets x 9 planners
# (see the script header for the drift-induced-CSV prerequisite)
bash scripts/run_regression.sh --seed 1

# Full CV grid: bdd100k, acdc, imagenet, imagenet_c x 9 planners
# (iwildcam is present but excluded — weights were never produced; see
# DECISIONS_PENDING.md DP21/DP22)
bash scripts/run_cv.sh --seed 1

# Recover true ground-truth accuracy (mAP/mIoU/top-1) for completed CV runs,
# merged into each run's run_manifest.json alongside the live proxy metrics
bash scripts/run_offline_eval.sh --seed 1

# Calibrate a per-dataset drift threshold from held-out in-distribution data
python experiments/calibrate_drift_threshold.py --dataset pems

# Generate a results dashboard (PNG charts + embedded HTML) from runs/
python scripts/plot_results.py
```

---

## Datasets

| ID | Config | Domain | Task | Models | Notes |
|---|---|---|---|---|---|
| R1 | `pems` / `pems_driftinduced` | regression | traffic flow | LSTM, Ridge, SVR | driftinduced variant streams a region scale/shift-injected test split |
| R2 | `uci_electricity` / `uci_electricity_driftinduced` | regression | electricity load | LSTM, Ridge, SVR | |
| R3 | `spot_prices` | regression | ERCOT spot price | LSTM, Ridge, SVR | |
| C1 | `bdd100k` | CV / detection | object detection | YOLOv8n/s/m | weather/time-of-day drift-ordered manifest |
| C2 | `iwildcam` | CV / classification | species classification | EfficientNet-B0/ResNet50/101 | present, not in the active grid — weights never produced |
| C3 | `acdc` | CV / segmentation | semantic segmentation | SegFormer B0/B1/B2 | adverse-condition drift-ordered manifest |
| C4 | `imagenet` / `imagenet_c` | CV / classification | 100-class subset classification | EfficientNet-B0/ResNet50/101 | trial replacement for C2; `imagenet_c` is a corruption-ordered synthetic-drift variant (TENT's own ImageNet-C benchmark) sharing C4's trained weights |

Per-dataset recipes: `tool/docs/datasets/`.

---

## Adding a dataset

1. Preprocess to the contract schema (`tool/docs/DATA_CONTRACT.md`).
2. Copy `tool/configs/datasets/_template.json`, fill in your values.
3. Validate: `python scripts/validate_dataset.py <name>`.
4. Init: `python scripts/init_regression.py` or `scripts/init_cv.py`.
5. Run: `python experiments/run_experiment.py --dataset <name> --planner <planner>`.

No code changes needed for a dataset that fits an existing task type. See
`tool/docs/datasets/` for worked examples across both domains.

---

## Configuration reference

Common threshold keys in each `configs/datasets/*.json` (and, for the live system,
`managed_system_*/knowledge/thresholds.json`):

| Key | Paper symbol | Meaning |
|-----|-------------|---------|
| `min_score` | S_min | Minimum acceptable HarmonE score |
| `min_accuracy` | — | Accuracy floor used by S5/S6/S7 (separate from the combined score) |
| `max_energy` | τ_E (initial) | Initial energy threshold |
| `E_ref` / `energy_reference` | E_ref | Reference normalised energy for the adaptive-threshold update and S6's scalarization |
| `delta` | δ | Step size for the adaptive energy-threshold update |
| `beta` | β | Accuracy weight in the composite score |
| `gamma` | γ | EMA smoothing factor |
| `alpha` | α | ε-greedy exploration probability (S3/S4) |
| `E_m` / `E_M` | E_m / E_M | Energy normalisation bounds (calibrate per hardware) |
| `tau_drift` | τ_drift | Drift detection threshold — calibrate with `calibrate_drift_threshold.py` |
| `drift_reference` | — | `"fixed"` / `"rolling"` / `"both"` |
| `drift_detector` | — | `"kl_fixed_ref"` / `"mmd_embedding"` / `"frechet_embedding"` / `"hellinger"` / `"energy_distance"` (CV harness currently always uses luminance KL regardless of this key — DP20, open) |
| `retrain_tactic` | — | `"pseudo_label"` (default) or `"tent"` (classification only) |
| `s5_switch_margin`, `pareto_switch_margin`, `pareto_augmentation`, `bandit_lambda_0`, `bandit_mu`, `bandit_min_observation_steps`, `tent_lr`, `tent_batch_size`, `tent_steps` | — | Planner/TTA-specific tuning; documented in each module's own docstring |

---

## Design decisions

- **pyJoules instead of pyRAPL**: pyRAPL is unmaintained; pyJoules provides the same RAPL/NVML access with active maintenance.
- **Fixed-reference drift as the primary signal**: a reference distribution captured at init time, not a rolling window, so gradual drift is not missed. Rolling KL is retained as a secondary/comparison signal.
- **Task-adapter unification**: CV MAPE logic is task-agnostic; task-specific I/O lives in `adapters/tasks/{detection,classification,segmentation}.py`.
- **Pinned embedding model**: one fixed model extracts drift embeddings regardless of which model is currently serving inference, to keep the embedding space consistent across switches.
- **Fail-loud philosophy**: energy metering, embedding extraction, and TENT's BatchNorm collection raise or warn loudly on missing/invalid state rather than silently imputing.
- **Planner hysteresis is shared, not per-planner**: S5/S6/S7 all switch through `core/planners/hysteresis.py::should_switch()` with a per-planner margin, instead of each planner implementing its own ad hoc thrash guard.
- **Drift is engineered, not assumed**: both regression and CV get a purpose-built synthetic-drift variant for controlled, reproducible evaluation, in the same spirit as Augur's "Drifter" step.
- **Plug-and-play contract**: a dataset is a JSON config + a preprocessed manifest; `core/dataset_validator.py` enforces the contract at startup.

Full change history, including every deviation from the original papers' published behaviour and its rationale, lives in `context/CHANGES_FROM_PAPER.md`.

---

## Testing

```bash
cd tool/
python -m pytest -q
```

436 tests pass (5 skipped — platform-gated, e.g. RAPL/Flask-dependent). Coverage spans: energy
metering (pyJoules backends + null fallback), scoring formulas, all 7 drift detectors, TENT,
all 9 planners, VMR, task adapters (detection/classification/segmentation), the offline
true-accuracy evaluators, dataset preprocessing/trimming, and the plug-and-play validator.

---

## Known limitations

- **ImageNet/ImageNet-C weight bootstrap needs `torchvision` + network access** — not available
  in every dev environment; `run_cv.sh`'s weight-sentinel check skips these datasets gracefully
  when unmet rather than failing.
- **DP20 — CV `drift_detector` config key currently has no effect**: the CV harness hardcodes a
  luminance KL detector for every CV dataset regardless of what a config specifies. The other
  detectors exist and are tested but have never been invoked by an actual CV run.
- **iWildCam excluded from the active CV grid**: its model weights were never produced; ImageNet
  has taken its place as the fourth CV trial dataset. iWildCam's config and code remain in place.
- **TENT is episodic-per-retrain-event, not continuous per-step online adaptation** — see
  `context/DECISIONS_PENDING.md` DP25 for the scope reasoning.
- **Thresholds require calibration**: `E_M` and `tau_drift` should be calibrated per hardware and
  dataset via `experiments/calibrate_drift_threshold.py` before treating results as paper-quality.
- **The live system's `managed_system_cv/retrain_tactics/` remain YOLO-specific**, not yet
  unified with the task-agnostic adapter layer the experiment harness uses (DP19, open).

Full list of open decisions: `context/DECISIONS_PENDING.md`.

---

## Papers & citation

This repository extends the approach introduced in the HarmonE paper (ECSA 2025), and builds
on Harmonica, the original reference implementation of that approach (2026).

```
[CITATION STUB — fill in with the actual paper references]
```

---

## License

Not yet finalised — a `LICENSE` file will be added before public release.

## Research questions

- **RQ1** Does the extended framework generalise the paper's regression results to CV tasks with proxy accuracy signals, without regressing the paper's numbers on PeMS?
- **RQ2** Do embedding drift detectors correctly identify semantic drift that luminance-KL misses (iWildCam geographic shift as the primary test)?
- **RQ3** How well do the three CV proxies (confidence, calibrated_confidence, agreement) track ground-truth accuracy? Measured as Spearman ρ over monitoring intervals.
- **RQ4** How much of the offline-oracle upper bound does HarmonE capture, and how does that vary by planner choice?
- **RQ5** What is the actual energy overhead of MAPE-K, including the pinned embedding model's extra forward pass?
- **RQ6** How does HarmonE behave on a structural-break dataset (spot prices) where drift is permanent and VMR reuse cannot help? Stress test where partial failure is the expected outcome.

## Datasets

Six datasets total. All documented as expected-format specs in `docs/datasets/` with matching config skeletons in `configs/datasets/`. No downloads for the prototype sprint. Skeletons are marked `"status": "awaiting_data"` and the validator reports SKIPPED (not FAIL) until data arrives.

### Regression

| ID | Name | Task | Drift character | Models |
| --- | --- | --- | --- | --- |
| R1 | pems | Traffic flow forecasting | Recurring daily + weekly | LSTM, Ridge, SVR |
| R2 | uci_electricity | Load forecasting | Seasonal recurring (VMR reuse showcase) | LSTM, Ridge, SVR |
| R3 | spot_prices | Day-ahead price forecasting | Structural break, no revert (stress test) | LSTM, Ridge, SVR |

R2 handling notes. Portugal DST duplicate-timestamp quirk in preprocessing, kW-per-15-min unit conversion documented. R3 licensing. 

### CV

| ID | Name | Task | Drift character | Models |
| --- | --- | --- | --- | --- |
| C1 | bdd100k | Object detection | Clear → overcast → dusk → night → rain (luminance-visible) | YOLOv8 n / s / m |
| C2 | iwildcam | Classification | Geographic (location clusters, invisible to luminance) | EfficientNet-B0, ResNet-50, ResNet-101 |
| C3 | acdc | Semantic segmentation | Clear → fog → rain → night → snow (mixed) | SegFormer-B0, SegFormer-B1, SegFormer-B2 |

C2 is the embedding-drift motivation dataset. C3 uses `ignore_index: 255` in mIoU computation. DeepLab variants for C3 are deferred (DP9).

## Baselines

- **Paper baseline (I8)** `harmone_original` planner plus `confidence` proxy plus `luminance_kl` detector plus published thresholds. Reproducibility check at every phase.

> **Current active dataset**: `pems` (31 days of flow data; pre-split into `flow_data_train.csv` + `flow_data_test.csv` at `data/pems/`). `pems_node1` data not yet downloaded; `pems_node1.json` kept as a placeholder for future lab-machine use.
- **S1 naive** always noop, lower bound. Run once per model via `--pin-model` (regression: lstm/ridge/svr; bdd100k: yolo_n/yolo_s/yolo_m; iwildcam: efficientnet_b0/resnet50/resnet101; acdc: segformer_b0/segformer_b1/segformer_b2). Each pinned naive run is a separate row in the results table showing the fixed-model baseline for that model.
- **S2 random_switch** ablation lower bound
- **S3 greedy_switch** deterministic exploitation, no exploration
- **S5 violation_aware** and **S6 pareto** ablation variants of HarmonE
- **S7 bandit** LinUCB contextual bandit; accumulates learning across runs; regression only this cycle (CV wiring deferred, DP11)
- **Offline oracle** retrospective planner with full label access, upper bound (methodology open, DP6)

Planners are selectable via the dashboard modal (D10). CV pareto, random_switch, and bandit fall back to harmone_original at runtime pending the CV dispatch refactor (DP11), so CV planner ablations have three behaviourally-distinct planners this cycle. Regression has all seven.

## Metrics

- Task-native accuracy. RMSE and MAE for regression. mAP@0.5 for detection. Top-1 for classification. mIoU for segmentation.
- `S_i` timeline (composite HarmonE score per model, per interval)
- Energy per prediction in Joules, split cpu vs gpu, with `cpu_valid` and `gpu_valid` flags surfaced
- Switch count and event-type breakdown (`switch`, `noop`, `retrain`, `vmr`, `retrain_rejected`, `retrain_skipped_budget`, `ema_head_start`)
- MAPE-K overhead as fraction of total energy (this is where the pinned embedding model's extra forward pass shows up)
- Time-to-recover after each drift event (intervals from drift detected to `S_i` back above `S_min`)
- Fraction of offline-oracle score
- Spearman ρ between proxy and ground truth, per proxy per dataset

## Ablation dimensions

- **Planner** All 7 planners (S1–S7 fully implemented). Modal exposes all 6 (bandit added 2026-08-11; CV marks it unimplemented pending DP11).
- **Proxy** 3 (confidence, calibrated_confidence, agreement). CV only.
- **Drift detector** Regression = kl_fixed_ref vs kl_rolling. CV = luminance_kl vs mmd_embedding vs frechet_embedding.
- **Retrain tactic** retrain_oracle vs pseudo_label vs human_in_loop. CV only.
- **EMA head-start** on vs off (A2, `ema_head_start` config gate)
- **VMR match strategy** best_score vs closest_distribution

## Fixed vs rolling drift comparison (RQ2)

For each CV dataset, run once per drift-detector configuration.

1. `luminance_kl` (paper baseline)
2. `mmd_embedding` fixed reference (novel, primary)
3. `mmd_local` rolling reference (bug baseline, demonstrates B3 in embedding space)

Compare on time-to-first-detection and false-positive rate. iWildCam is the crisp test because luminance is roughly constant across locations but embedding distributions differ. Direct evidence for R3 (fixed reference) and for the two-axis taxonomy argument.

## CV offline ground-truth evaluation

Implemented as **Option 2 (replay)** in `experiments/offline_eval.py`.

**Approach**: after a CV run completes, `offline_eval.py` replays the recorded model-switch history from `predictions.csv` against ground-truth labels from the dataset manifest (`label_path` column).  No predictions are re-generated; each model is loaded at most once and cached for the full evaluation.

**Metric per task**:

| Dataset | Task | Metric | Method |
| --- | --- | --- | --- |
| bdd100k | detection | mAP@0.5 | `Ultralytics YOLO.val()` per monitoring interval |
| acdc | segmentation | mIoU | `SegmentationAdapter.offline_accuracy()` per image |
| iwildcam | classification | top-1 accuracy | `ClassificationAdapter.offline_accuracy()` per image |

**Output**: `{run_dir}/offline_eval.json` with per-interval and overall metric.

**Limitation**: uses original model weights from the dataset config.  Post-retrain weight updates in VMR are not replayed — steps where the model was retrained mid-run reflect pre-retrain weights.  A future extension (DP-offline-exact) could replay from VMR snapshots.

**Shell script**: `scripts/run_offline_eval.sh` iterates over all completed CV run directories and evaluates each one.  Skips runs without `run_manifest.json` (incomplete) and runs that already have `offline_eval.json` (already evaluated; delete to re-run).

```bash
# Evaluate all completed CV runs (seed=1)
bash scripts/run_offline_eval.sh

# Evaluate bdd100k only, seed 42
bash scripts/run_offline_eval.sh --dataset bdd100k --seed 42

# Smaller reporting window (500 steps per interval)
bash scripts/run_offline_eval.sh --interval 500
```

## Proxy validation (RQ3)

Two-script pipeline.

1. `experiments/offline_eval.py` computes ground-truth accuracy per monitoring interval (see section above).
2. `experiments/proxy_validation.py` computes Spearman ρ for each of the three proxies against ground truth. Supports `-smoke` for 3-interval sanity runs.

Output per dataset picks the winning proxy as the default for the main runs. Calibration method (Platt vs temperature scaling) selection for `calibrated_confidence` is still open (DP2).

## Oracle upper bound (RQ4)

Open decision (DP6). Three options.

1. Real mAP@0.5 (or task metric) against `offline_labels()` per interval. Cleanest but ties oracle to label availability.
2. Calibrated proxy score. Trivially achievable when the runtime system uses the same proxy.
3. Human-annotated subset. Highest quality but least scalable.

Leaning toward option 1 for CV, with a paper-level note that oracle = best planner picks given retrospective ground truth.

## Grid and reproducibility

- `configs/experiments/baseline.yaml` defines the base grid (used by `run_grid.py`)
- `scripts/run_regression.sh` and `scripts/run_cv.sh` drive the experiment grid directly: single seed (default 1, override with `--seed N`). Naive runs 3× per dataset (once per model via `--pin-model`); each adaptive planner runs once.
- Per-run reset via `experiments/run_reset.py` clears knowledge state, predictions.csv, VMR, embedding store between runs
- 5 seeds per configuration for the reproduction runs (A4, pending lab machine)
- Every commit that changes structure updates `files.md` and `CHANGES_FROM_PAPER.md` in the same commit
- No absolute paths, no lab-machine-only dependencies, requirements.txt pinned
- Plug-and-play contract enforced by 15 conformance tests plus `dataset_validator.py`

## Hardware setup

- Development laptop. AMD Ryzen AI 7 350 with RTX 5060.
- WSL for smoke tests. Energy backends null as expected. Full functional smoke including telemetry, predictions.csv growth, dashboard, clean shutdown.
# HarmonE Journal Extension — Implementation Status

Last updated: 2026-08-11 (S7 bandit implemented; 323 pass, 2 skip / 325 total)

## Phase Overview

| Phase | Name | Status |
|---|---|---|
| 1 | Codebase Repair (B1–B7) | ✅ IMPLEMENTED (A3 complete; A4 pending lab machine) |
| 2 | Pluggable Interfaces | ✅ IMPLEMENTED |
| 3 | CV Generalisation | ✅ IMPLEMENTED |
| 4 | Energy Instrumentation | ✅ IMPLEMENTED |
| 5 | Experiment Harness | ✅ IMPLEMENTED |
| 6 | Live-Run Bug Fixes (L1–L6, E1–E3) | ✅ IMPLEMENTED |
| 7 | Dashboard — Planner Selection Modal | ✅ IMPLEMENTED |
| 8 | Integration Hardening (CP7) | ✅ IMPLEMENTED |

---

## Phase 1 — Bug Fixes

### Bugs

| ID | Description | File(s) | Status | Test File |
|---|---|---|---|---|
| B1 | Dynamic energy threshold broken — formula grows monotonically | `analyse.py` (both domains), `core/scoring.py` | ✅ FIXED | `tests/test_b1_energy_threshold.py` |
| B2 | VMR reuse tactic unreachable — dict key contract mismatch | `managed_system_regression/mape_logic/analyse.py`, `plan.py` | ✅ FIXED | `tests/test_b2_vmr_reuse.py` |
| B3 | Drift reference is rolling, not training distribution | `monitor.py`, `core/drift/kl_fixed_ref.py`, `core/drift/kl_rolling.py` | ✅ FIXED | `tests/test_b3_drift_reference.py` |
| B4 | Model reloaded from disk every inference step | `managed_system_regression/inference.py` | ✅ FIXED | `tests/test_b4_model_cache.py` |
| B5 | Random KL placeholder before 2400 samples — can fabricate drift | `managed_system_regression/mape_logic/monitor.py` | ✅ FIXED | `tests/test_b5_kl_placeholder.py` |
| B6 | Switch counter increments on no-op — inflates metrics | `managed_system_regression/mape_logic/execute.py` | ✅ FIXED | `tests/test_b6_switch_counter.py` |
| B7 | Scaler data leakage — fitted on streaming data | `managed_system_regression/inference.py`, `retrain.py` | ✅ FIXED | `tests/test_b7_scaler_leakage.py` |

### Additional Items

| ID | Description | Status |
|---|---|---|
| A1 | Rename `plan_simple_switch` → `plan_random_switch`; add `plan_greedy_switch` | ✅ DONE |
| A2 | CV EMA inflation behind config `ema_head_start` | ✅ DONE |
| A3 | Git archaeology — record which bugs predate paper commits | ✅ DONE (all B-bugs confirmed to predate all tracked commits) |
| A4 | Reproduction run — 5 seeds on PeMS, compare vs paper values | ⬜ PENDING (needs lab machine) |

### Phase 1 Exit Criteria

- [x] All tests green (`pytest tool/tests/ -v`) — 323 pass, 2 skip (Flask/WSL); +30 bandit tests
- [x] Smoke test: `reg_harmone` runs end-to-end (WSL verified 2026-07-28)
- [x] `CHANGES_FROM_PAPER.md` covers B1–B7, A1–A4, and all subsequent phases
- [x] `DECISIONS_PENDING.md` captures all deferred choices

---

## Smoke Test Command

```bash
cd tool
python3 run_managed_system.py
# approach.conf must read: reg_harmone
# expect: telemetry pushed to app.py, no crashes, predictions.csv grows
```

## Known Gaps (not blocking current work)

| Item | Status | Blocks |
|---|---|---|
| A4 — PeMS 5-seed reproduction | ⬜ PENDING (needs Arch + energy hardware) | Baseline confidence |
| DP11 — CV planner dispatch not wired | ⬜ PENDING (~1 day) | CV planner comparison rows (bandit also excluded from CV) |
| `experiments/calibrate_drift_threshold.py` | ⬜ NOT IMPLEMENTED | Per-dataset tau_drift calibration (DP4) |
| `scripts/validate_manifests.py` | ⬜ NOT IMPLEMENTED | Bulk manifest validation across all 6 datasets |
| `experiments/proxy_validation.py` API | ⬜ MISMATCH | Checkpoint spec wants `--config` + per-frame CSV; current API is `--run` + Spearman-ρ summary |
| S7 bandit — CV wiring | ⬜ PENDING (DP11) | CV bandit comparison rows |

**Test policy:** No test should be tagged "pre-existing failure" and left red across phases. A failing test is either fixed or removed with an explicit decision recorded here. The 4 init_regression path failures (fixed 2026-07-31) should have been caught at Phase 1 exit. Going forward: any red test blocks the phase exit criteria.

---

## Dataset Onboarding Audit (checkpoint.md — 2026-08-05)

Audit performed against `context/checkpoint.md`. Checks that do not require dataset files are complete. Dataset-dependent checks (path validation, preprocessing runs, manifest spot-checks) are deferred until datasets are placed on disk.

### Config file status

| Config | File | Schema gaps vs checkpoint | Data path |
|---|---|---|---|
| R1 PeMS node2 | `configs/datasets/pems_node2.json` | `tau_drift_source` added ✅ | `data/pems_node2/pems_node2.csv` |
| R2 UCI Electricity | `configs/datasets/uci_electricity.json` | `tau_drift_source` added ✅ | `data/uci_electricity/uci_electricity.csv` |
| R3 ERCOT Spot Prices | `configs/datasets/spot_prices.json` | `tau_drift_source` added ✅; confirmed ERCOT (not Nord Pool) | `data/spot_prices/spot_prices.csv` |
| C1 BDD100K | `configs/datasets/bdd100k.json` | `image_dir` updated to `data/bdd100k/images`; `manifest_csv` set; `energy_required` fixed to `["gpu"]`; `status: awaiting_data` added ✅ | `data/bdd100k/` |
| C2 iWildCam | `configs/datasets/iwildcam.json` | `energy_required` fixed to `["gpu"]` ✅ | `data/iwildcam/` |
| C3 ACDC | `configs/datasets/acdc.json` | `energy_required` fixed to `["gpu"]` ✅ | `data/acdc/` |

**Remaining schema divergence from checkpoint spec (intentional — codebase schema takes precedence):**
- Checkpoint uses `dataset_id`, `task`, `model_spectrum` list, `input_window`, nested `thresholds{}`, `energy_domain`, `proxy_metric`; codebase uses `name`, `domain`, `models` dict, `seq_length`, flat thresholds, `energy_required`. The codebase schema is what the code/tests use — do NOT rewrite configs to the checkpoint schema.
- `raw_data_path` field not present in any config; not needed by any current code path.
- `sensor_id` field not present in regression configs; the meter/sensor selection is documented in `_comment` fields and preprocessing scripts.

### Dataset placement guide

When datasets are moved to the repository machine:

```
tool/data/pems_node2/
  pems_node2.csv              ← preprocessed CSV (timestamp, value columns)

tool/data/uci_electricity/
  uci_electricity.csv         ← preprocessed CSV (run preprocess_uci_electricity.py)
  [raw] LD2011_2014.txt       ← optional; only needed to re-run preprocessing

tool/data/spot_prices/
  spot_prices.csv             ← preprocessed CSV (run preprocess_spot_prices.py)
  [raw] ercot_dam_*.csv       ← optional; only needed to re-run preprocessing

tool/data/bdd100k/
  images/100k/train/*.jpg
  images/100k/val/*.jpg
  images/100k/test/*.jpg
  labels/bdd100k_labels_images_train.json
  labels/bdd100k_labels_images_val.json
  bdd100k_manifest.csv        ← generated by: python scripts/preprocess_bdd100k.py --bdd-root data/bdd100k/ --output data/bdd100k/bdd100k_manifest.csv

tool/data/iwildcam/           ← WILDS v2.0 directory structure
  metadata.csv
  categories.csv
  images/
  iwildcam_manifest.csv       ← generated by: python scripts/preprocess_iwildcam.py --wilds-root data/iwildcam/ --output data/iwildcam/iwildcam_manifest.csv

tool/data/acdc/
  rgb_anon/{fog,night,rain,snow}/{train,val,test}/{seq}/*.png
  gt_trainval/gt/{fog,night,rain,snow}/{train,val,test}/{seq}/*_gt_labelTrainIds.png
  acdc_manifest.csv           ← generated by: python scripts/preprocess_acdc.py --acdc-root data/acdc/ --output data/acdc/acdc_manifest.csv
```

### Preprocessing scripts (new — 2026-08-05)

| Script | Status | Handles |
|---|---|---|
| `scripts/preprocess_uci_electricity.py` | ✅ CREATED | DST dedup, leading-zero drop, meter selection (MT_168 default), kW unit |
| `scripts/preprocess_spot_prices.py` | ✅ CREATED | Auto-detects ERCOT vs Nord Pool headers; area/hub selection |
| `scripts/preprocess_bdd100k.py` | ✅ CREATED | BDD100K attribute → domain mapping; drift-ordered manifest |
| `scripts/preprocess_iwildcam.py` | ✅ CREATED | WILDS metadata.csv; location-ordered manifest; OOD split routing |
| `scripts/preprocess_acdc.py` | ✅ CREATED | rgb_anon/gt_trainval pairing; condition-ordered manifest |
| `scripts/preprocess_pems_node2.py` | ⬜ NOT NEEDED | PeMS node2 is same CSV format as node1 — run `init_regression.py --config pems_node2` directly after placing the CSV |

### Checks that require dataset presence

These cannot be completed until datasets are placed in `tool/data/`:

- Validate `data_path` values point to real files
- Run R3 `head -5` column inspection (verify ERCOT column names match preprocess_spot_prices.py assumptions)
- Execute all 5 preprocessing scripts and validate output manifests
- Run `scripts/validate_dataset.py` for each of the 6 configs
- Spot-check 50 random `input_path` rows in CV manifests
- Confirm `timestamp` monotonically increasing and no NaN/Inf in regression outputs
- Confirm `domain` values are from expected sets in CV manifests
- Run regression smoke test with new datasets (init_regression → run_managed_system)

### MAPE-K contamination check

**PASS** — grep of all 8 MAPE-K files (`managed_system_{regression,cv}/mape_logic/*.py`) finds zero matches for: `bdd100k`, `pems`, `electricity`, `nord`, `ercot`, `iwildcam`, `acdc`, hardcoded `/home/` paths, hardcoded threshold numerics outside of thresholds.json loading.

### Bug fix verification (section 4)

All B1–B7 and L1–L6 confirmed fixed (code inspection):
- B1: `update_energy_threshold()` with signed delta in `analyse.py:56-62` ✅
- B2: `{"action": ..., "version": ..., "drift_detected": ...}` contract in `analyse.py:155` ✅
- B5: `{"kl_div": None}` during warmup in `monitor.py` ✅
- L2: `_clear_stale_command()` called at startup in `manage.py:84` ✅
- L4: None-safe boundary checks (confirmed by `test_live_run_fixes.py` passing)

### Energy module status

- **Intentional divergence from checkpoint spec:** checkpoint expects `core/energy/context.py` + `core/energy/gpu_polling.py`; actual implementation is `core/energy.py` (single file). This was an explicit decision in `execution.md §2 R1` — do NOT create the directory structure.
- API name differs: `EnergyMeter` not `EnergyContext`. CV inference must use `EnergyMeter(label, backend="nvml")` (which falls back to polling on RTX 5060).
- `get_backend_status()` is exported from `core/energy.py` ✅
- 50ms `_PollingGPUBackend` present for RTX 5060 Blackwell fallback ✅

---

## Key File Locations

| Purpose | Path |
|---|---|
| Active approach | `tool/approach.conf` |
| Regression knowledge | `tool/managed_system_regression/knowledge/` |
| CV knowledge | `tool/managed_system_cv/knowledge/` |
| Core library | `tool/core/` |
| Tests | `tool/tests/` |
| Policy files | `tool/policies/` |
| Configs (new) | `tool/configs/` |

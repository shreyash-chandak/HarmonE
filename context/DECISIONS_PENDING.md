# Decisions Pending

Unresolved decisions that require dataset knowledge or advisor sign-off.
Both implementation options are stubbed/selectable via config unless noted.

**Sprint status (July 2026):** Items DP1–DP6 are from the original planning phase.
Items marked ✅ DECIDED have been resolved in the prototype sprint. New P2 deferrals
(from `execution.md §0.1`) are listed at the bottom as DP7–DP9.

---

## ✅ DECIDED — DP-A: Dataset specs for prototype sprint

**Decided (July 2026):** Six datasets selected for the prototype sprint documentation
(no download required for the sprint itself):
- R: pems_node2, uci_electricity (seasonal VMR reuse), spot_prices (structural break stress test)
- CV: bdd100k (detection), iwildcam (classification, embedding drift motivation), acdc (segmentation)

Config skeletons with `"status": "awaiting_data"` live in `configs/datasets/`.
Preprocessing recipes live in `tool/docs/datasets/`.

## ✅ DECIDED — DP-B: Fixed vs. rolling drift reference

**Decided (July 2026, execution.md §2 R3):** Fixed reference is the primary signal.
Rolling reference is an optional secondary signal (emitted only when
`emit_local_drift: true` in config) and must NOT be the trigger for MAPE action.

Rationale: rolling reference allows the detector to track gradual drift and go blind
(B3 analogue in CV). Fixed reference captures the training-time distribution and
remains sensitive to persistent shifts. `init_cv.py --skip-embeddings` is the escape
hatch for WSL/no-weights runs.

## ✅ DECIDED — DP-C: Pinned embedding model (R4)

**Decided (July 2026):** `embedding_model` config key pins ONE model for drift
embedding extraction, regardless of which model is currently serving inference.
Cross-model embedding spaces are incomparable; the fixed pinned model is the reference
frame for all embedding drift signals.

---

## DP1 — Dataset Selection

**Decision needed:** Which datasets to use for the journal extension (beyond bundled PeMS node 1 and BDD100K).

**Options to consider:**
- Regression: PEMS-BAY (multi-sensor), METR-LA, electricity consumption (UCR archive)
- CV: nuScenes (self-driving), VisDrone (aerial), KITTI (autonomous driving)

**Config impact:** New `configs/datasets/<name>.json` per dataset. No code changes needed
once adapter interfaces (Phase 2) are in place.

**Implement now:** `configs/datasets/_template.json` documents all required fields.
Running the tool on a new dataset requires only filling the template.

---

## DP2 — CV Proxy Calibration Method

**Decision needed:** Whether to use temperature scaling (Guo et al. 2017) via NLL minimisation
on detection logits, or Platt scaling on confidence-vs-correctness at IoU 0.5.

**Why this matters:** Ultralytics YOLO may not expose raw pre-softmax logits cleanly
for temperature scaling. If logit access is too intrusive, Platt scaling on the
confidence-vs-IoU alignment is more practical but less principled.

**Both options implemented** in `core/proxies/calibrated_confidence.py`. Config key
`"calibration_method": "temperature" | "platt"` selects at runtime.

**Sign-off needed on:** which method is defensible as the primary for the paper.

---

## ✅ DECIDED — DP3 / DP8: LinUCB Bandit Planner (S7)

**Decided (2026-08-11):** LinUCB contextual bandit implemented in `core/planners/bandit.py`.

Key decisions:
- **Algorithm**: LinUCB with fixed alpha (not decaying). Fixed alpha chosen because sample
  count per run (~12 decisions) is too low for meaningful decay; fixed alpha provides
  LinUCB's theoretical regret guarantees.
- **Context**: 10 + 2×|models| dimensional vector. Features: violation type (one-hot 3),
  current EMA, normalised energy, drift signal, EMA slope, steps-since-switch, retrain
  count, VMR size, per-model EMA accuracy, per-model EMA energy.
- **Reward**: sustainability gain (weighted accuracy improvement + energy reduction) per
  joule of switching cost. Delayed by one monitoring interval.
- **State persistence**: `knowledge/bandit_state.json` keyed by dataset_id. NOT reset
  between runs. `run_reset.py` preserves it (bandit_pending.json IS cleared on reset).
- **Constructor injection**: module-level `_bandit_instance` in `bandit.py`; `manage.py`
  calls `set_bandit_instance()` at startup; `get_planner("bandit")` reads it.
- **plan() delegation**: `plan_mape()` detects `planner=="bandit"` and calls `dispatch_plan()`.
- **Sensitivity analysis**: alpha ∈ {0.1, 0.5, 1.0} to be run in experiments.
- **CV domain**: bandit NOT wired in CV domain (DP11 still pending).
- **Dashboard**: "LinUCB Bandit (S7)" added as sixth option; CV_UNIMPLEMENTED_PLANNERS.
- **API**: `/api/set-planner` now accepts "bandit" (previously HTTP 400).
- **Tests**: 30 new tests in `tests/test_bandit_planner.py`; total suite 323 pass.

---

## DP4 — Drift Threshold Calibration

**Decision needed:** Are `tau_drift` thresholds calibrated per dataset (empirically, via
`experiments/calibrate_drift_threshold.py`) or set globally from the existing default (0.5)?

**Current state:** Default `tau_drift = 0.5` (regression), `tau_drift = 0.07` (CV).
`experiments/calibrate_drift_threshold.py` is planned but **not yet implemented** — it would
compute p99 of the null distribution for a given dataset+detector.

**Issue:** Using the bundled PeMS tau_drift for a new dataset may give very different
false-positive rates. Calibration is the right approach but requires held-out in-distribution
windows per dataset.

---

## DP5 — Regression Train/Test Split Boundaries

**Decision needed:** How to split the bundled PeMS `dataset.csv` into train and test
splits for initial scaler fitting and reference distribution computation.

**Current default (Phase 1):** First 80% of rows = train, remainder = test (chronological).
This is implemented in `scripts/init_regression.py`. The exact split index is a config
field (`train_frac`, default 0.8) in `configs/datasets/pems_node1.json`.

**Sign-off needed on:** whether the paper's original split was 80/20 or something else.
Check the original experiment setup before finalising.

✅ **Partially resolved:** `train_frac: 0.8` is the default. Config-level `train_end_index`
and `val_end_index` override for exact splits. The init script uses this to fit the scaler.

---

## DP6 — Oracle Upper Bound Methodology

**Decision needed:** For the offline oracle planner (Phase 5), how to define "true
accuracy" for the CV domain in the absence of real-time labels.

**Options:**
- mAP@0.5 computed against the adapter's `offline_labels()` per monitoring interval.
- Calibrated proxy score (uses the same proxy as the runtime system, no extra labels).
- Human-annotated subset only.

**Impact:** The oracle row in the results table (upper bound) depends on this choice.
Using proxy for oracle and proxy for monitoring makes the baseline trivially achievable,
which inflates the claimed fraction-of-oracle metric.

---

## P2 Deferrals (execution.md §0.1 — explicitly deferred, do not build)

### DP7 — RT2: Contrastive fine-tuning retrain tactic

**Status:** Deferred (P2). `retrain_tactics/` directory exists with a stub.
`execution.md §0.1` explicitly lists RT2 as P2. Do not build until paper scope is
confirmed. Document as deferred in any paper draft table of contributions.

**Options when revisited:**
1. SimCLR/MoCo-style contrastive loss on unlabelled retrain window
2. SupCon if pseudo-labels are available from the pseudo_label tactic

### ✅ DP8 — S7: Bandit exploration planner (LinUCB)

**Status:** IMPLEMENTED (2026-08-11). See DP3 above for full decision record.
`core/planners/bandit.py` contains complete LinUCBBandit and BanditPlanner.
30 tests pass. Wired into regression manage.py and plan.py.

**Options when revisited:**
1. LinUCB with context = [EMA slope, drift signal, steps-since-switch, energy budget]
2. Thompson sampling over Beta(win, loss) per model

### DP10 — reg_random_switch standalone baseline routing (CP7 G12/DP6)

**Status:** Partially wired (CP7). The `reg_random_switch` preset can be invoked two ways: (1) via the HarmonE planner modal with planner=random_switch → uses `reg_harmone_score` policy + `execute_mape_plan` tactic; (2) via the policy dropdown → preset defines tactic `random_switch`, which now has a handler (G12b fix), but the approach still maps to `reg_harmone`, so the prefix scan finds `reg_harmone_score.json` rather than an isolated baseline file.

**Decision needed:** Introduce a dedicated approach token `reg_random` (new `approach.conf` value, new policy file `reg_random_switch_r2.json`) to give the standalone random-switch baseline its own isolated approach path. Currently the tactic routing works, but the approach mapping is shared with HarmonE.

### DP11 — CV planner registry / dispatch_plan (CP7 G_CV_PLAN)

**Status:** Documented limitation (CP7). CV domain calls `plan_mape()` / `execute_mape()` directly in `execute.py`, bypassing `dispatch_plan()` and the planner registry in `core/planners/`. `thresholds.json["planner"]` is written by `/api/set-planner` for CV but is never read at runtime.

**Decision needed:** Refactor CV `execute.py` to call `dispatch_plan()` (mirrors regression architecture from Phase 2). Requires: wiring `PlanningContext` in CV `plan.py`, routing all five modal planners through `get_planner()`, and verifying that `cv_greedy_switch` / `cv_violation_aware` / `cv_pareto` / `cv_random_switch` functions in CV `plan.py` return `Decision` objects compatible with `dispatch_plan()`'s return contract.

**Cost:** ~1 day. Not blocking current experiments (CV always runs `harmone_original` logic regardless of modal selection).

### DP12 — proxy_validation.py API alignment

**Status:** Diverges from checkpoint spec (2026-08-05). Current `experiments/proxy_validation.py` accepts `--run <run_dir>` and produces a Spearman ρ summary for BDD100K/confidence/calibrated/agreement proxies. Checkpoint spec wants `--config configs/datasets/bdd100k.json --max-frames 100`, per-frame iteration in drift order, and a CSV output with columns `interval, confidence_proxy, true_accuracy, model, dataset, condition`.

**Decision needed:** Extend or replace the current script to also support the config-driven per-frame API. The safest path is adding a second entrypoint (`validate_per_frame()`) rather than rewriting the existing Spearman pipeline.

**Cost:** ~0.5 day. Unblocks: the proxy validation smoke test from checkpoint §7.

### DP13 — validate_manifests.py (bulk dataset validator)

**Status:** `scripts/validate_dataset.py` validates one dataset config at a time. Checkpoint §7 smoke test requires `scripts/validate_manifests.py` that loops all 6 dataset configs in one invocation and produces a PASS/FAIL table. Current workaround: run `validate_dataset.py` six times.

**Cost:** ~1 hour. Unblocks: checkpoint §7 smoke test step 5.

### DP9 — DeepLab variants for ACDC segmentation

**Status:** Deferred (P2). ACDC config uses SegFormer-B0 and SegFormer-B2 only.
DeepLab-v3+ and DeepLab-v3 are mentioned in `cv_guide.md §4` as options.
These require different adapter code and are excluded from the prototype evaluation.

**Options when revisited:**
1. `adapters/tasks/segmentation.py` already supports HuggingFace SegFormer; add DeepLab
   adapter using `torchvision.models.segmentation.deeplabv3_resnet50`
2. Adapter follows the same `load_model / infer / extract_proxy / extract_embedding / offline_accuracy` interface

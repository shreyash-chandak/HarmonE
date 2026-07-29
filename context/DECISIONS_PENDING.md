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

## DP3 — Bandit Planner (S7)

**Decision needed:** Whether LinUCB bandit planner is fully implemented or left as a
documented stub for the paper deadline.

**Current state:** `core/planners/bandit.py` is a documented stub raising
`NotImplementedError` with the full algorithm specification as a comment.

**Cost to implement:** ~2 days engineering. Requires: feature vector definition
(EMA slope, drift signal, steps-since-switch, remaining energy budget), reward
definition (ΔS_i per Joule of switching cost), contextual bandit training loop.

---

## DP4 — Drift Threshold Calibration

**Decision needed:** Are `tau_drift` thresholds calibrated per dataset (empirically, via
`experiments/calibrate_drift_threshold.py`) or set globally from the existing default (0.5)?

**Current state:** Default `tau_drift = 0.5` (regression), `tau_drift = 0.07` (CV).
`experiments/calibrate_drift_threshold.py` (Phase 3) computes p99 of the null distribution
for a given dataset+detector.

**Issue:** Using the bundled PeMS tau_drift for a new dataset may give very different
false-positive rates. Calibration is the right approach but requires held-out in-distribution
windows per dataset.

---

## DP5 — Regression Train/Test Split Boundaries

**Decision needed:** How to split the bundled PeMS `dataset.csv` into train and test
splits for initial scaler fitting and reference distribution computation.

**Current default (Phase 1):** First 80% of rows = train, remainder = test (chronological).
This is implemented in `scripts/init_scaler.py`. The exact split index should be a config
field in `configs/datasets/pems_node1.json` once that file exists.

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

### DP8 — S7: Bandit exploration planner (LinUCB)

**Status:** Deferred (P2). `core/planners/bandit.py` is a documented stub raising
`NotImplementedError`. Feature vector and reward definition (see DP3) require empirical
tuning. Estimated cost: ~2 days engineering. Not included in the prototype evaluation.

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

### DP9 — DeepLab variants for ACDC segmentation

**Status:** Deferred (P2). ACDC config uses SegFormer-B0 and SegFormer-B2 only.
DeepLab-v3+ and DeepLab-v3 are mentioned in `cv_guide.md §4` as options.
These require different adapter code and are excluded from the prototype evaluation.

**Options when revisited:**
1. `adapters/tasks/segmentation.py` already supports HuggingFace SegFormer; add DeepLab
   adapter using `torchvision.models.segmentation.deeplabv3_resnet50`
2. Adapter follows the same `load_model / infer / extract_proxy / extract_embedding / offline_accuracy` interface

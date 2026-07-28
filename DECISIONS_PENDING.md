# Decisions Pending

Unresolved decisions that require dataset knowledge or advisor sign-off.
Both implementation options are stubbed/selectable via config unless noted.

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

# HarmonE Journal Extension — Implementation Plan for Coding Agent

> **Audience:** Claude Sonnet 4.6 acting as an autonomous coding agent inside the
> `HarmonE-tool` repository.
> **Goal:** Transform the current Harmonica tool into the generalised, bug-fixed,
> dataset-agnostic experimentation platform required for the journal extension.
> **Datasets are NOT chosen yet.** All code must be dataset-agnostic behind adapter
> interfaces and config files. Nothing may hardcode a dataset path, column name,
> class list, or drift parameter outside a config/adapter.
> **Reference documents:** `documentation.md` (current tool state) and the research
> context document. Read both before writing any code.

---

## 0. Ground Rules for the Agent (READ FIRST, APPLY ALWAYS)

1. **Never break the running tool.** After every phase, `reg_harmone` mode must
   still run end-to-end on the bundled PeMS `dataset.csv`. Treat this as the
   permanent smoke test.
2. **One phase = one branch = one PR-sized change.** Branch names:
   `fix/bugs-phase1`, `refactor/planner-interface`, `feat/cv-generalisation`,
   `feat/energy-gpu`, `feat/experiment-harness`, `chore/repo-hygiene`.
   Commit messages: imperative mood, reference the bug/feature ID from this plan
   (e.g. `fix: B1 dynamic energy threshold uses E_ref and delta (Eq.3)`).
3. **Every bug fix gets a regression test** in a new `tests/` directory (pytest).
   Tests must fail on the old code path and pass on the new one. Where the old
   behavior can't be imported cleanly, test the extracted pure function.
4. **Extract logic into pure functions before fixing it.** Much of the MAPE logic
   reads/writes files inside the same function. Refactor pattern: pure function
   `compute_x(inputs) -> outputs` + thin I/O wrapper. Test the pure function.
5. **No behavioral change without a config switch.** Old behavior must remain
   reachable via config (e.g. `"drift_reference": "rolling" | "fixed"`), because
   the paper needs pre-fix vs post-fix comparison numbers.
6. **Dataset-agnostic means:** every dataset-specific value (paths, column names,
   window sizes, thresholds, model lists, label availability) lives in a per-dataset
   config JSON under `configs/datasets/`. Code reads only from configs and adapter
   objects. When a dataset is unknown, ship a `configs/datasets/_template.json`
   with every field documented.
7. **Do not delete existing code paths** (luminance drift, rolling KL, random
   switch). Rename them honestly, keep them selectable, because they are baselines
   in the paper.
8. **Windows/Linux:** repo lives at `D:/Desktop/HarmonE-tool` but energy tooling
   (pyRAPL, NVML) is Linux-oriented. All energy measurement must go through the
   new `EnergyMeter` abstraction (Phase 4) which degrades gracefully (returns
   `None`, never fake zeros silently — log a warning and mark the run
   `energy_valid: false`).
9. **Ask via TODO, don't guess.** If a decision requires dataset knowledge or
   advisor sign-off, implement both options behind config and add a line to
   `DECISIONS_PENDING.md` at repo root.
10. **After each phase, update `CHANGES_FROM_PAPER.md`** with what changed and why.

---

## 1. Target Architecture (What the Repo Looks Like When Done)

```
HarmonE-tool/
├── tool/
│   ├── app.py                          # unchanged interface; internals cleaned
│   ├── run_managed_system.py
│   ├── configs/
│   │   ├── datasets/
│   │   │   ├── _template.json          # fully documented template
│   │   │   ├── pems_node1.json         # existing bundled dataset (R-baseline)
│   │   │   └── ...                     # filled in later when datasets chosen
│   │   └── experiments/
│   │       └── _template.yaml          # experiment grid definition
│   ├── core/                           # NEW: shared, domain-agnostic library
│   │   ├── energy.py                   # EnergyMeter abstraction (RAPL+NVML)
│   │   ├── drift/
│   │   │   ├── base.py                 # DriftDetector interface
│   │   │   ├── kl_fixed_ref.py         # fixed training-reference KL
│   │   │   ├── kl_rolling.py           # legacy rolling KL (renamed, kept)
│   │   │   ├── luminance_kl.py         # legacy CV luminance KL (kept)
│   │   │   ├── mmd_embedding.py        # MMD w/ RBF kernel on embeddings
│   │   │   └── frechet_embedding.py    # Fréchet distance on feature stats
│   │   ├── planners/
│   │   │   ├── base.py                 # Planner interface
│   │   │   ├── naive.py                # S1 static single model
│   │   │   ├── random_switch.py        # S2 honest rename of current Switch
│   │   │   ├── greedy_switch.py        # S3 highest-EMA switch
│   │   │   ├── harmone_original.py     # S4 bug-fixed original
│   │   │   ├── violation_aware.py      # S5
│   │   │   ├── pareto.py               # S6 primary novel contribution
│   │   │   └── bandit.py               # S7 optional, stub allowed
│   │   ├── proxies/
│   │   │   ├── base.py                 # AccuracyProxy interface
│   │   │   ├── confidence.py           # raw mean confidence
│   │   │   ├── calibrated_confidence.py# temperature-scaled
│   │   │   └── agreement.py            # multi-model agreement
│   │   ├── vmr.py                      # VMR store/match, pluggable signature
│   │   └── scoring.py                  # S_i, EMA, dynamic threshold (Eq.3)
│   ├── adapters/                       # NEW: per-domain dataset adapters
│   │   ├── base.py                     # DatasetAdapter interface
│   │   ├── regression_csv.py           # generic sliding-window CSV adapter
│   │   └── cv_imagedir.py              # generic image-directory adapter
│   ├── experiments/                    # NEW: harness
│   │   ├── run_experiment.py           # single (dataset × planner × seed) run
│   │   ├── run_grid.py                 # full baseline grid driver
│   │   ├── metrics.py                  # R²/mAP, energy, Pareto efficiency, ρ
│   │   └── offline_eval.py             # offline mAP / label-based scoring (CV)
│   ├── managed_system_regression/      # kept, refactored to call core/
│   ├── managed_system_cv/              # kept, refactored to call core/
│   └── tests/
├── CHANGES_FROM_PAPER.md
├── DECISIONS_PENDING.md
└── reproduce/                          # scripts per RQ (Phase 6)
```

Key principle: `managed_system_*` become thin domain shells; all intelligence
moves into `core/` where it is testable and shared. The Flask/ACP plumbing,
dashboard, and command.txt mechanism stay as-is unless a fix requires touching
them.

---

## 2. Phase 1 — Codebase Repair (Bugs B1–B7)

All file paths relative to `tool/`. Fix in this order. Each fix: extract pure
function → write failing test → fix → test passes → smoke test.

### B1 — Dynamic energy threshold broken (regression) [CRITICAL]

**File:** `managed_system_regression/mape_logic/analyse.py`

Current (broken — threshold only grows, energy violation can never fire because
`max_energy = 1` and `used_energy ∈ [0,1]`):

```python
new_energy_threshold = current_energy_threshold + 0.95 * (original_energy_threshold - used_energy)
```

Replace with the paper's Eq. 3, extracted into `core/scoring.py`:

```python
def update_energy_threshold(current: float, e_ref: float, e_used: float,
                            delta: float, lo: float = 0.1, hi: float = 1.0) -> float:
    """tau_E(i+1) = tau_E(i) + delta * (E_ref - E_used), clamped to [lo, hi]."""
    return max(lo, min(hi, current + delta * (e_ref - e_used)))
```

**Config changes** in `managed_system_regression/knowledge/thresholds.json`:
- Add `"E_ref": 0.7` and `"delta": 0.1` (defaults; per-dataset configs override).
- Change `"max_energy"` from `1` to `0.6` so the boundary is meaningful.
- Apply the same `update_energy_threshold` in the CV `analyse.py` (replacing its
  0.4-factor formula) so both domains use one implementation.

**Tests:**
- Sustained `e_used > e_ref` → threshold strictly decreases toward `lo`.
- Sustained `e_used < e_ref` → threshold strictly increases toward `hi`.
- Threshold never leaves `[lo, hi]`.
- With old shipped config (`max_energy=1`, factor 0.95), demonstrate the old
  function is monotonically increasing (documents the bug).

### B2 — VMR reuse tactic unreachable (regression) [CRITICAL]

**Files:** `managed_system_regression/mape_logic/analyse.py`, `plan.py`

`analyse_drift()` returns `{"drift_detected": ..., "best_version": ...}` but
`plan_drift()` reads keys `action` and `version` → every drift falls through to
retrain; VMR reuse can never fire.

**Fix:** align regression `analyse_drift()` with the (correct) CV contract:

```python
# analyse_drift() return contract (both domains, document in core/vmr.py):
{"drift_detected": True,  "action": "replace", "version": "<path>"}   # VMR hit
{"drift_detected": True,  "action": "retrain", "version": None}       # no match
{"drift_detected": False, "action": None,      "version": None}
```

**Test:** stub `get_best_version` to return a path → `plan_drift()` must return
the replace action with that path; stub it to return `None` → retrain action.

### B3 — Drift reference is rolling, not training distribution [METHODOLOGY]

**File:** `managed_system_regression/mape_logic/monitor.py`

**Fix:**
1. At train time (`retrain.py` and initial setup), persist the training-data
   value histogram to `knowledge/reference_distribution.json`.
2. Implement `core/drift/kl_fixed_ref.py`: KL(current_window ‖ fixed P_r).
3. Move the existing adjacent-window logic to `core/drift/kl_rolling.py`,
   rename its output key `kl_local`.
4. Config key `"drift_reference": "fixed" | "rolling" | "both"` (default
   `"both"`: report both signals, trigger on the fixed one). Both is the
   publishable comparison.

**Test:** synthetic slow linear drift over N windows — rolling KL stays below
threshold at every step while fixed-ref KL crosses it. This is the paper's
motivating example; keep the test data generator reusable for plots.

### B4 — Model reloaded from disk every inference step [MEASUREMENT]

**File:** `managed_system_regression/inference.py`

**Fix:** cache the loaded model in memory. Reload only when `model.csv` content
changes (compare the string read each iteration — the read is cheap, the
deserialisation is not) or when the executor touches a new
`knowledge/model_reload.flag` after a VMR replace/retrain writes new weights to
`models/`. Delete the flag after reload.

**Test:** monkeypatch `torch.load`/`pickle.load` with a counter; run 100 loop
iterations with no switch → loader called once; switch model at iteration 50 →
called twice.

### B5 — Random KL placeholder before 2400 samples [CORRECTNESS]

**File:** `managed_system_regression/mape_logic/monitor.py`

Current fallback `np.random.uniform(0.01, 0.15)` straddles the ACP secondary
boundary (0.1) → fabricated drift triggers during warmup.

**Fix:** return `{"kl_div": None}` on insufficient data. Everywhere `kl_div` is
consumed (`analyse_drift`, `app.py` secondary check, telemetry push,
dashboard): treat `None`/missing as "no signal", never as a violation. In
telemetry, omit the key rather than sending null if the dashboard chokes on it.

**Also:** unify the drift threshold. Local hardcoded 0.5 vs ACP policy 0.1 is a
conflict. Move the threshold into `thresholds.json` (`"tau_drift"`), read it in
both places, regenerate the policy JSONs from it. Add a startup assertion that
policy and thresholds agree, warn loudly if not.

**Test:** with 100 rows, `monitor_drift()` returns None-valued signal and
`analyse_drift()` returns `drift_detected: False` deterministically over 1000
calls (no randomness).

### B6 — Switch counter increments on no-op [MEASUREMENT]

**File:** `managed_system_regression/mape_logic/execute.py`

**Fix:** only call `record_event("switch", ...)` when `decision is not None`.
Record the no-op case as a new event type `"noop"` (energy still logged —
MAPE-K overhead is real) so overhead accounting stays complete.

**Test:** plan returns None → `model_switches` unchanged, `noop` count +1,
`mape_k_energy_uJ` increased.

### B7 — Scaler data leakage [METHODOLOGY]

**File:** `managed_system_regression/inference.py`

**Fix:** fit `MinMaxScaler` on training data only; persist to
`knowledge/scaler.pkl` at train time; `inference.py` loads and applies it,
never fits. `retrain.py` refits on its training window and overwrites the
persisted scaler atomically (write temp + rename) together with the new
reference distribution (B3).

**Test:** inference over data containing values outside the training range →
scaler params identical before/after; transformed values may exceed [0,1]
(that is correct behavior — clip only if the model requires it, and do the
clip explicitly with a comment).

### Additional Phase 1 items

- **A1 — Honest Switch rename:** rename `plan_simple_switch` to
  `plan_random_switch` everywhere (code, policies, dashboard labels). Add
  `plan_greedy_switch` (Strategy 3) as the fair informed baseline. Keep both.
- **A2 — CV EMA inflation flag:** the CV `execute_drift` inflates EMA by +0.1
  after a VMR deploy. Keep it but gate behind config
  `"ema_head_start": 0.1 | 0` and log the inflation as an event so it is
  visible in analysis.
- **A3 — git archaeology:** run `git log --follow` on the buggy files and record
  in `CHANGES_FROM_PAPER.md` whether each bug predates the paper's experiment
  commits (needed to phrase the reproduction section honestly).
- **A4 — Reproduction run:** after all fixes, run the original PeMS experiment
  in `harmone_original` planner mode, 5 runs, and record results vs paper
  (R²=0.8628, 20.62 mJ, 1.89 ms, 12 adaptations). Target ±5%; if outside,
  document the delta and the suspected cause (B4 and B6 will change numbers —
  that is expected and must be written up, not hidden).

**Phase 1 exit criteria:** all tests green; smoke test passes; reproduction run
logged; `CHANGES_FROM_PAPER.md` covers B1–B7 + A1–A4.

---

## 3. Phase 2 — Pluggable Interfaces (Planner, Drift, Proxy, Adapter)

This phase is pure refactor + new scaffolding. No new science yet.

### 3.1 Planner interface (`core/planners/base.py`)

```python
@dataclass
class PlanningContext:
    violation: str | None            # "score" | "energy" | "drift" | None
    ema_scores: dict[str, float]     # collapsed score per model (legacy)
    ema_accuracy: dict[str, float]   # separated signal (new, Phase 2.4)
    ema_energy: dict[str, float]     # separated signal (new, Phase 2.4)
    current_model: str
    available_models: list[str]
    thresholds: dict                 # full thresholds.json contents
    drift_result: dict | None        # analyse_drift contract from B2
    history: Any                     # accessor for telemetry history (bandit)

@dataclass
class PlanDecision:
    action: str                      # "switch" | "replace" | "retrain" | "noop"
    model: str | None = None
    version_path: str | None = None
    reason: str = ""                 # free text, logged to event log

class Planner(ABC):
    name: str
    @abstractmethod
    def plan(self, ctx: PlanningContext) -> PlanDecision: ...
```

Implementation notes:
- `plan.py` in each managed system becomes a dispatcher: build
  `PlanningContext` from knowledge files, instantiate the planner named in
  `thresholds.json["planner"]` (via a registry dict, no dynamic imports of
  arbitrary paths), translate `PlanDecision` back to the existing execute
  contract. `execute.py` behavior is unchanged.
- Policy JSONs gain `"planner": "<name>"`; dashboard presets updated to set it.
- Migrate existing logic: current ε-greedy → `harmone_original.py` (S4, uses
  the B1/B2-fixed analyse outputs); current random pick → `random_switch.py`
  (S2); `naive.py` (S1) always returns noop.

### 3.2 New planners

**S3 `greedy_switch.py`:** on any violation, switch to the non-current model
with highest `ema_scores`; noop if current is best. No ε exploration, no drift
tactics (drift violations also just trigger a greedy switch — that is the
honest "informed switch only" baseline).

**S5 `violation_aware.py`:**

```python
if ctx.violation == "energy":
    candidates = [m for m in others if ctx.ema_accuracy[m] >= thr["min_score"]]
    pick min ema_energy among candidates (fallback: min ema_energy overall)
elif ctx.violation == "score":
    candidates = [m for m in others if ctx.ema_energy[m] <= thr["current_energy_threshold"]]
    pick max ema_accuracy among candidates (fallback: max ema_accuracy overall)
elif ctx.violation == "drift":
    follow drift_result: replace if VMR hit else retrain
else: noop
```

**S6 `pareto.py` (primary novel contribution):**
- Requires separated EMAs (3.4 below).
- Build empirical Pareto front over `available_models` in
  (accuracy=ema_accuracy[m], energy=ema_energy[m]) space.
- Select the Pareto-optimal model minimising Chebyshev distance to the goal
  point `(S_min, E_ref)`:

```python
d(m) = max(w_acc * max(0, S_min - acc[m]), w_e * max(0, energy[m] - E_ref))
```

- `w_acc`, `w_e` from thresholds (default 1.0, 1.0). Note: use `max(0, ·)`
  so models exceeding the goal on an axis are not rewarded for overshoot;
  document this choice — it differs slightly from the sketch in the research
  doc and is deliberate. Tie-break: lower energy.
- Drift violations delegate to the same replace/retrain logic as S5.

**S7 `bandit.py` (optional):** implement LinUCB skeleton with context vector
[EMA slope over last k intervals, drift signal, steps-since-switch, remaining
energy budget], reward = ΔS_i per Joule of switching cost. Ship behind
`"planner": "bandit"` but mark experimental; a stub raising
`NotImplementedError` with a clear message is acceptable for the first pass —
record in `DECISIONS_PENDING.md`.

Also add **oracle switch** as an *experiment-harness-only* planner
(`experiments/` may import offline labels; `core/` planners may not): at each
interval pick the model with highest true offline accuracy. Used purely as the
upper bound row in the results grid.

**Tests:** table-driven per planner: given a fixed `PlanningContext`, assert
the exact `PlanDecision`. Pareto planner: hand-constructed 3-model fronts
covering (a) dominated model never chosen, (b) goal-point inside front,
(c) goal unreachable → nearest front point.

### 3.3 DriftDetector interface (`core/drift/base.py`)

```python
class DriftDetector(ABC):
    name: str
    def fit_reference(self, reference: Any) -> None: ...   # persist stats
    def score(self, window: Any) -> float | None: ...      # None = insufficient data
    def save(self, path) / load(path)                      # reference persistence
```

Wrap existing KL implementations (fixed, rolling, luminance) in this interface.
`monitor_drift()` becomes: instantiate detectors listed in
`thresholds.json["drift_detectors"]`, emit `{"<name>": score}` per detector,
plus the legacy `kl_div` key mapped from the configured primary detector (keeps
dashboard and policies working).

### 3.4 Separated EMA signals

**File:** `core/scoring.py` + both `monitor.py`s.

Alongside the legacy collapsed `score`, maintain:
- `ema_accuracy[model] = γ·A_i + (1−γ)·prev` (A_i = R² or proxy accuracy)
- `ema_energy[model] = γ·Ē_i + (1−γ)·prev`

Persist both in `mape_info.json` (new keys; keep `ema_scores` for backward
compat). All three updated only for the currently active model, same as today.

### 3.5 DatasetAdapter interface (`adapters/base.py`)

Because datasets are not chosen yet, this interface is the contract that makes
later integration a config-only task:

```python
class DatasetAdapter(ABC):
    """Constructed from configs/datasets/<name>.json only."""
    domain: str                      # "regression" | "cv"
    self_labeling: bool              # runtime ground truth available?
    def train_split(self) -> Any: ...
    def val_split(self) -> Any: ...
    def stream(self) -> Iterator[Sample]:  # chronological test stream
    def offline_labels(self) -> Any | None:  # for oracle/offline eval; None if absent
    def models(self) -> dict[str, ModelSpec]:  # name -> weights path, loader, cost class
```

Implement two generic adapters now:
- `regression_csv.py`: config specifies csv path, value column, window length,
  horizon, split boundaries (chronological indices/dates), model spec paths.
  Point it at the bundled PeMS csv via `configs/datasets/pems_node1.json` and
  make `managed_system_regression/inference.py` consume it (replacing its
  hardcoded loading, preserving the 0.15 s streaming sleep as a config value).
- `cv_imagedir.py`: config specifies image dir(s) with ordered file list or a
  manifest CSV (path + optional attributes + optional label path), model specs.
  Wire `managed_system_cv/inference.py` to it with a config for the bundled
  BDD100K test images.

`_template.json` must document every field with an inline `"_comment_*"` key.
Drift-induction transforms (scale/shift for regression, brightness/contrast for
CV) become adapter-level config-driven stream transforms (`"induced_drift"`
block), replacing the standalone `induce.py` coupling.

**Phase 2 exit criteria:** all planner/drift tests green; PeMS smoke test runs
through the adapter + dispatcher path with `planner: harmone_original` and
produces telemetry indistinguishable in schema from before; dashboard still
renders.

---

## 4. Phase 3 — CV Generalisation (Label-Free Loop)

The scientific core. CV has no runtime ground truth, so every mechanism that
consumed A_i needs a validated proxy, and pixel/luminance drift detection needs
an embedding-space replacement.

### 4.1 Embedding extraction (`core/drift/embeddings.py`)

- Detection models (YOLOv8): hook the backbone's final feature map before the
  detection head; spatial-average-pool to a D-dim vector per image. Implement
  via a forward hook registered on the loaded Ultralytics model; cache the
  hook per loaded model (interacts with B4's model cache — extend the cache to
  hold `(model, hook_handle)`).
- Classification models (future adapters): penultimate-layer activations.
- Store per-image embeddings in a ring buffer file
  `knowledge/embeddings.npy` (memory-mapped, fixed capacity = 2 ×
  `drift_window` from config) plus an index CSV mapping row → image name.
  Do NOT append embeddings to `predictions.csv`.

### 4.2 MMD drift detector (`core/drift/mmd_embedding.py`)

- Unbiased MMD² estimator, RBF kernel, bandwidth = median heuristic computed on
  the reference set at `fit_reference` time and frozen.
- `fit_reference`: embed a config-sized sample (default 500) of training
  images; persist embeddings + bandwidth to `versionedMR`-adjacent
  `knowledge/reference_embeddings.npz`.
- `score(window)`: MMD² between reference and last `drift_window` (default
  500) streamed embeddings; `None` if fewer available.
- Also implement `frechet_embedding.py` (mean+covariance, FID-style) as the
  cheap alternative; config selects.
- Keep `luminance_kl.py` selectable as legacy/secondary signal.
- Calibrate the trigger threshold empirically: provide
  `experiments/calibrate_drift_threshold.py` that runs the detector on
  held-out in-distribution windows and sets `tau_drift` at the p99 of the null
  distribution (config records the method used).

### 4.3 Accuracy proxies (`core/proxies/`)

```python
class AccuracyProxy(ABC):
    name: str
    def setup(self, models, val_split) -> None: ...   # e.g. fit temperature
    def score(self, inference_record) -> float: ...   # proxy A_i in [0,1]
```

- `confidence.py`: current behavior (mean box confidence / softmax max).
- `calibrated_confidence.py`: temperature scaling (Guo et al. 2017). `setup`
  fits scalar T per model on the val split by NLL minimisation (for detection,
  calibrate on box-classification logits; if Ultralytics logit access is too
  intrusive, fall back to Platt scaling on confidence-vs-correctness at IoU
  0.5 and document in `DECISIONS_PENDING.md`). Persist per-model T in
  `knowledge/calibration.json`.
- `agreement.py`: run the smallest other model on a config-fraction (default
  0.1) subsample of inputs; proxy = detection-level agreement (matched boxes
  IoU>0.5, same class) or class agreement. Log its extra energy under a
  dedicated event so overhead is accountable.
- `monitor.py` (CV) computes A_i via the configured proxy; `score` formula and
  EMAs unchanged otherwise.

### 4.4 Offline evaluation + proxy validation (`experiments/offline_eval.py`)

Strictly outside the runtime loop:
- Consumes the adapter's `offline_labels()`; computes true mAP@0.5 (detection)
  or accuracy (classification) per monitoring interval, aligned to the same
  interval boundaries the monitor used (persist interval boundaries to
  `knowledge/intervals.csv` from `monitor_mape` to make alignment exact).
- `experiments/proxy_validation.py`: given a completed run, produce per-interval
  table (proxy value, true metric, drift severity/attribute) and Spearman ρ
  overall and per drift condition, plus the proxy-vs-truth plot. This is RQ3's
  entire pipeline; make it a single command:
  `python experiments/proxy_validation.py --run <run_dir>`.
- Decision rule to encode in the paper workflow (document in README): ρ ≥ 0.8
  across conditions → calibrated confidence is the primary proxy; ρ collapses
  under drift → agreement proxy becomes primary. Both remain implemented.

### 4.5 Embedding-based VMR (`core/vmr.py`)

Unify both domains behind one store:

```
versionedMR/<model_name>/version_N/
    weights.(pt|pth|pkl)
    signature.npz        # regression: value histogram; CV: embedding mean+cov (+ optional sample)
    meta.json            # created_at, train_window_ref, drift_context, signature_type
```

- `VMR.store(model_name, weights, signature, meta)` /
  `VMR.match(current_signature, metric) -> (version_path, distance) | None`.
- Matching metric = same family as the active drift detector (KL for
  histograms, Fréchet/MMD for embeddings) — enforce signature_type
  compatibility, refuse cross-type matches.
- Migrate the CV flat-file layout (`yolo_s_v1.pt` + `_hist.json`) with a
  one-shot `scripts/migrate_vmr.py`; keep a reader shim for old layout for one
  release.
- Acceptance threshold `tau_vmr` per dataset config; calibrated by the same
  null-distribution script as 4.2.

### 4.6 Label-free retraining tactics (`managed_system_cv/retrain_tactics/`)

Replace oracle-coupled `retrain.py` (which reads ground-truth labels and
mirrors `induce.py` constants) with three selectable tactics, chosen by config
`"retrain_tactic"`:

1. `pseudo_label.py`: fine-tune detection head on recent frames using own
   high-confidence predictions (conf > `tau_pseudo`, default 0.6) as labels.
   Safety valve: after fine-tune, compare proxy score on a held-back recent
   window; if worse than pre-fine-tune, discard weights and fall back to a
   greedy switch (log event `retrain_rejected`).
2. `embedding_finetune.py`: contrastive/feature-alignment fine-tune of the
   backbone on unlabeled old+new frames (no task labels). Mark experimental;
   a documented stub is acceptable first pass → `DECISIONS_PENDING.md`.
3. `human_in_loop.py`: no training. Emit `labeling_required` event, switch to
   the lowest-energy model until an operator drops labels into a watched
   directory (`knowledge/incoming_labels/`), then run supervised fine-tune.
   This is the honest tactic and must be the documented default for
   deployments without labels.

Keep the old oracle retrain as `retrain_oracle.py`, selectable only when the
adapter reports `offline_labels()` — it becomes an upper-bound ablation, and
its augmentation parameters move from hardcoded constants into the dataset
config's `induced_drift` block (they must never silently mirror induction).

Energy accounting: every tactic runs under the EnergyMeter and logs
train-energy separately; the planner's drift branch must consult a config
`"retrain_energy_budget"` — if estimated cost (last known retrain energy)
exceeds remaining budget, prefer VMR/switch and log `retrain_skipped_budget`.

**Phase 3 exit criteria:** CV pipeline runs end-to-end on bundled BDD images
with `planner: harmone_original`, MMD drift, calibrated-confidence proxy, and
`human_in_loop` retrain tactic — zero reads of `labels/` anywhere under
`managed_system_cv/` at runtime (add a test that fails if any runtime module
opens a path containing `labels/`). Proxy-validation script produces ρ on a
small smoke run.

---

## 5. Phase 4 — Energy Instrumentation (`core/energy.py`)

```python
class EnergyMeter:
    """Context manager. Sums CPU (RAPL) + GPU (NVML) joules; each backend optional."""
    def __enter__ / __exit__
    @property result -> {"cpu_uJ": float|None, "gpu_uJ": float|None,
                         "total_uJ": float|None, "valid": bool, "backends": [...]}
```

- CPU backend: pyRAPL (existing). GPU backend: pynvml —
  `nvmlDeviceGetTotalEnergyConsumption` delta when supported (Volta+); else
  integrate `nvmlDeviceGetPowerUsage` samples on a 50 ms sampling thread and
  mark `"method": "power_integration"`.
- If a backend is unavailable: `None` for that component, `valid` reflects
  whether the run's configured requirement (`"energy_required": ["cpu"]` or
  `["cpu","gpu"]` per dataset config) was met. Never write 0 for missing.
- Replace every direct `pyRAPL.Measurement` call in both managed systems and
  MAPE logic with `EnergyMeter`.
- `E_m`/`E_M` normalisation bounds move into the dataset config and MUST be
  derived per hardware+dataset by `experiments/calibrate_energy_bounds.py`
  (runs each model over a pilot slice, records min/max interval energy).
  Never reuse regression bounds for CV.
- Results tables must never mix CPU-only and CPU+GPU totals in one column:
  `metrics.py` refuses to aggregate runs whose `backends` differ, unless
  `--force` with a warning.

---

## 6. Phase 5 — Experiment Harness

### 6.1 `experiments/run_experiment.py`

One invocation = one (dataset config × planner × drift detector × proxy ×
retrain tactic × seed) run. Responsibilities:
- Build adapter from config; reset knowledge dirs into a fresh
  `runs/<timestamp>_<tag>/` directory (never clobber `knowledge/` in place —
  copy the template knowledge dir per run).
- Launch the managed system headless (no dashboard needed): reuse
  `run_managed_system.py` but add a `--config` / `--run-dir` mode that
  bypasses `approach.conf` indirection; keep the ACP path working for
  dashboard use.
- Run to stream exhaustion; write `run_manifest.json` (full resolved config,
  git commit, hardware info, energy backends, seed) and copy all knowledge
  artifacts into the run dir.
- Enforce protocol knobs from the experiment config: cooldown seconds between
  runs, fixed seeds list, chronological splits only.

### 6.2 `experiments/run_grid.py`

Reads `configs/experiments/<name>.yaml`:

```yaml
datasets: [pems_node1, ...]        # filled when chosen
planners: [naive, random_switch, greedy_switch, oracle_switch,
           harmone_original, violation_aware, pareto]   # bandit optional
seeds: [1,2,3,4,5]
cooldown_minutes: 20
prt_baselines: {enabled: true, period_steps: 3200}
```

Runs the full grid sequentially, resumable (skips completed run dirs), and
emits a grid manifest. PRT baselines implemented as a planner-independent
stream hook (retrain every N steps regardless of signals).

### 6.3 `experiments/metrics.py`

Per run and aggregated (mean ± std over seeds):
- accuracy (R² / offline mAP / proxy), cumulative energy J, mean inference
  latency ms (excluding retrain intervals), counts {switches, retrains, VMR,
  noop, retrain_rejected, retrain_skipped_budget}, MAPE-K overhead % of total
  energy.
- Pareto efficiency: normalised area under the accuracy-vs-energy curve
  relative to the oracle upper bound (document formula in the module
  docstring; oracle from 3.2).
- Proxy Spearman ρ (CV, from proxy_validation).
- Paired Wilcoxon (pareto vs harmone_original per dataset) + effect size
  (matched by seed); output a tidy CSV + a LaTeX table generator.
- Sensitivity sweep driver for β / (w_acc,w_e) / tau_drift on configured
  datasets.

**Phase 5 exit criteria:** full grid runs end-to-end on `pems_node1` alone
(the only dataset that exists yet) across all planners × 2 seeds in one
command, producing the aggregated table without manual steps.

---

## 7. Phase 6 — Repo Hygiene & Handoff

- `reproduce/` scripts: one per RQ (RQ1–RQ6), each a thin wrapper over
  `run_grid.py` + `metrics.py` with the right experiment yaml. Where a dataset
  is missing, the script must exit with a clear message naming the config file
  to create — not crash.
- Update README: new architecture diagram (core/adapters/experiments), dataset
  onboarding guide ("to add a dataset: copy `_template.json`, fill fields, run
  calibrate scripts, add to experiment yaml — no code changes").
- `requirements_cv.txt` split out (ultralytics, pynvml, torchvision);
  base `requirements.txt` stays CPU-lean. Pin versions.
- Tag the commit used for experiments (`v2.0-journal-experiments`).
- Finalise `CHANGES_FROM_PAPER.md` and `DECISIONS_PENDING.md`.
- Ensure `cv_harmone_score.json` and all preset policies regenerate from
  thresholds via a `scripts/regen_policies.py` (single source of truth per B5).

---

## 8. Execution Order & Dependency Summary

```
Phase 1 (bugs)  ──►  Phase 2 (interfaces)  ──►  Phase 3 (CV)  ──►  Phase 5 (harness)
                                    │                                    ▲
                                    └──────────►  Phase 4 (energy) ──────┘
Phase 6 last.
```

Phase 4 can start any time after Phase 1 (EnergyMeter has no dependency on
planners). Phase 3.4/4.6 need Phase 4's meter for honest energy accounting —
if built earlier, leave the meter call sites as the abstraction with the RAPL
backend only.

## 9. Definition of Done (Whole Project)

1. All tests pass; smoke tests for both domains pass on bundled data.
2. Bugs B1–B7 fixed, each with a regression test and a CHANGES entry.
3. Seven planner strategies (S7 may be stub) selectable via config; oracle
   available in harness.
4. CV loop runs with zero runtime label access; proxy validation pipeline
   produces ρ tables from a run directory in one command.
5. MMD/Fréchet embedding drift + unified VMR with typed signatures.
6. EnergyMeter with CPU+GPU backends and per-dataset calibrated bounds; no
   mixed-backend aggregation.
7. Adding a new dataset requires only a config file + calibration scripts —
   verified by adding a second toy config and running the grid on it.
8. `run_grid.py` reproduces the full baseline grid unattended and resumable.
9. Reproduction of the original PeMS result documented with pre/post-fix
   numbers.
10. `DECISIONS_PENDING.md` lists every deferred choice with the options
    implemented.
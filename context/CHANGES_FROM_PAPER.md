# Changes From the Paper — Harmonica Journal Extension

This file records every change to behaviour (fixes, refactors, additions) relative to the
system described in the original paper, including whether each change was present during
the paper's experiments and the expected effect on reproduced numbers.

---

## B1 — Dynamic Energy Threshold Formula (CRITICAL)

**Bug:** `managed_system_regression/mape_logic/analyse.py` used:
```python
new_threshold = current + 0.95 * (original_max_energy - used_energy)
```
With `original_max_energy = max_energy = 1` and `used_energy ∈ [0,1]`, the expression
`(1 - used_energy)` is always ≥ 0, so `new_threshold` only ever grows. The threshold
asymptotically approaches 1.0; the energy violation branch **never fires** after the first
few cycles. The same bug existed in the CV `analyse.py` with factor 0.4.

**Fix (Phase 1):** Replaced with Eq. 3:
```python
new_threshold = clamp(current + delta * (E_ref - E_used), lo=0.1, hi=1.0)
```
Extracted to `core/scoring.py::update_energy_threshold()`. Config keys `E_ref=0.7` and
`delta=0.1` added to `thresholds.json`. `max_energy` corrected from `1` to `0.6`.

**Effect on paper numbers:** The energy violation arm of the planner was entirely
inoperative during all paper experiments. Published energy adaptation numbers are
therefore for the score-violation path only. Post-fix runs will likely show more
frequent model switches triggered by energy, potentially changing cumulative energy
totals and switch counts. Delta documented in A4 reproduction run.

**CV:** Same fix applied to `managed_system_cv/mape_logic/analyse.py` (factor was 0.4,
now replaced uniformly with the `update_energy_threshold()` call).

---

## B2 — VMR Reuse Tactic Unreachable (CRITICAL)

**Bug:** Regression `analyse_drift()` returned `{"drift_detected": ..., "best_version": ...}`
but `plan_drift()` read keys `action` and `version`. Key mismatch meant `drift_analysis.get("action")`
always returned `None`, so every drift event fell through to `{"action": "retrain"}`. The
VMR (Versioned Model Repository) replace path was **completely unreachable** in the
regression domain.

**Fix (Phase 1):** `analyse_drift()` now returns the aligned contract:
```python
{"drift_detected": True,  "action": "replace", "version": "<path>"}  # VMR hit
{"drift_detected": True,  "action": "retrain", "version": None}       # no match
{"drift_detected": False, "action": None,      "version": None}
```

**Effect on paper numbers:** Published VMR event counts for regression (0 in the
sample `mape_info.json`) are likely accurate for the wrong reason — the counter
never incremented because the code path was unreachable. Post-fix, VMR events will
increase and retrain events will decrease whenever a suitable archived version exists.

---

## B3 — Drift Reference Is Rolling, Not Training Distribution (METHODOLOGY)

**Bug:** `monitor_drift()` computed KL divergence between two adjacent windows of
predictions (`[-2400:-1200]` vs `[-1200:]`). This detects local distributional changes
but misses **gradual drift**: if the distribution shifts slowly over many windows, each
adjacent pair looks similar (low KL) even as the current distribution is far from the
original training distribution.

**Fix (Phase 1):**
- Added `core/drift/kl_fixed_ref.py`: KL(current_window ‖ training_reference)
  where the training reference is persisted in `knowledge/reference_distribution.json`.
- Renamed the old logic to `core/drift/kl_rolling.py` (key: `kl_local`).
- Config key `"drift_reference": "fixed" | "rolling" | "both"` (default `"both"`).
- Adaptation triggers on the fixed-ref signal; rolling signal is secondary telemetry.

**Effect on paper numbers:** Paper used rolling KL. Fixed-ref KL will detect
gradual drift earlier, potentially triggering more (and earlier) drift adaptations.
The two signals are now independently measurable — this comparison is one of the
paper's contributions.

---

## B4 — Model Reloaded From Disk Every Inference Step (MEASUREMENT)

**Bug:** `managed_system_regression/inference.py` called `torch.load()` or `pickle.load()`
on every iteration of the inference loop (0.15 s apart). Model deserialization is not
measured by pyRAPL as "inference energy" — it shows up in the `energy_uJ` column,
inflating reported energy per inference. LSTM state dicts are ~200 KB; pickle files
(SVM, Linear) are smaller but still non-trivial to deserialise repeatedly.

**Fix (Phase 1):** Module-level `_model_cache` dict. Model only reloaded when:
1. `model.csv` content changes (string comparison on every iteration — cheap).
2. `knowledge/model_reload.flag` exists (executor writes it after VMR/retrain).

**Effect on paper numbers:** Reported per-inference energy will decrease for runs
using LSTM (most affected). Inference latency will also drop. The paper's 1.89 ms
average likely included deserialization overhead; post-fix latency will be lower.

---

## B5 — Random KL Placeholder During Warmup (CORRECTNESS)

**Bug:** When fewer than 2400 samples exist, `monitor_drift()` returned:
```python
{"kl_div": round(np.random.uniform(0.01, 0.15), 4)}
```
The ACP secondary boundary threshold is 0.10. Values above this (possible ~50% of
draws) triggered the `handle_data_drift` tactic during the warmup period of every
run, before any real drift signal was available. This corrupted event counters and
could trigger spurious retrains.

**Fix (Phase 1):** Returns `{"kl_div": None}`. Consumers skip boundary checks on `None`.
Drift threshold unified in `thresholds.json["tau_drift"]`; startup assertion warns if
policy JSON and thresholds disagree.

**Effect on paper numbers:** Event counters (retrains, VMR) from very early in runs
may have been inflated by false drift events. Post-fix, early-run event counts will
be lower and more accurate.

---

## B6 — Switch Counter Increments on No-Op (MEASUREMENT)

**Bug:** `execute_mape()` called `record_event("switch", energy_consumed, ...)` even when
`plan_mape()` returned `None` (no switch needed). The `model_switches` counter in
`mape_info.json` thus counted planning cycles, not actual model switches.

**Fix (Phase 1):** `record_event("noop", ...)` when plan returns `None`. `record_event(
"switch", ...)` only when a new model is actually written. `mape_k_energy_uJ` still
accumulates for noop events (overhead is real).

**Effect on paper numbers:** Published `model_switches` counts (e.g. 68 in the sample
`mape_info.json`) are overstated; they include planning cycles that decided not to switch.
Post-fix counts will be strictly lower and accurately reflect adaptation actions.

---

## B7 — Scaler Data Leakage (METHODOLOGY)

**Bug:** `inference.py` fitted `MinMaxScaler` on the full streaming dataset (including
test data) before the loop. This allowed test distribution extremes to influence the
normalization bounds, leaking future information into model inputs during experiments.

**Fix (Phase 1):** Scaler fitted only on training data (the first split of `dataset.csv`,
or on `drift.csv` after retrain). Persisted to `knowledge/scaler.pkl`. `inference.py`
loads and applies `transform()` only; never calls `fit()`. Retrain overwrites atomically
(write to `.tmp` + `os.replace()`).

**Effect on paper numbers:** Models were receiving slightly different input distributions
than if the scaler had been fitted on training data alone. The magnitude of the effect
depends on how different the test distribution's extremes were. Minor for PeMS (relatively
stable traffic data); may be larger for datasets with strong non-stationarity.

---

## A1 — Rename Random Switch Baseline

**Change:** `plan_simple_switch` → `plan_random_switch`. Tactic ID `"switch_model_r2_baseline"`
kept as alias. New `plan_greedy_switch` (S3) added. Dashboard labels updated.

---

## A2 — CV EMA Head-Start Gated Behind Config

**Change:** The `+0.1` EMA inflation after VMR deployment in CV `execute_drift()` is now
conditional on `thresholds.json["ema_head_start"]` (default 0.1; 0 to disable). Logged
as event type `"ema_head_start"`.

---

---

## Phase 2 — Pluggable Interfaces

### Planner Registry (`core/planners/`)

Seven selectable planners (S1–S7) registered via `core/planners/base.py::get_planner()`.
Active planner set in `thresholds.json["planner"]`. Both managed systems delegate to the
registry; the `harmone_original` planner preserves exact paper behaviour.

New planners:
- S1 `naive`: always noop (useful as lower bound baseline).
- S2 `random_switch`: uniform random over non-current models (was paper's exploration arm only).
- S3 `greedy_switch`: max(ema_scores) — deterministic exploitation without ε.
- S4 `harmone_original`: paper's ε-greedy + VMR/retrain on drift (default).
- S5 `violation_aware`: separate energy / score violation handling.
- S6 `pareto`: Chebyshev-distance selection on (accuracy, energy) Pareto front.
- S7 `bandit`: LinUCB stub (NotImplementedError — future work).

### Dataset Adapter (`adapters/`)

`adapters/base.py` defines `DatasetAdapter` ABC with `train_split()`, `val_split()`,
`stream()`, `offline_labels()`, `models()`. `adapters/regression_csv.py` implements
the generic CSV adapter (PeMS). Dataset configs live in `configs/datasets/`.

### Separated EMA signals

`mape_info.json` now tracks `ema_accuracy[model]` and `ema_energy[model]` alongside
the legacy `ema_scores[model]`. Both regression and CV monitor phases call
`core/scoring.update_separated_emas()`. Planners can use either signal set.

---

## Phase 3 — CV Generalisation

### Accuracy Proxies (`core/proxies/`)

Three selectable proxies controlled by `thresholds.json["proxy"]`:
- `confidence` (default): mean detection confidence — the paper's implicit signal.
- `calibrated_confidence`: Platt/temperature scaling per model; calibration persisted in
  `knowledge/calibration.json`. Enables RQ3 proxy validation.
- `agreement`: multi-model detection-level agreement (IoU > threshold, same class).
  Falls back to raw confidence if companion predictions not available.

CV `mape_logic/monitor.py` now calls `proxy.score(inference_record)` instead of raw
`df["confidence"].mean()`. Paper behaviour is preserved as the `confidence` default.

### Embedding Drift Detectors (`core/drift/`)

Two new selectable drift detectors for CV:
- `mmd_embedding`: Unbiased MMD² with RBF kernel; bandwidth frozen at `fit_reference()`
  via median pairwise distance heuristic (Gretton et al. 2012).
- `frechet_embedding`: FID-style Fréchet distance between Gaussian approximations of
  reference and current embedding distributions.

The paper's `luminance_kl` detector is preserved as the default (`"drift_detector": "luminance_kl"`).

### Unified VMR (`core/vmr.py`)

`VMR` class with typed `store()` / `best_match()` / `restore()` API. Stores weights +
distribution JSON + meta per version under `knowledge/vmr/<model>/<timestamp>/`.
Match strategies: `"best_score"` (highest proxy_score) and `"closest_distribution"`
(KL for histograms, Euclidean mean distance for embeddings).

### CV Retrain Tactics (`managed_system_cv/retrain_tactics/`)

Three selectable tactics for post-drift adaptation:
- `retrain_oracle`: supervised fine-tune using augmented labelled data (paper's method,
  refactored); only fires when `offline_labels()` available.
- `pseudo_label`: label-free fine-tune using high-confidence predictions (conf > `tau_pseudo`);
  safety valve discards new weights if proxy score worsens by > `tau_regression`.
- `human_in_loop`: emits `labeling_required` event, switches to lowest-energy model,
  watches `knowledge/incoming_labels/` for YOLO .txt files from external annotators.

### Experiment Scripts (`experiments/`)

- `offline_eval.py`: offline mAP@0.5 per monitoring interval using Ultralytics `val()`.
- `proxy_validation.py`: Spearman ρ pipeline — single command computes ρ for all three
  proxies vs ground-truth mAP@0.5. Supports `--smoke` flag for 3-interval smoke run.

---

## Phase 4 — Energy Instrumentation

### EnergyMeter Abstraction (`core/energy.py`)

> **Note on Phase 6 revision:** Phase 4 originally wrapped pyRAPL. Phase 6 (E3
> live-run fix) rewrote `core/energy.py` to use pyJoules because pyRAPL is
> incompatible with AMD Ryzen AI 7 350. This section now documents the **actual
> pyJoules implementation** shipped in Phase 6. The earlier pyRAPL wrapper no
> longer exists anywhere in the codebase (`import pyRAPL` will not be found).

All direct pyRAPL calls have been removed from both managed systems and replaced
with the `EnergyMeter` context manager backed by pyJoules.

**Backends**, selected via `thresholds.json["energy_meter"]`:
- `"null"` (`_NullBackend`): always returns `None`; used on Windows / dev / CI.
- `"rapl"` (`_PyJoulesRaplBackend`): CPU energy via pyJoules `RaplPackageDomain`.
  Calls `_probe_rapl()` on first use; if the probe returns zero (missing powercap
  permissions or AMD quirks), falls back to `_NullBackend` with a `RuntimeWarning`.
- `"nvml"` (`_PyJoulesNvmlBackend` + `_PyJoulesRaplBackend`): GPU via pyJoules
  `NvidiaGPUDomain`; CPU RAPL alongside. If NVML probe returns zero (RTX 5060
  Laptop with no cumulative counter), falls back to `_PollingGPUBackend`.
- `"auto"` (default): tries `_probe_rapl()`; falls back to `_NullBackend` with
  a warning. Does not probe GPU automatically (use `"nvml"` explicitly).

**Backend probes** (`_probe_rapl`, `_probe_nvml`):
- Called lazily on first EnergyMeter construction; results cached in module-level
  `_RAPL_PROBED` / `_RAPL_AVAILABLE` and `_NVML_PROBED` / `_NVML_AVAILABLE`.
- Results also written to `knowledge/.energy_backends.json` for run manifest.
- GPU polling fallback (`_PollingGPUBackend`): spawns a daemon thread that reads
  `nvidia-smi --query-gpu=power.draw` at 50 ms intervals; integrates to µJ on stop.

**Re-entrancy guard**: module-level `_ACTIVE` bool prevents nested `EnergyMeter`
contexts from double-counting RAPL package-level counters.

**Result dict keys** (from `meter.result`):
`cpu_uJ`, `gpu_uJ`, `total_uJ`, `cpu_valid`, `gpu_valid`, `valid`, `backends`.
- `valid` is True iff at least one component produced a non-None reading.
- `backends` is a list of active backend names (e.g. `["rapl", "nvml_polling"]`).

**Call sites (all 9 usages across both domains):**
- `managed_system_regression/inference.py` — inference loop
- `managed_system_regression/mape_logic/execute.py` — `execute_mape`, `execute_drift`
- `managed_system_cv/inference.py` — inference loop
- `managed_system_cv/mape_logic/execute.py` — `execute_mape`, `execute_drift`
- `managed_system_cv/mape_logic/manage.py` — ACP tactic router
- `managed_system_cv/retrain.py` — legacy YOLO fine-tuning
- `managed_system_cv/retrain_tactics/pseudo_label.py` — pseudo-label fine-tuning
- `managed_system_cv/retrain_tactics/retrain_oracle.py` — oracle retrain
- `managed_system_cv/utility/raw_model_stats.py` — pilot evaluation utility

**Contract for callers:**
- `meter.total_uJ` is `float | None`; None when backend unavailable.
- `meter.result["valid"]` is False when no backend produced a reading.
- Call sites pass `meter.total_uJ or 0.0` to `record_event()`.
- Never silently returns `0.0` as a valid measurement.

**Effect on paper numbers:** The paper ran with pyRAPL on different hardware.
The pyJoules RAPL backend reads the same `/sys/class/powercap/intel-rapl/`
interface, so readings are numerically compatible when running on Linux with
RAPL permissions. E_m/E_M normalisation bounds must be recalibrated via
`scripts/probe_energy.py` after confirming backend availability (E1/E2 in
DECISIONS_PENDING.md).

---

## Phase 5 — Experiment Harness

### Headless Inline Runner (`experiments/run_experiment.py`)

Replaces the ACP/Flask subprocess approach with a pure Python inline MAPE loop.
All (dataset × planner × seed) combinations run in a single process with no
inter-process communication.

Key design decisions:
- `EnergyMeter(backend="null")` in the harness (no RAPL hardware dependency for
  batch experiment runs on Windows/CI); overridden in configs/experiments/baseline.yaml.
- Scaler fitted on `adapter.train_split()` and persisted to run_dir/scaler.pkl.
- Reference distribution for KL drift detection written to run_dir/reference_distribution.json
  at setup time (not shared across runs; each run gets an independent reference).
- Retrain/replace decisions are logged as `retrain_skipped` events rather than
  executing full retraining — the harness measures planners, not retrain tactics.
- Artifacts per run: `predictions.csv`, `mape_events.csv`, `mape_info.json`,
  `thresholds.json`, `run_manifest.json`, `scaler.pkl`, `reference_distribution.json`.

**Contract changes relative to live managed system:**
- `monitor_interval=50` (vs ~7 MAPE cycles in 1200-prediction live system) —
  configurable; shorter for faster experiments.
- `stream_delay_s=0.0` override (config key) — no real-time simulation in experiments.

### Model Loaders (`adapters/loaders.py`)

`lstm_loader`, `sklearn_loader`, `yolo_loader` (CV stub) with `get_loader()` registry.
`LSTMModel` class exported for shared use between harness and managed system inference.

### Grid Driver (`experiments/run_grid.py`)

Loads YAML or JSON grid configs; iterates dataset × planner × seed combos; resumes
incomplete grids; writes `grid_manifest.json` summarising the run.

### Metrics (`experiments/metrics.py`)

`compute_run_metrics(run_dir)` reads per-run artifacts. `aggregate_grid(grid_dir)`
collects across all runs. `aggregate_by_planner(rows)` collapses seeds. Pareto
efficiency marking. Wilcoxon signed-rank test for paper's statistical claims.
CSV and LaTeX output.

### Baseline Grid Config (`configs/experiments/baseline.yaml`)

`pems_node1` × [naive, random_switch, greedy_switch, harmone_original, violation_aware, pareto]
× seeds [1, 2]. `monitor_interval=50`, `cooldown_minutes=0`,
`energy_meter="null"` override (Windows-safe).

**Effect on paper numbers:** None — the harness is additive. The paper's numbers
came from the live managed system; the harness produces independently comparable
numbers for the paper's new planners (S1–S6).

---

---

## Phase 6 — Live-Run Bug Fixes (post July 24 2026 first live run)

These fixes address nine issues (L1–L6, E1–E3) identified from the first live
regression run that produced zero valid results. See `live-run-report.md` for
the full diagnostic and shell-output analysis.

### L1 — Scaler Initialisation

**Root cause:** B7 fix created a hard dependency on `knowledge/scaler.pkl` but
no script created it for the live managed system, and inference crashed on
startup.  Additionally, `monitor_mape()` re-served stale cached rows as fresh
telemetry when inference was dead, causing 112 fabricated model switches.

**Fixes:**
- `scripts/init_regression.py` — fits MinMaxScaler on train split only,
  persists scaler + reference_distribution.json.  `--config` driven,
  idempotent, `--force` flag.
- `managed_system_regression/mape_logic/monitor.py` — deleted stale-cache
  fallback; returns `{"fresh": False}` when no new rows.
- `run_managed_system.py` — startup artifact health check aborts with
  remediation command if any required artifact is missing; `push_telemetry()`
  skips POST on `fresh=False`; polls `Popen.poll()` each cycle and exits on
  dead child.

### L2 + L6 — Stale Command Replay and Cross-Session Counters

**Root cause:** `command.txt` carried commands from previous sessions into new
runs; event counters accumulated across sessions making adaptation counts
meaningless.

**Fixes:**
- Both `manage.py` files: `_clear_stale_command()` on startup; timestamp-gated
  command format `tactic_id|unix_ts`; `_parse_and_validate_command()` rejects
  commands older than 30 s (legacy no-timestamp format also rejected).
- `run_managed_system.py`: writes timestamped commands `tactic_id|time.time()`.
- `experiments/run_reset.py`: zeros all per-run state while preserving
  training artifacts.

### L3 — PyTorch sm_120 Arch Guard (CV)

**Root cause:** PyTorch 2.13.0+cu126 has no kernels for RTX 5060 Laptop
(compute capability 12.0), causing `AcceleratorError` mid-warmup rather than
a clear startup message.

**Fix:** `managed_system_cv/inference.py` — fast-fail GPU arch guard on startup
with clear message including remediation steps (reinstall cu129 or cu133).

### L4 — None-Safe Boundary Evaluation in ACP

**Root cause:** `periodic_secondary_checks()` in `app.py` crashed on `None`
KL divergence, silently killing the monitoring thread.

**Fix:** Extracted `evaluate_boundary(value, condition, threshold)` — returns
`False` on `None`; wraps loop body in try/except with heartbeat DEBUG log so
a dead thread becomes detectable.

### L5 — Planner Anti-Thrash Cooldown Guard

**Rationale:** Downstream of L1; resolves with live inference.  Added optional
guard to prevent re-switching faster than `switch_cooldown_s` (default 30 s
in `thresholds.json`) — a legitimate anti-thrash mechanism aligned with the
paper's "tolerate transient spikes" narrative.

**Effect on paper numbers:** The paper's runs had no cooldown; adding a 30 s
guard will reduce switch frequency in high-volatility windows. Documented as
a configurable departure: set `switch_cooldown_s=0` to reproduce the exact
paper planner behaviour.

### E3 — Energy Abstraction: pyRAPL → pyJoules

**Rationale:** pyRAPL is incompatible with AMD Ryzen AI 7 350; pyJoules
provides a unified interface for CPU RAPL + GPU NVML under one library.

**Fixes:**
- `core/energy.py` — completely rewritten: `_PyJoulesRaplBackend`,
  `_PyJoulesNvmlBackend`, `_PollingGPUBackend` (50 ms nvidia-smi integration
  for GPUs without cumulative NVML energy counter).
- Module-level `_ACTIVE` re-entrancy flag prevents nested contexts from
  double-counting package-wide RAPL counters.
- Probe functions `_probe_rapl()` / `_probe_nvml()` called lazily on first
  use; results cached to `knowledge/.energy_backends.json`.
- `scripts/probe_energy.py` — standalone E1/E2 probe with pynvml + nvidia-smi
  fallback chain.
- Test `TestNoPyRAPL` confirms `import pyRAPL` is absent from the entire tree.

**Effect on paper numbers:** The paper ran with pyRAPL on a different CPU
(Ryzen AI 7 350 is new hardware). All `E_m`/`E_M` normalisation bounds must
be recalibrated after E1/E2 probes confirm which backends are active.
Old bounds recorded in DECISIONS_PENDING.md pending post-probe recalibration.

### Dataset Plug-and-Play Contract (Block 4)

**Added:** `core/dataset_validator.py`, `docs/DATA_CONTRACT.md`,
`scripts/validate_dataset.py`, `scripts/make_toy_datasets.py`,
`configs/datasets/toy_regression.json`, `configs/datasets/toy_cv.json`.
A conformance test (`tests/test_plug_and_play.py`) proves config-only dataset
onboarding for both domains.  Adding a new dataset that requires code changes
is a contract violation.

---

## Phase 7 — Dashboard Planner Selection Modal (D10)

The `frontend/dashboard.html` was extended with a planner selection modal that intercepts any HarmonE button click (regression or CV). 

**Changes to dashboard.html:**
- `HARMONY_PRESETS` expanded from 6 → 12 entries: added 5 per-planner regression variants (`reg_greedy_switch`, `reg_violation_aware`, `reg_pareto`, `reg_random_switch`) and 5 CV equivalents.
- `PLANNERS` array: 5 entries driving the modal (harmone_original, greedy_switch, violation_aware, pareto, random_switch).
- `launchHarmonE(approachKey, domain)` function: shows modal, waits for planner selection, calls `/api/set-planner`, then proceeds with `reset` → `writeApproachConfig` → `savePolicyToFolder` → start button.
- CV HarmonE button re-enabled (was commented out).

**Changes to app.py:**
- `/api/set-planner` endpoint added. Writes the selected planner name to `thresholds.json["planner"]` for the relevant domain. The planner takes effect on the next MAPE cycle; no restart required.

**Effect on paper numbers:** None — only the dashboard UI and the `thresholds.json["planner"]` key are new. All existing planner implementations preexisted; the modal just exposes them without requiring manual JSON edits before each run.

---

## Phase 8 — Integration Hardening (CP7)

A full audit of the dashboard → backend → managed-system integration path was performed. Nine code bugs and five documentation drift issues were resolved.

**Code fixes:**

| ID | Fix |
|---|---|
| G1 | `import json` missing from `app.py` top-level; `/api/set-planner` would crash with `NameError` |
| G2 | `launchHarmonE()` used variant key (e.g. `reg_greedy_switch`) as policy_id; prefix scan in `run_managed_system.py` uses base approach (`reg_harmone`); policy file was never found. Fixed: use `approachKey` (`reg_harmone_score`) as policy_id |
| G7 | CV `execute_tactic_locally("handle_data_drift")` was `pass` — drift tactic silently dropped. Fixed: `execute_drift(trigger="acp")` |
| G8 | `/api/start-managed-system` returned HTTP 200 immediately; dashboard began polling even if the managed system died within seconds. Fixed: endpoint waits 4 s, polls process, returns HTTP 500 + last 20 log lines on early exit; dashboard keeps start button visible for retry |
| G9 | `/api/set-planner` had no input validation. Fixed: `_VALID_PLANNERS` set; bandit/unknown → HTTP 400 |
| G12b | `random_switch` tactic case missing in both `manage.py` files. Fixed: `execute_simple_switch(trigger="acp")` handler added to regression and CV manage.py |
| G5 | `random_switch` label ambiguous (same name as a standalone baseline). Fixed: renamed to "Random Switch (HarmonE policy)" in PLANNERS array |

**Documentation drift fixed:** Endpoint names corrected in all context docs (`/api/adaptor/upload` → `/api/upload-custom-mape`; `/api/set-approach` → `/api/write-approach`). Test count updated.

**Known limitation — G_CV_PLAN:** CV domain calls `plan_mape()` directly, not `dispatch_plan()`. `thresholds.json["planner"]` is written for CV but ignored at runtime. Planner selection in the CV HarmonE modal has no effect on planning behaviour. Documented as DP11; deferred to a future pass.

**Effect on paper numbers:** None — the bugs fixed were in the dashboard/ACP layer, not the MAPE-K algorithms themselves. CV drift tactic now fires when triggered by ACP (G7), but the paper's results came from local-mode runs where manage.py ran the MAPE loop independently without ACP-commanded tactics.

---

## A3 — Git Archaeology

**Evidence basis:** `git log --oneline --follow <file>` run on all B-bug files 2026-07-29.

**Finding:** The repository history is shallow and partially rewritten. The earliest
substantive commits are:
- `8808030a regression` — initial regression code upload
- `01dfbd5d cv integrated` — initial CV integration
- `2d8fe66b Add files via upload` — bulk file upload

All seven B-bugs exist in the code at the earliest available commit. No commit
introduces any of them — they were present from the initial upload, indicating
they predate all tracked experiments.

**Per-bug confirmation:**
- **B1** (energy threshold formula): `current + 0.95 * (original_max_energy - used_energy)` 
  present since `8808030a`. Confirmed predate: ✅
- **B2** (VMR key mismatch `action` vs result keys): `analyse_drift()` returned `best_version`,
  `plan_drift()` read `action` since `8808030a`. Confirmed predate: ✅
- **B3** (rolling drift reference): adjacent-window KL present since `8808030a`. Confirmed: ✅
- **B4** (model reload every step): `torch.load()` / `pickle.load()` inside loop since initial
  upload. Confirmed predate: ✅
- **B5** (`np.random.uniform` placeholder): present since `8808030a`. Confirmed: ✅
- **B6** (switch counter on no-op): `record_event("switch", ...)` before decision check since
  initial upload. Confirmed: ✅
- **B7** (scaler data leakage): `MinMaxScaler.fit()` on full dataset at top of script since
  `8808030a`. Confirmed: ✅

**Conclusion for paper:** All bugs were present during the paper's experiments. Published
numbers reflect these bugs' effects; the delta analysis in A4 quantifies the impact of
each fix.

---

## A4 — Reproduction Run

*TODO: run after lab machine access. Target: ±5% of paper values.*

**Paper values (reg_harmone, PeMS):**
- R² = 0.8628
- Energy = 20.62 mJ
- Latency = 1.89 ms
- Adaptations = 12

**Known expected deltas:**
- B4 fix → latency and per-inference energy will decrease.
- B6 fix → switch count will decrease (no-op cycles removed).
- B1 fix → energy adaptations will now fire; switch count may increase.
- B5 fix → early-run retrains removed; retrain count may decrease.

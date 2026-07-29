# HarmonE Tool — Post-Live-Run Improvement Plan for Coding Agent

> **Audience:** Claude Sonnet 4.6 acting as an autonomous coding agent inside
> the `HarmonE-tool` repository, branch `fix/bugs-phase1` (create
> `fix/live-run-issues` off it for this work).
> **Trigger:** First live run on July 24 2026 produced zero valid results.
> Nine issues (L1–L6, E1–E3) were identified from shell-output analysis; the
> full diagnostic is in `docs/live-run-report.md` (the bug report document).
> Read it in full before writing any code — this plan tells you WHAT to build
> and in what order; the report tells you WHY and shows the exact failures.
> **Two strategic changes in this plan:**
> 1. Energy measurement migrates from **pyRAPL to pyJoules** everywhere
>    (unified CPU RAPL + GPU NVML under one library, with a polling fallback).
> 2. The dataset layer is hardened into a **plug-and-play contract**: any
>    dataset whose preprocessed form matches the documented schema must run
>    with zero code changes — config file only.

---

## 0. Environment Facts (Treat as Ground Truth)

| Fact | Value | Consequence |
|---|---|---|
| Machine | Lenovo Legion 5 Gen 10 laptop — the ONLY experiment machine | No "lab machine" exists. Everything must run here. Thermals matter (see §7.4). |
| CPU | AMD Ryzen AI 7 350 (Zen 5) | User has confirmed the chip exposes an Intel-compatible RAPL interface. Use the pyJoules RAPL backend as primary CPU meter — but still run the E1 probe (§5.1) and hard-fail the run if it returns zeros, because "compatible" must be proven by a non-zero reading in logs, not assumed. |
| GPU | NVIDIA RTX 5060 Laptop, compute capability 12.0 (sm_120) | Current torch 2.13.0+cu126 has no sm_120 kernels → must reinstall (L3). NVML energy counter support unknown → probe, fall back to polling (E2). |
| Python | 3.13 | Constrains available torch wheels — check `cp313` availability. |
| OS | Linux (RAPL/NVML paths assume it) | `sudo modprobe msr` + powercap permissions needed once per boot; add to a setup script, not to runtime code. |

---

## 1. Ground Rules (Apply to Every Change)

1. **Read `live-run-report.md` first.** Every L/E number below maps to a
   section there with logs, root cause, and code sketches. Where this plan and
   the report conflict, this plan wins; note the conflict in the PR description.
2. **Branch `fix/live-run-issues`.** One commit per issue, message referencing
   the ID: `fix: L2 clear stale command.txt on startup + timestamp gate`.
3. **Every fix ships with a regression test** (pytest, `tool/tests/`). For
   process-level issues (L1, L2) the test exercises the extracted pure function
   or the startup check, not the full subprocess stack.
4. **Fail loudly, never silently.** The live run's core pathology was silent
   failure: dead subprocesses with the wrapper running on stale cache. Every
   fix must convert a silent failure into either a hard abort with a
   actionable message, or an explicitly logged degraded mode. This is the
   single most important principle in this plan — apply it beyond the listed
   issues wherever you touch code.
5. **No fabricated telemetry, ever.** Cached/stale/placeholder values must
   never enter the telemetry stream as if fresh (see L1-c below). `None` /
   omitted key = "no signal"; consumers must handle it (L4).
6. **Energy: pyJoules only.** After this plan, `import pyRAPL` must not exist
   anywhere in the repo (grep-enforced by a test). All measurement goes
   through `core/energy.py::EnergyContext`.
7. **Dataset plug-and-play is a contract, not a convention.** §6 defines the
   preprocessed-data schemas. A validator enforces them at startup. If the
   data validates, the tool must run it with no code edits.
8. **Update `CHANGES_FROM_PAPER.md`** for anything affecting comparability
   with the ECSA numbers (hardware change, energy library change, polling
   estimation). Update `DECISIONS_PENDING.md` for deferred choices.
9. **Keep the smoke tests green.** After every block (§8), both domains must
   run the 50-step smoke test successfully before moving on.

---

## 2. Block 1 — Get Regression Running (L1, L2, L4, L5, L6)

### 2.1 L1 — Scaler initialisation (Critical)

The B7 leakage fix created a hard dependency on `knowledge/scaler.pkl` that
nothing creates or checks. Three sub-fixes:

**L1-a — `scripts/init_scaler.py`.** Create (or verify) the script exactly as
specified in the report §L1 Step 1: fit `MinMaxScaler` on the chronological
first 80% of the dataset's value column only, persist to
`managed_system_regression/knowledge/scaler.pkl`. Two generalisations beyond
the report's sketch, required for §6 plug-and-play:
- Read the csv path, value column name, and train fraction from the active
  dataset config (`configs/datasets/<name>.json`), not hardcoded
  `data/dataset.csv` / `"flow"`. Accept `--config <name>` (default: the
  config named in `approach.conf` resolution or `pems_node1`).
- Also persist the training-split value histogram to
  `knowledge/reference_distribution.json` in the same run (the B3 fixed-ref
  drift detector needs it; one init script, all training-time artifacts).

**L1-b — Startup health check.** In `run_managed_system.py`, before spawning
any subprocess, verify every required knowledge artifact for the resolved
domain exists; abort the whole wrapper with the exact remediation command if
not:

```python
REQUIRED_ARTIFACTS = {
    "regression": ["knowledge/scaler.pkl", "knowledge/reference_distribution.json",
                   "knowledge/thresholds.json", "knowledge/model.csv"],
    "cv": ["knowledge/thresholds.json", "knowledge/model.csv",
           "knowledge/reference_embeddings.npz"],
}
# missing → raise RuntimeError(f"Missing {path}. Run: python scripts/init_{domain}.py --config <name>")
```

Wrap CV's training-time artifacts in a matching `scripts/init_cv.py`
(reference embeddings, calibration.json) so both domains have one documented
init entry point. If an artifact's init step is expensive, the script must be
idempotent and skip work when the artifact exists and `--force` is absent.

**L1-c — Kill the stale-cache telemetry path (root of the whole disaster).**
`monitor_mape()`'s fallback ("No new data → using recent data") converted a
dead inference process into 112 fake switches. Change the contract:
- If there are no NEW rows since `last_line`, `monitor_mape()` returns a
  payload with `"fresh": false` and the metric keys set to `None` (or
  omitted). It must NOT re-serve old rows as current telemetry.
- `push_telemetry` skips the POST entirely when `fresh` is false, and after
  `stale_limit` consecutive stale intervals (config, default 6 = 30 s) logs
  an ERROR and POSTs a distinct `{"system_alert": "inference_stalled"}`
  payload that `app.py` surfaces to the dashboard but never evaluates against
  boundaries.
- **Subprocess liveness:** the master wrapper must poll
  `Popen.poll()` on its children every telemetry cycle. A dead child →
  immediate loud shutdown of the whole run (log which child died, exit
  non-zero). A run where inference dies must never keep "running".

**Tests:** (1) init script on a toy csv → scaler params equal train-split
min/max, not full-data; (2) health check aborts on missing scaler with the
remediation string; (3) monitor with no new rows returns `fresh: false` and
no numeric metrics; (4) wrapper with a child that exits → wrapper exits
non-zero within one cycle.

### 2.2 L2 + L6 — Stale command replay and cross-session counters (Critical)

Implement all three sub-fixes from the report — they are complementary layers,
not alternatives:

**L2-a — Startup clear** in both `manage.py` files, before the listener
thread starts: truncate `knowledge/command.txt` if present, log
`"Cleared stale command.txt from previous session."`.

**L2-b — Timestamp-gated commands (make this mandatory, not optional).**
Change the command file format to `tactic_id|<unix_ts>` written by the
Adaptation Handler (`/adaptor/tactic` in `run_managed_system.py`). Listener
parses; executes only if `time.time() - ts < 30`; otherwise discards with a
WARNING. Handle the legacy no-timestamp format during transition by treating
it as expired (safe default). Update both manage.py parsers and the writer in
one commit so formats never mix silently.

**L2-c / L6 — Per-run reset + delta reporting in the harness.**
Add `experiments/run_reset.py` exposing `reset_run_state(domain_dir)`:
- zero all `event_counters` and `simple_switch_counters`, set
  `last_line: 0`, reset every `ema_scores` (and `ema_accuracy`/`ema_energy`
  if present) entry to 0.5, reset `current_energy_threshold` to the
  thresholds.json `max_energy`, clear `recovery_cycles`;
- truncate `predictions.csv` to header only; delete `command.txt`,
  `drift.csv`, `drift_kl.json`, `knowledge/intervals.csv` if present;
- leave `versionedMR/`, models, scaler, and reference artifacts untouched
  (they are training-time state, not run state).
`run_experiment.py` calls this before every run AND records
`counts_before`/`counts_after`, reporting deltas in `run_manifest.json` so a
missed reset can never corrupt reported counts (report §L6 pattern).

**Tests:** listener given an expired command executes nothing and warns;
fresh command executes; reset function leaves versionedMR intact and zeroes
exactly the specified keys; harness manifest contains delta counts.

### 2.3 L4 — None-safe boundary evaluation in ACP (Moderate)

In `tool/app.py`: extract `evaluate_boundary(value, condition, threshold)`
exactly per the report (returns `False` on `None`, handles both conditions,
returns `False` on unknown condition) and use it in BOTH
`analyze_telemetry()` (primary path) and `periodic_secondary_checks()`
(secondary path) — the report only caught the secondary crash, but the
primary path has the same latent bug if a primary QA ever goes `None` under
the L1-c contract. Add the debug log for skipped `None` checks.

**Additionally — thread resilience:** wrap the body of
`periodic_secondary_checks`'s loop iteration in try/except that logs the
traceback and continues. A monitoring thread must never die permanently from
one bad payload. Add a heartbeat DEBUG log every N iterations so a dead
thread is detectable in logs.

**Tests:** `evaluate_boundary(None, "GREATER_THAN", 0.1) is False`; thread
survives a poisoned telemetry entry (inject a string value) and processes the
next one.

### 2.4 L5 — Thrashing (verification only, plus one guard)

L5 is downstream of L1 — no planner change needed. Two required actions:
1. After Block 1 fixes, run a 10-minute live regression session and verify in
   logs: R² in a plausible 0.7–0.95 band, EMA scores stabilising, switch rate
   far below 1-per-5-seconds. Attach the log excerpt to the PR.
2. Confirm `reset_run_state` resets EMA scores to 0.5 (kills the corrupted
   priors: SVM −0.11, LINEAR −2.49 carried from the garbage session).

**Optional guard (implement, default on):** min-interval-between-switches
`switch_cooldown_s` in thresholds.json (default 30 s). The planner returns
noop with reason `"cooldown"` if the last switch is more recent. This is a
legitimate anti-thrash mechanism aligned with the paper's "tolerate transient
spikes" story; document it in CHANGES_FROM_PAPER.md.

---

## 3. Block 2 — Get CV Running (L3)

### 3.1 L3 — PyTorch sm_120 reinstall (Critical)

Follow report §L3 steps 1–5 verbatim inside `harmone_env`:
1. Probe wheel availability for cu129 first, then cu130/cu132, checking for
   `cp313` builds.
2. Uninstall torch/torchvision/torchaudio; reinstall from the first index
   that has cp313 wheels with sm_120 support.
3. Run the GPU verification snippet — the run is only valid if it prints
   compute capability `(12, 0)` and places a tensor on `cuda:0` without
   `AcceleratorError`.
4. `pip install ultralytics --upgrade`, then the YOLO-on-GPU verification
   snippet.
5. **Pin the outcome:** write the exact working versions (torch, torchvision,
   torchaudio, ultralytics, CUDA suffix) into `requirements_cv.txt` with the
   `--index-url` documented in a comment at the top. Record in
   `run_manifest.json` schema: `torch_version`, `cuda_arch_list`,
   `gpu_name`, `gpu_cc`.

**Guard for the future:** add a startup check in CV `inference.py`:

```python
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    arches = torch.cuda.get_arch_list()
    if f"sm_{cc[0]}{cc[1]}" not in arches and not any(int(a.split('_')[1]) >= cc[0]*10 for a in arches if a.startswith('sm_')):
        raise RuntimeError(f"Installed torch has no kernels for this GPU (cc {cc}, arches {arches}). Reinstall per requirements_cv.txt header.")
```

(Adjust the arch-match logic to whatever `torch.cuda.get_arch_list()` actually
returns on the working install — the intent is: crash at startup with a clear
message instead of crashing mid-warmup with `AcceleratorError`.) Combined with
the L1-c liveness poll, a CUDA-broken environment now aborts the run instead
of producing a session of confidence=0.5 fabrications.

**Test:** unit-test the guard function with mocked capability/arch lists for
(supported, unsupported, CPU-only) cases.

---

## 4. Block 3 — Energy Migration to pyJoules (E1, E2, E3)

Do this AFTER Blocks 1–2 so probes run against a working system.

### 4.1 E1 — CPU probe on the Ryzen AI 7 350 (§5.1 gate)

The user reports the chip is Intel-RAPL compatible. Verify, don't trust:
1. Ensure access: `sudo modprobe msr` and powercap permissions — put both in
   `scripts/setup_energy_permissions.sh` (documented in README as
   once-per-boot).
2. Run the report's E1 probe but through **pyJoules**
   (`RaplPackageDomain(0)`), not pyRAPL — pyRAPL is being removed. A non-zero
   pkg reading under a numpy workload = confirmed.
3. **Outcome A (expected):** proceed with RAPL as primary CPU backend. Add to
   threats section text (report §Threats): AMD RAPL accuracy ±10–15%.
4. **Outcome B (zeros/errors despite the compatibility claim):** do NOT
   silently fall back. Stop, record the probe output in
   `DECISIONS_PENDING.md` with the three options from report §E1 Outcome B,
   and implement Option B1's time×measured-power proxy as a
   `_TdpProxyCPUMeter` behind the same interface — but leave it disabled
   until the user picks. The run-level `energy_valid` flag stays false under
   a proxy meter.

### 4.2 E2 — GPU probe on the RTX 5060

After L3: run the report's E2 pyJoules NVML probe under a real torch matmul
load.
- **Outcome A:** NVML backend primary.
- **Outcome B:** `_PollingGPUMeter` (nvidia-smi power.draw @ 50 ms,
  integrate) becomes the GPU backend — implement it exactly per the report's
  class, with these hardening changes:
  - Reuse ONE long-lived polling thread per process (start/stop marks window
    boundaries by sample index) OR keep per-context threads but assert via a
    test that 10k sequential `EnergyContext` uses neither leak threads nor
    accumulate >1 s of total startup latency. Per-inference-step contexts are
    the hot path; thread churn there is a real risk — measure it.
  - Record `sample_count` per measurement; a window with 0 samples must set
    that measurement's `valid: false` (short windows under 50 ms will
    happen).
  - Prefer `pynvml.nvmlDeviceGetPowerUsage` polling over shelling out to
    `nvidia-smi` if pynvml imports and returns sane values — same
    integration math, ~100× cheaper per sample. Keep the subprocess variant
    as last-resort fallback. Probe order: NVML energy counter → pynvml power
    polling → nvidia-smi polling.

### 4.3 E3 — `core/energy.py` unified abstraction

Implement the report's `EnergyContext` design with these amendments (they
override the report where they differ):

1. **Validity over zeros.** The report's Null meters report `0.0` J. Keep
   `0.0` for arithmetic convenience BUT add per-component and overall
   validity: `to_dict()` returns
   `{"cpu_uJ", "gpu_uJ", "total_uJ", "cpu_valid", "gpu_valid", "valid",
   "cpu_backend", "gpu_backend"}`. `valid` = every component the domain
   requires is valid. Downstream (monitor normalisation, metrics aggregation)
   must check `valid` and exclude invalid rows from energy statistics rather
   than averaging zeros in — averaging silent zeros is fabrication.
2. **Probe at first use, cache to disk.** Module-load probing (report design)
   runs a GPU workload on import — too heavy and import-order-sensitive.
   Probe lazily on first `EnergyContext` construction; cache the result to
   `knowledge/../.energy_backends.json` keyed by hostname+boot_id so a
   session probes once. `get_backend_status()` reads the cache; the harness
   logs it in `run_manifest.json` for every run (report's logging
   requirement).
3. **Units discipline.** Internal canonical unit: microjoules (matches the
   existing csv schema). Every public attribute name carries its unit
   (`cpu_uJ` not `cpu`). One conversion at the meter boundary; no `*1e6`
   sprinkled at call sites. Verify the pyJoules trace unit empirically during
   the probe (RAPL domains report µJ; document what NVML/polling paths
   return) and normalise inside the meter classes.
4. **Nesting is forbidden.** RAPL counters are package-wide; nested contexts
   double-count. Keep a module-level re-entrancy flag; constructing an
   `EnergyContext` while one is active raises. Audit call sites for nesting
   during integration (MAPE execute wrapping a retrain that measures itself
   is the likely offender — measure at ONE level, pass results up).
5. **Domain mapping per call site** (report's integration table, confirmed):
   regression inference + regression MAPE execute → `"cpu"`; CV inference +
   CV MAPE execute → `"gpu"`; CV retrain → `"both"`; regression retrain →
   `"cpu"`.
6. **Import mechanics:** no `sys.path.insert` hacks at call sites (report
   sketch). Make `tool/` a proper package (add `__init__.py` files) or
   install with `pip install -e .` via a minimal `pyproject.toml`; import as
   `from core.energy import EnergyContext` everywhere. Pick one mechanism,
   apply repo-wide, document in README.
7. **Removal of pyRAPL:** delete every `import pyRAPL` and `pyRAPL.setup()`;
   remove pyRAPL from requirements; add pyjoules (and pynvml). Add
   `tests/test_no_pyrapl.py` that greps the tree (excluding docs/) and fails
   on any match.
8. **CSV schema change (CV):** `predictions.csv` gains `cpu_uJ`, `gpu_uJ`
   columns; the legacy `energy_uJ` column stays and equals total, so
   monitor/threshold code keeps working unmodified. Regression keeps
   `energy_uJ` (CPU-only) unchanged. Update the CV monitor's normalisation to
   run on `energy_uJ` (total) explicitly.
9. **Recalibrate E_m/E_M.** The switch from pyRAPL scope + hardware change
   invalidates all normalisation bounds. Re-run
   `experiments/calibrate_energy_bounds.py` on this laptop for both domains
   after integration; write new bounds into the dataset configs; note old
   vs new in CHANGES_FROM_PAPER.md.

**Tests:** context manager with mocked backends returns correct dict and
validity; nesting raises; unit conversion (mock trace in µJ → attribute in
µJ); polling meter with synthetic reading stream integrates correctly and
flags zero-sample windows; the no-pyrapl grep test.

---

## 5. Block 4 — Dataset Plug-and-Play Contract

Goal: a new dataset drops in with **a config file + preprocessed data in the
documented schema, zero code changes**. The adapter layer from the original
plan exists; this block turns "convention" into "contract" with schemas,
validation, and a conformance test.

### 5.1 Preprocessed data schemas (write to `docs/DATA_CONTRACT.md`)

**Regression (`domain: "regression"`):**
- One CSV, UTF-8, header row. Required: one numeric value column (name
  declared in config as `value_column`). Optional: `timestamp` column
  (ISO-8601 or unix; used only for plots/split boundaries, stream order is
  row order). Rows strictly chronological. No NaNs in the value column
  (validator rejects; preprocessing must handle imputation — the tool never
  imputes silently).
- Config declares: `csv_path`, `value_column`, `window_length`, `horizon`,
  `train_fraction` (or explicit `split_indices`), `stream_sleep_s`,
  optional `induced_drift` block.

**CV (`domain: "cv"`):**
- A directory of images plus a **manifest CSV** with required column
  `image_path` (relative to the data root), rows in stream order. Optional
  columns: `attributes` (free-form, e.g. weather/timeofday), `label_path`
  (YOLO-format txt, offline eval only — runtime never reads it),
  `split` (`train`/`val`/`stream`). If `split` is absent, config must give
  `train_glob`/`val_glob` or an explicit reference-image list for
  init_cv.py.
- Config declares: `data_root`, `manifest_path`, `models` (name → weights
  path + cost class), `drift_window`, optional `induced_drift`.

Both schemas: every field in `configs/datasets/_template.json` documented
with an inline comment key, including which fields the init scripts consume.

### 5.2 `core/dataset_validator.py`

`validate(config_path) -> ValidationReport` run automatically by
`run_managed_system.py` and both init scripts at startup; also exposed as
`python scripts/validate_dataset.py --config <name>`. Checks:
- config parses, all required keys present, types correct, paths exist;
- regression: value column exists, numeric dtype, no NaN, length ≥
  `window_length + horizon` × a sane minimum; monotonic timestamp if present;
- CV: manifest parses, every `image_path` exists (sample-check 100 + count
  mismatch), at least one model weight file per declared model, label paths
  (if declared) exist for the offline split;
- prints a PASS/FAIL table; FAIL aborts startup with the failing rows listed.
Failing validation must be impossible to bypass except with an explicit
`--skip-validation` flag that stamps `"validation_skipped": true` into the
run manifest.

### 5.3 Conformance proof

Create `configs/datasets/toy_regression.json` + a generated 5k-row synthetic
CSV, and `configs/datasets/toy_cv.json` + 60 generated images + manifest
(script: `scripts/make_toy_datasets.py`). Add
`tests/test_plug_and_play.py` that: generates toys → validates → runs
init scripts → runs a 30-step headless session per domain → asserts
predictions.csv grew with non-null metrics. This test IS the plug-and-play
guarantee; if adding a future dataset requires touching code, this test's
existence makes that a reviewable contract violation.

### 5.4 Init scripts finalised

`scripts/init_regression.py` (absorbs init_scaler.py: scaler + reference
distribution + optional pilot E_m/E_M) and `scripts/init_cv.py` (reference
embeddings + calibration + VMR v1 seeding), both `--config`-driven,
idempotent, validator-gated. README gets a "Adding a dataset" section:
preprocess to schema → copy template config → fill → validate → init → add
to experiment yaml. Nothing else.

---

## 6. Execution Order

Strict order, matching the report's priority blocks:

```
Block 1 (regression): L2 → L1 → L4 → (L5 verify)      §2
Block 2 (CV):         L3                               §3
Block 3 (energy):     E1 probe → E2 probe → E3 build → integrate → recalibrate   §4
Block 4 (datasets):   contract + validator + toys      §5
Block 5:              end-to-end verification          §7
```

L2 before L1 (report's order): clearing stale state first means the first
post-L1 session starts clean. Block 4 can proceed in parallel with Block 3
after Block 1 lands (no dependency), but land Block 3 first if serialising.

---

## 7. Block 5 — End-to-End Verification & Sign-off

### 7.1 Smoke test (must pass before the branch merges)

Regression 50 timesteps + CV 50 frames, each preceded by
`reset_run_state`, asserting ALL of:
- predictions.csv grows with real, varying values (no repeated stale row);
- energy columns non-zero AND `valid: true` in ≥95% of rows;
- counters start at 0; deltas in manifest match observed events;
- command.txt absent/empty at startup (check the startup log line);
- app.py secondary thread alive at session end (heartbeat present in log);
- no `AcceleratorError`; CV ran on `cuda:0` (assert from log);
- backend status logged in manifest with non-"none" cpu_backend and a named
  gpu_backend for CV.
Automate as `scripts/smoke_test.py --domain reg|cv`; wire into the test
suite as a slow/marked test.

### 7.2 Longer soak

One 30-minute regression session and one 15-minute CV session. Verify: no
thread deaths, no thrashing (switch count plausible), memory stable
(psutil RSS logged every minute, assert < 2× start), polling meter (if
active) thread count stable.

### 7.3 Documentation deliverables

- `docs/DATA_CONTRACT.md` (§5.1), README "Adding a dataset" + "Energy setup
  on this machine" (modprobe/permissions/backend cache) sections.
- CHANGES_FROM_PAPER.md: pyRAPL→pyJoules, AMD hardware, polling estimation
  (if used), E_m/E_M recalibration, switch_cooldown guard.
- Threats-to-validity text: copy the three paragraphs from the report's final
  section into `docs/threats_additions.md`, filling the E1/E2 outcome
  placeholders with the actual probe results.

### 7.4 Laptop protocol note (add to experiment docs)

Because this laptop is the lab: experiments run plugged in, on a fixed power
profile (document which), lid open, same physical setup; the existing
20-minute cooldown between runs stays; log CPU/GPU temperature at run start
(via psutil/pynvml) into the manifest and flag runs starting above a
threshold (default 60 °C) — thermal throttling on a laptop is a real
confound for energy comparisons.

## 8. Definition of Done

1. All nine issues closed with tests; smoke + soak pass.
2. `import pyRAPL` grep test green; all measurement via `EnergyContext` with
   probe-selected pyJoules backends and validity flags.
3. Dead subprocesses, stale telemetry, stale commands, and CUDA-kernel
   mismatches all abort loudly instead of fabricating data.
4. Per-run reset + delta counting in the harness; no cross-session counter
   bleed.
5. Toy-dataset conformance test proves config-only dataset onboarding for
   both domains.
6. Probe outcomes (E1/E2), pinned versions (L3), and recalibrated energy
   bounds recorded in manifests, requirements, and CHANGES_FROM_PAPER.md.
7. DECISIONS_PENDING.md updated (empty additions if both probes hit
   Outcome A).
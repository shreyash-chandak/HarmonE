# Architectural Decisions

This file records every significant architectural decision made during implementation,
including the rationale and any alternatives that were considered.

---

## D1 — Dynamic Energy Threshold Formula (B1)

**Decision:** Use `tau_E(i+1) = clamp(tau_E(i) + delta * (E_ref - E_used), lo, hi)` where
`E_ref` is a fixed reference point and `delta` controls the adaptation speed.

**Rationale:** The original formula `current + 0.95 * (original_max - used_energy)` was
monotonically increasing when `original_max = 1` and `used_energy ∈ [0,1]`, because
`original_max - used_energy ≥ 0` always. The threshold would grow toward 1 and the energy
violation branch would never fire. The Eq. 3 formula from the paper uses a signed delta
`(E_ref - E_used)`, which can be negative (threshold decreases when system uses too much
energy) or positive (threshold relaxes when system is energy-efficient).

**Config:** `E_ref` (target energy level, default 0.7) and `delta` (step size, default 0.1)
added to `thresholds.json`. `max_energy` changed from 1 → 0.6 as the initial/meaningful
boundary. `lo=0.1` and `hi=1.0` are clamp bounds in `thresholds.json`.

**Backward compat:** Old behavior (`max_energy=1`) is documented as the bug baseline.
Test `test_b1_energy_threshold.py` demonstrates the monotonic growth with old config.

---

## D2 — VMR Return Contract Alignment (B2)

**Decision:** Align the regression `analyse_drift()` return contract with the CV one,
using keys `action` ("replace"/"retrain"/None) and `version` (path or None), alongside
`drift_detected` (bool).

**Rationale:** Regression `analyse_drift()` was returning `{"drift_detected": ...,
"best_version": ...}` but `plan_drift()` was reading keys `action` and `version`.
This silent mismatch meant the replace path was unreachable; every drift ended in retrain.
The CV domain already had the correct contract. Unifying both domains to the same contract
makes the `plan_drift()` logic shareable (Phase 2).

**Documented in:** `core/vmr.py` (Phase 2 will move this there).

---

## D3 — Dual Drift Signals: Fixed Reference + Rolling (B3)

**Decision:** Maintain both drift detectors simultaneously:
- `kl_fixed_ref`: KL(current_window ‖ P_reference), where P_reference is the training
  distribution persisted at initial setup / after each retrain.
- `kl_rolling`: KL(current_window ‖ previous_window) — the legacy signal, renamed.

The config key `"drift_reference"` (default `"both"`) controls which signal(s) are emitted.
Adaptation triggers on the **fixed-reference** signal when `"both"` is set (so the paper's
comparison is automatic). The `kl_div` telemetry key sent to ACP is always from the primary
(fixed-ref) detector; `kl_local` carries the rolling signal for additional observability.

**Rationale:** Rolling KL misses gradual linear drift (each adjacent pair looks similar
even as the overall distribution shifts far from training). Fixed-ref KL catches it. The
plan explicitly says to keep both because they are comparison baselines for the paper.

**Reference file:** `managed_system_regression/knowledge/reference_distribution.json`
(histogram of `true_value` from the training split). Created by `retrain.py` and by a
new `scripts/init_reference.py` for the initial setup.

---

## D4 — Model Cache with Flag-Based Reload (B4)

**Decision:** Use a module-level dict `_model_cache = {"name": None, "model": None}` in
`inference.py`. On each loop iteration, compare `chosen_model` with `_model_cache["name"]`;
only deserialize if they differ OR if a `knowledge/model_reload.flag` file exists (created
by executor after VMR/retrain).

**Rationale:** Deserialising PyTorch state dicts and pickle files on every inference step
(150 ms apart) adds measurable latency and non-trivial energy overhead. The simple string
comparison is negligible. The `model_reload.flag` is needed because a VMR replace writes
new weights to `models/<name>.pt` while the model name in `model.csv` stays the same —
without the flag, the cache would serve stale weights after a version replacement.

**Flag lifecycle:** Written by `execute_drift()` after `shutil.copy`. Deleted by
`inference.py` immediately after reload. If `inference.py` is not running, the flag
persists until next startup (safe: next load will see it and reload).

---

## D5 — None Signal for Insufficient Drift Data (B5)

**Decision:** `monitor_drift()` returns `{"kl_div": None}` (not a random value) when fewer
than `window_size * 2` (2400) samples exist in `predictions.csv`. Consumers treat
`None` as "no signal": `analyse_drift()` returns `drift_detected: False`; `app.py`
secondary check skips the boundary evaluation; telemetry push omits the key.

**Rationale:** The original `np.random.uniform(0.01, 0.15)` straddled the ACP secondary
boundary (0.10), causing fabricated drift triggers during the warmup period of every run,
which corrupted event-counter metrics and could trigger spurious retrains. A missing signal
is epistemically honest.

**Drift threshold unification:** `tau_drift` (default 0.5 local, 0.10 ACP) is consolidated
into `thresholds.json` as a single `"tau_drift"` key. Policy JSON regenerator
(`scripts/regen_policies.py`, Phase 6) will sync policy files from this value.

---

## D6 — No-Op Event Type (B6)

**Decision:** When `plan_mape()` returns `None` (no switch needed), `execute_mape()` records
event type `"noop"` (not "switch") and still logs the energy consumed by the MAPE-K cycle.
`model_switches` counter is only incremented when a model is actually written to `model.csv`.

**Rationale:** MAPE-K overhead energy is real regardless of whether a switch occurs; hiding
it by not recording no-ops would understate adaptation cost. Separating "noop" from "switch"
makes the switch count an accurate measure of actual adaptations, which the paper uses to
compare planner aggressiveness.

**Event types:** `"switch"`, `"noop"`, `"retrain"`, `"vmr"`, `"retrain_rejected"` (future),
`"retrain_skipped_budget"` (future). All recorded in `mape_info.json["event_counters"]`.

---

## D7 — Scaler Persistence Pattern (B7)

**Decision:** `MinMaxScaler` is fitted once on the training data (`retrain.py` and the
initial `scripts/init_scaler.py`), then persisted to `knowledge/scaler.pkl`. `inference.py`
loads it at startup (not per-iteration) and applies `transform()` only. Refitting on
`drift.csv` at retrain time overwrites atomically (write to `.tmp` then `os.replace()`).

**Rationale:** Fitting the scaler on streaming test data allows data from the test
distribution to influence preprocessing bounds, which leaks future information into the
normalized values that are then passed to the model. This is a methodology bug that would
make results non-reproducible when test windows shift.

**Clip behavior:** After transform, values may exceed [0, 1] if test data has extremes
outside the training range. This is intentional and correct (do not silently clip);
`inference.py` adds an explicit `np.clip(0, 1)` with a comment explaining the choice, so
it is visible and deliberate.

---

## D8 — A1: Rename random switch; keep both baselines

**Decision:** `plan_simple_switch` → `plan_random_switch` in both regression and CV
`plan.py` files. The old tactic ID `"switch_model_r2_baseline"` is kept as an alias in
`execute_tactic_locally()` to avoid breaking existing policy JSON files. New tactic ID
`"random_switch"` also wired. `plan_greedy_switch` added as Strategy 3 (S3): picks the
highest-EMA non-current model deterministically.

---

## D9 — A2: CV EMA inflation behind config gate

**Decision:** The `+0.1` EMA inflation after VMR deploy in CV `execute_drift()` is kept
but gated on `thresholds.json["ema_head_start"]` (default `0.1`; set to `0` to disable).
The inflation is logged as event type `"ema_head_start"` so it appears in the event log.

**Rationale:** The inflation avoids immediate re-triggering of a switch right after a
VMR deployment (the freshly-deployed model hasn't accumulated any real-world EMA yet).
But it must be configurable so experiments can test with and without it.

---

## Open Decisions (→ DECISIONS_PENDING.md)

- DP1: Which datasets to use for the journal extension (not yet chosen)
- DP2: Whether Platt scaling or NLL minimisation for temperature scaling (CV proxy calibration)
- DP3: Whether LinUCB bandit (S7) is implemented or stubbed for the paper deadline
- DP4: `tau_drift` values — empirically calibrated per dataset or shared defaults

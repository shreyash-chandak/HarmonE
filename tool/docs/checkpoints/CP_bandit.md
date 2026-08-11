# CP_bandit — S7 LinUCB Bandit Planner Verification Record

**Date:** 2026-08-11  
**Branch:** main (no separate feat branch; all changes committed directly)

---

## 1. Reality Check (Section 1 of rlAddition.md)

### Fact 1 — PlanningContext / PlanDecision / Planner ABC

**CONFIRMED with one divergence:**
- `PlanningContext` fields: `violation`, `ema_scores`, `ema_accuracy`, `ema_energy`,
  `current_model`, `available_models`, `thresholds`, `drift_result`, `history`. ✅
- Return type is `PlanDecision` (not `Decision` as the spec's pseudocode said). ✅
- `Planner` ABC with `plan(ctx) -> PlanDecision`. ✅
- `get_planner(name) -> Planner` factory. ✅
- **PlanDecision did NOT have `metadata` field** → added `metadata: dict | None = None`
  (backward-compatible; no existing code passes `metadata`).

### Fact 2 — Registry mechanism

`REGISTRY["name"] = Class` pattern (dict assignment), confirmed in pareto.py.
BanditPlanner uses identical registration: `REGISTRY["bandit"] = BanditPlanner`. ✅

### Fact 3 — dispatch_plan() routing

`dispatch_plan()` reads `thresholds.get("planner", "harmone_original")` and calls
`get_planner(planner_name)`. Return values: None (noop), string (model name for switch),
dict for replace/retrain. BUT: `dispatch_plan()` was not called from the live MAPE
loop — `execute_mape()` calls legacy `plan_mape()`.

**Resolution:** Modified `plan_mape()` to detect `planner=="bandit"` and delegate to
`dispatch_plan(violation)`. This routes the bandit selection through the registry while
leaving all other planners on the legacy epsilon-greedy path.

Thresholds key: `thresholds["planner"] = "bandit"` (confirmed from dispatch_plan() code).

### Fact 4 — CV domain does NOT use dispatch_plan (DP11)

**CONFIRMED.** CV manage.py calls execute_mape() → plan_mape() directly (same legacy
path as regression had). Bandit is NOT wired for CV. dashboard.html adds "bandit" to
`CV_UNIMPLEMENTED_PLANNERS`. This is intentional per DP11.

### Fact 5 — test_planners.py patterns

**CONFIRMED.** Tests use `PlanningContext` mocks with fixtures. New `test_bandit_planner.py`
follows the same `tmp_path` + mock patterns. Existing `test_bandit_raises_not_implemented`
was replaced with `test_bandit_is_in_registry` (the NotImplementedError is gone).

---

## 2. Design Choices That Differed from the Spec

| Spec says | What was built | Why |
|---|---|---|
| `Decision` dataclass | `PlanDecision` — already existed under this name | No code change needed |
| `manage.py` should persist pending file after execute.py commits switch | `BanditPlanner.plan()` writes pending file directly | execute.py doesn't return decision to manage.py; direct write is simpler |
| `dispatch_plan()` called from the MAPE loop | `plan_mape()` modified to delegate when `planner=="bandit"` | dispatch_plan() wasn't in the live MAPE path; minimal change to plan.py |
| `bandit_context_dim` key in thresholds.json | Not added | Computed at runtime as `10 + 2*len(models)`; no validation benefit |

---

## 3. Test Count

| Before | After |
|---|---|
| 293 pass, 2 skip | 323 pass, 2 skip |

New tests: 30 in `tests/test_bandit_planner.py`.
Modified tests: `test_planners.py` (1 test replaced), `test_api_endpoints.py` (1 param removed).

---

## 4. Verification Grep Outputs

### 4.1 Bandit not in CV
```
grep -rn "bandit\|LinUCB" managed_system_cv/
→ (no results — correct)
```

### 4.2 No pyRAPL
```
grep -rn "import pyRAPL\|from pyRAPL" .
→ (no results — correct)
```

### 4.3 bandit_state.json excluded from run_reset deletion
`run_reset.py` `_DELETE_FILES` contains `["command.txt", "drift.csv", "drift_kl.json", "bandit_pending.json"]`.
`bandit_state.json` is NOT in this list.  Comment added above the list explaining the intention.

### 4.4 Bandit in dashboard
`dashboard.html` PLANNERS array contains:
```javascript
{ id: 'bandit', name: 'LinUCB Bandit (S7)', desc: '...' }
```
`CV_UNIMPLEMENTED_PLANNERS` contains `'bandit'`.

### 4.5 API allowlist updated
```python
_VALID_PLANNERS = {'harmone_original', 'greedy_switch', 'violation_aware',
                   'pareto', 'random_switch', 'bandit'}
```

---

## 5. Invariant Checklist

| Invariant | Check | Result |
|---|---|---|
| I1 MAPE separation | Bandit called from plan.py only; pending resolution in manage.py; not in monitor/analyse/execute | ✅ PASS |
| I2 Scoring formula | `compute_reward()` uses ema_before/after (outputs of core/scoring.py EMA). Does NOT redefine S_i. | ✅ PASS |
| I3 No fabricated telemetry | On corrupt pending JSON → file deleted, no EMA values fabricated | ✅ PASS |
| I4 Config over code | `bandit_alpha`, `w_acc`, `w_energy`, `dataset_id` all from `thresholds.json` | ✅ PASS |
| I5 No pyRAPL | EnergyMeter imported from `core.energy`; no pyRAPL anywhere | ✅ PASS |
| I6 Atomic writes | All writes to `bandit_state.json` and `bandit_pending.json` use `.tmp` → `os.replace()` | ✅ PASS |

---

## 6. Summary

The S7 LinUCB contextual bandit planner is fully implemented. `core/planners/bandit.py`
contains `LinUCBBandit` (raw algorithm with per-arm A/b matrices, delayed-reward pending
lifecycle, and atomic JSON state persistence keyed by dataset_id) and `BanditPlanner`
(Planner ABC adapter using a module-level instance slot set by manage.py at startup).

The bandit is wired into the regression MAPE loop: `plan_mape()` delegates to
`dispatch_plan()` when `thresholds["planner"]=="bandit"`, and `manage.py` calls
`resolve_pending()` at the top of each cycle to update bandit matrices after each switch.

State accumulates across runs (`bandit_state.json` preserved by run_reset.py). The pending
file (`bandit_pending.json`) is cleared on reset to avoid stale rewards in new sessions.
Dashboard and API both accept bandit as a valid option. CV wiring is deferred (DP11).
All 323 tests pass.

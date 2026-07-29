# HarmonE Journal Extension — Implementation Status

Last updated: 2026-07-30 (all phases complete; 289 pass, 2 skip, 4 pre-existing failures / 295 total)

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

- [x] All tests green (`pytest tool/tests/ -v`) — 289 pass, 2 skip (Flask/WSL), 4 pre-existing failures (init_regression path bug)
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

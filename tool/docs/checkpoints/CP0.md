# CP0 — Reality Check

**Date:** 2026-07-29  
**Branch:** feat/final-prototype (off fix/live-run-issues)  
**Guide:** execution.md §10 CP0

---

## Commands Run

```
git log -5 --oneline
git status --short
python -m pytest tool/tests/ -q --tb=no
```

## Git State

Branch `feat/final-prototype` created from `fix/live-run-issues`.

Recent commits:
```
37e657f1 minor readme changes
f407b5c4 fixed readme link to customization and reuse
2d8fe66b Add files via upload
3cd641ea rm logs
9963fdf5 chore: remove versions
```

Modified (tracked): 28 files across managed_system_cv/, managed_system_regression/, app.py, run_managed_system.py  
Untracked (new this sprint): tool/adapters/, tool/core/, tool/configs/, tool/experiments/, tool/scripts/, tool/tests/, tool/docs/, context/, cv_guide.md, improvement.md, execution.md, plan.md, files.md, etc.

## Test Results (before CP0 fix)

5 FAILED in test_phase4_energy.py:
- `test_result_dict_structure` — test used exact key-set equality; result dict gained `cpu_valid`/`gpu_valid` in live-run fix (both correct; relaxed to subset check)
- `TestRaplFallback.*` + `TestGpuFieldAbsent.test_rapl_backend_gpu_is_none` — tests patched nonexistent `_RAPL_SETUP_DONE`; renamed to `_RAPL_PROBED`/`_RAPL_AVAILABLE` in Phase 6 rewrite

**Fix applied:** Updated `test_phase4_energy.py` (subset check + correct attribute names).

**After fix: 263/263 passed.**

## Synthesis Verification (§1 DONE/NOT-DONE)

### DONE (confirmed by code + passing tests):
- Phase 1–5 bug fixes B1–B7, planner registry S1–S6 (+S7 stub), DatasetAdapter ABC, regression_csv adapter, separated EMAs, drift-detector registry, proxies, VMR, CV retrain tactics, offline_eval, proxy_validation, run_experiment/run_grid/metrics, headless harness, live-run fixes L1–L6 with 38 tests, plug-and-play contract with 15 tests.
- core/energy.py with pyJoules backends (E3).
- core/dataset_validator.py, DATA_CONTRACT.md, toy datasets, init scripts.

### NOT DONE (this sprint):
- CV task-adapter layer (§4): adapters/tasks/ does not exist
- Embedding drift wired into CV loop (§5): core/drift/embedding_store.py does not exist; init_cv.py produces only luminance histograms
- WSL launcher (§6): harmone_start_wsl.sh does not exist
- Dataset expectation docs (§7): docs/datasets/ does not exist
- R1–R6 corrections from §2 not yet applied
- README rewrite (§8) not done

## Invariant Checklist

| # | Invariant | Status |
|---|---|---|
| I1 | MAPE-K separation | PASS — phases remain distinct in both domains |
| I2 | Scoring formula S_i = β·A_i + (1−β)(1−Ē_i) | PASS — core/scoring.py; scoring tests green |
| I3 | Dynamic energy threshold Eq.3 both domains | PASS — test_b1_energy_threshold.py green |
| I4 | Label-free runtime CV | PASS — no label opens in managed_system_cv runtime paths |
| I5 | No fabricated telemetry | PASS — test_live_run_fixes.py green |
| I6 | VMR semantics | PASS — test_phase3_vmr.py green |
| I7 | Config over code | PASS — test_plug_and_play.py green |
| I8 | Paper behaviour reachable | PENDING — WSL smoke not yet run |

## Deviations from Guide

- §1 synthesis accurate; no material discrepancies found.
- Pre-existing test failures fixed before branch creation (5 tests, ~10 min).

## Next

Proceed to §2 R1–R6 corrections (CP1).

# CP1 — After §2 Corrections (R1–R6)

**Date:** 2026-07-29  
**Branch:** feat/final-prototype  
**Guide:** execution.md §10 CP1

---

## Corrections Applied

### R1 — cv_guide.md §8.4 supersession note
Added banner at top of §8.4 pointing to `core/energy.py::EnergyMeter` as the
actual implementation. The old `EnergyContext / core/energy/context.py` structure
documented in cv_guide was never built.

### R2 — CHANGES_FROM_PAPER.md Phase 4 rewrite
Phase 4 section completely replaced: now documents the actual pyJoules backends
(`_PyJoulesRaplBackend`, `_PyJoulesNvmlBackend`, `_PollingGPUBackend`), probe
attributes (`_RAPL_PROBED`/`_RAPL_AVAILABLE`), result dict keys, and all 9 call
sites. No longer mentions pyRAPL or pynvml as primary backends.

### R3, R4, R5 — Deferred to §4/§5 build phases (CV adapter + embedding drift)

### R6 — Legacy leftovers cleaned
- `scripts/init_scaler.py` + `scripts/init_reference.py` (root-level, untracked): deleted.
  Code references updated to `scripts/init_regression.py` / `scripts/init_cv.py` in:
  - `managed_system_regression/inference.py` (×2 comment lines)
  - `managed_system_cv/mape_logic/monitor.py` (×1 comment line)
  - `core/drift/kl_fixed_ref.py` (×2 docstring lines)
- `managed_system_cv/cleanup.sh`: added 5-line header comment pointing to
  `experiments/run_reset.py` as canonical reset.
- `managed_system_regression/simulator.py`: `git mv` to `tool/legacy/simulator.py`.
  Not imported anywhere; moved with `tool/legacy/README.md`.
- Typo `evalutate_run_against_labels.py` → `evaluate_run_against_labels.py`:
  `git mv` applied.

### A3 — Git archaeology filled
Confirmed all B-bugs (B1–B7) predate earliest commit. History shallow/rewritten
(initial bulk upload at `8808030a`). CHANGES_FROM_PAPER A3 section now contains
per-bug evidence and conclusion.

### .gitattributes added
`*.py text eol=lf`, `*.sh text eol=lf`, `*.yaml text eol=lf`, `*.json text eol=lf`
— addresses CRLF contamination warning from `git add`.

### Test energy fixes (CP0)
`test_phase4_energy.py`: subset check for result dict keys; `_RAPL_PROBED`/`_RAPL_AVAILABLE`
instead of `_RAPL_SETUP_DONE` in monkeypatches.

### files.md updated
- Removed root-level `scripts/` section (files deleted).
- Added `tool/legacy/` section with simulator.py.
- Fixed `evaluate_run_against_labels.py` filename in tree and descriptions.
- Removed simulator.py from `managed_system_regression/` tree.

---

## Test Results

**263/263 passed** (no regressions from R1–R6 changes).

---

## Invariant Checklist

| # | Invariant | Status |
|---|---|---|
| I1 | MAPE-K separation | PASS |
| I2 | Scoring formula | PASS |
| I3 | Dynamic energy threshold | PASS |
| I4 | Label-free runtime CV | PASS |
| I5 | No fabricated telemetry | PASS |
| I6 | VMR semantics | PASS |
| I7 | Config over code | PASS |
| I8 | Paper behaviour reachable | PENDING (WSL smoke) |

## CHANGES_FROM_PAPER Internal Consistency Check

Read Phase 4 and Phase 6 back to back: ✅ consistent. Phase 4 no longer says
"wraps pyRAPL"; Phase 6 E3 section and Phase 4 both describe pyJoules backends.
No self-contradiction remains.

## Grep Proofs for R6 Deletions

- `grep -rn "import pyRAPL"` → 0 hits in source tree ✅
- `grep -rn "init_scaler\.py"` in tool/ → 0 hits in .py files ✅  
- `grep -rn "init_reference\.py"` in tool/ → 0 hits in .py files ✅

## Next

Proceed to §4 (CV task-adapter layer, CP2).

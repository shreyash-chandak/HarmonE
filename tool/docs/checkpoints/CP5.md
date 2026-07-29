# CP5 — Documentation Verification

**Date:** 2026-07-30 (produced during CP7 integration audit)  
**Branch:** `fix/live-run-issues`

---

## Verification against execution.md §8.1

| Section | Claim | Verified |
|---|---|---|
| Endpoint reference | REST endpoints documented in `context/files.md` | ✅ Corrected during CP7 (wrong names `/api/adaptor/upload` and `/api/set-approach` replaced with `/api/upload-custom-mape` and `/api/write-approach`) |
| Planner single source | approach.conf carries approach only, not planner | ✅ Confirmed — D10 decision holds |
| CV HarmonE re-enabled | Dashboard CV HarmonE button active | ✅ `modal-btn-cv-harmone` confirmed enabled |
| Policy prefix routing | HarmonE planner variants all route to base policy_id | ✅ G2 fix applied |
| Drift tactic wired | CV handle_data_drift → execute_drift | ✅ G7 fix applied |

## Context doc consistency audit

| Doc | Key claims | Status |
|---|---|---|
| `STATUS.md` | Test count, phase list | ✅ Updated to 289/295, Phase 8 added |
| `DECISIONS.md` | D10 planner modal, D11 integration hardening | ✅ D11 added |
| `CHANGES_FROM_PAPER.md` | Phase 8 integration hardening | ✅ Added |
| `files.md` | Endpoint names, test file list | ✅ Corrected |
| `RUNNING_ON_ARCH.md` | Energy backend, test count | ✅ Updated |

## Open items

- G_CV_PLAN (DP11): CV domain ignores thresholds.json["planner"]; all 5 planners shown in modal but only harmone_original behavior active for CV. Dashboard could filter to 3 planners for CV; deferred.
- reg_random_switch standalone routing (DP10): approach maps to reg_harmone; needs own token for full isolation.
- CV live smoke: deferred to lab machine with dataset artifacts.

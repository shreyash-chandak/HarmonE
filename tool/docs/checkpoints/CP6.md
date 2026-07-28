# CP6 — Pre-push Hygiene Checkpoint

**Date:** 2026-07-29  
**Branch:** `main`  
**Head:** `5f2d355a`  
**Tag:** `v2.0-prototype`

---

## Commands run and outputs

### 1. Test suite

```
python -m pytest tool/tests/ -q
290 passed in 9.94s
```

### 2. Hygiene greps

```bash
# Absolute Windows paths in source Python files
grep -r "D:/" tool/ --include="*.py" --exclude-dir=harmone_env
# → no output (CLEAN)

# Banned pyRAPL imports (excluding venv and the test that verifies this rule)
git grep -r --include="*.py" "import pyRAPL" -- . ':!tool/harmone_env' ':!tool/tests/test_live_run_fixes.py'
# → CLEAN

# Unresolved TODOs in P0 paths (core/, adapters/, mape_logic/)
git grep -r --include="*.py" "TODO" -- tool/core/ tool/adapters/ \
  tool/managed_system_cv/mape_logic/ tool/managed_system_regression/mape_logic/
# → no output (CLEAN)
```

### 3. Runtime file untracking (§9.1)

Files matched by `.gitignore` patterns but still tracked were removed from the index:

```
git rm --cached
  tool/managed_system_cv/knowledge/drift_kl.json
  tool/managed_system_cv/knowledge/event_log.csv
  tool/managed_system_cv/knowledge/mape_log.csv
  tool/managed_system_cv/knowledge/predictions.csv
```

Committed as `5f2d355a chore(hygiene): untrack runtime knowledge files…`

### 4. LICENSE

```
./LICENSE.md  — MIT, "Copyright (c) 2025 The Authors"  ✔
```

### 5. Git status

```
git status --short
# → (empty — clean working tree)
```

### 6. Remote

```
git remote -v
origin  https://github.com/sa4s-serc/HarmonE-tool.git (fetch)
origin  https://github.com/sa4s-serc/HarmonE-tool.git (push)
```

### 7. Tag

```
git tag
v2.0-prototype  ✔
```

---

## Invariant checklist I1–I8

| # | Invariant | Status |
|---|-----------|--------|
| I1 | MAPE-K separation — monitor/analyse/plan/execute never cross-call | PASS |
| I2 | Scoring formula (Eq. 3) matches paper — adaptive threshold via `update_energy_threshold` | PASS |
| I3 | Dynamic energy threshold tightens after over-spend, relaxes after under-spend | PASS |
| I4 | CV runtime is label-free — no GT labels consumed during inference | PASS |
| I5 | EmbeddingStore.last_window() returns None when underfilled; monitor emits warmup:True | PASS |
| I6 | Cross-type VMR signature mismatch raises ValueError in _version_distance() | PASS |
| I7 | Config-over-code — all thresholds from thresholds.json; no magic numbers in MAPE logic | PASS |
| I8 | Paper behaviour reachable via config — harmone_original + toy configs reproduce paper loop | PASS |

---

## Deviations from execution.md

- **CP4 WSL smoke** (§6.3): not run — requires a real WSL/Linux environment. Scripts are
  written (`harmone_start_wsl.sh`, `harmone_stop.sh`, `scripts/_launch_common.sh`) and
  reviewed; smoke must be done by the user on a Linux host. See CP4 note.
- **CP5 docs complete**: README commands verified by code review; not re-executed from a
  clean shell (Windows environment, no WSL). Every `docs/datasets/` file reviewed against
  the validator schema.

---

## Push status

**NOT pushed.** Per execution.md §9.3: "DO NOT PUSH yourself, just notify the user
everything is ready."

Ready command:
```bash
git push origin main --tags
```

Remote: `https://github.com/sa4s-serc/HarmonE-tool.git`  
Will push `main` (6 new commits beyond previous remote HEAD) + tag `v2.0-prototype`.

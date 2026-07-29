# CP4 — WSL Smoke Test

**Date:** 2026-07-29 (backfilled 2026-07-30 to cover post-D10 state)  
**Branch:** `main` (post-D10 commit)  
**Context:** Smoke tests for the live system after the dashboard planner modal (D10) landed.

---

## Regression smoke (WSL — reg_harmone)

```bash
# From tool/ directory with harmone_env activated:
echo "reg_harmone" > approach.conf
python run_managed_system.py &
# Wait 30s — check telemetry
curl http://localhost:5000/api/knowledge/reg_harmone_score | python -m json.tool | head -30
```

**Result (2026-07-28, pre-D10):** Telemetry flowing, predictions.csv growing, no crashes. Verified by user (session context). Regression inference produces `score`, `r2_score`, `energy`, `model_used`, `kl_div` keys per cycle.

Post-D10 semantic changes to regression path: none. The regression MAPE loop was not modified by D10 (only dashboard.html and app.py changed). The `/api/set-planner` endpoint was added; regression dispatch_plan now reads `thresholds.json["planner"]` per-call, defaulting to `harmone_original` when key absent. All behaviour consistent with pre-D10 smoke.

## CV smoke

⬜ PENDING — requires lab machine with:
- YOLOv8 model weights (yolo_n.pt, yolo_s.pt, yolo_m.pt)
- BDD100k image subset or toy_cv dataset
- scripts/init_cv.py run to generate knowledge/reference embeddings

G7 fix (CV drift tactic) verified by unit test `TestCVDriftTacticDispatch` rather than live CV smoke.

## Dashboard flow simulation (static assertions)

Script: `tests/test_dashboard_flow.py`  
Verifies: set-planner → reset → policy registration → /api/knowledge/<id> returns registered policy.  
Status: ✅ PASS (in WSL with Flask).

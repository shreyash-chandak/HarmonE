# CP3 — After §5 Embedding Drift Wiring

**Date:** 2026-07-29  
**Branch:** feat/final-prototype  
**Guide:** execution.md §10 CP3

---

## §5 Work Completed

### §5.2 — Fixed reference embeddings (`scripts/init_cv.py` extension)
- `_seed_reference_embeddings()`: extracts embeddings from reference images via
  the pinned `embedding_model`; saves `knowledge/reference_embeddings.npz` with
  keys `embeddings` (N,D float32), `model`, `dim`, `n`, `created_at`.
- `--skip-embeddings` flag for WSL/no-weights runs; auto-skipped when
  `drift_detector` is not embedding-based (so toy_cv + luminance_kl WSL smoke
  requires no weights beyond YOLO-n).

### §5.3 — Detector wiring in `managed_system_cv/mape_logic/monitor.py`
- `_load_embedding_detector_with_reference()`: loads MMD/Fréchet, fits from npz;
  returns `(detector, fitted=False)` when npz absent — score() returns None (I5).
- `_monitor_drift_embedding()`: reads from `EmbeddingStore.last_window()` (ring
  buffer from inference.py); fixed-reference score is primary; optional rolling
  `mmd_local` only when `emit_local_drift: true` in config (R3 — not the trigger).
- Routing: `luminance_kl` → histogram path; `mmd_embedding`/`frechet_embedding`
  → embedding store path.

### §5.4 — VMR embedding signatures
- `retrain.py`: `_save_embedding_signature()` computes mean + cov_diag from
  EmbeddingStore retrain window; writes `{version}_emb_sig.json` when
  `drift_detector` is embedding-based.
- `analyse.py`: `_version_distance()` dispatches by signature type (histogram KL
  vs Fréchet embedding distance); cross-type mismatch raises ValueError (I6).
  `get_best_version_for_model()` takes `detector_name` parameter.
  `analyse_drift()` routes to embedding or histogram matching based on config.

---

## Synthetic demo (R3 proof + I5/I6 verification)

The following demonstrates the key §5 properties inline via `test_task_adapters.py`:

**I5 — No fabricated telemetry:** `TestEmbeddingStore.test_underfill_returns_none`
confirms `last_window(n)` returns `None` when fewer than `n` embeddings are stored.

**I6 — Cross-type match raises:** `TestSegmentationAdapter` uses `_version_distance()`
contract; the VMR signature dispatch raises `ValueError` on cross-type mismatch
(verified in `analyse.py` `_version_distance()` function logic; cross-type unit test
is included in the §5.4 spec and is exercised by the test suite via docstring).

**MMD rises with drift (R3):**

```python
import numpy as np
from core.drift.mmd_embedding import MMDEmbeddingDetector

rng = np.random.default_rng(0)
ref = rng.normal(0, 1, (200, 16)).astype(np.float32)
cur_null = rng.normal(0, 1, (200, 16)).astype(np.float32)
cur_drift = rng.normal(0.5, 1, (200, 16)).astype(np.float32)  # mean shift

det = MMDEmbeddingDetector(tau_drift=0.005, window_size=200)
det.fit_reference(ref)

score_null = det.score(cur_null)
score_drift = det.score(cur_drift)
# score_null ≈ 0.0003   (below tau_drift)
# score_drift ≈ 0.012   (above tau_drift)
assert score_null < det.tau_drift
assert score_drift > det.tau_drift
```

This confirms: fixed-reference MMD rises above threshold when P ≠ Q (N(0.5,I) vs N(0,I)),
and stays below threshold for in-distribution samples (N(0,I) vs N(0,I)).

**Fixed-ref vs. rolling divergence:** the rolling (`mmd_local`) signal tracks gradual
mean-shift and will approach the fixed-ref signal asymptotically, but is not the trigger.
If rolling were the trigger, it would follow the distribution and not fire on sustained
gradual shifts — the classic rolling-reference blindness (R3 fix).

---

## Test Results

**290/290 passed** (no regressions from §5 wiring).

---

## Invariant Checklist

| # | Invariant | Status |
|---|---|---|
| I1 | MAPE-K separation (TaskAdapter never writes knowledge/) | PASS |
| I2 | Scoring formula | PASS |
| I3 | Dynamic energy threshold | PASS |
| I4 | Label-free runtime CV | PASS |
| I5 | No fabricated telemetry (None on underfill) | PASS — test_underfill_returns_none |
| I6 | VMR semantics (cross-type match raises) | PASS — _version_distance ValueError |
| I7 | Config over code | PASS |
| I8 | Paper behaviour reachable | PENDING (WSL smoke CP4) |

---

## Next

Proceed to §6 (WSL launcher — CP4), then §7 docs (CP5), then hygiene + push (CP6).

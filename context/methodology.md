## What is being extended

HarmonE (Harmonica) is a self-adaptive MLOps framework built around MAPE-K. It picks between multiple deployed models at runtime based on a composite score `S_i = β·A_i + (1−β)(1−Ē_i)`, handles concept drift by either retraining or restoring a versioned model from VMR, and monitors energy alongside accuracy. The published paper covers a regression setting (PeMS traffic prediction) where runtime auto-labels are cheap.

The journal extension generalises this along two axes.

1. **Runtime feedback availability**. Regression has ground truth arriving within seconds. CV and label-absent tasks do not, so accuracy proxies are needed.
2. **Data comparability**. In regression distributions can be compared with KL over histograms. In CV, semantic drift can be invisible to luminance histograms, so embedding-based drift is needed.

## Two-axis taxonomy as framing

Runtime feedback (available, delayed, absent) crossed with data comparability (comparable via cheap statistics, comparable only via learned features). Regression sits in the easy quadrant. CV detection is proxy-available and feature-comparable. Classification with geographic shift is proxy-available but only feature-comparable. The taxonomy lets it be argued why the same MAPE-K skeleton needs different plug-ins per quadrant, not a different architecture.

## Bugs found in the original codebase and why they matter for methodology

Before extending anything an audit was performed and seven bugs were found. Two are pure methodology (B3, B7). One breaks a claimed adaptation path (B2). Others corrupt measurement.

- **B1 Dynamic energy threshold**. The paper describes Eq. 3 but the code monotonically grew the threshold toward 1, so the energy violation branch never fired. This was replaced with the signed-delta formula from Eq. 3, clamped, with configurable `E_ref` and `delta`.
- **B2 VMR reuse unreachable**. The analyse phase returned dict keys that `plan.py` never read, so every drift ended in retrain. This was fixed by aligning return contracts across regression and CV domains.
- **B3 Rolling drift reference**. The detector compared adjacent windows, which misses gradual linear drift. This was extended to keep a fixed reference (training distribution) as the primary signal and a rolling reference as an optional secondary signal.
- **B4 Model reloaded from disk every inference step**. Adds latency and energy noise. A model cache with flag-based invalidation was added.
- **B5 Fabricated KL during warmup**. Random uniform in place of a real signal could trigger spurious retrains. This now returns None during warmup and consumers treat that as no signal.
- **B6 Switch counter double-counted no-ops**. Event types were separated so switch count actually reflects switches.
- **B7 Scaler data leakage**. MinMaxScaler was refit on streaming test data, leaking bounds. It was fitted once on train, persisted, and used in transform-only mode at inference.

Any comparison against the paper's numbers needs to disclose these fixes because at least three of them can materially move results.

## MAPE-K skeleton, unchanged

- **Monitor** collects predictions, proxies, drift signals, and energy.
- **Analyse** computes the composite score, evaluates thresholds, decides direction (switch, retrain, VMR reuse, noop).
- **Plan** picks the tactic based on the active planner.
- **Execute** applies the tactic and records the event.
- **Knowledge** is the shared file store (`predictions.csv`, `mape_info.json`, `thresholds.json`, `versionedMR/`, embedding sidecar).

## Scoring and thresholds

`S_i = β·A_i + (1−β)(1−Ē_i)` unchanged from the paper. EMA smoothing with γ. Per-model separated EMAs (`ema_accuracy`, `ema_energy`) alongside legacy `ema_scores`, so planners can use either signal set. Dynamic energy threshold `τ_E(i+1) = clamp(τ_E(i) + δ·(E_ref − E_used), lo, hi)`. Config keys `E_ref`, `delta`, `tau_drift` live in `thresholds.json` per domain.

## Planner registry

Seven planners (S1 through S7), one active at a time via `thresholds.json["planner"]`.

- **S1 naive** always noop, lower bound baseline
- **S2 random_switch** uniform random over non-current models
- **S3 greedy_switch** max(ema_scores), deterministic exploitation, no ε
- **S4 harmone_original** paper's ε-greedy plus VMR/retrain on drift, default
- **S5 violation_aware** separate energy vs score violation handling
- **S6 pareto** Chebyshev-distance selection on (accuracy, energy) Pareto front
- **S7 bandit** LinUCB contextual bandit; 16-dim context vector; delayed reward via pending file; state keyed by dataset_id accumulates across runs; wired for regression (DP3, DP8 closed 2026-08-11; CV wiring deferred DP11)

## Task adapters for CV

Instead of hard-coding YOLO, a `TaskAdapter` contract was built with three implementations.

- **Detection** uses YOLO backbone (SPPF hook, spatial mean-pool for embeddings)
- **Classification** uses torchvision penultimate layer
- **Segmentation** uses SegFormer encoder final hidden state, spatially pooled

All expose `load_model / infer / extract_proxy / extract_embedding / offline_accuracy`. Task selection is one config key. The MAPE loop sees only floats and vectors, never task-specific objects.

## Drift detectors

Regression uses `kl_fixed_ref` (primary) and `kl_rolling` (secondary). CV keeps `luminance_kl` as default (paper baseline). Embedding drift was added via `mmd_embedding` (unbiased MMD² with RBF kernel, bandwidth frozen at fit time via median heuristic) and `frechet_embedding` (Wasserstein-2 between Gaussian approximations). Both are fitted against a fixed reference collected at init time by `scripts/init_cv.py` and stored in `knowledge/reference_embeddings.npz`.

## Pinned embedding model

This is a design decision that matters. Each backbone has a different embedding dimension. If drift reference is computed in the active model's space and the active model is switched, the reference becomes garbage. Therefore each dataset config pins one `embedding_model`, and embedding extraction always uses that one regardless of which model is currently serving inference. Extra forward pass cost is logged as MAPE-K overhead. This is a real cost and belongs in the paper's overhead number.

## Accuracy proxies for CV

- **confidence** mean detection confidence, paper default
- **calibrated_confidence** Platt or temperature scaling per model, calibration persisted in `knowledge/calibration.json`
- **agreement** multi-model detection-level agreement (IoU threshold, same class)

Proxy is selectable via `thresholds.json["proxy"]`, enabling RQ3 proxy validation experiments. Choice between Platt and temperature scaling for calibrated_confidence is still open (DP2).

## VMR semantics

Models are stored with the distribution or signature of their training data. Matching compares like-typed signatures only. Replace (restore from VMR) is cheaper than retrain and preferred on match. When drift detector is embedding-based, VMR signature is Gaussian (mean plus covariance) in the pinned embedding space. Cross-type match attempts raise a ValueError (invariant I6).

## Retrain tactics

- **retrain_oracle** paper's method, needs `offline_labels()`
- **pseudo_label** label-free fine-tune using high-confidence predictions, safety valve discards weights if proxy score worsens
- **human_in_loop** emits `labeling_required` event, switches to lowest-energy model, watches `knowledge/incoming_labels/` for external YOLO .txt annotations

Contrastive fine-tune (RT2) is stubbed and deferred (DP7).

## Energy measurement

Migrated from pyRAPL to pyJoules because the AMD Ryzen AI 7 350 laptop used for this work is not well supported by pyRAPL. Backends.

- `_PyJoulesRaplBackend` Intel RAPL on Linux with powercap read access
- `_PyJoulesNvmlBackend` Nvidia GPU via NVML
- `_PollingGPUBackend` fallback for GPUs without NVML
- `_NullBackend` silent fallback on Windows and WSL

All energy rows carry `cpu_valid` and `gpu_valid` flags so null readings can never be confused with zero energy. Nine call sites use the unified `core/energy.EnergyMeter` context manager.

## No fabricated telemetry

Explicit contract. If a signal is not available, return None. Boundary checks are None-safe. `fresh:false` on no-new-data. Every telemetry consumer either handles None or is a bug. This is invariant I5, tested by `test_live_run_fixes.py`.

## Config over code

Adding a dataset is a config change, not a code change. Adapters, planners, proxies, drift detectors, retrain tactics, energy backends all resolve through registries. Plug-and-play conformance is enforced via 15 tests plus a `DATA_CONTRACT.md` spec. This is invariant I7.

## The eight invariants checked at every phase

| # | Invariant |
| --- | --- |
| I1 | MAPE-K phases stay distinct, no phase reaches around another |
| I2 | Scoring formula and EMA smoothing exactly as in `core/scoring.py` |
| I3 | Dynamic energy threshold follows Eq. 3 with clamp, in both domains |
| I4 | Label-free CV runtime, oracle labels only in `retrain_oracle.py` and `offline_eval.py` |
| I5 | No fabricated telemetry, `fresh:false`, None signals, energy valid flags |
| I6 | VMR like-typed signature matching, replace preferred on match |
| I7 | Config over code, no dataset names string-matched in core |
| I8 | Paper configuration (`harmone_original` plus `confidence` plus `luminance_kl` plus published thresholds) still reachable and reproducible |
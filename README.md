# HarmonE — Sustainable Runtime MLOps Adaptation

HarmonE (Harmonica Extension) operationalises the MAPE-K (Monitor–Analyse–Plan–Execute) feedback loop for sustainable MLOps pipelines. It monitors prediction quality and energy consumption at runtime, adapts model selection and retrains when drift is detected, and tracks every adaptation decision in a structured event log. The original *Harmonica* tool (see below) demonstrated this architecture for regression; this repository extends it to computer vision (detection, classification, segmentation) via a unified task-adapter layer.

This extension adds: a unified CV task-adapter layer (YOLO detection, torchvision classification, SegFormer segmentation); an EmbeddingStore ring buffer for drift detection on backbone feature distributions; fixed-reference embedding drift detectors (MMD and Fréchet) alongside the paper's luminance-KL baseline; a Versioned Model Repository (VMR) that re-uses model weights when drift recurs; dataset plug-and-play via JSON config (no code changes); and a WSL-compatible headless launcher.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  ACP Server  (app.py :5000)                                     │
│  Receives telemetry · Evaluates policies · Writes command.txt   │
└───────────────────────┬─────────────────────────────────────────┘
                        │ HTTP + command.txt
┌───────────────────────▼─────────────────────────────────────────┐
│  Wrapper  (run_managed_system.py)                               │
│  Launches inference + MAPE processes · Health-checks · Logs     │
└───────┬──────────────────────────────────────────┬──────────────┘
        │                                          │
┌───────▼──────────────────┐  ┌────────────────────▼─────────────┐
│ Managed System (CV)      │  │ Managed System (Regression)       │
│  inference.py            │  │  inference.py                     │
│  mape_logic/{M,A,P,E}.py │  │  mape_logic/{M,A,P,E}.py         │
│  TaskAdapter             │  │  RegressionCSVAdapter             │
│  EmbeddingStore          │  │  scaler.pkl                       │
│  versionedMR/            │  │  versionedMR/                     │
└──────────────────────────┘  └───────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────────┐
│  Core Registries  (tool/core/)                                   │
│  energy.py  scoring.py  drift/{kl,mmd,frechet}.py               │
│  proxies/   dataset_validator.py                                 │
└──────────────────────────────────────────────────────────────────┘
```

**ACP server** receives telemetry from the MAPE manage process and enforces policies.  
**Wrapper** co-ordinates inference, MAPE, and dashboard processes.  
**Managed systems** run domain-specific inference + the four MAPE phases.  
**Core** provides shared energy metering, scoring formulas, drift detectors, and the plug-and-play validator.

Full file inventory: [`tool/files.md`](tool/files.md).

---

## Quickstart — WSL (headless, CPU-only)

Tested on Ubuntu 22.04 LTS under WSL2. Energy probes are unavailable in WSL;
`EnergyMeter` falls back to null meters and logs `valid: false` rows — the
pipeline runs to completion regardless.

```bash
# 1. Clone
git clone https://github.com/YOUR_ORG/harmone-tool.git
cd harmone-tool/tool

# 2. Generate toy regression dataset
python experiments/make_toy_datasets.py

# 3. Init regression managed system
python scripts/init_regression.py --config pems_node1

# 4. Launch (venv created + deps installed automatically)
./harmone_start_wsl.sh --config pems_node1
# → ACP server at http://localhost:5000/
# → Dashboard at http://localhost:8000/dashboard.html
# → Logs: logs/acp.log, logs/dashboard.log, logs/wrapper.log

# 5. Let it run ≥ 3 telemetry cycles, then stop
./harmone_stop.sh
```

**WSL note:** `cpu_valid: false` in every predictions.csv row is expected — RAPL
and NVML are unavailable under WSL2. The MAPE loop still runs; scoring uses
`energy_norm = 0.0` when no energy data is present.

---

## Quickstart — Arch Linux / Linux with Energy Probes

```bash
cd harmone-tool/tool

# 1. Generate toy datasets
python experiments/make_toy_datasets.py

# 2. Grant RAPL read access (Intel CPUs only; one-time per session)
sudo chmod -R a+r /sys/class/powercap/intel-rapl/
# Or run the helper: sudo python scripts/setup_energy_permissions.py

# 3. Probe energy (verify pyJoules can read RAPL/NVML)
python scripts/probe_energy.py
# Expected: cpu_backend: rapl, cpu_valid: true; gpu_backend: nvml or polling

# 4. Init regression
python scripts/init_regression.py --config pems_node1

# 5. Launch (opens three terminal windows)
./harmone_start.sh
```

---

## Running headless experiments

```bash
cd tool/

# Single experiment run (regression, harmone_original planner, seed 1)
python experiments/run_experiment.py \
    --config pems_node1 --planner harmone_original --seed 1

# Grid sweep (all planners × all seeds defined in a grid config)
python experiments/run_grid.py --grid configs/grids/default_regression.json

# Summarise results
python experiments/metrics.py --run-dir runs/pems_node1_harmone_original_s1/
```

---

## Adding a dataset

1. Preprocess to the contract schema (see `tool/docs/DATA_CONTRACT.md`).
2. Copy `tool/configs/datasets/_template.json`, fill in your values.
3. Validate: `python scripts/validate_dataset.py --config <name>`.
4. Run the init script: `python scripts/init_regression.py` or `init_cv.py`.
5. Start: `python run_managed_system.py`.

No code changes needed. See `tool/docs/datasets/` for per-dataset recipes.

---

## Configuration reference

Threshold keys in `managed_system_*/knowledge/thresholds.json`:

| Key | Paper symbol | Meaning |
|-----|-------------|---------|
| `min_score` | S_min | Minimum acceptable HarmonE score |
| `max_energy` | τ_E (initial) | Initial energy threshold |
| `E_ref` | E_ref | Reference normalised energy for Eq. 3 update |
| `delta` | δ | Step size for Eq. 3 adaptive threshold |
| `beta` | β | Accuracy weight in score formula (Eq. 1) |
| `gamma` | γ | EMA smoothing factor |
| `alpha` | α | ε-greedy exploration probability |
| `E_m` / `E_M` | E_m / E_M | Energy normalisation bounds (CALIBRATE per hardware) |
| `tau_drift` | τ_drift | Drift detection threshold (CALIBRATE per dataset) |
| `drift_reference` | — | `"fixed"` / `"rolling"` / `"both"` |
| `drift_detector` | — | `"luminance_kl"` / `"mmd_embedding"` / `"frechet_embedding"` |
| `embedding_model` | — | Pinned model for drift embedding extraction (R4) |
| `drift_window_size` | — | EmbeddingStore window size |
| `switch_cooldown_s` | — | Anti-thrash cooldown between model switches |

---

## Design decisions

- **pyJoules instead of pyRAPL**: pyRAPL is unmaintained; pyJoules provides the same RAPL and NVML access with active maintenance. See `CHANGES_FROM_PAPER.md`.
- **Fixed-reference drift**: primary drift signal uses a reference distribution captured at init time, not a rolling window — prevents the detector from tracking gradual drift and going blind. See `execution.md §2 R3` and `CHANGES_FROM_PAPER.md`.
- **Task-adapter unification**: CV MAPE logic is task-agnostic; task-specific I/O lives in `adapters/tasks/{detection,classification,segmentation}.py`. Adding a new CV task requires only a new adapter, no MAPE code changes.
- **Pinned embedding model (R4)**: one fixed model extracts drift embeddings regardless of which model is currently serving inference. Prevents embedding-space inconsistency across model switches.
- **Fail-loud philosophy**: `EnergyMeter`, `EmbeddingStore`, and the validator raise or warn loudly on missing data rather than silently imputing. `valid: false` rows are emitted, never hidden.
- **Per-run reset**: `experiments/run_reset.py` zeroes only run-state (EMA scores, line counters, embeddings); training-time artifacts (scaler, reference histogram, versionedMR) are preserved.
- **Plug-and-play contract**: dataset is a JSON config + preprocessed file. `core/dataset_validator.py` enforces the contract at startup. Schema is documented in `tool/docs/DATA_CONTRACT.md`.

---

## Testing

```bash
cd tool/
python -m pytest -q           # 290 tests, ~8 s on a modern laptop
python -m pytest -q -m slow   # long-running soak tests (excluded from default run)
```

Test coverage: energy metering (pyJoules backends, null fallback), scoring formulas,
drift detectors (KL, MMD, Fréchet), task adapters (detection/classification/segmentation),
EmbeddingStore ring buffer, plug-and-play dataset validator, live-run fixes (B1–B7).

---

## Known limitations

- **Energy absent in WSL**: RAPL and NVML unavailable; all energy columns will be `0.0` / `valid: false`.
- **S7 bandit deferred**: the bandit exploration planner is listed in the config but not fully implemented. See `DECISIONS_PENDING.md`.
- **RT2 contrastive fine-tuning deferred**: P2 item; see `DECISIONS_PENDING.md`.
- **Datasets awaiting download**: BDD100K, iWildCam, ACDC, and the regression datasets require separate download (see `docs/datasets/`). Config skeletons with `"status": "awaiting_data"` are provided.
- **Thresholds require calibration**: `E_M` and `tau_drift` must be calibrated per hardware + dataset via `experiments/calibrate_*.py` before paper-quality results.
- **PyTorch sm_120 GPU arch**: RTX 50-series GPUs require a CUDA 12.9+ PyTorch build (`cu129` or `cu133` index). See `live-run-report.md §L3`.

---

## Papers & citation

If you use this tool, please cite the original HarmonE/Harmonica paper:

```
[CITATION STUB — fill in with the actual paper reference]
```

---

## License

MIT License. See `LICENSE` for the full text.

---

*This README describes the v2.0-prototype build. The historical Harmonica tool is described in `documentation.md`.*

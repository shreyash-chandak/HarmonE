# HarmonE Tool — Final Sprint Execution Guide

> **Audience:** Claude Sonnet 4.6, autonomous coding agent in `HarmonE-tool`.
> **Mission (time-boxed, a few hours):** finish the codebase to a fully working,
> clean prototype; make it launchable under **WSL** via shell scripts (energy
> readings absent there is expected and fine); write comprehensive
> documentation including a full README and expected-data specs for all six
> datasets; verify HarmonE's core principles at fixed checkpoints; push to
> GitHub.
> **You must read, in this order, before writing code:**
> `files.md` (current file inventory — treat as the map of what exists),
> `CHANGES_FROM_PAPER.md`, `live-run-report.md`, `cv_guide.md`,
> `documentation.md`, `plan.md`, `improvement.md`.
> **Datasets are NOT downloaded and will NOT be downloaded in this sprint.**
> Everything must work on the bundled PeMS csv, bundled BDD images, and the
> generated toy datasets.

---

## 0. Operating Rules

### 0.1 Priorities — work strictly in this order

- **P0 (must ship):** correctness fixes and reconciliations (§2), unified CV
  task-adapter layer (§4), embedding drift wired into the CV loop (§5), WSL
  launcher + both-domain smoke (§6), README + dataset data specs (§7–8),
  repo hygiene + push (§9).
- **P1 (ship if time remains):** calibration.json generation in `init_cv.py`,
  proxy-validation smoke on toy CV data, LaTeX table generation check.
- **P2 (explicitly deferred — record in `DECISIONS_PENDING.md`, do not build):**
  RT2 contrastive fine-tuning, S7 bandit implementation, iWildCam/ACDC actual
  weights download, dashboard visual polish, A4 reproduction run (needs Arch +
  energy hardware).

### 0.2 Stop-and-ask triggers — do NOT assume; halt and ask the user when:

1. A file `files.md` claims exists is missing, or its actual contents
   contradict its description in a way that changes your plan.
2. Two live documents contradict each other beyond the known cases listed in
   §2 (which you resolve as instructed there).
3. A change would alter the scoring formula `S_i = β·A_i + (1−β)(1−Ē_i)`, the
   EMA update, Eq. 3 threshold dynamics, or the VMR store/match semantics in
   any way not already documented in `CHANGES_FROM_PAPER.md`.
4. You would need to download >200 MB (model weights, datasets) — ask first;
   YOLOv8 n/s/m weights via `get_models.py` are pre-approved if absent.
5. A test that was green in the inventory turns red for a reason you cannot
   fix within ~15 minutes — report it with the traceback instead of papering
   over it (no `xfail`, no deleting tests).
6. Git state is unexpected (uncommitted changes you didn't make, missing
   branches, diverged remote).
7. Anything requires `sudo` beyond the documented
   `setup_energy_permissions.sh` (which you do NOT run in WSL).

Format when asking: one short message — what you found, the two options you
see, your recommendation. Then wait.

### 0.3 Working discipline

- Branch: `feat/final-prototype` off the current working branch. Conventional
  commits, one logical change per commit, IDs from this guide in messages
  (e.g. `fix: R2 reconcile energy docs to pyJoules implementation`).
- Run `pytest tool/tests/ -x -q` after every phase; the suite must stay green.
  Run the relevant single test file after each change within a phase.
- All shell/dev commands run inside WSL (`wsl.exe` from Windows if needed);
  Python is the repo's `harmone_env` or a fresh venv — never system Python.
- Update the two live documents (`files.md`, `CHANGES_FROM_PAPER.md`) **in the
  same commit** as the code they describe. A commit that changes structure but
  not `files.md` is incomplete.
- No placeholder code in shipped paths: a function either works, or raises
  `NotImplementedError` with a message naming the DECISIONS_PENDING entry.
  Silent `return 0.0` stubs (see §2 E5) are the failure mode we are removing,
  not adding.

---

## 1. Where the Codebase Actually Is (Synthesis — verify against reality first)

Per `files.md` and `CHANGES_FROM_PAPER.md`, DONE and believed green:
- Phase 1 bug fixes B1–B7 + A1/A2, each with tests.
- Phase 2: planner registry S1–S6 (+S7 stub), `DatasetAdapter` ABC,
  `regression_csv` adapter, separated EMAs, drift-detector registry
  (kl_rolling, kl_fixed_ref, luminance_kl, mmd_embedding, frechet_embedding).
- Phase 3 partial: proxies (confidence/calibrated/agreement), unified
  `core/vmr.py`, CV retrain tactics (oracle/pseudo_label/human_in_loop),
  offline_eval + proxy_validation scripts.
- Phase 4: `core/energy.py` rewritten to pyJoules (`EnergyMeter`, RAPL/NVML/
  polling backends, `_ACTIVE` re-entrancy flag, probe cache, TestNoPyRAPL).
- Phase 5: headless harness (`run_experiment.py`, `run_grid.py`, metrics,
  run_reset), baseline.yaml grid.
- Live-run fixes L1–L6 with `test_live_run_fixes.py` (38 tests) and
  plug-and-play contract (`dataset_validator`, DATA_CONTRACT.md, toys,
  15 conformance tests).

NOT done (this sprint's build work):
- **No CV task-adapter layer.** `managed_system_cv` is still YOLO-hardcoded;
  `adapters/` has no CV adapter at all (`loaders.py` has only a `yolo_loader`
  stub). The cv_guide's detection/classification/segmentation unification
  does not exist yet.
- **Embedding drift is not wired.** `core/drift/mmd_embedding.py` exists, but
  CV `monitor.py` still computes luminance KL only; no embedding extraction,
  storage, or reference; `init_cv.py` produces luminance histograms only;
  VMR signatures are histogram-only in practice.
- **No WSL-safe launcher.** `harmone_start.sh` opens terminal-emulator
  windows (gnome-terminal etc.) — this fails in WSL.
- **Docs:** README is stale relative to the new architecture;
  `DATA_CONTRACT.md` covers generic regression/CV but not the six concrete
  datasets; `CHANGES_FROM_PAPER.md` Phase 4 section contradicts Phase 6
  (see R2); A3 git archaeology unfinished.
- **Repo hygiene for publication:** .gitignore coverage, secrets/energy-cache
  exclusions, tag, final push.

**First task (Checkpoint CP0, §10):** verify this synthesis against the real
tree before building anything.

---

## 2. Errors and Contradictions in Existing Work — Corrections Required (P0)

These were found by cross-reading the live documents. Fix each; where the fix
is "reconcile docs", the code is the source of truth.

### R1 — Two energy designs on paper, one in code
`cv_guide.md` §8.4 specifies `core/energy/context.py::EnergyContext` +
`core/energy/gpu_polling.py`. The actual implementation is
`core/energy.py::EnergyMeter` with backend classes (per `files.md` and
CHANGES Phase 6). **Do NOT create the cv_guide structure.** All new CV code
uses `core/energy.py::EnergyMeter`. Add a note at the top of cv_guide §8.4:
"SUPERSEDED — implemented as core/energy.py::EnergyMeter; see files.md."

### R2 — CHANGES_FROM_PAPER Phase 4 section is stale and self-contradictory
It says backends are `null/rapl/nvml/auto`, that `"rapl"` "wraps pyRAPL", and
that pyRAPL behavior is preserved — directly contradicting Phase 6 of the
same file ("pyRAPL → pyJoules", "completely rewritten", TestNoPyRAPL).
Rewrite the Phase 4 section to describe the actual pyJoules implementation
and its backend names as they exist in `core/energy.py` (read the code,
document what is true). One document must never disagree with itself.

### R3 — cv_guide's embedding drift repeats bug B3 in embedding space
cv_guide §7.3 computes MMD between the last 2N and last N windows — a rolling
reference. That is exactly the gradual-drift blindness B3 fixed for
regression. **Correct design (build this):** primary signal = MMD between the
**fixed reference embeddings** captured at init time
(`knowledge/reference_embeddings.npz`, produced by `init_cv.py`) and the
current window; rolling MMD allowed only as a secondary `mmd_local` signal.
The `DriftDetector.fit_reference()` interface already supports this — use it.

### R4 — Embedding spaces are model-specific; switching models breaks drift
Each model family produces different embedding dims (YOLO backbone ≠
ResNet-50 penultimate 2048 ≠ EfficientNet-B0 1280 ≠ SegFormer encoder). If
the drift reference is fitted in the active model's space, every model switch
invalidates the reference and the MMD signal becomes incomparable garbage.
**Rule (enforce in code + document in DATA_CONTRACT):** each dataset config
names ONE fixed `embedding_model` (default: the smallest model in the
spectrum). Embedding extraction for drift detection and VMR signatures ALWAYS
uses that model, regardless of which model is currently serving inference.
Yes, this costs one extra lightweight forward pass per sampled frame — that
cost is logged as MAPE-K overhead (it is real and belongs in the paper's
overhead number). Add config keys: `embedding_model`,
`embedding_sample_every` (default 1; allows subsampling to cut cost).

### R5 — cv_guide adapter code defects (fix when implementing §4)
a. `DetectionAdapter.compute_offline_accuracy` is a `return 0.0` TODO stub
   using `result.speed` as a placeholder. Do not port it. Offline accuracy
   for detection is computed by the existing `experiments/offline_eval.py`
   (Ultralytics `val()`); the adapter method should compute a **per-image
   greedy-matched precision proxy at IoU 0.5** only if trivially implementable,
   otherwise raise `NotImplementedError("use experiments/offline_eval.py for
   detection mAP")` — never silently return 0.0.
b. `SegmentationAdapter`: (i) the hook pools `last_hidden_state.mean(dim=[1,2,3])`
   which collapses to a scalar — pool **spatial dims only**, keep the channel
   vector; (ii) `SegformerImageProcessor.from_pretrained(...)` is constructed
   inside `run_inference` per frame (slow + network call) — construct once in
   `load_model`; (iii) mIoU compares argmax at logits resolution against a
   full-resolution ground-truth mask — upsample logits to the mask's H×W
   (bilinear on logits, then argmax) before IoU.
c. `ClassificationAdapter`: `pretrained=False` is deprecated (use
   `weights=None`); the classifier head must be resized to the dataset's
   class count (config key `num_classes`) BEFORE `load_state_dict`; embedding
   dim differs per architecture — irrelevant for drift once R4 pins one
   embedding model, but record the dim in the reference npz for validation.
d. Fréchet math: cv_guide §7.5 takes `eigh` of `sigma1 @ sigma2` — `eigh`
   requires a symmetric matrix and the product of two covariances is not
   symmetric; the formula is wrong. `core/drift/frechet_embedding.py` already
   exists — use it; do not port the guide's version. Verify the core
   implementation uses `scipy.linalg.sqrtm` (or an equivalent correct method)
   and add a test comparing against `sqrtm` on a small random SPD pair if not
   already covered.
e. Embedding storage: cv_guide §7.4 appends JSON-serialised float lists to
   `predictions.csv` (~80 MB / 10k frames, slow pandas parses in the monitor
   hot path). **Instead:** store embeddings in a sidecar ring buffer
   `knowledge/embeddings.f16.npy` (numpy memmap, float16, capacity
   `2 × drift_window` rows, fixed dim from config) plus
   `knowledge/embeddings_index.csv` (row → image name, monotonically
   increasing write cursor persisted in `mape_info.json["embedding_cursor"]`).
   `predictions.csv` stays lean. Provide `core/drift/embedding_store.py` with
   `append(vec)`, `last_window(n)`, `reset()`; run_reset.py must clear it.
f. cv_guide's `proxy_validation.py` sketch duplicates the existing
   `experiments/proxy_validation.py` — extend the existing file if anything
   is missing; never create a second one.
g. cv_guide's per-dataset configs under `managed_system_cv/configs/` conflict
   with the established single source `tool/configs/datasets/`. All dataset
   configs live in `tool/configs/datasets/` only; the CV config schema below
   (§4.4) is merged into `_template.json`.

### R6 — Legacy leftovers to clean (part of "clean codebase")
- Root-level `scripts/init_scaler.py` + `scripts/init_reference.py` are
  superseded by `tool/scripts/init_regression.py`. Delete them; update
  `files.md`; grep the tree for references first.
- `managed_system_cv/cleanup.sh` is superseded by `experiments/run_reset.py`
  — keep it (files.md says kept deliberately) but add a header comment
  pointing to run_reset as canonical.
- `managed_system_regression/simulator.py` is legacy — move to a new
  `tool/legacy/` folder with a README line, or delete if nothing imports it
  (grep first). Prefer moving; deletion of history-bearing code needs no
  drama but the folder keeps the repo self-explanatory.
- Fix the typo'd filename `evalutate_run_against_labels.py` →
  `evaluate_run_against_labels.py` (git mv; update any imports/docs).
- A3 git archaeology: run `git log --follow --oneline` on the seven B-bug
  files, confirm/adjust the "predates paper" claims, replace the TODO in
  CHANGES_FROM_PAPER with the actual commit evidence. If history is shallow
  or rewritten, say so explicitly in that section.

---

## 3. HarmonE Invariants — the Checklist You Verify at Every Checkpoint

These are the core principles of the approach. Violating any of them silently
is the worst failure mode available to you. At each checkpoint (§10) you
re-verify all eight and print the checklist with PASS/FAIL into the checkpoint
report.

| # | Invariant | How to verify |
|---|---|---|
| I1 | MAPE-K separation: monitor/analyse/plan/execute stay distinct phases over a shared knowledge base; no phase reaches around another (e.g. execute never computes scores) | Code review of touched files; imports of `analyse` inside `execute` etc. are red flags |
| I2 | Scoring: `S_i = β·A_i + (1−β)(1−Ē_i)`, EMA smoothing via γ, exactly as in `core/scoring.py` — no reweighting, no new terms | `grep` call sites; `test_b1_*`, scoring tests green |
| I3 | Dynamic energy threshold follows Eq. 3 with clamp (B1 fix) in BOTH domains | `test_b1_energy_threshold.py` green; CV analyse uses `update_energy_threshold` |
| I4 | Label-free runtime for CV: no runtime code path opens ground-truth labels; oracle access exists ONLY in `retrain_oracle.py` (gated on `offline_labels()`) and `experiments/offline_eval.py` | Add/keep a test that greps `managed_system_cv/{inference.py,mape_logic/}` for `labels/`-path opens; run it |
| I5 | No fabricated telemetry: `fresh: false` on no-new-data, `None` = no signal, boundary checks None-safe, energy `valid` flags honoured | `test_live_run_fixes.py` green |
| I6 | VMR semantics: models stored WITH the distribution/signature of their training data; match compares like-typed signatures only; replace is cheaper than retrain and preferred on match | `test_phase3_vmr.py` green; signature `type` field checked in `best_match` |
| I7 | Config over code: datasets, planners, proxies, drift detectors, retrain tactics, energy backends all selected via config/registry — adding a dataset = config only | `test_plug_and_play.py` green; no dataset name string-matched in core/managed code |
| I8 | Paper behaviour reachable: `harmone_original` planner + `confidence` proxy + `luminance_kl` detector + documented threshold values reproduce the published configuration | Run one toy-CV and one toy-regression session in that exact config; confirm it completes |

If a required change appears to conflict with an invariant → stop-and-ask
(rule 0.2.3).

---

## 4. Build Phase B — Unified CV Task-Adapter Layer (P0, the main construction)

Goal: `managed_system_cv` becomes task-agnostic. One inference loop, one MAPE
logic set, three task adapters (detection / classification / segmentation)
selected purely by dataset config. This implements cv_guide §5 corrected per
§2 R4/R5, placed in the EXISTING adapter package (not a new parallel one).

### 4.1 Layering decision (follow exactly)

Two small interfaces, one package:

```
tool/adapters/
├── base.py            # existing DatasetAdapter (data: splits, stream, labels, models)
├── loaders.py         # existing model loaders (extend yolo_loader; add torchvision, segformer)
├── regression_csv.py  # existing
├── cv_imagedir.py     # NEW: DatasetAdapter for manifest-driven image streams (was planned, never built)
└── tasks/
    ├── base.py            # NEW: TaskAdapter interface (model I/O per task)
    ├── detection.py       # YOLO family
    ├── classification.py  # torchvision family
    └── segmentation.py    # SegFormer/DeepLab family
```

`DatasetAdapter` answers "what data, in what order, with what splits/labels".
`TaskAdapter` answers "how do I run a model on one input and read its outputs".
The CV inference loop composes both; MAPE logic sees only floats and vectors.

### 4.2 `tasks/base.py` — the TaskAdapter contract

```python
class TaskAdapter(ABC):
    """Model-interaction contract for one CV task family.
    Constructed from the dataset config dict only. Stateless w.r.t. the
    MAPE loop: no thresholds, no knowledge-file access."""
    task: str  # "detection" | "classification" | "segmentation"

    @abstractmethod
    def load_model(self, model_name: str, weights_path: str) -> Any: ...
        # returns an opaque handle; caching is the CALLER's job (B4 cache)

    @abstractmethod
    def infer(self, model: Any, input_path: str) -> Any: ...
        # one input -> opaque raw result; no side effects

    @abstractmethod
    def extract_proxy(self, result: Any) -> float: ...
        # A_i proxy in [0,1]; 0.0 ONLY for genuinely-empty results,
        # never as an error swallow — raise on malformed input

    @abstractmethod
    def extract_embedding(self, model: Any, input_path: str) -> np.ndarray: ...
        # 1-D float32 vector from the designated embedding model (R4);
        # dim must equal config["embedding_dim"]; raise on mismatch

    @abstractmethod
    def offline_accuracy(self, result: Any, label_path: str) -> float: ...
        # true metric vs offline label; NotImplementedError allowed with
        # a message routing to experiments/offline_eval.py (R5-a)
```

Registry `get_task_adapter(name, config)` in the same file, mirroring the
planner registry pattern.

### 4.3 Concrete adapters — implementation notes (all R5 corrections apply)

- **detection.py:** port cv_guide §5.5 with: forward hook registered once per
  loaded model on the backbone output layer (verify the layer index against
  the installed ultralytics version at runtime — resolve by module class
  name, not a hardcoded `model.model.model[9]`, and raise with the discovered
  architecture printed if resolution fails); embedding = spatial-mean-pooled
  feature map, float32; proxy = mean box confidence, 0.0 on zero boxes.
- **classification.py:** torchvision with `weights=None`, head resized to
  `config["num_classes"]` before `load_state_dict`; transform built once;
  proxy = max softmax; embedding hook on the pooled penultimate layer.
- **segmentation.py:** processor constructed in `load_model`; proxy = mean
  per-pixel max softmax; embedding = channel vector from spatially-pooled
  encoder final hidden state; `offline_accuracy` = mIoU with logits
  bilinearly upsampled to mask resolution before argmax, ignore-index
  respected (config key `ignore_index`, default 255).
- **Weights availability:** only YOLO weights exist locally
  (`get_models.py`). classification/segmentation adapters must import their
  heavy deps lazily inside methods and be fully unit-testable with mocks;
  their real-model paths are exercised later on Arch. Toy-CV conformance
  runs use the detection adapter.

### 4.4 Config schema merge

Extend `configs/datasets/_template.json` (single source of truth) with the CV
task keys, documented inline: `task`, `task_adapter`, `model_spectrum`,
`model_paths`, `num_classes` (classification), `ignore_index` (segmentation),
`proxy` (existing key — registry name), `drift_detector` (existing),
`embedding_model`, `embedding_dim`, `embedding_sample_every`,
`drift_window_size`, `retrain_tactic`, `pseudo_label_threshold`,
`rollback_if_worse`, plus the standard `thresholds` block including `E_ref`,
`delta`, `tau_drift`, `switch_cooldown_s`. Update `bdd100k.json` and
`toy_cv.json` to the merged schema. `core/dataset_validator.py` gains the new
required-key checks for `domain=="cv"` (task in the allowed set, embedding
keys present and consistent, model paths exist for the toy/bdd configs).
Do NOT create `managed_system_cv/configs/` (R5-g).

### 4.5 Rewire `managed_system_cv`

- `inference.py`: replace direct `YOLO(...)` calls with
  `task_adapter.load_model/infer/extract_proxy`; keep the B4 model cache,
  the L3 GPU arch guard (now conditional: only when the active task adapter
  reports it needs CUDA and CUDA is present), `EnergyMeter` wrapping, and the
  predictions.csv schema (`confidence` column now holds the task proxy).
  Add embedding extraction per R4 (designated model, sample_every) writing to
  the embedding store (R5-e). Luminance histogram column: keep writing it
  (cheap, needed for luminance_kl comparisons) — the paper comparison
  requires both signals.
- `mape_logic/monitor.py`: drift section calls the configured detector via
  the registry; for embedding detectors it reads windows from the embedding
  store and scores against the fixed reference (R3); emits
  `{"kl_div": <luminance>, "mmd": <embedding>, "drift_detector": <primary>}`
  with the primary signal also mapped onto the key the policy watches.
  `fresh:false` contract untouched.
- `analyse.py`/`plan.py`/`execute.py`: should need only minimal changes —
  drift analysis passes the primary signal + tau_drift from config; VMR
  matching uses embedding signatures when the primary detector is
  embedding-based, histogram signatures when luminance (I6: like-typed only).
- `retrain_tactics/pseudo_label.py`: route model calls through the task
  adapter so the tactic is task-generic in interface even though only the
  detection path is exercised now.

### 4.6 Tests for this phase (write alongside, not after)

`tests/test_task_adapters.py`: registry resolution; detection adapter with a
mocked YOLO result object (proxy math, zero-box case, embedding dim
assertion); classification proxy from a mocked logits tensor; segmentation
mIoU on a tiny synthetic (4×4) pred/gt pair including upsample path and
ignore_index; TaskAdapter never touches knowledge files (assert no such
opens via monkeypatched open). Extend `test_plug_and_play.py`: toy_cv config
validates under the merged schema and a 30-frame headless session completes
through the task-adapter path.

---

## 5. Build Phase C — Embedding Drift, Store, Reference, VMR (P0)

### 5.1 Embedding store (`core/drift/embedding_store.py`) — per R5-e
Memmapped float16 ring buffer + index csv + cursor in `mape_info.json`.
Capacity `2 × drift_window_size`. API: `append`, `last_window(n) -> float32
array | None if fewer than n`, `reset()`. `experiments/run_reset.py` calls
`reset()`. Unit tests: wraparound correctness, dtype round-trip, None on
underfill.

### 5.2 Fixed reference (`scripts/init_cv.py` extension)
For the config's `embedding_model`: run it over the reference image set
(config `reference_images` glob or first-N of the manifest train split),
collect embeddings, persist `knowledge/reference_embeddings.npz`
(`embeddings` float32 array, `model`, `dim`, `n`, `created_at`). Existing
luminance-histogram outputs stay. Idempotent, `--force` respected. On WSL/no
weights this step may be skipped for toy runs via a validator-visible
`--skip-embeddings` flag that stamps the omission into the npz-absent state;
the CV startup health check then requires reference_embeddings.npz ONLY when
the config's `drift_detector` is embedding-based (keep luminance_kl the
default for toy_cv so WSL smoke runs need no weights beyond YOLO-n).

### 5.3 Detector wiring
`mmd_embedding` / `frechet_embedding` `fit_reference()` load the npz;
`score(window)` takes the store's `last_window(drift_window_size)`; return
`None` on underfill (I5). Secondary `mmd_local` (rolling) optional behind
config `"emit_local_drift": true` — NOT the trigger signal (R3).

### 5.4 VMR embedding signatures
When storing a CV version under an embedding-primary config, signature =
`{"type": "embedding", "mean": [...], "cov_diag_or_full": ..., "model":
embedding_model, "dim": d}` computed over the retrain window's stored
embeddings; matching via the existing `closest_distribution` strategy using
`core/drift/frechet_embedding.py` distance (correct math — R5-d).
Histogram-signature versions remain matchable under luminance configs;
cross-type match attempts must raise (I6). Tests: store/match round-trip for
both types; cross-type raises.

---

## 6. Build Phase D — WSL Launcher and Smoke (P0)

The current `harmone_start.sh` spawns GUI terminal emulators — unavailable in
WSL. Deliver a launch path that works in a single WSL shell.

### 6.1 `tool/harmone_start_wsl.sh` (new; do not break the Arch script)

```
#!/usr/bin/env bash
set -euo pipefail
# 1. venv check/create + pip install -r requirements.txt (CPU torch index)
# 2. Preflight: python scripts/validate_dataset.py --config <active>;
#    python -c "from core.energy import get_backend_status; print(...)"
#    (expected in WSL: all backends unavailable -> null meters, valid:false;
#     script prints a clear yellow notice, does NOT abort)
# 3. Launch in background with logs:
#      python app.py            > logs/acp.log 2>&1 &
#      (cd frontend && python -m http.server 8000) > logs/dashboard.log 2>&1 &
#      python run_managed_system.py > logs/wrapper.log 2>&1 &
#    PIDs written to logs/harmone.pids
# 4. Health wait: poll http://localhost:5000/ and :8080/adaptor/health up to
#    30 s; on failure, dump the last 30 lines of each log and exit 1.
# 5. Print: dashboard URL, log paths, and "./harmone_stop.sh to stop".
```

Companion `tool/harmone_stop.sh`: kill PIDs from the pidfile, then the psutil
sweep via a small python -c fallback. Both scripts `chmod +x`, LF line
endings enforced (`.gitattributes` entry `*.sh text eol=lf` — WSL chokes on
CRLF shebangs, and this repo has lived on Windows: check and fix line endings
on ALL .sh files).

Refactor `harmone_start.sh` (Arch) to source shared setup steps from a common
`scripts/_launch_common.sh` rather than duplicating; terminal-emulator UX
unchanged.

### 6.2 Energy in WSL — expected behavior, must be graceful
No RAPL in WSL; NVML sometimes partially works under WSL2. `EnergyMeter`
must: probe once, cache, log `cpu_backend: none` etc., produce
`valid: false` rows, and the pipeline runs to completion regardless. If any
component crashes on absent backends, that is a bug — fix in `core/energy.py`,
add a test with all probes mocked False.

### 6.3 WSL smoke procedure (this is Checkpoint CP4's substance)
1. `./harmone_start_wsl.sh` with `approach.conf` → regression / harmone /
   `pems_node1` (after `init_regression.py`). Let it run ≥ 3 telemetry
   cycles; verify predictions.csv grows, dashboard serves, telemetry visible
   at `/api/knowledge/...`, no dead subprocess, energy columns present with
   valid:false, `./harmone_stop.sh` leaves no orphan python processes
   (`pgrep -f inference.py` empty).
2. Same for CV with `toy_cv` config, `luminance_kl` detector, YOLO-n only if
   weights present — if YOLO weights are absent and >200 MB to fetch is
   needed, the n-model (~6 MB) is pre-approved; fetch just yolo_n via
   get_models.py. CPU inference is fine (no CUDA in this WSL check); the L3
   guard must not fire on a CPU-only environment.
3. Headless: `python experiments/run_experiment.py` toy_regression ×
   harmone_original × seed 1 completes and `metrics.py` summarises it.

---

## 7. Docs Phase E — Dataset Expectation Specs (P0, no downloads)

Create `docs/datasets/` with one file per dataset plus an index. Each file
follows the same template: **Purpose in the study · Task & models · Expected
RAW form (what the download gives) · Required PREPROCESSED form (exact
schema the validator will accept, per DATA_CONTRACT) · Drift-stream
construction recipe (how to order/split, from which metadata fields) ·
Config skeleton (a ready-to-fill `configs/datasets/<name>.json`) · Init &
calibration steps · Offline-label availability & where they plug in ·
License/registration notes.** Also add each config skeleton as an actual
file in `configs/datasets/` marked `"status": "awaiting_data"` so the
validator can distinguish "not downloaded yet" from "misconfigured" (add
that status handling: validator reports SKIPPED, not FAIL, when status is
awaiting_data and paths are absent).

The six, with the specifics to encode (draw details from plan.md §6 and
cv_guide §4 — restate them concretely, do not just link):
- **R1 `pems_node2.md`** — second PeMS sensor (different traffic character);
  same csv schema as pems_node1 (`flow` column, 5-min intervals,
  chronological); recurring daily/weekly drift; models LSTM/Ridge/SVR.
- **R2 `uci_electricity.md`** — UCI Electricity Load Diagrams 2011–2014;
  preprocess: pick one client column (document which and why), resample to a
  single `value` column csv, 15-min intervals; seasonal recurring drift —
  the VMR-reuse showcase; note unit conversion (kW per 15 min) and the
  portugal DST duplicate-timestamp quirk to handle in preprocessing.
- **R3 `spot_prices.md`** — Nord Pool or ERCOT day-ahead hourly prices over
  a crisis window (structural break, never reverts); expected columns
  timestamp+price → `value`; stress test where HarmonE is EXPECTED to
  partially fail (frame as in plan.md); licensing: Nord Pool restricted →
  ERCOT public fallback documented.
- **C1 `bdd100k.md`** — detection, YOLOv8 n/s/m; raw: images + JSON
  annotation files with `attributes` (weather, timeofday); preprocess:
  manifest.csv ordered clear→overcast→dusk→night→rain built from attributes,
  labels converted via `bdd_to_yolo_labels.py`; offline labels used ONLY by
  offline_eval/proxy validation (I4).
- **C2 `iwildcam.md`** — classification via WILDS package; models
  EfficientNet-B0/ResNet-50/ResNet-101, `num_classes: 182`; preprocess:
  manifest ordered by location clusters (WILDS metadata `location` field),
  per-image label txt with the integer class; geographic drift invisible to
  luminance — the embedding-drift motivation dataset.
- **C3 `acdc.md`** — segmentation; models per cv_guide (SegFormer-B0 at
  minimum; note DeepLab variants as P2); raw: condition-folders
  (fog/night/rain/snow) + Cityscapes-format masks; preprocess: manifest
  ordered clear→fog→rain→night→snow, `label_path` to per-image mask PNG,
  `ignore_index: 255`; correspondence structure enables exact per-condition
  mIoU.

Cross-check every schema statement against `core/dataset_validator.py` — the
doc must promise exactly what the validator enforces, nothing else. Extend
DATA_CONTRACT.md with a short "per-dataset specs live in docs/datasets/"
pointer and the new CV task keys.

---

## 8. Docs Phase F — README and Live Documents (P0)

### 8.1 README.md (rewrite, top-level of repo)
Sections, in order: project summary (2 paragraphs: HarmonE/Harmonica in one,
what this extension adds in one) · architecture overview (ASCII diagram of
ACP ↔ wrapper ↔ managed systems ↔ core registries; one paragraph per layer)
· repository map (short — link files.md for the full inventory) · quickstart
WSL (exact commands: clone, venv, pip, make_toy_datasets, init_regression,
harmone_start_wsl.sh, dashboard URL, stop) · quickstart Arch/energy
(setup_energy_permissions.sh, probe_energy.py, expected probe outputs, full
launcher) · running headless experiments (run_experiment/run_grid/metrics
with one worked example) · adding a dataset (the config-only recipe;
link docs/datasets/) · configuration reference (thresholds keys table with
one-line meanings and paper symbols β γ α S_min τ_E τ_drift E_ref δ) ·
design decisions (bullet list, each one sentence + link: pyJoules migration,
fixed-vs-rolling drift, task-adapter unification, pinned embedding model
(R4), fail-loud philosophy, per-run reset, plug-and-play contract) · testing
(pytest command, marker for slow smoke) · known limitations (energy absent in
WSL, S7/RT2 deferred, datasets awaiting download, thresholds awaiting
per-dataset calibration) · papers & citation stubs · license.
Every command in the README must be one you actually executed successfully
during this sprint — no aspirational commands.

### 8.2 Live documents
- `files.md`: full refresh after all structural changes (new tasks/ package,
  embedding_store, docs/datasets/, launchers, deletions/renames from R6).
- `CHANGES_FROM_PAPER.md`: R2 rewrite; new entries — task-adapter
  unification, pinned embedding model (R4, with the honest overhead note),
  fixed-reference embedding drift (R3), embedding sidecar store; A3 filled.
- `documentation.md`: append a "v2 architecture" section covering what
  changed since the Harmonica description (do not rewrite the historical
  Harmonica content — mark it as describing the original tool, add the
  extension delta after it).
- `DECISIONS_PENDING.md`: refresh — closed items marked with what was
  decided; new deferrals (P2 list) added with options.

---

## 9. Phase G — Repo Hygiene and GitHub Push (P0)

1. `.gitignore` must cover: `harmone_env/`, any venv, `__pycache__/`,
   `*.pyc`, `data/` (except toy generators' OUTPUT — toys are generated, so
   ignore `data/` wholesale and regenerate via script; verify tests generate
   toys themselves), `models/`, `base_models/`, `versionedMR/`, `runs/`,
   `runs_artifact/`, `logs/`, `knowledge/predictions.csv`,
   `knowledge/embeddings.f16.npy`, `knowledge/embeddings_index.csv`,
   `knowledge/.energy_backends.json`, `knowledge/command.txt`,
   `knowledge/drift*.{csv,json}`, `*.pids`, `.pytest_cache/`. Keep tracked:
   `thresholds.json`, config templates, policies. Check `git status` for
   large or generated files already tracked and `git rm --cached` them.
2. `.gitattributes`: `*.sh text eol=lf`, `*.py text eol=lf` (the repo has
   CRLF contamination — cv_guide.md itself is CRLF; normalise code files,
   leave docs as-is to keep the diff reviewable).
3. Sanity greps before push: no absolute `D:/` or `/home/<user>` paths in
   code; no `import pyRAPL`; no TODO in P0 paths without a DECISIONS_PENDING
   reference; `pip freeze`-verified `requirements.txt` +
   `requirements_cv.txt` (pinned; cu-index documented in the header comment).
4. LICENSE present (MIT per repo) — verify authors line.
5. Final: full `pytest -q` green → commit → merge `feat/final-prototype`
   into the main working branch → tag `v2.0-prototype` → push branch + tag
   to origin. If the remote is not configured or push is rejected:
   stop-and-ask (rule 0.2.6) — never force-push.

---

## 10. Checkpoints — run these verbatim, report each before proceeding

Each checkpoint produces a short report in `docs/checkpoints/CP<N>.md`:
commands run, outputs (trimmed), invariant checklist I1–I8 PASS/FAIL, and
"deviations from guide" (empty or explained).

- **CP0 — Reality check (before any code).** `git status`/`git log -5`;
  `pytest tool/tests/ -q` full run recorded; tree diff against `files.md`
  (list every discrepancy); confirm §1's DONE/NOT-done synthesis or correct
  it. If the synthesis is materially wrong → stop-and-ask.
- **CP1 — after §2 corrections (R1–R6).** Tests green; CHANGES internally
  consistent (read Phase 4 and Phase 6 back to back); grep proofs for R6
  deletions.
- **CP2 — after §4 (task adapters).** `test_task_adapters.py` +
  `test_plug_and_play.py` green; toy_cv headless 30-frame run through the
  adapter path; I1/I4/I7 explicitly re-verified.
- **CP3 — after §5 (embedding drift).** Store tests green; a scripted demo:
  feed the store synthetic embeddings drawn from N(0,I) as reference then
  N(0.5,I) as current → MMD score rises above a calibrated-on-null
  threshold; fixed-ref vs rolling divergence demonstrated (R3 proof); I5/I6
  re-verified.
- **CP4 — WSL smoke (§6.3 all three steps).** Attach log excerpts; zero
  orphan processes after stop; I8 verified (paper-config toy runs).
- **CP5 — docs complete.** README commands re-executed from a clean shell;
  every docs/datasets file cross-checked against the validator; files.md
  regenerated and spot-checked against `find`.
- **CP6 — pre-push.** Hygiene greps clean; `git status` clean; test suite
  green; tag created; push done (or stop-and-ask outcome recorded).

---

## 11. Definition of Done

1. CP0–CP6 reports exist, all invariants PASS at CP6.
2. Full pytest suite green including new task-adapter, embedding-store, and
   extended conformance tests.
3. `harmone_start_wsl.sh` boots ACP + dashboard + managed system in WSL for
   both domains (toy/bundled data), telemetry flows, stop script is clean.
4. CV pipeline is task-agnostic behind TaskAdapter; embedding drift runs
   against a fixed init-time reference with a pinned embedding model;
   luminance path preserved as the paper-default comparison signal.
5. All R1–R6 corrections applied; live documents mutually consistent and
   current; A3 filled with real git evidence.
6. Six dataset expectation docs + `awaiting_data` config skeletons in place;
   validator distinguishes SKIPPED from FAIL.
7. README enables a stranger to go from clone to a running WSL prototype and
   to understand the architecture and every major design decision.
8. Repo pushed with tag `v2.0-prototype`; no generated/large files tracked;
   no pyRAPL, no absolute paths, requirements pinned.
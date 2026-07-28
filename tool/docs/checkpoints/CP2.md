# CP2 — After §4 Unified CV Task-Adapter Layer

**Date:** 2026-07-29  
**Branch:** feat/final-prototype  
**Guide:** execution.md §10 CP2

---

## §4 Work Completed

### §4.1 — DatasetAdapter: `adapters/cv_imagedir.py`
`CVImageDirAdapter` — image-directory / manifest-CSV dataset adapter.
`from_config()` classmethod; `train_split()`, `val_split()`, `stream()` yield
`Sample(index, inputs=path, ground_truth=None)`; `offline_labels()` returns
label paths from manifest or `None`; `models()` returns `ModelSpec` dict.

### §4.2 — TaskAdapter ABC: `adapters/tasks/base.py`
`TaskAdapter` abstract base: `task`, `load_model()`, `infer()`, `extract_proxy()`,
`extract_embedding()`, `offline_accuracy()`. `_TASK_REGISTRY` dict + `@register_task`
decorator + `get_task_adapter(task, config)` factory.

### §4.3 — Concrete adapters

**Detection (`adapters/tasks/detection.py`):**
- SPPF hook resolved by class name (not index); falls back to C2f/C3/Conv.
- `extract_proxy()`: `boxes.conf.mean()`, 0.0 on empty, raises on `None`.
- `extract_embedding()`: spatial-mean-pool of hook feature map; validates against
  `config["embedding_dim"]`.
- `offline_accuracy()`: raises `NotImplementedError` pointing to `experiments/offline_eval.py`.

**Classification (`adapters/tasks/classification.py`):**
- `weights=None` (not deprecated `pretrained=False`).
- Classifier head resized to `num_classes` BEFORE `load_state_dict`.
- Transform built once in `load_model`; `AdaptiveAvgPool2d` hook for embedding.
- `offline_accuracy()`: top-1 from single-integer label file.

**Segmentation (`adapters/tasks/segmentation.py`):**
- `SegformerImageProcessor.from_pretrained()` called once in `load_model`.
- `extract_embedding()`: spatial dims pooled only, keeps channel vector.
- `offline_accuracy()`: bilinear upsample logits to mask H×W BEFORE argmax; mIoU
  with `ignore_index` (R5-b fix applied).

### §4.4 — Config + validator updates
- `configs/datasets/toy_cv.json`, `bdd100k.json`: new CV task keys added
  (`task`, `task_adapter`, `embedding_model`, `embedding_dim`, etc.)
- `configs/datasets/_template.json`: `_comment_cv_task` block documents all new keys.
- `core/dataset_validator.py`: `_VALID_TASKS`, `awaiting_data` early-exit,
  task validation, `embedding_model` catalogue check, `embedding_dim` type check.

### §4.5 — `managed_system_cv/inference.py` rewrite
Full rewrite: `get_task_adapter()` replaces hard-coded YOLO calls; model cache
`_model_cache` keyed by name; embedding model pinned per R4; `EmbeddingStore`
initialized from config; L3 GPU guard preserved; luminance histogram column kept.

### §5.1 — `core/drift/embedding_store.py`
Float16 ring-buffer: capacity = 2 × drift_window; files `embeddings.f16.npy` +
`embeddings_index.csv`; cursor in `mape_info.json["embedding_cursor"]`.
`append()`, `last_window(n)` returns `None` if < n entries, `reset()`.

`experiments/run_reset.py` updated: `_reset_embedding_store()` called from
`reset_run_state()` when domain is CV.

---

## Test Results

**290/290 passed** (27 new tests in `tests/test_task_adapters.py`).

New tests cover:
- Registry: detection / classification / segmentation resolution; unknown task raises.
- `DetectionAdapter`: proxy mean, single-box, zero-box → 0.0, None raises,
  offline_accuracy `NotImplementedError`, embedding dim mismatch raises.
- `ClassificationAdapter`: max-softmax proxy, None raises, offline_accuracy
  correct/wrong class.
- `SegmentationAdapter`: perfect/wrong mIoU, ignore_index exclusion, upsample path.
- `TaskAdapter` never opens knowledge files (verified via monkeypatched open).
- `EmbeddingStore`: append + last_window, underfill → None, wraparound, reset,
  float16 dtype roundtrip.
- Plug-and-play: `toy_cv.json` validates; unknown task caught; `awaiting_data`
  skips path check with correct warning.

---

## Invariant Checklist

| # | Invariant | Status |
|---|---|---|
| I1 | MAPE-K separation (TaskAdapter never writes knowledge/) | PASS |
| I2 | Scoring formula (proxy is float, adapter returns float) | PASS |
| I3 | Dynamic energy threshold | PASS |
| I4 | Label-free runtime CV (no label reads in inference loop) | PASS |
| I5 | No fabricated telemetry | PASS |
| I6 | VMR semantics | PASS |
| I7 | Config over code (task name drives adapter selection) | PASS |
| I8 | Paper behaviour reachable | PENDING (WSL smoke) |

---

## R3/R4/R5 Verification

- **R3** (TaskAdapter isolation): `test_adapter_knowledge_isolation` confirms no
  knowledge/ file opens during `extract_proxy`. ✅
- **R4** (pinned embedding model): `embedding_model` fixed in `inference.py`; separate
  from `chosen_model`; documented in `_template.json` comment. ✅
- **R5-a** (detection `NotImplementedError`): raises with message pointing to
  `offline_eval.py`. ✅
- **R5-b** (segmentation upsample before argmax): bilinear upsample at `orig_size`
  before argmax; test_upsample_path_different_logit_resolution confirms. ✅
- **R5-c** (classification `weights=None`, head resize before `load_state_dict`): ✅

---

## Next

Proceed to §5.2–§5.4: extend `scripts/init_cv.py` for reference embeddings,
wire embedding drift detector in CV monitor.py, VMR embedding signatures (CP3).

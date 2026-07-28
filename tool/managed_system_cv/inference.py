"""managed_system_cv/inference.py — Task-agnostic CV inference loop.

§4.5 rewrite: replaces hard-coded YOLO calls with TaskAdapter.load_model/infer/
extract_proxy.  The active inference model is still chosen by the MAPE planner
via knowledge/model.csv.  Embedding extraction always uses the config's fixed
embedding_model (R4) regardless of which model is currently serving inference.

Startup checks (unchanged from live-run fixes L1-L3):
  - L3: GPU arch guard (fast-fail, task-adapter-aware)
  - L1-b: requires knowledge/model.csv to exist

Embedding sidecar (§5): every embedding_sample_every frames, extract embedding
from the embedding_model and append to EmbeddingStore.

Luminance histogram column: kept (cheap, needed for luminance_kl comparisons).
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Make tool/ importable when this script runs from managed_system_cv/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.energy import EnergyMeter

# ── Config loading ─────────────────────────────────────────────────────────────

KNOWLEDGE_DIR = Path("knowledge")
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("versionedMR", exist_ok=True)
os.makedirs(KNOWLEDGE_DIR / "inferences", exist_ok=True)

def _load_thresholds() -> dict:
    try:
        with open(KNOWLEDGE_DIR / "thresholds.json") as f:
            return json.load(f)
    except Exception:
        return {}

def _load_dataset_config() -> dict:
    """Load the active dataset config via approach.conf or fallback to bdd100k."""
    try:
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read("approach.conf")
        dataset_name = cfg.get("dataset", "name", fallback="bdd100k")
    except Exception:
        dataset_name = "bdd100k"
    config_path = Path("../configs/datasets") / f"{dataset_name}.json"
    if not config_path.exists():
        config_path = Path("../configs/datasets/bdd100k.json")
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        return {}

_thresholds = _load_thresholds()
_dataset_cfg = _load_dataset_config()
_energy_backend = _thresholds.get("energy_meter", "auto")
_task_name = _dataset_cfg.get("task", "detection")
_embedding_model_name = _dataset_cfg.get("embedding_model", "yolo_n")
_embedding_sample_every = int(_dataset_cfg.get("embedding_sample_every", 1))
_embedding_dim = _dataset_cfg.get("embedding_dim")
_drift_window = int(_dataset_cfg.get("drift_window_size", 500))

# ── L3: GPU arch guard (task-adapter-aware) ───────────────────────────────────

try:
    import torch
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        arch_list = torch.cuda.get_arch_list()
        target_sm = f"sm_{cc[0]}{cc[1]}"
        supported = any(
            a == target_sm or (a.startswith("sm_") and int(a[3:]) >= cc[0] * 10 + cc[1])
            for a in arch_list
        )
        if not supported:
            raise RuntimeError(
                f"Installed torch has no kernels for this GPU "
                f"(compute capability {cc[0]}.{cc[1]} / {target_sm}). "
                f"Supported arches: {arch_list}. "
                f"Reinstall PyTorch with a CUDA build that includes {target_sm}. "
                f"See live-run-report.md §L3 for instructions (cu129 or cu133)."
            )
        print(f"✔ GPU: {torch.cuda.get_device_name(0)} (cc {cc[0]}.{cc[1]}) — "
              f"torch {torch.__version__} supports {target_sm}.")
except ImportError:
    pass  # torch not installed; CPU-only inference will proceed

# ── Task adapter and model cache ───────────────────────────────────────────────

def _resolve_model_paths() -> dict[str, str]:
    """Return {model_name: abs_weights_path} from dataset config or fallback."""
    models_cfg = _dataset_cfg.get("models", {})
    if models_cfg:
        tool_dir = Path(__file__).resolve().parent.parent
        result = {}
        for name, spec in models_cfg.items():
            wp = spec.get("weights_path", "")
            if not os.path.isabs(wp):
                wp = str(tool_dir / wp)
            result[name] = wp
        return result
    # Fallback for direct run without dataset config
    return {
        "yolo_n": "models/yolo_n.pt",
        "yolo_s": "models/yolo_s.pt",
        "yolo_m": "models/yolo_m.pt",
    }

MODEL_PATHS = _resolve_model_paths()

# Load task adapter
try:
    from adapters.tasks.base import get_task_adapter
    _task_adapter = get_task_adapter(_task_name, _dataset_cfg)
except Exception as e:
    print(f"[CV-INFERENCE] Could not load task adapter '{_task_name}': {e}")
    _task_adapter = None

# Model cache: {model_name: handle}
_model_cache: dict[str, object] = {}
_model_csv_cache: str = ""

def _get_model(model_name: str) -> object | None:
    """Load or retrieve the cached model handle via TaskAdapter."""
    global _model_csv_cache
    if _task_adapter is None:
        return None
    if model_name not in _model_cache:
        weights_path = MODEL_PATHS.get(model_name, "")
        if not os.path.exists(weights_path):
            print(f"[CV-INFERENCE] ⚠  Model {model_name} weights not found: {weights_path}")
            return None
        try:
            _model_cache[model_name] = _task_adapter.load_model(model_name, weights_path)
            print(f"[CV-INFERENCE] ✔ Loaded {model_name}")
        except Exception as exc:
            print(f"[CV-INFERENCE] ✗ Failed to load {model_name}: {exc}")
            return None
    return _model_cache.get(model_name)

# ── Embedding store (§5) ───────────────────────────────────────────────────────

_embedding_store = None
if _embedding_dim:
    try:
        from core.drift.embedding_store import EmbeddingStore
        _embedding_store = EmbeddingStore(
            str(KNOWLEDGE_DIR),
            embedding_dim=_embedding_dim,
            drift_window=_drift_window,
        )
    except Exception as exc:
        print(f"[CV-INFERENCE] Embedding store init failed (skipping): {exc}")

# ── VMR seed (unchanged from original) ────────────────────────────────────────

import shutil
from pathlib import Path as _Path
from tqdm import tqdm

def _calculate_and_save_initial_histogram(image_paths: list, output_path: _Path) -> None:
    import numpy as np
    try:
        from utility.drift_utils import luminance_histogram
    except ImportError:
        return
    print(f"Calculating initial histogram for {output_path.name}...")
    total_hist = None
    processed = 0
    for img_path in tqdm(image_paths, desc="  Analyzing", leave=False, ncols=80):
        h = luminance_histogram(img_path)
        if h is not None:
            if total_hist is None:
                total_hist = h * 0.0
            total_hist += h
            processed += 1
    if processed > 0:
        avg = total_hist / processed
        with open(output_path, "w") as f:
            json.dump({"average_histogram": avg.tolist()}, f, indent=4)

print("--- Initializing Model Versions ---")
_ref_image_dir = _Path(_dataset_cfg.get("image_dir", "data/bdd100k/images/test"))
if not _ref_image_dir.is_absolute():
    _ref_image_dir = _Path(__file__).resolve().parent.parent / _ref_image_dir
_ref_images = sorted(_ref_image_dir.glob("*.jpg"))[:1000] if _ref_image_dir.exists() else []

for _model_name, _model_path in MODEL_PATHS.items():
    if not os.path.exists(_model_path):
        continue
    _vmr_model = _Path("versionedMR") / f"{_model_name}_v1.pt"
    _vmr_hist = _Path("versionedMR") / f"{_model_name}_v1_hist.json"
    if not _vmr_model.exists():
        shutil.copy(_model_path, _vmr_model)
        print(f"✔ Seeded VMR v1 for {_model_name}")
    if not _vmr_hist.exists() and _ref_images:
        _calculate_and_save_initial_histogram([str(p) for p in _ref_images], _vmr_hist)

# ── Output CSV ────────────────────────────────────────────────────────────────

results_file = KNOWLEDGE_DIR / "predictions.csv"
_COLUMNS = ["image_name", "confidence", "model_used", "inference_time", "energy_uJ", "histogram"]
if not results_file.exists():
    pd.DataFrame(columns=_COLUMNS).to_csv(str(results_file), index=False)

# ── Main inference loop ───────────────────────────────────────────────────────

print("\n--- Starting Inference ---")
_image_dir = _Path(_dataset_cfg.get("image_dir", "data/bdd100k/images/test"))
if not _image_dir.is_absolute():
    _image_dir = _Path(__file__).resolve().parent.parent / _image_dir
_image_files = sorted(_image_dir.glob("*.jpg")) if _image_dir.exists() else []

_frame_idx = 0

try:
    from utility.drift_utils import luminance_histogram as _lum_hist
    _has_lum = True
except ImportError:
    _has_lum = False

for i, image_path in enumerate(_image_files):
    # Read active model
    try:
        with open(KNOWLEDGE_DIR / "model.csv") as f:
            chosen_model = f.read().strip().lower()
    except FileNotFoundError:
        chosen_model = "yolo_s"
        print("[CV-INFERENCE] knowledge/model.csv not found; defaulting to yolo_s.")

    model = _get_model(chosen_model)
    if model is None:
        print(f"[{i+1}/{len(_image_files)}] Skipping — model {chosen_model} unavailable.")
        continue

    print(f"[{i+1}/{len(_image_files)}] {image_path.name} | model={chosen_model.upper()}")

    with EnergyMeter("inference", backend=_energy_backend) as _em:
        t0 = time.time()
        result = _task_adapter.infer(model, str(image_path))
        inference_time = time.time() - t0
    energy_uJ = _em.total_uJ or 0.0

    proxy = _task_adapter.extract_proxy(result) if _task_adapter else 0.0

    # Luminance histogram (cheap; kept for luminance_kl signal)
    hist_str = ""
    if _has_lum:
        h = _lum_hist(str(image_path))
        if h is not None:
            hist_str = " ".join(f"{x:.8f}" for x in h)

    # Append to predictions.csv
    pd.DataFrame(
        [[image_path.name, proxy, chosen_model, inference_time, energy_uJ, hist_str]],
        columns=_COLUMNS,
    ).to_csv(str(results_file), mode="a", header=False, index=False)

    # YOLO inference .txt output (kept for compatibility with labelling / offline eval)
    infer_txt = KNOWLEDGE_DIR / "inferences" / f"{image_path.stem}.txt"
    _write_yolo_txt(result, infer_txt, image_path)

    # Embedding extraction (R4: fixed embedding_model, every embedding_sample_every)
    if _embedding_store is not None and _task_adapter is not None:
        if _frame_idx % _embedding_sample_every == 0:
            emb_model = _get_model(_embedding_model_name)
            if emb_model is not None:
                try:
                    emb = _task_adapter.extract_embedding(emb_model, str(image_path))
                    _embedding_store.append(emb, label=image_path.name)
                except Exception as exc:
                    print(f"[CV-INFERENCE] Embedding extraction failed: {exc}")

    _frame_idx += 1


def _write_yolo_txt(result, out_path: _Path, image_path: _Path) -> None:
    """Write per-image YOLO detection .txt from inference result."""
    from PIL import Image
    try:
        if result is None:
            return
        # Detection result (ultralytics)
        boxes = getattr(result[0], "boxes", None) if hasattr(result, "__getitem__") else None
        if boxes is None or len(boxes) == 0:
            out_path.write_text("")
            return
        img = Image.open(image_path)
        w, h = img.size
        lines = []
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.6f}")
        out_path.write_text("\n".join(lines))
    except Exception:
        pass  # Non-fatal: inference txt is for offline eval only


print("\nCV Inference completed. Results saved in knowledge/predictions.csv")

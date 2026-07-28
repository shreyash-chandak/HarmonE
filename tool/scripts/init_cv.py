"""scripts/init_cv.py — Initialise CV managed-system training-time artifacts.

Creates (idempotent):
  - managed_system_cv/versionedMR/{model}_v1.pt       (copy of base weights)
  - managed_system_cv/versionedMR/{model}_v1_hist.json (avg luminance hist)
  - managed_system_cv/knowledge/model.csv              (default active model)
  - managed_system_cv/knowledge/mape_info.json         (default MAPE state)
  - managed_system_cv/knowledge/reference_embeddings.npz  (§5.2 fixed ref)

reference_embeddings.npz keys:
  embeddings  float32 (N, D)  — raw embedding vectors from reference images
  model       str              — embedding_model name
  dim         int              — embedding dimension D
  n           int              — number of reference embeddings
  created_at  str              — ISO timestamp

The npz is required at runtime only when the config's drift_detector is
embedding-based (mmd_embedding / frechet_embedding). luminance_kl configs
(including the default toy_cv) do not need it; the CV health check skips the
npz requirement for those configs, so WSL smoke runs without model weights
work fine via --skip-embeddings.

Usage:
    cd tool/
    python scripts/init_cv.py [--config bdd100k] [--force] [--skip-embeddings]
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

_TOOL_DIR = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _TOOL_DIR / "scripts"
_CV_DIR = _TOOL_DIR / "managed_system_cv"

_DEFAULT_MODELS = ["yolo_n", "yolo_s", "yolo_m"]
_N_REF_IMAGES = 1000

_EMBEDDING_DETECTORS = {"mmd_embedding", "frechet_embedding"}


def _load_config(config_name: str) -> dict:
    cfg_path = _TOOL_DIR / "configs" / "datasets" / f"{config_name}.json"
    if not cfg_path.exists():
        print(f"❌ Config not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path) as f:
        return json.load(f)


def _resolve_path(raw: str) -> Path:
    """Resolve a path that may be relative to _TOOL_DIR."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return (_TOOL_DIR / p).resolve()


def _compute_avg_histogram(image_paths: list[Path]) -> list[float] | None:
    """Return average luminance histogram over image_paths (up to _N_REF_IMAGES)."""
    sys.path.insert(0, str(_CV_DIR))
    try:
        from utility.drift_utils import luminance_histogram
    except ImportError as exc:
        print(f"⚠  Cannot import drift_utils: {exc}. Skipping histogram init.")
        return None

    import numpy as np
    from tqdm import tqdm

    total: "np.ndarray | None" = None
    n = 0
    for img_path in tqdm(image_paths[:_N_REF_IMAGES], desc="  Building ref histogram", ncols=80):
        h = luminance_histogram(img_path)
        if h is not None:
            if total is None:
                total = np.zeros_like(h, dtype=np.float64)
            total = total + h
            n += 1

    if total is None or n == 0:
        return None
    avg = (total / n).tolist()
    return avg


def _seed_reference_embeddings(
    cfg: dict,
    ref_image_paths: list[Path],
    knowledge_dir: Path,
    force: bool,
) -> None:
    """Extract embeddings from reference images and persist knowledge/reference_embeddings.npz.

    Skipped when:
      - --skip-embeddings flag is passed (caller sets ref_image_paths=[])
      - config drift_detector is not embedding-based
      - embedding_model or embedding_dim not in config
      - embedding_model weights not available
    """
    drift_detector = cfg.get("drift_detector", "luminance_kl")
    if drift_detector not in _EMBEDDING_DETECTORS:
        print(f"  ⓘ drift_detector='{drift_detector}' is not embedding-based; skipping reference_embeddings.npz.")
        return

    if not ref_image_paths:
        print("  ⚠  No reference images available; skipping reference_embeddings.npz.")
        return

    emb_model_name = cfg.get("embedding_model", "")
    emb_dim = cfg.get("embedding_dim")
    if not emb_model_name or not emb_dim:
        print("  ⚠  embedding_model / embedding_dim not set in config; skipping reference_embeddings.npz.")
        return

    npz_path = knowledge_dir / "reference_embeddings.npz"
    if npz_path.exists() and not force:
        import numpy as np
        data = np.load(str(npz_path))
        n_stored = int(data["n"])
        print(f"  ✔ reference_embeddings.npz already exists ({n_stored} vectors, skip; use --force to regenerate)")
        return

    # Resolve weights for embedding_model
    models_cfg: dict = cfg.get("models", {})
    emb_weights_raw = (models_cfg.get(emb_model_name) or {}).get("weights_path", "")
    emb_weights_path = (_resolve_path(emb_weights_raw) if emb_weights_raw else None)
    if not emb_weights_path or not emb_weights_path.exists():
        print(
            f"  ⚠  Embedding model '{emb_model_name}' weights not found "
            f"at {emb_weights_path}; skipping reference_embeddings.npz.\n"
            "     Re-run with --force after placing weights to regenerate."
        )
        return

    # Load task adapter and embedding model
    task = cfg.get("task", "detection")
    try:
        sys.path.insert(0, str(_TOOL_DIR))
        from adapters.tasks.base import get_task_adapter
        adapter = get_task_adapter(task, cfg)
        model_handle = adapter.load_model(emb_model_name, str(emb_weights_path))
    except Exception as exc:
        print(f"  ⚠  Could not load embedding model '{emb_model_name}': {exc}; skipping npz.")
        return

    import numpy as np
    from tqdm import tqdm

    embeddings: list[np.ndarray] = []
    print(f"  Extracting reference embeddings via '{emb_model_name}' ({len(ref_image_paths)} images) …")
    failed = 0
    for img_path in tqdm(ref_image_paths, desc="  Embedding", ncols=80, leave=False):
        try:
            emb = adapter.extract_embedding(model_handle, str(img_path))
            embeddings.append(emb.astype(np.float32))
        except Exception:
            failed += 1

    if not embeddings:
        print(f"  ⚠  No embeddings extracted (all {failed} attempts failed); skipping npz.")
        return

    emb_array = np.stack(embeddings, axis=0)  # (N, D) float32
    np.savez_compressed(
        str(npz_path),
        embeddings=emb_array,
        model=np.bytes_(emb_model_name),
        dim=np.array([emb_array.shape[1]], dtype=np.int32),
        n=np.array([len(emb_array)], dtype=np.int32),
        created_at=np.bytes_(datetime.datetime.utcnow().isoformat()),
    )
    print(
        f"  ✔ Wrote reference_embeddings.npz: {len(emb_array)} vectors, "
        f"dim={emb_array.shape[1]}, model={emb_model_name}"
        + (f" ({failed} skipped)" if failed else "")
    )


def init_cv(config_name: str = "bdd100k", force: bool = False, skip_embeddings: bool = False) -> None:
    cfg = _load_config(config_name)
    if cfg.get("domain", "cv") != "cv":
        print(f"❌ Config '{config_name}' is domain={cfg.get('domain')}, not cv.")
        sys.exit(1)

    versioned_dir = _CV_DIR / "versionedMR"
    knowledge_dir = _CV_DIR / "knowledge"
    versioned_dir.mkdir(exist_ok=True)
    knowledge_dir.mkdir(exist_ok=True)

    models_cfg: dict = cfg.get("models", {})
    if not models_cfg:
        # Fall back to default model layout
        models_cfg = {
            m: {"weights_path": str(_CV_DIR / "models" / f"{m}.pt")}
            for m in _DEFAULT_MODELS
        }

    # Collect reference images
    image_dir_raw = cfg.get("image_dir", "")
    image_dir: Path | None = None
    ref_image_paths: list[Path] = []
    if image_dir_raw:
        image_dir = _resolve_path(image_dir_raw)
        if image_dir.is_dir():
            ref_image_paths = sorted(image_dir.glob("*.jpg"))[:_N_REF_IMAGES]
            if not ref_image_paths:
                ref_image_paths = sorted(image_dir.glob("*.png"))[:_N_REF_IMAGES]
        else:
            print(f"⚠  Reference image dir not found: {image_dir}. Histograms will be skipped.")

    any_weights_missing = False
    for model_name, model_cfg in models_cfg.items():
        weights_raw = model_cfg.get("weights_path", "")
        weights_path = _resolve_path(weights_raw) if weights_raw else None

        # ── versionedMR/{model}_v1.pt ──────────────────────────────────────────
        v1_pt = versioned_dir / f"{model_name}_v1.pt"
        if v1_pt.exists() and not force:
            print(f"  ✔ {v1_pt.name} already exists (skip, use --force to overwrite)")
        else:
            if weights_path and weights_path.exists():
                shutil.copy(weights_path, v1_pt)
                print(f"  ✔ Seeded {v1_pt.name} from {weights_path}")
            else:
                print(f"  ⚠  Weights not found for {model_name} at {weights_path}; skipping v1 seed.")
                any_weights_missing = True

        # ── versionedMR/{model}_v1_hist.json ──────────────────────────────────
        v1_hist = versioned_dir / f"{model_name}_v1_hist.json"
        if v1_hist.exists() and not force:
            print(f"  ✔ {v1_hist.name} already exists (skip)")
        else:
            if not ref_image_paths:
                print(f"  ⚠  No reference images available — skipping {v1_hist.name}.")
            else:
                print(f"  Computing histogram for {model_name}_v1 …")
                avg_hist = _compute_avg_histogram(ref_image_paths)
                if avg_hist is not None:
                    with open(v1_hist, "w") as fh:
                        json.dump({"average_histogram": avg_hist}, fh)
                    print(f"  ✔ Wrote {v1_hist.name} ({len(avg_hist)} bins, {len(ref_image_paths)} images)")
                else:
                    print(f"  ⚠  Histogram computation failed for {model_name}_v1.")

    # ── knowledge/model.csv ───────────────────────────────────────────────────
    model_csv = knowledge_dir / "model.csv"
    if model_csv.exists() and not force:
        print(f"  ✔ model.csv already exists: {model_csv.read_text().strip()}")
    else:
        default_model = cfg.get("default_model", "yolo_s")
        model_csv.write_text(default_model)
        print(f"  ✔ Wrote model.csv → {default_model}")

    # ── knowledge/mape_info.json ───────────────────────────────────────────────
    mape_info_path = knowledge_dir / "mape_info.json"
    if mape_info_path.exists() and not force:
        print(f"  ✔ mape_info.json already exists (skip)")
    else:
        model_names = list(models_cfg.keys()) or _DEFAULT_MODELS
        mape_info: dict = {
            "last_line": 0,
            "last_switch_ts": 0.0,
            "recovery_cycles": 0,
            "current_energy_threshold": cfg.get("max_energy", 0.43),
            "ema_scores": {m: 0.5 for m in model_names},
            "ema_accuracy": {m: 0.0 for m in model_names},
            "ema_energy": {m: 0.0 for m in model_names},
            "event_counters": {
                "model_switches": 0,
                "retrains": 0,
                "vmr_events": 0,
                "noops": 0,
                "mape_k_energy_uJ": 0.0,
            },
            "simple_switch_counters": {"simple_switches": 0},
        }
        with open(mape_info_path, "w") as f:
            json.dump(mape_info, f, indent=4)
        print(f"  ✔ Wrote mape_info.json with fresh state")

    # ── knowledge/reference_embeddings.npz (§5.2) ────────────────────────────
    if not skip_embeddings:
        _seed_reference_embeddings(cfg, ref_image_paths, knowledge_dir, force)
    else:
        print("  ⓘ --skip-embeddings: reference_embeddings.npz generation skipped.")

    print()
    if any_weights_missing:
        print(
            "⚠  Some model weights were missing — versionedMR seeding incomplete.\n"
            "   Download or train weights, then re-run with --force."
        )
    else:
        print("✅ CV init complete. System is ready to run.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise CV managed-system training-time artifacts.")
    parser.add_argument("--config", default="bdd100k",
                        help="Dataset config name (default: bdd100k)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing artifacts")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip reference_embeddings.npz generation (for WSL/no-weights runs)")
    args = parser.parse_args()

    print(f"[init_cv] Initialising CV artifacts for config '{args.config}' …")
    init_cv(config_name=args.config, force=args.force, skip_embeddings=args.skip_embeddings)


if __name__ == "__main__":
    main()

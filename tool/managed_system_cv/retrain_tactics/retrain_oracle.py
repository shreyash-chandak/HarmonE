"""
managed_system_cv/retrain_tactics/retrain_oracle.py — Supervised oracle retrain.

Refactored from the original retrain.py. Only callable when the dataset adapter
has offline_labels() available (self_labeling=True or explicit label directory).

Called by monitor/execute when drift action == "retrain" AND labels are available.
If labels are unavailable, returns {"status": "skipped", "reason": "no_labels"}.

The oracle tactic:
1. Deduces drift type from luminance statistics (dark / fog / clear).
2. Builds an augmented training set matching the detected drift.
3. Fine-tunes the last-N layers of the current model for 5 epochs.
4. Saves a new version to knowledge/vmr/ via VMR.store().
5. Copies new weights to models/<model>.pt.
6. Writes knowledge/model_reload.flag.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# Ensure core/ is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.vmr import VMR
from core.energy import EnergyMeter


def run(
    model_name: str,
    knowledge_dir: str = "knowledge",
    data_dir: str = "data/bdd100k",
    base_models_dir: str = "base_models",
    active_models_dir: str = "models",
    vmr_base_dir: str = "knowledge/vmr",
    offline_labels_available: bool = False,
    n_ref_images: int = 1000,
    epochs: int = 5,
) -> dict:
    """Run oracle-supervised retrain.

    Returns:
        dict with keys: status ("ok"|"skipped"|"error"), version_path (str|None),
        new_version_name (str|None), energy_uJ (float).
    """
    if not offline_labels_available:
        print("[ORACLE] No offline labels available — skipping oracle retrain.")
        return {"status": "skipped", "reason": "no_labels", "version_path": None}

    try:
        return _do_retrain(
            model_name=model_name,
            knowledge_dir=knowledge_dir,
            data_dir=data_dir,
            base_models_dir=base_models_dir,
            active_models_dir=active_models_dir,
            vmr_base_dir=vmr_base_dir,
            n_ref_images=n_ref_images,
            epochs=epochs,
        )
    except Exception as exc:
        print(f"[ORACLE] Retrain failed: {exc}")
        return {"status": "error", "reason": str(exc), "version_path": None}


def _do_retrain(
    model_name, knowledge_dir, data_dir, base_models_dir, active_models_dir,
    vmr_base_dir, n_ref_images, epochs
) -> dict:
    import pandas as pd
    from PIL import Image
    from ultralytics import YOLO

    knowledge_path = Path(knowledge_dir)
    data_path = Path(data_dir)
    base_models_path = Path(base_models_dir)
    active_models_path = Path(active_models_dir)

    ref_image_dir = data_path / "images" / "test"
    ref_label_dir = data_path / "labels" / "test"
    predictions_file = knowledge_path / "predictions.csv"

    model_path = base_models_path / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Base model not found: {model_path}")

    # Deduce drift type from luminance histograms in predictions.csv
    drift_type = "clear"
    try:
        df = pd.read_csv(predictions_file)
        if len(df) >= n_ref_images and "histogram" in df.columns:
            cur_hists_str = df["histogram"].iloc[-n_ref_images:]
            cur_hists = np.array([np.fromstring(h, sep=" ") for h in cur_hists_str if h])
            ref_image_paths = sorted(list(ref_image_dir.glob("*.jpg")))[:n_ref_images]

            sys.path.insert(0, str(Path(__file__).parent.parent))
            from utility.drift_utils import luminance_histogram
            ref_hists_list = [h for p in ref_image_paths if (h := luminance_histogram(p)) is not None]
            if ref_hists_list and cur_hists.size > 0:
                ref_dist = np.mean(np.array(ref_hists_list), axis=0)
                cur_dist = np.mean(cur_hists, axis=0)
                drift_type = _deduce_drift_type(ref_dist, cur_dist)
    except Exception as exc:
        print(f"[ORACLE] Drift deduction failed: {exc}, defaulting to 'clear'.")

    # Build augmented retrain set
    retrain_img_dir = data_path / "images" / "retrain_aug_oracle"
    retrain_lbl_dir = data_path / "labels" / "retrain_aug_oracle"
    image_paths = sorted(list(ref_image_dir.glob("*.jpg")))[:n_ref_images]
    avg_hist = _create_augmented_set(image_paths, ref_label_dir, retrain_img_dir, retrain_lbl_dir, drift_type)

    if avg_hist is None:
        raise RuntimeError("Failed to create augmented dataset.")

    train_yaml = knowledge_path / "retrain_temp.yaml"
    with open(train_yaml, "w") as f:
        f.write(f"""path: {data_path.resolve()}
train: {retrain_img_dir.relative_to(data_path)}
val: {retrain_img_dir.relative_to(data_path)}
nc: 80
names: {{ {', '.join([f'{i}: {i}' for i in range(80)])} }}
""")

    model = YOLO(str(model_path))
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.model.model[-1].parameters():
        param.requires_grad = True

    try:
        with open(knowledge_path / "thresholds.json") as _tf:
            _oracle_thresholds = json.load(_tf)
    except Exception:
        _oracle_thresholds = {}
    energy_backend = _oracle_thresholds.get("energy_meter", "auto")

    with EnergyMeter("oracle_retrain", backend=energy_backend) as _em:
        model.train(data=str(train_yaml), epochs=epochs, imgsz=640, batch=4, workers=2,
                    patience=3, pretrained=True, cache="disk")
    energy_uJ = _em.total_uJ or 0.0

    # Archive in VMR
    best_pt_candidates = list(Path("runs/detect").rglob("weights/best.pt"))
    new_weights_src = str(best_pt_candidates[-1]) if best_pt_candidates else str(model_path)

    vmr = VMR(base_dir=vmr_base_dir)
    distribution = {"type": "histogram", "data": avg_hist.tolist()}
    version = vmr.store(
        model_name=model_name,
        weights_src=new_weights_src,
        distribution=distribution,
        tag="oracle",
    )

    # Copy to active models
    active_weights = active_models_path / f"{model_name}.pt"
    shutil.copy2(version.weights_path, active_weights)

    # Write reload flag
    flag_path = knowledge_path / "model_reload.flag"
    flag_path.touch()

    # Cleanup
    train_yaml.unlink(missing_ok=True)
    shutil.rmtree(retrain_img_dir, ignore_errors=True)
    shutil.rmtree(retrain_lbl_dir, ignore_errors=True)

    print(f"[ORACLE] Retrain complete. Version: {version.version_dir}, energy={energy_uJ:.0f} uJ")
    return {
        "status": "ok",
        "version_path": version.weights_path,
        "new_version_name": f"{model_name}_{version.timestamp}",
        "energy_uJ": energy_uJ,
    }


def _deduce_drift_type(ref_hist: np.ndarray, cur_hist: np.ndarray) -> str:
    centers = np.linspace(0, 255, len(ref_hist)) + 255 / len(ref_hist) / 2
    ref_mean = np.sum(ref_hist * centers)
    cur_mean = np.sum(cur_hist * centers)
    ref_std = np.sqrt(max(np.sum(ref_hist * (centers - ref_mean) ** 2), 1e-9))
    cur_std = np.sqrt(max(np.sum(cur_hist * (centers - cur_mean) ** 2), 1e-9))
    is_dark = (ref_mean > 0) and (cur_mean / ref_mean < 0.80)
    is_foggy = (ref_std > 0) and (cur_std / ref_std < 0.85)
    if is_dark:
        return "dark"
    if is_foggy:
        return "fog"
    return "clear"


def _create_augmented_set(image_paths, label_dir, out_img_dir, out_lbl_dir, drift_type) -> np.ndarray | None:
    import numpy as np
    from PIL import Image
    from torchvision.transforms.functional import adjust_brightness

    if out_img_dir.exists():
        shutil.rmtree(out_img_dir)
    if out_lbl_dir.exists():
        shutil.rmtree(out_lbl_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utility.drift_utils import luminance_histogram

    total_hist, count = None, 0
    for img_path in image_paths:
        try:
            with Image.open(img_path).convert("RGB") as img:
                if drift_type == "dark":
                    img = adjust_brightness(img, 0.35)
                elif drift_type == "fog":
                    arr = np.array(img, dtype=np.float32)
                    mean = np.mean(arr, axis=(0, 1), keepdims=True)
                    arr = np.clip((arr - mean) * 0.25 + mean, 0, 255)
                    img = Image.fromarray(arr.astype(np.uint8))
                img.save(out_img_dir / img_path.name)

            lbl = label_dir / f"{img_path.stem}.txt"
            if lbl.exists():
                shutil.copy(lbl, out_lbl_dir / lbl.name)

            h = luminance_histogram(out_img_dir / img_path.name)
            if h is not None:
                total_hist = h if total_hist is None else total_hist + h
                count += 1
        except Exception:
            continue

    return total_hist / count if count > 0 else None

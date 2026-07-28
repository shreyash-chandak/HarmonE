"""
managed_system_cv/retrain_tactics/pseudo_label.py — Pseudo-label fine-tuning.

Label-free retrain tactic:
1. Run the current model on the recent drift window.
2. Keep only detections with confidence > tau_pseudo (default 0.6).
3. Fine-tune on this pseudo-labeled set (last N layers only).
4. Safety valve: after fine-tuning, compute proxy score on a held-out
   subsample. If proxy score worsens by more than tau_regression (default 0.05),
   discard the new weights and return {"status": "skipped", "reason": "regression"}.
5. On success: archive via VMR, update active weights, write model_reload.flag.

Config keys (thresholds.json):
  tau_pseudo:      float (default 0.6) — min confidence to include as pseudo-label
  tau_regression:  float (default 0.05) — max allowed proxy score drop before discard
  pseudo_n_images: int   (default 500)  — number of drift-window images to pseudo-label
  pseudo_epochs:   int   (default 3)    — fine-tuning epochs
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.vmr import VMR
from core.energy import EnergyMeter


def run(
    model_name: str,
    drift_images: list[str],
    proxy,  # AccuracyProxy instance
    thresholds: dict,
    knowledge_dir: str = "knowledge",
    active_models_dir: str = "models",
    vmr_base_dir: str = "knowledge/vmr",
) -> dict:
    """Run pseudo-label fine-tuning tactic.

    Args:
        model_name:    e.g. "yolo_n"
        drift_images:  Paths to recent images in the drift window.
        proxy:         AccuracyProxy instance for post-fine-tune safety check.
        thresholds:    Loaded thresholds.json dict.
        knowledge_dir: Path to knowledge/ directory.
        active_models_dir: Path where active .pt weights live.
        vmr_base_dir:  Base directory for VMR archive.

    Returns:
        dict with keys: status, version_path, reason, energy_uJ.
    """
    if not drift_images:
        return {"status": "skipped", "reason": "no_images", "version_path": None}

    tau_pseudo = thresholds.get("tau_pseudo", 0.6)
    tau_regression = thresholds.get("tau_regression", 0.05)
    n_images = thresholds.get("pseudo_n_images", 500)
    epochs = thresholds.get("pseudo_epochs", 3)

    selected = drift_images[:n_images]

    try:
        return _do_pseudo_label(
            model_name=model_name,
            image_paths=[Path(p) for p in selected],
            proxy=proxy,
            tau_pseudo=tau_pseudo,
            tau_regression=tau_regression,
            epochs=epochs,
            knowledge_dir=Path(knowledge_dir),
            active_models_dir=Path(active_models_dir),
            vmr_base_dir=vmr_base_dir,
        )
    except Exception as exc:
        print(f"[PSEUDO] Pseudo-label retrain failed: {exc}")
        return {"status": "error", "reason": str(exc), "version_path": None}


def _do_pseudo_label(
    model_name, image_paths, proxy, tau_pseudo, tau_regression, epochs,
    knowledge_dir, active_models_dir, vmr_base_dir
) -> dict:
    from ultralytics import YOLO

    active_weights = active_models_dir / f"{model_name}.pt"
    if not active_weights.exists():
        raise FileNotFoundError(f"Active weights not found: {active_weights}")

    model = YOLO(str(active_weights))

    # Step 1: measure pre-tune proxy score on a held-out subset
    holdout = image_paths[-max(1, len(image_paths) // 5):]
    pre_score = _proxy_score_on_images(model, holdout, proxy, model_name)
    print(f"[PSEUDO] Pre-tune proxy score: {pre_score:.4f}")

    # Step 2: generate pseudo labels on training subset
    train_images = image_paths[: -max(1, len(image_paths) // 5)]
    pseudo_dir = knowledge_dir / "pseudo_tmp" / "images"
    pseudo_lbl_dir = knowledge_dir / "pseudo_tmp" / "labels"
    if pseudo_dir.exists():
        shutil.rmtree(pseudo_dir.parent)
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    pseudo_lbl_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    for img_path in train_images:
        results = model.predict(str(img_path), conf=tau_pseudo, verbose=False)
        if not results or not results[0].boxes:
            continue
        # Write YOLO-format label file
        label_lines = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            cx, cy, w, h = box.xywhn[0].tolist()
            label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if label_lines:
            shutil.copy(img_path, pseudo_dir / img_path.name)
            (pseudo_lbl_dir / f"{img_path.stem}.txt").write_text("\n".join(label_lines))
            kept += 1

    if kept < 10:
        shutil.rmtree(pseudo_dir.parent, ignore_errors=True)
        return {"status": "skipped", "reason": "too_few_pseudo_labels", "version_path": None}

    print(f"[PSEUDO] Kept {kept}/{len(train_images)} images with pseudo-labels (tau={tau_pseudo})")

    # Step 3: fine-tune (last layers only)
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.model.model[-1].parameters():
        param.requires_grad = True

    train_yaml = knowledge_dir / "pseudo_retrain.yaml"
    data_parent = pseudo_dir.parent.parent
    train_yaml.write_text(f"""path: {pseudo_dir.parent.parent.resolve()}
train: {pseudo_dir.relative_to(data_parent)}
val: {pseudo_dir.relative_to(data_parent)}
nc: 80
names: {{ {', '.join([f'{i}: {i}' for i in range(80)])} }}
""")

    energy_backend = thresholds.get("energy_meter", "auto")
    with EnergyMeter("pseudo_retrain", backend=energy_backend) as _em:
        model.train(data=str(train_yaml), epochs=epochs, imgsz=640, batch=4, workers=2,
                    patience=2, pretrained=False, cache="disk")
    energy_uJ = _em.total_uJ or 0.0

    # Step 4: safety valve — measure post-tune proxy score
    post_score = _proxy_score_on_images(model, holdout, proxy, model_name)
    print(f"[PSEUDO] Post-tune proxy score: {post_score:.4f}")

    shutil.rmtree(pseudo_dir.parent, ignore_errors=True)
    train_yaml.unlink(missing_ok=True)

    if post_score < pre_score - tau_regression:
        print(f"[PSEUDO] Safety valve triggered: post ({post_score:.4f}) < pre - tau ({pre_score - tau_regression:.4f}). Discarding.")
        return {
            "status": "skipped",
            "reason": "regression",
            "version_path": None,
            "pre_score": pre_score,
            "post_score": post_score,
        }

    # Step 5: archive and deploy
    best_pt_candidates = list(Path("runs/detect").rglob("weights/best.pt"))
    new_weights_src = str(best_pt_candidates[-1]) if best_pt_candidates else str(active_weights)

    vmr = VMR(base_dir=vmr_base_dir)
    version = vmr.store(
        model_name=model_name,
        weights_src=new_weights_src,
        distribution={"type": "raw", "data": []},
        tag="pseudo",
        proxy_score=post_score,
    )

    shutil.copy2(version.weights_path, active_weights)
    (knowledge_dir / "model_reload.flag").touch()

    print(f"[PSEUDO] Fine-tune accepted. Version: {version.version_dir}, energy={energy_uJ:.0f} uJ")
    return {
        "status": "ok",
        "version_path": version.weights_path,
        "energy_uJ": energy_uJ,
        "pre_score": pre_score,
        "post_score": post_score,
    }


def _proxy_score_on_images(model, image_paths, proxy, model_name: str) -> float:
    all_confs = []
    for img_path in image_paths:
        results = model.predict(str(img_path), verbose=False)
        if results and results[0].boxes:
            confs = results[0].boxes.conf.tolist()
            all_confs.extend(confs)
    record = {"model": model_name, "confidences": all_confs}
    return proxy.score(record)

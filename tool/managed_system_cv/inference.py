import os
import sys
import time
import pandas as pd
import torch
import shutil
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import numpy as np
import json
from tqdm import tqdm

# Make core/ importable when this script runs from managed_system_cv/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.energy import EnergyMeter
from utility.drift_utils import luminance_histogram

# Setup directories
os.makedirs("knowledge", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("versionedMR", exist_ok=True)
os.makedirs("knowledge/inferences", exist_ok=True)

# Load energy backend from thresholds
try:
    with open("knowledge/thresholds.json") as _tf:
        _inf_thresholds = json.load(_tf)
except Exception:
    _inf_thresholds = {}
_energy_backend = _inf_thresholds.get("energy_meter", "auto")

# Output file for inference results
results_file = "knowledge/predictions.csv"
if not os.path.exists(results_file):
    pd.DataFrame(columns=["image_name", "confidence", "model_used", "inference_time", "energy_uJ", "histogram"]).to_csv(results_file, index=False)

def calculate_and_save_initial_histogram(image_paths, output_path):
    """Calculates the average luminance histogram for a list of images and saves it."""
    print(f"Calculating initial histogram for {output_path.name}...")
    total_hist = None
    images_processed = 0
    for img_path in tqdm(image_paths, desc="  Analyzing initial data", leave=False, ncols=80):
        hist = luminance_histogram(img_path)
        if hist is not None:
            if total_hist is None:
                total_hist = np.zeros_like(hist)
            total_hist += hist
            images_processed += 1
    
    if images_processed > 0:
        avg_hist = total_hist / images_processed
        with open(output_path, 'w') as f:
            json.dump({"average_histogram": avg_hist.tolist()}, f, indent=4)
        print(f"✔ Saved initial histogram to {output_path}")

MODEL_PATHS = {
    "yolo_n": "models/yolo_n.pt",
    "yolo_s": "models/yolo_s.pt",
    "yolo_m": "models/yolo_m.pt"    
}

# L3: GPU compatibility guard — fail fast with a clear message instead of
# crashing mid-warmup with AcceleratorError.
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()  # e.g. ['sm_50', 'sm_60', ..., 'sm_90']
    target_sm = f"sm_{cc[0]}{cc[1]}"
    # Accept if the exact SM is listed, or if a higher SM is (forward compat)
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

print("--- Initializing Model Versions ---")
REF_IMAGE_DIR = Path("data/bdd100k/images/test")
N_REF_IMAGES = 1000
ref_image_paths = sorted(list(REF_IMAGE_DIR.glob("*.jpg")))[:N_REF_IMAGES]

for model_name, model_path in MODEL_PATHS.items():
    if not os.path.exists(model_path):
        print(f"⚠️  Base model {model_path} not found. Skipping versioning.")
        continue

    version_base_name = f"{model_name}_v1"
    versioned_model_path = Path("versionedMR") / f"{version_base_name}.pt"
    versioned_hist_path = Path("versionedMR") / f"{version_base_name}_hist.json"

    # Save version 1 of the model if it doesn't exist
    if not versioned_model_path.exists():
        shutil.copy(model_path, versioned_model_path)
        print(f"✔ Saved version 1 of {model_name} to {versioned_model_path}")
    
    # Save the corresponding histogram if it doesn't exist
    if not versioned_hist_path.exists():
        if not ref_image_paths:
            print(f"❌ Cannot generate histogram for {model_name}_v1: No reference images found in {REF_IMAGE_DIR}")
        else:
            calculate_and_save_initial_histogram(ref_image_paths, versioned_hist_path)

print("\n--- Starting Inference ---")
image_dir = Path("data/bdd100k/images/test")
image_files = sorted(list(image_dir.glob("*.jpg")))

current_model_name = None
model = None

for i, image_path in enumerate(image_files):
    try:
        with open("knowledge/model.csv", "r") as f:
            chosen_model = f.read().strip().lower()
    except FileNotFoundError:
        print("knowledge/model.csv not found. Defaulting to yolo_s.")
        chosen_model = "yolo_s"
    model_path = MODEL_PATHS.get(chosen_model, MODEL_PATHS["yolo_s"])
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Skipping inference for this image.")
        continue
    print(f"[{i+1}/{len(image_files)}] Running inference on {image_path.name} with model {chosen_model.upper()}...")
    model = YOLO(model_path)
    
    with EnergyMeter("inference", backend=_energy_backend) as _em:
        start_time = time.time()
        results = model(image_path, verbose=False)
        inference_time = time.time() - start_time
    energy_usage_uJ = _em.total_uJ or 0.0

    boxes = results[0].boxes
    top_conf = float(boxes.conf.mean().item()) if boxes is not None and len(boxes.conf) > 0 else 0.0

    hist = luminance_histogram(image_path)
    hist_str = ' '.join(f"{x:.8f}" for x in hist) if hist is not None else ''

    pd.DataFrame([[image_path.name, top_conf, chosen_model, inference_time, energy_usage_uJ, hist_str]],
                 columns=["image_name", "confidence", "model_used", "inference_time", "energy_uJ", "histogram"]).to_csv(
        results_file, mode="a", header=False, index=False
    )
    inference_txt_path = Path("knowledge/inferences") / f"{image_path.stem}.txt"
    with open(inference_txt_path, "w") as f:
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                img = Image.open(image_path)
                img_w, img_h = img.size
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
        else:
            f.write("")

print("\nYOLO Inference completed. Results saved in knowledge/predictions.csv")
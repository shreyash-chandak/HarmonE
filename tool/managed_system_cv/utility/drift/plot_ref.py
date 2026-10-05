#!/usr/bin/env python3
"""
plot_drift.py

Compute and plot KL divergence of images in BDD100K train set
relative to first N_REF images in test set. Supports:
- RGB histogram
- Luminance histogram (Y channel)

Usage:
    python3 plot_drift.py
"""

import os
import numpy as np
from PIL import Image
from scipy.stats import entropy
import matplotlib.pyplot as plt
from tqdm import tqdm

# =======================
# CONFIG
# =======================
TRAIN_DIR = "data/bdd100k/images/train"
TEST_DIR = "data/bdd100k/images/test"
N_REF = 500        # number of reference images from test set
WINDOW_SIZE = 500  # sliding window size
STEP_SIZE = 500     # sliding step
BINS = 16           # histogram bins

# =======================
# UTILS
# =======================
def load_image_paths(folder, limit=None):
    exts = (".jpg", ".jpeg", ".png")
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
    if limit:
        return files[:limit]
    return files

def rgb_histogram(img, bins=BINS):
    arr = np.asarray(img).astype(np.float32) / 255.0
    hists = []
    for c in range(arr.shape[2]):
        hist, _ = np.histogram(arr[...,c].ravel(), bins=bins, range=(0,1), density=True)
        hist += 1e-8
        hists.append(hist/np.sum(hist))
    return np.concatenate(hists)

def luminance_histogram(img, bins=BINS):
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 3:
        lum = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
    else:
        lum = arr
    hist, _ = np.histogram(lum.ravel(), bins=bins, range=(0,1), density=True)
    hist += 1e-8
    return hist/np.sum(hist)

def kl_divergence(p, q):
    return entropy(p, q)

def average_distribution(features):
    return np.mean(np.stack(features, axis=0), axis=0)

# =======================
# MAIN
# =======================
def main():
    print("Loading reference dataset...")
    test_imgs = load_image_paths(TEST_DIR, limit=N_REF)

    rgb_ref, lum_ref = [], []
    for path in tqdm(test_imgs, desc="Reference stats"):
        try:
            img = Image.open(path).convert("RGB").resize((64,32))
        except:
            continue
        rgb_ref.append(rgb_histogram(img))
        lum_ref.append(luminance_histogram(img))

    rgb_ref_dist = average_distribution(rgb_ref)
    lum_ref_dist = average_distribution(lum_ref)

    print("Processing train set...")
    train_imgs = load_image_paths(TRAIN_DIR)

    rgb_vals, lum_vals = [], []
    window_indices = []

    for start in tqdm(range(0, len(train_imgs)-WINDOW_SIZE+1, STEP_SIZE), desc="Sliding windows"):
        end = start + WINDOW_SIZE
        window = train_imgs[start:end]

        rgb_cur, lum_cur = [], []
        for path in window:
            try:
                img = Image.open(path).convert("RGB").resize((64,32))
            except:
                continue
            rgb_cur.append(rgb_histogram(img))
            lum_cur.append(luminance_histogram(img))

        if len(rgb_cur) == 0:
            continue

        rgb_cur_dist = average_distribution(rgb_cur)
        lum_cur_dist = average_distribution(lum_cur)

        rgb_vals.append(kl_divergence(rgb_cur_dist, rgb_ref_dist))
        lum_vals.append(kl_divergence(lum_cur_dist, lum_ref_dist))
        window_indices.append(start)

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(window_indices, rgb_vals, label="RGB hist KL")
    plt.plot(window_indices, lum_vals, label="Luminance hist KL")
    plt.xlabel("Train dataset index (window start)")
    plt.ylabel("KL divergence")
    plt.title("Drift detection: KL divergence vs reference")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
plot_drift.py

Compute and plot KL divergence of images in BDD100K train set
using sliding windows. Each window is compared to the *previous*
window (instead of a fixed reference).

Supports:
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
WINDOW_SIZE = 500   # number of images per window
STEP_SIZE = 500     # stride between windows
BINS = 16           # histogram bins
RESIZE = (64, 32)   # resize for faster histogramming

# =======================
# UTILS
# =======================
def load_image_paths(folder):
    exts = (".jpg", ".jpeg", ".png")
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
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

def compute_window_distribution(image_paths):
    rgb_cur, lum_cur = [], []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB").resize(RESIZE)
        except:
            continue
        rgb_cur.append(rgb_histogram(img))
        lum_cur.append(luminance_histogram(img))
    if len(rgb_cur) == 0:
        return None, None
    return average_distribution(rgb_cur), average_distribution(lum_cur)

# =======================
# MAIN
# =======================
def main():
    print("Loading train set...")
    train_imgs = load_image_paths(TRAIN_DIR)

    rgb_vals, lum_vals = [], []
    window_indices = []

    prev_rgb, prev_lum = None, None

    for start in tqdm(range(0, len(train_imgs)-WINDOW_SIZE+1, STEP_SIZE), desc="Sliding windows"):
        end = start + WINDOW_SIZE
        window = train_imgs[start:end]

        rgb_dist, lum_dist = compute_window_distribution(window)
        if rgb_dist is None:
            continue

        if prev_rgb is not None:  # compare to previous window
            rgb_vals.append(kl_divergence(rgb_dist, prev_rgb))
            lum_vals.append(kl_divergence(lum_dist, prev_lum))
            window_indices.append(start)

        prev_rgb, prev_lum = rgb_dist, lum_dist

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(window_indices, rgb_vals, label="RGB hist KL")
    plt.plot(window_indices, lum_vals, label="Luminance hist KL")
    plt.xlabel("Train dataset index (window start)")
    plt.ylabel("KL divergence (current vs previous window)")
    plt.title("Drift detection: KL divergence between consecutive windows")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

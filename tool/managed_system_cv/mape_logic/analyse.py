import os
import json
import re
import numpy as np
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.drift_utils import kl_divergence
from monitor import monitor_mape, monitor_drift

# Allow importing from tool/core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from core.scoring import update_energy_threshold

# Define the base directory dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

thresholds_file = os.path.join(KNOWLEDGE_DIR, "thresholds.json")
mape_info_file = os.path.join(KNOWLEDGE_DIR, "mape_info.json")
current_model_file = os.path.join(KNOWLEDGE_DIR, "model.csv")
predictions_file = os.path.join(KNOWLEDGE_DIR, "predictions.csv")
drift_kl_file = os.path.join(KNOWLEDGE_DIR, "drift_kl.json")
versioned_dir = "versionedMR"

ALL_MODELS = ["yolo_n", "yolo_s", "yolo_m"]

def load_mape_info():
    """Load stored MAPE info including energy threshold and recovery cycles."""
    try:
        with open(mape_info_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "last_line": 0,
            "current_energy_threshold": json.load(open(thresholds_file))["max_energy"],
            "ema_scores": {m: 0.5 for m in ALL_MODELS},
            "recovery_cycles": 0
        }

def save_mape_info(data):
    """Save updated MAPE info."""
    with open(mape_info_file, "w") as f:
        json.dump(data, f, indent=4)

def analyse_mape():
    """Analyze performance and decide if switching is needed, using dynamic energy thresholds and recovery cycles."""
    data = monitor_mape()
    if not data:
        print("[MAPE] No monitoring data available for analysis.")
        return None

    thresholds = json.load(open(thresholds_file))
    min_score = thresholds["min_score"]
    original_energy_threshold = thresholds["max_energy"]
    e_ref = thresholds.get("E_ref", 0.5)
    delta = thresholds.get("delta", 0.05)

    mape_info = load_mape_info()
    current_energy_threshold = mape_info.get("current_energy_threshold", original_energy_threshold)
    recovery_cycles = mape_info.get("recovery_cycles", 0)

    # B1 fix: Eq. 3 — adaptive threshold (was monotonically increasing, never tightened)
    used_energy_norm = data["normalized_energy"]
    new_energy_threshold = update_energy_threshold(
        current_energy_threshold, e_ref, used_energy_norm, delta,
        lo=0.1, hi=1.0,
    )
    mape_info["current_energy_threshold"] = new_energy_threshold

    switch_needed = False
    threshold_violated = None

    if recovery_cycles > 0:
        recovery_cycles -= 1
        print(f"[MAPE] Recovery mode active: {recovery_cycles} cycles remaining. No switching allowed.")
    else:
        if data["score"] < min_score:
            switch_needed = True
            threshold_violated = "score"
            print(f"[MAPE] Threshold violated: score ({data['score']:.4f} < {min_score})")

        if used_energy_norm > current_energy_threshold:
            switch_needed = True
            threshold_violated = "energy"
            recovery_cycles = 3
            print(f"[MAPE] Threshold violated: energy ({used_energy_norm:.4f} > {current_energy_threshold:.4f}). Entering recovery mode.")

    mape_info["recovery_cycles"] = recovery_cycles
    save_mape_info(mape_info)
    print(f"[MAPE] Updated Energy Threshold: {new_energy_threshold:.4f}")

    return {"switch_needed": switch_needed, "threshold_violated": threshold_violated, "score": data["score"]}


_EMBEDDING_DETECTORS = {"mmd_embedding", "frechet_embedding"}


def _version_distance(current_signal, version_base_name: str, detector_name: str) -> float:
    """Compute distance between current signal and a stored version signature.

    For luminance_kl: current_signal is a histogram array; compares against _hist.json.
    For embedding detectors: current_signal is a (N, D) embedding array; compares
      against _emb_sig.json using Fréchet distance on Gaussian approximations.
    Cross-type mismatch raises ValueError (I6).
    """
    if detector_name not in _EMBEDDING_DETECTORS:
        # Histogram path
        hist_path = os.path.join(versioned_dir, f"{version_base_name}_hist.json")
        if not os.path.exists(hist_path):
            return float("inf")
        with open(hist_path) as f:
            sig = json.load(f)
        sig_type = sig.get("type", "histogram")
        if sig_type == "embedding":
            raise ValueError(
                f"I6: version '{version_base_name}' has embedding signature "
                f"but drift_detector='{detector_name}' expects histogram. "
                "Cross-type VMR matching is not allowed."
            )
        version_dist = np.array(sig["average_histogram"])
        return float(kl_divergence(current_signal, version_dist))
    else:
        # Embedding path
        sig_path = os.path.join(versioned_dir, f"{version_base_name}_emb_sig.json")
        if not os.path.exists(sig_path):
            return float("inf")
        with open(sig_path) as f:
            sig = json.load(f)
        sig_type = sig.get("type", "embedding")
        if sig_type != "embedding":
            raise ValueError(
                f"I6: version '{version_base_name}' has histogram signature "
                f"but drift_detector='{detector_name}' expects embedding. "
                "Cross-type VMR matching is not allowed."
            )
        # Fréchet distance between Gaussians: mean of current vs version
        from core.drift.frechet_embedding import frechet_distance
        cur = np.asarray(current_signal, dtype=np.float64)
        cur_mu = np.mean(cur, axis=0)
        cur_sigma = np.cov(cur.T) if len(cur) > 1 else np.zeros((cur.shape[1],) * 2)
        ver_mu = np.array(sig["mean"], dtype=np.float64)
        # Version stores diagonal cov only — build full diagonal matrix
        ver_cov_diag = np.array(sig.get("cov_diag", [1.0] * len(ver_mu)), dtype=np.float64)
        ver_sigma = np.diag(ver_cov_diag)
        return frechet_distance(cur_mu, cur_sigma, ver_mu, ver_sigma)


def _load_current_embedding_window(thresholds: dict) -> "np.ndarray | None":
    """Load the most recent embedding window from EmbeddingStore for VMR matching."""
    try:
        import sys as _sys
        _tool_dir = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
        if _tool_dir not in _sys.path:
            _sys.path.insert(0, _tool_dir)
        from core.drift.embedding_store import EmbeddingStore
        emb_dim = int(thresholds.get("embedding_dim", 256))
        drift_window = int(thresholds.get("drift_window_size", 500))
        store = EmbeddingStore(KNOWLEDGE_DIR, embedding_dim=emb_dim, drift_window=drift_window)
        return store.last_window(drift_window)
    except Exception as exc:
        print(f"[DRIFT] Could not load EmbeddingStore for VMR matching: {exc}")
        return None


def get_best_version_for_model(model_name, current_signal, detector_name: str = "luminance_kl"):
    """Finds the best previous version for a SINGLE model type.

    Returns (best_version_path, best_distance) using the appropriate
    distance metric (KL for luminance, Fréchet for embedding).
    Cross-type signature mismatches raise ValueError (I6).
    """
    pattern = re.compile(f"({model_name}_v(\\d+))\\.pt")
    versions = [m.group(1) for f in os.listdir(versioned_dir) if (m := pattern.match(f))]
    if len(versions) <= 1:
        return None, float("inf")

    best_dist = float("inf")
    best_path = None

    for version_base_name in versions:
        try:
            dist = _version_distance(current_signal, version_base_name, detector_name)
            if dist < best_dist:
                best_dist = dist
                best_path = os.path.join(versioned_dir, f"{version_base_name}.pt")
        except ValueError:
            raise
        except Exception:
            continue

    return best_path, best_dist


def analyse_drift():
    """
    Analyze data drift. If detected, search across ALL model types (n, s, m)
    for the best existing version before triggering a retrain.
    """
    drift = monitor_drift()
    if not drift:
        print("[DRIFT] No drift monitoring data available for analysis.")
        return None

    kl_div = drift["kl_div"]
    if kl_div is None:
        print("[DRIFT] Warmup period — insufficient data for drift detection.")
        return {"drift_detected": False, "action": None, "version": None}

    thresholds = json.load(open(thresholds_file))
    tau_drift = thresholds.get("tau_drift", 0.07)

    if kl_div <= tau_drift:
        print(f"[DRIFT] No significant drift detected. KL ({kl_div:.4f}) <= {tau_drift}")
        return {"drift_detected": False, "action": None, "version": None}

    print(f"[DRIFT] Drift detected! KL ({kl_div:.4f}) > {tau_drift}")

    # --- Drift detected: find best version across ALL model types ---
    thresholds = json.load(open(thresholds_file))
    detector_name = thresholds.get("drift_detector", "luminance_kl")

    if detector_name in _EMBEDDING_DETECTORS:
        # Embedding-based version matching: use EmbeddingStore window
        current_signal = _load_current_embedding_window(thresholds)
        if current_signal is None:
            print("[DRIFT] EmbeddingStore underfilled — cannot compare versions. Planning retrain.")
            return {"drift_detected": True, "action": "retrain", "version": None}
    else:
        # Histogram-based version matching (luminance_kl)
        try:
            df = pd.read_csv(predictions_file)
            if len(df) < 1000:
                print("[DRIFT] Not enough data to compare versions. Planning retrain.")
                return {"drift_detected": True, "best_version": None, "action": "retrain"}
            drift_hists_str = df["histogram"].iloc[-1000:]
            drift_hists = np.array([np.fromstring(h, sep=" ") for h in drift_hists_str if h])
            if drift_hists.size == 0:
                return {"drift_detected": True, "best_version": None, "action": "retrain"}
            current_signal = np.mean(drift_hists, axis=0)
        except Exception as e:
            print(f"[DRIFT] Error processing current data for version comparison: {e}. Planning retrain.")
            return {"drift_detected": True, "best_version": None, "action": "retrain"}

    overall_best_version_path = None
    overall_min_dist = float("inf")
    all_dist_results = {}

    print("[DRIFT] Searching all model types for the best version to handle drift...")
    for model_name in ALL_MODELS:
        try:
            best_path, min_dist = get_best_version_for_model(model_name, current_signal, detector_name)
        except ValueError as ve:
            print(f"[DRIFT] I6 violation for {model_name}: {ve}")
            continue
        all_dist_results[model_name] = min_dist
        metric_name = "Fréchet" if detector_name in _EMBEDDING_DETECTORS else "KL"
        print(f"[DRIFT] Best version for {model_name.upper()}: {metric_name}={min_dist:.4f}")
        if min_dist < overall_min_dist:
            overall_min_dist = min_dist
            overall_best_version_path = best_path
    all_kl_results = all_dist_results  # alias for logging compat

    # Store results for debugging
    with open(drift_kl_file, "w") as f:
        json.dump({
            "best_overall_version": overall_best_version_path,
            "min_overall_dist": overall_min_dist,
            "dist_per_model": all_dist_results,
            "drift_detector": detector_name,
        }, f, indent=4)

    # Decide on the final action
    if overall_min_dist < tau_drift:
        print(f"[DRIFT] Found suitable version across all models: {overall_best_version_path} (dist={overall_min_dist:.4f})")
        return {"drift_detected": True, "action": "switch_version", "version": overall_best_version_path}
    else:
        print("[DRIFT] No suitable previous version found across any model type. A full retrain is needed.")
        return {"drift_detected": True, "action": "retrain", "version": None}
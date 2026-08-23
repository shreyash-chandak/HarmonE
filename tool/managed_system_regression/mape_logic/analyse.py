import os
import sys
import json
import numpy as np
import pandas as pd
from monitor import monitor_mape, monitor_drift

# Allow importing from tool/core
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.scoring import update_energy_threshold
from core.vmr import VMR

# Define the base directory dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")
VMR_DIR = os.path.join(BASE_DIR, "..", "knowledge", "vmr")

thresholds_file = os.path.join(KNOWLEDGE_DIR, "thresholds.json")
mape_info_file = os.path.join(KNOWLEDGE_DIR, "mape_info.json")
current_model_file = os.path.join(KNOWLEDGE_DIR, "model.csv")
drift_kl_file = os.path.join(KNOWLEDGE_DIR, "drift_kl.json")
drift_data_file = os.path.join(KNOWLEDGE_DIR, "drift.csv")
predictions_file = os.path.join(KNOWLEDGE_DIR, "predictions.csv")

def load_mape_info():
    """Load stored MAPE info including energy debt and recovery cycles."""
    with open(mape_info_file, "r") as f:
        return json.load(f)

def save_mape_info(data):
    """Save updated MAPE info including energy debt and recovery cycles."""
    with open(mape_info_file, "w") as f:
        json.dump(data, f, indent=4)

def analyse_mape():
    """Analyze performance and decide if switching is needed, using dynamic energy thresholds."""
    mape_data = monitor_mape()
    if not mape_data:
        print("⚠️ No MAPE data available for analysis.")
        return None

    # Load thresholds
    with open(thresholds_file, "r") as f:
        thresholds = json.load(f)

    min_score = thresholds["min_score"]
    original_energy_threshold = thresholds["max_energy"]
    e_ref = thresholds.get("E_ref", 0.7)
    delta = thresholds.get("delta", 0.1)

    # Load current MAPE info
    mape_info = load_mape_info()
    current_energy_threshold = mape_info.get("current_energy_threshold", original_energy_threshold)
    recovery_cycles = mape_info["recovery_cycles"]

    # B1 fix: Eq. 3 — adaptive threshold (was monotonically increasing, never tightened)
    used_energy = mape_data["normalized_energy"]
    new_energy_threshold = update_energy_threshold(
        current_energy_threshold, e_ref, used_energy, delta,
        lo=0.1, hi=1.0,
    )
    mape_info["current_energy_threshold"] = new_energy_threshold

    # Check if switching is needed
    switch_needed = False
    threshold_violated = None

    if recovery_cycles > 0:
        recovery_cycles -= 1
        print(f"⏳ Recovery mode active: {recovery_cycles} cycles remaining. No switching allowed.")
    else:
        if mape_data["score"] < min_score:
            print("⚠️ Model score too low! Model switch required.")
            switch_needed = True
            threshold_violated = "score"

        if used_energy > current_energy_threshold:
            print(f"⚠️ Energy threshold exceeded! Used: {used_energy:.4f}, Limit: {current_energy_threshold:.4f}")
            switch_needed = True
            threshold_violated = "energy"
            recovery_cycles = 3

    # Save updated info
    mape_info["recovery_cycles"] = recovery_cycles
    save_mape_info(mape_info)

    print(f"📊 Updated Energy Threshold: {new_energy_threshold:.4f}")

    return {
        "switch_needed": switch_needed,
        "score": mape_data["score"],
        "threshold_violated": threshold_violated
    }


def _vmr_best_match(model_name: str, drift_values: np.ndarray):
    """Search the VMR for the closest training distribution to the current drift data.

    Returns a VMRVersion on match, or None if the VMR is empty.
    Uses core/vmr.py VMR.best_match() with 'closest_distribution' strategy.
    """
    vmr = VMR(base_dir=VMR_DIR)
    hist, _ = np.histogram(drift_values, bins=50, density=True)
    distribution = {"type": "histogram", "data": hist.tolist()}
    version = vmr.best_match(model_name, distribution, strategy="closest_distribution")
    if version:
        print(f"🔎 VMR match for {model_name}: {version.weights_path}")
    return version

def analyse_drift():
    """Analyze drift & decide if retraining is needed or if an existing version can be used.

    Return contract (B2 fix — was using "best_version" key, plan_drift reads "action"/"version"):
      drift not detected → {"drift_detected": False, "action": None, "version": None}
      drift, VMR hit     → {"drift_detected": True,  "action": "replace", "version": <path>}
      drift, no VMR      → {"drift_detected": True,  "action": "retrain", "version": None}
    """
    drift_data = monitor_drift()
    if not drift_data:
        return None

    kl_div = drift_data["kl_div"]
    if kl_div is None:
        # Warmup period — not enough data for a reliable signal
        return {"drift_detected": False, "action": None, "version": None}

    # Load tau_drift from config (was hardcoded 0.5)
    with open(thresholds_file, "r") as f:
        thresholds = json.load(f)
    tau_drift = thresholds.get("tau_drift", 0.5)

    drift_detected = kl_div > tau_drift

    if drift_detected:
        print(f"🚨 Drift detected! KL divergence = {kl_div:.4f}")
        try:
            df = pd.read_csv(predictions_file)
            df.columns = df.columns.str.strip()
            df.tail(1200).to_csv(drift_data_file, index=False)
        except FileNotFoundError:
            print("No predictions file found to store drift data.")

        # Get the currently used model
        with open(current_model_file, "r") as f:
            current_model = f.read().strip()

        # Load drift data for VMR distribution matching
        try:
            drift_df = pd.read_csv(drift_data_file)
            drift_values = drift_df["true_value"].values
        except Exception as e:
            print(f"❌ Cannot read drift.csv for VMR search: {e}")
            return {"drift_detected": True, "action": "retrain", "version": None}

        vmr_match = _vmr_best_match(current_model, drift_values)

        if vmr_match:
            print(f"✔ VMR match found: {vmr_match.weights_path}")
            return {"drift_detected": True, "action": "replace", "version": vmr_match.weights_path}

        # No suitable previous version found → Retrain needed
        return {"drift_detected": True, "action": "retrain", "version": None}

    return {"drift_detected": False, "action": None, "version": None}
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import entropy
import json
import os

# Get the absolute path of the current script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

# Allow importing from tool/core
sys.path.insert(0, os.path.join(BASE_DIR, '..', '..'))
from core.drift.kl_fixed_ref import KLFixedRefDetector
from core.drift.kl_rolling import KLRollingDetector
from core.scoring import update_separated_emas

mape_info_file = os.path.join(KNOWLEDGE_DIR, "mape_info.json")
thresholds_file = os.path.join(KNOWLEDGE_DIR, "thresholds.json")
model_file = os.path.join(KNOWLEDGE_DIR, "model.csv")
predictions_file = os.path.join(KNOWLEDGE_DIR, "predictions.csv")

def load_mape_info():
    """Load MAPE info from JSON file."""
    with open(mape_info_file, "r") as f:
        return json.load(f)

def save_mape_info(data):
    """Save updated MAPE info including model-specific EMA scores."""
    with open(mape_info_file, "w") as f:
        json.dump(data, f, indent=4)

def get_current_model():
    """Fetch the currently active model from knowledge."""
    try:
        with open(model_file, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def monitor_mape():
    """Monitor R² Score and Actual Energy, and Compute Score."""
    info = load_mape_info()
    last_line = info["last_line"]
    current_model = get_current_model()
    
    if current_model is None:
        print("⚠️ No model currently in use.")
        return None

    try:
        df = pd.read_csv(predictions_file, skiprows=range(1, last_line + 1))
        df.columns = df.columns.str.strip()
        
        # L1-c: No new rows since last check — do NOT re-serve stale cached data.
        # Return fresh=False so the telemetry pusher skips the ACP POST entirely.
        if df.empty:
            print("📉 No new data in predictions.csv — skipping telemetry (inference may have stalled).")
            return {"fresh": False}
            
    except FileNotFoundError:
        print("⚠️ No predictions.csv file found.")
        return None

    print(f"🆕 Processing {len(df)} new rows from predictions.csv for {current_model.upper()}")

    # Calculate R² score
    r2 = r2_score(df["true_value"], df["predicted_value"])

    # Compute Actual and Normalized Energy
    with open(thresholds_file, "r") as f:
        thresholds = json.load(f)
    energy_min, energy_max = thresholds["E_m"], thresholds["E_M"]
    
    avg_energy = df["energy"].mean()
    print(f"Average energy: {avg_energy}, Min: {energy_min}, Max: {energy_max}")
    
    # Ensure energy normalization doesn't cause division by zero
    if energy_max > energy_min:
        energy_normalized = (avg_energy - energy_min) / (energy_max - energy_min)
        energy_normalized = max(0.0, min(1.0, energy_normalized))  # Clamp between 0 and 1
    else:
        energy_normalized = 0.0

    # Calculate model score (still use normalized energy for scoring)
    beta = thresholds.get("beta", 0.5)
    model_score = beta * r2 + (1 - beta) * (1 - energy_normalized)

    # Compute Exponential Moving Average (EMA)
    gamma = thresholds.get("gamma", 0.8)
    prev_score = info["ema_scores"].get(current_model, 0.5)
    final_score = gamma * model_score + (1 - gamma) * prev_score

    # Update MAPE info (combined score + Phase 2.4 separated signals)
    info["ema_scores"][current_model] = final_score
    update_separated_emas(info, current_model, r2, energy_normalized, gamma)
    info["last_line"] += len(df)

    # Log computed values
    print(f"🔹 R² Score: {r2:.4f}")
    print(f"🔹 Actual Energy: {avg_energy:.2f}")
    print(f"🔹 Normalized Energy: {energy_normalized:.4f}")
    print(f"🔹 Model Score for {current_model.upper()}: {model_score:.4f}")
    print(f"🔹 Updated EMA Score for {current_model.upper()}: {final_score:.4f}")

    save_mape_info(info)

    # Include event counters in telemetry
    event_counters = info.get("event_counters", {
        "model_switches": 0,
        "retrains": 0,
        "vmr_events": 0,
        "mape_k_energy_uJ": 0.0
    })
    
    # Include simple switch counters
    simple_switch_counters = info.get("simple_switch_counters", {
        "simple_switches": 0
    })
    
    print(f"📊 Event Counters - Switches: {event_counters['model_switches']}, Retrains: {event_counters['retrains']}, VMR: {event_counters['vmr_events']}, MAPE-K Energy: {event_counters['mape_k_energy_uJ']:.2f} µJ")

    return {
        "r2_score": round(r2, 4),
        "energy": round(avg_energy, 2),  # Return actual energy for display
        "normalized_energy": round(energy_normalized, 4),  # Keep for internal calculations
        "score": round(final_score, 4),
        "model_used": current_model,
        "model_switches": event_counters["model_switches"],
        "retrains": event_counters["retrains"],
        "vmr_events": event_counters["vmr_events"],
        "mape_k_energy_uJ": round(event_counters["mape_k_energy_uJ"], 2),
        "simple_switches": simple_switch_counters["simple_switches"]
    }

def monitor_drift():
    """Monitor data drift using fixed-reference KL divergence (B3 fix).

    B3 fix: uses KLFixedRefDetector (current ‖ training reference) rather than
    adjacent-window KL, which cannot detect gradual drift. The rolling signal is
    also computed and returned as secondary telemetry ("kl_div_rolling").

    B5 fix: returns {"kl_div": None} during warmup instead of a random placeholder
    that could falsely straddle the ACP secondary threshold.
    """
    reference_path = os.path.join(KNOWLEDGE_DIR, "reference_distribution.json")

    with open(thresholds_file, "r") as f:
        thresholds = json.load(f)
    tau_drift = thresholds.get("tau_drift", 0.5)
    drift_ref_mode = thresholds.get("drift_reference", "both")

    try:
        df = pd.read_csv(predictions_file)
        df.columns = df.columns.str.strip()

        if df.empty:
            print("Drift Monitor: No predictions yet.")
            return None

        values = df["true_value"].tolist()

        # Fixed-reference detector (primary signal — B3 fix)
        fixed_detector = KLFixedRefDetector(
            reference_path=reference_path,
            tau_drift=tau_drift,
            window_size=1200,
            n_bins=50,
        )
        fixed_result = fixed_detector.detect(values)

        # Rolling detector (secondary telemetry — paper's original method, retained for comparison)
        rolling_detector = KLRollingDetector(tau_drift=tau_drift, window_size=1200, n_bins=50)
        rolling_result = rolling_detector.detect(values)

        kl_primary = fixed_result["kl_div"] if drift_ref_mode != "rolling" else rolling_result["kl_div"]
        kl_rolling = rolling_result["kl_div"]

        print(f"🌊 Drift: KL_fixed={fixed_result['kl_div']} KL_rolling={kl_rolling}")

        return {
            "kl_div": kl_primary,
            "kl_div_rolling": kl_rolling,
            "drift_detected": fixed_result["drift_detected"],
        }

    except FileNotFoundError:
        print("Drift Monitor: No predictions found.")
        return None
    except Exception as e:
        print(f"Drift Monitor Error: {e}")
        return None
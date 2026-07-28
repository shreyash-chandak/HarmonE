import os
import shutil
import logging
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.energy import EnergyMeter
from plan import plan_mape, plan_drift, plan_random_switch, plan_greedy_switch, plan_simple_switch

# Define the base directory dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

MODEL_FILE = os.path.join(KNOWLEDGE_DIR, "model.csv")
MAPE_INFO_FILE = os.path.join(KNOWLEDGE_DIR, "mape_info.json")

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [execute.py] - %(levelname)-8s - %(message)s',
    stream=sys.stdout
)

def _load_thresholds() -> dict:
    try:
        with open(os.path.join(KNOWLEDGE_DIR, "thresholds.json")) as _f:
            return json.load(_f)
    except Exception:
        return {}

_thresholds = _load_thresholds()
_energy_backend = _thresholds.get("energy_meter", "auto")


def load_mape_info():
    """Load MAPE info with event counters and energy tracking."""
    try:
        with open(MAPE_INFO_FILE, "r") as f:
            info = json.load(f)
    except FileNotFoundError:
        info = {}
    
    # Ensure event counters exist
    if "event_counters" not in info:
        info["event_counters"] = {
            "model_switches": 0,
            "retrains": 0,
            "vmr_events": 0,
            "mape_k_energy_uJ": 0.0
        }
    
    # Ensure simple switch counters exist (separate from MAPE counters)
    if "simple_switch_counters" not in info:
        info["simple_switch_counters"] = {
            "simple_switches": 0
        }
    
    return info

def save_mape_info(info):
    """Save updated MAPE info including event counters."""
    with open(MAPE_INFO_FILE, "w") as f:
        json.dump(info, f, indent=4)

def record_event(event_type, energy_consumed=0.0, details=None):
    """Record an event and update counters."""
    info = load_mape_info()
    
    # Update counters
    if event_type == "switch":
        info["event_counters"]["model_switches"] += 1
        info["last_switch_ts"] = time.time()  # for switch_cooldown_s guard
        logging.info(f"📊 Event recorded: Model switch #{info['event_counters']['model_switches']}")
    elif event_type == "retrain":
        info["event_counters"]["retrains"] += 1
        logging.info(f"📊 Event recorded: Retrain #{info['event_counters']['retrains']}")
    elif event_type == "vmr":
        info["event_counters"]["vmr_events"] += 1
        logging.info(f"📊 Event recorded: VMR event #{info['event_counters']['vmr_events']}")
    elif event_type == "noop":
        logging.info("📊 Event recorded: MAPE no-op (MAPE-K energy still tracked)")
    
    # Add MAPE-K energy consumption
    info["event_counters"]["mape_k_energy_uJ"] += energy_consumed
    
    if details:
        logging.info(f"📊 Event details: {details}")
    if energy_consumed > 0:
        logging.info(f"⚡ MAPE-K energy consumed: {energy_consumed:.2f} µJ (Total: {info['event_counters']['mape_k_energy_uJ']:.2f} µJ)")
    
    save_mape_info(info)

def record_simple_switch():
    """Record a simple switch event (no energy tracking, just count)."""
    info = load_mape_info()
    
    # Update simple switch counter
    info["simple_switch_counters"]["simple_switches"] += 1
    
    logging.info(f"📊 Simple switch recorded: #{info['simple_switch_counters']['simple_switches']}")
    
    save_mape_info(info)

# --- Tactic Execution ---
def execute_mape(trigger="local"):
    """Switch to the best model based on planning."""
    logging.info("Executing MAPE (model switch)...")

    decision = None
    event_type = "noop"
    detail = ""

    with EnergyMeter("mape_k_execution", backend=_energy_backend) as _em:
        decision = plan_mape(trigger=trigger)

        if decision:
            try:
                try:
                    with open(MODEL_FILE, "r") as f:
                        old_model = f.read().strip()
                except FileNotFoundError:
                    old_model = "unknown"

                with open(MODEL_FILE, "w") as f:
                    f.write(decision)

                event_type = "switch"
                detail = f"Model switched from {old_model} to {decision}"
                logging.info(f"⚡ EXECUTE: Switching model to {decision.upper()} in {MODEL_FILE}")
            except Exception as e:
                event_type = "switch"
                detail = f"Failed to write model file: {e}"
                logging.error(f"EXECUTE: Failed to write to {MODEL_FILE}: {e}")
        else:
            # B6 fix: record noop, not switch
            event_type = "noop"
            detail = "No switch needed - planning returned no decision"
            logging.info("EXECUTE: No action needed (plan was empty).")

    energy_consumed = _em.total_uJ or 0.0
    record_event(event_type, energy_consumed, detail)

def execute_drift(trigger="local"):
    """Replaces model with best version or retrains if necessary."""
    logging.info("Executing Drift handling...")

    event_type = "vmr"
    detail = ""

    with EnergyMeter("mape_k_drift_execution", backend=_energy_backend) as _em:
        decision = plan_drift(trigger=trigger)

        if not decision:
            detail = "No drift action needed"
            logging.info("EXECUTE (Drift): No action needed.")

        elif decision["action"] == "replace":
            best_version_path = decision["version"]
            if not best_version_path or "version" not in best_version_path:
                detail = f"Invalid version path: {best_version_path}"
                logging.warning(f"EXECUTE (Drift): Invalid version path provided: {best_version_path}")
            else:
                model_name = os.path.basename(os.path.dirname(best_version_path))
                model_extension = ".pkl" if model_name in ["linear", "svm"] else ".pth"
                model_target_path = os.path.join(BASE_DIR, "..", "models", f"{model_name}{model_extension}")
                try:
                    shutil.copy(best_version_path, model_target_path)
                    detail = f"VMR: Switched to version {best_version_path}"
                    logging.info(f"✔ EXECUTE (Drift): Switched to lower KL divergence model: {best_version_path}")
                except Exception as e:
                    detail = f"Failed to copy model: {e}"
                    logging.error(f"EXECUTE (Drift): Failed to copy model: {e}")

        elif decision["action"] == "retrain":
            event_type = "retrain"
            logging.info("🚀 EXECUTE (Drift): Triggering retraining...")
            try:
                if os.path.exists("retrain.py"):
                    os.system("python retrain.py")
                    detail = "Model retrained due to drift"
                    logging.info("EXECUTE (Drift): Retraining script finished.")
                else:
                    detail = "Retrain.py not found"
                    logging.warning("EXECUTE (Drift): 'retrain.py' not found. Skipping.")
            except Exception as e:
                detail = f"Retraining failed: {e}"
                logging.error(f"EXECUTE (Drift): Retraining failed: {e}")

    energy_consumed = _em.total_uJ or 0.0
    record_event(event_type, energy_consumed, detail)

def execute_random_switch(trigger="local"):
    """Switches model based on the random_switch plan (A1 rename of execute_simple_switch)."""
    logging.info("Executing Random Switch (R² Baseline)...")

    decision = plan_random_switch(trigger=trigger)
    
    if not decision:
        logging.info("EXECUTE (Simple Switch): No action needed.")
        return

    logging.info(f"⚡ EXECUTE (Simple Switch): Switching model to {decision.upper()} in {MODEL_FILE}")
    try:
        # Get current model for logging
        try:
            with open(MODEL_FILE, "r") as f:
                old_model = f.read().strip()
        except FileNotFoundError:
            old_model = "unknown"
        
        with open(MODEL_FILE, "w") as f:
            f.write(decision)
        
        # Record the simple switch event (no energy tracking)
        record_simple_switch()
        
        logging.info(f"📊 Random switch: {old_model} → {decision}")
        logging.info("EXECUTE (Random Switch): Model switch successful.")
    except Exception as e:
        logging.error(f"EXECUTE (Random Switch): Failed to write to {MODEL_FILE}: {e}")


# Backward-compat alias
execute_simple_switch = execute_random_switch
import os
import shutil
import sys
import time
import re
import json
import csv
import logging

# Define the base directory dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "..", "..")))
from core.energy import EnergyMeter

model_file = os.path.join(KNOWLEDGE_DIR, "model.csv")
mape_info_file = os.path.join(KNOWLEDGE_DIR, "mape_info.json")
event_log_file = os.path.join(KNOWLEDGE_DIR, "event_log.csv")
predictions_file = os.path.join(KNOWLEDGE_DIR, "predictions.csv")

# --- IMPORT ALL THREE planners ---
from plan import plan_mape, plan_drift, plan_random_switch, plan_greedy_switch, plan_simple_switch

models_dir = "models"

def _load_thresholds() -> dict:
    try:
        with open(os.path.join(KNOWLEDGE_DIR, "thresholds.json")) as _f:
            return json.load(_f)
    except Exception:
        return {}

_thresholds = _load_thresholds()
_energy_backend = _thresholds.get("energy_meter", "auto")


def load_mape_info():
    """Loads the mape_info JSON file with event counters and energy tracking."""
    try:
        with open(mape_info_file, "r") as f:
            info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        info = {
            "last_line": 0,
            "current_energy_threshold": 0.6,
            "ema_scores": {"yolo_n": 0.5, "yolo_s": 0.5, "yolo_m": 0.5},
            "recovery_cycles": 0
        }
    
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

def save_mape_info(data):
    """Saves data to the mape_info JSON file."""
    with open(mape_info_file, "w") as f:
        json.dump(data, f, indent=4)

def get_last_prediction_line():
    try:
        with open(predictions_file, "r") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

def log_event(event_type, model=None, version=None, details=None):
    last_line = get_last_prediction_line()
    log_entry = {
        "event_type": event_type,
        "last_line": last_line,
        "model": model or "",
        "version": version or "",
        "details": details or ""
    }
    file_exists = os.path.isfile(event_log_file)
    with open(event_log_file, "a", newline="") as csvfile:
        fieldnames = ["event_type", "last_line", "model", "version", "details"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

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

def execute_mape(trigger="local"):
    """Execute a model switch based on the MAPE plan."""
    print("[MAPE-EXEC] Planning model switch...")

    event_type = "noop"
    detail = ""
    decision = None

    with EnergyMeter("mape_k_cv_execution", backend=_energy_backend) as _em:
        decision = plan_mape(trigger=trigger)

        if decision:
            print(f"[MAPE-EXEC] Executing switch to model: {decision.upper()}")
            try:
                try:
                    with open(model_file, "r") as f:
                        old_model = f.read().strip()
                except FileNotFoundError:
                    old_model = "unknown"

                with open(model_file, "w") as f:
                    f.write(decision)

                log_event("switch", model=decision)
                event_type = "switch"
                detail = f"Model switched from {old_model} to {decision}"
                print(f"⚡ Switched active model to {decision.upper()}")
            except Exception as e:
                event_type = "switch"
                detail = f"Failed to write model file: {e}"
                print(f"[MAPE-EXEC] Error switching model: {e}")
        else:
            # B6 fix: record noop, not switch
            event_type = "noop"
            detail = "No switch needed - planning returned no decision"
            print("[MAPE-EXEC] No model switch needed.")

    energy_consumed = _em.total_uJ or 0.0
    record_event(event_type, energy_consumed, detail)

def execute_drift(trigger="local"):
    """Execute the drift response: switch to a previous version or trigger retraining."""
    print("[DRIFT-EXEC] Planning drift response...")

    event_type = "vmr"
    detail = ""

    with EnergyMeter("mape_k_cv_drift_execution", backend=_energy_backend) as _em:
        decision = plan_drift(trigger=trigger)

        if not decision:
            detail = "No drift action needed"
            print("[DRIFT-EXEC] No drift action needed.")

        else:
            action = decision.get("action")
            print(f"[DRIFT-EXEC] Drift action planned: {action}")

            if action == "switch_version":
                version_path = decision["version_path"]
                if not os.path.exists(version_path):
                    detail = f"Version path does not exist: {version_path}"
                    print(f"[DRIFT-EXEC] Error: Version path '{version_path}' does not exist. Cannot switch.")
                else:
                    base_name_match = re.search(r'(yolo_[nsm])', os.path.basename(version_path))
                    if not base_name_match:
                        detail = f"Could not determine base model name from {version_path}"
                        print(f"[DRIFT-EXEC] Error: Could not determine base model name from '{version_path}'.")
                    else:
                        base_name = base_name_match.group(1)
                        destination_path = os.path.join(models_dir, f"{base_name}.pt")
                        try:
                            shutil.copy(version_path, destination_path)
                            print(f"[DRIFT-EXEC] Copied '{version_path}' to '{destination_path}'.")
                            with open(model_file, "w") as f:
                                f.write(base_name)

                            log_event("vmr", model=base_name, version=os.path.basename(version_path),
                                      details=f"Switched to versioned model at {version_path}")
                            detail = f"VMR: Switched to version {version_path}"
                            print(f"⚡ Switched active model to version: {os.path.basename(version_path)}")

                            # A2 fix: EMA head-start is now config-driven (was hardcoded +0.1)
                            ema_head_start = _thresholds.get("ema_head_start", 0.1)
                            if ema_head_start > 0:
                                print(f"[DRIFT-EXEC] Applying EMA head-start of {ema_head_start} for {base_name.upper()} (A2)...")
                                mape_info = load_mape_info()
                                current_score = mape_info["ema_scores"].get(base_name, 0.5)
                                new_score = min(1.0, current_score + ema_head_start)
                                mape_info["ema_scores"][base_name] = new_score
                                save_mape_info(mape_info)
                                log_event("ema_head_start", model=base_name, details=f"+{ema_head_start} applied after VMR deploy")
                                print(f"[DRIFT-EXEC] EMA score for {base_name.upper()}: {current_score:.4f} → {new_score:.4f}.")
                            time.sleep(20)
                        except Exception as e:
                            detail = f"Failed to copy versioned model: {e}"
                            print(f"[DRIFT-EXEC] Error copying versioned model: {e}")

            elif action == "retrain":
                event_type = "retrain"
                print("[DRIFT-EXEC] Triggering retraining...")
                try:
                    RETRAIN_PATH = os.path.join(BASE_DIR, "..", "retrain.py")
                    os.system(f"{sys.executable} {RETRAIN_PATH}")
                    log_event("retrain", details="Retraining triggered by drift detection.")
                    detail = "Model retrained due to drift"
                    time.sleep(20)
                except Exception as e:
                    detail = f"Retraining failed: {e}"
                    print(f"[DRIFT-EXEC] Error during retraining: {e}")

    energy_consumed = _em.total_uJ or 0.0
    record_event(event_type, energy_consumed, detail)

def execute_random_switch(trigger="local"):
    """Executes a random model switch (A1 rename of execute_simple_switch)."""
    logging.info("Executing Random Switch (Confidence Baseline)...")

    decision = plan_random_switch(trigger=trigger)
    
    if not decision:
        logging.info("EXECUTE (Random Switch): No action needed.")
        return

    logging.info(f"⚡ EXECUTE (Random Switch): Switching model to {decision.upper()} in {model_file}")
    
    try:
        # Get current model for logging
        try:
            with open(model_file, "r") as f:
                old_model = f.read().strip()
        except FileNotFoundError:
            old_model = "unknown"
        
        with open(model_file, "w") as f:
            f.write(decision)
        
        # Record both old CSV log and new simple switch counter
        log_event("switch", model=decision, details="confidence_baseline_switch")
        record_simple_switch()
        
        logging.info(f"📊 Random switch: {old_model} → {decision}")
        logging.info("EXECUTE (Random Switch): Model switch successful.")

    except Exception as e:
        logging.error(f"EXECUTE (Random Switch): Failed to write to {model_file}: {e}")


# Backward-compat alias
execute_simple_switch = execute_random_switch
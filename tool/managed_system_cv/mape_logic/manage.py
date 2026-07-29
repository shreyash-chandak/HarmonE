import threading
import time
import csv
import os
import sys
import json
import pandas as pd
import logging

# --- IMPORTANT: Import all your execute functions ---
# Make sure your execute.py has all three of these
from execute import execute_mape, execute_drift, execute_simple_switch

# Make core/ importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.energy import EnergyMeter

# --- File Paths ---
# Define the base directory dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

# Load energy backend from thresholds
try:
    with open(os.path.join(KNOWLEDGE_DIR, "thresholds.json")) as _tf:
        _thresholds = json.load(_tf)
except Exception:
    _thresholds = {}
_energy_backend = _thresholds.get("energy_meter", "auto")

log_file = os.path.join(KNOWLEDGE_DIR, "mape_log.csv")
predictions_file = os.path.join(KNOWLEDGE_DIR, "predictions.csv")
drift_file = os.path.join(KNOWLEDGE_DIR, "drift.csv")
COMMAND_FILE_PATH = os.path.join(KNOWLEDGE_DIR, "command.txt")
config_file = os.path.join("approach.conf")

_CMD_MAX_AGE_S = 30

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CV-Manage] - %(levelname)s - %(message)s')

# Ensure 'knowledge' directory exists
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Ensure log file exists
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["function", "energy_uJ"])

# --- Helper Functions (Unchanged) ---
def log_energy(function_name, energy_uJ):
    """Appends the energy consumption of a function to the log file."""
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([function_name, energy_uJ])

def get_line_count(filepath):
    """Efficiently counts the number of lines in a file."""
    try:
        with open(filepath, 'r') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

# --- Legacy "Local Brain" Functions (Kept for standalone use) ---
# These functions are NO LONGER CALLED in ACP mode.
def run_execute_mape_local():
    logging.info("Starting local MAPE loop (timer-based)...")
    while True:
        time.sleep(40) # Fixed time-based trigger
        logging.info("[Local-Manage] Triggering local execute_mape()...")
        execute_mape(trigger="local") # Pass "local" trigger

def run_execute_drift_local():
    logging.info("Starting local Drift loop (file-based)...")
    last_run_lines = max(0, get_line_count(predictions_file) - 1)
    DRIFT_CHECK_INTERVAL = 500
    POLLING_SECONDS = 20
    while True:
        time.sleep(POLLING_SECONDS)
        current_lines = max(0, get_line_count(predictions_file) - 1)
        if current_lines >= last_run_lines + DRIFT_CHECK_INTERVAL:
            logging.info(f"[Local-Manage] Triggering local execute_drift()...")
            execute_drift(trigger="local") # Pass "local" trigger
            last_run_lines = current_lines

# --- NEW: ACP-Driven Tactic "Router" ---
def execute_tactic_locally(tactic_id):
    """
    Executes the correct local CV logic based on the tactic_id from the ACP.
    """
    logging.info(f"Command '{tactic_id}' received. Triggering local CV logic...")
    
    with EnergyMeter(tactic_id, backend=_energy_backend) as _em:
        # These IDs must match your policies in the /policies folder
        if tactic_id == "execute_mape_plan":
            execute_mape(trigger="acp")

        elif tactic_id == "handle_data_drift":
            execute_drift(trigger="acp")

        elif tactic_id == "switch_model_r2_baseline":
            execute_mape(trigger="acp")

        elif tactic_id == "random_switch":
            execute_simple_switch(trigger="acp")

        else:
            logging.warning(f"Unknown local tactic_id: '{tactic_id}'")

    log_energy(tactic_id, _em.total_uJ or 0.0)

def _clear_stale_command() -> None:
    """L2-a: truncate command.txt on startup to prevent cross-session replay."""
    if os.path.exists(COMMAND_FILE_PATH):
        open(COMMAND_FILE_PATH, "w").close()
        logging.info("Cleared stale command.txt from previous session.")


def _parse_and_validate_command(raw: str) -> "str | None":
    """L2-c: parse 'tactic_id|unix_ts' and reject commands older than _CMD_MAX_AGE_S."""
    if not raw:
        return None
    if '|' not in raw:
        logging.warning("Received legacy command without timestamp — discarding (L2-c).")
        return None
    tactic_id, ts_str = raw.rsplit('|', 1)
    tactic_id = tactic_id.strip()
    try:
        age = time.time() - float(ts_str)
        if age > _CMD_MAX_AGE_S:
            logging.warning(
                f"Discarding stale command '{tactic_id}' (age {age:.1f}s > {_CMD_MAX_AGE_S}s)."
            )
            return None
    except ValueError:
        logging.warning(f"Bad timestamp in command '{raw}' — discarding.")
        return None
    return tactic_id


# --- NEW: ACP-Driven Command Listener ---
def acp_command_listener():
    """
    Runs in a thread and waits for commands from the ACP (via command.txt).
    This REPLACES all the old timed execution loops.
    """
    logging.info(f"ACP Command Listener started. Waiting for '{COMMAND_FILE_PATH}'...")
    while True:
        if os.path.exists(COMMAND_FILE_PATH):
            try:
                with open(COMMAND_FILE_PATH, 'r') as f:
                    raw = f.read().strip()
                os.remove(COMMAND_FILE_PATH)

                tactic_id = _parse_and_validate_command(raw)
                if tactic_id:
                    logging.info(f"Received command '{tactic_id}' from ACP Wrapper.")
                    execute_tactic_locally(tactic_id)

            except Exception as e:
                logging.error(f"Error processing command file: {e}")
                if os.path.exists(COMMAND_FILE_PATH):
                    os.remove(COMMAND_FILE_PATH)

        time.sleep(1)

# --- Configuration Loader (Unchanged) ---
def get_approach_config():
    """Reads the approach configuration from the config file."""
    if not os.path.exists(config_file):
        logging.warning(f"'{config_file}' not found. Defaulting to 'harmone_acp'.")
        return "harmone_acp" # Default to ACP mode
    with open(config_file, "r") as f:
        return f.read().strip().lower()

# --- REVISED: Main Execution Logic ---
if __name__ == "__main__":
    _clear_stale_command()  # L2-a: clear any leftover command from a previous session
    approach = get_approach_config()
    logging.info(f"Running config: {approach}")

    threads = []
    
    if "acp" in approach:
        # --- NEW ACP-DRIVEN MODE ---
        # This will be true for "harmone_acp", "switch_acp", etc.
        logging.info("Running in ACP-Driven mode. Starting command listener.")
        t_listener = threading.Thread(target=acp_command_listener, daemon=True)
        threads.append(t_listener)

    elif approach in ["harmone", "switch"]:
        # --- LEGACY "LOCAL BRAIN" MODE ---
        logging.info("Running in LOCAL mode. Starting internal timers.")
        t_mape = threading.Thread(target=run_execute_mape_local, daemon=True)
        threads.append(t_mape)
        if approach == "harmone":
            t_drift = threading.Thread(target=run_execute_drift_local, daemon=True)
            threads.append(t_drift)
    
    elif "single" in approach:
         # --- SINGLE/MONITOR-ONLY MODE ---
        logging.info("Running in 'single' (monitor-only) mode. No management threads started.")
    
    else:
        logging.warning(f"Unknown approach '{approach}'. No management threads started.")

    if threads:
        logging.info(f"Starting {len(threads)} management thread(s)...")
        for t in threads:
            t.start()
    else:
        logging.info("No management threads to start for this approach.")

    try:
        # Keep the main thread alive to let daemon threads run
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Exiting.")
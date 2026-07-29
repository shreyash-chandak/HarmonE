import threading
import time
import os
import sys
import logging
from execute import execute_mape, execute_drift, execute_simple_switch

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [manage.py] - %(levelname)-8s - %(message)s',
    stream=sys.stdout
)

# Define the base directory dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

# File paths
COMMAND_FILE_PATH = os.path.join(KNOWLEDGE_DIR, "command.txt")
APPROACH_CONFIG_FILE = "approach.conf"
LOG_FILE = os.path.join(KNOWLEDGE_DIR, "mape_log.csv")
PREDICTIONS_FILE = os.path.join(KNOWLEDGE_DIR, "predictions.csv")
DRIFT_FILE = os.path.join(KNOWLEDGE_DIR, "drift.csv")

_CMD_MAX_AGE_S = 30  # discard commands older than this

# --- Local Tactic Execution ---
# def execute_tactic_locally(tactic_id):
#     """Executes the correct local logic based on the tactic_id."""
#     logging.info(f"Command '{tactic_id}' received. Triggering local logic...")
    
#     if tactic_id == "execute_mape_plan":
#         # This calls plan.py with the "acp" trigger
#         execute_mape(trigger="acp")
        
#     elif tactic_id == "handle_data_drift":
#         # This calls plan.py (for drift) and then execute_drift
#         execute_drift(trigger="acp")
        
#     else:
#         logging.warning(f"Unknown local tactic_id: '{tactic_id}'")

# --- Main MAPE Loop ---
def run_mape_loop(approach):
    """The main loop that drives the local MAPE logic."""
    
    if approach == "harmone_local":
        # --- Original HarmonE Logic ---
        logging.info("Running in 'harmone_local' mode. Using internal timer.")
        while True:
            time.sleep(40) # Original 40-second timer
            logging.info("Local timer triggered. Running MAPE plan...")
            execute_mape(trigger="local")
            
            # Add drift check logic here if needed
            # time.sleep(400)
            # execute_drift(trigger="local")

    elif approach in ["harmone_acp", "switch_acp"]:
        # --- ACP-Driven Logic ---
        logging.info("Running in 'harmone_acp' mode. Listening for commands...")
        while True:
            try:
                if os.path.exists(COMMAND_FILE_PATH):
                    with open(COMMAND_FILE_PATH, 'r') as f:
                        raw = f.read().strip()
                    os.remove(COMMAND_FILE_PATH)

                    tactic_id = _parse_and_validate_command(raw)
                    if tactic_id:
                        execute_tactic_locally(tactic_id)

            except FileNotFoundError:
                pass
            except Exception as e:
                logging.error(f"Error in ACP command loop: {e}")

            time.sleep(5)
    else:
        logging.info(f"Mode '{approach}' requires no local MAPE loop. Exiting.")


def _clear_stale_command() -> None:
    """L2-a: truncate command.txt on startup to prevent cross-session replay."""
    if os.path.exists(COMMAND_FILE_PATH):
        open(COMMAND_FILE_PATH, "w").close()
        logging.info("Cleared stale command.txt from previous session.")


def _parse_and_validate_command(raw: str) -> str | None:
    """L2-c: parse 'tactic_id|unix_ts' and reject commands older than _CMD_MAX_AGE_S.

    Legacy format (no pipe) is treated as expired — safe default.
    Returns the tactic_id string if fresh, else None.
    """
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


def execute_tactic_locally(tactic_id):
    """Executes the correct local logic based on the tactic_id."""
    logging.info(f"Command '{tactic_id}' received. Triggering local logic...")
    
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

# --- Startup ---
if __name__ == "__main__":
    _clear_stale_command()  # L2-a: clear any leftover command from a previous session

    try:
        with open(APPROACH_CONFIG_FILE, 'r') as f:
            approach = f.read().strip().lower()
    except FileNotFoundError:
        logging.error(f"'{APPROACH_CONFIG_FILE}' not found. Cannot start.")
        sys.exit(1)

    run_mape_loop(approach)


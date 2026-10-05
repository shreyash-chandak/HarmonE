import time
import requests
import threading
import logging
import subprocess
import os
import json
import importlib.util
import sys
import shutil
import signal
import psutil
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MasterWrapper] - %(levelname)s - %(message)s')

# --- Configuration ---
ACP_SERVER_URL = "http://localhost:5000"
APPROACH_CONFIG_FILE = "approach.conf"
POLICY_DIR = "policies"
HANDLER_PORT = 8080

# --- Dynamically set logic paths ---
LOGIC_PATH = ""
COMMAND_FILE_PATH = ""
monitor_mape = None
monitor_drift = None

# --- Global process tracking ---
subprocesses = []
should_shutdown = False

def get_python_command():
    """Detect whether to use 'python' or 'python3' command."""
    if shutil.which("python3"):
        return "python3"
    elif shutil.which("python"):
        return "python"
    else:
        logging.critical("FATAL: Neither 'python' nor 'python3' command found in PATH.")
        exit(1)

def cleanup_processes():
    """Clean up all subprocesses and related processes."""
    global subprocesses, should_shutdown
    should_shutdown = True
    
    logging.info("Starting process cleanup...")
    
    # Terminate direct subprocesses
    for p in subprocesses:
        try:
            if p.poll() is None:  # Process is still running
                logging.info(f"Terminating subprocess PID {p.pid}")
                p.terminate()
                
                # Wait for graceful termination, then force kill if needed
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logging.warning(f"Process PID {p.pid} didn't terminate gracefully, force killing...")
                    p.kill()
        except Exception as e:
            logging.error(f"Error terminating subprocess PID {p.pid}: {e}")
    
    # Clean up any orphaned Python processes related to our systems
    try:
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']):
            try:
                # Skip our own process
                if proc.info['pid'] == current_pid:
                    continue
                    
                # Look for Python processes that might be our inference/manage processes
                if proc.info['name'] in ['python', 'python3'] and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if ('inference.py' in cmdline or 'manage.py' in cmdline or 
                        'managed_system_cv' in cmdline or 'managed_system_regression' in cmdline):
                        logging.info(f"Killing orphaned process PID {proc.info['pid']}: {cmdline}")
                        proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logging.error(f"Error during orphaned process cleanup: {e}")
    
    subprocesses.clear()
    logging.info("Process cleanup completed")

# --- Adaptation Handler API (Listens for commands from ACP) ---
handler_app = Flask(__name__)

@handler_app.route('/adaptor/tactic', methods=['POST'])
def execute_tactic_from_acp():
    """L2-c: Writes 'tactic_id|unix_ts' to command file so manage.py can reject stale replays."""
    global should_shutdown
    if should_shutdown:
        return jsonify({"error": "System is shutting down"}), 503

    data = request.json
    tactic_id = data.get("tactic_id")
    logging.info(f"[ACP_Handler] Command received: '{tactic_id}'")
    try:
        with open(COMMAND_FILE_PATH, "w") as f:
            f.write(f"{tactic_id}|{time.time()}")
        logging.info(f"[ACP_Handler] Command '{tactic_id}' queued in {COMMAND_FILE_PATH}.")
        return jsonify({"message": "Command queued."}), 200
    except Exception as e:
        logging.error(f"[ACP_Handler] Failed to write command file: {e}")
        return jsonify({"error": "Failed to queue command"}), 500

@handler_app.route('/adaptor/shutdown', methods=['POST'])
def shutdown_system():
    """Endpoint to shutdown all managed system processes."""
    global should_shutdown
    logging.info("[ACP_Handler] Shutdown command received")
    should_shutdown = True
    
    # Clean up processes in a separate thread to avoid blocking the response
    threading.Thread(target=cleanup_processes, daemon=True).start()
    
    # Also exit this process after a short delay
    def delayed_exit():
        time.sleep(2)
        os._exit(0)
    
    threading.Thread(target=delayed_exit, daemon=True).start()
    
    return jsonify({"message": "System shutdown initiated"}), 200

@handler_app.route('/adaptor/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global should_shutdown
    if should_shutdown:
        return jsonify({"status": "shutting_down"}), 503
    return jsonify({"status": "running", "processes": len(subprocesses)}), 200

def run_handler_api():
    handler_app.run(host='0.0.0.0', port=HANDLER_PORT)

_STALE_LIMIT = 6   # consecutive stale intervals before system_alert (6 × 5s = 30s)

# --- Telemetry & Policy Functions ---
def push_telemetry():
    """L1-c: Pushes telemetry; skips POST on stale data; alerts after stale_limit cycles.

    Polls subprocess liveness each cycle — kills the wrapper if any child dies.
    """
    global monitor_mape, monitor_drift, should_shutdown
    if not monitor_mape and not monitor_drift:
        logging.critical("Monitors not imported. Exiting telemetry thread.")
        return

    logging.info("[Monitor] Telemetry thread started.")
    time.sleep(2)

    stale_cycles = 0

    while not should_shutdown:
        # L1-c: subprocess liveness check — dead child = abort immediately
        for p in list(subprocesses):
            rc = p.poll()
            if rc is not None:
                logging.critical(
                    f"[Monitor] Subprocess PID {p.pid} exited with code {rc}. "
                    f"Shutting down — fix the subprocess error and restart."
                )
                should_shutdown = True
                cleanup_processes()
                import os as _os
                _os._exit(1)

        try:
            mape_metrics = monitor_mape() if monitor_mape else None

            # L1-c: monitor returns {"fresh": False} when inference has no new data
            if mape_metrics and mape_metrics.get("fresh") is False:
                stale_cycles += 1
                logging.warning(
                    f"[Monitor] Stale cycle {stale_cycles}/{_STALE_LIMIT} — "
                    f"no new inference data."
                )
                if stale_cycles >= _STALE_LIMIT:
                    logging.error(
                        "[Monitor] Inference stalled — posting system_alert to ACP."
                    )
                    try:
                        requests.post(
                            f"{ACP_SERVER_URL}/api/telemetry",
                            json={"timestamp": time.time(), "system_alert": "inference_stalled"},
                            timeout=3,
                        )
                    except Exception:
                        pass
                    stale_cycles = 0  # reset so we don't flood
            else:
                stale_cycles = 0
                telemetry_payload = {"timestamp": time.time()}

                if mape_metrics:
                    telemetry_payload.update(mape_metrics)

                if monitor_drift:
                    drift_metrics = monitor_drift()
                    if drift_metrics:
                        telemetry_payload.update(drift_metrics)

                if len(telemetry_payload) > 1:
                    logging.info(f"[Monitor] Pushing telemetry: {telemetry_payload}")
                    requests.post(
                        f"{ACP_SERVER_URL}/api/telemetry",
                        json=telemetry_payload,
                        timeout=3,
                    )
                else:
                    logging.info("[Monitor] No new data from monitors.")

        except Exception as e:
            if not should_shutdown:
                logging.error(f"[Monitor] Error in telemetry loop: {e}", exc_info=True)

        time.sleep(5)

    logging.info("[Monitor] Telemetry thread shutting down")

def register_policies_with_acp(policy_prefix):
    """
    Registers policies from the POLICY_DIR that match the prefix
    (e.g., 'cv_harmone' prefix will load 'cv_harmone_score.json').
    """
    if 'single' in policy_prefix:
        logging.info("Running in 'single' mode. No policies will be registered.")
        return True

    try:
        policy_files = [
            f for f in os.listdir(POLICY_DIR) 
            if f.endswith('.json') and f.startswith(policy_prefix)
        ]
    except FileNotFoundError:
        logging.error(f"FATAL: Policy directory '{POLICY_DIR}' not found.")
        return False
        
    if not policy_files:
        logging.error(f"FATAL: No policies found in '{POLICY_DIR}' with prefix '{policy_prefix}'. Required for non-single modes.")
        logging.error(f"Expected policy file like: '{policy_prefix}_score.json' or similar in '{POLICY_DIR}' directory.")
        return False # Make this fatal for non-single modes

    logging.info(f"Found {len(policy_files)} policies to register: {policy_files}")
    
    for policy_file in policy_files:
        try:
            policy_path = os.path.join(POLICY_DIR, policy_file)
            with open(policy_path, 'r') as f:
                policy = json.load(f)
            
            policy_id = policy.get("policy_id")
            if not policy_id:
                logging.error(f"'{policy_file}' is missing 'policy_id'. Skipping.")
                continue

            requests.post(f"{ACP_SERVER_URL}/api/policy", json=policy, timeout=3)
            logging.info(f"Policy '{policy_id}' from '{policy_file}' registered.")
        
        except requests.exceptions.RequestException as e:
            logging.critical(f"FATAL: Could not connect to ACP at {ACP_SERVER_URL}.")
            return False
        except Exception as e:
            logging.error(f"FATAL: Error registering policy '{policy_file}': {e}")
            return False
            
    return True

def import_monitor_from_path(logic_path):
    """Helper function to dynamically import the monitor module."""
    global monitor_mape, monitor_drift
    try:
        monitor_path = os.path.join(logic_path, "mape_logic", "monitor.py")
        spec = importlib.util.spec_from_file_location("monitor", monitor_path)
        monitor_module = importlib.util.module_from_spec(spec)
        
        sys.path.insert(0, logic_path) # Add to path so monitor's internal imports work
        spec.loader.exec_module(monitor_module)
        sys.path.pop(0) # Clean up sys.path
        
        # We need to handle if a monitor doesn't exist (e.g. no drift)
        monitor_mape = getattr(monitor_module, "monitor_mape", None)
        monitor_drift = getattr(monitor_module, "monitor_drift", None)
        
        if not monitor_mape:
             logging.warning(f"Could not find 'monitor_mape' in {monitor_path}")
        if not monitor_drift:
             logging.warning(f"Could not find 'monitor_drift' in {monitor_path}")
             
    except Exception as e:
        logging.critical(f"FATAL: Could not import monitors from '{monitor_path}': {e}")
        exit(1)

# --- Main Execution Logic ---
if __name__ == '__main__':
    # 1. Read Master Configuration
    try:
        with open(APPROACH_CONFIG_FILE, 'r') as f:
            approach = f.read().strip().lower()
    except FileNotFoundError:
        logging.critical(f"FATAL: '{APPROACH_CONFIG_FILE}' not found. Please create it.")
        exit(1)
        
    # --- MODIFIED BLOCK STARTS HERE ---
    if approach.startswith("custom_"):
        # Handle Custom Mode
        system_type = "custom" # This helps us distinguish in logs
        run_mode = approach.split('_')[1] # 'regression' or 'cv' effectively
        LOGIC_PATH = "managed_system_custom" # <--- The key change: Point to the new dir
        logging.info(f"--- Running CUSTOM System based on: '{run_mode.upper()}' ---")
    
    # Standard Modes
    elif '_' in approach:
        system_type, run_mode = approach.split('_', 1)
        
        if system_type == "cv":
            LOGIC_PATH = "managed_system_cv"
        elif system_type == "reg":
            LOGIC_PATH = "managed_system_regression"
        else:
            logging.critical(f"FATAL: Unknown system_type '{system_type}'.")
            exit(1)
            
        logging.info(f"--- Running System: '{system_type.upper()}' in Mode: '{run_mode.upper()}' ---")
    else:
        logging.critical(f"FATAL: Invalid approach format '{approach}'.")
        exit(1)
    # --- MODIFIED BLOCK ENDS HERE ---

    # 2. Parse Configuration (e.g., "cv_harmone")
    # if not '_' in approach:
    #     logging.critical(f"FATAL: Invalid approach '{approach}'. Must be format 'system_mode' (e.g., 'cv_harmone').")
    #     exit(1)
        
    # system_type, run_mode = approach.split('_', 1)
    
    # if system_type == "cv":
    #     LOGIC_PATH = "managed_system_cv"
    # elif system_type == "reg":
    #     LOGIC_PATH = "managed_system_regression"
    # else:
    #     logging.critical(f"FATAL: Unknown system_type '{system_type}'. Must be 'cv' or 'reg'.")
    #     exit(1)
    
    # logging.info(f"--- Running System: '{system_type.upper()}' in Mode: '{run_mode.upper()}' ---")
    
    # 3. L1-b: Startup artifact health check — abort with remediation if missing
    _REQUIRED_ARTIFACTS = {
        "regression": [
            os.path.join(LOGIC_PATH, "knowledge", "scaler.pkl"),
            os.path.join(LOGIC_PATH, "knowledge", "reference_distribution.json"),
            os.path.join(LOGIC_PATH, "knowledge", "thresholds.json"),
            os.path.join(LOGIC_PATH, "knowledge", "model.csv"),
        ],
        "cv": [
            os.path.join(LOGIC_PATH, "knowledge", "thresholds.json"),
            os.path.join(LOGIC_PATH, "knowledge", "model.csv"),
        ],
    }
    domain_key = "regression" if system_type == "reg" else "cv"
    for artifact_path in _REQUIRED_ARTIFACTS.get(domain_key, []):
        if not os.path.exists(artifact_path):
            init_cmd = (
                "python3 scripts/init_regression.py"
                if domain_key == "regression"
                else "python3 scripts/init_cv.py"
            )
            logging.critical(
                f"FATAL: Required artifact missing: {artifact_path}\n"
                f"Run:  {init_cmd} --config <dataset_name>\n"
                f"Then restart."
            )
            exit(1)

    # 4. Dynamically import the correct logic
    import_monitor_from_path(LOGIC_PATH)

    # 5. Set up knowledge path and write the *local* config for manage.py
    KNOWLEDGE_PATH = os.path.join(LOGIC_PATH, "knowledge")
    LOCAL_APPROACH_CONFIG = os.path.join(LOGIC_PATH, "approach.conf")
    COMMAND_FILE_PATH = os.path.join(KNOWLEDGE_PATH, "command.txt") # Set global var

    os.makedirs(KNOWLEDGE_PATH, exist_ok=True)
    with open(LOCAL_APPROACH_CONFIG, "w") as f:
        # Pass the correct mode to the local manage.py
        f.write(f"{run_mode}_acp") # e.g., "harmone_acp", "switch_acp"
    
    # 5. Start Background Threads
    threading.Thread(target=push_telemetry, daemon=True).start()

    # 6. Start Subprocesses from the correct logic path
    try:
        python_cmd = get_python_command()
        inference_cmd = [python_cmd, "-u", "inference.py"]
        manage_cmd = [python_cmd, "-u", "mape_logic/manage.py"]

        # Pass the logic path as a working directory so all file paths are correct
        subprocesses.append(subprocess.Popen(inference_cmd, cwd=LOGIC_PATH))
        logging.info(f"Inference engine '{inference_cmd[2]}' started in '{LOGIC_PATH}'.")
        
        if 'single' not in run_mode:
            subprocesses.append(subprocess.Popen(manage_cmd, cwd=LOGIC_PATH))
            logging.info(f"MAPE logic '{manage_cmd[2]}' started in '{LOGIC_PATH}'.")
        else:
            logging.info("Running in 'single' (monitor-only) mode. 'manage.py' will not be started.")

    except Exception as e:
        logging.critical(f"Failed to start subprocesses: {e}")
        exit(1)

    # 7. Handle ACP-specific setup
    policy_prefix = f"{system_type}_{run_mode}" # e.g., "cv_harmone"
    
    if not register_policies_with_acp(policy_prefix):
        cleanup_processes()
        exit(1)
    
    if 'single' not in run_mode:
        threading.Thread(target=run_handler_api, daemon=True).start()
        logging.info(f"Adaptation Handler API listening on http://0.0.0.0:{HANDLER_PORT}...")

    # 8. Wait for processes to finish
    try:
        logging.info("Wrapper is running. (Press Ctrl+C to stop)")
        for p in subprocesses: p.wait()
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Terminating subprocesses...")
        cleanup_processes()
    finally:
        logging.info("Wrapper script finished.")
import time
import random
import requests
import threading
from flask import Flask, request, jsonify
import logging
import json

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ManagedSys] - %(levelname)s - %(message)s')

# --- Configuration ---
ACP_TELEMETRY_URL = "http://localhost:5000/api/telemetry"
ACP_POLICY_URL = "http://localhost:5000/api/policy"
ADAPTATION_HANDLER_PORT = 8080

# --- State Simulation ---
# This global variable simulates the state of the deployed model.
# The Adaptation Handler will change this state upon receiving a command.
CURRENT_MODEL_STATE = {
    "model_name": "LARGE_FP32",
    "adapted": False
}

# --- Adaptation Handler API (Listens for commands from ACP) ---
handler_app = Flask(__name__)

@handler_app.route('/adaptor/tactic', methods=['POST'])
def execute_tactic():
    """This endpoint receives the execution command from the ACP."""
    data = request.json
    tactic_id = data.get("tactic_id")
    logging.info(f"Received command to execute tactic: '{tactic_id}'")

    if tactic_id == "apply_model_quantization":
        logging.info("Executing the 'apply_model_quantization' tactic...")
        logging.info("--> Simulating model swap: Changing from LARGE_FP32 to QUANTIZED_INT8.")
        
        # Change the global state to simulate the effect of the adaptation
        CURRENT_MODEL_STATE["model_name"] = "QUANTIZED_INT8"
        CURRENT_MODEL_STATE["adapted"] = True
        
        logging.info("--> Tactic execution complete.")
        return jsonify({"message": f"Tactic '{tactic_id}' executed successfully."}), 200
    else:
        logging.warning(f"Received unknown tactic_id: '{tactic_id}'")
        return jsonify({"error": "Unknown tactic"}), 400

def run_handler_api():
    """Runs the Flask app in a separate thread."""
    handler_app.run(host='0.0.0.0', port=ADAPTATION_HANDLER_PORT)

# --- Model Monitoring Agent (Pushes telemetry to ACP) ---
def start_monitoring():
    """Generates and pushes telemetry data in a loop."""
    logging.info("Model Monitoring Agent started.")
    
    while True:
        try:
            if not CURRENT_MODEL_STATE["adapted"]:
                # State 1: Before adaptation (high energy, high accuracy)
                energy = random.uniform(480, 600) # Occasionally spikes above 500
                accuracy = random.uniform(0.94, 0.96)
            else:
                # State 2: After adaptation (low energy, slightly lower accuracy)
                energy = random.uniform(300, 350)
                accuracy = random.uniform(0.88, 0.91)

            telemetry_payload = {
                "timestamp": time.time(),
                "energy_consumption_mj": round(energy, 2),
                "current_model": CURRENT_MODEL_STATE["model_name"],
                "model_accuracy": round(accuracy, 3)
            }
            
            logging.info(f"[MONITOR] Pushing telemetry: {telemetry_payload}")
            requests.post(ACP_TELEMETRY_URL, json=telemetry_payload, timeout=3)
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to push telemetry to ACP: {e}")
        
        # Wait for the next monitoring cycle
        time.sleep(5)

# --- Main Execution ---
if __name__ == '__main__':
    # Start the Adaptation Handler API in a background thread
    handler_thread = threading.Thread(target=run_handler_api)
    handler_thread.daemon = True
    handler_thread.start()
    logging.info(f"Adaptation Handler API listening on port {ADAPTATION_HANDLER_PORT}...")
    time.sleep(2) # Give the server a moment to start

    # On startup, load the policy from the JSON file
    try:
        with open('policy.json', 'r') as f:
            SUSTAINABILITY_POLICY = json.load(f)
        logging.info("Policy configuration loaded from policy.json")
    except FileNotFoundError:
        logging.critical("Error: policy.json not found in the managed_system directory.")
        exit(1)
    except json.JSONDecodeError:
        logging.critical("Error: Could not decode policy.json. Please check for valid JSON format.")
        exit(1)

    # Push the loaded policy to the ACP to configure it
    try:
        logging.info("Registering adaptation policy with the ACP...")
        requests.post(ACP_POLICY_URL, json=SUSTAINABILITY_POLICY, timeout=3)
        logging.info("Policy successfully registered.")
    except requests.exceptions.RequestException as e:
        logging.critical(f"Could not register policy with ACP. Is the ACP server running? Error: {e}")
        exit(1)

    # Start the main monitoring loop
    start_monitoring()


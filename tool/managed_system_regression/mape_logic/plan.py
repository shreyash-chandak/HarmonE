import json
import random
import logging
import os
import sys
import time
from analyse import analyse_mape, analyse_drift  # analyse_mape is ONLY for local mode

# Allow importing from tool/core
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.planners.base import PlanningContext, get_planner

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [plan.py] - %(levelname)-8s - %(message)s',
    stream=sys.stdout
)

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

THRESHOLDS_FILE = os.path.join(KNOWLEDGE_DIR, "thresholds.json")
MODEL_FILE = os.path.join(KNOWLEDGE_DIR, "model.csv")
MAPE_INFO_FILE = os.path.join(KNOWLEDGE_DIR, "mape_info.json")

_AVAILABLE_MODELS = ["lstm", "linear", "svm"]

# --- Helper Functions ---
def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return {}

def get_current_model():
    try:
        with open(MODEL_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "lstm" # Default

# ---------------------------------------------------------------------------
# Phase 2 dispatcher: routes through the planner registry.
# ---------------------------------------------------------------------------

def dispatch_plan(violation: str | None, trigger: str = "local") -> dict | None:
    """Build a PlanningContext and delegate to the configured planner.

    Args:
        violation: "score" | "energy" | "drift" | None
        trigger:   "local" | "acp"

    Returns:
        None for noop, or a dict compatible with the existing execute.py contract:
          {"action": "switch",   "model": <name>}
          {"action": "replace",  "version": <path>}
          {"action": "retrain"}
    """
    thresholds = load_json(THRESHOLDS_FILE)
    mape_info = load_json(MAPE_INFO_FILE)
    current_model = get_current_model()

    # Run drift analysis only when needed
    drift_result = None
    if violation == "drift":
        drift_result = analyse_drift()
        if drift_result and not drift_result.get("drift_detected"):
            drift_result = None  # no real drift; treat as noop

    ctx = PlanningContext(
        violation=violation,
        ema_scores=mape_info.get("ema_scores", {}),
        ema_accuracy=mape_info.get("ema_accuracy", {}),
        ema_energy=mape_info.get("ema_energy", {}),
        current_model=current_model,
        available_models=_AVAILABLE_MODELS,
        thresholds=thresholds,
        drift_result=drift_result,
    )

    planner_name = thresholds.get("planner", "harmone_original")
    try:
        planner = get_planner(planner_name)
    except (KeyError, NotImplementedError) as exc:
        logging.error(f"DISPATCH: Failed to load planner '{planner_name}': {exc}")
        return None

    decision = planner.plan(ctx)
    logging.info(f"DISPATCH [{planner_name}]: {decision.action} reason={decision.reason!r}")

    if decision.action == "noop":
        return None
    if decision.action == "switch":
        return decision.model  # execute.py writes this string to model.csv
    if decision.action in ("replace", "switch_version"):
        return {"action": "replace", "version": decision.version_path}
    if decision.action == "retrain":
        return {"action": "retrain"}
    return None


# --- Main Planning Logic (legacy wrappers — kept for backward compat) ---
def plan_mape(trigger="local"):
    """
    Decides on the best model to use.
    - trigger='local': Runs full analysis first (original HarmonE).
    - trigger='acp': Skips analysis and proceeds to planning (ACP-driven).
    """
    logging.info(f"PLAN (MAPE) triggered by: {trigger.upper()}")

    # L5 anti-thrash cooldown: skip switch if the last one was too recent
    thresholds = load_json(THRESHOLDS_FILE)
    cooldown_s = thresholds.get("switch_cooldown_s", 30)
    mape_info = load_json(MAPE_INFO_FILE)
    last_switch_ts = mape_info.get("last_switch_ts", 0.0)
    elapsed = time.time() - last_switch_ts
    if elapsed < cooldown_s:
        logging.info(
            f"PLAN: Cooldown active — last switch was {elapsed:.1f}s ago "
            f"(cooldown={cooldown_s}s). Returning noop."
        )
        return None

    # 1. ANALYZE (Only for local mode)
    if trigger == "local":
        logging.info("Running in 'local' mode, performing local analysis...")
        analysis = analyse_mape()
        if not analysis or not analysis["switch_needed"]:
            logging.info("Local analysis: No switch needed.")
            return None
        logging.info("Local analysis: Violation detected, proceeding to plan.")
    
    elif trigger == "acp":
        logging.info("Running in 'acp' mode. ACP detected violation. Proceeding to plan.")
        # We skip the local 'analyse_mape()' because the ACP has already made the decision
    
    else:
        logging.warning(f"Unknown trigger '{trigger}'. Aborting plan.")
        return None

    # 2. PLAN (This logic is now shared by both modes)
    
    # Load local knowledge (thresholds for alpha, ema_scores for planning)
    thresholds = load_json(THRESHOLDS_FILE)
    mape_info = load_json(MAPE_INFO_FILE)
    ema_scores = mape_info.get("ema_scores", {})
    alpha = thresholds.get("alpha", 0.1) # Exploration probability
    
    current_model = get_current_model()

    # Tactic 1: Exploration (using alpha)
    if random.random() < alpha:
        available_models = [m for m in ema_scores.keys() if m != current_model]
        if not available_models:
             logging.warning("PLAN: Exploration tactic: No alternative models to explore.")
             return None
        chosen_model = random.choice(available_models)
        logging.info(f"🎲 PLAN: Exploratory tactic! Randomly selecting '{chosen_model.upper()}'.")
    
    # Tactic 2: Exploitation (choose best alternative)
    else:
        best_alternative = sorted(ema_scores.items(), key=lambda x: x[1], reverse=True)
        # Find the best model that is NOT the current one
        chosen_model = next((m for m, score in best_alternative if m != current_model), None)
        
        if not chosen_model:
            # This happens if all other models have a score of 0 or are not listed
            logging.warning("PLAN: Exploitation tactic: No valid alternatives found. Sticking with current model.")
            return None
            
        logging.info(f"🏆 PLAN: Exploitation tactic: Best alternative to '{current_model.upper()}' is '{chosen_model.upper()}' (Score: {ema_scores.get(chosen_model, 'N/A'):.2f}).")

    # Final check: Don't switch if we're already on the best model
    if chosen_model == current_model:
        logging.info(f"PLAN: Already using the chosen model ('{chosen_model.upper()}'). No switch needed.")
        return None

    return chosen_model

def plan_drift(trigger="local"):
    """Decides if retraining or model replacement is needed."""
    logging.info(f"PLAN (Drift) triggered by: {trigger.upper()}")
    
    drift_analysis = None # <-- Initialize variable
    
    # --- THIS IS THE FIX ---
    if trigger == "local":
        logging.info("Running in 'local' mode, performing local drift analysis...")
        drift_analysis = analyse_drift()
        if not drift_analysis or not drift_analysis["drift_detected"]:
            logging.info("PLAN (Drift): No drift detected. No action required.")
            return None
    
    elif trigger == "acp":
        logging.info("Running in 'acp' mode. ACP detected drift. Running analysis to find solution...")
        # Even when triggered by ACP, we still need to run analyse_drift() 
        # to find out *what* to do (replace or retrain).
        # But we skip the first "is drift detected?" check.
        drift_analysis = analyse_drift()
        if not drift_analysis:
            logging.warning("PLAN (Drift): ACP triggered, but local analysis failed.")
            return None
    # --- END OF FIX ---

    if drift_analysis.get("action") == "replace":
        logging.info(f"PLAN (Drift): Switching to lower KL divergence model: {drift_analysis['version']}")
        return {"action": "replace", "version": drift_analysis["version"]}
    
    logging.info("PLAN (Drift): Drift detected! No previous version available. Retraining required.")
    return {"action": "retrain"}

def plan_random_switch(trigger="local"):
    """Random baseline: picks one of the other available models uniformly at random.

    A1 rename: was plan_simple_switch. Kept deterministically seeded by the caller
    (random.choice) — this is the S2/random-switch baseline in the paper.
    """
    logging.info(f"PLAN (Random Switch): Triggered by {trigger.upper()} R² baseline.")
    current_model = get_current_model()
    available_models = ["lstm", "linear", "svm"]

    if current_model in available_models:
        available_models.remove(current_model)

    chosen_model = random.choice(available_models)
    logging.info(f"PLAN (Random Switch): Switching from '{current_model.upper()}' to '{chosen_model.upper()}'.")
    return chosen_model


# Backward-compat alias so existing callers (e.g. ACP policy files) keep working
plan_simple_switch = plan_random_switch


def plan_greedy_switch(trigger="local"):
    """Greedy baseline (S3): always switch to the highest-EMA non-current model.

    Unlike plan_mape which uses epsilon-greedy exploration, this planner exploits
    only. It is used as the S3 comparison baseline in the journal extension.
    """
    logging.info(f"PLAN (Greedy Switch): Triggered by {trigger.upper()}.")
    thresholds = load_json(THRESHOLDS_FILE)
    mape_info = load_json(MAPE_INFO_FILE)
    ema_scores = mape_info.get("ema_scores", {})
    current_model = get_current_model()

    best_alternative = sorted(ema_scores.items(), key=lambda x: x[1], reverse=True)
    chosen_model = next((m for m, _ in best_alternative if m != current_model), None)

    if not chosen_model:
        logging.warning("PLAN (Greedy Switch): No alternative models found.")
        return None

    if chosen_model == current_model:
        logging.info("PLAN (Greedy Switch): Already on best model, no switch.")
        return None

    logging.info(
        f"PLAN (Greedy Switch): Best alternative to '{current_model.upper()}' is "
        f"'{chosen_model.upper()}' (EMA={ema_scores.get(chosen_model, 'N/A')})."
    )
    return chosen_model
"""
managed_system_cv/retrain_tactics/human_in_loop.py — Human-in-the-loop tactic.

When drift is detected and neither oracle nor pseudo-label retrain is feasible:
1. Emit a "labeling_required" event to mape_info.json event log.
2. Switch the active model to the lowest-energy candidate (to reduce cost
   while waiting for labels).
3. Poll knowledge/incoming_labels/ on every subsequent MAPE-K tick.
4. When labels arrive (YOLO .txt files for queued images), run oracle retrain
   on those specific images, then clear the queue directory.

State file: knowledge/hil_state.json
  {"status": "waiting"|"idle", "queued_since": ISO timestamp, "images": [...]}

Directory: knowledge/incoming_labels/<image_stem>.txt
  Standard YOLO label format. Created externally by human annotators.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


HIL_STATE_FILE = "knowledge/hil_state.json"
INCOMING_LABELS_DIR = "knowledge/incoming_labels"


def request_labels(
    drift_images: list[str],
    ema_energy: dict[str, float],
    available_models: list[str],
    knowledge_dir: str = "knowledge",
) -> dict:
    """Emit labeling_required event and switch to lowest-energy model.

    Args:
        drift_images:     Recent image paths that need annotation.
        ema_energy:       {model_name: normalized_energy_ema} from mape_info.
        available_models: All candidate model names.
        knowledge_dir:    Path to knowledge/ directory.

    Returns:
        dict with keys: status, lowest_energy_model, queued_count.
    """
    knowledge_path = Path(knowledge_dir)
    labels_dir = knowledge_path / "incoming_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Persist images that need annotation as a JSON queue
    state = {
        "status": "waiting",
        "queued_since": datetime.now(tz=timezone.utc).isoformat(),
        "images": drift_images,
    }
    state_path = knowledge_path / "hil_state.json"
    tmp = str(state_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path)

    # Find lowest-energy model from EMA
    lowest_model = _lowest_energy_model(ema_energy, available_models)

    print(f"[HIL] Labeling required for {len(drift_images)} images. "
          f"Switching to lowest-energy model: {lowest_model}.")
    print(f"[HIL] Place YOLO label files in: {labels_dir}")

    # Log event to mape_info
    _log_event(knowledge_path, "labeling_required", {
        "queued_images": len(drift_images),
        "switched_to": lowest_model,
    })

    return {
        "status": "waiting",
        "lowest_energy_model": lowest_model,
        "queued_count": len(drift_images),
    }


def check_and_process(
    knowledge_dir: str = "knowledge",
    active_models_dir: str = "models",
    vmr_base_dir: str = "knowledge/vmr",
    model_name: str | None = None,
) -> dict:
    """Check incoming_labels/ for new annotations and retrain if ready.

    Returns:
        dict with keys: status ("idle"|"retrained"|"waiting"), details.
    """
    knowledge_path = Path(knowledge_dir)
    state_path = knowledge_path / "hil_state.json"
    labels_dir = knowledge_path / "incoming_labels"

    if not state_path.exists():
        return {"status": "idle"}

    with open(state_path) as f:
        state = json.load(f)

    if state.get("status") != "waiting":
        return {"status": "idle"}

    if not labels_dir.exists() or not any(labels_dir.glob("*.txt")):
        print("[HIL] Still waiting for labels.")
        return {"status": "waiting", "queued_since": state.get("queued_since")}

    # Labels arrived — run oracle-like retrain on labeled images
    label_files = list(labels_dir.glob("*.txt"))
    queued_images = state.get("images", [])

    # Match queued images to arrived labels
    labeled_stems = {lf.stem for lf in label_files}
    matched_images = [img for img in queued_images if Path(img).stem in labeled_stems]

    if not matched_images:
        print("[HIL] Label files present but no matching queued images.")
        return {"status": "waiting"}

    print(f"[HIL] {len(matched_images)} images matched with labels. Triggering oracle retrain.")

    if model_name is None:
        model_file = knowledge_path / "model.csv"
        model_name = model_file.read_text().strip().split("_v")[0] if model_file.exists() else "yolo_n"

    # Import retrain_oracle dynamically to avoid circular import
    from . import retrain_oracle
    result = retrain_oracle.run(
        model_name=model_name,
        knowledge_dir=str(knowledge_path),
        active_models_dir=active_models_dir,
        vmr_base_dir=vmr_base_dir,
        offline_labels_available=True,
    )

    if result.get("status") == "ok":
        # Clear queue
        for lf in label_files:
            lf.unlink(missing_ok=True)
        # Reset state
        state["status"] = "idle"
        tmp = str(state_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, state_path)

        _log_event(knowledge_path, "hil_retrain_complete", {
            "labeled_images": len(matched_images),
            "version_path": result.get("version_path"),
        })
        return {"status": "retrained", "details": result}

    return {"status": "waiting", "retrain_result": result}


def _lowest_energy_model(ema_energy: dict[str, float], available_models: list[str]) -> str:
    """Return the model with the lowest normalized energy EMA."""
    candidates = [m for m in available_models if m in ema_energy]
    if not candidates:
        return available_models[0] if available_models else ""
    return min(candidates, key=lambda m: ema_energy[m])


def _log_event(knowledge_path: Path, event_type: str, details: dict) -> None:
    mape_info_file = knowledge_path / "mape_info.json"
    if not mape_info_file.exists():
        return
    try:
        with open(mape_info_file) as f:
            info = json.load(f)
        events = info.setdefault("hil_events", [])
        events.append({
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "type": event_type,
            **details,
        })
        tmp = str(mape_info_file) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(info, f, indent=4)
        os.replace(tmp, mape_info_file)
    except Exception as exc:
        print(f"[HIL] Failed to log event: {exc}")

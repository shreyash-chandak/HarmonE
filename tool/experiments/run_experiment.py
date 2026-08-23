"""
experiments/run_experiment.py — Headless single-run experiment driver.

Runs one (dataset × planner × seed) combination entirely in-process (no Flask/ACP).
Reads model weights from the paths in the dataset config; scales inputs with a
MinMaxScaler fitted on the training split; runs the MAPE loop inline.

Usage (CLI):
    python experiments/run_experiment.py \\
        --dataset pems_node1 --planner harmone_original --seed 42 \\
        --run-dir runs/my_run

Or from Python:
    from experiments.run_experiment import run_experiment
    manifest = run_experiment("pems_node1", "harmone_original", seed=42,
                              run_dir="runs/my_run")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────

_TOOL_DIR = Path(__file__).resolve().parent.parent  # tool/
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

# ── Local imports ─────────────────────────────────────────────────────────────

from core.planners.base import PlanningContext, PlanDecision, get_planner
from core.scoring import (
    compute_harmone_score,
    normalize_energy,
    update_ema,
    update_energy_threshold,
    update_separated_emas,
)
from core.energy import EnergyMeter
from core.drift.kl_fixed_ref import KLFixedRefDetector
from core.vmr import VMR

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_MONITOR_INTERVAL = 50
_DRIFT_WINDOW_SIZE = 1200
_DRIFT_N_BINS = 50


# ── Setup helpers ─────────────────────────────────────────────────────────────

def setup_run_dir(run_dir: str) -> Path:
    """Create run directory and return as Path.  Idempotent."""
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_dataset_config(dataset_name: str, configs_dir: Path) -> dict:
    config_path = configs_dir / "datasets" / f"{dataset_name}.json"
    with open(config_path, "r") as f:
        return json.load(f)


def _build_adapter(dataset_config: dict, configs_dir: Path):
    """Instantiate the adapter class named in the config."""
    adapter_ref = dataset_config["adapter"]
    module_path, class_name = adapter_ref.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    domain = dataset_config.get("domain", "regression")
    if domain == "cv":
        try:
            return cls(dataset_config, tool_dir=str(_TOOL_DIR))
        except TypeError as exc:
            if "tool_dir" not in str(exc):
                raise
            return cls(dataset_config)
    else:
        try:
            return cls(dataset_config, config_dir=str(_TOOL_DIR))
        except TypeError as exc:
            if "config_dir" not in str(exc):
                raise
            return cls(dataset_config)


def _fit_scaler(train_values: np.ndarray):
    """Fit a MinMaxScaler on the training split."""
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_values.reshape(-1, 1))
    return scaler


def _build_reference_distribution(
    train_values: np.ndarray, ref_path: str, n_bins: int = _DRIFT_N_BINS
) -> None:
    """Write reference_distribution.json for the KL drift detector."""
    hist, bin_edges = np.histogram(train_values, bins=n_bins)
    with open(ref_path, "w") as f:
        json.dump(
            {"histogram": hist.tolist(), "bin_edges": bin_edges.tolist()},
            f,
            indent=2,
        )


def _seed_vmr_initial(
    model_names: list[str],
    dataset_config: dict,
    vmr: VMR,
    training_distribution: dict,
) -> None:
    """Seed the VMR with the initial trained model weights and training distribution.

    Mirrors managed_system_regression/train.py which seeds versionedMR/ after
    initial training. Skips models that already have an 'initial' version so
    repeated runs don't accumulate duplicate entries.
    """
    for model_name in model_names:
        existing = vmr.list_versions(model_name)
        if any(v.tag == "initial" for v in existing):
            logger.debug("VMR: '%s' already has an initial version — skipping seed.", model_name)
            continue

        spec = dataset_config.get("models", {}).get(model_name, {})
        wp = spec.get("weights_path", "")
        if not os.path.isabs(wp):
            wp = os.path.join(str(_TOOL_DIR), wp)
        if not os.path.exists(wp):
            logger.warning("VMR seed: weights not found for '%s' at %s — skipping.", model_name, wp)
            continue

        try:
            vmr.store(
                model_name, wp, training_distribution,
                tag="initial",
                proxy_score=None,
                drift_score=0.0,
            )
            logger.info("VMR seeded: '%s' (initial, training distribution)", model_name)
        except Exception as exc:
            logger.warning("VMR seed failed for '%s': %s", model_name, exc)


def _train_regression_models(
    dataset_config: dict, train_values: np.ndarray, scaler
) -> None:
    """Train LSTM + Ridge + SVR on train_values and save to the paths in config.

    Only called when weight files are absent.  Models are saved to the paths
    specified in dataset_config["models"][name]["weights_path"].
    """
    import pickle
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from adapters.loaders import LSTMModel

    seq_length = int(dataset_config.get("seq_length", 5))
    scaled = scaler.transform(train_values.reshape(-1, 1)).flatten()

    # Build (X, y) sequences
    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i : i + seq_length])
        y.append(scaled[i + seq_length])
    X = np.array(X)
    y = np.array(y)

    model_specs = dataset_config.get("models", {})
    for name, spec in model_specs.items():
        wp = spec["weights_path"]
        if not os.path.isabs(wp):
            wp = os.path.join(str(_TOOL_DIR), wp)
        os.makedirs(os.path.dirname(wp), exist_ok=True)

        if "lstm" in name:
            n_epochs = 50
            logger.info("Training LSTM for '%s' — %d epochs, %d sequences ...",
                        name, n_epochs, len(X))
            X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
            y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
            model = LSTMModel()
            opt = optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=16, shuffle=True)
            for epoch in range(n_epochs):
                epoch_loss = 0.0
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item()
                if (epoch + 1) % 10 == 0:
                    logger.info("  LSTM epoch %d/%d  loss=%.6f", epoch + 1, n_epochs,
                                epoch_loss / len(loader))
            torch.save(model.state_dict(), wp)
            logger.info("  Saved LSTM → %s", wp)

        elif "ridge" in name or "linear" in name:
            logger.info("Training Ridge for '%s' ...", name)
            m = Ridge(alpha=256)
            m.fit(X, y)
            with open(wp, "wb") as f:
                pickle.dump(m, f)
            logger.info("  Saved Ridge → %s", wp)

        elif "svr" in name or "svm" in name:
            logger.info("Training SVR for '%s' ...", name)
            m = SVR(kernel="linear", C=0.08, tol=0.16)
            m.fit(X, y)
            with open(wp, "wb") as f:
                pickle.dump(m, f)
            logger.info("  Saved SVR → %s", wp)

        else:
            logger.warning("No inline trainer for model '%s' — skipping.", name)


# ── CV helpers ────────────────────────────────────────────────────────────────

def _compute_image_luminance(image_path: str) -> float:
    """Return mean grayscale luminance (0–255) for an image. Returns 128.0 on failure."""
    try:
        from PIL import Image
        img = Image.open(image_path).convert("L")
        return float(np.array(img).mean())
    except Exception:
        return 128.0


def _build_cv_reference(
    train_paths: list[str], ref_path: str, n_bins: int = _DRIFT_N_BINS
) -> None:
    """Write luminance histogram reference for CV (luminance_kl) drift detection."""
    sample_every = max(1, len(train_paths) // 1000)
    luminances = [
        _compute_image_luminance(p)
        for i, p in enumerate(train_paths)
        if i % sample_every == 0
    ]
    if not luminances:
        luminances = [128.0]
    hist, bin_edges = np.histogram(luminances, bins=n_bins, range=(0, 256))
    with open(ref_path, "w") as f:
        json.dump({"histogram": hist.tolist(), "bin_edges": bin_edges.tolist()}, f, indent=2)


def _extract_cv_proxy(raw_pred: Any) -> float:
    """Extract a [0,1] accuracy proxy from model output.

    - dict {"proxy": float, "pred": ...}: from segformer_loader / torchvision_loader.
    - float/int: legacy scalar proxy — returned directly.
    - YOLO Results list: mean detection confidence across all boxes.
    - Returns 0.0 on failure or when no detections are found.
    """
    try:
        if raw_pred is None:
            return 0.0
        # Dict from segformer/torchvision loaders: {"proxy": float, "pred": data}
        if isinstance(raw_pred, dict):
            return float(np.clip(raw_pred.get("proxy", 0.0), 0.0, 1.0))
        # Scalar proxy returned directly by legacy loaders
        if isinstance(raw_pred, (int, float)):
            return float(np.clip(raw_pred, 0.0, 1.0))
        # YOLO-style Results list: extract box confidences
        results = list(raw_pred) if hasattr(raw_pred, "__iter__") else [raw_pred]
        confs: list[float] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            c = getattr(boxes, "conf", None)
            if c is None:
                continue
            try:
                confs.extend(c.tolist())
            except AttributeError:
                confs.extend(list(c))
        return float(sum(confs) / len(confs)) if confs else 0.0
    except Exception:
        return 0.0


def _save_cv_prediction(raw_pred: Any, step: int, pred_dir: Path, task: str) -> None:
    """Persist raw model output to pred_dir so offline eval needs no re-inference.

    segmentation → step_NNNNNN.png  (uint8 class mask, PNG lossless compression)
    classification → step_NNNNNN.txt (predicted class index as ASCII integer)
    detection → step_NNNNNN.npz     (boxes, scores, classes as compressed arrays)
    """
    try:
        if task == "segmentation":
            if isinstance(raw_pred, dict) and "pred" in raw_pred:
                from PIL import Image as _PILImage
                mask = raw_pred["pred"]  # uint8 ndarray (H, W)
                _PILImage.fromarray(mask, mode="L").save(
                    pred_dir / f"step_{step:06d}.png"
                )
        elif task == "classification":
            if isinstance(raw_pred, dict) and "pred" in raw_pred:
                with open(pred_dir / f"step_{step:06d}.txt", "w") as _f:
                    _f.write(str(raw_pred["pred"]))
        elif task == "detection":
            results = list(raw_pred) if hasattr(raw_pred, "__iter__") else []
            if results and getattr(results[0], "boxes", None) is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
                scores = results[0].boxes.conf.cpu().numpy().astype(np.float32)
                classes = results[0].boxes.cls.cpu().numpy().astype(np.int32)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros(0, dtype=np.float32)
                classes = np.zeros(0, dtype=np.int32)
            np.savez_compressed(
                pred_dir / f"step_{step:06d}.npz",
                boxes=boxes, scores=scores, classes=classes,
            )
    except Exception as exc:
        logger.debug("save_cv_prediction step %d: %s", step, exc)


def _load_models(dataset_config: dict, configs_dir: Path) -> dict[str, Any]:
    """Load all models listed in the config.  Returns {name: predict_fn | None}.

    A None entry means the weights file was not found — that model will be
    treated as unavailable and skipped in planning.
    """
    from adapters.loaders import get_loader

    models: dict[str, Any] = {}
    for name, spec in dataset_config.get("models", {}).items():
        weights_path = spec["weights_path"]
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(str(_TOOL_DIR), weights_path)
        if not os.path.exists(weights_path):
            logger.warning("Model '%s' weights not found at %s — skipping.", name, weights_path)
            models[name] = None
            continue
        try:
            loader_fn = get_loader(spec["loader"])
            models[name] = loader_fn(
                weights_path,
                seq_length=dataset_config.get("seq_length", 5),
                num_classes=dataset_config.get("num_classes", 1000),
            )
        except Exception as exc:
            logger.warning("Failed to load model '%s': %s", name, exc)
            models[name] = None

    return models


def _initial_mape_info(models: dict[str, Any]) -> dict:
    """Build a fresh mape_info dict for a new run."""
    return {
        "current_energy_threshold": 0.6,
        "ema_scores": {m: 0.5 for m in models},
        "ema_accuracy": {m: 0.5 for m in models},
        "ema_energy": {m: 0.5 for m in models},
        "event_counters": {
            "model_switches": 0,
            "retrains": 0,
            "retrain_skipped": 0,
            "vmr_events": 0,
            "noops": 0,
            "mape_k_energy_uJ": 0.0,
        },
    }


# ── MAPE logic (inline) ───────────────────────────────────────────────────────

def _monitor_batch(
    y_true: list[float],
    y_pred: list[float],
    energies_uJ: list[float],
    current_model: str,
    mape_info: dict,
    thresholds: dict,
    accuracy: float | None = None,
) -> dict:
    """Inline monitor: compute accuracy, energy, EMA; return telemetry dict.

    For regression, accuracy is computed as R²(y_true, y_pred).
    For CV, pass accuracy=<mean_confidence_proxy> directly (y_true/y_pred unused).
    """
    if accuracy is not None:
        r2 = max(0.0, min(1.0, accuracy))
    else:
        from sklearn.metrics import r2_score as _r2
        r2 = float(_r2(y_true, y_pred)) if len(y_true) > 1 else 0.0

    avg_energy_uJ = float(np.mean(energies_uJ)) if energies_uJ else 0.0
    e_min = thresholds.get("E_m", 0.0)
    e_max = thresholds.get("E_M", 25000.0)
    e_norm = normalize_energy(avg_energy_uJ, e_min, e_max)

    beta = thresholds.get("beta", 0.95)
    gamma = thresholds.get("gamma", 0.8)

    raw_score = compute_harmone_score(r2, e_norm, beta)
    prev_score = mape_info["ema_scores"].get(current_model, 0.5)
    ema = update_ema(prev_score, raw_score, gamma)

    mape_info["ema_scores"][current_model] = ema
    update_separated_emas(mape_info, current_model, r2, e_norm, gamma)

    return {
        "r2": round(r2, 6),
        "avg_energy_uJ": round(avg_energy_uJ, 4),
        "normalized_energy": round(e_norm, 6),
        "ema_score": round(ema, 6),
    }


def _analyse_violation(
    telemetry: dict,
    mape_info: dict,
    thresholds: dict,
) -> str | None:
    """Return 'score' | 'energy' | None."""
    min_score = thresholds.get("min_score", 0.78)
    energy_threshold = mape_info["current_energy_threshold"]

    if telemetry["ema_score"] < min_score:
        return "score"
    if telemetry["normalized_energy"] > energy_threshold:
        return "energy"
    return None


def _update_energy_boundary(
    mape_info: dict, telemetry: dict, thresholds: dict
) -> None:
    """Apply Eq. 3 (B1 fix) to update the adaptive energy threshold."""
    mape_info["current_energy_threshold"] = update_energy_threshold(
        current=mape_info["current_energy_threshold"],
        e_ref=thresholds.get("E_ref", 0.7),
        e_used=telemetry["normalized_energy"],
        delta=thresholds.get("delta", 0.1),
    )


def _analyse_drift(
    value_history: list[float],
    drift_detector: KLFixedRefDetector,
    thresholds: dict,
    vmr: VMR | None = None,
    current_model: str | None = None,
    current_distribution: dict | None = None,
) -> dict:
    """Run drift detection; return B2-aligned dict.

    When drift is detected, queries the VMR for a suitable archived version.
    Returns action="replace" + version path if found, else action="retrain".
    """
    detection = drift_detector.detect(value_history)
    kl = detection.get("kl_div")
    if not detection["drift_detected"]:
        return {"drift_detected": False, "action": None, "version": None, "kl_div": kl}

    if vmr is not None and current_model is not None:
        strategy = "closest_distribution" if current_distribution else "best_score"
        best = vmr.best_match(
            current_model,
            current_distribution=current_distribution,
            strategy=strategy,
        )
        if best is not None:
            return {
                "drift_detected": True,
                "action": "replace",
                "version": best.weights_path,
                "kl_div": kl,
            }

    return {"drift_detected": True, "action": "retrain", "version": None, "kl_div": kl}


def _plan(
    violation: str | None,
    drift_result: dict,
    current_model: str,
    available_models: list[str],
    mape_info: dict,
    thresholds: dict,
    planner,
    current_step: int = 0,
) -> PlanDecision:
    effective_violation = violation
    if drift_result["drift_detected"] and violation is None:
        effective_violation = "drift"

    ctx = PlanningContext(
        violation=effective_violation,
        ema_scores=dict(mape_info["ema_scores"]),
        ema_accuracy=dict(mape_info["ema_accuracy"]),
        ema_energy=dict(mape_info["ema_energy"]),
        current_model=current_model,
        available_models=available_models,
        thresholds=thresholds,
        drift_result=drift_result if drift_result["drift_detected"] else None,
        current_step=current_step,
    )
    return planner.plan(ctx)


def _load_model_store(dataset_config: dict) -> dict:
    """Load raw model objects for inline retraining (parallel to the predict closures).

    Returns {name: {"type": "sklearn"|"lstm", "model": obj} | None}.
    The model objects here are SEPARATE instances from those captured by the
    predict closures in _load_models(); _do_inline_retrain() rebuilds the
    closure after fitting so both stay in sync.
    """
    import pickle
    store: dict = {}
    for name, spec in dataset_config.get("models", {}).items():
        wp = spec["weights_path"]
        if not os.path.isabs(wp):
            wp = os.path.join(str(_TOOL_DIR), wp)
        if not os.path.exists(wp):
            store[name] = None
            continue
        try:
            if "lstm" in name:
                from adapters.loaders import LSTMModel
                import torch
                m = LSTMModel()
                m.load_state_dict(torch.load(wp, map_location="cpu", weights_only=False))
                m.eval()
                store[name] = {"type": "lstm", "model": m}
            else:
                with open(wp, "rb") as f:
                    m = pickle.load(f)
                store[name] = {"type": "sklearn", "model": m}
        except Exception as exc:
            logger.warning("Could not load model object for '%s' (retraining disabled): %s", name, exc)
            store[name] = None
    return store


def _do_inline_retrain(
    model_name: str,
    model_store: dict,
    models: dict,
    value_history: list[float],
    scaler,
    seq_length: int,
    drift_window: int,
) -> bool:
    """Retrain model_name on the most recent drift_window y_true values.

    Updates both model_store[model_name]["model"] (in-place fit) and
    models[model_name] (new predict closure) so subsequent inference sees the
    updated weights.  Returns True on success, False if skipped or failed.
    """
    info = model_store.get(model_name)
    if info is None:
        return False

    recent_raw = value_history[-drift_window:] if len(value_history) > drift_window else value_history
    if len(recent_raw) <= seq_length:
        return False

    try:
        arr = np.array(recent_raw)
        scaled = scaler.transform(arr.reshape(-1, 1)).flatten()
        X = np.array([scaled[i : i + seq_length] for i in range(len(scaled) - seq_length)])
        y = np.array([scaled[i + seq_length] for i in range(len(scaled) - seq_length)])
        if len(X) == 0:
            return False

        if info["type"] == "sklearn":
            info["model"].fit(X, y)
            _m = info["model"]
            def _new_predict(inputs: np.ndarray, m=_m) -> float:
                return float(m.predict(inputs.reshape(1, -1))[0])
            models[model_name] = _new_predict
            return True

        elif info["type"] == "lstm":
            import torch
            import torch.nn as nn
            import torch.optim as optim
            model_obj = info["model"]
            model_obj.train()
            opt = optim.Adam(model_obj.parameters(), lr=5e-4)
            loss_fn = nn.MSELoss()
            X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
            y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
            for _ in range(5):  # brief fine-tuning, not full re-training
                opt.zero_grad()
                loss_fn(model_obj(X_t), y_t).backward()
                opt.step()
            model_obj.eval()
            _m = model_obj
            def _new_predict(inputs: np.ndarray, m=_m) -> float:
                import torch as _t
                x = _t.tensor(inputs, dtype=_t.float32).view(1, -1, 1)
                with _t.no_grad():
                    return float(m(x).item())
            models[model_name] = _new_predict
            return True

    except Exception as exc:
        logger.warning("Inline retrain failed for '%s': %s", model_name, exc)

    return False


def _do_vmr_restore(
    version_path: str,
    model_name: str,
    models: dict,
    model_store: dict,
) -> bool:
    """Load model weights from a VMR version path; update models and model_store.

    Updates both the predict closure in models[model_name] and the raw object
    in model_store[model_name] so subsequent inline retrains see the restored weights.
    Returns True on success, False on failure.
    """
    import pickle
    try:
        if version_path.endswith((".pth", ".pt")):
            from adapters.loaders import LSTMModel
            import torch
            m = LSTMModel()
            m.load_state_dict(torch.load(version_path, map_location="cpu", weights_only=False))
            m.eval()
            model_store[model_name] = {"type": "lstm", "model": m}
            _m = m
            def _new_predict(inputs: np.ndarray, m=_m) -> float:
                import torch as _t
                x = _t.tensor(inputs, dtype=_t.float32).view(1, -1, 1)
                with _t.no_grad():
                    return float(m(x).item())
        else:
            with open(version_path, "rb") as f:
                m = pickle.load(f)
            model_store[model_name] = {"type": "sklearn", "model": m}
            _m = m
            def _new_predict(inputs: np.ndarray, m=_m) -> float:
                return float(m.predict(inputs.reshape(1, -1))[0])
        models[model_name] = _new_predict
        return True
    except Exception as exc:
        logger.warning("VMR restore failed for '%s' from %s: %s", model_name, version_path, exc)
        return False


def _archive_in_vmr(
    model_name: str,
    model_store: dict,
    vmr: VMR,
    value_history: list[float],
    drift_result: dict,
    run_path: Path,
    drift_window: int = _DRIFT_WINDOW_SIZE,
    n_bins: int = _DRIFT_N_BINS,
    tag: str = "retrain",
    proxy_score: float | None = None,
) -> None:
    """Atomically save model weights to the VMR.

    Writes weights to a temp file, hands it to vmr.store() (which copies it),
    then removes the temp file.  Fire-and-forget: logs a warning on failure.

    tag="pre_retrain"  — snapshot of the model BEFORE it is overwritten.
    tag="retrain"      — snapshot of the model AFTER retraining completes.
    tag="initial"      — initial seed written at run start.
    """
    import pickle
    info = model_store.get(model_name)
    if info is None:
        return

    try:
        if info["type"] == "lstm":
            import torch
            tmp_path = str(run_path / f"_vmr_tmp_{model_name}.pth")
            torch.save(info["model"].state_dict(), tmp_path)
        else:
            tmp_path = str(run_path / f"_vmr_tmp_{model_name}.pkl")
            with open(tmp_path, "wb") as f:
                pickle.dump(info["model"], f)
    except Exception as exc:
        logger.warning("VMR archive: could not save weights for '%s': %s", model_name, exc)
        return

    try:
        window = value_history[-drift_window:]
        if len(window) >= 10:
            hist, _ = np.histogram(window, bins=n_bins)
            distribution = {"type": "histogram", "data": hist.tolist()}
        else:
            distribution = {"type": "histogram", "data": []}
        vmr.store(
            model_name, tmp_path, distribution,
            tag=tag,
            proxy_score=proxy_score,
            drift_score=drift_result.get("kl_div"),
        )
        logger.debug("VMR archive: stored '%s' tag=%s (kl=%.4f)", model_name, tag,
                     drift_result.get("kl_div") or 0.0)
    except Exception as exc:
        logger.warning("VMR archive failed for '%s': %s", model_name, exc)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _execute(
    decision: PlanDecision,
    current_model: str,
    mape_info: dict,
    energy_uJ: float,
) -> str:
    """Apply decision; update mape_info counters.  Returns new current_model."""
    counters = mape_info["event_counters"]
    counters["mape_k_energy_uJ"] += energy_uJ

    if decision.action == "switch":
        new_model = decision.model
        if new_model and new_model != current_model:
            counters["model_switches"] += 1
            logger.debug("Switch: %s → %s  (%s)", current_model, new_model, decision.reason)
            return new_model

    elif decision.action in ("retrain", "replace"):
        # Retrain is handled upstream (regression-only inline path);
        # reaching here means the inline path was skipped or CV domain.
        counters["retrain_skipped"] += 1
        logger.debug("Retrain/replace skipped: %s", decision.reason)

    else:
        counters["noops"] += 1

    return current_model


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(
    dataset_name: str,
    planner_name: str,
    seed: int,
    run_dir: str,
    *,
    configs_dir: str | None = None,
    monitor_interval: int = _DEFAULT_MONITOR_INTERVAL,
    stream_delay_s: float = 0.0,
    max_steps: int | None = None,
    extra_thresholds: dict | None = None,
    pin_model: str | None = None,
) -> dict:
    """Run one (dataset × planner × seed) experiment and write artifacts.

    Args:
        dataset_name:      Key in configs/datasets/<name>.json.
        planner_name:      Registered planner name (e.g. "harmone_original").
        seed:              RNG seed for reproducibility.
        run_dir:           Directory where artifacts will be written.
        configs_dir:       Path to the configs/ directory; defaults to
                           <tool_dir>/configs.
        monitor_interval:  Predictions per MAPE cycle (default 50).
        stream_delay_s:    Seconds to sleep between predictions (0 for headless).
        max_steps:         Cap the stream at N steps (None = run to exhaustion).
        extra_thresholds:  Dict merged on top of dataset config thresholds
                           (for grid sweeps that override individual keys).

    Returns:
        The manifest dict (same content written to run_dir/run_manifest.json).
    """
    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)

    # ── Paths ─────────────────────────────────────────────────────────────────
    if configs_dir is None:
        configs_dir = str(_TOOL_DIR / "configs")
    configs_path = Path(configs_dir)
    run_path = setup_run_dir(run_dir)

    # ── Config ────────────────────────────────────────────────────────────────
    dataset_config = _load_dataset_config(dataset_name, configs_path)
    thresholds = dict(dataset_config)  # dataset config IS the thresholds dict
    thresholds["planner"] = planner_name
    if extra_thresholds:
        thresholds.update(extra_thresholds)

    energy_backend = thresholds.get("energy_meter", "null")  # null = no hardware needed

    # ── Adapter ───────────────────────────────────────────────────────────────
    adapter = _build_adapter(dataset_config, configs_path)
    effective_delay = float(thresholds.get("stream_delay_s", stream_delay_s))
    if hasattr(adapter, "_stream_delay"):
        adapter._stream_delay = effective_delay

    domain = dataset_config.get("domain", "regression")
    is_cv = (domain == "cv")
    cv_task: str = dataset_config.get("task", "detection")  # segmentation/classification/detection

    import pickle

    # ── Scaler / reference distribution (domain-specific) ─────────────────────
    ref_dist_path = str(run_path / "reference_distribution.json")

    pred_dir: Path | None = None
    if is_cv:
        pred_dir = run_path / "predictions"
        pred_dir.mkdir(exist_ok=True)

    if is_cv:
        train_paths: list[str] = adapter.train_split()
        scaler = None
        scaler_path = None
        logger.info("CV domain: building luminance reference from %d training images ...",
                    len(train_paths))
        _build_cv_reference(train_paths, ref_dist_path)
        # Re-use KLFixedRefDetector on luminance values (range 0–255 → n_bins buckets)
        drift_detector = KLFixedRefDetector(
            reference_path=ref_dist_path,
            tau_drift=thresholds.get("tau_drift", 0.07),
            window_size=thresholds.get("drift_window_size", _DRIFT_WINDOW_SIZE),
            n_bins=_DRIFT_N_BINS,
        )
    else:
        train_values: np.ndarray = adapter.train_split()
        scaler = _fit_scaler(train_values)
        scaler_path = run_path / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        _build_reference_distribution(train_values, ref_dist_path)
        drift_detector = KLFixedRefDetector(
            reference_path=ref_dist_path,
            tau_drift=thresholds.get("tau_drift", 0.5),
            window_size=_DRIFT_WINDOW_SIZE,
            n_bins=_DRIFT_N_BINS,
        )

    # ── Models ────────────────────────────────────────────────────────────────
    if not is_cv:
        any_missing = any(
            not os.path.exists(
                spec["weights_path"] if os.path.isabs(spec["weights_path"])
                else os.path.join(str(_TOOL_DIR), spec["weights_path"])
            )
            for spec in dataset_config.get("models", {}).values()
        )
        if any_missing:
            logger.info("Some model weights missing — training inline on training split ...")
            _train_regression_models(dataset_config, train_values, scaler)

    models = _load_models(dataset_config, configs_path)
    available_models = [m for m, fn in models.items() if fn is not None]
    if not available_models:
        raise RuntimeError(
            f"No loadable models found for dataset '{dataset_name}'. "
            "Check that weights_path entries in the config point to existing files."
        )

    # Pin to a single model (used for per-model naive baseline runs)
    if pin_model is not None:
        if pin_model not in available_models:
            raise RuntimeError(
                f"Pinned model '{pin_model}' not in available models {available_models} "
                f"for dataset '{dataset_name}'."
            )
        available_models = [pin_model]

    # Load raw model objects for inline retraining (regression only)
    model_store: dict = _load_model_store(dataset_config) if not is_cv else {}
    seq_length: int = int(dataset_config.get("seq_length", 5))

    # ── VMR (regression only) ─────────────────────────────────────────────────
    # Scoped per dataset so that pems/lstm and uci_electricity/lstm are
    # independent namespaces and cannot cross-contaminate each other.
    _vmr_dir = _TOOL_DIR / "knowledge" / "vmr" / dataset_name
    _vmr_dir.mkdir(parents=True, exist_ok=True)
    vmr = VMR(base_dir=str(_vmr_dir))

    if not is_cv:
        # Seed VMR with initial model weights + training distribution so that
        # vmr.best_match() has candidates from the very first drift event.
        # Mirrors managed_system_regression/train.py's versionedMR/ seeding.
        _train_hist, _ = np.histogram(train_values, bins=_DRIFT_N_BINS)
        _initial_dist = {"type": "histogram", "data": _train_hist.tolist()}
        _seed_vmr_initial(available_models, dataset_config, vmr, _initial_dist)

    # ── Planner ───────────────────────────────────────────────────────────────
    if planner_name == "bandit":
        from core.planners.bandit import load_or_create_bandit, set_bandit_instance
        _knowledge_dir = _TOOL_DIR / "knowledge"
        _knowledge_dir.mkdir(parents=True, exist_ok=True)
        _bandit = load_or_create_bandit(thresholds, available_models, _knowledge_dir, dataset_name)
        set_bandit_instance(_bandit)
    planner = get_planner(planner_name)

    # ── State ─────────────────────────────────────────────────────────────────
    current_model = available_models[0]
    mape_info = _initial_mape_info(models)
    value_history: list[float] = []

    # Batch accumulators (reset every monitor_interval steps)
    batch_y_true: list[float] = []
    batch_y_pred: list[float] = []
    batch_energy_uJ: list[float] = []

    # Predictions log (written to CSV at end)
    prediction_rows: list[dict] = []
    mape_events: list[dict] = []

    # ── Run metadata ──────────────────────────────────────────────────────────
    start_ts = datetime.now(timezone.utc).isoformat()
    start_wall = time.monotonic()

    logger.info(
        "run_experiment | dataset=%s planner=%s seed=%d monitor_interval=%d",
        dataset_name, planner_name, seed, monitor_interval,
    )

    # ── Stream loop ───────────────────────────────────────────────────────────
    step = 0
    for sample in adapter.stream():
        if max_steps is not None and step >= max_steps:
            break

        # Resolve active predict_fn (same for both domains)
        predict_fn = models.get(current_model)
        if predict_fn is None:
            for m in available_models:
                if models[m] is not None:
                    current_model = m
                    predict_fn = models[m]
                    break

        if is_cv:
            # ── CV: image path → YOLO → confidence proxy ──────────────────
            image_path = str(sample.inputs)

            with EnergyMeter(f"step_{step}", backend=energy_backend) as _em:
                try:
                    raw_pred = predict_fn(image_path)
                except Exception as exc:
                    logger.warning("CV prediction failed at step %d: %s", step, exc)
                    raw_pred = None

            energy_uJ = _em.total_uJ if _em.total_uJ is not None else 0.0
            proxy_acc = _extract_cv_proxy(raw_pred)

            # Persist prediction for offline eval (no re-inference needed later)
            if pred_dir is not None and raw_pred is not None:
                _save_cv_prediction(raw_pred, step, pred_dir, cv_task)

            # Luminance drives the drift detector (luminance_kl strategy)
            luminance = _compute_image_luminance(image_path)
            value_history.append(luminance)

            batch_y_true.append(proxy_acc)
            batch_y_pred.append(proxy_acc)  # placeholder; accuracy= kwarg used in monitor
            batch_energy_uJ.append(energy_uJ)

            prediction_rows.append({
                "step": step,
                "proxy_acc": round(proxy_acc, 6),
                "active_model": current_model,
                "energy_uJ": round(energy_uJ, 4),
                "energy_valid": _em.valid,
            })
            logger.info(
                "stream  step=%-5d  model=%-12s  conf=%.4f  energy=%.1f µJ",
                step, current_model, proxy_acc, energy_uJ,
            )

        else:
            # ── Regression: scalar inputs → scale → predict → inverse scale ─
            scaled_inputs = scaler.transform(sample.inputs.reshape(-1, 1)).flatten()

            with EnergyMeter(f"step_{step}", backend=energy_backend) as _em:
                try:
                    raw_pred = predict_fn(scaled_inputs)
                except Exception as exc:
                    logger.warning("Prediction failed at step %d: %s", step, exc)
                    raw_pred = 0.0

            energy_uJ = _em.total_uJ if _em.total_uJ is not None else 0.0
            y_pred_orig = float(scaler.inverse_transform([[raw_pred]])[0][0])
            y_true = float(sample.ground_truth)

            value_history.append(y_true)
            batch_y_true.append(y_true)
            batch_y_pred.append(y_pred_orig)
            batch_energy_uJ.append(energy_uJ)

            prediction_rows.append({
                "step": step,
                "y_true": round(y_true, 6),
                "y_pred": round(y_pred_orig, 6),
                "active_model": current_model,
                "energy_uJ": round(energy_uJ, 4),
                "energy_valid": _em.valid,
            })

        if stream_delay_s > 0:
            time.sleep(stream_delay_s)

        # ── MAPE cycle ────────────────────────────────────────────────────────
        step += 1
        if step % monitor_interval == 0 and batch_y_true:
            # Monitor
            if is_cv:
                batch_proxy = float(np.mean(batch_y_true))
                telemetry = _monitor_batch(
                    batch_y_true, batch_y_pred, batch_energy_uJ,
                    current_model, mape_info, thresholds,
                    accuracy=batch_proxy,
                )
            else:
                telemetry = _monitor_batch(
                    batch_y_true, batch_y_pred, batch_energy_uJ,
                    current_model, mape_info, thresholds,
                )

            # Analyse: violation?
            violation = _analyse_violation(telemetry, mape_info, thresholds)

            # Analyse: drift? Build distribution snapshot for VMR matching first.
            current_dist: dict | None = None
            if not is_cv and drift_detector._bin_edges is not None:
                _window = value_history[-_DRIFT_WINDOW_SIZE:]
                if len(_window) >= _DRIFT_WINDOW_SIZE:
                    _hist, _ = np.histogram(_window, bins=drift_detector._bin_edges)
                    current_dist = {"type": "histogram", "data": _hist.tolist()}
            drift_result = _analyse_drift(
                value_history, drift_detector, thresholds,
                vmr=vmr if not is_cv else None,
                current_model=current_model,
                current_distribution=current_dist,
            )

            # Update energy boundary (B1 fix)
            _update_energy_boundary(mape_info, telemetry, thresholds)

            # Plan
            with EnergyMeter("mape_plan", backend=energy_backend) as _plan_em:
                decision = _plan(
                    violation, drift_result, current_model,
                    available_models, mape_info, thresholds, planner,
                    current_step=step,
                )

            # EMA head-start: if the planner noops while some models have never
            # been observed (EMA still exactly at the 0.5 initialisation value),
            # force a round-robin trial so every model accumulates at least one
            # monitoring window before the planner makes exploitative comparisons.
            # Only applies on CV runs where ema_head_start > 0 in the config.
            if decision.action == "noop" and is_cv:
                _head_start = thresholds.get("ema_head_start", 0.0)
                if _head_start > 0.0:
                    _unobserved = [
                        m for m in available_models
                        if m != current_model
                        and abs(mape_info["ema_scores"].get(m, 0.5) - 0.5) < 1e-9
                    ]
                    if _unobserved:
                        _trial = _unobserved[0]
                        decision = PlanDecision(
                            action="switch",
                            model=_trial,
                            reason=f"ema_head_start: bootstrap {_trial} (never observed)",
                        )
                        logger.info("ema_head_start: forcing trial of %s", _trial)

            # Execute — VMR restore or inline retrain for drift; switch for score/energy
            plan_energy_uJ = _plan_em.total_uJ or 0.0
            if decision.action in ("retrain", "replace") and not is_cv:
                mape_info["event_counters"]["mape_k_energy_uJ"] += plan_energy_uJ
                vmr_done = False

                if decision.action == "replace" and decision.version_path:
                    vmr_done = _do_vmr_restore(
                        decision.version_path, current_model, models, model_store,
                    )
                    if vmr_done:
                        mape_info["event_counters"]["vmr_events"] += 1
                        logger.info(
                            "VMR restore: %s ← %s", current_model, decision.version_path,
                        )
                    else:
                        logger.warning(
                            "VMR restore failed for '%s'; falling back to retrain.", current_model,
                        )

                if not vmr_done:
                    # Snapshot current weights before overwriting — enables recovery
                    # if retrain degrades the model, and gives VMR a pre-retrain
                    # checkpoint for future closest_distribution matching.
                    _archive_in_vmr(
                        current_model, model_store, vmr,
                        value_history, drift_result, run_path,
                        tag="pre_retrain",
                        proxy_score=mape_info["ema_scores"].get(current_model),
                    )
                    retrained = _do_inline_retrain(
                        current_model, model_store, models, value_history,
                        scaler=scaler,
                        seq_length=seq_length,
                        drift_window=_DRIFT_WINDOW_SIZE,
                    )
                    if retrained:
                        mape_info["event_counters"]["retrains"] += 1
                        logger.debug("Inline retrain: %s  (%s)", current_model, decision.reason)
                        _archive_in_vmr(
                            current_model, model_store, vmr,
                            value_history, drift_result, run_path,
                            tag="retrain",
                            proxy_score=mape_info["ema_scores"].get(current_model),
                        )
                    else:
                        mape_info["event_counters"]["retrain_skipped"] += 1
                        logger.debug("Retrain skipped: %s", decision.reason)

                new_model = current_model
            else:
                new_model = _execute(
                    decision, current_model, mape_info,
                    energy_uJ=plan_energy_uJ,
                )
            old_model = current_model
            current_model = new_model

            mape_events.append({
                "step": step,
                "violation": violation,
                "drift_detected": drift_result["drift_detected"],
                "kl_div": drift_result.get("kl_div"),
                "decision_action": decision.action,
                "decision_model": decision.model,
                "decision_reason": decision.reason,
                "model_before": old_model,
                "model_after": current_model,
                "r2": telemetry["r2"],
                "ema_score": telemetry["ema_score"],
                "avg_energy_uJ": telemetry["avg_energy_uJ"],
                "energy_threshold": round(mape_info["current_energy_threshold"], 6),
            })

            # Per-cycle INFO line — shows accuracy + energy so log is human-readable
            _action_str = decision.action
            if decision.action == "switch" and decision.model and decision.model != old_model:
                _action_str = f"switch→{decision.model}"
            _flags = ""
            if violation:
                _flags += f"  violation={violation}"
            if drift_result["drift_detected"]:
                _flags += "  drift=yes"
            _metric_label = "conf" if is_cv else "R²  "
            logger.info(
                "MAPE[%d] model=%-8s  action=%-16s  %s=%.4f  energy=%.1f µJ/step  ema=%.4f%s",
                len(mape_events), old_model, _action_str, _metric_label,
                telemetry["r2"], telemetry["avg_energy_uJ"],
                telemetry["ema_score"], _flags,
            )

            # Reset batch
            batch_y_true = []
            batch_y_pred = []
            batch_energy_uJ = []

    # ── Task metrics (regression: R², RMSE, MAE, energy; CV: none yet) ─────────
    task_metrics: dict = {}
    energy_by_model: dict = {}
    if prediction_rows:
        for _r in prediction_rows:
            _m = _r.get("active_model", "unknown")
            if _m not in energy_by_model:
                energy_by_model[_m] = {"energy_uJ": 0.0, "n_steps": 0}
            energy_by_model[_m]["energy_uJ"] += _r["energy_uJ"]
            energy_by_model[_m]["n_steps"] += 1
        # Round for readability
        for _v in energy_by_model.values():
            _v["energy_uJ"] = round(_v["energy_uJ"], 2)
            _v["energy_mJ"] = round(_v["energy_uJ"] / 1000.0, 4)

    if not is_cv and prediction_rows:
        from sklearn.metrics import r2_score as _r2_fn
        _yt = [r["y_true"] for r in prediction_rows]
        _yp = [r["y_pred"] for r in prediction_rows]
        _n = len(_yt)
        _mae = sum(abs(a - b) for a, b in zip(_yt, _yp)) / _n
        _mse = sum((a - b) ** 2 for a, b in zip(_yt, _yp)) / _n
        _r2_global = float(_r2_fn(_yt, _yp)) if _n > 1 else 0.0
        _total_e_uJ = sum(r["energy_uJ"] for r in prediction_rows)
        task_metrics = {
            "r2": round(_r2_global, 6),
            "rmse": round(_mse ** 0.5, 6),
            "mae": round(_mae, 6),
            "n_samples": _n,
            "total_inference_energy_uJ": round(_total_e_uJ, 2),
            "total_inference_energy_mJ": round(_total_e_uJ / 1000.0, 4),
        }

    if is_cv and prediction_rows:
        _proxy_vals = [r["proxy_acc"] for r in prediction_rows]
        _n = len(_proxy_vals)
        _mean_conf = sum(_proxy_vals) / _n
        _std_conf = float(np.std(_proxy_vals))
        _total_e_uJ = sum(r["energy_uJ"] for r in prediction_rows)
        _model_confs: dict[str, list] = {}
        for _r in prediction_rows:
            _model_confs.setdefault(_r["active_model"], []).append(_r["proxy_acc"])
        task_metrics = {
            "mean_proxy_acc": round(_mean_conf, 6),
            "std_proxy_acc": round(_std_conf, 6),
            "min_proxy_acc": round(min(_proxy_vals), 6),
            "max_proxy_acc": round(max(_proxy_vals), 6),
            "n_samples": _n,
            "per_model": {
                m: {"mean_proxy_acc": round(sum(v) / len(v), 6), "n_steps": len(v)}
                for m, v in _model_confs.items()
            },
            "total_inference_energy_uJ": round(_total_e_uJ, 2),
            "total_inference_energy_mJ": round(_total_e_uJ / 1000.0, 4),
        }

    # ── Write artifacts ───────────────────────────────────────────────────────
    elapsed_s = time.monotonic() - start_wall
    end_ts = datetime.now(timezone.utc).isoformat()

    # predictions.csv
    import csv
    predictions_path = run_path / "predictions.csv"
    if prediction_rows:
        fieldnames = list(prediction_rows[0].keys())
        with open(predictions_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prediction_rows)

    # mape_events.csv
    events_path = run_path / "mape_events.csv"
    if mape_events:
        event_fields = list(mape_events[0].keys())
        with open(events_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=event_fields)
            writer.writeheader()
            writer.writerows(mape_events)

    # mape_info.json — final state
    mape_info_path = run_path / "mape_info.json"
    with open(mape_info_path, "w") as f:
        json.dump(mape_info, f, indent=2)

    # thresholds.json — copy of effective thresholds
    thresholds_path = run_path / "thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    # run_manifest.json — top-level metadata + summary
    manifest = {
        "run_id": run_path.name,
        "dataset": dataset_name,
        "planner": planner_name,
        "pin_model": pin_model,
        "seed": seed,
        "monitor_interval": monitor_interval,
        "started_at": start_ts,
        "finished_at": end_ts,
        "elapsed_s": round(elapsed_s, 3),
        "total_steps": step,
        "mape_cycles": len(mape_events),
        "final_model": current_model,
        "event_counters": dict(mape_info["event_counters"]),
        "final_ema_scores": dict(mape_info["ema_scores"]),
        "task_metrics": task_metrics,
        "energy_by_model": energy_by_model,
        "artifacts": {
            "predictions_csv": str(predictions_path),
            "mape_events_csv": str(events_path),
            "mape_info_json": str(mape_info_path),
            "thresholds_json": str(thresholds_path),
            "reference_distribution_json": ref_dist_path,
            "scaler_pkl": str(scaler_path) if scaler_path is not None else None,
            "cv_predictions_dir": str(pred_dir) if pred_dir is not None else None,
        },
        "status": "ok",
    }

    manifest_path = run_path / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "run_experiment done | steps=%d cycles=%d switches=%d retrains=%d elapsed=%.1fs",
        step, len(mape_events),
        mape_info["event_counters"]["model_switches"],
        mape_info["event_counters"]["retrains"],
        elapsed_s,
    )
    if task_metrics:
        if is_cv:
            logger.info(
                "accuracy summary  | mean_conf=%.4f  std=%.4f  min=%.4f  max=%.4f  n=%d",
                task_metrics["mean_proxy_acc"],
                task_metrics["std_proxy_acc"],
                task_metrics["min_proxy_acc"],
                task_metrics["max_proxy_acc"],
                task_metrics["n_samples"],
            )
            for _m, _ms in sorted(task_metrics.get("per_model", {}).items()):
                logger.info(
                    "conf by model     | %-12s  steps=%d  mean_conf=%.4f",
                    _m, _ms["n_steps"], _ms["mean_proxy_acc"],
                )
        else:
            logger.info(
                "accuracy summary  | R²=%.4f  RMSE=%.4f  MAE=%.4f  n=%d",
                task_metrics.get("r2", float("nan")),
                task_metrics.get("rmse", float("nan")),
                task_metrics.get("mae", float("nan")),
                task_metrics.get("n_samples", 0),
            )
        _inf_mJ = task_metrics.get("total_inference_energy_mJ", 0.0)
        _mape_mJ = mape_info["event_counters"]["mape_k_energy_uJ"] / 1000.0
        logger.info(
            "energy summary    | inference=%.3f mJ  mape_overhead=%.3f mJ  total=%.3f mJ",
            _inf_mJ, _mape_mJ, _inf_mJ + _mape_mJ,
        )
    if energy_by_model:
        for _m, _ev in sorted(energy_by_model.items()):
            logger.info(
                "energy by model   | %-10s  steps=%d  energy=%.3f mJ",
                _m, _ev["n_steps"], _ev["energy_mJ"],
            )

    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_run_id(dataset: str, planner: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{ts}_{dataset}_{planner}_s{seed}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a single HarmonE experiment.")
    parser.add_argument("--dataset", required=True, help="Dataset name (configs/datasets/<name>.json)")
    parser.add_argument("--planner", required=True, help="Planner name (e.g. harmone_original)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-dir", default=None, help="Output directory (auto-generated if omitted)")
    parser.add_argument("--runs-dir", default="runs", help="Parent dir for auto-generated run dirs")
    parser.add_argument("--monitor-interval", type=int, default=_DEFAULT_MONITOR_INTERVAL)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument("--pin-model", default=None,
                        help="Fix a single model for the entire run (used for per-model naive baseline).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_dir = args.run_dir
    if run_dir is None:
        run_id = _build_run_id(args.dataset, args.planner, args.seed)
        run_dir = str(Path(args.runs_dir) / run_id)

    manifest = run_experiment(
        dataset_name=args.dataset,
        planner_name=args.planner,
        seed=args.seed,
        run_dir=run_dir,
        configs_dir=args.configs_dir,
        monitor_interval=args.monitor_interval,
        max_steps=args.max_steps,
        pin_model=args.pin_model,
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

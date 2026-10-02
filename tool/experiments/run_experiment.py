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
from core.device import get_device as _dev, yolo_device, configure_device, device_info

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
            hp = spec.get("hyperparams", {}).get("train", {})
            n_epochs = int(hp.get("epochs", 50))
            lr = float(hp.get("lr", 0.001))
            batch_size = int(hp.get("batch_size", 16))
            logger.info("Training LSTM for '%s' — %d epochs, %d sequences ...",
                        name, n_epochs, len(X))
            X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
            y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
            model = LSTMModel()
            opt = optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
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
            hp = spec.get("hyperparams", {}).get("train", {})
            alpha = float(hp.get("alpha", 256))
            logger.info("Training Ridge for '%s' (alpha=%.0f) ...", name, alpha)
            m = Ridge(alpha=alpha)
            m.fit(X, y)
            with open(wp, "wb") as f:
                pickle.dump(m, f)
            logger.info("  Saved Ridge → %s", wp)

        elif "svr" in name or "svm" in name:
            hp = spec.get("hyperparams", {}).get("train", {})
            kernel = hp.get("kernel", "linear")
            C = float(hp.get("C", 0.08))
            tol = float(hp.get("tol", 0.16))
            epsilon = float(hp.get("epsilon", 0.1))  # sklearn default; datasets whose
            # stream extrapolates outside the scaled [0,1] training range (drift-induced
            # regression configs) need this much tighter — see DECISIONS_PENDING.md DP27.
            max_train_samples = int(hp.get("max_train_samples", 8000))
            # SVR's QP solver scales super-linearly with sample count — fitting on the
            # full sequence set (uci_electricity ~14k, spot_prices ~130k) took 4+ minutes
            # and 10+ minutes respectively during verification, impractical for something
            # that can also run inline during a drift-triggered retrain event. Subsampling
            # is empirically safe here (confirmed against the real drift-induced stream,
            # DP27): a linear-kernel SVR needs far fewer points than a complex RBF one to
            # find a good hyperplane, and R² on an 8000-point subsample matched the full
            # fit within noise for both datasets.
            X_fit, y_fit = X, y
            if len(X) > max_train_samples:
                idx = np.random.RandomState(0).choice(len(X), max_train_samples, replace=False)
                X_fit, y_fit = X[idx], y[idx]
            logger.info(
                "Training SVR for '%s' (kernel=%s, C=%.3f, tol=%.4f, epsilon=%.4f, "
                "n=%d%s) ...",
                name, kernel, C, tol, epsilon, len(X_fit),
                f" subsampled from {len(X)}" if len(X_fit) < len(X) else "",
            )
            m = SVR(kernel=kernel, C=C, tol=tol, epsilon=epsilon)
            m.fit(X_fit, y_fit)
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
        # Phase 1.4: step each model was last active (for staleness detection)
        "last_observed_step": {},
        # Phase 1: violation-aware noop counter and switch tracking
        "steps_since_last_switch": 0,
        "event_counters": {
            "model_switches": 0,
            "retrains": 0,
            "retrain_skipped": 0,
            "vmr_events": 0,
            "noops": 0,
            "noop_on_violation": 0,
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

    R² is mathematically unbounded below (a batch with low true-value variance can
    send it to large negative numbers even for a decent model — this is exactly
    what was observed with monitor_interval=50 batches in practice). Both branches
    clip to [0, 1] so the result honors the contract compute_harmone_score() and
    update_separated_emas() both document ("accuracy ... in [0, 1]") — previously
    only the CV branch did this, letting an unbounded R² poison a model's EMA score
    (and its ema_accuracy used by violation_aware/pareto's min_accuracy gate) after
    a single noisy window, effectively locking switching-based planners onto
    whichever model wasn't just hit by a bad batch.
    """
    if accuracy is not None:
        r2 = max(0.0, min(1.0, accuracy))
    else:
        from sklearn.metrics import r2_score as _r2
        raw_r2 = float(_r2(y_true, y_pred)) if len(y_true) > 1 else 0.0
        r2 = max(0.0, min(1.0, raw_r2))

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

    The VMR match is only accepted if its distribution distance clears
    thresholds["tau_drift"] — mirroring the original paper's own gate ("if
    some prior version's training data is a closer match [min KL < 0.75],
    that version... is returned... instead of retraining"). Reusing tau_drift
    here (rather than a separate config key) mirrors the original apparently
    using the same single constant for both "is this drift?" and "is this
    archived version close enough to reuse?". Previously there was no gate at
    all — any non-empty archive always "won" regardless of how distant the
    best match actually was, which meant a genuine retrain could never fire
    again once even one version existed for a model (confirmed happening in
    practice, 2026-09-03).
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
            threshold=thresholds.get("tau_drift") if strategy == "closest_distribution" else None,
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
        # Phase 1 plumbing: live adaptive threshold and staleness tracking
        current_energy_threshold=float(mape_info.get("current_energy_threshold", 0.6)),
        last_observed_step=dict(mape_info.get("last_observed_step", {})),
        staleness_window=int(thresholds.get("staleness_window", 500)),
    )
    decision = planner.plan(ctx)

    # Switch hold (audit A2/A3/D1, 2026-09-28) — mirrors the original HarmonE's
    # recovery_cycles (HarmonE/mape/analyse.py: after an energy violation, no
    # switching for the next 3 cycles), generalised to "after ANY switch".
    # Gives a newly activated model full monitoring windows before it can be
    # judged again, which damps thrashing. Retrain/replace/noop decisions are
    # never suppressed. Relies on mape_info["steps_since_last_switch"] being
    # incremented once per cycle BEFORE planning and reset to 0 on a switch
    # (both harnesses do this). Not applied before the first switch of a run.
    # switch_hold_cycles=0 (the default) disables it.
    hold = int(thresholds.get("switch_hold_cycles", 0))
    if (
        hold > 0
        and decision.action == "switch"
        and decision.model
        and decision.model != current_model
        and mape_info.get("event_counters", {}).get("model_switches", 0) > 0
        and mape_info.get("steps_since_last_switch", hold + 1) <= hold
    ):
        return PlanDecision(
            action="noop",
            reason=(
                f"switch hold: {mape_info.get('steps_since_last_switch')} cycle(s) since "
                f"last switch <= switch_hold_cycles={hold} (suppressed: {decision.reason})"
            ),
        )
    return decision


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
        retrain_params = spec.get("hyperparams", {}).get("retrain", {})
        try:
            if "lstm" in name:
                from adapters.loaders import LSTMModel
                import torch
                m = LSTMModel()
                m.load_state_dict(torch.load(wp, map_location="cpu", weights_only=False))
                m.eval()
                store[name] = {"type": "lstm", "model": m, "retrain_params": retrain_params}
            else:
                with open(wp, "rb") as f:
                    m = pickle.load(f)
                store[name] = {"type": "sklearn", "model": m, "retrain_params": retrain_params}
        except Exception as exc:
            logger.warning("Could not load model object for '%s' (retraining disabled): %s", name, exc)
            store[name] = None
    return store


def _build_torchvision_classifier(weights_path: str, num_classes: int):
    """Build an EfficientNet/ResNet classifier + its inference transform.

    Architecture inference and preprocessing mirror
    adapters/loaders.py::torchvision_loader() exactly, so pseudo-labels
    generated here are consistent with normal inference. Returns
    (model, transform, arch_name); raises ValueError on unrecognised filename.
    """
    import torch
    import torchvision.models as tv
    from torchvision import transforms

    name_lower = os.path.basename(weights_path).lower()
    if "efficientnet_b0" in name_lower:
        model = tv.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        arch = "efficientnet_b0"
    elif "resnet101" in name_lower:
        model = tv.resnet101(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        arch = "resnet101"
    elif "resnet50" in name_lower:
        model = tv.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        arch = "resnet50"
    else:
        raise ValueError(
            f"Cannot infer torchvision architecture from filename: {weights_path}. "
            "Expected 'efficientnet_b0', 'resnet50', or 'resnet101' in the name."
        )

    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(_dev())
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, transform, arch


def _build_segformer_model(weights_path: str, num_classes: int):
    """Build a SegFormer segmentation model + its HF image processor.

    Architecture inference and preprocessing mirror
    adapters/loaders.py::segformer_loader() exactly (reuses its hardcoded,
    network-free MiT backbone table), so pseudo-labels generated here are
    consistent with normal inference. Returns (model, processor, arch_key).
    """
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor
    from adapters.loaders import _SEGFORMER_ARCH

    name_lower = os.path.basename(weights_path).lower()
    if "segformer_b2" in name_lower or "segformer-b2" in name_lower:
        arch_key = "nvidia/mit-b2"
    elif "segformer_b1" in name_lower or "segformer-b1" in name_lower:
        arch_key = "nvidia/mit-b1"
    else:
        arch_key = "nvidia/mit-b0"

    cfg = SegformerConfig(**_SEGFORMER_ARCH[arch_key], num_labels=num_classes)
    model = SegformerForSemanticSegmentation(cfg)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(_dev())
    model.eval()

    processor = SegformerImageProcessor(
        do_resize=True, size={"height": 512, "width": 512},
        do_normalize=True, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
        do_rescale=True, rescale_factor=1.0 / 255,
    )
    return model, processor, arch_key


def _build_yolo_model(weights_path: str):
    """Load a YOLO model via ultralytics — a thin wrapper, kept as its own
    helper for symmetry with the other two _build_* functions."""
    from ultralytics import YOLO
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS.update({"sync": False})  # disable telemetry / update checks
    except Exception:
        pass
    model = YOLO(weights_path)
    model.to(_dev())
    return model


def _load_cv_model_store(dataset_config: dict, cv_task: str) -> dict:
    """Load raw CV model objects for inline periodic fine-tuning (PRT+VMR, CV).

    Covers all three CV task families:
      - "classification": torchvision EfficientNet/ResNet (_build_torchvision_classifier)
      - "segmentation":    SegFormer B0/B1/B2 (_build_segformer_model)
      - "detection":       YOLOv8 n/s/m via ultralytics (_build_yolo_model)

    An unrecognised task, or a model that fails to build, gets a None entry —
    _do_cv_inline_finetune then no-ops cleanly (counted as retrain_skipped)
    rather than silently pretending to fine-tune something that didn't load.

    Returns {name: {"type": ..., "model": ..., ...} | None} — the exact
    per-type dict shape is documented on each _build_* helper above and
    consumed by _do_cv_inline_finetune / _do_cv_vmr_restore / _archive_in_vmr.
    """
    store: dict = {}
    for name, spec in dataset_config.get("models", {}).items():
        wp = spec["weights_path"]
        if not os.path.isabs(wp):
            wp = os.path.join(str(_TOOL_DIR), wp)
        if not os.path.exists(wp):
            store[name] = None
            continue
        try:
            if cv_task == "classification":
                num_classes = int(dataset_config.get("num_classes", 1000))
                model, transform, arch = _build_torchvision_classifier(wp, num_classes)
                store[name] = {
                    "type": "torchvision_classifier",
                    "model": model, "transform": transform,
                    "arch": arch, "num_classes": num_classes,
                }
            elif cv_task == "segmentation":
                num_classes = int(dataset_config.get("num_classes", 19))
                model, processor, arch = _build_segformer_model(wp, num_classes)
                store[name] = {
                    "type": "segformer_segmentation",
                    "model": model, "processor": processor,
                    "arch": arch, "num_classes": num_classes,
                }
            elif cv_task == "detection":
                model = _build_yolo_model(wp)
                store[name] = {
                    "type": "yolo_detection",
                    "model": model,
                    "nc": int(dataset_config.get("num_classes", 80)),  # COCO default
                }
            else:
                logger.warning("Unrecognised cv_task '%s' — PRT disabled for '%s'.", cv_task, name)
                store[name] = None
        except Exception as exc:
            logger.warning("Could not load CV model object for '%s' (PRT disabled): %s", name, exc)
            store[name] = None
    return store


# ── CV initial train (weights-missing bootstrap) ────────────────────────────
#
# Mirrors _train_regression_models()'s "train if the weights file is absent"
# gate, with one deliberate difference: this uses REAL ground truth from
# train_split() (CVImageDirAdapter.train_labels()), not pseudo-labels.
# Invariant I4 (label-free) is a property of the runtime STREAMING phase —
# MAPE decisions can't peek at ground truth — it was never meant to restrict
# the initial supervised bootstrap, and regression's own train_split() is
# already fully supervised. As with PRT retrain, only the last
# finetune_n_layers layers are trained (full backbone training from a
# pretrained checkpoint is unnecessary and far more costly) — same knob,
# same _select_finetune_params() helper, shared config keys.

def _train_cv_models_if_missing(
    dataset_config: dict,
    cv_task: str,
    train_paths: list[str],
    train_label_paths: list[str | None],
    train_inline_labels: list[int | None],
    thresholds: dict,
    run_path: Path | None,
) -> None:
    """For each model whose weights_path is missing, fine-tune from a
    pretrained checkpoint on train_split() and save to weights_path.

    Called once per run, before _load_models()/_load_cv_model_store() — by
    the time those run, a model trained here loads normally like any other
    pre-existing checkpoint. A model that still has no weights afterward
    (training skipped or failed) is reported by _load_models() the same way
    a genuinely missing file always has been (None entry, excluded from
    available_models).
    """
    model_specs = dataset_config.get("models", {})
    missing = []
    for name, spec in model_specs.items():
        wp = spec["weights_path"]
        if not os.path.isabs(wp):
            wp = os.path.join(str(_TOOL_DIR), wp)
        if not os.path.exists(wp):
            missing.append((name, wp))
    if not missing:
        return

    logger.info(
        "CV initial train: %d model(s) missing weights — fine-tuning from pretrained "
        "on %d training images ...", len(missing), len(train_paths),
    )
    for name, wp in missing:
        os.makedirs(os.path.dirname(wp), exist_ok=True)
        try:
            if cv_task == "classification":
                num_classes = int(dataset_config.get("num_classes", 1000))
                ok = _initial_train_torchvision_classifier(
                    name, wp, num_classes, train_paths, train_inline_labels, thresholds,
                )
            elif cv_task == "segmentation":
                num_classes = int(dataset_config.get("num_classes", 19))
                ok = _initial_train_segformer_segmentation(
                    name, wp, num_classes, train_paths, train_label_paths, thresholds,
                )
            elif cv_task == "detection":
                ok = _initial_train_yolo_detection(
                    name, wp, train_paths, train_label_paths, thresholds, run_path,
                )
            else:
                logger.warning("CV initial train: unrecognised cv_task '%s' for '%s' — skipping.",
                                cv_task, name)
                ok = False
        except Exception as exc:
            logger.warning("CV initial train failed for '%s': %s", name, exc)
            ok = False
        if not ok:
            logger.warning(
                "CV initial train: '%s' still has no usable weights at %s — "
                "will be excluded from available_models.", name, wp,
            )


def _initial_train_torchvision_classifier(
    model_name: str,
    weights_path: str,
    num_classes: int,
    train_paths: list[str],
    train_inline_labels: list[int | None],
    thresholds: dict,
) -> bool:
    """Build an ImageNet-pretrained classifier with a fresh num_classes head,
    fine-tune it on real (path, label) pairs from train_split(), and save the
    resulting state_dict to weights_path.

    Two-phase fine-tune, not a single flat pass:
      Phase 1 (head warmup): backbone stays fully frozen at its pretrained
        weights; only the fresh, randomly-initialized head is trained, for
        finetune_warmup_epochs (default 1). Prevents the large early gradients
        a random-init head produces from being backpropagated into (and
        degrading) the last finetune_n_layers pretrained layers before the
        head has learned anything sensible — those layers stay untouched
        during this phase.
      Phase 2 (discriminative fine-tune): the last finetune_n_layers leaf
        modules (which include the now-warmed-up head) are unfrozen, and
        trained for finetune_epochs with TWO learning rates — the head at the
        full finetune_lr, everything else (the unfrozen backbone layers) at
        finetune_lr * finetune_backbone_lr_ratio (default 0.1). Standard
        discriminative-LR transfer-learning practice: the head still has the
        most to learn, the backbone layers only need small nudges.

    After this, _build_torchvision_classifier() loads the saved file exactly
    like any pre-existing checkpoint — no special-casing downstream.
    """
    import torch
    import torch.nn as nn
    import torchvision.models as tv
    from torchvision import transforms
    from PIL import Image

    paths = [p for p, lbl in zip(train_paths, train_inline_labels) if lbl is not None]
    labels = [int(lbl) for lbl in train_inline_labels if lbl is not None]

    min_images = int(thresholds.get("finetune_min_images", 20))
    if len(paths) < min_images:
        logger.warning(
            "CV initial train: only %d labeled training images for '%s' (need %d) — skipping.",
            len(paths), model_name, min_images,
        )
        return False

    name_lower = os.path.basename(weights_path).lower()
    if "efficientnet_b0" in name_lower:
        model = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        head_module = model.classifier[1]
    elif "resnet101" in name_lower:
        model = tv.resnet101(weights=tv.ResNet101_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head_module = model.fc
    elif "resnet50" in name_lower:
        model = tv.resnet50(weights=tv.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head_module = model.fc
    else:
        raise ValueError(
            f"Cannot infer torchvision architecture from filename: {weights_path}. "
            "Expected 'efficientnet_b0', 'resnet50', or 'resnet101' in the name."
        )

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    n_layers = int(thresholds.get("finetune_n_layers", 10))
    lr = float(thresholds.get("finetune_lr", 1e-4))
    epochs = int(thresholds.get("finetune_epochs", 3))
    batch_size = int(thresholds.get("finetune_batch_size", 8))
    warmup_epochs = int(thresholds.get("finetune_warmup_epochs", 1))
    backbone_lr_ratio = float(thresholds.get("finetune_backbone_lr_ratio", 0.1))

    model.to(_dev())
    loss_fn = nn.CrossEntropyLoss()

    def _run_epochs(n_epochs: int, optimizer) -> None:
        model.train()
        for _epoch in range(n_epochs):
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                batch_labels = labels[start : start + batch_size]
                if not batch_paths:
                    continue
                try:
                    imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
                except Exception as exc:
                    logger.debug("CV initial train batch load failed: %s", exc)
                    continue
                x = torch.stack(imgs).to(_dev())
                y = torch.tensor(batch_labels, dtype=torch.long).to(_dev())
                optimizer.zero_grad()
                loss_fn(model(x), y).backward()
                optimizer.step()

    for p in model.parameters():
        p.requires_grad_(False)

    # Phase 1: head-only warmup — backbone stays frozen at pretrained weights.
    head_params = list(head_module.parameters())
    if warmup_epochs > 0 and head_params:
        for p in head_params:
            p.requires_grad_(True)
        warmup_optimizer = torch.optim.Adam(head_params, lr=lr)
        _run_epochs(warmup_epochs, warmup_optimizer)

    # Phase 2: unfreeze the last n_layers (includes the head) and fine-tune
    # with a discriminative LR — head at lr, backbone layers at a fraction of it.
    finetune_params = _select_finetune_params(model, n_layers)
    if not finetune_params:
        logger.warning("CV initial train: no parameters selected for '%s' (n_layers=%d).",
                        model_name, n_layers)
        return False
    for p in finetune_params:
        p.requires_grad_(True)

    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in finetune_params if id(p) not in head_param_ids]
    param_groups = [{"params": head_params, "lr": lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_ratio})
    optimizer = torch.optim.Adam(param_groups)
    _run_epochs(epochs, optimizer)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    torch.save(model.state_dict(), weights_path)
    logger.info(
        "CV initial train: saved '%s' -> %s (%d images, %d warmup epoch(s) on head "
        "only, then %d epoch(s) with last %d layers unfrozen (%d parameter tensors) "
        "at backbone_lr_ratio=%.3f).",
        model_name, weights_path, len(paths), warmup_epochs, epochs, n_layers,
        len(finetune_params), backbone_lr_ratio,
    )
    return True


def _initial_train_segformer_segmentation(
    model_name: str,
    weights_path: str,
    num_classes: int,
    train_paths: list[str],
    train_label_paths: list[str | None],
    thresholds: dict,
) -> bool:
    """Build a SegFormer model from HuggingFace's ImageNet-pretrained MiT
    backbone with a fresh num_classes decode head, fine-tune the last
    finetune_n_layers layers on real (image, mask) pairs from train_split(),
    and save the resulting state_dict to weights_path.

    Unlike _finetune_segformer_segmentation's pseudo-mask (which is derived
    directly from the logits, so it's already at the logits' spatial
    resolution), a real ground-truth mask is full-resolution — logits are
    upsampled to match it before computing the loss (the standard SegFormer
    fine-tuning recipe), rather than downsampling the mask.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    ignore_index = 255
    pairs = [
        (p, lp) for p, lp in zip(train_paths, train_label_paths)
        if lp and os.path.exists(lp)
    ]
    min_images = int(thresholds.get("finetune_min_images", 20))
    if len(pairs) < min_images:
        logger.warning(
            "CV initial train: only %d training images with a GT mask for '%s' (need %d) — skipping.",
            len(pairs), model_name, min_images,
        )
        return False

    name_lower = os.path.basename(weights_path).lower()
    if "segformer_b2" in name_lower or "segformer-b2" in name_lower:
        arch_key = "nvidia/mit-b2"
    elif "segformer_b1" in name_lower or "segformer-b1" in name_lower:
        arch_key = "nvidia/mit-b1"
    else:
        arch_key = "nvidia/mit-b0"

    model = SegformerForSemanticSegmentation.from_pretrained(
        arch_key, num_labels=num_classes, ignore_mismatched_sizes=True,
    ).to(_dev())
    processor = SegformerImageProcessor(
        do_resize=True, size={"height": 512, "width": 512},
        do_normalize=True, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
        do_rescale=True, rescale_factor=1.0 / 255,
    )

    def _encode(path: str):
        img = Image.open(path).convert("RGB")
        return processor(images=img, return_tensors="pt")["pixel_values"].to(_dev())

    def _encode_mask(label_path: str) -> torch.Tensor:
        gt = np.array(Image.open(label_path), dtype=np.int64)
        gt[(gt < 0) | (gt >= num_classes)] = ignore_index
        return torch.from_numpy(gt).to(_dev())

    n_layers = int(thresholds.get("finetune_n_layers", 10))
    lr = float(thresholds.get("finetune_lr", 1e-4))
    epochs = int(thresholds.get("finetune_epochs", 3))
    batch_size = int(thresholds.get("finetune_batch_size", 4))

    for p in model.parameters():
        p.requires_grad_(False)
    finetune_params = _select_finetune_params(model, n_layers)
    if not finetune_params:
        logger.warning("CV initial train: no parameters selected for '%s' (n_layers=%d).",
                        model_name, n_layers)
        return False
    for p in finetune_params:
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(finetune_params, lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    model.train()
    for _epoch in range(epochs):
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            if not batch:
                continue
            try:
                xs = [_encode(p) for p, _lp in batch]
                masks = [_encode_mask(lp) for _p, lp in batch]
            except Exception as exc:
                logger.debug("CV initial train batch load failed: %s", exc)
                continue
            x = torch.cat(xs, dim=0)
            optimizer.zero_grad()
            logits = model(pixel_values=x).logits
            total_loss = None
            for i, mask in enumerate(masks):
                up = F.interpolate(
                    logits[i : i + 1], size=mask.shape, mode="bilinear", align_corners=False,
                )
                li = loss_fn(up, mask.unsqueeze(0))
                total_loss = li if total_loss is None else total_loss + li
            if total_loss is not None:
                total_loss.backward()
                optimizer.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    torch.save(model.state_dict(), weights_path)
    logger.info(
        "CV initial train: saved '%s' -> %s (%d images, last %d layers unfrozen "
        "(%d parameter tensors), %d epochs).",
        model_name, weights_path, len(pairs), n_layers, len(finetune_params), epochs,
    )
    return True


def _initial_train_yolo_detection(
    model_name: str,
    weights_path: str,
    train_paths: list[str],
    train_label_paths: list[str | None],
    thresholds: dict,
    run_path: Path | None,
) -> bool:
    """Load a COCO-pretrained YOLO checkpoint (ultralytics resolves/downloads
    it if weights_path is a recognised shorthand, e.g. "yolov8n.pt" — the
    same resolution _build_yolo_model already relies on), fine-tune the last
    finetune_n_layers layers on real GT boxes from train_split(), and save.

    No class-head resize needed: BDD100K detections are evaluated in COCO's
    own 80-class space (see offline_eval.py's _BDD_TO_COCO mapping) rather
    than a BDD-specific taxonomy, so the stock pretrained head already
    matches — this is a domain (not class-taxonomy) adaptation, same as
    _finetune_yolo_detection's PRT fine-tune.
    """
    if run_path is None:
        logger.warning("CV initial train (yolo): run_path not provided — cannot stage a temp dataset.")
        return False

    import shutil
    from PIL import Image as PILImage
    from experiments.offline_eval import _load_gt_boxes

    min_images = int(thresholds.get("finetune_min_images", 20))
    n_layers = int(thresholds.get("finetune_n_layers", 10))
    lr = float(thresholds.get("finetune_lr", 1e-4))
    epochs = int(thresholds.get("finetune_epochs", 3))
    batch_size = int(thresholds.get("finetune_batch_size", 4))
    nc = 80  # COCO head — matches the pretrained checkpoint's output space as-is

    train_dir = run_path / f"_yolo_initial_train_tmp_{model_name}"
    img_dir = train_dir / "images"
    lbl_dir = train_dir / "labels"
    if train_dir.exists():
        shutil.rmtree(train_dir, ignore_errors=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    model = None
    try:
        for p, lp in zip(train_paths, train_label_paths):
            if not lp or not os.path.exists(lp):
                continue
            try:
                with PILImage.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            boxes, classes = _load_gt_boxes(lp, w, h)
            if len(boxes) == 0:
                continue
            lines = []
            for (x1, y1, x2, y2), cls in zip(boxes, classes):
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            src = Path(p)
            shutil.copy(src, img_dir / src.name)
            (lbl_dir / f"{src.stem}.txt").write_text("\n".join(lines))
            kept += 1

        if kept < min_images:
            logger.warning(
                "CV initial train: only %d/%d training images with GT boxes for '%s' (need %d) — skipping.",
                kept, len(train_paths), model_name, min_images,
            )
            return False

        names_map = "{ " + ", ".join(f"{i}: {i}" for i in range(nc)) + " }"
        train_yaml = train_dir / "initial_train.yaml"
        train_yaml.write_text(
            f"path: {train_dir.resolve()}\n"
            f"train: images\n"
            f"val: images\n"
            f"nc: {nc}\n"
            f"names: {names_map}\n"
        )

        model = _build_yolo_model(weights_path)

        for p in model.model.parameters():
            p.requires_grad_(False)
        finetune_params = _select_finetune_params(model.model, n_layers)
        if not finetune_params:
            logger.warning("CV initial train: no parameters selected for '%s' (n_layers=%d).",
                            model_name, n_layers)
            return False
        for p in finetune_params:
            p.requires_grad_(True)

        try:
            model.train(
                data=str(train_yaml), epochs=epochs, imgsz=640, batch=batch_size,
                workers=2, patience=max(epochs, 1), pretrained=False, cache="disk",
                lr0=lr, verbose=False, device=yolo_device(),
            )
        except Exception as exc:
            logger.warning("CV initial train (yolo) train() failed for '%s': %s", model_name, exc)
            return False

    finally:
        if model is not None:
            for p in model.model.parameters():
                p.requires_grad_(False)
        shutil.rmtree(train_dir, ignore_errors=True)

    model.save(weights_path)
    logger.info(
        "CV initial train: saved '%s' -> %s (%d images, last %d layers unfrozen "
        "(%d parameter tensors), %d epochs).",
        model_name, weights_path, kept, n_layers, len(finetune_params), epochs,
    )
    return True


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
            retrain_params = info.get("retrain_params", {})
            if retrain_params:
                info["model"].set_params(**retrain_params)
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


def _select_finetune_params(model, n_layers: int) -> list:
    """Return the parameters of the last n_layers parameterised leaf modules.

    Architecture-agnostic: walks model.modules(), keeps leaf modules (no
    children) that actually own parameters (Conv2d, Linear, BatchNorm2d,
    etc.), and selects the LAST n_layers of them in forward-definition order
    — i.e. the layers closest to the output head, which is what "fine-tune
    the last N layers" means for a classifier. Works the same way for
    EfficientNet-B0 and ResNet-50/101 despite very different module trees.
    n_layers is the tunable knob requested for this feature (config key
    "finetune_n_layers", default 10).
    """
    leaf_param_modules = [
        m for m in model.modules()
        if len(list(m.children())) == 0 and list(m.parameters(recurse=False))
    ]
    selected = leaf_param_modules[-max(n_layers, 0):] if n_layers > 0 else []
    params = []
    for m in selected:
        for p in m.parameters(recurse=False):
            params.append(p)
    return params


def _do_cv_inline_finetune(
    model_name: str,
    model_store: dict,
    models: dict,
    image_path_history: list[str],
    thresholds: dict,
    drift_window: int,
    run_path: Path | None = None,
) -> bool:
    """Periodic-retrain (PRT) fine-tune dispatcher for CV models.

    Covers all three CV task families — dispatches on model_store[model_name]
    ["type"] (set by _load_cv_model_store):
      - "torchvision_classifier" -> _finetune_torchvision_classifier, or
        _finetune_torchvision_classifier_tent if thresholds["retrain_tactic"]
        == "tent" (imagenet/imagenet_c only — see DP25)
      - "segformer_segmentation" -> _finetune_segformer_segmentation
      - "yolo_detection"         -> _finetune_yolo_detection (needs run_path
        for a temp on-disk YOLO dataset — ultralytics' training API is
        filesystem-based, unlike the other two in-memory loops)

    All three share the same label-free design (invariant I4 — no ground
    truth read at runtime): pseudo-labels are the model's OWN high-confidence
    predictions on the recent drift window, and only the last
    `finetune_n_layers` parameterised layers are unfrozen (tunable, shared
    across all three task types via _select_finetune_params) — full
    end-to-end retraining of a CV backbone is far too costly to run inline,
    on a schedule, inside a headless experiment loop. Each also applies a
    safety valve comparing held-out confidence before/after, discarding the
    fine-tune and restoring pre-tune weights if it drops too far.

    Returns True on an accepted fine-tune, False if skipped (unsupported/
    unloaded model, too few confident pseudo-labels, or safety-valve reject).
    """
    info = model_store.get(model_name)
    if info is None:
        return False

    model_type = info.get("type")
    if model_type == "torchvision_classifier":
        # retrain_tactic (already present in every CV config as "pseudo_label",
        # but never consumed by anything until now — see DP19/DP25) selects
        # between the supervised pseudo-label path and TENT's entropy-
        # minimization adaptation — classification only.
        if thresholds.get("retrain_tactic", "pseudo_label") == "tent":
            return _finetune_torchvision_classifier_tent(
                model_name, model_store, models, image_path_history, thresholds, drift_window,
            )
        return _finetune_torchvision_classifier(
            model_name, model_store, models, image_path_history, thresholds, drift_window,
        )
    elif model_type == "segformer_segmentation":
        return _finetune_segformer_segmentation(
            model_name, model_store, models, image_path_history, thresholds, drift_window,
        )
    elif model_type == "yolo_detection":
        return _finetune_yolo_detection(
            model_name, model_store, models, image_path_history, thresholds, drift_window, run_path,
        )
    else:
        logger.debug("CV finetune: unrecognised model_store type '%s' for '%s'.",
                     model_type, model_name)
        return False


def _finetune_torchvision_classifier(
    model_name: str,
    model_store: dict,
    models: dict,
    image_path_history: list[str],
    thresholds: dict,
    drift_window: int,
) -> bool:
    """Pseudo-labeled last-N-layer fine-tune for a torchvision classifier.
    See _do_cv_inline_finetune's docstring for the shared design rationale.
    """
    info = model_store.get(model_name)
    if info is None or info.get("type") != "torchvision_classifier":
        return False

    import torch
    import torch.nn as nn
    from PIL import Image

    model = info["model"]
    transform = info["transform"]

    finetune_window = int(thresholds.get("finetune_window_images", drift_window))
    window = image_path_history[-finetune_window:]
    if not window:
        return False

    threshold = float(thresholds.get("pseudo_label_threshold", 0.8))
    min_images = int(thresholds.get("finetune_min_images", 20))
    n_layers = int(thresholds.get("finetune_n_layers", 10))
    lr = float(thresholds.get("finetune_lr", 1e-4))
    epochs = int(thresholds.get("finetune_epochs", 3))
    tau_regression = float(thresholds.get("tau_regression", 0.05))
    batch_size = int(thresholds.get("finetune_batch_size", 8))

    # Step 1 — pseudo-label the window with the CURRENT (pre-tune) weights.
    model.eval()
    pseudo_paths: list[str] = []
    pseudo_labels: list[int] = []
    pseudo_confs: list[float] = []
    with torch.no_grad():
        for p in window:
            try:
                img = Image.open(p).convert("RGB")
                x = transform(img).unsqueeze(0).to(_dev())
                probs = torch.softmax(model(x), dim=1)
                conf, cls = probs.max(dim=1)
                conf = float(conf.item())
                if conf >= threshold:
                    pseudo_paths.append(p)
                    pseudo_labels.append(int(cls.item()))
                    pseudo_confs.append(conf)
            except Exception as exc:
                logger.debug("CV finetune pseudo-label failed for %s: %s", p, exc)

    if len(pseudo_paths) < min_images:
        logger.debug(
            "CV finetune skipped for '%s': only %d/%d confident pseudo-labels (need %d).",
            model_name, len(pseudo_paths), len(window), min_images,
        )
        return False

    # Step 2 — hold out the last 20% (min 1) to sanity-check after fine-tuning.
    n_holdout = max(1, len(pseudo_paths) // 5)
    train_paths, holdout_paths = pseudo_paths[:-n_holdout], pseudo_paths[-n_holdout:]
    train_labels = pseudo_labels[:-n_holdout]
    pre_tune_conf = float(np.mean(pseudo_confs[-n_holdout:]))

    if len(train_paths) < 2:
        return False

    # Snapshot for rollback if the safety valve trips.
    pre_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    try:
        # Step 3 — freeze all but the last n_layers parameterised layers.
        for p in model.parameters():
            p.requires_grad_(False)
        finetune_params = _select_finetune_params(model, n_layers)
        if not finetune_params:
            logger.warning("CV finetune: no parameters selected for '%s' (n_layers=%d).",
                            model_name, n_layers)
            return False
        for p in finetune_params:
            p.requires_grad_(True)

        optimizer = torch.optim.Adam(finetune_params, lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for _epoch in range(epochs):
            for start in range(0, len(train_paths), batch_size):
                batch_paths = train_paths[start : start + batch_size]
                batch_labels = train_labels[start : start + batch_size]
                if not batch_paths:
                    continue
                try:
                    imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
                except Exception as exc:
                    logger.debug("CV finetune batch load failed: %s", exc)
                    continue
                x = torch.stack(imgs).to(_dev())
                y = torch.tensor(batch_labels, dtype=torch.long).to(_dev())
                optimizer.zero_grad()
                loss_fn(model(x), y).backward()
                optimizer.step()
        model.eval()

        # Step 4 — safety valve: re-check confidence on the SAME held-out images.
        post_confs = []
        with torch.no_grad():
            for p in holdout_paths:
                img = Image.open(p).convert("RGB")
                x = transform(img).unsqueeze(0).to(_dev())
                post_confs.append(float(torch.softmax(model(x), dim=1).max().item()))
        post_tune_conf = float(np.mean(post_confs)) if post_confs else 0.0

        if post_tune_conf < pre_tune_conf - tau_regression:
            logger.info(
                "CV finetune safety valve triggered for '%s': holdout confidence %.4f -> %.4f "
                "(drop > tau_regression=%.4f). Discarding fine-tuned weights.",
                model_name, pre_tune_conf, post_tune_conf, tau_regression,
            )
            model.load_state_dict(pre_state)
            model.eval()
            return False

    finally:
        for p in model.parameters():
            p.requires_grad_(False)

    # Step 5 — accepted: rebuild the predict closure so inference sees the new weights.
    _m, _t = model, transform

    def _new_predict(image_path, m=_m, t=_t) -> dict:
        img = Image.open(str(image_path)).convert("RGB")
        x = t(img).unsqueeze(0).to(_dev())
        with torch.no_grad():
            logits = m(x)
            probs = torch.softmax(logits, dim=1)
            proxy = float(probs.max().item())
            pred_class = int(probs.argmax().item())
        return {"proxy": proxy, "pred": pred_class}

    models[model_name] = _new_predict
    logger.info(
        "CV finetune accepted for '%s': %d images, last %d layers unfrozen "
        "(%d parameter tensors), holdout confidence %.4f -> %.4f.",
        model_name, len(train_paths), n_layers, len(finetune_params), pre_tune_conf, post_tune_conf,
    )
    return True


def _restore_bn_tracking(model, pre_state: dict) -> None:
    """Undo core.tta.tent.configure_model()'s BatchNorm mutation.

    configure_model() sets track_running_stats=False and running_mean/
    running_var=None on every BN layer (so BN always uses fresh per-batch
    statistics, per the paper). That De-registers those buffers entirely, so
    a plain model.load_state_dict(pre_state) cannot restore them — this
    helper re-enables tracking and re-registers the original buffers from a
    state_dict snapshot taken BEFORE configure_model() ran. Used only by the
    TENT safety-valve rollback path below.
    """
    import torch.nn as nn
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.track_running_stats = True
            rm_key, rv_key, nbt_key = (
                f"{name}.running_mean", f"{name}.running_var", f"{name}.num_batches_tracked",
            )
            if rm_key in pre_state:
                m.running_mean = pre_state[rm_key].clone()
                m.running_var = pre_state[rv_key].clone()
                m.num_batches_tracked = pre_state[nbt_key].clone()


def _finetune_torchvision_classifier_tent(
    model_name: str,
    model_store: dict,
    models: dict,
    image_path_history: list[str],
    thresholds: dict,
    drift_window: int,
) -> bool:
    """TENT (Wang, Shelhamer, Liu, Olshausen, Darrell — ICLR 2021) entropy-
    minimization adaptation for a torchvision classifier — selected via the
    "retrain_tactic": "tent" config key (present in every CV config as
    "pseudo_label" already, previously never consumed by anything — DP19),
    as an alternative to _finetune_torchvision_classifier's supervised
    pseudo-label fine-tune (the default for every other CV dataset). See
    context/DECISIONS_PENDING.md
    DP25 for the full design rationale: imagenet/imagenet_c specifically,
    since TENT's own headline robustness benchmark IS ImageNet-C, and
    classification only — detection/segmentation already have natural drift
    via their own domain-shift axes (weather/location/lighting), unlike
    ImageNet's engineered corruption-based drift.

    Reuses core/tta/tent.py's algorithm directly: configure_model() freezes
    every parameter except each BatchNorm layer's affine scale/shift (typically
    <1% of the model) and forces BN to estimate statistics fresh from each
    batch; forward_and_adapt() runs one entropy-minimization gradient step per
    batch. Unlike the supervised pseudo-label path:
      - No confidence gating on individual images (finetune_n_layers/
        pseudo_label_threshold do not apply) — TENT's entropy objective is
        well-defined regardless of the model's current confidence, and the
        paper adapts on every batch, confident or not.
      - Adapts ALL BatchNorm layers throughout the network, not a suffix of
        "last N layers" — this is the paper's own validated parameter set,
        not a reduction of the supervised path's knob.
      - Hyperparameters default to the paper's own ImageNet numbers
        (tent_lr=2.5e-4, tent_batch_size=64, tent_steps=1 — SGD+momentum),
        distinct config keys from finetune_lr/finetune_batch_size since the
        two tactics have different validated regimes.
    A confidence-based safety valve (tau_regression, same convention as the
    supervised path) is kept for consistency with this codebase's established
    caution, though the paper itself doesn't require one. Adaptation state
    persists across repeated calls (no reset) — this event continues from
    wherever the model's BN parameters already are, the harness-trigger-
    granularity equivalent of the paper's "online" mode (see DP25 for why
    this is episodic-per-retrain-event, not per individual inference step).
    """
    info = model_store.get(model_name)
    if info is None or info.get("type") != "torchvision_classifier":
        return False

    import torch
    from PIL import Image
    from core.tta.tent import configure_model, collect_params, forward_and_adapt

    model = info["model"]
    transform = info["transform"]

    finetune_window = int(thresholds.get("finetune_window_images", drift_window))
    window = image_path_history[-finetune_window:]
    if not window:
        return False

    min_images = int(thresholds.get("finetune_min_images", 20))
    if len(window) < min_images:
        logger.debug(
            "CV finetune (tent) skipped for '%s': only %d images in window (need %d).",
            model_name, len(window), min_images,
        )
        return False

    lr = float(thresholds.get("tent_lr", 2.5e-4))
    batch_size = int(thresholds.get("tent_batch_size", 64))
    steps = int(thresholds.get("tent_steps", 1))
    tau_regression = float(thresholds.get("tau_regression", 0.05))

    # Step 1 — hold out the last 20% (min 1) to sanity-check after adapting.
    n_holdout = max(1, len(window) // 5)
    holdout_paths = window[-n_holdout:]
    adapt_paths = window[:-n_holdout]
    if len(adapt_paths) < 2:
        return False

    def _mean_confidence(paths: list[str]) -> float:
        model.eval()
        confs = []
        with torch.no_grad():
            for p in paths:
                try:
                    img = Image.open(p).convert("RGB")
                    x = transform(img).unsqueeze(0).to(_dev())
                    confs.append(float(torch.softmax(model(x), dim=1).max().item()))
                except Exception as exc:
                    logger.debug("CV finetune (tent) confidence check failed for %s: %s", p, exc)
        return float(sum(confs) / len(confs)) if confs else 0.0

    # Snapshot BEFORE configure_model() mutates BN tracking — needed both for
    # ordinary weight rollback and (via _restore_bn_tracking) to undo that
    # BN-specific mutation if the safety valve rejects this event.
    pre_tune_conf = _mean_confidence(holdout_paths)
    pre_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    try:
        configure_model(model)
        try:
            params, _ = collect_params(model)
        except ValueError as exc:
            logger.warning("CV finetune (tent): %s", exc)
            return False
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

        adapted_batches = 0
        for start in range(0, len(adapt_paths), batch_size):
            batch_paths = adapt_paths[start : start + batch_size]
            try:
                imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
            except Exception as exc:
                logger.debug("CV finetune (tent) batch load failed: %s", exc)
                continue
            if not imgs:
                continue
            x = torch.stack(imgs).to(_dev())
            for _step in range(steps):
                forward_and_adapt(x, model, optimizer)
            adapted_batches += 1

        if adapted_batches == 0:
            _restore_bn_tracking(model, pre_state)
            model.load_state_dict(pre_state, strict=False)
            return False

        model.eval()

        post_tune_conf = _mean_confidence(holdout_paths)

        if post_tune_conf < pre_tune_conf - tau_regression:
            logger.info(
                "CV finetune (tent) safety valve triggered for '%s': holdout confidence "
                "%.4f -> %.4f (drop > tau_regression=%.4f). Discarding TENT-adapted weights.",
                model_name, pre_tune_conf, post_tune_conf, tau_regression,
            )
            _restore_bn_tracking(model, pre_state)
            model.load_state_dict(pre_state, strict=False)
            model.eval()
            return False

    finally:
        for p in model.parameters():
            p.requires_grad_(False)

    _m, _t = model, transform

    def _new_predict(image_path, m=_m, t=_t) -> dict:
        img = Image.open(str(image_path)).convert("RGB")
        x = t(img).unsqueeze(0).to(_dev())
        with torch.no_grad():
            logits = m(x)
            probs = torch.softmax(logits, dim=1)
            proxy = float(probs.max().item())
            pred_class = int(probs.argmax().item())
        return {"proxy": proxy, "pred": pred_class}

    models[model_name] = _new_predict
    logger.info(
        "CV finetune (tent) accepted for '%s': %d images (%d batches), "
        "holdout confidence %.4f -> %.4f.",
        model_name, len(adapt_paths), adapted_batches, pre_tune_conf, post_tune_conf,
    )
    return True


def _finetune_segformer_segmentation(
    model_name: str,
    model_store: dict,
    models: dict,
    image_path_history: list[str],
    thresholds: dict,
    drift_window: int,
) -> bool:
    """Pseudo-labeled last-N-layer fine-tune for a SegFormer segmentation model.

    Pseudo-labels are per-pixel: the argmax class at each pixel of the
    model's own logits, with low-confidence pixels marked `ignore_index`
    (255, matching the Cityscapes/ACDC convention already used elsewhere in
    this codebase) so the loss only trains on pixels the model is already
    confident about. The per-IMAGE gate (whether to include an image in the
    fine-tune set at all) reuses the same mean-max-softmax "proxy" value
    adapters/loaders.py::segformer_loader already computes for inference —
    so pseudo_label_threshold means the same thing here as it does for
    classification. See _do_cv_inline_finetune's docstring for the shared
    design (layer selection, safety valve).
    """
    info = model_store.get(model_name)
    if info is None or info.get("type") != "segformer_segmentation":
        return False

    import torch
    import torch.nn as nn
    from PIL import Image

    model = info["model"]
    processor = info["processor"]
    ignore_index = 255

    finetune_window = int(thresholds.get("finetune_window_images", drift_window))
    window = image_path_history[-finetune_window:]
    if not window:
        return False

    threshold = float(thresholds.get("pseudo_label_threshold", 0.7))
    min_images = int(thresholds.get("finetune_min_images", 20))
    n_layers = int(thresholds.get("finetune_n_layers", 10))
    lr = float(thresholds.get("finetune_lr", 1e-4))
    epochs = int(thresholds.get("finetune_epochs", 3))
    tau_regression = float(thresholds.get("tau_regression", 0.05))
    batch_size = int(thresholds.get("finetune_batch_size", 4))  # segmentation tensors are larger

    def _encode(path: str):
        img = Image.open(path).convert("RGB")
        return processor(images=img, return_tensors="pt")["pixel_values"].to(_dev())

    def _pseudo_mask(logits: torch.Tensor) -> tuple[torch.Tensor, float]:
        """(H, W) int64 mask with low-confidence pixels set to ignore_index,
        plus the mean-max-softmax proxy over the WHOLE image (matching
        segformer_loader's inference-time proxy definition exactly)."""
        probs = torch.softmax(logits, dim=1)
        conf, cls = probs.max(dim=1)  # (1, H, W) each
        proxy = float(conf.mean().item())
        mask = cls.clone()
        mask[conf < threshold] = ignore_index
        return mask.squeeze(0).long(), proxy

    # Step 1 — pseudo-label the window with the CURRENT (pre-tune) weights.
    model.eval()
    pseudo_paths: list[str] = []
    pseudo_masks: list[torch.Tensor] = []
    pseudo_confs: list[float] = []
    with torch.no_grad():
        for p in window:
            try:
                x = _encode(p)
                logits = model(pixel_values=x).logits
                mask, proxy = _pseudo_mask(logits)
                if proxy >= threshold and bool((mask != ignore_index).any()):
                    pseudo_paths.append(p)
                    pseudo_masks.append(mask)
                    pseudo_confs.append(proxy)
            except Exception as exc:
                logger.debug("CV finetune (segformer) pseudo-label failed for %s: %s", p, exc)

    if len(pseudo_paths) < min_images:
        logger.debug(
            "CV finetune skipped for '%s': only %d/%d confident pseudo-labels (need %d).",
            model_name, len(pseudo_paths), len(window), min_images,
        )
        return False

    # Step 2 — hold out the last 20% (min 1) to sanity-check after fine-tuning.
    n_holdout = max(1, len(pseudo_paths) // 5)
    train_paths = pseudo_paths[:-n_holdout]
    train_masks = pseudo_masks[:-n_holdout]
    holdout_paths = pseudo_paths[-n_holdout:]
    pre_tune_conf = float(np.mean(pseudo_confs[-n_holdout:]))

    if len(train_paths) < 2:
        return False

    pre_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    try:
        for p in model.parameters():
            p.requires_grad_(False)
        finetune_params = _select_finetune_params(model, n_layers)
        if not finetune_params:
            logger.warning("CV finetune: no parameters selected for '%s' (n_layers=%d).",
                            model_name, n_layers)
            return False
        for p in finetune_params:
            p.requires_grad_(True)

        optimizer = torch.optim.Adam(finetune_params, lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        model.train()
        for _epoch in range(epochs):
            for start in range(0, len(train_paths), batch_size):
                batch_paths = train_paths[start : start + batch_size]
                batch_masks = train_masks[start : start + batch_size]
                if not batch_paths:
                    continue
                try:
                    xs = [_encode(p) for p in batch_paths]
                except Exception as exc:
                    logger.debug("CV finetune (segformer) batch load failed: %s", exc)
                    continue
                x = torch.cat(xs, dim=0)
                optimizer.zero_grad()
                total_loss = None
                logits = model(pixel_values=x).logits
                for i, mask in enumerate(batch_masks):
                    # Logits and pseudo-mask share the same (decode-head) resolution
                    # by construction — both were derived from the same forward pass
                    # shape, no upsample/downsample needed for a self-training signal.
                    li = loss_fn(logits[i : i + 1], mask.unsqueeze(0))
                    total_loss = li if total_loss is None else total_loss + li
                if total_loss is not None:
                    total_loss.backward()
                    optimizer.step()
        model.eval()

        # Step 4 — safety valve: re-check confidence on the SAME held-out images.
        post_confs = []
        with torch.no_grad():
            for p in holdout_paths:
                x = _encode(p)
                logits = model(pixel_values=x).logits
                post_confs.append(float(torch.softmax(logits, dim=1).max(dim=1).values.mean().item()))
        post_tune_conf = float(np.mean(post_confs)) if post_confs else 0.0

        if post_tune_conf < pre_tune_conf - tau_regression:
            logger.info(
                "CV finetune safety valve triggered for '%s': holdout confidence %.4f -> %.4f "
                "(drop > tau_regression=%.4f). Discarding fine-tuned weights.",
                model_name, pre_tune_conf, post_tune_conf, tau_regression,
            )
            model.load_state_dict(pre_state)
            model.eval()
            return False

    finally:
        for p in model.parameters():
            p.requires_grad_(False)

    _m, _proc = model, processor

    def _new_predict(image_path, m=_m, proc=_proc) -> dict:
        img = Image.open(str(image_path)).convert("RGB")
        enc = {k: v.to(_dev()) for k, v in proc(images=img, return_tensors="pt").items()}
        with torch.no_grad():
            out = m(**enc)
            probs = torch.softmax(out.logits, dim=1)
            proxy = float(probs.max(dim=1).values.mean().item())
            pred_mask = out.logits.argmax(dim=1).squeeze(0).byte().cpu().numpy()
        return {"proxy": proxy, "pred": pred_mask}

    models[model_name] = _new_predict
    logger.info(
        "CV finetune accepted for '%s': %d images, last %d layers unfrozen "
        "(%d parameter tensors), holdout confidence %.4f -> %.4f.",
        model_name, len(train_paths), n_layers, len(finetune_params), pre_tune_conf, post_tune_conf,
    )
    return True


def _finetune_yolo_detection(
    model_name: str,
    model_store: dict,
    models: dict,
    image_path_history: list[str],
    thresholds: dict,
    drift_window: int,
    run_path: Path | None,
) -> bool:
    """Pseudo-labeled last-N-layer fine-tune for a YOLO detection model.

    Adapted from managed_system_cv/retrain_tactics/pseudo_label.py (the
    existing, if currently-disconnected — see DP19 — reference
    implementation for YOLO pseudo-label fine-tuning), with two changes to
    match this harness's design:
      1. Layer selection uses the same architecture-agnostic
         _select_finetune_params(model.model, n_layers) as the other two CV
         task types, instead of pseudo_label.py's "only the very last
         top-level child module" scheme — one consistent finetune_n_layers
         knob across all CV tasks, per the request that added this feature.
      2. A held-out-confidence safety valve is applied here too (pseudo_label.py
         has its own, functionally equivalent version).

    ultralytics' training API (model.train(...)) is filesystem-based, not an
    in-memory batch loop like the other two task types — this writes a small
    temporary YOLO-format dataset under run_path and cleans it up afterward.
    """
    info = model_store.get(model_name)
    if info is None or info.get("type") != "yolo_detection":
        return False
    if run_path is None:
        logger.warning("CV finetune (yolo): run_path not provided — cannot stage a temp dataset.")
        return False

    import shutil
    model = info["model"]
    nc = info.get("nc", 80)

    finetune_window = int(thresholds.get("finetune_window_images", drift_window))
    window = image_path_history[-finetune_window:]
    if not window:
        return False

    threshold = float(thresholds.get("pseudo_label_threshold", 0.5))
    min_images = int(thresholds.get("finetune_min_images", 20))
    n_layers = int(thresholds.get("finetune_n_layers", 10))
    lr = float(thresholds.get("finetune_lr", 1e-4))
    epochs = int(thresholds.get("finetune_epochs", 3))
    tau_regression = float(thresholds.get("tau_regression", 0.05))
    batch_size = int(thresholds.get("finetune_batch_size", 4))

    def _mean_conf(paths: list[str]) -> float:
        confs: list[float] = []
        for p in paths:
            results = model.predict(str(p), verbose=False, device=yolo_device())
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                confs.extend(results[0].boxes.conf.tolist())
        return float(np.mean(confs)) if confs else 0.0

    # Step 1 — held-out confidence BEFORE any pseudo-labeling/training touches the model.
    n_holdout = max(1, len(window) // 5)
    holdout_paths = window[-n_holdout:]
    train_candidates = window[:-n_holdout]
    pre_tune_conf = _mean_conf(holdout_paths)

    # Step 2 — pseudo-label the training candidates (boxes above threshold).
    from pathlib import Path as _Path
    pseudo_dir = (run_path or _Path(".")) / f"_yolo_pseudo_tmp_{model_name}"
    img_dir = pseudo_dir / "images"
    lbl_dir = pseudo_dir / "labels"
    if pseudo_dir.exists():
        shutil.rmtree(pseudo_dir, ignore_errors=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    try:
        for p in train_candidates:
            try:
                results = model.predict(str(p), conf=threshold, verbose=False, device=yolo_device())
            except Exception as exc:
                logger.debug("CV finetune (yolo) predict failed for %s: %s", p, exc)
                continue
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                continue
            lines = []
            for box in results[0].boxes:
                cls = int(box.cls[0])
                cx, cy, w, h = box.xywhn[0].tolist()
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if not lines:
                continue
            src = _Path(p)
            shutil.copy(src, img_dir / src.name)
            (lbl_dir / f"{src.stem}.txt").write_text("\n".join(lines))
            kept += 1

        if kept < min_images:
            logger.debug(
                "CV finetune skipped for '%s': only %d/%d images with confident pseudo-boxes (need %d).",
                model_name, kept, len(train_candidates), min_images,
            )
            return False

        # Step 3 — freeze all but the last n_layers parameterised layers (shared knob).
        for p in model.model.parameters():
            p.requires_grad_(False)
        finetune_params = _select_finetune_params(model.model, n_layers)
        if not finetune_params:
            logger.warning("CV finetune: no parameters selected for '%s' (n_layers=%d).",
                            model_name, n_layers)
            return False
        for p in finetune_params:
            p.requires_grad_(True)

        names_map = "{ " + ", ".join(f"{i}: {i}" for i in range(nc)) + " }"
        train_yaml = pseudo_dir / "pseudo_retrain.yaml"
        train_yaml.write_text(
            f"path: {pseudo_dir.resolve()}\n"
            f"train: images\n"
            f"val: images\n"
            f"nc: {nc}\n"
            f"names: {names_map}\n"
        )

        try:
            model.train(
                data=str(train_yaml), epochs=epochs, imgsz=640, batch=batch_size,
                workers=2, patience=max(epochs, 1), pretrained=False, cache="disk",
                lr0=lr, verbose=False, device=yolo_device(),
            )
        except Exception as exc:
            logger.warning("CV finetune (yolo) train() failed for '%s': %s", model_name, exc)
            return False

        # Step 4 — safety valve.
        post_tune_conf = _mean_conf(holdout_paths)
        if post_tune_conf < pre_tune_conf - tau_regression:
            logger.info(
                "CV finetune safety valve triggered for '%s': holdout confidence %.4f -> %.4f "
                "(drop > tau_regression=%.4f). Weights already mutated in-place by "
                "ultralytics — model_store entry is now stale; the caller's "
                "_archive_in_vmr(tag='pre_retrain') snapshot (taken before this call) "
                "is the recovery path, not an in-memory restore.",
                model_name, pre_tune_conf, post_tune_conf, tau_regression,
            )
            return False

    finally:
        for p in model.model.parameters():
            p.requires_grad_(False)
        shutil.rmtree(pseudo_dir, ignore_errors=True)

    def _new_predict(inputs, m=model) -> Any:
        return m(inputs, verbose=False, device=yolo_device())

    models[model_name] = _new_predict
    logger.info(
        "CV finetune accepted for '%s': %d images, last %d layers unfrozen "
        "(%d parameter tensors), holdout confidence %.4f -> %.4f.",
        model_name, kept, n_layers, len(finetune_params), pre_tune_conf, post_tune_conf,
    )
    return True


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
        existing = model_store.get(model_name) or {}
        retrain_params = existing.get("retrain_params", {})
        if version_path.endswith((".pth", ".pt")):
            from adapters.loaders import LSTMModel
            import torch
            m = LSTMModel()
            m.load_state_dict(torch.load(version_path, map_location="cpu", weights_only=False))
            m.eval()
            model_store[model_name] = {"type": "lstm", "model": m, "retrain_params": retrain_params}
            _m = m
            def _new_predict(inputs: np.ndarray, m=_m) -> float:
                import torch as _t
                x = _t.tensor(inputs, dtype=_t.float32).view(1, -1, 1)
                with _t.no_grad():
                    return float(m(x).item())
        else:
            with open(version_path, "rb") as f:
                m = pickle.load(f)
            model_store[model_name] = {"type": "sklearn", "model": m, "retrain_params": retrain_params}
            _m = m
            def _new_predict(inputs: np.ndarray, m=_m) -> float:
                return float(m.predict(inputs.reshape(1, -1))[0])
        models[model_name] = _new_predict
        return True
    except Exception as exc:
        logger.warning("VMR restore failed for '%s' from %s: %s", model_name, version_path, exc)
        return False


def _do_cv_vmr_restore(
    version_path: str,
    model_name: str,
    models: dict,
    model_store: dict,
) -> bool:
    """CV counterpart of _do_vmr_restore — dispatches on model_store's
    recorded type rather than file extension (LSTM, torchvision classifier,
    and SegFormer checkpoints are all .pt/.pth; only the YOLO/ultralytics
    checkpoint is self-describing enough to not need this).

    Requires model_store[model_name] to already describe a known CV type
    (set by _load_cv_model_store()); rebuilds a fresh model of that same
    architecture and loads the VMR-archived weights into it.
    """
    info = model_store.get(model_name)
    model_type = info.get("type") if info else None

    if model_type == "torchvision_classifier":
        return _cv_vmr_restore_torchvision_classifier(version_path, model_name, models, model_store, info)
    elif model_type == "segformer_segmentation":
        return _cv_vmr_restore_segformer(version_path, model_name, models, model_store, info)
    elif model_type == "yolo_detection":
        return _cv_vmr_restore_yolo(version_path, model_name, models, model_store)
    else:
        logger.warning(
            "CV VMR restore skipped for '%s': model_store has no known CV "
            "entry to restore into (type=%s).", model_name, model_type,
        )
        return False


def _cv_vmr_restore_torchvision_classifier(version_path, model_name, models, model_store, info) -> bool:
    try:
        import torch
        import torchvision.models as tv

        arch = info["arch"]
        num_classes = info["num_classes"]
        create_fn = getattr(tv, arch)
        model = create_fn(weights=None)
        if arch == "efficientnet_b0":
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        else:  # resnet50 / resnet101
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

        state = torch.load(version_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.to(_dev())
        model.eval()

        transform = info["transform"]
        model_store[model_name] = {
            "type": "torchvision_classifier",
            "model": model,
            "transform": transform,
            "arch": arch,
            "num_classes": num_classes,
        }

        from PIL import Image
        _m, _t = model, transform

        def _new_predict(image_path, m=_m, t=_t) -> dict:
            img = Image.open(str(image_path)).convert("RGB")
            x = t(img).unsqueeze(0).to(_dev())
            with torch.no_grad():
                logits = m(x)
                probs = torch.softmax(logits, dim=1)
                proxy = float(probs.max().item())
                pred_class = int(probs.argmax().item())
            return {"proxy": proxy, "pred": pred_class}

        models[model_name] = _new_predict
        return True
    except Exception as exc:
        logger.warning("CV VMR restore failed for '%s' from %s: %s", model_name, version_path, exc)
        return False


def _cv_vmr_restore_segformer(version_path, model_name, models, model_store, info) -> bool:
    try:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
        from adapters.loaders import _SEGFORMER_ARCH

        arch = info["arch"]
        num_classes = info["num_classes"]
        cfg = SegformerConfig(**_SEGFORMER_ARCH[arch], num_labels=num_classes)
        model = SegformerForSemanticSegmentation(cfg)
        state = torch.load(version_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)
        model.to(_dev())
        model.eval()

        processor = info["processor"]
        model_store[model_name] = {
            "type": "segformer_segmentation",
            "model": model,
            "processor": processor,
            "arch": arch,
            "num_classes": num_classes,
        }

        from PIL import Image
        _m, _proc = model, processor

        def _new_predict(image_path, m=_m, proc=_proc) -> dict:
            img = Image.open(str(image_path)).convert("RGB")
            enc = {k: v.to(_dev()) for k, v in proc(images=img, return_tensors="pt").items()}
            with torch.no_grad():
                out = m(**enc)
                probs = torch.softmax(out.logits, dim=1)
                proxy = float(probs.max(dim=1).values.mean().item())
                pred_mask = out.logits.argmax(dim=1).squeeze(0).byte().cpu().numpy()
            return {"proxy": proxy, "pred": pred_mask}

        models[model_name] = _new_predict
        return True
    except Exception as exc:
        logger.warning("CV VMR restore failed for '%s' from %s: %s", model_name, version_path, exc)
        return False


def _cv_vmr_restore_yolo(version_path, model_name, models, model_store) -> bool:
    """YOLO checkpoints (via ultralytics .save()) are self-describing —
    no architecture bookkeeping needed, unlike the other two CV types."""
    try:
        model = _build_yolo_model(version_path)
        model_store[model_name] = {
            "type": "yolo_detection",
            "model": model,
            "nc": model_store.get(model_name, {}).get("nc", 80) if model_store.get(model_name) else 80,
        }

        def _new_predict(inputs, m=model) -> Any:
            return m(inputs, verbose=False, device=yolo_device())

        models[model_name] = _new_predict
        return True
    except Exception as exc:
        logger.warning("CV VMR restore failed for '%s' from %s: %s", model_name, version_path, exc)
        return False


_VMR_RAW_MAX = 2000


def _subsample_raw(values, max_len: int = _VMR_RAW_MAX) -> list[float]:
    """Evenly subsample a value window to at most max_len floats (keeps VMR
    distribution.json small while preserving the window's distribution)."""
    arr = np.asarray(values, dtype=float)
    if len(arr) > max_len:
        idx = np.linspace(0, len(arr) - 1, max_len).astype(int)
        arr = arr[idx]
    return [round(float(x), 6) for x in arr]


def _load_reference_bin_edges(run_path: Path, n_bins: int) -> np.ndarray | None:
    """Load the fixed drift-reference bin edges for this run, if available.

    Returns None (never raises) if reference_distribution.json is missing or
    its bin count doesn't match n_bins — callers fall back to auto-ranged
    bins in that case rather than fail archiving outright.
    """
    ref_path = Path(run_path) / "reference_distribution.json"
    try:
        with open(ref_path) as f:
            data = json.load(f)
        edges = np.array(data["bin_edges"], dtype=float)
        if len(edges) == n_bins + 1:
            return edges
    except Exception:
        pass
    return None


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

    The archived distribution is histogrammed on the SAME fixed reference bin
    edges used to build the "current window" distribution at match time
    (core/vmr.py::_closest_histogram compares the two bin-for-bin, with no
    re-binning of its own) — previously this used `np.histogram(window,
    bins=n_bins)`, an integer bin COUNT that auto-ranges to that window's own
    min/max every call, so every archived version ended up on different,
    mutually incomparable bin edges. That made every closest_distribution
    match essentially noise: bin[i] in one archive and bin[i] in another (or
    in the live query) rarely represented the same value range at all, so the
    computed KL distance almost never cleared `tau_drift` regardless of how
    many versions were archived or how similar their source data actually
    was. Falls back to the old auto-ranged behavior only if
    reference_distribution.json can't be loaded (should not happen in normal
    operation — both harnesses write it before any archiving can occur).

    tag="pre_retrain"  — snapshot of the model BEFORE it is overwritten.
    tag="retrain"      — snapshot of the model AFTER retraining completes.
    tag="initial"      — initial seed written at run start.
    """
    import pickle
    info = model_store.get(model_name)
    if info is None:
        return

    try:
        if info["type"] in ("lstm", "torchvision_classifier", "segformer_segmentation"):
            import torch
            tmp_path = str(run_path / f"_vmr_tmp_{model_name}.pth")
            torch.save(info["model"].state_dict(), tmp_path)
        elif info["type"] == "yolo_detection":
            # ultralytics' own .save() writes a self-describing checkpoint
            # (architecture + weights) — reloadable directly via YOLO(path),
            # unlike a bare state_dict which would need external arch bookkeeping.
            tmp_path = str(run_path / f"_vmr_tmp_{model_name}.pt")
            info["model"].save(tmp_path)
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
            bin_edges = _load_reference_bin_edges(run_path, n_bins)
            hist, _ = np.histogram(window, bins=bin_edges if bin_edges is not None else n_bins)
            # "raw" (audit E2, 2026-09-28): also keep the raw window values,
            # like the original HarmonE kept each version's data.csv in
            # versionedMR/. core/vmr.py re-bins raw-vs-raw on edges spanning
            # BOTH windows, so values outside the training range are no longer
            # silently dropped by the fixed reference edges. The binned "data"
            # is kept for callers whose query has no raw window.
            distribution = {
                "type": "histogram",
                "data": hist.tolist(),
                "raw": _subsample_raw(window),
            }
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
    if is_cv:
        # All CV work runs on core.device's device (config "device"; "cuda"
        # fails fast if no GPU) — see core/device.py.
        configure_device(thresholds)

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
    else:
        train_label_paths, train_inline_labels = adapter.train_labels()
        _train_cv_models_if_missing(
            dataset_config, cv_task, train_paths, train_label_paths, train_inline_labels,
            thresholds, run_path,
        )

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

    # Load raw model objects for inline retraining/fine-tuning.
    # Regression: LSTM/sklearn objects (_load_model_store). CV: all three
    # task families — classification/segmentation/detection — via
    # _load_cv_model_store; a model that fails to load gets a None entry and
    # PRT/VMR cleanly no-ops for it (see that function's docstring).
    model_store: dict = (
        _load_cv_model_store(dataset_config, cv_task) if is_cv
        else _load_model_store(dataset_config)
    )
    seq_length: int = int(dataset_config.get("seq_length", 5))

    # ── VMR ───────────────────────────────────────────────────────────────────
    # Scoped per dataset AND per planner (knowledge/vmr/<dataset>/<planner>/<model>/
    # <version>/) so that pems/lstm and uci_electricity/lstm are independent
    # namespaces, AND so that e.g. naive_prt's retrains can't be silently reused
    # by a later harmone_original/violation_aware/pareto run in the same grid —
    # confirmed happening in practice (2026-09-03): naive_prt's retrain-archived
    # versions were being picked up via VMR "replace" by every switching planner
    # that ran afterward in the same dataset's grid, since the old path was only
    # keyed by dataset_name. Different seeds of the SAME planner still share this
    # pool deliberately (not scoped further) — only cross-planner leakage is
    # closed here. Pre-existing dataset-only VMR trees (accumulated since
    # 2026-08-30) are NOT migrated by this change — they're simply orphaned under
    # the old `knowledge/vmr/<dataset>/<model>/` path; a cleanup/migration pass
    # is tracked as separate follow-up work, not done here.
    _vmr_dir = _TOOL_DIR / "knowledge" / "vmr" / dataset_name / planner_name
    _vmr_dir.mkdir(parents=True, exist_ok=True)
    vmr = VMR(base_dir=str(_vmr_dir))

    if not is_cv:
        # Seed VMR with initial model weights + training distribution so that
        # vmr.best_match() has candidates from the very first drift event.
        # Mirrors managed_system_regression/train.py's versionedMR/ seeding.
        _train_hist, _ = np.histogram(train_values, bins=_DRIFT_N_BINS)
        _initial_dist = {
            "type": "histogram",
            "data": _train_hist.tolist(),
            "raw": _subsample_raw(train_values),  # audit E2: raw-vs-raw VMR matching
        }
        _seed_vmr_initial(available_models, dataset_config, vmr, _initial_dist)
    else:
        # Same idea for CV: seed with the luminance reference already computed
        # for the drift detector (_build_cv_reference), so a "replace" match is
        # reachable from the first drift event even before any PRT fine-tune
        # has happened. _seed_vmr_initial only copies the weights file — it is
        # architecture-agnostic and needs no CV-specific changes.
        with open(ref_dist_path) as _f:
            _cv_ref = json.load(_f)
        _initial_dist = {"type": "histogram", "data": _cv_ref["histogram"]}
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
    # CV only: rolling history of recent image paths, for PRT pseudo-label
    # fine-tuning (_do_cv_inline_finetune) — value_history holds luminance
    # floats for drift detection, not the paths themselves.
    image_path_history: list[str] = []

    # Batch accumulators (reset every monitor_interval steps)
    batch_y_true: list[float] = []
    batch_y_pred: list[float] = []
    batch_energy_uJ: list[float] = []

    # Predictions log (written to CSV at end)
    prediction_rows: list[dict] = []
    mape_events: list[dict] = []
    planner_decisions: list[dict] = []  # Phase 0.1: per-decision log with EMA snapshots

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
            image_path_history.append(image_path)

            batch_y_true.append(proxy_acc)
            batch_y_pred.append(proxy_acc)  # placeholder; accuracy= kwarg used in monitor
            batch_energy_uJ.append(energy_uJ)

            prediction_rows.append({
                "step": step,
                "proxy_acc": round(proxy_acc, 6),
                "active_model": current_model,
                "planner": planner_name,
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
            # Wrap the full MAPE cycle so that retrain, VMR restore, and heavy
            # planner computation are attributed to mape_k_energy_uJ instead of
            # bleeding into the adjacent inference step RAPL readings.
            _mape_em = EnergyMeter("mape_cycle", backend=energy_backend)
            _mape_em.__enter__()
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
            # Works identically for both domains — value_history holds true
            # values for regression, luminance for CV, and drift_detector.window_size
            # already reflects each domain's own configured window.
            current_dist: dict | None = None
            if drift_detector._bin_edges is not None:
                _window = value_history[-drift_detector.window_size:]
                if len(_window) >= drift_detector.window_size:
                    _hist, _ = np.histogram(_window, bins=drift_detector._bin_edges)
                    current_dist = {
                        "type": "histogram",
                        "data": _hist.tolist(),
                        "raw": _subsample_raw(_window),  # VMR raw-vs-raw matching (audit E2)
                    }
            drift_result = _analyse_drift(
                value_history, drift_detector, thresholds,
                vmr=vmr,
                current_model=current_model,
                current_distribution=current_dist,
            )

            # Update energy boundary (B1 fix)
            _update_energy_boundary(mape_info, telemetry, thresholds)

            # Phase 1.4: record that current_model was active at this step
            mape_info["last_observed_step"][current_model] = step
            mape_info["steps_since_last_switch"] = (
                mape_info.get("steps_since_last_switch", 0) + 1
            )

            # Plan (runs inside the outer mape_cycle EnergyMeter — nesting forbidden)
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

            # Phase 0.1: per-decision log with EMA snapshots captured at decision time
            planner_decisions.append({
                "step": step,
                "violation": violation,
                "drift_detected": drift_result["drift_detected"],
                "decision_action": decision.action,
                "decision_model": decision.model,
                "decision_reason": decision.reason,
                "current_model": current_model,
                **{f"ema_score_{m}": round(mape_info["ema_scores"].get(m, 0.5), 6)
                   for m in available_models},
                **{f"ema_acc_{m}": round(mape_info["ema_accuracy"].get(m, 0.5), 6)
                   for m in available_models},
                **{f"ema_eng_{m}": round(mape_info["ema_energy"].get(m, 0.5), 6)
                   for m in available_models},
            })

            # Phase 1: count noops that occur despite a violation being active
            if violation is not None and decision.action == "noop":
                mape_info["event_counters"]["noop_on_violation"] = (
                    mape_info["event_counters"].get("noop_on_violation", 0) + 1
                )

            # Execute — VMR restore or inline retrain/fine-tune for drift; switch for score/energy
            if decision.action in ("retrain", "replace"):
                vmr_done = False

                if decision.action == "replace" and decision.version_path:
                    if is_cv:
                        vmr_done = _do_cv_vmr_restore(
                            decision.version_path, current_model, models, model_store,
                        )
                    else:
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
                    if is_cv:
                        # PRT fine-tune: last finetune_n_layers layers only,
                        # pseudo-labeled (label-free, invariant I4). Covers
                        # all three CV task types (classification/segmentation/
                        # detection — see _do_cv_inline_finetune's docstring).
                        # No-ops cleanly (False) when a model failed to load
                        # or too few confident pseudo-labels exist in the window.
                        retrained = _do_cv_inline_finetune(
                            current_model, model_store, models, image_path_history,
                            thresholds=thresholds,
                            drift_window=int(thresholds.get("drift_window_size", _DRIFT_WINDOW_SIZE)),
                            run_path=run_path,
                        )
                    else:
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
                    energy_uJ=0.0,
                )
            _mape_em.__exit__(None, None, None)
            mape_info["event_counters"]["mape_k_energy_uJ"] += (_mape_em.total_uJ or 0.0)
            old_model = current_model
            current_model = new_model
            if current_model != old_model:
                mape_info["steps_since_last_switch"] = 0
                mape_info["last_observed_step"][current_model] = step

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

    # planner_decisions.csv — Phase 0.1: per-decision log with EMA snapshots
    decisions_path = run_path / "planner_decisions.csv"
    if planner_decisions:
        decision_fields = list(planner_decisions[0].keys())
        with open(decisions_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=decision_fields)
            writer.writeheader()
            writer.writerows(planner_decisions)
    else:
        decisions_path = None

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
        "compute_device": device_info() if is_cv else {"device": "cpu"},
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
            "planner_decisions_csv": str(decisions_path) if decisions_path else None,
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

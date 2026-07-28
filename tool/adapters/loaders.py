"""
adapters/loaders.py — Model loaders for the experiment harness.

Loader references in configs/datasets/<name>.json point here:
    "loader": "adapters.loaders.lstm_loader"
    "loader": "adapters.loaders.sklearn_loader"
    "loader": "adapters.loaders.yolo_loader"     (CV; stub for now)

Each loader function takes (weights_path, **kwargs) and returns a callable
    predict(inputs: np.ndarray) -> float | Any

LSTMModel is also exported for use by managed_system_regression/inference.py
so both code paths share one definition.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Callable

import numpy as np


# ── LSTM ──────────────────────────────────────────────────────────────────────

class LSTMModel:
    """PyTorch LSTM for scalar regression; loaded lazily so torch is optional."""

    def __new__(cls):
        import torch
        import torch.nn as nn

        class _LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
                self.fc = nn.Linear(50, 1)

            def forward(self, x):
                _, (h_n, _) = self.lstm(x)
                return self.fc(h_n[-1])

        return _LSTM()


def lstm_loader(weights_path: str, **kwargs) -> Callable[[np.ndarray], float]:
    """Load an LSTM model from a .pth weights file.

    Returns a callable predict(inputs: np.ndarray) -> float.
    inputs must be a 1-D array of shape (seq_length,).
    """
    import torch

    seq_length = kwargs.get("seq_length", 5)

    model = LSTMModel()
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()

    def predict(inputs: np.ndarray) -> float:
        x = torch.tensor(inputs, dtype=torch.float32).view(1, -1, 1)
        with torch.no_grad():
            return float(model(x).item())

    return predict


def sklearn_loader(weights_path: str, **kwargs) -> Callable[[np.ndarray], float]:
    """Load a scikit-learn model from a .pkl file.

    Returns a callable predict(inputs: np.ndarray) -> float.
    inputs must be a 1-D array of shape (seq_length,) — reshaped to (1, seq_length).
    """
    with open(weights_path, "rb") as f:
        model = pickle.load(f)

    def predict(inputs: np.ndarray) -> float:
        return float(model.predict(inputs.reshape(1, -1))[0])

    return predict


def yolo_loader(weights_path: str, **kwargs) -> Callable[[Any], Any]:
    """Load a YOLO model from a .pt file (CV domain).

    Returns a callable that runs detection on an image path or PIL Image.
    """
    from ultralytics import YOLO
    model = YOLO(weights_path)

    def predict(inputs: Any) -> Any:
        return model(inputs, verbose=False)

    return predict


# ── Registry ─────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Callable] = {
    "adapters.loaders.lstm_loader": lstm_loader,
    "adapters.loaders.sklearn_loader": sklearn_loader,
    "adapters.loaders.yolo_loader": yolo_loader,
}


def get_loader(loader_ref: str) -> Callable:
    """Look up a loader function by its dotted reference string.

    Args:
        loader_ref: e.g. "adapters.loaders.lstm_loader"

    Returns:
        The loader callable.

    Raises:
        KeyError if the reference is not registered.
    """
    if loader_ref in _REGISTRY:
        return _REGISTRY[loader_ref]
    raise KeyError(
        f"Unknown loader '{loader_ref}'. "
        f"Available: {sorted(_REGISTRY)}"
    )


def load_model(model_spec: dict, config_dir: str = ".", **kwargs) -> Callable:
    """Load a model from a ModelSpec dict.

    Args:
        model_spec: dict with keys "weights_path" and "loader".
        config_dir: Base directory for resolving relative weights_path.
        **kwargs:   Forwarded to the loader (e.g. seq_length).

    Returns:
        A predict callable.
    """
    weights_path = model_spec["weights_path"]
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(config_dir, weights_path)

    loader_fn = get_loader(model_spec["loader"])
    return loader_fn(weights_path, **kwargs)

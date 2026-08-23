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
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS.update({"sync": False})  # disable telemetry / update checks
    except Exception:
        pass
    model = YOLO(weights_path)

    def predict(inputs: Any) -> Any:
        return model(inputs, verbose=False)

    return predict


def torchvision_loader(weights_path: str, **kwargs) -> Callable[[Any], float]:
    """Load an EfficientNet or ResNet classification model from a .pt state-dict.

    Architecture is inferred from the filename (efficientnet_b0, resnet50, resnet101).
    Returns a callable predict(image_path) -> float (max softmax probability).
    """
    import torch
    import torchvision.models as tv
    from PIL import Image
    from torchvision import transforms

    num_classes = int(kwargs.get("num_classes", 182))
    name_lower = os.path.basename(weights_path).lower()

    if "efficientnet_b0" in name_lower:
        model = tv.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif "resnet101" in name_lower:
        model = tv.resnet101(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif "resnet50" in name_lower:
        model = tv.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(
            f"Cannot infer torchvision architecture from filename: {weights_path}. "
            "Expected 'efficientnet_b0', 'resnet50', or 'resnet101' in the name."
        )

    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()

    _tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def predict(inputs: Any) -> dict:
        img = Image.open(str(inputs)).convert("RGB")
        x = _tf(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            proxy = float(probs.max().item())
            pred_class = int(probs.argmax().item())
        return {"proxy": proxy, "pred": pred_class}

    return predict


# Hardcoded SegFormer MiT backbone architectures — avoids any HuggingFace network calls.
# Values match the official nvidia/mit-b{0,1,2} config.json files exactly.
_SEGFORMER_ARCH: dict[str, dict] = {
    "nvidia/mit-b0": dict(
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sequence_reduction_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
    ),
    "nvidia/mit-b1": dict(
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sequence_reduction_ratios=[8, 4, 2, 1],
        hidden_sizes=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
    ),
    "nvidia/mit-b2": dict(
        num_encoder_blocks=4,
        depths=[3, 4, 6, 3],
        sequence_reduction_ratios=[8, 4, 2, 1],
        hidden_sizes=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
    ),
}


def segformer_loader(weights_path: str, **kwargs) -> Callable[[Any], dict]:
    """Load a SegFormer segmentation model from a local .pt state-dict.

    Architecture (b0/b1/b2) is inferred from the filename.  All model and
    processor construction is fully local — no HuggingFace network calls.

    Returns a callable predict(image_path) -> {"proxy": float, "pred": np.ndarray}
    where "proxy" is mean max-class confidence and "pred" is a uint8 H×W class mask
    at 1/4 input resolution (suitable for offline mIoU without re-running inference).
    """
    import torch
    from PIL import Image
    from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor

    num_classes = int(kwargs.get("num_classes", 19))
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
    model.eval()

    # Standard ImageNet preprocessing — identical to what HF hub config.json specifies.
    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": 512, "width": 512},
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_rescale=True,
        rescale_factor=1.0 / 255,
    )

    def predict(inputs: Any) -> dict:
        img = Image.open(str(inputs)).convert("RGB")
        enc = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=1)
            proxy = float(probs.max(dim=1).values.mean().item())
            pred_mask = out.logits.argmax(dim=1).squeeze(0).byte().cpu().numpy()
        return {"proxy": proxy, "pred": pred_mask}

    return predict


# ── Registry ─────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Callable] = {
    "adapters.loaders.lstm_loader": lstm_loader,
    "adapters.loaders.sklearn_loader": sklearn_loader,
    "adapters.loaders.yolo_loader": yolo_loader,
    "adapters.loaders.torchvision_loader": torchvision_loader,
    "adapters.loaders.segformer_loader": segformer_loader,
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

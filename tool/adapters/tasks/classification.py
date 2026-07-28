"""adapters/tasks/classification.py — Classification task adapter (torchvision family).

R5-c corrections applied:
- weights=None replaces deprecated pretrained=False
- Classifier head resized to config["num_classes"] BEFORE load_state_dict
- Embedding dim differs per architecture but is documented in the reference npz
  (irrelevant for drift once R4 pins one embedding model)
- Transform built once in load_model, not per-frame
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import TaskAdapter, register_task


@register_task("classification")
class ClassificationAdapter(TaskAdapter):
    """torchvision-family image classification adapter.

    Config keys consumed:
        num_classes (int):     Number of output classes.  REQUIRED.
        embedding_dim (int):   Expected embedding dim.  Validated on first extraction.
        architecture (str):    torchvision model name, e.g. "resnet50" (default: "resnet50").
        image_size (int):      Resize side length for transform (default: 224).
    """

    task = "classification"

    def __init__(self, config: dict) -> None:
        self._config = config
        self._num_classes: int = config.get("num_classes", 2)
        self._arch: str = config.get("architecture", "resnet50")
        self._img_size: int = config.get("image_size", 224)
        self._embedding_dim: int | None = config.get("embedding_dim")

    def load_model(self, model_name: str, weights_path: str) -> "_LoadedClassifier":
        """Load a torchvision model with head resized to num_classes."""
        import torch
        import torchvision.models as tvm
        import torchvision.transforms as T

        # Build model with random weights — head will be resized, then state loaded
        create_fn = getattr(tvm, self._arch, None)
        if create_fn is None:
            raise ValueError(f"ClassificationAdapter: unknown torchvision model '{self._arch}'")

        model = create_fn(weights=None)

        # Resize final linear layer to num_classes before loading weights
        if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, self._num_classes)
        elif hasattr(model, "classifier"):
            clf = model.classifier
            if isinstance(clf, torch.nn.Sequential):
                last = clf[-1]
                if isinstance(last, torch.nn.Linear):
                    clf[-1] = torch.nn.Linear(last.in_features, self._num_classes)
            elif isinstance(clf, torch.nn.Linear):
                model.classifier = torch.nn.Linear(clf.in_features, self._num_classes)

        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.eval()

        # Build transform once
        transform = T.Compose([
            T.Resize((self._img_size, self._img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Register embedding hook on the penultimate pooling layer
        hook_state = _ClassifierHookState()
        _attach_pooling_hook(model, hook_state)

        return _LoadedClassifier(model=model, transform=transform, hook_state=hook_state)

    def infer(self, model: "_LoadedClassifier", input_path: str) -> Any:
        """Run classification and return (logits_tensor, top_class_idx)."""
        import torch
        from PIL import Image

        img = Image.open(input_path).convert("RGB")
        x = model.transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model.model(x)
        return logits

    def extract_proxy(self, result: Any) -> float:
        """Max softmax probability as confidence proxy."""
        import torch
        if result is None:
            raise ValueError("extract_proxy: result is None")
        probs = torch.softmax(result, dim=-1)
        return float(probs.max().item())

    def extract_embedding(self, model: "_LoadedClassifier", input_path: str) -> np.ndarray:
        """Run forward and return the penultimate pooled layer embedding."""
        import torch
        from PIL import Image

        img = Image.open(input_path).convert("RGB")
        x = model.transform(img).unsqueeze(0)
        with torch.no_grad():
            _ = model.model(x)

        feat = model.hook_state._embedding
        if feat is None:
            raise RuntimeError("ClassificationAdapter: embedding hook did not fire.")

        emb = feat.squeeze().cpu().float().numpy()
        if emb.ndim > 1:
            emb = emb.reshape(-1)

        if self._embedding_dim is not None and emb.shape[0] != self._embedding_dim:
            raise ValueError(
                f"ClassificationAdapter: expected dim={self._embedding_dim}, got {emb.shape[0]}."
            )
        return emb.astype(np.float32)

    def offline_accuracy(self, result: Any, label_path: str) -> float:
        """Top-1 accuracy against a text label file (single integer class index)."""
        import torch
        with open(label_path) as f:
            true_class = int(f.read().strip())
        pred_class = int(torch.argmax(result, dim=-1).item())
        return 1.0 if pred_class == true_class else 0.0


def _attach_pooling_hook(model: Any, state: "_ClassifierHookState") -> None:
    """Attach a forward hook to the adaptive average pooling layer."""
    import torch.nn as nn

    target = None
    for m in model.modules():
        if isinstance(m, nn.AdaptiveAvgPool2d):
            target = m
            break
    if target is not None:
        target.register_forward_hook(state._hook_fn)


class _ClassifierHookState:
    def __init__(self) -> None:
        self._embedding = None

    def _hook_fn(self, module: Any, input: Any, output: Any) -> None:
        self._embedding = output.detach()


class _LoadedClassifier:
    def __init__(self, model: Any, transform: Any, hook_state: _ClassifierHookState) -> None:
        self.model = model
        self.transform = transform
        self.hook_state = hook_state

"""adapters/tasks/detection.py — Detection task adapter (YOLO family).

Implements the TaskAdapter contract for object detection models loaded via
ultralytics.  Heavy imports are deferred to method bodies so the module is
importable in test environments that don't have ultralytics installed.

R5-a: offline_accuracy raises NotImplementedError — use experiments/offline_eval.py.
R4:   extract_embedding uses a fixed forward hook registered once per loaded model.
      The hook fires on the SPPF layer (backbone output) resolved by class name,
      not a hardcoded layer index.  Falls back to the last backbone layer if SPPF
      is not found (prints architecture for debugging).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import TaskAdapter, register_task


@register_task("detection")
class DetectionAdapter(TaskAdapter):
    """YOLO-family detection adapter.

    Config keys consumed (from dataset config dict):
        embedding_dim (int): expected embedding dimension.  Validated on first extraction.
    """

    task = "detection"

    def __init__(self, config: dict) -> None:
        self._config = config
        self._embedding_dim: int | None = config.get("embedding_dim")

    # ── Private ───────────────────────────────────────────────────────────────

    def _attach_embedding_hook(self, model: Any) -> "_HookState":
        """Register a forward hook on the backbone output layer.

        Resolves the target layer by looking for an SPPF module (Spatial Pyramid
        Pooling Fast) or the last Convolution before the detection head.

        Returns a _HookState object that holds the captured feature maps and
        the hook handle (call handle.remove() to clean up).
        """
        import torch.nn as nn

        state = _HookState()

        # Ultralytics stores layers in model.model (a nn.Sequential-like container)
        backbone_candidates = []
        yolo_inner = getattr(model, "model", None)  # YOLO object's inner nn.Module
        if yolo_inner is not None:
            module_seq = getattr(yolo_inner, "model", None)  # nn.Sequential of layers
            if module_seq is not None:
                for m in module_seq:
                    backbone_candidates.append((type(m).__name__, m))

        # Prefer SPPF; fall back to the last C2f, then the last Conv
        target: Any = None
        for cls_name, mod in backbone_candidates:
            if cls_name == "SPPF":
                target = mod
                break
        if target is None:
            for cls_name, mod in reversed(backbone_candidates):
                if cls_name in ("C2f", "C3", "Conv"):
                    target = mod
                    break
        if target is None and backbone_candidates:
            target = backbone_candidates[-1][1]

        if target is None:
            arch_desc = [n for n, _ in backbone_candidates]
            raise RuntimeError(
                f"DetectionAdapter: could not locate a backbone output layer "
                f"in the YOLO model architecture. "
                f"Discovered layers: {arch_desc}. "
                f"File an issue with this output so the resolver can be updated."
            )

        handle = target.register_forward_hook(state._hook_fn)
        state._handle = handle
        return state

    # ── TaskAdapter interface ──────────────────────────────────────────────────

    def load_model(self, model_name: str, weights_path: str) -> "_LoadedDetection":
        """Load a YOLO model and attach the embedding hook."""
        from ultralytics import YOLO

        from core.device import get_device
        yolo = YOLO(weights_path)
        yolo.to(get_device())
        hook_state = self._attach_embedding_hook(yolo)
        return _LoadedDetection(yolo=yolo, hook_state=hook_state)

    def infer(self, model: "_LoadedDetection", input_path: str) -> Any:
        """Run YOLO detection on a single image.  Returns ultralytics Results list."""
        from core.device import yolo_device
        return model.yolo(input_path, verbose=False, device=yolo_device())

    def extract_proxy(self, result: Any) -> float:
        """Mean detection confidence.  Returns 0.0 on genuinely empty detections."""
        if result is None:
            raise ValueError("extract_proxy: result is None")
        try:
            boxes = result[0].boxes
            if boxes is None or len(boxes) == 0:
                return 0.0
            confs = boxes.conf
            if confs is None or len(confs) == 0:
                return 0.0
            return float(confs.mean().item())
        except (IndexError, AttributeError) as exc:
            raise ValueError(f"extract_proxy: malformed result — {exc}") from exc

    def extract_embedding(self, model: "_LoadedDetection", input_path: str) -> np.ndarray:
        """Run a forward pass and return the spatially-mean-pooled backbone embedding."""
        import torch

        # Run forward — the hook captures the feature map
        from core.device import yolo_device
        with torch.no_grad():
            _ = model.yolo(input_path, verbose=False, device=yolo_device())

        feat = model.hook_state._embedding
        if feat is None:
            raise RuntimeError(
                "DetectionAdapter.extract_embedding: hook did not fire. "
                "The backbone hook may not have been attached correctly."
            )

        # feat shape: (batch, C, H, W) or (batch, C) — pool spatial dims
        if feat.ndim == 4:
            emb = feat[0].mean(dim=(-2, -1)).cpu().float().numpy()
        elif feat.ndim == 3:
            emb = feat[0].mean(dim=-1).cpu().float().numpy()
        elif feat.ndim == 2:
            emb = feat[0].cpu().float().numpy()
        else:
            emb = feat.reshape(-1).cpu().float().numpy()

        if self._embedding_dim is not None and emb.shape[0] != self._embedding_dim:
            raise ValueError(
                f"DetectionAdapter.extract_embedding: expected dim={self._embedding_dim}, "
                f"got {emb.shape[0]}. "
                f"Update config['embedding_dim'] to match this model's backbone output."
            )
        return emb.astype(np.float32)

    def offline_accuracy(self, result: Any, label_path: str) -> float:
        raise NotImplementedError(
            "Per-image detection accuracy requires full dataset mAP computation. "
            "Use experiments/offline_eval.py for detection mAP@0.5 via Ultralytics val()."
        )


class _HookState:
    """Holds the captured feature map from the registered forward hook."""

    def __init__(self) -> None:
        self._embedding = None
        self._handle = None

    def _hook_fn(self, module: Any, input: Any, output: Any) -> None:
        self._embedding = output.detach()

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class _LoadedDetection:
    """Bundle of a loaded YOLO model and its embedding hook state."""

    def __init__(self, yolo: Any, hook_state: _HookState) -> None:
        self.yolo = yolo
        self.hook_state = hook_state

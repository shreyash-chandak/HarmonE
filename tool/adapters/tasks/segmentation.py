"""adapters/tasks/segmentation.py — Segmentation task adapter (SegFormer/DeepLab family).

R5-b corrections applied:
- SegformerImageProcessor constructed once in load_model, not per-frame
- Embedding pools spatial dims only, keeps channel vector
- mIoU upsamples logits to mask resolution (bilinear) before argmax
- ignore_index respected (config key, default 255)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import TaskAdapter, register_task


from adapters.loaders import _SEGFORMER_ARCH  # hardcoded arch dicts, zero HF calls


@register_task("segmentation")
class SegmentationAdapter(TaskAdapter):
    """SegFormer/DeepLab-family semantic segmentation adapter.

    Config keys consumed:
        num_classes (int):     Number of segmentation classes.  REQUIRED.
        ignore_index (int):    Label value to ignore in mIoU (default: 255).
        model_name_or_path (str): HuggingFace model ID or local path for SegFormer
                                  (default: "nvidia/segformer-b0-finetuned-ade-512-512").
        embedding_dim (int):   Expected embedding dim for drift store.
    """

    task = "segmentation"

    def __init__(self, config: dict) -> None:
        self._config = config
        self._num_classes: int = config.get("num_classes", 150)
        self._ignore_index: int = config.get("ignore_index", 255)
        self._hf_model: str = config.get(
            "model_name_or_path",
            "nvidia/segformer-b0-finetuned-ade-512-512",
        )
        self._embedding_dim: int | None = config.get("embedding_dim")

    def load_model(self, model_name: str, weights_path: str) -> "_LoadedSegformer":
        """Load SegFormer model + processor (processor built once, not per-frame).

        Accepts two forms of weights_path:
          - A raw .pt/.pth state-dict file: loads the HuggingFace backbone by
            inferring the variant (b0/b1/b2) from model_name, then applies the
            state dict.  This matches the layout in configs/datasets/acdc.json.
          - A HuggingFace model directory or hub id (no .pt/.pth suffix, or
            empty): passed to from_pretrained(), a real network fetch.

        A weights_path that LOOKS like a local checkpoint (.pt/.pth suffix) but
        doesn't resolve to an actual file raises FileNotFoundError instead of
        silently falling through to the from_pretrained() branch. That silent
        fallback used to use self._hf_model unconditionally — a single,
        non-variant-aware default ("nvidia/segformer-b0-finetuned-ade-512-512")
        regardless of whether model_name was b0/b1/b2 — so a missing
        segformer_b2_acdc.pt would silently fetch a b0/ADE20K checkpoint over
        the network instead, and ignore_mismatched_sizes=True would then reinit
        the classifier head rather than erroring on the shape mismatch: a wrong
        model, silently, with no clear signal anything was off. Every current
        caller (managed_system_cv/inference.py, offline_eval_segmentation.py's
        slow path, scripts/init_cv.py) already checks the file exists before
        calling load_model() at all, so this never fires today — but the
        function itself was not safe by construction, only safe because three
        separate call sites each happened to duplicate the same guard.
        """
        import torch
        from pathlib import Path
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        wp = Path(weights_path) if weights_path else None
        looks_like_local_checkpoint = wp is not None and wp.suffix.lower() in (".pt", ".pth")
        is_pt_file = looks_like_local_checkpoint and wp.is_file()

        if looks_like_local_checkpoint and not is_pt_file:
            raise FileNotFoundError(
                f"SegFormer weights not found at {wp} for model '{model_name}'. "
                "This path has a local checkpoint extension (.pt/.pth) but doesn't "
                "resolve to an actual file — refusing to silently substitute a "
                "different HuggingFace checkpoint (which would previously have "
                "fetched a fixed, non-variant-aware default regardless of which "
                "model was requested). Train/place the weights file, or pass an "
                "explicit HuggingFace model id/directory (no .pt/.pth suffix) if "
                "a fresh pretrained checkpoint is actually what you want."
            )

        if is_pt_file:
            # Infer backbone variant from model_name (segformer_b0 / _b1 / _b2)
            name_lower = model_name.lower()
            if "b2" in name_lower:
                hf_id = "nvidia/mit-b2"
            elif "b1" in name_lower:
                hf_id = "nvidia/mit-b1"
            else:
                hf_id = "nvidia/mit-b0"

            # Build config and processor from hardcoded arch dicts — no HF network calls.
            from transformers import SegformerConfig
            cfg = SegformerConfig(**_SEGFORMER_ARCH[hf_id], num_labels=self._num_classes)
            processor = SegformerImageProcessor(
                do_resize=True,
                size={"height": 512, "width": 512},
                do_normalize=True,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
                do_rescale=True,
                rescale_factor=1.0 / 255,
            )
            model = SegformerForSemanticSegmentation(cfg)
            state = torch.load(str(wp), map_location="cpu", weights_only=False)
            model.load_state_dict(state, strict=False)
        else:
            # HuggingFace model directory (or empty → default)
            hf_path = weights_path if weights_path else self._hf_model
            processor = SegformerImageProcessor.from_pretrained(
                hf_path, local_files_only=bool(weights_path),
            )
            model = SegformerForSemanticSegmentation.from_pretrained(
                hf_path,
                local_files_only=bool(weights_path),
                ignore_mismatched_sizes=True,
                num_labels=self._num_classes,
            )

        model.eval()

        # Register hook on encoder last hidden state
        hook_state = _SegHookState()
        _attach_encoder_hook(model, hook_state)

        return _LoadedSegformer(model=model, processor=processor, hook_state=hook_state)

    def infer(self, model: "_LoadedSegformer", input_path: str) -> Any:
        """Run segmentation.  Returns (logits, (H, W)) for the original image size."""
        import torch
        from PIL import Image

        img = Image.open(input_path).convert("RGB")
        orig_size = (img.height, img.width)
        inputs = model.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model.model(**inputs)
        return outputs.logits, orig_size

    def extract_proxy(self, result: Any) -> float:
        """Mean per-pixel maximum softmax as segmentation confidence proxy."""
        import torch
        logits, _ = result
        probs = torch.softmax(logits, dim=1)         # (1, C, H, W)
        return float(probs.max(dim=1).values.mean().item())

    def extract_embedding(self, model: "_LoadedSegformer", input_path: str) -> np.ndarray:
        """Spatially-pooled encoder last hidden state (channel vector, not scalar)."""
        import torch
        from PIL import Image

        img = Image.open(input_path).convert("RGB")
        inputs = model.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            _ = model.model(**inputs)

        feat = model.hook_state._embedding
        if feat is None:
            raise RuntimeError("SegmentationAdapter: embedding hook did not fire.")

        # feat: (batch, C, H, W) or (batch, seq, C) — pool spatial dims, keep C
        if feat.ndim == 4:
            emb = feat[0].mean(dim=(-2, -1)).cpu().float().numpy()  # (C,)
        elif feat.ndim == 3:
            emb = feat[0].mean(dim=-2).cpu().float().numpy()         # (C,)
        else:
            emb = feat.reshape(-1).cpu().float().numpy()

        if self._embedding_dim is not None and emb.shape[0] != self._embedding_dim:
            raise ValueError(
                f"SegmentationAdapter: expected dim={self._embedding_dim}, got {emb.shape[0]}."
            )
        return emb.astype(np.float32)

    def offline_accuracy(self, result: Any, label_path: str) -> float:
        """mIoU vs a ground-truth mask PNG.

        result can be:
          - (logits_tensor, orig_size)  — live inference output
          - np.ndarray uint8 (H, W)     — saved prediction mask from predictions/
        ignore_index pixels are excluded from IoU computation.
        """
        from PIL import Image

        mask_img = Image.open(label_path)
        gt = np.array(mask_img, dtype=np.int64)
        h, w = gt.shape[:2]

        if isinstance(result, np.ndarray):
            # Saved prediction mask — resize to GT resolution with nearest-neighbour
            pred_raw = result.astype(np.int64)
            if pred_raw.shape != (h, w):
                pred_img = Image.fromarray(result.astype(np.uint8), mode="L")
                pred_img = pred_img.resize((w, h), resample=Image.NEAREST)
                pred_raw = np.array(pred_img, dtype=np.int64)
            pred = pred_raw
        else:
            import torch
            import torch.nn.functional as F
            logits, _ = result
            logits_up = F.interpolate(
                logits.float(), size=(h, w), mode="bilinear", align_corners=False
            )
            pred = logits_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)

        # Compute per-class IoU ignoring ignore_index
        iou_list = []
        for cls in range(self._num_classes):
            valid = gt != self._ignore_index
            tp = int(((pred == cls) & (gt == cls) & valid).sum())
            fp = int(((pred == cls) & (gt != cls) & valid).sum())
            fn = int(((pred != cls) & (gt == cls) & valid).sum())
            denom = tp + fp + fn
            if denom > 0:
                iou_list.append(tp / denom)
        return float(np.mean(iou_list)) if iou_list else 0.0


def _attach_encoder_hook(model: Any, state: "_SegHookState") -> None:
    """Attach hook to the encoder's final norm layer (or last encoder block)."""
    # SegFormer has model.segformer.encoder.layer_norm; try that first
    target = None
    segformer_core = getattr(model, "segformer", None)
    if segformer_core is not None:
        encoder = getattr(segformer_core, "encoder", None)
        if encoder is not None:
            # last layer norm (one per stage)
            layer_norms = getattr(encoder, "layer_norm", None)
            if layer_norms is not None and hasattr(layer_norms, "__len__"):
                target = layer_norms[-1]

    if target is None:
        # Fallback: last module with "norm" in class name
        for m in model.modules():
            if "norm" in type(m).__name__.lower():
                target = m

    if target is not None:
        target.register_forward_hook(state._hook_fn)


class _SegHookState:
    def __init__(self) -> None:
        self._embedding = None

    def _hook_fn(self, module: Any, input: Any, output: Any) -> None:
        if isinstance(output, (list, tuple)):
            self._embedding = output[0].detach()
        else:
            self._embedding = output.detach()


class _LoadedSegformer:
    def __init__(self, model: Any, processor: Any, hook_state: _SegHookState) -> None:
        self.model = model
        self.processor = processor
        self.hook_state = hook_state

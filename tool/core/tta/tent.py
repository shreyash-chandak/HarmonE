"""
core/tta/tent.py — Tent: Fully Test-Time Adaptation by Entropy Minimization.

Wang, Shelhamer, Liu, Olshausen, Darrell. ICLR 2021. arXiv:2006.10726.
See context/idea.md for the full incorporation writeup (why this is being
added, which of our datasets it targets, and how it relates to Augur).

What this module does (paper Section 3, Algorithm in Section 3.3):
    1. `configure_model()` puts every BatchNorm layer into "estimate batch
       statistics fresh every forward pass" mode (no running average carried
       from training), and freezes every parameter EXCEPT each BatchNorm
       layer's affine scale/shift (gamma, beta) — typically <1% of the model.
    2. Each adaptation step: run a batch through the model, compute the
       Shannon entropy of the softmax predictions (test entropy — no labels
       used), backpropagate, and update ONLY the unfrozen affine parameters.
    3. No source data, no altered training, no auxiliary/proxy task — the
       model's own prediction confidence is the sole training signal.

Model requirement: the model must contain torch.nn.BatchNorm{1,2,3}d layers
(true of every torchvision classifier this codebase uses — EfficientNet-B0,
ResNet-50, ResNet-101). GroupNorm/LayerNorm-only architectures (e.g. the
ConvNeXt family) are NOT supported by this implementation; that would need a
different modulation target (see the paper's Section 6 "Parameters"
discussion — out of scope here).

Standalone module, not wired into the CV retrain-tactic dispatch (see
context/idea.md and context/DECISIONS_PENDING.md DP19 for why: that dispatch
is presently detection/YOLO-specific). Usable directly:

    from core.tta.tent import configure_model, collect_params, Tent
    import torch

    model = configure_model(loaded_classifier.model)
    params, names = collect_params(model)
    optimizer = torch.optim.Adam(params, lr=1e-3)
    tented = Tent(model, optimizer, steps=1, episodic=False)

    for batch in image_batches:            # batch: (N, C, H, W) tensor
        logits = tented(batch)             # adapts online, one step per call
        ...

Or via the convenience wrapper `adapt_classifier()` below, which handles
image loading through a ClassificationAdapter-style `_LoadedClassifier`
(model + transform) directly from a list of file paths.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# ── Core algorithm (paper Section 3.2-3.3) ─────────────────────────────────

def configure_model(model: nn.Module) -> nn.Module:
    """Configure model for use with Tent: train mode, freeze all but BN affine
    params, force BN to estimate statistics fresh from each batch (no running
    average from the source/training data)."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.requires_grad_(True)
            # Force use of batch statistics in train and eval modes, and
            # discard the source-data running statistics (paper: "The
            # normalization statistics {mu, sigma} from the source data are
            # discarded").
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def collect_params(model: nn.Module) -> tuple[list[nn.Parameter], list[str]]:
    """Collect the affine scale/shift parameters of every BatchNorm layer —
    the only parameters Tent optimizes."""
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for p_name in ("weight", "bias"):
                p = getattr(m, p_name, None)
                if p is not None:
                    params.append(p)
                    names.append(f"{nm}.{p_name}")
    if not params:
        raise ValueError(
            "core.tta.tent: no BatchNorm layers found in this model. "
            "Tent requires BatchNorm affine parameters to adapt; "
            "GroupNorm/LayerNorm-only architectures are not supported."
        )
    return params, names


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample Shannon entropy of the softmax distribution, in nats."""
    log_probs = logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


@torch.enable_grad()
def forward_and_adapt(
    x: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer
) -> torch.Tensor:
    """One Tent adaptation step. Returns THIS batch's logits (computed before
    the parameter update — the paper's default efficiency scheme: 'the
    transformation update follows the prediction for the current batch, and
    so it only affects the next batch')."""
    logits = model(x)
    loss = softmax_entropy(logits).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return logits


def copy_model_and_optimizer(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[dict, dict]:
    """Snapshot state for episodic reset."""
    return (
        {k: v.detach().clone() for k, v in model.state_dict().items()},
        optimizer.state_dict(),
    )


def load_model_and_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_state: dict,
    optimizer_state: dict,
) -> None:
    """Restore a snapshot taken by copy_model_and_optimizer (episodic reset)."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Wraps a model already prepared with configure_model() and an optimizer
    already restricted to collect_params(model). Call repeatedly on batches
    of test data. Adaptation persists across calls unless episodic=True, in
    which case the model and optimizer reset to their initial state before
    every call (matches the paper's single-image / single-batch episodic
    mode used for the semantic segmentation experiment).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        steps: int = 1,
        episodic: bool = False,
    ) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("Tent requires >= 1 adaptation step per forward call.")
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self._model_state, self._optimizer_state = copy_model_and_optimizer(model, optimizer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    def reset(self) -> None:
        if self._model_state is None or self._optimizer_state is None:
            raise RuntimeError("Tent: cannot reset, no snapshot was captured at init.")
        load_model_and_optimizer(self.model, self.optimizer, self._model_state, self._optimizer_state)


# ── Convenience wrapper for adapters/tasks/classification.py-style models ──

def adapt_classifier(
    loaded_classifier: Any,
    image_paths: list[str],
    *,
    lr: float = 1e-3,
    batch_size: int = 32,
    steps: int = 1,
    episodic: bool = False,
    device: str = "cpu",
) -> tuple[Tent, list[int]]:
    """Run Tent adaptation over a list of image files using a
    ClassificationAdapter-style `_LoadedClassifier` (has `.model` and
    `.transform`; see adapters/tasks/classification.py).

    Mutates loaded_classifier.model in place (its BatchNorm affine params are
    updated). Returns (Tent wrapper, predicted class index per image, in the
    same order as image_paths — computed from the pre-update logits of each
    image's batch, so no extra forward pass is spent just for reporting).

    This is intentionally NOT wired into managed_system_cv's retrain-tactic
    dispatch (core/DECISIONS_PENDING.md DP19) — it is a standalone utility
    callable from an experiment script, a notebook, or a future retrain
    tactic module.
    """
    from PIL import Image

    model = configure_model(loaded_classifier.model).to(device)
    params, _ = collect_params(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    tented = Tent(model, optimizer, steps=steps, episodic=episodic)

    preds: list[int] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(loaded_classifier.transform(img))
        x = torch.stack(tensors).to(device)
        logits = tented(x)
        preds.extend(int(i) for i in logits.argmax(dim=-1).tolist())

    return tented, preds

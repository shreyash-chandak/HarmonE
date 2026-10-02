"""
core/device.py — single source of truth for where CV models run (2026-10-03).

Before this module, no CV code path in the harnesses ever moved a model or a
tensor off the CPU: SegFormer (acdc) and the torchvision classifiers
(imagenet/imagenet_c/iwildcam) always ran on CPU, and YOLO (bdd100k) only
reached a GPU by ultralytics' own auto-selection when no device was passed.
All CV work — inference, initial training, pseudo-label fine-tuning, TENT,
VMR restore — now goes through `get_device()`.

Configuration: the dataset config key "device":
  "cuda"  → REQUIRE a GPU. Startup fails with a clear error if CUDA is not
            available, instead of silently running on CPU. (All CV dataset
            configs set this.)
  "auto"  → GPU if available, else CPU (code default when the key is absent).
  "cpu"   → force CPU.
  "cuda:N"→ a specific GPU.

The resolved device is process-wide: each process (single-threaded harness,
concurrent inference.py, concurrent manage.py, run_concurrent.py's bootstrap)
calls `configure_device(dataset_config)` once at startup. If nothing calls it,
`get_device()` lazily resolves "auto".

Regression (LSTM/sklearn) is deliberately left on CPU: those models are tiny
and their code paths do not consult this module.

ultralytics note: `yolo_device()` returns "cuda:0"-style strings, never a bare
"cuda" — ultralytics' select_device() strips the "cuda:" prefix and treats the
remainder as a CUDA_VISIBLE_DEVICES index, so a bare "cuda" would be parsed as
an invalid device index.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DEVICE = None  # torch.device, resolved once per process


def resolve_device(requested: str | None = "auto"):
    """Turn a config value into a torch.device. Raises RuntimeError if CUDA is
    explicitly requested but unavailable."""
    import torch

    req = (requested or "auto").strip().lower()
    if req == "cpu":
        return torch.device("cpu")
    if req == "auto":
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Config requests device='{requested}' but torch.cuda.is_available() is False "
                f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}). Refusing to "
                "fall back to CPU silently — fix the CUDA install/driver, or set "
                "\"device\": \"cpu\" (or \"auto\") in the dataset config explicitly."
            )
        idx = req.split(":", 1)[1] if ":" in req else "0"
        dev = torch.device(f"cuda:{int(idx)}")
        if dev.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Config requests device='{requested}' but only {torch.cuda.device_count()} "
                "CUDA device(s) are visible."
            )
        return dev
    raise ValueError(f"Unknown device setting '{requested}' (use cuda / cuda:N / cpu / auto).")


def configure_device(config: dict | None) -> "object":
    """Resolve and fix the process-wide device from a dataset config. Logs the
    choice (and the GPU name) once. Returns the torch.device."""
    global _DEVICE
    requested = (config or {}).get("device", "auto")
    _DEVICE = resolve_device(requested)
    logger.info("CV device: %s (requested '%s')%s", _DEVICE, requested, _gpu_suffix(_DEVICE))
    return _DEVICE


def get_device():
    """The process-wide CV device (lazily 'auto' if never configured)."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = resolve_device("auto")
    return _DEVICE


def yolo_device() -> str:
    """Device string for ultralytics' device= argument ("cuda:0" or "cpu")."""
    dev = get_device()
    return "cpu" if dev.type == "cpu" else f"cuda:{dev.index or 0}"


def device_info() -> dict:
    """For run manifests."""
    dev = get_device()
    info = {"device": str(dev)}
    if dev.type == "cuda":
        import torch
        info["gpu_name"] = torch.cuda.get_device_name(dev)
    return info


def _gpu_suffix(dev) -> str:
    if dev.type != "cuda":
        return ""
    import torch
    return f" — {torch.cuda.get_device_name(dev)}"

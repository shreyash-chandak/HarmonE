"""adapters/tasks/base.py — TaskAdapter interface and registry.

TaskAdapter answers "how do I run a model on one input and read its outputs".
It is stateless w.r.t. the MAPE loop: no thresholds, no knowledge-file access.
It composes with DatasetAdapter (what data, in what order) inside the CV
inference loop.

Registry: get_task_adapter(task_name, config) returns the correct adapter
for the given task family, following the same pattern as the planner registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TaskAdapter(ABC):
    """Model-interaction contract for one CV task family.

    Constructed from the dataset config dict only.  Stateless w.r.t. the
    MAPE loop: no thresholds, no knowledge-file access.
    """

    task: str  # "detection" | "classification" | "segmentation"

    @abstractmethod
    def load_model(self, model_name: str, weights_path: str) -> Any:
        """Load model weights and return an opaque handle.

        Caching (avoiding reload on every call) is the CALLER's responsibility.
        Heavy imports (torch, ultralytics) must be deferred to this method body
        so the adapter is importable without those packages installed.
        """

    @abstractmethod
    def infer(self, model: Any, input_path: str) -> Any:
        """Run one input through the model.  No side effects; pure function."""

    @abstractmethod
    def extract_proxy(self, result: Any) -> float:
        """Derive the A_i accuracy proxy from a raw inference result.

        Returns a float in [0, 1].  Return 0.0 ONLY for genuinely empty results
        (e.g. zero detections) — never as an error swallow.  Raise on malformed
        input so silent failures are caught at the boundary.
        """

    @abstractmethod
    def extract_embedding(self, model: Any, input_path: str) -> np.ndarray:
        """Extract a 1-D float32 embedding vector from the designated embedding model.

        The caller is responsible for passing the embedding_model handle (R4:
        the embedding model is fixed per dataset config, independent of the
        model currently serving inference).

        Raises ValueError if the output dim does not match config["embedding_dim"].
        """

    @abstractmethod
    def offline_accuracy(self, result: Any, label_path: str) -> float:
        """Compute a true offline accuracy metric given a ground-truth label file.

        Implementations that cannot compute this trivially should raise:
            NotImplementedError("use experiments/offline_eval.py for <task> mAP")
        This is not a stub — it documents the correct routing.
        """


# ── Registry ──────────────────────────────────────────────────────────────────

_TASK_REGISTRY: dict[str, type[TaskAdapter]] = {}


def register_task(name: str):
    """Decorator to register a TaskAdapter subclass under a task name."""
    def _dec(cls: type[TaskAdapter]) -> type[TaskAdapter]:
        _TASK_REGISTRY[name] = cls
        return cls
    return _dec


def get_task_adapter(task: str, config: dict) -> TaskAdapter:
    """Construct and return the TaskAdapter for the given task family.

    Args:
        task:   "detection" | "classification" | "segmentation"
        config: full dataset config dict (passed through to the adapter)

    Raises:
        KeyError if the task is not registered.
    """
    # Trigger registration side-effects by importing concrete modules
    if task not in _TASK_REGISTRY:
        _import_task_modules()
    if task not in _TASK_REGISTRY:
        raise KeyError(
            f"Unknown task '{task}'. "
            f"Available tasks: {sorted(_TASK_REGISTRY)}"
        )
    return _TASK_REGISTRY[task](config)


def _import_task_modules() -> None:
    """Import concrete adapter modules to trigger @register_task decorators."""
    import importlib
    for mod in ("adapters.tasks.detection",
                "adapters.tasks.classification",
                "adapters.tasks.segmentation"):
        try:
            importlib.import_module(mod)
        except ImportError:
            pass

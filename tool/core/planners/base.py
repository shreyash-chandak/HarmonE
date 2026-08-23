"""
core/planners/base.py — Planner interface and data types.

All planners receive a PlanningContext and return a PlanDecision. They are
pure (no file I/O); the managed system's plan.py wrapper handles reading files
and persisting the decision.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanningContext:
    """All information a planner needs to make a decision."""
    violation: str | None               # "score" | "energy" | "drift" | None
    ema_scores: dict[str, float]        # collapsed EMA score per model (legacy)
    ema_accuracy: dict[str, float]      # separated accuracy EMA per model
    ema_energy: dict[str, float]        # separated energy EMA per model
    current_model: str
    available_models: list[str]
    thresholds: dict                    # full thresholds.json contents
    drift_result: dict | None           # analyse_drift() return contract (B2)
    history: Any = None                 # future: telemetry accessor for bandit
    current_step: int = 0               # stream step count at time of planning (for PRT)


@dataclass
class PlanDecision:
    """A planner's resolved action."""
    action: str                         # "switch" | "replace" | "retrain" | "noop"
    model: str | None = None            # target model name for "switch"
    version_path: str | None = None     # path to VMR weights for "replace"
    reason: str = ""                    # logged to event log verbatim
    metadata: dict | None = None        # planner-specific data (e.g. bandit pending record)


class Planner(ABC):
    """Base class for all planning strategies."""

    name: str  # must be set on each subclass

    @abstractmethod
    def plan(self, ctx: PlanningContext) -> PlanDecision:
        """Compute a planning decision from the given context.

        Must be pure: no file I/O, no randomness unless the strategy explicitly
        requires it (document it). Raises ValueError if ctx is structurally invalid.
        """


# Registry: maps planner name (from thresholds.json["planner"]) → class.
# Import each planner module below to populate; guard against import errors.
REGISTRY: dict[str, type[Planner]] = {}


def _register():
    """Lazy-populate REGISTRY from sibling modules.

    Python's import system is idempotent — re-importing an already-loaded
    module is a no-op, so this is safe to call multiple times.
    """
    from . import naive, random_switch, greedy_switch, harmone_original, violation_aware, pareto, bandit, naive_prt, random_switch_prt  # noqa: F401


def get_planner(name: str) -> Planner:
    """Instantiate a planner by its registered name.

    Raises KeyError if the name is not registered; ImportError if the module
    failed to load (e.g. bandit not yet implemented).
    """
    _register()
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown planner '{name}'. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]()

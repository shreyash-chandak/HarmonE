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
    # Phase 9 additions (all with defaults for backward compatibility)
    current_energy_threshold: float = 0.6  # live adaptive threshold (Phase 1.2)
    last_observed_step: dict = field(default_factory=dict)  # step each model was last active (Phase 1.4)
    staleness_window: int = 500        # steps after which an EMA is stale (Phase 1.4)
    # Audit A5: model -> has it been monitored yet. Empty = not supplied
    # (legacy callers); is_observed() then falls back to the 0.5 seed check.
    observed: dict = field(default_factory=dict)
    # This cycle's monitor output: accuracy, normalized_energy, kl_div and
    # served_model (the model that served the monitored batch). Used by the
    # bandit to learn from every batch (audit B1/B5).
    telemetry: dict = field(default_factory=dict)
    # S5 run-level accuracy credit B (mape_info["s5_credit"], updated every
    # cycle in Monitor; core/planners/credit.py). 0.0 when not tracked.
    accuracy_credit: float = 0.0
    # A PRT retrain was skipped because another retrain was running; it runs on
    # the next cycle that can (mape_info["prt_pending"], 2026-10-31).
    prt_pending: bool = False


def prt_due(ctx: "PlanningContext", default_interval: int) -> bool:
    """Whether this cycle crossed a multiple of prt_interval (2026-10-14).

    Planning happens once per monitor_interval rows, so with a cycle that does
    not divide prt_interval (260 vs 3200) `step % prt_interval == 0` almost
    never holds (first at lcm = 41 600). This fires on the first cycle at or
    past each multiple instead; identical to the modulo test whenever
    monitor_interval divides prt_interval (e.g. 50 vs 3200).
    """
    if ctx.prt_pending:
        return True
    p = int(ctx.thresholds.get("prt_interval", default_interval))
    mi = int(ctx.thresholds.get("monitor_interval", 50))
    s = ctx.current_step
    return s > 0 and s // p > max(s - mi, 0) // p


COST_RANK = {"light": 0, "medium": 1, "heavy": 2}


def cost_rank(thresholds: dict, model: str) -> int:
    """Configured cost class of a model as a rank (unknown class = medium)."""
    return COST_RANK.get(thresholds.get("models", {}).get(model, {}).get("cost_class", "medium"), 1)


def is_observed(ctx: "PlanningContext", model: str) -> bool:
    """Whether `model` has real EMA estimates (audit A5).

    Uses ctx.observed when the harness supplies it; otherwise falls back to
    the legacy test "EMAs are not at the 0.5 initial seed".
    """
    if ctx.observed:
        return bool(ctx.observed.get(model, False))
    acc = ctx.ema_accuracy.get(model, 0.5)
    eng = ctx.ema_energy.get(model, 0.5)
    return not (abs(acc - 0.5) < 1e-9 or abs(eng - 0.5) < 1e-9)


@dataclass
class PlanDecision:
    """A planner's resolved action."""
    action: str                         # "switch" | "replace" | "retrain" | "noop"
    model: str | None = None            # target model name for "switch"
    version_path: str | None = None     # path to VMR weights for "replace"
    reason: str = ""                    # logged to event log verbatim
    metadata: dict | None = None        # planner-specific data (e.g. bandit pending record)
    # Violation the planner was actually given (after the recovery hold and
    # drift folding), set by the harness's _plan(); logged next to the raw one.
    effective_violation: str | None = None
    # What execution did with it, set by the concurrent harness's execute_decision:
    # "executed", "started" (background), "skipped_busy", "failed"; None for a noop.
    executed: str | None = None


class Planner(ABC):
    """Base class for all planning strategies."""

    name: str  # must be set on each subclass
    # False for the switch-only baselines: the original HarmonE's "switch" and
    # "switch+retrain" approaches never start its drift thread, so they never
    # react to drift (2026-10-12). The harness then withholds drift from them.
    reacts_to_drift: bool = True

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
    from . import naive, random_switch, switch, greedy_switch, harmone_original, violation_aware, violation_aware_gated, pareto, bandit, naive_prt, random_switch_prt  # noqa: F401


def planner_reacts_to_drift(name: str) -> bool:
    """Class attribute lookup, no instance (BanditPlanner needs its bandit set first)."""
    _register()
    return bool(REGISTRY[name].reacts_to_drift)


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

"""PRT fires on the first planning cycle at or past each multiple of prt_interval,
also when monitor_interval does not divide it (2026-10-14: 260 vs 3200 never
fired before step 41 600)."""

from core.planners.base import PlanningContext
from core.planners.naive_prt import NaivePRTPlanner
from core.planners.random_switch_prt import RandomSwitchPRTPlanner

MODELS = ["ridge", "svr", "lstm"]


def _ctx(step, mi, p=3200):
    return PlanningContext(
        violation=None, ema_scores={m: 0.5 for m in MODELS}, ema_accuracy={m: 0.5 for m in MODELS},
        ema_energy={m: 0.5 for m in MODELS}, current_model="ridge", available_models=MODELS,
        thresholds={"prt_interval": p, "monitor_interval": mi}, drift_result=None, current_step=step,
    )


def _fires(planner, mi, n_rows):
    return [s for s in range(mi, n_rows + 1, mi) if planner.plan(_ctx(s, mi)).action == "retrain"]


def test_cycle_not_dividing_interval():
    for planner in (NaivePRTPlanner(), RandomSwitchPRTPlanner()):
        fired = _fires(planner, 260, 23313)
        assert fired == [3380, 6500, 9620, 13000, 16120, 19240, 22620]


def test_dividing_cycle_unchanged():
    assert _fires(NaivePRTPlanner(), 50, 23313) == list(range(3200, 23313, 3200))

"""
Tests for the 2026-09-28 fixes borrowed from the single-threaded harness and
the original HarmonE (context/audit_2026-09-28.md, "borrowed fixes"):

  - S4 per-cycle exploration (original plan_mape)             — audit A2
  - S5/S6 explore_prob probe of the stalest model             — audit A2/A7
  - switch hold in the shared _plan (original recovery_cycles) — audit A2/A3/D1
  - VMR raw-vs-raw matching (original versionedMR data.csv)   — audit E2
  - concurrent drift reference: overflow bins, re-base after
    adaptation, cooldown (original rolling ref + sleep(400))  — audit E1/E2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from core.planners.base import PlanningContext, PlanDecision
from core.planners.harmone_original import HarmonEOriginalPlanner
from core.planners.violation_aware import ViolationAwarePlanner
from core.planners.pareto import ParetoPlanner
from core.planners.exploration import stalest_model
from core.vmr import VMR
from experiments.run_experiment import _plan, _initial_mape_info, _analyse_drift

_TOOL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TOOL_DIR / "concurrent_harness" / "mape"))
from drift_ref import PerModelDriftReference, BoundDetector  # noqa: E402

MODELS = ["lstm", "ridge", "svr"]
EMA_S = {"lstm": 0.9, "ridge": 0.4, "svr": 0.5}
EMA_A = {"lstm": 0.95, "ridge": 0.02, "svr": 0.03}
EMA_E = {"lstm": 0.6, "ridge": 0.05, "svr": 0.1}


def _ctx(violation=None, thr=None, current="lstm", last_seen=None):
    base = {"alpha": 0.0, "min_score": 0.78, "min_accuracy": 0.78, "explore_prob": 0.0}
    base.update(thr or {})
    return PlanningContext(
        violation=violation, ema_scores=EMA_S, ema_accuracy=EMA_A, ema_energy=EMA_E,
        current_model=current, available_models=MODELS, thresholds=base,
        drift_result=None, current_step=1000,
        last_observed_step=last_seen if last_seen is not None else {"lstm": 1000, "ridge": 50, "svr": 400},
    )


class _Const:
    """Planner stub returning a fixed decision."""
    def __init__(self, d): self.d = d
    def plan(self, ctx): return self.d


# ── S4 per-cycle exploration ────────────────────────────────────────────────

class TestS4Exploration:
    def test_explores_without_violation(self):
        seen = {HarmonEOriginalPlanner().plan(_ctx(thr={"alpha": 1.0})).model for _ in range(60)}
        assert seen & {"ridge", "svr"}, "alpha=1 must explore even with no violation"

    def test_no_exploration_alpha_zero(self):
        assert HarmonEOriginalPlanner().plan(_ctx(thr={"alpha": 0.0})).action == "noop"

    def test_drift_takes_priority_over_exploration(self):
        ctx = _ctx("drift", thr={"alpha": 1.0})
        ctx.drift_result = {"drift_detected": True, "action": "retrain", "version": None}
        assert HarmonEOriginalPlanner().plan(ctx).action == "retrain"


# ── S5/S6 explore_prob probe ──────────────────────────────────────────────────

class TestStalestProbe:
    def test_stalest_prefers_never_observed(self):
        ctx = _ctx(last_seen={"lstm": 1000, "ridge": 50})
        assert stalest_model(ctx) == "svr"

    def test_stalest_oldest_observation(self):
        assert stalest_model(_ctx()) == "ridge"

    @pytest.mark.parametrize("planner_cls", [ViolationAwarePlanner, ParetoPlanner])
    def test_probe_switches_to_stalest(self, planner_cls):
        d = planner_cls().plan(_ctx(thr={"explore_prob": 1.0}))
        assert d.action == "switch" and d.model == "ridge"
        assert "exploration probe" in d.reason

    @pytest.mark.parametrize("planner_cls", [ViolationAwarePlanner, ParetoPlanner])
    def test_probe_off_by_default(self, planner_cls):
        assert planner_cls().plan(_ctx()).action == "noop"

    def test_s5_frozen_estimate_blocks_switch_without_probe(self):
        # Reproduces the audit A3 pattern: ridge/svr frozen near 0 accuracy,
        # so on an energy violation S5 keeps lstm. The probe is the way out.
        d = ViolationAwarePlanner().plan(_ctx("energy"))
        assert d.action == "noop"


# ── switch hold in shared _plan ──────────────────────────────────────────────

class TestSwitchHold:
    def _info(self, since, switches):
        info = _initial_mape_info({m: None for m in MODELS})
        info["steps_since_last_switch"] = since
        info["event_counters"]["model_switches"] = switches
        return info

    def _run(self, since, switches, hold=3, action="switch", model="ridge"):
        planner = _Const(PlanDecision(action=action, model=model, reason="stub"))
        return _plan("score", {"drift_detected": False}, "lstm", MODELS,
                     self._info(since, switches), {"switch_hold_cycles": hold}, planner)

    def test_switch_suppressed_within_hold(self):
        d = self._run(since=2, switches=1)
        assert d.action == "noop" and "switch hold" in d.reason

    def test_hold_covers_exactly_n_cycles(self):
        assert self._run(since=3, switches=1).action == "noop"
        assert self._run(since=4, switches=1).action == "switch"

    def test_no_hold_before_first_switch(self):
        assert self._run(since=1, switches=0).action == "switch"

    def test_hold_disabled_by_default(self):
        assert self._run(since=1, switches=5, hold=0).action == "switch"

    def test_retrain_never_suppressed(self):
        assert self._run(since=1, switches=5, action="retrain", model=None).action == "retrain"


# ── VMR raw-vs-raw matching ───────────────────────────────────────────────────

class TestVMRRawMatching:
    def _store(self, vmr, tmp_path, name, raw, data):
        w = tmp_path / f"{name}.pkl"
        w.write_bytes(b"x")
        return vmr.store("ridge", str(w), {"type": "histogram", "data": data, "raw": raw}, tag=name)

    def test_raw_match_sees_out_of_range_values(self, tmp_path):
        rng = np.random.default_rng(0)
        vmr = VMR(str(tmp_path / "vmr"))
        in_range = rng.normal(100, 5, 1200).tolist()
        drifted = rng.normal(300, 5, 1200).tolist()  # far beyond a [80,120] training range
        # Binned "data" identical (all-zero, as fixed edges would drop drifted
        # values) — only the raw windows can tell the versions apart.
        self._store(vmr, tmp_path, "near_training", in_range, [0] * 50)
        self._store(vmr, tmp_path, "drifted", drifted, [0] * 50)
        cur = rng.normal(300, 5, 1200).tolist()
        best = vmr.best_match("ridge", {"type": "histogram", "data": [0] * 50, "raw": cur},
                              strategy="closest_distribution", threshold=0.5)
        assert best is not None and best.tag == "drifted"

    def test_raw_match_respects_threshold(self, tmp_path):
        rng = np.random.default_rng(1)
        vmr = VMR(str(tmp_path / "vmr"))
        self._store(vmr, tmp_path, "old", rng.normal(100, 5, 1200).tolist(), [1] * 50)
        cur = rng.normal(300, 5, 1200).tolist()
        assert vmr.best_match("ridge", {"type": "histogram", "data": [1] * 50, "raw": cur},
                              strategy="closest_distribution", threshold=0.5) is None


# ── concurrent drift reference ────────────────────────────────────────────────

@pytest.fixture()
def ref_file(tmp_path):
    rng = np.random.default_rng(2)
    train = rng.normal(100, 10, 5000)
    hist, edges = np.histogram(train, bins=50)
    p = tmp_path / "reference_distribution.json"
    p.write_text(json.dumps({"histogram": hist.tolist(), "bin_edges": edges.tolist()}))
    return str(p)


class TestPerModelDriftReference:
    def test_out_of_range_values_raise_kl(self, ref_file):
        ref = PerModelDriftReference(ref_file, tau_drift=0.5, window_size=1200, cooldown_rows=1200)
        rng = np.random.default_rng(3)
        beyond = rng.normal(1000, 10, 1200)  # entirely above the training max
        r = ref.detect("lstm", beyond, current_step=5000)
        assert r["drift_detected"] and r["kl_div"] > 0.5

    def test_in_distribution_no_drift(self, ref_file):
        ref = PerModelDriftReference(ref_file, tau_drift=0.5, window_size=1200, cooldown_rows=1200)
        rng = np.random.default_rng(4)
        assert not ref.detect("lstm", rng.normal(100, 10, 1200), current_step=5000)["drift_detected"]

    def test_rebase_and_cooldown_after_adaptation(self, ref_file):
        ref = PerModelDriftReference(ref_file, tau_drift=0.5, window_size=1200, cooldown_rows=600)
        rng = np.random.default_rng(5)
        shifted = rng.normal(160, 10, 1200)
        assert ref.detect("lstm", shifted, 3000)["drift_detected"]
        assert ref.note_adaptation("lstm", 3000, shifted, 3000)
        # Within cooldown: suppressed even if the data moves again.
        moved = rng.normal(40, 10, 1200)
        r = ref.detect("lstm", moved, 3300)
        assert r["cooldown"] and not r["drift_detected"]
        # After cooldown: compared against the RE-BASED reference.
        assert not ref.detect("lstm", rng.normal(160, 10, 1200), 3700)["drift_detected"]
        assert ref.detect("lstm", moved, 3700)["drift_detected"]
        # Other models keep the training reference.
        assert ref.detect("ridge", rng.normal(160, 10, 1200), 3700)["drift_detected"]

    def test_note_adaptation_idempotent(self, ref_file):
        ref = PerModelDriftReference(ref_file, tau_drift=0.5, window_size=1200, cooldown_rows=600)
        w = np.random.default_rng(6).normal(100, 10, 1200)
        assert ref.note_adaptation("lstm", 3000, w, 3000)
        assert not ref.note_adaptation("lstm", 3000, w, 3050)

    def test_bound_detector_works_with_analyse_drift(self, ref_file, tmp_path):
        ref = PerModelDriftReference(ref_file, tau_drift=0.5, window_size=1200, cooldown_rows=600)
        beyond = np.random.default_rng(7).normal(1000, 10, 1200).tolist()
        out = _analyse_drift(beyond, BoundDetector(ref, "lstm", 5000), {"tau_drift": 0.5},
                             vmr=VMR(str(tmp_path / "vmr")), current_model="lstm",
                             current_distribution={"type": "histogram", "data": [0] * 50, "raw": beyond})
        assert out["drift_detected"] and out["action"] == "retrain"

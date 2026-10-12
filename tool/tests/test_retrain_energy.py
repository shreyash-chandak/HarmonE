"""Background retrain energy (2026-10-13): the retrain thread's Pkg0Meter
reading is added to event_counters.retrain_energy_uJ by the plan thread."""
import sys
import types
from pathlib import Path

_TOOL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TOOL))
sys.path.insert(0, str(_TOOL / "concurrent_harness"))
sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))  # Linux-only import

sys.path.insert(0, str(_TOOL / "concurrent_harness" / "mape"))
import execute  # noqa: E402


class _Store:
    def __init__(self):
        self.d = {"event_counters": {"retrains": 0, "retrain_skipped": 0}}

    def update(self, fn):
        fn(self.d)


def test_retrain_energy_accumulates(monkeypatch):
    monkeypatch.setattr(execute, "note_adaptation", lambda *a, **k: None)
    store = _Store()
    execute._bg_results.put(("ridge", True, None, "prt", 1500.0))
    execute._bg_results.put(("ridge", False, None, "prt", None))   # unreadable RAPL: not added
    execute._bg_results.put(("svr", True, None, "prt", 500.0))
    execute.apply_background_results(store, 3200)
    ec = store.d["event_counters"]
    assert ec["retrains"] == 2 and ec["retrain_skipped"] == 1
    assert ec["retrain_energy_uJ"] == 2000.0

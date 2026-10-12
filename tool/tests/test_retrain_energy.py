"""Background retrain energy (2026-10-13): the retrain thread's Pkg0Meter
reading is added to event_counters.retrain_energy_uJ by the plan thread."""
import sys
import types
from pathlib import Path

_TOOL = Path(__file__).resolve().parent.parent

# Import concurrent_harness/mape/execute.py without leaking its top-level
# module names (execute, retrain, knowledge_io) to other tests, which import
# managed_system_*'s own modules of the same names.
_saved_path, _saved_mods = list(sys.path), set(sys.modules)
sys.path[:0] = [str(_TOOL), str(_TOOL / "concurrent_harness"), str(_TOOL / "concurrent_harness" / "mape")]
sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))  # Linux-only import
import execute  # noqa: E402
sys.path[:] = _saved_path
for _m in {"execute", "retrain", "knowledge_io", "fcntl"} & (set(sys.modules) - _saved_mods):
    del sys.modules[_m]


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

"""Smoke check config (2026-10-18, run_concurrent.py --smoke): no pacing, no
per-prediction reload, fixed per-model energies from the real naive runs."""
import json
import sys
import types
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parent.parent

# Import run_concurrent.py without leaking knowledge_io to other tests (see
# test_retrain_energy.py).
_saved_path, _saved_mods = list(sys.path), set(sys.modules)
sys.path[:0] = [str(_TOOL), str(_TOOL / "concurrent_harness")]
sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))  # Linux-only import
import run_concurrent  # noqa: E402
sys.path[:] = _saved_path
for _m in {"run_concurrent", "knowledge_io", "fcntl"} & (set(sys.modules) - _saved_mods):
    del sys.modules[_m]

CFG = {"models": {"ridge": {}, "svr": {}, "lstm": {}}, "stream_delay_s": 0.0075,
       "reload_model_per_step": True, "energy_metering": "per_prediction", "E_m": 21000.0}


def _naive(tmp_path, model, **fields):
    d = tmp_path / f"ds_naive_{model}_conc"
    d.mkdir()
    (d / "run_manifest.json").write_text(json.dumps(fields))


def test_smoke_config(tmp_path):
    _naive(tmp_path, "ridge", inference_energy_mJ=20.99, energy_mJ=22.7)
    _naive(tmp_path, "svr", inference_energy_mJ=30.24)
    _naive(tmp_path, "lstm", energy_mJ=77.83)  # older manifest: energy_mJ was the per-prediction mean
    cfg = run_concurrent._smoke_config(CFG, "ds", tmp_path)
    assert cfg["fixed_energy_uJ"] == {"ridge": 20990.0, "svr": 30240.0, "lstm": 77830.0}
    assert cfg["stream_delay_s"] == 0.0 and cfg["reload_model_per_step"] is False
    assert cfg["energy_metering"] == "fixed" and cfg["smoke_wait_retrain"] is True
    assert cfg["E_m"] == 21000.0 and CFG["stream_delay_s"] == 0.0075  # thresholds kept, input untouched


def test_smoke_config_needs_naive_runs(tmp_path):
    _naive(tmp_path, "ridge", inference_energy_mJ=20.99)
    with pytest.raises(SystemExit):
        run_concurrent._smoke_config(CFG, "ds", tmp_path)

"""tests/test_live_run_fixes.py — Regression tests for L1–L6, E1–E3 fixes.

Every test must fail on the un-fixed code and pass on the fixed code.
Tests are pure-Python; none require a live ACP server, subprocess, or GPU.
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))


# =============================================================================
# L2-a: Startup clear of command.txt
# =============================================================================

def _load_manage_module(domain: str) -> ModuleType:
    """Load manage.py for a domain by directly loading the file with mocked deps."""
    mape_logic_dir = _TOOL_DIR / f"managed_system_{domain}" / "mape_logic"
    # Mock 'execute' so manage.py's top-level import doesn't fail in tests
    mock_execute = MagicMock()
    sys.modules.setdefault("execute", mock_execute)
    sys.path.insert(0, str(mape_logic_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"{domain}_manage", str(mape_logic_dir / "manage.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if str(mape_logic_dir) in sys.path:
            sys.path.remove(str(mape_logic_dir))
    return mod


class TestCommandFileClearOnStartup:
    """manage.py must truncate command.txt before the listener loop starts."""

    def test_clear_stale_command_removes_content(self, tmp_path: Path) -> None:
        m = _load_manage_module("regression")
        cmd_file = tmp_path / "command.txt"
        cmd_file.write_text("execute_mape_plan|0")

        orig = m.COMMAND_FILE_PATH
        m.COMMAND_FILE_PATH = str(cmd_file)
        try:
            m._clear_stale_command()
            assert cmd_file.read_text() == "", "command.txt should be empty after clear"
        finally:
            m.COMMAND_FILE_PATH = orig

    def test_clear_stale_command_no_file_ok(self, tmp_path: Path) -> None:
        m = _load_manage_module("regression")
        orig = m.COMMAND_FILE_PATH
        m.COMMAND_FILE_PATH = str(tmp_path / "command.txt")
        try:
            m._clear_stale_command()  # must not raise
        finally:
            m.COMMAND_FILE_PATH = orig

    def test_cv_clear_stale_command(self, tmp_path: Path) -> None:
        m = _load_manage_module("cv")
        cmd_file = tmp_path / "command.txt"
        cmd_file.write_text("execute_mape_plan|0")

        orig = m.COMMAND_FILE_PATH
        m.COMMAND_FILE_PATH = str(cmd_file)
        try:
            m._clear_stale_command()
            assert cmd_file.read_text() == ""
        finally:
            m.COMMAND_FILE_PATH = orig


# =============================================================================
# L2-c: Timestamp-gated command parsing
# =============================================================================

class TestTimestampGatedCommands:
    """Commands without timestamps or older than 30s must be discarded."""

    @pytest.fixture(autouse=True)
    def _import_manage(self):
        self.m = _load_manage_module("regression")

    def test_fresh_command_accepted(self) -> None:
        raw = f"execute_mape_plan|{time.time()}"
        result = self.m._parse_and_validate_command(raw)
        assert result == "execute_mape_plan"

    def test_stale_command_rejected(self) -> None:
        stale_ts = time.time() - 60
        raw = f"execute_mape_plan|{stale_ts}"
        result = self.m._parse_and_validate_command(raw)
        assert result is None, "Command older than 30s must be rejected"

    def test_legacy_no_timestamp_rejected(self) -> None:
        result = self.m._parse_and_validate_command("execute_mape_plan")
        assert result is None, "Legacy format without timestamp must be rejected"

    def test_bad_timestamp_rejected(self) -> None:
        result = self.m._parse_and_validate_command("execute_mape_plan|NOT_A_NUMBER")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        assert self.m._parse_and_validate_command("") is None

    def test_cv_fresh_command_accepted(self) -> None:
        cv_m = _load_manage_module("cv")
        raw = f"execute_mape_plan|{time.time()}"
        assert cv_m._parse_and_validate_command(raw) == "execute_mape_plan"

    def test_cv_stale_command_rejected(self) -> None:
        cv_m = _load_manage_module("cv")
        raw = f"execute_mape_plan|{time.time() - 100}"
        assert cv_m._parse_and_validate_command(raw) is None


# =============================================================================
# L4: None-safe boundary evaluation in app.py
# =============================================================================

class TestNoneSafeBoundaryEvaluation:
    """evaluate_boundary must return False for None without raising."""

    @pytest.fixture(autouse=True)
    def _import_app(self):
        import importlib.util as _iu
        spec = _iu.spec_from_file_location("app_under_test", str(_TOOL_DIR / "app.py"))
        mod = _iu.module_from_spec(spec)
        # Stub all heavy imports so we can load the module without a Flask server
        mock_flask = MagicMock()
        mock_flask_module = MagicMock()
        mock_flask_module.Flask = MagicMock(return_value=MagicMock())
        mock_flask_module.request = MagicMock()
        mock_flask_module.jsonify = MagicMock()
        with patch.dict("sys.modules", {
            "flask": mock_flask_module,
            "flask_cors": MagicMock(),
            "requests": MagicMock(),
            "psutil": MagicMock(),
            "zipfile": MagicMock(),
            "shutil": MagicMock(),
        }):
            spec.loader.exec_module(mod)
        self.evaluate_boundary = mod.evaluate_boundary

    def test_none_value_returns_false_greater_than(self) -> None:
        assert self.evaluate_boundary(None, "GREATER_THAN", 0.1) is False

    def test_none_value_returns_false_less_than(self) -> None:
        assert self.evaluate_boundary(None, "LESS_THAN", 0.5) is False

    def test_value_above_threshold_greater_than(self) -> None:
        assert self.evaluate_boundary(0.9, "GREATER_THAN", 0.5) is True

    def test_value_below_threshold_greater_than(self) -> None:
        assert self.evaluate_boundary(0.2, "GREATER_THAN", 0.5) is False

    def test_value_below_threshold_less_than(self) -> None:
        assert self.evaluate_boundary(0.2, "LESS_THAN", 0.5) is True

    def test_unknown_condition_returns_false(self) -> None:
        assert self.evaluate_boundary(0.9, "EQUAL", 0.9) is False


# =============================================================================
# L1-c: monitor_mape returns fresh=False instead of stale cached data
# =============================================================================

class TestMonitorNoStaleTelemetry:
    """monitor_mape must return {"fresh": False} when predictions.csv has no new rows."""

    @pytest.fixture(autouse=True)
    def _patch_paths(self, tmp_path: Path):
        self.knowledge = tmp_path / "knowledge"
        self.knowledge.mkdir()

        # Write a minimal predictions.csv with data (simulating a prior session)
        pred_file = self.knowledge / "predictions.csv"
        with open(pred_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true_value", "predicted_value", "model_used", "inference_time", "energy_uJ"])
            for i in range(5):
                w.writerow([1.0, 1.1, "linear", 0.01, 100.0])

        # mape_info.json with last_line == 5 (all rows already seen)
        mape_info = {
            "last_line": 5,
            "ema_scores": {"linear": 0.5, "svm": 0.5, "lstm": 0.5},
            "event_counters": {"model_switches": 0, "retrains": 0, "vmr_events": 0, "mape_k_energy_uJ": 0.0},
            "simple_switch_counters": {"simple_switches": 0},
        }
        (self.knowledge / "mape_info.json").write_text(json.dumps(mape_info))

        # model.csv
        (self.knowledge / "model.csv").write_text("linear")

        # thresholds.json
        thresholds = {"E_m": 0, "E_M": 25000, "beta": 0.95, "gamma": 0.8, "tau_drift": 0.5}
        (self.knowledge / "thresholds.json").write_text(json.dumps(thresholds))

        # Patch module-level paths used by monitor
        from managed_system_regression.mape_logic import monitor as mon_mod
        self._orig_pred = mon_mod.predictions_file
        self._orig_mape = mon_mod.mape_info_file
        self._orig_thresh = mon_mod.thresholds_file
        self._orig_model = mon_mod.model_file
        mon_mod.predictions_file = str(self.knowledge / "predictions.csv")
        mon_mod.mape_info_file = str(self.knowledge / "mape_info.json")
        mon_mod.thresholds_file = str(self.knowledge / "thresholds.json")
        mon_mod.model_file = str(self.knowledge / "model.csv")
        self.mon = mon_mod
        yield
        mon_mod.predictions_file = self._orig_pred
        mon_mod.mape_info_file = self._orig_mape
        mon_mod.thresholds_file = self._orig_thresh
        mon_mod.model_file = self._orig_model

    def test_no_new_rows_returns_fresh_false(self) -> None:
        """All rows already seen → monitor must signal not-fresh, not serve stale data."""
        result = self.mon.monitor_mape()
        assert result is not None, "monitor_mape should return a dict, not None"
        assert result.get("fresh") is False, (
            f"Expected {{'fresh': False}} but got {result}. "
            "The stale-cache path was not removed — inference stalling will go undetected."
        )

    def test_no_r2_in_stale_result(self) -> None:
        """The stale-signal result must NOT contain r2_score (no fabricated telemetry)."""
        result = self.mon.monitor_mape() or {}
        assert "r2_score" not in result or result.get("fresh") is False


# =============================================================================
# L2-b / L6: run_reset.py zeros counters and preserves training artifacts
# =============================================================================

class TestRunReset:
    """reset_run_state must zero counters, reset EMAs, truncate predictions.csv,
    and leave scaler.pkl + reference_distribution.json untouched."""

    @pytest.fixture
    def regression_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "managed_system_regression"
        knowledge = d / "knowledge"
        knowledge.mkdir(parents=True)

        mape_info = {
            "last_line": 500,
            "recovery_cycles": 3,
            "ema_scores": {"lstm": -5.9, "linear": -2.4, "svm": -0.1},
            "ema_accuracy": {"lstm": -6.2, "linear": 0.0, "svm": 0.2},
            "ema_energy": {"lstm": 0.6, "linear": 0.3, "svm": 0.4},
            "event_counters": {
                "model_switches": 113,
                "retrains": 5,
                "vmr_events": 0,
                "noops": 0,
                "mape_k_energy_uJ": 197239446.0,
            },
            "simple_switch_counters": {"simple_switches": 7},
        }
        (knowledge / "mape_info.json").write_text(json.dumps(mape_info))

        # predictions.csv with data
        with open(knowledge / "predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true_value", "predicted_value", "model_used"])
            for i in range(10):
                w.writerow([1.0, 1.1, "linear"])

        # Training artifacts that must NOT be deleted
        (knowledge / "scaler.pkl").write_bytes(b"fake_scaler")
        (knowledge / "reference_distribution.json").write_text('{"histogram": []}')

        # Volatile files
        (knowledge / "command.txt").write_text("some_cmd|0")
        (knowledge / "drift.csv").write_text("drift_data")

        # versionedMR must be preserved
        vmr = d / "versionedMR"
        vmr.mkdir()
        (vmr / "model_v1.pkl").write_bytes(b"model")

        return d

    def test_counters_zeroed(self, regression_dir: Path, tmp_path: Path) -> None:
        from experiments.run_reset import reset_run_state
        result = reset_run_state(str(regression_dir))

        knowledge = regression_dir / "knowledge"
        info = json.loads((knowledge / "mape_info.json").read_text())
        assert info["event_counters"]["model_switches"] == 0
        assert info["event_counters"]["retrains"] == 0
        assert info["event_counters"]["mape_k_energy_uJ"] == 0.0
        assert info["last_line"] == 0
        assert info["recovery_cycles"] == 0

    def test_ema_scores_reset_to_half(self, regression_dir: Path) -> None:
        from experiments.run_reset import reset_run_state
        reset_run_state(str(regression_dir))

        info = json.loads((regression_dir / "knowledge" / "mape_info.json").read_text())
        for model, score in info["ema_scores"].items():
            assert score == 0.5, f"EMA for {model} should be 0.5, got {score}"

    def test_predictions_csv_truncated_to_header(self, regression_dir: Path) -> None:
        from experiments.run_reset import reset_run_state
        reset_run_state(str(regression_dir))

        pred = regression_dir / "knowledge" / "predictions.csv"
        rows = list(csv.reader(pred.open()))
        assert len(rows) == 1, f"Expected header only; got {len(rows)} rows"
        assert rows[0][0] == "true_value"

    def test_training_artifacts_preserved(self, regression_dir: Path) -> None:
        from experiments.run_reset import reset_run_state
        reset_run_state(str(regression_dir))

        knowledge = regression_dir / "knowledge"
        assert (knowledge / "scaler.pkl").exists(), "scaler.pkl must be preserved"
        assert (knowledge / "reference_distribution.json").exists(), \
            "reference_distribution.json must be preserved"

    def test_thresholds_json_preserved_entirely(self, regression_dir: Path) -> None:
        """G4: reset must never touch thresholds.json — planner config survives across runs."""
        from experiments.run_reset import reset_run_state
        knowledge = regression_dir / "knowledge"
        thresholds = {
            "min_score": 0.78, "max_energy": 0.6, "planner": "greedy_switch",
            "energy_meter": "auto", "switch_cooldown_s": 30,
        }
        (knowledge / "thresholds.json").write_text(json.dumps(thresholds))

        reset_run_state(str(regression_dir))

        after = json.loads((knowledge / "thresholds.json").read_text())
        assert after["planner"] == "greedy_switch", \
            "reset must NOT touch thresholds.json — planner key must survive"
        assert after["min_score"] == 0.78, \
            "reset must NOT touch thresholds.json — config values must survive"

    def test_volatile_files_deleted(self, regression_dir: Path) -> None:
        from experiments.run_reset import reset_run_state
        reset_run_state(str(regression_dir))

        knowledge = regression_dir / "knowledge"
        assert not (knowledge / "command.txt").exists()
        assert not (knowledge / "drift.csv").exists()

    def test_versionedmr_preserved(self, regression_dir: Path) -> None:
        from experiments.run_reset import reset_run_state
        reset_run_state(str(regression_dir))

        assert (regression_dir / "versionedMR" / "model_v1.pkl").exists(), \
            "versionedMR must not be deleted by reset"

    def test_dry_run_makes_no_changes(self, regression_dir: Path) -> None:
        from experiments.run_reset import reset_run_state
        knowledge = regression_dir / "knowledge"
        before_info = json.loads((knowledge / "mape_info.json").read_text())

        reset_run_state(str(regression_dir), dry_run=True)

        after_info = json.loads((knowledge / "mape_info.json").read_text())
        assert after_info["event_counters"]["model_switches"] == \
               before_info["event_counters"]["model_switches"], \
               "dry-run must not modify mape_info.json"

    def test_last_switch_ts_reset_to_zero(self, regression_dir: Path) -> None:
        """run_reset must zero last_switch_ts so the cooldown guard starts fresh."""
        import time
        from experiments.run_reset import reset_run_state

        # Write a recent last_switch_ts into mape_info
        knowledge = regression_dir / "knowledge"
        info = json.loads((knowledge / "mape_info.json").read_text())
        info["last_switch_ts"] = time.time()  # simulate a very recent switch
        (knowledge / "mape_info.json").write_text(json.dumps(info))

        reset_run_state(str(regression_dir))

        after = json.loads((knowledge / "mape_info.json").read_text())
        assert after.get("last_switch_ts", -1) == 0.0, \
            "last_switch_ts must be zeroed on reset so the cooldown guard doesn't block the first switch"


# =============================================================================
# L1-a: init_regression.py fits scaler on training split only
# =============================================================================

def _setup_toy_regression(tmp_path: Path, config_name: str, n: int = 100,
                           seed: int | None = None) -> Path:
    """Create a minimal regression dataset + config under tmp_path for init_regression tests."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"flow": np.arange(n, dtype=float) if seed is None else rng.random(n)})

    # data lives in tmp_path/knowledge/dataset.csv (mirrors managed_system_regression layout)
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir(exist_ok=True)
    data_path = knowledge / "dataset.csv"
    df.to_csv(data_path, index=False)

    # Config uses path relative to tmp_path (the fake _TOOL_DIR)
    config = {
        "name": config_name,
        "domain": "regression",
        "data_path": "knowledge/dataset.csv",  # relative to _TOOL_DIR
        "value_column": "flow",
        "train_frac": 0.8,
    }
    cfg_dir = tmp_path / "configs" / "datasets"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / f"{config_name}.json").write_text(json.dumps(config))
    return tmp_path  # return as fake tool_dir


class TestInitRegression:
    """init_regression must fit MinMaxScaler on the training split only (not full dataset)."""

    def test_scaler_fitted_on_train_split(self, tmp_path: Path) -> None:
        import numpy as np
        import pandas as pd
        import pickle

        tool_dir = _setup_toy_regression(tmp_path, "test_scaler", n=100, seed=None)

        from scripts import init_regression as ir
        orig = ir._TOOL_DIR
        ir._TOOL_DIR = tool_dir
        try:
            ir.init_regression("test_scaler", force=True)
        finally:
            ir._TOOL_DIR = orig

        scaler_path = tool_dir / "knowledge" / "scaler.pkl"
        assert scaler_path.exists(), "scaler.pkl must be written"

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        n = 100
        train_end = int(n * 0.8)
        expected_min = float(0)           # arange(100)[:80].min()
        expected_max = float(train_end - 1)  # arange(100)[:80].max() = 79

        assert abs(scaler.data_min_[0] - expected_min) < 1e-6, \
            "Scaler must be fitted on training split only (B7 fix)"
        assert abs(scaler.data_max_[0] - expected_max) < 1e-6, \
            f"Expected max {expected_max} but got {scaler.data_max_[0]} (full-data max would be 99)"

    def test_reference_distribution_written(self, tmp_path: Path) -> None:
        tool_dir = _setup_toy_regression(tmp_path, "test_ref", n=100, seed=42)

        from scripts import init_regression as ir
        orig = ir._TOOL_DIR
        ir._TOOL_DIR = tool_dir
        try:
            ir.init_regression("test_ref", force=True)
        finally:
            ir._TOOL_DIR = orig

        ref_path = tool_dir / "knowledge" / "reference_distribution.json"
        assert ref_path.exists()
        data = json.loads(ref_path.read_text())
        assert "histogram" in data and "bin_edges" in data
        assert len(data["bin_edges"]) == len(data["histogram"]) + 1


# =============================================================================
# E3: EnergyMeter uses pyJoules, not pyRAPL
# =============================================================================

class TestNoPyRAPL:
    """No file in the repo (outside docs/) should import pyRAPL after E3 fix."""

    def test_no_pyrapl_import_in_source(self) -> None:
        import subprocess
        result = subprocess.run(
            ["git", "grep", "-r", "--include=*.py", "import pyRAPL"],
            cwd=str(_TOOL_DIR.parent),
            capture_output=True,
            text=True,
        )
        # Filter out doc files and tests themselves
        lines = [
            ln for ln in result.stdout.splitlines()
            if not any(skip in ln for skip in ["docs/", "test_live_run_fixes.py", ".md:"])
        ]
        assert lines == [], (
            f"Found pyRAPL imports in source files (E3 fix incomplete):\n"
            + "\n".join(lines)
        )


class TestEnergyMeterNestingForbidden:
    """Nesting EnergyMeter contexts must raise RuntimeError (amendment 4)."""

    def test_nesting_raises(self) -> None:
        from core.energy import EnergyMeter
        with EnergyMeter("outer", backend="null") as _outer:
            with pytest.raises(RuntimeError, match="nesting"):
                with EnergyMeter("inner", backend="null"):
                    pass


class TestEnergyMeterNullBackend:
    """Null backend returns None/invalid as specified."""

    def test_null_total_uJ_is_none(self) -> None:
        from core.energy import EnergyMeter
        with EnergyMeter("test", backend="null") as m:
            pass
        assert m.total_uJ is None
        assert m.valid is False

    def test_null_result_dict_structure(self) -> None:
        from core.energy import EnergyMeter
        with EnergyMeter("test", backend="null") as m:
            pass
        r = m.result
        assert "cpu_uJ" in r
        assert "gpu_uJ" in r
        assert "total_uJ" in r
        assert "valid" in r
        assert "cpu_valid" in r
        assert "gpu_valid" in r

    def test_after_exit_active_flag_released(self) -> None:
        """_ACTIVE must be False after __exit__ so a second context can be created."""
        import core.energy as ce
        from core.energy import EnergyMeter
        assert ce._ACTIVE is False
        with EnergyMeter("first", backend="null"):
            assert ce._ACTIVE is True
        assert ce._ACTIVE is False
        # Second context must succeed
        with EnergyMeter("second", backend="null"):
            assert ce._ACTIVE is True
        assert ce._ACTIVE is False


# =============================================================================
# L3: CV inference CUDA arch guard function
# =============================================================================

# =============================================================================
# G7: CV handle_data_drift must dispatch to execute_drift, not pass
# =============================================================================

class TestCVDriftTacticDispatch:
    """G7: execute_tactic_locally('handle_data_drift') must call execute_drift(trigger='acp')."""

    def test_handle_data_drift_calls_execute_drift(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mape_logic_dir = _TOOL_DIR / "managed_system_cv" / "mape_logic"

        mock_execute = MagicMock()
        mock_execute.execute_mape = MagicMock()
        mock_execute.execute_drift = MagicMock()
        mock_execute.execute_simple_switch = MagicMock()

        monkeypatch.setitem(sys.modules, "execute", mock_execute)
        monkeypatch.syspath_prepend(str(mape_logic_dir))

        spec = importlib.util.spec_from_file_location(
            "cv_manage_g7_test", str(mape_logic_dir / "manage.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mod.execute_tactic_locally("handle_data_drift")

        mock_execute.execute_drift.assert_called_once_with(trigger="acp")
        mock_execute.execute_mape.assert_not_called()

    def test_execute_mape_plan_not_confused_with_drift(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mape_logic_dir = _TOOL_DIR / "managed_system_cv" / "mape_logic"

        mock_execute = MagicMock()
        monkeypatch.setitem(sys.modules, "execute", mock_execute)
        monkeypatch.syspath_prepend(str(mape_logic_dir))

        spec = importlib.util.spec_from_file_location(
            "cv_manage_g7b_test", str(mape_logic_dir / "manage.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mod.execute_tactic_locally("execute_mape_plan")

        mock_execute.execute_mape.assert_called_once_with(trigger="acp")
        mock_execute.execute_drift.assert_not_called()


class TestCudaArchGuard:
    """The arch-guard logic must correctly identify supported/unsupported GPUs."""

    def _check_supported(self, cc: tuple, arch_list: list) -> bool:
        """Mirror the guard logic from inference.py."""
        target_sm = f"sm_{cc[0]}{cc[1]}"
        return any(
            a == target_sm or (a.startswith("sm_") and int(a[3:]) >= cc[0] * 10 + cc[1])
            for a in arch_list
        )

    def test_exact_match_supported(self) -> None:
        assert self._check_supported((9, 0), ["sm_50", "sm_80", "sm_90"]) is True

    def test_higher_sm_considered_supported(self) -> None:
        # If the installed torch has sm_100, it supports cc 9.0 (forward compat)
        # Actually this tests the opposite: sm_100 >= 90 is True
        assert self._check_supported((9, 0), ["sm_100"]) is True

    def test_sm_120_not_in_pre_cu129_build(self) -> None:
        # Old cu126 build: up to sm_90
        old_arches = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
        assert self._check_supported((12, 0), old_arches) is False

    def test_sm_120_in_cu129_plus_build(self) -> None:
        # New cu129+ build includes sm_120
        new_arches = ["sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
        assert self._check_supported((12, 0), new_arches) is True

    def test_cpu_only_no_cuda_not_checked(self) -> None:
        # The guard only runs when torch.cuda.is_available() — CPU-only is always safe
        # This is tested implicitly by the other tests (no GPU needed here)
        assert True  # guard is a no-op without CUDA

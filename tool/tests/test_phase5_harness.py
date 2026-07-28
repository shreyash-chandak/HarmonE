"""
tests/test_phase5_harness.py — Phase 5 experiment harness unit tests.

Tests target pure functions and IO helpers; no model weights required.
Each test builds its own isolated tmp directory and tears it down.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Path bootstrap ────────────────────────────────────────────────────────────

_TOOL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TOOL_DIR))

from experiments.run_experiment import (
    _analyse_violation,
    _build_reference_distribution,
    _initial_mape_info,
    _monitor_batch,
    _update_energy_boundary,
    setup_run_dir,
)
from experiments.metrics import (
    aggregate_by_planner,
    aggregate_grid,
    compute_run_metrics,
    pareto_efficiency,
    to_csv,
    wilcoxon_test,
)
from experiments.run_grid import load_grid_config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_predictions_csv(path: Path, n: int = 100) -> tuple[list[float], list[float]]:
    """Write a synthetic predictions.csv.  Returns (y_true, y_pred)."""
    rng = np.random.default_rng(0)
    y_true = rng.uniform(50, 100, size=n).tolist()
    noise = rng.normal(0, 3, size=n).tolist()
    y_pred = [t + e for t, e in zip(y_true, noise)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "y_true", "y_pred", "active_model", "energy_uJ"])
        writer.writeheader()
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            writer.writerow({
                "step": i,
                "y_true": round(t, 6),
                "y_pred": round(p, 6),
                "active_model": "lstm",
                "energy_uJ": round(rng.uniform(0.0, 100.0), 4),
            })
    return y_true, y_pred


def _make_run_dir(tmp: Path, run_id: str = "test_run", **manifest_extras) -> Path:
    """Synthesise a minimal completed run directory."""
    run_path = tmp / run_id
    run_path.mkdir(parents=True)

    y_true, y_pred = _make_predictions_csv(run_path / "predictions.csv")

    manifest = {
        "run_id": run_id,
        "dataset": "pems_node1",
        "planner": "harmone_original",
        "seed": 1,
        "monitor_interval": 50,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00",
        "elapsed_s": 60.0,
        "total_steps": 100,
        "mape_cycles": 2,
        "final_model": "lstm",
        "event_counters": {
            "model_switches": 3,
            "retrains": 0,
            "retrain_skipped": 1,
            "vmr_events": 0,
            "noops": 5,
            "mape_k_energy_uJ": 1234.5,
        },
        "final_ema_scores": {"lstm": 0.85, "linear": 0.72},
        "status": "ok",
        **manifest_extras,
    }
    with open(run_path / "run_manifest.json", "w") as f:
        json.dump(manifest, f)

    return run_path


# ── setup_run_dir ─────────────────────────────────────────────────────────────

class TestSetupRunDir:
    def test_creates_directory(self, tmp_path):
        target = str(tmp_path / "new_run" / "subdir")
        p = setup_run_dir(target)
        assert p.is_dir()

    def test_idempotent(self, tmp_path):
        target = str(tmp_path / "run1")
        setup_run_dir(target)
        setup_run_dir(target)  # must not raise
        assert Path(target).is_dir()

    def test_returns_path(self, tmp_path):
        target = str(tmp_path / "run2")
        p = setup_run_dir(target)
        assert isinstance(p, Path)
        assert str(p) == target


# ── _build_reference_distribution ────────────────────────────────────────────

class TestBuildReferenceDistribution:
    def test_writes_json(self, tmp_path):
        values = np.linspace(0, 100, 500)
        ref_path = str(tmp_path / "ref.json")
        _build_reference_distribution(values, ref_path)
        assert Path(ref_path).exists()

    def test_json_has_expected_keys(self, tmp_path):
        values = np.random.default_rng(1).uniform(0, 100, 300)
        ref_path = str(tmp_path / "ref.json")
        _build_reference_distribution(values, ref_path, n_bins=30)
        with open(ref_path) as f:
            data = json.load(f)
        assert "histogram" in data
        assert "bin_edges" in data

    def test_histogram_length(self, tmp_path):
        n_bins = 20
        values = np.arange(100, dtype=float)
        ref_path = str(tmp_path / "ref.json")
        _build_reference_distribution(values, ref_path, n_bins=n_bins)
        with open(ref_path) as f:
            data = json.load(f)
        assert len(data["histogram"]) == n_bins
        assert len(data["bin_edges"]) == n_bins + 1

    def test_histogram_sums_to_n(self, tmp_path):
        values = np.arange(50, dtype=float)
        ref_path = str(tmp_path / "ref.json")
        _build_reference_distribution(values, ref_path, n_bins=10)
        with open(ref_path) as f:
            data = json.load(f)
        assert sum(data["histogram"]) == len(values)


# ── _initial_mape_info ────────────────────────────────────────────────────────

class TestInitialMapeInfo:
    def test_has_all_models(self):
        models = {"lstm": None, "linear": None, "svm": None}
        info = _initial_mape_info(models)
        for m in models:
            assert m in info["ema_scores"]
            assert m in info["ema_accuracy"]
            assert m in info["ema_energy"]

    def test_counter_keys_present(self):
        info = _initial_mape_info({"a": None})
        ec = info["event_counters"]
        assert "model_switches" in ec
        assert "noops" in ec
        assert "mape_k_energy_uJ" in ec
        assert "retrain_skipped" in ec

    def test_initial_ema_is_float(self):
        info = _initial_mape_info({"m1": None})
        assert isinstance(info["ema_scores"]["m1"], float)


# ── _monitor_batch ────────────────────────────────────────────────────────────

class TestMonitorBatch:
    _thresholds = {
        "E_m": 0.0,
        "E_M": 1000.0,
        "beta": 0.95,
        "gamma": 0.8,
        "min_score": 0.78,
    }

    def _make_info(self, model="lstm"):
        return _initial_mape_info({model: None})

    def test_returns_r2(self):
        y_true = list(range(100))
        y_pred = [v + 0.01 for v in y_true]
        info = self._make_info()
        result = _monitor_batch(y_true, y_pred, [0.0] * 100, "lstm", info, self._thresholds)
        assert result["r2"] > 0.99

    def test_updates_ema_scores(self):
        y_true = [float(i) for i in range(50)]
        y_pred = [v + 0.1 for v in y_true]
        info = self._make_info()
        prev = info["ema_scores"]["lstm"]
        _monitor_batch(y_true, y_pred, [10.0] * 50, "lstm", info, self._thresholds)
        assert info["ema_scores"]["lstm"] != prev

    def test_updates_separated_emas(self):
        y_true = list(range(50))
        y_pred = list(range(50))
        info = self._make_info()
        _monitor_batch(y_true, y_pred, [0.0] * 50, "lstm", info, self._thresholds)
        assert "ema_accuracy" in info
        assert "ema_energy" in info
        assert "lstm" in info["ema_accuracy"]

    def test_avg_energy_reported(self):
        y_true = list(range(10))
        y_pred = list(range(10))
        energies = [200.0] * 10  # 200 µJ each
        info = self._make_info()
        result = _monitor_batch(y_true, y_pred, energies, "lstm", info, self._thresholds)
        assert pytest.approx(result["avg_energy_uJ"], rel=1e-3) == 200.0

    def test_single_sample_does_not_crash(self):
        info = self._make_info()
        result = _monitor_batch([5.0], [5.1], [10.0], "lstm", info, self._thresholds)
        assert "r2" in result


# ── _analyse_violation ────────────────────────────────────────────────────────

class TestAnalyseViolation:
    def _info(self):
        return {"current_energy_threshold": 0.6}

    def _thresholds(self):
        return {"min_score": 0.78}

    def test_no_violation(self):
        tel = {"ema_score": 0.85, "normalized_energy": 0.3}
        v = _analyse_violation(tel, self._info(), self._thresholds())
        assert v is None

    def test_score_violation(self):
        tel = {"ema_score": 0.50, "normalized_energy": 0.3}
        v = _analyse_violation(tel, self._info(), self._thresholds())
        assert v == "score"

    def test_energy_violation(self):
        tel = {"ema_score": 0.90, "normalized_energy": 0.8}
        v = _analyse_violation(tel, self._info(), self._thresholds())
        assert v == "energy"

    def test_score_takes_priority_over_energy(self):
        # Both violated: score check happens first
        tel = {"ema_score": 0.50, "normalized_energy": 0.9}
        v = _analyse_violation(tel, self._info(), self._thresholds())
        assert v == "score"


# ── _update_energy_boundary ───────────────────────────────────────────────────

class TestUpdateEnergyBoundary:
    def test_threshold_increases_when_under_ref(self):
        info = {"current_energy_threshold": 0.5}
        thresholds = {"E_ref": 0.7, "delta": 0.1}
        tel = {"normalized_energy": 0.3}  # well below E_ref
        _update_energy_boundary(info, tel, thresholds)
        assert info["current_energy_threshold"] > 0.5

    def test_threshold_decreases_when_over_ref(self):
        info = {"current_energy_threshold": 0.8}
        thresholds = {"E_ref": 0.7, "delta": 0.1}
        tel = {"normalized_energy": 0.95}  # above E_ref
        _update_energy_boundary(info, tel, thresholds)
        assert info["current_energy_threshold"] < 0.8

    def test_threshold_clamped_to_1(self):
        info = {"current_energy_threshold": 0.99}
        thresholds = {"E_ref": 0.7, "delta": 0.5}
        tel = {"normalized_energy": 0.0}
        _update_energy_boundary(info, tel, thresholds)
        assert info["current_energy_threshold"] <= 1.0

    def test_threshold_clamped_to_01(self):
        info = {"current_energy_threshold": 0.11}
        thresholds = {"E_ref": 0.0, "delta": 0.5}
        tel = {"normalized_energy": 1.0}
        _update_energy_boundary(info, tel, thresholds)
        assert info["current_energy_threshold"] >= 0.1


# ── compute_run_metrics ───────────────────────────────────────────────────────

class TestComputeRunMetrics:
    def test_reads_manifest_and_predictions(self, tmp_path):
        run_path = _make_run_dir(tmp_path, "r1")
        m = compute_run_metrics(str(run_path))
        assert m["dataset"] == "pems_node1"
        assert m["planner"] == "harmone_original"
        assert m["seed"] == 1

    def test_r2_computed(self, tmp_path):
        run_path = _make_run_dir(tmp_path, "r2_run")
        m = compute_run_metrics(str(run_path))
        assert m["r2_mean"] is not None
        assert -1.0 <= m["r2_mean"] <= 1.0

    def test_energy_totals(self, tmp_path):
        run_path = _make_run_dir(tmp_path, "r3")
        m = compute_run_metrics(str(run_path))
        assert m["energy_uJ_total"] >= 0
        assert m["energy_uJ_mean"] >= 0

    def test_counters_present(self, tmp_path):
        run_path = _make_run_dir(tmp_path, "r4")
        m = compute_run_metrics(str(run_path))
        assert "model_switches" in m
        assert m["model_switches"] == 3

    def test_missing_manifest_raises(self, tmp_path):
        run_path = tmp_path / "empty_run"
        run_path.mkdir()
        with pytest.raises(FileNotFoundError):
            compute_run_metrics(str(run_path))


# ── aggregate_grid ────────────────────────────────────────────────────────────

class TestAggregateGrid:
    def test_finds_all_runs(self, tmp_path):
        _make_run_dir(tmp_path, "run_a", planner="naive")
        _make_run_dir(tmp_path, "run_b", planner="harmone_original")
        rows = aggregate_grid(str(tmp_path))
        assert len(rows) == 2

    def test_skips_non_run_dirs(self, tmp_path):
        _make_run_dir(tmp_path, "run_ok")
        empty = tmp_path / "not_a_run"
        empty.mkdir()  # no manifest
        rows = aggregate_grid(str(tmp_path))
        assert len(rows) == 1

    def test_sorted_by_dataset_planner_seed(self, tmp_path):
        _make_run_dir(tmp_path, "r_b", planner="naive", seed=2)
        _make_run_dir(tmp_path, "r_a", planner="harmone_original", seed=1)
        rows = aggregate_grid(str(tmp_path))
        planners = [r["planner"] for r in rows]
        assert planners == sorted(planners)


# ── aggregate_by_planner ──────────────────────────────────────────────────────

class TestAggregateByPlanner:
    def _rows(self):
        return [
            {"dataset": "pems_node1", "planner": "naive", "seed": 1,
             "r2_mean": 0.70, "r2_std": 0.05, "energy_uJ_mean": 100.0,
             "energy_uJ_total": 10000.0, "model_switches": 0,
             "mape_k_energy_uJ": 0.0, "elapsed_s": 10.0, "total_steps": 100},
            {"dataset": "pems_node1", "planner": "naive", "seed": 2,
             "r2_mean": 0.72, "r2_std": 0.04, "energy_uJ_mean": 110.0,
             "energy_uJ_total": 11000.0, "model_switches": 0,
             "mape_k_energy_uJ": 0.0, "elapsed_s": 11.0, "total_steps": 100},
        ]

    def test_collapses_to_one_row_per_planner(self):
        rows = self._rows()
        summary = aggregate_by_planner(rows)
        assert len(summary) == 1
        assert summary[0]["planner"] == "naive"
        assert summary[0]["n_seeds"] == 2

    def test_mean_is_correct(self):
        rows = self._rows()
        summary = aggregate_by_planner(rows)
        assert pytest.approx(summary[0]["r2_mean_mean"], rel=1e-4) == 0.71

    def test_std_is_correct(self):
        rows = self._rows()
        summary = aggregate_by_planner(rows)
        # std of [0.70, 0.72] with ddof=1
        expected_std = float(np.std([0.70, 0.72], ddof=1))
        assert pytest.approx(summary[0]["r2_mean_std"], rel=1e-4) == expected_std


# ── pareto_efficiency ─────────────────────────────────────────────────────────

class TestParetoEfficiency:
    def _rows(self):
        return [
            {"dataset": "d1", "planner": "A", "r2_mean_mean": 0.90, "energy_uJ_mean_mean": 50.0},
            {"dataset": "d1", "planner": "B", "r2_mean_mean": 0.80, "energy_uJ_mean_mean": 20.0},
            {"dataset": "d1", "planner": "C", "r2_mean_mean": 0.70, "energy_uJ_mean_mean": 30.0},
        ]

    def test_dominated_row_marked_false(self):
        rows = pareto_efficiency(self._rows())
        c_row = next(r for r in rows if r["planner"] == "C")
        assert c_row["is_pareto"] is False  # B has lower energy AND higher R²

    def test_pareto_rows_have_flag(self):
        rows = pareto_efficiency(self._rows())
        for r in rows:
            assert "is_pareto" in r
            assert isinstance(r["is_pareto"], bool)

    def test_single_row_always_pareto(self):
        rows = [{"dataset": "d1", "planner": "X", "r2_mean_mean": 0.8, "energy_uJ_mean_mean": 100.0}]
        result = pareto_efficiency(rows)
        assert result[0]["is_pareto"] is True


# ── wilcoxon_test ─────────────────────────────────────────────────────────────

class TestWilcoxonTest:
    def _rows(self, base_vals: list[float], treat_vals: list[float]) -> list[dict]:
        rows = []
        for i, (b, t) in enumerate(zip(base_vals, treat_vals)):
            rows.append({"dataset": "d1", "planner": "baseline", "seed": i + 1, "r2_mean": b})
            rows.append({"dataset": "d1", "planner": "treatment", "seed": i + 1, "r2_mean": t})
        return rows

    def test_significantly_better(self):
        # n=8 samples needed: two-sided Wilcoxon min p-value = 2/2^8 ≈ 0.008 < 0.05
        base = [0.70, 0.71, 0.69, 0.70, 0.71, 0.68, 0.72, 0.69]
        treat = [0.90, 0.91, 0.89, 0.88, 0.92, 0.87, 0.91, 0.90]
        rows = self._rows(base, treat)
        result = wilcoxon_test(rows, "baseline", "treatment")
        assert result["significant"] is True
        assert result["direction"] == "better"

    def test_insufficient_pairs_returns_none(self):
        rows = [{"dataset": "d1", "planner": "baseline", "seed": 1, "r2_mean": 0.8}]
        result = wilcoxon_test(rows, "baseline", "treatment")
        assert result["p_value"] is None

    def test_identical_values(self):
        base = [0.80, 0.80, 0.80, 0.80, 0.80]
        treat = [0.80, 0.80, 0.80, 0.80, 0.80]
        rows = self._rows(base, treat)
        result = wilcoxon_test(rows, "baseline", "treatment")
        assert result["significant"] is False


# ── load_grid_config ──────────────────────────────────────────────────────────

class TestLoadGridConfig:
    """Tests use JSON format (stdlib) so PyYAML is not required."""

    def _write_json(self, tmp_path: Path, data: dict) -> str:
        import json
        p = tmp_path / "grid.json"
        p.write_text(json.dumps(data))
        return str(p)

    def test_parses_required_keys(self, tmp_path):
        path = self._write_json(tmp_path, {
            "datasets": ["pems_node1"],
            "planners": ["naive", "harmone_original"],
            "seeds": [1, 2],
        })
        cfg = load_grid_config(path)
        assert cfg["datasets"] == ["pems_node1"]
        assert cfg["planners"] == ["naive", "harmone_original"]
        assert cfg["seeds"] == [1, 2]

    def test_missing_required_key_raises(self, tmp_path):
        path = self._write_json(tmp_path, {"datasets": ["pems_node1"], "planners": ["naive"]})
        with pytest.raises(ValueError, match="seeds"):
            load_grid_config(path)

    def test_optional_keys_have_defaults(self, tmp_path):
        path = self._write_json(tmp_path, {
            "datasets": ["pems_node1"],
            "planners": ["naive"],
            "seeds": [1],
            "resume": False,
            "monitor_interval": 100,
        })
        cfg = load_grid_config(path)
        assert cfg.get("resume") is False
        assert cfg.get("monitor_interval") == 100


# ── to_csv ────────────────────────────────────────────────────────────────────

class TestToCsv:
    def test_creates_file(self, tmp_path):
        rows = [{"a": 1, "b": 2.0}, {"a": 3, "b": 4.0}]
        out = str(tmp_path / "out.csv")
        to_csv(rows, out)
        assert Path(out).exists()

    def test_correct_content(self, tmp_path):
        rows = [{"x": 10, "y": "hello"}]
        out = str(tmp_path / "out.csv")
        to_csv(rows, out)
        with open(out, newline="") as f:
            reader = csv.DictReader(f)
            result = list(reader)
        assert result[0]["x"] == "10"
        assert result[0]["y"] == "hello"

    def test_empty_rows_does_not_create_file(self, tmp_path):
        out = str(tmp_path / "should_not_exist.csv")
        to_csv([], out)
        assert not Path(out).exists()


# ── loaders registry ──────────────────────────────────────────────────────────

class TestLoadersRegistry:
    def test_known_loader_returns_callable(self):
        from adapters.loaders import get_loader
        fn = get_loader("adapters.loaders.lstm_loader")
        assert callable(fn)

    def test_unknown_loader_raises_key_error(self):
        from adapters.loaders import get_loader
        with pytest.raises(KeyError, match="Unknown loader"):
            get_loader("adapters.loaders.nonexistent")

    def test_sklearn_loader_uses_pickle(self, tmp_path):
        """sklearn_loader must load a pickle file and return a predict callable."""
        from sklearn.linear_model import LinearRegression
        from adapters.loaders import sklearn_loader

        model = LinearRegression()
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        model.fit(X, y)

        pkl_path = str(tmp_path / "lr.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)

        predict = sklearn_loader(pkl_path)
        result = predict(np.array([5.0]))
        assert pytest.approx(result, rel=0.01) == 10.0

    def test_get_loader_and_load_model_consistent(self, tmp_path):
        """get_loader and load_model must produce the same callable for sklearn."""
        from sklearn.linear_model import LinearRegression
        from adapters.loaders import get_loader, load_model

        model = LinearRegression()
        model.fit([[1], [2]], [2.0, 4.0])
        pkl_path = str(tmp_path / "m.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)

        loader_fn = get_loader("adapters.loaders.sklearn_loader")
        p1 = loader_fn(pkl_path)(np.array([3.0]))

        spec = {"weights_path": pkl_path, "loader": "adapters.loaders.sklearn_loader"}
        p2 = load_model(spec)(np.array([3.0]))

        assert pytest.approx(p1, rel=1e-6) == p2

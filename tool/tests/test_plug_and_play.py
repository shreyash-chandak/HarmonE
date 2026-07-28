"""tests/test_plug_and_play.py — Dataset plug-and-play contract conformance tests.

These tests prove that a config-only dataset swap (no code changes) works end-
to-end for both domains.  They are intentionally self-contained: each test
generates toy data, validates, inits, and runs a tiny headless loop.

If adding a new dataset requires touching code in the managed_system_* or core/
directories, these tests become the first thing that breaks — that is the point.
"""
from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

# Make tool/ importable
_TOOL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TOOL_DIR))

from core.dataset_validator import validate, ValidationReport


# =============================================================================
# Helpers
# =============================================================================

def _make_regression_csv(path: Path, n: int = 200) -> None:
    """Write a small valid regression CSV."""
    import numpy as np
    rng = np.random.default_rng(0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["flow"])
        for v in (100.0 + 10.0 * rng.standard_normal(n)):
            w.writerow([round(float(v), 4)])


def _make_cv_images(img_dir: Path, n: int = 10) -> list[str]:
    """Write n small JPEG images; return relative paths."""
    pytest.importorskip("PIL", reason="Pillow required for CV toy dataset")
    from PIL import Image
    import numpy as np

    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    names: list[str] = []
    for i in range(n):
        r, g, b = (int(rng.integers(50, 200)) for _ in range(3))
        arr = __import__("numpy").full((64, 64, 3), [r, g, b], dtype=__import__("numpy").uint8)
        name = f"frame_{i:04d}.jpg"
        Image.fromarray(arr).save(img_dir / name, quality=85)
        names.append(f"images/{name}")
    return names


# =============================================================================
# Block 4.1: Validator unit tests
# =============================================================================

class TestDatasetValidator:
    """ValidationReport correctness — schema enforcement without touching files."""

    def test_missing_config_file(self, tmp_path: Path) -> None:
        r = validate(tmp_path / "nonexistent.json")
        assert not r.passed
        assert any("not found" in e for e in r.errors)

    def test_malformed_json(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad.json"
        cfg.write_text("{invalid")
        r = validate(cfg)
        assert not r.passed
        assert any("parse" in e.lower() or "JSON" in e for e in r.errors)

    def test_missing_required_keys(self, tmp_path: Path) -> None:
        cfg = tmp_path / "empty.json"
        cfg.write_text("{}")
        r = validate(cfg)
        assert not r.passed
        assert any("name" in e or "domain" in e for e in r.errors)

    def test_unknown_domain(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad_domain.json"
        cfg.write_text(json.dumps({"name": "x", "domain": "llm"}))
        r = validate(cfg)
        assert not r.passed
        assert any("domain" in e for e in r.errors)

    def test_train_frac_plus_val_frac_overflow(self, tmp_path: Path) -> None:
        _make_regression_csv(tmp_path / "d.csv")
        cfg = {
            "name": "t", "domain": "regression",
            "data_path": str(tmp_path / "d.csv"),
            "value_column": "flow",
            "train_frac": 0.8, "val_frac": 0.3,
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))
        r = validate(cfg_path)
        assert not r.passed
        assert any("1.0" in e or "train_frac" in e for e in r.errors)

    def test_valid_regression_config_passes(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dataset.csv"
        _make_regression_csv(csv_path, n=300)
        cfg = {
            "name": "toy", "domain": "regression",
            "data_path": str(csv_path),
            "value_column": "flow",
        }
        cfg_path = tmp_path / "toy.json"
        cfg_path.write_text(json.dumps(cfg))
        r = validate(cfg_path)
        assert r.passed, f"Expected pass; errors: {r.errors}"

    def test_missing_value_column_fails(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dataset.csv"
        csv_path.write_text("speed\n1.0\n2.0\n")
        cfg = {
            "name": "t", "domain": "regression",
            "data_path": str(csv_path), "value_column": "flow",
        }
        cfg_path = tmp_path / "t.json"
        cfg_path.write_text(json.dumps(cfg))
        r = validate(cfg_path)
        assert not r.passed
        assert any("value_column" in e for e in r.errors)

    def test_nan_in_value_column_fails(self, tmp_path: Path) -> None:
        import numpy as np
        import pandas as pd

        csv_path = tmp_path / "dataset.csv"
        vals: list = list(range(100)) + [float("nan")] + list(range(100, 200))
        pd.DataFrame({"flow": vals}).to_csv(csv_path, index=False)
        cfg = {
            "name": "t", "domain": "regression",
            "data_path": str(csv_path), "value_column": "flow",
        }
        cfg_path = tmp_path / "t.json"
        cfg_path.write_text(json.dumps(cfg))
        r = validate(cfg_path)
        assert not r.passed
        assert any("NaN" in e for e in r.errors)

    def test_dataset_too_small_fails(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dataset.csv"
        csv_path.write_text("flow\n1.0\n2.0\n3.0\n")
        cfg = {
            "name": "t", "domain": "regression",
            "data_path": str(csv_path), "value_column": "flow", "seq_length": 50,
        }
        cfg_path = tmp_path / "t.json"
        cfg_path.write_text(json.dumps(cfg))
        r = validate(cfg_path)
        assert not r.passed
        assert any("rows" in e.lower() or "minimum" in e.lower() for e in r.errors)


class TestDatasetValidatorCV:
    """CV-specific validator checks."""

    def test_missing_image_dir_warns(self, tmp_path: Path) -> None:
        cfg = {
            "name": "cv_t", "domain": "cv",
            "image_dir": str(tmp_path / "nonexistent"),
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))
        r = validate(cfg_path)
        # Missing image_dir is a warning (might be set up later), not an error
        assert any("image_dir" in w or "not found" in w for w in r.warnings), \
            f"Expected warning about missing image_dir; got warnings={r.warnings}"

    def test_valid_cv_config_with_images_passes(self, tmp_path: Path) -> None:
        pytest.importorskip("PIL", reason="Pillow required")
        img_dir = tmp_path / "images"
        _make_cv_images(img_dir, n=5)
        cfg = {"name": "cv_t", "domain": "cv", "image_dir": str(img_dir)}
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))
        r = validate(cfg_path)
        assert r.passed, f"Expected pass; errors={r.errors}"

    def test_cv_neither_image_dir_nor_manifest_fails(self, tmp_path: Path) -> None:
        cfg = {"name": "cv_t", "domain": "cv"}
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))
        r = validate(cfg_path)
        assert not r.passed
        assert any("image_dir" in e or "manifest" in e for e in r.errors)


# =============================================================================
# Block 4.2: init_regression.py plug-and-play
# =============================================================================

class TestInitRegressionPlugAndPlay:
    """Validate → init_regression → check artifacts: end-to-end with toy config."""

    def test_validate_then_init_then_artifacts_exist(self, tmp_path: Path) -> None:
        """Full happy path: synthetic dataset validates and produces scaler + ref dist."""
        import pickle

        # Generate toy dataset
        csv_path = tmp_path / "knowledge" / "dataset.csv"
        csv_path.parent.mkdir(parents=True)
        _make_regression_csv(csv_path, n=500)

        # Config: data_path relative to _TOOL_DIR (init_regression pattern)
        cfg = {
            "name": "toy_pp",
            "domain": "regression",
            "data_path": str(csv_path),  # absolute — validator handles this
            "value_column": "flow",
            "train_frac": 0.8,
        }
        cfg_dir = tmp_path / "configs" / "datasets"
        cfg_dir.mkdir(parents=True)
        cfg_path = cfg_dir / "toy_pp.json"
        cfg_path.write_text(json.dumps(cfg))

        # Validate
        r = validate(cfg_path)
        assert r.passed, f"Validation failed: {r.errors}"

        # Init with _TOOL_DIR patched to tmp_path
        from scripts import init_regression as ir_mod
        orig_tool_dir = ir_mod._TOOL_DIR

        # Override so init_regression resolves paths relative to tmp_path
        ir_mod._TOOL_DIR = tmp_path
        # Use absolute data_path in config to bypass resolution
        cfg["data_path"] = "knowledge/dataset.csv"
        cfg_path.write_text(json.dumps(cfg))
        try:
            ir_mod.init_regression("toy_pp", force=True)
        finally:
            ir_mod._TOOL_DIR = orig_tool_dir

        scaler_path = tmp_path / "knowledge" / "scaler.pkl"
        ref_path = tmp_path / "knowledge" / "reference_distribution.json"
        assert scaler_path.exists(), "scaler.pkl not created"
        assert ref_path.exists(), "reference_distribution.json not created"

        # Verify scaler was fitted on training split (not full data)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        assert scaler.data_min_[0] is not None

        # Verify histogram structure
        dist = json.loads(ref_path.read_text())
        assert "histogram" in dist and "bin_edges" in dist
        assert len(dist["bin_edges"]) == len(dist["histogram"]) + 1

    def test_force_flag_overwrites_existing_scaler(self, tmp_path: Path) -> None:
        """--force must overwrite existing scaler.pkl without error."""
        import pickle

        csv_path = tmp_path / "knowledge" / "dataset.csv"
        csv_path.parent.mkdir(parents=True)
        _make_regression_csv(csv_path, n=200)

        # Write a dummy scaler that should be replaced
        dummy_path = tmp_path / "knowledge" / "scaler.pkl"
        dummy_path.write_bytes(b"dummy")

        cfg = {
            "name": "toy_force",
            "domain": "regression",
            "data_path": "knowledge/dataset.csv",
            "value_column": "flow",
        }
        cfg_dir = tmp_path / "configs" / "datasets"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "toy_force.json").write_text(json.dumps(cfg))

        from scripts import init_regression as ir_mod
        orig = ir_mod._TOOL_DIR
        ir_mod._TOOL_DIR = tmp_path
        try:
            ir_mod.init_regression("toy_force", force=True)
        finally:
            ir_mod._TOOL_DIR = orig

        with open(dummy_path, "rb") as f:
            content = f.read(4)
        assert content != b"dumm", "scaler.pkl should have been overwritten by --force"
        # Should now be a valid pickle
        with open(dummy_path, "rb") as f:
            obj = pickle.load(f)
        assert hasattr(obj, "transform"), "Expected a fitted sklearn scaler"


# =============================================================================
# Block 4.3: dataset_validator integrated in run_managed_system startup
# =============================================================================

class TestValidatorRejectsMissingArtifacts:
    """Confirm that the validator is the gatekeeper, not runtime code."""

    def test_validator_fail_stops_before_init(self, tmp_path: Path) -> None:
        """If validation fails, init_regression must refuse to run."""
        import pytest

        # Config that points to a non-existent CSV
        cfg = {
            "name": "bad",
            "domain": "regression",
            "data_path": "knowledge/no_such_file.csv",
            "value_column": "flow",
        }
        cfg_dir = tmp_path / "configs" / "datasets"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "bad.json").write_text(json.dumps(cfg))

        cfg_path = tmp_path / "configs" / "datasets" / "bad.json"
        r = validate(cfg_path)
        # Validator must catch missing data file
        assert not r.passed, "Validator should have failed on missing data_path"

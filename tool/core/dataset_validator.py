"""core/dataset_validator.py — Dataset config + data validation.

`validate(config_path) -> ValidationReport` enforces the plug-and-play
contract documented in docs/DATA_CONTRACT.md. Called automatically by
run_managed_system.py and both init scripts; also exposed as a CLI:

    python scripts/validate_dataset.py --config pems_node1

A ValidationReport is a dataclass with:
  .passed  — bool; True if all checks pass
  .errors  — list[str] of blocking issues (empty when passed)
  .warnings— list[str] of advisory notes
  .summary — printable one-line status
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_TOOL_DIR = Path(__file__).resolve().parent.parent

# ── report ────────────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    config_name: str = ""
    passed: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.config_name}: "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )

    def print_table(self) -> None:
        print(self.summary)
        for e in self.errors:
            print(f"  ❌ {e}")
        for w in self.warnings:
            print(f"  ⚠  {w}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve(raw: str, base: Path) -> Path:
    """Resolve raw path: if absolute return as-is, else relative to base."""
    p = Path(raw)
    return p if p.is_absolute() else (base / p).resolve()


def _check_type(value: Any, expected: type, name: str, errors: list[str]) -> bool:
    if not isinstance(value, expected):
        errors.append(f"'{name}' must be {expected.__name__}, got {type(value).__name__}")
        return False
    return True


_REQUIRED_TOP = ["name", "domain"]
_OPTIONAL_NUMERIC = {
    "train_frac": (0.0, 1.0),
    "val_frac": (0.0, 1.0),
    "stream_delay_s": (0.0, 600.0),
    "alpha": (0.0, 1.0),
    "beta": (0.0, 1.0),
    "gamma": (0.0, 1.0),
}


# ── domain-specific validators ────────────────────────────────────────────────

def _validate_regression(cfg: dict, base: Path, report: ValidationReport) -> None:
    errors = report.errors
    warnings = report.warnings

    # data_path
    data_path_raw = cfg.get("data_path")
    if not data_path_raw:
        errors.append("'data_path' is required for regression domain")
        return
    data_path = _resolve(data_path_raw, base)
    if not data_path.exists():
        errors.append(f"data_path not found: {data_path}")
        return

    # value_column
    value_col = cfg.get("value_column")
    if not value_col:
        errors.append("'value_column' is required for regression domain")
        return

    # Parse CSV
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
    except Exception as exc:
        errors.append(f"Cannot read CSV at {data_path}: {exc}")
        return

    if value_col not in df.columns:
        errors.append(f"value_column '{value_col}' not found in CSV (columns: {list(df.columns)})")
        return

    # numeric dtype
    try:
        df[value_col] = pd.to_numeric(df[value_col], errors="raise")
    except Exception:
        errors.append(f"Column '{value_col}' contains non-numeric values")
        return

    # NaN check
    nan_count = int(df[value_col].isna().sum())
    if nan_count > 0:
        errors.append(
            f"Column '{value_col}' has {nan_count} NaN value(s). "
            "Impute or drop before running HarmonE (silent imputation is not performed)."
        )

    # Minimum length
    seq_len = int(cfg.get("seq_length", 5))
    horizon = int(cfg.get("horizon", 1))
    min_rows = max(100, (seq_len + horizon) * 10)
    if len(df) < min_rows:
        errors.append(
            f"Dataset has only {len(df)} rows; minimum is {min_rows} "
            f"(seq_length={seq_len}, horizon={horizon})."
        )

    # Optional: timestamp monotonicity
    if "timestamp" in df.columns:
        try:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            if ts.isna().any():
                warnings.append("'timestamp' column has unparseable values; time-based plots may break.")
            elif not ts.is_monotonic_increasing:
                warnings.append("'timestamp' column is not monotonically increasing — rows are not chronologically ordered.")
        except Exception:
            warnings.append("Could not parse 'timestamp' column for monotonicity check.")

    # model weights
    models_cfg = cfg.get("models", {})
    for m_name, m_cfg in models_cfg.items():
        w_raw = m_cfg.get("weights_path", "")
        if not w_raw:
            warnings.append(f"Model '{m_name}' has no weights_path declared.")
            continue
        w_path = _resolve(w_raw, base)
        if not w_path.exists():
            warnings.append(f"Model '{m_name}' weights not found: {w_path} (init will still succeed)")


def _validate_cv(cfg: dict, base: Path, report: ValidationReport) -> None:
    errors = report.errors
    warnings = report.warnings

    image_dir_raw = cfg.get("image_dir")
    manifest_raw = cfg.get("manifest_csv")

    if not image_dir_raw and not manifest_raw:
        errors.append("CV config must specify 'image_dir' or 'manifest_csv'.")
        return

    # image_dir check
    if image_dir_raw:
        image_dir = _resolve(image_dir_raw, base)
        if not image_dir.exists():
            warnings.append(
                f"image_dir not found: {image_dir}. "
                "Image-based drift detection and histograms will fail at runtime."
            )
        else:
            imgs = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
            if len(imgs) == 0:
                warnings.append(f"image_dir has no .jpg/.png files: {image_dir}")
            else:
                # Sample check up to 100 images
                sample = random.sample(imgs, min(100, len(imgs)))
                unreadable = []
                try:
                    from PIL import Image as _PILImage
                    for p in sample:
                        try:
                            _PILImage.open(p).verify()
                        except Exception:
                            unreadable.append(p.name)
                except ImportError:
                    warnings.append("Pillow not installed — cannot sample-check image integrity.")
                if unreadable:
                    errors.append(
                        f"{len(unreadable)} sampled images are unreadable: {unreadable[:5]}…"
                    )

    # manifest check
    if manifest_raw:
        manifest_path = _resolve(manifest_raw, base)
        if not manifest_path.exists():
            errors.append(f"manifest_csv not found: {manifest_path}")
        else:
            try:
                import pandas as pd
                mdf = pd.read_csv(manifest_path)
                if "image_path" not in mdf.columns:
                    errors.append(
                        f"manifest_csv must have 'image_path' column; found: {list(mdf.columns)}"
                    )
                else:
                    # Sample existence check
                    root = _resolve(cfg.get("data_root", "."), base)
                    sample = mdf["image_path"].dropna().sample(min(100, len(mdf))).tolist()
                    missing = [p for p in sample if not (root / p).exists()]
                    if missing:
                        errors.append(
                            f"{len(missing)}/{len(sample)} sampled image_path entries do not exist "
                            f"under data_root={root}. First missing: {missing[0]}"
                        )
            except Exception as exc:
                errors.append(f"Cannot read manifest_csv: {exc}")

    # model weights
    models_cfg = cfg.get("models", {})
    for m_name, m_cfg in models_cfg.items():
        w_raw = m_cfg.get("weights_path", "")
        if not w_raw:
            warnings.append(f"Model '{m_name}' has no weights_path declared.")
            continue
        w_path = _resolve(w_raw, base)
        if not w_path.exists():
            warnings.append(
                f"Model '{m_name}' weights not found: {w_path}. "
                "Inference will fail for this model."
            )


# ── public API ────────────────────────────────────────────────────────────────

def validate(config_path: str | Path, *, sample: bool = True) -> ValidationReport:
    """Validate a dataset config + data files.

    Args:
        config_path: Path to the JSON config (absolute, or relative to cwd).
        sample:      If True, sample-check image integrity for CV configs.

    Returns:
        ValidationReport (check .passed and .errors).
    """
    config_path = Path(config_path).resolve()
    report = ValidationReport(config_name=config_path.stem)

    if not config_path.exists():
        report.errors.append(f"Config file not found: {config_path}")
        return report

    # Parse JSON
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception as exc:
        report.errors.append(f"Cannot parse config JSON: {exc}")
        return report

    base = _TOOL_DIR  # paths in configs are relative to tool/

    # Top-level required keys
    for key in _REQUIRED_TOP:
        if key not in cfg:
            report.errors.append(f"Required config key '{key}' is missing")

    if report.errors:
        return report  # can't proceed without domain

    # Domain check
    domain = cfg.get("domain", "")
    if domain not in ("regression", "cv"):
        report.errors.append(f"'domain' must be 'regression' or 'cv', got '{domain}'")
        return report

    # Numeric range checks
    for key, (lo, hi) in _OPTIONAL_NUMERIC.items():
        if key in cfg:
            val = cfg[key]
            if not isinstance(val, (int, float)):
                report.errors.append(f"'{key}' must be numeric, got {type(val).__name__}")
            elif not (lo <= val <= hi):
                report.errors.append(f"'{key}'={val} out of range [{lo}, {hi}]")

    # train_frac + val_frac sanity
    train_frac = cfg.get("train_frac", 0.8)
    val_frac = cfg.get("val_frac", 0.0)
    if isinstance(train_frac, float) and isinstance(val_frac, float):
        if train_frac + val_frac > 1.0:
            report.errors.append(
                f"train_frac ({train_frac}) + val_frac ({val_frac}) > 1.0"
            )

    # Domain-specific
    if domain == "regression":
        _validate_regression(cfg, base, report)
    elif domain == "cv":
        _validate_cv(cfg, base, report)

    report.passed = len(report.errors) == 0
    return report

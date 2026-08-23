"""
tests/test_phase4_energy.py — Tests for core/energy.py (EnergyMeter abstraction).

Tests cover:
  - NullMeter: total_uJ is None; valid is False; result dict structure
  - Context manager: __enter__ returns self; __exit__ populates fields
  - from_thresholds: reads "energy_meter" key; defaults to "auto"
  - Backend fallback: RAPL unavailable → NullMeter behavior (no crash)
  - "auto" with unavailable RAPL → graceful degradation
  - Unknown backend → null fallback
  - GPU field absent when only CPU backend configured
"""

import sys
import os
import warnings

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Always run with null backend so tests work on Windows/CI without RAPL/NVML
import core.energy as energy_mod
from core.energy import EnergyMeter


# ── helpers ────────────────────────────────────────────────────────────────────

def _null_meter(label: str = "test") -> EnergyMeter:
    return EnergyMeter(label, backend="null")


# ── NullMeter ─────────────────────────────────────────────────────────────────

class TestNullMeter:
    def test_total_uJ_is_none(self):
        with _null_meter() as m:
            pass
        assert m.total_uJ is None

    def test_cpu_uJ_is_none(self):
        with _null_meter() as m:
            pass
        assert m.cpu_uJ is None

    def test_gpu_uJ_is_none(self):
        with _null_meter() as m:
            pass
        assert m.gpu_uJ is None

    def test_valid_is_false(self):
        with _null_meter() as m:
            pass
        assert m.valid is False

    def test_result_dict_structure(self):
        with _null_meter() as m:
            pass
        r = m.result
        required = {"cpu_uJ", "gpu_uJ", "total_uJ", "valid", "cpu_backend", "gpu_backend"}
        assert required.issubset(r.keys()), f"Missing keys: {required - r.keys()}"
        assert r["cpu_uJ"] is None
        assert r["gpu_uJ"] is None
        assert r["total_uJ"] is None
        assert r["valid"] is False
        assert isinstance(r["cpu_backend"], str)
        assert isinstance(r["gpu_backend"], str)

    def test_null_backend_no_crash_on_double_enter(self):
        m = EnergyMeter("t", backend="null")
        m.__enter__()
        m.__exit__()
        m.__enter__()
        m.__exit__()
        assert m.total_uJ is None


# ── Context manager semantics ──────────────────────────────────────────────────

class TestContextManager:
    def test_enter_returns_self(self):
        m = EnergyMeter("t", backend="null")
        returned = m.__enter__()
        m.__exit__()
        assert returned is m

    def test_with_block_populates_result(self):
        with EnergyMeter("t", backend="null") as m:
            x = 1 + 1  # noqa: F841
        assert m.result["valid"] is False  # null returns None

    def test_result_accessible_after_context(self):
        with EnergyMeter("t", backend="null") as m:
            pass
        # Should not raise
        _ = m.total_uJ
        _ = m.result

    def test_fields_none_before_exit(self):
        m = EnergyMeter("t", backend="null")
        # Before context: fields are None (initial state)
        assert m.cpu_uJ is None
        assert m.total_uJ is None

    def test_exception_inside_context_still_exits(self):
        try:
            with EnergyMeter("t", backend="null") as m:
                raise ValueError("oops")
        except ValueError:
            pass
        # __exit__ was still called
        assert m.result is not None


# ── from_thresholds factory ───────────────────────────────────────────────────

class TestFromThresholds:
    def test_reads_energy_meter_key(self):
        thresholds = {"energy_meter": "null"}
        m = EnergyMeter.from_thresholds("t", thresholds)
        with m:
            pass
        assert m.total_uJ is None  # null backend

    def test_defaults_to_auto_when_key_absent(self):
        # With no pyRAPL on Windows, auto falls back to null gracefully
        thresholds = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = EnergyMeter.from_thresholds("t", thresholds)
        with m:
            pass
        # total_uJ may be None (null fallback) or a float (if RAPL available)
        assert m.total_uJ is None or isinstance(m.total_uJ, float)

    def test_null_key_gives_null_meter(self):
        thresholds = {"energy_meter": "null", "min_score": 0.8}
        m = EnergyMeter.from_thresholds("t", thresholds)
        with m:
            pass
        assert not m.valid


# ── RAPL fallback ─────────────────────────────────────────────────────────────

class TestRaplFallback:
    def test_rapl_unavailable_does_not_crash(self, monkeypatch):
        """Simulate unavailable RAPL; backend should degrade to null."""
        # Inject a cached probe result that reports unavailable
        _unavailable = {"kind": "null", "available": False, "msg": "CPU energy backend: unavailable"}
        monkeypatch.setattr(energy_mod, "_cpu_probe", _unavailable)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = EnergyMeter("t", backend="rapl")
        with m:
            pass
        assert m.total_uJ is None
        assert m.valid is False

    def test_auto_rapl_unavailable_falls_back_to_null(self, monkeypatch):
        _unavailable = {"kind": "null", "available": False, "msg": "CPU energy backend: unavailable"}
        monkeypatch.setattr(energy_mod, "_cpu_probe", _unavailable)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = EnergyMeter("t", backend="auto")
        with m:
            pass
        assert m.total_uJ is None

    def test_rapl_unavailable_emits_warning(self, monkeypatch):
        _unavailable = {"kind": "null", "available": False, "msg": "CPU energy backend: unavailable"}
        monkeypatch.setattr(energy_mod, "_cpu_probe", _unavailable)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            EnergyMeter("t", backend="rapl")
        runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warns) >= 1
        assert "RAPL" in str(runtime_warns[0].message)


# ── Unknown backend ───────────────────────────────────────────────────────────

class TestUnknownBackend:
    def test_unknown_backend_warns_and_uses_null(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = EnergyMeter("t", backend="bogus_backend")
        with m:
            pass
        assert m.total_uJ is None
        warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert any("bogus_backend" in str(w.message) for w in warns)

    def test_unknown_backend_result_is_valid_dict(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = EnergyMeter("t", backend="bogus_backend")
        with m:
            pass
        r = m.result
        assert "total_uJ" in r
        assert "valid" in r
        assert "cpu_backend" in r
        assert "gpu_backend" in r


# ── GPU field absent for CPU-only config ─────────────────────────────────────

class TestGpuFieldAbsent:
    def test_null_backend_gpu_is_none(self):
        with EnergyMeter("t", backend="null") as m:
            pass
        assert m.gpu_uJ is None
        assert m.result["gpu_backend"] == "null"

    def test_rapl_backend_gpu_is_none(self, monkeypatch):
        _unavailable = {"kind": "null", "available": False, "msg": "CPU energy backend: unavailable"}
        monkeypatch.setattr(energy_mod, "_cpu_probe", _unavailable)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with EnergyMeter("t", backend="rapl") as m:
                pass
        assert m.gpu_uJ is None

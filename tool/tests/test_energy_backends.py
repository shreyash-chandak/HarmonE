"""tests/test_energy_backends.py — Regression tests for energy measurement bugs.

Covers the exact failure modes that existed before the fix:
  A: Sub-ms CPU workload legitimately returns 0 (not treated as failure)
  B: Longer CPU workload returns positive delta
  C: GPU NVML path does NOT use _PollingGPUBackend
  D: NVML unavailable → gpu_uJ=None, not 0
  E: CPU unavailable does not prevent GPU from working
  F: GPU unavailable does not prevent CPU from working
  G: Both available → components reported separately; total = sum

Run from tool/:
    python -m pytest tests/test_energy_backends.py -v
"""

from __future__ import annotations

import math
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))


class TestEnergyMeterInterface(unittest.TestCase):
    """Core invariant tests that do not require real hardware."""

    def setUp(self):
        # Reset probe cache before each test so patches take effect
        import core.energy as e
        e._cpu_probe = None
        e._gpu_probe = None
        e._ACTIVE = False

    def tearDown(self):
        import core.energy as e
        e._ACTIVE = False

    # ── Case A: sub-ms CPU workload → valid zero ──────────────────────────────

    def test_A_zero_cpu_delta_is_valid(self):
        """A zero hardware delta must be valid=True, not None."""
        from core.energy import EnergyMeter, _NullBackend

        class _ZeroCpuBackend:
            name = "fake_rapl_zero"
            def start(self): pass
            def stop(self): return 0.0  # valid measurement, counter did not advance

        import core.energy as e
        e._cpu_probe = {"kind": "fake", "available": True, "msg": ""}
        with patch.object(e, "_make_cpu_backend", return_value=_ZeroCpuBackend()):
            with EnergyMeter("case_a", backend="rapl") as em:
                pass
        self.assertIsNotNone(em.cpu_uJ, "cpu_uJ must not be None for a valid zero reading")
        self.assertEqual(em.cpu_uJ, 0.0)
        self.assertTrue(em.cpu_valid)
        self.assertIsNotNone(em.total_uJ)
        self.assertTrue(em.valid)

    # ── Case B: longer workload → positive delta ──────────────────────────────

    def test_B_longer_cpu_workload_positive(self):
        """Backend returning a positive value must be fully propagated."""
        from core.energy import EnergyMeter

        class _PosCpuBackend:
            name = "fake_rapl_positive"
            def start(self): pass
            def stop(self): return 54321.0  # µJ

        import core.energy as e
        e._cpu_probe = {"kind": "fake", "available": True, "msg": ""}
        with patch.object(e, "_make_cpu_backend", return_value=_PosCpuBackend()):
            with EnergyMeter("case_b", backend="rapl") as em:
                pass
        self.assertEqual(em.cpu_uJ, 54321.0)
        self.assertTrue(em.cpu_valid)
        self.assertEqual(em.total_uJ, 54321.0)

    # ── Case C: nvml backend must not use _PollingGPUBackend ─────────────────

    def test_C_nvml_backend_does_not_use_polling(self):
        """backend='nvml' must never instantiate _PollingGPUBackend."""
        from core.energy import EnergyMeter, _PollingGPUBackend

        class _FakeNvmlBackend:
            name = "fake_nvml"
            def start(self): pass
            def stop(self): return 1_000_000.0  # µJ

        import core.energy as e
        e._cpu_probe = {"kind": "null", "available": False, "msg": ""}
        e._gpu_probe = {"kind": "fake_nvml", "available": True, "gpu_name": "FakeGPU", "msg": ""}
        with patch.object(e, "_make_cpu_backend", return_value=e._NullBackend()):
            with patch.object(e, "_make_gpu_backend", return_value=_FakeNvmlBackend()):
                with EnergyMeter("case_c", backend="nvml") as em:
                    pass
        self.assertNotIsInstance(em._gpu_backend, _PollingGPUBackend,
                                 "nvml backend activated _PollingGPUBackend — must not happen")
        self.assertTrue(em.gpu_valid)
        self.assertEqual(em.gpu_uJ, 1_000_000.0)

    # ── Case D: NVML unavailable → gpu_uJ=None, not 0 ────────────────────────

    def test_D_nvml_unavailable_is_None_not_zero(self):
        """When GPU probe fails, gpu_uJ must be None, never 0.0."""
        import core.energy as e
        e._cpu_probe = {"kind": "null", "available": False, "msg": ""}
        e._gpu_probe = {"kind": "null", "available": False, "gpu_name": None, "msg": ""}
        from core.energy import EnergyMeter
        with patch.object(e, "_make_cpu_backend", return_value=e._NullBackend()):
            with patch.object(e, "_make_gpu_backend", return_value=e._NullBackend()):
                with EnergyMeter("case_d", backend="nvml") as em:
                    pass
        self.assertIsNone(em.gpu_uJ, "unavailable GPU must produce None, not 0")
        self.assertFalse(em.gpu_valid)
        self.assertIsNone(em.total_uJ, "no components measured → total must be None")
        self.assertFalse(em.valid)

    # ── Case E: CPU unavailable, GPU works ────────────────────────────────────

    def test_E_gpu_works_independently_of_cpu(self):
        """GPU measurement proceeds even when CPU backend is unavailable."""

        class _FakeGpu:
            name = "fake_gpu"
            def start(self): pass
            def stop(self): return 987_654.0

        import core.energy as e
        e._cpu_probe = {"kind": "null", "available": False, "msg": ""}
        e._gpu_probe = {"kind": "fake", "available": True, "gpu_name": "FakeGPU", "msg": ""}
        from core.energy import EnergyMeter
        with patch.object(e, "_make_cpu_backend", return_value=e._NullBackend()):
            with patch.object(e, "_make_gpu_backend", return_value=_FakeGpu()):
                with EnergyMeter("case_e", backend="nvml") as em:
                    pass
        self.assertIsNone(em.cpu_uJ)
        self.assertFalse(em.cpu_valid)
        self.assertEqual(em.gpu_uJ, 987_654.0)
        self.assertTrue(em.gpu_valid)
        self.assertEqual(em.total_uJ, 987_654.0)  # GPU-only total
        self.assertTrue(em.valid)
        self.assertFalse(em.total_complete)  # CPU was requested but missing

    # ── Case F: GPU unavailable, CPU works ────────────────────────────────────

    def test_F_cpu_works_independently_of_gpu(self):
        """CPU measurement proceeds even when GPU backend is unavailable."""

        class _FakeCpu:
            name = "fake_cpu"
            def start(self): pass
            def stop(self): return 12_345.0

        import core.energy as e
        e._cpu_probe = {"kind": "fake", "available": True, "msg": ""}
        e._gpu_probe = {"kind": "null", "available": False, "gpu_name": None, "msg": ""}
        from core.energy import EnergyMeter
        with patch.object(e, "_make_cpu_backend", return_value=_FakeCpu()):
            with patch.object(e, "_make_gpu_backend", return_value=e._NullBackend()):
                with EnergyMeter("case_f", backend="nvml") as em:
                    pass
        self.assertEqual(em.cpu_uJ, 12_345.0)
        self.assertTrue(em.cpu_valid)
        self.assertIsNone(em.gpu_uJ)
        self.assertFalse(em.gpu_valid)
        self.assertEqual(em.total_uJ, 12_345.0)  # CPU-only total
        self.assertFalse(em.total_complete)

    # ── Case G: both available → separate + sum ───────────────────────────────

    def test_G_both_available_sum(self):
        """When both backends provide readings, total = cpu + gpu."""

        class _FakeCpu:
            name = "fake_cpu"
            def start(self): pass
            def stop(self): return 10_000.0

        class _FakeGpu:
            name = "fake_gpu"
            def start(self): pass
            def stop(self): return 500_000.0

        import core.energy as e
        e._cpu_probe = {"kind": "fake", "available": True, "msg": ""}
        e._gpu_probe = {"kind": "fake", "available": True, "gpu_name": "FakeGPU", "msg": ""}
        from core.energy import EnergyMeter
        with patch.object(e, "_make_cpu_backend", return_value=_FakeCpu()):
            with patch.object(e, "_make_gpu_backend", return_value=_FakeGpu()):
                with EnergyMeter("case_g", backend="nvml") as em:
                    pass
        self.assertEqual(em.cpu_uJ, 10_000.0)
        self.assertEqual(em.gpu_uJ, 500_000.0)
        self.assertAlmostEqual(em.total_uJ, 510_000.0)
        self.assertTrue(em.total_complete)
        self.assertTrue(em.valid)
        # result dict
        r = em.result
        self.assertEqual(r["cpu_uJ"], 10_000.0)
        self.assertEqual(r["gpu_uJ"], 500_000.0)
        self.assertTrue(r["total_complete"])

    # ── Nesting guard ─────────────────────────────────────────────────────────

    def test_nesting_raises(self):
        import core.energy as e
        e._cpu_probe = {"kind": "null", "available": False, "msg": ""}
        from core.energy import EnergyMeter
        with self.assertRaises(RuntimeError):
            with EnergyMeter("outer", backend="null") as _outer:
                with EnergyMeter("inner", backend="null") as _inner:
                    pass

    # ── GPU units: mJ → µJ conversion ────────────────────────────────────────

    def test_pyjoules_nvml_unit_conversion(self):
        """_PyJoulesNvmlBackend must multiply mJ by 1000 to produce µJ."""
        import core.energy as e

        # Mock pyJoules NvidiaGPUDevice to return 1000 mJ
        fake_sample = MagicMock()
        fake_sample.energy = {"nvidia_gpu_0": 1000}  # 1000 mJ = 1,000,000 µJ
        fake_trace = [fake_sample]

        fake_meter = MagicMock()
        fake_meter.get_trace.return_value = fake_trace

        fake_device = MagicMock()
        fake_domains = [MagicMock()]

        with patch.dict("sys.modules", {
            "pyJoules": MagicMock(__version__="0.5.1"),
            "pyJoules.device": MagicMock(),
            "pyJoules.device.nvidia_device": MagicMock(
                NvidiaGPUDevice=MagicMock(
                    return_value=fake_device,
                    available_domains=MagicMock(return_value=fake_domains),
                ),
                NvidiaGPUDomain=MagicMock(),
            ),
            "pyJoules.energy_meter": MagicMock(
                EnergyMeter=MagicMock(return_value=fake_meter),
            ),
        }):
            import importlib
            # Reload backend with mocked imports
            from core.energy import _PyJoulesNvmlBackend as _Backend

            backend = _Backend.__new__(_Backend)
            backend._device = fake_device
            backend._PyEM = MagicMock(return_value=fake_meter)
            backend._meter = fake_meter

            result = backend.stop()

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1_000_000.0, places=0,
                               msg="pyJoules GPU energy must be converted from mJ to µJ (×1000)")

    # ── Negative GPU delta → None ─────────────────────────────────────────────

    def test_negative_gpu_delta_returns_None(self):
        """A negative NVML delta (counter reset) must produce None, not a positive number."""
        from core.energy import _DirectNvmlBackend

        backend = _DirectNvmlBackend.__new__(_DirectNvmlBackend)
        backend._before_mJ = 5000
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetTotalEnergyConsumption.return_value = 100  # < before → negative delta
        backend._nvml = mock_nvml
        backend._handle = MagicMock()

        result = backend.stop()
        self.assertIsNone(result, "negative NVML delta must return None, not a converted negative value")


class TestRaplApiCorrectness(unittest.TestCase):
    """Verify the pyJoules RAPL API usage is correct (no AttributeError)."""

    def setUp(self):
        import core.energy as e
        e._cpu_probe = None
        e._gpu_probe = None
        e._ACTIVE = False

    def test_rapl_passes_device_not_domain(self):
        """EnergyMeter must be constructed with Device objects, not Domain objects.

        The AttributeError in the old code was caused by passing RaplPackageDomain
        directly to EnergyMeter instead of wrapping it in a RaplDevice first.
        Verify the new code uses the correct pattern.
        """
        import inspect
        from core.energy import _PyJoulesRaplBackend
        src = inspect.getsource(_PyJoulesRaplBackend.__init__)
        # Must use RaplDevice, not bare RaplPackageDomain passed to EnergyMeter
        self.assertIn("RaplDevice", src,
                      "Backend must construct a RaplDevice object")
        self.assertIn("configure", src,
                      "Backend must call device.configure() with domain list")
        # The EnergyMeter call must pass the device, not the domain
        start_src = inspect.getsource(_PyJoulesRaplBackend.start)
        self.assertIn("devices=", start_src,
                      "EnergyMeter must be called with devices= kwarg containing Device objects")

    def test_no_domain_passed_directly_to_energymeter(self):
        """Old bug: EnergyMeter([RaplPackageDomain(0)]) → AttributeError.
        New code: EnergyMeter(devices=[rapl_device_instance]).
        """
        import inspect
        from core.energy import _PyJoulesRaplBackend
        start_src = inspect.getsource(_PyJoulesRaplBackend.start)
        # Must not pass RaplPackageDomain directly to EnergyMeter
        self.assertNotIn("RaplPackageDomain", start_src,
                         "RaplPackageDomain must not be passed directly to EnergyMeter in start()")


if __name__ == "__main__":
    unittest.main()

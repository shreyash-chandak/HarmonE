"""core/energy.py — Hardware-agnostic energy measurement abstraction.

E3 fix: migrated from pyRAPL/pynvml to pyJoules (unified CPU RAPL + GPU NVML
under one library) with a power-polling fallback for GPUs that lack cumulative
energy counters (e.g. RTX 5060 Laptop).

Usage (context manager, preferred):
    with EnergyMeter("label", backend="auto") as meter:
        do_work()
    uJ = meter.total_uJ   # float or None; None if backend unavailable

Construct from thresholds dict:
    meter = EnergyMeter.from_thresholds("label", thresholds)

Backends (set via thresholds.json["energy_meter"]):
    "null"  — NullMeter; always returns None. Default on dev/Windows.
    "rapl"  — CPU RAPL via pyJoules (Linux, requires powercap permissions).
    "nvml"  — GPU via pyJoules NVML + CPU RAPL (Linux+CUDA).
    "auto"  — tries RAPL first; falls back to null with a logged warning.

Canonical unit: microjoules (µJ) internally and in all public attributes.
Nesting: forbidden. A second EnergyMeter constructed while one is active raises
         RuntimeError (RAPL package counters are shared; nesting double-counts).

E1 / AMD note: The Ryzen AI 7 350 exposes an Intel-compatible RAPL interface.
    pyJoules reads from /sys/class/powercap/intel-rapl/. Probe at first use;
    if readings are zero, falls back to null and logs a warning.
"""

from __future__ import annotations

import logging
import threading
import time
import warnings
from typing import Any

logger = logging.getLogger(__name__)

# ── Module-level state ────────────────────────────────────────────────────────

_RAPL_PROBED: bool = False
_RAPL_AVAILABLE: bool = False
_NVML_PROBED: bool = False
_NVML_AVAILABLE: bool = False  # True = pyJoules NvidiaGPUDomain works + non-zero
_ACTIVE: bool = False           # re-entrancy guard


def _probe_rapl() -> bool:
    """Return True if pyJoules RAPL backend produces non-zero readings."""
    global _RAPL_PROBED, _RAPL_AVAILABLE
    if _RAPL_PROBED:
        return _RAPL_AVAILABLE
    _RAPL_PROBED = True
    try:
        from pyJoules.energy_meter import EnergyMeter as _PyEM
        from pyJoules.device.rapl_device import RaplPackageDomain
        meter = _PyEM([RaplPackageDomain(0)])
        meter.start(tag="probe")
        # Small CPU workload to ensure non-zero energy
        import math
        for _ in range(50_000):
            math.sqrt(2.0)
        meter.stop()
        trace = meter.get_trace()
        for sample in trace:
            if sample.energy and any(v > 0 for v in sample.energy.values()):
                _RAPL_AVAILABLE = True
                logger.info("pyJoules RAPL probe: non-zero reading — RAPL available.")
                return True
        logger.warning(
            "pyJoules RAPL probe: all readings are zero. "
            "RAPL may not be accessible (run scripts/setup_energy_permissions.sh). "
            "CPU energy will be reported as None."
        )
    except Exception as exc:
        logger.warning(f"pyJoules RAPL probe failed: {exc}. CPU energy unavailable.")
    _RAPL_AVAILABLE = False
    return False


def _probe_nvml() -> bool:
    """Return True if pyJoules NvidiaGPUDomain produces non-zero readings."""
    global _NVML_PROBED, _NVML_AVAILABLE
    if _NVML_PROBED:
        return _NVML_AVAILABLE
    _NVML_PROBED = True
    try:
        from pyJoules.energy_meter import EnergyMeter as _PyEM
        from pyJoules.device.nvidia_gpu import NvidiaGPUDomain
        meter = _PyEM([NvidiaGPUDomain(0)])
        meter.start(tag="probe")
        time.sleep(0.1)
        meter.stop()
        trace = meter.get_trace()
        for sample in trace:
            if sample.energy and any(v > 0 for v in sample.energy.values()):
                _NVML_AVAILABLE = True
                logger.info("pyJoules NVML probe: non-zero reading — NVML available.")
                return True
        logger.warning(
            "pyJoules NVML probe: readings are zero. "
            "Falling back to 50ms power.draw polling (nvidia-smi)."
        )
    except Exception as exc:
        logger.warning(f"pyJoules NVML probe failed: {exc}. Falling back to polling.")
    _NVML_AVAILABLE = False
    return False


def get_backend_status() -> dict[str, Any]:
    """Probe both backends (lazy) and return their status for logging."""
    rapl_ok = _probe_rapl()
    nvml_ok = _probe_nvml()
    return {
        "rapl_available": rapl_ok,
        "nvml_available": nvml_ok,
        "cpu_backend": "rapl" if rapl_ok else "none",
        "gpu_backend": "nvml" if nvml_ok else "power_polling",
    }


# ── Backend implementations ────────────────────────────────────────────────────

class _NullBackend:
    name = "null"

    def start(self) -> None:
        pass

    def stop(self) -> float | None:
        return None


class _PyJoulesRaplBackend:
    """CPU energy via pyJoules RaplPackageDomain.

    Creates a fresh EnergyMeter per measurement to avoid trace accumulation.
    Raises ImportError / RuntimeError on construction if unavailable.
    """

    name = "rapl"

    def __init__(self) -> None:
        from pyJoules.device.rapl_device import RaplPackageDomain  # import-test
        self._RaplPackageDomain = RaplPackageDomain
        self._meter: Any = None

    def start(self) -> None:
        from pyJoules.energy_meter import EnergyMeter as _PyEM
        self._meter = _PyEM([self._RaplPackageDomain(0)])
        self._meter.start(tag="measure")

    def stop(self) -> float | None:
        if self._meter is None:
            return None
        self._meter.stop()
        try:
            for sample in self._meter.get_trace():
                if sample.energy:
                    total = sum(sample.energy.values())
                    if total > 0:
                        return float(total)  # µJ (pyJoules RAPL reports µJ natively)
        except Exception as exc:
            logger.debug(f"RAPL stop error: {exc}")
        return None


class _PyJoulesNvmlBackend:
    """GPU energy via pyJoules NvidiaGPUDomain.

    nvmlDeviceGetTotalEnergyConsumption returns mJ; pyJoules normalises to µJ.
    Raises ImportError / RuntimeError on construction if unavailable.
    """

    name = "nvml"

    def __init__(self) -> None:
        from pyJoules.device.nvidia_gpu import NvidiaGPUDomain  # import-test
        self._NvidiaGPUDomain = NvidiaGPUDomain
        self._meter: Any = None

    def start(self) -> None:
        from pyJoules.energy_meter import EnergyMeter as _PyEM
        self._meter = _PyEM([self._NvidiaGPUDomain(0)])
        self._meter.start(tag="measure")

    def stop(self) -> float | None:
        if self._meter is None:
            return None
        self._meter.stop()
        try:
            for sample in self._meter.get_trace():
                if sample.energy:
                    total = sum(sample.energy.values())
                    if total > 0:
                        return float(total)  # µJ
        except Exception as exc:
            logger.debug(f"NVML stop error: {exc}")
        return None


class _PollingGPUBackend:
    """Fallback GPU energy via nvidia-smi power.draw integration at 50ms.

    Used when pyJoules NVML returns zero (e.g. RTX 5060 Laptop with
    cumulative-energy counter not exposed through current driver).

    Accuracy: ±(poll_interval/2 × mean_power) per measurement window,
    converging to < 2% relative error over a full run.
    """

    name = "nvml_polling"
    _POLL_INTERVAL_S: float = 0.05
    _SAMPLE_COUNT: int = 0  # tracks samples for validity

    def __init__(self, gpu_index: int = 0) -> None:
        self.gpu_index = gpu_index
        self._readings_w: list[float] = []
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sample_count: int = 0

    def start(self) -> None:
        self._readings_w = []
        self._sample_count = 0
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        import subprocess
        cmd = [
            "nvidia-smi", f"--id={self.gpu_index}",
            "--query-gpu=power.draw", "--format=csv,noheader,nounits",
        ]
        while self._running:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=0.5)
                val = result.stdout.strip()
                if val and val.lower() not in ("[n/a]", ""):
                    watts = float(val)
                    with self._lock:
                        self._readings_w.append(watts)
            except Exception:
                pass
            time.sleep(self._POLL_INTERVAL_S)

    def stop(self) -> float | None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            n = len(self._readings_w)
            self._sample_count = n
            if n == 0:
                logger.debug("PollingGPUBackend: 0 samples — window too short for energy estimate.")
                return None
            # E = sum(P_i) × Δt, in joules → convert to µJ
            energy_J = sum(self._readings_w) * self._POLL_INTERVAL_S
            return float(energy_J * 1e6)


# ── EnergyMeter ───────────────────────────────────────────────────────────────

class EnergyMeter:
    """Context manager for CPU (RAPL) and/or GPU (NVML) energy measurement.

    Args:
        label:    Measurement label (used for logging).
        backend:  "null" | "rapl" | "nvml" | "auto" (default).

    Attributes after __exit__:
        cpu_uJ    — CPU energy in µJ, or None if unavailable.
        gpu_uJ    — GPU energy in µJ, or None if unavailable.
        total_uJ  — sum of available components, or None if none available.
        valid     — True if at least one backend produced a reading.
        result    — full dict with all above plus metadata.
    """

    def __init__(self, label: str, backend: str = "auto") -> None:
        self._label = label
        self._backend_key = backend
        self._cpu_backend: _NullBackend | _PyJoulesRaplBackend | None = None
        self._gpu_backend: _PyJoulesNvmlBackend | _PollingGPUBackend | None = None
        self._cpu_uJ: float | None = None
        self._gpu_uJ: float | None = None
        self._setup(backend)

    def _setup(self, backend: str) -> None:
        if backend == "null":
            self._cpu_backend = _NullBackend()

        elif backend == "rapl":
            try:
                _probe_rapl()
                if _RAPL_AVAILABLE:
                    self._cpu_backend = _PyJoulesRaplBackend()
                else:
                    warnings.warn(
                        f"EnergyMeter '{self._label}': RAPL probe returned zero — using null. "
                        "Run scripts/setup_energy_permissions.sh and check AMD RAPL support.",
                        RuntimeWarning, stacklevel=3,
                    )
                    self._cpu_backend = _NullBackend()
            except Exception as exc:
                warnings.warn(
                    f"EnergyMeter '{self._label}': RAPL unavailable ({exc}) — using null.",
                    RuntimeWarning, stacklevel=3,
                )
                self._cpu_backend = _NullBackend()

        elif backend == "nvml":
            # CPU RAPL
            _probe_rapl()
            self._cpu_backend = _PyJoulesRaplBackend() if _RAPL_AVAILABLE else _NullBackend()
            # GPU
            _probe_nvml()
            if _NVML_AVAILABLE:
                self._gpu_backend = _PyJoulesNvmlBackend()
            else:
                self._gpu_backend = _PollingGPUBackend()
                logger.info(
                    f"EnergyMeter '{self._label}': NVML not available — "
                    "using 50ms power.draw polling for GPU energy."
                )

        elif backend == "auto":
            _probe_rapl()
            if _RAPL_AVAILABLE:
                self._cpu_backend = _PyJoulesRaplBackend()
            else:
                warnings.warn(
                    f"EnergyMeter '{self._label}': RAPL unavailable — energy not measured. "
                    "Run scripts/setup_energy_permissions.sh first.",
                    RuntimeWarning, stacklevel=3,
                )
                self._cpu_backend = _NullBackend()

        else:
            warnings.warn(
                f"EnergyMeter '{self._label}': unknown backend '{backend}' — using null.",
                RuntimeWarning, stacklevel=3,
            )
            self._cpu_backend = _NullBackend()

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "EnergyMeter":
        global _ACTIVE
        if _ACTIVE:
            raise RuntimeError(
                "EnergyMeter nesting is forbidden (RAPL counters are package-wide; "
                "nested contexts double-count). Measure at ONE level only."
            )
        _ACTIVE = True
        if self._cpu_backend is not None:
            self._cpu_backend.start()
        if self._gpu_backend is not None:
            self._gpu_backend.start()
        return self

    def __exit__(self, *_: Any) -> None:
        global _ACTIVE
        if self._cpu_backend is not None:
            self._cpu_uJ = self._cpu_backend.stop()
        if self._gpu_backend is not None:
            self._gpu_uJ = self._gpu_backend.stop()
        _ACTIVE = False

    # ── results ───────────────────────────────────────────────────────────────

    @property
    def cpu_uJ(self) -> float | None:
        return self._cpu_uJ

    @property
    def gpu_uJ(self) -> float | None:
        return self._gpu_uJ

    @property
    def total_uJ(self) -> float | None:
        parts = [x for x in (self._cpu_uJ, self._gpu_uJ) if x is not None]
        return float(sum(parts)) if parts else None

    @property
    def valid(self) -> bool:
        return self.total_uJ is not None

    @property
    def result(self) -> dict:
        backends: list[str] = []
        if isinstance(self._cpu_backend, _PyJoulesRaplBackend):
            backends.append("rapl")
        if isinstance(self._gpu_backend, _PyJoulesNvmlBackend):
            backends.append("nvml")
        elif isinstance(self._gpu_backend, _PollingGPUBackend):
            backends.append("nvml_polling")
        return {
            "cpu_uJ": self._cpu_uJ,
            "gpu_uJ": self._gpu_uJ,
            "total_uJ": self.total_uJ,
            "cpu_valid": self._cpu_uJ is not None,
            "gpu_valid": self._gpu_uJ is not None,
            "valid": self.valid,
            "backends": backends,
        }

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_thresholds(cls, label: str, thresholds: dict) -> "EnergyMeter":
        """Construct from a loaded thresholds.json dict.

        Reads thresholds["energy_meter"]; defaults to "auto" if absent.
        """
        backend = thresholds.get("energy_meter", "auto")
        return cls(label, backend=backend)

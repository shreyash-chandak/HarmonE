"""core/energy.py — Hardware-agnostic energy measurement abstraction.

CPU and GPU are probed and managed independently.  A failure in one does NOT
affect the other.  The key architectural invariant:

    cpu_uJ = None  →  CPU measurement unavailable (backend failed / no perms)
    cpu_uJ = 0.0   →  valid measurement; hardware counter did not advance
    cpu_uJ > 0.0   →  valid measurement with non-zero energy

Same invariant applies to gpu_uJ.  Never silently treat an unavailable backend
as "zero energy".

Backend configuration (set via thresholds.json["energy_meter"]):
    "null"     — NullMeter; always returns None.
    "rapl"     — CPU via Intel RAPL (pyJoules or sysfs fallback); no GPU.
    "nvml"     — CPU via RAPL + GPU via NVML cumulative-energy counter.
    "auto"     — tries RAPL for CPU; no GPU; falls back to null with a warning.
    "polling"  — EXPERIMENTAL: CPU via RAPL + GPU via 50 ms nvidia-smi polling.
                 Only use when NVML cumulative counter is unavailable.

CPU probe order:
    1. pyJoules RaplDevice (correct 0.5.1 API: Device, not Domain, passed to EnergyMeter)
    2. Direct sysfs powercap read (no pyJoules required)
    3. /proc/driver/amd_energy (AMD kernel module)
    4. Null (CPU measurement unavailable)

GPU probe order (backend "nvml" only):
    1. pyJoules NvidiaGPUDevice (nvidia_device module, cumulative mJ counter)
    2. pynvml direct (same NVML API, no pyJoules wrapper)
    3. Null (GPU measurement unavailable)

Units: all public attributes use microjoules (µJ).
    pyJoules RAPL:  reads energy_uj sysfs → already µJ.
    pyJoules GPU:   nvmlDeviceGetTotalEnergyConsumption returns mJ → ×1000 → µJ.
    pynvml direct:  same NVML function → mJ → ×1000 → µJ.

Nesting: forbidden.  A second EnergyMeter constructed while one is active
raises RuntimeError (RAPL package counters are shared).
"""

from __future__ import annotations

import logging
import threading
import time
import warnings
from typing import Any

logger = logging.getLogger(__name__)

# ── Module-level probe cache (each probed once per process) ───────────────────

_cpu_probe: dict | None = None   # set by _probe_cpu(); {kind, available, msg}
_gpu_probe: dict | None = None   # set by _probe_gpu(); {kind, available, msg}
_ACTIVE: bool = False            # re-entrancy guard

# ── sysfs helpers ─────────────────────────────────────────────────────────────

_RAPL_SYSFS_CANDIDATES = [
    "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
    "/sys/class/powercap/intel-rapl:0/energy_uj",
    "/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj",
    "/sys/class/powercap/intel-rapl:1/energy_uj",
]


def _read_rapl_sysfs_uj() -> float | None:
    import math
    from pathlib import Path as _P
    for p in _RAPL_SYSFS_CANDIDATES:
        try:
            val = float(_P(p).read_text().strip())
            if not math.isnan(val):
                return val
        except (OSError, ValueError):
            continue
    return None


def read_package_counter() -> dict | None:
    """Passive read of the CPU package RAPL counter (sysfs): {"uJ", "max_range_uJ"}.

    Used for the whole-stream package energy (2026-10-07): read once when the
    stream starts and once when the run ends, outside any EnergyMeter (so it
    never conflicts with the meter re-entrancy guard or the energy lock).
    Returns None when RAPL is not readable.
    """
    from pathlib import Path as _P
    path = _rapl_sysfs_path()
    if path is None:
        return None
    try:
        val = float(_P(path).read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        max_range = float((_P(path).parent / "max_energy_range_uj").read_text().strip())
    except (OSError, ValueError):
        max_range = None
    return {"uJ": val, "max_range_uJ": max_range}


def package_energy_between(start: dict | None, end: dict | None) -> float | None:
    """Energy (µJ) between two read_package_counter() readings, correcting one
    counter wrap. None if either reading is missing."""
    if not start or not end:
        return None
    delta = end["uJ"] - start["uJ"]
    if delta < 0 and start.get("max_range_uJ"):
        delta += start["max_range_uJ"]
    return delta if delta >= 0 else None


class Pkg0Meter:
    """The original HarmonE's energy reading (2026-10-09): pyRAPL's
    Measurement.begin()/end() with result.pkg[0], i.e. the socket-0 CPU
    package counter (intel-rapl:0/energy_uj, the same sysfs file pyRAPL reads)
    read at begin() and at end(); the reading is the difference in µJ. No DRAM,
    no GPU, no cross-process lock. One addition: a single counter wrap inside
    the window is corrected (pyRAPL would report a negative value).
    end() returns None when RAPL is not readable.
    """

    def __init__(self) -> None:
        from pathlib import Path as _P
        path = _rapl_sysfs_path()
        self._file = _P(path) if path else None
        self._max_range: float | None = None
        if self._file is not None:
            try:
                self._max_range = float((self._file.parent / "max_energy_range_uj").read_text().strip())
            except (OSError, ValueError):
                pass
        self._begin: float | None = None

    def _read(self) -> float | None:
        if self._file is None:
            return None
        try:
            return float(self._file.read_text())
        except (OSError, ValueError):
            return None

    def begin(self) -> None:
        self._begin = self._read()

    def end(self) -> float | None:
        after = self._read()
        if after is None or self._begin is None:
            return None
        delta = after - self._begin
        if delta < 0 and self._max_range:
            delta += self._max_range
        return delta if delta >= 0 else None


def _rapl_sysfs_path() -> str | None:
    from pathlib import Path as _P
    for p in _RAPL_SYSFS_CANDIDATES:
        if _P(p).exists():
            try:
                _P(p).read_text()
                return p
            except OSError:
                continue
    return None


# ── CPU probe ─────────────────────────────────────────────────────────────────

def _probe_cpu() -> dict:
    """Probe CPU RAPL backends once; cache result.

    Returns {kind: str, available: bool, msg: str}.
    """
    global _cpu_probe
    if _cpu_probe is not None:
        return _cpu_probe

    import math
    from pathlib import Path as _P

    def _workload():
        acc = 0.0
        for i in range(100_000):
            acc += math.sqrt(float(i) + 1.0)

    # ── Path 1: pyJoules RaplDevice (correct 0.5.1 API) ──────────────────────
    # The API takes Device objects, not Domain objects.
    # EnergyMeter(devices=[RaplDevice()]) — NOT EnergyMeter([RaplPackageDomain(0)])
    try:
        from pyJoules.device.rapl_device import RaplDevice, RaplPackageDomain
        from pyJoules.energy_meter import EnergyMeter as _PyEM
        domains = RaplDevice.available_package_domains()
        if domains:
            device = RaplDevice()
            device.configure(domains=domains)
            meter = _PyEM(devices=[device])
            meter.start(tag="probe")
            _workload()
            meter.stop()
            trace = meter.get_trace()
            for sample in trace:
                if sample.energy and any(v > 0 for v in sample.energy.values()):
                    _cpu_probe = {"kind": "pyjoules_rapl", "available": True,
                                  "msg": "CPU energy backend: pyJoules RAPL"}
                    logger.info("CPU energy backend: pyJoules RAPL (RaplDevice)")
                    return _cpu_probe
            # Domains exist but all readings were zero → sysfs is readable but
            # workload window too short; still consider pyJoules available.
            _cpu_probe = {"kind": "pyjoules_rapl", "available": True,
                          "msg": "CPU energy backend: pyJoules RAPL (zero probe — counter may need time)"}
            logger.info("CPU energy backend: pyJoules RAPL (probe read zero; counter resolution low)")
            return _cpu_probe
    except PermissionError:
        logger.debug("CPU probe path 1 (pyJoules RAPL): permission denied on energy_uj — trying sysfs fallback")
    except Exception as exc:
        logger.debug("CPU probe path 1 (pyJoules RAPL): %s — trying sysfs fallback", exc)

    # ── Path 2: direct sysfs powercap ────────────────────────────────────────
    try:
        before = _read_rapl_sysfs_uj()
        if before is not None:
            _workload()
            after = _read_rapl_sysfs_uj()
            if after is not None:
                delta = after - before
                if delta < 0:
                    sysfs_p = _rapl_sysfs_path()
                    if sysfs_p:
                        max_p = _P(sysfs_p).parent / "max_energy_range_uj"
                        if max_p.exists():
                            delta += float(max_p.read_text().strip())
                _cpu_probe = {"kind": "sysfs_direct", "available": True,
                              "msg": "CPU energy backend: sysfs RAPL fallback"}
                logger.info("CPU energy backend: sysfs RAPL fallback (pyJoules unavailable or denied)")
                return _cpu_probe
    except Exception as exc:
        logger.debug("CPU probe path 2 (sysfs direct): %s", exc)

    # ── Path 3: /proc/driver/amd_energy ──────────────────────────────────────
    try:
        proc_path = _P("/proc/driver/amd_energy")
        if proc_path.exists():
            def _read_amd() -> float | None:
                lines = proc_path.read_text().splitlines()
                total = 0.0
                found = False
                for line in lines:
                    if "socket" in line.lower() or "package" in line.lower():
                        for part in line.split():
                            try:
                                total += float(part)
                                found = True
                                break
                            except ValueError:
                                continue
                return total if found else None
            before = _read_amd()
            if before is not None:
                _workload()
                after = _read_amd()
                if after is not None and (after - before) > 0:
                    _cpu_probe = {"kind": "amd_proc", "available": True,
                                  "msg": "CPU energy backend: AMD /proc energy"}
                    logger.info("CPU energy backend: /proc/driver/amd_energy (AMD kernel module)")
                    return _cpu_probe
    except Exception as exc:
        logger.debug("CPU probe path 3 (/proc/driver/amd_energy): %s", exc)

    logger.warning(
        "CPU energy measurement unavailable. "
        "Run: sudo bash scripts/setup_energy_permissions.sh  (once per boot)"
    )
    _cpu_probe = {"kind": "null", "available": False, "msg": "CPU energy backend: unavailable"}
    return _cpu_probe


# ── GPU probe ─────────────────────────────────────────────────────────────────

def _probe_gpu() -> dict:
    """Probe GPU NVML backends once; cache result.

    Returns {kind: str, available: bool, msg: str, gpu_name: str | None}.
    """
    global _gpu_probe
    if _gpu_probe is not None:
        return _gpu_probe

    # ── Path 1: pyJoules NvidiaGPUDevice ─────────────────────────────────────
    # Module is nvidia_device, NOT nvidia_gpu.
    # pyJoules GPU values are in mJ (from nvmlDeviceGetTotalEnergyConsumption).
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", FutureWarning)  # suppress pynvml deprecation noise
            from pyJoules.device.nvidia_device import NvidiaGPUDevice, NvidiaGPUDomain
        import pynvml as _nvml
        _nvml.nvmlInit()
        n_gpus = _nvml.nvmlDeviceGetCount()
        if n_gpus == 0:
            raise RuntimeError("No NVIDIA GPUs detected by NVML")
        handle = _nvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = _nvml.nvmlDeviceGetName(handle)
        # Verify cumulative energy counter works
        e_before = _nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        time.sleep(0.1)
        e_after = _nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        if e_after < e_before:
            raise RuntimeError("NVML energy counter went backwards (driver reset?)")
        # Now verify pyJoules wrapper produces the same reading
        domains = NvidiaGPUDevice.available_domains()
        device = NvidiaGPUDevice()
        device.configure(domains=domains)
        from pyJoules.energy_meter import EnergyMeter as _PyEM
        meter = _PyEM(devices=[device])
        meter.start(tag="probe")
        time.sleep(0.1)
        meter.stop()
        trace = meter.get_trace()
        pyj_ok = any(
            v >= 0 for s in trace for v in s.energy.values()
        )
        if pyj_ok:
            _gpu_probe = {
                "kind": "pyjoules_nvml", "available": True,
                "gpu_name": gpu_name,
                "msg": f"GPU energy backend: pyJoules NVML ({gpu_name})",
            }
            logger.info("GPU energy backend: pyJoules NVML  GPU=%s", gpu_name)
            return _gpu_probe
    except ImportError as exc:
        logger.debug("GPU probe path 1 (pyJoules nvidia_device): import failed: %s — trying pynvml direct", exc)
    except Exception as exc:
        logger.debug("GPU probe path 1 (pyJoules nvidia_device): %s — trying pynvml direct", exc)

    # ── Path 2: pynvml direct (no pyJoules wrapper) ───────────────────────────
    try:
        import pynvml as _nvml
        _nvml.nvmlInit()
        n_gpus = _nvml.nvmlDeviceGetCount()
        if n_gpus == 0:
            raise RuntimeError("No NVIDIA GPUs detected by NVML")
        handle = _nvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = _nvml.nvmlDeviceGetName(handle)
        e_before = _nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        time.sleep(0.1)
        e_after = _nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        if e_after < e_before:
            raise RuntimeError("NVML energy counter went backwards")
        _gpu_probe = {
            "kind": "pynvml_direct", "available": True,
            "gpu_name": gpu_name,
            "msg": f"GPU energy backend: pynvml direct ({gpu_name})",
        }
        logger.info("GPU energy backend: pynvml direct  GPU=%s  (pyJoules nvidia_device unavailable)", gpu_name)
        return _gpu_probe
    except ImportError:
        logger.debug("GPU probe path 2 (pynvml direct): pynvml not installed")
    except Exception as exc:
        logger.debug("GPU probe path 2 (pynvml direct): %s", exc)

    logger.warning(
        "GPU energy measurement unavailable. "
        "Check: pip install pyJoules[nvidia]  and that the NVIDIA driver/NVML is accessible."
    )
    _gpu_probe = {"kind": "null", "available": False, "gpu_name": None,
                  "msg": "GPU energy backend: unavailable"}
    return _gpu_probe


def get_backend_status() -> dict[str, Any]:
    """Probe both backends lazily and return status dict for logging."""
    cpu = _probe_cpu()
    gpu = _probe_gpu()
    return {
        "cpu_available": cpu["available"],
        "cpu_kind": cpu["kind"],
        "gpu_available": gpu["available"],
        "gpu_kind": gpu["kind"],
        "gpu_name": gpu.get("gpu_name"),
    }


# ── Backend implementations ────────────────────────────────────────────────────

class _NullBackend:
    name = "null"

    def start(self) -> None:
        pass

    def stop(self) -> float | None:
        return None


class _PyJoulesRaplBackend:
    """CPU energy via pyJoules RaplDevice (correct 0.5.1 API)."""
    name = "pyjoules_rapl"

    def __init__(self) -> None:
        import warnings as _w
        from pyJoules.device.rapl_device import RaplDevice
        from pyJoules.energy_meter import EnergyMeter as _PyEM
        domains = RaplDevice.available_package_domains()
        if not domains:
            raise RuntimeError("No RAPL package domains available")
        self._device = RaplDevice()
        self._device.configure(domains=domains)
        self._PyEM = _PyEM
        self._meter: Any = None

    @staticmethod
    def _max_range_uj() -> float | None:
        path = _rapl_sysfs_path()
        if path is None:
            return None
        try:
            from pathlib import Path as _P
            return float((_P(path).parent / "max_energy_range_uj").read_text().strip())
        except Exception:
            return None

    def start(self) -> None:
        self._meter = self._PyEM(devices=[self._device])
        self._meter.start(tag="measure")

    def stop(self) -> float | None:
        if self._meter is None:
            return None
        self._meter.stop()
        try:
            for sample in self._meter.get_trace():
                if sample.energy:
                    # Per-domain wraparound correction (audit C6), mirroring
                    # _SysfsRaplBackend: a negative delta straddled one
                    # counter wrap, so add the counter range back.
                    wrap = self._max_range_uj()
                    vals = [v + wrap if v < 0 and wrap else v for v in sample.energy.values()]
                    if any(v < 0 for v in vals):
                        logger.warning("pyJoules RAPL: negative energy delta after wrap correction — reading dropped.")
                        return None
                    # RAPL values are µJ; total may be 0 (valid, counter resolution)
                    return float(sum(vals))
        except Exception as exc:
            logger.debug("pyJoules RAPL stop error: %s", exc)
        return None


class _SysfsRaplBackend:
    """CPU energy via direct sysfs powercap read — no pyJoules required."""
    name = "sysfs_direct"

    def __init__(self) -> None:
        from pathlib import Path as _P
        self._path: str | None = _rapl_sysfs_path()
        if self._path is None:
            raise RuntimeError("No readable sysfs powercap energy_uj path found.")
        self._max_path: str | None = None
        max_p = _P(self._path).parent / "max_energy_range_uj"
        if max_p.exists():
            self._max_path = str(max_p)
        self._before: float | None = None

    def start(self) -> None:
        try:
            self._before = _read_rapl_sysfs_uj()
        except Exception:
            self._before = None

    def stop(self) -> float | None:
        try:
            after = _read_rapl_sysfs_uj()
            if after is None or self._before is None:
                return None
            delta = after - self._before
            if delta < 0 and self._max_path:
                from pathlib import Path as _P
                delta += float(_P(self._max_path).read_text().strip())
            if delta < 0:
                logger.warning("SysfsRaplBackend: negative energy delta after wrap correction — reading dropped.")
                return None
            return float(delta)
        except Exception as exc:
            logger.debug("SysfsRaplBackend stop error: %s", exc)
            return None


class _AmdProcRaplBackend:
    """CPU energy via /proc/driver/amd_energy (AMD kernel module, kernel ≥5.8)."""
    name = "amd_proc"
    _PROC_PATH = "/proc/driver/amd_energy"

    def __init__(self) -> None:
        from pathlib import Path as _P
        self._path = _P(self._PROC_PATH)
        if not self._path.exists():
            raise RuntimeError(f"{self._PROC_PATH} not found — amd_energy module not loaded.")
        self._before: float | None = None

    def _read(self) -> float | None:
        lines = self._path.read_text().splitlines()
        total = 0.0
        found = False
        for line in lines:
            if "socket" in line.lower() or "package" in line.lower():
                for part in line.split():
                    try:
                        total += float(part)
                        found = True
                        break
                    except ValueError:
                        continue
        return total if found else None

    def start(self) -> None:
        try:
            self._before = self._read()
        except Exception:
            self._before = None

    def stop(self) -> float | None:
        try:
            after = self._read()
            if after is None or self._before is None:
                return None
            delta = after - self._before
            return float(delta) if delta >= 0 else None
        except Exception as exc:
            logger.debug("AmdProcRaplBackend stop error: %s", exc)
            return None


class _PyJoulesNvmlBackend:
    """GPU energy via pyJoules NvidiaGPUDevice.

    nvmlDeviceGetTotalEnergyConsumption returns mJ; converted here to µJ (×1000).
    A negative delta (driver counter reset) is treated as an invalid measurement.
    """
    name = "pyjoules_nvml"

    def __init__(self) -> None:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", FutureWarning)
            from pyJoules.device.nvidia_device import NvidiaGPUDevice
        from pyJoules.energy_meter import EnergyMeter as _PyEM
        domains = NvidiaGPUDevice.available_domains()
        if not domains:
            raise RuntimeError("No NVIDIA GPU domains available")
        self._device = NvidiaGPUDevice()
        self._device.configure(domains=domains)
        self._PyEM = _PyEM
        self._meter: Any = None

    @staticmethod
    def _max_range_uj() -> float | None:
        path = _rapl_sysfs_path()
        if path is None:
            return None
        try:
            from pathlib import Path as _P
            return float((_P(path).parent / "max_energy_range_uj").read_text().strip())
        except Exception:
            return None

    def start(self) -> None:
        self._meter = self._PyEM(devices=[self._device])
        self._meter.start(tag="measure")

    def stop(self) -> float | None:
        if self._meter is None:
            return None
        self._meter.stop()
        try:
            for sample in self._meter.get_trace():
                if sample.energy:
                    total_mJ = sum(sample.energy.values())
                    if total_mJ < 0:
                        # Counter reset or driver anomaly — treat as invalid
                        logger.debug("pyJoules NVML: negative energy delta (%.0f mJ) — skipping", total_mJ)
                        return None
                    return float(total_mJ) * 1000.0  # mJ → µJ
        except Exception as exc:
            logger.debug("pyJoules NVML stop error: %s", exc)
        return None


class _DirectNvmlBackend:
    """GPU energy via pynvml directly (no pyJoules wrapper).

    Used when pyJoules nvidia_device is unavailable but pynvml is installed.
    Same unit semantics: nvmlDeviceGetTotalEnergyConsumption → mJ → µJ.
    """
    name = "pynvml_direct"

    def __init__(self, gpu_index: int = 0) -> None:
        import pynvml as _nvml
        _nvml.nvmlInit()
        self._handle = _nvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._nvml = _nvml
        self._before_mJ: int | None = None

    def start(self) -> None:
        try:
            self._before_mJ = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        except Exception:
            self._before_mJ = None

    def stop(self) -> float | None:
        try:
            after_mJ = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            if self._before_mJ is None:
                return None
            delta_mJ = after_mJ - self._before_mJ
            if delta_mJ < 0:
                logger.debug("pynvml direct: negative energy delta (%.0f mJ) — skipping", delta_mJ)
                return None
            return float(delta_mJ) * 1000.0  # mJ → µJ
        except Exception as exc:
            logger.debug("pynvml direct stop error: %s", exc)
            return None


class _PollingGPUBackend:
    """EXPERIMENTAL GPU fallback: 50 ms nvidia-smi power.draw integration.

    Only used when backend="polling" is explicitly configured.
    NOT an implicit fallback for "nvml" — configure explicitly if needed.

    Accuracy is limited by poll interval and window duration.
    A short inference window may produce 0 samples and return None.
    """
    name = "nvml_polling"
    _POLL_INTERVAL_S: float = 0.05

    def __init__(self, gpu_index: int = 0) -> None:
        self.gpu_index = gpu_index
        self._readings_w: list[float] = []
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._readings_w = []
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        import subprocess
        cmd = [
            "nvidia-smi", f"--id={self.gpu_index}",
            "--query-gpu=power.draw", "--format=csv,noheader,nounits",
        ]
        _logged = False
        while self._running:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=0.5)
                val = result.stdout.strip()
                if val and val.lower() not in ("[n/a]", "n/a", ""):
                    with self._lock:
                        self._readings_w.append(float(val))
                elif not _logged and val.lower() in ("[n/a]", "n/a"):
                    logger.debug("PollingGPUBackend: nvidia-smi power.draw returned N/A")
                    _logged = True
            except FileNotFoundError:
                if not _logged:
                    logger.debug("PollingGPUBackend: nvidia-smi not found")
                    _logged = True
            except Exception as exc:
                if not _logged:
                    logger.debug("PollingGPUBackend: nvidia-smi error: %s", exc)
                    _logged = True
            time.sleep(self._POLL_INTERVAL_S)

    def stop(self) -> float | None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            n = len(self._readings_w)
            if n == 0:
                logger.debug("PollingGPUBackend: 0 samples — window too short for energy estimate")
                return None
            energy_J = sum(self._readings_w) * self._POLL_INTERVAL_S
            return float(energy_J * 1e6)  # J → µJ


# ── Backend factory helpers ────────────────────────────────────────────────────

def _make_cpu_backend() -> Any:
    """Create the best available CPU backend based on probe result."""
    probe = _probe_cpu()
    kind = probe["kind"]
    if not probe["available"]:
        return _NullBackend()
    try:
        if kind == "pyjoules_rapl":
            return _PyJoulesRaplBackend()
        if kind == "sysfs_direct":
            return _SysfsRaplBackend()
        if kind == "amd_proc":
            return _AmdProcRaplBackend()
    except Exception as exc:
        logger.debug("Could not construct %s backend: %s — using null", kind, exc)
    return _NullBackend()


def _make_gpu_backend() -> Any:
    """Create the best available GPU backend based on probe result."""
    probe = _probe_gpu()
    kind = probe["kind"]
    if not probe["available"]:
        return _NullBackend()
    try:
        if kind == "pyjoules_nvml":
            return _PyJoulesNvmlBackend()
        if kind == "pynvml_direct":
            return _DirectNvmlBackend()
    except Exception as exc:
        logger.debug("Could not construct %s backend: %s — using null", kind, exc)
    return _NullBackend()


# ── EnergyMeter ───────────────────────────────────────────────────────────────

class EnergyMeter:
    """Context manager for CPU (RAPL) and/or GPU (NVML) energy measurement.

    Args:
        label:    Measurement label (used for logging/provenance).
        backend:  "null" | "rapl" | "nvml" | "auto" | "polling" (see module docstring).

    Attributes after __exit__:
        cpu_uJ    — CPU energy in µJ; None if CPU measurement unavailable.
        gpu_uJ    — GPU energy in µJ; None if GPU measurement unavailable or not requested.
        total_uJ  — sum of available components; None if nothing measured.
        cpu_valid — True if CPU backend ran and produced a reading (even if 0).
        gpu_valid — True if GPU backend ran and produced a reading (even if 0).
        total_complete — True if every requested component produced a valid reading.
        valid     — True if total_uJ is not None (at least one component measured).

    Unit invariant:
        None  ≠  0.0
        None  → measurement unavailable (backend missing, permissions denied, etc.)
        0.0   → valid measurement; hardware counter did not advance in this interval
    """

    def __init__(self, label: str, backend: str = "auto") -> None:
        self._label = label
        self._backend_key = backend
        self._cpu_backend: Any = _NullBackend()
        self._gpu_backend: Any = _NullBackend()
        self._cpu_requested: bool = False
        self._gpu_requested: bool = False
        self._cpu_uJ: float | None = None
        self._gpu_uJ: float | None = None
        self._setup(backend)

    def _setup(self, backend: str) -> None:
        if backend == "null":
            pass  # both remain NullBackend

        elif backend == "rapl":
            self._cpu_requested = True
            _probe_cpu()
            self._cpu_backend = _make_cpu_backend()
            if _cpu_probe and not _cpu_probe["available"]:
                warnings.warn(
                    f"EnergyMeter '{self._label}': RAPL unavailable — energy not measured. "
                    "Run scripts/setup_energy_permissions.sh first.",
                    RuntimeWarning, stacklevel=3,
                )

        elif backend == "nvml":
            # CPU and GPU initialized independently — one failure does not affect the other
            self._cpu_requested = True
            self._gpu_requested = True
            _probe_cpu()
            self._cpu_backend = _make_cpu_backend()
            _probe_gpu()
            self._gpu_backend = _make_gpu_backend()

        elif backend == "auto":
            self._cpu_requested = True
            _probe_cpu()
            if _cpu_probe and _cpu_probe["available"]:
                self._cpu_backend = _make_cpu_backend()
            else:
                warnings.warn(
                    f"EnergyMeter '{self._label}': RAPL unavailable — energy not measured. "
                    "Run scripts/setup_energy_permissions.sh first.",
                    RuntimeWarning, stacklevel=3,
                )

        elif backend == "polling":
            # Explicit opt-in for 50 ms nvidia-smi polling
            self._cpu_requested = True
            self._gpu_requested = True
            _probe_cpu()
            self._cpu_backend = _make_cpu_backend()
            self._gpu_backend = _PollingGPUBackend()
            logger.info(
                "EnergyMeter '%s': using experimental 50 ms nvidia-smi polling for GPU.", self._label
            )

        else:
            warnings.warn(
                f"EnergyMeter '{self._label}': unknown backend '{backend}' — using null.",
                RuntimeWarning, stacklevel=3,
            )

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "EnergyMeter":
        global _ACTIVE
        if _ACTIVE:
            raise RuntimeError(
                "EnergyMeter nesting is forbidden (RAPL counters are package-wide; "
                "nested contexts double-count). Measure at ONE level only."
            )
        _ACTIVE = True
        self._cpu_backend.start()
        self._gpu_backend.start()
        return self

    def __exit__(self, *_: Any) -> None:
        global _ACTIVE
        self._cpu_uJ = self._cpu_backend.stop()
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
    def cpu_valid(self) -> bool:
        return self._cpu_uJ is not None

    @property
    def gpu_valid(self) -> bool:
        return self._gpu_uJ is not None

    @property
    def total_uJ(self) -> float | None:
        parts = [x for x in (self._cpu_uJ, self._gpu_uJ) if x is not None]
        return float(sum(parts)) if parts else None

    @property
    def total_complete(self) -> bool:
        """True only if every requested component produced a valid reading."""
        if self._cpu_requested and not self.cpu_valid:
            return False
        if self._gpu_requested and not self.gpu_valid:
            return False
        return True

    @property
    def valid(self) -> bool:
        return self.total_uJ is not None

    @property
    def result(self) -> dict:
        return {
            "cpu_uJ": self._cpu_uJ,
            "gpu_uJ": self._gpu_uJ,
            "total_uJ": self.total_uJ,
            "cpu_valid": self.cpu_valid,
            "gpu_valid": self.gpu_valid,
            "total_complete": self.total_complete,
            "valid": self.valid,
            "cpu_backend": self._cpu_backend.name,
            "gpu_backend": self._gpu_backend.name,
        }

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_thresholds(cls, label: str, thresholds: dict) -> "EnergyMeter":
        backend = thresholds.get("energy_meter", "auto")
        return cls(label, backend=backend)

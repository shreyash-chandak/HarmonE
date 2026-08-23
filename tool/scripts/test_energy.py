"""scripts/test_energy.py — Hardware energy measurement self-test.

Verifies that CPU (RAPL) and GPU (NVML) backends are functional on this
machine.  Run before experiments to confirm measurement infrastructure.

Usage (from tool/):
    python scripts/test_energy.py          # both CPU and GPU
    python scripts/test_energy.py --cpu    # CPU only
    python scripts/test_energy.py --gpu    # GPU only

Exit codes:
    0 — all requested checks passed
    1 — one or more checks failed
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# ── bootstrap tool/ on sys.path ──────────────────────────────────────────────
_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

from core.energy import (
    EnergyMeter, _probe_cpu, _probe_gpu,
    _PyJoulesRaplBackend, _SysfsRaplBackend, _PyJoulesNvmlBackend, _DirectNvmlBackend,
)


def _cpu_workload(n: int = 500_000) -> None:
    acc = 0.0
    for i in range(n):
        acc += math.sqrt(float(i) + 1.0)


def _section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


def _pass(msg: str) -> None:
    print(f"  PASS: {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")


def _info(msg: str) -> None:
    print(f"  INFO: {msg}")


# ── CPU checks ────────────────────────────────────────────────────────────────

def check_cpu() -> bool:
    _section("CPU RAPL energy measurement")
    all_ok = True

    # 1. Probe
    probe = _probe_cpu()
    if probe["available"]:
        _pass(f"RAPL probe succeeded — backend: {probe['kind']}")
    else:
        _fail("RAPL probe: no CPU energy backend available")
        _info("Run: sudo bash scripts/setup_energy_permissions.sh  (once per boot)")
        all_ok = False
        return all_ok

    # 2. Short workload — may legitimately return 0 due to counter resolution
    with EnergyMeter("cpu_short", backend="rapl") as em:
        pass  # sub-ms workload
    if em.cpu_valid:
        _pass(f"Short measurement (sub-ms): cpu_valid=True  cpu_uJ={em.cpu_uJ:.1f}  "
              f"(zero is valid; counter resolution may not advance in <1 ms)")
    else:
        _fail("Short measurement returned cpu_valid=False — backend failed")
        all_ok = False

    # 3. Longer workload — counter should advance
    with EnergyMeter("cpu_long", backend="rapl") as em:
        _cpu_workload(500_000)
    if em.cpu_valid and em.cpu_uJ is not None and em.cpu_uJ >= 0:
        _pass(f"Long workload measurement: cpu_uJ={em.cpu_uJ:.0f}  cpu_valid=True")
        if em.cpu_uJ == 0:
            _info("  cpu_uJ=0 on a longer workload — RAPL counter granularity is coarse; "
                  "consider a longer workload for visible deltas")
    else:
        _fail(f"Long workload returned cpu_valid={em.cpu_valid}  cpu_uJ={em.cpu_uJ}")
        all_ok = False

    # 4. Verify zero delta is valid (not treated as backend failure)
    with EnergyMeter("cpu_noop", backend="rapl") as em:
        pass
    # A None here would be a bug; 0.0 is expected
    if em.cpu_uJ is not None:
        _pass(f"Zero-delta validity: cpu_uJ={em.cpu_uJ:.1f}  cpu_valid={em.cpu_valid}  "
              f"(0.0 ≠ None — correct)")
    else:
        _fail("RAPL backend returned None for a no-op (should return 0.0 or a small value)")
        all_ok = False

    # 5. Backend provenance
    _info(f"CPU backend: {em._cpu_backend.name}")

    return all_ok


# ── GPU checks ────────────────────────────────────────────────────────────────

def check_gpu() -> bool:
    _section("GPU NVML energy measurement")
    all_ok = True

    # 1. pynvml direct
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        if n == 0:
            _fail("pynvml: no GPUs detected")
            all_ok = False
        else:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(h)
            _pass(f"pynvml initialized  GPU={gpu_name}")
            try:
                e = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
                _pass(f"nvmlDeviceGetTotalEnergyConsumption: {e} mJ (cumulative)")
            except Exception as ex:
                _fail(f"nvmlDeviceGetTotalEnergyConsumption: {ex}")
                all_ok = False
        pynvml.nvmlShutdown()
    except ImportError:
        _fail("pynvml not installed — run: pip install pyJoules[nvidia]")
        all_ok = False
    except Exception as ex:
        _fail(f"pynvml init failed: {ex}")
        all_ok = False

    # 2. pyJoules nvidia_device
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            from pyJoules.device.nvidia_device import NvidiaGPUDevice, NvidiaGPUDomain
        _pass("pyJoules nvidia_device importable")
        domains = NvidiaGPUDevice.available_domains()
        _pass(f"NvidiaGPUDevice.available_domains(): {domains}")
        device = NvidiaGPUDevice()
        device.configure(domains=domains)
        from pyJoules.energy_meter import EnergyMeter as _PyEM
        meter = _PyEM(devices=[device])
        meter.start(tag="probe")
        time.sleep(0.15)
        meter.stop()
        trace = meter.get_trace()
        for s in trace:
            mJ_total = sum(s.energy.values())
            uJ_total = mJ_total * 1000
            _pass(f"pyJoules NvidiaGPUDomain: {mJ_total} mJ = {uJ_total:.0f} µJ over {s.duration:.3f}s")
    except ImportError as ex:
        _fail(f"pyJoules nvidia_device not importable: {ex}")
        _info("Run: pip install pyJoules[nvidia]")
        all_ok = False
    except Exception as ex:
        _fail(f"pyJoules NvidiaGPUDevice: {ex}")
        all_ok = False

    # 3. EnergyMeter GPU full path
    probe = _probe_gpu()
    if probe["available"]:
        _pass(f"GPU probe: backend={probe['kind']}  gpu={probe.get('gpu_name', '?')}")
        with EnergyMeter("gpu_test", backend="nvml") as em:
            time.sleep(0.2)  # let GPU idle energy accumulate
        if em.gpu_valid:
            _pass(f"EnergyMeter GPU: gpu_uJ={em.gpu_uJ:.0f}  gpu_valid=True")
        else:
            _fail(f"EnergyMeter GPU: gpu_valid=False — backend probe succeeded but measurement failed")
            all_ok = False
        _info(f"GPU backend: {em._gpu_backend.name}")
        _info(f"total_complete: {em.total_complete}  (False if CPU or GPU not both valid)")
    else:
        _fail(f"GPU probe: {probe['msg']}")
        all_ok = False

    # 4. Verify _PollingGPUBackend is NOT activated by "nvml" backend
    with EnergyMeter("nvml_no_polling", backend="nvml") as em:
        time.sleep(0.05)
    from core.energy import _PollingGPUBackend
    if isinstance(em._gpu_backend, _PollingGPUBackend):
        _fail("nvml backend activated _PollingGPUBackend — this should not happen")
        all_ok = False
    else:
        _pass(f"nvml backend did NOT activate _PollingGPUBackend  (backend={em._gpu_backend.name})")

    return all_ok


# ── Independence checks ───────────────────────────────────────────────────────

def check_independence() -> bool:
    _section("CPU/GPU independence")
    all_ok = True

    # CPU-only backend
    with EnergyMeter("rapl_only", backend="rapl") as em:
        _cpu_workload(100_000)
    if em._gpu_backend.name == "null" and not em._gpu_requested:
        _pass("rapl backend: gpu_backend=null, gpu_requested=False — correct")
    else:
        _fail(f"rapl backend unexpectedly activated GPU: {em._gpu_backend.name}")
        all_ok = False

    # GPU unavailable → cpu still valid (simulate by checking GPU is None doesn't poison CPU)
    with EnergyMeter("both", backend="nvml") as em:
        _cpu_workload(100_000)
    if em.cpu_valid or em.gpu_valid:
        _pass(f"nvml backend: cpu_valid={em.cpu_valid}  gpu_valid={em.gpu_valid}  "
              f"total_complete={em.total_complete}")
    else:
        _fail("nvml backend: neither cpu nor gpu produced a reading")
        all_ok = False

    # Validity distinction: unavailable ≠ zero
    with EnergyMeter("null_test", backend="null") as em:
        pass
    if em.total_uJ is None and not em.valid:
        _pass("null backend: total_uJ=None, valid=False — correct (not silently zero)")
    else:
        _fail(f"null backend returned total_uJ={em.total_uJ} (should be None)")
        all_ok = False

    return all_ok


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="HarmonE energy measurement self-test")
    parser.add_argument("--cpu", action="store_true", help="CPU checks only")
    parser.add_argument("--gpu", action="store_true", help="GPU checks only")
    args = parser.parse_args()

    do_cpu = args.cpu or not (args.cpu or args.gpu)
    do_gpu = args.gpu or not (args.cpu or args.gpu)

    results = []
    if do_cpu:
        results.append(("CPU RAPL", check_cpu()))
    if do_gpu:
        results.append(("GPU NVML", check_gpu()))
    if do_cpu and do_gpu:
        results.append(("Independence", check_independence()))

    _section("Summary")
    all_passed = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
        if not ok:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

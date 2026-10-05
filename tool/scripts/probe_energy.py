"""scripts/probe_energy.py — E1/E2 energy backend probe.

Runs on the target hardware to verify which pyJoules backends produce
non-zero readings, then writes the result to knowledge/.energy_backends.json
(keyed by hostname so the result survives reboots without re-probing on a
different machine).

Run once (or with --force to re-probe) before any live experiment:

    cd tool/
    python3 scripts/probe_energy.py          # probe all
    python3 scripts/probe_energy.py --cpu    # CPU RAPL only
    python3 scripts/probe_energy.py --gpu    # GPU NVML only
    python3 scripts/probe_energy.py --force  # ignore cached result

The probe results are also printed so they can be manually recorded in
DECISIONS_PENDING.md when an Outcome B (zeros/errors) occurs.

Exit codes:
  0 — at least one backend available
  1 — all probes failed / returned zero (action required)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

_TOOL_DIR = Path(__file__).resolve().parent.parent
_CACHE_FILE = _TOOL_DIR / "knowledge" / ".energy_backends.json"

# ── helpers ──────────────────────────────────────────────────────────────────

def _hostname_key() -> str:
    return f"{socket.gethostname()}:{platform.node()}"


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(data: dict) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(data, indent=2))


# ── E1: CPU RAPL probe ────────────────────────────────────────────────────────

def probe_cpu_rapl() -> dict:
    """Probe pyJoules RAPL backend (E1).

    Returns a dict with keys: available, reading_uJ, notes.
    """
    result: dict = {"backend": "rapl", "available": False, "reading_uJ": None, "notes": ""}
    try:
        from pyJoules.energy_meter import EnergyMeter as _PyEM
        from pyJoules.device.rapl_device import RaplPackageDomain
    except ImportError as exc:
        result["notes"] = f"pyJoules import failed: {exc}"
        return result

    try:
        meter = _PyEM([RaplPackageDomain(0)])
        meter.start(tag="e1_probe")

        # Intentional CPU workload — enough to guarantee non-zero delta
        acc = 0.0
        for i in range(100_000):
            acc += math.sqrt(float(i) + 1.0)

        meter.stop()
        trace = meter.get_trace()

        total_uJ = 0.0
        for sample in trace:
            if sample.energy:
                total_uJ += sum(sample.energy.values())

        result["reading_uJ"] = total_uJ
        if total_uJ > 0:
            result["available"] = True
            result["notes"] = (
                f"Non-zero reading: {total_uJ:.1f} µJ. "
                "RAPL is accessible — CPU energy measurement active."
            )
        else:
            result["notes"] = (
                "All RAPL readings are zero. RAPL interface may exist but is unreadable. "
                "Try: sudo modprobe msr && sudo chmod -R 777 /sys/class/powercap/intel-rapl/. "
                "See scripts/setup_energy_permissions.sh."
            )
    except Exception as exc:
        result["notes"] = f"RAPL probe exception: {type(exc).__name__}: {exc}"

    return result


# ── E2: GPU NVML probe ────────────────────────────────────────────────────────

def probe_gpu_nvml() -> dict:
    """Probe pyJoules NVML backend (E2).

    Returns a dict with keys: available, reading_uJ, notes.
    Falls back to pynvml power polling if NVML energy counter returns zero.
    """
    result: dict = {
        "backend": "nvml",
        "available": False,
        "reading_uJ": None,
        "fallback": None,
        "notes": "",
    }

    # — primary: pyJoules NvidiaGPUDomain —
    try:
        from pyJoules.energy_meter import EnergyMeter as _PyEM
        from pyJoules.device.nvidia_gpu import NvidiaGPUDomain
    except ImportError as exc:
        result["notes"] = f"pyJoules GPU import failed: {exc}"
        # Fall through to pynvml
    else:
        try:
            meter = _PyEM([NvidiaGPUDomain(0)])
            meter.start(tag="e2_probe_nvml")
            time.sleep(0.5)  # idle window — NVML energy counters still tick
            meter.stop()
            trace = meter.get_trace()

            total_uJ = 0.0
            for sample in trace:
                if sample.energy:
                    total_uJ += sum(sample.energy.values())

            result["reading_uJ"] = total_uJ
            if total_uJ > 0:
                result["available"] = True
                result["backend"] = "nvml"
                result["notes"] = (
                    f"NVML cumulative energy counter: {total_uJ:.1f} µJ over 0.5 s. "
                    "GPU energy measurement active (pyJoules NvidiaGPUDomain)."
                )
                return result
            else:
                result["notes"] = (
                    "NVML energy counter returned zero — counter may not be exposed on this GPU "
                    "(common on RTX 5060 Laptop / driver < 535). "
                    "Falling back to pynvml power.draw polling."
                )
        except Exception as exc:
            result["notes"] = f"pyJoules NVML probe exception: {exc}. Falling back to pynvml."

    # — fallback A: pynvml power polling —
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)

        # Collect 10 power samples × 50ms
        readings_mW: list[float] = []
        for _ in range(10):
            readings_mW.append(pynvml.nvmlDeviceGetPowerUsage(handle))  # mW
            time.sleep(0.05)
        pynvml.nvmlShutdown()

        avg_w = sum(readings_mW) / len(readings_mW) / 1000.0  # → W
        energy_uJ = avg_w * (10 * 0.05) * 1e6  # over 0.5 s window → µJ

        result["backend"] = "pynvml_polling"
        result["fallback"] = "pynvml"
        result["reading_uJ"] = energy_uJ
        result["available"] = avg_w > 0
        result["notes"] = (
            f"pynvml power.draw polling: avg {avg_w:.2f} W on {gpu_name}. "
            "PollingGPUBackend will be used (50 ms integration). "
            f"Estimated accuracy: ±{50/1000/2*avg_w*1e6:.0f} µJ per 50ms window."
        )
        return result
    except Exception as exc:
        result["fallback"] = "pynvml_failed"
        result["notes"] += f" pynvml fallback also failed: {exc}."

    # — fallback B: nvidia-smi subprocess —
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        raw = proc.stdout.strip()
        if raw and raw.lower() not in ("[n/a]", ""):
            watts = float(raw)
            result["backend"] = "nvml_subprocess"
            result["fallback"] = "nvidia_smi"
            result["reading_uJ"] = watts * 0.5 * 1e6  # rough 0.5 s window
            result["available"] = watts > 0
            result["notes"] = (
                f"nvidia-smi power.draw: {watts:.2f} W. "
                "Subprocess polling fallback will be used (~100× slower than pynvml). "
                "Consider installing pynvml: pip install pynvml."
            )
        else:
            result["notes"] += " nvidia-smi returned N/A."
    except Exception as exc:
        result["notes"] += f" nvidia-smi subprocess failed: {exc}."

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Probe CPU/GPU energy backends.")
    parser.add_argument("--cpu", action="store_true", help="Probe CPU RAPL only")
    parser.add_argument("--gpu", action="store_true", help="Probe GPU NVML only")
    parser.add_argument("--force", action="store_true", help="Ignore cached results")
    args = parser.parse_args()

    if not args.cpu and not args.gpu:
        args.cpu = args.gpu = True  # default: probe both

    hkey = _hostname_key()
    cache = _load_cache()
    host_cache = cache.get(hkey, {}) if not args.force else {}

    print(f"[probe_energy] Host: {hkey}")
    print(f"[probe_energy] Cache file: {_CACHE_FILE}")
    print()

    cpu_result: dict | None = None
    gpu_result: dict | None = None

    # ── E1: CPU RAPL ──
    if args.cpu:
        if "cpu" in host_cache and not args.force:
            cpu_result = host_cache["cpu"]
            print(f"[E1 CPU]  Using cached result (--force to re-probe)")
        else:
            print("[E1 CPU]  Probing pyJoules RAPL …")
            cpu_result = probe_cpu_rapl()

        status = "✅ AVAILABLE" if cpu_result.get("available") else "❌ UNAVAILABLE"
        print(f"[E1 CPU]  {status}")
        print(f"          Backend:  {cpu_result.get('backend')}")
        print(f"          Reading:  {cpu_result.get('reading_uJ')} µJ")
        print(f"          Notes:    {cpu_result.get('notes')}")
        print()

        if not cpu_result.get("available"):
            print(
                "[E1 CPU]  OUTCOME B: RAPL returned zeros.\n"
                "          Action options (record in DECISIONS_PENDING.md):\n"
                "          B1 — Use time×TDP proxy (energy_valid=False in reports)\n"
                "          B2 — Skip CPU energy entirely\n"
                "          B3 — Fix RAPL access (run scripts/setup_energy_permissions.sh)\n"
            )

    # ── E2: GPU NVML ──
    if args.gpu:
        if "gpu" in host_cache and not args.force:
            gpu_result = host_cache["gpu"]
            print(f"[E2 GPU]  Using cached result (--force to re-probe)")
        else:
            print("[E2 GPU]  Probing GPU energy (NVML → pynvml → nvidia-smi) …")
            gpu_result = probe_gpu_nvml()

        status = "✅ AVAILABLE" if gpu_result.get("available") else "❌ UNAVAILABLE"
        print(f"[E2 GPU]  {status}")
        print(f"          Backend:  {gpu_result.get('backend')}")
        print(f"          Reading:  {gpu_result.get('reading_uJ')} µJ")
        print(f"          Fallback: {gpu_result.get('fallback')}")
        print(f"          Notes:    {gpu_result.get('notes')}")
        print()

    # ── Write cache ──
    new_host_cache: dict = dict(host_cache)
    if cpu_result is not None:
        new_host_cache["cpu"] = cpu_result
    if gpu_result is not None:
        new_host_cache["gpu"] = gpu_result
    cache[hkey] = new_host_cache
    _save_cache(cache)
    print(f"[probe_energy] Results cached to {_CACHE_FILE}")

    # ── Summary & exit code ──
    any_available = any([
        cpu_result.get("available") if cpu_result else False,
        gpu_result.get("available") if gpu_result else False,
    ])
    if not any_available:
        print("\n⚠  All energy probes failed. Energy measurement will report None for all runs.")
        print("   See DECISIONS_PENDING.md for remediation options.")
        sys.exit(1)
    else:
        backends_ok = []
        if cpu_result and cpu_result.get("available"):
            backends_ok.append(f"CPU:{cpu_result['backend']}")
        if gpu_result and gpu_result.get("available"):
            backends_ok.append(f"GPU:{gpu_result['backend']}")
        print(f"✅ Active backends: {', '.join(backends_ok)}")


if __name__ == "__main__":
    main()

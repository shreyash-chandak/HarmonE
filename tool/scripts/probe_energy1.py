"""scripts/probe_energy.py — E1/E2 energy backend probe.

Runs on the target hardware to verify which energy backends produce non-zero
readings, then writes the result to knowledge/.energy_backends.json (keyed by
hostname so the result survives reboots without re-probing on a different
machine).

Run once (or with --force to re-probe) before any live experiment:

    cd tool/
    python scripts/probe_energy.py          # probe all
    python scripts/probe_energy.py --cpu    # CPU RAPL only
    python scripts/probe_energy.py --gpu    # GPU NVML only
    python scripts/probe_energy.py --force  # ignore cached result

The probe results are also printed so they can be manually recorded in
DECISIONS_PENDING.md when an Outcome B (zeros/errors) occurs.

Exit codes:
  0 — at least one backend available
  1 — all probes failed / returned zero (action required)

─── CPU (E1) — four paths tried in order ────────────────────────────────────

  Path 1: pyJoules EnergyMeter (standard API).
    Works on pyJoules ≤0.4. BROKEN on pyJoules 0.5.1 with:
      AttributeError: 'RaplPackageDomain' object has no attribute 'get_energy'
    If this fires, fall through to Path 2.

  Path 2: pyJoules low-level — reads the sysfs energy_uj file that
    RaplPackageDomain wraps, bypassing the broken EnergyMeter call chain.
    Requires pyJoules installed (for the domain object) but does not use
    EnergyMeter at all.

  Path 3: Direct sysfs powercap read — no pyJoules required.
    Reads /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj directly.
    Works on Intel AND AMD Zen 2+ (Ryzen 3000+, EPYC Rome+, Ryzen AI 7 350).
    AMD exposes the powercap interface under the same sysfs path as Intel.
    Requires read access to the powercap sysfs; run setup_energy_permissions.sh
    if permission denied.

  Path 4: /proc/driver/amd_energy — AMD kernel module (kernel ≥5.8).
    Load with: sudo modprobe amd_energy
    Falls back to this if powercap sysfs is also unavailable.

─── GPU (E2) — three paths tried in order ───────────────────────────────────

  Path 1: pyJoules NvidiaGPUDomain (nvmlDeviceGetTotalEnergyConsumption).
    Not available on RTX 5060 Laptop / most consumer laptop GPUs.

  Path 2: pynvml power.draw polling at 50ms intervals.
    Works on RTX 5060 (nvmlDeviceGetPowerUsage is available even when the
    cumulative energy counter is not). This is the primary GPU path.

  Path 3: nvidia-smi subprocess fallback (slowest, last resort).
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

# pyJoules 0.5.1 has a known breakage: RaplPackageDomain no longer exposes
# get_energy() as an instance method in the way EnergyMeter calls it internally,
# resulting in AttributeError: 'RaplPackageDomain' object has no attribute
# 'get_energy'.  We therefore try four paths in order:
#
#   Path 1 — pyJoules EnergyMeter (works on pyJoules ≤0.4 or patched builds)
#   Path 2 — pyJoules low-level: read RaplPackageDomain._rapl_file directly
#   Path 3 — sysfs powercap (works on Intel AND AMD Zen 2+ without pyJoules)
#   Path 4 — /proc/driver/amd_energy (AMD-specific, kernel ≥5.8)
#
# The first path that returns a non-zero reading wins.

def _cpu_workload() -> None:
    """Intentional CPU workload — guarantees a non-zero energy delta."""
    acc = 0.0
    for i in range(200_000):
        acc += math.sqrt(float(i) + 1.0)


def _read_rapl_sysfs() -> float | None:
    """Read RAPL energy counter directly from sysfs powercap.

    Works on Intel and on AMD Zen 2+ (Ryzen 3000+, EPYC Rome+).
    Returns energy in µJ, or None if the interface is not present / unreadable.

    Typical paths:
      Intel: /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
      AMD:   /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
             (AMD exposes the same sysfs node under the same name since Zen 2)
    """
    candidate_paths = [
        # Standard powercap path (Intel + AMD Zen 2+)
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
        # Some kernels expose package under a numbered subdirectory
        "/sys/class/powercap/intel-rapl:0/energy_uj",
        # Fallback: try package 1 (multi-socket or AMD CCX numbering)
        "/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj",
        "/sys/class/powercap/intel-rapl:1/energy_uj",
    ]
    for p in candidate_paths:
        path = Path(p)
        if path.exists():
            try:
                return float(path.read_text().strip())
            except (OSError, ValueError):
                continue
    return None


def _read_amd_energy_sysfs() -> float | None:
    """Read AMD energy accumulator from /proc/driver/amd_energy (kernel ≥5.8).

    Returns total energy in µJ across all cores/packages, or None.
    """
    proc_path = Path("/proc/driver/amd_energy")
    if not proc_path.exists():
        return None
    try:
        lines = proc_path.read_text().splitlines()
        # Each line: "CPU<N>: <value> uJ" or "socket<N>: <value> uJ"
        # Sum all socket-level readings (lines containing "socket")
        total = 0.0
        found = False
        for line in lines:
            lower = line.lower()
            if "socket" in lower or "package" in lower:
                parts = line.split()
                for i, part in enumerate(parts):
                    try:
                        total += float(part)
                        found = True
                        break
                    except ValueError:
                        continue
        return total if found else None
    except (OSError, ValueError):
        return None


def probe_cpu_rapl() -> dict:
    """Probe CPU energy measurement capability (E1).

    Tries four paths in order:
      1. pyJoules EnergyMeter (standard API — fails on pyJoules 0.5.1 with
         AttributeError: 'RaplPackageDomain' has no attribute 'get_energy')
      2. pyJoules low-level sysfs file read (bypasses EnergyMeter)
      3. Direct sysfs powercap read (no pyJoules needed — works on AMD Zen 2+)
      4. /proc/driver/amd_energy (AMD kernel module, kernel ≥5.8)

    Returns a dict with keys: available, reading_uJ, backend, notes.
    """
    result: dict = {
        "backend": "rapl",
        "available": False,
        "reading_uJ": None,
        "notes": "",
    }

    # ── Path 1: pyJoules EnergyMeter (standard API) ──────────────────────────
    try:
        from pyJoules.energy_meter import EnergyMeter as _PyEM
        from pyJoules.device.rapl_device import RaplPackageDomain

        meter = _PyEM([RaplPackageDomain(0)])
        meter.start(tag="e1_probe_p1")
        _cpu_workload()
        meter.stop()
        trace = meter.get_trace()

        total_uJ = 0.0
        for sample in trace:
            if sample.energy:
                total_uJ += sum(sample.energy.values())

        if total_uJ > 0:
            result["available"] = True
            result["backend"] = "pyjoules_energy_meter"
            result["reading_uJ"] = total_uJ
            result["notes"] = (
                f"pyJoules EnergyMeter: {total_uJ:.1f} µJ. "
                "Standard API works on this installation."
            )
            return result
        else:
            result["notes"] += (
                "Path 1 (pyJoules EnergyMeter): returned zero — "
                "RAPL interface present but unreadable via this API. "
            )
    except AttributeError as exc:
        # This is the pyJoules 0.5.1 breakage:
        # AttributeError: 'RaplPackageDomain' object has no attribute 'get_energy'
        result["notes"] += (
            f"Path 1 (pyJoules EnergyMeter): AttributeError — {exc}. "
            "This is a known pyJoules 0.5.1 API breakage. Trying lower-level paths. "
        )
    except ImportError as exc:
        result["notes"] += f"Path 1 (pyJoules EnergyMeter): import failed — {exc}. "
    except Exception as exc:
        result["notes"] += (
            f"Path 1 (pyJoules EnergyMeter): {type(exc).__name__}: {exc}. "
        )

    # ── Path 2: pyJoules low-level — read the sysfs file that RaplPackageDomain
    #            wraps, bypassing the broken EnergyMeter call chain ────────────
    try:
        from pyJoules.device.rapl_device import RaplPackageDomain

        domain = RaplPackageDomain(0)

        # pyJoules stores the sysfs path in different attributes depending on version
        rapl_file: Path | None = None
        for attr in ("_rapl_file", "_energy_file", "_path", "path"):
            candidate = getattr(domain, attr, None)
            if candidate is not None:
                rapl_file = Path(str(candidate))
                break

        if rapl_file is not None and rapl_file.exists():
            before = float(rapl_file.read_text().strip())
            _cpu_workload()
            after = float(rapl_file.read_text().strip())
            delta_uJ = after - before

            # Handle counter wraparound (max_energy_range_uj)
            if delta_uJ < 0:
                max_path = rapl_file.parent / "max_energy_range_uj"
                if max_path.exists():
                    max_val = float(max_path.read_text().strip())
                    delta_uJ += max_val

            if delta_uJ > 0:
                result["available"] = True
                result["backend"] = "pyjoules_lowlevel_sysfs"
                result["reading_uJ"] = delta_uJ
                result["notes"] += (
                    f"Path 2 (pyJoules low-level sysfs via {rapl_file}): "
                    f"{delta_uJ:.1f} µJ. "
                    "Bypasses broken EnergyMeter; reads the powercap file directly."
                )
                return result
            else:
                result["notes"] += (
                    f"Path 2 (pyJoules low-level sysfs): delta was zero (before={before}, "
                    f"after={after}). "
                )
        else:
            result["notes"] += (
                "Path 2 (pyJoules low-level sysfs): could not find _rapl_file attribute "
                "or file does not exist. "
            )
    except ImportError:
        result["notes"] += "Path 2 (pyJoules low-level): pyJoules not installed. "
    except Exception as exc:
        result["notes"] += (
            f"Path 2 (pyJoules low-level): {type(exc).__name__}: {exc}. "
        )

    # ── Path 3: direct sysfs powercap read (no pyJoules) ─────────────────────
    # Works on Intel and AMD Zen 2+ (Ryzen 3000+, EPYC Rome+, Ryzen AI 7 350)
    try:
        before = _read_rapl_sysfs()
        if before is not None:
            _cpu_workload()
            after = _read_rapl_sysfs()

            if after is not None:
                delta_uJ = after - before

                # Handle counter wraparound
                if delta_uJ < 0:
                    # Find max_energy_range_uj alongside the energy_uj file
                    for energy_path_str in [
                        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
                        "/sys/class/powercap/intel-rapl:0/energy_uj",
                    ]:
                        ep = Path(energy_path_str)
                        if ep.exists():
                            max_path = ep.parent / "max_energy_range_uj"
                            if max_path.exists():
                                delta_uJ += float(max_path.read_text().strip())
                            break

                if delta_uJ > 0:
                    result["available"] = True
                    result["backend"] = "sysfs_powercap_direct"
                    result["reading_uJ"] = delta_uJ
                    result["notes"] += (
                        f"Path 3 (direct sysfs powercap): {delta_uJ:.1f} µJ. "
                        "pyJoules not used — reads /sys/class/powercap/intel-rapl directly. "
                        "Works on AMD Zen 2+ and Intel without pyJoules. "
                        "core/energy.py will use this backend via the SysfsRaplBackend class."
                    )
                    return result
                else:
                    result["notes"] += (
                        f"Path 3 (direct sysfs): delta was zero (before={before}, "
                        f"after={after}). Counter may not increment — "
                        "check /sys/class/powercap/ permissions. "
                    )
            else:
                result["notes"] += (
                    "Path 3 (direct sysfs): could read before-value but not after. "
                )
        else:
            result["notes"] += (
                "Path 3 (direct sysfs): /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj "
                "not found. This is expected on WSL2 (Hyper-V blocks MSR). "
                "On bare-metal Arch, try: sudo modprobe msr. "
            )
    except Exception as exc:
        result["notes"] += f"Path 3 (direct sysfs): {type(exc).__name__}: {exc}. "

    # ── Path 4: /proc/driver/amd_energy (AMD kernel module, kernel ≥5.8) ─────
    try:
        before = _read_amd_energy_sysfs()
        if before is not None:
            _cpu_workload()
            after = _read_amd_energy_sysfs()
            if after is not None:
                delta_uJ = after - before
                if delta_uJ > 0:
                    result["available"] = True
                    result["backend"] = "amd_energy_proc"
                    result["reading_uJ"] = delta_uJ
                    result["notes"] += (
                        f"Path 4 (/proc/driver/amd_energy): {delta_uJ:.1f} µJ. "
                        "AMD energy kernel module active. "
                        "core/energy.py will use this backend via AmdEnergyBackend."
                    )
                    return result
                else:
                    result["notes"] += (
                        "Path 4 (/proc/driver/amd_energy): delta was zero. "
                    )
            else:
                result["notes"] += (
                    "Path 4 (/proc/driver/amd_energy): could not read after-value. "
                )
        else:
            result["notes"] += (
                "Path 4 (/proc/driver/amd_energy): file not found. "
                "Load with: sudo modprobe amd_energy. "
            )
    except Exception as exc:
        result["notes"] += f"Path 4 (amd_energy): {type(exc).__name__}: {exc}. "

    # ── All paths failed ──────────────────────────────────────────────────────
    result["notes"] += (
        "\nAll CPU energy paths failed. On bare-metal Arch Linux, run: "
        "sudo modprobe msr && sudo bash scripts/setup_energy_permissions.sh. "
        "On WSL2, CPU energy measurement is unavailable (Hyper-V blocks MSR access) — "
        "this is expected and documented in CHANGES_FROM_PAPER.md."
    )
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
                "[E1 CPU]  OUTCOME B: All four CPU energy paths failed.\n"
                "\n"
                "          Paths tried:\n"
                "            1. pyJoules EnergyMeter       (fails on pyJoules 0.5.1 with AttributeError)\n"
                "            2. pyJoules low-level sysfs   (reads _rapl_file directly)\n"
                "            3. Direct sysfs powercap      (/sys/class/powercap/intel-rapl/...)\n"
                "            4. /proc/driver/amd_energy    (AMD kernel module)\n"
                "\n"
                "          If on bare-metal Arch Linux:\n"
                "            sudo modprobe msr\n"
                "            sudo modprobe amd_energy        # AMD-specific, kernel ≥5.8\n"
                "            sudo bash scripts/setup_energy_permissions.sh\n"
                "            python scripts/probe_energy.py --cpu --force\n"
                "\n"
                "          If on WSL2:\n"
                "            CPU energy unavailable — Hyper-V blocks MSR access.\n"
                "            This is expected and documented. Set energy_meter: null\n"
                "            in regression dataset configs. GPU energy still works via pynvml.\n"
                "\n"
                "          Action options (record decision in DECISIONS_PENDING.md):\n"
                "            B1 — Use time×TDP proxy (energy_valid=False in reports)\n"
                "            B2 — Skip CPU energy entirely (regression only)\n"
                "            B3 — Fix access and re-probe (preferred on bare-metal)\n"
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

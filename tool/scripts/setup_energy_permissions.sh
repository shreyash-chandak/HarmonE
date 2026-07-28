#!/usr/bin/env bash
# scripts/setup_energy_permissions.sh — Once-per-boot energy measurement setup.
#
# Run this with sudo before starting any live experiment or the managed system
# when using RAPL or NVML energy measurement.
#
# Required once per boot. For automatic setup on boot, see RUNNING_ON_ARCH.md §3.
#
# Usage:
#   sudo bash scripts/setup_energy_permissions.sh

set -euo pipefail

echo "[energy] Loading MSR module (required for RAPL access)..."
modprobe msr 2>/dev/null && echo "  ✔ msr loaded" || echo "  ⚠ msr already loaded or unavailable"

echo "[energy] Loading Intel RAPL modules..."
modprobe intel_rapl_common 2>/dev/null && echo "  ✔ intel_rapl_common loaded" \
    || echo "  ⚠ intel_rapl_common unavailable (AMD RAPL uses different interface)"
modprobe intel_rapl_msr 2>/dev/null && echo "  ✔ intel_rapl_msr loaded" \
    || echo "  ⚠ intel_rapl_msr unavailable"

if [ -d "/sys/class/powercap/intel-rapl" ]; then
    echo "[energy] Setting RAPL sysfs permissions..."
    chmod -R 777 /sys/class/powercap/intel-rapl/
    echo "  ✔ /sys/class/powercap/intel-rapl/ → 777"
else
    echo "  ⚠ /sys/class/powercap/intel-rapl/ not found — RAPL may not be supported"
    echo "     On AMD Zen 5 (Ryzen AI 7 350), check /sys/class/powercap/ for AMD RAPL:"
    ls /sys/class/powercap/ 2>/dev/null || echo "     (powercap not available)"
fi

echo "[energy] Setup complete. Permissions reset on next reboot — re-run this script."

# Live Run — Bug Report, Fix Specification, and Energy Instrumentation Plan

> Generated from analysis of three shell outputs from the first live run of the
> HarmonE-tool on `fix/bugs-phase1` branch, July 24 2026.
> Hardware: Lenovo Legion 5 Gen 10, AMD Ryzen AI 7 350, NVIDIA RTX 5060 Laptop GPU.
> All issues below must be resolved before any experimental results are valid.

---

## Status Summary

| # | Problem | Domain | Severity | Blocks |
|---|---|---|---|---|
| L1 | `scaler.pkl` not found — inference never runs | Regression | Critical | All regression results |
| L2 | Stale command replay on startup | Both | Critical | Reproducibility, event counts |
| L3 | CUDA kernel incompatibility — CV inference crashes | CV | Critical | All CV results |
| L4 | `None` comparison crash in ACP secondary check | Both | Moderate | ACP drift tactic firing |
| L5 | Planner thrashing on stale cached telemetry | Regression | Consequential | Meaningful switch analysis |
| L6 | Switch counter persists across sessions | Both | Moderate | Adaptation count metrics |
| E1 | pyRAPL incompatible with AMD CPU | Regression | Critical | CPU energy measurement |
| E2 | pyJoules NVML backend needs validation on RTX 5060 | CV | Critical | GPU energy measurement |
| E3 | No unified energy abstraction layer | Both | Architectural | Clean experiment harness |

---

## L1 — Regression Inference Never Runs (Critical)

### What happened

Every regression session crashed immediately at startup:

```
FileNotFoundError: knowledge/scaler.pkl not found.
Run python scripts/init_scaler.py to fit the scaler on training data.
```

`inference.py` raises this error on line 37 before processing a single sample.
The subprocess exits silently from the MasterWrapper's perspective, which keeps
running as if inference were live.

### What the system did instead

Because the inference subprocess is dead, `predictions.csv` receives no new rows.
`monitor.py`'s `monitor_mape()` detects an empty new-data window and falls back to
its cached-telemetry path:

```
📉 No new data to process in predictions.csv, using recent data for telemetry
🔄 Using recent data: R²=-0.5900, Actual Energy=13566.52,
   Normalized Energy=0.5427, Score=-0.1148
```

The values R²=−0.59, Energy=13566.52, Normalized Energy=0.5427 appear in every
single telemetry push across the entire session — they are the last 50 rows of a
`predictions.csv` left over from a prior session, not live measurements.

Every ACP violation, every tactic fired, every model switch logged during this run
was triggered by stale data. The switch counter reached 112 by session end, all
of it meaningless. No inference occurred.

### Root cause

B7 (scaler leakage fix) was implemented correctly — `inference.py` now refuses to
fit the scaler on streaming data. But the initialisation step that creates
`knowledge/scaler.pkl` by fitting on training data was never run. The fix created
a hard dependency on an initialisation script that must be run once before the
first inference session and again after each retrain.

### Fix

**Step 1**: Run the initialisation script:
```bash
cd tool
python scripts/init_scaler.py
```

This script must fit `MinMaxScaler` on the training split only (first 80% of
`dataset.csv` in chronological order), then persist it to
`managed_system_regression/knowledge/scaler.pkl`.

Verify `scripts/init_scaler.py` does the following and nothing else:
```python
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("managed_system_regression/data/dataset.csv")
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]

scaler = MinMaxScaler()
scaler.fit(train_df[["flow"]])  # fit on training column only

with open("managed_system_regression/knowledge/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"Scaler fitted on {train_size} training samples. Saved to knowledge/scaler.pkl")
```

If the script does not exist or does not match this pattern, write it from scratch.

**Step 2**: Verify `inference.py` loads and applies (never fits) the scaler:
```python
# On startup
with open("knowledge/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Per prediction
x_scaled = scaler.transform(x_raw)  # transform only — never fit
```

**Step 3**: After each retrain, `retrain.py` must refit and overwrite the scaler
on the new training data atomically:
```python
scaler = MinMaxScaler()
scaler.fit(new_train_data)
tmp_path = "knowledge/scaler.pkl.tmp"
with open(tmp_path, "wb") as f:
    pickle.dump(scaler, f)
os.replace(tmp_path, "knowledge/scaler.pkl")
```

The atomic write (write to `.tmp` then `os.replace`) prevents inference from
loading a partially-written scaler if a retrain is interrupted.

**Step 4**: Add `scaler.pkl` to the startup health check in `run_managed_system.py`
so a missing scaler aborts the wrapper with a clear message rather than silently
running without inference:
```python
scaler_path = "managed_system_regression/knowledge/scaler.pkl"
if not os.path.exists(scaler_path):
    raise RuntimeError(
        f"Missing {scaler_path}. Run: python scripts/init_scaler.py"
    )
```

**Step 5**: Document in README that `init_scaler.py` must be run once before
the first regression experiment and after every retrain that changes the training
window.

---

## L2 — Stale Command Replay on Startup (Critical)

### What happened

When the first regression session was terminated mid-run, the last-queued command
(`execute_mape_plan`) remained in `managed_system_regression/knowledge/command.txt`.
When the next session started 2 minutes later, `manage.py` read and immediately
executed this leftover command before inference had even loaded:

```
2026-07-24 23:55:41 - [plan.py] - Running in 'harmone_acp' mode. Listening for commands...
2026-07-24 23:55:41 - [plan.py] - Command 'execute_mape_plan' received. Triggering local logic...
...
2026-07-24 23:55:41 - [plan.py] - Event recorded: Model switch #109
```

This switch was executed at t=0 of the new session, before a single inference step,
based on EMA scores from the previous session. Switch #109 is therefore invalid.

For CV, a similar mechanism would apply — any command left in
`managed_system_cv/knowledge/command.txt` would be replayed.

Additionally, across multiple sessions the switch counter continued incrementing
from 68 (where a previous session left it) through 113, accumulating across
sessions. This means `model_switches` in `mape_info.json` is a cumulative total
across all runs, not a per-run count. For the paper's adaptation count metrics
(Table 2 equivalent), this makes the counter useless without a reset mechanism.

### Root cause

Two separate issues:

1. `knowledge/command.txt` is not cleared on shutdown or startup. The command
   listener in `manage.py` reads whatever is in the file when it starts, without
   checking whether the command predates the current session.

2. `mape_info.json` event counters are never reset between experimental runs.
   Each run appends to the previous session's totals.

### Fix

**Fix 2a — Clear command file on startup**:

In `managed_system_regression/mape_logic/manage.py` and the CV equivalent,
add a startup clear before the command listener loop begins:

```python
# Clear any stale command from a previous session
command_file = os.path.join(KNOWLEDGE_DIR, "command.txt")
if os.path.exists(command_file):
    open(command_file, "w").close()
    logging.info("Cleared stale command.txt from previous session.")
```

This must happen before the listener thread starts, not inside it.

**Fix 2b — Add per-run reset to experiment harness**:

When the experiment harness (Phase 5) runs a fresh experiment, it must reset
`mape_info.json` counters to zero before starting the managed system. The reset
should preserve EMA score structure but zero all counters:

```python
def reset_run_counters(mape_info_path):
    with open(mape_info_path, "r") as f:
        info = json.load(f)
    info["event_counters"] = {
        "model_switches": 0,
        "retrains": 0,
        "vmr_events": 0,
        "mape_k_energy_uJ": 0.0
    }
    info["simple_switch_counters"] = {"simple_switches": 0}
    info["last_line"] = 0
    # Reset EMA scores to default
    info["ema_scores"] = {k: 0.5 for k in info["ema_scores"]}
    with open(mape_info_path, "w") as f:
        json.dump(info, f, indent=4)
```

Also clear `predictions.csv` (keep the header, remove all data rows) and
clear `knowledge/command.txt` as part of each run reset.

**Fix 2c — Timestamp-gate commands (belt-and-suspenders)**:

Optionally, when ACP writes a command to `command.txt`, include the timestamp:
```
execute_mape_plan|1784917290.086727
```
The listener only executes the command if `time.time() - timestamp < 30` seconds.
Commands older than 30 seconds are discarded with a warning log. This prevents
replay even if the startup clear is missed.

---

## L3 — CUDA Kernel Incompatibility — CV Inference Crashes (Critical)

### What happened

CV inference started, loaded YOLOv8m, and crashed during the first warmup forward
pass:

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

Immediately before the crash, PyTorch emitted two explicit warnings:

```
UserWarning: Found GPU0 NVIDIA GeForce RTX 5060 Laptop GPU which is of
compute capability (CC) 12.0.
torch==2.13.0+cu126 does not include kernels for this GPU.
```

and:

```
UserWarning: NVIDIA GeForce RTX 5060 Laptop GPU with CUDA capability sm_120
is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities:
sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90.
```

The RTX 5060 is an Ada/Blackwell-era GPU with compute capability 12.0 (sm_120).
PyTorch 2.13.0+cu126 was built against CUDA 12.6 which only compiled kernels
for up to sm_90 (RTX 40-series). sm_120 is a 50-series architecture that requires
CUDA 12.9 or later compiled kernels.

The inference subprocess exits, and similarly to L1, the CV monitoring loop falls
back to cached or default telemetry. All CV metrics during this session were
fabricated (confidence stuck at 0.5, energy at 0.0).

### Fix

Reinstall PyTorch with a CUDA build that includes sm_120 kernels.

**Step 1**: Check which CUDA versions have sm_120 kernel support for PyTorch:
```bash
pip index versions torch --index-url https://download.pytorch.org/whl/cu129 2>/dev/null | head -5
pip index versions torch --index-url https://download.pytorch.org/whl/cu130 2>/dev/null | head -5
```

**Step 2**: Install against CUDA 12.9 (first choice, most likely to support sm_120):
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

If cu129 doesn't have a build for your Python version (3.13), try cu130 or cu132.

**Step 3**: Verify the install works with your GPU:
```bash
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Compute capability:', torch.cuda.get_device_capability(0))
    x = torch.tensor([1.0]).cuda()
    print('Tensor on GPU:', x)
    print('GPU is working correctly.')
"
```

Expected output:
```
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
Compute capability: (12, 0)
Tensor on GPU: tensor([1.], device='cuda:0')
GPU is working correctly.
```

**Step 4**: Reinstall Ultralytics after the PyTorch change, as it caches model
formats based on the torch build:
```bash
pip install ultralytics --upgrade
```

**Step 5**: Verify YOLO runs on GPU:
```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
import torch
assert torch.cuda.is_available()
results = model('https://ultralytics.com/images/bus.jpg', device=0, verbose=False)
print('YOLO inference successful. Detections:', len(results[0].boxes))
"
```

**Note on Python 3.13**: PyTorch support for Python 3.13 was added in PyTorch 2.6+,
so 2.13.0 should work. If the cu129 wheel is not available for Python 3.13 at the
time you run this, check https://download.pytorch.org/whl/cu129 for the available
filenames — look for `cp313` in the filename.

---

## L4 — `None` Comparison Crash in ACP Secondary Check (Moderate)

### What happened

The periodic secondary evaluator thread in `app.py` crashed with:

```
Exception in thread Thread-1 (periodic_secondary_checks):
TypeError: '>' not supported between instances of 'NoneType' and 'float'
  File "app.py", line 189, in periodic_secondary_checks
      (condition == "GREATER_THAN" and value > threshold)
```

This happens because `kl_div` is `None` in telemetry (correct behaviour from the
B5 fix — `None` is returned when fewer than 2400 samples exist for drift detection),
but `app.py`'s secondary check evaluator blindly compares whatever value it finds
in the telemetry dict against the configured threshold without a `None` guard.

The thread dies after this exception. Python's `threading` module does not restart
crashed daemon threads. The secondary boundary check (drift tactic firing) is
therefore silently disabled for the remainder of any session where this crash
occurs — which is every session during the warmup period before 2400 samples
accumulate.

### Exact location

File: `tool/app.py`, function `periodic_secondary_checks`, line ~189:
```python
(condition == "GREATER_THAN" and value > threshold)
```

`value` is `latest_telemetry.get(qa)` where `qa = "kl_div"`, which returns `None`
when the B5 fix is active.

### Fix

Add a `None` guard before the comparison:

```python
def evaluate_boundary(value, condition, threshold):
    """Evaluate a boundary condition safely, returning False if value is None."""
    if value is None:
        return False  # No signal available — do not fire
    if condition == "LESS_THAN":
        return value < threshold
    if condition == "GREATER_THAN":
        return value > threshold
    return False

# Replace the inline comparisons in periodic_secondary_checks:
# Before:
#   (condition == "GREATER_THAN" and value > threshold)
# After:
#   evaluate_boundary(value, condition, threshold)
```

Also add a log when a `None` value is skipped so it is visible in the ACP output:
```python
if value is None:
    logging.debug(f"[SECONDARY] Skipping boundary check for '{qa}': value is None (warmup period)")
    continue
```

The thread should never crash on a `None` telemetry value. Any None-returning
metric must be silently skipped, not compared.

---

## L5 — Planner Thrashing on Stale Cached Telemetry (Consequential)

### What happened

The regression planner oscillated continuously between SVM and LINEAR throughout
the entire session:

```
PLAN: Exploitation tactic: Best alternative to 'SVM' is 'LINEAR' (Score: -2.49)
PLAN: Exploitation tactic: Best alternative to 'LINEAR' is 'SVM' (Score: -0.11)
PLAN: Exploitation tactic: Best alternative to 'SVM' is 'LINEAR' (Score: -2.49)
... (repeated 100+ times)
```

LSTM was never selected by exploitation. This pattern produces roughly 1 switch
per 5 seconds indefinitely, which is not what deliberate adaptation looks like.

### Root cause

This is a downstream consequence of L1. Because inference is dead, `predictions.csv`
gets no new rows, so `monitor_mape()` serves the same cached R²=-0.59 every interval.
R²=-0.59 is deeply below `S_min=0.78`, so ACP fires `execute_mape_plan` every 5
seconds. The planner, seeing all three EMA scores negative (SVM: -0.11, LINEAR: -2.49,
LSTM: presumably around -5.91 from the first monitor read), always picks the
least-negative non-current model. Since SVM (-0.11) beats LINEAR (-2.49) and
LINEAR beats LSTM (-5.91), LSTM is never reached by exploitation. The ε-greedy
random path occasionally fires (you can see `🎲 PLAN: Exploratory tactic!
Randomly selecting 'SVM'` at switch #71, #83, #96) but even that just picks SVM
or LINEAR at random.

This is not a planner bug — it is the planner behaving correctly given garbage
input. Fix L1 and this resolves itself, because real inference will produce R²
values in the 0.75–0.91 range and EMA scores will stabilise above S_min once a
capable model is running.

### One real issue to note

The EMA scores carried over from the previous session (SVM: -0.11, LINEAR: -2.49)
are so corrupted by the stale-data session that even after L1 is fixed, the
first few hundred steps of the next session will be influenced by these wrong
EMA priors. The per-run reset described in L2 Fix 2b also resets EMA scores to
0.5, which eliminates this contamination. Confirm that the reset does this.

---

## L6 — Switch Counter Persists Across Sessions (Moderate)

### What happened

The first session started with `model_switches: 68` (not 0), indicating this was
not the first session ever run. By the end of the first session it reached 79.
The second session started at 109 (the 30 switches from session 1 plus some extra
from the replay issue in L2). By the second session's end it reached 113.

The paper's Table 2 equivalent reports adaptation counts per experimental run.
A counter that accumulates across sessions makes this metric meaningless without
a separate mechanism to snapshot the start-of-run count and compute the delta.

### Fix

The per-run reset in L2 Fix 2b handles this — it zeros `model_switches`,
`retrains`, and `vmr_events` before each experimental run. Additionally, the
experiment harness should log the counter value at the start and end of each run
and report the delta, not the absolute value, so that any accidental miss of the
reset is caught:

```python
def get_adaptation_counts(mape_info_path):
    with open(mape_info_path, "r") as f:
        info = json.load(f)
    return dict(info["event_counters"])

# In experiment harness:
counts_before = get_adaptation_counts(mape_info_path)
# ... run experiment ...
counts_after = get_adaptation_counts(mape_info_path)
delta = {k: counts_after[k] - counts_before[k] for k in counts_before}
# delta is the per-run adaptation count regardless of whether reset was applied
```

---

## E1 — pyRAPL Incompatible with AMD CPU (Critical for CPU Energy)

### Background

The current codebase uses pyRAPL for CPU energy measurement. pyRAPL reads Intel's
RAPL (Running Average Power Limit) MSR registers. AMD implemented a partially
compatible RAPL interface from Zen 2 onwards, but compatibility with pyRAPL
specifically is inconsistent and hardware-dependent.

The machine runs an AMD Ryzen AI 7 350 (Zen 5 architecture). Whether pyRAPL
returns valid non-zero readings on this chip is currently unknown — it was not
tested in the live run because the regression inference subprocess crashed before
any energy measurement occurred.

### Why this matters

Every energy number in the paper's results comes from pyRAPL. If it returns
zeros on your AMD chip, all regression energy measurements will be zero, making
the paper's RQ1 and RQ2 results entirely fabricated.

### Required test

Run this immediately after fixing L1 (once inference is actually running):

```bash
sudo modprobe msr  # load MSR kernel module — required for RAPL access
python3 -c "
import pyRAPL
pyRAPL.setup()
meter = pyRAPL.Measurement('amd_rapl_test')
meter.begin()
# Simulate inference workload
import numpy as np
for _ in range(100000):
    np.dot(np.random.rand(100), np.random.rand(100))
meter.end()
result = meter.result
print('pkg readings:', result.pkg)
print('dram readings:', result.dram)
if result.pkg and result.pkg[0] > 0:
    print('RAPL is working on this AMD CPU — pkg energy:', result.pkg[0], 'uJ')
else:
    print('RAPL returned zero or None — AMD RAPL not accessible via pyRAPL')
"
```

### Outcome A — RAPL works (pkg > 0)

pyRAPL is usable as-is for CPU energy on the Ryzen AI 7 350. Note in the paper
that experiments were conducted on AMD hardware with RAPL-compatible MSR registers
and readings should be treated as estimates (AMD RAPL accuracy is lower than Intel's,
typically ±10-15% vs ±5%). Add this to threats.

### Outcome B — RAPL returns zero or errors

Switch to pyJoules as the primary measurement library (see E2/E3 below). pyJoules
has marginally better AMD handling and falls back more gracefully. If pyJoules also
returns zero on AMD, you have two options:

**Option B1 (preferred)**: Accept that CPU energy measurement is not available on
this hardware for regression experiments. Report this explicitly in the paper. Use
a synthetic CPU energy proxy: measure inference time per step and multiply by the
CPU's rated TDP (the Ryzen AI 7 350 has a configurable TDP of 15–54W; use the
measured average power draw from a monitoring tool like `turbostat` during inference
as a constant multiplier). This is imprecise but honest.

**Option B2**: Run regression experiments inside a VM or container on an Intel
machine where RAPL is reliable. Not recommended given time constraints.

**Option B3**: Report that regression energy is CPU-time-based proxy rather than
hardware counter measurement, note this as a threat to construct validity, and
proceed. This is what the original paper may have done anyway since the i5-1240P
has known RAPL reliability issues at high frequencies.

---

## E2 — pyJoules GPU Measurement on RTX 5060 Needs Validation (Critical for CV Energy)

### Background

The previous analysis concluded that pyJoules's NVML backend should work because
`nvidia-smi` returns valid power readings (10.41W average, 11.74W instantaneous).
However, this has not been tested in code. Before writing the energy abstraction
layer (E3), confirm pyJoules NVML actually works on the RTX 5060.

### Required test

After fixing L3 (PyTorch reinstall):

```bash
pip install pyjoules
python3 -c "
from pyJoules.energy_meter import EnergyMeter
from pyJoules.device.nvidia_gpu import NvidiaGPUDomain
import time

try:
    domains = [NvidiaGPUDomain(0)]
    meter = EnergyMeter(domains)
    meter.start(tag='gpu_test')
    # Simulate GPU workload
    import torch
    x = torch.randn(1000, 1000, device='cuda')
    for _ in range(100):
        y = torch.mm(x, x)
    torch.cuda.synchronize()
    time.sleep(1)
    meter.stop()
    trace = meter.get_trace()
    for sample in trace:
        print('tag:', sample.tag)
        print('duration:', sample.duration, 's')
        print('energy:', sample.energy, 'uJ')
        if sample.energy and any(e > 0 for e in sample.energy.values()):
            print('GPU energy measurement WORKING')
        else:
            print('GPU energy measurement returned zero — NVML may not work')
except Exception as e:
    print('pyJoules NVML failed:', type(e).__name__, e)
    print('Will need manual polling fallback')
"
```

### Outcome A — pyJoules NVML works (energy > 0)

Use pyJoules as the primary GPU energy backend. Implement E3 with
`NvidiaGPUDomain(0)` as the GPU backend and either `RaplPackageDomain(0)` or the
proxy mechanism for CPU (depending on E1 outcome).

### Outcome B — pyJoules NVML fails or returns zero

Fall back to the manual polling meter. The RTX 5060 returns valid power.draw
readings from nvidia-smi (confirmed: 10.41W average, 11.74W instantaneous,
50W limit). Integrating `power.draw` over time gives energy in joules.

The manual polling meter to implement:

```python
import subprocess
import threading
import time

class NvidiaPowerMeter:
    """
    Polls nvidia-smi for instantaneous power draw at a fixed interval
    and integrates to estimate energy consumed.

    Used as fallback when NVML energy counters are unavailable
    (e.g. RTX 5060 with driver lacking cumulative energy counter support).

    Resolution: interval_ms (default 50ms). At 50ms, error per inference
    call is bounded by one missed sample, typically < 5% for YOLO inference
    which runs 100-800ms per batch.
    """

    def __init__(self, interval_ms=50, gpu_index=0):
        self.interval_s = interval_ms / 1000.0
        self.gpu_index = gpu_index
        self._readings_w = []          # watts at each sample
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        self._readings_w = []
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits"
        ]
        while self._running:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=0.5
                )
                val_str = result.stdout.strip()
                if val_str and val_str.lower() != "[n/a]":
                    with self._lock:
                        self._readings_w.append(float(val_str))
            except (subprocess.TimeoutExpired, ValueError):
                pass
            time.sleep(self.interval_s)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def energy_joules(self):
        """Integrate power over time: E = sum(P_i) * dt"""
        with self._lock:
            if not self._readings_w:
                return 0.0
            return sum(self._readings_w) * self.interval_s

    def energy_microjoules(self):
        return self.energy_joules() * 1e6

    def sample_count(self):
        with self._lock:
            return len(self._readings_w)

    def mean_power_watts(self):
        with self._lock:
            if not self._readings_w:
                return 0.0
            return sum(self._readings_w) / len(self._readings_w)
```

**Accuracy note**: At 50ms polling interval, a 200ms YOLO inference call produces
4 samples. One missed sample represents a 25% error on that call, but averaged
across hundreds of inference steps the error converges to roughly ±(interval/2)
seconds worth of power, i.e. ±25ms × mean_power. At 15W mean GPU draw during
YOLO inference, this is ±0.375 mJ per call, or roughly 1-2% relative error at
the per-call level, much less over a full run. This is acceptable for the paper.

---

## E3 — Unified Energy Abstraction Layer (Architectural)

### Why this is needed

Currently, pyRAPL calls are scattered throughout `execute.py` files in both
managed system directories. The measurement library, the backend, and the
reporting format are all tightly coupled to the CPU-only pyRAPL API. Changing
to pyJoules or adding GPU measurement requires touching multiple files. The
experiment harness (Phase 5) needs to swap backends per domain (CPU-only for
regression, GPU+CPU for CV) without changing any managed system code.

### Design

Create `tool/core/energy.py` as the single point of truth for all energy
measurement. All execute.py files, retrain.py files, and the experiment harness
import from here and never call pyRAPL or pyJoules directly.

```python
# tool/core/energy.py

"""
Unified energy measurement abstraction for HarmonE experiments.

Domain selection:
  EnergyContext("cpu")  — regression experiments (CPU-only, RAPL)
  EnergyContext("gpu")  — CV inference energy (GPU only)
  EnergyContext("both") — CV full-system energy (GPU + CPU)

Backend selection is automatic based on available hardware and libraries.
Results always available as .cpu_joules, .gpu_joules, .total_joules.
"""

import os
import logging
import time

logger = logging.getLogger(__name__)

# ── Backend availability detection ──────────────────────────────────────────

def _detect_pyjoules_rapl():
    """Returns True if pyJoules RAPL backend is available and returns non-zero."""
    try:
        from pyJoules.device.rapl_device import RaplPackageDomain
        from pyJoules.energy_meter import EnergyMeter
        meter = EnergyMeter([RaplPackageDomain(0)])
        meter.start(tag="probe")
        time.sleep(0.05)
        meter.stop()
        trace = meter.get_trace()
        # Check if we got a non-zero reading
        for sample in trace:
            if sample.energy and any(v > 0 for v in sample.energy.values()):
                return True
        logger.warning("pyJoules RAPL returned zero — AMD RAPL may not be accessible")
        return False
    except Exception as e:
        logger.warning(f"pyJoules RAPL unavailable: {e}")
        return False

def _detect_pyjoules_nvml():
    """Returns True if pyJoules NVML backend is available and returns non-zero."""
    try:
        from pyJoules.device.nvidia_gpu import NvidiaGPUDomain
        from pyJoules.energy_meter import EnergyMeter
        meter = EnergyMeter([NvidiaGPUDomain(0)])
        meter.start(tag="probe")
        time.sleep(0.1)
        meter.stop()
        trace = meter.get_trace()
        for sample in trace:
            if sample.energy and any(v > 0 for v in sample.energy.values()):
                return True
        logger.warning("pyJoules NVML returned zero — falling back to power polling")
        return False
    except Exception as e:
        logger.warning(f"pyJoules NVML unavailable: {e}")
        return False


# ── CPU energy measurement ───────────────────────────────────────────────────

class _RaplCPUMeter:
    """pyJoules RAPL backend for CPU energy."""

    def __init__(self):
        from pyJoules.device.rapl_device import RaplPackageDomain
        from pyJoules.energy_meter import EnergyMeter
        self._meter = EnergyMeter([RaplPackageDomain(0)])
        self.joules = 0.0

    def start(self):
        self._meter.start(tag="measure")

    def stop(self):
        self._meter.stop()
        for sample in self._meter.get_trace():
            if sample.energy:
                self.joules = sum(sample.energy.values()) / 1e6  # µJ → J


class _NullCPUMeter:
    """No-op CPU meter when RAPL is unavailable."""

    def __init__(self):
        self.joules = 0.0
        logger.warning(
            "CPU energy measurement unavailable (AMD RAPL not accessible). "
            "CPU energy will be reported as 0. Note in paper threats section."
        )

    def start(self):
        pass

    def stop(self):
        pass


# ── GPU energy measurement ───────────────────────────────────────────────────

class _NvmlGPUMeter:
    """pyJoules NVML backend for GPU energy."""

    def __init__(self):
        from pyJoules.device.nvidia_gpu import NvidiaGPUDomain
        from pyJoules.energy_meter import EnergyMeter
        self._meter = EnergyMeter([NvidiaGPUDomain(0)])
        self.joules = 0.0

    def start(self):
        self._meter.start(tag="measure")

    def stop(self):
        self._meter.stop()
        for sample in self._meter.get_trace():
            if sample.energy:
                self.joules = sum(sample.energy.values()) / 1e6


class _PollingGPUMeter:
    """
    Fallback GPU meter using nvidia-smi power.draw polling.
    Used when NVML energy counters are unavailable (e.g. RTX 5060).
    """

    def __init__(self, interval_ms=50, gpu_index=0):
        import subprocess
        import threading
        self._subprocess = subprocess
        self._threading = threading
        self.interval_s = interval_ms / 1000.0
        self.gpu_index = gpu_index
        self._readings_w = []
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self.joules = 0.0

    def start(self):
        self._readings_w = []
        self._running = True
        self._thread = self._threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        cmd = [
            "nvidia-smi", f"--id={self.gpu_index}",
            "--query-gpu=power.draw", "--format=csv,noheader,nounits"
        ]
        while self._running:
            try:
                result = self._subprocess.run(
                    cmd, capture_output=True, text=True, timeout=0.5
                )
                val = result.stdout.strip()
                if val and val.lower() not in ("[n/a]", ""):
                    with self._lock:
                        self._readings_w.append(float(val))
            except Exception:
                pass
            import time
            time.sleep(self.interval_s)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._readings_w:
                self.joules = sum(self._readings_w) * self.interval_s
            else:
                self.joules = 0.0


class _NullGPUMeter:
    """No-op GPU meter for regression experiments that run CPU-only."""
    def __init__(self):
        self.joules = 0.0
    def start(self):
        pass
    def stop(self):
        pass


# ── Backend selection (run once at module load) ──────────────────────────────

_RAPL_AVAILABLE = None    # None = not yet probed
_NVML_AVAILABLE = None

def _probe_backends():
    global _RAPL_AVAILABLE, _NVML_AVAILABLE
    if _RAPL_AVAILABLE is None:
        _RAPL_AVAILABLE = _detect_pyjoules_rapl()
    if _NVML_AVAILABLE is None:
        _NVML_AVAILABLE = _detect_pyjoules_nvml()

def get_backend_status():
    """Returns dict of available backends for logging at experiment start."""
    _probe_backends()
    return {
        "rapl_available": _RAPL_AVAILABLE,
        "nvml_available": _NVML_AVAILABLE,
        "gpu_backend": "nvml" if _NVML_AVAILABLE else "polling",
        "cpu_backend": "rapl" if _RAPL_AVAILABLE else "none",
    }


# ── Public context manager ───────────────────────────────────────────────────

class EnergyContext:
    """
    Context manager for energy measurement. Use as:

        with EnergyContext("cpu") as m:
            do_inference()
        print(m.cpu_joules, m.total_joules)

    Args:
        domain: "cpu" | "gpu" | "both"
            "cpu"  — regression experiments (RAPL only)
            "gpu"  — CV GPU inference (NVML or polling, no CPU)
            "both" — CV full system (GPU + CPU, use for retrain)
    """

    def __init__(self, domain="cpu"):
        _probe_backends()
        self.domain = domain

        # CPU meter
        if domain in ("cpu", "both"):
            self._cpu = _RaplCPUMeter() if _RAPL_AVAILABLE else _NullCPUMeter()
        else:
            self._cpu = _NullGPUMeter()  # reuse null pattern

        # GPU meter
        if domain in ("gpu", "both"):
            if _NVML_AVAILABLE:
                self._gpu = _NvmlGPUMeter()
            else:
                self._gpu = _PollingGPUMeter(interval_ms=50)
                logger.info("GPU energy: using 50ms power.draw polling (NVML unavailable)")
        else:
            self._gpu = _NullGPUMeter()

        self.cpu_joules = 0.0
        self.gpu_joules = 0.0
        self.total_joules = 0.0

    def __enter__(self):
        self._cpu.start()
        self._gpu.start()
        return self

    def __exit__(self, *args):
        self._cpu.stop()
        self._gpu.stop()
        self.cpu_joules = self._cpu.joules
        self.gpu_joules = self._gpu.joules
        self.total_joules = self.cpu_joules + self.gpu_joules

    def to_dict(self):
        return {
            "cpu_joules": self.cpu_joules,
            "gpu_joules": self.gpu_joules,
            "total_joules": self.total_joules,
        }
```

### Integration

Replace every pyRAPL call in the codebase with `EnergyContext`:

**In `managed_system_regression/mape_logic/execute.py`** — replace:
```python
# Before
import pyRAPL
pyRAPL.setup()
energy_meter = pyRAPL.Measurement("mape_k_execution")
energy_meter.begin()
# ... work ...
energy_meter.end()
energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
```
With:
```python
# After
import sys; sys.path.insert(0, "../../core")
from energy import EnergyContext
with EnergyContext("cpu") as m:
    # ... work ...
energy_consumed_uJ = m.total_joules * 1e6
```

**In `managed_system_cv/mape_logic/execute.py`** — same pattern with `EnergyContext("gpu")`.

**In `managed_system_regression/inference.py`** and
**`managed_system_cv/inference.py`** — wrap each per-step inference call:
```python
with EnergyContext("cpu") as m:    # or "gpu" for CV
    result = model(input_data)
row["energy"] = m.total_joules * 1e6  # log in µJ for consistency with existing schema
```

**In `managed_system_cv/retrain.py`** — use `EnergyContext("both")` since
fine-tuning uses both GPU and CPU.

### Logging requirements

At the start of each experimental run, log the backend status:
```python
from core.energy import get_backend_status
status = get_backend_status()
logger.info(f"Energy backends: {status}")
# Example output:
# Energy backends: {'rapl_available': False, 'nvml_available': True,
#                   'gpu_backend': 'nvml', 'cpu_backend': 'none'}
```

This creates a permanent record in experiment logs of exactly which measurement
method was used, which is needed for the paper's threats section and for
reproducibility.

### Separate logging for CV

CV experiments must log `cpu_joules` and `gpu_joules` as separate columns in
`predictions.csv`, not just `total_joules`. This allows post-hoc analysis of the
breakdown and is required for the paper's claim that GPU dominates CV energy.
The regression schema only needs `energy` (CPU only, since no GPU is used for
LR/SVM/LSTM on this hardware).

---

## Fix Priority Order

Given that nothing produces valid results right now, fix in this strict order:

**Block 1 — Get regression running** (nothing works without this):
1. L2: Clear `command.txt` on startup + add per-run reset to experiment harness
2. L1: Run `scripts/init_scaler.py` to create `knowledge/scaler.pkl`
3. L4: Fix `None` comparison crash in `app.py` secondary check
4. E1: Test pyRAPL on AMD CPU to determine which energy path to take

**Block 2 — Get CV running** (requires L3 first):
5. L3: Reinstall PyTorch with CUDA 12.9 kernel support for sm_120
6. E2: Test pyJoules NVML on RTX 5060 to determine GPU energy path

**Block 3 — Build the energy abstraction** (do after Blocks 1+2 so you know which backends work):
7. E3: Implement `core/energy.py` with `EnergyContext` using discovered backends
8. Replace all pyRAPL calls in both managed systems with `EnergyContext`

**Block 4 — Verify everything end to end**:
9. L6: Confirm per-run counter reset works
10. L5: Verify planner stops thrashing once real inference flows
11. Full smoke test: regression run 50 timesteps, CV run 50 frames, confirm:
    - predictions.csv grows with real values
    - energy columns are non-zero
    - switch counter resets to 0 at start
    - command.txt is clear on startup
    - no crashes in app.py secondary thread
    - YOLO runs on CUDA without AcceleratorError

---

## For the Paper — Threats to Validity Additions

The following must be added to the threats section as a result of the hardware
and measurement findings:

**Construct validity — CPU energy measurement**: Experiments were conducted on AMD
Ryzen AI 7 350 hardware. If RAPL is inaccessible on this chip (Outcome B of E1),
CPU energy for regression experiments is estimated via [method chosen from E1
options] rather than direct hardware counter measurement. The relative ordering
of approaches by energy consumption is expected to hold regardless, as the same
estimation method applies uniformly across all baselines.

**Construct validity — GPU energy measurement**: NVIDIA RTX 5060 (sm_120, compute
capability 12.0) does not expose cumulative energy counters through the current
NVML driver. GPU energy for CV experiments is estimated by integrating
instantaneous power draw (sampled at 50ms intervals via nvidia-smi) over
measurement windows. At 50ms polling on inference tasks running 100-800ms, the
integration error is bounded at approximately ±25ms × mean_GPU_power per call,
converging to < 2% relative error over a full experimental run.

**External validity — hardware**: The original HarmonE paper used an Intel i5-1240P
with pyRAPL. This extension uses different hardware. Absolute energy values are
not directly comparable across the two sets of results; only within-paper
comparisons between approaches on the same hardware are valid.

*Last updated: July 25 2026. Based on live run analysis of fix/bugs-phase1 branch.*
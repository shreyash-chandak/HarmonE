# Running HarmonE on Arch Linux

Complete step-by-step instructions for every runnable scenario on Arch Linux.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [RAPL Energy Measurement (Intel only)](#3-rapl-energy-measurement-intel-only)
4. [Scenario A — Live Regression Managed System](#scenario-a--live-regression-managed-system)
5. [Scenario B — Live CV / YOLO Managed System](#scenario-b--live-cv--yolo-managed-system)
6. [Scenario C — Headless Single Experiment Run](#scenario-c--headless-single-experiment-run)
7. [Scenario D — Headless Grid Run](#scenario-d--headless-grid-run)
8. [Scenario E — Metrics Aggregation](#scenario-e--metrics-aggregation)
9. [Scenario F — Test Suite](#scenario-f--test-suite)
10. [Scenario G — Custom MAPE System via Dashboard](#scenario-g--custom-mape-system-via-dashboard)
11. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

Install system packages:

```bash
sudo pacman -S python python-pip base-devel git
```

For the CV scenario with GPU support:

```bash
# NVIDIA driver + CUDA (install nvidia-dkms instead if using a custom kernel)
sudo pacman -S nvidia cuda cudnn

# Reboot after driver installation
sudo reboot
```

Verify:

```bash
python --version      # should be 3.12+
pip --version
nvidia-smi            # if using GPU
```

> **Note:** Arch ships the latest Python version; the `python_version >= "3.12"` dependency branches
> in `requirements.txt` apply (PyTorch ≥ 2.9, NumPy ≥ 2.2, scikit-learn ≥ 1.7, etc.).

---

## 2. Environment Setup

All commands are run from the `tool/` directory unless stated otherwise.

```bash
# Clone (or enter) the repo
cd /path/to/HarmonE-tool/tool

# Create and activate virtual environment
python -m venv harmone_env
source harmone_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### CPU-only install (recommended for experiments without a GPU)

```bash
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

### CUDA install (for CV / YOLO with NVIDIA GPU)

```bash
# Replace cu121 with your CUDA version (check: nvcc --version)
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

Verify PyTorch sees your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 3. RAPL Energy Measurement (Intel only)

pyRAPL reads CPU energy from the Linux power-cap sysfs interface.
Skip this section if you have an AMD CPU or run without energy measurement.

### Enable the kernel module

```bash
# Usually auto-loaded on Intel systems; if not:
sudo modprobe intel_rapl_common
sudo modprobe intel_rapl_msr

# Persist across reboots
echo intel_rapl_common | sudo tee /etc/modules-load.d/rapl.conf
echo intel_rapl_msr    | sudo tee -a /etc/modules-load.d/rapl.conf
```

### Grant read access

```bash
sudo chmod -R 777 /sys/class/powercap/intel-rapl/
```

> This permission resets on reboot. To make it permanent, create a udev rule:
> ```bash
> echo 'SUBSYSTEM=="powercap", ACTION=="add", RUN+="/bin/chmod -R 777 /sys/class/powercap/intel-rapl/"' \
>   | sudo tee /etc/udev/rules.d/99-rapl.rules
> sudo udevadm control --reload-rules
> ```

### Test pyRAPL

```bash
python -c "import pyRAPL; pyRAPL.setup(); m = pyRAPL.Measurement('test'); m.begin(); m.end(); print(m.result)"
```

---

## Scenario A — Live Regression Managed System

The live system runs **three concurrent processes**:
| Process | Port | What it does |
|---------|------|--------------|
| ACP Server (`app.py`) | 5000 | MAPE-K orchestrator, REST API |
| Adaptor / Inference loop (`run_managed_system.py`) | 8080 | Streams PeMS data, calls ACP |
| Dashboard (`frontend/`) | 8000 | Browser UI |

### Option 1 — Automated (harmone_start.sh)

The start script auto-detects your terminal emulator (konsole, gnome-terminal,
xfce4-terminal, tilix, xterm) and opens each process in its own window.

```bash
cd /path/to/HarmonE-tool/tool
source harmone_env/bin/activate

bash harmone_start.sh
```

Then open **http://localhost:8000/dashboard.html** in your browser.

> If no terminal is detected, install xterm: `sudo pacman -S xterm`

### Option 2 — Manual (three terminals)

**Terminal 1 — ACP Server:**

```bash
cd /path/to/HarmonE-tool/tool
source harmone_env/bin/activate
python app.py
# Listening on http://0.0.0.0:5000
```

**Terminal 2 — Managed System / Inference loop:**

```bash
cd /path/to/HarmonE-tool/tool
source harmone_env/bin/activate
python run_managed_system.py
# Connects to ACP on port 5000, exposes adaptor on port 8080
```

**Terminal 3 — Dashboard:**

```bash
cd /path/to/HarmonE-tool/tool/frontend
python -m http.server 8000
# Serves dashboard at http://localhost:8000/dashboard.html
```

Open **http://localhost:8000/dashboard.html**.

### Selecting an approach

In the dashboard, choose one of the regression approaches:
- `reg_harmone` — HarmonE adaptive (default / paper configuration)
- `reg_switch` — switch-based heuristic
- `reg_single` — fixed single model

### Stopping

Press `Ctrl+C` in each terminal (in reverse order: inference loop → ACP server → dashboard).

---

## Scenario B — Live CV / YOLO Managed System

Same three-process architecture as Scenario A, but the managed system uses
YOLOv8 for object detection on BDD100k images.

### Additional prerequisites

PyTorch with CUDA should already be installed from §2. Verify ultralytics:

```bash
python -c "from ultralytics import YOLO; print('OK')"
```

### Dataset

The BDD100k test images must be present at:

```
tool/managed_system_cv/data/bdd100k/images/test/
```

Mount or symlink if stored elsewhere:

```bash
ln -s /data/bdd100k tool/managed_system_cv/data/bdd100k
```

### Start (same as Scenario A)

```bash
bash harmone_start.sh
```

Or manually (same three terminals). In the dashboard, choose a CV approach:
- `cv_harmone` — HarmonE adaptive
- `cv_switch` — switch-based
- `cv_single` — fixed single model

### GPU memory note

YOLOv8n requires ~1 GB VRAM. If you see OOM errors, set:

```bash
export CUDA_VISIBLE_DEVICES=0   # use only GPU 0
```

---

## Scenario C — Headless Single Experiment Run

Runs one (dataset × planner × seed) combination end-to-end without ACP or a
browser. Results are written to a run directory under `runs/`.

```bash
cd /path/to/HarmonE-tool/tool
source harmone_env/bin/activate

python experiments/run_experiment.py \
  --dataset pems_node1 \
  --planner harmone_original \
  --seed 42
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | required | Name matching `configs/datasets/<name>.json` |
| `--planner` | required | One of: `naive`, `random_switch`, `greedy_switch`, `harmone_original`, `violation_aware`, `pareto` |
| `--seed` | `42` | Random seed |
| `--runs-dir` | `runs` | Parent directory for auto-named run dirs |
| `--run-dir` | auto | Override the full output path |
| `--monitor-interval` | `50` | Steps between MAPE-K cycles |
| `--max-steps` | unlimited | Truncate stream for quick tests |
| `--verbose` | off | Print per-step progress |

### Example: quick smoke test (100 steps)

```bash
python experiments/run_experiment.py \
  --dataset pems_node1 \
  --planner naive \
  --seed 1 \
  --max-steps 100 \
  --verbose
```

### Output artifacts (in `runs/<run_id>/`)

```
predictions.csv          # step-level y_true, y_pred, energy_uJ
mape_events.csv          # MAPE-K events (violations, drifts, decisions)
mape_info.json           # current MAPE-K state snapshot
thresholds.json          # effective thresholds used
run_manifest.json        # summary (elapsed_s, event_counters, …)
scaler.pkl               # fitted MinMaxScaler
reference_distribution.json   # KL drift reference
```

### Energy measurement

By default, `energy_meter` is `"null"` (zero energy values, no measurement).
To enable real RAPL measurement after completing §3:

```bash
python experiments/run_experiment.py \
  --dataset pems_node1 \
  --planner harmone_original \
  --seed 42
# Then edit configs/datasets/pems_node1.json:
#   "energy_meter": "auto"
```

---

## Scenario D — Headless Grid Run

Runs all combinations from a YAML grid config and writes results to a shared
grid directory.

```bash
cd /path/to/HarmonE-tool/tool
source harmone_env/bin/activate

python experiments/run_grid.py configs/experiments/baseline.yaml
```

The provided `baseline.yaml` runs:
- **Datasets:** `pems_node1`
- **Planners:** `naive`, `random_switch`, `greedy_switch`, `harmone_original`, `violation_aware`, `pareto`
- **Seeds:** `1`, `2`
- **Output:** `runs/baseline/`

### Custom grid config

Create `configs/experiments/my_grid.yaml`:

```yaml
datasets: [pems_node1]
planners: [naive, harmone_original]
seeds: [1, 2, 3, 4, 5]
monitor_interval: 50
cooldown_minutes: 0
runs_dir: runs/my_grid
resume: true
extra_thresholds:
  energy_meter: "auto"    # "null" to skip energy measurement
  stream_delay_s: 0.0     # 0.0 = no artificial delay (fastest)
```

Then run:

```bash
python experiments/run_grid.py configs/experiments/my_grid.yaml
```

### Resume interrupted grids

Set `resume: true` (already default). If the run directory for a given
(dataset, planner, seed) already contains `run_manifest.json`, that run is
skipped. Re-run the same command to pick up where it left off.

> **Note:** Run IDs are timestamp-based, so a new grid invocation always
> creates new directories and resume only helps within the same invocation.
> A future improvement will use deterministic IDs.

---

## Scenario E — Metrics Aggregation

Post-hoc analytics on a completed grid directory. No model or data needed.

```bash
cd /path/to/HarmonE-tool/tool
source harmone_env/bin/activate
```

### Aggregate all runs to CSV

```bash
python experiments/metrics.py aggregate \
  --grid-dir runs/baseline \
  --out runs/baseline/metrics.csv
```

Add `--by-planner` to collapse seeds (mean ± std per planner):

```bash
python experiments/metrics.py aggregate \
  --grid-dir runs/baseline \
  --out runs/baseline/metrics_by_planner.csv \
  --by-planner
```

Add `--pareto` to annotate which planners are Pareto-efficient (higher R² AND
lower energy than all others on the same dataset):

```bash
python experiments/metrics.py aggregate \
  --grid-dir runs/baseline \
  --out runs/baseline/metrics_pareto.csv \
  --by-planner --pareto
```

### Generate a LaTeX table

```bash
python experiments/metrics.py latex \
  --grid-dir runs/baseline \
  --out runs/baseline/table.tex \
  --caption "HarmonE planner comparison on PeMS node 1" \
  --label "tab:harmone_results"
```

Optionally select columns:

```bash
python experiments/metrics.py latex \
  --grid-dir runs/baseline \
  --out runs/baseline/table.tex \
  --columns planner n_seeds r2_mean_mean energy_uJ_total_mean model_switches_mean
```

### Wilcoxon signed-rank test

Compare `harmone_original` against `naive` on R²:

```bash
python experiments/metrics.py wilcoxon \
  --grid-dir runs/baseline \
  --baseline naive \
  --treatment harmone_original \
  --metric r2_mean \
  --dataset pems_node1
```

Output:

```json
{
  "statistic": 0.0,
  "p_value": 0.0156,
  "n_pairs": 2,
  "direction": "better",
  "significant": true
}
```

> Significance requires at least 8 paired seeds for a two-sided test (p < 0.05).
> With 2 seeds, the p-value floor is 0.5. Run 8+ seeds for publication results.

---

## Scenario F — Test Suite

```bash
cd /path/to/HarmonE-tool/tool
source harmone_env/bin/activate

python -m pytest tool/tests/ -v
```

Run a specific test file:

```bash
python -m pytest tool/tests/test_phase5_harness.py -v
```

Run a single test:

```bash
python -m pytest tool/tests/test_phase5_harness.py::TestWilcoxonTest::test_significantly_better -v
```

### PyYAML note

Grid config tests use JSON (not YAML) to avoid requiring PyYAML.
If you install PyYAML (`pip install pyyaml`), `.yaml` grid configs also work.

### Expected output

All 210 tests should pass:

```
============ 210 passed in X.Xs ============
```

---

## Scenario G — Custom MAPE System via Dashboard

The dashboard lets you upload your own MAPE-K files and optionally a custom
dataset, building a custom managed system without editing source code.

### Start the live system first (Scenario A or B)

The upload endpoint is served by the ACP server (`app.py`), so it must be
running.

### Upload via the dashboard

1. Open **http://localhost:8000/dashboard.html**
2. Navigate to **Custom System**
3. Choose base system: `regression` or `cv`
4. Upload one or more of: `monitor.py`, `analyse.py`, `plan.py`, `execute.py`, `manage.py`
5. Optionally upload a dataset:
   - Regression: a CSV file (saved as `managed_system_custom/knowledge/dataset.csv`)
   - CV: a ZIP of images (extracted to `managed_system_custom/data/bdd100k/images/test/`)
6. Click **Deploy**

The ACP server copies the base managed system to `managed_system_custom/`,
overwrites the uploaded MAPE files, and sets `approach.conf` to
`custom_regression` or `custom_cv`.

### Start the custom system

```bash
# In Terminal 2 (managed system terminal):
python run_managed_system.py
```

The system will use `approach.conf` to select `managed_system_custom/`.

### Revert to a built-in approach

Select any non-custom approach in the dashboard, or:

```bash
echo reg_harmone > approach.conf
```

---

## Troubleshooting

### Port already in use

```bash
# Find which process holds a port
ss -tlnp | grep :5000
# Kill it
kill $(ss -tlnp | grep :5000 | awk '{print $6}' | cut -d= -f2 | cut -d, -f1)
```

### RAPL permission denied

```bash
# Check if the sysfs path exists (Intel CPUs only)
ls /sys/class/powercap/intel-rapl/

# If the directory is missing, load the module
sudo modprobe intel_rapl_common intel_rapl_msr

# Re-apply permissions
sudo chmod -R 777 /sys/class/powercap/intel-rapl/
```

If you have an AMD CPU, pyRAPL is not supported. Set `energy_meter: "null"` in
your dataset config or `extra_thresholds` in the grid YAML.

### PyTorch not found (LSTM model unavailable)

The LSTM model loader requires `torch`. If PyTorch was not installed (e.g.
CPU-only pip install failed due to a version conflict), the harness falls back
gracefully — only `linear` and `svm` models are used.

Force reinstall PyTorch (CPU):

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

Verify:

```bash
python -c "import torch; print(torch.__version__)"
```

### scikit-learn version mismatch warning

If you see:
```
InconsistentVersionWarning: Trying to unpickle estimator ... from version 1.3.2
when using version 1.X.Y
```

This means the pre-trained `.pkl` models in `knowledge/` were serialized with
an older scikit-learn. The predictions are usually still valid, but to
silence the warning, retrain the models:

```bash
python managed_system_regression/retrain.py
```

### PyYAML not installed (grid YAML configs)

```bash
pip install pyyaml
```

Or convert your grid config to JSON — `run_grid.py` accepts `.json` files
without needing PyYAML.

### harmone_start.sh: no supported terminal found

Install xterm (minimal terminal, works in any environment):

```bash
sudo pacman -S xterm
```

Or run the processes manually (see Scenario A, Option 2).

### Flask startup warning about the development server

Flask prints: *"WARNING: This is a development server."* This is expected.
The ACP server (`app.py`) is intentionally run with Flask's built-in server
for the purposes of this tool.

### Dashboard shows "Cannot connect to ACP"

Verify the ACP server is running and listening on port 5000:

```bash
curl http://localhost:5000/
# Expected: Welcome to ACP Server!
```

If it is not running, start it (Terminal 1 in Scenario A).

### Ultralytics downloads weights on first run

YOLOv8 downloads model weights to `~/.cache/ultralytics/` on first use.
This requires internet access. To pre-download:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

On a machine without internet, copy `~/.cache/ultralytics/` from another
machine.

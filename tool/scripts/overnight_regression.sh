#!/usr/bin/env bash
# scripts/overnight_regression.sh — unattended regression pipeline in tmux:
#   1. single-model baselines (naive, naive_prt; seed 1) -> runs_calib/
#   2. calibrate E_m/E_M/min_accuracy/min_score from them (--write)
#   3. adaptive planners for seeds 1..5    -> runs_seed1/ .. runs_seed5/
#      (the baselines run once — their R² does not depend on the seed — and
#      the summary takes them from runs_calib; 2026-10-13)
#   4. summary table                       -> overnight/summary.txt
#
# Usage (from tool/, after git pull):
#   bash scripts/overnight_regression.sh          # pre-flight here, then starts tmux session "harmone"
#   tmux attach -t harmone                        # watch; detach with Ctrl-b d
#   tail -f overnight/pipeline.log                # step log (per-run output: overnight/grid.log)
#
# Resumable: re-running the same command continues where it stopped (finished
# steps are marked in overnight/; a half-finished grid resumes from its
# existing run manifests). Start from scratch with --fresh (moves old results
# into old/<timestamp>/).

set -uo pipefail

TOOL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TOOL_DIR" || exit 1
STATE="$TOOL_DIR/overnight"
LOG="$STATE/pipeline.log"
SESSION="harmone"
SEEDS=(1 2 3 4 5)
DATASETS=(pems_driftinduced spot_prices_driftinduced uci_electricity_driftinduced)
COOLDOWN_S=60
VENV="$TOOL_DIR/../.venv/bin/activate"

log() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }

# ── inner: the actual pipeline (runs inside tmux) ────────────────────────────
if [[ "${1:-}" == "--inner" ]]; then
    # shellcheck disable=SC1090
    source "$VENV"
    log "=== pipeline start (git $(git rev-parse --short HEAD)) ==="

    # 1. naive runs for the calibration
    if [[ ! -f "$STATE/naive.done" ]]; then
        log "[1/4] naive runs (seed 1)"
        if [[ -d runs_calib ]]; then mv runs_calib runs; fi   # resume a partial step
        bash concurrent_harness/run_concurrent_regression_grid.sh --seed 1 --only-naive >> "$STATE/grid.log" 2>&1 \
            || bash concurrent_harness/run_concurrent_regression_grid.sh --seed 1 --only-naive >> "$STATE/grid.log" 2>&1 \
            || die "naive grid failed twice — see runs/logs/failed_regression_concurrent.log"
        python3 - <<'PY' >> "$LOG" 2>&1 || die "naive runs failed, have no energy readings (RAPL?) or R² off the expected pool — see pipeline.log; not calibrating"
import glob, json, sys
# Expected static stream R² (laptop, 2026-10-13 configs). The LSTMs are deliberately
# under-trained to match the paper's pool, so they are sensitive to the machine; stop
# here rather than run five seeds on a different pool.
EXPECTED = {"pems_driftinduced": {"ridge": 0.7595, "svr": 0.7895, "lstm": 0.9213},
            "spot_prices_driftinduced": {"ridge": 0.7586, "svr": 0.7973, "lstm": 0.9328},
            "uci_electricity_driftinduced": {"ridge": 0.7596, "svr": 0.7898, "lstm": 0.9256}}
bad = []
for m in sorted(glob.glob("runs/*_naive_*_conc/run_manifest.json")):
    j = json.load(open(m))
    if "_naive_prt_" in m:
        continue
    e = j.get("energy_mJ")
    r2 = j["task_metrics"].get("r2")
    exp = EXPECTED.get(j["dataset"], {}).get(j.get("pin_model"))
    print(f"  {j['run_id']:45s} status={j.get('status')} energy_mJ={e} R2={r2} expected={exp}")
    if j.get("status") != "ok" or not e or e <= 0:
        bad.append(j["run_id"])
    elif exp is not None and (r2 is None or abs(r2 - exp) > 0.02):
        bad.append(f"{j['run_id']} (R2 {r2} vs expected {exp})")
if len(glob.glob("runs/*_naive_*_conc/run_manifest.json")) < 18 or bad:
    print("bad or missing naive runs:", bad); sys.exit(1)
PY
        mv runs runs_calib
        touch "$STATE/naive.done"
    fi

    # 2. calibration
    if [[ ! -f "$STATE/calibration.done" ]]; then
        log "[2/4] calibration from runs_calib"
        python3 experiments/calibrate_from_naive_runs.py --runs-dir runs_calib \
            $(printf -- '--dataset %s ' "${DATASETS[@]}") --write > "$STATE/calibration.json" 2>> "$LOG" \
            || die "calibration failed"
        python3 - <<'PY' >> "$LOG" 2>&1 || die "calibration values look wrong — configs left as written, check overnight/calibration.json"
import json, sys
for ds in ["pems_driftinduced", "spot_prices_driftinduced", "uci_electricity_driftinduced"]:
    c = json.load(open(f"configs/datasets/{ds}.json"))
    print(f"  {ds}: E_m={c['E_m']} E_M={c['E_M']} min_accuracy={c['min_accuracy']} min_score={c['min_score']}")
    if not (0 < c["E_m"] < c["E_M"]):
        sys.exit(1)
PY
        git diff configs/ > "$STATE/calibration.diff"
        touch "$STATE/calibration.done"
    fi

    # 3. five seeds
    for s in "${SEEDS[@]}"; do
        [[ -f "$STATE/seed$s.done" ]] && continue
        log "[3/4] seed $s grid"
        if [[ -d "runs_seed$s" ]]; then mv "runs_seed$s" runs; fi   # resume a partial seed
        if ! bash concurrent_harness/run_concurrent_regression_grid.sh --seed "$s" --skip-naive >> "$STATE/grid.log" 2>&1; then
            log "seed $s: some runs failed — retrying the missing ones once"
            bash concurrent_harness/run_concurrent_regression_grid.sh --seed "$s" --skip-naive >> "$STATE/grid.log" 2>&1 \
                || log "seed $s: still has failed runs (kept going) — see runs_seed$s/logs/failed_regression_concurrent.log"
        fi
        mv runs "runs_seed$s"
        touch "$STATE/seed$s.done"
        log "seed $s done ($(ls -d runs_seed$s/*_conc 2>/dev/null | wc -l) runs)"
        sleep "$COOLDOWN_S"
    done

    # 4. summary
    log "[4/4] summary"
    python3 scripts/aggregate_seeds.py --runs runs_calib $(printf 'runs_seed%s ' "${SEEDS[@]}") > "$STATE/summary.txt" 2>&1
    python3 experiments/drift_detection_eval.py --runs $(printf 'runs_seed%s ' "${SEEDS[@]}") \
        --csv "$STATE/drift_detection.csv" >> "$STATE/summary.txt" 2>&1
    cat "$STATE/summary.txt" >> "$LOG"
    log "=== pipeline DONE — baselines in runs_calib, planners in runs_seed1..5, summary in overnight/summary.txt ==="
    log "configs/ now hold the new calibration (uncommitted on this machine; see overnight/calibration.diff)"
    exit 0
fi

# ── outer: pre-flight checks, then launch tmux ───────────────────────────────
mkdir -p "$STATE"
command -v tmux >/dev/null || { echo "tmux not installed"; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "tmux session '$SESSION' already exists — attach with: tmux attach -t $SESSION"; exit 1; }
[[ -f "$VENV" ]] || { echo "venv not found at $VENV"; exit 1; }

if [[ "${1:-}" == "--fresh" ]]; then
    ts="$(date +%Y%m%d_%H%M%S)"
    mkdir -p "old/$ts"
    for d in runs runs_calib runs_seed* overnight; do
        [[ -e "$d" ]] && mv "$d" "old/$ts/"
    done
    mkdir -p "$STATE"
    echo "moved previous results to old/$ts/"
elif [[ ! -f "$STATE/naive.done" ]]; then
    # first start: move stale results out of the way (never on a resume)
    ts="$(date +%Y%m%d_%H%M%S)"
    for d in runs runs_seed*; do
        if [[ -e "$d" ]]; then mkdir -p "old/$ts"; mv "$d" "old/$ts/"; echo "moved $d -> old/$ts/"; fi
    done
fi

# RAPL must be readable (energy is the whole point); fix it now, while you're here
if ! cat /sys/class/powercap/intel-rapl:0/energy_uj >/dev/null 2>&1; then
    echo "RAPL not readable — running the permission setup (needs sudo):"
    sudo bash scripts/setup_energy_permissions.sh || { echo "RAPL setup failed"; exit 1; }
    cat /sys/class/powercap/intel-rapl:0/energy_uj >/dev/null 2>&1 || { echo "RAPL still not readable"; exit 1; }
fi
echo "RAPL ok: $(cat /sys/class/powercap/intel-rapl:0/energy_uj) uJ"

# configs must use the original per-prediction metering, data must exist
for ds in "${DATASETS[@]}"; do
    grep -q '"energy_metering": "per_prediction"' "configs/datasets/$ds.json" \
        || { echo "$ds.json is not on per_prediction metering — git pull?"; exit 1; }
done
for f in data/pems/flow_data_test_driftInduced.csv data/uci_electricity/uci_electricity_driftInduced.csv \
         data/spot_prices/spot_prices_driftInduced.csv; do
    [[ -f "$f" ]] || { echo "missing $f"; exit 1; }
done
if [[ ! -f "$STATE/calibration.done" ]] && ! git diff --quiet configs/; then
    echo "WARNING: configs/ has local changes (they will be overwritten by the calibration):"
    git diff --stat configs/
fi
echo "disk free: $(df -h . | awk 'NR==2{print $4}')"

tmux new-session -d -s "$SESSION" "bash '$TOOL_DIR/scripts/overnight_regression.sh' --inner; echo; echo 'pipeline finished — press enter to close'; read"
echo
echo "started in tmux session '$SESSION'."
echo "  watch:   tmux attach -t $SESSION   (detach: Ctrl-b d)"
echo "  log:     tail -f $LOG"
echo "  resume:  bash scripts/overnight_regression.sh   (after a crash/reboot)"

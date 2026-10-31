#!/usr/bin/env bash
# scripts/overnight_regression.sh — unattended regression pipeline in tmux:
#   1. calibration baselines (naive ridge/svr/lstm, seed 1)  -> runs_calib/
#   2. calibrate E_m/E_M/min_accuracy/min_score/s5_credit_cap from them (--write)
#   3. for seeds 1..5: 6 baselines + 8 adaptive planners x 3 datasets, interleaved
#      in an order shuffled by the seed, 60 s apart           -> runs_seed1/ .. runs_seed5/
#   4. summary table (baselines: mean ± sd over seeds)        -> overnight/summary.txt
#
# Step 1 exists because the adaptive planners need the calibration before they
# start, so the interleaved seed-1 baselines cannot provide it. Naive runs do not
# depend on the calibration.
#
# Usage (from tool/, after git pull):
#   bash scripts/overnight_regression.sh --fresh  # everything from scratch, seeds 1-5
#   bash scripts/overnight_regression.sh          # pre-flight here, then starts tmux session "harmone"
#   bash scripts/overnight_regression.sh --seeds "1"            # calibration + seed 1 only
#   bash scripts/overnight_regression.sh --seeds "1 2 3 4 5"    # later: resumes, runs seeds 2-5, summarises all
#   bash scripts/overnight_regression.sh --static-naive-only    # baselines in the seeds: naive only, no naive_prt
#   tmux attach -t harmone                        # watch; detach with Ctrl-b d
#   tail -f overnight/pipeline.log                # step log (per-run output: overnight/grid.log)
# Pipeline check without RAPL or tmux (no pacing, fixed energies, its own folder and
# a copy of the configs, nothing else is touched):
#   bash scripts/overnight_regression.sh --smoke-from runs_thinkpad_2810/runs_calib --out smoke_pipeline --seeds "1 2"
#
# Resumable: re-running the same command continues where it stopped (finished
# steps are marked in overnight/; a half-finished grid resumes from its
# existing run manifests). Start from scratch with --fresh (moves old results
# into old/<timestamp>/).

set -uo pipefail

TOOL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TOOL_DIR" || exit 1
# Passed from the outer invocation to --inner through the environment.
OUT="${HARMONE_OUT:-$TOOL_DIR}"
SMOKE_FROM="${HARMONE_SMOKE_FROM:-}"
STATIC_FLAG="${HARMONE_STATIC_FLAG:-}"
STATE="$OUT/overnight"
LOG="$STATE/pipeline.log"
SESSION="harmone"
read -r -a SEEDS <<< "${HARMONE_SEEDS:-1 2 3 4 5}"
DATASETS=(pems_driftinduced spot_prices_driftinduced uci_electricity_driftinduced)
COOLDOWN_S=60
VENV="${HARMONE_VENV:-$TOOL_DIR/../.venv/bin/activate}"
CONFIGS_DIR="$TOOL_DIR/configs"

log() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }

# ── inner: the actual pipeline ───────────────────────────────────────────────
if [[ "${1:-}" == "--inner" ]]; then
    # shellcheck disable=SC1090
    source "$VENV"
    GRID_COMMON=(--runs-dir "$OUT/runs")
    if [[ -n "$SMOKE_FROM" ]]; then
        COOLDOWN_S=0
        CONFIGS_DIR="$OUT/configs"
        GRID_COMMON+=(--configs-dir "$CONFIGS_DIR" --smoke-from "$SMOKE_FROM")
    fi
    export RUNS_DIR_ENV="$OUT/runs" CONFIGS_DIR_ENV="$CONFIGS_DIR"
    log "=== pipeline start (git $(git rev-parse --short HEAD)${SMOKE_FROM:+, SMOKE from $SMOKE_FROM}) ==="

    # 1. calibration baselines
    if [[ ! -f "$STATE/naive.done" ]]; then
        log "[1/4] calibration baselines: naive ridge/svr/lstm (seed 1)"
        if [[ -d "$OUT/runs_calib" ]]; then mv "$OUT/runs_calib" "$OUT/runs"; fi   # resume a partial step
        bash concurrent_harness/run_concurrent_regression_grid.sh --seed 1 --only-naive --static-naive-only "${GRID_COMMON[@]}" >> "$STATE/grid.log" 2>&1 \
            || bash concurrent_harness/run_concurrent_regression_grid.sh --seed 1 --only-naive --static-naive-only "${GRID_COMMON[@]}" >> "$STATE/grid.log" 2>&1 \
            || die "naive grid failed twice — see $OUT/runs/logs/failed_regression_concurrent.log"
        python3 - <<'PY' >> "$LOG" 2>&1 || die "naive runs failed, have no energy readings (RAPL?) or R² off the expected pool — see pipeline.log; not calibrating"
import glob, json, os, sys
# Expected static stream R² (laptop, 2026-10-13 configs). The LSTMs are deliberately
# under-trained to match the paper's pool, so they are sensitive to the machine; stop
# here rather than run five seeds on a different pool.
EXPECTED = {"pems_driftinduced": {"ridge": 0.7595, "svr": 0.7895, "lstm": 0.9213},
            "spot_prices_driftinduced": {"ridge": 0.7586, "svr": 0.7973, "lstm": 0.9328},
            "uci_electricity_driftinduced": {"ridge": 0.7596, "svr": 0.7898, "lstm": 0.9256}}
runs = os.environ["RUNS_DIR_ENV"]
bad = []
found = sorted(glob.glob(f"{runs}/*_naive_*_conc/run_manifest.json"))
for m in found:
    j = json.load(open(m))
    e = j.get("energy_mJ")
    r2 = j["task_metrics"].get("r2")
    exp = EXPECTED.get(j["dataset"], {}).get(j.get("pin_model"))
    print(f"  {j['run_id']:45s} status={j.get('status')} energy_mJ={e} R2={r2} expected={exp}")
    if j.get("status") != "ok" or not e or e <= 0:
        bad.append(j["run_id"])
    elif exp is not None and (r2 is None or abs(r2 - exp) > 0.02):
        bad.append(f"{j['run_id']} (R2 {r2} vs expected {exp})")
if len(found) < 9 or bad:
    print("bad or missing naive runs:", bad); sys.exit(1)
PY
        python3 scripts/check_runs.py --runs "$OUT/runs" > "$STATE/check_calib.txt" 2>&1 || die "baseline runs failed scripts/check_runs.py — see $STATE/check_calib.txt; not calibrating"
        mv "$OUT/runs" "$OUT/runs_calib"
        touch "$STATE/naive.done"
    fi

    # 2. calibration
    if [[ ! -f "$STATE/calibration.done" ]]; then
        log "[2/4] calibration from runs_calib (naive runs at seed 1, before the seed grids)"
        python3 experiments/calibrate_from_naive_runs.py --runs-dir "$OUT/runs_calib" --configs-dir "$CONFIGS_DIR/datasets" \
            $(printf -- '--dataset %s ' "${DATASETS[@]}") --write > "$STATE/calibration.json" 2>> "$LOG" \
            || die "calibration failed"
        python3 - <<'PY' >> "$LOG" 2>&1 || die "calibration values look wrong — configs left as written, check calibration.json"
import json, os, sys
for ds in ["pems_driftinduced", "spot_prices_driftinduced", "uci_electricity_driftinduced"]:
    c = json.load(open(f"{os.environ['CONFIGS_DIR_ENV']}/datasets/{ds}.json"))
    print(f"  {ds}: E_m={c['E_m']} E_M={c['E_M']} min_accuracy={c['min_accuracy']} min_score={c['min_score']} "
          f"s5_credit_cap={c.get('s5_credit_cap')}")
    if not (0 < c["E_m"] < c["E_M"]) or not c.get("s5_credit_cap", 0) > 0:
        sys.exit(1)
PY
        git diff configs/ > "$STATE/calibration.diff"
        touch "$STATE/calibration.done"
    fi

    # 3. seeds: baselines and adaptive planners interleaved
    for s in "${SEEDS[@]}"; do
        [[ -f "$STATE/seed$s.done" ]] && continue
        log "[3/4] seed $s grid"
        if [[ -d "$OUT/runs_seed$s" ]]; then mv "$OUT/runs_seed$s" "$OUT/runs"; fi   # resume a partial seed
        # shellcheck disable=SC2086
        if ! bash concurrent_harness/run_concurrent_regression_grid.sh --seed "$s" $STATIC_FLAG "${GRID_COMMON[@]}" >> "$STATE/grid.log" 2>&1; then
            log "seed $s: some runs failed — retrying them once"
            # shellcheck disable=SC2086
            bash concurrent_harness/run_concurrent_regression_grid.sh --seed "$s" $STATIC_FLAG "${GRID_COMMON[@]}" >> "$STATE/grid.log" 2>&1 \
                || log "seed $s: still has failed runs (kept going) — see runs_seed$s/logs/failed_regression_concurrent.log"
        fi
        mv "$OUT/runs" "$OUT/runs_seed$s"
        python3 scripts/check_runs.py --runs "$OUT/runs_seed$s" > "$STATE/check_seed$s.txt" 2>&1 || die "seed $s failed scripts/check_runs.py — see $STATE/check_seed$s.txt; stopping before the next seed"
        touch "$STATE/seed$s.done"
        log "seed $s done ($(ls -d "$OUT/runs_seed$s"/*_conc 2>/dev/null | wc -l) runs, checks passed)"
        sleep "$COOLDOWN_S"
    done

    # 4. summary (baselines from the seed folders, mean ± sd; runs_calib only calibrates)
    log "[4/4] summary"
    SEED_DIRS=()
    for s in "${SEEDS[@]}"; do SEED_DIRS+=("$OUT/runs_seed$s"); done
    python3 scripts/aggregate_seeds.py --runs "${SEED_DIRS[@]}" > "$STATE/summary.txt" 2>&1
    python3 experiments/drift_detection_eval.py --runs "${SEED_DIRS[@]}" \
        --csv "$STATE/drift_detection.csv" >> "$STATE/summary.txt" 2>&1
    cat "$STATE/summary.txt" >> "$LOG"
    log "=== pipeline DONE — calibration runs in runs_calib, all runs in runs_seed*, summary in $STATE/summary.txt ==="
    [[ -n "$SMOKE_FROM" ]] || log "configs/ now hold the new calibration (uncommitted on this machine; see overnight/calibration.diff)"
    exit 0
fi

# ── outer: pre-flight checks, then launch ────────────────────────────────────
FRESH=0
SEEDS_ARG="${SEEDS[*]}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fresh) FRESH=1; shift ;;
        --seeds) SEEDS_ARG="$2"; shift 2 ;;
        --static-naive-only) STATIC_FLAG="--static-naive-only"; shift ;;
        --smoke-from) SMOKE_FROM="$2"; shift 2 ;;
        --out) mkdir -p "$2"; OUT="$(cd "$2" && pwd)"; shift 2 ;;
        *) echo "unknown argument: $1 (use --fresh, --seeds \"1 2 3\", --static-naive-only, --smoke-from DIR, --out DIR)"; exit 1 ;;
    esac
done
read -r -a SEEDS <<< "$SEEDS_ARG"
STATE="$OUT/overnight"
LOG="$STATE/pipeline.log"
mkdir -p "$STATE"
[[ -f "$VENV" ]] || { echo "venv not found at $VENV (set HARMONE_VENV)"; exit 1; }
echo "seeds: ${SEEDS[*]}"

if [[ -z "$SMOKE_FROM" ]]; then
    command -v tmux >/dev/null || { echo "tmux not installed"; exit 1; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "tmux session '$SESSION' already exists — attach with: tmux attach -t $SESSION"; exit 1; }
fi

(
    cd "$OUT" || exit 1
    if [[ "$FRESH" -eq 1 ]]; then
        ts="$(date +%Y%m%d_%H%M%S)"
        mkdir -p "old/$ts"
        for d in runs runs_calib runs_seed* overnight; do
            [[ -e "$d" ]] && mv "$d" "old/$ts/"
        done
        echo "moved previous results to old/$ts/"
    elif [[ ! -f "$STATE/naive.done" ]]; then
        # first start: move stale results out of the way (never on a resume)
        ts="$(date +%Y%m%d_%H%M%S)"
        for d in runs runs_seed*; do
            if [[ -e "$d" ]]; then mkdir -p "old/$ts"; mv "$d" "old/$ts/"; echo "moved $d -> old/$ts/"; fi
        done
    fi
)
mkdir -p "$STATE"

if [[ -n "$SMOKE_FROM" ]]; then
    # Its own copy of the configs: the calibration must not touch configs/.
    mkdir -p "$OUT/configs"
    cp -r configs/datasets "$OUT/configs/"
    [[ -d "$SMOKE_FROM" ]] || { echo "no $SMOKE_FROM"; exit 1; }
    HARMONE_OUT="$OUT" HARMONE_SMOKE_FROM="$SMOKE_FROM" HARMONE_STATIC_FLAG="$STATIC_FLAG" \
        HARMONE_SEEDS="${SEEDS[*]}" HARMONE_VENV="$VENV" bash "$TOOL_DIR/scripts/overnight_regression.sh" --inner
    exit $?
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
# base weights must be the shared ones (a missing file would be retrained on
# this machine, giving a different pool); print checksums to compare machines
for ds in "${DATASETS[@]}"; do
    for w in $(grep -o '"weights_path": "[^"]*"' "configs/datasets/$ds.json" | cut -d'"' -f4); do
        [[ -f "$w" ]] || { echo "missing model weights $w — copy managed_system_regression/models/ from the other machine"; exit 1; }
    done
done
echo "model checksums (compare across machines):"
(cd managed_system_regression/models && sha256sum *.pt *.pkl | cut -c1-12,65-)
if [[ ! -f "$STATE/calibration.done" ]] && ! git diff --quiet configs/; then
    echo "WARNING: configs/ has local changes (they will be overwritten by the calibration):"
    git diff --stat configs/
fi
echo "disk free: $(df -h . | awk 'NR==2{print $4}')"

tmux new-session -d -s "$SESSION" "HARMONE_OUT='$OUT' HARMONE_STATIC_FLAG='$STATIC_FLAG' HARMONE_SEEDS='${SEEDS[*]}' HARMONE_VENV='$VENV' bash '$TOOL_DIR/scripts/overnight_regression.sh' --inner; echo; echo 'pipeline finished — press enter to close'; read"
echo
echo "started in tmux session '$SESSION'."
echo "  watch:   tmux attach -t $SESSION   (detach: Ctrl-b d)"
echo "  log:     tail -f $LOG"
echo "  resume:  bash scripts/overnight_regression.sh --seeds \"${SEEDS[*]}\"   (after a crash/reboot)"

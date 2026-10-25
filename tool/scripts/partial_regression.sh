#!/usr/bin/env bash
# scripts/partial_regression.sh — partial rerun on a machine that already has
# its single-model baselines (2026-10-26):
#   1. runs/ holds the baselines (the old runs_calib, renamed); run only the
#      naive / naive_prt runs whose manifest is missing, check them, then
#      runs/ -> runs_calib/
#   2. calibrate from runs_calib (--write): E_m/E_M/min_accuracy/min_score as
#      before (the naive runs are unchanged) plus s5_credit_cap
#   3. all 8 adaptive planners x 3 datasets for ONE seed -> runs_seed<N>/,
#      checked with check_runs.py; then a summary (aggregate_seeds + s5_report)
#
# Usage (from tool/, inside tmux, after git checkout -- configs/ && git pull):
#   source ../.venv/bin/activate
#   bash scripts/partial_regression.sh             # seed 1
#   bash scripts/partial_regression.sh --seed 2
# Resumable: re-run the same command; finished steps are marked in
# overnight_partial/, a half-finished grid resumes from its run manifests.
# Log: overnight_partial/pipeline.log (per-run output: overnight_partial/grid.log)

set -uo pipefail

TOOL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TOOL_DIR" || exit 1
SEED=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        *) echo "unknown argument: $1 (use --seed N)"; exit 1 ;;
    esac
done
STATE="$TOOL_DIR/overnight_partial"
LOG="$STATE/pipeline.log"
DATASETS=(pems_driftinduced spot_prices_driftinduced uci_electricity_driftinduced)
mkdir -p "$STATE"

log() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }

# ── pre-flight ────────────────────────────────────────────────────────────────
grep -q violation_aware_gated concurrent_harness/run_concurrent_regression_grid.sh \
    || die "old checkout (no violation_aware_gated in the grid) — git pull first"
cat /sys/class/powercap/intel-rapl:0/energy_uj >/dev/null 2>&1 \
    || die "RAPL not readable — sudo bash scripts/setup_energy_permissions.sh"
if [[ ! -f "$STATE/naive.done" ]]; then
    [[ -d runs ]] || die "no runs/ — rename the existing baselines folder to runs first"
    [[ -e runs_calib ]] && die "runs_calib already exists — move it away (the baselines must be in runs/)"
fi
if [[ -e "runs_seed$SEED" && ! -f "$STATE/seed$SEED.started" ]]; then
    die "runs_seed$SEED already exists — move it away first"
fi
if [[ ! -f "$STATE/calibration.done" ]] && ! git diff --quiet configs/; then
    log "WARNING: configs/ has local changes; the calibration will overwrite its keys"
fi
log "=== partial pipeline start (git $(git rev-parse --short HEAD), seed $SEED) ==="

# ── 1. missing baselines ──────────────────────────────────────────────────────
if [[ ! -f "$STATE/naive.done" ]]; then
    missing=()
    for ds in "${DATASETS[@]}"; do
        for p in naive naive_prt; do
            for m in ridge svr lstm; do
                id="${ds}_${p}_${m}_conc"
                [[ -f "runs/$id/run_manifest.json" ]] || missing+=("$id")
            done
        done
    done
    log "[1/3] baselines missing in runs/: ${missing[*]:-none}"
    if [[ ${#missing[@]} -gt 0 ]]; then
        bash concurrent_harness/run_concurrent_regression_grid.sh --seed 1 --only-naive >> "$STATE/grid.log" 2>&1 \
            || die "baseline runs failed — see runs/logs/failed_regression_concurrent.log"
    fi
    n=$(ls runs/*_naive_*_conc/run_manifest.json 2>/dev/null | wc -l)
    [[ "$n" -eq 18 ]] || die "$n of 18 baseline manifests in runs/"
    python3 scripts/check_runs.py --runs runs > "$STATE/check_calib.txt" 2>&1 \
        || die "baselines failed scripts/check_runs.py — see overnight_partial/check_calib.txt"
    grep -E "spot_prices_driftinduced_naive_prt_(ridge|svr)" "$STATE/check_calib.txt" | tee -a "$LOG"
    mv runs runs_calib
    touch "$STATE/naive.done"
fi

# ── 2. calibration ────────────────────────────────────────────────────────────
if [[ ! -f "$STATE/calibration.done" ]]; then
    log "[2/3] calibration from runs_calib"
    python3 experiments/calibrate_from_naive_runs.py --runs-dir runs_calib \
        $(printf -- '--dataset %s ' "${DATASETS[@]}") --write > "$STATE/calibration.json" 2>> "$LOG" \
        || die "calibration failed"
    python3 - <<'PY' >> "$LOG" 2>&1 || die "calibration values look wrong — check overnight_partial/calibration.json"
import json, sys
for ds in ["pems_driftinduced", "spot_prices_driftinduced", "uci_electricity_driftinduced"]:
    c = json.load(open(f"configs/datasets/{ds}.json"))
    print(f"  {ds}: E_m={c['E_m']} E_M={c['E_M']} min_accuracy={c['min_accuracy']} min_score={c['min_score']} "
          f"s5_credit_cap={c.get('s5_credit_cap')}")
    if not (0 < c["E_m"] < c["E_M"]) or not c.get("s5_credit_cap", 0) > 0:
        sys.exit(1)
PY
    git diff configs/ > "$STATE/calibration.diff"
    touch "$STATE/calibration.done"
fi

# ── 3. adaptive planners, one seed ────────────────────────────────────────────
if [[ ! -f "$STATE/seed$SEED.done" ]]; then
    log "[3/3] seed $SEED: all adaptive planners"
    touch "$STATE/seed$SEED.started"
    if [[ -d "runs_seed$SEED" ]]; then mv "runs_seed$SEED" runs; fi   # resume a partial seed
    if ! bash concurrent_harness/run_concurrent_regression_grid.sh --seed "$SEED" --skip-naive >> "$STATE/grid.log" 2>&1; then
        log "seed $SEED: some runs failed — retrying the missing ones once"
        bash concurrent_harness/run_concurrent_regression_grid.sh --seed "$SEED" --skip-naive >> "$STATE/grid.log" 2>&1 \
            || log "seed $SEED: still has failed runs — see runs_seed$SEED/logs/failed_regression_concurrent.log"
    fi
    mv runs "runs_seed$SEED"
    python3 scripts/check_runs.py --runs "runs_seed$SEED" --naive-from runs_calib > "$STATE/check_seed$SEED.txt" 2>&1 \
        || die "seed $SEED failed scripts/check_runs.py — see overnight_partial/check_seed$SEED.txt"
    touch "$STATE/seed$SEED.done"
    log "seed $SEED done ($(ls -d runs_seed$SEED/*_conc 2>/dev/null | wc -l) runs, checks passed)"
fi

# ── summary ───────────────────────────────────────────────────────────────────
python3 scripts/aggregate_seeds.py --runs runs_calib "runs_seed$SEED" > "$STATE/summary.txt" 2>&1
python3 scripts/s5_report.py --runs "runs_seed$SEED" >> "$STATE/summary.txt" 2>&1
log "=== DONE — baselines in runs_calib, planners in runs_seed$SEED, summary in overnight_partial/summary.txt ==="
log "configs/ hold the new calibration (uncommitted; see overnight_partial/calibration.diff)"

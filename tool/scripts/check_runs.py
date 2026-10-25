"""
scripts/check_runs.py — sanity checks over finished regression runs (smoke or
real), so a broken run is caught before hours of seeds are spent on it
(2026-10-18).

Per run (runs/<dataset>_<planner>[_<model>]_conc/):
  FAIL  status not ok; wall time per row outside (delay, 1.5 x delay + 4 ms) on
        real runs (catches doubled or missing pacing); rows != stream length; MAPE cycles != rows // monitor_interval;
        mape_events rows != cycles; drift checks missing; reload failures; NaN
        predictions; a pinned run served another model; batches served by mixed
        models; naive runs that adapted; naive static R² off the expected pool;
        PRT runs whose retrain count (incl. failed/busy-skipped) != the schedule;
        failed retrains; retrained weights that never reached the stream
        (predictions after the first retrain identical to the naive run's);
        missing/zero energy; Python tracebacks in the run log; smoke runs with
        busy-skipped retrains (they wait for retrains).
  WARN  ridge/SVR retrain gain over the 0.075 cap; naive_prt R² over PRT_MAX; a retrain that lowers R²;
        busy-skipped adaptations (real runs); "no drift result from t2" log lines;
        energy readings missing on > 1 % of rows (real runs).

Usage (from tool/):
    python3 scripts/check_runs.py --runs runs_smoke
    python3 scripts/check_runs.py --runs runs_calib runs_seed1
Exit code 1 if anything FAILs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

DATASETS = ["pems_driftinduced", "spot_prices_driftinduced", "uci_electricity_driftinduced"]
# Expected static stream R² of the pool (same as overnight_regression.sh's guard).
EXPECTED = {"pems_driftinduced": {"ridge": 0.7595, "svr": 0.7895, "lstm": 0.9213},
            "spot_prices_driftinduced": {"ridge": 0.7586, "svr": 0.7973, "lstm": 0.9328},
            "uci_electricity_driftinduced": {"ridge": 0.7596, "svr": 0.7898, "lstm": 0.9256}}
GAIN_CAP = 0.075  # user cap on ridge/SVR retraining gains
# user caps on naive_prt stream R² (2026-10-26)
PRT_MAX = {"spot_prices_driftinduced": {"ridge": 0.81, "svr": 0.85}}


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return 1.0 - float(((y - p) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def stream_length(ds: str, cache: dict) -> int:
    if ds not in cache:
        from experiments.run_experiment import _build_adapter
        cfg = json.loads((_TOOL_DIR / "configs" / "datasets" / f"{ds}.json").read_text(encoding="utf-8"))
        cfg["stream_delay_s"] = 0.0
        cache[ds] = sum(1 for _ in _build_adapter(cfg, _TOOL_DIR / "configs").stream())
    return cache[ds]


def prt_schedule(n_rows: int, mi: int, p: int) -> list[int]:
    """Cycle steps at which PRT fires (core/planners/base.py prt_due)."""
    return [s for s in range(mi, n_rows + 1, mi) if s // p > (s - mi) // p]


def load_preds(run: Path) -> list[dict]:
    with open(run / "predictions.csv", newline="") as f:
        return list(csv.DictReader(f))


def check_run(run: Path, naive_dirs: list[Path], cache: dict) -> tuple[dict, list[str], list[str]]:
    fail, warn = [], []
    j = json.loads((run / "run_manifest.json").read_text())
    ds, planner, pin = j["dataset"], j["planner"], j.get("pin_model")
    thr = json.loads((run / "thresholds.json").read_text(encoding="utf-8"))
    mi = int(thr.get("monitor_interval", 50))
    smoke = bool(j.get("smoke"))
    ec = j.get("event_counters", {})
    tm = j.get("task_metrics", {})
    rows = load_preds(run)
    n = len(rows)
    info = {"run": run.name, "smoke": smoke, "rows": n, "r2": tm.get("r2"),
            "sw": ec.get("model_switches", 0), "rt": ec.get("retrains", 0), "rt_exp": "",
            "vmr": ec.get("vmr_events", 0), "busy": ec.get("adaptation_skipped_busy", 0),
            "share": ""}

    if j.get("status") != "ok":
        fail.append(f"status {j.get('status')}: {j.get('failure_reason')}")
    exp_n = stream_length(ds, cache)
    if n != exp_n:
        fail.append(f"{n} prediction rows, stream has {exp_n}")
    if j.get("mape_cycles") != n // mi:
        fail.append(f"{j.get('mape_cycles')} MAPE cycles, expected {n // mi}")
    with open(run / "mape_events.csv", newline="") as f:
        n_ev = sum(1 for _ in csv.DictReader(f))
    if n_ev != j.get("mape_cycles"):
        fail.append(f"mape_events.csv has {n_ev} rows for {j.get('mape_cycles')} cycles")
    if int(j.get("drift_checks", 0)) < n // mi - 1:
        fail.append(f"only {j.get('drift_checks')} drift checks for {n // mi} batches")
    if j.get("reload_failures", 0):
        fail.append(f"{j['reload_failures']} weight reload failures")
    yp = [float(r["y_pred"]) for r in rows]
    if any(not math.isfinite(v) for v in yp):
        fail.append("non-finite predictions")
    served = [r["active_model"] for r in rows]
    shares = {m: served.count(m) / max(n, 1) for m in ("ridge", "svr", "lstm")}
    info["share"] = "/".join(f"{100 * shares[m]:.0f}" for m in ("ridge", "svr", "lstm"))
    if pin and set(served) != {pin}:
        fail.append(f"pinned to {pin} but served {sorted(set(served))}")
    mixed = sum(1 for a in range(0, n - n % mi, mi) if len(set(served[a:a + mi])) > 1)
    if mixed:
        fail.append(f"{mixed} batches served by more than one model")

    # pacing: wall time per row = one stream_delay_s + inference + overhead
    delay = float(thr.get("stream_delay_s", 0.0))
    if not smoke and delay > 0 and n:
        per_row = float(j.get("elapsed_s", 0.0)) / n
        if not delay < per_row < 1.5 * delay + 0.004:
            fail.append(f"{1000 * per_row:.1f} ms per row for a {1000 * delay:.1f} ms delay")

    # energy
    if not j.get("energy_mJ") or j["energy_mJ"] <= 0:
        fail.append(f"energy_mJ {j.get('energy_mJ')}")
    inv = int(tm.get("energy_invalid_steps", 0) or 0)
    if not smoke and inv > 0.01 * n:
        warn.append(f"{inv} rows without an energy reading")

    # run log
    for logs in (run.parent / "logs", run.parent.parent / "overnight"):
        lp = logs / f"{run.name}.log"
        if lp.exists():
            text = lp.read_text(errors="replace")
            if "Traceback" in text:
                fail.append(f"{text.count('Traceback')} traceback(s) in {lp.name}")
            k = text.count("no drift result from t2")
            if k:
                warn.append(f"{k}x 'no drift result from t2'")
            break

    # planner-specific
    if planner == "naive":
        if info["sw"] or info["rt"] or info["vmr"]:
            fail.append("naive run adapted")
        exp = EXPECTED.get(ds, {}).get(pin)
        if exp is not None and tm.get("r2") is not None and abs(tm["r2"] - exp) > 0.02:
            fail.append(f"static R² {tm['r2']:.4f}, expected {exp}")
    if planner in ("naive_prt", "random_switch_prt"):
        sched = prt_schedule(n, mi, int(thr.get("prt_interval", 3200)))
        info["rt_exp"] = len(sched)
        done = info["rt"] + ec.get("retrain_skipped", 0) + info["busy"]
        if done != len(sched):
            fail.append(f"{done} PRT retrains (ok+failed+busy), schedule has {len(sched)}")
        if ec.get("retrain_skipped", 0):
            fail.append(f"{ec['retrain_skipped']} failed retrains")
        if planner == "naive_prt" and sched:
            base = None
            for d in naive_dirs:
                bp = d / f"{ds}_naive_{pin}_conc"
                if (bp / "predictions.csv").exists():
                    base = load_preds(bp)
                    break
            if base is None:
                warn.append("no naive run to compare against")
            else:
                start = sched[0] + 2 * mi  # first retrain applied by then
                diff = np.mean([abs(float(a["y_pred"]) - float(b["y_pred"])) > 1e-6
                                for a, b in zip(rows[start:], base[start:])])
                if diff < 0.5:
                    fail.append(f"only {100 * diff:.0f} % of predictions after the first retrain differ "
                                "from the naive run — new weights not served?")
                gain = tm["r2"] - r2([float(r["y_true"]) for r in base], [float(r["y_pred"]) for r in base])
                info["gain"] = gain
                if pin in ("ridge", "svr") and gain > GAIN_CAP:
                    warn.append(f"retrain gain {gain:+.3f} over the {GAIN_CAP} cap")
                cap = PRT_MAX.get(ds, {}).get(pin)
                if cap is not None and tm["r2"] > cap:
                    warn.append(f"PRT R² {tm['r2']:.4f} over the {cap} cap")
                if gain < -0.01:
                    warn.append(f"retraining lowers R² ({gain:+.3f})")
    if info["busy"]:
        (fail if smoke else warn).append(f"{info['busy']} adaptations skipped (retrain busy)")
    return info, fail, warn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--naive-from", nargs="*", default=None,
                   help="Dirs with the naive runs to compare naive_prt against (default: --runs, then runs_calib).")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # R² on a Windows console
    run_dirs = [Path(r) for r in args.runs]
    naive_dirs = [Path(r) for r in (args.naive_from if args.naive_from is not None
                                    else args.runs + ["runs_calib"])]
    cache: dict = {}
    n_fail = 0
    print("| Run | Rows | R² | Switches | Retrains (schedule) | VMR | Busy | ridge/svr/lstm % | Result |")
    print("|---|---|---|---|---|---|---|---|---|")
    notes = []
    for rd in run_dirs:
        for mf in sorted(rd.glob("*_conc/run_manifest.json")):
            try:
                info, fail, warn = check_run(mf.parent, naive_dirs, cache)
            except Exception as exc:  # a malformed run is itself a failure
                info, fail, warn = {"run": mf.parent.name}, [f"checker crashed: {exc!r}"], []
            n_fail += bool(fail)
            res = "FAIL" if fail else ("warn" if warn else "ok")
            rt = f"{info.get('rt', '')}" + (f" ({info['rt_exp']})" if info.get("rt_exp") != "" else "")
            r2v = f"{info['r2']:.4f}" if isinstance(info.get("r2"), float) else "?"
            if "gain" in info:
                r2v += f" ({info['gain']:+.3f})"
            print(f"| {rd.name}/{info['run']} | {info.get('rows', '')} | {r2v} | {info.get('sw', '')} | {rt} "
                  f"| {info.get('vmr', '')} | {info.get('busy', '')} | {info.get('share', '')} | {res} |")
            notes += [f"FAIL {rd.name}/{info['run']}: {m}" for m in fail]
            notes += [f"warn {rd.name}/{info['run']}: {m}" for m in warn]
    print()
    print("\n".join(notes) if notes else "no problems found")
    print(f"\n{n_fail} run(s) failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

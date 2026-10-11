"""
experiments/power_diagnostic.py — CPU package power (RAPL, socket 0) in
fixed states, to separate idle power, harness overhead and inference
(2026-10-12, stage 1 diagnostics).

States (each measured for --seconds after a settle period, --repeats times):
  idle            nothing of ours running
  harness_paused  run_concurrent.py up (inference process + manager threads)
                  but inference paused: a huge stream delay after the first
                  prediction, so both processes just wait
  running_lstm    normal run, naive pinned LSTM (per-prediction reload)
  running_ridge   normal run, naive pinned ridge (extra: ridge also read ~100 W)

Run directories go under <out>/power_runs/ and are killed after measuring.
Output: <out>/power.csv (one row per state x repeat, plus per-second watts in
<out>/power_trace.csv).

Usage (from tool/, on the machine to measure, with nothing else running):
    python3 experiments/power_diagnostic.py --out diagnostics/<folder> --dataset uci_electricity_driftinduced
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))
from core.energy import read_package_counter, package_energy_between  # noqa: E402


def measure(seconds: float) -> tuple[float | None, list[float]]:
    """Package energy (J) over `seconds`, plus per-second watts."""
    start = prev = read_package_counter()
    if start is None:          # no RAPL (--allow-no-rapl test runs): keep the timing
        time.sleep(seconds)
        return None, []
    watts = []
    t_end = time.monotonic() + seconds
    t_prev = time.monotonic()
    while time.monotonic() < t_end:
        time.sleep(min(1.0, max(0.0, t_end - time.monotonic())))
        cur = read_package_counter()
        now = time.monotonic()
        d = package_energy_between(prev, cur)
        if d is not None and now > t_prev:
            watts.append(d / 1e6 / (now - t_prev))
        prev, t_prev = cur, now
    total = package_energy_between(start, read_package_counter())
    return (None if total is None else total / 1e6), watts


def _launch(out: Path, dataset: str, model: str, run_id: str, delay: float | None) -> subprocess.Popen:
    cmd = [sys.executable, str(_TOOL_DIR / "concurrent_harness" / "run_concurrent.py"),
           "--dataset", dataset, "--planner", "naive", "--pin-model", model,
           "--runs-dir", str(out / "power_runs"), "--run-id", run_id]
    if delay is not None:
        cmd += ["--stream-delay-s", str(delay)]
    log = open(out / "power_runs" / f"{run_id}.log", "w")
    return subprocess.Popen(cmd, cwd=str(_TOOL_DIR), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)


def _rows_written(out: Path, run_id: str) -> int:
    p = out / "power_runs" / run_id / "predictions.csv"
    try:
        with open(p) as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def _wait_rows(out: Path, run_id: str, proc: subprocess.Popen, rows: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _rows_written(out, run_id) >= rows:
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="uci_electricity_driftinduced",
                   help="Longest regression stream, so a 30 s window fits inside one run.")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--settle", type=float, default=10.0)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--allow-no-rapl", action="store_true",
                   help="Testing only: run the states even without RAPL (energies are empty).")
    args = p.parse_args()

    out = Path(args.out)
    (out / "power_runs").mkdir(parents=True, exist_ok=True)
    if read_package_counter() is None and not args.allow_no_rapl:
        sys.exit("RAPL package counter not readable — run scripts/setup_energy_permissions.sh first.")

    results, trace = [], []

    def record(state: str, rep: int, note: str = "") -> None:
        energy_j, watts = measure(args.seconds)
        avg = None if energy_j is None else energy_j / args.seconds
        results.append({"state": state, "repeat": rep, "seconds": args.seconds,
                        "energy_J": None if energy_j is None else round(energy_j, 3),
                        "avg_W": None if avg is None else round(avg, 2),
                        "min_W": round(min(watts), 2) if watts else None,
                        "max_W": round(max(watts), 2) if watts else None, "note": note})
        trace.extend({"state": state, "repeat": rep, "second": i + 1, "W": round(w, 2)}
                     for i, w in enumerate(watts))
        print(f"  {state:15s} repeat {rep}: {avg if avg is None else round(avg, 1)} W  {note}", flush=True)

    for rep in range(1, args.repeats + 1):
        print(f"repeat {rep}", flush=True)
        time.sleep(args.settle)
        record("idle", rep)

        run_id = f"paused_{rep}"
        proc = _launch(out, args.dataset, "lstm", run_id, delay=3600.0)
        ok = _wait_rows(out, run_id, proc, 1, timeout=600)
        time.sleep(args.settle)
        record("harness_paused", rep, "" if ok else "WARNING: no prediction row seen before measuring")
        _kill(proc)

        for model in ("lstm", "ridge"):
            run_id = f"running_{model}_{rep}"
            proc = _launch(out, args.dataset, model, run_id, delay=None)
            ok = _wait_rows(out, run_id, proc, 2000, timeout=600)
            n0 = _rows_written(out, run_id)
            record(f"running_{model}", rep, "" if ok else "WARNING: run not streaming before measuring")
            n1 = _rows_written(out, run_id)
            results[-1]["note"] = (results[-1]["note"] + f" predictions during window: {n1 - n0}").strip()
            _kill(proc)

    with open(out / "power.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    with open(out / "power_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["state", "repeat", "second", "W"])
        w.writeheader()
        w.writerows(trace)
    print(f"wrote {out / 'power.csv'}")


if __name__ == "__main__":
    main()

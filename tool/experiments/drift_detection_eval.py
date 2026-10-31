"""
experiments/drift_detection_eval.py — precision/recall of the drift detector
against the injected drift segments of the *_driftinduced streams.

Ground truth: the rows where a *_driftinduced stream differs from its clean
sibling (pems / spot_prices / uci_electricity), merged across gaps shorter
than --merge-gap rows (spot has zero prices that the scale+shift leaves at 0).
The detector judges a trailing window of drift_window_size rows, so a segment
[s, e] is "visible" to checks at steps s .. e + window.

From each run's mape_events.csv (one row per MAPE cycle, step = rows seen):
  - cycle level: a flagged cycle is a true positive if its step lies in a
    visible span; precision = TP / flagged, recall = TP cycles / visible
    cycles (cycle-level recall is low by nature: the cooldown after an
    adaptation suppresses checks inside a segment);
  - segment level: a segment counts as detected if at least one cycle in its
    visible span is flagged; also the median delay (rows) to the first flag.
Both for raw detections (drift_detected) and for accuracy-confirmed ones
(drift_confirmed, what planners act on).

Usage (from tool/):
    python3 experiments/drift_detection_eval.py --runs runs_seed1 runs_seed2 ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

CLEAN = {
    "pems_driftinduced": "pems",
    "spot_prices_driftinduced": "spot_prices",
    "uci_electricity_driftinduced": "uci_electricity",
}


def injected_segments(dataset: str, merge_gap: int) -> tuple[list[tuple[int, int]], int]:
    from experiments.run_experiment import _build_adapter
    cfg_dir = _TOOL_DIR / "configs"
    load = lambda name: [s.ground_truth for s in _build_adapter(
        json.loads((cfg_dir / "datasets" / f"{name}.json").read_text()), cfg_dir).stream()]
    a, b = np.asarray(load(CLEAN[dataset]), float), np.asarray(load(dataset), float)
    n = min(len(a), len(b))
    diff = np.flatnonzero(np.abs(a[:n] - b[:n]) > 1e-9)
    segs: list[list[int]] = []
    for i in diff:
        if segs and i - segs[-1][1] <= merge_gap:
            segs[-1][1] = int(i)
        else:
            segs.append([int(i), int(i)])
    return [tuple(s) for s in segs], n


def evaluate(run_dirs: list[Path], merge_gap: int) -> list[dict]:
    seg_cache: dict[str, tuple] = {}
    out = []
    for run_dir in run_dirs:
        for ev_path in sorted(run_dir.glob("*_conc/mape_events.csv")):
            man = json.loads((ev_path.parent / "run_manifest.json").read_text())
            ds = man["dataset"]
            if ds not in CLEAN or not man.get("drift_checks"):
                continue  # planners that ignore drift run no detector (2026-10-31)
            if ds not in seg_cache:
                seg_cache[ds] = injected_segments(ds, merge_gap)
            segs, n_rows = seg_cache[ds]
            thr = json.loads((ev_path.parent / "thresholds.json").read_text())
            window = int(thr.get("drift_window_size", 1200))
            spans = [(s, min(e + window, n_rows)) for s, e in segs]
            with open(ev_path, newline="") as f:
                events = list(csv.DictReader(f))
            steps = np.array([int(r["step"]) for r in events])
            in_span = np.zeros(len(steps), bool)
            for s, e in spans:
                in_span |= (steps >= s) & (steps <= e)
            row = {"run": f"{run_dir.name}/{ev_path.parent.name}", "dataset": ds,
                   "planner": man["planner"] + (f"_{man['pin_model']}" if man.get("pin_model") else ""),
                   "cycles": len(steps), "visible_cycles": int(in_span.sum()), "segments": len(segs)}
            for col in ("drift_detected", "drift_confirmed"):
                flag = np.array([r.get(col) == "True" for r in events])
                tp = int((flag & in_span).sum())
                delays, hit = [], 0
                for s, e in spans:
                    idx = np.flatnonzero(flag & (steps >= s) & (steps <= e))
                    if len(idx):
                        hit += 1
                        delays.append(int(steps[idx[0]] - s))
                row[f"{col}_flags"] = int(flag.sum())
                row[f"{col}_precision"] = tp / flag.sum() if flag.sum() else float("nan")
                row[f"{col}_cycle_recall"] = tp / in_span.sum() if in_span.sum() else float("nan")
                row[f"{col}_segment_recall"] = hit / len(segs) if segs else float("nan")
                row[f"{col}_median_delay"] = float(np.median(delays)) if delays else float("nan")
            out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs="+", required=True, help="Run folders, e.g. runs_seed1 runs_seed2")
    p.add_argument("--merge-gap", type=int, default=50)
    p.add_argument("--csv", default=None, help="Optional per-run CSV output path.")
    args = p.parse_args()
    rows = evaluate([Path(r) for r in args.runs], args.merge_gap)
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["dataset"], r["planner"])].append(r)
    print("| Dataset | Planner | runs | flags det / conf | precision det / conf | segment recall det / conf "
          "| cycle recall det / conf | median delay (rows) det / conf |")
    print("|---|---|---|---|---|---|---|---|")
    m = lambda v: float(np.nanmean(v)) if len(v) else float("nan")
    for (ds, pl), rs in sorted(agg.items()):
        if pl.startswith("naive"):
            continue
        g = lambda k: m([r[k] for r in rs])
        print(f"| {ds} | {pl} | {len(rs)} | {g('drift_detected_flags'):.0f} / {g('drift_confirmed_flags'):.0f} | "
              f"{g('drift_detected_precision'):.2f} / {g('drift_confirmed_precision'):.2f} | "
              f"{g('drift_detected_segment_recall'):.2f} / {g('drift_confirmed_segment_recall'):.2f} | "
              f"{g('drift_detected_cycle_recall'):.2f} / {g('drift_confirmed_cycle_recall'):.2f} | "
              f"{g('drift_detected_median_delay'):.0f} / {g('drift_confirmed_median_delay'):.0f} |")
    for ds in sorted({r["dataset"] for r in rows}):
        r0 = next(r for r in rows if r["dataset"] == ds)
        print(f"{ds}: {r0['segments']} injected segments, {r0['visible_cycles']} of {r0['cycles']} cycles "
              f"see an injected segment (base rate {r0['visible_cycles'] / r0['cycles']:.2f})")


if __name__ == "__main__":
    main()

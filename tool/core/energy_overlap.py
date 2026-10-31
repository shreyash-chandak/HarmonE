"""
core/energy_overlap.py — count package energy once across overlapping meters (2026-10-31).

Every reading in a run is a difference of the same package counter over a time
window. Prediction readings (no lock since per-prediction metering), MAPE cycles,
drift checks and background retrains overlap in time, so summing them counts the
energy of an overlap more than once.

Each meter has a window and a reading. Readings are placed on one timeline in
order of resolution (short windows first): prediction, then mape / drift, then
retrain. Within a stretch of time covered by several meters, the highest-resolution
reading already placed gives the energy of that stretch, and it is removed from the
lower-resolution reading. What remains of that reading is spread over the part of
its window that nothing covered yet, so a later, longer meter sees it there.

    net = gross - sum(overlap)

A window partly covering a reading takes the share of that reading proportional to
the time overlap (power within one window is taken as uniform). A reading never
loses more than its own value, so net >= the largest single meter kind's readings
that were placed first, and net <= gross.

Pure: no I/O except compute_from_run(), which reads a run's two CSVs.
"""

from __future__ import annotations

import bisect
import csv
from pathlib import Path

PRIORITY = {"prediction": 0, "mape": 1, "drift": 1, "retrain": 2}


def _pair(kind: str, covered_by: str) -> str:
    return f"{kind}_x_{covered_by}"


def compute_overlaps(predictions: list[tuple[float, float, float]],
                     windows: list[tuple[str, float, float, float]]) -> dict[str, float]:
    """Energy (in the readings' unit) counted more than once, per pair of meter kinds.

    predictions: (start, end, energy), one per reading.
    windows:     (kind, start, end, energy) for every other meter; kind in PRIORITY.
    """
    placed = sorted((float(s), float(e), float(en), "prediction") for s, e, en in predictions)
    starts = [p[0] for p in placed]
    segments = placed  # sorted by start; non-overlapping by construction after this point
    out: dict[str, float] = {}
    for kind, t0, t1, energy in sorted(windows, key=lambda w: (PRIORITY.get(w[0], 9), w[1])):
        if t1 <= t0 or energy is None:
            continue
        i = max(bisect.bisect_right(starts, t0) - 1, 0)
        removed: dict[str, float] = {}
        gaps: list[tuple[float, float]] = []
        cursor = t0
        while i < len(segments) and segments[i][0] < t1:
            s, e, en, source = segments[i]
            i += 1
            if e <= t0:
                continue
            if s > cursor:
                gaps.append((cursor, s))
            cursor = max(cursor, e)
            duration = e - s
            lo, hi = max(s, t0), min(e, t1)
            share = en if duration <= 0 else en * max(hi - lo, 0.0) / duration
            removed[source] = removed.get(source, 0.0) + share
        if cursor < t1:
            gaps.append((cursor, t1))
        total = sum(removed.values())
        scale = energy / total if total > energy and total > 0 else 1.0
        for source, value in removed.items():
            key = _pair(kind, source)
            out[key] = out.get(key, 0.0) + value * scale
        remainder = max(energy - total * scale, 0.0)
        gap_time = sum(b - a for a, b in gaps)
        for a, b in gaps:
            at = bisect.bisect_left(starts, a)
            starts.insert(at, a)
            segments.insert(at, (a, b, remainder * (b - a) / gap_time, kind))
    return out


def compute_from_run(predictions_csv: Path, meter_windows_csv: Path) -> dict[str, float]:
    """Overlaps (µJ) of a finished regression run; {} when windows were not recorded."""
    preds: list[tuple[float, float, float]] = []
    if not Path(predictions_csv).exists() or not Path(meter_windows_csv).exists():
        return {}
    with open(predictions_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("energy_valid") == "True" and r.get("start_monotonic") and r.get("end_monotonic"):
                preds.append((float(r["start_monotonic"]), float(r["end_monotonic"]), float(r["energy_uJ"])))
    with open(meter_windows_csv, newline="") as f:
        windows = [(r["kind"], float(r["start_monotonic"]), float(r["end_monotonic"]), float(r["energy_uJ"]))
                   for r in csv.DictReader(f) if r.get("energy_uJ")]
    return compute_overlaps(preds, windows)

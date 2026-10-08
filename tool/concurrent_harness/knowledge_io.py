"""concurrent_harness/knowledge_io.py — shared knowledge-dir I/O helpers.

Every concurrent-harness process (inference.py, mape/manage.py and its
threads) talks to the outside world only through a per-run "knowledge"
directory, mirroring the original HarmonE's knowledge/ blackboard — but scoped
per-run instead of one shared global directory, so concurrent runs never
collide with each other or with the single-threaded harness's own state.

Nothing here is imported by, or modifies, experiments/run_experiment.py or any
other single-threaded-path file — this whole package is purely additive.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable


def setup_logging(verbose: bool = False) -> None:
    """Identical to experiments/run_experiment.py's own logging.basicConfig
    call (same level rule, same format string) so every concurrent-harness
    process's log output looks exactly like the single-threaded harness's —
    same timestamp/level/logger-name prefix, same INFO-line conventions
    reused verbatim at each call site (see plan_thread.py/drift_thread.py's
    "MAPE[...]" line, inference.py's "stream step=..." line, and
    run_concurrent.py's final summary block)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# ── Fixed per-run knowledge-dir layout ─────────────────────────────────────────

MODEL_FILE = "model.txt"
PREDICTIONS_FILE = "predictions.csv"
IMAGE_HISTORY_FILE = "image_history.csv"          # CV only: step,image_path
MAPE_INFO_FILE = "mape_info.json"
MAPE_EVENTS_FILE = "mape_events.csv"
THRESHOLDS_FILE = "thresholds.json"
SCALER_FILE = "scaler.pkl"                          # regression only
REFERENCE_DIST_FILE = "reference_distribution.json"
STREAM_DONE_SENTINEL = "_stream_done"
SHUTDOWN_SENTINEL = "_shutdown"
# Written by run_concurrent.py if mape/manage.py dies before the stream ends,
# so a barrier-blocked inference.py can exit instead of waiting forever.
ABORT_SENTINEL = "_abort"
# Barrier pointer (audit D1/D2, 2026-09-28): number of prediction rows t1 has
# fully processed (monitored + planned + executed). inference.py may not run
# more than max_batches_ahead batches past it — see inference.py.
PROCESSED_FILE = "_processed"
# t2 -> t1 handoff (audit D4/D8): t2 only DETECTS drift and publishes its latest
# result here; t1 is the only thread that plans/executes.
DRIFT_STATE_FILE = "drift_state.json"
PLANNER_DECISIONS_FILE = "planner_decisions.csv"
# inference.py logs every failed weight reload here (audit D6); the count
# goes into run_manifest.json as "reload_failures".
RELOAD_FAILURES_FILE = "reload_failures.csv"
MODELS_SUBDIR = "models"                            # per-run copies of live weights

PREDICTION_FIELDS_REGRESSION = [
    "step", "y_true", "y_pred", "active_model", "energy_uJ", "energy_valid",
    "inference_time_s",
]
PREDICTION_FIELDS_CV = [
    "step", "proxy_acc", "active_model", "planner", "energy_uJ", "energy_valid",
]
# Same schema experiments/run_experiment.py's own mape_events.append({...})
# writes, so scripts/plot_results.py's chart_ema_timeseries() (which reads
# step/ema_score from this file) works unmodified against a concurrent run.
MAPE_EVENT_FIELDS = [
    "step", "violation", "drift_detected", "drift_confirmed", "kl_div",
    "decision_action", "decision_model", "decision_reason",
    "model_before", "model_after",
    "r2", "accuracy", "ema_score", "avg_energy_uJ", "energy_threshold",
]

# Serializes all core.energy.EnergyMeter usage *within mape/manage.py's process*
# (plan_thread / drift_thread run as threads in that one process).
# core/energy.py's own re-entrancy guard (_ACTIVE) is a plain unlocked bool —
# safe against nothing across threads. inference.py is a *separate* process
# with its own single loop, so it never needs this lock. This is
# process-local only — see energy_lock() below for the cross-process half of
# the story (which covers inference.py <-> mape/manage.py, the pair this
# lock cannot).
ENERGY_LOCK = threading.Lock()

_ENERGY_LOCK_FILE = "_energy.lock"


@contextlib.contextmanager
def energy_lock(knowledge_dir: Path, blocking: bool = True):
    """Cross-process advisory lock guaranteeing at most one EnergyMeter is
    open, system-wide, at any instant.

    RAPL/NVML/polling energy counters are global to the machine, not
    per-process — inference.py and mape/manage.py (via mape_cycle.py) each
    used to open their own EnergyMeter completely independently, so their
    measurement windows could (and did) overlap in real wall-clock time,
    double-counting whatever the hardware actually drew during the overlap.
    A plain threading.Lock (see ENERGY_LOCK above) only serializes threads
    *within* one process — it's invisible to the other process entirely.
    An flock on a shared per-run file works across the process boundary,
    which is the actual fix: once every EnergyMeter anywhere in this run is
    guaranteed never to overlap another, summing every measured delta into
    one running total (see add_energy_used()) is correct instead of an
    overestimate — there's nothing left to double-count.

    blocking=True: wait for the lock. As of 2026-09-28 (audit C3) every
    caller uses this — mape_cycle.py, drift_thread.py AND inference.py. The
    previous non-blocking inference-side usage skipped measuring any step
    that collided with a MAPE cycle (recorded as energy_uJ=0,
    energy_valid=False; 4-18% of steps), biasing per-model energy. Waiting
    is now affordable because the inference/manager barrier (see
    inference.py) already bounds how far the stream can run ahead, i.e.
    inference was going to pause for a long retrain anyway — exactly like
    the single-threaded harness, which measures every step.

    blocking=False is still supported (yields False if the lock is held).

    Yields True if the lock was acquired, False otherwise (only possible
    when blocking=False).
    """
    lock_path = Path(knowledge_dir) / _ENERGY_LOCK_FILE
    lock_path.touch(exist_ok=True)
    fd = open(lock_path, "r+")
    acquired = False
    try:
        if blocking:
            fcntl.flock(fd, fcntl.LOCK_EX)
            acquired = True
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                acquired = False
        try:
            yield acquired
        finally:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        fd.close()


def add_mape_energy(mape_store: "MapeInfoStore", energy_uJ: float | None) -> None:
    """Add a MAPE-side EnergyMeter reading to event_counters.mape_k_energy_uJ.

    2026-09-28 (audit C3): energy is now split exactly like the
    single-threaded harness — inference energy lives in predictions.csv
    (every step measured; summed into task_metrics.total_inference_energy_*
    at manifest time) and MAPE-K overhead accumulates here. inference.py no
    longer touches mape_info.json at all. Because every EnergyMeter in the
    run is opened under the blocking cross-process energy_lock(), the two
    totals never overlap in time, so run_concurrent.py can also report their
    sum (event_counters.energy_used_uJ) without double counting.
    """
    def _add(d: dict) -> None:
        ec = d["event_counters"]
        if energy_uJ is None:  # unavailable reading (audit C6) — count, don't add 0
            ec["mape_energy_invalid_cycles"] = ec.get("mape_energy_invalid_cycles", 0) + 1
        else:
            ec["mape_k_energy_uJ"] = ec.get("mape_k_energy_uJ", 0.0) + energy_uJ
    mape_store.update(_add)


def build_initial_mape_info(model_names) -> dict:
    """experiments.run_experiment._initial_mape_info() plus two concurrent-only
    fields:
      - last_adaptation: {model: step of its last successful replace/retrain},
        written by t1 (mape/execute.py), read by t2 to re-base that model's
        drift reference and start its cooldown (mape/drift_ref.py).
      - drift_checks: number of t2 drift detections, reported separately from
        "cycles" (t1 planning cycles, one per monitor_interval batch — the
        same meaning as single-threaded's mape_cycles).
    """
    from experiments.run_experiment import _initial_mape_info
    info = _initial_mape_info(model_names)
    info["last_adaptation"] = {}
    info["drift_checks"] = 0
    return info


def knowledge_paths(knowledge_dir: str | Path) -> dict[str, Path]:
    d = Path(knowledge_dir)
    return {
        "root": d,
        "model_file": d / MODEL_FILE,
        "predictions_file": d / PREDICTIONS_FILE,
        "image_history_file": d / IMAGE_HISTORY_FILE,
        "mape_info_file": d / MAPE_INFO_FILE,
        "mape_events_file": d / MAPE_EVENTS_FILE,
        "thresholds_file": d / THRESHOLDS_FILE,
        "scaler_file": d / SCALER_FILE,
        "reference_dist_file": d / REFERENCE_DIST_FILE,
        "stream_done": d / STREAM_DONE_SENTINEL,
        "shutdown": d / SHUTDOWN_SENTINEL,
        "abort": d / ABORT_SENTINEL,
        "processed": d / PROCESSED_FILE,
        "drift_state": d / DRIFT_STATE_FILE,
        "planner_decisions_file": d / PLANNER_DECISIONS_FILE,
        "models_dir": d / MODELS_SUBDIR,
    }


# ── barrier pointer / drift-state handoff (2026-09-28) ─────────────────────────

def _atomic_write_text(path: Path, text: str) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def write_processed(path: Path, n_rows: int) -> None:
    _atomic_write_text(path, str(int(n_rows)))


def read_processed(path: Path) -> int:
    try:
        with open(path) as f:
            return int(f.read().strip() or 0)
    except (FileNotFoundError, ValueError):
        return 0


def write_drift_state(path: Path, state: dict) -> None:
    _atomic_write_text(path, json.dumps(state))


def read_drift_state(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def append_dict_row(csv_path: Path, row: dict) -> None:
    """Append a row whose header is the first row's keys (schema fixed per
    run) — used for planner_decisions.csv, whose per-model columns depend on
    the dataset's model list, same as run_experiment.py's own writer."""
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ── model.txt ───────────────────────────────────────────────────────────────

def read_current_model(model_file: Path, default: str | None = None) -> str | None:
    try:
        with open(model_file, "r") as f:
            name = f.read().strip()
            return name or default
    except FileNotFoundError:
        return default


def write_current_model(model_file: Path, name: str) -> None:
    tmp = str(model_file) + ".tmp"
    with open(tmp, "w") as f:
        f.write(name)
    os.replace(tmp, model_file)


# ── predictions.csv / image_history.csv ────────────────────────────────────────

def append_row(csv_path: Path, row: dict, fieldnames: list[str]) -> None:
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


class RowBuffer:
    """Incrementally stages new CSV rows so a caller can drain them in fixed
    chunks without re-reading the file's live tail on every drain.

    Fix (2026-09-06): plan_thread.py/drift_thread.py both used to track only
    a row *count* and, on each boundary crossing, re-fetch data via
    read_tail() — the file's CURRENT state at call time, not a snapshot tied
    to that specific boundary. If a thread ever fell behind real inference
    progress for a stretch (GIL contention with mape_cycle's lock, or simply
    a fast model letting inference race ahead between poll ticks), many
    boundary crossings could accumulate at once; the inner drain loop would
    then process all of them in one synchronous burst with zero new rows
    arriving in between, so every iteration's read_tail() call returned the
    IDENTICAL data. Confirmed directly: 19 consecutive "drift detected" /
    VMR-replace cycles, all evaluating one single window, each logging a
    different (fictitious) current_step. A RowBuffer per source file
    guarantees each drained chunk is the actual next slice of real rows,
    however many chunks a burst contains.
    """

    def __init__(self, path: Path):
        self._path = path
        self._offset = 0                      # bytes consumed so far
        self._fields: list[str] | None = None  # header, parsed on first read
        self._rows_seen = 0
        self._staging: list[dict] = []

    def poll(self) -> None:
        """Pull any rows written since the last poll() into staging.

        Incremental (audit D5, 2026-10-04): reads only the bytes appended
        since the previous poll instead of re-parsing the whole file every
        tick (O(N^2) over a run; on uci that is 126k rows re-read every
        20 ms per thread, and the CPU time was charged to inference energy,
        audit C2). Only bytes up to the last newline are consumed; a trailing
        partial row (inference.py mid-append) stays unread until its newline
        lands, so torn rows cannot be parsed by construction.
        """
        try:
            with open(self._path, "rb") as f:
                f.seek(self._offset)
                data = f.read()
        except FileNotFoundError:
            return
        end = data.rfind(b"\n")
        if end < 0:
            return
        complete = data[: end + 1]
        self._offset += len(complete)
        lines = complete.decode("utf-8").splitlines()
        for values in csv.reader(lines):
            if self._fields is None:
                self._fields = values
                continue
            # Counted even if malformed, so committed_count stays aligned
            # with inference.py's step numbers (it drives the barrier).
            self._rows_seen += 1
            if len(values) != len(self._fields):
                logging.getLogger(__name__).warning(
                    "RowBuffer %s: skipping malformed row (%d fields, expected %d)",
                    self._path.name, len(values), len(self._fields))
                continue
            self._staging.append(dict(zip(self._fields, values)))

    def drain_chunk(self, n: int) -> list[dict] | None:
        """Pop exactly n staged rows (oldest first), or None if fewer than
        n are currently staged."""
        if len(self._staging) < n:
            return None
        chunk, self._staging = self._staging[:n], self._staging[n:]
        return chunk

    @property
    def committed_count(self) -> int:
        """Rows actually drained via drain_chunk() so far (i.e. total rows
        seen minus whatever's still sitting in staging)."""
        return self._rows_seen - len(self._staging)


# ── mape_info.json (event counters / EMA state) ────────────────────────────────

class MapeInfoStore:
    """Read-modify-write wrapper around knowledge_dir/mape_info.json, safe
    across both threads (within mape/manage.py) and processes.

    As of 2026-09-06 (single shared energy_used_uJ total), inference.py also
    holds its own MapeInfoStore instance on the same file — previously only
    manage.py's process ever touched it. A plain threading.Lock (still kept
    below, for the same-process t1/t2 fast path) is invisible to a different
    process, so two processes racing to initialize or update this file could
    (and did, verified directly: `FileNotFoundError` on the shared `.tmp`
    path, and `JSONDecodeError: Extra data` from an interleaved write)
    corrupt it. An flock on a dedicated lock file closes that gap — same
    pattern as energy_lock() above, but this one guards mape_info.json's
    read-modify-write cycle specifically, not EnergyMeter usage.
    """

    def __init__(self, path: Path, initial: dict):
        self._path = path
        self._lock = threading.Lock()
        self._lock_path = Path(str(path) + ".flock")
        with self._cross_process_lock():
            if not path.exists():
                self._write(initial)

    @contextlib.contextmanager
    def _cross_process_lock(self):
        self._lock_path.touch(exist_ok=True)
        fd = open(self._lock_path, "r+")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            fd.close()

    def _write(self, data: dict) -> None:
        tmp = str(self._path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self._path)

    def load(self) -> dict:
        with self._lock, self._cross_process_lock():
            with open(self._path, "r") as f:
                return json.load(f)

    def update(self, fn: Callable[[dict], None]) -> dict:
        """Load, call fn(data) to mutate in place, persist, return the result."""
        with self._lock, self._cross_process_lock():
            with open(self._path, "r") as f:
                data = json.load(f)
            fn(data)
            self._write(data)
            return data


# ── sentinels ───────────────────────────────────────────────────────────────

def touch(path: Path) -> None:
    path.write_text(str(time.time()))


def exists(path: Path) -> bool:
    return path.exists()


def interruptible_sleep(total_s: float, shutdown_path: Path, tick: float = 1.0) -> bool:
    """Sleep up to total_s seconds, waking early (and returning False) if
    shutdown_path appears. Returns True if the full sleep completed.

    The original HarmonE's threads sleep in one uninterruptible block (e.g.
    time.sleep(40)) because nothing ever needed to stop them early — a human
    just killed the process. This harness needs a bounded run (so it can write
    a final manifest), so threads here poll for shutdown on short ticks instead.
    """
    elapsed = 0.0
    while elapsed < total_s:
        if shutdown_path.exists():
            return False
        step = min(tick, total_s - elapsed)
        time.sleep(step)
        elapsed += step
    return not shutdown_path.exists()


# ── per-model reload flags (retrain -> inference cross-process signal) ────────

def _reload_flag_path(knowledge_dir: Path, model_name: str) -> Path:
    return knowledge_dir / f"_reload_{model_name}"


def write_reload_flag(knowledge_dir: Path, model_name: str) -> None:
    touch(_reload_flag_path(knowledge_dir, model_name))


def consume_reload_flag(knowledge_dir: Path, model_name: str) -> bool:
    p = _reload_flag_path(knowledge_dir, model_name)
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass
        return True
    return False


# ── model spec localization ────────────────────────────────────────────────

def local_weights_path(knowledge_dir: Path, model_name: str, weights_path: str) -> Path:
    """Per-run copy of a model's weights, under knowledge_dir/models/.

    Concurrent runs must never write back to the shared weights_path files
    configs/datasets/*.json point at (those are read by the single-threaded
    harness and other concurrent runs too) — every retrain/replace in this
    package writes here instead.
    """
    ext = Path(weights_path).suffix
    return knowledge_dir / MODELS_SUBDIR / f"{model_name}{ext}"


def localize_model_spec(spec: dict, knowledge_dir: Path, model_name: str) -> dict:
    """Return a copy of a model spec dict with weights_path pointed at the
    per-run local copy instead of the shared production path."""
    local = dict(spec)
    local["weights_path"] = str(local_weights_path(knowledge_dir, model_name, spec["weights_path"]))
    return local


def localize_dataset_config(dataset_config: dict, knowledge_dir: Path) -> dict:
    """Copy of dataset_config with every model's weights_path localized."""
    cfg = dict(dataset_config)
    cfg["models"] = {
        name: localize_model_spec(spec, knowledge_dir, name)
        for name, spec in dataset_config.get("models", {}).items()
    }
    return cfg

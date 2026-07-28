"""
experiments/run_grid.py — Full grid driver.

Loads a YAML grid config and runs all (dataset × planner × seed) combinations.
Skips already-completed runs when --resume is set.

Usage (CLI):
    python experiments/run_grid.py configs/experiments/baseline.yaml

Or from Python:
    from experiments.run_grid import run_grid
    manifest = run_grid("configs/experiments/baseline.yaml")
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

from experiments.run_experiment import run_experiment, _build_run_id

logger = logging.getLogger(__name__)


# ── Config loading ────────────────────────────────────────────────────────────

def load_grid_config(config_path: str) -> dict:
    """Parse a grid config file (YAML or JSON).

    JSON is supported as a fallback so tests can run without PyYAML installed.
    YAML is the recommended format for human-authored configs (.yaml / .yml).
    JSON configs can use the same extension (.json) or .yaml — the parser is
    chosen by file extension, defaulting to JSON when yaml is unavailable.

    Required keys:
        datasets:         list of dataset names
        planners:         list of planner names
        seeds:            list of integer seeds

    Optional keys:
        cooldown_minutes: seconds to sleep between runs (default 0)
        monitor_interval: MAPE cycle length (default 50)
        max_steps:        cap per run (default None)
        runs_dir:         where to create run dirs (default "runs")
        resume:           skip runs where run_manifest.json already exists (default True)
        extra_thresholds: dict overlaid on the dataset config thresholds
    """
    import json as _json

    ext = Path(config_path).suffix.lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
        except ImportError:
            raise ImportError(
                "PyYAML is required for .yaml grid configs. "
                "Install it with: pip install pyyaml  "
                "— or use a .json grid config instead."
            )
    else:
        with open(config_path, "r") as f:
            cfg = _json.load(f)

    # Validate required keys
    for key in ("datasets", "planners", "seeds"):
        if key not in cfg:
            raise ValueError(f"Grid config missing required key: '{key}'")

    return cfg


def _enumerate_runs(cfg: dict) -> list[dict]:
    """Return all (dataset, planner, seed) combinations from the config."""
    combos = []
    for dataset in cfg["datasets"]:
        for planner in cfg["planners"]:
            for seed in cfg["seeds"]:
                combos.append({"dataset": dataset, "planner": planner, "seed": int(seed)})
    return combos


# ── Grid runner ───────────────────────────────────────────────────────────────

def run_grid(
    config_path: str,
    *,
    runs_dir: str | None = None,
    configs_dir: str | None = None,
    resume: bool | None = None,
    dry_run: bool = False,
) -> dict:
    """Execute the full experiment grid defined in config_path.

    Args:
        config_path:  Path to the YAML grid config.
        runs_dir:     Parent directory for run dirs; overrides config value.
        configs_dir:  Path to configs/; defaults to <tool_dir>/configs.
        resume:       If True, skip runs where run_manifest.json exists.
                      Overrides the config's ``resume`` key when provided.
        dry_run:      Print the plan without running anything.

    Returns:
        Grid manifest dict (also written to <runs_dir>/grid_manifest.json).
    """
    cfg = load_grid_config(config_path)

    _runs_dir = runs_dir or cfg.get("runs_dir", "runs")
    _resume = resume if resume is not None else cfg.get("resume", True)
    _monitor_interval = cfg.get("monitor_interval", 50)
    _cooldown_s = cfg.get("cooldown_minutes", 0) * 60
    _max_steps = cfg.get("max_steps", None)
    _extra_thresholds = cfg.get("extra_thresholds", None)

    runs_path = Path(_runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)

    combos = _enumerate_runs(cfg)
    total = len(combos)

    logger.info(
        "Grid: %d runs (%d datasets × %d planners × %d seeds)%s",
        total,
        len(cfg["datasets"]),
        len(cfg["planners"]),
        len(cfg["seeds"]),
        " [DRY RUN]" if dry_run else "",
    )

    results: list[dict] = []
    skipped = 0
    failed = 0
    start_ts = datetime.now(timezone.utc).isoformat()

    for i, combo in enumerate(combos):
        dataset, planner, seed = combo["dataset"], combo["planner"], combo["seed"]
        run_id = _build_run_id(dataset, planner, seed)
        run_dir = str(runs_path / run_id)

        label = f"[{i+1}/{total}] {dataset} × {planner} × seed={seed}"

        # Check for existing run (resume mode)
        manifest_check = Path(run_dir) / "run_manifest.json"
        if _resume and manifest_check.exists():
            logger.info("%s → SKIP (already completed)", label)
            skipped += 1
            results.append({
                "run_id": run_id,
                "dataset": dataset,
                "planner": planner,
                "seed": seed,
                "status": "skipped",
            })
            continue

        if dry_run:
            logger.info("%s → DRY RUN (would write to %s)", label, run_dir)
            results.append({
                "run_id": run_id,
                "dataset": dataset,
                "planner": planner,
                "seed": seed,
                "status": "dry_run",
            })
            continue

        logger.info("%s → RUNNING …", label)
        try:
            manifest = run_experiment(
                dataset_name=dataset,
                planner_name=planner,
                seed=seed,
                run_dir=run_dir,
                configs_dir=configs_dir,
                monitor_interval=_monitor_interval,
                max_steps=_max_steps,
                extra_thresholds=_extra_thresholds,
            )
            results.append({
                "run_id": run_id,
                "dataset": dataset,
                "planner": planner,
                "seed": seed,
                "status": "ok",
                "total_steps": manifest.get("total_steps"),
                "model_switches": manifest.get("event_counters", {}).get("model_switches"),
                "elapsed_s": manifest.get("elapsed_s"),
            })
            logger.info(
                "%s → DONE  steps=%d switches=%d  (%.1fs)",
                label,
                manifest.get("total_steps", 0),
                manifest.get("event_counters", {}).get("model_switches", 0),
                manifest.get("elapsed_s", 0),
            )
        except Exception as exc:
            logger.error("%s → FAILED: %s", label, exc, exc_info=True)
            failed += 1
            results.append({
                "run_id": run_id,
                "dataset": dataset,
                "planner": planner,
                "seed": seed,
                "status": "failed",
                "error": str(exc),
            })

        if _cooldown_s > 0 and i < total - 1:
            logger.info("Cooling down for %.0f s …", _cooldown_s)
            time.sleep(_cooldown_s)

    end_ts = datetime.now(timezone.utc).isoformat()
    grid_manifest = {
        "config": str(Path(config_path).resolve()),
        "started_at": start_ts,
        "finished_at": end_ts,
        "total": total,
        "completed": sum(1 for r in results if r["status"] == "ok"),
        "skipped": skipped,
        "failed": failed,
        "dry_run": dry_run,
        "runs": results,
    }

    grid_manifest_path = runs_path / "grid_manifest.json"
    with open(grid_manifest_path, "w") as f:
        import json
        json.dump(grid_manifest, f, indent=2)

    logger.info(
        "Grid done: %d ok, %d skipped, %d failed → %s",
        grid_manifest["completed"], skipped, failed, grid_manifest_path,
    )
    return grid_manifest


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a HarmonE experiment grid.")
    parser.add_argument("config", help="Path to YAML grid config")
    parser.add_argument("--runs-dir", default=None, help="Override runs_dir from config")
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument("--no-resume", action="store_true", help="Re-run completed runs")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import json as _json
    manifest = run_grid(
        config_path=args.config,
        runs_dir=args.runs_dir,
        configs_dir=args.configs_dir,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )
    print(_json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

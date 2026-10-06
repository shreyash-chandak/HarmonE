"""
experiments/calibrate_from_naive_runs.py — derive the planner thresholds of a
dataset from its naive single-model runs (audit A4/A6, A1 follow-up, N11).

Planner-independent by construction: only the pinned naive runs (one per model,
no adaptation) are read, never a planner's own replay (audit A6: E_ref used to
be calibrated from harmone_original's run).

Energy (normalisation bounds and reference):
    E_m   = mean per-step inference energy (µJ) of the cheapest model
    E_M   = mean per-step inference energy (µJ) of the most expensive model
    E_ref = energy_budget_rho (normalised), i.e. the point rho of the way from
            the cheapest to the most expensive model
  so normalised energy is ~0 on the cheapest model and ~1 on the most
  expensive one. (Previously E_m/E_M came from per-step logs that were mostly
  RAPL zeros: on uci E_M was 17,500 µJ while lstm uses ~58,900 µJ/step, and
  ridge already sat above E_ref, so energy violations fired on every model.)

Accuracy gates (same budget rule on the accuracy axis):
    min_accuracy = acc_cheap + accuracy_budget_rho * (acc_best - acc_cheap)
    min_score    = beta * min_accuracy + (1 - beta) * (1 - E_ref)
  where acc_* are the models' mean per-batch accuracy under the configured
  accuracy_signal (regression: "window_r2", recomputed here from each naive
  run's predictions.csv; CV: mean proxy confidence). min_score is the
  HarmonE score of a model sitting exactly at the target operating point.

These values depend on the model pool and on the energy metering, so re-run
this after either changes (pool positioning, batch metering) — the run
manifests record which metering produced them.

Usage (from tool/):
    python3 experiments/calibrate_from_naive_runs.py --dataset pems_driftinduced \\
        --runs-dir concurrent_harness/runs [--write]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_TOOL_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_WINDOW = 1200
_DEFAULT_INTERVAL = 50


def _naive_run_dir(runs_dir: Path, dataset: str, model: str) -> Path | None:
    for name in (f"{dataset}_naive_{model}_conc", f"{dataset}_naive_{model}"):
        if (runs_dir / name / "run_manifest.json").exists():
            return runs_dir / name
    return None


def _window_r2_mean(pred_csv: Path, interval: int, window: int) -> float:
    """Mean per-batch accuracy under accuracy_signal="window_r2" (see
    experiments/run_experiment._monitor_batch)."""
    with open(pred_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    yt = np.array([float(r["y_true"]) for r in rows])
    yp = np.array([float(r["y_pred"]) for r in rows])
    accs = []
    for i in range(len(yt) // interval):
        end = (i + 1) * interval
        mse = float(np.mean((yt[end - interval:end] - yp[end - interval:end]) ** 2))
        ref_var = float(np.var(yt[max(0, end - window):end]))
        accs.append(max(0.0, min(1.0, 1.0 - mse / ref_var)) if ref_var > 0 else 0.0)
    return float(np.mean(accs)) if accs else 0.0


def calibrate(dataset: str, runs_dir: Path, configs_dir: Path) -> dict:
    cfg_path = configs_dir / f"{dataset}.json"
    cfg = json.loads(cfg_path.read_text())
    is_cv = cfg.get("domain", "regression") == "cv"
    interval = int(cfg.get("monitor_interval", _DEFAULT_INTERVAL))
    window = int(cfg.get("drift_window_size", _DEFAULT_WINDOW))
    beta = float(cfg.get("beta", 0.95))
    rho_e = float(cfg.get("energy_budget_rho", 0.5))
    rho_a = float(cfg.get("accuracy_budget_rho", 0.5))

    per_model: dict[str, dict] = {}
    for model in cfg.get("models", {}):
        run = _naive_run_dir(runs_dir, dataset, model)
        if run is None:
            raise SystemExit(f"missing naive run for {dataset}/{model} under {runs_dir}")
        man = json.loads((run / "run_manifest.json").read_text())
        e = man["energy_by_model"][model]
        e_step = e.get("energy_per_inference_mJ")
        if e_step is None:
            e_step = e["energy_mJ"] / max(e.get("n_valid_steps", e["n_steps"]), 1)
        if is_cv:
            acc = float(man["task_metrics"]["mean_proxy_acc"])
        else:
            acc = _window_r2_mean(run / "predictions.csv", interval, window)
        per_model[model] = {
            "energy_uJ_per_step": float(e_step) * 1000.0,
            "accuracy": acc,
            "run": run.name,
        }

    cheap = min(per_model, key=lambda m: per_model[m]["energy_uJ_per_step"])
    costly = max(per_model, key=lambda m: per_model[m]["energy_uJ_per_step"])
    best = max(per_model, key=lambda m: per_model[m]["accuracy"])
    e_m = per_model[cheap]["energy_uJ_per_step"]
    e_M = per_model[costly]["energy_uJ_per_step"]
    acc_cheap = per_model[cheap]["accuracy"]
    acc_best = per_model[best]["accuracy"]
    min_acc = acc_cheap + rho_a * (acc_best - acc_cheap)
    e_ref = rho_e
    min_score = beta * min_acc + (1.0 - beta) * (1.0 - e_ref)
    return {
        "dataset": dataset,
        "per_model": per_model,
        "values": {
            "E_m": round(e_m, 1),
            "E_M": round(e_M, 1),
            "E_ref": round(e_ref, 4),
            "energy_reference": round(e_ref, 4),
            "min_accuracy": round(min_acc, 4),
            "min_score": round(min_score, 4),
        },
        "source": f"calibrate_from_naive_runs:{runs_dir.name}",
    }


def write_config(dataset: str, configs_dir: Path, result: dict) -> None:
    """Update the threshold keys in place, editing only their lines (keeps the
    file's formatting); keys that are absent are added after "min_score"."""
    import re
    cfg_path = configs_dir / f"{dataset}.json"
    text = cfg_path.read_text()
    updates = dict(result["values"])
    updates["energy_bounds_source"] = result["source"]
    updates["energy_reference_source"] = result["source"] + " (E_ref = energy_budget_rho)"
    updates["accuracy_gate_source"] = result["source"] + " (min_accuracy = accuracy_budget_rho rule)"
    missing = []
    for key, val in updates.items():
        pat = re.compile(rf'^(\s*)"{re.escape(key)}":\s*(?:"(?:[^"\\]|\\.)*"|[^,\n]+)(,?)$', re.M)
        if pat.search(text):
            text = pat.sub(lambda m: f'{m.group(1)}"{key}": {json.dumps(val)}{m.group(2)}', text, count=1)
        else:
            missing.append((key, val))
    if missing:
        anchor = re.compile(r'^(\s*)"min_score":[^\n]*\n', re.M)
        m = anchor.search(text)
        if m is None:
            raise SystemExit(f'{cfg_path}: no "min_score" line to insert {missing} after')
        extra = "".join(f'{m.group(1)}"{k}": {json.dumps(v)},\n' for k, v in missing)
        text = text[:m.end()] + extra + text[m.end():]
    json.loads(text)  # still valid JSON
    cfg_path.write_text(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True, action="append",
                   help="Dataset config name; repeat for several.")
    p.add_argument("--runs-dir", default=str(_TOOL_DIR / "concurrent_harness" / "runs"))
    p.add_argument("--configs-dir", default=str(_TOOL_DIR / "configs" / "datasets"))
    p.add_argument("--write", action="store_true", help="Write the values into the configs.")
    args = p.parse_args()
    for ds in args.dataset:
        res = calibrate(ds, Path(args.runs_dir), Path(args.configs_dir))
        print(json.dumps(res, indent=2))
        if args.write:
            write_config(ds, Path(args.configs_dir), res)
            print(f"wrote {ds}.json", file=sys.stderr)


if __name__ == "__main__":
    main()

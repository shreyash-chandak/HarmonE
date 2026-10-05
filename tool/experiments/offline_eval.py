"""experiments/offline_eval.py — Offline GT-accuracy evaluation for CV runs (dispatcher).

Reads run_experiment.py output (runs/{run_id}/) and replays the recorded
model-switch decisions against ground-truth labels to compute true accuracy
metrics. As of the 2026-08-30 audit this is a thin dispatcher — the actual
per-task logic lives in one file each, cleanly split by task:

  experiments/offline_eval_detection.py       — mAP@0.5 / mAP@0.75 / mAP@0.90
  experiments/offline_eval_classification.py  — accuracy/precision/recall/F1 + TP/FP/FN/TN
  experiments/offline_eval_segmentation.py    — mIoU
  experiments/offline_eval_common.py          — shared config/manifest/predictions loading

Design (Option 2 — replay): the run already logged which model was active at
each step (predictions.csv: step, active_model, planner, proxy_acc/confidence,
energy) and, for most runs, the raw per-step prediction (predictions/*.{npz,
txt,png}). We re-read the dataset manifest for GT and compare — no
predictions are re-generated unless that directory is missing (legacy runs),
and model loading (when needed) is cached so each model loads at most once.

Limitation: uses the original model weights from the dataset config, not VMR
snapshots. Post-retrain weight changes are not reproduced exactly — this
gives the accuracy of the pre-retrain model at retrain-triggered steps.

Results are written to two places:
  1. {run_dir}/offline_eval.json — full detail (overall + per-interval).
  2. {run_dir}/run_manifest.json's "offline_task_metrics" key — merged in
     alongside the live run's existing "task_metrics" (proxy-based), and a
     one-line "accuracy summary | ..." is printed in the same format
     run_experiment.py already uses for regression's R²/RMSE/MAE, so a true
     CV score is surfaced the same way a regression score already is.

Usage (from inside tool/):
    python3 experiments/offline_eval.py \\
        --run-dir runs/bdd100k_harmone_original_s1 \\
        --dataset bdd100k \\
        [--interval 1000] \\
        [--output runs/bdd100k_harmone_original_s1/offline_eval.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Invoked directly as `python3 experiments/offline_eval.py` (see this file's
# __main__ block / run_offline_eval.sh), Python puts experiments/ itself on
# sys.path[0], not tool/ — so `experiments` isn't importable as a package yet
# without this. Must run before the `from experiments....` imports below.
_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

from experiments.offline_eval_common import build_stream_index, load_config, load_predictions
from experiments.offline_eval_classification import evaluate_classification
from experiments.offline_eval_detection import evaluate_detection
from experiments.offline_eval_segmentation import evaluate_segmentation


def _print_summary(task: str, result: dict) -> None:
    overall = result.get("overall")
    if overall is None:
        print("[EVAL] no valid intervals — check that label files exist")
        return
    if task == "detection":
        print(
            "accuracy summary  | mAP50=%.4f  mAP75=%.4f  mAP90=%.4f  n=%d"
            % (overall["map50"], overall["map75"], overall["map90"], overall["n_images"])
        )
    elif task == "classification":
        print(
            "accuracy summary  | Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  "
            "TP=%d  TN=%d  FP=%d  FN=%d  n=%d"
            % (overall["accuracy"], overall["precision_macro"], overall["recall_macro"],
               overall["f1_macro"], overall["tp"], overall["tn"], overall["fp"], overall["fn"],
               overall["n_samples"])
        )
    else:  # segmentation
        print("accuracy summary  | mIoU=%.4f  n=%d" % (overall["miou"], overall["n_images"]))


def _merge_into_run_manifest(run_path: Path, task: str, result: dict) -> None:
    """Add offline_task_metrics to run_manifest.json, alongside the live
    run's proxy-based task_metrics — same artifact, same convention regression
    uses for its own accuracy summary."""
    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.exists():
        print(f"[EVAL] no run_manifest.json at {manifest_path} — skipping manifest merge")
        return
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["offline_task_metrics"] = {"task": task, **result}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[EVAL] merged offline_task_metrics into {manifest_path}")
    except Exception as exc:
        print(f"[EVAL] could not merge into {manifest_path}: {exc}")


def evaluate_run(
    run_dir: str,
    dataset: str,
    interval_size: int = 1000,
    output_path: str | None = None,
) -> dict:
    """Evaluate a CV run against ground-truth labels.

    Args:
        run_dir:       Path to runs/{run_id}/ produced by run_experiment.py.
        dataset:       Dataset name matching configs/datasets/<name>.json.
        interval_size: Steps per reporting interval (default 1000).
        output_path:   Where to write JSON results. Defaults to {run_dir}/offline_eval.json.

    Returns:
        Results dict (same content written to output_path).
    """
    run_path = Path(run_dir)
    config = load_config(dataset)
    task = config.get("task", "detection")

    print(f"[EVAL] run_dir={run_path}  dataset={dataset}  task={task}")
    pred_df = load_predictions(run_path)
    stream_images, stream_labels, stream_inline_labels = build_stream_index(config)
    print(
        f"[EVAL] predictions={len(pred_df)}  stream_images={len(stream_images)}"
        f"  interval_size={interval_size}"
    )

    if task == "detection":
        result = evaluate_detection(config, pred_df, stream_images, stream_labels, interval_size, run_path=run_path)
    elif task == "classification":
        result = evaluate_classification(
            config, pred_df, stream_images, stream_labels, stream_inline_labels, interval_size,
            run_path=run_path,
        )
    elif task == "segmentation":
        result = evaluate_segmentation(config, pred_df, stream_images, stream_labels, interval_size, run_path=run_path)
    else:
        raise ValueError(
            f"Unsupported task '{task}'. Must be detection, segmentation, or classification."
        )

    output = {
        "dataset": dataset,
        "task": task,
        "run_dir": str(run_path.resolve()),
        "interval_size": interval_size,
        **result,
    }

    if output_path is None:
        output_path = str(run_path / "offline_eval.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[EVAL] results written to {output_path}")

    _print_summary(task, result)
    _merge_into_run_manifest(run_path, task, result)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline GT-accuracy evaluation for CV runs")
    parser.add_argument("--run-dir", required=True, help="Path to runs/{run_id}/ directory")
    parser.add_argument("--dataset", required=True, help="Dataset key (bdd100k | acdc | iwildcam | imagenet | imagenet_c)")
    parser.add_argument(
        "--interval", type=int, default=1000,
        help="Steps per reporting interval (default: 1000)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: {run-dir}/offline_eval.json)",
    )
    args = parser.parse_args()

    results = evaluate_run(
        run_dir=args.run_dir,
        dataset=args.dataset,
        interval_size=args.interval,
        output_path=args.output,
    )
    print(json.dumps(results, indent=2))

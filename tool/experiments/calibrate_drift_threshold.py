"""
experiments/calibrate_drift_threshold.py — Per-dataset drift-threshold
calibration, closing DP4 (context/DECISIONS_PENDING.md).

Implements the calibration half of Augur's methodology (Lewis, Echeverría,
Pons, Chrabaszcz — "Augur: A Step Towards Realistic Drift Detection in
Production ML Systems", SE4RAI'22): rather than picking tau_drift from a
single global default, compute a NULL distribution of drift scores on
held-out in-distribution data (data the detector has not seen but which is
NOT drifted), then set the threshold at a high percentile of that null
distribution (Augur Section 2.3 / Table 2's "Detection Threshold" column;
default here is the p99 rule already named as the intended approach in DP4).

This does NOT reproduce Augur's full toolset (their Trainer/Drifter/Predictor
pipeline with 8 engineered drift scenarios and a supplementary ARIMA
time-series model of expected event rate) — see context/idea.md for why that
full pipeline is out of scope here and what a reduced version would look
like. This script reuses OUR existing (and Augur-inspired) drift detectors
against OUR existing held-out data, which is the part of Augur's process
directly actionable without new infrastructure.

Detector "kl_overflow" (2026-10-06, audit N1) is the one the concurrent
harness actually runs (concurrent_harness/mape/drift_ref.py): KL against the
training histogram with 50 bins plus two open-ended overflow bins, so values
outside the training range raise the score. Calibrate tau_drift with it.

Null pool: --null-stream-rows N uses the first N stream rows (before the first
injected drift region) instead of val_split(): pems_driftinduced 2000,
spot_prices_driftinduced 2364, uci_electricity_driftinduced 4975 (rows where
the *_driftInduced stream first differs from the clean one). --write stores
the kl_overflow threshold as tau_drift in the dataset config.

Usage:
    cd tool/
    python3 experiments/calibrate_drift_threshold.py --dataset pems_driftinduced \
        --detectors kl_overflow --window-size 1200 --null-stream-rows 2000 --write
    python3 experiments/calibrate_drift_threshold.py --dataset pems
    python3 experiments/calibrate_drift_threshold.py --dataset imagenet_c \\
        --detectors kl_fixed_ref,hellinger_fixed_ref --window-size 300
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_TOOL_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))
_DEFAULT_WINDOW_SIZE = 200
_DEFAULT_PERCENTILE = 99.0

class _KLOverflowDetector:
    """KL vs the training histogram with overflow bins — same scoring as
    concurrent_harness/mape/drift_ref.PerModelDriftReference."""

    def __init__(self, n_bins: int = 50) -> None:
        self.n_bins = n_bins
        self.edges = None
        self.ref = None

    def fit_reference(self, values) -> None:
        hist, base_edges = np.histogram(np.asarray(values, dtype=float), bins=self.n_bins)
        self.edges = np.concatenate(([-np.inf], base_edges, [np.inf]))
        self.ref = np.concatenate(([0.0], hist.astype(float), [0.0]))

    def score(self, window) -> float:
        from core.drift.kl_fixed_ref import _kl_divergence
        hist, _ = np.histogram(np.asarray(window, dtype=float), bins=self.edges)
        return float(_kl_divergence(hist.astype(float), self.ref))


_DETECTOR_BUILDERS: dict[str, str] = {
    "kl_overflow": "kl_overflow",
    "kl_fixed_ref": "kl_fixed_ref",
    "hellinger_fixed_ref": "hellinger_fixed_ref",
    "energy_distance": "energy_distance",
}


def _make_detector(name: str, window_size: int, n_bins: int = 50):
    if name == "kl_overflow":
        return _KLOverflowDetector(n_bins=n_bins)
    if name == "kl_fixed_ref":
        from core.drift.kl_fixed_ref import KLFixedRefDetector
        # KLFixedRefDetector loads its reference from a JSON file rather than
        # via fit_reference(); the caller writes a temp reference file first
        # (see _fit_reference_for below) and passes its path in via a
        # detector-specific hook. We build it here with a placeholder path
        # and let _fit_reference_for overwrite it.
        return KLFixedRefDetector(
            reference_path="", tau_drift=0.0, window_size=window_size, n_bins=n_bins
        )
    if name == "hellinger_fixed_ref":
        from core.drift.hellinger import HellingerFixedRefDetector
        return HellingerFixedRefDetector(tau_drift=0.0, window_size=window_size, n_bins=n_bins)
    if name == "energy_distance":
        from core.drift.energy_distance import EnergyDistanceDetector
        return EnergyDistanceDetector(tau_drift=0.0, window_size=window_size)
    raise ValueError(f"Unknown detector '{name}'. Available: {list(_DETECTOR_BUILDERS)}")


def _fit_reference_for(detector, name: str, train_values: np.ndarray, tmp_ref_path: Path) -> None:
    if name == "kl_fixed_ref":
        hist, bin_edges = np.histogram(train_values, bins=detector.n_bins)
        with open(tmp_ref_path, "w") as f:
            json.dump({"histogram": hist.tolist(), "bin_edges": bin_edges.tolist()}, f)
        detector.reference_path = str(tmp_ref_path)
        detector._load_reference()
    else:
        detector.fit_reference(train_values)


def _null_window_scores(
    detector, null_values: np.ndarray, window_size: int, stride: int
) -> list[float]:
    scores = []
    for start in range(0, max(1, len(null_values) - window_size + 1), stride):
        window = null_values[start : start + window_size]
        if len(window) < window_size:
            break
        s = detector.score(list(window))
        if s is not None:
            scores.append(s)
    return scores


def _load_regression_values(dataset_config: dict, tool_dir: Path, null_stream_rows: int | None = None):
    from experiments.run_experiment import _build_adapter
    adapter = _build_adapter(dataset_config, tool_dir / "configs" / "datasets")
    train_values = np.asarray(adapter.train_split(), dtype=float)
    if null_stream_rows:
        # first rows of the stream, before the first injected drift region
        val_values = np.asarray([s.ground_truth for _, s in zip(range(null_stream_rows), adapter.stream())],
                                dtype=float)
    else:
        val_values = np.asarray(adapter.val_split(), dtype=float) if hasattr(adapter, "val_split") else np.array([])
    return train_values, val_values


def _load_cv_luminance_values(dataset_config: dict, tool_dir: Path, sample_cap: int = 2000):
    from experiments.run_experiment import _build_adapter, _compute_image_luminance
    adapter = _build_adapter(dataset_config, tool_dir / "configs" / "datasets")
    train_paths = adapter.train_split()
    val_paths = adapter.val_split() if hasattr(adapter, "val_split") else []

    def _luminances(paths):
        if len(paths) > sample_cap:
            idx = np.random.default_rng(0).choice(len(paths), sample_cap, replace=False)
            paths = [paths[i] for i in sorted(idx)]
        return np.array([_compute_image_luminance(p) for p in paths], dtype=float)

    return _luminances(train_paths), _luminances(val_paths)


def calibrate(
    dataset_name: str,
    detector_names: list[str],
    window_size: int,
    percentile: float,
    stride: int | None,
    tool_dir: Path,
    null_stream_rows: int | None = None,
) -> dict:
    config_path = tool_dir / "configs" / "datasets" / f"{dataset_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    with open(config_path) as f:
        dataset_config = json.load(f)

    is_cv = dataset_config.get("domain") == "cv"
    if is_cv:
        train_values, val_values = _load_cv_luminance_values(dataset_config, tool_dir)
    else:
        train_values, val_values = _load_regression_values(dataset_config, tool_dir, null_stream_rows)

    if len(train_values) < window_size:
        raise ValueError(
            f"Training split ({len(train_values)} values) is smaller than "
            f"window_size ({window_size}) - cannot fit a meaningful reference."
        )

    null_values = val_values if len(val_values) >= window_size else train_values
    if val_values is None or len(val_values) < window_size:
        logger.warning(
            "val_split() has fewer than window_size values (%d) - falling back to "
            "the training split for the null distribution. This UNDERSTATES the "
            "true null variance (the detector was fit on the same data it is being "
            "scored against). Prefer a dataset config with val_frac > 0 for a "
            "trustworthy calibration.",
            len(val_values) if val_values is not None else 0,
        )

    # small null pools: ~20 (overlapping) windows rather than 1-2
    stride = stride or max(1, min(window_size // 2, (len(null_values) - window_size) // 20 or 1))
    tmp_ref_path = tool_dir / "runs" / f"_calibrate_tmp_ref_{dataset_name}.json"
    tmp_ref_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {"dataset": dataset_name, "window_size": window_size,
                     "percentile": percentile, "n_train": len(train_values),
                     "n_null_pool": len(null_values), "detectors": {}}

    print(f"\n{'Metric':<22}{'Threshold':>12}{'Mean':>12}{'Std':>12}{'N windows':>12}")
    print("-" * 70)
    for name in detector_names:
        detector = _make_detector(name, window_size)
        _fit_reference_for(detector, name, train_values, tmp_ref_path)
        scores = _null_window_scores(detector, null_values, window_size, stride)
        if not scores:
            logger.warning("Detector '%s': no valid windows scored - skipping.", name)
            continue
        arr = np.array(scores)
        threshold = float(np.percentile(arr, percentile))
        entry = {
            "detection_threshold": round(threshold, 6),
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "n_windows": len(scores),
        }
        report["detectors"][name] = entry
        print(f"{name:<22}{entry['detection_threshold']:>12.6f}{entry['mean']:>12.6f}"
              f"{entry['std']:>12.6f}{entry['n_windows']:>12d}")

    tmp_ref_path.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate per-dataset drift-detection thresholds (DP4, Augur-inspired)."
    )
    parser.add_argument("--dataset", required=True, help="Dataset config name (no .json).")
    parser.add_argument("--detectors", default="kl_fixed_ref,hellinger_fixed_ref,energy_distance",
                        help="Comma-separated detector names.")
    parser.add_argument("--window-size", type=int, default=_DEFAULT_WINDOW_SIZE,
                        help=f"Must match the runtime drift window for the calibrated "
                             f"threshold to be meaningful (default {_DEFAULT_WINDOW_SIZE}; "
                             "regression's live default is 1200 - pass --window-size 1200 "
                             "to match it).")
    parser.add_argument("--percentile", type=float, default=_DEFAULT_PERCENTILE,
                        help="Null-distribution percentile used as the threshold (default p99).")
    parser.add_argument("--stride", type=int, default=None,
                        help="Window stride for the null-score sweep (default window_size // 2).")
    parser.add_argument("--null-stream-rows", type=int, default=None,
                        help="Use the first N stream rows (pre-drift) as the null pool.")
    parser.add_argument("--write", action="store_true",
                        help="Write the kl_overflow threshold into the config as tau_drift.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: runs/calibration_<dataset>.json).")
    args = parser.parse_args()

    detector_names = [d.strip() for d in args.detectors.split(",") if d.strip()]
    report = calibrate(
        args.dataset, detector_names, args.window_size, args.percentile, args.stride, _TOOL_DIR,
        null_stream_rows=args.null_stream_rows,
    )

    out_path = Path(args.output) if args.output else _TOOL_DIR / "runs" / f"calibration_{args.dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved calibration report to {out_path}")
    if args.write:
        _write_tau(args.dataset, report, args)
    else:
        print("Pass --write to store the kl_overflow threshold as tau_drift.")


def _write_tau(dataset: str, report: dict, args) -> None:
    """Edit only the tau_drift / tau_drift_source lines of the config."""
    import re
    entry = report["detectors"].get("kl_overflow")
    if entry is None:
        raise SystemExit("--write needs the kl_overflow detector in --detectors")
    path = _TOOL_DIR / "configs" / "datasets" / f"{dataset}.json"
    text = path.read_text()
    tau = entry["detection_threshold"]
    source = (f"calibrate_drift_threshold kl_overflow p{args.percentile:g} "
              f"window={args.window_size} null={'stream[:%d]' % args.null_stream_rows if args.null_stream_rows else 'val_split'}")
    for key, val in (("tau_drift", tau), ("tau_drift_source", source)):
        pat = re.compile(rf'^(\s*)"{key}":\s*[^,\n]+(,?)$', re.M)
        if not pat.search(text):
            raise SystemExit(f'{path}: no "{key}" line')
        text = pat.sub(lambda m: f'{m.group(1)}"{key}": {json.dumps(val)}{m.group(2)}', text, count=1)
    json.loads(text)
    path.write_text(text)
    print(f"wrote tau_drift={tau} to {path.name}")


if __name__ == "__main__":
    main()

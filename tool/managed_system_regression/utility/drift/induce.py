"""
induce.py — non-interactive drift-injection CLI for regression datasets
(pems, uci_electricity, spot_prices).

Matches the original HarmonE paper's methodology (context/HarmonE.pdf §4.2-4.3):
models are trained on the actual (clean) dataset; a SEPARATE drift-induced
copy of the stream/test data is what gets used for streaming inference and
accuracy evaluation. §4.3 specifically: "we induce controlled data drift by
applying a consistent scale-and-shift transformation twice to designated
segments of the test data ... simulate a scenario where the data distribution
first diverges and later realigns with a previously seen pattern" — pass the
same scale/shift to two --region flags to reproduce this exactly (the tool
supports any number of regions, not just two).

Non-interactive by design (was originally an interactive input()-driven tool;
revised so it can be invoked from a pipeline, e.g. run_regression.sh, with
the dataset and drift regions passed as arguments):

    python managed_system_regression/utility/drift/induce.py \\
        --dataset pems \\
        --region 1000 1500 1.5 50 \\
        --region 5000 5500 1.5 50

The ORIGINAL file is NEVER modified — this writes a new file,
`<name>_driftInduced<ext>`, alongside it:
    data/pems/flow_data_test.csv  ->  data/pems/flow_data_test_driftInduced.csv

uci_electricity/spot_prices are not pre-split (adapters/regression_csv.py
Mode A: one CSV, train/stream split by train_frac at read time) — the output
file is the FULL dataset with the training rows copied verbatim (byte-
identical to the original — "trained on the actual dataset" holds exactly)
and only the stream rows (index >= the train/stream boundary, read from that
dataset's own configs/datasets/<name>.json) transformed. A drift region can
never be placed before that boundary — rejected, not clipped — so it is not
possible to accidentally leak drift into what a model would be trained on.

Run from `tool/` — paths (including configs/datasets/) are relative to cwd,
not to this file, matching every other script in this repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

_CONFIGS_DIR = "configs/datasets"


def _resolve_target(dataset_name: str) -> dict:
    """Read configs/datasets/<name>.json and resolve where the stream data
    lives and which row index it is safe to start mutating from.

    Returns:
        {
          "file_path":    CSV to read (never written to),
          "value_column": column to mutate,
          "safe_start":   smallest row index allowed to be touched (0 for
                          pre-split/Mode B datasets — pems — since the whole
                          file is streaming data; > 0 for Mode A datasets,
                          resolved once the file length is known),
          "train_frac":   only present for Mode A datasets,
        }
    """
    config_path = os.path.join(_CONFIGS_DIR, f"{dataset_name}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Dataset config not found: {config_path}. Run this script from tool/."
        )
    with open(config_path) as f:
        cfg = json.load(f)

    value_column = cfg["value_column"]

    if "train_path" in cfg and "stream_path" in cfg:
        # Mode B — pre-split (pems). The stream file IS the streaming portion;
        # nothing in it belongs to training.
        return {
            "file_path": cfg["stream_path"],
            "value_column": value_column,
            "safe_start": 0,
        }

    # Mode A — single file + train_frac (uci_electricity, spot_prices).
    return {
        "file_path": cfg["data_path"],
        "value_column": value_column,
        "safe_start": None,
        "train_frac": float(cfg.get("train_frac", 0.8)),
    }


def _driftinduced_path(file_path: str) -> str:
    """data/pems/flow_data_test.csv -> data/pems/flow_data_test_driftInduced.csv"""
    stem, ext = os.path.splitext(file_path)
    return f"{stem}_driftInduced{ext}"


def _parse_region(raw: list[str]) -> tuple[int, int, float, float]:
    try:
        start, end = int(raw[0]), int(raw[1])
        scale, shift = float(raw[2]), float(raw[3])
    except ValueError as exc:
        raise ValueError(
            f"--region expects START END SCALE SHIFT (int int float float), got {raw}"
        ) from exc
    return start, end, scale, shift


def induce_drift(
    dataset_name: str,
    regions: list[tuple[int, int, float, float]],
    force: bool = False,
    plot_output: str | None = None,
) -> str:
    """Apply drift regions to dataset_name's stream data and write the result
    to a new `_driftInduced` file. Returns the output path on success; raises
    ValueError/FileNotFoundError/FileExistsError on any rejected input —
    never partially writes on failure.
    """
    target = _resolve_target(dataset_name)
    file_path = target["file_path"]
    value_column = target["value_column"]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    out_path = _driftinduced_path(file_path)
    if os.path.exists(out_path) and not force:
        raise FileExistsError(
            f"{out_path} already exists. Pass --force to overwrite it "
            "(the ORIGINAL source file is never touched either way)."
        )

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    # Cast to float64 unconditionally: pems's `flow` column is int64 (whole
    # vehicle counts) on disk, and a non-integer scale/shift produces floats
    # that modern pandas refuses to write back into an int column
    # (LossySetitemError). This changes representation, not values — every
    # downstream consumer (scaler, models) already treats it as float.
    df[value_column] = df[value_column].astype("float64")

    safe_start = target["safe_start"]
    if safe_start is None:
        safe_start = int(len(df) * target["train_frac"])
        print(
            f"[induce] {dataset_name} is not pre-split — rows 0..{safe_start - 1} "
            f"of {file_path} are the TRAINING portion (train_frac="
            f"{target['train_frac']}). Only rows {safe_start}..{len(df) - 1} are "
            "streaming data and eligible for drift regions."
        )

    if not regions:
        raise ValueError("At least one --region is required.")

    for start, end, scale, shift in regions:
        if start < safe_start:
            raise ValueError(
                f"Region [{start}, {end}] starts before the train/stream boundary "
                f"({safe_start}) — this would inject drift into training data. "
                f"Choose start >= {safe_start}."
            )
        if end < start or end >= len(df):
            raise ValueError(
                f"Region [{start}, {end}] is out of range for [{safe_start}, {len(df) - 1}]."
            )

    df_drift = df.copy()
    for start, end, scale, shift in regions:
        df_drift.loc[start:end, value_column] = df_drift.loc[start:end, value_column] * scale + shift
        print(f"[induce] {dataset_name}: rows [{start}, {end}] -> "
              f"{value_column} = {value_column} * {scale} + {shift}")

    if plot_output:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(df[value_column], label=f"Original {value_column}", alpha=0.5)
        plt.plot(df_drift[value_column], label=f"Drifted {value_column}", color="red")
        if safe_start > 0:
            plt.axvline(safe_start, color="gray", linestyle="--", label="train/stream boundary")
        plt.title(f"{dataset_name} {value_column} with induced drift")
        plt.xlabel("Index")
        plt.ylabel(value_column)
        plt.legend()
        plt.savefig(plot_output, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[induce] Saved comparison plot to {plot_output}")

    df_drift.to_csv(out_path, index=False)
    print(f"[induce] Wrote {out_path} ({len(df_drift)} rows). "
          f"Original {file_path} was not modified.")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Non-interactive drift injection for a regression dataset's "
                     "stream data. Writes <name>_driftInduced<ext> alongside the "
                     "original file; never modifies the original.",
    )
    parser.add_argument("--dataset", required=True,
                         choices=["pems", "uci_electricity", "spot_prices"])
    parser.add_argument(
        "--region", action="append", nargs=4, metavar=("START", "END", "SCALE", "SHIFT"),
        required=True,
        help="Row range [START, END] (inclusive) and the transform "
             "value = value*SCALE + SHIFT to apply to it. Repeatable — pass "
             "the same SCALE/SHIFT twice (per HarmonE.pdf §4.3) to simulate "
             "drift that diverges then realigns with a previously seen pattern.",
    )
    parser.add_argument("--force", action="store_true",
                         help="Overwrite an existing <name>_driftInduced<ext> file.")
    parser.add_argument("--plot-output", default=None,
                         help="Optional path to save a before/after comparison PNG "
                              "(no interactive display — this tool never blocks on a GUI).")
    args = parser.parse_args()

    try:
        regions = [_parse_region(r) for r in args.region]
        induce_drift(args.dataset, regions, force=args.force, plot_output=args.plot_output)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"[induce] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

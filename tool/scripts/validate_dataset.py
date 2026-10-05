"""scripts/validate_dataset.py — CLI wrapper for core/dataset_validator.py.

Usage:
    cd tool/
    python3 scripts/validate_dataset.py --config pems_node1
    python3 scripts/validate_dataset.py --config bdd100k
    python3 scripts/validate_dataset.py --config /absolute/path/to/cfg.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TOOL_DIR))

from core.dataset_validator import validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a HarmonE dataset config + data.")
    parser.add_argument(
        "--config", required=True,
        help="Config name (e.g. pems_node1) or absolute path to .json file",
    )
    args = parser.parse_args()

    cfg_arg = args.config
    cfg_path = Path(cfg_arg)
    if not cfg_path.is_absolute():
        cfg_path = _TOOL_DIR / "configs" / "datasets" / f"{cfg_arg}.json"

    report = validate(cfg_path)
    report.print_table()
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()

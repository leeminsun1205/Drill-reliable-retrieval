#!/usr/bin/env python3
"""Print a machine-readable validation report for DRiLL Parquet files."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.cli import validate_main


if __name__ == "__main__":
    validate_main()

#!/usr/bin/env python3
"""Run one configured retrieval experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.cli import run_main


if __name__ == "__main__":
    run_main()

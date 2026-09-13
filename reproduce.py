#!/usr/bin/env python
"""Regenerate every table and figure this paper reports.

Runs against the aggregate result files committed here, so it needs no
credentialed data and no GPU. It reads `results/`, rebuilds the table panels,
the cell index every number in the manuscript resolves to, and the six figures,
and writes them back into `results/`.

    python reproduce.py            # tables and figures
    python reproduce.py --tables   # tables only
    python reproduce.py --figures  # figures only

Re-deriving those result files from the raw data is a different and much longer
path: see the README, which gives the stages in order.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

TABLES = [("make_table_main.py", "the table panels and the cell index")]
FIGURES = [
    ("make_figures.py", "the four data figures"),
    ("make_calibration_figures.py", "the reliability curves"),
]


def run(script: str, what: str) -> int:
    print(f"==> {what} ({script})")
    completed = subprocess.run([sys.executable, str(SCRIPTS / script)],
                               cwd=ROOT)
    return completed.returncode


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", action="store_true")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args(argv)

    steps = []
    if args.tables or not args.figures:
        steps += TABLES
    if args.figures or not args.tables:
        steps += FIGURES

    failed = [s for s, what in steps if run(s, what) != 0]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nRegenerated. Tables are results/table_main.md and "
          "results/table_main_cells.csv; figures are results/fig/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python
"""Score the exploratory triage-acuity comparator against the declared arms.

EXPLORATORY. Both external reviews asked how much the acquisition-context block
adds beyond the clinical acuity a triage nurse already records, and no arm in the
paper isolated that feature. `R9_triage` does, and this scores it.

It is deliberately NOT folded into `score_arms.py`. That script produces the
declared family's results, and an exploratory arm added after those results
existed has no business changing the file they live in: a reader comparing two
versions of `p3_arms.csv` should see a difference only when a declared arm
changed. So this writes `results/exploratory_triage.csv` and touches nothing
else, on the same reasoning as the matching sensitivity in
`build_matched_cohort.py --sensitivity`.

Two quantities, both patient-clustered and both on the same 10,000 resamples the
rest of the paper uses. The arm's own AUROC with an interval, and the paired
difference `R3_acqctx_pre - R9_triage`, which is the quantity the reviews are
really asking about: what the other eight pre-acquisition features are worth once
acuity is already in hand. The difference is paired because both arms score the
same records, and a marginal comparison of two overlapping intervals is exactly
what this project refuses to do elsewhere.

One property of this arm is worth knowing before reading its spread. It fits a
single ordinal feature, so every seed produces an identical model and the
seed-to-seed spread is exactly zero. That is a fact about a one-feature tree
rather than a scoring bug, and it means the noise floor this project judges
differences against does not exist for this arm; the interval is what carries the
uncertainty here.

    python scripts/score_exploratory_triage.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import clustered_statistic_ci  # noqa: E402
from contract import load_base_config  # noqa: E402
from paths import REPO_ROOT, derived_root  # noqa: E402
from score_arms import Cell, build_universe, join_arms, ledger_index, pooled_arm  # noqa: E402

RESULTS = REPO_ROOT / "results"

ARM = "R9_triage"
REFERENCE = "R3_acqctx_pre"
FLOOR = "R2_demo"

LABELS = [
    "deterioration_icu_24h",
    "deterioration_mortality_28d",
    "deterioration_mortality_365d",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    base = load_base_config()
    seeds = list(range(int(base["seeds"]))) if isinstance(base.get("seeds"), int) \
        else list(base.get("seeds", [0, 1, 2, 3, 4]))
    derived = derived_root()
    index = ledger_index()

    pooled: dict = {}
    rows: list[dict] = []
    for label in LABELS:
        for arm in (ARM, REFERENCE, FLOOR):
            pooled[(arm, label)] = pooled_arm(arm, label, seeds, index, derived)

        joined = join_arms(pooled, [ARM, REFERENCE, FLOOR], label)
        if joined is None:
            print(f"  {label}: not all arms have predictions; skipped",
                  file=sys.stderr)
            continue
        cell = Cell(joined, [ARM, REFERENCE, FLOOR])
        indices, n_groups = build_universe({"c": cell})

        for arm in (ARM, REFERENCE, FLOOR):
            def statistic(gathered, cell=cell, arm=arm):
                return cell.auc(arm, gathered["c"])

            result = clustered_statistic_ci(indices, n_groups, statistic,
                                            n_boot=args.n_boot, seed=args.seed)
            result.update(arm=arm, label=label, unit="record",
                          quantity="auroc", n_items=cell.n_items)
            rows.append(result)

        # The quantity the reviews are asking for, paired on the same patients.
        def difference(gathered, cell=cell):
            rowsel = gathered["c"]
            return cell.auc(REFERENCE, rowsel) - cell.auc(ARM, rowsel)

        result = clustered_statistic_ci(indices, n_groups, difference,
                                        n_boot=args.n_boot, seed=args.seed)
        result.update(arm=f"{REFERENCE} - {ARM}", label=label, unit="record",
                      quantity="paired_difference", n_items=cell.n_items)
        rows.append(result)

        print(f"  {label}: {ARM} "
              f"{rows[-4]['estimate']:.4f}, {REFERENCE} {rows[-3]['estimate']:.4f}, "
              f"{FLOOR} {rows[-2]['estimate']:.4f}, difference "
              f"{rows[-1]['estimate']:+.4f} "
              f"[{rows[-1]['ci_lo']:+.4f}, {rows[-1]['ci_hi']:+.4f}]")

    frame = pd.DataFrame(rows)
    frame["exploratory"] = True
    frame["comparison"] = "exploratory_triage"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "exploratory_triage.csv"
    frame.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

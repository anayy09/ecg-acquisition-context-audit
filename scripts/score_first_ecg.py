#!/usr/bin/env python
"""Score the waveform arms on the first ECG of each stay, as the benchmark does.

The audited benchmark restricts validation and test scoring to the first record
per visit. This project scores all 6,439 test records, which is a different
denominator, and until now the manuscript compared its all-records macro against
a published first-record one without saying so. Review 2 asked for the
restriction to be reported as a separate analysis rather than folded into the
reproduction, and that is what this is.

Nothing is retrained. The stored test-fold predictions carry `study_id`, so the
restriction is a row selection over scores that already exist, and the arms are
byte-identical to the ones the rest of the paper reports.

Three things about the restriction are worth knowing before reading the output,
and all three are in `results/cohort_reconciliation.csv` too.

`ecg_no_within_stay` is ZERO-based: the first ECG of a stay carries 0. Reading it
as a count selects the SECOND ECG of each stay and 318 rows instead of 6,080.

Four stays carry two records both numbered 0, because their acquisition times tie
and the ordering does not break ties. This script keeps both rather than picking
one, since picking one would be an undeclared tie-break, and reports the count so
the restricted set's 6,080 records over 6,076 stays is legible rather than
puzzling.

And the restriction removes 359 records, 5.6 percent of the test fold. It is a
real subset rather than a rounding, which is why it is worth scoring, but it is
small enough that a large move would be surprising.

    python scripts/score_first_ecg.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import REPO_ROOT, derived_root  # noqa: E402
from score_reproduction import load_runs, macro_clustered_ci  # noqa: E402

RESULTS = REPO_ROOT / "results"

#: Zero-based. 0 is the first ECG of the stay.
FIRST = 0


def first_ecg_study_ids(derived: Path) -> tuple[set[int], dict]:
    """The test-fold study ids that are first in their stay, and the counts."""
    acq = pd.read_parquet(
        derived / "features_acq.parquet",
        columns=["study_id", "split", "ecg_no_within_stay", "stay_id"])
    test = acq[acq["split"] == "test"]
    first = test[test["ecg_no_within_stay"] == FIRST]
    ties = first["stay_id"].value_counts()
    return set(first["study_id"].astype(int)), {
        "test_records": int(len(test)),
        "first_records": int(len(first)),
        "first_stays": int(first["stay_id"].nunique()),
        "tied_stays": int((ties > 1).sum()),
        "dropped_records": int(len(test) - len(first)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    # Only the reproduction arm by default. R7 is waveform plus demographics
    # plus acquisition context; it reproduces nothing, so a first-record
    # sensitivity on it answers no question anyone asked, and scoring it doubles
    # a job that takes twenty minutes and has twice been killed for memory on
    # this machine. Pass --arm to widen it.
    parser.add_argument("--arm", action="append", default=None,
                        help="arm to score; repeatable. Default: R6_waveform")
    args = parser.parse_args(argv)
    wanted = set(args.arm or ["R6_waveform"])

    derived = derived_root()
    keep, counts = first_ecg_study_ids(derived)
    print(f"test fold {counts['test_records']:,} records; "
          f"first of stay {counts['first_records']:,} over "
          f"{counts['first_stays']:,} stays "
          f"({counts['tied_stays']} stays tie); "
          f"dropped {counts['dropped_records']:,}")

    runs = [r for r in load_runs(derived) if r["arm"] in wanted]
    if not runs:
        print("no waveform runs with predictions; nothing to score", file=sys.stderr)
        return 1

    rows = []
    for run in runs:
        blob = np.load(run["predictions"], allow_pickle=True)
        targets = [str(t) for t in blob["targets"]]
        study = blob["study_id"].astype(int)
        subject = blob["subject_id"]
        selected = np.fromiter((s in keep for s in study), dtype=bool,
                               count=len(study))

        for unit in ("crop", "record"):
            if unit == "crop":
                local = blob["crop_local"]
                rowsel = selected[local]
                y = blob["crop_y"][rowsel]
                sc = blob["crop_scores"][rowsel]
                mk = blob["crop_mask"][rowsel]
                groups = subject[local][rowsel]
            else:
                y = blob["record_y"][selected]
                sc = blob["record_scores"][selected]
                mk = blob["record_mask"][selected]
                groups = subject[selected]

            res = macro_clustered_ci(y, sc, mk, groups, targets,
                                     n_boot=args.n_boot, seed=args.seed)
            rows.append({
                "run_id": run["run_id"], "arm": run["arm"], "seed": run["seed"],
                "unit": unit, "restriction": "first_ecg_per_stay",
                "macro": res["macro"], "macro_lo": res["macro_lo"],
                "macro_hi": res["macro_hi"], "n_items": res["n_items"],
                "n_groups": res["n_groups"], "n_boot": res["n_boot"],
                "n_targets_scored": res["n_targets_scored"],
            })
        print(f"  {run['run_id']} {run['arm']} seed {run['seed']}: "
              f"crop {rows[-2]['macro']:.4f}, record {rows[-1]['macro']:.4f}")

    frame = pd.DataFrame(rows)
    for key, value in counts.items():
        frame[key] = value
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "first_ecg_sensitivity.csv"
    frame.to_csv(out, index=False)

    # The comparison the reproduction section needs: our macro on the protocol
    # the published figure was computed under, beside our macro on all records.
    allrecords = pd.read_csv(RESULTS / "p2_macro.csv")
    print()
    for arm in sorted(frame["arm"].unique()):
        for unit in ("crop", "record"):
            restricted = frame[(frame["arm"] == arm) & (frame["unit"] == unit)]
            full = allrecords[(allrecords["arm"] == arm)
                              & (allrecords["unit"] == unit)]
            if restricted.empty or full.empty:
                continue
            a, b = float(restricted["macro"].mean()), float(full["macro"].mean())
            print(f"  {arm:<22} {unit:<7} first-ECG {a:.4f}  "
                  f"all records {b:.4f}  difference {a - b:+.4f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

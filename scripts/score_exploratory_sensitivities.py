#!/usr/bin/env python
"""Score the two exploratory sensitivities declared in `exploratory_sensitivities`.

EXPLORATORY. Neither arm here tests a registered prediction and neither enters a
multiplicity family. Both qualify something the manuscript already says.

**The informative-features arm.** Two of the nine pre-acquisition features are
emptied by MIMIC-IV's de-identification: a random whole-day offset per patient
rotates the weekday and smears the month, so the recorded values carry nothing
about local practice. The declared arm keeps them because the feature set was
fixed before that was established. This scores the same arm without them, and
the paired difference says what they were worth.

**The shared-configuration arms.** The recovery ratio divides a tabular arm
tuned per target over fifty trials by a waveform arm selected once, from four
candidates, on a validation macro across all fifteen deterioration targets. That
inequality reaches the ratio and inflates it. A comparably tuned waveform arm is
out of reach here; refitting the TABULAR side under the waveform arm's own
procedure, four candidates and one macro, is not, and it bounds the inequality
from the side that can be measured.

Every quantity is patient-clustered on the same 10,000 resamples as the rest of
the paper, and every difference is paired on the same records.

    python scripts/score_exploratory_sensitivities.py
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
from score_arms import (Cell, build_universe, join_arms, ledger_index,  # noqa: E402
                      pooled_arm, recovery_ratio)

RESULTS = REPO_ROOT / "results"

DENOMINATOR = "R6_waveform"

#: (arm under test, the declared arm it qualifies, what the pair measures)
PAIRS = [
    ("R10_acqctx_informative", "R3_acqctx_pre", "calendar_features"),
    ("R3s_acqctx_shared", "R3_acqctx_pre", "shared_configuration"),
    ("R4s_demo_acq_shared", "R4_demo_acq", "shared_configuration"),
]

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
    seeds = list(base["training"]["seeds"])
    derived = derived_root()
    index = ledger_index()

    pooled: dict = {}
    rows: list[dict] = []

    for arm, declared, question in PAIRS:
        for label in LABELS:
            wanted = [arm, declared, DENOMINATOR]
            for member in wanted:
                if (member, label) not in pooled:
                    pooled[(member, label)] = pooled_arm(
                        member, label, seeds, index, derived)

            joined = join_arms(pooled, wanted, label)
            if joined is None:
                print(f"  {arm} {label}: not all arms scored; skipped",
                      file=sys.stderr)
                continue
            cell = Cell(joined, wanted)
            indices, n_groups = build_universe({"c": cell})

            # Each arm's own AUROC, and its recovery ratio on the same cell as
            # the arm it qualifies, so the two ratios are comparable rather
            # than merely similar.
            for member in (arm, declared):
                for name, fn in (
                    ("auroc",
                     lambda rows_, c=cell, m=member: c.auc(m, rows_["c"])),
                    ("recovery_ratio",
                     lambda rows_, c=cell, m=member: recovery_ratio(
                         c.auc(m, rows_["c"]), c.auc(DENOMINATOR, rows_["c"]))),
                ):
                    result = clustered_statistic_ci(
                        indices, n_groups, fn, n_boot=args.n_boot,
                        seed=args.seed)
                    result.update(
                        arm=member, label=label, unit="record", quantity=name,
                        question=question, declared_arm=declared,
                        n_items=cell.n_items)
                    rows.append(result)

            # The paired difference, which is what each question is about.
            def difference(rows_, c=cell, a=arm, d=declared):
                return c.auc(d, rows_["c"]) - c.auc(a, rows_["c"])

            result = clustered_statistic_ci(indices, n_groups, difference,
                                            n_boot=args.n_boot, seed=args.seed)
            result.update(arm=f"{declared} - {arm}", label=label, unit="record",
                          quantity="paired_difference", question=question,
                          declared_arm=declared, n_items=cell.n_items)
            rows.append(result)

            auc_arm = cell.auc(arm, np.arange(cell.n_items))
            auc_dec = cell.auc(declared, np.arange(cell.n_items))
            print(f"  {question:22} {arm:24} {label:32} "
                  f"{auc_arm:.4f} vs {auc_dec:.4f} declared, "
                  f"difference {auc_dec - auc_arm:+.4f}")

    frame = pd.DataFrame(rows)
    frame["exploratory"] = True
    frame["comparison"] = "exploratory_sensitivities"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "exploratory_sensitivities.csv"
    frame.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

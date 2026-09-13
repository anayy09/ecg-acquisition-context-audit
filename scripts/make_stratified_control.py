#!/usr/bin/env python
"""Write the within-stratum controls into a generated artifact.

Review round 1, priority fix 3. The manuscript quotes the unstratified value of
the constant-within-stratum control score, and it quoted 0.72 while the control
produced 0.7406. That survived because the corresponding check listed "0.72" among
the decimals a manuscript may state without a pointer, described as a
definitional constant. It is not a constant; it is a measurement, and exempting
it from the pointer discipline is what stopped it being compared to anything.

So the control's measured values become a generated cell like every other number
in the paper. `stratified.control_values()` is the single definition, the
module's own `selftest()` asserts against it, and this writes it out.

    python scripts/make_stratified_control.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import REPO_ROOT  # noqa: E402
from stratified import control_values, selftest  # noqa: E402

OUT = REPO_ROOT / "results" / "p4_control.csv"

#: What each control demonstrates, in the order the methods section states them.
ROWS = [
    ("single_stratum",
     "single_stratum_plain", "single_stratum_within",
     "one stratum reproduces plain AUROC exactly"),
    ("constant_within_stratum",
     "shortcut_plain", "shortcut_within",
     "a score constant within every stratum collapses to chance"),
    ("duplicated_rows",
     "single_stratum_plain", "doubled_within",
     "duplicating every row leaves the statistic unchanged"),
]


def main() -> int:
    problems = selftest()
    if problems:
        print("the within-stratum controls do not pass; refusing to write a "
              "table from a statistic that fails its own checks:", file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        return 1

    values = control_values()
    frame = pd.DataFrame(
        [
            {
                "control": name,
                "unstratified": values[plain_key],
                "within_stratum": values[within_key],
                "n_items": int(values["n"]),
                "n_strata": int(values["n_strata"]) if name != "single_stratum" else 1,
                "demonstrates": text,
            }
            for name, plain_key, within_key, text in ROWS
        ]
    )
    OUT.write_text(frame.to_csv(index=False), encoding="utf-8")

    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    for _, row in frame.iterrows():
        print(f"  {row['control']:24s} unstratified={row['unstratified']:.4f} "
              f"within={row['within_stratum']:.4f}")
    print("STRATIFIED CONTROL WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Apply gate the pre-declared rule's rule to the scored waveform runs and write the verdict.

The rule is the reproduction tolerance's, not the project's design notes's original: the paper reports the model-variant
comparison only as a macro over all 15 deterioration targets, so there is no
published per-label number to compare against and the flat 0.02 AUROC tolerance
in the original reproduction tolerance is dropped. pre-declared rule 2 passes if the local macro falls inside the published
interval for the ECG-waveforms-only variant.

This script computes the verdict. the corresponding check recomputes it
independently from the same CSVs and refuses to agree by construction, which is
how the this stage verdict was checked and how a bug in the rule itself was caught there.

One thing the verdict must say out loud rather than bank quietly. Six of the 15
targets carry fewer than 50 positives in the test fold, the macro weights them
equally with targets carrying 800, and that is why the published interval spans
0.046 AUROC while the tabular variant's spans 0.003. pre-declared rule 2 is an easy gate. A pass
is reported with that stated.

    python scripts/evaluate_reproduction.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import REPO_ROOT  # noqa: E402

RESULTS = REPO_ROOT / "results"
AUDIT_ARM = "R6_waveform"


def evaluate(gap: pd.DataFrame, per_target: pd.DataFrame) -> dict:
    # Two published versions report this arm differently, so the gap table has a
    # row per anchor and the verdict has to say which one it is judged against.
    # Taking the first row would have picked whichever the scorer wrote first.
    rows = gap[gap["arm"] == AUDIT_ARM]
    if rows.empty:
        raise SystemExit(f"no {AUDIT_ARM} row in p2_reproduction_gap.csv")
    if "anchor_role" not in rows.columns:
        raise SystemExit(
            "p2_reproduction_gap.csv carries no anchor_role column, so the "
            "verdict cannot say which published figure it is judged against. "
            "Re-run scripts/score_reproduction.py")
    primary = rows[rows["anchor_role"] == "primary"]
    if primary.empty:
        raise SystemExit("no primary anchor row in p2_reproduction_gap.csv")
    row = primary.iloc[0]

    local = float(row["local_macro_mean"])
    lo, hi = float(row["published_lo"]), float(row["published_hi"])
    inside = lo <= local <= hi

    secondary = [
        {"anchor": str(other["anchor"]),
         "published_macro": float(other["published_macro"]),
         "published_ci": [float(other["published_lo"]), float(other["published_hi"])],
         "gap": float(other["gap"]),
         "inside_published_ci": bool(other["inside_published_ci"])}
        for _, other in rows[rows["anchor_role"] != "primary"].iterrows()
    ]

    # Sparsity is counted at RECORD level even though the gate is scored at crop
    # level, because a crop-level count is exactly four times the record count:
    # every ECG contributes the same event four times. Reporting 36 crop
    # positives for `deterioration_ecmo` would describe nine distinct events as
    # though they were thirty-six, and would make the macro's weakest targets
    # look four times better informed than they are. Counting distinct ECGs puts
    # six targets under fifty rather than two.
    sparse = (
        per_target[(per_target["arm"] == AUDIT_ARM) & (per_target["unit"] == "record")]
        .groupby("target")["n_positive"].min().sort_values()
    )
    n_sparse = int((sparse < 50).sum())

    verdict = {
        "gate": "G2",
        "rule": "the reproduction tolerance: the local macro over all 15 deterioration targets falls "
                "inside the published ECG-waveforms-only interval",
        "supersedes": "the project's design notes's original 'within 0.02 AUROC' text, and the original reproduction tolerance",
        "arm": AUDIT_ARM,
        "unit": "crop",
        "unit_note": "crop level, because that is what the published macro computes "
                     "over; see the crop-level and record-level separation",
        "local_macro": local,
        "local_macro_seed_spread": float(row["seed_spread"]),
        "n_seeds": int(row["n_seeds"]),
        "anchor": str(row["anchor"]),
        "published_macro": float(row["published_macro"]),
        "published_ci": [lo, hi],
        "other_anchors": secondary,
        "gap": float(row["gap"]),
        "inside_published_ci": bool(inside),
        "result": "PASS" if inside else "FAIL",
        "branch": (
            "Proceed. The local reproduction lands inside the published interval, so "
            "the comparison rows 3 and 6 rest on is fair."
            if inside else
            "Do NOT proceed to the headline claim on this comparison. the project's design notes the pre-declared rule's "
            "second branch applies: reframe as a reproducibility audit plus a metadata "
            "baseline, which is weaker and still publishable, and record the reframe."
        ),
        "caveat": (
            f"pre-declared rule 2 as the reproduction tolerance defines it is an easy gate to pass. The published interval "
            f"spans {hi - lo:.4f} AUROC because {n_sparse} of the 15 targets carry "
            f"fewer than 50 distinct positive ECGs in the test fold, two of them only "
            f"nine, and the unweighted macro gives each the same weight as targets "
            f"carrying over 800. The tabular variant's published interval, over the "
            f"same 15 targets, spans 0.0033. A pass here is evidence that the "
            f"reproduction is not badly wrong; it is not evidence that it is precisely "
            f"right."
        ),
        "sparsity_counted_at": "record level; a crop-level count would be 4x larger "
                               "for the same events and would overstate how well "
                               "informed the sparse targets are",
        "sparse_targets": {t: int(n) for t, n in sparse.items() if n < 50},
    }

    # The full-strength branch of pre-declared rule 1 was left PENDING in the first verdict for want of a
    # locally reproduced waveform number. It resolves here.
    verdict["g1_full_strength_branch"] = {
        "rule": "the project's design notes pre-declared rule 1: continue at full strength if R3_acqctx_pre comes within "
                "0.03 AUROC of the waveform arm on at least one label",
        "status": "resolvable now that a local row 6 exists; evaluated in this stage against "
                  "the record-level macro, since rows 3 and 6 must be compared on the "
                  "same unit",
    }
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    gap_path = RESULTS / "p2_reproduction_gap.csv"
    per_path = RESULTS / "p2_per_target.csv"
    for path in (gap_path, per_path):
        if not path.is_file():
            print(f"{path} is missing; run scripts/score_reproduction.py first", file=sys.stderr)
            return 1

    verdict = evaluate(pd.read_csv(gap_path), pd.read_csv(per_path))
    (RESULTS / "p2_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(f"Gate pre-declared rule 2: {verdict['result']}")
    print(f"  local macro   {verdict['local_macro']:.4f} "
          f"(seed spread {verdict['local_macro_seed_spread']:.4f}, "
          f"{verdict['n_seeds']} seeds)")
    print(f"  published     {verdict['published_macro']:.4f} "
          f"[{verdict['published_ci'][0]:.4f}, {verdict['published_ci'][1]:.4f}]")
    print(f"  gap           {verdict['gap']:+.4f}")
    print(f"\n{verdict['branch']}")
    print(f"\nCaveat: {verdict['caveat']}")
    print(f"\nwrote {RESULTS / 'p2_verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Reconcile the reproduced cohort against the published one, line by line.

Review 2 could not tell, from the manuscript, whether the 38-record difference
between our 129,057 records and the release's 129,095 was a cohort built
differently or a release withheld differently. The cohort reconciliation has had the answer since this stage
and the paper never carried it. Worse, the question generalises: the review also
asked about unique stays, ECG sequence numbers within a stay, label exclusions
and how validation and test rows are selected, none of which the manuscript
states and all of which a reader of an audit is entitled to.

This writes `results/cohort_reconciliation.csv`, one row per quantity, with the
local value, the published value where one exists, and the gap. `make_table_main.py`
renders it; nothing here is typed into prose.

Three of the rows exist because building it turned something up.

The benchmark's own instructions restrict validation and test scoring to the
first ECG per stay, and this project scores every test record. That is a
different denominator and the reconciliation says how different: 6,080 of the
6,439 test records are the first of their stay.

`ecg_no_within_stay` is ZERO-based. The first ECG of a stay carries 0, not 1.
Anything reading it as a count rather than an index gets the restriction exactly
inverted, which is a mistake worth making impossible to repeat by writing the
convention into the table.

And four stays carry two records both numbered 0, because their acquisition
times tie and the ordering does not break ties. So "the first ECG of the stay"
is ambiguous for four of the 6,077 test stays, and the restricted set covers
6,076 distinct stays rather than 6,080. It cannot move a macro at four rows in
six thousand, and a reconciliation that quietly rounded it away would be the
kind of table this paper exists to complain about.

    python scripts/make_cohort_reconciliation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import REPO_ROOT, derived_root  # noqa: E402

RESULTS = REPO_ROOT / "results"
PUBLISHED = REPO_ROOT / "configs" / "published_reference.yaml"

#: The sentinel the released label columns use for an unobservable outcome.
MISSING = -999.0

#: The four labels the manuscript reports, in the order it reports them.
REPORTED_LABELS = [
    "deterioration_icu_24h",
    "deterioration_icu_stay",
    "deterioration_mortality_28d",
    "deterioration_mortality_365d",
]


def row(section: str, quantity: str, local, published=None, note: str = "") -> dict:
    """One reconciliation line. `published` is None where nothing is published."""
    gap = "" if published is None else int(local) - int(published)
    return {
        "section": section,
        "quantity": quantity,
        "local": int(local),
        "published": "" if published is None else int(published),
        "gap": gap,
        "note": note,
    }


def build(cohort: pd.DataFrame, published: dict) -> list[dict]:
    rows: list[dict] = []
    counts = published["cohort"]
    positives = published["label_positives"]

    rows.append(row("Cohort", "ECG records", len(cohort), counts["n_samples"],
                    "the whole released cohort"))
    rows.append(row("Cohort", "Patients", cohort["subject_id"].nunique(),
                    counts["n_patients"], "matches exactly"))
    rows.append(row("Cohort", "Emergency department stays",
                    cohort["stay_id"].nunique(), counts["n_visits"],
                    "matches exactly"))

    # The label columns are floats carrying -999.0 where the outcome is not
    # observable. Summing them is meaningless; the first run of this script did
    # exactly that and reported minus half a million positives. Equality with 1
    # is the count, and the sentinel is not discarded: it IS the label exclusion
    # the reconciliation is being asked for.
    for label in REPORTED_LABELS:
        if label not in cohort:
            continue
        column = pd.to_numeric(cohort[label], errors="coerce")
        if label in positives:
            rows.append(row(
                "Label positives", label, int((column == 1).sum()),
                positives[label],
                "counted over the whole cohort, as the release reports them"))
        rows.append(row(
            "Label exclusions", label, int((column == MISSING).sum()), None,
            "records where the outcome is not observable for this label; the "
            "tabular arms drop them, which is why the analysed sample differs "
            "by label"))

    for name, value in (("Training folds 0 to 17", "train"),
                        ("Validation fold 18", "val"),
                        ("Test fold 19", "test")):
        rows.append(row("Split", name, int((cohort["split"] == value).sum()),
                        None, "the published 20-fold split, unmodified"))

    test = cohort[cohort["split"] == "test"]
    rows.append(row("Test fold", "ECG records", len(test), None, ""))
    rows.append(row("Test fold", "Patients", test["subject_id"].nunique(), None, ""))
    rows.append(row("Test fold", "Stays", test["stay_id"].nunique(), None, ""))

    # The sequence number is zero-based: 0 is the first ECG of the stay.
    sequence = test["ecg_no_within_stay"]
    first = test[sequence == 0]
    rows.append(row(
        "First ECG per stay", "Records numbered 0 within their stay", len(first),
        None, "zero-based: 0 is the FIRST ECG of the stay, not the second"))
    rows.append(row(
        "First ECG per stay", "Distinct stays among them",
        first["stay_id"].nunique(), None,
        "fewer than the records above, because some stays tie"))
    ties = first["stay_id"].value_counts()
    rows.append(row(
        "First ECG per stay", "Stays carrying two records numbered 0",
        int((ties > 1).sum()), None,
        "acquisition times tie and the ordering does not break ties, so the "
        "first ECG of these stays is ambiguous"))
    rows.append(row(
        "First ECG per stay", "Records after the first of their stay",
        int((sequence > 0).sum()), None,
        "excluded by the benchmark's stated validation and test protocol and "
        "retained by ours"))

    for level in sorted(sequence.unique()):
        rows.append(row(
            "Sequence within stay", f"Records numbered {int(level)}",
            int((sequence == level).sum()), None,
            "first of the stay" if level == 0 else f"the {int(level) + 1}th"))

    return rows


def main() -> int:
    cohort = pd.read_parquet(derived_root() / "cohort_mdsed.parquet")
    published = yaml.safe_load(PUBLISHED.read_text(encoding="utf-8"))

    frame = pd.DataFrame(build(cohort, published))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "cohort_reconciliation.csv"
    frame.to_csv(out, index=False)

    width = max(len(str(q)) for q in frame["quantity"])
    for _, line in frame.iterrows():
        gap = f"  gap {line['gap']:+d}" if line["gap"] != "" else ""
        print(f"  {line['section']:<22} {line['quantity']:<{width}} "
              f"{line['local']:>8,}{gap}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

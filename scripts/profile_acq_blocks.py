#!/usr/bin/env python
"""Profile every acquisition-context feature: how much information it can carry.

Review round 1, priority fix 1. The manuscript offered the small gap between the
leakage-suspect arm and the confirmatory arm as evidence that no leak is
operating, and named `care_unit_at_acquisition` as the variable that made the
comparison informative. That variable is CONSTANT on all 129,057 records:
MDS-ED admits only patients with an ECG inside the first 90 minutes of the ED
stay, so the patient is in the Emergency Department at acquisition by
construction. A constant cannot leak, and a comparison against an arm whose
leak-prone variable is constant had little room to move.

The manuscript cannot state that without a table to point at, so this writes
one. It profiles BOTH blocks rather than just the suspect one, because the same
question applies in the other direction: if an ACQ_PRE feature were degenerate,
the confirmatory arm would be resting on fewer features than it claims. None is.

Aggregate only. Level counts, modal share and undefined share, never a row.
The containment rule and the row-level data rule keep row-level data out of this repository.

    python scripts/profile_acq_blocks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import load_contract  # noqa: E402
from paths import REPO_ROOT, data_root  # noqa: E402

OUT = REPO_ROOT / "results" / "acq_block_profile.csv"
FEATURES = "derived/features_acq.parquet"


def main() -> int:
    contract = load_contract()
    path = data_root() / FEATURES
    if not path.is_file():
        print(f"feature table not found at {path}", file=sys.stderr)
        return 1

    declared: list[tuple[str, str, str]] = []
    for block in ("acq_pre", "acq_window"):
        for derivation in contract.block(block):
            declared.append((block, derivation.name, derivation.available_at))

    frame = pd.read_parquet(path)
    rows = []
    for block, name, available_at in declared:
        if name not in frame.columns:
            rows.append({
                "block": block,
                "feature": name,
                "fitted": False,
                "n_levels": 0,
                "modal_share": float("nan"),
                "undefined_share": float("nan"),
                "available_at": available_at,
                "carries_information": False,
                "status": "dropped",
            })
            continue
        series = frame[name]
        defined = series.notna()
        counts = series.value_counts(dropna=True)
        modal = float(counts.iloc[0] / defined.sum()) if defined.sum() else float("nan")
        n_levels = int(series.nunique(dropna=True))
        rows.append({
            "block": block,
            "feature": name,
            "fitted": True,
            "n_levels": n_levels,
            "modal_share": modal,
            "undefined_share": float(series.isna().mean()),
            "available_at": available_at,
            # A single level carries no information at all, so it cannot be a
            # leak whatever its availability says. That is a statement about
            # this cohort, not about the feature in general. Recorded as a
            # BOOLEAN as well as a word, so the corresponding check can re-derive
            # the polarity of the table cell rather than trusting the word.
            "carries_information": bool(n_levels > 1),
            "status": "constant" if n_levels <= 1 else "varies",
        })

    profile = pd.DataFrame(rows)
    profile.insert(0, "n_records", len(frame))
    OUT.write_text(profile.to_csv(index=False), encoding="utf-8")

    print(f"wrote {OUT.relative_to(REPO_ROOT)} over {len(frame):,} records")
    for _, row in profile.iterrows():
        share = ("-" if pd.isna(row["modal_share"])
                 else f"{100 * row['modal_share']:.2f}%")
        print(f"  {row['block']:11s} {row['feature']:32s} {row['status']:8s} "
              f"levels={row['n_levels']:<6d} modal={share}")
    constants = profile[(profile["status"] == "constant")]
    if len(constants):
        print(f"  {len(constants)} feature(s) carry a single level and cannot "
              "contribute anything, in either direction")
    print("ACQ BLOCK PROFILE WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""The ACQ_PRE leakage guard.

One assertion, applied row by row to every feature in the confirmatory block:

    available_at__<feature> <= index_time

the two-block split promises that ACQ_PRE contains only values fixed at or before the index
acquisition. This is the test that makes the promise checkable rather than
rhetorical. It runs on the emitted table, not on the plan, so a derivation that
quietly starts reading a later timestamp fails here even though the config still
says the right thing.

Null handling is explicit: a null availability stamp is *not applicable*, never
*safe*. A row whose feature has a value but no stamp is a violation, because an
unstamped value cannot be shown to predate anything.

    python scripts/leakage_test.py --fixtures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import load_contract  # noqa: E402
from paths import derived_root, fixtures_root  # noqa: E402


class Violation(dict):
    """One feature's failure record."""


def audit(frame: pd.DataFrame, contract) -> tuple[list[Violation], list[dict]]:
    """Return (violations, per-feature summary) for the ACQ_PRE block."""
    violations: list[Violation] = []
    summary: list[dict] = []

    if "index_time" not in frame.columns:
        raise RuntimeError("feature table has no index_time column; cannot audit")

    index_time = frame["index_time"]

    for derivation in contract.block("acq_pre"):
        name = derivation.name
        stamp_column = f"available_at__{name}"
        if name not in frame.columns:
            # A dropped ACQ_PRE feature is a decision, not a leak; report it and
            # let the dictionary check decide whether the drop was recorded.
            summary.append({"feature": name, "status": "absent", "rows_checked": 0})
            continue
        if stamp_column not in frame.columns:
            violations.append(
                Violation(
                    feature=name,
                    kind="unstamped",
                    detail=f"{stamp_column} is missing; availability cannot be proven",
                    rows=int(frame[name].notna().sum()),
                )
            )
            continue

        stamp = frame[stamp_column]
        has_value = frame[name].notna()
        # Unstamped but valued: cannot be shown to predate the index.
        unstamped = has_value & stamp.isna()
        # Stamped after the index acquisition: an outright leak.
        comparable = stamp.notna() & index_time.notna()
        late = comparable & (stamp > index_time)

        if int(unstamped.sum()):
            violations.append(
                Violation(
                    feature=name,
                    kind="unstamped_value",
                    detail="feature has a value on rows with no availability stamp",
                    rows=int(unstamped.sum()),
                )
            )
        if int(late.sum()):
            worst = (stamp[late] - index_time[late]).max()
            violations.append(
                Violation(
                    feature=name,
                    kind="available_after_index",
                    detail=(
                        f"availability postdates the index acquisition by up to {worst}"
                    ),
                    rows=int(late.sum()),
                )
            )

        summary.append(
            {
                "feature": name,
                "status": "ok" if not (unstamped.sum() or late.sum()) else "VIOLATION",
                "rows_checked": int(comparable.sum()),
                "declared_rule": derivation.available_at,
            }
        )

    return violations, summary


def window_report(frame: pd.DataFrame, contract) -> list[dict]:
    """Informational: confirm ACQ_WINDOW really does resolve after the index.

    A window feature that turns out to be index-fixed is not dangerous, but it
    means the block split is describing something other than what the data does,
    and the leakage-suspect classification rests on that split being real.
    """
    rows: list[dict] = []
    index_time = frame["index_time"]
    for derivation in contract.block("acq_window"):
        stamp_column = f"available_at__{derivation.name}"
        if derivation.name not in frame.columns:
            rows.append({"feature": derivation.name, "status": "dropped"})
            continue
        stamp = frame[stamp_column]
        comparable = stamp.notna() & index_time.notna()
        after = int((comparable & (stamp > index_time)).sum())
        rows.append(
            {
                "feature": derivation.name,
                "status": "post-index" if after else "INDEX-FIXED",
                "rows_after_index": after,
                "rows_checked": int(comparable.sum()),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument("--features", default=None, help="path to features_acq.parquet")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    contract = load_contract()
    if args.features:
        path = Path(args.features)
    else:
        base = (fixtures_root() / "derived") if args.fixtures else derived_root()
        path = base / "features_acq.parquet"
    if not path.is_file():
        print(f"feature table not found at {path}", file=sys.stderr)
        return 2

    frame = pd.read_parquet(path)
    violations, summary = audit(frame, contract)
    windows = window_report(frame, contract)

    if not args.quiet:
        print(f"ACQ_PRE leakage audit: {path}")
        print(f"  rows: {len(frame)}   patients: {frame['subject_id'].nunique()}\n")
        for row in summary:
            print(
                f"  {row['status']:<10} {row['feature']:<34} "
                f"rows_checked={row['rows_checked']}"
            )
        print("\nACQ_WINDOW placement (informational, the leakage-suspect classification):")
        for row in windows:
            print(f"  {row['status']:<12} {row['feature']}")

    if violations:
        print(f"\nLEAKAGE GUARD FAILED: {len(violations)} violation(s)", file=sys.stderr)
        for violation in violations:
            print(
                f"  - {violation['feature']}: {violation['kind']} on "
                f"{violation['rows']} row(s); {violation['detail']}",
                file=sys.stderr,
            )
        return 1

    if not args.quiet:
        print("\nLEAKAGE GUARD PASSED: every ACQ_PRE feature is fixed at or before index_time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Emit the benchmark's tabular feature block for row 5, passed through unmodified.

The pass-through rule for row 5 fixes this arm's status: row 5 is the benchmark's own vitals, labs and
biometrics block, taken from the release rather than re-derived. Re-deriving it
would turn a clinical reference point into another of our own constructions, and
the point of row 5 is that it is not ours.

So this script selects, it does not compute. Every column whose name carries one
of the contract's `tabular_prefixes` is copied across as-is, with no imputation,
no scaling, no clipping and no feature engineering. Missingness is left in place
for the model to handle, which is what LightGBM does natively and what any
imputation here would quietly override.

The column count is a check in itself. The published signal-only config declares
`input_channels_cont: 463`, and biometrics (3) plus vitals (55) plus labvalues
(405) is exactly 463. A mismatch means the release is not the one the config was
written against.

    python scripts/build_tabular.py
    python scripts/build_tabular.py --dry-run    # report the column set only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import load_contract  # noqa: E402
from loaders import resolve_path  # noqa: E402
from paths import assert_outside_repo, data_root, derived_root  # noqa: E402

#: From the published config's `input_channels_cont`.
EXPECTED_CONTINUOUS = 463
KEY_SOURCE = "general_study_id"
KEY = "study_id"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    contract = load_contract()
    root = data_root(args.data_root)
    derived = derived_root(args.data_root)

    spec = contract.table_spec("mds_ed.cohort")
    prefixes = tuple(spec["tabular_prefixes"])
    source = resolve_path("mds_ed.cohort", root, contract)

    with source.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    selected = [c for c in header if c.startswith(prefixes)]
    by_prefix = {p: sum(1 for c in selected if c.startswith(p)) for p in prefixes}

    print(f"source: {source.name}, {len(header)} columns")
    for prefix, count in by_prefix.items():
        print(f"  {prefix:<14} {count:4d}")
    print(f"  {'total':<14} {len(selected):4d}")

    if len(selected) != EXPECTED_CONTINUOUS:
        print(
            f"\nFAILED: selected {len(selected)} tabular columns but the published "
            f"config declares input_channels_cont: {EXPECTED_CONTINUOUS}. The release "
            "staged here is not the one that config was written against, or the "
            "prefixes in the contract are wrong. Not writing a table on that basis.",
            file=sys.stderr,
        )
        return 1
    if KEY_SOURCE not in header:
        print(f"FAILED: {KEY_SOURCE} absent from {source.name}", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    frame = pd.read_csv(source, usecols=[KEY_SOURCE, *selected], low_memory=False)
    frame = frame.rename(columns={KEY_SOURCE: KEY})

    cohort = pd.read_parquet(derived / "cohort_mdsed.parquet", columns=[KEY])
    frame = cohort.merge(frame, on=KEY, how="left")
    if len(frame) != len(cohort):
        print(f"FAILED: join produced {len(frame)} rows against a cohort of "
              f"{len(cohort)}; {KEY} is not unique in the release", file=sys.stderr)
        return 1

    unmatched = int(frame[selected].isna().all(axis=1).sum())
    out = assert_outside_repo(derived / "features_tabular.parquet", "tabular features")
    frame.to_parquet(out, index=False)

    missing = frame[selected].isna().mean()
    meta = {
        "cohort": "mdsed",
        "block": "tabular",
        "status": "passthrough from the release, unmodified; see the pass-through rule for row 5",
        "source_file": source.name,
        "prefixes": list(prefixes),
        "n_columns": len(selected),
        "n_columns_by_prefix": by_prefix,
        "expected_continuous": EXPECTED_CONTINUOUS,
        "n_rows": int(len(frame)),
        "rows_with_no_tabular_data": unmatched,
        "missingness": {
            "mean": float(missing.mean()),
            "median": float(missing.median()),
            "max": float(missing.max()),
            "columns_over_90pct_missing": int((missing > 0.9).sum()),
            "columns_fully_missing": int((missing == 1.0).sum()),
        },
        "imputation": "none. Missingness is left for the model, per the pass-through rule for row 5.",
        "hash": _sha256(out),
        "contract_sha256": contract.sha256,
    }
    (derived / "features_tabular_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"\nwrote {out.name}: {len(frame):,} rows x {len(selected)} columns")
    print(f"  rows with no tabular data at all : {unmatched:,}")
    print(f"  mean missingness across columns  : {missing.mean():.1%}")
    print(f"  columns over 90% missing         : {int((missing > 0.9).sum())}")
    print(f"  sha256 {meta['hash'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

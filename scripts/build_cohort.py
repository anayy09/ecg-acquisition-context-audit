#!/usr/bin/env python
"""Build the MDS-ED cohort and C_SUPERSET, and pin the published split.

MDS-ED is the primary source: one row per ECG, carrying acquisition time, ED
arrival, the 90-minute bound, triage acuity, the fold assignment, and all 15
deterioration labels. Deriving the audited arms from the audited artifact keeps
the cohort and split aligned with the benchmark by construction.

C_SUPERSET is every adult ED stay in MIMIC-IV-ED, flagged by whether it appears
in MDS-ED. That flag is exactly `ecg_in_first_90min`, and it is the thing MDS-ED
cannot see about itself.

Outputs go to ``$ECG_DATA_ROOT/derived/`` and never into the repo.

    python scripts/build_cohort.py
    python scripts/build_cohort.py --reproduce-check
    python scripts/build_cohort.py --list-labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import load_contract  # noqa: E402
from loaders import load_source, resolve_path  # noqa: E402
from paths import derived_root, fixtures_root  # noqa: E402

ADULT_MIN_AGE = 18

#: MDS-ED column -> our canonical name.
RENAME = {
    "general_subject_id": "subject_id",
    "general_ed_stay_id": "stay_id",
    "general_ed_hadm_id": "hadm_id",
    "general_study_id": "study_id",
    "general_ecg_time": "index_time",
    "general_intime": "intime",
    "general_outtime": "outtime",
    "general_90min": "window_close",
    "general_ecg_no_within_stay": "ecg_no_within_stay",
    "general_strat_fold": "strat_fold",
    "demographics_age": "age",
    "demographics_gender": "sex",
    "vitals_acuity": "acuity",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


PUBLISHED_PATH = Path(__file__).resolve().parent.parent / "configs" / "published_reference.yaml"


def load_published() -> dict | None:
    """Published MDS-ED figures, read from the paper. None if not yet recorded."""
    import yaml

    if not PUBLISHED_PATH.is_file():
        return None
    return yaml.safe_load(PUBLISHED_PATH.read_text(encoding="utf-8"))


def reproduce_check(built: dict) -> pd.DataFrame:
    """Local versus published cohort counts and positives (the protocol task 1, the corresponding check).

    The published column comes from `configs/published_reference.yaml`, which was
    read from the paper rather than recalled. A gap larger than rounding is a
    finding and belongs in RESULTS-LOG.md, not in a footnote.
    """
    mdsed = built["mdsed"]
    published = load_published() or {}
    cohort = published.get("cohort") or {}
    positives = published.get("label_positives") or {}
    sentinel = built["sentinel"]

    rows: list[dict] = [
        {"quantity": "n_samples", "local": int(len(mdsed)),
         "published": cohort.get("n_samples")},
        {"quantity": "n_patients", "local": int(mdsed["subject_id"].nunique()),
         "published": cohort.get("n_patients")},
        {"quantity": "n_visits", "local": int(mdsed["stay_id"].nunique()),
         "published": cohort.get("n_visits")},
    ]
    for label, published_n in positives.items():
        if label in mdsed.columns:
            valid = mdsed[label] != sentinel
            rows.append({
                "quantity": f"positives[{label}]",
                "local": int((mdsed.loc[valid, label] == 1).sum()),
                "published": published_n,
            })

    frame = pd.DataFrame(rows)
    frame["gap"] = [
        None if r["published"] is None else int(r["local"]) - int(r["published"])
        for _, r in frame.iterrows()
    ]
    frame["pct"] = [
        None if not r["published"] else
        round(100.0 * (int(r["local"]) - int(r["published"])) / int(r["published"]), 4)
        for _, r in frame.iterrows()
    ]
    return frame


def label_columns(contract) -> list[str]:
    spec = contract.table_spec("mds_ed.cohort")
    return list((spec.get("label_columns") or {}).keys())


def split_of(fold: pd.Series, contract) -> pd.Series:
    assignment = contract.raw["split"]["assignment"]
    mapping: dict[int, str] = {}
    for name, folds in assignment.items():
        for value in folds:
            mapping[int(value)] = name
    return fold.map(mapping)


def load_mdsed(root: Path, contract) -> pd.DataFrame:
    """Read only the columns we need out of the 1,936-column release."""
    path = resolve_path("mds_ed.cohort", root, contract)
    spec = contract.table_spec("mds_ed.cohort")
    wanted = list(spec["columns"]) + label_columns(contract)
    dates = [
        name
        for name, column in spec["columns"].items()
        if str(column.get("dtype", "")).startswith("datetime")
    ]
    frame = pd.read_csv(path, usecols=wanted, parse_dates=dates, low_memory=False)

    missing = [c for c in wanted if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{path} is missing contract columns {missing}. Correct "
            "configs/source_schema.yaml rather than working around it."
        )
    return frame


def build(root: Path, contract) -> dict:
    sentinel = contract.raw.get("label_missing_sentinel", -999)
    labels = label_columns(contract)

    raw = load_mdsed(root, contract)
    mdsed = raw.rename(columns=RENAME)
    mdsed["stay_id"] = mdsed["stay_id"].astype("Int64")
    mdsed["hadm_id"] = mdsed["hadm_id"].astype("Int64")
    mdsed["split"] = split_of(mdsed["strat_fold"], contract)

    # Sanity: every acquisition must sit inside the window the release claims.
    offset = (mdsed["index_time"] - mdsed["intime"]).dt.total_seconds() / 60.0
    if float(offset.min()) < 0 or float(offset.max()) > 90.0:
        raise RuntimeError(
            f"acquisition offsets fall outside [0, 90] minutes "
            f"(min {offset.min():.1f}, max {offset.max():.1f}); the 90-minute "
            "conditioning assumed in the dataset request does not hold for this release"
        )

    # ---- C_SUPERSET ------------------------------------------------------
    edstays = load_source("mimic_iv_ed.edstays", root, contract, required=False)
    patients = load_source("mimic_iv_hosp.patients", root, contract, required=False)
    superset = None
    if edstays is not None and patients is not None:
        superset = edstays.merge(
            patients[["subject_id", "anchor_age", "anchor_year"]],
            on="subject_id",
            how="left",
        )
        superset["age"] = superset["anchor_age"] + (
            superset["intime"].dt.year - superset["anchor_year"]
        )
        superset = superset[superset["age"] >= ADULT_MIN_AGE].copy()
        superset["window_close"] = superset["intime"] + pd.Timedelta(minutes=90)
        # The flag is definitionally "is this stay in MDS-ED", which is exactly
        # what MDS-ED conditions on. Deriving it any other way would risk
        # disagreeing with the benchmark about its own inclusion rule.
        in_mdsed = set(mdsed["stay_id"].dropna().astype("int64").tolist())
        superset["ecg_in_first_90min"] = superset["stay_id"].isin(in_mdsed)
        superset = superset[
            [
                "subject_id",
                "stay_id",
                "hadm_id",
                "intime",
                "outtime",
                "window_close",
                "age",
                "arrival_transport",
                "disposition",
                "ecg_in_first_90min",
            ]
        ]

    split = mdsed[["subject_id", "study_id", "strat_fold", "split"]].copy()

    return {
        "mdsed": mdsed,
        "superset": superset,
        "split": split,
        "labels": labels,
        "sentinel": sentinel,
    }


def label_table(mdsed: pd.DataFrame, labels: list[str], sentinel: int) -> pd.DataFrame:
    rows = []
    for label in labels:
        valid = mdsed[label] != sentinel
        rows.append(
            {
                "label": label,
                "n_valid": int(valid.sum()),
                "n_missing": int((~valid).sum()),
                "n_positive": int((mdsed.loc[valid, label] == 1).sum()),
                "prevalence": round(float(mdsed.loc[valid, label].mean()), 5),
                "n_patients": int(mdsed.loc[valid, "subject_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def check_split_disjoint(mdsed: pd.DataFrame) -> list[str]:
    counts = mdsed.groupby("subject_id")["split"].nunique()
    offenders = int((counts > 1).sum())
    if offenders:
        return [
            f"{offenders} patient(s) appear in more than one split; the split unit "
            "and the resampling unit must agree"
        ]
    return []


def write_outputs(built: dict, out: Path, contract) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    pairs = [("cohort_mdsed", built["mdsed"]), ("split_published", built["split"])]
    if built["superset"] is not None:
        pairs.append(("cohort_superset", built["superset"]))

    for name, frame in pairs:
        path = out / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        written[name] = _sha256(path)

    labels = label_table(built["mdsed"], built["labels"], built["sentinel"])
    labels.to_csv(out / "label_prevalence.csv", index=False)

    (out / "cohort_meta.json").write_text(
        json.dumps(
            {
                "contract_sha256": contract.sha256,
                "labels_provisional": False,
                "label_missing_sentinel": built["sentinel"],
                "split": contract.raw["split"],
                "n_ecgs": int(len(built["mdsed"])),
                "n_patients": int(built["mdsed"]["subject_id"].nunique()),
                "n_stays": int(built["mdsed"]["stay_id"].nunique()),
                "n_superset_stays": (
                    int(len(built["superset"])) if built["superset"] is not None else None
                ),
                "split_counts": built["mdsed"]["split"].value_counts().to_dict(),
                "hashes": written,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument("--reproduce-check", action="store_true")
    parser.add_argument("--list-labels", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    contract = load_contract()
    if args.fixtures:
        root = fixtures_root()
    else:
        from paths import data_root

        root = data_root(args.data_root)

    built = build(root, contract)
    mdsed = built["mdsed"]

    if args.list_labels:
        table = label_table(mdsed, built["labels"], built["sentinel"])
        print(table.to_string(index=False))
        print(
            f"\n{len(built['labels'])} deterioration labels. "
            f"{built['sentinel']} marks 'not defined for this row' and is dropped "
            "per label, never globally."
        )
        return 0

    problems = check_split_disjoint(mdsed)

    if args.reproduce_check:
        print("Cohort reproduction, local versus published (the protocol task 1, the corresponding check)\n")
        frame = reproduce_check(built)
        print(frame.to_string(index=False))
        published = load_published()
        if published is None:
            print("\nNo published reference recorded; there is nothing to compare "
                  "against. Record configs/published_reference.yaml first.")
        else:
            print(f"\nPublished figures read from {published['source_key']} on "
                  f"{published['verified_on']}.")
            print("Split: " + ", ".join(
                f"{k}={v}" for k, v in published["split"].items()
                if k in ("ratio", "val_folds", "test_folds")
            ))
        print("\nLabel prevalence, all 15 targets:")
        print(label_table(mdsed, built["labels"], built["sentinel"]).to_string(index=False))
        print("\nSplit sizes (ECGs):")
        print(mdsed["split"].value_counts().to_string())
        for problem in problems:
            print(f"\nSPLIT PROBLEM: {problem}")
        return 1 if problems else 0

    if problems:
        for problem in problems:
            print(f"SPLIT PROBLEM: {problem}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else (
        (root / "derived") if args.fixtures else derived_root()
    )
    hashes = write_outputs(built, out, contract)

    if not args.quiet:
        print(f"derived root: {out}")
        print(
            f"  MDS-ED cohort  {len(mdsed):>8,} ECGs  "
            f"{mdsed['subject_id'].nunique():>7,} patients  "
            f"{mdsed['stay_id'].nunique():>7,} stays"
        )
        if built["superset"] is not None:
            sup = built["superset"]
            print(
                f"  C_SUPERSET     {len(sup):>8,} stays  "
                f"{sup['subject_id'].nunique():>7,} patients  "
                f"ecg_in_first_90min share {sup['ecg_in_first_90min'].mean():.3f}"
            )
        else:
            print("  C_SUPERSET     not built (MIMIC-IV-ED not staged)")
        counts = mdsed["split"].value_counts()
        print(f"  split (ECGs)   " + "  ".join(f"{k}={v:,}" for k, v in counts.items()))
        for name, digest in hashes.items():
            print(f"  sha256 {name:<18} {digest[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

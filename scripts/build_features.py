#!/usr/bin/env python
"""Build the acquisition-context feature table in two strictly separated blocks.

Every feature column ``X`` is emitted with a companion ``available_at__X``
column holding the per-row timestamp at which that value became knowable. The
leakage guard in ``leakage_test.py`` reads those columns; nothing else decides
whether a feature is safe. That is the mechanism the two-block split promises, made literal.

    python scripts/build_features.py
    python scripts/build_features.py --probe-availability
    python scripts/build_features.py --inject-leak triage_acuity   # test only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import load_contract  # noqa: E402
from loaders import load_source  # noqa: E402
from paths import derived_root, fixtures_root  # noqa: E402

#: Maps a contract availability rule to the cohort column holding its timestamp.
AVAILABILITY_COLUMN = {
    "ed_arrival_time": "intime",
    "index_time": "index_time",
    "window_close": "window_close",
}

OVERREAD_MARKERS = ("cardiolog", "confirmed by", "over-read", "overread", "reviewed by")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prior_ecg(cohort: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Latest ECG strictly before this acquisition, via a backward as-of join.

    An as-of join rather than a full merge: the full merge is quadratic in a
    patient's ECG count, and MIMIC-IV-ECG has patients with hundreds.
    """
    left = (
        cohort[["study_id", "subject_id", "index_time"]]
        .sort_values("index_time")
        .reset_index(drop=True)
    )
    right = (
        records[["subject_id", "ecg_time"]]
        .dropna(subset=["ecg_time"])
        .sort_values("ecg_time")
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        left,
        right.rename(columns={"ecg_time": "prior_ecg_time"}),
        left_on="index_time",
        right_on="prior_ecg_time",
        by="subject_id",
        direction="backward",
        allow_exact_matches=False,
    )
    merged["prior_ecg_exists"] = merged["prior_ecg_time"].notna()
    merged["days_since_prior_ecg"] = (
        merged["index_time"] - merged["prior_ecg_time"]
    ).dt.total_seconds() / 86400.0
    return merged[["study_id", "prior_ecg_exists", "days_since_prior_ecg"]]


def _prior_visits(cohort: pd.DataFrame, edstays: pd.DataFrame) -> pd.DataFrame:
    """Count of this patient's earlier ED arrivals, keyed by stay."""
    ordered = edstays[["subject_id", "stay_id", "intime"]].sort_values(
        ["subject_id", "intime"]
    )
    ordered["prior_ed_visit_count"] = ordered.groupby("subject_id").cumcount()
    return (
        cohort[["study_id", "stay_id"]]
        .merge(
            ordered[["stay_id", "prior_ed_visit_count"]].astype({"stay_id": "Int64"}),
            on="stay_id",
            how="left",
        )[["study_id", "prior_ed_visit_count"]]
    )


def _window_stats(cohort: pd.DataFrame) -> pd.DataFrame:
    """ECG count and mean inter-ECG gap inside the 90-minute window, per stay.

    MDS-ED contains exactly the acquisitions inside that window, so the stay's
    own rows are the window. Both columns are ACQ_WINDOW: they count and space
    acquisitions that in general happen after the index one.
    """
    per_stay = cohort.groupby("stay_id", dropna=False).agg(
        n_ecgs_first_90min=("study_id", "size")
    )
    ordered = cohort[["stay_id", "index_time"]].sort_values(["stay_id", "index_time"])
    ordered["gap"] = (
        ordered.groupby("stay_id")["index_time"].diff().dt.total_seconds() / 60.0
    )
    gaps = ordered.groupby("stay_id")["gap"].mean().rename("inter_ecg_interval_minutes")
    joined = per_stay.join(gaps).reset_index()
    return cohort[["study_id", "stay_id"]].merge(joined, on="stay_id", how="left")[
        ["study_id", "n_ecgs_first_90min", "inter_ecg_interval_minutes"]
    ]


def _care_unit(cohort: pd.DataFrame, icustays: pd.DataFrame) -> pd.DataFrame:
    """ICU unit already covering the acquisition, else the ED. The leakage-suspect classification: suspect."""
    icu = (
        icustays[["hadm_id", "intime", "first_careunit"]]
        .rename(columns={"intime": "icu_intime"})
        .astype({"hadm_id": "Int64"})
    )
    merged = cohort[["study_id", "hadm_id", "index_time"]].merge(
        icu, on="hadm_id", how="left"
    )
    covering = merged[
        merged["icu_intime"].notna() & (merged["icu_intime"] <= merged["index_time"])
    ]
    latest = (
        covering.sort_values(["study_id", "icu_intime"])
        .groupby("study_id", as_index=False)
        .last()[["study_id", "first_careunit"]]
    )
    out = cohort[["study_id"]].merge(latest, on="study_id", how="left")
    out["care_unit_at_acquisition"] = out["first_careunit"].fillna("Emergency Department")
    return out[["study_id", "care_unit_at_acquisition"]]


def probe_overread(machine: pd.DataFrame | None, contract) -> dict:
    """Decide whether a cardiologist over-read flag is derivable."""
    spec = contract.table_spec("mimic_iv_ecg.machine_measurements")
    candidates = spec.get("candidate_overread_columns") or []
    report: dict = {
        "table_present": machine is not None,
        "candidates_declared": list(candidates),
        "candidates_present": [],
        "chosen": None,
        "derivable": False,
        "degenerate": {},
        "reason": "",
    }
    if machine is None:
        report["reason"] = "machine_measurements is not staged or failed validation"
        return report

    present = [c for c in candidates if c in machine.columns]
    report["candidates_present"] = present
    for column in present:
        values = machine[column].astype("string").fillna("")
        hit = values.str.lower().str.contains("|".join(OVERREAD_MARKERS), regex=True)
        rate = float(hit.mean())
        # A flag that is almost always or almost never set carries no signal and
        # would be a fake feature. Treat degeneracy as "not derivable".
        if 0.01 <= rate <= 0.99:
            report.update(
                chosen=column,
                derivable=True,
                positive_rate=round(rate, 4),
                reason=f"column {column!r} carries an over-read marker in {rate:.1%} of rows",
            )
            return report
        report["degenerate"][column] = round(rate, 6)

    report["reason"] = (
        "no declared candidate column carries a non-degenerate over-read marker; "
        "MIMIC-IV-ECG ships machine-generated statements and no separate "
        "cardiologist over-read flag was found"
    )
    return report


def build(root: Path, contract, cohort_name: str = "mdsed") -> dict:
    derived = (root / "derived") if (root / "derived").is_dir() else derived_root()
    cohort_path = derived / f"cohort_{cohort_name}.parquet"
    if not cohort_path.is_file():
        raise FileNotFoundError(
            f"{cohort_path} not found. Run scripts/build_cohort.py first."
        )
    frame = pd.read_parquet(cohort_path)

    records = load_source("mimic_iv_ecg.record_list", root, contract, required=False)
    edstays = load_source("mimic_iv_ed.edstays", root, contract, required=False)
    icustays = load_source("mimic_iv_icu.icustays", root, contract, required=False)
    machine = load_source(
        "mimic_iv_ecg.machine_measurements", root, contract, required=False
    )

    dropped: list[str] = []

    # ---- ACQ_PRE, from MDS-ED itself -------------------------------------
    frame["arrival_to_acquisition_minutes"] = (
        frame["index_time"] - frame["intime"]
    ).dt.total_seconds() / 60.0
    frame["acquisition_hour_of_day"] = frame["index_time"].dt.hour.astype("int64")
    frame["acquisition_day_of_week"] = (
        frame["index_time"].dt.isocalendar().day.astype("int64")
    )
    frame["acquisition_month"] = frame["index_time"].dt.month.astype("int64")
    frame["triage_acuity"] = frame["acuity"].astype("float64")
    # ecg_no_within_stay and age/sex already arrive on the cohort.

    # ---- ACQ_PRE, needing the parent tables ------------------------------
    if records is not None:
        frame = frame.merge(_prior_ecg(frame, records), on="study_id", how="left")
    else:
        dropped += ["prior_ecg_exists", "days_since_prior_ecg"]
    if edstays is not None:
        frame = frame.merge(_prior_visits(frame, edstays), on="study_id", how="left")
    else:
        dropped.append("prior_ed_visit_count")

    # ---- ACQ_WINDOW -------------------------------------------------------
    frame = frame.merge(_window_stats(frame), on="study_id", how="left")
    if icustays is not None:
        frame = frame.merge(_care_unit(frame, icustays), on="study_id", how="left")
    else:
        dropped.append("care_unit_at_acquisition")

    overread = probe_overread(machine, contract)
    if overread["derivable"]:
        column = overread["chosen"]
        flags = machine[["study_id", column]].copy()
        values = flags[column].astype("string").fillna("")
        flags["cardiologist_overread_exists"] = values.str.lower().str.contains(
            "|".join(OVERREAD_MARKERS), regex=True
        )
        frame = frame.merge(
            flags[["study_id", "cardiologist_overread_exists"]],
            on="study_id",
            how="left",
        )
        frame["cardiologist_overread_exists"] = frame[
            "cardiologist_overread_exists"
        ].fillna(False)
    else:
        dropped.append("cardiologist_overread_exists")

    # ---- availability stamps ---------------------------------------------
    emitted: dict[str, list[str]] = {}
    for block in ("demographics", "acq_pre", "acq_window"):
        names = []
        for derivation in contract.block(block):
            if derivation.name in dropped:
                continue
            if derivation.name not in frame.columns:
                raise RuntimeError(
                    f"derivation {derivation.name!r} is declared in the contract "
                    "but was not produced by build_features.py"
                )
            stamp = AVAILABILITY_COLUMN[derivation.available_at]
            frame[f"available_at__{derivation.name}"] = frame[stamp]
            names.append(derivation.name)
        emitted[block] = names

    return {
        "frame": frame,
        "emitted": emitted,
        "dropped": dropped,
        "overread": overread,
        "cohort_name": cohort_name,
    }


def inject_leak(built: dict, feature: str) -> dict:
    """TEST ONLY. Backdate an ACQ_PRE feature's availability past the index time.

    The positive control for the leakage guard (the project's design notes pre-declared rule 7). A guard that never
    fires has not been shown to work.
    """
    frame = built["frame"]
    column = f"available_at__{feature}"
    if feature not in built["emitted"]["acq_pre"]:
        raise KeyError(f"{feature!r} is not an emitted ACQ_PRE feature")
    poisoned = frame.copy()
    mask = np.zeros(len(poisoned), dtype=bool)
    mask[:: max(1, len(poisoned) // 20)] = True
    poisoned.loc[mask, column] = poisoned.loc[mask, "window_close"]
    built = dict(built)
    built["frame"] = poisoned
    built["poisoned_feature"] = feature
    built["poisoned_rows"] = int(mask.sum())
    return built


def write_outputs(built: dict, out: Path, contract) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    frame = built["frame"]
    keys = [
        "subject_id",
        "stay_id",
        "study_id",
        "intime",
        "index_time",
        "window_close",
        "split",
    ]
    keys = [k for k in keys if k in frame.columns]

    written: dict[str, str] = {}
    for name, blocks in (
        ("features_demo", ["demographics"]),
        ("features_acq", ["acq_pre", "acq_window"]),
    ):
        columns = list(keys)
        for block in blocks:
            for feature in built["emitted"][block]:
                columns += [feature, f"available_at__{feature}"]
        path = out / f"{name}.parquet"
        frame[columns].to_parquet(path, index=False)
        written[name] = _sha256(path)

    (out / "features_meta.json").write_text(
        json.dumps(
            {
                "cohort": built["cohort_name"],
                "contract_sha256": contract.sha256,
                "blocks": built["emitted"],
                "dropped": built["dropped"],
                "overread_probe": built["overread"],
                "hashes": written,
                "poisoned": built.get("poisoned_feature"),
                "n_rows": int(len(frame)),
                "n_patients": int(frame["subject_id"].nunique()),
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
    parser.add_argument("--cohort", default="mdsed", choices=["mdsed", "superset"])
    parser.add_argument("--probe-availability", action="store_true")
    parser.add_argument("--inject-leak", default=None, metavar="FEATURE")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    contract = load_contract()
    if args.fixtures:
        root = fixtures_root()
    else:
        from paths import data_root

        root = data_root(args.data_root)

    if args.probe_availability:
        machine = load_source(
            "mimic_iv_ecg.machine_measurements", root, contract, required=False
        )
        report = probe_overread(machine, contract)
        print(json.dumps(report, indent=2))
        print(
            "\nD-019 closes on this result. `derivable: false` means "
            "cardiologist_overread_exists is dropped automatically; record the drop."
        )
        return 0

    built = build(root, contract, cohort_name=args.cohort)
    if args.inject_leak:
        built = inject_leak(built, args.inject_leak)

    out = Path(args.out) if args.out else (
        (root / "derived") if args.fixtures else derived_root()
    )
    hashes = write_outputs(built, out, contract)

    if not args.quiet:
        print(f"derived root: {out}")
        print(f"  rows {len(built['frame']):,}   patients "
              f"{built['frame']['subject_id'].nunique():,}")
        for block, names in built["emitted"].items():
            print(f"  {block:<14} {len(names):>2}: {', '.join(names)}")
        if built["dropped"]:
            print(f"  dropped        {', '.join(built['dropped'])}")
        print(f"  over-read probe: {built['overread']['reason']}")
        for name, digest in hashes.items():
            print(f"  sha256 {name:<16} {digest[:16]}")
        if built.get("poisoned_feature"):
            print(f"  POISONED (test only): {built['poisoned_feature']} on "
                  f"{built['poisoned_rows']} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

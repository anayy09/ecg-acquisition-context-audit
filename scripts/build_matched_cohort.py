#!/usr/bin/env python
"""Build row 8's availability-matched strata, its balance table, and its diagnostic.

The within-stratum specification fixes the specification, before any row 8 number exists: coarsened exact
matching on the ACQ_PRE block, and a discrimination statistic restricted to
positive-negative pairs drawn from the SAME stratum, so no compared pair differs
in acquisition context.

    arrival-to-acquisition delay   <=15, 15-30, 30-60, >60 minutes
    acquisition hour of day        00-06, 06-12, 12-18, 18-24
    day of week                    weekday / weekend
    triage acuity                  as recorded 1-5, missing as its own level

Why matching cannot instead be to the superset. `cohort_superset.parquet` holds
425,011 adult ED stays but no deterioration label, and its ECG-absent stays have
no waveform, so it cannot supply an evaluation row to row 8 under any
specification. Its role here is the diagnostic at the bottom: how strongly does
pre-acquisition context predict whether an early ECG happens at all. That is
reported as a diagnostic and is never row 8.

Balance is reported the way this design makes meaningful. Within a stratum the
two arms see the same rows, so an SMD between arms would be identically zero and
would prove nothing. What the design claims is that the POSITIVES and NEGATIVES
being compared no longer differ in acquisition context, so that is what is
measured: the standardised mean difference between outcome-positive and
outcome-negative records, before and after the pair weighting the within-stratum
statistic implies. On the coarsened variables it must fall to zero exactly. On
the underlying continuous values it does not, because coarsening leaves residual
imbalance inside a bin, and that residual is reported rather than hidden.

    python scripts/build_matched_cohort.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import load_base_config  # noqa: E402
from paths import REPO_ROOT, assert_outside_repo, derived_root  # noqa: E402

RESULTS = REPO_ROOT / "results"

#: The within-stratum specification's coarsening, in one place. Changing any of it is a new decision entry,
#: not an edit: the specification was fixed before any row 8 number existed.
DELAY_EDGES = [-np.inf, 15, 30, 60, np.inf]
DELAY_LABELS = ["<=15", "15-30", "30-60", ">60"]
HOUR_EDGES = [-np.inf, 5.5, 11.5, 17.5, np.inf]
HOUR_LABELS = ["00-06", "06-12", "12-18", "18-24"]

MATCHING_VARIABLES = {
    "arrival_to_acquisition_minutes": "delay_bucket",
    "acquisition_hour_of_day": "hour_bucket",
    "acquisition_day_of_week": "is_weekend",
    "triage_acuity": "acuity_level",
}

#: Every pre-acquisition feature the confirmatory arm is given, matched or not.
#: Balance is reported over all nine rather than over the four that are held
#: fixed, because "matched on acquisition context" is a claim about the block and
#: a table covering only the matched part cannot be read against it.
ALL_ACQ_PRE = [
    "arrival_to_acquisition_minutes",
    "acquisition_hour_of_day",
    "acquisition_day_of_week",
    "acquisition_month",
    "triage_acuity",
    "ecg_no_within_stay",
    "prior_ecg_exists",
    "days_since_prior_ecg",
    "prior_ed_visit_count",
]

#: Declared sensitivities. Each names the variables it strata on; anything not
#: listed here cannot be run, so a sensitivity is a decision rather than a flag
#: somebody passed. Every one is EXPLORATORY: the declared specification is
#: The within-stratum specification's and is what the paper's row 8 reports.
SENSITIVITIES = {
    "no_weekday": {
        "variables": ["arrival_to_acquisition_minutes",
                      "acquisition_hour_of_day", "triage_acuity"],
        "why": ("the calendar-feature finding: the recorded weekday is a de-identification artefact, "
                "so the fourth stratum variable splits on noise. This matches "
                "on the three that carry information and buys back the "
                "effective pairs the split costs."),
    },
}


def coarsen(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["delay_bucket"] = pd.cut(
        out["arrival_to_acquisition_minutes"], DELAY_EDGES, labels=DELAY_LABELS
    ).astype(str)
    out["hour_bucket"] = pd.cut(
        out["acquisition_hour_of_day"], HOUR_EDGES, labels=HOUR_LABELS
    ).astype(str)
    out["is_weekend"] = (out["acquisition_day_of_week"] >= 5).astype(int).astype(str)
    # Missing acuity is its own level, not an imputation and not a drop: 170 of
    # 6,439 test records carry no acuity, and assigning them a value would invent
    # the very context this design holds fixed.
    out["acuity_level"] = out["triage_acuity"].fillna(-1).astype(float).astype(str)
    out["stratum"] = stratum_key(out, list(MATCHING_VARIABLES))
    return out


def stratum_key(frame: pd.DataFrame, variables: list[str]) -> pd.Series:
    """The stratum label, over whichever variables are being held fixed."""
    columns = [MATCHING_VARIABLES[v] for v in variables]
    key = frame[columns[0]].astype(str)
    for column in columns[1:]:
        key = key + "|" + frame[column].astype(str)
    return key


def smd(values: np.ndarray, positive: np.ndarray, weights: np.ndarray) -> float:
    """Weighted standardised mean difference between positives and negatives."""
    pos_w, neg_w = weights[positive], weights[~positive]
    if pos_w.sum() <= 0 or neg_w.sum() <= 0:
        return float("nan")
    pos_v, neg_v = values[positive], values[~positive]
    mean_p = float(np.average(pos_v, weights=pos_w))
    mean_n = float(np.average(neg_v, weights=neg_w))
    var_p = float(np.average((pos_v - mean_p) ** 2, weights=pos_w))
    var_n = float(np.average((neg_v - mean_n) ** 2, weights=neg_w))
    pooled = np.sqrt((var_p + var_n) / 2.0)
    if pooled <= 0:
        return 0.0 if abs(mean_p - mean_n) < 1e-12 else float("inf")
    return (mean_p - mean_n) / pooled


def pair_weights(stratum: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """Weights implied by counting only within-stratum positive-negative pairs.

    In a stratum with P positives and N negatives the statistic counts P*N pairs,
    so each positive there enters N of them and each negative P. Reweighting the
    marginal distributions this way is what makes the balance table describe the
    comparison the statistic actually performs rather than the raw cohort.
    """
    weights = np.zeros(len(stratum), dtype=float)
    frame = pd.DataFrame({"s": stratum, "p": positive})
    counts = frame.groupby("s")["p"].agg(["sum", "count"])
    n_pos = counts["sum"].to_dict()
    n_neg = (counts["count"] - counts["sum"]).to_dict()
    for i, (s, p) in enumerate(zip(stratum, positive)):
        weights[i] = n_neg.get(s, 0) if p else n_pos.get(s, 0)
    return weights


def propensity_diagnostic(derived: Path) -> dict:
    """How strongly does pre-acquisition context predict early-ECG availability?

    Fitted on C_SUPERSET, reported as a diagnostic, and never used to select or
    weight a row 8 evaluation row. It answers the question the superset CAN
    answer — MDS-ED conditions on an early ECG, so presence-versus-absence is not
    estimable inside it — without pretending row 8 is evaluated there.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    superset = pd.read_parquet(derived / "cohort_superset.parquet")
    frame = pd.DataFrame({
        "age": superset["age"].astype(float),
        "arrival_transport": superset["arrival_transport"].astype("category"),
        "arrival_hour": superset["intime"].dt.hour.astype(float),
        "arrival_weekday": superset["intime"].dt.dayofweek.astype(float),
        "arrival_month": superset["intime"].dt.month.astype(float),
    })
    target = superset["ecg_in_first_90min"].astype(int).to_numpy()

    rng = np.random.default_rng(0)
    holdout = rng.random(len(frame)) < 0.2
    model = lgb.LGBMClassifier(objective="binary", n_estimators=300, num_leaves=31,
                               learning_rate=0.05, random_state=0, n_jobs=-1,
                               verbose=-1)
    model.fit(frame[~holdout], target[~holdout])
    scores = model.predict_proba(frame[holdout])[:, 1]
    return {
        "n_stays": int(len(superset)),
        "n_patients": int(superset["subject_id"].nunique()),
        "early_ecg_rate": float(target.mean()),
        "auroc": float(roc_auc_score(target[holdout], scores)),
        "n_holdout": int(holdout.sum()),
        "covariates": list(frame.columns),
        "note": "diagnostic only. Fitted on C_SUPERSET, never used to select or "
                "weight a row 8 evaluation row.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--skip-propensity", action="store_true")
    parser.add_argument(
        "--sensitivity", choices=sorted(SENSITIVITIES), default=None,
        help="run a declared EXPLORATORY variant instead of the within-stratum specification's "
             "specification. Writes to its own files and never touches the "
             "declared artifacts")
    args = parser.parse_args()

    variant = SENSITIVITIES[args.sensitivity] if args.sensitivity else None
    variables = variant["variables"] if variant else list(MATCHING_VARIABLES)
    prefix = f"p4_sensitivity_{args.sensitivity}" if variant else "p4"
    if variant:
        print(f"EXPLORATORY sensitivity {args.sensitivity!r}: "
              f"strata on {', '.join(variables)}")
        print(f"  {variant['why']}")

    base = load_base_config()
    labels = list(base["labels"])
    derived = derived_root(args.data_root)
    RESULTS.mkdir(parents=True, exist_ok=True)

    acq = pd.read_parquet(derived / "features_acq.parquet")
    cohort = pd.read_parquet(derived / "cohort_mdsed.parquet")
    test = acq[acq["split"] == "test"].copy()
    test = coarsen(test)
    if variant:
        # Re-key the strata over the variant's variables. Everything downstream
        # reads `stratum`, so the rest of this script is unchanged and the two
        # analyses cannot diverge in anything except what they hold fixed.
        test["stratum"] = stratum_key(test, variables)

    sentinel = int(base.get("label_missing_sentinel", -999))
    label_frame = cohort[["study_id", *labels]]
    test = test.merge(label_frame, on="study_id", how="left")

    print(f"test fold: {len(test):,} records, "
          f"{test['subject_id'].nunique():,} patients")
    sizes = test["stratum"].value_counts()
    print(f"strata: {len(sizes)} non-empty, median size {sizes.median():.0f}, "
          f"max {sizes.max()}, {int(sizes[sizes >= 10].sum()):,} records in strata "
          f"of at least 10")

    # ---- row-level assignment goes outside the repo, the containment rule ----------------
    out_path = assert_outside_repo(
        derived / (f"matched_strata_{args.sensitivity}.parquet" if variant
                   else "matched_strata.parquet"),
        "matched strata")
    test[["study_id", "subject_id", "stratum", "delay_bucket", "hour_bucket",
          "is_weekend", "acuity_level"]].to_parquet(out_path, index=False)

    # ---- per-stratum aggregate, and per-label informativeness -------------
    stratum_rows = []
    for stratum, group in test.groupby("stratum"):
        row = {
            "stratum": stratum,
            "n_records": int(len(group)),
            "n_patients": int(group["subject_id"].nunique()),
        }
        for label in labels:
            defined = group[group[label] != sentinel]
            n_pos = int((defined[label] == 1).sum())
            n_neg = int((defined[label] == 0).sum())
            row[f"n_pos__{label}"] = n_pos
            row[f"n_neg__{label}"] = n_neg
            row[f"n_pairs__{label}"] = n_pos * n_neg
        stratum_rows.append(row)
    strata = pd.DataFrame(stratum_rows).sort_values("n_records", ascending=False)
    strata.to_csv(RESULTS / f"{prefix}_strata.csv", index=False)

    # ---- effective sample: what actually bounds every row 8 claim ---------
    effective_rows = []
    for label in labels:
        informative = strata[strata[f"n_pairs__{label}"] > 0]["stratum"]
        keep = test[test["stratum"].isin(informative) & (test[label] != sentinel)]
        defined = test[test[label] != sentinel]
        effective_rows.append({
            "label": label,
            "n_strata_total": int(len(strata)),
            "n_strata_informative": int(len(informative)),
            "n_records_defined": int(len(defined)),
            "n_records_informative": int(len(keep)),
            "n_patients_defined": int(defined["subject_id"].nunique()),
            "n_patients_effective": int(keep["subject_id"].nunique()),
            "n_within_stratum_pairs": int(strata[f"n_pairs__{label}"].sum()),
            "n_all_pairs": int((defined[label] == 1).sum()
                               * (defined[label] == 0).sum()),
        })
        effective_rows[-1]["pair_retention"] = (
            effective_rows[-1]["n_within_stratum_pairs"]
            / effective_rows[-1]["n_all_pairs"]
            if effective_rows[-1]["n_all_pairs"] else float("nan")
        )
    effective = pd.DataFrame(effective_rows)
    effective.to_csv(RESULTS / f"{prefix}_effective.csv", index=False)

    # ---- balance: positives against negatives, before and after -----------
    balance_rows = []
    for label in labels:
        defined = test[test[label] != sentinel].copy()
        positive = (defined[label] == 1).to_numpy()
        stratum = defined["stratum"].to_numpy()
        flat = np.ones(len(defined), dtype=float)
        weighted = pair_weights(stratum, positive)
        for source in variables:
            coarse = MATCHING_VARIABLES[source]
            # the coarsened variable, one indicator per level; balance must be
            # exact after weighting, because a pair shares its stratum
            for level in sorted(defined[coarse].unique()):
                indicator = (defined[coarse] == level).to_numpy().astype(float)
                balance_rows.append({
                    "label": label, "variable": coarse, "level": level,
                    "scale": "coarsened", "held_fixed": "yes",
                    "smd_before": smd(indicator, positive, flat),
                    "smd_after": smd(indicator, positive, weighted),
                })
        # Every pre-acquisition feature on its underlying scale, whether or not
        # the strata hold it fixed. The five that are not held fixed are the
        # point: a balance table covering only the matched variables says
        # nothing about what "matched on acquisition context" leaves free.
        for source in ALL_ACQ_PRE:
            if source not in defined:
                continue
            values = pd.to_numeric(defined[source], errors="coerce").to_numpy()
            usable = np.isfinite(values)
            if usable.sum() > 1:
                balance_rows.append({
                    "label": label, "variable": source, "level": "(continuous)",
                    "scale": "underlying",
                    "held_fixed": "yes" if source in variables else "no",
                    "smd_before": smd(values[usable], positive[usable], flat[usable]),
                    "smd_after": smd(values[usable], positive[usable],
                                     weighted[usable]),
                })
    balance = pd.DataFrame(balance_rows)
    balance.to_csv(RESULTS / f"{prefix}_balance.csv", index=False)

    worst_coarse = balance[balance["scale"] == "coarsened"]["smd_after"].abs().max()
    worst_under = balance[balance["scale"] == "underlying"]["smd_after"].abs().max()
    print(f"balance after pair weighting: worst |SMD| on a coarsened variable "
          f"{worst_coarse:.2e}, on an underlying value {worst_under:.4f}")

    meta = {
        "specification": "within-stratum on coarsened exact strata",
        "coarsening": {
            "arrival_to_acquisition_minutes": DELAY_LABELS,
            "acquisition_hour_of_day": HOUR_LABELS,
            "acquisition_day_of_week": ["weekday", "weekend"],
            "triage_acuity": "as recorded, missing as its own level",
        },
        "n_strata": int(len(strata)),
        "n_records": int(len(test)),
        "n_patients": int(test["subject_id"].nunique()),
        "strata_path": str(out_path),
        "worst_abs_smd_after_coarsened": float(worst_coarse),
        "worst_abs_smd_after_underlying": float(worst_under),
    }
    if not args.skip_propensity:
        meta["superset_propensity_diagnostic"] = propensity_diagnostic(derived)
        d = meta["superset_propensity_diagnostic"]
        print(f"superset diagnostic: early-ECG rate {d['early_ecg_rate']:.4f} over "
              f"{d['n_stays']:,} stays; context predicts availability at AUROC "
              f"{d['auroc']:.4f}")
    (RESULTS / f"{prefix}_matching.json").write_text(json.dumps(meta, indent=2),
                                              encoding="utf-8")

    # The prefix, not a literal. A sensitivity writes to its own files, and a
    # message naming the declared ones reads as though it had overwritten
    # them, which is the first thing anyone running it will want to rule out.
    print(f"\nwrote {prefix}_strata.csv ({len(strata)}), "
          f"{prefix}_effective.csv ({len(effective)}), "
          f"{prefix}_balance.csv ({len(balance)}), {prefix}_matching.json")
    print(f"row-level stratum assignment -> {out_path}")
    for _, row in effective.iterrows():
        print(f"  {row['label']:<32} {row['n_patients_effective']:,} effective "
              f"patients, {row['n_strata_informative']}/{row['n_strata_total']} "
              f"informative strata, {row['pair_retention']:.4f} of pairs retained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

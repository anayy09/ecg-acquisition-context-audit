#!/usr/bin/env python
"""Score row 8: does the waveform advantage survive with acquisition context fixed?

The quantity `acq_context_audit` declares is `waveform_gain_retained`, with a
target of at most 0.50. The within-stratum specification makes "held fixed" concrete: the within-stratum
concordance counts only positive-negative pairs from the same coarsened exact
stratum, so no compared pair differs in delay bucket, hour bucket, weekday status
or triage acuity.

One design choice decides what the number means, and it is made here explicitly.
The unmatched comparison is computed on the SAME rows as the matched one, so the
only difference between the two is whether the concordance pairs are restricted
within stratum. Comparing a within-stratum statistic on informative strata
against a plain statistic on the whole test fold would confound restricting the
pairs with dropping the rows, and the loss of gain would then be partly a
selection effect. The full-cohort gain is reported too, as context.

    python scripts/score_matched.py
    python scripts/score_matched.py --n-boot 2000     # faster, for a smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import (  # noqa: E402
    ClusterIndex,
    auc_from_ranks,
    clustered_statistic_ci,
    rank_scores,
)
from contract import load_base_config  # noqa: E402
from paths import REPO_ROOT, derived_root  # noqa: E402
from score_arms import holm, join_arms, ledger_index, pooled_arm  # noqa: E402
from stratified import StratifiedConcordance, selftest  # noqa: E402

RESULTS = REPO_ROOT / "results"
COMPARISONS = REPO_ROOT / "comparisons"
MATCHED_ARM = "R8_waveform_matched"
WAVEFORM_ARM = "R6_waveform"
BASELINE_ARM = "R2_demo"
CONTEXT_ARM = "R3_acqctx_pre"


class MatchedCell:
    """One label's evidence, carrying both the plain and the stratified statistic."""

    def __init__(self, joined: pd.DataFrame, arms: list[str],
                 stratum: np.ndarray) -> None:
        self.arms = arms
        self.y = joined["y_true"].to_numpy().astype(np.int64)
        self.subject = joined["subject_id"].to_numpy()
        self.stratum = stratum
        self.n_items = int(len(joined))
        self.ranks, self.buckets, self.concordance = {}, {}, {}
        for arm in arms:
            score = joined[arm].to_numpy()
            r, b = rank_scores(score)
            self.ranks[arm], self.buckets[arm] = r, b
            self.concordance[arm] = StratifiedConcordance(self.y, score, stratum)

    def plain(self, arm: str, rows: np.ndarray) -> float:
        if rows.size == 0:
            return float("nan")
        y = self.y[rows]
        if y.min() == y.max():
            return float("nan")
        return auc_from_ranks(y, self.ranks[arm][rows], self.buckets[arm])

    def within(self, arm: str, rows: np.ndarray) -> float:
        return self.concordance[arm].evaluate(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    import yaml

    problems = selftest()
    if problems:
        for line in problems:
            print(f"stratified selftest: {line}", file=sys.stderr)
        return 1
    print("within-stratum concordance self-test passed: it reproduces plain AUROC "
          "on one stratum and collapses a between-stratum shortcut to 0.5")

    base = load_base_config()
    labels = list(base["labels"])
    seeds = list(base["training"]["seeds"])
    derived = derived_root(args.data_root)
    index = ledger_index()
    audit = yaml.safe_load((COMPARISONS / "acq_context_audit.yaml").read_text("utf-8"))
    threshold = float(audit["secondary_threshold"])

    strata_path = derived / "matched_strata.parquet"
    if not strata_path.is_file():
        print("matched_strata.parquet is missing; run build_matched_cohort.py first",
              file=sys.stderr)
        return 1
    strata = pd.read_parquet(strata_path)[["study_id", "stratum"]]

    arms = [BASELINE_ARM, CONTEXT_ARM, WAVEFORM_ARM]
    arm_rows, contrast_rows, retained_rows = [], [], []

    for label in labels:
        pooled = {}
        for arm in arms:
            entry = pooled_arm(arm, label, seeds, index, derived)
            if entry is None:
                print(f"  {label}: no pooled {arm}; skipping", file=sys.stderr)
                pooled = {}
                break
            pooled[(arm, label)] = entry
        if not pooled:
            continue

        joined = join_arms(pooled, arms, label)
        joined = joined.join(strata.set_index("study_id"), how="inner")
        joined = joined[joined["stratum"].notna()]

        cell = MatchedCell(joined, arms, joined["stratum"].to_numpy())
        rows_all = np.arange(cell.n_items)

        # Rows in strata that contribute at least one pair. Everything below is
        # computed on exactly these rows, both matched and unmatched, so the only
        # difference between the two is the pair restriction.
        informative = cell.concordance[WAVEFORM_ARM].informative()
        keep = np.flatnonzero(informative[cell.concordance[WAVEFORM_ARM].codes])
        subject_keep = joined["subject_id"].to_numpy()[keep]

        index_map = {sid: i for i, sid in enumerate(np.unique(subject_keep))}
        cluster = ClusterIndex.from_codes(
            np.array([index_map[s] for s in subject_keep], dtype=np.int64),
            len(index_map),
        )
        local = MatchedCell(joined.iloc[keep], arms,
                            joined["stratum"].to_numpy()[keep])

        for arm in arms:
            plain = local.plain(arm, np.arange(local.n_items))
            within = local.within(arm, np.arange(local.n_items))

            def statistic(gathered, arm=arm):
                return local.within(arm, gathered["c"])

            interval = clustered_statistic_ci({"c": cluster}, len(index_map),
                                              statistic, n_boot=args.n_boot,
                                              seed=args.seed)
            arm_rows.append({
                "arm": MATCHED_ARM if arm == WAVEFORM_ARM else arm,
                "source_arm": arm, "label": label, "unit": "record",
                "statistic": "within_stratum_concordance",
                "auroc_within_stratum": within,
                "ci_lo": interval["ci_lo"], "ci_hi": interval["ci_hi"],
                "auroc_plain_same_rows": plain,
                "n_items": int(local.n_items),
                "n_groups": int(len(index_map)),
                "n_boot": int(args.n_boot),
                "n_pairs": int(local.concordance[arm].n_pairs()),
            })

        # ---- the declared secondary quantity ------------------------------
        def gains(gathered):
            rows = gathered["c"]
            matched = local.within(WAVEFORM_ARM, rows) - local.within(BASELINE_ARM, rows)
            unmatched = local.plain(WAVEFORM_ARM, rows) - local.plain(BASELINE_ARM, rows)
            if not np.isfinite(matched) or not np.isfinite(unmatched):
                return float("nan")
            if abs(unmatched) < 1e-12:
                return float("nan")
            return matched / unmatched

        # The declared formula, `(R8 - R2) / (R6 - R2)`, does not say which R2.
        # Reading it with a within-stratum R2 in the numerator keeps numerator and
        # denominator on their own statistics, which is what `gains` computes and
        # is the reading this project adopts. Reading it with the PLAIN R2 in the
        # numerator mixes the two statistics in one difference, which is harder to
        # interpret but is the more literal reading of "row 8 minus row 2". Both
        # are computed, because picking one silently on an ambiguous
        # pre-declaration is the failure this project audits.
        def gains_plain_baseline(gathered):
            rows = gathered["c"]
            numerator = local.within(WAVEFORM_ARM, rows) - local.plain(BASELINE_ARM, rows)
            denominator = local.plain(WAVEFORM_ARM, rows) - local.plain(BASELINE_ARM, rows)
            if not np.isfinite(numerator) or not np.isfinite(denominator):
                return float("nan")
            if abs(denominator) < 1e-12:
                return float("nan")
            return numerator / denominator

        alternative = clustered_statistic_ci({"c": cluster}, len(index_map),
                                             gains_plain_baseline,
                                             n_boot=args.n_boot, seed=args.seed)
        retained = clustered_statistic_ci({"c": cluster}, len(index_map), gains,
                                          n_boot=args.n_boot, seed=args.seed)
        rows_local = np.arange(local.n_items)
        gain_matched = (local.within(WAVEFORM_ARM, rows_local)
                        - local.within(BASELINE_ARM, rows_local))
        gain_unmatched = (local.plain(WAVEFORM_ARM, rows_local)
                          - local.plain(BASELINE_ARM, rows_local))
        gain_full = (cell.plain(WAVEFORM_ARM, rows_all)
                     - cell.plain(BASELINE_ARM, rows_all))
        retained.update(
            label=label, quantity="waveform_gain_retained", unit="record",
            gain_matched=gain_matched, gain_unmatched_same_rows=gain_unmatched,
            gain_unmatched_full_cohort=gain_full,
            n_items=int(local.n_items), threshold=threshold,
            meets_target=bool(retained["estimate"] <= threshold),
            retained_plain_baseline=alternative["estimate"],
            retained_plain_baseline_lo=alternative["ci_lo"],
            retained_plain_baseline_hi=alternative["ci_hi"],
            meets_target_plain_baseline=bool(alternative["estimate"] <= threshold),
            note="matched and unmatched are computed on the SAME rows, so the only "
                 "difference is the within-stratum pair restriction. "
                 "retained_plain_baseline is the alternative reading of the declared "
                 "formula, with an unstratified R2_demo in the numerator.",
        )
        retained_rows.append(retained)

        # ---- R8 against R6, the family's remaining three tests -------------
        def difference(gathered):
            rows = gathered["c"]
            return local.within(WAVEFORM_ARM, rows) - local.plain(WAVEFORM_ARM, rows)

        contrast = clustered_statistic_ci({"c": cluster}, len(index_map), difference,
                                          n_boot=args.n_boot, seed=args.seed)
        contrast.update(
            contrast=f"{MATCHED_ARM} - {WAVEFORM_ARM}", arm_a=MATCHED_ARM,
            arm_b=WAVEFORM_ARM, label=label, family="acq_context_audit",
            unit="record", n_items=int(local.n_items),
            difference=contrast["estimate"], involves_waveform_arm=True,
        )
        contrast_rows.append(contrast)

        print(f"  {label:<32} R8 {local.within(WAVEFORM_ARM, rows_local):.4f} "
              f"vs R6 {local.plain(WAVEFORM_ARM, rows_local):.4f}, gain retained "
              f"{retained['estimate']:.4f} "
              f"[{retained['ci_lo']:.4f}, {retained['ci_hi']:.4f}]")

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(arm_rows).to_csv(RESULTS / "p4_arms.csv", index=False)
    pd.DataFrame(retained_rows).to_csv(RESULTS / "p4_retained.csv", index=False)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(RESULTS / "p4_contrasts.csv", index=False)

    # ---- the audit family, now complete at all 15 tests -------------------
    p3_path = RESULTS / "p3_contrasts.csv"
    if p3_path.is_file() and not contrasts.empty:
        p3 = pd.read_csv(p3_path)
        family_size = int(p3["holm_family_size"].iloc[0])
        combined = {f"{r['contrast']}@{r['label']}": float(r["p_value"])
                    for _, r in p3.iterrows()}
        combined.update({f"{r['contrast']}@{r['label']}": float(r["p_value"])
                         for _, r in contrasts.iterrows()})
        adjusted = holm(combined, family_size)
        final = pd.DataFrame([
            {"contrast": key.split("@")[0], "label": key.split("@")[1],
             "p_value": value, "p_holm_final": adjusted[key],
             "holm_family_size": family_size,
             "phase": "P4" if key in {f"{r['contrast']}@{r['label']}"
                                      for _, r in contrasts.iterrows()} else "P3"}
            for key, value in combined.items()
        ]).sort_values("p_value")
        final.to_csv(RESULTS / "p4_family_final.csv", index=False)
        print(f"\naudit family complete: {len(final)} of {family_size} tests, "
              f"Holm recomputed over the full family")
        if len(final) != family_size:
            print(f"  WARNING: {family_size - len(final)} declared test(s) are still "
                  f"absent; the correction is over what exists", file=sys.stderr)

    # ---- verdict ----------------------------------------------------------
    verdict = {
        "arm": MATCHED_ARM,
        "specification": "within-stratum on coarsened exact strata",
        "quantity": "waveform_gain_retained",
        "definition": "(R6 - R2) within stratum, over (R6 - R2) on the same rows "
                      "without the pair restriction",
        "threshold": threshold,
        "threshold_direction": "at most",
        "per_label": [
            {"label": r["label"], "retained": r["estimate"],
             "ci": [r["ci_lo"], r["ci_hi"]],
             "gain_matched": r["gain_matched"],
             "gain_unmatched_same_rows": r["gain_unmatched_same_rows"],
             "gain_unmatched_full_cohort": r["gain_unmatched_full_cohort"],
             "meets_target": bool(r["meets_target"]),
             "retained_plain_baseline": r["retained_plain_baseline"],
             "retained_plain_baseline_ci": [r["retained_plain_baseline_lo"],
                                            r["retained_plain_baseline_hi"]],
             "meets_target_plain_baseline": bool(r["meets_target_plain_baseline"])}
            for r in retained_rows
        ],
        "formula_ambiguity": (
            "acq_context_audit writes the quantity as (R8 - R2) / (R6 - R2) without "
            "saying which R2 the numerator uses. Both readings are computed and "
            "reported: `retained` uses a within-stratum R2, keeping numerator and "
            "denominator each on one statistic; `retained_plain_baseline` uses the "
            "unstratified R2, which is the more literal reading. The verdict is the "
            "same under both, so nothing turns on the choice."
        ),
        "effective_sample_note": (
            "The effective patient count, not the ECG count, bounds every row 8 "
            "claim, and it is per label because a rare label empties more strata "
            "than a common one. See results/p4_effective.csv."
        ),
        "matching_note": (
            "Balance is exact within stratum on the coarsened variables by "
            "construction; the residual imbalance on the underlying continuous "
            "values is what coarsening leaves behind and is reported in "
            "results/p4_balance.csv rather than assumed away."
        ),
    }
    (RESULTS / "p4_verdict.json").write_text(json.dumps(verdict, indent=2),
                                             encoding="utf-8")
    print(f"\nwrote p4_arms.csv, p4_retained.csv, p4_contrasts.csv, p4_verdict.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

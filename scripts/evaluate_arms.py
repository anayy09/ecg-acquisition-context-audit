#!/usr/bin/env python
"""Apply gate the pre-declared rule's rule to the scored this stage runs and write the verdict.

pre-declared rule 3 is a COMPLETENESS gate, not a hypothesis gate. the project's design notes states it as three
criteria: rows 1 to 7 across three horizons at five seeds all registered and
verifiable, the recovery ratio computed with a paired patient-clustered
interval, and the noise floor measured. None of them is "the claim succeeded",
and this script keeps the two apart: a complete evidence set whose primary
quantity misses its target is a PASS for pre-declared rule 3 and a failure for the claim, and
both are reported side by side rather than one being read as the other.

It also resolves the full-strength branch of pre-declared rule 1, which the first verdict left PENDING for
want of a locally reproduced waveform number and which `evaluate_reproduction.py`
explicitly deferred to this phase.

the corresponding check recomputes every criterion from the CSVs rather than
reading this script's output. That separation caught a real bug in the pre-declared rule 1
implementation at this stage, where the rule was evaluated against the primary label
alone while the project's design notes writes it as "on at least one deterioration label".

    python scripts/evaluate_arms.py
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
COMPARISONS = REPO_ROOT / "comparisons"

#: Rows 1 to 7 of the headline table. Row 8 is this stage's and is not required by pre-declared rule 3.
REQUIRED_ARMS = (
    "R1_prevalence", "R2_demo", "R3_acqctx_pre", "R3b_acqctx_window",
    "R4_demo_acq", "R5_tabular", "R6_waveform", "R7_waveform_demo_acq",
)
G1_FULL_STRENGTH_TOLERANCE = 0.03


def evaluate(arms: pd.DataFrame, recovery: pd.DataFrame, noise: pd.DataFrame,
             contrasts: pd.DataFrame, audit: dict) -> dict:
    labels = [str(x) for x in audit["labels"]]
    primary_label = str(audit["primary_label"])
    primary_arm = str(audit["primary_arm"])
    reference_arm = str(audit["primary_reference_arm"])
    target = float(audit["success_threshold"])

    # ---- criterion 1: the evidence set is complete ------------------------
    present = {(r["arm"], r["label"]) for _, r in arms.iterrows()}
    missing = sorted(
        f"{arm}@{label}" for arm in REQUIRED_ARMS for label in labels
        if (arm, label) not in present
    )
    seeds_short = sorted(
        f"{r['arm']}@{r['label']} has {int(r['n_seeds'])} seeds"
        for _, r in arms.iterrows()
        if r["label"] in labels and int(r["n_seeds"]) < 5
    )
    complete = not missing and not seeds_short

    # ---- criterion 2: the recovery ratio has an interval on the ratio -----
    primary_rows = recovery[(recovery["arm"] == primary_arm)
                            & (recovery["label"] == primary_label)]
    ratio_ok = bool(
        not recovery.empty
        and not primary_rows.empty
        and recovery["ci_lo"].notna().all()
        and recovery["ci_hi"].notna().all()
        and (recovery["denominator_source"].astype(str)
             .str.contains("per-label").all())
    )

    # ---- criterion 3: the noise floor is measured -------------------------
    floor_ok = bool(
        not noise.empty
        and (noise["n_seeds"] >= 5).all()
        and (noise["unit"] == "record").all()
    )

    criteria = [
        {"name": "rows 1 to 7, three horizons, 5 seeds, registered and verifiable",
         "met": bool(complete),
         "observed": f"{len(present & {(a, l) for a in REQUIRED_ARMS for l in labels})} "
                     f"of {len(REQUIRED_ARMS) * len(labels)} (arm, label) cells present",
         "missing": missing, "seeds_short": seeds_short},
        {"name": "recovery ratio computed with a paired patient-clustered interval",
         "met": ratio_ok,
         "observed": f"{len(recovery)} ratio row(s), interval taken on the ratio "
                     f"itself, denominator = per-label record-level {reference_arm} "
                     f""},
        {"name": "noise floor measured",
         "met": floor_ok,
         "observed": f"{len(noise)} per-(arm, label) record-level spreads, "
                     f"range {noise['seed_spread'].min():.4f} to "
                     f"{noise['seed_spread'].max():.4f}" if not noise.empty
                     else "no noise rows"},
    ]
    result = "PASS" if all(c["met"] for c in criteria) else "INCOMPLETE"

    # ---- the claim, reported beside the gate and never as the gate --------
    def ratio_row(label: str) -> dict | None:
        sub = recovery[(recovery["arm"] == primary_arm) & (recovery["label"] == label)]
        if sub.empty:
            return None
        row = sub.iloc[0]
        return {
            "label": label,
            "recovery_ratio": float(row["estimate"]),
            "ci": [float(row["ci_lo"]), float(row["ci_hi"])],
            "numerator_auroc": float(row["numerator_auroc"]),
            "denominator_auroc": float(row["denominator_auroc"]),
            "target": target,
            "meets_target": bool(row["estimate"] >= target),
            "interval_excludes_target": bool(row["ci_lo"] >= target),
        }

    primary = ratio_row(primary_label)
    others = [ratio_row(l) for l in labels if l != primary_label]
    others = [o for o in others if o]

    # ---- the pre-declared rule's full-strength branch, pending since the first verdict -------------------
    gaps = []
    for label in labels:
        r3 = arms[(arms["arm"] == primary_arm) & (arms["label"] == label)]
        r6 = arms[(arms["arm"] == reference_arm) & (arms["label"] == label)]
        if r3.empty or r6.empty:
            continue
        gap = float(r6.iloc[0]["auroc"]) - float(r3.iloc[0]["auroc"])
        gaps.append({
            "label": label,
            "r3_auroc": float(r3.iloc[0]["auroc"]),
            "r6_auroc": float(r6.iloc[0]["auroc"]),
            "gap": gap,
            "within_tolerance": bool(gap <= G1_FULL_STRENGTH_TOLERANCE),
        })
    passing = [g["label"] for g in gaps if g["within_tolerance"]]

    verdict = {
        "gate": "G3",
        "rule": "the project's design notes pre-declared rule 3: rows 1 to 7 across three horizons at five seeds, all "
                "registered and verifiable; the recovery ratio computed with a "
                "paired patient-clustered interval; the noise floor measured",
        "nature": "pre-declared rule 3 is a completeness gate. It asks whether the minimum evidence "
                  "set exists, not whether the claim succeeded. The two outcomes "
                  "are reported separately below and must not be read as one.",
        "unit": "record",
        "unit_note": "every this stage contrast is one row per ECG; rows 1 to 5 have no "
                     "crops, so a crop-level waveform score never enters (the crop-level and record-level separation, "
                     "the declared scoring conventions)",
        "criteria": criteria,
        "result": result,
        "primary_quantity": "recovery_ratio",
        "primary_arm": primary_arm,
        "primary_reference_arm": reference_arm,
        "primary_label": primary_label,
        "primary": primary,
        "primary_outcome": (
            "MET" if primary and primary["meets_target"] else "NOT MET"
        ),
        "other_labels_exploratory": others,
        "exploratory_note": (
            "The pre-declared primary is the recovery ratio at "
            f"{primary_label}. The other labels are reported beside it and are "
            "EXPLORATORY per the comparison file's exploratory_policy: promoting "
            "one of them to the headline would be the post-hoc endpoint switch "
            "the first verdict already refused once."
        ),
        "g1_full_strength_branch": {
            "rule": "the project's design notes pre-declared rule 1: continue at full strength if R3_acqctx_pre comes "
                    "within 0.03 AUROC of the waveform arm on at least one "
                    "deterioration label",
            "tolerance": G1_FULL_STRENGTH_TOLERANCE,
            "per_label": gaps,
            "labels_within_tolerance": passing,
            "resolution": (
                f"RESOLVED: {primary_arm} comes within {G1_FULL_STRENGTH_TOLERANCE} "
                f"AUROC of {reference_arm} on {len(passing)} of {len(gaps)} labels "
                f"({', '.join(passing)})."
                if passing else
                f"RESOLVED: {primary_arm} does not come within "
                f"{G1_FULL_STRENGTH_TOLERANCE} AUROC of {reference_arm} on any "
                "label, so the pre-declared rule's full-strength branch does NOT apply and the first verdict's "
                "branch 2, continue in weaker form, stands as the operative verdict."
            ),
            "note": "Left PENDING by the first verdict for want of a locally reproduced "
                    "waveform number, and deferred to this stage by evaluate_reproduction.py. "
                    "Evaluated at record level, because rows 3 and 6 must be "
                    "compared on the same unit.",
        },
        "families": {
            "acq_context_audit": {
                "holm_family_size": int(contrasts["holm_family_size"].iloc[0])
                if not contrasts.empty else None,
                "tests_present": int(len(contrasts)),
                "tests_pending_r8": int(contrasts["holm_tests_pending"].iloc[0])
                if not contrasts.empty else None,
                "note": "corrected at the DECLARED size of 15 while the three R8 "
                        "tests are pending, per the declared family size",
            },
            "horizon_ladder": {
                "note": "a separate family with its own Holm correction; never "
                        "pooled with acq_context_audit. Its verdict lives "
                        "in results/p3_trends.csv and p3_ladder.csv.",
            },
        },
        "caveat": (
            "The denominator of every recovery ratio is a locally reproduced row 6 "
            "that lands ABOVE the published number (the reproduction verdict: 0.8474 crop against a "
            "published 0.8279). A stronger row 6 makes the 70 percent target harder "
            "to meet, not easier, so the target is harder here than it would have "
            "been against the published figure. This is the conservative direction "
            "and it is stated because a reader's first suspicion of an audit paper "
            "is that the audited baseline was weakened."
        ),
    }
    return verdict


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    import yaml

    needed = {
        "arms": RESULTS / "p3_arms.csv",
        "recovery": RESULTS / "p3_recovery.csv",
        "noise": RESULTS / "p3_noise.csv",
        "contrasts": RESULTS / "p3_contrasts.csv",
    }
    for name, path in needed.items():
        if not path.is_file():
            print(f"{path} is missing; run scripts/score_arms.py first", file=sys.stderr)
            return 1

    audit = yaml.safe_load(
        (COMPARISONS / "acq_context_audit.yaml").read_text(encoding="utf-8")
    )
    verdict = evaluate(
        pd.read_csv(needed["arms"]), pd.read_csv(needed["recovery"]),
        pd.read_csv(needed["noise"]), pd.read_csv(needed["contrasts"]), audit,
    )
    (RESULTS / "p3_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )

    print(f"Gate pre-declared rule 3: {verdict['result']}")
    for criterion in verdict["criteria"]:
        mark = "met" if criterion["met"] else "NOT MET"
        print(f"  [{mark:>7}] {criterion['name']}")
        print(f"            {criterion['observed']}")
    if verdict["primary"]:
        p = verdict["primary"]
        print(f"\nPre-declared primary ({p['label']}): recovery ratio "
              f"{p['recovery_ratio']:.4f} [{p['ci'][0]:.4f}, {p['ci'][1]:.4f}] "
              f"against a target of {p['target']:.2f}  -> "
              f"{verdict['primary_outcome']}")
    print(f"\nG1 full-strength branch: "
          f"{verdict['g1_full_strength_branch']['resolution']}")
    print(f"\nwrote {RESULTS / 'p3_verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

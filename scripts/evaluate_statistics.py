#!/usr/bin/env python
"""Evaluate what the statistics stage decides, and write it where another script can re-derive it.

the statistics stage decides two narrow things and neither is a claim about discrimination:

  usable probabilities   does an arm's ECE sit inside the simulated null band,
                         under BOTH declared binning schemes?
  clinical usefulness    at the pre-declared threshold, does an arm's
                         net benefit exceed treat-all and treat-none, with a
                         PAIRED clustered interval on the difference?

They are separate decisions and the second depends on the first, which is the
whole reason this file writes the dependency down rather than leaving a reader
to notice it. Net benefit reads the probability scale; AUROC reads only the
ranking. An arm can rank cases well and still be useless at a fixed threshold
because its probabilities are inflated, and that is a statement about the
probability scale and not about the signal the model found.

The declared threshold anticipated exactly that for the waveform arms — focal BCE at gamma 2
, then a mean over four crops, then a mean over five seeds
 — and said so before any of these numbers existed. The verdict therefore
carries the caveat as a field rather than as prose someone can drop, and
the corresponding check fails if it goes missing.

    python scripts/evaluate_statistics.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import load_base_config  # noqa: E402
from paths import REPO_ROOT  # noqa: E402

RESULTS = REPO_ROOT / "results"
CALIBRATION = RESULTS / "calibration"
CURVES = RESULTS / "decision_curve"

CAVEAT = (
    "Net benefit reads the PROBABILITY SCALE; AUROC reads only the ranking. An "
    "arm that ranks well can still lose at a fixed threshold because its "
    "probabilities are inflated, and that is a finding about the probability "
    "scale rather than about discrimination. R6_waveform and "
    "R7_waveform_demo_acq were fitted under focal BCE at gamma 2 and "
    "their reported score is a mean over four crops and then over five "
    "seeds. The declared threshold stated this before any of these numbers existed, and "
    "recalibration was declared out of scope in the same entry because only "
    "test-fold predictions are stored and refitting on them is circular. No "
    "net-benefit result may be read as evidence about discrimination, and no "
    "claim in the abstract rests on one."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_base_config()
    declared = config["evaluation"]["decision_curve"]
    threshold = float(declared["threshold"])
    quote_labels = sorted(set(declared.get("quote_point_at_labels")
                              or [declared["threshold_label"]]))

    ece = pd.read_csv(CALIBRATION / "ece.csv")
    points = pd.read_csv(CURVES / "nb_at_threshold.csv")
    schemes = sorted(set(ece["scheme"]))

    calibration_rows = []
    for (arm, label), group in ece.groupby(["arm", "label"]):
        inside = {str(r["scheme"]): bool(r["inside_null_band"])
                  for _, r in group.iterrows()}
        calibration_rows.append({
            "arm": arm,
            "label": label,
            "inside_null_band": {s: inside.get(s) for s in schemes},
            # A verdict of "calibrated" requires BOTH schemes. One scheme
            # agreeing is not a result: the choice of binning is a choice, and
            # reporting only the flattering one is the failure this project is
            # about.
            "usable_probabilities": all(inside.get(s, False) for s in schemes),
            "ece": {str(r["scheme"]): float(r["ece"]) for _, r in group.iterrows()},
            "seed_ece_spread": {str(r["scheme"]): float(r["seed_ece_spread"])
                                for _, r in group.iterrows()},
            "mean_predicted": float(group["mean_predicted"].iloc[0]),
            "observed_rate": float(group["observed_rate"].iloc[0]),
        })

    usable = {(r["arm"], r["label"]): r["usable_probabilities"]
              for r in calibration_rows}

    benefit_rows = []
    for _, row in points.iterrows():
        benefit_rows.append({
            "arm": str(row["arm"]),
            "label": str(row["label"]),
            "net_benefit": float(row["net_benefit"]),
            "ci": [float(row["ci_lo"]), float(row["ci_hi"])],
            "net_benefit_treat_all": float(row["net_benefit_treat_all"]),
            "vs_treat_all": float(row["vs_treat_all"]),
            "vs_treat_all_ci": [float(row["vs_treat_all_lo"]),
                                float(row["vs_treat_all_hi"])],
            "beats_treat_all": bool(row["vs_treat_all_lo"] > 0),
            "beats_treat_none": bool(row["ci_lo"] > 0),
            "flagged_fraction": float(row["flagged_fraction"]),
            "mean_predicted": float(row["mean_predicted"]),
            "prevalence": float(row["prevalence"]),
            "seed_nb_spread": float(row["seed_nb_spread"]),
            "usable_probabilities": bool(usable.get((str(row["arm"]),
                                                     str(row["label"])), False)),
            "n_groups": int(row["n_groups"]),
        })

    ranked = sorted(benefit_rows, key=lambda r: -r["net_benefit"])
    best = ranked[0] if ranked else None

    payload = {
        "phase": "P5",
        "threshold": threshold,
        "threshold_declared_by": "the declared threshold, closing the declared decision threshold",
        "threshold_quoted_at": quote_labels,
        "threshold_rationale": str(declared.get("threshold_rationale", "")).strip(),
        "unit": "record",
        "recalibration": str(declared.get("recalibration", "none")),
        "calibration_reference": (
            "the simulated null band, not zero. At 15 bins over roughly 6,400 "
            "records a perfectly calibrated score still shows an ECE of order "
            "0.01, so the reference is what a calibrated score of this size and "
            "shape would produce, computed per cell."
        ),
        "interval_note": (
            "ECE is a mean of ABSOLUTE per-bin gaps and is therefore positively "
            "biased under resampling: a clustered resample duplicates patients, "
            "which adds noise that can only push each bin's gap up. The "
            "percentile interval can consequently exclude its own point estimate "
            "from below. The declared protocol is a percentile interval "
            "and it is not silently swapped for a bias-corrected one; the null "
            "band, not the interval, is what answers whether an arm is calibrated."
        ),
        "calibration": calibration_rows,
        "net_benefit": benefit_rows,
        "ranked_by_net_benefit": [r["arm"] for r in ranked],
        "best_arm_at_threshold": best["arm"] if best else None,
        "caveat": CAVEAT,
        "what_this_does_not_decide": (
            "Nothing here changes any pre-declared outcome. All three "
            "quantitative targets failed at the endpoints they named (the first verdict, "
            "the second verdict, the matched-arm verdict) and the horizon-ladder trend is unsupported. "
            "Calibration and net benefit are secondary analyses 9 and 10 in the project's design notes "
            "section 4, cut first if the schedule slips, and the confirmatory set "
            "depends on neither."
        ),
    }

    out = RESULTS / "p5_verdict.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"wrote {out.relative_to(REPO_ROOT)}")
        for row in ranked:
            print(f"  {row['arm']:24s} NB {row['net_benefit']:.4f}  "
                  f"flags {row['flagged_fraction']:.3f}  "
                  f"beats treat-all {row['beats_treat_all']}  "
                  f"usable probabilities {row['usable_probabilities']}")
    print("the statistics stage VERDICT WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

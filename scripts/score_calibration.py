#!/usr/bin/env python
"""Score calibration for the headline arms: ECE, reliability curves, noise floor.

Everything here is at RECORD level and reuses this stage's loaders unchanged, so the
probability being scored is exactly the probability whose AUROC appears in
Table 1 and no second convention is introduced.

What the reported probability actually is, stated because it changes the
quantity rather than merely qualifying it:

  the tabular arms  a mean over five seeds of a LightGBM probability
  rows 6 and 7      a mean over four 2.5-second crops of a sigmoid
                    fitted under focal BCE at gamma 2, then a mean over
                    five seeds

Averaging probabilities shrinks them toward the middle, so these are calibration
figures for the seed ensemble and not for any single trained model. The per-seed
ECE spread is reported beside every cell as this statistic's own noise floor, by
the same argument the declared scoring conventions makes for AUROC: a difference smaller than run-to-run
variation is not a difference.

    python scripts/score_calibration.py
    python scripts/score_calibration.py --n-boot 2000     # faster smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import ClusterIndex, clustered_statistic_ci  # noqa: E402
from calibration import (  # noqa: E402
    ece,
    equal_mass_bins,
    equal_width_bins,
    null_band,
    reliability,
    selftest,
)
from contract import load_base_config  # noqa: E402
from paths import REPO_ROOT, derived_root  # noqa: E402
from score_arms import (  # noqa: E402
    _tabular_seed_frame,
    _waveform_seed_frame,
    ledger_index,
    WAVEFORM_ARMS,
)

RESULTS = REPO_ROOT / "results"
CALIBRATION = RESULTS / "calibration"
COMPARISONS = REPO_ROOT / "comparisons"

#: The headline arms. Row 8 is absent by construction: it is row 6's scores
#: under a restricted pair statistic, so it has no probability of its own and a
#: calibration figure for it would be row 6's figure under another name.
ARMS = (
    "R1_prevalence", "R2_demo", "R3_acqctx_pre", "R3b_acqctx_window",
    "R4_demo_acq", "R5_tabular", "R6_waveform", "R7_waveform_demo_acq",
)


def seed_frames(arm: str, label: str, seeds: list[int], index: dict,
                derived: Path) -> list[pd.DataFrame] | None:
    frames = []
    for seed in seeds:
        run_id = index.get((arm, label, seed)) or index.get((arm, "all15", seed))
        if run_id is None:
            return None
        if arm in WAVEFORM_ARMS:
            frames.append(_waveform_seed_frame(run_id, derived, label))
        else:
            frames.append(_tabular_seed_frame(run_id, derived))
    return frames


def pooled_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """The seed-pooled probability, on the intersection of the seeds' rows."""
    indexed = [f.set_index("study_id") for f in frames]
    wide = pd.concat([f["y_score"].rename(f"s{i}") for i, f in enumerate(indexed)],
                     axis=1, join="inner")
    base = indexed[0].loc[wide.index, ["subject_id", "y_true"]].copy()
    base["y_score"] = wide.mean(axis=1)
    return base.reset_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--null-draws", type=int, default=400)
    args = parser.parse_args()

    problems = selftest()
    if problems:
        print("calibration selftest FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    config = load_base_config()
    n_bins = int(config["evaluation"]["calibration"]["bins"])
    schemes = list(config["evaluation"]["calibration"]["schemes"])
    seeds = list(config["training"]["seeds"])
    declared = yaml.safe_load(
        (COMPARISONS / "acq_context_audit.yaml").read_text(encoding="utf-8")
    )
    labels = [str(x) for x in declared["labels"]]

    derived = derived_root()
    index = ledger_index()
    CALIBRATION.mkdir(parents=True, exist_ok=True)

    ece_rows: list[dict] = []
    curve_rows: list[dict] = []
    bin_rows: list[dict] = []

    for arm in ARMS:
        for label in labels:
            frames = seed_frames(arm, label, seeds, index, derived)
            if frames is None:
                raise RuntimeError(f"no registered runs for {arm} at {label}")
            pooled = pooled_frame(frames)
            y = pooled["y_true"].to_numpy().astype(np.int64)
            p = pooled["y_score"].to_numpy(dtype=float)
            groups = pooled["subject_id"].to_numpy()
            cluster = ClusterIndex(groups)

            for scheme in schemes:
                point = ece(y, p, n_bins, scheme)

                # The interval is taken on the ECE itself: patients are
                # resampled once and the statistic is recomputed end to end on
                # the gathered rows, exactly as the recovery ratio is.
                # Re-binning inside the replicate is deliberate — the bins are
                # part of the statistic, not a fixed frame it is evaluated in.
                interval = clustered_statistic_ci(
                    {"cell": cluster},
                    cluster.n_groups,
                    lambda rows, y=y, p=p: ece(y[rows["cell"]], p[rows["cell"]],
                                               n_bins, scheme),
                    n_boot=args.n_boot,
                    seed=args.seed,
                )

                # The per-seed spread is this statistic's own noise floor.
                per_seed = [ece(f["y_true"].to_numpy().astype(np.int64),
                                f["y_score"].to_numpy(dtype=float), n_bins, scheme)
                            for f in frames]
                per_seed = np.asarray(per_seed, dtype=float)

                # What a PERFECTLY calibrated score of this size and shape would
                # show. An ECE inside this band is not evidence of
                # miscalibration; see calibration.py for why zero is the wrong
                # reference.
                null_lo, null_hi = null_band(p, n_bins, scheme,
                                             n_draws=args.null_draws,
                                             seed=args.seed + 1)

                bins = (equal_width_bins(p, n_bins) if scheme == "equal_width"
                        else equal_mass_bins(p, n_bins))
                count, mean_p, mean_y = reliability(y, p, bins, n_bins)

                ece_rows.append({
                    "arm": arm,
                    "label": label,
                    "unit": "record",
                    "scheme": scheme,
                    "n_bins": n_bins,
                    "ece": point,
                    "ci_lo": interval["ci_lo"],
                    "ci_hi": interval["ci_hi"],
                    "n_boot": interval["n_boot"],
                    "n_items": int(len(pooled)),
                    "n_groups": int(cluster.n_groups),
                    "seed_ece_min": float(per_seed.min()),
                    "seed_ece_max": float(per_seed.max()),
                    "seed_ece_spread": float(per_seed.max() - per_seed.min()),
                    "n_seeds": len(per_seed),
                    "null_lo": null_lo,
                    "null_hi": null_hi,
                    "inside_null_band": bool(null_lo <= point <= null_hi),
                    # ECE is a mean of ABSOLUTE per-bin gaps, so it is
                    # positively biased: any noise added to a bin's observed
                    # rate pushes it up and never down. A clustered resample
                    # duplicates patients and so adds exactly that noise, which
                    # means the bootstrap distribution sits systematically above
                    # the observed value and the percentile interval can exclude
                    # its own point estimate from below. That is a property of
                    # the statistic, not a coding error, and it is recorded per
                    # cell rather than smoothed over. The declared protocol is a
                    # percentile interval and it is not silently swapped
                    # for a bias-corrected one; the null band above, not the
                    # interval, is what answers "is this arm calibrated".
                    "point_below_ci_lo": bool(point < interval["ci_lo"]),
                    "n_bins_occupied": int((count > 0).sum()),
                    "bin_count_min": int(count[count > 0].min()) if (count > 0).any() else 0,
                    "bin_count_max": int(count.max()),
                    "mean_predicted": float(p.mean()),
                    "observed_rate": float(y.mean()),
                })

                for b in range(n_bins):
                    if count[b] == 0:
                        continue
                    curve_rows.append({
                        "arm": arm, "label": label, "scheme": scheme, "bin": b,
                        "n": int(count[b]), "mean_predicted": float(mean_p[b]),
                        "observed_rate": float(mean_y[b]),
                        "gap": float(mean_y[b] - mean_p[b]),
                    })
                bin_rows.append({
                    "arm": arm, "label": label, "scheme": scheme,
                    "n_bins": n_bins, "n_occupied": int((count > 0).sum()),
                    "count_min": int(count[count > 0].min()) if (count > 0).any() else 0,
                    "count_max": int(count.max()),
                })

    pd.DataFrame(ece_rows).to_csv(CALIBRATION / "ece.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(CALIBRATION / "reliability.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(CALIBRATION / "bin_occupancy.csv", index=False)

    print(f"wrote {len(ece_rows)} ECE rows and {len(curve_rows)} reliability "
          f"points into {CALIBRATION}")
    print("CALIBRATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

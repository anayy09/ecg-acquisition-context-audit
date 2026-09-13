#!/usr/bin/env python
"""Net benefit over the declared grid, and at the one pre-declared threshold.

This script REFUSES TO COMPUTE ANYTHING until the operating threshold has been declared and the threshold
in `configs/base.yaml` is a number rather than `TBD`. That is the same
before-the-fact refusal the matched-arm stage makes, and it exists for the
same reason: checking afterwards that a threshold predated the curve cannot stop
a threshold from being chosen for how the curve reads.

The threshold is quoted at `deterioration_icu_24h` only. Curves are
drawn for all three labels because the shape is informative, but no point is
quoted on the two mortality labels, since no action is taken at ED disposition
on a 28-day or one-year mortality prediction and inventing one to have a number
to report would be this project's own failure mode.

    python scripts/score_decision_curve.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import ClusterIndex, clustered_statistic_ci  # noqa: E402
from contract import load_base_config  # noqa: E402
from decision_curve import (  # noqa: E402
    flagged_fraction,
    grid,
    net_benefit,
    net_benefit_all,
    selftest,
)
from paths import REPO_ROOT, derived_root  # noqa: E402
from score_calibration import ARMS, pooled_frame, seed_frames  # noqa: E402
from score_arms import ledger_index  # noqa: E402

RESULTS = REPO_ROOT / "results"
CURVES = RESULTS / "decision_curve"
COMPARISONS = REPO_ROOT / "comparisons"
DECLARED = COMPARISONS / "declared_decisions.yaml"


def threshold_declaration_status() -> str:
    """Whether the operating threshold was fixed before any curve ran.

    The threshold had to be chosen from the clinical framing and recorded
    before a decision curve was computed, because a threshold chosen for
    how the resulting number reads is the failure this audit looks for in
    other people's work. The record travels with this repository.
    """
    if not DECLARED.is_file():
        return "missing"
    record = yaml.safe_load(DECLARED.read_text(encoding="utf-8"))
    return str(record.get("net_benefit_threshold", {})
               .get("status", "missing"))


def refuse_if_undeclared() -> list[str]:
    """Every reason this script must not run yet."""
    problems: list[str] = []

    status = threshold_declaration_status()
    if status == "open":
        problems.append(
            "The operating threshold is not recorded as declared. The net-benefit threshold must be fixed from the "
            "clinical framing and recorded BEFORE any curve is computed. A "
            "threshold chosen for how the resulting number reads is the failure "
            "this project audits in other people's work"
        )
    elif status == "missing":
        problems.append("the threshold declaration record is absent")

    config = load_base_config()
    curve = config["evaluation"]["decision_curve"]
    threshold = curve.get("threshold")
    if threshold is None or str(threshold).strip().upper() == "TBD":
        problems.append(
            "configs/base.yaml still reads evaluation.decision_curve.threshold: "
            "TBD. The config and the decision close together"
        )
    else:
        try:
            value = float(threshold)
        except (TypeError, ValueError):
            problems.append(f"the declared threshold {threshold!r} is not a number")
        else:
            if not (0.0 < value < 1.0):
                problems.append(f"the declared threshold {value} is not a "
                                "probability strictly between 0 and 1")
    if not curve.get("threshold_label"):
        problems.append("no threshold_label is declared, so the quoted point "
                        "belongs to no particular decision")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    blockers = refuse_if_undeclared()
    if blockers:
        print("REFUSING TO COMPUTE A DECISION CURVE:", file=sys.stderr)
        for blocker in blockers:
            print(f"  - {blocker}", file=sys.stderr)
        return 2

    problems = selftest()
    if problems:
        print("decision-curve selftest FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    config = load_base_config()
    curve = config["evaluation"]["decision_curve"]
    threshold = float(curve["threshold"])
    quote_labels = set(curve.get("quote_point_at_labels")
                       or [curve["threshold_label"]])
    thresholds = grid(float(curve["grid_lo"]), float(curve["grid_hi"]),
                      float(curve["grid_step"]))
    seeds = list(config["training"]["seeds"])
    labels = [str(x) for x in yaml.safe_load(
        (COMPARISONS / "acq_context_audit.yaml").read_text(encoding="utf-8"))["labels"]]

    derived = derived_root()
    index = ledger_index()
    CURVES.mkdir(parents=True, exist_ok=True)

    curve_rows: list[dict] = []
    point_rows: list[dict] = []

    for label in labels:
        for arm in ARMS:
            frames = seed_frames(arm, label, seeds, index, derived)
            if frames is None:
                raise RuntimeError(f"no registered runs for {arm} at {label}")
            pooled = pooled_frame(frames)
            y = pooled["y_true"].to_numpy().astype(np.int64)
            p = pooled["y_score"].to_numpy(dtype=float)
            cluster = ClusterIndex(pooled["subject_id"].to_numpy())
            prevalence = float(y.mean())

            for t in thresholds:
                curve_rows.append({
                    "arm": arm, "label": label, "unit": "record",
                    "threshold": float(t),
                    "net_benefit": net_benefit(y, p, float(t)),
                    "net_benefit_treat_all": net_benefit_all(y, float(t)),
                    "net_benefit_treat_none": 0.0,
                    "flagged_fraction": flagged_fraction(p, float(t)),
                    "prevalence": prevalence,
                    "n_items": int(len(pooled)),
                    "n_groups": int(cluster.n_groups),
                })

            if label not in quote_labels:
                continue

            # The pre-declared point, with an interval on the net benefit itself.
            interval = clustered_statistic_ci(
                {"cell": cluster},
                cluster.n_groups,
                lambda rows, y=y, p=p: net_benefit(y[rows["cell"]],
                                                   p[rows["cell"]], threshold),
                n_boot=args.n_boot,
                seed=args.seed,
            )
            # Treat-all is not a constant across replicates either: its value
            # depends on the resampled prevalence, so it gets the SAME draw
            # rather than a separately resampled one. The difference between an
            # arm and treat-all is therefore paired, which is the same rule
            # every contrast in this project follows.
            against_all = clustered_statistic_ci(
                {"cell": cluster},
                cluster.n_groups,
                lambda rows, y=y, p=p: (
                    net_benefit(y[rows["cell"]], p[rows["cell"]], threshold)
                    - net_benefit_all(y[rows["cell"]], threshold)
                ),
                n_boot=args.n_boot,
                seed=args.seed,
            )
            per_seed = np.asarray([
                net_benefit(f["y_true"].to_numpy().astype(np.int64),
                            f["y_score"].to_numpy(dtype=float), threshold)
                for f in frames
            ], dtype=float)

            point_rows.append({
                "arm": arm, "label": label, "unit": "record",
                "threshold": threshold,
                "net_benefit": interval["estimate"],
                "ci_lo": interval["ci_lo"],
                "ci_hi": interval["ci_hi"],
                "n_boot": interval["n_boot"],
                "net_benefit_treat_all": net_benefit_all(y, threshold),
                "net_benefit_treat_none": 0.0,
                "vs_treat_all": against_all["estimate"],
                "vs_treat_all_lo": against_all["ci_lo"],
                "vs_treat_all_hi": against_all["ci_hi"],
                "beats_treat_all": bool(against_all["ci_lo"] > 0),
                "beats_treat_none": bool(interval["ci_lo"] > 0),
                "flagged_fraction": flagged_fraction(p, threshold),
                "prevalence": prevalence,
                "mean_predicted": float(p.mean()),
                "seed_nb_min": float(per_seed.min()),
                "seed_nb_max": float(per_seed.max()),
                "seed_nb_spread": float(per_seed.max() - per_seed.min()),
                "n_seeds": len(per_seed),
                "n_items": int(len(pooled)),
                "n_groups": int(cluster.n_groups),
            })

    pd.DataFrame(curve_rows).to_csv(CURVES / "net_benefit.csv", index=False)
    pd.DataFrame(point_rows).to_csv(CURVES / "nb_at_threshold.csv", index=False)

    print(f"wrote {len(curve_rows)} curve points and {len(point_rows)} "
          f"pre-declared-threshold rows into {CURVES}")
    print(f"threshold {threshold} quoted at {sorted(quote_labels)} only")
    print("DECISION CURVE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

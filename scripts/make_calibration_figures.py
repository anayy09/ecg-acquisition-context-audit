#!/usr/bin/env python
"""Reliability curves for the headline arms, drawn from the stored points.

The figures are drawn from `results/calibration/reliability.csv` and never from
the predictions directly, so no number exists only inside an image. That is the
whole of the corresponding check: a figure is a rendering of a table, and if the two can disagree
the table is the one that counts.

One figure per (label, binning scheme). Diagonal is perfect calibration; a curve
below the diagonal is over-prediction.

    python scripts/make_calibration_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import REPO_ROOT  # noqa: E402

CALIBRATION = REPO_ROOT / "results" / "calibration"

#: Drawn in ladder order so a reader can follow the nesting across figures.
ARM_ORDER = [
    "R1_prevalence", "R2_demo", "R3_acqctx_pre", "R3b_acqctx_window",
    "R4_demo_acq", "R5_tabular", "R6_waveform", "R7_waveform_demo_acq",
]


def label_display(label: str) -> str:
    stem = label[len("deterioration_"):] if label.startswith("deterioration_") else label
    if stem.startswith("icu_"):
        return f"ICU admission, {stem[len('icu_'):]}"
    if stem.startswith("mortality_"):
        return f"Mortality, {stem[len('mortality_'):]}"
    return stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else CALIBRATION / "fig"
    out_dir.mkdir(parents=True, exist_ok=True)

    points = pd.read_csv(CALIBRATION / "reliability.csv")
    ece = pd.read_csv(CALIBRATION / "ece.csv")
    if points.empty:
        raise RuntimeError("reliability.csv is empty; run score_calibration.py")

    written: list[Path] = []
    for label in [str(x) for x in dict.fromkeys(points["label"])]:
        for scheme in [str(x) for x in dict.fromkeys(points["scheme"])]:
            subset = points[(points["label"] == label) & (points["scheme"] == scheme)]
            if subset.empty:
                continue
            figure, axis = plt.subplots(figsize=(5.2, 5.0))
            axis.plot([0, 1], [0, 1], color="0.6", linewidth=1,
                      linestyle="--", zorder=1)
            for arm in ARM_ORDER:
                arm_points = subset[subset["arm"] == arm].sort_values("mean_predicted")
                if arm_points.empty:
                    continue
                cell = ece[(ece["arm"] == arm) & (ece["label"] == label)
                           & (ece["scheme"] == scheme)]
                suffix = ""
                if not cell.empty:
                    suffix = f"  (ECE {cell.iloc[0]['ece']:.3f})"
                axis.plot(arm_points["mean_predicted"], arm_points["observed_rate"],
                          marker="o", markersize=3, linewidth=1.2,
                          label=f"{arm}{suffix}", zorder=2)

            top = float(max(subset["mean_predicted"].max(),
                            subset["observed_rate"].max())) * 1.05
            axis.set_xlim(0, top)
            axis.set_ylim(0, top)
            axis.set_xlabel("mean predicted probability")
            axis.set_ylabel("observed frequency")
            axis.set_title(f"{label_display(label)} — {scheme.replace('_', ' ')} bins")
            axis.legend(fontsize=6, loc="upper left", frameon=False)
            figure.tight_layout()

            path = out_dir / f"reliability_{label}_{scheme}.png"
            figure.savefig(path, dpi=200)
            plt.close(figure)
            written.append(path)

    if not args.quiet:
        print(f"wrote {len(written)} reliability figure(s) into {out_dir}")
    print("CALIBRATION FIGURES OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

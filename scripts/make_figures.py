#!/usr/bin/env python
"""The manuscript's figures, drawn only from generated CSV cells.

Review round 1, priority fix 4: the manuscript cited no figure at all, and three
of its arguments are shape arguments that a reader had to assemble from
four-decimal table cells.

The rule the corresponding check established for the reliability curves holds for all four figures
here and the corresponding check enforces it: every plotted point must already
exist as a row of a generated CSV. Nothing is computed in this file except
axis limits. A figure is the last place a number should first appear, because it
is the one place a reader cannot check it.

Sizes are the generic profile in the `manuscript-figures` standards, 89 mm
single column and 183 mm double, because the venue decision is open and no venue is fixed.
That is a deliberate assumption rather than an oversight: the figures are code,
so once the venue is chosen this script regenerates them at that venue's exact
column width and DPI without any hand editing.

    python scripts/make_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import REPO_ROOT  # noqa: E402

#: The style module ships with the manuscript-figures skill pack. It is not
#: pinned in tool_pins.json because it draws rather than computes: no number in
#: this repository depends on it, and a drift would change how a figure looks
#: and nothing else. That is the same test the tool pinning rule applied to the two literature
#: tools, applied here and reaching the opposite conclusion for the same reason.
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import figstyle  # noqa: E402

RESULTS = REPO_ROOT / "results"

#: Overridable so the corresponding check can re-run this generator into a temporary
#: directory rather than over the committed figures. Regenerating the working
#: tree in order to verify the working tree is how this stage deleted paper/ once.
FIGDIR = Path(os.environ.get("ECG_FIG_DIR", str(RESULTS / "fig")))

LABEL_SHORT = {
    "deterioration_icu_24h": "ICU admission, 24 h",
    "deterioration_mortality_28d": "Mortality, 28 d",
    "deterioration_mortality_365d": "Mortality, 365 d",
}

#: The ladder in the order the manuscript presents it, nested input sets.
ARM_ORDER = [
    "R1_prevalence", "R2_demo", "R3_acqctx_pre", "R3b_acqctx_window",
    "R4_demo_acq", "R5_tabular", "R6_waveform", "R7_waveform_demo_acq",
]
ARM_SHORT = {
    "R1_prevalence": "1. Prevalence floor",
    "R2_demo": "2. Age and sex",
    "R3_acqctx_pre": "3. Acquisition context",
    "R3b_acqctx_window": "3b. + index window",
    "R4_demo_acq": "4. Demographics + context",
    "R5_tabular": "5. Vitals and labs",
    "R6_waveform": "6. Waveform",
    "R7_waveform_demo_acq": "7. Waveform + both",
}

#: Arms drawn on the decision curve and the reliability panels. The full set of
#: eight is unreadable at 89 mm, so these are the four the prose discusses, and
#: the caption says the rest are in the tables.
FOCUS_ARMS = ["R3_acqctx_pre", "R5_tabular", "R6_waveform", "R2_demo"]


def fig_ladder(arms: pd.DataFrame) -> tuple[plt.Figure, int]:
    """Figure 1. The nested ladder, one panel per horizon, with clustered CIs."""
    C = figstyle.OKABE_ITO
    labels = list(LABEL_SHORT)
    fig, axes = plt.subplots(
        1, 3, sharey=True,
        figsize=figstyle.fig_size("double", aspect=0.42), layout="constrained")

    plotted = 0
    y = np.arange(len(ARM_ORDER))[::-1]
    for ax, label in zip(axes, labels):
        sub = arms[(arms["label"] == label) & (arms["unit"] == "record")]
        sub = sub.set_index("arm")
        # reindex() would silently turn a missing arm into a NaN row that draws
        # nothing while still counting as a plotted point, which is how the
        # point count stopped measuring what it claimed to measure. A figure
        # that quietly loses an arm must fail rather than render.
        missing = [a for a in ARM_ORDER if a not in sub.index]
        if missing:
            raise RuntimeError(
                f"p3_arms.csv has no record-level row for {missing} at {label}; "
                "refusing to draw a ladder with a silently missing arm")
        sub = sub.reindex(ARM_ORDER)
        est = sub["auroc"].to_numpy(float)
        lo = sub["ci_lo"].to_numpy(float)
        hi = sub["ci_hi"].to_numpy(float)
        if not np.isfinite(est).all():
            raise RuntimeError(f"non-finite AUROC in the ladder at {label}")
        # Colour carries the acquisition-context arm and the waveform arm, the
        # two the claim is about; everything else is neutral. Marker shape is
        # redundant with colour so the figure survives grayscale.
        for i, arm in enumerate(ARM_ORDER):
            if arm == "R3_acqctx_pre":
                colour, marker = C["vermillion"], "o"
            elif arm == "R6_waveform":
                colour, marker = C["blue"], "s"
            else:
                colour, marker = "0.35", "D"
            ax.errorbar(est[i], y[i],
                        xerr=[[est[i] - lo[i]], [hi[i] - est[i]]],
                        fmt=marker, ms=3.0, lw=0.9, capsize=1.8, color=colour)
            plotted += 1
        ax.axvline(0.5, ls="--", lw=0.6, color="0.6", zorder=0)
        ax.set_title(LABEL_SHORT[label], fontsize=7, pad=3)
        ax.set_xlabel("AUROC")
        ax.set_xlim(0.45, 1.0)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    axes[0].set_yticks(y, [ARM_SHORT[a] for a in ARM_ORDER])
    axes[0].set_ylim(-0.6, len(ARM_ORDER) - 0.4)
    return fig, plotted


def fig_horizon(ladder: pd.DataFrame) -> tuple[plt.Figure, int]:
    """Figure 2. The horizon ladder, registered horizons against generating ones."""
    C = figstyle.OKABE_ITO
    fig, ax = plt.subplots(figsize=figstyle.fig_size("single", aspect=0.72),
                           layout="constrained")

    data = ladder.sort_values("horizon_days")
    plotted = 0
    for role, colour, marker, name in (
        ("primary", C["vermillion"], "o", "Registered (4 horizons)"),
        ("generating", "0.45", "^", "Generated the hypothesis"),
    ):
        sub = data[data["role"] == role]
        x = sub["horizon_days"].to_numpy(float)
        est = sub["difference"].to_numpy(float)
        lo = sub["ci_lo"].to_numpy(float)
        hi = sub["ci_hi"].to_numpy(float)
        ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt=marker, ms=3.4,
                    lw=0.9, capsize=1.8, color=colour, label=name, ls="none")
        plotted += len(sub)

    ax.axhline(0.0, ls="--", lw=0.6, color="0.6", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([1, 7, 28, 90, 180, 365])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Outcome horizon (days, log scale)")
    ax.set_ylabel("Acquisition context minus\nage and sex (AUROC)")
    ax.legend(loc="upper right", frameon=False)
    return fig, plotted


def fig_reliability(reliability: pd.DataFrame) -> tuple[plt.Figure, int]:
    """Figure 3. Reliability at the label a decision is actually taken on."""
    C = figstyle.OKABE_ITO
    label = "deterioration_icu_24h"
    colours = {"R3_acqctx_pre": C["vermillion"], "R5_tabular": C["green"],
               "R6_waveform": C["blue"], "R2_demo": "0.45"}
    markers = {"R3_acqctx_pre": "o", "R5_tabular": "D",
               "R6_waveform": "s", "R2_demo": "^"}

    fig, axes = plt.subplots(
        1, 2, figsize=figstyle.fig_size("double", aspect=0.40),
        layout="constrained")

    plotted = 0
    for ax, scheme in zip(axes, ("equal_width", "equal_mass")):
        ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color="0.6", zorder=0)
        for arm in FOCUS_ARMS:
            sub = reliability[(reliability["arm"] == arm)
                              & (reliability["label"] == label)
                              & (reliability["scheme"] == scheme)]
            sub = sub.sort_values("mean_predicted")
            if sub.empty:
                continue
            ax.plot(sub["mean_predicted"], sub["observed_rate"],
                    marker=markers[arm], ms=2.6, lw=0.9, color=colours[arm],
                    label=ARM_SHORT[arm])
            plotted += len(sub)
        ax.set_xlim(0, 0.75)
        ax.set_ylim(0, 0.75)
        ax.set_aspect("equal")
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(scheme.replace("_", " ") + " bins", fontsize=7, pad=3)
    axes[0].set_ylabel("Observed rate")
    axes[1].legend(loc="upper left", frameon=False)
    return fig, plotted


def fig_decision(net_benefit: pd.DataFrame) -> tuple[plt.Figure, int]:
    """Figure 4. Net benefit at the one label with an action at ED disposition."""
    C = figstyle.OKABE_ITO
    label = "deterioration_icu_24h"
    colours = {"R3_acqctx_pre": C["vermillion"], "R5_tabular": C["green"],
               "R6_waveform": C["blue"], "R2_demo": "0.45"}
    styles = {"R3_acqctx_pre": "-", "R5_tabular": "-",
              "R6_waveform": (0, (4, 1.5)), "R2_demo": (0, (1, 1.5))}

    fig, ax = plt.subplots(figsize=figstyle.fig_size("single", aspect=0.72),
                           layout="constrained")

    plotted = 0
    for arm in FOCUS_ARMS:
        sub = net_benefit[(net_benefit["arm"] == arm)
                          & (net_benefit["label"] == label)].sort_values("threshold")
        ax.plot(sub["threshold"], sub["net_benefit"], lw=1.0,
                color=colours[arm], ls=styles[arm], label=ARM_SHORT[arm])
        plotted += len(sub)

    reference = net_benefit[(net_benefit["arm"] == FOCUS_ARMS[0])
                            & (net_benefit["label"] == label)].sort_values("threshold")
    ax.plot(reference["threshold"], reference["net_benefit_treat_all"],
            lw=0.8, color="0.25", ls=(0, (5, 2, 1, 2)), label="Treat all")
    ax.plot(reference["threshold"], reference["net_benefit_treat_none"],
            lw=0.8, color="0.6", label="Treat none")
    plotted += 2 * len(reference)

    threshold = 0.10
    ax.axvline(threshold, lw=0.6, color="0.4", ls=":", zorder=0)
    ax.annotate("declared threshold", xy=(threshold, 0.115),
                xytext=(threshold + 0.035, 0.118), fontsize=5.5,
                color="0.3", va="center")

    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(-0.02, 0.13)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(loc="upper right", frameon=False, fontsize=5.5)
    return fig, plotted


def main() -> int:
    figstyle.apply_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    arms = pd.read_csv(RESULTS / "p3_arms.csv")
    ladder = pd.read_csv(RESULTS / "p3_ladder.csv")
    reliability = pd.read_csv(RESULTS / "calibration" / "reliability.csv")
    net_benefit = pd.read_csv(RESULTS / "decision_curve" / "net_benefit.csv")

    builders = [
        ("fig3_ladder", fig_ladder, (arms,)),
        ("fig4_horizon", fig_horizon, (ladder,)),
        ("fig5_reliability", fig_reliability, (reliability,)),
        ("fig6_decision_curve", fig_decision, (net_benefit,)),
    ]

    total = 0
    for stem, builder, args in builders:
        fig, plotted = builder(*args)
        figstyle.save_figure(fig, str(FIGDIR / stem))
        plt.close(fig)
        total += plotted
        print(f"  {stem:22s} {plotted:5d} plotted points")

    if total == 0:
        print("no figure plotted anything; refusing to report success",
              file=sys.stderr)
        return 1

    try:
        where = FIGDIR.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        # ECG_FIG_DIR pointed outside the repository, which is what
        # the corresponding check does deliberately so it never overwrites the
        # committed figures while verifying them.
        where = str(FIGDIR)
    print(f"wrote {len(builders)} figures into {where} ({total} plotted points)")
    print("FIGURES WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

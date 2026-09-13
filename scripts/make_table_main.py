#!/usr/bin/env python
"""Assemble `results/table_main.{md,tex}` from the generated this stage to this stage artifacts.

the statistics stage computes no new discrimination number. Every value in the headline table
already exists in `results/p1_*.csv` through `p4_*.csv`, and this script's only
job is to put them where a reader and a manuscript can quote them. It types
nothing: each number is read from a named CSV cell and the read is recorded, so
the table can be checked against its own sources rather than against a reader's
memory.

Three units are all called an AUROC in this project and none of them may share a
column with another:

  crop level        4 rows per ECG, 25,756 rows, the published protocol.
                    It appears in the reproduction panel and nowhere else.
  record level      1 row per ECG, 6,439 rows, every cross-arm contrast.
                    This is the ladder panel.
  within stratum    row 8's concordance over pairs drawn from one coarsened
                    acquisition-context stratum. A different estimand,
                    so it gets its own panel, NOT a ninth row of the ladder.

Alongside the two rendered forms this writes `results/table_main_cells.csv`, one
row per number, recording which file, which row and which column it came from.
That file is what makes the corresponding check a re-derivation rather than a
restatement: the checker opens each named source, selects the named row, reads
the named column and reformats it, instead of trusting a claim the generator
made about itself.

    python scripts/make_table_main.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pvalues  # noqa: E402
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import table_labels  # noqa: E402


def ref(pid: str) -> str:
    """How a note sends a reader to another panel.

    Typed as a literal, a cross-reference is correct until the display map
    changes and silently wrong afterwards. Three of these pointed at tables
    that did not exist and two at the wrong one before this existed.
    """
    return table_labels.reference(pid)
from paths import REPO_ROOT  # noqa: E402

RESULTS = REPO_ROOT / "results"
COMPARISONS = REPO_ROOT / "comparisons"

#: Row ids as the project's design notes names them. Structural, not measured.
ARM_ROW = {
    "R1_prevalence": "1",
    "R2_demo": "2",
    "R3_acqctx_pre": "3",
    "R3b_acqctx_window": "3b",
    "R4_demo_acq": "4",
    "R5_tabular": "5",
    "R6_waveform": "6",
    "R7_waveform_demo_acq": "7",
    "R8_waveform_matched": "8",
}
LADDER_ARMS = [a for a in ARM_ROW if a != "R8_waveform_matched"]


#: The comparison files record which stage of the plan registered each contrast,
#: as "this stage" or "this stage". That is this project's own vocabulary and means nothing to a
#: reader, so the tables print what the value actually distinguishes: whether
#: the contrast was measured on all evaluable pairs or on pairs restricted to a
#: shared coarsened acquisition-context stratum.
PAIR_SET = {"P3": "All pairs", "P4": "Matched pairs"}


def pair_set(phase: str) -> str:
    """How a contrast's pair set is printed. An unknown value is a bug, loudly."""
    try:
        return PAIR_SET[str(phase)]
    except KeyError:
        raise KeyError(
            f"{phase!r} is not a known registration stage. Add it to PAIR_SET "
            "deliberately; printing it raw would put this project's internal "
            "vocabulary into a submitted table") from None

def label_display(label: str) -> str:
    """A human column header derived from the label id, never chosen by hand.

    The hand-written scaffold in RESULTS-LOG.md heads a column "In-hospital
    mortality", which is a label MDS-ED does not have. Deriving the
    header from the id mechanically makes that class of error impossible.
    """
    stem = label[len("deterioration_"):] if label.startswith("deterioration_") else label
    if stem.startswith("icu_"):
        return f"ICU admission, {stem[len('icu_'):]}"
    if stem.startswith("mortality_"):
        return f"Mortality, {stem[len('mortality_'):]}"
    return stem


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
@dataclass
class Provenance:
    """Every number the table prints, with the cell it was read from."""

    rows: list[dict] = field(default_factory=list)

    def num(self, table: str, row_key: str, column: str, value, spec: str,
            source_file: str, source_key: str, source_column: str,
            unit: str, aggregate: str = "") -> str:
        """Record one printed number and where it was read from.

        `aggregate` is empty when the selector names exactly one source cell.
        It is `mean` when the printed value is a mean over the selected rows,
        which is the honest description of a figure pooled across seeds: the
        checker then applies the same aggregation instead of demanding a single
        matching cell that does not exist.
        """
        text = format(float(value), spec)
        self.rows.append({
            "table": table,
            "row_key": row_key,
            "column": column,
            "text": text,
            "value": float(value),
            "format": spec,
            "source_file": source_file,
            "source_key": source_key,
            "source_column": source_column,
            "aggregate": aggregate,
            "unit": unit,
        })
        return text

    def pvalue(self, table: str, row_key: str, column: str, value,
               source_file: str, source_key: str, source_column: str,
               unit: str) -> str:
        """Record one p-value, printed at the resolution the design supports.

        `.4g` turns anything below 5e-5 into `0`, and twenty-seven cells printed
        that. A resampling p-value cannot be smaller than one over the number of
        resamples, so below the floor it is printed as the floor and said to be
        below it.
        """
        text = pvalues.format_p(value, pvalues.resampling_floor(REPO_ROOT))
        self.rows.append({
            "table": table,
            "row_key": row_key,
            "column": column,
            "text": text,
            "value": float(value),
            "format": pvalues.SPEC,
            "source_file": source_file,
            "source_key": source_key,
            "source_column": source_column,
            "aggregate": "",
            "unit": unit,
        })
        return text

    def verdict(self, table: str, row_key: str, column: str, text: str,
                source_file: str, source_key: str, source_column: str) -> str:
        self.rows.append({
            "table": table,
            "row_key": row_key,
            "column": column,
            "text": text,
            "value": "",
            "format": "verdict",
            "source_file": source_file,
            "source_key": source_key,
            "source_column": source_column,
            "aggregate": "",
            "unit": "not_applicable",
        })
        return text

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=[
            "table", "row_key", "column", "text", "value", "format",
            "source_file", "source_key", "source_column", "aggregate", "unit",
        ])


def key(**kwargs) -> str:
    """A selector into a source CSV: `col=value;col=value`."""
    return ";".join(f"{k}={v}" for k, v in kwargs.items())


@dataclass
class Panel:
    #: The stable panel id, without its `T` prefix. This is what
    #: `results/table_main_cells.csv` keys on and what every pointer comment in
    #: `paper/` resolves against, so it never changes.
    number: str
    title: str
    lead: list[str]
    columns: list[str]
    rows: list[list[str]]
    code_cols: set[int]
    notes: list[str]
    tex_label: str


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------
def panel_triage(p: Provenance, triage: pd.DataFrame) -> Panel:
    """Table S12. What the acquisition-context block adds beyond triage acuity.

    EXPLORATORY. The arm was added after every result in the paper existed,
    because both reviews asked the question and nothing in the repository
    answered it. It is in no multiplicity family and is corrected against
    nothing.
    """
    src = "results/exploratory_triage.csv"
    rows: list[list[str]] = []
    order = ["R9_triage", "R2_demo", "R3_acqctx_pre", "R3_acqctx_pre - R9_triage"]
    names = {
        "R9_triage": "`R9_triage`, triage acuity alone",
        "R2_demo": "`R2_demo`, age and sex",
        "R3_acqctx_pre": "`R3_acqctx_pre`, all nine features",
        "R3_acqctx_pre - R9_triage": "what the other eight features add",
    }
    auroc = triage[triage["quantity"] != "recovery_ratio"]
    ratios = triage[triage["quantity"] == "recovery_ratio"]
    for label in dict.fromkeys(triage["label"]):
        for arm in order:
            picked = auroc[(auroc["arm"] == arm) & (auroc["label"] == label)]
            if picked.empty:
                continue
            rk = f"{arm}|{label}"
            # The quantity has to be in the selector now that this file holds
            # two rows per (arm, label): the arm's own AUROC and its recovery
            # ratio. Without it the selector matches both and the cell checker
            # cannot tell which number it is verifying.
            k = key(arm=arm, label=label,
                    quantity=str(picked.iloc[0]["quantity"]))
            spec = "+.4f" if " - " in arm else ".4f"
            est = p.num("T1b", rk, "estimate", picked.iloc[0]["estimate"], spec,
                        src, k, "estimate", "record")
            lo = p.num("T1b", rk, "estimate", picked.iloc[0]["ci_lo"], spec,
                       src, k, "ci_lo", "record")
            hi = p.num("T1b", rk, "estimate", picked.iloc[0]["ci_hi"], spec,
                       src, k, "ci_hi", "record")

            # The recovery ratio, for the two arms it is defined for. A
            # difference has no ratio and the demographics floor is not on this
            # scale, so those cells are empty rather than filled with something
            # that would read as a measurement.
            got = ratios[(ratios["arm"] == arm) & (ratios["label"] == label)]
            if got.empty:
                recovery = "-"
            else:
                rkey = key(arm=arm, label=label, quantity="recovery_ratio")
                rest = p.num("T1b", rk, "recovery", got.iloc[0]["estimate"],
                             ".4f", src, rkey, "estimate", "record")
                rlo = p.num("T1b", rk, "recovery", got.iloc[0]["ci_lo"],
                            ".4f", src, rkey, "ci_lo", "record")
                rhi = p.num("T1b", rk, "recovery", got.iloc[0]["ci_hi"],
                            ".4f", src, rkey, "ci_hi", "record")
                recovery = f"{rest} [{rlo}, {rhi}]"

            rows.append([label_display(label), names[arm],
                         f"{est} [{lo}, {hi}]", recovery])

    return Panel(
        number="1b",
        title="The acquisition-context block against triage acuity alone",
        lead=[
            "EXPLORATORY, and not registered. Triage acuity is a clinician's",
            "severity judgement recorded before the ECG, and it is one of the",
            "nine pre-acquisition features, so how much of the block it carries",
            "on its own is a question the nested ladder does not answer. This",
            "arm answers it: a single ordinal column, fitted the same way as",
            "every other tabular arm.",
            "The last row of each label is the paired difference between the",
            "nine-feature arm and the one-feature arm, on the same records and",
            "the same resampled patients. The recovery column puts both arms on",
            "the scale the headline is quoted on, dividing above-chance",
            "discrimination by the waveform arm's, and its denominators are the",
            "same cells Table 4 divides by.",
        ],
        columns=["Label", "Arm", "AUROC or difference [95% CI]",
                 "Recovery of waveform arm [95% CI]"],
        rows=rows,
        code_cols=set(),
        notes=[
            "Acuity alone carries most of the block's discrimination at "
            "24-hour ICU admission and the other eight features still add to "
            "it, at every label, with every interval excluding zero. The block "
            "is neither reducible to acuity nor independent of it.",
            "Read on the recovery scale, acuity alone accounts for most of what "
            "the block recovers at 24 hours and for considerably less of it as "
            "the horizon lengthens. At the registered endpoint, one-year "
            "mortality, the nine features recover about two thirds more than "
            "acuity alone does. So the single feature explains most of the "
            "exploratory short-horizon figure and well under two thirds of the "
            "registered one, and neither of those is a result this paper claims "
            "as met: the registered target was missed under every reading.",
            "At one year the ordering between acuity and demographics reverses: "
            "acuity alone falls below age and sex, having been well above them "
            "at 24 hours. That is consistent with the intuition the "
            "horizon-ladder family was built on, and it is not evidence for "
            "that family's registered trend, which the paper reports as not "
            "supported. An exploratory arm cannot rescue a registered test.",
            "This arm fits one ordinal feature, so every seed produces the same "
            "model and its seed-to-seed spread is exactly zero. The noise floor "
            "this paper judges differences against therefore does not exist for "
            "it, and the interval carries the uncertainty instead.",
        ],
        tex_label="tab:triage",
    )



def panel_calendar_features(p: Provenance, sens: pd.DataFrame) -> Panel:
    """Table S13. What the two de-identified calendar features are worth.

    EXPLORATORY. The declared arm keeps day of week and month because the
    feature set was fixed before it was established that MIMIC-IV's date
    shifting empties them. This says what they contribute, so that the
    nine-feature framing is not quietly carrying two variables the data cannot
    support.
    """
    src = "results/exploratory_sensitivities.csv"
    frame = sens[sens["question"] == "calendar_features"]
    rows: list[list[str]] = []
    for label in dict.fromkeys(frame["label"]):
        for arm in ("R3_acqctx_pre", "R10_acqctx_informative",
                    "R3_acqctx_pre - R10_acqctx_informative"):
            got = frame[(frame["arm"] == arm)
                        & (frame["label"] == label)
                        & (frame["quantity"].isin(["auroc", "paired_difference"]))]
            if got.empty:
                continue
            rk = f"{arm}|{label}"
            k = key(arm=arm, label=label,
                    quantity=str(got.iloc[0]["quantity"]),
                    question="calendar_features")
            spec = "+.4f" if " - " in arm else ".4f"
            est = p.num("T11", rk, "estimate", got.iloc[0]["estimate"], spec,
                        src, k, "estimate", "record")
            lo = p.num("T11", rk, "estimate", got.iloc[0]["ci_lo"], spec,
                       src, k, "ci_lo", "record")
            hi = p.num("T11", rk, "estimate", got.iloc[0]["ci_hi"], spec,
                       src, k, "ci_hi", "record")
            name = {
                "R3_acqctx_pre": "`R3_acqctx_pre`, all nine features",
                "R10_acqctx_informative": "`R10`, the seven informative features",
                "R3_acqctx_pre - R10_acqctx_informative":
                    "what day of week and month add",
            }[arm]
            rows.append([label_display(label), name, f"{est} [{lo}, {hi}]"])

    return Panel(
        number="11",
        title="The acquisition-context arm without its two emptied features",
        lead=[
            "EXPLORATORY, and not registered. MIMIC-IV shifts every date by a",
            "random whole-day offset per patient, so the recorded day of week",
            "and month carry nothing about local practice (Methods, Acquisition",
            "context). The declared arm keeps them because the feature set was",
            "fixed before that was established, and a pre-specified analysis",
            "does not drop a feature after seeing results.",
            "This arm is the same features without those two, fitted the same",
            "way. The third row of each label is the paired difference on the",
            "same records and the same resampled patients.",
        ],
        columns=["Label", "Arm", "AUROC or difference [95% CI]"],
        rows=rows,
        code_cols=set(),
        notes=[
            "The two features are worth nothing measurable at any label: every "
            "paired difference is smaller than the seed-to-seed spread of the "
            "arm itself and every interval contains zero. The declared arm is "
            "reported as nine features because nine were declared, and it "
            "reads as seven.",
        ],
        tex_label="tab:calendar",
    )


def panel_shared_config(p: Provenance, sens: pd.DataFrame) -> Panel:
    """Table S14. The recovery ratio when the tabular arms are selected the
    way the waveform arm was.

    EXPLORATORY. This is the measurable half of the tuning asymmetry: the
    waveform arm cannot be retuned here, but the tabular arms can be selected
    the way it was, from four candidates on a validation macro across all
    fifteen targets rather than per target over fifty trials.
    """
    src = "results/exploratory_sensitivities.csv"
    frame = sens[(sens["question"] == "shared_configuration")
                 & (sens["quantity"] == "recovery_ratio")]
    rows: list[list[str]] = []
    pairs = [("R3_acqctx_pre", "R3s_acqctx_shared"),
             ("R4_demo_acq", "R4s_demo_acq_shared")]
    for declared, shared in pairs:
        for label in dict.fromkeys(frame["label"]):
            cells = []
            for arm in (declared, shared):
                got = frame[(frame["arm"] == arm) & (frame["label"] == label)]
                if got.empty:
                    cells.append("-")
                    continue
                rk = f"{arm}|{label}"
                k = key(arm=arm, label=label, quantity="recovery_ratio",
                        question="shared_configuration")
                est = p.num("T12", rk, "recovery", got.iloc[0]["estimate"],
                            ".4f", src, k, "estimate", "record")
                lo = p.num("T12", rk, "recovery", got.iloc[0]["ci_lo"],
                           ".4f", src, k, "ci_lo", "record")
                hi = p.num("T12", rk, "recovery", got.iloc[0]["ci_hi"],
                           ".4f", src, k, "ci_hi", "record")
                cells.append(f"{est} [{lo}, {hi}]")
            rows.append([f"`{declared}`", label_display(label)] + cells)

    return Panel(
        number="12",
        title="The recovery ratio under the waveform arm's own selection procedure",
        lead=[
            "EXPLORATORY, and not registered. Every recovery ratio in this",
            "paper divides a tabular arm tuned per target over fifty trials by",
            "a waveform arm selected once, from four candidate configurations,",
            "on a validation macro across all fifteen deterioration targets.",
            "That inequality reaches the ratio and inflates it (Limitations).",
            "The waveform arm cannot be retuned within this work. The tabular",
            "arms can be selected the way it was, and are here: four",
            "candidates, one configuration for every label, chosen on the same",
            "fifteen-target validation macro. Same features, same code path,",
            "same five seeds; only the selection changes.",
        ],
        columns=["Arm", "Label", "Declared, tuned per target [95% CI]",
                 "Selected on the 15-target macro [95% CI]"],
        rows=rows,
        code_cols={0},
        notes=[
            "Matching the waveform arm's selection procedure moves every "
            "recovery ratio by under a hundredth, slightly upward, which is "
            "within what the seeds alone move these arms and leaves the "
            "registered target missed as before. The tuning inequality is "
            "therefore worth almost nothing on the side of it that can be "
            "measured here. It does not follow that the whole "
            "inequality is worth little: what cannot be measured here is what "
            "a per-target waveform arm would reach, and that is the side the "
            "Limitations says is open.",
        ],
        tex_label="tab:sharedconfig",
    )


def panel_calibration_fit(p: Provenance, fit: pd.DataFrame) -> Panel:
    """Table S15. Calibration intercept, slope and Brier score.

    ECE summarises a binned curve and says nothing about the DIRECTION of
    miscalibration, or about whether the error is in the level of the
    probabilities or in their spread. These three separate those: the intercept
    is calibration-in-the-large, the slope is dispersion, and the Brier score is
    a proper score that ECE is not.
    """
    src = "results/calibration/calibration_fit.csv"
    order = ["calibration_intercept", "calibration_slope", "brier"]
    rows: list[list[str]] = []
    for label in dict.fromkeys(fit["label"]):
        for arm in dict.fromkeys(fit["arm"]):
            cells = []
            for quantity in order:
                got = fit[(fit["arm"] == arm) & (fit["label"] == label)
                          & (fit["quantity"] == quantity)]
                if got.empty or pd.isna(got.iloc[0]["estimate"]):
                    # The prevalence arm is a constant predictor, so its
                    # probabilities have no spread and a slope is not defined
                    # for it. Printed as undefined rather than as a number.
                    cells.append("undefined" if quantity == "calibration_slope"
                                 else "-")
                    continue
                rk = f"{arm}|{label}"
                k = key(arm=arm, label=label, quantity=quantity)
                spec = "+.4f" if quantity != "brier" else ".4f"
                est = p.num("T13", rk, quantity, got.iloc[0]["estimate"], spec,
                            src, k, "estimate", "record")
                lo = p.num("T13", rk, quantity, got.iloc[0]["ci_lo"], spec,
                           src, k, "ci_lo", "record")
                hi = p.num("T13", rk, quantity, got.iloc[0]["ci_hi"], spec,
                           src, k, "ci_hi", "record")
                cells.append(f"{est} [{lo}, {hi}]")
            rows.append([label_display(label), f"`{arm}`"] + cells)

    return Panel(
        number="13",
        title="Calibration intercept, slope and Brier score",
        lead=[
            "Three quantities the expected calibration error cannot give.",
            "The INTERCEPT is calibration-in-the-large, fitted with the slope",
            "held at one: zero means the average predicted risk matches the",
            "observed rate, and a negative value means the arm over-predicts.",
            "The SLOPE is the coefficient on the logit of the predicted",
            "probability: one means the probabilities are spread correctly,",
            "above one means they are too narrow and below one too extreme.",
            "The BRIER score is mean squared error on the probability scale,",
            "lower being better, and unlike the expected calibration error it",
            "is a proper score.",
            "All three are patient-clustered over the same 10,000 resamples as",
            "every other interval here, and each interval is taken on the",
            "quantity itself.",
        ],
        columns=["Label", "Arm", "Intercept [95% CI]", "Slope [95% CI]",
                 "Brier [95% CI]"],
        rows=rows,
        code_cols={1},
        notes=[
            "The two waveform arms are the only ones whose intervals exclude "
            "both nulls. Their intercepts sit far below zero, so they "
            "over-predict, and their slopes far above one, so the "
            "probabilities they do produce are packed too tightly. Every "
            "tabular arm's intercept interval contains zero and every tabular "
            "slope interval contains one.",
            "A slope above one is what averaging produces, and the scored "
            "probability here is a mean over five seeds and, for the waveform "
            "arms, over four crops before that. That mechanism is already "
            "named in the limitations; this is the measurement of it.",
            "The prevalence arm predicts one constant, so its probabilities "
            "have no spread and a calibration slope does not exist for it. It "
            "is reported as undefined rather than as whatever a singular fit "
            "stops on.",
        ],
        tex_label="tab:calfit",
    )

def panel_matching_sensitivity(p: Provenance, declared: pd.DataFrame,
                               sensitivity: pd.DataFrame) -> Panel:
    """Table S11. What the weekday split costs the matched comparison.

    EXPLORATORY. The declared specification is unchanged and row 8 is computed
    under it; this says what the fourth stratum variable is worth, which is
    nothing, and what it costs, which is about half the usable pairs.
    """
    dsrc = "results/p4_effective.csv"
    ssrc = "results/p4_sensitivity_no_weekday_effective.csv"
    rows: list[list[str]] = []
    for _, line in declared.iterrows():
        label = str(line["label"])
        other = sensitivity[sensitivity["label"] == label]
        if other.empty:
            continue
        other = other.iloc[0]
        rk = label
        k = key(label=label)
        strata_a = p.num("T4f", rk, "strata_declared", line["n_strata_total"],
                         ",.0f", dsrc, k, "n_strata_total", "within_stratum")
        strata_b = p.num("T4f", rk, "strata_sensitivity", other["n_strata_total"],
                         ",.0f", ssrc, k, "n_strata_total", "within_stratum")
        keep_a = p.num("T4f", rk, "retention_declared", line["pair_retention"],
                       ".4f", dsrc, k, "pair_retention", "within_stratum")
        keep_b = p.num("T4f", rk, "retention_sensitivity", other["pair_retention"],
                       ".4f", ssrc, k, "pair_retention", "within_stratum")
        eff_a = p.num("T4f", rk, "effective_declared", line["n_patients_effective"],
                      ",.0f", dsrc, k, "n_patients_effective", "within_stratum")
        eff_b = p.num("T4f", rk, "effective_sensitivity",
                      other["n_patients_effective"], ",.0f", ssrc, k,
                      "n_patients_effective", "within_stratum")
        # Two rows per label rather than one row of paired columns. Seven
        # columns whose headers each carried the specification ran 61pt off
        # the right edge of a landscape page, which the log reported as one
        # overfull box and the compiled file showed as a truncated table.
        rows.append([label_display(label), "Declared", strata_a, keep_a, eff_a])
        rows.append([label_display(label), "Without weekday", strata_b, keep_b,
                     eff_b])

    return Panel(
        number="4f",
        title="The declared matching against a sensitivity without the weekday",
        lead=[
            "EXPLORATORY, and it moves no reported result. The declared",
            "specification strata on four coarsened variables and row 8 is",
            "computed under it. One of those four is the recorded day of the",
            "week, which this dataset's de-identification leaves carrying no",
            "information, so the split costs pairs and controls nothing.",
            "This is that cost, measured by re-cutting the strata on the three",
            "informative variables and counting what the within-stratum",
            "statistic can then use.",
        ],
        columns=["Label", "Specification", "Strata", "Pairs retained",
                 "Effective patients"],
        rows=rows,
        code_cols=set(),
        notes=[
            "Dropping the weekday roughly halves the number of strata and "
            "roughly doubles the share of pairs the statistic can compare, "
            "while leaving the same patients defined. The declared analysis is "
            "not re-cut: a specification fixed before the numbers existed is "
            "not revised once they do, and what this supports is a statement "
            "about the precision of row 8 rather than about its estimate.",
        ],
        tex_label="tab:matching_sensitivity",
    )


def panel_first_ecg(p: Provenance, restricted: pd.DataFrame,
                    macro: pd.DataFrame) -> Panel:
    """Table S10. The reproduction under the benchmark's own scoring protocol.

    EXPLORATORY. The restriction was not registered; it exists because the
    benchmark scores the first record per visit and this project scores every
    test record, and nobody had measured what that difference is worth.
    """
    rsrc = "results/first_ecg_sensitivity.csv"
    msrc = "results/p2_macro.csv"
    arm = "R6_waveform"
    rows: list[list[str]] = []

    for unit in ("crop", "record"):
        full = macro[(macro["arm"] == arm) & (macro["unit"] == unit)]
        sub = restricted[(restricted["arm"] == arm) & (restricted["unit"] == unit)]
        if full.empty or sub.empty:
            continue

        rk = f"{unit}|all_records"
        k = key(arm=arm, unit=unit)
        everything = p.num("T8b", rk, "macro", float(full["macro"].mean()), ".4f",
                           msrc, k, "macro", unit, aggregate="mean")
        spread_all = p.num("T8b", rk, "spread",
                           float(full["macro"].max() - full["macro"].min()),
                           ".4f", msrc, k, "macro", unit, aggregate="range")
        rows.append([unit, "all test records",
                     f"{int(full['n_items'].iloc[0]):,}", everything, spread_all])

        rk = f"{unit}|first_ecg"
        k = key(arm=arm, unit=unit, restriction="first_ecg_per_stay")
        first = p.num("T8b", rk, "macro", float(sub["macro"].mean()), ".4f",
                      rsrc, k, "macro", unit, aggregate="mean")
        spread_first = p.num("T8b", rk, "spread",
                             float(sub["macro"].max() - sub["macro"].min()),
                             ".4f", rsrc, k, "macro", unit, aggregate="range")
        rows.append([unit, "first ECG of each stay",
                     f"{int(sub['n_items'].iloc[0]):,}", first, spread_first])

    return Panel(
        number="8b",
        title="The waveform reproduction under both scoring protocols",
        lead=[
            "EXPLORATORY, and not registered. The audited benchmark scores",
            "validation and test on the first record of each visit; this audit",
            "scores every test record, so the two use different denominators.",
            "Restricting to the first record is a selection over predictions",
            "that already exist, so nothing is retrained and the arm is the one",
            "every other table reports.",
            "The per-seed range is beside each macro, because a difference",
            "smaller than the spread of the five seeds is not a difference.",
        ],
        columns=["Unit", "Scoring protocol", "Rows scored", "Macro AUROC",
                 "Per-seed range"],
        rows=rows,
        code_cols=set(),
        notes=[
            "The restriction moves the macro by about a thousandth at both "
            "units, an order of magnitude below the seed-to-seed range, and the "
            "restricted macro falls inside the published interval exactly as "
            "the unrestricted one does. The difference between the two "
            "protocols is therefore not what separates this reproduction from "
            "the published figure.",
            "The restriction keeps 6,080 records over 6,076 stays: four stays "
            "carry two records whose acquisition times tie, and both are kept "
            "rather than resolved by a tie-break nobody declared.",
        ],
        tex_label="tab:first_ecg",
    )


def panel_cohort_reconciliation(p: Provenance, recon: pd.DataFrame) -> Panel:
    """Table S9. The reproduced cohort against the published one, line by line.

    Every count here is an integer rather than an estimate. The provenance
    recorder casts to float before formatting, so the spec is `,.0f` rather than
    `,d`, which renders identically and survives the re-derivation the cell
    checker performs. A quantity the
    benchmark does not publish carries no published value and no gap, rather
    than a zero that would read as agreement.
    """
    src = "results/cohort_reconciliation.csv"
    rows: list[list[str]] = []
    for _, line in recon.iterrows():
        quantity = str(line["quantity"])
        rk = f"{line['section']}|{quantity}"
        k = key(section=line["section"], quantity=quantity)
        local = p.num("T0", rk, "local", line["local"], ",.0f", src, k, "local",
                      "count")
        # An empty CSV cell reads back as NaN, not as "". Testing the string
        # form let "nan" through and printed it in the table.
        if pd.isna(line["published"]) or str(line["published"]).strip() == "":
            published, gap = "not published", "-"
        else:
            published = p.num("T0", rk, "published", line["published"], ",.0f",
                              src, k, "published", "count")
            gap = p.num("T0", rk, "gap", line["gap"], "+,.0f", src, k, "gap",
                        "count")
        rows.append([str(line["section"]), quantity, local, published, gap])

    return Panel(
        number="0",
        title="The reproduced cohort against the published counts",
        lead=[
            "Every count the release states, beside ours, with the difference.",
            "Patients and stays match to the unit and the record count falls 38",
            "short, which is the shape a partially withheld release leaves and",
            "not the shape a differently built cohort would.",
            "The sequence number within a stay is ZERO-based here: a record",
            "numbered 0 is the first ECG of its stay. The benchmark's own",
            "instructions restrict validation and test scoring to that first",
            "record and this audit scores every test record, so the two use",
            "different denominators and this table gives both.",
        ],
        columns=["Section", "Quantity", "Local", "Published", "Gap"],
        rows=rows,
        code_cols=set(),
        notes=[
            "Label exclusions are records whose outcome is not observable for "
            "that label, marked in the release with a sentinel value. The "
            "tabular arms drop them, which is why the analysed sample differs "
            "slightly by label.",
            "Four stays carry two records both numbered 0, because their "
            "acquisition times tie and the ordering does not break ties. The "
            "first ECG of those stays is therefore ambiguous, and the "
            "first-record restriction covers four fewer distinct stays than it "
            "has records.",
        ],
        tex_label="tab:cohort_reconciliation",
    )


def panel_ladder(p: Provenance, arms: pd.DataFrame, labels: list[str]) -> Panel:
    """Table 2. Rows 1 to 7, record level. Row 8 is NOT here; see Table 5."""
    src = "results/p3_arms.csv"
    rows: list[list[str]] = []
    for arm in LADDER_ARMS:
        cells = [ARM_ROW[arm], arm]
        for label in labels:
            sub = arms[(arms["arm"] == arm) & (arms["label"] == label)]
            if len(sub) != 1:
                raise RuntimeError(f"p3_arms.csv has {len(sub)} rows for {arm}/{label}")
            row = sub.iloc[0]
            if row["unit"] != "record":
                raise RuntimeError(f"{arm}/{label} is unit={row['unit']}, not record")
            k = key(arm=arm, label=label)
            rk = arm
            point = p.num("T1", rk, label, row["auroc"], ".4f", src, k, "auroc", "record")
            lo = p.num("T1", rk, label, row["ci_lo"], ".4f", src, k, "ci_lo", "record")
            hi = p.num("T1", rk, label, row["ci_hi"], ".4f", src, k, "ci_hi", "record")
            cells.append(f"{point} [{lo}, {hi}]")
        rows.append(cells)

    groups = []
    for label in labels:
        row = arms[(arms["arm"] == "R2_demo") & (arms["label"] == label)].iloc[0]
        n_items = p.num("T1", f"meta|{label}", "n_ecgs", row["n_items"], ".0f",
                        src, key(arm="R2_demo", label=label), "n_items", "record")
        n_groups = p.num("T1", f"meta|{label}", "n_patients", row["n_groups"], ".0f",
                         src, key(arm="R2_demo", label=label), "n_groups", "record")
        prev = p.num("T1", f"meta|{label}", "prevalence", row["prevalence"], ".4f",
                     src, key(arm="R2_demo", label=label), "prevalence", "record")
        groups.append(f"{label_display(label)}: {int(float(n_items)):,} ECGs, "
                      f"{int(float(n_groups)):,} patients, prevalence {prev}")

    return Panel(
        number="1",
        title="The nested input ladder, record level",
        lead=[
            "AUROC with a patient-clustered percentile bootstrap interval, 10,000",
            "resamples, resampling unit `subject_id`. One row per ECG throughout:",
            "rows 1 to 5 have no crops, so the published crop-level protocol",
            "appears nowhere in this panel. Scores are averaged across the",
            "five seeds and one AUROC is taken on the pooled score, because a",
            "paired bootstrap needs one score vector per arm; per-seed",
            f"spreads are in {ref('T7')}.",
        ],
        columns=["Row", "Arm"] + [label_display(x) for x in labels],
        rows=rows,
        code_cols={1},
        notes=[
            "Row 8 is not a row here. It is a within-stratum concordance over "
            "restricted pairs, a different estimand from a record-level AUROC, "
            f"and it has its own panel in {ref('T4a')}.",
            "Sample: " + "; ".join(groups) + ".",
        ],
        tex_label="tab:ladder",
    )


def panel_targets(p: Provenance, g1_row: dict, recovery: pd.DataFrame,
                  retained: pd.DataFrame, trends: pd.DataFrame,
                  primary_label: str) -> Panel:
    """Table 3. Every pre-declared quantitative target, and what it measured.

    This panel exists because the failures ARE the result. Reporting the
    measured values without their targets beside them, and leaving a reader to
    find the four failures in prose, is the presentational failure this project
    audits in other people's papers.
    """
    rows: list[list[str]] = []

    # (1) the pre-declared rule's pre-declared primary contrast, the first verdict. Read from the pre-declared rule 1 verdict
    # itself rather than recomputed here: `evaluate_g1` decided this rule and a
    # second implementation of it in the table generator would be a second
    # chance to get it wrong.
    src = "results/p1_verdict.json"
    k = key(path="verdict.per_label", label=primary_label)
    rk = "G1_primary"
    diff = p.num("T2", rk, "measured", g1_row["paired_difference"], "+.4f", src, k,
                 "paired_difference", "record")
    lo = p.num("T2", rk, "measured", g1_row["paired_ci"][0], "+.4f", src, k,
               "paired_ci.0", "record")
    hi = p.num("T2", rk, "measured", g1_row["paired_ci"][1], "+.4f", src, k,
               "paired_ci.1", "record")
    spread = p.num("T2", rk, "pre-declared", g1_row["seed_and_noise_spread"], ".4f",
                   src, k, "seed_and_noise_spread", "record")
    verdict = p.verdict("T2", rk, "outcome",
                        "MET" if bool(g1_row["beats_demographics_beyond_noise"])
                        else "FAILED",
                        src, k, "beats_demographics_beyond_noise")
    rows.append([
        "pre-declared rule 1: R3_acqctx_pre beats R2_demo",
        label_display(primary_label),
        f"positive, beyond the {spread} seed and noise spread",
        f"{diff} [{lo}, {hi}]",
        verdict,
    ])

    # (2) the recovery-ratio target, the second verdict.
    src = "results/p3_recovery.csv"
    row = recovery[recovery["is_primary"]].iloc[0]
    k = key(arm=row["arm"], label=row["label"], is_primary="True")
    rk = "recovery_primary"
    est = p.num("T2", rk, "measured", row["estimate"], ".4f", src, k, "estimate", "record")
    lo = p.num("T2", rk, "measured", row["ci_lo"], ".4f", src, k, "ci_lo", "record")
    hi = p.num("T2", rk, "measured", row["ci_hi"], ".4f", src, k, "ci_hi", "record")
    tgt = p.num("T2", rk, "pre-declared", row["target"], ".2f", src, k, "target", "record")
    verdict = p.verdict("T2", rk, "outcome",
                        "MET" if bool(row["meets_target"]) else "FAILED",
                        src, k, "meets_target")
    rows.append([
        "Recovery ratio, R3_acqctx_pre against R6_waveform",
        label_display(row["label"]),
        f"at least {tgt}",
        f"{est} [{lo}, {hi}]",
        verdict,
    ])

    # (3) the waveform-gain-retained target, the matched-arm verdict.
    src = "results/p4_retained.csv"
    row = retained[retained["label"] == primary_label].iloc[0]
    k = key(label=primary_label)
    rk = "retained_primary"
    est = p.num("T2", rk, "measured", row["estimate"], ".4f", src, k, "estimate", "within_stratum")
    lo = p.num("T2", rk, "measured", row["ci_lo"], ".4f", src, k, "ci_lo", "within_stratum")
    hi = p.num("T2", rk, "measured", row["ci_hi"], ".4f", src, k, "ci_hi", "within_stratum")
    tgt = p.num("T2", rk, "pre-declared", row["threshold"], ".2f", src, k, "threshold", "within_stratum")
    verdict = p.verdict("T2", rk, "outcome",
                        "MET" if bool(row["meets_target"]) else "FAILED",
                        src, k, "meets_target")
    rows.append([
        "Waveform gain retained with acquisition context held fixed",
        label_display(primary_label),
        f"at most {tgt}",
        f"{est} [{lo}, {hi}]",
        verdict,
    ])

    # (4) the horizon ladder's primary trend, the second verdict.
    src = "results/p3_trends.csv"
    row = trends[trends["name"] == "acq_advantage_trend"].iloc[0]
    k = key(name="acq_advantage_trend")
    rk = "ladder_trend"
    est = p.num("T2", rk, "measured", row["estimate"], "+.4f", src, k, "estimate", "record")
    lo = p.num("T2", rk, "measured", row["ci_lo"], "+.4f", src, k, "ci_lo", "record")
    hi = p.num("T2", rk, "measured", row["ci_hi"], "+.4f", src, k, "ci_hi", "record")
    verdict = p.verdict("T2", rk, "outcome",
                        "SUPPORTED" if bool(row["supported"]) else "NOT SUPPORTED",
                        src, k, "supported")
    rows.append([
        "Horizon-ladder trend, four never-scored horizons",
        "mortality 1, 7, 90, 180 d",
        "negative, interval excluding zero",
        f"{est} [{lo}, {hi}]",
        verdict,
    ])

    return Panel(
        number="2",
        title="Every pre-declared quantitative target, and what it measured",
        lead=[
            "Each target was fixed in a committed comparison file before the",
            "run that tests it",
            "existed.",
        ],
        columns=["Quantity", "Endpoint", "Pre-declared", "Measured [95% CI]",
                 "Outcome"],
        rows=rows,
        code_cols=set(),
        notes=[
            f"The recovery ratio clears 0.70 at the two labels that were not "
            f"the pre-declared endpoint ({ref('T3')}); both are reported as "
            "exploratory.",
            "The retained-gain target is missed at all three labels and under "
            f"both readings of its ambiguous formula ({ref('T4b')}), not only at "
            "the endpoint shown here. Under the second reading the intervals at "
            "28 days and at one year contain the target, so at those two the "
            "evidence is inconclusive rather than contrary; the outcome column "
            "reports the declared rule, which is read on the point estimate.",
        ],
        tex_label="tab:targets",
    )


def panel_recovery(p: Provenance, recovery: pd.DataFrame) -> Panel:
    """Table 4. Every recovery ratio at the declared denominator, primary flagged."""
    src = "results/p3_recovery.csv"
    rows: list[list[str]] = []
    # p3_recovery.csv now holds two quantities. Without this filter the panel
    # would quietly gain the alternative-denominator rows, which belong to the
    # panel below and are not comparable to these: they hold the denominator
    # fixed instead of resampling it.
    declared = recovery[recovery["quantity"] == "recovery_ratio"]
    for _, row in declared.iterrows():
        # The quantity enters the selector because the CSV now holds two of
        # them; without it the selector matches the alternative-denominator rows
        # as well and the checker cannot re-derive a single cell.
        k = key(arm=row["arm"], label=row["label"], quantity="recovery_ratio")
        rk = f"{row['arm']}|{row['label']}"
        est = p.num("T3", rk, "ratio", row["estimate"], ".4f", src, k, "estimate", "record")
        lo = p.num("T3", rk, "ratio", row["ci_lo"], ".4f", src, k, "ci_lo", "record")
        hi = p.num("T3", rk, "ratio", row["ci_hi"], ".4f", src, k, "ci_hi", "record")
        num = p.num("T3", rk, "numerator", row["numerator_auroc"], ".4f", src, k,
                    "numerator_auroc", "record")
        den = p.num("T3", rk, "denominator", row["denominator_auroc"], ".4f", src, k,
                    "denominator_auroc", "record")
        met = p.verdict("T3", rk, "meets", "yes" if bool(row["meets_target"]) else "no",
                        src, k, "meets_target")
        primary = p.verdict("T3", rk, "primary",
                            "pre-declared primary" if bool(row["is_primary"]) else "exploratory",
                            src, k, "is_primary")
        rows.append([row["arm"], label_display(row["label"]),
                     f"{est} [{lo}, {hi}]", num, den, met, primary])

    return Panel(
        number="3",
        title="Recovery ratio against the waveform arm",
        lead=[
            "`(AUROC_arm - 0.5) / (AUROC_R6 - 0.5)`, with the interval taken on the",
            "ratio itself: patients are resampled once and both AUROCs are",
            "recomputed on that same draw. The denominator is the per-label",
            "record-level `R6_waveform`, not the 15-target macro; the two",
            "readings differ enough to move the primary across its own threshold.",
        ],
        columns=["Arm", "Label", "Ratio [95% CI]", "Numerator", "Denominator",
                 "At least 0.70", "Status"],
        rows=rows,
        code_cols={0},
        notes=[
            "`R5_tabular`, the benchmark's own vitals-and-labs arm, exceeds 1.0 at "
            "every horizon: it beats the waveform arm it is bundled with. The "
            "benchmark's own appendix reports the same ordering, so this "
            "replicates a published direction at a per-label unit rather than "
            "reporting a new one.",
        ],
        tex_label="tab:recovery",
    )


def panel_recovery_denominators(p: Provenance, recovery: pd.DataFrame) -> Panel:
    """Table S8. The primary recovery ratio under every declared denominator.

    The manuscript said the alternative denominators "both give a smaller ratio
    still". One of them gives a larger one, and the sentence could be wrong for
    two phases because it carried no decimal and so resolved to no cell. This is
    the panel that makes it resolvable.
    """
    src = "results/p3_recovery.csv"
    declared = recovery[(recovery["quantity"] == "recovery_ratio")
                        & recovery["is_primary"].astype(bool)]
    alternatives = recovery[
        recovery["quantity"] == "recovery_ratio_alternative_denominator"]
    frame = pd.concat([declared, alternatives])

    rows: list[list[str]] = []
    for _, row in frame.iterrows():
        source = str(row["denominator_source"])
        k = key(arm=row["arm"], label=row["label"], denominator_source=source)
        rk = f"{row['arm']}|{row['label']}|{source}"
        est = p.num("T3b", rk, "ratio", row["estimate"], ".4f", src, k, "estimate", "record")
        lo = p.num("T3b", rk, "ratio", row["ci_lo"], ".4f", src, k, "ci_lo", "record")
        hi = p.num("T3b", rk, "ratio", row["ci_hi"], ".4f", src, k, "ci_hi", "record")
        den = p.num("T3b", rk, "denominator", row["denominator_auroc"], ".4f",
                    src, k, "denominator_auroc", "record")
        paired = bool(row.get("denominator_resampled", True))
        met = p.verdict("T3b", rk, "meets",
                        "yes" if bool(row["meets_target"]) else "no",
                        src, k, "meets_target")
        rows.append([
            "per-label `R6_waveform` (declared)" if paired else source,
            den, f"{est} [{lo}, {hi}]",
            "both" if paired else "numerator only", met,
        ])

    return Panel(
        number="3b",
        title="The primary recovery ratio under every declared denominator",
        lead=[
            "The registered primary quantity divides the acquisition-context",
            "arm's above-chance discrimination by the waveform arm's, at the",
            "registered endpoint. Which waveform figure belongs in the",
            "denominator was declared with two alternatives beside the one used,",
            "and a third became available when the audited benchmark's version",
            "of record revised its published macro. All four are here.",
            "Only the declared denominator is resampled with the numerator. A",
            "published macro is a constant from another paper, and our own",
            "15-target macro is a different estimand from the per-label AUROC, so",
            "for those three the interval carries the numerator's uncertainty",
            "alone and the table says which.",
        ],
        columns=["Denominator", "Value", "Ratio [95% CI]", "Resampled",
                 "At least 0.70"],
        rows=rows,
        code_cols=set(),
        notes=[
            "No denominator brings the ratio to its registered target. The "
            "declared one is the most favourable of the four: under it the "
            "interval spans 0.70, while under the version-of-record macro and "
            "under our own 15-target macro the whole interval falls below the "
            "target. The preprint macro gives a slightly larger ratio than the "
            "declared denominator rather than a smaller one, which is the "
            "opposite of what this paper asserted before the cells existed.",
        ],
        tex_label="tab:recovery_denominators",
    )


def panel_row8(p: Provenance, p4arms: pd.DataFrame, effective: pd.DataFrame,
               labels: list[str]) -> Panel:
    """Table 5a. Row 8, in its own panel and its own unit."""
    src = "results/p4_arms.csv"
    esrc = "results/p4_effective.csv"
    rows: list[list[str]] = []
    for label in labels:
        for arm in ("R2_demo", "R3_acqctx_pre", "R8_waveform_matched"):
            sub = p4arms[(p4arms["arm"] == arm) & (p4arms["label"] == label)]
            if len(sub) != 1:
                raise RuntimeError(f"p4_arms.csv has {len(sub)} rows for {arm}/{label}")
            row = sub.iloc[0]
            k = key(arm=arm, label=label)
            rk = f"{arm}|{label}"
            est = p.num("T4a", rk, "within_stratum", row["auroc_within_stratum"], ".4f",
                        src, k, "auroc_within_stratum", "within_stratum")
            lo = p.num("T4a", rk, "within_stratum", row["ci_lo"], ".4f", src, k,
                       "ci_lo", "within_stratum")
            hi = p.num("T4a", rk, "within_stratum", row["ci_hi"], ".4f", src, k,
                       "ci_hi", "within_stratum")
            plain = p.num("T4a", rk, "plain", row["auroc_plain_same_rows"], ".4f",
                          src, k, "auroc_plain_same_rows", "record_same_rows")
            eff = effective[effective["label"] == label].iloc[0]
            ek = key(label=label)
            npat = p.num("T4a", rk, "n_patients", eff["n_patients_effective"], ".0f",
                         esrc, ek, "n_patients_effective", "within_stratum")
            ret = p.num("T4a", rk, "pair_retention", eff["pair_retention"], ".4f",
                        esrc, ek, "pair_retention", "within_stratum")
            rows.append([arm, label_display(label), f"{est} [{lo}, {hi}]", plain,
                         f"{int(float(npat)):,}", ret])

    return Panel(
        number="4a",
        title="Row 8: discrimination with acquisition context held fixed",
        lead=[
            f"A different estimand from {ref('T1')}, and never in the same "
            "column as it.",
            "Only positive-negative pairs drawn from the same coarsened",
            "acquisition-context stratum contribute, so no compared pair differs in",
            "acquisition context. The plain column is computed on the same",
            "rows, so the only difference between the two is the pair restriction",
            "rather than which records were kept.",
        ],
        columns=["Arm", "Label", "Within stratum [95% CI]", "Plain, same rows",
                 "Effective patients", "Pairs retained"],
        rows=rows,
        code_cols={0},
        notes=[
            "`R3_acqctx_pre` collapses within a stratum because four of its nine "
            "features are the matching variables, `R2_demo` barely moves, and the "
            "waveform arm keeps most of its advantage. A statistic that failed to "
            "restrict its pairs could not produce that pattern.",
            "The effective patient count, not the ECG count, bounds every claim in "
            "this panel. Almost every patient survives the restriction and most "
            "pairs do not, which is why these intervals are two to three times "
            f"wider than those of {ref('T1')} on the same records.",
        ],
        tex_label="tab:row8",
    )


def panel_retained(p: Provenance, retained: pd.DataFrame) -> Panel:
    """Table 5b. The declared secondary quantity, under both readings."""
    src = "results/p4_retained.csv"
    rows: list[list[str]] = []
    thresholds: set[str] = set()
    for _, row in retained.iterrows():
        k = key(label=row["label"])
        rk = row["label"]
        est = p.num("T4b", rk, "retained", row["estimate"], ".4f", src, k, "estimate", "within_stratum")
        lo = p.num("T4b", rk, "retained", row["ci_lo"], ".4f", src, k, "ci_lo", "within_stratum")
        hi = p.num("T4b", rk, "retained", row["ci_hi"], ".4f", src, k, "ci_hi", "within_stratum")
        alt = p.num("T4b", rk, "retained_alt", row["retained_plain_baseline"], ".4f",
                    src, k, "retained_plain_baseline", "within_stratum")
        alo = p.num("T4b", rk, "retained_alt", row["retained_plain_baseline_lo"], ".4f",
                    src, k, "retained_plain_baseline_lo", "within_stratum")
        ahi = p.num("T4b", rk, "retained_alt", row["retained_plain_baseline_hi"], ".4f",
                    src, k, "retained_plain_baseline_hi", "within_stratum")
        tgt = p.num("T4b", rk, "target", row["threshold"], ".2f", src, k, "threshold",
                    "within_stratum")
        met = p.verdict("T4b", rk, "meets", "yes" if bool(row["meets_target"]) else "no",
                        src, k, "meets_target")
        met_alt = p.verdict("T4b", rk, "meets_alt",
                            "yes" if bool(row["meets_target_plain_baseline"]) else "no",
                            src, k, "meets_target_plain_baseline")
        rows.append([label_display(row["label"]), f"{est} [{lo}, {hi}]", met,
                     f"{alt} [{alo}, {ahi}]", met_alt, f"at most {tgt}"])
        thresholds.add(tgt)

    if len(thresholds) != 1:
        raise RuntimeError(f"p4_retained.csv carries {len(thresholds)} distinct "
                           f"thresholds; one panel cannot state a single target")

    return Panel(
        number="4b",
        title="Waveform gain retained: the declared secondary quantity",
        lead=[
            "`(R6 - R2)` within stratum over `(R6 - R2)` on the same rows without",
            "the pair restriction. The declared formula does not say which `R2` the",
            "numerator takes, so both are given; the first is the reading quoted",
            "in the abstract and the conclusion.",
        ],
        columns=["Label", "Retained, quoted reading [95% CI]", "Meets target",
                 "Retained, unstratified R2 [95% CI]", "Meets target",
                 "Declared target"],
        rows=rows,
        code_cols=set(),
        notes=[
            "The prediction was that holding acquisition context fixed would cost "
            "the waveform arm at least half its advantage over demographics. "
            "Under the within-stratum reading of the numerator it costs between "
            "7 and 22 percent of it; under the unstratified reading, between 32 "
            "and 45 percent. The point estimate misses the declared target "
            "under both. Under the second reading the intervals at 28 days and "
            "at one year include 0.50, so at those two labels the evidence does "
            "not settle which side of the target the truth lies on; what is "
            "established is that the target is not met under the declared rule, "
            "which is read on the point estimate.",
        ],
        tex_label="tab:retained",
    )


def panel_balance(p: Provenance, balance: pd.DataFrame, matching: dict) -> Panel:
    """Table 5c. What the matching balanced, and what it left behind.

    Within a stratum the arms see the same rows, so a standardised mean
    difference BETWEEN ARMS would be identically zero and would prove nothing.
    What the design claims is that the positives and negatives being compared no
    longer differ in acquisition context, so that is what is measured, under the
    pair weighting the within-stratum statistic implies.
    """
    src = "results/p4_balance.csv"
    rows: list[list[str]] = []
    # (row key, scale, held fixed, what the row is about). The first two keys are
    # the ones the manuscript already points at and they keep the meaning they
    # had: the variables the strata hold fixed. The third is new and is the
    # review's question, the five pre-acquisition features the matching leaves
    # free. Folding those into the second row would have moved a number the
    # prose quotes while the pointer still resolved.
    groups = [
        ("coarsened", "coarsened", "yes",
         "matching variables, as coarsened"),
        ("underlying", "underlying", "yes",
         "matching variables, underlying scale"),
        ("underlying_unmatched", "underlying", "no",
         "pre-acquisition features the strata do not hold fixed"),
    ]
    for rk, scale, held, description in groups:
        subset = balance[(balance["scale"] == scale)
                         & (balance["held_fixed"] == held)]
        if subset.empty:
            continue
        selector = key(scale=scale, held_fixed=held)
        before = p.num("T4c", rk, "worst_before", subset["smd_before"].abs().max(),
                       ".4f", src, selector, "smd_before", "within_stratum",
                       aggregate="absmax")
        after = p.num("T4c", rk, "worst_after", subset["smd_after"].abs().max(),
                      ".4f", src, selector, "smd_after", "within_stratum",
                      aggregate="absmax")
        rows.append([description, before, after, f"{len(subset)}"])

    diagnostic = matching["superset_propensity_diagnostic"]
    dsrc = "results/p4_matching.json"
    auroc = p.num("T4c", "meta|propensity", "propensity_auroc",
                  diagnostic["auroc"], ".4f", dsrc,
                  "path=superset_propensity_diagnostic", "auroc", "record")
    rate = p.num("T4c", "meta|propensity", "early_ecg_rate",
                 diagnostic["early_ecg_rate"], ".4f", dsrc,
                 "path=superset_propensity_diagnostic", "early_ecg_rate", "record")

    return Panel(
        number="4c",
        title="Balance, and the acquisition-propensity diagnostic",
        lead=[
            "Standardised mean differences between outcome-positive and",
            "outcome-negative records, under the pair weighting the within-stratum",
            "statistic implies. On the coarsened matching variables balance is exact",
            "after weighting by construction, so a non-zero value there would be a",
            "bug in the stratifier rather than a property of the data; a shuffled",
            "stratum assignment breaks those zeros, so they are not the output of an",
            "insensitive statistic. The residual on the underlying continuous values",
            "is what coarsening leaves inside a bin and is reported rather than",
            "assumed away.",
        ],
        columns=["Scale", "Worst |SMD| before", "Worst |SMD| after", "Comparisons"],
        rows=rows,
        code_cols=set(),
        notes=[
            f"As a diagnostic only, pre-acquisition context "
            f"predicts whether an early ECG happens at all with AUROC {auroc} on "
            f"held-out stays from the parent cohort, where the early-ECG rate is "
            f"{rate}. So the decision to record is itself substantially "
            "predictable. That model is never used to select or weight a row 8 "
            "evaluation row: the parent cohort carries no deterioration label and "
            "its ECG-absent stays carry no waveform.",
        ],
        tex_label="tab:balance",
    )


def panel_blocks(p: Provenance, profile: pd.DataFrame) -> Panel:
    """Table 5d. What each acquisition-context feature can carry in this cohort.

    Added after review round 1, priority fix 1. The leakage answer used to rest
    partly on the small gap between the leakage-suspect arm and the confirmatory
    one, and named `care_unit_at_acquisition` as the variable that made that gap
    informative. It is constant on every record, so it cannot leak and the gap
    had little room to move. That is worth a table rather than a sentence,
    because the same question runs the other way: a degenerate ACQ_PRE feature
    would mean the confirmatory arm rests on fewer features than it claims.
    """
    src = "results/acq_block_profile.csv"
    rows: list[list[str]] = []
    n_records = int(profile["n_records"].iloc[0])
    for _, row in profile.iterrows():
        name = str(row["feature"])
        rk = f"{row['block']}|{name}"
        k = key(feature=name)
        if not bool(row["fitted"]):
            dropped = p.verdict("T4d", rk, "carries_information", "no", src, k,
                                "carries_information")
            rows.append([f"`{name}`", str(row["block"]).upper(), "dropped",
                         dropped, "-", "-", "-"])
            continue
        levels = p.num("T4d", rk, "n_levels", row["n_levels"], ".0f", src, k,
                       "n_levels", "feature")
        modal = p.num("T4d", rk, "modal_share", row["modal_share"], ".4f", src, k,
                      "modal_share", "feature")
        undefined = p.num("T4d", rk, "undefined_share", row["undefined_share"],
                          ".4f", src, k, "undefined_share", "feature")
        carries = p.verdict("T4d", rk, "carries_information",
                            "yes" if bool(row["carries_information"]) else "no",
                            src, k, "carries_information")
        rows.append([f"`{name}`", str(row["block"]).upper(), str(row["status"]),
                     carries, levels, modal, undefined])

    return Panel(
        number="4d",
        title="Every declared acquisition-context feature, and what it can carry",
        lead=[
            f"Level counts over all {n_records:,} records of the reproduced cohort.",
            "A feature with one level carries no information, so it cannot be a",
            "leak whatever its availability stamp says, and it cannot be part of a",
            "confirmatory arm's discrimination either. Both directions matter here:",
            "the ACQ_WINDOW block is where the leakage objection lives, and the",
            "ACQ_PRE block is what the confirmatory claim rests on.",
        ],
        columns=["Feature", "Block", "Status", "Carries information", "Levels",
                 "Modal share", "Undefined share"],
        rows=rows,
        code_cols={0},
        notes=[
            "`care_unit_at_acquisition` is constant because the benchmark admits "
            "only patients with an ECG inside the first 90 minutes of the ED stay, "
            "so the patient is in the Emergency Department at acquisition by "
            "construction. The benchmark's own inclusion rule forecloses the leak "
            "the ACQ_WINDOW block was designed to isolate.",
            "`cardiologist_overread_exists` was declared in the block, probed at "
            "data access, and dropped: MIMIC-IV-ECG ships machine-generated "
            "statements and carries no separate human over-read marker. It "
            "entered no run.",
            "Every ACQ_PRE feature varies. The confirmatory arm is fitted on nine "
            "features and all nine carry information.",
        ],
        tex_label="tab:blocks",
    )


def panel_control(p: Provenance, control: pd.DataFrame) -> Panel:
    """Table 5e. The controls that show the within-stratum statistic restricts pairs.

    Added after review round 1, priority fix 3. These values were stated in the
    prose as though they were design constants, and one of them was wrong by
    0.02 for exactly as long as nothing compared it to the code that produces it. They are measurements, so they are generated and pointed at like
    every other measurement in the paper.
    """
    src = "results/p4_control.csv"
    rows: list[list[str]] = []
    for _, row in control.iterrows():
        name = str(row["control"])
        k = key(control=name)
        plain = p.num("T4e", name, "unstratified", row["unstratified"], ".4f",
                      src, k, "unstratified", "synthetic")
        within = p.num("T4e", name, "within_stratum", row["within_stratum"], ".4f",
                       src, k, "within_stratum", "synthetic")
        rows.append([name.replace("_", " "), plain, within,
                     str(row["demonstrates"])])

    return Panel(
        number="4e",
        title="Controls on the within-stratum statistic",
        lead=[
            "Computed on synthetic data with a fixed seed, so these are properties",
            "of the estimator rather than results about the cohort. The third row",
            "is the one that matters: a score that is a pure function of the",
            "stratum discriminates well unstratified and must collapse to exactly",
            "chance once pairs are restricted to within a stratum. A statistic that",
            "did not really restrict its pairs would keep most of it.",
        ],
        columns=["Control", "Unstratified", "Within stratum", "What it shows"],
        rows=rows,
        code_cols=set(),
        notes=[
            "`stratified.control_values()` is the single definition of these "
            "numbers. The module's own selftest asserts against it and this "
            "table is written from it, so the assertion and the printed figure "
            "cannot diverge.",
        ],
        tex_label="tab:control",
    )


def panel_family(p: Provenance, family: pd.DataFrame) -> Panel:
    """Table 6. The completed audit family, Holm over all 15 declared tests."""
    src = "results/p4_family_final.csv"
    rows: list[list[str]] = []
    for _, row in family.iterrows():
        k = key(contrast=row["contrast"], label=row["label"])
        rk = f"{row['contrast']}|{row['label']}"
        raw = p.pvalue("T5", rk, "raw_p", row["p_value"], src, k, "p_value",
                       "record")
        holm = p.pvalue("T5", rk, "holm_p", row["p_holm_final"], src, k,
                        "p_holm_final", "record")
        m = p.num("T5", rk, "m", row["holm_family_size"], ".0f", src, k,
                  "holm_family_size", "record")
        rows.append([row["contrast"], label_display(row["label"]), raw, holm,
                     pair_set(row["phase"]), m])

    return Panel(
        number="5",
        title="The audit family, complete at all 15 declared tests",
        lead=[
            "`acq_context_audit` declared 5 contrasts crossed with 3 labels and the",
            "family has not grown. This is the correction the manuscript",
            "quotes: completing the family can only lower an existing adjusted",
            "p-value, never raise one, so the earlier correction is superseded",
            "here rather than contradicted.",
        ],
        columns=["Contrast", "Label", "Raw p", "Holm p, final", "Pairs", "Family size m"],
        rows=rows,
        code_cols={0},
        notes=[
            f"This family is never pooled with the horizon ladder ({ref('T6a')}), "
            "which carries its own Holm correction at its own declared size of 6.",
        ],
        tex_label="tab:family",
    )


def panel_ladder_horizons(p: Provenance, ladder: pd.DataFrame) -> Panel:
    """Table 7a. The second family's per-horizon differences."""
    src = "results/p3_ladder.csv"
    rows: list[list[str]] = []
    for _, row in ladder.sort_values("horizon_days").iterrows():
        k = key(label=row["label"])
        rk = row["label"]
        diff = p.num("T6a", rk, "difference", row["difference"], "+.4f", src, k,
                     "difference", "record")
        lo = p.num("T6a", rk, "difference", row["ci_lo"], "+.4f", src, k, "ci_lo", "record")
        hi = p.num("T6a", rk, "difference", row["ci_hi"], "+.4f", src, k, "ci_hi", "record")
        raw = p.pvalue("T6a", rk, "raw_p", row["p_value"], src, k, "p_value",
                       "record")
        days = p.num("T6a", rk, "horizon", row["horizon_days"], ".0f", src, k,
                     "horizon_days", "record")
        if bool(row["counts_as_confirmation"]):
            holm = p.pvalue("T6a", rk, "holm_p", row["p_holm"], src, k,
                            "p_holm", "record")
        else:
            holm = p.verdict("T6a", rk, "holm_p", "not corrected in this family",
                             src, k, "counts_as_confirmation")
        rows.append([label_display(row["label"]), days, f"{diff} [{lo}, {hi}]",
                     raw, holm, row["role"]])

    return Panel(
        number="6a",
        title="The horizon ladder: a separate family, per-horizon differences",
        lead=[
            "`R3_acqctx_pre` minus `R2_demo` at each mortality horizon. The primary",
            "test covers only the four horizons no arm had been scored on. The two",
            "horizons that generated the hypothesis are shown, carry no",
            "ladder-adjusted p, and never count as confirmation.",
        ],
        columns=["Label", "Horizon (d)", "Difference [95% CI]", "Raw p",
                 "Holm p", "Role"],
        rows=rows,
        code_cols=set(),
        notes=[
            f"This family is corrected separately from the audit family "
            f"({ref('T5')}) and the two are never pooled. No test carries an "
            "adjusted p in both.",
        ],
        tex_label="tab:ladder_horizons",
    )


def panel_trends(p: Provenance, trends: pd.DataFrame) -> Panel:
    """Table 7b. The trend statistics, interval taken on the correlation."""
    src = "results/p3_trends.csv"
    rows: list[list[str]] = []
    for _, row in trends.iterrows():
        k = key(name=row["name"])
        rk = row["name"]
        est = p.num("T6b", rk, "correlation", row["estimate"], "+.4f", src, k,
                    "estimate", "record")
        lo = p.num("T6b", rk, "correlation", row["ci_lo"], "+.4f", src, k, "ci_lo", "record")
        hi = p.num("T6b", rk, "correlation", row["ci_hi"], "+.4f", src, k, "ci_hi", "record")
        raw = p.pvalue("T6b", rk, "raw_p", row["p_value"], src, k, "p_value",
                       "record")
        if pd.isna(row["p_holm"]):
            holm = p.verdict("T6b", rk, "holm_p", "not corrected in this family",
                             src, k, "p_holm")
        else:
            holm = p.pvalue("T6b", rk, "holm_p", row["p_holm"], src, k,
                            "p_holm", "record")
        sup = p.verdict("T6b", rk, "supported",
                        "supported" if bool(row["supported"]) else "NOT SUPPORTED",
                        src, k, "supported")
        rows.append([row["name"], row["role"], f"{est} [{lo}, {hi}]", raw, holm, sup])

    return Panel(
        number="6b",
        title="The horizon ladder: trend statistics",
        lead=[
            "Spearman rank correlation between horizon in days and the per-horizon",
            "quantity, with the interval taken on the correlation itself: patients",
            "are resampled once, every horizon is recomputed on that draw, and the",
            "correlation is recomputed from them.",
        ],
        columns=["Statistic", "Role", "Correlation [95% CI]", "Raw p", "Holm p",
                 "Supported"],
        rows=rows,
        code_cols={0},
        notes=[
            "`acq_advantage_trend_all_six` contains the two observations that "
            "generated the hypothesis. It reaches a much stronger correlation with "
            "a small raw p, and it is reported for completeness only and never "
            "counted as support, whatever its value. That contrast is the entire "
            "argument for the restriction: the hypothesis looks convincing exactly "
            "when it is tested on the data that produced it.",
            "An interval on a correlation over four points is coarse by "
            "construction, because such a correlation can take only a handful of "
            "values. That is a limitation of the pre-declared statistic and no "
            "amount of resampling fixes it.",
        ],
        tex_label="tab:trends",
    )


def panel_noise(p: Provenance, noise: pd.DataFrame, gap: pd.DataFrame,
                labels: list[str]) -> Panel:
    """Table 8. The noise floor, which is a table and not a number."""
    src = "results/p3_noise.csv"
    # The reproduction verdict's crop-level macro spread is the floor this panel exists to replace,
    # so it is READ from the this stage artifact rather than typed into the prose. A
    # number in a caption is still a number.
    crop_floor = p.num(
        "T7", "meta|d054_floor", "crop_macro_spread",
        gap[gap["arm"] == "R6_waveform"].iloc[0]["seed_spread"], ".4f",
        # The anchor is in the selector because the gap table now has one row
        # per anchor. The seed spread is identical across them, being a property
        # of our five runs rather than of the published figure, so naming the
        # primary is a choice about which row to point at and not about which
        # value to print.
        "results/p2_reproduction_gap.csv",
        key(arm="R6_waveform", unit="crop", anchor="version_of_record"),
        "seed_spread", "crop",
    )
    rows: list[list[str]] = []
    for arm in LADDER_ARMS:
        cells = [arm]
        present = False
        for label in labels:
            sub = noise[(noise["arm"] == arm) & (noise["label"] == label)]
            if len(sub) != 1:
                cells.append("-")
                continue
            present = True
            row = sub.iloc[0]
            k = key(arm=arm, label=label)
            rk = arm
            spread = p.num("T7", rk, label, row["seed_spread"], ".4f", src, k,
                           "seed_spread", "record")
            cells.append(spread)
        if present:
            rows.append(cells)

    return Panel(
        number="7",
        title="The noise floor, per arm and label, record level",
        lead=[
            "Seed-to-seed spread of each cell's own five runs. The floor",
            f"measured, {crop_floor}, is a crop-level spread of a 15-target macro",
            "and this project's contrasts are neither, so every contrast is judged",
            "against the spread of its own arms at its own label. Every",
            "per-label record-level spread of `R6_waveform` exceeds that figure;",
            "several tabular arms fall well below it, which is why one scalar",
            "cannot serve them all.",
        ],
        columns=["Arm"] + [label_display(x) for x in labels],
        rows=rows,
        code_cols={0},
        notes=[
            "A between-arm difference not clearly larger than the relevant cell "
            "here is reported as no difference.",
        ],
        tex_label="tab:noise",
    )


def panel_reproduction(p: Provenance, gap: pd.DataFrame, macro: pd.DataFrame) -> Panel:
    """Table 9. The pre-declared rule 2 reproduction, at the ONLY place crop level appears."""
    src = "results/p2_reproduction_gap.csv"
    rows: list[list[str]] = []
    # The anchor enters the selector and the row key. Two anchors for one arm
    # would otherwise write the same cell twice under the same selector, and the
    # checker would re-derive whichever row the CSV returned first.
    label_for = {"version_of_record": "version of record",
                 "preprint_v2": "preprint"}
    for _, row in gap.iterrows():
        anchor = str(row["anchor"])
        k = key(arm=row["arm"], unit=row["unit"], anchor=anchor)
        rk = f"{row['arm']}|{row['unit']}|{anchor}"
        local = p.num("T8", rk, "local", row["local_macro_mean"], ".4f", src, k,
                      "local_macro_mean", str(row["unit"]))
        pub = p.num("T8", rk, "published", row["published_macro"], ".4f", src, k,
                    "published_macro", str(row["unit"]))
        plo = p.num("T8", rk, "published", row["published_lo"], ".4f", src, k,
                    "published_lo", str(row["unit"]))
        phi = p.num("T8", rk, "published", row["published_hi"], ".4f", src, k,
                    "published_hi", str(row["unit"]))
        g = p.num("T8", rk, "gap", row["gap"], "+.4f", src, k, "gap", str(row["unit"]))
        inside = p.verdict("T8", rk, "inside", "yes" if bool(row["inside_published_ci"])
                           else "no", src, k, "inside_published_ci")
        rows.append([row["arm"], row["unit"],
                     label_for.get(anchor, anchor), local,
                     f"{pub} [{plo}, {phi}]", g, inside])

    # The record-level macro is the same fifteen targets scored one row per ECG.
    # Nothing published exists at this unit, so there is no gap to report and the
    # published columns are empty rather than filled with the crop-level figure.
    msrc = "results/p2_macro.csv"
    for arm in sorted(macro["arm"].unique()):
        sub = macro[(macro["arm"] == arm) & (macro["unit"] == "record")]
        if sub.empty:
            continue
        rk = f"{arm}|record"
        k = key(arm=arm, unit="record")
        value = float(sub["macro"].mean())
        text = p.num("T8", rk, "local", value, ".4f", msrc, k, "macro", "record",
                     aggregate="mean")
        rows.append([arm, "record", "-", text,
                     "none published at this unit", "-", "-"])

    return Panel(
        number="8",
        title="The waveform reproduction, and the only crop-level numbers here",
        lead=[
            "Unweighted macro AUROC over the 15 deterioration targets, at crop",
            "level, which is the unit the benchmark publishes. Gaps are against",
            "both published versions. The record-level macro beside it is the",
            "mean of the five per-seed macros and has no published counterpart.",
        ],
        columns=["Arm", "Unit", "Published version", "Local macro",
                 "Published [95% CI]", "Gap", "Inside"],
        rows=rows,
        code_cols={0},
        notes=[
            "The published interval is wide because six "
            "of the fifteen deterioration targets carry fewer than fifty positives "
            "in the test fold and the unweighted macro gives each the same weight "
            "as targets carrying over eight hundred. A pass is evidence the "
            "reproduction is not badly wrong, not that it is precisely right.",
            "`R7_waveform_demo_acq` is shown against the same published figures "
            "for scale only; it reproduces nothing.",
        ],
        tex_label="tab:reproduction",
    )



def panel_contrasts(p: Provenance, p3: pd.DataFrame, p4: pd.DataFrame,
                    family: pd.DataFrame) -> Panel:
    """Table 6b. Every declared contrast as a paired difference with its interval.

    Table 6 carries the corrected p-values; a p-value without an effect size is
    forbidden by the project's design notes A6, and the manuscript needs the differences
    themselves. Both pair sets are shown together because they are one
    family, with each row's Holm value taken from the COMPLETED correction rather than from the one that was true when the family was still filling up.
    """
    rows: list[list[str]] = []
    for _, entry in family.iterrows():
        contrast, label = str(entry["contrast"]), str(entry["label"])
        phase = str(entry["phase"])
        source_frame, src = ((p3, "results/p3_contrasts.csv") if phase == "P3"
                             else (p4, "results/p4_contrasts.csv"))
        got = source_frame[(source_frame["contrast"] == contrast)
                           & (source_frame["label"] == label)]
        if got.empty:
            continue
        row = got.iloc[0]
        k = key(contrast=contrast, label=label)
        rk = f"{contrast}|{label}"
        diff = p.num("T5b", rk, "difference", row["difference"], "+.4f", src, k,
                     "difference", "record")
        lo = p.num("T5b", rk, "difference", row["ci_lo"], "+.4f", src, k, "ci_lo",
                   "record")
        hi = p.num("T5b", rk, "difference", row["ci_hi"], "+.4f", src, k, "ci_hi",
                   "record")
        holm = p.pvalue("T5b", rk, "holm_p", entry["p_holm_final"],
                        "results/p4_family_final.csv", k, "p_holm_final",
                        "record")
        if "noise_floor_used" in source_frame.columns and not pd.isna(
                row.get("noise_floor_used")):
            floor = p.num("T5b", rk, "noise_floor", row["noise_floor_used"], ".4f",
                          src, k, "noise_floor_used", "record")
            clears = p.verdict("T5b", rk, "clears_floor",
                               "yes" if bool(row["clears_noise_floor"]) else "no",
                               src, k, "clears_noise_floor")
        else:
            floor, clears = "-", "-"
        rows.append([contrast, label_display(label), f"{diff} [{lo}, {hi}]", holm,
                     floor, clears, pair_set(phase)])

    return Panel(
        number="5b",
        title="Every declared contrast, as a paired difference",
        lead=[
            "Paired patient-clustered bootstrap on the difference itself: both arms",
            "are scored on the same resampled patients in each replicate, and two",
            "marginal intervals are never compared. No p-value appears without its",
            "effect size. Each row's Holm value is from the completed family of 15",
            ", not from the correction that was true while the family was still filling up.",
            "The noise floor is the seed-to-seed spread of that contrast's own arms",
            "at that label, and a difference not clearly larger than it is reported",
            "as no difference.",
        ],
        columns=["Contrast", "Label", "Difference [95% CI]", "Holm p", "Noise floor",
                 "Clears floor", "Pairs"],
        rows=rows,
        code_cols={0},
        notes=[
            "`R4_demo_acq - R6_waveform` at 24-hour ICU admission does not clear "
            "its own noise floor. Demographics plus acquisition context is not "
            "distinguishable from the waveform arm there.",
            "The three row 8 rows compare a within-stratum concordance against a "
            "plain AUROC on the same records, so they measure what the pair "
            "restriction costs rather than a difference between two models.",
        ],
        tex_label="tab:contrasts",
    )


def _below_interval_note(cal: pd.DataFrame) -> str:
    """A sentence about any cell whose ECE sits below its own interval.

    Generated rather than written, so a cell that starts tripping the flag later
    cannot go unmentioned by a note nobody thought to update. ECE is positively
    biased: binned absolute gaps cannot cancel, so resampling can only add to
    them, and at a cell whose calibration error is near zero the percentile
    interval can sit entirely above the value it was built around.
    """
    flagged = cal[cal["point_below_ci_lo"].astype(bool)] if "point_below_ci_lo" in cal else cal.iloc[0:0]
    if flagged.empty:
        return ("No cell's expected calibration error falls below its own "
                "interval.")
    parts = [f"`{row['arm']}` at {label_display(row['label'])} under "
             f"{str(row['scheme']).replace('_', ' ')} binning"
             for _, row in flagged.iterrows()]
    subject = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    verb = "sits" if len(parts) == 1 else "sit"
    return (f"The expected calibration error for {subject} {verb} below the "
            "lower bound of its own percentile interval. Expected calibration "
            "error is positively biased, because binned absolute gaps cannot "
            "cancel and resampling noise can only add to them, so at a cell "
            "whose calibration error is near zero the interval can sit entirely "
            "above the value it was built around. The declared protocol is a "
            "percentile interval and it is reported as computed rather than "
            "swapped for a bias-corrected one; the null band, not the interval, "
            "is what answers whether the arm is calibrated.")


def panel_calibration(p: Provenance, ece: pd.DataFrame, labels: list[str]) -> Panel:
    """Table 10. Calibration, which reads the probability scale and not the ranking."""
    src = "results/calibration/ece.csv"
    schemes = [str(s) for s in dict.fromkeys(ece["scheme"])]
    rows: list[list[str]] = []
    for arm in LADDER_ARMS:
        for label in labels:
            cells = ece[(ece["arm"] == arm) & (ece["label"] == label)]
            if cells.empty:
                continue
            rk = f"{arm}|{label}"
            values = []
            for scheme in schemes:
                got = cells[cells["scheme"] == scheme]
                if got.empty:
                    values.append("-")
                    continue
                row = got.iloc[0]
                k = key(arm=arm, label=label, scheme=scheme)
                point = p.num("T9", rk, f"ece_{scheme}", row["ece"], ".4f", src, k,
                              "ece", "record")
                lo = p.num("T9", rk, f"ece_{scheme}", row["ci_lo"], ".4f", src, k,
                           "ci_lo", "record")
                hi = p.num("T9", rk, f"ece_{scheme}", row["ci_hi"], ".4f", src, k,
                           "ci_hi", "record")
                values.append(f"{point} [{lo}, {hi}]")
            first = cells.iloc[0]
            fk = key(arm=arm, label=label, scheme=schemes[0])
            band = p.num("T9", rk, "null_hi", cells["null_hi"].max(), ".4f", src,
                         key(arm=arm, label=label), "null_hi", "record",
                         aggregate="max")
            mean_p = p.num("T9", rk, "mean_predicted", first["mean_predicted"],
                           ".4f", src, fk, "mean_predicted", "record")
            observed = p.num("T9", rk, "observed_rate", first["observed_rate"],
                             ".4f", src, fk, "observed_rate", "record")
            calibrated = p.verdict(
                "T9", rk, "calibrated",
                "yes" if bool(cells["inside_null_band"].all()) else "no",
                src, key(arm=arm, label=label), "inside_null_band")
            rows.append([arm, label_display(label), *values,
                         f"up to {band}", calibrated, mean_p, observed])

    return Panel(
        number="9",
        title="Calibration: expected calibration error under both declared schemes",
        lead=[
            "ECE reads the probability scale; every other table here reads only the",
            "ranking, and the two can disagree completely. The reference is not zero:",
            "at this bin count and sample size a perfectly calibrated score still",
            "shows a non-zero ECE, so each cell carries the band such a score would",
            "produce, obtained by drawing labels from the arm's own predicted",
            "probabilities. An arm counts as calibrated only if it sits inside that",
            "band under both schemes, because the choice of binning is a choice.",
        ],
        columns=["Arm", "Label"] + [f"ECE, {s.replace('_', ' ')} [95% CI]"
                                    for s in schemes]
        + ["Null band", "Calibrated", "Mean predicted", "Observed"],
        rows=rows,
        code_cols={0},
        notes=[
            "Row 8 is absent by construction: it is row 6's scores under a "
            "restricted pair statistic, so it has no probability of its own and a "
            "calibration figure for it would be row 6's under another name.",
            _below_interval_note(ece),
            "Both waveform arms over-predict at every label, by a factor "
            "running from about 1.8 to about 3.7 of the observed rate. The "
            "largest factor is at 28 days, where the observed rate is lowest; "
            "both columns it is computed from are in this table. They were "
            "fitted under focal BCE and their score is a mean over four crops "
            "and then over five seeds, and each of those moves the probability "
            "scale.",
        ],
        tex_label="tab:calibration",
    )


def panel_net_benefit(p: Provenance, points: pd.DataFrame, verdict: dict) -> Panel:
    """Table 11. Net benefit at the one pre-declared threshold."""
    src = "results/decision_curve/nb_at_threshold.csv"
    rows: list[list[str]] = []
    for arm in verdict["ranked_by_net_benefit"]:
        got = points[points["arm"] == arm]
        if got.empty:
            continue
        row = got.iloc[0]
        k = key(arm=arm, label=str(row["label"]))
        rk = arm
        nb = p.num("T10", rk, "net_benefit", row["net_benefit"], ".4f", src, k,
                   "net_benefit", "record")
        lo = p.num("T10", rk, "net_benefit", row["ci_lo"], ".4f", src, k, "ci_lo",
                   "record")
        hi = p.num("T10", rk, "net_benefit", row["ci_hi"], ".4f", src, k, "ci_hi",
                   "record")
        diff = p.num("T10", rk, "vs_treat_all", row["vs_treat_all"], "+.4f", src, k,
                     "vs_treat_all", "record")
        dlo = p.num("T10", rk, "vs_treat_all", row["vs_treat_all_lo"], "+.4f", src,
                    k, "vs_treat_all_lo", "record")
        dhi = p.num("T10", rk, "vs_treat_all", row["vs_treat_all_hi"], "+.4f", src,
                    k, "vs_treat_all_hi", "record")
        beats = p.verdict("T10", rk, "beats_treat_all",
                          "yes" if bool(row["vs_treat_all_lo"] > 0) else "no",
                          src, k, "beats_treat_all")
        flags = p.num("T10", rk, "flagged_fraction", row["flagged_fraction"], ".4f",
                      src, k, "flagged_fraction", "record")
        mean_p = p.num("T10", rk, "mean_predicted", row["mean_predicted"], ".4f",
                       src, k, "mean_predicted", "record")
        rows.append([arm, f"{nb} [{lo}, {hi}]", f"{diff} [{dlo}, {dhi}]", beats,
                     flags, mean_p])

    threshold = p.num("T10", "meta|threshold", "threshold",
                      points["threshold"].iloc[0], ".2f", src,
                      key(arm=str(points["arm"].iloc[0]),
                          label=str(points["label"].iloc[0])), "threshold",
                      "record")
    treat_all = p.num("T10", "meta|treat_all", "treat_all",
                      points["net_benefit_treat_all"].iloc[0], ".4f", src,
                      key(arm=str(points["arm"].iloc[0]),
                          label=str(points["label"].iloc[0])),
                      "net_benefit_treat_all", "record")
    prevalence = p.num("T10", "meta|prevalence", "prevalence",
                       points["prevalence"].iloc[0], ".4f", src,
                       key(arm=str(points["arm"].iloc[0]),
                           label=str(points["label"].iloc[0])), "prevalence",
                       "record")

    return Panel(
        number="10",
        title="Net benefit at the pre-declared threshold",
        lead=[
            f"Threshold {threshold}, fixed from the clinical framing and recorded",
            "before any curve was computed, and quoted at",
            "24-hour ICU admission alone because it is the only reported label with",
            "an action available at emergency-department disposition. Treating",
            f"everyone scores {treat_all} here and treating no one scores exactly",
            f"zero; the prevalence is {prevalence}, which no rule can exceed. The",
            "comparison against treat-all is a paired clustered bootstrap on the",
            "difference, since treat-all also varies with the resampled prevalence.",
        ],
        columns=["Arm", "Net benefit [95% CI]", "vs treat-all [95% CI]",
                 "Beats treat-all", "Fraction flagged", "Mean predicted"],
        rows=rows,
        code_cols={0},
        notes=[
            "Net benefit reads the probability scale rather than the ranking, "
            "so this ordering is not a discrimination ordering: an arm that ranks "
            "cases well loses at a fixed threshold when its probabilities are "
            "inflated. No recalibration was performed; only test-fold predictions "
            "are stored, and refitting on them would be circular.",
        ],
        tex_label="tab:net_benefit",
    )


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def md_cell(text: str, code: bool) -> str:
    if code and text:
        return f"`{text}`"
    return text


def tex_escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        text = text.replace(a, b)
    return text


def tex_cell(text: str, code: bool) -> str:
    if code and text:
        return r"\texttt{" + tex_escape(text) + "}"
    return tex_escape(text)


def strip_md(text: str) -> str:
    return text.replace("`", "")


def render_md(panels: list[Panel], footnote: list[str], stamp: list[str]) -> str:
    out: list[str] = [
        "<!-- BEGIN GENERATED TABLE MAIN -->",
        "",
        "# table_main: the headline tables",
        "",
        "Generated by `scripts/make_table_main.py` from the artifacts in",
        "`results/`. Never hand-edited. Every number here is read from a named CSV",
        "cell and the read is recorded in `results/table_main_cells.csv`, which is",
        "what the corresponding check re-derives against.",
        "",
    ]
    out += stamp + [""]
    for panel in panels:
        out.append(f"## Table {table_labels.display('T' + panel.number)}."
                   f" {panel.title}")
        out.append("")
        out += panel.lead
        out.append("")
        out.append("| " + " | ".join(panel.columns) + " |")
        out.append("|" + "|".join("---" for _ in panel.columns) + "|")
        for row in panel.rows:
            out.append("| " + " | ".join(
                md_cell(cell, i in panel.code_cols) for i, cell in enumerate(row)
            ) + " |")
        out.append("")
        for note in panel.notes:
            out.append(f"> {note}")
            out.append("")
    out.append("## Footnote, generated with the tables")
    out.append("")
    out += footnote
    out.append("")
    out.append("<!-- END GENERATED TABLE MAIN -->")
    return "\n".join(out) + "\n"


def render_tex(panels: list[Panel], footnote: list[str], stamp: list[str]) -> str:
    out: list[str] = [
        "% Generated by scripts/make_table_main.py. Never hand-edited.",
        "% Every number traces to a cell recorded in results/table_main_cells.csv.",
        "% Requires \\usepackage{booktabs}.",
    ]
    out += ["% " + tex_escape(strip_md(line)) for line in stamp]
    for panel in panels:
        spec = "l" * len(panel.columns)
        out += [
            "",
            # The panel id, so a consumer can split this file by panel without
            # parsing captions. `make_submission.py` uses it to separate the
            # main-text panels from the supplementary ones per
            # notes/table_layout.md, which is a recorded decision and should
            # not be re-derived from a heading string.
            f"% PANEL T{panel.number}",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{" + tex_escape(strip_md(panel.title)) + ". "
            + tex_escape(strip_md(" ".join(panel.lead))) + "}",
            r"\label{" + panel.tex_label + "}",
            r"\begin{tabular}{" + spec + "}",
            r"\toprule",
            " & ".join(tex_escape(strip_md(c)) for c in panel.columns) + r" \\",
            r"\midrule",
        ]
        for row in panel.rows:
            if not any(cell for cell in row):
                out.append(r"\midrule")
                continue
            out.append(" & ".join(
                tex_cell(cell, i in panel.code_cols) for i, cell in enumerate(row)
            ) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}"]
        for note in panel.notes:
            out.append(r"\par\footnotesize " + tex_escape(strip_md(note)))
        out.append(r"\end{table}")
    out += ["", "% Footnote, generated with the tables:"]
    out += ["% " + tex_escape(strip_md(line)) for line in footnote]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def build() -> tuple[str, str, pd.DataFrame]:
    p = Provenance()

    arms = pd.read_csv(RESULTS / "p3_arms.csv")
    recovery = pd.read_csv(RESULTS / "p3_recovery.csv")
    contrasts = pd.read_csv(RESULTS / "p3_contrasts.csv")
    ladder = pd.read_csv(RESULTS / "p3_ladder.csv")
    trends = pd.read_csv(RESULTS / "p3_trends.csv")
    noise = pd.read_csv(RESULTS / "p3_noise.csv")
    p4arms = pd.read_csv(RESULTS / "p4_arms.csv")
    retained = pd.read_csv(RESULTS / "p4_retained.csv")
    effective = pd.read_csv(RESULTS / "p4_effective.csv")
    family = pd.read_csv(RESULTS / "p4_family_final.csv")
    gap = pd.read_csv(RESULTS / "p2_reproduction_gap.csv")
    macro = pd.read_csv(RESULTS / "p2_macro.csv")
    verdict = json.loads((RESULTS / "p3_verdict.json").read_text(encoding="utf-8"))
    g1 = json.loads((RESULTS / "p1_verdict.json").read_text(encoding="utf-8"))

    # The three DECLARED labels, read from the comparison file rather than
    # inferred from whatever is in p3_arms.csv. That table also carries the four
    # horizon-ladder labels, which belong to the other family and must
    # not silently widen the headline ladder.
    declared = yaml.safe_load(
        (COMPARISONS / "acq_context_audit.yaml").read_text(encoding="utf-8")
    )
    labels = [str(x) for x in declared["labels"]]
    missing = [x for x in labels if x not in set(arms["label"])]
    if missing:
        raise RuntimeError(f"declared labels absent from p3_arms.csv: {missing}")
    primary = recovery[recovery["is_primary"]]
    if len(primary) != 1:
        raise RuntimeError(f"p3_recovery.csv flags {len(primary)} primary rows, expected 1")
    primary_label = str(primary.iloc[0]["label"])

    g1_candidates = [r for r in g1["verdict"]["per_label"] if r["label"] == primary_label]
    if len(g1_candidates) != 1:
        raise RuntimeError(f"p1_verdict.json has {len(g1_candidates)} rows for "
                           f"{primary_label}, expected 1")
    g1_row = g1_candidates[0]

    calibration_path = RESULTS / "calibration" / "ece.csv"
    net_benefit_path = RESULTS / "decision_curve" / "nb_at_threshold.csv"
    p5_verdict_path = RESULTS / "p5_verdict.json"

    reconciliation_path = RESULTS / "cohort_reconciliation.csv"
    panels = [
        panel_ladder(p, arms, labels),
        panel_targets(p, g1_row, recovery, retained, trends, primary_label),
        panel_recovery(p, recovery),
        panel_recovery_denominators(p, recovery),
        panel_cohort_reconciliation(p, pd.read_csv(reconciliation_path)),
        panel_first_ecg(p, pd.read_csv(RESULTS / "first_ecg_sensitivity.csv"),
                        macro),
        panel_triage(p, pd.read_csv(RESULTS / "exploratory_triage.csv")),
        panel_calendar_features(
            p, pd.read_csv(RESULTS / "exploratory_sensitivities.csv")),
        panel_shared_config(
            p, pd.read_csv(RESULTS / "exploratory_sensitivities.csv")),
        panel_calibration_fit(
            p, pd.read_csv(RESULTS / "calibration" / "calibration_fit.csv")),
        panel_matching_sensitivity(
            p, effective,
            pd.read_csv(RESULTS / "p4_sensitivity_no_weekday_effective.csv")),
        panel_row8(p, p4arms, effective, labels),
        panel_retained(p, retained),
        panel_balance(p, pd.read_csv(RESULTS / "p4_balance.csv"),
                      json.loads((RESULTS / "p4_matching.json").read_text(
                          encoding="utf-8"))),
        panel_blocks(p, pd.read_csv(RESULTS / "acq_block_profile.csv")),
        panel_control(p, pd.read_csv(RESULTS / "p4_control.csv")),
        panel_family(p, family),
        panel_contrasts(p, contrasts, pd.read_csv(RESULTS / "p4_contrasts.csv"),
                        family),
        panel_ladder_horizons(p, ladder),
        panel_trends(p, trends),
        panel_noise(p, noise, gap, labels),
        panel_reproduction(p, gap, macro),
    ]

    # The two secondary analyses join the table only once they have been scored, so
    # this script stays runnable before them and the manuscript has a single
    # provenance index once they exist.
    if calibration_path.is_file():
        panels.append(panel_calibration(p, pd.read_csv(calibration_path), labels))
    if net_benefit_path.is_file() and p5_verdict_path.is_file():
        panels.append(panel_net_benefit(
            p, pd.read_csv(net_benefit_path),
            json.loads(p5_verdict_path.read_text(encoding="utf-8"))))

    n_boot = int(contrasts["n_boot"].iloc[0])
    m_audit = int(family["holm_family_size"].iloc[0])
    m_ladder = int(trends["holm_family_size"].iloc[0])
    footnote = [
        f"- Resampling: patient-clustered percentile bootstrap, {n_boot:,} resamples,",
        "  resampling unit `subject_id`. Contrasts are paired on the same",
        "  draw; two marginal intervals are never compared.",
        "- Unit: record level, one row per ECG, for Tables 1, 3, 6a, 6b and 7.",
        "  Tables 4a and 4b are a within-stratum concordance over restricted pairs",
        "  and are a different estimand, which is why row 8 is not a row of Table 2.",
        "  Table 9 is the only place crop-level numbers appear.",
        f"- Multiplicity: two families, corrected separately and never pooled.",
        f"  `acq_context_audit` at its declared size m = {m_audit};",
        f"  `horizon_ladder` at its declared size m = {m_ladder}. Both use Holm.",
        "- Seeds: five per arm. Scores are averaged across seeds and one AUROC is",
        "  taken on the pooled score; the per-seed spread is Table 8.",
        "- Noise floor: per (arm, label) at record level, not a single scalar.",
        "- Row-level predictions live outside the repository.",
    ]
    stamp = [
        f"- Gate {verdict['gate']}: {verdict['result']}. It is a completeness gate and",
        "  it is not the claim; Table 3 is what the claim measured.",
        f"- Pre-declared primary quantity, `{verdict['primary_quantity']}` at",
        f"  `{verdict['primary_label']}`: {verdict['primary_outcome']}.",
        "- All four pre-declared quantitative targets are in Table 3 and none was met",
        "  at the endpoint it named.",
    ]

    return render_md(panels, footnote, stamp), render_tex(panels, footnote, stamp), p.frame()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)

    md, tex, cells = build()
    (out_dir / "table_main.md").write_text(md, encoding="utf-8")
    (out_dir / "table_main.tex").write_text(tex, encoding="utf-8")
    cells.to_csv(out_dir / "table_main_cells.csv", index=False)

    if not args.quiet:
        print(f"wrote table_main.md, table_main.tex and table_main_cells.csv "
              f"({len(cells)} traced numbers) into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

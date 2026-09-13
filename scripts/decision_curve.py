"""Net benefit, following Vickers and Elkin, with controls that can fail.

At a threshold probability t, a patient is flagged when the predicted risk
exceeds t. Net benefit trades true positives against false positives at the
exchange rate the threshold implies:

    NB(t) = TP/n - (FP/n) * t / (1 - t)

The odds factor `t / (1 - t)` is the whole content of the measure: choosing t
declares how many unnecessary actions one true case is worth. The declared threshold fixes
t = 0.10 for `deterioration_icu_24h`, an exchange rate of nine to one, and does
so from the clinical framing before any curve exists.

Unlike AUROC, net benefit reads the PROBABILITY SCALE. An arm whose ranking is
excellent but whose probabilities are systematically too high will flag far too
many patients at any fixed threshold and can score below treat-all. That is a
statement about the probability scale and not about discrimination, and the declared threshold
states it in advance precisely so it cannot be produced later as an excuse.

`selftest()` carries three analytic identities, each of which a sign error, a
misplaced odds conversion, or a threshold-versus-odds confusion would break:

    treat everyone   NB = prevalence - (1 - prevalence) * t / (1 - t), exactly
    treat no one     NB = 0, exactly, at every threshold
    a perfect score  NB = prevalence, exactly, at every threshold below 1
"""

from __future__ import annotations

import numpy as np


def net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    """Net benefit of acting when the predicted risk is AT LEAST `threshold`.

    The comparison is `>=`, which is the rule Vickers and Elkin state. The
    distinction from `>` is invisible on continuous scores and is not invisible
    on a constant one: the prevalence-floor arm predicts a single value for
    every patient, so at a threshold equal to that value the two conventions
    differ by the entire cohort. The selftest below separates them explicitly,
    because an earlier version of this file used `>` and no control noticed.
    """
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=float)
    n = y.size
    if n == 0 or not (0.0 <= threshold < 1.0):
        return float("nan")
    flagged = p >= threshold
    tp = float(np.count_nonzero(flagged & (y == 1)))
    fp = float(np.count_nonzero(flagged & (y == 0)))
    return tp / n - (fp / n) * (threshold / (1.0 - threshold))


def net_benefit_all(y: np.ndarray, threshold: float) -> float:
    """Net benefit of flagging everyone: the analytic treat-all reference."""
    y = np.asarray(y, dtype=np.int64)
    if y.size == 0 or not (0.0 <= threshold < 1.0):
        return float("nan")
    prevalence = float(y.mean())
    return prevalence - (1.0 - prevalence) * (threshold / (1.0 - threshold))


def flagged_fraction(p: np.ndarray, threshold: float) -> float:
    """The share of patients the rule would act on. A capacity statement."""
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.count_nonzero(p >= threshold) / p.size)


def grid(lo: float, hi: float, step: float) -> np.ndarray:
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 10)


def selftest(n: int = 5000, seed: int = 0) -> list[str]:
    """Three analytic identities and one ordering property."""
    problems: list[str] = []
    rng = np.random.default_rng(seed)

    prevalence = 0.126          # the icu_24h base rate, near enough
    y = (rng.random(n) < prevalence).astype(np.int64)
    observed = float(y.mean())

    for t in (0.01, 0.05, 0.10, 0.25, 0.50, 0.90):
        # (1) treat everyone. A score above every threshold flags everyone, so
        # the empirical and analytic values must agree exactly.
        empirical = net_benefit(y, np.ones(n), t)
        analytic = observed - (1.0 - observed) * (t / (1.0 - t))
        if not np.isclose(empirical, analytic, atol=1e-12):
            problems.append(
                f"treat-all at t={t}: net benefit {empirical} against the "
                f"analytic {analytic}"
            )
        if not np.isclose(net_benefit_all(y, t), analytic, atol=1e-12):
            problems.append(f"net_benefit_all disagrees with its own formula at t={t}")

        # (2) treat no one. Exactly zero, not approximately.
        if net_benefit(y, np.zeros(n), t) != 0.0:
            problems.append(f"treat-none at t={t} is not exactly zero")

        # (3) a perfect predictor. Every positive flagged, no false positive, so
        # net benefit is the prevalence at every threshold.
        perfect = net_benefit(y, y.astype(float) * 0.998 + 0.001, t)
        if not np.isclose(perfect, observed, atol=1e-12):
            problems.append(
                f"a perfect predictor at t={t} scores {perfect}, not the "
                f"prevalence {observed}"
            )

    # (4) the odds factor must be applied, not the threshold itself. A
    # implementation that used t instead of t/(1-t) agrees at small t and
    # diverges as t grows, so the check is made where they differ most.
    t = 0.5
    flagged_none_but_one = np.zeros(n)
    flagged_none_but_one[0] = 0.9              # one false or true positive
    value = net_benefit(y, flagged_none_but_one, t)
    expected = (1.0 if y[0] == 1 else 0.0) / n - (0.0 if y[0] == 1 else 1.0) / n * 1.0
    if not np.isclose(value, expected, atol=1e-12):
        problems.append(
            f"at t=0.5 the odds factor must be 1.0; a single flagged case gives "
            f"{value} against {expected}"
        )

    # (5) the boundary convention. A score EXACTLY at the threshold is acted
    # on, per Vickers and Elkin. This is the one control that separates `>=`
    # from `>`, and it matters for the constant prevalence-floor arm, whose
    # every prediction is the same number.
    for t in (0.05, 0.10, 0.25):
        at_threshold = np.full(n, t)
        value = net_benefit(y, at_threshold, t)
        expected = net_benefit_all(y, t)
        if not np.isclose(value, expected, atol=1e-12):
            problems.append(
                f"a score exactly at the threshold t={t} scores {value}, not the "
                f"treat-all value {expected}. The rule must act at p >= t"
            )
        if not np.isclose(flagged_fraction(at_threshold, t), 1.0, atol=1e-12):
            problems.append(
                f"a score exactly at t={t} flags "
                f"{flagged_fraction(at_threshold, t)} of patients, not all of them"
            )

    # (6) net benefit is bounded above by the prevalence for any rule.
    scores = rng.random(n)
    for t in (0.02, 0.1, 0.3):
        value = net_benefit(y, scores, t)
        if value > observed + 1e-12:
            problems.append(
                f"a random score at t={t} scores {value}, above the prevalence "
                f"{observed}, which no rule can exceed"
            )

    return problems

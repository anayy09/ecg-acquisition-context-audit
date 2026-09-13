"""Within-stratum concordance, the statistic row 8 is defined on.

Restricting AUROC to positive-negative pairs drawn from the same stratum is the
concrete form of "acquisition context held fixed": no compared pair differs in
delay bucket, hour bucket, weekday status or triage acuity. Pooling the
per-stratum Wilcoxon statistics by their pair counts is the Mantel-Haenszel
weighting, and it is what makes the result a single number rather than 143.

The whole point of the design is that it is bootstrapped, so the naive
implementation — loop over strata, call `roc_auc_score` — is not affordable: 143
strata times 10,000 replicates times several arms is millions of calls. Two
precomputations make it a handful of vectorised operations per replicate.

1. Each row carries a rank that is DENSE WITHIN ITS OWN STRATUM. Resampling
   changes how many times a row appears, never the relative order of scores, so
   the ranks are computed once and reused. Local ranks keep the count matrix at
   `n_strata x max_stratum_size` (143 x 415 here) instead of
   `n_strata x n_rows` (143 x 6,439), which is the difference between a few
   seconds and several minutes.
2. Positives and negatives are counted into that matrix with one `bincount`
   each, and the number of negatives below each positive comes from one
   exclusive cumulative sum along the rank axis.
"""

from __future__ import annotations

import numpy as np


class StratifiedConcordance:
    """Pair-weighted within-stratum AUROC, evaluable on any row resample."""

    def __init__(self, y: np.ndarray, score: np.ndarray, stratum: np.ndarray) -> None:
        self.y = np.asarray(y).astype(np.int64)
        stratum = np.asarray(stratum)
        codes, self.n_strata = _dense(stratum)
        self.codes = codes

        score = np.asarray(score, dtype=float)
        local = np.empty(len(score), dtype=np.int64)
        width = 1
        for s in range(self.n_strata):
            rows = np.flatnonzero(codes == s)
            if rows.size == 0:
                continue
            ranks, k = _dense(score[rows])
            local[rows] = ranks
            width = max(width, k)
        self.local = local
        self.width = int(width)
        self.key = codes * self.width + local
        self.size = self.n_strata * self.width

    def evaluate(self, rows: np.ndarray) -> float:
        """Pooled within-stratum concordance over `rows`, NaN if no pair exists."""
        if rows.size == 0:
            return float("nan")
        key = self.key[rows]
        y = self.y[rows]
        pos = np.bincount(key[y == 1], minlength=self.size).reshape(
            self.n_strata, self.width)
        neg = np.bincount(key[y == 0], minlength=self.size).reshape(
            self.n_strata, self.width)
        below = np.cumsum(neg, axis=1) - neg
        n_pos = pos.sum(axis=1)
        n_neg = neg.sum(axis=1)
        denominator = float((n_pos.astype(np.int64) * n_neg.astype(np.int64)).sum())
        if denominator <= 0:
            return float("nan")
        numerator = float((pos * below).sum()) + 0.5 * float((pos * neg).sum())
        return numerator / denominator

    def informative(self) -> np.ndarray:
        """Strata that contribute at least one positive-negative pair."""
        pos = np.bincount(self.codes[self.y == 1], minlength=self.n_strata)
        neg = np.bincount(self.codes[self.y == 0], minlength=self.n_strata)
        return (pos > 0) & (neg > 0)

    def n_pairs(self) -> int:
        pos = np.bincount(self.codes[self.y == 1], minlength=self.n_strata)
        neg = np.bincount(self.codes[self.y == 0], minlength=self.n_strata)
        return int((pos.astype(np.int64) * neg.astype(np.int64)).sum())


def _dense(values: np.ndarray) -> tuple[np.ndarray, int]:
    _, inverse = np.unique(values, return_inverse=True)
    inverse = inverse.astype(np.int64)
    if inverse.size == 0:
        return inverse, 1
    return inverse, int(inverse.max()) + 1


def control_values() -> dict[str, float]:
    """The measured values of the three controls, computed once.

    These numbers are quoted in the manuscript, so they must have exactly one
    definition. An earlier draft stated the shortcut score's unstratified AUROC
    as 0.72 while this control produces 0.7406, and the mismatch survived
    because the manuscript's value was declared a definitional constant and so
    was never compared to anything. `selftest()` asserts against this
    function and `make_stratified_control.py` writes it into a generated table
    cell, so the assertion and the printed figure cannot diverge again.

    Deterministic: seeded, fixed sizes, and the draw ORDER is part of the
    definition. Do not reorder the rng calls below without regenerating the
    table, because every value here would move.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)

    # Control 1: a single stratum must reproduce plain AUROC exactly.
    n = 800
    y = (rng.random(n) < 0.3).astype(int)
    score = rng.normal(size=n) + y * 0.8
    one = StratifiedConcordance(y, score, np.zeros(n, dtype=int))

    # Control 2: a shortcut living entirely BETWEEN strata must vanish within
    # them. This is the substantive one and it is the shape row 8 exists to
    # detect: inside a stratum the score carries no ordering at all.
    stratum = rng.integers(0, 8, n)
    y2 = (rng.random(n) < 0.15 + 0.1 * stratum).astype(int)
    shortcut = stratum.astype(float)

    return {
        "single_stratum_within": float(one.evaluate(np.arange(n))),
        "single_stratum_plain": float(roc_auc_score(y, score)),
        "shortcut_plain": float(roc_auc_score(y2, shortcut)),
        "shortcut_within": float(
            StratifiedConcordance(y2, shortcut, stratum).evaluate(np.arange(n))),
        "doubled_within": float(
            one.evaluate(np.repeat(np.arange(n), 2))),
        "n": float(n),
        "n_strata": float(8),
    }


def selftest() -> list[str]:
    """One stratum must reproduce plain AUROC; strata must destroy a pure shortcut.

    The second control is the substantive one. If every stratum's scores are a
    pure function of the stratum, the unstratified AUROC can be high while the
    within-stratum concordance is exactly 0.5, because inside a stratum the
    scores carry no ordering at all. A statistic that did not actually restrict
    its pairs would not show that collapse.
    """
    problems: list[str] = []
    v = control_values()

    if not np.isclose(v["single_stratum_within"], v["single_stratum_plain"],
                      atol=1e-10):
        problems.append(
            f"with a single stratum the within-stratum statistic is "
            f"{v['single_stratum_within']} but plain AUROC is "
            f"{v['single_stratum_plain']}; they must agree"
        )

    if v["shortcut_plain"] <= 0.6:
        problems.append(
            f"the control's between-stratum shortcut only reaches "
            f"{v['shortcut_plain']:.3f} unstratified, so the collapse it is meant "
            "to demonstrate is not visible"
        )
    if not np.isclose(v["shortcut_within"], 0.5, atol=1e-9):
        problems.append(
            f"a score that is constant within every stratum gives a within-stratum "
            f"concordance of {v['shortcut_within']}, not 0.5; the pairs are not "
            "being restricted"
        )

    # Resampling must reach the same number when it reproduces the full set.
    if not np.isclose(v["doubled_within"], v["single_stratum_plain"], atol=1e-10):
        problems.append("duplicating every row changed the statistic")
    return problems

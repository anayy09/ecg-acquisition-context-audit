"""Fast patient-clustered bootstrap for AUROC.

A naive clustered bootstrap rebuilds the resampled index with a Python loop over
groups and calls `roc_auc_score` per draw. At 10,000 resamples over ~3,600
patients that costs minutes per interval, and the paper needs dozens of
intervals.

Two changes make it seconds instead:

1. The ragged group gather is vectorised with `np.repeat`, so no per-draw Python
   loop over groups.
2. AUROC is computed from the Mann-Whitney U statistic over precomputed score
   ranks with two `bincount` calls, instead of re-sorting every draw. Ranks are
   computed once against the full score vector; a resample only changes which
   items are counted, not their relative order, so the statistic is identical.

`selftest()` checks the fast AUC against scikit-learn, including ties, and is run
as a gate rather than trusted.
"""

from __future__ import annotations

import numpy as np


class ClusterIndex:
    """Precomputed group -> item mapping for repeated clustered resampling."""

    def __init__(self, groups: np.ndarray) -> None:
        self.unique, inverse = np.unique(groups, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        self.order = order
        counts = np.bincount(inverse, minlength=len(self.unique))
        self.counts = counts.astype(np.int64)
        self.starts = np.concatenate(([0], np.cumsum(self.counts)[:-1])).astype(np.int64)
        self.n_groups = len(self.unique)

    @classmethod
    def from_codes(cls, codes: np.ndarray, n_groups: int) -> "ClusterIndex":
        """An index over a GLOBAL group universe, where some groups have no items.

        `__init__` derives the universe from the groups it is handed, which makes
        two cells with different row sets disagree about what group 7 is. this stage
        resamples one patient list and recomputes several labels on that same
        draw, and the labels do not share a row set: the tabular arms drop the
        `-999` sentinel per label, so `deterioration_icu_24h` has 6,407 rows where
        `deterioration_mortality_28d` has 6,428. Every cell is therefore indexed
        against one shared patient ordering, and a patient absent from a cell
        contributes zero rows to it rather than shifting its neighbours.
        """
        obj = cls.__new__(cls)
        codes = np.asarray(codes, dtype=np.int64)
        obj.unique = np.arange(n_groups)
        obj.order = np.argsort(codes, kind="stable")
        obj.counts = np.bincount(codes, minlength=n_groups).astype(np.int64)
        obj.starts = np.concatenate(([0], np.cumsum(obj.counts)[:-1])).astype(np.int64)
        obj.n_groups = int(n_groups)
        return obj

    def gather(self, picked: np.ndarray) -> np.ndarray:
        """Rows belonging to `picked` groups, in order, with multiplicity."""
        cnt = self.counts[picked]
        total = int(cnt.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        cum_before = np.concatenate(([0], np.cumsum(cnt)[:-1]))
        offsets = np.repeat(self.starts[picked] - cum_before, cnt)
        return self.order[offsets + np.arange(total)]

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """One clustered resample: draw groups with replacement, take all their items."""
        return self.gather(rng.integers(0, self.n_groups, self.n_groups))


def rank_scores(score: np.ndarray) -> tuple[np.ndarray, int]:
    """Map scores to dense integer buckets preserving order, ties shared."""
    _, inverse = np.unique(score, return_inverse=True)
    return inverse.astype(np.int64), int(inverse.max()) + 1


def auc_from_ranks(y: np.ndarray, ranks: np.ndarray, n_buckets: int) -> float:
    """AUROC via Mann-Whitney U over precomputed rank buckets, ties at 0.5."""
    positive = y == 1
    pos = np.bincount(ranks[positive], minlength=n_buckets)
    neg = np.bincount(ranks[~positive], minlength=n_buckets)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    below = np.concatenate(([0], np.cumsum(neg)[:-1]))
    u = float(np.dot(pos, below) + 0.5 * np.dot(pos, neg))
    return u / (n_pos * n_neg)


def clustered_auc_ci(
    y: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Patient-clustered percentile bootstrap interval for AUROC."""
    y = np.asarray(y).astype(np.int64)
    ranks, k = rank_scores(np.asarray(score, dtype=float))
    point = auc_from_ranks(y, ranks, k)

    index = ClusterIndex(np.asarray(groups))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = index.draw(rng)
        draws[b] = auc_from_ranks(y[idx], ranks[idx], k)
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def paired_clustered_auc_diff(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Paired clustered bootstrap on the AUROC difference b - a.

    Both arms are scored on the *same* resample, which is what makes the interval
    paired. Comparing two independently drawn marginal intervals is the error
    the patient resampling rule exists to prevent.
    """
    y = np.asarray(y).astype(np.int64)
    ra, ka = rank_scores(np.asarray(score_a, dtype=float))
    rb, kb = rank_scores(np.asarray(score_b, dtype=float))
    point = auc_from_ranks(y, rb, kb) - auc_from_ranks(y, ra, ka)

    index = ClusterIndex(np.asarray(groups))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = index.draw(rng)
        yy = y[idx]
        draws[i] = auc_from_ranks(yy, rb[idx], kb) - auc_from_ranks(yy, ra[idx], ka)

    finite = draws[np.isfinite(draws)]
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min(float((finite <= 0).mean()), float((finite >= 0).mean()))
    return {
        "difference": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_value": float(min(1.0, p)),
        "n_boot": int(n_boot),
        "n_groups": int(index.n_groups),
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, average ranks for ties, NaN if either is flat."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or not np.isfinite(y).all():
        return float("nan")
    rx, ry = _average_ranks(x), _average_ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # average tied ranks, so a replicate with two equal deltas is not ordered
    # by array position
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.bincount(inverse, weights=ranks, minlength=len(unique))
        ranks = (sums / counts)[inverse]
    return ranks


def clustered_statistic_ci(
    cells: dict[str, ClusterIndex],
    n_groups: int,
    statistic,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Patient-clustered percentile interval for an ARBITRARY statistic.

    The recovery ratio and the horizon-ladder trend are not differences of two
    AUROCs, so `paired_clustered_auc_diff` cannot produce their intervals, and
    combining the endpoints of marginal intervals would produce an interval for a
    quantity nobody reported. the project's design notes the protocol requires the interval on the ratio
    itself; the ladder requires it on the correlation itself.

    So: patients are drawn ONCE per replicate from a shared universe of
    `n_groups`, every cell is gathered on that same draw, and `statistic` is
    recomputed end to end from the gathered rows. Correlation between labels, and
    between arms within a label, is carried rather than assumed away.

    `statistic` receives ``{cell_name: row_index_array}`` and returns a float. It
    may return NaN — an undefined recovery ratio, a flat rank vector — and those
    replicates are dropped from the percentile rather than clipped or imputed,
    which is what `horizon_ladder.yaml` requires of an undefined denominator.
    """
    point = statistic({name: index.gather(np.arange(n_groups))
                       for name, index in cells.items()})

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        picked = rng.integers(0, n_groups, n_groups)
        draws[i] = statistic({name: index.gather(picked)
                              for name, index in cells.items()})

    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return {"estimate": float(point), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p_value": float("nan"),
                "n_boot": int(n_boot), "n_groups": int(n_groups),
                "n_finite_replicates": 0}
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min(float((finite <= 0).mean()), float((finite >= 0).mean()))
    return {
        "estimate": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_value": float(min(1.0, p)),
        "n_boot": int(n_boot),
        "n_groups": int(n_groups),
        "n_finite_replicates": int(finite.size),
    }


def selftest(n: int = 4000, seed: int = 0) -> list[str]:
    """Check the fast AUC against scikit-learn, and the gather against a loop."""
    from sklearn.metrics import roc_auc_score

    problems: list[str] = []
    rng = np.random.default_rng(seed)

    for name, score in (
        ("continuous", rng.normal(size=n)),
        ("heavy ties", rng.integers(0, 5, n).astype(float)),
        ("constant", np.zeros(n)),
    ):
        y = (rng.random(n) < 0.2).astype(np.int64)
        ranks, k = rank_scores(score)
        mine = auc_from_ranks(y, ranks, k)
        theirs = float(roc_auc_score(y, score))
        if not np.isclose(mine, theirs, atol=1e-10):
            problems.append(
                f"fast AUC disagrees with sklearn on {name}: {mine} vs {theirs}"
            )

    # The vectorised gather must reproduce a plain per-group loop exactly.
    groups = rng.integers(0, 300, n)
    index = ClusterIndex(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    by_group = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    for _ in range(5):
        fast = index.draw(rng_a)
        picked = rng_b.integers(0, index.n_groups, index.n_groups)
        slow = np.concatenate([by_group[p] for p in picked])
        if not np.array_equal(np.sort(fast), np.sort(slow)):
            problems.append("the vectorised clustered gather differs from a loop")
            break

    # `from_codes` must agree with `__init__` when the universe happens to be
    # complete, and must tolerate groups that own no rows when it does not.
    dense = ClusterIndex.from_codes(inverse, len(unique))
    picked = np.random.default_rng(3).integers(0, len(unique), len(unique))
    if not np.array_equal(np.sort(dense.gather(picked)), np.sort(index.gather(picked))):
        problems.append("ClusterIndex.from_codes disagrees with the derived index")
    sparse_codes = inverse[inverse % 3 != 0]
    sparse_rows = np.flatnonzero(inverse % 3 != 0)
    sparse = ClusterIndex.from_codes(sparse_codes, len(unique))
    got = sparse_rows[sparse.gather(np.arange(len(unique)))]
    if not np.array_equal(np.sort(got), np.sort(sparse_rows)):
        problems.append(
            "a cell missing some groups does not gather its own rows; a patient "
            "absent from one label would shift that label's neighbours"
        )

    # Spearman against scipy, on a vector with a tie.
    try:
        from scipy.stats import spearmanr

        for vector in ([1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 1.0, 3.0], [2.0, 9.0, 4.0, 1.0]):
            mine = spearman(np.array([1.0, 7.0, 90.0, 180.0]), np.array(vector))
            theirs = float(spearmanr([1.0, 7.0, 90.0, 180.0], vector).statistic)
            if not np.isclose(mine, theirs, atol=1e-12):
                problems.append(f"spearman disagrees with scipy on {vector}: "
                                f"{mine} vs {theirs}")
    except ImportError:
        problems.append("scipy is unavailable, so spearman is unchecked")
    return problems

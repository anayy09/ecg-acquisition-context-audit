"""Expected calibration error and reliability curves, with their own controls.

AUROC reads a ranking. ECE reads the probability scale, and nothing else in this
project does. Two arms with identical discrimination can have very different
ECE, and the waveform arms in particular were fitted under focal BCE at gamma 2
 and then averaged twice, over four crops and over five seeds
, both of which shrink probabilities toward the middle. So the quantity
here is a property of the reported probability, not of the model's ability to
separate cases, and the two must not be read as one.

`selftest()` is run as a gate rather than trusted, exactly as `bootstrap.py` is.
It carries the control that matters and refuses the one that cannot work:

  "a perfectly calibrated score has ECE zero" IS FALSE at finite n. With 15 bins
  over roughly 6,400 records, sampling noise alone puts ECE at order 0.01. A
  check asserting ECE near zero would fail forever; one asserting it below 0.05
  would pass on almost anything.

So the null is SIMULATED: labels are drawn repeatedly from a known probability
vector, the ECE of each draw is recorded, and a genuinely calibrated score must
sit inside that band while a distorted one must sit outside it. The distortion
used is p -> p**2, which is strictly monotone and therefore leaves AUROC exactly
unchanged, so the control separates calibration from discrimination rather than
measuring the same thing twice.
"""

from __future__ import annotations

import numpy as np


def equal_width_bins(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin index per item, over n_bins equal-width intervals of [0, 1]."""
    idx = np.floor(np.asarray(p, dtype=float) * n_bins).astype(np.int64)
    return np.clip(idx, 0, n_bins - 1)


def equal_mass_bins(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin index per item, over n_bins bins of as-equal-as-possible COUNT.

    The sorted order is split into contiguous blocks of equal size and each
    boundary is then advanced to the end of whatever tie run it lands inside, so
    items with the SAME predicted probability always share a bin.

    That refinement is not cosmetic and the selftest below found the need for it.
    A plain index split gives exactly equal counts but scatters tied scores
    across bins, and each of those bins then shows a different observed rate
    purely from sampling noise. On a constant predictor — which is exactly what
    the prevalence-floor arm is — that manufactures an ECE of about 0.015 where
    the true calibration error is the aggregate gap and is near zero. Reporting
    a constant, correctly-centred predictor as miscalibrated would be an
    artefact of the binning, not a property of the arm.

    The cost is that bin counts are equal only up to the size of a tie run,
    which is why the gate asserts equal counts on a tie-free score and reports
    the achieved counts on the real ones.
    """
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    edges = (np.arange(n_bins + 1) * n) // n_bins

    # Advance every interior boundary past the end of the tie run it sits in.
    snapped = [0]
    for cut in edges[1:-1]:
        cut = max(cut, snapped[-1])
        while 0 < cut < n and sorted_p[cut] == sorted_p[cut - 1]:
            cut += 1
        snapped.append(int(cut))
    snapped.append(n)

    out = np.empty(n, dtype=np.int64)
    for b in range(n_bins):
        out[order[snapped[b]:snapped[b + 1]]] = b
    return out


def reliability(y: np.ndarray, p: np.ndarray, bins: np.ndarray,
                n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin count, mean predicted probability, and observed frequency."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    count = np.bincount(bins, minlength=n_bins).astype(np.float64)
    mean_p = np.divide(np.bincount(bins, weights=p, minlength=n_bins), count,
                       out=np.zeros(n_bins), where=count > 0)
    mean_y = np.divide(np.bincount(bins, weights=y, minlength=n_bins), count,
                       out=np.zeros(n_bins), where=count > 0)
    return count, mean_p, mean_y


def ece(y: np.ndarray, p: np.ndarray, n_bins: int, scheme: str) -> float:
    """Expected calibration error: count-weighted mean gap over occupied bins."""
    if scheme == "equal_width":
        bins = equal_width_bins(p, n_bins)
    elif scheme == "equal_mass":
        bins = equal_mass_bins(p, n_bins)
    else:
        raise ValueError(f"unknown binning scheme {scheme!r}")
    count, mean_p, mean_y = reliability(y, p, bins, n_bins)
    total = count.sum()
    if total == 0:
        return float("nan")
    return float((count * np.abs(mean_y - mean_p)).sum() / total)


def null_band(p: np.ndarray, n_bins: int, scheme: str, n_draws: int = 400,
              seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """The ECE a PERFECTLY calibrated score of this size and shape would show.

    Labels are drawn from `p` itself, so the only departure from perfect
    calibration in each draw is sampling noise. The returned interval is what
    "calibrated" looks like at this n and this bin count; an observed ECE inside
    it is not evidence of miscalibration, and one far above it is.
    """
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        y = (rng.random(p.size) < p).astype(np.int64)
        draws[i] = ece(y, p, n_bins, scheme)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def selftest(n: int = 6000, n_bins: int = 15, seed: int = 0) -> list[str]:
    """Controls that can actually fail. Run as a gate, never assumed."""
    problems: list[str] = []
    rng = np.random.default_rng(seed)

    p = rng.beta(2.0, 8.0, size=n)
    y = (rng.random(n) < p).astype(np.int64)

    for scheme in ("equal_width", "equal_mass"):
        lo, hi = null_band(p, n_bins, scheme, n_draws=300, seed=seed + 1)
        observed = ece(y, p, n_bins, scheme)
        if not (lo <= observed <= hi):
            problems.append(
                f"{scheme}: a calibrated score gives ECE {observed:.4f}, outside "
                f"the simulated null band [{lo:.4f}, {hi:.4f}]. Either the "
                "statistic or the band is wrong"
            )

        # A strictly monotone distortion leaves the RANKING untouched, so it
        # cannot change AUROC, and must move ECE well outside the null band.
        distorted = p ** 2
        moved = ece(y, distorted, n_bins, scheme)
        if moved <= hi:
            problems.append(
                f"{scheme}: squaring the probabilities left ECE at {moved:.4f}, "
                f"inside the null band [{lo:.4f}, {hi:.4f}]. The statistic does "
                "not detect miscalibration"
            )

    # The distortion really is rank-preserving, so the control separates
    # calibration from discrimination rather than measuring both at once.
    try:
        from sklearn.metrics import roc_auc_score

        before = float(roc_auc_score(y, p))
        after = float(roc_auc_score(y, p ** 2))
        if not np.isclose(before, after, atol=1e-12):
            problems.append(
                f"the p -> p**2 control changed AUROC from {before} to {after}; "
                "it is meant to be rank-preserving"
            )
    except ImportError:
        problems.append("scikit-learn is unavailable, so the rank-preservation "
                        "of the ECE control is unchecked")

    # On a TIE-FREE score, equal-mass bins must carry equal counts to within
    # one, or the splitting logic has silently degraded into something else.
    # The qualifier matters: with ties the counts cannot be equal, because tied
    # items are deliberately kept together.
    continuous = rng.random(n)
    counts = np.bincount(equal_mass_bins(continuous, n_bins), minlength=n_bins)
    if counts.max() - counts.min() > 1:
        problems.append(
            f"on a tie-free score, equal-mass bins carry {counts.min()} to "
            f"{counts.max()} items; they must be equal to within one"
        )
    if (counts == 0).any():
        problems.append("an equal-mass bin is empty on a tie-free score")

    # Tied scores must never be split across bins. This is the property that
    # keeps a constant predictor from being reported as miscalibrated.
    tied = np.repeat(np.linspace(0.05, 0.95, 6), n // 6)
    tied_bins = equal_mass_bins(tied, n_bins)
    for value in np.unique(tied):
        if len(np.unique(tied_bins[tied == value])) != 1:
            problems.append(
                f"equal-mass binning split the tied score {value} across bins; "
                "each of those bins would then show a different observed rate "
                "from sampling noise alone"
            )
            break

    # Equal-width bins must actually partition [0, 1] by value.
    widths = equal_width_bins(np.array([0.0, 0.5, 1.0]), n_bins)
    if list(widths) != [0, n_bins // 2, n_bins - 1]:
        problems.append(f"equal-width binning misplaces the endpoints: {widths}")

    # A constant prediction equal to the prevalence is perfectly calibrated in
    # aggregate, and both schemes must agree it is, since there is one occupied
    # bin under either. This is the shape of the prevalence-floor arm.
    prevalence = 0.137
    y_const = (rng.random(n) < prevalence).astype(np.int64)
    p_const = np.full(n, prevalence)
    for scheme in ("equal_width", "equal_mass"):
        value = ece(y_const, p_const, n_bins, scheme)
        if not np.isclose(value, abs(y_const.mean() - prevalence), atol=1e-12):
            problems.append(
                f"{scheme}: a constant prediction gives ECE {value}, not the "
                "absolute gap between the prediction and the observed rate"
            )

    return problems

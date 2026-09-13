#!/usr/bin/env python
"""How a p-value from a resampling procedure is allowed to be printed.

Twenty-seven cells across two panels printed `0`. The estimator is twice the
smaller tail proportion over 10,000 patient-clustered replicates, so the values
it can take are 0, 0.0002, 0.0004 and upwards; `0` is not a small p-value, it is
the statement that no replicate fell on the far side of zero, and a reader who
takes it at face value reads certainty out of a resampling floor.

The floor is read from `configs/base.yaml` rather than written down here,
because it is a property of the design and not of the printing. Raising the
resample count moves the floor, the notation, and the gate that checks it, all
from the one declaration.

One module, three users, on purpose. `make_table_main.py` prints through it,
the corresponding check re-derives the printed text through it, and
the corresponding check asserts against the same notation. A formatter and its
checker that each carry their own copy of the rule are a pair that can disagree.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

#: The format spec recorded in `table_main_cells.csv` for a p-value cell.
#: the corresponding check dispatches on it, which is how a cell printed as
#: `<0.0001` is still re-derived from its source rather than trusted.
SPEC = "pfloor"

#: How a p-value at or above the floor is printed. Unchanged from what the
#: tables already did, so no cell that was legible moves.
ABOVE = ".4g"


def resampling_floor(repo_root: Path) -> float:
    r"""The smallest non-zero p-value this estimator can produce: 2 / B.

    `scripts/bootstrap.py` computes

        p = 2.0 * min((draws <= 0).mean(), (draws >= 0).mean())

    so a tail is k/B and a p-value is 2k/B. The attainable values are 0, 2/B,
    4/B and upwards, with nothing strictly between zero and 2/B. A computed zero
    means no replicate fell on the far side of zero, and what can be said about
    it is that p is below 2/B.

    It is 2/B and not 1/B on purpose. Printing `<0.0001` would assert a p-value
    smaller than the design can resolve, which is a claim made larger by a
    rounding convention.
    """
    config = yaml.safe_load(
        (repo_root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    count = _find(config, "n_resamples")
    if not count:
        raise RuntimeError(
            "configs/base.yaml declares no n_resamples, so there is no "
            "resolution to report a p-value at and nothing here can be honest "
            "about one")
    return 2.0 / float(count)


def _find(node, key: str):
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if isinstance(current.get(key), int):
                return current[key]
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return None


def notation(floor: float) -> str:
    """How a value at or below the floor is printed: `<0.0001` at 10,000."""
    digits = max(1, int(round(-math.log10(floor))))
    return f"<{floor:.{digits}f}"


def format_p(value: float, floor: float) -> str:
    """A p-value, printed at the resolution the resampling can support."""
    value = float(value)
    if value < floor:
        return notation(floor)
    return format(value, ABOVE)

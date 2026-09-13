"""Path resolution and the containment guard.

One rule, enforced in one place: nothing derived from credentialed data, and
nothing that merely looks like it, is ever written inside the repository. Every
writer in this project asks for its output directory here.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Extensions that the containment rule forbids inside the repository.
FORBIDDEN_SUFFIXES = frozenset(
    {".parquet", ".csv", ".npy", ".dat", ".hea", ".pt", ".ckpt", ".gz", ".zip"}
)

#: The four datasets this work requires, in the order they are requested.
REQUIRED_DATASETS = (
    "mimic-iv-ext-mds-ed",
    "mimic-iv-ecg",
    "mimic-iv-ed",
    "mimic-iv",
)


class ContainmentError(RuntimeError):
    """Raised when a write would land inside the repository."""


def is_inside_repo(path: os.PathLike[str] | str) -> bool:
    """True when *path* resolves to the repo root or anything beneath it."""
    candidate = Path(path).expanduser().resolve()
    return candidate == REPO_ROOT or REPO_ROOT in candidate.parents


def assert_outside_repo(path: os.PathLike[str] | str, what: str = "output") -> Path:
    """Return *path* resolved, or raise if it would write inside the repo."""
    resolved = Path(path).expanduser().resolve()
    if is_inside_repo(resolved):
        raise ContainmentError(
            f"containment violation: refusing to write {what} inside the repository.\n"
            f"  requested: {resolved}\n"
            f"  repo root: {REPO_ROOT}\n"
            "Set ECG_DATA_ROOT to a location outside the repository."
        )
    return resolved


def data_root(explicit: str | None = None) -> Path:
    """Resolve the credentialed-data root.

    Order: explicit argument, then ``$ECG_DATA_ROOT``. There is deliberately no
    in-repo default: a missing data root should stop the pipeline, not quietly
    redirect a write into the working tree.
    """
    raw = explicit or os.environ.get("ECG_DATA_ROOT")
    if not raw:
        raise RuntimeError(
            "ECG_DATA_ROOT is not set and no --data-root was given.\n"
            "It must point outside the repository."
        )
    return assert_outside_repo(raw, what="data root")


def derived_root(explicit: str | None = None) -> Path:
    """``$ECG_DATA_ROOT/derived`` — where every derived table lives."""
    return data_root(explicit) / "derived"


def fixtures_root(explicit: str | None = None) -> Path:
    """Synthetic-fixture root.

    Fixtures carry no credentialed content, but they are row-level tables with
    exactly the shape of real ones. Keeping them outside the repo means the
    hygiene guard never has to distinguish a safe table from an unsafe one.
    """
    raw = explicit or os.environ.get("ECG_FIXTURES_ROOT")
    if raw:
        return assert_outside_repo(raw, what="fixtures root")
    base = os.environ.get("ECG_DATA_ROOT")
    if base:
        return assert_outside_repo(Path(base) / "fixtures", what="fixtures root")
    return assert_outside_repo(
        Path.home() / ".cache" / "ecg-acq-context" / "fixtures", what="fixtures root"
    )

"""Locate and read the source tables, validating each against the contract.

Every read in this project goes through :func:`load_source`. That is what makes
the contract binding rather than decorative: a column the pipeline needs but the
release does not have becomes an abort here, with the table and column named,
instead of a silently all-null feature two steps later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from contract import Contract, ContractError, validate_table

#: Where each contract table sits relative to a staged (or fixture) data root.
#: Both layouts are identical by construction, so fixture-tested code paths are
#: the same code paths the real release will take.
FIXTURE_LAYOUT: dict[str, str] = {
    "mimic_iv_ed.edstays": "mimic-iv-ed/edstays.csv.gz",
    "mimic_iv_ed.triage": "mimic-iv-ed/triage.csv.gz",
    "mimic_iv_ecg.record_list": "mimic-iv-ecg/record_list.csv",
    "mimic_iv_ecg.machine_measurements": "mimic-iv-ecg/machine_measurements.csv",
    "mimic_iv_hosp.patients": "mimic-iv/hosp/patients.csv.gz",
    "mimic_iv_hosp.admissions": "mimic-iv/hosp/admissions.csv.gz",
    "mimic_iv_icu.icustays": "mimic-iv/icu/icustays.csv.gz",
    "mds_ed.cohort": "mimic-iv-ext-mds-ed/mds_ed.csv",
}


def resolve_ci(root: Path, relpath: str) -> Path | None:
    """Resolve a relative path one segment at a time, case-insensitively.

    PhysioNet directory names are staged with whatever casing the download used
    (`MIMIC-IV-ECG` here, `mimic-iv-ecg` in the fixtures). Windows hides the
    difference and Linux does not, so the pipeline must not depend on it: the
    same data root must resolve regardless of how the download cased it.
    """
    current = root
    for segment in relpath.split("/"):
        if not current.is_dir():
            return None
        direct = current / segment
        if direct.exists():
            current = direct
            continue
        lowered = segment.lower()
        match = next(
            (child for child in current.iterdir() if child.name.lower() == lowered),
            None,
        )
        if match is None:
            return None
        current = match
    return current

#: Tables the pipeline can proceed without, and what is lost when they are absent.
OPTIONAL_TABLES = {
    "mimic_iv_ecg.machine_measurements": "cardiologist_overread_exists is dropped",
}


class SourceMissing(RuntimeError):
    """A required source table is not present under the data root."""


def read_table(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Read a MIMIC-style CSV, gzipped or not, with the given date columns."""
    return pd.read_csv(
        path,
        compression="gzip" if path.name.endswith(".gz") else None,
        parse_dates=parse_dates or None,
        low_memory=False,
    )


def date_columns(table_ref: str, contract: Contract) -> list[str]:
    spec = contract.table_spec(table_ref)
    return [
        name
        for name, column in (spec.get("columns") or {}).items()
        if str(column.get("dtype", "")).startswith("datetime")
    ]


def resolve_path(table_ref: str, root: Path, contract: Contract) -> Path:
    """Resolve a contract table to a file under *root*.

    Tries the contract's pinned filename first, then the canonical layout, then a
    stem search inside the dataset directory. Every step is case-insensitive, so
    a tree staged as `MIMIC-IV-ECG/` on Windows and one staged as `mimic-iv-ecg/`
    from a case-sensitive filesystem both resolve.
    """
    spec = contract.table_spec(table_ref)
    pinned = spec.get("file")
    layout = FIXTURE_LAYOUT[table_ref]
    parent_rel = "/".join(layout.split("/")[:-1])

    relatives: list[str] = []
    if pinned:
        relatives.append(f"{parent_rel}/{pinned}" if parent_rel else pinned)
    relatives.append(layout)

    for relative in relatives:
        found = resolve_ci(root, relative)
        if found is not None and found.is_file():
            return found

    source_dir = resolve_ci(root, layout.split("/")[0])
    if source_dir is not None and source_dir.is_dir():
        stem = Path(layout).name.split(".")[0]
        for found in sorted(source_dir.rglob(f"*{stem}*.csv*")):
            return found

    raise SourceMissing(
        f"{table_ref}: no file found under {root}.\n"
        f"  tried, case-insensitively: {', '.join(relatives)}\n"
        "Stage the dataset under ECG_DATA_ROOT, or pin the filename in "
        "configs/source_schema.yaml."
    )


def load_source(
    table_ref: str,
    root: Path,
    contract: Contract,
    *,
    required: bool = True,
) -> pd.DataFrame | None:
    """Load one contract table and validate it. Returns ``None`` if optional and absent."""
    try:
        path = resolve_path(table_ref, root, contract)
    except SourceMissing:
        if required and table_ref not in OPTIONAL_TABLES:
            raise
        return None

    frame = read_table(path, parse_dates=date_columns(table_ref, contract))
    problems = validate_table(frame, table_ref, contract)
    if problems:
        raise ContractError(
            f"{table_ref} at {path} violates the source-schema contract:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\nThe contract records an assumption that this release does not "
            "meet. Correct configs/source_schema.yaml and record the correction "
            "in the project's design notes; do not work around it in code."
        )
    return frame


def load_all(
    root: Path, contract: Contract, *, include: list[str] | None = None
) -> dict[str, Any]:
    """Load every contract table under *root*, skipping absent optional ones."""
    wanted = include or list(FIXTURE_LAYOUT)
    loaded: dict[str, Any] = {}
    for table_ref in wanted:
        loaded[table_ref] = load_source(
            table_ref,
            root,
            contract,
            required=table_ref not in OPTIONAL_TABLES,
        )
    return loaded

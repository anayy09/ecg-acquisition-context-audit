"""Load the source-schema contract and validate real tables against it.

The contract in ``configs/source_schema.yaml`` states what the pipeline needs
from the credentialed sources. Nothing in it has been checked against a release.
This module turns that honest ignorance into a loud failure: a table that does
not match the contract stops the pipeline instead of yielding a wrong feature.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from paths import REPO_ROOT

CONTRACT_PATH = REPO_ROOT / "configs" / "source_schema.yaml"
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"

#: Feature blocks the contract and base config must agree on, in report order.
FEATURE_BLOCKS = ("demographics", "acq_pre", "acq_window")

#: Availability rules a derivation may declare, and whether the value is fixed
#: at or before the index acquisition. Anything ``False`` here is leakage-suspect
#: by construction and may not appear in ACQ_PRE.
AVAILABILITY_RULES = {
    "ed_arrival_time": True,
    "index_time": True,
    "window_close": False,
}


class ContractError(RuntimeError):
    """The contract is internally inconsistent, or data violates it."""


@dataclass(frozen=True)
class Derivation:
    name: str
    block: str
    source: tuple[str, ...]
    transform: str
    available_at: str
    dtype: str
    droppable: bool = False
    availability: str | None = None
    leakage_risk: str | None = None

    @property
    def fixed_by_index(self) -> bool:
        return AVAILABILITY_RULES[self.available_at]


@dataclass
class Contract:
    raw: dict[str, Any]
    path: Path
    derivations: dict[str, Derivation] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    @property
    def sources(self) -> dict[str, Any]:
        return self.raw.get("sources", {})

    def block(self, name: str) -> list[Derivation]:
        return [d for d in self.derivations.values() if d.block == name]

    def block_names(self, name: str) -> list[str]:
        return [d.name for d in self.block(name)]

    # -- source-table access -------------------------------------------------

    def table_spec(self, dotted: str) -> dict[str, Any]:
        """Look up ``source.table`` and return its spec."""
        try:
            source, table = dotted.split(".", 1)
        except ValueError as exc:
            raise ContractError(f"malformed table reference {dotted!r}") from exc
        try:
            return self.sources[source]["tables"][table]
        except KeyError as exc:
            raise ContractError(f"undeclared table {dotted!r} in contract") from exc

    def iter_table_refs(self) -> Iterable[tuple[str, dict[str, Any]]]:
        for source_name, source in self.sources.items():
            for table_name, table in source.get("tables", {}).items():
                yield f"{source_name}.{table_name}", table


def load_contract(path: Path | None = None) -> Contract:
    path = Path(path) if path else CONTRACT_PATH
    if not path.is_file():
        raise ContractError(f"source-schema contract not found at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ContractError(f"{path} did not parse to a mapping")

    contract = Contract(raw=raw, path=path)
    derivations = raw.get("derivations") or {}
    if not derivations:
        raise ContractError(f"{path} declares no derivations")

    for name, spec in derivations.items():
        if not isinstance(spec, dict):
            raise ContractError(f"derivation {name!r} is not a mapping")
        missing = [k for k in ("block", "source", "transform", "available_at", "dtype")
                   if k not in spec]
        if missing:
            raise ContractError(f"derivation {name!r} is missing {missing}")
        block = spec["block"]
        if block not in FEATURE_BLOCKS:
            raise ContractError(
                f"derivation {name!r} declares unknown block {block!r}; "
                f"expected one of {FEATURE_BLOCKS}"
            )
        rule = spec["available_at"]
        if rule not in AVAILABILITY_RULES:
            raise ContractError(
                f"derivation {name!r} declares unknown availability rule {rule!r}; "
                f"expected one of {sorted(AVAILABILITY_RULES)}"
            )
        source = spec["source"]
        if isinstance(source, str):
            source = [source]
        contract.derivations[name] = Derivation(
            name=name,
            block=block,
            source=tuple(source),
            transform=spec["transform"],
            available_at=rule,
            dtype=spec["dtype"],
            droppable=bool(spec.get("droppable", False)),
            availability=spec.get("availability"),
            leakage_risk=(
                str(spec["leakage_risk"]).strip() if spec.get("leakage_risk") else None
            ),
        )
    return contract


def load_base_config(path: Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else BASE_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ContractError(f"{path} did not parse to a mapping")
    return raw


def check_contract_consistency(
    contract: Contract, base_config: dict[str, Any]
) -> list[str]:
    """Return every inconsistency between the config, the contract, and the two-block split.

    An empty list is the only acceptable result. This is the check that stops
    ``configs/base.yaml`` and the feature pipeline from drifting apart, which is
    the quiet failure that would let a leakage-suspect column reach a
    confirmatory arm without anyone editing a line that mentions leakage.
    """
    problems: list[str] = []
    declared = base_config.get("features", {}) or {}

    for block in FEATURE_BLOCKS:
        config_names = list(declared.get(block) or [])
        contract_names = contract.block_names(block)

        for name in sorted(set(config_names) - set(contract_names)):
            problems.append(
                f"features.{block}: {name!r} is in configs/base.yaml but has no "
                "derivation in configs/source_schema.yaml"
            )
        for name in sorted(set(contract_names) - set(config_names)):
            problems.append(
                f"features.{block}: {name!r} has a derivation but is not declared "
                "in configs/base.yaml"
            )
        duplicates = {n for n in config_names if config_names.count(n) > 1}
        for name in sorted(duplicates):
            problems.append(f"features.{block}: {name!r} is listed more than once")

    # nothing in ACQ_PRE may resolve after the index acquisition.
    for derivation in contract.block("acq_pre"):
        if not derivation.fixed_by_index:
            problems.append(
                f"block violation: acq_pre feature {derivation.name!r} declares "
                f"available_at={derivation.available_at!r}, which resolves after "
                "the index acquisition"
            )

    # An ACQ_WINDOW feature that claims to be index-fixed is either misfiled or
    # mislabelled. Either way a human has to look at it.
    for derivation in contract.block("acq_window"):
        if derivation.fixed_by_index:
            problems.append(
                f"acq_window feature {derivation.name!r} declares "
                f"available_at={derivation.available_at!r}, which is index-fixed; "
                "either move it to acq_pre with a decision entry, or correct the rule"
            )
        if not derivation.leakage_risk:
            problems.append(
                f"acq_window feature {derivation.name!r} states no leakage_risk; "
                "every leakage-suspect feature must say why it is suspect"
            )

    # Every source referenced by a derivation must be declared as a table.
    for derivation in contract.derivations.values():
        for ref in derivation.source:
            parts = ref.split(".")
            if len(parts) < 2:
                problems.append(
                    f"derivation {derivation.name!r} references {ref!r}, which is "
                    "not a source.table[.column] path"
                )
                continue
            table_ref = ".".join(parts[:2])
            try:
                spec = contract.table_spec(table_ref)
            except ContractError as exc:
                problems.append(f"derivation {derivation.name!r}: {exc}")
                continue
            if len(parts) == 3:
                column = parts[2]
                if column not in (spec.get("columns") or {}):
                    problems.append(
                        f"derivation {derivation.name!r} references column "
                        f"{ref!r}, which the contract does not declare"
                    )

    return problems


def validate_table(
    frame, table_ref: str, contract: Contract, *, strict_dtypes: bool = False
) -> list[str]:
    """Validate a loaded table against its contract entry.

    Returns a list of problems. Column presence is always checked; dtype checking
    is opt-in because a fixture writer and a real ``csv.gz`` reader legitimately
    disagree about integer width and nullability.
    """
    problems: list[str] = []
    spec = contract.table_spec(table_ref)
    columns = spec.get("columns") or {}
    present = set(map(str, frame.columns))

    for column, column_spec in columns.items():
        if column not in present:
            problems.append(f"{table_ref}: required column {column!r} is absent")
            continue
        if not strict_dtypes:
            continue
        expected = str(column_spec.get("dtype", ""))
        actual = str(frame[column].dtype)
        if expected.startswith("datetime") and not actual.startswith("datetime"):
            problems.append(
                f"{table_ref}.{column}: expected a datetime dtype, found {actual!r}"
            )
    return problems

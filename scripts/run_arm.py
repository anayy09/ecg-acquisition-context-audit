#!/usr/bin/env python
"""Fit one tabular arm for one label and one seed, and register it in the ledger.

Both arms of the falsification test go through this single code path. That is the
point: a metadata arm and a demographics arm that differ in their training code
as well as their inputs would confound the comparison the paper rests on. The
only thing that changes between them is which feature block is read.

Tuning is cached per (arm, label) and the identical budget is spent on every arm,
which the analysis protocol forbids getting wrong in either direction.

    python scripts/run_arm.py --arm R3_acqctx_pre --label deterioration_mortality_365d --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import load_base_config, load_contract  # noqa: E402
from paths import REPO_ROOT, data_root, derived_root  # noqa: E402

GENERATED_DIR = REPO_ROOT / "configs" / "generated"

#: Which feature blocks each arm reads. The confirmatory arm reads ACQ_PRE and
#: nothing else; that constraint is enforced here and asserted again by the
#: leakage guard on the emitted table.
ARM_BLOCKS = {
    "R1_prevalence": [],
    "R2_demo": ["demographics"],
    "R3_acqctx_pre": ["acq_pre"],
    # Row 3 and row 4 again, under the waveform arm's selection procedure
    # rather than their own. Same features; see SHARED_CONFIG_ARMS.
    "R3s_acqctx_shared": ["acq_pre"],
    "R4s_demo_acq_shared": ["demographics", "acq_pre"],
    "R3b_acqctx_window": ["acq_pre", "acq_window"],
    "R4_demo_acq": ["demographics", "acq_pre"],
    # Row 5's block is not in the contract as a name list. It is the benchmark's
    # own 463 vitals/labs/biometrics columns, passed through from the release
    # so its names come from the emitted table rather than from a
    # declaration we wrote. `tabular` is resolved by `load_tabular_block`.
    "R5_tabular": ["tabular"],
    # Take no whole block; their features are named in ARM_FEATURES.
    "R9_triage": [],
    "R10_acqctx_informative": [],
}

#: Arms that name their features directly instead of taking whole blocks.
#: Consulted BEFORE `ARM_BLOCKS` and empty for every arm declared before this stage, so
#: no registered arm's feature set can be changed through it.
#:
#: R9 exists because both reviews ask what the acquisition-context block adds
#: beyond the acuity a triage nurse already records, and no arm isolates that
#: feature. It cannot be expressed as a block: `triage_acuity` is declared in
#: `acq_pre` in the source schema, a feature belongs to one block, and moving it
#: would take it out of the confirmatory arm. It is EXPLORATORY: added after the
#: results existed, outside every declared comparison and every Holm family.
#: R10 is the pre-acquisition block with the two calendar features removed. It
#: is named feature by feature for the same reason R9 is: `acq_pre` is one block
#: in the schema and an arm that takes a SUBSET of a block cannot be expressed
#: as a block without redefining the block the confirmatory arm uses. Also
#: EXPLORATORY, and also outside every declared comparison.
ARM_FEATURES = {
    "R9_triage": ["triage_acuity"],
    "R10_acqctx_informative": [
        "arrival_to_acquisition_minutes",
        "acquisition_hour_of_day",
        "triage_acuity",
        "ecg_no_within_stay",
        "prior_ecg_exists",
        "days_since_prior_ecg",
        "prior_ed_visit_count",
    ],
}

#: Blocks whose names come from the contract. `tabular` is deliberately absent.
CONTRACT_BLOCKS = ("demographics", "acq_pre", "acq_window")

#: Row 5's expected column count, from the published `input_channels_cont: 463`.
TABULAR_COLUMNS = 463

SEARCH_SPACE = {
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [10, 20, 50, 100],
    "learning_rate": [0.02, 0.05, 0.1],
    "feature_fraction": [0.6, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "lambda_l2": [0.0, 1.0, 10.0],
}


#: Paths whose modification does not make a run irreproducible. The ledger is
#: append-only and tracked, so registering run N necessarily modifies it and
#: every later run in the same batch then sees a "dirty" tree. Run outputs are
#: likewise products of the batch, not inputs to it.
NON_SOURCE_PREFIXES = ("ledger/", "runs/", "results/", "notes/")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_dirty() -> tuple[bool, list[str]]:
    """Is any *source* file modified relative to HEAD?

    This is the question the ledger's `git_dirty` flag is trying to answer but
    cannot, because the registry it writes to is itself tracked. Anything under
    NON_SOURCE_PREFIXES is excluded; everything else — code, configs, the
    comparison declaration — counts.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain", "-uno"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        return True, ["git status failed"]
    offenders = []
    for line in result.stdout.splitlines():
        path = line[3:].strip().strip('"')
        if path and not path.startswith(NON_SOURCE_PREFIXES):
            offenders.append(path)
    return bool(offenders), offenders


def sanitise(name: str) -> str:
    """LightGBM refuses JSON-special characters in a feature name.

    45 of row 5's 463 passthrough columns carry a comma, for example
    `labvalues_bilirubin,_direct_mean`, and LightGBM fails the fit outright with
    "Do not support special JSON characters in feature name". This maps names
    only. No value, no ordering and no missingness is touched, and the stored
    parquet keeps the release's own names. Injectivity is asserted at load time,
    because a mapping that collided would silently merge two lab columns.
    """
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name)


def load_tabular_block(derived: Path) -> tuple[pd.DataFrame, list[str]]:
    """Row 5's block, verified against the hash `build_tabular.py` recorded.

    The pass-through rule for row 5 makes row 5 the benchmark's own table rather than one of ours, so the
    thing that matters is that the table has not moved underneath the run. The
    hash is checked here rather than trusted, on every run.
    """
    table = derived / "features_tabular.parquet"
    meta_path = derived / "features_tabular_meta.json"
    if not table.is_file() or not meta_path.is_file():
        raise RuntimeError(
            f"row 5 needs {table.name} and {meta_path.name}; run scripts/build_tabular.py"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    digest = _sha256(table)
    if digest != meta.get("hash"):
        raise RuntimeError(
            f"{table.name} hashes to {digest[:12]} but {meta_path.name} pins "
            f"{str(meta.get('hash'))[:12]}. The passthrough table has changed since "
            "it was built; refusing to fit row 5 against an unpinned table."
        )

    frame = pd.read_parquet(table)
    columns = [c for c in frame.columns if c != "study_id"]
    if len(columns) != TABULAR_COLUMNS:
        raise RuntimeError(
            f"{table.name} carries {len(columns)} feature columns; the published "
            f"config declares input_channels_cont: {TABULAR_COLUMNS}"
        )
    mapped = {c: sanitise(c) for c in columns}
    if len(set(mapped.values())) != len(columns):
        collisions = sorted(
            {v for v in mapped.values() if list(mapped.values()).count(v) > 1}
        )
        raise RuntimeError(
            f"the feature-name map is not injective on this release: {collisions[:5]} "
            "collide. Two columns would silently merge; refusing."
        )
    frame = frame.rename(columns=mapped)
    return frame, [mapped[c] for c in columns]


def load_matrix(derived: Path, arm: str, label: str, contract) -> dict:
    """Assemble X, y, split, and groups for one arm and label."""
    acq = pd.read_parquet(derived / "features_acq.parquet")
    demo = pd.read_parquet(derived / "features_demo.parquet")
    cohort = pd.read_parquet(derived / "cohort_mdsed.parquet")

    frame = cohort[["study_id", "subject_id", "split", label]].merge(
        demo.drop(columns=[c for c in demo.columns if c.startswith("available_at__")]),
        on="study_id",
        how="left",
        suffixes=("", "_demo"),
    )
    acq_cols = [c for c in acq.columns if not c.startswith("available_at__")]
    frame = frame.merge(acq[acq_cols], on="study_id", how="left", suffixes=("", "_acq"))

    tabular_names: list[str] = []
    if "tabular" in ARM_BLOCKS[arm]:
        tabular, tabular_names = load_tabular_block(derived)
        frame = frame.merge(tabular, on="study_id", how="left", suffixes=("", "_tab"))

    sentinel = contract.raw.get("label_missing_sentinel", -999)
    before = len(frame)
    frame = frame[frame[label] != sentinel].copy()
    dropped = before - len(frame)

    features: list[str] = []
    if arm in ARM_FEATURES:
        # An explicitly named feature set. Missing columns are fatal here rather
        # than filtered away: an arm that asked for one feature and quietly fit
        # on none would score at chance and read as a finding.
        features = list(ARM_FEATURES[arm])
        absent = [f for f in features if f not in frame.columns]
        if absent:
            raise SystemExit(
                f"{arm} names feature(s) {absent} that are not in the built "
                "frame. An arm fitting on fewer features than it declares is "
                "not the arm that was declared")
    else:
        for block in ARM_BLOCKS[arm]:
            if block == "tabular":
                features += tabular_names
            else:
                features += contract.block_names(block)
        features = [f for f in features if f in frame.columns]

    for column in features:
        if frame[column].dtype == bool:
            frame[column] = frame[column].astype("int8")
        elif str(frame[column].dtype) in {"object", "category", "string"}:
            frame[column] = frame[column].astype("category")

    return {
        "frame": frame,
        "features": features,
        "label": label,
        "dropped_sentinel_rows": dropped,
    }


def _fit(params: dict, matrix: dict, train, valid, seed: int, n_estimators: int = 2000):
    import lightgbm as lgb

    frame, features, label = matrix["frame"], matrix["features"], matrix["label"]
    if not features:
        return None  # prevalence floor: constant predictor, no model

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    model.fit(
        frame.loc[train, features],
        frame.loc[train, label],
        eval_set=[(frame.loc[valid, features], frame.loc[valid, label])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def tune(matrix: dict, train, valid, budget: int, cache: Path) -> dict:
    """Random search on the validation split only. Cached per (arm, label)."""
    from sklearn.metrics import roc_auc_score

    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))

    rng = np.random.default_rng(0)
    frame, features, label = matrix["frame"], matrix["features"], matrix["label"]
    best = {"params": {}, "val_auroc": -1.0, "trials": 0}

    for _ in range(budget):
        params = {k: rng.choice(v).item() for k, v in SEARCH_SPACE.items()}
        params["bagging_freq"] = 1
        model = _fit(params, matrix, train, valid, seed=0)
        if model is None:
            break
        score = roc_auc_score(
            frame.loc[valid, label],
            model.predict_proba(frame.loc[valid, features])[:, 1],
        )
        if score > best["val_auroc"]:
            best = {"params": params, "val_auroc": float(score), "trials": budget}

    best["trials"] = budget
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


#: Arms selected the way the waveform arm was: ONE configuration for every
#: label, chosen on a validation macro across all the deterioration targets,
#: rather than a per-target search. The value is the arm whose features they
#: borrow, which is also what makes the pairing checkable: a shared-config arm
#: must read exactly the blocks its source arm reads.
SHARED_CONFIG_ARMS = {
    "R3s_acqctx_shared": "R3_acqctx_pre",
    "R4s_demo_acq_shared": "R4_demo_acq",
}

for _shared, _source in SHARED_CONFIG_ARMS.items():
    if ARM_BLOCKS[_shared] != ARM_BLOCKS[_source]:
        raise RuntimeError(
            f"{_shared} is meant to be {_source} under a different selection "
            "procedure and reads different blocks, so any difference between "
            "them would confound the thing it exists to measure")


def macro_labels(derived) -> list[str]:
    """The deterioration targets the shared configuration is selected on.

    Read off the built cohort rather than listed here, so this is the same set
    the waveform arm's own selection macro ran over instead of a second list
    that can drift away from it. The waveform arm is multi-label over all 15 of
    them in one run; a shared tabular configuration has to be chosen on the same
    set for the comparison to mean anything.
    """
    import pandas as pd

    frame = pd.read_parquet(derived / "cohort_mdsed.parquet")
    labels = sorted(c for c in frame.columns
                    if str(c).startswith("deterioration_"))
    if len(labels) != 15:
        raise RuntimeError(
            f"the cohort carries {len(labels)} deterioration targets and the "
            "waveform arm was selected on 15; a shared configuration chosen on "
            "a different set is not the comparison this arm exists to make")
    return labels


def tune_shared(arm: str, derived, contract, budget: int, cache: Path) -> dict:
    """One configuration for all labels, scored on a validation macro.

    The waveform arm was selected from four candidate configurations on a
    validation macro across every deterioration target, and this mirrors that:
    the same random search this project already uses, but each candidate is
    scored once as the mean validation AUROC over all the targets rather than
    separately per target. The winner is then fitted at every reported label.

    Cached per ARM, not per (arm, label), because one configuration for all
    labels is the whole point. A cache key that carried the label would quietly
    restore per-target tuning.
    """
    import json

    import numpy as np
    from sklearn.metrics import roc_auc_score

    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))

    labels = macro_labels(derived)
    matrices = {}
    for label in labels:
        try:
            matrices[label] = load_matrix(derived, arm, label, contract)
        except Exception as exc:  # a label with no usable rows is skipped, loudly
            print(f"  {label}: not loadable for {arm} ({exc})", file=sys.stderr)
    if not matrices:
        raise RuntimeError(f"no label matrices loaded for {arm}")

    rng = np.random.default_rng(0)
    best = {"params": {}, "val_macro": -1.0, "trials": 0,
            "n_labels": len(matrices), "labels": sorted(matrices)}

    for trial in range(budget):
        params = {k: rng.choice(v).item() for k, v in SEARCH_SPACE.items()}
        params["bagging_freq"] = 1
        scores = []
        for label, matrix in matrices.items():
            frame = matrix["frame"]
            train = frame["split"] == "train"
            valid = frame["split"] == "val"
            model = _fit(params, matrix, train, valid, seed=0)
            if model is None:
                continue
            y = frame.loc[valid, label]
            if y.nunique() < 2:
                continue
            scores.append(roc_auc_score(
                y, model.predict_proba(frame.loc[valid, matrix["features"]])[:, 1]))
        if not scores:
            continue
        macro = float(np.mean(scores))
        if macro > best["val_macro"]:
            best.update(params=params, val_macro=macro,
                        n_scored=len(scores), trial=trial)

    best["trials"] = budget
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


COMPARISONS_DIR = REPO_ROOT / "comparisons"


def comparison_members(name: str) -> tuple[set[str], set[str]]:
    """The arms and labels a declared comparison actually covers.

    Two families exist and they are never pooled. `acq_context_audit` was
    fixed on 2026-08-23 and does not grow, so a run tagged to it whose label it
    never declared would be a silent family expansion — exactly what this project
    audits other work for. The membership is read from the declaration file
    rather than from a list in code, so the two cannot drift.
    """
    import yaml

    path = COMPARISONS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise RuntimeError(f"no declared comparison at {path}")
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))

    arms = {a["id"] for a in spec.get("arms", []) if isinstance(a, dict) and "id" in a}
    labels: set[str] = set()
    for value in spec.get("labels", []) or []:
        labels.add(str(value))
    for key in ("primary_labels", "generating_labels", "anchor_label"):
        for item in spec.get(key, []) or []:
            if isinstance(item, dict) and "label" in item:
                labels.add(str(item["label"]))
    return arms, labels


def assert_declared(comparison: str, arm: str, label: str) -> None:
    arms, labels = comparison_members(comparison)
    if arm not in arms:
        raise RuntimeError(
            f"comparison {comparison!r} does not declare arm {arm!r}; tagging this "
            "run to it would expand a fixed family"
        )
    if label not in labels:
        raise RuntimeError(
            f"comparison {comparison!r} does not declare label {label!r}; tagging "
            "this run to it would expand a fixed family"
        )


def write_overlay(arm: str, label: str, seed: int, comparison: str) -> Path:
    """One committed config per run. The run_id hashes it, so seeds differ."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{arm}__{label}__seed{seed}.yaml"
    path.write_text(
        "# Generated by scripts/run_arm.py. One file per (arm, label, seed) so the\n"
        "# resolved config, and therefore the run_id, is distinct per seed.\n"
        f"arm: {arm}\n"
        f"comparison: {comparison}\n"
        f"label_active: {label}\n"
        "training:\n"
        f"  seed: {seed}\n",
        encoding="utf-8",
    )
    return path


def register(arm: str, label: str, seed: int, overlay: Path, split_file: Path,
             comparison: str) -> str:
    result = subprocess.run(
        [
            sys.executable, "scripts/ledger.py", "new",
            "--config", "configs/base.yaml",
            "--overlay", f"configs/arms/{arm}.yaml",
            "--overlay", str(overlay.relative_to(REPO_ROOT)).replace("\\", "/"),
            "--tag", f"arm={arm}", "--tag", f"label={label}", "--tag", f"seed={seed}",
            "--tag", f"comparison={comparison}",
            "--tag", f"group={arm}__{label}",
            "--split", str(split_file),
            "--entrypoint", "scripts/run_arm.py",
            "--quiet", "--force",
        ],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"ledger new failed:\n{result.stdout}{result.stderr}")
    return result.stdout.strip().splitlines()[-1].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARM_BLOCKS))
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--comparison", default="acq_context_audit",
                        help="the declared family this run belongs to. It must "
                             "declare both this arm and this label; the two "
                             "families are never pooled.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    assert_declared(args.comparison, args.arm, args.label)

    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    contract = load_contract()
    base = load_base_config()
    root = data_root(args.data_root)
    derived = derived_root(args.data_root)

    started = time.time()
    matrix = load_matrix(derived, args.arm, args.label, contract)
    frame = matrix["frame"]
    train = frame["split"] == "train"
    valid = frame["split"] == "val"
    test = frame["split"] == "test"

    budget = int(base["model"]["tabular"]["tuning_budget_trials"])
    if args.arm in SHARED_CONFIG_ARMS:
        # One configuration for every label, selected on a validation macro
        # across all the deterioration targets, which is how the waveform arm
        # was selected. The BUDGET matches too: the waveform arm chose from four
        # candidate configurations, so these choose from four. Giving them the
        # fifty-trial tabular budget would have handed the shared-configuration
        # arm more search than the arm it is imitating, which is the opposite of
        # the comparison this sensitivity exists to make. The cache is keyed by
        # arm alone, because one configuration for all labels is the point.
        shared_budget = len(base["model"]["waveform"]["tuning_candidates"])
        cache = derived / "tuning" / f"{args.arm}__shared.json"
        best = tune_shared(args.arm, derived, contract, shared_budget, cache)
        best.setdefault("val_auroc", best.get("val_macro", float("nan")))
    else:
        cache = derived / "tuning" / f"{args.arm}__{args.label}.json"
        best = tune(matrix, train, valid, budget, cache)

    model = _fit(best["params"], matrix, train, valid, seed=args.seed)
    if model is None:
        scores = np.full(int(test.sum()), float(frame.loc[train, args.label].mean()))
    else:
        scores = model.predict_proba(frame.loc[test, matrix["features"]])[:, 1]

    y_true = frame.loc[test, args.label].to_numpy()
    metrics = {
        "auroc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "n_test_items": int(test.sum()),
        "n_test_groups": int(frame.loc[test, "subject_id"].nunique()),
        "n_test_positives": int(y_true.sum()),
        "prevalence_test": float(y_true.mean()),
        "n_train_items": int(train.sum()),
        "dropped_sentinel_rows": int(matrix["dropped_sentinel_rows"]),
        "n_features": len(matrix["features"]),
        "tuning_trials": int(best["trials"]),
        "val_auroc_at_tuning": float(best["val_auroc"]),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "fit_seconds": round(time.time() - started, 2),
    }
    dirty, offenders = source_dirty()
    # metrics.json feeds `ledger record`, which accepts numeric values only, so
    # the offending path list goes in a sibling provenance file.
    metrics["source_dirty"] = int(dirty)

    overlay = write_overlay(args.arm, args.label, args.seed, args.comparison)
    split_file = derived / "split_published.parquet"
    run_id = register(args.arm, args.label, args.seed, overlay, split_file,
                      args.comparison)

    # Row-level predictions carry subject_id, so they live under the data root,
    # never in the repository. The containment rule, and the row-level data rule for this specific choice.
    pred_dir = derived / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{run_id}.parquet"
    pd.DataFrame(
        {
            "subject_id": frame.loc[test, "subject_id"].to_numpy(),
            "study_id": frame.loc[test, "study_id"].to_numpy(),
            "y_true": y_true,
            "y_score": scores,
        }
    ).to_parquet(pred_path, index=False)

    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.pointer.json").write_text(
        json.dumps(
            {
                "path": str(pred_path),
                "sha256": _sha256(pred_path),
                "n_rows": int(test.sum()),
                "note": "Row-level predictions are keyed by subject_id and are kept "
                        "outside the repository under $ECG_DATA_ROOT. See the containment rule, the row-level data rule.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "source_dirty": bool(dirty),
                "source_dirty_paths": sorted(offenders),
                "non_source_prefixes": list(NON_SOURCE_PREFIXES),
                "note": "ledger.py's git_dirty flag also fires on the tracked "
                        "append-only registry, so it cannot distinguish a code "
                        "change from a run being registered.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/ledger.py", "record", "--run", run_id,
         "--metrics", f"runs/{run_id}/metrics.json"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"ledger record failed:\n{result.stdout}{result.stderr}")

    if not args.quiet:
        print(
            f"{run_id}  {args.arm:<16} {args.label:<30} seed={args.seed}  "
            f"AUROC={metrics['auroc']:.4f}  n={metrics['n_test_items']:,} "
            f"groups={metrics['n_test_groups']:,}"
        )
    else:
        print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

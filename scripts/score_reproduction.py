#!/usr/bin/env python
"""Score the waveform arms: per-target AUROC, the macro, and the reproduction gap.

The quantity the corresponding check tests is a macro over 15 targets, so its interval has to be
computed on the macro itself. Averaging fifteen separately-computed intervals
would be wrong twice over: it ignores that one patient contributes to all fifteen
targets at once, and it produces an interval for a quantity nobody reported. So a
patient is resampled once per replicate, all fifteen AUROCs are recomputed on
that same resample, and the macro is taken from them. The correlation between
targets is then carried rather than assumed away.

Two units are reported for every arm, per the crop-level and record-level separation. The crop-level macro is what the
published protocol computes and therefore what the reproduction gap is measured
against. The record-level macro is one row per ECG and is what every cross-arm
contrast uses, because rows 1 to 5 have no crops.

    python scripts/score_reproduction.py
    python scripts/score_reproduction.py --n-boot 2000     # faster, for a smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import ClusterIndex, auc_from_ranks, rank_scores, selftest  # noqa: E402
from paths import REPO_ROOT, derived_root  # noqa: E402

WAVEFORM_ARMS = ("R6_waveform", "R7_waveform_demo_acq")
RESULTS = REPO_ROOT / "results"


def macro_clustered_ci(
    y: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    groups: np.ndarray,
    targets: list[str],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Per-target AUROC and the macro, all on one patient-clustered bootstrap.

    `mask` marks where a target is defined. A target undefined for a row simply
    contributes no items for that target on that row; it does not remove the row
    from the others. Targets that lose a class within a replicate return NaN for
    that replicate and are skipped by the nanmean, which is what makes the
    interval on the macro honest when six of fifteen targets carry fewer than 50
    positives.
    """
    index = ClusterIndex(np.asarray(groups))
    rng = np.random.default_rng(seed)
    n_targets = len(targets)

    ranks, buckets = [], []
    point = np.full(n_targets, np.nan)
    for k in range(n_targets):
        keep = mask[:, k] > 0
        r, b = rank_scores(scores[:, k])
        ranks.append(r)
        buckets.append(b)
        if keep.any() and len(np.unique(y[keep, k])) > 1:
            point[k] = auc_from_ranks(y[keep, k].astype(np.int64), r[keep], b)

    draws = np.full((n_boot, n_targets), np.nan)
    for i in range(n_boot):
        idx = index.draw(rng)
        for k in range(n_targets):
            sel = idx[mask[idx, k] > 0]
            if sel.size == 0:
                continue
            yy = y[sel, k].astype(np.int64)
            if yy.min() == yy.max():
                continue
            draws[i, k] = auc_from_ranks(yy, ranks[k][sel], buckets[k])

    macro_draws = np.nanmean(draws, axis=1)
    lo, hi = np.nanpercentile(macro_draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    per = {}
    for k, name in enumerate(targets):
        col = draws[:, k]
        if np.isfinite(col).any():
            tlo, thi = np.nanpercentile(col, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        else:
            tlo = thi = float("nan")
        per[name] = {
            "auroc": float(point[k]),
            "lo": float(tlo),
            "hi": float(thi),
            "n_defined": int((mask[:, k] > 0).sum()),
            "n_positive": int(y[mask[:, k] > 0, k].sum()),
            "nan_replicates": int(np.isnan(col).sum()),
        }
    return {
        "per_target": per,
        "macro": float(np.nanmean(point)),
        "macro_lo": float(lo),
        "macro_hi": float(hi),
        "n_boot": int(n_boot),
        "n_groups": int(index.n_groups),
        "n_items": int(len(y)),
        "n_targets_scored": int(np.isfinite(point).sum()),
    }


def load_runs(derived: Path) -> list[dict]:
    """Every registered waveform run with predictions on disk."""
    tags: dict[str, dict] = {}
    ledger = REPO_ROOT / "ledger" / "runs.jsonl"
    for line in ledger.read_text(encoding="utf-8").splitlines() if ledger.is_file() else []:
        if line.strip():
            rec = json.loads(line)
            if rec.get("tags"):
                tags[rec["run_id"]] = rec["tags"]
    out = []
    for run_dir in sorted((REPO_ROOT / "runs").iterdir()):
        tag = tags.get(run_dir.name, {})
        if tag.get("arm") not in WAVEFORM_ARMS:
            continue
        pointer = run_dir / "predictions.pointer.json"
        if not pointer.is_file():
            continue
        info = json.loads(pointer.read_text(encoding="utf-8"))
        path = Path(info["path"])
        if not path.is_file():
            print(f"  {run_dir.name}: predictions missing at {path}", file=sys.stderr)
            continue
        out.append({"run_id": run_dir.name, "arm": tag["arm"], "seed": int(tag["seed"]),
                    "predictions": path})
    return out


def published_anchors(published: dict) -> list[dict]:
    """The declared published figures for the waveform-only deterioration macro.

    Two versions of the audited benchmark exist and they disagree about this
    number, which is the whole of review 1's first objection. The gate had
    compared against one of them since this stage without naming it. Returned in
    reporting order, primary first, with the role carried so that a table cannot
    show one and imply it is the only one.
    """
    declared = published.get("reproduction_anchors")
    if not declared:
        raise RuntimeError(
            "configs/published_reference.yaml declares no reproduction_anchors, "
            "so there is no way to say which published figure the gate is "
            "judged against, and that ambiguity is the defect this replaced")

    known = {
        "preprint_v2": (
            published["model_auroc_macro_deterioration"]["ecg_waveforms_only"],
            published["model_auroc_macro_deterioration"]["source"]),
        "version_of_record": (
            published["version_of_record"]["macro_deterioration"]["ecg_waveforms_only"],
            published["version_of_record"]["macro_deterioration"]["source"]),
    }

    anchors = []
    for role in ("primary", "secondary"):
        name = declared.get(role)
        if not name:
            continue
        if name not in known:
            raise RuntimeError(
                f"reproduction_anchors.{role} names {name!r}, which this script "
                "cannot resolve to a published figure")
        figure, source = known[name]
        anchors.append({"name": name, "role": role, "source": source,
                        "point": float(figure["point"]),
                        "ci": [float(figure["ci"][0]), float(figure["ci"][1])]})
    return anchors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    import yaml

    failures = selftest()
    if failures:
        for line in failures:
            print(f"bootstrap selftest: {line}", file=sys.stderr)
        return 1
    print("bootstrap self-test passed against scikit-learn")

    derived = derived_root(args.data_root)
    runs = load_runs(derived)
    if not runs:
        print("no waveform runs with predictions; nothing to score", file=sys.stderr)
        return 1
    print(f"scoring {len(runs)} run(s)")

    published = yaml.safe_load(
        (REPO_ROOT / "configs" / "published_reference.yaml").read_text(encoding="utf-8")
    )
    anchors = published_anchors(published)

    rows, per_target_rows = [], []
    for run in runs:
        blob = np.load(run["predictions"], allow_pickle=True)
        targets = [str(t) for t in blob["targets"]]
        subject = blob["subject_id"]
        for unit in ("crop", "record"):
            if unit == "crop":
                y, sc, mk = blob["crop_y"], blob["crop_scores"], blob["crop_mask"]
                groups = subject[blob["crop_local"]]
            else:
                y, sc, mk = blob["record_y"], blob["record_scores"], blob["record_mask"]
                groups = subject
            res = macro_clustered_ci(y, sc, mk, groups, targets,
                                     n_boot=args.n_boot, seed=args.seed)
            rows.append({
                "run_id": run["run_id"], "arm": run["arm"], "seed": run["seed"],
                "unit": unit, "macro": res["macro"], "macro_lo": res["macro_lo"],
                "macro_hi": res["macro_hi"], "n_items": res["n_items"],
                "n_groups": res["n_groups"], "n_boot": res["n_boot"],
                "n_targets_scored": res["n_targets_scored"],
            })
            for name, stats in res["per_target"].items():
                per_target_rows.append({
                    "run_id": run["run_id"], "arm": run["arm"], "seed": run["seed"],
                    "unit": unit, "target": name, **stats,
                })
        print(f"  {run['run_id']} {run['arm']} seed {run['seed']}: "
              f"crop {rows[-2]['macro']:.4f}, record {rows[-1]['macro']:.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "p2_macro.csv", index=False)
    pd.DataFrame(per_target_rows).to_csv(RESULTS / "p2_per_target.csv", index=False)

    # The reproduction gap, on the crop-level macro the published number uses,
    # against every declared anchor rather than against one unnamed figure.
    gap_rows = []
    for arm in sorted(frame["arm"].unique()):
        sub = frame[(frame["arm"] == arm) & (frame["unit"] == "crop")]
        if sub.empty:
            continue
        seed_macros = sub["macro"].to_numpy()
        mean_macro = float(seed_macros.mean())
        for anchor in anchors:
            gap_rows.append({
                "arm": arm,
                "unit": "crop",
                "local_macro_mean": mean_macro,
                "local_macro_min": float(seed_macros.min()),
                "local_macro_max": float(seed_macros.max()),
                "seed_spread": float(seed_macros.max() - seed_macros.min()),
                "n_seeds": int(len(seed_macros)),
                "local_ci_lo": float(sub["macro_lo"].mean()),
                "local_ci_hi": float(sub["macro_hi"].mean()),
                "anchor": anchor["name"],
                "anchor_role": anchor["role"],
                "anchor_source": anchor["source"],
                "published_macro": anchor["point"],
                "published_lo": anchor["ci"][0],
                "published_hi": anchor["ci"][1],
                "gap": mean_macro - anchor["point"],
                "inside_published_ci": bool(
                    anchor["ci"][0] <= mean_macro <= anchor["ci"][1]
                ),
            })
    pd.DataFrame(gap_rows).to_csv(RESULTS / "p2_reproduction_gap.csv", index=False)

    for row in gap_rows:
        verdict = "INSIDE" if row["inside_published_ci"] else "OUTSIDE"
        print(f"\n{row['arm']} vs {row['anchor']} ({row['anchor_role']}): "
              f"local crop macro {row['local_macro_mean']:.4f} "
              f"(seed spread {row['seed_spread']:.4f}, {row['n_seeds']} seeds)")
        print(f"  published {row['published_macro']:.4f} "
              f"[{row['published_lo']:.4f}, {row['published_hi']:.4f}]  "
              f"gap {row['gap']:+.4f}  -> {verdict} the published interval")
    print(f"\nwrote {RESULTS / 'p2_macro.csv'}, p2_per_target.csv, p2_reproduction_gap.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

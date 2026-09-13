#!/usr/bin/env python
"""Score this stage: the nested rows, the two families, the recovery ratio and the trends.

Everything here is at RECORD level, one row per ECG, because rows 1 to 5 have no
crops. The crop-level waveform arrays are never opened.

Four conventions are fixed by the declared scoring conventions and implemented once, here:

  unit          record level throughout; `_waveform_pooled` reads `record_*` only
  denominator   the recovery ratio divides by R6 AT THE SAME LABEL, not by the
                15-target macro
  noise floor   measured per (arm, label) at record level, not as one scalar
  seed pooling  scores are averaged across the five seeds and one AUROC is taken
                on the pooled score, because a paired bootstrap needs one score
                vector per arm; per-seed AUROCs are reported as the spread

The two families are computed separately and corrected separately.
They are never pooled and this file has no code path that could pool them.

    python scripts/score_arms.py
    python scripts/score_arms.py --n-boot 2000     # faster, for a smoke test
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import (  # noqa: E402
    ClusterIndex,
    auc_from_ranks,
    clustered_auc_ci,
    clustered_statistic_ci,
    paired_clustered_auc_diff,
    rank_scores,
    selftest,
    spearman,
)
from contract import load_base_config  # noqa: E402
from paths import REPO_ROOT, derived_root  # noqa: E402

RESULTS = REPO_ROOT / "results"
COMPARISONS = REPO_ROOT / "comparisons"
WAVEFORM_ARMS = ("R6_waveform", "R7_waveform_demo_acq")
REFERENCE_ARM = "R6_waveform"

#: Rows 1 to 7 of the headline table. Row 8 arrives in this stage.
P3_ARMS = (
    "R1_prevalence", "R2_demo", "R3_acqctx_pre", "R3b_acqctx_window",
    "R4_demo_acq", "R5_tabular", "R6_waveform", "R7_waveform_demo_acq",
)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def ledger_index() -> dict[tuple[str, str, int], str]:
    """(arm, label, seed) -> run_id for the CURRENT run of each key.

    The registry is append-only and re-registers a key under a new id whenever
    the resolved config changes, so the last `created` event for a key wins and
    ids whose run directory no longer exists are dropped.
    """
    current: dict[tuple[str, str, int], str] = {}
    path = REPO_ROOT / "ledger" / "runs.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") != "created":
            continue
        tags = event.get("tags") or {}
        if {"arm", "label", "seed"} <= tags.keys():
            current[(tags["arm"], tags["label"], int(tags["seed"]))] = event["run_id"]
    return {k: v for k, v in current.items() if (REPO_ROOT / "runs" / v).is_dir()}


def _tabular_seed_frame(run_id: str, derived: Path) -> pd.DataFrame:
    frame = pd.read_parquet(derived / "predictions" / f"{run_id}.parquet")
    return frame[["study_id", "subject_id", "y_true", "y_score"]]


def _waveform_seed_frame(run_id: str, derived: Path, label: str) -> pd.DataFrame:
    """One record-level column of a multi-label waveform run.

    The archive holds `crop_*` and `record_*` side by side. Only `record_*` is
    read: a crop-level score is four correlated rows per ECG and must never share
    a column with a tabular arm. `record_mask` marks where the
    target is defined, which is the waveform side of the same `-999` sentinel the
    tabular arms drop.
    """
    blob = np.load(derived / "predictions" / f"{run_id}.npz", allow_pickle=True)
    targets = [str(t) for t in blob["targets"]]
    if label not in targets:
        raise RuntimeError(f"{run_id} does not carry target {label}")
    k = targets.index(label)
    keep = blob["record_mask"][:, k] > 0
    return pd.DataFrame({
        "study_id": blob["study_id"][keep],
        "subject_id": blob["subject_id"][keep],
        "y_true": blob["record_y"][keep, k].astype(np.int64),
        "y_score": blob["record_scores"][keep, k].astype(float),
    })


def pooled_arm(arm: str, label: str, seeds: list[int], index: dict, derived: Path):
    """Seed-pooled scores for one (arm, label), plus each seed's own AUROC.

    The declared scoring conventions: the pooled score is the mean across seeds and carries the reported
    AUROC; the per-seed AUROCs are the spread, and their max-minus-min is this
    cell's noise floor.
    """
    frames, per_seed = [], []
    for seed in seeds:
        run_id = index.get((arm, label, seed)) or index.get((arm, "all15", seed))
        if run_id is None:
            return None
        if arm in WAVEFORM_ARMS:
            frame = _waveform_seed_frame(run_id, derived, label)
        else:
            frame = _tabular_seed_frame(run_id, derived)
        ranks, buckets = rank_scores(frame["y_score"].to_numpy())
        per_seed.append(auc_from_ranks(frame["y_true"].to_numpy().astype(np.int64),
                                       ranks, buckets))
        frames.append(frame.set_index("study_id"))

    base = frames[0]
    wide = pd.concat([f["y_score"].rename(f"s{i}") for i, f in enumerate(frames)],
                     axis=1, join="inner")
    pooled = base.loc[wide.index, ["subject_id", "y_true"]].copy()
    pooled["y_score"] = wide.mean(axis=1)
    return pooled.reset_index(), np.asarray(per_seed, dtype=float)


# --------------------------------------------------------------------------
# cells: one joined frame per label, shared patient universe across labels
# --------------------------------------------------------------------------
def join_arms(pooled: dict, arms: list[str], label: str):
    """Inner-join several arms on study_id for one label.

    The arms do not share a row set. Tabular arms drop the label's `-999`
    sentinel rows and the waveform arm drops rows its mask leaves undefined, so
    the contrast's unit is the intersection and its item and group counts are
    reported from the intersection rather than from either arm alone.
    """
    frames = []
    for arm in arms:
        entry = pooled.get((arm, label))
        if entry is None:
            return None
        frame = entry[0].set_index("study_id")
        frames.append(frame["y_score"].rename(arm))
    joined = pooled[(arms[0], label)][0].set_index("study_id")[["subject_id", "y_true"]]
    for frame in frames:
        joined = joined.join(frame, how="inner")
    return joined.dropna()


class Cell:
    """One label's joined evidence, with ranks precomputed per arm."""

    def __init__(self, joined: pd.DataFrame, arms: list[str]) -> None:
        self.arms = arms
        self.y = joined["y_true"].to_numpy().astype(np.int64)
        self.subject = joined["subject_id"].to_numpy()
        self.n_items = int(len(joined))
        self.n_groups = int(joined["subject_id"].nunique())
        self.ranks, self.buckets = {}, {}
        for arm in arms:
            r, b = rank_scores(joined[arm].to_numpy())
            self.ranks[arm], self.buckets[arm] = r, b

    def auc(self, arm: str, rows: np.ndarray) -> float:
        if rows.size == 0:
            return float("nan")
        y = self.y[rows]
        if y.min() == y.max():
            return float("nan")
        return auc_from_ranks(y, self.ranks[arm][rows], self.buckets[arm])


def build_universe(cells: dict[str, Cell]):
    """One shared patient ordering across every cell, and an index per cell."""
    universe = np.unique(np.concatenate([c.subject for c in cells.values()]))
    lookup = {sid: i for i, sid in enumerate(universe)}
    indices = {
        name: ClusterIndex.from_codes(
            np.array([lookup[s] for s in cell.subject], dtype=np.int64), len(universe)
        )
        for name, cell in cells.items()
    }
    return indices, len(universe)


def recovery_ratio(numerator: float, denominator: float) -> float:
    """(AUROC_arm - 0.5) / (AUROC_R6 - 0.5), undefined at a zero denominator.

    `horizon_ladder.yaml` requires an undefined ratio to be reported as undefined
    rather than clipped, imputed or dropped, so this returns NaN and the caller's
    percentile drops that replicate.
    """
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    below = denominator - 0.5
    if abs(below) < 1e-12:
        return float("nan")
    return (numerator - 0.5) / below


def holm(pvalues: dict, m: int) -> dict:
    """Holm at a DECLARED family size, which may exceed the tests in hand.

    The declared family size: `acq_context_audit` declares 15 tests and three of them need R8, which
    arrives in this stage. Correcting at m = 12 would be anti-conservative with respect to
    the family fixed on 2026-08-23 and could let a this stage conclusion evaporate in this stage
    with no new evidence about it.
    """
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


# --------------------------------------------------------------------------
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
    print("bootstrap self-test passed against scikit-learn and scipy")

    base = load_base_config()
    seeds = list(base["training"]["seeds"])
    derived = derived_root(args.data_root)
    index = ledger_index()
    RESULTS.mkdir(parents=True, exist_ok=True)

    audit = yaml.safe_load((COMPARISONS / "acq_context_audit.yaml").read_text("utf-8"))
    ladder = yaml.safe_load((COMPARISONS / "horizon_ladder.yaml").read_text("utf-8"))
    audit_labels = [str(x) for x in audit["labels"]]
    ladder_primary = [(str(d["label"]), int(d["horizon_days"]))
                      for d in ladder["primary_labels"]]
    ladder_generating = [(str(d["label"]), int(d["horizon_days"]))
                         for d in ladder["generating_labels"]]
    all_labels = sorted({*audit_labels, *[l for l, _ in ladder_primary],
                         *[l for l, _ in ladder_generating]})

    # ---- 1. pooled scores, per-arm intervals, per-cell noise floor ---------
    pooled: dict[tuple[str, str], tuple] = {}
    arm_rows, noise_rows = [], []
    for arm, label in itertools.product(P3_ARMS, all_labels):
        entry = pooled_arm(arm, label, seeds, index, derived)
        if entry is None:
            continue
        pooled[(arm, label)] = entry
        frame, per_seed = entry
        point, lo, hi = clustered_auc_ci(
            frame["y_true"].to_numpy(), frame["y_score"].to_numpy(),
            frame["subject_id"].to_numpy(), n_boot=args.n_boot, seed=args.seed,
        )
        arm_rows.append({
            "arm": arm, "label": label, "unit": "record",
            "auroc": point, "ci_lo": lo, "ci_hi": hi,
            "seed_mean": float(per_seed.mean()),
            "seed_min": float(per_seed.min()), "seed_max": float(per_seed.max()),
            "seed_spread": float(per_seed.max() - per_seed.min()),
            "n_seeds": int(len(per_seed)),
            "n_items": int(len(frame)),
            "n_groups": int(frame["subject_id"].nunique()),
            "n_positives": int(frame["y_true"].sum()),
            "prevalence": float(frame["y_true"].mean()),
        })
        noise_rows.append({
            "arm": arm, "label": label, "unit": "record",
            "seed_spread": float(per_seed.max() - per_seed.min()),
            "seed_sd": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
            "n_seeds": int(len(per_seed)),
            "source": "per-seed record-level AUROC of this exact (arm, label)",
        })
        print(f"  {arm:<22} {label:<32} {point:.4f} "
              f"[{lo:.4f}, {hi:.4f}]  spread {per_seed.max() - per_seed.min():.4f}")

    arms_table = pd.DataFrame(arm_rows)
    noise_table = pd.DataFrame(noise_rows)
    arms_table.to_csv(RESULTS / "p3_arms.csv", index=False)
    noise_table.to_csv(RESULTS / "p3_noise.csv", index=False)

    floor = {(r["arm"], r["label"]): r["seed_spread"] for _, r in noise_table.iterrows()}

    # ---- 2. the acq_context_audit family: paired contrasts ----------------
    declared = [tuple(c) for c in audit["family"]["contrasts"]]
    family_size = len(declared) * len(audit_labels)
    contrast_rows, pending = [], []
    for (arm_a, arm_b), label in itertools.product(declared, audit_labels):
        # the file writes [a, b] meaning "a against b"; the reported difference
        # is a minus b, matching this stage's `R3_acqctx_pre - R2_demo`
        joined = join_arms(pooled, [arm_b, arm_a], label)
        if joined is None:
            pending.append(f"{arm_a}-{arm_b}@{label}")
            continue
        result = paired_clustered_auc_diff(
            joined["y_true"].to_numpy(), joined[arm_b].to_numpy(),
            joined[arm_a].to_numpy(), joined["subject_id"].to_numpy(),
            n_boot=args.n_boot, seed=args.seed,
        )
        involves_waveform = arm_a in WAVEFORM_ARMS or arm_b in WAVEFORM_ARMS
        applicable = max(floor.get((arm_a, label), 0.0), floor.get((arm_b, label), 0.0))
        result.update(
            contrast=f"{arm_a} - {arm_b}", arm_a=arm_a, arm_b=arm_b, label=label,
            family="acq_context_audit", unit="record",
            n_items=int(len(joined)),
            noise_floor_used=applicable,
            noise_floor_source=(
                f"max per-seed record-level spread of {arm_a} and {arm_b} at {label}"
            ),
            involves_waveform_arm=bool(involves_waveform),
            clears_noise_floor=bool(abs(result["difference"]) > applicable),
        )
        contrast_rows.append(result)

    contrasts = pd.DataFrame(contrast_rows)
    if not contrasts.empty:
        keys = {f"{r['contrast']}@{r['label']}": r["p_value"]
                for _, r in contrasts.iterrows()}
        adjusted = holm(keys, family_size)
        contrasts["p_holm"] = [adjusted[f"{r['contrast']}@{r['label']}"]
                               for _, r in contrasts.iterrows()]
        contrasts["holm_family_size"] = family_size
        contrasts["holm_tests_present"] = len(contrasts)
        contrasts["holm_tests_pending"] = family_size - len(contrasts)
    contrasts.to_csv(RESULTS / "p3_contrasts.csv", index=False)

    # ---- 3. the recovery ratio, interval on the ratio itself --------------
    ratio_rows = []
    for arm, label in itertools.product(
        ("R3_acqctx_pre", "R4_demo_acq", "R5_tabular", "R7_waveform_demo_acq"),
        audit_labels,
    ):
        joined = join_arms(pooled, [arm, REFERENCE_ARM], label)
        if joined is None:
            continue
        cell = Cell(joined, [arm, REFERENCE_ARM])
        indices, n_groups = build_universe({"c": cell})

        def statistic(gathered, cell=cell, arm=arm):
            rows = gathered["c"]
            return recovery_ratio(cell.auc(arm, rows), cell.auc(REFERENCE_ARM, rows))

        result = clustered_statistic_ci(indices, n_groups, statistic,
                                        n_boot=args.n_boot, seed=args.seed)
        result.update(
            arm=arm, reference=REFERENCE_ARM, label=label, unit="record",
            quantity="recovery_ratio", n_items=cell.n_items,
            numerator_auroc=cell.auc(arm, np.arange(cell.n_items)),
            denominator_auroc=cell.auc(REFERENCE_ARM, np.arange(cell.n_items)),
            denominator_source="per-label record-level R6_waveform",
            target=float(audit["success_threshold"]),
            is_primary=bool(arm == audit["primary_arm"]
                            and label == audit["primary_label"]),
        )
        result["meets_target"] = bool(result["estimate"] >= result["target"])
        ratio_rows.append(result)
        print(f"  recovery {arm:<22} {label:<32} {result['estimate']:.4f} "
              f"[{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]")
    # ---- 3b. the same ratio under every declared denominator --------------
    # The prose claimed a direction for the alternative denominators and got one
    # of them backwards, and could do so because a comparative claim with no
    # decimal in it resolves to no cell. These are the cells.
    #
    # Only the per-label denominator is paired. A published macro is a constant
    # from another paper, and our own 15-target macro is a different estimand
    # from the per-label AUROC, so for those the resampling carries the
    # numerator alone and the row says so.
    published = yaml.safe_load(
        (REPO_ROOT / "configs" / "published_reference.yaml").read_text(encoding="utf-8"))
    macro = pd.read_csv(RESULTS / "p2_macro.csv")
    our_macro = float(
        macro[(macro["arm"] == REFERENCE_ARM) & (macro["unit"] == "record")]
        ["macro"].mean())

    alternatives = [
        ("published macro, version of record",
         float(published["version_of_record"]["macro_deterioration"]
               ["ecg_waveforms_only"]["point"]),
         "configs/published_reference.yaml version_of_record"),
        ("published macro, preprint",
         float(published["model_auroc_macro_deterioration"]
               ["ecg_waveforms_only"]["point"]),
         "configs/published_reference.yaml model_auroc_macro_deterioration"),
        ("our own 15-target record-level macro", our_macro,
         "results/p2_macro.csv"),
    ]

    primary_arm = audit["primary_arm"]
    primary_label = audit["primary_label"]
    joined = join_arms(pooled, [primary_arm, REFERENCE_ARM], primary_label)
    if joined is not None:
        cell = Cell(joined, [primary_arm, REFERENCE_ARM])
        indices, n_groups = build_universe({"c": cell})
        for name, denominator, origin in alternatives:
            def statistic(gathered, cell=cell, denominator=denominator):
                rows = gathered["c"]
                return recovery_ratio(cell.auc(primary_arm, rows), denominator)

            result = clustered_statistic_ci(indices, n_groups, statistic,
                                            n_boot=args.n_boot, seed=args.seed)
            result.update(
                arm=primary_arm, reference=REFERENCE_ARM, label=primary_label,
                unit="record", quantity="recovery_ratio_alternative_denominator",
                n_items=cell.n_items,
                numerator_auroc=cell.auc(primary_arm, np.arange(cell.n_items)),
                denominator_auroc=denominator,
                denominator_source=name,
                denominator_origin=origin,
                denominator_resampled=False,
                target=float(audit["success_threshold"]),
                is_primary=False,
            )
            result["meets_target"] = bool(result["estimate"] >= result["target"])
            ratio_rows.append(result)
            print(f"  recovery {primary_arm:<22} {name:<38} "
                  f"{result['estimate']:.4f} "
                  f"[{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]")

    frame = pd.DataFrame(ratio_rows)
    if "denominator_resampled" in frame:
        frame["denominator_resampled"] = frame["denominator_resampled"].fillna(True)
    else:
        frame["denominator_resampled"] = True
    frame.to_csv(RESULTS / "p3_recovery.csv", index=False)

    # ---- 4. the horizon ladder, a separate family with its own Holm -------
    ladder_rows, trend_rows = [], []
    for label, days in ladder_primary + ladder_generating:
        joined = join_arms(pooled, ["R2_demo", "R3_acqctx_pre"], label)
        if joined is None:
            continue
        result = paired_clustered_auc_diff(
            joined["y_true"].to_numpy(), joined["R2_demo"].to_numpy(),
            joined["R3_acqctx_pre"].to_numpy(), joined["subject_id"].to_numpy(),
            n_boot=args.n_boot, seed=args.seed,
        )
        result.update(
            contrast="R3_acqctx_pre - R2_demo", label=label, horizon_days=days,
            family="horizon_ladder", unit="record", n_items=int(len(joined)),
            role="primary" if (label, days) in ladder_primary else "generating",
            counts_as_confirmation=bool((label, days) in ladder_primary),
        )
        ladder_rows.append(result)
    ladder_table = pd.DataFrame(ladder_rows)

    def trend_over(pairs, arms, quantity):
        """Spearman between horizon and a per-label quantity, resampled jointly.

        Patients are drawn once and all four labels are recomputed on that same
        draw, so the correlation carries the dependence between horizons instead
        of treating four overlapping patient sets as independent.
        """
        cells, days = {}, []
        for label, horizon in pairs:
            joined = join_arms(pooled, arms, label)
            if joined is None:
                return None
            cells[label] = Cell(joined, arms)
            days.append(horizon)
        indices, n_groups = build_universe(cells)
        order = [label for label, _ in pairs]
        horizons = np.asarray(days, dtype=float)

        def statistic(gathered):
            values = []
            for label in order:
                cell, rows = cells[label], gathered[label]
                if quantity == "delta":
                    values.append(cell.auc(arms[1], rows) - cell.auc(arms[0], rows))
                else:
                    values.append(recovery_ratio(cell.auc(arms[0], rows),
                                                 cell.auc(arms[1], rows)))
            values = np.asarray(values, dtype=float)
            if not np.isfinite(values).all():
                return float("nan")
            return spearman(horizons, values)

        result = clustered_statistic_ci(indices, n_groups, statistic,
                                        n_boot=args.n_boot, seed=args.seed)
        result.update(quantity=quantity, labels=",".join(order),
                      horizons=",".join(str(int(d)) for d in days),
                      n_labels=len(order), family="horizon_ladder", unit="record")
        result["supported"] = bool(
            np.isfinite(result["ci_hi"]) and result["estimate"] < 0
            and result["ci_hi"] < 0
        )
        return result

    acq_trend = trend_over(ladder_primary, ["R2_demo", "R3_acqctx_pre"], "delta")
    if acq_trend:
        acq_trend["name"] = "acq_advantage_trend"
        acq_trend["role"] = "primary"
        trend_rows.append(acq_trend)
    rr_trend = trend_over(ladder_primary, ["R3_acqctx_pre", REFERENCE_ARM], "ratio")
    if rr_trend:
        rr_trend["name"] = "recovery_ratio_trend"
        rr_trend["role"] = "secondary"
        trend_rows.append(rr_trend)
    all_six = trend_over(sorted(ladder_primary + ladder_generating, key=lambda t: t[1]),
                         ["R2_demo", "R3_acqctx_pre"], "delta")
    if all_six:
        all_six["name"] = "acq_advantage_trend_all_six"
        all_six["role"] = "tertiary_contains_generating_labels"
        all_six["supported"] = False   # reported for completeness, never confirmation
        trend_rows.append(all_six)

    # Holm over this family only: its four primary paired differences and its two
    # trend statistics. Six tests, its own correction, never pooled with the
    # audit family.
    ladder_keys: dict[str, float] = {}
    if not ladder_table.empty:
        for _, row in ladder_table[ladder_table["counts_as_confirmation"]].iterrows():
            ladder_keys[f"paired@{row['label']}"] = row["p_value"]
    for row in trend_rows:
        if row["role"] in ("primary", "secondary"):
            ladder_keys[f"trend@{row['name']}"] = row["p_value"]
    ladder_family_size = len(ladder["family"]["members"])
    ladder_adjusted = holm(ladder_keys, ladder_family_size)
    if not ladder_table.empty:
        ladder_table["p_holm"] = [
            ladder_adjusted.get(f"paired@{r['label']}", float("nan"))
            for _, r in ladder_table.iterrows()
        ]
        ladder_table["holm_family_size"] = ladder_family_size
    for row in trend_rows:
        row["p_holm"] = ladder_adjusted.get(f"trend@{row['name']}", float("nan"))
        row["holm_family_size"] = ladder_family_size
    ladder_table.to_csv(RESULTS / "p3_ladder.csv", index=False)
    pd.DataFrame(trend_rows).to_csv(RESULTS / "p3_trends.csv", index=False)

    print(f"\nwrote p3_arms.csv ({len(arms_table)} rows), p3_contrasts.csv "
          f"({len(contrasts)}), p3_recovery.csv ({len(ratio_rows)}), "
          f"p3_ladder.csv ({len(ladder_table)}), p3_trends.csv ({len(trend_rows)}), "
          f"p3_noise.csv ({len(noise_table)})")
    if pending:
        print(f"pending family members, awaiting this stage: {sorted(pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Datasets over the 100 Hz memmap, cropped the way the published config crops.

Two modes, and the difference between them is the whole of the crop-level and record-level separation:

  train  one random 250-sample crop per record per epoch. `chunk_length_train`
         is 0 in the published config, which means the record is one chunk and a
         crop of `output_size` is drawn from it, and `sample_items_per_record`
         is 1, so one item per record per epoch.

  eval   four fixed non-overlapping crops per record, at offsets 0, 250, 500 and
         750, because `chunk_length_valtest` and `stride_valtest` both resolve to
         250. Each crop is a separate row. They are NOT averaged by the published
         code: `PoolingHeadConfig.multi_prediction` is False, so
         `is_multi_prediction()` is False and the aggregation branch never runs.
         We keep the crop index on every row so a record-level score can be built
         afterwards for cross-arm contrasts.

Targets are the 15 deterioration labels in the contract's order. MDS-ED marks a
target undefined for a row with -999; that becomes mask 0 and target 0, and the
loss drops it. Treating it as a negative would mislabel 41,125 rows on
`deterioration_severe_hypoxemia` alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from paths import derived_root  # noqa: E402

MEMMAP_DIRNAME = "waveforms_100hz"
SENTINEL = -999

#: The 15 deterioration targets, in the order the contract lists them. Fixed
#: here so the head's output columns mean the same thing in every run.
TARGETS = [
    "deterioration_severe_hypoxemia",
    "deterioration_ecmo",
    "deterioration_vasopressors",
    "deterioration_inotropes",
    "deterioration_mechanical_ventilation",
    "deterioration_cardiac_arrest",
    "deterioration_icu_24h",
    "deterioration_icu_stay",
    "deterioration_mortality_1d",
    "deterioration_mortality_7d",
    "deterioration_mortality_28d",
    "deterioration_mortality_90d",
    "deterioration_mortality_180d",
    "deterioration_mortality_365d",
    "deterioration_mortality_stay",
]


def load_memmap_meta(data_root: str | None = None) -> dict:
    path = derived_root(data_root) / MEMMAP_DIRNAME / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/preprocess_waveforms.py` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_static_block(
    derived: Path, blocks: list[str], contract, split: pd.Series
) -> tuple[np.ndarray, list[str], dict]:
    """Assemble and standardise the static feature block for `R7`.

    Standardisation uses train-fold statistics only, so no validation or test
    information reaches the model through the scaler. Missing values become 0
    after standardisation, which is the train-fold mean, and every column with
    any missingness gains an explicit indicator so "missing" is a signal the
    model can use rather than a silent imputation. `triage_acuity` is missing on
    2.33 percent of rows by the arrival lower bound and must stay visible.
    """
    demo = pd.read_parquet(derived / "features_demo.parquet")
    acq = pd.read_parquet(derived / "features_acq.parquet")
    frame = demo.merge(
        acq[[c for c in acq.columns if not c.startswith("available_at__")]],
        on="study_id", how="left",
    )
    frame = frame[[c for c in frame.columns if not c.startswith("available_at__")]]

    names: list[str] = []
    for block in blocks:
        names += contract.block_names(block)
    names = [n for n in names if n in frame.columns]

    stats: dict = {}
    value_cols: list[np.ndarray] = []
    columns: list[str] = []
    indicator_cols: list[np.ndarray] = []
    indicator_names: list[str] = []

    is_train = split.to_numpy() == "train"
    for name in names:
        col = frame[name]
        if str(col.dtype) in {"object", "category", "string", "bool"}:
            codes = pd.Categorical(col).codes.astype("float64")
            col = pd.Series(np.where(codes < 0, np.nan, codes), index=col.index)
        col = pd.to_numeric(col, errors="coerce").astype("float64")
        train_vals = col[is_train]
        mu = float(train_vals.mean()) if train_vals.notna().any() else 0.0
        sd = float(train_vals.std(ddof=0))
        sd = sd if sd > 0 else 1.0
        missing = col.isna().to_numpy()
        value_cols.append(((col.fillna(mu) - mu) / sd).to_numpy(dtype=np.float32))
        columns.append(name)
        stats[name] = {"mean": mu, "std": sd, "missing_rate": float(missing.mean())}
        if missing.any():
            indicator_cols.append(missing.astype(np.float32))
            indicator_names.append(f"{name}__missing")

    if not value_cols:
        return np.zeros((len(frame), 0), dtype=np.float32), [], stats
    values = np.stack(value_cols + indicator_cols, axis=1).astype(np.float32)
    return values, columns + indicator_names, stats


class WaveformDataset(Dataset):
    """One split of the cohort, over the shared 100 Hz memmap."""

    def __init__(
        self,
        split: str,
        *,
        data_root: str | None = None,
        crop: int = 250,
        train: bool = True,
        seed: int = 0,
        static: np.ndarray | None = None,
        targets: list[str] | None = None,
    ) -> None:
        derived = derived_root(data_root)
        self.meta = load_memmap_meta(data_root)
        self.mm_path = derived / MEMMAP_DIRNAME / "signals.f32"
        self.shape = tuple(self.meta["shape"])
        self.crop = int(crop)
        self.train = bool(train)
        self.targets = list(targets or TARGETS)
        self.n_samples = self.shape[1]
        if self.n_samples % self.crop:
            raise ValueError(f"crop {self.crop} does not divide record length {self.n_samples}")
        self.crops_per_record = self.n_samples // self.crop

        cohort = pd.read_parquet(
            derived / "cohort_mdsed.parquet",
            columns=["study_id", "subject_id", "split", *self.targets],
        ).reset_index(drop=True)
        done = np.memmap(derived / MEMMAP_DIRNAME / "done.u8", dtype=np.uint8, mode="r")
        # Completeness is recorded, not assumed. A memmap still being written
        # grows under a running experiment: two "identical" runs then see
        # different record sets and disagree for a reason that looks like
        # nondeterminism and is not. Callers that need a fixed dataset check
        # `is_complete` and refuse.
        self.n_preprocessed = int(np.asarray(done[: len(cohort)]).sum())
        self.n_cohort = int(len(cohort))
        self.is_complete = self.n_preprocessed == self.n_cohort
        keep = (cohort["split"].to_numpy() == split) & (np.asarray(done[: len(cohort)]) == 1)
        self.rows = np.flatnonzero(keep).astype(np.int64)
        self.n_missing_signal = int(((cohort["split"].to_numpy() == split) & ~keep).sum())

        raw = cohort.loc[self.rows, self.targets].to_numpy(dtype=np.float32)
        self.mask = (raw != SENTINEL).astype(np.float32)
        self.y = np.where(raw == SENTINEL, 0.0, raw).astype(np.float32)
        self.subject_id = cohort.loc[self.rows, "subject_id"].to_numpy()
        self.study_id = cohort.loc[self.rows, "study_id"].to_numpy()
        self.static = None if static is None else static[self.rows].astype(np.float32)

        self.seed = int(seed)
        self.epoch = 0
        self._mm: np.memmap | None = None

    def set_epoch(self, epoch: int) -> None:
        """Advance the crop schedule. Must be called before each training epoch."""
        self.epoch = int(epoch)

    def _crop_offset(self, row: int) -> int:
        """Deterministic crop start for (seed, epoch, row).

        Deliberately not a stateful RNG. A stateful generator would break twice
        over: on resume, a fresh dataset would restart its stream and replay
        different crops than the uninterrupted run, and with DataLoader workers
        every worker would inherit the same generator state and emit correlated
        crops. A pure function of the three things that should determine the
        crop has neither problem, and it makes an epoch exactly reproducible
        from its index alone. SplitMix64 finaliser.
        """
        # Plain Python ints with an explicit 64-bit mask. numpy uint64 would do
        # the same arithmetic but emits an overflow RuntimeWarning per call for
        # the wraparound this algorithm depends on, which would bury real
        # warnings in the training log.
        m = 0xFFFFFFFFFFFFFFFF
        x = (row * 0x9E3779B97F4A7C15) & m
        x ^= (self.epoch * 0xBF58476D1CE4E5B9) & m
        x ^= (self.seed * 0x94D049BB133111EB) & m
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & m
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & m
        x ^= x >> 31
        return x % (self.n_samples - self.crop + 1)

    def __len__(self) -> int:
        return len(self.rows) * (1 if self.train else self.crops_per_record)

    def _signals(self) -> np.memmap:
        # Opened lazily so DataLoader workers each get their own handle rather
        # than inheriting one across a fork/spawn boundary.
        if self._mm is None:
            self._mm = np.memmap(self.mm_path, dtype=np.float32, mode="r", shape=self.shape)
        return self._mm

    def __getitem__(self, i: int):
        if self.train:
            local = i
            start = self._crop_offset(int(self.rows[local]))
            crop_idx = -1
        else:
            local, crop_idx = divmod(i, self.crops_per_record)
            start = crop_idx * self.crop
        row = int(self.rows[local])
        seq = np.asarray(self._signals()[row, start : start + self.crop, :], dtype=np.float32)
        seq = np.nan_to_num(seq, nan=0.0)  # matches template_model's input NaN guard
        item = {
            "seq": torch.from_numpy(seq),
            "y": torch.from_numpy(self.y[local]),
            "mask": torch.from_numpy(self.mask[local]),
            "local": local,
            "crop": crop_idx,
        }
        if self.static is not None:
            item["static"] = torch.from_numpy(self.static[local])
        return item

    def describe(self) -> dict:
        return {
            "n_records": int(len(self.rows)),
            "n_preprocessed": self.n_preprocessed,
            "n_cohort": self.n_cohort,
            "memmap_complete": self.is_complete,
            "n_items": int(len(self)),
            "crops_per_record": int(1 if self.train else self.crops_per_record),
            "n_patients": int(pd.unique(self.subject_id).size),
            "n_missing_signal": self.n_missing_signal,
            "targets": self.targets,
            "defined_per_target": {
                t: int(self.mask[:, k].sum()) for k, t in enumerate(self.targets)
            },
            "positives_per_target": {
                t: int((self.y[:, k] * self.mask[:, k]).sum()) for k, t in enumerate(self.targets)
            },
        }

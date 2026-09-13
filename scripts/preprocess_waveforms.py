#!/usr/bin/env python
"""Turn the fetched 500 Hz WFDB records into the 100 Hz array the model trains on.

This reproduces `ecg_utils.prepare_mimicecg` from the MDS-ED release exactly,
because the corresponding check measures a reproduction gap and a preprocessing difference would
be indistinguishable from an architecture difference in that number. The order of
operations is not incidental and is not ours to choose:

  1. ``wfdb.rdsamp``                          -> (5000, 12) float64, mV, 500 Hz
  2. NaN interpolation *and* clip to +/-3 mV, **at 500 Hz, before resampling**
  3. ``resampy.resample`` 500 -> 100 Hz, per channel, cast to float32
  4. no re-clip afterwards

Step 2 preceding step 3 matters twice over. Interpolating after resampling would
be too late, because resampy's sinc kernel smears a single NaN across the whole
signal. Clipping before rather than after leaves resampling ringing free to
overshoot +/-3 slightly, which is what the published pipeline does and therefore
what we do.

Leads are mapped by name, never by position. The staged headers carry
``I, II, III, aVR, aVF, aVL, V1..V6``; MDS-ED's ``channel_stoi_default`` stores
``I, II, V1..V6, III, aVR, aVL, aVF``. A positional mapping trains cleanly,
converges, and reproduces nothing, so the mapping is by lowercased ``sig_name``
and the corresponding check asserts a deliberately mis-ordered map fails.

Output, under ``$ECG_DATA_ROOT/derived/waveforms_100hz/``:

  signals.f32   memmap, (n_records, 1000, 12) float32, in cohort row order
  done.u8       memmap, (n_records,) uint8 progress flags, makes this resumable
  index.parquet one row per record: study_id, subject_id, split, digest, stats
  meta.json     shape, dtype, the spec, and the hashes that pin all of it

    python scripts/preprocess_waveforms.py                 # all records
    python scripts/preprocess_waveforms.py --limit 2000    # a slice, for timing
    python scripts/preprocess_waveforms.py --verify-only   # re-check, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import load_base_config  # noqa: E402
from paths import assert_outside_repo, data_root, derived_root  # noqa: E402

#: MDS-ED `channel_stoi_default`, restricted to the 12 leads it keeps. Lowercase
#: keys, because `resample_data` lowercases the header's sig_name first.
CHANNEL_STOI = {
    "i": 0, "ii": 1, "v1": 2, "v2": 3, "v3": 4, "v4": 5,
    "v5": 6, "v6": 7, "iii": 8, "avr": 9, "avl": 10, "avf": 11,
}
N_CHANNELS = 12
SOURCE_FS = 500
TARGET_FS = 100
N_SAMPLES = 1000          # int(5000 * 100/500)
CLIP_AMP = 3.0
OUT_DIRNAME = "waveforms_100hz"

#: The staged tree drops MDS-ED's leading archive-name component.
_ARCHIVE_PREFIX = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"

_STATE: dict = {}


def spec() -> dict:
    """The preprocessing specification, hashed to pin every arm to one recipe."""
    return {
        "reader": "wfdb.rdsamp",
        "source_fs_hz": SOURCE_FS,
        "target_fs_hz": TARGET_FS,
        "n_samples": N_SAMPLES,
        "n_channels": N_CHANNELS,
        "resampler": "resampy.resample",
        "clip_amplitude_mv": CLIP_AMP,
        "nan_handling": "pandas.DataFrame.interpolate, per channel",
        "order": [
            "rdsamp",
            "nan_interpolate_and_clip_at_source_fs",
            "resample_to_target_fs",
            "cast_float32",
        ],
        "reclip_after_resample": False,
        "channel_stoi": CHANNEL_STOI,
        "dtype": "float32",
        "layout": "(n_records, n_samples, n_channels)",
        "source": "ecg_utils.prepare_mimicecg, MDS-ED release; see the pinned architecture",
    }


def preprocess_hash() -> str:
    """Content hash of the recipe. Two arms with different values cannot be compared."""
    canonical = json.dumps(spec(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def record_path(ecg_root: Path, general_file_name: str) -> Path:
    rel = general_file_name
    if rel.startswith(_ARCHIVE_PREFIX):
        rel = rel[len(_ARCHIVE_PREFIX):]
    elif "/" in rel and rel.split("/", 1)[0].startswith("mimic-iv-ecg"):
        rel = rel.split("/", 1)[1]
    return ecg_root / rel


def fix_nans_and_clip(signal: np.ndarray, clip_amp: float = CLIP_AMP) -> None:
    """MDS-ED `fix_nans_and_clip`, in place, at the source sampling rate.

    `interpolate()` fills interior gaps only; a NaN run at the very start has
    nothing to interpolate from and survives. That is the published behaviour,
    so it is reproduced rather than improved, and the survivors are counted in
    `index.parquet` so the decision to leave them is visible.
    """
    for i in range(signal.shape[1]):
        tmp = pd.DataFrame(signal[:, i]).interpolate().values.ravel()
        signal[:, i] = np.clip(tmp, a_max=clip_amp, a_min=-clip_amp) if clip_amp > 0 else tmp


def resample_data(sigbufs: np.ndarray, channel_labels, lead_map=None) -> np.ndarray:
    """MDS-ED `resample_data`: map by lowercased name, resample per channel."""
    import resampy

    stoi = CHANNEL_STOI if lead_map is None else lead_map
    labels = [str(c).lower() for c in channel_labels]
    data = np.zeros((N_SAMPLES, N_CHANNELS), dtype=np.float32)
    mapped = 0
    for i, cl in enumerate(labels):
        if cl in stoi and stoi[cl] < N_CHANNELS:
            data[:, stoi[cl]] = resampy.resample(
                sigbufs[:, i], SOURCE_FS, TARGET_FS
            ).astype(np.float32)
            mapped += 1
    return data, mapped


def process_one(path: Path, lead_map=None) -> tuple[np.ndarray, dict]:
    """Read and preprocess a single record. Returns the array and its statistics."""
    import wfdb

    sigbufs, header = wfdb.rdsamp(str(path))
    if int(header["fs"]) != SOURCE_FS:
        raise ValueError(f"{path}: fs is {header['fs']}, expected {SOURCE_FS}")
    sigbufs = np.asarray(sigbufs, dtype=np.float64)
    nan_before = int(np.isnan(sigbufs).sum())

    if nan_before > 0:
        fix_nans_and_clip(sigbufs, clip_amp=CLIP_AMP)
    elif CLIP_AMP > 0:
        sigbufs = np.clip(sigbufs, a_max=CLIP_AMP, a_min=-CLIP_AMP)

    data, mapped = resample_data(sigbufs, header["sig_name"], lead_map=lead_map)
    stats = {
        "nan_before": nan_before,
        "nan_after": int(np.isnan(data).sum()),
        "leads_mapped": mapped,
        "amp_min": float(np.nanmin(data)) if data.size else 0.0,
        "amp_max": float(np.nanmax(data)) if data.size else 0.0,
        "digest": hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()[:16],
    }
    return data, stats


def _init_worker(sig_path: str, done_path: str, n: int, ecg_root: str) -> None:
    _STATE["signals"] = np.memmap(
        sig_path, dtype=np.float32, mode="r+", shape=(n, N_SAMPLES, N_CHANNELS)
    )
    _STATE["done"] = np.memmap(done_path, dtype=np.uint8, mode="r+", shape=(n,))
    _STATE["ecg_root"] = Path(ecg_root)


def _work(item: tuple[int, str]) -> tuple[int, dict | None, str]:
    idx, general_file_name = item
    try:
        path = record_path(_STATE["ecg_root"], general_file_name)
        data, stats = process_one(path)
        _STATE["signals"][idx] = data
        _STATE["done"][idx] = 1
        return idx, stats, ""
    except Exception as exc:  # a bad record must not kill the whole pass
        return idx, None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--limit", type=int, default=0, help="process only the first N records")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--reindex", action="store_true",
                        help="rebuild index.parquet and meta.json from the memmap, "
                             "processing nothing")
    parser.add_argument("--force", action="store_true", help="ignore done flags and redo")
    args = parser.parse_args()

    base = load_base_config()
    derived = derived_root(args.data_root)
    ecg_root = data_root(args.data_root) / "MIMIC-IV-ECG"
    out_dir = assert_outside_repo(derived / OUT_DIRNAME, "waveform memmap")
    out_dir.mkdir(parents=True, exist_ok=True)

    declared = list(base["features"]["waveform"]["lead_order"])
    ordered = [lead for lead, _ in sorted(CHANNEL_STOI.items(), key=lambda kv: kv[1])]
    if [d.lower() for d in declared] != ordered:
        raise SystemExit(
            f"lead_order in configs/base.yaml is {declared}, but CHANNEL_STOI gives "
            f"{ordered}. These must agree;"
        )

    cohort = pd.read_parquet(
        derived / "cohort_mdsed.parquet",
        columns=["general_file_name", "study_id", "subject_id", "split"],
    ).reset_index(drop=True)
    n_total = len(cohort)
    n = min(args.limit, n_total) if args.limit else n_total

    sig_path = out_dir / "signals.f32"
    done_path = out_dir / "done.u8"
    expected_bytes = n_total * N_SAMPLES * N_CHANNELS * 4
    if not sig_path.exists() or sig_path.stat().st_size != expected_bytes:
        if args.verify_only:
            raise SystemExit(f"{sig_path} missing or wrong size; nothing to verify")
        print(f"allocating {expected_bytes / 2**30:.2f} GiB at {sig_path}")
        np.memmap(sig_path, dtype=np.float32, mode="w+",
                  shape=(n_total, N_SAMPLES, N_CHANNELS)).flush()
    if not done_path.exists() or done_path.stat().st_size != n_total:
        np.memmap(done_path, dtype=np.uint8, mode="w+", shape=(n_total,)).flush()

    done = np.memmap(done_path, dtype=np.uint8, mode="r+", shape=(n_total,))
    if args.force:
        done[:n] = 0
    todo = [(i, cohort.at[i, "general_file_name"]) for i in range(n) if not done[i]]
    if args.reindex:
        todo = []
    print(f"{n:,} records in scope, {n - len(todo):,} already done, {len(todo):,} to do")

    if args.verify_only:
        remaining = int((done[:n] == 0).sum())
        print(f"VERIFY: {n - remaining:,}/{n:,} complete, {remaining:,} missing")
        return 0 if remaining == 0 else 1

    stats_path = out_dir / "stats.jsonl"
    failures: list[tuple[int, str]] = []
    started = time.time()
    written = 0
    if todo:
        with stats_path.open("a", encoding="utf-8") as sink:
            with mp.Pool(
                processes=args.workers,
                initializer=_init_worker,
                initargs=(str(sig_path), str(done_path), n_total, str(ecg_root)),
            ) as pool:
                for idx, stats, err in pool.imap_unordered(_work, todo, chunksize=32):
                    if err:
                        failures.append((idx, err))
                        continue
                    sink.write(json.dumps({"i": idx, **stats}) + "\n")
                    written += 1
                    if written % 2000 == 0:
                        rate = written / max(time.time() - started, 1e-9)
                        left = (len(todo) - written) / max(rate, 1e-9)
                        print(f"  {written:,}/{len(todo):,}  {rate:.1f} rec/s  "
                              f"eta {left / 60:.1f} min", flush=True)
                        sink.flush()
    elapsed = time.time() - started

    for idx, err in failures[:20]:
        print(f"  FAILED {idx}: {cohort.at[idx, 'general_file_name']}  {err}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more failures")

    # Rebuild the index. Statistics that describe the STORED array are recomputed
    # from the memmap rather than taken from the append-only log, because the log
    # is flushed periodically and loses its tail when a run is killed: after one
    # such kill, 220 perfectly good records had no log line, showed up as
    # "fewer than 12 leads mapped", and contributed an empty digest to the
    # content hash, which would then not have detected corruption in them. The
    # artifact is the memmap, so the memmap is what gets measured. Only
    # `nan_before` and `leads_mapped` describe the INPUT and can come from the
    # log; where the log lost them they are recorded as unrecorded, not as zero.
    rows: dict[int, dict] = {}
    if stats_path.is_file():
        for line in stats_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                rows[rec["i"]] = rec

    index = cohort.iloc[:n].copy()
    index["row"] = np.arange(n)
    index["ok"] = [bool(done[i]) for i in range(n)]
    index["nan_before"] = [rows.get(i, {}).get("nan_before", -1) for i in range(n)]
    index["leads_mapped"] = [rows.get(i, {}).get("leads_mapped", -1) for i in range(n)]

    signals = np.memmap(sig_path, dtype=np.float32, mode="r",
                        shape=(n_total, N_SAMPLES, N_CHANNELS))
    digests, nan_after, amp_min, amp_max, leads_nonzero = [], [], [], [], []
    for i in range(n):
        if not done[i]:
            digests.append("")
            nan_after.append(-1)
            amp_min.append(np.nan)
            amp_max.append(np.nan)
            leads_nonzero.append(-1)
            continue
        arr = np.ascontiguousarray(signals[i])
        digests.append(hashlib.sha256(arr.tobytes()).hexdigest()[:16])
        nan_after.append(int(np.isnan(arr).sum()))
        finite = arr[np.isfinite(arr)]
        amp_min.append(float(finite.min()) if finite.size else np.nan)
        amp_max.append(float(finite.max()) if finite.size else np.nan)
        leads_nonzero.append(int((np.abs(arr).max(axis=0) > 0).sum()))
    index["digest"] = digests
    index["nan_after"] = nan_after
    index["amp_min"] = amp_min
    index["amp_max"] = amp_max
    index["leads_nonzero"] = leads_nonzero
    index.to_parquet(out_dir / "index.parquet", index=False)

    complete = int(index["ok"].sum())
    combined = hashlib.sha256(
        "".join(index["digest"].tolist()).encode("utf-8")
    ).hexdigest() if complete == n else ""

    meta = {
        "n_records": int(n),
        "n_records_cohort": int(n_total),
        "n_complete": complete,
        "n_failed": len(failures),
        "shape": [int(n_total), N_SAMPLES, N_CHANNELS],
        "dtype": "float32",
        "bytes": expected_bytes,
        "spec": spec(),
        "preprocess_hash": preprocess_hash(),
        "content_hash": combined,
        "lead_order_declared": declared,
        "cohort_order": "cohort_mdsed.parquet row order, index 0..n-1",
        "elapsed_seconds": round(elapsed, 1),
        "records_per_second": round(written / elapsed, 2) if elapsed > 0 and written else None,
        "workers": args.workers,
        "nan_after_total": int(index.loc[index["ok"], "nan_after"].clip(lower=0).sum()),
        "records_with_residual_nan": int(
            ((index["nan_after"] > 0) & index["ok"]).sum()
        ),
        # Measured on the stored array: a channel that is identically zero
        # everywhere was never written, because no source lead mapped to it.
        "records_with_incomplete_leads": int(
            ((index["leads_nonzero"] != N_CHANNELS) & index["ok"]).sum()
        ),
        # Provenance about the INPUT, from the append-only log. A gap here means
        # the log lost its tail to a kill, not that the record is bad.
        "records_without_input_stats": int(
            ((index["leads_mapped"] < 0) & index["ok"]).sum()
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"\n{complete:,}/{n:,} complete, {len(failures)} failed, "
        f"{elapsed / 60:.1f} min, preprocess_hash={meta['preprocess_hash'][:12]}"
    )
    if meta["records_with_residual_nan"]:
        print(f"  residual NaN in {meta['records_with_residual_nan']:,} records "
              f"(leading-NaN runs interpolate() cannot fill; published behaviour)")
    if meta["records_with_incomplete_leads"]:
        print(f"  fewer than 12 non-zero leads in "
              f"{meta['records_with_incomplete_leads']:,} stored records")
    if meta["records_without_input_stats"]:
        print(f"  input-side stats unrecorded for "
              f"{meta['records_without_input_stats']:,} records (log tail lost to a "
              f"kill; the stored arrays are measured and fine)")
    return 0 if complete == n and not failures else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

#!/usr/bin/env python
"""Verify the fetched waveform subset against the manifest.

A fetch that reports failures is not the same as a fetch that is incomplete: a
failed request may have been retried into a later success, and a "successful"
run can still leave a truncated file. This checks the artifact on disk rather
than trusting the fetch log.

Three assertions per record:
  * both the ``.hea`` header and the ``.dat`` signal exist
  * neither is empty, and no ``.part`` temporary is left behind
  * the ``.dat`` size matches what its own header declares it should be

That last one is the point. A short ``.dat`` is the failure mode that survives a
size>0 check and then silently trains a model on truncated signals.

    python scripts/verify_waveforms.py
    python scripts/verify_waveforms.py --write-missing     # emit a retry list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import load_contract  # noqa: E402
from loaders import resolve_path  # noqa: E402
from paths import data_root, derived_root  # noqa: E402


def read_manifest(manifest_dir: Path) -> list[tuple[str, Path]]:
    config = manifest_dir / "curl_config.txt"
    if not config.is_file():
        raise FileNotFoundError(f"{config} not found; run make_waveform_manifest.py")
    urls: list[str] = []
    outs: list[Path] = []
    for line in config.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("url ="):
            urls.append(line.split("=", 1)[1].strip().strip('"'))
        elif line.startswith("output ="):
            outs.append(Path(line.split("=", 1)[1].strip().strip('"')))
    return list(zip(urls, outs))


def expected_dat_bytes(header: Path) -> int | None:
    """Bytes the.dat should hold, from the WFDB header's own declaration.

    Line 1 is ``<name> <n_signals> <fs> <n_samples> ...``; each signal line names
    a format such as ``16``. For a fixed-width format the file is
    n_signals * n_samples * bytes_per_sample.
    """
    try:
        lines = header.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if not lines:
        return None
    fields = lines[0].split()
    if len(fields) < 4:
        return None
    try:
        n_signals = int(fields[1])
        n_samples = int(fields[3])
    except ValueError:
        return None
    widths = {"8": 1, "80": 1, "16": 2, "61": 2, "160": 2, "24": 3, "32": 4}
    formats = set()
    for line in lines[1: 1 + n_signals]:
        parts = line.split()
        if len(parts) >= 2:
            formats.add(parts[1].split("x")[0].split(":")[0].split("+")[0])
    if len(formats) != 1:
        return None
    width = widths.get(formats.pop())
    if width is None:
        return None
    return n_signals * n_samples * width


def split_of_record(root: Path, contract) -> dict[str, str]:
    """Map record stem -> split, so completeness is reportable per fold."""
    path = resolve_path("mds_ed.cohort", root, contract)
    assignment = contract.raw["split"]["assignment"]
    fold_to_split = {
        int(f): name for name, folds in assignment.items() for f in folds
    }
    mapping: dict[str, str] = {}
    # Chunked: even with usecols the C parser tokenises all 1,936 columns per row,
    # and reading 129k of them at once exhausts memory on this machine.
    for chunk in pd.read_csv(
        path,
        usecols=["general_file_name", "general_strat_fold"],
        chunksize=20_000,
    ):
        for raw, fold in zip(chunk["general_file_name"], chunk["general_strat_fold"]):
            if pd.isna(raw):
                continue
            text = str(raw).replace("\\", "/")
            stem = "files/" + text.split("/files/", 1)[1] if "/files/" in text else text
            mapping[stem] = fold_to_split.get(int(fold), "train")
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--write-missing", action="store_true")
    parser.add_argument("--check-sizes", action="store_true", default=True)
    args = parser.parse_args()

    contract = load_contract()
    root = data_root(args.data_root)
    manifest_dir = derived_root(args.data_root) / "waveforms"
    targets = read_manifest(manifest_dir)
    splits = split_of_record(root, contract)

    missing: list[tuple[str, Path]] = []
    empty: list[Path] = []
    truncated: list[tuple[Path, int, int]] = []
    stray_parts: list[Path] = []
    by_split_missing: Counter = Counter()
    by_split_total: Counter = Counter()

    # Index the.hea for each record so.dat sizes can be checked against it.
    headers: dict[str, Path] = {}
    for _, destination in targets:
        if destination.suffix == ".hea":
            headers[destination.with_suffix("").as_posix()] = destination

    for url, destination in targets:
        stem = "files/" + destination.as_posix().split("/files/", 1)[1]
        record = stem.rsplit(".", 1)[0]
        split = splits.get(record, "unknown")
        by_split_total[split] += 1

        part = destination.with_suffix(destination.suffix + ".part")
        if part.exists():
            stray_parts.append(part)

        try:
            size = os.path.getsize(destination)
        except OSError:
            missing.append((url, destination))
            by_split_missing[split] += 1
            continue

        if size == 0:
            empty.append(destination)
            by_split_missing[split] += 1
            continue

        if args.check_sizes and destination.suffix == ".dat":
            header = headers.get(destination.with_suffix("").as_posix())
            if header is not None and header.exists():
                want = expected_dat_bytes(header)
                if want is not None and size != want:
                    truncated.append((destination, size, want))
                    by_split_missing[split] += 1

    total = len(targets)
    bad = len(missing) + len(empty) + len(truncated)
    print(f"waveform verification against the manifest")
    print(f"  expected files   {total:>9,}")
    print(f"  missing          {len(missing):>9,}")
    print(f"  empty            {len(empty):>9,}")
    print(f"  truncated.dat   {len(truncated):>9,}")
    print(f"  stray.part      {len(stray_parts):>9,}")
    print(f"  complete         {total - bad:>9,}  ({100.0 * (total - bad) / total:.3f}%)")
    print("\n  per split:")
    for split in ("test", "val", "train", "unknown"):
        if not by_split_total.get(split):
            continue
        have = by_split_total[split] - by_split_missing[split]
        print(f"    {split:<8} {have:>8,}/{by_split_total[split]:<8,} "
              f"({100.0 * have / by_split_total[split]:6.2f}%)")

    for path, size, want in truncated[:5]:
        print(f"\n  TRUNCATED {path.name}: {size:,} bytes, header declares {want:,}")

    if args.write_missing and (missing or empty or truncated):
        retry = manifest_dir / "retry_config.txt"
        lines = ["# Retry list from verify_waveforms.py", ""]
        seen: set[str] = set()
        for url, destination in missing:
            lines += [f'url = "{url}"', f'output = "{destination.as_posix()}"']
            seen.add(str(destination))
        for destination in empty + [t[0] for t in truncated]:
            if str(destination) in seen:
                continue
            stem = "files/" + destination.as_posix().split("/files/", 1)[1]
            url = f"https://physionet.org/files/mimic-iv-ecg/1.0/{stem}"
            lines += [f'url = "{url}"', f'output = "{destination.as_posix()}"']
        retry.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n  wrote {retry} with {(len(lines) - 2) // 2:,} files to refetch")

    (manifest_dir / "verify_report.json").write_text(
        json.dumps(
            {
                "expected": total,
                "missing": len(missing),
                "empty": len(empty),
                "truncated": len(truncated),
                "stray_parts": len(stray_parts),
                "complete": total - bad,
                "per_split": {
                    s: {"have": by_split_total[s] - by_split_missing[s],
                        "total": by_split_total[s]}
                    for s in by_split_total
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

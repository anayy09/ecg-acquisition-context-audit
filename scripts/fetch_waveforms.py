#!/usr/bin/env python
"""Fetch the MDS-ED waveform subset, in parallel, resumably.

Replaces the `curl -K` route, which does not pair `output` with `url`
positionally inside a config file and collapses to one global `-o`. This does the
job directly: a thread pool of HTTP workers, a size check so an interrupted run
resumes for free, and live throughput reporting so a slow link is visible
immediately rather than after two days.

Only the 129,057 records MDS-ED references are fetched, out of MIMIC-IV-ECG's
800,035. See scripts/make_waveform_manifest.py.

    python scripts/fetch_waveforms.py --workers 16
    python scripts/fetch_waveforms.py --probe 400        # measure, download nothing
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import derived_root  # noqa: E402

USER_AGENT = "ecg-acquisition-context/1.0 (research; contact via repo)"
TIMEOUT = 60
RETRIES = 3


def load_targets(manifest_dir: Path, name: str = "curl_config.txt") -> list[tuple[str, Path]]:
    """(url, destination) pairs, read from the generated manifest."""
    config = manifest_dir / name
    if not config.is_file():
        raise FileNotFoundError(
            f"{config} not found. Run scripts/make_waveform_manifest.py first."
        )
    urls: list[str] = []
    outs: list[str] = []
    for line in config.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("url ="):
            urls.append(line.split("=", 1)[1].strip().strip('"'))
        elif line.startswith("output ="):
            outs.append(line.split("=", 1)[1].strip().strip('"'))
    if len(urls) != len(outs):
        raise ValueError(f"manifest is malformed: {len(urls)} urls, {len(outs)} outputs")
    return list(zip(urls, [Path(o) for o in outs]))


def fetch(url: str, destination: Path) -> tuple[str, int]:
    """Download one file. Returns (status, bytes). Skips a complete file."""
    if destination.exists() and destination.stat().st_size > 0:
        return "skip", destination.stat().st_size

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                payload = response.read()
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp name and move, so an interrupted write never leaves
            # a truncated file that the skip check would later accept as done.
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            return "ok", len(payload)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    return f"fail:{type(last).__name__}", 0


class Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.ok = 0
        self.skipped = 0
        self.failed = 0
        self.bytes = 0
        self.started = time.time()
        self.lock = threading.Lock()

    def record(self, status: str, size: int) -> None:
        with self.lock:
            self.done += 1
            self.bytes += size
            if status == "ok":
                self.ok += 1
            elif status == "skip":
                self.skipped += 1
            else:
                self.failed += 1

    def line(self) -> str:
        elapsed = max(1e-6, time.time() - self.started)
        rate = self.done / elapsed
        remaining = (self.total - self.done) / rate if rate > 0 else float("inf")
        return (
            f"  {self.done:>7,}/{self.total:,}  "
            f"{100.0 * self.done / max(1, self.total):5.1f}%  "
            f"{rate:6.1f} files/s  {self.bytes / elapsed / 1e6:6.2f} MB/s  "
            f"ok={self.ok:,} skip={self.skipped:,} fail={self.failed:,}  "
            f"eta {remaining / 3600:5.2f} h"
        )


def run(targets: list[tuple[str, Path]], workers: int, report_every: float) -> Progress:
    progress = Progress(len(targets))
    work: queue.Queue = queue.Queue()
    for item in targets:
        work.put(item)

    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            try:
                url, destination = work.get_nowait()
            except queue.Empty:
                return
            status, size = fetch(url, destination)
            progress.record(status, size)
            work.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for thread in threads:
        thread.start()

    try:
        last = 0.0
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
            if time.time() - last >= report_every:
                print(progress.line(), flush=True)
                last = time.time()
    except KeyboardInterrupt:
        stop.set()
        print("\n  interrupted; rerun to resume from where it stopped", flush=True)
    for thread in threads:
        thread.join(timeout=5)
    return progress


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--probe", type=int, default=0,
                        help="download only N files, to measure throughput")
    parser.add_argument("--offset", type=int, default=0,
                        help="skip the first N targets; use with --probe to measure "
                             "fresh files rather than counting instant skips")
    parser.add_argument("--report-every", type=float, default=15.0)
    parser.add_argument("--config", default="curl_config.txt",
                        help="manifest file under derived/waveforms/; use "
                             "retry_config.txt to refetch only what verification "
                             "found missing")
    args = parser.parse_args()

    manifest_dir = derived_root(args.data_root) / "waveforms"
    targets = load_targets(manifest_dir, args.config)
    if args.offset:
        targets = targets[args.offset:]
    if args.probe:
        targets = targets[: args.probe]

    present = sum(1 for _, d in targets if d.exists() and d.stat().st_size > 0)
    print(f"waveform fetch: {len(targets):,} files, {present:,} already present, "
          f"{args.workers} workers")

    progress = run(targets, args.workers, args.report_every)
    elapsed = time.time() - progress.started
    print(progress.line())
    print(f"\ndone in {elapsed / 60:.1f} min: ok={progress.ok:,} "
          f"skipped={progress.skipped:,} failed={progress.failed:,}, "
          f"{progress.bytes / 1e9:.2f} GB")

    report = manifest_dir / ("probe_report.json" if args.probe else "fetch_report.json")
    report.write_text(
        json.dumps(
            {
                "n_targets": len(targets),
                "ok": progress.ok,
                "skipped": progress.skipped,
                "failed": progress.failed,
                "bytes": progress.bytes,
                "elapsed_seconds": round(elapsed, 1),
                "files_per_second": round(progress.done / max(1e-6, elapsed), 2),
                "mb_per_second": round(progress.bytes / max(1e-6, elapsed) / 1e6, 3),
                "workers": args.workers,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 1 if progress.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

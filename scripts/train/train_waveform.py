#!/usr/bin/env python
"""Train one waveform arm, resumably, against a wall clock.

The local-compute constraint removed the cluster, and with it the requeue. A local run that dies has
nothing to restart it, so resume is not a convenience here: it is the only thing
between a power cut and a lost day. Every checkpoint therefore carries the model,
the optimizer, the epoch and step counters, and the RNG state of torch, numpy and
python, so a resumed run continues the same trajectory rather than a similar one.
the corresponding check interrupts a short run and asserts the
resumed weights match an uninterrupted run bit for bit.

Model selection follows the published setup: the checkpoint with the best
validation macro AUROC is the one evaluated on test, as `ModelCheckpoint(
save_top_k=1, monitor='macro', mode='max')` and `trainer.test(ckpt_path='best')`
specify.

    python scripts/train/train_waveform.py --arm R6_waveform --seed 0
    python scripts/train/train_waveform.py --arm R6_waveform --seed 0 --resume
    python scripts/train/train_waveform.py --arm R6_waveform --seed 0 --max-hours 6
    python scripts/train/train_waveform.py --arm R6_waveform --seed 0 --epochs 1 --time-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for extra in (str(_REPO), str(_REPO / "scripts"), str(_HERE)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from contract import load_base_config, load_contract  # noqa: E402
from paths import derived_root  # noqa: E402

from data import TARGETS, WaveformDataset, build_static_block  # noqa: E402
from model import build_from_config, focal_bce_masked  # noqa: E402

#: Which static block each waveform arm reads. R6 sees signal only.
ARM_STATIC_BLOCKS = {
    "R6_waveform": [],
    "R7_waveform_demo_acq": ["demographics", "acq_pre"],
    "R8_waveform_matched": [],
}


#: Touch this file to ask a running batch to stop at the next epoch boundary.
#: One file stops every arm and seed, because a three-day batch is one thing to
#: the person who wants their GPU back.
GLOBAL_STOP = "STOP_P2"


def save_atomic(payload: dict, path: Path) -> None:
    """Write a checkpoint that a kill cannot truncate.

    `torch.save` straight to `last.pt` leaves a window, tens of megabytes wide,
    in which the process can die and leave a file that exists, has a plausible
    size, and fails to load. On a three-day unattended run that window is the
    difference between losing an epoch and losing the run. Writing beside the
    target and renaming closes it: `os.replace` is atomic on one volume.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def stop_requested(out_dir: Path, derived: Path) -> str:
    """Has anyone asked this run, or the whole batch, to stop?"""
    for path, scope in ((derived / GLOBAL_STOP, "batch"), (out_dir / "STOP", "run")):
        if path.exists():
            return f"{scope} stop requested via {path}"
    return ""


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    # `torch.load(..., map_location=device)` moves every tensor in the payload,
    # including the RNG states, onto the GPU. `set_rng_state` requires a CPU
    # ByteTensor and rejects a CUDA one with a message that names the type but
    # not the cause, so the states are forced back explicitly here.
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(
            [s.cpu().to(torch.uint8) for s in state["torch_cuda"]]
        )


@torch.no_grad()
def evaluate(model, loader, device, n_targets: int):
    """Score every crop. Returns crop-level scores plus the indices to fold by."""
    model.eval()
    scores, ys, masks, locals_, crops = [], [], [], [], []
    for batch in loader:
        seq = batch["seq"].to(device, non_blocking=True)
        static = batch.get("static")
        logits = model(seq, static.to(device) if static is not None else None)
        scores.append(torch.sigmoid(logits.float()).cpu())
        ys.append(batch["y"])
        masks.append(batch["mask"])
        locals_.append(batch["local"])
        crops.append(batch["crop"])
    return (
        torch.cat(scores).numpy(),
        torch.cat(ys).numpy(),
        torch.cat(masks).numpy(),
        torch.cat(locals_).numpy(),
        torch.cat(crops).numpy(),
    )


def macro_auroc(scores: np.ndarray, y: np.ndarray, mask: np.ndarray, targets: list[str]) -> dict:
    """Per-target AUROC and their unweighted mean, the published macro."""
    from sklearn.metrics import roc_auc_score

    per = {}
    for k, name in enumerate(targets):
        keep = mask[:, k] > 0
        yt, st = y[keep, k], scores[keep, k]
        per[name] = float(roc_auc_score(yt, st)) if len(np.unique(yt)) > 1 else float("nan")
    finite = [v for v in per.values() if np.isfinite(v)]
    return {
        "per_target": per,
        "macro": float(np.mean(finite)) if finite else float("nan"),
        "n_targets_scored": len(finite),
    }


def fold_to_records(scores, y, mask, locals_, n_records: int):
    """Average the crops of each record, giving one row per ECG."""
    n_t = scores.shape[1]
    summed = np.zeros((n_records, n_t), dtype=np.float64)
    counts = np.zeros((n_records, 1), dtype=np.float64)
    y_out = np.zeros((n_records, n_t), dtype=np.float32)
    m_out = np.zeros((n_records, n_t), dtype=np.float32)
    np.add.at(summed, locals_, scores.astype(np.float64))
    np.add.at(counts, locals_, 1.0)
    y_out[locals_] = y
    m_out[locals_] = mask
    seen = counts[:, 0] > 0
    summed[seen] /= counts[seen]
    return summed[seen].astype(np.float32), y_out[seen], m_out[seen], np.flatnonzero(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARM_STATIC_BLOCKS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override configured epochs")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=0.0,
                        help="stop cleanly after this much wall clock; 0 disables")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default=None, help="checkpoint dir; defaults under $ECG_DATA_ROOT")
    parser.add_argument("--tag", default="", help="suffix for the checkpoint dir, e.g. a tuning id")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=0, help="debug/timing only")
    parser.add_argument("--time-only", action="store_true",
                        help="run the configured epochs, report timing, write no predictions")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="train on a partially preprocessed memmap; timing runs only")
    args = parser.parse_args()

    base = load_base_config()
    contract = load_contract()
    w = base["model"]["waveform"]
    epochs = args.epochs if args.epochs is not None else int(w["epochs"])
    lr = args.lr if args.lr is not None else float(w["lr"])
    dropout = args.dropout if args.dropout is not None else float(w["dropout"])
    batch_size = args.batch_size or int(w["batch_size"])
    crop = int(base["features"]["waveform"]["crop_samples"])
    weight_decay = float(w["weight_decay"])
    gamma = float(w["loss_gamma"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device; this will be unusably slow", file=sys.stderr)

    derived = derived_root(args.data_root)
    out_dir = Path(args.out) if args.out else (
        derived / "checkpoints" / f"{args.arm}__seed{args.seed}{('__' + args.tag) if args.tag else ''}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)

    blocks = ARM_STATIC_BLOCKS[args.arm]
    static, static_cols, static_stats = None, [], {}
    if blocks:
        import pandas as pd
        split_col = pd.read_parquet(derived / "cohort_mdsed.parquet", columns=["split"])["split"]
        static, static_cols, static_stats = build_static_block(derived, blocks, contract, split_col)

    common = dict(data_root=args.data_root, crop=crop, static=static, targets=TARGETS)
    ds_train = WaveformDataset("train", train=True, seed=args.seed, **common)
    ds_val = WaveformDataset("val", train=False, seed=args.seed, **common)
    ds_test = WaveformDataset("test", train=False, seed=args.seed, **common)

    # A partial memmap grows while the job runs, so the record set is not fixed
    # and the run is not reproducible from its config. Refuse rather than
    # produce a plausible number over whatever happened to be ready.
    if not ds_train.is_complete and not args.allow_incomplete:
        print(
            f"REFUSING: the 100 Hz memmap holds {ds_train.n_preprocessed:,} of "
            f"{ds_train.n_cohort:,} records. Finish "
            "`python scripts/preprocess_waveforms.py`, or pass --allow-incomplete "
            "for a timing run that will not be registered.",
            file=sys.stderr,
        )
        return 2

    dl_kwargs = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    if args.num_workers:
        dl_kwargs["persistent_workers"] = True
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False, **dl_kwargs)
    dl_val = DataLoader(ds_val, batch_size=batch_size * 2, shuffle=False, **dl_kwargs)
    dl_test = DataLoader(ds_test, batch_size=batch_size * 2, shuffle=False, **dl_kwargs)

    model = build_from_config(
        base, n_targets=len(TARGETS),
        static_dim=0 if static is None else static.shape[1],
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ckpt_last = out_dir / "last.pt"
    ckpt_best = out_dir / "best.pt"
    start_epoch, global_step, best_macro = 0, 0, -1.0
    history: list[dict] = []
    elapsed_before = 0.0
    if args.resume and ckpt_last.is_file():
        state = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_macro = float(state["best_macro"])
        history = list(state.get("history", []))
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
        restore_rng(state["rng"])
        print(f"resumed from {ckpt_last} at epoch {start_epoch}, step {global_step}")

    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    started = time.time()
    stopped_early = ""
    epoch_seconds: list[float] = []

    for epoch in range(start_epoch, epochs):
        model.train()
        # The crop schedule is a pure function of (seed, epoch, row), so this
        # is what makes a resumed epoch see the same crops as an uninterrupted
        # one. See WaveformDataset._crop_offset.
        ds_train.set_epoch(epoch)
        epoch_start = time.time()
        running, n_batches = 0.0, 0
        for bi, batch in enumerate(dl_train):
            if args.limit_train_batches and bi >= args.limit_train_batches:
                break
            seq = batch["seq"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            st = batch.get("static")
            optimizer.zero_grad(set_to_none=True)
            loss = focal_bce_masked(model(seq, st.to(device) if st is not None else None),
                                    y, mask, gamma=gamma)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            n_batches += 1
            global_step += 1
        epoch_seconds.append(time.time() - epoch_start)

        val = macro_auroc(*evaluate(model, dl_val, device, len(TARGETS))[:3], TARGETS)
        history.append({
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "val_macro": val["macro"],
            "epoch_seconds": round(epoch_seconds[-1], 1),
        })
        print(f"epoch {epoch:3d}  loss {history[-1]['train_loss']:.4f}  "
              f"val_macro {val['macro']:.4f}  {epoch_seconds[-1] / 60:.1f} min", flush=True)

        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_macro": max(best_macro, val["macro"]),
            "history": history,
            "rng": rng_state(),
            "elapsed_seconds": elapsed_before + (time.time() - started),
            "args": vars(args),
        }
        save_atomic(payload, ckpt_last)
        if val["macro"] > best_macro:
            best_macro = val["macro"]
            save_atomic({**payload, "best_macro": best_macro, "val": val}, ckpt_best)

        if args.max_hours and (elapsed_before + time.time() - started) > args.max_hours * 3600:
            stopped_early = f"wall clock budget of {args.max_hours} h reached after epoch {epoch}"
            print(stopped_early, flush=True)
            break
        # Checked only here, once the epoch's checkpoint is durably on disk, so
        # a cooperative stop always lands on a resumable boundary.
        asked = stop_requested(out_dir, derived)
        if asked:
            stopped_early = f"{asked}; stopped cleanly after epoch {epoch}"
            print(stopped_early, flush=True)
            print(f"resume with: python scripts/run_p2.py   (delete {GLOBAL_STOP} first)",
                  flush=True)
            break

    total_seconds = elapsed_before + (time.time() - started)
    peak_vram = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0

    summary = {
        "arm": args.arm,
        "seed": args.seed,
        "epochs_configured": epochs,
        "epochs_completed": len(history),
        "lr": lr,
        "dropout": dropout,
        "batch_size": batch_size,
        "crop_samples": crop,
        "loss_gamma": gamma,
        "best_val_macro": best_macro,
        "wall_seconds": round(total_seconds, 1),
        "gpu_hours": round(total_seconds / 3600, 3),
        "median_epoch_seconds": round(float(np.median(epoch_seconds)), 1) if epoch_seconds else None,
        "peak_vram_mib": round(peak_vram, 1),
        "gpu_model": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "static_dim": 0 if static is None else int(static.shape[1]),
        "static_columns": static_cols,
        "n_train_records": len(ds_train.rows),
        "n_val_items": len(ds_val),
        "n_test_items": len(ds_test),
        "stopped_early": stopped_early,
        "history": history,
        "memmap_complete": ds_train.is_complete,
        "n_preprocessed": ds_train.n_preprocessed,
        "preprocess_hash": ds_train.meta["preprocess_hash"],
        "content_hash": ds_train.meta.get("content_hash", ""),
    }
    if static_stats:
        summary["static_standardisation"] = static_stats
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if stopped_early and not args.time_only and len(history) < epochs:
        # Scoring a half-trained model would register a run that looks complete.
        print("", file=sys.stderr)
        print(stopped_early, file=sys.stderr)
        print("not scoring: this run is unfinished. Re-run with --resume to continue.",
              file=sys.stderr)
        return 3

    if args.time_only:
        print(json.dumps({k: summary[k] for k in
                          ("median_epoch_seconds", "peak_vram_mib", "gpu_hours",
                           "epochs_completed", "best_val_macro")}, indent=2))
        return 0

    if not ckpt_best.is_file():
        print("no best checkpoint was written; nothing to score", file=sys.stderr)
        return 1
    model.load_state_dict(torch.load(ckpt_best, map_location=device)["model"])
    scores, y, mask, locals_, crops = evaluate(model, dl_test, device, len(TARGETS))
    crop_level = macro_auroc(scores, y, mask, TARGETS)
    r_scores, r_y, r_mask, r_rows = fold_to_records(scores, y, mask, locals_, len(ds_test.rows))
    record_level = macro_auroc(r_scores, r_y, r_mask, TARGETS)

    np.savez_compressed(
        out_dir / "test_predictions.npz",
        crop_scores=scores, crop_y=y, crop_mask=mask, crop_local=locals_, crop_index=crops,
        record_scores=r_scores, record_y=r_y, record_mask=r_mask,
        subject_id=ds_test.subject_id[r_rows], study_id=ds_test.study_id[r_rows],
        targets=np.array(TARGETS),
    )
    result = {
        "crop_level": crop_level,
        "record_level": record_level,
        "n_crop_rows": int(scores.shape[0]),
        "n_record_rows": int(r_scores.shape[0]),
        "n_patients": int(np.unique(ds_test.subject_id[r_rows]).size),
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\ncrop-level macro   {crop_level['macro']:.4f}  over {result['n_crop_rows']:,} rows")
    print(f"record-level macro {record_level['macro']:.4f}  over {result['n_record_rows']:,} rows, "
          f"{result['n_patients']:,} patients")
    print(f"peak VRAM {peak_vram:.0f} MiB, {total_seconds / 3600:.2f} GPU-hours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""The published waveform arm, assembled from vendored S4 plus a pooling head.

This is the composition the MDS-ED signal-only deterioration config specifies.
Every argument below traces to a field in `configs/published_reference.yaml`,
which was read from the authors' committed config rather than from the paper;
nothing here is a design choice of ours. The mapping, for a reader checking it
against `third_party/clinical_ts/reference/`:

  ts/enc: none        -> no encoder stage; the 12 raw channels are the input
  ts/pred: s4         -> S4Predictor, which builds S4Model with
                         d_input=channels (12) because channels != model_dim,
                         transposed_input=False, pooling=False, d_output=None,
                         bidirectional=not causal, layer_norm=not batchnorm,
                         l_max=input length (250)
  ts/head: pool       -> PoolingHead with multi_prediction False, so a global
                         AdaptiveAvgPool1d(1) over time, then, because
                         output_layer is True, nn.Linear(model_dim, n_targets)

Shapes, for the published settings:

  input   (B, 250, 12)
  S4Model (B, 250, 512)
  pool    (B, 512)
  linear  (B, 15)        one logit per deterioration target
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from third_party.clinical_ts.ts.s4_modules.s4_model import S4Model  # noqa: E402


class WaveformS4(nn.Module):
    """S4 backbone, global average pool over time, linear head over 15 targets.

    `static_dim` is 0 for `R6_waveform`. `R7_waveform_demo_acq` sets it to the
    width of the demographics-plus-ACQ_PRE block, which is concatenated to the
    pooled representation before the linear head. That is the only structural
    difference between the two arms, and it is deliberately the smallest one
    that can carry the extra inputs: anything more elaborate would make row 7 a
    different model rather than row 6 plus context.
    """

    def __init__(
        self,
        n_targets: int = 15,
        input_channels: int = 12,
        input_length: int = 250,
        model_dim: int = 512,
        state_dim: int = 8,
        layers: int = 4,
        dropout: float = 0.2,
        tie_dropout: bool = True,
        prenorm: bool = False,
        layer_norm: bool = True,
        bidirectional: bool = True,
        backbone: str = "s42",
        static_dim: int = 0,
    ) -> None:
        super().__init__()
        self.n_targets = n_targets
        self.input_length = input_length
        self.static_dim = static_dim

        self.backbone = S4Model(
            d_input=input_channels if input_channels != model_dim else None,
            d_output=None,          # decoder disabled; the head owns the output layer
            d_state=state_dim,
            d_model=model_dim,
            n_layers=layers,
            dropout=dropout,
            tie_dropout=tie_dropout,
            prenorm=prenorm,
            l_max=input_length,
            transposed_input=False,  # we feed (B, L, C)
            bidirectional=bidirectional,
            layer_norm=layer_norm,
            pooling=False,           # PoolingHead below does it
            backbone=backbone,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(model_dim + static_dim, n_targets)

    def forward(self, seq: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        """seq: (B, L, C) -> logits (B, n_targets)."""
        x = self.backbone(seq)              # (B, L, model_dim)
        x = self.pool(x.transpose(1, 2))    # (B, model_dim, 1)
        x = x.squeeze(-1)                   # (B, model_dim)
        if self.static_dim:
            if static is None:
                raise ValueError("static_dim > 0 but no static tensor was passed")
            x = torch.cat([x, static], dim=-1)
        return self.head(x)


def focal_bce_masked(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """The published `bcef` loss with `ignore_nans: True`, reproduced exactly.

    Transcribed from `clinical_ts.loss.supervised.BinaryCrossEntropyFocalLoss`,
    `ignore_nans` branch. Three details are load-bearing and none of them is
    what "multi-label BCE" would suggest:

      1. It is *focal* BCE, weighting each entry by ``(1 - p_t) ** gamma`` with
         ``gamma = 2`` from `BCEFLossConfig`. On targets whose prevalence is
         under one percent this is not a small correction.
      2. The reduction is a **mean within each target, then a sum across the 15
         targets**, not a global mean over observed entries. The sum makes the
         gradient scale roughly fifteen times what a global mean would give, so
         the published learning rate is only the published learning rate under
         this reduction.
      3. Every target is weighted equally regardless of how many rows define
         it. `deterioration_severe_hypoxemia` is undefined on 41,125 of 129,057
         rows and still contributes the same as a fully observed target.

    `mask` is 1 where the target is defined and 0 where MDS-ED's -999 sentinel
    marks it undefined. A target with no defined row in a batch contributes
    nothing, as in the original, where the empty case is skipped.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    modulation = torch.pow(1.0 - p_t, gamma)
    per_entry = modulation * bce * mask

    counts = mask.sum(dim=0)                       # defined rows per target
    per_target = per_entry.sum(dim=0) / counts.clamp(min=1.0)
    return (per_target * (counts > 0).to(per_target.dtype)).sum()


def build_from_config(
    base: dict, n_targets: int, static_dim: int = 0, dropout: float | None = None
) -> WaveformS4:
    """Instantiate straight from `configs/base.yaml`, so the config is the spec.

    `dropout` overrides the configured value for the tuning search. It is a
    constructor argument rather than something set on the built model, because
    with `tie_dropout: true` the S4 stack uses `DropoutNd`, not `nn.Dropout`:
    walking the modules and setting `.p` on every `nn.Dropout` would silently
    miss every dropout that matters and tune a parameter that has no effect.
    """
    w = base["model"]["waveform"]
    f = base["features"]["waveform"]
    if w["type"] != "s4":
        raise ValueError(f"model.waveform.type is {w['type']!r}, expected 's4'; see the pinned architecture")
    return WaveformS4(
        n_targets=n_targets,
        input_channels=int(f["leads"]),
        input_length=int(f["crop_samples"]),
        model_dim=int(w["model_dim"]),
        state_dim=int(w["state_dim"]),
        layers=int(w["layers"]),
        dropout=float(w["dropout"] if dropout is None else dropout),
        tie_dropout=bool(w["tie_dropout"]),
        prenorm=bool(w["prenorm"]),
        layer_norm=not bool(w["batchnorm"]),
        bidirectional=not bool(w["causal"]),
        backbone=str(w["backbone"]),
        static_dim=static_dim,
    )

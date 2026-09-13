"""Manuscript figure style, vendored so this repository runs standalone.

Source: anayy09/claude-research-skills, manuscript-figures/scripts/figstyle.py
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from cycler import cycler

MM_PER_INCH = 25.4

# Okabe-Ito colorblind-safe palette, ordered for plotting priority.
OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "pink": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
}
OKABE_ITO_LIST = list(OKABE_ITO.values())

# Column widths in mm. Verify against current author guidelines for real
# submissions; these are the widely published values.
COLUMN_WIDTHS_MM = {
    "default":  {"single": 89.0,  "onehalf": 136.0, "double": 183.0},
    "nature":   {"single": 89.0,  "onehalf": 136.0, "double": 183.0},
    "elsevier": {"single": 90.0,  "onehalf": 140.0, "double": 190.0},
    "springer": {"single": 84.0,  "onehalf": 129.0, "double": 174.0},
    "ieee":     {"single": 88.9,  "onehalf": 135.0, "double": 181.9},
    "science":  {"single": 57.0,  "onehalf": 121.0, "double": 184.0},
    "neurips":  {"single": 139.7, "onehalf": 139.7, "double": 139.7},  # one-column format
    "icml":     {"single": 82.55, "onehalf": 127.0, "double": 171.45},
}

GOLDEN = 0.6180339887


def fig_size(
    width: str = "single",
    journal: str = "default",
    aspect: float = GOLDEN,
    height_mm: float | None = None,
    width_mm: float | None = None,
) -> tuple[float, float]:
    """Figure size in inches for a given column slot.

    width:    "single" | "onehalf" | "double" (ignored if width_mm given)
    journal:  key into COLUMN_WIDTHS_MM
    aspect:   height / width (ignored if height_mm given)
    """
    if width_mm is None:
        try:
            width_mm = COLUMN_WIDTHS_MM[journal.lower()][width]
        except KeyError as e:
            raise KeyError(
                f"Unknown journal/slot {journal!r}/{width!r}. "
                f"Journals: {sorted(COLUMN_WIDTHS_MM)}; slots: single, onehalf, double. "
                f"Or pass width_mm= directly."
            ) from e
    h_mm = height_mm if height_mm is not None else width_mm * aspect
    return (width_mm / MM_PER_INCH, h_mm / MM_PER_INCH)


def resolve_family(candidates: Iterable[str]) -> str:
    """First installed family from candidates, else matplotlib's own fallback.

    Used to keep mathtext on the same typeface as the rest of the figure.
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


def apply_style(
    base_size: float = 7.0,
    font: Sequence[str] = ("Arial", "Helvetica", "DejaVu Sans"),
    palette: Iterable[str] = OKABE_ITO_LIST,
) -> None:
    """Set rcParams for manuscript output. Call once, before creating figures.

    base_size is the axis-label size in pt at FINAL printed size; ticks and
    legends are one point smaller. Text stays text in exported PDFs
    (fonttype 42) and SVGs (fonttype 'none').
    """
    small = max(base_size - 1.0, 5.0)
    family = resolve_family(font)
    mpl.rcParams.update({
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": list(font),
        "font.size": base_size,
        "axes.labelsize": base_size,
        "axes.titlesize": base_size,      # discouraged, but if used, not larger
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "legend.fontsize": small,
        "legend.title_fontsize": small,
        "figure.titlesize": base_size,
        # Mathtext rides on the figure's own family. The matplotlib default
        # ("dejavusans") silently embeds a SECOND typeface the moment a label
        # contains $...$, which breaks the one-font rule everywhere else in
        # this skill and shows up in the PDF as a stray DejaVu subset.
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        # Geometry
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "errorbar.capsize": 2.0,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        # Legend
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.4,
        # Color
        "axes.prop_cycle": cycler(color=list(palette)),
        "image.cmap": "viridis",
        # Export: keep text as text
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,        # screen preview only
        "savefig.dpi": 600,       # raster export default
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def label_panels(
    axes,
    labels: Sequence[str] | None = None,
    fontsize: float = 8.0,
    uppercase: bool = False,
    dx_pt: float = -2.0,
    dy_pt: float = 2.0,
) -> None:
    """Bold panel labels (a, b, c ...) just outside each axes' top-left corner.

    axes: dict from subplot_mosaic (labels default to its keys), or a
    sequence of Axes (labels default to a, b, c ...). uppercase=True for
    venues using A, B, C. Nudge with dx_pt/dy_pt if a label collides with a
    y-axis label.
    """
    if isinstance(axes, Mapping):
        items = list(axes.items())
        if labels is not None:
            items = list(zip(labels, [ax for _, ax in items]))
    else:
        seq = list(axes)
        if labels is None:
            labels = [chr(ord("a") + i) for i in range(len(seq))]
        items = list(zip(labels, seq))
    for lab, ax in items:
        text = str(lab).upper() if uppercase else str(lab)
        ax.annotate(
            text, xy=(0, 1), xycoords="axes fraction",
            xytext=(dx_pt, dy_pt), textcoords="offset points",
            fontsize=fontsize, fontweight="bold", ha="right", va="bottom",
            annotation_clip=False,
        )


def save_figure(
    fig,
    stem: str,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 600,
    transparent: bool = False,
) -> list[str]:
    """Save fig as stem.<fmt> for each format, at the figure's declared size.

    Deliberately no bbox_inches='tight': the figure was built at final size
    and tight-cropping would change it. Returns the written paths.
    """
    stem = os.fspath(stem)
    d = os.path.dirname(stem)
    if d:
        os.makedirs(d, exist_ok=True)
    written = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        path = f"{stem}.{fmt}"
        kwargs = {"transparent": transparent}
        if fmt in ("png", "tif", "tiff", "jpg", "jpeg"):
            kwargs["dpi"] = dpi
        if fmt in ("tif", "tiff"):
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        written.append(path)
    return written


if __name__ == "__main__":
    # Smoke test: renders a demo figure into ./figstyle_demo.pdf/.png
    import numpy as np

    apply_style()
    fig = plt.figure(figsize=fig_size("double", aspect=0.42), layout="constrained")
    axs = fig.subplot_mosaic([["a", "b"]])
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 60)
    for i, name in enumerate(["Proposed", "Baseline"]):
        m = np.sin(x + i) * 0.3 + 0.6 + 0.02 * i
        s = 0.05 + 0.02 * rng.random(x.size)
        line, = axs["a"].plot(x, m, label=name)
        axs["a"].fill_between(x, m - s, m + s, color=line.get_color(), alpha=0.18, lw=0)
    axs["a"].set_xlabel("Epoch"); axs["a"].set_ylabel("Validation AUROC")
    axs["a"].legend()
    vals = [rng.normal(0.7 + 0.05 * i, 0.04, 20) for i in range(3)]
    axs["b"].boxplot(vals, widths=0.5, showfliers=False)
    for i, v in enumerate(vals, start=1):
        axs["b"].plot(rng.normal(i, 0.06, v.size), v, "o", ms=2.2, alpha=0.6,
                      mew=0, color=OKABE_ITO["blue"])
    axs["b"].set_xticklabels(["A", "B", "C"]); axs["b"].set_ylabel("Dice score")
    label_panels(axs)
    print("wrote:", save_figure(fig, "figstyle_demo"))

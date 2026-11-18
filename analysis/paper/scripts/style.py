"""Shared style and paths for all paper figures (no re-analysis; plots verified result files only).

Style v2 (2026-11-05): one typeface (Arial), recessive axes, hairline grid, and a semantic colour system whose
categorical sets were checked with the dataviz validator (validate_palette.js, light surface, --pairs all):
  MODEL  B2 / CNN-GRU / CNN-only            #2a78d6 / #eb6834 / #1baf7a     PASS (aqua < 3:1 -> always direct-labelled)
  ANOM   sequence / tempo                   #6250d6 / #c98500               PASS (amber 2.99:1 -> labelled)
  ROLE   calibration / sequence / tempo     #1baf7a / #6250d6 / #c98500     PASS
Neutrals (B1, normal cycles, training data, test data) are greys and carry no categorical hue.
Never use blue (#2a78d6) next to violet (#6250d6) in one figure: that pair FAILS (CVD dE 4.5, normal 10.5).
Old names (C, MODEL_COL, COL, FULL, panel, save, setup) are kept so every figure script keeps working.
"""
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

COL = 3.35   # single column, 8.5 cm (inches)
FULL = 6.69  # full width = \textwidth of main.tex (a4paper, 2 cm margins: 17.0 cm); figures are included at width=\textwidth

# ---------------------------------------------------------------- ink and neutrals
INK = "#1f1f1e"        # primary text, axes labels
INK2 = "#52514e"       # secondary text, spines, ticks
MUTED = "#8a8985"      # tertiary text; neutral series (B1, normal cycles)
GRID = "#e8e7e3"       # hairline grid
RULE = "#c9c8c3"       # reference lines (chance diagonal, thresholds when not emphasised)
SURF = "#ffffff"       # figure surface
TRAIN = "#d9d8d3"      # training / fitting data (neutral fill)
TEST = "#3d3c39"       # held-out test data (dark neutral fill)
PANEL_BG = "#f6f6f4"   # optional background for schematic groups

# ---------------------------------------------------------------- validated categorical sets
BLUE, ORANGE, AQUA, VIOLET, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c98500"
MODEL = {"B1": MUTED, "B2": BLUE, "CNN-GRU": ORANGE, "CNN-only": AQUA}
ANOM = {"normal": MUTED, "sequence": VIOLET, "tempo": AMBER, "all": INK}
ROLE = {"train": TRAIN, "calib": AQUA, "test": TEST, "sequence": VIOLET, "tempo": AMBER}
# light tints for bands / fills behind lines (same hue, high lightness)
TINT = {BLUE: "#cde2fb", ORANGE: "#fbd9c9", AQUA: "#c6ecdf", VIOLET: "#ddd8f6", AMBER: "#f6e3b4", MUTED: "#e3e2de"}

# ---------------------------------------------------------------- set-level conventions (critique round, 2026-11-05)
# AQUA means calibration ONLY (Figs 1, 2). CNN-only is drawn as a light orange variant of the CNN-GRU (it is the
# CNN-GRU without the GRU): ORANGE edge, TINT[ORANGE] face. Identity never depends on hue alone: models also differ by marker.
MODEL["CNN-only"] = ORANGE
MODEL_FACE = {"B1": MUTED, "B2": BLUE, "CNN-GRU": ORANGE, "CNN-only": TINT[ORANGE]}
MARKER = {"B2": "s", "CNN-GRU": "D", "CNN-only": "^", "B1": "v"}          # same shapes in every figure
# anomaly-type series: tempo solid AMBER, sequence dashed VIOLET, all anomalies solid INK (thinner, drawn on top);
# anomaly-type points use circles only (colour + line style separate them)
LS_ANOM = {"tempo": "-", "sequence": (0, (4, 2)), "all": "-"}
LW_ANOM = {"tempo": 1.5, "sequence": 1.5, "all": 1.1}
ORDER_ANOM = ["Seq 1–3–5", "Seq 1–4–5", "Fast", "Slow"]                  # group order everywhere (en dashes)
# reference lines: dotted = target level (AUROC 0.90, 95 %, all-data value); dashed = threshold/boundary (alpha, cycle end, 11.8 cm)
REF_TARGET = dict(color=INK2, lw=0.6, ls=(0, (1, 1.5)))
REF_THRESH = dict(color=INK2, lw=0.6, ls=(0, (3, 2)))
RATE_LABEL = "Alarm rate"   # the per-group rate at alpha = 0.10 (detection for anomalous, false alarm for normal cycles)
# layout rules: panel letters are the top-most elements; a shared key sits on the panel-letter row ~0.15 in right of 'a';
# legend entries in sentence case; final figure width within 0.03 in of FULL so every PDF is included at scale 1.00.

# backward-compatible aliases (old scripts)
C = dict(black=INK, orange=ORANGE, sky=TINT[BLUE], green=AQUA, yellow=AMBER, blue=BLUE, verm=ORANGE,
         pink=VIOLET, grey=MUTED)
MODEL_COL = MODEL


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold", "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.3,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.5, "lines.markersize": 4.5, "lines.solid_capstyle": "round",
        "patch.linewidth": 0, "hatch.linewidth": 0.5,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100,
    })


def panel(ax, letter, x=-0.14, y=1.04, title=None):
    """Bold lower-case panel letter at the top-left; optional short panel title after it."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)
    if title:
        ax.text(x + 0.045, y, title, transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left", color=INK2)


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def box(ax, x, y, w, h, text="", fc=SURF, ec=INK2, lw=0.7, ls="-", r=0.012, fontsize=7, color=INK, weight="normal",
        zorder=2, ha="center", transform=None):
    """Rounded box for schematics (data or axes coordinates)."""
    tr = transform or ax.transData
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls,
                       transform=tr, zorder=zorder, mutation_aspect=1)
    ax.add_patch(p)
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.01
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fontsize, color=color, fontweight=weight,
                transform=tr, zorder=zorder + 1, linespacing=1.25)
    return p


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=0.8, ls="-", style="-|>", ms=7, shrink=0, zorder=1, transform=None):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                        shrinkA=shrink, shrinkB=shrink, transform=transform or ax.transData, zorder=zorder)
    ax.add_patch(a)
    return a


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", name)


setup()

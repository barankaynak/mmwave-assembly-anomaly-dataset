"""Fig 5: frame errors -> cycle-level detection. Re-plot (no re-analysis).
Sources:
 reports/faz_a3/01_error_propagation.csv  (synthetic-noise grid: iid / bursty; mean, lo=2.5%, hi=97.5% over seeds; and obs_* rows giving real-source r)
 reports/faz_a/04_anomaly_table.csv       (real sources, D3, pred, 'tum anomali': AUROC, FAR@0.10; r=0 reference = ToF oracle x pred)
 reports/faz_a3/01_error_propagation.md   (section 'Simulasyon gercekci mi?' table: false visits per ToF cycle; parsed from the text)

Style v2 + set conventions (style.py, critique round 2026-11-05):
 - real estimators carry MODEL hues, MODEL_FACE fills and the shared MARKER shapes (square B2, diamond CNN-GRU,
   up-triangle CNN-only drawn as a light-orange face with an orange edge, down-triangle B1);
 - the two synthetic noise types are neutral (INK2) and differ by line style, marker fill and band treatment
   (independent: solid line, filled dots, shaded band, filled bars; bursts: dashed line, open dots, dashed band edges,
   open bars);
 - reference lines: dotted REF_TARGET (AUROC 0.90), dashed REF_THRESH (alpha = 0.10);
 - panel letters are top-most; the shared key sits on the letter row ~0.15 in right of 'a';
 - the canvas is laid out so that the tight-cropped PDF is FULL (= \\textwidth) wide and is included at scale 1.00."""
import re
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import text_to_path
from matplotlib.legend_handler import HandlerTuple
from style import *

ep = pd.read_csv(REP / "faz_a3/01_error_propagation.csv")
tab = pd.read_csv(REP / "faz_a/04_anomaly_table.csv")
tab = tab[(tab.variant == "D3") & (tab.segmentation == "pred") & (tab.atype == "tum anomali")].set_index("state_source")

# r at which each real source sits (from the obs_* tags)
robs = {}
for tag, nm in [("obs_b2", "b2"), ("obs_cnngru", "cnngru"), ("obs_cnnonly", "cnnonly"), ("obs_b1", "b1")]:
    robs[nm] = float(ep[ep.tag == tag].r.iloc[0])
print("robs", robs)

# ------------------------------------------------------------------ encodings
SYN = INK2                                       # one neutral for both synthetic noise types
SYN_STYLE = {                                    # noise -> (legend label, line style, marker face)
    "iid": ("Independent flips", "-", SYN),
    "bursty": ("Bursts fitted to B2", (0, (4.2, 1.8)), SURF),
}
BAND = TINT[MUTED]                               # independent flips: shaded band
BAND_A = 0.75
EDGE = dict(color=INK2, lw=0.4, ls=(0, (3, 2)))  # bursts: band drawn as its two dashed edges only (no fill)
REAL = {  # key -> model name (colour, face and marker from style.py) and direct label
    "b2": ("B2", "B2"),
    "cnngru": ("CNN-GRU", "CNN–GRU"),
    "cnnonly": ("CNN-only", "CNN-only"),
    "b1": ("B1", "B1"),
}
MSIZE = {"s": 30, "D": 25, "^": 36, "v": 36}       # scatter areas equalised by eye across shapes
LW = 1.3
DASH = "–"                                         # en dash for CNN–GRU (CNN-only keeps its hyphen)

H = 2.6
fig = plt.figure(figsize=(FULL, H))
Y_LET = (H - 0.13) / H                             # baseline of the panel letters and of the key's first row
gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.12, 0.86], wspace=0.40, left=0.07, right=0.995,
                      top=(H - 0.38) / H, bottom=0.16)
axA, axB, axC = [fig.add_subplot(gs[0, i]) for i in range(3)]

ref = tab.loc["tof"]
r0_auc, r0_far = ref.auroc_mean, ref.far_a10
for ax, metric, ylab in [(axA, "auroc_all", "D3 AUROC (all anomalies)"),
                         (axB, "far", r"False-alarm rate at $\alpha$ = 0.10")]:
    for noise, (lab, ls, mfc) in SYN_STYLE.items():
        g = ep[(ep.noise == noise) & (ep.tag == "grid") & (ep.metric == metric)].sort_values("r")
        x = np.r_[0, g.r.values]
        y0 = r0_auc if metric == "auroc_all" else r0_far
        ym = np.r_[y0, g["mean"].values]
        lo = np.r_[y0, g.lo.values]; hi = np.r_[y0, g.hi.values]
        if noise == "iid":
            ax.fill_between(x, lo, hi, color=BAND, alpha=BAND_A, lw=0, zorder=1)
        else:
            for yy in (lo, hi):
                ax.plot(x, yy, zorder=2, dash_capstyle="butt", **EDGE)
        # line without the r=0 marker (the shared ToF point gets its own symbol below)
        ax.plot(x, ym, color=SYN, lw=LW, ls=ls, zorder=3, dash_capstyle="butt")
        ax.plot(x[1:], ym[1:], ls="none", marker="o", ms=3.0, mfc=mfc, mec=SYN, mew=0.8, zorder=4)
    # shared starting point: ToF labels (r = 0)
    ax.plot(0, r0_auc if metric == "auroc_all" else r0_far, ls="none", marker="o", ms=4.6, mfc=SURF, mec=INK,
            mew=1.0, zorder=6)
    ax.set_xlabel("Frame-error rate $r$")
    ax.set_ylabel(ylab)
    ax.set_xlim(-0.012, 0.40)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xticklabels(["0", "0.1", "0.2", "0.3", "0.4"])
    ygrid(ax)


def real_style(name):
    """Scatter kwargs for one real estimator: shared marker; CNN-only = light-orange face with an orange edge."""
    mk = MARKER[name]
    if name == "CNN-only":
        return dict(marker=mk, s=MSIZE[mk], facecolors=MODEL_FACE[name], edgecolors=MODEL[name], linewidths=0.8)
    return dict(marker=mk, s=MSIZE[mk], facecolors=MODEL_FACE[name], edgecolors=SURF, linewidths=0.7)


for nm, (name, _) in REAL.items():
    for ax, yv in [(axA, tab.loc[nm].auroc_mean), (axB, tab.loc[nm].far_a10)]:
        ax.scatter(robs[nm], yv, zorder=8, **real_style(name))

# reference lines: dotted = target level (AUROC 0.90), dashed = threshold (alpha)
axA.axhline(0.9, zorder=2, **REF_TARGET)
axA.text(0.398, 0.906, "0.90", fontsize=6.8, ha="right", va="bottom", color=INK2)
axB.axhline(0.10, zorder=2, **REF_THRESH)
axB.text(0.398, 0.1025, r"$\alpha$ = 0.10", fontsize=6.8, ha="right", va="bottom", color=INK2)
axA.set_ylim(0.5, 1.0)
axA.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axB.set_ylim(0, 0.15)
axB.set_yticks([0, 0.05, 0.10, 0.15])
axB.set_yticklabels(["0", "0.05", "0.10", "0.15"])


# direct labels for the real estimators in (a); leader line where the symbol sits inside a band
def lab(ax, xy, txt, xyt, c, ha="left", leader=False):
    ax.annotate(txt, xy=xy, xytext=xyt, textcoords="data", ha=ha, va="center", fontsize=6.8, color=INK,
                arrowprops=(dict(arrowstyle="-", color=c, lw=0.6, shrinkA=0.5, shrinkB=3.2) if leader else None),
                zorder=9)


yA = {nm: tab.loc[nm].auroc_mean for nm in REAL}
lab(axA, (robs["b2"], yA["b2"]), "B2", (robs["b2"] + 0.013, yA["b2"] + 0.012), MODEL["B2"])
lab(axA, (robs["cnngru"], yA["cnngru"]), REAL["cnngru"][1], (0.168, 0.925), MODEL["CNN-GRU"], leader=True)
lab(axA, (robs["cnnonly"], yA["cnnonly"]), "CNN-only", (robs["cnnonly"] + 0.016, yA["cnnonly"]), MODEL["CNN-only"])
lab(axA, (robs["b1"], yA["b1"]), "B1", (robs["b1"] - 0.016, yA["b1"]), MODEL["B1"], ha="right")

# ------------------------------------------------------------------ panel c: false visits per ToF cycle (md table)
txt = open(REP / "faz_a3/01_error_propagation.md", encoding="utf-8").read()
blk = txt.split("Simulasyon gercekci mi?")[1].split("Sekiller:")[0]
rows = {}
for ln in blk.splitlines():
    m = re.match(r"\|\s*(B2|CNN-GRU|CNN-only|B1)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", ln)
    if m:
        rows[m.group(1)] = dict(r=float(m.group(2)), real=float(m.group(3)), bursty=float(m.group(4)), iid=float(m.group(5)))
print(rows)
names = ["B2", "CNN-GRU", "CNN-only", "B1"]
w, gap = 0.25, 0.03
xc = np.arange(4)
off = {"real": -(w + gap), "iid": 0.0, "bursty": (w + gap)}
real_ec = [MODEL[n] if n == "CNN-only" else "none" for n in names]
real_lw = [0.8 if n == "CNN-only" else 0 for n in names]
axC.bar(xc + off["real"], [rows[n]["real"] for n in names], w, color=[MODEL_FACE[n] for n in names],
        edgecolor=real_ec, linewidth=real_lw, zorder=2)
axC.bar(xc + off["iid"], [rows[n]["iid"] for n in names], w, color=SYN, lw=0, zorder=2)
axC.bar(xc + off["bursty"], [rows[n]["bursty"] for n in names], w, fc=SURF, ec=SYN, lw=0.8, zorder=2)
axC.set_xticks(xc); axC.set_xticklabels(["B2", f"CNN{DASH}\nGRU", "CNN-\nonly", "B1"])
axC.tick_params(axis="x", length=0)
axC.set_xlim(-0.6, 3.6)
axC.set_ylim(0, 2.2)
axC.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
axC.set_yticklabels(["0", "0.5", "1.0", "1.5", "2.0"])
axC.set_ylabel("False visits per cycle")
ygrid(axC)
h_real = tuple(Patch(fc=MODEL_FACE[n], ec=(MODEL[n] if n == "CNN-only" else "none"),
                     lw=(0.8 if n == "CNN-only" else 0)) for n in names)
axC.legend([h_real, Patch(fc=SYN, lw=0), Patch(fc=SURF, ec=SYN, lw=0.8)],
           ["Real errors", "Independent flips", "Bursts"],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}, loc="upper left", bbox_to_anchor=(0.0, 1.02),
           fontsize=6.8, handlelength=2.4, handleheight=0.9, labelspacing=0.35, borderaxespad=0.2,
           frameon=False, borderpad=0.25)

# ------------------------------------------------------------------ width: tight-cropped figure = FULL (scale 1.00)
fig.canvas.draw()
rdr = fig.canvas.get_renderer()
left = gs.left
for _ in range(4):
    wc = fig.get_tightbbox(rdr).width                # content width in inches (save() adds 2 x 0.03 in pad)
    left -= ((FULL - 0.06) - wc) / FULL
    gs.update(left=left)
    fig.canvas.draw()
print("content width (in)", round(fig.get_tightbbox(rdr).width, 4), "left", round(left, 4))

# ------------------------------------------------------------------ panel letters (top-most) aligned to each panel's outer left edge
inv = fig.transFigure.inverted()
letters = {}
for ax, letter in zip((axA, axB, axC), "abc"):
    x0 = inv.transform(ax.yaxis.label.get_window_extent(rdr))[0][0]
    letters[letter] = fig.text(x0, Y_LET, letter, fontsize=9, fontweight="bold", ha="left", va="baseline", color=INK)


# ------------------------------------------------------------------ shared key for a and b, on the letter row
# Drawn by hand (not fig.legend) so that every key label is placed with va="baseline" at exactly the letter baseline,
# in the PDF as well as in the PNG (legend packing aligns the mixed math/plain labels differently per backend).
# Layout in inches: row 1 = [ToF | independent flips] on the letter row, row 2 = [ | bursts] below it.
TI = mtransforms.Affine2D().scale(1 / FULL, 1 / H) + fig.transFigure   # inches from the canvas origin; built on
#                                   transFigure so it follows the shift that save(bbox_inches="tight") applies
FS = 7
PT = 1 / 72                                                # one point in inches
HL, HH, PAD, CSP = 2.1 * FS * PT, 0.7 * FS * PT, 0.5 * FS * PT, 1.1 * FS * PT   # handle length/height, text pad, column gap
ROW = 0.155                                                # baseline-to-baseline distance of the two key rows (in)
YC = 0.034                                                 # handle centre above the baseline (about half the x-height)
y1 = Y_LET * H                                             # letter baseline (in)
y2 = y1 - ROW
a_x1 = letters["a"].get_window_extent(rdr).x1 / fig.dpi    # right edge of the letter 'a' (in)


def key_text(x, y, s):
    return fig.text(x, y, s, transform=TI, fontsize=FS, color=INK, ha="left", va="baseline")


def key_line(xs, ys, **kw):
    ln = Line2D(xs, ys, transform=TI, **kw)
    fig.add_artist(ln)
    return ln


def syn_handle(x0, yb, noise):
    """Band swatch (shaded or dashed edges) + line + centred dot, as drawn in panels a and b."""
    _, ls, mfc = SYN_STYLE[noise]
    yc = yb + YC
    if noise == "iid":
        fig.add_artist(Rectangle((x0, yc - HH / 2), HL, HH, transform=TI, fc=BAND, alpha=BAND_A, lw=0))
    else:
        for yy in (yc - HH / 2, yc + HH / 2):
            key_line([x0, x0 + HL], [yy, yy], dash_capstyle="butt", **EDGE)
    key_line([x0, x0 + HL], [yc, yc], color=SYN, lw=LW, ls=ls, dash_capstyle="butt")
    key_line([x0 + HL / 2], [yc], ls="none", marker="o", ms=3.0, mfc=mfc, mec=SYN, mew=0.8)


# column 1: ToF point (marker starts 0.15 in right of 'a')
r_tof = 4.6 / 2 * PT
x_tof = a_x1 + 0.15 + r_tof
key_line([x_tof], [y1 + YC], ls="none", marker="o", ms=4.6, mfc=SURF, mec=INK, mew=1.0)
# "ToF labels (r = 0)" as three plain Text pieces (italic r) instead of mathtext: the PDF backend rounds mathtext depth
# up to whole points, which lifts a mathtext label ~0.5 pt above the letter baseline
# pieces are spaced by their vector advance widths (the metrics the PDF uses), plus a small italic correction after r
xt = x_tof + r_tof + PAD
for piece, style, kern in (("ToF labels (", "normal", 0.0), ("r", "italic", 0.3), (" = 0)", "normal", 0.0)):
    t_tof = key_text(xt, y1, piece)
    t_tof.set_fontstyle(style)
    adv = text_to_path.get_text_width_height_descent(piece, FontProperties(size=FS, style=style), ismath=False)[0]
    xt += (adv + kern) * PT
fig.canvas.draw()
# column 2: synthetic noise (independent flips on the letter row, bursts below)
x2 = t_tof.get_window_extent(rdr).x1 / fig.dpi + CSP
syn_handle(x2, y1, "iid")
t_iid = key_text(x2 + HL + PAD, y1, SYN_STYLE["iid"][0])
syn_handle(x2, y2, "bursty")
t_bur = key_text(x2 + HL + PAD, y2, SYN_STYLE["bursty"][0])
fig.canvas.draw()

# report the clearances that matter (inches)
key_x1 = max(t.get_window_extent(rdr).x1 for t in (t_iid, t_bur))
key_y0 = t_bur.get_window_extent(rdr).y0
print("key right -> letter b left (in):", round((letters["b"].get_window_extent(rdr).x0 - key_x1) / fig.dpi, 3))
print("key bottom -> axes a top (in):", round((key_y0 - axA.get_window_extent(rdr).y1) / fig.dpi, 3))
print("tight width (in):", round(fig.get_tightbbox(rdr).width + 0.06, 4))
save(fig, "fig5_error_propagation")

"""Fig 3: S0 (random window) vs S1 (cycle-grouped) for B1/B2/CNN-GRU, slope view with broken y-axis.
Sources: reports/<LEAK_DIR>/02_leakage_per_run.csv (per run: small symbols and means) and 02_leakage_summary.csv (S0 - S1
effect, Welch 95 % CI; printed in the effect column beside each panel). LEAK_DIR defaults to 'faz_d' (Phase D: all CNN-GRU
runs on the RTX 3090); set LEAK_DIR=faz_a2 to draw the Phase A2 result files. No re-analysis: means are plain averages of
the per-run rows and are checked against m0/m1 of the summary file; printed effects are the summary values in percentage
points.
Style (set conventions in style.py): model shapes from MARKER (B2 square, CNN-GRU diamond, B1 down-triangle) for runs and
means; means filled (MODEL_FACE, white edge), runs light (TINT fill, model-colour edge). The shared size key sits on the
panel-letter row. The figure is drawn on a fixed FULL-wide canvas and saved without tight cropping, so the PDF is exactly
\\textwidth wide and is included at scale 1.00; the script checks that no artist leaves the canvas.
Layout: both y segments (B2/CNN-GRU above, B1 below) and both panels share one scale (inches per percentage point)."""
import os
import re

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
from style import *

LEAK_DIR = os.environ.get("LEAK_DIR", "faz_d")
print("LEAK_DIR =", LEAK_DIR)
pr = pd.read_csv(REP / LEAK_DIR / "02_leakage_per_run.csv")
sm = pd.read_csv(REP / LEAK_DIR / "02_leakage_summary.csv")

MODELS = ["b1", "b2", "cnngru"]
SKEY = {"b1": "B1", "b2": "B2", "cnngru": "CNN-GRU"}   # keys of the style.py dictionaries
NAME = {"b1": "B1", "b2": "B2", "cnngru": "CNN–GRU"}
COLR = {mod: MODEL[SKEY[mod]] for mod in MODELS}
FACE = {mod: MODEL_FACE[SKEY[mod]] for mod in MODELS}
MARK = {mod: MARKER[SKEY[mod]] for mod in MODELS}
# marker sizes (pt): a square or diamond of size ms covers ms^2, a triangle only ms^2/2, so triangles are scaled up
# (slightly less than sqrt 2: their pointed outline already reads large)
SHAPE_K = {"s": 1.0, "D": 1.0, "v": 1.33}
FOOT = {"s": 1.0, "D": 2 ** 0.5, "v": 1.0}             # footprint width / ms (for the swarm spacing)
MS_MEAN, MS_RUN, MEW_RUN = 4.5, 2.5, 0.5
ND = {"b1": 2, "b2": 2, "cnngru": 1}                   # decimals: caption rule (two for B1/B2, one for CNN-GRU)
METRICS = [("acc", "Accuracy (%)"), ("macro_f1", "Macro-F1 (%)")]
SPLITS = [("S0", 0.0, "random window"), ("S1", 1.0, "cycle-grouped")]
MINUS = "−"


def fmt(v, nd):
    return f"{v:+.{nd}f}".replace("-", MINUS)


def ms_of(mk, base):
    return base * SHAPE_K[mk]


# ---------------------------------------------------------------- data (means + self-check against the summary file)
mean, runs, eff = {}, {}, {}
for m, _ in METRICS:
    for mod in MODELS:
        r = sm[(sm.model == mod) & (sm.metric == m)]
        assert len(r) == 1, (mod, m, len(r))
        r = r.iloc[0]
        for sp, _, _ in SPLITS:
            v = pr[(pr.model == mod) & (pr.split == sp)][m].to_numpy() * 100
            assert len(v) > 0, (mod, sp, m)
            runs[m, mod, sp] = v
            mean[m, mod, sp] = v.mean()
        assert abs(mean[m, mod, "S0"] / 100 - r.m0) < 1e-9 and abs(mean[m, mod, "S1"] / 100 - r.m1) < 1e-9, (mod, m)
        eff[m, mod] = (r.d * 100, r.lo * 100, r.hi * 100)
        nd = ND[mod]  # console print in ASCII (Windows code page)
        print(f"{m:9s} {mod:7s} S0 {mean[m, mod, 'S0']:6.2f}  S1 {mean[m, mod, 'S1']:6.2f}  "
              f"effect {r.d * 100:+.{nd}f} [{r.lo * 100:+.{nd}f}, {r.hi * 100:+.{nd}f}]")

# printed strings of the effect column: integer part, fraction part (decimal-aligned), CI
EFF_STR = {}
for (m, mod), (d, l, h) in eff.items():
    ip, fp = fmt(d, ND[mod]).split(".")
    EFF_STR[m, mod] = (ip, "." + fp, f"[{fmt(l, ND[mod])}, {fmt(h, ND[mod])}]")

# ---------------------------------------------------------------- y ranges on one common scale
PAD = 1.0  # percentage points above / below the extreme run
def span(m, mods):
    v = np.concatenate([runs[m, mod, sp] for mod in mods for sp, _, _ in SPLITS])
    return v.min(), v.max()

UP_SPAN = max(np.diff(span(m, ["b2", "cnngru"]))[0] for m, _ in METRICS) + 2 * PAD
LO_SPAN = max(np.diff(span(m, ["b1"]))[0] for m, _ in METRICS) + 2 * PAD
STEP = MaxNLocator(nbins=6, steps=[1, 2, 5, 10]).tick_values(0, UP_SPAN)
STEP = STEP[1] - STEP[0]

# ---------------------------------------------------------------- geometry (inches) on a fixed FULL-wide canvas
W = FULL
SCALE = 0.128                       # inches per percentage point (both segments, both panels)
H_UP, H_LO = UP_SPAN * SCALE, LO_SPAN * SCALE
TOP, GAP, BOT = 0.236, 0.09, 0.30   # letter/key row, break gap, x labels
H = TOP + H_UP + GAP + H_LO + BOT
LEFT = 0.40                          # panel start -> slope axis (y label + y tick labels)
YLAB = 0.315                         # y-label centre, left of the axis
XL = 0.02                            # panel letter, right of the panel start
XGAP, RMARG = 0.12, 0.035            # clear gap between panel a's effect column and panel b; right margin
XLIM = (-0.41, 1.41)
STRIP = {"b1": 0.14, "b2": 0.14, "cnngru": 0.29}   # run strips sit outside the means: S0 left, S1 right

fig = plt.figure(figsize=(W, H))
REND = fig.canvas.get_renderer()

NUMKW = dict(fontsize=6.5, color=INK2)
NAMEKW = dict(fontsize=7, fontweight="bold")


def text_w(s, **kw):
    """rendered width of a string in inches."""
    t = fig.text(0, 0, s, **kw)
    w = t.get_window_extent(REND).width / fig.dpi
    t.remove()
    return w


# effect column widths measured from the strings actually printed (so the faz_d re-run lays out the same way)
W_NAME = max(text_w(NAME[mod], **NAMEKW) for mod in MODELS)
W_IP = max(text_w(s[0], **NUMKW) for s in EFF_STR.values())
W_FP = max(text_w(s[1], **NUMKW) for s in EFF_STR.values())
W_CI = max(text_w(s[2], **NUMKW) for s in EFF_STR.values())
G_AX, G_GLYPH, G_NAME, G_NUM = 0.10, 0.068, 0.07, 0.075   # axis -> glyph centre -> name -> number -> CI
COLW = G_AX + G_GLYPH + W_NAME + G_NAME + W_IP + W_FP + G_NUM + W_CI
PANEL_W = (W - XGAP - RMARG) / 2
PW = PANEL_W - LEFT - COLW           # slope plot width
print(f"slope width {PW:.3f} in, effect column {COLW:.3f} in")


def add_ax(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def swarm(v, xc, mk):
    """Deterministic beeswarm: shift a run marker sideways only if it would touch one already placed."""
    sx, sy = PW / (XLIM[1] - XLIM[0]), SCALE          # inches per x unit / per percentage point
    d = 0.9 * (ms_of(mk, MS_RUN) * FOOT[mk] + MEW_RUN) / 72   # minimum centre distance (inches)
    placed, xs = [], np.empty(len(v))
    for i in np.argsort(v):
        for o in (0, 1, -1, 2, -2):
            x = xc + o * d / sx
            if all(np.hypot((x - px) * sx, (v[i] - py) * sy) >= d for px, py in placed):
                break
        placed.append((x, v[i])); xs[i] = x
    return xs


def ycoord(ax, y):
    """data y -> points from figure bottom (for label de-confliction)."""
    return ax.transData.transform((0, y))[1] * 72 / fig.dpi


# letter row: one baseline for both panel letters and the shared key
Y_BASE = H - 0.041 - 0.716 * 9 / 72        # baseline of the 9 pt letters, ascender 0.041 in below the canvas top

rows, xpanel = {}, {}
for k, (m, ylab) in enumerate(METRICS):
    xp = k * (PANEL_W + XGAP)
    x0 = xp + LEFT
    xpanel[k] = (xp, x0)
    lo = add_ax(x0, BOT, PW, H_LO)
    up = add_ax(x0, BOT + H_LO + GAP, PW, H_UP)

    # y limits: centre the data of each segment in its fixed span
    for ax, mods, sp_ in ((up, ["b2", "cnngru"], UP_SPAN), (lo, ["b1"], LO_SPAN)):
        a, b = span(m, mods)
        c = (a + b) / 2
        ax.set_ylim(c - sp_ / 2, c + sp_ / 2)
        ax.yaxis.set_major_locator(MultipleLocator(STEP))
        ax.set_xlim(*XLIM)
        ygrid(ax)
        ax.tick_params(axis="y", length=2.5, pad=2)

    # marks: light runs in strips beside the means, filled means joined by the slope line
    for mod in MODELS:
        ax = lo if mod == "b1" else up
        col, mk = COLR[mod], MARK[mod]
        for sp, x, _ in SPLITS:
            v = runs[m, mod, sp]
            xc = x - STRIP[mod] if sp == "S0" else x + STRIP[mod]
            ax.plot(swarm(v, xc, mk), v, ls="none", marker=mk, ms=ms_of(mk, MS_RUN), mfc=TINT[col], mec=col,
                    mew=MEW_RUN, zorder=2, clip_on=False)
        ax.plot([0, 1], [mean[m, mod, "S0"], mean[m, mod, "S1"]], color=col, lw=1.4, zorder=3, solid_capstyle="round")
        ax.plot([0, 1], [mean[m, mod, "S0"], mean[m, mod, "S1"]], ls="none", marker=mk, ms=ms_of(mk, MS_MEAN),
                mfc=FACE[mod], mec="white", mew=0.7, zorder=4, clip_on=False)

    # axes cosmetics: broken y-axis, categorical x
    up.spines["bottom"].set_visible(False)
    up.tick_params(axis="x", bottom=False, labelbottom=False)
    lo.spines["bottom"].set_bounds(XLIM[0] + 0.06, XLIM[1] - 0.06)
    lo.set_xticks([0, 1])
    lo.set_xticklabels([s for s, _, _ in SPLITS], fontsize=7.5, color=INK)
    lo.tick_params(axis="x", length=2.5, pad=2)
    for sp, x, sub in SPLITS:
        lo.annotate(sub, xy=(x, 0), xycoords=("data", "axes fraction"), xytext=(0, -13.5), textcoords="offset points",
                    ha="center", va="top", fontsize=6.5, color=INK2)
    brk = dict(marker=[(-1, -0.55), (1, 0.55)], ms=5, ls="none", color=INK2, mec=INK2, mew=0.6, clip_on=False, zorder=5)
    up.plot([0], [0], transform=up.transAxes, **brk)
    lo.plot([0], [1], transform=lo.transAxes, **brk)

    # y label centred on the two segments
    fig.text((x0 - YLAB) / W, (BOT + (H_LO + GAP + H_UP) / 2) / H, ylab, rotation=90, ha="center", va="center",
             fontsize=7.5, color=INK)
    # panel letter (top-most element)
    fig.text((xp + XL) / W, Y_BASE / H, "ab"[k], fontsize=9, fontweight="bold", ha="left", va="baseline", color=INK)

    # effect-column rows: one per model at its S1 mean, lower rows pushed down if they crowd the row above
    for ax, mods in ((up, ["b2", "cnngru"]), (lo, ["b1"])):
        ys = {mod: ycoord(ax, mean[m, mod, "S1"]) for mod in mods}
        order = sorted(mods, key=lambda q: -ys[q])
        sep = 9.0  # points between baselines
        for i in range(1, len(order)):
            a, b = order[i - 1], order[i]
            if ys[a] - ys[b] < sep:
                mid = (ys[a] + ys[b]) / 2
                ys[a], ys[b] = mid + sep / 2, mid - sep / 2
        rows.update({(k, mod): y for mod, y in ys.items()})

# ---------------------------------------------------------------- effect columns (header at one height in both panels)
Y_HEAD = (max(rows[k, mod] for k in range(len(METRICS)) for mod in ("b2", "cnngru")) + 10.5) / (H * 72)
for k, (m, _) in enumerate(METRICS):
    xp, x0 = xpanel[k]
    xg = x0 + PW + G_AX               # model glyph (centre)
    xn = xg + G_GLYPH                 # model name (left-aligned)
    xd = xn + W_NAME + G_NAME + W_IP  # decimal point of the effect
    xc = xd + W_FP + G_NUM            # opening bracket of the CI
    for mod in MODELS:
        yf = rows[k, mod] / (H * 72)
        ip, fp, ci = EFF_STR[m, mod]
        mk = MARK[mod]
        fig.add_artist(Line2D([xg / W], [yf], transform=fig.transFigure, ls="none", marker=mk,
                              ms=ms_of(mk, MS_MEAN) * 0.9, mfc=FACE[mod], mec=FACE[mod], mew=0))
        fig.text(xn / W, yf, NAME[mod], va="center", ha="left", color=COLR[mod], **NAMEKW)
        fig.text(xd / W, yf, ip, ha="right", va="center", **NUMKW)
        fig.text(xd / W, yf, fp, ha="left", va="center", **NUMKW)
        fig.text(xc / W, yf, ci, ha="left", va="center", **NUMKW)
    fig.text((xd + W_FP) / W, Y_HEAD, f"S0 {MINUS} S1 (pp)", ha="right", va="center", fontsize=6.5, color=MUTED)
    fig.text(xc / W, Y_HEAD, "95% CI", ha="left", va="center", fontsize=6.5, color=MUTED)

# ---------------------------------------------------------------- shared key on the letter row, ~0.15 in right of 'a'
# neutral grey circles: the key explains size and fill only (shape and colour identify the model)
yk = (Y_BASE + 0.716 * 6.5 / 2 / 72) / H   # glyph centres at half the cap height of the 6.5 pt key text
xk = XL + text_w("a", fontsize=9, fontweight="bold") + 0.15
KEY = [(MS_RUN * 1.13, TINT[MUTED], INK2, MEW_RUN, "Seed (S0) or fold (S1)"),
       (MS_MEAN * 1.13, INK2, "white", 0.7, "Mean")]
for ms, mfc, mec, mew, lab in KEY:
    r = ms / 72 / 2
    fig.add_artist(Line2D([(xk + r) / W], [yk], transform=fig.transFigure, ls="none", marker="o", ms=ms, mfc=mfc,
                          mec=mec, mew=mew))
    fig.text((xk + 2 * r + 0.05) / W, Y_BASE / H, lab, ha="left", va="baseline", fontsize=6.5, color=INK2)
    xk += 2 * r + 0.05 + text_w(lab, fontsize=6.5) + 0.16

# ---------------------------------------------------------------- save on the fixed canvas (no tight crop) + checks
fig.canvas.draw()
bb = fig.get_tightbbox(fig.canvas.get_renderer())
marg = dict(left=bb.x0, right=W - bb.x1, bottom=bb.y0, top=H - bb.y1)
print("canvas %.3f x %.3f in; content margins (in): " % (W, H) + ", ".join(f"{a} {b:.3f}" for a, b in marg.items()))
assert min(marg.values()) >= 0, "an artist leaves the canvas"
for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
    fig.savefig(FIG / f"fig3_ranking_inversion.{ext}", bbox_inches=None, **kw)
plt.close(fig)
box_pt = [float(v) for v in re.search(rb"MediaBox\s*\[([^\]]*)\]",
                                     (FIG / "fig3_ranking_inversion.pdf").read_bytes()).group(1).split()]
w_in = (box_pt[2] - box_pt[0]) / 72
print(f"saved fig3_ranking_inversion: PDF {w_in:.3f} x {(box_pt[3] - box_pt[1]) / 72:.3f} in")
assert abs(w_in - FULL) <= 0.03, w_in

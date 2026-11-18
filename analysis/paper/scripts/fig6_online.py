"""Fig 6: within-cycle alarm timing (B2 x pred, alpha=0.10).
Sources: reports/faz_a3/03_online_alarms.csv (frac_lit, alarm_lit; (cycle, fold) pairs pooled, alarmed pairs only),
reports/faz_a3/03_online_summary.csv (medians / IQR, alarm rates printed in the legend/text),
reports/faz_a3/03_online_check.csv (offline == online identity check, 10/10 cell x fold).
Style v2 + set conventions (style.py): group order ORDER_ANOM (Seq 1–3–5, Seq 1–4–5, Fast, Slow); ANOM hues
(sequence = violet, tempo = amber); within a family the first type is solid with a filled circle and the second is
dashed (LS_ANOM dash) with an open circle, so no series is told apart by colour alone; anomaly-type points are circles
only. Cycle end = threshold/boundary line (REF_THRESH). Rate column = RATE_LABEL. Fixed inch-based layout fitted to
FULL so the PDF is included at scale 1.00."""
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from style import *

al = pd.read_csv(REP / "faz_a3/03_online_alarms.csv")
sm = pd.read_csv(REP / "faz_a3/03_online_summary.csv").set_index(["cell", "atype"])
chk = pd.read_csv(REP / "faz_a3/03_online_check.csv")
print("identity check: all keys equal:", bool(chk.keys_equal.all()), "n_diff total:", int(chk.n_diff.sum()), "rows", len(chk))
a = al[(al.state_source == "b2") & (al.segmentation == "pred") & (al.alarm_lit == True)].copy()

SOLID, DASH = "-", LS_ANOM["sequence"]          # second type of each family: the set's (4, 2) dash
KEY = {"Seq 1–3–5": "seq135", "Seq 1–4–5": "seq145", "Fast": "fast", "Slow": "slow"}
FAM = {"seq135": "sequence", "seq145": "sequence", "fast": "tempo", "slow": "tempo"}
SECOND = {"seq145", "slow"}                     # second type within its family: dashed line, open circle
# (key, label, colour, line style in a, open marker?) in the set-wide group order
types = [(KEY[lab], lab, ANOM[FAM[KEY[lab]]], DASH if KEY[lab] in SECOND else SOLID, KEY[lab] in SECOND)
         for lab in ORDER_ANOM]
LW = LW_ANOM["tempo"]
XMAX = 1.10
MS, MEW_OPEN = 4.6, 1.0


def xticks(ax):
    ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])


def circle(open_, c):
    return dict(marker="o", ms=MS, mfc=SURF if open_ else c, mec=c, mew=MEW_OPEN if open_ else 0.0)


CYC = dict(REF_THRESH, zorder=1)                # cycle end is a boundary -> dashed reference line

# ------------------------------------------------------------------ fixed canvas, inch-based axes (fitted below)
H = 2.35
Y0, AH = 0.42, 1.68            # axes bottom and height (in)
ROW = Y0 + AH + 0.075          # baseline of the panel-letter row (in)
fig = plt.figure(figsize=(FULL, H))
ax_a = fig.add_axes([0.1, Y0 / H, 0.5, AH / H])
ax_b = fig.add_axes([0.7, Y0 / H, 0.25, AH / H])
yrow = ROW / H

# ------------------------------------------------------------------ panel a: CDF of the alarm fraction
ax = ax_a
stats = {}
for i, (t, lab, c, ls, op) in enumerate(types):
    f = np.sort(a[a.atype == t].frac_lit.dropna().values)
    y = np.arange(1, len(f) + 1) / len(f)
    stats[t] = (len(f), np.median(f))
    ax.step(np.r_[0, f], np.r_[0, y], where="post", color=c, lw=LW, ls=ls, zorder=3 + i,
            solid_joinstyle="miter", dash_capstyle="butt")
ax.axvline(1.0, **CYC)
ax.text(1.0, yrow, "cycle end", transform=blended_transform_factory(ax.transData, fig.transFigure),
        ha="center", va="baseline", fontsize=6.8, color=INK2, clip_on=False)
ax.set_xlim(-0.015, XMAX); ax.set_ylim(-0.02, 1.02)
xticks(ax)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
ygrid(ax)
ax.set_xlabel("Fraction of the cycle elapsed when the alarm is raised")
ax.set_ylabel("Cumulative share of alarms")

# legend as a small table (line key | type | n | median) in the empty area right of the curves. No ground patch:
# the rows are placed in data units so the 0.75 gridline falls between the header and the first row and the 0.50
# gridline between the sequence and the tempo rows; the hairlines run behind the gaps, never through the text.
trt = blended_transform_factory(ax.transAxes, ax.transData)
x_key, x_name, x_n, x_med = 0.385, 0.475, 0.735, 0.88   # axes coordinates; median column ends left of x = 1.0
y_hdr = 0.776                                          # header baseline just above the 0.75 gridline (data units)
y_rows = [0.65, 0.54, 0.43, 0.32]                        # row baselines: sequence rows above, tempo rows below 0.50
hdr = dict(fontsize=6.8, color=INK2, va="baseline", transform=trt, zorder=10)
ax.text(x_n, y_hdr, "$n$", ha="right", **hdr)
ax.text(x_med, y_hdr, "Median", ha="right", **hdr)
for i, (t, lab, c, ls, op) in enumerate(types):
    yy = y_rows[i]
    ym = yy + 0.024  # visual centre of a 7 pt text line above its baseline (data units)
    # line-only key (panel a has no markers; a marker in the key would hide the dash of the second type)
    ax.add_line(Line2D([x_key, x_key + 0.075], [ym, ym], transform=trt, color=c, lw=LW, ls=ls,
                       dash_capstyle="butt", zorder=10))
    n_, med_ = stats[t]
    row = dict(fontsize=7, color=INK, va="baseline", transform=trt, zorder=10)
    ax.text(x_name, yy, lab, ha="left", **row)
    ax.text(x_n, yy, f"{n_}", ha="right", **row)
    ax.text(x_med, yy, f"{med_:.2f}", ha="right", **row)

# ------------------------------------------------------------------ panel b: median [IQR] + alarm rate
ax = ax_b
ypos = np.arange(len(types))[::-1].astype(float)
for yp, (t, lab, c, ls, op) in zip(ypos, types):
    r = sm.loc[("b2xpred", t)]
    ax.errorbar(r.frac_med, yp, xerr=[[r.frac_med - r.frac_q25], [r.frac_q75 - r.frac_med]], fmt="none", ecolor=c,
                elinewidth=LW, capsize=2.2, capthick=LW, zorder=3)
    ax.plot(r.frac_med, yp, ls="none", zorder=5, **circle(op, c))
ax.axvline(1.0, **CYC)
ax.axhline(1.5, color=GRID, lw=0.6, zorder=0)  # sequence | tempo
ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
ax.set_xlim(-0.015, XMAX); ax.set_ylim(-0.6, 3.6)
xticks(ax)
ax.set_yticks(ypos)
ax.set_yticklabels([t[1] for t in types])
ax.tick_params(axis="y", length=0, pad=4)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Fraction of the cycle elapsed (median, IQR)")
# alarm rate (= detection rate for these anomalous groups) as a right-hand column; header on the panel-letter row
trb = blended_transform_factory(ax.transAxes, ax.transData)
X_RATE = 1.13
ax.text(X_RATE, yrow, RATE_LABEL, transform=blended_transform_factory(ax.transAxes, fig.transFigure), ha="right",
        va="baseline", fontsize=6.8, color=INK2, clip_on=False)
for yp, (t, lab, c, ls, op) in zip(ypos, types):
    ax.text(X_RATE, yp, format(sm.loc[("b2xpred", t)].alarm_rate * 100, ".0f") + "%", transform=trb,
            ha="right", va="center", fontsize=7, color=INK, clip_on=False)

# ------------------------------------------------------------------ fit the two panels to FULL (scale 1.00 in main.tex)
PAD = 0.03      # save() pads the tight box by 0.03 in on each side
GAP = 0.36      # whitespace between panel a's content and panel b's tick labels (in)
RATIO = 1.42    # width of panel a relative to panel b


def fit(fig, ax_a, ax_b):
    for _ in range(3):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        d = fig.dpi
        ta, tb = ax_a.get_tightbbox(r), ax_b.get_tightbbox(r)
        pa, pb = ax_a.get_window_extent(r), ax_b.get_window_extent(r)
        lo_a, ro_a = (pa.x0 - ta.x0) / d, (ta.x1 - pa.x1) / d
        lo_b, ro_b = (pb.x0 - tb.x0) / d, (tb.x1 - pb.x1) / d
        w = FULL - 2 * PAD - lo_a - max(ro_a, 0) - GAP - lo_b - ro_b
        wa, wb = w * RATIO / (1 + RATIO), w / (1 + RATIO)
        xa = PAD + lo_a
        xb = xa + wa + max(ro_a, 0) + GAP + lo_b
        ax_a.set_position([xa / FULL, Y0 / H, wa / FULL, AH / H])
        ax_b.set_position([xb / FULL, Y0 / H, wb / FULL, AH / H])


fit(fig, ax_a, ax_b)


# ------------------------------------------------------------------ panel letters on one row (top-most), flush left
def letters(fig, axes, labels):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, lab in zip(axes, labels):
        x0 = inv.transform(ax.yaxis.get_tightbbox(rend).p0)[0]
        fig.text(x0, yrow, lab, fontsize=9, fontweight="bold", ha="left", va="baseline", color=INK)


letters(fig, [ax_a, ax_b], ["a", "b"])
fig.canvas.draw()
rend = fig.canvas.get_renderer()
# check: no gridline of panel a passes through a table glyph box (clearance in pt; must be > 0)
tab_txt = [t for t in ax_a.texts if t.get_transform() is trt]
grid_px = [ax_a.transData.transform((0, g))[1] for g in (0.25, 0.5, 0.75)]
clear = min(min(abs(g - e.y0), abs(g - e.y1)) * 72 / fig.dpi if not (e.y0 < g < e.y1) else -1.0
            for e in (t.get_window_extent(rend) for t in tab_txt) for g in grid_px)
print(f"table text vs gridlines: min clearance {clear:.1f} pt")
bb = fig.get_tightbbox(rend)
print(f"tight content width {bb.width:.3f} in -> saved width {bb.width + 2 * PAD:.3f} in (FULL = {FULL})")
save(fig, "fig6_online_alarm_timing")
for t, _, _, _, _ in types:
    r = sm.loc[("b2xpred", t)]
    print(t, round(r.frac_med, 3), round(r.frac_q25, 3), round(r.frac_q75, 3), round(r.alarm_rate, 3), r.n_alarm,
          "| legend n, median:", stats[t][0], f"{stats[t][1]:.2f}", "| alarm rate:", format(r.alarm_rate * 100, ".0f") + "%")

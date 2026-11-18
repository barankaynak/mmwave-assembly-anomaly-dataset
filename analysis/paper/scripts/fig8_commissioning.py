"""Fig 8: commissioning curves. x-axis in CYCLES (minute conversions are not yet unified, OUTLINE 9.1).
Sources: reports/faz_a2/04_comm_anomaly_summary.csv (D3 AUROC vs number of normal training cycles; 50 resamples of the 51 s02 cycles; 20 calibration cycles extra),
reports/faz_a2/04_comm_state_summary.csv (B2/B1 state macro-F1 vs labelled cycles per session; 10 resamples),
reports/faz_a/04_anomaly_table.csv (Phase-A reference: B2 x pred D3 AUROC 0.924 at ~20 training cycles).
Style v2 (2026-11-05): same data and computations as the archived version; only the visual encoding changed
(s.d. drawn as bands instead of error bars, ANOM / MODEL colours, distinct markers and line styles, direct labels).
Critique round (2026-11-05): set-level conventions from style.py (circles + LS_ANOM/LW_ANOM for anomaly types in a,
MARKER/MODEL_FACE for B2/B1 in b, REF_TARGET for the 0.90 and all-data lines), the cycle-grouped star moved just right of
n = 20 with a hairline and a direct label, panel b legend replaced by direct labels. BLUE (B2, b) and VIOLET (sequence, a)
share this figure only because they sit in separate panels and every series is direct-labelled and has its own marker
or line style, so identity never depends on hue alone."""
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
from style import *

an = pd.read_csv(REP / "faz_a2/04_comm_anomaly_summary.csv")
st = pd.read_csv(REP / "faz_a2/04_comm_state_summary.csv")
ref = pd.read_csv(REP / "faz_a/04_anomaly_table.csv")
ref = ref[(ref.state_source == "b2") & (ref.segmentation == "pred") & (ref.variant == "D3") & (ref.atype == "tum anomali")].auroc_mean.iloc[0]

WHITE_EDGE = dict(mec=SURF, mew=0.6)
BAND_ALL = TINT[MUTED]            # light neutral band behind the INK 'all anomalies' curve
BAND_ALPHA = 0.7                  # bands slightly lighter than the stock tint so the curves stay in front
YLAB_X = -0.135                   # common y-label offset (axes fraction; both axes have equal width)
LET_X = -0.175                    # common panel-letter offset
MS_ANOM = {"all": 3.9, "sequence": 3.6, "tempo": 3.6}   # anomaly-type points: circles only
MS_MOD = {"s": 3.7, "v": 4.3}                            # square / down-triangle, matched in visual area
LAB = dict(fontsize=7, ha="left", va="center")          # direct labels at line ends


def hgrid(ax, ys, x0, x1):
    """Hairline y-grid limited to the data span, so it never runs behind the direct labels."""
    ax.hlines(ys, x0, x1, color=GRID, lw=0.5, zorder=0)

# fixed canvas; W_CANVAS is chosen so that the tight bounding box written by save() is FULL (6.69 in = \textwidth)
W_CANVAS = 6.846
fig, axes = plt.subplots(1, 2, figsize=(W_CANVAS, 2.35), gridspec_kw=dict(wspace=0.30, left=0.075, right=0.975, bottom=0.17, top=0.94))

# ---------------------------------------------------------------- (a) detector: D3 AUROC vs normal training cycles
ax = axes[0]
a = an[an.cell == "B2 x pred"]
A_X0, A_X1 = 3.2, 20.8
hgrid(ax, [0.80, 0.85, 0.95, 1.00], A_X0, A_X1)
ax.hlines(0.9, A_X0, A_X1, zorder=2, **REF_TARGET)

g = a[a.group == "tum anomali"].sort_values("n")
ax.fill_between(g.n, g.auroc_mean - g.auroc_sd, g.auroc_mean + g.auroc_sd, color=BAND_ALL, alpha=BAND_ALPHA, lw=0, zorder=1)
ax.plot(g.n, g.auroc_mean, color=ANOM["all"], lw=LW_ANOM["all"], ls=LS_ANOM["all"], marker="o", ms=MS_ANOM["all"],
        zorder=5, **WHITE_EDGE)

STAR_X = 21.1                     # cycle-grouped protocol drawn just right of n = 20 (caption: 'about 20 training cycles')
LAB_X = 21.95                     # x of the Tempo / Sequence direct labels (clear of the star)
series = [("seq(135+145)", "Sequence", "sequence"), ("tempo(fast+slow)", "Tempo", "tempo")]
for grp, lab, key in series:
    g = a[a.group == grp].sort_values("n")
    ax.plot(g.n, g.auroc_mean, color=ANOM[key], lw=LW_ANOM[key], ls=LS_ANOM[key], marker="o", ms=MS_ANOM[key],
            zorder=4, **WHITE_EDGE)
    last = g.iloc[-1]
    ax.text(LAB_X, last.auroc_mean, lab, color=ANOM[key], **LAB)

# primary (cycle-grouped) protocol: star on top, offset right of the n = 20 point (0.928) and tied to it by a hairline
ax.plot([20, STAR_X], [ref, ref], color=RULE, lw=0.4, zorder=3, solid_capstyle="butt")
ax.plot([STAR_X], [ref], ls="none", marker="*", ms=7.5, mfc=INK, mec=SURF, mew=0.6, zorder=6)
# direct label in the free space under the band, joined to the star by a vertical leader (no legend entry)
STAR_LAB_Y = 0.826
ax.plot([STAR_X, STAR_X], [ref - 0.0085, STAR_LAB_Y + 0.0085], color=RULE, lw=0.4, zorder=3, solid_capstyle="butt")
ax.text(STAR_X + 0.45, STAR_LAB_Y, f"{ref:.3f}, cycle-grouped protocol", fontsize=7, color=INK2, ha="right", va="center",
        zorder=6)

ax.set_xlim(3.2, 25.2)
ax.xaxis.set_major_locator(FixedLocator([5, 10, 15, 20]))
ax.set_ylim(0.755, 1.005)
ax.yaxis.set_major_locator(FixedLocator([0.80, 0.85, 0.90, 0.95, 1.00]))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax.spines["bottom"].set_bounds(A_X0, A_X1)
ax.set_xlabel("Normal training cycles (plus 20 calibration cycles)")
ax.set_ylabel("D3 AUROC (B2, predicted cycles)")
ax.yaxis.set_label_coords(YLAB_X, 0.5)

h_all = (Patch(fc=BAND_ALL, alpha=BAND_ALPHA, lw=0),
         Line2D([], [], color=ANOM["all"], lw=LW_ANOM["all"], marker="o", ms=MS_ANOM["all"], **WHITE_EDGE))
ax.legend([h_all], ["All anomalies (± 1 s.d.)"], loc="lower right", bbox_to_anchor=(1.0, 0.0), handlelength=2.0,
          borderaxespad=0.2)
panel(ax, "a", x=LET_X)

# ---------------------------------------------------------------- (b) state estimator: macro-F1 vs labelled cycles
ax = axes[1]
B_X1 = 20
hgrid(ax, [0.5, 0.6, 0.7, 0.8, 0.9], 0.8, B_X1)
for mod, lab in [("b2", "B2"), ("b1", "B1")]:
    c, mk = MODEL[lab], MARKER[lab]
    g = st[(st.model == mod) & (st.k != "all")].copy()
    g["kk"] = g.k.astype(float)
    g = g.sort_values("kk")
    full = st[(st.model == mod) & (st.k == "all")].mf1_mean.iloc[0]
    ax.fill_between(g.kk, g.mf1_mean - g.mf1_sd_reps, g.mf1_mean + g.mf1_sd_reps, color=TINT[c], alpha=BAND_ALPHA, lw=0, zorder=1)
    ax.plot([0.8, B_X1], [full, full], zorder=2, **REF_TARGET)
    ax.text(0.86, full + 0.007, f"All data, {full:.3f}", color=INK2, fontsize=7, ha="left", va="bottom")
    ax.plot(g.kk, g.mf1_mean, color=c, lw=1.5, marker=mk, ms=MS_MOD[mk], mfc=MODEL_FACE[lab], zorder=4, **WHITE_EDGE)
    last = g.iloc[-1]
    ax.text(last.kk * 1.09, last.mf1_mean, lab, color=c, **LAB)
ax.set_xscale("log")
ax.set_xlim(0.8, 25.5)
ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(["1", "2", "5", "10", "20"])
ax.spines["bottom"].set_bounds(0.8, B_X1)
ax.set_ylim(0.45, 0.90)
ax.yaxis.set_major_locator(FixedLocator([0.5, 0.6, 0.7, 0.8, 0.9]))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax.set_xlabel("Labelled cycles per session")
ax.set_ylabel("State macro-F1 (cycle-grouped folds)")
ax.yaxis.set_label_coords(YLAB_X, 0.5)
panel(ax, "b", x=LET_X)

save(fig, "fig8_commissioning")
print(a[a.group == "tum anomali"][["n", "auroc_mean", "auroc_sd"]].to_string()); print(st[st.model == "b2"][["k", "mf1_mean", "mf1_sd_reps"]].to_string())

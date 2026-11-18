"""Fig 4: D3 cycle scores (B2 x pred, role=test) per anomaly type, alpha=0.10 operating point.
Sources: reports/faz_a/04_cycle_scores.csv.gz (p3 -> score 1-p3); AUROC / alarm-rate annotations = values from
reports/faz_a/04_anomaly_table.csv (fold means; NOT recomputed from the pooled scores).
Panel b is a pooled descriptive ROC (anomaly cycles are repeated once per fold); it is not the verified AUROC.
Style v2 (2026-11-05) + set-level conventions of style.py (critique round): ANOM colours (normal MUTED, sequence
VIOLET, tempo AMBER, all INK) with LS_ANOM / LW_ANOM (tempo solid, sequence dashed, all anomalies solid INK on top);
threshold = REF_THRESH; group labels = ORDER_ANOM; header row name = RATE_LABEL; density-scaled jitter (x only,
y values untouched); medians drawn under the points with a white halo and protruding past every strip, so no cycle
is hidden. Fixed canvas FULL x H (inch-based axes), saved without tight trimming so the PDF is exactly \\textwidth wide
and is included at scale 1.00."""
import numpy as np, pandas as pd
from style import *
from sklearn.metrics import roc_curve
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import matplotlib.patheffects as pe

d = pd.read_csv(REP / "faz_a/04_cycle_scores.csv.gz")
d = d[(d.state_source == "b2") & (d.segmentation == "pred") & (d.role == "test")].copy()
d["score"] = 1 - d.p3
tab = pd.read_csv(REP / "faz_a/04_anomaly_table.csv")
tab = tab[(tab.state_source == "b2") & (tab.segmentation == "pred") & (tab.variant == "D3")].set_index("atype")
types = ["seq135", "seq145", "fast", "slow"]
lab = {"normal": "Normal (s02)", **dict(zip(types, ORDER_ANOM))}       # Seq 1–3–5, Seq 1–4–5, Fast, Slow
colr = {"normal": ANOM["normal"], "seq135": ANOM["sequence"], "seq145": ANOM["sequence"], "fast": ANOM["tempo"],
        "slow": ANOM["tempo"]}

# self-checks against verified numbers
nrm = d[d.atype == "normal"]
far = (nrm.p3 <= 0.10 + 1e-9).mean()
print("normal n", len(nrm), "FAR@0.10", far)
for t in types:
    print(t, "det", (d[d.atype == t].p3 <= 0.10 + 1e-9).mean(), "table", tab.loc[t, "det_a10"])
assert abs(far - tab.loc["tum anomali", "far_a10"]) < 1e-9
print("min attainable p3 in test rows:", d.p3.min())

THR = 0.90                      # alarm threshold at alpha = 0.10 (score >= 0.90)
HW = 0.40                       # half-width of the median segments (x units); widest strip half-width is 0.28
YLO, YHI = -0.035, 1.02         # shared y range of both panels

# ------------------------------------------------------------------ fixed canvas, inch-based layout
W, H = FULL, 2.445
LA, BA, WA, HA = 0.565, 0.36, 3.46, 1.62           # panel a axes (inches)
HDR = (0.075, 0.215)                                # header rows above panel a (baseline offsets, inches)
LET = HA + 0.335                                    # panel-letter baseline above the axes bottom (inches)
WB = 1.64                                           # panel b axes width (inches)
LB = W - 0.135 - WB                                 # right end leaves room for the '1.00' tick label overhang
HB = HDR[1] + 0.105 + HA                            # panel b: from panel a's bottom to the top of the header strip
fig = plt.figure(figsize=(W, H))
axA = fig.add_axes([LA / W, BA / H, WA / W, HA / H])
axB = fig.add_axes([LB / W, BA / H, WB / W, HB / H])

# ------------------------------------------------------------------ (a) score strip plot
ax = axA
rng = np.random.default_rng(5)
order = ["normal"] + types
win = 0.012                     # y window for the local count that sets the jitter width
dens = {t: np.array([(np.abs(d[d.atype == t].score.values - v) <= win).sum() for v in d[d.atype == t].score.values])
        for t in order}
cmax = max(c.max() for c in dens.values())
halo = [pe.Stroke(linewidth=2.6, foreground=SURF), pe.Normal()]
ax.axhspan(THR, YHI, color=PANEL_BG, lw=0, zorder=0)              # alarm region (score >= 0.90)
ax.axhline(THR, zorder=1, **REF_THRESH)
for i, t in enumerate(order):
    s = d[d.atype == t].score.values
    w = 0.04 + 0.24 * np.sqrt(dens[t] / cmax)                     # sina-style: wider where cycles pile up
    x = i + rng.uniform(-1, 1, len(s)) * w
    ax.scatter(x, s, s=7 if t == "normal" else 6, color=colr[t], alpha=1.0, edgecolors=SURF, linewidths=0.15,
               zorder=3)
    ax.hlines(np.median(s), i - HW, i + HW, color=INK, lw=1.1, zorder=2, capstyle="butt", path_effects=halo)
ax.text(len(order) - 0.5 + 0.05, THR, r"$\alpha = 0.10$", ha="left", va="center", fontsize=6.5, color=INK2,
        clip_on=False)
ax.set_xlim(-0.5, len(order) - 0.5)
ax.set_ylim(YLO, YHI)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"{lab[t]}\n" for t in order], fontsize=7)
ax.tick_params(axis="x", length=0, pad=4)
for i, t in enumerate(order):                                     # n under each group label, muted
    ax.annotate(f"n = {(d.atype == t).sum()}", (i, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -14.5),
                textcoords="offset points", ha="center", va="top", fontsize=6.5, color=MUTED)
ax.set_ylabel(r"D3 score ($1-p_3$)")

# header strip: fold-mean AUROC and alarm rate (detection rate; false-alarm rate for the normal group)
hx = blended_transform_factory(ax.transData, ax.transAxes)
rows = {"AUROC": 1 + HDR[1] / HA, RATE_LABEL: 1 + HDR[0] / HA}
for name, yy in rows.items():
    ax.text(-0.5, yy, name, transform=hx, ha="right", va="baseline", fontsize=6.5, color=INK2, clip_on=False)
printed = []
for i, t in enumerate(types, start=1):
    printed += [f"{tab.loc[t, 'auroc_mean']:.2f}", f"{tab.loc[t, 'det_a10'] * 100:.0f}%"]
    ax.text(i, rows["AUROC"], printed[-2], transform=hx, ha="center", va="baseline", fontsize=7, color=INK,
            clip_on=False)
    ax.text(i, rows[RATE_LABEL], printed[-1], transform=hx, ha="center", va="baseline", fontsize=7, color=INK,
            clip_on=False)
printed.append(f"{far * 100:.1f}%")
ax.text(0, rows[RATE_LABEL], printed[-1], transform=hx, ha="center", va="baseline", fontsize=7, color=INK,
        clip_on=False)
ax.plot([-0.5, len(order) - 0.5], [1 + 0.035 / HA, 1 + 0.035 / HA], transform=hx, color=RULE, lw=0.5, clip_on=False)
print("printed header values:", printed)
assert printed == ["0.88", "25%", "0.92", "48%", "0.95", "92%", "0.96", "93%", "7.8%"], printed
assert [int((d.atype == t).sum()) for t in order] == [51, 280, 240, 240, 255]

# ------------------------------------------------------------------ (b) pooled ROC
ax = axB
ax.plot([0, 1], [0, 1], color=RULE, lw=0.6, zorder=1)               # chance diagonal
curves = [(["seq135", "seq145"], "Sequence (1–3–5, 1–4–5)", "sequence", 2),
          (["fast", "slow"], "Tempo (Fast, Slow)", "tempo", 3),
          (types, "All anomalies", "all", 4)]                          # all anomalies drawn on top
handles = []
for grp, nm, key, z in curves:
    sub = d[d.atype.isin(grp + ["normal"])]
    y = (sub.atype != "normal").astype(int)
    fpr, tpr, _ = roc_curve(y, sub.score)
    ax.step(fpr, tpr, where="post", color=ANOM[key], lw=LW_ANOM[key], ls=LS_ANOM[key], zorder=z,
            solid_joinstyle="miter")
    handles.append(Line2D([], [], color=ANOM[key], lw=LW_ANOM[key], ls=LS_ANOM[key], label=nm))
det_all = (d[d.atype.isin(types)].p3 <= 0.10 + 1e-9).mean()
print("operating point: FAR", far, "detection", det_all)
ax.scatter([far], [det_all], marker="o", s=26, facecolor=SURF, edgecolors=INK, linewidths=0.9, zorder=6)
# direct label for the operating point, led out into the empty region above the chance line
ax.annotate(r"$\alpha = 0.10$", (far, det_all), xytext=(0.30, 0.56), textcoords="data", ha="left", va="center",
            fontsize=6.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=1.5, shrinkB=3.2))
ax.set_xlabel("False-alarm rate"); ax.set_ylabel("Detection rate")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
ax.spines["left"].set_bounds(0, 1); ax.spines["bottom"].set_bounds(0, 1)
ax.legend(handles=handles, loc="lower right", fontsize=6.5, handlelength=2.4, borderaxespad=0.45, labelspacing=0.35,
          frameon=False, borderpad=0.25)

# ------------------------------------------------------------------ panel letters: top-most, level with each other
ylet = (BA + LET) / H
fig.text(0.03 / W, ylet, "a", fontsize=9, fontweight="bold", va="baseline", ha="left", color=INK)
fig.canvas.draw()
r = fig.canvas.get_renderer()
ylab_x0 = axB.yaxis.label.get_window_extent(r).x0 / fig.dpi           # left edge of 'Detection rate' (inches)
fig.text((ylab_x0 - 0.03) / W, ylet, "b", fontsize=9, fontweight="bold", va="baseline", ha="left", color=INK)

# ------------------------------------------------------------------ layout checks, then save at exactly W x H
fig.canvas.draw()
bb = fig.get_tightbbox(r)
print(f"content bbox (in): x {bb.x0:.3f}-{bb.x1:.3f} of {W}, y {bb.y0:.3f}-{bb.y1:.3f} of {H}")
assert 0.0 <= bb.x0 and bb.x1 <= W and 0.0 <= bb.y0 and bb.y1 <= H, "content outside the canvas"
texts = [t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
tops = {t.get_text(): t.get_window_extent(r).y1 / fig.dpi for t in texts}
others_top = max(v for k, v in tops.items() if k not in ("a", "b"))
print(f"letter baseline {ylet * H:.3f} in; highest other text top {others_top:.3f} in")
assert ylet * H > others_top, "panel letters must be the top-most elements"
print("smallest font (pt):", min(t.get_fontsize() for t in texts))
fig.savefig(FIG / "fig4_d3_scores.pdf", bbox_inches=None)
fig.savefig(FIG / "fig4_d3_scores.png", dpi=300, bbox_inches=None)
plt.close(fig)
print("saved fig4_d3_scores", W, "x", H, "in")

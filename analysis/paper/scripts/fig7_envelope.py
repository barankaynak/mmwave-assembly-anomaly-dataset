"""Fig 7: operating envelope (B2). Descriptive re-plot; no fits, no d50, no declared limit.
Sources: reports/faz_a3/02_envelope_detection.csv (per session x touch-duration bin, Wilson 95% CIs; definition D-ham = raw label),
reports/faz_a3/02_envelope_spatial.csv (14 neighbouring box pairs: spacing in cm vs symmetric confusion 'sim_hata').

Style v2 (2026-11-05): the whole figure is B2, so every data mark is MODEL["B2"] (blue); sessions are told apart by
marker shape only (one shared key for both panels). Rates and confusions are drawn as percent (value x 100, same
computation). Panel b uses a broken y axis (same units per inch in both segments) so that the four pairs closer than
11.8 cm can be read next to the single 14.7 % outlier.

Critique round (2026-11-05): reference lines use the shared specs (95 % = REF_TARGET dotted, 11.8 cm = REF_THRESH
dashed); panel a separates bins with hairlines instead of alternating fills, so PANEL_BG only marks the
sub-resolution region in b; coincident zero-confusion pairs in b (s02, s05, s06 at about 14 and 25 cm) are kept at
their true value y = 0 and stacked downward by a display offset in points (confusion cannot be negative, so the strip
below the 0 line carries no value); the canvas is sized so the saved PDF is FULL wide (included at scale 1.00).
"""
import numpy as np, pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory, offset_copy
from style import *

B2 = MODEL["B2"]

det = pd.read_csv(REP / "faz_a3/02_envelope_detection.csv")
det = det[(det.model == "B2") & (det.tanim == "D-ham")]
sp = pd.read_csv(REP / "faz_a3/02_envelope_spatial.csv")
bins = ["<0.2", "0.2-0.3", "0.3-0.5", "0.5-1.0", ">1.0"]
blab = ["< 0.2", "0.2–0.3", "0.3–0.5", "0.5–1.0", "> 1.0"]
sess = ["s01", "s02", "s03", "s04", "s05", "s06"]
smk = dict(zip(sess, ["o", "s", "^", "D", "v", "P"]))
# optically equal marker sizes (squares and diamonds look larger than circles and triangles at the same ms)
sms = dict(zip(sess, [4.3, 3.8, 4.7, 3.7, 4.7, 4.6]))

# zero-confusion pairs whose spacings lie within 1 cm of each other overprint; stack them (session order, top to bottom)
STEP = 6.0                                   # pt between stacked markers (markers are about 4.5 pt across)
zero = sp[sp.sim_hata == 0].sort_values("aralik_cm")
stack = {}                                   # (oturum, cift) -> stack position k (0 = on the 0 line)
cl = []
for _, r in zero.iterrows():
    if cl and r.aralik_cm - cl[-1].aralik_cm > 1.0:
        for k, q in enumerate(sorted(cl, key=lambda q: q.oturum)):
            stack[(q.oturum, q.cift)] = k
        cl = []
    cl.append(r)
for k, q in enumerate(sorted(cl, key=lambda q: q.oturum)):
    stack[(q.oturum, q.cift)] = k
KMAX = max(stack.values())

H = 2.62
RES = 11.8
top_lim, bot_lim = (14.35, 15.05), (-0.43, 1.60)          # percent; equal units per inch in both segments
# bot_lim[0] must leave at least KMAX * STEP + 3 pt below 0, or the lowest stacked marker is clipped


def build(W):
    fig = plt.figure(figsize=(W, H))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.42, 1], wspace=0.25, left=0.075, right=0.995, top=0.86,
                  bottom=0.21)

    # -------------------------------------------------------------- (a) detection by touch duration
    ax = fig.add_subplot(gs[0])
    ygrid(ax)
    for j in range(4):                                     # hairline separators between duration bins
        ax.axvline(j + 0.5, color=GRID, lw=0.4, zorder=0.3)
    ntot = {}
    dx = 0.13
    for i, s in enumerate(sess):
        g = det[det.session == s].set_index("bin")
        off = (i - 2.5) * dx
        for j, b in enumerate(bins):
            if b not in g.index or g.loc[b].n_eval == 0:
                continue
            r = g.loc[b]
            ntot[b] = ntot.get(b, 0) + int(r.n_eval)
            x, few = j + off, r.n_eval < 5
            ax.vlines(x, 100 * r.wilson_lo, 100 * r.wilson_hi, color=B2, lw=0.8, alpha=(0.45 if few else 1.0), zorder=2)
            ax.plot(x, 100 * r.rate, ls="none", marker=smk[s], ms=sms[s], zorder=3,
                    mfc=(SURF if few else B2), mec=(B2 if few else SURF), mew=(0.8 if few else 0.45))
    ax.axhline(95, zorder=1, **REF_TARGET)
    ax.text(-0.46, 93.4, "95%", fontsize=6.5, color=INK2, va="top", ha="left")
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-4, 104)
    ax.set_yticks(range(0, 101, 20))
    ax.spines["left"].set_bounds(0, 100)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(range(5))
    ax.set_xticklabels(blab)
    ax.tick_params(axis="x", length=0, pad=3)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for j, b in enumerate(bins):
        ax.annotate(f"N = {ntot.get(b, 0)}", xy=(j, 0), xycoords=tr, xytext=(0, -13.5), textcoords="offset points",
                    ha="center", va="top", fontsize=6.5, color=INK2)
    ax.set_xlabel("Touch duration (s; 30 Hz ToF reference)", labelpad=13)
    ax.set_ylabel("Touches detected by B2 (%)")

    # -------------------------------------------------------------- (b) pair confusion vs spacing, broken y axis
    gsb = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.10,
                                  height_ratios=[top_lim[1] - top_lim[0], bot_lim[1] - bot_lim[0]])
    bt = fig.add_subplot(gsb[0])
    bb = fig.add_subplot(gsb[1], sharex=bt)
    for a in (bt, bb):
        a.axvspan(5.0, RES, color=PANEL_BG, lw=0, zorder=0.2)
        ygrid(a)
        a.axvline(RES, zorder=1, **REF_THRESH)
    for _, r in sp.iterrows():
        mk = dict(ls="none", marker=smk[r.oturum], ms=sms[r.oturum] + 0.3, mfc=B2, mec=SURF, mew=0.35, zorder=3)
        k = stack.get((r.oturum, r.cift))
        if k is None:                                      # non-zero (or isolated zero): plotted in both segments
            for a in (bt, bb):
                a.plot(r.aralik_cm, 100 * r.sim_hata, **mk)
        else:                                              # zero: true y = 0, shifted down k * STEP points on screen
            bb.plot(r.aralik_cm, 0.0, transform=offset_copy(bb.transData, fig=fig, x=0, y=-k * STEP, units="points"),
                    **mk)
    bt.set_ylim(*top_lim); bb.set_ylim(*bot_lim)
    bt.set_yticks([14.5, 15.0]); bb.set_yticks([0, 0.5, 1.0, 1.5])
    fmt = FuncFormatter(lambda v, p: "0" if abs(v) < 1e-9 else f"{v:.1f}")
    bt.yaxis.set_major_formatter(fmt); bb.yaxis.set_major_formatter(fmt)
    bb.spines["left"].set_bounds(0, bot_lim[1])           # value axis ends at 0; the strip below holds stacked zeros
    bt.spines["bottom"].set_visible(False)
    bt.tick_params(axis="x", bottom=False, labelbottom=False)
    bb.set_xlim(5.0, 32.0)
    bb.set_xticks([5, 10, 15, 20, 25, 30])
    bb.set_xlabel("Spacing of neighbouring boxes (cm)")
    # break marks on the left spine
    kw = dict(marker=[(-1, -0.55), (1, 0.55)], ms=5, ls="none", color=INK2, mec=INK2, mew=0.7, clip_on=False, zorder=5)
    bt.plot([0], [0], transform=bt.transAxes, **kw)
    bb.plot([0], [1], transform=bb.transAxes, **kw)
    # direct labels: pair (session is given by the marker shape)
    lab_left = {("s01", "3-4")}
    for _, r in sp[sp.sim_hata > 0].iterrows():
        a = bt if 100 * r.sim_hata > top_lim[0] else bb
        left = (r.oturum, r.cift) in lab_left
        a.annotate(r.cift.replace("-", "–"), (r.aralik_cm, 100 * r.sim_hata), xytext=(-4.5 if left else 4.5, 0),
                   textcoords="offset points", ha=("right" if left else "left"), va="center", fontsize=6.5, color=INK)
    bb.text(RES + 0.6, 1.52, "Nominal azimuth\nresolution, 11.8 cm", fontsize=6.5, color=INK2, ha="left", va="top",
            linespacing=1.15)
    # the stacked strip: note in the empty lowest row between the two stacks (no marker there)
    xs = sorted(zero[zero.set_index(["oturum", "cift"]).index.map(lambda t: stack[t] > 0)].aralik_cm)
    bb.annotate("Stacked, all 0%", xy=((xs[0] + xs[-1]) / 2, 0), xytext=(0, -KMAX * STEP),
                textcoords="offset points", ha="center", va="center", fontsize=6.5, color=INK2)

    # y label of b centred over both segments
    pt, pb = bt.get_position(), bb.get_position()
    bb.set_ylabel("Symmetric confusion (%)")
    bb.yaxis.set_label_coords(-0.135, ((pb.y0 + pt.y1) / 2 - pb.y0) / (pb.y1 - pb.y0))

    # -------------------------------------------------------------- panel letters (aligned) and shared session key
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    ytop = pt.y1 + 0.035
    xa = inv.transform((ax.get_tightbbox(rend).x0, 0))[0]
    xb = inv.transform((min(bt.get_tightbbox(rend).x0, bb.get_tightbbox(rend).x0), 0))[0]
    for x, l in ((xa, "a"), (xb, "b")):
        fig.text(x, ytop, l, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)

    hand = [Line2D([], [], ls="none", marker=smk[s], ms=sms[s], mfc=B2, mec=SURF, mew=0.45) for s in sess]
    hand.append(Line2D([], [], ls="none", marker="o", ms=sms["s01"], mfc=SURF, mec=B2, mew=0.8))
    labs = sess + ["Fewer than 5 touches"]
    fig.legend(hand, labs, loc="lower left", bbox_to_anchor=(xa + 0.15 / W, ytop - 0.012), bbox_transform=fig.transFigure,
               ncol=7, fontsize=7, handletextpad=0.25, columnspacing=1.05, borderaxespad=0, borderpad=0,
               handlelength=1.0)
    return fig, ntot


# size the canvas so the tight-cropped output (save() pads 0.03 in on each side) is exactly FULL wide
W = FULL
for _ in range(4):
    fig, ntot = build(W)
    fig.canvas.draw()
    w = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * 0.03
    if abs(w - FULL) < 0.004:
        break
    plt.close(fig)
    W += FULL - w
print(f"canvas {W:.3f} in -> output {w:.3f} in")
print("N per bin:", ntot, "sum", sum(ntot.values()))
save(fig, "fig7_envelope")
print(len(sp), "pairs;", "below 11.8 cm:", int((sp.aralik_cm < 11.8).sum()), "; stacked zeros:",
      {k: v for k, v in stack.items() if v > 0})

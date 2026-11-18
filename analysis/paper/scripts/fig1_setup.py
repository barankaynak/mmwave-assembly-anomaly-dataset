"""Fig 1: set-up photographs (reused from the earlier submission, folder 4) and pipeline schematic (drawn here).
Photos: $PHOTO_DIR/{img_side_view, img_step1, img_mmwave_sensor, img_tof_sensor}.jpg (set-up photographs; not in this package)
(resampled to ~600 dpi at print size for the PDF). Schematic: matplotlib patches; the content follows OUTLINE s5 / faz_a/DESIGN.md.
Style v2 (2026-11-05): photos at equal height with thin gutters; panel e drawn in inch coordinates (true round corners),
run-time path left to right, training-only ToF lane dashed on a muted band, calibration path dotted (ROLE colours).
Critique round (2026-11-05): canvas exactly FULL wide, saved without tight-bbox trimming (included at scale 1.00);
photo (a) callouts: 'radar' = the grey 3D-printed case on the overhead arm (verified on the original photo; pixel
box x 1395-1795, y 245-420 of 2923 x 3831), 'webcam' = the device on the desk tripod (x 795-1075, y 1505-1655), which
recorded a debugging video only; box titles top-anchored (one title baseline for the run-time row); set-style header
brackets (RULE 0.5 pt, 0.04 in ticks, 7 pt INK2 label above); D2 detail 'tempo score'; shorter training lane."""
import os
import numpy as np
import matplotlib.patheffects as pe
from PIL import Image
from style import *

SRC = Path(os.environ.get("PHOTO_DIR", ROOT / "photos"))

# ---------------------------------------------------------------- canvas
M = 0.025                      # inset on every side of the FULL-wide canvas (no stroke is cut at the edge)
W = FULL - 2 * M               # drawable width (in)
LANE_FILL = "#efeee9"          # training-lane band (slightly darker than PANEL_BG so it survives print)

# ---------------------------------------------------------------- text helpers (inch-based layout)
FS_T, FS_D = 7.0, 6.5          # stage title / stage detail (pt)
LS = 1.22                      # line pitch in units of font size
PT = 1 / 72                    # points -> inches
CAP, DESC = 0.716, 0.21        # Arial cap height / descender depth (em)
T_CAP = 0.06                   # box top edge -> cap top of the first title line (in); same for every box
GAP_TD = 0.022                 # extra space between the last title line and the first detail line (in)
_meas = plt.figure(figsize=(2, 2))
_rend = _meas.canvas.get_renderer()


def tw(s, fs, weight="normal"):
    """Rendered width of a one-line string in inches."""
    t = _meas.text(0, 0, s, fontsize=fs, fontweight=weight)
    w = t.get_window_extent(_rend).width / _meas.dpi
    t.remove()
    return w


def lines_of(title, detail, tcol=INK):
    return [(s, FS_T, "bold", tcol) for s in title] + [(s, FS_D, "normal", INK2) for s in detail]


def baselines(lines, ls=LS, gap=GAP_TD):
    """Baseline offsets below the box top edge, first title cap top at T_CAP."""
    out, b, prev = [], None, None
    for s, fs, w, c in lines:
        if b is None:
            b = T_CAP + CAP * fs * PT
        else:
            b += fs * ls * PT + (gap if prev == "bold" and w == "normal" else 0)
        out.append(b)
        prev = w
    return out


def depth(lines, ls=LS, gap=GAP_TD):
    """Box top edge -> bottom of the last line's descenders (in)."""
    return baselines(lines, ls, gap)[-1] + DESC * lines[-1][1] * PT


def block_w(lines):
    return max(tw(s, fs, w) for s, fs, w, _ in lines)


def stage(ax, x, y, w, h, lines, fc=SURF, ec=INK2, lw=0.7, ls="-", r=0.045, pitch=LS, gap=GAP_TD):
    """Rounded box; text top-anchored (titles of boxes with a common top edge share one baseline)."""
    box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, ls=ls, r=r)
    top = y + h
    for (s, fs, wt, c), b in zip(lines, baselines(lines, pitch, gap)):
        ax.text(x + w / 2, top - b, s, ha="center", va="baseline", fontsize=fs, fontweight=wt, color=c, zorder=4)


def hbracket(ax, x0, x1, y, label, tick=0.04):
    """Set-style header bracket: RULE 0.5 pt line, 0.04 in end ticks pointing down, 7 pt INK2 label centred above."""
    ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color=RULE, lw=0.5, solid_capstyle="butt",
            solid_joinstyle="miter", zorder=1)
    ax.text((x0 + x1) / 2, y + 0.035, label, ha="center", va="baseline", fontsize=7, color=INK2, zorder=2)


def broken_arrow(ax, x0, y0, x1, y1, color, ls, lw=0.8, head=0.05):
    """Dashed/dotted shaft with a solid arrowhead (a patterned head renders as a smudge)."""
    v = np.array([x1 - x0, y1 - y0]); L = np.hypot(*v); u = v / L
    xe, ye = np.array([x1, y1]) - u * head
    ax.plot([x0, xe], [y0, ye], color=color, lw=lw, ls=ls, dash_capstyle="butt", zorder=1)
    arrow(ax, xe - u[0] * 0.005, ye - u[1] * 0.005, x1, y1, color=color, lw=lw)


# ---------------------------------------------------------------- photographs (equal height, thin gutters)
photos = [("img_side_view.jpg", "a", "Workstation"),
          ("img_step1.jpg", "b", "Top view, hand entering Box 1"),
          ("img_mmwave_sensor.jpg", "c", "IWR6843ISK radar"),
          ("img_tof_sensor.jpg", "d", "ToF sensor on a box wall")]
raw = [Image.open(SRC / p[0]).convert("RGB") for p in photos]
GUT = 0.05                                             # gutter between photos (in)
asp = [im.size[0] / im.size[1] for im in raw]
H_PH = (W - GUT * (len(raw) - 1)) / sum(asp)           # common photo height (in)
DPI_PH = 600
ims = []
for im in raw:
    target_h = int(round(H_PH * DPI_PH))
    if im.size[1] > target_h:
        im = im.resize((int(round(im.size[0] * target_h / im.size[1])), target_h), Image.LANCZOS)
    ims.append(np.asarray(im))

# callouts in photo (a), positions as fractions of the original 2923 x 3831 photo (x from left, y from top)
RADAR_XY = (1400 / 2923, 335 / 3831)                   # left end of the 3D-printed radar case, mid-height
CAM_XY = (935 / 2923, 1508 / 3831)                     # top edge of the webcam on the desk tripod

# ---------------------------------------------------------------- pipeline content (every text fact of the old schematic)
S = [  # run-time stages, left to right
    lines_of(["On-chip", "point cloud"], ["(x, y, v, SNR)", "10 Hz, 2-D"]),
    lines_of(["5-frame window"], ["0.5 s; 21 features (B2)", "or 2×48×48 heatmaps", "(CNN–GRU)"]),
    lines_of(["State estimator"], ["none / Box 1–5", "per frame"]),
    lines_of(["Debounce and", "cycle segmentation"], ["(Box 1 entry)"]),
    None,  # D1 / D2 stacked column
    lines_of(["Conformal $p$-values"], ["D3: $p_3$ = min(1,", "2 min($p_1$, $p_2$))", r"alarm if $p_3 \leq \alpha$"]),
]
D1 = lines_of(["D1 Markov"], ["sequence score"])
D2 = lines_of(["D2 step duration"], ["tempo score"])      # sentence case; main.tex 'D2, step duration'
TOF = lines_of(["5 ToF sensors"], ["one per box, 30 Hz"], tcol=INK2)   # off-line boxes recede
LAB = lines_of(["Smoothed frame", "labels"], [], tcol=INK2)            # 'training' is said by the lane label
CAL = lines_of(["Normal-only", "calibration cycles"], [], tcol=INK2)

PADX = 0.07
need = [block_w(s) + 2 * PADX if s else max(block_w(D1), block_w(D2)) + 2 * PADX for s in S]
GAP_MIN = 0.17
spare = W - sum(need) - GAP_MIN * (len(S) - 1)
assert spare > 0, f"stages do not fit: short by {-spare:.3f} in"
wid = [n + 0.5 * spare / len(S) for n in need]         # half the spare width widens the boxes ...
GAP = GAP_MIN + 0.5 * spare / (len(S) - 1)             # ... and half lengthens the arrows
xs = M + np.r_[0, np.cumsum([w + GAP for w in wid])][:-1]
assert abs(xs[-1] + wid[-1] - (FULL - M)) < 1e-9

# D boxes: same top anchor as the run-time row (D1 title on the row's title baseline), tighter pitch inside
D_PITCH, D_GAP, D_BOT = 1.12, 0.012, 0.045
DGAP = 0.045                                           # vertical gap between D1 and D2
DH = max(depth(D1, D_PITCH, D_GAP), depth(D2, D_PITCH, D_GAP)) + D_BOT
BOT_MIN = 0.06                                         # minimum space under the deepest run-time text block
MH = max(max(depth(s) for s in S if s) + BOT_MIN, 2 * DH + DGAP)

# vertical layout of panel e (inch coordinates, origin bottom-left of the schematic axes)
LB_PAD = 0.05                                          # lane band -> lane boxes
bh = max(depth(TOF), depth(LAB), depth(CAL)) + T_CAP   # lane boxes: equal pad above the cap top and below the text
LANE_H = bh + 2 * LB_PAD
LANE_Y = M
VGAP = 0.28                                            # between lane and main row (arrows + 2-line notes)
ROW_Y = LANE_Y + LANE_H + VGAP
HDR_Y = ROW_Y + MH + 0.075                             # frame/cycle header rule (ticks end 0.035 above the boxes)
E_TOP = HDR_Y + 0.035 + CAP * 7 * PT + 0.05            # top of the header labels + clearance
TITLE_ROW = 0.20                                       # letter + sub-caption row above photos and above panel e
GAP_PE = 0.10                                          # photos -> panel-e title row
FIGH = TITLE_ROW + H_PH + GAP_PE + TITLE_ROW + E_TOP

fig = plt.figure(figsize=(FULL, FIGH))


def label(x_in, y_in, letter, text):
    """Panel letter (bold, 9 pt) with a short sub-caption (7.5 pt INK2, as style.panel), baseline at y_in (in)."""
    fig.text(x_in / FULL, y_in / FIGH, letter, fontsize=9, fontweight="bold", color=INK, ha="left", va="baseline")
    fig.text((x_in + 0.13) / FULL, y_in / FIGH, text, fontsize=7.5, color=INK2, ha="left", va="baseline")


HALO = [pe.withStroke(linewidth=0.5 + 2 * 0.8, foreground="white")]


def callout(ax, text, target, anchor, dark):
    """Leader (0.5 pt, white 0.8 pt halo) from a 6.5 pt chip at `anchor` to `target` (photo fractions from top-left).
    dark: white bold text on an INK chip (the radar); otherwise INK text on a white chip (secondary device)."""
    tx, ty = target[0], 1 - target[1]
    ax.annotate(text, xy=(tx, ty), xytext=(anchor[0], 1 - anchor[1]), xycoords="axes fraction",
                textcoords="axes fraction", fontsize=6.5, fontweight="bold" if dark else "normal",
                color="white" if dark else INK, ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.5", fc=INK if dark else "white",
                          ec="none" if dark else RULE, lw=0.4, alpha=1.0 if dark else 0.92),
                arrowprops=dict(arrowstyle="-", color=INK if dark else INK2, lw=0.5, shrinkA=0, shrinkB=0,
                                path_effects=HALO))
    ax.plot([tx], [ty], marker="o", ms=2.4, mfc=INK if dark else INK2, mec="white", mew=0.5,
            transform=ax.transAxes, zorder=7, clip_on=False)


y_ph = FIGH - TITLE_ROW - H_PH
x = M
for (fn, let, cap), im, a in zip(photos, ims, asp):
    w = a * H_PH
    ax = fig.add_axes([x / FULL, y_ph / FIGH, w / FULL, H_PH / FIGH])
    ax.imshow(im, interpolation="lanczos", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(RULE); sp.set_linewidth(0.4)
    assert tw(cap, 7.5) + 0.13 <= w + 1e-6, f"sub-caption too wide for panel {let}"
    label(x, y_ph + H_PH + 0.06, let, cap)
    if let == "a":
        callout(ax, "radar", RADAR_XY, (0.20, RADAR_XY[1]), dark=True)
        callout(ax, "webcam", CAM_XY, (0.30, 0.255), dark=False)
    x += w + GUT

# ---------------------------------------------------------------- panel e: processing pipeline
axs = fig.add_axes([0, 0, 1, E_TOP / FIGH])
axs.set_xlim(0, FULL); axs.set_ylim(0, E_TOP); axs.axis("off")
label(M, E_TOP + 0.06, "e", "Processing pipeline")

# training-only lane (muted band) under stages 1-3
lane_x0, lane_x1 = M, xs[2] + wid[2] + 0.06
box(axs, lane_x0, LANE_Y, lane_x1 - lane_x0, LANE_H, fc=LANE_FILL, ec="none", r=0.05, zorder=0)
axs.text(lane_x0 + 0.08, LANE_Y + LANE_H / 2, "training\nonly", ha="left", va="center", fontsize=6.5,
         fontweight="bold", color=INK2, linespacing=1.2, zorder=1)

# main row: run-time stages
ymid = ROW_Y + MH / 2
for i, s in enumerate(S):
    if s is None:
        continue
    out = i == len(S) - 1
    stage(axs, xs[i], ROW_Y, wid[i], MH, s, ec=INK if out else INK2, lw=0.9 if out else 0.7)
# D1 / D2 stacked (ANOM colours: sequence violet, tempo amber)
xd, wd = xs[4], wid[4]
yd1 = ROW_Y + MH - DH
yd2 = ROW_Y
stage(axs, xd, yd1, wd, DH, D1, fc=TINT[VIOLET], ec=VIOLET, pitch=D_PITCH, gap=D_GAP)
stage(axs, xd, yd2, wd, DH, D2, fc=TINT[AMBER], ec=AMBER, pitch=D_PITCH, gap=D_GAP)

# run-time arrows (solid, left to right)
for i in range(3):
    arrow(axs, xs[i] + wid[i], ymid, xs[i + 1], ymid)
for yy in (yd1 + DH / 2, yd2 + DH / 2):
    arrow(axs, xs[3] + wid[3], yy, xd, yy)              # debounce -> D1, D2
    arrow(axs, xd + wd, yy, xs[5], yy)                  # D1, D2 -> conformal

# header rules: frame level vs cycle level
hbracket(axs, xs[0], xs[3] + wid[3], HDR_Y, "frame level")
hbracket(axs, xs[4], xs[5] + wid[5], HDR_Y, "cycle level")

# training lane boxes (dashed) and arrows
DASH = (0, (3, 2))
by = LANE_Y + LB_PAD
stage(axs, xs[1], by, wid[1], bh, TOF, fc=SURF, ec=MUTED, ls=DASH)
stage(axs, xs[2], by, wid[2], bh, LAB, fc=SURF, ec=MUTED, ls=DASH)
broken_arrow(axs, xs[1] + wid[1], by + bh / 2, xs[2], by + bh / 2, color=MUTED, ls=DASH)
xv = xs[2] + wid[2] / 2
broken_arrow(axs, xv, by + bh, xv, ROW_Y, color=MUTED, ls=DASH)
axs.text(xv + 0.07, (by + bh + ROW_Y) / 2, "training labels only;\nabsent at run time", ha="left", va="center",
         fontsize=6.5, style="italic", color=INK2, linespacing=1.2)

# calibration path (dotted, ROLE calibration colour) into the conformal block
xc, wc = xs[5], wid[5]
stage(axs, xc, by, wc, bh, CAL, fc=TINT[AQUA], ec=AQUA, ls=(0, (1, 1.5)), lw=0.9)
xcv = xc + wc / 2
broken_arrow(axs, xcv, by + bh, xcv, ROW_Y, color=AQUA, ls=(0, (1, 1.5)), lw=0.9)
axs.text(xcv - 0.07, (by + bh + ROW_Y) / 2, "sets the alarm\nthreshold", ha="right", va="center", fontsize=6.5,
         style="italic", color=INK2, linespacing=1.2)

for art in axs.patches + axs.lines + axs.texts:       # nothing in panel e is clipped by its own axes
    art.set_clip_on(False)

print(f"photo height {H_PH:.3f} in, photo a width {asp[0] * H_PH:.3f} in, figure {FULL} x {FIGH:.3f} in, "
      f"stage spare {spare:.3f} in, MH {MH:.3f}, DH {DH:.3f}, lane {LANE_H:.3f}")
# fixed canvas, no tight-bbox trimming: the PDF is exactly FULL wide and is included at scale 1.00
fig.savefig(FIG / "fig1_setup_pipeline.pdf", bbox_inches=None)
fig.savefig(FIG / "fig1_setup_pipeline.png", dpi=300, bbox_inches=None)
plt.close(fig)
print("saved fig1_setup_pipeline")

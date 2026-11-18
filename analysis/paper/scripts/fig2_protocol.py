"""Fig 2: evaluation protocol schematic (S0 / S1 / S2 for state estimation; normal-only cross-fitting for anomaly detection).
Content follows faz_a/DESIGN.md and reports/faz_a/04_anomaly.md ('normal cycles from s02: train 2 folds, calibration 2 folds, test 1 fold;
anomalies from s03-s06 tested in every fold') and reports/faz_a3/03_online.md (roles: train f+3, f+4; calibration f+1, f+2; test f).
Panel a is illustrative (cell counts are not data) except the S2 bar, whose segment widths are the Table 1 frame counts.
Panel b draws one cell per normal s02 cycle; the cycles per fold set (11, 10, 10, 10, 10; n = 51) are counted from
reports/faz_a/folds_anom.csv (use == 'normal_crossfit'), the same file 00_folds.md reports. The anomalous sessions are drawn
as one tinted block each (no count is drawn for them).
Style v2 (2026-11-05): drawn in inch coordinates on one full-figure axes so both panels share the same columns;
TRAIN / TEST neutrals, ROLE['calib'] (AQUA, calibration only) for calibration, ANOM sequence / tempo tints for the anomalous
sessions (no BLUE). Set conventions (critique round): canvas = FULL wide and saved untrimmed (included at scale 1.00),
panel letters top-most, panel-a key on the letter row, sentence-case key entries placed on one baseline, one bracket
style (RULE 0.5 pt, 0.04 in end ticks, 7 pt INK2 label), dashed REF_THRESH guides for the shared-frame boundaries."""
import numpy as np
import pandas as pd
from style import *
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- data used in panel b (counts only, no analysis)
fa = pd.read_csv(REP / "faz_a/folds_anom.csv")
n_fold = fa[fa.use == "normal_crossfit"].groupby("fold").size().reindex(range(5)).astype(int).tolist()
assert n_fold == [11, 10, 10, 10, 10], n_fold          # = 00_folds.md (51 normal cycles of s02)
N_NORMAL = sum(n_fold)

# ---------------------------------------------------------------- canvas in inches (one axes = whole figure)
W, H = FULL, 3.65
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(-H, 0); ax.axis("off")

LX = 0.02                      # left edge of the panel letters
X0, X1 = 1.33, 4.62            # main data column (strips in a, normal folds in b)
IX0, IX1 = 4.88, W - 0.02      # right column (zoom inset in a, anomalous sessions in b)
XL = X0 - 0.07                 # right edge of the row labels
FS, FS2 = 7.0, 6.5             # primary / secondary text size (pt)
PT = 1 / 72                    # one point in inches


def rect(x, y, w, h, fc, ec="none", lw=0.0, z=2, **kw):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, **kw))


def T(x, y, s, size=FS, color=INK, ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, **kw)


def text_w(t, size=FS):
    """Rendered width of a string in inches."""
    tx = T(0, 0, t, size=size)
    w = tx.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    tx.remove()
    return w


def legend_row(x, y, items, sw=0.13, sh=0.085, gap=0.06, sep=0.22, size=FS, span=None):
    """Hand-placed key row. items: (face, edge, label); edge None = no edge, else a 0.6 pt edge of that colour.
    All labels sit on one baseline y; each swatch is centred 0.35 font heights above it (no drift from descenders
    or italics). With span=(xa, xb) the items are justified to fill the span."""
    ws = [sw + gap + text_w(t, size) for _, _, t in items]
    if span is not None:
        x = span[0]
        sep = (span[1] - span[0] - sum(ws)) / (len(items) - 1)
        assert sep > 0.12, sep
    yc = y + 0.35 * size * PT
    for (fc, ec, t), w in zip(items, ws):
        if ec is None:
            rect(x, yc - sh / 2, sw, sh, fc)
        else:
            rect(x, yc - sh / 2, sw, sh, fc, ec=ec, lw=0.6)
        T(x + sw + gap, y, t, size=size, va="baseline")
        x += w + sep
    return x


def letter(y, s):
    T(LX, y, s, size=9, weight="bold", va="top")


def row_label(yc, code, name, sub):
    T(XL, yc + 0.045, rf"$\mathbf{{{code}}}$  {name}", ha="right")
    T(XL, yc - 0.065, sub, size=FS2, color=INK2, ha="right")


def bracket(xa, xb, y, label, below=False):
    """Shared bracket style (Figs 1, 2): RULE 0.5 pt line with 0.04 in end ticks and a 7 pt INK2 label.
    Header bracket (default): label centred above the line, ticks pointing down to the columns it spans.
    below=True: measuring brace under an object, ticks pointing up, label centred under the line."""
    tk = 0.04
    s = 1 if below else -1
    ax.plot([xa, xa, xb, xb], [y + s * tk, y, y, y + s * tk], color=RULE, lw=0.5, solid_capstyle="butt",
            solid_joinstyle="miter", zorder=3)
    if below:
        T((xa + xb) / 2, y - 0.03, label, size=7, color=INK2, ha="center", va="top")
    else:
        T((xa + xb) / 2, y + 0.03, label, size=7, color=INK2, ha="center", va="baseline")


# ================================================================ panel a: state estimation
yA = -0.02
letter(yA, "a")
CYC = [5, 4, 6, 5, 4, 5, 6, 4, 5, 5]            # windows per cycle in the illustrative excerpt (10 cycles)
nwin = sum(CYC)
GW, GC, SH = 0.010, 0.060, 0.150                 # gap between windows, gap between cycles, strip height
cw = (X1 - X0 - (len(CYC) - 1) * GC - (nwin - len(CYC)) * GW) / nwin
xs, cyc_of, spans = [], [], []
x = X0
for c, k in enumerate(CYC):
    a0 = x
    for j in range(k):
        xs.append(x); cyc_of.append(c)
        x += cw + (GW if j < k - 1 else 0)
    spans.append((a0, x))
    x += GC
xs, cyc_of = np.array(xs), np.array(cyc_of)
bounds = [(spans[c][1] + spans[c + 1][0]) / 2 for c in range(len(CYC) - 1)]

# key for panel a on the letter row, ~0.15 in right of the letter (set rule); set slightly below the letter's baseline
# so the bold letter stays the top-most element (as in Fig 7)
legend_row(LX + 0.15, yA - 0.108,[(TRAIN, None, "Training window"), (TEST, None, "Test window")])

# cycle header
yH = -0.265
T(XL, yH, "assembly cycles (excerpt)", size=FS2, color=INK2, ha="right")
for c, (a0, a1) in enumerate(spans):
    T((a0 + a1) / 2, yH, f"{c + 1}", size=FS2, color=MUTED, ha="center")


def strip(y, test_mask):
    for i in range(nwin):
        rect(xs[i], y, cw, SH, TEST if test_mask[i] else TRAIN)
    for b in bounds:                                  # visible cycle boundaries
        ax.plot([b, b], [y - 0.035, y + SH + 0.035], color=INK2, lw=0.6, solid_capstyle="butt", zorder=3)


# S0: random window split (illustrative ~20 % test), last two windows = the zoomed pair (training, test)
yS0 = -0.515
rng = np.random.default_rng(11)
test0 = rng.random(nwin) < 0.2
test0[-2], test0[-1] = False, True
strip(yS0, test0)
row_label(yS0 + SH / 2, "S0", "random window split", "80/20, leakage control")
T(X0, yS0 - 0.075, "neighbouring windows share 4 of 5 frames,", size=FS2, color=INK2, va="top")
T(X0, yS0 - 0.185, "yet land on both sides of the split", size=FS2, color=INK2, va="top")
pair_x0, pair_x1 = xs[-2] - 0.018, xs[-1] + cw + 0.018
pair_y0, pair_y1 = yS0 - 0.018, yS0 + SH + 0.018
rect(pair_x0, pair_y0, pair_x1 - pair_x0, pair_y1 - pair_y0, "none", ec=INK, lw=0.7, z=4)

# S1: cycle-grouped (one of five folds shown: cycles 3 and 8 in test)
yS1 = -1.025
test_cyc = {2, 7}
strip(yS1, np.isin(cyc_of, list(test_cyc)))
row_label(yS1 + SH / 2, "S1", "cycle-grouped, 5 folds", "primary; one fold shown")
T(X0, yS1 - 0.075, "a whole cycle (session, cycle counter) is either in training", size=FS2, color=INK2, va="top")
T(X0, yS1 - 0.185, "or in test; folds stratified by session", size=FS2, color=INK2, va="top")

# S2: leave one session out; segment widths = Table 1 frame counts
yS2 = -1.535
sizes = [4683, 2274, 2407, 1928, 1780, 3200]; names = ["s01", "s02", "s03", "s04", "s05", "s06"]
edges = X0 + np.r_[0, np.cumsum(sizes)] / sum(sizes) * (X1 - X0)
for k in range(6):
    held = k == 4
    rect(edges[k] + (0 if k == 0 else 0.012), yS2, edges[k + 1] - edges[k] - (0 if k == 0 else 0.012), SH,
         TEST if held else TRAIN)
    T((edges[k] + edges[k + 1]) / 2 + 0.006, yS2 + SH / 2, names[k], size=FS2, ha="center",
      color=SURF if held else INK)
row_label(yS2 + SH / 2, "S2", "leave one session out", "6 folds; one fold shown")
T(X0, yS2 - 0.075, "one session held out per fold (shown: s05); tests transfer", size=FS2, color=INK2, va="top")
T(X0, yS2 - 0.185, "between sessions, not between people", size=FS2, color=INK2, va="top")

# ---------------------------------------------------------------- zoom inset: two neighbouring S0 windows
iy1, iy0 = yS0 + SH + 0.005, yS0 + SH + 0.005 - 0.93          # inset top / bottom
ax.add_patch(FancyBboxPatch((IX0, iy0), IX1 - IX0, iy1 - iy0, boxstyle="round,pad=0,rounding_size=0.035",
                            fc=PANEL_BG, ec="none", zorder=0.5))
for (xa, ya), (xb, yb) in [((pair_x1, pair_y1), (IX0, iy1)), ((pair_x1, pair_y0), (IX0, iy0))]:
    ax.plot([xa, xb], [ya, yb], color=MUTED, lw=0.5, zorder=1)
T(IX0 + 0.08, iy1 - 0.1, "zoom: two neighbouring S0 windows", size=FS2, color=INK2)
NF, fw, fg = 6, 0.235, 0.035                       # frames t-4 .. t+1
fx0 = IX0 + (IX1 - IX0 - (NF * fw + (NF - 1) * fg)) / 2
fxs = [fx0 + i * (fw + fg) for i in range(NF)]
fy, fh = iy1 - 0.39, 0.095
flab = ["t−4", "t−3", "t−2", "t−1", "t", "t+1"]
sh0, sh1 = fxs[1] - fg / 2, fxs[4] + fw + fg / 2   # shared frames t-3 .. t
for i in range(NF):                                 # frame row: white; shared frames hatched (belong to both windows)
    shared = 1 <= i <= 4
    rect(fxs[i], fy, fw, fh, SURF, ec=MUTED, lw=0.5, z=2)
    if shared:
        rect(fxs[i], fy, fw, fh, "none", ec="none", z=2.1, hatch="//////", hatchcolor=INK2, hatch_linewidth=0.4)
    T(fxs[i] + fw / 2, fy + fh + 0.035, flab[i], size=FS2, color=INK2, ha="center", va="baseline")
bh = 0.105
wy_a, wy_b = fy - 0.165, fy - 0.31                 # window t, window t+1
rect(fxs[0], wy_a, fxs[4] + fw - fxs[0], bh, TRAIN, z=2)
T((fxs[0] + fxs[4] + fw) / 2, wy_a + bh / 2, "window t: training", size=FS2, ha="center")
rect(fxs[1], wy_b, fxs[5] + fw - fxs[1], bh, TEST, z=2)
T((fxs[1] + fxs[5] + fw) / 2, wy_b + bh / 2, "window t+1: test", size=FS2, ha="center", color=SURF)
by = wy_b - 0.06
for xg in (sh0, sh1):                              # dashed guides bounding the shared frames (boundary = REF_THRESH)
    ax.plot([xg, xg], [fy - 0.012, by + 0.04], zorder=1, **REF_THRESH)
bracket(sh0, sh1, by, "4 of 5 frames shared", below=True)

# ================================================================ panel b: cycle-level anomaly detection
yb = -2.03
letter(yb, "b")
yh0, yhA, yhB = yb - 0.075, yb - 0.255, yb - 0.385  # header lines: group / bracket rule (label above) / column
RH, RP = 0.13, 0.195                               # row height, row pitch
yr0 = yb - 0.58                                    # top row (fold 0), bottom edge

BG = 0.07                                          # gap between fold sets
pc = (X1 - X0 - 4 * BG) / N_NORMAL                 # width per normal cycle
cg = 0.007                                         # gap between cycles
bx = [X0]
for k in range(4):
    bx.append(bx[-1] + n_fold[k] * pc + BG)
T((X0 + X1) / 2, yh0, f"normal cycles of s02 (n = {N_NORMAL})", ha="center")
bracket(X0, X1, yhA, "shuffled assignment to folds 0–4")
for k in range(5):
    T(bx[k] + n_fold[k] * pc / 2, yhB, f"{k}", size=FS2, color=INK2, ha="center")

# anomalous sessions: equal-width tinted blocks with a coloured edge (all cycles scored in every fold); tints keep
# them behind the per-cycle normal grid, which carries the rotating fold roles
ag = 0.05
aw = (IX1 - IX0 - 3 * ag) / 4
ax_ = [IX0 + k * (aw + ag) for k in range(4)]
acol = [ROLE["sequence"], ROLE["sequence"], ROLE["tempo"], ROLE["tempo"]]
T((IX0 + IX1) / 2, yh0, "anomalous cycles, scored in every fold", ha="center")
bracket(ax_[0], ax_[1] + aw, yhA, "sequence")
bracket(ax_[2], ax_[3] + aw, yhA, "tempo")
for k, s in enumerate(["s03", "s04", "s05", "s06"]):
    T(ax_[k] + aw / 2, yhB, s, size=FS2, color=INK2, ha="center")

role_col = {0: ROLE["test"], 1: ROLE["calib"], 2: ROLE["calib"], 3: ROLE["train"], 4: ROLE["train"]}
for f in range(5):
    y = yr0 - f * RP
    T(XL, y + RH / 2, f"fold {f}", ha="right")
    for blk in range(5):
        rel = (blk - f) % 5                        # 0: test (fold f); 1, 2: calibration; 3, 4: fitting
        for j in range(n_fold[blk]):
            rect(bx[blk] + j * pc + cg / 2, y, pc - cg, RH, role_col[rel])
    for k in range(4):                             # edge inset by half its width so blocks keep their footprint
        rect(ax_[k] + 0.0042, y + 0.0042, aw - 0.0083, RH - 0.0083, TINT[acol[k]], ec=acol[k], lw=0.6)

# key for panel b: one row under the grid, justified over both columns
yleg = yr0 - 4 * RP - 0.20
legend_row(None, yleg, [(ROLE["train"], None, "Fit D1, D2 parameters"),
                        (ROLE["calib"], None, "Calibrate conformal $p$"),
                        (ROLE["test"], None, "Test, normal"),
                        (TINT[ROLE["sequence"]], ROLE["sequence"], "Test, sequence"),
                        (TINT[ROLE["tempo"]], ROLE["tempo"], "Test, tempo")], span=(X0, IX1))

# ---------------------------------------------------------------- save untrimmed: width = FULL, included at scale 1.00
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig2_protocol.{ext}", dpi=300, bbox_inches=None)
plt.close(fig)
from PIL import Image
wpx, hpx = Image.open(FIG / "fig2_protocol.png").size
assert 6.66 <= wpx / 300 <= 6.72, wpx
print("saved fig2_protocol", f"{wpx / 300:.3f} x {hpx / 300:.3f} in")

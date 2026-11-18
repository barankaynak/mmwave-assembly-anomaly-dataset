"""Fig 0 (v2): one-glance data flow of the method, drawn with real data (replaces v1, archived in paper/archive).

Left to right, every method is a box and the data it passes on is drawn inside it:
  radar frames (2-D points of one real 5-frame window) -> window statistics (its 21 numbers) -> state estimator B2
  (box per frame) -> visits and cycle (debounced visits with dwell and transit times) -> two scores (sequence D1 with
  E1 dashed as secondary, duration D2) -> calibration and decision (score / largest calibration score, conformal p,
  p = 2 min(p1, p2), alarm or no alarm).
Three thin rows carry three real test cycles of conformal fold f = 4 through stages 3-6 (stages 1-2 show one window of
the first, normal cycle): a normal s02 cycle (no alarm), an s03 cycle read 1-3-5 and a slow s06 cycle read 1-2-5
(D3 alarm through the duration channel). Fold 4 is the typical fold for 1-3-5 cycles: D3 flags 7.1 % of them there
(5-7 % in folds 0, 2, 3, 4; 100 % only in fold 1; 25.4 % overall, main.tex). In fold 4 the s03 cycle's D1 ties the
largest calibration D1, which belongs to the normal flicker cycle (s02 cycle 19, read 1-3-2-5; main.tex, D1 diagnosis),
so p1 = 2/21 and D3 gives no alarm (p = 0.19); E1 (secondary, with D2) flags it (p = 0.095). The figure shows both:
the tie on the dashed line, the D3 chip "no alarm" and a dashed violet E1 chip "alarm". The bottom cards give the
36 % (planned rule) and the 104 of 104 (E1 + D2) next to it.
Side lanes: dashed grey training-only lane (ToF, 30 Hz -> frame labels -> train B2); dotted aqua normal-only lane
(fit the scores, calibrate the p-values). Bottom: four result cards, every number copied from main.tex.

Sources (read only; nothing here changes a result):
  reports/faz_e/02_oneshot_cycles.csv.gz      D1, D2, p1, p2, D3 (= p3), S_E, p_E, D3' per (cycle, fold)
  reports/faz_a/04_cycle_scores.csv.gz        b2 x pred rows: role, natural_dev, and the fold-4 calibration D1 / D2
  reports/faz_a/oof/b2__S1__f*.csv.gz         B2 out-of-fold frame labels (y_pred), via faz_a2/a2_cycles.source_labels
  dataset/data (DATA_PATH)                    points of the shown window (probe/common.load_points)
The visit list, dwell and transit times are rebuilt with faz_a2/a2_cycles (the Phase A code); the script asserts that
the rebuilt D1 / D2 of the three cycles and of the 20 calibration cycles equal the stored values.
Display choices: stage 1 is mirrored in x (radar x runs from +0.21 m at Box 1 to -0.18 m at Box 5) so that the boxes
read 1 to 5 left to right, and carries a 10 cm scale bar instead of axes; box outlines are drawn at the median frame
centroid of each ToF box label (display only). The 21 statistics are shaded by their percentile among all 16,248
windows. Stage 6 divides each score by the largest of the 20 fold-4 calibration scores of its channel, so the dashed
line (1.0) is the alarm line of both channels: with 20 calibration cycles a channel reaches p <= 0.05, and D3 alarms
at alpha = 0.10, only above every calibration score (a tie gives p = 2/21). 19 of the 20 calibration D1 values are
equal and are drawn as one dot labelled with their count. Markers are neutral unless their channel fires.
Style: style.py colours (B2 blue only in stage 3, sequence violet, tempo amber, calibration aqua, normal grey; blue
and violet never share an edge); box / arrow look of fig1_setup.py panel (e); canvas exactly FULL wide, saved without
tight-bbox trimming (included at scale 1.00). The script asserts text fits its box, no text overlaps, no font is
below 6.5 pt and nothing leaves the canvas.
"""
import os
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # 'python -I' does not put the script directory on sys.path
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from style import *

os.environ.setdefault("DATA_PATH", str(ROOT / "data"))   # set DATA_PATH to the dataset's data/ folder
sys.path.insert(0, str(ROOT / "faz_a2"))
import a2_cycles as ac   # noqa: E402  (also puts faz_a and probe on sys.path)
import fa_common as fc   # noqa: E402
import common            # noqa: E402

# ================================================================= data
F = 4                                                   # conformal fold of the three example cycles (typical fold)
PICK = [("session_02_normal_seq125", 39, "normal (s02)", MUTED),
        ("session_03_normal_seq135", 2, "order 1–3–5 (s03)", VIOLET),
        ("session_06_anomaly_slow", 3, "slow (s06)", AMBER)]
ALPHA = 0.10

one = pd.read_csv(REP / "faz_e" / "02_oneshot_cycles.csv.gz")
sc = pd.read_csv(REP / "faz_a" / "04_cycle_scores.csv.gz")
sc = sc[(sc.state_source == "b2") & (sc.segmentation == "pred")]
cal = sc[(sc.role == "cal") & (sc.f == F)]
assert len(cal) == 20, len(cal)
CALMAX = {"D1": float(cal.D1.max()), "D2": float(cal.D2.max())}

ft = fc.frame_table()
LAB = ac.source_labels(sources=("b2",), verbose=False)["b2"]          # {session: B2 S1 out-of-fold label per frame row}
C, _ = ac.build_cycles("b2", "pred", LAB)
cl = C.to_dict("records")
tr_, ca_, te_ = ac._split_roles(cl, fc.anom_roles(F))
par = ac.fit_params(tr_)
reb_cal = ac.score_cycles(ca_, par)
m = reb_cal.merge(cal[["session", "cycle_idx", "D1", "D2"]], on=["session", "cycle_idx"], suffixes=("", "_csv"))
assert len(m) == 20 and np.allclose(m.D1, m.D1_csv, atol=1e-9) and np.allclose(m.D2, m.D2_csv, atol=1e-9)
oof = pd.concat([pd.read_csv(p) for p in sorted((REP / "faz_a" / "oof").glob("b2__S1__f*.csv.gz"))], ignore_index=True)

EX = []
for ses, ci, tag, col in PICK:
    r = one[(one.session == ses) & (one.cycle_idx == ci) & (one.f == F)].iloc[0]
    s4 = sc[(sc.session == ses) & (sc.cycle_idx == ci) & (sc.f == F) & (sc.role == "test")].iloc[0]
    c = next(x for x in cl if x["session"] == ses and x["cycle_idx"] == ci)
    rs = ac.score_cycle(c, par)
    assert abs(rs["D1"] - r.D1) < 1e-9 and abs(rs["D2"] - r.D2) < 1e-9, (ses, ci)
    assert not bool(s4.natural_dev) and int(c["start_frame_id"]) == int(r.start_frame_id)
    zs = {k: abs(v - par.d2[k][0]) / par.d2[k][1] for k, v in ac.steps_of(c) if k in par.d2}
    e = dict(session=ses, cycle_idx=ci, tag=tag, col=col, reading=r.reading_c, D1=r.D1, D2=r.D2, p1=r.p1, p2=r.p2,
             p3=r.D3_phaseA, S_E=r.S_E, p_E=r.p_E, p3E=r.D3prime, seq=c["seq"], dw=c["dw"], gp=c["gp"],
             dur=c["dur_s"], i0=c["start_idx"], i1=c["end_idx"], zmax=max(zs, key=zs.get), z=zs)
    e["alarm"] = e["p3"] <= ALPHA + 1e-9
    fr = ft[ft.session == ses].reset_index(drop=True)
    e["t"] = fr["t"].to_numpy()[e["i0"]:e["i1"] + 1]                    # row times incl. the next cycle's first row
    e["lab"] = LAB[ses][e["i0"]:e["i1"]]
    fid = fr["frame_id"].to_numpy()[e["i0"]:e["i1"]]
    st = oof[(oof.session == ses) & oof.frame_id.isin(fid)].set_index("frame_id")
    assert len(st) == len(fid) and (st["y_pred"] == pd.Series(e["lab"], index=fid).loc[st.index]).all()
    e["state_folds"] = sorted(st["fold"].unique().tolist())
    EX.append(e)

nrm, sq, slw = EX
for e in EX:
    e["alarmE"] = e["p3E"] <= ALPHA + 1e-9                                             # D3' (E1 in place of D1, with D2)
assert nrm["reading"] == "1-2-5" and not nrm["alarm"] and not nrm["alarmE"]            # normal: no alarm (D3 and D3')
assert sq["reading"] == "1-3-5" and not sq["alarm"] and sq["p1"] > ALPHA / 2           # s03: D3 does not flag ...
assert abs(sq["D1"] - CALMAX["D1"]) < 1e-9                                             # ... because D1 ties the largest
assert sq["p_E"] <= ALPHA / 2 and sq["alarmE"]                                         # ... E1 (secondary) flags it
assert slw["reading"] == "1-2-5" and slw["alarm"] and slw["p2"] <= ALPHA / 2 and slw["p1"] > ALPHA / 2   # s06: via D2
N_STACK = int(np.isclose(cal.D1, cal.D1.min(), atol=1e-12).sum())                     # equal calibration D1 values
assert N_STACK == 19, N_STACK
CAL_TOP = cal.loc[cal.D1.idxmax()]                                                    # the cycle that sets the D1 line

# one real 5-frame window of the normal cycle: inside its Box 2 visit (B2 and ToF agree), contiguous frame ids
W_ = fc.windows()
R_ = fc.win_rows(W_)
FEATS = ["cx", "cy", "n_points", "snr_sum", "mean_abs_v", "std_x", "std_y"]          # faz_a/02_state_baselines.py
FILL = {"cx": -2.0, "cy": -2.0, "n_points": 0.0, "snr_sum": 0.0, "mean_abs_v": 0.0, "std_x": 0.0, "std_y": 0.0}
tc = np.arange(5.0) - 2.0


def b2_features(rows):
    cols = []
    for f_ in FEATS:
        v = ft[f_].to_numpy(float)
        v = np.where(np.isnan(v), FILL[f_], v)[rows]
        cols += [v[:, -1], v.mean(1), (v * tc).sum(1) / (tc ** 2).sum()]
    return np.stack(cols, 1)


XALL = b2_features(R_)                                                                # (16248, 21)
assert XALL.shape == (16248, 21)
rows_n = ft.index[ft.session == nrm["session"]].to_numpy()
cand = []
for k in range(nrm["i0"] + 4, nrm["i1"]):
    w = np.flatnonzero(R_[:, -1] == rows_n[k])
    if len(w) != 1 or not bool(W_.contiguous.iloc[w[0]]):
        continue
    rr = R_[w[0]]
    if (LAB[nrm["session"]][k - 4:k + 1] == 2).all() and (ft.y_true.to_numpy()[rr] == 2).all():
        cand.append((k, w[0]))
assert cand, "no clean Box 2 window in the normal cycle"
K_LAST, WIN = cand[len(cand) // 2]
WROWS = R_[WIN]
WFID = ft.frame_id.to_numpy()[WROWS]
pctl = (XALL < XALL[WIN][None, :]).mean(0) + 0.5 * (XALL == XALL[WIN][None, :]).mean(0)   # mid-rank percentile
GRID = np.array([pctl[0::3], pctl[1::3], pctl[2::3]])                                    # rows: last, mean, slope
pts = common.load_points(nrm["session"])
pts = pts[pts.frame_id.isin(WFID)]
BOXC = {k: (float(ft.loc[ft.y_true == k, "cx"].median()), float(ft.loc[ft.y_true == k, "cy"].median())) for k in range(1, 6)}
_hp = pd.concat([common.load_points(s_) for s_ in common.sessions()], ignore_index=True)
_hp = _hp[_hp.label > 0]
BOX_Y = (min(float(_hp.y.quantile(0.02)), float(pts.y.min())) - 0.005, max(float(_hp.y.quantile(0.98)), float(pts.y.max())) + 0.005)

print(f"fold f = {F}; calibration: 20 s02 normal cycles; largest calibration D1 {CALMAX['D1']:.4f} "
      f"(s02 cycle {int(CAL_TOP.cycle_idx)}, read {CAL_TOP.visit_seq}; {N_STACK} of 20 at {cal.D1.min():.4f}), "
      f"largest D2 {CALMAX['D2']:.4f}")
for e in EX:
    print(f"{e['session']} cycle {e['cycle_idx']} (conformal fold {F}; B2 state folds {e['state_folds']}): "
          f"reading {e['reading']}, visits {'-'.join(map(str, e['seq']))}, dwell {[round(x, 2) for x in e['dw']]} s, "
          f"transit {[round(x, 2) for x in e['gp']]} s, cycle {e['dur']:.2f} s | D1 {e['D1']:.4f} p1 {e['p1']:.4f} | "
          f"D2 {e['D2']:.4f} p2 {e['p2']:.4f} (largest step {e['zmax']} z {e['z'][e['zmax']]:.2f}) | "
          f"D3 p3 {e['p3']:.4f} -> {'ALARM' if e['alarm'] else 'no alarm'} (primary) | E1 S_E {e['S_E']:.3f} p_E {e['p_E']:.4f} "
          f"D3' {e['p3E']:.4f} -> {'ALARM' if e['alarmE'] else 'no alarm'} (secondary)")
print(f"window shown in stages 1-2: {nrm['session']} frame ids {WFID.tolist()} ({len(pts)} points with v != 0); "
      f"21 statistics percentiles {np.round(pctl, 2).tolist()}")

# ================================================================= type and measuring
M = 0.025
Wd = FULL - 2 * M
FS_T, FS_D, FS_N = 7.0, 6.5, 7.0
PT, CAP, DESC, LEAD = 1 / 72, 0.716, 0.21, 0.294
T_CAP, GAP_TD, PADX = 0.06, 0.022, 0.05
DASH, DOT = (0, (3, 2)), (0, (1, 1.5))
LANE_FILL = "#efeee9"
RAMP = {0: "#ffffff", 1: "#dedcd6", 2: "#bdbbb5", 3: "#99978f", 4: "#75736d", 5: "#4d4c48"}   # none, Box 1..5
RAMP_TXT = {1: INK, 2: INK, 3: INK, 4: "white", 5: "white"}

_meas = plt.figure(figsize=(2, 2), dpi=600)
_rend = _meas.canvas.get_renderer()
_wc = {}


def _ext(s, fs, weight, style):
    t = _meas.text(0, 0, s, fontsize=fs, fontweight=weight, fontstyle=style)
    w = t.get_window_extent(_rend).width / _meas.dpi
    t.remove()
    return w


def tw(s, fs, weight="normal", style="normal"):
    """Advance width of a one-line string in inches; a sentinel keeps trailing spaces (segments of one line abut)."""
    k = (s, fs, weight, style)
    if k not in _wc:
        _wc[k] = _ext(s + "|", fs, weight, style) - _ext("|", fs, weight, style) if s.endswith(" ") else _ext(s, fs, weight, style)
    return _wc[k]


def segs_of(s):
    """'a **b** c' -> [(a, normal), (b, bold), (c, normal)]."""
    return [(p, "bold" if i % 2 else "normal") for i, p in enumerate(s.split("**")) if p]


def sw(segs, fs, style="normal"):
    return sum(tw(t, fs, w, style) for t, w in segs)


def L(s, fs=FS_D, color=INK2, kind="detail", style="normal", bold=False):
    return dict(segs=[(s, "bold")] if bold else segs_of(s), fs=fs, color=color, kind=kind, style=style)


def Tl(s, color=INK):
    return L(s, FS_T, color, "title", bold=True)


def baselines(lines):
    out, b, prev = [], None, None
    for ln in lines:
        fs = ln["fs"]
        if b is None:
            b = T_CAP + CAP * fs * PT
        else:
            b += (DESC * prev["fs"] + CAP * fs + LEAD * max(fs, prev["fs"])) * PT
            b += GAP_TD if prev["kind"] == "title" and ln["kind"] != "title" else 0
        out.append(b)
        prev = ln
    return out


def depth(lines):
    return baselines(lines)[-1] + DESC * lines[-1]["fs"] * PT


def block_w(lines):
    return max(sw(l["segs"], l["fs"], l["style"]) for l in lines)


# ================================================================= content
HEAD = [
    [Tl("Radar frames"), L("on-chip point cloud,"), L("10 Hz")],
    [Tl("Window"), Tl("statistics")],
    [Tl("State estimator B2"), L("random forest")],
    [Tl("Visits and cycle"), L("debounce; cut at"), L("each Box 1 entry")],
    None,                                                     # two-part header, drawn separately
    [Tl("Calibration and decision"), L("conformal p-values (D3)"), L("p = min(1, 2·min(p1, p2))")],
]
H5 = ([Tl("Sequence"), Tl("score")], [Tl("Duration"), Tl("score")])
CAPS = [None, ["every window"], ["box per frame", "(white: none)"], ["transit above,", "dwell below (s)"],
        ["one number each"], ["alarm if p ≤ 0.10"]]       # bottom-aligned, at most two lines

# ================================================================= horizontal layout
STRIP_S = max(e["dur"] for e in EX)                       # longest cycle (s) sets the time scale of stage 3
CH_W, CON_W = 0.13, 0.19                                 # stage 4 chip width, connector length
num_w = max(tw(f"{v:.2f}", FS_N, "bold") for e in EX for v in (e["D1"], e["D2"]))
COLW = max(num_w, tw("15.3", FS_N, "bold"), tw("D1", FS_D)) + 0.05
need = [None] * 6
need[0] = max(block_w(HEAD[0]) + 2 * PADX, 0.84)
CELL = 0.056
need[1] = max(block_w(HEAD[1]) + 2 * PADX, tw("slope", FS_D) + 0.04 + 7 * CELL + 2 * 0.045)
need[2] = 1.05
need[3] = max(block_w(HEAD[3]) + 2 * PADX, 3 * CH_W + 2 * CON_W + 2 * 0.045)
w5l = max(block_w(H5[0]) + 2 * 0.03, 2 * COLW + 0.04)
w5r = max(block_w(H5[1]) + 2 * 0.03, COLW + 0.04)
need[4] = w5l + w5r
CHIP_W = max(tw("no alarm", FS_D), tw("alarm", FS_D, "bold"), tw("E1: ", FS_D) + tw("alarm", FS_D, "bold")) + 0.07
TRK_W = 0.42
TRK_LAB = max(tw("D1", FS_D, "bold"), tw("D2", FS_D, "bold"))
need[5] = max(block_w(HEAD[5]) + 2 * PADX, 0.05 + TRK_LAB + 0.03 + TRK_W + 0.05 + CHIP_W + 0.04)
G12 = max(tw("5 frames", FS_D), tw("(0.5 s)", FS_D)) + 0.07
GAP = 0.10
spare = Wd - sum(need) - G12 - 4 * GAP
assert spare >= 0, f"stages do not fit: short by {-spare:.3f} in ({[round(n, 3) for n in need]})"
WS = list(need)
WS[2] += 0.8 * spare                                      # longer strips, room for the window label
WS[5] += 0.2 * spare
gaps = [G12, GAP, GAP, GAP, GAP]
XS = [M]
for w, g in zip(WS[:-1], gaps):
    XS.append(XS[-1] + w + g)
assert abs(XS[-1] + WS[-1] - (FULL - M)) < 1e-9
X5M = XS[4] + w5l * WS[4] / need[4]                       # split of the two-score box
w5l, w5r = X5M - XS[4], XS[4] + WS[4] - X5M

# ================================================================= vertical layout (d = inches below the canvas top)
d_hlab = M + CAP * 7 * PT                                  # header-bracket label baseline
d_rule = d_hlab + 0.035
D0 = d_rule + 0.075                                        # top of the stage boxes
HZ = max(max(depth(h) for h in HEAD if h), max(depth(h + [L("D1")]) for h in H5)) + 0.045
BT = D0 + HZ                                               # top of the row area
RP = 0.44                                                  # row pitch (room for the secondary E1 chip in stage 6)
LP = (DESC + CAP + LEAD) * FS_D * PT                       # line pitch at 6.5 pt
ROW_TAG = [BT + i * RP + 0.012 + CAP * FS_D * PT for i in range(3)]
ROW_MID = [d + 0.11 for d in ROW_TAG]
ROW_SUB = [d + 0.072 + CAP * FS_D * PT for d in ROW_MID]
BB = ROW_SUB[-1] + DESC * FS_D * PT                        # bottom of the row area
d_cap = BB + 0.07 + CAP * FS_D * PT + LP                   # baseline of the last caption line
DB = d_cap + DESC * FS_D * PT + 0.055                      # bottom of the stage boxes
LANE_T = DB + 0.20                                         # lanes
LB_PAD = 0.045
LANE_BOX = [L("ToF sensors", color=INK2), L("30 Hz")]
lane_bh = depth(LANE_BOX) + 0.05
KEY_H = 4 * LP + 2 * 0.02 + 0.03
LANE_H = max(2 * LB_PAD + lane_bh, KEY_H)
LANE_B = LANE_T + LANE_H
CARD_LINES = [                                             # wording and numbers as in main.tex (abstract, results)
    (MUTED, [Tl("False alarms"), L("**4 of 51** normal cycles (**7.8%**)", color=INK)], []),
    (AMBER, [Tl("Tempo deviations"), L("**93%** detected", color=INK), L("fast and slow cycles", color=INK2)], []),
    (VIOLET, [Tl("Sequence deviations"), L("**36%** detected by the planned rule", color=INK)],
     [L("**100%** (104 of 104 cycles) with E1 + D2", color=INK)]),
    (INK, [Tl("Evaluation"), L("macro-F1, cycle-grouped folds", color=INK2)], [L("accuracy inflation, random split", color=INK2)]),
]
EVAL_COLS = ("B2", "CNN–GRU")                              # card 4 is a small table: rows = the two lines above
EVAL_VALS = [("0.856", "0.841"), ("+0.45", "+4.0")]
EVAL_CW = [max(tw(EVAL_COLS[j], FS_D), *(tw(r[j], FS_D, "bold") for r in EVAL_VALS)) for j in range(2)]
EVAL_GAP = 0.10
SEC_GAP = 0.035
card_d = []
for _, pl, sl in CARD_LINES:
    dd = depth(pl + sl) + (SEC_GAP if sl else 0)
    card_d.append(dd)
d_rlab = LANE_B + 0.115 + CAP * 7 * PT
d_rrule = d_rlab + 0.035
CT = d_rrule + 0.075
CH = max(card_d) + 0.05
H = CT + CH + M
print(f"canvas {FULL} x {H:.3f} in; widths {[round(w, 3) for w in WS]}; spare {spare:.3f}; box height {DB - D0:.3f}")
assert 3.5 <= H <= 4.4, H


def Y(d):
    return H - d


# ================================================================= drawing helpers
fig = plt.figure(figsize=(FULL, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FULL)
ax.set_ylim(0, H)
ax.axis("off")
OWNER, LINE_ID, KEEP_CLIP = {}, {}, set()


def put(x, d, s, fs=FS_D, color=INK2, ha="left", weight=None, style="normal", owner=None, z=5):
    """One line; '**' marks bold parts; ha in left / center / right; d = baseline."""
    segs = [(s, weight)] if weight else segs_of(s)
    tot = sw(segs, fs, style)
    x0 = x - tot / 2 if ha == "center" else (x - tot if ha == "right" else x)
    lid = len(LINE_ID) + 1
    out = []
    for t, wt in segs:
        a = ax.text(x0, Y(d), t, ha="left", va="baseline", fontsize=fs, fontweight=wt, fontstyle=style, color=color,
                    zorder=z)
        if owner:
            OWNER[a] = owner
        LINE_ID[a] = lid
        out.append(a)
        x0 += tw(t, fs, wt, style)
    return out


def put_block(x, d_top, w, lines, align="center", padx=PADX):
    assert block_w(lines) + 2 * padx <= w + 1e-6, f"text too wide: {lines[0]['segs']}"
    for ln, b in zip(lines, baselines(lines)):
        s = "".join(f"**{t}**" if wt == "bold" and ln["kind"] != "title" else t for t, wt in ln["segs"])
        put(x + (w / 2 if align == "center" else padx), d_top + b, s, ln["fs"], ln["color"],
            "center" if align == "center" else "left", weight="bold" if ln["kind"] == "title" else None,
            style=ln["style"], owner=(x, x + w))


def rbox(x, d_top, w, h, fc=SURF, ec=INK2, lw=0.7, ls="-", r=0.045, z=2):
    return box(ax, x, Y(d_top + h), w, h, fc=fc, ec=ec, lw=lw, ls=ls, r=r, zorder=z)


def band(patch, x, d0, w, h, color):
    """Tinted header band clipped to a rounded box."""
    b = Rectangle((x, Y(d0 + h)), w, h, fc=color, ec="none", zorder=patch.get_zorder() - 0.5)
    ax.add_patch(b)
    b.set_clip_path(patch)
    KEEP_CLIP.add(b)


def harrow(x0, x1, d, color=INK2, lw=0.8):
    arrow(ax, x0, Y(d), x1, Y(d), color=color, lw=lw)


def broken(xa, da, xb, db, color, ls, lw=0.8, head=0.05):
    """Dashed / dotted shaft with a solid head (as fig1)."""
    p0, p1 = np.array([xa, Y(da)]), np.array([xb, Y(db)])
    u = (p1 - p0) / np.hypot(*(p1 - p0))
    pe_ = p1 - u * head
    ax.plot([p0[0], pe_[0]], [p0[1], pe_[1]], color=color, lw=lw, ls=ls, dash_capstyle="butt", zorder=1)
    arrow(ax, pe_[0] - u[0] * 0.005, pe_[1] - u[1] * 0.005, p1[0], p1[1], color=color, lw=lw)


def hbracket(x0, x1, d_line, d_label, label):
    ax.plot([x0, x0, x1, x1], [Y(d_line + 0.04), Y(d_line), Y(d_line), Y(d_line + 0.04)], color=RULE, lw=0.5,
            solid_capstyle="butt", solid_joinstyle="miter", zorder=1)
    put((x0 + x1) / 2, d_label, label, 7, INK2, "center")


def vline(x, d0, d1, color, lw, ls="-", z=3):
    ax.plot([x, x], [Y(d0), Y(d1)], color=color, lw=lw, ls=ls, solid_capstyle="butt", dash_capstyle="butt", zorder=z)


def hline(x0, x1, d, color, lw, ls="-", z=3):
    ax.plot([x0, x1], [Y(d), Y(d)], color=color, lw=lw, ls=ls, solid_capstyle="butt", dash_capstyle="butt", zorder=z)


# ================================================================= headers: frame level / cycle level
hbracket(XS[0], XS[2] + WS[2], d_rule, d_hlab, "frame level")
hbracket(XS[3], XS[5] + WS[5], d_rule, d_hlab, "cycle level")

# ================================================================= stage boxes
HB = HZ - 0.02                                            # tinted header band depth (stages 3 and 5)
P = []
for i in range(6):
    if i == 4:
        continue
    out = i == 5
    ec, lw = (BLUE, 0.9) if i == 2 else ((INK, 0.9) if out else (INK2, 0.7))
    p = rbox(XS[i], D0, WS[i], DB - D0, ec=ec, lw=lw)
    if i == 2:
        band(p, XS[i], D0, WS[i], HB, TINT[BLUE])
    put_block(XS[i], D0, WS[i], HEAD[i])
p5 = rbox(XS[4], D0, WS[4], DB - D0, ec=INK2, lw=0.7)
band(p5, XS[4], D0, w5l, HB, TINT[VIOLET])
band(p5, X5M, D0, w5r, HB, TINT[AMBER])
vline(X5M, D0, BB + 0.02, INK2, 0.7, z=2.5)
hline(XS[4], XS[4] + WS[4], D0 + HB, RULE, 0.5, z=2.5)
hline(XS[2], XS[2] + WS[2], D0 + HB, RULE, 0.5, z=2.5)
put_block(XS[4], D0, w5l, H5[0], padx=0.03)
put_block(X5M, D0, w5r, H5[1], padx=0.03)
for i, cp in enumerate(CAPS):
    for q, ln in enumerate(cp or []):
        put(XS[i] + WS[i] / 2, d_cap - (len(cp) - 1 - q) * LP, ln, FS_D, INK2, "center", style="italic",
            owner=(XS[i], XS[i] + WS[i]))

# ---------------------------------------------------------------- stage 1: points of the shown window
x1, w1 = XS[0], WS[0]
PX0, PX1 = x1 + 0.05, x1 + w1 - 0.05
xm = sorted(-BOXC[k][0] for k in BOXC)                    # mirrored x
xlo, xhi = xm[0] - 0.045, xm[-1] + 0.045
sx = (PX1 - PX0) / (xhi - xlo)                             # in per m (equal aspect)
yc = (BOX_Y[0] + BOX_Y[1]) / 2
BW, BH_ = 0.064, BOX_Y[1] - BOX_Y[0]                       # box outline size (m)
d_pc = ROW_MID[0] - 0.01                                   # centre of the point view (d)


def P1(x_m, y_m):
    return PX0 + (-x_m - xlo) * sx, Y(d_pc) + (y_m - yc) * sx


for k, (bx, by) in BOXC.items():
    X_, Y_ = P1(bx, yc)
    ax.add_patch(Rectangle((X_ - BW / 2 * sx, Y_ - BH_ / 2 * sx), BW * sx, BH_ * sx, fc="none", ec=RULE, lw=0.6,
                           zorder=2))
    put(X_, H - (Y_ - BH_ / 2 * sx) + 0.03 + CAP * FS_D * PT, str(k), FS_D, INK2, "center", owner=(x1, x1 + w1))
fr_order = {f_: i for i, f_ in enumerate(WFID)}
shade = ["#c9c8c3", "#a8a7a2", "#86857f", "#5f5e5a", INK]
for f_, g in pts.groupby("frame_id"):
    X_, Y_ = P1(g.x.to_numpy(), g.y.to_numpy())
    ax.scatter(X_, Y_, s=4.0, c=shade[fr_order[f_]], lw=0, zorder=4)
ys_all = [P1(0, y)[1] for y in pts.y]
assert min(ys_all) >= Y(d_pc) - BH_ / 2 * sx - 0.01, "points fall below the outlines"
d_sb = H - (Y(d_pc) - BH_ / 2 * sx) + 0.03 + CAP * FS_D * PT + DESC * FS_D * PT + 0.075   # under the box numbers
hline(PX0, PX0 + 0.10 * sx, d_sb, INK2, 0.8)
put(PX0 + 0.10 * sx + 0.04, d_sb + CAP * FS_D * PT / 2, "10 cm", FS_D, INK2, owner=(x1, x1 + w1))
NOTES = (["no raw-data", "capture board"], ["about **200 B**", "per frame"])
notes_h = sum(depth([L(s) for s in nt]) + 0.03 for nt in NOTES) + 0.045 * (len(NOTES) - 1)
n_d0 = max(d_sb + 0.10, (d_sb + 0.10 + DB - 0.06 - notes_h) / 2)        # centred in the free space below the scale
for NOTE in NOTES:
    nlines = [L(s, color=INK) for s in NOTE]
    nh = depth(nlines) + 0.03
    rbox(x1 + 0.05, n_d0, w1 - 0.10, nh, fc=PANEL_BG, ec="none", r=0.03, z=1.5)
    put_block(x1 + 0.05, n_d0 - 0.02, w1 - 0.10, nlines, padx=0.03)
    n_d0 += nh + 0.045
assert n_d0 - 0.045 <= DB - 0.04, "stage 1 notes do not fit"

# arrow 1 -> 2 with its data label
xa, xb = x1 + w1, XS[1]
harrow(xa, xb, ROW_MID[0])
put((xa + xb) / 2, ROW_MID[0] - 0.045 - (DESC + CAP + LEAD) * FS_D * PT, "5 frames", FS_D, INK, "center")
put((xa + xb) / 2, ROW_MID[0] - 0.045, "(0.5 s)", FS_D, INK2, "center")

# ---------------------------------------------------------------- stage 2: the window's 21 statistics
x2, w2 = XS[1], WS[1]
lab_w = max(tw(s, FS_D) for s in ("last", "mean", "slope"))
gx0 = x2 + (w2 - (lab_w + 0.04 + 7 * CELL)) / 2 + lab_w + 0.04
CELLH = 0.09
gd0 = ROW_MID[0] - 1.5 * CELLH
gre = ["#f4f3f0", "#2f2e2b"]


def mix(p):
    a, b = np.array(mpl.colors.to_rgb(gre[0])), np.array(mpl.colors.to_rgb(gre[1]))
    return a + (b - a) * float(p)


for i, nm in enumerate(("last", "mean", "slope")):
    for j in range(7):
        ax.add_patch(Rectangle((gx0 + j * CELL, Y(gd0 + (i + 1) * CELLH)), CELL, CELLH, fc=mix(GRID[i, j]), ec="white",
                               lw=0.6, zorder=3))
    put(gx0 - 0.04, gd0 + (i + 0.5) * CELLH + CAP * FS_D * PT / 2, nm, FS_D, INK2, "right", owner=(x2, x2 + w2))
d21 = gd0 + 3 * CELLH + 0.06 + CAP * FS_D * PT
put(x2 + w2 / 2, d21, "21 window", FS_D, INK, "center", owner=(x2, x2 + w2))
put(x2 + w2 / 2, d21 + LP, "statistics", FS_D, INK, "center", owner=(x2, x2 + w2))
for q, s_ in enumerate(["7 per frame:", "position, points,", "SNR, speed,", "spread"]):    # grid column order
    put(x2 + w2 / 2, d21 + (2.4 + q) * LP, s_, FS_D, INK2, "center", owner=(x2, x2 + w2))

# arrows 2 -> 3 (every cycle's windows), 3 -> 4, 4 -> 5, 5 -> 6 per row
for i in range(3):
    for a_, b_ in ((1, 2), (2, 3), (3, 4), (4, 5)):
        harrow(XS[a_] + WS[a_], XS[b_], ROW_MID[i], lw=0.7)

# ---------------------------------------------------------------- stage 3: B2 box per frame
x3, w3 = XS[2], WS[2]
SX0 = x3 + 0.055
TSC = (w3 - 0.11) / (STRIP_S + 0.15)                       # in per s
SH = 0.105
for i, e in enumerate(EX):
    put(SX0, ROW_TAG[i], e["tag"], FS_D, INK, owner=(x3, x3 + w3))
    t = e["t"] - e["t"][0]
    lab = e["lab"]
    d_top = ROW_MID[i] - SH / 2
    ax.add_patch(Rectangle((SX0, Y(d_top + SH)), t[-1] * TSC, SH, fc="white", ec=RULE, lw=0.5, zorder=2))
    for l, s0, s1 in ac.runs_of(lab):
        if l == 0:
            continue
        xa_, xb_ = SX0 + t[s0] * TSC, SX0 + t[s1] * TSC
        ax.add_patch(Rectangle((xa_, Y(d_top + SH)), xb_ - xa_, SH, fc=RAMP[l], ec="none", zorder=3))
        if xb_ - xa_ >= 0.055:
            put((xa_ + xb_) / 2, ROW_MID[i] + CAP * FS_D * PT / 2, str(l), FS_D, RAMP_TXT[l], "center", z=6,
                owner=(x3, x3 + w3))
    if i == 0:                                             # the window shown in stages 1-2
        wa = SX0 + t[K_LAST - 4 - e["i0"]] * TSC
        wb = SX0 + t[K_LAST + 1 - e["i0"]] * TSC
        dd = ROW_MID[i] + SH / 2 + 0.04
        ax.plot([wa, wa, wb, wb], [Y(dd - 0.025), Y(dd), Y(dd), Y(dd - 0.025)], color=INK, lw=0.6, zorder=4,
                solid_joinstyle="miter")
        put(wb + 0.03, dd + CAP * FS_D * PT * 0.5, "window shown left", FS_D, INK2, owner=(x3, x3 + w3))
# 1 s scale bar under the longest strip
d_sc = ROW_SUB[2]
xe = SX0 + EX[2]["dur"] * TSC
hline(xe - TSC, xe, d_sc - CAP * FS_D * PT / 2, INK2, 0.8)
put(xe - TSC - 0.035, d_sc, "1 s", FS_D, INK2, "right", owner=(x3, x3 + w3))

# ---------------------------------------------------------------- stage 4: visits with dwell and transit times
x4, w4 = XS[3], WS[3]
CHH = 0.115
cx0 = x4 + (w4 - (3 * CH_W + 2 * CON_W)) / 2
for i, e in enumerate(EX):
    assert len(e["seq"]) == 3
    for j, (b, dw) in enumerate(zip(e["seq"], e["dw"])):
        xc = cx0 + j * (CH_W + CON_W)
        ec, lw = (VIOLET, 1.1) if (i == 1 and b == 3) else ("none", 0)
        rbox(xc, ROW_MID[i] - CHH / 2, CH_W, CHH, fc=RAMP[b], ec=ec, lw=lw, r=0.02, z=3)
        put(xc + CH_W / 2, ROW_MID[i] + CAP * FS_D * PT / 2, str(b), FS_D, RAMP_TXT[b], "center", weight="bold",
            owner=(x4, x4 + w4))
        hot = f"dwell_{b}" == e["zmax"] and e["p2"] <= ALPHA / 2
        put(xc + CH_W / 2, ROW_SUB[i], f"{dw:.1f}", FS_D, AMBER if hot else INK2, "center",
            weight="bold" if hot else None, owner=(x4, x4 + w4))
        if j < 2:
            xa_ = xc + CH_W
            arrow(ax, xa_ + 0.012, Y(ROW_MID[i]), xa_ + CON_W - 0.012, Y(ROW_MID[i]), color=INK2, lw=0.6, ms=5, zorder=4)
            g = e["gp"][j]
            hot = f"gap_{b}_{e['seq'][j + 1]}" == e["zmax"] and e["p2"] <= ALPHA / 2
            put(xa_ + CON_W / 2, ROW_MID[i] - 0.03, f"{g:.1f}", FS_D, AMBER if hot else INK2, "center",
                weight="bold" if hot else None, owner=(x4, x4 + w4))

# ---------------------------------------------------------------- stage 5: two scores
x5 = XS[4]
cxs = [x5 + w5l * 0.27, x5 + w5l * 0.73, X5M + w5r / 2]
d_col = BT - 0.045                                         # column labels (last header line)
for xc, nm in zip(cxs, ("D1", "E1", "D2")):
    put(xc, d_col, nm, FS_D, INK2 if nm == "E1" else INK, "center", weight="bold", owner=(x5, x5 + WS[4]))
e1w = COLW
rbox(cxs[1] - e1w / 2, d_col - CAP * FS_D * PT - 0.035, e1w, ROW_SUB[2] - (d_col - CAP * FS_D * PT - 0.035), fc="none",
     ec=VIOLET, lw=0.6, ls=DASH, r=0.03, z=2.6)
for i, e in enumerate(EX):
    dn = ROW_MID[i] + CAP * FS_N * PT / 2
    hot1, hot2 = e["p1"] <= ALPHA / 2, e["p2"] <= ALPHA / 2
    put(cxs[0], dn, f"{e['D1']:.2f}", FS_N, VIOLET if hot1 else INK, "center", weight="bold" if hot1 else "normal",
        owner=(x5, X5M))
    put(cxs[1], dn, f"{e['S_E']:.1f}", FS_N, VIOLET if e["p_E"] <= ALPHA / 2 else INK2, "center", weight="normal",
        owner=(x5, X5M))
    put(cxs[2], dn, f"{e['D2']:.2f}", FS_N, AMBER if hot2 else INK, "center", weight="bold" if hot2 else "normal",
        owner=(X5M, X5M + w5r))

# ---------------------------------------------------------------- stage 6: calibration and decision
x6, w6 = XS[5], WS[5]
TX0 = x6 + 0.05 + TRK_LAB + 0.03
CHX = x6 + w6 - 0.04 - CHIP_W
TX1 = CHX - 0.05
vals = [e["D1"] / CALMAX["D1"] for e in EX] + [e["D2"] / CALMAX["D2"] for e in EX]
XMAX = max(1.18, max(vals) * 1.07)
TS = (TX1 - TX0) / XMAX
XTH = TX0 + 1.0 * TS
OFF = 0.042                                                # track offset from the row mid
CH_H, E_H = 0.12, 0.11                                     # decision chip, secondary E1 chip
calv = {"D1": cal.D1.to_numpy() / CALMAX["D1"], "D2": cal.D2.to_numpy() / CALMAX["D2"]}
CH_COL = {"D1": VIOLET, "D2": AMBER}
P_OF = {"D1": "p1", "D2": "p2"}
vline(XTH, ROW_TAG[0] + 0.02, ROW_MID[2] + OFF + 0.045, INK2, 0.6, ls=(0, (3, 2)), z=2.8)
put(XTH - 0.03, ROW_TAG[0], "normal", FS_D, INK2, "right", owner=(x6, x6 + w6))          # key: aqua dots
ax.scatter([XTH - 0.06 - tw("normal", FS_D)], [Y(ROW_TAG[0] - CAP * FS_D * PT / 2)], s=4.5, c=AQUA, lw=0, zorder=4)
put(XTH + 0.03, ROW_TAG[0], "largest", FS_D, INK2, "left", owner=(x6, x6 + w6))
for i, e in enumerate(EX):
    for k, ch in enumerate(("D1", "D2")):
        dt = ROW_MID[i] + (-OFF if k == 0 else OFF)
        put(x6 + 0.05, dt + CAP * FS_D * PT / 2, ch, FS_D, CH_COL[ch], weight="bold", owner=(x6, x6 + w6))
        ax.add_patch(Rectangle((XTH, Y(dt) - 0.022), TX1 - XTH, 0.044, fc=TINT[MUTED], ec="none", zorder=2.1))
        hline(TX0, TX1, dt, RULE, 0.5, z=2.2)                # alarm zone (tint) right of the dashed line
        ax.scatter(TX0 + calv[ch] * TS, np.full(len(calv[ch]), Y(dt)), s=4.5, c=AQUA, lw=0, zorder=3.5)
        v = e[ch] / CALMAX[ch]
        fire = e[P_OF[ch]] <= ALPHA / 2 + 1e-9
        assert fire == (v > 1 + 1e-9), (e["session"], ch)  # beyond the dashed line <=> this channel's p <= 0.05
        ax.scatter([TX0 + v * TS], [Y(dt)], s=22 if fire else 20, facecolors="none",
                   edgecolors=CH_COL[ch] if fire else INK2, linewidths=1.1 if fire else 0.7, zorder=4.5)
        if i == 0 and ch == "D1":                          # the 19 equal calibration values are one dot
            xs_ = TX0 + float(cal.D1.min()) / CALMAX["D1"] * TS + 0.055
            for a_ in put(xs_, dt + CAP * FS_D * PT / 2, f"×{N_STACK}", FS_D, INK2, owner=(TX0, XTH - 0.02), z=4):
                a_.set_bbox(dict(fc=SURF, ec="none", pad=0.4))
        if abs(v - 1.0) < 1e-9:                            # a tie with the largest calibration score: p = 2/21
            put(XTH + 0.03, ROW_TAG[i], "tie", FS_D, INK2, owner=(x6, x6 + w6))
    al = e["alarm"]
    ac_ = (AMBER if e["p2"] <= ALPHA / 2 else VIOLET) if al else None      # colour of the channel that fired
    rbox(CHX, ROW_MID[i] - CH_H / 2, CHIP_W, CH_H, fc=TINT[ac_] if al else SURF, ec=ac_ if al else RULE,
         lw=0.9 if al else 0.6, r=0.03, z=3)
    put(CHX + CHIP_W / 2, ROW_MID[i] + CAP * FS_D * PT / 2, "alarm" if al else "no alarm", FS_D, INK if al else INK2,
        "center", weight="bold" if al else "normal", owner=(CHX, CHX + CHIP_W))
    p3 = e["p3"]
    pv = "1" if p3 > 0.995 else (f"{p3:.3f}".rstrip("0") if p3 < 0.1 else f"{p3:.2f}")
    put(CHX + CHIP_W / 2, ROW_SUB[i], f"p = {pv}", FS_D, INK2, "center", owner=(x6, x6 + w6))
    if e["alarmE"] and not al:                             # secondary rule (E1 with D2) flags what D3 misses
        d_e = ROW_SUB[i] + DESC * FS_D * PT + 0.03
        rbox(CHX, d_e, CHIP_W, E_H, fc=TINT[VIOLET], ec=VIOLET, lw=0.8, ls=DASH, r=0.03, z=3)
        put(CHX + CHIP_W / 2, d_e + E_H / 2 + CAP * FS_D * PT / 2, "E1: **alarm**", FS_D, INK, "center",
            owner=(CHX, CHX + CHIP_W))
        assert d_e + E_H < ROW_MID[i + 1] - OFF - 0.06, "E1 chip crowds the next row"

# ================================================================= lanes
# training only (dashed grey band under stages 1-3)
lx0, lx1 = XS[0], XS[2] + WS[2]
box(ax, lx0, Y(LANE_B), lx1 - lx0, LANE_B - LANE_T, fc=LANE_FILL, ec="none", r=0.05, zorder=0)
put(lx0 + 0.07, LANE_T + LB_PAD + T_CAP + CAP * FS_D * PT, "training only", FS_D, INK2, weight="bold",
    owner=(lx0, XS[1]))
put(lx0 + 0.07, LANE_T + LB_PAD + T_CAP + CAP * FS_D * PT + (DESC + CAP + LEAD) * FS_D * PT, "absent at run time",
    FS_D, INK2, style="italic", owner=(lx0, XS[1] + 0.1))
by = LANE_T + (LANE_H - lane_bh) / 2
tof_x, tof_w = XS[1] + 0.03, WS[1] - 0.06
rbox(tof_x, by, tof_w, lane_bh, fc=SURF, ec=MUTED, ls=DASH)
put_block(tof_x, by, tof_w, LANE_BOX, padx=0.03)
lab_lines = [L("frame labels", color=INK2)]
lab_x, lab_w_ = XS[2] + 0.06, block_w(lab_lines) + 0.12
rbox(lab_x, by, lab_w_, lane_bh, fc=SURF, ec=MUTED, ls=DASH)
put(lab_x + lab_w_ / 2, by + lane_bh / 2 + CAP * FS_D * PT / 2, "frame labels", FS_D, INK2, "center",
    owner=(lab_x, lab_x + lab_w_))
broken(tof_x + tof_w, by + lane_bh / 2, lab_x, by + lane_bh / 2, MUTED, DASH)
xv = lab_x + lab_w_ / 2
broken(xv, by, xv, DB, MUTED, DASH)
put(xv + 0.05, (by + DB) / 2 + CAP * FS_D * PT / 2, "train B2", FS_D, INK2, style="italic", owner=(x3, x3 + w3))

# normal-only calibration lane (dotted aqua) under stages 5-6
cal_lines = [L("**s02** normal cycles only", color=INK), L("about 20 fit + 20 calibrate", color=INK2)]
cx_, cw_ = XS[4] + 0.04, XS[5] + WS[5] - 0.04 - (XS[4] + 0.04)
ch_ = max(lane_bh, depth(cal_lines) + 0.05)
rbox(cx_, by, cw_, ch_, fc=TINT[AQUA], ec=AQUA, ls=DOT, lw=0.9)
put_block(cx_, by, cw_, cal_lines)
for xx, lab_ in ((XS[4] + WS[4] / 2, "fit"), (XS[5] + WS[5] / 2, "calibrate")):
    broken(xx, by, xx, DB, AQUA, DOT, lw=0.9)
    put(xx + 0.05, (by + DB) / 2 + CAP * FS_D * PT / 2, lab_, FS_D, INK2, style="italic")

# line-style key under stage 4
kx = XS[3] + 0.02
KEY = [(["primary"], INK2, "-", 0.8), (["secondary or", "training only"], MUTED, DASH, 0.8),
       (["calibration"], AQUA, DOT, 0.9)]
dk = LANE_T + 0.015 + CAP * FS_D * PT
for labs, colr, ls, lw in KEY:
    hline(kx, kx + 0.2, dk - CAP * FS_D * PT / 2, colr, lw, ls=ls)
    for q, lab_ in enumerate(labs):
        put(kx + 0.25, dk + q * LP, lab_, FS_D, INK2, owner=(kx, XS[4] - 0.02))
    dk += len(labs) * LP + 0.02
put(XS[3] + WS[3] / 2, DB + 0.115, f"rows: three test cycles (fold {F})", FS_D, INK2, "center", style="italic",
    owner=(xv + 0.05 + tw("train B2", FS_D) + 0.04, XS[4] + WS[4] / 2 - 0.03))

# ================================================================= results
hbracket(M, FULL - M, d_rrule, d_rlab, "results over all cycles")
nat = [block_w(pl + sl) + 0.17 for _, pl, sl in CARD_LINES]
nat[3] = block_w(CARD_LINES[3][1][1:] + CARD_LINES[3][2]) + 0.11 + 2 * EVAL_GAP + sum(EVAL_CW) + 0.06
CG = 0.08
csp = Wd - sum(nat) - CG * 3
assert csp >= 0, f"cards too wide by {-csp:.3f}"
xx = M
for (colr, pl, sl), wn, dd in zip(CARD_LINES, nat, card_d):
    wc = wn + csp / 4
    rbox(xx, CT, wc, CH, fc=SURF, ec=RULE, lw=0.5, r=0.04, z=1)
    put_block(xx + 0.06, CT, wc - 0.06, pl, align="left", padx=0.05)
    bl = baselines(pl)
    d_end = CT + bl[-1] + DESC * FS_D * PT
    vline(xx + 0.07, CT + T_CAP - 0.01, d_end, colr, 2.0)
    if sl:
        s_top = d_end + SEC_GAP
        put_block(xx + 0.06, s_top - T_CAP, wc - 0.06, sl, align="left", padx=0.05)
        vline(xx + 0.07, s_top, s_top - T_CAP + depth(sl), colr, 1.1, ls=(0, (1.6, 1.2)))   # secondary: dashed
    if pl[0]["segs"][0][0] == "Evaluation":               # value columns of the table card
        xr = [xx + wc - 0.06 - EVAL_CW[1] - EVAL_GAP, xx + wc - 0.06]
        bl_s = baselines(sl)
        rows_d = [CT + bl[0], CT + bl[1], s_top - T_CAP + bl_s[0]]
        for j in range(2):
            put(xr[j], rows_d[0], EVAL_COLS[j], FS_D, INK2, "right", owner=(xx, xx + wc))
            for q in range(2):
                put(xr[j], rows_d[q + 1], EVAL_VALS[q][j], FS_D, INK, "right", weight="bold", owner=(xx, xx + wc))
    xx += wc + CG

for art in ax.patches + ax.lines + ax.texts + ax.collections:
    if art not in KEEP_CLIP:
        art.set_clip_on(False)

# ================================================================= checks
rend = fig.canvas.get_renderer()
texts = [t for t in ax.texts if t.get_text().strip()]
assert min(t.get_fontsize() for t in texts) >= 6.5
ext = {t: t.get_window_extent(rend) for t in texts}
px = fig.dpi
PROBLEMS = 0
for t, (xa_, xb_) in OWNER.items():
    e_ = ext[t]
    if not (e_.x0 >= xa_ * px - 0.5 and e_.x1 <= xb_ * px + 0.5):
        print(f"[layout] text leaves its box: {t.get_text()!r} by {max(xa_ * px - e_.x0, e_.x1 - xb_ * px) / px:.3f} in")
        PROBLEMS += 1
bad = []
for a, b in combinations(texts, 2):
    if LINE_ID.get(a) == LINE_ID.get(b):
        continue
    ea, eb = ext[a], ext[b]
    if min(ea.x1, eb.x1) - max(ea.x0, eb.x0) > 0.6 and min(ea.y1, eb.y1) - max(ea.y0, eb.y0) > 0.6:
        bad.append((a.get_text(), b.get_text()))
for b_ in bad:
    print(f"[layout] overlapping text: {b_}")
assert PROBLEMS == 0 and not bad, "layout problems (see [layout] lines)"
assert all(0 <= e_.x0 and e_.x1 <= FULL * px and 0 <= e_.y0 and e_.y1 <= H * px for e_ in ext.values()), "text off canvas"

fig.savefig(FIG / "fig0_overview.pdf", bbox_inches=None)
fig.savefig(FIG / "fig0_overview.png", dpi=300, bbox_inches=None)
plt.close(fig)
print("saved fig0_overview")

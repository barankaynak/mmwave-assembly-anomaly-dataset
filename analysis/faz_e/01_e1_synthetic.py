"""Phase E - 01: E1 normal-side quantities (DESIGN_E section 3) and synthetic validation battery (section 4).

Loads ONLY session_01_training and session_02_normal_seq125:
  - B2 S1 OOF per-frame outputs  reports/faz_a/oof/b2__S1__f{0..4}.csv.gz  (rows of other sessions dropped chunk by chunk)
  - normal folds                 reports/faz_a/folds_anom.csv               (s01/s02 rows only)
  - Phase A reference scores     reports/faz_a/04_cycle_scores.csv.gz      (b2 x pred x s02 x normal rows only)
The all-session frame table of faz_a/fa_common.py is never built: a2_cycles.ctx() is replaced by an s02-only context
assembled from the OOF rows, and every raw-data loader of fa_common / probe.common is replaced by a tripwire.

Steps
  0. Gate: the 51 s02 normal B2 x pred cycles rebuilt with faz_a2/a2_cycles.py must reproduce D0/D1/D2/p0..p3 of
     04_cycle_scores (all 255 (cycle, f, role) rows) - otherwise stop.
  1. Normal side: route/mu per fold, 51 S_E, calibration maxima, p_E, D3' = min(1, 2 min(p_E, p_2)), FAR, alarm set
     against Phase A D3.
  2. Synthetic battery (kinds 1-12) from the 50 s02 normal parents in their Phase A test role (collapsed reading 1-2-5)
     and ToF-true Box-3/Box-4 runs of s01 (complete loops), D1/D2/p_1/p_2 recomputed for every synthetic cycle.
  3. Acceptance items of section 4.
Outputs: stdout (every number), reports/faz_e/01_synthetic.md and CSVs in reports/faz_e/.
Run:  bash tools/local_run.sh faz_e/01_e1_synthetic.py      (CPU only)
"""
from __future__ import annotations

import datetime as _dt
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a2"))
sys.path.insert(0, str(ROOT / "faz_e"))
import e1_core as E  # noqa: E402  (unit tests of DESIGN_E 2.4 run at import)
import a2_cycles as ac  # noqa: E402
import fa_common as fc  # noqa: E402
import common  # noqa: E402  (probe/common.py; only its loaders are referenced here, to disable them)

T_START = time.time()
DESIGN_SHA = E.assert_design_frozen()
S01, S02 = "session_01_training", "session_02_normal_seq125"
KEEP = (S01, S02)
OUT = ROOT / "reports" / "faz_e"
OUT.mkdir(parents=True, exist_ok=True)
DT = ac.DT
ALPHA = 0.10
TOL = 1e-9
NONE_P = np.array([1.0, 0, 0, 0, 0, 0])
TEMPO = (0.6, 0.7, 0.8, 1.25, 1.4, 1.7)
SPLIT_GAPS = (2, 3, 5)
TRUNC_KEEP = (0, 1, 2, 3)
MID_DWELL_MIN = 6        # Box-2 block >= 6 frames: each half keeps >= 3 frames (= MIN_BOX + 1)
FLICKER_EXPECTED = 1.09
INPUTS = {f"oof_b2_S1_f{k}": fc.oof_path("b2", "S1", k) for k in range(5)}
INPUTS["folds_anom"] = fc.OUT / "folds_anom.csv"
INPUTS["cycle_scores"] = fc.OUT / "04_cycle_scores.csv.gz"


# ------------------------------------------------------------------ tripwires: no raw / all-session loader may run
TRIPPED = []


def _forbidden(name):
    def f(*a, **k):
        TRIPPED.append(name)
        raise RuntimeError(f"[E1-01] forbidden loader called: {name} (would load sessions other than s01/s02)")
    return f


for _mod, _names in ((fc, ("frame_table", "anom_loops", "load_oof", "heatmaps", "windows", "window_batch", "win_rows")),
                     (common, ("sessions", "load_merged", "load_points", "load_points_raw", "load_frames",
                               "load_tof_raw", "tof_segments", "loops"))):
    for _n in _names:
        assert hasattr(_mod, _n), _n
        setattr(_mod, _n, _forbidden(f"{_mod.__name__}.{_n}"))


def hr(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def read_filtered(path, keep_fn, chunksize=5000):
    """Read a CSV chunk by chunk and keep only the rows selected by keep_fn (other rows are dropped per chunk)."""
    parts = [ch[keep_fn(ch)] for ch in pd.read_csv(path, chunksize=chunksize)]
    return pd.concat(parts, ignore_index=True)


def fmt(x, nd=3):
    if isinstance(x, (float, np.floating)):
        return "nan" if np.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def md_table(df: pd.DataFrame, nd=3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(fmt(v, nd) for v in r) + " |")
    return "\n".join(out)


def alarm(p, a):
    return np.asarray(p, float) <= a + TOL


# ================================================================== 0. load (s01/s02 only) and reproduction gate
hr("0. Inputs (s01 / s02 only) and Phase A reproduction gate")
oof = pd.concat([read_filtered(INPUTS[f"oof_b2_S1_f{k}"], lambda ch: ch["session"].isin(KEEP)) for k in range(5)],
                ignore_index=True)
assert list(oof.columns) == fc.OOF_COLS, list(oof.columns)
oof = oof.sort_values(["session", "frame_id"]).reset_index(drop=True)
AL = read_filtered(INPUTS["folds_anom"], lambda ch: ch["session"].isin(KEEP))
CS = read_filtered(INPUTS["cycle_scores"], lambda ch: (ch["state_source"] == "b2") & (ch["segmentation"] == "pred")
                   & (ch["session"] == S02) & (ch["atype"] == "normal"))
for nm, d in (("OOF", oof), ("folds_anom", AL), ("04_cycle_scores", CS)):
    assert set(d["session"].unique()) <= set(KEEP), (nm, set(d["session"].unique()))
assert set(CS["session"].unique()) == {S02}
CS["natural_dev"] = CS["natural_dev"].astype(str).str.lower().eq("true")
assert int(CS["natural_dev"].sum()) == 0, "natural_dev cycles in the b2 x pred normal pool (plan: none)"
AL["natural_dev"] = AL["natural_dev"].astype(bool)
AL["is_cycle"] = AL["is_cycle"].astype(bool)
print(f"OOF rows kept: {len(oof)} ({dict(Counter(oof['session']))}); sessions in memory: {sorted(set(oof['session']))}")
print(f"folds_anom rows kept: {len(AL)} ({dict(Counter(AL['session']))}); 04_cycle_scores rows kept: {len(CS)} "
      f"(b2 x pred x s02 x normal), natural_dev rows among them: 0")
for s, g in oof.groupby("session"):
    fid = g["frame_id"].to_numpy()
    assert fid[0] == fc.L - 1 and (np.diff(fid) == 1).all(), s
    print(f"  {s}: {len(g)} window rows, frame_id {fid[0]}..{fid[-1]} contiguous (first {fc.L - 1} frames have no window)")

# s02-only replacement of a2_cycles.ctx(): frame table rebuilt from the OOF rows (+ the 4 window-less leading frames)
g2 = oof[oof["session"] == S02].reset_index(drop=True)
pad = pd.DataFrame({"session": S02, "frame_id": np.arange(fc.L - 1), "t": g2["t"].iloc[0] - DT * (fc.L - 1 - np.arange(fc.L - 1)),
                    "loop_id": int(g2["loop_id"].iloc[0]), "y_true": -1})
ft2 = pd.concat([pad, g2[["session", "frame_id", "t", "loop_id", "y_true"]]], ignore_index=True)
ft2["row"] = np.arange(len(ft2))
assert (ft2["frame_id"].to_numpy() == np.arange(len(ft2))).all()
CTX = SimpleNamespace(ft=ft2, AL=AL, SESS=[S02], OFF={S02: int(AL.loc[AL["session"] == S02, "first_row"].min())},
                      SE={S02: ft2.reset_index(drop=True)},
                      ALX={(r.session, int(r.loop_id)): r for r in AL.itertuples(index=False)})
ac.ctx = lambda: CTX
LAB = ac.labels_from_oof(oof, strict=False)
assert set(LAB) == {S02}
C, CNT = ac.build_cycles("b2", "pred", LAB, sessions=[S02])
print(f"s02 B2 x pred cycles rebuilt: {len(C)}; counts: {CNT[0]}")
assert len(C) == 51, len(C)
assert int(C["natural_dev"].sum()) == 0
SA, NCAL = ac.score_group(C, verbose=False)
print("Phase A n_cal per fold:", {x["f"]: x["n_cal"] for x in NCAL}, " n_train:", {x["f"]: x["n_train"] for x in NCAL})

KEY = ["cycle_idx", "f", "role"]
ok, gate = E.compare_to_reference(SA, CS, KEY)
print("reproduction gate:", gate)
if not ok:
    raise SystemExit("[E1-01] STOP: rebuilt s02 normal cycles do not reproduce reports/faz_a/04_cycle_scores.csv.gz")
print(f"GATE PASSED: {gate['n_matched']} (cycle, f, role) rows reproduce Phase A D0/D1/D2/p0..p3 "
      f"(max |diff| {max(v for k, v in gate.items() if k.startswith('maxdiff')):.2e})")

# per-frame posteriors of s02 aligned with the rebuilt frame table (window-less leading frames: one-hot none)
PCOL = [f"p{c}" for c in range(6)]
P2 = np.vstack([np.tile(NONE_P, (fc.L - 1, 1)), g2[PCOL].to_numpy(float)])
assert P2.shape == (len(ft2), 6) and (ft2["frame_id"].to_numpy()[fc.L - 1:] == g2["frame_id"].to_numpy()).all()
TT2 = ft2["t"].to_numpy(float)
FID2 = ft2["frame_id"].to_numpy()
RAW2 = LAB[S02]
DEB2 = ac.debounce_labels(RAW2)
CL = C.to_dict("records")
for c in CL:
    assert c["start_idx"] >= fc.L - 1
    c.update(E.e1_inputs(DEB2, P2, int(c["start_idx"]), int(c["end_idx"])))
    assert tuple(v.box for v in c["vis"]) == tuple(c["seq"]), "E1 visits differ from Phase A visits"
    # isolated re-debounce of the cycle equals the session-wide debounce (used by every synthetic composite)
    assert (ac.debounce_labels(RAW2[c["start_idx"]:c["end_idx"]]) == DEB2[c["start_idx"]:c["end_idx"]]).all()
print("E1 visits == Phase A visits for all 51 cycles; isolated re-debounce == session-wide debounce for all 51 cycles")

# ================================================================== 1. normal side (section 3)
hr("1. Normal side: route, mu, calibration, S_E, p_E, D3' (51 s02 normal cycles, Phase A roles)")
RE, FINFO = E.score_roles(CL, frame_ids_by_session={S02: FID2})
RE = RE.merge(SA[KEY + ["D0", "D1", "D2", "d2_nref", "p1", "p2", "p3"]], on=KEY, how="left", validate="1:1")
RE["D3p"] = np.where(RE["role"] == "test", ac.d3(RE["p_E"].fillna(1.0), RE["p2"].fillna(1.0)), np.nan)
# plain variant on the same cycles (only for the synthetic tabulation)
FOLD = {}
for fi in FINFO:
    f = fi["f"]
    ro = fc.anom_roles(f)
    tr, ca, te = ac._split_roles(CL, ro)
    route = tuple(int(x) for x in fi["route"].split("-"))
    mu = {int(k.split("_")[1]): v for k, v in fi.items() if k.startswith("mu_")}
    par = ac.fit_params(tr)
    cal_rows = RE[(RE["f"] == f) & (RE["role"] == "cal")].set_index("cycle_idx")
    cal_ids = [int(c["cycle_idx"]) for c in ca]
    cal_SE = cal_rows.loc[cal_ids, "S_E"].to_numpy(float)
    cal_SEp = np.array([E.align(c["vis_c"], route, mu, multimatch=False)[0] for c in ca])
    sc = pd.DataFrame([ac.score_cycle(c, par) for c in ca])
    ref = SA[(SA["f"] == f) & (SA["role"] == "cal")].set_index("cycle_idx").loc[cal_ids]
    assert np.allclose(sc["D1"].to_numpy(), ref["D1"].to_numpy(), atol=1e-12, rtol=0)
    assert np.allclose(sc["D2"].to_numpy(), ref["D2"].to_numpy(), atol=1e-12, rtol=0)
    FOLD[f] = SimpleNamespace(f=f, route=route, mu=mu, par=par, cal_SE=cal_SE, cal_SEp=cal_SEp,
                              cal_D1=sc["D1"].to_numpy(float), cal_D2=sc["D2"].to_numpy(float), n_cal=len(ca),
                              n_train=len(tr), cal_ids=cal_ids)
    fi["cal_max_plain"] = float(cal_SEp.max())
FOLDS = pd.DataFrame([{k: v for k, v in fi.items() if k != "train_readings"} | {"train_readings": ", ".join(
    f"{'-'.join(map(str, k))} x{v}" for k, v in sorted(fi["train_readings"].items(), key=lambda kv: -kv[1]))} for fi in FINFO])
print(FOLDS.to_string(index=False))

TN = RE[RE["role"] == "test"].sort_values("cycle_idx").reset_index(drop=True)
assert len(TN) == 51 and TN["cycle_idx"].nunique() == 51 and not TN["natural_dev"].any()
pos = TN[TN["S_E"] > 0]
n_zero = int((TN["S_E"] == 0).sum())
print(f"\n51 normal test S_E: {n_zero} x exactly 0; positive: "
      + "; ".join(f"cycle {int(r.cycle_idx)} (f{int(r.f)}, {r.reading}) S_E = {r.S_E:.4f}" for r in pos.itertuples()))
PHA = CS[(CS["role"] == "test")].copy()
SET_A = sorted(PHA.loc[alarm(PHA["p3"], ALPHA), "cycle_idx"].astype(int))
SET_E = sorted(TN.loc[alarm(TN["D3p"], ALPHA), "cycle_idx"].astype(int))
SET_PE10 = sorted(TN.loc[alarm(TN["p_E"], ALPHA), "cycle_idx"].astype(int))
SET_PE05 = sorted(TN.loc[alarm(TN["p_E"], 0.05), "cycle_idx"].astype(int))
SE_FAR = float(np.sqrt(ALPHA * (1 - ALPHA) / 51))
NORMAL = dict(n=51, n_zero=n_zero, positive=[(int(r.cycle_idx), float(r.S_E)) for r in pos.itertuples()],
              far_D3p=len(SET_E) / 51, far_pE10=len(SET_PE10) / 51, far_pE05=len(SET_PE05) / 51, far_D3=len(SET_A) / 51,
              set_D3=SET_A, set_D3p=SET_E, set_pE10=SET_PE10, se=SE_FAR)
print(f"FAR p_E <= 0.10: {len(SET_PE10)}/51 = {len(SET_PE10) / 51:.3f}  cycles {SET_PE10}")
print(f"FAR p_E <= 0.05: {len(SET_PE05)}/51 = {len(SET_PE05) / 51:.3f}  cycles {SET_PE05}")
print(f"FAR D3'  <= 0.10: {len(SET_E)}/51 = {len(SET_E) / 51:.3f}  cycles {SET_E}")
print(f"FAR Phase A D3 (loaded table) <= 0.10: {len(SET_A)}/51 = {len(SET_A) / 51:.3f}  cycles {SET_A}")
print(f"binomial SE at alpha 0.10, n = 51: {SE_FAR:.3f}; 2SE band [{ALPHA - 2 * SE_FAR:.3f}, {ALPHA + 2 * SE_FAR:.3f}]")
print(f"D3' alarm set == Phase A D3 alarm set: {SET_E == SET_A}")
ALARM_ROWS = TN[TN["cycle_idx"].isin(sorted(set(SET_A) | set(SET_E) | set(SET_PE10)))][
    ["cycle_idx", "f", "reading", "reading_c", "S_E", "p_E", "p1", "p2", "p3", "D3p", "explain"]]
ALARM_ROWS = ALARM_ROWS.assign(path=np.where(alarm(2 * ALARM_ROWS["p_E"], ALPHA) & alarm(2 * ALARM_ROWS["p2"], ALPHA), "both",
                                             np.where(alarm(2 * ALARM_ROWS["p_E"], ALPHA), "p_E", np.where(alarm(2 * ALARM_ROWS["p2"], ALPHA), "p_2", "-"))))
print(ALARM_ROWS.to_string(index=False))

# ================================================================== 2. synthetic battery (section 4)
hr("2. Synthetic battery: material")


class Seg(NamedTuple):
    lab: np.ndarray
    P: np.ndarray
    dur: np.ndarray


def compose(*segs):
    lab = np.concatenate([s.lab for s in segs]).astype(int)
    P = np.vstack([s.P for s in segs])
    dur = np.concatenate([s.dur for s in segs]).astype(float)
    assert len(lab) == len(P) == len(dur) and len(lab) > 0
    return lab, P, np.r_[0.0, np.cumsum(dur)]


def score_composite(lab, P, t_ext, F):
    """Debounce the composite, E1 (frozen + plain), Phase A D1/D2 recomputed from its own labels and times,
    p-values against the parent fold's unmodified calibration cycles."""
    deb = ac.debounce_labels(lab)
    vs = E.visits_with_mass(deb, P)
    vc = E.collapse(vs, lab=deb)
    S, ops = E.align(vc, F.route, F.mu, multimatch=True)
    Sp = E.align(vc, F.route, F.mu, multimatch=False)[0]
    vt = ac.visits_of(np.r_[deb, 0], t_ext)          # extra none frame -> a visit ending at the cycle end uses t_ext[n]
    seq, dw, gp = ac.cycle_from(vt)
    assert tuple(seq) == tuple(v.box for v in vs)
    sc = ac.score_cycle(dict(seq=seq, dw=dw, gp=gp, dur_s=float(t_ext[-1] - t_ext[0])), F.par)
    pE = float(ac.conf_p(F.cal_SE, [S])[0])
    pEp = float(ac.conf_p(F.cal_SEp, [Sp])[0])
    p1 = float(ac.conf_p(F.cal_D1, [sc["D1"]])[0])
    p2 = float(ac.conf_p(F.cal_D2, [sc["D2"]])[0])
    off = [v for v in vc if v.box not in F.route]
    return dict(n_frames=len(lab), reading=E.reading(vs), reading_c=E.reading(vc), S_E=S, S_plain=Sp,
                offroute_mass=float(sum(v.mass for v in off)), offroute_frames=int(sum(v.end - v.start for v in off)),
                D1=sc["D1"], D2=sc["D2"], d2_nref=sc["d2_nref"], p_E=pE, p_E_plain=pEp, p1=p1, p2=p2,
                D3=float(ac.d3(p1, p2)), D3p=float(ac.d3(pE, p2)), D3p_plain=float(ac.d3(pEp, p2)),
                boxes=set(v.box for v in vs), explain=E.explain(S, ops, vc, F.route, F.mu))


# ---- parents
NCYC = {int(c["cycle_idx"]): c for c in CL}
PARENTS = []
for r in TN.itertuples():
    c = NCYC[int(r.cycle_idx)]
    i0, i1 = int(c["start_idx"]), int(c["end_idx"])
    rec = dict(cycle_idx=int(r.cycle_idx), f=int(r.f), i0=i0, i1=i1, n=i1 - i0, raw=RAW2[i0:i1].copy(), P=P2[i0:i1].copy(),
               dur=np.diff(TT2[i0:i1 + 1]), reading=r.reading, reading_c=r.reading_c,
               blocks=[(v.box, v.start - i0, v.end - i0) for v in c["vis_c"]])
    rec["max_dt"] = float(rec["dur"].max())
    if r.reading_c == "1-2-5":
        (b1, s1_, e1_), (b2, s2, e2), (b5, s5, e5) = rec["blocks"]
        assert (b1, b2, b5) == (1, 2, 5) and 0 == s1_ < e1_ <= s2 < e2 <= s5 < e5 <= rec["n"]
        rec.update(b1=(s1_, e1_), b2=(s2, e2), b5=(s5, e5))
        PARENTS.append(rec)
assert not any(TN["natural_dev"])
print(f"parents (test-role s02 normal cycles with collapsed reading 1-2-5): {len(PARENTS)} of 51; per fold "
      f"{dict(sorted(Counter(p['f'] for p in PARENTS).items()))}; uncollapsed readings {dict(Counter(p['reading'] for p in PARENTS))}")
B2LEN = Counter(p["b2"][1] - p["b2"][0] for p in PARENTS)
print(f"Box-2 block length (frames) over parents: {dict(sorted(B2LEN.items()))}")
print(f"Box-5 block length (frames) over parents: {dict(sorted(Counter(p['b5'][1] - p['b5'][0] for p in PARENTS).items()))}")
print(f"2->5 transit length (frames): {dict(sorted(Counter(p['b5'][0] - p['b2'][1] for p in PARENTS).items()))}")
print(f"1->2 transit length (frames): {dict(sorted(Counter(p['b2'][0] - p['b1'][1] for p in PARENTS).items()))}")
HOLES = []
for p in PARENTS:
    for k in np.flatnonzero(p["dur"] > 0.15):
        where = next((f"Box-{b} block" for b, s, e in p["blocks"] if s <= k < e), "transit")
        HOLES.append(dict(cycle_idx=p["cycle_idx"], f=p["f"], frame_in_cycle=int(k), frame_id=int(FID2[p["i0"] + k]),
                          interval_s=float(p["dur"][k]), where=where))
HOLES = pd.DataFrame(HOLES)
HOLE_SUM = dict(n_intervals_gt_0p15s=len(HOLES), n_parents_with_one=int(HOLES["cycle_idx"].nunique()),
                interval_min_med_max=(round(float(HOLES["interval_s"].min()), 3), round(float(HOLES["interval_s"].median()), 3),
                                      round(float(HOLES["interval_s"].max()), 3)),
                where=dict(Counter(HOLES["where"])))
HOLE_MAX = HOLES.loc[HOLES["interval_s"].idxmax()].to_dict()
print("frame intervals > 0.15 s inside parent cycles (frames without radar points are absent from the frame table):", HOLE_SUM)
print("largest:", HOLE_MAX)

# ---- the real 2-frame Box-3 flicker of the 1-3-2-5 normal cycle
off = [c for c in CL if any(v.box not in (1, 2, 5) for v in c["vis_c"])]
assert len(off) == 1, [c["cycle_idx"] for c in off]
FC = off[0]
fv = [v for v in FC["vis_c"] if v.box not in (1, 2, 5)]
assert len(fv) == 1 and fv[0].box == 3 and fv[0].end - fv[0].start == 2 and E.reading(FC["vis_c"]) == "1-3-2-5"
fv = fv[0]
FLK = Seg(RAW2[fv.start:fv.end].copy(), P2[fv.start:fv.end].copy(), np.diff(TT2[fv.start:fv.end + 1]))
FLK_INFO = dict(cycle_idx=int(FC["cycle_idx"]), frame_ids=(int(FID2[fv.start]), int(FID2[fv.end - 1])), raw=FLK.lab.tolist(),
                p3=[round(float(x), 4) for x in FLK.P[:, 3]], mass=fv.mass)
print(f"flicker block: {FLK_INFO}")
assert abs(fv.mass - FLICKER_EXPECTED) < 0.005

# ---- s01 ToF-true Box-3 / Box-4 runs inside complete loops, with their own B2 OOF labels and posteriors
s1 = oof[oof["session"] == S01].reset_index(drop=True)
Y1, YP1, P1, T1, L1, F1 = (s1["y_true"].to_numpy(int), s1["y_pred"].to_numpy(int), s1[PCOL].to_numpy(float),
                           s1["t"].to_numpy(float), s1["loop_id"].to_numpy(int), s1["frame_id"].to_numpy(int))
COMPLETE = set(AL.loc[(AL["session"] == S01) & AL["is_cycle"], "loop_id"].astype(int))
RUNS, RUN_COUNTS = [], Counter()
for b, s, e in E.label_runs(Y1):
    if b <= 0 or len(set(L1[s:e])) != 1 or int(L1[s]) not in COMPLETE:
        continue
    RUN_COUNTS[int(b)] += 1
    if b in (3, 4):
        assert e < len(Y1)
        sg = Seg(YP1[s:e].copy(), P1[s:e].copy(), T1[s + 1:e + 1] - T1[s:e])
        iso = ac.debounce_labels(sg.lab)
        RUNS.append(dict(run_id=f"s01_b{b}_{len(RUNS):03d}", box=int(b), loop_id=int(L1[s]), first_frame_id=int(F1[s]),
                         last_frame_id=int(F1[e - 1]), n_frames=e - s, mass_true_box=float(P1[s:e, b].sum()),
                         b2_raw="".join(map(str, sg.lab)), block_alone_has_visit=bool((iso == b).any()), seg=sg))
print(f"s01 ToF-true runs inside single complete loops, per box: {dict(sorted(RUN_COUNTS.items()))}")
for b in (3, 4):
    rr = [r for r in RUNS if r["box"] == b]
    ln = np.array([r["n_frames"] for r in rr])
    ms = np.array([r["mass_true_box"] for r in rr])
    print(f"  Box {b}: {len(rr)} runs; length min/median/max {ln.min()}/{np.median(ln):.1f}/{ln.max()} frames; "
          f"B2 mass at Box {b} min/median/max {ms.min():.2f}/{np.median(ms):.2f}/{ms.max():.2f}; "
          f"block alone debounces to a Box-{b} visit: {sum(r['block_alone_has_visit'] for r in rr)}/{len(rr)}")

print("\nDetails the plan leaves open (fixed here, before scoring):")
OPEN = [
    "Blocks = frames [first, last+1) of the parent's collapsed visit (session-wide debounced labels; == isolated re-debounce for all 51 cycles); a composite carries the parent's RAW B2 labels and posteriors and is debounced again as a whole.",
    "Time axis of every non-tempo composite = spliced real frame intervals (each frame keeps t[k+1]-t[k] of its source: s02 parent, s01 run or the s02 flicker frames; inserted none frames 0.1 s). The unmodified parent therefore reproduces Phase A D2 exactly; dropped-frame holes travel with their block.",
    "Insertion points = floor midpoints: foreign at e2 + (s5-e2)//2 (2->5 transit); reach flicker at e1 + (s2-e1)//2 (1->2 transit); mid-dwell flicker and split gaps at s2 + (e2-s2)//2.",
    "swap_1-5-2 = pre + B5 + (2->5 transit) + B2 + post. repeat_2_far = B2 copy inserted right after the last frame of the Box-5 block. repeat_2_immediate = B2 + (copy of the parent's own 2->5 transit) + B2 copy + (2->5 transit) + B5.",
    "subst_b = s01 run replaces B2; foreign_b = s01 run inserted in the 2->5 transit; full cross products (50 x 43 Box-3 runs, 50 x 30 Box-4 runs), no RNG. 'Readable' = the debounced composite has a visit of the pasted box.",
    "truncation keepK = the first K frames of the Box-5 block are kept, the rest of the block is deleted (K = 0, 1, 2, 3).",
    "tempo xs: n' = round(T*s/0.1) frames on a 0.1 s grid (T = parent duration); frame k takes the parent frame whose real start time is nearest to 0.1*k/s (ties -> earlier frame); labels and posteriors copied, then debounced.",
    f"mid-dwell flicker parents: Box-2 block >= {MID_DWELL_MIN} frames (each half keeps >= 3 frames = MIN_BOX + 1); shorter parents are run as a supplementary row outside the acceptance.",
    "split_visit gapG: G none frames (posterior one-hot none, 0.1 s each) inserted at the middle of the Box-2 block.",
    "Every synthetic cycle: parent fold f, route/mu and Phase A parameters fitted on the real training normals (f+3, f+4), calibrated on the real calibration normals (f+1, f+2), conformal ties conservative (a2_cycles.conf_p).",
]
for x in OPEN:
    print("  -", x)

# ---- identity check: every normal cycle rebuilt unmodified as a composite
hr("2a. Identity check (51 normal cycles rebuilt unmodified as composites)")
ID_ROWS = []
for r in TN.itertuples():
    c = NCYC[int(r.cycle_idx)]
    i0, i1 = int(c["start_idx"]), int(c["end_idx"])
    lab, P, t_ext = compose(Seg(RAW2[i0:i1], P2[i0:i1], np.diff(TT2[i0:i1 + 1])))
    o = score_composite(lab, P, t_ext, FOLD[int(r.f)])
    ID_ROWS.append(dict(cycle_idx=int(r.cycle_idx), f=int(r.f), d_S_E=abs(o["S_E"] - r.S_E), d_D1=abs(o["D1"] - r.D1),
                        d_D2=abs(o["D2"] - r.D2), d_p_E=abs(o["p_E"] - r.p_E), d_p1=abs(o["p1"] - r.p1), d_p2=abs(o["p2"] - r.p2),
                        reading_same=o["reading"] == r.reading, flip_D3p=bool(alarm(o["D3p"], ALPHA) != alarm(r.D3p, ALPHA))))
IDC = pd.DataFrame(ID_ROWS)
ID_SUM = {k: float(IDC[k].max()) for k in ("d_S_E", "d_D1", "d_D2", "d_p_E", "d_p1", "d_p2")}
ID_SUM.update(readings_same=int(IDC["reading_same"].sum()), D3p_flips=int(IDC["flip_D3p"].sum()))
print("max |diff| composite vs real:", ID_SUM)
assert ID_SUM["d_S_E"] < 1e-9 and ID_SUM["d_D1"] < 1e-9 and ID_SUM["d_D2"] < 1e-9 and ID_SUM["d_p2"] == 0 and ID_SUM["D3p_flips"] == 0

# ---- battery
hr("2b. Synthetic battery: scoring")
ROWS = []


def add(kind, variant, p, lab, P, t_ext, **extra):
    o = score_composite(lab, P, t_ext, FOLD[p["f"]])
    o.update(kind=kind, variant=variant, parent=p["cycle_idx"], f=p["f"], **extra)
    o["boxes"] = "".join(map(str, sorted(o["boxes"])))
    ROWS.append(o)


def S(p, a, b):
    return Seg(p["raw"][a:b], p["P"][a:b], p["dur"][a:b])


def NONE_SEG(g):
    return Seg(np.zeros(g, dtype=int), np.tile(NONE_P, (g, 1)), np.full(g, DT))


def tempo_composite(p, s):
    tr = np.r_[0.0, np.cumsum(p["dur"])][:-1]            # real start time of every parent frame (relative)
    T = float(p["dur"].sum())
    n2 = int(round(T * s / DT))
    tau = DT * np.arange(n2) / s
    j = np.clip(np.searchsorted(tr, tau), 1, len(tr) - 1)
    j = np.where(np.abs(tr[j - 1] - tau) <= np.abs(tr[j] - tau), j - 1, j)
    return p["raw"][j].astype(int), p["P"][j], DT * np.arange(n2 + 1)


for p in PARENTS:
    n = p["n"]
    (s1_, e1_), (s2, e2), (s5, e5) = p["b1"], p["b2"], p["b5"]
    add("omission_2", "", p, *compose(S(p, 0, s2), S(p, e2, n)))
    add("swap_1-5-2", "", p, *compose(S(p, 0, s2), S(p, s5, e5), S(p, e2, s5), S(p, s2, e2), S(p, e5, n)))
    add("repeat_2_far", "", p, *compose(S(p, 0, e5), S(p, s2, e2), S(p, e5, n)))
    add("repeat_2_immediate", "", p, *compose(S(p, 0, e2), S(p, e2, s5), S(p, s2, e2), S(p, e2, n)), transit_len=s5 - e2)
    mid25 = e2 + (s5 - e2) // 2
    for r in RUNS:
        b = r["box"]
        add(f"subst_{b}", r["run_id"], p, *compose(S(p, 0, s2), r["seg"], S(p, e2, n)))
        add(f"foreign_{b}", r["run_id"], p, *compose(S(p, 0, mid25), r["seg"], S(p, mid25, n)))
    for k in TRUNC_KEEP:
        ke = min(k, e5 - s5)
        add("truncation", f"keep{k}", p, *compose(S(p, 0, s5 + ke), S(p, e5, n)), k_eff=ke,
            kept_raw="".join(map(str, p["raw"][s5:s5 + ke])))
    for sc in TEMPO:
        add("tempo", f"x{sc:.2f}", p, *tempo_composite(p, sc), scale=sc)
    mid12 = e1_ + (s2 - e1_) // 2
    add("flicker_reach", "", p, *compose(S(p, 0, mid12), FLK, S(p, mid12, n)))
    mid2 = s2 + (e2 - s2) // 2
    elig = (e2 - s2) >= MID_DWELL_MIN
    add("flicker_mid_dwell" if elig else "flicker_mid_dwell_short", "", p, *compose(S(p, 0, mid2), FLK, S(p, mid2, n)),
        b2_len=e2 - s2)
    for g in SPLIT_GAPS:
        add("split_visit", f"gap{g}", p, *compose(S(p, 0, mid2), NONE_SEG(g), S(p, mid2, n)), gap=g)

SY = pd.DataFrame(ROWS)
for k in ("transit_len", "k_eff", "scale", "b2_len", "gap"):
    if k not in SY:
        SY[k] = np.nan
SY["a_D3p"] = alarm(SY["D3p"], ALPHA)
SY["a_pE10"] = alarm(SY["p_E"], ALPHA)
SY["a_pE05"] = alarm(SY["p_E"], 0.05)
SY["a_D3"] = alarm(SY["D3"], ALPHA)
SY["a_D1_05"] = alarm(SY["p1"], 0.05)
SY["a_p2_05"] = alarm(SY["p2"], 0.05)
SY["a_D3p_plain"] = alarm(SY["D3p_plain"], ALPHA)
SY["readable"] = [float(str(r.kind).split("_")[-1] in r.boxes) if str(r.kind).startswith(("subst_", "foreign_")) else np.nan
                  for r in SY.itertuples()]
print(f"synthetic cycles scored: {len(SY)}  ({dict(Counter(SY['kind']))})")

KIND_ORDER = ["omission_2", "swap_1-5-2", "repeat_2_far", "repeat_2_immediate", "subst_3", "subst_4", "foreign_3",
              "foreign_4", "truncation", "tempo", "flicker_reach", "flicker_mid_dwell", "flicker_mid_dwell_short",
              "split_visit"]


def top_readings(s, k=4):
    return ", ".join(f"{a} x{b}" for a, b in Counter(s).most_common(k))


def summarize(d, by):
    rows = []
    for key, g in d.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(by, key)) | dict(
            n=len(g), S_E_min=g["S_E"].min(), S_E_med=g["S_E"].median(), S_E_max=g["S_E"].max(),
            plain_min=g["S_plain"].min(), plain_med=g["S_plain"].median(), plain_max=g["S_plain"].max(),
            det_D3p=g["a_D3p"].mean(), det_pE10=g["a_pE10"].mean(), det_pE05=g["a_pE05"].mean(),
            det_D3_phaseA=g["a_D3"].mean(), det_D1_05=g["a_D1_05"].mean(), det_p2_05=g["a_p2_05"].mean(),
            det_D3p_plain=g["a_D3p_plain"].mean(), readings=top_readings(g["reading_c"])))
    return pd.DataFrame(rows)


SY["kind"] = pd.Categorical(SY["kind"], KIND_ORDER, ordered=True)
SY = SY.sort_values(["kind", "variant", "parent"], kind="stable").reset_index(drop=True)
SY["vgroup"] = np.where(SY["kind"].astype(str).str.startswith(("subst_", "foreign_")), "all runs", SY["variant"])
SUM = summarize(SY, ["kind", "vgroup"]).rename(columns={"vgroup": "variant"})
SUM = SUM[SUM["n"] > 0].reset_index(drop=True)
PER_RUN = summarize(SY[SY["kind"].astype(str).str.startswith(("subst_", "foreign_"))], ["kind", "variant"])
PER_RUN = PER_RUN.merge(pd.DataFrame([{k: v for k, v in r.items() if k != "seg"} for r in RUNS]).rename(columns={"run_id": "variant"})
                        [["variant", "n_frames", "mass_true_box", "b2_raw"]], on="variant", how="left")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 70)
hr("2c. Battery summary (detection = share of synthetic cycles alarmed; D3' alarm <= 0.10; p_E / p_1 / p_2 at the stated level)")
print(SUM.drop(columns=["readings"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print()
print(SUM[["kind", "variant", "n", "readings"]].to_string(index=False))

# readable split
hr("2d. Substitution / foreign insertion split by readability (composite has a visit of the pasted box)")
RD = summarize(SY[SY["kind"].astype(str).str.startswith(("subst_", "foreign_"))], ["kind", "readable"])
print(RD.drop(columns=["plain_min", "plain_med", "plain_max"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
RD_RUN = (SY[SY["kind"].astype(str).str.startswith("subst_")].groupby(["kind", "variant"], observed=True)["readable"]
          .agg(["mean", "size"]).reset_index())
print("\nsubst runs readable in all / no / some parents:",
      {k: (int((g["mean"] == 1).sum()), int((g["mean"] == 0).sum()), int(((g["mean"] > 0) & (g["mean"] < 1)).sum()))
       for k, g in RD_RUN.groupby("kind", observed=True)})

# per-fold tables
hr("2e. Per-fold detection (D3') and minimum S_E for kinds 1-3, 5, 6; D1 (p_1 <= 0.05) per fold")
PF_KINDS = ["omission_2", "swap_1-5-2", "repeat_2_far", "subst_3", "subst_4", "foreign_3", "foreign_4"]
PFT = (SY[SY["kind"].isin(PF_KINDS)].groupby(["kind", "f"], observed=True)
       .agg(n=("S_E", "size"), det_D3p=("a_D3p", "mean"), det_D1_05=("a_D1_05", "mean"), S_E_min=("S_E", "min"),
            S_E_med=("S_E", "median")).reset_index())
print(PFT.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# specific checks
hr("2f. Kind-specific checks")
CHK = {}
om = SY[SY["kind"] == "omission_2"]
mu2 = om["f"].map({f: FOLD[f].mu[2] for f in FOLD})
CHK["omission_reading_1-5"] = int((om["reading_c"] == "1-5").sum())
CHK["omission_S_E_eq_mu2"] = int((np.abs(om["S_E"] - mu2) < 1e-9).sum())
tr_ = SY[SY["kind"] == "truncation"]
no5 = tr_[~tr_["reading_c"].str.contains("5")]
mu5 = no5["f"].map({f: FOLD[f].mu[5] for f in FOLD})
CHK["truncation_lacking_box5"] = int(len(no5))
CHK["truncation_lacking_box5_S_E_eq_mu5"] = int((np.abs(no5["S_E"] - mu5) < 1e-9).sum())
CHK["truncation_lacking_box5_det_D3p"] = float(no5["a_D3p"].mean()) if len(no5) else np.nan
fr = SY[SY["kind"] == "flicker_reach"]
CHK["flicker_reach_S_E_eq_flicker_mass"] = int((np.abs(fr["S_E"] - FLK_INFO["mass"]) < 1e-9).sum())
CHK["flicker_reach_alarm_D3p_by_fold"] = {int(f): float(g["a_D3p"].mean()) for f, g in fr.groupby("f")}
CHK["flicker_reach_alarm_pE05_by_fold"] = {int(f): float(g["a_pE05"].mean()) for f, g in fr.groupby("f")}
CHK["flicker_reach_alarm_pE10_by_fold"] = {int(f): float(g["a_pE10"].mean()) for f, g in fr.groupby("f")}
CHK["flicker_reach_D3p_alarms_via_p2_only"] = int((fr["a_D3p"] & ~alarm(2 * fr["p_E"], ALPHA)).sum())
md_ = SY[SY["kind"] == "flicker_mid_dwell"]
CHK["mid_dwell_n"] = int(len(md_))
CHK["mid_dwell_readings"] = dict(Counter(md_["reading"]))
CHK["mid_dwell_S_E_values"] = dict(Counter(np.round(md_["S_E"], 4)))
CHK["mid_dwell_S_E_max"] = float(md_["S_E"].max())
CHK["mid_dwell_S_E_eq_offroute_visit_mass"] = f"{int((np.abs(md_['S_E'] - md_['offroute_mass']) < 1e-9).sum())}/{len(md_)}"
CHK["mid_dwell_offroute_visit_frames"] = dict(Counter(md_["offroute_frames"]))
CHK["mid_dwell_plain_min_med_max"] = (float(md_["S_plain"].min()), float(md_["S_plain"].median()), float(md_["S_plain"].max()))
CHK["mid_dwell_alarm_D3p_by_fold"] = {int(f): float(g["a_D3p"].mean()) for f, g in md_.groupby("f")}
# diagnosis of any mid-dwell flicker scoring above the acceptance bound (normal data only)
MD_FAIL = []
for r in md_[md_["S_E"] > 1.2].itertuples():
    p = next(q for q in PARENTS if q["cycle_idx"] == r.parent)
    s2, e2 = p["b2"]
    mid2 = s2 + (e2 - s2) // 2
    MD_FAIL.append(dict(parent=int(r.parent), f=int(r.f), b2_block_raw="".join(map(str, p["raw"][s2:e2])),
                        b2_block_p3=" ".join(f"{x:.2f}" for x in p["P"][s2:e2, 3]), insert_after_block_frame=mid2 - s2 - 1,
                        composite_box3_frames=int(r.offroute_frames), S_E=float(r.S_E), S_plain=float(r.S_plain),
                        explain=r.explain))
CHK["mid_dwell_above_1.2"] = MD_FAIL
ri = SY[SY["kind"] == "repeat_2_immediate"]
CHK["repeat_immediate_transit_lt2"] = int((ri["transit_len"] < 2).sum())
CHK["repeat_immediate_S_E_zero"] = int((ri["S_E"] == 0).sum())
sv = SY[SY["kind"] == "split_visit"]
CHK["split_visit_S_E_zero"] = f"{int((sv['S_E'] == 0).sum())}/{len(sv)}"
for k, v in CHK.items():
    print(f"  {k}: {v}")

TRU = summarize(SY[SY["kind"] == "truncation"].assign(lacks5=lambda d: ~d["reading_c"].str.contains("5")), ["variant", "lacks5"])
print("\ntruncation by kept frames and whether the reading lacks Box 5:")
print(TRU.drop(columns=["plain_min", "plain_med", "plain_max"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
TEM = SY[SY["kind"] == "tempo"].groupby("variant").agg(
    n=("S_E", "size"), S_E_med=("S_E", "median"), S_E_max=("S_E", "max"), seq_ch_pE05=("a_pE05", "mean"),
    det_D3p=("a_D3p", "mean"), det_D3_phaseA=("a_D3", "mean"), det_D2_p2_05=("a_p2_05", "mean"),
    det_D1_05=("a_D1_05", "mean"), readings=("reading_c", top_readings)).reset_index()
print("\ntempo scaling (D1/D2 recomputed on the scaled, re-debounced cycles):")
print(TEM.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
HOLE_TEMPO = SY[(SY["kind"] == "tempo") & (SY["parent"] == int(HOLE_MAX["cycle_idx"]))][
    ["parent", "variant", "n_frames", "reading_c", "S_E", "D2", "p2", "D3", "D3p"]]
print(f"\ntempo rows of the parent with the largest frame interval ({HOLE_MAX['interval_s']:.2f} s, cycle {int(HOLE_MAX['cycle_idx'])}):")
print(HOLE_TEMPO.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
SPL = summarize(SY[SY["kind"] == "split_visit"], ["variant"])
print("\nsplit visit (negative control; D2 is the only channel that can react):")
print(SPL[["variant", "n", "S_E_max", "det_D3p", "det_pE10", "det_p2_05", "det_D3_phaseA", "readings"]].to_string(
    index=False, float_format=lambda x: f"{x:.3f}"))

# D1 exhibit
hr("2g. D1 exhibit: Phase A D1 (p_1 <= 0.05, the D3 path) vs E1 (p_E <= 0.05) on the same synthetic cycles")
EX = SY[SY["kind"].isin(PF_KINDS)].groupby("kind", observed=True).agg(
    n=("S_E", "size"), E1_pE05=("a_pE05", "mean"), E1_D3p=("a_D3p", "mean"), D1_p1_05=("a_D1_05", "mean"),
    PhaseA_D3=("a_D3", "mean")).reset_index()
print(EX.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
D1PF = {k: {int(f): round(float(g["a_D1_05"].mean()), 3) for f, g in SY[SY["kind"] == k].groupby("f")}
        for k in ("subst_4", "omission_2", "subst_3")}
print("D1 detection per fold:", D1PF)
D1VAL = {k: {int(f): sorted(set(np.round(g["D1"], 3))) [:4] for f, g in SY[SY["kind"] == k].groupby("f")} for k in ("subst_4", "omission_2")}
print("D1 values per fold (first 4 distinct):", D1VAL)
print("calibration D1 max per fold:", {f: round(float(FOLD[f].cal_D1.max()), 3) for f in FOLD})

# ================================================================== 3. acceptance
hr("3. Acceptance items (stated before the battery was run; DESIGN_E section 4)")
det = {k: float(SY.loc[SY["kind"] == k, "a_D3p"].mean()) for k in ("omission_2", "swap_1-5-2", "repeat_2_far", "subst_4", "foreign_4")}
s3r = SY[(SY["kind"] == "subst_3") & (SY["readable"] == True)]  # noqa: E712
nz = TN[TN["S_E"] > 0]
d1_s4 = float(SY.loc[SY["kind"] == "subst_4", "a_D1_05"].mean())
d1_om = float(SY.loc[SY["kind"] == "omission_2", "a_D1_05"].mean())
ACC = [
    dict(item="A1 detection >= 0.95 at D3' for omission, swap, far repeat, subst_4, foreign_4",
         value="; ".join(f"{k} {v:.3f}" for k, v in det.items()), passed=all(v >= 0.95 for v in det.values())),
    dict(item="A2 subst_3 readable-conditional detection >= 0.95 (D3')",
         value=f"{s3r['a_D3p'].mean():.3f} (n = {len(s3r)} readable of {int((SY['kind'] == 'subst_3').sum())})",
         passed=bool(len(s3r)) and s3r["a_D3p"].mean() >= 0.95),
    dict(item=f"A3 mid-dwell flicker multi-match S_E <= 1.2 in every parent (Box-2 block >= {MID_DWELL_MIN} frames)",
         value=f"max {md_['S_E'].max():.4f} over n = {len(md_)}", passed=bool(len(md_)) and md_["S_E"].max() <= 1.2),
    dict(item="A4a the 51 real normal S_E are 50 x 0 and 1 x 1.09 (cycle 19)",
         value=f"{n_zero} x 0; positive: " + ", ".join(f"cycle {int(r.cycle_idx)} = {r.S_E:.4f}" for r in nz.itertuples()),
         passed=n_zero == 50 and len(nz) == 1 and int(nz["cycle_idx"].iloc[0]) == 19 and abs(nz["S_E"].iloc[0] - FLICKER_EXPECTED) < 0.005),
    dict(item="A4b D3' normal alarm set equals the Phase A D3 set",
         value=f"D3' {SET_E} vs Phase A D3 {SET_A}", passed=SET_E == SET_A),
    dict(item="A5 unit cases of section 2.4 pass", value=f"{E.UNIT_TESTS_PASSED} asserted DP cases (+ collapse/traceback/plain asserts) at import",
         passed=E.UNIT_TESTS_PASSED >= 9),
    dict(item="A6 D1 exhibit reproduces the tie mechanism (D1 detection < 0.5 on subst_4 or omission)",
         value=f"subst_4 {d1_s4:.3f}; omission {d1_om:.3f}", passed=(d1_s4 < 0.5) or (d1_om < 0.5)),
]
ACCD = pd.DataFrame(ACC)
for a in ACC:
    print(f"  [{'PASS' if a['passed'] else 'FAIL'}] {a['item']}: {a['value']}")
ALL_PASS = bool(ACCD["passed"].all())
print(f"\nALL ACCEPTANCE ITEMS PASS: {ALL_PASS}")

# ---- smoke test of the one-shot tabulation helper (e1_core.oneshot_tables) on s02 normals + relabelled synthetic rows
SMOKE_MAP = {("omission_2", ""): "seq135", ("subst_4", None): "seq145", ("tempo", "x0.60"): "fast", ("tempo", "x1.70"): "slow"}
parts = [TN.assign(atype="normal", natural_dev=False)[["session", "cycle_idx", "f", "atype", "natural_dev", "reading_c", "S_E", "p_E", "p2", "p3", "D3p"]]]
expect = {}
for (k, v), at in SMOKE_MAP.items():
    g = SY[(SY["kind"] == k) & ((SY["variant"] == v) if v is not None else True)]
    expect[at] = (float(g["a_D3p"].mean()), float(g["a_D3"].mean()), len(g))
    parts.append(pd.DataFrame(dict(session="synthetic_" + at, cycle_idx=np.arange(len(g)), f=g["f"].to_numpy(), atype=at,
                                   natural_dev=False, reading_c=g["reading_c"].to_numpy(), S_E=g["S_E"].to_numpy(),
                                   p_E=g["p_E"].to_numpy(), p2=g["p2"].to_numpy(), p3=g["D3"].to_numpy(), D3p=g["D3p"].to_numpy())))
SMOKE_DF = pd.concat(parts, ignore_index=True)
# natural_dev handling (fix 2026-11-06, verifier finding B1): anomaly-session natural_dev rows must stay in the rates,
# a normal natural_dev row must be dropped (count only). Flag some stand-in seq145 rows and add one synthetic normal row.
_ix = SMOKE_DF.index[SMOKE_DF["atype"] == "seq145"][:7]
SMOKE_DF.loc[_ix, "natural_dev"] = True
_fake = SMOKE_DF[SMOKE_DF["atype"] == "normal"].iloc[[0]].assign(session="synthetic_normal_natdev", natural_dev=True)
SMK = E.oneshot_tables(pd.concat([SMOKE_DF, _fake], ignore_index=True), ALPHA)
assert SMK["n_natural_dev_dropped"] == 1, "exactly the one normal natural_dev row must be dropped"
assert int(SMK["rows"]["natural_dev"].astype(bool).sum()) == 7, "anomaly-session natural_dev rows must stay"
rt = SMK["rates"].set_index("atype")
assert abs(rt.loc["normal", "E1_D3p"] - len(SET_E) / 51) < 1e-12 and abs(rt.loc["normal", "phaseA_D3"] - len(SET_A) / 51) < 1e-12
assert SMK["normal_alarm_sets"] == dict(phaseA_D3=SET_A, E1_D3p=SET_E)
for at, (dp, d3, nn) in expect.items():
    assert abs(rt.loc[at, "E1_D3p"] - dp) < 1e-12 and abs(rt.loc[at, "phaseA_D3"] - d3) < 1e-12 and rt.loc[at, "n_pairs"] == nn, at
SMOKE_LABEL = ", ".join(f"{k[0]}{(' ' + k[1]) if k[1] else ''} -> {v}" for k, v in SMOKE_MAP.items())
print(f"smoke test of e1_core.oneshot_tables (51 normals + synthetic rows relabelled: {SMOKE_LABEL}): passed")
print(SMK["rates"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(SMK["auc_sum"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ================================================================== 4. outputs
hr("4. Outputs")
assert not TRIPPED, TRIPPED
assert fc._frame_table.cache_info().currsize == 0 and fc._anom_loops.cache_info().currsize == 0
assert fc._windows.cache_info().currsize == 0 and ac._default_labels.cache_info().currsize == 0
for nm, d in (("OOF", oof), ("folds_anom", AL), ("04_cycle_scores", CS), ("cycles", C), ("Phase A rows", SA), ("E1 rows", RE)):
    assert set(d["session"].unique()) <= set(KEEP), nm
SESSIONS_IN_MEMORY = sorted(set(oof["session"]) | set(AL["session"]) | set(CS["session"]) | set(C["session"]))
print("sessions in memory at the end:", SESSIONS_IN_MEMORY, "| tripwires fired:", TRIPPED,
      "| fa_common frame-table cache size:", fc._frame_table.cache_info().currsize)

RUNS_DF = pd.DataFrame([{k: v for k, v in r.items() if k != "seg"} for r in RUNS])
RE_OUT = RE.drop(columns=["natural_dev"])
paths = {
    "normal": OUT / "01_normal_rows.csv", "folds": OUT / "01_folds.csv", "synthetic": OUT / "01_synthetic_cycles.csv.gz",
    "summary": OUT / "01_synthetic_summary.csv", "acceptance": OUT / "01_acceptance.csv", "s01_runs": OUT / "01_s01_runs.csv",
    "identity": OUT / "01_identity_check.csv", "perfold": OUT / "01_synthetic_perfold.csv", "readable": OUT / "01_readable_split.csv",
    "tempo": OUT / "01_tempo.csv", "per_run": OUT / "01_synthetic_per_run.csv", "holes": OUT / "01_frame_intervals.csv",
    "md": OUT / "01_synthetic.md",
}
PER_RUN.to_csv(paths["per_run"], index=False)
HOLES.to_csv(paths["holes"], index=False)
s3 = SY[SY["kind"] == "subst_3"]
PRELIM = pd.DataFrame([
    ("omission_2 S_E min / D3' det", "7.86 / 1.000", f"{om['S_E'].min():.3f} / {om['a_D3p'].mean():.3f}"),
    ("swap_1-5-2 S_E min", "9.30", f"{SY.loc[SY['kind'] == 'swap_1-5-2', 'S_E'].min():.3f}"),
    ("repeat_2_far S_E min", "3.06", f"{SY.loc[SY['kind'] == 'repeat_2_far', 'S_E'].min():.3f}"),
    ("subst_3 p_E<=0.05 overall / readable", "0.656 (500 sampled pairs) / 327 of 327",
     f"{s3['a_pE05'].mean():.3f} (all {len(s3)} pairs) / {int(s3r['a_pE05'].sum())} of {len(s3r)}"),
    ("subst_4 p_E<=0.05", "500/500", f"{int(SY.loc[SY['kind'] == 'subst_4', 'a_pE05'].sum())}/{int((SY['kind'] == 'subst_4').sum())}"),
    ("foreign_4 S_E min", "1.95", f"{SY.loc[SY['kind'] == 'foreign_4', 'S_E'].min():.3f}"),
    ("reach flicker S_E", "1.09", f"{fr['S_E'].min():.4f}-{fr['S_E'].max():.4f}"),
    ("mid-dwell flicker multi-match / plain", "1.09 every parent / 2.29-8.70 (n = 46)",
     f"{md_['S_E'].min():.2f}-{md_['S_E'].max():.2f} / {md_['S_plain'].min():.2f}-{md_['S_plain'].max():.2f} (n = {len(md_)})"),
    ("D1 p_1<=0.05: subst_4 / subst_3 / omission / swap / repeat / foreign_4", "0.354 / 0.126 / 0.360 / 0.800 / 0.800 / 0.800",
     " / ".join(f"{SY.loc[SY['kind'] == k, 'a_D1_05'].mean():.3f}" for k in ("subst_4", "subst_3", "omission_2", "swap_1-5-2", "repeat_2_far", "foreign_4"))),
], columns=["quantity", "preliminary (workflow logs)", "this run"])
print("\npreliminary vs this run:")
print(PRELIM.to_string(index=False))
RE_OUT.to_csv(paths["normal"], index=False)
FOLDS.to_csv(paths["folds"], index=False)
SY.to_csv(paths["synthetic"], index=False, compression="gzip")
SUM.to_csv(paths["summary"], index=False)
ACCD.to_csv(paths["acceptance"], index=False)
RUNS_DF.to_csv(paths["s01_runs"], index=False)
IDC.to_csv(paths["identity"], index=False)
PFT.to_csv(paths["perfold"], index=False)
RD.to_csv(paths["readable"], index=False)
TEM.to_csv(paths["tempo"], index=False)

# ---- markdown report
now = _dt.datetime.now().astimezone()
src_sha = {p.name: E.file_sha256(p) for p in (ROOT / "faz_e" / "e1_core.py", Path(__file__).resolve(), ROOT / "faz_a2" / "a2_cycles.py",
                                               ROOT / "faz_a" / "fa_common.py")}
in_sha = {k: E.file_sha256(v) for k, v in INPUTS.items()}
L = []
w = L.append
w("# Phase E - E1 normal side and synthetic battery (secondary analysis) [S]")
w("")
w(f"**Written:** {now:%Y-%m-%d %H:%M %Z} by `faz_e/01_e1_synthetic.py` (run time {time.time() - T_START:.0f} s, CPU). "
  f"**Plan:** `faz_e/DESIGN_E.md`, frozen part sha256 `{DESIGN_SHA}` (equals the value logged in `paper/review/CHANGELOG_A4.md`, item 12).")
w("")
w("**Status.** Secondary analysis, planned after the Phase A-D results were known (DESIGN_E section 1). The Phase A rule "
  "(D3, B2 x pred, alpha = 0.10) stays the planned primary result. This report covers sections 3 and 4 of the plan: "
  "normal-side quantities and the synthetic battery, computed on **s01 and s02 only**. No session s03-s06 was loaded "
  "(OOF / fold / score tables filtered to s01+s02 at load, chunk by chunk; the all-session frame table of `fa_common` "
  "was never built; every raw-data loader was replaced by a tripwire, none fired). The one-shot script "
  "`faz_e/02_e1_oneshot.py` was written but **not run**.")
w("")
w(f"**Verdict:** all acceptance items of section 4 {'PASS' if ALL_PASS else 'do NOT all pass (see section 5)'}.")
w("")
w("## 1. Rule (DESIGN_E section 2-3, as implemented)")
w("- Visits: maximal runs of the debounced B2 label (Phase A debouncer MIN_BOX 2, MAX_GAP 1); evidence E = sum of p_b over the run. "
  "Collapse: consecutive same-box visits (only none frames between) merge, masses add.")
w("- Per fold f: route R = modal collapsed reading of the training normals (folds f+3, f+4); mu_b = mean whole-cycle posterior mass of box b over them.")
w("- S_E = multi-match evidence-weighted edit distance (insertion = visit mass, deletion = mu of the step, an open step absorbs later visits of its own box at zero cost). "
  "p_E = (1 + #{cal >= s - 1e-12}) / (n_cal + 1) on the calibration normals (folds f+1, f+2). D3' = min{1, 2 min(p_E, p_2)}, alarm if D3' <= 0.10; p_2 = unchanged Phase A D2 p-value.")
w(f"- Unit cases of section 2.4: {E.UNIT_TESTS_PASSED} DP cases asserted at import of `faz_e/e1_core.py` (masses chosen so every table row is the minimising path; "
  "a second, realistic-mass set asserts the general identities `1-5-2` = min(mu_2 + E_2, mu_5 + E_5) and `1-2-5-2` = min(E_2, E_5 + mu_5)), plus collapse, traceback and plain-variant asserts.")
w("")
w("## 2. Phase A reproduction gate")
w(f"The 51 s02 normal B2 x pred cycles were rebuilt with `faz_a2/a2_cycles.py` (`labels_from_oof`, `build_cycles`, `score_group`) from the s01/s02 OOF rows and compared with "
  f"`reports/faz_a/04_cycle_scores.csv.gz` (b2, pred, s02, normal): {gate['n_matched']} of {gate['n_ref']} (cycle, f, role) rows matched; "
  f"start/end frame ids and visit readings identical; max |diff| D0 {gate['maxdiff_D0']:.1e}, D1 {gate['maxdiff_D1']:.1e}, D2 {gate['maxdiff_D2']:.1e}, "
  f"p0-p3 {max(gate['maxdiff_p0'], gate['maxdiff_p1'], gate['maxdiff_p2'], gate['maxdiff_p3']):.1e}. **Gate passed.** "
  "E1 visits equal the Phase A visits for all 51 cycles, and re-debouncing a cycle in isolation equals the session-wide debounce for all 51 (this is what the composites rely on). "
  "natural_dev cycles in the B2 x pred normal pool: 0.")
w("")
w("## 3. Normal side (section 3)")
w("Per fold (route, mu and calibration from real s02 normal cycles; 'plain' = the pre-fix collapse rule, tabulated only):")
w("")
w(md_table(FOLDS[["f", "route", "mu_1", "mu_2", "mu_5", "n_train", "n_cal", "n_test", "cal_max", "n_cal_pos", "cal_max_plain", "train_readings"]], 3))
w("")
w(f"- 51 normal test S_E: **{n_zero} x 0**; positive: " + ", ".join(f"cycle {int(r.cycle_idx)} (f{int(r.f)}, reading {r.reading}) **{r.S_E:.4f}**" for r in pos.itertuples()) + ".")
w(f"- FAR p_E <= 0.10: {len(SET_PE10)}/51 = {len(SET_PE10) / 51:.3f} (cycles {SET_PE10}); p_E <= 0.05: {len(SET_PE05)}/51.")
w(f"- FAR **D3' <= 0.10: {len(SET_E)}/51 = {len(SET_E) / 51:.3f}** (cycles {SET_E}); Phase A D3 (loaded table): {len(SET_A)}/51 = {len(SET_A) / 51:.3f} (cycles {SET_A}). "
  f"Binomial SE at alpha 0.10 and n = 51: {SE_FAR:.3f} (2SE band {ALPHA - 2 * SE_FAR:.3f}-{ALPHA + 2 * SE_FAR:.3f}). **Same alarm set: {SET_E == SET_A}.**")
w("")
w(md_table(ALARM_ROWS[["cycle_idx", "f", "reading", "S_E", "p_E", "p1", "p2", "p3", "D3p", "path"]].rename(columns={"p3": "D3 (Phase A)", "D3p": "D3'"}), 3))
w("")
w("The decision layer is degenerate, as the plan states: 50/51 normal scores are exactly 0 and the calibration maximum is 0 in folds 1-3, "
  "so there any collapsed off-route visit alarms; the one positive score is the 0.2 s reach flicker of cycle 19.")
w("")
w("## 4. Synthetic battery (section 4)")
w(f"**Material.** Parents: {len(PARENTS)} s02 normal cycles in their Phase A test role with collapsed reading 1-2-5 "
  f"(per fold {dict(sorted(Counter(p['f'] for p in PARENTS).items()))}). Foreign runs: s01 ToF-true runs inside single complete loops "
  f"(counts per box {dict(sorted(RUN_COUNTS.items()))}; used: {sum(r['box'] == 3 for r in RUNS)} Box-3 and {sum(r['box'] == 4 for r in RUNS)} Box-4 runs) "
  f"with their own B2 S1 OOF labels and posteriors. Flicker block: cycle {FLK_INFO['cycle_idx']}, frames {FLK_INFO['frame_ids'][0]}-{FLK_INFO['frame_ids'][1]}, "
  f"raw {FLK_INFO['raw']}, p3 {FLK_INFO['p3']}, mass {FLK_INFO['mass']:.4f}. "
  f"Identity check (51 cycles rebuilt unmodified as composites): max |diff| S_E {ID_SUM['d_S_E']:.1e}, D1 {ID_SUM['d_D1']:.1e}, D2 {ID_SUM['d_D2']:.1e}, p_2 {ID_SUM['d_p2']:.1e}; D3' flips {ID_SUM['D3p_flips']}.")
w("")
w("Columns: S_E (frozen multi-match) and plain (pre-fix rule) min/median/max; detection = share alarmed: D3' <= 0.10 (the Phase E rule), p_E <= 0.10 / 0.05, "
  "Phase A D3 <= 0.10, Phase A D1 p_1 <= 0.05 (its D3 path), D2 p_2 <= 0.05, and D3' built with the plain score. D1/D2 are recomputed for every synthetic cycle.")
w("")
w(md_table(SUM.drop(columns=["readings"]).rename(columns={"det_D3_phaseA": "det_D3(A)", "det_D3p_plain": "det_D3'plain", "det_D3p": "det_D3'"}), 3))
w("")
w("Collapsed readings (most frequent):")
w("")
w(md_table(SUM[["kind", "variant", "n", "readings"]], 3))
w("")
w("**Substitution / foreign insertion by readability** (readable = the debounced composite has a visit of the pasted box):")
w("")
w(md_table(RD.drop(columns=["plain_min", "plain_med", "plain_max"]).rename(columns={"det_D3p": "det_D3'", "det_D3_phaseA": "det_D3(A)", "det_D3p_plain": "det_D3'plain"}), 3))
w("")
w("Readability is mostly a property of the s01 run (B2 reads it as the pasted box or not, whatever the parent): runs readable in all / no / some parents: "
  + "; ".join(f"{k} {a}/{b}/{c}" for k, (a, b, c) in {k: (int((g['mean'] == 1).sum()), int((g['mean'] == 0).sum()), int(((g['mean'] > 0) & (g['mean'] < 1)).sum()))
                                                     for k, g in RD_RUN.groupby('kind', observed=True)}.items())
  + ". Per-run rows: `reports/faz_e/01_synthetic_per_run.csv`.")
w("")
w("**Per fold** (kinds 1-3, 5, 6):")
w("")
w(md_table(PFT, 3))
w("")
w("**Kind-specific checks:**")
w("")
for k, v in CHK.items():
    w(f"- {k}: {v}")
w("")
w("**Truncation** by kept Box-5 frames and whether the debounced reading lacks Box 5:")
w("")
w(md_table(TRU[["variant", "lacks5", "n", "S_E_min", "S_E_med", "S_E_max", "det_D3p", "det_pE05", "det_D3_phaseA", "readings"]].rename(columns={"det_D3p": "det_D3'", "det_D3_phaseA": "det_D3(A)"}), 3))
w("")
w("**Tempo scaling** (0.1 s grid; D1 and D2 recomputed; Phase A D3 and D3' on the same scaled cycles):")
w("")
w(md_table(TEM.rename(columns={"det_D3p": "det_D3'", "det_D3_phaseA": "det_D3(A)", "seq_ch_pE05": "seq channel p_E<=0.05"}), 3))
w("")
w(f"Frame intervals inside the parents: {HOLE_SUM['n_intervals_gt_0p15s']} intervals > 0.15 s in {HOLE_SUM['n_parents_with_one']} of {len(PARENTS)} parents "
  f"(min/median/max {HOLE_SUM['interval_min_med_max']} s; location {HOLE_SUM['where']}); frames without radar points are absent from the frame table, so the real "
  "time axis has holes (locations as listed). Literal nearest-in-time resampling keeps their real duration (the frame before a hole is repeated on the 0.1 s grid). "
  f"Tempo rows of the parent with the largest interval ({HOLE_MAX['interval_s']:.2f} s, cycle {int(HOLE_MAX['cycle_idx'])}):")
w("")
w(md_table(HOLE_TEMPO.rename(columns={"D3p": "D3'"}), 3))
w("")
w("**Split visit** (negative control; S_E must be 0, D2 may react):")
w("")
w(md_table(SPL[["variant", "n", "S_E_max", "det_D3p", "det_pE10", "det_p2_05", "det_D3_phaseA", "readings"]].rename(columns={"det_D3p": "det_D3'", "det_D3_phaseA": "det_D3(A)"}), 3))
w("")
w("**D1 exhibit (kind 12).** Phase A D1 at its D3 path (p_1 <= 0.05) against E1 (p_E <= 0.05) on the same synthetic cycles:")
w("")
w(md_table(EX.rename(columns={"E1_D3p": "E1 D3'", "PhaseA_D3": "Phase A D3"}), 3))
w("")
w(f"D1 detection per fold: {D1PF}. Calibration D1 maximum per fold: {({f: round(float(FOLD[f].cal_D1.max()), 3) for f in FOLD})}; "
  f"D1 values the synthetic deviations receive per fold (first distinct values): {D1VAL}. "
  "Every unseen transition gets the same Laplace score, so a single calibration cycle with any unseen transition ties with all deviations of its fold.")
w("")
w("## 5. Acceptance (stated in the plan before the battery was run)")
w("")
w(md_table(ACCD.assign(passed=ACCD["passed"].map({True: "PASS", False: "FAIL"})), 3))
w("")
w(f"**All items pass: {ALL_PASS}.**")
w("")
if MD_FAIL:
    w("**Failed item A3, as observed (normal data only; the method was not changed).** "
      f"{len(MD_FAIL)} of {len(md_)} mid-dwell flicker composites score above 1.2:")
    w("")
    w(md_table(pd.DataFrame(MD_FAIL), 3))
    w("")
    w(("In this case" if len(MD_FAIL) == 1 else "In each case") + " the parent's own Box-2 dwell already holds a raw Box-3 frame (a 1-frame B2 misreading that the Phase A debouncer had absorbed into the Box-2 visit) "
      "immediately before the midpoint where the 2-frame flicker block is inserted. The composite therefore carries a "
      f"{'/'.join(str(x['composite_box3_frames']) for x in MD_FAIL)}-frame Box-3 run, and S_E equals that run's whole posterior mass "
      f"(S_E = off-route visit mass in {CHK['mid_dwell_S_E_eq_offroute_visit_mass']} mid-dwell composites). The multi-match still charges only the foreign visit "
      "(the split Box-2 dwell costs nothing; the plain rule charges more), but the 1.2 bound assumed the inserted visit carries only the block's own mass. "
      "Plan section 4 says that a failed item leads to an amendment on normal data only, re-dated, before the one-shot; that decision is left to the authors.")
    w("")
w("**Preliminary values (workflow logs, `faz_e/prelim/`) against this run:**")
w("")
w(md_table(PRELIM, 3))
w("")
w("## 6. Method notes: details the plan left open (fixed in code before scoring)")
for x in OPEN:
    w(f"- {x}")
w("- p-values, D1, D2 and debouncing are the Phase A functions of `faz_a2/a2_cycles.py` (`conf_p`, `fit_params`, `score_cycle`, `debounce_labels`, `visits_of`, `cycle_from`).")
w("- Explanations: every synthetic and normal row carries the DP traceback text (`explain` column of the CSVs).")
w("- Code shared with the one-shot: `e1_core.e1_inputs`, `score_roles` and `compare_to_reference` produced every normal-side number above; "
  "`e1_core.oneshot_tables` was smoke-tested here on the 51 normal test rows plus synthetic rows relabelled as stand-in anomaly groups "
  f"({SMOKE_LABEL}); its rates matched the direct computation. "
  "`faz_e/02_e1_oneshot.py` itself was only parsed (ast), never executed.")
w("")
w("## 7. Files")
for k, p in paths.items():
    w(f"- `{p.relative_to(ROOT).as_posix()}`")
w("- `reports/faz_e/01_synthetic_stdout.log` (stdout of this run, kept by the caller: "
  "`bash tools/local_run.sh faz_e/01_e1_synthetic.py 2>&1 | tee reports/faz_e/01_synthetic_stdout.log`)")
w("")
w("## 8. Provenance")
w(f"- Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}; host {platform.node()}; CPU only.")
w(f"- DESIGN_E.md frozen part sha256 `{DESIGN_SHA}`.")
for k, v in src_sha.items():
    w(f"- code `{k}` sha256 `{v}`")
for k, v in in_sha.items():
    w(f"- input `{INPUTS[k].relative_to(ROOT).as_posix()}` sha256 `{v}`")
w(f"- Sessions in memory: {SESSIONS_IN_MEMORY}; tripwires fired: {TRIPPED or 'none'}; `fa_common` all-session frame table built: no.")
paths["md"].write_text("\n".join(L) + "\n", encoding="utf-8")
for k, p in paths.items():
    print(f"wrote {p.relative_to(ROOT).as_posix()}")
print(f"done in {time.time() - T_START:.1f} s")

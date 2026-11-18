"""Gap 1 (OUTLINE s9.1): recording minutes for commissioning (A2.4) with ONE duration definition.

Definition (timestamp-based, from the frame table; frames with no radar points never appear as rows, so n_frames*0.1
UNDER-counts wall time):
    dur_b(loop) = t(first frame of the NEXT loop_id) - t(first frame of this loop)      [primary: contiguous partition,
                  sum over a session = session wall time]; last loop of a session: dur_a.
    dur_a(loop) = t(last frame) + 0.1 - t(first frame)                                   [= `dur_s` of the ToF segmentation]
    dur_c(loop) = n_frames * 0.1                                                         [OLD definition used in 04_commissioning.md]

Nothing is re-trained. (a) re-uses the A2.4 random assignment (RandomState([20261104, r]) on the 51 s02 normal ToF loops)
and the Faz A cycle tables only to convert counts to minutes; (b) re-uses the A2.4 loop-sampling RNG (RandomState([SEED, f, rep]))
and is validated against the stored `rec_min_total` (old frame-count definition) before any timestamp number is trusted.

Run from ROOT:  PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg DATA_PATH=... CACHE_DIR=... <venv python> paper/scripts/gap1_minutes.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DATA_PATH", str(ROOT / "data"))
os.environ.setdefault("CACHE_DIR", str(ROOT / ".cache"))
sys.path.insert(0, str(ROOT / "faz_a"))
sys.path.insert(0, str(ROOT / "faz_a2"))
sys.path.insert(0, str(ROOT / "probe"))
import numpy as np
import pandas as pd

import a2_cycles as ac
import fa_common as fc

OUT = ROOT / "reports" / "paper_prep"
DT = 0.1
SEED = 20261104


def P(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ per-loop durations
ft = fc.frame_table()
AL = fc.anom_loops()
g = ft.groupby("gid")
L = pd.DataFrame({"session": g["session"].first(), "loop_id": g["loop_id"].first(), "t_first": g["t"].min(), "t_last": g["t"].max(),
                  "n_frames": g.size()}).reset_index()
L = L.sort_values(["session", "loop_id"]).reset_index(drop=True)
L["dur_a"] = L["t_last"] + DT - L["t_first"]
nxt = L.groupby("session")["t_first"].shift(-1)
L["dur_b"] = (nxt - L["t_first"]).fillna(L["dur_a"])
L["dur_c"] = L["n_frames"] * DT
# loops must be consecutive in a session
assert all((np.diff(d["loop_id"].to_numpy()) == 1).all() for _, d in L.groupby("session"))
L = L.merge(AL[["session", "loop_id", "is_cycle", "use", "fold"]], on=["session", "loop_id"], how="left", validate="1:1")
L["is_cycle"] = L["is_cycle"].astype(bool)
P("per-loop table:", L.shape)

# session wall time check: sum of dur_b over loops == t_last_of_session + 0.1 - t_first_of_session  (when last loop uses dur_a)
chk = L.groupby("session").agg(sum_b=("dur_b", "sum"), sum_c=("dur_c", "sum"), n=("loop_id", "size"))
chk["wall"] = ft.groupby("session")["t"].agg(lambda s: s.max() - s.min() + DT)
P(chk)

# ------------------------------------------------------------------ cross-check vs dur_s already stored in the Faz A cycle table
cs = pd.read_csv(ROOT / "reports" / "faz_a" / "04_cycle_scores.csv.gz")
tof_n = cs[(cs.state_source == "tof") & (cs.segmentation == "tof") & (cs.atype == "normal") & (cs.role == "test")]
chk_a = L[(L.session == "session_02_normal_seq125") & (L.use == "normal_crossfit")]
P(f"cross-check: stored dur_s (tof x tof, s02 normal, test rows n={len(tof_n)}) mean = {tof_n.dur_s.mean():.4f}; "
  f"recomputed dur_a over the 51 s02 normal loops = {chk_a.dur_a.mean():.4f} (n={len(chk_a)})")
merged = tof_n.merge(chk_a[["loop_id", "dur_a"]], left_on="cycle_idx", right_on="loop_id")
P(f"  per-loop max |stored - dur_a| = {np.abs(merged.dur_s - merged.dur_a).max():.2e} over {len(merged)} loops")
pn = cs[(cs.state_source == "b2") & (cs.segmentation == "pred") & (cs.atype == "normal") & (cs.role == "test")]
P(f"  B2 x pred normal test cycles: n={len(pn)}, mean dur_s = {pn.dur_s.mean():.4f}")

# ================================================================== (a) anomaly detector: n training + 20 calibration cycles
loops51 = sorted(chk_a["loop_id"].astype(int))
assert len(loops51) == 51
dur = chk_a.set_index("loop_id")
rows_a = []
simple = []
for name, col in (("dur_b (primary)", "dur_b"), ("dur_a", "dur_a"), ("dur_c (old, frames*0.1)", "dur_c")):
    d = dur[col]
    simple.append(dict(definition=name, n_loops=len(d), mean_s=d.mean(), median_s=d.median(), sd_s=d.std(ddof=1), q25=d.quantile(.25), q75=d.quantile(.75)))
SIMPLE = pd.DataFrame(simple)
P(SIMPLE)

# exact A2.4 assignment, ToF x tof cell: cycles == ToF loops
N_TEST, N_CAL = 11, 20
R = 50
for r in range(R):
    perm = np.random.RandomState([SEED, r]).permutation(np.array(loops51))
    ca, po = perm[N_TEST:N_TEST + N_CAL], perm[N_TEST + N_CAL:]
    for n in (5, 10, 15, 20):
        tr = po[:n]
        rows_a.append(dict(cell="ToF x tof", rep=r, n=n, n_train=len(tr), n_cal=len(ca),
                           train_min=dur.loc[tr, "dur_b"].sum() / 60, cal_min=dur.loc[ca, "dur_b"].sum() / 60,
                           train_min_old=n * 5.39 / 60, cal_min_old=20 * 5.39 / 60,
                           train_min_cyclesum=dur.loc[tr, "dur_b"].sum() / 60, cal_min_cyclesum=dur.loc[ca, "dur_b"].sum() / 60))   # old doc: n * 5.39 s

# B2 x pred cell: predicted cycles (debounced Box1 entry -> next Box1 entry; dur_s timestamp-based) mapped to the same ToF loops
LAB = ac.source_labels(sources=("b2",), verbose=False)
C, _ = ac.build_cycles("b2", "pred", LAB["b2"])
cl = C.to_dict("records")
norm = [c for c in cl if c["atype"] == "normal"]
by_loop = {}
for c in sorted(norm, key=lambda c: c["cycle_idx"]):
    by_loop.setdefault(int(c["anchor_loop"]), []).append(c)
assert set(by_loop) <= set(loops51)
for r in range(R):
    perm = np.random.RandomState([SEED, r]).permutation(np.array(loops51))
    ca_l, po_l = perm[N_TEST:N_TEST + N_CAL], perm[N_TEST + N_CAL:]
    gat = lambda Ls: [c for l in Ls for c in by_loop.get(int(l), [])]
    ca_c, pool = gat(ca_l), gat(po_l)
    for n in (5, 10, 15, 20):
        tr = pool[:n]
        tr_loops = sorted({int(c["anchor_loop"]) for c in tr})          # ToF loops that had to be recorded to obtain these cycles
        rows_a.append(dict(cell="B2 x pred", rep=r, n=n, n_train=len(tr), n_cal=len(ca_c),
                           train_min=dur.loc[tr_loops, "dur_b"].sum() / 60, cal_min=dur.loc[ca_l, "dur_b"].sum() / 60,     # loop-based (recorded time)
                           train_min_cyclesum=sum(c["dur_s"] for c in tr) / 60, cal_min_cyclesum=sum(c["dur_s"] for c in ca_c) / 60,
                           train_min_old=len(tr) * 5.00 / 60, cal_min_old=20 * 5.00 / 60))   # old doc: n * 5.00 s (+20 cal cycles * 5.00 s)
A = pd.DataFrame(rows_a)
A["total_min"] = A["train_min"] + A["cal_min"]
A["total_min_old"] = A["train_min_old"] + A["cal_min_old"]
A["total_min_cyclesum"] = A["train_min_cyclesum"] + A["cal_min_cyclesum"]
SA = A.groupby(["cell", "n"]).agg(n_train_mean=("n_train", "mean"), n_cal_mean=("n_cal", "mean"),
                                   train_min_mean=("train_min", "mean"), train_min_sd=("train_min", "std"),
                                   cal_min_mean=("cal_min", "mean"), cal_min_sd=("cal_min", "std"),
                                   total_min_mean=("total_min", "mean"), total_min_sd=("total_min", "std"),
                                   total_min_q05=("total_min", lambda x: x.quantile(.05)), total_min_q95=("total_min", lambda x: x.quantile(.95)),
                                   total_min_cyclesum=("total_min_cyclesum", "mean"), total_min_old=("total_min_old", "mean")).reset_index()
# closed form: (n + 20) cycles x mean duration of s02 normal loops (dur_b)
m_b = float(dur["dur_b"].mean())
SA["closed_form_min_(n+20)xmean_dur_b"] = (SA["n"] + 20) * m_b / 60
P(SA.round(3).to_string())
SA.to_csv(OUT / "05_gaps_minutes_anomaly.csv", index=False)
SIMPLE.to_csv(OUT / "05_gaps_minutes_s02_duration_defs.csv", index=False)

# ================================================================== (b) state estimator: k labelled cycles per session
sess = sorted(L["session"].unique())
rows_s = []
for s in sess:
    d = L[L.session == s]
    full = d[d.is_cycle] if s != "session_01_training" else d
    rows_s.append(dict(session=s, n_loop_groups=len(d), n_complete_cycles=int(d.is_cycle.sum()),
                       all_groups_mean_b=d.dur_b.mean(), all_groups_mean_a=d.dur_a.mean(), all_groups_mean_c=d.dur_c.mean(),
                       complete_mean_b=full.dur_b.mean(), complete_mean_a=full.dur_a.mean(), complete_mean_c=full.dur_c.mean(),
                       session_wall_min=ft[ft.session == s]["t"].agg(lambda x: x.max() - x.min() + DT) / 60))
DS = pd.DataFrame(rows_s)
P(DS.round(3).to_string())
DS.to_csv(OUT / "05_gaps_minutes_state_session_durations.csv", index=False)

# exact A2.4 sampling replicate (no training): validate against stored rec_min_total (old def)
W = fc.windows()
gid = W["gid"].to_numpy(); sidx = W["session_idx"].to_numpy(); s1 = W["s1"].to_numpy()
nsess = len(sess)
gdur = L.set_index("gid")
rows_k = []
for f in range(fc.N_FOLDS):
    tr_all = np.flatnonzero(s1 != f)
    for k in (1, 2, 5, 10, 20):
        for rep in range(10):
            rng = np.random.RandomState([SEED, f, rep])
            ch = []
            for si in range(nsess):
                gg = np.unique(gid[tr_all][sidx[tr_all] == si])
                ch += rng.permutation(gg)[:k].tolist()
            ch = np.array(ch)
            rows_k.append(dict(fold=f, k=k, rep=rep, n_loops=len(ch), min_old=gdur.loc[ch, "n_frames"].sum() * DT / 60,
                               min_b=gdur.loc[ch, "dur_b"].sum() / 60, min_a=gdur.loc[ch, "dur_a"].sum() / 60))
    ch = np.unique(gid[tr_all])
    rows_k.append(dict(fold=f, k=-1, rep=0, n_loops=len(ch), min_old=gdur.loc[ch, "n_frames"].sum() * DT / 60,
                       min_b=gdur.loc[ch, "dur_b"].sum() / 60, min_a=gdur.loc[ch, "dur_a"].sum() / 60))
K = pd.DataFrame(rows_k)
# validation against stored run file
runs = pd.read_csv(ROOT / "reports" / "faz_a2" / "04_comm_state_runs.csv")
runs = runs[runs.model == "b2"].copy()
runs["k"] = runs["k"].astype(str)
Kv = K.copy(); Kv["k"] = Kv["k"].map(lambda x: "all" if x == -1 else str(x))
mm = runs.merge(Kv, on=["fold", "k", "rep"], suffixes=("_stored", "_replic"))
P(f"validation: {len(mm)} rows matched; max |rec_min_total(stored) - min_old(replicated)| = {np.abs(mm.rec_min_total - mm.min_old).max():.2e}; "
  f"n_loops equal = {(mm.n_loops_stored == mm.n_loops_replic).all()}")
assert len(mm) == len(runs) and np.abs(mm.rec_min_total - mm.min_old).max() < 1e-9, "sampling replicate does not match A2.4"
SK = Kv.groupby("k").agg(n_loops=("n_loops", "mean"), min_old=("min_old", "mean"), min_b=("min_b", "mean"), min_b_sd=("min_b", "std"), min_a=("min_a", "mean")).reset_index()
order = {"1": 1, "2": 2, "5": 3, "10": 4, "20": 5, "all": 6}
SK = SK.sort_values("k", key=lambda s: s.map(order)).reset_index(drop=True)
# per-session (single station) closed form: k x mean of the session's loop groups (dur_b)
for kk in (1, 2, 5, 10, 20):
    SK.loc[SK.k == str(kk), "closed_form_total_6sess_min"] = kk * DS["all_groups_mean_b"].sum() / 60
    SK.loc[SK.k == str(kk), "closed_form_per_session_min_mean"] = kk * DS["all_groups_mean_b"].mean() / 60
P(SK.round(3).to_string())
SK.to_csv(OUT / "05_gaps_minutes_state_total.csv", index=False)
# per-session minutes for k in (5,10,20)
ps = []
for kk in (1, 2, 5, 10, 20):
    for r in DS.itertuples():
        ps.append(dict(k=kk, session=r.session, minutes_dur_b=kk * r.all_groups_mean_b / 60, minutes_complete_dur_b=kk * r.complete_mean_b / 60,
                       minutes_old_frames=kk * r.all_groups_mean_c / 60))
PS = pd.DataFrame(ps)
PS.to_csv(OUT / "05_gaps_minutes_state_per_session.csv", index=False)
P(PS[PS.k.isin([5, 10, 20])].pivot(index="session", columns="k", values="minutes_dur_b").round(2))
P("done")

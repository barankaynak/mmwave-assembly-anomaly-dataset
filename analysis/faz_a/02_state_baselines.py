"""Faz A / 02 - durum tahmini taban cizgileri B1 (en-yakin-merkez) ve B2 (el yapimi ozellik + RandomForest).
DESIGN.md'ye birebir. S1 (5 kat) ve S2 (LOSO, 6 kat). OOF dosyalari fa_common.write_oof ile yazilir."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probe"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import fa_common as fc

SEED = 20261104
TAU_GRID = np.concatenate([np.arange(0.002, 0.02, 0.002), np.arange(0.02, 0.62, 0.02), [np.inf]])
FEATS = ["cx", "cy", "n_points", "snr_sum", "mean_abs_v", "std_x", "std_y"]
FILL = {"cx": -2.0, "cy": -2.0, "n_points": 0.0, "snr_sum": 0.0, "mean_abs_v": 0.0, "std_x": 0.0, "std_y": 0.0}

ft = fc.frame_table()
W = fc.windows()
R = fc.win_rows(W)                       # (Nw,5) frame rows
CX = ft["cx"].to_numpy(float); CY = ft["cy"].to_numpy(float)
print(f"frames={len(ft)} windows={len(W)}; frames with NaN centroid={int(np.isnan(CX).sum())}")

def b2_features(rows):
    cols, names = [], []
    t = np.arange(fc.L, dtype=float); tc = t - t.mean()
    for f in FEATS:
        v = ft[f].to_numpy(float)
        v = np.where(np.isnan(v), FILL[f], v)[rows]            # (n,5)
        cols += [v[:, -1], v.mean(1), (v * tc).sum(1) / (tc ** 2).sum()]
        names += [f + "_last", f + "_mean", f + "_slope"]
    return np.stack(cols, 1).astype(np.float32), names

Xb2_all, B2_NAMES = b2_features(R)
LASTX, LASTY = CX[R[:, -1]], CY[R[:, -1]]
Y = W["y_true"].to_numpy()

def centroids(mask):
    c = np.full((5, 2), np.nan)
    for k in range(1, 6):
        m = mask & (Y == k) & ~np.isnan(LASTX)
        if m.sum(): c[k - 1] = [LASTX[m].mean(), LASTY[m].mean()]
    return c

def b1_predict(idx, cent, tau):
    d = np.sqrt((LASTX[idx, None] - cent[None, :, 0]) ** 2 + (LASTY[idx, None] - cent[None, :, 1]) ** 2)
    d = np.where(np.isnan(d), np.inf, d)
    k = d.argmin(1); dm = d.min(1)
    return np.where(dm > tau, 0, k + 1)

def macro(y, p): return f1_score(y, p, labels=list(range(6)), average="macro", zero_division=0)

summary = []
timing = {"b1": [], "b2": []}
def run_fold(split, f, tr, te):
    gid = W["gid"].to_numpy(); sidx = W["session_idx"].to_numpy()
    tr_idx = np.flatnonzero(tr); te_idx = np.flatnonzero(te)
    val = fc.inner_val_split(gid[tr_idx], seed=SEED + f, strata=sidx[tr_idx])
    itr = np.zeros(len(W), bool); itr[tr_idx[~val]] = True
    iva = tr_idx[val]
    cent_in = centroids(itr)
    sc = [macro(Y[iva], b1_predict(iva, cent_in, t)) for t in TAU_GRID]
    tau = float(TAU_GRID[int(np.argmax(sc))])
    cent = centroids(tr)
    p1 = b1_predict(te_idx, cent, tau)
    wins = W.iloc[te_idx]
    fc.write_oof("b1", split, f, wins, y_pred=p1)
    # B2
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=8)
    rf.fit(Xb2_all[tr_idx], Y[tr_idx])
    pr = np.zeros((len(te_idx), 6), np.float32)
    pr[:, rf.classes_.astype(int)] = rf.predict_proba(Xb2_all[te_idx])
    fc.write_oof("b2", split, f, wins, proba=pr)
    m1, m2 = fc.state_metrics(Y[te_idx], p1), fc.state_metrics(Y[te_idx], pr.argmax(1))
    print(f"{split} f{f}: n_te={len(te_idx)} tau={tau:.2f} | B1 acc={m1['acc']:.4f} mF1={m1['macro_f1']:.4f} | B2 acc={m2['acc']:.4f} mF1={m2['macro_f1']:.4f}")
    summary.append({"split": split, "fold": f, "tau": tau, "b1_acc": m1["acc"], "b1_mf1": m1["macro_f1"], "b2_acc": m2["acc"], "b2_mf1": m2["macro_f1"]})
    return rf, cent, tau

last = None
for f in range(fc.N_FOLDS):
    te = W["s1"].to_numpy() == f
    last = run_fold("S1", f, ~te, te)
for f in range(6):
    te = W["s2"].to_numpy() == f
    run_fold("S2", f, ~te, te)

# --- CPU inference time per window (single window, single thread, incl. feature extraction)
rf, cent, tau = last
rf.set_params(n_jobs=1)
rng = np.random.RandomState(0); sel = rng.choice(len(W), 300, replace=False)
t0 = time.perf_counter()
for i in sel: b1_predict(np.array([i]), cent, tau)
b1_ms = (time.perf_counter() - t0) / len(sel) * 1e3
t0 = time.perf_counter()
for i in sel:
    X, _ = b2_features(R[i:i + 1]); rf.predict(X)
b2_ms = (time.perf_counter() - t0) / len(sel) * 1e3
print(f"CPU latency per window (1 thread): B1={b1_ms:.4f} ms  B2={b2_ms:.3f} ms (feature extraction + RF predict, 300 trees)")
json.dump({"b1_ms": b1_ms, "b2_ms": b2_ms, "b2_n_trees": 300, "b2_n_features": len(B2_NAMES)},
          open(fc.OUT / "baseline_runs.json", "w"))
pd.DataFrame(summary).to_csv(fc.OUT / "02_baseline_folds.csv", index=False)
print("done")

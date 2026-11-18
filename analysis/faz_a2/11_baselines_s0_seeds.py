"""Faz A2 / A2.2 (a) - B1 ve B2 S0 (rastgele pencere bolmesi), tohumlar 42,1,2,3,4.
B1/B2 tanimlari faz_a/02_state_baselines.py'den BIREBIR kopya (02'nin __main__ korumasi yok; import edilemez).
Degisen tek sey bolme: S0 tohum ailesi (a2_s0.py) + ic dogrulama = sinif-tabakali rastgele %10 (CNN S0 dali gibi; B1 tau secimi icin).
B2: tum egitim pencereleriyle RF(300, balanced, random_state=20261104+tohum, n_jobs=3).
Cikti: reports/faz_a2/oof/{b1,b2}__S0s{seed}__f0.csv.gz, reports/faz_a2/11_baselines_s0.csv"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import a2_s0 as s0
fc = s0.fc

TAU_GRID = np.concatenate([np.arange(0.002, 0.02, 0.002), np.arange(0.02, 0.62, 0.02), [np.inf]])
FEATS = ["cx", "cy", "n_points", "snr_sum", "mean_abs_v", "std_x", "std_y"]
FILL = {"cx": -2.0, "cy": -2.0, "n_points": 0.0, "snr_sum": 0.0, "mean_abs_v": 0.0, "std_x": 0.0, "std_y": 0.0}

ft = fc.frame_table()
W = fc.windows()
R = fc.win_rows(W)
CX = ft["cx"].to_numpy(float); CY = ft["cy"].to_numpy(float)
Y = W["y_true"].to_numpy()
print(f"frames={len(ft)} windows={len(W)}", flush=True)
s0.assert_seed42_matches_faz_a(W)
print("seed 42 S0 split == Faz A split_s0.csv == windows()['s0_test'] (assert OK)", flush=True)

def b2_features(rows):
    cols, names = [], []
    t = np.arange(fc.L, dtype=float); tc = t - t.mean()
    for f in FEATS:
        v = ft[f].to_numpy(float)
        v = np.where(np.isnan(v), FILL[f], v)[rows]
        cols += [v[:, -1], v.mean(1), (v * tc).sum(1) / (tc ** 2).sum()]
        names += [f + "_last", f + "_mean", f + "_slope"]
    return np.stack(cols, 1).astype(np.float32), names

Xb2_all, B2_NAMES = b2_features(R)
LASTX, LASTY = CX[R[:, -1]], CY[R[:, -1]]

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

rows = []
for seed in s0.SEEDS:
    t0 = time.time()
    trn, te = s0.s0_masks(W, seed)
    itr_idx, iva = s0.inner_val(trn, Y, seed)
    assert len(np.intersect1d(trn, te)) == 0 and len(trn) + len(te) == len(W)
    itr = np.zeros(len(W), bool); itr[itr_idx] = True
    tr = np.zeros(len(W), bool); tr[trn] = True
    cent_in = centroids(itr)
    sc = [macro(Y[iva], b1_predict(iva, cent_in, t)) for t in TAU_GRID]
    tau = float(TAU_GRID[int(np.argmax(sc))])
    cent = centroids(tr)
    p1 = b1_predict(te, cent, tau)
    wins = W.iloc[te]
    s0.write_oof2("b1", seed, wins, y_pred=p1)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=s0.MODEL_SEED0 + seed, n_jobs=3)
    rf.fit(Xb2_all[trn], Y[trn])
    pr = np.zeros((len(te), 6), np.float32)
    pr[:, rf.classes_.astype(int)] = rf.predict_proba(Xb2_all[te])
    s0.write_oof2("b2", seed, wins, proba=pr)
    m1, m2 = fc.state_metrics(Y[te], p1), fc.state_metrics(Y[te], pr.argmax(1))
    print(f"S0 seed {seed}: n_tr={len(trn)} (inner tr {len(itr_idx)} / val {len(iva)}) n_te={len(te)} tau={tau:.3f} | "
          f"B1 acc={m1['acc']:.4f} mF1={m1['macro_f1']:.4f} | B2 acc={m2['acc']:.4f} mF1={m2['macro_f1']:.4f} [{time.time()-t0:.0f}s]", flush=True)
    rows.append({"seed": seed, "n_train": len(trn), "n_test": len(te), "tau": tau,
                 "b1_acc": m1["acc"], "b1_mf1": m1["macro_f1"], "b2_acc": m2["acc"], "b2_mf1": m2["macro_f1"]})
pd.DataFrame(rows).to_csv(s0.OUT2 / "11_baselines_s0.csv", index=False)
print("done", flush=True)

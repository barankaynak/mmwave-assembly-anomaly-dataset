"""Faz B / B.2 ek (brief disi, ikincil): her seviye (ve birlesimler) RF tohumlari 1-4 ile yeniden egitilir -> seviye farklarinin tohumdan bagimsiz mi oldugu.
Girdi: reports/faz_b/02_sensing_budget.csv (seviye parametreleri). Tohum 20261104 (ana tablo) bu dosyada YOK. Cikti: 02_level_seeds.csv. n_jobs=3.
OOF'lar reports/faz_b/oof/sbs_<tag>_s<seed>__S1__f*.csv.gz."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probe")); sys.path.insert(0, str(ROOT / "faz_a")); sys.path.insert(0, str(ROOT / "faz_a2")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import fa_common as fc
import a2_cycles as ac
import b_frames as bf

REP = ROOT / "reports" / "faz_b"
OOFD = REP / "oof"
FEATS = bf.STAT_COLS
FILL = {"cx": -2.0, "cy": -2.0, "n_points": 0.0, "snr_sum": 0.0, "mean_abs_v": 0.0, "std_x": 0.0, "std_y": 0.0}
SEEDS = [1, 2, 3, 4]
W = fc.windows(); R = fc.win_rows(W); Y = W["y_true"].to_numpy(); S1 = W["s1"].to_numpy()
lv = pd.read_csv(REP / "02_sensing_budget.csv")[["tag", "axis", "tau_db", "vel", "pts", "group"]]
out_path = REP / "02_level_seeds.csv"
rows = []
tcen = np.arange(fc.L, dtype=float); tc = tcen - tcen.mean()
for _, l in lv.iterrows():
    params = dict(tau_db=float(l.tau_db), vel=l["vel"], pts=l.pts, group=bool(l.group))
    st = bf.frame_stats(params)
    cols = []
    for f in FEATS:
        v = np.where(np.isnan(st[f].to_numpy(float)), FILL[f], st[f].to_numpy(float))[R]
        cols += [v[:, -1], v.mean(1), (v * tc).sum(1) / (tc ** 2).sum()]
    X = np.stack(cols, 1).astype(np.float32)
    for seed in SEEDS:
        t0 = time.time()
        tag = f"sbs_{l.tag}_s{seed}"
        f1s = []
        for f in range(5):
            te, tr = np.flatnonzero(S1 == f), np.flatnonzero(S1 != f)
            rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=3).fit(X[tr], Y[tr])
            pr = np.zeros((len(te), 6), np.float32)
            pr[:, rf.classes_.astype(int)] = rf.predict_proba(X[te])
            wins = W.iloc[te]
            d = pd.DataFrame({"session": wins["session"].to_numpy(), "frame_id": wins["frame_id"].to_numpy(), "t": wins["t"].to_numpy(),
                              "loop_id": wins["loop_id"].to_numpy(), "fold": f, "y_true": wins["y_true"].to_numpy(),
                              "y_raw": wins["y_raw"].to_numpy(), "y_pred": pr.argmax(1)})
            for c in range(6):
                d[f"p{c}"] = pr[:, c]
            d[fc.OOF_COLS].to_csv(OOFD / f"{tag}__S1__f{f}.csv.gz", index=False, compression="gzip")
            f1s.append(fc.state_metrics(Y[te], pr.argmax(1))["macro_f1"])
        oof = pd.concat([pd.read_csv(OOFD / f"{tag}__S1__f{f}.csv.gz") for f in range(5)], ignore_index=True)
        labs = ac.labels_from_oof(oof, strict=True)
        C, _ = ac.build_cycles(tag, "pred", labs)
        S, _ = ac.score_group(C, verbose=False)
        PF, SM = ac.evaluate(S)
        TAB = ac.summarize(PF, SM)
        a = TAB[(TAB.variant == "D3") & (TAB.atype == "tum anomali")].iloc[0]
        rows.append(dict(tag=l.tag, axis=l.axis, seed=seed, f1_mean=float(np.mean(f1s)), auroc=a.auroc_mean, det_a10=a.det_a10, far_a10=a.far_a10, n_cycles=len(C)))
        print(f"[{l.tag} s{seed}] mF1={rows[-1]['f1_mean']:.4f} AUROC={rows[-1]['auroc']:.4f} det10={a.det_a10:.3f} FAR10={a.far_a10:.4f} cycles={len(C)} {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
d = pd.DataFrame(rows)
g = d.groupby("tag", sort=False).agg(f1=("f1_mean", "mean"), f1_sd=("f1_mean", "std"), auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
                                      det=("det_a10", "mean"), far=("far_a10", "mean"), far_sd=("far_a10", "std"))
b = g.loc["base"]
g["dF1"] = g.f1 - b.f1
g["dAUROC"] = g.auroc - b.auroc
print(g.round(4).to_string(), flush=True)
print("done", flush=True)

"""Faz B / B.2 ek (brief disi, ikincil): taban seviyesinde (merged/tam) yalniz RF tohumu degisince B2 macro-F1 ve anomali AUROC/FAR ne kadar oynar?
Amac: B.2 seviye farklarini (ozellikle AUROC) tohum gurultusuyle kiyaslamak. Tohumlar 1..8 (Faz A tohumu 20261104 ayrica referans). n_jobs=3."""
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
W = fc.windows(); R = fc.win_rows(W); Y = W["y_true"].to_numpy(); S1 = W["s1"].to_numpy()
st = bf.frame_stats(dict(tau_db=15.0, vel="full", pts="merged", group=False))
cols = []
t = np.arange(fc.L, dtype=float); tc = t - t.mean()
for f in FEATS:
    v = np.where(np.isnan(st[f].to_numpy(float)), FILL[f], st[f].to_numpy(float))[R]
    cols += [v[:, -1], v.mean(1), (v * tc).sum(1) / (tc ** 2).sum()]
X = np.stack(cols, 1).astype(np.float32)

rows = []
for seed in range(1, 9):
    t0 = time.time()
    f1s = []
    for f in range(5):
        te, tr = np.flatnonzero(S1 == f), np.flatnonzero(S1 != f)
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=3).fit(X[tr], Y[tr])
        pr = np.zeros((len(te), 6), np.float32)
        pr[:, rf.classes_.astype(int)] = rf.predict_proba(X[te])
        wins = W.iloc[te]
        df = pd.DataFrame({"session": wins["session"].to_numpy(), "frame_id": wins["frame_id"].to_numpy(), "t": wins["t"].to_numpy(),
                           "loop_id": wins["loop_id"].to_numpy(), "fold": f, "y_true": wins["y_true"].to_numpy(),
                           "y_raw": wins["y_raw"].to_numpy(), "y_pred": pr.argmax(1)})
        for c in range(6):
            df[f"p{c}"] = pr[:, c]
        df[fc.OOF_COLS].to_csv(OOFD / f"sb_noise_s{seed}__S1__f{f}.csv.gz", index=False, compression="gzip")
        f1s.append(fc.state_metrics(Y[te], pr.argmax(1))["macro_f1"])
    oof = pd.concat([pd.read_csv(OOFD / f"sb_noise_s{seed}__S1__f{f}.csv.gz") for f in range(5)], ignore_index=True)
    labs = ac.labels_from_oof(oof, strict=True)
    C, _ = ac.build_cycles(f"noise{seed}", "pred", labs)
    S, _ = ac.score_group(C, verbose=False)
    PF, SM = ac.evaluate(S)
    TAB = ac.summarize(PF, SM)
    a = TAB[(TAB.variant == "D3") & (TAB.atype == "tum anomali")].iloc[0]
    row = dict(seed=seed, f1_mean=float(np.mean(f1s)), auroc=a.auroc_mean, det_a10=a.det_a10, far_a10=a.far_a10, n_cycles=len(C))
    rows.append(row)
    print(f"[seed {seed}] mF1={row['f1_mean']:.4f} AUROC={row['auroc']:.4f} det10={row['det_a10']:.3f} FAR10={row['far_a10']:.4f} cycles={len(C)} {time.time()-t0:.0f}s", flush=True)
d = pd.DataFrame(rows)
d.to_csv(REP / "02_noise_floor.csv", index=False)
print(f"[tohum gurultusu, 8 tohum] macro-F1 mean={d.f1_mean.mean():.4f} sd={d.f1_mean.std(ddof=1):.4f} range=[{d.f1_mean.min():.4f},{d.f1_mean.max():.4f}]; "
      f"AUROC mean={d.auroc.mean():.4f} sd={d.auroc.std(ddof=1):.4f} range=[{d.auroc.min():.4f},{d.auroc.max():.4f}]; "
      f"det10 sd={d.det_a10.std(ddof=1):.3f}; FAR10 mean={d.far_a10.mean():.4f} sd={d.far_a10.std(ddof=1):.4f} range=[{d.far_a10.min():.4f},{d.far_a10.max():.4f}]", flush=True)
print("done", flush=True)

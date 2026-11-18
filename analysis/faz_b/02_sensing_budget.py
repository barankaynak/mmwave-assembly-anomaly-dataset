"""Faz B / B.2 - Algilama butcesi: cip ustu nokta bulutunun ne kadari yetiyor? (oykunme). DESIGN_B.md B.2'ye birebir.
Her seviyede: ham mmwave_data.csv -> kare istatistikleri (b_frames) -> B2 ozellikleri (7 x son/ortalama/egim) -> RF 300 (balanced, seed 20261104,
n_jobs<=3) S1 5 kat -> tam OOF (reports/faz_b/oof/sb_<tag>__S1__f*.csv.gz) -> anomali hatti (pred, D3, alpha 0,10) -> AUROC/FAR; bayt/kare, bayt/s.
Tek eksen degisir (digerleri taban). Taban satiri esdeger kapi: Faz A B2'yi birebir uretmeli. Sonra eksen-basina diz secimi ve
(ek, ikincil) secilen seviyelerin birlesimi kosulur. Her sayi stdout'a da basilir."""
import sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probe")); sys.path.insert(0, str(ROOT / "faz_a")); sys.path.insert(0, str(ROOT / "faz_a2")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import fa_common as fc
import a2_cycles as ac
import b_frames as bf

SEED = 20261104
NJOBS = 3
FEATS = bf.STAT_COLS
FILL = {"cx": -2.0, "cy": -2.0, "n_points": 0.0, "snr_sum": 0.0, "mean_abs_v": 0.0, "std_x": 0.0, "std_y": 0.0}
REP = ROOT / "reports" / "faz_b"
OOFD = REP / "oof"
OOFD.mkdir(parents=True, exist_ok=True)

BASE = dict(tau_db=15.0, vel="full", pts="merged", group=False)
LEVELS = [("base", "-", "merged / tam (taban)", BASE)]
for t in (17.5, 20.0, 22.5, 25.0):
    LEVELS.append((f"a_tau{t:g}", "a", f"SNR >= {t:g} dB", {**BASE, "tau_db": t}))
LEVELS.append(("b_sign", "b", "hiz: yalniz isaret", {**BASE, "vel": "sign"}))
LEVELS.append(("b_removed", "b", "hiz: cikarilmis", {**BASE, "vel": "removed"}))
LEVELS.append(("c_raw", "c", "nokta: ham (v=0 dahil)", {**BASE, "pts": "raw"}))
LEVELS.append(("c_static", "c", "nokta: yalniz v=0", {**BASE, "pts": "static"}))
LEVELS.append(("d_group", "d", "tepe gruplama acik", {**BASE, "group": True}))

ft = fc.frame_table()
W = fc.windows()
R = fc.win_rows(W)
Y = W["y_true"].to_numpy()
S1 = W["s1"].to_numpy()
print(f"frames={len(ft)} windows={len(W)} bins={bf.bin_info()}", flush=True)


def b2_features(st):
    cols = []
    t = np.arange(fc.L, dtype=float); tc = t - t.mean()
    for f in FEATS:
        v = st[f].to_numpy(float)
        v = np.where(np.isnan(v), FILL[f], v)[R]
        cols += [v[:, -1], v.mean(1), (v * tc).sum(1) / (tc ** 2).sum()]
    return np.stack(cols, 1).astype(np.float32)


def write_oof(tag, f, wins, proba):
    y_pred = proba.argmax(1)
    df = pd.DataFrame({"session": wins["session"].to_numpy(), "frame_id": wins["frame_id"].to_numpy(), "t": wins["t"].to_numpy(),
                       "loop_id": wins["loop_id"].to_numpy(), "fold": int(f), "y_true": wins["y_true"].to_numpy(),
                       "y_raw": wins["y_raw"].to_numpy(), "y_pred": y_pred})
    for c in range(fc.N_CLASSES):
        df[f"p{c}"] = proba[:, c]
    df[fc.OOF_COLS].to_csv(OOFD / f"{tag}__S1__f{f}.csv.gz", index=False, compression="gzip")


def anomaly(oof, name):
    labs = ac.labels_from_oof(oof, strict=True)
    assert labs is not None, name
    C, cnt = ac.build_cycles(name, "pred", labs)
    S, _ = ac.score_group(C, verbose=False)
    PF, SM = ac.evaluate(S)
    TAB = ac.summarize(PF, SM)
    d3 = TAB[TAB.variant == "D3"].set_index("atype")
    pf = PF[PF.variant == "D3"][["atype", "f", "auroc"]]
    return d3, pf, len(C)


def fold_f1(pred_by_fold):
    return np.array([fc.state_metrics(Y[S1 == f], pred_by_fold[f])["macro_f1"] for f in range(5)])


# ---- referans: Faz A B2 (OOF) -> per-fold F1 ve per-fold AUROC (04_anomaly_table'da kat bazli yok, ayni yoldan uretilir)
b2 = fc.load_oof("b2", "S1")
assert len(b2) == 16248
mm = W[["session", "frame_id", "s1"]].merge(b2[["session", "frame_id", "y_pred"]], on=["session", "frame_id"], how="left")
REF_F1 = np.array([fc.state_metrics(Y[S1 == f], mm.loc[mm.s1 == f, "y_pred"].to_numpy())["macro_f1"] for f in range(5)])
d3_ref, pf_ref, ncyc_ref = anomaly(b2, "b2ref")
REF_AUC = pf_ref[pf_ref.atype == "tum anomali"].sort_values("f")["auroc"].to_numpy()
rr = d3_ref.loc["tum anomali"]
print(f"[ref] Faz A B2 per-fold macro-F1 {np.round(REF_F1,4)} mean={REF_F1.mean():.6f}; per-fold AUROC(D3 tum) {np.round(REF_AUC,4)} mean={REF_AUC.mean():.6f}; "
      f"FAR={rr.far_a10:.4f} det={rr.det_a10:.4f} cycles={ncyc_ref}", flush=True)
assert abs(REF_AUC.mean() - 0.923697) < 1e-5, "referans AUROC Faz A ile ortusmuyor"

BASE_PTS = None
rows, fold_rows = [], []


def run_level(tag, axis, label, params):
    global BASE_PTS
    t0 = time.time()
    st = bf.frame_stats(params)
    by = bf.frame_bytes(params, st)
    if tag == "base":
        BASE_PTS = by["n_points_mean"]
    X = b2_features(st)
    preds, accs = {}, []
    for f in range(5):
        te, tr = np.flatnonzero(S1 == f), np.flatnonzero(S1 != f)
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=NJOBS)
        rf.fit(X[tr], Y[tr])
        pr = np.zeros((len(te), 6), np.float32)
        pr[:, rf.classes_.astype(int)] = rf.predict_proba(X[te])
        write_oof(f"sb_{tag}", f, W.iloc[te], pr)
        preds[f] = pr.argmax(1)
        accs.append(fc.state_metrics(Y[te], preds[f])["acc"])
    f1 = fold_f1(preds)
    oof = pd.concat([pd.read_csv(OOFD / f"sb_{tag}__S1__f{f}.csv.gz") for f in range(5)], ignore_index=True)
    d3, pf, ncyc = anomaly(oof, f"sb_{tag}")
    auc = pf[pf.atype == "tum anomali"].sort_values("f")["auroc"].to_numpy()
    assert len(auc) == 5 and not np.isnan(auc).any(), (tag, auc)
    c1, c2 = bf.paired_ci(f1, REF_F1), bf.paired_ci(auc, REF_AUC)
    kp = bf.knee_pass(c1, c2)
    a, sq, tp = d3.loc["tum anomali"], d3.loc["seq(135+145)"], d3.loc["tempo(fast+slow)"]
    row = dict(tag=tag, axis=axis, label=label, **{k: v for k, v in params.items()},
               f1_mean=f1.mean(), f1_sd=f1.std(ddof=1), acc_mean=float(np.mean(accs)),
               df1_mean=c1["mean"], df1_sd=c1["sd"], df1_lo=c1["lo"], df1_hi=c1["hi"], df1_min=c1["min"], df1_max=c1["max"],
               auroc=auc.mean(), auroc_sd=auc.std(ddof=1), dauc_mean=c2["mean"], dauc_sd=c2["sd"], dauc_lo=c2["lo"], dauc_hi=c2["hi"],
               dauc_min=c2["min"], dauc_max=c2["max"],
               auroc_seq=sq.auroc_mean, auroc_tempo=tp.auroc_mean, det_a10=a.det_a10, det_seq=sq.det_a10, det_tempo=tp.det_a10,
               far_a10=a.far_a10, far_se_a10=a.far_se_a10, n_pred_cycles=ncyc,
               pass_f1_point=kp["f1_point"], pass_f1_ci=kp["f1_ci"], pass_auc_point=kp["auc_point"], pass_auc_ci=kp["auc_ci"],
               pass_literal=kp["literal"], pass_point_only=kp["point_only"],
               pts_pct_of_base=100.0 * by["n_points_mean"] / BASE_PTS, sec=time.time() - t0, **by)
    rows.append(row)
    for f in range(5):
        fold_rows.append(dict(tag=tag, fold=f, macro_f1=f1[f], ref_macro_f1=REF_F1[f], auroc_d3_all=auc[f], ref_auroc=REF_AUC[f]))
    print(f"[{tag}] mF1={row['f1_mean']:.4f}±{row['f1_sd']:.4f} dF1={c1['mean']:+.4f} CI[{c1['lo']:+.4f},{c1['hi']:+.4f}] "
          f"| AUROC={row['auroc']:.4f}±{row['auroc_sd']:.4f} dAUC={c2['mean']:+.4f} CI[{c2['lo']:+.4f},{c2['hi']:+.4f}] "
          f"seq={row['auroc_seq']:.4f} tempo={row['auroc_tempo']:.4f} det10={row['det_a10']:.3f} FAR10={row['far_a10']:.4f} "
          f"| pts/frame={by['n_points_mean']:.2f} ({row['pts_pct_of_base']:.1f}% of base) empty={by['empty_frames']} V2={by['v2_B_frame']:.1f}B/frame "
          f"{by['v2_B_s']:.0f}B/s V3={by['v3_B_frame']:.1f}B ADCx={by['adc_ratio_v2']:.0f} | pass F1(pt,ci)=({kp['f1_point']},{kp['f1_ci']}) "
          f"AUC(pt,ci)=({kp['auc_point']},{kp['auc_ci']}) cycles={ncyc} {row['sec']:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(REP / "02_sensing_budget.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(REP / "02_sensing_budget_folds.csv", index=False)


for tag, axis, label, params in LEVELS:
    run_level(tag, axis, label, params)
    if tag == "base":
        fmax = max(abs(fold_rows[i]["macro_f1"] - fold_rows[i]["ref_macro_f1"]) for i in range(5))
        amax = max(abs(fold_rows[i]["auroc_d3_all"] - fold_rows[i]["ref_auroc"]) for i in range(5))
        print(f"[kapi2] taban (yeniden uretilmis istatistik + RF) vs Faz A B2: max|dF1 fold|={fmax:.2e}; max|d AUROC fold|={amax:.2e}; "
              f"FAR {rows[0]['far_a10']:.4f} vs {rr.far_a10:.4f}", flush=True)
        if fmax > 1e-6:
            print("[kapi2] DURDU: taban Faz A B2 ile ozdes degil", flush=True)
            sys.exit(3)

# ---- eksen basina diz secimi (en yalin = en az bayt/kare (V2); taban dahil)
df = pd.DataFrame(rows)
sel = {}
for axis in "abcd":
    g = df[(df.axis == axis) | (df.tag == "base")].sort_values(["v2_B_frame", "pts_pct_of_base"])
    ok = g[g.pass_literal]
    ok_pt = g[g.pass_point_only]
    sel[axis] = {"literal": ok.iloc[0].tag if len(ok) else None, "point_only": ok_pt.iloc[0].tag if len(ok_pt) else None,
                 "order": list(g.tag)}
    print(f"[diz] eksen {axis}: bayt sirasi {list(g.tag)} -> literal secim: {sel[axis]['literal']}; yalniz-nokta-tahmini secim: {sel[axis]['point_only']}", flush=True)
json.dump(sel, open(REP / "02_sensing_budget_selection.json", "w"), indent=1)

# ---- ikincil: eksen-basina secilen seviyelerin birlesimi (literal ve yalniz-nokta-tahmini)
ptag = {t: p for t, _, _, p in LEVELS}


def combine(key):
    c = dict(BASE)
    for axis in "abcd":
        t = sel[axis][key] or "base"
        for k in BASE:
            if ptag[t][k] != BASE[k]:
                c[k] = ptag[t][k]
    return c


comb = combine("literal")
if comb != BASE:
    print(f"[birlesim] literal secimlerin birlesimi: {comb}", flush=True)
    run_level("combo_literal", "combo", f"birlesim (literal secimler)", comb)
else:
    print("[birlesim] hicbir eksende taban disinda gecen seviye yok -> birlesim = taban, kosulmadi", flush=True)
comb2 = combine("point_only")
if comb2 != BASE and comb2 != comb:
    print(f"[birlesim2] yalniz nokta tahmini kuralina gore birlesim: {comb2}", flush=True)
    run_level("combo_point", "combo", f"birlesim (nokta-tahmin secimleri)", comb2)
else:
    print("[birlesim2] ek birlesim yok (taban ya da literal birlesimle ayni)", flush=True)
print("done", flush=True)

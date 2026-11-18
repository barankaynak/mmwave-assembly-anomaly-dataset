"""Faz B / B.1 - Kucuk ormanlar: B2 (7 kare istatistigi x last/mean/slope, RF) cip belleğine sigar mi?
DESIGN_B.md B.1'e birebir. Izgara: n_trees {10,25,50,100,300} x max_depth {4,6,8,10,12,None}; S1 5 kat; ozellik/tohum/sinif agirligi
Faz A B2 ile ayni (faz_a/02_state_baselines.py). Her yapilandirma icin tam OOF yazilir (reports/faz_b/oof/), anomali hatti
(pred segmentasyonu, D3 standart conformal, Faz A rolleri) bu OOF ile kosulur. Sayilarin hepsi stdout'a da basilir."""
import os, sys, time, json, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probe")); sys.path.insert(0, str(ROOT / "faz_a")); sys.path.insert(0, str(ROOT / "faz_a2"))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import fa_common as fc
import a2_cycles as ac

SEED = 20261104
NJOBS = 3
N_TREES = [10, 25, 50, 100, 300]
DEPTHS = [4, 6, 8, 10, 12, None]
FEATS = ["cx", "cy", "n_points", "snr_sum", "mean_abs_v", "std_x", "std_y"]
FILL = {"cx": -2.0, "cy": -2.0, "n_points": 0.0, "snr_sum": 0.0, "mean_abs_v": 0.0, "std_x": 0.0, "std_y": 0.0}
REP = ROOT / "reports" / "faz_b"
OOFD = REP / "oof"
OOFD.mkdir(parents=True, exist_ok=True)
F1_REF, AUC_REF = 0.856, 0.924          # onceden yazilmis kural sabitleri (B2: macro-F1, D3 AUROC 'tum anomali')
TOL_F1, TOL_AUC = 0.02, 0.03

ft = fc.frame_table()
W = fc.windows()
R = fc.win_rows(W)
Y = W["y_true"].to_numpy()
S1 = W["s1"].to_numpy()


def b2_features(rows):
    names = []
    cols = []
    t = np.arange(fc.L, dtype=float); tc = t - t.mean()
    for f in FEATS:
        v = ft[f].to_numpy(float)
        v = np.where(np.isnan(v), FILL[f], v)[rows]
        cols += [v[:, -1], v.mean(1), (v * tc).sum(1) / (tc ** 2).sum()]
        names += [f + "_last", f + "_mean", f + "_slope"]
    return np.stack(cols, 1).astype(np.float32), names


X, NAMES = b2_features(R)
print(f"windows={len(W)} features={len(NAMES)} folds={sorted(set(S1))}", flush=True)


def tree_depths(tree):
    """Dugum derinligi (kok=0); dugumler on-sirada oldugundan cocuk id > ebeveyn id."""
    n = tree.node_count
    d = np.zeros(n, np.int32)
    L_, R_ = tree.children_left, tree.children_right
    for i in range(n):
        if L_[i] != -1:
            d[L_[i]] = d[i] + 1
            d[R_[i]] = d[i] + 1
    return d


def write_oof(tag, f, wins, proba):
    y_pred = proba.argmax(1)
    df = pd.DataFrame({"session": wins["session"].to_numpy(), "frame_id": wins["frame_id"].to_numpy(), "t": wins["t"].to_numpy(),
                       "loop_id": wins["loop_id"].to_numpy(), "fold": int(f), "y_true": wins["y_true"].to_numpy(),
                       "y_raw": wins["y_raw"].to_numpy(), "y_pred": y_pred})
    for c in range(fc.N_CLASSES):
        df[f"p{c}"] = proba[:, c]
    df[fc.OOF_COLS].to_csv(OOFD / f"{tag}__S1__f{f}.csv.gz", index=False, compression="gzip")


def fit_cfg(nt, md, tr_idx):
    rf = RandomForestClassifier(n_estimators=nt, max_depth=md, class_weight="balanced", random_state=SEED, n_jobs=NJOBS)
    rf.fit(X[tr_idx], Y[tr_idx])
    return rf


# ---- referans B2 (Faz A OOF) ve Faz A B2 x pred D3 sayilari
b2_oof = fc.load_oof("b2", "S1")
assert len(b2_oof) == 16248
mm = W[["session", "frame_id", "s1"]].merge(b2_oof[["session", "frame_id", "y_pred", "y_true"]], on=["session", "frame_id"], how="left")
B2_FOLD_F1 = np.array([fc.state_metrics(mm.loc[mm.s1 == f, "y_true"].to_numpy(), mm.loc[mm.s1 == f, "y_pred"].to_numpy())["macro_f1"] for f in range(5)])
print("Faz A B2 S1 macro-F1 per fold:", np.round(B2_FOLD_F1, 4), "mean=%.4f sd=%.4f" % (B2_FOLD_F1.mean(), B2_FOLD_F1.std(ddof=1)), flush=True)
TA = pd.read_csv(ROOT / "reports" / "faz_a" / "04_anomaly_table.csv")
ref = TA[(TA.state_source == "b2") & (TA.segmentation == "pred") & (TA.variant == "D3") & (TA.atype == "tum anomali")].iloc[0]
print(f"Faz A B2 x pred D3 tum anomali: AUROC={ref.auroc_mean:.6f} sd={ref.auroc_sd:.4f} det_a10={ref.det_a10:.4f} FAR_a10={ref.far_a10:.4f}", flush=True)

rows = []
t_all = time.time()
for nt, md in itertools.product(N_TREES, DEPTHS):
    t0 = time.time()
    tag = f"tf_n{nt}_d{'N' if md is None else md}"
    f1s, accs, nodes_f, comps_f, worst_f, maxd_f = [], [], [], [], [], []
    parts = []
    for f in range(5):
        te_idx = np.flatnonzero(S1 == f); tr_idx = np.flatnonzero(S1 != f)
        rf = fit_cfg(nt, md, tr_idx)
        pr = np.zeros((len(te_idx), 6), np.float32)
        pr[:, rf.classes_.astype(int)] = rf.predict_proba(X[te_idx])
        wins = W.iloc[te_idx]
        write_oof(tag, f, wins, pr)
        m = fc.state_metrics(Y[te_idx], pr.argmax(1))
        f1s.append(m["macro_f1"]); accs.append(m["acc"])
        nodes_f.append(sum(e.tree_.node_count for e in rf.estimators_))
        # karsilastirma sayisi: her agacta kok->yaprak ic dugum sayisi (= yaprak derinligi)
        tot = np.zeros(len(te_idx))
        for e in rf.estimators_:
            tot += tree_depths(e.tree_)[e.apply(X[te_idx])]
        comps_f.append(tot.mean())
        worst_f.append(sum(e.tree_.max_depth for e in rf.estimators_))
        maxd_f.append(max(e.tree_.max_depth for e in rf.estimators_))
    # tam-veri modeli (dagitim boyutu icin; OOF'a girmez)
    rf_full = fit_cfg(nt, md, np.arange(len(W)))
    nodes_full = sum(e.tree_.node_count for e in rf_full.estimators_)
    # anomali hatti: bu yapilandirmanin tam OOF'u
    oof = pd.concat([pd.read_csv(OOFD / f"{tag}__S1__f{f}.csv.gz") for f in range(5)], ignore_index=True)
    labs = ac.labels_from_oof(oof, strict=True)
    assert labs is not None, tag
    C, cnt = ac.build_cycles(tag, "pred", labs)
    S, _ = ac.score_group(C, verbose=False)
    PF, SM = ac.evaluate(S)
    TAB = ac.summarize(PF, SM)
    d3 = TAB[(TAB.variant == "D3")].set_index("atype")
    a = d3.loc["tum anomali"]
    f1s = np.array(f1s)
    row = dict(config=tag, n_trees=nt, max_depth=("None" if md is None else md),
               f1_mean=f1s.mean(), f1_sd=f1s.std(ddof=1), acc_mean=float(np.mean(accs)),
               d_f1_vs_b2_mean=float((f1s - B2_FOLD_F1).mean()), d_f1_vs_b2_sd=float((f1s - B2_FOLD_F1).std(ddof=1)),
               nodes_mean=float(np.mean(nodes_f)), nodes_max=int(np.max(nodes_f)), nodes_full=int(nodes_full),
               bytes8_mean=float(np.mean(nodes_f)) * 8, bytes12_mean=float(np.mean(nodes_f)) * 12,
               bytes8_full=nodes_full * 8, bytes12_full=nodes_full * 12,
               comps_per_window=float(np.mean(comps_f)), mean_path_len=float(np.mean(comps_f)) / nt,
               worst_comps=float(np.mean(worst_f)), max_depth_realised=int(np.max(maxd_f)),
               auroc_all=a.auroc_mean, auroc_all_sd=a.auroc_sd, auroc_seq=d3.loc["seq(135+145)", "auroc_mean"],
               auroc_tempo=d3.loc["tempo(fast+slow)", "auroc_mean"], det_a10_all=a.det_a10,
               far_a10=a.far_a10, far_se_a10=a.far_se_a10, n_pred_cycles=int(len(C)),
               sec=time.time() - t0)
    rows.append(row)
    print(f"[{tag}] mF1={row['f1_mean']:.4f}±{row['f1_sd']:.4f} dB2={row['d_f1_vs_b2_mean']:+.4f} nodes={row['nodes_mean']:.0f} (full {nodes_full}) "
          f"B8={row['bytes8_mean']/1024:.1f}KB B12={row['bytes12_mean']/1024:.1f}KB comps={row['comps_per_window']:.1f} "
          f"(path {row['mean_path_len']:.2f}, worst {row['worst_comps']:.0f}) | AUROC={row['auroc_all']:.4f}±{row['auroc_all_sd']:.4f} "
          f"seq={row['auroc_seq']:.4f} tempo={row['auroc_tempo']:.4f} det10={row['det_a10_all']:.3f} FAR10={row['far_a10']:.4f} "
          f"cycles={row['n_pred_cycles']} {row['sec']:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(REP / "01_tiny_forests.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv(REP / "01_tiny_forests.csv", index=False)
print(f"grid done in {time.time()-t_all:.0f}s", flush=True)

# ---- esdegerlik kapisi: (300, None) = Faz A B2
g = df[(df.n_trees == 300) & (df.max_depth == "None")].iloc[0]
print(f"[kapi] (300,None) vs Faz A B2: mF1 {g.f1_mean:.6f} vs {B2_FOLD_F1.mean():.6f}; AUROC {g.auroc_all:.6f} vs {ref.auroc_mean:.6f}; FAR {g.far_a10:.6f} vs {ref.far_a10:.6f}; "
      f"max|dF1 fold| = {abs(g.d_f1_vs_b2_mean):.2e} (sd {g.d_f1_vs_b2_sd:.2e})", flush=True)

# ---- onceden yazilmis secim kurali (hazir sabitler: 0.856-0.02, 0.924-0.03)
f1_thr, auc_thr = F1_REF - TOL_F1, AUC_REF - TOL_AUC
ok = df[(df.f1_mean >= f1_thr) & (df.auroc_all >= auc_thr)].copy()
print(f"[kural] macro-F1 >= {f1_thr:.3f} VE AUROC >= {auc_thr:.3f}: {len(ok)} / {len(df)} yapilandirma saglar", flush=True)
if len(ok):
    ok = ok.sort_values(["nodes_mean", "comps_per_window"])
    print(ok[["config", "f1_mean", "auroc_all", "nodes_mean", "bytes8_mean", "bytes12_mean", "comps_per_window"]].to_string(), flush=True)
    sel = ok.iloc[0]
    print(f"[kural] SECILEN: {sel.config} nodes={sel.nodes_mean:.0f} B8={sel.bytes8_mean/1024:.1f}KB B12={sel.bytes12_mean/1024:.1f}KB", flush=True)
    json.dump({"selected": sel.config, "n_pass": int(len(ok))}, open(REP / "01_tiny_forests_selected.json", "w"))
else:
    print("[kural] sigan yapilandirma yok", flush=True)
    json.dump({"selected": None, "n_pass": 0}, open(REP / "01_tiny_forests_selected.json", "w"))
print("done", flush=True)

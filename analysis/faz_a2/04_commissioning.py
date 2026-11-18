"""Faz A2 / A2.4 - devreye alma maliyeti (DESIGN_A2.md, on-kayitli).

(a) Anomali: R=50 tohumlu rastgele dagitim; s02'nin 51 normal ToF dongusu -> test 11 / kal 20 / egitim havuzu 20.
    Havuzdan ilk n dongu (n in 5,10,15,20) ile D0/D1/D2 ogrenilir; kaynak B2 x pred ve ToF x tof (mevcut S1 OOF).
    D3 AUROC (tur basina + tum anomali) ve alpha=0.10'da FAR; tekrarlar uzerinden ortalama +- sd.
(b) Durum: S1 katlarinda egitim havuzundan OTURUM BASINA k etiketli dongu (k in 1,2,5,10,20,hepsi), R=10;
    B2 (21 ozellik + RF 300) ve B1 egitilir, kat f'de test; macro-F1 ortalama +- sd; k -> kayit dakikasi.

Kullanim:  tools/local_run.sh faz_a2/04_commissioning.py --part a|b|report|all
CSV'ler reports/faz_a2/ altina; rapor 04_commissioning.md (part report/all).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a"))
sys.path.insert(0, str(ROOT / "faz_a2"))
sys.path.insert(0, str(ROOT / "probe"))
import numpy as np
import pandas as pd

import a2_cycles as ac
import fa_common as fc

OUT2 = ROOT / "reports" / "faz_a2"
OUT2.mkdir(parents=True, exist_ok=True)
SEED = 20261104
T0 = time.time()


def P(*a):
    print(*a, flush=True)


# =============================================================================== (a) ANOMALI
R_A = 50
N_TEST, N_CAL, N_POOL = 11, 20, 20
N_LIST = (5, 10, 15, 20)
CELLS = [("b2", "pred"), ("tof", "tof")]
CELL_NAME = {("b2", "pred"): "B2 x pred", ("tof", "tof"): "ToF x tof"}
GROUPS = list(ac.AGROUPS)            # seq135, seq145, fast, slow, seq(135+145), tempo(fast+slow), tum anomali


def part_a():
    P(f"[{time.time()-T0:.0f}s] (a) anomali: kaynak etiketleri")
    LAB = ac.source_labels(sources=("tof", "b2"), verbose=True)
    assert "b2" in LAB and "tof" in LAB
    AL = ac.ctx().AL
    loops = sorted(AL[(AL["session"] == "session_02_normal_seq125") & (AL["use"] == "normal_crossfit")]["loop_id"].astype(int))
    P(f"s02 normal ToF dongu sayisi (katlanan): {len(loops)}")
    assert len(loops) == 51 == N_TEST + N_CAL + N_POOL

    rows_g, rows_f, info = [], [], []
    for (src, seg) in CELLS:
        C, _ = ac.build_cycles(src, seg, LAB[src])
        cl = C.to_dict("records")
        norm = [c for c in cl if c["atype"] == "normal"]
        anom = [c for c in cl if c["atype"] != "normal"]
        by_loop = {}
        for c in sorted(norm, key=lambda c: c["cycle_idx"]):
            by_loop.setdefault(int(c["anchor_loop"]), []).append(c)
        assert set(by_loop) <= set(loops)
        n_per = [len(by_loop.get(l, [])) for l in loops]
        info.append(dict(cell=CELL_NAME[(src, seg)], n_normal_cycles=len(norm), n_anom_cycles=len(anom),
                         loops_with_0=sum(1 for x in n_per if x == 0), loops_with_2plus=sum(1 for x in n_per if x >= 2),
                         mean_dur_normal_s=float(np.mean([c["dur_s"] for c in norm]))))
        P(f"  {CELL_NAME[(src, seg)]}: normal dongu={len(norm)} anomali dongu={len(anom)}; "
          f"0-dongulu ToF dongusu={info[-1]['loops_with_0']}, 2+={info[-1]['loops_with_2plus']}; "
          f"ort. normal sure={info[-1]['mean_dur_normal_s']:.2f}s")
        for r in range(R_A):
            perm = np.random.RandomState([SEED, r]).permutation(np.array(loops))      # tum kaynaklarda AYNI dagitim
            te_l, ca_l, po_l = perm[:N_TEST], perm[N_TEST:N_TEST + N_CAL], perm[N_TEST + N_CAL:]
            gat = lambda L_: [c for l in L_ for c in by_loop.get(int(l), [])]
            te_n, ca_n, pool = gat(te_l), gat(ca_l), gat(po_l)           # havuz: karistirilmis dongu sirasi, dongu ici zaman sirasi
            for n in N_LIST:
                tr = pool[:n]
                par = ac.fit_params(tr)
                sc_ca = ac.score_cycles(ca_n, par)
                sc_te = ac.score_cycles(te_n, par)
                sc_an = ac.score_cycles(anom, par)
                p = {}
                for v, k in (("D1", "p1"), ("D2", "p2")):
                    p[k + "n"] = ac.conf_p(sc_ca[v].to_numpy(), sc_te[v].to_numpy())
                    p[k + "a"] = ac.conf_p(sc_ca[v].to_numpy(), sc_an[v].to_numpy())
                p3n = ac.d3(p["p1n"], p["p2n"])
                p3a = ac.d3(p["p1a"], p["p2a"])
                nat = sc_te["natural_dev"].to_numpy(bool)
                far = {f"far_a{int(a*100):02d}": float(ac.alarm(p3n, a).mean()) for a in ac.ALPHAS}
                far["far_excl_nat_a10"] = float(ac.alarm(p3n[~nat], 0.10).mean()) if (~nat).any() else float("nan")
                rows_f.append(dict(src=src, seg=seg, cell=CELL_NAME[(src, seg)], rep=r, n=n, n_train_actual=len(tr),
                                   n_cal_cycles=len(ca_n), n_test_normal=len(te_n), n_natdev_test=int(nat.sum()), **far))
                at = sc_an["atype"].to_numpy()
                for g in GROUPS:
                    m = np.isin(at, ac.AGROUPS[g])
                    au, ap = ac.auc_ap(1.0 - p3n, 1.0 - p3a[m])
                    rows_g.append(dict(src=src, seg=seg, cell=CELL_NAME[(src, seg)], rep=r, n=n, group=g, n_pos=int(m.sum()),
                                       auroc=au, auprc=ap, det_a10=float(ac.alarm(p3a[m], 0.10).mean()),
                                       det_a05=float(ac.alarm(p3a[m], 0.05).mean())))
            if r % 10 == 9:
                P(f"  [{time.time()-T0:.0f}s] {CELL_NAME[(src, seg)]} tekrar {r+1}/{R_A}")
    G = pd.DataFrame(rows_g)
    F = pd.DataFrame(rows_f)
    G.to_csv(OUT2 / "04_comm_anomaly_groups.csv", index=False)
    F.to_csv(OUT2 / "04_comm_anomaly_far.csv", index=False)
    pd.DataFrame(info).to_csv(OUT2 / "04_comm_anomaly_info.csv", index=False)
    P(f"[{time.time()-T0:.0f}s] (a) bitti: {len(G)} grup satiri, {len(F)} FAR satiri")


def _msd(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan"), len(x)) if len(x) else (float("nan"),) * 2 + (0,)


def summarize_a():
    G = pd.read_csv(OUT2 / "04_comm_anomaly_groups.csv")
    F = pd.read_csv(OUT2 / "04_comm_anomaly_far.csv")
    rows = []
    for (cell, n, g), d in G.groupby(["cell", "n", "group"], sort=False):
        m1, s1, k1 = _msd(d["auroc"]); m2, s2, _ = _msd(d["det_a10"]); m3, s3, _ = _msd(d["det_a05"])
        rows.append(dict(cell=cell, n=n, group=g, n_reps_valid=k1, auroc_mean=m1, auroc_sd=s1, det_a10_mean=m2, det_a10_sd=s2,
                         det_a05_mean=m3, det_a05_sd=s3))
    SG = pd.DataFrame(rows)
    rows = []
    for (cell, n), d in F.groupby(["cell", "n"], sort=False):
        r = dict(cell=cell, n=n, n_train_actual_mean=float(d["n_train_actual"].mean()), n_cal_mean=float(d["n_cal_cycles"].mean()),
                 n_test_mean=float(d["n_test_normal"].mean()))
        for c in ("far_a10", "far_a05", "far_excl_nat_a10"):
            m, s, _ = _msd(d[c]); r[c + "_mean"], r[c + "_sd"] = m, s
        rows.append(r)
    SF = pd.DataFrame(rows)
    SG.to_csv(OUT2 / "04_comm_anomaly_summary.csv", index=False)
    SF.to_csv(OUT2 / "04_comm_anomaly_far_summary.csv", index=False)
    return SG, SF


# =============================================================================== (b) DURUM
R_B = 10
K_LIST = (1, 2, 5, 10, 20)           # + 'hepsi'
FEATS = ["cx", "cy", "n_points", "snr_sum", "mean_abs_v", "std_x", "std_y"]
FILL = {"cx": -2.0, "cy": -2.0, "n_points": 0.0, "snr_sum": 0.0, "mean_abs_v": 0.0, "std_x": 0.0, "std_y": 0.0}
TAU_GRID = np.concatenate([np.arange(0.002, 0.02, 0.002), np.arange(0.02, 0.62, 0.02), [np.inf]])   # faz_a/02 ile ayni


def part_b():
    from sklearn.ensemble import RandomForestClassifier
    ft = fc.frame_table()
    W = fc.windows()
    RW = fc.win_rows(W)
    CX, CY = ft["cx"].to_numpy(float), ft["cy"].to_numpy(float)

    def b2_features(rows):                          # faz_a/02 ile birebir
        cols = []
        t = np.arange(fc.L, dtype=float); tc = t - t.mean()
        for f in FEATS:
            v = ft[f].to_numpy(float)
            v = np.where(np.isnan(v), FILL[f], v)[rows]
            cols += [v[:, -1], v.mean(1), (v * tc).sum(1) / (tc ** 2).sum()]
        return np.stack(cols, 1).astype(np.float32)

    X = b2_features(RW)
    LX, LY = CX[RW[:, -1]], CY[RW[:, -1]]
    Y = W["y_true"].to_numpy()
    gid = W["gid"].to_numpy(); sidx = W["session_idx"].to_numpy(); s1 = W["s1"].to_numpy()
    sessions = sorted(W["session"].unique())
    gframes = ft.groupby("gid").size().to_dict()               # gid -> kare sayisi (dongu suresi = kare/10 s)

    def centroids(mask):
        c = np.full((5, 2), np.nan)
        for k in range(1, 6):
            m = mask & (Y == k) & ~np.isnan(LX)
            if m.sum():
                c[k - 1] = [LX[m].mean(), LY[m].mean()]
        return c

    def b1_predict(idx, cent, tau):
        d = np.sqrt((LX[idx, None] - cent[None, :, 0]) ** 2 + (LY[idx, None] - cent[None, :, 1]) ** 2)
        d = np.where(np.isnan(d), np.inf, d)
        return np.where(d.min(1) > tau, 0, d.argmin(1) + 1)

    def macro(y, p):
        from sklearn.metrics import f1_score
        return f1_score(y, p, labels=list(range(6)), average="macro", zero_division=0)

    def fit_b1(tr_idx, f):
        """Faz A ile ayni: ic dogrulama (grup bazli, oturum tabakali, %10) ile tau; ic dogrulama bos/ic egitim bos ise
        (k=1) resubstitution: tau egitim pencerelerinin kendisinde secilir (Sapmalar'a yazildi)."""
        val = fc.inner_val_split(gid[tr_idx], seed=SEED + f, strata=sidx[tr_idx])
        mode = "inner_val"
        if val.sum() == 0 or (~val).sum() == 0:
            itr = np.zeros(len(W), bool); itr[tr_idx] = True; iva = tr_idx; mode = "resub"
        else:
            itr = np.zeros(len(W), bool); itr[tr_idx[~val]] = True; iva = tr_idx[val]
        cin = centroids(itr)
        sc = [macro(Y[iva], b1_predict(iva, cin, t)) for t in TAU_GRID]
        tau = float(TAU_GRID[int(np.argmax(sc))])
        return centroids(np.isin(np.arange(len(W)), tr_idx)), tau, mode

    def run_one(tr_idx, te_idx, f):
        out = {}
        cent, tau, mode = fit_b1(tr_idx, f)
        p1 = b1_predict(te_idx, cent, tau)
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=3)
        rf.fit(X[tr_idx], Y[tr_idx])
        pr = np.zeros((len(te_idx), 6), np.float32)
        pr[:, rf.classes_.astype(int)] = rf.predict_proba(X[te_idx])
        p2 = pr.argmax(1)
        for name, p in (("b1", p1), ("b2", p2)):
            m = fc.state_metrics(Y[te_idx], p)
            r = dict(acc=m["acc"], macro_f1=m["macro_f1"])
            for si, s in enumerate(sessions):
                mm = sidx[te_idx] == si
                r[f"mf1_{s[:10]}"] = fc.state_metrics(Y[te_idx][mm], p[mm])["macro_f1"] if mm.any() else float("nan")
            out[name] = r
        return out, p1, p2, mode, tau

    rows, agree = [], []
    for f in range(fc.N_FOLDS):
        te_idx = np.flatnonzero(s1 == f)
        tr_all = np.flatnonzero(s1 != f)
        for k in list(K_LIST) + ["all"]:
            reps = [0] if k == "all" else range(R_B)
            for rep in reps:
                if k == "all":
                    tr_idx, ch = tr_all, np.unique(gid[tr_all])
                else:
                    rng = np.random.RandomState([SEED, f, rep])                 # n-ic-ice: ilk k = permutasyonun basi
                    ch = []
                    for si in range(len(sessions)):
                        g = np.unique(gid[tr_all][sidx[tr_all] == si])
                        ch += rng.permutation(g)[:k].tolist()
                    ch = np.array(ch)
                    tr_idx = tr_all[np.isin(gid[tr_all], ch)]
                out, p1, p2, mode, tau = run_one(tr_idx, te_idx, f)
                frames = int(sum(gframes[int(g)] for g in ch))
                for name in ("b1", "b2"):
                    rows.append(dict(model=name, fold=f, k=("all" if k == "all" else k), rep=rep, n_loops=len(ch),
                                     n_train_windows=len(tr_idx), rec_min_total=frames * ac.DT / 60.0, b1_tau=tau if name == "b1" else np.nan,
                                     b1_mode=mode if name == "b1" else "", **out[name]))
                if k == "all":
                    for name, pp in (("b1", p1), ("b2", p2)):
                        oof = fc.load_oof(name, "S1"); oo = oof[oof["fold"] == f].set_index(["session", "frame_id"])["y_pred"]
                        mine = pd.Series(pp, index=pd.MultiIndex.from_frame(W.iloc[te_idx][["session", "frame_id"]]))
                        a = float((mine.sort_index().to_numpy() == oo.sort_index().reindex(mine.sort_index().index).to_numpy()).mean())
                        agree.append(dict(model=name, fold=f, agree_with_fazA_oof=a))
                        P(f"   'hepsi' f{f} {name}: Faz A OOF ile tahmin uyumu = {a:.6f}")
            sub = [r for r in rows if r["fold"] == f and str(r["k"]) == str(k)]
            P(f"[{time.time()-T0:.0f}s] f{f} k={k}: " + "  ".join(
                f"{m} mF1={np.mean([r['macro_f1'] for r in sub if r['model']==m]):.4f}" for m in ("b1", "b2")))
    D = pd.DataFrame(rows)
    D.to_csv(OUT2 / "04_comm_state_runs.csv", index=False)
    pd.DataFrame(agree).to_csv(OUT2 / "04_comm_state_agree.csv", index=False)
    # ortalama dongu suresi (oturum basina)
    AL = ac.ctx().AL
    dur = []
    for s in sessions:
        a = AL[AL["session"] == s]
        full = a[a["is_cycle"].astype(bool)] if s != "session_01_training" else a
        dur.append(dict(session=s, n_loops=len(a), n_complete_cycles=int(a["is_cycle"].sum()),
                        mean_cycle_s=float(full["n_frames"].mean() * ac.DT), mean_all_loops_s=float(a["n_frames"].mean() * ac.DT)))
    pd.DataFrame(dur).to_csv(OUT2 / "04_comm_state_durations.csv", index=False)
    P(f"[{time.time()-T0:.0f}s] (b) bitti: {len(D)} satir")


def summarize_b():
    D = pd.read_csv(OUT2 / "04_comm_state_runs.csv")
    D["k"] = D["k"].astype(str)
    dur = pd.read_csv(OUT2 / "04_comm_state_durations.csv")
    rows = []
    for (model, k), d in D.groupby(["model", "k"], sort=False):
        # tekrar basina: 5 katin ortalamasi; sonra tekrarlar uzerinden ortalama +- sd
        pr = d.groupby("rep")[["macro_f1", "acc", "rec_min_total", "n_train_windows"]].mean()
        sessc = [c for c in d.columns if c.startswith("mf1_")]
        ps = d.groupby("rep")[sessc].mean()
        r = dict(model=model, k=k, n_reps=len(pr), mf1_mean=float(pr["macro_f1"].mean()),
                 mf1_sd_reps=float(pr["macro_f1"].std(ddof=1)) if len(pr) > 1 else float("nan"),
                 mf1_sd_folds=float(d.groupby("fold")["macro_f1"].mean().std(ddof=1)),
                 mf1_min_rep=float(pr["macro_f1"].min()), mf1_max_rep=float(pr["macro_f1"].max()),
                 acc_mean=float(pr["acc"].mean()), acc_sd_reps=float(pr["acc"].std(ddof=1)) if len(pr) > 1 else float("nan"),
                 rec_min_total_actual=float(pr["rec_min_total"].mean()), n_train_windows_mean=float(pr["n_train_windows"].mean()))
        for c in sessc:
            r[c] = float(ps[c].mean())
        rows.append(r)
    S = pd.DataFrame(rows)
    # nominal dakika: k x ortalama dongu suresi
    tot_cycle_s = float(dur["mean_cycle_s"].sum()); mean_cycle_s = float(dur["mean_cycle_s"].mean())
    S["k_num"] = S["k"].map(lambda x: np.nan if x == "all" else float(x))
    S["nominal_min_per_session"] = S["k_num"] * mean_cycle_s / 60.0
    S["nominal_min_total_6sessions"] = S["k_num"] * tot_cycle_s / 60.0
    S.to_csv(OUT2 / "04_comm_state_summary.csv", index=False)
    return S, dur, D


# =============================================================================== RAPOR
def fm(m, s, d=3):
    if pd.isna(m):
        return "NaN"
    return f"{m:.{d}f} ± {s:.{d}f}" if not pd.isna(s) else f"{m:.{d}f}"


def pct(m, s, d=1):
    if pd.isna(m):
        return "NaN"
    return f"{100*m:.{d}f} ± {100*s:.{d}f}" if not pd.isna(s) else f"{100*m:.{d}f}"


def report():
    SG, SF = summarize_a()
    S, dur, D = summarize_b()
    info = pd.read_csv(OUT2 / "04_comm_anomaly_info.csv")
    agree = pd.read_csv(OUT2 / "04_comm_state_agree.csv")
    tab = pd.read_csv(fc.OUT / "04_anomaly_table.csv")
    st_a = pd.read_csv(fc.OUT / "03_state_table.csv") if (fc.OUT / "03_state_table.csv").exists() else None
    L = []
    w = L.append
    mean_norm = {r.cell: r.mean_dur_normal_s for r in info.itertuples()}
    w("# A2.4 — Devreye alma maliyeti")
    w("")
    w("`faz_a2/04_commissioning.py` tarafından üretilir (DESIGN_A2.md A2.4). **İkincil analiz**: makalenin birincil sonuçları Faz A'dır; "
      "bu analiz ana model seçimini (B2 + D3) değiştirmez. Tüm sayılar stdout'a da yazılır; seçici raporlama yoktur.")
    w("")
    w("## (a) Anomali dedektörü: kaç normal döngü gerekir?")
    w("")
    w("**Kurulum.** R = 50 tekrar; her tekrarda `RandomState([20261104, r])` ile s02'nin 51 katlanmış normal **ToF döngüsü** (loop_id) "
      "test 11 / kalibrasyon 20 / eğitim havuzu 20 olarak dağıtılır. Dağıtım iki kaynakta **aynıdır** (eşleşik). `pred` segmentasyonunda "
      "döngüler Faz A'daki gibi ToF döngüsüne eşlenir; dağıtım ToF döngüsü bazındadır, bir ToF döngüsüne 0, 1 ya da 2+ tahmin döngüsü düşebilir. "
      "Eğitimde havuzun **ilk n döngüsü** (n ∈ {5, 10, 15, 20}; havuz sırası = karıştırılmış ToF döngü sırası, aynı ToF döngüsü içinde zaman sırası) "
      "kullanılır; `pred` hücresinde havuz 20 ToF döngüsünden gelen tahmin döngülerinin tamamıdır, yani n **döngü sayısıdır** (ToF döngüsü sayısı değil). "
      "Anomali döngüleri (s03–s06) her tekrarda test. D0/D1/D2 `a2_cycles.fit_params` ile n döngüden öğrenilir; kalibrasyon skorları 20 kalibrasyon "
      "döngüsündendir; p = standart conformal (Faz A, `cal >= s`); D3 = min(1, 2·min(p1, p2)); AUROC skoru = 1 − p3 (Faz A ile aynı). "
      "FAR = test normal döngülerinin p3 ≤ α oranı (doğal sapmalar dâhil; hariç sütunu ayrıca). Yalnız mevcut S1 OOF tahminleri (yeniden eğitim yok).")
    w("")
    w("| hücre | normal döngü | anomali döngü | 0 döngülü ToF döngüsü | 2+ döngülü | ort. normal döngü süresi (s) |")
    w("|---|---|---|---|---|---|")
    for r in info.itertuples():
        w(f"| {r.cell} | {r.n_normal_cycles} | {r.n_anom_cycles} | {r.loops_with_0} | {r.loops_with_2plus} | {r.mean_dur_normal_s:.2f} |")
    w("")
    w("### D3 AUROC (ortalama ± sd, 50 tekrar)")
    w("")
    for cell in ["B2 x pred", "ToF x tof"]:
        w(f"**{cell}**")
        w("")
        w("| n eğitim | " + " | ".join(ac.AGROUPS) + " | FAR@0,10 | FAR@0,05 | FAR@0,10 (doğal sapma hariç) | ort. fiili eğitim / kal. / test |")
        w("|---|" + "---|" * (len(ac.AGROUPS) + 4))
        for n in ac.N_LIST if hasattr(ac, "N_LIST") else N_LIST:
            row = [str(n)]
            for g in ac.AGROUPS:
                x = SG[(SG["cell"] == cell) & (SG["n"] == n) & (SG["group"] == g)].iloc[0]
                row.append(fm(x.auroc_mean, x.auroc_sd))
            f = SF[(SF["cell"] == cell) & (SF["n"] == n)].iloc[0]
            row += [fm(f.far_a10_mean, f.far_a10_sd), fm(f.far_a05_mean, f.far_a05_sd), fm(f.far_excl_nat_a10_mean, f.far_excl_nat_a10_sd),
                    f"{f.n_train_actual_mean:.1f} / {f.n_cal_mean:.1f} / {f.n_test_mean:.1f}"]
            w("| " + " | ".join(row) + " |")
        w("")
    w("### Tespit oranı α = 0,10 ve 0,05 (D3; ortalama ± sd)")
    w("")
    for cell in ["B2 x pred", "ToF x tof"]:
        w(f"**{cell}**")
        w("")
        w("| n | " + " | ".join(f"{g} @0,10" for g in ("seq(135+145)", "tempo(fast+slow)", "tum anomali")) + " | tum anomali @0,05 |")
        w("|---|---|---|---|---|")
        for n in N_LIST:
            row = [str(n)]
            for g in ("seq(135+145)", "tempo(fast+slow)", "tum anomali"):
                x = SG[(SG["cell"] == cell) & (SG["n"] == n) & (SG["group"] == g)].iloc[0]
                row.append(pct(x.det_a10_mean, x.det_a10_sd))
            x = SG[(SG["cell"] == cell) & (SG["n"] == n) & (SG["group"] == "tum anomali")].iloc[0]
            row.append(pct(x.det_a05_mean, x.det_a05_sd))
            w("| " + " | ".join(row) + " |")
        w("")
        w("(yüzde; tüm tür başına tespit sayıları CSV'de)")
        w("")
    # Faz A referansi
    w("### Faz A referansı (D3, aynı hücreler; birincil, ön-kayıtlı)")
    w("")
    w("Faz A: eğitim ≈ 20 (2 kat), kalibrasyon ≈ 20,5 (2 kat), test ≈ 10 normal döngü/kat, 5 kat çapraz-uyum. Bu A2.4 n = 20 satırının en yakın karşılığıdır "
      "(farklar: tekrarlı rastgele dağıtım ve 11 test döngüsü).")
    w("")
    w("| hücre | AUROC tüm anomali | AUROC seq(135+145) | AUROC tempo | FAR@0,10 | FAR@0,05 | det@0,10 tüm |")
    w("|---|---|---|---|---|---|---|")
    for (src, seg), nm in CELL_NAME.items():
        t = tab[(tab.state_source == src) & (tab.segmentation == seg) & (tab.variant == "D3")]
        g = lambda a: t[t.atype == a].iloc[0]
        a, b_, c_ = g("tum anomali"), g("seq(135+145)"), g("tempo(fast+slow)")
        w(f"| {nm} | {a.auroc_mean:.3f} ± {a.auroc_sd:.3f} | {b_.auroc_mean:.3f} ± {b_.auroc_sd:.3f} | {c_.auroc_mean:.3f} ± {c_.auroc_sd:.3f} | "
          f"{a.far_a10:.3f} | {a.far_a05:.3f} | {100*a.det_a10:.1f}% |")
    w("")
    # betimleyici yeterlilik olcutu
    w("### Betimleyici \"yeter\" ölçütü (sonradan tanımlı, ön-kayıtlı DEĞİL)")
    w("")
    for cell in ["B2 x pred", "ToF x tof"]:
        t = SG[(SG["cell"] == cell) & (SG["group"] == "tum anomali")].set_index("n")
        ref = t.loc[20, "auroc_mean"]
        ok = [n for n in N_LIST if t.loc[n, "auroc_mean"] >= ref - 0.02]
        nmin = min(ok) if ok else None
        mn = nmin * mean_norm[cell] / 60 if nmin else float("nan")
        w(f"- {cell}: tüm-anomali D3 AUROC'u n = 20 değerinin (**{ref:.3f}**) 0,02 puan içinde kalan en küçük n = **{nmin}** "
          f"(≈ {mn:.1f} dk normal kayıt; ort. döngü {mean_norm[cell]:.1f} s). Kalibrasyon için ek **{N_CAL}** normal döngü "
          f"(≈ {N_CAL*mean_norm[cell]/60:.1f} dk) ayrıca gerekir; toplam normal kayıt = n + {N_CAL} döngü.")
    w("")
    w("### Uyarılar (a)")
    w("")
    w("- **Zamansal sürüklenme:** s02 döngüleri zaman içinde bağımsız değil (operatör yorgunluğu/ısınma, hız sürüklenmesi). Rastgele dağıtım eğitim/kalibrasyon/test döngülerini zamanda "
      "iç içe geçirir; bu **iyimser** olabilir. Gerçek devreye almada eğitim yakın geçmişten, test sonradan gelir. Zaman sıralı bir dağıtım koşulmadı.")
    w("- Tek operatör ve tek oturum (s02); tekrarlar aynı 51 döngüden yeniden örnekleme olduğu için sd **yeni bir istasyona/operatöre genellemeyi** değil, "
      "dağıtım şansını ölçer. R = 50 tekrar bağımsız değildir (aynı 51 döngü).")
    w("- Test kümesi küçük (11 normal döngü): tekrar başına FAR ∈ {0, 1/11, 2/11, …}; FAR'ın nominal 0,10 ile karşılaştırılması gürültülüdür. "
      "Anomali döngüleri her tekrarda aynıdır; AUROC'taki sd yalnız normal tarafın ve parametre öğrenmenin değişkenliğini içerir.")
    w("- n küçükken D0 σ ve D2 referansları (min-ref 3) kararsızdır (σ tabanı 0,05); `pred`'de n döngü sayısıdır.")
    w("")
    # ---------------- (b)
    w("## (b) Durum tahmini: kaç etiketli döngü gerekir?")
    w("")
    w("**Kurulum.** S1 katları (Faz A `folds_state.csv`); her f için eğitim = f dışındaki katlar. Eğitim havuzundan **her oturum için k döngü** "
      "(döngü = `loop_id` grubu; 6 oturum, s01 dâhil → toplam 6k döngü) rastgele seçilir; seçimler tekrar içinde **iç içedir** (aynı permütasyonun ilk k'sı). "
      "B2 = 21 özellik + RandomForest(300, balanced, random_state 20261104, n_jobs = 3), B1 = en-yakın-merkez; test = kat f'nin tamamı (3.2–3.3 bin pencere). "
      "k ∈ {1, 2, 5, 10, 20}, R = 10 tekrar; 'hepsi' = o kattaki tüm eğitim döngüleri (belirlenimci, tek koşu = Faz A S1). "
      "Macro-F1: her tekrar için 5 katın ortalaması; tablo bu 10 tekrarın ortalaması ± sd'sidir ('sd katlar' sütunu ise katlar arası sd).")
    w("")
    w("B1 için `tau` Faz A'daki gibi grup bazlı iç doğrulama (%10, oturum tabakalı) ile seçilir; **k = 1**'de iç doğrulama kümesi boş kalır "
      "(her oturumda tek döngü) → `tau` eğitim pencerelerinin kendisinde seçilir (resubstitution). Bu Sapmalar'da kayıtlıdır.")
    w("")
    ag = agree.groupby("model")["agree_with_fazA_oof"].agg(["min", "mean"])
    w("**Doğrulama:** 'hepsi' koşusunun tahminleri Faz A S1 OOF ile karşılaştırıldı (pencere bazında uyum): " +
      "; ".join(f"{m}: min {r['min']:.6f}, ort. {r['mean']:.6f}" for m, r in ag.iterrows()) + ".")
    w("")
    w("### Ortalama döngü süresi (oturum başına) ve kayıt dakikası")
    w("")
    w("| oturum | loop_id grubu | tam döngü | ort. döngü süresi (s) | ort. tüm-loop süresi (s) |")
    w("|---|---|---|---|---|")
    for r in dur.itertuples():
        w(f"| {r.session} | {r.n_loops} | {r.n_complete_cycles} | {r.mean_cycle_s:.2f} | {r.mean_all_loops_s:.2f} |")
    w("")
    w(f"Ortalama döngü süresi (6 oturum ortalaması) = {dur['mean_cycle_s'].mean():.2f} s; 6 oturumun toplamı = {dur['mean_cycle_s'].sum():.1f} s "
      f"(s01 için tüm-loop ortalaması; s01 rastgele toplama, döngü = tam 4 adımlı döngü değil). Dakika = k × ortalama döngü süresi; "
      "'toplam' = bütün 6 oturumu birlikte kaydetmenin maliyeti (deneyde eğitim bütün oturumlardan havuzlandığı için), 'oturum başına' = tek bir oturum/istasyon "
      "için k döngü. 'fiili toplam' = seçilen loop'ların gerçek kare sayısından hesaplanan toplam kayıt dakikası (6 oturum).")
    w("")
    for model, nm in (("b2", "B2 (el yapımı özellik + RF)"), ("b1", "B1 (en-yakın-merkez)")):
        w(f"### {nm}: macro-F1 / doğruluk")
        w("")
        w("| k (oturum başına) | macro-F1 (ort ± sd tekrarlar) | sd katlar | min–maks tekrar | doğruluk | dk/oturum (nominal) | dk toplam 6 oturum (nominal) | dk toplam (fiili) | eğitim penceresi |")
        w("|---|---|---|---|---|---|---|---|---|")
        t = S[S["model"] == model]
        t = t.assign(_o=t["k"].map(lambda x: 10**6 if x == "all" else int(x))).sort_values("_o")
        for r in t.itertuples():
            nm_ = "—" if r.k == "all" else f"{r.nominal_min_per_session:.1f}"
            nt = "—" if r.k == "all" else f"{r.nominal_min_total_6sessions:.1f}"
            w(f"| {r.k} | {fm(r.mf1_mean, r.mf1_sd_reps, 4)} | {r.mf1_sd_folds:.4f} | {r.mf1_min_rep:.3f}–{r.mf1_max_rep:.3f} | "
              f"{fm(r.acc_mean, r.acc_sd_reps, 4)} | {nm_} | {nt} | {r.rec_min_total_actual:.1f} | {r.n_train_windows_mean:.0f} |")
        w("")
    # 95% olcutu
    w("### Betimleyici \"yeter\" ölçütü (sonradan tanımlı, ön-kayıtlı DEĞİL)")
    w("")
    for model in ("b2", "b1"):
        t = S[S["model"] == model].set_index("k")
        full = t.loc["all", "mf1_mean"]
        ks = [k for k in map(str, K_LIST) if t.loc[k, "mf1_mean"] >= 0.95 * full]
        kmin = min(ks, key=int) if ks else None
        if kmin:
            r = t.loc[kmin]
            w(f"- {model.upper()}: tüm eğitim verisiyle macro-F1 = {full:.4f}; bunun %95'ine (= {0.95*full:.4f}) ulaşan en küçük k = **{kmin}** döngü/oturum "
              f"(macro-F1 {r.mf1_mean:.4f}; ≈ {r.nominal_min_per_session:.1f} dk/oturum nominal, 6 oturum toplam ≈ {r.nominal_min_total_6sessions:.1f} dk).")
        else:
            w(f"- {model.upper()}: tam veriyle macro-F1 = {full:.4f}; k ≤ 20'nin hiçbiri %95'ine ulaşmıyor.")
    w("")
    w("### B2 oturum başına macro-F1 (5 kat × tekrar ortalaması; test pencerelerinde oturum bazında)")
    w("")
    sc = [c for c in S.columns if c.startswith("mf1_") and c not in ("mf1_mean", "mf1_sd_reps", "mf1_sd_folds", "mf1_min_rep", "mf1_max_rep")]
    w("| k | " + " | ".join(c[4:] for c in sc) + " |")
    w("|---|" + "---|" * len(sc))
    t = S[S["model"] == "b2"]
    t = t.assign(_o=t["k"].map(lambda x: 10**6 if x == "all" else int(x))).sort_values("_o")
    for r in t.itertuples():
        w(f"| {r.k} | " + " | ".join(f"{getattr(r, c):.3f}" for c in sc) + " |")
    w("")
    w("### Uyarılar (b)")
    w("")
    w("- Katlar `StratifiedGroupKFold` ile (oturum, loop_id) gruplarına göre ayrıldığından seçilen döngüler **ardışık zaman** değil rastgeledir (aynı sürüklenme uyarısı); "
      "test döngüleri eğitim döngülerinin zamansal komşusu olabilir.")
    w("- Eğitim **6 oturumdan havuzlanmış**tır (hand-box durum kestirimi senaryodan bağımsız varsayımı, Faz A ile aynı). Yeni bir istasyon için "
      "'6k döngü' pratikte tek istasyondan değil, mevcut 6 oturumluk çeşitlilikten gelir; tek oturumdan k döngü ile eğitilen model bu deneyde ölçülmedi. "
      "Dolayısıyla 'toplam 6 oturum' dakikası bir üst sınır değil, deneyin gerçek maliyetidir; tek istasyon maliyeti için 'oturum başına' sütunu kullanılır ama genelleme iddiası taşımaz.")
    w("- CNN-GRU bu analize dâhil değildir (kapsam dışı; brifte yalnız B2 ve B1). Etiketler ToF'tan bedava; maliyet = kayıt dakikası.")
    w("- 'döngü' = `loop_id` grubu; s01 (rastgele toplama) ve oturum başı/sonu parçaları tam 4 adımlı döngü olmayabilir; fiili dakika bu yüzden nominal değerden sapabilir.")
    w("")
    w("## Makale cümlesi için özet (ikincil)")
    w("")
    sgs = SG[SG["group"] == "tum anomali"].set_index(["cell", "n"])
    sfs = SF.set_index(["cell", "n"])
    for cell in ("B2 x pred", "ToF x tof"):
        a = " ; ".join(f"n={n}: AUROC {sgs.loc[(cell,n),'auroc_mean']:.3f}, FAR@0,10 {sfs.loc[(cell,n),'far_a10_mean']:.3f}" for n in N_LIST)
        w(f"- {cell}: {a}")
    for model in ("b2", "b1"):
        t = S[S["model"] == model].set_index("k")
        a = " ; ".join(f"k={k}: {t.loc[k,'mf1_mean']:.3f}" for k in list(map(str, K_LIST)) + ["all"])
        w(f"- {model.upper()} macro-F1: {a}")
    w("")
    (OUT2 / "04_commissioning.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    P("rapor yazildi:", OUT2 / "04_commissioning.md")
    P("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["a", "b", "report", "all"])
    a = ap.parse_args()
    if a.part in ("a", "all"):
        part_a()
    if a.part in ("b", "all"):
        part_b()
    if a.part in ("report", "all"):
        report()

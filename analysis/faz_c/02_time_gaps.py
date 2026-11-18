"""Faz C / C.2 - kayit zaman bosluklarinin nedeni (betimsel). Plan: faz_c/DESIGN_C.md "C.2".

Salt okunur: veri klasoru, paper/main.tex, reports/* yalnizca OKUNUR. Yalniz reports/faz_c/ ve faz_c/ altina yazar.
Her sayi stdout'a yazilir ve CSV'de bulunur (rec()). md, ayni degiskenlerden uretilir ve stdout'a da basilir.

Onceden yazilmis okuma kurali (isleyisi, calistirmadan once sabit):
  n_points: 'dusuk' = komsu kare yuzdelik-sira (oturum ici) medyani < 0,5 ve MWU(tek yonlu) p < 0,01; 'yuksek' simetrik; yoksa 'benzer'.
  kutu yok: komsu karelerde 'kutu yok' payi, oturumdaki tum karelere gore Fisher (tek yonlu) p<0,01 ile yuksek/dusuk; yoksa 'benzer'.
            Bosluk ARALIGINDAKI ham ToF (30 Hz) aktif payi, ayni uzunlukta rastgele ToF pencerelerine gore (permutasyon, iki yonlu p<0,01).
            Iki kaynak ayni yonu gostermiyorsa 'karisik'.
  H-bos desteklenir: n_points dusuk VE kutu-yok yuksek (iki kaynak da). H-yuk: n_points yuksek VE kutu-yok dusuk (iki kaynak da).
  Aksi halde 'belirsiz'. Karar havuzlanmis komsular (onceki+sonraki) uzerinden; onceki/sonraki ayri ayri da raporlanir.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probe"))
sys.path.insert(0, str(ROOT / "faz_a"))
import common  # noqa: E402
import fa_common as fc  # noqa: E402

OUT = ROOT / "reports" / "faz_c"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.RandomState(20261105)
GAP_DT = 0.15
P = 0.1

ROWS: list[dict] = []
MD: list[str] = []


def rec(section, item, stat, value, n="", note=""):
    ROWS.append(dict(section=section, item=item, stat=stat, value=value, n=n, note=note))
    v = f"{value:.6g}" if isinstance(value, (float, np.floating)) else value
    print(f"[{section}] {item} | {stat} = {v} (n={n}) {note}")


def H(s=""):
    MD.append(s)


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ======================================================================================= veri
SESS = common.sessions()
ft = fc.frame_table()
W = fc.windows()
lag_rows = []
raw = {}
tof = {}
vid = {}
for s in SESS:
    r = pd.read_csv(common.DATA_PATH / s / "mmwave_data.csv")
    ts = pd.to_datetime(r["timestamp"], format="mixed")
    r["t"] = (ts - common.t0(s)).dt.total_seconds()
    g = r.groupby("frame_id", sort=True)
    rf = pd.DataFrame({"t": g["t"].first(), "n_all": g.size(), "n_mov": g["velocity"].apply(lambda v: int((v != 0).sum()))}).reset_index()
    raw[s] = rf
    d = pd.read_csv(common.DATA_PATH / s / "distance_sensors.csv")
    d["t"] = (pd.to_datetime(d["timestamp"], format="mixed") - common.t0(s)).dt.total_seconds()
    d["label"] = common.label_from_active(d[common.ACTIVE_COLS].to_numpy())
    d["fid"] = d["frame_id"]
    d = d.sort_values("t", kind="stable").reset_index(drop=True)
    tof[s] = d
    v = pd.read_csv(common.DATA_PATH / s / "video_frames.csv")
    v["t"] = (pd.to_datetime(v["timestamp"], format="mixed") - common.t0(s)).dt.total_seconds()
    vid[s] = v.sort_values("t").reset_index(drop=True)
    # ToF-sayac gecikmesi: ilk ToF satirinin zamani - radar zaman damgasi (kare 0 haric)
    fm = d.groupby("fid")["t"].min()
    j = rf.set_index("frame_id")["t"].to_frame("tr").join(fm.rename("tt"), how="inner")
    j = j[j.index > 0]
    lag_rows.append(j["tt"] - j["tr"])
LAG = float(np.median(pd.concat(lag_rows)))
rec("veri", "tof_sayac_gecikmesi", "medyan (ilk ToF satiri - radar zaman damgasi) s", LAG, n=sum(len(x) for x in lag_rows))

# kare basina menzil (merged noktalarindan)
rng_mean = []
for s in SESS:
    p = common.load_points(s)
    rng_mean.append(p.groupby("frame_id")["range"].mean().rename("range_mean").reset_index().assign(session=s))
ft = ft.merge(pd.concat(rng_mean), on=["session", "frame_id"], how="left", validate="1:1")
assert ft["range_mean"].notna().all()
ft = ft.merge(pd.concat([raw[s].assign(session=s)[["session", "frame_id", "n_all"]] for s in SESS]), on=["session", "frame_id"], how="left")

rec("veri", "kare_sayisi_merged", "toplam", len(ft), n=len(ft))
rec("veri", "kare_sayisi_ham_mmwave_data", "toplam", sum(len(raw[s]) for s in SESS), n="")
for s in SESS:
    rec("veri", s, "ham kare / merged kare", f"{len(raw[s])} / {int((ft.session == s).sum())}")

# ======================================================================================= aralik tablosu
iv = []
for s in SESS:
    g = ft[ft.session == s]
    t = g["t"].to_numpy()
    dt = np.diff(t)
    ridx = g["row"].to_numpy()
    for j, d_ in enumerate(dt):
        iv.append((s, ridx[j], ridx[j + 1], d_))
IV = pd.DataFrame(iv, columns=["session", "pre_row", "post_row", "dt"])
IV["gap"] = IV["dt"] > GAP_DT
IV["u"] = IV["dt"] / P
IV["k_missing"] = np.where(IV["gap"], np.round(IV["u"]).astype(int) - 1, 0)
nG = int(IV["gap"].sum())
rec("genel", "aralik_sayisi", "toplam", len(IV))
rec("genel", "bosluk_sayisi (dt>0,15 s)", "toplam", nG, n=len(IV), note="B.0(f): 696")
rec("genel", "en_buyuk_dt", "s", float(IV.dt.max()))
assert nG == 696, nG
MISS = int(IV["k_missing"].sum())
N_LOG = len(ft)
MISS_SHARE = MISS / (N_LOG + MISS)
rec("genel", "kayip_kare_yuvasi_tahmini", "toplam sum(round(dt/0,1)-1)", MISS, note="frame_id sayaci ardisik oldugundan yalniz zaman damgasindan kestirilir")
rec("genel", "kayip_pay", "kayip/(kayitli+kayip)", MISS_SHARE, n=N_LOG + MISS)

# ======================================================================================= TEST 1 tam kat
reg = IV[~IV["gap"]]
dev_reg = np.abs(reg["dt"] - P)
p95 = float(np.percentile(dev_reg, 95))
TOL = 2 * p95
TOL_U = TOL / P
rec("T1", "duzenli_aralik", "n (dt<=0,15)", len(reg))
rec("T1", "duzenli_aralik |dt-0,1|", "medyan s", float(np.median(dev_reg)))
rec("T1", "duzenli_aralik |dt-0,1|", "p95 s", p95)
rec("T1", "tolerans", "2 x p95 (s)", TOL)
rec("T1", "tolerans", "birim (0,1 s'nin kati)", TOL_U)
G = IV[IV["gap"]].copy()
G["resid_u"] = G["u"] - np.round(G["u"])
G["resid_abs_s"] = np.abs(G["resid_u"]) * P
frac_in = float((G["resid_abs_s"] <= TOL).mean())
rec("T1", "bosluklar", "tolerans icinde pay", frac_in, n=nG, note="esik >= 0,90")
T1_PASS = frac_in >= 0.90
rec("T1", "karar", "tam kat testi (>=0,90)", "GECTI" if T1_PASS else "GECMEDI")
rec("T1", "bosluklar |dt/0,1 - round|", "medyan (birim)", float(np.median(np.abs(G.resid_u))))
rec("T1", "bosluklar |dt/0,1 - round|", "p95 (birim)", float(np.percentile(np.abs(G.resid_u), 95)))
rec("T1", "bosluklar", "ort. resid s", float(G.resid_u.mean() * P), note="isaretli yanlilik")
# rastgele zamanlama: ayni dt'lere U(0,0,1) faz eklenir
nulls = []
for _ in range(2000):
    u2 = (G["dt"].to_numpy() + rng.uniform(0, P, nG)) / P
    nulls.append((np.abs(u2 - np.round(u2)) * P <= TOL).mean())
nulls = np.array(nulls)
rec("T1", "rastgele-faz basi", "tolerans icinde pay (ort)", float(nulls.mean()))
rec("T1", "rastgele-faz basi", "tolerans icinde pay (maks, 2000 tekrar)", float(nulls.max()))
rec("T1", "analitik tekdüze", "2*tol_birim", float(min(1.0, 2 * TOL_U)))
# k siniflari
kb = pd.cut(G["k_missing"], [0, 1, 2, 4, 9, 10 ** 6], labels=["k=1", "k=2", "k=3-4", "k=5-9", "k>=10"])
T1K = []
for lab, g in G.groupby(kb, observed=True):
    T1K.append(dict(sinif=str(lab), n=len(g), tol_ici=float((g.resid_abs_s <= TOL).mean()), medyan_resid_ms=float(np.median(g.resid_u) * P * 1000),
                    p95_abs_ms=float(np.percentile(g.resid_abs_s, 95) * 1000)))
    for kk, vv in T1K[-1].items():
        if kk != "sinif":
            rec("T1k", str(lab), kk, vv)
T1K = pd.DataFrame(T1K)
# resid dagilimi (tam), 0,1 birimlik kutular [-0,5, 0,5)
edges = np.round(np.arange(-0.5, 0.5001, 0.1), 2)
cnt, _ = np.histogram(G["resid_u"], bins=edges)
HIST = pd.DataFrame({"kutu_alt": edges[:-1], "kutu_ust": edges[1:], "n": cnt, "pay": cnt / nG})
for r_ in HIST.itertuples():
    rec("T1hist", f"[{r_.kutu_alt:+.1f},{r_.kutu_ust:+.1f})", "bosluk sayisi", int(r_.n), n=nG, note=f"pay {r_.pay:.4f}")
# ince dagilim: tolerans civarinda
for q in (0.5, 0.75, 0.9, 0.95, 0.99):
    rec("T1q", f"|resid| bosluk", f"q{int(q*100)} s", float(np.quantile(G.resid_abs_s, q)))
# yedek: duzenli aralik medyani ile
per_fit = float(np.median(reg["dt"]))
Gf = G["dt"] / per_fit
rec("T1", "yedek (bilgi): duzenli medyan periyot", "s", per_fit)
rec("T1", "yedek: medyan periyotla tolerans icinde pay", "pay", float((np.abs(Gf - np.round(Gf)) * per_fit <= TOL).mean()), n=nG)

# ======================================================================================= TEST 2 komsu kareler
ft["pct_n"] = ft.groupby("session")["n_points"].rank(method="average") / ft.groupby("session")["n_points"].transform("size")
ft["pct_nall"] = ft.groupby("session")["n_all"].rank(method="average") / ft.groupby("session")["n_all"].transform("size")
ft["pct_rng"] = ft.groupby("session")["range_mean"].rank(method="average") / ft.groupby("session")["range_mean"].transform("size")
ft["pct_v"] = ft.groupby("session")["mean_abs_v"].rank(method="average") / ft.groupby("session")["mean_abs_v"].transform("size")
ft["none"] = (ft["y_raw"] == 0).astype(int)
pre_rows = G["pre_row"].to_numpy()
post_rows = G["post_row"].to_numpy()
nb_rows = np.unique(np.r_[pre_rows, post_rows])
SETS = {"onceki": np.unique(pre_rows), "sonraki": np.unique(post_rows), "havuz": nb_rows}
for k_, v_ in SETS.items():
    rec("T2", f"komsu kume {k_}", "benzersiz kare", len(v_))


def mwu(a, b, alt):
    return max(float(stats.mannwhitneyu(a, b, alternative=alt).pvalue), 1e-300)   # 1e-300 = sayisal taban (alttan tasma)


VARS = [("n_points", "n_points (merged, v!=0)", "pct_n"), ("n_all", "n_all (ham, tum noktalar)", "pct_nall"),
        ("range_mean", "kare ort. menzil (m)", "pct_rng"), ("mean_abs_v", "ort. |hiz| (m/s)", "pct_v")]
T2 = []
for setn, rows in SETS.items():
    sub = ft.iloc[rows]
    for col, lab, pc in VARS:
        base_med = float(ft[col].median())
        nb_med = float(sub[col].median())
        pct_med = float(sub[pc].median())
        p_two = mwu(sub[pc].to_numpy(), ft[pc].to_numpy(), "two-sided")
        p_less = mwu(sub[pc].to_numpy(), ft[pc].to_numpy(), "less")
        p_gr = mwu(sub[pc].to_numpy(), ft[pc].to_numpy(), "greater")
        # oturum ici medyanlar (bilgi)
        T2.append(dict(komsu=setn, degisken=lab, n_komsu=len(sub), komsu_medyan=nb_med, tum_kare_medyan=base_med, komsu_pctl_medyan=pct_med,
                       p_iki_yonlu=p_two, p_dusuk=p_less, p_yuksek=p_gr))
        rec("T2", f"{setn} | {lab}", "komsu medyan / tum-kare medyan / yuzdelik medyan / MWU p(iki yonlu, dusuk, yuksek)",
            f"{nb_med:.4g} / {base_med:.4g} / {pct_med:.4f} / {p_two:.3g}, {p_less:.3g}, {p_gr:.3g}", n=len(sub))
T2 = pd.DataFrame(T2)
# oturum bazinda n_points medyanlari (bilgi)
for s in SESS:
    g = ft[ft.session == s]
    nbs = g[g["row"].isin(nb_rows)]
    rec("T2s", s, "n_points medyan: komsu / tum", f"{nbs.n_points.median():.3g} / {g.n_points.median():.3g}", n=len(nbs))
# kutu yok payi
none_all = float(ft["none"].mean())
rec("T2", "tum kareler", "'kutu yok' payi", none_all, n=len(ft))
NONE_RES = {}
for setn, rows in SETS.items():
    sub = ft.iloc[rows]
    a = int(sub["none"].sum()); b = len(sub) - a
    c = int(ft["none"].sum()); d_ = len(ft) - c
    pg = float(stats.fisher_exact([[a, b], [c, d_]], alternative="greater")[1])
    pl = float(stats.fisher_exact([[a, b], [c, d_]], alternative="less")[1])
    NONE_RES[setn] = dict(share=a / len(sub), p_yuksek=pg, p_dusuk=pl, n=len(sub))
    rec("T2", f"komsu {setn}", "'kutu yok' payi", a / len(sub), n=len(sub), note=f"Fisher tek yonlu p(yuksek)={pg:.3g}, p(dusuk)={pl:.3g}; tum kareler {none_all:.4f}")
# oturum bazinda kutu yok
for s in SESS:
    g = ft[ft.session == s]
    nbs = g[g["row"].isin(nb_rows)]
    rec("T2s", s, "'kutu yok' payi: komsu / tum", f"{nbs.none.mean():.3f} / {g.none.mean():.3f}", n=len(nbs))
# onceki kare nokta sayisi egilimi (5 kare, indeks tabanli egim)
slopes_all, slopes_pre = [], []
pre_set = set(pre_rows.tolist())
for s in SESS:
    g = ft[ft.session == s]
    n_ = g["n_points"].to_numpy(float)
    rws = g["row"].to_numpy()
    if len(n_) < 5:
        continue
    X = np.arange(-2, 3)
    sl = np.convolve(n_, X[::-1], mode="valid") / 10.0     # i = 4..N-1 icin son 5 karenin egimi
    ends = rws[4:]
    slopes_all.append(sl)
    mask = np.isin(ends, pre_rows)
    slopes_pre.append(sl[mask])
slopes_all = np.concatenate(slopes_all)
slopes_pre = np.concatenate(slopes_pre)
sl_p_less = mwu(slopes_pre, slopes_all, "less")
sl_p_two = mwu(slopes_pre, slopes_all, "two-sided")
rec("T2", "onceki 5 kare egimi (nokta/kare)", "medyan: bosluk oncesi / tum", f"{np.median(slopes_pre):.4f} / {np.median(slopes_all):.4f}", n=len(slopes_pre))
rec("T2", "onceki 5 kare egimi", "negatif pay: bosluk oncesi / tum", f"{(slopes_pre < 0).mean():.4f} / {(slopes_all < 0).mean():.4f}", n=len(slopes_pre))
rec("T2", "onceki 5 kare egimi", "MWU p (dusuk, iki yonlu)", f"{sl_p_less:.3g}, {sl_p_two:.3g}", n=len(slopes_pre))
rec("T2", "onceki 5 kare egimi", "ort.: bosluk oncesi / tum", f"{slopes_pre.mean():.4f} / {slopes_all.mean():.4f}")
# n_points dagilimi, tek-nokta karelerin payi
for kk in (1, 2, 3):
    rec("T2d", f"n_points=={kk}", "pay tum / komsu havuz", f"{(ft.n_points == kk).mean():.4f} / {(ft.iloc[nb_rows].n_points == kk).mean():.4f}")

# ======================================================================================= TEST 3 oturum bazinda + ham ToF
rows3 = []
for s in SESS:
    g = ft[ft.session == s]
    iv_s = IV[IV.session == s]
    gs = iv_s[iv_s.gap]
    wall = float(g.t.iloc[-1] + P - g.t.iloc[0])
    miss = int(gs.k_missing.sum())
    rows3.append(dict(session=s, kare=len(g), aralik=len(iv_s), bosluk=len(gs), bosluk_orani_kare_basina=len(gs) / len(g), kayip_yuva=miss,
                      kayip_pay=miss / (len(g) + miss), en_uzun_bosluk_s=float(gs.dt.max()), medyan_bosluk_s=float(gs.dt.median()),
                      bosluk_ge_1s=int((gs.dt >= 1.0).sum()), wall_s=wall))
    for kk in ("kare", "bosluk", "bosluk_orani_kare_basina", "kayip_yuva", "kayip_pay", "en_uzun_bosluk_s", "medyan_bosluk_s", "bosluk_ge_1s"):
        rec("T3", s, kk, rows3[-1][kk])
T3 = pd.DataFrame(rows3)

# ham ToF bosluk araliklari ve rastgele pencere nulli
R = 1000
gap_info = []
null_act = np.zeros((R,)); null_act_w = np.zeros((R,)); null_chg = np.zeros((R,))
obs_cnt = []  # (n_rows, n_active, n_changes, session)
per_sess_arr = {}
for s in SESS:
    d = tof[s]
    lab = d["label"].to_numpy()
    act = (lab > 0).astype(np.int64)
    chg = np.r_[0, (lab[1:] != lab[:-1]).astype(np.int64)]
    per_sess_arr[s] = (d["t"].to_numpy(), d["fid"].to_numpy(), lab, np.r_[0, np.cumsum(act)], np.r_[0, np.cumsum(chg)])
SLOT_LAB = []   # kayip yuvalarin ToF etiketi
for g_ in G.itertuples():
    s = g_.session
    t_arr, fid_arr, lab, cA, cC = per_sess_arr[s]
    pre = ft.iloc[g_.pre_row]
    post = ft.iloc[g_.post_row]
    i0 = int(np.searchsorted(t_arr, pre.t, side="left"))          # birincil: (ts_onceki, ts_sonraki) tam aralik, ayni saat tabani
    i1 = int(np.searchsorted(t_arr, post.t, side="left"))
    n_r = max(i1 - i0, 0)
    n_act = int(cA[i1] - cA[i0]) if n_r else 0
    n_ch = int(cC[i1] - cC[min(i0 + 1, i1)]) if n_r > 1 else 0
    sens = {}
    for sh in (0.1, 0.2):                                          # duyarlilik: radar damgasi fiziksel yakalamadan sonra basilir -> aralik geriye kaydirilir
        j0 = int(np.searchsorted(t_arr, pre.t - sh, side="left")); j1 = int(np.searchsorted(t_arr, post.t - sh, side="left"))
        sens[sh] = (int(cA[j1] - cA[j0]), max(j1 - j0, 0))
    tt = t_arr[i0:i1]
    vt = vid[s]["t"].to_numpy()
    v0 = int(np.searchsorted(vt, pre.t)); v1 = int(np.searchsorted(vt, post.t))
    vtt = vt[v0:v1]
    dur = float(post.t - pre.t)
    for j in range(1, int(g_.k_missing) + 1):                     # kayip yuva j: ts_onceki + 0,1 j
        ii = min(int(np.searchsorted(t_arr, pre.t + P * j, side="left")), len(lab) - 1)
        SLOT_LAB.append((s, int(lab[ii])))
    same_pre = int((fid_arr[i0:i1] == pre.frame_id).sum()) if n_r else 0
    same_post = int((fid_arr[i0:i1] == pre.frame_id + 1).sum()) if n_r else 0
    gap_info.append(dict(session=s, pre_frame_id=int(pre.frame_id), t_pre=float(pre.t), dt=dur, k=int(g_.k_missing), tof_rows=n_r, tof_active=n_act,
                         tof_changes=n_ch, tof_max_dt=float(np.max(np.diff(tt))) if len(tt) > 1 else np.nan, vid_rows=len(vtt),
                         vid_max_dt=float(np.max(np.diff(vtt))) if len(vtt) > 1 else np.nan, tof_fid_eq_pre=same_pre, tof_fid_eq_pre_plus1=same_post,
                         act_shift01=sens[0.1][0], rows_shift01=sens[0.1][1], act_shift02=sens[0.2][0], rows_shift02=sens[0.2][1],
                         n_pre=int(pre.n_points), n_post=int(post.n_points), lab_pre=int(pre.y_raw), lab_post=int(post.y_raw),
                         range_pre=float(pre.range_mean), range_post=float(post.range_mean)))
GI = pd.DataFrame(gap_info)
GI["act_frac"] = np.where(GI.tof_rows > 0, GI.tof_active / GI.tof_rows.clip(lower=1), np.nan)
GI["chg_rate"] = np.where(GI.tof_rows > 1, GI.tof_changes / (GI.tof_rows - 1).clip(lower=1), np.nan)
ok = GI["tof_rows"] > 0
rec("T3", "bosluk ToF araligi", "araliginda ToF satiri olan bosluk", int(ok.sum()), n=len(GI))
obs_act_pool = float(GI.tof_active.sum() / GI.tof_rows.sum())
obs_act_mean = float(GI.loc[ok, "act_frac"].mean())
obs_chg_pool = float(GI.tof_changes.sum() / max((GI.tof_rows - 1).clip(lower=0).sum(), 1))
rec("T3", "bosluk ToF araligi", "aktif pay (havuz satir)", obs_act_pool, n=int(GI.tof_rows.sum()))
rec("T3", "bosluk ToF araligi", "aktif pay (bosluk ortalamasi)", obs_act_mean, n=int(ok.sum()))
rec("T3", "bosluk ToF araligi", "etiket degisim orani (satir basina)", obs_chg_pool)
rec("T3", "bosluk ToF araligi", "aktif pay = 0 olan bosluk payi", float((GI.loc[ok, "act_frac"] == 0).mean()), n=int(ok.sum()))
rec("T3", "bosluk ToF araligi", "aktif pay = 1 olan bosluk payi", float((GI.loc[ok, "act_frac"] == 1).mean()), n=int(ok.sum()))
# rastgele ayni uzunlukta (satir sayisi) ToF pencereleri
for r_i in range(R):
    tot_rows = 0; tot_act = 0; tot_chg = 0; fr = []
    for s in SESS:
        sub = GI[(GI.session == s) & ok]
        if len(sub) == 0:
            continue
        t_arr, fid_arr, lab, cA, cC = per_sess_arr[s]
        valid0 = int(np.searchsorted(fid_arr, 1))      # radar basladiktan sonra (fid>=1)
        nrow = sub["tof_rows"].to_numpy()
        hi = len(lab) - nrow
        st = rng.randint(valid0, np.maximum(hi, valid0 + 1))
        en = st + nrow
        a = cA[en] - cA[st]
        c = cC[en] - cC[np.minimum(st + 1, en)]
        tot_rows += nrow.sum(); tot_act += a.sum(); tot_chg += c.sum()
        fr.extend((a / nrow).tolist())
    null_act[r_i] = tot_act / tot_rows
    null_act_w[r_i] = np.mean(fr)
    null_chg[r_i] = tot_chg / max(tot_rows - len(GI[ok]), 1)
fid_sorted_ok = all(np.all(np.diff(per_sess_arr[s][1]) >= 0) for s in SESS)
rec("T3", "ToF fid siralamasi", "tum oturumlarda monoton", bool(fid_sorted_ok))
p_lo = float((null_act <= obs_act_pool).mean()); p_hi = float((null_act >= obs_act_pool).mean())
perm_p_two = min(1.0, 2 * min(p_lo, p_hi) + 1 / R)
rec("T3", "bosluk ToF aktif pay vs rastgele pencere", "null ort / null q2,5-q97,5", f"{null_act.mean():.4f} / [{np.quantile(null_act, .025):.4f}, {np.quantile(null_act, .975):.4f}]", n=R)
rec("T3", "bosluk ToF aktif pay vs rastgele pencere", "gozlenen havuz", obs_act_pool)
rec("T3", "bosluk ToF aktif pay vs rastgele pencere", "permutasyon p (iki yonlu)", perm_p_two, n=R)
p_lo_w = float((null_act_w <= obs_act_mean).mean()); p_hi_w = float((null_act_w >= obs_act_mean).mean())
perm_p_two_w = min(1.0, 2 * min(p_lo_w, p_hi_w) + 1 / R)
rec("T3", "bosluk ToF aktif pay (bosluk ort.) vs rastgele", "null ort", float(null_act_w.mean()), n=R)
rec("T3", "bosluk ToF aktif pay (bosluk ort.) vs rastgele", "permutasyon p (iki yonlu)", perm_p_two_w, n=R)
chg_null_mean = float(null_chg.mean())
p_lo_c = float((null_chg <= obs_chg_pool).mean()); p_hi_c = float((null_chg >= obs_chg_pool).mean())
rec("T3", "bosluk ToF etiket degisimi vs rastgele", "gozlenen / null ort", f"{obs_chg_pool:.4f} / {chg_null_mean:.4f}", n=R, note=f"p(dusuk)={p_lo_c:.3g}, p(yuksek)={p_hi_c:.3g}")
# oturum baslangic etiket dagilimi
tof_all_active = float(np.mean(np.concatenate([(per_sess_arr[s][2][np.searchsorted(per_sess_arr[s][1], 1):] > 0) for s in SESS])))
rec("T3", "tum ToF (radar basladiktan sonra)", "aktif pay", tof_all_active)
# bosluk suresine gore
for lab_, lo, hi in (("k=1", 1, 1), ("k=2-4", 2, 4), ("k>=5", 5, 10 ** 6)):
    sub = GI[(GI.k >= lo) & (GI.k <= hi) & ok]
    rec("T3k", lab_, "bosluk sayisi / aktif pay (havuz) / medyan ToF satiri", f"{len(sub)} / {sub.tof_active.sum() / max(sub.tof_rows.sum(), 1):.4f} / {sub.tof_rows.median():.0f}")
# kayip yuva basina tehlike
SL = pd.DataFrame(SLOT_LAB, columns=["session", "lab"])
nm = len(SL); nm_none = int((SL.lab == 0).sum())
LOGLAB = []
for s_ in SESS:
    t_arr_, _, lab_, _, _ = per_sess_arr[s_]
    tf_ = ft[ft.session == s_]["t"].to_numpy()
    LOGLAB.append(lab_[np.minimum(np.searchsorted(t_arr_, tf_, side="left"), len(lab_) - 1)])
LOGLAB = np.concatenate(LOGLAB)
nl_none = int((LOGLAB == 0).sum()); nl = len(LOGLAB)
haz_none = nm_none / (nm_none + nl_none); haz_box = (nm - nm_none) / ((nm - nm_none) + (nl - nl_none))
rec("S1", "kayip yuva etiketi (ToF)", "'kutu yok' payi", nm_none / nm, n=nm, note=f"kayitli karelerde {nl_none / nl:.4f}")
rec("S1", "yuva basina kayip tehlikesi", "P(kayip | kutu yok)", haz_none, n=nm_none + nl_none)
rec("S1", "yuva basina kayip tehlikesi", "P(kayip | kutu aktif)", haz_box, n=(nm - nm_none) + (nl - nl_none))
rec("S1", "yuva basina kayip tehlikesi", "oran yok/aktif", haz_none / haz_box)
for b in range(1, 6):
    mb = int((SL.lab == b).sum()); lb = int((LOGLAB == b).sum())
    rec("S1", f"kutu {b}", "P(kayip | kutu b)", mb / (mb + lb) if mb + lb else np.nan, n=mb + lb)
# ToF / video canliligi
alive = GI[GI.dt >= 1.0]
rec("S3", "bosluk >= 1 s", "sayi", len(alive))
tof_rate = alive.tof_rows / alive.dt
vid_rate = alive.vid_rows / alive.dt
rec("S3", "bosluk >= 1 s: ToF oran (satir/s)", "min / medyan", f"{tof_rate.min():.1f} / {tof_rate.median():.1f}", n=len(alive))
rec("S3", "bosluk >= 1 s: video oran (kare/s)", "min / medyan", f"{vid_rate.min():.1f} / {vid_rate.median():.1f}", n=len(alive))
rec("S3", "tum bosluklar: ToF ic-aralik max dt", "s (max)", float(GI.tof_max_dt.max()))
rec("S3", "tum bosluklar: video ic-aralik max dt", "s (max)", float(GI.vid_max_dt.max()))
rec("S3", "bosluk ToF satirlarinda sayac kimligi", "onceki kare kimligine esit pay", float(GI.tof_fid_eq_pre.sum() / GI.tof_rows.sum()), n=int(GI.tof_rows.sum()))
rec("S3", "bosluk ToF satirlarinda sayac kimligi", "onceki kare kimligi+1 (= sonraki kare) esit pay", float(GI.tof_fid_eq_pre_plus1.sum() / GI.tof_rows.sum()), n=int(GI.tof_rows.sum()))
for sh, ca, cr in ((0.1, "act_shift01", "rows_shift01"), (0.2, "act_shift02", "rows_shift02")):
    rec("T3", f"bosluk ToF araligi geri kaydirma {sh} s", "aktif pay (havuz)", float(GI[ca].sum() / GI[cr].sum()), n=int(GI[cr].sum()), note="duyarlilik")
for s in SESS:
    dd = tof[s]["t"].diff().dropna(); vv = vid[s]["t"].diff().dropna()
    rec("S3", s, "ToF max dt / video max dt (tum oturum, s)", f"{dd.max():.3f} / {vv.max():.3f}")
# en uzun 10 bosluk
LONG = GI.sort_values("dt", ascending=False).head(10)
for r_ in LONG.itertuples():
    rec("T3long", f"{r_.session} t={r_.t_pre:.1f}s", "dt s / k / ToF aktif pay / n_once,n_sonra / etiket once,sonra / menzil once,sonra",
        f"{r_.dt:.2f} / {r_.k} / {r_.act_frac:.2f} / {r_.n_pre},{r_.n_post} / {r_.lab_pre},{r_.lab_post} / {r_.range_pre:.2f},{r_.range_post:.2f}")

# S2: nokta sayisi dagiliminin sifira ekstrapolasyonu (sezgisel)
cn = ft["n_points"].value_counts().sort_index()
p1 = cn.get(1, 0) / len(ft); p2 = cn.get(2, 0) / len(ft)
odds_obs = MISS / N_LOG
p0rel_pois = p1 ** 2 / (2 * p2); p0rel_geo = p1 ** 2 / p2
rec("S2", "n_points payi", "p1 / p2 (kayitli kareler)", f"{p1:.4f} / {p2:.4f}", n=len(ft))
rec("S2", "kayip/kayitli oran (gozlenen)", "M/N", odds_obs)
rec("S2", "sifir-nokta kare / kayitli (Poisson ekstrapolasyonu p1^2/(2 p2))", "oran", p0rel_pois, note="SEZGISEL; model varsayimi")
rec("S2", "sifir-nokta kare / kayitli (geometrik ekstrapolasyon p1^2/p2)", "oran", p0rel_geo, note="SEZGISEL; model varsayimi")
# S4: statik (v==0) noktalar
allraw = pd.concat([pd.read_csv(common.DATA_PATH / s / "mmwave_data.csv", usecols=["frame_id", "velocity"]).assign(session=s) for s in SESS])
rec("S4", "ham noktalar", "v==0 payi", float((allraw.velocity == 0).mean()), n=len(allraw))
fr_static = allraw.groupby(["session", "frame_id"])["velocity"].agg(n="size", ns=lambda v: int((v == 0).sum())).reset_index()
rec("S4", "ham kareler", "yalniz-statik kare sayisi (n_all>0, hepsi v==0)", int((fr_static.n == fr_static.ns).sum()), n=len(fr_static))
rec("S4", "ham kareler", "en az 1 statik nokta iceren kare", int((fr_static.ns > 0).sum()), n=len(fr_static))
for nn in (1, 2, 3):
    sub = fr_static[fr_static.n == nn]
    rec("S4", f"n_all=={nn} kareler", "yalniz-statik pay / tek-nokta statik beklenen (nokta payi^n)", f"{(sub.n == sub.ns).mean():.4f} / {(float((allraw.velocity == 0).mean()) ** nn):.4f}", n=len(sub))

# ======================================================================================= TEST 3b pencere etkisi
rows_w = fc.win_rows(W)
tt_ = ft["t"].to_numpy()
gap_row = np.zeros(len(ft), dtype=bool)         # satir i ile i+1 arasi bosluk
gap_row[G["pre_row"].to_numpy()] = True
wg = np.zeros(len(W), dtype=bool)
for k in range(4):
    wg |= gap_row[rows_w[:, k]] & (rows_w[:, k + 1] == rows_w[:, k] + 1)
span = tt_[rows_w[:, 4]] - tt_[rows_w[:, 0]]
n_win_gap = int(wg.sum())
rec("TW", "pencere", "sayi", len(W))
rec("TW", "bosluklu pencere (en az bir dt>0,15 s)", "sayi", n_win_gap, n=len(W), note="B.0(f): 2114")
rec("TW", "bosluklu pencere", "pay", n_win_gap / len(W))
rec("TW", "bosluklu pencere", "sinif 'none' (y_true==0) olan sayi", int((wg & (W["y_true"].to_numpy() == 0)).sum()), note="B.0(f): 2073")
rec("TW", "pencere suresi (son-ilk kare, s)", "medyan / p95 / max", f"{np.median(span):.3f} / {np.percentile(span, 95):.3f} / {span.max():.2f}", n=len(W))
rec("TW", "bosluklu pencere suresi (son-ilk kare, s)", "medyan / p95 / max", f"{np.median(span[wg]):.3f} / {np.percentile(span[wg], 95):.3f} / {span[wg].max():.2f}", n=n_win_gap)
span_ext = span + P
rec("TW", "pencere kapsami (son-ilk + 0,1 s) > 0,5 s + tol", "pay", float((span_ext > 0.5 + TOL).mean()), n=len(W))
rec("TW", "pencere kapsami > 0,6 s (>=1 kayip yuva)", "pay", float((span_ext > 0.6 - 1e-9).mean()), n=len(W))
rec("TW", "bosluksuz pencere", "pay", 1 - n_win_gap / len(W))

# ======================================================================================= TEST 5 sure denetimi
rows5 = []
for s in SESS:
    rf = raw[s]
    g = ft[ft.session == s]
    wall = float(rf.t.iloc[-1] - rf.t.iloc[0] + P)
    nom_raw = len(rf) * P
    nom_mer = len(g) * P
    mis = int(IV[(IV.session == s) & IV.gap].k_missing.sum())
    rows5.append(dict(session=s, kare_ham=len(rf), kare_merged=len(g), wall_s=wall, kare_x_0p1_ham_s=nom_raw, kare_x_0p1_merged_s=nom_mer,
                      oran_wall_per_kare_ham=wall / nom_raw, oran_wall_per_kare_merged=wall / nom_mer, fark_s=wall - nom_mer, kayip_yuva=mis,
                      kare_plus_kayip_x_0p1_s=(len(g) + mis) * P, wall_min=wall / 60, kare_min_merged=nom_mer / 60))
    for kk in ("kare_ham", "kare_merged", "wall_s", "kare_x_0p1_merged_s", "oran_wall_per_kare_merged", "fark_s", "kayip_yuva", "kare_plus_kayip_x_0p1_s"):
        rec("T5", s, kk, rows5[-1][kk])
T5 = pd.DataFrame(rows5)
tot_wall = T5.wall_s.sum(); tot_nom = T5.kare_x_0p1_merged_s.sum()
rec("T5", "toplam", "wall s / kare x 0,1 s / oran", f"{tot_wall:.1f} / {tot_nom:.1f} / {tot_wall / tot_nom:.4f}")
rec("T5", "toplam", "wall dk / kare dk", f"{tot_wall / 60:.2f} / {tot_nom / 60:.2f}")
RATIO = dict(zip(T5.session, T5.oran_wall_per_kare_merged))
WALLMIN = dict(zip(T5.session, T5.wall_min))
sidx = {s: i for i, s in enumerate(SESS)}

# --- onceki sayilarin kaynak denetimi (salt okunur)
mins = pd.read_csv(ROOT / "reports" / "faz_a4" / "04_unseen_minutes.csv")
Wl = W.copy()
sess_of_win = Wl["session"].to_numpy()


def corrected_row(r):
    k = r["kaynak"]
    if k.startswith("Faz A"):
        vals = []
        for f in range(5):
            m = Wl["s1"].to_numpy() != f
            vals.append(sum((m & (sess_of_win == s)).sum() * RATIO[s] for s in SESS) / 600)
        return float(np.mean(vals))
    if k.startswith("(i) S2: s02"):
        m = Wl["s2"].to_numpy() != 1
        return sum((m & (sess_of_win == s)).sum() * RATIO[s] for s in SESS) / 600
    if k.startswith("(i) S2: s03"):
        m = Wl["s2"].to_numpy() != 2
        return sum((m & (sess_of_win == s)).sum() * RATIO[s] for s in SESS) / 600
    if k.startswith("(i) S2: 6"):
        vals = []
        for j in range(6):
            m = Wl["s2"].to_numpy() != j
            vals.append(sum((m & (sess_of_win == s)).sum() * RATIO[s] for s in SESS) / 600)
        return float(np.mean(vals))
    if k.startswith("(ii)"):
        nums = [int(x) for x in re.findall(r"\((\d+)\)", r["etiketli_oturumlar"])]
        return (nums[0] * RATIO[SESS[0]] + nums[1] * RATIO[SESS[1]]) / 600
    return np.nan


mins["etiketli_dakika_zaman_damgasi_onerisi"] = mins.apply(corrected_row, axis=1)
MINS = mins
for r_ in MINS.itertuples():
    rec("T5kaynak", f"04_unseen_minutes.csv | {r_.kaynak}", "mevcut dk (pencere/600) -> zaman damgali oneri dk",
        f"{r_.etiketli_dakika:.2f} -> {r_.etiketli_dakika_zaman_damgasi_onerisi:.2f}", n=int(r_.egitim_penceresi))
rec("T5kaynak", "04_unseen_sessions.py L261-273", "etiketli_dakika = egitim_penceresi / 600", "kare sayisindan (zaman damgasi degil)")

# --- main.tex / supplementary.tex dakika cumleleri (salt okunur)
TEXT_HITS = []
pat = re.compile(r"\d\\,\s*min\b|\d\s+min\b|\bminutes?\b|recording time", re.I)
for fn in ("main.tex", "supplementary.tex"):
    p = ROOT / "paper" / fn
    for ln, line in enumerate(p.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue
        if pat.search(line) and re.search(r"\d", line):
            TEXT_HITS.append((fn, ln, line.strip()))
for fn, ln, line in TEXT_HITS:
    rec("T5metin", f"{fn}:{ln}", "dakika/sure cumlesi", line[:600])

# --- paper_prep/05_gaps*: nasil hesaplanmis (metin denetimi)
gp = (ROOT / "reports" / "paper_prep" / "05_gaps.md").read_text(encoding="utf-8")
claim_line = [l for l in gp.split("\n") if "contain only radar frames that had at least one point" in l]
rec("T5kaynak", "reports/paper_prep/05_gaps.md sec.1", "'yalniz en az bir noktali kareler kayitli' iddiasi bulundu mu", bool(claim_line), note="bu, H-bos'un olgu gibi yazilmis hali")
gm = pd.read_csv(ROOT / "reports" / "paper_prep" / "05_gaps_minutes_state_session_durations.csv")
for r_ in gm.itertuples():
    rec("T5kaynak", f"05_gaps_minutes_state_session_durations | {r_.session}", "session_wall_min (paper_prep) vs bu betik wall_min",
        f"{r_.session_wall_min:.4f} vs {WALLMIN[r_.session]:.4f}")

# ======================================================================================= KARAR (onceden yazilmis kural)
res = {}
for setn in ("havuz", "onceki", "sonraki"):
    row = T2[(T2.komsu == setn) & (T2.degisken.str.startswith("n_points"))].iloc[0]
    n_state = "dusuk" if (row.komsu_pctl_medyan < 0.5 and row.p_dusuk < 0.01) else ("yuksek" if (row.komsu_pctl_medyan > 0.5 and row.p_yuksek < 0.01) else "benzer")
    nr = NONE_RES[setn]
    none_state = "yuksek" if nr["p_yuksek"] < 0.01 else ("dusuk" if nr["p_dusuk"] < 0.01 else "benzer")
    res[setn] = (n_state, none_state, float(row.komsu_pctl_medyan), float(row.p_dusuk), float(row.p_yuksek))
tof_state = "dusuk(aktif az)" if (perm_p_two < 0.01 and obs_act_pool < null_act.mean()) else ("yuksek(aktif cok)" if (perm_p_two < 0.01 and obs_act_pool > null_act.mean()) else "benzer")
n_s, none_s = res["havuz"][0], res["havuz"][1]
none_dir = {"yuksek": +1, "dusuk": -1, "benzer": 0}[none_s]            # kutu-yok payi yonu
tof_dir = {"dusuk(aktif az)": +1, "yuksek(aktif cok)": -1, "benzer": 0}[tof_state]   # aktif az = kutu-yok cok
if n_s == "dusuk" and none_dir == 1 and tof_dir == 1:
    DECISION = "H-bos DESTEKLENDI"
elif n_s == "yuksek" and none_dir == -1 and tof_dir == -1:
    DECISION = "H-yuk DESTEKLENDI"
else:
    DECISION = "BELIRSIZ (iki hipotez de reddedilmez)"
rec("KARAR", "n_points", "havuz komsu durum", n_s)
rec("KARAR", "kutu yok (komsu kare)", "havuz durum", none_s)
rec("KARAR", "ToF bosluk araligi aktif pay", "durum", tof_state)
rec("KARAR", "onceden yazilmis kural", "sonuc", DECISION)
rec("KARAR", "onceki / sonraki ayri", "n_points,kutu yok", f"{res['onceki'][:2]} / {res['sonraki'][:2]}")

# ======================================================================================= CSV
pd.DataFrame(ROWS).to_csv(OUT / "02_time_gaps.csv", index=False, encoding="utf-8")
# ek tablolar (aynı numaralar rec ile CSV'de var; ayri CSV'ler yalniz kolaylik)
T1K.to_csv(OUT / "02_time_gaps_t1_by_k.csv", index=False)
HIST.to_csv(OUT / "02_time_gaps_t1_resid_hist.csv", index=False)
T2.to_csv(OUT / "02_time_gaps_t2_neighbors.csv", index=False)
T3.to_csv(OUT / "02_time_gaps_t3_sessions.csv", index=False)
GI.to_csv(OUT / "02_time_gaps_gaps_detail.csv", index=False)
T5.to_csv(OUT / "02_time_gaps_t5_wall.csv", index=False)
MINS.to_csv(OUT / "02_time_gaps_t5_minutes_proposal.csv", index=False)

# ======================================================================================= MD
def md_table(df, fmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.to_list()) + " |")
    return "\n".join(out)


H("# C.2 — Kayıttaki zaman boşluklarının nedeni (betimsel)")
H()
H("`faz_c/02_time_gaps.py` üretir; plan `faz_c/DESIGN_C.md` C.2. Statü: KEŞİF AMAÇLI; kullanıcı görmeden `main.tex`'e girmez. Her sayı stdout'a ve `02_time_gaps.csv`'ye de yazıldı (ek ayrıntı CSV'leri: `02_time_gaps_*.csv`). Yalnız okuma: veri klasörü, `reports/*`, `paper/*.tex`. Doğal sapan s02 döngülerinin alarm durumuna bakılmadı.")
H()
H("## 0. Özet")
H()
H(f"- **Karar (önceden yazılmış kurala harfiyen): {DECISION}.**")
H(f"- Boşluk sayısı {nG} (dt > 0,15 s) — Faz B B.0(f) ile aynı; ham `mmwave_data.csv` ile kare tablosu aynı 696 aralığı verir. Tahmini kayıp kare yuvası {MISS} (kayıp payı {pct(MISS_SHARE)}: {MISS}/{N_LOG + MISS}). `frame_id` sayacı ardışık olduğundan bu sayı yalnız zaman damgasından kestirilir.")
H(f"- **Test (i) tam kat: {'GEÇTİ' if T1_PASS else 'GEÇMEDİ'}** — boşluk dt'lerinin {pct(frac_in)}'i 0,1 s'nin tam katına ±{TOL * 1000:.1f} ms içinde (eşik ≥ %90; rastgele fazda ~{pct(nulls.mean())}). Bu iki hipotezle de uyumludur; ayrım (ii)'dedir.")
H("- Boşluktan sonra 100 ms kare ızgarasının aynı fazda sürmesi, kaybın tam kare birimleriyle olduğunu gösterir; kaybın radar çıktısında mı (boş kare gönderilmedi) yoksa kaydedici/ayrıştırıcıda mı olduğunu ayırt etmez.")
H(f"- Test (ii) n_points: komşu karelerin oturum-içi yüzdelik medyanı {res['havuz'][2]:.3f} (0,5 = fark yok); MWU tek yönlü p(düşük) = {res['havuz'][3]:.3g}, p(yüksek) = {res['havuz'][4]:.3g} (1e-300 = sayısal taban; gerçek p daha küçük) → **{n_s}**. 'Kutu yok' payı komşularda {pct(NONE_RES['havuz']['share'])} (tüm kareler {pct(none_all)}) → **{none_s}**. Boşluk aralığındaki ham ToF aktif payı {pct(obs_act_pool)} (rastgele pencere {pct(null_act.mean())}, permütasyon p = {perm_p_two:.3g}) → **{tof_state}**.")
H(f"- Test (iv) kayıt kodu: kullanıcının diskinde (<LOCAL_HOME>, D:\\) kayıt/UART ayrıştırma kodu **bulunamadı**; `experiment_info.txt` kayıt makinesinin Linux'ta `<REMOTE_HOME>/Github/mmwave/mmwave_sensor` altında olduğunu söylüyor, bu kod diskte yok. Bu nedenle (iii) 'kesin kanıt' yolu kapalı.")
H(f"- Test (v): zaman damgası süresi, kare × 0,1 s'den toplamda {tot_wall:.0f} s'ye karşı {tot_nom:.0f} s (oran {tot_wall / tot_nom:.3f}). Kare sayısından hesaplanmış dakikalar var (§6, düzeltme ÖNERİSİ).")
H()

H("## 1. Test (i) — tam kat testi")
H()
H(f"Düzenli aralıklar (dt ≤ 0,15 s; n = {len(reg)}): |dt − 0,1| medyanı {np.median(dev_reg) * 1000:.2f} ms, 95. yüzdelik {p95 * 1000:.2f} ms → tolerans = 2 × p95 = **{TOL * 1000:.2f} ms** ({TOL_U:.3f} birim). Boşluklar: n = {nG}.")
H()
H(f"- Tolerans içinde: **{pct(frac_in)}** (eşik ≥ %90 → {'GEÇTİ' if T1_PASS else 'GEÇMEDİ'}).")
H(f"- Rastgele zamanlama karşılaştırması: aynı dt'lere U(0, 0,1 s) rastgele faz eklenince tolerans içinde pay ortalama {pct(nulls.mean())} (2000 tekrarın en büyüğü {pct(nulls.max())}); analitik beklenti 2 × tol_birim = {pct(min(1, 2 * TOL_U))}.")
H(f"- |dt/0,1 − round| medyanı {np.median(np.abs(G.resid_u)):.3f} birim, p95 {np.percentile(np.abs(G.resid_u), 95):.3f} birim; ortalama işaretli sapma {G.resid_u.mean() * P * 1000:+.2f} ms.")
H(f"- Yedek (bilgi): düzenli medyan periyot {per_fit:.4f} s ile aynı test: {pct(float((np.abs(Gf - np.round(Gf)) * per_fit <= TOL).mean()))}.")
H()
H("Kayıp kare sayısına (k = round(dt/0,1) − 1) göre:")
H()
H(md_table(T1K))
H()
H("`dt/0,1 − round(dt/0,1)` dağılımının tamamı (0,1 birimlik kutular; birim = 0,1 s):")
H()
H(md_table(HIST, "{:.4f}"))
H()
H(f"Okuma: {'Boşluk süreleri radar kare periyodunun tam katlarıdır; kayıp olan şey bütün radar kareleridir (kısmi/yarım kare değil) ve boşluktan sonra zaman ızgarası aynı fazda sürer.' if T1_PASS else 'Boşluk süreleri tam kat değildir.'} Bu, kare sınırından bağımsız (rastgele fazda) bir takılma ile tutarlı DEĞİLDİR; kare sınırlarında kayıpla tutarlıdır, ve \"boş kare kaydedilmedi\" ile \"kare kaydedilemedi\" (UART taşması vb.) ikisi de bunu öngörür, yani ayırt edemez. Aykırılar: tolerans dışında kalan {nG - int(round(frac_in * nG))} boşluk (bunların {int((np.abs(G.resid_u) > 0.1).sum())}'ü ±0,1 birimin ötesinde, kalanı ±0,1 içinde ama toleransın dışında). Not: zaman damgaları ana bilgisayar tarafında basıldığı için ±{TOL * 1000:.0f} ms jitter vardır; sapma uzun boşluklarda (k ≥ 10) da tolerans içinde kalır (k bölmesi tablosu), yani radar saati ile ana bilgisayar saati arasında ölçülebilir kayma yok.")
H()

H("## 2. Test (ii) — ayrım testi: boşluğun komşu kareleri")
H()
H("Komşu = boşluktan hemen önceki ve sonraki kayıtlı kare (benzersiz kareler: "
  f"önceki {len(SETS['onceki'])}, sonraki {len(SETS['sonraki'])}, havuz {len(SETS['havuz'])}). Kıyas tabanı: aynı oturumun tüm kareleri; MWU, oturum-içi yüzdelik-sıra (tüm kareler içindeki oturum içi yüzdelik) üzerinde (oturum farklarını yok eder). Komşu kareler tabana dahil olduğu için (≈%{100 * len(nb_rows) / len(ft):.1f}) testler hafif tutucudur; komşuluk örtüşmesi p-değerlerini kabaca yorumlanır kılar.")
H()
H("Önceden yazılmış okuma kuralı (çalıştırmadan önce sabit): n_points 'düşük' = yüzdelik medyan < 0,5 ve MWU tek yönlü p < 0,01 ('yüksek' simetrik); 'kutu yok' Fisher tek yönlü p < 0,01; boşluk aralığındaki ham ToF aktif payı rastgele pencerelere göre permütasyon p < 0,01 (iki yönlü). İki 'kutu yok' kaynağı aynı yönü göstermezse belirsiz. (Bu kural brifin 'düşük/yüksek'ini sayısallaştırır; brifteki hipotez tahminlerini değiştirmez.)")
H()
H(md_table(T2.assign(**{c: T2[c] for c in T2.columns})))
H()
H(f"- 'Kutu yok' payı: tüm kareler {pct(none_all)}; komşu havuz {pct(NONE_RES['havuz']['share'])} (Fisher p(yüksek) = {NONE_RES['havuz']['p_yuksek']:.3g}, p(düşük) = {NONE_RES['havuz']['p_dusuk']:.3g}); önceki {pct(NONE_RES['onceki']['share'])}; sonraki {pct(NONE_RES['sonraki']['share'])}.")
H(f"- Boşluk öncesi son 5 karenin nokta-sayısı eğimi (kare başına): medyan {np.median(slopes_pre):+.3f} (tüm pencereler {np.median(slopes_all):+.3f}); negatif eğim payı {pct((slopes_pre < 0).mean())} (tümü {pct((slopes_all < 0).mean())}); MWU p(düşük) = {sl_p_less:.3g}. Ortalama eğim {slopes_pre.mean():+.3f} vs {slopes_all.mean():+.3f}.")
H(f"- Nokta sayısı dağılımı: n=1 payı tüm kareler {pct((ft.n_points == 1).mean())}, komşu havuz {pct((ft.iloc[nb_rows].n_points == 1).mean())}; n=2: {pct((ft.n_points == 2).mean())} vs {pct((ft.iloc[nb_rows].n_points == 2).mean())}.")
H()
H("Oturum bazında (komşu havuz / tüm kareler):")
H()
sess_tab = []
for s in SESS:
    g = ft[ft.session == s]; nbs = g[g["row"].isin(nb_rows)]
    sess_tab.append(dict(oturum=s, n_komsu=len(nbs), npts_medyan_komsu=nbs.n_points.median(), npts_medyan_tum=g.n_points.median(),
                         kutuyok_komsu=nbs.none.mean(), kutuyok_tum=g.none.mean(), menzil_komsu=nbs.range_mean.median(), menzil_tum=g.range_mean.median()))
H(md_table(pd.DataFrame(sess_tab), "{:.3f}"))
H()

H("## 3. Test (iii-a) — oturum bazında boşluk oranı ve boşluk anlarında ham ToF")
H()
H(md_table(T3.assign(session=T3.session.str.slice(0, 10)), "{:.4g}"))
H()
H(f"Ham ToF (~30 Hz) ve video radar karelerinden bağımsız örneklenir ve aynı ana bilgisayar saatini kullanır. Boşluk aralığı = (önceki kaydın zaman damgası, sonraki kaydın zaman damgası); duyarlılık için aralık 0,1 s ve 0,2 s geriye kaydırıldı (radar zaman damgası ToF sayaç değişiminden medyan {abs(LAG) * 1000:.0f} ms sonra basılır; fiziksel yakalama damgadan önce olduğundan geri kaydırma haklıdır). ToF/video satırlarındaki `frame_id` 'şimdiye dek yazılan radar karesi sayısıdır' (ToF satırı, yazılmış son radar karesinin damgasından ~{(0.1 + LAG) * 1000:.0f} ms sonra bir sonraki kimliğe geçer).")
H()
H(f"- {int(ok.sum())} boşlukta aralığa düşen ToF satırı var ({int(GI.tof_rows.sum())} satır). Aktif ('kutu' ToF < 140 mm) pay: havuz **{pct(obs_act_pool)}**, boşluk ortalaması {pct(obs_act_mean)}; aynı satır sayısında rastgele ToF pencereleri: ort. {pct(null_act.mean())} [q2,5–q97,5 = {pct(np.quantile(null_act, .025))}, {pct(np.quantile(null_act, .975))}] (R = {R}); permütasyon p (iki yönlü) = {perm_p_two:.3g} (boşluk-ortalaması için p = {perm_p_two_w:.3g}). Tüm ToF'de aktif pay {pct(tof_all_active)}.")
H(f"- Boşlukların {pct(float((GI.loc[ok, 'act_frac'] == 0).mean()))}'i tümüyle 'kutu yok' iken {pct(float((GI.loc[ok, 'act_frac'] == 1).mean()))}'i baştan sona 'kutu aktif'.")
H(f"- ToF etiket değişim oranı (satır başına): boşluklarda {obs_chg_pool:.4f}, rastgele pencerelerde {chg_null_mean:.4f} (p(düşük) = {p_lo_c:.3g}, p(yüksek) = {p_hi_c:.3g}) — el boşlukta durağan mı?")
H(f"- Kayıp yuva başına tehlike (S1): P(kayıp | ToF 'kutu yok') = {haz_none:.4f}; P(kayıp | kutu aktif) = {haz_box:.4f}; oran {haz_none / haz_box:.2f}. Kayıp yuvaların {pct(nm_none / nm)}'i 'kutu yok' (kayıtlı karelerde {pct(nl_none / nl)}).")
H(f"- Kaydedici canlılığı (S3): boşluk ≥ 1 s olan {len(alive)} boşlukta ToF oranı min/medyan {tof_rate.min():.1f}/{tof_rate.median():.1f} satır/s, video {vid_rate.min():.1f}/{vid_rate.median():.1f} kare/s; tüm boşluklarda ToF iç-aralık max dt {GI.tof_max_dt.max():.3f} s, video {GI.vid_max_dt.max():.3f} s. Boşluk aralığındaki ToF satırlarının {pct(float(GI.tof_fid_eq_pre_plus1.sum() / GI.tof_rows.sum()))}'inde `frame_id` = önceki kayıtlı kare + 1 (yani sonraki kaydın kimliği; önceki kimliğe eşit olanlar {pct(float(GI.tof_fid_eq_pre.sum() / GI.tof_rows.sum()))}): sayaç boşluk boyunca ilerlemiyor, yalnız YAZILAN radar karesiyle artıyor. Bu yüzden `frame_id` ardışıklığı iki hipotezi de aynı şekilde öngörür.")
H()
H(f"- Not (şeffaflık): permütasyon p-değerleri R = {R} ile sınırlıdır (en küçük ölçülebilir p ≈ 1/R = {1 / R:.3f}); gözlenen aktif pay, {R} rastgele pencere kümesinin hiçbirinde bu kadar düşük çıkmadı. İlk koşuda ToF aralığı yanlışlıkla `frame_id`'nin 'önceki kare' kimliği taşıdığı varsayımıyla (sayaç gecikmesi düzeltmeli) kurulmuştu; ToF satırlarının aslında 'şimdiye dek yazılan kare sayısı' (= sonraki kare kimliği) taşıdığı görülünce aralık (ts_önceki, ts_sonraki) olarak yeniden tanımlandı. Yön değişmedi (aktif pay %0,58 → %0,87), karar kuralı aynı.")
H()
H("En uzun 10 boşluk:")
H()
H(md_table(LONG[["session", "t_pre", "dt", "k", "act_frac", "n_pre", "n_post", "lab_pre", "lab_post", "range_pre", "range_post"]].assign(session=lambda d: d.session.str.slice(0, 10)), "{:.3g}"))
H()

H("## 4. Ek ölçümler (sezgisel; karar kuralına girmez)")
H()
H(f"- **S2 (sıfır-nokta ekstrapolasyonu).** Kayıtlı karelerde nokta sayısı payı p1 = {p1:.4f}, p2 = {p2:.4f}. Boş karelerin kayıtlı karelere oranı (gözlenen) M/N = {odds_obs:.4f}. n_points dağılımının sıfıra Poisson (p1²/(2·p2)) ekstrapolasyonu {p0rel_pois:.4f}, geometrik (p1²/p2) {p0rel_geo:.4f}. Bu karşılaştırma model varsayımına bağlıdır; H-boş'u KANITLAMAZ, yalnız büyüklük mertebesi kontrolüdür.")
H(f"- **S4 (statik noktalar).** Ham noktaların {pct(float((allraw.velocity == 0).mean()))}'i v = 0 (merged_df bunları atar). {int((fr_static.ns > 0).sum())} karede ≥ 1 statik nokta var ama yalnız-statik kare sayısı {int((fr_static.n == fr_static.ns).sum())}. Tek-nokta karelerinde yalnız-statik payı {pct(float((fr_static[fr_static.n == 1].n == fr_static[fr_static.n == 1].ns).mean()))} (bağımsızlık altında beklenen ≈ {pct(float((allraw.velocity == 0).mean()))}) — kayıtlı kareler daima en az bir hareketli nokta içeriyor.")
H()

H("## 5. Test (iv) — kayıt kodu araması (salt okunur, çalıştırılmadı)")
H()
H("Arama: `rg -l` ile `<LOCAL_HOME>` ve `D:\\` altında (AppData, .git, node_modules, site-packages, venv hariç; ≤ 5 MB; *.py/ipynb/c/cpp/h/js/ts/m/sh/txt/md/cs/java), anahtarlar `numDetectedObj`, `0x0102`, `0x0201`, `mmwave_data.csv`, `magic word`; ikinci tur: `serial.Serial`, `experiment_data`, `distance_sensors`, `video_frames.csv`, `mmwave_sensor`, `IWR6843`; ayrıca dosya/klasör adı taraması (`*mmwave*`, `*uart*`, `*radar*`, `*recorder*`) ve `mmwave_anomaly.zip` içerik listesi. Eşleşen dosya listesi: `faz_c/02_time_gaps_search.txt`.")
H()
H("- **Kayıt/UART ayrıştırma kodu bulunamadı.** `mmwave_data.csv`'yi yalnız *okuyan* kod var: `00_dataprepare_all.ipynb` (üç kopya: `Documents\\GitHub\\mmwave_anomaly`, `Github\\jupyter_notebook\\data\\working`, dondurulmuş 4 numaralı klasör `dataset\\notebooks`) ve çıkarım not defterleri. Bunlar kayıt sonrası hazırlık: hücre 0, satır 40–42 `velocity != 0` ile v = 0 noktalarını atar; satır 46–58 `mmwave`/ToF `frame_id` kesişimini iç birleştirir (`how='inner'`) ve ToF'de `frame_id` başına ilk satırı tutar. **Bu kod boş kareyi atlayan bir koşul içermez ve içeremez** (kayıt sonrası; boş kareler zaten dosyada satır üretmez). Kayıt kodu (`mmwave_sensor`, FastAPI + pyserial + opencv; `experiment_info.txt`'e göre `<REMOTE_HOME>/Github/mmwave/mmwave_sensor/venv` altında) bu diskte yok; muhtemelen kayıt makinesinde (Linux).")
H("- İlgili diğer bulgu: `mmwave_data.csv` uzun biçimli (nokta başına bir satır). Sıfır noktalı bir kare, kaydedici onu sayaçla numaralasa bile **dosyada hiç satır üretmez**. `frame_id` ardışık olduğuna göre kaydedici ya boş kareleri sayaca dahil etmiyor ya da hiç görmedi — dosyadan ayırt edilemez.")
H("- Sonuç: (iii) 'kesin kanıt' yolu bu oturumda kapalı. **Önerilen eylem:** kayıt makinesindeki `mmwave_sensor` kodunda UART okuma/ayrıştırma döngüsünde `numDetectedObj == 0` ya da magic-word hatası durumunda `continue`/`return` olup olmadığına ve `frame_id` artışının nerede yapıldığına bakılmalı (kullanıcı veya erişimi olan oturum).")
H()

H("## 6. Test (v) — süre hesabı denetimi")
H()
H("Oturum başına duvar saati (ilk → son radar zaman damgası + 0,1 s) ve kare × 0,1 s:")
H()
H(md_table(T5[["session", "kare_ham", "kare_merged", "wall_s", "kare_x_0p1_merged_s", "oran_wall_per_kare_merged", "fark_s", "kayip_yuva", "kare_plus_kayip_x_0p1_s"]].assign(session=lambda d: d.session.str.slice(0, 10)), "{:.4g}"))
H()
H(f"Toplam: duvar saati {tot_wall:.1f} s ({tot_wall / 60:.2f} dk), kare × 0,1 s = {tot_nom:.1f} s ({tot_nom / 60:.2f} dk); oran {tot_wall / tot_nom:.4f}. (kayıtlı + tahmini kayıp) × 0,1 s duvar saatiyle ±0,1 s'lik jitter ölçeğinde örtüşür (tablo, son sütun). Ham dosyada 16,274, merged'de 16,272 kare var; s05'te 2 kare (`frame_id` 2 ve 4) ToF'de karşılığı olmadığından iç birleştirmede düşmüş (merge artefaktı, boşluk nedeni değil).")
H()
H("**Kaynak denetimi.**")
H()
H("| Kaynak | Hesap | Sonuç |")
H("|---|---|---|")
H("| `reports/faz_a4/04_unseen_minutes.csv` (`04_unseen_sessions.py` L261–273) | `etiketli_dakika = egitim_penceresi / 600` | **Kare sayısından**; zaman damgası değil → düzeltme önerisi aşağıda |")
H("| `reports/paper_prep/05_gaps_minutes_*.csv` | döngü süresi `t(sonraki döngünün ilk karesi) − t(bu döngünün ilk karesi)` | **Zaman damgasından** (`session_wall_min` bu betikteki wall_min ile aynı; tablo CSV'de) |")
H("| `reports/paper_prep/05_gaps.md` §1 | 'merged_df … yalnız en az bir noktalı kareleri içerir' | Bu, H-boş hipotezinin **olgu gibi yazılmış hali**; bu rapor onu doğrulayamadı (bkz. §0, §5) |")
H()
H("`04_unseen_minutes.csv` için zaman damgası temelli ÖNERİ (oturum oranı wall/(kare × 0,1 s) ile ölçeklenmiş; dosyalara dokunulmadı):")
H()
H(md_table(MINS[["kaynak", "egitim_penceresi", "etiketli_dakika", "etiketli_dakika_zaman_damgasi_onerisi"]], "{:.2f}"))
H()
H("`paper/main.tex` / `supplementary.tex` içindeki dakika/süre cümleleri (satır numarası, kaynak sınıfı; düzeltme ÖNERİSİ, dosyaya dokunulmadı):")
H()
for fn, ln, line in TEXT_HITS:
    H(f"- `{fn}:{ln}`: {line[:300]}{'…' if len(line) > 300 else ''}")
H()
_FA = float(MINS.loc[MINS.kaynak.str.startswith("Faz A"), "etiketli_dakika_zaman_damgasi_onerisi"].iloc[0])
_LO = float(MINS.loc[MINS.kaynak.str.contains("6 modelin"), "etiketli_dakika_zaman_damgasi_onerisi"].iloc[0])
_IIF = float(MINS.loc[MINS.kaynak.str.contains("ii_f"), "etiketli_dakika_zaman_damgasi_onerisi"].mean())
_IIA = float(MINS.loc[MINS.kaynak.str.contains("ii_all"), "etiketli_dakika_zaman_damgasi_onerisi"].iloc[0])
_W01, _W02 = WALLMIN[SESS[0]], WALLMIN[SESS[1]]
CLS = {
    ("main.tex", 506): ("zaman damgası", "değişiklik gerekmez (05_gaps §1a; döngü süreleri zaman damgasından)"),
    ("main.tex", 508): ("zaman damgası", "değişiklik gerekmez (5,88 dk = 05_gaps_minutes_state_total.csv `min_b`)"),
    ("main.tex", 510): ("**kare sayısı** (pencere/600)", f"\"about 7.8 and 3.8 min (11.6 min …)\" yerine zaman damgalı kayıt süresi: s01 {_W01:.1f} ve s02 {_W02:.1f} dk ({_W01 + _W02:.1f} dk); \"21.7 min of labelled frames\" etiketli veri miktarı olarak kalabilir (kare sayısı), ama kayıt süresiyle yan yana konursa duvar saati karşılığı {_FA:.1f} dk; \"about 1.7 min each\" zaman damgalı (20 × 5,0 s), tamam"),
    ("main.tex", 515): ("zaman damgası (başlık bunu söylüyor)", "değişiklik gerekmez"),
    ("main.tex", 558): ("**kare sayısı** (pencere/600)", f"\"about 11.6 min in total\" → \"about {_W01 + _W02:.0f} min\" ({_W01 + _W02:.2f} dk duvar saati)"),
    ("supplementary.tex", 85): ("zaman damgası", "değişiklik gerekmez"),
    ("supplementary.tex", 306): ("**kare sayısı** (metin 'windows divided by 600' diye açıkça söylüyor)", f"tanım açık olduğu için kendi içinde tutarlı; ama ana metin 510/558 aynı sayıları kayıt süresi diye aktarıyor. Öneri: duvar saati karşılıklarını yan yana ver (pooled S1 kat modeli ≈ {_FA:.1f} dk, LOSO ≈ {_LO:.1f} dk, s01 {_W01:.1f}, s02 {_W02:.1f}, (ii) kat modelleri ≈ {_IIF:.1f}, tüm-veri ≈ {_IIA:.1f} dk)"),
    ("supplementary.tex", 307): ("zaman damgası", "değişiklik gerekmez"),
    ("supplementary.tex", 363): ("zaman damgası", "değişiklik gerekmez"),
    ("supplementary.tex", 365): ("zaman damgası", "değişiklik gerekmez"),
}
H("| Satır | Sayı/cümle kaynağı | Öneri |")
H("|---|---|---|")
for fn, ln, line in TEXT_HITS:
    c = CLS.get((fn, ln), ("(sınıflanmadı)", ""))
    H(f"| `{fn}:{ln}` | {c[0]} | {c[1]} |")
H()
H("Ek: `main.tex` 'What the radar stream contains' bölümündeki \"About 14.7% of radar periods produced no frame row\" cümlesi bu betikle uyumlu (kayıp yuva payı " + pct(MISS_SHARE) + ": " + f"{MISS}/{N_LOG + MISS}" + "; 05_gaps 2801/19075'i, ham kare toplamıyla, kullanır) ve nedeni iddia etmiyor. **Nedeni olgu gibi yazan yer `reports/paper_prep/05_gaps.md` §1.**")
H()

H("## 7. Okuma, makale cümlesi ve sınırlar")
H()
H(f"**Önceden yazılmış kurala göre karar: {DECISION}.** Dayanak (hepsi önceden yazılmış eşiklerle): (a) komşu karelerde nokta sayısı belirgin düşük (medyan {ft.iloc[nb_rows].n_points.median():.0f} nokta, tüm kareler {ft.n_points.median():.0f}; n = 1 payı {pct((ft.iloc[nb_rows].n_points == 1).mean())} vs {pct((ft.n_points == 1).mean())}); (b) komşu karelerin {pct(NONE_RES['havuz']['share'])}'i 'kutu yok' (tüm kareler {pct(none_all)}); (c) boşluk aralığındaki ham ToF'de aktif pay {pct(obs_act_pool)} (rastgele pencere {pct(null_act.mean())}) ve aralıkta etiket değişimi yok; (d) boşluk öncesi son 5 karede nokta sayısı eğimi medyan {np.median(slopes_pre):+.2f} (tümü {np.median(slopes_all):+.2f}); (e) yuva başına kayıp tehlikesi 'kutu yok'ta {haz_none:.3f}, kutu aktifken {haz_box:.4f} (oran ≈ {haz_none / haz_box:.0f}). Önceki ve sonraki komşular ayrı ayrı aynı yönü verir.")
H()
H("**Ne gösterilebildi, ne gösterilemedi.**")
H()
H(f"- Gösterilen: kayıp kareler bütün radar kareleridir (test (i)); kayıp kareler el kutu dışındayken ve nokta sayısı zaten azalırken kümelenmiştir; ToF/video kaydı boşluk boyunca çalışmaya devam eder (ToF ≈ {tof_rate.median():.0f} satır/s, video ≈ {vid_rate.median():.0f} kare/s), yani tüm kayıt süreci durmuyor. Bu örüntü 'boş kareler kaydedilmedi' ile uyumlu; 'kaydedici yük altında içerikten bağımsız kare düşürdü' ile uyumsuz (o durumda komşu kareler tüm karelere benzerdi).")
H("- Gösterilemeyen: boş karelerin **kim tarafından** atlandığı (radar yazılımı mı, UART ayrıştırıcı mı, kaydedici mi) ve atlamanın bilinçli bir koşul mu, yoksa 'sıfır nesne → ayrıştırma başarısız → kare düştü' yan etkisi mi olduğu. Kayıt kodu diskte yok (§5). `frame_id` sayacı boşluk boyunca ilerlemediği için ardışıklığı iki hipotezde de aynıdır ve ayrım sağlamaz.")
H(f"- Sınırlar: (1) 'nokta az' ile 'kare yok' sebep-sonuç değil korelasyondur; ortak bir neden (ör. el uzakken ayrıştırıcının yavaşlaması) bu verilerle tamamen dışlanamaz, ama içerikten bağımsız rastgele düşürme dışlanır. (2) Nokta sayısı dağılımından sıfıra Poisson/geometrik ekstrapolasyon (S2) kayıp/kayıtlı oranından ({odds_obs:.3f}) çok küçük ({p0rel_pois:.3f}–{p0rel_geo:.3f}); bu, H-boş'u nicel olarak ne destekler ne çürütür (nokta sayısı 'kutu yok' anlarında bimodal olabilir: boş sahnede ya hiç ya birkaç nokta), yalnız ekstrapolasyon kaba bir araçtır. (3) Zaman damgaları sunucu saatiyle basılır; tolerans ±{TOL * 1000:.1f} ms.")
H()
H("**Fiziksel yorum (varsayım, doğrulanmadı).** Yapılandırmada `clutterRemoval -1 1` (statik kirlilik temizleme açık). El kutu dışında ve durağanken radar hareketli nokta üretmeyebilir; sıfır nesneli kare çıktıda ya hiç yer almaz ya da kaydedilmez. Ham dosyada kayıtlı karelerin tümü ≥ 1 hareketli nokta içerir (yalnız-statik kare yok, §4 S4) — bu da 'v = 0 olan ama hareketli noktası olmayan kare' diye bir sınıfın hiç kaydedilmediğini gösterir. Kayıt kodu görülünce doğrulanır.")
H()
H("**Makale cümlesi.** Plan kuralına göre H-boş destekleniyor; önerilen cümle (kullanıcı onayı olmadan `main.tex`'e girmez; \"About 14.7% of radar periods produced no frame row\" cümlesinin yerine geçebilir):")
H()
H(f"> *Frames in which the radar reported no points were not logged ({MISS:,} slots, {pct(MISS_SHARE)} of radar periods); a 5-frame window therefore spans more than 0.5 s in {pct(n_win_gap / len(W))} of windows.*")
H()
n_win_none = int((wg & (W['y_true'].to_numpy() == 0)).sum())
H(f"Doğrulanabilirlik uyarısı: \"no points … were not logged\" kısmı kayıt koduyla doğrulanmadı (yalnız dolaylı kanıt). Temkinli alternatif: \"Radar periods without a logged frame ({MISS:,} slots, {pct(MISS_SHARE)}) occurred almost only while no box was active and the point count was already low; a 5-frame window therefore spans more than 0.5 s in {pct(n_win_gap / len(W))} of windows.\" Pencere payı ({n_win_gap}/{len(W)}) Faz B B.0(f) ile aynı; boşluklu pencerelerin {n_win_none}'ü ({pct(n_win_none / max(n_win_gap, 1))}) 'none' sınıfında, yani boşluk neredeyse hiç kutu aktifken olmuyor. Pencere payı tanımı: en az bir ardışık aralığın > 0,15 s olması (%{100 * n_win_gap / len(W):.1f}); alternatif tanım (son−ilk kare + 0,1 s > 0,5 s + tolerans) %{100 * float((span_ext > 0.5 + TOL).mean()):.1f} verir (jitter birikimi nedeniyle biraz şişer); ≥ 1 kayıp yuva içeren (kapsam > 0,6 s) pencere payı %{100 * float((span_ext > 0.6 - 1e-9).mean()):.1f}.")
H()
H("**Düzeltme önerileri (özet; dosyalara dokunulmadı).**")
H()
_r12 = (_W01 + _W02) * 60 / ((len(ft[ft.session == SESS[0]]) + len(ft[ft.session == SESS[1]])) * P)
H(f"1. `main.tex` 510 ve 558 ile `supplementary.tex` 306: kare sayısından hesaplanmış dakikalar (11.6 / 21.7 / 7.8 / 3.8 / 10.8 / 22.6) kayıt süresi diye kullanılıyorsa zaman damgalı karşılıklarla değiştirilmeli (s01 {_W01:.1f} dk, s02 {_W02:.1f} dk, toplam {_W01 + _W02:.1f} dk). Fark tüm oturumlarda %{100 * (tot_wall / tot_nom - 1):.0f} (toplam oran {tot_wall / tot_nom:.3f}); s01+s02 için oran {_r12:.3f}.")
H("2. `reports/faz_a4/04_unseen_minutes.csv` ve `04_unseen_sessions.py` L261–273: `etiketli_dakika` kare sayısıdır; sütun adı 'etiketli_kare_dakikası' yapılmalı ya da zaman damgalı sütun eklenmeli (öneri sütunu `02_time_gaps_t5_minutes_proposal.csv`).")
H("3. `reports/paper_prep/05_gaps.md` §1: \"`merged_df`/the frame table contain only radar frames that had at least one point\" cümlesi, olgu gibi yazılmış bir hipotezdir; 'dolaylı kanıtla uyumlu, kayıt koduyla doğrulanmamış' diye yumuşatılmalı.")
H()
H("## 8. Dosyalar")
H()
H("`faz_c/02_time_gaps.py`, `faz_c/02_time_gaps_search.txt` (test 4 arama çıktısı), `reports/faz_c/02_time_gaps.md`, `02_time_gaps.csv` (tüm sayılar; sütunlar section,item,stat,value,n,note), `02_time_gaps_t1_by_k.csv`, `02_time_gaps_t1_resid_hist.csv`, `02_time_gaps_t2_neighbors.csv`, `02_time_gaps_t3_sessions.csv`, `02_time_gaps_gaps_detail.csv` (696 boşluğun tek tek özellikleri), `02_time_gaps_t5_wall.csv`, `02_time_gaps_t5_minutes_proposal.csv`.")
H()
open(OUT / "02_time_gaps.md", "w", encoding="utf-8").write("\n".join(MD) + "\n")
print("\n".join(MD))
print("[done] 02_time_gaps")

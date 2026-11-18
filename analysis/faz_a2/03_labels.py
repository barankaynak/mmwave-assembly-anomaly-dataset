"""Faz A2.3 -- etiket duyarliligi (s05'te silinen Box5).  Brief: faz_a2/DESIGN_A2.md  (A2.3).

Yeniden egitim YOK. Mevcut S1 OOF tahminleri (b2, cnngru, cnnonly, b1) alternatif dogrulara karsi yeniden skorlanir.
Pencere etiketi = son karenin etiketi (OOF satiri = pencerenin son karesi).

  L0  Faz A y_true  (ham 10 Hz etiket, +-3 kare mod)
  L1  ham 10 Hz etiket (ft.y_raw; karenin ilk ToF ornegi, argmin(mesafe)<140 mm)
  L2  +-1 kare mod (fc.mode_smooth ile ayni beraberlik kurali: en kucuk sinif id)
  L3  30 Hz ham ToF "dokunus" etiketi: karenin (t-100 ms, t] araligindaki ToF orneklerinde mesafe<140 mm olan kutu;
      birden fazlaysa en cok ornek alan, beraberlikte en son ornek (en son aktif oldugu zaman damgasi).
  L3b (duyarlilik kontrolu) ayni oylama, ama ToF satirlari zaman penceresiyle degil distance_sensors.csv'deki
      frame_id ile karelere atanir (veri setinin merged_df'te kullandigi eslestirme).

Cikti: reports/faz_a2/03_labels.md (+ 03_labels_metrics.csv, 03_labels_loopcov.csv, 03_labels_frames.csv.gz)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a"))
import fa_common as fc  # noqa: E402  (adds probe/ to sys.path)
import common as pc  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = ROOT / "reports" / "faz_a2"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ("b2", "cnngru", "cnnonly", "b1")
MNAME = {"b2": "B2", "cnngru": "CNN-GRU", "cnnonly": "CNN-only", "b1": "B1"}
LABELS = ("L0", "L1", "L2", "L3")
LDESC = {"L0": "Faz A y_true (+-3 mod)", "L1": "ham 10 Hz", "L2": "+-1 mod", "L3": "30 Hz ToF dokunus"}
SESS = pc.sessions()
SNAME = {s: s.split("_")[0] for s in SESS}          # session_05_... -> session ; short id below
SHORT = {s: "s" + s.split("_")[1] for s in SESS}
NC = fc.N_CLASSES
WIN_S = 0.1
B_BOOT = 1000
SEED_BOOT = 20261104

P = print


# ------------------------------------------------------------------------------------------ L3 (30 Hz ToF)
def vote(tof_dist_lt: np.ndarray, lo: np.ndarray, hi: np.ndarray, dist: np.ndarray):
    """Per frame: samples tof[lo:hi]; active[b] = dist_b < 140.  Winner = box with most active samples;
    tie -> box whose LAST active sample is latest; still tied (same sample) -> smaller distance at that sample.
    Returns (label int array, n_samples, n_tie_count, n_tie_unresolved)."""
    n = len(tof_dist_lt)
    nf = len(lo)
    csum = np.vstack([np.zeros((1, 5), dtype=np.int64), np.cumsum(tof_dist_lt, axis=0, dtype=np.int64)])
    cnt = csum[hi] - csum[lo]                                           # (nf, 5) active-sample counts
    # last active index (<= i) per box, forward-filled
    idx = np.arange(n)[:, None]
    last = np.maximum.accumulate(np.where(tof_dist_lt > 0, idx, -1), axis=0)   # (n, 5)
    lastw = np.full((nf, 5), -1, dtype=np.int64)
    ok = hi > lo
    li = last[np.clip(hi - 1, 0, n - 1)]                                 # last active idx <= hi-1
    lastw = np.where(ok[:, None] & (li >= lo[:, None]), li, -1)
    mx = cnt.max(axis=1)
    lab = np.zeros(nf, dtype=np.int64)
    ntie = 0
    nunres = 0
    for k in range(nf):
        if mx[k] == 0:
            continue
        cand = np.flatnonzero(cnt[k] == mx[k])
        if len(cand) > 1:
            ntie += 1
            lt = lastw[k, cand]
            cand = cand[lt == lt.max()]
            if len(cand) > 1:
                nunres += 1
                i = lastw[k, cand[0]]
                cand = cand[[int(np.nanargmin(dist[i, cand]))]]
        lab[k] = int(cand[0]) + 1
    return lab, (hi - lo), ntie, nunres


def tof_labels(ft_s: pd.DataFrame, s: str) -> dict:
    d = pc.load_tof_raw(s)                                     # sorted by t, stable; has frame_id
    tt = d["t"].to_numpy()
    dist = d[pc.DIST_COLS].to_numpy(dtype=float)
    lt = np.where(np.isnan(dist), False, dist < pc.TOF_ACTIVE_MM).astype(np.int64)
    dist_f = np.where(np.isnan(dist), np.inf, dist)
    t = ft_s["t"].to_numpy()
    # time window (t-100ms, t]  -> indices lo..hi  (right-inclusive: side='right' for both ends)
    lo = np.searchsorted(tt, t - WIN_S, side="right")
    hi = np.searchsorted(tt, t, side="right")
    l3, ns, ntie, nun = vote(lt, lo, hi, dist_f)
    # frame_id based (dataset's own assignment)
    fid = d["frame_id"].to_numpy()
    assert (np.diff(fid) >= 0).all()
    f_ids = ft_s["frame_id"].to_numpy()
    lo2 = np.searchsorted(fid, f_ids, side="left")
    hi2 = np.searchsorted(fid, f_ids, side="right")
    l3b, ns2, ntie2, nun2 = vote(lt, lo2, hi2, dist_f)
    # ToF satiri -> kare zamani farki (frame_id atamasi): dt = t_kare(frame_id) - t_tof  (kare zaman damgasi - ToF zaman damgasi)
    ft_t = pd.Series(ft_s["t"].to_numpy(), index=ft_s["frame_id"].to_numpy())
    tk = ft_t.reindex(fid).to_numpy()
    in_ft = ~np.isnan(tk)
    dt = (tk - tt)[in_ft]
    # zaman kuralina (t-100ms, t] gore ayni satir ayni kareye mi dusuyor?
    pos = np.clip(np.searchsorted(ft_s["t"].to_numpy(), tt, side="left"), 0, len(ft_s) - 1)
    in_win = (ft_s["t"].to_numpy()[pos] - tt) < WIN_S
    assigned_time = np.where(in_win, ft_s["frame_id"].to_numpy()[pos], -1)
    row_agree = float(np.mean(assigned_time == fid))
    dtinfo = dict(tof_in_ft=float(in_ft.mean()), dt_med=float(np.median(dt)), dt_p90=float(np.percentile(dt, 90)),
                  dt_gt100=float(np.mean(dt > WIN_S)), dt_neg=float(np.mean(dt < 0)))
    return dict(L3=l3, L3b=l3b, n_samp=ns, n_samp_b=ns2, ntie=ntie, nunres=nun, ntie_b=ntie2, nunres_b=nun2,
                row_agree=row_agree, n_tof=len(d), **dtinfo)


# ------------------------------------------------------------------------------------------ metrics from confusion
def cm_of(y, p) -> np.ndarray:
    return np.bincount(y * NC + p, minlength=NC * NC).reshape(NC, NC)


def metrics_from_cm(cm: np.ndarray) -> dict:
    """acc, macro-F1 over classes present in y_true (rows), per-class F1; == fc.state_metrics."""
    tp = np.diag(cm).astype(float)
    sup = cm.sum(axis=1).astype(float)
    pred = cm.sum(axis=0).astype(float)
    den = sup + pred
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    present = sup > 0
    n = cm.sum()
    out = {"n": int(n), "acc": float(tp.sum() / n) if n else float("nan"),
           "macro_f1": float(f1[present].mean()) if present.any() else float("nan")}
    p5 = present.copy()
    p5[5] = False
    out["macro_f1_no5"] = float(f1[p5].mean()) if p5.any() else float("nan")
    for c in range(NC):
        out[f"f1_{c}"] = float(f1[c]) if present[c] else float("nan")
    return out


def main():
    P("=== A2.3 etiket duyarliligi ===")
    ft = fc.frame_table()
    P(f"kare tablosu: {len(ft)} kare")

    # ---------------- alternatif etiketler
    alt = ft[["session", "frame_id", "loop_id", "t", "y_true", "y_raw"]].copy()
    alt["L0"] = ft["y_true"].to_numpy()
    alt["L1"] = ft["y_raw"].to_numpy()
    l2 = np.zeros(len(ft), dtype=np.int64)
    l3 = np.zeros(len(ft), dtype=np.int64)
    l3b = np.zeros(len(ft), dtype=np.int64)
    nsamp = np.zeros(len(ft), dtype=np.int64)
    nsamp_b = np.zeros(len(ft), dtype=np.int64)
    tof_info = {}
    for s in SESS:
        m = (ft["session"] == s).to_numpy()
        ys = ft.loc[m, "y_raw"].to_numpy()
        l2[m] = fc.mode_smooth(ys, w=1)
        # kontrol: w=3 yeniden uretimi L0'a esit mi
        assert (fc.mode_smooth(ys, w=fc.SMOOTH_W) == ft.loc[m, "y_true"].to_numpy()).all()
        r = tof_labels(ft.loc[m].reset_index(drop=True), s)
        l3[m], l3b[m], nsamp[m], nsamp_b[m] = r["L3"], r["L3b"], r["n_samp"], r["n_samp_b"]
        tof_info[s] = r
    alt["L2"], alt["L3"], alt["L3b"] = l2, l3, l3b
    alt["n_tof_win"], alt["n_tof_fid"] = nsamp, nsamp_b
    alt.drop(columns=["y_true", "y_raw"]).to_csv(OUT / "03_labels_frames.csv.gz", index=False, compression="gzip",
                                                 lineterminator="\n")

    # ---------------- L3 uyum / hizalama tanisi
    P("\n--- ToF -> kare hizalamasi (L3) ---")
    rows = []
    for s in SESS:
        m = (alt["session"] == s).to_numpy()
        a = alt[m]
        r = tof_info[s]
        rows.append({
            "oturum": SHORT[s], "kare": int(m.sum()), "tof_satir": r["n_tof"],
            "ornek/kare_medyan": float(np.median(a["n_tof_win"])), "ornek/kare_min": int(a["n_tof_win"].min()),
            "ornek/kare_max": int(a["n_tof_win"].max()), "bos_pencere": int((a["n_tof_win"] == 0).sum()),
            "tof_satir_karesi_var": r["tof_in_ft"], "dt_medyan_s": r["dt_med"], "dt_p90_s": r["dt_p90"],
            "dt>0.1s_orani": r["dt_gt100"], "dt<0_orani": r["dt_neg"],
            "satir_zaman==frame_id": r["row_agree"],
            "L3==L3b": float((a["L3"] == a["L3b"]).mean()),
            "L3==L1": float((a["L3"] == a["L1"]).mean()), "L3==L0": float((a["L3"] == a["L0"]).mean()),
            "L1==L0": float((a["L1"] == a["L0"]).mean()), "L2==L0": float((a["L2"] == a["L0"]).mean()),
            "beraberlik_L3": r["ntie"], "cozulmeyen_L3": r["nunres"],
        })
    tdiag = pd.DataFrame(rows)
    P(pc.md_table(tdiag, "{:.4f}"))

    # ---------------- sinif dagilimi & dongu kapsami
    P("\n--- kare sayisi (sinif basina) ---")
    drows = []
    for s in SESS + ["ALL"]:
        a = alt if s == "ALL" else alt[alt["session"] == s]
        for L in LABELS + ("L3b",):
            cnt = np.bincount(a[L].to_numpy(), minlength=NC)
            drows.append({"oturum": SHORT.get(s, s), "etiket": L, **{f"n{c}": int(cnt[c]) for c in range(NC)}})
    cdist = pd.DataFrame(drows)
    P(pc.md_table(cdist))

    AL = fc.anom_loops()
    cyc = AL[AL["is_cycle"]].set_index(["session", "loop_id"]).index
    lc_rows = []
    for s in SESS:
        a = alt[(alt["session"] == s) & (alt["loop_id"] >= 1)]
        loops_ids = np.sort(a["loop_id"].unique())
        iscyc = np.array([(s, int(l)) in cyc for l in loops_ids])
        for L in LABELS + ("L3b",):
            for b in range(1, 6):
                has = a.groupby("loop_id")[L].apply(lambda v: bool((v == b).any())).reindex(loops_ids).to_numpy()
                lc_rows.append({"oturum": SHORT[s], "etiket": L, "kutu": b, "n_dongu": len(loops_ids),
                                "n_var": int(has.sum()), "oran": float(has.mean()),
                                "n_tamdongu": int(iscyc.sum()),
                                "n_var_tamdongu": int(has[iscyc].sum()) if iscyc.any() else 0,
                                "oran_tamdongu": float(has[iscyc].mean()) if iscyc.any() else float("nan")})
    lc = pd.DataFrame(lc_rows)
    lc.to_csv(OUT / "03_labels_loopcov.csv", index=False, lineterminator="\n")
    P("\n--- en az bir Box5 karesi olan dongu orani (loop_id>=1; tamdongu = ToF-tamam dongu) ---")
    b5 = lc[lc["kutu"] == 5].drop(columns=["kutu"])
    P(pc.md_table(b5, "{:.4f}"))
    P("\n--- ayni oran, Box1..Box4 (kontrol) ---")
    for b in range(1, 5):
        P(f"Box{b}:")
        P(pc.md_table(lc[lc["kutu"] == b].pivot(index="oturum", columns="etiket", values="oran").reset_index(), "{:.4f}"))

    # ---------------- OOF yukle ve alternatif etiketlerle hizala
    key = alt.set_index(["session", "frame_id"])
    oofs = {}
    for mdl in MODELS:
        o = fc.load_oof(mdl, "S1")
        assert len(o) == 16248 and o["fold"].nunique() == 5, (mdl, len(o))
        o = o.merge(alt[["session", "frame_id", "L0", "L1", "L2", "L3", "L3b"]], on=["session", "frame_id"],
                    how="left", validate="1:1")
        assert o[["L0", "L1", "L2", "L3"]].notna().all().all()
        assert (o["L0"].to_numpy() == o["y_true"].to_numpy()).all(), "OOF y_true != L0"
        oofs[mdl] = o
    P(f"\nOOF: {len(oofs['b2'])} satir/model, S1 5 kat; OOF y_true == L0 dogrulandi")

    # ---------------- Faz A tablosuyla capraz kontrol (kat basina ortalama)
    st = pd.read_csv(ROOT / "reports" / "faz_a" / "03_state_table.csv")
    st = st[(st["split"] == "S1") & (st["subset"] == "tum")].set_index("model")
    P("\n--- L0 capraz kontrol: kat-ortalamasi == Faz A 03_state_table.csv (S1, tum) ---")
    chk = []
    for mdl in MODELS:
        o = oofs[mdl]
        per = [fc.state_metrics(g["L0"], g["y_pred"]) for _, g in o.groupby("fold")]
        a_ = float(np.mean([p["acc"] for p in per]))
        f_ = float(np.mean([p["macro_f1"] for p in per]))
        chk.append({"model": MNAME[mdl], "acc_kat_ort": a_, "acc_FazA": float(st.loc[mdl, "acc_mean"]),
                    "mF1_kat_ort": f_, "mF1_FazA": float(st.loc[mdl, "macro_f1_mean"]),
                    "maks_fark": max(abs(a_ - st.loc[mdl, "acc_mean"]), abs(f_ - st.loc[mdl, "macro_f1_mean"]))})
    chk = pd.DataFrame(chk)
    P(pc.md_table(chk, "{:.6f}"))

    # ---------------- oturum x etiket x model metrikleri (fc.state_metrics)
    mrows = []
    for mdl in MODELS:
        o = oofs[mdl]
        for s in SESS + ["ALL"]:
            g = o if s == "ALL" else o[o["session"] == s]
            for L in LABELS + ("L3b",):
                r = fc.state_metrics(g[L].to_numpy(), g["y_pred"].to_numpy())
                # hizli cm-tabanli hesapla esit mi
                r2 = metrics_from_cm(cm_of(g[L].to_numpy().astype(int), g["y_pred"].to_numpy().astype(int)))
                assert abs(r["acc"] - r2["acc"]) < 1e-12 and abs(r["macro_f1"] - r2["macro_f1"]) < 1e-12, (mdl, s, L)
                r["macro_f1_no5"] = r2["macro_f1_no5"]
                present = "".join(str(c) for c in range(NC) if not np.isnan(r[f"f1_{c}"]))
                mrows.append({"model": MNAME[mdl], "oturum": SHORT.get(s, s), "etiket": L, **r, "siniflar": present,
                              "n_pred5": int((g["y_pred"] == 5).sum())})
    M = pd.DataFrame(mrows)
    M.to_csv(OUT / "03_labels_metrics.csv", index=False, lineterminator="\n")
    P("\nfc.state_metrics == hizli cm metrikleri: dogrulandi (acc, macro-F1 < 1e-12)")
    # sinif seti uyarisi
    sets = M.groupby(["oturum", "etiket"])["siniflar"].first().unstack()
    P("\n--- y_true'da mevcut siniflar (macro-F1 bunlar uzerinden) ---")
    P(sets.to_string())

    # ---------------- paired cluster bootstrap (donguler uzerinden) L_k - L0
    P(f"\n--- paired cluster bootstrap: dongu (session,loop_id) yeniden orneklemi, B={B_BOOT}, tohum={SEED_BOOT} ---")
    P("gruplar: tek oturum, ALL (havuz), EX5 (s05 disindaki tum oturumlar: s01-s04, s06); DiD = delta(s05) - delta(EX5) "
      "(s05 ve EX5 donguleri bagimsiz kumeler; ayri cekilisler)")
    rng = np.random.RandomState(SEED_BOOT)
    boot_rows = []
    base = oofs["b2"]
    groups = {SHORT[s]: (base["session"] == s).to_numpy() for s in SESS}
    groups["ALL"] = np.ones(len(base), bool)
    groups["EX5"] = (base["session"] != "session_05_anomaly_fast").to_numpy()
    deltas = {}                                         # (grup, model, etiket, metrik) -> (pt, array B)
    for gname, msk in groups.items():
        gid = (base.loc[msk, "session"].astype(str) + "|" + base.loc[msk, "loop_id"].astype(str)).to_numpy()
        ug, inv = np.unique(gid, return_inverse=True)
        nL = len(ug)
        draws = rng.randint(0, nL, size=(B_BOOT, nL))          # ayni cekilis tum model/etiketler icin (paired)
        for mdl in MODELS:
            o = oofs[mdl]
            yp = o.loc[msk, "y_pred"].to_numpy().astype(int)
            cms = {}
            for Lb in LABELS:
                yt = o.loc[msk, Lb].to_numpy().astype(int)
                cms[Lb] = np.bincount(inv * (NC * NC) + yt * NC + yp, minlength=nL * NC * NC).reshape(nL, NC, NC)
            res = {Lb: {k: np.empty(B_BOOT) for k in ("acc", "macro_f1", "macro_f1_no5")} for Lb in LABELS}
            for Lb in LABELS:
                for b in range(B_BOOT):
                    r = metrics_from_cm(cms[Lb][draws[b]].sum(axis=0))
                    for k in res[Lb]:
                        res[Lb][k][b] = r[k]
            pt0 = metrics_from_cm(cms["L0"].sum(axis=0))
            for Lb in LABELS[1:]:
                ptl = metrics_from_cm(cms[Lb].sum(axis=0))
                for nm in ("acc", "macro_f1", "macro_f1_no5"):
                    dlt = res[Lb][nm] - res["L0"][nm]
                    deltas[(gname, mdl, Lb, nm)] = (ptl[nm] - pt0[nm], dlt)
                    boot_rows.append({"model": MNAME[mdl], "oturum": gname, "etiket": Lb, "metrik": nm,
                                      "delta": ptl[nm] - pt0[nm], "lo95": float(np.percentile(dlt, 2.5)),
                                      "hi95": float(np.percentile(dlt, 97.5)), "n_dongu": nL})
    for mdl in MODELS:
        for Lb in LABELS[1:]:
            for nm in ("acc", "macro_f1", "macro_f1_no5"):
                p5, d5 = deltas[("s05", mdl, Lb, nm)]
                px, dx = deltas[("EX5", mdl, Lb, nm)]
                dd = d5 - dx
                boot_rows.append({"model": MNAME[mdl], "oturum": "DiD(s05-EX5)", "etiket": Lb, "metrik": nm,
                                  "delta": p5 - px, "lo95": float(np.percentile(dd, 2.5)),
                                  "hi95": float(np.percentile(dd, 97.5)), "n_dongu": -1})
    BT = pd.DataFrame(boot_rows)
    BT.to_csv(OUT / "03_labels_bootstrap.csv", index=False, lineterminator="\n")

    # ---------------- tahmin tarafi: s05'te tahmin edilen Box5
    P("\n--- tahmin tarafinda Box5 (s05): modellerin Box5 tahmin ettigi kare ve dongu orani ---")
    prow = []
    for mdl in MODELS:
        o = oofs[mdl]
        for s in SESS:
            g = o[(o["session"] == s) & (o["loop_id"] >= 1)]
            has = g.groupby("loop_id")["y_pred"].apply(lambda v: bool((v == 5).any()))
            prow.append({"model": MNAME[mdl], "oturum": SHORT[s], "n_pred5_kare": int((g["y_pred"] == 5).sum()),
                         "pred5_dongu_orani": float(has.mean()), "n_dongu": len(has),
                         "L0_5_kare": int((g["L0"] == 5).sum()), "L3_5_kare": int((g["L3"] == 5).sum())})
    PR = pd.DataFrame(prow)
    P(pc.md_table(PR, "{:.4f}"))

    # ====================================================================================== rapor
    P("\n--- metrik tablosu (oturum x etiket x model) ---")
    show = M[["model", "oturum", "etiket", "n", "acc", "macro_f1", "macro_f1_no5", "f1_0", "f1_1", "f1_2", "f1_3",
              "f1_4", "f1_5", "siniflar"]]
    P(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    P("\n--- delta (L_k - L0) ve %95 bootstrap GA ---")
    P(BT.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    md = build_md(alt, tdiag, cdist, b5, lc, chk, M, sets, BT, PR)
    (OUT / "03_labels.md").write_text(md, encoding="utf-8")
    P("\n=== RAPOR (reports/faz_a2/03_labels.md) ===")
    P(md)


def summary_md(M, BT, lc, PR, cdist) -> str:
    """Ozet: sayilar tablolardan programatik uretilir (ayni sayilar stdout'ta da basilir)."""
    o = ["## Ozet (sayilar asagidaki tablolardan alinmistir)\n"]
    b5 = lc[(lc["kutu"] == 5) & (lc["oturum"] == "s05")].set_index("etiket")
    c5 = cdist[cdist["oturum"] == "s05"].set_index("etiket")
    o.append("**1. Silme etiket tarafinda.** s05'te en az bir Box5 karesi olan dongu: " +
             ", ".join(f"{l} {int(b5.loc[l, 'n_var'])}/{int(b5.loc[l, 'n_dongu'])} ({b5.loc[l, 'oran']:.3f})"
                       for l in ("L0", "L1", "L2", "L3", "L3b")) +
             ". s05 Box5 kare sayisi: " + ", ".join(f"{l} {int(c5.loc[l, 'n5'])}" for l in ("L0", "L1", "L2", "L3", "L3b")) +
             ". (L1 icin oran yapisal olarak ~1: `loop_id` ham Box5 ile kapaniyor.) L3, L0'daki silmeyi bagimsiz 30 Hz ToF'tan dogruluyor: "
             f"L3'te s05 dongulerinin {int(b5.loc['L3', 'n_var'])}/{int(b5.loc['L3', 'n_dongu'])}'inde Box5 temasi var; L0'da {int(b5.loc['L0', 'n_var'])}'unda kaliyor.\n")

    def g(model, ses, lab, met):
        r = BT[(BT["model"] == model) & (BT["oturum"] == ses) & (BT["etiket"] == lab) & (BT["metrik"] == met)].iloc[0]
        return r["delta"], r["lo95"], r["hi95"]

    def mm(model, ses, lab, col):
        return float(M[(M["model"] == model) & (M["oturum"] == ses) & (M["etiket"] == lab)][col].iloc[0])

    o.append("**2. s05 durum skoru (macro-F1; A = L0 = Faz A, B = L3).**\n")
    rows = []
    for mdl in MODELS:
        m = MNAME[mdl]
        d3, l3_, h3 = g(m, "s05", "L3", "macro_f1")
        dn, ln, hn = g(m, "s05", "L3", "macro_f1_no5")
        dd, ld, hd = g(m, "DiD(s05-EX5)", "L3", "macro_f1")
        dx, lx, hx = g(m, "EX5", "L3", "macro_f1")
        rows.append({"model": m, "A: L0": mm(m, "s05", "L0", "macro_f1"), "B: L3": mm(m, "s05", "L3", "macro_f1"),
                     "B-A [95% GA]": f"{d3:+.4f} [{l3_:+.4f}, {h3:+.4f}]",
                     "L1": mm(m, "s05", "L1", "macro_f1"), "L2": mm(m, "s05", "L2", "macro_f1"),
                     "F1(Box5) L0->L3": f"{mm(m, 's05', 'L0', 'f1_5'):.4f} -> {mm(m, 's05', 'L3', 'f1_5'):.4f}",
                     "no5 fark [95% GA]": f"{dn:+.4f} [{ln:+.4f}, {hn:+.4f}]",
                     "EX5 fark [95% GA]": f"{dx:+.4f} [{lx:+.4f}, {hx:+.4f}]",
                     "DiD [95% GA]": f"{dd:+.4f} [{ld:+.4f}, {hd:+.4f}]"})
    o.append(pc.md_table(pd.DataFrame(rows), "{:.4f}") + "\n")
    o.append("`no5` = sinif 5 disarida macro-F1 farki (Box5 katkisi cikarildiktan sonra); `EX5` = s05 disindaki oturumlarin (s01-s04, s06) havuzunda L3-L0 farki; "
             "`DiD` = s05 farki - EX5 farki (L3'un tum oturumlara getirdigi genel zamanlama/etiket kaymasini cikarir).\n")
    b2 = "B2"
    d3, l3_, h3 = g(b2, "s05", "L3", "macro_f1")
    dd, ld, hd = g(b2, "DiD(s05-EX5)", "L3", "macro_f1")
    dn, ln, hn = g(b2, "s05", "L3", "macro_f1_no5")
    o.append(f"**3. Makale cumlesi (ikincil).** B2'nin s05 macro-F1'i L0 (Faz A) etiketiyle {mm(b2, 's05', 'L0', 'macro_f1'):.4f}, "
             f"30 Hz dokunus etiketi L3 ile {mm(b2, 's05', 'L3', 'macro_f1'):.4f} (fark {d3:+.4f}, %95 GA [{l3_:+.4f}, {h3:+.4f}]; "
             f"{'GA 0 icermiyor' if (l3_ > 0 or h3 < 0) else 'GA 0 iceriyor'}). "
             f"Kazanc sinif 5'ten geliyor: Box5 F1 {mm(b2, 's05', 'L0', 'f1_5'):.4f} -> {mm(b2, 's05', 'L3', 'f1_5'):.4f}; "
             f"Box5 disindaki siniflarin ortalamasi {dn:+.4f} [{ln:+.4f}, {hn:+.4f}] degisiyor. "
             f"s05 farki, diger oturumlardaki L3 farkindan {dd:+.4f} [{ld:+.4f}, {hd:+.4f}] ({'GA 0 icermiyor' if (ld > 0 or hd < 0) else 'GA 0 iceriyor'}) ayrisiyor. "
             f"Ham 10 Hz etiket (L1) ve +-1 mod (L2) ile de artis var ({mm(b2, 's05', 'L1', 'macro_f1'):.4f} / {mm(b2, 's05', 'L2', 'macro_f1'):.4f}); "
             f"B2 icin L1, L2 ve L3'te s05 skoru L0'dan yuksek; ancak L3 (30 Hz, ~3 ornek oylamasi) L1/L2'den dusuk, yani kazancin buyuklugu etiket tanimina duyarli (diger modellerde sira farkli: bkz. tablo 2).\n")
    pr = PR[(PR["oturum"] == "s05")].set_index("model")
    o.append("**4. Tahmin tarafi.** s05'te modellerin Box5 tahmin ettigi kare sayisi (L0'da " + f"{int(pr.loc['B2', 'L0_5_kare'])}" + " Box5 karesi, L3'te " +
             f"{int(pr.loc['B2', 'L3_5_kare'])}" + "; loop_id>=1 kareler): " +
             ", ".join(f"{m} {int(pr.loc[m, 'n_pred5_kare'])} (donguler: {pr.loc[m, 'pred5_dongu_orani']:.3f})" for m in pr.index) +
             ". Modeller L0 ile (silinmis Box5'li s05 egitim pencereleriyle de) egitildigi halde s05'te L0'dan cok daha fazla Box5 tahmin ediyor: "
             "bu, Box5'in diger oturumlardaki etiketlerden ogrenildigiyle tutarlidir; s05'te L0'a gore fazla tahmin edilen Box5 kareleri L0'da yanlis-pozitif, L3'te kismen dogru-pozitif sayilir (Box5 F1 sutunu).\n")
    o.append("**5. Sinirlamalar.** (i) Egitim etiketleri L0'dir; L3 ile yeniden egitim yapilmadi (A2 kapsami disinda). "
             "(ii) L3, L1/L0'dan farkli olarak kare basina ~3 ToF ornegini oyladigi icin kucuk bir zamanlama kaymasi da getirir; bu s05 disindaki oturumlarda "
             "L3'te gorulen kucuk dusus/artislarda gorunur (EX5 ve DiD sutunlari). (iii) macro-F1 yalniz y_true'da bulunan siniflar uzerinden; sinif kumeleri tum etiketlerde "
             "ayni (tablo 'Mevcut siniflar'). (iv) GA: dongu bazli bootstrap, tek operator/tek gun verisi; oturumlar arasi degisim kapsanmiyor.\n")
    return "\n".join(o)


def build_md(alt, tdiag, cdist, b5, lc, chk, M, sets, BT, PR) -> str:
    T = lambda df, ff="{:.4f}": pc.md_table(df, ff)
    L = []
    L.append("# Faz A2.3 - Etiket duyarliligi (s05'te silinen Box5)\n")
    L.append("Brief: `faz_a2/DESIGN_A2.md` A2.3. **Yeniden egitim yok**; mevcut Faz A S1 OOF tahminleri (B2, CNN-GRU, CNN-only, B1) "
             "alternatif dogru etiketlere karsi yeniden skorlandi. Pencere etiketi = son karenin etiketi. "
             "Her sayi `faz_a2/03_labels.py` ciktisinda da basilidir; ham tablolar `03_labels_metrics.csv`, `03_labels_loopcov.csv`, "
             "`03_labels_bootstrap.csv`, `03_labels_frames.csv.gz`.\n")
    L.append("**Statu:** ikincil/sonradan analiz. Birincil makale sayilari Faz A'nin on-kayitli sayilaridir (L0). Ana model secimi degismez (B2 + D3).\n")
    L.append("## Onemli uyari (kapsam)\n"
             "- Egitim etiketleri de ayni silmeyi tasiyordu: Faz A modelleri **L0 etiketleriyle egitildi** (s05'in egitim pencerelerinde de Box5 silinmisti). "
             "Bu analiz yalnizca **degerlendirme tarafindaki** artefakti olcer; L1-L3 ile yeniden egitim yapilmadi. "
             "Bu yuzden L3'te s05 skorunun yukselmemesi, etiket artefaktinin olmadigini kanitlamaz (modeller Box5'i s05'te zaten ogrenmemis olabilir); "
             "yukselmesi de tek basina 'model dogru, etiket yanlis' demek degildir (L3 baska siniflari da degistirir; asagida kontroller var).\n")
    L.append(summary_md(M, BT, lc, PR, cdist))
    L.append("## Etiket tanimlari\n")
    L.append("| etiket | tanim |\n|---|---|\n"
             "| L0 | Faz A `y_true`: ham 10 Hz etiket uzerinde +-3 kare mod filtresi (beraberlik: en kucuk sinif id) |\n"
             "| L1 | ham 10 Hz etiket, yumusatma yok (`y_raw`: karenin merged_df'teki ToF ornegi; argmin(mesafe) < 140 mm) |\n"
             "| L2 | +-1 kare mod filtresi (`fc.mode_smooth(w=1)`, ayni beraberlik kurali) |\n"
             "| L3 | 30 Hz ham ToF 'dokunus': karenin (t-100 ms, t] araligindaki ToF ornekleri, kutu mesafesi < 140 mm sayilir; en cok ornek alan kutu kazanir, "
             "beraberlikte en son aktif olan; hic ornek yoksa/hicbir kutu < 140 mm degilse 0 |\n"
             "| L3b | kontrol: ayni oylama, ToF satirlari `distance_sensors.csv`'deki `frame_id` ile atanir |\n")
    L.append("### ToF -> kare hizalamasi (L3 icin belgelendi)\n"
             "- `merged_df.csv` `00_dataprepare_all.ipynb` ile uretildi: `distance_sensors.csv` satirlari **`frame_id` ile** radar karelerine eslenir; "
             "kare basina `drop_duplicates(subset='frame_id', keep='first')` ile **ilk** ToF satiri tutulur. Yani `box*_active` / `y_raw` (L1) bir karenin "
             "yaklasik 3 ToF ornegindeki **ilk** ornegi yansitir (kare zaman damgasindan ortanca ~86 ms once); diger ~2 ornek atilir.\n"
             "- ToF ve radar zaman damgalari ayni saatten; dt = kare zaman damgasi - ToF zaman damgasi, `dt<0_orani` = 0 (ToF satiri hep kendi karesinden once). "
             "Bir karenin ~3 ToF satiri (~32 ms aralikli) cogunlukla (t-100 ms, t] icine denk gelir, ama hepsi degil: `dt>0.1s_orani` kadar satir, "
             "kendi karesinin zaman damgasindan 100 ms'den daha once (radar kare zamanlamasi titresimi/duraklamalar; oturumlarda 0.15 s'den uzun kare aralari var).\n"
             "- **L3 icin kullanilan atama:** brief'teki zaman penceresi (t-100 ms, t] (t = karenin radar zaman damgasi, `frame_table().t`; "
             "`common.load_tof_raw().t` ile ayni olcek; `searchsorted(side='right')`). **L3b** ayni oylamayi veri setinin kendi `frame_id` atamasiyla yapar. "
             "Iki kural satir duzeyinde `satir_zaman==frame_id` oraninda ayni kareye atar; ama **kare etiketlerinin uyumu** `L3==L3b` cok yuksek, "
             "yani L3 sonuclari atama kuralina duyarsiz. Ana sonuclar L3 ile; L3b CSV'de ve metrik tablolarinda ayri satir.\n"
             "- L3, L1'den farkli olarak kare basina ~3 ornegi oylar; L1 tek (ilk) ornektir. L3'un zamansal merkezi bu yuzden L0/L1'den ~0.5 kare (~50 ms) kadar "
             "farkli olabilir; modeller L0 ile egitildigi icin L3 tum oturumlarda kucuk bir zamanlama uyumsuzlugu da tasir "
             "(s05 disi oturumlar bu yuzden kontrol olarak kullanildi: EX5 ve DiD).\n")
    L.append(T(tdiag) + "\n")
    L.append("- `satir_zaman==frame_id`: ToF satirlarinin zaman kuralina gore ve `frame_id`'ye gore ayni kareye atanma orani. "
             "`L3==L3b`: iki atamayla uretilen kare etiketlerinin uyumu. `beraberlik_L3`: oylamada beraberlik cikan kare sayisi, `cozulmeyen_L3`: 'en son' kuraliyla da cozulmeyen.\n")
    L.append("## Kare sayilari (sinif basina)\n")
    L.append(T(cdist) + "\n")
    L.append("## 'En az bir Box5 karesi olan dongu' orani (oturum x etiket)\n")
    L.append("Dongu = `loop_id >= 1` (merged_df durum makinesi: Box1 ile baslar). `tamdongu` = ToF'a gore tamamlanmis dongu (`anom_loops.is_cycle`). "
             "**Not:** `loop_id` ham (L1) Box5 ile kapandigi icin L1'de oran yapisal olarak ~1'dir (son, kesilmis dongu haric); L1 bu olcut icin bagimsiz kanit degildir.\n")
    L.append(T(b5) + "\n")
    L.append("Box1-Box4 icin ayni oran (kontrol; tum degerler `03_labels_loopcov.csv`'de):\n")
    for b in range(1, 5):
        piv = lc[lc["kutu"] == b].pivot(index="oturum", columns="etiket", values="oran").reset_index()
        L.append(f"Box{b}:\n\n" + T(piv) + "\n")
    L.append("## Durum tahmini: oturum x etiket x model\n")
    L.append("`macro_f1`: `fc.state_metrics` (y_true'da bulunan siniflar uzerinden), `macro_f1_no5`: ayni, ama sinif 5 disarida (Box5 katkisini ayirmak icin). "
             "Havuz = `ALL` (16,248 pencere, tum 6 oturum). Oturum basina OOF, 5 katin birlesimi.\n")
    L.append("### L0 capraz kontrol (Faz A 03_state_table ile)\n")
    L.append(T(chk, "{:.6f}") + "\n")
    L.append("### y_true'da mevcut siniflar\n")
    L.append("```\n" + sets.to_string() + "\n```\n")
    for mdl in MODELS:
        sub = M[M["model"] == MNAME[mdl]]
        L.append(f"### {MNAME[mdl]}\n")
        L.append(T(sub[["oturum", "etiket", "n", "acc", "macro_f1", "macro_f1_no5", "f1_0", "f1_1", "f1_2", "f1_3", "f1_4", "f1_5", "n_pred5"]]) + "\n")
    L.append("## Etiket farki (L_k - L0) ve %95 bootstrap GA\n")
    L.append("Paired cluster bootstrap: (oturum, loop_id) donguleri yerine koyarak yeniden orneklenir (B=%d, tohum=%d); "
             "ayni cekilis tum modeller ve etiketler icin kullanilir. Gruplar: tek oturum, 'ALL' (havuz), 'EX5' (s05 disindaki tum oturumlar), 'DiD(s05-EX5)' = s05 farki - EX5 farki (bagimsiz kume cekilisleri).\n" % (B_BOOT, SEED_BOOT))
    piv = BT.copy()
    piv["delta [95% GA]"] = piv.apply(lambda r: f"{r['delta']:+.4f} [{r['lo95']:+.4f}, {r['hi95']:+.4f}]", axis=1)
    for met in ("macro_f1", "macro_f1_no5", "acc"):
        L.append(f"### {met}\n")
        for mdl in MODELS:
            sub = piv[(piv["metrik"] == met) & (piv["model"] == MNAME[mdl])]
            p = sub.pivot(index="oturum", columns="etiket", values="delta [95% GA]").reset_index()
            L.append(f"**{MNAME[mdl]}**\n\n" + T(p) + "\n")
    L.append("## Tahmin tarafinda Box5 (egitim etiketi silmesinin izi)\n")
    L.append(T(PR) + "\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()

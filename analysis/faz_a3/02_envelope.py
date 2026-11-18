"""Faz A3.2 -- calisma zarfi: sistem neyi goremiyor?  Brief: faz_a3/DESIGN_A3.md (A3.2, on-kayitli).

Yeniden egitim YOK. Faz A S1 OOF tahminleri (B2 birincil; CNN-GRU, CNN-only, B1 ikincil) 30 Hz ToF dokunuslarina karsi okunur.
Faz A 04_cycle_scores / natural-deviant dongu sonuclarina BAKILMAZ (yasak).

Birim: ToF dokunusu = kutu ve oturum basina, 30 Hz'de mesafe < 140 mm olan kesintisiz ornek kosusu (>= 1 esik-ustu ornek boler;
NaN mesafe = esik-ustu).  Sure = t[son] - t[ilk] + medyan ToF dt.
Hizalama (A2.3 ile ayni): ToF ornegi, (t_kare - 100 ms, t_kare] araligina dusen kareye atanir (kare t = frame_table().t).
Bir dokunusun kare kumesi = en az bir ornegini alan kareler.  Kare kumesi BOS olan dokunuslar (ornekleri hicbir karenin
penceresine dusmeyen) icin tespit TANIMSIZ: paydadan dusulur, n_nomap olarak raporlanir.
D-ham: kare kumesinde en az bir karenin ham S1 OOF y_pred'i o kutu.
D-deb: ayni, ama y_pred akisina ac.debounce_labels(MIN_BOX=2, MAX_GAP=1) uygulandiktan sonra (D1/D2'yi yoneten tanim, pred segmentasyonu).

Cikti: reports/faz_a3/02_envelope.md (+ 02_envelope_*.csv, *.png).
"""
from __future__ import annotations

import importlib
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a2"))
import a2_cycles as ac  # noqa: E402  (faz_a/ ve probe/ yollarini ekler; fc'yi acar)

fc = ac.fc
import common as pc  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import statsmodels.api as sm  # noqa: E402

OUT = ROOT / "reports" / "faz_a3"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ("b2", "cnngru", "cnnonly", "b1")
MNAME = {"b2": "B2", "cnngru": "CNN-GRU", "cnnonly": "CNN-only", "b1": "B1"}
DEFS = ("ham", "deb")
DNAME = {"ham": "D-ham", "deb": "D-deb"}
SESS = pc.sessions()
SHORT = {s: "s" + s.split("_")[1] for s in SESS}
EDGES = [0.0, 0.2, 0.3, 0.5, 1.0, np.inf]                       # sol-kapali [lo, hi); son kutu >= 1.0
BINLAB = ["<0.2", "0.2-0.3", "0.3-0.5", "0.5-1.0", ">1.0"]
THR_MM = 140.0
WIN_S = 0.1
Z = 1.96
DX_RES = 0.118
NORMAL_PACE = ("s02", "s03", "s04")                              # "normal tempolu" yorumu (Sapmalar'da kayitli)
COL = {"ham": "#0072B2", "deb": "#D55E00"}                       # Okabe-Ito mavi / turuncu (CVD-guvenli)

MD: list[str] = []


def emit(text: str = "") -> None:
    MD.append(text)
    print(text, flush=True)


def wilson(k: int, n: int):
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return (p, min(p, max(0.0, c - h)), max(p, min(1.0, c + h)))


def fmt_w(k, n):
    if n <= 0:
        return "-"
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {p:.3f} [{lo:.3f}, {hi:.3f}]"


def bin_of(d: float) -> int:
    return int(np.searchsorted(EDGES, d, side="right") - 1)


# ============================================================================================ logistic fit
def fit_logit(x, y) -> dict:
    """logit P(tespit) = b0 + b1 * sure(s).  d50 = -b0/b1, d90 = (ln 9 - b0)/b1.  Duzenlilestirmesiz ML (statsmodels Logit).
    Tahmin edilemez durumlar: n=0, tum sonuclar ayni, tam ayrisma, b1 <= 0, yakinsamama / sonsuz SE.  Aralik [lo50, hi50] =
    d50'nin dayandigi belirsizlik araligi (ok: nokta; ayrisma: [en uzun kacirilan, en kisa yakalanan])."""
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    n, h = len(x), int(y.sum())
    r = dict(n=n, hits=h, status="", b0=np.nan, b1=np.nan, d50=np.nan, d90=np.nan, d50_se=np.nan, d90_se=np.nan,
             lo50=np.nan, hi50=np.nan, extrap=False, xmin=float(x.min()) if n else np.nan, xmax=float(x.max()) if n else np.nan)
    if n == 0:
        r["status"] = "n=0"
        return r
    if h == n:
        r.update(status="tumu tespit (d50 < min sure)", lo50=0.0, hi50=r["xmin"])
        return r
    if h == 0:
        r.update(status="hic tespit yok (d50 > max sure)", lo50=r["xmax"], hi50=np.inf)
        return r
    if x[y == 0].max() < x[y == 1].min():
        r.update(status="tam ayrisma (tahmin edilemez)", lo50=float(x[y == 0].max()), hi50=float(x[y == 1].min()))
        return r
    if x[y == 1].max() < x[y == 0].min():
        r.update(status="ters ayrisma (b1<0, tahmin edilemez)")
        return r
    X = sm.add_constant(x)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(y, X).fit(disp=0, maxiter=200)
        b0, b1 = float(res.params[0]), float(res.params[1])
        cov = np.asarray(res.cov_params())
        ok = bool(res.mle_retvals.get("converged", False)) and np.all(np.isfinite(cov)) and np.all(np.isfinite(res.bse)) and abs(b1) < 1e3
    except Exception as e:  # noqa: BLE001
        r["status"] = f"yakinsamadi ({type(e).__name__})"
        return r
    if not ok:
        r["status"] = "yakinsamadi/yari-ayrisma (tahmin edilemez)"
        return r
    if b1 <= 0:
        r.update(status="b1<=0 (tahmin edilemez)", b0=b0, b1=b1)
        return r
    ln9 = np.log(9.0)
    d50, d90 = -b0 / b1, (ln9 - b0) / b1
    g50 = np.array([-1.0 / b1, b0 / b1**2])                       # d(d50)/d(b0,b1)
    g90 = np.array([-1.0 / b1, -(ln9 - b0) / b1**2])
    se50, se90 = float(np.sqrt(g50 @ cov @ g50)), float(np.sqrt(g90 @ cov @ g90))
    ext = bool((d50 < r["xmin"]) or (d50 > r["xmax"]) or (d90 < r["xmin"]) or (d90 > r["xmax"]))
    r.update(status="ok", b0=b0, b1=b1, d50=d50, d90=d90, d50_se=se50, d90_se=se90, lo50=d50, hi50=d50, extrap=ext)
    # d50 gozlenen sure araliginin disindaysa nokta degeri anlamsizdir (ornegin negatif sure): yalniz sinir olarak okunur
    if d50 < r["xmin"]:
        r.update(lo50=0.0, hi50=r["xmin"])
    elif d50 > r["xmax"]:
        r.update(lo50=r["xmax"], hi50=np.inf)
    return r


def verdict_interval(lo, hi, a, b) -> str:
    if not (np.isfinite(lo) or np.isfinite(hi)):
        return "belirsiz (aralik yok)"
    lo_ = lo if np.isfinite(lo) else -np.inf
    if lo_ >= a and hi <= b:
        return "TUTAR"
    if hi < a or lo_ > b:
        return "TUTMAZ"
    return "belirsiz"


# ============================================================================================ veri yukleme
def main():
    emit("# Faz A3.2 - Calisma zarfi: sistem neyi goremiyor?")
    emit()
    emit("Brief: `faz_a3/DESIGN_A3.md` (A3.2; on-kayitli). **Yeniden egitim yok.** Statu: ikincil/sonradan analiz; birincil sonuclar Faz A'nin on-kayitli sayilaridir "
         "(ana model B2 + D3 degismez). Her sayi `faz_a3/02_envelope.py` ciktisinda da basilidir. Faz A 04 dongu skorlarina / dogal sapma dongulerine bakilmadi.")
    emit()
    ft = fc.frame_table()
    assert len(ft) == 16272, len(ft)

    # ---------- OOF akislari (s01 dahil 6 oturum) ----------
    RAW, DEB = {}, {}
    oofs = {}
    for m in MODELS:
        oof = fc.load_oof(m, "S1")
        assert len(oof) == 16248 and oof["fold"].nunique() == 5, (m, len(oof))
        mm = ft[["session", "frame_id"]].merge(oof[["session", "frame_id", "y_pred"]], on=["session", "frame_id"], how="left", validate="1:1")
        assert int(mm["y_pred"].notna().sum()) == 16248
        y = mm["y_pred"].fillna(0).astype(int).to_numpy()
        ses = ft["session"].to_numpy()
        RAW[m] = {s: y[ses == s] for s in SESS}
        DEB[m] = {s: ac.debounce_labels(RAW[m][s], ac.MIN_BOX, ac.MAX_GAP) for s in SESS}
        oofs[m] = oof
    # capraz kontrol: s02-s06 icin modulun kendi akislariyla ayni mi
    ref = ac.source_labels(sources=MODELS, verbose=False)
    eq = True
    for m in MODELS:
        for s in ac.ctx().SESS:
            eq &= bool((RAW[m][s] == ref[m][s]).all()) and bool((DEB[m][s] == ac.debounce_labels(ref[m][s], ac.MIN_BOX, ac.MAX_GAP)).all())
    emit(f"Capraz kontrol (s02-s06 y_pred akisi == `ac.source_labels`, debounce == `ac.debounce_labels`): **equal: {eq}**")
    assert eq
    emit(f"- Kare tablosu {len(ft)} kare (6 oturum); OOF'lar 16,248 pencere / 5 kat (dort model icin dogrulandi); oturum basina ilk 4 kare penceresiz -> y_pred = none (0) (A2 ile ayni).")
    emit("- s01 (egitim oturumu) de kapsamdadir (brief: kisa dokunuslar s05 Box5 ve s01 Box4'te); `labels_from_oof` s01'i dusurdugu icin s01 akisi bu betikte elle kuruldu, ayni `debounce_labels` ile.")
    emit()

    frames_alt = pd.read_csv(ROOT / "reports" / "faz_a2" / "03_labels_frames.csv.gz")
    fa = ft[["session", "frame_id"]].merge(frames_alt[["session", "frame_id", "L0", "L3"]], on=["session", "frame_id"], how="left", validate="1:1")
    assert fa["L3"].notna().all()
    L3 = {s: fa.loc[(ft["session"] == s).to_numpy(), "L3"].astype(int).to_numpy() for s in SESS}
    L0 = {s: ft.loc[(ft["session"] == s).to_numpy(), "y_true"].astype(int).to_numpy() for s in SESS}
    assert all((fa.loc[(ft["session"] == s).to_numpy(), "L0"].to_numpy() == L0[s]).all() for s in SESS)

    # ---------- dokunuslar ----------
    rows = []
    gap_warn = 0
    for s in SESS:
        d = pc.load_tof_raw(s)
        tt = d["t"].to_numpy()
        dist = d[pc.DIST_COLS].to_numpy(float)
        act = np.where(np.isnan(dist), False, dist < THR_MM)
        dtm = float(np.median(np.diff(tt)))
        fs = ft[ft["session"] == s]
        tf = fs["t"].to_numpy()
        assert (np.diff(tf) >= 0).all()
        pos = np.searchsorted(tf, tt, side="left")                 # ilk kare, t_kare >= t_tof
        posc = np.minimum(pos, len(tf) - 1)
        inwin = (pos < len(tf)) & ((tf[posc] - tt) < WIN_S - 1e-12)
        fof = np.where(inwin, posc, -1)
        for b in range(5):
            a = act[:, b].astype(np.int8)
            ch = np.flatnonzero(np.diff(np.r_[0, a, 0]))
            for i0, i1e in zip(ch[0::2], ch[1::2]):
                i1 = i1e - 1
                dur = float(tt[i1] - tt[i0] + dtm)
                gaps = np.diff(tt[i0:i1 + 1])
                mg = float(gaps.max()) if len(gaps) else 0.0
                gap_warn += int(mg > 0.2)
                fr = np.unique(fof[i0:i1 + 1])
                fr = fr[fr >= 0]
                rec = dict(session=SHORT[s], box=b + 1, t_start=float(tt[i0]), t_end=float(tt[i1] + dtm), dur_s=dur,
                           n_samples=int(i1 - i0 + 1), max_internal_gap_s=mg, n_frames=int(len(fr)), bin=BINLAB[bin_of(dur)])
                rec["L0_vis"] = bool((L0[s][fr] == b + 1).any()) if len(fr) else np.nan
                rec["L3_vis"] = bool((L3[s][fr] == b + 1).any()) if len(fr) else np.nan
                for m in MODELS:
                    rec[f"{m}_ham"] = bool((RAW[m][s][fr] == b + 1).any()) if len(fr) else np.nan
                    rec[f"{m}_deb"] = bool((DEB[m][s][fr] == b + 1).any()) if len(fr) else np.nan
                rows.append(rec)
    T = pd.DataFrame(rows)
    T["nomap"] = T["n_frames"] == 0
    T.to_csv(OUT / "02_envelope_touches.csv.gz", index=False, compression="gzip", lineterminator="\n")
    emit("## 0. Tanimlar ve hizalama (ozet; on-kayitli tanimlar degismedi)")
    emit()
    emit("- **Dokunus:** kutu ve oturum basina, ham 30 Hz ToF'ta `mesafe < 140 mm` olan kesintisiz ornek kosusu; >= 1 esik-ustu (ya da NaN) ornek dokunusu boler. "
         "**Sure** = t[son ornek] - t[ilk ornek] + oturumun medyan ToF dt'si (tek orneklik dokunus ~1 dt; sure tanimi brifte yoktu, bkz. Sapmalar).")
    emit(f"- **ToF -> kare hizalamasi (A2.3 `reports/faz_a2/03_labels.md` ile ayni):** ToF ornegi, `(t_kare - 100 ms, t_kare]` araligina dusen kareye atanir (`t` = `frame_table().t`; "
         f"ToF `t` ayni saat; `searchsorted(side='left')` + `t_kare - t_tof < 0.1 s`). Bir dokunusun *kare kumesi* = en az bir orneginin atandigi kareler. "
         f"`L3` etiketindeki `(t-100ms, t]` kuraliyla ayni; `frame_id` atamasi (`merged_df`) kullanilmadi.")
    emit("- **D-ham:** kare kumesinde en az bir karenin (= pencerenin son karesi) ham S1 OOF `y_pred`'i o kutu. "
         "**D-deb:** ayni, `debounce_labels(MIN_BOX=2, MAX_GAP=1)` sonrasi akista (**D1/D2'yi yoneten tanim, pred segmentasyonu**).")
    emit("- **Kare kumesi bos dokunus (n_nomap):** ornekleri hicbir karenin pencerenin icine dusmeyen dokunuslar (radar kare aralarindaki bosluklar; A2.3: ToF satirlarinin %7-26'si kendi karesinden >100 ms once - ayri satir olarak degil, dokunus duzeyinde sayilir; bu veride sayi asagida). "
         "Bunlar icin tespit **tanimsizdir** (kacirma degil); payda disi birakilir ve her tabloda ayri raporlanir.")
    emit(f"- Sure kutulari (sol-kapali `[lo, hi)`; son kutu >= 1.0 s): {', '.join(BINLAB)} s. Wilson %95. Dokunus ici >0.2 s ornek-arasi bosluklu dokunus sayisi: {gap_warn} (bilgi).")
    emit(f"- Toplam dokunus: {len(T)} (oturum x kutu); kare kumesi bos: {int(T['nomap'].sum())}.")
    emit()

    # ---------- sayim ----------
    emit("## 1. Dokunus sayimi (oturum x kutu x sure kutusu; kare kumesi bos olanlar `(+n)` ile)")
    emit()
    cens = []
    for s in SESS:
        for b in range(1, 6):
            g = T[(T["session"] == SHORT[s]) & (T["box"] == b)]
            if len(g) == 0:
                continue
            rec = {"oturum": SHORT[s], "kutu": b, "n": len(g), "medyan_sure_s": float(g["dur_s"].median()), "min_s": float(g["dur_s"].min()), "max_s": float(g["dur_s"].max())}
            for bl in BINLAB:
                gg = g[g["bin"] == bl]
                rec[bl] = f"{int((~gg['nomap']).sum())}" + (f" (+{int(gg['nomap'].sum())})" if gg["nomap"].any() else "")
            cens.append(rec)
    emit(pc.md_table(pd.DataFrame(cens), "{:.3f}"))
    emit()
    pd.DataFrame(cens).to_csv(OUT / "02_envelope_census.csv", index=False, lineterminator="\n")

    # ---------- tespit tablolari ----------
    DET = []
    FITS = []
    for m in MODELS:
        for s in SESS:
            g = T[T["session"] == SHORT[s]]
            for dn in DEFS:
                col = f"{m}_{dn}"
                for bi, bl in enumerate(BINLAB):
                    gg = g[g["bin"] == bl]
                    ev = gg[~gg["nomap"]]
                    k, n = int(ev[col].astype(bool).sum()), len(ev)
                    p, lo, hi = wilson(k, n)
                    DET.append(dict(model=MNAME[m], session=SHORT[s], tanim=DNAME[dn], bin=bl, n_touch=len(gg), n_nomap=int(gg["nomap"].sum()),
                                    n_eval=n, k_detect=k, rate=p, wilson_lo=lo, wilson_hi=hi))
                ev = g[~g["nomap"]]
                f = fit_logit(ev["dur_s"].to_numpy(), ev[col].astype(int).to_numpy())
                f.update(model=MNAME[m], session=SHORT[s], tanim=DNAME[dn])
                FITS.append(f)
    DETdf = pd.DataFrame(DET)
    FITdf = pd.DataFrame(FITS)
    DETdf.to_csv(OUT / "02_envelope_detection.csv", index=False, lineterminator="\n")
    FITdf.to_csv(OUT / "02_envelope_fits.csv", index=False, lineterminator="\n")

    def det_table(model: str, dn: str) -> pd.DataFrame:
        r = []
        for s in SESS:
            rec = {"oturum": SHORT[s]}
            for bl in BINLAB:
                x = DETdf[(DETdf["model"] == MNAME[model]) & (DETdf["session"] == SHORT[s]) & (DETdf["tanim"] == DNAME[dn]) & (DETdf["bin"] == bl)].iloc[0]
                rec[bl] = fmt_w(int(x.k_detect), int(x.n_eval)) + (f" (+{int(x.n_nomap)} bos)" if x.n_nomap else "")
            r.append(rec)
        return pd.DataFrame(r)

    emit("## 2. Tespit orani ~ sure kutusu (B2, birincil; Wilson %95; `k/n = oran [alt, ust]`; oturum basina, havuzlanmadi)")
    emit()
    for dn in DEFS:
        emit(f"### B2 - {DNAME[dn]}" + (" (D1/D2'yi yoneten tanim)" if dn == "deb" else ""))
        emit()
        emit(pc.md_table(det_table("b2", dn)))
        emit()

    # ---------- lojistik ----------
    def fit_row(f):
        return {"model": f["model"], "oturum": f["session"], "tanim": f["tanim"], "n": f["n"], "yakalanan": f["hits"],
                "durum": f["status"], "b0": f["b0"], "b1": f["b1"], "d50_s": f["d50"], "d50_SE": f["d50_se"], "d90_s": f["d90"], "d90_SE": f["d90_se"],
                "d50_araligi": ("" if not np.isfinite(f["lo50"]) and not np.isfinite(f["hi50"]) else f"[{f['lo50']:.3f}, {f['hi50']:.3f}]"),
                "ekstrapolasyon": ("EVET" if f["extrap"] else ""), "sure_aralik_s": f"[{f['xmin']:.3f}, {f['xmax']:.3f}]" if f["n"] else ""}

    emit("## 3. Lojistik uydurma: logit P(tespit) = b0 + b1 * sure(s); d50 = -b0/b1, d90 = (ln 9 - b0)/b1")
    emit()
    emit("Duzenlilestirmesiz ML (`statsmodels.Logit`), oturum x model x tanim basina, kare kumesi bos olmayan dokunuslar uzerinde, **sure dogrusal saniye olarak**. SE: delta yontemi (yaklasik). "
         "**Tahmin edilemez** durumlar: n=0; tum sonuclar ayni; tam ayrisma; b1<=0; yakinsamama/sonsuz SE. Tam ayrismada d50 icin yalniz **aralik** verilir: [en uzun kacirilan, en kisa yakalanan]; "
         "'tumu tespit' durumunda d50 < min sure. `ekstrapolasyon` = d50 ya da d90 uydurulan surelerin [min, max] disinda.")
    emit()
    for m in MODELS:
        for dn in DEFS:
            emit(f"### {MNAME[m]} - {DNAME[dn]}")
            emit()
            sub = FITdf[(FITdf["model"] == MNAME[m]) & (FITdf["tanim"] == DNAME[dn])]
            emit(pc.md_table(pd.DataFrame([fit_row(r) for r in sub.to_dict("records")]), "{:.3f}"))
            emit()

    # ---------- on-yazili tahminler ----------
    emit("Not: `ekstrapolasyon = EVET` iken d50/d90 gozlenen sure araliginin disindadir (orn. negatif sure); bu satirlar yalniz **sinir** olarak okunur (d50 < min sure ya da > max sure; Bolum 4'te aralik olarak kullanilir), nokta tahmini olarak degil. "
         "Dogrusal-saniye lojistik, kisa dokunuslarda plato yapan oranlari (orn. s01'de <0.3 s kutularinda ~0.6) iyi yakalamaz; bu yuzden kutu orani tablolari (Bolum 2) esas, uydurma ozet olarak okunmalidir.")
    emit()
    emit("## 4. On-yazili tahminler ve sabit kural (B2; kurallar kelimesi kelimesine)")
    emit()
    fb = {(r["session"], r["tanim"]): r for r in FITdf[FITdf["model"] == "B2"].to_dict("records")}
    V = []
    emit("### (a) B2'de d50 in [0.2, 0.4] s (D-ham) - oturum basina")
    emit()
    rows_a = []
    for s in SESS:
        f = fb[(SHORT[s], "D-ham")]
        v = verdict_interval(f["lo50"], f["hi50"], 0.2, 0.4)
        rows_a.append({"oturum": SHORT[s], "n": f["n"], "durum": f["status"], "d50_ham_s": f["d50"], "d50_araligi": f"[{f['lo50']:.3f}, {f['hi50']:.3f}]" if np.isfinite(f["lo50"]) else "", "[0.2, 0.4] icinde?": v})
        V.append(("a", SHORT[s], v))
    emit(pc.md_table(pd.DataFrame(rows_a), "{:.3f}"))
    na = [r for r in rows_a]
    emit()
    cnt_a = {k: sum(1 for r in na if r["[0.2, 0.4] icinde?"] == k) for k in ("TUTAR", "TUTMAZ", "belirsiz")}
    est = [r for r in na if r["durum"] == "ok"]
    allok = all(r["[0.2, 0.4] icinde?"] == "TUTAR" for r in na)
    emit(f"Ozet (a): TUTAR {cnt_a['TUTAR']}, TUTMAZ {cnt_a['TUTMAZ']}, belirsiz/aralik-yok {len(na) - cnt_a['TUTAR'] - cnt_a['TUTMAZ']} (6 oturumdan); d50 nokta olarak tahmin edilebilen oturum: {len(est)}. "
         f"**Tahmin (a) tum oturumlarda TUTUYOR mu: {'EVET' if allok else 'HAYIR'}.**")
    emit()

    emit("### (b) Normal tempolu oturumlarda > 0.5 s dokunuslar icin tespit >= 0.9 (B2; D-ham; verdikt kumesi: s02, s03, s04; s01/s05/s06 bilgi)")
    emit()
    rows_b = []
    okb = []
    for s in SESS:
        g = T[(T["session"] == SHORT[s]) & (~T["nomap"])]
        rec = {"oturum": SHORT[s], "normal tempo (verdikt kumesi)": "EVET" if SHORT[s] in NORMAL_PACE else "hayir"}
        for lab, sel in (("0.5-1.0", g["bin"] == "0.5-1.0"), (">1.0", g["bin"] == ">1.0"), (">=0.5 (birlesik)", g["dur_s"] >= 0.5)):
            for dn in DEFS:
                gg = g[sel]
                rec[f"{lab} {DNAME[dn]}"] = fmt_w(int(gg[f"b2_{dn}"].astype(bool).sum()), len(gg))
        gg = g[g["dur_s"] >= 0.5]
        k, n = int(gg["b2_ham"].astype(bool).sum()), len(gg)
        rate = k / n if n else np.nan
        rec["D-ham >=0.9?"] = "EVET" if n and rate >= 0.9 else ("HAYIR" if n else "n=0")
        if SHORT[s] in NORMAL_PACE:
            okb.append(bool(n and rate >= 0.9))
        rows_b.append(rec)
    emit(pc.md_table(pd.DataFrame(rows_b)))
    emit()
    emit(f"**Tahmin (b) (s02, s03, s04'te D-ham >= 0.9, nokta tahmin): {'TUTUYOR' if all(okb) else 'TUTMUYOR'}** ({sum(okb)}/3 oturum). "
         f"s01 (egitim oturumu, rastgele toplama) dahil edilirse: {'TUTUYOR' if all(okb) and rows_b[0]['D-ham >=0.9?'] == 'EVET' else 'TUTMUYOR'}.")
    emit()

    emit("### (c) D-deb'de d50 D-ham'dan yuksek (B2; oturum basina; aralik karsilastirmasi)")
    emit()
    rows_c = []
    for s in SESS:
        fh, fd = fb[(SHORT[s], "D-ham")], fb[(SHORT[s], "D-deb")]
        if all(np.isfinite([fh["lo50"], fh["hi50"], fd["lo50"], fd["hi50"]])) or (np.isfinite(fh["lo50"]) and np.isfinite(fd["lo50"])):
            hh_lo, hh_hi, dd_lo, dd_hi = fh["lo50"], fh["hi50"], fd["lo50"], fd["hi50"]
            if dd_lo > hh_hi:
                v = "EVET (yuksek)"
            elif dd_hi < hh_lo:
                v = "HAYIR (dusuk)"
            elif dd_lo == dd_hi == hh_lo == hh_hi:
                v = "esit"
            else:
                v = "belirsiz (araliklar ortusuyor)"
        else:
            v = "belirsiz (aralik yok)"
        rows_c.append({"oturum": SHORT[s], "D-ham durum": fh["status"], "d50_ham": fh["d50"], "D-deb durum": fd["status"], "d50_deb": fd["d50"],
                       "D-ham araligi": f"[{fh['lo50']:.3f}, {fh['hi50']:.3f}]" if np.isfinite(fh["lo50"]) else "",
                       "D-deb araligi": f"[{fd['lo50']:.3f}, {fd['hi50']:.3f}]" if np.isfinite(fd["lo50"]) else "", "d50_deb > d50_ham?": v})
    emit(pc.md_table(pd.DataFrame(rows_c), "{:.3f}"))
    emit()
    ce = sum(1 for r in rows_c if r["d50_deb > d50_ham?"].startswith("EVET"))
    ch = sum(1 for r in rows_c if r["d50_deb > d50_ham?"].startswith("HAYIR"))
    emit(f"Ozet (c): EVET {ce}, HAYIR {ch}, belirsiz/esit {6 - ce - ch} (6 oturumdan). **Tahmin (c) tum oturumlarda EVET: {'EVET' if ce == 6 else 'HAYIR'}.**")
    emit()

    emit("### (d) Kural: d50 > 0.4 s ise beyan edilen sinir gozlenen d90'a yukseltilir (yeniden ayar yok)")
    emit()
    emit("Yorum (Sapmalar'da kayitli): baslangic beyan siniri = on-yazili d50 araliginin ust ucu, 0.4 s; kural tetiklenirse (d50 > 0.4 s; aralik varsa alt ucun 0.4'u asmasi) sinir = o oturumun gozlenen d90'i; "
         "tam ayrismada d90 tanimsiz oldugundan sinir = aralik ust ucu (en kisa yakalanan dokunus) olarak, acikca etiketli. Tetiklenmezse sinir 0.4 s olarak kalir; ayrica d90 her durumda gosterilir.")
    emit()
    rows_d = []
    for s in SESS:
        for dn in ("D-ham", "D-deb"):
            f = fb[(SHORT[s], dn)]
            lo, hi = f["lo50"], f["hi50"]
            if f["status"] == "ok" and not (f["d50"] < f["xmin"] or f["d50"] > f["xmax"]):
                trig = "EVET" if f["d50"] > 0.4 else "hayir"
            elif np.isfinite(lo) and lo > 0.4:
                trig = "EVET (araliktan)"
            elif np.isfinite(hi) and hi <= 0.4:
                trig = "hayir (araliktan)"
            else:
                trig = "belirsiz"
            if trig.startswith("EVET"):
                lim = f"{f['d90']:.3f} (d90{'; ekstrapolasyon' if f['extrap'] else ''})" if f["status"] == "ok" else (f"{hi:.3f} (aralik ust ucu; d90 tanimsiz)" if np.isfinite(hi) else "tanimsiz")
            elif trig.startswith("hayir"):
                lim = "0.400 (kural tetiklenmedi)"
            else:
                lim = "belirsiz"
            rows_d.append({"oturum": SHORT[s], "tanim": dn, "durum": f["status"], "d50_s": f["d50"], "d90_s": f["d90"], "d50>0.4?": trig, "beyan edilen sinir_s": lim})
    emit(pc.md_table(pd.DataFrame(rows_d), "{:.3f}"))
    emit()
    pd.DataFrame(rows_d).to_csv(OUT / "02_envelope_declared_limits.csv", index=False, lineterminator="\n")
    ext_map = {(r["session"], r["tanim"]): r["extrap"] for r in FITdf[FITdf["model"] == "B2"].to_dict("records")}
    d90s = [(r["oturum"], r["tanim"], r["d90_s"]) for r in rows_d if np.isfinite(r["d90_s"]) and not ext_map[(r["oturum"], r["tanim"])]]
    if d90s:
        emit("Betimleyici ozet (on-kayitli degil, havuzlama degil; yalniz gozlenen sure araliginda kalan, ekstrapolasyonsuz d90 degerleri) = " + "; ".join(f"{a} {b}: {c:.3f} s" for a, b, c in d90s) + f"; en buyuk = {max(x[2] for x in d90s):.3f} s.")
    else:
        emit("Betimleyici ozet: hicbir oturumda d90 nokta olarak tahmin edilemedi.")
    emit()
    pd.DataFrame(V, columns=["pred", "oturum", "verdikt"]).to_csv(OUT / "02_envelope_verdicts_a.csv", index=False, lineterminator="\n")

    # ---------- ikincil modeller ----------
    emit("## 5. Ikincil modeller (CNN-GRU, CNN-only, B1): tespit orani (oturum x kutu; Wilson)")
    emit()
    for m in ("cnngru", "cnnonly", "b1"):
        for dn in DEFS:
            emit(f"### {MNAME[m]} - {DNAME[dn]}")
            emit()
            emit(pc.md_table(det_table(m, dn)))
            emit()

    # ---------- L0'da gorunmezlik ----------
    emit("## 6. Egitim-etiketi uyarisinin sayisi: dokunus egitim/degerlendirme etiketinde (L0) ve 30 Hz oylamada (L3) gorunuyor mu?")
    emit()
    emit("`L0` = Faz A `y_true` (modellerin egitildigi etiket; +-3 kare mod), `L3` = 30 Hz oylama etiketi. Bir dokunus *L0'da gorunur* = kare kumesinde en az bir karenin L0 etiketi o kutu. "
         "Kare kumesi bos dokunuslar disarida. Hucre: gorunur/n (oran).")
    emit()
    rows6 = []
    for s in SESS:
        g = T[(T["session"] == SHORT[s]) & (~T["nomap"])]
        for bl in BINLAB:
            gg = g[g["bin"] == bl]
            if len(gg) == 0:
                continue
            n = len(gg)
            rows6.append({"oturum": SHORT[s], "bin": bl, "n": n, "L0'da gorunur": f"{int(gg['L0_vis'].astype(bool).sum())}/{n} ({gg['L0_vis'].astype(bool).mean():.3f})",
                          "L3'te gorunur": f"{int(gg['L3_vis'].astype(bool).sum())}/{n} ({gg['L3_vis'].astype(bool).mean():.3f})",
                          "B2 D-ham": f"{int(gg['b2_ham'].astype(bool).sum())}/{n} ({gg['b2_ham'].astype(bool).mean():.3f})"})
    emit(pc.md_table(pd.DataFrame(rows6)))
    emit()
    pd.DataFrame(rows6).to_csv(OUT / "02_envelope_label_visibility.csv", index=False, lineterminator="\n")
    gl = T[~T["nomap"]]
    for bl in BINLAB:
        gg = gl[gl["bin"] == bl]
        if len(gg):
            emit(f"- Tum oturumlar (yalniz bilgi, havuzlanmis ham oran): {bl}: n={len(gg)}, L0'da gorunur {gg['L0_vis'].astype(bool).mean():.3f}, L3'te gorunur {gg['L3_vis'].astype(bool).mean():.3f}, B2 D-ham {gg['b2_ham'].astype(bool).mean():.3f}.")
    emit()

    # ---------- mekansal eksen ----------
    emit("## 7. Konum ekseni (yalniz betimleyici): komsu kutu cifti ikili karisikligi vs kutu araligi")
    emit()
    q1 = importlib.import_module("01_dwell_modes")
    exp_sp = {}
    for ln in (ROOT / "reports" / "probe" / "q1.md").read_text(encoding="utf-8").splitlines():
        mt = re.match(r"\|\s*(s0\d)\s*\|\s*(\d-\d)\s*\|\s*([0-9.]+)\s*\|", ln)
        if mt:
            exp_sp[(mt.group(1), mt.group(2))] = float(mt.group(3))
    pairs = pd.concat([q1.adjacent_pairs(q1.box_centroids(pc.load_frames(s)), s) for s in SESS], ignore_index=True)
    assert len(pairs) == 14, len(pairs)
    mism = [(r.session, r.pair) for r in pairs.itertuples() if abs(round(r.spacing_m, 3) - exp_sp.get((r.session, r.pair), -9)) > 1e-9]
    emit(f"Kutu araligi = radar kare-merkezi (cx) medyan farki, `probe/01_dwell_modes.py` `adjacent_pairs` ile **ayni yontemle yeniden hesaplandi** (komsu = oturumda gorulen kutular arasinda cx sirasinda komsu; "
         f"cift sayisi {len(pairs)}); `reports/probe/q1.md` degerleriyle uyum (3 ondalik): **{'TAMAM, 14/14' if not mism else f'UYUMSUZ {mism}'}**. Bu aralik bant boyunca radar-merkez ayrimidir, mezura ile olculmus mesafe degildir.")
    emit(f"Karisiklik: `b2__S1` OOF'ta `y_true in {{a, b}}` satirlari; `y_pred` ucuncu bir sinif (none dahil) olan satirlar **disarida**; "
         f"hata a->b = y_true=a, y_pred=b orani (kalan a satirlari icinde), b->a benzer; `sim_hata` = (a->b + b->a) / (kalan a + kalan b). Dogru etiket = OOF `y_true` (L0, modelin egitildigi etiket). "
         f"Uydurulmus bir tasarim sabiti yoktur; cizgi = lateral cozunurluk {DX_RES * 100:.1f} cm (r*dtheta = 0.45 m x 15 derece).")
    emit()
    ob = oofs["b2"]
    srows = []
    for r in pairs.itertuples():
        s = next(x for x in SESS if SHORT[x] == r.session)
        a, b = (int(v) for v in r.pair.split("-"))
        g = ob[ob["session"] == s]
        ga, gb = g[g["y_true"] == a], g[g["y_true"] == b]
        ka, kb = ga[ga["y_pred"].isin([a, b])], gb[gb["y_pred"].isin([a, b])]
        a2b = int((ka["y_pred"] == b).sum())
        b2a = int((kb["y_pred"] == a).sum())
        srows.append(dict(oturum=r.session, cift=r.pair, aralik_cm=r.spacing_m * 100, aralik_lt_11_8cm=bool(r.spacing_m < DX_RES), n_a=len(ga), n_b=len(gb),
                          n_a_kalan=len(ka), n_b_kalan=len(kb), disarida_3_sinif=(len(ga) - len(ka)) + (len(gb) - len(kb)),
                          a_to_b=a2b, b_to_a=b2a, hata_a_to_b=a2b / len(ka) if len(ka) else np.nan, hata_b_to_a=b2a / len(kb) if len(kb) else np.nan,
                          sim_hata=(a2b + b2a) / (len(ka) + len(kb)) if (len(ka) + len(kb)) else np.nan, sep_ratio=r.sep_ratio))
    SP = pd.DataFrame(srows)
    SP.to_csv(OUT / "02_envelope_spatial.csv", index=False, lineterminator="\n")
    emit(pc.md_table(SP, "{:.4f}"))
    emit()
    sub = SP[SP["aralik_lt_11_8cm"]]
    sup = SP[~SP["aralik_lt_11_8cm"]]
    emit(f"Betimleyici: aralik < 11.8 cm olan {len(sub)} cift: sim_hata araligi [{sub['sim_hata'].min():.4f}, {sub['sim_hata'].max():.4f}], medyan {sub['sim_hata'].median():.4f}; "
         f"aralik >= 11.8 cm olan {len(sup)} cift: [{sup['sim_hata'].min():.4f}, {sup['sim_hata'].max():.4f}], medyan {sup['sim_hata'].median():.4f}. (Fit yok; n=14 cift, cogu tek oturum; yalniz betimleyici.)")
    s45 = SP[(SP["oturum"] == "s04") & (SP["cift"] == "4-5")].iloc[0]
    emit(f"**s04 Box4-Box5 (ayri): aralik {s45.aralik_cm:.1f} cm; n_4={int(s45.n_a)}, n_5={int(s45.n_b)}; 4->5 hata {s45.hata_a_to_b:.4f} ({int(s45.a_to_b)}/{int(s45.n_a_kalan)}), 5->4 hata {s45.hata_b_to_a:.4f} ({int(s45.b_to_a)}/{int(s45.n_b_kalan)}); sim_hata {s45.sim_hata:.4f}; ucuncu sinif nedeniyle disarida {int(s45.disarida_3_sinif)} satir.**")
    emit()

    # ---------- sekiller ----------
    plot_detection(T, FITdf)
    plot_spatial(SP)
    emit("Sekiller: `02_envelope_detection.png` (oturum basina tespit ~ sure: B2, D-ham ve D-deb, Wilson %95, lojistik egri), `02_envelope_spatial.png` (komsu cift hatasi vs kutu araligi, 11.8 cm cizgisi).")
    emit()

    # ---------- ozet ----------
    emit("## 7b. Ozet (yalniz yukaridaki tablolardan; yorum degil sayim)")
    emit()
    for dn in DEFS:
        for bl in ("0.5-1.0", ">1.0"):
            sub_ = DETdf[(DETdf["model"] == "B2") & (DETdf["tanim"] == DNAME[dn]) & (DETdf["bin"] == bl) & (DETdf["n_eval"] > 0)]
            w = sub_.loc[sub_["rate"].idxmin()]
            emit(f"- B2 {DNAME[dn]}, {bl} s kutusu: oturumlar arasi en dusuk tespit {w['rate']:.3f} ({int(w['k_detect'])}/{int(w['n_eval'])}, {w['session']}; Wilson alt {w['wilson_lo']:.3f}); tum oturumlarda >= 0.9: {bool((sub_['rate'] >= 0.9).all())}.")
    for dn in DEFS:
        for bl in ("<0.2", "0.2-0.3", "0.3-0.5"):
            sub_ = DETdf[(DETdf["model"] == "B2") & (DETdf["tanim"] == DNAME[dn]) & (DETdf["bin"] == bl) & (DETdf["n_eval"] >= 8)]
            if len(sub_):
                emit(f"- B2 {DNAME[dn]}, {bl} s kutusu (n >= 8 olan oturumlar): " + "; ".join(f"{r.session} {int(r.k_detect)}/{int(r.n_eval)} = {r.rate:.3f}" for r in sub_.itertuples()) + ".")
    emit(f"- Kare kumesi bos dokunus: {int(T['nomap'].sum())}/{len(T)} (paydadan dusulen dokunus yok; kural bu veride bagayici olmadi).")
    emit(f"- Tahmin (a) d50 in [0.2, 0.4] (D-ham): {sum(1 for r in rows_a if r['[0.2, 0.4] icinde?'] == 'TUTAR')}/6 oturumda TUTAR; nokta d50 tahmin edilebilen oturumlar: "
         + ", ".join(f"{r['oturum']} {r['d50_ham_s']:.3f} s" for r in rows_a if r['durum'] == 'ok') + " (ikisi de aralik **altinda**). Yani sistem, on-yazili tahminden **daha kisa** dokunuslari yakaliyor; tahmin (a) desteklenmedi.")
    gl_ = T[~T["nomap"]]
    emit("- Egitim etiketi (L0) gorunurlugu vs B2 D-ham (tum oturumlar, yalniz bilgi): " + "; ".join(
        f"{bl}: L0 {gl_[gl_['bin'] == bl]['L0_vis'].astype(bool).mean():.3f} / B2 {gl_[gl_['bin'] == bl]['b2_ham'].astype(bool).mean():.3f}" for bl in BINLAB if (gl_['bin'] == bl).any())
         + ". L0'da neredeyse hic gorunmeyen kisa dokunuslarin bir kismi yine de ham akista yakalaniyor (kare kumesindeki baska karelerin / L=5 pencerelerinin tasmasi ve 100 ms hizalama payi dahil; Bolum 8 uyarisi).")
    emit(f"- Tahmin (b): {'TUTUYOR' if all(okb) else 'TUTMUYOR'}. Tahmin (c): EVET {ce}, HAYIR {ch}, belirsiz {6 - ce - ch}. Kural (d): d50 > 0.4 s olan oturum yok -> sinir d90'a yukseltilmedi (s06'da D-ham/D-deb tum dokunuslar yakalandi ve en kisa dokunus 0.477 s > 0.4 s oldugundan kural veriden degerlendirilemedi: 'belirsiz').")
    mx = SP.loc[SP["sim_hata"].idxmax()]
    lt = SP[SP["aralik_lt_11_8cm"]]
    emit(f"- Konum ekseni: aralik < 11.8 cm olan {len(lt)} cifte B2 sim. hata en fazla {lt['sim_hata'].max():.4f}; en yuksek hata aralik >= 11.8 cm bir cifte ait: {mx['oturum']} {mx['cift']} (aralik {mx['aralik_cm']:.1f} cm, sim_hata {mx['sim_hata']:.4f}, 3->2 yonu {mx['hata_b_to_a']:.4f}). "
         f"Yani 14 ciftte hata kutu araligiyla monoton degil; 11.8 cm cizgisi B2'nin ikili karisikligini ayirmiyor. (Kosullar: S1 ayni oturumun diger dongulerinden ogrenir; ucuncu sinif/none tahminleri haric; Bolum 8.)")
    emit("- Destekledigi cumle (on-yazili) ile karsilastirma: *>= d90 s suren dokunuslarda guvenilir* kismi verilerle tutarli (d90: s01 ~0.49 s D-ham / ~0.53 s D-deb; s05 ~0.27 s / ~0.33 s; diger oturumlarda d90 tahmin edilemedi cunku neredeyse hic kacirma yok; 0.5 s ustu kutularda B2 oranlari yukaridaki maddede). "
         "*>= ~12 cm aralikli kutularda* kismi B2'nin ikili karisikligiyle **desteklenmiyor**: aralik < 11.8 cm olan ciftlerde sim. hata en fazla %1.3.")
    emit()

    # ---------- uyarilar ----------
    emit("## 8. On-kayitli uyarilar (brifteki gibi)")
    emit()
    emit("- Bu sinir **egitilmis sistemin** sinirdir, radarin degil: egitim etiketleri (L0, +-3 kare mod) kisa dokunuslari da silmisti (bkz. Bolum 6'daki L0-gorunurluk sayilari).")
    emit("- L = 5 pencere (0.5 s) ve hizli tempo hareketleriyle karisik: kisa dokunus, penceredeki baska karelerin etkisiyle de kacirilabilir/yakalanabilir; sure ile hareket hizi ayrisamaz.")
    emit("- Tek operator, tek mesafe (~0.45 m); kutu araligi radar-merkez farkidir; s01 (egitim oturumu) rastgele toplama icerir.")
    emit("- **S1 kapsami:** OOF tahminleri 6 oturumun havuzunda dongu-gruplu 5-kat ile uretildi; her katin egitim kumesi ayni oturumlarin (ayni kutu yerlesimi) diger dongulerini icerir. Dolayisiyla hem tespit hem konum ekseni 'ayni yerlesimde, baska dongu' kosulunu olcer; yeni bir yerlesime/oturuma genelleme (S2/LOSO) degildir ve 8 cm ciftlerdeki dusuk karisiklik yerlesimin egitimde gorulmus olmasindan da gelebilir.")
    emit("- Ikili karisiklik ucuncu sinif (none dahil) tahminlerini disarida birakir; none'a dusen/none'dan gelen hatalar (gecisler) bu olcude gorunmez, yani olcut yalniz kutu-kutu karisikligidir.")
    emit("- Dogrusal-saniye lojistik uydurmasi kisa dokunuslarda plato yapan oranlari zayif temsil eder (orn. s01); d50/d90 yalniz oturum basina ozet olarak, kutu oranlariyla birlikte okunmalidir.")
    emit("- Bolum 7 betimleyicidir: 14 cift, sabit/tasarim esigi uydurulmadi. Tespit tanimlari kare kumesi bos dokunuslari payda disi birakir (Bolum 0).")
    emit("- Faz A birincil sonuc; bu analiz ikincil ve sonradan yapildi. Tum on-kayitli sayilar (olumsuz olanlar dahil) yukarida raporlanmistir.")
    emit()
    (OUT / "02_envelope.md").write_text("\n".join(MD) + "\n", encoding="utf-8")


# ============================================================================================ sekiller
def plot_detection(T: pd.DataFrame, FITdf: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), sharey=True)
    rng = np.random.RandomState(0)
    for ax, s in zip(axes.ravel(), SESS):
        g = T[(T["session"] == SHORT[s]) & (~T["nomap"])]
        xs = g["dur_s"].to_numpy()
        for dn, off, mk in (("ham", 0.97, "o"), ("deb", 1.03, "s")):
            y = g[f"b2_{dn}"].astype(bool).to_numpy()
            jit = rng.uniform(-0.035, 0.035, len(g))
            ax.scatter(xs[y] * off, 1.09 + (0.0 if dn == "ham" else 0.05) + jit[y] * 0.3, s=6, color=COL[dn], alpha=0.25, marker="|")
            ax.scatter(xs[~y] * off, -0.09 - (0.0 if dn == "ham" else 0.05) + jit[~y] * 0.3, s=6, color=COL[dn], alpha=0.5, marker="|")
            for bi, bl in enumerate(BINLAB):
                gg = g[g["bin"] == bl]
                if len(gg) == 0:
                    continue
                k, n = int(gg[f"b2_{dn}"].astype(bool).sum()), len(gg)
                p, lo, hi = wilson(k, n)
                xm = float(np.median(gg["dur_s"])) * off
                ax.errorbar([xm], [p], yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]], color=COL[dn], marker=mk, ms=6, lw=1.2, capsize=2.5, mec="white", mew=0.8, label=f"{DNAME[dn]} (Wilson %95)" if bi == 0 else None)
            f = FITdf[(FITdf["model"] == "B2") & (FITdf["session"] == SHORT[s]) & (FITdf["tanim"] == DNAME[dn])].iloc[0]
            if f["status"] == "ok" and not f["extrap"]:
                xx = np.geomspace(max(0.05, f["xmin"]), max(f["xmax"], 0.5), 200)
                ax.plot(xx, 1 / (1 + np.exp(-(f["b0"] + f["b1"] * xx))), color=COL[dn], lw=1.6, ls="-" if dn == "ham" else "--", alpha=0.9)
        for e in EDGES[1:-1]:
            ax.axvline(e, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0.4, color="0.45", lw=0.9, ls=":", zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(0.04, max(6.0, xs.max() * 1.2 if len(xs) else 6.0))
        ax.set_ylim(-0.25, 1.3)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.axhline(0.9, color="0.8", lw=0.6)
        ax.set_title(f"{SHORT[s]}  (n={len(g)} dokunus)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("dokunus suresi (s, log)")
    axes[0, 0].set_ylabel("B2 tespit orani")
    axes[1, 0].set_ylabel("B2 tespit orani")
    axes[0, 0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("A3.2 - B2 tespit orani ~ ToF dokunus suresi (oturum basina). Ust/alt cizgiler: tek tek dokunus (yakalanan/kacirilan); kesik dikey: 0.4 s; egri: lojistik uydurma (yalniz ekstrapolasyonsuz olanlar)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "02_envelope_detection.png", dpi=150)
    plt.close(fig)


def plot_spatial(SP: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.axvline(DX_RES * 100, color="0.4", lw=1.2, ls="--")
    ax.text(DX_RES * 100 + 0.3, 0.09, "11.8 cm (lateral cozunurluk)", fontsize=8, color="0.3", va="top")
    for r in SP.itertuples():
        hi = (r.oturum == "s04" and r.cift == "4-5")
        ax.scatter(r.aralik_cm, r.sim_hata, s=50 if hi else 30, color="#D55E00" if hi else "#0072B2", edgecolor="white", linewidth=0.8, zorder=3)
    # etiketler: ayni (cift, ~aralik) olanlari birlestir; sifir hatali noktalar dikey yazilir
    grp = {}
    for r in SP.itertuples():
        grp.setdefault((r.cift, round(r.aralik_cm)), []).append(r)
    for (cift, _), rs in grp.items():
        lab = "/".join(x.oturum for x in rs) + f" {cift}"
        x0, y0 = float(np.mean([x.aralik_cm for x in rs])), float(np.mean([x.sim_hata for x in rs]))
        if y0 < 0.002:
            ax.annotate(lab, (x0, y0), textcoords="offset points", xytext=(0, 7), fontsize=7, color="0.2", rotation=90, ha="center", va="bottom")
        else:
            ax.annotate(lab, (x0, y0), textcoords="offset points", xytext=(5, 4), fontsize=7, color="0.2")
    ax.set_xlabel("komsu kutu araligi (cm; radar kare-merkezi cx medyan farki)")
    ax.set_ylabel("B2 ikili karisiklik (sim. hata; 3. sinif tahminleri haric)")
    ax.set_title("A3.2 - komsu cift karisikligi vs kutu araligi (betimleyici, fit yok; turuncu: s04 4-5)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.004, 0.155)
    ax.set_xlim(6, 33)
    fig.tight_layout()
    fig.savefig(OUT / "02_envelope_spatial.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Faz A2 / A2.1 - Conformal esik: esitlikler ve kalibrasyon cozunurlugu (DESIGN_A2.md, A2.1).

Hucreler: B2 x pred (ana), B2 x tof, ToF oracle x tof, CNN-GRU x pred. Yalniz mevcut S1 OOF tahminleri; yeniden egitim yok.
Yontemler (hepsi ayni test dongu kumesi uzerinde: kat f normal + tum anomali donguleri, f = 0..4):
  std          : Faz A standart conformal p (rol: egitim f+3,f+4 / kal f+1,f+2) -> Faz A ile BIREBIR esit olmali (kontrol basilir)
  smooth       : duzgunlestirilmis (randomize) p, ayni roller; 200 cekilis, sabit tohum (D1 ve D2 icin bagimsiz U)
  cross        : cross-conformal (4 kalan katin her biri k icin parametre kalan 3 kattan; kat k kat-disi kalibrasyon), standart p
  cross_smooth : cross-conformal + duzgunlestirme (ayni tohumlar)
Kural: D3 = min(1, 2 min(p1, p2)) (Bonferroni, birincil). D1, D2 ikincil. 'D1orD2' = (p1<=a) | (p2<=a): BETIMLEYICI, calisma noktasi DEGIL.
Cikti: reports/faz_a2/01_conformal.md (+ 01_conformal_table.csv uzun format, 01_conformal_meta.csv). Her sayi stdout'a da basilir.
Rapor sonundaki '<!-- YORUM -->' isaretinden sonraki metin yeniden kosuda korunur.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a"))
sys.path.insert(0, str(ROOT / "faz_a2"))
sys.path.insert(0, str(ROOT / "probe"))
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import a2_cycles as ac
import fa_common as fc
from common import md_table

OUT2 = ROOT / "reports" / "faz_a2"
OUT2.mkdir(parents=True, exist_ok=True)
MD = OUT2 / "01_conformal.md"
MARK = "<!-- YORUM -->"
SEED0 = 20261104
N_DRAWS = 200
CELLS = [("b2", "pred"), ("b2", "tof"), ("tof", "tof"), ("cnngru", "pred")]
CELLNAME = {("b2", "pred"): "B2 x pred (ana)", ("b2", "tof"): "B2 x tof", ("tof", "tof"): "ToF oracle x tof",
            ("cnngru", "pred"): "CNN-GRU x pred"}
METHODS = ["std", "smooth", "cross", "cross_smooth"]
MNAME = {"std": "Faz A standart", "smooth": "duzgunlestirilmis", "cross": "cross-conformal",
         "cross_smooth": "cross + duzgunlestirme"}
VARS = ["D1", "D2", "D3", "D1orD2"]
GROUPS = list(ac.AGROUPS)            # seq135, seq145, fast, slow, seq(135+145), tempo(fast+slow), tum anomali
t0 = time.time()


def P(*a):
    print(*a, flush=True)


def seed_for(f, k):
    """f. kat, k=1 (D1) / k=2 (D2): D1 ve D2 icin BAGIMSIZ U; ayni tohum cifti butun hucrelerde (eslesik)."""
    return SEED0 + 100 * f + k


# ------------------------------------------------------------------ metrikler (vektorlestirilmis; nd cekilis x n satir)
def auroc_rows(Q, neg, pos):
    """Q (nd, n) skor (yuksek = anomali); neg/pos bool maskeleri. ac.auc_ap ile ayni sozlesme (NaN: <2 negatif ya da <1 pozitif)."""
    nn, npos = int(neg.sum()), int(pos.sum())
    if nn < 2 or npos < 1:
        return np.full(Q.shape[0], np.nan)
    idx = np.r_[np.flatnonzero(neg), np.flatnonzero(pos)]
    R = rankdata(Q[:, idx], axis=1)
    return (R[:, nn:].sum(axis=1) - npos * (npos + 1) / 2.0) / (nn * npos)


class Rows:
    """Bir (hucre, yontem) icin test satirlari: meta + P1, P2 (nd, n). std/cross icin nd = 1."""

    def __init__(self, meta: pd.DataFrame, P1, P2, raw=None):
        self.meta = meta.reset_index(drop=True)
        self.P1, self.P2 = np.atleast_2d(P1), np.atleast_2d(P2)
        self.P3 = ac.d3(self.P1, self.P2)
        self.raw = raw                         # std icin ham skorlar {'D1': (1,n), 'D2': (1,n)}
        m = self.meta
        self.f = m["f"].to_numpy(int)
        self.normal = (m["atype"] == "normal").to_numpy()
        self.nat = m["natural_dev"].to_numpy(bool)
        self.gmask = {g: m["atype"].isin(ac.AGROUPS[g]).to_numpy() for g in GROUPS}
        self.nd = self.P1.shape[0]

    def alarm(self, var, a):
        if var == "D1":
            return ac.alarm(self.P1, a)
        if var == "D2":
            return ac.alarm(self.P2, a)
        if var == "D3":
            return ac.alarm(self.P3, a)
        return ac.alarm(self.P1, a) | ac.alarm(self.P2, a)

    def pmat(self, var):
        return {"D1": self.P1, "D2": self.P2, "D3": self.P3}[var]


def agg(x):
    """(nd,) -> (mean, lo, hi); nd = 1 icin lo = hi = mean."""
    x = np.asarray(x, float)
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    return float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def metrics(R: Rows, cell, method, rec: list):
    """Uzun-format kayitlar: (cell, method, variant, alpha, atype, metric, value, lo, hi, extra)."""
    ck = dict(cell=cell, method=method)
    nn_pairs = int(R.normal.sum())
    for var in VARS:
        for a in ac.ALPHAS:
            al = R.alarm(var, a)
            far = al[:, R.normal].mean(axis=1)
            nonat = R.normal & ~R.nat
            far_x = al[:, nonat].mean(axis=1)
            v, lo, hi = agg(far)
            rec.append(dict(ck, variant=var, alpha=a, atype="normal", metric="far", value=v, lo=lo, hi=hi, n=nn_pairs))
            v, lo, hi = agg(far_x)
            rec.append(dict(ck, variant=var, alpha=a, atype="normal", metric="far_excl_nat", value=v, lo=lo, hi=hi,
                            n=int(nonat.sum())))
            for g in GROUPS:
                v, lo, hi = agg(al[:, R.gmask[g]].mean(axis=1))
                rec.append(dict(ck, variant=var, alpha=a, atype=g, metric="det", value=v, lo=lo, hi=hi,
                                n=int(R.gmask[g].sum())))
    # AUROC (kat bazli; sonra katlar uzerinden ortalama).  Skor: 1 - p  (std icin ayrica D1/D2 ham skor)
    scorings = {v: (1.0 - R.pmat(v), "p") for v in ("D1", "D2", "D3")}
    if R.raw is not None:
        scorings["D1raw"] = (R.raw["D1"], "ham")
        scorings["D2raw"] = (R.raw["D2"], "ham")
    for var, (Q, kind) in scorings.items():
        for g in GROUPS:
            per_fold = []
            for f in sorted(set(R.f)):
                m = R.f == f
                per_fold.append(auroc_rows(Q[:, m], R.normal[m], R.gmask[g][m]))
            A = np.vstack(per_fold)                      # (nfold, nd)
            with np.errstate(all="ignore"):
                mean_f = np.nanmean(A, axis=0)           # kat ortalamasi, cekilis basina
                sd_f = np.nanstd(A, axis=0, ddof=1) if A.shape[0] > 1 else np.full(A.shape[1], np.nan)
            v, lo, hi = agg(mean_f)
            rec.append(dict(ck, variant=var, alpha=np.nan, atype=g, metric="auroc", value=v, lo=lo, hi=hi,
                            n=int(np.isfinite(A).any(axis=1).sum())))
            rec.append(dict(ck, variant=var, alpha=np.nan, atype=g, metric="auroc_sd", value=float(np.nanmean(sd_f)),
                            lo=np.nan, hi=np.nan, n=A.shape[0]))


# ------------------------------------------------------------------ kuruluslar
def build_std(S: pd.DataFrame) -> Rows:
    T = S[S["role"] == "test"].sort_values("f", kind="stable").reset_index(drop=True)
    return Rows(T[["f", "atype", "natural_dev"]], T["p1"].to_numpy()[None], T["p2"].to_numpy()[None],
                raw={"D1": T["D1"].to_numpy()[None], "D2": T["D2"].to_numpy()[None]}), T


def build_smooth(S: pd.DataFrame, T_std: pd.DataFrame):
    """Faz A rolleri; duzgunlestirilmis p. Ayrica D1/D2 icin ANALITIK beklenen alarm olasiligi (MC kontrolu)."""
    metas, P1, P2, ex = [], [], [], {}
    for f in sorted(S["f"].unique()):
        cal = S[(S["f"] == f) & (S["role"] == "cal")]
        te = S[(S["f"] == f) & (S["role"] == "test")]
        metas.append(te[["f", "atype", "natural_dev"]])
        P1.append(ac.smoothed_draws(cal["D1"].to_numpy(), te["D1"].to_numpy(), N_DRAWS, seed_for(f, 1)))
        P2.append(ac.smoothed_draws(cal["D2"].to_numpy(), te["D2"].to_numpy(), N_DRAWS, seed_for(f, 2)))
        for v in ("D1", "D2"):
            gt, eq = ac._gt_eq(cal[v].to_numpy(), te[v].to_numpy())
            ex.setdefault(v, []).append((gt, eq, len(cal)))
    meta = pd.concat(metas, ignore_index=True)
    assert (meta[["f", "atype"]].to_numpy() == T_std[["f", "atype"]].to_numpy()).all(), "satir sirasi std ile ayni degil"
    return Rows(meta, np.hstack(P1), np.hstack(P2)), ex


def build_cross(C: pd.DataFrame, T_std: pd.DataFrame):
    metas, P1s, P2s, P1d, P2d, ex, info = [], [], [], [], [], {}, []
    for f in range(ac.N_FOLDS):
        cf = ac.cross_scores(C, f)
        m = cf["test"][["atype", "natural_dev"]].copy()
        m.insert(0, "f", f)
        metas.append(m)
        P1s.append(ac.cross_p(cf, "D1"))
        P2s.append(ac.cross_p(cf, "D2"))
        P1d.append(ac.cross_p_draws(cf, "D1", N_DRAWS, seed_for(f, 1)))
        P2d.append(ac.cross_p_draws(cf, "D2", N_DRAWS, seed_for(f, 2)))
        for v in ("D1", "D2"):
            gt, eq = ac._cross_counts(cf, v)
            ex.setdefault(v, []).append((gt, eq, cf["n_cal"]))
        info.append(dict(f=f, n_cal=cf["n_cal"], n_train_models=cf["n_train"], folds_k=cf["folds_k"],
                         n_test=len(cf["test"])))
    meta = pd.concat(metas, ignore_index=True)
    assert (meta[["f", "atype"]].to_numpy() == T_std[["f", "atype"]].to_numpy()).all(), "cross satir sirasi std ile ayni degil"
    return (Rows(meta, np.concatenate(P1s)[None], np.concatenate(P2s)[None]),
            Rows(meta, np.hstack(P1d), np.hstack(P2d)), ex, info)


def exact_alarm_prob(ex, v, a):
    """Duzgunlestirilmis p icin P(p <= a) = clip((a (n+1) - gt) / (eq+1), 0, 1) (1e-9 toleransli); (n_rows,)."""
    out = []
    for gt, eq, n in ex[v]:
        out.append(np.clip(((a + 1e-9) * (n + 1.0) - gt) / (eq + 1.0), 0.0, 1.0))
    return np.concatenate(out)


# ------------------------------------------------------------------ calistir
P(f"[{time.time()-t0:.0f}s] kaynak etiketleri yukleniyor")
LAB = ac.source_labels(sources=("tof", "cnngru", "b2"))
REC, META, CHK = [], [], []
faza = pd.read_csv(fc.OUT / "04_anomaly_table.csv")
MAXDIFF = []

for src, seg in CELLS:
    cell = CELLNAME[(src, seg)]
    P(f"\n[{time.time()-t0:.0f}s] ===== {cell} =====")
    C, _ = ac.build_cycles(src, seg, LAB[src])
    S, ncal = ac.score_group(C, verbose=True)
    ncal = pd.DataFrame(ncal)
    Rs, Tstd = build_std(S)
    Rm, exm = build_smooth(S, Tstd)
    Rc, Rcs, exc, cinfo = build_cross(C, Tstd)
    for meth, R in (("std", Rs), ("smooth", Rm), ("cross", Rc), ("cross_smooth", Rcs)):
        metrics(R, cell, meth, REC)

    # --- (i) Faz A ile esitlik kontrolu (std; D1/D2/D3; far/det/auroc)
    cur = pd.DataFrame(REC)
    cur = cur[(cur["cell"] == cell) & (cur["method"] == "std")]
    ref = faza[(faza["state_source"] == src) & (faza["segmentation"] == seg)]
    diffs = []
    for var in ("D1", "D2", "D3"):
        for g in GROUPS:
            r = ref[(ref["variant"] == var) & (ref["atype"] == g)].iloc[0]
            for a, k in ((0.10, "a10"), (0.05, "a05")):
                x = cur[(cur["variant"] == var) & (cur["alpha"] == a) & (cur["atype"] == g) & (cur["metric"] == "det")]["value"].iloc[0]
                diffs.append(abs(x - r[f"det_{k}"]))
                x = cur[(cur["variant"] == var) & (cur["alpha"] == a) & (cur["atype"] == "normal") & (cur["metric"] == "far")]["value"].iloc[0]
                diffs.append(abs(x - r[f"far_{k}"]))
                x = cur[(cur["variant"] == var) & (cur["alpha"] == a) & (cur["atype"] == "normal") & (cur["metric"] == "far_excl_nat")]["value"].iloc[0]
                diffs.append(abs(x - r[f"far_excl_nat_{k}"]))
            vv = {"D1": "D1raw", "D2": "D2raw", "D3": "D3"}[var]
            x = cur[(cur["variant"] == vv) & (cur["atype"] == g) & (cur["metric"] == "auroc")]["value"].iloc[0]
            diffs.append(abs(x - r["auroc_mean"]))
            x = cur[(cur["variant"] == vv) & (cur["atype"] == g) & (cur["metric"] == "auroc_sd")]["value"].iloc[0]
            if np.isfinite(r["auroc_sd"]):
                diffs.append(abs(x - r["auroc_sd"]))
    md_ = float(np.nanmax(diffs))
    MAXDIFF.append((cell, md_, len(diffs)))
    P(f"[kontrol (i)] {cell}: Faz A 04_anomaly_table.csv ile max |fark| = {md_:.3g} ({len(diffs)} sayi; D1/D2/D3 x 7 grup x far/far_excl_nat/det x a10,a05 + AUROC mean/sd)")

    # --- MC vs analitik beklenen deger (D1/D2; smooth ve cross_smooth)
    for nm, R, ex in (("smooth", Rm, exm), ("cross_smooth", Rcs, exc)):
        worst = 0.0
        for v, AR in (("D1", Rm.P1 if nm == "smooth" else Rcs.P1), ("D2", Rm.P2 if nm == "smooth" else Rcs.P2)):
            for a in ac.ALPHAS:
                pe = exact_alarm_prob(ex, v, a)
                mc = (AR <= a + 1e-9).mean(axis=0)
                worst = max(worst, abs(pe[R.normal].mean() - mc[R.normal].mean()),
                            abs(pe[R.gmask["tum anomali"]].mean() - mc[R.gmask["tum anomali"]].mean()))
        P(f"[kontrol] {cell} {nm}: |analitik beklenen - MC ortalama| (D1/D2, FAR ve tum-anomali tespit) max = {worst:.4f}")

    # --- n_cal, en kucuk ulasilabilir p
    nc = ncal["n_cal"]
    ntr = ncal["n_train"]
    cx = pd.DataFrame(cinfo)
    row = dict(cell=cell,
               std_ncal_min=int(nc.min()), std_ncal_mean=float(nc.mean()), std_ncal_max=int(nc.max()),
               std_ntrain_mean=float(ntr.mean()),
               std_minp_D1D2_at_ncal_max=1 / (nc.max() + 1), std_minp_D1D2_at_ncal_min=1 / (nc.min() + 1),
               std_minp_D3_at_ncal_max=2 / (nc.max() + 1), std_minp_D3_at_ncal_min=2 / (nc.min() + 1),
               std_obs_min_p1=float(Rs.P1.min()), std_obs_min_p2=float(Rs.P2.min()), std_obs_min_p3=float(Rs.P3.min()),
               cross_ncal_mean=float(cx["n_cal"].mean()), cross_ncal_min=int(cx["n_cal"].min()), cross_ncal_max=int(cx["n_cal"].max()),
               cross_ntrain_per_model=",".join(str(x) for x in cx.loc[0, "n_train_models"]),
               cross_minp_D1D2=1 / (cx["n_cal"].max() + 1), cross_minp_D3=2 / (cx["n_cal"].max() + 1),
               cross_obs_min_p1=float(Rc.P1.min()), cross_obs_min_p2=float(Rc.P2.min()), cross_obs_min_p3=float(Rc.P3.min()),
               smooth_obs_min_p1=float(Rm.P1.min()), smooth_obs_min_p3=float(Rm.P3.min()),
               cross_smooth_obs_min_p3=float(Rcs.P3.min()),
               n_normal_pairs=int(Rs.normal.sum()), n_anom_cycles_per_f=int((~Rs.normal & (Rs.f == 0)).sum()),
               n_natdev_normal=int((Rs.normal & Rs.nat).sum()))
    META.append(row)
    P("[n_cal / min p]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

DF = pd.DataFrame(REC)
DF.to_csv(OUT2 / "01_conformal_table.csv", index=False, lineterminator="\n", float_format="%.6g")
MET = pd.DataFrame(META)
MET.to_csv(OUT2 / "01_conformal_meta.csv", index=False, lineterminator="\n", float_format="%.6g")
P(f"\n[{time.time()-t0:.0f}s] CSV yazildi: {len(DF)} satir")
P("[kontrol (i) ozet]", [(c, f"{d:.3g}") for c, d, n in MAXDIFF])


# ------------------------------------------------------------------ rapor
def get(cell, meth, var, a, atype, metric):
    q = DF[(DF["cell"] == cell) & (DF["method"] == meth) & (DF["variant"] == var) & (DF["atype"] == atype) & (DF["metric"] == metric)]
    if a is not None:
        q = q[q["alpha"] == a]
    r = q.iloc[0]
    return r["value"], r["lo"], r["hi"], int(r["n"])


def fmt(v, lo, hi, meth, dec=3):
    if meth in ("smooth", "cross_smooth"):
        return f"{v:.{dec}f} [{lo:.{dec}f}; {hi:.{dec}f}]"
    return f"{v:.{dec}f}"


def far_label(far, a, n, var):
    if var == "D1orD2":
        se = np.sqrt(2 * a * (1 - 2 * a) / n)
        return "<=2a sinir icinde" if far <= 2 * a + 2 * se else "2a SINIRI ASILDI"
    se = np.sqrt(a * (1 - a) / n)
    if abs(far - a) <= 2 * se:
        return "nominal +-2SE icinde"
    return "muhafazakar (< a-2SE)" if far < a else "ASIRI (> a+2SE; gecersiz)"


L = []
H = L.append
H("# A2.1 - Conformal esik: esitlikler ve kalibrasyon cozunurlugu")
H("")
H("Uretim: `faz_a2/01_conformal.py` (yerel makine). Ham sayilar: `01_conformal_table.csv` (uzun format), `01_conformal_meta.csv`. "
  "Her sayi betigin stdout'una da basilir. **Statu:** ikincil, Faz A'dan SONRA (DESIGN_A2.md); makalenin birincil sayilari Faz A'dir "
  "ve bu rapor onlari degistirmez. Yeniden egitim yok: yalniz mevcut S1 OOF tahminleri.")
H("")
H("## 0. Yontem ozeti")
H("")
H("- **Birincil kural degismedi:** D3 = min(1, 2 min(p1, p2)) (Bonferroni; alarm = 'D1 ya da D2, her biri a/2'de'). D1 ve D2 ayrica tek tek verilir (ikincil). "
  "**D1orD2** = '(p1 <= a) ya da (p2 <= a)': nominal FAR siniri <= 2a; **betimleyicidir, bir calisma noktasi olarak sunulmaz.**")
H("- **(i) Faz A standart:** p = (1 + #{cal >= s}) / (n_cal + 1); roller Faz A ile ayni (egitim f+3,f+4; kal f+1,f+2; test f). "
  "Bu satirlar `04_anomaly_table.csv` ile karsilastirildi (asagida).")
H(f"- **(ii) Duzgunlestirilmis p:** p = (#{{cal > s}} + U (#{{cal = s}} + 1)) / (n_cal + 1), U ~ Unif(0,1); ayni roller. {N_DRAWS} cekilis; "
  f"tohum = {SEED0} + 100 f + k (k = 1: D1, k = 2: D2; yani D1 ve D2 icin BAGIMSIZ U; ayni tohum cifti her hucrede). "
  "Tablolarda 'ortalama [%2,5; %97,5]' = cekilisler arasi aralik (yalniz U rastgeleligi; ornekleme belirsizligi **degil**).")
H("- **(iii) Cross-conformal (s02 normal donguleri):** test kati f (`folds_anom.csv`); kalan 4 katin her biri k icin parametreler diger 3 kattan (~30 dongu) "
  "ogrenilir, k kati kat-disi kalibrasyon skorlarini verir, test/anomali donguleri model k ile skorlanir; "
  "p = (1 + sum_k #{cal_k >= s^(k)}) / (n_cal + 1). 'cross + duzgunlestirme' = ayni sayimlarla (ii)'deki formul (ayni tohumlar).")
H("- **FAR** = havuzlanmis normal test donguleri (5 kat), **SE = sqrt(a(1-a)/n_normal)** (Faz A kurali); etiket: |FAR - a| <= 2 SE ise 'nominal icinde', "
  "altindaysa 'muhafazakar', ustundeyse 'ASIRI'. D1orD2 icin sinir 2a. Duzgunlestirilmis satirlarda etiket **ortalama** FAR'a uygulanir. "
  "**Tespit** = (dongu, f) ciftleri uzerinden anomali dongulerinde alarm orani (cross-conformal'da anomali donguleri her f'de yeniden skorlandigindan f uzerinden ortalama ile ayni).")
H("- **AUROC:** kat bazli, sonra katlar uzerinden ortalama (Faz A kurali). Skor: D3 ve p-tabanli D1/D2 icin 1 - p; ayrica Faz A'nin ham-skor D1/D2 AUROC'u (std icin). "
  "Cross-conformal'da tek bir ham skor yoktur (her f icin 4 model), bu yuzden D1/D2 AUROC'u p-tabanlidir.")
H("")
H("### Faz A ile esitlik kontrolu (i)")
H("")
H("Yukaridaki std satirlari `reports/faz_a/04_anomaly_table.csv` ile karsilastirildi (D1, D2, D3 x 7 anomali grubu x FAR / FAR (dogal sapma haric) / tespit x a = 0,10 ve 0,05, "
  "ve AUROC ortalama ile sd):")
H("")
H(md_table(pd.DataFrame([dict(hucre=c, karsilastirilan_sayi=n, max_mutlak_fark=f"{d:.3g}") for c, d, n in MAXDIFF])))
H("")

# --- n_cal / min p
H("## 1. Kalibrasyon cozunurlugu: n_cal ve ulasilabilir en kucuk p")
H("")
mt = MET.copy()
H("Standart conformal'da en kucuk p = 1/(n_cal+1); D3 = min(1, 2 min(p1,p2)) oldugundan D3'un en kucugu 2/(n_cal+1). "
  "Duzgunlestirmede p surekli (U/(n_cal+1)'e kadar iner); 'gozlenen en kucuk p' test satirlari uzerindendir (std/cross: tek deger; smooth: 200 cekilis uzerinden en kucuk).")
H("")
H(md_table(pd.DataFrame([dict(
    hucre=r["cell"],
    std_ncal=f'{r["std_ncal_min"]}/{r["std_ncal_mean"]:.1f}/{r["std_ncal_max"]}',
    std_egitim=f'{r["std_ntrain_mean"]:.1f}',
    std_minp_D1D2=f'{r["std_minp_D1D2_at_ncal_max"]:.4f}-{r["std_minp_D1D2_at_ncal_min"]:.4f}',
    std_minp_D3=f'{r["std_minp_D3_at_ncal_max"]:.4f}-{r["std_minp_D3_at_ncal_min"]:.4f}',
    std_gozlenen_min_p3=f'{r["std_obs_min_p3"]:.4f}',
    cross_ncal=f'{r["cross_ncal_min"]}-{r["cross_ncal_max"]}',
    cross_egitim_model_basi=r["cross_ntrain_per_model"],
    cross_minp_D1D2=f'{r["cross_minp_D1D2"]:.4f}',
    cross_minp_D3=f'{r["cross_minp_D3"]:.4f}',
    cross_gozlenen_min_p3=f'{r["cross_obs_min_p3"]:.4f}',
    smooth_gozlenen_min_p3=f'{r["smooth_obs_min_p3"]:.5f}',
) for _, r in mt.iterrows()])))
H("")
H("(std_ncal = min/ortalama/maks kat basina kalibrasyon dongusu; std_minp_* aralik = n_cal maks ... n_cal min.) "
  "**Okuma:** std'de D3'un ulasilabilir en kucuk p'si ~0,09-0,10 oldugundan a = 0,05'te D3 alarmi yapisal olarak imkansiz, a = 0,10'da yalniz tum kalibrasyonu asan dongulerde mumkun. "
  "cross-conformal'da n_cal ~40 oldugundan D3 icin en kucuk p ~0,049: a = 0,05'e ancak sinirda ulasilir.")
H("")
H(f"Test kumeleri (hucre basina): normal test cifti sayisi = {', '.join(f'{r.cell}: {r.n_normal_pairs}' for r in mt.itertuples())}; "
  f"her f'de anomali dongusu = {', '.join(f'{r.cell}: {r.n_anom_cycles_per_f}' for r in mt.itertuples())}; "
  f"normal test dongulerinde dogal sapma = {', '.join(f'{r.cell}: {r.n_natdev_normal}' for r in mt.itertuples())}.")
H("")

# --- ana tablolar
for ai, a in enumerate(ac.ALPHAS):
    H(f"## {2 + ai}. FAR ve tespit, a = {a:.2f}")
    H("")
    H("Sutunlar: FAR (SE; etiket), tur basina tespit, ayrica seq(135+145), tempo(fast+slow) ve tum anomali havuzlari. "
      "Duzgunlestirilmis satirlar 'ortalama [%2,5; %97,5]'.")
    H("")
    for src, seg in CELLS:
        cell = CELLNAME[(src, seg)]
        H(f"### {cell}, a = {a:.2f}")
        H("")
        rows = []
        for var in VARS:
            for meth in METHODS:
                far, flo, fhi, n = get(cell, meth, var, a, "normal", "far")
                se = np.sqrt(a * (1 - a) / n) if var != "D1orD2" else np.sqrt(2 * a * (1 - 2 * a) / n)
                lab = far_label(far, a, n, var)
                r = dict(degisken=(var if var != "D1orD2" else "D1orD2 (betimleyici, nominal<=2a)"), yontem=MNAME[meth],
                         FAR=fmt(far, flo, fhi, meth) + f" (SE {se:.3f})", gecerlilik=lab)
                for g in GROUPS:
                    v, lo, hi, _ = get(cell, meth, var, a, g, "det")
                    r[g] = fmt(v, lo, hi, meth)
                rows.append(r)
        H(md_table(pd.DataFrame(rows)))
        H("")

# --- AUROC
H(f"## {2 + len(ac.ALPHAS)}. AUROC (kat ortalamasi)")
H("")
H("std satirlari Faz A ile ayni (D1/D2 ham skor; 'p' satirlari 1-p tabanli). (ii) icin D1/D2 **ham skor** AUROC'u degismez (skor ayni); "
  "p-tabanli ve D3 satirlari duzgunlestirme rastgeleligi nedeniyle degisebilir ve 200 cekilis uzerinden ortalama [%2,5; %97,5] verilir. "
  "(iii) icin tum satirlar p-tabanlidir; AUROC kat basina yalniz ~10 normal donguye dayanir (SE ~0,03-0,05; 0,02'lik farklar anlamli degil).")
H("")
for src, seg in CELLS:
    cell = CELLNAME[(src, seg)]
    H(f"### {cell}")
    H("")
    rows = []
    for var in ("D1", "D2", "D3"):
        for meth in METHODS:
            vs = [var]
            if meth == "std":
                vs = [var + "raw", var] if var != "D3" else [var]
            for vv in vs:
                r = dict(degisken=vv.replace("raw", " (ham skor)") if "raw" in vv else (vv + " (1-p)" if var != "D3" else vv),
                         yontem=MNAME[meth])
                for g in GROUPS:
                    v, lo, hi, _ = get(cell, meth, vv, None, g, "auroc")
                    r[g] = fmt(v, lo, hi, meth)
                rows.append(r)
    H(md_table(pd.DataFrame(rows)))
    H("")

# --- baslik sayilari ve makale cumlesi
H(f"## {3 + len(ac.ALPHAS)}. Desteklenen makale cumlesi icin sayilar (B2 x pred, D3)")
H("")
cell = CELLNAME[("b2", "pred")]
srows = []
for a in ac.ALPHAS:
    for meth in METHODS:
        far, flo, fhi, n = get(cell, meth, "D3", a, "normal", "far")
        seq = get(cell, meth, "D3", a, "seq(135+145)", "det")
        tmp = get(cell, meth, "D3", a, "tempo(fast+slow)", "det")
        al = get(cell, meth, "D3", a, "tum anomali", "det")
        srows.append(dict(alpha=a, yontem=MNAME[meth], FAR=fmt(far, flo, fhi, meth), sira_sapmalari=fmt(*seq[:3], meth),
                          tempo_sapmalari=fmt(*tmp[:3], meth), tum_anomali=fmt(*al[:3], meth), etiket=far_label(far, a, n, "D3")))
H(md_table(pd.DataFrame(srows)))
H("")
H("Cumle kalibi: 'Esitlige duyarli cross-conformal kalibrasyonla (~40 normal dongu) iki kanalli detektor, dongu duzeyinde %Z yanlis alarmda sira sapmalarinin %X'ini, tempo sapmalarinin %Y'sini yakaliyor.' "
  "X, Y, Z yukaridaki 'cross + duzgunlestirme' ve 'cross-conformal' satirlarindan okunur; secici rapor yok: tum yontemler birlikte verilir.")
H("")
H("## Karistirici ve uyarilar")
H("")
H("- **Egitim kumesi boyutu (karistirici):** Faz A protokolunde parametreler ~20 normal donguden (2 kat), cross-conformal'da ~30 donguden (3 kat) ogrenilir "
  "(tablo 1: `std_egitim` ve `cross_egitim_model_basi`). Dolayisiyla (ii) -> (iii) farki hem **kalibrasyon kumesinin buyumesi (~20 -> ~40) hem de egitim kumesinin buyumesi**ni birlikte icerir; "
  "ikisi ayrilamaz. Yalniz (i) -> (ii) farki egitim kumesini sabit tutar ve esitlik duzeltmesini ayri olcer.")
H("- **Cross-conformal kalibrasyon kumesi** dort ayri modelin (k = 1..4) kat-disi skorlarindan olusur; her skor kendi modelinin test skoruyla karsilastirilir (p'nin toplamasi bunu yapar). "
  "Teorik FAR siniri ~2a'dir; ampirik olarak ~a beklenir. Gozlenen FAR yukarida etiketlenmistir.")
H("- **Dongu duzeyinde bagimsizlik:** normal dongular zaman icinde bagimsiz degildir (surukleme); komsu dongulerin farkli katlara dagilmasi kalibrasyonu iyimser yapabilir. "
  "Test normal dongusu sayisi ~50 oldugundan FAR'in SE'si ~0,04 (a = 0,10) ve ~0,03 (a = 0,05); %1-2'lik FAR farklari anlamli degil.")
H("- Anomali donguleri her f'de test oldugu icin ayni anomali donguleri 5 kez (farkli kalibrasyonla) sayilir; tespit orani bu (dongu, f) ciftleri uzerinden ortalamadir.")
H("- Dogal sapmali normal dongular (`natural_dev`) FAR'a dahildir; FAR (dogal sapma haric) CSV'de `far_excl_nat` olarak verilir.")
H("- 'D1orD2' betimleyicidir: nominal FAR siniri 2a'dir, bir calisma noktasi olarak sunulamaz.")
H("")

old = ""
if MD.exists():
    t = MD.read_text(encoding="utf-8")
    if MARK in t:
        old = t.split(MARK, 1)[1]
body = "\n".join(L) + "\n\n" + MARK + (old if old else "\n## Yorum\n\n(henuz yazilmadi)\n")
MD.write_text(body, encoding="utf-8", newline="\n")
P(f"[{time.time()-t0:.0f}s] yazildi: {MD}")

# --- stdout: rapordaki ana sayilar
P("\n===== STDOUT OZET (rapordaki ana tablolarin ham sayilari) =====")
pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 2000)
for src, seg in CELLS:
    cell = CELLNAME[(src, seg)]
    for a in ac.ALPHAS:
        P(f"\n--- {cell}  a={a}  (deger [lo; hi]) ---")
        for var in VARS:
            for meth in METHODS:
                far = get(cell, meth, var, a, "normal", "far")
                parts = [f"FAR={far[0]:.4f}[{far[1]:.4f};{far[2]:.4f}]"]
                for g in GROUPS:
                    v = get(cell, meth, var, a, g, "det")
                    parts.append(f"{g}={v[0]:.3f}[{v[1]:.3f};{v[2]:.3f}]")
                P(f"{var:7s} {meth:13s} " + " | ".join(parts))
for src, seg in CELLS:
    cell = CELLNAME[(src, seg)]
    P(f"\n--- AUROC {cell} ---")
    sub = DF[(DF["cell"] == cell) & (DF["metric"] == "auroc")]
    for (meth, var), g in sub.groupby(["method", "variant"], sort=False):
        P(f"{meth:13s} {var:6s} " + " | ".join(f"{r.atype}={r.value:.3f}[{r.lo:.3f};{r.hi:.3f}]" for r in g.itertuples()))
P(f"\n[{time.time()-t0:.0f}s] bitti")

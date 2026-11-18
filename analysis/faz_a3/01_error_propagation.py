"""Faz A3.1 - Hata yayilimi (DESIGN_A3.md 'A3.1'; on-kayitli, Sapmalar satiri DESIGN_A3.md sonunda).

Kol (a): analitik model (hata-kosusu uzunluk dagilimlarindan sahte ziyaret / dongu bolunme orani) vs gozlenen.
Kol (b): ToF-oracle kare etiketlerini r in {0.02,..,0.30} oraninda boz (iid / B2'den uydurulan geometrik patlama),
         donguleri BOZUK etiketlerden kes (pred segmentasyonu), D3'u bozuk normal dongulerle kalibre et, R=50 tohum.
Cikti: reports/faz_a3/01_error_propagation.{md,csv}, 01_error_propagation_seeds.csv, 01_error_propagation_arm_a.csv, *.png
Yeniden egitim yok; Faz A / A2 dosyalarina yazilmaz. Hicbir dongu-duzeyi natural_dev bilgisi uretilmez / okunmaz.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a2"))
import numpy as np
import pandas as pd

import a2_cycles as ac  # noqa: E402  (faz_a/fa_common'i fc olarak da getirir)

fc = ac.fc
OUT = ROOT / "reports" / "faz_a3"
GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
R_SEEDS = 50
ALPHA = 0.10
MODELS = ["b2", "cnngru", "cnnonly", "b1"]
MNAME = {"b2": "B2", "cnngru": "CNN-GRU", "cnnonly": "CNN-only", "b1": "B1"}
NOISES = ["iid", "bursty"]
GROUPS = {"all": ac.ATYPES, "seq135": ["seq135"], "seq145": ["seq145"], "fast": ["fast"], "slow": ["slow"],
          "seq(135+145)": ["seq135", "seq145"], "tempo(fast+slow)": ["fast", "slow"]}
BASE_SEED = 20261104
G = None          # surec-yerel onbellek (_init)


# ------------------------------------------------------------------ ortak durum
def _init():
    global G
    if G is not None:
        return G
    c = ac.ctx()
    LAB = ac.source_labels(verbose=False)
    y = {s: LAB["tof"][s].astype(int) for s in c.SESS}
    tt = {s: c.SE[s]["t"].to_numpy() for s in c.SESS}
    nref = {s: int(((c.AL["session"] == s) & c.AL["use"].isin(ac.USED)).sum()) for s in c.SESS}
    # B2 hata-kosullu karisiklik (L0 -> B2), s02-s06 tum kareleri
    M = np.zeros((6, 6))
    for s in c.SESS:
        np.add.at(M, (y[s], LAB["b2"][s]), 1)
    Q = M.copy()
    np.fill_diagonal(Q, 0.0)
    for k in range(6):
        if Q[k].sum() == 0:
            Q[k] = 1.0
            Q[k, k] = 0.0
    Q = Q / Q.sum(axis=1, keepdims=True)
    cdf = np.cumsum(Q, axis=1)
    cdf = np.minimum(cdf / cdf[:, -1:], 1.0)
    # B2 hata kosulari (p != y, oturum icinde maksimal kosu) -> ortalama uzunluk
    n_err_frames = n_err_runs = 0
    for s in c.SESS:
        e = (LAB["b2"][s] != y[s]).astype(int)
        for l, a, b in ac.runs_of(e):
            if l == 1:
                n_err_runs += 1
                n_err_frames += b - a
    ell = n_err_frames / n_err_runs
    G = dict(c=c, LAB=LAB, y=y, tt=tt, nref=nref, M=M, Q=Q, cdf=cdf, ell=ell, q=1.0 / ell)
    return G


# ------------------------------------------------------------------ bozma
def corrupt_iid(y, r, rng):
    n = len(y)
    m = rng.random(n) < r
    lab = y.copy()
    lab[m] = (y[m] + 1 + rng.integers(0, 5, size=int(m.sum()))) % 6      # dogru etiket disindaki 5 sinifdan duzgun
    return lab


def corrupt_bursty(y, r, rng):
    """Dongusel (bosluk, patlama) geometrik; patlama uzunlugu ort. ell (B2'den); durağan kapsama = r.
    Patlama basina TEK ortak U ile B2 hata-kosullu karisiklik satirindan ters-CDF etiket."""
    g = _init()
    q = g["q"]
    n = len(y)
    a = r * q / (1.0 - r)
    in_b = rng.random() < r
    bid = np.full(n, -1, dtype=np.int64)
    pos = k = 0
    while pos < n:
        L = int(rng.geometric(q if in_b else a))
        if in_b:
            bid[pos:pos + L] = k
            k += 1
        pos += L
        in_b = not in_b
    lab = y.copy()
    m = bid >= 0
    if m.any():
        Ub = 1.0 - rng.random(max(k, 1))                       # (0, 1]
        u = Ub[bid[m]]
        cd = g["cdf"][y[m]]
        new = np.minimum((cd < u[:, None]).sum(axis=1), 5)
        new = np.where(new == y[m], (y[m] + 1) % 6, new)       # (yalniz sayisal kenar durumu)
        lab[m] = new
    return lab


# ------------------------------------------------------------------ ziyaret istatistikleri (kol a ve simulasyon tanilamasi)
def visit_stats(lab_raw, y, tt):
    """debounce sonrasi pred ziyaretleri. sahte = ayni sinifli hicbir L0 karesi icermeyen; sahte Box1; Box1 parcalanmasi."""
    d = ac.debounce_labels(lab_raw)
    vs = ac.visits_of(d, tt)
    rid = np.zeros(len(y), dtype=np.int64)
    rid[1:] = np.cumsum(y[1:] != y[:-1])
    n_sp = n_sp1 = 0
    runs1 = []
    for (l, s, e, _, _) in vs:
        m = y[s:e] == l
        if not m.any():
            n_sp += 1
            n_sp1 += int(l == 1)
        elif l == 1:
            runs1.append(int(rid[s + int(np.argmax(m))]))
    frag = len(runs1) - len(set(runs1))
    return n_sp, n_sp1, frag, len(vs)


def wrong_box_runs(p, y):
    """sabit yanlis KUTU etiketli kosular: q_t = p_t if (p_t>0 and p_t != y_t) else -1; kosu (sinif, uzunluk) listesi."""
    q = np.where((p > 0) & (p != y), p, -1)
    n = len(y)
    out = []
    for l, a, b in ac.runs_of(q):
        if l > 0:
            interior = (a == 0 or y[a - 1] != l) and (b == n or y[b] != l)     # iki komsu L0 karesi de bu kutu degil
            out.append((l, b - a, bool(interior)))
    return out


# ------------------------------------------------------------------ yalin degerlendirme (Faz A evaluate'in D3 kismi)
def _auc(neg, pos):
    from sklearn.metrics import roc_auc_score
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) < 2 or len(pos) < 1:
        return float("nan")
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def eval_d3(S):
    T = S[S["role"] == "test"]
    at = T["atype"].to_numpy()
    f_ = T["f"].to_numpy()
    p3 = T["p3"].to_numpy(float)
    D1 = T["D1"].to_numpy(float)
    D2 = T["D2"].to_numpy(float)
    folds = sorted(set(f_.tolist()))
    out = {}
    per = {k: [] for k in GROUPS}
    per1, per2 = [], []
    for f in folds:
        mf = f_ == f
        neg = mf & (at == "normal")
        for k, mem in GROUPS.items():
            pos = mf & np.isin(at, mem)
            per[k].append(_auc(1.0 - p3[neg], 1.0 - p3[pos]))
        pos = mf & (at != "normal")
        per1.append(_auc(D1[neg], D1[pos]))
        per2.append(_auc(D2[neg], D2[pos]))
    for k in GROUPS:
        out["auroc_" + k] = float(np.nanmean(per[k])) if len(per[k]) else float("nan")
    out["auroc_D1_all"] = float(np.nanmean(per1)) if per1 else float("nan")
    out["auroc_D2_all"] = float(np.nanmean(per2)) if per2 else float("nan")
    nm = at == "normal"
    out["far"] = float(np.mean(p3[nm] <= ALPHA + 1e-9))
    out["n_pairs"] = int(nm.sum())
    for k, mem in GROUPS.items():
        m = np.isin(at, mem)
        out["det_" + k] = float(np.mean(p3[m] <= ALPHA + 1e-9)) if m.any() else float("nan")
    out["n_folds"] = len(folds)
    return out


def run_pipeline(lab):
    """lab: {session: etiket} -> (D3 metrikleri, dongu sayilari). Dagitilan yolun tamami: pred segmentasyonu, Faz A rolleri."""
    C, _ = ac.build_cycles("sim", "pred", lab)
    S, _ = ac.score_group(C, verbose=False)
    m = eval_d3(S)
    m["n_cyc"] = int(len(C))
    m["n_cyc_normal"] = int((C["atype"] == "normal").sum())
    return m


def sim_task(args):
    noise, tag, r, ridx, rep = args
    g = _init()
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 301, NOISES.index(noise), ridx, rep]))
    fn = corrupt_iid if noise == "iid" else corrupt_bursty
    lab = {s: fn(g["y"][s], r, rng) for s in g["c"].SESS}
    nerr = sum(int((lab[s] != g["y"][s]).sum()) for s in g["c"].SESS)
    ntot = sum(len(lab[s]) for s in g["c"].SESS)
    nsp = nsp1 = nfrag = nvis = 0
    for s in g["c"].SESS:
        a, b, cfr, nv = visit_stats(lab[s], g["y"][s], g["tt"][s])
        nsp += a
        nsp1 += b
        nfrag += cfr
        nvis += nv
    nref = sum(g["nref"].values())
    m = run_pipeline(lab)
    m.update(noise=noise, tag=tag, r=r, rep=rep, realized_r=nerr / ntot, svr=nsp / nref, csr=nsp1 / nref,
             split_alt=(nsp1 + nfrag) / nref, vis_per_tof_cycle=nvis / nref)
    return m


# ------------------------------------------------------------------ raporlama yardimcilari
LINES = []


def H(s=""):
    LINES.append(s)
    print(s, flush=True)


def md_table(df, floatfmt=None):
    cols = list(df.columns)
    H("| " + " | ".join(cols) + " |")
    H("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)) and not (c == "L0"):
                f = (floatfmt or {}).get(c, "{:.4f}")
                cells.append("nan" if np.isnan(v) else f.format(v))
            else:
                cells.append(str(v))
        H("| " + " | ".join(cells) + " |")


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    g = _init()
    c = g["c"]
    P = lambda *a: print(*a, flush=True)  # noqa: E731
    H("# Faz A3.1 - Hata yayilimi (kare dogrulugu -> dongu duzeyi anomali tespiti)")
    H()
    H("Brief: `faz_a3/DESIGN_A3.md` A3.1 (on-kayitli; ayrintilar 'Sapmalar' satirinda). **Statu: ikincil/sonradan analiz.** Birincil sonuclar Faz A'nin on-kayitli sayilaridir; "
      "ana model B2 + D3 degismez. Yeni veri/yeniden egitim yok. Gurultu **sentetiktir**; gercek hatalar gecis-yapilidir; tek operator. "
      "Her sayi `faz_a3/01_error_propagation.py` ciktisinda da basilidir.")
    H()

    # ---------------------------------------------------------------- 0) kapi: kendi hattim Faz A'yi birebir uretiyor mu?
    tab = pd.read_csv(ROOT / "reports" / "faz_a" / "04_anomaly_table.csv")
    tab = tab[(tab["variant"] == "D3") & (tab["segmentation"] == "pred")]
    gate_rows, gate_ok = [], True
    for src in ["tof"] + MODELS:
        m = run_pipeline(g["LAB"][src])
        for k, an in [("all", "tum anomali"), ("seq135", "seq135"), ("seq145", "seq145"), ("fast", "fast"), ("slow", "slow"),
                      ("seq(135+145)", "seq(135+145)"), ("tempo(fast+slow)", "tempo(fast+slow)")]:
            ref = float(tab[(tab["state_source"] == src) & (tab["atype"] == an)]["auroc_mean"].iloc[0])
            d = abs(m["auroc_" + k] - ref)
            gate_rows.append(dict(source=src, group=k, mine=m["auroc_" + k], faz_a=ref, absdiff=d))
            gate_ok &= d < 1e-9
        reff = float(tab[(tab["state_source"] == src) & (tab["atype"] == "tum anomali")]["far_a10"].iloc[0])
        gate_rows.append(dict(source=src, group="FAR@0.10", mine=m["far"], faz_a=reff, absdiff=abs(m["far"] - reff)))
        gate_ok &= abs(m["far"] - reff) < 1e-9
    GT = pd.DataFrame(gate_rows)
    H("## 0. Kapi (hattin Faz A'yi birebir uretmesi)")
    H(f"Kendi yalin degerlendirici + `build_cycles('pred')` + `score_group` hattim, bozmasiz etiketlerle (ToF = r=0 referansi ve 4 gercek kaynak) "
      f"Faz A `04_anomaly_table.csv` D3/pred satirlarini (AUROC: tum anomali + 4 tur + 2 grup; FAR@0.10) uretiyor mu? "
      f"Maks |fark| = {GT['absdiff'].max():.2e}; **{'PASS' if gate_ok else 'FAIL'}** (esik 1e-9).")
    P(GT.to_string())
    if not gate_ok:
        H("**KAPI BASARISIZ - analiz durduruldu.**")
        (OUT / "01_error_propagation.md").write_text("\n".join(LINES), encoding="utf-8")
        sys.exit(2)
    ref0 = run_pipeline(g["LAB"]["tof"])
    H()
    H(f"r = 0 referans (ToF oracle, bozmasiz; ihtiyac duyulan ust sinir): D3 AUROC(tum) = {ref0['auroc_all']:.4f}, FAR@0.10 = {ref0['far']:.4f} "
      f"(n_pairs = {ref0['n_pairs']}).")
    H()

    # ---------------------------------------------------------------- kare hata orani tanimi
    n_tot = sum(len(g["y"][s]) for s in c.SESS)
    rate = {}
    rate_win = {}
    for m in MODELS:
        ne = sum(int((g["LAB"][m][s] != g["y"][s]).sum()) for s in c.SESS)
        rate[m] = ne / n_tot
        nw = sum(len(g["y"][s]) - 4 for s in c.SESS)
        ne_w = sum(int((g["LAB"][m][s][4:] != g["y"][s][4:]).sum()) for s in c.SESS)
        rate_win[m] = ne_w / nw
    H("## 1. 'Kare hata orani' tanimi (overlay ve kol b ekseni)")
    H(f"Kaynak m icin **kare hata orani = #{{kare: OOF y_pred != L0 (frame_table.y_true)}} / #kare**, s02-s06'nin TUM {n_tot} kareleri uzerinden "
      f"(oturum basina pencere-siz ilk 4 kare `source_labels` geregi none tahmin edilir ve L0 != none ise hata sayilir; yalniz pencereli kareler uzerinden hesap asagida karsilastirma icin). "
      "Simulasyonda r ise ayni tanimda 'bozulan kare payi' (gerceklesen oran tohum basina olculur).")
    md_table(pd.DataFrame([dict(kaynak=MNAME[m], hata_orani=rate[m], hata_orani_yalniz_pencereli=rate_win[m]) for m in MODELS]))
    H()

    # ---------------------------------------------------------------- hata-kosusu dagilimlari
    H("## 2. Hata-kosusu uzunluk dagilimlari (S1 OOF, s02-s06)")
    H("Hata kosusu = oturum icinde ardisik kareler boyunca y_pred != L0 olan maksimal kosu. 'sabit yanlis etiket (>=2)' = uzunlugu >= 2 olan hata kosularinin tek bir yanlis etiket tasiyanlarinin payi.")
    rows = []
    for m in MODELS:
        Ls, const = [], []
        for s in c.SESS:
            p, y = g["LAB"][m][s], g["y"][s]
            for l, a, b in ac.runs_of((p != y).astype(int)):
                if l == 1:
                    Ls.append(b - a)
                    const.append(len(set(p[a:b].tolist())) == 1)
        Ls, const = np.array(Ls), np.array(const)
        rows.append(dict(kaynak=MNAME[m], n_kosu=len(Ls), ort_uzunluk=Ls.mean(), L1=int((Ls == 1).sum()), L2=int((Ls == 2).sum()),
                         L3=int((Ls == 3).sum()), L4_5=int(((Ls >= 4) & (Ls <= 5)).sum()), L6_10=int(((Ls >= 6) & (Ls <= 10)).sum()),
                         L_gt10=int((Ls > 10).sum()), pay_L_ge2=float((Ls >= 2).mean()), sabit_yanlis_etiket_ge2=float(const[Ls >= 2].mean()),
                         maks=int(Ls.max())))
    md_table(pd.DataFrame(rows), {"ort_uzunluk": "{:.4f}", "pay_L_ge2": "{:.4f}", "sabit_yanlis_etiket_ge2": "{:.4f}"})
    H()
    H(f"**B2'den (YALNIZ B2) uydurulan geometrik patlama: ortalama uzunluk ell = {g['ell']:.4f} kare, kosuldan ayrilma olasiligi q = 1/ell = {g['q']:.4f}.** "
      f"Diger kaynaklar uydurmada kullanilmaz; yalniz sinamada yer alir. Geometrik modelde P(L>=2) = 1-q = {1-g['q']:.3f}; B2 gercek P(L>=2) = {rows[0]['pay_L_ge2']:.3f}.")
    H(f"B2 hata-kosullu karisiklik satirlari Q[y, :] (L0 sinifi -> yedek etiket olasiligi, diag = 0), sinif 0..5:")
    P(np.array2string(g["Q"], precision=3))
    md_table(pd.DataFrame(g["Q"], columns=[f"->{k}" for k in range(6)]).assign(L0=[str(k) for k in range(6)])[["L0"] + [f"->{k}" for k in range(6)]],
             {f"->{k}": "{:.3f}" for k in range(6)})
    H()

    ncs = {s_: c.SE[s_]["near_change"].to_numpy() for s_ in c.SESS}
    base_nc = float(np.mean(np.concatenate([ncs[s_] for s_ in c.SESS])))
    H("Betimleyici (post-hoc, Sapmalar 2. satir): hata karelerinin L0 gecisine yakinligi (`near_change` = L0 degisim sinirinin 2 kare oncesi + 2 kare sonrasi). "
      f"Tum karelerde near_change payi = {base_nc:.4f}.")
    ncrow = []
    for m in MODELS:
        e_all = np.concatenate([g["LAB"][m][s_] != g["y"][s_] for s_ in c.SESS])
        nc_all = np.concatenate([ncs[s_] for s_ in c.SESS])
        ncrow.append(dict(kaynak=MNAME[m], hata_karesi=int(e_all.sum()), hata_karesi_near_change_payi=float(nc_all[e_all].mean()),
                          P_hata_given_near_change=float(e_all[nc_all].mean()), P_hata_given_uzak=float(e_all[~nc_all].mean())))
    md_table(pd.DataFrame(ncrow))
    NC = pd.DataFrame(ncrow)
    H()

    # ---------------------------------------------------------------- KOL (a)
    H("## 3. Kol (a) - analitik model vs gozlenen (4 kaynak x 5 oturum)")
    H("**Tanimlar (Sapmalar satiri):** oranlar = olay sayisi / oturumun ToF referans dongu sayisi `n_tof_ref`. **Gozlenen (birincil)**: debounce sonrasi pred ziyaretlerinden, ayni siniftan hicbir L0 karesi icermeyenler = *sahte ziyaret*; "
      "bunlarin Box1 olanlari = *dongu bolunmesi*. **Analitik tahmin**: kaynak m icin havuzlanmis (s02-s06) 'sabit yanlis kutu etiketli, uzunluk >= 2' kosu uzunlugu dagilimindan kare basina tehlike x "
      "oturumun kare sayisi / `n_tof_ref` (sahte ziyaret icin tum kutular, bolunme icin yalniz Box1; debounce'un 1-karelik bosluk birlestirmesi modellenmez). LOSO: tehlike o oturum haric dort oturumdan. "
      "**Gecme (sik)**: dort kaynagin hepsinde, tum oturumlarda ve iki oranda max |tahmin - gozlenen| <= 0.05 (birincil gozlenen, havuzlanmis tehlike). Ikincil gozlenen: `04_cycle_counts.csv` `n_loops_with_2plus/n_tof_ref` "
      "(ToF dongusu basina >= 2 pred donguye dusen oran; sinir kaymasi dahil).")
    cnt = pd.read_csv(ROOT / "reports" / "faz_a" / "04_cycle_counts.csv")
    cnt = cnt[(cnt["segmentation"] == "pred") & cnt["state_source"].isin(MODELS)]
    ra = []
    for m in MODELS:
        runs = {s: wrong_box_runs(g["LAB"][m][s], g["y"][s]) for s in c.SESS}
        nfr = {s: len(g["y"][s]) for s in c.SESS}
        for s in c.SESS:
            cr = cnt[(cnt["state_source"] == m) & (cnt["session"] == s)].iloc[0]
            assert int(cr["n_tof_ref"]) == g["nref"][s], (m, s)
            C_s = g["nref"][s]
            sp, sp1, frag, nv = visit_stats(g["LAB"][m][s], g["y"][s], g["tt"][s])

            def haz(sessions, only1, interior=False):
                num = sum(1 for ss in sessions for (l, L, ic) in runs[ss] if L >= 2 and (l == 1 or not only1) and (ic or not interior))
                return num / sum(nfr[ss] for ss in sessions)

            others = [x for x in c.SESS if x != s]
            raw_sp = sum(1 for (l, L, ic) in runs[s] if L >= 2)
            raw_sp1 = sum(1 for (l, L, ic) in runs[s] if L >= 2 and l == 1)
            ra.append(dict(kaynak=MNAME[m], oturum=s.replace("session_", "s")[:3], n_tof_ref=C_s,
                           gz_sahte=sp / C_s, tah_sahte=haz(c.SESS, False) * nfr[s] / C_s, tah_sahte_loso=haz(others, False) * nfr[s] / C_s,
                           ham_sahte=raw_sp / C_s,
                           gz_bol=sp1 / C_s, tah_bol=haz(c.SESS, True) * nfr[s] / C_s, tah_bol_loso=haz(others, True) * nfr[s] / C_s,
                           ham_bol=raw_sp1 / C_s,
                           tah_sahte_ic=haz(c.SESS, False, True) * nfr[s] / C_s, tah_bol_ic=haz(c.SESS, True, True) * nfr[s] / C_s,
                           gz_bol_parca=(sp1 + frag) / C_s, gz_bol_sayim=int(cr["n_loops_with_2plus"]) / C_s,
                           gz_net_giris=(float(cr["n_pred_box1_entries"]) - C_s) / C_s))
    RA = pd.DataFrame(ra)
    RA["fark_sahte"] = RA["tah_sahte"] - RA["gz_sahte"]
    RA["fark_bol"] = RA["tah_bol"] - RA["gz_bol"]
    RA["fark_sahte_loso"] = RA["tah_sahte_loso"] - RA["gz_sahte"]
    RA["fark_bol_loso"] = RA["tah_bol_loso"] - RA["gz_bol"]
    RA["fark_bol_sayim"] = RA["tah_bol"] - RA["gz_bol_sayim"]
    RA["fark_sahte_ic"] = RA["tah_sahte_ic"] - RA["gz_sahte"]
    RA["fark_bol_ic"] = RA["tah_bol_ic"] - RA["gz_bol"]
    RA.to_csv(OUT / "01_error_propagation_arm_a.csv", index=False)
    md_table(RA[["kaynak", "oturum", "n_tof_ref", "gz_sahte", "tah_sahte", "fark_sahte", "tah_sahte_loso", "ham_sahte", "gz_bol", "tah_bol", "fark_bol",
                 "tah_bol_loso", "ham_bol"]])
    H()
    H("**POST-HOC ikincil varyant (Sapmalar 2. satir): 'ic patlama' analitik tahmini** (yanlis-kutu kosusunun iki komsu L0 karesi de o kutu degil -> sinir kaymasi dislanir); ayni gozlenen, ayni 0.05 olcutu:")
    md_table(RA[["kaynak", "oturum", "gz_sahte", "tah_sahte_ic", "fark_sahte_ic", "gz_bol", "tah_bol_ic", "fark_bol_ic"]])
    H()
    H("Ikincil gozlenen dongu-bolunme olculeri (ayni oturumlar):")
    md_table(RA[["kaynak", "oturum", "gz_bol", "gz_bol_parca", "gz_bol_sayim", "gz_net_giris", "tah_bol", "fark_bol_sayim"]])
    H()
    verdict = []
    for m in MODELS:
        d = RA[RA["kaynak"] == MNAME[m]]
        mx = max(d["fark_sahte"].abs().max(), d["fark_bol"].abs().max())
        mx_l = max(d["fark_sahte_loso"].abs().max(), d["fark_bol_loso"].abs().max())
        mx_c = d["fark_bol_sayim"].abs().max()
        # havuz (oturumlar toplami)
        ps = (d["tah_sahte"] * d["n_tof_ref"]).sum() / d["n_tof_ref"].sum() - (d["gz_sahte"] * d["n_tof_ref"]).sum() / d["n_tof_ref"].sum()
        pb = (d["tah_bol"] * d["n_tof_ref"]).sum() / d["n_tof_ref"].sum() - (d["gz_bol"] * d["n_tof_ref"]).sum() / d["n_tof_ref"].sum()
        mx_ic = max(d["fark_sahte_ic"].abs().max(), d["fark_bol_ic"].abs().max())
        verdict.append(dict(kaynak=MNAME[m], maks_abs_fark_ic_patlama=mx_ic, gecti_ic=bool(mx_ic <= 0.05), maks_abs_fark_havuz_tehlike=mx, gecti_sik=bool(mx <= 0.05), maks_abs_fark_LOSO=mx_l,
                            gecti_LOSO=bool(mx_l <= 0.05), maks_abs_fark_bolunme_vs_sayim=mx_c, gecti_sayim=bool(mx_c <= 0.05),
                            havuz_fark_sahte=ps, havuz_fark_bolunme=pb))
    V = pd.DataFrame(verdict)
    md_table(V)
    H()
    H("Betimleyici (post-hoc): s05'te gozlenen sahte ziyaretlerin sinif dagilimi (A2.3: s05 L0 etiketinde Box5 silinmis; sahte sayilan Box5 ziyaretlerinin onemli kismi bu artefakttan gelebilir):")
    s05 = [x for x in c.SESS if x.startswith("session_05")][0]
    rows5 = []
    for m in MODELS:
        d_ = ac.debounce_labels(g["LAB"][m][s05])
        yy = g["y"][s05]
        tot = {k: 0 for k in range(1, 6)}
        for (l, a_, e_, _, _) in ac.visits_of(d_, g["tt"][s05]):
            if not (yy[a_:e_] == l).any():
                tot[l] += 1
        rows5.append(dict(kaynak=MNAME[m], sahte_toplam=sum(tot.values()), **{f"Box{k}": v for k, v in tot.items()}))
    md_table(pd.DataFrame(rows5))
    armA_pass = bool(V["gecti_sik"].all())
    H()
    H(f"**Kol (a) karari (on-kayitli sik kural; birincil gozlenen; havuzlanmis tehlike): {'PASS' if armA_pass else 'FAIL'}** "
      f"(gecen kaynaklar: {', '.join(MNAME[m] for m, p in zip(MODELS, V['gecti_sik']) if p) or 'yok'}; "
      f"kalan: {', '.join(MNAME[m] for m, p in zip(MODELS, V['gecti_sik']) if not p) or 'yok'}). "
      f"LOSO tehlike ile: {'PASS' if V['gecti_LOSO'].all() else 'FAIL'}; post-hoc 'ic patlama' varyantiyla (ikincil): {'PASS' if V['gecti_ic'].all() else 'FAIL'} (gecen: {', '.join(MNAME[m] for m, p in zip(MODELS, V['gecti_ic']) if p) or 'yok'}); ikincil (sayim-tabanli bolunme) ile: {'PASS' if V['gecti_sayim'].all() else 'FAIL'}.")
    H()

    # ---------------------------------------------------------------- KOL (b)
    H("## 4. Kol (b) - simulasyon: tum dagitilan yol bozulur")
    tasks = []
    for ni, noise in enumerate(NOISES):
        for ri, r in enumerate(GRID):
            for rep in range(R_SEEDS):
                tasks.append((noise, "grid", r, ri, rep))
        for mi, m in enumerate(MODELS):
            for rep in range(R_SEEDS):
                tasks.append((noise, "obs_" + m, rate[m], 100 + mi, rep))
    P(f"[{time.time()-t0:.0f}s] {len(tasks)} simulasyon, 3 isci surec")
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(3, initializer=_init) as pool:
        res = pool.map(sim_task, tasks, chunksize=10)
    SD = pd.DataFrame(res)
    SD.to_csv(OUT / "01_error_propagation_seeds.csv", index=False)
    P(f"[{time.time()-t0:.0f}s] simulasyon bitti")
    skipped = int((SD["n_folds"] < 5).sum())
    H(f"Toplam {len(SD)} simulasyon (6 r x 2 tur x {R_SEEDS} tohum + {len(MODELS)} gozlenen-r noktasi x 2 tur x {R_SEEDS}); atlanan kat (<3 egitim/kal. dongusu) olan simulasyon sayisi: {skipped}; "
      f"min kat sayisi = {int(SD['n_folds'].min())}.")
    H()

    metric_cols = [c_ for c_ in SD.columns if c_.startswith("auroc_") or c_.startswith("det_")] + ["far", "n_pairs", "realized_r", "svr", "csr", "split_alt",
                                                                                                      "vis_per_tof_cycle", "n_cyc", "n_cyc_normal"]
    rows = []
    for (noise, tag, r), d in SD.groupby(["noise", "tag", "r"], sort=False):
        for mc in metric_cols:
            x = d[mc].to_numpy(float)
            rows.append(dict(noise=noise, tag=tag, r=r, metric=mc, mean=float(np.nanmean(x)), lo=float(np.nanpercentile(x, 2.5)),
                             hi=float(np.nanpercentile(x, 97.5)), sd=float(np.nanstd(x, ddof=1)), n_seeds=int(len(x))))
    AG = pd.DataFrame(rows)
    AG.to_csv(OUT / "01_error_propagation.csv", index=False)

    def agg(noise, tag, r, metric, what="mean"):
        return float(AG[(AG["noise"] == noise) & (AG["tag"] == tag) & (np.isclose(AG["r"], r)) & (AG["metric"] == metric)][what].iloc[0])

    # --- gerceklesen oran
    H("### 4.1 Gerceklesen kare hata orani (tohumlar uzerinden ortalama) ve dongu yapisi")
    rr = []
    for noise in NOISES:
        for r in GRID:
            rr.append(dict(tur=noise, r=r, gerceklesen_r=agg(noise, "grid", r, "realized_r"), dongu_sayisi=agg(noise, "grid", r, "n_cyc"),
                           normal_dongu=agg(noise, "grid", r, "n_cyc_normal"), ziyaret_per_ToF_dongu=agg(noise, "grid", r, "vis_per_tof_cycle"),
                           sahte_ziyaret_per_ToF_dongu=agg(noise, "grid", r, "svr"), bolunme_per_ToF_dongu=agg(noise, "grid", r, "csr")))
    md_table(pd.DataFrame(rr), {"r": "{:.2f}"})
    H()

    # --- ana tablo: AUROC / FAR vs r
    H("### 4.2 D3 AUROC (tum anomali) ve normal FAR@0.10 vs r (50 tohum; aralik = tohumlar arasi 2.5-97.5 yuzdelik)")
    rows = []
    for noise in NOISES:
        for r in GRID:
            rows.append(dict(tur=noise, r=r, AUROC=agg(noise, "grid", r, "auroc_all"), AUROC_lo=agg(noise, "grid", r, "auroc_all", "lo"),
                             AUROC_hi=agg(noise, "grid", r, "auroc_all", "hi"), FAR=agg(noise, "grid", r, "far"), FAR_lo=agg(noise, "grid", r, "far", "lo"),
                             FAR_hi=agg(noise, "grid", r, "far", "hi"), n_pairs=agg(noise, "grid", r, "n_pairs")))
    T42 = pd.DataFrame(rows)
    md_table(T42, {"r": "{:.2f}", "n_pairs": "{:.1f}"})
    H()
    H("### 4.3 Anomali turu basina D3 AUROC (ortalama [2.5, 97.5])")
    rows = []
    for noise in NOISES:
        for r in GRID:
            row = dict(tur=noise, r=r)
            for k in GROUPS:
                row[k] = f"{agg(noise, 'grid', r, 'auroc_' + k):.3f} [{agg(noise, 'grid', r, 'auroc_' + k, 'lo'):.3f}, {agg(noise, 'grid', r, 'auroc_' + k, 'hi'):.3f}]"
            rows.append(row)
    md_table(pd.DataFrame(rows), {"r": "{:.2f}"})
    H()
    H("### 4.4 Ek tanilama: kanal bazli ham AUROC (D1 dizi, D2 sure; tum anomali) ve D3 aksiyon orani (alfa=0.10)")
    rows = []
    for noise in NOISES:
        for r in GRID:
            rows.append(dict(tur=noise, r=r, D1_AUROC=agg(noise, "grid", r, "auroc_D1_all"), D2_AUROC=agg(noise, "grid", r, "auroc_D2_all"),
                             det_all=agg(noise, "grid", r, "det_all"), det_seq=agg(noise, "grid", r, "det_seq(135+145)"),
                             det_tempo=agg(noise, "grid", r, "det_tempo(fast+slow)")))
    md_table(pd.DataFrame(rows), {"r": "{:.2f}"})
    H()

    # --- karar kurali ve bant
    H("### 4.5 'Yeterli dogruluk' kurali ve bant (on-kayitli)")
    H("Kural: AUROC(tum) >= 0.90 **ve** FAR@0.10 <= alfa + 2SE (SE = sqrt(alfa(1-alfa)/n_pairs), n_pairs o r icin ortalama normal test cifti). "
      "r* = kurali saglayan en buyuk izgara r (izgara: 0.02 ... 0.30). Bant: nokta (ortalama AUROC) / alt sinir (AUROC tohum-2.5 yuzdeligi) / ust sinir (97.5 yuzdelik); FAR kosulu hepsinde ortalama FAR ile.")
    rstar = {}
    brows = []
    for noise in NOISES:
        pts = {}
        for key, what in [("nokta", "mean"), ("alt", "lo"), ("ust", "hi")]:
            ok = []
            for r in GRID:
                a = agg(noise, "grid", r, "auroc_all", what)
                far = agg(noise, "grid", r, "far")
                se = np.sqrt(ALPHA * (1 - ALPHA) / agg(noise, "grid", r, "n_pairs"))
                ok.append((a >= 0.90) and (far <= ALPHA + 2 * se))
            pts[key] = max([r for r, o in zip(GRID, ok) if o], default=0.0)
            pts[key + "_ok"] = ok
        rstar[noise] = pts
        brows.append(dict(tur=noise, r_star_nokta=pts["nokta"], r_star_alt=pts["alt"], r_star_ust=pts["ust"],
                          saglayan_r_nokta=",".join(f"{r:.2f}" for r, o in zip(GRID, pts["nokta_ok"]) if o) or "yok",
                          monoton=bool(all(pts["nokta_ok"][i] >= pts["nokta_ok"][i + 1] for i in range(len(GRID) - 1)))))
    md_table(pd.DataFrame(brows), {"r_star_nokta": "{:.2f}", "r_star_alt": "{:.2f}", "r_star_ust": "{:.2f}"})
    H("(r* = 0.00 -> izgaranin en kucugu 0.02'de bile kural saglanmiyor.)")
    H()
    for noise in NOISES:
        pts = rstar[noise]
        H(f"- **{noise}: yeterli dogruluk bandi r in [{pts['alt']:.2f} (alt), {pts['nokta']:.2f} (nokta), {pts['ust']:.2f} (ust)]** "
          f"(kare hata orani; referans r=0 AUROC {ref0['auroc_all']:.3f}, FAR {ref0['far']:.3f}).")
    H()

    # --- tahmin 1
    H("### 4.6 On-yazili tahmin 1: patlamali gurultu iid'den daha dusuk r'de bozar")
    diff_rows = []
    for r in GRID:
        ai, ab = agg("iid", "grid", r, "auroc_all"), agg("bursty", "grid", r, "auroc_all")
        di = SD[(SD["noise"] == "iid") & (SD["tag"] == "grid") & np.isclose(SD["r"], r)].sort_values("rep")["auroc_all"].to_numpy()
        db = SD[(SD["noise"] == "bursty") & (SD["tag"] == "grid") & np.isclose(SD["r"], r)].sort_values("rep")["auroc_all"].to_numpy()
        dd = db - di      # eslesmemis tohumlar; fark dagilimi icin yalniz ozet
        diff_rows.append(dict(r=r, AUROC_iid=ai, AUROC_bursty=ab, fark_bursty_eksi_iid=ab - ai,
                              FAR_iid=agg("iid", "grid", r, "far"), FAR_bursty=agg("bursty", "grid", r, "far"),
                              bursty_dusuk=bool(ab < ai), tohum_sd_iid=di.std(ddof=1), tohum_sd_bursty=db.std(ddof=1),
                              fark_MC_SE=float(np.sqrt(di.var(ddof=1) / len(di) + db.var(ddof=1) / len(db)))))
    DF = pd.DataFrame(diff_rows)
    md_table(DF, {"r": "{:.2f}", "fark_bursty_eksi_iid": "{:+.4f}"})
    p1 = bool(rstar["bursty"]["nokta"] < rstar["iid"]["nokta"])
    H()
    H(f"r*(nokta): patlamali {rstar['bursty']['nokta']:.2f} vs iid {rstar['iid']['nokta']:.2f}. On-kayitli olcut (r*_patlamali < r*_iid): **{'GECTI' if p1 else 'GECMEDI'}**. "
      f"Ek: AUROC_patlamali < AUROC_iid {int(DF['bursty_dusuk'].sum())}/{len(DF)} izgara noktasinda; ortalama fark {DF['fark_bursty_eksi_iid'].mean():+.4f}; "
      f"fark MC SE'leri {DF['fark_MC_SE'].min():.4f}-{DF['fark_MC_SE'].max():.4f}.")
    H()

    # --- tahmin 2
    H("### 4.7 On-yazili tahmin 2: gercek CNN-GRU / CNN-only noktalari, B2'den uydurulan patlamali egrinin 0.03 AUROC yakininda")
    real = {}
    for m in MODELS:
        row = tab[(tab["state_source"] == m) & (tab["atype"] == "tum anomali")].iloc[0]
        real[m] = dict(auroc=float(row["auroc_mean"]), far=float(row["far_a10"]))
    curve = [agg("bursty", "grid", r, "auroc_all") for r in GRID]
    curve_i = [agg("iid", "grid", r, "auroc_all") for r in GRID]
    p2rows = []
    for m in MODELS:
        ro = rate[m]
        inside = GRID[0] <= ro <= GRID[-1]
        cb = float(np.interp(ro, GRID, curve)) if inside else float("nan")
        ci = float(np.interp(ro, GRID, curve_i)) if inside else float("nan")
        ddir = agg("bursty", "obs_" + m, ro, "auroc_all")
        ddir_i = agg("iid", "obs_" + m, ro, "auroc_all")
        p2rows.append(dict(kaynak=MNAME[m], rol=("uydurma-kaynagi (kendi kendine kontrol, tahmin degil)" if m == "b2" else
                                                ("TAHMIN" if m in ("cnngru", "cnnonly") else "izgara disi, bilgi")),
                           r_gozlenen=ro, AUROC_gercek=real[m]["auroc"], egri_enterpolasyon=cb, fark=real[m]["auroc"] - cb if inside else float("nan"),
                           abs_fark_le_003=("-" if not inside else str(abs(real[m]["auroc"] - cb) <= 0.03)),
                           dogrudan_bursty_r_obs=ddir, fark_dogrudan=real[m]["auroc"] - ddir, iid_enterp=ci, dogrudan_iid_r_obs=ddir_i,
                           FAR_gercek=real[m]["far"], FAR_sim_bursty=agg("bursty", "obs_" + m, ro, "far")))
    P2 = pd.DataFrame(p2rows)
    md_table(P2, {"r_gozlenen": "{:.4f}", "fark": "{:+.4f}", "fark_dogrudan": "{:+.4f}"})
    p2_g = {m: bool(abs(real[m]["auroc"] - float(np.interp(rate[m], GRID, curve))) <= 0.03) for m in ("cnngru", "cnnonly")}
    p2_d = {m: bool(abs(real[m]["auroc"] - agg("bursty", "obs_" + m, rate[m], "auroc_all")) <= 0.03) for m in ("cnngru", "cnnonly")}
    p2 = all(p2_g.values())
    H()
    H(f"Tahmin 2 (izgara enterpolasyonu; on-kayitli): CNN-GRU {'GECTI' if p2_g['cnngru'] else 'GECMEDI'}, CNN-only {'GECTI' if p2_g['cnnonly'] else 'GECMEDI'} -> **{'GECTI' if p2 else 'GECMEDI'}** "
      f"(her ikisi de gerekli). Dogrudan r_obs simulasyonu ile: CNN-GRU {'GECTI' if p2_d['cnngru'] else 'GECMEDI'}, CNN-only {'GECTI' if p2_d['cnnonly'] else 'GECMEDI'}. "
      "B1 (r=0.376) izgaranin (0.30) disindadir; bursty/iid dogrudan r_obs simulasyonu tabloda. B2 satiri uydurma kaynagi oldugu icin tahmin degil, kendi kendine tutarlilik kontroludur.")
    H()
    H("Simulasyon gercekci mi? (dongu yapisi tanilamasi; gercek kaynak vs ayni r'de patlamali / iid simulasyon; ToF dongusu basina havuzlanmis oranlar)")
    rows = []
    for m in MODELS:
        nref = sum(g["nref"].values())
        sp = sp1 = frag = nv = 0
        for s in c.SESS:
            a, b, cc, n_ = visit_stats(g["LAB"][m][s], g["y"][s], g["tt"][s])
            sp += a
            sp1 += b
            frag += cc
            nv += n_
        rows.append(dict(kaynak=MNAME[m], r=rate[m], sahte_gercek=sp / nref, sahte_bursty=agg("bursty", "obs_" + m, rate[m], "svr"),
                         sahte_iid=agg("iid", "obs_" + m, rate[m], "svr"), bol_gercek=sp1 / nref, bol_bursty=agg("bursty", "obs_" + m, rate[m], "csr"),
                         bol_iid=agg("iid", "obs_" + m, rate[m], "csr"), ziyaret_gercek=nv / nref,
                         ziyaret_bursty=agg("bursty", "obs_" + m, rate[m], "vis_per_tof_cycle"), ziyaret_iid=agg("iid", "obs_" + m, rate[m], "vis_per_tof_cycle")))
    md_table(pd.DataFrame(rows), {"r": "{:.4f}"})
    H()

    # ---------------------------------------------------------------- sekil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    col = {"iid": "#4C72B0", "bursty": "#DD8452"}
    xs = [0.0] + GRID
    for noise in NOISES:
        mu = [ref0["auroc_all"]] + [agg(noise, "grid", r, "auroc_all") for r in GRID]
        lo = [ref0["auroc_all"]] + [agg(noise, "grid", r, "auroc_all", "lo") for r in GRID]
        hi = [ref0["auroc_all"]] + [agg(noise, "grid", r, "auroc_all", "hi") for r in GRID]
        ax[0].plot(xs, mu, "-o", color=col[noise], label=f"simulasyon: {noise}", ms=4)
        ax[0].fill_between(xs, lo, hi, color=col[noise], alpha=0.2)
        mu = [ref0["far"]] + [agg(noise, "grid", r, "far") for r in GRID]
        lo = [ref0["far"]] + [agg(noise, "grid", r, "far", "lo") for r in GRID]
        hi = [ref0["far"]] + [agg(noise, "grid", r, "far", "hi") for r in GRID]
        ax[1].plot(xs, mu, "-o", color=col[noise], label=f"simulasyon: {noise}", ms=4)
        ax[1].fill_between(xs, lo, hi, color=col[noise], alpha=0.2)
    mk = {"b2": ("s", "k"), "cnngru": ("D", "#2ca02c"), "cnnonly": ("^", "#d62728"), "b1": ("v", "#7f7f7f")}
    for m in MODELS:
        ax[0].scatter([rate[m]], [real[m]["auroc"]], marker=mk[m][0], c=mk[m][1], s=70, zorder=5, label=f"gercek {MNAME[m]} (r={rate[m]:.3f})")
        ax[1].scatter([rate[m]], [real[m]["far"]], marker=mk[m][0], c=mk[m][1], s=70, zorder=5, label=f"gercek {MNAME[m]}")
    ax[0].axhline(0.90, color="k", ls=":", lw=1)
    ax[0].text(0.001, 0.905, "0.90", fontsize=8)
    ax[0].set_xlabel("kare hata orani r (L0'a gore)")
    ax[0].set_ylabel("D3 AUROC (tum anomali, kat ortalamasi)")
    ax[0].set_title("AUROC vs r (r=0: ToF oracle)")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].grid(alpha=0.3)
    ax[1].axhline(ALPHA, color="k", ls="--", lw=1)
    se_ = np.sqrt(ALPHA * (1 - ALPHA) / float(np.mean([agg("iid", "grid", r, "n_pairs") for r in GRID])))
    ax[1].axhline(ALPHA + 2 * se_, color="k", ls=":", lw=1)
    ax[1].text(0.001, ALPHA + 0.003, "alfa=0.10", fontsize=8)
    ax[1].text(0.30, ALPHA + 2 * se_ + 0.003, "alfa+2SE", fontsize=8)
    ax[1].set_ylim(-0.005, 0.25)
    ax[1].set_xlabel("kare hata orani r (L0'a gore)")
    ax[1].set_ylabel("normal FAR @ alfa=0.10")
    ax[1].set_title("Normal yanlis alarm vs r")
    ax[1].legend(fontsize=7, loc="upper left", ncol=2)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "01_error_propagation.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for a_, k in zip(axs, ["seq135", "seq145", "fast", "slow"]):
        for noise in NOISES:
            mu = [agg(noise, "grid", r, "auroc_" + k) for r in GRID]
            lo = [agg(noise, "grid", r, "auroc_" + k, "lo") for r in GRID]
            hi = [agg(noise, "grid", r, "auroc_" + k, "hi") for r in GRID]
            a_.plot(GRID, mu, "-o", color=col[noise], ms=3, label=noise)
            a_.fill_between(GRID, lo, hi, color=col[noise], alpha=0.2)
        ra_ = tab[(tab["state_source"] == "tof") & (tab["atype"] == k)]["auroc_mean"].iloc[0]
        a_.scatter([0.0], [ra_], c="k", s=20, zorder=5)
        for m in MODELS:
            v = float(tab[(tab["state_source"] == m) & (tab["atype"] == k)]["auroc_mean"].iloc[0])
            a_.scatter([rate[m]], [v], marker=mk[m][0], c=mk[m][1], s=45, zorder=5)
        a_.axhline(0.90, color="k", ls=":", lw=1)
        a_.set_title(k)
        a_.set_xlabel("r")
        a_.grid(alpha=0.3)
    axs[0].set_ylabel("D3 AUROC")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "01_error_propagation_pertype.png", dpi=150)
    plt.close(fig)
    H("Sekiller: `01_error_propagation.png` (AUROC / FAR vs r, iid + patlamali, gercek 4 kaynak overlay), `01_error_propagation_pertype.png` (tur basina AUROC). "
      "Veri: `01_error_propagation.csv` (ozet; tohumlar arasi ortalama/2.5/97.5/sd), `01_error_propagation_seeds.csv` (tohum basina), `01_error_propagation_arm_a.csv` (kol a).")
    H()

    # ---------------------------------------------------------------- yorum
    pz = rstar["bursty"]
    pi = rstar["iid"]
    H("## 5. Ozet ve desteklenen cumle")
    H(f"- **Kol (a):** {'PASS' if armA_pass else 'FAIL'} (sik kural). Ayrinti 3. bolumde; kaynak bazinda en buyuk |tahmin-gozlenen| = " +
      ", ".join(f"{MNAME[m]} {v:.3f}" for m, v in zip(MODELS, V["maks_abs_fark_havuz_tehlike"])) + ".")
    H(f"- **Kol (b) bant (D3, tum anomali):** iid r* = {pi['nokta']:.2f} [{pi['alt']:.2f}, {pi['ust']:.2f}], patlamali r* = {pz['nokta']:.2f} [{pz['alt']:.2f}, {pz['ust']:.2f}] "
      f"(r=0: AUROC {ref0['auroc_all']:.3f}). Gercek B2 (r={rate['b2']:.3f}) AUROC {real['b2']['auroc']:.3f}, CNN-GRU (r={rate['cnngru']:.3f}) {real['cnngru']['auroc']:.3f}, "
      f"CNN-only (r={rate['cnnonly']:.3f}) {real['cnnonly']['auroc']:.3f}, B1 (r={rate['b1']:.3f}) {real['b1']['auroc']:.3f}.")
    H(f"- **Tahmin 1:** {'GECTI' if p1 else 'GECMEDI'}. **Tahmin 2:** {'GECTI' if p2 else 'GECMEDI'} (izgara enterpolasyonu).")
    H(f"- **Desteklenen cumle (veriye gore, sentetik gurultu icin):** kalibre edilmis D3 dedektoru, kare hata orani iid gurultude ~%{pi['nokta']*100:.0f} (bant %{pi['alt']*100:.0f}-%{pi['ust']*100:.0f}), "
      f"B2-uydurmali patlamali gurultude ~%{pz['nokta']*100:.0f} (bant %{pz['alt']*100:.0f}-%{pz['ust']*100:.0f}; 0 = 0.02'de bile AUROC 0.90'in altinda) seviyesine kadar AUROC >= 0.90 saglar ve FAR'i korur; "
      f"hatanin zaman yapisi (patlama vs bagimsiz) orani kadar onemlidir (ayni r'de AUROC farki {DF['fark_bursty_eksi_iid'].min():+.3f} .. {DF['fark_bursty_eksi_iid'].max():+.3f}). "
      f"Ancak gercek B2 %{rate['b2']*100:.1f} hata oraninda AUROC {real['b2']['auroc']:.3f} verdigi icin sentetik bant gercek hata yapisi icin karamsardir; '~%X' tek sayisi bu analizden gercek hatalar icin cikarilamaz.")
    H()
    rr_b2 = next(r_ for r_ in rows if r_["kaynak"] == "B2")
    H("**Yorum notu (sayilar yukaridaki tablolardan).** (i) Sentetik gurultu, ayni kare hata oraninda gercek hatalardan cok daha zarar vericidir: "
      f"B2 r = {rate['b2']:.3f}'te gercek B2 AUROC {real['b2']['auroc']:.3f}, ayni r'de B2-uydurmali patlamali simulasyon {agg('bursty', 'obs_b2', rate['b2'], 'auroc_all'):.3f}, iid {agg('iid', 'obs_b2', rate['b2'], 'auroc_all'):.3f}; "
      f"sahte ziyaret / ToF dongusu: gercek {rr_b2['sahte_gercek']:.3f}, patlamali sim {rr_b2['sahte_bursty']:.3f}, iid sim {rr_b2['sahte_iid']:.3f}. "
      f"Gercek hatalar L0 gecislerine yogunlasiyor (B2 hata karelerinin near_change payi {NC.iloc[0]['hata_karesi_near_change_payi']:.3f}, tum karelerde {base_nc:.3f}); "
      "bu, gercek hatalarin buyuk bolumunun zamanlama kaymasi oldugu ve (patlamali simulasyona gore) daha az sahte ziyaret urettigi ile tutarlidir (nedensellik bu analizde sinanmadi). Sentetik patlamalar ise zamanda duzgun dagilir. Bu nedenle sentetik egri, bu veri/model icin gercek hata yapisina gore KARAMSAR (alt-sinir benzeri) bir egridir. "
      "(ii) Debounce (MIN_BOX=2) tek-kareli bagimsiz hatalari suzer; iid gurultu bu yuzden dusuk r'de zararsiz, >= 2 karelik patlamalar suzgeci gecer (4.1: sahte ziyaret/ToF dongusu iid vs patlamali). "
      f"(iii) FAR: izgaradaki tum (tur, r) noktalarinda ortalama FAR en fazla {T42['FAR'].max():.4f} (esik alfa+2SE ~ {ALPHA + 2*np.sqrt(ALPHA*(1-ALPHA)/T42['n_pairs'].mean()):.3f}); "
      f"FAR kosulu {'baglayici DEGIL' if (T42['FAR'] <= ALPHA + 2*np.sqrt(ALPHA*(1-ALPHA)/T42['n_pairs'])).all() else 'baglayici'} (kalibrasyon bozuk normal donguler uzerinde yapilir), baglayici olan AUROC kosuludur.")
    H()
    H("## 6. Onceden yazilmis uyarilar")
    H("- Gurultu sentetiktir: patlamalar zamanda duzgun dagilir ve gecis karelerinde yogunlasmaz; gercek hatalar gecis-yapilidir (kosu uzunlugu / yanlis etiket secimi gercek model hatalarina bagimli).")
    H("- Tek operator, tek gun, tek mesafe; r ekseni 'kare hata orani'dir, hatanin zaman yapisi ayri (iid / patlamali) eksendir.")
    H("- Yalniz sorgu hatti bozuldu; B2'nin kendi egitimi / pencere (L=5) yapisi simule edilmedi. 'Dagitilan yolun tamami' = debounce + Box1 giris segmentasyonu + D1/D2 + conformal kalibrasyon (bozuk normal donguler).")
    H("- Dongu sayilari bozuk segmentasyonla degisir (bkz. 4.1); anomali dongusu ile normal dongu ayni bozuk hattan gecer; ToF-dongu eslemesi (fold/atype) bozuk girisin ait oldugu ToF dongusunden alinir (Faz A gibi).")
    H("- AUROC SE'si ~0.03-0.05 (Faz A kat sd'si); tohum araligi yalniz gurultu gerceklesmesi belirsizligini gosterir, dongu-kumesi (n~250) belirsizligini DEGIL. Bant bu yuzden nokta/alt/ust olarak verilmistir.")
    H()
    H(f"_Toplam sure: {time.time()-t0:.0f} s._")
    (OUT / "01_error_propagation.md").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    P(f"[{time.time()-t0:.0f}s] bitti")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    main()

"""Faz B / B.0 - hakem raporu kalan maddeleri (e) D0+D1 Bonferroni, (f) bosluklu pencereler, (g) zaman sirali kalibrasyon.
DESIGN_B.md B.0'a gore. Cikti: reports/faz_b/00_referee_leftovers.md + 00_referee_leftovers_{e,f,g}.csv. Her sayi stdout'a da yazilir.
Calistirma: bash tools/local_run.sh faz_b/00_referee_leftovers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faz_a2"))
sys.path.insert(0, str(ROOT / "faz_a"))
import a2_cycles as ac  # noqa: E402
import fa_common as fc  # noqa: E402

OUTD = ROOT / "reports" / "faz_b"
OUTD.mkdir(parents=True, exist_ok=True)
MD: list[str] = []
Z = 1.959964


def emit(t: str = "") -> None:
    MD.append(t)
    print(t, flush=True)


def wilson(k: float, n: int):
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return (p, min(p, max(0.0, c - h)), max(p, min(1.0, c + h)))


def fw(k: float, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}] ({k:.0f}/{n})" if abs(k - round(k)) < 1e-9 else f"{p:.3f} [{lo:.3f}, {hi:.3f}] (n={n})"


GROUPS = ["seq135", "seq145", "fast", "slow", "seq(135+145)", "tempo(fast+slow)", "tum anomali"]
GMEM = ac.AGROUPS

# =====================================================================================================
emit("# B.0 - Hakem raporunun kalan maddeleri (e, f, g)")
emit("")
emit("`faz_b/00_referee_leftovers.py` uretir; plan `faz_b/DESIGN_B.md` B.0. Statu: KESIF AMACLI; kullanici gormeden makaleye girmez. "
     "B.0 icin onceden yazilmis sayisal beklenti yoktur; yalnizca (g)'deki 'onceden yazilmis okuma' ve '~0,15' cumlesi vardir (asagida denetlenir). "
     "Her sayi stdout'a da yazildi. Sapmalar/ek ayrintilar: DESIGN_B.md 'Sapmalar' satirinda (tek satir).")
emit("")

# =====================================================================================================
# (e) D0 + D1 Bonferroni, B2 x pred
# =====================================================================================================
emit("## (e) D0 + D1 kombinasyonu (B2 x pred)")
emit("")
S = pd.read_csv(ROOT / "reports" / "faz_a" / "04_cycle_scores.csv.gz")
T = S[(S.state_source == "b2") & (S.segmentation == "pred") & (S.role == "test")].copy()
cal = S[(S.state_source == "b2") & (S.segmentation == "pred") & (S.role == "cal")]
ncal_f = cal.groupby("f").size()
T["p01"] = np.minimum(1.0, 2.0 * np.minimum(T["p0"], T["p1"]))
emit("Protokol: Faz A `04_cycle_scores.csv.gz`, role=test satirlari (B2 x pred); D01 = min(1, 2 min(p0, p1)) (Bonferroni); D3 = min(1, 2 min(p1, p2)) (Faz A kolonu p3). "
     f"Kalibrasyon sayisi f basina n_cal = { {int(k): int(v) for k, v in ncal_f.items()} }; ulasilabilir en kucuk p = 1/(n_cal+1) = {[round(float(1/(n+1)),4) for n in ncal_f.values]}; "
     f"ulasilabilir en kucuk Bonferroni p = {[round(float(2/(n+1)),4) for n in ncal_f.values]}. "
     "**alpha=0,05'te iki Bonferroni kurali da yapisal olarak alarm veremez** (en kucuk p ~ 0,095 > 0,05); bu, Faz A D3'te de boyleydi (det_a05 = 0).")
emit("")
VAR = [("D3 (D1+D2)", "p3", "p3"), ("D01 (D0+D1)", "p01", "p01"), ("D0", "p0", "p0"), ("D1", "p1", "p1")]
rows_e = []
neg_all = T[T.atype == "normal"]
n_neg = len(neg_all)
uniq = {a: T[T.atype == a].drop_duplicates(["session", "cycle_idx"]).shape[0] for a in ac.ATYPES}
uniq_g = {g: sum(uniq[a] for a in GMEM[g]) for g in GROUPS}
emit(f"Normal test cycles (havuzlanmis, her biri tek f'de test): n = {n_neg}. Benzersiz anomali dongusu: {uniq}.")
emit("")
for vn, pc, _ in VAR:
    for g in GROUPS:
        mem = GMEM[g]
        aur = []
        for f, gf in T.groupby("f"):
            neg = 1.0 - gf[gf.atype == "normal"][pc].to_numpy(float)
            pos = 1.0 - gf[gf.atype.isin(mem)][pc].to_numpy(float)
            aur.append(ac.auc_ap(neg, pos)[0])
        pos_all = T[T.atype.isin(mem)]
        row = dict(variant=vn, group=g, n_unique_cycles=uniq_g[g], auroc_mean=float(np.nanmean(aur)), auroc_sd=float(np.nanstd(aur, ddof=1)))
        for a in (0.10, 0.05):
            k = f"a{int(a * 100):02d}"
            dr = float(ac.alarm(pos_all[pc].to_numpy(float), a).mean())
            row[f"det_{k}"] = dr
            row[f"det_{k}_wlo"], row[f"det_{k}_whi"] = wilson(dr * uniq_g[g], uniq_g[g])[1:]
        rows_e.append(row)
    far_row = dict(variant=vn, group="normal (FAR)", n_unique_cycles=n_neg)
    for a in (0.10, 0.05):
        k = f"a{int(a * 100):02d}"
        kk = int(ac.alarm(neg_all[pc].to_numpy(float), a).sum())
        far_row[f"far_{k}"] = kk / n_neg
        far_row[f"far_{k}_k"] = kk
        far_row[f"far_{k}_wlo"], far_row[f"far_{k}_whi"] = wilson(kk, n_neg)[1:]
    rows_e.append(far_row)
E = pd.DataFrame(rows_e)
E.to_csv(OUTD / "00_referee_leftovers_e.csv", index=False)

# gate: D3 'tum anomali' AUROC must reproduce Faz A table
TAB = pd.read_csv(ROOT / "reports" / "faz_a" / "04_anomaly_table.csv")
ref = TAB[(TAB.state_source == "b2") & (TAB.segmentation == "pred") & (TAB.variant == "D3") & (TAB.atype == "tum anomali")].iloc[0]
mine = E[(E.variant == "D3 (D1+D2)") & (E.group == "tum anomali")].iloc[0]
gate_e = abs(ref.auroc_mean - mine.auroc_mean) < 1e-9 and abs(ref.det_a10 - mine.det_a10) < 1e-9
emit(f"Dogrulama (kapi): D3 tum-anomali AUROC Faz A tablosu {ref.auroc_mean:.4f} vs burada {mine.auroc_mean:.4f}; det@0,10 {ref.det_a10:.4f} vs {mine.det_a10:.4f} -> "
     f"{'ESLESTI' if gate_e else 'ESLESMEDI'}.")
emit("")
emit("### (e) FAR (normal test dongusu, n = %d; Wilson %%95)" % n_neg)
emit("")
emit("| kural | FAR @ 0,10 | FAR @ 0,05 |")
emit("|---|---|---|")
for vn, pc, _ in VAR:
    r = E[(E.variant == vn) & (E.group == "normal (FAR)")].iloc[0]
    emit(f"| {vn} | {fw(r.far_a10_k, n_neg)} | {fw(r.far_a05_k, n_neg)} |")
emit("")
emit("### (e) Tur basina tespit (alpha = 0,10; Wilson %95, n = benzersiz anomali dongusu, k = oran x n; ayni dongu 5 f'de tekrar eder, Wilson bu yuzden iyimser olabilir)")
emit("")
emit("| kural | " + " | ".join(GROUPS) + " |")
emit("|---|" + "---|" * len(GROUPS))
for vn, pc, _ in VAR:
    cells = []
    for g in GROUPS:
        r = E[(E.variant == vn) & (E.group == g)].iloc[0]
        cells.append(f"{r.det_a10:.3f} [{r.det_a10_wlo:.3f}, {r.det_a10_whi:.3f}]")
    emit(f"| {vn} | " + " | ".join(cells) + " |")
emit("")
emit("### (e) Tur basina tespit (alpha = 0,05)")
emit("")
emit("| kural | " + " | ".join(GROUPS) + " |")
emit("|---|" + "---|" * len(GROUPS))
for vn, pc, _ in VAR:
    cells = []
    for g in GROUPS:
        r = E[(E.variant == vn) & (E.group == g)].iloc[0]
        cells.append(f"{r.det_a05:.3f} [{r.det_a05_wlo:.3f}, {r.det_a05_whi:.3f}]")
    emit(f"| {vn} | " + " | ".join(cells) + " |")
emit("")
emit("### (e) AUROC (kat basina, ortalama +- sd; skor = 1 - p; normal vs ilgili anomali grubu)")
emit("")
emit("| kural | " + " | ".join(GROUPS) + " |")
emit("|---|" + "---|" * len(GROUPS))
for vn, pc, _ in VAR:
    cells = []
    for g in GROUPS:
        r = E[(E.variant == vn) & (E.group == g)].iloc[0]
        cells.append(f"{r.auroc_mean:.3f} +- {r.auroc_sd:.3f}")
    emit(f"| {vn} | " + " | ".join(cells) + " |")
emit("")
d3r = E[(E.variant == "D3 (D1+D2)")]
d01 = E[(E.variant == "D01 (D0+D1)")]
emit("**Okuma (e):** D01 (D0+D1) ve D3 (D1+D2) yan yana yukarida. "
     f"tum-anomali AUROC: D3 {d3r[d3r.group=='tum anomali'].auroc_mean.iloc[0]:.3f}, D01 {d01[d01.group=='tum anomali'].auroc_mean.iloc[0]:.3f}; "
     f"seq(135+145) AUROC: D3 {d3r[d3r.group=='seq(135+145)'].auroc_mean.iloc[0]:.3f}, D01 {d01[d01.group=='seq(135+145)'].auroc_mean.iloc[0]:.3f}; "
     f"tempo(fast+slow) AUROC: D3 {d3r[d3r.group=='tempo(fast+slow)'].auroc_mean.iloc[0]:.3f}, D01 {d01[d01.group=='tempo(fast+slow)'].auroc_mean.iloc[0]:.3f}.")
emit("")

# =====================================================================================================
# (f) eksik kare / bosluklu pencereler
# =====================================================================================================
emit("## (f) Eksik kare / bosluklu pencereler")
emit("")
W = fc.windows()
FT = fc.frame_table()
tt = FT["t"].to_numpy()
rr = fc.win_rows(W)
tw = tt[rr]                                    # (Nw, 5)
dts = np.diff(tw, axis=1)                      # (Nw, 4)
fidw = FT["frame_id"].to_numpy()[rr]
W["max_dt"] = dts.max(axis=1)
W["span_s"] = tw[:, -1] - tw[:, 0]
W["lost_slots"] = np.clip(np.rint(dts / 0.1) - 1, 0, None).sum(axis=1)
W["gap_fid"] = ~W["contiguous"].to_numpy()
G_THR = 0.15
W["gap_t"] = W["max_dt"] > G_THR
W["gap_t25"] = W["max_dt"] > 0.25
W["gap_any"] = W["gap_fid"] | W["gap_t"]
nW = len(W)
emit("Tanimlar: (i) **frame_id boslugu** = `windows()['contiguous'] == False` (son-ilk frame_id farki != 4). (ii) **zaman boslugu** = pencere icinde ardisik kareler arasi dt > 0,15 s "
     "(nominal 0,10 s; dt >= ~0,2 s en az bir kayip kare yuvasi demektir). Cunku kayittaki `frame_id` ardisik bir sayacdir (kare tablosunda yalniz s05'te 2 bosluk var), "
     "ama `t` uzerinde cok sayida bosluk vardir: frame_id sayaci bos kareleri atlamis gorunuyor.")
emit("")
dt_all = np.concatenate([np.diff(g["t"].to_numpy()) for _, g in FT.groupby("session")])
emit(f"Kare dt dagilimi (tum oturumlar, {len(dt_all)} aralik): medyan {np.median(dt_all):.4f} s; %1-%99 = [{np.quantile(dt_all, .01):.3f}, {np.quantile(dt_all, .99):.3f}]; "
     f"dt <= 0,12 s: {(dt_all <= 0.12).mean():.4f}; (0,12-0,15]: {((dt_all > 0.12) & (dt_all <= 0.15)).mean():.4f}; (0,15-0,25]: {((dt_all > 0.15) & (dt_all <= 0.25)).mean():.4f}; > 0,25 s: {(dt_all > 0.25).mean():.4f}. "
     f"Aralik sayisi: dt > 0,15 s = {(dt_all > 0.15).sum()}; dt > 0,25 s = {(dt_all > 0.25).sum()}; en buyuk dt = {dt_all.max():.2f} s.")
emit("")
emit(f"Pencere sayisi {nW}. frame_id bosluklu: **{int(W.gap_fid.sum())}** ({W.gap_fid.mean():.4%}); zaman bosluklu (dt > 0,15 s): **{int(W.gap_t.sum())}** ({W.gap_t.mean():.4%}); "
     f"dt > 0,25 s: {int(W.gap_t25.sum())} ({W.gap_t25.mean():.4%}); ikisinden biri: {int(W.gap_any.sum())} ({W.gap_any.mean():.4%}). "
     f"Bosluklu pencerelerde ortalama kayip kare yuvasi (zaman bosluklu olanlar): {W.loc[W.gap_t, 'lost_slots'].mean():.2f}; pencere suresi medyan {W.span_s.median():.3f} s (bosluksuz {W.loc[~W.gap_any, 'span_s'].median():.3f} s).")
emit("")
tb = W.groupby("session").agg(n=("win", "size"), fid_gap=("gap_fid", "sum"), t_gap=("gap_t", "sum"), t_gap25=("gap_t25", "sum"))
tb["t_gap_pay"] = tb.t_gap / tb.n
emit("| oturum | pencere | frame_id bosluklu | zaman bosluklu (>0,15 s) | dt>0,25 s | zaman bosluklu pay |")
emit("|---|---|---|---|---|---|")
for s, r in tb.iterrows():
    emit(f"| {s} | {int(r.n)} | {int(r.fid_gap)} | {int(r.t_gap)} | {int(r.t_gap25)} | {r.t_gap_pay:.3%} |")
emit("")
cl = W.groupby("y_true").agg(n=("win", "size"), t_gap=("gap_t", "sum"))
cl["pay"] = cl.t_gap / cl.n
emit("Son karenin sinifina gore zaman bosluklu pay: " + "; ".join(f"sinif {int(c)}: {r.pay:.3%} ({int(r.t_gap)}/{int(r.n)})" for c, r in cl.iterrows()) + ".")
emit("")

# S1 metrikleri
f1_rows = []
emit("### (f) S1 metrikleri: bosluklu pencereler dahil / haric (B2, CNN-GRU, B1)")
emit("")
emit("Havuzlanmis dogruluk ve macro-F1 (`fc.state_metrics`) ile kat basina macro-F1 ortalama +- sd (5 kat). 'haric-fid' = frame_id bosluklu pencereler cikarildi; 'haric-t' = zaman bosluklu (dt > 0,15 s) cikarildi; "
     "'yalniz-t' = yalniz zaman bosluklu pencereler (kucuk altorneklem, yorumlama icin n'e bakin).")
emit("")
emit("| model | altkume | n pencere | dogruluk | macro-F1 (havuz) | macro-F1 kat ort +- sd | dogr. kat ort |")
emit("|---|---|---|---|---|---|---|")
state_ref = pd.read_csv(ROOT / "reports" / "faz_a" / "03_state_table.csv")
gate_f = True
for m in ("b2", "cnngru", "b1"):
    o = fc.load_oof(m, "S1")
    assert len(o) == 16248 and o.fold.nunique() == 5, m
    d = W[["win", "session", "frame_id", "gap_fid", "gap_t", "gap_any"]].merge(o[["session", "frame_id", "fold", "y_true", "y_pred"]], on=["session", "frame_id"], how="left", validate="1:1")
    assert d["y_pred"].notna().all()
    subs = [("tum (A)", np.ones(len(d), bool)), ("haric-fid", ~d.gap_fid.to_numpy()), ("haric-t", ~d.gap_t.to_numpy()),
            ("haric-fid+t", ~d.gap_any.to_numpy()), ("yalniz-t", d.gap_t.to_numpy())]
    for nm, msk in subs:
        dd = d[msk]
        if len(dd) == 0:
            continue
        pm = fc.state_metrics(dd.y_true, dd.y_pred)
        pf = [fc.state_metrics(g.y_true, g.y_pred) for _, g in dd.groupby("fold")]
        f1m = float(np.mean([x["macro_f1"] for x in pf]))
        f1s = float(np.std([x["macro_f1"] for x in pf], ddof=1)) if len(pf) > 1 else float("nan")
        acm = float(np.mean([x["acc"] for x in pf]))
        f1_rows.append(dict(model=m, subset=nm, n=len(dd), acc=pm["acc"], macro_f1_pooled=pm["macro_f1"], macro_f1_fold_mean=f1m, macro_f1_fold_sd=f1s, acc_fold_mean=acm, n_folds=len(pf)))
        emit(f"| {m} | {nm} | {len(dd)} | {pm['acc']:.4f} | {pm['macro_f1']:.4f} | {f1m:.4f} +- {f1s:.4f} | {acm:.4f} |")
        if nm == "tum (A)":
            r0 = state_ref[(state_ref.split == "S1") & (state_ref.model == m) & (state_ref.subset == "tum")]
            if len(r0):
                ok = abs(r0.macro_f1_mean.iloc[0] - f1m) < 1e-9
                gate_f &= bool(ok)
                print(f"[kapi f] {m}: Faz A 03 macro-F1 {r0.macro_f1_mean.iloc[0]:.6f} vs {f1m:.6f} -> {'ESLESTI' if ok else 'ESLESMEDI'}", flush=True)
F = pd.DataFrame(f1_rows)
F.to_csv(OUTD / "00_referee_leftovers_f.csv", index=False)
emit("")
emit(f"Dogrulama (kapi): 'tum' satirlari Faz A `03_state_table.csv` (S1, tum) kat-ortalama macro-F1 ile {'ESLESTI' if gate_f else 'ESLESMEDI'} (kat ortalamasi birebir).")
emit("")
dl = []
for m in ("b2", "cnngru", "b1"):
    a = F[(F.model == m) & (F.subset == "tum (A)")].iloc[0]
    b = F[(F.model == m) & (F.subset == "haric-t")].iloc[0]
    c = F[(F.model == m) & (F.subset == "yalniz-t")].iloc[0]
    dl.append(f"{m}: haric-t - tum = {b.macro_f1_fold_mean - a.macro_f1_fold_mean:+.4f} (kat ort macro-F1), dogruluk {b.acc - a.acc:+.4f}; bosluklu altkumede dogruluk {c.acc:.4f} (n={int(c.n)}) vs bosluksuz {b.acc:.4f}")
emit("**Okuma (f):** " + "; ".join(dl) + ".")
emit("")

# =====================================================================================================
# (g) zaman sirali kalibrasyon
# =====================================================================================================
emit("## (g) Zaman sirali kalibrasyon (s02'nin normal donguleri)")
emit("")
emit("s02'nin normal donguleri `start_idx` (oturum ici kare indeksi) sirasiyla dizildi. Duzen (A): uydurma 1-15 / kalibrasyon 16-30 / izleme 31-51; duzen (B): uydurma 1-20 / kalibrasyon 21-35 / izleme 36-51. "
     "Her duzende D0/D1/D2 parametreleri `ac.fit_params` ile yalniz uydurma dongulerinden ogrenilir; kalibrasyon dongulerinde standart conformal (`ac.conf_p`, p = (1 + #{cal >= s})/(n_cal+1)); "
     "test = izleme normalleri + TUM anomali donguleri (s03-s06); D3 = min(1, 2 min(p1, p2)); alpha = 0,10. Hucreler: B2 x pred ve ToF x tof.")
emit("")
LAB = ac.source_labels(sources=("tof", "b2"), verbose=False)
CELLS = [("b2", "pred"), ("tof", "tof")]
ORD = {"A": (15, 15), "B": (20, 15)}
ALPHA = 0.10
rows_g = []
d1diag = []
verdicts = {}
for src, seg in CELLS:
    C, _ = ac.build_cycles(src, seg, LAB[src])
    N = C[C.atype == "normal"].sort_values("start_idx").reset_index(drop=True)
    A_ = C[C.atype != "normal"]
    n_nat = int(N.natural_dev.sum())
    assert (np.diff(N.start_idx.to_numpy()) > 0).all() and len(N) == 51, (src, seg, len(N))
    emit(f"### Hucre {src} x {seg}: {len(N)} normal dongu; anomali donguleri: " + ", ".join(f"{a} {int((A_.atype == a).sum())}" for a in ac.ATYPES)
         + (f". Havuzda {n_nat} dogal-sapma isaretli normal dongu var; Faz A'daki gibi havuzda birakildi, alarm durumuna bakilmadi (kural)." if n_nat else "."))
    emit("")
    for on, (nf, nc) in ORD.items():
        fit = N.iloc[:nf].to_dict("records")
        cal_ = N.iloc[nf:nf + nc].to_dict("records")
        mon = N.iloc[nf + nc:].to_dict("records")
        par = ac.fit_params(fit)
        Sc = ac.score_cycles(cal_, par)
        Sm = ac.score_cycles(mon, par)
        Sa = ac.score_cycles(A_.to_dict("records"), par)
        Sm["atype"] = "normal"
        for df in (Sm, Sa):
            for k, v in (("0", "D0"), ("1", "D1"), ("2", "D2")):
                df["p" + k] = ac.conf_p(Sc[v].to_numpy(), df[v].to_numpy())
            df["p3"] = ac.d3(df["p1"], df["p2"])
        nm_ = len(Sm)
        _sq = Sa[Sa.atype.isin(["seq135", "seq145"])]
        d1diag.append(dict(cell=f"{src} x {seg}", order=on, n_cal=nc, d1_cal_max=float(Sc.D1.max()), n_cal_ge_max=int((Sc.D1 >= Sc.D1.max() - 1e-12).sum()),
                           frac_seq_le_max=float((_sq.D1 <= Sc.D1.max() + 1e-12).mean()), n_seq=len(_sq)))
        se = float(np.sqrt(ALPHA * (1 - ALPHA) / nm_))
        band = (ALPHA - 2 * se, ALPHA + 2 * se)
        kmin = int(np.floor((ALPHA + 2 * se) * nm_ + 1e-9)) + 1               # smallest k with k/n > alpha + 2SE
        mdi = kmin / nm_ - ALPHA
        pmin = 1.0 / (nc + 1)
        # mean durations (descriptive)
        du = [np.mean([c["dur_s"] for c in lst]) for lst in (fit, cal_, mon)]
        sdu = [np.std([c["dur_s"] for c in lst], ddof=1) / np.sqrt(len(lst)) for lst in (fit, cal_, mon)]
        emit(f"**Duzen {on}** (uydurma {nf}, kalibrasyon {nc}, izleme {nm_}): en kucuk p = 1/(n_cal+1) = {pmin:.4f}; en kucuk D3 p = {min(1, 2*pmin):.4f} -> "
             f"D3 alpha = {ALPHA} altinda alarm {'verebilir' if min(1, 2*pmin) <= ALPHA + 1e-9 else 'VEREMEZ (yapisal)'}. "
             f"alpha +- 2SE = [{band[0]:.3f}, {band[1]:.3f}] (SE = {se:.4f}); saptanabilir en kucuk FAR artisi (k/n > ust sinir): k = {kmin}/{nm_} -> +{mdi:.3f}. "
             f"Ortalama sure (s) uydurma/kal./izleme: {du[0]:.2f} (SE {sdu[0]:.2f}) / {du[1]:.2f} ({sdu[1]:.2f}) / {du[2]:.2f} ({sdu[2]:.2f}).")
        emit("")
        emit("| kural | izleme FAR (Wilson) | alpha +- 2SE icinde | " + " | ".join(GROUPS) + " |")
        emit("|---|---|---|" + "---|" * len(GROUPS))
        for vn, pc, sc in (("D3 (birincil)", "p3", None), ("D0 (ek)", "p0", "D0"), ("D1 (ek)", "p1", "D1"), ("D2 (ek)", "p2", "D2")):
            kk = int(ac.alarm(Sm[pc].to_numpy(float), ALPHA).sum())
            far = kk / nm_
            within = band[0] <= far <= band[1]
            cells = []
            dets = {}
            for g in GROUPS:
                sub = Sa[Sa.atype.isin(GMEM[g])]
                dr = float(ac.alarm(sub[pc].to_numpy(float), ALPHA).mean())
                dets[g] = dr
                cells.append(f"{dr:.3f} [{wilson(dr * len(sub), len(sub))[1]:.2f}, {wilson(dr * len(sub), len(sub))[2]:.2f}] (n={len(sub)})")
            # AUROC izleme normal vs tum anomali (skor: ham ya da 1-p)
            sc_m = Sm[sc].to_numpy(float) if sc else 1 - Sm["p3"].to_numpy(float)
            sc_a = Sa[sc].to_numpy(float) if sc else 1 - Sa["p3"].to_numpy(float)
            au = ac.auc_ap(sc_m, sc_a)[0]
            emit(f"| {vn} | {fw(kk, nm_)} | {'evet' if within else 'HAYIR'} | " + " | ".join(cells) + " |")
            rows_g.append(dict(cell=f"{src}x{seg}", order=on, rule=vn, n_fit=nf, n_cal=nc, n_mon=nm_, far=far, far_k=kk, far_wlo=wilson(kk, nm_)[1], far_whi=wilson(kk, nm_)[2],
                               band_lo=band[0], band_hi=band[1], within_2se=bool(within), min_detectable_increase=mdi, auroc_mon_vs_all_anom=au,
                               **{f"det_{g}": dets[g] for g in GROUPS}, d3_can_alarm=bool(min(1, 2 * pmin) <= ALPHA + 1e-9)))
            if vn.startswith("D3"):
                verdicts[(src, seg, on)] = (within, far)
        au3 = [r for r in rows_g if r["cell"] == f"{src}x{seg}" and r["order"] == on and r["rule"].startswith("D3")][-1]["auroc_mon_vs_all_anom"]
        emit("")
        emit(f"(D3 skoru 1-p3 ile izleme normali vs tum anomali AUROC = {au3:.3f}; D0/D1/D2 satirlari yalniz ek bilgidir, birincil okuma D3 satirina dayanir.)")
        emit("")
G = pd.DataFrame(rows_g)
G.to_csv(OUTD / "00_referee_leftovers_g.csv", index=False)

emit("### (g) Onceden yazilmis okumanin harfiyen uygulanmasi")
emit("")
all_in = all(v[0] for v in verdicts.values())
emit("Kural: iki duzende de FAR alpha +- 2SE icindeyse 'surukleme saptanmadi'. D3, alpha = 0,10, izleme normalleri:")
emit("")
emit("| hucre | duzen | izleme FAR | alpha +- 2SE icinde |")
emit("|---|---|---|---|")
for (src, seg, on), (w_, far) in verdicts.items():
    emit(f"| {src} x {seg} | {on} | {far:.3f} | {'evet' if w_ else 'HAYIR'} |")
emit("")
cellok = {}
for src, seg in CELLS:
    cellok[(src, seg)] = verdicts[(src, seg, "A")][0] and verdicts[(src, seg, "B")][0]
emit("Harfiyen sonuc: " + "; ".join(f"{s} x {g}: **{'surukleme saptanmadi' if cellok[(s, g)] else 'kural saglanmadi (surukleme isareti)'}**" for s, g in CELLS) + ".")
emit("")
mdiA = G[(G.order == "A")].min_detectable_increase.iloc[0]
mdiB = G[(G.order == "B")].min_detectable_increase.iloc[0]
d3_can = bool(G[G.rule.str.startswith("D3")].d3_can_alarm.any())
emit(f"Yazilacak cumle (plana gore): bu orneklemde saptanabilir en kucuk FAR artisi ~0,15 mertebesindedir. Hesaplanan: duzen A +{mdiA:.2f} (izleme {int(G[G.order=='A'].n_mon.iloc[0])} dongu), duzen B +{mdiB:.2f} "
     f"(izleme {int(G[G.order=='B'].n_mon.iloc[0])} dongu) - yani ~0,15 duzen A icin yaklasik dogru, duzen B icin daha kaba (~0,2).")
emit("")
if not d3_can:
    emit("**Onemli uyari (okumanin gucu):** her iki duzende n_cal = 15 oldugu icin en kucuk standart-conformal p = 1/16 = 0,0625 ve en kucuk D3 p = 0,125 > 0,10. "
         "Yani D3 bu duzenlerde alpha = 0,10'da HICBIR dongu icin alarm veremez: FAR = 0 ve anomali tespiti = 0 yapisaldir (tablodaki D3 satirlari). "
         "Harfiyen okuma 'surukleme saptanmadi' dese de, D3 icin bu denetimin gucu yoktur; FAR ~0'in alpha +- 2SE bandina girmesi kuralin bir sonucudur, surukleme olmadiginin kaniti degildir. "
         "Ek D0/D1/D2 satirlari (tek tek, alarm verebilen kurallar) bu nedenle eklendi; bunlarin FAR'i ve band durumu yukarida verilir.")
    emit("")
sup = G[(G.rule != "D3 (birincil)")]
sup_in = bool(sup.within_2se.all())
emit(f"Ek (alarm verebilen) D0/D1/D2 kurallari icin ayni bant denetimi: iki hucrede, iki duzende, uc kuralin FAR'i alpha +- 2SE icinde mi? -> {'evet, hepsi icinde' if sup_in else 'HAYIR, en az biri disinda (tabloya bakin)'}. "
     "Bu ek kontrol harfiyen okumanin parcasi degildir; bu kurallar alarm verebildigi icin D3'ten daha anlamlidir, yine de saptanabilir FAR artisi ~0,14-0,21 ile sinirlidir (kucuk izleme orneklemi).")
emit("")
emit("Not: n_cal = 15 ile D1'in B2 x pred hucresindeki tespiti ~0'a duser (Faz A'da n_cal ~ 19-22 iken D1 seq(135+145) tespiti ~0,69). Mekanizma dogrudan hesaptan cikar: p <= 0,10 icin #{cal >= s} = 0 gerekir "
     "(n_cal=15: en kucuk p = 1/16 = 0,0625, bir esitlik p = 2/16 = 0,125). Gorulmemis gecis skoru ayni kaynak durum icin ozdestir; kalibrasyonda tek bir sahte gecisli dongu bu skoru doldurursa ayni durumdan "
     "gorulmemis gecisi olan tum anomaliler onunla baglasir (n_cal ~ 20'de #{cal >= s} <= 1 hala alarm verirdi). Olculen:")
emit("")
emit("| hucre | duzen | n_cal | max cal D1 | max'a esit/ustu cal sayisi | seq anomalilerde D1 <= max cal D1 payi | seq anomali sayisi |")
emit("|---|---|---|---|---|---|---|")
for r in d1diag:
    emit(f"| {r['cell']} | {r['order']} | {r['n_cal']} | {r['d1_cal_max']:.3f} | {r['n_cal_ge_max']} | {r['frac_seq_le_max']:.3f} | {r['n_seq']} |")
emit("")
emit("Payi ~1 olan hucrede D1 seq alarmi imkansizdir (anomali skoru kalibrasyon maksimumunu asmiyor); bu zaman sirali duzenin guc kaybidir, surukleme bulgusu degildir.")
emit("")

# ozet
emit("## Ozet sayilar")
emit("")
d3a = E[(E.variant == "D3 (D1+D2)") & (E.group == "normal (FAR)")].iloc[0]
d01a = E[(E.variant == "D01 (D0+D1)") & (E.group == "normal (FAR)")].iloc[0]
emit(f"- (e) B2 x pred, alpha 0,10: D01 FAR {d01a.far_a10:.3f} (k={int(d01a.far_a10_k)}/{n_neg}), tum-anomali tespit {d01[d01.group=='tum anomali'].det_a10.iloc[0]:.3f}, AUROC {d01[d01.group=='tum anomali'].auroc_mean.iloc[0]:.3f}; "
     f"D3 FAR {d3a.far_a10:.3f}, tespit {d3r[d3r.group=='tum anomali'].det_a10.iloc[0]:.3f}, AUROC {d3r[d3r.group=='tum anomali'].auroc_mean.iloc[0]:.3f}; alpha 0,05'te ikisi de 0 tespit (yapisal).")
emit(f"- (f) bosluklu pencere payi: frame_id {W.gap_fid.mean():.4%} ({int(W.gap_fid.sum())}); zaman (dt>0,15 s) {W.gap_t.mean():.2%} ({int(W.gap_t.sum())}/{nW}).")
emit("- (g) D3 duzenleri: " + "; ".join(f"{s}x{g} A:{verdicts[(s, g, 'A')][1]:.3f} B:{verdicts[(s, g, 'B')][1]:.3f}" for s, g in CELLS) + f"; D3 alarm verebilir mi: {d3_can}.")

(OUTD / "00_referee_leftovers.md").write_text("\n".join(MD) + "\n", encoding="utf-8")
print("[ok] yazildi:", OUTD / "00_referee_leftovers.md")

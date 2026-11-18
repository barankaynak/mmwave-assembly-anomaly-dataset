"""Faz B / B.3 rapor + sekil: reports/faz_b/03_frame_rate.csv (+ _folds, _detection, _fits) -> 03_frame_rate.md, 03_frame_rate.png.
Hicbir model egitilmez; yalniz 03_frame_rate.py ciktilarini okur. Her sayi stdout'a da basilir (md metni)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = ROOT / "reports" / "faz_b"
D = pd.read_csv(REP / "03_frame_rate.csv")
DET = pd.read_csv(REP / "03_frame_rate_detection.csv")
FIT = pd.read_csv(REP / "03_frame_rate_fits.csv")
FOLD = pd.read_csv(REP / "03_frame_rate_folds.csv")
MD = []


def emit(t=""):
    MD.append(t); print(t, flush=True)


def tab(df):
    c = list(df.columns)
    out = ["| " + " | ".join(c) + " |", "|" + "|".join(["---"] * len(c)) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def yn(b): return "evet" if b else "hayır"


M = D[D.kind == "main"].set_index("k")
K = [1, 2, 3, 5]
HZ = {1: "10", 2: "5", 3: "3,33", 5: "2"}
T975 = 2.7764451051977987

emit("# B.3 — Kare hızı: her k'inci kare tutulursa ne kalıyor? (öykünme)")
emit()
emit("Üretim: `faz_b/03_frame_rate.py` (hesap, kapılar, tüm seviyeler), `faz_b/03b_frame_rate_report.py` (bu rapor). Ham sayılar: `03_frame_rate.csv` (seviye başına tek satır), `03_frame_rate_folds.csv`, `03_frame_rate_detection.csv` (dokunuş tespiti, oturum × süre kutusu), `03_frame_rate_fits.csv` (lojistik d50/d90); OOF'lar `reports/faz_b/oof/fr_*__S1__f*.csv.gz`. "
     "**Statü: keşif amaçlı, plan koşudan önce yazıldı (`DESIGN_B.md` B.3); `main.tex`'e girmez.** Hiçbir şey radarda çalıştırılmadı: seyreltme kaydedilmiş 10 Hz karelerden yazılımla öykünüldü. **B.3'te önceden yazılmış sayısal bir beklenti yoktur** (yalnız diz kuralı); bu yüzden aşağıda 'beklenti tuttu/tutmadı' satırı yoktur.")
emit()
emit("## Yöntem (DESIGN_B.md'ye göre)")
emit("- **Seyreltme**: her oturumda `frame_idx_in_session % k == 0` olan kareler tutulur (ana koşu faz kaydırması 0; fazlar 1…k−1 ikincil). k ∈ {2, 3, 5} → 5 / 3,33 / 2 Hz; k = 1 (10 Hz) kapı satırı.")
emit("- **Pencere** L_k = ceil(5/k): k = 2 → 3, k = 3 → 2, k = 5 → 1 kare. Pencerenin etiketi son tutulan karenin 10 Hz `y_true`'su (yeniden yumuşatma yok). L = 1'de eğim özellikleri 0; özellikler yine 7 istatistik × (son, ortalama, eğim) = 21 sütun (L = 1'de son = ortalama). Model B2: RF 300 ağaç, `class_weight=balanced`, tohum 20261104, `n_jobs = 3`.")
emit("- **Katlar**: pencerenin son karesinin döngüsünün (gid) Faz A S1 katı; yani aynı döngüler aynı katta (doğrulandı: her gid tek katta). Eşli fark: seviye − Faz A B2, kat bazında (5 kat), ortalama ve %95 t-GA (df = 4).")
emit("- **Debounce süre olarak sabit** (MIN_BOX 0,2 s, MAX_GAP 0,1 s), kareye yuvarlanıp en az 1 (MIN_BOX) ve 0 (MAX_GAP): " + "; ".join(f"k = {k}: MIN_BOX {int(M.loc[k,'min_box_fr'])}, MAX_GAP {int(M.loc[k,'max_gap_fr'])} kare" for k in K) +
     ". **k = 2'de MAX_GAP = 0,1 s / 0,2 s = 0,5 kare** ve yuvarlama kuralı belirsiz: ana koşu Python `round` (→ 0) kullandı; yarı-yukarı (→ 1) ikincil olarak koşuldu (aşağıda; sonuç çok farklı).")
emit("- **Anomali hattı**: tutulan akışın `y_pred`'i (ilk L_k − 1 tutulan kare penceresiz → none) → yukarıdaki debounce → 10 Hz'e sıfırıncı-derece tutma (bir tutulan karenin etiketi sonraki tutulan kareye kadar tutulur; ziyaret süreleri yani 0,1·k s'ye nicemlenir) → `ac.build_cycles('pred')` → D1 + D2 → D3, Faz A rolleri ve standart conformal, α = 0,10. `build_cycles`'ın içindeki sabit (2, 1) debounce'unun tutulmuş akışta etkisiz olduğu her koşuda iddia (assert) ile doğrulandı.")
emit("- **Dokunuş tespiti (D-ham, A3.2 yöntemi)**: ToF dokunuşu (mesafe < 140 mm kesintisiz koşu, süre = span + medyan dt); dokunuşun kare kümesi tam hızda A3.2'deki `(t−100 ms, t]` atamasıyla belirlenir, **sonra tutulan karelerle kesiştirilir** (radar yalnız tutulan karelerde ölçer). D-ham = kesişimde son karesi bulunan en az bir penceresi o kutuyu ham `y_pred` ile tahmin ediyor. **Ana payda**: tam hızda değerlendirilebilen dokunuşlar (n_frames > 0; aynı 1111 dokunuş, hepsi); tutulan karesi hiç olmayan dokunuş **kaçırma sayılır** (örnekleme kaybı, ayrıca raporlanır). Koşullu varyant (yalnız ≥ 1 tutulan karesi olanlar) ikincil.")
emit("- **Bayt/s**: V2 taban 157,3 B/kare (B.2 ile aynı sözleşme) × 10/k Hz. Not: seyreltme bayt/kare'yi değiştirmez, yalnız kare sayısını; B.2'deki özel-biçim varsayımları geçerli değildir (stok akış).")
emit()
emit("## Kapılar")
g1 = M.loc[1]
emit(f"- **Kapı 1 (k = 1 ≡ Faz A B2)**: kat bazında macro-F1 ve kat bazında D3 AUROC Faz A B2 ile **birebir** (max |fark| = 0,0; ortalamalar {g1.f1_mean:.4f} ve {g1.auroc:.4f}); FAR {g1.far_a10:.4f}, tespit {g1.det_a10:.4f} (Faz A ile aynı).")
emit("- **Kapı 2 (k = 1 D-ham ≡ A3.2)**: 6 oturum × 5 süre kutusunda B2 / D-ham `k_detect`, `n_eval` ve oturum başına lojistik `b0, b1, d50, d90` `reports/faz_a3/02_envelope_*.csv` ile **birebir** (fark 0 / 0).")
emit("- **Sonuç: kapılar geçti; B.3 koşuldu.**")
emit()

# -------------------------------------------------------------------------- ana tablo
emit("## Ana sonuç (tohum 20261104, faz 0; S1 5 kat)")
emit()
emit("`pencere` = test+eğitim pencere sayısı; `span` = pencerenin süresi (0,1·(k(L−1)+1) s); ΔF1/ΔAUROC = Faz A B2'ye göre eşli kat farkı, ortalama [%95 GA]; `ΔF1 (kısıtlı ref.)` = fark, referans = Faz A B2'nin yalnız aynı tutulan karelerdeki macro-F1'i (modelin düşük hızda eğitilmesi ile test karesinin azalmasını ayırmak için); AUROC = D3 `tüm anomali`, kat ort. ± sd; tespit / FAR α = 0,10 (FAR: SE); `kaçırılan dok.` = tutulan karesi olmayan dokunuş / 1111.")
emit()
rows = []
for k in K:
    r = M.loc[k]
    rows.append({"k": k, "Hz": HZ[k], "L": int(r.L), "span (s)": f"{r.window_span_s:.1f}", "pencere": int(r.n_windows), "V2 B/s": f"{r.v2_B_s:.0f}",
                 "macro-F1 ± sd": f"{r.f1_mean:.4f} ± {r.f1_sd:.4f}", "ΔF1 [GA]": f"{r.df1_mean:+.4f} [{r.df1_lo:+.4f}, {r.df1_hi:+.4f}]",
                 "ΔF1 (kısıtlı ref.)": f"{r.df1_restricted_mean:+.4f}",
                 "AUROC ± sd": f"{r.auroc:.4f} ± {r.auroc_sd:.4f}", "ΔAUROC [GA]": f"{r.dauc_mean:+.4f} [{r.dauc_lo:+.4f}, {r.dauc_hi:+.4f}]",
                 "seq / tempo AUROC": f"{r.auroc_seq:.4f} / {r.auroc_tempo:.4f}", "tespit": f"{r.det_a10:.3f}", "FAR (SE)": f"{r.far_a10:.4f} ({r.far_se_a10:.3f})",
                 "döngü": int(r.n_pred_cycles), "kaçırılan dok.": f"{int(r.touch_lost_sampling)}"})
emit(tab(pd.DataFrame(rows)))
emit()
emit("Sınıf bazında F1 (havuzlanmış OOF; 0 = none): " + "; ".join(f"k = {k}: " + " / ".join(f"{M.loc[k, f'f1_pooled_{c}']:.3f}" for c in range(6)) for k in K))
emit()

# -------------------------------------------------------------------------- diz kurali
emit("## Önceden yazılmış diz kuralı (B.2 ile aynı)")
emit("Kural: en yalın seviye (burada: en düşük kare hızı = en büyük k), macro-F1 düşüşü ≤ 0,02 **ve** AUROC düşüşü ≤ 0,03 sağlayan seviyedir; kat GA'ları daha büyük bir düşüşü dışlamalı. Dört ölçüt ayrı ayrı: (i) ΔF1 ≥ −0,02, (ii) ΔF1 GA alt ≥ −0,02, (iii) ΔAUROC ≥ −0,03, (iv) ΔAUROC GA alt ≥ −0,03. **Literal geçiş = dördü birden**; yalnız nokta tahmini (i + iii) ayrıca.")
emit()
rows = []
for k in K[1:]:
    r = M.loc[k]
    rows.append({"k (Hz)": f"{k} ({HZ[k]})", "(i) ΔF1 ≥ −0,02": f"{yn(r.pass_f1_point)} ({r.df1_mean:+.4f})", "(ii) GA alt ≥ −0,02": f"{yn(r.pass_f1_ci)} ({r.df1_lo:+.4f})",
                 "(iii) ΔAUROC ≥ −0,03": f"{yn(r.pass_auc_point)} ({r.dauc_mean:+.4f})", "(iv) GA alt ≥ −0,03": f"{yn(r.pass_auc_ci)} ({r.dauc_lo:+.4f})",
                 "**literal**": yn(r.pass_literal), "yalnız nokta": yn(r.pass_point_only)})
emit(tab(pd.DataFrame(rows)))
emit()
FF2 = (FOLD[FOLD.tag == "k2"].macro_f1 - FOLD[FOLD.tag == "k2"].ref_macro_f1)
lit = [k for k in K[1:] if M.loc[k].pass_literal]
pt = [k for k in K[1:] if M.loc[k].pass_point_only]
emit(f"**Sonuç (ana koşu):** literal kuralı geçen k: {lit if lit else 'yok'}; yalnız-nokta kuralını geçen k: {pt if pt else 'yok'}. "
     + ("Hiçbir seyreltme düzeyi kuralı geçmiyor: **tam kare hızı (10 Hz) gerekli** (bu model ve bu hat için). " if not lit else "")
     + f"Kural k = 2'de bile macro-F1'den takılıyor (ΔF1 {M.loc[2].df1_mean:+.4f}, GA [{M.loc[2].df1_lo:+.4f}, {M.loc[2].df1_hi:+.4f}]; 5 katın hepsinde kat farkı {FF2.min():+.4f} … {FF2.max():+.4f}, yani hepsi −0,02'nin altında); AUROC ayrıca ana debounce kuralında (ΔAUROC {M.loc[2].dauc_mean:+.4f}) takılıyor, ancak o ölçüt debounce yuvarlamasına çok duyarlı (aşağıda).")
emit()

# -------------------------------------------------------------------------- debounce varyantlari
emit("## Ek (brief dışı): debounce yuvarlamasına duyarlılık")
emit("Macro-F1 debounce'tan etkilenmez (debounce yalnız döngü hattında); AUROC/FAR/tespit etkilenir. Aynı OOF, farklı debounce (tohum 20261104, faz 0): **ana** = süre sabit, Python `round`; **MAX_GAP = 1 kare** (k = 2'de yarı-yukarı yuvarlama ile aynı); **kare sayısı sabit** (MIN_BOX 2, MAX_GAP 1, yani 10 Hz'deki sayılar; saniye karşılığı 0,1·k ile büyür). Bu üç düzenleme arasından en iyiyi seçmek seçim yanlılığı olurdu: **tablo bir tarama olarak okunmalı, bir sonuç olarak değil**.")
emit()
rows = []
for k in K[1:]:
    for lab, tag in (("ana", f"k{k}"), ("MAX_GAP = 1 kare", f"k{k}_mg1" if k != 2 else "k2_mg1"), ("kare sayısı sabit (2/1)", f"k{k}_deb21")):
        r = D[D.tag == tag].iloc[0]
        rows.append({"k": k, "debounce": lab, "MIN_BOX/MAX_GAP (kare)": f"{int(r.min_box_fr)}/{int(r.max_gap_fr)}", "AUROC ± sd": f"{r.auroc:.4f} ± {r.auroc_sd:.4f}",
                     "ΔAUROC [GA]": f"{r.dauc_mean:+.4f} [{r.dauc_lo:+.4f}, {r.dauc_hi:+.4f}]", "seq / tempo": f"{r.auroc_seq:.4f} / {r.auroc_tempo:.4f}",
                     "tespit": f"{r.det_a10:.3f}", "FAR": f"{r.far_a10:.4f}", "döngü": int(r.n_pred_cycles), "literal": yn(r.pass_literal)})
emit(tab(pd.DataFrame(rows)))
emit()

# -------------------------------------------------------------------------- faz ve tohum
emit("## Ek (brief dışı): faz kaydırması ve RF tohumu")
emit("Faz: aynı k için tutulan karelerin başlangıcı 0…k−1 (tohum 20261104; k = 2: 2, k = 3: 3, k = 5: 5 faz). Tohum: RF tohumu 1–4, faz 0. k = 1 için 4 tohum aynı tablodadır. Δ = faz/tohum ortalaması − k = 1 tohum ortalaması (macro-F1 0,8567; AUROC 0,9285).")
emit()
b1 = D[(D.k == 1) & (D.kind == "seed")]
bF, bA = b1.f1_mean.mean(), b1.auroc.mean()
rows = []
nlit = {}
for k in K:
    ph = D[(D.k == k) & (D.kind.isin(["main", "phase"]))]
    sd = D[(D.k == k) & (D.kind == "seed")]
    allr = pd.concat([ph, sd]) if k > 1 else sd
    nlit[k] = (int(allr.pass_literal.sum()), len(allr))
    rows.append({"k": k, "faz sayısı": len(ph) if k > 1 else "-", "macro-F1 faz ort. (min–maks)": f"{ph.f1_mean.mean():.4f} ({ph.f1_mean.min():.4f}–{ph.f1_mean.max():.4f})" if k > 1 else "-",
                 "AUROC faz ort. (min–maks)": f"{ph.auroc.mean():.4f} ({ph.auroc.min():.4f}–{ph.auroc.max():.4f})" if k > 1 else "-",
                 "macro-F1 tohum ort. ± sd": f"{sd.f1_mean.mean():.4f} ± {sd.f1_mean.std(ddof=1):.4f}", "AUROC tohum ort. ± sd": f"{sd.auroc.mean():.4f} ± {sd.auroc.std(ddof=1):.4f}",
                 "ΔF1 (tohum ort.)": f"{sd.f1_mean.mean() - bF:+.4f}", "ΔAUROC (tohum ort.)": f"{sd.auroc.mean() - bA:+.4f}",
                 "tespit tohum ort.": f"{sd.det_a10.mean():.3f}", "FAR tohum ort.": f"{sd.far_a10.mean():.4f}",
                 "literal geçen / koşu (faz + tohum)": f"{nlit[k][0]} / {nlit[k][1]}"})
emit(tab(pd.DataFrame(rows)))
emit()

# -------------------------------------------------------------------------- dokunus tespiti
emit("## D-ham dokunuş tespiti ~ süre (B2; Wilson %95; oturum başına, havuzlanmadı)")
emit("`k/n = oran [alt, üst]`; payda = tam hızda değerlendirilebilen dokunuş; **tutulan karesi olmayan dokunuş kaçırma sayılır**, parantezde `(−x)` = o hücredeki örnekleme kaybı sayısı. k = 1 sütunları A3.2 ile aynıdır (kapı 2).")
emit()
BL = ["<0.2", "0.2-0.3", "0.3-0.5", "0.5-1.0", ">1.0"]
SES = ["s01", "s02", "s03", "s04", "s05", "s06"]


def wil(k, n):
    from math import sqrt
    if n <= 0: return "-"
    z = 1.96; p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return f"{k}/{n} = {p:.3f} [{max(0, c - h):.3f}, {min(1, c + h):.3f}]"


for k in K:
    emit(f"### k = {k} ({HZ[k]} Hz)")
    emit()
    rows = []
    for s in SES:
        rec = {"oturum": s}
        for bl in BL:
            x = DET[(DET.tag == f"k{k}") & (DET.session == s) & (DET.bin == bl)].iloc[0]
            rec[bl] = wil(int(x.k_detect), int(x.n_eval)) + (f" (−{int(x.n_lost_sampling)})" if x.n_lost_sampling else "")
        rows.append(rec)
    emit(tab(pd.DataFrame(rows)))
    emit()

emit("### Özet: süre kutusu başına havuzlanmış tespit (6 oturum; betimleyici — A3.2 havuzlamaz, burada yalnız k'ler arası karşılaştırma için)")
emit()
rows = []
for bl in BL:
    rec = {"süre kutusu (s)": bl}
    for k in K:
        x = DET[(DET.tag == f"k{k}") & (DET.bin == bl)]
        n, kk, lost = int(x.n_eval.sum()), int(x.k_detect.sum()), int(x.n_lost_sampling.sum())
        rec[f"k={k}"] = f"{kk}/{n} = {kk/n:.3f}" + (f" (−{lost})" if lost else "")
    rows.append(rec)
emit(tab(pd.DataFrame(rows)))
emit()
emit("### Normal tempolu oturumlar (s02–s04), süre ≥ 0,5 s (A3.2 'verdikt kümesi'; ön-yazılı A3.2 tahmini 'tespit ≥ 0,9')")
emit()
rows = []
for k in K:
    r = M.loc[k]
    rows.append({"k": k, "Hz": HZ[k], "tespit": f"{int(r.vs_hit)}/{int(r.vs_n)} = {r.vs_rate:.3f} [{r.vs_lo:.3f}, {r.vs_hi:.3f}]",
                 "tüm dokunuş tespiti (6 oturum)": f"{r.touch_detect_all:.3f}", "yalnız ≥1 tutulan karesi olanlar": f"{r.touch_detect_cond:.3f}",
                 "kaçırılan (örnekleme)": f"{int(r.touch_lost_sampling)}/{int(r.touch_n_eval)}"})
emit(tab(pd.DataFrame(rows)))
emit()
ph = D[D.kind.isin(["main", "phase"]) & (D.k > 1)]
emit("Faz kaydırması bu üç sayının yayılımı: s02–s04 (havuz, 3 oturum, n = 347) ≥ 0,5 s tespit oranı — " + "; ".join(f"k = {k}: {ph[ph.k==k].vs_rate.min():.3f}–{ph[ph.k==k].vs_rate.max():.3f}" for k in K[1:]) +
     "; tüm dokunuş tespiti — " + "; ".join(f"k = {k}: {ph[ph.k==k].touch_detect_all.min():.3f}–{ph[ph.k==k].touch_detect_all.max():.3f}" for k in K[1:]) + ".")
emit()
emit("**Oturum bazında (A3.2'nin kendi sınaması: verdikt kümesi s02, s03, s04'te her oturum için ≥ 0,5 s birleşik D-ham ≥ 0,9, nokta tahmini).** Birleşik = 0,5–1,0 s ve > 1,0 s kutuları toplamı; `ana` = faz 0, `faz aralığı` = o k'nin tüm fazlarında (k = 2: 2, k = 3: 3, k = 5: 5 faz) en düşük–en yüksek oran. 'Havuz' sütunu yukarıdaki tablonun aynısıdır.")
emit()
def _comb(tag, ses):
    q = DET[(DET.tag == tag) & (DET.session == ses) & (DET.bin.isin(["0.5-1.0", ">1.0"]))]
    return int(q.k_detect.sum()), int(q.n_eval.sum())
rows = []; _s_ok = {}
for k in K:
    tags = D[(D.k == k) & D.kind.isin(["main", "phase"])].tag.tolist()
    cells = {"k": k, "Hz": HZ[k]}
    for ses in ("s02", "s03", "s04"):
        h, n = _comb(f"k{k}", ses); rs = [_comb(t, ses) for t in tags]; rr = [x[0] / x[1] for x in rs]
        cells[f"{ses} ana"] = f"{h}/{n} = {h/n:.3f}"
        cells[f"{ses} faz aralığı"] = "-" if k == 1 else f"{min(rr):.3f}–{max(rr):.3f}"
        _s_ok[(k, ses)] = (h / n, min(rr), max(rr), h, n)
    hp = sum(_comb(f"k{k}", s_)[0] for s_ in ("s02", "s03", "s04")); npp = sum(_comb(f"k{k}", s_)[1] for s_ in ("s02", "s03", "s04"))
    assert (hp, npp) == (int(M.loc[k].vs_hit), int(M.loc[k].vs_n)), (k, hp, npp)   # havuz = DET toplami
    cells["havuz (s02+s03+s04)"] = f"{hp}/{npp} = {hp/npp:.3f}"
    pr = [sum(_comb(t, s_)[0] for s_ in ("s02", "s03", "s04")) / sum(_comb(t, s_)[1] for s_ in ("s02", "s03", "s04")) for t in tags]
    cells["havuz faz aralığı"] = "-" if k == 1 else f"{min(pr):.3f}–{max(pr):.3f}"
    rows.append(cells)
emit(tab(pd.DataFrame(rows)))
emit()
_fail = [(k, s_) for k in K for s_ in ("s02", "s03", "s04") if _s_ok[(k, s_)][0] < 0.9]
_failphase = [(k, s_) for k in K[1:] for s_ in ("s02", "s03", "s04") if _s_ok[(k, s_)][1] < 0.9]
emit("A3.2 sınaması ana koşuda (faz 0): " + ("tüm k ve oturumlarda ≥ 0,9" if not _fail else "başarısız: " + ", ".join(f"k = {k} {s_} ({_s_ok[(k, s_)][3]}/{_s_ok[(k, s_)][4]} = {_s_ok[(k, s_)][0]:.3f})" for k, s_ in _fail)) +
     ". Faz kaydırmasıyla en az bir fazda < 0,9 olan hücreler: " + (", ".join(f"k = {k} {s_} (en düşük {_s_ok[(k, s_)][1]:.3f})" for k, s_ in _failphase) if _failphase else "yok") + ".")
emit()

emit("### Lojistik uydurma: d50 / d90 (s), oturum başına, ana payda (kaçırma = 0)")
emit("A3.2 ile aynı uydurma (düzenlileştirmesiz ML; süre doğrusal saniye). `ok*` = nokta tahmini gözlenen süre aralığının dışında (ekstrapolasyon; yalnız sınır olarak okunur); diğer durumlar: `tümü` = tüm dokunuşlar yakalandı (d50 < en kısa dokunuş), `ayrışma`, `b1≤0` (uydurma tespit için süreyle artış vermiyor). Aralık = d50 için kullanılabilir belirsizlik aralığı.")
emit()
f = FIT[(FIT["mode"] == "all") & FIT.tag.isin([f"k{k}" for k in K])]
rows = []
for s in SES:
    rec = {"oturum": s}
    for k in K:
        x = f[(f.tag == f"k{k}") & (f.session == s)].iloc[0]
        st = x.status
        if st == "ok":
            if bool(x.extrap):
                txt = f"ok*: d50 {x.d50:.2f}, d90 {x.d90:.2f} → d50 ∈ [{x.lo50:.2f}, {'∞' if not np.isfinite(x.hi50) else format(x.hi50, '.2f')}]"
            else:
                txt = f"d50 {x.d50:.3f} (SE {x.d50_se:.3f}); d90 {x.d90:.3f} (SE {x.d90_se:.3f})"
        elif st.startswith("tumu"):
            txt = f"tümü ({int(x.hits)}/{int(x.n)}); d50 < {x.xmin:.2f}"
        elif st.startswith("tam ayrisma"):
            txt = f"ayrışma ({int(x.hits)}/{int(x.n)}); d50 ∈ [{x.lo50:.2f}, {x.hi50:.2f}]"
        else:
            txt = f"{st.split(' (')[0]} ({int(x.hits)}/{int(x.n)})"
        rec[f"k={k}"] = txt
    rows.append(rec)
emit(tab(pd.DataFrame(rows)))
emit()
emit("Ham d50/d90 sayıları (ekstrapolasyon dahil) ve koşullu (yalnız ≥ 1 tutulan karesi olan dokunuşlar) uydurma `03_frame_rate_fits.csv`'de (`mode` = all / cond).")
emit()

# -------------------------------------------------------------------------- okuma
r2, r3, r5 = M.loc[2], M.loc[3], M.loc[5]
emit("## Okuma (tahmini; nedensel iddia yok)")
emit(f"- **Durum tahmini (macro-F1) kare hızıyla monoton düşüyor**: ΔF1 k = 2 / 3 / 5 için {r2.df1_mean:+.4f} / {r3.df1_mean:+.4f} / {r5.df1_mean:+.4f}; GA'lar 0'ı ve −0,02'yi dışlıyor (k = 2: [{r2.df1_lo:+.4f}, {r2.df1_hi:+.4f}]). "
     f"Bu düşüş test karesinin azalmasından değil: aynı tutulan karelerde Faz A B2'ye göre fark benzer ({r2.df1_restricted_mean:+.4f} / {r3.df1_restricted_mean:+.4f} / {r5.df1_restricted_mean:+.4f}). "
     f"Tohum ve faz değişimi macro-F1'i az oynatıyor (faz aralığı k = 2 / 3 / 5: {D[(D.k==2)&D.kind.isin(['main','phase'])].f1_mean.max()-D[(D.k==2)&D.kind.isin(['main','phase'])].f1_mean.min():.4f} / {D[(D.k==3)&D.kind.isin(['main','phase'])].f1_mean.max()-D[(D.k==3)&D.kind.isin(['main','phase'])].f1_mean.min():.4f} / {D[(D.k==5)&D.kind.isin(['main','phase'])].f1_mean.max()-D[(D.k==5)&D.kind.isin(['main','phase'])].f1_mean.min():.4f}; tohum sd: k = 2 {D[(D.k==2)&(D.kind=='seed')].f1_mean.std(ddof=1):.4f}, k = 3 {D[(D.k==3)&(D.kind=='seed')].f1_mean.std(ddof=1):.4f}, k = 5 {D[(D.k==5)&(D.kind=='seed')].f1_mean.std(ddof=1):.4f}); yani sıra gürültüden net ayrılıyor. "
     f"k = 5'te pencere tek kare (span 0,1 s): düşüş en sert ({r5.f1_mean:.4f}); L = 1 hareket (hız/eğim) bağlamını yok ediyor, ama bu modelde bu nedensel olarak ayrıştırılmadı (k ve L birlikte değişti).")
emit(f"- **Anomali AUROC'u gürültülü ve debounce'a bağımlı.** Ana debounce kuralıyla (süre sabit) AUROC {r2.auroc:.3f} / {r3.auroc:.3f} / {r5.auroc:.3f} (ΔAUROC {r2.dauc_mean:+.3f} / {r3.dauc_mean:+.3f} / {r5.dauc_mean:+.3f}); GA'lar çok geniş (k = 2: [{r2.dauc_lo:+.3f}, {r2.dauc_hi:+.3f}]) çünkü kat AUROC'u (tüm koşularda) {FOLD[FOLD.tag.str.match(r'^k[1235](_[so]\d)?$')].auroc_d3_all.min():.2f}–{FOLD[FOLD.tag.str.match(r'^k[1235](_[so]\d)?$')].auroc_d3_all.max():.2f} arasında oynuyor. "
     f"Aynı OOF'la debounce değişince k = 2 AUROC'u {D[D.tag=='k2'].auroc.iloc[0]:.3f} → {D[D.tag=='k2_mg1'].auroc.iloc[0]:.3f} (MAX_GAP = 1 kare) → {D[D.tag=='k2_deb21'].auroc.iloc[0]:.3f} (2/1 kare) oynuyor; yani k = 2'de AUROC farkı kare hızının değil, süre-sabit yuvarlama kuralının sonucu olabilir. "
     f"Faz ve tohum ortalamalarında k = 2 / 3 / 5 AUROC'u (ana debounce) tohum ort. {D[(D.k==2)&(D.kind=='seed')].auroc.mean():.3f} / {D[(D.k==3)&(D.kind=='seed')].auroc.mean():.3f} / {D[(D.k==5)&(D.kind=='seed')].auroc.mean():.3f}; k = 1: {bA:.3f}. Tohumlar arası sd (ortalama AUROC): k = 1 {D[(D.k==1)&(D.kind=='seed')].auroc.std(ddof=1):.3f}, k = 2 {D[(D.k==2)&(D.kind=='seed')].auroc.std(ddof=1):.3f}, k = 3 {D[(D.k==3)&(D.kind=='seed')].auroc.std(ddof=1):.3f}, k = 5 {D[(D.k==5)&(D.kind=='seed')].auroc.std(ddof=1):.3f}; k ≥ 2 ortalamaları k = 1'in altında kalıyor ve bu ayrım tohumlar arasında tekrarlıyor, ama büyüklüğü (özellikle k = 2'de) debounce kuralına bağlı.")
emit(f"- **Tespit/FAR**: ana debounce kuralıyla tespit α = 0,10'da {r2.det_a10:.3f} / {r3.det_a10:.3f} / {r5.det_a10:.3f} (k = 1: {M.loc[1].det_a10:.3f}); FAR {r2.far_a10:.3f} / {r3.far_a10:.3f} / {r5.far_a10:.3f}. FAR α civarında kalıyor (SE ≈ 0,04), tespit düşüyor: kayıp duyarlılıkta, yanlış alarmda değil. Tohum ortalamasıyla tespit: k = 1 / 2 / 3 / 5: {D[(D.k==1)&(D.kind=='seed')].det_a10.mean():.3f} / {D[(D.k==2)&(D.kind=='seed')].det_a10.mean():.3f} / {D[(D.k==3)&(D.kind=='seed')].det_a10.mean():.3f} / {D[(D.k==5)&(D.kind=='seed')].det_a10.mean():.3f}.")
emit(f"- **Dokunuş tespiti seyreltmeye kısa dokunuşlar yüzünden duyarlı, uzun dokunuşlar değil**: normal tempolu oturumlarda (s02–s04) ≥ 0,5 s dokunuşlarda D-ham k = 1 / 2 / 3 / 5'te {M.loc[1].vs_rate:.3f} / {r2.vs_rate:.3f} / {r3.vs_rate:.3f} / {r5.vs_rate:.3f} (havuz, n = 347; A3.2'nin '≥ 0,9' tahmini havuzda k = 5'e kadar sürüyor, ama oturum bazında okunduğunda s04'te k = 5'te tutmuyor: oturum tablosuna bakın). "
     f"Kayıp yapısı örneklemeyle uyumlu: k = 5'te tutulan karesi olmayan {int(r5.touch_lost_sampling)} dokunuşun hepsi < 0,5 s (k = 2'de {int(r2.touch_lost_sampling)}, hepsi < 0,2 s). Tüm dokunuş tespiti {M.loc[1].touch_detect_all:.3f} → {r2.touch_detect_all:.3f} → {r3.touch_detect_all:.3f} → {r5.touch_detect_all:.3f}; yani 'kısa dokunuşlar' sınırı kare periyodu (0,1·k s) ile birlikte kayıyor, ama k ≤ 3 için 0,5 s üstü dokunuşlar neredeyse tam. "
     "D-ham kare düzeyinde bir tespittir; D1/D2'yi yöneten debounce'lu tanım (D-deb) bu çalışmada hesaplanmadı.")
_d = FIT[(FIT["mode"] == "all") & (FIT.session.isin(["s01", "s05"]))]
def _v(tag, ses, c): return _d[(_d.tag == tag) & (_d.session == ses)][c].iloc[0]
emit("- **Kısa dokunuşlu oturumlarda (s01, s05; lojistik uydurması mümkün olan tek oturumlar) d50 / d90 kare periyodu ile kayıyor**: s01 d90 " + " → ".join(f"{_v(f'k{k}','s01','d90'):.2f}" for k in K) + " s, s05 d90 " + " → ".join(f"{_v(f'k{k}','s05','d90'):.2f}" for k in K) + " s (k = 1 / 2 / 3 / 5); d50: s01 " + " → ".join(f"{_v(f'k{k}','s01','d50'):.2f}" for k in K) + ", s05 " + " → ".join(f"{_v(f'k{k}','s05','d50'):.2f}" for k in K) + " s. Yani 'güvenilir tespit için asgari dokunuş süresi' k büyüdükçe artıyor (s05'te 0,26 → 0,57 s), ama k = 1'de 0,26–0,49 s olan d90 aralığı k = 2'de 0,33–0,52 s'ye çıkıyor (yani 5 Hz'de sınır ≤ 0,1 s kaydı). Diğer oturumlarda neredeyse hiç kaçırma olmadığı için uydurma tahmin edilemiyor ya da ekstrapolasyon (tablo).")
emit("- **Tek cümlelik toplam**: bu model ve hat için 10 Hz'den inmek durum doğruluğunu (macro-F1) en az 0,03 düşürüyor ve literal diz kuralını geçen bir kare hızı yok; buna karşılık normal tempolu oturumlarda (s02–s04 havuzu, n = 347) ≥ 0,5 s dokunuşlar 2 Hz'e kadar faz 0'da ≥ 0,94 oranında yakalanıyor (tüm fazlarda en düşük havuz oranı k = 5'te " + f"{ph[ph.k==5].vs_rate.min():.3f}" + "; ama oturum bazında s04'te k = 5'te 98/109 = 0,899 < 0,9; faz aralığı için oturum tablosu) ve anomali AUROC'u debounce ayarına bağlı olarak 0,75–0,98 arasında değişiyor. Diz, bu ızgarada 10 Hz ile 5 Hz arasında; ara hızlar (örn. 7,5 Hz) denenmedi (tek model, tek operatör; keşif amaçlı).")
emit()

emit("## Uyarılar")
emit("1. **Öykünme, çip değil.** Düşük hızda gerçek bir radar farklı chirp/döngü ayarı kullanabilir (örn. kare başına daha fazla çırpı, daha fazla nokta); burada her tutulan kare 10 Hz kaydın aynen kopyası. Güç/CPU/bayt kazancı ölçülmedi, yalnız kare sayısı (bayt/s ∝ 1/k) hesaplandı.")
emit("2. **k ve L birlikte değişir** (brief'in tanımı): L_k = ceil(5/k) pencere süresi k = 2'de 0,5 s'yi korur ama k = 3'te 0,4 s, k = 5'te 0,1 s'dir; k'nin tek başına etkisi ile pencere uzunluğunun etkisi ayrıştırılmadı.")
emit("3. **Etiketler 10 Hz'de yumuşatılmış** (`y_true` ±3 kare mod süzgeci); seyreltilmiş modeller bu 10 Hz etiketi son tutulan karede öğreniyor. Etiketin kendisi seyreltme altında yeniden üretilmedi.")
emit("4. **Ziyaret süreleri 0,1·k s'ye nicelenir** (sıfırıncı-derece tutma); süre tabanlı D0/D2 skorları bunun bedelini taşır. Alternatif yeniden yapılandırma (ara değerleme) denenmedi.")
emit("5. **Debounce kural duyarlılığı büyük** (yukarıdaki tablo); k ≥ 2 için 'doğru' debounce ayrıca ayarlanabilir, ama bu bir hiperparametre taraması olurdu ve brief'in dışında; ana satır brief'in süre-sabit kuralına sadıktır.")
emit("6. **AUROC gürültülü** (50 civarı normal döngü, kat sd 0,03–0,15): literal kural fiilen 'AUROC GA'sı yeterince dar mı' sorusuna dönüşüyor; k ≥ 2'de ana debounce kuralında GA alt sınırı hiçbir koşuda (ana, faz, tohum) −0,03'ü dışlamıyor (yalnız k = 2 'kare sayısı sabit' varyantında alt sınır +0,0006). Bu yüzden asıl bağlayıcı olan kural macro-F1 ölçütüdür.")
emit("7. Tek operatör, tek görev, S1 5 kat (aynı oturumlarda). D-ham, ToF dokunuşuna karşı kare düzeyi tespit; anomali döngülerine ilişkin sonuçlar s02 doğal-sapma döngülerinin alarm durumuna bakılmadan hesaplandı (yasak kural uyuldu).")
emit()
emit("## Sapmalar")
emit("`DESIGN_B.md` 'Sapmalar (tarihli)' altına tek satır eklendi (2026-11-05 B.3).")
(REP / "03_frame_rate.md").write_text("\n".join(MD) + "\n", encoding="utf-8")

# -------------------------------------------------------------------------- sekil
OI = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "gray": "0.45", "sky": "#56B4E9", "verm": "#CC79A7"}
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))
xs = {1: 10, 2: 5, 3: 10 / 3, 5: 2}
# panel A macro-F1
a = ax[0]
for k in K:
    sd = D[(D.k == k) & (D.kind == "seed")]
    ph = D[(D.k == k) & (D.kind.isin(["main", "phase"]))]
    a.scatter([xs[k]] * len(sd), sd.f1_mean, s=14, color=OI["gray"], alpha=0.7, zorder=2, label="RF tohumu 1–4" if k == 1 else None)
    if k > 1:
        a.scatter([xs[k] * 1.0] * len(ph), ph.f1_mean, s=22, facecolors="none", edgecolors=OI["orange"], zorder=2, label="faz kaydırması" if k == 2 else None)
    fo = FOLD[FOLD.tag == f"k{k}"]
    a.errorbar(xs[k], M.loc[k].f1_mean, yerr=M.loc[k].f1_sd / np.sqrt(5) * T975, fmt="o", color=OI["blue"], capsize=3, zorder=3, label="ana (kat %95 GA)" if k == 1 else None)
a.axhline(M.loc[1].f1_mean - 0.02, color=OI["verm"], ls="--", lw=1)
a.text(1.95, M.loc[1].f1_mean - 0.0185, "Faz A B2 − 0,02", color=OI["verm"], fontsize=8)
a.set_xlabel("kare hızı (Hz)"); a.set_ylabel("S1 macro-F1"); a.set_title("A. Durum tahmini", fontsize=10)
a.set_xticks([10, 5, 10 / 3, 2]); a.set_xticklabels(["10", "5", "3,33", "2"]); a.invert_xaxis(); a.legend(fontsize=7, frameon=False, loc="upper right")
# panel B AUROC
b = ax[1]
for k in K:
    sd = D[(D.k == k) & (D.kind == "seed")]
    b.scatter([xs[k]] * len(sd), sd.auroc, s=14, color=OI["gray"], alpha=0.7, zorder=2, label="RF tohumu 1–4 (ana debounce)" if k == 1 else None)
    b.scatter([xs[k]], [M.loc[k].auroc], color=OI["blue"], s=40, zorder=3, label="ana (tohum 20261104)" if k == 1 else None)
    if k > 1:
        for tag, mk, lab in ((f"k{k}_mg1", "s", "MAX_GAP = 1 kare"), (f"k{k}_deb21", "^", "kare sayısı sabit 2/1")):
            r = D[D.tag == tag].iloc[0]
            b.scatter([xs[k]], [r.auroc], marker=mk, facecolors="none", edgecolors=OI["green"] if mk == "s" else OI["orange"], s=40, zorder=3, label=lab if k == 2 else None)
b.axhline(M.loc[1].auroc - 0.03, color=OI["verm"], ls="--", lw=1)
b.text(1.95, M.loc[1].auroc - 0.027, "Faz A B2 − 0,03", color=OI["verm"], fontsize=8)
b.set_xlabel("kare hızı (Hz)"); b.set_ylabel("D3 AUROC (tüm anomali)"); b.set_title("B. Anomali hattı (debounce'a duyarlı)", fontsize=10)
b.set_xticks([10, 5, 10 / 3, 2]); b.set_xticklabels(["10", "5", "3,33", "2"]); b.invert_xaxis(); b.legend(fontsize=7, frameon=False, loc="lower left")
# panel C touch detection by duration bin (pooled over 6 sessions)
c = ax[2]
cols = {1: OI["blue"], 2: OI["green"], 3: OI["orange"], 5: OI["verm"]}
for k in K:
    ys = []
    for bl in BL:
        x = DET[(DET.tag == f"k{k}") & (DET.bin == bl)]
        ys.append(x.k_detect.sum() / x.n_eval.sum())
    c.plot(range(len(BL)), ys, marker="o", color=cols[k], lw=1.6, label=f"{HZ[k]} Hz")
c.set_xticks(range(len(BL))); c.set_xticklabels(BL); c.set_xlabel("ToF dokunuş süresi (s)"); c.set_ylabel("D-ham tespit oranı (6 oturum havuzlu)")
c.set_ylim(0, 1.02); c.set_title("C. Dokunuş tespiti ~ süre (betimleyici)", fontsize=10); c.legend(fontsize=8, frameon=False, loc="lower right")
for axx in ax:
    axx.spines[["top", "right"]].set_visible(False)
fig.suptitle("B.3 — kare hızı öykünmesi (keşif; makaleye girmez)", fontsize=10)
fig.tight_layout()
fig.savefig(REP / "03_frame_rate.png", dpi=150)
print("done")

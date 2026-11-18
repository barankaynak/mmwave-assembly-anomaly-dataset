"""Faz B / B.1 rapor + sekil: reports/faz_b/01_tiny_forests.csv -> 01_tiny_forests.md, 01_tiny_forests.png.
Secim kurali 01_tiny_forests.py ile ayni sabitlerle (0.856-0.02, 0.924-0.03) burada da uygulanir ve stdout'a basilir."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = ROOT / "reports" / "faz_b"
df = pd.read_csv(REP / "01_tiny_forests.csv", dtype={"max_depth": str}, keep_default_na=False)
df["max_depth"] = df["max_depth"].astype(str)
F1_THR, AUC_THR = 0.856 - 0.02, 0.924 - 0.03
B2_F1, B2_AUC = 0.8564, 0.923697           # Faz A (rapor: 03_state.md / 04_anomaly_table.csv)
KB = 1024
# TI IWR6843 veri sayfasi SWRS219F (Nisan 2025): 1.75 MB = MSS prog RAM 512 + MSS data RAM 192 + DSP L1 64 + L2 256 + L3 768 (KB)
CHIP = [("MSS veri RAM 192 KB", 192), ("DSP L2 256 KB", 256), ("L3 768 KB", 768), ("toplam 1,75 MB", 1792)]
LINES = [("128 KB", 128), ("256 KB", 256), ("512 KB", 512)]
ALL_LINES = LINES + CHIP

df["ok_f1"] = df.f1_mean >= F1_THR
df["ok_auc"] = df.auroc_all >= AUC_THR
df["ok_both"] = df.ok_f1 & df.ok_auc
ok = df[df.ok_both].sort_values(["nodes_mean", "comps_per_window"])
sel = ok.iloc[0] if len(ok) else None

out = []
def H(s=""):
    out.append(s); print(s)

def kb(x): return f"{x / KB:.1f}"
def cfgname(r): return f"n={int(r.n_trees)}, d={r.max_depth}"

H("# B.1 — Küçük ormanlar: B2 modeli radar çipinin belleğine sığar mı? (tahmin; port yok)")
H()
H("Üretim: `faz_b/01_tiny_forests.py` (ızgara + OOF + anomali hattı) ve `faz_b/01b_tiny_forests_report.py` (bu rapor, şekil). Ham sayılar: `01_tiny_forests.csv`; her yapılandırmanın tam S1 OOF'u `reports/faz_b/oof/tf_n*_d*__S1__f*.csv.gz`. **Statü: keşif amaçlı, plan koşudan önce yazıldı (`DESIGN_B.md` B.1); makaleye girmez.** Çip üzerinde hiçbir şey çalıştırılmadı; yalnız model boyutu ve işlem sayısı hesaplandı.")
H()
H("## Yöntem (DESIGN_B.md'ye göre)")
H("- Özellikler, tohum (20261104), sınıf ağırlığı (`balanced`) ve S1 5 katı Faz A B2 ile aynı: 7 kare istatistiği × (son, ortalama, eğim) = 21 özellik, L = 5. Yalnız `n_estimators` ve `max_depth` değişir (30 yapılandırma × 5 kat = 150 model).")
H("- **Düğüm sayısı** = 5 kat modelinin toplam düğüm (iç + yaprak) sayısı ortalaması (`nodes_mean`); ek olarak tüm 16.248 pencereyle eğitilmiş modelin düğüm sayısı (`nodes_full`, dağıtılacak modele yakın; kurala girmez).")
H("- **Bayt** = düğüm × 8 B ve düğüm × 12 B (brief'teki varsayım; aşağıdaki uyarıya bakın).")
H("- **Pencere başına karşılaştırma** = test pencerelerinde her ağaçta kökten yaprağa geçilen iç düğüm sayısı, ağaçlar üzerinden toplanıp pencereler üzerinden ortalanır (= ortalama yol uzunluğu × ağaç sayısı). En kötü durum (ağaç `max_depth` toplamı) ayrıca verilir.")
H("- **Anomali hattı** her yapılandırmanın tam OOF `y_pred` çıktısıyla Faz A protokolünde koşuldu: ±3 kare debounce (MIN_BOX 2, MAX_GAP 1), `pred` segmentasyonu (tahmin edilen Box1 girişinden sonrakine), D1 + D2, D3 = min(1, 2 min(p1, p2)), standart conformal, Faz A rolleri (eğitim f+3,f+4; kal. f+1,f+2; test f). AUROC = kat bazlı AUROC'un ortalaması (± sd), `tüm anomali` grubu (seq135 + seq145 + fast + slow); FAR = α = 0,10'da 5 katın havuzlanmış normal test döngüleri (n ≈ 50, SE ≈ 0,042).")
H()

g = df[(df.n_trees == 300) & (df.max_depth == "None")].iloc[0]
H("## Eşdeğerlik kapısı (300 ağaç, derinlik sınırsız = Faz A B2)")
H(f"- macro-F1: {g.f1_mean:.4f} ± {g.f1_sd:.4f} (Faz A B2: {B2_F1:.4f}); B2'ye göre eşli kat farkı ortalaması {g.d_f1_vs_b2_mean:+.2e} (sd {g.d_f1_vs_b2_sd:.2e}).")
H(f"- B2 × pred D3 AUROC (tüm anomali): {g.auroc_all:.6f} ± {g.auroc_all_sd:.4f} (Faz A: {B2_AUC:.6f} ± 0,0352); FAR α = 0,10: {g.far_a10:.4f} (Faz A: 0,0784); tespit α = 0,10: {g.det_a10_all:.4f} (Faz A: 0,6335).")
gate = abs(g.f1_mean - B2_F1) < 5e-4 and abs(g.auroc_all - B2_AUC) < 1e-5
H(f"- **Sonuç: {'kapı geçti (Faz A B2 yeniden üretildi)' if gate else 'KAPI GEÇMEDİ — aşağıdaki sayılar Faz A ile birebir örtüşmüyor'}.**")
H()
H("## Çip belleği (kaynak)")
H("- **IWR6843**, Texas Instruments veri sayfası **SWRS219F** (Ekim 2018, Nisan 2025 revizyonu), *Features → Internal memory with ECC* ve bellek tablosu: 1,75 MB = MSS program RAM (TCM-A) 512 KB + MSS veri RAM (TCM-B) 192 KB + DSP L1 64 KB (L1P 32 + L1D 32) + DSP L2 256 KB + L3 radar veri küpü RAM 768 KB.")
H("- URL: https://www.ti.com/lit/ds/symlink/iwr6843.pdf (ürün sayfası https://www.ti.com/product/IWR6843).")
H("- Çizgiler: 128 / 256 / 512 KB (brief) ve çip: MSS veri RAM 192 KB, DSP L2 256 KB, L3 768 KB, toplam 1792 KB. **Uyarı:** bu bellekler paylaşılır (L3 radar küpünü tutar; program RAM üretici yazılım + uygulama kodunu tutar; DSP L2 DSP kodu/verisi + ara tamponlar). Model sabitleri için fiilen kullanılabilir pay bu toplamdan küçüktür; aşağıdaki \"sığar\" yalnız model sabitlerinin tek başına bu kovaya girdiği anlamına gelir.")
H()
H("## Tüm ızgara (S1, 5 kat)")
H("`nodes` = kat ortalaması; `full` = tüm veriyle eğitilmiş modelin düğüm sayısı; B8/B12 = KB (1 KB = 1024 B), kat ortalaması düğüm × 8 / 12 B; `kars.` = pencere başına karşılaştırma (ortalama; en kötü durum); `Δ B2` = macro-F1 eşli kat farkı (ortalama ± sd); AUROC = D3, tüm anomali, kat ortalaması ± sd; tespit/FAR α = 0,10.")
H()
H("| ağaç | derinlik | macro-F1 | Δ B2 | düğüm | full | B8 KB | B12 KB | kars. (en kötü) | yol uz. | AUROC D3 | seq | tempo | tespit | FAR (SE) | F1 ≥ 0,836 | AUROC ≥ 0,894 |")
H("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in df.itertuples():
    H(f"| {r.n_trees} | {r.max_depth} | {r.f1_mean:.4f} ± {r.f1_sd:.4f} | {r.d_f1_vs_b2_mean:+.4f} ± {r.d_f1_vs_b2_sd:.4f} | {r.nodes_mean:,.0f} | {r.nodes_full:,} | "
      f"{kb(r.bytes8_mean)} | {kb(r.bytes12_mean)} | {r.comps_per_window:.1f} ({r.worst_comps:.0f}) | {r.mean_path_len:.2f} | {r.auroc_all:.4f} ± {r.auroc_all_sd:.4f} | "
      f"{r.auroc_seq:.3f} | {r.auroc_tempo:.3f} | {r.det_a10_all:.3f} | {r.far_a10:.3f} ({r.far_se_a10:.3f}) | {'evet' if r.ok_f1 else 'hayır'} | {'evet' if r.ok_auc else 'hayır'} |")
H()
H(f"Referans (Faz A B2 = 300 ağaç, sınırsız derinlik): macro-F1 {B2_F1:.4f} ± 0,0122, AUROC {B2_AUC:.4f} ± 0,0352, FAR 0,0784; düğüm {g.nodes_mean:,.0f} (kat ort.), {g.nodes_full:,} (tam veri) → 8 B/düğümle {g.bytes8_mean/KB/KB:.2f} MB, 12 B/düğümle {g.bytes12_mean/KB/KB:.2f} MB; pencere başına {g.comps_per_window:.0f} karşılaştırma.")
H()
H("## Önceden yazılmış seçim kuralı (kural sabitleri aynen)")
H(f"Kural: macro-F1 ≥ 0,856 − 0,02 = **{F1_THR:.3f}** **ve** AUROC ≥ 0,924 − 0,03 = **{AUC_THR:.3f}** olan en küçük (bayt) yapılandırma; yoksa \"sığan yapılandırma yok\". Bayt sırası 8 ve 12 B/düğümde aynıdır (düğüm sayısı); eşitlikte az karşılaştırma.")
H(f"- Yalnız macro-F1 koşulunu sağlayan: {int(df.ok_f1.sum())} / {len(df)}; yalnız AUROC koşulunu sağlayan: {int(df.ok_auc.sum())} / {len(df)}; ikisini birden: **{int(df.ok_both.sum())} / {len(df)}**.")
if sel is None:
    H("- **Sonuç: sığan yapılandırma yok.**")
else:
    H("- Kuralı sağlayanlar (küçükten büyüğe):")
    H()
    H("| ağaç | derinlik | macro-F1 | AUROC D3 | düğüm (kat ort.) | B8 KB | B12 KB | kars. |")
    H("|---|---|---|---|---|---|---|---|")
    for r in ok.itertuples():
        H(f"| {r.n_trees} | {r.max_depth} | {r.f1_mean:.4f} | {r.auroc_all:.4f} | {r.nodes_mean:,.0f} | {kb(r.bytes8_mean)} | {kb(r.bytes12_mean)} | {r.comps_per_window:.1f} |")
    H()
    H(f"- **Seçilen: {cfgname(sel)}** — macro-F1 {sel.f1_mean:.4f} ± {sel.f1_sd:.4f} (B2'ye göre {sel.d_f1_vs_b2_mean:+.4f} ± {sel.d_f1_vs_b2_sd:.4f}), AUROC {sel.auroc_all:.4f} ± {sel.auroc_all_sd:.4f} (B2: {B2_AUC:.4f}), FAR {sel.far_a10:.3f}, tespit {sel.det_a10_all:.3f} (B2: 0,078 / 0,633); "
      f"{sel.nodes_mean:,.0f} düğüm (kat ort.; tam veri {sel.nodes_full:,}) = {kb(sel.bytes8_mean)} KB (8 B) / {kb(sel.bytes12_mean)} KB (12 B) [tam veri: {kb(sel.bytes8_full)} / {kb(sel.bytes12_full)} KB]; pencere başına {sel.comps_per_window:.1f} karşılaştırma (en kötü {sel.worst_comps:.0f}).")
    fits = [nm for nm, k in ALL_LINES if sel.bytes8_mean <= k * KB]
    fits12 = [nm for nm, k in ALL_LINES if sel.bytes12_mean <= k * KB]
    fitsf8 = [nm for nm, k in ALL_LINES if sel.bytes8_full <= k * KB]
    fitsf12 = [nm for nm, k in ALL_LINES if sel.bytes12_full <= k * KB]
    H(f"- Seçilen yapılandırmanın sığdığı çizgiler: kat ortalaması 8 B → {', '.join(fits)}; 12 B → {', '.join(fits12)}; tam-veri modeli 8 B → {', '.join(fitsf8)}; 12 B → {', '.join(fitsf12)}. **128 KB çizgisine sığmaz** (kat ortalaması 8 B bile {kb(sel.bytes8_mean)} KB).")
    H(f"- B2 ile yan yana: B2 {B2_F1:.4f} / {B2_AUC:.4f} ile {g.bytes8_mean/KB:,.0f}–{g.bytes12_mean/KB:,.0f} KB; seçilen yapılandırma ≈ {g.nodes_mean/sel.nodes_mean:.0f} kat daha az düğümle {sel.f1_mean:.4f} / {sel.auroc_all:.4f}. **B2 birincil model olarak kalır.**")
H()
H("## Hangi bellek kovasına ne sığar (düğüm kat ortalaması; bayt = düğüm × B/düğüm)")
H("Her kova için kovaya sığan yapılandırmalar arasında en yüksek macro-F1 ve en yüksek AUROC. Son sütun: kovaya sığan ve kuralın iki koşulunu da sağlayan yapılandırma sayısı.")
H()
H("| bellek çizgisi | B/düğüm | sığan yapılandırma sayısı | en yüksek macro-F1 (yapılandırma) | en yüksek AUROC (yapılandırma) | kural koşullarını sağlayan sığan sayısı |")
H("|---|---|---|---|---|---|")
fits_json = {}
for nm, k in ALL_LINES:
    for bpn in (8, 12):
        d = df[df.nodes_mean * bpn <= k * KB]
        n_ok = int(d.ok_both.sum()) if len(d) else 0
        if len(d):
            a = d.loc[d.f1_mean.idxmax()]; b = d.loc[d.auroc_all.idxmax()]
            H(f"| {nm} | {bpn} | {len(d)} | {a.f1_mean:.4f} ({cfgname(a)}) | {b.auroc_all:.4f} ({cfgname(b)}) | {n_ok} |")
        else:
            H(f"| {nm} | {bpn} | 0 | – | – | 0 |")
H()
H("## Dikkat edilecekler (koşudan önce yazılan + sonuçtan çıkan)")
H("- **B/düğüm varsayımı:** 8 B = özellik indeksi + eşik + iki çocuk işaretçisi (yaprakta sınıf); 12 B = yaprakta sınıf olasılıklarını da taşıyan daha cömert bir düzen. Gerçek bir C uygulaması farklı olabilir (ör. yapraklarda 6 sınıf olasılığı için ağaç başına ek dizi; ağaç sayısı büyükse ofset tabloları). Sıralama (hangi yapılandırma daha küçük) bu varsayımlardan etkilenmez; mutlak bayt etkilenir.")
H("- **Oy sayımı:** burada `predict_proba` ortalamasının argmax'ı kullanıldı (Faz A B2 ile aynı); çipte sınıf oylarının sayılması ya da yaprak olasılıklarının toplanması gerekir. Özellik çıkarımı (kare istatistikleri, 5 karelik pencere) ve nokta bulutu tamponları bu boyuta dahil değildir.")
H("- **Anomali AUROC'u gürültülüdür:** kat sd'si 0,04–0,19; ~50 normal döngü ve 5 kat. Küçük AUROC farkları (≈ 0,03) kat sd'sinin içindedir; AUROC'un düğüm sayısıyla tekdüze artmaması (aşağıdaki şekil) buna uyar. FAR'ın SE'si ≈ 0,04.")
H("- Çip belleği: bkz. yukarıdaki uyarı. Hiçbir çip üzerinde ölçüm, derleme ya da gecikme testi yapılmadı; \"sığar\" yalnız boyut hesabıdır.")
H("- Kural sabitleri (0,856 / 0,924) Faz A B2'nin yuvarlanmış değerleridir (B2: 0,8564 / 0,9237); kural bu yuvarlanmış sabitlerle aynen uygulandı.")
H()
H("## Şekil")
H("`01_tiny_forests.png`: üst sıra macro-F1, alt sıra D3 AUROC (tüm anomali); sol sütun 8 B/düğüm, sağ sütun 12 B/düğüm; x ekseni model baytı (log). Renk = ağaç sayısı, işaret = derinlik. Kesikli yatay: Faz A B2 (0,8564 / 0,9237); noktalı yatay: kural eşikleri (0,836 / 0,894). Siyah dikey: 128 / 256 / 512 KB; kırmızı dikey: çip (MSS veri RAM 192 KB, DSP L2 256 KB, L3 768 KB, toplam 1792 KB). Siyah 256 KB çizgisi ile kırmızı DSP L2 256 KB çizgisi aynı x'tedir (üst üste biner). Not: rapordaki hesaplanan sayılar noktalı, elle yazılan Faz A sabitleri virgüllü yazılmıştır; aynı nicelikler.")
H("Bayt sırası kat ortalaması düğüm sayısıyla ve tam-veri modeli düğüm sayısıyla aynıdır (n=10/None 20.460 < n=25/None 50.479 < ...), yani seçim hangi bayt tanımı kullanılsa değişmez.")
(REP / "01_tiny_forests.md").write_text("\n".join(out) + "\n", encoding="utf-8")

# ---- sekil
cm = {10: "#d7301f", 25: "#fc8d59", 50: "#fdbb84", 100: "#41b6c4", 300: "#225ea8"}
mk = {"4": "o", "6": "s", "8": "^", "10": "D", "12": "v", "None": "*"}
fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
for j, bpn in enumerate((8, 12)):
    for i, (col, ref, thr, lab) in enumerate((("f1_mean", B2_F1, F1_THR, "S1 macro-F1"), ("auroc_all", B2_AUC, AUC_THR, "D3 AUROC (tüm anomali)"))):
        ax = axs[i, j]
        for r in df.itertuples():
            ax.scatter(r.nodes_mean * bpn, getattr(r, col), c=cm[r.n_trees], marker=mk[r.max_depth], s=90 if r.max_depth != "None" else 170,
                       edgecolors="k", linewidths=0.5, zorder=3)
        ax.axhline(ref, ls="--", c="gray", lw=1); ax.axhline(thr, ls=":", c="gray", lw=1)
        for nm, k in LINES:
            ax.axvline(k * KB, c="k", lw=0.8);
            if i == 0: ax.text(k * KB, ax.get_ylim()[0], " " + nm, rotation=90, va="bottom", ha="right", fontsize=8)
        for nm, k in CHIP:
            ax.axvline(k * KB, c="#cb181d", lw=0.8, ls="-.")
            if i == 0: ax.text(k * KB, ax.get_ylim()[0], " " + nm, rotation=90, va="bottom", ha="left", fontsize=8, color="#cb181d")
        ax.set_xscale("log"); ax.set_ylabel(lab); ax.set_title(f"{lab}, {bpn} B/düğüm"); ax.grid(alpha=0.25)
        if i == 1: ax.set_xlabel(f"model baytı (düğüm × {bpn} B)")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", c=c, mec="k", label=f"{n} ağaç") for n, c in cm.items()]
h += [Line2D([], [], marker=m, ls="", c="w", mec="k", label=f"derinlik {d}") for d, m in mk.items()]
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.legend(handles=h, loc="lower center", fontsize=9, ncol=11)
fig.set_size_inches(15, 10.4); fig.savefig(REP / "01_tiny_forests.png", dpi=130)
print("rapor + sekil yazildi")

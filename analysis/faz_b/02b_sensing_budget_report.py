"""Faz B / B.2 rapor + sekil: reports/faz_b/02_sensing_budget*.csv -> 02_sensing_budget.md, 02_sensing_budget.png. Diz kurali ve beklentiler
burada CSV'den bagimsiz yeniden hesaplanir ve stdout'a da basilir."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import b_frames as bf  # noqa (yalniz paired_ci/knee sabitleri icin)

REP = ROOT / "reports" / "faz_b"
df = pd.read_csv(REP / "02_sensing_budget.csv")
fo = pd.read_csv(REP / "02_sensing_budget_folds.csv")
sel_json = json.load(open(REP / "02_sensing_budget_selection.json"))
nf = REP / "02_noise_floor.csv"
nz = pd.read_csv(nf) if nf.exists() else None
base = df[df.tag == "base"].iloc[0]
REF_F1 = fo[fo.tag == "base"].sort_values("fold")["ref_macro_f1"].to_numpy()
REF_AUC = fo[fo.tag == "base"].sort_values("fold")["ref_auroc"].to_numpy()

out = []
def H(s=""):
    out.append(s); print(s)

def f4(x): return f"{x:.4f}"
def sg(x): return f"{x:+.4f}"
def ci(m, lo, hi): return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
def yn(b): return "evet" if bool(b) else "hayır"
def lab(r): return f"`{r.tag}` — {r.label}"

H("# B.2 — Algılama bütçesi: çip içi nokta bulutunun ne kadarı yetiyor? (öykünme)")
H()
H("Üretim: `faz_b/b_frames.py` (ham noktalardan kare istatistikleri + bayt hesabı), `faz_b/02a_gate.py` (kapı 1), `faz_b/02_sensing_budget.py` (seviyeler, eğitim, anomali hattı, diz seçimi), `faz_b/02c_noise_floor.py` (ek: tohum gürültüsü), `faz_b/02d_level_seeds.py` (ek: seviye başına tohum 1–4), `faz_b/02b_sensing_budget_report.py` (bu rapor). Ham sayılar: `02_sensing_budget.csv` (seviye başına tek satır, tüm sütunlar), `02_sensing_budget_folds.csv` (kat bazlı macro-F1 ve AUROC), `02_sensing_budget_selection.json`, `02_noise_floor.csv`, `02_level_seeds.csv`; OOF'lar `reports/faz_b/oof/sb_*__S1__f*.csv.gz` (her seviye için tam 16.248 pencere, 5 kat). **Statü: keşif amaçlı, plan koşudan önce yazıldı (`DESIGN_B.md` B.2); `main.tex`'e girmez.** Hiçbir şey çipte çalıştırılmadı; bütün seviyeler kaydedilmiş ham noktalardan yazılımla öykünüldü.")
H()
H("## Yöntem (DESIGN_B.md'ye göre)")
H("- **Kare kümesi** her seviyede aynı 16.272 kare (`fa_common.frame_table()`; etiketler ve katlar bunlara bağlı). Bir öykünme bir karede hiç nokta bırakırsa o karede `n_points = 0`, `snr_sum = 0` olur, diğer istatistikler NaN'dır ve B2'nin Faz A dolgusuyla (cx, cy = −2; diğerleri 0) doldurulur.")
H("- **İstatistikler** ham `mmwave_data.csv` noktalarından yeniden hesaplanır: cx, cy (SNR ağırlıklı), n_points, snr_sum, mean_abs_v, std_x, std_y. İşlem sırası: nokta kümesi (c) → SNR eşiği (a) → tepe gruplama (d) → hız (b).")
H("- **B2**: 7 istatistik × (son, ortalama, eğim) = 21 özellik, L = 5, RandomForest 300 ağaç, `class_weight=balanced`, tohum 20261104, `n_jobs = 3`, S1 5 kat (Faz A ile aynı katlar). Her seviye için tam OOF yazıldı.")
H("- **Anomali hattı**: OOF `y_pred` → ±3 kare debounce (MIN_BOX 2, MAX_GAP 1) → `pred` segmentasyonu → D1 + D2 → D3 = min(1, 2 min(p1, p2)), standart conformal, Faz A rolleri; AUROC = kat bazlı AUROC'un ortalaması, `tüm anomali` grubu; FAR α = 0,10'da 5 katın havuzlanmış normal test döngüleri (n ≈ 50, SE ≈ 0,04).")
H("- **Eşli fark**: seviye − Faz A B2, kat bazında (5 kat), ortalama ve %95 t-GA (df = 4, t = 2,776, sd/√5). Faz A B2'nin kat bazlı AUROC'u `04_anomaly_table.csv`'de yok; aynı hat `fc.load_oof('b2','S1')` ile çalıştırılarak üretildi (ortalama 0,923697 = Faz A).")
H("- **Eksenler tek tek değişir** (diğerleri taban: τ = 15 dB, hız tam, noktalar merged, gruplama kapalı): 4 + 2 + 2 + 1 = 9 seviye + taban = 10 seviye, 50 RF modeli. Birleşimler ayrıca, ikincil olarak koşuldu (aşağıda).")
H("- **Ölçek**: kayıttaki SNR 0,1 dB biriminde; tüm oturumlarda `min(snr) = 150` (= 15,0 dB, cfg `cfarCfg … 15`). τ dB'e karşılık gelen eşik `snr ≥ 10·τ` (17,5 → 175, 20 → 200, 22,5 → 225, 25 → 250).")
H("- **Hız seviyeleri**: *tam* = |v| (Faz A). *Çıkarılmış* = `mean_abs_v` sabit 0. *Yalnız işaret* = bir işaret-biti akışı |v| vermez; merged kümede (her nokta v ≠ 0) `mean(|sign v|)` ≡ 1 olurdu, yani seviye *çıkarılmış* ile aynı sabit özelliğe düşerdi. İşaret akışının destekleyebildiği tek hız istatistiği `mean(sign v)` olduğundan bu, `mean_abs_v` özellik yuvasına konuldu (özellik sayısı 21 kalır). Bkz. Sapmalar.")
H("- **Tepe gruplama (açık)**: her (kare, menzil kutusu, açı kutusu) hücresinde yalnız en yüksek SNR'li nokta; eşitlikte dosyadaki ilk. Kayıttaki değerler kesiklidir: 6 menzil kutusu (adım ≈ 0,038 m, ayrım eşiği 1e-4 m), 57 açı kutusu (eşik 1e-3°). Aynı hücredeki noktalar Doppler kutusuyla ayrışır (hız adımı 0,1207 m/s).")
H("- **Bayt**: (V2, birincil) yalnız nokta bulutu TLV'leri: TLV1 8 B başlık + n × (x, y, z float32 = 12 B + hız alanı) ve TLV7 8 B başlık + n × 4 B (SNR, gürültü uint16); hız alanı tam 4 B, işaret 1 B (bayt hizalı, özel biçim), çıkarılmış 0 B. Taban: 16 + 20·n (n = 9 → 196 B, `05_gaps.md` V2). (V3, ikincil) hattın okuduğu yük: x, y float32 (8 B) + hız + SNR uint16 (2 B), başlıksız. Nokta kalmayan karede 0 B. Bayt/s = 10 Hz × bayt/kare (`05_gaps.md` geleneği: kayıtlı kareler; boş radar karelerini saymaz). Ham ADC 57.344 B/kare (`05_gaps.md`). Tabloda ortalama bayt/kare (16.272 karede); medyan sütunu CSV'de (`v2_B_frame_median`).")
H()

# ------------------------------------------------------------------ kapilar
H("## Kapılar")
H("- **Kapı 1 (kare istatistikleri)**: `merged / tam` seviyesinin 7 istatistiği `fa_common.frame_table()` ile **16.272 karenin hepsinde birebir aynı**: her istatistik için max |fark| = 0,0 (bit düzeyinde; NaN yok). Not: s05'in ham dosyasında `merged_df.csv`'de olmayan 2 kare (9 nokta) var; bu kareler kare kümesinin dışında bırakıldı. Birebir `n_points` eşleşmesi bunların tek fark olduğunu gösterir.")
H(f"- **Kapı 2 (taban satırı = Faz A B2)**: yeniden üretilmiş istatistiklerle eğitilen taban modeli kat bazında Faz A B2 ile aynı: macro-F1 {f4(base.f1_mean)} ± {f4(base.f1_sd)} (Faz A 0,856448), max |kat farkı| = 0 (OOF birebir); AUROC {f4(base.auroc)} ± {f4(base.auroc_sd)} (Faz A 0,923697); FAR {f4(base.far_a10)} (Faz A 0,0784); tespit {f4(base.det_a10)} (Faz A 0,6335).")
H("- **Sonuç: kapılar geçti; B.2 koşuldu.**")
H()

# ------------------------------------------------------------------ ana tablo
H("## Tüm seviyeler (S1, 5 kat, tek eksen değişir)")
H("`nokta/kare` = kare başına ortalama nokta (taban yüzdesi); `boş` = nokta kalmayan kare sayısı (16.272'den); bayt V2 / V3 = ortalama bayt/kare (ADC÷ = 57.344 / V2); ΔF1 ve ΔAUROC = Faz A B2'ye göre eşli kat farkı, ortalama [%95 GA]; AUROC = D3, `tüm anomali`, kat ortalaması ± sd; tespit / FAR α = 0,10 (FAR: SE).")
H()
H("| seviye | nokta/kare (taban %) | boş | V2 B/kare | V2 B/s | V3 B/kare | ADC÷ | macro-F1 ± sd | ΔF1 [GA] | AUROC ± sd | ΔAUROC [GA] | seq / tempo AUROC | tespit | FAR (SE) |")
H("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in df.iterrows():
    H(f"| {lab(r)} | {r.n_points_mean:.2f} ({r.pts_pct_of_base:.0f}%) | {int(r.empty_frames)} | {r.v2_B_frame:.1f} | {r.v2_B_s:.0f} | {r.v3_B_frame:.1f} | {r.adc_ratio_v2:.0f}× | "
      f"{f4(r.f1_mean)} ± {f4(r.f1_sd)} | {ci(r.df1_mean, r.df1_lo, r.df1_hi)} | {f4(r.auroc)} ± {f4(r.auroc_sd)} | {ci(r.dauc_mean, r.dauc_lo, r.dauc_hi)} | "
      f"{f4(r.auroc_seq)} / {f4(r.auroc_tempo)} | {r.det_a10:.3f} | {r.far_a10:.4f} ({r.far_se_a10:.3f}) |")
H()
H("Kat bazlı en küçük / en büyük fark (macro-F1; AUROC) CSV'de (`df1_min`, `df1_max`, `dauc_min`, `dauc_max`). Birleşim satırlarının tanımı: `combo_literal` = eksen başına literal kuralla seçilen seviyelerin birleşimi; `combo_point` = yalnız nokta tahmini kuralıyla seçilenlerin birleşimi (aşağıda). İkisi de brief dışı, ikincil.")
H()

# ------------------------------------------------------------------ diz kurali
H("## Önceden yazılmış diz kuralı")
H("Kural (brief): en yalın seviye (burada: eksen içinde en az bayt/kare, V2), macro-F1 düşüşü ≤ 0,02 **ve** AUROC düşüşü ≤ 0,03 sağlayan seviyedir; kat GA'ları daha büyük bir düşüşü dışlamalı. Uygulama: dört ölçüt ayrı ayrı — (i) ΔF1 ≥ −0,02, (ii) ΔF1'in GA alt sınırı ≥ −0,02, (iii) ΔAUROC ≥ −0,03, (iv) ΔAUROC'un GA alt sınırı ≥ −0,03. **Literal geçiş = dördü birden.** Yalnız nokta tahmini (i + iii) ayrıca verilir; bu GA koşulunu gevşetir ve brief'in kuralı değildir.")
H()
H("| seviye | (i) ΔF1 ≥ −0,02 | (ii) GA alt ≥ −0,02 | (iii) ΔAUROC ≥ −0,03 | (iv) GA alt ≥ −0,03 | **literal** | yalnız nokta |")
H("|---|---|---|---|---|---|---|")
for _, r in df[df.tag != "base"].iterrows():
    H(f"| {lab(r)} | {yn(r.pass_f1_point)} | {yn(r.pass_f1_ci)} ({r.df1_lo:+.4f}) | {yn(r.pass_auc_point)} | {yn(r.pass_auc_ci)} ({r.dauc_lo:+.4f}) | **{yn(r.pass_literal)}** | {yn(r.pass_point_only)} |")
H()
# yeniden hesap
sel = {}
for axis in "abcd":
    g = df[(df.axis == axis) | (df.tag == "base")].sort_values(["v2_B_frame", "pts_pct_of_base"])
    ok = g[g.pass_literal]; okp = g[g.pass_point_only]
    sel[axis] = (ok.iloc[0].tag if len(ok) else None, okp.iloc[0].tag if len(okp) else None, list(g.tag))
    assert sel[axis][0] == sel_json[axis]["literal"] and sel[axis][1] == sel_json[axis]["point_only"], axis
names = {"a": "(a) SNR eşiği", "b": "(b) hız", "c": "(c) noktalar", "d": "(d) tepe gruplama"}
H("| eksen | bayt sırası (en yalından) | **literal seçim** | yalnız nokta tahmini seçimi | sonuç |")
H("|---|---|---|---|---|")
verdict = {}
for axis in "abcd":
    lit, pt, order = sel[axis]
    if lit == "base" or lit is None:
        v = "taban dışında geçen yalın seviye yok: **tam bütçe gerekli** (bu eksende)" if axis != "c" else "**tam bütçe (= merged, Faz A) gerekli**; ham gerekmiyor (daha pahalı ve F1 daha düşük), yalnız v = 0 geçmiyor"
    else:
        v = f"`{lit}` yeterli"
    verdict[axis] = v
    H(f"| {names[axis]} | {' < '.join(order)} | `{lit}` | `{pt}` | {v} |")
H()
for a_ in "abcd":
    print(f"[diz] {a_}: literal={sel[a_][0]} nokta={sel[a_][1]}")

# birlesimler
H("### Birleşimler (ikincil; brief dışı)")
cmb = df[df.axis == "combo"]
if len(cmb):
    for _, r in cmb.iterrows():
        H(f"- `{r.tag}`: τ = {r.tau_db:g} dB, hız = {r['vel']}, noktalar = {r.pts}, gruplama = {'açık' if r.group else 'kapalı'} → {r.n_points_mean:.2f} nokta/kare ({r.pts_pct_of_base:.0f}%), V2 {r.v2_B_frame:.1f} B/kare = {r.v2_B_s:.0f} B/s (ADC÷ {r.adc_ratio_v2:.0f}×); macro-F1 {f4(r.f1_mean)} (ΔF1 {ci(r.df1_mean, r.df1_lo, r.df1_hi)}); AUROC {f4(r.auroc)} (ΔAUROC {ci(r.dauc_mean, r.dauc_lo, r.dauc_hi)}); tespit {r.det_a10:.3f}; FAR {r.far_a10:.4f}. Literal geçiş: **{yn(r.pass_literal)}** (ΔF1 alt sınır {r.df1_lo:+.4f}; ΔAUROC alt sınır {r.dauc_lo:+.4f}). Yalnız nokta tahmini: {yn(r.pass_point_only)}.")
    H("- Eksen başına geçen seviyelerin birleşimi eksenler arası etkileşimi test eder; hiçbir zaman eksen başına sonucun yerine geçmez.")
else:
    H("- Birleşim koşulmadı (hiçbir eksende taban dışında geçen seviye yok).")
H()

# ------------------------------------------------------------------ beklentiler
H("## Önceden yazılmış beklentiler (literal)")
rc = df[df.tag == "c_raw"].iloc[0]; rd = df[df.tag == "d_group"].iloc[0]
c_ok = abs(rc.df1_mean) <= 0.01; d_ok = abs(rd.df1_mean) <= 0.01
H(f"- **(c) ham ile merged arasında |Δ macro-F1| ≤ 0,01:** ΔF1 = {sg(rc.df1_mean)} (GA [{rc.df1_lo:+.4f}, {rc.df1_hi:+.4f}]; ham {f4(rc.f1_mean)} vs merged {f4(base.f1_mean)}) → |Δ| = {abs(rc.df1_mean):.4f}. **Beklenti {'TUTTU' if c_ok else 'TUTMADI'}** (nokta tahmini). Not: kat GA'sı 0'ı dışlıyor (ham v = 0 noktaları eklemek F1'i hafifçe düşürüyor); beklenti nokta tahminine göre yazılmıştı.")
H(f"- **(d) gruplama açıkken |Δ macro-F1| ≤ 0,01:** ΔF1 = {sg(rd.df1_mean)} (GA [{rd.df1_lo:+.4f}, {rd.df1_hi:+.4f}]; gruplamalı {f4(rd.f1_mean)}) → |Δ| = {abs(rd.df1_mean):.4f}. **Beklenti {'TUTTU' if d_ok else 'TUTMADI'}**" + (f" (eşiği {abs(rd.df1_mean)-0.01:.4f} aştı; yani sınırda ama literal olarak tutmadı)." if not d_ok else "."))
print(f"[beklenti] c: |dF1|={abs(rc.df1_mean):.4f} -> {'TUTTU' if c_ok else 'TUTMADI'}; d: |dF1|={abs(rd.df1_mean):.4f} -> {'TUTTU' if d_ok else 'TUTMADI'}")
H()

# ------------------------------------------------------------------ tohum gurultusu
H("## Ek: tohum gürültüsü (brief dışı; AUROC farklarını okumak için)")
if nz is not None:
    H(f"Taban seviyesinde (merged / tam) yalnız RF tohumu değişti (tohum 1–8, 40 model; diğer her şey sabit). **macro-F1** ortalama {nz.f1_mean.mean():.4f}, sd {nz.f1_mean.std(ddof=1):.4f}, aralık [{nz.f1_mean.min():.4f}, {nz.f1_mean.max():.4f}] (Faz A tohumu 20261104: {base.f1_mean:.4f}). **AUROC (D3, tüm anomali)** ortalama {nz.auroc.mean():.4f}, sd {nz.auroc.std(ddof=1):.4f}, aralık [{nz.auroc.min():.4f}, {nz.auroc.max():.4f}] (Faz A tohumu: {base.auroc:.4f}). tespit α = 0,10: ortalama {nz.det_a10.mean():.3f}, sd {nz.det_a10.std(ddof=1):.3f}; **FAR α = 0,10**: ortalama {nz.far_a10.mean():.4f}, sd {nz.far_a10.std(ddof=1):.4f}, aralık [{nz.far_a10.min():.4f}, {nz.far_a10.max():.4f}].")
    amax = df[df.axis != "combo"].dauc_mean.abs().max()
    H(f"Okuma: tohum değişince macro-F1 yalnız ±{2*nz.f1_mean.std(ddof=1):.4f} (2 sd) oynar, yani ΔF1 farkları (a_tau17.5'in −0,0017'si hariç, o gürültü düzeyinde) tohum gürültüsünün üstünde okunabilir. AUROC ise ±{2*nz.auroc.std(ddof=1):.3f} (2 sd) ve FAR/tespit çok oynak (tespit sd {nz.det_a10.std(ddof=1):.2f}; FAR 0,078–0,137 aralığında yalnız tohumla). Seviyeler arasındaki en büyük |ΔAUROC| {amax:.3f}.")
    H("Kural açısından sonuç: AUROC ölçütü (iv) kat farklarının GA'sını kullandığından, tek tohumlu satırlar için bu GA'nın genişliği (±0,05–0,14) tohum gürültüsünden büyüktür; tek tohumlu bir AUROC farkı kendi başına bir etki sayılmamalı. Tohum 1–4 tekrarı (aşağıda) farkların tohumlar arasında tekrarlanıp tekrarlanmadığını gösterir.")
else:
    H("- Tohum gürültüsü koşusu bulunamadı (`02_noise_floor.csv`).")
H()
H("## Ek: her seviye için RF tohumu 1–4 tekrarı (brief dışı; kurala girmez)")
sf = REP / "02_level_seeds.csv"
if sf.exists():
    sd_ = pd.read_csv(sf)
    g = sd_.groupby("tag", sort=False).agg(n=("seed", "size"), f1=("f1_mean", "mean"), f1_sd=("f1_mean", "std"), auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
                                            det=("det_a10", "mean"), far=("far_a10", "mean"), far_sd=("far_a10", "std"))
    gb = g.loc["base"]
    H(f"Aynı seviyeler, RF tohumu 1, 2, 3, 4 (n_jobs = 3; her tohum için tam OOF + anomali hattı). Taban (aynı 4 tohum): macro-F1 {gb.f1:.4f} ± {gb.f1_sd:.4f}, AUROC {gb.auroc:.4f} ± {gb.auroc_sd:.4f}, tespit {gb.det:.3f}, FAR {gb.far:.4f}. Δ = seviye − taban (aynı tohumlar, tohum ortalaması). Ana tablodaki tohum (20261104) bu ortalamaya girmez.")
    H()
    H("| seviye | n tohum | macro-F1 (ort ± sd) | ΔF1 | AUROC (ort ± sd) | ΔAUROC | tespit | FAR (ort ± sd) |")
    H("|---|---|---|---|---|---|---|---|")
    for t, r in g.iterrows():
        H(f"| `{t}` | {int(r.n)} | {r.f1:.4f} ± {r.f1_sd:.4f} | {r.f1-gb.f1:+.4f} | {r.auroc:.4f} ± {r.auroc_sd:.4f} | {r.auroc-gb.auroc:+.4f} | {r.det:.3f} | {r.far:.4f} ± {r.far_sd:.4f} |")
    H()
    se_d = (2 * (gb.auroc_sd ** 2) / 4) ** 0.5
    big = g[(g.auroc - gb.auroc).abs() > 2 * se_d].index.tolist()
    H(f"Ölçek: tohum başına AUROC sd ≈ {gb.auroc_sd:.3f} (taban); 4 tohum ortalamasının iki ortalama farkı için SE ≈ {se_d:.3f}, yani 2 SE ≈ {2*se_d:.3f}. Tohum-ortalamalı |ΔAUROC| > 2 SE olan seviyeler: {', '.join('`'+x+'`' for x in big) if big else 'yok'} (eşiği en çok ≈1,2 katı aşıyorlar; 4 tohumla bu kesin bir ayrım değil). macro-F1 farkları ise tohum sd'sinin (≈ {gb.f1_sd:.4f}) çok üstünde ve ana tablodaki sırayı aynen koruyor. AUROC ile macro-F1 aynı yönde hareket etmiyor (örn. hız çıkarılmış: ΔF1 {g.loc['b_removed'].f1-gb.f1:+.4f} iken ΔAUROC {g.loc['b_removed'].auroc-gb.auroc:+.4f}); bu ayrışmanın nedeni bu koşularla açıklanmadı.")
    print(g.round(4).to_string())
else:
    H("- Tohum tekrarı bulunamadı (`02_level_seeds.csv`).")
H()

# ------------------------------------------------------------------ okuma
H("## Okuma (tahmini; nedensel iddia yok)")
def row(t): return df[df.tag == t].iloc[0]
a20, a25, a225 = row("a_tau20"), row("a_tau25"), row("a_tau22.5")
bs, br = row("b_sign"), row("b_removed")
cs = row("c_static")
H(f"- **(a) SNR eşiği.** macro-F1 τ yükseldikçe monoton düşüyor: ΔF1 τ = 17,5 / 20 / 22,5 / 25 dB için {sg(row('a_tau17.5').df1_mean)} / {sg(a20.df1_mean)} / {sg(a225.df1_mean)} / {sg(a25.df1_mean)}; 22,5 ve 25 dB'de GA 0'ı dışlıyor. Nokta sayısı %{a25.pts_pct_of_base:.0f}'e iniyor, {int(a25.empty_frames)} kare ({100*a25.empty_frames/16272:.1f}%) boş kalıyor; V2 {a25.v2_B_s:.0f} B/s ({a25.v2_B_s/base.v2_B_s:.2f}× taban). Literal kuralı 17,5 ve 20 dB geçiyor (en yalın geçen 20 dB); 22,5 ve 25 dB AUROC'un GA alt sınırından ({a225.dauc_lo:+.3f}, {a25.dauc_lo:+.3f}) takılıyor — GA çok geniş (±0,10–0,14), çünkü kat bazlı AUROC'lar 0,03–0,19 arasında oynuyor.")
H(f"- **(b) hız.** Hızın tamamen çıkarılması macro-F1'i {sg(br.df1_mean)} düşürüyor (GA [{br.df1_lo:+.4f}, {br.df1_hi:+.4f}]; ayrıca eşik 0,02'yi aşıyor): durum tahmini için hız bilgisi (mean_abs_v) gerekli. Yalnız işaret (mean(sign v)) ise macro-F1'i {sg(bs.df1_mean)} (GA [{bs.df1_lo:+.4f}, {bs.df1_hi:+.4f}]) *artırıyor*; AUROC farkı {sg(bs.dauc_mean)} (GA alt {bs.dauc_lo:+.3f}), literal kuralda yalnızca bu AUROC GA'sı takılıyor. Not: `mean(sign v)` tam |v| yerine konan farklı bir özellik; bu sonuç \"işaret yeter\" demekten çok \"|v| yerine işaret ortalaması da en az onun kadar bilgi taşıyor\" demektir. Bayt kazancı: {bs.v2_B_s:.0f} B/s ({bs.v2_B_s/base.v2_B_s:.2f}× taban) — 1 B/nokta varsayımıyla.")
H(f"- **(c) noktalar.** v = 0 noktalarını eklemek (ham) macro-F1'i {sg(rc.df1_mean)} değiştiriyor ve veriyi %{rc.pts_pct_of_base-100:.0f} artırıyor; v ≠ 0 noktaları atmak (yalnız v = 0) macro-F1'i {sg(cs.df1_mean)} düşürüyor (en kötü ΔF1) ve {int(cs.empty_frames)} kareyi boş bırakıyor. Yalnız-v = 0 kanalı bu radar yapılandırmasında bile bilgi taşıyor (macro-F1 {f4(cs.f1_mean)} ≫ şans), ama merged'den çok zayıf.")
H(f"- **(d) tepe gruplama.** Gruplama nokta sayısını %{rd.pts_pct_of_base:.0f}'e indiriyor (V2 {rd.v2_B_s:.0f} B/s); ΔF1 {sg(rd.df1_mean)} (GA [{rd.df1_lo:+.4f}, {rd.df1_hi:+.4f}]) — F1 düşüşü küçük ama 0'ı dışlıyor; literal kuralı geçiyor.")
_nc = df[df.axis != "combo"]
H(f"- **Bayt ölçeği.** Tüm seviyelerde V2 bayt/kare ham ADC'nin (57.344 B/kare) çok altında: {_nc.adc_ratio_v2.min():.0f}× (en pahalı: ham) ile {_nc.adc_ratio_v2.max():.0f}× (yalnız v = 0) arası; taban {base.v2_B_frame:.1f} B/kare = {base.v2_B_s:.0f} B/s ≈ {base.adc_ratio_v2:.0f}× daha az (merged noktalar, kare başına ortalama {base.n_points_mean:.2f} nokta; `05_gaps.md`'deki 196 B / 293× medyan 9 *ham* nokta içindi, bu yüzden farklı). En yalın birleşim ({df[df.tag=='combo_point'].iloc[0].v2_B_s:.0f} B/s) {df[df.tag=='combo_point'].iloc[0].adc_ratio_v2:.0f}×. Bütün seviyeler 10³ B/s mertebesinde; öykünülen en yalın birleşim tabanın %{100*df[df.tag=='combo_point'].iloc[0].v2_B_s/base.v2_B_s:.0f}'i kadar bayt gönderiyor, yani seviyeler arasındaki tasarruf, ham ADC'den nokta bulutuna geçişteki (≈ 365×) tasarrufun yanında küçük.")
H("- **Sınır.** Her seviye bir RF modeli, bir operatör, bir görev, S1 5 kat (pencereler aynı oturumlarda). Bunlar \"B2 durum tahmini + Faz A anomali hattı\" için sonuçlardır; çipte çalışan bir alıcı kanıtlanmadı.")
H()

# ------------------------------------------------------------------ uyarilar
H("## Öykünme uyarıları")
H("Koşudan önce yazılanlar:")
H("1. SNR kesimi yalnız menzil-CFAR'ı yaklaşık taklit eder (çipin CFAR'ı kendi gürültü tabanına göre eşikler; burada kayıttaki SNR'a sabit eşik konuyor; eşik altı noktalar zaten kayıtta yok, yani ≥ 15 dB'den aşağısı denenemez).")
H("2. Clutter removal (cfg `clutterRemoval -1 1`) geri alınamaz: ham ADC'de hareketsiz yansımalar çıkarılmadan önceki durum öykünülemez. Kayıttaki v = 0 noktaları (ham dosyada %20) clutter removal açıkken bile çıktı; bu yüzden \"yalnız v = 0\" temiz bir statik-clutter kanalı değildir.")
H("3. Daha küçük bir anten dizisi taklit edilmez (açı kuantizasyonu iyimser olurdu).")
H("Koşu sırasında eklenenler:")
H("4. **Tepe gruplama bir vekil.** TI'ın tepe gruplaması (cfg `cfarCfg … peakGrouping`; burada kapalı) menzil-Doppler algılama matrisinde çalışır. Burada brief'in tanımı kullanıldı: kayıtlı (menzil kutusu, açı kutusu) hücresi başına en yüksek SNR'li nokta. Aynı hücredeki noktalar farklı Doppler kutularından geliyor; yani bu işlem SDK'nın işlemi değil, onun noktasal bir yaklaşığı. Menzil yalnız 6 kutu olduğundan hücreler kaba.")
H("5. **Kare kümesi sabit** (16.272 kare); bir öykünmede hiç nokta kalmayan kare 0 bayt sayıldı ve B2 dolgusuyla özelliklendi. Gerçek bir akışta boş kare ayrı davranabilir.")
H("6. **Bayt varsayımları belgelenmiş sabitler, tel üstünde ölçülmedi** (`05_gaps.md`): TI SDK 3.6 TLV düzeni; işaret için 1 B/nokta ve hız alanını tamamen çıkarmak (nokta başına 4 B silmek), **stok firmware'in desteklemediği özel biçimlerdir**; stok akış hızı her zaman float olarak gönderir, kazanç yalnız özel bir firmware'de oluşur. z (4 B/nokta) V2'de sayıldı; z her zaman 0 olduğundan (`channelCfg 15 5 0`, 2B bulut) ek 4 B/nokta atılabilirdi, bu bir seviye olarak denenmedi (bayt sayısını etkilerdi, tahmin doğruluğunu etkilemezdi).")
H("7. **SNR eşiği ve gruplama kaydın ürettiği değerler üzerinde**; çipte eşik değişirse CFAR'ın gürültü tabanı da değişir, algılanan nokta kümesi bu öykünmenin sonucundan farklı olur.")
H("8. **Tek eksen değişimi** etkileşimleri görmez; birleşimler yalnız iki seçim kümesi için ikincil olarak koşuldu.")
H("9. **AUROC gürültülü** (kat sd 0,03–0,09; 50 civarı normal döngü): kural AUROC için GA koşulunu da içerdiğinden literal kural fiilen \"AUROC GA'sı yeterince dar mı\" sorusuna dönüşüyor. Tohum gürültüsü bölümüne bakın.")
H()
H("## Sapmalar / brief'te belirtilmemiş ayrıntılar")
H("`DESIGN_B.md` \"Sapmalar (tarihli)\" altına tek satır eklendi (2026-11-05 B.2). Özet: yalnız-işaret seviyesinde `mean_abs_v` yuvasına `mean(sign v)` konması; tek eksen değişimi ve \"en yalın\" = en az V2 bayt; GA tanımı (5 kat eşli t-GA, df = 4); bayt/kare sözleşmesi; hücre ve eşitlik kuralları; ek koşular (birleşimler, tohum gürültüsü 1–8, her seviye için tohum 1–4 tekrarı).")

(REP / "02_sensing_budget.md").write_text("\n".join(out) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ sekil
col = {"-": "#444444", "a": "#1f77b4", "b": "#d62728", "c": "#2ca02c", "d": "#9467bd", "combo": "#ff7f0e"}
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for k, (m, lo, hi, yl, thr) in enumerate([("df1_mean", "df1_lo", "df1_hi", "Δ macro-F1 (seviye − Faz A B2)", -0.02),
                                          ("dauc_mean", "dauc_lo", "dauc_hi", "Δ AUROC D3 (seviye − Faz A B2)", -0.03)]):
    a = ax[k]
    for _, r in df.iterrows():
        a.errorbar(r.v2_B_s, r[m], yerr=[[r[m] - r[lo]], [r[hi] - r[m]]], fmt="o", color=col[r.axis], capsize=3, ms=6)
        a.annotate(r.tag.replace("combo_", "c_"), (r.v2_B_s, r[m]), textcoords="offset points", xytext=(4, 5 + 9 * (int(_)%3 - 1)), fontsize=7)
    a.axhline(0, color="#999999", lw=0.8)
    a.axhline(thr, color="#d62728", lw=1, ls="--")
    a.set_xlabel("V2 bayt/s (10 Hz × bayt/kare)")
    a.set_ylabel(yl)
    a.set_title(f"{yl}\nçubuklar: kat eşli %95 t-GA; kesik çizgi: diz eşiği {thr}")
    a.grid(alpha=0.25)
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for l, c in
           [("taban", col["-"]), ("(a) SNR eşiği", col["a"]), ("(b) hız", col["b"]), ("(c) noktalar", col["c"]), ("(d) gruplama", col["d"]), ("birleşim", col["combo"])]]
ax[1].legend(handles=handles, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(REP / "02_sensing_budget.png", dpi=130)
print("rapor + sekil yazildi")

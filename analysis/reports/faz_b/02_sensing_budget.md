# B.2 — Algılama bütçesi: çip içi nokta bulutunun ne kadarı yetiyor? (öykünme)

Üretim: `faz_b/b_frames.py` (ham noktalardan kare istatistikleri + bayt hesabı), `faz_b/02a_gate.py` (kapı 1), `faz_b/02_sensing_budget.py` (seviyeler, eğitim, anomali hattı, diz seçimi), `faz_b/02c_noise_floor.py` (ek: tohum gürültüsü), `faz_b/02d_level_seeds.py` (ek: seviye başına tohum 1–4), `faz_b/02b_sensing_budget_report.py` (bu rapor). Ham sayılar: `02_sensing_budget.csv` (seviye başına tek satır, tüm sütunlar), `02_sensing_budget_folds.csv` (kat bazlı macro-F1 ve AUROC), `02_sensing_budget_selection.json`, `02_noise_floor.csv`, `02_level_seeds.csv`; OOF'lar `reports/faz_b/oof/sb_*__S1__f*.csv.gz` (her seviye için tam 16.248 pencere, 5 kat). **Statü: keşif amaçlı, plan koşudan önce yazıldı (`DESIGN_B.md` B.2); `main.tex`'e girmez.** Hiçbir şey çipte çalıştırılmadı; bütün seviyeler kaydedilmiş ham noktalardan yazılımla öykünüldü.

## Yöntem (DESIGN_B.md'ye göre)
- **Kare kümesi** her seviyede aynı 16.272 kare (`fa_common.frame_table()`; etiketler ve katlar bunlara bağlı). Bir öykünme bir karede hiç nokta bırakırsa o karede `n_points = 0`, `snr_sum = 0` olur, diğer istatistikler NaN'dır ve B2'nin Faz A dolgusuyla (cx, cy = −2; diğerleri 0) doldurulur.
- **İstatistikler** ham `mmwave_data.csv` noktalarından yeniden hesaplanır: cx, cy (SNR ağırlıklı), n_points, snr_sum, mean_abs_v, std_x, std_y. İşlem sırası: nokta kümesi (c) → SNR eşiği (a) → tepe gruplama (d) → hız (b).
- **B2**: 7 istatistik × (son, ortalama, eğim) = 21 özellik, L = 5, RandomForest 300 ağaç, `class_weight=balanced`, tohum 20261104, `n_jobs = 3`, S1 5 kat (Faz A ile aynı katlar). Her seviye için tam OOF yazıldı.
- **Anomali hattı**: OOF `y_pred` → ±3 kare debounce (MIN_BOX 2, MAX_GAP 1) → `pred` segmentasyonu → D1 + D2 → D3 = min(1, 2 min(p1, p2)), standart conformal, Faz A rolleri; AUROC = kat bazlı AUROC'un ortalaması, `tüm anomali` grubu; FAR α = 0,10'da 5 katın havuzlanmış normal test döngüleri (n ≈ 50, SE ≈ 0,04).
- **Eşli fark**: seviye − Faz A B2, kat bazında (5 kat), ortalama ve %95 t-GA (df = 4, t = 2,776, sd/√5). Faz A B2'nin kat bazlı AUROC'u `04_anomaly_table.csv`'de yok; aynı hat `fc.load_oof('b2','S1')` ile çalıştırılarak üretildi (ortalama 0,923697 = Faz A).
- **Eksenler tek tek değişir** (diğerleri taban: τ = 15 dB, hız tam, noktalar merged, gruplama kapalı): 4 + 2 + 2 + 1 = 9 seviye + taban = 10 seviye, 50 RF modeli. Birleşimler ayrıca, ikincil olarak koşuldu (aşağıda).
- **Ölçek**: kayıttaki SNR 0,1 dB biriminde; tüm oturumlarda `min(snr) = 150` (= 15,0 dB, cfg `cfarCfg … 15`). τ dB'e karşılık gelen eşik `snr ≥ 10·τ` (17,5 → 175, 20 → 200, 22,5 → 225, 25 → 250).
- **Hız seviyeleri**: *tam* = |v| (Faz A). *Çıkarılmış* = `mean_abs_v` sabit 0. *Yalnız işaret* = bir işaret-biti akışı |v| vermez; merged kümede (her nokta v ≠ 0) `mean(|sign v|)` ≡ 1 olurdu, yani seviye *çıkarılmış* ile aynı sabit özelliğe düşerdi. İşaret akışının destekleyebildiği tek hız istatistiği `mean(sign v)` olduğundan bu, `mean_abs_v` özellik yuvasına konuldu (özellik sayısı 21 kalır). Bkz. Sapmalar.
- **Tepe gruplama (açık)**: her (kare, menzil kutusu, açı kutusu) hücresinde yalnız en yüksek SNR'li nokta; eşitlikte dosyadaki ilk. Kayıttaki değerler kesiklidir: 6 menzil kutusu (adım ≈ 0,038 m, ayrım eşiği 1e-4 m), 57 açı kutusu (eşik 1e-3°). Aynı hücredeki noktalar Doppler kutusuyla ayrışır (hız adımı 0,1207 m/s).
- **Bayt**: (V2, birincil) yalnız nokta bulutu TLV'leri: TLV1 8 B başlık + n × (x, y, z float32 = 12 B + hız alanı) ve TLV7 8 B başlık + n × 4 B (SNR, gürültü uint16); hız alanı tam 4 B, işaret 1 B (bayt hizalı, özel biçim), çıkarılmış 0 B. Taban: 16 + 20·n (n = 9 → 196 B, `05_gaps.md` V2). (V3, ikincil) hattın okuduğu yük: x, y float32 (8 B) + hız + SNR uint16 (2 B), başlıksız. Nokta kalmayan karede 0 B. Bayt/s = 10 Hz × bayt/kare (`05_gaps.md` geleneği: kayıtlı kareler; boş radar karelerini saymaz). Ham ADC 57.344 B/kare (`05_gaps.md`). Tabloda ortalama bayt/kare (16.272 karede); medyan sütunu CSV'de (`v2_B_frame_median`).

## Kapılar
- **Kapı 1 (kare istatistikleri)**: `merged / tam` seviyesinin 7 istatistiği `fa_common.frame_table()` ile **16.272 karenin hepsinde birebir aynı**: her istatistik için max |fark| = 0,0 (bit düzeyinde; NaN yok). Not: s05'in ham dosyasında `merged_df.csv`'de olmayan 2 kare (9 nokta) var; bu kareler kare kümesinin dışında bırakıldı. Birebir `n_points` eşleşmesi bunların tek fark olduğunu gösterir.
- **Kapı 2 (taban satırı = Faz A B2)**: yeniden üretilmiş istatistiklerle eğitilen taban modeli kat bazında Faz A B2 ile aynı: macro-F1 0.8564 ± 0.0122 (Faz A 0,856448), max |kat farkı| = 0 (OOF birebir); AUROC 0.9237 ± 0.0352 (Faz A 0,923697); FAR 0.0784 (Faz A 0,0784); tespit 0.6335 (Faz A 0,6335).
- **Sonuç: kapılar geçti; B.2 koşuldu.**

## Tüm seviyeler (S1, 5 kat, tek eksen değişir)
`nokta/kare` = kare başına ortalama nokta (taban yüzdesi); `boş` = nokta kalmayan kare sayısı (16.272'den); bayt V2 / V3 = ortalama bayt/kare (ADC÷ = 57.344 / V2); ΔF1 ve ΔAUROC = Faz A B2'ye göre eşli kat farkı, ortalama [%95 GA]; AUROC = D3, `tüm anomali`, kat ortalaması ± sd; tespit / FAR α = 0,10 (FAR: SE).

| seviye | nokta/kare (taban %) | boş | V2 B/kare | V2 B/s | V3 B/kare | ADC÷ | macro-F1 ± sd | ΔF1 [GA] | AUROC ± sd | ΔAUROC [GA] | seq / tempo AUROC | tespit | FAR (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `base` — merged / tam (taban) | 7.07 (100%) | 0 | 157.3 | 1573 | 98.9 | 365× | 0.8564 ± 0.0122 | +0.0000 [+0.0000, +0.0000] | 0.9237 ± 0.0352 | +0.0000 [+0.0000, +0.0000] | 0.8970 / 0.9518 | 0.633 | 0.0784 (0.042) |
| `a_tau17.5` — SNR >= 17.5 dB | 6.81 (96%) | 241 | 152.0 | 1520 | 95.3 | 377× | 0.8548 ± 0.0154 | -0.0017 [-0.0074, +0.0041] | 0.9713 ± 0.0255 | +0.0476 [-0.0104, +0.1056] | 0.9653 / 0.9777 | 0.778 | 0.0385 (0.042) |
| `a_tau20` — SNR >= 20 dB | 6.50 (92%) | 670 | 145.3 | 1453 | 91.0 | 395× | 0.8530 ± 0.0103 | -0.0034 [-0.0076, +0.0008] | 0.9755 ± 0.0314 | +0.0518 [-0.0181, +0.1218] | 0.9800 / 0.9709 | 0.929 | 0.0392 (0.042) |
| `a_tau22.5` — SNR >= 22.5 dB | 6.15 (87%) | 1144 | 137.9 | 1379 | 86.1 | 416× | 0.8511 ± 0.0107 | -0.0053 [-0.0089, -0.0018] | 0.9127 ± 0.0932 | -0.0110 [-0.1433, +0.1213] | 0.8982 / 0.9278 | 0.347 | 0.0980 (0.042) |
| `a_tau25` — SNR >= 25 dB | 5.74 (81%) | 1676 | 129.1 | 1291 | 80.4 | 444× | 0.8431 ± 0.0105 | -0.0134 [-0.0195, -0.0072] | 0.9266 ± 0.0839 | +0.0029 [-0.1030, +0.1088] | 0.9235 / 0.9298 | 0.376 | 0.0784 (0.042) |
| `b_sign` — hiz: yalniz isaret | 7.07 (100%) | 0 | 136.1 | 1361 | 77.7 | 421× | 0.8670 ± 0.0128 | +0.0105 [+0.0056, +0.0155] | 0.8976 ± 0.0565 | -0.0261 [-0.0879, +0.0358] | 0.8608 / 0.9355 | 0.521 | 0.0800 (0.042) |
| `b_removed` — hiz: cikarilmis | 7.07 (100%) | 0 | 129.1 | 1291 | 70.7 | 444× | 0.8142 ± 0.0129 | -0.0423 [-0.0474, -0.0372] | 0.9727 ± 0.0318 | +0.0490 [-0.0148, +0.1129] | 0.9798 / 0.9661 | 0.474 | 0.0392 (0.042) |
| `c_raw` — nokta: ham (v=0 dahil) | 8.85 (125%) | 0 | 192.9 | 1929 | 123.8 | 297× | 0.8481 ± 0.0105 | -0.0084 [-0.0138, -0.0030] | 0.9324 ± 0.0211 | +0.0087 [-0.0293, +0.0466] | 0.9245 / 0.9407 | 0.860 | 0.0980 (0.042) |
| `c_static` — nokta: yalniz v=0 | 1.78 (25%) | 3278 | 48.4 | 484 | 24.9 | 1186× | 0.7760 ± 0.0071 | -0.0805 [-0.0919, -0.0690] | 0.8974 ± 0.0873 | -0.0263 [-0.1389, +0.0862] | 0.9149 / 0.8799 | 0.413 | 0.1020 (0.043) |
| `d_group` — tepe gruplama acik | 5.19 (73%) | 0 | 119.8 | 1198 | 72.7 | 479× | 0.8457 ± 0.0109 | -0.0107 [-0.0135, -0.0080] | 0.9703 ± 0.0429 | +0.0466 [-0.0090, +0.1021] | 0.9698 / 0.9707 | 0.671 | 0.0392 (0.042) |
| `combo_literal` — birlesim (literal secimler) | 4.76 (67%) | 670 | 110.5 | 1105 | 66.6 | 519× | 0.8397 ± 0.0117 | -0.0167 [-0.0206, -0.0128] | 0.9676 ± 0.0542 | +0.0439 [-0.0628, +0.1505] | 0.9818 / 0.9523 | 0.683 | 0.0000 (0.042) |
| `combo_point` — birlesim (nokta-tahmin secimleri) | 4.20 (59%) | 1676 | 85.8 | 858 | 46.2 | 669× | 0.8407 ± 0.0069 | -0.0158 [-0.0271, -0.0045] | 0.9336 ± 0.0757 | +0.0099 [-0.0783, +0.0980] | 0.9296 / 0.9376 | 0.461 | 0.0800 (0.042) |

Kat bazlı en küçük / en büyük fark (macro-F1; AUROC) CSV'de (`df1_min`, `df1_max`, `dauc_min`, `dauc_max`). Birleşim satırlarının tanımı: `combo_literal` = eksen başına literal kuralla seçilen seviyelerin birleşimi; `combo_point` = yalnız nokta tahmini kuralıyla seçilenlerin birleşimi (aşağıda). İkisi de brief dışı, ikincil.

## Önceden yazılmış diz kuralı
Kural (brief): en yalın seviye (burada: eksen içinde en az bayt/kare, V2), macro-F1 düşüşü ≤ 0,02 **ve** AUROC düşüşü ≤ 0,03 sağlayan seviyedir; kat GA'ları daha büyük bir düşüşü dışlamalı. Uygulama: dört ölçüt ayrı ayrı — (i) ΔF1 ≥ −0,02, (ii) ΔF1'in GA alt sınırı ≥ −0,02, (iii) ΔAUROC ≥ −0,03, (iv) ΔAUROC'un GA alt sınırı ≥ −0,03. **Literal geçiş = dördü birden.** Yalnız nokta tahmini (i + iii) ayrıca verilir; bu GA koşulunu gevşetir ve brief'in kuralı değildir.

| seviye | (i) ΔF1 ≥ −0,02 | (ii) GA alt ≥ −0,02 | (iii) ΔAUROC ≥ −0,03 | (iv) GA alt ≥ −0,03 | **literal** | yalnız nokta |
|---|---|---|---|---|---|---|
| `a_tau17.5` — SNR >= 17.5 dB | evet | evet (-0.0074) | evet | evet (-0.0104) | **evet** | evet |
| `a_tau20` — SNR >= 20 dB | evet | evet (-0.0076) | evet | evet (-0.0181) | **evet** | evet |
| `a_tau22.5` — SNR >= 22.5 dB | evet | evet (-0.0089) | evet | hayır (-0.1433) | **hayır** | evet |
| `a_tau25` — SNR >= 25 dB | evet | evet (-0.0195) | evet | hayır (-0.1030) | **hayır** | evet |
| `b_sign` — hiz: yalniz isaret | evet | evet (+0.0056) | evet | hayır (-0.0879) | **hayır** | evet |
| `b_removed` — hiz: cikarilmis | hayır | hayır (-0.0474) | evet | evet (-0.0148) | **hayır** | hayır |
| `c_raw` — nokta: ham (v=0 dahil) | evet | evet (-0.0138) | evet | evet (-0.0293) | **evet** | evet |
| `c_static` — nokta: yalniz v=0 | hayır | hayır (-0.0919) | evet | hayır (-0.1389) | **hayır** | hayır |
| `d_group` — tepe gruplama acik | evet | evet (-0.0135) | evet | evet (-0.0090) | **evet** | evet |
| `combo_literal` — birlesim (literal secimler) | evet | hayır (-0.0206) | evet | hayır (-0.0628) | **hayır** | evet |
| `combo_point` — birlesim (nokta-tahmin secimleri) | evet | hayır (-0.0271) | evet | hayır (-0.0783) | **hayır** | evet |

| eksen | bayt sırası (en yalından) | **literal seçim** | yalnız nokta tahmini seçimi | sonuç |
|---|---|---|---|---|
| (a) SNR eşiği | a_tau25 < a_tau22.5 < a_tau20 < a_tau17.5 < base | `a_tau20` | `a_tau25` | `a_tau20` yeterli |
| (b) hız | b_removed < b_sign < base | `base` | `b_sign` | taban dışında geçen yalın seviye yok: **tam bütçe gerekli** (bu eksende) |
| (c) noktalar | c_static < base < c_raw | `base` | `base` | **tam bütçe (= merged, Faz A) gerekli**; ham gerekmiyor (daha pahalı ve F1 daha düşük), yalnız v = 0 geçmiyor |
| (d) tepe gruplama | d_group < base | `d_group` | `d_group` | `d_group` yeterli |

### Birleşimler (ikincil; brief dışı)
- `combo_literal`: τ = 20 dB, hız = full, noktalar = merged, gruplama = açık → 4.76 nokta/kare (67%), V2 110.5 B/kare = 1105 B/s (ADC÷ 519×); macro-F1 0.8397 (ΔF1 -0.0167 [-0.0206, -0.0128]); AUROC 0.9676 (ΔAUROC +0.0439 [-0.0628, +0.1505]); tespit 0.683; FAR 0.0000. Literal geçiş: **hayır** (ΔF1 alt sınır -0.0206; ΔAUROC alt sınır -0.0628). Yalnız nokta tahmini: evet.
- `combo_point`: τ = 25 dB, hız = sign, noktalar = merged, gruplama = açık → 4.20 nokta/kare (59%), V2 85.8 B/kare = 858 B/s (ADC÷ 669×); macro-F1 0.8407 (ΔF1 -0.0158 [-0.0271, -0.0045]); AUROC 0.9336 (ΔAUROC +0.0099 [-0.0783, +0.0980]); tespit 0.461; FAR 0.0800. Literal geçiş: **hayır** (ΔF1 alt sınır -0.0271; ΔAUROC alt sınır -0.0783). Yalnız nokta tahmini: evet.
- Eksen başına geçen seviyelerin birleşimi eksenler arası etkileşimi test eder; hiçbir zaman eksen başına sonucun yerine geçmez.

## Önceden yazılmış beklentiler (literal)
- **(c) ham ile merged arasında |Δ macro-F1| ≤ 0,01:** ΔF1 = -0.0084 (GA [-0.0138, -0.0030]; ham 0.8481 vs merged 0.8564) → |Δ| = 0.0084. **Beklenti TUTTU** (nokta tahmini). Not: kat GA'sı 0'ı dışlıyor (ham v = 0 noktaları eklemek F1'i hafifçe düşürüyor); beklenti nokta tahminine göre yazılmıştı.
- **(d) gruplama açıkken |Δ macro-F1| ≤ 0,01:** ΔF1 = -0.0107 (GA [-0.0135, -0.0080]; gruplamalı 0.8457) → |Δ| = 0.0107. **Beklenti TUTMADI** (eşiği 0.0007 aştı; yani sınırda ama literal olarak tutmadı).

## Ek: tohum gürültüsü (brief dışı; AUROC farklarını okumak için)
Taban seviyesinde (merged / tam) yalnız RF tohumu değişti (tohum 1–8, 40 model; diğer her şey sabit). **macro-F1** ortalama 0.8573, sd 0.0010, aralık [0.8558, 0.8592] (Faz A tohumu 20261104: 0.8564). **AUROC (D3, tüm anomali)** ortalama 0.9255, sd 0.0155, aralık [0.9031, 0.9487] (Faz A tohumu: 0.9237). tespit α = 0,10: ortalama 0.701, sd 0.104; **FAR α = 0,10**: ortalama 0.0956, sd 0.0221, aralık [0.0784, 0.1373].
Okuma: tohum değişince macro-F1 yalnız ±0.0020 (2 sd) oynar, yani ΔF1 farkları (a_tau17.5'in −0,0017'si hariç, o gürültü düzeyinde) tohum gürültüsünün üstünde okunabilir. AUROC ise ±0.031 (2 sd) ve FAR/tespit çok oynak (tespit sd 0.10; FAR 0,078–0,137 aralığında yalnız tohumla). Seviyeler arasındaki en büyük |ΔAUROC| 0.052.
Kural açısından sonuç: AUROC ölçütü (iv) kat farklarının GA'sını kullandığından, tek tohumlu satırlar için bu GA'nın genişliği (±0,05–0,14) tohum gürültüsünden büyüktür; tek tohumlu bir AUROC farkı kendi başına bir etki sayılmamalı. Tohum 1–4 tekrarı (aşağıda) farkların tohumlar arasında tekrarlanıp tekrarlanmadığını gösterir.

## Ek: her seviye için RF tohumu 1–4 tekrarı (brief dışı; kurala girmez)
Aynı seviyeler, RF tohumu 1, 2, 3, 4 (n_jobs = 3; her tohum için tam OOF + anomali hattı). Taban (aynı 4 tohum): macro-F1 0.8567 ± 0.0007, AUROC 0.9285 ± 0.0221, tespit 0.769, FAR 0.1029. Δ = seviye − taban (aynı tohumlar, tohum ortalaması). Ana tablodaki tohum (20261104) bu ortalamaya girmez.

| seviye | n tohum | macro-F1 (ort ± sd) | ΔF1 | AUROC (ort ± sd) | ΔAUROC | tespit | FAR (ort ± sd) |
|---|---|---|---|---|---|---|---|
| `base` | 4 | 0.8567 ± 0.0007 | +0.0000 | 0.9285 ± 0.0221 | +0.0000 | 0.769 | 0.1029 ± 0.0294 |
| `a_tau17.5` | 4 | 0.8544 ± 0.0002 | -0.0023 | 0.9621 ± 0.0117 | +0.0335 | 0.928 | 0.0637 ± 0.0188 |
| `a_tau20` | 4 | 0.8524 ± 0.0009 | -0.0043 | 0.9467 ± 0.0026 | +0.0182 | 0.870 | 0.0880 ± 0.0202 |
| `a_tau22.5` | 4 | 0.8489 ± 0.0011 | -0.0078 | 0.9180 ± 0.0039 | -0.0105 | 0.361 | 0.0784 ± 0.0000 |
| `a_tau25` | 4 | 0.8448 ± 0.0012 | -0.0119 | 0.9244 ± 0.0060 | -0.0041 | 0.404 | 0.0825 ± 0.0091 |
| `b_sign` | 4 | 0.8662 ± 0.0016 | +0.0095 | 0.8936 ± 0.0152 | -0.0349 | 0.575 | 0.0934 ± 0.0241 |
| `b_removed` | 4 | 0.8143 ± 0.0013 | -0.0424 | 0.9668 ± 0.0078 | +0.0383 | 0.524 | 0.0344 ± 0.0096 |
| `c_raw` | 4 | 0.8488 ± 0.0011 | -0.0079 | 0.9431 ± 0.0139 | +0.0145 | 0.810 | 0.0735 ± 0.0294 |
| `c_static` | 4 | 0.7759 ± 0.0009 | -0.0808 | 0.8962 ± 0.0094 | -0.0323 | 0.421 | 0.0765 ± 0.0195 |
| `d_group` | 4 | 0.8442 ± 0.0008 | -0.0125 | 0.9645 ± 0.0186 | +0.0359 | 0.733 | 0.0392 ± 0.0392 |
| `combo_literal` | 4 | 0.8412 ± 0.0008 | -0.0155 | 0.9530 ± 0.0203 | +0.0244 | 0.745 | 0.0343 ± 0.0435 |
| `combo_point` | 4 | 0.8407 ± 0.0007 | -0.0160 | 0.9094 ± 0.0091 | -0.0191 | 0.597 | 0.1189 ± 0.0288 |

Ölçek: tohum başına AUROC sd ≈ 0.022 (taban); 4 tohum ortalamasının iki ortalama farkı için SE ≈ 0.016, yani 2 SE ≈ 0.031. Tohum-ortalamalı |ΔAUROC| > 2 SE olan seviyeler: `a_tau17.5`, `b_sign`, `b_removed`, `c_static`, `d_group` (eşiği en çok ≈1,2 katı aşıyorlar; 4 tohumla bu kesin bir ayrım değil). macro-F1 farkları ise tohum sd'sinin (≈ 0.0007) çok üstünde ve ana tablodaki sırayı aynen koruyor. AUROC ile macro-F1 aynı yönde hareket etmiyor (örn. hız çıkarılmış: ΔF1 -0.0424 iken ΔAUROC +0.0383); bu ayrışmanın nedeni bu koşularla açıklanmadı.

## Okuma (tahmini; nedensel iddia yok)
- **(a) SNR eşiği.** macro-F1 τ yükseldikçe monoton düşüyor: ΔF1 τ = 17,5 / 20 / 22,5 / 25 dB için -0.0017 / -0.0034 / -0.0053 / -0.0134; 22,5 ve 25 dB'de GA 0'ı dışlıyor. Nokta sayısı %81'e iniyor, 1676 kare (10.3%) boş kalıyor; V2 1291 B/s (0.82× taban). Literal kuralı 17,5 ve 20 dB geçiyor (en yalın geçen 20 dB); 22,5 ve 25 dB AUROC'un GA alt sınırından (-0.143, -0.103) takılıyor — GA çok geniş (±0,10–0,14), çünkü kat bazlı AUROC'lar 0,03–0,19 arasında oynuyor.
- **(b) hız.** Hızın tamamen çıkarılması macro-F1'i -0.0423 düşürüyor (GA [-0.0474, -0.0372]; ayrıca eşik 0,02'yi aşıyor): durum tahmini için hız bilgisi (mean_abs_v) gerekli. Yalnız işaret (mean(sign v)) ise macro-F1'i +0.0105 (GA [+0.0056, +0.0155]) *artırıyor*; AUROC farkı -0.0261 (GA alt -0.088), literal kuralda yalnızca bu AUROC GA'sı takılıyor. Not: `mean(sign v)` tam |v| yerine konan farklı bir özellik; bu sonuç "işaret yeter" demekten çok "|v| yerine işaret ortalaması da en az onun kadar bilgi taşıyor" demektir. Bayt kazancı: 1361 B/s (0.87× taban) — 1 B/nokta varsayımıyla.
- **(c) noktalar.** v = 0 noktalarını eklemek (ham) macro-F1'i -0.0084 değiştiriyor ve veriyi %25 artırıyor; v ≠ 0 noktaları atmak (yalnız v = 0) macro-F1'i -0.0805 düşürüyor (en kötü ΔF1) ve 3278 kareyi boş bırakıyor. Yalnız-v = 0 kanalı bu radar yapılandırmasında bile bilgi taşıyor (macro-F1 0.7760 ≫ şans), ama merged'den çok zayıf.
- **(d) tepe gruplama.** Gruplama nokta sayısını %73'e indiriyor (V2 1198 B/s); ΔF1 -0.0107 (GA [-0.0135, -0.0080]) — F1 düşüşü küçük ama 0'ı dışlıyor; literal kuralı geçiyor.
- **Bayt ölçeği.** Tüm seviyelerde V2 bayt/kare ham ADC'nin (57.344 B/kare) çok altında: 297× (en pahalı: ham) ile 1186× (yalnız v = 0) arası; taban 157.3 B/kare = 1573 B/s ≈ 365× daha az (merged noktalar, kare başına ortalama 7.07 nokta; `05_gaps.md`'deki 196 B / 293× medyan 9 *ham* nokta içindi, bu yüzden farklı). En yalın birleşim (858 B/s) 669×. Bütün seviyeler 10³ B/s mertebesinde; öykünülen en yalın birleşim tabanın %55'i kadar bayt gönderiyor, yani seviyeler arasındaki tasarruf, ham ADC'den nokta bulutuna geçişteki (≈ 365×) tasarrufun yanında küçük.
- **Sınır.** Her seviye bir RF modeli, bir operatör, bir görev, S1 5 kat (pencereler aynı oturumlarda). Bunlar "B2 durum tahmini + Faz A anomali hattı" için sonuçlardır; çipte çalışan bir alıcı kanıtlanmadı.

## Öykünme uyarıları
Koşudan önce yazılanlar:
1. SNR kesimi yalnız menzil-CFAR'ı yaklaşık taklit eder (çipin CFAR'ı kendi gürültü tabanına göre eşikler; burada kayıttaki SNR'a sabit eşik konuyor; eşik altı noktalar zaten kayıtta yok, yani ≥ 15 dB'den aşağısı denenemez).
2. Clutter removal (cfg `clutterRemoval -1 1`) geri alınamaz: ham ADC'de hareketsiz yansımalar çıkarılmadan önceki durum öykünülemez. Kayıttaki v = 0 noktaları (ham dosyada %20) clutter removal açıkken bile çıktı; bu yüzden "yalnız v = 0" temiz bir statik-clutter kanalı değildir.
3. Daha küçük bir anten dizisi taklit edilmez (açı kuantizasyonu iyimser olurdu).
Koşu sırasında eklenenler:
4. **Tepe gruplama bir vekil.** TI'ın tepe gruplaması (cfg `cfarCfg … peakGrouping`; burada kapalı) menzil-Doppler algılama matrisinde çalışır. Burada brief'in tanımı kullanıldı: kayıtlı (menzil kutusu, açı kutusu) hücresi başına en yüksek SNR'li nokta. Aynı hücredeki noktalar farklı Doppler kutularından geliyor; yani bu işlem SDK'nın işlemi değil, onun noktasal bir yaklaşığı. Menzil yalnız 6 kutu olduğundan hücreler kaba.
5. **Kare kümesi sabit** (16.272 kare); bir öykünmede hiç nokta kalmayan kare 0 bayt sayıldı ve B2 dolgusuyla özelliklendi. Gerçek bir akışta boş kare ayrı davranabilir.
6. **Bayt varsayımları belgelenmiş sabitler, tel üstünde ölçülmedi** (`05_gaps.md`): TI SDK 3.6 TLV düzeni; işaret için 1 B/nokta ve hız alanını tamamen çıkarmak (nokta başına 4 B silmek), **stok firmware'in desteklemediği özel biçimlerdir**; stok akış hızı her zaman float olarak gönderir, kazanç yalnız özel bir firmware'de oluşur. z (4 B/nokta) V2'de sayıldı; z her zaman 0 olduğundan (`channelCfg 15 5 0`, 2B bulut) ek 4 B/nokta atılabilirdi, bu bir seviye olarak denenmedi (bayt sayısını etkilerdi, tahmin doğruluğunu etkilemezdi).
7. **SNR eşiği ve gruplama kaydın ürettiği değerler üzerinde**; çipte eşik değişirse CFAR'ın gürültü tabanı da değişir, algılanan nokta kümesi bu öykünmenin sonucundan farklı olur.
8. **Tek eksen değişimi** etkileşimleri görmez; birleşimler yalnız iki seçim kümesi için ikincil olarak koşuldu.
9. **AUROC gürültülü** (kat sd 0,03–0,09; 50 civarı normal döngü): kural AUROC için GA koşulunu da içerdiğinden literal kural fiilen "AUROC GA'sı yeterince dar mı" sorusuna dönüşüyor. Tohum gürültüsü bölümüne bakın.

## Sapmalar / brief'te belirtilmemiş ayrıntılar
`DESIGN_B.md` "Sapmalar (tarihli)" altına tek satır eklendi (2026-11-05 B.2). Özet: yalnız-işaret seviyesinde `mean_abs_v` yuvasına `mean(sign v)` konması; tek eksen değişimi ve "en yalın" = en az V2 bayt; GA tanımı (5 kat eşli t-GA, df = 4); bayt/kare sözleşmesi; hücre ve eşitlik kuralları; ek koşular (birleşimler, tohum gürültüsü 1–8, her seviye için tohum 1–4 tekrarı).

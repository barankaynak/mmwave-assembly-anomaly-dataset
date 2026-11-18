# B.3 — Kare hızı: her k'inci kare tutulursa ne kalıyor? (öykünme)

Üretim: `faz_b/03_frame_rate.py` (hesap, kapılar, tüm seviyeler), `faz_b/03b_frame_rate_report.py` (bu rapor). Ham sayılar: `03_frame_rate.csv` (seviye başına tek satır), `03_frame_rate_folds.csv`, `03_frame_rate_detection.csv` (dokunuş tespiti, oturum × süre kutusu), `03_frame_rate_fits.csv` (lojistik d50/d90); OOF'lar `reports/faz_b/oof/fr_*__S1__f*.csv.gz`. **Statü: keşif amaçlı, plan koşudan önce yazıldı (`DESIGN_B.md` B.3); `main.tex`'e girmez.** Hiçbir şey radarda çalıştırılmadı: seyreltme kaydedilmiş 10 Hz karelerden yazılımla öykünüldü. **B.3'te önceden yazılmış sayısal bir beklenti yoktur** (yalnız diz kuralı); bu yüzden aşağıda 'beklenti tuttu/tutmadı' satırı yoktur.

## Yöntem (DESIGN_B.md'ye göre)
- **Seyreltme**: her oturumda `frame_idx_in_session % k == 0` olan kareler tutulur (ana koşu faz kaydırması 0; fazlar 1…k−1 ikincil). k ∈ {2, 3, 5} → 5 / 3,33 / 2 Hz; k = 1 (10 Hz) kapı satırı.
- **Pencere** L_k = ceil(5/k): k = 2 → 3, k = 3 → 2, k = 5 → 1 kare. Pencerenin etiketi son tutulan karenin 10 Hz `y_true`'su (yeniden yumuşatma yok). L = 1'de eğim özellikleri 0; özellikler yine 7 istatistik × (son, ortalama, eğim) = 21 sütun (L = 1'de son = ortalama). Model B2: RF 300 ağaç, `class_weight=balanced`, tohum 20261104, `n_jobs = 3`.
- **Katlar**: pencerenin son karesinin döngüsünün (gid) Faz A S1 katı; yani aynı döngüler aynı katta (doğrulandı: her gid tek katta). Eşli fark: seviye − Faz A B2, kat bazında (5 kat), ortalama ve %95 t-GA (df = 4).
- **Debounce süre olarak sabit** (MIN_BOX 0,2 s, MAX_GAP 0,1 s), kareye yuvarlanıp en az 1 (MIN_BOX) ve 0 (MAX_GAP): k = 1: MIN_BOX 2, MAX_GAP 1 kare; k = 2: MIN_BOX 1, MAX_GAP 0 kare; k = 3: MIN_BOX 1, MAX_GAP 0 kare; k = 5: MIN_BOX 1, MAX_GAP 0 kare. **k = 2'de MAX_GAP = 0,1 s / 0,2 s = 0,5 kare** ve yuvarlama kuralı belirsiz: ana koşu Python `round` (→ 0) kullandı; yarı-yukarı (→ 1) ikincil olarak koşuldu (aşağıda; sonuç çok farklı).
- **Anomali hattı**: tutulan akışın `y_pred`'i (ilk L_k − 1 tutulan kare penceresiz → none) → yukarıdaki debounce → 10 Hz'e sıfırıncı-derece tutma (bir tutulan karenin etiketi sonraki tutulan kareye kadar tutulur; ziyaret süreleri yani 0,1·k s'ye nicemlenir) → `ac.build_cycles('pred')` → D1 + D2 → D3, Faz A rolleri ve standart conformal, α = 0,10. `build_cycles`'ın içindeki sabit (2, 1) debounce'unun tutulmuş akışta etkisiz olduğu her koşuda iddia (assert) ile doğrulandı.
- **Dokunuş tespiti (D-ham, A3.2 yöntemi)**: ToF dokunuşu (mesafe < 140 mm kesintisiz koşu, süre = span + medyan dt); dokunuşun kare kümesi tam hızda A3.2'deki `(t−100 ms, t]` atamasıyla belirlenir, **sonra tutulan karelerle kesiştirilir** (radar yalnız tutulan karelerde ölçer). D-ham = kesişimde son karesi bulunan en az bir penceresi o kutuyu ham `y_pred` ile tahmin ediyor. **Ana payda**: tam hızda değerlendirilebilen dokunuşlar (n_frames > 0; aynı 1111 dokunuş, hepsi); tutulan karesi hiç olmayan dokunuş **kaçırma sayılır** (örnekleme kaybı, ayrıca raporlanır). Koşullu varyant (yalnız ≥ 1 tutulan karesi olanlar) ikincil.
- **Bayt/s**: V2 taban 157,3 B/kare (B.2 ile aynı sözleşme) × 10/k Hz. Not: seyreltme bayt/kare'yi değiştirmez, yalnız kare sayısını; B.2'deki özel-biçim varsayımları geçerli değildir (stok akış).

## Kapılar
- **Kapı 1 (k = 1 ≡ Faz A B2)**: kat bazında macro-F1 ve kat bazında D3 AUROC Faz A B2 ile **birebir** (max |fark| = 0,0; ortalamalar 0.8564 ve 0.9237); FAR 0.0784, tespit 0.6335 (Faz A ile aynı).
- **Kapı 2 (k = 1 D-ham ≡ A3.2)**: 6 oturum × 5 süre kutusunda B2 / D-ham `k_detect`, `n_eval` ve oturum başına lojistik `b0, b1, d50, d90` `reports/faz_a3/02_envelope_*.csv` ile **birebir** (fark 0 / 0).
- **Sonuç: kapılar geçti; B.3 koşuldu.**

## Ana sonuç (tohum 20261104, faz 0; S1 5 kat)

`pencere` = test+eğitim pencere sayısı; `span` = pencerenin süresi (0,1·(k(L−1)+1) s); ΔF1/ΔAUROC = Faz A B2'ye göre eşli kat farkı, ortalama [%95 GA]; `ΔF1 (kısıtlı ref.)` = fark, referans = Faz A B2'nin yalnız aynı tutulan karelerdeki macro-F1'i (modelin düşük hızda eğitilmesi ile test karesinin azalmasını ayırmak için); AUROC = D3 `tüm anomali`, kat ort. ± sd; tespit / FAR α = 0,10 (FAR: SE); `kaçırılan dok.` = tutulan karesi olmayan dokunuş / 1111.

| k | Hz | L | span (s) | pencere | V2 B/s | macro-F1 ± sd | ΔF1 [GA] | ΔF1 (kısıtlı ref.) | AUROC ± sd | ΔAUROC [GA] | seq / tempo AUROC | tespit | FAR (SE) | döngü | kaçırılan dok. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 5 | 0.5 | 16248 | 1573 | 0.8564 ± 0.0122 | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0.9237 ± 0.0352 | +0.0000 [+0.0000, +0.0000] | 0.8970 / 0.9518 | 0.633 | 0.0784 (0.042) | 254 | 0 |
| 2 | 5 | 3 | 0.5 | 8125 | 787 | 0.8265 ± 0.0117 | -0.0300 [-0.0360, -0.0239] | -0.0325 | 0.8471 ± 0.1551 | -0.0766 [-0.2521, +0.0989] | 0.7840 / 0.9133 | 0.192 | 0.0980 (0.042) | 256 | 6 |
| 3 | 3,33 | 2 | 0.4 | 5420 | 524 | 0.8015 ± 0.0120 | -0.0549 [-0.0605, -0.0494] | -0.0461 | 0.8467 ± 0.0953 | -0.0770 [-0.1798, +0.0258] | 0.8878 / 0.8039 | 0.245 | 0.0784 (0.042) | 251 | 10 |
| 5 | 2 | 1 | 0.1 | 3256 | 315 | 0.7076 ± 0.0278 | -0.1489 [-0.1765, -0.1213] | -0.1471 | 0.7968 ± 0.0826 | -0.1269 [-0.2314, -0.0225] | 0.8562 / 0.7361 | 0.260 | 0.0377 (0.041) | 257 | 51 |

Sınıf bazında F1 (havuzlanmış OOF; 0 = none): k = 1: 0.902 / 0.909 / 0.851 / 0.803 / 0.845 / 0.838; k = 2: 0.875 / 0.876 / 0.819 / 0.771 / 0.801 / 0.827; k = 3: 0.855 / 0.875 / 0.774 / 0.729 / 0.787 / 0.802; k = 5: 0.744 / 0.781 / 0.726 / 0.683 / 0.632 / 0.689

## Önceden yazılmış diz kuralı (B.2 ile aynı)
Kural: en yalın seviye (burada: en düşük kare hızı = en büyük k), macro-F1 düşüşü ≤ 0,02 **ve** AUROC düşüşü ≤ 0,03 sağlayan seviyedir; kat GA'ları daha büyük bir düşüşü dışlamalı. Dört ölçüt ayrı ayrı: (i) ΔF1 ≥ −0,02, (ii) ΔF1 GA alt ≥ −0,02, (iii) ΔAUROC ≥ −0,03, (iv) ΔAUROC GA alt ≥ −0,03. **Literal geçiş = dördü birden**; yalnız nokta tahmini (i + iii) ayrıca.

| k (Hz) | (i) ΔF1 ≥ −0,02 | (ii) GA alt ≥ −0,02 | (iii) ΔAUROC ≥ −0,03 | (iv) GA alt ≥ −0,03 | **literal** | yalnız nokta |
|---|---|---|---|---|---|---|
| 2 (5) | hayır (-0.0300) | hayır (-0.0360) | hayır (-0.0766) | hayır (-0.2521) | hayır | hayır |
| 3 (3,33) | hayır (-0.0549) | hayır (-0.0605) | hayır (-0.0770) | hayır (-0.1798) | hayır | hayır |
| 5 (2) | hayır (-0.1489) | hayır (-0.1765) | hayır (-0.1269) | hayır (-0.2314) | hayır | hayır |

**Sonuç (ana koşu):** literal kuralı geçen k: yok; yalnız-nokta kuralını geçen k: yok. Hiçbir seyreltme düzeyi kuralı geçmiyor: **tam kare hızı (10 Hz) gerekli** (bu model ve bu hat için). Kural k = 2'de bile macro-F1'den takılıyor (ΔF1 -0.0300, GA [-0.0360, -0.0239]; 5 katın hepsinde kat farkı -0.0372 … -0.0258, yani hepsi −0,02'nin altında); AUROC ayrıca ana debounce kuralında (ΔAUROC -0.0766) takılıyor, ancak o ölçüt debounce yuvarlamasına çok duyarlı (aşağıda).

## Ek (brief dışı): debounce yuvarlamasına duyarlılık
Macro-F1 debounce'tan etkilenmez (debounce yalnız döngü hattında); AUROC/FAR/tespit etkilenir. Aynı OOF, farklı debounce (tohum 20261104, faz 0): **ana** = süre sabit, Python `round`; **MAX_GAP = 1 kare** (k = 2'de yarı-yukarı yuvarlama ile aynı); **kare sayısı sabit** (MIN_BOX 2, MAX_GAP 1, yani 10 Hz'deki sayılar; saniye karşılığı 0,1·k ile büyür). Bu üç düzenleme arasından en iyiyi seçmek seçim yanlılığı olurdu: **tablo bir tarama olarak okunmalı, bir sonuç olarak değil**.

| k | debounce | MIN_BOX/MAX_GAP (kare) | AUROC ± sd | ΔAUROC [GA] | seq / tempo | tespit | FAR | döngü | literal |
|---|---|---|---|---|---|---|---|---|---|
| 2 | ana | 1/0 | 0.8471 ± 0.1551 | -0.0766 [-0.2521, +0.0989] | 0.7840 / 0.9133 | 0.192 | 0.0980 | 256 | hayır |
| 2 | MAX_GAP = 1 kare | 1/1 | 0.9443 ± 0.0451 | +0.0206 [-0.0600, +0.1012] | 0.9297 / 0.9597 | 0.315 | 0.0196 | 254 | hayır |
| 2 | kare sayısı sabit (2/1) | 2/1 | 0.9829 ± 0.0141 | +0.0592 [+0.0006, +0.1178] | 0.9923 / 0.9731 | 0.563 | 0.0000 | 248 | hayır |
| 3 | ana | 1/0 | 0.8467 ± 0.0953 | -0.0770 [-0.1798, +0.0258] | 0.8878 / 0.8039 | 0.245 | 0.0784 | 251 | hayır |
| 3 | MAX_GAP = 1 kare | 1/1 | 0.8484 ± 0.0791 | -0.0753 [-0.1608, +0.0101] | 0.8948 / 0.8000 | 0.307 | 0.0800 | 250 | hayır |
| 3 | kare sayısı sabit (2/1) | 2/1 | 0.8537 ± 0.0879 | -0.0700 [-0.1874, +0.0473] | 0.9091 / 0.7952 | 0.276 | 0.0800 | 239 | hayır |
| 5 | ana | 1/0 | 0.7968 ± 0.0826 | -0.1269 [-0.2314, -0.0225] | 0.8562 / 0.7361 | 0.260 | 0.0377 | 257 | hayır |
| 5 | MAX_GAP = 1 kare | 1/1 | 0.8266 ± 0.0399 | -0.0971 [-0.1791, -0.0151] | 0.9005 / 0.7489 | 0.313 | 0.0784 | 250 | hayır |
| 5 | kare sayısı sabit (2/1) | 2/1 | 0.7510 ± 0.1125 | -0.1727 [-0.3219, -0.0234] | 0.7272 / 0.7721 | 0.000 | 0.0000 | 181 | hayır |

## Ek (brief dışı): faz kaydırması ve RF tohumu
Faz: aynı k için tutulan karelerin başlangıcı 0…k−1 (tohum 20261104; k = 2: 2, k = 3: 3, k = 5: 5 faz). Tohum: RF tohumu 1–4, faz 0. k = 1 için 4 tohum aynı tablodadır. Δ = faz/tohum ortalaması − k = 1 tohum ortalaması (macro-F1 0,8567; AUROC 0,9285).

| k | faz sayısı | macro-F1 faz ort. (min–maks) | AUROC faz ort. (min–maks) | macro-F1 tohum ort. ± sd | AUROC tohum ort. ± sd | ΔF1 (tohum ort.) | ΔAUROC (tohum ort.) | tespit tohum ort. | FAR tohum ort. | literal geçen / koşu (faz + tohum) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | - | - | - | 0.8567 ± 0.0007 | 0.9285 ± 0.0221 | +0.0000 | +0.0000 | 0.769 | 0.1029 | 2 / 4 |
| 2 | 2 | 0.8238 (0.8212–0.8265) | 0.8447 (0.8424–0.8471) | 0.8265 ± 0.0008 | 0.8536 ± 0.0066 | -0.0302 | -0.0749 | 0.183 | 0.0731 | 0 / 6 |
| 3 | 3 | 0.8032 (0.8015–0.8046) | 0.8365 (0.7999–0.8629) | 0.8022 ± 0.0021 | 0.8561 ± 0.0126 | -0.0545 | -0.0725 | 0.291 | 0.0649 | 0 / 7 |
| 5 | 5 | 0.7083 (0.7035–0.7147) | 0.7589 (0.7395–0.7968) | 0.7115 ± 0.0016 | 0.8132 ± 0.0244 | -0.1452 | -0.1154 | 0.198 | 0.0330 | 0 / 9 |

## D-ham dokunuş tespiti ~ süre (B2; Wilson %95; oturum başına, havuzlanmadı)
`k/n = oran [alt, üst]`; payda = tam hızda değerlendirilebilen dokunuş; **tutulan karesi olmayan dokunuş kaçırma sayılır**, parantezde `(−x)` = o hücredeki örnekleme kaybı sayısı. k = 1 sütunları A3.2 ile aynıdır (kapı 2).

### k = 1 (10 Hz)

| oturum | <0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1.0 | >1.0 |
|---|---|---|---|---|---|
| s01 | 5/8 = 0.625 [0.306, 0.863] | 14/22 = 0.636 [0.430, 0.803] | 88/106 = 0.830 [0.747, 0.890] | 160/166 = 0.964 [0.923, 0.983] | 26/26 = 1.000 [0.871, 1.000] |
| s02 | 0/1 = 0.000 [0.000, 0.793] | - | 36/36 = 1.000 [0.904, 1.000] | 105/105 = 1.000 [0.965, 1.000] | 13/13 = 1.000 [0.772, 1.000] |
| s03 | - | - | 51/51 = 1.000 [0.930, 1.000] | 111/111 = 1.000 [0.967, 1.000] | 9/9 = 1.000 [0.701, 1.000] |
| s04 | - | 1/1 = 1.000 [0.207, 1.000] | 38/38 = 1.000 [0.908, 1.000] | 102/102 = 1.000 [0.964, 1.000] | 7/7 = 1.000 [0.646, 1.000] |
| s05 | 4/9 = 0.444 [0.189, 0.733] | 24/28 = 0.857 [0.685, 0.943] | 63/63 = 1.000 [0.943, 1.000] | 43/43 = 1.000 [0.918, 1.000] | 10/10 = 1.000 [0.722, 1.000] |
| s06 | - | - | 2/2 = 1.000 [0.342, 1.000] | 110/110 = 1.000 [0.966, 1.000] | 44/44 = 1.000 [0.920, 1.000] |

### k = 2 (5 Hz)

| oturum | <0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1.0 | >1.0 |
|---|---|---|---|---|---|
| s01 | 0/8 = 0.000 [0.000, 0.324] (−5) | 13/22 = 0.591 [0.387, 0.767] | 84/106 = 0.792 [0.706, 0.859] | 157/166 = 0.946 [0.900, 0.971] | 26/26 = 1.000 [0.871, 1.000] |
| s02 | 1/1 = 1.000 [0.207, 1.000] | - | 36/36 = 1.000 [0.904, 1.000] | 105/105 = 1.000 [0.965, 1.000] | 13/13 = 1.000 [0.772, 1.000] |
| s03 | - | - | 51/51 = 1.000 [0.930, 1.000] | 111/111 = 1.000 [0.967, 1.000] | 9/9 = 1.000 [0.701, 1.000] |
| s04 | - | 1/1 = 1.000 [0.207, 1.000] | 38/38 = 1.000 [0.908, 1.000] | 102/102 = 1.000 [0.964, 1.000] | 7/7 = 1.000 [0.646, 1.000] |
| s05 | 3/9 = 0.333 [0.121, 0.646] (−1) | 22/28 = 0.786 [0.605, 0.898] | 59/63 = 0.937 [0.848, 0.975] | 43/43 = 1.000 [0.918, 1.000] | 10/10 = 1.000 [0.722, 1.000] |
| s06 | - | - | 2/2 = 1.000 [0.342, 1.000] | 110/110 = 1.000 [0.966, 1.000] | 44/44 = 1.000 [0.920, 1.000] |

### k = 3 (3,33 Hz)

| oturum | <0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1.0 | >1.0 |
|---|---|---|---|---|---|
| s01 | 1/8 = 0.125 [0.022, 0.471] (−6) | 12/22 = 0.545 [0.347, 0.731] | 79/106 = 0.745 [0.655, 0.819] | 157/166 = 0.946 [0.900, 0.971] | 26/26 = 1.000 [0.871, 1.000] |
| s02 | 0/1 = 0.000 [0.000, 0.793] (−1) | - | 36/36 = 1.000 [0.904, 1.000] | 104/105 = 0.990 [0.948, 0.998] | 12/13 = 0.923 [0.667, 0.986] |
| s03 | - | - | 51/51 = 1.000 [0.930, 1.000] | 109/111 = 0.982 [0.937, 0.995] | 9/9 = 1.000 [0.701, 1.000] |
| s04 | - | 1/1 = 1.000 [0.207, 1.000] | 36/38 = 0.947 [0.827, 0.985] | 98/102 = 0.961 [0.903, 0.985] | 6/7 = 0.857 [0.487, 0.974] |
| s05 | 2/9 = 0.222 [0.063, 0.547] (−3) | 20/28 = 0.714 [0.529, 0.847] | 59/63 = 0.937 [0.848, 0.975] | 42/43 = 0.977 [0.879, 0.996] | 10/10 = 1.000 [0.722, 1.000] |
| s06 | - | - | 2/2 = 1.000 [0.342, 1.000] | 110/110 = 1.000 [0.966, 1.000] | 44/44 = 1.000 [0.920, 1.000] |

### k = 5 (2 Hz)

| oturum | <0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1.0 | >1.0 |
|---|---|---|---|---|---|
| s01 | 1/8 = 0.125 [0.022, 0.471] (−7) | 6/22 = 0.273 [0.132, 0.482] (−11) | 60/106 = 0.566 [0.471, 0.656] (−8) | 146/166 = 0.880 [0.821, 0.921] | 26/26 = 1.000 [0.871, 1.000] |
| s02 | 0/1 = 0.000 [0.000, 0.793] (−1) | - | 34/36 = 0.944 [0.819, 0.985] (−1) | 101/105 = 0.962 [0.906, 0.985] | 12/13 = 0.923 [0.667, 0.986] |
| s03 | - | - | 48/51 = 0.941 [0.841, 0.980] | 107/111 = 0.964 [0.911, 0.986] | 9/9 = 1.000 [0.701, 1.000] |
| s04 | - | 1/1 = 1.000 [0.207, 1.000] | 33/38 = 0.868 [0.727, 0.942] (−2) | 92/102 = 0.902 [0.829, 0.946] | 6/7 = 0.857 [0.487, 0.974] |
| s05 | 3/9 = 0.333 [0.121, 0.646] (−5) | 17/28 = 0.607 [0.424, 0.764] (−9) | 48/63 = 0.762 [0.644, 0.850] (−7) | 41/43 = 0.953 [0.845, 0.987] | 10/10 = 1.000 [0.722, 1.000] |
| s06 | - | - | 2/2 = 1.000 [0.342, 1.000] | 107/110 = 0.973 [0.923, 0.991] | 44/44 = 1.000 [0.920, 1.000] |

### Özet: süre kutusu başına havuzlanmış tespit (6 oturum; betimleyici — A3.2 havuzlamaz, burada yalnız k'ler arası karşılaştırma için)

| süre kutusu (s) | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|
| <0.2 | 9/18 = 0.500 | 4/18 = 0.222 (−6) | 3/18 = 0.167 (−10) | 4/18 = 0.222 (−13) |
| 0.2-0.3 | 39/51 = 0.765 | 36/51 = 0.706 | 33/51 = 0.647 | 24/51 = 0.471 (−20) |
| 0.3-0.5 | 278/296 = 0.939 | 270/296 = 0.912 | 263/296 = 0.889 | 225/296 = 0.760 (−18) |
| 0.5-1.0 | 631/637 = 0.991 | 628/637 = 0.986 | 620/637 = 0.973 | 594/637 = 0.932 |
| >1.0 | 109/109 = 1.000 | 109/109 = 1.000 | 107/109 = 0.982 | 107/109 = 0.982 |

### Normal tempolu oturumlar (s02–s04), süre ≥ 0,5 s (A3.2 'verdikt kümesi'; ön-yazılı A3.2 tahmini 'tespit ≥ 0,9')

| k | Hz | tespit | tüm dokunuş tespiti (6 oturum) | yalnız ≥1 tutulan karesi olanlar | kaçırılan (örnekleme) |
|---|---|---|---|---|---|
| 1 | 10 | 347/347 = 1.000 [0.989, 1.000] | 0.959 | 0.959 | 0/1111 |
| 2 | 5 | 347/347 = 1.000 [0.989, 1.000] | 0.942 | 0.948 | 6/1111 |
| 3 | 3,33 | 338/347 = 0.974 [0.951, 0.986] | 0.923 | 0.932 | 10/1111 |
| 5 | 2 | 327/347 = 0.942 [0.913, 0.962] | 0.859 | 0.900 | 51/1111 |

Faz kaydırması bu üç sayının yayılımı: s02–s04 (havuz, 3 oturum, n = 347) ≥ 0,5 s tespit oranı — k = 2: 0.991–1.000; k = 3: 0.974–0.988; k = 5: 0.919–0.957; tüm dokunuş tespiti — k = 2: 0.942–0.942; k = 3: 0.923–0.932; k = 5: 0.834–0.866.

**Oturum bazında (A3.2'nin kendi sınaması: verdikt kümesi s02, s03, s04'te her oturum için ≥ 0,5 s birleşik D-ham ≥ 0,9, nokta tahmini).** Birleşik = 0,5–1,0 s ve > 1,0 s kutuları toplamı; `ana` = faz 0, `faz aralığı` = o k'nin tüm fazlarında (k = 2: 2, k = 3: 3, k = 5: 5 faz) en düşük–en yüksek oran. 'Havuz' sütunu yukarıdaki tablonun aynısıdır.

| k | Hz | s02 ana | s02 faz aralığı | s03 ana | s03 faz aralığı | s04 ana | s04 faz aralığı | havuz (s02+s03+s04) | havuz faz aralığı |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 118/118 = 1.000 | - | 120/120 = 1.000 | - | 109/109 = 1.000 | - | 347/347 = 1.000 | - |
| 2 | 5 | 118/118 = 1.000 | 0.983–1.000 | 120/120 = 1.000 | 1.000–1.000 | 109/109 = 1.000 | 0.991–1.000 | 347/347 = 1.000 | 0.991–1.000 |
| 3 | 3,33 | 116/118 = 0.983 | 0.983–0.992 | 118/120 = 0.983 | 0.983–1.000 | 104/109 = 0.954 | 0.954–0.982 | 338/347 = 0.974 | 0.974–0.988 |
| 5 | 2 | 113/118 = 0.958 | 0.941–0.958 | 116/120 = 0.967 | 0.942–0.975 | 98/109 = 0.899 | 0.835–0.954 | 327/347 = 0.942 | 0.919–0.957 |

A3.2 sınaması ana koşuda (faz 0): başarısız: k = 5 s04 (98/109 = 0.899). Faz kaydırmasıyla en az bir fazda < 0,9 olan hücreler: k = 5 s04 (en düşük 0.835).

### Lojistik uydurma: d50 / d90 (s), oturum başına, ana payda (kaçırma = 0)
A3.2 ile aynı uydurma (düzenlileştirmesiz ML; süre doğrusal saniye). `ok*` = nokta tahmini gözlenen süre aralığının dışında (ekstrapolasyon; yalnız sınır olarak okunur); diğer durumlar: `tümü` = tüm dokunuşlar yakalandı (d50 < en kısa dokunuş), `ayrışma`, `b1≤0` (uydurma tespit için süreyle artış vermiyor). Aralık = d50 için kullanılabilir belirsizlik aralığı.

| oturum | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|
| s01 | d50 0.053 (SE 0.087); d90 0.487 (SE 0.041) | d50 0.212 (SE 0.039); d90 0.520 (SE 0.032) | d50 0.230 (SE 0.037); d90 0.542 (SE 0.033) | d50 0.339 (SE 0.027); d90 0.687 (SE 0.041) |
| s02 | ayrışma (154/155); d50 ∈ [0.03, 0.35] | tümü (155/155); d50 < 0.03 | ok*: d50 -2.39, d90 -0.70 → d50 ∈ [0.00, 0.03] | ok*: d50 -0.76, d90 0.27 → d50 ∈ [0.00, 0.03] |
| s03 | tümü (171/171); d50 < 0.32 | tümü (171/171); d50 < 0.32 | b1<=0 (169/171) | ok*: d50 -3.08, d90 -0.52 → d50 ∈ [0.00, 0.32] |
| s04 | tümü (148/148); d50 < 0.29 | tümü (148/148); d50 < 0.29 | b1<=0 (141/148) | ok*: d50 -0.53, d90 0.64 → d50 ∈ [0.00, 0.29] |
| s05 | d50 0.117 (SE 0.042); d90 0.265 (SE 0.029) | d50 0.167 (SE 0.036); d90 0.334 (SE 0.028) | d50 0.155 (SE 0.047); d90 0.390 (SE 0.037) | d50 0.182 (SE 0.058); d90 0.568 (SE 0.067) |
| s06 | tümü (156/156); d50 < 0.48 | tümü (156/156); d50 < 0.48 | tümü (156/156); d50 < 0.48 | ok*: d50 -1.08, d90 -0.01 → d50 ∈ [0.00, 0.48] |

Ham d50/d90 sayıları (ekstrapolasyon dahil) ve koşullu (yalnız ≥ 1 tutulan karesi olan dokunuşlar) uydurma `03_frame_rate_fits.csv`'de (`mode` = all / cond).

## Okuma (tahmini; nedensel iddia yok)
- **Durum tahmini (macro-F1) kare hızıyla monoton düşüyor**: ΔF1 k = 2 / 3 / 5 için -0.0300 / -0.0549 / -0.1489; GA'lar 0'ı ve −0,02'yi dışlıyor (k = 2: [-0.0360, -0.0239]). Bu düşüş test karesinin azalmasından değil: aynı tutulan karelerde Faz A B2'ye göre fark benzer (-0.0325 / -0.0461 / -0.1471). Tohum ve faz değişimi macro-F1'i az oynatıyor (faz aralığı k = 2 / 3 / 5: 0.0053 / 0.0031 / 0.0112; tohum sd: k = 2 0.0008, k = 3 0.0021, k = 5 0.0016); yani sıra gürültüden net ayrılıyor. k = 5'te pencere tek kare (span 0,1 s): düşüş en sert (0.7076); L = 1 hareket (hız/eğim) bağlamını yok ediyor, ama bu modelde bu nedensel olarak ayrıştırılmadı (k ve L birlikte değişti).
- **Anomali AUROC'u gürültülü ve debounce'a bağımlı.** Ana debounce kuralıyla (süre sabit) AUROC 0.847 / 0.847 / 0.797 (ΔAUROC -0.077 / -0.077 / -0.127); GA'lar çok geniş (k = 2: [-0.252, +0.099]) çünkü kat AUROC'u (tüm koşularda) 0.60–1.00 arasında oynuyor. Aynı OOF'la debounce değişince k = 2 AUROC'u 0.847 → 0.944 (MAX_GAP = 1 kare) → 0.983 (2/1 kare) oynuyor; yani k = 2'de AUROC farkı kare hızının değil, süre-sabit yuvarlama kuralının sonucu olabilir. Faz ve tohum ortalamalarında k = 2 / 3 / 5 AUROC'u (ana debounce) tohum ort. 0.854 / 0.856 / 0.813; k = 1: 0.929. Tohumlar arası sd (ortalama AUROC): k = 1 0.022, k = 2 0.007, k = 3 0.013, k = 5 0.024; k ≥ 2 ortalamaları k = 1'in altında kalıyor ve bu ayrım tohumlar arasında tekrarlıyor, ama büyüklüğü (özellikle k = 2'de) debounce kuralına bağlı.
- **Tespit/FAR**: ana debounce kuralıyla tespit α = 0,10'da 0.192 / 0.245 / 0.260 (k = 1: 0.633); FAR 0.098 / 0.078 / 0.038. FAR α civarında kalıyor (SE ≈ 0,04), tespit düşüyor: kayıp duyarlılıkta, yanlış alarmda değil. Tohum ortalamasıyla tespit: k = 1 / 2 / 3 / 5: 0.769 / 0.183 / 0.291 / 0.198.
- **Dokunuş tespiti seyreltmeye kısa dokunuşlar yüzünden duyarlı, uzun dokunuşlar değil**: normal tempolu oturumlarda (s02–s04) ≥ 0,5 s dokunuşlarda D-ham k = 1 / 2 / 3 / 5'te 1.000 / 1.000 / 0.974 / 0.942 (havuz, n = 347; A3.2'nin '≥ 0,9' tahmini havuzda k = 5'e kadar sürüyor, ama oturum bazında okunduğunda s04'te k = 5'te tutmuyor: oturum tablosuna bakın). Kayıp yapısı örneklemeyle uyumlu: k = 5'te tutulan karesi olmayan 51 dokunuşun hepsi < 0,5 s (k = 2'de 6, hepsi < 0,2 s). Tüm dokunuş tespiti 0.959 → 0.942 → 0.923 → 0.859; yani 'kısa dokunuşlar' sınırı kare periyodu (0,1·k s) ile birlikte kayıyor, ama k ≤ 3 için 0,5 s üstü dokunuşlar neredeyse tam. D-ham kare düzeyinde bir tespittir; D1/D2'yi yöneten debounce'lu tanım (D-deb) bu çalışmada hesaplanmadı.
- **Kısa dokunuşlu oturumlarda (s01, s05; lojistik uydurması mümkün olan tek oturumlar) d50 / d90 kare periyodu ile kayıyor**: s01 d90 0.49 → 0.52 → 0.54 → 0.69 s, s05 d90 0.26 → 0.33 → 0.39 → 0.57 s (k = 1 / 2 / 3 / 5); d50: s01 0.05 → 0.21 → 0.23 → 0.34, s05 0.12 → 0.17 → 0.15 → 0.18 s. Yani 'güvenilir tespit için asgari dokunuş süresi' k büyüdükçe artıyor (s05'te 0,26 → 0,57 s), ama k = 1'de 0,26–0,49 s olan d90 aralığı k = 2'de 0,33–0,52 s'ye çıkıyor (yani 5 Hz'de sınır ≤ 0,1 s kaydı). Diğer oturumlarda neredeyse hiç kaçırma olmadığı için uydurma tahmin edilemiyor ya da ekstrapolasyon (tablo).
- **Tek cümlelik toplam**: bu model ve hat için 10 Hz'den inmek durum doğruluğunu (macro-F1) en az 0,03 düşürüyor ve literal diz kuralını geçen bir kare hızı yok; buna karşılık normal tempolu oturumlarda (s02–s04 havuzu, n = 347) ≥ 0,5 s dokunuşlar 2 Hz'e kadar faz 0'da ≥ 0,94 oranında yakalanıyor (tüm fazlarda en düşük havuz oranı k = 5'te 0.919; ama oturum bazında s04'te k = 5'te 98/109 = 0,899 < 0,9; faz aralığı için oturum tablosu) ve anomali AUROC'u debounce ayarına bağlı olarak 0,75–0,98 arasında değişiyor. Diz, bu ızgarada 10 Hz ile 5 Hz arasında; ara hızlar (örn. 7,5 Hz) denenmedi (tek model, tek operatör; keşif amaçlı).

## Uyarılar
1. **Öykünme, çip değil.** Düşük hızda gerçek bir radar farklı chirp/döngü ayarı kullanabilir (örn. kare başına daha fazla çırpı, daha fazla nokta); burada her tutulan kare 10 Hz kaydın aynen kopyası. Güç/CPU/bayt kazancı ölçülmedi, yalnız kare sayısı (bayt/s ∝ 1/k) hesaplandı.
2. **k ve L birlikte değişir** (brief'in tanımı): L_k = ceil(5/k) pencere süresi k = 2'de 0,5 s'yi korur ama k = 3'te 0,4 s, k = 5'te 0,1 s'dir; k'nin tek başına etkisi ile pencere uzunluğunun etkisi ayrıştırılmadı.
3. **Etiketler 10 Hz'de yumuşatılmış** (`y_true` ±3 kare mod süzgeci); seyreltilmiş modeller bu 10 Hz etiketi son tutulan karede öğreniyor. Etiketin kendisi seyreltme altında yeniden üretilmedi.
4. **Ziyaret süreleri 0,1·k s'ye nicelenir** (sıfırıncı-derece tutma); süre tabanlı D0/D2 skorları bunun bedelini taşır. Alternatif yeniden yapılandırma (ara değerleme) denenmedi.
5. **Debounce kural duyarlılığı büyük** (yukarıdaki tablo); k ≥ 2 için 'doğru' debounce ayrıca ayarlanabilir, ama bu bir hiperparametre taraması olurdu ve brief'in dışında; ana satır brief'in süre-sabit kuralına sadıktır.
6. **AUROC gürültülü** (50 civarı normal döngü, kat sd 0,03–0,15): literal kural fiilen 'AUROC GA'sı yeterince dar mı' sorusuna dönüşüyor; k ≥ 2'de ana debounce kuralında GA alt sınırı hiçbir koşuda (ana, faz, tohum) −0,03'ü dışlamıyor (yalnız k = 2 'kare sayısı sabit' varyantında alt sınır +0,0006). Bu yüzden asıl bağlayıcı olan kural macro-F1 ölçütüdür.
7. Tek operatör, tek görev, S1 5 kat (aynı oturumlarda). D-ham, ToF dokunuşuna karşı kare düzeyi tespit; anomali döngülerine ilişkin sonuçlar s02 doğal-sapma döngülerinin alarm durumuna bakılmadan hesaplandı (yasak kural uyuldu).

## Sapmalar
`DESIGN_B.md` 'Sapmalar (tarihli)' altına tek satır eklendi (2026-11-05 B.3).

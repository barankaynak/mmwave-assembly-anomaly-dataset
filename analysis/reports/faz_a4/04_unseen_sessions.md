# A4.4 — Anomali oturumlarını hiç görmemiş durum tahmincisi (M4)

`faz_a4/04_unseen_sessions.py` tarafından üretilir (DESIGN_A4.md A4.4). **İkincil analiz**; Faz A sayıları birincil kalır. Tüm sayılar stdout'a da yazılır; seçici raporlama yoktur.

## Kurulum

- **Protokol değişmedi:** `a2_cycles` ile Faz A rolleri (eğitim katları f+3,f+4; kalibrasyon f+1,f+2; test f), pred segmentasyonu, D0–D3, standart conformal p, D3 = min(1, 2·min(p1, p2)). Yalnız B2 durum etiketlerinin kaynağı değişti.
- **Faz A (S1 OOF):** mevcut `b2__S1` — referans sütunu. Bu hücre protokolün yeniden koşusunda `reports/faz_a/04_anomaly_table.csv` ile birebir çıktı (max |fark| = 1e-16).
- **(i) S2:** mevcut `b2__S2__f0..f5` OOF (oturum dışarıda). **Dikkat:** her oturumu tahmin eden model **diğer anomali oturumlarını görmüş** olabilir (ör. s03'ü tahmin eden model s04–s06'yı ve s02'yi görür); yalnız kendi oturumunu görmedi. Bu yüzden (i) soruyu yalnız yarım yanıtlar.
- **(ii) devreye alma gerçekçi:** B2 yalnız s01 + s02 ile; **anomali oturumları s03–s06 eğitimde hiç yok.** s02 pencereleri anomali katı k'nin dışarıda bırakıldığı modelle (k = 0..4), s03–s06 pencereleri s01 + s02'nin tamamıyla eğitilen tek modelle tahmin edildi. Özellikler (21), RF (300 ağaç, balanced, random_state 20261104) ve pencere tanımı B2 ile aynı; `n_jobs = 3` (sonucu etkilemez).
- **Eşdeğerlik koruması:** betikteki B2 kopyası, Faz A OOF'unu yeniden üretti: S1 f0: y_pred uyumu 1.000000, max |proba farkı| 2.9e-08; S2 f1: y_pred uyumu 1.000000, max |proba farkı| 2.9e-08.
- **Tamlık:** (i) 16.248 pencere, 6 kat, `windows()` ile aynı (session, frame_id) kümesi. (ii) 11569 pencere = s02–s06 (16.248 − s01'in 4679); eksik y_pred yok. `labels_from_oof(strict=False)` yalnız bu iki doğrulamadan sonra çağrıldı.
- s02'nin tek parça döngüsü (loop 0, fold −1; 1 pencere) hiçbir eğitim kümesine alınmadı; k = 0 modeliyle tahmin edildi (örnek-içi değil).
- Eğitim penceresi sayıları (ii): ii_f0: 6464; ii_f1: 6528; ii_f2: 6506; ii_f3: 6499; ii_f4: 6474; ii_all: 6948.

## 1. Ana karşılaştırma — B2 × pred, D3 (Faz A Tablo 3 ile yan yana)

**α = 0.10**

| kaynak | FAR | Wilson %95 | det seq135 | det seq145 | det fast | det slow | det tum anomali | AUROC tüm (kat ort ± sd) |
|---|---|---|---|---|---|---|---|---|
| Faz A (S1 OOF) | 0.078 (4/51) | [0.031, 0.185] | 0.254 | 0.475 | 0.921 | 0.929 | 0.633 | 0.924 ± 0.035 |
| (i) S2 OOF | 0.118 (6/51) | [0.055, 0.234] | 0.607 | 0.600 | 0.646 | 0.552 | 0.601 | 0.928 ± 0.051 |
| (ii) s01+s02 | 0.060 (3/50) | [0.021, 0.162] | 0.264 | 0.480 | 0.929 | 0.973 | 0.653 | 0.938 ± 0.053 |

**α = 0.05**

| kaynak | FAR | Wilson %95 | det seq135 | det seq145 | det fast | det slow | det tum anomali | AUROC tüm (kat ort ± sd) |
|---|---|---|---|---|---|---|---|---|
| Faz A (S1 OOF) | 0.000 (0/51) | [0.000, 0.070] | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.924 ± 0.035 |
| (i) S2 OOF | 0.000 (0/51) | [0.000, 0.070] | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.928 ± 0.051 |
| (ii) s01+s02 | 0.000 (0/50) | [0.000, 0.071] | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.938 ± 0.053 |

**AUROC (D3, kat ortalaması ± sd), tür başına**

| kaynak | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|
| Faz A (S1 OOF) | 0.879 ± 0.078 | 0.918 ± 0.046 | 0.947 ± 0.029 | 0.956 ± 0.022 | 0.897 ± 0.051 | 0.952 ± 0.025 | 0.924 ± 0.035 |
| (i) S2 OOF | 0.933 ± 0.052 | 0.933 ± 0.052 | 0.929 ± 0.054 | 0.916 ± 0.048 | 0.933 ± 0.052 | 0.922 ± 0.051 | 0.928 ± 0.051 |
| (ii) s01+s02 | 0.886 ± 0.126 | 0.942 ± 0.056 | 0.962 ± 0.030 | 0.969 ± 0.029 | 0.911 ± 0.082 | 0.966 ± 0.029 | 0.938 ± 0.053 |

## 2. D1 (sıra) ve D2 (süre) tek başına — pred

**D1**

| kaynak | FAR@0,10 | FAR@0,05 | seq135 | seq145 | fast | slow | tum anomali |
|---|---|---|---|---|---|---|---|
| Faz A (S1 OOF) | 0.059 [0.020, 0.159] | 0.020 [0.003, 0.103] | AUROC 0.969; det 0.604 / 0.211 | AUROC 0.979; det 0.800 / 0.400 | AUROC 0.604; det 0.217 / 0.217 | AUROC 0.470; det 0.008 / 0.000 | AUROC 0.760; det 0.409 / 0.204 |
| (i) S2 OOF | 0.039 [0.011, 0.132] | 0.020 [0.003, 0.103] | AUROC 0.990; det 1.000 / 0.607 | AUROC 0.990; det 1.000 / 0.600 | AUROC 0.635; det 0.338 / 0.267 | AUROC 0.450; det 0.004 / 0.000 | AUROC 0.771; det 0.594 / 0.373 |
| (ii) s01+s02 | 0.060 [0.021, 0.162] | 0.020 [0.004, 0.105] | AUROC 0.971; det 0.607 / 0.214 | AUROC 0.980; det 0.800 / 0.400 | AUROC 0.589; det 0.192 / 0.133 | AUROC 0.478; det 0.024 / 0.016 | AUROC 0.755; det 0.402 / 0.186 |

**D2**

| kaynak | FAR@0,10 | FAR@0,05 | seq135 | seq145 | fast | slow | tum anomali |
|---|---|---|---|---|---|---|---|
| Faz A (S1 OOF) | 0.078 [0.031, 0.185] | 0.059 [0.020, 0.159] | AUROC 0.485; det 0.068 / 0.054 | AUROC 0.535; det 0.125 / 0.125 | AUROC 0.931; det 0.800 / 0.771 | AUROC 0.990; det 0.992 / 0.929 | AUROC 0.729; det 0.487 / 0.460 |
| (i) S2 OOF | 0.137 [0.068, 0.257] | 0.098 [0.043, 0.210] | AUROC 0.437; det 0.107 / 0.032 | AUROC 0.534; det 0.226 / 0.106 | AUROC 0.943; det 0.892 / 0.533 | AUROC 0.963; det 0.860 / 0.552 | AUROC 0.711; det 0.509 / 0.299 |
| (ii) s01+s02 | 0.100 [0.043, 0.214] | 0.040 [0.011, 0.135] | AUROC 0.484; det 0.186 / 0.061 | AUROC 0.566; det 0.236 / 0.133 | AUROC 0.952; det 0.896 / 0.883 | AUROC 0.998; det 1.000 / 0.969 | AUROC 0.746; det 0.575 / 0.506 |

(det hücreleri: α = 0,10 / α = 0,05 tespit oranı.)

## 3. Önceden yazılmış okuma kuralı — harfi harfine

Kural: *Tümü-AUROC düşüşü ≤ 0,05 ve tespit düşüşü ≤ 0,10 ise "Faz A sonuçları anomali oturumlarının görülmesine bağlı değil"; değilse makale (ii)'yi dağıtım gerçekçi tahmin olarak öne alır ve Faz A sayılarını iyimser diye niteler.* Ölçüm: B2 × pred, D3, "tüm anomali" grubu; AUROC = kat ortalaması; tespit = α = 0,10 havuzlanmış oran; düşüş = Faz A − yeni (pozitif = kötüleşme).

| kaynak | AUROC Faz A | AUROC yeni | AUROC düşüşü | AUROC ≤ 0,05 | det Faz A | det yeni | det düşüşü | det ≤ 0,10 | kural sağlandı |
|---|---|---|---|---|---|---|---|---|---|
| (i) S2 OOF | 0.924 | 0.928 | -0.004 | evet | 0.633 | 0.601 | +0.033 | evet | EVET |
| (ii) s01+s02 | 0.924 | 0.938 | -0.014 | evet | 0.633 | 0.653 | -0.020 | evet | EVET |

Tür başına tespit düşüşü (α = 0,10, D3; ek betimleyici) ve tür başına AUROC düşüşü:

| kaynak | det düşüş seq135 | det düşüş seq145 | det düşüş fast | det düşüş slow | AUROC düşüş seq135 | AUROC düşüş seq145 | AUROC düşüş fast | AUROC düşüş slow |
|---|---|---|---|---|---|---|---|---|
| (i) S2 OOF | -0.354 | -0.125 | +0.275 | +0.377 | -0.054 | -0.014 | +0.019 | +0.040 |
| (ii) s01+s02 | -0.011 | -0.005 | -0.008 | -0.043 | -0.007 | -0.023 | -0.015 | -0.013 |

**Hüküm (kuralın harfi):** her iki kaynakta da iki şart sağlandı → "Faz A sonuçları anomali oturumlarının görülmesine bağlı değil".

Not: kural "tespit düşüşü"nü tek bir sayı olarak veriyor; ana okuma "tüm anomali" havuzudur. Tür başına en büyük tespit düşüşü (ii) için -0.005, (i) için +0.377; tür başına en büyük AUROC düşüşü (ii) için -0.007, (i) için +0.040. Katlar arası AUROC sd ve ~10 normal döngü/kat nedeniyle ±0,03–0,05'lik AUROC farkları gürültü düzeyindedir (Faz A uyarı 6).

## 4. Durum macro-F1 (s02–s06)

Kare/pencere düzeyi, `fc.state_metrics` (macro-F1: y_true'da bulunan sınıflar üzerinden). Aynı 11.569 pencere (s02–s06) üzerinde.

| oturum | macro-F1 Faz A (S1 OOF) | macro-F1 (i) S2 OOF | macro-F1 (ii) s01+s02 | acc Faz A (S1 OOF) | acc (i) S2 OOF | acc (ii) s01+s02 |
|---|---|---|---|---|---|---|
| s02-s06 havuz | 0.892 | 0.867 | 0.870 | 0.902 | 0.884 | 0.885 |
| s03-s06 havuz | 0.887 | 0.861 | 0.862 | 0.897 | 0.877 | 0.877 |
| session_02 | 0.914 | 0.905 | 0.913 | 0.920 | 0.910 | 0.917 |
| session_03 | 0.897 | 0.891 | 0.892 | 0.908 | 0.902 | 0.903 |
| session_04 | 0.902 | 0.892 | 0.885 | 0.909 | 0.900 | 0.892 |
| session_05 | 0.758 | 0.715 | 0.709 | 0.873 | 0.834 | 0.830 |
| session_06 | 0.889 | 0.865 | 0.871 | 0.895 | 0.869 | 0.874 |

s05'te Box5 etiketten silindiği için (±3 kare mod yumuşatması; Faz A uyarı 1) s05 skorları kısmen hatalı referansla ölçülür.

## 5. Segmentasyon sayımları (pred)

| state_source | session | atype | n_tof_ref | n_pred_box1_entries | n_used | n_dropped_unmapped_loop | n_dropped_open_last | n_loops_with_0 | n_loops_with_1 | n_loops_with_2plus |
|---|---|---|---|---|---|---|---|---|---|---|
| Faz A (S1 OOF) | session_02_normal_seq125 | normal | 51 | 52 | 51 | 0 | 1 | 10 | 31 | 10 |
| Faz A (S1 OOF) | session_03_normal_seq135 | seq135 | 57 | 57 | 56 | 0 | 1 | 12 | 34 | 11 |
| Faz A (S1 OOF) | session_04_normal_seq145 | seq145 | 49 | 49 | 48 | 0 | 1 | 8 | 34 | 7 |
| Faz A (S1 OOF) | session_05_anomaly_fast | fast | 49 | 49 | 48 | 0 | 1 | 4 | 42 | 3 |
| Faz A (S1 OOF) | session_06_anomaly_slow | slow | 52 | 52 | 51 | 0 | 1 | 14 | 25 | 13 |
| (i) S2 OOF | session_02_normal_seq125 | normal | 51 | 52 | 51 | 0 | 1 | 11 | 29 | 11 |
| (i) S2 OOF | session_03_normal_seq135 | seq135 | 57 | 57 | 56 | 0 | 1 | 12 | 34 | 11 |
| (i) S2 OOF | session_04_normal_seq145 | seq145 | 49 | 48 | 47 | 0 | 1 | 8 | 35 | 6 |
| (i) S2 OOF | session_05_anomaly_fast | fast | 49 | 49 | 48 | 0 | 1 | 3 | 44 | 2 |
| (i) S2 OOF | session_06_anomaly_slow | slow | 52 | 52 | 50 | 1 | 1 | 8 | 38 | 6 |
| (ii) s01+s02 | session_02_normal_seq125 | normal | 51 | 51 | 50 | 0 | 1 | 10 | 32 | 9 |
| (ii) s01+s02 | session_03_normal_seq135 | seq135 | 57 | 57 | 56 | 0 | 1 | 10 | 38 | 9 |
| (ii) s01+s02 | session_04_normal_seq145 | seq145 | 49 | 47 | 45 | 1 | 1 | 8 | 37 | 4 |
| (ii) s01+s02 | session_05_anomaly_fast | fast | 49 | 49 | 48 | 0 | 1 | 3 | 44 | 2 |
| (ii) s01+s02 | session_06_anomaly_slow | slow | 52 | 52 | 51 | 0 | 1 | 9 | 35 | 8 |

## 6. Devreye alma dakikaları (eğitim setine göre yeniden etiketlendi)

Tanım: **pencere sayısı / 600** (0,1 s kare, 'fiili' tanım; A2.4(b)'deki 21,7 dk ile aynı). Bu, ToF ile **etiketli** durum-eğitim kaydı süresidir. A2.4'ün 'dk toplam 6 oturum' sütunu eğitimin altı oturumdan havuzlandığını varsayıyordu; (ii)'de eğitim yalnız s01 + s02'dir.

| kaynak | etiketli_oturumlar | egitim_penceresi | etiketli_dakika |
|---|---|---|---|
| Faz A (S1 OOF) kat modeli | 6 oturumun %80'i (5 katin ort.) | 12998.4 | 21.7 |
| (i) S2: s02'yi tahmin eden model | s01 + s03..s06 | 13978.0 | 23.3 |
| (i) S2: s03'ü tahmin eden model | s01, s02, s04..s06 | 13845.0 | 23.1 |
| (i) S2: 6 modelin ort. | diger 5 oturum | 13540.0 | 22.6 |
| (ii) ii_f0 | s01 (4679) + s02 (1785) | 6464.0 | 10.8 |
| (ii) ii_f1 | s01 (4679) + s02 (1849) | 6528.0 | 10.9 |
| (ii) ii_f2 | s01 (4679) + s02 (1827) | 6506.0 | 10.8 |
| (ii) ii_f3 | s01 (4679) + s02 (1820) | 6499.0 | 10.8 |
| (ii) ii_f4 | s01 (4679) + s02 (1795) | 6474.0 | 10.8 |
| (ii) ii_all | s01 (4679) + s02 (2269) | 6948.0 | 11.6 |

Ham kayıt süreleri: s01 = 4679 pencere ≈ 7.8 dk; s02 = 2270 pencere ≈ 3.8 dk. s03–s06'nın etiketlenmesine gerek yoktur; yalnız anomali tespiti için normal döngülerin süresi gerekir. Anomali dedektörü normal döngü süresi (Faz A rolleri, pred döngüleri, f ortalaması):

| source | eğitim dk | eğitim döngü | kalibrasyon dk | kalibrasyon döngü |
|---|---|---|---|---|
| Faz A (S1 OOF) | 1.70 | 20.40 | 1.70 | 20.40 |
| (i) S2 OOF | 1.70 | 20.40 | 1.70 | 20.40 |
| (ii) s01+s02 | 1.67 | 20.00 | 1.67 | 20.00 |

(ii) için kayıt gereksinimi böylece: durum-eğitim etiketi için s01 + s02, anomali dedektörü için s02'nin normal döngüleri (aynı oturumdan). Hiçbir anomali oturumu kaydı gerekmez.

## 7. Ek (betimleyici)

**(ii)'de test penceresi ile eğitim penceresinin aynı kareyi paylaşması** (kat k test pencereleri, ≥ 1 ortak kare; döngü sınırlarında, Faz A S1'deki %12,9'a benzer):

| fold | n_test | n_share |
|---|---|---|
| 0 | 484 | 48 |
| 1 | 420 | 56 |
| 2 | 442 | 80 |
| 3 | 449 | 68 |
| 4 | 474 | 52 |

**tof segmentasyonu (ek; okuma kuralı pred'e göre), D3, α = 0,10:**

| kaynak | FAR | det seq135 | det seq145 | det fast | det slow | det tum anomali | AUROC tüm |
|---|---|---|---|---|---|---|---|
| Faz A (S1 OOF) | 0.059 [0.020, 0.159] | 0.305 | 0.473 | 0.788 | 0.827 | 0.590 | 0.944 |
| (i) S2 OOF | 0.098 [0.043, 0.210] | 0.600 | 0.600 | 0.616 | 0.592 | 0.602 | 0.934 |
| (ii) s01+s02 | 0.039 [0.011, 0.132] | 0.211 | 0.200 | 0.294 | 0.581 | 0.321 | 0.911 |

**Doğal sapma notu:** Faz A tablosundaki `far_excl_nat` ve döngü başına skorlar bu raporda **verilmedi/yazılmadı** (s02 doğal sapma döngülerinin alarm durumuna bakmak yasak); FAR havuzlanmış, doğal sapmalar dahil standart tanımdır.

Çıktılar: `04_unseen_anomaly_table.csv` (Faz A `04_anomaly_table.csv` sütunları, 3 kaynak × 2 segmentasyon), `04_unseen_far.csv`, `04_unseen_state_metrics.csv`, `04_unseen_verdict.csv`, `04_unseen_minutes.csv`, `04_unseen_cycle_counts.csv`, `oof/b2_s0102__f0..f5.csv.gz` (+ `__all`).

<!-- YORUM-BEGIN -->

## Yorum (elle yazıldı; script yeniden koşulursa korunur)

**1. Okuma kuralı (pred, D3, tüm anomali, α = 0,10) harfi harfine sağlandı — iki kaynakta da.**
- (ii) s01 + s02, anomali oturumları eğitimde hiç yok: AUROC 0,938 (Faz A 0,924; düşüş −0,014), tespit 0,653 (Faz A 0,633; düşüş −0,020). Tür başına tespit Faz A'ya çok yakın (seq135 0,264 / 0,254; seq145 0,480 / 0,475; fast 0,929 / 0,921; slow 0,973 / 0,929); tür başına AUROC düşüşleri −0,007…−0,023, yani iyileşme yönünde ve gürültü içinde. FAR 3/50 = 0,060, Wilson [0,021, 0,162]; nominal 0,10 aralığın içinde.
- Hüküm (kuralın harfi): **"Faz A sonuçları anomali oturumlarının görülmesine bağlı değil."** "Değilse" dalı (ii)'yi öne alma/Faz A'yı iyimser niteleme **tetiklenmedi**.
- (ii) tek başına bu soruyu temiz yanıtlar: s03–s06 modelin eğitiminde yok. (i) yalnız yarım yanıt (her oturumu tahmin eden model diğer anomali oturumlarını görmüş olabilir).

**2. (i) S2: havuzlanmış kural geçti ama tür bileşimi belirgin kaydı (dürüstlük notu).**
- Havuzlanmış tespit 0,601 (Faz A 0,633; düşüş +0,033), AUROC 0,928 (düşüş −0,004) → kural sağlandı.
- Tür başına tespit ise çok farklı: sıra anomalileri **yükseldi** (seq135 0,254 → 0,607; seq145 0,475 → 0,600), tempo anomalileri **düştü** (fast 0,921 → 0,646; slow 0,929 → 0,552). Tür başına en büyük düşüş +0,377 (slow). Kural "tespit düşüşü"nü "tüm anomali" havuzu için okuyunca geçer; tür başına okunsaydı (i) için **geçmezdi** (fast ve slow > 0,10). Kuralın yazımı tek sayı verdiği için ana okuma havuzdur; bu tür kayması yine de raporlanır.
- (i)'de FAR 6/51 = 0,118 [0,055, 0,234] (nominal 0,10 aralıkta). D2 tek başına FAR 0,137 [0,068, 0,257] (α = 0,10), yani süre kanalı bu kaynakta muhafazakâr değil, hafifçe yüksek; α = 0,05'te 0,098 [0,043, 0,210] (nominal 0,05 aralığın içinde ama sınıra yakın).
- Olası neden (betimleyici, test edilmedi): S2 modelleri hedef oturumun hızını/yolunu görmeden etiketler; durum etiketi hatası oturuma göre değiştiği için süre ve sıra puanlarının dağılımı kayıyor. Havuzlanmış kural bunu görmüyor.

**3. Ek tof segmentasyonu (okuma kuralının konusu değil; yine de kayıtlı):** (ii) tof segmentasyonunda tespit 0,321 (Faz A tof 0,590; fark +0,269), AUROC 0,911 (Faz A 0,944; fark +0,033); fast tespiti 0,788 → 0,294. Bu hücre çalışma anında ToF gerektirdiği için dağıtım hücresi değildir ve kural pred için yazılmıştı; ama "devreye alma gerçekçi durum tahmincisi her hücrede zararsız" denemez. (i)'de tof tespiti 0,602, AUROC 0,934.

**4. Durum macro-F1 (s02–s06 havuz):** Faz A (S1 OOF) 0,892; (i) S2 0,867; (ii) s01+s02 0,870. Yani (ii) durum tahminini ~2,2 puan düşürüyor (acc 0,902 → 0,885), ama anomali sonuçları aynı kalıyor; durum düzeyindeki bu kayıp döngü düzeyi anomaliye yansımadı. En büyük oturum düşüşü s05 (0,758 → 0,709; Box5 etiket sorunu geçerli), s06 0,889 → 0,871. s02 (ii'de kat dışı) 0,913, Faz A'dan pratikte aynı.

**5. Devreye alma maliyeti (yeniden etiketlendi):** (ii) için ToF etiketli durum eğitimi yalnız s01 (≈ 7,8 dk) + s02 (≈ 3,8 dk) = **≈ 11,6 dk** (kat modelleri ≈ 10,8 dk); Faz A S1 modelleri ≈ 21,7 dk, S2 modelleri ≈ 22,6 dk. Anomali dedektörü ek olarak s02'den ≈ 20 eğitim + ≈ 20 kalibrasyon döngüsü (≈ 1,7 + 1,7 dk) kullanır. Hiçbir anomali oturumunun (s03–s06) etiketlenmesi gerekmez. Not: s01 rastgele toplama oturumu olduğundan (ii)'nin 11,6 dk'sı "yalnız normal akış" kaydı değildir.

**DÜZELTME (2026-11-05, Faz C.2):** Bu bölümdeki ve §6'daki dakikalar kare sayısından (pencere/600) hesaplandı; kayıtta olmayan radar periyotları (%14,7) bu hesaba girmiyor. Makale artık duvar saatini kullanıyor: s01 ≈ 8,5 dk, s02 ≈ 4,6 dk, (ii) ≈ 13 dk, kat modelleri ≈ 12,1 dk, S1 ≈ 25,4 dk, S2 ≈ 26,5 dk. Kaynak: `faz_c/02_time_gaps_t5_minutes_proposal.csv` ve `faz_c/02_time_gaps_t5_wall.csv`. `04_unseen_minutes.csv` bilerek değiştirilmedi (kare tabanlı değerlerin kaynağı olarak kalıyor; ek dosya iki tanımı da veriyor).

**6. Sınırlamalar.** Tek operatör/tek görev; (ii)'de s02 kat modelleri s01'i ve s02'nin diğer döngülerini görür (zaman olarak iç içe; sürüklenme uyarısı geçerli). (ii)'de test–eğitim pencere örtüşmesi var (kat başına 48–80 / ≈ 420–485 test penceresi; Faz A S1'deki gibi döngü sınırlarında). ~10 normal döngü/kat nedeniyle AUROC SE ≈ 0,03–0,05 ve tespit/FAR Wilson aralıkları geniştir; ±0,03'lük farklar yorumlanmamalı. Pred segmentasyonunda (ii) normal döngü sayısı 50 (Faz A 51): bir döngü eşleşmedi. s05 etiket sorunu (Faz A uyarı 1) bu analizde de geçerli.

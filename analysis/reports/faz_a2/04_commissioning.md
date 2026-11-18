# A2.4 — Devreye alma maliyeti

`faz_a2/04_commissioning.py` tarafından üretilir (DESIGN_A2.md A2.4). **İkincil analiz**: makalenin birincil sonuçları Faz A'dır; bu analiz ana model seçimini (B2 + D3) değiştirmez. Tüm sayılar stdout'a da yazılır; seçici raporlama yoktur.

## (a) Anomali dedektörü: kaç normal döngü gerekir?

**Kurulum.** R = 50 tekrar; her tekrarda `RandomState([20261104, r])` ile s02'nin 51 katlanmış normal **ToF döngüsü** (loop_id) test 11 / kalibrasyon 20 / eğitim havuzu 20 olarak dağıtılır. Dağıtım iki kaynakta **aynıdır** (eşleşik). `pred` segmentasyonunda döngüler Faz A'daki gibi ToF döngüsüne eşlenir; dağıtım ToF döngüsü bazındadır, bir ToF döngüsüne 0, 1 ya da 2+ tahmin döngüsü düşebilir. Eğitimde havuzun **ilk n döngüsü** (n ∈ {5, 10, 15, 20}; havuz sırası = karıştırılmış ToF döngü sırası, aynı ToF döngüsü içinde zaman sırası) kullanılır; `pred` hücresinde havuz 20 ToF döngüsünden gelen tahmin döngülerinin tamamıdır, yani n **döngü sayısıdır** (ToF döngüsü sayısı değil). Anomali döngüleri (s03–s06) her tekrarda test. D0/D1/D2 `a2_cycles.fit_params` ile n döngüden öğrenilir; kalibrasyon skorları 20 kalibrasyon döngüsündendir; p = standart conformal (Faz A, `cal >= s`); D3 = min(1, 2·min(p1, p2)); AUROC skoru = 1 − p3 (Faz A ile aynı). FAR = test normal döngülerinin p3 ≤ α oranı (doğal sapmalar dâhil). Yalnız mevcut S1 OOF tahminleri (yeniden eğitim yok).

| hücre | normal döngü | anomali döngü | 0 döngülü ToF döngüsü | 2+ döngülü | ort. normal döngü süresi (s) |
|---|---|---|---|---|---|
| B2 x pred | 51 | 203 | 10 | 10 | 5.00 |
| ToF x tof | 51 | 207 | 0 | 0 | 5.39 |

### D3 AUROC (ortalama ± sd, 50 tekrar)

**B2 x pred**

| n eğitim | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali | FAR@0,10 | FAR@0,05 | ort. fiili eğitim / kal. / test |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.880 ± 0.116 | 0.882 ± 0.115 | 0.883 ± 0.118 | 0.929 ± 0.081 | 0.881 ± 0.115 | 0.907 ± 0.092 | 0.893 ± 0.098 | 0.058 ± 0.102 | 0.000 ± 0.000 | 5.0 / 19.8 / 10.8 |
| 10 | 0.903 ± 0.096 | 0.906 ± 0.096 | 0.925 ± 0.070 | 0.949 ± 0.062 | 0.904 ± 0.096 | 0.937 ± 0.062 | 0.920 ± 0.073 | 0.050 ± 0.085 | 0.000 ± 0.000 | 10.0 / 19.8 / 10.8 |
| 15 | 0.913 ± 0.075 | 0.919 ± 0.071 | 0.929 ± 0.067 | 0.954 ± 0.048 | 0.916 ± 0.073 | 0.942 ± 0.052 | 0.928 ± 0.058 | 0.048 ± 0.079 | 0.000 ± 0.000 | 15.0 / 19.8 / 10.8 |
| 20 | 0.908 ± 0.081 | 0.917 ± 0.077 | 0.937 ± 0.054 | 0.954 ± 0.049 | 0.912 ± 0.078 | 0.946 ± 0.048 | 0.928 ± 0.059 | 0.049 ± 0.078 | 0.000 ± 0.000 | 19.4 / 19.8 / 10.8 |

**ToF x tof**

| n eğitim | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali | FAR@0,10 | FAR@0,05 | ort. fiili eğitim / kal. / test |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.975 ± 0.043 | 0.975 ± 0.043 | 0.954 ± 0.044 | 0.939 ± 0.060 | 0.975 ± 0.043 | 0.946 ± 0.047 | 0.961 ± 0.043 | 0.051 ± 0.086 | 0.000 ± 0.000 | 5.0 / 20.0 / 11.0 |
| 10 | 0.968 ± 0.048 | 0.968 ± 0.048 | 0.951 ± 0.046 | 0.947 ± 0.050 | 0.968 ± 0.048 | 0.949 ± 0.046 | 0.959 ± 0.046 | 0.064 ± 0.096 | 0.000 ± 0.000 | 10.0 / 20.0 / 11.0 |
| 15 | 0.975 ± 0.037 | 0.975 ± 0.037 | 0.958 ± 0.038 | 0.958 ± 0.043 | 0.975 ± 0.037 | 0.958 ± 0.039 | 0.967 ± 0.037 | 0.051 ± 0.074 | 0.000 ± 0.000 | 15.0 / 20.0 / 11.0 |
| 20 | 0.973 ± 0.040 | 0.973 ± 0.040 | 0.959 ± 0.041 | 0.957 ± 0.047 | 0.973 ± 0.040 | 0.958 ± 0.042 | 0.965 ± 0.040 | 0.055 ± 0.080 | 0.000 ± 0.000 | 20.0 / 20.0 / 11.0 |

### Tespit oranı α = 0,10 ve 0,05 (D3; ortalama ± sd)

**B2 x pred**

| n | seq(135+145) @0,10 | tempo(fast+slow) @0,10 | tum anomali @0,10 | tum anomali @0,05 |
|---|---|---|---|---|
| 5 | 17.6 ± 33.9 | 47.5 ± 35.7 | 32.1 ± 28.5 | 0.0 ± 0.0 |
| 10 | 18.7 ± 34.0 | 52.6 ± 37.0 | 35.2 ± 30.7 | 0.0 ± 0.0 |
| 15 | 28.5 ± 40.3 | 57.5 ± 37.0 | 42.6 ± 34.1 | 0.0 ± 0.0 |
| 20 | 27.8 ± 37.8 | 60.6 ± 39.6 | 43.8 ± 34.5 | 0.0 ± 0.0 |

(yüzde; tüm tür başına tespit sayıları CSV'de)

**ToF x tof**

| n | seq(135+145) @0,10 | tempo(fast+slow) @0,10 | tum anomali @0,10 | tum anomali @0,05 |
|---|---|---|---|---|
| 5 | 100.0 ± 0.0 | 66.3 ± 22.7 | 83.6 ± 11.1 | 0.0 ± 0.0 |
| 10 | 100.0 ± 0.0 | 68.8 ± 23.6 | 84.8 ± 11.5 | 0.0 ± 0.0 |
| 15 | 100.0 ± 0.0 | 70.2 ± 23.8 | 85.4 ± 11.6 | 0.0 ± 0.0 |
| 20 | 100.0 ± 0.0 | 71.1 ± 24.1 | 85.9 ± 11.8 | 0.0 ± 0.0 |

(yüzde; tüm tür başına tespit sayıları CSV'de)

### Faz A referansı (D3, aynı hücreler; birincil, ön-kayıtlı)

Faz A: eğitim ≈ 20 (2 kat), kalibrasyon ≈ 20,5 (2 kat), test ≈ 10 normal döngü/kat, 5 kat çapraz-uyum. Bu A2.4 n = 20 satırının en yakın karşılığıdır (farklar: tekrarlı rastgele dağıtım ve 11 test döngüsü).

| hücre | AUROC tüm anomali | AUROC seq(135+145) | AUROC tempo | FAR@0,10 | FAR@0,05 | det@0,10 tüm |
|---|---|---|---|---|---|---|
| B2 x pred | 0.924 ± 0.035 | 0.897 ± 0.051 | 0.952 ± 0.025 | 0.078 | 0.000 | 63.3% |
| ToF x tof | 0.959 ± 0.052 | 0.962 ± 0.052 | 0.957 ± 0.051 | 0.078 | 0.000 | 88.3% |

### Betimleyici "yeter" ölçütü (sonradan tanımlı, ön-kayıtlı DEĞİL)

- B2 x pred: tüm-anomali D3 AUROC'u n = 20 değerinin (**0.928**) 0,02 puan içinde kalan en küçük n = **10** (≈ 0.8 dk normal kayıt; ort. döngü 5.0 s). Kalibrasyon için ek **20** normal döngü (≈ 1.7 dk) ayrıca gerekir; toplam normal kayıt = n + 20 döngü.
- ToF x tof: tüm-anomali D3 AUROC'u n = 20 değerinin (**0.965**) 0,02 puan içinde kalan en küçük n = **5** (≈ 0.4 dk normal kayıt; ort. döngü 5.4 s). Kalibrasyon için ek **20** normal döngü (≈ 1.8 dk) ayrıca gerekir; toplam normal kayıt = n + 20 döngü.

### Uyarılar (a)

- **Zamansal sürüklenme:** s02 döngüleri zaman içinde bağımsız değil (operatör yorgunluğu/ısınma, hız sürüklenmesi). Rastgele dağıtım eğitim/kalibrasyon/test döngülerini zamanda iç içe geçirir; bu **iyimser** olabilir. Gerçek devreye almada eğitim yakın geçmişten, test sonradan gelir. Zaman sıralı bir dağıtım koşulmadı.
- Tek operatör ve tek oturum (s02); tekrarlar aynı 51 döngüden yeniden örnekleme olduğu için sd **yeni bir istasyona/operatöre genellemeyi** değil, dağıtım şansını ölçer. R = 50 tekrar bağımsız değildir (aynı 51 döngü).
- Test kümesi küçük (11 normal döngü): tekrar başına FAR ∈ {0, 1/11, 2/11, …}; FAR'ın nominal 0,10 ile karşılaştırılması gürültülüdür. Anomali döngüleri her tekrarda aynıdır; AUROC'taki sd yalnız normal tarafın ve parametre öğrenmenin değişkenliğini içerir.
- n küçükken D0 σ ve D2 referansları (min-ref 3) kararsızdır (σ tabanı 0,05); `pred`'de n döngü sayısıdır.

## (b) Durum tahmini: kaç etiketli döngü gerekir?

**Kurulum.** S1 katları (Faz A `folds_state.csv`); her f için eğitim = f dışındaki katlar. Eğitim havuzundan **her oturum için k döngü** (döngü = `loop_id` grubu; 6 oturum, s01 dâhil → toplam 6k döngü) rastgele seçilir; seçimler tekrar içinde **iç içedir** (aynı permütasyonun ilk k'sı). B2 = 21 özellik + RandomForest(300, balanced, random_state 20261104, n_jobs = 3), B1 = en-yakın-merkez; test = kat f'nin tamamı (3.2–3.3 bin pencere). k ∈ {1, 2, 5, 10, 20}, R = 10 tekrar; 'hepsi' = o kattaki tüm eğitim döngüleri (belirlenimci, tek koşu = Faz A S1). Macro-F1: her tekrar için 5 katın ortalaması; tablo bu 10 tekrarın ortalaması ± sd'sidir ('sd katlar' sütunu ise katlar arası sd).

B1 için `tau` Faz A'daki gibi grup bazlı iç doğrulama (%10, oturum tabakalı) ile seçilir; **k = 1**'de iç doğrulama kümesi boş kalır (her oturumda tek döngü) → `tau` eğitim pencerelerinin kendisinde seçilir (resubstitution). Bu Sapmalar'da kayıtlıdır.

**Doğrulama:** 'hepsi' koşusunun tahminleri Faz A S1 OOF ile karşılaştırıldı (pencere bazında uyum): b1: min 1.000000, ort. 1.000000; b2: min 1.000000, ort. 1.000000.

### Ortalama döngü süresi (oturum başına) ve kayıt dakikası

| oturum | loop_id grubu | tam döngü | ort. döngü süresi (s) | ort. tüm-loop süresi (s) |
|---|---|---|---|---|
| session_01_training | 57 | 55 | 8.22 | 8.22 |
| session_02_normal_seq125 | 52 | 51 | 4.45 | 4.37 |
| session_03_normal_seq135 | 58 | 57 | 4.21 | 4.15 |
| session_04_normal_seq145 | 50 | 49 | 3.92 | 3.86 |
| session_05_anomaly_fast | 50 | 49 | 3.62 | 3.56 |
| session_06_anomaly_slow | 53 | 52 | 6.14 | 6.04 |

Ortalama döngü süresi (6 oturum ortalaması) = 5.09 s; 6 oturumun toplamı = 30.6 s (s01 için tüm-loop ortalaması; s01 rastgele toplama, döngü = tam 4 adımlı döngü değil). Dakika = k × ortalama döngü süresi; 'toplam' = bütün 6 oturumu birlikte kaydetmenin maliyeti (deneyde eğitim bütün oturumlardan havuzlandığı için), 'oturum başına' = tek bir oturum/istasyon için k döngü. 'fiili toplam' = seçilen loop'ların gerçek kare sayısından hesaplanan toplam kayıt dakikası (6 oturum).

### B2 (el yapımı özellik + RF): macro-F1 / doğruluk

| k (oturum başına) | macro-F1 (ort ± sd tekrarlar) | sd katlar | min–maks tekrar | doğruluk | dk/oturum (nominal) | dk toplam 6 oturum (nominal) | dk toplam (fiili) | eğitim penceresi |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.7133 ± 0.0202 | 0.0058 | 0.685–0.736 | 0.7897 ± 0.0089 | 0.1 | 0.5 | 0.5 | 304 |
| 2 | 0.7701 ± 0.0130 | 0.0164 | 0.752–0.794 | 0.8208 ± 0.0059 | 0.2 | 1.0 | 1.0 | 612 |
| 5 | 0.8098 ± 0.0038 | 0.0113 | 0.805–0.816 | 0.8471 ± 0.0023 | 0.4 | 2.5 | 2.5 | 1526 |
| 10 | 0.8287 ± 0.0038 | 0.0115 | 0.824–0.834 | 0.8620 ± 0.0029 | 0.8 | 5.1 | 5.0 | 3003 |
| 20 | 0.8424 ± 0.0019 | 0.0122 | 0.840–0.846 | 0.8732 ± 0.0011 | 1.7 | 10.2 | 10.0 | 6013 |
| all | 0.8564 | 0.0122 | 0.856–0.856 | 0.8841 | — | — | 21.7 | 12998 |

### B1 (en-yakın-merkez): macro-F1 / doğruluk

| k (oturum başına) | macro-F1 (ort ± sd tekrarlar) | sd katlar | min–maks tekrar | doğruluk | dk/oturum (nominal) | dk toplam 6 oturum (nominal) | dk toplam (fiili) | eğitim penceresi |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.4913 ± 0.0134 | 0.0118 | 0.474–0.515 | 0.5765 ± 0.0184 | 0.1 | 0.5 | 0.5 | 304 |
| 2 | 0.4959 ± 0.0143 | 0.0112 | 0.466–0.512 | 0.5670 ± 0.0189 | 0.2 | 1.0 | 1.0 | 612 |
| 5 | 0.5081 ± 0.0112 | 0.0129 | 0.497–0.531 | 0.5817 ± 0.0216 | 0.4 | 2.5 | 2.5 | 1526 |
| 10 | 0.5188 ± 0.0070 | 0.0131 | 0.507–0.528 | 0.6047 ± 0.0144 | 0.8 | 5.1 | 5.0 | 3003 |
| 20 | 0.5272 ± 0.0054 | 0.0166 | 0.513–0.532 | 0.6110 ± 0.0015 | 1.7 | 10.2 | 10.0 | 6013 |
| all | 0.5265 | 0.0136 | 0.527–0.527 | 0.6130 | — | — | 21.7 | 12998 |

### Betimleyici "yeter" ölçütü (sonradan tanımlı, ön-kayıtlı DEĞİL)

- B2: tüm eğitim verisiyle macro-F1 = 0.8564; bunun %95'ine (= 0.8136) ulaşan en küçük k = **10** döngü/oturum (macro-F1 0.8287; ≈ 0.8 dk/oturum nominal, 6 oturum toplam ≈ 5.1 dk).
- B1: tüm eğitim verisiyle macro-F1 = 0.5265; bunun %95'ine (= 0.5002) ulaşan en küçük k = **5** döngü/oturum (macro-F1 0.5081; ≈ 0.4 dk/oturum nominal, 6 oturum toplam ≈ 2.5 dk).

### B2 oturum başına macro-F1 (5 kat × tekrar ortalaması; test pencerelerinde oturum bazında)

| k | session_01 | session_02 | session_03 | session_04 | session_05 | session_06 |
|---|---|---|---|---|---|---|
| 1 | 0.611 | 0.823 | 0.796 | 0.791 | 0.673 | 0.799 |
| 2 | 0.666 | 0.846 | 0.841 | 0.839 | 0.705 | 0.829 |
| 5 | 0.710 | 0.872 | 0.869 | 0.865 | 0.725 | 0.857 |
| 10 | 0.736 | 0.890 | 0.880 | 0.880 | 0.739 | 0.870 |
| 20 | 0.751 | 0.902 | 0.889 | 0.891 | 0.749 | 0.882 |
| all | 0.770 | 0.915 | 0.896 | 0.901 | 0.755 | 0.889 |

### Uyarılar (b)

- Katlar `StratifiedGroupKFold` ile (oturum, loop_id) gruplarına göre ayrıldığından seçilen döngüler **ardışık zaman** değil rastgeledir (aynı sürüklenme uyarısı); test döngüleri eğitim döngülerinin zamansal komşusu olabilir.
- Eğitim **6 oturumdan havuzlanmış**tır (hand-box durum kestirimi senaryodan bağımsız varsayımı, Faz A ile aynı). Yeni bir istasyon için '6k döngü' pratikte tek istasyondan değil, mevcut 6 oturumluk çeşitlilikten gelir; tek oturumdan k döngü ile eğitilen model bu deneyde ölçülmedi. Dolayısıyla 'toplam 6 oturum' dakikası bir üst sınır değil, deneyin gerçek maliyetidir; tek istasyon maliyeti için 'oturum başına' sütunu kullanılır ama genelleme iddiası taşımaz.
- CNN-GRU bu analize dâhil değildir (kapsam dışı; brifte yalnız B2 ve B1). Etiketler ToF'tan bedava; maliyet = kayıt dakikası.
- 'döngü' = `loop_id` grubu; s01 (rastgele toplama) ve oturum başı/sonu parçaları tam 4 adımlı döngü olmayabilir; fiili dakika bu yüzden nominal değerden sapabilir.

## Makale cümlesi için özet (ikincil)

- B2 x pred: n=5: AUROC 0.893, FAR@0,10 0.058 ; n=10: AUROC 0.920, FAR@0,10 0.050 ; n=15: AUROC 0.928, FAR@0,10 0.048 ; n=20: AUROC 0.928, FAR@0,10 0.049
- ToF x tof: n=5: AUROC 0.961, FAR@0,10 0.051 ; n=10: AUROC 0.959, FAR@0,10 0.064 ; n=15: AUROC 0.967, FAR@0,10 0.051 ; n=20: AUROC 0.965, FAR@0,10 0.055
- B2 macro-F1: k=1: 0.713 ; k=2: 0.770 ; k=5: 0.810 ; k=10: 0.829 ; k=20: 0.842 ; k=all: 0.856
- B1 macro-F1: k=1: 0.491 ; k=2: 0.496 ; k=5: 0.508 ; k=10: 0.519 ; k=20: 0.527 ; k=all: 0.527


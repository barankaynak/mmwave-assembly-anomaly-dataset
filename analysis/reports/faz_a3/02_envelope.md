# Faz A3.2 - Calisma zarfi: sistem neyi goremiyor?

Brief: `faz_a3/DESIGN_A3.md` (A3.2; on-kayitli). **Yeniden egitim yok.** Statu: ikincil/sonradan analiz; birincil sonuclar Faz A'nin on-kayitli sayilaridir (ana model B2 + D3 degismez). Her sayi `faz_a3/02_envelope.py` ciktisinda da basilidir. Faz A 04 dongu skorlarina / dogal sapma dongulerine bakilmadi.

Capraz kontrol (s02-s06 y_pred akisi == `ac.source_labels`, debounce == `ac.debounce_labels`): **equal: True**
- Kare tablosu 16272 kare (6 oturum); OOF'lar 16,248 pencere / 5 kat (dort model icin dogrulandi); oturum basina ilk 4 kare penceresiz -> y_pred = none (0) (A2 ile ayni).
- s01 (egitim oturumu) de kapsamdadir (brief: kisa dokunuslar s05 Box5 ve s01 Box4'te); `labels_from_oof` s01'i dusurdugu icin s01 akisi bu betikte elle kuruldu, ayni `debounce_labels` ile.

## 0. Tanimlar ve hizalama (ozet; on-kayitli tanimlar degismedi)

- **Dokunus:** kutu ve oturum basina, ham 30 Hz ToF'ta `mesafe < 140 mm` olan kesintisiz ornek kosusu; >= 1 esik-ustu (ya da NaN) ornek dokunusu boler. **Sure** = t[son ornek] - t[ilk ornek] + oturumun medyan ToF dt'si (tek orneklik dokunus ~1 dt; sure tanimi brifte yoktu, bkz. Sapmalar).
- **ToF -> kare hizalamasi (A2.3 `reports/faz_a2/03_labels.md` ile ayni):** ToF ornegi, `(t_kare - 100 ms, t_kare]` araligina dusen kareye atanir (`t` = `frame_table().t`; ToF `t` ayni saat; `searchsorted(side='left')` + `t_kare - t_tof < 0.1 s`). Bir dokunusun *kare kumesi* = en az bir orneginin atandigi kareler. `L3` etiketindeki `(t-100ms, t]` kuraliyla ayni; `frame_id` atamasi (`merged_df`) kullanilmadi.
- **D-ham:** kare kumesinde en az bir karenin (= pencerenin son karesi) ham S1 OOF `y_pred`'i o kutu. **D-deb:** ayni, `debounce_labels(MIN_BOX=2, MAX_GAP=1)` sonrasi akista (**D1/D2'yi yoneten tanim, pred segmentasyonu**).
- **Kare kumesi bos dokunus (n_nomap):** ornekleri hicbir karenin pencerenin icine dusmeyen dokunuslar (radar kare aralarindaki bosluklar; A2.3: ToF satirlarinin %7-26'si kendi karesinden >100 ms once - ayri satir olarak degil, dokunus duzeyinde sayilir; bu veride sayi asagida). Bunlar icin tespit **tanimsizdir** (kacirma degil); payda disi birakilir ve her tabloda ayri raporlanir.
- Sure kutulari (sol-kapali `[lo, hi)`; son kutu >= 1.0 s): <0.2, 0.2-0.3, 0.3-0.5, 0.5-1.0, >1.0 s. Wilson %95. Dokunus ici >0.2 s ornek-arasi bosluklu dokunus sayisi: 0 (bilgi).
- Toplam dokunus: 1111 (oturum x kutu); kare kumesi bos: 0.

## 1. Dokunus sayimi (oturum x kutu x sure kutusu; kare kumesi bos olanlar `(+n)` ile)

| oturum | kutu | n | medyan_sure_s | min_s | max_s | <0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1.0 | >1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| s01 | 1 | 57 | 0.538 | 0.222 | 1.458 | 0 | 3 | 24 | 24 | 6 |
| s01 | 2 | 56 | 0.541 | 0.191 | 1.367 | 1 | 3 | 22 | 28 | 2 |
| s01 | 3 | 51 | 0.509 | 0.032 | 1.559 | 2 | 3 | 20 | 22 | 4 |
| s01 | 4 | 58 | 0.381 | 0.032 | 1.399 | 5 | 10 | 16 | 22 | 5 |
| s01 | 5 | 106 | 0.541 | 0.223 | 2.799 | 0 | 3 | 24 | 70 | 9 |
| s02 | 1 | 51 | 0.635 | 0.507 | 1.425 | 0 | 0 | 0 | 45 | 6 |
| s02 | 2 | 51 | 0.636 | 0.445 | 1.877 | 0 | 0 | 2 | 43 | 6 |
| s02 | 5 | 53 | 0.477 | 0.032 | 1.208 | 1 | 0 | 34 | 17 | 1 |
| s03 | 1 | 57 | 0.665 | 0.475 | 1.710 | 0 | 0 | 1 | 48 | 8 |
| s03 | 3 | 57 | 0.604 | 0.446 | 1.431 | 0 | 0 | 2 | 54 | 1 |
| s03 | 5 | 57 | 0.445 | 0.317 | 0.573 | 0 | 0 | 48 | 9 | 0 |
| s04 | 1 | 49 | 0.697 | 0.476 | 1.237 | 0 | 0 | 1 | 43 | 5 |
| s04 | 4 | 50 | 0.604 | 0.413 | 1.241 | 0 | 0 | 3 | 45 | 2 |
| s04 | 5 | 49 | 0.445 | 0.286 | 0.636 | 0 | 1 | 34 | 14 | 0 |
| s05 | 1 | 49 | 0.571 | 0.285 | 1.457 | 0 | 1 | 14 | 24 | 10 |
| s05 | 2 | 52 | 0.461 | 0.032 | 0.954 | 2 | 1 | 30 | 19 | 0 |
| s05 | 5 | 52 | 0.286 | 0.032 | 0.445 | 7 | 26 | 19 | 0 | 0 |
| s06 | 1 | 52 | 1.029 | 0.539 | 1.711 | 0 | 0 | 0 | 22 | 30 |
| s06 | 2 | 52 | 0.858 | 0.636 | 2.163 | 0 | 0 | 0 | 39 | 13 |
| s06 | 5 | 52 | 0.667 | 0.477 | 1.082 | 0 | 0 | 2 | 49 | 1 |

## 2. Tespit orani ~ sure kutusu (B2, birincil; Wilson %95; `k/n = oran [alt, ust]`; oturum basina, havuzlanmadi)

### B2 - D-ham

| oturum | <0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1.0 | >1.0 |
|---|---|---|---|---|---|
| s01 | 5/8 = 0.625 [0.306, 0.863] | 14/22 = 0.636 [0.430, 0.803] | 88/106 = 0.830 [0.747, 0.890] | 160/166 = 0.964 [0.923, 0.983] | 26/26 = 1.000 [0.871, 1.000] |
| s02 | 0/1 = 0.000 [0.000, 0.793] | - | 36/36 = 1.000 [0.904, 1.000] | 105/105 = 1.000 [0.965, 1.000] | 13/13 = 1.000 [0.772, 1.000] |
| s03 | - | - | 51/51 = 1.000 [0.930, 1.000] | 111/111 = 1.000 [0.967, 1.000] | 9/9 = 1.000 [0.701, 1.000] |
| s04 | - | 1/1 = 1.000 [0.207, 1.000] | 38/38 = 1.000 [0.908, 1.000] | 102/102 = 1.000 [0.964, 1.000] | 7/7 = 1.000 [0.646, 1.000] |
| s05 | 4/9 = 0.444 [0.189, 0.733] | 24/28 = 0.857 [0.685, 0.943] | 63/63 = 1.000 [0.943, 1.000] | 43/43 = 1.000 [0.918, 1.000] | 10/10 = 1.000 [0.722, 1.000] |
| s06 | - | - | 2/2 = 1.000 [0.342, 1.000] | 110/110 = 1.000 [0.966, 1.000] | 44/44 = 1.000 [0.920, 1.000] |

### B2 - D-deb (D1/D2'yi yoneten tanim)

| oturum | <0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1.0 | >1.0 |
|---|---|---|---|---|---|
| s01 | 5/8 = 0.625 [0.306, 0.863] | 13/22 = 0.591 [0.387, 0.767] | 84/106 = 0.792 [0.706, 0.859] | 158/166 = 0.952 [0.908, 0.975] | 26/26 = 1.000 [0.871, 1.000] |
| s02 | 0/1 = 0.000 [0.000, 0.793] | - | 36/36 = 1.000 [0.904, 1.000] | 105/105 = 1.000 [0.965, 1.000] | 12/13 = 0.923 [0.667, 0.986] |
| s03 | - | - | 51/51 = 1.000 [0.930, 1.000] | 111/111 = 1.000 [0.967, 1.000] | 9/9 = 1.000 [0.701, 1.000] |
| s04 | - | 1/1 = 1.000 [0.207, 1.000] | 38/38 = 1.000 [0.908, 1.000] | 102/102 = 1.000 [0.964, 1.000] | 7/7 = 1.000 [0.646, 1.000] |
| s05 | 3/9 = 0.333 [0.121, 0.646] | 22/28 = 0.786 [0.605, 0.898] | 60/63 = 0.952 [0.869, 0.984] | 43/43 = 1.000 [0.918, 1.000] | 10/10 = 1.000 [0.722, 1.000] |
| s06 | - | - | 2/2 = 1.000 [0.342, 1.000] | 110/110 = 1.000 [0.966, 1.000] | 44/44 = 1.000 [0.920, 1.000] |

## 3. Lojistik uydurma: logit P(tespit) = b0 + b1 * sure(s); d50 = -b0/b1, d90 = (ln 9 - b0)/b1

Duzenlilestirmesiz ML (`statsmodels.Logit`), oturum x model x tanim basina, kare kumesi bos olmayan dokunuslar uzerinde, **sure dogrusal saniye olarak**. SE: delta yontemi (yaklasik). **Tahmin edilemez** durumlar: n=0; tum sonuclar ayni; tam ayrisma; b1<=0; yakinsamama/sonsuz SE. Tam ayrismada d50 icin yalniz **aralik** verilir: [en uzun kacirilan, en kisa yakalanan]; 'tumu tespit' durumunda d50 < min sure. `ekstrapolasyon` = d50 ya da d90 uydurulan surelerin [min, max] disinda.

### B2 - D-ham

| model | oturum | tanim | n | yakalanan | durum | b0 | b1 | d50_s | d50_SE | d90_s | d90_SE | d50_araligi | ekstrapolasyon | sure_aralik_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B2 | s01 | D-ham | 328 | 293 | ok | -0.270 | 5.071 | 0.053 | 0.087 | 0.487 | 0.041 | [0.053, 0.053] |  | [0.032, 2.799] |
| B2 | s02 | D-ham | 155 | 154 | tam ayrisma (tahmin edilemez) | nan | nan | nan | nan | nan | nan | [0.032, 0.350] |  | [0.032, 1.877] |
| B2 | s03 | D-ham | 171 | 171 | tumu tespit (d50 < min sure) | nan | nan | nan | nan | nan | nan | [0.000, 0.317] |  | [0.317, 1.710] |
| B2 | s04 | D-ham | 148 | 148 | tumu tespit (d50 < min sure) | nan | nan | nan | nan | nan | nan | [0.000, 0.286] |  | [0.286, 1.241] |
| B2 | s05 | D-ham | 153 | 144 | ok | -1.755 | 14.934 | 0.117 | 0.042 | 0.265 | 0.029 | [0.117, 0.117] |  | [0.032, 1.457] |
| B2 | s06 | D-ham | 156 | 156 | tumu tespit (d50 < min sure) | nan | nan | nan | nan | nan | nan | [0.000, 0.477] |  | [0.477, 2.163] |

### B2 - D-deb

| model | oturum | tanim | n | yakalanan | durum | b0 | b1 | d50_s | d50_SE | d90_s | d90_SE | d50_araligi | ekstrapolasyon | sure_aralik_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B2 | s01 | D-deb | 328 | 286 | ok | -0.471 | 5.004 | 0.094 | 0.076 | 0.533 | 0.043 | [0.094, 0.094] |  | [0.032, 2.799] |
| B2 | s02 | D-deb | 155 | 153 | ok | 3.937 | 0.629 | -6.255 | 36.644 | -2.764 | 18.067 | [0.000, 0.032] | EVET | [0.032, 1.877] |
| B2 | s03 | D-deb | 171 | 171 | tumu tespit (d50 < min sure) | nan | nan | nan | nan | nan | nan | [0.000, 0.317] |  | [0.317, 1.710] |
| B2 | s04 | D-deb | 148 | 148 | tumu tespit (d50 < min sure) | nan | nan | nan | nan | nan | nan | [0.000, 0.286] |  | [0.286, 1.241] |
| B2 | s05 | D-deb | 153 | 138 | ok | -2.120 | 13.242 | 0.160 | 0.037 | 0.326 | 0.028 | [0.160, 0.160] |  | [0.032, 1.457] |
| B2 | s06 | D-deb | 156 | 156 | tumu tespit (d50 < min sure) | nan | nan | nan | nan | nan | nan | [0.000, 0.477] |  | [0.477, 2.163] |

### CNN-GRU - D-ham

| model | oturum | tanim | n | yakalanan | durum | b0 | b1 | d50_s | d50_SE | d90_s | d90_SE | d50_araligi | ekstrapolasyon | sure_aralik_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN-GRU | s01 | D-ham | 328 | 284 | ok | -1.610 | 7.705 | 0.209 | 0.038 | 0.494 | 0.030 | [0.209, 0.209] |  | [0.032, 2.799] |
| CNN-GRU | s02 | D-ham | 155 | 153 | ok | 3.937 | 0.629 | -6.255 | 36.644 | -2.764 | 18.067 | [0.000, 0.032] | EVET | [0.032, 1.877] |
| CNN-GRU | s03 | D-ham | 171 | 169 | ok | -0.782 | 10.485 | 0.075 | 0.267 | 0.284 | 0.127 | [0.000, 0.317] | EVET | [0.317, 1.710] |
| CNN-GRU | s04 | D-ham | 148 | 146 | ok | 1.028 | 5.992 | -0.172 | 0.727 | 0.195 | 0.344 | [0.000, 0.286] | EVET | [0.286, 1.241] |
| CNN-GRU | s05 | D-ham | 153 | 145 | ok | -1.669 | 15.354 | 0.109 | 0.043 | 0.252 | 0.030 | [0.109, 0.109] |  | [0.032, 1.457] |
| CNN-GRU | s06 | D-ham | 156 | 156 | tumu tespit (d50 < min sure) | nan | nan | nan | nan | nan | nan | [0.000, 0.477] |  | [0.477, 2.163] |

### CNN-GRU - D-deb

| model | oturum | tanim | n | yakalanan | durum | b0 | b1 | d50_s | d50_SE | d90_s | d90_SE | d50_araligi | ekstrapolasyon | sure_aralik_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN-GRU | s01 | D-deb | 328 | 274 | ok | -1.386 | 6.400 | 0.217 | 0.042 | 0.560 | 0.036 | [0.217, 0.217] |  | [0.032, 2.799] |
| CNN-GRU | s02 | D-deb | 155 | 152 | ok | 3.867 | 0.089 | -43.256 | 1223.952 | -18.679 | 538.822 | [0.000, 0.032] | EVET | [0.032, 1.877] |
| CNN-GRU | s03 | D-deb | 171 | 169 | ok | -0.782 | 10.485 | 0.075 | 0.267 | 0.284 | 0.127 | [0.000, 0.317] | EVET | [0.317, 1.710] |
| CNN-GRU | s04 | D-deb | 148 | 145 | b1<=0 (tahmin edilemez) | 4.679 | -1.251 | nan | nan | nan | nan |  |  | [0.286, 1.241] |
| CNN-GRU | s05 | D-deb | 153 | 132 | ok | -0.197 | 5.182 | 0.038 | 0.108 | 0.462 | 0.063 | [0.038, 0.038] |  | [0.032, 1.457] |
| CNN-GRU | s06 | D-deb | 156 | 155 | b1<=0 (tahmin edilemez) | 6.526 | -1.514 | nan | nan | nan | nan |  |  | [0.477, 2.163] |

### CNN-only - D-ham

| model | oturum | tanim | n | yakalanan | durum | b0 | b1 | d50_s | d50_SE | d90_s | d90_SE | d50_araligi | ekstrapolasyon | sure_aralik_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN-only | s01 | D-ham | 328 | 302 | ok | -1.702 | 10.025 | 0.170 | 0.039 | 0.389 | 0.025 | [0.170, 0.170] |  | [0.032, 2.799] |
| CNN-only | s02 | D-ham | 155 | 155 | tumu tespit (d50 < min sure) | nan | nan | nan | nan | nan | nan | [0.000, 0.032] |  | [0.032, 1.877] |
| CNN-only | s03 | D-ham | 171 | 171 | tumu tespit (d50 < min sure) | nan | nan | nan | nan | nan | nan | [0.000, 0.317] |  | [0.317, 1.710] |
| CNN-only | s04 | D-ham | 148 | 148 | tumu tespit (d50 < min sure) | nan | nan | nan | nan | nan | nan | [0.000, 0.286] |  | [0.286, 1.241] |
| CNN-only | s05 | D-ham | 153 | 150 | ok | -2.408 | 33.990 | 0.071 | 0.030 | 0.135 | 0.037 | [0.071, 0.071] |  | [0.032, 1.457] |
| CNN-only | s06 | D-ham | 156 | 156 | tumu tespit (d50 < min sure) | nan | nan | nan | nan | nan | nan | [0.000, 0.477] |  | [0.477, 2.163] |

### CNN-only - D-deb

| model | oturum | tanim | n | yakalanan | durum | b0 | b1 | d50_s | d50_SE | d90_s | d90_SE | d50_araligi | ekstrapolasyon | sure_aralik_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN-only | s01 | D-deb | 328 | 285 | ok | -1.107 | 6.498 | 0.170 | 0.049 | 0.508 | 0.034 | [0.170, 0.170] |  | [0.032, 2.799] |
| CNN-only | s02 | D-deb | 155 | 154 | b1<=0 (tahmin edilemez) | 8.478 | -4.020 | nan | nan | nan | nan |  |  | [0.032, 1.877] |
| CNN-only | s03 | D-deb | 171 | 170 | ok | 3.851 | 2.238 | -1.721 | 6.563 | -0.739 | 3.732 | [0.000, 0.317] | EVET | [0.317, 1.710] |
| CNN-only | s04 | D-deb | 148 | 145 | ok | 3.454 | 0.701 | -4.926 | 26.165 | -1.792 | 11.336 | [0.000, 0.286] | EVET | [0.286, 1.241] |
| CNN-only | s05 | D-deb | 153 | 143 | ok | -0.267 | 8.444 | 0.032 | 0.089 | 0.292 | 0.042 | [0.000, 0.032] | EVET | [0.032, 1.457] |
| CNN-only | s06 | D-deb | 156 | 156 | tumu tespit (d50 < min sure) | nan | nan | nan | nan | nan | nan | [0.000, 0.477] |  | [0.477, 2.163] |

### B1 - D-ham

| model | oturum | tanim | n | yakalanan | durum | b0 | b1 | d50_s | d50_SE | d90_s | d90_SE | d50_araligi | ekstrapolasyon | sure_aralik_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | s01 | D-ham | 328 | 241 | ok | -1.416 | 4.808 | 0.295 | 0.040 | 0.752 | 0.056 | [0.295, 0.295] |  | [0.032, 2.799] |
| B1 | s02 | D-ham | 155 | 153 | b1<=0 (tahmin edilemez) | 5.872 | -2.103 | nan | nan | nan | nan |  |  | [0.032, 1.877] |
| B1 | s03 | D-ham | 171 | 169 | ok | -7.053 | 26.808 | 0.263 | 0.056 | 0.345 | 0.029 | [0.000, 0.317] | EVET | [0.317, 1.710] |
| B1 | s04 | D-ham | 148 | 146 | b1<=0 (tahmin edilemez) | 5.727 | -2.173 | nan | nan | nan | nan |  |  | [0.286, 1.241] |
| B1 | s05 | D-ham | 153 | 144 | ok | 0.299 | 6.875 | -0.043 | 0.137 | 0.276 | 0.052 | [0.000, 0.032] | EVET | [0.032, 1.457] |
| B1 | s06 | D-ham | 156 | 155 | ok | -0.377 | 7.418 | 0.051 | 0.643 | 0.347 | 0.339 | [0.000, 0.477] | EVET | [0.477, 2.163] |

### B1 - D-deb

| model | oturum | tanim | n | yakalanan | durum | b0 | b1 | d50_s | d50_SE | d90_s | d90_SE | d50_araligi | ekstrapolasyon | sure_aralik_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | s01 | D-deb | 328 | 210 | ok | -1.120 | 3.146 | 0.356 | 0.048 | 1.054 | 0.113 | [0.356, 0.356] |  | [0.032, 2.799] |
| B1 | s02 | D-deb | 155 | 148 | b1<=0 (tahmin edilemez) | 3.427 | -0.563 | nan | nan | nan | nan |  |  | [0.032, 1.877] |
| B1 | s03 | D-deb | 171 | 162 | ok | 1.331 | 2.735 | -0.487 | 0.874 | 0.317 | 0.224 | [0.000, 0.317] | EVET | [0.317, 1.710] |
| B1 | s04 | D-deb | 148 | 129 | ok | 1.004 | 1.525 | -0.658 | 1.223 | 0.782 | 0.256 | [0.000, 0.286] | EVET | [0.286, 1.241] |
| B1 | s05 | D-deb | 153 | 131 | ok | 1.005 | 1.751 | -0.574 | 0.635 | 0.681 | 0.216 | [0.000, 0.032] | EVET | [0.032, 1.457] |
| B1 | s06 | D-deb | 156 | 153 | ok | 3.035 | 1.044 | -2.908 | 8.560 | -0.802 | 3.766 | [0.000, 0.477] | EVET | [0.477, 2.163] |

Not: `ekstrapolasyon = EVET` iken d50/d90 gozlenen sure araliginin disindadir (orn. negatif sure); bu satirlar yalniz **sinir** olarak okunur (d50 < min sure ya da > max sure; Bolum 4'te aralik olarak kullanilir), nokta tahmini olarak degil. Dogrusal-saniye lojistik, kisa dokunuslarda plato yapan oranlari (orn. s01'de <0.3 s kutularinda ~0.6) iyi yakalamaz; bu yuzden kutu orani tablolari (Bolum 2) esas, uydurma ozet olarak okunmalidir.

## 4. On-yazili tahminler ve sabit kural (B2; kurallar kelimesi kelimesine)

### (a) B2'de d50 in [0.2, 0.4] s (D-ham) - oturum basina

| oturum | n | durum | d50_ham_s | d50_araligi | [0.2, 0.4] icinde? |
|---|---|---|---|---|---|
| s01 | 328 | ok | 0.053 | [0.053, 0.053] | TUTMAZ |
| s02 | 155 | tam ayrisma (tahmin edilemez) | nan | [0.032, 0.350] | belirsiz |
| s03 | 171 | tumu tespit (d50 < min sure) | nan | [0.000, 0.317] | belirsiz |
| s04 | 148 | tumu tespit (d50 < min sure) | nan | [0.000, 0.286] | belirsiz |
| s05 | 153 | ok | 0.117 | [0.117, 0.117] | TUTMAZ |
| s06 | 156 | tumu tespit (d50 < min sure) | nan | [0.000, 0.477] | belirsiz |

Ozet (a): TUTAR 0, TUTMAZ 2, belirsiz/aralik-yok 4 (6 oturumdan); d50 nokta olarak tahmin edilebilen oturum: 2. **Tahmin (a) tum oturumlarda TUTUYOR mu: HAYIR.**

### (b) Normal tempolu oturumlarda > 0.5 s dokunuslar icin tespit >= 0.9 (B2; D-ham; verdikt kumesi: s02, s03, s04; s01/s05/s06 bilgi)

| oturum | normal tempo (verdikt kumesi) | 0.5-1.0 D-ham | 0.5-1.0 D-deb | >1.0 D-ham | >1.0 D-deb | >=0.5 (birlesik) D-ham | >=0.5 (birlesik) D-deb | D-ham >=0.9? |
|---|---|---|---|---|---|---|---|---|
| s01 | hayir | 160/166 = 0.964 [0.923, 0.983] | 158/166 = 0.952 [0.908, 0.975] | 26/26 = 1.000 [0.871, 1.000] | 26/26 = 1.000 [0.871, 1.000] | 186/192 = 0.969 [0.934, 0.986] | 184/192 = 0.958 [0.920, 0.979] | EVET |
| s02 | EVET | 105/105 = 1.000 [0.965, 1.000] | 105/105 = 1.000 [0.965, 1.000] | 13/13 = 1.000 [0.772, 1.000] | 12/13 = 0.923 [0.667, 0.986] | 118/118 = 1.000 [0.968, 1.000] | 117/118 = 0.992 [0.954, 0.999] | EVET |
| s03 | EVET | 111/111 = 1.000 [0.967, 1.000] | 111/111 = 1.000 [0.967, 1.000] | 9/9 = 1.000 [0.701, 1.000] | 9/9 = 1.000 [0.701, 1.000] | 120/120 = 1.000 [0.969, 1.000] | 120/120 = 1.000 [0.969, 1.000] | EVET |
| s04 | EVET | 102/102 = 1.000 [0.964, 1.000] | 102/102 = 1.000 [0.964, 1.000] | 7/7 = 1.000 [0.646, 1.000] | 7/7 = 1.000 [0.646, 1.000] | 109/109 = 1.000 [0.966, 1.000] | 109/109 = 1.000 [0.966, 1.000] | EVET |
| s05 | hayir | 43/43 = 1.000 [0.918, 1.000] | 43/43 = 1.000 [0.918, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 53/53 = 1.000 [0.932, 1.000] | 53/53 = 1.000 [0.932, 1.000] | EVET |
| s06 | hayir | 110/110 = 1.000 [0.966, 1.000] | 110/110 = 1.000 [0.966, 1.000] | 44/44 = 1.000 [0.920, 1.000] | 44/44 = 1.000 [0.920, 1.000] | 154/154 = 1.000 [0.976, 1.000] | 154/154 = 1.000 [0.976, 1.000] | EVET |

**Tahmin (b) (s02, s03, s04'te D-ham >= 0.9, nokta tahmin): TUTUYOR** (3/3 oturum). s01 (egitim oturumu, rastgele toplama) dahil edilirse: TUTUYOR.

### (c) D-deb'de d50 D-ham'dan yuksek (B2; oturum basina; aralik karsilastirmasi)

| oturum | D-ham durum | d50_ham | D-deb durum | d50_deb | D-ham araligi | D-deb araligi | d50_deb > d50_ham? |
|---|---|---|---|---|---|---|---|
| s01 | ok | 0.053 | ok | 0.094 | [0.053, 0.053] | [0.094, 0.094] | EVET (yuksek) |
| s02 | tam ayrisma (tahmin edilemez) | nan | ok | -6.255 | [0.032, 0.350] | [0.000, 0.032] | belirsiz (araliklar ortusuyor) |
| s03 | tumu tespit (d50 < min sure) | nan | tumu tespit (d50 < min sure) | nan | [0.000, 0.317] | [0.000, 0.317] | belirsiz (araliklar ortusuyor) |
| s04 | tumu tespit (d50 < min sure) | nan | tumu tespit (d50 < min sure) | nan | [0.000, 0.286] | [0.000, 0.286] | belirsiz (araliklar ortusuyor) |
| s05 | ok | 0.117 | ok | 0.160 | [0.117, 0.117] | [0.160, 0.160] | EVET (yuksek) |
| s06 | tumu tespit (d50 < min sure) | nan | tumu tespit (d50 < min sure) | nan | [0.000, 0.477] | [0.000, 0.477] | belirsiz (araliklar ortusuyor) |

Ozet (c): EVET 2, HAYIR 0, belirsiz/esit 4 (6 oturumdan). **Tahmin (c) tum oturumlarda EVET: HAYIR.**

### (d) Kural: d50 > 0.4 s ise beyan edilen sinir gozlenen d90'a yukseltilir (yeniden ayar yok)

Yorum (Sapmalar'da kayitli): baslangic beyan siniri = on-yazili d50 araliginin ust ucu, 0.4 s; kural tetiklenirse (d50 > 0.4 s; aralik varsa alt ucun 0.4'u asmasi) sinir = o oturumun gozlenen d90'i; tam ayrismada d90 tanimsiz oldugundan sinir = aralik ust ucu (en kisa yakalanan dokunus) olarak, acikca etiketli. Tetiklenmezse sinir 0.4 s olarak kalir; ayrica d90 her durumda gosterilir.

| oturum | tanim | durum | d50_s | d90_s | d50>0.4? | beyan edilen sinir_s |
|---|---|---|---|---|---|---|
| s01 | D-ham | ok | 0.053 | 0.487 | hayir | 0.400 (kural tetiklenmedi) |
| s01 | D-deb | ok | 0.094 | 0.533 | hayir | 0.400 (kural tetiklenmedi) |
| s02 | D-ham | tam ayrisma (tahmin edilemez) | nan | nan | hayir (araliktan) | 0.400 (kural tetiklenmedi) |
| s02 | D-deb | ok | -6.255 | -2.764 | hayir (araliktan) | 0.400 (kural tetiklenmedi) |
| s03 | D-ham | tumu tespit (d50 < min sure) | nan | nan | hayir (araliktan) | 0.400 (kural tetiklenmedi) |
| s03 | D-deb | tumu tespit (d50 < min sure) | nan | nan | hayir (araliktan) | 0.400 (kural tetiklenmedi) |
| s04 | D-ham | tumu tespit (d50 < min sure) | nan | nan | hayir (araliktan) | 0.400 (kural tetiklenmedi) |
| s04 | D-deb | tumu tespit (d50 < min sure) | nan | nan | hayir (araliktan) | 0.400 (kural tetiklenmedi) |
| s05 | D-ham | ok | 0.117 | 0.265 | hayir | 0.400 (kural tetiklenmedi) |
| s05 | D-deb | ok | 0.160 | 0.326 | hayir | 0.400 (kural tetiklenmedi) |
| s06 | D-ham | tumu tespit (d50 < min sure) | nan | nan | belirsiz | belirsiz |
| s06 | D-deb | tumu tespit (d50 < min sure) | nan | nan | belirsiz | belirsiz |

Betimleyici ozet (on-kayitli degil, havuzlama degil; yalniz gozlenen sure araliginda kalan, ekstrapolasyonsuz d90 degerleri) = s01 D-ham: 0.487 s; s01 D-deb: 0.533 s; s05 D-ham: 0.265 s; s05 D-deb: 0.326 s; en buyuk = 0.533 s.

## 5. Ikincil modeller (CNN-GRU, CNN-only, B1): tespit orani (oturum x kutu; Wilson)

### CNN-GRU - D-ham

| oturum | <0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1.0 | >1.0 |
|---|---|---|---|---|---|
| s01 | 3/8 = 0.375 [0.137, 0.694] | 15/22 = 0.682 [0.473, 0.836] | 78/106 = 0.736 [0.645, 0.810] | 162/166 = 0.976 [0.940, 0.991] | 26/26 = 1.000 [0.871, 1.000] |
| s02 | 0/1 = 0.000 [0.000, 0.793] | - | 36/36 = 1.000 [0.904, 1.000] | 105/105 = 1.000 [0.965, 1.000] | 12/13 = 0.923 [0.667, 0.986] |
| s03 | - | - | 50/51 = 0.980 [0.897, 0.997] | 110/111 = 0.991 [0.951, 0.998] | 9/9 = 1.000 [0.701, 1.000] |
| s04 | - | 1/1 = 1.000 [0.207, 1.000] | 37/38 = 0.974 [0.865, 0.995] | 101/102 = 0.990 [0.947, 0.998] | 7/7 = 1.000 [0.646, 1.000] |
| s05 | 4/9 = 0.444 [0.189, 0.733] | 26/28 = 0.929 [0.774, 0.980] | 62/63 = 0.984 [0.915, 0.997] | 43/43 = 1.000 [0.918, 1.000] | 10/10 = 1.000 [0.722, 1.000] |
| s06 | - | - | 2/2 = 1.000 [0.342, 1.000] | 110/110 = 1.000 [0.966, 1.000] | 44/44 = 1.000 [0.920, 1.000] |

### CNN-GRU - D-deb

| oturum | <0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1.0 | >1.0 |
|---|---|---|---|---|---|
| s01 | 3/8 = 0.375 [0.137, 0.694] | 14/22 = 0.636 [0.430, 0.803] | 73/106 = 0.689 [0.595, 0.769] | 158/166 = 0.952 [0.908, 0.975] | 26/26 = 1.000 [0.871, 1.000] |
| s02 | 0/1 = 0.000 [0.000, 0.793] | - | 36/36 = 1.000 [0.904, 1.000] | 104/105 = 0.990 [0.948, 0.998] | 12/13 = 0.923 [0.667, 0.986] |
| s03 | - | - | 50/51 = 0.980 [0.897, 0.997] | 110/111 = 0.991 [0.951, 0.998] | 9/9 = 1.000 [0.701, 1.000] |
| s04 | - | 1/1 = 1.000 [0.207, 1.000] | 37/38 = 0.974 [0.865, 0.995] | 101/102 = 0.990 [0.947, 0.998] | 6/7 = 0.857 [0.487, 0.974] |
| s05 | 4/9 = 0.444 [0.189, 0.733] | 19/28 = 0.679 [0.493, 0.821] | 58/63 = 0.921 [0.827, 0.966] | 41/43 = 0.953 [0.845, 0.987] | 10/10 = 1.000 [0.722, 1.000] |
| s06 | - | - | 2/2 = 1.000 [0.342, 1.000] | 110/110 = 1.000 [0.966, 1.000] | 43/44 = 0.977 [0.882, 0.996] |

### CNN-only - D-ham

| oturum | <0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1.0 | >1.0 |
|---|---|---|---|---|---|
| s01 | 4/8 = 0.500 [0.215, 0.785] | 15/22 = 0.682 [0.473, 0.836] | 91/106 = 0.858 [0.780, 0.912] | 166/166 = 1.000 [0.977, 1.000] | 26/26 = 1.000 [0.871, 1.000] |
| s02 | 1/1 = 1.000 [0.207, 1.000] | - | 36/36 = 1.000 [0.904, 1.000] | 105/105 = 1.000 [0.965, 1.000] | 13/13 = 1.000 [0.772, 1.000] |
| s03 | - | - | 51/51 = 1.000 [0.930, 1.000] | 111/111 = 1.000 [0.967, 1.000] | 9/9 = 1.000 [0.701, 1.000] |
| s04 | - | 1/1 = 1.000 [0.207, 1.000] | 38/38 = 1.000 [0.908, 1.000] | 102/102 = 1.000 [0.964, 1.000] | 7/7 = 1.000 [0.646, 1.000] |
| s05 | 6/9 = 0.667 [0.354, 0.879] | 28/28 = 1.000 [0.879, 1.000] | 63/63 = 1.000 [0.943, 1.000] | 43/43 = 1.000 [0.918, 1.000] | 10/10 = 1.000 [0.722, 1.000] |
| s06 | - | - | 2/2 = 1.000 [0.342, 1.000] | 110/110 = 1.000 [0.966, 1.000] | 44/44 = 1.000 [0.920, 1.000] |

### CNN-only - D-deb

| oturum | <0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1.0 | >1.0 |
|---|---|---|---|---|---|
| s01 | 4/8 = 0.500 [0.215, 0.785] | 14/22 = 0.636 [0.430, 0.803] | 80/106 = 0.755 [0.665, 0.827] | 161/166 = 0.970 [0.931, 0.987] | 26/26 = 1.000 [0.871, 1.000] |
| s02 | 1/1 = 1.000 [0.207, 1.000] | - | 36/36 = 1.000 [0.904, 1.000] | 105/105 = 1.000 [0.965, 1.000] | 12/13 = 0.923 [0.667, 0.986] |
| s03 | - | - | 51/51 = 1.000 [0.930, 1.000] | 110/111 = 0.991 [0.951, 0.998] | 9/9 = 1.000 [0.701, 1.000] |
| s04 | - | 1/1 = 1.000 [0.207, 1.000] | 38/38 = 1.000 [0.908, 1.000] | 99/102 = 0.971 [0.917, 0.990] | 7/7 = 1.000 [0.646, 1.000] |
| s05 | 4/9 = 0.444 [0.189, 0.733] | 25/28 = 0.893 [0.728, 0.963] | 63/63 = 1.000 [0.943, 1.000] | 41/43 = 0.953 [0.845, 0.987] | 10/10 = 1.000 [0.722, 1.000] |
| s06 | - | - | 2/2 = 1.000 [0.342, 1.000] | 110/110 = 1.000 [0.966, 1.000] | 44/44 = 1.000 [0.920, 1.000] |

### B1 - D-ham

| oturum | <0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1.0 | >1.0 |
|---|---|---|---|---|---|
| s01 | 2/8 = 0.250 [0.071, 0.591] | 10/22 = 0.455 [0.269, 0.653] | 59/106 = 0.557 [0.462, 0.648] | 144/166 = 0.867 [0.807, 0.911] | 26/26 = 1.000 [0.871, 1.000] |
| s02 | 1/1 = 1.000 [0.207, 1.000] | - | 35/36 = 0.972 [0.858, 0.995] | 105/105 = 1.000 [0.965, 1.000] | 12/13 = 0.923 [0.667, 0.986] |
| s03 | - | - | 49/51 = 0.961 [0.868, 0.989] | 111/111 = 1.000 [0.967, 1.000] | 9/9 = 1.000 [0.701, 1.000] |
| s04 | - | 1/1 = 1.000 [0.207, 1.000] | 37/38 = 0.974 [0.865, 0.995] | 102/102 = 1.000 [0.964, 1.000] | 6/7 = 0.857 [0.487, 0.974] |
| s05 | 6/9 = 0.667 [0.354, 0.879] | 24/28 = 0.857 [0.685, 0.943] | 62/63 = 0.984 [0.915, 0.997] | 42/43 = 0.977 [0.879, 0.996] | 10/10 = 1.000 [0.722, 1.000] |
| s06 | - | - | 2/2 = 1.000 [0.342, 1.000] | 109/110 = 0.991 [0.950, 0.998] | 44/44 = 1.000 [0.920, 1.000] |

### B1 - D-deb

| oturum | <0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1.0 | >1.0 |
|---|---|---|---|---|---|
| s01 | 1/8 = 0.125 [0.022, 0.471] | 9/22 = 0.409 [0.233, 0.613] | 48/106 = 0.453 [0.361, 0.548] | 128/166 = 0.771 [0.701, 0.828] | 24/26 = 0.923 [0.759, 0.979] |
| s02 | 1/1 = 1.000 [0.207, 1.000] | - | 34/36 = 0.944 [0.819, 0.985] | 101/105 = 0.962 [0.906, 0.985] | 12/13 = 0.923 [0.667, 0.986] |
| s03 | - | - | 46/51 = 0.902 [0.790, 0.957] | 107/111 = 0.964 [0.911, 0.986] | 9/9 = 1.000 [0.701, 1.000] |
| s04 | - | 1/1 = 1.000 [0.207, 1.000] | 31/38 = 0.816 [0.666, 0.908] | 91/102 = 0.892 [0.817, 0.939] | 6/7 = 0.857 [0.487, 0.974] |
| s05 | 5/9 = 0.556 [0.267, 0.811] | 22/28 = 0.786 [0.605, 0.898] | 57/63 = 0.905 [0.807, 0.956] | 37/43 = 0.860 [0.727, 0.934] | 10/10 = 1.000 [0.722, 1.000] |
| s06 | - | - | 2/2 = 1.000 [0.342, 1.000] | 108/110 = 0.982 [0.936, 0.995] | 43/44 = 0.977 [0.882, 0.996] |

## 6. Egitim-etiketi uyarisinin sayisi: dokunus egitim/degerlendirme etiketinde (L0) ve 30 Hz oylamada (L3) gorunuyor mu?

`L0` = Faz A `y_true` (modellerin egitildigi etiket; +-3 kare mod), `L3` = 30 Hz oylama etiketi. Bir dokunus *L0'da gorunur* = kare kumesinde en az bir karenin L0 etiketi o kutu. Kare kumesi bos dokunuslar disarida. Hucre: gorunur/n (oran).

| oturum | bin | n | L0'da gorunur | L3'te gorunur | B2 D-ham |
|---|---|---|---|---|---|
| s01 | <0.2 | 8 | 1/8 (0.125) | 8/8 (1.000) | 5/8 (0.625) |
| s01 | 0.2-0.3 | 22 | 0/22 (0.000) | 22/22 (1.000) | 14/22 (0.636) |
| s01 | 0.3-0.5 | 106 | 68/106 (0.642) | 106/106 (1.000) | 88/106 (0.830) |
| s01 | 0.5-1.0 | 166 | 166/166 (1.000) | 166/166 (1.000) | 160/166 (0.964) |
| s01 | >1.0 | 26 | 26/26 (1.000) | 26/26 (1.000) | 26/26 (1.000) |
| s02 | <0.2 | 1 | 0/1 (0.000) | 1/1 (1.000) | 0/1 (0.000) |
| s02 | 0.3-0.5 | 36 | 36/36 (1.000) | 36/36 (1.000) | 36/36 (1.000) |
| s02 | 0.5-1.0 | 105 | 105/105 (1.000) | 105/105 (1.000) | 105/105 (1.000) |
| s02 | >1.0 | 13 | 13/13 (1.000) | 13/13 (1.000) | 13/13 (1.000) |
| s03 | 0.3-0.5 | 51 | 42/51 (0.824) | 51/51 (1.000) | 51/51 (1.000) |
| s03 | 0.5-1.0 | 111 | 111/111 (1.000) | 111/111 (1.000) | 111/111 (1.000) |
| s03 | >1.0 | 9 | 9/9 (1.000) | 9/9 (1.000) | 9/9 (1.000) |
| s04 | 0.2-0.3 | 1 | 0/1 (0.000) | 1/1 (1.000) | 1/1 (1.000) |
| s04 | 0.3-0.5 | 38 | 35/38 (0.921) | 38/38 (1.000) | 38/38 (1.000) |
| s04 | 0.5-1.0 | 102 | 102/102 (1.000) | 102/102 (1.000) | 102/102 (1.000) |
| s04 | >1.0 | 7 | 7/7 (1.000) | 7/7 (1.000) | 7/7 (1.000) |
| s05 | <0.2 | 9 | 2/9 (0.222) | 9/9 (1.000) | 4/9 (0.444) |
| s05 | 0.2-0.3 | 28 | 1/28 (0.036) | 28/28 (1.000) | 24/28 (0.857) |
| s05 | 0.3-0.5 | 63 | 48/63 (0.762) | 63/63 (1.000) | 63/63 (1.000) |
| s05 | 0.5-1.0 | 43 | 43/43 (1.000) | 43/43 (1.000) | 43/43 (1.000) |
| s05 | >1.0 | 10 | 10/10 (1.000) | 10/10 (1.000) | 10/10 (1.000) |
| s06 | 0.3-0.5 | 2 | 2/2 (1.000) | 2/2 (1.000) | 2/2 (1.000) |
| s06 | 0.5-1.0 | 110 | 110/110 (1.000) | 110/110 (1.000) | 110/110 (1.000) |
| s06 | >1.0 | 44 | 44/44 (1.000) | 44/44 (1.000) | 44/44 (1.000) |

- Tum oturumlar (yalniz bilgi, havuzlanmis ham oran): <0.2: n=18, L0'da gorunur 0.167, L3'te gorunur 1.000, B2 D-ham 0.500.
- Tum oturumlar (yalniz bilgi, havuzlanmis ham oran): 0.2-0.3: n=51, L0'da gorunur 0.020, L3'te gorunur 1.000, B2 D-ham 0.765.
- Tum oturumlar (yalniz bilgi, havuzlanmis ham oran): 0.3-0.5: n=296, L0'da gorunur 0.780, L3'te gorunur 1.000, B2 D-ham 0.939.
- Tum oturumlar (yalniz bilgi, havuzlanmis ham oran): 0.5-1.0: n=637, L0'da gorunur 1.000, L3'te gorunur 1.000, B2 D-ham 0.991.
- Tum oturumlar (yalniz bilgi, havuzlanmis ham oran): >1.0: n=109, L0'da gorunur 1.000, L3'te gorunur 1.000, B2 D-ham 1.000.

## 7. Konum ekseni (yalniz betimleyici): komsu kutu cifti ikili karisikligi vs kutu araligi

Kutu araligi = radar kare-merkezi (cx) medyan farki, `probe/01_dwell_modes.py` `adjacent_pairs` ile **ayni yontemle yeniden hesaplandi** (komsu = oturumda gorulen kutular arasinda cx sirasinda komsu; cift sayisi 14); `reports/probe/q1.md` degerleriyle uyum (3 ondalik): **TAMAM, 14/14**. Bu aralik bant boyunca radar-merkez ayrimidir, mezura ile olculmus mesafe degildir.
Karisiklik: `b2__S1` OOF'ta `y_true in {a, b}` satirlari; `y_pred` ucuncu bir sinif (none dahil) olan satirlar **disarida**; hata a->b = y_true=a, y_pred=b orani (kalan a satirlari icinde), b->a benzer; `sim_hata` = (a->b + b->a) / (kalan a + kalan b). Dogru etiket = OOF `y_true` (L0, modelin egitildigi etiket). Uydurulmus bir tasarim sabiti yoktur; cizgi = lateral cozunurluk 11.8 cm (r*dtheta = 0.45 m x 15 derece).

| oturum | cift | aralik_cm | aralik_lt_11_8cm | n_a | n_b | n_a_kalan | n_b_kalan | disarida_3_sinif | a_to_b | b_to_a | hata_a_to_b | hata_b_to_a | sim_hata | sep_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01 | 1-2 | 8.0282 | True | 293 | 287 | 270 | 263 | 47 | 0 | 3 | 0.0000 | 0.0114 | 0.0056 | 1.6456 |
| s01 | 2-3 | 12.8252 | False | 287 | 278 | 260 | 257 | 48 | 0 | 76 | 0.0000 | 0.2957 | 0.1470 | 2.7698 |
| s01 | 3-4 | 9.7956 | True | 278 | 224 | 183 | 209 | 110 | 2 | 3 | 0.0109 | 0.0144 | 0.0128 | 2.2784 |
| s01 | 4-5 | 8.5096 | True | 224 | 644 | 207 | 607 | 54 | 1 | 0 | 0.0048 | 0.0000 | 0.0012 | 2.2374 |
| s02 | 1-2 | 14.3230 | False | 372 | 378 | 368 | 354 | 28 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 5.6333 |
| s02 | 2-5 | 25.0767 | False | 378 | 262 | 354 | 246 | 40 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 5.5654 |
| s03 | 1-3 | 21.6881 | False | 427 | 381 | 416 | 362 | 30 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 7.1375 |
| s03 | 3-5 | 17.3339 | False | 381 | 221 | 362 | 211 | 29 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 7.5170 |
| s04 | 1-4 | 30.5112 | False | 356 | 334 | 333 | 317 | 40 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 7.0734 |
| s04 | 4-5 | 8.2492 | True | 334 | 212 | 319 | 192 | 35 | 2 | 0 | 0.0063 | 0.0000 | 0.0039 | 2.5130 |
| s05 | 1-2 | 14.2230 | False | 345 | 253 | 298 | 224 | 76 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 3.9999 |
| s05 | 2-5 | 24.7563 | False | 253 | 37 | 224 | 26 | 40 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 8.9154 |
| s06 | 1-2 | 14.1769 | False | 568 | 486 | 560 | 480 | 14 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 7.2251 |
| s06 | 2-5 | 25.1437 | False | 486 | 343 | 480 | 338 | 11 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 13.2142 |

Betimleyici: aralik < 11.8 cm olan 4 cift: sim_hata araligi [0.0012, 0.0128], medyan 0.0048; aralik >= 11.8 cm olan 10 cift: [0.0000, 0.1470], medyan 0.0000. (Fit yok; n=14 cift, cogu tek oturum; yalniz betimleyici.)
**s04 Box4-Box5 (ayri): aralik 8.2 cm; n_4=334, n_5=212; 4->5 hata 0.0063 (2/319), 5->4 hata 0.0000 (0/192); sim_hata 0.0039; ucuncu sinif nedeniyle disarida 35 satir.**

Sekiller: `02_envelope_detection.png` (oturum basina tespit ~ sure: B2, D-ham ve D-deb, Wilson %95, lojistik egri), `02_envelope_spatial.png` (komsu cift hatasi vs kutu araligi, 11.8 cm cizgisi).

## 7b. Ozet (yalniz yukaridaki tablolardan; yorum degil sayim)

- B2 D-ham, 0.5-1.0 s kutusu: oturumlar arasi en dusuk tespit 0.964 (160/166, s01; Wilson alt 0.923); tum oturumlarda >= 0.9: True.
- B2 D-ham, >1.0 s kutusu: oturumlar arasi en dusuk tespit 1.000 (26/26, s01; Wilson alt 0.871); tum oturumlarda >= 0.9: True.
- B2 D-deb, 0.5-1.0 s kutusu: oturumlar arasi en dusuk tespit 0.952 (158/166, s01; Wilson alt 0.908); tum oturumlarda >= 0.9: True.
- B2 D-deb, >1.0 s kutusu: oturumlar arasi en dusuk tespit 0.923 (12/13, s02; Wilson alt 0.667); tum oturumlarda >= 0.9: True.
- B2 D-ham, <0.2 s kutusu (n >= 8 olan oturumlar): s01 5/8 = 0.625; s05 4/9 = 0.444.
- B2 D-ham, 0.2-0.3 s kutusu (n >= 8 olan oturumlar): s01 14/22 = 0.636; s05 24/28 = 0.857.
- B2 D-ham, 0.3-0.5 s kutusu (n >= 8 olan oturumlar): s01 88/106 = 0.830; s02 36/36 = 1.000; s03 51/51 = 1.000; s04 38/38 = 1.000; s05 63/63 = 1.000.
- B2 D-deb, <0.2 s kutusu (n >= 8 olan oturumlar): s01 5/8 = 0.625; s05 3/9 = 0.333.
- B2 D-deb, 0.2-0.3 s kutusu (n >= 8 olan oturumlar): s01 13/22 = 0.591; s05 22/28 = 0.786.
- B2 D-deb, 0.3-0.5 s kutusu (n >= 8 olan oturumlar): s01 84/106 = 0.792; s02 36/36 = 1.000; s03 51/51 = 1.000; s04 38/38 = 1.000; s05 60/63 = 0.952.
- Kare kumesi bos dokunus: 0/1111 (paydadan dusulen dokunus yok; kural bu veride bagayici olmadi).
- Tahmin (a) d50 in [0.2, 0.4] (D-ham): 0/6 oturumda TUTAR; nokta d50 tahmin edilebilen oturumlar: s01 0.053 s, s05 0.117 s (ikisi de aralik **altinda**). Yani sistem, on-yazili tahminden **daha kisa** dokunuslari yakaliyor; tahmin (a) desteklenmedi.
- Egitim etiketi (L0) gorunurlugu vs B2 D-ham (tum oturumlar, yalniz bilgi): <0.2: L0 0.167 / B2 0.500; 0.2-0.3: L0 0.020 / B2 0.765; 0.3-0.5: L0 0.780 / B2 0.939; 0.5-1.0: L0 1.000 / B2 0.991; >1.0: L0 1.000 / B2 1.000. L0'da neredeyse hic gorunmeyen kisa dokunuslarin bir kismi yine de ham akista yakalaniyor (kare kumesindeki baska karelerin / L=5 pencerelerinin tasmasi ve 100 ms hizalama payi dahil; Bolum 8 uyarisi).
- Tahmin (b): TUTUYOR. Tahmin (c): EVET 2, HAYIR 0, belirsiz 4. Kural (d): d50 > 0.4 s olan oturum yok -> sinir d90'a yukseltilmedi (s06'da D-ham/D-deb tum dokunuslar yakalandi ve en kisa dokunus 0.477 s > 0.4 s oldugundan kural veriden degerlendirilemedi: 'belirsiz').
- Konum ekseni: aralik < 11.8 cm olan 4 cifte B2 sim. hata en fazla 0.0128; en yuksek hata aralik >= 11.8 cm bir cifte ait: s01 2-3 (aralik 12.8 cm, sim_hata 0.1470, 3->2 yonu 0.2957). Yani 14 ciftte hata kutu araligiyla monoton degil; 11.8 cm cizgisi B2'nin ikili karisikligini ayirmiyor. (Kosullar: S1 ayni oturumun diger dongulerinden ogrenir; ucuncu sinif/none tahminleri haric; Bolum 8.)
- Destekledigi cumle (on-yazili) ile karsilastirma: *>= d90 s suren dokunuslarda guvenilir* kismi verilerle tutarli (d90: s01 ~0.49 s D-ham / ~0.53 s D-deb; s05 ~0.27 s / ~0.33 s; diger oturumlarda d90 tahmin edilemedi cunku neredeyse hic kacirma yok; 0.5 s ustu kutularda B2 oranlari yukaridaki maddede). *>= ~12 cm aralikli kutularda* kismi B2'nin ikili karisikligiyle **desteklenmiyor**: aralik < 11.8 cm olan ciftlerde sim. hata en fazla %1.3.

## 8. On-kayitli uyarilar (brifteki gibi)

- Bu sinir **egitilmis sistemin** sinirdir, radarin degil: egitim etiketleri (L0, +-3 kare mod) kisa dokunuslari da silmisti (bkz. Bolum 6'daki L0-gorunurluk sayilari).
- L = 5 pencere (0.5 s) ve hizli tempo hareketleriyle karisik: kisa dokunus, penceredeki baska karelerin etkisiyle de kacirilabilir/yakalanabilir; sure ile hareket hizi ayrisamaz.
- Tek operator, tek mesafe (~0.45 m); kutu araligi radar-merkez farkidir; s01 (egitim oturumu) rastgele toplama icerir.
- **S1 kapsami:** OOF tahminleri 6 oturumun havuzunda dongu-gruplu 5-kat ile uretildi; her katin egitim kumesi ayni oturumlarin (ayni kutu yerlesimi) diger dongulerini icerir. Dolayisiyla hem tespit hem konum ekseni 'ayni yerlesimde, baska dongu' kosulunu olcer; yeni bir yerlesime/oturuma genelleme (S2/LOSO) degildir ve 8 cm ciftlerdeki dusuk karisiklik yerlesimin egitimde gorulmus olmasindan da gelebilir.
- Ikili karisiklik ucuncu sinif (none dahil) tahminlerini disarida birakir; none'a dusen/none'dan gelen hatalar (gecisler) bu olcude gorunmez, yani olcut yalniz kutu-kutu karisikligidir.
- Dogrusal-saniye lojistik uydurmasi kisa dokunuslarda plato yapan oranlari zayif temsil eder (orn. s01); d50/d90 yalniz oturum basina ozet olarak, kutu oranlariyla birlikte okunmalidir.
- Bolum 7 betimleyicidir: 14 cift, sabit/tasarim esigi uydurulmadi. Tespit tanimlari kare kumesi bos dokunuslari payda disi birakir (Bolum 0).
- Faz A birincil sonuc; bu analiz ikincil ve sonradan yapildi. Tum on-kayitli sayilar (olumsuz olanlar dahil) yukarida raporlanmistir.


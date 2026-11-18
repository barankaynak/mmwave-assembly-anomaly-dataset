# A4.1 (M1) Anomali turu oturumla karisik mi?

**Statu:** A4 ikincil analizdir (Faz A sayilari birincil kalir); on-kayit `faz_a4/DESIGN_A4.md` A4.1. Kaynak: `faz_a4/01_session_confound.py`. Yasak uyarisi: s02 dogal sapma dongulerinin alarm durumu ayrica incelenmedi; s02 normal Faz A FAR'i olarak butun kullanildi.

## A4.1(a) Baslangictan ONCE verilen alarmlar (B2 x pred, A3.3 literal tanim)

### 0. Yeniden uretim kontrolu (A3.3: baslangic anina esit/oncesi alarm 17/71 ve 17/114)

| atype | pre_onset | alarm_with_onset | expected | ok | n_pairs | n_alarm_all | trig_kinds |
|---|---|---|---|---|---|---|---|
| seq135 | 17 | 71 | 17/71 | True | 280 | 71 | {'dwell': 9, 'trans_mid': 8} |
| seq145 | 17 | 114 | 17/114 | True | 240 | 114 | {'trans_mid': 12, 'dwell': 5} |

- belirsiz olay->ziyaret eslemesi sayisi: 0
- **Yeniden uretim: GECTI**

### 1. Ozet: alarm veren (dongu,f) ciftleri; baslangic = Box3/Box4'e ilk ToF girisi

| tur | n_cift | n_alarm | n_baslangic_oncesi_veya_esit | oran_alarm_icinde | n_kesin_once | n_esit | n_tekil_dongu | n_tekil_dongu_alarmli |
|---|---|---|---|---|---|---|---|---|
| seq135 | 280 | 71 | 17 | 0.239 | 9 | 8 | 10 | 56 |
| seq145 | 240 | 114 | 17 | 0.149 | 9 | 8 | 7 | 48 |

### 2. Baslangic oncesi/esit alarmlar: tetikleyen olay x alarm anindaki B2 tahmini (ham y_pred)

**seq135** (n=17)

| kind | trig_boxes | pred=0 | pred=3 |
|---|---|---|---|
| dwell | 1 | 9 | 0 |
| trans_mid | 1->3 | 0 | 8 |

**seq145** (n=17)

| kind | trig_boxes | pred=0 | pred=4 |
|---|---|---|---|
| dwell | 1 | 5 | 0 |
| trans_mid | 1->4 | 2 | 10 |

### 3. Iki aciklamanin ayrimi (kural: (i) alarm anindaki B2 durumu = anomali kutusu; (ii) degilse ve tetik Box1 dwell ya da Box1->X; (iii) diger)

BIRINCIL tanim = alarm mantiginin gordugu debounce'lu B2 durumu (`cat`); duyarlilik = ham y_pred (`cat_raw`). Fark: seq145'te 2 alarm (1->4 gecisi, ham tahmin alarm karesinde 0, debounce'lu 4).

| durum_tanimi | tur | kategori | n | oran | n_kesin_once | n_esit | n_tekil_dongu |
|---|---|---|---|---|---|---|---|
| debounce'lu (BIRINCIL) | seq135 | (i) radar yanlis kutuyu gordu (tahmin = anomali kutusu) | 8 | 0.471 | 0 | 8 | 8 |
| debounce'lu (BIRINCIL) | seq135 | (ii) oturuma ozgu zamanlama (Box1 dwell / Box1->X) | 9 | 0.529 | 9 | 0 | 2 |
| debounce'lu (BIRINCIL) | seq135 | TOPLAM | 17 | 1.000 | 9 | 8 | 10 |
| debounce'lu (BIRINCIL) | seq145 | (i) radar yanlis kutuyu gordu (tahmin = anomali kutusu) | 12 | 0.706 | 4 | 8 | 6 |
| debounce'lu (BIRINCIL) | seq145 | (ii) oturuma ozgu zamanlama (Box1 dwell / Box1->X) | 5 | 0.294 | 5 | 0 | 1 |
| debounce'lu (BIRINCIL) | seq145 | TOPLAM | 17 | 1.000 | 9 | 8 | 7 |
| debounce'lu (BIRINCIL) | ikisi | (i) radar yanlis kutuyu gordu (tahmin = anomali kutusu) | 20 | 0.588 | 4 | 16 | 14 |
| debounce'lu (BIRINCIL) | ikisi | (ii) oturuma ozgu zamanlama (Box1 dwell / Box1->X) | 14 | 0.412 | 14 | 0 | 3 |
| debounce'lu (BIRINCIL) | ikisi | TOPLAM | 34 | 1.000 | 18 | 16 | 17 |
| ham y_pred (duyarlilik) | seq135 | (i) radar yanlis kutuyu gordu (tahmin = anomali kutusu) | 8 | 0.471 | 0 | 8 | 8 |
| ham y_pred (duyarlilik) | seq135 | (ii) oturuma ozgu zamanlama (Box1 dwell / Box1->X) | 9 | 0.529 | 9 | 0 | 2 |
| ham y_pred (duyarlilik) | seq135 | TOPLAM | 17 | 1.000 | 9 | 8 | 10 |
| ham y_pred (duyarlilik) | seq145 | (i) radar yanlis kutuyu gordu (tahmin = anomali kutusu) | 10 | 0.588 | 4 | 6 | 5 |
| ham y_pred (duyarlilik) | seq145 | (ii) oturuma ozgu zamanlama (Box1 dwell / Box1->X) | 7 | 0.412 | 5 | 2 | 2 |
| ham y_pred (duyarlilik) | seq145 | TOPLAM | 17 | 1.000 | 9 | 8 | 7 |
| ham y_pred (duyarlilik) | ikisi | (i) radar yanlis kutuyu gordu (tahmin = anomali kutusu) | 18 | 0.529 | 4 | 14 | 13 |
| ham y_pred (duyarlilik) | ikisi | (ii) oturuma ozgu zamanlama (Box1 dwell / Box1->X) | 16 | 0.471 | 14 | 2 | 4 |
| ham y_pred (duyarlilik) | ikisi | TOPLAM | 34 | 1.000 | 18 | 16 | 17 |

Duyarlilik sayilari (ayni 34 alarm): ham tahmin = anomali kutusu: 18; ham tahmin in {3,4}: 18; debounce'lu tahmin = anomali kutusu: 20; ToF (oracle) durumu alarm karesinde = anomali kutusu: 16.

(ii) alarmlarini veren tekil donguler (Box1 dwell'i ToF 30 Hz'den; s02 Box1-dwell dagilimindaki yuzdelik konumuyla):

Alarm karesi ile baslangic arasindaki fark (kare; 10 Hz, pozitif = alarm baslangictan once):

| atype | count | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|
| seq135 | 17.000 | 0.000 | 0.000 | 4.000 | 4.000 | 4.000 |
| seq145 | 17.000 | 0.000 | 0.000 | 1.000 | 8.000 | 8.000 |

### 4. Baglam: tum alarm veren ciftler (baslangic oncesi + sonrasi) - kategori dagilimi

| tur | zaman | kategori | n |
|---|---|---|---|
| seq135 | baslangic oncesi/esit | (i) radar yanlis kutuyu gordu (tahmin = anomali kutusu) | 8 |
| seq135 | baslangic oncesi/esit | (ii) oturuma ozgu zamanlama (Box1 dwell / Box1->X) | 9 |
| seq135 | baslangic sonrasi | (i) radar yanlis kutuyu gordu (tahmin = anomali kutusu) | 47 |
| seq135 | baslangic sonrasi | (iii) diger | 7 |
| seq145 | baslangic oncesi/esit | (i) radar yanlis kutuyu gordu (tahmin = anomali kutusu) | 12 |
| seq145 | baslangic oncesi/esit | (ii) oturuma ozgu zamanlama (Box1 dwell / Box1->X) | 5 |
| seq145 | baslangic sonrasi | (i) radar yanlis kutuyu gordu (tahmin = anomali kutusu) | 82 |
| seq145 | baslangic sonrasi | (iii) diger | 15 |

Referans (A3.3, ToF x tof oracle): baslangic anina esit/oncesi alarm 4/285 (seq135) ve 0/245 (seq145); oracle etiket ToF esiginden once degisemez.

## A4.1(b) Karsi kanal = negatif kontrol (B2 x pred, Faz A standart p; role=test)

Birim: (dongu, f) cifti (anomali donguleri her f'de test; normal s02 donguleri tek f'de). Alarm: kanal p <= alpha (+1e-9), p = Faz A standart conformal. Wilson GA havuzlu ciftler uzerinde (anomali donguleri 5 f'de tekrar ettigi icin dar kalir); bootstrap = dongu-kumeli (kume = oturum#dongu, tum f'leri birlikte), B=10000, tohum 20261105, yuzdelik %95.

Grup buyuklukleri (cift / tekil dongu): s02 (normal): 51/51; s03 (seq135): 280/56; s04 (seq145): 240/48; s05 (fast): 240/48; s06 (slow): 255/51; seq (s03+s04): 520/104; tempo (s05+s06): 495/99

### Tum kanal x grup oranlari (oran [Wilson GA] / [kume-bootstrap GA])

**D1** (karsi-kanal hucreleri: tempo oturumlari)

| grup | n_cift/n_dongu | alpha=0.10 | alpha=0.05 |
|---|---|---|---|
| s02 (normal) | 51/51 | 0.059 [0.020, 0.159] / [0.000, 0.137] | 0.020 [0.003, 0.103] / [0.000, 0.059] |
| s03 (seq135) | 280/56 | 0.604 [0.545, 0.659] / [0.600, 0.611] | 0.211 [0.167, 0.262] / [0.200, 0.232] |
| s04 (seq145) | 240/48 | 0.800 [0.745, 0.846] / [0.800, 0.800] | 0.400 [0.340, 0.463] / [0.400, 0.400] |
| s05 (fast) | 240/48 | 0.217 [0.169, 0.273] / [0.117, 0.317] | 0.217 [0.169, 0.273] / [0.117, 0.317] |
| s06 (slow) | 255/51 | 0.008 [0.002, 0.028] / [0.000, 0.024] | 0.000 [0.000, 0.015] / [0.000, 0.000] |
| seq (s03+s04) | 520/104 | 0.694 [0.653, 0.732] / [0.675, 0.713] | 0.298 [0.260, 0.339] / [0.277, 0.319] |
| tempo (s05+s06) | 495/99 | 0.109 [0.085, 0.140] / [0.061, 0.166] | 0.105 [0.081, 0.135] / [0.057, 0.162] |

**D2** (karsi-kanal hucreleri: sira oturumlari)

| grup | n_cift/n_dongu | alpha=0.10 | alpha=0.05 |
|---|---|---|---|
| s02 (normal) | 51/51 | 0.078 [0.031, 0.185] / [0.020, 0.157] | 0.059 [0.020, 0.159] / [0.000, 0.137] |
| s03 (seq135) | 280/56 | 0.068 [0.044, 0.104] / [0.014, 0.139] | 0.054 [0.033, 0.086] / [0.004, 0.118] |
| s04 (seq145) | 240/48 | 0.125 [0.089, 0.173] / [0.042, 0.229] | 0.125 [0.089, 0.173] / [0.042, 0.229] |
| s05 (fast) | 240/48 | 0.800 [0.745, 0.846] / [0.692, 0.900] | 0.771 [0.714, 0.819] / [0.654, 0.883] |
| s06 (slow) | 255/51 | 0.992 [0.972, 0.998] / [0.980, 1.000] | 0.929 [0.891, 0.955] / [0.875, 0.976] |
| seq (s03+s04) | 520/104 | 0.094 [0.072, 0.122] / [0.038, 0.152] | 0.087 [0.065, 0.114] / [0.038, 0.142] |
| tempo (s05+s06) | 495/99 | 0.899 [0.869, 0.923] / [0.840, 0.949] | 0.853 [0.819, 0.881] / [0.786, 0.913] |

**D3** (karsi-kanal hucreleri: yok; referans)

| grup | n_cift/n_dongu | alpha=0.10 | alpha=0.05 |
|---|---|---|---|
| s02 (normal) | 51/51 | 0.078 [0.031, 0.185] / [0.020, 0.157] | 0.000 [0.000, 0.070] / [0.000, 0.000] |
| s03 (seq135) | 280/56 | 0.254 [0.206, 0.308] / [0.211, 0.311] | 0.000 [0.000, 0.014] / [0.000, 0.000] |
| s04 (seq145) | 240/48 | 0.475 [0.413, 0.538] / [0.425, 0.537] | 0.000 [0.000, 0.016] / [0.000, 0.000] |
| s05 (fast) | 240/48 | 0.921 [0.880, 0.949] / [0.858, 0.967] | 0.000 [0.000, 0.016] / [0.000, 0.000] |
| s06 (slow) | 255/51 | 0.929 [0.891, 0.955] / [0.875, 0.976] | 0.000 [0.000, 0.015] / [0.000, 0.000] |
| seq (s03+s04) | 520/104 | 0.356 [0.316, 0.398] / [0.315, 0.400] | 0.000 [0.000, 0.007] / [0.000, 0.000] |
| tempo (s05+s06) | 495/99 | 0.925 [0.899, 0.945] / [0.885, 0.960] | 0.000 [0.000, 0.008] / [0.000, 0.000] |

### Karsi kanal karari (on-yazili okuma: oran normal FAR'in Wilson GA'si ICINDEYSE 'saptanamadi', DISINDAYSA 'oturum etkisi var' + buyukluk)

'Icinde' = karsi kanal oraninin nokta tahmini, ayni kanalin s02 normal FAR'inin Wilson %95 GA'si icinde (n_normal = 51). Ikincil: karsi kanal bootstrap GA'si ile normal Wilson GA'sinin ortusmesi; fark = oran - normal FAR, kume-bootstrap GA.

| kanal | alpha | grup | oran [boot GA] | normal FAR [Wilson] | fark pp [boot GA] | oran_normal_GA_icinde | GA_ortusuyor | hukum |
|---|---|---|---|---|---|---|---|---|
| D1 | 0.100 | s05 (fast) | 0.217 [0.117, 0.317] | 0.059 [0.020, 0.159] | +15.8 [+3.8, +27.7] | False | True | oturum etkisi VAR (normal FAR GA'sinin ustunde: alarm fazlasi) |
| D1 | 0.100 | s06 (slow) | 0.008 [0.000, 0.024] | 0.059 [0.020, 0.159] | -5.1 [-12.2, +0.8] | False | True | oturum etkisi VAR (normal FAR GA'sinin ALTINDA: daha az alarm; yanlis alarm sisirmesi degil) |
| D1 | 0.100 | tempo (s05+s06) | 0.109 [0.061, 0.166] | 0.059 [0.020, 0.159] | +5.0 [-3.7, +13.0] | True | True | oturum etkisi bu kanalda saptanamadi |
| D1 | 0.050 | s05 (fast) | 0.217 [0.117, 0.317] | 0.020 [0.003, 0.103] | +19.7 [+9.4, +30.0] | False | False | oturum etkisi VAR (normal FAR GA'sinin ustunde: alarm fazlasi) |
| D1 | 0.050 | s06 (slow) | 0.000 [0.000, 0.000] | 0.020 [0.003, 0.103] | -2.0 [-5.9, +0.0] | False | False | oturum etkisi VAR (normal FAR GA'sinin ALTINDA: daha az alarm; yanlis alarm sisirmesi degil) |
| D1 | 0.050 | tempo (s05+s06) | 0.105 [0.057, 0.162] | 0.020 [0.003, 0.103] | +8.5 [+2.2, +15.0] | False | True | oturum etkisi VAR (normal FAR GA'sinin ustunde: alarm fazlasi) |
| D2 | 0.100 | s03 (seq135) | 0.068 [0.014, 0.139] | 0.078 [0.031, 0.185] | -1.1 [-10.7, +8.4] | True | True | oturum etkisi bu kanalda saptanamadi |
| D2 | 0.100 | s04 (seq145) | 0.125 [0.042, 0.229] | 0.078 [0.031, 0.185] | +4.7 [-7.5, +16.9] | True | True | oturum etkisi bu kanalda saptanamadi |
| D2 | 0.100 | seq (s03+s04) | 0.094 [0.038, 0.152] | 0.078 [0.031, 0.185] | +1.6 [-8.0, +10.5] | True | True | oturum etkisi bu kanalda saptanamadi |
| D2 | 0.050 | s03 (seq135) | 0.054 [0.004, 0.118] | 0.059 [0.020, 0.159] | -0.5 [-9.4, +7.9] | True | True | oturum etkisi bu kanalda saptanamadi |
| D2 | 0.050 | s04 (seq145) | 0.125 [0.042, 0.229] | 0.059 [0.020, 0.159] | +6.6 [-5.3, +18.9] | True | True | oturum etkisi bu kanalda saptanamadi |
| D2 | 0.050 | seq (s03+s04) | 0.087 [0.038, 0.142] | 0.059 [0.020, 0.159] | +2.8 [-6.0, +10.8] | True | True | oturum etkisi bu kanalda saptanamadi |

Not (A4.2'ye atif): s05'teki (fast) D1 orani Box5 silinmesiyle (kisalmis dizi) kirlenmistir; temiz tempo-D1 kontrolu s06 (slow) satiridir. 'tempo (s05+s06)' havuzu s05 kirliligini icerir.

## A4.1(c) Oturum basina adim sureleri (30 Hz ToF segmentleri; betimleyici)

Hucre = medyan [Q25, Q75], saniye. Adim sureleri yalniz ToF dizisi tam beklenen diziyle (s02, s05, s06: 1-2-5; s03: 1-3-5; s04: 1-4-5) eslesen donguler icin; dongu suresi tum donguler icin (Faz A ToF dongu siniri, kare tablosu, 10 Hz cozunurluk). Gecis = onceki kutu segmentinin sonu ile sonraki kutu segmentinin basi arasi (ToF debounce: 0.1 s / 0.15 s).

| oturum | n_dongu | n_beklenen_dizi | Box1 kalisi | Box1->sonraki kutu gecisi | ikinci kutu kalisi | ->Box5 gecisi | Box5 kalisi | dongu suresi |
|---|---|---|---|---|---|---|---|---|
| session_02(normal) | 51 | 50 | 0.63 [0.60, 0.85] | 0.57 [0.54, 0.61] | 0.65 [0.57, 0.85] | 1.84 [1.69, 2.06] | 0.48 [0.44, 0.51] | 5.00 [4.55, 5.25] |
| session_03(seq135) | 57 | 57 | 0.66 [0.60, 0.79] | 0.48 [0.45, 0.54] | 0.60 [0.54, 0.76] | 1.69 [1.59, 1.91] | 0.45 [0.41, 0.48] | 4.60 [4.40, 5.00] |
| session_04(seq145) | 49 | 48 | 0.70 [0.60, 0.80] | 0.45 [0.41, 0.49] | 0.60 [0.54, 0.79] | 1.72 [1.56, 1.94] | 0.45 [0.41, 0.51] | 4.50 [4.20, 4.90] |
| session_05(fast) | 49 | 47 | 0.57 [0.48, 0.85] | 0.38 [0.32, 0.41] | 0.48 [0.41, 0.53] | 1.50 [1.32, 1.70] | 0.29 [0.25, 0.32] | 3.80 [3.50, 4.20] |
| session_06(slow) | 52 | 52 | 1.03 [0.95, 1.23] | 1.40 [1.23, 1.56] | 0.86 [0.76, 0.99] | 2.72 [2.55, 3.05] | 0.67 [0.57, 0.74] | 8.30 [7.48, 8.70] |

s02'ye gore medyan orani (oturum medyani / s02 medyani):

| oturum | Box1 kalisi | Box1->sonraki kutu gecisi | ikinci kutu kalisi | ->Box5 gecisi | Box5 kalisi | dongu suresi |
|---|---|---|---|---|---|---|
| session_02(normal) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| session_03(seq135) | 1.05 | 0.83 | 0.93 | 0.91 | 0.93 | 0.92 |
| session_04(seq145) | 1.10 | 0.78 | 0.93 | 0.93 | 0.93 | 0.90 |
| session_05(fast) | 0.90 | 0.67 | 0.73 | 0.81 | 0.60 | 0.76 |
| session_06(slow) | 1.62 | 2.45 | 1.32 | 1.47 | 1.40 | 1.66 |

Dizisi beklenenle eslesmeyen donguler (sayim, oturum basina): session_02: 1; session_03: 0; session_04: 1; session_05: 2; session_06: 0

(a) -> (c) baglantisi: (ii) kategorisindeki alarmlari veren tekil donguler ve Box1 kalis sureleri

| tur | oturum | dongu | n_alarm_cifti | tetik | tetik_ziyaret_kutusu | B2_pred_dwell_s | s02_pred_yuzdelik | s02_pred_aralik | ToF_box1_dwell_s | s02_ToF_yuzdelik | s02_ToF_aralik | kendi_oturum_ToF_medyan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seq135 | session_03 | 24 | 4 | dwell | 1 | 1.801 | 100.000 | [0.50, 1.30] | 1.173 | 96.000 | [0.51, 1.43] | 0.665 |
| seq135 | session_03 | 27 | 5 | dwell | 1 | 0.300 | 0.000 | [0.50, 1.30] | 0.633 | 44.000 | [0.51, 1.43] | 0.665 |
| seq145 | session_04 | 6 | 5 | dwell | 1 | 0.300 | 0.000 | [0.50, 1.30] | 0.697 | 62.000 | [0.51, 1.43] | 0.697 |

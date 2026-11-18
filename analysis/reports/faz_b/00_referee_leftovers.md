# B.0 - Hakem raporunun kalan maddeleri (e, f, g)

`faz_b/00_referee_leftovers.py` uretir; plan `faz_b/DESIGN_B.md` B.0. Statu: KESIF AMACLI; kullanici gormeden makaleye girmez. B.0 icin onceden yazilmis sayisal beklenti yoktur; yalnizca (g)'deki 'onceden yazilmis okuma' ve '~0,15' cumlesi vardir (asagida denetlenir). Her sayi stdout'a da yazildi. Sapmalar/ek ayrintilar: DESIGN_B.md 'Sapmalar' satirinda (tek satir).

## (e) D0 + D1 kombinasyonu (B2 x pred)

Protokol: Faz A `04_cycle_scores.csv.gz`, role=test satirlari (B2 x pred); D01 = min(1, 2 min(p0, p1)) (Bonferroni); D3 = min(1, 2 min(p1, p2)) (Faz A kolonu p3). Kalibrasyon sayisi f basina n_cal = {0: 19, 1: 20, 2: 21, 3: 22, 4: 20}; ulasilabilir en kucuk p = 1/(n_cal+1) = [0.05, 0.0476, 0.0455, 0.0435, 0.0476]; ulasilabilir en kucuk Bonferroni p = [0.1, 0.0952, 0.0909, 0.087, 0.0952]. **alpha=0,05'te iki Bonferroni kurali da yapisal olarak alarm veremez** (en kucuk p ~ 0,095 > 0,05); bu, Faz A D3'te de boyleydi (det_a05 = 0).

Normal test cycles (havuzlanmis, her biri tek f'de test): n = 51. Benzersiz anomali dongusu: {'seq135': 56, 'seq145': 48, 'fast': 48, 'slow': 51}.

Dogrulama (kapi): D3 tum-anomali AUROC Faz A tablosu 0.9237 vs burada 0.9237; det@0,10 0.6335 vs 0.6335 -> ESLESTI.

### (e) FAR (normal test dongusu, n = 51; Wilson %95)

| kural | FAR @ 0,10 | FAR @ 0,05 |
|---|---|---|
| D3 (D1+D2) | 0.078 [0.031, 0.185] (4/51) | 0.000 [0.000, 0.070] (0/51) |
| D01 (D0+D1) | 0.078 [0.031, 0.185] (4/51) | 0.000 [0.000, 0.070] (0/51) |
| D0 | 0.098 [0.043, 0.210] (5/51) | 0.059 [0.020, 0.159] (3/51) |
| D1 | 0.059 [0.020, 0.159] (3/51) | 0.020 [0.003, 0.103] (1/51) |

### (e) Tur basina tespit (alpha = 0,10; Wilson %95, n = benzersiz anomali dongusu, k = oran x n; ayni dongu 5 f'de tekrar eder, Wilson bu yuzden iyimser olabilir)

| kural | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|
| D3 (D1+D2) | 0.254 [0.158, 0.381] | 0.475 [0.341, 0.613] | 0.921 [0.810, 0.969] | 0.929 [0.825, 0.973] | 0.356 [0.270, 0.451] | 0.925 [0.856, 0.963] | 0.633 [0.565, 0.697] |
| D01 (D0+D1) | 0.332 [0.223, 0.463] | 0.517 [0.379, 0.651] | 0.700 [0.560, 0.811] | 1.000 [0.930, 1.000] | 0.417 [0.327, 0.513] | 0.855 [0.772, 0.911] | 0.631 [0.562, 0.694] |
| D0 | 0.243 [0.149, 0.369] | 0.308 [0.196, 0.449] | 0.787 [0.653, 0.880] | 1.000 [0.930, 1.000] | 0.273 [0.197, 0.366] | 0.897 [0.822, 0.943] | 0.577 [0.509, 0.643] |
| D1 | 0.604 [0.473, 0.721] | 0.800 [0.667, 0.889] | 0.217 [0.124, 0.352] | 0.008 [0.001, 0.084] | 0.694 [0.600, 0.775] | 0.109 [0.062, 0.186] | 0.409 [0.344, 0.478] |

### (e) Tur basina tespit (alpha = 0,05)

| kural | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|
| D3 (D1+D2) | 0.000 [0.000, 0.064] | 0.000 [0.000, 0.074] | 0.000 [0.000, 0.074] | 0.000 [0.000, 0.070] | 0.000 [0.000, 0.036] | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.019] |
| D01 (D0+D1) | 0.000 [0.000, 0.064] | 0.000 [0.000, 0.074] | 0.000 [0.000, 0.074] | 0.000 [0.000, 0.070] | 0.000 [0.000, 0.036] | 0.000 [0.000, 0.037] | 0.000 [0.000, 0.019] |
| D0 | 0.168 [0.092, 0.286] | 0.183 [0.099, 0.315] | 0.662 [0.521, 0.780] | 1.000 [0.930, 1.000] | 0.175 [0.114, 0.259] | 0.836 [0.751, 0.896] | 0.498 [0.429, 0.566] |
| D1 | 0.211 [0.124, 0.334] | 0.400 [0.274, 0.541] | 0.217 [0.124, 0.352] | 0.000 [0.000, 0.070] | 0.298 [0.219, 0.392] | 0.105 [0.059, 0.181] | 0.204 [0.154, 0.265] |

### (e) AUROC (kat basina, ortalama +- sd; skor = 1 - p; normal vs ilgili anomali grubu)

| kural | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|
| D3 (D1+D2) | 0.879 +- 0.078 | 0.918 +- 0.046 | 0.947 +- 0.029 | 0.956 +- 0.022 | 0.897 +- 0.051 | 0.952 +- 0.025 | 0.924 +- 0.035 |
| D01 (D0+D1) | 0.896 +- 0.075 | 0.901 +- 0.071 | 0.872 +- 0.024 | 0.962 +- 0.039 | 0.898 +- 0.073 | 0.918 +- 0.030 | 0.908 +- 0.048 |
| D0 | 0.592 +- 0.095 | 0.631 +- 0.075 | 0.884 +- 0.037 | 0.973 +- 0.041 | 0.610 +- 0.085 | 0.930 +- 0.034 | 0.766 +- 0.052 |
| D1 | 0.961 +- 0.039 | 0.961 +- 0.039 | 0.601 +- 0.032 | 0.470 +- 0.039 | 0.961 +- 0.039 | 0.534 +- 0.035 | 0.753 +- 0.037 |

**Okuma (e):** D01 (D0+D1) ve D3 (D1+D2) yan yana yukarida. tum-anomali AUROC: D3 0.924, D01 0.908; seq(135+145) AUROC: D3 0.897, D01 0.898; tempo(fast+slow) AUROC: D3 0.952, D01 0.918.

## (f) Eksik kare / bosluklu pencereler

Tanimlar: (i) **frame_id boslugu** = `windows()['contiguous'] == False` (son-ilk frame_id farki != 4). (ii) **zaman boslugu** = pencere icinde ardisik kareler arasi dt > 0,15 s (nominal 0,10 s; dt >= ~0,2 s en az bir kayip kare yuvasi demektir). Cunku kayittaki `frame_id` ardisik bir sayacdir (kare tablosunda yalniz s05'te 2 bosluk var), ama `t` uzerinde cok sayida bosluk vardir: frame_id sayaci bos kareleri atlamis gorunuyor.

Kare dt dagilimi (tum oturumlar, 16266 aralik): medyan 0.1000 s; %1-%99 = [0.097, 0.600]; dt <= 0,12 s: 0.9569; (0,12-0,15]: 0.0003; (0,15-0,25]: 0.0162; > 0,25 s: 0.0266. Aralik sayisi: dt > 0,15 s = 696; dt > 0,25 s = 433; en buyuk dt = 8.40 s.

Pencere sayisi 16248. frame_id bosluklu: **3** (0.0185%); zaman bosluklu (dt > 0,15 s): **2114** (13.0108%); dt > 0,25 s: 1488 (9.1581%); ikisinden biri: 2116 (13.0231%). Bosluklu pencerelerde ortalama kayip kare yuvasi (zaman bosluklu olanlar): 5.13; pencere suresi medyan 0.400 s (bosluksuz 0.400 s).

| oturum | pencere | frame_id bosluklu | zaman bosluklu (>0,15 s) | dt>0,25 s | zaman bosluklu pay |
|---|---|---|---|---|---|
| session_01_training | 4679 | 0 | 438 | 256 | 9.361% |
| session_02_normal_seq125 | 2270 | 0 | 293 | 229 | 12.907% |
| session_03_normal_seq135 | 2403 | 0 | 322 | 238 | 13.400% |
| session_04_normal_seq145 | 1924 | 0 | 254 | 224 | 13.202% |
| session_05_anomaly_fast | 1776 | 3 | 238 | 134 | 13.401% |
| session_06_anomaly_slow | 3196 | 0 | 569 | 407 | 17.804% |

Son karenin sinifina gore zaman bosluklu pay: sinif 0: 21.714% (2073/9547); sinif 1: 0.593% (14/2361); sinif 2: 0.000% (0/1404); sinif 3: 0.000% (0/659); sinif 4: 0.000% (0/558); sinif 5: 1.571% (27/1719).

### (f) S1 metrikleri: bosluklu pencereler dahil / haric (B2, CNN-GRU, B1)

Havuzlanmis dogruluk ve macro-F1 (`fc.state_metrics`) ile kat basina macro-F1 ortalama +- sd (5 kat). 'haric-fid' = frame_id bosluklu pencereler cikarildi; 'haric-t' = zaman bosluklu (dt > 0,15 s) cikarildi; 'yalniz-t' = yalniz zaman bosluklu pencereler (kucuk altorneklem, yorumlama icin n'e bakin).

| model | altkume | n pencere | dogruluk | macro-F1 (havuz) | macro-F1 kat ort +- sd | dogr. kat ort |
|---|---|---|---|---|---|---|
| b2 | tum (A) | 16248 | 0.8840 | 0.8578 | 0.8564 +- 0.0122 | 0.8841 |
| b2 | haric-fid | 16245 | 0.8841 | 0.8579 | 0.8565 +- 0.0121 | 0.8842 |
| b2 | haric-t | 14134 | 0.8678 | 0.8533 | 0.8519 +- 0.0131 | 0.8679 |
| b2 | haric-fid+t | 14132 | 0.8680 | 0.8534 | 0.8520 +- 0.0129 | 0.8680 |
| b2 | yalniz-t | 2114 | 0.9920 | 0.9004 | 0.9098 +- 0.0749 | 0.9918 |
| cnngru | tum (A) | 16248 | 0.8692 | 0.8410 | 0.8408 +- 0.0143 | 0.8693 |
| cnngru | haric-fid | 16245 | 0.8692 | 0.8410 | 0.8408 +- 0.0143 | 0.8693 |
| cnngru | haric-t | 14134 | 0.8523 | 0.8370 | 0.8368 +- 0.0146 | 0.8523 |
| cnngru | haric-fid+t | 14132 | 0.8523 | 0.8370 | 0.8368 +- 0.0147 | 0.8523 |
| cnngru | yalniz-t | 2114 | 0.9825 | 0.8173 | 0.8804 +- 0.1035 | 0.9824 |
| b1 | tum (A) | 16248 | 0.6129 | 0.5271 | 0.5265 +- 0.0136 | 0.6130 |
| b1 | haric-fid | 16245 | 0.6130 | 0.5271 | 0.5266 +- 0.0136 | 0.6130 |
| b1 | haric-t | 14134 | 0.5976 | 0.5431 | 0.5427 +- 0.0151 | 0.5976 |
| b1 | haric-fid+t | 14132 | 0.5976 | 0.5431 | 0.5427 +- 0.0151 | 0.5976 |
| b1 | yalniz-t | 2114 | 0.7157 | 0.3984 | 0.4833 +- 0.0807 | 0.7166 |

Dogrulama (kapi): 'tum' satirlari Faz A `03_state_table.csv` (S1, tum) kat-ortalama macro-F1 ile ESLESTI (kat ortalamasi birebir).

**Okuma (f):** b2: haric-t - tum = -0.0045 (kat ort macro-F1), dogruluk -0.0161; bosluklu altkumede dogruluk 0.9920 (n=2114) vs bosluksuz 0.8678; cnngru: haric-t - tum = -0.0040 (kat ort macro-F1), dogruluk -0.0169; bosluklu altkumede dogruluk 0.9825 (n=2114) vs bosluksuz 0.8523; b1: haric-t - tum = +0.0161 (kat ort macro-F1), dogruluk -0.0154; bosluklu altkumede dogruluk 0.7157 (n=2114) vs bosluksuz 0.5976.

## (g) Zaman sirali kalibrasyon (s02'nin normal donguleri)

s02'nin normal donguleri `start_idx` (oturum ici kare indeksi) sirasiyla dizildi. Duzen (A): uydurma 1-15 / kalibrasyon 16-30 / izleme 31-51; duzen (B): uydurma 1-20 / kalibrasyon 21-35 / izleme 36-51. Her duzende D0/D1/D2 parametreleri `ac.fit_params` ile yalniz uydurma dongulerinden ogrenilir; kalibrasyon dongulerinde standart conformal (`ac.conf_p`, p = (1 + #{cal >= s})/(n_cal+1)); test = izleme normalleri + TUM anomali donguleri (s03-s06); D3 = min(1, 2 min(p1, p2)); alpha = 0,10. Hucreler: B2 x pred ve ToF x tof.

### Hucre b2 x pred: 51 normal dongu; anomali donguleri: seq135 56, seq145 48, fast 48, slow 51.

**Duzen A** (uydurma 15, kalibrasyon 15, izleme 21): en kucuk p = 1/(n_cal+1) = 0.0625; en kucuk D3 p = 0.1250 -> D3 alpha = 0.1 altinda alarm VEREMEZ (yapisal). alpha +- 2SE = [-0.031, 0.231] (SE = 0.0655); saptanabilir en kucuk FAR artisi (k/n > ust sinir): k = 5/21 -> +0.138. Ortalama sure (s) uydurma/kal./izleme: 5.08 (SE 0.12) / 5.01 (0.15) / 4.92 (0.09).

| kural | izleme FAR (Wilson) | alpha +- 2SE icinde | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|---|
| D3 (birincil) | 0.000 [0.000, 0.155] (0/21) | evet | 0.000 [0.00, 0.06] (n=56) | 0.000 [0.00, 0.07] (n=48) | 0.000 [0.00, 0.07] (n=48) | 0.000 [0.00, 0.07] (n=51) | 0.000 [0.00, 0.04] (n=104) | 0.000 [0.00, 0.04] (n=99) | 0.000 [0.00, 0.02] (n=203) |
| D0 (ek) | 0.000 [0.000, 0.155] (0/21) | evet | 0.089 [0.04, 0.19] (n=56) | 0.062 [0.02, 0.17] (n=48) | 0.646 [0.50, 0.77] (n=48) | 1.000 [0.93, 1.00] (n=51) | 0.077 [0.04, 0.14] (n=104) | 0.828 [0.74, 0.89] (n=99) | 0.443 [0.38, 0.51] (n=203) |
| D1 (ek) | 0.000 [0.000, 0.155] (0/21) | evet | 0.000 [0.00, 0.06] (n=56) | 0.000 [0.00, 0.07] (n=48) | 0.000 [0.00, 0.07] (n=48) | 0.000 [0.00, 0.07] (n=51) | 0.000 [0.00, 0.04] (n=104) | 0.000 [0.00, 0.04] (n=99) | 0.000 [0.00, 0.02] (n=203) |
| D2 (ek) | 0.048 [0.008, 0.227] (1/21) | evet | 0.018 [0.00, 0.09] (n=56) | 0.021 [0.00, 0.11] (n=48) | 0.292 [0.18, 0.43] (n=48) | 0.529 [0.40, 0.66] (n=51) | 0.019 [0.01, 0.07] (n=104) | 0.414 [0.32, 0.51] (n=99) | 0.212 [0.16, 0.27] (n=203) |

(D3 skoru 1-p3 ile izleme normali vs tum anomali AUROC = 0.936; D0/D1/D2 satirlari yalniz ek bilgidir, birincil okuma D3 satirina dayanir.)

**Duzen B** (uydurma 20, kalibrasyon 15, izleme 16): en kucuk p = 1/(n_cal+1) = 0.0625; en kucuk D3 p = 0.1250 -> D3 alpha = 0.1 altinda alarm VEREMEZ (yapisal). alpha +- 2SE = [-0.050, 0.250] (SE = 0.0750); saptanabilir en kucuk FAR artisi (k/n > ust sinir): k = 5/16 -> +0.212. Ortalama sure (s) uydurma/kal./izleme: 5.02 (SE 0.10) / 5.09 (0.15) / 4.88 (0.12).

| kural | izleme FAR (Wilson) | alpha +- 2SE icinde | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|---|
| D3 (birincil) | 0.000 [0.000, 0.194] (0/16) | evet | 0.000 [0.00, 0.06] (n=56) | 0.000 [0.00, 0.07] (n=48) | 0.000 [0.00, 0.07] (n=48) | 0.000 [0.00, 0.07] (n=51) | 0.000 [0.00, 0.04] (n=104) | 0.000 [0.00, 0.04] (n=99) | 0.000 [0.00, 0.02] (n=203) |
| D0 (ek) | 0.000 [0.000, 0.194] (0/16) | evet | 0.089 [0.04, 0.19] (n=56) | 0.062 [0.02, 0.17] (n=48) | 0.646 [0.50, 0.77] (n=48) | 1.000 [0.93, 1.00] (n=51) | 0.077 [0.04, 0.14] (n=104) | 0.828 [0.74, 0.89] (n=99) | 0.443 [0.38, 0.51] (n=203) |
| D1 (ek) | 0.000 [0.000, 0.194] (0/16) | evet | 0.000 [0.00, 0.06] (n=56) | 0.000 [0.00, 0.07] (n=48) | 0.000 [0.00, 0.07] (n=48) | 0.000 [0.00, 0.07] (n=51) | 0.000 [0.00, 0.04] (n=104) | 0.000 [0.00, 0.04] (n=99) | 0.000 [0.00, 0.02] (n=203) |
| D2 (ek) | 0.062 [0.011, 0.283] (1/16) | evet | 0.054 [0.02, 0.15] (n=56) | 0.125 [0.06, 0.25] (n=48) | 0.833 [0.70, 0.91] (n=48) | 0.902 [0.79, 0.96] (n=51) | 0.087 [0.05, 0.16] (n=104) | 0.869 [0.79, 0.92] (n=99) | 0.468 [0.40, 0.54] (n=203) |

(D3 skoru 1-p3 ile izleme normali vs tum anomali AUROC = 0.951; D0/D1/D2 satirlari yalniz ek bilgidir, birincil okuma D3 satirina dayanir.)

### Hucre tof x tof: 51 normal dongu; anomali donguleri: seq135 57, seq145 49, fast 49, slow 52.

**Duzen A** (uydurma 15, kalibrasyon 15, izleme 21): en kucuk p = 1/(n_cal+1) = 0.0625; en kucuk D3 p = 0.1250 -> D3 alpha = 0.1 altinda alarm VEREMEZ (yapisal). alpha +- 2SE = [-0.031, 0.231] (SE = 0.0655); saptanabilir en kucuk FAR artisi (k/n > ust sinir): k = 5/21 -> +0.138. Ortalama sure (s) uydurma/kal./izleme: 5.09 (SE 0.13) / 5.01 (0.16) / 5.88 (0.96).

| kural | izleme FAR (Wilson) | alpha +- 2SE icinde | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|---|
| D3 (birincil) | 0.000 [0.000, 0.155] (0/21) | evet | 0.000 [0.00, 0.06] (n=57) | 0.000 [0.00, 0.07] (n=49) | 0.000 [0.00, 0.07] (n=49) | 0.000 [0.00, 0.07] (n=52) | 0.000 [0.00, 0.03] (n=106) | 0.000 [0.00, 0.04] (n=101) | 0.000 [0.00, 0.02] (n=207) |
| D0 (ek) | 0.048 [0.008, 0.227] (1/21) | evet | 0.140 [0.07, 0.25] (n=57) | 0.122 [0.06, 0.24] (n=49) | 0.714 [0.58, 0.82] (n=49) | 1.000 [0.93, 1.00] (n=52) | 0.132 [0.08, 0.21] (n=106) | 0.861 [0.78, 0.92] (n=101) | 0.488 [0.42, 0.56] (n=207) |
| D1 (ek) | 0.000 [0.000, 0.155] (0/21) | evet | 1.000 [0.94, 1.00] (n=57) | 1.000 [0.93, 1.00] (n=49) | 0.857 [0.73, 0.93] (n=49) | 0.000 [0.00, 0.07] (n=52) | 1.000 [0.97, 1.00] (n=106) | 0.416 [0.32, 0.51] (n=101) | 0.715 [0.65, 0.77] (n=207) |
| D2 (ek) | 0.048 [0.008, 0.227] (1/21) | evet | 0.000 [0.00, 0.06] (n=57) | 0.000 [0.00, 0.07] (n=49) | 0.143 [0.07, 0.27] (n=49) | 0.231 [0.14, 0.36] (n=52) | 0.000 [0.00, 0.03] (n=106) | 0.188 [0.12, 0.28] (n=101) | 0.092 [0.06, 0.14] (n=207) |

(D3 skoru 1-p3 ile izleme normali vs tum anomali AUROC = 0.966; D0/D1/D2 satirlari yalniz ek bilgidir, birincil okuma D3 satirina dayanir.)

**Duzen B** (uydurma 20, kalibrasyon 15, izleme 16): en kucuk p = 1/(n_cal+1) = 0.0625; en kucuk D3 p = 0.1250 -> D3 alpha = 0.1 altinda alarm VEREMEZ (yapisal). alpha +- 2SE = [-0.050, 0.250] (SE = 0.0750); saptanabilir en kucuk FAR artisi (k/n > ust sinir): k = 5/16 -> +0.212. Ortalama sure (s) uydurma/kal./izleme: 5.02 (SE 0.10) / 5.09 (0.15) / 6.13 (1.26).

| kural | izleme FAR (Wilson) | alpha +- 2SE icinde | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|---|
| D3 (birincil) | 0.000 [0.000, 0.194] (0/16) | evet | 0.000 [0.00, 0.06] (n=57) | 0.000 [0.00, 0.07] (n=49) | 0.000 [0.00, 0.07] (n=49) | 0.000 [0.00, 0.07] (n=52) | 0.000 [0.00, 0.03] (n=106) | 0.000 [0.00, 0.04] (n=101) | 0.000 [0.00, 0.02] (n=207) |
| D0 (ek) | 0.062 [0.011, 0.283] (1/16) | evet | 0.070 [0.03, 0.17] (n=57) | 0.102 [0.04, 0.22] (n=49) | 0.633 [0.49, 0.75] (n=49) | 1.000 [0.93, 1.00] (n=52) | 0.085 [0.05, 0.15] (n=106) | 0.822 [0.74, 0.88] (n=101) | 0.444 [0.38, 0.51] (n=207) |
| D1 (ek) | 0.000 [0.000, 0.194] (0/16) | evet | 1.000 [0.94, 1.00] (n=57) | 1.000 [0.93, 1.00] (n=49) | 0.857 [0.73, 0.93] (n=49) | 0.000 [0.00, 0.07] (n=52) | 1.000 [0.97, 1.00] (n=106) | 0.416 [0.32, 0.51] (n=101) | 0.715 [0.65, 0.77] (n=207) |
| D2 (ek) | 0.062 [0.011, 0.283] (1/16) | evet | 0.000 [0.00, 0.06] (n=57) | 0.000 [0.00, 0.07] (n=49) | 0.143 [0.07, 0.27] (n=49) | 0.385 [0.26, 0.52] (n=52) | 0.000 [0.00, 0.03] (n=106) | 0.267 [0.19, 0.36] (n=101) | 0.130 [0.09, 0.18] (n=207) |

(D3 skoru 1-p3 ile izleme normali vs tum anomali AUROC = 0.962; D0/D1/D2 satirlari yalniz ek bilgidir, birincil okuma D3 satirina dayanir.)

### (g) Onceden yazilmis okumanin harfiyen uygulanmasi

Kural: iki duzende de FAR alpha +- 2SE icindeyse 'surukleme saptanmadi'. D3, alpha = 0,10, izleme normalleri:

| hucre | duzen | izleme FAR | alpha +- 2SE icinde |
|---|---|---|---|
| b2 x pred | A | 0.000 | evet |
| b2 x pred | B | 0.000 | evet |
| tof x tof | A | 0.000 | evet |
| tof x tof | B | 0.000 | evet |

Harfiyen sonuc: b2 x pred: **surukleme saptanmadi**; tof x tof: **surukleme saptanmadi**.

Yazilacak cumle (plana gore): bu orneklemde saptanabilir en kucuk FAR artisi ~0,15 mertebesindedir. Hesaplanan: duzen A +0.14 (izleme 21 dongu), duzen B +0.21 (izleme 16 dongu) - yani ~0,15 duzen A icin yaklasik dogru, duzen B icin daha kaba (~0,2).

**Onemli uyari (okumanin gucu):** her iki duzende n_cal = 15 oldugu icin en kucuk standart-conformal p = 1/16 = 0,0625 ve en kucuk D3 p = 0,125 > 0,10. Yani D3 bu duzenlerde alpha = 0,10'da HICBIR dongu icin alarm veremez: FAR = 0 ve anomali tespiti = 0 yapisaldir (tablodaki D3 satirlari). Harfiyen okuma 'surukleme saptanmadi' dese de, D3 icin bu denetimin gucu yoktur; FAR ~0'in alpha +- 2SE bandina girmesi kuralin bir sonucudur, surukleme olmadiginin kaniti degildir. Ek D0/D1/D2 satirlari (tek tek, alarm verebilen kurallar) bu nedenle eklendi; bunlarin FAR'i ve band durumu yukarida verilir.

Ek (alarm verebilen) D0/D1/D2 kurallari icin ayni bant denetimi: iki hucrede, iki duzende, uc kuralin FAR'i alpha +- 2SE icinde mi? -> evet, hepsi icinde. Bu ek kontrol harfiyen okumanin parcasi degildir; bu kurallar alarm verebildigi icin D3'ten daha anlamlidir, yine de saptanabilir FAR artisi ~0,14-0,21 ile sinirlidir (kucuk izleme orneklemi).

Not: n_cal = 15 ile D1'in B2 x pred hucresindeki tespiti ~0'a duser (Faz A'da n_cal ~ 19-22 iken D1 seq(135+145) tespiti ~0,69). Mekanizma dogrudan hesaptan cikar: p <= 0,10 icin #{cal >= s} = 0 gerekir (n_cal=15: en kucuk p = 1/16 = 0,0625, bir esitlik p = 2/16 = 0,125). Gorulmemis gecis skoru ayni kaynak durum icin ozdestir; kalibrasyonda tek bir sahte gecisli dongu bu skoru doldurursa ayni durumdan gorulmemis gecisi olan tum anomaliler onunla baglasir (n_cal ~ 20'de #{cal >= s} <= 1 hala alarm verirdi). Olculen:

| hucre | duzen | n_cal | max cal D1 | max'a esit/ustu cal sayisi | seq anomalilerde D1 <= max cal D1 payi | seq anomali sayisi |
|---|---|---|---|---|---|---|
| b2 x pred | A | 15 | 3.045 | 2 | 1.000 | 104 |
| b2 x pred | B | 15 | 3.258 | 2 | 1.000 | 104 |
| tof x tof | A | 15 | 0.272 | 15 | 0.000 | 106 |
| tof x tof | B | 15 | 0.214 | 15 | 0.000 | 106 |

Payi ~1 olan hucrede D1 seq alarmi imkansizdir (anomali skoru kalibrasyon maksimumunu asmiyor); bu zaman sirali duzenin guc kaybidir, surukleme bulgusu degildir.

## Ozet sayilar

- (e) B2 x pred, alpha 0,10: D01 FAR 0.078 (k=4/51), tum-anomali tespit 0.631, AUROC 0.908; D3 FAR 0.078, tespit 0.633, AUROC 0.924; alpha 0,05'te ikisi de 0 tespit (yapisal).
- (f) bosluklu pencere payi: frame_id 0.0185% (3); zaman (dt>0,15 s) 13.01% (2114/16248).
- (g) D3 duzenleri: b2xpred A:0.000 B:0.000; tofxtof A:0.000 B:0.000; D3 alarm verebilir mi: False.

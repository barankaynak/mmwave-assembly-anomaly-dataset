# A2.1 - Conformal esik: esitlikler ve kalibrasyon cozunurlugu

Uretim: `faz_a2/01_conformal.py` (yerel makine). Ham sayilar: `01_conformal_table.csv` (uzun format), `01_conformal_meta.csv`. Her sayi betigin stdout'una da basilir. **Statu:** ikincil, Faz A'dan SONRA (DESIGN_A2.md); makalenin birincil sayilari Faz A'dir ve bu rapor onlari degistirmez. Yeniden egitim yok: yalniz mevcut S1 OOF tahminleri.

## 0. Yontem ozeti

- **Birincil kural degismedi:** D3 = min(1, 2 min(p1, p2)) (Bonferroni; alarm = 'D1 ya da D2, her biri a/2'de'). D1 ve D2 ayrica tek tek verilir (ikincil). **D1orD2** = '(p1 <= a) ya da (p2 <= a)': nominal FAR siniri <= 2a; **betimleyicidir, bir calisma noktasi olarak sunulmaz.**
- **(i) Faz A standart:** p = (1 + #{cal >= s}) / (n_cal + 1); roller Faz A ile ayni (egitim f+3,f+4; kal f+1,f+2; test f). Bu satirlar `04_anomaly_table.csv` ile karsilastirildi (asagida).
- **(ii) Duzgunlestirilmis p:** p = (#{cal > s} + U (#{cal = s} + 1)) / (n_cal + 1), U ~ Unif(0,1); ayni roller. 200 cekilis; tohum = 20261104 + 100 f + k (k = 1: D1, k = 2: D2; yani D1 ve D2 icin BAGIMSIZ U; ayni tohum cifti her hucrede). Tablolarda 'ortalama [%2,5; %97,5]' = cekilisler arasi aralik (yalniz U rastgeleligi; ornekleme belirsizligi **degil**).
- **(iii) Cross-conformal (s02 normal donguleri):** test kati f (`folds_anom.csv`); kalan 4 katin her biri k icin parametreler diger 3 kattan (~30 dongu) ogrenilir, k kati kat-disi kalibrasyon skorlarini verir, test/anomali donguleri model k ile skorlanir; p = (1 + sum_k #{cal_k >= s^(k)}) / (n_cal + 1). 'cross + duzgunlestirme' = ayni sayimlarla (ii)'deki formul (ayni tohumlar).
- **FAR** = havuzlanmis normal test donguleri (5 kat), **SE = sqrt(a(1-a)/n_normal)** (Faz A kurali); etiket: |FAR - a| <= 2 SE ise 'nominal icinde', altindaysa 'muhafazakar', ustundeyse 'ASIRI'. D1orD2 icin sinir 2a. Duzgunlestirilmis satirlarda etiket **ortalama** FAR'a uygulanir. **Tespit** = (dongu, f) ciftleri uzerinden anomali dongulerinde alarm orani (cross-conformal'da anomali donguleri her f'de yeniden skorlandigindan f uzerinden ortalama ile ayni).
- **AUROC:** kat bazli, sonra katlar uzerinden ortalama (Faz A kurali). Skor: D3 ve p-tabanli D1/D2 icin 1 - p; ayrica Faz A'nin ham-skor D1/D2 AUROC'u (std icin). Cross-conformal'da tek bir ham skor yoktur (her f icin 4 model), bu yuzden D1/D2 AUROC'u p-tabanlidir.

### Faz A ile esitlik kontrolu (i)

Yukaridaki std satirlari `reports/faz_a/04_anomaly_table.csv` ile karsilastirildi (D1, D2, D3 x 7 anomali grubu x FAR / tespit x a = 0,10 ve 0,05, ve AUROC ortalama ile sd):

| hucre | karsilastirilan_sayi | max_mutlak_fark |
|---|---|---|
| B2 x pred (ana) | 168 | 2.22e-16 |
| B2 x tof | 168 | 3.33e-16 |
| ToF oracle x tof | 168 | 3.33e-16 |
| CNN-GRU x pred | 168 | 1.39e-16 |

## 1. Kalibrasyon cozunurlugu: n_cal ve ulasilabilir en kucuk p

Standart conformal'da en kucuk p = 1/(n_cal+1); D3 = min(1, 2 min(p1,p2)) oldugundan D3'un en kucugu 2/(n_cal+1). Duzgunlestirmede p surekli (U/(n_cal+1)'e kadar iner); 'gozlenen en kucuk p' test satirlari uzerindendir (std/cross: tek deger; smooth: 200 cekilis uzerinden en kucuk).

| hucre | std_ncal | std_egitim | std_minp_D1D2 | std_minp_D3 | std_gozlenen_min_p3 | cross_ncal | cross_egitim_model_basi | cross_minp_D1D2 | cross_minp_D3 | cross_gozlenen_min_p3 | smooth_gozlenen_min_p3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B2 x pred (ana) | 19/20.4/22 | 20.4 | 0.0435-0.0500 | 0.0870-0.1000 | 0.0870 | 40-42 | 31,30,30,29 | 0.0233 | 0.0465 | 0.0465 | 0.00000 |
| B2 x tof | 20/20.4/21 | 20.4 | 0.0455-0.0476 | 0.0909-0.0952 | 0.0909 | 40-41 | 30,30,30,30 | 0.0238 | 0.0476 | 0.0476 | 0.00000 |
| ToF oracle x tof | 20/20.4/21 | 20.4 | 0.0455-0.0476 | 0.0909-0.0952 | 0.0909 | 40-41 | 30,30,30,30 | 0.0238 | 0.0476 | 0.0476 | 0.00000 |
| CNN-GRU x pred | 18/19.6/22 | 19.6 | 0.0435-0.0526 | 0.0870-0.1053 | 0.0870 | 38-42 | 31,27,27,29 | 0.0233 | 0.0465 | 0.0465 | 0.00000 |

(std_ncal = min/ortalama/maks kat basina kalibrasyon dongusu; std_minp_* aralik = n_cal maks ... n_cal min.) **Okuma:** std'de D3'un ulasilabilir en kucuk p'si ~0,09-0,10 oldugundan a = 0,05'te D3 alarmi yapisal olarak imkansiz, a = 0,10'da yalniz tum kalibrasyonu asan dongulerde mumkun. cross-conformal'da n_cal ~40 oldugundan D3 icin en kucuk p ~0,049: a = 0,05'e ancak sinirda ulasilir.

Test kumeleri (hucre basina): normal test cifti sayisi = B2 x pred (ana): 51, B2 x tof: 51, ToF oracle x tof: 51, CNN-GRU x pred: 49; her f'de anomali dongusu = B2 x pred (ana): 203, B2 x tof: 207, ToF oracle x tof: 207, CNN-GRU x pred: 210.

## 2. FAR ve tespit, a = 0.10

Sutunlar: FAR (SE; etiket), tur basina tespit, ayrica seq(135+145), tempo(fast+slow) ve tum anomali havuzlari. Duzgunlestirilmis satirlar 'ortalama [%2,5; %97,5]'.

### B2 x pred (ana), a = 0.10

| degisken | yontem | FAR | gecerlilik | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | Faz A standart | 0.059 (SE 0.042) | nominal +-2SE icinde | 0.604 | 0.800 | 0.217 | 0.008 | 0.694 | 0.109 | 0.409 |
| D1 | duzgunlestirilmis | 0.099 [0.059; 0.157] (SE 0.042) | nominal +-2SE icinde | 0.706 [0.678; 0.736] | 0.947 [0.925; 0.975] | 0.276 [0.254; 0.296] | 0.041 [0.024; 0.063] | 0.817 [0.800; 0.840] | 0.155 [0.141; 0.170] | 0.494 [0.482; 0.507] |
| D1 | cross-conformal | 0.078 (SE 0.042) | nominal +-2SE icinde | 1.000 | 1.000 | 0.271 | 0.012 | 1.000 | 0.137 | 0.579 |
| D1 | cross + duzgunlestirme | 0.102 [0.078; 0.137] (SE 0.042) | nominal +-2SE icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.289 [0.275; 0.308] | 0.042 [0.024; 0.063] | 1.000 [1.000; 1.000] | 0.162 [0.152; 0.176] | 0.591 [0.586; 0.598] |
| D2 | Faz A standart | 0.078 (SE 0.042) | nominal +-2SE icinde | 0.068 | 0.125 | 0.800 | 0.992 | 0.094 | 0.899 | 0.487 |
| D2 | duzgunlestirilmis | 0.098 [0.078; 0.137] (SE 0.042) | nominal +-2SE icinde | 0.090 [0.079; 0.104] | 0.143 [0.129; 0.158] | 0.808 [0.800; 0.821] | 0.993 [0.992; 1.000] | 0.115 [0.106; 0.125] | 0.903 [0.899; 0.909] | 0.499 [0.495; 0.504] |
| D2 | cross-conformal | 0.078 (SE 0.042) | nominal +-2SE icinde | 0.071 | 0.125 | 0.829 | 1.000 | 0.096 | 0.917 | 0.497 |
| D2 | cross + duzgunlestirme | 0.083 [0.078; 0.098] (SE 0.042) | nominal +-2SE icinde | 0.086 [0.075; 0.100] | 0.140 [0.129; 0.154] | 0.832 [0.829; 0.838] | 1.000 [1.000; 1.000] | 0.111 [0.102; 0.121] | 0.919 [0.917; 0.921] | 0.505 [0.500; 0.510] |
| D3 | Faz A standart | 0.078 (SE 0.042) | nominal +-2SE icinde | 0.254 | 0.475 | 0.921 | 0.929 | 0.356 | 0.925 | 0.633 |
| D3 | duzgunlestirilmis | 0.087 [0.078; 0.098] (SE 0.042) | nominal +-2SE icinde | 0.445 [0.418; 0.479] | 0.719 [0.683; 0.758] | 0.936 [0.925; 0.950] | 0.932 [0.929; 0.937] | 0.571 [0.546; 0.598] | 0.934 [0.927; 0.939] | 0.748 [0.735; 0.762] |
| D3 | cross-conformal | 0.059 (SE 0.042) | nominal +-2SE icinde | 0.257 | 0.650 | 0.925 | 0.937 | 0.438 | 0.931 | 0.679 |
| D3 | cross + duzgunlestirme | 0.091 [0.059; 0.118] (SE 0.042) | nominal +-2SE icinde | 0.377 [0.350; 0.407] | 0.862 [0.829; 0.896] | 0.951 [0.938; 0.958] | 0.942 [0.937; 0.949] | 0.601 [0.575; 0.625] | 0.946 [0.939; 0.952] | 0.769 [0.756; 0.782] |
| D1orD2 (betimleyici, nominal<=2a) | Faz A standart | 0.118 (SE 0.056) | <=2a sinir icinde | 0.629 | 0.825 | 0.929 | 0.992 | 0.719 | 0.962 | 0.837 |
| D1orD2 (betimleyici, nominal<=2a) | duzgunlestirilmis | 0.177 [0.137; 0.235] (SE 0.056) | <=2a sinir icinde | 0.738 [0.707; 0.768] | 0.954 [0.929; 0.975] | 0.959 [0.946; 0.971] | 0.994 [0.992; 1.000] | 0.837 [0.819; 0.856] | 0.977 [0.970; 0.984] | 0.905 [0.896; 0.917] |
| D1orD2 (betimleyici, nominal<=2a) | cross-conformal | 0.137 (SE 0.056) | <=2a sinir icinde | 1.000 | 1.000 | 0.979 | 1.000 | 1.000 | 0.990 | 0.995 |
| D1orD2 (betimleyici, nominal<=2a) | cross + duzgunlestirme | 0.164 [0.137; 0.196] (SE 0.056) | <=2a sinir icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.980 [0.979; 0.983] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.990 [0.990; 0.992] | 0.995 [0.995; 0.996] |

### B2 x tof, a = 0.10

| degisken | yontem | FAR | gecerlilik | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | Faz A standart | 0.059 (SE 0.042) | nominal +-2SE icinde | 0.670 | 0.800 | 0.212 | 0.008 | 0.730 | 0.107 | 0.426 |
| D1 | duzgunlestirilmis | 0.102 [0.059; 0.157] (SE 0.042) | nominal +-2SE icinde | 0.758 [0.733; 0.786] | 0.845 [0.828; 0.865] | 0.256 [0.241; 0.273] | 0.052 [0.031; 0.077] | 0.798 [0.783; 0.815] | 0.151 [0.135; 0.164] | 0.482 [0.472; 0.494] |
| D1 | cross-conformal | 0.078 (SE 0.042) | nominal +-2SE icinde | 1.000 | 1.000 | 0.265 | 0.012 | 1.000 | 0.135 | 0.578 |
| D1 | cross + duzgunlestirme | 0.101 [0.078; 0.138] (SE 0.042) | nominal +-2SE icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.270 [0.265; 0.278] | 0.044 [0.023; 0.065] | 1.000 [1.000; 1.000] | 0.153 [0.143; 0.166] | 0.587 [0.582; 0.593] |
| D2 | Faz A standart | 0.039 (SE 0.042) | nominal +-2SE icinde | 0.095 | 0.151 | 0.698 | 0.877 | 0.121 | 0.790 | 0.447 |
| D2 | duzgunlestirilmis | 0.042 [0.039; 0.059] (SE 0.042) | nominal +-2SE icinde | 0.112 [0.102; 0.126] | 0.168 [0.155; 0.184] | 0.699 [0.698; 0.706] | 0.883 [0.877; 0.892] | 0.138 [0.128; 0.149] | 0.794 [0.790; 0.800] | 0.458 [0.453; 0.464] |
| D2 | cross-conformal | 0.078 (SE 0.042) | nominal +-2SE icinde | 0.063 | 0.122 | 0.800 | 0.908 | 0.091 | 0.855 | 0.464 |
| D2 | cross + duzgunlestirme | 0.082 [0.078; 0.098] (SE 0.042) | nominal +-2SE icinde | 0.063 [0.063; 0.063] | 0.122 [0.122; 0.122] | 0.803 [0.800; 0.808] | 0.917 [0.908; 0.927] | 0.091 [0.091; 0.091] | 0.862 [0.857; 0.867] | 0.467 [0.465; 0.470] |
| D3 | Faz A standart | 0.059 (SE 0.042) | nominal +-2SE icinde | 0.305 | 0.473 | 0.788 | 0.827 | 0.383 | 0.808 | 0.590 |
| D3 | duzgunlestirilmis | 0.070 [0.059; 0.098] (SE 0.042) | nominal +-2SE icinde | 0.504 [0.467; 0.537] | 0.659 [0.628; 0.686] | 0.823 [0.804; 0.841] | 0.830 [0.827; 0.838] | 0.576 [0.547; 0.596] | 0.826 [0.820; 0.836] | 0.698 [0.684; 0.710] |
| D3 | cross-conformal | 0.059 (SE 0.042) | nominal +-2SE icinde | 0.368 | 0.649 | 0.922 | 0.831 | 0.498 | 0.875 | 0.682 |
| D3 | cross + duzgunlestirme | 0.093 [0.059; 0.118] (SE 0.042) | nominal +-2SE icinde | 0.526 [0.495; 0.558] | 0.774 [0.747; 0.800] | 0.945 [0.935; 0.955] | 0.836 [0.831; 0.846] | 0.640 [0.619; 0.662] | 0.889 [0.881; 0.895] | 0.762 [0.750; 0.774] |
| D1orD2 (betimleyici, nominal<=2a) | Faz A standart | 0.078 (SE 0.056) | <=2a sinir icinde | 0.684 | 0.824 | 0.841 | 0.877 | 0.749 | 0.859 | 0.803 |
| D1orD2 (betimleyici, nominal<=2a) | duzgunlestirilmis | 0.125 [0.078; 0.176] (SE 0.056) | <=2a sinir icinde | 0.768 [0.744; 0.793] | 0.865 [0.849; 0.882] | 0.864 [0.853; 0.873] | 0.887 [0.877; 0.896] | 0.813 [0.798; 0.830] | 0.876 [0.869; 0.883] | 0.844 [0.836; 0.854] |
| D1orD2 (betimleyici, nominal<=2a) | cross-conformal | 0.137 (SE 0.056) | <=2a sinir icinde | 1.000 | 1.000 | 0.976 | 0.908 | 1.000 | 0.941 | 0.971 |
| D1orD2 (betimleyici, nominal<=2a) | cross + duzgunlestirme | 0.159 [0.137; 0.197] (SE 0.056) | <=2a sinir icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.976 [0.976; 0.976] | 0.918 [0.908; 0.927] | 1.000 [1.000; 1.000] | 0.946 [0.941; 0.950] | 0.973 [0.971; 0.976] |

### ToF oracle x tof, a = 0.10

| degisken | yontem | FAR | gecerlilik | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | Faz A standart | 0.000 (SE 0.042) | muhafazakar (< a-2SE) | 1.000 | 1.000 | 0.857 | 0.000 | 1.000 | 0.416 | 0.715 |
| D1 | duzgunlestirilmis | 0.098 [0.020; 0.196] (SE 0.042) | nominal +-2SE icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.872 [0.861; 0.886] | 0.099 [0.065; 0.138] | 1.000 [1.000; 1.000] | 0.474 [0.455; 0.497] | 0.743 [0.734; 0.755] |
| D1 | cross-conformal | 0.000 (SE 0.042) | muhafazakar (< a-2SE) | 1.000 | 1.000 | 0.857 | 0.000 | 1.000 | 0.416 | 0.715 |
| D1 | cross + duzgunlestirme | 0.098 [0.020; 0.196] (SE 0.042) | nominal +-2SE icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.872 [0.861; 0.886] | 0.099 [0.065; 0.138] | 1.000 [1.000; 1.000] | 0.474 [0.455; 0.497] | 0.743 [0.734; 0.755] |
| D2 | Faz A standart | 0.098 (SE 0.042) | nominal +-2SE icinde | 0.021 | 0.000 | 0.322 | 0.881 | 0.011 | 0.610 | 0.303 |
| D2 | duzgunlestirilmis | 0.098 [0.098; 0.098] (SE 0.042) | nominal +-2SE icinde | 0.023 [0.021; 0.028] | 0.002 [0.000; 0.008] | 0.341 [0.327; 0.355] | 0.900 [0.888; 0.915] | 0.014 [0.011; 0.017] | 0.629 [0.620; 0.640] | 0.314 [0.309; 0.321] |
| D2 | cross-conformal | 0.078 (SE 0.042) | nominal +-2SE icinde | 0.028 | 0.000 | 0.347 | 1.000 | 0.015 | 0.683 | 0.341 |
| D2 | cross + duzgunlestirme | 0.081 [0.078; 0.098] (SE 0.042) | nominal +-2SE icinde | 0.030 [0.028; 0.035] | 0.000 [0.000; 0.000] | 0.347 [0.347; 0.347] | 1.000 [1.000; 1.000] | 0.016 [0.015; 0.019] | 0.683 [0.683; 0.683] | 0.341 [0.341; 0.343] |
| D3 | Faz A standart | 0.078 (SE 0.042) | nominal +-2SE icinde | 1.000 | 1.000 | 0.931 | 0.600 | 1.000 | 0.760 | 0.883 |
| D3 | duzgunlestirilmis | 0.122 [0.078; 0.176] (SE 0.042) | nominal +-2SE icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.934 [0.931; 0.943] | 0.639 [0.623; 0.662] | 1.000 [1.000; 1.000] | 0.782 [0.772; 0.796] | 0.894 [0.889; 0.900] |
| D3 | cross-conformal | 0.039 (SE 0.042) | nominal +-2SE icinde | 1.000 | 1.000 | 0.922 | 0.738 | 1.000 | 0.828 | 0.916 |
| D3 | cross + duzgunlestirme | 0.086 [0.039; 0.137] (SE 0.042) | nominal +-2SE icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.927 [0.922; 0.935] | 0.766 [0.750; 0.789] | 1.000 [1.000; 1.000] | 0.844 [0.836; 0.855] | 0.924 [0.920; 0.929] |
| D1orD2 (betimleyici, nominal<=2a) | Faz A standart | 0.098 (SE 0.056) | <=2a sinir icinde | 1.000 | 1.000 | 0.935 | 0.881 | 1.000 | 0.907 | 0.955 |
| D1orD2 (betimleyici, nominal<=2a) | duzgunlestirilmis | 0.184 [0.118; 0.275] (SE 0.056) | <=2a sinir icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.943 [0.935; 0.955] | 0.910 [0.896; 0.931] | 1.000 [1.000; 1.000] | 0.926 [0.917; 0.937] | 0.964 [0.959; 0.969] |
| D1orD2 (betimleyici, nominal<=2a) | cross-conformal | 0.078 (SE 0.056) | <=2a sinir icinde | 1.000 | 1.000 | 0.939 | 1.000 | 1.000 | 0.970 | 0.986 |
| D1orD2 (betimleyici, nominal<=2a) | cross + duzgunlestirme | 0.170 [0.098; 0.255] (SE 0.056) | <=2a sinir icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.945 [0.939; 0.955] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.973 [0.970; 0.978] | 0.987 [0.986; 0.989] |

### CNN-GRU x pred, a = 0.10

| degisken | yontem | FAR | gecerlilik | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | Faz A standart | 0.000 (SE 0.043) | muhafazakar (< a-2SE) | 0.011 | 0.600 | 0.265 | 0.129 | 0.276 | 0.191 | 0.232 |
| D1 | duzgunlestirilmis | 0.088 [0.041; 0.143] (SE 0.043) | nominal +-2SE icinde | 0.525 [0.496; 0.554] | 0.896 [0.865; 0.926] | 0.363 [0.343; 0.388] | 0.231 [0.214; 0.251] | 0.692 [0.671; 0.712] | 0.291 [0.276; 0.306] | 0.486 [0.471; 0.498] |
| D1 | cross-conformal | 0.020 (SE 0.043) | nominal +-2SE icinde | 0.214 | 1.000 | 0.367 | 0.207 | 0.569 | 0.280 | 0.420 |
| D1 | cross + duzgunlestirme | 0.104 [0.082; 0.122] (SE 0.043) | nominal +-2SE icinde | 0.812 [0.764; 0.857] | 1.000 [1.000; 1.000] | 0.394 [0.384; 0.404] | 0.250 [0.237; 0.261] | 0.897 [0.871; 0.922] | 0.315 [0.307; 0.322] | 0.598 [0.585; 0.611] |
| D2 | Faz A standart | 0.122 (SE 0.043) | nominal +-2SE icinde | 0.036 | 0.100 | 0.396 | 0.651 | 0.065 | 0.535 | 0.307 |
| D2 | duzgunlestirilmis | 0.126 [0.122; 0.143] (SE 0.043) | nominal +-2SE icinde | 0.040 [0.036; 0.046] | 0.159 [0.148; 0.170] | 0.525 [0.506; 0.543] | 0.703 [0.692; 0.719] | 0.094 [0.088; 0.100] | 0.623 [0.611; 0.633] | 0.366 [0.360; 0.371] |
| D2 | cross-conformal | 0.082 (SE 0.043) | nominal +-2SE icinde | 0.021 | 0.113 | 0.408 | 0.644 | 0.063 | 0.537 | 0.307 |
| D2 | cross + duzgunlestirme | 0.082 [0.082; 0.082] (SE 0.043) | nominal +-2SE icinde | 0.021 [0.021; 0.021] | 0.174 [0.161; 0.187] | 0.539 [0.514; 0.559] | 0.687 [0.675; 0.698] | 0.090 [0.084; 0.096] | 0.620 [0.609; 0.630] | 0.363 [0.356; 0.369] |
| D3 | Faz A standart | 0.082 (SE 0.043) | nominal +-2SE icinde | 0.021 | 0.409 | 0.359 | 0.376 | 0.196 | 0.369 | 0.285 |
| D3 | duzgunlestirilmis | 0.120 [0.102; 0.143] (SE 0.043) | nominal +-2SE icinde | 0.285 [0.246; 0.329] | 0.751 [0.713; 0.787] | 0.605 [0.584; 0.629] | 0.694 [0.674; 0.715] | 0.495 [0.465; 0.524] | 0.653 [0.639; 0.669] | 0.577 [0.562; 0.591] |
| D3 | cross-conformal | 0.020 (SE 0.043) | nominal +-2SE icinde | 0.029 | 0.626 | 0.563 | 0.495 | 0.298 | 0.526 | 0.415 |
| D3 | cross + duzgunlestirme | 0.088 [0.061; 0.102] (SE 0.043) | nominal +-2SE icinde | 0.237 [0.221; 0.250] | 0.992 [0.978; 1.000] | 0.681 [0.669; 0.686] | 0.635 [0.620; 0.647] | 0.577 [0.569; 0.588] | 0.656 [0.648; 0.663] | 0.617 [0.612; 0.623] |
| D1orD2 (betimleyici, nominal<=2a) | Faz A standart | 0.122 (SE 0.057) | <=2a sinir icinde | 0.046 | 0.622 | 0.596 | 0.729 | 0.306 | 0.669 | 0.492 |
| D1orD2 (betimleyici, nominal<=2a) | duzgunlestirilmis | 0.169 [0.143; 0.204] (SE 0.057) | <=2a sinir icinde | 0.538 [0.511; 0.564] | 0.904 [0.870; 0.931] | 0.776 [0.751; 0.804] | 0.831 [0.814; 0.847] | 0.703 [0.682; 0.725] | 0.806 [0.791; 0.822] | 0.756 [0.741; 0.769] |
| D1orD2 (betimleyici, nominal<=2a) | cross-conformal | 0.102 (SE 0.057) | <=2a sinir icinde | 0.232 | 1.000 | 0.722 | 0.776 | 0.578 | 0.752 | 0.668 |
| D1orD2 (betimleyici, nominal<=2a) | cross + duzgunlestirme | 0.150 [0.122; 0.163] (SE 0.057) | <=2a sinir icinde | 0.815 [0.768; 0.857] | 1.000 [1.000; 1.000] | 0.849 [0.829; 0.865] | 0.824 [0.810; 0.837] | 0.898 [0.873; 0.922] | 0.835 [0.824; 0.846] | 0.866 [0.852; 0.878] |

## 3. FAR ve tespit, a = 0.05

Sutunlar: FAR (SE; etiket), tur basina tespit, ayrica seq(135+145), tempo(fast+slow) ve tum anomali havuzlari. Duzgunlestirilmis satirlar 'ortalama [%2,5; %97,5]'.

### B2 x pred (ana), a = 0.05

| degisken | yontem | FAR | gecerlilik | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | Faz A standart | 0.020 (SE 0.031) | nominal +-2SE icinde | 0.211 | 0.400 | 0.217 | 0.000 | 0.298 | 0.105 | 0.204 |
| D1 | duzgunlestirilmis | 0.029 [0.020; 0.040] (SE 0.031) | nominal +-2SE icinde | 0.413 [0.379; 0.447] | 0.679 [0.637; 0.729] | 0.236 [0.221; 0.250] | 0.006 [0.000; 0.012] | 0.536 [0.508; 0.564] | 0.118 [0.109; 0.125] | 0.332 [0.318; 0.348] |
| D1 | cross-conformal | 0.020 (SE 0.031) | nominal +-2SE icinde | 0.211 | 0.600 | 0.217 | 0.000 | 0.390 | 0.105 | 0.251 |
| D1 | cross + duzgunlestirme | 0.048 [0.020; 0.078] (SE 0.031) | nominal +-2SE icinde | 0.340 [0.307; 0.372] | 0.843 [0.808; 0.883] | 0.254 [0.237; 0.267] | 0.008 [0.000; 0.016] | 0.572 [0.544; 0.600] | 0.127 [0.119; 0.133] | 0.355 [0.340; 0.369] |
| D2 | Faz A standart | 0.059 (SE 0.031) | nominal +-2SE icinde | 0.054 | 0.125 | 0.771 | 0.929 | 0.087 | 0.853 | 0.460 |
| D2 | duzgunlestirilmis | 0.059 [0.059; 0.059] (SE 0.031) | nominal +-2SE icinde | 0.055 [0.054; 0.061] | 0.125 [0.125; 0.125] | 0.772 [0.771; 0.779] | 0.932 [0.929; 0.937] | 0.087 [0.087; 0.090] | 0.855 [0.853; 0.859] | 0.461 [0.460; 0.463] |
| D2 | cross-conformal | 0.039 (SE 0.031) | nominal +-2SE icinde | 0.054 | 0.125 | 0.767 | 0.937 | 0.087 | 0.855 | 0.461 |
| D2 | cross + duzgunlestirme | 0.043 [0.039; 0.059] (SE 0.031) | nominal +-2SE icinde | 0.055 [0.054; 0.061] | 0.125 [0.125; 0.125] | 0.768 [0.767; 0.775] | 0.942 [0.937; 0.949] | 0.087 [0.087; 0.090] | 0.858 [0.855; 0.863] | 0.463 [0.461; 0.466] |
| D3 | Faz A standart | 0.000 (SE 0.031) | nominal +-2SE icinde | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| D3 | duzgunlestirilmis | 0.047 [0.000; 0.078] (SE 0.031) | nominal +-2SE icinde | 0.234 [0.193; 0.272] | 0.402 [0.346; 0.454] | 0.517 [0.454; 0.575] | 0.497 [0.439; 0.553] | 0.312 [0.279; 0.340] | 0.507 [0.463; 0.552] | 0.407 [0.380; 0.431] |
| D3 | cross-conformal | 0.039 (SE 0.031) | nominal +-2SE icinde | 0.239 | 0.300 | 0.871 | 0.780 | 0.267 | 0.824 | 0.539 |
| D3 | cross + duzgunlestirme | 0.041 [0.039; 0.059] (SE 0.031) | nominal +-2SE icinde | 0.245 [0.239; 0.254] | 0.586 [0.542; 0.633] | 0.908 [0.892; 0.933] | 0.788 [0.780; 0.796] | 0.402 [0.381; 0.423] | 0.846 [0.836; 0.857] | 0.619 [0.606; 0.631] |
| D1orD2 (betimleyici, nominal<=2a) | Faz A standart | 0.078 (SE 0.042) | <=2a sinir icinde | 0.254 | 0.475 | 0.921 | 0.929 | 0.356 | 0.925 | 0.633 |
| D1orD2 (betimleyici, nominal<=2a) | duzgunlestirilmis | 0.087 [0.078; 0.098] (SE 0.042) | <=2a sinir icinde | 0.445 [0.418; 0.479] | 0.719 [0.683; 0.758] | 0.936 [0.925; 0.950] | 0.932 [0.929; 0.937] | 0.571 [0.546; 0.598] | 0.934 [0.927; 0.939] | 0.748 [0.735; 0.762] |
| D1orD2 (betimleyici, nominal<=2a) | cross-conformal | 0.059 (SE 0.042) | <=2a sinir icinde | 0.257 | 0.650 | 0.925 | 0.937 | 0.438 | 0.931 | 0.679 |
| D1orD2 (betimleyici, nominal<=2a) | cross + duzgunlestirme | 0.091 [0.059; 0.118] (SE 0.042) | <=2a sinir icinde | 0.377 [0.350; 0.407] | 0.862 [0.829; 0.896] | 0.951 [0.938; 0.958] | 0.942 [0.937; 0.949] | 0.601 [0.575; 0.625] | 0.946 [0.939; 0.952] | 0.769 [0.756; 0.782] |

### B2 x tof, a = 0.05

| degisken | yontem | FAR | gecerlilik | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | Faz A standart | 0.020 (SE 0.031) | nominal +-2SE icinde | 0.274 | 0.400 | 0.159 | 0.000 | 0.332 | 0.077 | 0.208 |
| D1 | duzgunlestirilmis | 0.034 [0.020; 0.059] (SE 0.031) | nominal +-2SE icinde | 0.480 [0.446; 0.519] | 0.612 [0.579; 0.645] | 0.206 [0.184; 0.224] | 0.008 [0.000; 0.015] | 0.541 [0.513; 0.566] | 0.104 [0.093; 0.115] | 0.328 [0.313; 0.342] |
| D1 | cross-conformal | 0.020 (SE 0.031) | nominal +-2SE icinde | 0.344 | 0.600 | 0.212 | 0.000 | 0.462 | 0.103 | 0.287 |
| D1 | cross + duzgunlestirme | 0.052 [0.020; 0.078] (SE 0.031) | nominal +-2SE icinde | 0.505 [0.474; 0.537] | 0.742 [0.714; 0.771] | 0.240 [0.229; 0.253] | 0.009 [0.000; 0.015] | 0.615 [0.592; 0.640] | 0.121 [0.113; 0.131] | 0.374 [0.361; 0.386] |
| D2 | Faz A standart | 0.039 (SE 0.031) | nominal +-2SE icinde | 0.046 | 0.102 | 0.673 | 0.827 | 0.072 | 0.752 | 0.404 |
| D2 | duzgunlestirilmis | 0.039 [0.039; 0.039] (SE 0.031) | nominal +-2SE icinde | 0.048 [0.046; 0.056] | 0.104 [0.102; 0.110] | 0.675 [0.673; 0.682] | 0.829 [0.827; 0.838] | 0.074 [0.072; 0.079] | 0.755 [0.752; 0.758] | 0.406 [0.404; 0.409] |
| D2 | cross-conformal | 0.039 (SE 0.031) | nominal +-2SE icinde | 0.039 | 0.122 | 0.759 | 0.831 | 0.077 | 0.796 | 0.428 |
| D2 | cross + duzgunlestirme | 0.044 [0.039; 0.059] (SE 0.031) | nominal +-2SE icinde | 0.040 [0.039; 0.046] | 0.122 [0.122; 0.122] | 0.761 [0.759; 0.767] | 0.836 [0.831; 0.846] | 0.078 [0.077; 0.081] | 0.800 [0.796; 0.806] | 0.430 [0.428; 0.434] |
| D3 | Faz A standart | 0.000 (SE 0.031) | nominal +-2SE icinde | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| D3 | duzgunlestirilmis | 0.036 [0.000; 0.078] (SE 0.031) | nominal +-2SE icinde | 0.265 [0.221; 0.305] | 0.357 [0.302; 0.400] | 0.451 [0.400; 0.498] | 0.441 [0.388; 0.492] | 0.307 [0.275; 0.340] | 0.446 [0.410; 0.485] | 0.375 [0.352; 0.397] |
| D3 | cross-conformal | 0.039 (SE 0.031) | nominal +-2SE icinde | 0.228 | 0.298 | 0.873 | 0.750 | 0.260 | 0.810 | 0.529 |
| D3 | cross + duzgunlestirme | 0.042 [0.039; 0.059] (SE 0.031) | nominal +-2SE icinde | 0.262 [0.246; 0.281] | 0.460 [0.428; 0.490] | 0.907 [0.890; 0.922] | 0.754 [0.750; 0.762] | 0.354 [0.336; 0.370] | 0.828 [0.820; 0.836] | 0.585 [0.575; 0.594] |
| D1orD2 (betimleyici, nominal<=2a) | Faz A standart | 0.059 (SE 0.042) | <=2a sinir icinde | 0.305 | 0.473 | 0.788 | 0.827 | 0.383 | 0.808 | 0.590 |
| D1orD2 (betimleyici, nominal<=2a) | duzgunlestirilmis | 0.070 [0.059; 0.098] (SE 0.042) | <=2a sinir icinde | 0.504 [0.467; 0.537] | 0.659 [0.628; 0.686] | 0.823 [0.804; 0.841] | 0.830 [0.827; 0.838] | 0.576 [0.547; 0.596] | 0.826 [0.820; 0.836] | 0.698 [0.684; 0.710] |
| D1orD2 (betimleyici, nominal<=2a) | cross-conformal | 0.059 (SE 0.042) | <=2a sinir icinde | 0.368 | 0.649 | 0.922 | 0.831 | 0.498 | 0.875 | 0.682 |
| D1orD2 (betimleyici, nominal<=2a) | cross + duzgunlestirme | 0.093 [0.059; 0.118] (SE 0.042) | <=2a sinir icinde | 0.526 [0.495; 0.558] | 0.774 [0.747; 0.800] | 0.945 [0.935; 0.955] | 0.836 [0.831; 0.846] | 0.640 [0.619; 0.662] | 0.889 [0.881; 0.895] | 0.762 [0.750; 0.774] |

### ToF oracle x tof, a = 0.05

| degisken | yontem | FAR | gecerlilik | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | Faz A standart | 0.000 (SE 0.031) | nominal +-2SE icinde | 1.000 | 1.000 | 0.857 | 0.000 | 1.000 | 0.416 | 0.715 |
| D1 | duzgunlestirilmis | 0.047 [0.000; 0.099] (SE 0.031) | nominal +-2SE icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.865 [0.857; 0.878] | 0.050 [0.027; 0.081] | 1.000 [1.000; 1.000] | 0.445 [0.432; 0.461] | 0.729 [0.723; 0.737] |
| D1 | cross-conformal | 0.000 (SE 0.031) | nominal +-2SE icinde | 1.000 | 1.000 | 0.857 | 0.000 | 1.000 | 0.416 | 0.715 |
| D1 | cross + duzgunlestirme | 0.047 [0.000; 0.099] (SE 0.031) | nominal +-2SE icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.865 [0.857; 0.878] | 0.050 [0.027; 0.081] | 1.000 [1.000; 1.000] | 0.445 [0.432; 0.461] | 0.729 [0.723; 0.737] |
| D2 | Faz A standart | 0.078 (SE 0.031) | nominal +-2SE icinde | 0.014 | 0.000 | 0.265 | 0.600 | 0.008 | 0.438 | 0.217 |
| D2 | duzgunlestirilmis | 0.079 [0.078; 0.079] (SE 0.031) | nominal +-2SE icinde | 0.015 [0.014; 0.018] | 0.000 [0.000; 0.000] | 0.268 [0.265; 0.274] | 0.620 [0.608; 0.635] | 0.008 [0.008; 0.009] | 0.449 [0.442; 0.459] | 0.223 [0.219; 0.228] |
| D2 | cross-conformal | 0.039 (SE 0.031) | nominal +-2SE icinde | 0.004 | 0.000 | 0.184 | 0.738 | 0.002 | 0.469 | 0.230 |
| D2 | cross + duzgunlestirme | 0.040 [0.039; 0.059] (SE 0.031) | nominal +-2SE icinde | 0.004 [0.004; 0.007] | 0.000 [0.000; 0.000] | 0.185 [0.184; 0.192] | 0.754 [0.742; 0.769] | 0.002 [0.002; 0.004] | 0.478 [0.471; 0.487] | 0.234 [0.231; 0.239] |
| D3 | Faz A standart | 0.000 (SE 0.031) | nominal +-2SE icinde | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| D3 | duzgunlestirilmis | 0.067 [0.020; 0.118] (SE 0.031) | nominal +-2SE icinde | 0.538 [0.484; 0.589] | 0.535 [0.477; 0.600] | 0.551 [0.494; 0.612] | 0.337 [0.285; 0.377] | 0.537 [0.494; 0.577] | 0.441 [0.408; 0.479] | 0.490 [0.461; 0.515] |
| D3 | cross-conformal | 0.039 (SE 0.031) | nominal +-2SE icinde | 1.000 | 1.000 | 0.922 | 0.192 | 1.000 | 0.547 | 0.779 |
| D3 | cross + duzgunlestirme | 0.062 [0.039; 0.098] (SE 0.031) | nominal +-2SE icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.925 [0.922; 0.931] | 0.236 [0.212; 0.262] | 1.000 [1.000; 1.000] | 0.570 [0.558; 0.584] | 0.790 [0.785; 0.797] |
| D1orD2 (betimleyici, nominal<=2a) | Faz A standart | 0.078 (SE 0.042) | <=2a sinir icinde | 1.000 | 1.000 | 0.931 | 0.600 | 1.000 | 0.760 | 0.883 |
| D1orD2 (betimleyici, nominal<=2a) | duzgunlestirilmis | 0.122 [0.078; 0.176] (SE 0.042) | <=2a sinir icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.934 [0.931; 0.943] | 0.639 [0.623; 0.662] | 1.000 [1.000; 1.000] | 0.782 [0.772; 0.796] | 0.894 [0.889; 0.900] |
| D1orD2 (betimleyici, nominal<=2a) | cross-conformal | 0.039 (SE 0.042) | <=2a sinir icinde | 1.000 | 1.000 | 0.922 | 0.738 | 1.000 | 0.828 | 0.916 |
| D1orD2 (betimleyici, nominal<=2a) | cross + duzgunlestirme | 0.086 [0.039; 0.137] (SE 0.042) | <=2a sinir icinde | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.927 [0.922; 0.935] | 0.766 [0.750; 0.789] | 1.000 [1.000; 1.000] | 0.844 [0.836; 0.855] | 0.924 [0.920; 0.929] |

### CNN-GRU x pred, a = 0.05

| degisken | yontem | FAR | gecerlilik | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | Faz A standart | 0.000 (SE 0.031) | nominal +-2SE icinde | 0.007 | 0.400 | 0.147 | 0.081 | 0.184 | 0.111 | 0.147 |
| D1 | duzgunlestirilmis | 0.028 [0.000; 0.062] (SE 0.031) | nominal +-2SE icinde | 0.267 [0.229; 0.314] | 0.737 [0.696; 0.774] | 0.300 [0.278; 0.327] | 0.168 [0.149; 0.186] | 0.479 [0.447; 0.510] | 0.228 [0.215; 0.241] | 0.350 [0.334; 0.366] |
| D1 | cross-conformal | 0.000 (SE 0.031) | nominal +-2SE icinde | 0.011 | 0.600 | 0.310 | 0.136 | 0.276 | 0.215 | 0.245 |
| D1 | cross + duzgunlestirme | 0.038 [0.020; 0.082] (SE 0.031) | nominal +-2SE icinde | 0.222 [0.207; 0.236] | 0.990 [0.978; 1.000] | 0.372 [0.363; 0.384] | 0.214 [0.203; 0.224] | 0.569 [0.561; 0.580] | 0.285 [0.280; 0.293] | 0.423 [0.417; 0.429] |
| D2 | Faz A standart | 0.082 (SE 0.031) | nominal +-2SE icinde | 0.014 | 0.065 | 0.237 | 0.322 | 0.037 | 0.283 | 0.164 |
| D2 | duzgunlestirilmis | 0.104 [0.082; 0.122] (SE 0.031) | nominal +-2SE icinde | 0.025 [0.021; 0.029] | 0.090 [0.083; 0.096] | 0.361 [0.351; 0.371] | 0.586 [0.569; 0.600] | 0.054 [0.051; 0.057] | 0.484 [0.474; 0.493] | 0.275 [0.269; 0.280] |
| D2 | cross-conformal | 0.020 (SE 0.031) | nominal +-2SE icinde | 0.018 | 0.061 | 0.265 | 0.380 | 0.037 | 0.328 | 0.187 |
| D2 | cross + duzgunlestirme | 0.060 [0.041; 0.061] (SE 0.031) | nominal +-2SE icinde | 0.018 [0.018; 0.018] | 0.094 [0.087; 0.096] | 0.343 [0.335; 0.347] | 0.479 [0.468; 0.488] | 0.052 [0.049; 0.053] | 0.418 [0.411; 0.424] | 0.240 [0.237; 0.244] |
| D3 | Faz A standart | 0.000 (SE 0.031) | nominal +-2SE icinde | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| D3 | duzgunlestirilmis | 0.061 [0.020; 0.122] (SE 0.031) | nominal +-2SE icinde | 0.145 [0.111; 0.189] | 0.414 [0.361; 0.474] | 0.331 [0.286; 0.380] | 0.377 [0.325; 0.427] | 0.266 [0.235; 0.300] | 0.356 [0.318; 0.394] | 0.313 [0.289; 0.340] |
| D3 | cross-conformal | 0.000 (SE 0.031) | nominal +-2SE icinde | 0.014 | 0.400 | 0.237 | 0.207 | 0.188 | 0.220 | 0.205 |
| D3 | cross + duzgunlestirme | 0.029 [0.020; 0.041] (SE 0.031) | nominal +-2SE icinde | 0.125 [0.100; 0.150] | 0.805 [0.761; 0.843] | 0.567 [0.551; 0.588] | 0.474 [0.458; 0.492] | 0.431 [0.408; 0.455] | 0.516 [0.504; 0.528] | 0.475 [0.463; 0.488] |
| D1orD2 (betimleyici, nominal<=2a) | Faz A standart | 0.082 (SE 0.043) | <=2a sinir icinde | 0.021 | 0.409 | 0.359 | 0.376 | 0.196 | 0.369 | 0.285 |
| D1orD2 (betimleyici, nominal<=2a) | duzgunlestirilmis | 0.120 [0.102; 0.143] (SE 0.043) | <=2a sinir icinde | 0.285 [0.246; 0.329] | 0.751 [0.713; 0.787] | 0.605 [0.584; 0.629] | 0.694 [0.674; 0.715] | 0.495 [0.465; 0.524] | 0.653 [0.639; 0.669] | 0.577 [0.562; 0.591] |
| D1orD2 (betimleyici, nominal<=2a) | cross-conformal | 0.020 (SE 0.043) | <=2a sinir icinde | 0.029 | 0.626 | 0.563 | 0.495 | 0.298 | 0.526 | 0.415 |
| D1orD2 (betimleyici, nominal<=2a) | cross + duzgunlestirme | 0.088 [0.061; 0.102] (SE 0.043) | <=2a sinir icinde | 0.237 [0.221; 0.250] | 0.992 [0.978; 1.000] | 0.681 [0.669; 0.686] | 0.635 [0.620; 0.647] | 0.577 [0.569; 0.588] | 0.656 [0.648; 0.663] | 0.617 [0.612; 0.623] |

## 4. AUROC (kat ortalamasi)

std satirlari Faz A ile ayni (D1/D2 ham skor; 'p' satirlari 1-p tabanli). (ii) icin D1/D2 **ham skor** AUROC'u degismez (skor ayni); p-tabanli ve D3 satirlari duzgunlestirme rastgeleligi nedeniyle degisebilir ve 200 cekilis uzerinden ortalama [%2,5; %97,5] verilir. (iii) icin tum satirlar p-tabanlidir; AUROC kat basina yalniz ~10 normal donguye dayanir (SE ~0,03-0,05; 0,02'lik farklar anlamli degil).

### B2 x pred (ana)

| degisken | yontem | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|
| D1 (ham skor) | Faz A standart | 0.969 | 0.979 | 0.604 | 0.470 | 0.974 | 0.535 | 0.760 |
| D1 (1-p) | Faz A standart | 0.961 | 0.961 | 0.601 | 0.470 | 0.961 | 0.534 | 0.753 |
| D1 (1-p) | duzgunlestirilmis | 0.950 [0.916; 0.977] | 0.966 [0.947; 0.987] | 0.599 [0.537; 0.666] | 0.470 [0.383; 0.564] | 0.958 [0.932; 0.981] | 0.532 [0.464; 0.609] | 0.750 [0.714; 0.790] |
| D1 (1-p) | cross-conformal | 0.961 | 0.961 | 0.604 | 0.470 | 0.961 | 0.535 | 0.753 |
| D1 (1-p) | cross + duzgunlestirme | 0.955 [0.933; 0.979] | 0.968 [0.950; 0.985] | 0.599 [0.542; 0.666] | 0.469 [0.383; 0.563] | 0.961 [0.941; 0.981] | 0.532 [0.463; 0.609] | 0.752 [0.717; 0.790] |
| D2 (ham skor) | Faz A standart | 0.485 | 0.535 | 0.931 | 0.990 | 0.508 | 0.961 | 0.729 |
| D2 (1-p) | Faz A standart | 0.494 | 0.543 | 0.908 | 0.969 | 0.517 | 0.939 | 0.723 |
| D2 (1-p) | duzgunlestirilmis | 0.493 [0.484; 0.503] | 0.542 [0.533; 0.552] | 0.908 [0.891; 0.924] | 0.969 [0.951; 0.987] | 0.516 [0.507; 0.524] | 0.939 [0.922; 0.956] | 0.722 [0.712; 0.732] |
| D2 (1-p) | cross-conformal | 0.476 | 0.528 | 0.919 | 0.986 | 0.500 | 0.954 | 0.721 |
| D2 (1-p) | cross + duzgunlestirme | 0.476 [0.469; 0.482] | 0.527 [0.520; 0.534] | 0.919 [0.911; 0.927] | 0.986 [0.978; 0.995] | 0.500 [0.493; 0.505] | 0.954 [0.946; 0.961] | 0.721 [0.716; 0.726] |
| D3 | Faz A standart | 0.879 | 0.918 | 0.947 | 0.956 | 0.897 | 0.952 | 0.924 |
| D3 | duzgunlestirilmis | 0.878 [0.837; 0.911] | 0.931 [0.908; 0.953] | 0.943 [0.918; 0.964] | 0.951 [0.924; 0.976] | 0.902 [0.873; 0.929] | 0.947 [0.923; 0.970] | 0.924 [0.903; 0.946] |
| D3 | cross-conformal | 0.906 | 0.937 | 0.968 | 0.971 | 0.920 | 0.969 | 0.944 |
| D3 | cross + duzgunlestirme | 0.904 [0.881; 0.927] | 0.950 [0.933; 0.966] | 0.963 [0.948; 0.977] | 0.967 [0.954; 0.980] | 0.925 [0.907; 0.944] | 0.965 [0.951; 0.978] | 0.945 [0.931; 0.959] |

### B2 x tof

| degisken | yontem | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|
| D1 (ham skor) | Faz A standart | 0.970 | 0.970 | 0.460 | 0.579 | 0.970 | 0.521 | 0.751 |
| D1 (1-p) | Faz A standart | 0.960 | 0.960 | 0.460 | 0.579 | 0.960 | 0.521 | 0.746 |
| D1 (1-p) | duzgunlestirilmis | 0.958 [0.930; 0.980] | 0.959 [0.930; 0.980] | 0.462 [0.427; 0.494] | 0.577 [0.538; 0.619] | 0.959 [0.932; 0.979] | 0.522 [0.487; 0.551] | 0.745 [0.727; 0.767] |
| D1 (1-p) | cross-conformal | 0.960 | 0.960 | 0.457 | 0.573 | 0.960 | 0.517 | 0.744 |
| D1 (1-p) | cross + duzgunlestirme | 0.957 [0.929; 0.978] | 0.955 [0.925; 0.978] | 0.459 [0.425; 0.493] | 0.574 [0.533; 0.616] | 0.956 [0.927; 0.978] | 0.519 [0.484; 0.547] | 0.743 [0.720; 0.763] |
| D2 (ham skor) | Faz A standart | 0.496 | 0.543 | 0.901 | 0.985 | 0.518 | 0.944 | 0.726 |
| D2 (1-p) | Faz A standart | 0.492 | 0.533 | 0.893 | 0.970 | 0.511 | 0.933 | 0.717 |
| D2 (1-p) | duzgunlestirilmis | 0.492 [0.481; 0.501] | 0.533 [0.523; 0.543] | 0.892 [0.879; 0.909] | 0.970 [0.955; 0.986] | 0.511 [0.502; 0.520] | 0.932 [0.920; 0.947] | 0.716 [0.708; 0.726] |
| D2 (1-p) | cross-conformal | 0.485 | 0.534 | 0.901 | 0.977 | 0.507 | 0.940 | 0.718 |
| D2 (1-p) | cross + duzgunlestirme | 0.485 [0.478; 0.490] | 0.533 [0.528; 0.539] | 0.901 [0.894; 0.908] | 0.977 [0.969; 0.985] | 0.507 [0.502; 0.512] | 0.940 [0.932; 0.948] | 0.718 [0.713; 0.723] |
| D3 | Faz A standart | 0.932 | 0.935 | 0.955 | 0.956 | 0.933 | 0.956 | 0.944 |
| D3 | duzgunlestirilmis | 0.932 [0.907; 0.954] | 0.935 [0.910; 0.958] | 0.946 [0.923; 0.968] | 0.946 [0.923; 0.966] | 0.933 [0.909; 0.954] | 0.946 [0.924; 0.966] | 0.940 [0.919; 0.958] |
| D3 | cross-conformal | 0.916 | 0.927 | 0.967 | 0.959 | 0.921 | 0.963 | 0.942 |
| D3 | cross + duzgunlestirme | 0.918 [0.889; 0.941] | 0.931 [0.906; 0.952] | 0.962 [0.947; 0.978] | 0.952 [0.939; 0.965] | 0.924 [0.899; 0.947] | 0.957 [0.944; 0.971] | 0.940 [0.921; 0.957] |

### ToF oracle x tof

| degisken | yontem | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|
| D1 (ham skor) | Faz A standart | 1.000 | 1.000 | 0.929 | 0.500 | 1.000 | 0.708 | 0.857 |
| D1 (1-p) | Faz A standart | 1.000 | 1.000 | 0.929 | 0.500 | 1.000 | 0.708 | 0.857 |
| D1 (1-p) | duzgunlestirilmis | 0.978 [0.945; 1.000] | 0.978 [0.945; 1.000] | 0.910 [0.871; 0.941] | 0.496 [0.404; 0.579] | 0.978 [0.946; 1.000] | 0.697 [0.633; 0.751] | 0.841 [0.801; 0.877] |
| D1 (1-p) | cross-conformal | 1.000 | 1.000 | 0.929 | 0.500 | 1.000 | 0.708 | 0.857 |
| D1 (1-p) | cross + duzgunlestirme | 0.989 [0.964; 1.000] | 0.989 [0.965; 1.000] | 0.920 [0.885; 0.947] | 0.496 [0.404; 0.579] | 0.989 [0.965; 1.000] | 0.702 [0.644; 0.753] | 0.849 [0.814; 0.880] |
| D2 (ham skor) | Faz A standart | 0.244 | 0.230 | 0.789 | 0.967 | 0.238 | 0.881 | 0.552 |
| D2 (1-p) | Faz A standart | 0.252 | 0.237 | 0.790 | 0.961 | 0.245 | 0.878 | 0.554 |
| D2 (1-p) | duzgunlestirilmis | 0.252 [0.236; 0.267] | 0.236 [0.221; 0.254] | 0.789 [0.779; 0.799] | 0.960 [0.940; 0.983] | 0.245 [0.228; 0.260] | 0.877 [0.862; 0.893] | 0.553 [0.542; 0.563] |
| D2 (1-p) | cross-conformal | 0.229 | 0.194 | 0.794 | 0.976 | 0.213 | 0.888 | 0.542 |
| D2 (1-p) | cross + duzgunlestirme | 0.229 [0.223; 0.235] | 0.194 [0.188; 0.199] | 0.794 [0.786; 0.801] | 0.975 [0.960; 0.990] | 0.213 [0.208; 0.218] | 0.887 [0.876; 0.898] | 0.542 [0.536; 0.548] |
| D3 | Faz A standart | 0.962 | 0.962 | 0.952 | 0.961 | 0.962 | 0.957 | 0.959 |
| D3 | duzgunlestirilmis | 0.941 [0.899; 0.974] | 0.941 [0.901; 0.972] | 0.927 [0.884; 0.958] | 0.920 [0.866; 0.963] | 0.941 [0.899; 0.973] | 0.924 [0.880; 0.958] | 0.932 [0.889; 0.965] |
| D3 | cross-conformal | 0.980 | 0.980 | 0.968 | 0.976 | 0.980 | 0.972 | 0.976 |
| D3 | cross + duzgunlestirme | 0.968 [0.937; 0.989] | 0.969 [0.938; 0.991] | 0.950 [0.921; 0.974] | 0.941 [0.893; 0.978] | 0.969 [0.938; 0.991] | 0.945 [0.906; 0.975] | 0.957 [0.922; 0.982] |

### CNN-GRU x pred

| degisken | yontem | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|
| D1 (ham skor) | Faz A standart | 0.946 | 0.982 | 0.664 | 0.587 | 0.962 | 0.622 | 0.787 |
| D1 (1-p) | Faz A standart | 0.944 | 0.973 | 0.660 | 0.585 | 0.957 | 0.619 | 0.783 |
| D1 (1-p) | duzgunlestirilmis | 0.933 [0.903; 0.961] | 0.972 [0.947; 0.992] | 0.657 [0.597; 0.709] | 0.582 [0.507; 0.651] | 0.950 [0.924; 0.973] | 0.616 [0.548; 0.677] | 0.778 [0.743; 0.812] |
| D1 (1-p) | cross-conformal | 0.946 | 0.991 | 0.666 | 0.588 | 0.966 | 0.624 | 0.790 |
| D1 (1-p) | cross + duzgunlestirme | 0.937 [0.914; 0.960] | 0.991 [0.980; 1.000] | 0.664 [0.602; 0.715] | 0.585 [0.511; 0.652] | 0.961 [0.946; 0.977] | 0.620 [0.561; 0.680] | 0.786 [0.751; 0.819] |
| D2 (ham skor) | Faz A standart | 0.329 | 0.449 | 0.807 | 0.865 | 0.383 | 0.839 | 0.618 |
| D2 (1-p) | Faz A standart | 0.331 | 0.441 | 0.787 | 0.855 | 0.381 | 0.824 | 0.609 |
| D2 (1-p) | duzgunlestirilmis | 0.330 [0.320; 0.341] | 0.441 [0.429; 0.452] | 0.787 [0.771; 0.801] | 0.855 [0.835; 0.873] | 0.380 [0.371; 0.390] | 0.824 [0.807; 0.841] | 0.608 [0.597; 0.621] |
| D2 (1-p) | cross-conformal | 0.346 | 0.464 | 0.811 | 0.881 | 0.399 | 0.849 | 0.631 |
| D2 (1-p) | cross + duzgunlestirme | 0.346 [0.338; 0.354] | 0.464 [0.458; 0.470] | 0.811 [0.806; 0.816] | 0.881 [0.871; 0.891] | 0.399 [0.393; 0.406] | 0.850 [0.841; 0.856] | 0.631 [0.626; 0.635] |
| D3 | Faz A standart | 0.797 | 0.912 | 0.889 | 0.921 | 0.849 | 0.907 | 0.879 |
| D3 | duzgunlestirilmis | 0.829 [0.800; 0.857] | 0.927 [0.898; 0.953] | 0.886 [0.858; 0.913] | 0.916 [0.888; 0.944] | 0.873 [0.849; 0.896] | 0.902 [0.877; 0.930] | 0.888 [0.864; 0.912] |
| D3 | cross-conformal | 0.894 | 0.974 | 0.928 | 0.945 | 0.930 | 0.937 | 0.934 |
| D3 | cross + duzgunlestirme | 0.895 [0.875; 0.916] | 0.979 [0.967; 0.990] | 0.921 [0.909; 0.934] | 0.936 [0.923; 0.949] | 0.933 [0.919; 0.948] | 0.929 [0.917; 0.941] | 0.931 [0.919; 0.943] |

## 5. Desteklenen makale cumlesi icin sayilar (B2 x pred, D3)

| alpha | yontem | FAR | sira_sapmalari | tempo_sapmalari | tum_anomali | etiket |
|---|---|---|---|---|---|---|
| 0.100 | Faz A standart | 0.078 | 0.356 | 0.925 | 0.633 | nominal +-2SE icinde |
| 0.100 | duzgunlestirilmis | 0.087 [0.078; 0.098] | 0.571 [0.546; 0.598] | 0.934 [0.927; 0.939] | 0.748 [0.735; 0.762] | nominal +-2SE icinde |
| 0.100 | cross-conformal | 0.059 | 0.438 | 0.931 | 0.679 | nominal +-2SE icinde |
| 0.100 | cross + duzgunlestirme | 0.091 [0.059; 0.118] | 0.601 [0.575; 0.625] | 0.946 [0.939; 0.952] | 0.769 [0.756; 0.782] | nominal +-2SE icinde |
| 0.050 | Faz A standart | 0.000 | 0.000 | 0.000 | 0.000 | nominal +-2SE icinde |
| 0.050 | duzgunlestirilmis | 0.047 [0.000; 0.078] | 0.312 [0.279; 0.340] | 0.507 [0.463; 0.552] | 0.407 [0.380; 0.431] | nominal +-2SE icinde |
| 0.050 | cross-conformal | 0.039 | 0.267 | 0.824 | 0.539 | nominal +-2SE icinde |
| 0.050 | cross + duzgunlestirme | 0.041 [0.039; 0.059] | 0.402 [0.381; 0.423] | 0.846 [0.836; 0.857] | 0.619 [0.606; 0.631] | nominal +-2SE icinde |

Cumle kalibi: 'Esitlige duyarli cross-conformal kalibrasyonla (~40 normal dongu) iki kanalli detektor, dongu duzeyinde %Z yanlis alarmda sira sapmalarinin %X'ini, tempo sapmalarinin %Y'sini yakaliyor.' X, Y, Z yukaridaki 'cross + duzgunlestirme' ve 'cross-conformal' satirlarindan okunur; secici rapor yok: tum yontemler birlikte verilir.

## Karistirici ve uyarilar

- **Egitim kumesi boyutu (karistirici):** Faz A protokolunde parametreler ~20 normal donguden (2 kat), cross-conformal'da ~30 donguden (3 kat) ogrenilir (tablo 1: `std_egitim` ve `cross_egitim_model_basi`). Dolayisiyla (ii) -> (iii) farki hem **kalibrasyon kumesinin buyumesi (~20 -> ~40) hem de egitim kumesinin buyumesi**ni birlikte icerir; ikisi ayrilamaz. Yalniz (i) -> (ii) farki egitim kumesini sabit tutar ve esitlik duzeltmesini ayri olcer.
- **Cross-conformal kalibrasyon kumesi** dort ayri modelin (k = 1..4) kat-disi skorlarindan olusur; her skor kendi modelinin test skoruyla karsilastirilir (p'nin toplamasi bunu yapar). Teorik FAR siniri ~2a'dir; ampirik olarak ~a beklenir. Gozlenen FAR yukarida etiketlenmistir.
- **Dongu duzeyinde bagimsizlik:** normal dongular zaman icinde bagimsiz degildir (surukleme); komsu dongulerin farkli katlara dagilmasi kalibrasyonu iyimser yapabilir. Test normal dongusu sayisi ~50 oldugundan FAR'in SE'si ~0,04 (a = 0,10) ve ~0,03 (a = 0,05); %1-2'lik FAR farklari anlamli degil.
- Anomali donguleri her f'de test oldugu icin ayni anomali donguleri 5 kez (farkli kalibrasyonla) sayilir; tespit orani bu (dongu, f) ciftleri uzerinden ortalamadir.
- Dogal sapmali normal dongular (`natural_dev`) FAR'a dahildir; (FAR dogal sapma haric) bu depozitte verilmez.
- 'D1orD2' betimleyicidir: nominal FAR siniri 2a'dir, bir calisma noktasi olarak sunulamaz.


<!-- YORUM -->
## Yorum

(Bu bölüm elle yazıldı; betik yeniden koşulunca korunur. Tüm sayılar yukarıdaki tablolardan; seçici raporlama yok.)

**Kontroller.** (i) Faz A ile eşitlik: dört hücrede 168 sayının max |fark| ≤ 3,3e-16. Düzgünleştirilmiş p için analitik beklenen alarm olasılığı ile 200 çekilişli MC ortalaması arasındaki fark ≤ 0,003 (FAR ve tüm-anomali tespitinde, D1/D2). U = 1 durumu standart p'ye eşit (modül öz-sınaması).

**Ana hücre (B2 × pred, D3, α = 0,10; FAR ± SE ≈ 0,042, 51 normal test döngüsü).**

| yöntem | FAR | sıra (seq135+145) | seq135 / seq145 | tempo | tüm anomali |
|---|---|---|---|---|---|
| Faz A standart | 0,078 | 0,356 | 0,254 / 0,475 | 0,925 | 0,633 |
| düzgünleştirilmiş | 0,087 [0,078; 0,098] | 0,571 [0,546; 0,598] | 0,445 / 0,719 | 0,934 | 0,748 |
| cross-conformal | 0,059 | 0,438 | 0,257 / 0,650 | 0,931 | 0,679 |
| cross + düzgünleştirme | 0,091 [0,059; 0,118] | 0,601 [0,575; 0,625] | 0,377 / 0,862 | 0,946 [0,939; 0,952] | 0,769 |

α = 0,05'te Faz A standart D3 her yerde 0 (en küçük p ≈ 0,087-0,10 > 0,025). Düzgünleştirme ile FAR 0,047, sıra 0,312, tempo 0,507; cross-conformal FAR 0,039, sıra 0,267, tempo 0,824; cross + düzgünleştirme FAR 0,041, sıra 0,402, tempo 0,846.

**Ne söylenebilir.**
1. **Tespit artışı çözünürlük/eşitlik etkisidir, daha iyi skor değil.** Skorlar aynı (D1/D2 ham-skor AUROC'ları hiç değişmez; B2 × pred seq(135+145) D1 ham 0,974); düzgünleştirmede p-tabanlı D1/D2/D3 AUROC'ları Faz A ile MC aralığında aynı (B2 × pred tüm-anomali D3: 0,924 → 0,924 [0,903; 0,946]). Aynı skorla α = 0,10'da sıra tespiti 0,356 → 0,571 çıkıyor, çünkü FAR beklentisi yine ≤ α (0,087). D3'ün Bonferroni maliyeti (her kanal α/2'de) kalıyor.
2. **Cross-conformal iyileşmesi (i)→(iii) iki etkiyi karıştırır** (n_cal ≈ 20 → 40 ve eğitim ≈ 20 → 30 döngü; ayrılamaz). D3 tüm-anomali AUROC farkı (cross − standart) B2 × pred +0,020 (0,924 → 0,944), B2 × tof −0,003, ToF oracle +0,017, CNN-GRU × pred +0,055 (0,879 → 0,934); en büyüğü bile kat başına ~10 normal döngünün AUROC SE'sine (≈ 0,03-0,05) yakın, tutarlı bir yön de yok; "cross-conformal daha iyi skorluyor" diye yazılmamalı. D1 tespitindeki sıçrama (seq 0,694 → 1,000, FAR 0,059 → 0,078) bir sayım etkisi: p ≤ 0,10 için standartta ≤ 1, cross'ta ≤ 3 kalibrasyon skorunun anomaliye eşit/üstte olmasına izin veriliyor; D1 AUROC'u değişmedi (0,961 → 0,961).
3. **FAR geçerliliği.** Faz A kuralıyla (|FAR − α| ≤ 2 SE) tüm D3 satırları (4 hücre × 4 yöntem × 2 α) "nominal içinde" çıkıyor; yalnız D1 standart/cross ToF oracle × tof ve D1 standart CNN-GRU × pred (FAR 0) "muhafazakâr". Kural gevşektir: α = 0,05'te Faz A standart D3 FAR = 0,000 de "nominal içinde" etiketli, oysa alarm yapısal olarak imkânsız. Düzgünleştirilmiş/cross D3 beklenen FAR'ı 0,087-0,091 (α'nın altında; Bonferroni muhafazakârlığı). Çekiliş aralığı yalnız U rastgeleliğini gösterir; 51 normal döngü yüzünden asıl belirsizlik SE ≈ 0,042'dir. Aynı gevşeklik: CNN-GRU × pred D1 cross FAR 0,020 "nominal içinde" etiketli ama açıkça muhafazakâr. CNN-GRU × pred D3 düzgünleştirilmiş FAR 0,120 (α'nın üstünde, ±2 SE içinde); cross + düzgünleştirme 0,088.
4. **ToF oracle D1 dejenere.** Oracle'da tüm normal döngüler aynı sıraya (1-2-5) sahip → tüm kalibrasyon ve normal test D1 skorları eşit; Faz A'da FAR = 0 (muhafazakâr). Düzgünleştirmede normal döngülerin p'si = U olur: FAR ≈ α "tasarım gereği" (0,098), tespit yalnız "görülmemiş geçiş" olgusuna dayanır (seq = 1,000). smooth ve cross+smooth satırlarının aynı çıkması bu yüzdendir (aynı U'lar), hata değil.
5. **Hücreler arası (D3, α = 0,10, cross + düzgünleştirme; FAR / sıra / tempo):** B2 × pred 0,091 / 0,601 / 0,946; B2 × tof 0,093 / 0,640 / 0,889; ToF oracle × tof 0,086 / 1,000 / 0,844; CNN-GRU × pred 0,088 / 0,577 / 0,656 (Faz A standart: 0,082 / 0,196 / 0,369). Sıra sırası: oracle ≥ B2 ≈ CNN-GRU; tempoda B2 × pred 0,946 > B2 × tof 0,889 > oracle 0,844 (Faz A raporundaki uyarı geçerli: oracle tempo anomalileri için gerçek üst sınır değildir).
6. **D1orD2 (betimleyici).** α = 0,10'da B2 × pred FAR 0,118 (standart) ... 0,177 (düzgünleştirilmiş), nominal sınır 0,20; tespit 0,84-0,995. Bir çalışma noktası olarak sunulmaz; yalnız iki kanalın her birinin α'da neyi yakaladığını gösterir.

**Makale cümlesi için (ikincil, Faz A sayılarını değiştirmez).** Cross-conformal + düzgünleştirmeyle (~40 normal döngü, B2 × pred, α = 0,10): "döngü düzeyinde beklenen yanlış alarm ≈ %9'da (51 normal döngüde SE ≈ 4 puan) sıra sapmalarının ≈ %60'ını (seq135 %38, seq145 %86), tempo sapmalarının ≈ %95'ini yakalıyor". Faz A standart D3 aynı α'da %36 / %93'tür. Cümle "eşitlik düzeltmesi + daha büyük kalibrasyon/eğitim" etkisini birlikte taşıdığı için "daha iyi skor" diye sunulmamalı; seq135 (%38) açıkça zayıf kalıyor. α = 0,05 için aynı yöntemle FAR ≈ %4: sıra ≈ %40, tempo ≈ %85.

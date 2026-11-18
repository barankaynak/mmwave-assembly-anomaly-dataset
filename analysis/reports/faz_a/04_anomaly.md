# 04 — Dongu duzeyi anomali protokolu (Faz A)

`faz_a/04_cycle_anomaly.py` tarafindan uretilir; DESIGN.md 'Anomali tespiti' protokolu. Tum sayilar stdout'a da yazilir. Sapma yok (asagidaki 'Kesinlestirilen parametreler' DESIGN'da acik birakilan ayrintilardir).

## Protokol ve dongu anahtari
- Durum kaynaklari: **ToF oracle** (`y_true`) + S1 OOF `y_pred`: CNN-GRU, CNN-only, B1, B2.
- OOF'ta pencere = son kare; her oturumun ilk 4 karesinde pencere yok -> **none sayilir** (ilk kismi dongu zaten atilir).
- **Debounce** (kare sayisiyla): once ayni kutu arasindaki <= 1 karelik none boslugu birlestirilir, sonra < 2 karelik kutu run'i none yapilir; sabit noktaya kadar tekrarlanir. Ziyaret = kutu run'i; kalis = run suresi (kare zamanlarindan), gecis = ardisik ziyaretler arasi bosluk (>= 0).
- **Segmentasyon:** `pred` = debounce edilmis Box1 girisi -> bir sonraki Box1 girisi (ToF kullanilmaz); ilk (Box1 oncesi) ve son (kapanmamis) kismi dongu atilir. `tof` = `anom_loops` donguleri (`use` in normal_crossfit/anomaly_test_all_folds; kismi/tamamlanmamis donguler atilir).
- **Kat esleme (pred):** tahmini dongunun kati, Box1-giris karesini iceren ToF dongusunun katidir; o dongu kullanilmayan bir dongu ise (parca/tamamlanmamis) tahmini dongu atilir ve 'unmapped' sayilir. Etiket/natural_dev de o ToF dongusunden gelir.
- Oturum etiketleri: s02 normal; s03 seq135, s04 seq145 (sira anomalisi); s05 fast, s06 slow (tempo anomalisi); s01 kullanilmaz.
- **Dongu anahtari (05 ile hizalama):** `(state_source, segmentation, session, cycle_idx, start_frame_id)`; `segmentation=tof` icin cycle_idx = ToF loop_id, start_frame_id = loop'un ilk karesi; `segmentation=pred` icin cycle_idx = oturumdaki tahmini Box1 girisinin 0-tabanli sirasi (debounce sonrasi, atilanlar dahil), start_frame_id = giris karesi. `04_cycle_scores.csv.gz` satiri = (anahtar, f, role) ve `start_frame_id`/`end_frame_id` (dahil) ile `dur_s` icerir; 05 ayni dongu sinirlariyla OLD skorlarini `05_old_cycle_scores.csv.gz` icinde anahtar + `f` + `old_tft` + `old_z` olarak yazmalidir.

### Kesinlestirilen parametreler (DESIGN'da acik birakilanlar)
- **D0:** log(dongu suresi) icin |z|; mu, sigma s02 egitim donguleri; sigma tabani 0.05.
- **D1:** durumlar {1..5, END}, ek olarak bir **START** baglami (ilk ziyaretin olasiligi da puanlanir; pred segmentasyonda ilk ziyaret hep Box1'dir). Laplace alpha = 1: p = (sayi + 1)/(satir toplami + 6); egitimde hic gorulmeyen baglam -> duz 1/6. Skor = max −log p.
- **D2:** adim anahtarlari `dwell_i` ve `gap_i_j` (ardisik ziyaretler); log(d + 0.05 s) ile (sifir uzunluklu gecisler icin); referans = egitimde >= 3 gozlem, sigma tabani 0.05; referansi olmayan adim atlanir; hic referansli adimi olmayan dongunun D2 skoru **0** (`d2_nref` sutunu sayiyi tutar). Donguye baslangic/bitis boslugu (START→i, son→END) adim sayilmaz.
- **D3:** 2 · min(p1, p2) (Bonferroni), 1 ile sinirli; AUROC icin skor = 1 − p3.
- **Conformal:** p = (1 + #{cal >= s})/(n_cal + 1); alarm p <= alpha; cal = `anom_roles(f)['cal']` (2 kat, s02 normal). Esitlikte (cal >= s) muhafazakar tarafta kalir.

## Dongu sayilari (2×2 hucre × oturum)
Hucre = durum kaynagi × segmentasyon. Deger: **kullanilan dongu** (ToF referansi = `anom_loops` kullanilan dongu sayisi). pred segmentasyonda ayrica atilan: unmapped (Box1 girisi kullanilmayan bir ToF dongusunde) / acik son dongu.
| kaynak | segmentasyon | s02 | s03 | s04 | s05 | s06 | toplam |
|---|---|---|---|---|---|---|---|
| ToF oracle | tof | 51 (51) | 57 (57) | 49 (49) | 49 (49) | 52 (52) | 258 |
| ToF oracle | pred | 50 (51) | 56 (57) | 48 (49) | 46 (49) | 51 (52) | 251 |
| CNN-GRU | tof | 51 (51) | 57 (57) | 49 (49) | 49 (49) | 52 (52) | 258 |
| CNN-GRU | pred | 49 (51) | 56 (57) | 46 (49) | 49 (49) | 59 (52) | 259 |
| CNN-only | tof | 51 (51) | 57 (57) | 49 (49) | 49 (49) | 52 (52) | 258 |
| CNN-only | pred | 52 (51) | 56 (57) | 49 (49) | 47 (49) | 88 (52) | 292 |
| B1 | tof | 51 (51) | 57 (57) | 49 (49) | 49 (49) | 52 (52) | 258 |
| B1 | pred | 55 (51) | 60 (57) | 43 (49) | 52 (49) | 97 (52) | 307 |
| B2 | tof | 51 (51) | 57 (57) | 49 (49) | 49 (49) | 52 (52) | 258 |
| B2 | pred | 51 (51) | 56 (57) | 48 (49) | 48 (49) | 51 (52) | 254 |

### pred segmentasyon: ToF dongusu basina tahmini dongu sayisi (0 = kacirilan, >= 2 = bolunen) ve atilanlar
| kaynak | Box1 giris (toplam) | kullanilan | ToF ref | loop'ta 0 | loop'ta 1 | loop'ta >=2 | atilan unmapped | atilan acik-son |
|---|---|---|---|---|---|---|---|---|
| ToF oracle | 256 | 251 | 258 | 8 | 249 | 1 | 0 | 5 |
| CNN-GRU | 267 | 259 | 258 | 43 | 173 | 42 | 3 | 5 |
| CNN-only | 301 | 292 | 258 | 68 | 98 | 92 | 4 | 5 |
| B1 | 316 | 307 | 258 | 55 | 118 | 85 | 4 | 5 |
| B2 | 259 | 254 | 258 | 48 | 166 | 44 | 0 | 5 |

Not: ToF×ToF hucresi dogrudan `anom_loops` kullanilan dongu sayisidir (toplam 258; s02 normal = 51). Oturum basina ~50 dongu beklenir.

### Oracle (ToF) tanisi: `loop_id` sinirlari ile 10 Hz yumusatilmis etiket (oturum bazinda)
| oturum | kullanilan loop | hic Box1 karesi yok | hic Box5 karesi yok | ilk kare onceki loop'tan suren Box1 run'inda |
|---|---|---|---|---|
| 02 | 51 | 0 | 0 | 0 |
| 03 | 57 | 0 | 9 | 0 |
| 04 | 49 | 0 | 4 | 1 |
| 05 | 49 | 2 | 40 | 0 |
| 06 | 52 | 0 | 0 | 0 |
±3 kare mod yumusatma ve 10 Hz ornekleme ~<4 karelik dokunuslari siler; `anom_loops.box_order` ise 30 Hz ham ToF'tan gelir, bu yuzden kare-duzeyi dizi bazi donguler icin 1-2-5'ten (ya da oturumun beklenen dizisinden) eksik cikar; `loop_id` ayrica Box1 girisinde arttigindan Box1 run'i dongu sinirini asabilir. Sonuc (tablo): s05 (fast) oracle dizisi 49 dongunun cogunda 1-2 (Box5 kare duzeyinde gorunmuyor; olasi neden: cok kisa Box5 dokunusu yumusatmada siliniyor), s03/s04'te de az sayida 1-3 / 1-4 / (s04: 4) kisaltma var. Bu yuzden **D1'in fast AUROC'u (~0.93) bir hiz sinyali degil, kare-duzeyi kisaltma ('2'den sonra END') artefaktidir** ve D1'in tempo sutunlari 'sira' kanalinin yan etkisi olarak okunmali; s03/s04 icin ise sira anomalisi sinyali (1→3, 1→4, 3→5, 4→5 gecisleri egitimde hic yok) gecerlidir. ToF×pred hucresinde 'loop'ta 0' sayisi ayrica her oturumun son (kapanmamis) dongusunu icerir.

### Ziyaret dizisi dagilimi: oturum × hucre (tum durum kaynaklari; her oturum icin beklenen dizi: s02 1-2-5, s03 1-3-5, s04 1-4-5, s05/s06 1-2-5)
| kaynak | seg | oturum | n | beklenen dizi orani | farkli dizi | en sik 3 dizi |
|---|---|---|---|---|---|---|
| ToF oracle | tof | 02 | 51 | 1.00 | 1 | 1-2-5×51 |
| ToF oracle | tof | 03 | 57 | 0.84 | 2 | 1-3-5×48; 1-3×9 |
| ToF oracle | tof | 04 | 49 | 0.88 | 4 | 1-4-5×43; 1-4×4; 1-4-5-1×1 |
| ToF oracle | tof | 05 | 49 | 0.14 | 5 | 1-2×36; 1-2-5×7; 2×2 |
| ToF oracle | tof | 06 | 52 | 1.00 | 1 | 1-2-5×52 |
| ToF oracle | pred | 02 | 50 | 1.00 | 1 | 1-2-5×50 |
| ToF oracle | pred | 03 | 56 | 0.84 | 2 | 1-3-5×47; 1-3×9 |
| ToF oracle | pred | 04 | 48 | 0.92 | 2 | 1-4-5×44; 1-4×4 |
| ToF oracle | pred | 05 | 46 | 0.13 | 5 | 1-2×34; 1-2-5×6; 1-2-2×2 |
| ToF oracle | pred | 06 | 51 | 1.00 | 1 | 1-2-5×51 |
| CNN-GRU | tof | 02 | 51 | 0.49 | 8 | 1-2-5×25; 2-5×8; 1-2-5-1×7 |
| CNN-GRU | tof | 03 | 57 | 0.60 | 7 | 1-3-5×34; 1-3-5-1×8; 3-5×8 |
| CNN-GRU | tof | 04 | 49 | 0.69 | 7 | 1-4-5×34; 4-5×7; 1-4-5-1×3 |
| CNN-GRU | tof | 05 | 49 | 0.51 | 11 | 1-2-5×25; 1-2×9; 1-5×4 |
| CNN-GRU | tof | 06 | 52 | 0.23 | 12 | 2-5×13; 1-2-5×12; 1-2-5-1×8 |
| CNN-GRU | pred | 02 | 49 | 0.88 | 4 | 1-2-5×43; 1-2-2-5×4; 1-3-5×1 |
| CNN-GRU | pred | 03 | 56 | 0.95 | 4 | 1-3-5×53; 1-5×1; 1-3-3×1 |
| CNN-GRU | pred | 04 | 46 | 0.91 | 5 | 1-4-5×42; 1-4-5-4-5×1; 1-4-3-5×1 |
| CNN-GRU | pred | 05 | 49 | 0.59 | 5 | 1-2-5×29; 1-2×11; 1-5×5 |
| CNN-GRU | pred | 06 | 59 | 0.73 | 7 | 1-2-5×43; 1×9; 1-2-2-5×3 |
| CNN-only | tof | 02 | 51 | 0.14 | 11 | 1-2-5-1×14; 2-5×13; 2-5-1×7 |
| CNN-only | tof | 03 | 57 | 0.16 | 11 | 3-5-1×14; 1-3-5-1×14; 3-5×12 |
| CNN-only | tof | 04 | 49 | 0.06 | 12 | 1-4-5-1×13; 4-5×12; 4-5-1×11 |
| CNN-only | tof | 05 | 49 | 0.16 | 16 | 1-2-5-1×12; 2-5×9; 1-2-5×8 |
| CNN-only | tof | 06 | 52 | 0.00 | 25 | 2-5-1×8; 1-2-5-1×8; 3-2-5-1×4 |
| CNN-only | pred | 02 | 52 | 0.79 | 6 | 1-2-5×41; 1-2-2-5×6; 1×2 |
| CNN-only | pred | 03 | 56 | 0.88 | 5 | 1-3-5×49; 1-3-4-5×3; 1-3-3-5×2 |
| CNN-only | pred | 04 | 49 | 0.82 | 6 | 1-4-5×40; 1-4-4-5×3; 1-4-3-5×2 |
| CNN-only | pred | 05 | 47 | 0.64 | 9 | 1-2-5×30; 1-2-2-5×6; 1-2×4 |
| CNN-only | pred | 06 | 88 | 0.26 | 15 | 1×38; 1-2-5×23; 1-2-2-5×8 |
| B1 | tof | 02 | 51 | 0.12 | 34 | 1-2-5×6; 1-2-2-5×4; 2-2-5-1×3 |
| B1 | tof | 03 | 57 | 0.12 | 19 | 3-5-1×8; 1-3-5-1×7; 1-3-5×7 |
| B1 | tof | 04 | 49 | 0.12 | 23 | 4-5×9; 1-4-5×6; 1-4-5-1×5 |
| B1 | tof | 05 | 49 | 0.10 | 31 | 1-2-5-1×5; 1-2-5×5; 2-2-5-1×4 |
| B1 | tof | 06 | 52 | 0.06 | 36 | 1-1-2-5-1×5; 1-2-5-5-1×4; 1-2-5-5×4 |
| B1 | pred | 02 | 55 | 0.20 | 20 | 1-2-5×11; 1-2-2-5×9; 1×8 |
| B1 | pred | 03 | 60 | 0.42 | 12 | 1-3-5×25; 1-3-3-5×13; 1×8 |
| B1 | pred | 04 | 43 | 0.42 | 15 | 1-4-5×18; 1-4-4-5×4; 1-4-3-5×3 |
| B1 | pred | 05 | 52 | 0.27 | 16 | 1-2-5×14; 1-2-2-5×7; 1×6 |
| B1 | pred | 06 | 97 | 0.11 | 16 | 1×49; 1-2-5-5×11; 1-2-5×11 |
| B2 | tof | 02 | 51 | 0.37 | 6 | 1-2-5×19; 1-2-5-1×10; 2-5-1×9 |
| B2 | tof | 03 | 57 | 0.46 | 6 | 1-3-5×26; 1-3-5-1×11; 3-5×11 |
| B2 | tof | 04 | 49 | 0.63 | 5 | 1-4-5×31; 1-4-5-1×8; 4-5×8 |
| B2 | tof | 05 | 49 | 0.59 | 8 | 1-2-5×29; 1-2×9; 1-2-5-1×3 |
| B2 | tof | 06 | 52 | 0.15 | 5 | 2-5-1×17; 1-2-5-1×13; 2-5×13 |
| B2 | pred | 02 | 51 | 0.92 | 3 | 1-2-5×47; 1-2-2-5×3; 1-3-2-5×1 |
| B2 | pred | 03 | 56 | 0.96 | 3 | 1-3-5×54; 1-3-5-2×1; 1-3-3-5×1 |
| B2 | pred | 04 | 48 | 0.98 | 2 | 1-4-5×47; 1-4-5-5×1 |
| B2 | pred | 05 | 48 | 0.73 | 4 | 1-2-5×35; 1-2×11; 1-2-4×1 |
| B2 | pred | 06 | 51 | 0.98 | 2 | 1-2-5×50; 1-2-2-5×1 |

## Conformal ayrintisi: kalibrasyon boyutu ve ulasilabilir en kucuk p
| state_source | segmentation | n_cal_min | n_cal_mean | n_cal_max | n_train_mean | en kucuk p (1/(n+1)), n=min | D3 en kucuk p (2/(n+1)), n=min |
|---|---|---|---|---|---|---|---|
| B1 | pred | 17 | 22.000 | 26 | 22.000 | 0.056 | 0.111 |
| B1 | tof | 20 | 20.400 | 21 | 20.400 | 0.048 | 0.095 |
| B2 | pred | 19 | 20.400 | 22 | 20.400 | 0.050 | 0.100 |
| B2 | tof | 20 | 20.400 | 21 | 20.400 | 0.048 | 0.095 |
| CNN-GRU | pred | 18 | 19.600 | 22 | 19.600 | 0.053 | 0.105 |
| CNN-GRU | tof | 20 | 20.400 | 21 | 20.400 | 0.048 | 0.095 |
| CNN-only | pred | 19 | 20.800 | 23 | 20.800 | 0.050 | 0.100 |
| CNN-only | tof | 20 | 20.400 | 21 | 20.400 | 0.048 | 0.095 |
| ToF oracle | pred | 19 | 20.000 | 21 | 20.000 | 0.050 | 0.100 |
| ToF oracle | tof | 20 | 20.400 | 21 | 20.400 | 0.048 | 0.095 |

Yapisal uyari (on-kayitli kural degistirilmedi): D3 = 2·min(p1, p2) oldugundan alpha'da alarm icin min(p1, p2) <= alpha/2 gerekir. n_cal ≈ 20'de en kucuk p ≈ 0.048; bu yuzden **D3 alpha = 0.10'da (esik 0.05) yalniz kalibrasyondaki tum skorlari asan donguler icin alarm verebilir ve alpha = 0.05'te (esik 0.025) hic alarm veremez** (gerekli n_cal >= 39; burada min n_cal = 17). n_cal < 19 olan katlarda (ornegin B1×pred) D3 alpha = 0.10'da da hic alarm veremez (en kucuk D3 p = 2/(n_cal+1) > 0.10). D0 skoru tof segmentasyonunda durum kaynagindan bagimsizdir (dongu suresi ToF dongu sinirlarindan gelir), bu yuzden o hucrelerde D0 satirlari ozdestir. D1/D2/D0 tek basina alpha = 0.05'te alarm verebilir (p = 1/(n_cal+1) <= 0.05 icin n_cal >= 19). p-degerleri ayriktir ve cok sayida esitlik icerir (ozellikle D1: normal dongulerin cogu ayni sirayi paylasir) -> D3'un AUROC'u kaba cozunurludur; AUPRC/AUROC'ta esitlikler 0.5 sayilir.

## AUROC (normal test donguleri vs anomali turu; kat bazli ortalama ± sd, 5 kat)
Her katta ~10.2 normal test dongusu (anomali: tip basina ~57); AUROC SE ≈ 0.03–0.05 oldugundan 0.02 farklari cozulemez.

### Segmentasyon = ToF dongu sinirlari (`tof`)
| kaynak | skor | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|
| ToF oracle | D0 sure | 0.595 ± 0.081 | 0.637 ± 0.088 | 0.894 ± 0.050 | 0.975 ± 0.043 | 0.614 ± 0.084 | 0.936 ± 0.042 | 0.771 ± 0.054 |
| ToF oracle | D1 Markov | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.929 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.708 ± 0.000 | 0.857 ± 0.000 |
| ToF oracle | D2 adim-suresi | 0.244 ± 0.076 | 0.230 ± 0.084 | 0.789 ± 0.071 | 0.967 ± 0.065 | 0.238 ± 0.078 | 0.881 ± 0.053 | 0.552 ± 0.046 |
| ToF oracle | D3 min-p | 0.962 ± 0.052 | 0.962 ± 0.052 | 0.952 ± 0.050 | 0.961 ± 0.051 | 0.962 ± 0.052 | 0.957 ± 0.051 | 0.959 ± 0.052 |
| ToF oracle | OLD TFT | 0.512 ± 0.184 | 0.550 ± 0.209 | 0.130 ± 0.059 | 0.894 ± 0.118 | 0.529 ± 0.194 | 0.532 ± 0.069 | 0.531 ± 0.084 |
| ToF oracle | OLD Z | 0.433 ± 0.047 | 0.372 ± 0.047 | 0.670 ± 0.112 | 0.907 ± 0.072 | 0.405 ± 0.045 | 0.792 ± 0.073 | 0.594 ± 0.056 |
| ToF oracle | OLD min-p | 0.565 ± 0.135 | 0.565 ± 0.158 | 0.584 ± 0.138 | 0.878 ± 0.090 | 0.565 ± 0.145 | 0.739 ± 0.099 | 0.649 ± 0.067 |
| CNN-GRU | D0 sure | 0.595 ± 0.081 | 0.637 ± 0.088 | 0.894 ± 0.050 | 0.975 ± 0.043 | 0.614 ± 0.084 | 0.936 ± 0.042 | 0.771 ± 0.054 |
| CNN-GRU | D1 Markov | 0.951 ± 0.049 | 0.989 ± 0.024 | 0.571 ± 0.054 | 0.654 ± 0.068 | 0.969 ± 0.029 | 0.613 ± 0.061 | 0.795 ± 0.034 |
| CNN-GRU | D2 adim-suresi | 0.340 ± 0.092 | 0.446 ± 0.074 | 0.824 ± 0.061 | 0.943 ± 0.033 | 0.389 ± 0.083 | 0.885 ± 0.046 | 0.631 ± 0.062 |
| CNN-GRU | D3 min-p | 0.908 ± 0.082 | 0.981 ± 0.026 | 0.913 ± 0.020 | 0.938 ± 0.023 | 0.942 ± 0.042 | 0.926 ± 0.021 | 0.934 ± 0.026 |
| CNN-GRU | OLD TFT | 0.663 ± 0.322 | 0.450 ± 0.383 | 0.454 ± 0.226 | 0.683 ± 0.100 | 0.565 ± 0.286 | 0.574 ± 0.099 | 0.569 ± 0.186 |
| CNN-GRU | OLD Z | 0.328 ± 0.066 | 0.356 ± 0.064 | 0.650 ± 0.147 | 0.911 ± 0.039 | 0.341 ± 0.062 | 0.784 ± 0.087 | 0.557 ± 0.054 |
| CNN-GRU | OLD min-p | 0.583 ± 0.295 | 0.449 ± 0.202 | 0.571 ± 0.135 | 0.872 ± 0.039 | 0.521 ± 0.219 | 0.729 ± 0.061 | 0.623 ± 0.122 |
| CNN-only | D0 sure | 0.595 ± 0.081 | 0.637 ± 0.088 | 0.894 ± 0.050 | 0.975 ± 0.043 | 0.614 ± 0.084 | 0.936 ± 0.042 | 0.771 ± 0.054 |
| CNN-only | D1 Markov | 0.936 ± 0.064 | 0.944 ± 0.053 | 0.553 ± 0.061 | 0.774 ± 0.082 | 0.940 ± 0.058 | 0.667 ± 0.070 | 0.807 ± 0.061 |
| CNN-only | D2 adim-suresi | 0.324 ± 0.075 | 0.389 ± 0.094 | 0.756 ± 0.106 | 0.877 ± 0.060 | 0.354 ± 0.084 | 0.818 ± 0.082 | 0.581 ± 0.081 |
| CNN-only | D3 min-p | 0.867 ± 0.105 | 0.876 ± 0.105 | 0.769 ± 0.114 | 0.883 ± 0.062 | 0.871 ± 0.105 | 0.828 ± 0.087 | 0.850 ± 0.087 |
| B1 | D0 sure | 0.595 ± 0.081 | 0.637 ± 0.088 | 0.894 ± 0.050 | 0.975 ± 0.043 | 0.614 ± 0.084 | 0.936 ± 0.042 | 0.771 ± 0.054 |
| B1 | D1 Markov | 0.838 ± 0.090 | 0.898 ± 0.084 | 0.535 ± 0.046 | 0.642 ± 0.052 | 0.865 ± 0.076 | 0.590 ± 0.043 | 0.731 ± 0.029 |
| B1 | D2 adim-suresi | 0.631 ± 0.228 | 0.393 ± 0.292 | 0.442 ± 0.045 | 0.749 ± 0.134 | 0.521 ± 0.207 | 0.600 ± 0.089 | 0.560 ± 0.099 |
| B1 | D3 min-p | 0.772 ± 0.114 | 0.800 ± 0.082 | 0.540 ± 0.081 | 0.730 ± 0.120 | 0.785 ± 0.088 | 0.638 ± 0.097 | 0.713 ± 0.077 |
| B2 | D0 sure | 0.595 ± 0.081 | 0.637 ± 0.088 | 0.894 ± 0.050 | 0.975 ± 0.043 | 0.614 ± 0.084 | 0.936 ± 0.042 | 0.771 ± 0.054 |
| B2 | D1 Markov | 0.970 ± 0.041 | 0.970 ± 0.042 | 0.460 ± 0.075 | 0.579 ± 0.119 | 0.970 ± 0.042 | 0.521 ± 0.097 | 0.751 ± 0.050 |
| B2 | D2 adim-suresi | 0.496 ± 0.069 | 0.543 ± 0.091 | 0.901 ± 0.034 | 0.985 ± 0.015 | 0.518 ± 0.079 | 0.944 ± 0.023 | 0.726 ± 0.052 |
| B2 | D3 min-p | 0.932 ± 0.073 | 0.935 ± 0.069 | 0.955 ± 0.028 | 0.956 ± 0.027 | 0.933 ± 0.071 | 0.956 ± 0.027 | 0.944 ± 0.048 |

### Segmentasyon = tahmini Box1 girisleri (`pred`)
| kaynak | skor | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|
| ToF oracle | D0 sure | 0.576 ± 0.107 | 0.605 ± 0.101 | 0.893 ± 0.038 | 1.000 ± 0.000 | 0.589 ± 0.104 | 0.949 ± 0.018 | 0.763 ± 0.063 |
| ToF oracle | D1 Markov | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.935 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.706 ± 0.000 | 0.858 ± 0.000 |
| ToF oracle | D2 adim-suresi | 0.282 ± 0.097 | 0.299 ± 0.111 | 0.803 ± 0.064 | 0.986 ± 0.022 | 0.290 ± 0.103 | 0.899 ± 0.030 | 0.584 ± 0.064 |
| ToF oracle | D3 min-p | 0.971 ± 0.042 | 0.971 ± 0.042 | 0.960 ± 0.041 | 0.970 ± 0.041 | 0.971 ± 0.042 | 0.965 ± 0.041 | 0.968 ± 0.041 |
| ToF oracle | OLD TFT | 0.513 ± 0.260 | 0.550 ± 0.298 | 0.086 ± 0.076 | 0.854 ± 0.205 | 0.530 ± 0.277 | 0.496 ± 0.105 | 0.514 ± 0.149 |
| ToF oracle | OLD Z | 0.454 ± 0.064 | 0.404 ± 0.087 | 0.656 ± 0.103 | 0.932 ± 0.053 | 0.431 ± 0.074 | 0.801 ± 0.062 | 0.609 ± 0.067 |
| ToF oracle | OLD min-p | 0.627 ± 0.214 | 0.617 ± 0.253 | 0.597 ± 0.105 | 0.917 ± 0.068 | 0.623 ± 0.231 | 0.768 ± 0.066 | 0.692 ± 0.131 |
| CNN-GRU | D0 sure | 0.606 ± 0.049 | 0.633 ± 0.044 | 0.910 ± 0.014 | 0.992 ± 0.005 | 0.618 ± 0.047 | 0.955 ± 0.008 | 0.791 ± 0.026 |
| CNN-GRU | D1 Markov | 0.946 ± 0.058 | 0.982 ± 0.025 | 0.664 ± 0.027 | 0.587 ± 0.031 | 0.962 ± 0.040 | 0.622 ± 0.030 | 0.787 ± 0.033 |
| CNN-GRU | D2 adim-suresi | 0.329 ± 0.094 | 0.449 ± 0.066 | 0.807 ± 0.047 | 0.865 ± 0.042 | 0.383 ± 0.082 | 0.839 ± 0.044 | 0.618 ± 0.059 |
| CNN-GRU | D3 min-p | 0.797 ± 0.147 | 0.912 ± 0.081 | 0.889 ± 0.055 | 0.921 ± 0.051 | 0.849 ± 0.104 | 0.907 ± 0.053 | 0.879 ± 0.076 |
| CNN-GRU | OLD TFT | 0.800 ± 0.150 | 0.546 ± 0.337 | 0.334 ± 0.190 | 0.579 ± 0.080 | 0.684 ± 0.161 | 0.472 ± 0.058 | 0.575 ± 0.095 |
| CNN-GRU | OLD Z | 0.329 ± 0.094 | 0.388 ± 0.080 | 0.765 ± 0.057 | 0.862 ± 0.036 | 0.356 ± 0.087 | 0.818 ± 0.026 | 0.593 ± 0.047 |
| CNN-GRU | OLD min-p | 0.712 ± 0.246 | 0.566 ± 0.238 | 0.650 ± 0.095 | 0.824 ± 0.034 | 0.645 ± 0.218 | 0.748 ± 0.043 | 0.698 ± 0.102 |
| CNN-only | D0 sure | 0.496 ± 0.227 | 0.483 ± 0.260 | 0.569 ± 0.418 | 0.954 ± 0.081 | 0.490 ± 0.242 | 0.820 ± 0.148 | 0.676 ± 0.189 |
| CNN-only | D1 Markov | 0.926 ± 0.095 | 0.956 ± 0.044 | 0.588 ± 0.059 | 0.788 ± 0.048 | 0.940 ± 0.068 | 0.718 ± 0.052 | 0.815 ± 0.051 |
| CNN-only | D2 adim-suresi | 0.271 ± 0.067 | 0.340 ± 0.082 | 0.731 ± 0.121 | 0.792 ± 0.085 | 0.303 ± 0.073 | 0.771 ± 0.097 | 0.566 ± 0.080 |
| CNN-only | D3 min-p | 0.826 ± 0.114 | 0.854 ± 0.108 | 0.762 ± 0.103 | 0.869 ± 0.069 | 0.839 ± 0.107 | 0.831 ± 0.080 | 0.835 ± 0.090 |
| B1 | D0 sure | 0.439 ± 0.068 | 0.396 ± 0.079 | 0.300 ± 0.162 | 0.875 ± 0.062 | 0.421 ± 0.072 | 0.674 ± 0.073 | 0.571 ± 0.069 |
| B1 | D1 Markov | 0.716 ± 0.074 | 0.924 ± 0.061 | 0.502 ± 0.045 | 0.478 ± 0.074 | 0.803 ± 0.056 | 0.486 ± 0.063 | 0.616 ± 0.029 |
| B1 | D2 adim-suresi | 0.660 ± 0.138 | 0.275 ± 0.112 | 0.409 ± 0.105 | 0.441 ± 0.135 | 0.499 ± 0.123 | 0.429 ± 0.120 | 0.458 ± 0.121 |
| B1 | D3 min-p | 0.675 ± 0.122 | 0.728 ± 0.251 | 0.470 ± 0.114 | 0.405 ± 0.138 | 0.697 ± 0.154 | 0.428 ± 0.127 | 0.538 ± 0.134 |
| B2 | D0 sure | 0.592 ± 0.088 | 0.630 ± 0.068 | 0.906 ± 0.028 | 1.000 ± 0.000 | 0.610 ± 0.078 | 0.954 ± 0.013 | 0.778 ± 0.047 |
| B2 | D1 Markov | 0.969 ± 0.045 | 0.979 ± 0.029 | 0.604 ± 0.030 | 0.470 ± 0.039 | 0.974 ± 0.037 | 0.535 ± 0.035 | 0.760 ± 0.029 |
| B2 | D2 adim-suresi | 0.485 ± 0.067 | 0.535 ± 0.068 | 0.931 ± 0.028 | 0.990 ± 0.010 | 0.508 ± 0.067 | 0.961 ± 0.015 | 0.729 ± 0.041 |
| B2 | D3 min-p | 0.879 ± 0.078 | 0.918 ± 0.046 | 0.947 ± 0.029 | 0.956 ± 0.022 | 0.897 ± 0.051 | 0.952 ± 0.025 | 0.924 ± 0.035 |

## AUPRC (pozitif = anomali; kat bazli ortalama ± sd)
Taban cizgisi (prevalans, rastgele skor AUPRC'si) tur basina: seq135=0.85, seq145=0.82, fast=0.83, slow=0.85, seq(135+145)=0.91, tempo(fast+slow)=0.91, tum anomali=0.95. Normal azinlikta oldugundan AUPRC yuksek gorunur; AUROC ile birlikte okunmali.

### Segmentasyon = ToF dongu sinirlari (`tof`)
| kaynak | skor | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|
| ToF oracle | D0 sure | 0.895 ± 0.030 | 0.898 ± 0.034 | 0.968 ± 0.031 | 0.985 ± 0.030 | 0.943 ± 0.019 | 0.987 ± 0.018 | 0.983 ± 0.009 |
| ToF oracle | D1 Markov | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.975 ± 0.001 | 0.836 ± 0.006 | 1.000 ± 0.000 | 0.946 ± 0.002 | 0.987 ± 0.001 |
| ToF oracle | D2 adim-suresi | 0.765 ± 0.034 | 0.722 ± 0.036 | 0.939 ± 0.023 | 0.983 ± 0.035 | 0.847 ± 0.026 | 0.983 ± 0.012 | 0.968 ± 0.006 |
| ToF oracle | D3 min-p | 0.986 ± 0.019 | 0.984 ± 0.021 | 0.982 ± 0.022 | 0.985 ± 0.020 | 0.993 ± 0.010 | 0.991 ± 0.011 | 0.996 ± 0.005 |
| ToF oracle | OLD TFT | 0.870 ± 0.059 | 0.859 ± 0.073 | 0.702 ± 0.045 | 0.964 ± 0.045 | 0.924 ± 0.036 | 0.933 ± 0.024 | 0.966 ± 0.012 |
| ToF oracle | OLD Z | 0.848 ± 0.029 | 0.784 ± 0.040 | 0.875 ± 0.054 | 0.964 ± 0.037 | 0.898 ± 0.020 | 0.963 ± 0.021 | 0.966 ± 0.011 |
| ToF oracle | OLD min-p | 0.875 ± 0.047 | 0.855 ± 0.058 | 0.849 ± 0.054 | 0.951 ± 0.037 | 0.925 ± 0.031 | 0.951 ± 0.023 | 0.968 ± 0.010 |
| CNN-GRU | D0 sure | 0.895 ± 0.030 | 0.898 ± 0.034 | 0.968 ± 0.031 | 0.985 ± 0.030 | 0.943 ± 0.019 | 0.987 ± 0.018 | 0.983 ± 0.009 |
| CNN-GRU | D1 Markov | 0.988 ± 0.012 | 0.995 ± 0.011 | 0.888 ± 0.010 | 0.895 ± 0.022 | 0.995 ± 0.005 | 0.940 ± 0.009 | 0.985 ± 0.003 |
| CNN-GRU | D2 adim-suresi | 0.792 ± 0.035 | 0.842 ± 0.027 | 0.960 ± 0.016 | 0.986 ± 0.010 | 0.898 ± 0.019 | 0.986 ± 0.006 | 0.976 ± 0.005 |
| CNN-GRU | D3 min-p | 0.976 ± 0.018 | 0.992 ± 0.011 | 0.975 ± 0.006 | 0.981 ± 0.009 | 0.991 ± 0.006 | 0.988 ± 0.004 | 0.995 ± 0.002 |
| CNN-GRU | OLD TFT | 0.906 ± 0.100 | 0.815 ± 0.138 | 0.813 ± 0.060 | 0.917 ± 0.027 | 0.929 ± 0.059 | 0.934 ± 0.011 | 0.967 ± 0.017 |
| CNN-GRU | OLD Z | 0.809 ± 0.021 | 0.788 ± 0.029 | 0.889 ± 0.052 | 0.971 ± 0.021 | 0.885 ± 0.014 | 0.967 ± 0.016 | 0.966 ± 0.007 |
| CNN-GRU | OLD min-p | 0.893 ± 0.071 | 0.833 ± 0.052 | 0.870 ± 0.032 | 0.961 ± 0.015 | 0.926 ± 0.034 | 0.960 ± 0.005 | 0.971 ± 0.008 |
| CNN-only | D0 sure | 0.895 ± 0.030 | 0.898 ± 0.034 | 0.968 ± 0.031 | 0.985 ± 0.030 | 0.943 ± 0.019 | 0.987 ± 0.018 | 0.983 ± 0.009 |
| CNN-only | D1 Markov | 0.981 ± 0.017 | 0.978 ± 0.021 | 0.857 ± 0.027 | 0.931 ± 0.025 | 0.989 ± 0.010 | 0.945 ± 0.014 | 0.984 ± 0.006 |
| CNN-only | D2 adim-suresi | 0.794 ± 0.041 | 0.792 ± 0.048 | 0.927 ± 0.040 | 0.960 ± 0.026 | 0.880 ± 0.029 | 0.972 ± 0.016 | 0.967 ± 0.011 |
| CNN-only | D3 min-p | 0.962 ± 0.029 | 0.958 ± 0.030 | 0.927 ± 0.032 | 0.960 ± 0.020 | 0.978 ± 0.016 | 0.971 ± 0.014 | 0.987 ± 0.007 |
| B1 | D0 sure | 0.895 ± 0.030 | 0.898 ± 0.034 | 0.968 ± 0.031 | 0.985 ± 0.030 | 0.943 ± 0.019 | 0.987 ± 0.018 | 0.983 ± 0.009 |
| B1 | D1 Markov | 0.947 ± 0.027 | 0.961 ± 0.031 | 0.847 ± 0.016 | 0.891 ± 0.026 | 0.974 ± 0.016 | 0.927 ± 0.013 | 0.975 ± 0.009 |
| B1 | D2 adim-suresi | 0.899 ± 0.075 | 0.808 ± 0.104 | 0.840 ± 0.029 | 0.922 ± 0.053 | 0.927 ± 0.046 | 0.938 ± 0.025 | 0.967 ± 0.013 |
| B1 | D3 min-p | 0.925 ± 0.039 | 0.932 ± 0.026 | 0.847 ± 0.040 | 0.906 ± 0.046 | 0.959 ± 0.020 | 0.934 ± 0.024 | 0.972 ± 0.010 |
| B2 | D0 sure | 0.895 ± 0.030 | 0.898 ± 0.034 | 0.968 ± 0.031 | 0.985 ± 0.030 | 0.943 ± 0.019 | 0.987 ± 0.018 | 0.983 ± 0.009 |
| B2 | D1 Markov | 0.990 ± 0.014 | 0.988 ± 0.016 | 0.858 ± 0.012 | 0.862 ± 0.030 | 0.994 ± 0.008 | 0.918 ± 0.013 | 0.981 ± 0.004 |
| B2 | D2 adim-suresi | 0.874 ± 0.030 | 0.875 ± 0.049 | 0.978 ± 0.012 | 0.997 ± 0.003 | 0.931 ± 0.022 | 0.994 ± 0.003 | 0.984 ± 0.004 |
| B2 | D3 min-p | 0.978 ± 0.023 | 0.975 ± 0.026 | 0.983 ± 0.011 | 0.984 ± 0.010 | 0.988 ± 0.013 | 0.992 ± 0.005 | 0.995 ± 0.005 |

### Segmentasyon = tahmini Box1 girisleri (`pred`)
| kaynak | skor | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) | tum anomali |
|---|---|---|---|---|---|---|---|---|
| ToF oracle | D0 sure | 0.895 ± 0.035 | 0.893 ± 0.036 | 0.977 ± 0.008 | 1.000 ± 0.000 | 0.942 ± 0.020 | 0.995 ± 0.002 | 0.986 ± 0.005 |
| ToF oracle | D1 Markov | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.977 ± 0.001 | 0.836 ± 0.010 | 1.000 ± 0.000 | 0.945 ± 0.004 | 0.987 ± 0.001 |
| ToF oracle | D2 adim-suresi | 0.779 ± 0.046 | 0.746 ± 0.056 | 0.942 ± 0.021 | 0.997 ± 0.005 | 0.859 ± 0.035 | 0.988 ± 0.004 | 0.972 ± 0.006 |
| ToF oracle | D3 min-p | 0.990 ± 0.015 | 0.988 ± 0.018 | 0.984 ± 0.019 | 0.988 ± 0.017 | 0.994 ± 0.008 | 0.993 ± 0.009 | 0.997 ± 0.004 |
| ToF oracle | OLD TFT | 0.865 ± 0.087 | 0.853 ± 0.109 | 0.669 ± 0.052 | 0.948 ± 0.083 | 0.919 ± 0.058 | 0.922 ± 0.043 | 0.961 ± 0.021 |
| ToF oracle | OLD Z | 0.860 ± 0.033 | 0.796 ± 0.051 | 0.867 ± 0.055 | 0.981 ± 0.018 | 0.907 ± 0.025 | 0.971 ± 0.013 | 0.971 ± 0.009 |
| ToF oracle | OLD min-p | 0.897 ± 0.060 | 0.871 ± 0.085 | 0.854 ± 0.046 | 0.969 ± 0.026 | 0.936 ± 0.040 | 0.960 ± 0.014 | 0.973 ± 0.012 |
| CNN-GRU | D0 sure | 0.903 ± 0.022 | 0.896 ± 0.023 | 0.982 ± 0.004 | 0.999 ± 0.001 | 0.945 ± 0.013 | 0.996 ± 0.001 | 0.988 ± 0.003 |
| CNN-GRU | D1 Markov | 0.981 ± 0.019 | 0.991 ± 0.012 | 0.895 ± 0.021 | 0.888 ± 0.024 | 0.992 ± 0.008 | 0.941 ± 0.013 | 0.981 ± 0.006 |
| CNN-GRU | D2 adim-suresi | 0.794 ± 0.041 | 0.840 ± 0.037 | 0.956 ± 0.016 | 0.971 ± 0.018 | 0.898 ± 0.025 | 0.982 ± 0.009 | 0.975 ± 0.007 |
| CNN-GRU | D3 min-p | 0.945 ± 0.047 | 0.961 ± 0.037 | 0.960 ± 0.026 | 0.974 ± 0.021 | 0.973 ± 0.022 | 0.983 ± 0.012 | 0.989 ± 0.009 |
| CNN-GRU | OLD TFT | 0.922 ± 0.075 | 0.839 ± 0.111 | 0.776 ± 0.084 | 0.898 ± 0.035 | 0.941 ± 0.047 | 0.920 ± 0.027 | 0.964 ± 0.021 |
| CNN-GRU | OLD Z | 0.823 ± 0.051 | 0.814 ± 0.042 | 0.942 ± 0.024 | 0.976 ± 0.011 | 0.898 ± 0.028 | 0.981 ± 0.007 | 0.975 ± 0.007 |
| CNN-GRU | OLD min-p | 0.913 ± 0.081 | 0.865 ± 0.070 | 0.907 ± 0.025 | 0.964 ± 0.014 | 0.942 ± 0.042 | 0.970 ± 0.008 | 0.979 ± 0.010 |
| CNN-only | D0 sure | 0.849 ± 0.088 | 0.826 ± 0.110 | 0.833 ± 0.162 | 0.988 ± 0.025 | 0.906 ± 0.059 | 0.979 ± 0.019 | 0.978 ± 0.016 |
| CNN-only | D1 Markov | 0.978 ± 0.025 | 0.984 ± 0.016 | 0.857 ± 0.037 | 0.955 ± 0.016 | 0.990 ± 0.011 | 0.961 ± 0.012 | 0.985 ± 0.007 |
| CNN-only | D2 adim-suresi | 0.764 ± 0.061 | 0.764 ± 0.075 | 0.913 ± 0.050 | 0.958 ± 0.023 | 0.858 ± 0.045 | 0.971 ± 0.016 | 0.968 ± 0.012 |
| CNN-only | D3 min-p | 0.948 ± 0.037 | 0.951 ± 0.034 | 0.912 ± 0.042 | 0.969 ± 0.017 | 0.972 ± 0.020 | 0.975 ± 0.012 | 0.985 ± 0.008 |
| B1 | D0 sure | 0.842 ± 0.053 | 0.764 ± 0.078 | 0.769 ± 0.086 | 0.972 ± 0.015 | 0.892 ± 0.039 | 0.963 ± 0.016 | 0.969 ± 0.013 |
| B1 | D1 Markov | 0.924 ± 0.029 | 0.973 ± 0.028 | 0.836 ± 0.053 | 0.898 ± 0.028 | 0.968 ± 0.017 | 0.928 ± 0.023 | 0.971 ± 0.011 |
| B1 | D2 adim-suresi | 0.913 ± 0.037 | 0.727 ± 0.048 | 0.821 ± 0.046 | 0.895 ± 0.030 | 0.919 ± 0.024 | 0.929 ± 0.019 | 0.960 ± 0.011 |
| B1 | D3 min-p | 0.906 ± 0.038 | 0.927 ± 0.047 | 0.823 ± 0.035 | 0.884 ± 0.026 | 0.951 ± 0.020 | 0.923 ± 0.017 | 0.965 ± 0.010 |
| B2 | D0 sure | 0.898 ± 0.020 | 0.901 ± 0.020 | 0.981 ± 0.005 | 1.000 ± 0.000 | 0.945 ± 0.011 | 0.995 ± 0.001 | 0.987 ± 0.003 |
| B2 | D1 Markov | 0.991 ± 0.012 | 0.992 ± 0.011 | 0.868 ± 0.011 | 0.830 ± 0.012 | 0.995 ± 0.007 | 0.918 ± 0.007 | 0.978 ± 0.003 |
| B2 | D2 adim-suresi | 0.864 ± 0.033 | 0.873 ± 0.030 | 0.986 ± 0.005 | 0.998 ± 0.002 | 0.927 ± 0.018 | 0.996 ± 0.001 | 0.984 ± 0.003 |
| B2 | D3 min-p | 0.959 ± 0.025 | 0.967 ± 0.018 | 0.980 ± 0.011 | 0.983 ± 0.008 | 0.980 ± 0.010 | 0.991 ± 0.005 | 0.992 ± 0.004 |

## Calisma noktasi alpha = 0.10 (BIRINCIL): gozlenen FAR ve tespit orani
FAR = havuzlanmis normal test donguleri (her normal dongu bir kez test); SE = sqrt(alpha(1−alpha)/n_test); 'gecerli' = |FAR − alpha| <= 2·SE (on-kayitli olcut; FAR < alpha − 2·SE ise 'muhafazakar': kural gecerli ama gucsuz, esitlikler p'yi buyutur; FAR > alpha + 2·SE ise 'asiri': gecersiz). Tespit orani = alarm verilen (dongu, f) ciftlerinin orani (anomali donguleri her f'de test edilir; = f ortalamasi).

### Segmentasyon = `tof`
| kaynak | skor | n_test | FAR | SE | gecerli | tespit seq135 | tespit seq145 | tespit fast | tespit slow | tespit tum anomali |
|---|---|---|---|---|---|---|---|---|---|---|
| ToF oracle | D0 sure | 51 | 0.078 | 0.042 | evet | 0.168 | 0.220 | 0.673 | 0.988 | 0.506 |
| ToF oracle | D1 Markov | 51 | 0.000 | 0.042 | HAYIR (muhafazakar) | 1.000 | 1.000 | 0.857 | 0.000 | 0.715 |
| ToF oracle | D2 adim-suresi | 51 | 0.098 | 0.042 | evet | 0.021 | 0.000 | 0.322 | 0.881 | 0.303 |
| ToF oracle | D3 min-p | 51 | 0.078 | 0.042 | evet | 1.000 | 1.000 | 0.931 | 0.600 | 0.883 |
| ToF oracle | OLD TFT | 48 | 0.104 | 0.043 | evet | 0.222 | 0.217 | 0.064 | 0.820 | 0.337 |
| ToF oracle | OLD Z | 48 | 0.083 | 0.043 | evet | 0.141 | 0.057 | 0.114 | 0.543 | 0.217 |
| ToF oracle | OLD min-p | 48 | 0.146 | 0.043 | evet | 0.230 | 0.196 | 0.027 | 0.535 | 0.253 |
| CNN-GRU | D0 sure | 51 | 0.078 | 0.042 | evet | 0.168 | 0.220 | 0.673 | 0.988 | 0.506 |
| CNN-GRU | D1 Markov | 51 | 0.020 | 0.042 | evet | 0.698 | 1.000 | 0.327 | 0.192 | 0.555 |
| CNN-GRU | D2 adim-suresi | 51 | 0.078 | 0.042 | evet | 0.025 | 0.147 | 0.518 | 0.735 | 0.349 |
| CNN-GRU | D3 min-p | 51 | 0.020 | 0.042 | evet | 0.218 | 0.812 | 0.620 | 0.612 | 0.553 |
| CNN-GRU | OLD TFT | 47 | 0.085 | 0.044 | evet | 0.263 | 0.257 | 0.098 | 0.360 | 0.248 |
| CNN-GRU | OLD Z | 47 | 0.128 | 0.044 | evet | 0.107 | 0.074 | 0.289 | 0.720 | 0.298 |
| CNN-GRU | OLD min-p | 47 | 0.085 | 0.044 | evet | 0.259 | 0.248 | 0.147 | 0.428 | 0.274 |
| CNN-only | D0 sure | 51 | 0.078 | 0.042 | evet | 0.168 | 0.220 | 0.673 | 0.988 | 0.506 |
| CNN-only | D1 Markov | 51 | 0.078 | 0.042 | evet | 0.709 | 0.727 | 0.167 | 0.404 | 0.508 |
| CNN-only | D2 adim-suresi | 51 | 0.059 | 0.042 | evet | 0.039 | 0.041 | 0.294 | 0.546 | 0.227 |
| CNN-only | D3 min-p | 51 | 0.039 | 0.042 | evet | 0.014 | 0.024 | 0.208 | 0.246 | 0.121 |
| B1 | D0 sure | 51 | 0.078 | 0.042 | evet | 0.168 | 0.220 | 0.673 | 0.988 | 0.506 |
| B1 | D1 Markov | 51 | 0.039 | 0.042 | evet | 0.000 | 0.118 | 0.045 | 0.019 | 0.043 |
| B1 | D2 adim-suresi | 51 | 0.137 | 0.042 | evet | 0.204 | 0.204 | 0.122 | 0.162 | 0.174 |
| B1 | D3 min-p | 51 | 0.137 | 0.042 | evet | 0.112 | 0.196 | 0.114 | 0.154 | 0.143 |
| B2 | D0 sure | 51 | 0.078 | 0.042 | evet | 0.168 | 0.220 | 0.673 | 0.988 | 0.506 |
| B2 | D1 Markov | 51 | 0.059 | 0.042 | evet | 0.670 | 0.800 | 0.212 | 0.008 | 0.426 |
| B2 | D2 adim-suresi | 51 | 0.039 | 0.042 | evet | 0.095 | 0.151 | 0.698 | 0.877 | 0.447 |
| B2 | D3 min-p | 51 | 0.059 | 0.042 | evet | 0.305 | 0.473 | 0.788 | 0.827 | 0.590 |

### Segmentasyon = `pred`
| kaynak | skor | n_test | FAR | SE | gecerli | tespit seq135 | tespit seq145 | tespit fast | tespit slow | tespit tum anomali |
|---|---|---|---|---|---|---|---|---|---|---|
| ToF oracle | D0 sure | 50 | 0.100 | 0.042 | evet | 0.196 | 0.225 | 0.717 | 1.000 | 0.526 |
| ToF oracle | D1 Markov | 50 | 0.000 | 0.042 | HAYIR (muhafazakar) | 1.000 | 1.000 | 0.870 | 0.000 | 0.716 |
| ToF oracle | D2 adim-suresi | 50 | 0.080 | 0.042 | evet | 0.025 | 0.000 | 0.365 | 0.988 | 0.341 |
| ToF oracle | D3 min-p | 50 | 0.060 | 0.042 | evet | 1.000 | 1.000 | 0.930 | 0.882 | 0.954 |
| ToF oracle | OLD TFT | 47 | 0.106 | 0.044 | evet | 0.358 | 0.373 | 0.038 | 0.800 | 0.403 |
| ToF oracle | OLD Z | 47 | 0.064 | 0.044 | evet | 0.125 | 0.040 | 0.090 | 0.662 | 0.234 |
| ToF oracle | OLD min-p | 47 | 0.106 | 0.044 | evet | 0.226 | 0.182 | 0.005 | 0.588 | 0.259 |
| CNN-GRU | D0 sure | 49 | 0.082 | 0.043 | evet | 0.171 | 0.196 | 0.727 | 0.969 | 0.530 |
| CNN-GRU | D1 Markov | 49 | 0.000 | 0.043 | HAYIR (muhafazakar) | 0.011 | 0.600 | 0.265 | 0.129 | 0.232 |
| CNN-GRU | D2 adim-suresi | 49 | 0.122 | 0.043 | evet | 0.036 | 0.100 | 0.396 | 0.651 | 0.307 |
| CNN-GRU | D3 min-p | 49 | 0.082 | 0.043 | evet | 0.021 | 0.409 | 0.359 | 0.376 | 0.285 |
| CNN-GRU | OLD TFT | 46 | 0.109 | 0.044 | evet | 0.434 | 0.262 | 0.062 | 0.148 | 0.230 |
| CNN-GRU | OLD Z | 46 | 0.065 | 0.044 | evet | 0.079 | 0.062 | 0.267 | 0.679 | 0.291 |
| CNN-GRU | OLD min-p | 46 | 0.043 | 0.044 | evet | 0.042 | 0.036 | 0.182 | 0.355 | 0.162 |
| CNN-only | D0 sure | 52 | 0.135 | 0.042 | evet | 0.061 | 0.057 | 0.149 | 0.636 | 0.288 |
| CNN-only | D1 Markov | 52 | 0.058 | 0.042 | evet | 0.400 | 0.400 | 0.179 | 0.298 | 0.319 |
| CNN-only | D2 adim-suresi | 52 | 0.077 | 0.042 | evet | 0.036 | 0.024 | 0.336 | 0.427 | 0.236 |
| CNN-only | D3 min-p | 52 | 0.058 | 0.042 | evet | 0.000 | 0.000 | 0.149 | 0.118 | 0.072 |
| B1 | D0 sure | 55 | 0.055 | 0.040 | evet | 0.067 | 0.023 | 0.035 | 0.179 | 0.096 |
| B1 | D1 Markov | 55 | 0.073 | 0.040 | evet | 0.020 | 0.372 | 0.027 | 0.023 | 0.083 |
| B1 | D2 adim-suresi | 55 | 0.127 | 0.040 | evet | 0.320 | 0.042 | 0.123 | 0.076 | 0.138 |
| B1 | D3 min-p | 55 | 0.091 | 0.040 | evet | 0.123 | 0.200 | 0.058 | 0.037 | 0.090 |
| B2 | D0 sure | 51 | 0.098 | 0.042 | evet | 0.243 | 0.308 | 0.787 | 1.000 | 0.577 |
| B2 | D1 Markov | 51 | 0.059 | 0.042 | evet | 0.604 | 0.800 | 0.217 | 0.008 | 0.409 |
| B2 | D2 adim-suresi | 51 | 0.078 | 0.042 | evet | 0.068 | 0.125 | 0.800 | 0.992 | 0.487 |
| B2 | D3 min-p | 51 | 0.078 | 0.042 | evet | 0.254 | 0.475 | 0.921 | 0.929 | 0.633 |

## Calisma noktasi alpha = 0.05 (ikincil): gozlenen FAR ve tespit orani
FAR = havuzlanmis normal test donguleri (her normal dongu bir kez test); SE = sqrt(alpha(1−alpha)/n_test); 'gecerli' = |FAR − alpha| <= 2·SE (on-kayitli olcut; FAR < alpha − 2·SE ise 'muhafazakar': kural gecerli ama gucsuz, esitlikler p'yi buyutur; FAR > alpha + 2·SE ise 'asiri': gecersiz). Tespit orani = alarm verilen (dongu, f) ciftlerinin orani (anomali donguleri her f'de test edilir; = f ortalamasi).

### Segmentasyon = `tof`
| kaynak | skor | n_test | FAR | SE | gecerli | tespit seq135 | tespit seq145 | tespit fast | tespit slow | tespit tum anomali |
|---|---|---|---|---|---|---|---|---|---|---|
| ToF oracle | D0 sure | 51 | 0.078 | 0.031 | evet | 0.119 | 0.171 | 0.473 | 0.588 | 0.333 |
| ToF oracle | D1 Markov | 51 | 0.000 | 0.031 | evet | 1.000 | 1.000 | 0.857 | 0.000 | 0.715 |
| ToF oracle | D2 adim-suresi | 51 | 0.078 | 0.031 | evet | 0.014 | 0.000 | 0.265 | 0.600 | 0.217 |
| ToF oracle | D3 min-p | 51 | 0.000 | 0.031 | evet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ToF oracle | OLD TFT | 48 | 0.104 | 0.031 | evet | 0.189 | 0.191 | 0.005 | 0.392 | 0.199 |
| ToF oracle | OLD Z | 48 | 0.083 | 0.031 | evet | 0.044 | 0.004 | 0.027 | 0.216 | 0.075 |
| ToF oracle | OLD min-p | 48 | 0.000 | 0.031 | evet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CNN-GRU | D0 sure | 51 | 0.078 | 0.031 | evet | 0.119 | 0.171 | 0.473 | 0.588 | 0.333 |
| CNN-GRU | D1 Markov | 51 | 0.000 | 0.031 | evet | 0.196 | 0.800 | 0.306 | 0.162 | 0.357 |
| CNN-GRU | D2 adim-suresi | 51 | 0.020 | 0.031 | evet | 0.021 | 0.086 | 0.351 | 0.542 | 0.245 |
| CNN-GRU | D3 min-p | 51 | 0.000 | 0.031 | evet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CNN-GRU | OLD TFT | 47 | 0.064 | 0.032 | evet | 0.244 | 0.248 | 0.044 | 0.292 | 0.211 |
| CNN-GRU | OLD Z | 47 | 0.043 | 0.032 | evet | 0.052 | 0.022 | 0.111 | 0.300 | 0.122 |
| CNN-GRU | OLD min-p | 47 | 0.000 | 0.032 | evet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CNN-only | D0 sure | 51 | 0.078 | 0.031 | evet | 0.119 | 0.171 | 0.473 | 0.588 | 0.333 |
| CNN-only | D1 Markov | 51 | 0.020 | 0.031 | evet | 0.000 | 0.008 | 0.053 | 0.046 | 0.026 |
| CNN-only | D2 adim-suresi | 51 | 0.020 | 0.031 | evet | 0.014 | 0.020 | 0.155 | 0.200 | 0.096 |
| CNN-only | D3 min-p | 51 | 0.000 | 0.031 | evet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B1 | D0 sure | 51 | 0.078 | 0.031 | evet | 0.119 | 0.171 | 0.473 | 0.588 | 0.333 |
| B1 | D1 Markov | 51 | 0.039 | 0.031 | evet | 0.000 | 0.012 | 0.033 | 0.019 | 0.015 |
| B1 | D2 adim-suresi | 51 | 0.098 | 0.031 | evet | 0.112 | 0.184 | 0.082 | 0.135 | 0.128 |
| B1 | D3 min-p | 51 | 0.000 | 0.031 | evet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B2 | D0 sure | 51 | 0.078 | 0.031 | evet | 0.119 | 0.171 | 0.473 | 0.588 | 0.333 |
| B2 | D1 Markov | 51 | 0.020 | 0.031 | evet | 0.274 | 0.400 | 0.159 | 0.000 | 0.208 |
| B2 | D2 adim-suresi | 51 | 0.039 | 0.031 | evet | 0.046 | 0.102 | 0.673 | 0.827 | 0.404 |
| B2 | D3 min-p | 51 | 0.000 | 0.031 | evet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Segmentasyon = `pred`
| kaynak | skor | n_test | FAR | SE | gecerli | tespit seq135 | tespit seq145 | tespit fast | tespit slow | tespit tum anomali |
|---|---|---|---|---|---|---|---|---|---|---|
| ToF oracle | D0 sure | 50 | 0.100 | 0.031 | evet | 0.146 | 0.167 | 0.639 | 1.000 | 0.481 |
| ToF oracle | D1 Markov | 50 | 0.000 | 0.031 | evet | 1.000 | 1.000 | 0.870 | 0.000 | 0.716 |
| ToF oracle | D2 adim-suresi | 50 | 0.060 | 0.031 | evet | 0.014 | 0.000 | 0.304 | 0.882 | 0.298 |
| ToF oracle | D3 min-p | 50 | 0.000 | 0.031 | evet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ToF oracle | OLD TFT | 47 | 0.085 | 0.032 | evet | 0.170 | 0.182 | 0.005 | 0.246 | 0.155 |
| ToF oracle | OLD Z | 47 | 0.064 | 0.032 | evet | 0.057 | 0.000 | 0.000 | 0.412 | 0.121 |
| ToF oracle | OLD min-p | 47 | 0.000 | 0.032 | evet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CNN-GRU | D0 sure | 49 | 0.020 | 0.031 | evet | 0.061 | 0.065 | 0.367 | 0.576 | 0.278 |
| CNN-GRU | D1 Markov | 49 | 0.000 | 0.031 | evet | 0.007 | 0.400 | 0.147 | 0.081 | 0.147 |
| CNN-GRU | D2 adim-suresi | 49 | 0.082 | 0.031 | evet | 0.014 | 0.065 | 0.237 | 0.322 | 0.164 |
| CNN-GRU | D3 min-p | 49 | 0.000 | 0.031 | evet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CNN-GRU | OLD TFT | 46 | 0.022 | 0.032 | evet | 0.008 | 0.000 | 0.000 | 0.010 | 0.005 |
| CNN-GRU | OLD Z | 46 | 0.043 | 0.032 | evet | 0.034 | 0.036 | 0.182 | 0.348 | 0.158 |
| CNN-GRU | OLD min-p | 46 | 0.000 | 0.032 | evet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CNN-only | D0 sure | 52 | 0.058 | 0.030 | evet | 0.032 | 0.029 | 0.111 | 0.568 | 0.243 |
| CNN-only | D1 Markov | 52 | 0.019 | 0.030 | evet | 0.000 | 0.000 | 0.077 | 0.041 | 0.030 |
| CNN-only | D2 adim-suresi | 52 | 0.038 | 0.030 | evet | 0.000 | 0.000 | 0.072 | 0.084 | 0.045 |
| CNN-only | D3 min-p | 52 | 0.000 | 0.030 | evet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B1 | D0 sure | 55 | 0.036 | 0.029 | evet | 0.033 | 0.005 | 0.023 | 0.111 | 0.056 |
| B1 | D1 Markov | 55 | 0.036 | 0.029 | evet | 0.000 | 0.186 | 0.008 | 0.004 | 0.035 |
| B1 | D2 adim-suresi | 55 | 0.055 | 0.029 | evet | 0.123 | 0.019 | 0.050 | 0.035 | 0.056 |
| B1 | D3 min-p | 55 | 0.000 | 0.029 | evet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B2 | D0 sure | 51 | 0.059 | 0.031 | evet | 0.168 | 0.183 | 0.662 | 1.000 | 0.498 |
| B2 | D1 Markov | 51 | 0.020 | 0.031 | evet | 0.211 | 0.400 | 0.217 | 0.000 | 0.204 |
| B2 | D2 adim-suresi | 51 | 0.059 | 0.031 | evet | 0.054 | 0.125 | 0.771 | 0.929 | 0.460 |
| B2 | D3 min-p | 51 | 0.000 | 0.031 | evet | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## OLD boru hatti: sabit esik (TFT > 0.5, Z > 1.0) ve ayni conformal prosedur
| kaynak | seg | skor | AUROC(tum) | FAR sabit | tespit sabit | FAR a=0.10 | tespit a=0.10 |
|---|---|---|---|---|---|---|---|
| ToF oracle | tof | OLD TFT | 0.531 ± 0.084 | 0.979 | 0.977 | 0.104 | 0.337 |
| ToF oracle | tof | OLD Z | 0.594 ± 0.056 | 0.875 | 0.835 | 0.083 | 0.217 |
| ToF oracle | tof | OLD min-p | 0.649 ± 0.067 | 1.000 | 0.999 | 0.146 | 0.253 |
| ToF oracle | pred | OLD TFT | 0.514 ± 0.149 | 0.979 | 0.979 | 0.106 | 0.403 |
| ToF oracle | pred | OLD Z | 0.609 ± 0.067 | 0.915 | 0.874 | 0.064 | 0.234 |
| ToF oracle | pred | OLD min-p | 0.692 ± 0.131 | 1.000 | 1.000 | 0.106 | 0.259 |
| CNN-GRU | tof | OLD TFT | 0.569 ± 0.186 | 1.000 | 0.991 | 0.085 | 0.248 |
| CNN-GRU | tof | OLD Z | 0.557 ± 0.054 | 0.979 | 0.931 | 0.128 | 0.298 |
| CNN-GRU | tof | OLD min-p | 0.623 ± 0.122 | 1.000 | 0.999 | 0.085 | 0.274 |
| CNN-GRU | pred | OLD TFT | 0.575 ± 0.095 | 1.000 | 0.972 | 0.109 | 0.230 |
| CNN-GRU | pred | OLD Z | 0.593 ± 0.047 | 1.000 | 0.933 | 0.065 | 0.291 |
| CNN-GRU | pred | OLD min-p | 0.698 ± 0.102 | 1.000 | 0.992 | 0.043 | 0.162 |

## Eslesik kat farki: durum kaynagi X − CNN-GRU (D3 AUROC; karar olcutu 2'nin 2. kosulu)
| seg | X | anomali | fark (X−CNN-GRU) | %95 GA | 0'i iceriyor ya da |fark|<=0.05 |
|---|---|---|---|---|---|
| tof | ToF oracle | seq135 | +0.054 | [-0.104, +0.212] | evet |
| tof | ToF oracle | seq145 | -0.020 | [-0.090, +0.051] | evet |
| tof | ToF oracle | fast | +0.039 | [-0.030, +0.109] | evet |
| tof | ToF oracle | slow | +0.023 | [-0.044, +0.089] | evet |
| tof | CNN-only | seq135 | -0.041 | [-0.205, +0.122] | evet |
| tof | CNN-only | seq145 | -0.106 | [-0.213, +0.002] | evet |
| tof | CNN-only | fast | -0.144 | [-0.286, -0.001] | HAYIR |
| tof | CNN-only | slow | -0.055 | [-0.135, +0.025] | evet |
| tof | B1 | seq135 | -0.137 | [-0.347, +0.074] | evet |
| tof | B1 | seq145 | -0.182 | [-0.306, -0.058] | HAYIR |
| tof | B1 | fast | -0.373 | [-0.478, -0.267] | HAYIR |
| tof | B1 | slow | -0.209 | [-0.339, -0.078] | HAYIR |
| tof | B2 | seq135 | +0.024 | [-0.095, +0.142] | evet |
| tof | B2 | seq145 | -0.047 | [-0.161, +0.068] | evet |
| tof | B2 | fast | +0.042 | [-0.012, +0.096] | evet |
| tof | B2 | slow | +0.018 | [-0.043, +0.079] | evet |
| pred | ToF oracle | seq135 | +0.173 | [-0.053, +0.399] | evet |
| pred | ToF oracle | seq145 | +0.058 | [-0.074, +0.191] | evet |
| pred | ToF oracle | fast | +0.071 | [-0.033, +0.175] | evet |
| pred | ToF oracle | slow | +0.048 | [-0.055, +0.151] | evet |
| pred | CNN-only | seq135 | +0.029 | [-0.140, +0.198] | evet |
| pred | CNN-only | seq145 | -0.058 | [-0.186, +0.069] | evet |
| pred | CNN-only | fast | -0.128 | [-0.249, -0.006] | HAYIR |
| pred | CNN-only | slow | -0.053 | [-0.121, +0.016] | evet |
| pred | B1 | seq135 | -0.122 | [-0.333, +0.089] | evet |
| pred | B1 | seq145 | -0.184 | [-0.529, +0.160] | evet |
| pred | B1 | fast | -0.420 | [-0.597, -0.243] | HAYIR |
| pred | B1 | slow | -0.516 | [-0.730, -0.303] | HAYIR |
| pred | B2 | seq135 | +0.081 | [-0.034, +0.197] | evet |
| pred | B2 | seq145 | +0.006 | [-0.081, +0.093] | evet |
| pred | B2 | fast | +0.058 | [+0.006, +0.110] | HAYIR |
| pred | B2 | slow | +0.035 | [-0.022, +0.092] | evet |

## Dogrulama kapilari (ToF oracle hucresi)
- Oracle × `tof`: D1 AUROC seq135 = 1.000, seq145 = 1.000 (beklenen ≈ 1); D0 AUROC fast = 0.894, slow = 0.975; D2 AUROC fast = 0.789, slow = 0.967; FAR@0.10: D0 0.078 / D1 0.000 / D2 0.098 / D3 0.078.
- Oracle × `pred`: D1 AUROC seq135 = 1.000, seq145 = 1.000 (beklenen ≈ 1); D0 AUROC fast = 0.893, slow = 1.000; D2 AUROC fast = 0.803, slow = 0.986; FAR@0.10: D0 0.100 / D1 0.000 / D2 0.080 / D3 0.060.

Yorum ve karar olcutleri (B1/B2 vs CNN-GRU hukmu) bu asamada verilmez; yalniz protokolun sayilari uretilir.

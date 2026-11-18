# Faz A3.1 - Hata yayilimi (kare dogrulugu -> dongu duzeyi anomali tespiti)

Brief: `faz_a3/DESIGN_A3.md` A3.1 (on-kayitli; ayrintilar 'Sapmalar' satirinda). **Statu: ikincil/sonradan analiz.** Birincil sonuclar Faz A'nin on-kayitli sayilaridir; ana model B2 + D3 degismez. Yeni veri/yeniden egitim yok. Gurultu **sentetiktir**; gercek hatalar gecis-yapilidir; tek operator. Her sayi `faz_a3/01_error_propagation.py` ciktisinda da basilidir.

## 0. Kapi (hattin Faz A'yi birebir uretmesi)
Kendi yalin degerlendirici + `build_cycles('pred')` + `score_group` hattim, bozmasiz etiketlerle (ToF = r=0 referansi ve 4 gercek kaynak) Faz A `04_anomaly_table.csv` D3/pred satirlarini (AUROC: tum anomali + 4 tur + 2 grup; FAR@0.10) uretiyor mu? Maks |fark| = 1.11e-16; **PASS** (esik 1e-9).

r = 0 referans (ToF oracle, bozmasiz; ihtiyac duyulan ust sinir): D3 AUROC(tum) = 0.9680, FAR@0.10 = 0.0600 (n_pairs = 50).

## 1. 'Kare hata orani' tanimi (overlay ve kol b ekseni)
Kaynak m icin **kare hata orani = #{kare: OOF y_pred != L0 (frame_table.y_true)} / #kare**, s02-s06'nin TUM 11589 kareleri uzerinden (oturum basina pencere-siz ilk 4 kare `source_labels` geregi none tahmin edilir ve L0 != none ise hata sayilir; yalniz pencereli kareler uzerinden hesap asagida karsilastirma icin). Simulasyonda r ise ayni tanimda 'bozulan kare payi' (gerceklesen oran tohum basina olculur).
| kaynak | hata_orani | hata_orani_yalniz_pencereli |
|---|---|---|
| B2 | 0.0983 | 0.0985 |
| CNN-GRU | 0.1144 | 0.1146 |
| CNN-only | 0.2075 | 0.2079 |
| B1 | 0.3760 | 0.3767 |

## 2. Hata-kosusu uzunluk dagilimlari (S1 OOF, s02-s06)
Hata kosusu = oturum icinde ardisik kareler boyunca y_pred != L0 olan maksimal kosu. 'sabit yanlis etiket (>=2)' = uzunlugu >= 2 olan hata kosularinin tek bir yanlis etiket tasiyanlarinin payi.
| kaynak | n_kosu | ort_uzunluk | L1 | L2 | L3 | L4_5 | L6_10 | L_gt10 | pay_L_ge2 | sabit_yanlis_etiket_ge2 | maks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B2 | 801 | 1.4220 | 604 | 117 | 47 | 26 | 7 | 0 | 0.2459 | 0.9543 | 9 |
| CNN-GRU | 848 | 1.5637 | 588 | 153 | 53 | 41 | 11 | 2 | 0.3066 | 0.9231 | 15 |
| CNN-only | 1620 | 1.4846 | 1135 | 314 | 92 | 67 | 11 | 1 | 0.2994 | 0.6309 | 11 |
| B1 | 2466 | 1.7672 | 1436 | 589 | 226 | 167 | 43 | 5 | 0.4177 | 0.6010 | 18 |

**B2'den (YALNIZ B2) uydurulan geometrik patlama: ortalama uzunluk ell = 1.4220 kare, kosuldan ayrilma olasiligi q = 1/ell = 0.7032.** Diger kaynaklar uydurmada kullanilmaz; yalniz sinamada yer alir. Geometrik modelde P(L>=2) = 1-q = 0.297; B2 gercek P(L>=2) = 0.246.
B2 hata-kosullu karisiklik satirlari Q[y, :] (L0 sinifi -> yedek etiket olasiligi, diag = 0), sinif 0..5:
| L0 | ->0 | ->1 | ->2 | ->3 | ->4 | ->5 |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.253 | 0.218 | 0.063 | 0.048 | 0.417 |
| 1 | 0.978 | 0.000 | 0.022 | 0.000 | 0.000 | 0.000 |
| 2 | 0.864 | 0.000 | 0.000 | 0.136 | 0.000 | 0.000 |
| 3 | 0.737 | 0.000 | 0.000 | 0.000 | 0.263 | 0.000 |
| 4 | 0.824 | 0.000 | 0.000 | 0.059 | 0.000 | 0.118 |
| 5 | 0.855 | 0.097 | 0.000 | 0.000 | 0.048 | 0.000 |

Betimleyici (post-hoc, Sapmalar 2. satir): hata karelerinin L0 gecisine yakinligi (`near_change` = L0 degisim sinirinin 2 kare oncesi + 2 kare sonrasi). Tum karelerde near_change payi = 0.4888.
| kaynak | hata_karesi | hata_karesi_near_change_payi | P_hata_given_near_change | P_hata_given_uzak |
|---|---|---|---|---|
| B2 | 1139 | 0.7550 | 0.1518 | 0.0471 |
| CNN-GRU | 1326 | 0.7006 | 0.1640 | 0.0670 |
| CNN-only | 2405 | 0.5098 | 0.2164 | 0.1990 |
| B1 | 4358 | 0.5301 | 0.4078 | 0.3457 |

## 3. Kol (a) - analitik model vs gozlenen (4 kaynak x 5 oturum)
**Tanimlar (Sapmalar satiri):** oranlar = olay sayisi / oturumun ToF referans dongu sayisi `n_tof_ref`. **Gozlenen (birincil)**: debounce sonrasi pred ziyaretlerinden, ayni siniftan hicbir L0 karesi icermeyenler = *sahte ziyaret*; bunlarin Box1 olanlari = *dongu bolunmesi*. **Analitik tahmin**: kaynak m icin havuzlanmis (s02-s06) 'sabit yanlis kutu etiketli, uzunluk >= 2' kosu uzunlugu dagilimindan kare basina tehlike x oturumun kare sayisi / `n_tof_ref` (sahte ziyaret icin tum kutular, bolunme icin yalniz Box1; debounce'un 1-karelik bosluk birlestirmesi modellenmez). LOSO: tehlike o oturum haric dort oturumdan. **Gecme (sik)**: dort kaynagin hepsinde, tum oturumlarda ve iki oranda max |tahmin - gozlenen| <= 0.05 (birincil gozlenen, havuzlanmis tehlike). Ikincil gozlenen: `04_cycle_counts.csv` `n_loops_with_2plus/n_tof_ref` (ToF dongusu basina >= 2 pred donguye dusen oran; sinir kaymasi dahil).
| kaynak | oturum | n_tof_ref | gz_sahte | tah_sahte | fark_sahte | tah_sahte_loso | ham_sahte | gz_bol | tah_bol | fark_bol | tah_bol_loso | ham_bol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B2 | s02 | 51 | 0.0588 | 0.6464 | 0.5876 | 0.7372 | 0.2745 | 0.0196 | 0.1424 | 0.1227 | 0.1484 | 0.1176 |
| B2 | s03 | 57 | 0.1754 | 0.6122 | 0.4367 | 0.6715 | 0.3860 | 0.0000 | 0.1348 | 0.1348 | 0.1472 | 0.0877 |
| B2 | s04 | 49 | 0.1020 | 0.5704 | 0.4684 | 0.6354 | 0.2449 | 0.0000 | 0.1256 | 0.1256 | 0.1466 | 0.0204 |
| B2 | s05 | 49 | 0.7143 | 0.5266 | -0.1877 | 0.4963 | 0.6939 | 0.0408 | 0.1160 | 0.0752 | 0.1296 | 0.0408 |
| B2 | s06 | 52 | 0.0000 | 0.8921 | 0.8921 | 0.6015 | 1.6538 | 0.0000 | 0.1965 | 0.1965 | 0.1027 | 0.4423 |
| CNN-GRU | s02 | 51 | 0.0980 | 0.6964 | 0.5984 | 0.7994 | 0.2745 | 0.0000 | 0.0923 | 0.0923 | 0.1005 | 0.0588 |
| CNN-GRU | s03 | 57 | 0.1404 | 0.6595 | 0.5192 | 0.6715 | 0.6140 | 0.0000 | 0.0875 | 0.0875 | 0.1058 | 0.0175 |
| CNN-GRU | s04 | 49 | 0.1020 | 0.6145 | 0.5125 | 0.6557 | 0.4082 | 0.0000 | 0.0815 | 0.0815 | 0.0896 | 0.0408 |
| CNN-GRU | s05 | 49 | 0.7347 | 0.5674 | -0.1673 | 0.5111 | 0.8776 | 0.0408 | 0.0752 | 0.0344 | 0.0815 | 0.0408 |
| CNN-GRU | s06 | 52 | 0.2500 | 0.9611 | 0.7111 | 0.8216 | 1.3269 | 0.1346 | 0.1274 | -0.0072 | 0.0587 | 0.3077 |
| CNN-only | s02 | 51 | 0.2157 | 1.1696 | 0.9539 | 1.2924 | 0.6667 | 0.0588 | 0.3963 | 0.3375 | 0.4500 | 0.1765 |
| CNN-only | s03 | 57 | 0.2632 | 1.1077 | 0.8446 | 1.2187 | 0.6842 | 0.0000 | 0.3753 | 0.3753 | 0.4415 | 0.1228 |
| CNN-only | s04 | 49 | 0.2041 | 1.0321 | 0.8281 | 1.1159 | 0.6122 | 0.0204 | 0.3497 | 0.3293 | 0.3869 | 0.1633 |
| CNN-only | s05 | 49 | 1.0408 | 0.9529 | -0.0879 | 0.9147 | 1.1633 | 0.0408 | 0.3229 | 0.2820 | 0.3629 | 0.1020 |
| CNN-only | s06 | 52 | 1.3654 | 1.6143 | 0.2489 | 1.1737 | 2.7692 | 0.7115 | 0.5469 | -0.1646 | 0.2127 | 1.4231 |
| B1 | s02 | 51 | 0.7059 | 1.5967 | 0.8908 | 1.5892 | 1.6275 | 0.0392 | 0.1885 | 0.1493 | 0.2154 | 0.0784 |
| B1 | s03 | 57 | 0.5439 | 1.5122 | 0.9683 | 1.5637 | 1.3158 | 0.0351 | 0.1785 | 0.1435 | 0.1932 | 0.1228 |
| B1 | s04 | 49 | 0.4490 | 1.4090 | 0.9600 | 1.4988 | 0.9592 | 0.0000 | 0.1664 | 0.1664 | 0.1914 | 0.0408 |
| B1 | s05 | 49 | 1.5918 | 1.3008 | -0.2910 | 1.2110 | 1.7959 | 0.0612 | 0.1536 | 0.0924 | 0.1704 | 0.0612 |
| B1 | s06 | 52 | 1.1346 | 2.2037 | 1.0691 | 2.1493 | 2.3462 | 0.5000 | 0.2602 | -0.2398 | 0.1174 | 0.6346 |

**POST-HOC ikincil varyant (Sapmalar 2. satir): 'ic patlama' analitik tahmini** (yanlis-kutu kosusunun iki komsu L0 karesi de o kutu degil -> sinir kaymasi dislanir); ayni gozlenen, ayni 0.05 olcutu:
| kaynak | oturum | gz_sahte | tah_sahte_ic | fark_sahte_ic | gz_bol | tah_bol_ic | fark_bol_ic |
|---|---|---|---|---|---|---|---|
| B2 | s02 | 0.0588 | 0.2039 | 0.1451 | 0.0196 | 0.0192 | -0.0004 |
| B2 | s03 | 0.1754 | 0.1931 | 0.0177 | 0.0000 | 0.0182 | 0.0182 |
| B2 | s04 | 0.1020 | 0.1799 | 0.0779 | 0.0000 | 0.0170 | 0.0170 |
| B2 | s05 | 0.7143 | 0.1661 | -0.5482 | 0.0408 | 0.0157 | -0.0251 |
| B2 | s06 | 0.0000 | 0.2814 | 0.2814 | 0.0000 | 0.0266 | 0.0266 |
| CNN-GRU | s02 | 0.0980 | 0.2578 | 0.1597 | 0.0000 | 0.0346 | 0.0346 |
| CNN-GRU | s03 | 0.1404 | 0.2441 | 0.1038 | 0.0000 | 0.0328 | 0.0328 |
| CNN-GRU | s04 | 0.1020 | 0.2275 | 0.1254 | 0.0000 | 0.0306 | 0.0306 |
| CNN-GRU | s05 | 0.7347 | 0.2100 | -0.5247 | 0.0408 | 0.0282 | -0.0126 |
| CNN-GRU | s06 | 0.2500 | 0.3558 | 0.1058 | 0.1346 | 0.0478 | -0.0868 |
| CNN-only | s02 | 0.2157 | 0.7464 | 0.5307 | 0.0588 | 0.2578 | 0.1990 |
| CNN-only | s03 | 0.2632 | 0.7069 | 0.4437 | 0.0000 | 0.2441 | 0.2441 |
| CNN-only | s04 | 0.2041 | 0.6587 | 0.4546 | 0.0204 | 0.2275 | 0.2071 |
| CNN-only | s05 | 1.0408 | 0.6081 | -0.4327 | 0.0408 | 0.2100 | 0.1692 |
| CNN-only | s06 | 1.3654 | 1.0302 | -0.3352 | 0.7115 | 0.3558 | -0.3558 |
| B1 | s02 | 0.7059 | 0.7503 | 0.0444 | 0.0392 | 0.1154 | 0.0762 |
| B1 | s03 | 0.5439 | 0.7105 | 0.1667 | 0.0351 | 0.1093 | 0.0742 |
| B1 | s04 | 0.4490 | 0.6621 | 0.2131 | 0.0000 | 0.1019 | 0.1019 |
| B1 | s05 | 1.5918 | 0.6112 | -0.9806 | 0.0612 | 0.0940 | 0.0328 |
| B1 | s06 | 1.1346 | 1.0355 | -0.0992 | 0.5000 | 0.1593 | -0.3407 |

Ikincil gozlenen dongu-bolunme olculeri (ayni oturumlar):
| kaynak | oturum | gz_bol | gz_bol_parca | gz_bol_sayim | gz_net_giris | tah_bol | fark_bol_sayim |
|---|---|---|---|---|---|---|---|
| B2 | s02 | 0.0196 | 0.0196 | 0.1961 | 0.0196 | 0.1424 | -0.0537 |
| B2 | s03 | 0.0000 | 0.0000 | 0.1930 | 0.0000 | 0.1348 | -0.0582 |
| B2 | s04 | 0.0000 | 0.0000 | 0.1429 | 0.0000 | 0.1256 | -0.0172 |
| B2 | s05 | 0.0408 | 0.0408 | 0.0612 | 0.0000 | 0.1160 | 0.0548 |
| B2 | s06 | 0.0000 | 0.0000 | 0.2500 | 0.0000 | 0.1965 | -0.0535 |
| CNN-GRU | s02 | 0.0000 | 0.0000 | 0.1569 | 0.0000 | 0.0923 | -0.0645 |
| CNN-GRU | s03 | 0.0000 | 0.0000 | 0.1404 | 0.0000 | 0.0875 | -0.0529 |
| CNN-GRU | s04 | 0.0000 | 0.0000 | 0.0816 | -0.0204 | 0.0815 | -0.0001 |
| CNN-GRU | s05 | 0.0408 | 0.0612 | 0.0612 | 0.0204 | 0.0752 | 0.0140 |
| CNN-GRU | s06 | 0.1346 | 0.1731 | 0.3654 | 0.1731 | 0.1274 | -0.2379 |
| CNN-only | s02 | 0.0588 | 0.0588 | 0.3137 | 0.0588 | 0.3963 | 0.0826 |
| CNN-only | s03 | 0.0000 | 0.0175 | 0.2632 | 0.0175 | 0.3753 | 0.1122 |
| CNN-only | s04 | 0.0204 | 0.0408 | 0.3061 | 0.0408 | 0.3497 | 0.0436 |
| CNN-only | s05 | 0.0408 | 0.0408 | 0.3265 | -0.0204 | 0.3229 | -0.0037 |
| CNN-only | s06 | 0.7115 | 0.7308 | 0.5769 | 0.7308 | 0.5469 | -0.0300 |
| B1 | s02 | 0.0392 | 0.1569 | 0.2745 | 0.1176 | 0.1885 | -0.0860 |
| B1 | s03 | 0.0351 | 0.1579 | 0.2632 | 0.0877 | 0.1785 | -0.0846 |
| B1 | s04 | 0.0000 | 0.0408 | 0.2041 | -0.1020 | 0.1664 | -0.0377 |
| B1 | s05 | 0.0612 | 0.1837 | 0.2449 | 0.1020 | 0.1536 | -0.0913 |
| B1 | s06 | 0.5000 | 0.9615 | 0.6538 | 0.9038 | 0.2602 | -0.3937 |

| kaynak | maks_abs_fark_ic_patlama | gecti_ic | maks_abs_fark_havuz_tehlike | gecti_sik | maks_abs_fark_LOSO | gecti_LOSO | maks_abs_fark_bolunme_vs_sayim | gecti_sayim | havuz_fark_sahte | havuz_fark_bolunme |
|---|---|---|---|---|---|---|---|---|---|---|
| B2 | 0.5482 | False | 0.8921 | False | 0.6783 | False | 0.0582 | False | 0.4457 | 0.1318 |
| CNN-GRU | 0.5247 | False | 0.7111 | False | 0.7013 | False | 0.2379 | False | 0.4419 | 0.0581 |
| CNN-only | 0.5307 | False | 0.9539 | False | 1.0767 | False | 0.1122 | False | 0.5659 | 0.2326 |
| B1 | 0.9806 | False | 1.0691 | False | 1.0498 | False | 0.3937 | False | 0.7326 | 0.0620 |

Betimleyici (post-hoc): s05'te gozlenen sahte ziyaretlerin sinif dagilimi (A2.3: s05 L0 etiketinde Box5 silinmis; sahte sayilan Box5 ziyaretlerinin onemli kismi bu artefakttan gelebilir):
| kaynak | sahte_toplam | Box1 | Box2 | Box3 | Box4 | Box5 |
|---|---|---|---|---|---|---|
| B2 | 35 | 2 | 5 | 0 | 1 | 27 |
| CNN-GRU | 36 | 2 | 5 | 0 | 0 | 29 |
| CNN-only | 51 | 2 | 10 | 4 | 0 | 35 |
| B1 | 78 | 3 | 19 | 16 | 4 | 36 |

**Kol (a) karari (on-kayitli sik kural; birincil gozlenen; havuzlanmis tehlike): FAIL** (gecen kaynaklar: yok; kalan: B2, CNN-GRU, CNN-only, B1). LOSO tehlike ile: FAIL; post-hoc 'ic patlama' varyantiyla (ikincil): FAIL (gecen: yok); ikincil (sayim-tabanli bolunme) ile: FAIL.

## 4. Kol (b) - simulasyon: tum dagitilan yol bozulur
Toplam 1000 simulasyon (6 r x 2 tur x 50 tohum + 4 gozlenen-r noktasi x 2 tur x 50); atlanan kat (<3 egitim/kal. dongusu) olan simulasyon sayisi: 0; min kat sayisi = 5.

### 4.1 Gerceklesen kare hata orani (tohumlar uzerinden ortalama) ve dongu yapisi
| tur | r | gerceklesen_r | dongu_sayisi | normal_dongu | ziyaret_per_ToF_dongu | sahte_ziyaret_per_ToF_dongu | bolunme_per_ToF_dongu |
|---|---|---|---|---|---|---|---|
| iid | 0.02 | 0.0199 | 251.5200 | 50.0800 | 2.7787 | 0.0047 | 0.0009 |
| iid | 0.05 | 0.0504 | 254.6600 | 50.6200 | 2.8149 | 0.0280 | 0.0049 |
| iid | 0.10 | 0.1005 | 264.8600 | 51.8400 | 2.9419 | 0.1198 | 0.0210 |
| iid | 0.15 | 0.1501 | 277.0600 | 54.0000 | 3.0956 | 0.2414 | 0.0427 |
| iid | 0.20 | 0.2003 | 293.1400 | 56.1200 | 3.3227 | 0.4355 | 0.0735 |
| iid | 0.30 | 0.2992 | 319.1400 | 60.9800 | 3.7897 | 0.9258 | 0.1567 |
| bursty | 0.02 | 0.0204 | 260.2600 | 51.5000 | 2.8918 | 0.1012 | 0.0237 |
| bursty | 0.05 | 0.0501 | 272.1600 | 53.6600 | 3.0517 | 0.2395 | 0.0564 |
| bursty | 0.10 | 0.0998 | 294.0000 | 57.6400 | 3.3354 | 0.4810 | 0.1138 |
| bursty | 0.15 | 0.1501 | 312.8200 | 60.4000 | 3.6167 | 0.7387 | 0.1702 |
| bursty | 0.20 | 0.1992 | 336.6000 | 65.0600 | 3.9126 | 0.9979 | 0.2326 |
| bursty | 0.30 | 0.2995 | 375.5600 | 72.0000 | 4.4773 | 1.5460 | 0.3615 |

### 4.2 D3 AUROC (tum anomali) ve normal FAR@0.10 vs r (50 tohum; aralik = tohumlar arasi 2.5-97.5 yuzdelik)
| tur | r | AUROC | AUROC_lo | AUROC_hi | FAR | FAR_lo | FAR_hi | n_pairs |
|---|---|---|---|---|---|---|---|---|
| iid | 0.02 | 0.9622 | 0.9347 | 0.9828 | 0.0551 | 0.0200 | 0.1000 | 50.1 |
| iid | 0.05 | 0.9251 | 0.8779 | 0.9709 | 0.0702 | 0.0200 | 0.1195 | 50.6 |
| iid | 0.10 | 0.8553 | 0.7837 | 0.9256 | 0.0655 | 0.0196 | 0.1334 | 51.8 |
| iid | 0.15 | 0.7909 | 0.7010 | 0.8786 | 0.0567 | 0.0039 | 0.1185 | 54.0 |
| iid | 0.20 | 0.7247 | 0.5953 | 0.8081 | 0.0511 | 0.0038 | 0.0954 | 56.1 |
| iid | 0.30 | 0.6284 | 0.5561 | 0.7078 | 0.0457 | 0.0037 | 0.0970 | 61.0 |
| bursty | 0.02 | 0.8949 | 0.8308 | 0.9412 | 0.0670 | 0.0191 | 0.1275 | 51.5 |
| bursty | 0.05 | 0.8465 | 0.7609 | 0.9068 | 0.0674 | 0.0231 | 0.1144 | 53.7 |
| bursty | 0.10 | 0.7652 | 0.6803 | 0.8402 | 0.0605 | 0.0173 | 0.1259 | 57.6 |
| bursty | 0.15 | 0.7265 | 0.6445 | 0.7972 | 0.0548 | 0.0160 | 0.1118 | 60.4 |
| bursty | 0.20 | 0.6912 | 0.6282 | 0.7473 | 0.0498 | 0.0146 | 0.0958 | 65.1 |
| bursty | 0.30 | 0.6316 | 0.5565 | 0.6754 | 0.0449 | 0.0000 | 0.0882 | 72.0 |

### 4.3 Anomali turu basina D3 AUROC (ortalama [2.5, 97.5])
| tur | r | all | seq135 | seq145 | fast | slow | seq(135+145) | tempo(fast+slow) |
|---|---|---|---|---|---|---|---|---|
| iid | 0.02 | 0.962 [0.935, 0.983] | 0.969 [0.935, 0.990] | 0.969 [0.935, 0.990] | 0.954 [0.923, 0.976] | 0.956 [0.923, 0.980] | 0.969 [0.935, 0.990] | 0.955 [0.931, 0.975] |
| iid | 0.05 | 0.925 [0.878, 0.971] | 0.930 [0.874, 0.987] | 0.929 [0.875, 0.987] | 0.916 [0.868, 0.967] | 0.924 [0.857, 0.964] | 0.929 [0.875, 0.987] | 0.920 [0.864, 0.960] |
| iid | 0.10 | 0.855 [0.784, 0.926] | 0.851 [0.688, 0.955] | 0.847 [0.736, 0.954] | 0.840 [0.753, 0.933] | 0.880 [0.795, 0.947] | 0.849 [0.713, 0.955] | 0.862 [0.786, 0.919] |
| iid | 0.15 | 0.791 [0.701, 0.879] | 0.801 [0.685, 0.926] | 0.813 [0.683, 0.939] | 0.718 [0.578, 0.871] | 0.821 [0.722, 0.907] | 0.807 [0.706, 0.927] | 0.775 [0.681, 0.868] |
| iid | 0.20 | 0.725 [0.595, 0.808] | 0.757 [0.584, 0.891] | 0.764 [0.622, 0.907] | 0.621 [0.528, 0.724] | 0.744 [0.639, 0.823] | 0.760 [0.604, 0.888] | 0.690 [0.579, 0.775] |
| iid | 0.30 | 0.628 [0.556, 0.708] | 0.694 [0.517, 0.834] | 0.705 [0.570, 0.820] | 0.508 [0.430, 0.625] | 0.599 [0.527, 0.684] | 0.699 [0.567, 0.816] | 0.561 [0.496, 0.658] |
| bursty | 0.02 | 0.895 [0.831, 0.941] | 0.894 [0.817, 0.956] | 0.898 [0.830, 0.956] | 0.876 [0.782, 0.932] | 0.910 [0.846, 0.958] | 0.896 [0.827, 0.956] | 0.894 [0.835, 0.940] |
| bursty | 0.05 | 0.847 [0.761, 0.907] | 0.866 [0.760, 0.951] | 0.870 [0.777, 0.958] | 0.782 [0.646, 0.907] | 0.860 [0.742, 0.927] | 0.868 [0.784, 0.952] | 0.824 [0.705, 0.902] |
| bursty | 0.10 | 0.765 [0.680, 0.840] | 0.812 [0.676, 0.908] | 0.844 [0.724, 0.936] | 0.652 [0.544, 0.748] | 0.745 [0.609, 0.844] | 0.827 [0.706, 0.911] | 0.704 [0.592, 0.791] |
| bursty | 0.15 | 0.727 [0.644, 0.797] | 0.795 [0.665, 0.912] | 0.801 [0.662, 0.905] | 0.607 [0.504, 0.701] | 0.697 [0.600, 0.794] | 0.797 [0.676, 0.907] | 0.658 [0.584, 0.752] |
| bursty | 0.20 | 0.691 [0.628, 0.747] | 0.781 [0.659, 0.859] | 0.802 [0.716, 0.877] | 0.538 [0.460, 0.625] | 0.643 [0.545, 0.770] | 0.791 [0.731, 0.845] | 0.598 [0.523, 0.702] |
| bursty | 0.30 | 0.632 [0.556, 0.675] | 0.724 [0.641, 0.801] | 0.741 [0.622, 0.803] | 0.491 [0.410, 0.573] | 0.579 [0.500, 0.663] | 0.731 [0.652, 0.782] | 0.542 [0.465, 0.612] |

### 4.4 Ek tanilama: kanal bazli ham AUROC (D1 dizi, D2 sure; tum anomali) ve D3 aksiyon orani (alfa=0.10)
| tur | r | D1_AUROC | D2_AUROC | det_all | det_seq | det_tempo |
|---|---|---|---|---|---|---|
| iid | 0.02 | 0.8561 | 0.5651 | 0.7871 | 0.8488 | 0.7211 |
| iid | 0.05 | 0.8455 | 0.5437 | 0.5237 | 0.5241 | 0.5237 |
| iid | 0.10 | 0.8130 | 0.5453 | 0.2774 | 0.2314 | 0.3257 |
| iid | 0.15 | 0.7728 | 0.5545 | 0.2056 | 0.2267 | 0.1847 |
| iid | 0.20 | 0.7348 | 0.5604 | 0.1538 | 0.1901 | 0.1181 |
| iid | 0.30 | 0.6535 | 0.5571 | 0.1168 | 0.1763 | 0.0603 |
| bursty | 0.02 | 0.8266 | 0.5581 | 0.3351 | 0.3322 | 0.3389 |
| bursty | 0.05 | 0.8080 | 0.5458 | 0.3171 | 0.3828 | 0.2489 |
| bursty | 0.10 | 0.7638 | 0.5349 | 0.2059 | 0.2876 | 0.1256 |
| bursty | 0.15 | 0.7364 | 0.5411 | 0.1878 | 0.2690 | 0.1093 |
| bursty | 0.20 | 0.7122 | 0.5272 | 0.1800 | 0.2759 | 0.0905 |
| bursty | 0.30 | 0.6579 | 0.5234 | 0.1414 | 0.2209 | 0.0682 |

### 4.5 'Yeterli dogruluk' kurali ve bant (on-kayitli)
Kural: AUROC(tum) >= 0.90 **ve** FAR@0.10 <= alfa + 2SE (SE = sqrt(alfa(1-alfa)/n_pairs), n_pairs o r icin ortalama normal test cifti). r* = kurali saglayan en buyuk izgara r (izgara: 0.02 ... 0.30). Bant: nokta (ortalama AUROC) / alt sinir (AUROC tohum-2.5 yuzdeligi) / ust sinir (97.5 yuzdelik); FAR kosulu hepsinde ortalama FAR ile.
| tur | r_star_nokta | r_star_alt | r_star_ust | saglayan_r_nokta | monoton |
|---|---|---|---|---|---|
| iid | 0.05 | 0.02 | 0.10 | 0.02,0.05 | True |
| bursty | 0.00 | 0.00 | 0.05 | yok | True |
(r* = 0.00 -> izgaranin en kucugu 0.02'de bile kural saglanmiyor.)

- **iid: yeterli dogruluk bandi r in [0.02 (alt), 0.05 (nokta), 0.10 (ust)]** (kare hata orani; referans r=0 AUROC 0.968, FAR 0.060).
- **bursty: yeterli dogruluk bandi r in [0.00 (alt), 0.00 (nokta), 0.05 (ust)]** (kare hata orani; referans r=0 AUROC 0.968, FAR 0.060).

### 4.6 On-yazili tahmin 1: patlamali gurultu iid'den daha dusuk r'de bozar
| r | AUROC_iid | AUROC_bursty | fark_bursty_eksi_iid | FAR_iid | FAR_bursty | bursty_dusuk | tohum_sd_iid | tohum_sd_bursty | fark_MC_SE |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.9622 | 0.8949 | -0.0673 | 0.0551 | 0.0670 | True | 0.0124 | 0.0269 | 0.0042 |
| 0.05 | 0.9251 | 0.8465 | -0.0786 | 0.0702 | 0.0674 | True | 0.0252 | 0.0396 | 0.0066 |
| 0.10 | 0.8553 | 0.7652 | -0.0901 | 0.0655 | 0.0605 | True | 0.0405 | 0.0424 | 0.0083 |
| 0.15 | 0.7909 | 0.7265 | -0.0644 | 0.0567 | 0.0548 | True | 0.0468 | 0.0408 | 0.0088 |
| 0.20 | 0.7247 | 0.6912 | -0.0335 | 0.0511 | 0.0498 | True | 0.0512 | 0.0329 | 0.0086 |
| 0.30 | 0.6284 | 0.6316 | +0.0032 | 0.0457 | 0.0449 | False | 0.0425 | 0.0302 | 0.0074 |

r*(nokta): patlamali 0.00 vs iid 0.05. On-kayitli olcut (r*_patlamali < r*_iid): **GECTI**. Ek: AUROC_patlamali < AUROC_iid 5/6 izgara noktasinda; ortalama fark -0.0551; fark MC SE'leri 0.0042-0.0088.

### 4.7 On-yazili tahmin 2: gercek CNN-GRU / CNN-only noktalari, B2'den uydurulan patlamali egrinin 0.03 AUROC yakininda
| kaynak | rol | r_gozlenen | AUROC_gercek | egri_enterpolasyon | fark | abs_fark_le_003 | dogrudan_bursty_r_obs | fark_dogrudan | iid_enterp | dogrudan_iid_r_obs | FAR_gercek | FAR_sim_bursty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B2 | uydurma-kaynagi (kendi kendine kontrol, tahmin degil) | 0.0983 | 0.9237 | 0.7680 | +0.1557 | False | 0.7822 | +0.1415 | 0.8577 | 0.8572 | 0.0784 | 0.0672 |
| CNN-GRU | TAHMIN | 0.1144 | 0.8788 | 0.7540 | +0.1248 | False | 0.7703 | +0.1085 | 0.8367 | 0.8224 | 0.0816 | 0.0606 |
| CNN-only | TAHMIN | 0.2075 | 0.8349 | 0.6867 | +0.1482 | False | 0.6913 | +0.1436 | 0.7174 | 0.7068 | 0.0577 | 0.0516 |
| B1 | izgara disi, bilgi | 0.3760 | 0.5378 | nan | nan | - | 0.6023 | -0.0645 | nan | 0.5744 | 0.0909 | 0.0468 |

Tahmin 2 (izgara enterpolasyonu; on-kayitli): CNN-GRU GECMEDI, CNN-only GECMEDI -> **GECMEDI** (her ikisi de gerekli). Dogrudan r_obs simulasyonu ile: CNN-GRU GECMEDI, CNN-only GECMEDI. B1 (r=0.376) izgaranin (0.30) disindadir; bursty/iid dogrudan r_obs simulasyonu tabloda. B2 satiri uydurma kaynagi oldugu icin tahmin degil, kendi kendine tutarlilik kontroludur.

Simulasyon gercekci mi? (dongu yapisi tanilamasi; gercek kaynak vs ayni r'de patlamali / iid simulasyon; ToF dongusu basina havuzlanmis oranlar)
| kaynak | r | sahte_gercek | sahte_bursty | sahte_iid | bol_gercek | bol_bursty | bol_iid | ziyaret_gercek | ziyaret_bursty | ziyaret_iid |
|---|---|---|---|---|---|---|---|---|---|---|
| B2 | 0.0983 | 0.2054 | 0.4716 | 0.1121 | 0.0116 | 0.1075 | 0.0172 | 2.9922 | 3.3217 | 2.9266 |
| CNN-GRU | 0.1144 | 0.2597 | 0.5572 | 0.1525 | 0.0349 | 0.1264 | 0.0240 | 3.0233 | 3.4149 | 2.9838 |
| CNN-only | 0.2075 | 0.6124 | 1.0268 | 0.4680 | 0.1667 | 0.2447 | 0.0826 | 3.4380 | 3.9374 | 3.3510 |
| B1 | 0.3760 | 0.8760 | 2.0216 | 1.3658 | 0.1279 | 0.4825 | 0.2402 | 3.8876 | 4.9368 | 4.1199 |

Sekiller: `01_error_propagation.png` (AUROC / FAR vs r, iid + patlamali, gercek 4 kaynak overlay), `01_error_propagation_pertype.png` (tur basina AUROC). Veri: `01_error_propagation.csv` (ozet; tohumlar arasi ortalama/2.5/97.5/sd), `01_error_propagation_seeds.csv` (tohum basina), `01_error_propagation_arm_a.csv` (kol a).

## 5. Ozet ve desteklenen cumle
- **Kol (a):** FAIL (sik kural). Ayrinti 3. bolumde; kaynak bazinda en buyuk |tahmin-gozlenen| = B2 0.892, CNN-GRU 0.711, CNN-only 0.954, B1 1.069.
- **Kol (b) bant (D3, tum anomali):** iid r* = 0.05 [0.02, 0.10], patlamali r* = 0.00 [0.00, 0.05] (r=0: AUROC 0.968). Gercek B2 (r=0.098) AUROC 0.924, CNN-GRU (r=0.114) 0.879, CNN-only (r=0.208) 0.835, B1 (r=0.376) 0.538.
- **Tahmin 1:** GECTI. **Tahmin 2:** GECMEDI (izgara enterpolasyonu).
- **Desteklenen cumle (veriye gore, sentetik gurultu icin):** kalibre edilmis D3 dedektoru, kare hata orani iid gurultude ~%5 (bant %2-%10), B2-uydurmali patlamali gurultude ~%0 (bant %0-%5; 0 = 0.02'de bile AUROC 0.90'in altinda) seviyesine kadar AUROC >= 0.90 saglar ve FAR'i korur; hatanin zaman yapisi (patlama vs bagimsiz) orani kadar onemlidir (ayni r'de AUROC farki -0.090 .. +0.003). Ancak gercek B2 %9.8 hata oraninda AUROC 0.924 verdigi icin sentetik bant gercek hata yapisi icin karamsardir; '~%X' tek sayisi bu analizden gercek hatalar icin cikarilamaz.

**Yorum notu (sayilar yukaridaki tablolardan).** (i) Sentetik gurultu, ayni kare hata oraninda gercek hatalardan cok daha zarar vericidir: B2 r = 0.098'te gercek B2 AUROC 0.924, ayni r'de B2-uydurmali patlamali simulasyon 0.782, iid 0.857; sahte ziyaret / ToF dongusu: gercek 0.205, patlamali sim 0.472, iid sim 0.112. Gercek hatalar L0 gecislerine yogunlasiyor (B2 hata karelerinin near_change payi 0.755, tum karelerde 0.489); bu, gercek hatalarin buyuk bolumunun zamanlama kaymasi oldugu ve (patlamali simulasyona gore) daha az sahte ziyaret urettigi ile tutarlidir (nedensellik bu analizde sinanmadi). Sentetik patlamalar ise zamanda duzgun dagilir. Bu nedenle sentetik egri, bu veri/model icin gercek hata yapisina gore KARAMSAR (alt-sinir benzeri) bir egridir. (ii) Debounce (MIN_BOX=2) tek-kareli bagimsiz hatalari suzer; iid gurultu bu yuzden dusuk r'de zararsiz, >= 2 karelik patlamalar suzgeci gecer (4.1: sahte ziyaret/ToF dongusu iid vs patlamali). (iii) FAR: izgaradaki tum (tur, r) noktalarinda ortalama FAR en fazla 0.0702 (esik alfa+2SE ~ 0.179); FAR kosulu baglayici DEGIL (kalibrasyon bozuk normal donguler uzerinde yapilir), baglayici olan AUROC kosuludur.

## 6. Onceden yazilmis uyarilar
- Gurultu sentetiktir: patlamalar zamanda duzgun dagilir ve gecis karelerinde yogunlasmaz; gercek hatalar gecis-yapilidir (kosu uzunlugu / yanlis etiket secimi gercek model hatalarina bagimli).
- Tek operator, tek gun, tek mesafe; r ekseni 'kare hata orani'dir, hatanin zaman yapisi ayri (iid / patlamali) eksendir.
- Yalniz sorgu hatti bozuldu; B2'nin kendi egitimi / pencere (L=5) yapisi simule edilmedi. 'Dagitilan yolun tamami' = debounce + Box1 giris segmentasyonu + D1/D2 + conformal kalibrasyon (bozuk normal donguler).
- Dongu sayilari bozuk segmentasyonla degisir (bkz. 4.1); anomali dongusu ile normal dongu ayni bozuk hattan gecer; ToF-dongu eslemesi (fold/atype) bozuk girisin ait oldugu ToF dongusunden alinir (Faz A gibi).
- AUROC SE'si ~0.03-0.05 (Faz A kat sd'si); tohum araligi yalniz gurultu gerceklesmesi belirsizligini gosterir, dongu-kumesi (n~250) belirsizligini DEGIL. Bant bu yuzden nokta/alt/ust olarak verilmistir.

_Toplam sure: 77 s._

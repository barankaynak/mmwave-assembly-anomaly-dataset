# 00 — Katlar, pencereler, önbellek (INFRA)

Bu rapor `faz_a/00_folds.py` tarafından üretilir (DESIGN.md ön-kayıt tasarımı; sapma yok). Tüm sayılar stdout'a da yazılır.

## Kare ve pencere
- Kare: **16272** (6 oturum); etiket = ToF < 140 mm, ±3 kare mod yumuşatma (oturum içi, beraberlikte en küçük sınıf; `run_all_pt.derive_labels` ile birebir). Ham etiket (`y_raw`) kayıtlı `box*_active` ve mesafeden türetilen etiketle %100 aynı (assert).
- Heatmap: (16272, 2, 48, 48) float32, X ∈ [−0,5; 0,5], Y ∈ [0,1; 1,0]; kanal 0 = hücre başına en büyük SNR, kanal 1 = hücredeki hızların **toplamı** (orijinal `build_heatmap` ikili 'dolu' göstergesine böler; adı ortalama ama aslında toplamdır — port bunu aynen korur). Vektörleştirilmiş inşa, orijinal döngüyle 600 rastgele karede **maks |fark| = 0**. Ortanca dolu hücre/kare = 4, dolu hücresi olmayan kare = 0.
- Pencere: L = 5, etiket/döngü = son karenin; **16248** pencere (16272 − 6×4). Kare-id boşluğu olan pencere: 3. 5 karesi iki döngüye yayılan pencere: **1256** (pencerenin grubu = son karesinin döngüsü; bu sınır pencereleri katlar arası ≤ 4 karelik örtüşme taşır — S1'de bilinen küçük artık sızıntı, DESIGN'da purge öngörülmemiştir).

| session | pencere | kare | döngü(loop_id) |
|---|---|---|---|
| session_01_training | 4679 | 4683 | 57 |
| session_02_normal_seq125 | 2270 | 2274 | 52 |
| session_03_normal_seq135 | 2403 | 2407 | 58 |
| session_04_normal_seq145 | 1924 | 1928 | 50 |
| session_05_anomaly_fast | 1776 | 1780 | 50 |
| session_06_anomaly_slow | 3196 | 3200 | 53 |

## Sınıf dağılımı (pencere, y_true)
| session | c0 | c1 | c2 | c3 | c4 | c5 | toplam |
|---|---|---|---|---|---|---|---|
| session_01_training | 2953 | 293 | 287 | 278 | 224 | 644 | 4679 |
| session_02_normal_seq125 | 1258 | 372 | 378 | 0 | 0 | 262 | 2270 |
| session_03_normal_seq135 | 1374 | 427 | 0 | 381 | 0 | 221 | 2403 |
| session_04_normal_seq145 | 1022 | 356 | 0 | 0 | 334 | 212 | 1924 |
| session_05_anomaly_fast | 1141 | 345 | 253 | 0 | 0 | 37 | 1776 |
| session_06_anomaly_slow | 1799 | 568 | 486 | 0 | 0 | 343 | 3196 |
| TOPLAM | 9547 | 2361 | 1404 | 659 | 558 | 1719 | 16248 |

## S1 — 5 katlı StratifiedGroupKFold (grup = (session, loop_id), strata = oturum, seed 20261104)
Her pencere tam bir katta (assert); her grup tek katta; her katta her oturum var (assert).

**Pencere sayısı, oturum × kat**
| session | f0 | f1 | f2 | f3 | f4 | toplam |
|---|---|---|---|---|---|---|
| session_01_training | 933 | 942 | 926 | 941 | 937 | 4679 |
| session_02_normal_seq125 | 449 | 446 | 447 | 482 | 446 | 2270 |
| session_03_normal_seq135 | 500 | 467 | 501 | 467 | 468 | 2403 |
| session_04_normal_seq145 | 387 | 390 | 390 | 368 | 389 | 1924 |
| session_05_anomaly_fast | 359 | 360 | 338 | 360 | 359 | 1776 |
| session_06_anomaly_slow | 672 | 619 | 621 | 619 | 665 | 3196 |
| TOPLAM | 3300 | 3224 | 3223 | 3237 | 3264 | 16248 |

**Döngü (grup) sayısı, oturum × kat**
| session | f0 | f1 | f2 | f3 | f4 | toplam |
|---|---|---|---|---|---|---|
| session_01_training | 10 | 11 | 12 | 12 | 12 | 57 |
| session_02_normal_seq125 | 10 | 10 | 11 | 11 | 10 | 52 |
| session_03_normal_seq135 | 12 | 11 | 12 | 11 | 12 | 58 |
| session_04_normal_seq145 | 10 | 10 | 10 | 10 | 10 | 50 |
| session_05_anomaly_fast | 10 | 10 | 10 | 10 | 10 | 50 |
| session_06_anomaly_slow | 11 | 10 | 11 | 10 | 11 | 53 |
| TOPLAM | 63 | 62 | 66 | 64 | 65 | 320 |

**Pencere sayısı, sınıf × kat**
| sınıf | f0 | f1 | f2 | f3 | f4 | toplam |
|---|---|---|---|---|---|---|
| 0 | 1987 | 1870 | 1829 | 1931 | 1930 | 9547 |
| 1 | 468 | 465 | 480 | 459 | 489 | 2361 |
| 2 | 242 | 289 | 287 | 305 | 281 | 1404 |
| 3 | 152 | 113 | 127 | 116 | 151 | 659 |
| 4 | 114 | 112 | 160 | 84 | 88 | 558 |
| 5 | 337 | 375 | 340 | 342 | 325 | 1719 |
| TOPLAM | 3300 | 3224 | 3223 | 3237 | 3264 | 16248 |

## S2 — leave-one-session-out (6 kat)
| fold | session | test_pencere | eğitim_pencere |
|---|---|---|---|
| 0 | session_01_training | 4679 | 11569 |
| 1 | session_02_normal_seq125 | 2270 | 13978 |
| 2 | session_03_normal_seq135 | 2403 | 13845 |
| 3 | session_04_normal_seq145 | 1924 | 14324 |
| 4 | session_05_anomaly_fast | 1776 | 14472 |
| 5 | session_06_anomaly_slow | 3196 | 13052 |

## S0 — rastgele sınıf-tabakalı %80/20 pencere bölmesi (sızıntı gösterimi; random_state 42)
Eğitim **12998**, test **3250** (eski makalenin 3.250 test penceresiyle aynı; assert).

| sınıf | test | toplam | eğitim |
|---|---|---|---|
| 0 | 1910 | 9547 | 7637 |
| 1 | 472 | 2361 | 1889 |
| 2 | 281 | 1404 | 1123 |
| 3 | 132 | 659 | 527 |
| 4 | 111 | 558 | 447 |
| 5 | 344 | 1719 | 1375 |
| TOPLAM | 3250 | 16248 | 12998 |

## Anomali çapraz uyum katları (s02 normal döngüleri; seed 20261104)
Döngüler karıştırılıp (RandomState.permutation) sırayla 5 kata dağıtılır (round-robin) → dengeli ve aralıklı; yalnız **tam** döngüler (ToF kuralı `complete`) katlanır. Loop 0 (ilk Box1 girişinden önceki parça) ve tam olmayanlar `fold = -1` / `use = fragment_dropped`. Roller: test = f; kalibrasyon = (f+1)%5, (f+2)%5; eğitim = (f+3)%5, (f+4)%5. s03–s06 döngüleri her f'de test.

| fold | döngü | kare |
|---|---|---|
| 0 | 11 | 484 |
| 1 | 10 | 420 |
| 2 | 10 | 442 |
| 3 | 10 | 449 |
| 4 | 10 | 474 |
| TOPLAM | 51 | 2269 |

| f | train | cal | test | n_train | n_cal | n_test |
|---|---|---|---|---|---|---|
| 0 | [3, 4] | [1, 2] | [0] | 20 | 20 | 11 |
| 1 | [4, 0] | [2, 3] | [1] | 21 | 20 | 10 |
| 2 | [0, 1] | [3, 4] | [2] | 21 | 20 | 10 |
| 3 | [1, 2] | [4, 0] | [3] | 20 | 21 | 10 |
| 4 | [2, 3] | [0, 1] | [4] | 20 | 21 | 10 |
- Her f için en küçük ulaşılabilir conformal p = 1/(n_cal+1): f0: 0.0476, f1: 0.0476, f2: 0.0476, f3: 0.0455, f4: 0.0455 (α = 0,05 için n_cal ≥ 19 gerekir).

**Tüm oturumlar — döngü kullanımı**
| session | use | döngü | kare |
|---|---|---|---|
| session_01_training | state_training_session | 57 | 4683 |
| session_02_normal_seq125 | fragment_dropped | 1 | 5 |
| session_02_normal_seq125 | normal_crossfit | 51 | 2269 |
| session_03_normal_seq135 | anomaly_test_all_folds | 57 | 2401 |
| session_03_normal_seq135 | fragment_dropped | 1 | 6 |
| session_04_normal_seq145 | anomaly_test_all_folds | 49 | 1922 |
| session_04_normal_seq145 | fragment_dropped | 1 | 6 |
| session_05_anomaly_fast | anomaly_test_all_folds | 49 | 1775 |
| session_05_anomaly_fast | fragment_dropped | 1 | 5 |
| session_06_anomaly_slow | anomaly_test_all_folds | 52 | 3194 |
| session_06_anomaly_slow | fragment_dropped | 1 | 6 |

## Dosyalar
- `folds_state.csv`: pencere başına S1/S2/S0 katı (`s1_fold`, `s2_fold`, `s0_test`), `gid` = (session, loop_id) grup kimliği.
- `folds_anom.csv`: döngü başına anomali kat/rol (`fold`, `use`; the shipped copy has no per-cycle box-order or deviation columns, see README section 3).
- `split_s0.csv`: S0 test bayrağı.
- Önbellek: `$CACHE_DIR/heatmaps_v1_K48_L5.npy`; OOF sözleşmesi: `reports/faz_a/oof/{model}__{split}__f{k}.csv.gz`.


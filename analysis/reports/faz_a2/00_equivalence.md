# A2.0 — Yerel = uzak boru hattı denkliği

`faz_a2/00_equivalence.py` tarafından üretilir (DESIGN_A2.md A2.0; sapma yok). Tüm sayılar stdout'a da yazılır. Süre: 11 s.

## SONUÇ: **PASS**

Ortam: python 3.12.13, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1 (requirements.txt ile aynı sürümler). Yerel heatmap önbelleği: `heatmaps_v1_K48_L5.npy` (CACHE_DIR outside the project folder).

| denetim | sonuç | ayrıntı |
|---|---|---|
| hm_shape | PASS | (16272, 2, 48, 48) float32 |
| hm_fresh_rebuild_equals_cache | PASS |  |
| hm_600_frames_maxdiff_0 | PASS | max|diff|=0 (600 kare, 600 tanesi bos olmayan) |
| n_frames_16272 | PASS | 16272 |
| n_windows_16248 | PASS | 16248 |
| s0_split_3250_12998 | PASS | test 3250 / egitim 12998 |
| loop_counts_per_session_equal | PASS | 01=57; 02=52; 03=58; 04=50; 05=50; 06=53 |
| fold_file_folds_state.csv | PASS | bayt=True bayt(LF-normal)=True deger=True n=16248 |
| fold_file_folds_anom.csv | PASS | bayt=True bayt(LF-normal)=True deger=True n=320 |
| fold_file_split_s0.csv | PASS | bayt=True bayt(LF-normal)=True deger=True n=16248 |
| b2_f0_keys | PASS |  |
| b2_f0_agreement>=99.9% | PASS | 1.000000 (farkli 0/3300); max|dp|=2.86e-08 |
| b2_f1_keys | PASS |  |
| b2_f1_agreement>=99.9% | PASS | 1.000000 (farkli 0/3224); max|dp|=2.86e-08 |

## Heatmap
- Yerel önbellek (16272, 2, 48, 48) float32; orijinal `build_heatmap` döngüsüyle 600 rastgele karede (RandomState(0), `faz_a/00_folds.py` ile aynı seçim) **maks |fark| = 0**. Önbellek zaten vardı; taze yeniden inşa ile birebir eşit: True.

## Sayılar ve katlar
- Kare **16272**, pencere **16248** (16272 − 6×4), S0 test **3250** / eğitim **12998**; s02'de katlanan tam döngü: 51.
- Döngü (loop_id grubu) sayısı, oturum başına (Faz A `folds_state.csv` ile aynı): 01=57, 02=52, 03=58, 04=50, 05=50, 06=53.

| dosya | bayt_ayni | bayt_ayni(CRLF->LF) | deger_ayni | satir | uzak_satir |
|---|---|---|---|---|---|
| folds_state.csv | True | True | True | 16248 | 16248 |
| folds_anom.csv | True | True | True | 320 | 320 |
| split_s0.csv | True | True | True | 16248 | 16248 |

Not: yerel CSV'ler `lineterminator="
"` ile yazılır (Faz A uzak Linux'ta, LF); bayt eşitliği ham ve ayrıca CRLF→LF normalize edilerek raporlanır. Geçme ölçütü: normalize bayt eşitliği **ve** değer eşitliği (float'lar toleranssız).

## B2 (RandomForest) yeniden eğitimi — S1 kat 0 ve 1
`02_state_baselines.py` ile aynı: 7 kare istatistiği × (son, ortalama, eğim) = 21 özellik, RF(300, class_weight=balanced, random_state=20261104; n_jobs=3 — sonuç n_jobs'tan bağımsız), eğitim = S1 katı ≠ f tüm pencereler.

| kat | n_test | anahtarlar_ayni | y_pred_uyum | farkli_tahmin | max_abs_dp | max_abs_dp_f32 | acc_yerel | acc_uzak | mF1_yerel | mF1_uzak |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3300 | True | 1 | 0 | 2.86102e-08 | 0 | 0.868788 | 0.868788 | 0.845535 | 0.845535 |
| 1 | 3224 | True | 1 | 0 | 2.86102e-08 | 0 | 0.897953 | 0.897953 | 0.870145 | 0.870145 |

- y_pred özdeş mi (iki katta da %100): **True**. Olasılıklar: `max |Δp|` yukarıda (OOF dosyası float32'yi metin olarak yazdığından 1e-7 mertebesi yuvarlama beklenir).

Geçme ölçütü: heatmap farkı 0; sayılar ve katlar özdeş; B2 tahmin uyumu ≥ %99,9 (%100 değilse oran raporlanır).

<!-- SELFTEST-BEGIN -->

## A2.0b — `a2_cycles.py` öz-sınaması (Faz A rolleriyle birebir yeniden üretim)

`faz_a2/00b_cycles_selftest.py` tarafından üretilir. Süre: 11 s.

### SONUÇ: **PASS**

| denetim | sonuç | ayrıntı |
|---|---|---|
| rows_equal | PASS | yerel 13265 / Faz A 13265 |
| keys_unique | PASS |  |
| keys_identical | PASS |  |
| counts_per_cell_role_identical | PASS |  |
| outer_merge_all_matched | PASS | {'both': 13265, 'left_only': 0, 'right_only': 0} |
| scores_maxabsdiff<=1e-9 | PASS | max |fark| = 7.11e-15; NaN deseni ayni: True |
| cycle_counts_table_identical | PASS | 50 satir |
| anomaly_table_rows_matched | PASS | yerel 280 / Faz A(D0..D3) 280 |
| anomaly_table_all_cells<=1e-9 | PASS | 21 sutun, max |fark| = 1.11e-16; sorunlu: [] |
| anomaly_perfold_all_rows<=1e-9 | PASS | 1400 satir yerel / 1400 Faz A; max |fark| = 1.11e-16 |
| smoothed_p_u1_equals_standard | PASS |  |
| smoothed_draws_deterministic_and_bounded | PASS |  |
| smoothed_p_nominal_coverage | PASS | [0.048, 0.099, 0.24925] |
| d3_def | PASS |  |
| cross_n_cal_equals_remaining_folds | PASS | n_cal=40 (beklenen 40); egitim/model=[30, 30, 30, 30]; kat sayilari={0: 11, 1: 10, 2: 10, 3: 10, 4: 10} |
| cross_test_matrix_shape | PASS | (218, 4) |
| cross_p_range | PASS | min p=0.0244 |
| cross_fold_k_scores_reproducible | PASS |  |
| cross_p_smoothed_u1_equals_standard | PASS |  |
| cross_p_draws_le_standard | PASS |  |
| fit_params_arbitrary_list_reproduces_train_rows | PASS |  |

- Yerel yeniden hesap: **13265** (dönüş, f, rol) satırı = Faz A `04_cycle_scores.csv.gz` (13265); D0/D1/D2/p0..p3 için **max |fark| = 7.11e-15** (eşik 1e-9); `04_anomaly_table.csv` (D0–D3, 21 sütun × 280 satır) max |fark| = 1.11e-16; `04_anomaly_perfold.csv` max |fark| = 1.11e-16.
- Sütun bazında max |fark|: D0=3.6e-15, D1=4.4e-16, D2=7.1e-15, p0=1.1e-16, p1=1.1e-16, p2=1.1e-16, p3=1.1e-16, dur_s=1.8e-15, d2_nref=0, n_visits=0, end_frame_id=0, anchor_loop=0, tof_fold=0, atype=0, visit_seq=0.

Hücre özeti (yerel; parantez içi Faz A değeri):

| hücre | skor | AUROC seq(135+145) | AUROC tempo(fast+slow) | AUROC tum anomali | FAR@0.10 | FAR@0.05 | tespit@0.10 tüm |
|---|---|---|---|---|---|---|---|
| B2 × pred | D3 | 0.8970 (FA 0.8970) | 0.9518 (FA 0.9518) | 0.9237 (FA 0.9237) | 0.0784 (FA 0.0784) | 0.0000 (FA 0.0000) | 0.6335 (FA 0.6335) |
| B2 × pred | D1 | 0.9737 (FA 0.9737) | 0.5351 (FA 0.5351) | 0.7598 (FA 0.7598) | 0.0588 (FA 0.0588) | 0.0196 (FA 0.0196) | 0.4089 (FA 0.4089) |
| B2 × pred | D2 | 0.5082 (FA 0.5082) | 0.9613 (FA 0.9613) | 0.7291 (FA 0.7291) | 0.0784 (FA 0.0784) | 0.0588 (FA 0.0588) | 0.4867 (FA 0.4867) |
| B2 × tof | D3 | 0.9334 (FA 0.9334) | 0.9556 (FA 0.9556) | 0.9442 (FA 0.9442) | 0.0588 (FA 0.0588) | 0.0000 (FA 0.0000) | 0.5903 (FA 0.5903) |
| B2 × tof | D1 | 0.9702 (FA 0.9702) | 0.5211 (FA 0.5211) | 0.7511 (FA 0.7511) | 0.0588 (FA 0.0588) | 0.0196 (FA 0.0196) | 0.4261 (FA 0.4261) |
| B2 × tof | D2 | 0.5177 (FA 0.5177) | 0.9442 (FA 0.9442) | 0.7258 (FA 0.7258) | 0.0392 (FA 0.0392) | 0.0392 (FA 0.0392) | 0.4473 (FA 0.4473) |
| ToF oracle × tof | D3 | 0.9618 (FA 0.9618) | 0.9565 (FA 0.9565) | 0.9592 (FA 0.9592) | 0.0784 (FA 0.0784) | 0.0000 (FA 0.0000) | 0.8831 (FA 0.8831) |
| ToF oracle × tof | D1 | 1.0000 (FA 1.0000) | 0.7079 (FA 0.7079) | 0.8575 (FA 0.8575) | 0.0000 (FA 0.0000) | 0.0000 (FA 0.0000) | 0.7150 (FA 0.7150) |
| ToF oracle × tof | D2 | 0.2380 (FA 0.2380) | 0.8809 (FA 0.8809) | 0.5517 (FA 0.5517) | 0.0980 (FA 0.0980) | 0.0784 (FA 0.0784) | 0.3034 (FA 0.3034) |
| CNN-GRU × pred | D3 | 0.8492 (FA 0.8492) | 0.9068 (FA 0.9068) | 0.8788 (FA 0.8788) | 0.0816 (FA 0.0816) | 0.0000 (FA 0.0000) | 0.2848 (FA 0.2848) |
| CNN-GRU × pred | D1 | 0.9622 (FA 0.9622) | 0.6219 (FA 0.6219) | 0.7872 (FA 0.7872) | 0.0000 (FA 0.0000) | 0.0000 (FA 0.0000) | 0.2324 (FA 0.2324) |
| CNN-GRU × pred | D2 | 0.3835 (FA 0.3835) | 0.8387 (FA 0.8387) | 0.6176 (FA 0.6176) | 0.1224 (FA 0.1224) | 0.0816 (FA 0.0816) | 0.3067 (FA 0.3067) |

Faz A rolleri: test = f; kalibrasyon = (f+1)%5, (f+2)%5; eğitim = (f+3)%5, (f+4)%5 (n_cal min/ort/maks = 17/20.5/26).


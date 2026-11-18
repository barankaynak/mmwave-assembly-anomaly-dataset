# 00 — Veri şeması ve doğrulama
DATA_PATH = `<DATA_PATH>`  
Bulunan oturumlar (6): session_01_training, session_02_normal_seq125, session_03_normal_seq135, session_04_normal_seq145, session_05_anomaly_fast, session_06_anomaly_slow

## merged_df.csv kolonları
`frame_id`:int64, `timestamp_mmwave`:str, `x`:float64, `y`:float64, `z`:float64, `velocity`:float64, `range`:float64, `azimuth`:float64, `elevation`:float64, `snr`:int64, `noise`:int64, `timestamp_dist`:str, `box1_distance`:float64, `box2_distance`:float64, `box3_distance`:float64, `box4_distance`:float64, `box5_distance`:float64, `box1_active`:int64, `box2_active`:int64, `box3_active`:int64, `box4_active`:int64, `box5_active`:int64, `timestamp`:str, `loop_id`:int64

## Oturum başına özet
| session | scenario | frames_merged | frames_raw | points_merged | points_raw | frame_dt_med_s | tof_dt_med_s | pts_per_frame_med | n_loops | loops_complete | loop_min | loop_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| session_01_training | training | 4683 | 4683 | 33229 | 42096 | 0.100 | 0.032 | 7.000 | 57 | 55 | 0 | 56 |
| session_02_normal_seq125 | normal | 2274 | 2274 | 16273 | 20457 | 0.100 | 0.032 | 7.000 | 52 | 51 | 0 | 51 |
| session_03_normal_seq135 | seq135 | 2407 | 2407 | 17153 | 21437 | 0.100 | 0.032 | 7.000 | 58 | 57 | 0 | 57 |
| session_04_normal_seq145 | seq145 | 1928 | 1928 | 13600 | 17098 | 0.100 | 0.032 | 7.000 | 50 | 49 | 0 | 49 |
| session_05_anomaly_fast | fast | 1780 | 1782 | 13273 | 16249 | 0.100 | 0.032 | 7.000 | 50 | 49 | 0 | 49 |
| session_06_anomaly_slow | slow | 3200 | 3200 | 21451 | 26600 | 0.100 | 0.032 | 6.000 | 53 | 52 | 0 | 52 |

**Toplam kare (merged): 16272** (beklenen 16.272); toplam kare (mmwave_data): 16274
Not: frames_raw − frames_merged farkı (varsa) merged_df'e girmeyen birkaç ham karedir; analiz kare etiketini merged_df'ten alır.  
Not: 314 = 320 loop − 6 (her oturumun loop 0'ı, ilk Box1 girişinden önceki parça). `loop_id` her Box1 girişinde artar.  
**Toplam loop: 320; tam (beklenen sıra/kurala göre): 313** (makale: 314)

## Geometri / hız / etiket kontrolleri
| session | z_eq_0 | elev_eq_0 | vel_unique | absv_lt_0.05 | frames_label_mixed | frames_loop_mixed | rec_vs_rule_merged | rec_vs_rule_tof | multi_active_tof |
|---|---|---|---|---|---|---|---|---|---|
| session_01_training | 1.000 | 1.000 | 14 | 0.000 | 0 | 0 | 1.000 | 1.000 | 0 |
| session_02_normal_seq125 | 1.000 | 1.000 | 11 | 0.000 | 0 | 0 | 1.000 | 1.000 | 0 |
| session_03_normal_seq135 | 1.000 | 1.000 | 12 | 0.000 | 0 | 0 | 1.000 | 1.000 | 0 |
| session_04_normal_seq145 | 1.000 | 1.000 | 11 | 0.000 | 0 | 0 | 1.000 | 1.000 | 0 |
| session_05_anomaly_fast | 1.000 | 1.000 | 15 | 0.000 | 0 | 0 | 1.000 | 1.000 | 0 |
| session_06_anomaly_slow | 1.000 | 1.000 | 15 | 0.000 | 0 | 0 | 1.000 | 1.000 | 0 |

## Kare etiket payları (0 = none)
| session | share_lab0 | share_lab1 | share_lab2 | share_lab3 | share_lab4 | share_lab5 |
|---|---|---|---|---|---|---|
| session_01_training | 0.596 | 0.073 | 0.067 | 0.063 | 0.061 | 0.140 |
| session_02_normal_seq125 | 0.555 | 0.164 | 0.166 | 0.000 | 0.000 | 0.115 |
| session_03_normal_seq135 | 0.561 | 0.177 | 0.000 | 0.158 | 0.000 | 0.103 |
| session_04_normal_seq145 | 0.529 | 0.184 | 0.000 | 0.000 | 0.172 | 0.116 |
| session_05_anomaly_fast | 0.580 | 0.193 | 0.146 | 0.000 | 0.000 | 0.081 |
| session_06_anomaly_slow | 0.563 | 0.177 | 0.152 | 0.000 | 0.000 | 0.107 |

## Hız kuantizasyonu (session_02_normal_seq125): en sık 12 değer
+0.121:3977, -0.121:3932, +0.241:2715, -0.241:2552, -0.362:1186, +0.362:1049, -0.483:436, +0.483:311, -0.604:81, +0.604:30, -0.724:4

## Ham nokta bulutu vs merged_df: düşürülen noktalar
merged_df.csv, mmwave_data.csv'den **tam olarak v == 0 noktalarını** düşürmüş mü? Ve v == 0 noktaları hangi kare etiketinde yoğun?
| session | raw_pts | static_pts | raw_minus_merged | static_share | static_share_lab0 | static_share_lab1 | static_share_lab2 | static_share_lab3 | static_share_lab4 | static_share_lab5 |
|---|---|---|---|---|---|---|---|---|---|---|
| session_01_training | 42096 | 8867 | 8867 | 0.211 | 0.207 | 0.208 | 0.213 | 0.205 | 0.213 | 0.231 |
| session_02_normal_seq125 | 20457 | 4184 | 4184 | 0.205 | 0.196 | 0.215 | 0.201 | 0.000 | 0.000 | 0.245 |
| session_03_normal_seq135 | 21437 | 4284 | 4284 | 0.200 | 0.189 | 0.214 | 0.000 | 0.197 | 0.000 | 0.245 |
| session_04_normal_seq145 | 17098 | 3498 | 3498 | 0.205 | 0.190 | 0.224 | 0.000 | 0.000 | 0.204 | 0.243 |
| session_05_anomaly_fast | 16249 | 2967 | 2976 | 0.183 | 0.174 | 0.193 | 0.189 | 0.000 | 0.000 | 0.206 |
| session_06_anomaly_slow | 26600 | 5149 | 5149 | 0.194 | 0.172 | 0.215 | 0.213 | 0.000 | 0.000 | 0.239 |

## Loop yapısı (ToF segmentlerinden, debounce'lu)
- **session_01_training**: süre medyan 5.60 s (IQR 4.70–10.20); en sık sıralar: `1-4-5`×10; `1-2-5`×7; `1-3-5`×7; `1-5`×3; `1-5-2`×2; tam olmayan loop'lar: 0(``, 0.7s), 56(`1`, 1.5s)
- **session_02_normal_seq125**: süre medyan 5.00 s (IQR 4.50–5.23); en sık sıralar: `1-2-5`×50; ``×1; `1-2-5-5`×1; tam olmayan loop'lar: 0(``, 0.5s)
- **session_03_normal_seq135**: süre medyan 4.60 s (IQR 4.40–5.00); en sık sıralar: `1-3-5`×57; ``×1; tam olmayan loop'lar: 0(``, 1.4s)
- **session_04_normal_seq145**: süre medyan 4.50 s (IQR 4.20–4.87); en sık sıralar: `1-4-5`×48; ``×1; `1-4-4-5`×1; tam olmayan loop'lar: 0(``, 0.6s)
- **session_05_anomaly_fast**: süre medyan 3.80 s (IQR 3.50–4.18); en sık sıralar: `1-2-5`×47; ``×1; `1-2-2-2-5`×1; `1-2-5-5`×1; tam olmayan loop'lar: 0(``, 1.2s)
- **session_06_anomaly_slow**: süre medyan 8.30 s (IQR 7.40–8.70); en sık sıralar: `1-2-5`×52; ``×1; tam olmayan loop'lar: 0(``, 0.6s)

## loop_id sınırı hangi olaya denk geliyor?
- session_01_training: n=56; önceki/sonraki kare etiketi en sık: 0→1×56; son Box5 çıkışından beri medyan 1.13 s; sonraki Box1 girişine medyan 5.48 s
- session_02_normal_seq125: n=51; önceki/sonraki kare etiketi en sık: 0→1×51; son Box5 çıkışından beri medyan 0.63 s; sonraki Box1 girişine medyan 4.89 s
- session_03_normal_seq135: n=57; önceki/sonraki kare etiketi en sık: 0→1×57; son Box5 çıkışından beri medyan 0.61 s; sonraki Box1 girişine medyan 4.44 s
- session_04_normal_seq145: n=49; önceki/sonraki kare etiketi en sık: 0→1×49; son Box5 çıkışından beri medyan 0.58 s; sonraki Box1 girişine medyan 4.38 s
- session_05_anomaly_fast: n=49; önceki/sonraki kare etiketi en sık: 0→1×49; son Box5 çıkışından beri medyan 0.55 s; sonraki Box1 girişine medyan 3.67 s
- session_06_anomaly_slow: n=52; önceki/sonraki kare etiketi en sık: 0→1×52; son Box5 çıkışından beri medyan 1.22 s; sonraki Box1 girişine medyan 8.22 s

## Kalış süreleri: 30 Hz ToF vs 10 Hz kare etiketi (medyan s)
| session | box | dwell_med_s_frames | dwell_med_s_tof | n_visits_frames | n_visits_tof |
|---|---|---|---|---|---|
| session_01_training | 1 | 0.599 | 0.538 | 57.000 | 57.000 |
| session_01_training | 2 | 0.502 | 0.540 | 56.000 | 56.000 |
| session_01_training | 3 | 0.501 | 0.509 | 50.000 | 50.000 |
| session_01_training | 4 | 0.450 | 0.508 | 54.000 | 53.000 |
| session_01_training | 5 | 0.600 | 0.542 | 106.000 | 106.000 |
| session_02_normal_seq125 | 1 | 0.699 | 0.634 | 51.000 | 51.000 |
| session_02_normal_seq125 | 2 | 0.699 | 0.637 | 51.000 | 51.000 |
| session_02_normal_seq125 | 5 | 0.500 | 0.477 | 51.000 | 52.000 |
| session_03_normal_seq135 | 1 | 0.700 | 0.665 | 57.000 | 57.000 |
| session_03_normal_seq135 | 3 | 0.601 | 0.604 | 57.000 | 57.000 |
| session_03_normal_seq135 | 5 | 0.401 | 0.445 | 57.000 | 57.000 |
| session_04_normal_seq145 | 1 | 0.700 | 0.697 | 49.000 | 49.000 |
| session_04_normal_seq145 | 4 | 0.601 | 0.604 | 50.000 | 50.000 |
| session_04_normal_seq145 | 5 | 0.500 | 0.445 | 49.000 | 49.000 |
| session_05_anomaly_fast | 1 | 0.603 | 0.571 | 49.000 | 49.000 |
| session_05_anomaly_fast | 2 | 0.499 | 0.477 | 51.000 | 51.000 |
| session_05_anomaly_fast | 5 | 0.300 | 0.286 | 50.000 | 50.000 |
| session_06_anomaly_slow | 1 | 1.051 | 1.029 | 52.000 | 52.000 |
| session_06_anomaly_slow | 2 | 0.899 | 0.858 | 52.000 | 52.000 |
| session_06_anomaly_slow | 5 | 0.700 | 0.667 | 52.000 | 52.000 |

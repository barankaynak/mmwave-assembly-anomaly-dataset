# 03 — Durum tahmini raporu (Faz A)

`faz_a/03_state_report.py` tarafindan uretilir; DESIGN.md protokolu. Mevcut OOF dosyalari: CNN-GRU/S1 (16248 pencere, 5 kat), CNN-only/S1 (16248 pencere, 5 kat), B1 en-yakin-merkez/S1 (16248 pencere, 5 kat), B2 el-yapimi+RF/S1 (16248 pencere, 5 kat), CNN-GRU/S0 (3250 pencere, 1 kat), CNN-GRU/S2 (16248 pencere, 6 kat), B1 en-yakin-merkez/S2 (16248 pencere, 6 kat), B2 el-yapimi+RF/S2 (16248 pencere, 6 kat).
Etiket = ±3 kare mod yumusatmali ToF etiketi (y_true). Metrikler OOF tahminler uzerinden kat bazli hesaplanir; ortalama ± sd (ddof=1) kat sayisi uzerinden.
macro-F1 yalniz y_true'da bulunan siniflar uzerinden (zero_division=0).

## S1 — 5 katli StratifiedGroupKFold (ana)

### Genel (kat ortalamasi ± sd)
| model | kat | n | accuracy | macro-F1 | F1 none | F1 B1 | F1 B2 | F1 B3 | F1 B4 | F1 B5 |
|---|---|---|---|---|---|---|---|---|---|---|
| CNN-GRU | 5 | 16248 | 0.8693 ± 0.0130 | 0.8408 ± 0.0143 | 0.8877 ± 0.0102 | 0.9042 ± 0.0132 | 0.8053 ± 0.0224 | 0.7751 ± 0.0247 | 0.8422 ± 0.0233 | 0.8302 ± 0.0399 |
| CNN-only | 5 | 16248 | 0.7759 ± 0.0369 | 0.7322 ± 0.0346 | 0.7969 ± 0.0446 | 0.8174 ± 0.0233 | 0.7080 ± 0.0359 | 0.6550 ± 0.0767 | 0.6366 ± 0.0600 | 0.7795 ± 0.0226 |
| B1 en-yakin-merkez | 5 | 16248 | 0.6130 ± 0.0123 | 0.5265 ± 0.0136 | 0.6696 ± 0.0117 | 0.6274 ± 0.0139 | 0.4970 ± 0.0203 | 0.4179 ± 0.0072 | 0.3752 ± 0.0414 | 0.5722 ± 0.0354 |
| B2 el-yapimi+RF | 5 | 16248 | 0.8841 ± 0.0116 | 0.8564 ± 0.0122 | 0.9016 ± 0.0095 | 0.9095 ± 0.0150 | 0.8503 ± 0.0231 | 0.8016 ± 0.0157 | 0.8384 ± 0.0451 | 0.8373 ± 0.0295 |

### Alt kumeler: (a) kutu kareleri, (b) none kareleri, (c) gecis kareleri (etiket degisimine ±2 kare)
(b) icin accuracy = none sinifinin recall'u. macro-F1 alt kumedeki mevcut siniflar uzerinden. Kat basina hesap, ortalama ± sd.
| model | (a) kutu n | (a) kutu acc | (a) kutu mF1 | (b) none n | (b) none acc | (c) gecis n | (c) gecis acc | (c) gecis mF1 |
|---|---|---|---|---|---|---|---|---|
| CNN-GRU | 6701 | 0.8942 ± 0.0206 | 0.9343 ± 0.0077 | 9547 | 0.8518 ± 0.0171 | 7727 | 0.8178 ± 0.0109 | 0.8098 ± 0.0086 |
| CNN-only | 6701 | 0.8580 ± 0.0098 | 0.8914 ± 0.0046 | 9547 | 0.7181 ± 0.0622 | 7727 | 0.7695 ± 0.0556 | 0.7579 ± 0.0515 |
| B1 en-yakin-merkez | 6701 | 0.5497 ± 0.0341 | 0.6854 ± 0.0283 | 9547 | 0.6584 ± 0.0313 | 7727 | 0.5801 ± 0.0135 | 0.5463 ± 0.0114 |
| B2 el-yapimi+RF | 6701 | 0.9325 ± 0.0078 | 0.9473 ± 0.0063 | 9547 | 0.8501 ± 0.0143 | 7727 | 0.8342 ± 0.0113 | 0.8278 ± 0.0097 |

### (d) Oturum basina (OOF havuzlanmis; s01 = kutular cozunurluk altinda)
| session | CNN-GRU acc | CNN-GRU mF1 | CNN-only acc | CNN-only mF1 | B1 en-yakin-merkez acc | B1 en-yakin-merkez mF1 | B2 el-yapimi+RF acc | B2 el-yapimi+RF mF1 | n |
|---|---|---|---|---|---|---|---|---|---|
| session_01 **(s01)** | 0.829 | 0.774 | 0.736 | 0.678 | 0.587 | 0.449 | 0.841 | 0.784 | 4679 |
| session_02 | 0.911 | 0.906 | 0.831 | 0.838 | 0.619 | 0.620 | 0.920 | 0.914 | 2270 |
| session_03 | 0.898 | 0.885 | 0.823 | 0.824 | 0.641 | 0.627 | 0.908 | 0.897 | 2403 |
| session_04 | 0.887 | 0.880 | 0.805 | 0.810 | 0.615 | 0.604 | 0.909 | 0.902 | 1924 |
| session_05 | 0.847 | 0.726 | 0.764 | 0.675 | 0.588 | 0.497 | 0.873 | 0.758 | 1776 |
| session_06 | 0.878 | 0.872 | 0.749 | 0.773 | 0.637 | 0.627 | 0.895 | 0.889 | 3196 |

**Eslesik kat farki (macro-F1; fark = CNN-GRU − diger):**
- CNN-only: fark +0.1086 (sd 0.0433, 5 kat). DESIGN sart 1: fark <= 0.02 ise taban 'yakin'.
- B1 en-yakin-merkez: fark +0.3142 (sd 0.0152, 5 kat). DESIGN sart 1: fark <= 0.02 ise taban 'yakin'.
- B2 el-yapimi+RF: fark -0.0157 (sd 0.0059, 5 kat). DESIGN sart 1: fark <= 0.02 ise taban 'yakin'.

## S2 — leave-one-session-out (6 kat)

### Genel (kat ortalamasi ± sd)
| model | kat | n | accuracy | macro-F1 | F1 none | F1 B1 | F1 B2 | F1 B3 | F1 B4 | F1 B5 |
|---|---|---|---|---|---|---|---|---|---|---|
| CNN-GRU | 6 | 16248 | 0.8415 ± 0.0334 | 0.8034 ± 0.0778 | 0.8615 ± 0.0250 | 0.8665 ± 0.0852 | 0.6797 ± 0.2247 | 0.7306 ± 0.1304 | 0.7952 ± 0.0820 | 0.7519 ± 0.1851 |
| B1 en-yakin-merkez | 6 | 16248 | 0.6145 ± 0.0202 | 0.5676 ± 0.0755 | 0.6706 ± 0.0106 | 0.6283 ± 0.0704 | 0.4625 ± 0.2082 | 0.4370 ± 0.1169 | 0.4344 ± 0.1178 | 0.5231 ± 0.1507 |
| B2 el-yapimi+RF | 6 | 16248 | 0.8682 ± 0.0458 | 0.8286 ± 0.0938 | 0.8911 ± 0.0263 | 0.8751 ± 0.0958 | 0.7407 ± 0.1905 | 0.7649 ± 0.1709 | 0.8065 ± 0.1142 | 0.7647 ± 0.1970 |

### Alt kumeler: (a) kutu kareleri, (b) none kareleri, (c) gecis kareleri (etiket degisimine ±2 kare)
(b) icin accuracy = none sinifinin recall'u. macro-F1 alt kumedeki mevcut siniflar uzerinden. Kat basina hesap, ortalama ± sd.
| model | (a) kutu n | (a) kutu acc | (a) kutu mF1 | (b) none n | (b) none acc | (c) gecis n | (c) gecis acc | (c) gecis mF1 |
|---|---|---|---|---|---|---|---|---|
| CNN-GRU | 6701 | 0.8707 ± 0.0711 | 0.9151 ± 0.0708 | 9547 | 0.8163 ± 0.0544 | 7727 | 0.7791 ± 0.0552 | 0.7792 ± 0.0782 |
| B1 en-yakin-merkez | 6701 | 0.5398 ± 0.0572 | 0.6913 ± 0.0693 | 9547 | 0.6659 ± 0.0236 | 7727 | 0.5856 ± 0.0198 | 0.5718 ± 0.0580 |
| B2 el-yapimi+RF | 6701 | 0.8868 ± 0.0996 | 0.9235 ± 0.0760 | 9547 | 0.8501 ± 0.0521 | 7727 | 0.8027 ± 0.0729 | 0.7992 ± 0.0857 |

### (d) Oturum basina (OOF havuzlanmis; s01 = kutular cozunurluk altinda)
| session | CNN-GRU acc | CNN-GRU mF1 | B1 en-yakin-merkez acc | B1 en-yakin-merkez mF1 | B2 el-yapimi+RF acc | B2 el-yapimi+RF mF1 | n |
|---|---|---|---|---|---|---|---|
| session_01 **(s01)** | 0.788 | 0.684 | 0.594 | 0.451 | 0.794 | 0.703 | 4679 |
| session_02 | 0.857 | 0.860 | 0.628 | 0.613 | 0.910 | 0.905 | 2270 |
| session_03 | 0.874 | 0.857 | 0.626 | 0.611 | 0.902 | 0.891 | 2403 |
| session_04 | 0.871 | 0.869 | 0.621 | 0.617 | 0.900 | 0.892 | 1924 |
| session_05 | 0.840 | 0.730 | 0.584 | 0.493 | 0.834 | 0.715 | 1776 |
| session_06 | 0.819 | 0.820 | 0.633 | 0.622 | 0.869 | 0.865 | 3196 |

**Eslesik kat farki (macro-F1; fark = CNN-GRU − diger):**
- B1 en-yakin-merkez: fark +0.2358 (sd 0.0195, 6 kat). DESIGN sart 1: fark <= 0.02 ise taban 'yakin'.
- B2 el-yapimi+RF: fark -0.0252 (sd 0.0225, 6 kat). DESIGN sart 1: fark <= 0.02 ise taban 'yakin'.

## Sizinti etkisi: S0 (rastgele pencere bolmesi) vs S1 (grup bazli) — CNN-GRU
| kosul | accuracy | macro-F1 |
|---|---|---|
| Keras (eski makale, S0, 3250 test) | 0.9366 | nan |
| PyTorch port S0 (n=3250) | 0.9009 | 0.8781 |
| PyTorch port S1 (kat ort.) | 0.8693 | 0.8408 |

**Sizinti etkisi (S0_pt − S1_pt): accuracy +0.0316, macro-F1 +0.0373.**
Port farki (S0_pt − Keras 0.9366): accuracy -0.0357.

Port farklari (DESIGN): Conv→BN→ReLU sirasi; Keras GRU dropout 0.3/recurrent 0.2 yerine GRU cikisinda dropout 0.3; kayitli 89.4 vs 93.7 ablasyon farki esit-epoch (40 vs 120) karsilastirmadan kaynaklanir.

## Parametre sayisi ve CPU cikarim suresi (pencere basina)
| model | parametre | CPU ms/pencere |
|---|---|---|
| B1 en-yakin-merkez | 5 merkez (10 deger) + tau | 0.0120 |
| B2 el-yapimi+RF | RF 300 agac, 21 ozellik | 13.323 (tek is parcacigi, ozellik cikarimi dahil) |

cnn_runs.csv (ham):

| model | split | fold | epochs_run | best_epoch | best_val_loss | wall_s | n_params | device | n_train | n_val | n_test | test_acc | test_macro_f1 | max_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnngru | S1 | 0 | 74 | 49 | 0.361 | 372.900 | 1921446 | cuda | 11749 | 1199 | 3300 | 0.855 | 0.827 | 120 |
| cnngru | S1 | 1 | 74 | 49 | 0.624 | 364.400 | 1921446 | cuda | 11422 | 1602 | 3224 | 0.886 | 0.858 | 120 |
| cnngru | S1 | 2 | 67 | 42 | 0.487 | 331.900 | 1921446 | cuda | 11476 | 1549 | 3223 | 0.877 | 0.853 | 120 |
| cnngru | S1 | 3 | 73 | 48 | 0.285 | 372.000 | 1921446 | cuda | 11751 | 1260 | 3237 | 0.871 | 0.837 | 120 |
| cnngru | S1 | 4 | 70 | 45 | 0.362 | 351.100 | 1921446 | cuda | 11753 | 1231 | 3264 | 0.858 | 0.828 | 120 |
| cnngru | S0 | 0 | 79 | 54 | 0.280 | 395.900 | 1921446 | cuda | 11698 | 1300 | 3250 | 0.901 | 0.878 | 120 |
| cnnonly | S1 | 0 | 74 | 49 | 0.488 | 159.500 | 388774 | cuda | 11749 | 1199 | 3300 | 0.783 | 0.744 | 120 |
| cnnonly | S1 | 1 | 45 | 20 | 0.827 | 95.600 | 388774 | cuda | 11422 | 1602 | 3224 | 0.711 | 0.674 | 120 |
| cnnonly | S1 | 2 | 83 | 58 | 0.614 | 177.200 | 388774 | cuda | 11476 | 1549 | 3223 | 0.802 | 0.764 | 120 |
| cnnonly | S1 | 3 | 95 | 70 | 0.484 | 202.500 | 388774 | cuda | 11751 | 1260 | 3237 | 0.787 | 0.733 | 120 |
| cnnonly | S1 | 4 | 90 | 65 | 0.550 | 193.000 | 388774 | cuda | 11753 | 1231 | 3264 | 0.797 | 0.746 | 120 |
| cnngru | S2 | 0 | 63 | 38 | 0.328 | 281.600 | 1921446 | cuda | 10444 | 1125 | 4679 | 0.788 | 0.684 | 120 |
| cnngru | S2 | 1 | 60 | 35 | 0.337 | 325.200 | 1921446 | cuda | 12611 | 1367 | 2270 | 0.857 | 0.860 | 120 |
| cnngru | S2 | 2 | 76 | 51 | 0.435 | 402.500 | 1921446 | cuda | 12271 | 1574 | 2403 | 0.874 | 0.857 | 120 |
| cnngru | S2 | 3 | 67 | 42 | 0.465 | 370.900 | 1921446 | cuda | 12821 | 1503 | 1924 | 0.871 | 0.869 | 120 |
| cnngru | S2 | 4 | 66 | 41 | 0.379 | 366.000 | 1921446 | cuda | 12989 | 1483 | 1776 | 0.840 | 0.730 | 120 |
| cnngru | S2 | 5 | 69 | 44 | 0.387 | 351.300 | 1921446 | cuda | 11983 | 1069 | 3196 | 0.819 | 0.820 | 120 |

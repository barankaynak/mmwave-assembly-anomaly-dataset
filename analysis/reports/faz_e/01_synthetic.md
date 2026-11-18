# Phase E - E1 normal side and synthetic battery (secondary analysis) [S]

**Written:** 2026-11-06 01:30 Turkey Standard Time by `faz_e/01_e1_synthetic.py` (run time 6 s, CPU). **Plan:** `faz_e/DESIGN_E.md`, frozen part sha256 `d8c2cafd4a788f0b87e8e222e2ec8ae4e6e8dc74c81b687a885b7207df168d76` (equals the value logged in `paper/review/CHANGELOG_A4.md`, item 12).

**Status.** Secondary analysis, planned after the Phase A-D results were known (DESIGN_E section 1). The Phase A rule (D3, B2 x pred, alpha = 0.10) stays the planned primary result. This report covers sections 3 and 4 of the plan: normal-side quantities and the synthetic battery, computed on **s01 and s02 only**. No session s03-s06 was loaded (OOF / fold / score tables filtered to s01+s02 at load, chunk by chunk; the all-session frame table of `fa_common` was never built; every raw-data loader was replaced by a tripwire, none fired). The one-shot script `faz_e/02_e1_oneshot.py` was written but **not run**.

**Verdict:** all acceptance items of section 4 do NOT all pass (see section 5).

## 1. Rule (DESIGN_E section 2-3, as implemented)
- Visits: maximal runs of the debounced B2 label (Phase A debouncer MIN_BOX 2, MAX_GAP 1); evidence E = sum of p_b over the run. Collapse: consecutive same-box visits (only none frames between) merge, masses add.
- Per fold f: route R = modal collapsed reading of the training normals (folds f+3, f+4); mu_b = mean whole-cycle posterior mass of box b over them.
- S_E = multi-match evidence-weighted edit distance (insertion = visit mass, deletion = mu of the step, an open step absorbs later visits of its own box at zero cost). p_E = (1 + #{cal >= s - 1e-12}) / (n_cal + 1) on the calibration normals (folds f+1, f+2). D3' = min{1, 2 min(p_E, p_2)}, alarm if D3' <= 0.10; p_2 = unchanged Phase A D2 p-value.
- Unit cases of section 2.4: 16 DP cases asserted at import of `faz_e/e1_core.py` (masses chosen so every table row is the minimising path; a second, realistic-mass set asserts the general identities `1-5-2` = min(mu_2 + E_2, mu_5 + E_5) and `1-2-5-2` = min(E_2, E_5 + mu_5)), plus collapse, traceback and plain-variant asserts.

## 2. Phase A reproduction gate
The 51 s02 normal B2 x pred cycles were rebuilt with `faz_a2/a2_cycles.py` (`labels_from_oof`, `build_cycles`, `score_group`) from the s01/s02 OOF rows and compared with `reports/faz_a/04_cycle_scores.csv.gz` (b2, pred, s02, normal): 255 of 255 (cycle, f, role) rows matched; start/end frame ids and visit readings identical; max |diff| D0 4.4e-16, D1 4.4e-16, D2 4.4e-16, p0-p3 1.1e-16. **Gate passed.** E1 visits equal the Phase A visits for all 51 cycles, and re-debouncing a cycle in isolation equals the session-wide debounce for all 51 (this is what the composites rely on).

## 3. Normal side (section 3)
Per fold (route, mu and calibration from real s02 normal cycles; 'plain' = the pre-fix collapse rule, tabulated only):

| f | route | mu_1 | mu_2 | mu_5 | n_train | n_cal | n_test | cal_max | n_cal_pos | cal_max_plain | train_readings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1-2-5 | 8.373 | 8.046 | 5.640 | 21 | 19 | 11 | 1.090 | 1 | 1.090 | 1-2-5 x21 |
| 1 | 1-2-5 | 8.304 | 8.524 | 5.554 | 22 | 20 | 9 | 0.000 | 0 | 0.000 | 1-2-5 x22 |
| 2 | 1-2-5 | 7.996 | 8.254 | 5.485 | 20 | 21 | 10 | 0.000 | 0 | 0.000 | 1-2-5 x19, 1-3-2-5 x1 |
| 3 | 1-2-5 | 8.032 | 7.858 | 5.740 | 19 | 22 | 10 | 0.000 | 0 | 0.000 | 1-2-5 x18, 1-3-2-5 x1 |
| 4 | 1-2-5 | 8.563 | 8.145 | 5.571 | 20 | 20 | 11 | 1.090 | 1 | 1.090 | 1-2-5 x20 |

- 51 normal test S_E: **50 x 0**; positive: cycle 19 (f1, reading 1-3-2-5) **1.0900**.
- FAR p_E <= 0.10: 1/51 = 0.020 (cycles [19]); p_E <= 0.05: 1/51.
- FAR **D3' <= 0.10: 4/51 = 0.078** (cycles [0, 4, 19, 48]); Phase A D3 (loaded table): 4/51 = 0.078 (cycles [0, 4, 19, 48]). Binomial SE at alpha 0.10 and n = 51: 0.042 (2SE band 0.016-0.184). **Same alarm set: True.**

| cycle_idx | f | reading | S_E | p_E | p1 | p2 | D3 (Phase A) | D3' | path |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 1-2-5 | 0.000 | 1.000 | 1.000 | 0.043 | 0.087 | 0.087 | p_2 |
| 4 | 2 | 1-2-5 | 0.000 | 1.000 | 1.000 | 0.045 | 0.091 | 0.091 | p_2 |
| 19 | 1 | 1-3-2-5 | 1.090 | 0.048 | 0.048 | 0.429 | 0.095 | 0.095 | p_E |
| 48 | 4 | 1-2-2-5 | 0.000 | 1.000 | 0.095 | 0.048 | 0.095 | 0.095 | p_2 |

The decision layer is degenerate, as the plan states: 50/51 normal scores are exactly 0 and the calibration maximum is 0 in folds 1-3, so there any collapsed off-route visit alarms; the one positive score is the 0.2 s reach flicker of cycle 19.

## 4. Synthetic battery (section 4)
**Material.** Parents: 50 s02 normal cycles in their Phase A test role with collapsed reading 1-2-5 (per fold {0: 11, 1: 8, 2: 10, 3: 10, 4: 11}). Foreign runs: s01 ToF-true runs inside single complete loops (counts per box {1: 40, 2: 46, 3: 43, 4: 30, 5: 101}; used: 43 Box-3 and 30 Box-4 runs) with their own B2 S1 OOF labels and posteriors. Flicker block: cycle 19, frames 859-860, raw [3, 3], p3 [0.7033, 0.3867], mass 1.0900. Identity check (51 cycles rebuilt unmodified as composites): max |diff| S_E 0.0e+00, D1 0.0e+00, D2 0.0e+00, p_2 0.0e+00; D3' flips 0.

Columns: S_E (frozen multi-match) and plain (pre-fix rule) min/median/max; detection = share alarmed: D3' <= 0.10 (the Phase E rule), p_E <= 0.10 / 0.05, Phase A D3 <= 0.10, Phase A D1 p_1 <= 0.05 (its D3 path), D2 p_2 <= 0.05, and D3' built with the plain score. D1/D2 are recomputed for every synthetic cycle.

| kind | variant | n | S_E_min | S_E_med | S_E_max | plain_min | plain_med | plain_max | det_D3' | det_pE10 | det_pE05 | det_D3(A) | det_D1_05 | det_p2_05 | det_D3'plain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| omission_2 |  | 50 | 7.858 | 8.145 | 8.524 | 7.858 | 8.145 | 8.524 | 1.000 | 1.000 | 1.000 | 0.380 | 0.360 | 0.020 | 1.000 |
| swap_1-5-2 |  | 50 | 9.301 | 10.329 | 12.995 | 9.301 | 10.329 | 12.995 | 1.000 | 1.000 | 1.000 | 0.820 | 0.800 | 0.040 | 1.000 |
| repeat_2_far |  | 50 | 3.063 | 5.822 | 10.150 | 3.063 | 5.822 | 17.300 | 1.000 | 1.000 | 1.000 | 0.820 | 0.800 | 0.060 | 1.000 |
| repeat_2_immediate |  | 50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.060 | 0.000 | 0.000 | 0.060 | 0.000 | 0.060 | 0.060 |
| subst_3 | all runs | 2150 | 0.000 | 11.328 | 22.525 | 0.000 | 11.328 | 22.525 | 0.669 | 0.653 | 0.653 | 0.148 | 0.119 | 0.035 | 0.669 |
| subst_4 | all runs | 1500 | 9.808 | 13.643 | 20.628 | 9.808 | 13.643 | 20.628 | 1.000 | 1.000 | 1.000 | 0.373 | 0.353 | 0.020 | 1.000 |
| foreign_3 | all runs | 2150 | 0.000 | 3.320 | 13.723 | 0.000 | 3.320 | 13.723 | 0.983 | 0.651 | 0.641 | 0.876 | 0.521 | 0.386 | 0.983 |
| foreign_4 | all runs | 1500 | 1.950 | 5.475 | 12.103 | 1.950 | 5.475 | 12.103 | 1.000 | 1.000 | 1.000 | 0.820 | 0.800 | 0.060 | 1.000 |
| truncation | keep0 | 50 | 5.485 | 5.571 | 5.740 | 5.485 | 5.571 | 5.740 | 1.000 | 1.000 | 1.000 | 0.800 | 0.800 | 0.040 | 1.000 |
| truncation | keep1 | 50 | 5.485 | 5.571 | 5.740 | 5.485 | 5.571 | 5.740 | 1.000 | 1.000 | 1.000 | 0.800 | 0.800 | 0.040 | 1.000 |
| truncation | keep2 | 50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| truncation | keep3 | 50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| tempo | x0.60 | 50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.920 | 0.000 | 0.000 | 0.920 | 0.000 | 0.920 | 0.920 |
| tempo | x0.70 | 50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.580 | 0.000 | 0.000 | 0.580 | 0.000 | 0.580 | 0.580 |
| tempo | x0.80 | 50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.120 | 0.000 | 0.000 | 0.120 | 0.000 | 0.120 | 0.120 |
| tempo | x1.25 | 50 | 0.000 | 0.000 | 1.407 | 0.000 | 0.000 | 2.730 | 0.240 | 0.040 | 0.020 | 0.240 | 0.020 | 0.220 | 0.240 |
| tempo | x1.40 | 50 | 0.000 | 0.000 | 1.407 | 0.000 | 0.000 | 2.730 | 0.520 | 0.020 | 0.020 | 0.500 | 0.000 | 0.500 | 0.520 |
| tempo | x1.70 | 50 | 0.000 | 0.000 | 1.407 | 0.000 | 0.000 | 3.470 | 0.960 | 0.060 | 0.040 | 0.960 | 0.020 | 0.960 | 0.960 |
| flicker_reach |  | 50 | 1.090 | 1.090 | 1.090 | 1.090 | 1.090 | 1.090 | 0.580 | 1.000 | 0.560 | 0.200 | 0.160 | 0.040 | 0.580 |
| flicker_mid_dwell |  | 46 | 1.090 | 1.090 | 1.793 | 2.670 | 3.755 | 9.637 | 0.609 | 1.000 | 0.587 | 0.804 | 0.783 | 0.065 | 1.000 |
| flicker_mid_dwell_short |  | 4 | 1.090 | 1.090 | 1.090 | 2.380 | 2.920 | 2.987 | 0.250 | 1.000 | 0.250 | 1.000 | 1.000 | 0.000 | 1.000 |
| split_visit | gap2 | 50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.060 | 0.000 | 0.000 | 0.060 | 0.000 | 0.060 | 0.060 |
| split_visit | gap3 | 50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.060 | 0.000 | 0.000 | 0.060 | 0.000 | 0.060 | 0.060 |
| split_visit | gap5 | 50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.060 | 0.000 | 0.000 | 0.060 | 0.000 | 0.060 | 0.060 |

Collapsed readings (most frequent):

| kind | variant | n | readings |
|---|---|---|---|
| omission_2 |  | 50 | 1-5 x50 |
| swap_1-5-2 |  | 50 | 1-5-2 x50 |
| repeat_2_far |  | 50 | 1-2-5-2 x50 |
| repeat_2_immediate |  | 50 | 1-2-5 x50 |
| subst_3 | all runs | 2150 | 1-3-5 x1350, 1-2-5 x747, 1-2-3-5 x50, 1-3-2-5 x3 |
| subst_4 | all runs | 1500 | 1-4-5 x1450, 1-3-5 x50 |
| foreign_3 | all runs | 2150 | 1-2-3-5 x1400, 1-2-5 x750 |
| foreign_4 | all runs | 1500 | 1-2-4-5 x1450, 1-2-3-5 x50 |
| truncation | keep0 | 50 | 1-2 x50 |
| truncation | keep1 | 50 | 1-2 x50 |
| truncation | keep2 | 50 | 1-2-5 x50 |
| truncation | keep3 | 50 | 1-2-5 x50 |
| tempo | x0.60 | 50 | 1-2-5 x50 |
| tempo | x0.70 | 50 | 1-2-5 x50 |
| tempo | x0.80 | 50 | 1-2-5 x50 |
| tempo | x1.25 | 50 | 1-2-5 x48, 1-2-4-5 x1, 1-2-3-2-5 x1 |
| tempo | x1.40 | 50 | 1-2-5 x49, 1-2-3-2-5 x1 |
| tempo | x1.70 | 50 | 1-2-5 x47, 1-2-4-5 x1, 1-3-2-5 x1, 1-2-3-2-5 x1 |
| flicker_reach |  | 50 | 1-3-2-5 x50 |
| flicker_mid_dwell |  | 46 | 1-2-3-2-5 x46 |
| flicker_mid_dwell_short |  | 4 | 1-2-3-2-5 x4 |
| split_visit | gap2 | 50 | 1-2-5 x50 |
| split_visit | gap3 | 50 | 1-2-5 x50 |
| split_visit | gap5 | 50 | 1-2-5 x50 |

**Substitution / foreign insertion by readability** (readable = the debounced composite has a visit of the pasted box):

| kind | readable | n | S_E_min | S_E_med | S_E_max | det_D3' | det_pE10 | det_pE05 | det_D3(A) | det_D1_05 | det_p2_05 | det_D3'plain | readings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| subst_3 | 1.000 | 1403 | 1.117 | 12.965 | 22.525 | 1.000 | 1.000 | 1.000 | 0.202 | 0.182 | 0.029 | 1.000 | 1-3-5 x1350, 1-2-3-5 x50, 1-3-2-5 x3 |
| subst_3 | 0.000 | 747 | 0.000 | 0.000 | 0.000 | 0.047 | 0.000 | 0.000 | 0.047 | 0.000 | 0.047 | 0.047 | 1-2-5 x747 |
| subst_4 | 1.000 | 1450 | 9.808 | 13.668 | 20.628 | 1.000 | 1.000 | 1.000 | 0.380 | 0.360 | 0.020 | 1.000 | 1-4-5 x1450 |
| subst_4 | 0.000 | 50 | 10.028 | 10.315 | 11.165 | 1.000 | 1.000 | 1.000 | 0.180 | 0.160 | 0.020 | 1.000 | 1-3-5 x50 |
| foreign_3 | 1.000 | 1400 | 1.063 | 4.875 | 13.723 | 0.985 | 1.000 | 0.984 | 0.820 | 0.800 | 0.067 | 0.985 | 1-2-3-5 x1400 |
| foreign_3 | 0.000 | 750 | 0.000 | 0.000 | 0.000 | 0.980 | 0.000 | 0.000 | 0.980 | 0.000 | 0.980 | 0.980 | 1-2-5 x750 |
| foreign_4 | 1.000 | 1450 | 1.950 | 5.597 | 12.103 | 1.000 | 1.000 | 1.000 | 0.820 | 0.800 | 0.060 | 1.000 | 1-2-4-5 x1450 |
| foreign_4 | 0.000 | 50 | 2.170 | 2.170 | 2.170 | 1.000 | 1.000 | 1.000 | 0.820 | 0.800 | 0.060 | 1.000 | 1-2-3-5 x50 |

Readability is mostly a property of the s01 run (B2 reads it as the pasted box or not, whatever the parent): runs readable in all / no / some parents: subst_3 28/12/3; subst_4 29/1/0. Per-run rows: `reports/faz_e/01_synthetic_per_run.csv`.

**Per fold** (kinds 1-3, 5, 6):

| kind | f | n | det_D3p | det_D1_05 | S_E_min | S_E_med |
|---|---|---|---|---|---|---|
| omission_2 | 0 | 11 | 1.000 | 0.000 | 8.046 | 8.046 |
| omission_2 | 1 | 8 | 1.000 | 1.000 | 8.524 | 8.524 |
| omission_2 | 2 | 10 | 1.000 | 0.000 | 8.254 | 8.254 |
| omission_2 | 3 | 10 | 1.000 | 1.000 | 7.858 | 7.858 |
| omission_2 | 4 | 11 | 1.000 | 0.000 | 8.145 | 8.145 |
| swap_1-5-2 | 0 | 11 | 1.000 | 1.000 | 9.850 | 10.233 |
| swap_1-5-2 | 1 | 8 | 1.000 | 1.000 | 9.964 | 10.726 |
| swap_1-5-2 | 2 | 10 | 1.000 | 0.000 | 9.755 | 10.245 |
| swap_1-5-2 | 3 | 10 | 1.000 | 1.000 | 9.367 | 10.272 |
| swap_1-5-2 | 4 | 11 | 1.000 | 1.000 | 9.301 | 10.751 |
| repeat_2_far | 0 | 11 | 1.000 | 1.000 | 4.363 | 5.870 |
| repeat_2_far | 1 | 8 | 1.000 | 1.000 | 5.223 | 5.727 |
| repeat_2_far | 2 | 10 | 1.000 | 0.000 | 3.493 | 5.890 |
| repeat_2_far | 3 | 10 | 1.000 | 1.000 | 3.063 | 6.728 |
| repeat_2_far | 4 | 11 | 1.000 | 1.000 | 4.320 | 5.630 |
| subst_3 | 0 | 473 | 0.651 | 0.023 | 0.000 | 11.246 |
| subst_3 | 1 | 344 | 0.651 | 0.651 | 0.000 | 11.724 |
| subst_3 | 2 | 430 | 0.686 | 0.000 | 0.000 | 11.453 |
| subst_3 | 3 | 430 | 0.698 | 0.023 | 0.000 | 11.058 |
| subst_3 | 4 | 473 | 0.658 | 0.023 | 0.000 | 11.345 |
| subst_4 | 0 | 330 | 1.000 | 0.000 | 9.996 | 13.521 |
| subst_4 | 1 | 240 | 1.000 | 1.000 | 10.474 | 13.999 |
| subst_4 | 2 | 300 | 1.000 | 0.000 | 10.203 | 13.729 |
| subst_4 | 3 | 300 | 1.000 | 0.967 | 9.808 | 13.333 |
| subst_4 | 4 | 330 | 1.000 | 0.000 | 10.095 | 13.620 |
| foreign_3 | 0 | 473 | 0.977 | 0.651 | 0.000 | 3.320 |
| foreign_3 | 1 | 344 | 1.000 | 0.651 | 0.000 | 3.320 |
| foreign_3 | 2 | 430 | 1.000 | 0.000 | 0.000 | 3.320 |
| foreign_3 | 3 | 430 | 0.965 | 0.651 | 0.000 | 3.320 |
| foreign_3 | 4 | 473 | 0.979 | 0.651 | 0.000 | 3.320 |
| foreign_4 | 0 | 330 | 1.000 | 1.000 | 1.950 | 5.475 |
| foreign_4 | 1 | 240 | 1.000 | 1.000 | 1.950 | 5.475 |
| foreign_4 | 2 | 300 | 1.000 | 0.000 | 1.950 | 5.475 |
| foreign_4 | 3 | 300 | 1.000 | 1.000 | 1.950 | 5.475 |
| foreign_4 | 4 | 330 | 1.000 | 1.000 | 1.950 | 5.475 |

**Kind-specific checks:**

- omission_reading_1-5: 50
- omission_S_E_eq_mu2: 50
- truncation_lacking_box5: 100
- truncation_lacking_box5_S_E_eq_mu5: 100
- truncation_lacking_box5_det_D3p: 1.0
- flicker_reach_S_E_eq_flicker_mass: 50
- flicker_reach_alarm_D3p_by_fold: {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.09090909090909091}
- flicker_reach_alarm_pE05_by_fold: {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0}
- flicker_reach_alarm_pE10_by_fold: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
- flicker_reach_D3p_alarms_via_p2_only: 1
- mid_dwell_n: 46
- mid_dwell_readings: {'1-2-3-2-5': 46}
- mid_dwell_S_E_values: {1.09: 45, 1.7933: 1}
- mid_dwell_S_E_max: 1.79333326
- mid_dwell_S_E_eq_offroute_visit_mass: 46/46
- mid_dwell_offroute_visit_frames: {2: 45, 3: 1}
- mid_dwell_plain_min_med_max: (2.67000002, 3.75499997, 9.6366667)
- mid_dwell_alarm_D3p_by_fold: {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.09090909090909091}
- mid_dwell_above_1.2: [{'parent': 47, 'f': 2, 'b2_block_raw': '223222', 'b2_block_p3': '0.14 0.22 0.70 0.17 0.11 0.18', 'insert_after_block_frame': 2, 'composite_box3_frames': 3, 'S_E': 1.79333326, 'S_plain': 3.11666656, 'explain': 'S_E = 1.79: extra visit Box 3 frames 14-16 (mass 1.79); reading 1-2-3-2-5'}]
- repeat_immediate_transit_lt2: 0
- repeat_immediate_S_E_zero: 50
- split_visit_S_E_zero: 150/150

**Truncation** by kept Box-5 frames and whether the debounced reading lacks Box 5:

| variant | lacks5 | n | S_E_min | S_E_med | S_E_max | det_D3' | det_pE05 | det_D3(A) | readings |
|---|---|---|---|---|---|---|---|---|---|
| keep0 | True | 50 | 5.485 | 5.571 | 5.740 | 1.000 | 1.000 | 0.800 | 1-2 x50 |
| keep1 | True | 50 | 5.485 | 5.571 | 5.740 | 1.000 | 1.000 | 0.800 | 1-2 x50 |
| keep2 | False | 50 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1-2-5 x50 |
| keep3 | False | 50 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1-2-5 x50 |

**Tempo scaling** (0.1 s grid; D1 and D2 recomputed; Phase A D3 and D3' on the same scaled cycles):

| variant | n | S_E_med | S_E_max | seq channel p_E<=0.05 | det_D3' | det_D3(A) | det_D2_p2_05 | det_D1_05 | readings |
|---|---|---|---|---|---|---|---|---|---|
| x0.60 | 50 | 0.000 | 0.000 | 0.000 | 0.920 | 0.920 | 0.920 | 0.000 | 1-2-5 x50 |
| x0.70 | 50 | 0.000 | 0.000 | 0.000 | 0.580 | 0.580 | 0.580 | 0.000 | 1-2-5 x50 |
| x0.80 | 50 | 0.000 | 0.000 | 0.000 | 0.120 | 0.120 | 0.120 | 0.000 | 1-2-5 x50 |
| x1.25 | 50 | 0.000 | 1.407 | 0.020 | 0.240 | 0.240 | 0.220 | 0.020 | 1-2-5 x48, 1-2-4-5 x1, 1-2-3-2-5 x1 |
| x1.40 | 50 | 0.000 | 1.407 | 0.020 | 0.520 | 0.500 | 0.500 | 0.000 | 1-2-5 x49, 1-2-3-2-5 x1 |
| x1.70 | 50 | 0.000 | 1.407 | 0.040 | 0.960 | 0.960 | 0.960 | 0.020 | 1-2-5 x47, 1-2-4-5 x1, 1-3-2-5 x1, 1-2-3-2-5 x1 |

Frame intervals inside the parents: 103 intervals > 0.15 s in 50 of 50 parents (min/median/max (0.197, 0.3, 1.199) s; location {'transit': 102, 'Box-1 block': 1}); frames without radar points are absent from the frame table, so the real time axis has holes (locations as listed). Literal nearest-in-time resampling keeps their real duration (the frame before a hole is repeated on the 0.1 s grid). Tempo rows of the parent with the largest interval (1.20 s, cycle 44):

| parent | variant | n_frames | reading_c | S_E | D2 | p2 | D3 | D3' |
|---|---|---|---|---|---|---|---|---|
| 44 | x0.60 | 29 | 1-2-5 | 0.000 | 7.859 | 0.048 | 0.095 | 0.095 |
| 44 | x0.70 | 34 | 1-2-5 | 0.000 | 6.570 | 0.048 | 0.095 | 0.095 |
| 44 | x0.80 | 39 | 1-2-5 | 0.000 | 5.609 | 0.048 | 0.095 | 0.095 |
| 44 | x1.25 | 61 | 1-2-5 | 0.000 | 2.285 | 0.381 | 0.190 | 0.762 |
| 44 | x1.40 | 69 | 1-2-5 | 0.000 | 3.991 | 0.095 | 0.190 | 0.190 |
| 44 | x1.70 | 83 | 1-2-5 | 0.000 | 4.879 | 0.048 | 0.095 | 0.095 |

**Split visit** (negative control; S_E must be 0, D2 may react):

| variant | n | S_E_max | det_D3' | det_pE10 | det_p2_05 | det_D3(A) | readings |
|---|---|---|---|---|---|---|---|
| gap2 | 50 | 0.000 | 0.060 | 0.000 | 0.060 | 0.060 | 1-2-5 x50 |
| gap3 | 50 | 0.000 | 0.060 | 0.000 | 0.060 | 0.060 | 1-2-5 x50 |
| gap5 | 50 | 0.000 | 0.060 | 0.000 | 0.060 | 0.060 | 1-2-5 x50 |

**D1 exhibit (kind 12).** Phase A D1 at its D3 path (p_1 <= 0.05) against E1 (p_E <= 0.05) on the same synthetic cycles:

| kind | n | E1_pE05 | E1 D3' | D1_p1_05 | Phase A D3 |
|---|---|---|---|---|---|
| omission_2 | 50 | 1.000 | 1.000 | 0.360 | 0.380 |
| swap_1-5-2 | 50 | 1.000 | 1.000 | 0.800 | 0.820 |
| repeat_2_far | 50 | 1.000 | 1.000 | 0.800 | 0.820 |
| subst_3 | 2150 | 0.653 | 0.669 | 0.119 | 0.148 |
| subst_4 | 1500 | 1.000 | 1.000 | 0.353 | 0.373 |
| foreign_3 | 2150 | 0.641 | 0.983 | 0.521 | 0.876 |
| foreign_4 | 1500 | 1.000 | 1.000 | 0.800 | 0.820 |

D1 detection per fold: {'subst_4': {0: 0.0, 1: 1.0, 2: 0.0, 3: 0.967, 4: 0.0}, 'omission_2': {0: 0.0, 1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0}, 'subst_3': {0: 0.023, 1: 0.651, 2: 0.0, 3: 0.023, 4: 0.023}}. Calibration D1 maximum per fold: {0: 3.296, 1: 2.303, 2: 3.258, 3: 2.565, 4: 3.258}; D1 values the synthetic deviations receive per fold (first distinct values): {'subst_4': {0: [3.296], 1: [3.332], 2: [2.565, 3.258], 3: [2.526, 3.219], 4: [3.258]}, 'omission_2': {0: [3.296], 1: [3.332], 2: [3.258], 3: [3.219], 4: [3.258]}}. Every unseen transition gets the same Laplace score, so a single calibration cycle with any unseen transition ties with all deviations of its fold.

## 5. Acceptance (stated in the plan before the battery was run)

| item | value | passed |
|---|---|---|
| A1 detection >= 0.95 at D3' for omission, swap, far repeat, subst_4, foreign_4 | omission_2 1.000; swap_1-5-2 1.000; repeat_2_far 1.000; subst_4 1.000; foreign_4 1.000 | PASS |
| A2 subst_3 readable-conditional detection >= 0.95 (D3') | 1.000 (n = 1403 readable of 2150) | PASS |
| A3 mid-dwell flicker multi-match S_E <= 1.2 in every parent (Box-2 block >= 6 frames) | max 1.7933 over n = 46 | FAIL |
| A4a the 51 real normal S_E are 50 x 0 and 1 x 1.09 (cycle 19) | 50 x 0; positive: cycle 19 = 1.0900 | PASS |
| A4b D3' normal alarm set equals the Phase A D3 set | D3' [0, 4, 19, 48] vs Phase A D3 [0, 4, 19, 48] | PASS |
| A5 unit cases of section 2.4 pass | 16 asserted DP cases (+ collapse/traceback/plain asserts) at import | PASS |
| A6 D1 exhibit reproduces the tie mechanism (D1 detection < 0.5 on subst_4 or omission) | subst_4 0.353; omission 0.360 | PASS |

**All items pass: False.**

**Failed item A3, as observed (normal data only; the method was not changed).** 1 of 46 mid-dwell flicker composites score above 1.2:

| parent | f | b2_block_raw | b2_block_p3 | insert_after_block_frame | composite_box3_frames | S_E | S_plain | explain |
|---|---|---|---|---|---|---|---|---|
| 47 | 2 | 223222 | 0.14 0.22 0.70 0.17 0.11 0.18 | 2 | 3 | 1.793 | 3.117 | S_E = 1.79: extra visit Box 3 frames 14-16 (mass 1.79); reading 1-2-3-2-5 |

In this case the parent's own Box-2 dwell already holds a raw Box-3 frame (a 1-frame B2 misreading that the Phase A debouncer had absorbed into the Box-2 visit) immediately before the midpoint where the 2-frame flicker block is inserted. The composite therefore carries a 3-frame Box-3 run, and S_E equals that run's whole posterior mass (S_E = off-route visit mass in 46/46 mid-dwell composites). The multi-match still charges only the foreign visit (the split Box-2 dwell costs nothing; the plain rule charges more), but the 1.2 bound assumed the inserted visit carries only the block's own mass. Plan section 4 says that a failed item leads to an amendment on normal data only, re-dated, before the one-shot; that decision is left to the authors.

**Preliminary values (workflow logs, `faz_e/prelim/`) against this run:**

| quantity | preliminary (workflow logs) | this run |
|---|---|---|
| omission_2 S_E min / D3' det | 7.86 / 1.000 | 7.858 / 1.000 |
| swap_1-5-2 S_E min | 9.30 | 9.301 |
| repeat_2_far S_E min | 3.06 | 3.063 |
| subst_3 p_E<=0.05 overall / readable | 0.656 (500 sampled pairs) / 327 of 327 | 0.653 (all 2150 pairs) / 1403 of 1403 |
| subst_4 p_E<=0.05 | 500/500 | 1500/1500 |
| foreign_4 S_E min | 1.95 | 1.950 |
| reach flicker S_E | 1.09 | 1.0900-1.0900 |
| mid-dwell flicker multi-match / plain | 1.09 every parent / 2.29-8.70 (n = 46) | 1.09-1.79 / 2.67-9.64 (n = 46) |
| D1 p_1<=0.05: subst_4 / subst_3 / omission / swap / repeat / foreign_4 | 0.354 / 0.126 / 0.360 / 0.800 / 0.800 / 0.800 | 0.353 / 0.119 / 0.360 / 0.800 / 0.800 / 0.800 |

## 6. Method notes: details the plan left open (fixed in code before scoring)
- Blocks = frames [first, last+1) of the parent's collapsed visit (session-wide debounced labels; == isolated re-debounce for all 51 cycles); a composite carries the parent's RAW B2 labels and posteriors and is debounced again as a whole.
- Time axis of every non-tempo composite = spliced real frame intervals (each frame keeps t[k+1]-t[k] of its source: s02 parent, s01 run or the s02 flicker frames; inserted none frames 0.1 s). The unmodified parent therefore reproduces Phase A D2 exactly; dropped-frame holes travel with their block.
- Insertion points = floor midpoints: foreign at e2 + (s5-e2)//2 (2->5 transit); reach flicker at e1 + (s2-e1)//2 (1->2 transit); mid-dwell flicker and split gaps at s2 + (e2-s2)//2.
- swap_1-5-2 = pre + B5 + (2->5 transit) + B2 + post. repeat_2_far = B2 copy inserted right after the last frame of the Box-5 block. repeat_2_immediate = B2 + (copy of the parent's own 2->5 transit) + B2 copy + (2->5 transit) + B5.
- subst_b = s01 run replaces B2; foreign_b = s01 run inserted in the 2->5 transit; full cross products (50 x 43 Box-3 runs, 50 x 30 Box-4 runs), no RNG. 'Readable' = the debounced composite has a visit of the pasted box.
- truncation keepK = the first K frames of the Box-5 block are kept, the rest of the block is deleted (K = 0, 1, 2, 3).
- tempo xs: n' = round(T*s/0.1) frames on a 0.1 s grid (T = parent duration); frame k takes the parent frame whose real start time is nearest to 0.1*k/s (ties -> earlier frame); labels and posteriors copied, then debounced.
- mid-dwell flicker parents: Box-2 block >= 6 frames (each half keeps >= 3 frames = MIN_BOX + 1); shorter parents are run as a supplementary row outside the acceptance.
- split_visit gapG: G none frames (posterior one-hot none, 0.1 s each) inserted at the middle of the Box-2 block.
- Every synthetic cycle: parent fold f, route/mu and Phase A parameters fitted on the real training normals (f+3, f+4), calibrated on the real calibration normals (f+1, f+2), conformal ties conservative (a2_cycles.conf_p).
- p-values, D1, D2 and debouncing are the Phase A functions of `faz_a2/a2_cycles.py` (`conf_p`, `fit_params`, `score_cycle`, `debounce_labels`, `visits_of`, `cycle_from`).
- Explanations: every synthetic and normal row carries the DP traceback text (`explain` column of the CSVs).
- Code shared with the one-shot: `e1_core.e1_inputs`, `score_roles` and `compare_to_reference` produced every normal-side number above; `e1_core.oneshot_tables` was smoke-tested here on the 51 normal test rows plus synthetic rows relabelled as stand-in anomaly groups (omission_2 -> seq135, subst_4 -> seq145, tempo x0.60 -> fast, tempo x1.70 -> slow); its rates matched the direct computation. `faz_e/02_e1_oneshot.py` itself was only parsed (ast), never executed.

## 7. Files
- `reports/faz_e/01_normal_rows.csv`
- `reports/faz_e/01_folds.csv`
- `reports/faz_e/01_synthetic_cycles.csv.gz`
- `reports/faz_e/01_synthetic_summary.csv`
- `reports/faz_e/01_acceptance.csv`
- `reports/faz_e/01_s01_runs.csv`
- `reports/faz_e/01_identity_check.csv`
- `reports/faz_e/01_synthetic_perfold.csv`
- `reports/faz_e/01_readable_split.csv`
- `reports/faz_e/01_tempo.csv`
- `reports/faz_e/01_synthetic_per_run.csv`
- `reports/faz_e/01_frame_intervals.csv`
- `reports/faz_e/01_synthetic.md`
- `reports/faz_e/01_synthetic_stdout.log` (stdout of this run, kept by the caller: `bash tools/local_run.sh faz_e/01_e1_synthetic.py 2>&1 | tee reports/faz_e/01_synthetic_stdout.log`)

## 8. Provenance
- Python 3.12.13, numpy 2.5.4, pandas 3.0.6; host <LOCAL_HOST>; CPU only.
- DESIGN_E.md frozen part sha256 `d8c2cafd4a788f0b87e8e222e2ec8ae4e6e8dc74c81b687a885b7207df168d76`.
- code `e1_core.py` sha256 `ea5b927640107b3ca7a9d2d32f9adfb38aa0a764a6fee9d83757185a1d87f2c6`
- code `01_e1_synthetic.py` sha256 `50ff64e5c73a77a54d2af92fcdab0790c2863b82629c73384fd858358b4555af`
- code `a2_cycles.py` sha256 `775b8c7e8ee8daf2bcd53caa55394204923d3bb6f61ecbc19ec5e4536a9bc70d`
- code `fa_common.py` sha256 `abc322022283df657b534fd1da6ae3a04b11d5707d49ad2a022b336a15ca6dce`
- input `reports/faz_a/oof/b2__S1__f0.csv.gz` sha256 `7096b1970c872ba602a2cfd8f528c5e54618ac768491b56e8995b7ebfce65603`
- input `reports/faz_a/oof/b2__S1__f1.csv.gz` sha256 `2f587a2fae582e58a5486e6dff507e5c1341c2cfb2ca8df55f9be42f34011214`
- input `reports/faz_a/oof/b2__S1__f2.csv.gz` sha256 `287893bd8407148f34a928c67b2b181ff47072589057f25c9e93881edcb9c008`
- input `reports/faz_a/oof/b2__S1__f3.csv.gz` sha256 `73b1cd285416c2efb7d556f58d391bb3f83a57dd563a662c492ce48b3929dffd`
- input `reports/faz_a/oof/b2__S1__f4.csv.gz` sha256 `687a3a257fa7c509c7b3563d113649dbd77dfd02df88bfa930bd4974753c67fd`
- input `reports/faz_a/folds_anom.csv` sha256 `3a36560b4e0e7c02afb868eb60fd72b0404ac11c0382ba69ce302f39cf35da3e`
- input `reports/faz_a/04_cycle_scores.csv.gz` sha256 `fc74eb7de34752e7cb56693fef5e46955e88be1f6761d4fd9b218ceb39609a77`
- Sessions in memory: ['session_01_training', 'session_02_normal_seq125']; tripwires fired: none; `fa_common` all-session frame table built: no.

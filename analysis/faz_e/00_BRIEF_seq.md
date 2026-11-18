# Phase E brief: why sequence deviations are missed, and what a better sequence channel must satisfy (2026-11-05)

Written for the method-search workflow (proposers on Fable). Facts only; every number is from the files named.
ROOT = `<ROOT>`.

## 1. Task and data (settled; do not re-derive)
- One operator, one bench. A TI IWR6843ISK radar streams its on-chip 2-D point cloud (x azimuth, y range; z = 0) at **10 Hz**. Five parts boxes stand in a row along x: Box1 +0.21 m, Box2 +0.13 (s01) / +0.07 (s02, s05), Box3 0.00, Box4 -0.09, Box5 -0.18; y about 0.43-0.47 m. Nominal azimuth resolution at 0.45 m is about 11.8 cm, so neighbouring boxes are about one resolution cell apart.
- ToF sensors over the boxes give frame labels (0 = none, 1-5 = box) **for training only**.
- Sessions: s01 training (random picks over all five boxes), s02 normal (route 1-2-5, 51 usable cycles), s03 sequence deviation (1-3-5, 57 cycles), s04 sequence deviation (1-4-5, 49), s05 fast and s06 slow (route 1-2-5, tempo deviations). A cycle = one full assembly (about 4-6 s).
- State estimator used by the deployable detector: **B2** = random forest (300 trees) on 21 statistics of a five-frame window; cycle-grouped 5-fold cross-fitting (S1). Out-of-fold per-frame outputs with class probabilities: `ROOT/reports/faz_a/oof/b2__S1__f{0..4}.csv.gz` (columns session, frame_id, t, loop_id, fold, y_true, y_raw, y_pred, p0..p5).
- Cycle table of the primary analysis: `ROOT/reports/faz_a/04_cycle_scores.csv.gz` (filter state_source == 'b2', segmentation == 'pred'; columns include f, role in {train, cal, test}, atype, natural_dev, start_frame_id, end_frame_id, visit_seq, D0, D1, D2, p1, p2, p3). Normal folds: `ROOT/reports/faz_a/folds_anom.csv`.
- Code of the primary detector: `ROOT/faz_a/04_cycle_anomaly.py` (+ `faz_a/fa_common.py`); conformal variants: `ROOT/faz_a2/01_conformal.py`.

## 2. The primary detector (planned in writing before it was run; stays reported as is)
- Debounce of the B2 frame labels (merge a <= 1-frame none gap inside the same box, then turn box runs < 2 frames into none; repeat to a fixed point). Visit = box run. Cycle = predicted Box1 entry to the next one.
- **D1 sequence channel:** first-order Markov over {START, 1..5, END}, Laplace alpha = 1, fitted on the ~20 training normal cycles of the fold; cycle score = max over transitions of -log p.
- **D2 duration channel:** per-step log-duration z-scores (dwells and gaps), reference from training normal cycles, sigma floor 0.05.
- Conformal p = (1 + #{cal >= s}) / (n_cal + 1), n_cal = 19-22 calibration normal cycles per fold. Fold roles: test f, calibration f+1, f+2, training f+3, f+4 (mod 5). Each normal cycle is tested once; anomalous cycles are tested in every fold.
- **D3 = min{1, 2 min(p1, p2)}** (Bonferroni); alarm if p3 <= alpha = 0.10.

## 3. Results (B2 x pred, alpha = 0.10) - `ROOT/reports/faz_a/04_anomaly.md`, `ROOT/reports/faz_a2/01_conformal.md`
| rule | FAR (51 normal) | seq135 | seq145 | fast | slow |
|---|---|---|---|---|---|
| D3 primary | 0.078 | 0.254 | 0.475 | 0.921 | 0.929 |
| D1 alone | 0.059 | 0.604 | 0.800 | 0.217 | 0.008 |
| D2 alone | 0.078 | 0.068 | 0.125 | 0.800 | 0.992 |
| D3, smoothed p (ties broken at random) | 0.087 | 0.445 | 0.719 | 0.936 | 0.932 |
| D3, cross-conformal (n_cal about 40) | 0.059 | 0.257 | 0.650 | 0.925 | 0.937 |
| D3, cross-conformal + smoothed | 0.091 | 0.377 | 0.862 | 0.951 | 0.942 |
| D1, ToF oracle labels | 0.000 | 1.000 | 1.000 | - | - |
- D1 AUROC for sequence deviations = 0.974 (B2 x pred). B2 recovers the expected visit order in 54/56 s03 and 47/48 s04 predicted cycles, and reads 1-2-5 in 47 of 51 normal cycles (others: 3 x `1-2-2-5`, 1 x `1-3-2-5`).
- **The signal is there; the decision layer loses it.**

## 4. Diagnosis (computed 2026-11-05 from the files above; script kept in the session scratchpad)
Per fold, max calibration D1 and the D1 value every sequence anomaly gets:
| fold | n_cal | calibration cycles that are not 1-2-5 | top calibration D1 | seq anomaly D1 | #cal >= anomaly | D3 det seq135 / seq145 |
|---|---|---|---|---|---|---|
| 0 | 19 | 1-3-2-5, 1-2-2-5 | 3.296 | 3.296 | 1 | 0.07 / 0.13 |
| 1 | 20 | 1-2-2-5 | 2.303 | 3.332 | 0 | 1.00 / 1.00 |
| 2 | 21 | 1-2-2-5 x 2 | 3.258 | 2.565 / 3.258 | 2 | 0.05 / 0.13 |
| 3 | 22 | 1-2-2-5 x 2 | 2.565 | 2.526 (135) / 3.219 (145) | 2 / 0 | 0.07 / 1.00 |
| 4 | 20 | 1-3-2-5 | 3.258 | 3.258 | 1 | 0.07 / 0.13 |
- Every unseen transition gets the same Laplace score (about -log 1/(row total + 6)), so all sequence anomalies in a fold share **one** D1 value. A single calibration cycle with any unseen transition ties with all of them and kills the D3 alarm (D3 needs p1 <= 0.05, i.e. #cal >= s = 0).
- The tying normal cycles are **artefacts**: `1-2-2-5` is one Box-2 dwell split in two by the debouncer (the 2 -> 2 transition is "unseen"), and the one `1-3-2-5` is a 2-frame (0.2 s) Box-3 flicker (max p3 = 0.70) during the 1 -> 2 reach.
- D1 measures how rare a transition was in about 20 training cycles, not how much evidence there is for a wrong box.

**Disclosure (must be carried into any plan):** during this diagnosis, before any Phase E plan existed, two candidate quantities were also computed on the sequence-deviation sessions: longest raw B2 run at the substituted box per cycle (median 7 frames, minimum 4, 10th percentile 5) and summed posterior mass p3 + p4 over the cycle (seq135 median 6.83, minimum 4.05; seq145 median 6.68, minimum 4.55), against normal cycles (p3 + p4 mass: median 0.91, 90th percentile 1.58, maximum 2.30; the 1-3-2-5 flicker has a 2-frame Box-3 run). Any method built on these quantities is therefore not blind and must say so; no parameter may be set from the anomaly numbers.

## 5. Requirements for a new sequence channel (Phase E, secondary analysis)
1. **Fit and calibrate on normal cycles only** (s02 training/calibration folds; s01 may be used because it is the commissioning/training session). Nothing learned from s03-s06. Same fold roles as Phase A, or a declared alternative (e.g. cross-conformal) whose validity argument is stated.
2. **FAR control at alpha = 0.10** with an explicit argument (exchangeability of normal cycles; what is approximate).
3. **General, not "Box 3 or 4 present":** it must also respond to omissions (skip Box 2), order swaps (1-5-2), extra/foreign stations and repeats, with the route learned from normal cycles. It must not be triggered by split visits or sub-0.3 s flickers *by construction* (e.g. continuous evidence), not by a threshold tuned to these data.
4. **Computable from the existing OOF outputs** (no retraining; CPU only; the RTX 3090 is not available).
5. **Must not break tempo detection** (93%) or raise FAR above the alpha band; combination with D2 must keep a stated FAR bound.
6. **Explainable in one paragraph** of a measurement journal (the target journal).
7. **Validation without the anomaly sessions:** synthetic sequence deviations can be built from normal cycles (e.g. delete the Box-2 dwell frames, swap dwell blocks, insert frames from another box taken from s01) to estimate detection before the one-shot test on s03/s04.
8. The Phase A rule (36%) stays reported as the planned primary result; Phase E is labelled secondary, planned after the Phase A results were known. Never call anything "pre-registered".
9. s02 has natural-deviation cycles (flag `natural_dev`); keep them in the pool but **never look up or report their alarm status**.
